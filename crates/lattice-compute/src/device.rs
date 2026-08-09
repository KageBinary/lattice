//! The backend boundary: who owns memory, and who is allowed to say no.
//!
//! §15.4 makes the portable GPU backend the product baseline and requires that *"the
//! architecture should permit later native backends such as CUDA without forcing the
//! simulation IR to change."* That sentence is the whole design brief for this module, and
//! it has a consequence that is easy to miss: the trait has to be shaped by what a CUDA
//! backend and a WebGPU backend *have in common*, which is much less than what either has
//! in common with the CPU.
//!
//! # Why this is not `Executor` with more implementations
//!
//! `lattice_cpu::Executor` answers "how do I split this loop across threads that share
//! memory". Every part of that question is wrong for a GPU. There is no `&mut [f64]` to
//! hand out, because the data is not in host memory; there is no loop, because the kernel
//! *is* the loop body; and the split is not the caller's to choose. The two abstractions
//! coexist rather than nest — `CpuDevice` uses an `Executor` internally, the way a wgpu
//! device uses a command encoder.
//!
//! # Why an associated buffer type rather than `Box<dyn Buffer>`
//!
//! Runtime backend selection happens *once per run* — the CLI matches on `--backend` and
//! calls a generic function — not once per dispatch. Paying for a vtable at the boundary
//! would buy nothing and would cost the thing §15.1 asks for, so [`Device`] is generic and
//! the polymorphism lives at the top where the choice is actually made.
//!
//! # The host speaks `f64`, always
//!
//! [`Device::write`] takes `&[f64]` and [`Device::read`] fills `&mut [f64]`, whatever the
//! device stores. §24.1 calls the scalar CPU path *"the executable specification for
//! accelerated kernels"*, so the reference representation is the one the specification is
//! written in, and a backend converts on the way in and on the way out.
//!
//! That puts every narrowing at exactly one place. It costs a conversion pass a
//! device-native API would avoid, and it buys the property the milestone actually needs:
//! when a GPU result differs from the CPU's, the difference has one entrance and can be
//! attributed to it. See [`Tolerance`](crate::Tolerance).

use core::fmt;

use crate::Precision;

/// Which implementation is running the kernels.
///
/// §19.3 requires the backend to be published with any performance number, and FR-011
/// requires a run to be identifiable. This is the value both of those print.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Backend {
    /// The scalar CPU path: §24.1's executable specification.
    CpuScalar,
    /// The CPU path with loops split across a thread pool. Bit-identical to
    /// [`Backend::CpuScalar`] by construction — see `lattice_cpu`.
    CpuParallel,
    /// The portable WebGPU/wgpu backend. §15.4's product baseline.
    Wgpu,
}

impl Backend {
    /// The short name used in run artifacts and validation reports.
    pub const fn name(self) -> &'static str {
        match self {
            Backend::CpuScalar => "cpu-scalar",
            Backend::CpuParallel => "cpu-parallel",
            Backend::Wgpu => "wgpu",
        }
    }

    /// True if this backend is the reference the others are compared against.
    ///
    /// Exactly one backend may claim this, and it is the scalar CPU path. `CpuParallel`
    /// deliberately does *not*: it agrees with the reference bit-for-bit, which is a
    /// stronger statement than being one.
    pub const fn is_reference(self) -> bool {
        matches!(self, Backend::CpuScalar)
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What a device can actually do.
///
/// §15.5 makes this part of the kernel cache key — a kernel specialized for one device's
/// limits must not be reused on a device with different ones — so it is a value that
/// hashes, not a bag of methods.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Capabilities {
    /// Human-readable adapter name, for §19.3's "publish exact hardware".
    pub adapter: String,
    /// Precisions this device will run. See [`Capabilities::supports`].
    pub precisions: Vec<Precision>,
    /// Largest number of invocations in one workgroup, or `None` for a CPU device where
    /// the concept does not apply.
    pub max_workgroup: Option<u32>,
    /// Largest single allocation, in bytes.
    pub max_buffer_bytes: u64,
}

impl Capabilities {
    /// True if `precision` can be run here.
    ///
    /// A device that cannot run a mode must say so rather than substituting one it likes.
    /// The portable GPU backend returns `false` for [`Precision::Accurate64`], and that
    /// single fact is the largest constraint in M4 — WGSL has no 64-bit float type at all,
    /// so the baseline backend cannot execute the reference precision even in principle.
    pub fn supports(&self, precision: Precision) -> bool {
        self.precisions.contains(&precision)
    }

    /// The best precision this device offers, preferring the reference.
    ///
    /// Returns `None` for a device that supports nothing, which is not a state any
    /// constructor here can produce but is not worth an `unwrap` in a caller.
    pub fn best_precision(&self) -> Option<Precision> {
        [Precision::Accurate64, Precision::Mixed, Precision::Fast32]
            .into_iter()
            .find(|&candidate| self.supports(candidate))
    }
}

/// Why a device operation could not be performed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeviceError {
    /// No device of the requested kind could be opened.
    Unavailable { backend: Backend, detail: String },
    /// The device does not implement the requested precision.
    ///
    /// A distinct variant rather than a string, because a caller may legitimately want to
    /// fall back — but only *deliberately*, having seen which mode it is giving up.
    UnsupportedPrecision { backend: Backend, requested: Precision, available: Vec<Precision> },
    /// An allocation exceeded [`Capabilities::max_buffer_bytes`].
    AllocationTooLarge { requested: u64, limit: u64 },
    /// A transfer's host slice did not match the buffer's element count.
    LengthMismatch { buffer: usize, host: usize },
    /// The backend failed for a reason of its own.
    Backend(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::Unavailable { backend, detail } => {
                write!(f, "no {backend} device available: {detail}")
            }
            DeviceError::UnsupportedPrecision { backend, requested, available } => {
                let names: Vec<_> = available.iter().map(|p| p.name()).collect();
                write!(
                    f,
                    "the {backend} backend cannot run {requested}; it offers {}",
                    names.join(", ")
                )
            }
            DeviceError::AllocationTooLarge { requested, limit } => {
                write!(f, "allocation of {requested} bytes exceeds the device limit of {limit}")
            }
            DeviceError::LengthMismatch { buffer, host } => {
                write!(f, "buffer holds {buffer} elements but the host slice has {host}")
            }
            DeviceError::Backend(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for DeviceError {}

/// How a buffer will be used.
///
/// §15.2 asks for *"double/triple buffering for asynchronous compute, rendering, and
/// observation readback where useful"*, and a backend cannot arrange any of that without
/// being told which buffers are read back and which never leave the device. Declaring it
/// at allocation is the only point where the answer is known.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Usage {
    /// Written once from the host, then read by kernels. Face coefficients, for instance.
    Upload,
    /// Lives on the device across steps and is never transferred. Scratch and work fields.
    Resident,
    /// Read back to the host on an observation cadence.
    Readback,
}

/// A region of device memory holding `len` elements of one precision.
pub trait Buffer: fmt::Debug + Send + Sync {
    /// Elements, not bytes.
    fn len(&self) -> usize;

    /// True if the buffer holds nothing.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How each element is stored.
    fn precision(&self) -> Precision;

    /// What the buffer was allocated for.
    fn usage(&self) -> Usage;

    /// Bytes occupied on the device, ignoring any padding the backend adds.
    fn bytes(&self) -> u64 {
        self.len() as u64 * self.precision().state_bytes() as u64
    }
}

/// A backend that owns memory and runs kernels over it.
///
/// Implementations: `lattice_cpu::CpuDevice` and `lattice_wgpu::GpuDevice`. The trait is
/// deliberately small — allocation, transfer, and self-description — because everything
/// else differs enough between the two that a shared signature would be a lie. Kernels are
/// declared by the crates that own the physics, against the buffers this hands out.
pub trait Device: fmt::Debug + Send + Sync {
    /// This device's buffer type.
    type Buffer: Buffer;

    /// Which backend this is.
    fn backend(&self) -> Backend;

    /// What it can do.
    fn capabilities(&self) -> &Capabilities;

    /// The precision it was opened with.
    ///
    /// Fixed at construction rather than per-kernel: a run that computed half its steps in
    /// `fast32` and half in `accurate64` would have an error budget nobody could write
    /// down, and §10.4 requires that budget to exist.
    fn precision(&self) -> Precision;

    /// A one-line description for the run artifact, per §19.3.
    fn label(&self) -> String {
        format!(
            "{} ({}, {})",
            self.backend(),
            self.capabilities().adapter,
            self.precision()
        )
    }

    /// Reserve `elements` elements of this device's precision.
    fn alloc(&self, elements: usize, usage: Usage) -> Result<Self::Buffer, DeviceError>;

    /// Copy `data` into `buffer`, narrowing to the device precision if it is narrower.
    ///
    /// This is the *only* narrowing point in the system, which is what makes
    /// [`Tolerance`](crate::Tolerance) able to name a mechanism.
    fn write(&self, buffer: &mut Self::Buffer, data: &[f64]) -> Result<(), DeviceError>;

    /// Copy `buffer` back into `out`, widening to `f64`.
    fn read(&self, buffer: &Self::Buffer, out: &mut [f64]) -> Result<(), DeviceError>;

    /// Block until all work submitted to this device has completed.
    ///
    /// §15.1 asks for end-to-end measurement, which is impossible against a backend that
    /// returns before it has finished. A CPU device implements this as a no-op and that is
    /// honest, because its dispatches have already completed.
    fn finish(&self) -> Result<(), DeviceError>;

    /// Convenience: allocate and fill in one step.
    fn upload(&self, data: &[f64], usage: Usage) -> Result<Self::Buffer, DeviceError> {
        let mut buffer = self.alloc(data.len(), usage)?;
        self.write(&mut buffer, data)?;
        Ok(buffer)
    }

    /// Convenience: read a whole buffer into a fresh vector.
    fn download(&self, buffer: &Self::Buffer) -> Result<Vec<f64>, DeviceError> {
        let mut out = vec![0.0; buffer.len()];
        self.read(buffer, &mut out)?;
        Ok(out)
    }
}

/// Check a requested precision against a device's capabilities.
///
/// Shared so that every backend refuses in the same words, and so that no backend can
/// quietly implement "refuse" as "substitute".
pub fn require_precision(
    backend: Backend,
    capabilities: &Capabilities,
    requested: Precision,
) -> Result<(), DeviceError> {
    if capabilities.supports(requested) {
        return Ok(());
    }
    Err(DeviceError::UnsupportedPrecision {
        backend,
        requested,
        available: capabilities.precisions.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(precisions: &[Precision]) -> Capabilities {
        Capabilities {
            adapter: "test".to_string(),
            precisions: precisions.to_vec(),
            max_workgroup: Some(256),
            max_buffer_bytes: 1 << 30,
        }
    }

    #[test]
    fn exactly_one_backend_is_the_reference() {
        let all = [Backend::CpuScalar, Backend::CpuParallel, Backend::Wgpu];
        let references: Vec<_> = all.iter().filter(|b| b.is_reference()).collect();
        assert_eq!(references, [&Backend::CpuScalar]);
    }

    #[test]
    fn a_device_that_cannot_run_a_precision_says_which_it_offers() {
        let capabilities = caps(&[Precision::Fast32]);
        let error =
            require_precision(Backend::Wgpu, &capabilities, Precision::Accurate64).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("accurate64"), "{message}");
        assert!(message.contains("fast32"), "{message}");
    }

    /// The failure this guards against is silent substitution, so the test is that the
    /// call *fails* rather than that it returns something plausible.
    #[test]
    fn an_unsupported_precision_is_refused_rather_than_downgraded() {
        let capabilities = caps(&[Precision::Fast32, Precision::Mixed]);
        assert!(require_precision(Backend::Wgpu, &capabilities, Precision::Accurate64).is_err());
        assert!(require_precision(Backend::Wgpu, &capabilities, Precision::Fast32).is_ok());
    }

    #[test]
    fn best_precision_prefers_the_reference() {
        assert_eq!(
            caps(&[Precision::Fast32, Precision::Accurate64]).best_precision(),
            Some(Precision::Accurate64)
        );
        assert_eq!(
            caps(&[Precision::Fast32, Precision::Mixed]).best_precision(),
            Some(Precision::Mixed)
        );
        assert_eq!(caps(&[]).best_precision(), None);
    }

    #[test]
    fn capabilities_hash_so_they_can_key_a_kernel_cache() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        assert!(seen.insert(caps(&[Precision::Fast32])));
        assert!(!seen.insert(caps(&[Precision::Fast32])), "equal capabilities collide");
        assert!(seen.insert(caps(&[Precision::Accurate64])));
    }
}
