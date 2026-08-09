//! The CPU as a [`Device`], so that §24.1's executable specification is reachable through
//! the same boundary the accelerators are.
//!
//! This is not how a CPU solver *should* be driven. A solver running on the host already
//! has its data in host memory, and going through [`Device::write`] to hand it to itself
//! copies for nothing. `HeatDomain` and its neighbours are right to keep using
//! [`Executor`](crate::Executor) directly, and they do.
//!
//! It exists for two things the direct path cannot do.
//!
//! **It is the control in the attribution experiment.** A GPU result differs from the
//! reference for several reasons at once, and the loudest of them is that WGSL has no
//! `f64`. A `CpuDevice` opened at [`Precision::Fast32`] runs the *same* host code over
//! `f32` storage, so the difference between it and the `f64` reference is `f32` storage and
//! nothing else. Subtract it from the GPU's disagreement and what remains is what the GPU
//! itself contributed. Without a control, the tolerance decomposition in
//! [`lattice_compute::Tolerance`] would be an argument rather than a measurement.
//!
//! **It keeps the trait honest.** A boundary with one implementation is shaped like that
//! implementation. Two, one of which stores its state in a different width than the host
//! speaks, is enough to have found the places where the boundary was assuming things.

use lattice_compute::{
    Backend, Buffer, Capabilities, Device, DeviceError, Precision, Usage, require_precision,
};

use crate::Executor;

/// Host memory, stored at the device's precision.
///
/// Two variants rather than a byte vector with a tag: the whole point of this device is to
/// store `f32` where the reference stores `f64`, and a representation that made that
/// invisible would defeat it.
#[derive(Debug)]
enum Storage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A buffer in host memory.
#[derive(Debug)]
pub struct CpuBuffer {
    storage: Storage,
    usage: Usage,
}

impl CpuBuffer {
    /// The contents as `f64`, if that is how they are stored.
    ///
    /// `None` under [`Precision::Fast32`] — deliberately, rather than converting. A kernel
    /// that wanted `f64` and silently got a widened `f32` would be computing at a
    /// precision nobody asked for.
    pub fn as_f64(&self) -> Option<&[f64]> {
        match &self.storage {
            Storage::F64(values) => Some(values),
            Storage::F32(_) => None,
        }
    }

    /// The contents as `f32`, if that is how they are stored.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::F32(values) => Some(values),
            Storage::F64(_) => None,
        }
    }

    /// Mutable `f64` contents, if that is how they are stored.
    pub fn as_f64_mut(&mut self) -> Option<&mut [f64]> {
        match &mut self.storage {
            Storage::F64(values) => Some(values),
            Storage::F32(_) => None,
        }
    }

    /// Mutable `f32` contents, if that is how they are stored.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.storage {
            Storage::F32(values) => Some(values),
            Storage::F64(_) => None,
        }
    }
}

impl Buffer for CpuBuffer {
    fn len(&self) -> usize {
        match &self.storage {
            Storage::F32(values) => values.len(),
            Storage::F64(values) => values.len(),
        }
    }

    fn precision(&self) -> Precision {
        match &self.storage {
            Storage::F32(_) => Precision::Fast32,
            Storage::F64(_) => Precision::Accurate64,
        }
    }

    fn usage(&self) -> Usage {
        self.usage
    }
}

/// The CPU behind [`lattice_compute::Device`].
#[derive(Debug)]
pub struct CpuDevice {
    executor: Executor,
    precision: Precision,
    capabilities: Capabilities,
}

impl CpuDevice {
    /// A device at `precision`, running kernels on `executor`.
    ///
    /// # Errors
    ///
    /// [`DeviceError::UnsupportedPrecision`] for [`Precision::Mixed`] — see
    /// [`CpuDevice::capabilities`] for why it is not claimed.
    pub fn new(executor: Executor, precision: Precision) -> Result<CpuDevice, DeviceError> {
        let capabilities = Capabilities {
            adapter: host_description(&executor),
            // `Mixed` is absent on purpose. It differs from `Fast32` only in the width
            // accumulations are carried at, no reduction runs on a device yet, and §11's
            // lesson about contracts that overclaim applies to capabilities too: a mode
            // whose distinguishing behaviour is untested must not be advertised.
            precisions: vec![Precision::Fast32, Precision::Accurate64],
            max_workgroup: None,
            max_buffer_bytes: u64::MAX,
        };
        let backend = backend_of(&executor);
        require_precision(backend, &capabilities, precision)?;
        Ok(CpuDevice { executor, precision, capabilities })
    }

    /// A device at the reference precision, running on one thread.
    ///
    /// §24.1's executable specification, reachable through the backend boundary.
    pub fn reference() -> CpuDevice {
        CpuDevice::new(Executor::sequential(), Precision::Accurate64)
            .expect("the reference precision is always supported on the host")
    }

    /// The executor this device splits its loops across.
    pub fn executor(&self) -> &Executor {
        &self.executor
    }
}

fn backend_of(executor: &Executor) -> Backend {
    if executor.is_sequential() { Backend::CpuScalar } else { Backend::CpuParallel }
}

fn host_description(executor: &Executor) -> String {
    executor.label()
}

impl Device for CpuDevice {
    type Buffer = CpuBuffer;

    fn backend(&self) -> Backend {
        backend_of(&self.executor)
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn alloc(&self, elements: usize, usage: Usage) -> Result<CpuBuffer, DeviceError> {
        let storage = match self.precision {
            Precision::Accurate64 => Storage::F64(vec![0.0; elements]),
            Precision::Fast32 | Precision::Mixed => Storage::F32(vec![0.0; elements]),
        };
        Ok(CpuBuffer { storage, usage })
    }

    fn write(&self, buffer: &mut CpuBuffer, data: &[f64]) -> Result<(), DeviceError> {
        if buffer.len() != data.len() {
            return Err(DeviceError::LengthMismatch { buffer: buffer.len(), host: data.len() });
        }
        match &mut buffer.storage {
            Storage::F64(values) => values.copy_from_slice(data),
            // The narrowing. One line, one place, and the only reason a `Fast32` run
            // disagrees with the reference before any arithmetic has happened.
            Storage::F32(values) => {
                for (slot, &value) in values.iter_mut().zip(data) {
                    *slot = value as f32;
                }
            }
        }
        Ok(())
    }

    fn read(&self, buffer: &CpuBuffer, out: &mut [f64]) -> Result<(), DeviceError> {
        if buffer.len() != out.len() {
            return Err(DeviceError::LengthMismatch { buffer: buffer.len(), host: out.len() });
        }
        match &buffer.storage {
            Storage::F64(values) => out.copy_from_slice(values),
            Storage::F32(values) => {
                for (slot, &value) in out.iter_mut().zip(values) {
                    *slot = f64::from(value);
                }
            }
        }
        Ok(())
    }

    fn finish(&self) -> Result<(), DeviceError> {
        // Every dispatch this device makes has already completed by the time it returns —
        // `Executor` is synchronous. Saying so is more useful than leaving the default.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_device_is_the_scalar_path_at_full_width() {
        let device = CpuDevice::reference();
        assert_eq!(device.backend(), Backend::CpuScalar);
        assert!(device.backend().is_reference());
        assert_eq!(device.precision(), Precision::Accurate64);
        assert!(device.precision().is_reference());
    }

    #[test]
    fn a_parallel_executor_reports_the_parallel_backend() {
        let device = CpuDevice::new(Executor::with_threads(4), Precision::Accurate64).unwrap();
        assert_eq!(device.backend(), Backend::CpuParallel);
        assert!(device.label().contains("accurate64"), "{}", device.label());
    }

    #[test]
    fn mixed_is_refused_rather_than_treated_as_fast32() {
        let error = CpuDevice::new(Executor::sequential(), Precision::Mixed).unwrap_err();
        assert!(matches!(error, DeviceError::UnsupportedPrecision { .. }));
    }

    #[test]
    fn a_round_trip_at_the_reference_precision_is_exact() {
        let device = CpuDevice::reference();
        let data = [300.0, 1.0 / 3.0, -std::f64::consts::E, f64::MIN_POSITIVE];
        let buffer = device.upload(&data, Usage::Resident).unwrap();
        assert_eq!(device.download(&buffer).unwrap(), data);
    }

    /// The control experiment, in one test: a `Fast32` round trip loses exactly what `f32`
    /// storage loses and nothing else.
    #[test]
    fn a_round_trip_at_fast32_loses_exactly_the_f32_narrowing() {
        let device = CpuDevice::new(Executor::sequential(), Precision::Fast32).unwrap();
        let data = [300.0, 1.0 / 3.0, -std::f64::consts::E];
        let buffer = device.upload(&data, Usage::Resident).unwrap();
        let out = device.download(&buffer).unwrap();

        for (&want, &got) in data.iter().zip(&out) {
            assert_eq!(got, f64::from(want as f32), "not a plain f32 narrowing");
        }
        assert_ne!(out[1], data[1], "1/3 is not representable in f32");
    }

    #[test]
    fn storage_width_follows_the_device_precision() {
        let reference = CpuDevice::reference();
        let buffer = reference.alloc(8, Usage::Resident).unwrap();
        assert_eq!(buffer.precision(), Precision::Accurate64);
        assert_eq!(buffer.bytes(), 64);
        assert!(buffer.as_f64().is_some());
        assert!(buffer.as_f32().is_none(), "must not widen on request");

        let fast = CpuDevice::new(Executor::sequential(), Precision::Fast32).unwrap();
        let buffer = fast.alloc(8, Usage::Resident).unwrap();
        assert_eq!(buffer.bytes(), 32);
        assert!(buffer.as_f32().is_some());
        assert!(buffer.as_f64().is_none(), "must not widen on request");
    }

    #[test]
    fn a_mismatched_transfer_is_an_error_rather_than_a_truncation() {
        let device = CpuDevice::reference();
        let mut buffer = device.alloc(4, Usage::Upload).unwrap();
        let error = device.write(&mut buffer, &[1.0, 2.0]).unwrap_err();
        assert_eq!(error, DeviceError::LengthMismatch { buffer: 4, host: 2 });
    }

    #[test]
    fn usage_survives_allocation() {
        let device = CpuDevice::reference();
        assert_eq!(device.alloc(1, Usage::Readback).unwrap().usage(), Usage::Readback);
        assert_eq!(device.alloc(1, Usage::Upload).unwrap().usage(), Usage::Upload);
    }
}
