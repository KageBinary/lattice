//! Explicit diffusion, resident on the device.
//!
//! The point of this solver is not that one stencil runs on a GPU. It is that a *run* does:
//! the field is uploaded once, `steps` steps execute without touching host memory, and the
//! result comes back once. A kernel that round-tripped every step would measure the PCIe
//! bus and would answer none of the questions M4.2 exists to answer.
//!
//! # What it deliberately does not do
//!
//! **Only insulated boundaries.** Zero-gradient Neumann is a copy; Dirichlet, Robin and
//! periodic each need their own halo kernel and their own test. Asking for one of the
//! others is [an error](DiffusionError::UnsupportedBoundary) rather than a silent
//! substitution, because a boundary condition quietly replaced by a different one produces
//! a run that looks entirely plausible and is wrong — the failure mode `HaloMode` was
//! introduced to prevent on the CPU side.
//!
//! **Only the explicit scheme.** Crank–Nicolson needs conjugate gradient, and CG needs
//! inner products, and a reduction is where `Mechanism::ReductionOrder` stops being zero.
//! That is a real piece of work with a real decision in it and it is not this one.

use lattice_compute::{Device, DeviceError, KernelKey, KernelSource, Precision, Usage};

use crate::device::{GpuDevice, WORKGROUP};

/// Everything the solver needs, in plain host data.
///
/// Deliberately not a `HeatDomain`. This crate sits beside `lattice-cpu` at the bottom of
/// the workspace and does not depend on the IR, so the caller marshals. That keeps `wgpu`
/// — a few hundred crates — out of every dependency graph that does not ask for it, which
/// is the property §24.1's dependency policy is protecting.
#[derive(Clone, Copy, Debug)]
pub struct DiffusionSetup<'a> {
    /// Interior cells along x.
    pub nx: usize,
    /// Interior cells along y.
    pub ny: usize,
    /// Ghost cells on each side. Must be at least 1.
    pub halo: usize,
    /// Elements between vertically adjacent cells: `nx + 2 * halo`.
    pub stride: usize,
    /// `1 / dx²`.
    pub inv_dx2: f64,
    /// `1 / dy²`.
    pub inv_dy2: f64,
    /// The whole field buffer including halo: `stride * (ny + 2 * halo)` elements.
    pub field: &'a [f64],
    /// Harmonic-mean face coefficients along x: `(nx + 1) * ny` elements.
    pub face_x: &'a [f64],
    /// Harmonic-mean face coefficients along y: `nx * (ny + 1)` elements.
    pub face_y: &'a [f64],
    /// An optional source term, laid out exactly like `field`.
    pub source: Option<&'a [f64]>,
}

/// Why a device-resident diffusion run could not be set up or executed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DiffusionError {
    /// The device could not do something.
    Device(DeviceError),
    /// A buffer's length did not match the geometry it was described by.
    Shape {
        what: &'static str,
        expected: usize,
        given: usize,
    },
    /// A boundary condition this solver does not implement.
    UnsupportedBoundary(&'static str),
    /// Geometry that cannot be run: no cells, or no halo for the stencil to read.
    Geometry(String),
}

impl core::fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DiffusionError::Device(error) => write!(f, "{error}"),
            DiffusionError::Shape {
                what,
                expected,
                given,
            } => {
                write!(f, "{what} should have {expected} elements but has {given}")
            }
            DiffusionError::UnsupportedBoundary(name) => write!(
                f,
                "the wgpu diffusion solver implements insulated boundaries only, not {name}"
            ),
            DiffusionError::Geometry(detail) => f.write_str(detail),
        }
    }
}

impl std::error::Error for DiffusionError {}

impl From<DeviceError> for DiffusionError {
    fn from(error: DeviceError) -> DiffusionError {
        DiffusionError::Device(error)
    }
}

/// The WGSL this solver compiles. Kept as a file so §24.1's *"shader sources must be
/// cacheable and inspectable"* means an actual file someone can open.
const SHADER: &str = include_str!("diffusion.wgsl");

/// An explicit diffusion solver holding its state on the device.
#[derive(Debug)]
pub struct GpuDiffusion {
    params: wgpu::Buffer,
    // Ping-pong. `parity` says which of the two currently holds the field.
    buffers: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    halo_pipeline: wgpu::ComputePipeline,
    step_pipeline: wgpu::ComputePipeline,
    nx: usize,
    ny: usize,
    stride: usize,
    halo: usize,
    total: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    parity: usize,
    steps: usize,
    key: KernelKey,
}

impl GpuDiffusion {
    /// Upload `setup` and prepare the pipelines.
    pub fn new(
        device: &GpuDevice,
        setup: DiffusionSetup<'_>,
    ) -> Result<GpuDiffusion, DiffusionError> {
        let DiffusionSetup {
            nx,
            ny,
            halo,
            stride,
            ..
        } = setup;

        let total = crate::geometry::grid(device, nx, ny, halo, stride)
            .map_err(DiffusionError::Geometry)?;
        crate::geometry::coefficients(setup.inv_dx2, setup.inv_dy2, setup.face_x, setup.face_y)
            .map_err(DiffusionError::Geometry)?;
        check(setup.field.len(), total, "the field")?;
        check(setup.face_x.len(), (nx + 1) * ny, "face_x")?;
        check(setup.face_y.len(), nx * (ny + 1), "face_y")?;
        if let Some(source) = setup.source {
            check(source.len(), total, "the source")?;
        }

        // §15.5: specialize on the dimensions and the boundary mode, and make all of it
        // part of the cache key. Two runs on differently shaped grids must not share a
        // pipeline compiled for one of them.
        let source_key = KernelSource::new("grid2d.diffusion.explicit", SHADER)
            .specialize("nx", nx)
            .specialize("ny", ny)
            .specialize("halo", halo)
            .specialize("boundary", "insulated")
            .specialize("workgroup", WORKGROUP);
        let key = KernelKey::new(
            &source_key,
            device.backend(),
            device.precision(),
            device.capabilities(),
        );
        let kernels = device.pipeline_set(
            key,
            &source_key,
            &[
                crate::bindings::uniform(0),
                crate::bindings::storage(1, false),
                crate::bindings::storage(2, false),
                crate::bindings::storage(3, true),
                crate::bindings::storage(4, true),
                crate::bindings::storage(5, true),
            ],
            &["halo_insulated", "step"],
        )?;
        let layout = kernels.layout;
        let halo_pipeline = kernels.pipelines[0].clone();
        let step_pipeline = kernels.pipelines[1].clone();

        // Buffers. `dt` is filled in per run, so params is written again in `run`.
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diffusion params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut field_a = device.alloc(total, Usage::Resident)?;
        device.write(&mut field_a, setup.field)?;
        let field_b = device.alloc(total, Usage::Resident)?;

        let mut face_x = device.alloc((nx + 1) * ny, Usage::Upload)?;
        device.write(&mut face_x, setup.face_x)?;
        let mut face_y = device.alloc(nx * (ny + 1), Usage::Upload)?;
        device.write(&mut face_y, setup.face_y)?;

        let mut source = device.alloc(total, Usage::Upload)?;
        if let Some(values) = setup.source {
            device.write(&mut source, values)?;
        } else {
            device.write(&mut source, &vec![0.0; total])?;
        }

        let buffers = [field_a.buffer, field_b.buffer];
        let bind_groups = [
            bind_group(
                device,
                &layout,
                &params,
                &buffers[0],
                &buffers[1],
                &face_x.buffer,
                &face_y.buffer,
                &source.buffer,
            ),
            bind_group(
                device,
                &layout,
                &params,
                &buffers[1],
                &buffers[0],
                &face_x.buffer,
                &face_y.buffer,
                &source.buffer,
            ),
        ];

        // The face and source buffers are owned by the bind groups from here on. wgpu
        // resources are reference-counted handles, so dropping the `GpuBuffer` wrappers
        // does not free the memory the bind groups still refer to.
        Ok(GpuDiffusion {
            params,
            buffers,
            bind_groups,
            halo_pipeline,
            step_pipeline,
            nx,
            ny,
            stride,
            halo,
            total,
            inv_dx2: setup.inv_dx2,
            inv_dy2: setup.inv_dy2,
            parity: 0,
            steps: 0,
            key,
        })
    }

    /// Advance `steps` steps of size `dt`, entirely on the device.
    pub fn run(&mut self, device: &GpuDevice, dt: f64, steps: usize) -> Result<(), DiffusionError> {
        if steps == 0 {
            return Ok(());
        }

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("diffusion"),
            });
        self.encode(device, &mut encoder, dt, steps);
        device.submit_and_wait(encoder)?;
        Ok(())
    }

    /// Record compute into a shared command stream; submit before rendering on the
    /// same queue. One timestep per submission: uniform writes precede all its commands.
    pub fn encode(
        &mut self,
        device: &GpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        dt: f64,
        steps: usize,
    ) {
        device
            .queue
            .write_buffer(&self.params, 0, &self.params_bytes(dt));

        let halo_threads = (2 * self.nx + 2 * self.ny) as u32;
        let halo_groups = halo_threads.div_ceil(64);
        let groups_x = (self.nx as u32).div_ceil(WORKGROUP);
        let groups_y = (self.ny as u32).div_ceil(WORKGROUP);

        // One encoder for the whole run. Every step is two dispatches against the same
        // command buffer, so the per-step cost is a dispatch and not a submission.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diffusion"),
                timestamp_writes: None,
            });
            for _ in 0..steps {
                let bind = &self.bind_groups[self.parity];
                pass.set_bind_group(0, bind, &[]);
                // Fill the ghost ring of whichever buffer currently holds the field, then
                // read it into the other. Dispatches inside one pass execute in order with
                // a memory barrier between them, which is what makes this correct without
                // an explicit fence.
                pass.set_pipeline(&self.halo_pipeline);
                pass.dispatch_workgroups(halo_groups, 1, 1);
                pass.set_pipeline(&self.step_pipeline);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
                self.parity ^= 1;
            }
        }
        self.steps += steps;
    }

    /// Both ping-pong fields, for persistent renderer bind groups.
    pub fn buffers(&self) -> [&wgpu::Buffer; 2] {
        [&self.buffers[0], &self.buffers[1]]
    }
    pub fn parity(&self) -> usize {
        self.parity
    }

    /// Read the field back, halo included, in the host's `f64`.
    pub fn field(&self, device: &GpuDevice) -> Result<Vec<f64>, DiffusionError> {
        let mut out = vec![0.0; self.total];
        device.read_raw(&self.buffers[self.parity], &mut out)?;
        Ok(out)
    }

    /// The interior cells only, row-major, `nx * ny` of them.
    pub fn interior(&self, device: &GpuDevice) -> Result<Vec<f64>, DiffusionError> {
        let whole = self.field(device)?;
        let origin = self.halo * self.stride + self.halo;
        let mut out = Vec::with_capacity(self.nx * self.ny);
        for j in 0..self.ny {
            let row = origin + j * self.stride;
            out.extend_from_slice(&whole[row..row + self.nx]);
        }
        Ok(out)
    }

    /// How many steps have run.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The cache key its pipelines were compiled under (§15.5).
    pub fn kernel_key(&self) -> KernelKey {
        self.key
    }

    fn params_bytes(&self, dt: f64) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        let origin = self.halo * self.stride + self.halo;
        bytes[0..4].copy_from_slice(&(self.nx as u32).to_le_bytes());
        bytes[4..8].copy_from_slice(&(self.ny as u32).to_le_bytes());
        bytes[8..12].copy_from_slice(&(self.stride as u32).to_le_bytes());
        bytes[12..16].copy_from_slice(&(origin as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(self.inv_dx2 as f32).to_le_bytes());
        bytes[20..24].copy_from_slice(&(self.inv_dy2 as f32).to_le_bytes());
        bytes[24..28].copy_from_slice(&(dt as f32).to_le_bytes());
        bytes
    }
}

fn check(given: usize, expected: usize, what: &'static str) -> Result<(), DiffusionError> {
    if given == expected {
        Ok(())
    } else {
        Err(DiffusionError::Shape {
            what,
            expected,
            given,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_group(
    device: &GpuDevice,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    face_x: &wgpu::Buffer,
    face_y: &wgpu::Buffer,
    source: &wgpu::Buffer,
) -> wgpu::BindGroup {
    use crate::bindings::entry;
    device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("diffusion"),
        layout,
        entries: &[
            entry(0, params),
            entry(1, src),
            entry(2, dst),
            entry(3, face_x),
            entry(4, face_y),
            entry(5, source),
        ],
    })
}

/// The precision this solver runs at, stated so a caller building a
/// [`Tolerance`](lattice_compute::Tolerance) does not have to guess.
pub const SOLVER_PRECISION: Precision = Precision::Fast32;
