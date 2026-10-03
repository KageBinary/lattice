//! Crank–Nicolson diffusion on the device, and the conjugate gradient that makes it
//! implicit.
//!
//! `GpuDiffusion` is one stencil per step and no reduction. This is the other kind of
//! solver: a linear system per step, solved iteratively, where the number of iterations is
//! decided by a dot product. §10.3 makes both of those visible —
//!
//! > conjugate gradient for symmetric positive-definite systems […] residual histories are
//! > always observable and can stop a run on divergence.
//!
//! — and the second half is the sentence that shapes everything here.
//!
//! # The one thing that has to come home
//!
//! A device-resident explicit run touches host memory twice: once up, once down. An
//! implicit run cannot, because *stopping* is a host decision and the quantity it depends
//! on is computed on the device. So `α` and `β` stay on the device — they are computed by
//! one-invocation kernels reading the same scalar buffer the reductions write — and exactly
//! one thing crosses per iteration: eight bytes holding `‖r‖²` and `pᵀAp`.
//!
//! Getting that to *one* readback rather than two took the trick in `iterating()`. The CPU
//! checks `pᵀAp > 0` before applying the step, and reports divergence without having
//! applied it. The device cannot ask the host mid-command-buffer, so the update kernels
//! carry the test themselves: a whole iteration is dispatched at once, and if the curvature
//! turns out to be non-positive the updates have already declined to run. The host learns
//! about it one readback later and finds `x` and `r` exactly where the CPU would have left
//! them.
//!
//! # What it deliberately does not do
//!
//! **Insulated and Dirichlet boundaries.** Conjugate gradient reads homogeneous
//! ghosts. The right-hand side adds the affine boundary contribution, matching the
//! CPU's `HaloMode` split. Robin, periodic and nonzero Neumann remain unsupported.
//!
//! **Only a tolerance it can actually reach.** See [`GpuCrankNicolson::tolerance_floor`].
//! This is the finding of the milestone and it is enforced rather than documented.

use lattice_compute::{Device, DeviceError, KernelKey, KernelSource, Precision, Usage};

use crate::device::GpuDevice;
use crate::reduction::{DotPipelines, GpuDot, Interior};

/// Slots in the device-side scalar buffer. Kept in step with the constants at the top of
/// `crank_nicolson.wgsl`.
mod scalar {
    /// Written by `compute_beta` and read by `compute_alpha`, both on the device, so the
    /// host never touches this slot. It is declared anyway because
    /// `the_scalar_slots_match_the_shader` checks the layout against the shader's own text,
    /// and a layout with a hole in it is one the test cannot check.
    #[allow(dead_code)]
    pub(super) const RS_OLD: u32 = 0;
    pub(super) const RS_NEW: u32 = 1;
    pub(super) const CURVATURE: u32 = 2;
    pub(super) const B_DOT: u32 = 5;
    pub(super) const COUNT: usize = 8;
}

/// Everything the solver needs, in plain host data.
///
/// Deliberately not a `HeatDomain`, for the dependency reason [`DiffusionSetup`](crate::DiffusionSetup) gives:
/// this crate sits at the bottom of the workspace and the caller marshals.
#[derive(Clone, Copy, Debug)]
pub struct CrankNicolsonSetup<'a> {
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
    /// Implicitness. `0.5` is Crank–Nicolson, `1.0` backward Euler. Must be in `(0, 1]`;
    /// `0` is the explicit scheme and belongs to [`GpuDiffusion`](crate::GpuDiffusion),
    /// which does it without a solve.
    pub theta: f64,
    /// The timestep, fixed for the solver's lifetime so that
    /// [`GpuCrankNicolson::tolerance_floor`] can be checked once, at setup, rather than
    /// discovered in the middle of a run.
    pub dt: f64,
    /// The whole field buffer including halo: `stride * (ny + 2 * halo)` elements.
    pub field: &'a [f64],
    /// Harmonic-mean face coefficients along x: `(nx + 1) * ny` elements.
    pub face_x: &'a [f64],
    /// Harmonic-mean face coefficients along y: `nx * (ny + 1)` elements.
    pub face_y: &'a [f64],
    /// An optional source term, laid out exactly like `field`.
    pub source: Option<&'a [f64]>,
    /// Relative residual tolerance: CG stops at `‖r‖ ≤ tolerance · ‖b‖`.
    pub tolerance: f64,
    /// Iteration cap per step.
    pub max_iterations: usize,
}

/// How a conjugate-gradient solve ended.
///
/// Mirrors `lattice_ir::SolveOutcome` variant for variant. It is a separate type because
/// this crate does not depend on the IR — §24.1's dependency policy keeps `wgpu` out of
/// every graph that has not asked for it — and the caller that owns both maps between them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CgOutcome {
    /// `‖r‖ ≤ tolerance · ‖b‖` was reached.
    Converged { iterations: usize, residual: f64 },
    /// The curvature `pᵀAp` was not positive, or the residual stopped being finite.
    Diverged { iterations: usize, residual: f64 },
    /// The iteration cap was hit first.
    MaxIterations {
        iterations: usize,
        residual: f64,
        target: f64,
    },
}

impl CgOutcome {
    /// True only for [`CgOutcome::Converged`].
    pub fn is_converged(self) -> bool {
        matches!(self, CgOutcome::Converged { .. })
    }

    /// Iterations performed.
    pub fn iterations(self) -> usize {
        match self {
            CgOutcome::Converged { iterations, .. }
            | CgOutcome::Diverged { iterations, .. }
            | CgOutcome::MaxIterations { iterations, .. } => iterations,
        }
    }

    /// The residual norm it finished at.
    pub fn residual(self) -> f64 {
        match self {
            CgOutcome::Converged { residual, .. }
            | CgOutcome::Diverged { residual, .. }
            | CgOutcome::MaxIterations { residual, .. } => residual,
        }
    }
}

impl core::fmt::Display for CgOutcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CgOutcome::Converged {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "converged in {iterations} iterations at residual {residual:.3e}"
                )
            }
            CgOutcome::Diverged {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "diverged after {iterations} iterations at residual {residual:.3e}"
                )
            }
            CgOutcome::MaxIterations {
                iterations,
                residual,
                target,
            } => write!(
                f,
                "hit the cap of {iterations} iterations at residual {residual:.3e}, \
                 needing {target:.3e}"
            ),
        }
    }
}

/// Why a device-resident implicit run could not be set up or executed.
#[derive(Clone, PartialEq, Debug)]
pub enum ImplicitError {
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
    /// Geometry that cannot be run.
    Geometry(String),
    /// `theta` outside `(0, 1]`.
    Implicitness(f64),
    /// A residual tolerance below what this precision can reach.
    ///
    /// The one error here that is a *finding* rather than a mistake. See
    /// [`GpuCrankNicolson::tolerance_floor`].
    ToleranceBelowFloor {
        requested: f64,
        floor: f64,
        operator_norm: f64,
        precision: Precision,
    },
}

impl core::fmt::Display for ImplicitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImplicitError::Device(error) => write!(f, "{error}"),
            ImplicitError::Shape {
                what,
                expected,
                given,
            } => {
                write!(f, "{what} should have {expected} elements but has {given}")
            }
            ImplicitError::UnsupportedBoundary(name) => write!(
                f,
                "the wgpu Crank-Nicolson solver implements insulated boundaries only, not {name}"
            ),
            ImplicitError::Geometry(detail) => f.write_str(detail),
            ImplicitError::Implicitness(theta) => write!(
                f,
                "theta must be in (0, 1] for an implicit scheme; {theta} was given, and \
                 theta = 0 is the explicit scheme, which needs no solve"
            ),
            ImplicitError::ToleranceBelowFloor {
                requested,
                floor,
                operator_norm,
                precision,
            } => write!(
                f,
                "a relative residual of {requested:.3e} is below what {precision} can reach: \
                 the floor is {floor:.3e} = eps * (1 + ||A||) with ||A|| <= {operator_norm:.3}, \
                 because the residual itself cannot be formed more accurately than that"
            ),
        }
    }
}

impl std::error::Error for ImplicitError {}

impl From<DeviceError> for ImplicitError {
    fn from(error: DeviceError) -> ImplicitError {
        ImplicitError::Device(error)
    }
}

const SHADER: &str = include_str!("crank_nicolson.wgsl");

/// Every compute entry point in `crank_nicolson.wgsl`, in the order they are created.
const ENTRY_POINTS: [&str; 13] = [
    "halo_p",
    "copy_x_to_p",
    "copy_r_to_p",
    "zero_x",
    "assemble_rhs",
    "apply_a",
    "sub_r",
    "axpy_x",
    "axpy_r",
    "xpby_p",
    "seed_rs_old",
    "compute_alpha",
    "compute_beta",
];

/// Index into [`ENTRY_POINTS`], and therefore into `GpuCrankNicolson::pipelines`.
#[derive(Clone, Copy)]
enum Kernel {
    HaloP = 0,
    CopyXToP = 1,
    CopyRToP = 2,
    ZeroX = 3,
    AssembleRhs = 4,
    ApplyA = 5,
    SubR = 6,
    AxpyX = 7,
    AxpyR = 8,
    XpbyP = 9,
    SeedRsOld = 10,
    ComputeAlpha = 11,
    ComputeBeta = 12,
}

/// An implicit diffusion solver holding its state, its workspace and its scalars on the
/// device.
#[derive(Debug)]
pub struct GpuCrankNicolson {
    fields: [wgpu::Buffer; 5],
    scalars: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipelines: Vec<wgpu::ComputePipeline>,
    dots: DotPipelines,
    dot_bb: GpuDot,
    dot_rr: GpuDot,
    dot_pap: GpuDot,

    interior: Interior,
    total: usize,
    tolerance: f64,
    max_iterations: usize,
    operator_norm: f64,

    history: Vec<f64>,
    last_outcome: Option<CgOutcome>,
    steps: usize,
    non_converged: usize,
    iterations: usize,
    readbacks: usize,
    largest_rhs_norm: f64,
    key: KernelKey,
}

// Buffer slots in `fields`, matching bindings 1..=5 in the shader.
const X: usize = 0;
const B: usize = 1;
const R: usize = 2;
const P: usize = 3;
const AP: usize = 4;

impl GpuCrankNicolson {
    /// Upload `setup`, compile the kernels, and check the tolerance against the floor.
    pub fn new(
        device: &GpuDevice,
        setup: CrankNicolsonSetup<'_>,
    ) -> Result<GpuCrankNicolson, ImplicitError> {
        Self::with_dirichlet(device, setup, [None; 4])
    }

    /// Prescribed face values in left, right, bottom, top order. `None` is insulated.
    /// The linear halo and affine right-hand side are kept separate, as on the CPU.
    pub fn with_dirichlet(
        device: &GpuDevice,
        setup: CrankNicolsonSetup<'_>,
        boundaries: [Option<f64>; 4],
    ) -> Result<GpuCrankNicolson, ImplicitError> {
        let CrankNicolsonSetup {
            nx,
            ny,
            halo,
            stride,
            theta,
            dt,
            ..
        } = setup;

        let total =
            crate::geometry::grid(device, nx, ny, halo, stride).map_err(ImplicitError::Geometry)?;
        crate::geometry::coefficients(setup.inv_dx2, setup.inv_dy2, setup.face_x, setup.face_y)
            .map_err(ImplicitError::Geometry)?;
        if !(theta > 0.0 && theta <= 1.0) {
            return Err(ImplicitError::Implicitness(theta));
        }
        if !(dt.is_finite() && dt > 0.0 && (dt as f32).is_normal()) {
            return Err(ImplicitError::Geometry(format!(
                "dt must be positive and finite, not {dt}"
            )));
        }

        check(setup.field.len(), total, "the field")?;
        check(setup.face_x.len(), (nx + 1) * ny, "face_x")?;
        check(setup.face_y.len(), nx * (ny + 1), "face_y")?;
        if let Some(source) = setup.source {
            check(source.len(), total, "the source")?;
        }
        if !setup.tolerance.is_finite() || setup.tolerance <= 0.0 {
            return Err(ImplicitError::Geometry(
                "tolerance must be positive and finite".into(),
            ));
        }
        if boundaries
            .iter()
            .flatten()
            .any(|&v| !(2.0 * v).is_finite() || !(2.0 * v as f32).is_finite())
        {
            return Err(ImplicitError::Geometry(
                "boundary values must be representable in f32".into(),
            ));
        }

        // The refusal, before anything is allocated. `theta * dt` is what makes the
        // operator, so the floor is knowable here and nowhere earlier.
        let coefficient = theta * dt;
        let operator_norm = operator_norm_bound(&setup, coefficient);
        let precision = device.precision();
        let floor = precision.epsilon() * (1.0 + operator_norm);
        if setup.tolerance < floor {
            return Err(ImplicitError::ToleranceBelowFloor {
                requested: setup.tolerance,
                floor,
                operator_norm,
                precision,
            });
        }

        let interior = Interior {
            nx,
            ny,
            stride,
            origin: halo * stride + halo,
        };

        // §15.5: the dimensions, the boundary mode and the scheme are all specializations,
        // and all of them are in the key. `theta` belongs there as much as `nx` does — a
        // module compiled for backward Euler must not be reused for Crank–Nicolson.
        let source_key = KernelSource::new("grid2d.diffusion.implicit", SHADER)
            .specialize("nx", nx)
            .specialize("ny", ny)
            .specialize("halo", halo)
            .specialize("theta", theta)
            .specialize("boundary", "insulated-or-dirichlet")
            .specialize("workgroup", 8);
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
                crate::bindings::storage(3, false),
                crate::bindings::storage(4, false),
                crate::bindings::storage(5, false),
                crate::bindings::storage(6, true),
                crate::bindings::storage(7, false),
            ],
            &ENTRY_POINTS,
        )?;
        let layout = kernels.layout;
        let pipelines = kernels.pipelines;

        // face_x ++ face_y ++ source in one buffer. Three separate ones would put the
        // shader at nine storage bindings and WebGPU's portable floor is eight.
        let face_y_off = (nx + 1) * ny;
        let source_off = face_y_off + nx * (ny + 1);
        let mut packed = Vec::with_capacity(source_off + total);
        packed.extend_from_slice(setup.face_x);
        packed.extend_from_slice(setup.face_y);
        match setup.source {
            Some(values) => packed.extend_from_slice(values),
            None => packed.resize(source_off + total, 0.0),
        }
        let coeffs = device.upload(&packed, Usage::Upload)?;

        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crank-nicolson params"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(
            &params,
            0,
            &boundary_params(
                &setup,
                interior,
                coefficient,
                face_y_off,
                source_off,
                boundaries,
            ),
        );

        let mut x = device.alloc(total, Usage::Readback)?;
        device.write(&mut x, setup.field)?;
        let make = |usage| device.alloc(total, usage);
        let fields = [
            x.buffer,
            make(Usage::Resident)?.buffer,
            make(Usage::Resident)?.buffer,
            make(Usage::Resident)?.buffer,
            make(Usage::Resident)?.buffer,
        ];
        let scalars = device
            .upload(&[0.0; scalar::COUNT], Usage::Readback)?
            .buffer;

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("crank-nicolson"),
            layout: &layout,
            entries: &[
                crate::bindings::entry(0, &params),
                crate::bindings::entry(1, &fields[X]),
                crate::bindings::entry(2, &fields[B]),
                crate::bindings::entry(3, &fields[R]),
                crate::bindings::entry(4, &fields[P]),
                crate::bindings::entry(5, &fields[AP]),
                crate::bindings::entry(6, &coeffs.buffer),
                crate::bindings::entry(7, &scalars),
            ],
        });

        // The staging buffer the per-iteration readback maps, allocated once.
        //
        // NFR-001 asks for no heap allocation in a validated hot loop, and creating a wgpu
        // buffer per conjugate-gradient iteration is exactly that with a driver round trip
        // attached. See `submit_and_read`.
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crank-nicolson scalars"),
            size: (scalar::COUNT * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dots = DotPipelines::new(device, interior)?;
        let dot = |a: &wgpu::Buffer, b: &wgpu::Buffer, slot: u32| {
            GpuDot::new(device, &dots, interior, a, b, &scalars, slot)
        };
        let dot_bb = dot(&fields[B], &fields[B], scalar::B_DOT);
        let dot_rr = dot(&fields[R], &fields[R], scalar::RS_NEW);
        let dot_pap = dot(&fields[P], &fields[AP], scalar::CURVATURE);

        Ok(GpuCrankNicolson {
            fields,
            scalars,
            staging,
            bind_group,
            pipelines,
            dots,
            dot_bb,
            dot_rr,
            dot_pap,
            interior,
            total,
            tolerance: setup.tolerance,
            max_iterations: setup.max_iterations,
            operator_norm,
            history: Vec::new(),
            last_outcome: None,
            steps: 0,
            non_converged: 0,
            iterations: 0,
            readbacks: 0,
            largest_rhs_norm: 0.0,
            key,
        })
    }

    /// The smallest relative residual this precision can reach, and therefore the smallest
    /// one the solver will accept.
    ///
    /// **This is the finding of M4.4, and it is why an implicit GPU solve cannot simply be
    /// held to the CPU's settings.** `HeatDomain` defaults to `1e-10`, which is a reasonable
    /// ask of an `f64` path and is three orders of magnitude below anything `f32` can
    /// represent about this problem.
    ///
    /// The derivation is short. CG stops on the *recursively updated* residual, but what a
    /// caller is promised is a statement about the true one, `b − Ax`. Even given a
    /// perfectly exact `x`, forming that difference in a precision with unit round-off `ε`
    /// leaves an error of about `ε·(‖b‖ + ‖A‖·‖x‖)`. For `A = I − θ·dt·L` with `L` negative
    /// semi-definite every eigenvalue of `A` is at least 1, so `‖A⁻¹‖₂ ≤ 1` and therefore
    /// `‖x‖ ≤ ‖b‖`. Dividing through by `‖b‖`:
    ///
    /// ```text
    /// floor = ε · (1 + ‖A‖₂)
    /// ```
    ///
    /// `‖A‖₂` is bounded above by Gershgorin over the assembled rows — see
    /// [`GpuCrankNicolson::operator_norm_bound`] — so the whole number is computable at
    /// setup from the face coefficients and `θ·dt`, with nothing fitted and nothing
    /// measured.
    ///
    /// Asking for less is not a slow run, it is a run that iterates to its cap every step
    /// and reports `MaxIterations` while looking like a performance problem. Refusing is
    /// [`ImplicitError::ToleranceBelowFloor`], which names the floor and the `‖A‖` it came
    /// from.
    pub fn tolerance_floor(&self) -> f64 {
        Precision::Fast32.epsilon() * (1.0 + self.operator_norm)
    }

    /// [`GpuCrankNicolson::tolerance_floor`] for a setup that has not been built yet.
    ///
    /// Without this, the floor is only reachable through a solver whose construction
    /// *refuses* tolerances below it, and a caller wanting to pick a legal one would have to
    /// guess and retry. `setup.tolerance` is ignored.
    pub fn floor_for(precision: Precision, setup: &CrankNicolsonSetup<'_>) -> f64 {
        precision.epsilon() * (1.0 + operator_norm_bound(setup, setup.theta * setup.dt))
    }

    /// A Gershgorin bound on `‖A‖₂`, where `A = I − θ·dt·L`.
    ///
    /// Row `k` of `L` has diagonal `−Σ face·inv` and off-diagonals the same faces with the
    /// opposite sign, so the row's absolute sum is twice the face total and Gershgorin
    /// gives `|λ(L)| ≤ max_k 2·[(fx_w + fx_e)·inv_dx2 + (fy_s + fy_n)·inv_dy2]`. An upper
    /// bound is what the floor needs, and this one is loose only where the boundary
    /// coefficients are, which is a ring of cells.
    pub fn operator_norm_bound(&self) -> f64 {
        self.operator_norm
    }

    /// Advance one step, returning how the solve ended.
    pub fn step(&mut self, device: &GpuDevice) -> Result<CgOutcome, ImplicitError> {
        self.history.clear();

        // Everything up to and including the first residual is one command buffer. There is
        // nothing to decide in it, so there is nothing to wait for.
        let mut encoder = self.encoder(device, "cn setup");
        {
            let mut pass = self.pass(&mut encoder, "cn setup");
            // rhs = u^n + dt·[(1−θ)·L(u^n) + S], with u^n staged into `p` for its halo.
            self.field_kernel(&mut pass, Kernel::CopyXToP);
            self.halo(&mut pass);
            self.field_kernel(&mut pass, Kernel::AssembleRhs);
            self.dot_bb.record(&self.dots, &mut pass);

            // r = b − A·x. `assemble_rhs` writes only `b`, so `p` still holds `x` with its
            // halo filled and the copy does not need repeating — an implicit step is
            // already four times the dispatches of an explicit one.
            pass.set_bind_group(0, &self.bind_group, &[]);
            self.field_kernel(&mut pass, Kernel::ApplyA);
            self.field_kernel(&mut pass, Kernel::SubR);
            self.field_kernel(&mut pass, Kernel::CopyRToP);
            self.dot_rr.record(&self.dots, &mut pass);
            pass.set_bind_group(0, &self.bind_group, &[]);
            self.scalar_kernel(&mut pass, Kernel::SeedRsOld);
        }
        let mut opening = [0.0; 5];
        self.submit_and_read(device, encoder, scalar::RS_NEW, &mut opening)?;
        let (rs_initial, b_squared) = (opening[0], opening[4]);
        let b_norm = residual_of(b_squared);
        self.largest_rhs_norm = self.largest_rhs_norm.max(b_norm);

        self.steps += 1;

        // The zero vector solves `A·x = 0` exactly, and `target` would otherwise be zero.
        if b_norm == 0.0 {
            let mut encoder = self.encoder(device, "cn zero");
            {
                let mut pass = self.pass(&mut encoder, "cn zero");
                self.field_kernel(&mut pass, Kernel::ZeroX);
            }
            device
                .submit_and_wait(encoder)
                .map_err(ImplicitError::Device)?;
            self.history.push(0.0);
            return Ok(self.finish(CgOutcome::Converged {
                iterations: 0,
                residual: 0.0,
            }));
        }

        let target = self.tolerance * b_norm;
        let mut r_norm = residual_of(rs_initial);
        self.history.push(r_norm);
        if meets(r_norm, target) {
            return Ok(self.finish(CgOutcome::Converged {
                iterations: 0,
                residual: r_norm,
            }));
        }

        for iteration in 1..=self.max_iterations {
            // One command buffer for the whole iteration. The guards in `axpy_x`, `axpy_r`,
            // `xpby_p` and `compute_beta` are what make that safe: a non-positive curvature
            // means none of them ran.
            let mut encoder = self.encoder(device, "cn iteration");
            {
                let mut pass = self.pass(&mut encoder, "cn iteration");
                self.halo(&mut pass);
                self.field_kernel(&mut pass, Kernel::ApplyA);
                self.dot_pap.record(&self.dots, &mut pass);
                pass.set_bind_group(0, &self.bind_group, &[]);
                self.scalar_kernel(&mut pass, Kernel::ComputeAlpha);
                self.field_kernel(&mut pass, Kernel::AxpyX);
                self.field_kernel(&mut pass, Kernel::AxpyR);
                self.dot_rr.record(&self.dots, &mut pass);
                pass.set_bind_group(0, &self.bind_group, &[]);
                self.scalar_kernel(&mut pass, Kernel::ComputeBeta);
                self.field_kernel(&mut pass, Kernel::XpbyP);
            }
            // The one crossing. Slots 1 and 2 are adjacent so that it is one copy, and the
            // copy rides on the iteration's own submission so that it is one fence.
            let mut pair = [0.0; 2];
            self.submit_and_read(device, encoder, scalar::RS_NEW, &mut pair)?;
            let (rs_new, curvature) = (pair[0], pair[1]);
            self.iterations += 1;

            // For an SPD operator the curvature is strictly positive; anything else means
            // the operator is not what CG assumes (NFR-007). `r` is untouched in that case,
            // so `r_norm` is still the previous iteration's — which is exactly what
            // `conjugate_gradient` reports.
            if !curvature.is_finite() || curvature <= 0.0 {
                return Ok(self.finish(CgOutcome::Diverged {
                    iterations: iteration - 1,
                    residual: r_norm,
                }));
            }

            r_norm = residual_of(rs_new);
            self.history.push(r_norm);
            if !r_norm.is_finite() {
                return Ok(self.finish(CgOutcome::Diverged {
                    iterations: iteration,
                    residual: r_norm,
                }));
            }
            if meets(r_norm, target) {
                return Ok(self.finish(CgOutcome::Converged {
                    iterations: iteration,
                    residual: r_norm,
                }));
            }
        }

        Ok(self.finish(CgOutcome::MaxIterations {
            iterations: self.max_iterations,
            residual: r_norm,
            target,
        }))
    }

    /// Advance `steps` steps, returning the last one's outcome.
    ///
    /// A step that fails to converge does not stop the run, because `HeatDomain::advance`
    /// does not stop either — it records the outcome and presses on, and a GPU path that
    /// bailed out early would not be running the same thing. [`Self::non_converged_steps`]
    /// is how a caller finds out, and a comparison that ignores it is comparing against a
    /// run that gave up.
    pub fn run(&mut self, device: &GpuDevice, steps: usize) -> Result<CgOutcome, ImplicitError> {
        let mut last = self.last_outcome.unwrap_or(CgOutcome::Converged {
            iterations: 0,
            residual: 0.0,
        });
        for _ in 0..steps {
            last = self.step(device)?;
        }
        Ok(last)
    }

    /// Read the field back, halo included, in the host's `f64`.
    pub fn field(&self, device: &GpuDevice) -> Result<Vec<f64>, ImplicitError> {
        let mut out = vec![0.0; self.total];
        device.read_raw(&self.fields[X], &mut out)?;
        Ok(out)
    }

    /// Resident field for rendering; interior layout is the one supplied at setup.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.fields[X]
    }

    /// The interior cells only, row-major, `nx * ny` of them.
    pub fn interior(&self, device: &GpuDevice) -> Result<Vec<f64>, ImplicitError> {
        let whole = self.field(device)?;
        let mut out = Vec::with_capacity(self.interior.cells());
        for j in 0..self.interior.ny {
            let row = self.interior.origin + j * self.interior.stride;
            out.extend_from_slice(&whole[row..row + self.interior.nx]);
        }
        Ok(out)
    }

    /// §10.3's observable residual history, for the most recent step.
    pub fn residual_history(&self) -> &[f64] {
        &self.history
    }

    /// How the most recent step's solve ended.
    pub fn last_outcome(&self) -> Option<CgOutcome> {
        self.last_outcome
    }

    /// Steps run.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The relative residual tolerance every solve stopped at.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The largest `‖b‖₂` any step's right-hand side reached.
    ///
    /// An input to [`ImplicitSolve`](lattice_compute::ImplicitSolve), whose bound is stated
    /// relative to `‖b‖₂` and has to be converted to something relative to `‖x‖₂` before it
    /// can be checked against a field. Taking the largest rather than the last is what makes
    /// it a bound over the whole run rather than a snapshot of the end of it — and it is
    /// free, because the reduction that produces it is one CG has to do anyway.
    pub fn largest_rhs_norm(&self) -> f64 {
        self.largest_rhs_norm
    }

    /// Steps whose solve did not converge. Zero is the only value a published comparison
    /// may have.
    pub fn non_converged_steps(&self) -> usize {
        self.non_converged
    }

    /// CG iterations *dispatched* across every step.
    ///
    /// One more than the outcomes report for a step that diverged, because the iteration
    /// whose curvature turned out to be non-positive was dispatched and read back before
    /// anyone could know that. It is the dispatched count that
    /// [`readbacks`](Self::readbacks) is one-to-one with, and the one a performance figure
    /// should be divided by.
    pub fn total_iterations(&self) -> usize {
        self.iterations
    }

    /// Device-to-host transfers performed, which for this solver is one per CG iteration
    /// plus one per step.
    ///
    /// Published because it is the cost of §10.3's stopping rule and the number M4.5 would
    /// have to attack. Nothing else about an implicit run touches host memory.
    pub fn readbacks(&self) -> usize {
        self.readbacks
    }

    /// Roundings any one term passes through inside a dot product. See
    /// [`Interior::depth`].
    pub fn reduction_depth(&self) -> usize {
        self.interior.depth()
    }

    /// The cache key its pipelines were compiled under (§15.5).
    pub fn kernel_key(&self) -> KernelKey {
        self.key
    }

    /// The cache key the *reduction* was compiled under.
    ///
    /// A second key rather than a component of the first, because the reduction is a
    /// separate module specialized on a different set of constants — the grid shape it
    /// shares, the split into workgroups it does not. Two solvers on the same grid with
    /// different `theta` reuse this module and not the other one, and §15.5's inspectability
    /// is worth more if that is visible than if it is merged away.
    pub fn reduction_kernel_key(&self) -> KernelKey {
        self.dots.kernel_key()
    }

    fn finish(&mut self, outcome: CgOutcome) -> CgOutcome {
        if !outcome.is_converged() {
            self.non_converged += 1;
        }
        self.last_outcome = Some(outcome);
        outcome
    }

    /// Submit `encoder` and bring `out.len()` scalars home, starting at slot `first`.
    ///
    /// Two things here are worth more than they look.
    ///
    /// **The copy rides along with the work.** Appending `copy_buffer_to_buffer` to the
    /// encoder that just recorded the iteration means one submission and one fence per
    /// iteration instead of two. A separate readback would submit the compute, wait for it,
    /// then submit a copy and wait again — and the second wait is a full round trip to a
    /// device that is already idle.
    ///
    /// **The staging buffer is allocated once.** NFR-001 asks for no heap allocation in a
    /// validated hot loop; creating a `wgpu::Buffer` per iteration is that, with a driver
    /// allocation attached. The buffer is mapped, read and unmapped each time, which is what
    /// makes it reusable as a copy destination on the next iteration.
    fn submit_and_read(
        &mut self,
        device: &GpuDevice,
        mut encoder: wgpu::CommandEncoder,
        first: u32,
        out: &mut [f64],
    ) -> Result<(), ImplicitError> {
        let bytes = out.len() as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.scalars, u64::from(first) * 4, &self.staging, 0, bytes);
        device.queue.submit(Some(encoder.finish()));

        // Mapped whole rather than to `bytes`: `map_async` wants an eight-byte-aligned size
        // and the buffer is a multiple of that, while a two-scalar read is not.
        let slice = self.staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.wait()?;
        receiver
            .recv()
            .map_err(|error| DeviceError::Backend(error.to_string()))?
            .map_err(|error| DeviceError::Backend(error.to_string()))?;

        {
            let view = slice
                .get_mapped_range()
                .map_err(|error| DeviceError::Backend(error.to_string()))?;
            for (slot, chunk) in out.iter_mut().zip(view.chunks_exact(4)) {
                let bits = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                *slot = f64::from(f32::from_bits(bits));
            }
        }
        self.staging.unmap();

        self.readbacks += 1;
        Ok(())
    }

    fn encoder(&self, device: &GpuDevice, label: &str) -> wgpu::CommandEncoder {
        device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    fn pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
    ) -> wgpu::ComputePass<'e> {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass
    }

    fn field_kernel(&self, pass: &mut wgpu::ComputePass<'_>, kernel: Kernel) {
        pass.set_pipeline(&self.pipelines[kernel as usize]);
        pass.dispatch_workgroups(
            (self.interior.nx as u32).div_ceil(8),
            (self.interior.ny as u32).div_ceil(8),
            1,
        );
    }

    fn scalar_kernel(&self, pass: &mut wgpu::ComputePass<'_>, kernel: Kernel) {
        pass.set_pipeline(&self.pipelines[kernel as usize]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn halo(&self, pass: &mut wgpu::ComputePass<'_>) {
        let threads = (2 * self.interior.nx + 2 * self.interior.ny) as u32;
        pass.set_pipeline(&self.pipelines[Kernel::HaloP as usize]);
        pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
    }
}

fn boundary_params(
    setup: &CrankNicolsonSetup<'_>,
    interior: Interior,
    coefficient: f64,
    face_y_off: usize,
    source_off: usize,
    boundaries: [Option<f64>; 4],
) -> [u8; 80] {
    let mut bytes = [0u8; 80];
    bytes[..48].copy_from_slice(&params_bytes(
        setup,
        interior,
        coefficient,
        face_y_off,
        source_off,
    ));
    bytes[44..48].copy_from_slice(&(setup.theta as f32).to_le_bytes());
    for (side, value) in boundaries.into_iter().enumerate() {
        let sign: f32 = if value.is_some() { -1.0 } else { 1.0 };
        bytes[48 + side * 4..52 + side * 4].copy_from_slice(&sign.to_le_bytes());
        bytes[64 + side * 4..68 + side * 4]
            .copy_from_slice(&(2.0 * value.unwrap_or(0.0) as f32).to_le_bytes());
    }
    bytes
}

/// Gershgorin over the assembled rows of `A = I − θ·dt·L`.
///
/// See [`GpuCrankNicolson::operator_norm_bound`]. Computed in `f64` on the host from the
/// same coefficients the device is given, because a bound derived from rounded inputs would
/// be a bound on a different matrix.
fn operator_norm_bound(setup: &CrankNicolsonSetup<'_>, coefficient: f64) -> f64 {
    let (nx, ny) = (setup.nx, setup.ny);
    let mut worst = 0.0f64;
    for j in 0..ny {
        for i in 0..nx {
            let fx = setup.face_x[j * (nx + 1) + i] + setup.face_x[j * (nx + 1) + i + 1];
            let fy = setup.face_y[j * nx + i] + setup.face_y[(j + 1) * nx + i];
            let row = 2.0 * (fx * setup.inv_dx2 + fy * setup.inv_dy2);
            worst = worst.max(row);
        }
    }
    1.0 + coefficient * worst
}

fn params_bytes(
    setup: &CrankNicolsonSetup<'_>,
    interior: Interior,
    coefficient: f64,
    face_y_off: usize,
    source_off: usize,
) -> [u8; 48] {
    let mut bytes = [0u8; 48];
    let mut put_u32 = |slot: usize, value: u32| {
        bytes[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
    };
    put_u32(0, interior.nx as u32);
    put_u32(1, interior.ny as u32);
    put_u32(2, interior.stride as u32);
    put_u32(3, interior.origin as u32);
    put_u32(8, face_y_off as u32);
    put_u32(9, source_off as u32);

    let mut put_f32 = |slot: usize, value: f64| {
        bytes[slot * 4..slot * 4 + 4].copy_from_slice(&(value as f32).to_le_bytes());
    };
    put_f32(4, setup.inv_dx2);
    put_f32(5, setup.inv_dy2);
    put_f32(6, coefficient);
    put_f32(7, 1.0 - setup.theta);
    put_f32(10, setup.dt);
    bytes
}

/// A residual norm from a device-side sum of squares.
///
/// A plain `sqrt`, and the absence of a clamp is the point. Guarding the input with
/// `rs.max(0.0)` looks like harmless defensiveness — a sum of squares cannot round negative,
/// so the clamp protects against nothing — but `f64::max` **ignores NaN and returns the
/// other operand**. A NaN residual would become `0.0`, compare as converged, and report a
/// step that produced garbage as a clean success. NFR-007 is exactly the rule against
/// confident wrong answers, so the NaN is propagated to the checks that exist to catch it.
fn residual_of(sum_of_squares: f64) -> f64 {
    sum_of_squares.sqrt()
}

/// Whether `r_norm` satisfies `target`, given that either may be non-finite.
///
/// `NaN <= target` is already false, so this looks redundant — and it is, right up until
/// something upstream launders the NaN into a number. It is spelled out because the failure
/// it prevents is silent and the cost of preventing it is one comparison per iteration.
fn meets(r_norm: f64, target: f64) -> bool {
    r_norm.is_finite() && r_norm <= target
}

fn check(given: usize, expected: usize, what: &'static str) -> Result<(), ImplicitError> {
    if given == expected {
        Ok(())
    } else {
        Err(ImplicitError::Shape {
            what,
            expected,
            given,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one coupling in this file that nothing else can catch.
    ///
    /// The host reads `scalars[1..3]` to learn the residual and the curvature, and the
    /// shader writes them by name. Swap two of the shader's constants and every kernel
    /// still compiles, every dispatch still runs, and conjugate gradient quietly checks the
    /// curvature against the residual — a wrong answer with no error anywhere. So the
    /// shader's own text is the source of truth and this test reads it.
    #[test]
    fn the_scalar_slots_match_the_shader() {
        let declared = |name: &str| -> u32 {
            let needle = format!("const {name}: u32 = ");
            let line = SHADER
                .lines()
                .find(|line| line.trim_start().starts_with(&needle))
                .unwrap_or_else(|| panic!("{name} is not declared in crank_nicolson.wgsl"));
            line.rsplit_once('=')
                .and_then(|(_, value)| {
                    value
                        .trim()
                        .trim_end_matches(';')
                        .trim_end_matches('u')
                        .parse()
                        .ok()
                })
                .unwrap_or_else(|| panic!("could not read {name}'s value from {line:?}"))
        };

        assert_eq!(declared("RS_OLD"), scalar::RS_OLD);
        assert_eq!(declared("RS_NEW"), scalar::RS_NEW);
        assert_eq!(declared("CURVATURE"), scalar::CURVATURE);
        assert_eq!(
            scalar::CURVATURE,
            scalar::RS_NEW + 1,
            "the per-iteration readback copies one adjacent pair; separating them makes it two"
        );
        assert!(
            (scalar::B_DOT as usize) < scalar::COUNT,
            "every slot must fit the buffer that is allocated for them"
        );
    }

    /// Every entry point named on the host must exist in the shader, and every entry point
    /// in the shader must be created as a pipeline. A kernel that is compiled and never
    /// dispatched is dead weight; one that is dispatched and does not exist is a panic
    /// inside `create_compute_pipeline` on a machine that has a GPU, which is not where
    /// anyone wants to find out.
    /// The bug this pair of helpers exists for, written as the thing that made it a bug.
    ///
    /// `rs.max(0.0)` stood in `step` as a clamp on a sum of squares, which cannot round
    /// negative — so it guarded nothing, and cost everything: a NaN residual came out as a
    /// clean `0.0`, compared as converged, and would have reported a step full of NaN as a
    /// success with residual zero. Only the curvature test caught it, and only because that
    /// path happened to run first.
    #[test]
    fn a_non_finite_residual_is_never_laundered_into_convergence() {
        assert_eq!(
            f64::NAN.max(0.0),
            0.0,
            "f64::max ignores NaN; this is what the clamp did"
        );

        assert!(
            residual_of(f64::NAN).is_nan(),
            "the NaN must survive to be checked"
        );
        assert!(
            !meets(residual_of(f64::NAN), 1.0),
            "a NaN residual is not convergence"
        );
        assert!(
            !meets(residual_of(f64::INFINITY), f64::INFINITY),
            "nor is an infinite one"
        );

        // And the ordinary cases still behave.
        assert!(meets(residual_of(4.0), 2.0), "2 meets a target of 2");
        assert!(!meets(residual_of(9.0), 2.0), "3 does not");
        assert_eq!(residual_of(0.0), 0.0);
    }

    #[test]
    fn the_entry_points_and_the_shader_agree() {
        // The first `fn NAME(` after each `@compute` attribute. Helper functions have no
        // attribute and so are not collected.
        let declared: Vec<&str> = SHADER
            .split("@compute")
            .skip(1)
            .filter_map(|block| block.split("fn ").nth(1))
            .filter_map(|rest| rest.split('(').next())
            .map(str::trim)
            .collect();

        for entry in ENTRY_POINTS {
            assert!(
                declared.contains(&entry),
                "{entry} is dispatched but not declared"
            );
        }
        assert_eq!(
            declared.len(),
            ENTRY_POINTS.len(),
            "shader declares {declared:?}, the host knows {ENTRY_POINTS:?}"
        );
    }
}
