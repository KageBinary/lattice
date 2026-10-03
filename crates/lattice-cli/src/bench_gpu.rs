//! Benchmarks on the portable GPU backend.
//!
//! §19.3 governs what may be published, and two of its clauses bite harder here than they
//! did on the CPU.
//!
//! **Precision is part of the result.** The portable backend runs `fast32` because WGSL has
//! no `f64`. A GPU number printed beside the CPU's is therefore not the same computation
//! done faster — it is a different computation — and [`Executed::precision`] carries that
//! into every report and every JSON record rather than leaving it to a footnote.
//!
//! **§15.1 says measure end-to-end**, *"including model compile time, upload/download,
//! solver work […] not kernel time alone."* Opening a device, compiling two pipelines and
//! uploading the field are real costs that a kernel timing hides, so they are timed as
//! `setup` and the readback as `observe`, and both are printed beside the compute figure.
//! The headline throughput remains the stepping loop, because that is what the CPU
//! benchmark measures and a comparison between differently-scoped timings is worse than no
//! comparison — but the phase table is right there, and for a short run it dominates.

use std::time::Instant;

use lattice_compute::{Device, DeviceError, Precision};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme, gaussian};
use lattice_ir::{BoundarySet, Domain, Grid2d};
use lattice_observe::{MemoryReport, Profile, Throughput, phase};
use lattice_wgpu::{CrankNicolsonSetup, DiffusionSetup, GpuCrankNicolson, GpuDevice, GpuDiffusion};

use crate::bench::{BenchOutcome, Check, Executed, Info};

#[path = "bench_gpu_particles.rs"]
mod particles;

/// A benchmark that runs on a GPU device.
#[derive(Clone, Copy)]
pub struct GpuBenchmark {
    /// Identifier, matching its CPU counterpart so the two can be compared by name.
    pub name: &'static str,
    /// What the scene is.
    pub description: &'static str,
    /// The §15.6 correctness condition.
    pub correctness: &'static str,
    /// Run at the given scale factor.
    pub run: fn(usize, &GpuDevice) -> Result<BenchOutcome, DeviceError>,
}

impl GpuBenchmark {
    /// The backend-independent description.
    pub fn info(&self) -> Info {
        Info {
            name: self.name,
            description: self.description,
            correctness: self.correctness,
        }
    }
}

/// Every benchmark with a GPU implementation.
///
/// Still short. `particles-gravity` and `particles-lj` have no GPU kernels — §15.6's "local
/// particles" and "Lennard-Jones MD" targets are GPU targets and the pair-force question is
/// still open — and listing them here with a CPU fallback behind them would report a GPU
/// number for work that never touched the GPU.
///
/// `heat-crank-nicolson` joined in M4.4. It is a different shape of benchmark from
/// everything above it: the work per step is not fixed, because the number of conjugate
/// gradient iterations depends on the problem, and each iteration stalls once on an
/// eight-byte readback that §10.3's stopping rule requires. Both are reported.
pub fn all() -> &'static [GpuBenchmark] {
    &[
        GpuBenchmark {
            name: "particles-gravity",
            description: "resident velocity-Verlet particles in uniform gravity",
            correctness: "analytic trajectories within accumulated f32 rounding",
            run: particles::gravity,
        },
        GpuBenchmark {
            name: "particles-lj",
            description: "resident velocity-Verlet Lennard-Jones particles with sorted spatial bins",
            correctness: "bounded shifted-potential energy and momentum drift; finite state",
            run: particles::lj,
        },
        GpuBenchmark {
            name: "heat-explicit",
            description: "explicit diffusion on a uniform grid, resident on the device — \
                          the §15.6 'heat/diffusion grid' target",
            correctness: "the field integral is conserved on a closed domain, to the bound \
                          f32 rounding allows",
            run: bench_heat_explicit,
        },
        GpuBenchmark {
            name: "heat-crank-nicolson",
            description: "implicit diffusion with a device-resident conjugate-gradient \
                          solve, at ten times the explicit stability limit",
            correctness: "the integral is conserved, every linear solve converged, and the \
                          residual tolerance was one f32 can reach",
            run: bench_heat_implicit,
        },
    ]
}

/// Look up GPU benchmarks whose name contains `pattern`.
pub fn matching(pattern: &str) -> Vec<GpuBenchmark> {
    all()
        .iter()
        .copied()
        .filter(|b| b.name.contains(pattern))
        .collect()
}

/// How a GPU run was executed.
pub fn executed(device: &GpuDevice) -> Executed {
    Executed {
        label: device.label(),
        precision: match device.precision() {
            Precision::Fast32 => "fast32",
            Precision::Mixed => "mixed",
            Precision::Accurate64 => "accurate64",
        },
        threads: None,
    }
}

/// The same scene the CPU benchmark builds, so the two are comparable.
fn heat_domain(side: usize) -> HeatDomain {
    let grid = Grid2d::new(side, side, [1.0, 1.0]);
    HeatDomain::new("u", grid, Diffusivity::Uniform(1e-2))
        .with_scheme(TimeScheme::Explicit)
        .with_boundaries(BoundarySet::INSULATED)
        .with_initial(gaussian([0.5, 0.5], 0.004, 1.0))
}

const STEPS: usize = 400;

fn bench_heat_explicit(scale: usize, device: &GpuDevice) -> Result<BenchOutcome, DeviceError> {
    let side = 256 * scale;
    let mut profile = Profile::new();

    // Everything lazy has to happen before the clock starts.
    //
    // `GpuDevice::open` already pays the driver's one-time transfer-path initialization,
    // which is worth about 56 ms on this machine against a 160 µs steady-state round trip.
    // This covers what is lazy *per shape* instead: the first dispatch of a given pipeline
    // at a given size, and the first allocation of buffers that large.
    //
    // A throwaway solver of the same shape runs first and is discarded. Rebuilding is cheap
    // because §15.5's kernel cache hands the second one the same compiled module — the
    // warm-up costs one extra allocation and upload, not a recompilation.
    {
        let domain = heat_domain(side);
        let dt = 0.8 * domain.stable_step().max;
        let mut warm = build_solver(device, &domain, side)?;
        warm.run(device, dt, 2)
            .map_err(|error| DeviceError::Backend(error.to_string()))?;
        warm.interior(device)
            .map_err(|error| DeviceError::Backend(error.to_string()))?;
        device.finish()?;
    }

    let setup_start = Instant::now();
    let domain = heat_domain(side);
    // 0.8 of the stability limit, matching the CPU benchmark exactly.
    let dt = 0.8 * domain.stable_step().max;
    let field = domain.field();
    let cell_area = domain.grid().cell_area();
    let before: f64 = interior(domain.field()).iter().sum::<f64>() * cell_area;

    let mut solver = build_solver(device, &domain, side)?;
    // Device work submitted during setup must be drained before the clock stops, or its
    // cost lands in the compute phase and flatters it.
    device.finish()?;
    profile.record(phase::SETUP, setup_start.elapsed());

    let compute_start = Instant::now();
    solver
        .run(device, dt, STEPS)
        .map_err(|error| DeviceError::Backend(error.to_string()))?;
    let compute = compute_start.elapsed();
    profile.record(phase::COMPUTE, compute);

    let readback_start = Instant::now();
    let interior_values = solver
        .interior(device)
        .map_err(|error| DeviceError::Backend(error.to_string()))?;
    profile.record(phase::OBSERVE, readback_start.elapsed());

    let after: f64 = interior_values.iter().sum::<f64>() * cell_area;
    let drift = (after - before).abs() / before.abs();
    let non_finite = f64::from(u8::from(
        interior_values.iter().any(|value| !value.is_finite()),
    ));

    let mut memory = MemoryReport::new();
    // Two ping-pong fields, two face arrays and a source, all at the device's width.
    let width = device.precision().state_bytes();
    let cells = field.len();
    memory.record(
        "device buffers",
        (2 * cells + (side + 1) * side + side * (side + 1) + cells) * width,
    );

    Ok(BenchOutcome {
        throughput: Throughput {
            steps: STEPS as u64,
            simulated_seconds: dt * STEPS as f64,
            wall_clock: compute,
            elements: (side * side) as u64,
        },
        profile,
        memory,
        checks: vec![
            Check::new(
                "relative integral drift on a closed domain",
                drift,
                drift_limit(STEPS),
            ),
            Check::new("cells holding a non-finite value", non_finite, 0.0),
        ],
    })
}

/// Steps the implicit benchmark runs. Fewer than the explicit one's 400 because each is a
/// solve, and matching the CPU `heat-crank-nicolson` benchmark exactly so the two figures
/// are about the same work.
const IMPLICIT_STEPS: usize = 100;

/// How far past the explicit stability limit the implicit benchmark steps. Ten, matching the
/// CPU benchmark — and the whole reason to pay for a solve.
const OVER_LIMIT: f64 = 10.0;

/// Implicit diffusion, solved on the device.
///
/// # What is being measured, and what a reader must not conclude from it
///
/// The throughput figure is steps per second, and a step here is *not* a fixed amount of
/// work: it is a right-hand side assembly plus however many conjugate-gradient iterations
/// the problem needs. Comparing it against `heat-explicit`'s steps per second compares two
/// different quantities. The iteration count is published beside it for exactly that reason,
/// and so is the readback count — §10.3 requires the residual history to be able to stop a
/// run, and the only way to honour that on a device is to bring one scalar home per
/// iteration.
///
/// The residual tolerance is not the CPU's. It cannot be: `1e-10` is three orders of
/// magnitude below what `f32` can say about this operator, and asking for it would produce a
/// run that hits its iteration cap every step and reports as a slow GPU rather than as a
/// numerical impossibility. The floor is computed from the problem and the tolerance is set
/// to ten times it, which is published as a correctness condition rather than buried.
fn bench_heat_implicit(scale: usize, device: &GpuDevice) -> Result<BenchOutcome, DeviceError> {
    let side = 256 * scale;
    let mut profile = Profile::new();

    let domain = heat_domain_implicit(side);
    let dt = OVER_LIMIT * domain.operator().explicit_stability_limit().max;

    // Everything lazy has to happen before the clock starts, for the reason
    // `bench_heat_explicit` gives. An implicit solver compiles fifteen pipelines rather than
    // two, so this matters more here, not less.
    {
        let mut warm = build_implicit(device, &domain, side, dt)?;
        warm.step(device).map_err(backend)?;
        warm.interior(device).map_err(backend)?;
        device.finish()?;
    }

    let setup_start = Instant::now();
    let domain = heat_domain_implicit(side);
    let cell_area = domain.grid().cell_area();
    let before: f64 = interior(domain.field()).iter().sum::<f64>() * cell_area;

    let mut solver = build_implicit(device, &domain, side, dt)?;
    device.finish()?;
    profile.record(phase::SETUP, setup_start.elapsed());

    let compute_start = Instant::now();
    solver.run(device, IMPLICIT_STEPS).map_err(backend)?;
    let compute = compute_start.elapsed();
    profile.record(phase::COMPUTE, compute);

    let readback_start = Instant::now();
    let interior_values = solver.interior(device).map_err(backend)?;
    profile.record(phase::OBSERVE, readback_start.elapsed());

    let after: f64 = interior_values.iter().sum::<f64>() * cell_area;
    let drift = (after - before).abs() / before.abs();
    let non_finite = f64::from(u8::from(
        interior_values.iter().any(|value| !value.is_finite()),
    ));

    let mut memory = MemoryReport::new();
    let width = device.precision().state_bytes();
    let cells = domain.field().len();
    memory.record(
        "device buffers",
        // x, b, r, p, ap, plus the packed face and source coefficients.
        (5 * cells + (side + 1) * side + side * (side + 1) + cells) * width,
    );

    let iterations = solver.total_iterations();
    Ok(BenchOutcome {
        throughput: Throughput {
            steps: IMPLICIT_STEPS as u64,
            simulated_seconds: dt * IMPLICIT_STEPS as f64,
            wall_clock: compute,
            elements: (side * side) as u64,
        },
        profile,
        memory,
        checks: vec![
            // The integral drift bound is the explicit one plus what the solves are entitled
            // to: each stops at a relative residual, and ||A^-1|| <= 1 carries that straight
            // to the solution.
            Check::new(
                "relative integral drift on a closed domain",
                drift,
                drift_limit(IMPLICIT_STEPS) + IMPLICIT_STEPS as f64 * solver.tolerance(),
            ),
            Check::new("cells holding a non-finite value", non_finite, 0.0),
            Check::new(
                "steps whose linear solve did not converge",
                solver.non_converged_steps() as f64,
                0.0,
            ),
            // Not a correctness condition about the answer — one about the *question*. A run
            // asking for less than the floor is not measuring the solver.
            Check::new(
                "residual tolerance below the f32 floor, as a ratio",
                solver.tolerance_floor() / solver.tolerance(),
                1.0,
            ),
            // Published as a check with a generous limit rather than as a note, because it is
            // the quantity M4.5 would attack and a silent regression in it is invisible.
            Check::new(
                "conjugate-gradient iterations per step",
                iterations as f64 / IMPLICIT_STEPS as f64,
                64.0,
            ),
        ],
    })
}

fn backend(error: impl core::fmt::Display) -> DeviceError {
    DeviceError::Backend(error.to_string())
}

/// The same scene as [`heat_domain`], solved implicitly.
fn heat_domain_implicit(side: usize) -> HeatDomain {
    heat_domain(side).with_scheme(TimeScheme::CrankNicolson)
}

/// Build a device-resident implicit solver, at ten times the floor its precision allows.
fn build_implicit(
    device: &GpuDevice,
    domain: &HeatDomain,
    side: usize,
    dt: f64,
) -> Result<GpuCrankNicolson, DeviceError> {
    let operator = domain.operator();
    let field = domain.field();
    let mut setup = CrankNicolsonSetup {
        nx: side,
        ny: side,
        halo: field.halo(),
        stride: field.stride(),
        inv_dx2: operator.inv_dx2(),
        inv_dy2: operator.inv_dy2(),
        theta: TimeScheme::CrankNicolson.theta(),
        dt,
        field: field.as_slice(),
        face_x: operator.face_x(),
        face_y: operator.face_y(),
        source: None,
        tolerance: 0.0,
        max_iterations: 500,
    };
    setup.tolerance = 10.0 * GpuCrankNicolson::floor_for(device.precision(), &setup);
    GpuCrankNicolson::new(device, setup).map_err(backend)
}

/// The largest integral drift `f32` rounding can produce over `steps` steps.
///
/// Derived, not fitted. Each step writes `center + dt*(lap + s)` once per cell, and that
/// store rounds by at most half an ulp — a relative error of `ε/2` on a quantity of size
/// `T`. Summed over `N` cells the worst case is `N·T·ε/2`, and the integral itself is about
/// `N·T`, so one step can move it by `ε/2` relative. Over `n` steps, `n·ε/2`.
///
/// That is the worst case in which every cell's rounding pushes the same way every step;
/// real round-off has mixed signs and cancels, so the measured figure sits orders of
/// magnitude below. Two structural facts help further and are *not* claimed in the bound:
/// the flux between two cells is computed as `face·(a−b)` on one side and `face·(b−a)` on
/// the other, and IEEE negation is exact, so the pair cancels to the bit.
///
/// The CPU benchmark's limit for the same check is a flat `1e-9`, which is looser than this
/// formula gives at `f64` (`4.4e-14` for 400 steps). It is left alone: it is an established
/// baseline, and tightening it is a separate change with its own risk of false failures.
fn drift_limit(steps: usize) -> f64 {
    steps as f64 * Precision::Fast32.epsilon() / 2.0
}

/// Build a device-resident solver for `domain`, taking the face coefficients from the CPU
/// operator that computed them rather than deriving a second set.
fn build_solver(
    device: &GpuDevice,
    domain: &HeatDomain,
    side: usize,
) -> Result<GpuDiffusion, DeviceError> {
    let operator = domain.operator();
    let field = domain.field();
    GpuDiffusion::new(
        device,
        DiffusionSetup {
            nx: side,
            ny: side,
            halo: field.halo(),
            stride: field.stride(),
            inv_dx2: operator.inv_dx2(),
            inv_dy2: operator.inv_dy2(),
            field: field.as_slice(),
            face_x: operator.face_x(),
            face_y: operator.face_y(),
            source: None,
        },
    )
    .map_err(|error| DeviceError::Backend(error.to_string()))
}

fn interior(field: &lattice_ir::ScalarField) -> Vec<f64> {
    let mut out = Vec::with_capacity(field.nx() * field.ny());
    for j in 0..field.ny() {
        out.extend_from_slice(field.row(j));
    }
    out
}
