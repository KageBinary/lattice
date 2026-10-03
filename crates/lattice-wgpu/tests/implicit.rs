//! The implicit solver and the reduction under it, against properties that hold whatever
//! precision they run at.
//!
//! Every test here skips when there is no adapter, for the reason `diffusion.rs` gives.
//! §19.1's cross-backend level compares these against the CPU; this file is the properties
//! the GPU path must have on its own, and it is where a mis-sized dispatch or a swapped
//! scalar slot gets caught before a comparison is ever run.

use lattice_compute::{DeviceError, Precision};
use lattice_wgpu::{
    CgOutcome, CrankNicolsonSetup, GpuCrankNicolson, GpuDevice, GpuDot, ImplicitError, Interior,
};

const NX: usize = 96;
const NY: usize = 64;
const HALO: usize = 1;
const STRIDE: usize = NX + 2 * HALO;
const TOTAL: usize = STRIDE * (NY + 2 * HALO);
const DIFFUSIVITY: f64 = 1e-3;

fn device() -> Option<GpuDevice> {
    match GpuDevice::open_default() {
        Ok(device) => Some(device),
        Err(DeviceError::Unavailable { detail, .. }) => {
            eprintln!("SKIPPED: no portable GPU adapter available ({detail})");
            None
        }
        Err(error) => panic!("unexpected device error: {error}"),
    }
}

const fn interior() -> Interior {
    Interior {
        nx: NX,
        ny: NY,
        stride: STRIDE,
        origin: HALO * STRIDE + HALO,
    }
}

struct Problem {
    field: Vec<f64>,
    face_x: Vec<f64>,
    face_y: Vec<f64>,
    inv_dx2: f64,
    inv_dy2: f64,
    explicit_limit: f64,
    dt: f64,
    theta: f64,
    tolerance: f64,
}

impl Problem {
    /// A pulse on a uniform-diffusivity plate, stepped `over_limit` times further than an
    /// explicit scheme could manage. Being able to do that is the entire reason to solve a
    /// system instead of applying a stencil.
    fn new(initial: impl Fn(usize, usize) -> f64, over_limit: f64) -> Problem {
        let (dx, dy) = (1.0 / NX as f64, 0.75 / NY as f64);
        let (inv_dx2, inv_dy2) = (1.0 / (dx * dx), 1.0 / (dy * dy));

        let mut field = vec![0.0; TOTAL];
        for j in 0..NY {
            for i in 0..NX {
                field[(j + HALO) * STRIDE + i + HALO] = initial(i, j);
            }
        }

        let explicit_limit = 1.0 / (2.0 * DIFFUSIVITY * (inv_dx2 + inv_dy2));
        let mut problem = Problem {
            field,
            face_x: vec![DIFFUSIVITY; (NX + 1) * NY],
            face_y: vec![DIFFUSIVITY; NX * (NY + 1)],
            inv_dx2,
            inv_dy2,
            explicit_limit,
            dt: over_limit * explicit_limit,
            theta: 0.5,
            tolerance: 0.0,
        };
        // Ten times the floor: comfortably reachable, and still tight enough that the
        // solve is doing real work rather than stopping on its first residual.
        problem.tolerance = 10.0 * GpuCrankNicolson::floor_for(Precision::Fast32, &problem.setup());
        problem
    }

    fn setup(&self) -> CrankNicolsonSetup<'_> {
        CrankNicolsonSetup {
            nx: NX,
            ny: NY,
            halo: HALO,
            stride: STRIDE,
            inv_dx2: self.inv_dx2,
            inv_dy2: self.inv_dy2,
            theta: self.theta,
            dt: self.dt,
            field: &self.field,
            face_x: &self.face_x,
            face_y: &self.face_y,
            source: None,
            tolerance: self.tolerance,
            max_iterations: 500,
        }
    }
}

fn pulse(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / NX as f64, j as f64 / NY as f64);
    let r2 = (x - 0.35).powi(2) + (y - 0.6).powi(2);
    300.0 + 60.0 * (-r2 / 0.01).exp()
}

// ---------------------------------------------------------------------------
// The refusal
// ---------------------------------------------------------------------------

/// **The finding of the milestone.** `HeatDomain` defaults to a relative residual of
/// `1e-10`, which is an ordinary ask of an `f64` solver and is three orders of magnitude
/// below anything an `f32` one can say about this problem.
///
/// The failure mode being prevented is not a wrong answer, it is a *misread* one: a solver
/// accepting `1e-10` iterates to its cap every step, reports `MaxIterations`, and looks
/// exactly like a slow GPU. Refusing at setup with the floor and the `‖A‖` it came from is
/// the difference between a numerical fact and a performance mystery.
#[test]
fn the_cpu_default_tolerance_is_refused_with_the_floor_it_missed() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);

    let mut setup = problem.setup();
    setup.tolerance = 1e-10;
    let error = GpuCrankNicolson::new(&device, setup).unwrap_err();

    let ImplicitError::ToleranceBelowFloor {
        floor,
        operator_norm,
        precision,
        ..
    } = error
    else {
        panic!("expected a floor refusal, got {error}");
    };
    assert_eq!(precision, Precision::Fast32);
    assert!(
        floor > 1e-10,
        "the floor must actually exceed what was asked for"
    );
    assert!(
        operator_norm > 1.0,
        "theta*dt*L must contribute something to ||A||"
    );

    let message = ImplicitError::ToleranceBelowFloor {
        requested: 1e-10,
        floor,
        operator_norm,
        precision,
    }
    .to_string();
    eprintln!("{message}");
    assert!(message.contains("fast32"), "{message}");
    assert!(message.contains("||A||"), "{message}");
}

/// The floor has to be a function of the problem, not a constant wearing a derivation. A
/// larger step makes `A` further from the identity and the residual harder to form.
#[test]
fn the_floor_rises_with_the_timestep() {
    let small = Problem::new(pulse, 1.0);
    let large = Problem::new(pulse, 100.0);
    let floor = |p: &Problem| GpuCrankNicolson::floor_for(Precision::Fast32, &p.setup());

    let (low, high) = (floor(&small), floor(&large));
    eprintln!("floor at 1x the explicit limit: {low:.3e}, at 100x: {high:.3e}");
    assert!(high > low, "{high:.3e} should exceed {low:.3e}");
    assert!(
        low >= Precision::Fast32.epsilon(),
        "the floor is never below one rounding"
    );
}

/// `theta = 0` is the explicit scheme, which needs no solve and has its own solver. A
/// silent fallthrough would divide by zero in `A = I - 0*L` — or worse, not.
#[test]
fn an_explicit_or_impossible_theta_is_refused() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);

    for theta in [0.0, -0.5, 1.5, f64::NAN] {
        let mut setup = problem.setup();
        setup.theta = theta;
        let error = GpuCrankNicolson::new(&device, setup).unwrap_err();
        assert!(
            matches!(error, ImplicitError::Implicitness(_)),
            "theta = {theta} gave {error}"
        );
    }
}

// ---------------------------------------------------------------------------
// The solver
// ---------------------------------------------------------------------------

/// A field with no gradient is the one implicit result that can be asserted at zero
/// tolerance, and it is stronger than the explicit version of the same test.
///
/// `L(u) = 0`, so `b = u` and `A·u = u`, so the initial residual is *exactly* zero and CG
/// returns before its first iteration. Anything that made the halo wrong, the rhs assembly
/// wrong, or the operator wrong would produce a non-zero residual here and iterate.
#[test]
fn a_uniform_field_is_solved_exactly_and_without_iterating() {
    let Some(device) = device() else { return };
    let problem = Problem::new(|_, _| 300.0, 20.0);
    let mut solver = GpuCrankNicolson::new(&device, problem.setup()).unwrap();

    let outcome = solver.run(&device, 20).unwrap();
    assert_eq!(
        outcome,
        CgOutcome::Converged {
            iterations: 0,
            residual: 0.0
        },
        "{outcome}"
    );
    assert_eq!(
        solver.total_iterations(),
        0,
        "a gradient-free field needs no iteration"
    );

    for (index, value) in solver.interior(&device).unwrap().iter().enumerate() {
        assert_eq!(*value, 300.0, "cell {index} drifted from a uniform field");
    }
}

#[test]
fn repeated_solver_setup_reuses_pipelines_and_keeps_state_independent() {
    let Some(device) = device() else { return };
    let cold = Problem::new(|_, _| 300.0, 2.0);
    let hot = Problem::new(|_, _| 400.0, 2.0);
    let mut a = GpuCrankNicolson::new(&device, cold.setup()).unwrap();
    let families = device.pipeline_cache_len();
    let mut b = GpuCrankNicolson::new(&device, hot.setup()).unwrap();
    assert_eq!(device.pipeline_cache_len(), families);
    assert_eq!(families, 2, "implicit and dot-product pipeline families");
    a.run(&device, 2).unwrap();
    b.run(&device, 2).unwrap();
    assert!(a.interior(&device).unwrap().iter().all(|&v| v == 300.0));
    assert!(b.interior(&device).unwrap().iter().all(|&v| v == 400.0));
}

/// The property the whole scheme exists for. At twenty times the explicit stability limit
/// an explicit stencil is `NaN` within a few dozen steps; Crank–Nicolson is unconditionally
/// stable and simply solves a harder system.
#[test]
fn crank_nicolson_is_stable_far_above_the_explicit_limit() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);
    assert!(problem.dt > 20.0 * problem.explicit_limit * 0.99);

    let mut solver = GpuCrankNicolson::new(&device, problem.setup()).unwrap();
    let start = solver.interior(&device).unwrap();
    let outcome = solver.run(&device, 60).unwrap();
    let end = solver.interior(&device).unwrap();

    assert!(outcome.is_converged(), "{outcome}");
    assert_eq!(
        solver.non_converged_steps(),
        0,
        "every step must have converged"
    );
    assert!(end.iter().all(|v| v.is_finite()), "the run went non-finite");

    let (low, high) = extremes(&start);
    let (new_low, new_high) = extremes(&end);
    eprintln!(
        "60 steps at {:.1}x the explicit limit: peak {high:.2} -> {new_high:.2}, \
         trough {low:.2} -> {new_low:.2}, {} CG iterations",
        problem.dt / problem.explicit_limit,
        solver.total_iterations()
    );
    assert!(new_high < high, "the peak must fall: {new_high} vs {high}");
    assert!(
        new_low >= low - 1e-2,
        "no meaningful undershoot below the initial minimum"
    );
}

/// Insulated boundaries move heat and never lose any.
///
/// The tolerance is looser than the explicit run's `1e-5` and that is the honest number,
/// not a concession: this run stops each step at a *relative residual*, so each step is
/// permitted an error the explicit scheme has no analogue for. It is asserted against a
/// bound derived from that permission — `steps · tolerance` — rather than against a round
/// number.
#[test]
fn insulated_implicit_diffusion_conserves_heat() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);
    let mut solver = GpuCrankNicolson::new(&device, problem.setup()).unwrap();

    let steps = 60;
    let before: f64 = solver.interior(&device).unwrap().iter().sum();
    solver.run(&device, steps).unwrap();
    let after: f64 = solver.interior(&device).unwrap().iter().sum();

    let drift = (after - before).abs() / before.abs();
    let permitted = steps as f64 * problem.tolerance;
    eprintln!(
        "heat drift over {steps} implicit steps: {drift:.3e}, against {permitted:.3e} \
         permitted by {steps} solves at a relative residual of {:.3e}",
        problem.tolerance
    );
    assert!(drift < permitted, "insulated boundaries leaked {drift:.3e}");
}

/// NFR-007: no confident wrong answers. A field that is already non-finite must not come
/// back as a converged solve.
///
/// This is the end-to-end half of the claim, and it is the weaker half — the curvature test
/// catches this particular path before the residual test is reached, so it would still pass
/// against the `rs.max(0.0)` clamp that `residual_of` was written to remove. The unit test
/// `a_non_finite_residual_is_never_laundered_into_convergence` is what actually holds that
/// line; a NaN reaching the residual check while `‖b‖` stays finite needs an overflow
/// mid-iteration, which is real and is not something a test can arrange from the outside.
#[test]
fn a_non_finite_field_is_never_reported_as_converged() {
    let Some(device) = device() else { return };
    let mut problem = Problem::new(pulse, 20.0);
    problem.field[(NY / 2 + HALO) * STRIDE + NX / 2 + HALO] = f64::NAN;

    let mut solver = GpuCrankNicolson::new(&device, problem.setup()).unwrap();
    let outcome = solver.step(&device).unwrap();

    eprintln!("a NaN in the initial field gives: {outcome}");
    assert!(
        !outcome.is_converged(),
        "a NaN field reported as converged: {outcome}"
    );
    assert!(
        solver.non_converged_steps() > 0,
        "the step must be counted as not converged"
    );
}

/// §10.3: *"residual histories are always observable"*. Observable means a caller can read
/// them, and useful means they fall.
#[test]
fn the_residual_history_is_observable_and_falls() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);
    let mut solver = GpuCrankNicolson::new(&device, problem.setup()).unwrap();

    let outcome = solver.step(&device).unwrap();
    let history = solver.residual_history();

    assert!(outcome.is_converged(), "{outcome}");
    assert!(
        history.len() > 2,
        "a real solve records more than a first and last residual"
    );
    assert_eq!(
        history.len(),
        outcome.iterations() + 1,
        "one residual per iteration, plus the initial one"
    );
    assert!(history.iter().all(|value| value.is_finite()));
    assert!(
        history.last().unwrap() < history.first().unwrap(),
        "the residual must fall: {:?}",
        &history[..history.len().min(5)]
    );
}

/// The cost of §10.3's stopping rule, stated as a number rather than as a worry.
///
/// One transfer per step to learn `‖b‖` and the opening residual, and one per iteration to
/// learn `‖r‖²` and `pᵀAp` together. Not two per iteration: `iterating()` in the shader is
/// what buys that, by letting the curvature test happen on the device.
#[test]
fn exactly_one_readback_per_iteration_and_one_per_step() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);
    let mut solver = GpuCrankNicolson::new(&device, problem.setup()).unwrap();

    solver.run(&device, 12).unwrap();
    let expected = solver.steps() + solver.total_iterations();
    eprintln!(
        "{} steps, {} CG iterations, {} readbacks",
        solver.steps(),
        solver.total_iterations(),
        solver.readbacks()
    );
    assert_eq!(
        solver.readbacks(),
        expected,
        "the solve is stalling more often than it must"
    );
}

/// §15.5's specialization, checked where it would be tempting to skip it. Two solvers that
/// differ only in `theta` are different schemes and must not share a compiled module — but
/// they *do* share a reduction, because a dot product does not know what it is summing.
#[test]
fn the_scheme_specializes_the_solver_but_not_the_reduction() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse, 20.0);

    let crank = GpuCrankNicolson::new(&device, problem.setup()).unwrap();
    let mut backward = problem.setup();
    backward.theta = 1.0;
    let euler = GpuCrankNicolson::new(&device, backward).unwrap();

    assert_ne!(
        crank.kernel_key(),
        euler.kernel_key(),
        "theta must be part of the key"
    );
    assert_eq!(
        crank.reduction_kernel_key(),
        euler.reduction_kernel_key(),
        "the reduction depends on the grid, not on the scheme"
    );
}

// ---------------------------------------------------------------------------
// The reduction, on its own
// ---------------------------------------------------------------------------

/// A field of positive values, whose halo is deliberately poisonous.
fn reduction_field(scale: f64) -> Vec<f64> {
    let mut data = vec![-1.0e6; TOTAL];
    for j in 0..NY {
        for i in 0..NX {
            let (x, y) = (i as f64 / NX as f64, j as f64 / NY as f64);
            data[(j + HALO) * STRIDE + i + HALO] = scale * (1.0 + x + 2.0 * y * y);
        }
    }
    data
}

/// The device's association order against an essentially exact sum of the same `f32`
/// values, held to the depth the order implies and nothing looser.
///
/// Every term is positive, so `Σ|xᵢ| = |Σxᵢ|` and the bound is exactly `depth · ε` with no
/// cancellation factor to argue about. A sequential `f32` sum of the same 6 144 values
/// would be entitled to `6143 · ε`; this one is entitled to 16.
#[test]
fn the_device_reduction_lands_inside_the_depth_it_declares() {
    let Some(device) = device() else { return };
    let field = reduction_field(0.01);
    let interior = interior();

    let measured = GpuDot::over_host(&device, interior, &field, &field).unwrap();

    // The reference: the same values the device holds — narrowed to f32 first, so this is
    // a statement about summation order and not about storage — added in f64, which is
    // exact enough at this magnitude to be the exact sum.
    let mut exact = 0.0f64;
    for j in 0..NY {
        for i in 0..NX {
            let value = f64::from(field[(j + HALO) * STRIDE + i + HALO] as f32);
            exact += value * value;
        }
    }

    let relative = (measured - exact).abs() / exact.abs();
    let permitted = interior.depth() as f64 * Precision::Fast32.epsilon();
    eprintln!(
        "device dot {measured:.9e} vs exact {exact:.9e}: {relative:.3e} relative, \
         against {permitted:.3e} permitted by a depth of {} at f32 epsilon",
        interior.depth()
    );
    assert!(
        relative <= permitted,
        "{relative:.3e} exceeds the declared depth's bound"
    );
}

/// The property that makes a budget possible at all. A reduction that returned a slightly
/// different sum each run would need a tolerance nobody could derive, and would make every
/// CG iteration count non-reproducible on one machine.
#[test]
fn the_device_reduction_is_bit_reproducible() {
    let Some(device) = device() else { return };
    let field = reduction_field(1.0);
    let interior = interior();

    let first = GpuDot::over_host(&device, interior, &field, &field).unwrap();
    for attempt in 1..5 {
        let again = GpuDot::over_host(&device, interior, &field, &field).unwrap();
        assert_eq!(
            again.to_bits(),
            first.to_bits(),
            "run {attempt} gave {again:.17e}, run 0 gave {first:.17e}"
        );
    }
}

/// Halo values are boundary bookkeeping, not degrees of freedom. `interior_dot` on the CPU
/// skips them; so must this, and the halo here is large and negative so that including any
/// of it would be unmissable.
#[test]
fn the_device_reduction_never_reads_the_halo() {
    let Some(device) = device() else { return };
    let interior = interior();
    let quiet = reduction_field(1.0);
    let mut loud = quiet.clone();
    for (index, slot) in loud.iter_mut().enumerate() {
        let (row, column) = (index / STRIDE, index % STRIDE);
        let inside = (HALO..HALO + NY).contains(&row) && (HALO..HALO + NX).contains(&column);
        if !inside {
            *slot = 1.0e9;
        }
    }

    let with_quiet_halo = GpuDot::over_host(&device, interior, &quiet, &quiet).unwrap();
    let with_loud_halo = GpuDot::over_host(&device, interior, &loud, &loud).unwrap();
    assert_eq!(
        with_quiet_halo.to_bits(),
        with_loud_halo.to_bits(),
        "a halo value reached the sum"
    );
}

/// The depth is only meaningful if the split it describes is the split that runs. At a size
/// where every invocation takes one element the depth is the two trees alone.
#[test]
fn the_declared_depth_matches_the_grid_it_was_built_for() {
    let interior = interior();
    assert!(
        interior.cells() < 256 * 256,
        "this grid should fit one element per invocation"
    );
    assert_eq!(interior.depth(), 16, "two 256-wide trees and no serial run");
}

fn extremes(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}
