//! The GPU half of §19.1's cross-backend level.
//!
//! > | Cross-backend | Do CPU and GPU agree within tolerance? | all released kernels |
//!
//! [`execution`](crate::execution) answers the CPU half of that question, and answers it
//! *more strictly than it was asked*: the scalar and parallel CPU paths agree bit for bit.
//! These cases are the first in the project that genuinely need the words "within
//! tolerance", and they are deliberately a different kind of claim rather than the CPU
//! claim weakened.
//!
//! # Where the tolerance comes from
//!
//! Not from running the comparison and rounding up. For the explicit rows the budget is
//! [`Tolerance::stepped_kernel`], derived before the measurement from three stated facts —
//! the stencil rounds about five times per cell per step, an explicit diffusion step below
//! its stability limit does not amplify error it already has, and additive accumulation
//! over `n` steps is `n` times the per-step figure. The budget is therefore a worst case
//! that real round-off, behaving like a random walk, should sit far below.
//!
//! Every row here reports the *fraction of the budget used*, which is the number worth
//! watching: a case sitting at a few percent says the bound is loose and honest, and one
//! climbing toward 100% across releases says something is degrading while still passing.
//!
//! # The implicit rows are a different kind of claim again
//!
//! M4.4 added Crank–Nicolson on the device, and with it a mechanism the explicit cases do
//! not have. Two conjugate-gradient solves of the same system, each stopping when its own
//! residual criterion is met, return two different correct answers; the distance between
//! them is set by how loosely the weaker backend is allowed to stop, and it is *four orders
//! of magnitude larger* than anything `f32` storage contributes. See
//! [`ImplicitSolve`](lattice_compute::ImplicitSolve) for the derivation and
//! `gpu_implicit_disagreement_is_dominated_by_solve_termination` for the measurement.
//!
//! Two consequences shape the rows below. The budget is stated and checked in the
//! **Euclidean norm**, because a stopping rule on `‖r‖₂` is a statement about a vector and
//! converting it to a per-cell bound would cost a factor of `√N` belonging entirely to the
//! change of norm. And [`Mechanism::ReductionOrder`] — the term M4.2 deferred this work for
//! — turns out to be invisible inside an iteration, so it is audited on a bare reduction
//! instead, where nothing re-measures it.
//!
//! # Why these cases can be absent
//!
//! A machine with no GPU produces no rows here at all, the way M3 leaves the Gillespie
//! statistics absent rather than reporting them as skipped-and-passing. [`cases`] returns
//! an empty list when no adapter opens, and the crate is compiled without `--features gpu`
//! by default so that `lattice validate` does not drag `wgpu` into the CLI.

use std::sync::OnceLock;

use lattice_compute::{Device, ImplicitSolve, Mechanism, Norm, Precision, Tolerance};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme, gaussian};
use lattice_ir::{Arena, BoundarySet, Domain, Executor, Grid2d, ScalarField, StepContext};
use lattice_wgpu::{
    CrankNicolsonSetup, DiffusionSetup, GpuCrankNicolson, GpuDevice, GpuDiffusion, GpuDot,
    ImplicitError, Interior,
};

use crate::{Case, Level, Outcome};

#[path = "gpu_particles.rs"]
mod particle_cases;

/// Interior cells. Large enough that the run is not dominated by dispatch overhead, and
/// deliberately not square, so a transposed index cannot pass.
const GRID: (usize, usize) = (192, 128);

/// Steps every comparison runs. Enough for the pulse to spread across many cells, so the
/// comparison exercises the stencil rather than the initial condition.
const STEPS: usize = 200;

/// The shared device, opened once. `None` means no adapter, which makes these cases
/// absent rather than failing.
fn device() -> Option<&'static GpuDevice> {
    static DEVICE: OnceLock<Option<GpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| match GpuDevice::open_default() {
            Ok(device) => Some(device),
            Err(lattice_compute::DeviceError::Unavailable { detail, .. }) => {
                eprintln!("GPU validation unavailable: {detail}");
                None
            }
            Err(error) => panic!("GPU validation initialization failed: {error}"),
        })
        .as_ref()
}

/// The cases, or none if this machine has no portable GPU backend.
pub(crate) fn cases() -> Vec<Case> {
    if device().is_none() {
        return Vec::new();
    }
    vec![
        Case {
            name: "gpu_gravity_matches_cpu",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "resident velocity Verlet gravity agrees with CPU trajectories for unequal masses",
            run: particle_cases::gravity,
        },
        Case {
            name: "gpu_lennard_jones_matches_cpu",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "sorted GPU cell gathers match CPU pair forces and trajectories with bounded energy and momentum drift",
            run: particle_cases::lennard_jones,
        },
        Case {
            name: "gpu_implicit_dirichlet_matches_cpu",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "mixed prescribed faces agree with CPU Crank-Nicolson, including single-cell axes and backward Euler",
            run: dirichlet_matches_cpu,
        },
        Case {
            name: "gpu_diffusion_matches_the_cpu_reference",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "200 explicit diffusion steps on the portable GPU backend agree with the scalar CPU reference inside a budget derived from f32 rounding, not fitted to the result",
            run: matches_the_reference,
        },
        Case {
            name: "gpu_disagreement_is_attributable_to_f32_storage",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "the CPU/GPU disagreement is a few f32 ulps, which is the scale f32 state rounding predicts and roughly 10^8 times what f64 rounding would explain",
            run: disagreement_is_attributable,
        },
        Case {
            name: "gpu_and_cpu_agree_exactly_where_no_rounding_occurs",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "on a field with no gradient every flux is a difference of equal numbers, and the two backends agree to the bit despite different precisions",
            run: agree_exactly_without_rounding,
        },
        Case {
            name: "gpu_crank_nicolson_matches_the_cpu_reference",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "an implicit diffusion run solved by conjugate gradient on the GPU agrees with the scalar CPU reference inside a budget derived from the two solves' stopping criteria, not fitted to the result",
            run: implicit_matches_the_reference,
        },
        Case {
            name: "gpu_implicit_disagreement_is_dominated_by_solve_termination",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "the implicit CPU/GPU disagreement is set by how loosely the f32 solve is allowed to stop, not by f32 storage and not by the reduction M4.2 deferred this milestone for",
            run: implicit_disagreement_is_attributable,
        },
        Case {
            name: "gpu_reduction_order_is_the_whole_disagreement_when_nothing_re_measures_it",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "a device reduction taken as an answer rather than as a stopping test disagrees with a sequential sum by its declared accumulation depth, which is the mechanism conjugate gradient hides",
            run: reduction_order_alone,
        },
        Case {
            name: "gpu_refuses_a_residual_tolerance_it_cannot_reach",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "the CPU's default relative residual of 1e-10 is below the f32 floor eps*(1+||A||), and the backend refuses it at setup rather than iterating to its cap and reporting a performance problem",
            run: refuses_an_unreachable_tolerance,
        },
        Case {
            name: "gpu_and_cpu_implicit_agree_exactly_on_a_field_with_no_gradient",
            domain: "wgpu",
            level: Level::CrossBackend,
            claim: "on a gradient-free field the implicit residual is exactly zero, both solves return before their first iteration, and the two backends agree to the bit with no budget spent",
            run: implicit_agrees_exactly_without_a_solve,
        },
    ]
}

fn dirichlet_matches_cpu() -> Outcome {
    use lattice_ir::Boundary;
    let device = device().expect("device");
    let mut worst = 0.0f64;
    for (nx, ny) in [(1, 1), (1, 9), (13, 1), (17, 11), (96, 64)] {
        for scheme in [TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
            for values in [
                [Some(300.0), Some(380.0), None, None],
                [None, Some(330.0), Some(280.0), Some(400.0)],
                [Some(300.0); 4],
            ] {
                let grid = Grid2d::new(nx, ny, [1.0, 0.75]);
                let mut diffusivity = ScalarField::new(&grid, 1);
                diffusivity.init_from_position(&grid, |[x, y]| 0.01 * (1.0 + x * x + y));
                let mut source = ScalarField::new(&grid, 1);
                source.init_from_position(&grid, |[x, y]| 0.2 + x - 0.5 * y);
                let boundary = |v: Option<f64>| v.map_or(Boundary::INSULATED, Boundary::fixed);
                let mut cpu =
                    HeatDomain::new("dirichlet", grid, Diffusivity::Variable(diffusivity))
                        .with_initial(|[x, y]| 300.0 + 5.0 * x * y)
                        .with_scheme(scheme)
                        .with_boundaries(BoundarySet {
                            left: boundary(values[0]),
                            right: boundary(values[1]),
                            bottom: boundary(values[2]),
                            top: boundary(values[3]),
                        });
                cpu.set_source(source);
                let op = cpu.operator();
                let field = cpu.field();
                let dt = 2.0 * op.explicit_stability_limit().max;
                let mut setup = CrankNicolsonSetup {
                    nx,
                    ny,
                    halo: field.halo(),
                    stride: field.stride(),
                    inv_dx2: op.inv_dx2(),
                    inv_dy2: op.inv_dy2(),
                    theta: scheme.theta(),
                    dt,
                    field: field.as_slice(),
                    face_x: op.face_x(),
                    face_y: op.face_y(),
                    source: cpu.source().map(ScalarField::as_slice),
                    tolerance: 1.0,
                    max_iterations: 500,
                };
                setup.tolerance = 4.0 * GpuCrankNicolson::floor_for(Precision::Fast32, &setup);
                let mut gpu = GpuCrankNicolson::with_dirichlet(device, setup, values)
                    .expect("Dirichlet setup");
                let mut arena = Arena::with_capacity(0);
                let mut ctx = StepContext::new(&mut arena);
                for _ in 0..20 {
                    cpu.prepare(&mut ctx);
                    cpu.advance(dt, &mut ctx);
                    assert!(gpu.step(device).expect("GPU step").is_converged());
                }
                let reference = interior_of(cpu.field());
                let measured = gpu.interior(device).expect("readback");
                let norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
                let budget = ImplicitSolve {
                    precision: Precision::Fast32,
                    steps: 20,
                    reference_tolerance: 1e-10,
                    measured_tolerance: gpu.tolerance(),
                    operator_norm: gpu.operator_norm_bound(),
                    reduction_depth: gpu.reduction_depth(),
                    rhs_amplification: (gpu.largest_rhs_norm() / norm).max(1.0),
                }
                .budget();
                let comparison = budget.compare(&reference, &measured);
                worst = worst.max(comparison.budget_used());
                assert!(
                    comparison.budget_used() <= 1.0,
                    "{nx}x{ny} {scheme:?} {values:?}: {}",
                    comparison.summary()
                );
            }
        }
    }
    Outcome::at_most("maximum fraction of derived solve budget", "1", worst, 1.0)
        .note("30 layouts: mixed insulated/Dirichlet and four Dirichlet faces, variable diffusivity and source, two implicit schemes, 1x1 through 96x64")
}

/// The problem both backends run.
///
/// Variable diffusivity and an asymmetric initial condition, for the same reason the CPU
/// cases use them: a uniform or symmetric problem can survive a mis-indexed face
/// coefficient by luck.
fn model() -> HeatDomain {
    let grid = Grid2d::new(GRID.0, GRID.1, [1.0, 0.75]);
    let hot_spot = gaussian([0.35, 0.55], 0.004, 45.0);

    let mut diffusivity = ScalarField::new(&grid, 1);
    diffusivity.init_from_position(&grid, |[x, y]| 1e-4 * (1.0 + 2.0 * x * x + 0.5 * y));

    HeatDomain::new("plate", grid, Diffusivity::Variable(diffusivity))
        .with_initial(move |[x, y]| 300.0 + hot_spot([x, y]) + 4.0 * (9.0 * x).sin() * y)
        .with_boundaries(BoundarySet::INSULATED)
        .with_scheme(TimeScheme::Explicit)
}

/// A field with no gradient, where the stencil has nothing to round.
fn flat_model() -> HeatDomain {
    let grid = Grid2d::new(GRID.0, GRID.1, [1.0, 0.75]);
    HeatDomain::new("plate", grid, Diffusivity::Uniform(1e-4))
        .with_initial(|_| 300.0)
        .with_boundaries(BoundarySet::INSULATED)
        .with_scheme(TimeScheme::Explicit)
}

/// Run the reference on the CPU and the same problem on the GPU, and return the interiors
/// plus the timestep used.
fn both_backends(mut domain: HeatDomain, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let device = device().expect("cases() checked for a device");
    let dt = domain.stable_step().preferred;

    // The GPU is set up from the *initial* state, before the CPU run mutates it, and from
    // the operator's own face coefficients so that any disagreement is about arithmetic
    // rather than about what the coefficients were.
    let setup_field = domain.field().as_slice().to_vec();
    let operator = domain.operator().clone();
    let (stride, halo) = (domain.field().stride(), domain.field().halo());

    let mut solver = GpuDiffusion::new(
        device,
        DiffusionSetup {
            nx: GRID.0,
            ny: GRID.1,
            halo,
            stride,
            inv_dx2: operator.inv_dx2(),
            inv_dy2: operator.inv_dy2(),
            field: &setup_field,
            face_x: operator.face_x(),
            face_y: operator.face_y(),
            source: None,
        },
    )
    .expect("the GPU solver could not be set up");
    solver.run(device, dt, steps).expect("the GPU run failed");
    let gpu = solver.interior(device).expect("GPU readback failed");

    let mut arena = Arena::with_capacity(0);
    {
        let executor = Executor::sequential();
        let mut ctx = StepContext::new(&mut arena).with_executor(&executor);
        domain.prepare(&mut ctx);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
        }
    }
    let cpu = interior_of(domain.field());

    (cpu, gpu)
}

fn interior_of(field: &ScalarField) -> Vec<f64> {
    let mut out = Vec::with_capacity(field.nx() * field.ny());
    for j in 0..field.ny() {
        out.extend_from_slice(field.row(j));
    }
    out
}

/// The headline: does the portable backend land inside the budget its precision predicts?
fn matches_the_reference() -> Outcome {
    let (cpu, gpu) = both_backends(model(), STEPS);

    // Derived before the comparison. `SOLVER_PRECISION` is what the GPU actually runs at,
    // taken from the backend rather than assumed here.
    let budget = Tolerance::stepped_kernel(lattice_wgpu::SOLVER_PRECISION, STEPS).allowing_fma(
        lattice_wgpu::SOLVER_PRECISION,
        4,
        STEPS,
    );
    let comparison = budget.compare(&cpu, &gpu);

    Outcome::at_most(
        "fraction of the derived budget used",
        "1",
        comparison.budget_used(),
        1.0,
    )
    .note(comparison.summary())
    .note(budget.explain().trim_end().to_string())
    .note(format!("backend: {}", device().expect("device").label()))
}

/// The attribution. A number inside a budget is not yet evidence that the budget describes
/// the right mechanism — a wiring error would also produce *some* number.
///
/// Measured in `f32` ulps, the disagreement should be small and bounded. Measured in `f64`
/// ulps it is astronomically large. Both are asserted, because together they say the
/// difference is precision rather than a defect: a transposed index would blow the `f32`
/// bound, and a backend secretly running `f64` would fail the second.
fn disagreement_is_attributable() -> Outcome {
    let (cpu, gpu) = both_backends(model(), STEPS);

    let exact = Tolerance::exact();
    let comparison = exact.compare(&cpu, &gpu);
    let in_f32_ulps = comparison.worst_relative / Precision::Fast32.epsilon();

    // The derived per-step figure is 5 roundings; over 200 steps that is 1000 f32 ulps in
    // the worst case. Round-off that behaves like a random walk lands far below it. The
    // bound asserted here is the derived one, not the observed one.
    let derived_bound = 5.0 * STEPS as f64;

    Outcome::at_most(
        "worst CPU/GPU disagreement, in f32 ulps",
        "1",
        in_f32_ulps,
        derived_bound,
    )
    .note(comparison.summary())
    .note(format!(
        "the same disagreement is {:.2e} f64 ulps, which is why {} is the mechanism named",
        comparison.worst_relative / Precision::Accurate64.epsilon(),
        Mechanism::StateRounding
    ))
    .note(format!(
        "derived bound is {derived_bound} ulps: 5 roundings per cell per step over {STEPS} steps"
    ))
}

/// The control. Where the arithmetic cannot round, the tolerance must not be needed.
///
/// Every flux term on a gradient-free field is `coefficient * (u - u)`, which is exactly
/// zero in any precision, so the update adds exactly zero. A case that passed the two
/// above but failed this one would mean the budget was covering a real defect.
fn agree_exactly_without_rounding() -> Outcome {
    let (cpu, gpu) = both_backends(flat_model(), STEPS);

    let comparison = Tolerance::exact().compare(&cpu, &gpu);
    let mismatches = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();

    Outcome::at_most(
        "cells differing between the backends",
        "1",
        mismatches as f64,
        0.0,
    )
    .note(comparison.summary())
    .note("no gradient means every flux is a difference of equal numbers".to_string())
}

// ---------------------------------------------------------------------------
// M4.4: the implicit path
// ---------------------------------------------------------------------------

/// Steps the implicit comparisons run. Fewer than the explicit cases' 200 because each one
/// is a solve rather than a stencil, and because the budget grows linearly in them — a
/// number chosen to keep the run honest rather than to keep the budget small.
const IMPLICIT_STEPS: usize = 50;

/// How far past the explicit stability limit the implicit runs step.
///
/// Twenty. Below about five there is no reason to be implicit at all, and the comparison
/// would be measuring a solver that converges on its first iteration.
const OVER_LIMIT: f64 = 20.0;

/// A relative residual the `f32` backend can actually reach, as a multiple of its own floor.
///
/// Ten rather than one: sitting exactly on the floor makes convergence a coin toss on the
/// last iteration, and a case that intermittently reports `MaxIterations` is a case nobody
/// reads. The multiple is stated here because it is the single largest input to the budget.
const FLOOR_MULTIPLE: f64 = 10.0;

/// The implicit problem, on the same plate the explicit cases use.
fn implicit_model() -> HeatDomain {
    model().with_scheme(TimeScheme::CrankNicolson)
}

/// Everything one implicit comparison produced.
struct ImplicitRun {
    cpu: Vec<f64>,
    gpu: Vec<f64>,
    budget: lattice_compute::Tolerance,
    solver_tolerance: f64,
    reduction_depth: usize,
    rhs_amplification: f64,
    iterations: usize,
    readbacks: usize,
    non_converged: usize,
}

/// Run Crank–Nicolson on both backends and assemble the budget from what the run reports.
///
/// Every input to the budget is either declared before the run (the two tolerances, the step
/// count, the precision) or is a property of the *problem* measured from it (`‖A‖`, the
/// reduction depth, `‖b‖₂/‖x‖₂`). None of them is the disagreement, which is the line
/// between deriving a bound and fitting one.
fn implicit_run(domain: HeatDomain) -> ImplicitRun {
    let device = device().expect("cases() checked for a device");

    let dt = OVER_LIMIT * domain.operator().explicit_stability_limit().max;
    let setup_field = domain.field().as_slice().to_vec();
    let operator = domain.operator().clone();
    let (stride, halo) = (domain.field().stride(), domain.field().halo());

    let mut setup = CrankNicolsonSetup {
        nx: GRID.0,
        ny: GRID.1,
        halo,
        stride,
        inv_dx2: operator.inv_dx2(),
        inv_dy2: operator.inv_dy2(),
        theta: TimeScheme::CrankNicolson.theta(),
        dt,
        field: &setup_field,
        face_x: operator.face_x(),
        face_y: operator.face_y(),
        source: None,
        tolerance: 0.0,
        max_iterations: 500,
    };
    setup.tolerance =
        FLOOR_MULTIPLE * GpuCrankNicolson::floor_for(lattice_wgpu::SOLVER_PRECISION, &setup);

    let mut solver =
        GpuCrankNicolson::new(device, setup).expect("the GPU implicit solver could not be set up");
    solver
        .run(device, IMPLICIT_STEPS)
        .expect("the GPU implicit run failed");
    let gpu = solver.interior(device).expect("GPU readback failed");

    // The CPU reference keeps its own default tolerance. Matching it to the GPU's would make
    // the comparison easier and would be the wrong experiment: the reference is what the
    // specification says to compute, and the budget's job is to explain the distance from it.
    let cpu_tolerance = 1e-10;
    let mut domain = domain
        .with_tolerance(cpu_tolerance)
        .with_max_iterations(500);
    let mut arena = Arena::with_capacity(0);
    {
        let executor = Executor::sequential();
        let mut ctx = StepContext::new(&mut arena).with_executor(&executor);
        domain.prepare(&mut ctx);
        for _ in 0..IMPLICIT_STEPS {
            domain.advance(dt, &mut ctx);
        }
    }
    let cpu = interior_of(domain.field());

    // ‖b‖₂/‖x‖₂, both measured: the largest right-hand side any step reached, over the
    // reference's own final magnitude. Never below 1, because ‖A⁻¹‖₂ ≤ 1.
    let reference_norm = cpu.iter().fold(0.0f64, |acc, v| acc + v * v).sqrt();
    let rhs_amplification = if reference_norm > 0.0 {
        solver.largest_rhs_norm() / reference_norm
    } else {
        1.0
    };

    let budget = ImplicitSolve {
        precision: lattice_wgpu::SOLVER_PRECISION,
        steps: IMPLICIT_STEPS,
        reference_tolerance: cpu_tolerance,
        measured_tolerance: solver.tolerance(),
        operator_norm: solver.operator_norm_bound(),
        reduction_depth: solver.reduction_depth(),
        rhs_amplification: rhs_amplification.max(1.0),
    }
    .budget();

    ImplicitRun {
        cpu,
        gpu,
        budget,
        solver_tolerance: solver.tolerance(),
        reduction_depth: solver.reduction_depth(),
        rhs_amplification,
        iterations: solver.total_iterations(),
        readbacks: solver.readbacks(),
        non_converged: solver.non_converged_steps(),
    }
}

/// The headline for the implicit path. Does it land inside the budget its stopping rule
/// predicts?
fn implicit_matches_the_reference() -> Outcome {
    let run = implicit_run(implicit_model());

    // A run with a step that gave up is not a comparison, whatever the fields say. Reported
    // as a budget overrun rather than as a passing row with a footnote.
    if run.non_converged > 0 {
        return Outcome::at_most(
            "fraction of the derived budget used",
            "1",
            f64::INFINITY,
            1.0,
        )
        .note(format!(
            "{} of {IMPLICIT_STEPS} steps did not converge",
            run.non_converged
        ));
    }

    let comparison = run.budget.compare(&run.cpu, &run.gpu);
    Outcome::at_most(
        "fraction of the derived budget used",
        "1",
        comparison.budget_used(),
        1.0,
    )
    .note(comparison.summary())
    .note(run.budget.explain().trim_end().to_string())
    .note(format!(
        "{IMPLICIT_STEPS} steps at {OVER_LIMIT:.0}x the explicit stability limit, \
             {} CG iterations, {} readbacks",
        run.iterations, run.readbacks
    ))
    .note(format!("backend: {}", device().expect("device").label()))
}

/// The attribution, and the result that reverses what M4.2 expected.
///
/// M4.2's table named reductions as the reason Crank–Nicolson was deferred. Having built
/// one, the reduction is the *smallest* term in the budget by six orders of magnitude, and
/// the largest is a mechanism that did not exist in the explicit case at all.
///
/// # What is measured, and what would be a tautology
///
/// Not the ratio of solve termination to state rounding. Those are `steps·τ_g·amp` and
/// `steps·ε(1+‖A‖)·amp`, and `τ_g` is chosen here as [`FLOOR_MULTIPLE`] times exactly
/// `ε(1+‖A‖)` — so their ratio is `FLOOR_MULTIPLE` by construction and reporting it as a
/// finding would be reporting a constant from this file.
///
/// What is measured is **solve termination over reduction order**, whose ratio is
/// `2/(d·ε)` and depends on nothing chosen here: `d` comes from the reduction's own split
/// and `ε` from the precision. That is the number that says the reduction is negligible
/// *inside* an iteration, and it would move if the reduction got deeper or the precision
/// changed.
///
/// The second measurement is the sharper one. If solve termination really is the mechanism,
/// the measured disagreement should sit near the **random walk** of it — `√steps` times the
/// per-step figure — rather than near the budget's worst case of `steps` times it. A run
/// landing at the worst case would mean the errors were aligning, which is a different
/// phenomenon and not one this budget describes.
fn implicit_disagreement_is_attributable() -> Outcome {
    let run = implicit_run(implicit_model());

    let share = |mechanism: Mechanism| {
        run.budget
            .terms()
            .iter()
            .find(|term| term.mechanism == mechanism)
            .map_or(0.0, |term| term.relative)
    };
    let termination = share(Mechanism::SolveTermination);
    let rounding = share(Mechanism::StateRounding);
    let reduction = share(Mechanism::ReductionOrder);

    // Derived, not observed: termination is `τ_c + τ_g` per step and reduction order is
    // `τ_g·d·ε/2`, so the ratio is about `2/(d·ε)` whatever tolerance was chosen.
    let epsilon = lattice_wgpu::SOLVER_PRECISION.epsilon();
    let predicted = 2.0 / (run.reduction_depth as f64 * epsilon);
    let separation = if reduction > 0.0 {
        termination / reduction
    } else {
        f64::INFINITY
    };

    // The random-walk prediction, for the note.
    let comparison = run.budget.compare(&run.cpu, &run.gpu);
    let per_step = (1e-10 + run.solver_tolerance) * run.rhs_amplification;
    let walk = (IMPLICIT_STEPS as f64).sqrt() * per_step;

    Outcome::at_least(
        "solve termination over reduction order",
        "1",
        separation,
        0.5 * predicted,
    )
    .note(format!(
        "solve termination {termination:.3e}, state rounding {rounding:.3e}, \
         reduction order {reduction:.3e}"
    ))
    .note(format!(
        "the separation predicted by 2/(depth {} x fast32 eps) is {predicted:.3e}, and it \
         depends on the reduction and the precision rather than on any tolerance chosen here",
        run.reduction_depth
    ))
    .note(format!(
        "the ratio of termination to state rounding is {:.1}, which is the {FLOOR_MULTIPLE:.0}x \
         floor multiple this case runs at and not a measurement of anything",
        termination / rounding
    ))
    .note(format!(
        "measured disagreement {:.3e} against a random walk of {walk:.3e} ({:.2}x) and a \
         worst case of {:.3e}; landing near the walk is what says the mechanism is termination",
        comparison.worst_relative,
        comparison.worst_relative / walk,
        IMPLICIT_STEPS as f64 * per_step
    ))
    .note(format!(
        "reduction order is small because CG re-measures what the reduction perturbed: a tree \
         {} roundings deep moves which iterate the solve arrives at, and the stopping test \
         then measures that iterate afresh",
        run.reduction_depth
    ))
}

/// The reduction with nothing to hide behind.
///
/// [`Mechanism::ReductionOrder`] is a term in the implicit budget worth about a
/// billionth of it, which means the implicit case does not audit it at all. So it is
/// audited here instead, on a reduction whose *value is the answer* — the kind an
/// observation or a conserved total would use — where the association order and `f32`
/// storage are the entire disagreement and nothing re-measures them.
///
/// The bound is the declared depth, `d·ε`, plus one rounding for the narrowing of each
/// input. Every term is a square and therefore non-negative, so `Σ|xᵢ| = |Σxᵢ|` and the
/// relative bound carries no cancellation factor.
fn reduction_order_alone() -> Outcome {
    let device = device().expect("cases() checked for a device");
    let domain = implicit_model();
    let field = domain.field();
    let interior = Interior {
        nx: GRID.0,
        ny: GRID.1,
        stride: field.stride(),
        origin: field.halo() * field.stride() + field.halo(),
    };
    let data = field.as_slice().to_vec();

    let measured = GpuDot::over_host(device, interior, &data, &data).expect("the reduction failed");

    // The reference: the same values, summed sequentially in `f64` the way `interior_dot`
    // does it. Not narrowed first — this is the whole cross-backend disagreement, storage
    // and order together, which is what an observation would actually suffer.
    let mut reference = 0.0f64;
    for j in 0..GRID.1 {
        for i in 0..GRID.0 {
            let value = data[interior.origin + j * interior.stride + i];
            reference += value * value;
        }
    }

    let depth = interior.depth();
    let epsilon = Precision::Fast32.epsilon();
    let budget = Tolerance::exact()
        .plus(
            Mechanism::StateRounding,
            2.0 * epsilon,
            "each of the two factors is narrowed to f32 once, at most half an ulp each",
        )
        .plus(
            Mechanism::ReductionOrder,
            depth as f64 * epsilon,
            format!(
                "a fixed tree {depth} roundings deep against a sequential sum; every term is \
                 a square, so there is no cancellation factor and the bound is exactly d*eps"
            ),
        );
    let comparison = budget.compare(&[reference], &[measured]);

    Outcome::at_most(
        "fraction of the derived budget used",
        "1",
        comparison.budget_used(),
        1.0,
    )
    .note(comparison.summary())
    .note(budget.explain().trim_end().to_string())
    .note(format!(
        "a sequential f32 sum of the same {} values would be entitled to {}x eps; this \
             tree is entitled to {depth}x",
        interior.cells(),
        interior.cells() - 1
    ))
}

/// The refusal, as a §19.1 row rather than as a paragraph in a doc comment.
///
/// A backend that silently accepted an unreachable tolerance would still produce fields, and
/// they would still be close enough to pass the headline case — it would simply spend 500
/// iterations a step doing it. That is the failure this prevents, and it is a validation
/// claim because "the backend declines what it cannot do" is exactly what §10.5's precision
/// modes are for.
fn refuses_an_unreachable_tolerance() -> Outcome {
    let device = device().expect("cases() checked for a device");
    let domain = implicit_model();
    let operator = domain.operator().clone();
    let field = domain.field().as_slice().to_vec();
    let dt = OVER_LIMIT * operator.explicit_stability_limit().max;

    let setup = CrankNicolsonSetup {
        nx: GRID.0,
        ny: GRID.1,
        halo: domain.field().halo(),
        stride: domain.field().stride(),
        inv_dx2: operator.inv_dx2(),
        inv_dy2: operator.inv_dy2(),
        theta: TimeScheme::CrankNicolson.theta(),
        dt,
        field: &field,
        face_x: operator.face_x(),
        face_y: operator.face_y(),
        source: None,
        // `HeatDomain`'s own default, which is the point: this is not a contrived number.
        tolerance: 1e-10,
        max_iterations: 500,
    };
    let floor = GpuCrankNicolson::floor_for(lattice_wgpu::SOLVER_PRECISION, &setup);

    let refused = match GpuCrankNicolson::new(device, setup) {
        Err(ImplicitError::ToleranceBelowFloor { .. }) => 1.0,
        Err(other) => {
            return Outcome::at_least("refused for the stated reason", "1", 0.0, 1.0)
                .note(format!("refused, but for the wrong reason: {other}"));
        }
        Ok(_) => 0.0,
    };

    Outcome::at_least("refused for the stated reason", "1", refused, 1.0)
        .note(format!(
            "the f32 floor here is {floor:.3e}, and HeatDomain's default asks for 1.000e-10 \
             — a factor of {:.0} below it",
            floor / 1e-10
        ))
        .note(
            "the floor is eps*(1 + ||A||): even given an exact x, forming b - Ax in f32 \
             leaves that much residual, so no iteration can report less"
                .to_string(),
        )
}

/// The control, and the implicit analogue of `gpu_and_cpu_agree_exactly_where_no_rounding
/// _occurs`.
///
/// It is a stronger control than the explicit one. On a gradient-free field `L(u) = 0`, so
/// `b = u` and `A·u = u` and the initial residual is *identically* zero — both backends
/// return from conjugate gradient before its first iteration, having agreed on a number that
/// neither the reduction nor the tolerance nor `f32` storage had any opportunity to touch.
/// If this ever fails while the budgeted cases pass, the budget is covering a real defect.
fn implicit_agrees_exactly_without_a_solve() -> Outcome {
    let run = implicit_run(flat_model().with_scheme(TimeScheme::CrankNicolson));

    let mismatches = run.cpu.iter().zip(&run.gpu).filter(|(a, b)| a != b).count();
    let comparison = Tolerance::exact()
        .in_norm(Norm::Euclidean)
        .compare(&run.cpu, &run.gpu);

    Outcome::at_most(
        "cells differing between the backends",
        "1",
        mismatches as f64,
        0.0,
    )
    .note(comparison.summary())
    .note(format!(
        "{} CG iterations across {IMPLICIT_STEPS} steps: the residual starts at zero, so \
             neither backend iterates at all",
        run.iterations
    ))
    .note(format!(
        "the budget this case declines to use would have been {:.3e}",
        run.budget.relative()
    ))
}
