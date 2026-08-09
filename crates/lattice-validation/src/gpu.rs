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
//! Not from running the comparison and rounding up. The budget is
//! [`Tolerance::stepped_kernel`], derived before the measurement from three stated facts —
//! the stencil rounds about five times per cell per step, an explicit diffusion step below
//! its stability limit does not amplify error it already has, and additive accumulation
//! over `n` steps is `n` times the per-step figure. The budget is therefore a worst case
//! that real round-off, behaving like a random walk, should sit far below.
//!
//! `gpu_disagreement_stays_within_its_derived_budget` reports the *fraction of the budget
//! used*, which is the number worth watching: a case sitting at a few percent says the
//! bound is loose and honest, and one climbing toward 100% across releases says something
//! is degrading while still passing.
//!
//! # Why these cases can be absent
//!
//! A machine with no GPU produces no rows here at all, the way M3 leaves the Gillespie
//! statistics absent rather than reporting them as skipped-and-passing. [`cases`] returns
//! an empty list when no adapter opens, and the crate is compiled without `--features gpu`
//! by default so that `lattice validate` does not drag `wgpu` into the CLI.

use std::sync::OnceLock;

use lattice_compute::{Device, Mechanism, Precision, Tolerance};
use lattice_domain_grid2d::{gaussian, Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, BoundarySet, Domain, Executor, Grid2d, ScalarField, StepContext};
use lattice_wgpu::{DiffusionSetup, GpuDevice, GpuDiffusion};

use crate::{Case, Level, Outcome};

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
    DEVICE.get_or_init(|| GpuDevice::open_default().ok()).as_ref()
}

/// The cases, or none if this machine has no portable GPU backend.
pub(crate) fn cases() -> Vec<Case> {
    if device().is_none() {
        return Vec::new();
    }
    vec![
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
    ]
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
    let budget = Tolerance::stepped_kernel(lattice_wgpu::SOLVER_PRECISION, STEPS)
        .allowing_fma(lattice_wgpu::SOLVER_PRECISION, 4, STEPS);
    let comparison = budget.compare(&cpu, &gpu);

    Outcome::at_most("fraction of the derived budget used", "1", comparison.budget_used(), 1.0)
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

    Outcome::at_most("worst CPU/GPU disagreement, in f32 ulps", "1", in_f32_ulps, derived_bound)
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

    Outcome::at_most("cells differing between the backends", "1", mismatches as f64, 0.0)
        .note(comparison.summary())
        .note("no gradient means every flux is a difference of equal numbers".to_string())
}
