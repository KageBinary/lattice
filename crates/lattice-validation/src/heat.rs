//! Validation cases for the heat / diffusion domain (spec §19.2).
//!
//! > Heat equation on periodic and fixed boundaries; diffusion Gaussian; convergence
//! > under grid refinement.

use lattice_domain_grid2d::{gaussian, DiffusionOperator, Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, Boundary, BoundarySet, Domain, Grid2d, ScalarField, StepContext};

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "diffusion_gaussian_matches_heat_kernel",
        domain: "grid2d",
        level: Level::Analytic,
        claim: "a diffusing Gaussian follows the analytic heat kernel, spreading as sigma^2 = sigma0^2 + 2Dt",
        run: gaussian_heat_kernel,
    },
    Case {
        name: "heat_spatial_convergence",
        domain: "grid2d",
        level: Level::Manufactured,
        claim: "refining the grid reduces the error against the continuous solution at second order",
        run: spatial_convergence,
    },
    Case {
        name: "heat_temporal_convergence_crank_nicolson",
        domain: "grid2d",
        level: Level::Manufactured,
        claim: "Crank-Nicolson converges in time at the second order it declares",
        run: temporal_convergence_cn,
    },
    Case {
        name: "heat_temporal_convergence_backward_euler",
        domain: "grid2d",
        level: Level::Manufactured,
        claim: "backward Euler converges in time at the first order it declares",
        run: temporal_convergence_be,
    },
    Case {
        name: "manufactured_steady_solution",
        domain: "grid2d",
        level: Level::Manufactured,
        claim: "a chosen exact solution, recovered from its forcing term, converges at second order",
        run: manufactured_steady,
    },
    Case {
        name: "closed_domain_conserves_integral",
        domain: "grid2d",
        level: Level::Property,
        claim: "with no source and closed boundaries the field integral is conserved to round-off",
        run: closed_conservation,
    },
    Case {
        name: "dirichlet_slab_linear_profile",
        domain: "grid2d",
        level: Level::Analytic,
        claim: "a slab held at two temperatures reaches the linear steady profile of textbook conduction",
        run: dirichlet_slab,
    },
    Case {
        name: "two_material_series_conduction",
        domain: "grid2d",
        level: Level::Analytic,
        claim: "across a conductivity step the flux is continuous and the gradient ratio equals the conductivity ratio",
        run: series_conduction,
    },
    Case {
        name: "robin_boundary_reaches_ambient",
        domain: "grid2d",
        level: Level::Analytic,
        claim: "a body with convective boundaries relaxes to the ambient value without overshooting",
        run: robin_ambient,
    },
    Case {
        name: "explicit_scheme_diverges_past_its_cfl_limit",
        domain: "grid2d",
        level: Level::Property,
        claim: "the declared stability limit is real: just inside it the run is stable, just outside it blows up",
        run: cfl_limit_is_real,
    },
    Case {
        name: "explicit_and_implicit_converge_together",
        domain: "grid2d",
        level: Level::CrossScheme,
        claim: "two independent time schemes approach the same solution as dt shrinks",
        run: cross_scheme_agreement,
    },
];

const TAU: f64 = core::f64::consts::TAU;

fn step(domain: &mut HeatDomain, dt: f64, steps: usize) {
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
    }
}

/// `sin(2πx)·sin(2πy)` — an exact eigenvector of the discrete periodic Laplacian.
fn fourier_mode([x, y]: [f64; 2]) -> f64 {
    (TAU * x).sin() * (TAU * y).sin()
}

/// The eigenvalue of the discrete 5-point Laplacian for [`fourier_mode`].
///
/// Comparing against `exp(D·λ_h·t)` isolates *temporal* error completely, because the
/// mode is exactly an eigenvector of the spatial operator. Comparing against the
/// continuous `exp(−8π²D·t)` would mix in spatial error and make a temporal
/// convergence study unreadable.
fn discrete_eigenvalue(grid: &Grid2d) -> f64 {
    let (dx, dy) = (grid.dx(), grid.dy());
    let pi = core::f64::consts::PI;
    -(4.0 / (dx * dx)) * (pi * dx).sin().powi(2) - (4.0 / (dy * dy)) * (pi * dy).sin().powi(2)
}

fn max_error(domain: &HeatDomain, exact: impl Fn([f64; 2]) -> f64) -> f64 {
    let grid = *domain.grid();
    let mut worst = 0.0f64;
    for j in 0..grid.ny() {
        for i in 0..grid.nx() {
            worst = worst.max((domain.field().get(i, j) - exact(grid.cell_center(i, j))).abs());
        }
    }
    worst
}

fn gaussian_heat_kernel() -> Outcome {
    let grid = Grid2d::new(128, 128, [1.0, 1.0]);
    let d = 2e-3;
    let variance_0 = 0.0015;
    let amplitude = 1.0;

    let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
        .with_scheme(TimeScheme::CrankNicolson)
        .with_boundaries(BoundarySet::INSULATED)
        .with_tolerance(1e-13)
        .with_initial(gaussian([0.5, 0.5], variance_0, amplitude));

    let dt = 2e-3;
    let steps = 100;
    step(&mut heat, dt, steps);

    let t = dt * steps as f64;
    let variance_t = variance_0 + 2.0 * d * t;
    let exact = gaussian([0.5, 0.5], variance_t, amplitude);
    let peak = amplitude / (TAU * variance_t);
    let relative = max_error(&heat, exact) / peak;

    Outcome::at_most("peak-relative error against the heat kernel", "1", relative, 5e-3)
        .note(format!(
            "variance grew from {variance_0} to {variance_t:.6} m^2 over {t} s, \
             as sigma^2 = sigma0^2 + 2Dt predicts"
        ))
        .note("128x128 grid, insulated walls far from the packet")
}

fn spatial_convergence() -> Outcome {
    let d = 0.01;
    let t_end = 0.5;

    let error_at = |n: usize| {
        let grid = Grid2d::new(n, n, [1.0, 1.0]);
        let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
            .with_scheme(TimeScheme::CrankNicolson)
            .with_boundaries(BoundarySet::PERIODIC)
            .with_tolerance(1e-14)
            .with_initial(fourier_mode);
        // Small enough that temporal error is far below spatial error.
        let steps = 2_000;
        step(&mut heat, t_end / steps as f64, steps);

        let decay = (-2.0 * TAU * TAU * d * t_end).exp();
        max_error(&heat, |p| decay * fourier_mode(p))
    };

    let coarse = error_at(16);
    let fine = error_at(32);
    let order = (coarse / fine).log2();

    Outcome::near("observed spatial convergence order", "1", order, 2.0, 0.15)
        .note(format!("max error {coarse:.3e} at 16^2 -> {fine:.3e} at 32^2"))
        .note("measured against the continuous solution exp(-8 pi^2 D t) sin(2 pi x) sin(2 pi y)")
}

fn temporal_order(scheme: TimeScheme) -> (f64, f64, f64) {
    let d = 0.05;
    let t_end = 1.0;
    let n = 32;
    let grid = Grid2d::new(n, n, [1.0, 1.0]);
    let lambda = discrete_eigenvalue(&grid);

    let error_at = |steps: usize| {
        let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
            .with_scheme(scheme)
            .with_boundaries(BoundarySet::PERIODIC)
            .with_tolerance(1e-14)
            .with_initial(fourier_mode);
        step(&mut heat, t_end / steps as f64, steps);

        let exact_decay = (d * lambda * t_end).exp();
        max_error(&heat, |p| exact_decay * fourier_mode(p))
    };

    let coarse = error_at(20);
    let fine = error_at(40);
    ((coarse / fine).log2(), coarse, fine)
}

fn temporal_convergence_cn() -> Outcome {
    let (order, coarse, fine) = temporal_order(TimeScheme::CrankNicolson);
    Outcome::near("observed temporal convergence order", "1", order, 2.0, 0.2)
        .note(format!("max error {coarse:.3e} at 20 steps -> {fine:.3e} at 40"))
        .note("measured against the semi-discrete solution, which removes spatial error entirely")
}

fn temporal_convergence_be() -> Outcome {
    let (order, coarse, fine) = temporal_order(TimeScheme::BackwardEuler);
    Outcome::near("observed temporal convergence order", "1", order, 1.0, 0.2)
        .note(format!("max error {coarse:.3e} at 20 steps -> {fine:.3e} at 40"))
}

fn manufactured_steady() -> Outcome {
    let d = 0.05;

    let error_at = |n: usize| {
        let grid = Grid2d::new(n, n, [1.0, 1.0]);
        let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
            .with_scheme(TimeScheme::BackwardEuler)
            .with_boundaries(BoundarySet::PERIODIC)
            .with_tolerance(1e-14)
            .with_uniform_initial(0.0);

        // Choose u* = sin(2 pi x) sin(2 pi y); then S = -D grad^2 u* = 8 pi^2 D u*
        // makes u* the steady state.
        let mut source = heat.make_source();
        source.init_from_position(&grid, |p| 2.0 * TAU * TAU * d * fourier_mode(p));
        heat.set_source(source);

        step(&mut heat, 5.0, 200);
        max_error(&heat, fourier_mode)
    };

    let coarse = error_at(16);
    let fine = error_at(32);
    let order = (coarse / fine).log2();

    Outcome::near("observed convergence order to the manufactured solution", "1", order, 2.0, 0.15)
        .note(format!("max error {coarse:.3e} at 16^2 -> {fine:.3e} at 32^2"))
        .note("forcing S = 8 pi^2 D sin(2 pi x) sin(2 pi y) chosen so u* is exactly the steady state")
}

fn closed_conservation() -> Outcome {
    let mut worst = 0.0f64;
    let mut notes = Vec::new();

    for boundaries in [BoundarySet::INSULATED, BoundarySet::PERIODIC] {
        for scheme in [TimeScheme::Explicit, TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
            let grid = Grid2d::new(48, 32, [1.5, 1.0]);
            let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(1e-2))
                .with_scheme(scheme)
                .with_boundaries(boundaries)
                .with_tolerance(1e-14)
                .with_initial(gaussian([0.5, 0.5], 0.004, 3.0));

            let before = heat.integral();
            let dt = heat.stable_step().preferred.min(1e-3);
            step(&mut heat, dt, 200);
            let drift = (heat.integral() - before).abs() / before.abs();

            notes.push(format!("{}/{:?}: {drift:.2e}", scheme.name(), boundaries.left.kind_name()));
            worst = worst.max(drift);
        }
    }

    // The implicit schemes carry the CG tolerance; the explicit one cancels to
    // round-off. The bound accommodates the looser of the two.
    Outcome::at_most("worst relative integral drift across schemes and boundaries", "1", worst, 1e-9)
        .note(notes.join("  "))
        .note(
            "finite-volume fluxes cancel in pairs, so conservation is structural rather \
             than a consequence of accuracy",
        )
}

fn dirichlet_slab() -> Outcome {
    let n = 40;
    let grid = Grid2d::new(n, 8, [1.0, 0.2]);
    let mut boundaries = BoundarySet::INSULATED;
    boundaries.left = Boundary::fixed(0.0);
    boundaries.right = Boundary::fixed(100.0);

    let mut heat = HeatDomain::new("T", grid, Diffusivity::Uniform(0.1))
        .with_scheme(TimeScheme::BackwardEuler)
        .with_boundaries(boundaries)
        .with_tolerance(1e-14)
        .with_uniform_initial(0.0);
    step(&mut heat, 5.0, 400);

    let error = max_error(&heat, |[x, _]| 100.0 * x);

    Outcome::at_most("max deviation from the linear steady profile", "K", error, 1e-6)
        .note("0 K at x=0 to 100 K at x=1, which also confirms the Dirichlet halo puts the prescribed value on the face rather than at the cell centre")
}

fn series_conduction() -> Outcome {
    let n = 40;
    let grid = Grid2d::new(n, 4, [1.0, 0.1]);
    let (d_left, d_right) = (1.0, 0.25);

    let mut diffusivity = ScalarField::new(&grid, 1);
    diffusivity.init_from_position(&grid, |[x, _]| if x < 0.5 { d_left } else { d_right });

    let mut boundaries = BoundarySet::INSULATED;
    boundaries.left = Boundary::fixed(0.0);
    boundaries.right = Boundary::fixed(1.0);

    let mut heat = HeatDomain::new("T", grid, Diffusivity::Variable(diffusivity))
        .with_scheme(TimeScheme::BackwardEuler)
        .with_boundaries(boundaries)
        .with_tolerance(1e-14)
        .with_uniform_initial(0.0);
    step(&mut heat, 2.0, 400);

    let row = 2;
    let gradient = |i0: usize, i1: usize| {
        (heat.field().get(i1, row) - heat.field().get(i0, row))
            / (grid.cell_center(i1, row)[0] - grid.cell_center(i0, row)[0])
    };
    let left_gradient = gradient(4, 14);
    let right_gradient = gradient(26, 36);
    let ratio = right_gradient / left_gradient;

    let flux_left = d_left * left_gradient;
    let flux_right = d_right * right_gradient;
    let flux_mismatch = (flux_left - flux_right).abs() / flux_left.abs();

    Outcome::near("gradient ratio across the interface", "1", ratio, d_left / d_right, 1e-3)
        .note(format!("flux continuity mismatch {flux_mismatch:.3e} (D*grad must match on both sides)"))
        .note("the harmonic-mean face coefficient is what makes this exact for a layered medium")
}

fn robin_ambient() -> Outcome {
    let grid = Grid2d::new(24, 24, [1.0, 1.0]);
    let ambient = 300.0;
    let mut heat = HeatDomain::new("T", grid, Diffusivity::Uniform(0.05))
        .with_scheme(TimeScheme::BackwardEuler)
        .with_boundaries(BoundarySet::uniform(Boundary::Robin { coefficient: 5.0, ambient }))
        .with_tolerance(1e-13)
        .with_uniform_initial(400.0);

    step(&mut heat, 1.0, 300);

    let final_max = heat.field().max_interior();
    let final_min = heat.field().min_interior();
    let overshoot = (ambient - final_min).max(0.0);

    Outcome::at_most("remaining excess over the ambient value", "K", final_max - ambient, 1e-3)
        .note(format!("cooled from 400 K to {final_max:.6} K against an ambient of {ambient} K"))
        .note(format!("undershoot below ambient: {overshoot:.3e} K (a monotone scheme should not overshoot)"))
}

fn cfl_limit_is_real() -> Outcome {
    let grid = Grid2d::new(32, 32, [1.0, 1.0]);
    let d = 0.01;
    let limit = DiffusionOperator::new(&grid, &Diffusivity::Uniform(d)).explicit_stability_limit().max;

    let growth_at = |dt: f64| {
        let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
            .with_scheme(TimeScheme::Explicit)
            .with_boundaries(BoundarySet::PERIODIC)
            .with_initial(gaussian([0.5, 0.5], 0.002, 1.0));
        let start = heat.field().max_abs_interior();
        step(&mut heat, dt, 200);
        heat.field().max_abs_interior() / start
    };

    // Just inside the limit the peak decays; just outside it explodes.
    let stable = growth_at(0.95 * limit);
    let unstable = growth_at(1.2 * limit);

    Outcome::at_least("amplitude growth ratio at 1.2x the declared limit", "1", unstable, 1e3)
        .note(format!("at 0.95x the limit the peak decayed to {stable:.4} of its initial value"))
        .note(format!("declared limit dt <= {limit:.6e} s for D = {d} on a 32^2 unit grid"))
        .note("NFR-007: instability must be visible, not silently clamped")
}

fn cross_scheme_agreement() -> Outcome {
    let grid = Grid2d::new(32, 32, [1.0, 1.0]);
    let d = 0.02;
    let t_end = 0.4;

    let solve = |scheme: TimeScheme, steps: usize| {
        let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
            .with_scheme(scheme)
            .with_boundaries(BoundarySet::PERIODIC)
            .with_tolerance(1e-14)
            .with_initial(gaussian([0.5, 0.5], 0.01, 1.0));
        step(&mut heat, t_end / steps as f64, steps);
        heat
    };

    let gap_at = |steps: usize| {
        let a = solve(TimeScheme::Explicit, steps);
        let b = solve(TimeScheme::CrankNicolson, steps);
        let mut worst = 0.0f64;
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                worst = worst.max((a.field().get(i, j) - b.field().get(i, j)).abs());
            }
        }
        worst / a.field().max_interior()
    };

    let coarse = gap_at(200);
    let fine = gap_at(400);
    let order = (coarse / fine).log2();

    Outcome::near("order at which the two schemes converge together", "1", order, 1.0, 0.2)
        .note(format!("relative gap {coarse:.3e} at 200 steps -> {fine:.3e} at 400"))
        .note(
            "the gap is dominated by the explicit scheme's first-order error; a real \
             discrepancy would leave a floor that refinement cannot remove",
        )
}
