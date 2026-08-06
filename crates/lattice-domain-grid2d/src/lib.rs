//! 2D grid PDE solvers: heat and diffusion.
//!
//! Spec §11.4 asks for an *"explicit scheme for interactive low-stiffness cases with
//! CFL diagnostics"*, an *"implicit or Crank–Nicolson scheme for larger timesteps and
//! validated runs"*, *"spatially varying conductivity, diffusivity, heat capacity, and
//! sources"*, and *"Dirichlet, Neumann, Robin, periodic […] boundaries"*. All of that
//! is here, solving
//!
//! ```text
//!   ∂u/∂t = ∇·(D ∇u) + S
//! ```
//!
//! on a uniform cell-centred grid.
//!
//! # Example
//!
//! ```
//! use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme, gaussian};
//! use lattice_ir::{Arena, BoundarySet, Domain, Grid2d, StepContext};
//!
//! let grid = Grid2d::new(64, 64, [1.0, 1.0]);
//! let diffusivity = 1e-3;
//! let mut heat = HeatDomain::new("temperature", grid, Diffusivity::Uniform(diffusivity))
//!     .with_scheme(TimeScheme::CrankNicolson)
//!     .with_boundaries(BoundarySet::INSULATED)
//!     .with_initial(gaussian([0.5, 0.5], 0.002, 1.0));
//!
//! // A closed domain with no source conserves the integral exactly.
//! assert!(heat.is_closed());
//! let before = heat.integral();
//!
//! let mut arena = Arena::with_capacity(0);
//! let mut ctx = StepContext::new(&mut arena);
//! for _ in 0..100 {
//!     heat.advance(1e-3, &mut ctx);
//! }
//!
//! let drift = (heat.integral() - before).abs() / before;
//! assert!(drift < 1e-9, "integral drifted {drift:e}");
//! // ...and the peak has spread out.
//! assert!(heat.field().max_interior() < 1.0 / (std::f64::consts::TAU * 0.002));
//! ```
//!
//! # Choosing a scheme
//!
//! | Scheme | Order | Stability | Reach for it when |
//! |---|---|---|---|
//! | [`TimeScheme::Explicit`] | 1 | `dt ≤ 1/(2D(1/dx²+1/dy²))` | the step is small anyway and you want no solve |
//! | [`TimeScheme::CrankNicolson`] | 2 | unconditional | initial data is smooth and you want large steps |
//! | [`TimeScheme::BackwardEuler`] | 1 | unconditional | initial data has a jump, or the problem is stiff |
//!
//! "Unconditionally stable" is the most over-read phrase in numerical PDEs. It means
//! the solution will not grow without bound — not that it will be right. Crank–Nicolson
//! at large `dt` produces decaying *oscillations* around a discontinuity rather than
//! smooth decay, because its amplification factor for the shortest modes approaches
//! −1. Backward Euler damps those modes monotonically and is the better choice there,
//! at the cost of an order of accuracy.

mod boundary;
mod domain;
mod operator;
mod solver;

pub use boundary::{apply_boundaries, HaloMode};
pub use domain::{gaussian, heated_edge, left_half, HeatDomain, TimeScheme};
pub use operator::{DiffusionOperator, Diffusivity};
pub use solver::{conjugate_gradient, CgWorkspace};

#[cfg(test)]
mod validation {
    //! Physics validation for the heat/diffusion module (spec §19.2).
    //!
    //! These are not unit tests of the code — they are tests of the *mathematics*.
    //! Each one compares against something known independently of this
    //! implementation: an analytic solution, a manufactured solution, a conservation
    //! law, or a convergence rate the theory predicts.

    use super::*;
    use lattice_ir::{Arena, Boundary, BoundarySet, Domain, Grid2d, ScalarField, Side, StepContext};

    fn run(domain: &mut HeatDomain, dt: f64, steps: usize) {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
        }
    }

    /// The eigenvalue of the discrete 5-point Laplacian for the mode
    /// `sin(2πx)·sin(2πy)` on a periodic cell-centred grid.
    ///
    /// This mode is an *exact* eigenvector of the discrete operator, so comparing
    /// against `exp(D·λ_h·t)` isolates time-stepping error from spatial error
    /// completely. Comparing against the continuous `exp(−8π²D·t)` instead would
    /// conflate the two and make a convergence study unreadable.
    fn discrete_eigenvalue(grid: &Grid2d) -> f64 {
        let (dx, dy) = (grid.dx(), grid.dy());
        let pi = core::f64::consts::PI;
        -(4.0 / (dx * dx)) * (pi * dx).sin().powi(2)
            - (4.0 / (dy * dy)) * (pi * dy).sin().powi(2)
    }

    fn fourier_mode(grid: &Grid2d) -> impl Fn([f64; 2]) -> f64 {
        let tau = core::f64::consts::TAU;
        let _ = grid;
        move |[x, y]| (tau * x).sin() * (tau * y).sin()
    }

    /// A diffusing Gaussian is the analytic solution of the heat equation: its
    /// variance grows as `σ² = σ₀² + 2Dt` while its integral is unchanged.
    #[test]
    fn diffusing_gaussian_matches_the_analytic_heat_kernel() {
        // A domain large enough that the tails never reach the insulated walls.
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
        run(&mut heat, dt, steps);

        let t = dt * steps as f64;
        let variance_t = variance_0 + 2.0 * d * t;
        let expected = gaussian([0.5, 0.5], variance_t, amplitude);

        let mut worst = 0.0f64;
        let mut peak = 0.0f64;
        for j in 0..128 {
            for i in 0..128 {
                let want = expected(grid.cell_center(i, j));
                worst = worst.max((heat.field().get(i, j) - want).abs());
                peak = peak.max(want.abs());
            }
        }
        let relative = worst / peak;
        assert!(relative < 5e-3, "peak-relative error {relative:e} against the heat kernel");

        // The variance really did grow: the peak dropped by the predicted ratio.
        let observed_peak = heat.field().max_interior();
        let predicted_peak = amplitude / (core::f64::consts::TAU * variance_t);
        assert!(
            (observed_peak - predicted_peak).abs() / predicted_peak < 5e-3,
            "peak {observed_peak} vs predicted {predicted_peak}"
        );
    }

    /// On a closed domain with no source, `∫u dA` is conserved by construction — the
    /// finite-volume fluxes cancel in pairs. This must hold to round-off, not merely
    /// to truncation error, and for every scheme.
    #[test]
    fn a_closed_domain_conserves_the_field_integral() {
        for boundaries in [BoundarySet::INSULATED, BoundarySet::PERIODIC] {
            for scheme in
                [TimeScheme::Explicit, TimeScheme::CrankNicolson, TimeScheme::BackwardEuler]
            {
                let grid = Grid2d::new(48, 32, [1.5, 1.0]);
                let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(1e-2))
                    .with_scheme(scheme)
                    .with_boundaries(boundaries)
                    .with_tolerance(1e-14)
                    .with_initial(gaussian([0.5, 0.5], 0.004, 3.0));

                assert!(heat.is_closed(), "{scheme:?} setup should be closed");
                let before = heat.integral();
                let dt = heat.stable_step().preferred.min(1e-3);
                run(&mut heat, dt, 200);

                let drift = (heat.integral() - before).abs() / before.abs();
                // Explicit and periodic cases cancel to round-off. The implicit
                // schemes carry the CG tolerance, hence the looser bound.
                let bound = if scheme == TimeScheme::Explicit { 1e-13 } else { 1e-9 };
                assert!(
                    drift < bound,
                    "{scheme:?} with {boundaries:?} drifted {drift:e} (bound {bound:e})"
                );
            }
        }
    }

    /// Spatial convergence: refining the grid must reduce the error against the
    /// *continuous* solution at second order.
    #[test]
    fn spatial_convergence_is_second_order() {
        let d = 0.01;
        let t_end = 0.5;
        let tau = core::f64::consts::TAU;

        let error_at = |n: usize| {
            let grid = Grid2d::new(n, n, [1.0, 1.0]);
            let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
                .with_scheme(TimeScheme::CrankNicolson)
                .with_boundaries(BoundarySet::PERIODIC)
                .with_tolerance(1e-14)
                .with_initial(fourier_mode(&grid));

            // A timestep small enough that temporal error is far below spatial error.
            let steps = 2000;
            run(&mut heat, t_end / steps as f64, steps);

            // Continuous solution: exp(-8 pi^2 D t) * sin(2 pi x) sin(2 pi y).
            let decay = (-2.0 * tau * tau * d * t_end).exp();
            let mut worst = 0.0f64;
            for j in 0..n {
                for i in 0..n {
                    let [x, y] = grid.cell_center(i, j);
                    let exact = decay * (tau * x).sin() * (tau * y).sin();
                    worst = worst.max((heat.field().get(i, j) - exact).abs());
                }
            }
            worst
        };

        let coarse = error_at(16);
        let fine = error_at(32);
        let order = (coarse / fine).log2();
        assert!(
            (order - 2.0).abs() < 0.15,
            "observed spatial order {order:.3} (errors {coarse:e} -> {fine:e})"
        );
    }

    /// Temporal convergence: each scheme must show the order it declares.
    ///
    /// Measured against the semi-discrete solution `exp(D·λ_h·t)`, which removes
    /// spatial error entirely and leaves only the time-stepping error.
    #[test]
    fn temporal_convergence_matches_each_declared_order() {
        let d = 0.05;
        let t_end = 1.0;
        let n = 32;
        let grid = Grid2d::new(n, n, [1.0, 1.0]);
        let lambda = discrete_eigenvalue(&grid);

        for scheme in [TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
            let error_at = |steps: usize| {
                let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
                    .with_scheme(scheme)
                    .with_boundaries(BoundarySet::PERIODIC)
                    .with_tolerance(1e-14)
                    .with_initial(fourier_mode(&grid));
                run(&mut heat, t_end / steps as f64, steps);

                let exact_decay = (d * lambda * t_end).exp();
                let mut worst = 0.0f64;
                for j in 0..n {
                    for i in 0..n {
                        let [x, y] = grid.cell_center(i, j);
                        let tau = core::f64::consts::TAU;
                        let exact = exact_decay * (tau * x).sin() * (tau * y).sin();
                        worst = worst.max((heat.field().get(i, j) - exact).abs());
                    }
                }
                worst
            };

            let coarse = error_at(20);
            let fine = error_at(40);
            let order = (coarse / fine).log2();
            let declared = f64::from(scheme.order());
            assert!(
                (order - declared).abs() < 0.2,
                "{scheme:?}: observed temporal order {order:.3}, declared {declared} \
                 (errors {coarse:e} -> {fine:e})"
            );
        }
    }

    /// A manufactured steady solution. Choosing `u* = sin(2πx)sin(2πy)` and setting
    /// `S = −D∇²u* = 8π²D·u*` makes `u*` the steady state; running to equilibrium
    /// from zero must find it, at second order in `h`.
    #[test]
    fn manufactured_steady_solution_converges_at_second_order() {
        let d = 0.05;
        let tau = core::f64::consts::TAU;

        let error_at = |n: usize| {
            let grid = Grid2d::new(n, n, [1.0, 1.0]);
            let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
                .with_scheme(TimeScheme::BackwardEuler)
                .with_boundaries(BoundarySet::PERIODIC)
                .with_tolerance(1e-14)
                .with_uniform_initial(0.0);

            let mut source = heat.make_source();
            source.init_from_position(&grid, |[x, y]| {
                2.0 * tau * tau * d * (tau * x).sin() * (tau * y).sin()
            });
            heat.set_source(source);

            // Backward Euler at a large step drives straight to equilibrium.
            run(&mut heat, 5.0, 200);

            let mut worst = 0.0f64;
            for j in 0..n {
                for i in 0..n {
                    let [x, y] = grid.cell_center(i, j);
                    let exact = (tau * x).sin() * (tau * y).sin();
                    worst = worst.max((heat.field().get(i, j) - exact).abs());
                }
            }
            worst
        };

        let coarse = error_at(16);
        let fine = error_at(32);
        let order = (coarse / fine).log2();
        assert!(
            (order - 2.0).abs() < 0.15,
            "observed order {order:.3} (errors {coarse:e} -> {fine:e})"
        );
    }

    /// A slab held at two different temperatures reaches a linear profile — the
    /// textbook steady conduction solution, and a direct check that the Dirichlet
    /// halo puts the prescribed value on the *face* rather than at the cell centre.
    #[test]
    fn dirichlet_slab_reaches_a_linear_steady_profile() {
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

        assert!(!heat.is_closed(), "Dirichlet edges make this an open system");
        run(&mut heat, 5.0, 400);

        for i in 0..n {
            let x = grid.cell_center(i, 4)[0];
            let expected = 100.0 * x; // linear from 0 at x=0 to 100 at x=1
            let got = heat.field().get(i, 4);
            assert!((got - expected).abs() < 1e-6, "cell {i}: {got} vs {expected}");
        }
    }

    /// Explicit and Crank–Nicolson must converge to the *same* answer as `dt → 0`.
    ///
    /// Asserting that they agree closely at one particular `dt` would be a weaker and
    /// more arbitrary claim — the two schemes have different orders, so at any fixed
    /// step they differ by whatever the first-order scheme's error happens to be.
    /// What actually matters is that the gap closes at that order: a genuine
    /// discrepancy (a boundary sign error, a wrong face coefficient) would leave a
    /// floor that refinement never removes.
    #[test]
    fn explicit_and_implicit_converge_to_the_same_solution() {
        let grid = Grid2d::new(32, 32, [1.0, 1.0]);
        let d = 0.02;
        let t_end = 0.4;

        let solve = |scheme: TimeScheme, steps: usize| {
            let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
                .with_scheme(scheme)
                .with_boundaries(BoundarySet::PERIODIC)
                .with_tolerance(1e-14)
                .with_initial(gaussian([0.5, 0.5], 0.01, 1.0));
            run(&mut heat, t_end / steps as f64, steps);
            heat
        };

        let gap_at = |steps: usize| {
            let a = solve(TimeScheme::Explicit, steps);
            let b = solve(TimeScheme::CrankNicolson, steps);
            let mut worst = 0.0f64;
            for j in 0..32 {
                for i in 0..32 {
                    worst = worst.max((a.field().get(i, j) - b.field().get(i, j)).abs());
                }
            }
            worst / a.field().max_interior()
        };

        // Both step counts are inside the explicit stability limit.
        let limit = DiffusionOperator::new(&grid, &Diffusivity::Uniform(d))
            .explicit_stability_limit()
            .max;
        assert!(t_end / 200.0 < limit, "the coarse run must still be stable");

        let coarse = gap_at(200);
        let fine = gap_at(400);
        let order = (coarse / fine).log2();

        // The gap is dominated by the explicit scheme's first-order error.
        assert!(
            (order - 1.0).abs() < 0.2,
            "the schemes should converge together at first order, observed {order:.3} \
             (gaps {coarse:e} -> {fine:e})"
        );
        assert!(fine < coarse, "refinement must actually reduce the gap");
    }

    /// Variable diffusivity: a two-material slab in series has a known steady profile,
    /// and the temperature *gradient* is steeper in the less conductive half. This is
    /// what the harmonic-mean face coefficient exists to get right.
    #[test]
    fn a_two_material_slab_matches_series_conduction() {
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
        run(&mut heat, 2.0, 400);

        // In steady state the flux is uniform: D_left * grad_left = D_right * grad_right.
        // Sample gradients well away from the interface and the walls.
        let row = 2;
        let gradient_between = |i0: usize, i1: usize| {
            (heat.field().get(i1, row) - heat.field().get(i0, row))
                / (grid.cell_center(i1, row)[0] - grid.cell_center(i0, row)[0])
        };
        let left_gradient = gradient_between(4, 14);
        let right_gradient = gradient_between(26, 36);

        let flux_left = d_left * left_gradient;
        let flux_right = d_right * right_gradient;
        assert!(
            (flux_left - flux_right).abs() / flux_left.abs() < 1e-6,
            "flux must be continuous: {flux_left} vs {flux_right}"
        );
        // The steeper gradient is on the less conductive side, by exactly the
        // conductivity ratio.
        let ratio = right_gradient / left_gradient;
        assert!((ratio - d_left / d_right).abs() < 1e-3, "gradient ratio {ratio}, expected 4");
    }

    /// Exceeding the explicit stability limit must produce a visible failure, not a
    /// quietly wrong answer. NFR-007: *"the engine must expose numerical instability
    /// rather than silently clamp or hide it."*
    #[test]
    fn exceeding_the_explicit_limit_blows_up_visibly() {
        let grid = Grid2d::new(32, 32, [1.0, 1.0]);
        let d = 0.01;
        let limit = DiffusionOperator::new(&grid, &Diffusivity::Uniform(d))
            .explicit_stability_limit()
            .max;

        let mut heat = HeatDomain::new("u", grid, Diffusivity::Uniform(d))
            .with_scheme(TimeScheme::Explicit)
            .with_boundaries(BoundarySet::PERIODIC)
            .with_initial(gaussian([0.5, 0.5], 0.002, 1.0));

        let start = heat.field().max_abs_interior();
        // 20% past the limit is enough for the shortest mode to run away.
        run(&mut heat, 1.2 * limit, 200);
        let end = heat.field().max_abs_interior();

        assert!(end > 1e3 * start, "the run should be obviously diverging: {start} -> {end}");
        // And the stability report says why, before it happens.
        let reported = heat.stable_step();
        assert_eq!(reported.reason, lattice_ir::StabilityReason::DiffusionExplicit);
        assert!(reported.margin(1.2 * limit) > 1.0, "the margin must flag the over-large step");
    }

    /// Robin boundaries relax toward the ambient value at a rate the coefficient sets.
    #[test]
    fn robin_boundaries_relax_toward_the_ambient_value() {
        let grid = Grid2d::new(24, 24, [1.0, 1.0]);
        let mut heat = HeatDomain::new("T", grid, Diffusivity::Uniform(0.05))
            .with_scheme(TimeScheme::BackwardEuler)
            .with_boundaries(BoundarySet::uniform(Boundary::Robin {
                coefficient: 5.0,
                ambient: 300.0,
            }))
            .with_tolerance(1e-13)
            .with_uniform_initial(400.0);

        let start = heat.field().max_interior();
        run(&mut heat, 1.0, 300);
        let end = heat.field().max_interior();

        assert!(end < start, "the body must cool toward the ambient");
        assert!((end - 300.0).abs() < 1e-3, "it should reach the ambient, got {end}");
        assert!(end > 299.0, "and must not overshoot below it");
    }

    /// Every solver contract must be complete before the module ships (P1, NFR-003).
    #[test]
    fn every_scheme_publishes_a_complete_contract() {
        let grid = Grid2d::new(8, 8, [1.0, 1.0]);
        for scheme in [TimeScheme::Explicit, TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
            let heat = HeatDomain::new("u", grid, Diffusivity::Uniform(1.0)).with_scheme(scheme);
            let contract = heat.contract();
            assert!(contract.audit().is_empty(), "{}: {:?}", contract.name, contract.audit());
            assert!(
                !contract.known_non_conservation.is_empty(),
                "{} claims exact conservation of everything",
                contract.name
            );
        }
    }

    #[test]
    fn observations_report_the_integral_and_solver_state() {
        let grid = Grid2d::new(16, 16, [1.0, 1.0]);
        let mut heat = HeatDomain::new("temperature", grid, Diffusivity::Uniform(1e-3))
            .with_boundaries(BoundarySet::INSULATED)
            .with_uniform_initial(300.0);
        run(&mut heat, 1e-2, 5);

        let mut obs = lattice_ir::Observations::new();
        heat.observe(&mut obs);
        assert!((obs.value("temperature.integral").unwrap() - 300.0).abs() < 1e-9);
        assert_eq!(obs.value("temperature.min"), Some(300.0));
        assert!(obs.value("temperature.solver_iterations").is_some());
        assert!(obs.first_non_finite().is_none());
        assert!(heat.last_solve().unwrap().is_converged());
    }

    /// The set of rows a domain publishes has to depend only on how it is configured,
    /// never on how far it has run. A table that gains two rows after the first step and
    /// loses them again on a reset reads as a fault, and there isn't one.
    #[test]
    fn the_published_rows_do_not_depend_on_how_far_the_run_has_got() {
        for scheme in [TimeScheme::Explicit, TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
            let grid = Grid2d::new(16, 16, [1.0, 1.0]);
            let mut domain = HeatDomain::new("plate", grid, Diffusivity::Uniform(1e-4))
                .with_scheme(scheme)
                .with_uniform_initial(300.0);

            let mut fresh = lattice_ir::Observations::new();
            domain.observe(&mut fresh);

            run(&mut domain, 1e-2, 5);
            let mut stepped = lattice_ir::Observations::new();
            domain.observe(&mut stepped);

            let before: Vec<&str> = fresh.iter().map(|o| o.name.as_ref()).collect();
            let after: Vec<&str> = stepped.iter().map(|o| o.name.as_ref()).collect();
            assert_eq!(before, after, "{scheme:?} changed its rows by running");

            // And an explicit scheme reports no solver at all, rather than a solver that
            // did zero iterations — there is no linear solve in a forward Euler step.
            assert_eq!(
                stepped.value("plate.solver_iterations").is_some(),
                scheme.is_implicit(),
                "{scheme:?}"
            );
        }
    }

    #[test]
    fn heated_edge_helper_builds_the_expected_boundary_set() {
        let bs = heated_edge(Side::Top, 500.0);
        assert_eq!(bs.top, Boundary::fixed(500.0));
        assert_eq!(bs.bottom, Boundary::INSULATED);
        assert!(!bs.is_closed());
    }

    #[test]
    fn left_half_helper_splits_the_domain() {
        let grid = Grid2d::new(10, 2, [1.0, 0.2]);
        let f = left_half(&grid, 2.0);
        assert_eq!(f(grid.cell_center(0, 0)), 2.0);
        assert_eq!(f(grid.cell_center(9, 0)), 0.0);
    }

    #[test]
    #[should_panic(expected = "periodicity must be declared on both edges")]
    fn unpaired_periodicity_is_rejected_at_construction() {
        let grid = Grid2d::new(8, 8, [1.0, 1.0]);
        let mut bs = BoundarySet::INSULATED;
        bs.left = Boundary::Periodic;
        let _ = HeatDomain::new("u", grid, Diffusivity::Uniform(1.0)).with_boundaries(bs);
    }
}
