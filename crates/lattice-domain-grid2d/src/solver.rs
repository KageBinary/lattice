//! Conjugate gradient for the symmetric positive-definite systems implicit schemes
//! produce.
//!
//! Spec §10.3 lists *"conjugate gradient for symmetric positive-definite systems"*
//! first, and adds that *"residual histories are always observable and can stop a run
//! on divergence."* Both are here: the solve is matrix-free (the operator is applied
//! as a stencil, never assembled), and every iteration's residual norm is recorded.
//!
//! # Why the system is SPD
//!
//! The implicit update solves `(I − θ·dt·L)u = b`. The discrete `L` built by
//! [`crate::DiffusionOperator`] is symmetric — cell `a` influences cell `b` through
//! exactly the same face coefficient that `b` influences `a` — and negative
//! semi-definite. Subtracting a negative semi-definite operator from the identity
//! gives a positive-definite one for any `θ·dt > 0`, so CG applies and converges.
//!
//! If the operator ever stops being SPD (a negative diffusivity, a bug in the face
//! coefficients), CG does not fail loudly on its own: it wanders. So the curvature
//! `pᵀAp` is checked every iteration and a non-positive value is reported as
//! divergence rather than pressed on with.

use lattice_ir::{Grid2d, ResidualHistory, ScalarField, SolveOutcome};

/// The three vectors conjugate gradient needs, allocated once.
#[derive(Clone, Debug)]
pub struct CgWorkspace {
    residual: ScalarField,
    direction: ScalarField,
    a_direction: ScalarField,
}

impl CgWorkspace {
    /// Allocate a workspace for a grid.
    pub fn new(grid: &Grid2d, halo: usize) -> Self {
        Self {
            residual: ScalarField::new(grid, halo),
            direction: ScalarField::new(grid, halo),
            a_direction: ScalarField::new(grid, halo),
        }
    }

    /// Bytes of scratch held, for the memory report.
    pub fn bytes(&self) -> usize {
        3 * self.residual.len() * core::mem::size_of::<f64>()
    }
}

/// Solve `A·x = b` by conjugate gradient.
///
/// `apply` computes `output = A·input`. It receives `input` mutably so it can fill
/// that field's halo with *homogeneous* boundary conditions before applying the
/// stencil — see [`crate::HaloMode`] for why homogeneous is the only correct choice
/// inside the iteration.
///
/// `x` is used as the initial guess; passing the previous timestep's solution
/// typically halves the iteration count.
///
/// Convergence is measured as `‖r‖ ≤ tolerance · ‖b‖`, a relative criterion, so the
/// same tolerance means the same thing across grid sizes and field magnitudes.
pub fn conjugate_gradient<A>(
    x: &mut ScalarField,
    b: &ScalarField,
    mut apply: A,
    workspace: &mut CgWorkspace,
    tolerance: f64,
    max_iterations: usize,
    history: &mut ResidualHistory,
) -> SolveOutcome
where
    A: FnMut(&mut ScalarField, &mut ScalarField),
{
    let CgWorkspace { residual, direction, a_direction } = workspace;
    history.clear();

    let b_norm = interior_norm(b);
    if b_norm == 0.0 {
        // The zero vector solves it exactly; iterating would divide by zero.
        x.fill_interior(0.0);
        history.push(0.0);
        return SolveOutcome::Converged { iterations: 0, residual: 0.0 };
    }
    let target = tolerance * b_norm;

    // r = b - A x
    apply(x, a_direction);
    interior_sub_into(residual, b, a_direction);
    direction.copy_interior_from(residual);

    let mut rs_old = interior_dot(residual, residual);
    let mut r_norm = rs_old.sqrt();
    history.push(r_norm);
    if r_norm <= target {
        return SolveOutcome::Converged { iterations: 0, residual: r_norm };
    }

    for iteration in 1..=max_iterations {
        apply(direction, a_direction);
        let curvature = interior_dot(direction, a_direction);

        // For an SPD operator this is strictly positive. Anything else means the
        // operator is not what CG assumes, and continuing would produce a confident
        // wrong answer (NFR-007).
        if !curvature.is_finite() || curvature <= 0.0 {
            return SolveOutcome::Diverged { iterations: iteration - 1, residual: r_norm };
        }

        let alpha = rs_old / curvature;
        interior_axpy(x, alpha, direction);
        interior_axpy(residual, -alpha, a_direction);

        let rs_new = interior_dot(residual, residual);
        r_norm = rs_new.sqrt();
        history.push(r_norm);

        if !r_norm.is_finite() {
            return SolveOutcome::Diverged { iterations: iteration, residual: r_norm };
        }
        if r_norm <= target {
            return SolveOutcome::Converged { iterations: iteration, residual: r_norm };
        }

        // p = r + (rs_new / rs_old) p
        interior_xpby(direction, residual, rs_new / rs_old);
        rs_old = rs_new;
    }

    SolveOutcome::MaxIterations { iterations: max_iterations, residual: r_norm, tolerance: target }
}

/// Dot product over interior cells only. Halo values are boundary bookkeeping, not
/// degrees of freedom, and including them would corrupt every norm.
fn interior_dot(a: &ScalarField, b: &ScalarField) -> f64 {
    let mut sum = 0.0;
    for j in 0..a.ny() {
        let (ra, rb) = (a.row(j), b.row(j));
        for i in 0..ra.len() {
            sum += ra[i] * rb[i];
        }
    }
    sum
}

fn interior_norm(a: &ScalarField) -> f64 {
    interior_dot(a, a).sqrt()
}

/// `out = a − b` over the interior.
fn interior_sub_into(out: &mut ScalarField, a: &ScalarField, b: &ScalarField) {
    let nx = out.nx();
    for j in 0..out.ny() {
        let start = out.index(0, j);
        let (ra, rb) = (a.row(j), b.row(j));
        let dst = &mut out.as_mut_slice()[start..start + nx];
        for i in 0..nx {
            dst[i] = ra[i] - rb[i];
        }
    }
}

/// `y += alpha·x` over the interior.
fn interior_axpy(y: &mut ScalarField, alpha: f64, x: &ScalarField) {
    let nx = y.nx();
    for j in 0..y.ny() {
        let start = y.index(0, j);
        let xs = x.row(j);
        let ys = &mut y.as_mut_slice()[start..start + nx];
        for i in 0..nx {
            ys[i] += alpha * xs[i];
        }
    }
}

/// `y = x + beta·y` over the interior.
fn interior_xpby(y: &mut ScalarField, x: &ScalarField, beta: f64) {
    let nx = y.nx();
    for j in 0..y.ny() {
        let start = y.index(0, j);
        let xs = x.row(j);
        let ys = &mut y.as_mut_slice()[start..start + nx];
        for i in 0..nx {
            ys[i] = xs[i] + beta * ys[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Grid2d {
        Grid2d::new(n, n, [1.0, 1.0])
    }

    /// The identity operator: CG must converge in a single iteration.
    #[test]
    fn identity_converges_immediately() {
        let g = grid(8);
        let mut x = ScalarField::new(&g, 1);
        let mut b = ScalarField::new(&g, 1);
        b.init_from_position(&g, |[px, py]| px + py);

        let mut ws = CgWorkspace::new(&g, 1);
        let mut history = ResidualHistory::default();
        let outcome = conjugate_gradient(
            &mut x,
            &b,
            |input, output| output.copy_interior_from(input),
            &mut ws,
            1e-12,
            50,
            &mut history,
        );

        assert!(outcome.is_converged(), "{outcome}");
        assert!(outcome.iterations() <= 1, "identity should need at most one step");
        for j in 0..8 {
            for i in 0..8 {
                assert!((x.get(i, j) - b.get(i, j)).abs() < 1e-10);
            }
        }
    }

    /// A diagonal operator with distinct entries. CG on an n-dimensional SPD system
    /// converges in at most n steps in exact arithmetic; here the number of *distinct
    /// eigenvalues* bounds it, which is a sharper and more interesting check.
    #[test]
    fn diagonal_operator_converges_within_the_eigenvalue_count() {
        let g = grid(6);
        let mut diagonal = ScalarField::new(&g, 1);
        // Only three distinct values, so CG must finish in at most three iterations.
        diagonal.init_from_position(&g, |[px, _]| if px < 0.34 { 1.0 } else if px < 0.67 { 4.0 } else { 9.0 });

        let mut b = ScalarField::new(&g, 1);
        b.fill_interior(1.0);
        let mut x = ScalarField::new(&g, 1);
        let mut ws = CgWorkspace::new(&g, 1);
        let mut history = ResidualHistory::default();

        let outcome = conjugate_gradient(
            &mut x,
            &b,
            |input, output| {
                for j in 0..input.ny() {
                    for i in 0..input.nx() {
                        output.set(i, j, diagonal.get(i, j) * input.get(i, j));
                    }
                }
            },
            &mut ws,
            1e-12,
            50,
            &mut history,
        );

        assert!(outcome.is_converged(), "{outcome}");
        assert!(outcome.iterations() <= 3, "took {} iterations", outcome.iterations());
        // x = b / diagonal.
        for j in 0..6 {
            for i in 0..6 {
                let expected = 1.0 / diagonal.get(i, j);
                assert!((x.get(i, j) - expected).abs() < 1e-9, "({i},{j})");
            }
        }
    }

    #[test]
    fn a_zero_right_hand_side_gives_the_zero_solution() {
        let g = grid(4);
        let mut x = ScalarField::new(&g, 1);
        x.fill_interior(99.0);
        let b = ScalarField::new(&g, 1);
        let mut ws = CgWorkspace::new(&g, 1);
        let mut history = ResidualHistory::default();

        let outcome = conjugate_gradient(
            &mut x,
            &b,
            |input, output| output.copy_interior_from(input),
            &mut ws,
            1e-12,
            10,
            &mut history,
        );
        assert!(outcome.is_converged());
        assert_eq!(x.max_abs_interior(), 0.0);
    }

    #[test]
    fn residual_history_is_recorded_and_monotone_enough() {
        let g = grid(16);
        let mut diagonal = ScalarField::new(&g, 1);
        diagonal.init_from_position(&g, |[px, py]| 1.0 + 10.0 * px + 5.0 * py);
        let mut b = ScalarField::new(&g, 1);
        b.init_from_position(&g, |[px, py]| (px * 7.0).sin() + py);
        let mut x = ScalarField::new(&g, 1);
        let mut ws = CgWorkspace::new(&g, 1);
        let mut history = ResidualHistory::default();

        let outcome = conjugate_gradient(
            &mut x,
            &b,
            |input, output| {
                for j in 0..input.ny() {
                    for i in 0..input.nx() {
                        output.set(i, j, diagonal.get(i, j) * input.get(i, j));
                    }
                }
            },
            &mut ws,
            1e-10,
            100,
            &mut history,
        );

        assert!(outcome.is_converged(), "{outcome}");
        assert_eq!(history.iterations(), outcome.iterations());
        assert!(!history.is_diverging());
        assert!(history.reduction() < 1e-9, "reduction {}", history.reduction());
        assert!(history.initial().unwrap() > history.latest().unwrap());
    }

    /// A non-SPD operator must be reported, not silently iterated on. Negating the
    /// identity makes the curvature negative on the first iteration.
    #[test]
    fn a_non_spd_operator_is_reported_as_divergence() {
        let g = grid(4);
        let mut x = ScalarField::new(&g, 1);
        let mut b = ScalarField::new(&g, 1);
        b.fill_interior(1.0);
        let mut ws = CgWorkspace::new(&g, 1);
        let mut history = ResidualHistory::default();

        let outcome = conjugate_gradient(
            &mut x,
            &b,
            |input, output| {
                output.copy_interior_from(input);
                for j in 0..output.ny() {
                    for i in 0..output.nx() {
                        let v = output.get(i, j);
                        output.set(i, j, -v);
                    }
                }
            },
            &mut ws,
            1e-12,
            10,
            &mut history,
        );
        assert!(matches!(outcome, SolveOutcome::Diverged { .. }), "{outcome}");
    }

    #[test]
    fn hitting_the_iteration_cap_is_reported_not_hidden() {
        let g = grid(8);
        let mut diagonal = ScalarField::new(&g, 1);
        // A viciously ill-conditioned diagonal so CG cannot finish in two steps.
        diagonal.init_from_position(&g, |[px, py]| 1e-8 + px * px * 1e8 + py);
        let mut b = ScalarField::new(&g, 1);
        b.fill_interior(1.0);
        let mut x = ScalarField::new(&g, 1);
        let mut ws = CgWorkspace::new(&g, 1);
        let mut history = ResidualHistory::default();

        let outcome = conjugate_gradient(
            &mut x,
            &b,
            |input, output| {
                for j in 0..input.ny() {
                    for i in 0..input.nx() {
                        output.set(i, j, diagonal.get(i, j) * input.get(i, j));
                    }
                }
            },
            &mut ws,
            1e-16,
            2,
            &mut history,
        );
        assert!(matches!(outcome, SolveOutcome::MaxIterations { iterations: 2, .. }), "{outcome}");
        assert!(!outcome.is_converged());
    }

    #[test]
    fn a_good_initial_guess_reduces_the_iteration_count() {
        let g = grid(24);
        let mut diagonal = ScalarField::new(&g, 1);
        diagonal.init_from_position(&g, |[px, py]| 1.0 + 20.0 * px + 7.0 * py);
        let mut b = ScalarField::new(&g, 1);
        b.init_from_position(&g, |[px, py]| px * py + 1.0);

        let solve = |initial: &ScalarField| {
            let mut x = initial.clone();
            let mut ws = CgWorkspace::new(&g, 1);
            let mut history = ResidualHistory::default();
            let outcome = conjugate_gradient(
                &mut x,
                &b,
                |input, output| {
                    for j in 0..input.ny() {
                        for i in 0..input.nx() {
                            output.set(i, j, diagonal.get(i, j) * input.get(i, j));
                        }
                    }
                },
                &mut ws,
                1e-10,
                200,
                &mut history,
            );
            (outcome.iterations(), x)
        };

        let (cold, exact) = solve(&ScalarField::new(&g, 1));
        let (warm, _) = solve(&exact);
        assert!(warm < cold, "warm start took {warm}, cold took {cold}");
    }

    #[test]
    fn interior_helpers_ignore_the_halo() {
        let g = grid(3);
        let mut a = ScalarField::filled(&g, 2, 1000.0);
        let mut b = ScalarField::filled(&g, 2, -1000.0);
        a.fill_interior(2.0);
        b.fill_interior(3.0);

        assert!((interior_dot(&a, &b) - 54.0).abs() < 1e-12, "3x3 cells of 2*3 = 54");

        interior_axpy(&mut a, 2.0, &b);
        assert_eq!(a.get(0, 0), 8.0);
        // Halo untouched.
        assert_eq!(a.as_slice()[a.index_signed(-1, -1)], 1000.0);

        interior_xpby(&mut a, &b, 0.5);
        assert_eq!(a.get(1, 1), 3.0 + 0.5 * 8.0);
    }
}
