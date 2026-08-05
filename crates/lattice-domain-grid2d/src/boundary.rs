//! Applying boundary conditions by writing the halo.
//!
//! Every scheme in this crate reads a 5-point stencil over the interior. Filling the
//! halo first means the stencil never needs a bounds test, and boundary handling
//! lives in exactly one place instead of being smeared through every kernel.
//!
//! # Homogeneous vs inhomogeneous
//!
//! An implicit solve needs the operator `A` to be *linear*: `A(u + v) = Au + Av`. A
//! Dirichlet condition `u_ghost = 2V − u_last` is affine, not linear — the `2V` is a
//! constant. So it is split:
//!
//! - The **right-hand side** is assembled with the full, inhomogeneous conditions.
//! - Conjugate gradient applies the operator with **homogeneous** conditions
//!   (`V = 0`, `g = 0`, `ambient = 0`), which is the linear part.
//!
//! Getting this wrong is subtle and expensive: the solve converges, the answer looks
//! plausible, and the boundary values are quietly wrong. [`HaloMode`] makes the
//! choice explicit at every call site.

use lattice_ir::{Boundary, BoundarySet, ScalarField, Side};

/// Whether to apply the full boundary condition or only its linear part.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HaloMode {
    /// The condition as declared, including prescribed values, fluxes and ambients.
    ///
    /// Used when assembling explicit updates and implicit right-hand sides.
    Inhomogeneous,
    /// The condition with every prescribed constant set to zero.
    ///
    /// Used inside an iterative solver, where the operator must be linear.
    Homogeneous,
}

/// Fill a field's halo so that the declared boundary conditions hold at the faces.
///
/// The halo must be at least one cell wide. Wider halos have only their innermost
/// ring written, which is all the 5-point stencils here read.
pub fn apply_boundaries(
    field: &mut ScalarField,
    boundaries: &BoundarySet,
    dx: f64,
    dy: f64,
    mode: HaloMode,
) {
    debug_assert!(field.halo() >= 1, "boundary application needs at least one halo cell");
    let (nx, ny) = (field.nx() as isize, field.ny() as isize);

    for side in Side::ALL {
        let condition = boundaries.get(side);
        // Spacing along the outward normal of this side.
        let spacing = match side {
            Side::Left | Side::Right => dx,
            Side::Bottom | Side::Top => dy,
        };

        match side {
            Side::Left | Side::Right => {
                let (ghost_i, edge_i, opposite_i) = match side {
                    Side::Left => (-1isize, 0isize, nx - 1),
                    _ => (nx, nx - 1, 0isize),
                };
                for j in 0..ny {
                    let ghost = field.index_signed(ghost_i, j);
                    let edge = field.index_signed(edge_i, j);
                    let opposite = field.index_signed(opposite_i, j);
                    let value = ghost_value(condition, mode, field, edge, opposite, spacing);
                    field.as_mut_slice()[ghost] = value;
                }
            }
            Side::Bottom | Side::Top => {
                let (ghost_j, edge_j, opposite_j) = match side {
                    Side::Bottom => (-1isize, 0isize, ny - 1),
                    _ => (ny, ny - 1, 0isize),
                };
                for i in 0..nx {
                    let ghost = field.index_signed(i, ghost_j);
                    let edge = field.index_signed(i, edge_j);
                    let opposite = field.index_signed(i, opposite_j);
                    let value = ghost_value(condition, mode, field, edge, opposite, spacing);
                    field.as_mut_slice()[ghost] = value;
                }
            }
        }
    }

    // Corners are read by nothing in a 5-point stencil, but leaving them stale makes
    // any future 9-point scheme or visualization sample garbage. Fill them by
    // averaging the two adjacent edge ghosts.
    fill_corners(field);
}

/// The value a ghost cell must take for the declared condition to hold at the face
/// between `edge` and the ghost.
///
/// All four conditions are derived from the same picture: the boundary face sits
/// halfway between the last interior cell and the ghost, so the face value is their
/// average and the outward derivative is their difference over the spacing.
#[inline]
fn ghost_value(
    condition: Boundary,
    mode: HaloMode,
    field: &ScalarField,
    edge: usize,
    opposite: usize,
    spacing: f64,
) -> f64 {
    let data = field.as_slice();
    match condition {
        // Periodicity is purely linear, so the mode does not affect it.
        Boundary::Periodic => data[opposite],

        // Face value V = (u_edge + u_ghost)/2  =>  u_ghost = 2V - u_edge.
        Boundary::Dirichlet { value } => {
            let v = if mode == HaloMode::Homogeneous { 0.0 } else { value };
            2.0 * v - data[edge]
        }

        // Outward derivative g = (u_ghost - u_edge)/spacing  =>  u_ghost = u_edge + g*h.
        Boundary::Neumann { gradient } => {
            let g = if mode == HaloMode::Homogeneous { 0.0 } else { gradient };
            data[edge] + g * spacing
        }

        // -du/dn = c(u_face - ambient), with u_face = (u_edge + u_ghost)/2 and
        // du/dn = (u_ghost - u_edge)/h. Solving for u_ghost:
        //     u_ghost = [u_edge(1 - c*h/2) + c*h*ambient] / (1 + c*h/2)
        Boundary::Robin { coefficient, ambient } => {
            let a = if mode == HaloMode::Homogeneous { 0.0 } else { ambient };
            let half = 0.5 * coefficient * spacing;
            (data[edge] * (1.0 - half) + coefficient * spacing * a) / (1.0 + half)
        }
    }
}

/// Give corner ghosts the mean of their two neighbouring edge ghosts.
fn fill_corners(field: &mut ScalarField) {
    let (nx, ny) = (field.nx() as isize, field.ny() as isize);
    let corners = [
        ((-1isize, -1isize), (0isize, -1isize), (-1isize, 0isize)),
        ((nx, -1), (nx - 1, -1), (nx, 0)),
        ((-1, ny), (0, ny), (-1, ny - 1)),
        ((nx, ny), (nx - 1, ny), (nx, ny - 1)),
    ];
    for ((ci, cj), (ai, aj), (bi, bj)) in corners {
        let corner = field.index_signed(ci, cj);
        let a = field.index_signed(ai, aj);
        let b = field.index_signed(bi, bj);
        let data = field.as_mut_slice();
        data[corner] = 0.5 * (data[a] + data[b]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::Grid2d;

    fn field_with(nx: usize, ny: usize, f: impl Fn(usize, usize) -> f64) -> (Grid2d, ScalarField) {
        let grid = Grid2d::new(nx, ny, [nx as f64, ny as f64]);
        let mut field = ScalarField::new(&grid, 1);
        for j in 0..ny {
            for i in 0..nx {
                field.set(i, j, f(i, j));
            }
        }
        (grid, field)
    }

    #[test]
    fn periodic_halo_mirrors_the_opposite_edge() {
        let (grid, mut f) = field_with(4, 3, |i, j| (i + 10 * j) as f64);
        apply_boundaries(&mut f, &BoundarySet::PERIODIC, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        for j in 0..3isize {
            assert_eq!(f.as_slice()[f.index_signed(-1, j)], f.get(3, j as usize));
            assert_eq!(f.as_slice()[f.index_signed(4, j)], f.get(0, j as usize));
        }
        for i in 0..4isize {
            assert_eq!(f.as_slice()[f.index_signed(i, -1)], f.get(i as usize, 2));
            assert_eq!(f.as_slice()[f.index_signed(i, 3)], f.get(i as usize, 0));
        }
    }

    /// The defining property of the Dirichlet ghost: the *face* value, which is the
    /// average of the last interior cell and the ghost, equals the prescribed value.
    #[test]
    fn dirichlet_puts_the_prescribed_value_on_the_face() {
        let (grid, mut f) = field_with(4, 4, |_, _| 100.0);
        let bs = BoundarySet::uniform(Boundary::fixed(300.0));
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        for j in 0..4isize {
            let ghost = f.as_slice()[f.index_signed(-1, j)];
            let face = 0.5 * (ghost + f.get(0, j as usize));
            assert!((face - 300.0).abs() < 1e-12, "left face = {face}");
        }
        let ghost = f.as_slice()[f.index_signed(4, 0)];
        assert!((0.5 * (ghost + f.get(3, 0)) - 300.0).abs() < 1e-12);
    }

    #[test]
    fn homogeneous_dirichlet_drops_the_prescribed_value() {
        let (grid, mut f) = field_with(3, 3, |_, _| 50.0);
        let bs = BoundarySet::uniform(Boundary::fixed(300.0));
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Homogeneous);
        // u_ghost = -u_edge, so the face value is zero.
        let ghost = f.as_slice()[f.index_signed(-1, 0)];
        assert!((ghost + 50.0).abs() < 1e-12, "ghost = {ghost}");
    }

    /// A zero-gradient Neumann halo must copy the edge value, which is what makes an
    /// insulated boundary produce no flux.
    #[test]
    fn insulated_neumann_copies_the_edge() {
        let (grid, mut f) = field_with(4, 4, |i, _| i as f64);
        apply_boundaries(&mut f, &BoundarySet::INSULATED, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        assert_eq!(f.as_slice()[f.index_signed(-1, 2)], f.get(0, 2));
        assert_eq!(f.as_slice()[f.index_signed(4, 2)], f.get(3, 2));
    }

    /// Signs follow the *outward* normal, so the same positive gradient produces
    /// opposite ghost offsets on opposite edges. Getting this backwards is the
    /// classic boundary sign error.
    #[test]
    fn neumann_gradient_follows_the_outward_normal() {
        let (grid, mut f) = field_with(4, 1, |_, _| 10.0);
        let bs = BoundarySet::uniform(Boundary::Neumann { gradient: 2.0 });
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let h = grid.dx();
        let left_ghost = f.as_slice()[f.index_signed(-1, 0)];
        let right_ghost = f.as_slice()[f.index_signed(4, 0)];
        // Both ghosts sit *above* their edge cell by g*h, because "outward" points
        // left on the left and right on the right.
        assert!((left_ghost - (10.0 + 2.0 * h)).abs() < 1e-12, "left {left_ghost}");
        assert!((right_ghost - (10.0 + 2.0 * h)).abs() < 1e-12, "right {right_ghost}");
    }

    #[test]
    fn homogeneous_neumann_ignores_the_prescribed_gradient() {
        let (grid, mut f) = field_with(3, 1, |_, _| 7.0);
        let bs = BoundarySet::uniform(Boundary::Neumann { gradient: 5.0 });
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Homogeneous);
        assert_eq!(f.as_slice()[f.index_signed(-1, 0)], 7.0);
    }

    /// Robin interpolates between its two limits: `c → 0` is insulating, `c → ∞`
    /// pins the face to the ambient value.
    #[test]
    fn robin_spans_neumann_and_dirichlet() {
        let (grid, _) = field_with(4, 1, |_, _| 0.0);
        let h = grid.dx();

        let ghost_for = |c: f64| {
            let (_, mut f) = field_with(4, 1, |_, _| 100.0);
            let bs = BoundarySet::uniform(Boundary::Robin { coefficient: c, ambient: 0.0 });
            apply_boundaries(&mut f, &bs, h, grid.dy(), HaloMode::Inhomogeneous);
            f.as_slice()[f.index_signed(-1, 0)]
        };

        // c = 0 is exactly insulating.
        assert!((ghost_for(0.0) - 100.0).abs() < 1e-12);
        // Very large c drives the face to the ambient value of 0, so the ghost
        // approaches -u_edge.
        let stiff = ghost_for(1e12);
        assert!((stiff + 100.0).abs() < 1e-3, "stiff Robin ghost = {stiff}");
        // In between, the ghost lies between the two.
        let middle = ghost_for(1.0);
        assert!(middle < 100.0 && middle > -100.0, "middle = {middle}");
    }

    #[test]
    fn robin_with_ambient_pulls_toward_the_reservoir() {
        let (grid, mut f) = field_with(4, 1, |_, _| 0.0);
        let bs = BoundarySet::uniform(Boundary::Robin { coefficient: 1e9, ambient: 273.15 });
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        let ghost = f.as_slice()[f.index_signed(-1, 0)];
        let face = 0.5 * (ghost + f.get(0, 0));
        assert!((face - 273.15).abs() < 1e-3, "face = {face}");
    }

    #[test]
    fn mixed_boundaries_apply_independently_per_side() {
        let (grid, mut f) = field_with(4, 4, |_, _| 20.0);
        let bs = BoundarySet {
            left: Boundary::fixed(0.0),
            right: Boundary::INSULATED,
            bottom: Boundary::Neumann { gradient: 1.0 },
            top: Boundary::fixed(100.0),
        };
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        assert!((f.as_slice()[f.index_signed(-1, 0)] + 20.0).abs() < 1e-12, "left Dirichlet 0");
        assert_eq!(f.as_slice()[f.index_signed(4, 0)], 20.0, "right insulated");
        let top_ghost = f.as_slice()[f.index_signed(0, 4)];
        assert!((0.5 * (top_ghost + 20.0) - 100.0).abs() < 1e-12, "top Dirichlet 100");
    }

    #[test]
    fn corners_are_filled_rather_than_left_stale() {
        let grid = Grid2d::new(3, 3, [3.0, 3.0]);
        let mut f = ScalarField::filled(&grid, 1, f64::NAN);
        f.fill_interior(5.0);
        apply_boundaries(&mut f, &BoundarySet::INSULATED, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        for (i, j) in [(-1isize, -1isize), (3, -1), (-1, 3), (3, 3)] {
            let v = f.as_slice()[f.index_signed(i, j)];
            assert!(v.is_finite(), "corner ({i},{j}) left as {v}");
            assert!((v - 5.0).abs() < 1e-12);
        }
    }

    #[test]
    fn applying_twice_is_idempotent() {
        let (grid, mut f) = field_with(5, 5, |i, j| (i * j) as f64);
        let bs = BoundarySet {
            left: Boundary::fixed(1.0),
            right: Boundary::Robin { coefficient: 0.5, ambient: 2.0 },
            bottom: Boundary::Periodic,
            top: Boundary::Periodic,
        };
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        let first = f.as_slice().to_vec();
        apply_boundaries(&mut f, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        assert_eq!(first, f.as_slice(), "halo values must not depend on prior halo contents");
    }
}
