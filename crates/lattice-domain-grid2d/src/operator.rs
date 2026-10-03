//! The discrete diffusion operator `∇·(D∇u)`.
//!
//! # Finite volume, not finite difference
//!
//! The stencil is assembled from *face fluxes* rather than by differencing `u` twice:
//!
//! ```text
//!   L[u]_ij = [ D_{i+½,j}(u_{i+1,j} − u_ij) − D_{i−½,j}(u_ij − u_{i−1,j}) ] / dx²
//!           + [ D_{i,j+½}(u_{i,j+1} − u_ij) − D_{i,j−½}(u_ij − u_{i,j−1}) ] / dy²
//! ```
//!
//! Whatever leaves one cell through a face enters its neighbour through the same
//! face, with the same coefficient. Interior fluxes therefore cancel exactly in the
//! sum over all cells, and `∫u dA` is conserved *to round-off* on a closed domain —
//! not merely to truncation error. For constant `D` this reduces to the familiar
//! 5-point stencil, so nothing is lost in the common case.
//!
//! # Harmonic means at faces
//!
//! Face diffusivities use the harmonic mean `2D₁D₂/(D₁+D₂)` rather than the
//! arithmetic mean. For layered media the two cells act like resistors in series, and
//! the harmonic mean is the exact effective conductivity of that pair. The arithmetic
//! mean over-predicts transport across a sharp contrast — a well-known way to get
//! plausible-looking but wrong answers at material interfaces.

use lattice_ir::{Executor, Grain, Grid2d, ScalarField, StabilityReason, StableStep};

/// How much of a grid is worth splitting, in cells.
///
/// A grid below 16k cells — a 128² model, say — runs its whole stencil in a few tens of
/// microseconds, and a dispatch barrier costs a few of those on its own. Below the floor
/// the partition collapses to one band and the model runs exactly as it did before M4.
/// Above it, bands are at least 4k cells each so a 160×110 grid is not cut into slivers.
///
/// Stated in cells rather than rows because a row is not a fixed amount of work: ten
/// rows of a 512-wide grid and ten of a 32-wide one differ by a factor of sixteen.
/// [`Grain::per_row`] converts.
///
/// Measured on a 20-thread machine with `lattice bench heat-explicit --threads auto
/// --compare`, best of three release runs: 3.4x at 256², 4.8x at 512², 4.7x at 1024².
/// The stencil is memory-bound — six loads and a handful of flops per cell — so those
/// are a long way below linear and are expected to be. The flattening past 512² is the
/// working set outgrowing cache, not the partition failing.
/// Public so a test can ask whether a given grid would actually be split. A
/// cross-backend case run on a grid below the floor compares the sequential path with
/// itself and passes for the wrong reason.
pub const BAND_GRAIN: Grain = Grain::new(16_384, 4_096);

/// How the diffusion coefficient varies over the grid.
#[derive(Clone, Debug)]
pub enum Diffusivity {
    /// One value everywhere, m²/s.
    Uniform(f64),
    /// A cell-centred field of values, m²/s.
    Variable(ScalarField),
}

impl Diffusivity {
    /// Value at an interior cell.
    pub fn at(&self, i: usize, j: usize) -> f64 {
        match self {
            Diffusivity::Uniform(d) => *d,
            Diffusivity::Variable(field) => field.get(i, j),
        }
    }

    /// Largest value anywhere, which is what sets the explicit stability limit.
    pub fn maximum(&self) -> f64 {
        match self {
            Diffusivity::Uniform(d) => *d,
            Diffusivity::Variable(field) => field.max_interior(),
        }
    }

    /// True if every value is finite and non-negative.
    ///
    /// A negative diffusivity makes the heat equation ill-posed — it runs the
    /// diffusion backwards, which amplifies every high-frequency mode without bound.
    /// Rejecting it at setup is far kinder than letting the field explode later.
    pub fn is_physical(&self) -> bool {
        match self {
            Diffusivity::Uniform(d) => d.is_finite() && *d >= 0.0,
            Diffusivity::Variable(field) => {
                field.first_non_finite().is_none() && field.min_interior() >= 0.0
            }
        }
    }
}

/// Precomputed face diffusivities and the stencil that uses them.
#[derive(Clone, Debug)]
pub struct DiffusionOperator {
    nx: usize,
    ny: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    /// `(nx+1) × ny` diffusivities on x-normal faces.
    face_x: Vec<f64>,
    /// `nx × (ny+1)` diffusivities on y-normal faces.
    face_y: Vec<f64>,
    max_diffusivity: f64,
}

impl DiffusionOperator {
    /// Build the operator for a grid and diffusivity distribution.
    ///
    /// # Panics
    ///
    /// If the diffusivity is negative or non-finite anywhere.
    pub fn new(grid: &Grid2d, diffusivity: &Diffusivity) -> Self {
        assert!(
            diffusivity.is_physical(),
            "diffusivity must be finite and non-negative everywhere; \
             a negative value makes the heat equation ill-posed"
        );
        let (nx, ny) = (grid.nx(), grid.ny());
        let mut op = Self {
            nx,
            ny,
            inv_dx2: 1.0 / (grid.dx() * grid.dx()),
            inv_dy2: 1.0 / (grid.dy() * grid.dy()),
            face_x: vec![0.0; (nx + 1) * ny],
            face_y: vec![0.0; nx * (ny + 1)],
            max_diffusivity: 0.0,
        };
        op.recompute_faces(diffusivity);
        op
    }

    /// Recompute face values after the diffusivity changes.
    ///
    /// Temperature-dependent conductivity (spec §11.4) makes this a per-step
    /// operation, so it is separated from construction.
    pub fn recompute_faces(&mut self, diffusivity: &Diffusivity) {
        let (nx, ny) = (self.nx, self.ny);

        for j in 0..ny {
            for i in 0..=nx {
                // At the domain edge there is no second cell, so the face takes the
                // adjacent interior value — a zero-gradient extrapolation of D. The
                // boundary condition on `u` is applied separately, through the halo.
                let value = if i == 0 {
                    diffusivity.at(0, j)
                } else if i == nx {
                    diffusivity.at(nx - 1, j)
                } else {
                    harmonic_mean(diffusivity.at(i - 1, j), diffusivity.at(i, j))
                };
                self.face_x[j * (nx + 1) + i] = value;
            }
        }

        for j in 0..=ny {
            for i in 0..nx {
                let value = if j == 0 {
                    diffusivity.at(i, 0)
                } else if j == ny {
                    diffusivity.at(i, ny - 1)
                } else {
                    harmonic_mean(diffusivity.at(i, j - 1), diffusivity.at(i, j))
                };
                self.face_y[j * nx + i] = value;
            }
        }

        self.max_diffusivity = diffusivity.maximum();
    }

    /// Largest diffusivity in the domain, m²/s.
    pub fn max_diffusivity(&self) -> f64 {
        self.max_diffusivity
    }

    /// The `(nx+1) × ny` harmonic-mean diffusivities on x-normal faces.
    ///
    /// Exposed so an accelerated backend can upload the coefficients this operator
    /// computed rather than deriving its own. §24.1 makes the scalar CPU path *"the
    /// executable specification for accelerated kernels"*, and a GPU that recomputed the
    /// harmonic means would be a second implementation of a rule the specification
    /// already fixes — with its own opportunity to get the boundary faces wrong, which is
    /// precisely where this operator is easiest to get wrong.
    pub fn face_x(&self) -> &[f64] {
        &self.face_x
    }

    /// The `nx × (ny+1)` harmonic-mean diffusivities on y-normal faces.
    ///
    /// See [`DiffusionOperator::face_x`] for why this is public.
    pub fn face_y(&self) -> &[f64] {
        &self.face_y
    }

    /// `1 / dx²`, the coefficient the x-flux difference is scaled by.
    pub fn inv_dx2(&self) -> f64 {
        self.inv_dx2
    }

    /// `1 / dy²`, the coefficient the y-flux difference is scaled by.
    pub fn inv_dy2(&self) -> f64 {
        self.inv_dy2
    }

    /// Apply `L[u] = ∇·(D∇u)`, writing into `out`'s interior, on the calling thread.
    ///
    /// `u`'s halo must already hold the boundary condition — see
    /// [`crate::apply_boundaries`]. `out`'s halo is left untouched.
    pub fn apply(&self, u: &ScalarField, out: &mut ScalarField) {
        self.apply_with(Executor::shared_sequential(), u, out);
    }

    /// [`DiffusionOperator::apply`], with the row loop split across `executor`.
    ///
    /// Rows are the natural split: cell `(i, j)`'s stencil reads rows `j-1`, `j` and
    /// `j+1` of `u` and writes only row `j` of `out`, so bands of rows write disjoint
    /// memory while reading freely across the boundary between them. Nothing is copied
    /// and no band needs its neighbour's *output*.
    ///
    /// Each output cell is computed by the same expression over the same inputs however
    /// the rows are divided, so this is bit-identical to [`DiffusionOperator::apply`] at
    /// any thread count — the guarantee `lattice_cpu` documents and
    /// `parallel_diffusion_is_bit_identical_to_the_scalar_path` checks.
    pub fn apply_with(&self, executor: &Executor, u: &ScalarField, out: &mut ScalarField) {
        debug_assert_eq!((u.nx(), u.ny()), (self.nx, self.ny), "field size mismatch");
        debug_assert_eq!((out.nx(), out.ny()), (self.nx, self.ny), "output size mismatch");

        let (nx, ny) = (self.nx, self.ny);
        let src = u.as_slice();
        let src_stride = u.stride();
        let src_origin = u.interior_origin();

        let out_stride = out.stride();
        let out_halo = out.halo();
        debug_assert_eq!(out.ny(), ny);

        let grain = BAND_GRAIN.per_row(nx);

        executor.for_each_row_band_mut(out.row_span_mut(), out_stride, grain, |first, band| {
            for local in 0..band.len() / out_stride {
                let j = first + local;
                // Row bases into the source field and the two face arrays.
                let row = src_origin + j * src_stride;
                let fx_row = j * (nx + 1);
                let fy_row = j * nx;
                let fy_next = (j + 1) * nx;
                let dst = local * out_stride + out_halo;

                for i in 0..nx {
                    let k = row + i;
                    let center = src[k];

                    let west = self.face_x[fx_row + i] * (src[k - 1] - center);
                    let east = self.face_x[fx_row + i + 1] * (src[k + 1] - center);
                    let south = self.face_y[fy_row + i] * (src[k - src_stride] - center);
                    let north = self.face_y[fy_next + i] * (src[k + src_stride] - center);

                    band[dst + i] = (east + west) * self.inv_dx2 + (north + south) * self.inv_dy2;
                }
            }
        });
    }

    /// The largest timestep an explicit update remains stable at.
    ///
    /// Von Neumann analysis of the FTCS stencil gives
    /// `dt ≤ 1 / (2·D_max·(1/dx² + 1/dy²))`. Exceeding it does not degrade the answer
    /// gracefully — the shortest-wavelength mode grows by a factor greater than one
    /// every step, and the field is `NaN` within a few dozen steps.
    pub fn explicit_stability_limit(&self) -> StableStep {
        if self.max_diffusivity <= 0.0 {
            // Nothing diffuses, so nothing constrains the step.
            return StableStep::unconditional(f64::INFINITY);
        }
        let max = 1.0 / (2.0 * self.max_diffusivity * (self.inv_dx2 + self.inv_dy2));
        // 80% of the limit: close enough to be efficient, far enough that a small
        // change in D (from a temperature-dependent conductivity, say) does not
        // silently cross the line.
        StableStep::limited(0.8 * max, max, StabilityReason::DiffusionExplicit)
    }
}

/// Effective conductivity of two cells in series.
///
/// Returns zero if either value is zero — an insulator in series with anything is an
/// insulator, and the naive formula would divide by zero.
#[inline]
fn harmonic_mean(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum <= 0.0 { 0.0 } else { 2.0 * a * b / sum }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{apply_boundaries, HaloMode};
    use lattice_ir::{Boundary, BoundarySet};

    fn setup(nx: usize, ny: usize, d: f64) -> (Grid2d, ScalarField, DiffusionOperator) {
        let grid = Grid2d::new(nx, ny, [1.0, 1.0]);
        let field = ScalarField::new(&grid, 1);
        let op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(d));
        (grid, field, op)
    }

    /// The stencil reads `(i±1, j)` and `(i, j±1)` and nothing diagonal, so the four
    /// corner ghost cells cannot affect any output.
    ///
    /// Asserted rather than assumed because an accelerated backend is entitled to rely on
    /// it: `lattice-wgpu`'s halo kernel fills the four edge strips and skips the corners,
    /// which is only correct while this holds. If the operator ever gains a nine-point
    /// form, this test fails and names the backend that has to change with it.
    #[test]
    fn the_five_point_stencil_never_reads_a_corner_ghost() {
        let (grid, mut u, op) = setup(6, 4, 1.5);
        u.init_from_position(&grid, |[x, y]| 300.0 + 10.0 * x + 3.0 * y * y);
        apply_boundaries(&mut u, &BoundarySet::INSULATED, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let mut baseline = ScalarField::new(&grid, 1);
        op.apply(&u, &mut baseline);

        // Poison every corner ghost with a value no correct result could survive.
        let (nx, ny) = (grid.nx() as isize, grid.ny() as isize);
        for (i, j) in [(-1, -1), (nx, -1), (-1, ny), (nx, ny)] {
            let index = u.index_signed(i, j);
            u.as_mut_slice()[index] = 1e9;
        }

        let mut poisoned = ScalarField::new(&grid, 1);
        op.apply(&u, &mut poisoned);

        assert_eq!(
            baseline.as_slice(),
            poisoned.as_slice(),
            "a corner ghost reached the output; lattice-wgpu's halo kernel skips corners"
        );
    }

    /// `L(0) = 0` under insulated boundaries, and *identically* so — not to within a
    /// rounding, but bit for bit, including the sign of the zero.
    ///
    /// This is the affine constant `c` that `HeatDomain::step_implicit` measures and moves
    /// to the right-hand side. Insulated is the one boundary set where it vanishes: a zero
    /// interior gives zero ghosts, so every flux is `face · (0 − 0)`, and face coefficients
    /// are harmonic means and therefore non-negative, so not even a `−0.0` survives to
    /// propagate.
    ///
    /// Asserted rather than assumed because `lattice-wgpu`'s `assemble_rhs` omits the term
    /// entirely. The Dirichlet half of the test is the reason that omission cannot simply be
    /// generalized: there, `c` is the boundary values and dropping it would produce a
    /// converged, plausible, wrong answer.
    #[test]
    fn l_of_zero_is_identically_zero_under_insulated_boundaries() {
        let (grid, mut zero, op) = setup(7, 5, 2.5);
        zero.fill_interior(0.0);
        apply_boundaries(&mut zero, &BoundarySet::INSULATED, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let mut constant = ScalarField::filled(&grid, 1, f64::NAN);
        op.apply(&zero, &mut constant);

        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let value = constant.get(i, j);
                assert_eq!(
                    value.to_bits(),
                    0.0f64.to_bits(),
                    "L(0) at ({i}, {j}) is {value}, not a positive zero"
                );
            }
        }

        // The contrast that makes the claim specific to insulated boundaries. A prescribed
        // value is exactly what an affine constant is, and it does not vanish.
        let mut dirichlet = ScalarField::new(&grid, 1);
        dirichlet.fill_interior(0.0);
        let boundaries = BoundarySet::uniform(Boundary::Dirichlet { value: 400.0 });
        apply_boundaries(&mut dirichlet, &boundaries, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        op.apply(&dirichlet, &mut constant);
        assert!(
            constant.max_abs_interior() > 0.0,
            "a Dirichlet boundary must leave a non-zero affine constant, or step_implicit \
             would not need to measure one"
        );
    }

    #[test]
    fn harmonic_mean_behaves_at_the_extremes() {
        assert!((harmonic_mean(2.0, 2.0) - 2.0).abs() < 1e-15, "equal values pass through");
        assert_eq!(harmonic_mean(0.0, 5.0), 0.0, "an insulator blocks the pair");
        assert_eq!(harmonic_mean(0.0, 0.0), 0.0);
        // Harmonic mean is always at or below the arithmetic mean.
        let (a, b) = (1.0, 9.0);
        assert!(harmonic_mean(a, b) < 0.5 * (a + b));
        assert!((harmonic_mean(a, b) - 1.8).abs() < 1e-12);
    }

    /// For uniform D the operator must reduce exactly to the 5-point stencil.
    #[test]
    fn uniform_diffusivity_gives_the_five_point_stencil() {
        let (grid, mut u, op) = setup(5, 5, 2.0);
        u.fill_interior(0.0);
        u.set(2, 2, 1.0);
        apply_boundaries(&mut u, &BoundarySet::PERIODIC, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let mut out = ScalarField::new(&grid, 1);
        op.apply(&u, &mut out);

        let h2 = grid.dx() * grid.dx();
        // Centre: -4 D / h^2. Neighbours: +D / h^2. Everything else: 0.
        assert!((out.get(2, 2) - (-4.0 * 2.0 / h2)).abs() < 1e-6, "{}", out.get(2, 2));
        for (i, j) in [(1, 2), (3, 2), (2, 1), (2, 3)] {
            assert!((out.get(i, j) - 2.0 / h2).abs() < 1e-6, "({i},{j}) = {}", out.get(i, j));
        }
        assert!(out.get(0, 0).abs() < 1e-12);
    }

    /// A linear field has zero second derivative, so the Laplacian must vanish in the
    /// interior regardless of the boundary treatment.
    #[test]
    fn the_laplacian_of_a_linear_field_is_zero() {
        let grid = Grid2d::new(8, 8, [1.0, 1.0]);
        let mut u = ScalarField::new(&grid, 1);
        u.init_from_position(&grid, |[x, y]| 3.0 * x - 2.0 * y + 7.0);
        // Neumann conditions matching the true gradient keep the halo consistent.
        let bs = BoundarySet {
            left: Boundary::Neumann { gradient: -3.0 },
            right: Boundary::Neumann { gradient: 3.0 },
            bottom: Boundary::Neumann { gradient: 2.0 },
            top: Boundary::Neumann { gradient: -2.0 },
        };
        apply_boundaries(&mut u, &bs, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(1.0));
        let mut out = ScalarField::new(&grid, 1);
        op.apply(&u, &mut out);
        assert!(out.max_abs_interior() < 1e-9, "max |L[u]| = {}", out.max_abs_interior());
    }

    /// The operator must reproduce a known analytic Laplacian to second order.
    #[test]
    fn matches_an_analytic_laplacian_and_converges_at_second_order() {
        // u = sin(2*pi*x) sin(2*pi*y) on the unit square, periodic.
        // Laplacian = -8*pi^2 * u.
        let error_at = |n: usize| {
            let grid = Grid2d::new(n, n, [1.0, 1.0]);
            let mut u = ScalarField::new(&grid, 1);
            let tau = core::f64::consts::TAU;
            u.init_from_position(&grid, |[x, y]| (tau * x).sin() * (tau * y).sin());
            apply_boundaries(&mut u, &BoundarySet::PERIODIC, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

            let op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(1.0));
            let mut out = ScalarField::new(&grid, 1);
            op.apply(&u, &mut out);

            let mut worst = 0.0f64;
            for j in 0..n {
                for i in 0..n {
                    let expected = -2.0 * tau * tau * u.get(i, j);
                    worst = worst.max((out.get(i, j) - expected).abs());
                }
            }
            worst
        };

        let coarse = error_at(32);
        let fine = error_at(64);
        let order = (coarse / fine).log2();
        assert!((order - 2.0).abs() < 0.1, "observed order {order:.3}, expected 2");
    }

    /// The conservation property the finite-volume form exists for: on a closed
    /// domain the operator sums to zero over all cells, so `∫u dA` cannot drift.
    #[test]
    fn the_operator_sums_to_zero_on_a_closed_domain() {
        for boundaries in [BoundarySet::PERIODIC, BoundarySet::INSULATED] {
            let grid = Grid2d::new(16, 12, [2.0, 1.5]);
            let mut u = ScalarField::new(&grid, 1);
            // An arbitrary, definitely non-uniform field.
            u.init_from_position(&grid, |[x, y]| (3.0 * x).sin() + (5.0 * y).cos() + x * y);
            apply_boundaries(&mut u, &boundaries, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

            let op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(0.7));
            let mut out = ScalarField::new(&grid, 1);
            op.apply(&u, &mut out);

            let total = out.sum_interior();
            let scale = out.max_abs_interior() * (grid.cell_count() as f64);
            assert!(
                total.abs() < 1e-12 * scale,
                "{boundaries:?}: interior fluxes did not cancel, sum = {total:e}"
            );
        }
    }

    /// Variable diffusivity must still conserve, because the two cells sharing a face
    /// use the *same* face coefficient.
    #[test]
    fn variable_diffusivity_still_conserves() {
        let grid = Grid2d::new(20, 20, [1.0, 1.0]);
        let mut d = ScalarField::new(&grid, 1);
        d.init_from_position(&grid, |[x, y]| 0.1 + x * x + y);
        let mut u = ScalarField::new(&grid, 1);
        u.init_from_position(&grid, |[x, y]| (10.0 * x).sin() * (7.0 * y).cos());
        apply_boundaries(&mut u, &BoundarySet::INSULATED, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let op = DiffusionOperator::new(&grid, &Diffusivity::Variable(d));
        let mut out = ScalarField::new(&grid, 1);
        op.apply(&u, &mut out);

        let total = out.sum_interior();
        let scale = out.max_abs_interior() * (grid.cell_count() as f64);
        assert!(total.abs() < 1e-12 * scale, "sum = {total:e}, scale = {scale:e}");
    }

    #[test]
    fn zero_diffusivity_produces_no_transport() {
        let (grid, mut u, op) = setup(6, 6, 0.0);
        u.set(3, 3, 1.0);
        apply_boundaries(&mut u, &BoundarySet::PERIODIC, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        let mut out = ScalarField::new(&grid, 1);
        op.apply(&u, &mut out);
        assert_eq!(out.max_abs_interior(), 0.0);
        assert!(op.explicit_stability_limit().max.is_infinite());
    }

    #[test]
    fn stability_limit_matches_the_von_neumann_formula() {
        let grid = Grid2d::new(100, 50, [1.0, 1.0]);
        let d = 0.25;
        let op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(d));
        let (dx, dy) = (grid.dx(), grid.dy());
        let expected = 1.0 / (2.0 * d * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        let limit = op.explicit_stability_limit();
        assert!((limit.max - expected).abs() < 1e-15 * expected, "{} vs {expected}", limit.max);
        assert_eq!(limit.reason, StabilityReason::DiffusionExplicit);
        assert!(limit.preferred < limit.max);
    }

    #[test]
    fn refining_the_grid_tightens_the_stability_limit_quadratically() {
        let coarse = DiffusionOperator::new(&Grid2d::new(50, 50, [1.0, 1.0]), &Diffusivity::Uniform(1.0));
        let fine = DiffusionOperator::new(&Grid2d::new(100, 100, [1.0, 1.0]), &Diffusivity::Uniform(1.0));
        let ratio = coarse.explicit_stability_limit().max / fine.explicit_stability_limit().max;
        assert!((ratio - 4.0).abs() < 1e-9, "halving h should quarter dt, got {ratio}");
    }

    /// §19.1's cross-backend level, at its strictest. Between two CPU threads there is
    /// no tolerance to hide in: the same expression over the same inputs must give the
    /// same bits, whatever the row bands are. A variable diffusivity is used so the
    /// face coefficients differ from cell to cell and a mis-indexed band cannot pass by
    /// symmetry.
    #[test]
    fn parallel_diffusion_is_bit_identical_to_the_scalar_path() {
        let grid = Grid2d::new(61, 37, [1.3, 0.9]);
        let mut d = ScalarField::new(&grid, 1);
        d.init_from_position(&grid, |[x, y]| 0.05 + x * x + 0.5 * y);
        let op = DiffusionOperator::new(&grid, &Diffusivity::Variable(d));

        let mut u = ScalarField::new(&grid, 1);
        u.init_from_position(&grid, |[x, y]| (11.0 * x).sin() * (7.0 * y).cos() + x);
        apply_boundaries(&mut u, &BoundarySet::INSULATED, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);

        let mut reference = ScalarField::new(&grid, 1);
        op.apply(&u, &mut reference);

        for threads in [2usize, 3, 4, 8] {
            let executor = Executor::with_threads(threads);
            let mut out = ScalarField::new(&grid, 1);
            op.apply_with(&executor, &u, &mut out);
            assert_eq!(
                out.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{threads} threads changed the stencil result"
            );
        }
    }

    /// A grid short enough that every band collapses to one still has to be right —
    /// this is the path a small interactive scene takes.
    #[test]
    fn a_grid_too_small_to_split_still_gives_the_same_answer() {
        let grid = Grid2d::new(5, 4, [1.0, 1.0]);
        let mut u = ScalarField::new(&grid, 1);
        u.init_from_position(&grid, |[x, y]| x + 2.0 * y);
        apply_boundaries(&mut u, &BoundarySet::PERIODIC, grid.dx(), grid.dy(), HaloMode::Inhomogeneous);
        let op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(0.3));

        let mut reference = ScalarField::new(&grid, 1);
        op.apply(&u, &mut reference);
        let mut parallel = ScalarField::new(&grid, 1);
        op.apply_with(&Executor::with_threads(8), &u, &mut parallel);
        assert_eq!(parallel.as_slice(), reference.as_slice());
    }

    #[test]
    #[should_panic(expected = "non-negative")]
    fn negative_diffusivity_is_rejected() {
        DiffusionOperator::new(&Grid2d::new(4, 4, [1.0, 1.0]), &Diffusivity::Uniform(-1.0));
    }

    #[test]
    fn faces_can_be_recomputed_when_diffusivity_changes() {
        let grid = Grid2d::new(4, 4, [1.0, 1.0]);
        let mut op = DiffusionOperator::new(&grid, &Diffusivity::Uniform(1.0));
        assert_eq!(op.max_diffusivity(), 1.0);
        op.recompute_faces(&Diffusivity::Uniform(5.0));
        assert_eq!(op.max_diffusivity(), 5.0);
        assert!(op.explicit_stability_limit().max < 1.0);
    }
}
