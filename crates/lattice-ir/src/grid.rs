//! Uniform 2D grids and the fields defined on them.
//!
//! Spec §11.3 specifies cell-centred scalar and vector fields on uniform 2D grids as
//! the MVP discretization. [`Grid2d`] holds the geometry; [`ScalarField`] holds the
//! values.
//!
//! # Halo cells
//!
//! A field carries a ring of *halo* (ghost) cells around its interior. Boundary
//! conditions are applied by writing the halo once per step, after which every
//! interior stencil reads its neighbours unconditionally:
//!
//! ```text
//!   halo=1, nx=4, ny=3           stride = nx + 2*halo = 6
//!   ┌───┬───┬───┬───┬───┬───┐
//!   │ h │ h │ h │ h │ h │ h │    j = -1  (halo row)
//!   ├───┼───┼───┼───┼───┼───┤
//!   │ h │ 0 │ 1 │ 2 │ 3 │ h │    j =  0
//!   ├───┼───┼───┼───┼───┼───┤
//!   │ h │ . │ . │ . │ . │ h │    j =  1
//!   ├───┼───┼───┼───┼───┼───┤
//!   │ h │ . │ . │ . │ . │ h │    j =  2
//!   ├───┼───┼───┼───┼───┼───┤
//!   │ h │ h │ h │ h │ h │ h │    j =  3  (halo row)
//!   └───┴───┴───┴───┴───┴───┘
//! ```
//!
//! Without a halo, every stencil evaluation needs four bounds tests, which defeats
//! vectorization and costs more than the halo's memory. This is the standard trade
//! and it is why boundary handling is a separate, explicit step rather than something
//! folded into the stencil.

use core::fmt;

/// A uniform, cell-centred 2D grid.
///
/// Cell `(i, j)` covers `[origin.x + i·dx, origin.x + (i+1)·dx] ×
/// [origin.y + j·dy, origin.y + (j+1)·dy]`, and its sample point is the centre.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Grid2d {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    origin: [f64; 2],
}

impl Grid2d {
    /// A grid of `nx × ny` cells spanning `extent` metres, with its lower-left corner
    /// at the origin.
    ///
    /// # Panics
    ///
    /// If either dimension is zero or either extent is not positive and finite. A
    /// degenerate grid produces infinite or NaN spacings that would surface much later
    /// as an unexplained NaN field (NFR-007 wants the failure at its source).
    pub fn new(nx: usize, ny: usize, extent: [f64; 2]) -> Self {
        Self::with_origin(nx, ny, extent, [0.0, 0.0])
    }

    /// A grid with an explicit lower-left corner.
    pub fn with_origin(nx: usize, ny: usize, extent: [f64; 2], origin: [f64; 2]) -> Self {
        assert!(nx > 0 && ny > 0, "grid must have at least one cell per axis, got {nx}x{ny}");
        assert!(
            extent[0].is_finite() && extent[0] > 0.0 && extent[1].is_finite() && extent[1] > 0.0,
            "grid extent must be positive and finite, got {extent:?}"
        );
        Self { nx, ny, dx: extent[0] / nx as f64, dy: extent[1] / ny as f64, origin }
    }

    /// Cells along x.
    pub const fn nx(&self) -> usize {
        self.nx
    }
    /// Cells along y.
    pub const fn ny(&self) -> usize {
        self.ny
    }
    /// Cell width, m.
    pub const fn dx(&self) -> f64 {
        self.dx
    }
    /// Cell height, m.
    pub const fn dy(&self) -> f64 {
        self.dy
    }
    /// Lower-left corner, m.
    pub const fn origin(&self) -> [f64; 2] {
        self.origin
    }

    /// Total interior cells.
    pub const fn cell_count(&self) -> usize {
        self.nx * self.ny
    }

    /// Area of one cell, m². The weight in every conservation integral.
    pub fn cell_area(&self) -> f64 {
        self.dx * self.dy
    }

    /// Physical size of the grid, m.
    pub fn extent(&self) -> [f64; 2] {
        [self.dx * self.nx as f64, self.dy * self.ny as f64]
    }

    /// Total area, m².
    pub fn area(&self) -> f64 {
        let e = self.extent();
        e[0] * e[1]
    }

    /// Centre of cell `(i, j)`, m.
    pub fn cell_center(&self, i: usize, j: usize) -> [f64; 2] {
        [
            self.origin[0] + (i as f64 + 0.5) * self.dx,
            self.origin[1] + (j as f64 + 0.5) * self.dy,
        ]
    }

    /// The cell containing a point, or `None` if the point is outside the grid.
    pub fn cell_at(&self, position: [f64; 2]) -> Option<(usize, usize)> {
        let fx = (position[0] - self.origin[0]) / self.dx;
        let fy = (position[1] - self.origin[1]) / self.dy;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (i, j) = (fx as usize, fy as usize);
        (i < self.nx && j < self.ny).then_some((i, j))
    }

    /// The smallest cell dimension, which sets the stability limit of explicit
    /// stencils on anisotropic grids.
    pub fn min_spacing(&self) -> f64 {
        self.dx.min(self.dy)
    }
}

impl fmt::Display for Grid2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = self.extent();
        write!(f, "{}x{} cells over {:.4}x{:.4} m", self.nx, self.ny, e[0], e[1])
    }
}

/// A cell-centred scalar field with a halo.
///
/// Interior coordinates run `0..nx` and `0..ny`; halo cells are addressed with
/// signed coordinates through [`ScalarField::index_signed`].
#[derive(Clone, PartialEq, Debug)]
pub struct ScalarField {
    nx: usize,
    ny: usize,
    halo: usize,
    stride: usize,
    data: Vec<f64>,
}

impl ScalarField {
    /// A zeroed field sized for `grid` with `halo` ghost cells on each side.
    pub fn new(grid: &Grid2d, halo: usize) -> Self {
        Self::filled(grid, halo, 0.0)
    }

    /// A field with every cell — interior and halo — set to `value`.
    pub fn filled(grid: &Grid2d, halo: usize, value: f64) -> Self {
        let stride = grid.nx() + 2 * halo;
        let rows = grid.ny() + 2 * halo;
        Self { nx: grid.nx(), ny: grid.ny(), halo, stride, data: vec![value; stride * rows] }
    }

    /// Interior cells along x.
    pub const fn nx(&self) -> usize {
        self.nx
    }
    /// Interior cells along y.
    pub const fn ny(&self) -> usize {
        self.ny
    }
    /// Ghost cells on each side.
    pub const fn halo(&self) -> usize {
        self.halo
    }
    /// Distance in elements between vertically adjacent cells.
    pub const fn stride(&self) -> usize {
        self.stride
    }
    /// Interior cell count.
    pub const fn interior_len(&self) -> usize {
        self.nx * self.ny
    }
    /// Total stored cells, including halo.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Always false; a field always has at least one cell.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat index of interior cell `(i, j)`.
    ///
    /// Debug builds check bounds; release builds do not, because this is called once
    /// per cell per stencil and the halo already guarantees neighbours are in range.
    #[inline]
    pub fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.nx && j < self.ny, "interior index ({i},{j}) out of {}x{}", self.nx, self.ny);
        (j + self.halo) * self.stride + (i + self.halo)
    }

    /// Flat index allowing halo coordinates, e.g. `(-1, 0)` for the left ghost cell.
    #[inline]
    pub fn index_signed(&self, i: isize, j: isize) -> usize {
        let h = self.halo as isize;
        debug_assert!(
            i >= -h && i < self.nx as isize + h && j >= -h && j < self.ny as isize + h,
            "index ({i},{j}) is outside the field including its halo"
        );
        ((j + h) as usize) * self.stride + (i + h) as usize
    }

    /// Flat index of interior cell `(0, 0)`.
    #[inline]
    pub const fn interior_origin(&self) -> usize {
        self.halo * self.stride + self.halo
    }

    /// Value at an interior cell.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.index(i, j)]
    }

    /// Set an interior cell.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        let k = self.index(i, j);
        self.data[k] = value;
    }

    /// The whole buffer, halo included.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The whole buffer, halo included, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// One interior row.
    pub fn row(&self, j: usize) -> &[f64] {
        let start = self.index(0, j);
        &self.data[start..start + self.nx]
    }

    /// One interior row, mutable.
    pub fn row_mut(&mut self, j: usize) -> &mut [f64] {
        let start = self.index(0, j);
        &mut self.data[start..start + self.nx]
    }

    /// Set every interior cell, leaving the halo alone.
    pub fn fill_interior(&mut self, value: f64) {
        for j in 0..self.ny {
            self.row_mut(j).fill(value);
        }
    }

    /// Set every cell, halo included.
    pub fn fill_all(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Initialize each interior cell from its physical position.
    ///
    /// The closure receives the cell centre in metres, which is what an initial
    /// condition like a Gaussian or a `left_half(...)` region is written in terms of.
    pub fn init_from_position(&mut self, grid: &Grid2d, mut f: impl FnMut([f64; 2]) -> f64) {
        debug_assert_eq!((grid.nx(), grid.ny()), (self.nx, self.ny), "grid/field size mismatch");
        for j in 0..self.ny {
            for i in 0..self.nx {
                let v = f(grid.cell_center(i, j));
                self.set(i, j, v);
            }
        }
    }

    /// Copy interior values from another field of the same interior size.
    pub fn copy_interior_from(&mut self, other: &ScalarField) {
        assert_eq!(
            (self.nx, self.ny),
            (other.nx, other.ny),
            "interior sizes must match to copy"
        );
        for j in 0..self.ny {
            let src_start = other.index(0, j);
            let dst_start = self.index(0, j);
            self.data[dst_start..dst_start + self.nx]
                .copy_from_slice(&other.data[src_start..src_start + other.nx]);
        }
    }

    /// Sum over interior cells. The building block of every conservation check.
    pub fn sum_interior(&self) -> f64 {
        (0..self.ny).map(|j| self.row(j).iter().sum::<f64>()).sum()
    }

    /// Integral over the grid: `sum · cell_area`.
    pub fn integrate(&self, grid: &Grid2d) -> f64 {
        self.sum_interior() * grid.cell_area()
    }

    /// Smallest interior value.
    pub fn min_interior(&self) -> f64 {
        (0..self.ny).flat_map(|j| self.row(j).iter().copied()).fold(f64::INFINITY, f64::min)
    }

    /// Largest interior value.
    pub fn max_interior(&self) -> f64 {
        (0..self.ny).flat_map(|j| self.row(j).iter().copied()).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Largest absolute interior value, used for residual norms.
    pub fn max_abs_interior(&self) -> f64 {
        (0..self.ny)
            .flat_map(|j| self.row(j).iter().map(|v| v.abs()))
            .fold(0.0, f64::max)
    }

    /// The first interior cell holding a NaN or infinity, if any.
    ///
    /// NFR-007 requires the engine to surface instability rather than hide it. The
    /// runtime calls this on a cadence and reports the *location*, not just the fact.
    pub fn first_non_finite(&self) -> Option<(usize, usize)> {
        for j in 0..self.ny {
            for (i, &v) in self.row(j).iter().enumerate() {
                if !v.is_finite() {
                    return Some((i, j));
                }
            }
        }
        None
    }
}

/// A 2D vector field stored as two collocated scalar components.
///
/// Collocated rather than staggered: staggered layouts are better for incompressible
/// flow but the MVP's electrostatic and force-density fields are naturally
/// cell-centred. A staggered variant belongs with the fluid module (§11.5), where the
/// choice actually matters.
#[derive(Clone, PartialEq, Debug)]
pub struct VectorField {
    /// x component.
    pub x: ScalarField,
    /// y component.
    pub y: ScalarField,
}

impl VectorField {
    /// A zeroed vector field on `grid`.
    pub fn new(grid: &Grid2d, halo: usize) -> Self {
        Self { x: ScalarField::new(grid, halo), y: ScalarField::new(grid, halo) }
    }

    /// Value at an interior cell.
    pub fn get(&self, i: usize, j: usize) -> [f64; 2] {
        [self.x.get(i, j), self.y.get(i, j)]
    }

    /// Set an interior cell.
    pub fn set(&mut self, i: usize, j: usize, value: [f64; 2]) {
        self.x.set(i, j, value[0]);
        self.y.set(i, j, value[1]);
    }

    /// Zero both components everywhere.
    pub fn fill_all(&mut self, value: [f64; 2]) {
        self.x.fill_all(value[0]);
        self.y.fill_all(value[1]);
    }
}

/// Which edge of the grid a boundary condition applies to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Side {
    /// `i = -1`, the low-x edge.
    Left,
    /// `i = nx`, the high-x edge.
    Right,
    /// `j = -1`, the low-y edge.
    Bottom,
    /// `j = ny`, the high-y edge.
    Top,
}

impl Side {
    /// All four sides.
    pub const ALL: [Side; 4] = [Side::Left, Side::Right, Side::Bottom, Side::Top];

    /// The side facing this one.
    pub const fn opposite(self) -> Side {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
            Side::Bottom => Side::Top,
            Side::Top => Side::Bottom,
        }
    }

    /// Name for diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
            Side::Bottom => "bottom",
            Side::Top => "top",
        }
    }
}

/// A boundary condition on one edge of a field.
///
/// Signs follow the outward normal: a positive [`Boundary::Neumann`] gradient means
/// the value increases as you leave the domain, so a positive value on the left edge
/// and a positive value on the right edge describe opposite physical fluxes. Stating
/// this once here avoids the sign errors that otherwise appear in every solver.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Boundary {
    /// Wraps to the opposite edge. Must be declared on both edges of an axis.
    Periodic,
    /// Fixed value at the boundary face.
    Dirichlet {
        /// The prescribed value.
        value: f64,
    },
    /// Fixed outward normal derivative `∂u/∂n` at the boundary face.
    ///
    /// `Neumann { gradient: 0.0 }` is the insulating / no-flux condition.
    Neumann {
        /// The prescribed outward derivative, in field units per metre.
        gradient: f64,
    },
    /// Mixed condition `-∂u/∂n = coefficient · (u − ambient)`.
    ///
    /// Models convective exchange with a reservoir. `coefficient` is `h/k` for heat
    /// transfer, with units of 1/metre.
    Robin {
        /// Exchange coefficient, 1/m.
        coefficient: f64,
        /// Reservoir value.
        ambient: f64,
    },
}

impl Boundary {
    /// The insulating condition, `∂u/∂n = 0`.
    pub const INSULATED: Boundary = Boundary::Neumann { gradient: 0.0 };

    /// A fixed value.
    pub const fn fixed(value: f64) -> Boundary {
        Boundary::Dirichlet { value }
    }

    /// Name for diagnostics.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Boundary::Periodic => "periodic",
            Boundary::Dirichlet { .. } => "Dirichlet",
            Boundary::Neumann { .. } => "Neumann",
            Boundary::Robin { .. } => "Robin",
        }
    }

    /// True for conditions that let the quantity leave the domain.
    ///
    /// A field with only [`Boundary::Periodic`] and zero-gradient Neumann edges is
    /// closed, so its integral must be conserved and the ledger can say so. Any other
    /// combination is an open system and drift is expected rather than a bug.
    pub fn is_closed(&self) -> bool {
        match self {
            Boundary::Periodic => true,
            Boundary::Neumann { gradient } => *gradient == 0.0,
            Boundary::Dirichlet { .. } | Boundary::Robin { .. } => false,
        }
    }
}

/// Boundary conditions for all four edges.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoundarySet {
    /// Low-x edge.
    pub left: Boundary,
    /// High-x edge.
    pub right: Boundary,
    /// Low-y edge.
    pub bottom: Boundary,
    /// High-y edge.
    pub top: Boundary,
}

impl BoundarySet {
    /// The same condition on all four edges.
    pub const fn uniform(b: Boundary) -> Self {
        Self { left: b, right: b, bottom: b, top: b }
    }

    /// Fully periodic in both directions.
    pub const PERIODIC: BoundarySet = BoundarySet::uniform(Boundary::Periodic);

    /// Insulated on all four edges — a closed box.
    pub const INSULATED: BoundarySet = BoundarySet::uniform(Boundary::INSULATED);

    /// The condition on one side.
    pub const fn get(&self, side: Side) -> Boundary {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
            Side::Bottom => self.bottom,
            Side::Top => self.top,
        }
    }

    /// Replace the condition on one side.
    pub fn set(&mut self, side: Side, b: Boundary) {
        match side {
            Side::Left => self.left = b,
            Side::Right => self.right = b,
            Side::Bottom => self.bottom = b,
            Side::Top => self.top = b,
        }
    }

    /// Check that periodicity is declared consistently.
    ///
    /// A grid that is periodic on the left but Dirichlet on the right has no
    /// well-defined meaning. Spec §8.4 step 8 calls for "boundary completeness"
    /// validation at compile time; this is that check.
    pub fn validate(&self) -> Result<(), BoundaryError> {
        for (a, b, axis) in [
            (self.left, self.right, "x"),
            (self.bottom, self.top, "y"),
        ] {
            let a_periodic = matches!(a, Boundary::Periodic);
            let b_periodic = matches!(b, Boundary::Periodic);
            if a_periodic != b_periodic {
                return Err(BoundaryError::UnpairedPeriodic {
                    axis,
                    low: a.kind_name(),
                    high: b.kind_name(),
                });
            }
        }
        Ok(())
    }

    /// True when no quantity can enter or leave through any edge.
    pub fn is_closed(&self) -> bool {
        Side::ALL.iter().all(|&s| self.get(s).is_closed())
    }
}

/// A malformed boundary specification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryError {
    /// One edge of an axis is periodic and the other is not.
    UnpairedPeriodic {
        /// `"x"` or `"y"`.
        axis: &'static str,
        /// Condition on the low edge.
        low: &'static str,
        /// Condition on the high edge.
        high: &'static str,
    },
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundaryError::UnpairedPeriodic { axis, low, high } => write!(
                f,
                "the {axis} axis has mismatched boundaries: {low} on the low edge and {high} on \
                 the high edge; periodicity must be declared on both edges of an axis"
            ),
        }
    }
}

impl core::error::Error for BoundaryError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid2d {
        // The spec §25.1 chamber: 512x256 cells over 2 m x 1 m.
        Grid2d::new(512, 256, [2.0, 1.0])
    }

    #[test]
    fn grid_spacing_follows_from_size_and_extent() {
        let g = grid();
        assert!((g.dx() - 2.0 / 512.0).abs() < 1e-15);
        assert!((g.dy() - 1.0 / 256.0).abs() < 1e-15);
        // Square cells, as it happens.
        assert!((g.dx() - g.dy()).abs() < 1e-15);
        assert_eq!(g.cell_count(), 512 * 256);
        assert!((g.area() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cell_centers_are_offset_by_half_a_cell() {
        let g = Grid2d::new(2, 2, [2.0, 2.0]);
        assert_eq!(g.cell_center(0, 0), [0.5, 0.5]);
        assert_eq!(g.cell_center(1, 1), [1.5, 1.5]);
    }

    #[test]
    fn cell_lookup_round_trips_and_rejects_outside_points() {
        let g = Grid2d::new(4, 4, [4.0, 4.0]);
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(g.cell_at(g.cell_center(i, j)), Some((i, j)));
            }
        }
        assert_eq!(g.cell_at([-0.1, 1.0]), None);
        assert_eq!(g.cell_at([4.1, 1.0]), None);
        assert_eq!(g.cell_at([1.0, -0.1]), None);
        assert_eq!(g.cell_at([1.0, 4.1]), None);
    }

    #[test]
    fn grid_with_origin_shifts_coordinates() {
        let g = Grid2d::with_origin(2, 2, [2.0, 2.0], [-1.0, -1.0]);
        assert_eq!(g.cell_center(0, 0), [-0.5, -0.5]);
        assert_eq!(g.cell_at([-0.5, -0.5]), Some((0, 0)));
    }

    #[test]
    #[should_panic(expected = "at least one cell")]
    fn degenerate_grids_are_rejected() {
        Grid2d::new(0, 4, [1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "positive and finite")]
    fn zero_extent_is_rejected() {
        Grid2d::new(4, 4, [0.0, 1.0]);
    }

    #[test]
    fn field_indexing_accounts_for_the_halo() {
        let g = Grid2d::new(4, 3, [4.0, 3.0]);
        let f = ScalarField::new(&g, 1);
        assert_eq!(f.stride(), 6);
        assert_eq!(f.len(), 6 * 5);
        // halo * stride + halo, with halo = 1 and stride = 6.
        assert_eq!(f.interior_origin(), 7);
        assert_eq!(f.index(0, 0), 7);
        assert_eq!(f.index(3, 2), 3 * 6 + 4);
        // Halo addressing.
        assert_eq!(f.index_signed(-1, 0), 6);
        assert_eq!(f.index_signed(4, 2), 3 * 6 + 5);
    }

    #[test]
    fn zero_halo_fields_are_allowed() {
        let g = Grid2d::new(3, 2, [3.0, 2.0]);
        let f = ScalarField::new(&g, 0);
        assert_eq!(f.stride(), 3);
        assert_eq!(f.len(), 6);
        assert_eq!(f.index(0, 0), 0);
        assert_eq!(f.index(2, 1), 5);
    }

    #[test]
    fn fill_interior_leaves_the_halo_untouched() {
        let g = Grid2d::new(2, 2, [2.0, 2.0]);
        let mut f = ScalarField::filled(&g, 1, -1.0);
        f.fill_interior(5.0);
        assert_eq!(f.get(0, 0), 5.0);
        assert_eq!(f.get(1, 1), 5.0);
        // Halo still holds the sentinel.
        assert_eq!(f.as_slice()[f.index_signed(-1, -1)], -1.0);
        assert_eq!(f.as_slice()[f.index_signed(2, 2)], -1.0);
    }

    #[test]
    fn rows_are_contiguous_interior_slices() {
        let g = Grid2d::new(4, 3, [4.0, 3.0]);
        let mut f = ScalarField::new(&g, 2);
        for j in 0..3 {
            f.row_mut(j).fill(j as f64);
        }
        assert_eq!(f.row(0), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(f.row(2), &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(f.get(3, 1), 1.0);
    }

    #[test]
    fn integrate_weights_by_cell_area() {
        let g = Grid2d::new(10, 10, [1.0, 1.0]);
        let mut f = ScalarField::new(&g, 1);
        f.fill_interior(3.0);
        assert!((f.sum_interior() - 300.0).abs() < 1e-12);
        // 3 per unit area over a 1 m^2 domain.
        assert!((f.integrate(&g) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn init_from_position_sees_cell_centers() {
        let g = Grid2d::new(2, 1, [2.0, 1.0]);
        let mut f = ScalarField::new(&g, 1);
        f.init_from_position(&g, |[x, _y]| x);
        assert_eq!(f.get(0, 0), 0.5);
        assert_eq!(f.get(1, 0), 1.5);
    }

    #[test]
    fn copy_interior_ignores_differing_halos() {
        let g = Grid2d::new(3, 3, [3.0, 3.0]);
        let mut src = ScalarField::new(&g, 1);
        src.fill_interior(7.0);
        let mut dst = ScalarField::filled(&g, 3, -1.0);
        dst.copy_interior_from(&src);
        assert_eq!(dst.get(1, 1), 7.0);
        // dst's wider halo is untouched.
        assert_eq!(dst.as_slice()[dst.index_signed(-3, -3)], -1.0);
    }

    #[test]
    fn non_finite_detection_reports_the_first_cell() {
        let g = Grid2d::new(4, 4, [4.0, 4.0]);
        let mut f = ScalarField::new(&g, 1);
        assert_eq!(f.first_non_finite(), None);
        f.set(2, 1, f64::NAN);
        assert_eq!(f.first_non_finite(), Some((2, 1)));
    }

    #[test]
    fn min_max_scan_only_the_interior() {
        let g = Grid2d::new(3, 3, [3.0, 3.0]);
        let mut f = ScalarField::filled(&g, 1, 1000.0);
        f.fill_interior(0.0);
        f.set(1, 1, -5.0);
        f.set(2, 2, 4.0);
        assert_eq!(f.min_interior(), -5.0);
        assert_eq!(f.max_interior(), 4.0);
        assert_eq!(f.max_abs_interior(), 5.0);
    }

    #[test]
    fn vector_field_components_are_independent() {
        let g = Grid2d::new(2, 2, [2.0, 2.0]);
        let mut v = VectorField::new(&g, 1);
        v.set(1, 0, [3.0, -4.0]);
        assert_eq!(v.get(1, 0), [3.0, -4.0]);
        assert_eq!(v.get(0, 0), [0.0, 0.0]);
    }

    #[test]
    fn unpaired_periodicity_is_rejected() {
        let mut bs = BoundarySet::INSULATED;
        bs.left = Boundary::Periodic;
        let err = bs.validate().unwrap_err();
        assert!(matches!(err, BoundaryError::UnpairedPeriodic { axis: "x", .. }));
        assert!(err.to_string().contains("both edges"));

        bs.right = Boundary::Periodic;
        assert!(bs.validate().is_ok());
    }

    #[test]
    fn closedness_matches_the_physics() {
        assert!(BoundarySet::PERIODIC.is_closed());
        assert!(BoundarySet::INSULATED.is_closed());

        let mut leaky = BoundarySet::INSULATED;
        leaky.left = Boundary::fixed(300.0);
        assert!(!leaky.is_closed(), "a Dirichlet edge is a source/sink");

        let mut flux = BoundarySet::INSULATED;
        flux.top = Boundary::Neumann { gradient: 1.0 };
        assert!(!flux.is_closed(), "a nonzero gradient drives flux across the edge");
    }

    #[test]
    fn sides_are_paired_correctly() {
        assert_eq!(Side::Left.opposite(), Side::Right);
        assert_eq!(Side::Top.opposite(), Side::Bottom);
        for s in Side::ALL {
            assert_eq!(s.opposite().opposite(), s);
        }
    }
}
