//! What the particle moves through: a real potential, and an absorbing layer.
//!
//! Spec §13.1 asks for *"barriers and wells"* and for potentials given as an
//! *"analytic expression, image/field, coupled electrostatic potential"*. The analytic
//! shapes the language can name are here — walls with slits, rectangles, a harmonic
//! trap — and [`Potential::from_fn`] takes anything else.
//!
//! # Absorbing boundaries
//!
//! A split-step Fourier propagator is periodic: a packet leaving on the right comes
//! back on the left. Spec §25.2's `boundary: absorbing(width=…)` is a *complex
//! absorbing potential*, `V → V − iW`, with `W` rising from zero to its strength over
//! the layer:
//!
//! ```text
//!   W(s) = W₀ s²,   s = depth into the layer / width ∈ [0, 1]
//! ```
//!
//! An imaginary potential removes probability at the rate `2W/ħ` per unit density,
//! so what the layer takes is known exactly, step by step, and the domain reports it.
//! Norm plus absorbed probability is then an invariant — conserved to round-off — even
//! though the norm on its own is not.
//!
//! The quadratic ramp is the standard compromise: a step in `W` would reflect like any
//! other step in a potential, and a ramp too gentle would let the packet through to the
//! far side. [`Absorber::for_speed`] sets `W₀` from the speed of what it must stop;
//! the validation suite measures how much it reflects.

use lattice_ir::Grid2d;

/// A real potential energy per cell, joules.
#[derive(Clone, Debug, PartialEq)]
pub struct Potential {
    grid: Grid2d,
    values: Vec<f64>,
}

impl Potential {
    /// Zero everywhere.
    pub fn zero(grid: Grid2d) -> Self {
        Self { grid, values: vec![0.0; grid.nx() * grid.ny()] }
    }

    /// `V(r)` sampled at each cell centre.
    pub fn from_fn(grid: Grid2d, f: impl Fn([f64; 2]) -> f64) -> Self {
        let mut values = Vec::with_capacity(grid.nx() * grid.ny());
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                values.push(f(grid.cell_center(i, j)));
            }
        }
        Self { grid, values }
    }

    /// Add a shape's contribution.
    pub fn add(&mut self, shape: &Shape) {
        for j in 0..self.grid.ny() {
            for i in 0..self.grid.nx() {
                self.values[j * self.grid.nx() + i] += shape.value_at(self.grid.cell_center(i, j));
            }
        }
    }

    /// The grid.
    pub fn grid(&self) -> Grid2d {
        self.grid
    }

    /// The values, row-major, joules.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The largest `|V|`, joules.
    pub fn max_abs(&self) -> f64 {
        self.values.iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    /// `Σ V |ψ|² ΔA` for a density given per cell.
    pub fn expectation(&self, density: impl Iterator<Item = f64>) -> f64 {
        self.values.iter().zip(density).map(|(v, p)| v * p).sum::<f64>() * self.grid.cell_area()
    }
}

/// An analytic potential the language can name.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// A wall parallel to the y axis, `|x − x₀| ≤ thickness/2`, open where `y` lies in
    /// one of the slits.
    Wall {
        /// Centre line, m.
        x: f64,
        /// Thickness, m.
        thickness: f64,
        /// Openings as `(centre y, width)`, m.
        slits: Vec<(f64, f64)>,
        /// Height, J.
        height: f64,
    },
    /// A rectangle of constant height: a barrier when positive, a well when negative.
    Rectangle {
        /// `[x_min, x_max]`, m.
        x: [f64; 2],
        /// `[y_min, y_max]`, m.
        y: [f64; 2],
        /// Height, J.
        height: f64,
    },
    /// `V = ½ m ω² |r − c|²`.
    Harmonic {
        /// Centre, m.
        center: [f64; 2],
        /// Angular frequency, rad/s.
        omega: f64,
        /// The particle's mass, kg — the trap is defined by its frequency, so the
        /// stiffness depends on what is in it.
        mass: f64,
    },
}

impl Shape {
    /// `V` at a point, joules.
    pub fn value_at(&self, [x, y]: [f64; 2]) -> f64 {
        match self {
            Shape::Wall { x: x0, thickness, slits, height } => {
                let inside = (x - x0).abs() <= 0.5 * thickness;
                let open = slits.iter().any(|(center, width)| (y - center).abs() <= 0.5 * width);
                if inside && !open { *height } else { 0.0 }
            }
            Shape::Rectangle { x: [x0, x1], y: [y0, y1], height } => {
                if (*x0..=*x1).contains(&x) && (*y0..=*y1).contains(&y) { *height } else { 0.0 }
            }
            Shape::Harmonic { center, omega, mass } => {
                let (dx, dy) = (x - center[0], y - center[1]);
                0.5 * mass * omega * omega * (dx * dx + dy * dy)
            }
        }
    }

    /// One line for the model report.
    pub fn describe(&self) -> String {
        match self {
            Shape::Wall { x, thickness, slits, height } => format!(
                "wall at x = {x:.4e} m, {thickness:.4e} m thick, {height:.4e} J high, {} slit{}",
                slits.len(),
                if slits.len() == 1 { "" } else { "s" }
            ),
            Shape::Rectangle { x, y, height } => format!(
                "rectangle x in [{:.4e}, {:.4e}] m, y in [{:.4e}, {:.4e}] m, {height:.4e} J",
                x[0], x[1], y[0], y[1]
            ),
            Shape::Harmonic { center, omega, .. } => format!(
                "harmonic trap at ({:.4e}, {:.4e}) m, omega = {omega:.4e} rad/s",
                center[0], center[1]
            ),
        }
    }
}

/// A complex absorbing layer along the grid's edges.
#[derive(Clone, Debug, PartialEq)]
pub struct Absorber {
    width: f64,
    strength: f64,
    profile: Vec<f64>,
}

impl Absorber {
    /// A layer `width` deep on every edge, rising quadratically to `strength` joules.
    ///
    /// # Panics
    ///
    /// On a non-positive width or strength, or a layer so wide the two sides meet.
    pub fn new(grid: Grid2d, width: f64, strength: f64) -> Self {
        Self::on_axes(grid, width, strength, [true, true])
    }

    /// A layer on the edges of the axes marked `true` only.
    ///
    /// A grid one row high is a line, and a layer on its y edges would cover all of it;
    /// this is how such a grid absorbs at its ends and nowhere else.
    ///
    /// # Panics
    ///
    /// As [`Absorber::new`], for the axes that have a layer, or if neither has one.
    pub fn on_axes(grid: Grid2d, width: f64, strength: f64, axes: [bool; 2]) -> Self {
        assert!(width > 0.0 && strength > 0.0, "an absorber needs a positive width and strength");
        assert!(axes[0] || axes[1], "an absorber needs at least one axis");
        let [lx, ly] = grid.extent();
        for (axis, extent) in [(0, lx), (1, ly)] {
            assert!(
                !axes[axis] || 2.0 * width < extent,
                "a {width:e} m absorbing layer on both sides leaves nothing of a {lx:e} x {ly:e} m grid"
            );
        }
        let origin = grid.origin();
        let mut profile = Vec::with_capacity(grid.nx() * grid.ny());
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let [x, y] = grid.cell_center(i, j);
                let to_x = if axes[0] { (x - origin[0]).min(origin[0] + lx - x) } else { f64::INFINITY };
                let to_y = if axes[1] { (y - origin[1]).min(origin[1] + ly - y) } else { f64::INFINITY };
                let edge = to_x.min(to_y);
                let depth = ((width - edge) / width).clamp(0.0, 1.0);
                profile.push(strength * depth * depth);
            }
        }
        Self { width, strength, profile }
    }

    /// A layer strong enough to stop a particle moving at `speed`.
    ///
    /// The fraction of a wave that survives one pass through the layer is
    /// `exp(−2∫W dx / ħv) = exp(−2W₀·width / 3ħv)`. A packet that survives the first pass
    /// on a periodic grid goes through the layer again on the far side, so two passes
    /// stand between it and the region of interest. `W₀ = 10ħv/width` makes that
    /// `e^(−40/3)`, about `2·10⁻⁶`. Reflection off the ramp itself is the other
    /// error, and it grows with `W₀`; the validation suite measures both.
    pub fn for_speed(grid: Grid2d, width: f64, speed: f64, hbar: f64) -> Self {
        Self::new(grid, width, Self::strength_for_speed(width, speed, hbar))
    }

    /// The strength [`Absorber::for_speed`] chooses, `10ħv / width`, J.
    pub fn strength_for_speed(width: f64, speed: f64, hbar: f64) -> f64 {
        10.0 * hbar * speed / width
    }

    /// Layer depth, m.
    pub fn width(&self) -> f64 {
        self.width
    }

    /// Peak `W`, joules.
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// `W` per cell, row-major, joules.
    pub fn profile(&self) -> &[f64] {
        &self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid2d {
        Grid2d::with_origin(40, 20, [4.0, 2.0], [-2.0, -1.0])
    }

    #[test]
    fn a_wall_is_open_only_at_its_slits() {
        let wall = Shape::Wall { x: 0.0, thickness: 0.2, slits: vec![(-0.5, 0.2), (0.5, 0.2)], height: 7.0 };
        assert_eq!(wall.value_at([0.0, 0.0]), 7.0);
        assert_eq!(wall.value_at([0.0, 0.5]), 0.0);
        assert_eq!(wall.value_at([0.0, -0.55]), 0.0);
        assert_eq!(wall.value_at([0.3, 0.0]), 0.0, "beside the wall");
    }

    #[test]
    fn shapes_add_into_a_potential() {
        let mut v = Potential::zero(grid());
        v.add(&Shape::Rectangle { x: [-0.5, 0.5], y: [-1.0, 1.0], height: 2.0 });
        v.add(&Shape::Rectangle { x: [0.0, 0.5], y: [-1.0, 1.0], height: 1.0 });
        assert_eq!(v.max_abs(), 3.0);
        let count = v.values().iter().filter(|&&x| x > 0.0).count();
        assert_eq!(count, 10 * 20, "a 1 m wide strip of 0.1 m cells, all 20 rows");
    }

    #[test]
    fn a_harmonic_trap_is_quadratic() {
        let trap = Shape::Harmonic { center: [0.0, 0.0], omega: 2.0, mass: 3.0 };
        assert_eq!(trap.value_at([1.0, 0.0]), 6.0);
        assert_eq!(trap.value_at([0.0, -2.0]), 24.0);
    }

    #[test]
    fn the_absorber_is_zero_inside_and_rises_to_its_strength_at_the_edge() {
        let layer = Absorber::new(grid(), 0.5, 4.0);
        let g = grid();
        let at = |i: usize, j: usize| layer.profile()[j * g.nx() + i];
        assert_eq!(at(20, 10), 0.0, "the interior is untouched");
        assert!(at(0, 10) > 3.0 && at(0, 10) <= 4.0, "{}", at(0, 10));
        assert!(at(2, 10) < at(1, 10) && at(1, 10) < at(0, 10), "monotone into the layer");
        assert_eq!(at(10, 10), 0.0, "beyond the layer's depth");
    }

    #[test]
    fn a_line_absorbs_only_at_its_ends() {
        let line = Grid2d::with_origin(40, 1, [4.0, 0.01], [-2.0, 0.0]);
        let layer = Absorber::on_axes(line, 0.5, 1.0, [true, false]);
        assert!(layer.profile()[0] > 0.0 && layer.profile()[39] > 0.0);
        assert_eq!(layer.profile()[20], 0.0);
    }

    #[test]
    #[should_panic(expected = "leaves nothing")]
    fn an_absorber_wider_than_half_the_grid_is_refused() {
        let _ = Absorber::new(grid(), 1.5, 1.0);
    }
}
