//! Detectors: how much probability crossed a line, and where along it.
//!
//! Spec §25.2's `detector screen at x=4.5 nanometer` is a screen parallel to the y
//! axis. What a screen records is *flux*: the probability current through it,
//! integrated over time,
//!
//! ```text
//!   P(y) = ∫ J_x(x_d, y, t) dt,     J_x = (ħ/m) Im(ψ* ∂ψ/∂x)
//! ```
//!
//! `P(y)` is the interference pattern — a probability per unit length along the
//! screen — and its integral over `y` is the probability that has arrived.
//!
//! # Discretization
//!
//! The screen sits on the cell face nearest `x_d`, between columns `i` and `i + 1`.
//! There the current is
//!
//! ```text
//!   J = (ħ / mΔx) · Im(ψᵢ* ψᵢ₊₁)
//! ```
//!
//! which is *exactly* the current that satisfies the discrete continuity equation of
//! the five-point Hamiltonian: under Crank–Nicolson, what a screen counts and what
//! leaves the cells behind it agree to the solver's tolerance. Under split-step it is a
//! second-order approximation to the spectral current. Time integration is by the
//! trapezoid rule over each step.

use lattice_ir::Grid2d;

use crate::hamiltonian::Hamiltonian;
use crate::wavefunction::Wavefunction;

/// A screen parallel to the y axis.
#[derive(Clone, Debug)]
pub struct Detector {
    name: String,
    x: f64,
    /// The screen lies on the face between column `column` and `column + 1`.
    column: usize,
    /// `∫J dt` per row, 1/m.
    integrated: Vec<f64>,
    /// `J` per row at the end of the last step, 1/(m·s).
    last: Vec<f64>,
    /// Scratch for the current flux.
    current: Vec<f64>,
}

impl Detector {
    /// A screen at `x` metres.
    ///
    /// # Panics
    ///
    /// If `x` does not fall between two cell centres of the grid.
    pub fn new(name: impl Into<String>, grid: Grid2d, x: f64) -> Self {
        let first = grid.cell_center(0, 0)[0];
        let position = (x - first) / grid.dx();
        assert!(
            position >= 0.0 && position <= (grid.nx() - 1) as f64,
            "a screen at x = {x:e} m is outside the grid's interior faces"
        );
        let column = (position.floor() as usize).min(grid.nx() - 2);
        let rows = grid.ny();
        Self {
            name: name.into(),
            x,
            column,
            integrated: vec![0.0; rows],
            last: vec![0.0; rows],
            current: vec![0.0; rows],
        }
    }

    /// The detector's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where it was asked to be, m.
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Where it is: the face it was placed on, m.
    pub fn face_x(&self, grid: Grid2d) -> f64 {
        grid.origin()[0] + (self.column + 1) as f64 * grid.dx()
    }

    fn flux(&mut self, h: &Hamiltonian, psi: &Wavefunction) {
        let grid = h.grid();
        let scale = h.hbar() / (h.mass() * grid.dx());
        for (j, slot) in self.current.iter_mut().enumerate() {
            let left = psi.get(self.column, j);
            let right = psi.get(self.column + 1, j);
            *slot = scale * (left.conj() * right).im;
        }
    }

    /// Record the flux of the state a run starts from, so the first step's trapezoid
    /// has a left end. Clears anything recorded so far.
    pub fn reset(&mut self, h: &Hamiltonian, psi: &Wavefunction) {
        self.flux(h, psi);
        self.last.copy_from_slice(&self.current);
        self.integrated.fill(0.0);
    }

    /// Accumulate one step of `dt` seconds ending at `psi`.
    pub fn record(&mut self, h: &Hamiltonian, psi: &Wavefunction, dt: f64) {
        self.flux(h, psi);
        for ((total, last), now) in self.integrated.iter_mut().zip(&mut self.last).zip(&self.current) {
            *total += 0.5 * (*last + *now) * dt;
            *last = *now;
        }
    }

    /// The net probability that has crossed in the `+x` direction.
    pub fn arrived(&self, grid: Grid2d) -> f64 {
        self.integrated.iter().sum::<f64>() * grid.dy()
    }

    /// The arrival pattern: row centres, m, and `∫J dt` there, 1/m.
    pub fn pattern(&self, grid: Grid2d) -> (Vec<f64>, Vec<f64>) {
        let y = (0..grid.ny()).map(|j| grid.cell_center(0, j)[1]).collect();
        (y, self.integrated.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::hamiltonian::Kinetic;

    #[test]
    fn a_plane_wave_carries_density_times_velocity() {
        let grid = Grid2d::new(64, 8, [1.0, 0.125]);
        let (mass, hbar) = (2.0, 1.0);
        let k = core::f64::consts::TAU * 4.0;
        let h = Hamiltonian::new(grid, mass, hbar, Kinetic::Spectral);
        let psi = Wavefunction::from_fn(grid, |[x, _]| Complex::cis(k * x));
        let mut screen = Detector::new("s", grid, 0.5);
        screen.reset(&h, &psi);
        screen.record(&h, &psi, 0.1);
        // |ψ|² = 1, and the lattice current is (ħ/mΔx)·sin(kΔx).
        let expected = hbar / (mass * grid.dx()) * (k * grid.dx()).sin() * 0.1 * grid.extent()[1];
        assert!((screen.arrived(grid) - expected).abs() < 1e-12, "{} vs {expected}", screen.arrived(grid));
        assert!((screen.face_x(grid) - 0.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "outside the grid")]
    fn a_screen_off_the_grid_is_refused() {
        let _ = Detector::new("s", Grid2d::new(8, 8, [1.0, 1.0]), 1.5);
    }
}
