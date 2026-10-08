//! The state: one particle's complex wavefunction on a cell-centred grid.
//!
//! `ψ` is stored row-major, `ψ[j·nx + i]` at the centre of cell `(i, j)`, with no halo:
//! the split-step propagator is periodic by construction and the Crank–Nicolson one
//! applies its walls through the stencil. `|ψ|²` is a probability per unit *area*, so
//! the norm is `Σ |ψ|² ΔA` and is 1 for a particle that is certainly somewhere on the
//! grid.

use lattice_ir::Grid2d;

use crate::complex::Complex;
use crate::hamiltonian::{Hamiltonian, Kinetic};

/// A wavefunction on a grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Wavefunction {
    grid: Grid2d,
    psi: Vec<Complex>,
}

impl Wavefunction {
    /// `ψ = 0` everywhere — no particle at all.
    pub fn zeros(grid: Grid2d) -> Self {
        Self { grid, psi: vec![Complex::ZERO; grid.nx() * grid.ny()] }
    }

    /// `ψ(r)` sampled at each cell centre.
    pub fn from_fn(grid: Grid2d, f: impl Fn([f64; 2]) -> Complex) -> Self {
        let mut psi = Vec::with_capacity(grid.nx() * grid.ny());
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                psi.push(f(grid.cell_center(i, j)));
            }
        }
        Self { grid, psi }
    }

    /// A Gaussian wave packet, normalized on the grid.
    ///
    /// ```text
    ///   ψ(r) ∝ exp(−(x−x₀)²/4σx² − (y−y₀)²/4σy²) · e^{i p·r/ħ}
    /// ```
    ///
    /// `sigma` is the standard deviation of *position* — the width of `|ψ|²` — so the
    /// momentum spread is `ħ/2σ` on each axis and the packet is minimum-uncertainty.
    /// The discrete sum is normalized rather than the continuum integral, so the norm
    /// starts at 1 to round-off however coarse the grid.
    ///
    /// # Panics
    ///
    /// On a non-positive width, or a packet that is zero on every cell centre.
    pub fn gaussian(grid: Grid2d, center: [f64; 2], sigma: [f64; 2], momentum: [f64; 2], hbar: f64) -> Self {
        assert!(sigma[0] > 0.0 && sigma[1] > 0.0, "a wave packet needs a positive width, got {sigma:?}");
        let mut packet = Self::from_fn(grid, |[x, y]| {
            let (dx, dy) = (x - center[0], y - center[1]);
            let envelope = (-(dx * dx) / (4.0 * sigma[0] * sigma[0]) - (dy * dy) / (4.0 * sigma[1] * sigma[1])).exp();
            Complex::from_polar(envelope, (momentum[0] * x + momentum[1] * y) / hbar)
        });
        let norm = packet.normalize();
        assert!(norm > 0.0, "the packet at {center:?} has no amplitude on the grid");
        packet
    }

    /// The grid.
    pub fn grid(&self) -> Grid2d {
        self.grid
    }

    /// The amplitudes, row-major.
    pub fn as_slice(&self) -> &[Complex] {
        &self.psi
    }

    /// The amplitudes, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [Complex] {
        &mut self.psi
    }

    /// The amplitude at cell `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> Complex {
        self.psi[j * self.grid.nx() + i]
    }

    /// `Σ |ψ|² ΔA`.
    pub fn norm(&self) -> f64 {
        self.psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * self.grid.cell_area()
    }

    /// Scale to unit norm, returning the norm it had. A zero state is left alone.
    pub fn normalize(&mut self) -> f64 {
        let norm = self.norm();
        if norm > 0.0 {
            let scale = 1.0 / norm.sqrt();
            for z in &mut self.psi {
                *z = z.scale(scale);
            }
        }
        norm
    }

    /// `⟨self|other⟩ = Σ conj(ψ_a) ψ_b ΔA`.
    ///
    /// # Panics
    ///
    /// If the two live on different grids.
    pub fn inner(&self, other: &Wavefunction) -> Complex {
        assert_eq!(self.grid, other.grid, "an inner product needs both states on one grid");
        let sum = self.psi.iter().zip(&other.psi).fold(Complex::ZERO, |acc, (a, b)| acc + a.conj() * *b);
        sum.scale(self.grid.cell_area())
    }

    /// Subtract the projection onto a normalized state: `ψ ← ψ − ⟨φ|ψ⟩ φ`.
    pub fn project_out(&mut self, state: &Wavefunction) {
        let overlap = state.inner(self);
        for (z, phi) in self.psi.iter_mut().zip(&state.psi) {
            *z -= *phi * overlap;
        }
    }

    /// `⟨x⟩` and `⟨y⟩`, metres, for the state as it is (not renormalized).
    ///
    /// Divided by the current norm, so a packet half-absorbed by the boundary reports
    /// where the half that remains *is* rather than a position pulled toward the origin.
    pub fn mean_position(&self) -> [f64; 2] {
        let mut weight = 0.0;
        let mut sum = [0.0; 2];
        for j in 0..self.grid.ny() {
            for i in 0..self.grid.nx() {
                let p = self.psi[j * self.grid.nx() + i].norm_sqr();
                let [x, y] = self.grid.cell_center(i, j);
                weight += p;
                sum[0] += p * x;
                sum[1] += p * y;
            }
        }
        if weight > 0.0 { [sum[0] / weight, sum[1] / weight] } else { [f64::NAN, f64::NAN] }
    }

    /// The probability on the cells whose centres satisfy `inside`.
    pub fn probability_where(&self, inside: impl Fn([f64; 2]) -> bool) -> f64 {
        let mut total = 0.0;
        for j in 0..self.grid.ny() {
            for i in 0..self.grid.nx() {
                if inside(self.grid.cell_center(i, j)) {
                    total += self.psi[j * self.grid.nx() + i].norm_sqr();
                }
            }
        }
        total * self.grid.cell_area()
    }

    /// The probability current `J = (ħ/m) Im(ψ* ∇ψ)` at every cell centre, as
    /// row-major `[J_x, J_y]`, 1/(m·s).
    ///
    /// The gradient is the central difference, so at a cell the current is the mean of
    /// the two face currents `(ħ/mΔ) Im(ψᵢ* ψᵢ₊₁)` either side of it — the current that
    /// satisfies the five-point Hamiltonian's discrete continuity equation, which is
    /// what the detectors integrate. The edges follow `h`: periodic for the spectral
    /// operator, and the antisymmetric ghost `ψ = 0` walls for the finite-difference
    /// one, which carry no current through them.
    pub fn probability_current(&self, h: &Hamiltonian) -> [Vec<f64>; 2] {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let mut current = [vec![0.0; nx * ny], vec![0.0; nx * ny]];
        for (axis, out) in current.iter_mut().enumerate() {
            for (j, row) in out.chunks_exact_mut(nx).enumerate() {
                self.current_row(h, axis, j, row);
            }
        }
        current
    }

    /// One row of one component of [`Wavefunction::probability_current`].
    pub(crate) fn current_row(&self, h: &Hamiltonian, axis: usize, j: usize, out: &mut [f64]) {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let periodic = h.kinetic() == Kinetic::Spectral;
        let (cells, spacing) = if axis == 0 { (nx, self.grid.dx()) } else { (ny, self.grid.dy()) };
        let scale = h.hbar() / (2.0 * h.mass() * spacing);
        for (i, slot) in out.iter_mut().enumerate() {
            let at = |n: usize| if axis == 0 { self.psi[j * nx + n] } else { self.psi[n * nx + i] };
            let n = if axis == 0 { i } else { j };
            let centre = at(n);
            let before = match n {
                0 if periodic => at(cells - 1),
                0 => -centre,
                _ => at(n - 1),
            };
            let after = match n + 1 {
                m if m < cells => at(m),
                _ if periodic => at(0),
                _ => -centre,
            };
            *slot = scale * (centre.conj() * (after - before)).im;
        }
    }

    /// The first cell whose amplitude is not finite.
    pub fn first_non_finite(&self) -> Option<(usize, usize)> {
        let nx = self.grid.nx();
        self.psi.iter().position(|z| !z.is_finite()).map(|k| (k % nx, k / nx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HBAR: f64 = 1.0;

    fn grid() -> Grid2d {
        Grid2d::with_origin(128, 96, [20.0, 15.0], [-10.0, -7.5])
    }

    #[test]
    fn a_gaussian_is_normalized_and_centred_where_asked() {
        let packet = Wavefunction::gaussian(grid(), [-2.0, 1.0], [1.2, 0.8], [3.0, 0.0], HBAR);
        assert!((packet.norm() - 1.0).abs() < 1e-14);
        let [x, y] = packet.mean_position();
        // Sampling at cell centres is spectrally accurate but not exact: ~1e-10 here.
        assert!((x + 2.0).abs() < 1e-8 && (y - 1.0).abs() < 1e-8, "{x} {y}");
    }

    #[test]
    fn the_density_has_the_declared_width() {
        let sigma = 1.3;
        let packet = Wavefunction::gaussian(grid(), [0.0, 0.0], [sigma, sigma], [0.0, 0.0], HBAR);
        let g = packet.grid();
        let mut second = 0.0;
        for j in 0..g.ny() {
            for i in 0..g.nx() {
                let [x, _] = g.cell_center(i, j);
                second += x * x * packet.get(i, j).norm_sqr() * g.cell_area();
            }
        }
        assert!((second.sqrt() - sigma).abs() < 1e-6, "{}", second.sqrt());
    }

    #[test]
    fn projection_removes_the_overlap() {
        let a = Wavefunction::gaussian(grid(), [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], HBAR);
        let mut b = Wavefunction::gaussian(grid(), [0.5, 0.0], [1.0, 1.0], [1.0, 0.0], HBAR);
        assert!(a.inner(&b).abs() > 0.1);
        b.project_out(&a);
        assert!(a.inner(&b).abs() < 1e-14);
    }

    #[test]
    fn a_plane_wave_carries_the_lattice_current_and_walls_carry_none() {
        let grid = Grid2d::new(32, 8, [1.0, 0.5]);
        let (mass, k) = (2.0, core::f64::consts::TAU * 3.0);
        let psi = Wavefunction::from_fn(grid, |[x, _]| Complex::cis(k * x));
        let h = Hamiltonian::new(grid, mass, HBAR, Kinetic::Spectral);
        let [jx, jy] = psi.probability_current(&h);
        // |ψ|² = 1, and the central difference of e^{ikx} gives (ħ/m)·sin(kΔx)/Δx.
        let expected = HBAR / mass * (k * grid.dx()).sin() / grid.dx();
        assert!(jx.iter().all(|j| (j - expected).abs() < 1e-12), "{} vs {expected}", jx[0]);
        assert!(jy.iter().all(|j| j.abs() < 1e-12));

        // In a box, the outermost cells see the wall's ghost and carry only half the
        // current — none of it through the wall itself.
        let boxed = Hamiltonian::new(grid, mass, HBAR, Kinetic::FiniteDifference);
        let [jx, _] = psi.probability_current(&boxed);
        let face = HBAR / (mass * grid.dx()) * (k * grid.dx()).sin();
        assert!((jx[0] - 0.5 * face).abs() < 1e-12, "{} vs {}", jx[0], 0.5 * face);
    }

    #[test]
    fn probability_where_splits_the_norm() {
        let packet = Wavefunction::gaussian(grid(), [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], HBAR);
        let left = packet.probability_where(|[x, _]| x < 0.0);
        let right = packet.probability_where(|[x, _]| x >= 0.0);
        assert!((left + right - 1.0).abs() < 1e-14);
        assert!((left - 0.5).abs() < 1e-12, "a centred packet is symmetric: {left}");
    }
}
