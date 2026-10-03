//! `H = −ħ²∇²/2m + V`, in the two discretizations the propagators use.
//!
//! The kinetic operator is where the two schemes differ, and an observable has to use
//! the *same* one as the propagator or it measures a different Hamiltonian from the one
//! being solved:
//!
//! - **Spectral** — `−∇² → |k|²` in Fourier space. Exact for every mode the grid can
//!   represent, and periodic by construction. The split-step propagator's.
//! - **Finite difference** — the five-point Laplacian with walls at the grid's outer
//!   faces, where `ψ = 0`. Second-order accurate. Crank–Nicolson's.
//!
//! # Where the wall is
//!
//! The grid is cell-centred, so the outer faces are half a cell beyond the outermost
//! centres. The wall is put *on the face* by an antisymmetric ghost, `ψ₋₁ = −ψ₀`, which
//! makes the box exactly as wide as the grid's extent: its discrete eigenvalues are
//!
//! ```text
//!   E = (2ħ²/m) Σ_axes sin²(nπ / 2N) / Δ²,    n = 1, 2, …
//! ```
//!
//! and tend to the continuum `ħ²π²n²/2mL²` at second order — both of which the
//! validation suite checks.

use lattice_ir::Grid2d;

use crate::complex::Complex;
use crate::fft::{wavenumbers, Fft2};
use crate::potential::{Absorber, Potential};
use crate::wavefunction::Wavefunction;

/// How the kinetic operator is discretized, and with it what the edges are.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kinetic {
    /// `|k|²` in Fourier space; the grid is periodic.
    Spectral,
    /// The five-point Laplacian; the grid is a box with `ψ = 0` on its outer faces.
    FiniteDifference,
}

impl Kinetic {
    /// Short name for reports.
    pub const fn name(self) -> &'static str {
        match self {
            Kinetic::Spectral => "spectral",
            Kinetic::FiniteDifference => "finite_difference",
        }
    }
}

/// A single particle's Hamiltonian on a grid.
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    grid: Grid2d,
    mass: f64,
    hbar: f64,
    kinetic: Kinetic,
    potential: Potential,
    absorber: Option<Absorber>,
    /// Angular wavenumbers in FFT order, rad/m.
    kx: Vec<f64>,
    ky: Vec<f64>,
}

impl Hamiltonian {
    /// A free particle of `mass` kilograms.
    ///
    /// # Panics
    ///
    /// On a non-positive mass or `ħ`.
    pub fn new(grid: Grid2d, mass: f64, hbar: f64, kinetic: Kinetic) -> Self {
        assert!(mass > 0.0 && mass.is_finite(), "a particle needs a positive mass, got {mass}");
        assert!(hbar > 0.0 && hbar.is_finite(), "ħ must be positive, got {hbar}");
        let [lx, ly] = grid.extent();
        Self {
            grid,
            mass,
            hbar,
            kinetic,
            potential: Potential::zero(grid),
            absorber: None,
            kx: wavenumbers(grid.nx(), lx),
            ky: wavenumbers(grid.ny(), ly),
        }
    }

    /// Replace the potential.
    ///
    /// # Panics
    ///
    /// If it lives on a different grid.
    pub fn with_potential(mut self, potential: Potential) -> Self {
        assert_eq!(potential.grid(), self.grid, "the potential must live on the Hamiltonian's grid");
        self.potential = potential;
        self
    }

    /// Add an absorbing layer, `V → V − iW`.
    pub fn with_absorber(mut self, absorber: Absorber) -> Self {
        self.absorber = Some(absorber);
        self
    }

    /// The grid.
    pub fn grid(&self) -> Grid2d {
        self.grid
    }

    /// The particle's mass, kg.
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// `ħ`, J·s.
    pub fn hbar(&self) -> f64 {
        self.hbar
    }

    /// The kinetic discretization.
    pub fn kinetic(&self) -> Kinetic {
        self.kinetic
    }

    /// The real potential.
    pub fn potential(&self) -> &Potential {
        &self.potential
    }

    /// The absorbing layer, if any.
    pub fn absorber(&self) -> Option<&Absorber> {
        self.absorber.as_ref()
    }

    /// Wavenumbers along x, FFT order, rad/m.
    pub fn kx(&self) -> &[f64] {
        &self.kx
    }

    /// Wavenumbers along y, FFT order, rad/m.
    pub fn ky(&self) -> &[f64] {
        &self.ky
    }

    /// The largest kinetic energy the grid can represent, J: `ħ²k²/2m` at the
    /// Nyquist corner for the spectral operator, the top of the five-point band for
    /// the finite-difference one.
    pub fn max_kinetic(&self) -> f64 {
        let factor = self.hbar * self.hbar / (2.0 * self.mass);
        let (cx, cy) = self.stencil_weights();
        match self.kinetic {
            // The Nyquist wavenumber is π/Δ, so k² there is π²·(1/Δ²).
            Kinetic::Spectral => factor * core::f64::consts::PI.powi(2) * (cx + cy),
            Kinetic::FiniteDifference => factor * 4.0 * (cx + cy),
        }
    }

    /// `1/Δ²` per axis, or zero for an axis one cell wide.
    ///
    /// A grid one cell high is a line, not a very narrow box: with walls a cell apart,
    /// the y direction would add a constant `(ħ²/2m)·4/Δy²` — fifteen electronvolts for a
    /// 0.1 nm row — which is harmless to an exact propagator and not to Crank–Nicolson,
    /// whose phase error grows with the energy.
    fn stencil_weights(&self) -> (f64, f64) {
        let weight = |cells: usize, d: f64| if cells > 1 { 1.0 / (d * d) } else { 0.0 };
        (weight(self.grid.nx(), self.grid.dx()), weight(self.grid.ny(), self.grid.dy()))
    }

    /// `out = −∇²ψ` by the five-point stencil with walls on the outer faces.
    pub fn negative_laplacian(&self, psi: &[Complex], out: &mut [Complex]) {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let (cx, cy) = self.stencil_weights();
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                let centre = psi[k];
                // Antisymmetric ghosts: a missing neighbour is −ψ at the centre.
                let west = if i > 0 { psi[k - 1] } else { -centre };
                let east = if i + 1 < nx { psi[k + 1] } else { -centre };
                let south = if j > 0 { psi[k - nx] } else { -centre };
                let north = if j + 1 < ny { psi[k + nx] } else { -centre };
                out[k] = (centre.scale(2.0) - west - east).scale(cx) + (centre.scale(2.0) - south - north).scale(cy);
            }
        }
    }

    /// The diagonal of `−∇²` under [`Hamiltonian::negative_laplacian`].
    pub fn negative_laplacian_diagonal(&self, i: usize, j: usize) -> f64 {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let (cx, cy) = self.stencil_weights();
        let x_count = 2.0 + f64::from(u8::from(i == 0)) + f64::from(u8::from(i + 1 == nx));
        let y_count = 2.0 + f64::from(u8::from(j == 0)) + f64::from(u8::from(j + 1 == ny));
        x_count * cx + y_count * cy
    }

    /// `out = Hψ` with the finite-difference kinetic operator. The absorber is not
    /// included: it is not part of `H`.
    pub fn apply_fd(&self, psi: &[Complex], out: &mut [Complex]) {
        self.negative_laplacian(psi, out);
        let factor = self.hbar * self.hbar / (2.0 * self.mass);
        for ((o, p), v) in out.iter_mut().zip(psi).zip(self.potential.values()) {
            *o = o.scale(factor) + p.scale(*v);
        }
    }

    /// `|k|²` of FFT bin `(i, j)`.
    #[inline]
    pub fn k_squared(&self, i: usize, j: usize) -> f64 {
        self.kx[i] * self.kx[i] + self.ky[j] * self.ky[j]
    }
}

/// Expectation values that need a Fourier transform, with the workspace to compute
/// them.
#[derive(Clone, Debug)]
pub struct Spectral {
    fft: Fft2,
    buffer: Vec<Complex>,
}

/// What [`Spectral::measure`] found.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Moments {
    /// `Σ|ψ|²ΔA`.
    pub norm: f64,
    /// `⟨p⟩ / norm`, kg·m/s.
    pub momentum: [f64; 2],
    /// `⟨T⟩ / norm`, J.
    pub kinetic: f64,
    /// `⟨V⟩ / norm`, J.
    pub potential: f64,
}

impl Moments {
    /// `⟨H⟩ / norm`, J.
    pub fn energy(&self) -> f64 {
        self.kinetic + self.potential
    }
}

impl Spectral {
    /// Workspace for a grid.
    pub fn new(grid: Grid2d) -> Self {
        Self { fft: Fft2::new(grid.nx(), grid.ny()), buffer: vec![Complex::ZERO; grid.nx() * grid.ny()] }
    }

    /// `out = Hψ` in the Hamiltonian's own discretization. The absorber is not part of
    /// `H` and is not applied.
    pub fn apply(&mut self, h: &Hamiltonian, psi: &Wavefunction, out: &mut [Complex]) {
        match h.kinetic() {
            Kinetic::FiniteDifference => h.apply_fd(psi.as_slice(), out),
            Kinetic::Spectral => {
                let grid = h.grid();
                self.buffer.copy_from_slice(psi.as_slice());
                self.fft.forward(&mut self.buffer);
                let factor = h.hbar() * h.hbar() / (2.0 * h.mass());
                for j in 0..grid.ny() {
                    for i in 0..grid.nx() {
                        let k = j * grid.nx() + i;
                        self.buffer[k] = self.buffer[k].scale(factor * h.k_squared(i, j));
                    }
                }
                self.fft.inverse(&mut self.buffer);
                for ((o, t), (p, v)) in out.iter_mut().zip(&self.buffer).zip(psi.as_slice().iter().zip(h.potential().values())) {
                    *o = *t + p.scale(*v);
                }
            }
        }
    }

    /// Norm, momentum and energy of `psi` under `h`, each per unit of remaining norm.
    ///
    /// Momentum is always measured spectrally: `⟨p⟩ = Σ ħk|φ_k|² / Σ|φ_k|²` is exact
    /// for every representable mode, whichever propagator produced the state. Kinetic
    /// energy uses the Hamiltonian's own discretization, so `⟨H⟩` is the quantity the
    /// propagator actually conserves.
    pub fn measure(&mut self, h: &Hamiltonian, psi: &Wavefunction) -> Moments {
        let grid = h.grid();
        let (nx, ny) = (grid.nx(), grid.ny());
        let norm = psi.norm();
        if norm <= 0.0 {
            return Moments::default();
        }
        let potential = h.potential().expectation(psi.as_slice().iter().map(|z| z.norm_sqr())) / norm;

        self.buffer.copy_from_slice(psi.as_slice());
        self.fft.forward(&mut self.buffer);
        let (mut weight, mut px, mut py, mut k2) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..ny {
            for i in 0..nx {
                let w = self.buffer[j * nx + i].norm_sqr();
                weight += w;
                px += w * h.kx()[i];
                py += w * h.ky()[j];
                k2 += w * h.k_squared(i, j);
            }
        }
        let hbar = h.hbar();
        let momentum = [hbar * px / weight, hbar * py / weight];
        let kinetic = match h.kinetic() {
            Kinetic::Spectral => hbar * hbar * k2 / (2.0 * h.mass() * weight),
            Kinetic::FiniteDifference => {
                h.negative_laplacian(psi.as_slice(), &mut self.buffer);
                let sum: f64 = psi.as_slice().iter().zip(&self.buffer).map(|(a, b)| (a.conj() * *b).re).sum();
                hbar * hbar * sum * grid.cell_area() / (2.0 * h.mass() * norm)
            }
        };
        Moments { norm, momentum, kinetic, potential }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_grid() -> Grid2d {
        Grid2d::new(24, 16, [2.4, 1.6])
    }

    #[test]
    fn the_box_ground_state_is_an_exact_eigenvector_of_the_stencil() {
        let grid = box_grid();
        let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::FiniteDifference);
        let [lx, ly] = grid.extent();
        let psi = Wavefunction::from_fn(grid, |[x, y]| {
            Complex::real((core::f64::consts::PI * x / lx).sin() * (core::f64::consts::PI * y / ly).sin())
        });
        let mut out = vec![Complex::ZERO; psi.as_slice().len()];
        h.apply_fd(psi.as_slice(), &mut out);
        let s = |n: usize, d: f64| (core::f64::consts::PI / (2.0 * n as f64)).sin().powi(2) / (d * d);
        let expected = 2.0 * (s(grid.nx(), grid.dx()) + s(grid.ny(), grid.dy()));
        for (o, p) in out.iter().zip(psi.as_slice()) {
            assert!((*o - p.scale(expected)).abs() < 1e-12 * expected, "{o:?} vs {:?}", p.scale(expected));
        }
    }

    #[test]
    fn the_diagonal_matches_the_stencil() {
        let grid = Grid2d::new(5, 4, [1.0, 1.0]);
        let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::FiniteDifference);
        let n = grid.nx() * grid.ny();
        let mut out = vec![Complex::ZERO; n];
        for k in 0..n {
            let mut unit = vec![Complex::ZERO; n];
            unit[k] = Complex::ONE;
            h.negative_laplacian(&unit, &mut out);
            let (i, j) = (k % grid.nx(), k / grid.nx());
            assert!((out[k].re - h.negative_laplacian_diagonal(i, j)).abs() < 1e-12, "cell ({i},{j})");
        }
    }

    #[test]
    fn a_plane_wave_has_its_momentum_and_kinetic_energy() {
        let grid = Grid2d::new(32, 32, [1.0, 1.0]);
        let h = Hamiltonian::new(grid, 2.0, 1.0, Kinetic::Spectral);
        let k = [core::f64::consts::TAU * 3.0, -core::f64::consts::TAU * 2.0];
        let psi = Wavefunction::from_fn(grid, |[x, y]| Complex::cis(k[0] * x + k[1] * y));
        let moments = Spectral::new(grid).measure(&h, &psi);
        assert!((moments.momentum[0] - k[0]).abs() < 1e-9 && (moments.momentum[1] - k[1]).abs() < 1e-9);
        let expected = (k[0] * k[0] + k[1] * k[1]) / (2.0 * 2.0);
        assert!((moments.kinetic - expected).abs() < 1e-9 * expected, "{} vs {expected}", moments.kinetic);
        assert_eq!(moments.potential, 0.0);
    }
}
