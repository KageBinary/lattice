//! Crank–Nicolson propagation in a box.
//!
//! ```text
//!   (1 + K) ψ(t + Δt) = (1 − K) ψ(t),     K = (Δt/2ħ)(iH + W)
//! ```
//!
//! with `H` the five-point finite-difference Hamiltonian and `W` the absorbing layer.
//! Without `W` this is the Cayley transform of `−iHΔt/ħ`, which is unitary: the norm and
//! the discrete energy `⟨H⟩` are both conserved, exactly in exact arithmetic and to the
//! linear solver's tolerance in practice. Spec §13.1 lists it as the alternative to
//! split-step Fourier for exactly the case this module uses it for — boundaries that
//! are not periodic.
//!
//! # The linear solve
//!
//! `1 + K` is not Hermitian, so conjugate gradient does not apply. It is *positive
//! real* — its Hermitian part `1 + (Δt/2ħ)W` is at least the identity — so every
//! eigenvalue sits at least 1 from the origin and Krylov methods converge quickly. The
//! solver is Jacobi-preconditioned BiCGSTAB, restarted on breakdown.
//!
//! COCG — conjugate gradient with the unconjugated form `xᵀy`, the textbook choice for a
//! complex-symmetric matrix — was the first implementation. It broke down on the 23rd
//! step of a 256-cell free packet: `rᵀz` passes through zero for complex vectors where
//! `r*z` cannot, the recurrence divides by it, and the iterate ran away to a norm of
//! ten. BiCGSTAB's inner products are conjugated.
//!
//! # What the absorber removes
//!
//! Writing `m = (ψ_new + ψ_old)/2`, the scheme gives `ψ_new − ψ_old = −2Km`, and
//! therefore
//!
//! ```text
//!   ‖ψ_new‖² − ‖ψ_old‖² = −4 Re⟨m|K|m⟩ = −(2Δt/ħ) ⟨m|W|m⟩
//! ```
//!
//! since `iH` contributes nothing real. That identity is used as the *measurement* of
//! what was absorbed. It holds for the exact solution, so the norm plus the absorbed
//! probability is conserved to the solver's tolerance rather than by construction —
//! which keeps the check honest.

use crate::complex::Complex;
use crate::hamiltonian::{Hamiltonian, Kinetic};
use crate::wavefunction::Wavefunction;

/// How a linear solve ended.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Solve {
    /// Iterations taken.
    pub iterations: usize,
    /// `‖r‖₂ / ‖b‖₂` at the end.
    pub residual: f64,
    /// Whether it met the tolerance.
    pub converged: bool,
}

/// A Crank–Nicolson propagator with its solver workspace.
#[derive(Clone, Debug)]
pub struct CrankNicolson {
    tolerance: f64,
    max_iterations: usize,
    rhs: Vec<Complex>,
    r: Vec<Complex>,
    shadow: Vec<Complex>,
    p: Vec<Complex>,
    v: Vec<Complex>,
    s: Vec<Complex>,
    t: Vec<Complex>,
    /// `M⁻¹p` and `M⁻¹s`.
    p_hat: Vec<Complex>,
    s_hat: Vec<Complex>,
    scratch: Vec<Complex>,
    inverse_diagonal: Vec<Complex>,
    /// `(Δt, imaginary)` the diagonal was built for.
    cached: (f64, bool),
}

/// `K = alpha·H + beta·W`.
#[derive(Clone, Copy, Debug)]
struct Coefficients {
    alpha: Complex,
    beta: f64,
}

impl CrankNicolson {
    /// The default relative residual, `1e-12`: well above `f64` round-off for the
    /// grids this module runs, and far below anything a norm diagnostic would show.
    pub const DEFAULT_TOLERANCE: f64 = 1e-12;

    /// A propagator for `h`.
    ///
    /// # Panics
    ///
    /// If `h` uses the spectral kinetic operator, which this scheme cannot apply.
    pub fn new(h: &Hamiltonian) -> Self {
        assert_eq!(
            h.kinetic(),
            Kinetic::FiniteDifference,
            "Crank-Nicolson here applies the finite-difference Hamiltonian of a box"
        );
        let n = h.grid().nx() * h.grid().ny();
        Self {
            tolerance: Self::DEFAULT_TOLERANCE,
            max_iterations: 10 * n.max(100),
            rhs: vec![Complex::ZERO; n],
            r: vec![Complex::ZERO; n],
            shadow: vec![Complex::ZERO; n],
            p: vec![Complex::ZERO; n],
            v: vec![Complex::ZERO; n],
            s: vec![Complex::ZERO; n],
            t: vec![Complex::ZERO; n],
            p_hat: vec![Complex::ZERO; n],
            s_hat: vec![Complex::ZERO; n],
            scratch: vec![Complex::ZERO; n],
            inverse_diagonal: vec![Complex::ZERO; n],
            cached: (f64::NAN, false),
        }
    }

    /// Set the relative residual the solve stops at.
    ///
    /// # Panics
    ///
    /// On a tolerance that is not positive and finite.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        assert!(tolerance > 0.0 && tolerance.is_finite(), "a solver tolerance must be positive, got {tolerance}");
        self.tolerance = tolerance;
        self
    }

    /// The relative residual the solve stops at.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn coefficients(h: &Hamiltonian, dt: f64, imaginary: bool) -> Coefficients {
        let scale = dt / (2.0 * h.hbar());
        if imaginary {
            // Backward Euler, not Crank-Nicolson: K = HΔτ/ħ, the whole step implicit.
            Coefficients { alpha: Complex::real(2.0 * scale), beta: 0.0 }
        } else {
            Coefficients { alpha: Complex::new(0.0, scale), beta: if h.absorber().is_some() { scale } else { 0.0 } }
        }
    }

    /// `out = v + sign·K v`.
    fn apply(h: &Hamiltonian, c: Coefficients, sign: f64, v: &[Complex], out: &mut [Complex]) {
        h.apply_fd(v, out);
        let absorber = h.absorber().map(|a| a.profile());
        for (k, (o, x)) in out.iter_mut().zip(v).enumerate() {
            let w = absorber.map_or(0.0, |profile| profile[k]);
            let kv = c.alpha * *o + x.scale(c.beta * w);
            *o = *x + kv.scale(sign);
        }
    }

    fn prepare(&mut self, h: &Hamiltonian, dt: f64, imaginary: bool) {
        if self.cached == (dt, imaginary) {
            return;
        }
        let c = Self::coefficients(h, dt, imaginary);
        let grid = h.grid();
        let factor = h.hbar() * h.hbar() / (2.0 * h.mass());
        let absorber = h.absorber().map(|a| a.profile());
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let k = j * grid.nx() + i;
                let h_diag = factor * h.negative_laplacian_diagonal(i, j) + h.potential().values()[k];
                let w = absorber.map_or(0.0, |profile| profile[k]);
                let diagonal = Complex::ONE + c.alpha * Complex::real(h_diag) + Complex::real(c.beta * w);
                self.inverse_diagonal[k] = Complex::ONE / diagonal;
            }
        }
        self.cached = (dt, imaginary);
    }

    /// Advance `psi` by `dt` seconds. Returns the probability the absorber removed and
    /// how the solve went.
    pub fn step(&mut self, h: &Hamiltonian, psi: &mut Wavefunction, dt: f64) -> (f64, Solve) {
        let solve = self.solve(h, psi, dt, false);
        let absorbed = match h.absorber() {
            Some(absorber) => {
                // m = (ψ_new + ψ_old)/2, and the identity in the module docs.
                // `scratch` still holds the state the solve started from.
                let sum: f64 = psi
                    .as_slice()
                    .iter()
                    .zip(&self.scratch)
                    .zip(absorber.profile())
                    .map(|((new, old), w)| (*new + *old).scale(0.5).norm_sqr() * w)
                    .sum();
                2.0 * dt / h.hbar() * sum * h.grid().cell_area()
            }
            None => 0.0,
        };
        (absorbed, solve)
    }

    /// Advance `psi` by `dtau` seconds of imaginary time, `(1 + HΔτ/ħ)ψ' = ψ`, without
    /// renormalizing.
    ///
    /// Backward Euler rather than Crank–Nicolson. The Cayley form multiplies an
    /// eigencomponent by `(1 − x)/(1 + x)` with `x = λΔτ/2ħ`, which tends to *−1* as `λ`
    /// grows: once `Δτ` is large enough to be useful, the top of the spectrum is the
    /// slowest thing to decay, and a search converges onto the grid's highest states.
    /// Backward Euler's factor `1/(1 + λΔτ/ħ)` falls monotonically, so the lowest state
    /// always wins, and its eigenvectors are still exactly those of the discrete `H`.
    pub fn step_imaginary(&mut self, h: &Hamiltonian, psi: &mut Wavefunction, dtau: f64) -> Solve {
        self.solve(h, psi, dtau, true)
    }

    fn solve(&mut self, h: &Hamiltonian, psi: &mut Wavefunction, dt: f64, imaginary: bool) -> Solve {
        self.prepare(h, dt, imaginary);
        let c = Self::coefficients(h, dt, imaginary);
        self.scratch.copy_from_slice(psi.as_slice());
        if imaginary {
            self.rhs.copy_from_slice(psi.as_slice());
        } else {
            Self::apply(h, c, -1.0, psi.as_slice(), &mut self.rhs);
        }
        let b_norm = self.rhs.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if b_norm == 0.0 {
            return Solve { iterations: 0, residual: 0.0, converged: true };
        }

        // BiCGSTAB from the old state as the initial guess.
        let x = psi.as_mut_slice();
        Self::apply(h, c, 1.0, x, &mut self.v);
        for ((r, b), ax) in self.r.iter_mut().zip(&self.rhs).zip(&self.v) {
            *r = *b - *ax;
        }
        let mut residual = norm(&self.r) / b_norm;
        let mut iterations = 0;
        let mut restart = true;
        let (mut rho, mut alpha, mut omega) = (Complex::ONE, Complex::ONE, Complex::ONE);
        while residual > self.tolerance && iterations < self.max_iterations {
            if restart {
                // A fresh shadow residual: at the start, and whenever the recurrence
                // has lost its footing.
                self.shadow.copy_from_slice(&self.r);
                self.p.fill(Complex::ZERO);
                self.v.fill(Complex::ZERO);
                (rho, alpha, omega) = (Complex::ONE, Complex::ONE, Complex::ONE);
                restart = false;
            }
            let next = dot(&self.shadow, &self.r);
            if next.abs() <= f64::MIN_POSITIVE {
                restart = true;
                if iterations > 0 {
                    continue;
                }
                break;
            }
            let beta = (next / rho) * (alpha / omega);
            rho = next;
            for ((p, r), v) in self.p.iter_mut().zip(&self.r).zip(&self.v) {
                *p = *r + beta * (*p - omega * *v);
            }
            for ((ph, p), d) in self.p_hat.iter_mut().zip(&self.p).zip(&self.inverse_diagonal) {
                *ph = *p * *d;
            }
            Self::apply(h, c, 1.0, &self.p_hat, &mut self.v);
            let denominator = dot(&self.shadow, &self.v);
            if denominator.abs() <= f64::MIN_POSITIVE {
                restart = true;
                continue;
            }
            alpha = rho / denominator;
            for ((s, r), v) in self.s.iter_mut().zip(&self.r).zip(&self.v) {
                *s = *r - alpha * *v;
            }
            iterations += 1;
            if norm(&self.s) / b_norm <= self.tolerance {
                for (xi, ph) in x.iter_mut().zip(&self.p_hat) {
                    *xi += alpha * *ph;
                }
                self.r.copy_from_slice(&self.s);
                residual = norm(&self.r) / b_norm;
                break;
            }
            for ((sh, s), d) in self.s_hat.iter_mut().zip(&self.s).zip(&self.inverse_diagonal) {
                *sh = *s * *d;
            }
            Self::apply(h, c, 1.0, &self.s_hat, &mut self.t);
            let tt = dot(&self.t, &self.t).re;
            omega = if tt > 0.0 { dot(&self.t, &self.s) / Complex::real(tt) } else { Complex::ZERO };
            for ((xi, ph), sh) in x.iter_mut().zip(&self.p_hat).zip(&self.s_hat) {
                *xi += alpha * *ph + omega * *sh;
            }
            for ((r, s), t) in self.r.iter_mut().zip(&self.s).zip(&self.t) {
                *r = *s - omega * *t;
            }
            residual = norm(&self.r) / b_norm;
            if omega.abs() <= f64::MIN_POSITIVE {
                restart = true;
            }
        }
        Solve { iterations, residual, converged: residual <= self.tolerance }
    }
}

/// `x*y`, conjugating the first argument.
fn dot(x: &[Complex], y: &[Complex]) -> Complex {
    x.iter().zip(y).fold(Complex::ZERO, |acc, (a, b)| acc + a.conj() * *b)
}

/// `‖x‖₂`.
fn norm(x: &[Complex]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}
#[cfg(test)]
mod tests {
    use lattice_ir::Grid2d;

    use super::*;
    use crate::hamiltonian::Spectral;
    use crate::potential::{Absorber, Potential, Shape};

    fn grid() -> Grid2d {
        Grid2d::with_origin(48, 40, [12.0, 10.0], [-6.0, -5.0])
    }

    #[test]
    fn norm_and_energy_are_conserved_to_the_solver_tolerance() {
        let mut potential = Potential::zero(grid());
        potential.add(&Shape::Rectangle { x: [1.0, 2.0], y: [-5.0, 5.0], height: 2.0 });
        let h = Hamiltonian::new(grid(), 1.0, 1.0, Kinetic::FiniteDifference).with_potential(potential);
        let mut psi = Wavefunction::gaussian(grid(), [-2.0, 0.0], [1.0, 1.0], [1.5, 0.0], 1.0);
        let mut spectral = Spectral::new(grid());
        let e0 = spectral.measure(&h, &psi).energy();
        let mut propagator = CrankNicolson::new(&h);
        for _ in 0..100 {
            let (absorbed, solve) = propagator.step(&h, &mut psi, 0.02);
            assert_eq!(absorbed, 0.0);
            assert!(solve.converged, "{solve:?}");
        }
        assert!((psi.norm() - 1.0).abs() < 1e-10, "{:e}", psi.norm() - 1.0);
        let e = spectral.measure(&h, &psi).energy();
        assert!((e - e0).abs() < 1e-9 * e0.abs(), "{e} vs {e0}");
    }

    #[test]
    fn the_absorbed_probability_balances_the_norm() {
        let h = Hamiltonian::new(grid(), 1.0, 1.0, Kinetic::FiniteDifference)
            .with_absorber(Absorber::for_speed(grid(), 2.5, 2.0, 1.0));
        let mut psi = Wavefunction::gaussian(grid(), [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], 1.0);
        let mut propagator = CrankNicolson::new(&h);
        let mut absorbed = 0.0;
        for _ in 0..200 {
            let (taken, solve) = propagator.step(&h, &mut psi, 0.02);
            assert!(solve.converged);
            absorbed += taken;
        }
        assert!(absorbed > 0.5, "absorbed {absorbed}");
        assert!((psi.norm() + absorbed - 1.0).abs() < 1e-9, "{:e}", psi.norm() + absorbed - 1.0);
    }
}
