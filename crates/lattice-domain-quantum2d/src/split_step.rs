//! Strang-split Fourier propagation.
//!
//! ```text
//!   ψ(t + Δt) ≈ e^{−iVΔt/2ħ} · F⁻¹ e^{−iħk²Δt/2m} F · e^{−iVΔt/2ħ} ψ(t)
//! ```
//!
//! Each factor is applied exactly — the potential is diagonal in position, the kinetic
//! operator in momentum — so each is unitary, and so is their product: the norm is
//! conserved to the round-off of two FFTs, whatever the step. What the step size buys
//! is accuracy. The neglected commutator `[T, V]` makes the scheme second order, and
//! makes the energy oscillate at O(Δt²) rather than drift, as with any symmetric
//! splitting.
//!
//! With an absorbing layer the potential half-steps carry `e^{−WΔt/2ħ}` as well, and the
//! probability each removes, `Σ|ψ|²(1 − e^{−WΔt/ħ})ΔA`, is accumulated exactly as it is
//! applied. Nothing else changes the norm.
//!
//! The same factors with `Δt → −iΔτ` give imaginary-time propagation, `e^{−HΔτ/ħ}`,
//! which damps every state relative to the ground state; see [`crate::eigen`].

use lattice_ir::{Executor, Grain};

use crate::complex::Complex;
use crate::fft::Fft2;
use crate::hamiltonian::{Hamiltonian, Kinetic};
use crate::wavefunction::Wavefunction;

/// A split-step propagator with its cached phase factors.
#[derive(Clone, Debug)]
pub struct SplitStep {
    fft: Fft2,
    /// The step the factors below were built for; NaN until the first step.
    cached: (f64, bool),
    /// `e^{−iħk²Δt/2m}` per Fourier bin.
    kinetic: Vec<Complex>,
    /// `e^{−iVΔt/2ħ} · e^{−WΔt/2ħ}` per cell.
    half: Vec<Complex>,
    /// `1 − e^{−WΔt/ħ}` per cell: the fraction a half-step removes. Empty without an
    /// absorber.
    loss: Vec<f64>,
}

impl SplitStep {
    /// A propagator for `h`.
    ///
    /// # Panics
    ///
    /// If `h` uses the finite-difference kinetic operator: a Fourier propagator can
    /// only apply the spectral one, and pairing it with walls would silently replace
    /// the box with a torus.
    pub fn new(h: &Hamiltonian) -> Self {
        assert_eq!(
            h.kinetic(),
            Kinetic::Spectral,
            "split-step Fourier propagation is periodic; a box with walls needs Crank-Nicolson"
        );
        let n = h.grid().nx() * h.grid().ny();
        Self {
            fft: Fft2::new(h.grid().nx(), h.grid().ny()),
            cached: (f64::NAN, false),
            kinetic: vec![Complex::ZERO; n],
            half: vec![Complex::ZERO; n],
            loss: Vec::new(),
        }
    }

    fn prepare(&mut self, h: &Hamiltonian, dt: f64, imaginary: bool) {
        if self.cached.0 == dt && self.cached.1 == imaginary {
            return;
        }
        let (nx, ny) = (h.grid().nx(), h.grid().ny());
        let hbar = h.hbar();
        let kinetic_scale = hbar * dt / (2.0 * h.mass());
        for j in 0..ny {
            for i in 0..nx {
                let theta = kinetic_scale * h.k_squared(i, j);
                self.kinetic[j * nx + i] = if imaginary { Complex::real((-theta).exp()) } else { Complex::cis(-theta) };
            }
        }
        let absorber = h.absorber().filter(|_| !imaginary);
        for (k, v) in h.potential().values().iter().enumerate() {
            let theta = v * dt / (2.0 * hbar);
            let decay = absorber.map_or(1.0, |a| (-a.profile()[k] * dt / (2.0 * hbar)).exp());
            self.half[k] = if imaginary { Complex::real((-theta).exp()) } else { Complex::from_polar(decay, -theta) };
        }
        self.loss = match absorber {
            Some(a) => a.profile().iter().map(|w| -(-w * dt / hbar).exp_m1()).collect(),
            None => Vec::new(),
        };
        self.cached = (dt, imaginary);
    }

    /// Advance `psi` by `dt` seconds, returning the probability the absorber removed.
    pub fn step(&mut self, h: &Hamiltonian, psi: &mut Wavefunction, dt: f64) -> f64 {
        self.step_with(h, psi, dt, Executor::shared_sequential())
    }

    /// [`SplitStep::step`] split across `executor`, with the same bits.
    ///
    /// The transforms and the phase multiplies are split; both are exact under any
    /// partition. The absorbed total is a sum, which `docs/execution.md` keeps on the
    /// calling thread so that its bits do not depend on the machine.
    pub fn step_with(&mut self, h: &Hamiltonian, psi: &mut Wavefunction, dt: f64, executor: &Executor) -> f64 {
        self.prepare(h, dt, false);
        let area = h.grid().cell_area();
        let mut absorbed = self.potential_half(psi, executor);
        self.fft.forward_with(psi.as_mut_slice(), executor);
        multiply(psi.as_mut_slice(), &self.kinetic, executor);
        self.fft.inverse_with(psi.as_mut_slice(), executor);
        absorbed += self.potential_half(psi, executor);
        absorbed * area
    }

    /// One half-step of the potential, returning `Σ|ψ|²·loss` before it is applied.
    fn potential_half(&self, psi: &mut Wavefunction, executor: &Executor) -> f64 {
        let removed = psi.as_slice().iter().zip(&self.loss).fold(0.0, |sum, (z, loss)| sum + z.norm_sqr() * loss);
        multiply(psi.as_mut_slice(), &self.half, executor);
        removed
    }

    /// Advance `psi` by `dtau` seconds of imaginary time, `ψ ← e^{−HΔτ/ħ}ψ`, without
    /// renormalizing. The absorber plays no part.
    pub fn step_imaginary(&mut self, h: &Hamiltonian, psi: &mut Wavefunction, dtau: f64) {
        self.prepare(h, dtau, true);
        for (z, f) in psi.as_mut_slice().iter_mut().zip(&self.half) {
            *z *= *f;
        }
        self.fft.forward(psi.as_mut_slice());
        for (z, f) in psi.as_mut_slice().iter_mut().zip(&self.kinetic) {
            *z *= *f;
        }
        self.fft.inverse(psi.as_mut_slice());
        for (z, f) in psi.as_mut_slice().iter_mut().zip(&self.half) {
            *z *= *f;
        }
    }
}

/// How many cells of `z *= f` are worth a dispatch. A complex multiply is a particle
/// update's order of work, and the floor is the particle integrator's.
const PHASE_GRAIN: Grain = Grain::new(65_536, 8_192);

/// `psi[k] *= factors[k]`, in chunks across `executor`.
fn multiply(psi: &mut [Complex], factors: &[Complex], executor: &Executor) {
    executor.for_each_chunk_mut(psi, PHASE_GRAIN, |offset, chunk| {
        for (z, f) in chunk.iter_mut().zip(&factors[offset..]) {
            *z *= *f;
        }
    });
}

#[cfg(test)]
mod tests {
    use lattice_ir::Grid2d;

    use super::*;
    use crate::potential::{Absorber, Potential, Shape};

    fn grid() -> Grid2d {
        Grid2d::with_origin(64, 64, [20.0, 20.0], [-10.0, -10.0])
    }

    #[test]
    fn the_norm_is_conserved_to_round_off_through_a_barrier() {
        let mut potential = Potential::zero(grid());
        potential.add(&Shape::Rectangle { x: [1.0, 2.0], y: [-10.0, 10.0], height: 3.0 });
        let h = Hamiltonian::new(grid(), 1.0, 1.0, Kinetic::Spectral).with_potential(potential);
        let mut psi = Wavefunction::gaussian(grid(), [-3.0, 0.0], [1.0, 1.0], [2.0, 0.0], 1.0);
        let mut propagator = SplitStep::new(&h);
        for _ in 0..400 {
            assert_eq!(propagator.step(&h, &mut psi, 0.01), 0.0);
        }
        assert!((psi.norm() - 1.0).abs() < 1e-12, "{:e}", psi.norm() - 1.0);
    }

    #[test]
    fn what_the_absorber_removes_is_exactly_what_the_norm_loses() {
        let h = Hamiltonian::new(grid(), 1.0, 1.0, Kinetic::Spectral)
            .with_absorber(Absorber::for_speed(grid(), 3.0, 3.0, 1.0));
        let mut psi = Wavefunction::gaussian(grid(), [3.0, 0.0], [1.0, 1.0], [3.0, 0.0], 1.0);
        let mut propagator = SplitStep::new(&h);
        let mut absorbed = 0.0;
        for _ in 0..600 {
            absorbed += propagator.step(&h, &mut psi, 0.01);
        }
        assert!(absorbed > 0.9, "the packet ran into the layer: absorbed {absorbed}");
        assert!((psi.norm() + absorbed - 1.0).abs() < 1e-12, "{:e}", psi.norm() + absorbed - 1.0);
    }

    #[test]
    fn a_free_packet_moves_at_its_group_velocity() {
        let (mass, p) = (2.0, 3.0);
        let h = Hamiltonian::new(grid(), mass, 1.0, Kinetic::Spectral);
        let mut psi = Wavefunction::gaussian(grid(), [-4.0, 0.0], [1.0, 1.0], [p, 0.0], 1.0);
        let mut propagator = SplitStep::new(&h);
        for _ in 0..200 {
            propagator.step(&h, &mut psi, 0.01);
        }
        let [x, y] = psi.mean_position();
        // Ehrenfest is exact for the continuum; the cell-centre position operator
        // measures it to ~1e-8 at this resolution.
        assert!((x - (-4.0 + p / mass * 2.0)).abs() < 1e-7 && y.abs() < 1e-12, "{x} {y}");
    }

    #[test]
    #[should_panic(expected = "needs Crank-Nicolson")]
    fn a_box_is_refused() {
        let h = Hamiltonian::new(grid(), 1.0, 1.0, Kinetic::FiniteDifference);
        let _ = SplitStep::new(&h);
    }
}
