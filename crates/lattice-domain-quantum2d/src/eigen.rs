//! Stationary states by imaginary-time propagation.
//!
//! Spec §13.1: *"imaginary-time eigenstate search"*. Propagating in imaginary time,
//! `ψ ← e^{−HΔτ/ħ}ψ`, multiplies each eigencomponent by `e^{−E_nΔτ/ħ}`, so after
//! renormalizing the lowest state wins. A *block* of trial states, re-orthonormalized
//! after every step, converges to the lowest `count` states together — which is what
//! makes degenerate levels (the 2D oscillator's `n + 1`-fold shells) come out as a
//! basis of the shell rather than one arbitrary member of it.
//!
//! The propagator is the domain's own, split-step or Crank–Nicolson, so the states
//! found are the eigenstates of the *discretized* Hamiltonian being solved — the ones
//! a real-time run started from them will hold still.
//!
//! # What converges, and how fast
//!
//! Component `m` of trial state `n` decays relative to the target as
//! `e^{−(E_m − E_n)Δτ/ħ}`, so the block converges at the rate of the gap between its
//! highest state and the first one outside it. A block that ends partway through a
//! degenerate shell has no gap there and does not converge in that shell; choose
//! `count` to close shells.
//!
//! The split-step form introduces an O(Δτ²) splitting error into the *state*. The
//! energy is the Rayleigh quotient `⟨ψ|H|ψ⟩` with the exact discrete `H`, whose error is
//! quadratic in the state's, so it is good to O(Δτ⁴).

use lattice_ir::Pcg32;

use crate::complex::Complex;
use crate::crank_nicolson::CrankNicolson;
use crate::hamiltonian::{Hamiltonian, Kinetic, Spectral};
use crate::split_step::SplitStep;
use crate::wavefunction::Wavefunction;

/// How to search.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EigenSearch {
    /// How many of the lowest states to find.
    pub count: usize,
    /// Imaginary-time step, seconds.
    pub dtau: f64,
    /// Stop when no energy has changed by more than this fraction over one check.
    pub tolerance: f64,
    /// Steps between convergence checks.
    pub check_every: usize,
    /// Give up after this many steps.
    pub max_steps: usize,
}

/// One state found.
#[derive(Clone, Debug)]
pub struct Eigenstate {
    /// `⟨ψ|H|ψ⟩`, joules.
    pub energy: f64,
    /// The normalized state.
    pub state: Wavefunction,
    /// `‖Hψ − Eψ‖ / |E|`, with the norm taken over the grid as `Σ|·|²ΔA`.
    pub residual: f64,
}

/// What a search found, and whether it finished.
#[derive(Clone, Debug)]
pub struct Spectrum {
    /// The states, lowest energy first.
    pub states: Vec<Eigenstate>,
    /// Steps taken.
    pub steps: usize,
    /// Whether the tolerance was met before `max_steps`.
    pub converged: bool,
}

enum Propagator {
    Split(SplitStep),
    Crank(CrankNicolson),
}

/// Find the lowest `search.count` eigenstates of `h`.
///
/// # Panics
///
/// On a zero count or a non-positive step.
pub fn eigenstates(h: &Hamiltonian, search: &EigenSearch) -> Spectrum {
    assert!(search.count > 0, "an eigenstate search needs at least one state");
    assert!(search.dtau > 0.0 && search.dtau.is_finite(), "imaginary-time step must be positive");
    let grid = h.grid();
    let mut propagator = match h.kinetic() {
        Kinetic::Spectral => Propagator::Split(SplitStep::new(h)),
        Kinetic::FiniteDifference => Propagator::Crank(CrankNicolson::new(h).with_tolerance(1e-13)),
    };

    // Real random trial states under a broad envelope: every eigenstate overlaps them,
    // and a real start keeps the imaginary-time solve real.
    let [lx, ly] = grid.extent();
    let centre = [grid.origin()[0] + 0.5 * lx, grid.origin()[1] + 0.5 * ly];
    let mut rng = Pcg32::seed_from_u64(0x5eed_e16e);
    let mut block: Vec<Wavefunction> = (0..search.count)
        .map(|_| {
            let noise: Vec<f64> = (0..grid.nx() * grid.ny()).map(|_| rng.normal()).collect();
            let mut k = 0;
            let mut state = Wavefunction::from_fn(grid, |_| Complex::ZERO);
            for j in 0..grid.ny() {
                for i in 0..grid.nx() {
                    let [x, y] = grid.cell_center(i, j);
                    let (u, v) = ((x - centre[0]) / (0.35 * lx), (y - centre[1]) / (0.35 * ly));
                    state.as_mut_slice()[k] = Complex::real(noise[k] * (-(u * u + v * v)).exp());
                    k += 1;
                }
            }
            state
        })
        .collect();
    orthonormalize(&mut block);

    let mut spectral = Spectral::new(grid);
    let mut energies: Vec<f64> = block.iter().map(|s| spectral.measure(h, s).energy()).collect();
    let mut steps = 0;
    let mut converged = false;
    while steps < search.max_steps {
        for _ in 0..search.check_every {
            for state in &mut block {
                match &mut propagator {
                    Propagator::Split(p) => p.step_imaginary(h, state, search.dtau),
                    Propagator::Crank(p) => {
                        p.step_imaginary(h, state, search.dtau);
                    }
                }
            }
            orthonormalize(&mut block);
            steps += 1;
        }
        let next: Vec<f64> = block.iter().map(|s| spectral.measure(h, s).energy()).collect();
        let change = next
            .iter()
            .zip(&energies)
            .map(|(a, b)| (a - b).abs() / a.abs().max(f64::MIN_POSITIVE))
            .fold(0.0, f64::max);
        energies = next;
        if change < search.tolerance {
            converged = true;
            break;
        }
    }

    // Rayleigh-Ritz within the block: rotate the converged subspace onto H's
    // eigenvectors, so a degenerate shell and its neighbours come out cleanly sorted.
    let mut states = ritz(h, &mut spectral, block);
    states.sort_by(|a, b| a.energy.total_cmp(&b.energy));
    Spectrum { states, steps, converged }
}

/// Modified Gram-Schmidt, in order.
fn orthonormalize(block: &mut [Wavefunction]) {
    for n in 0..block.len() {
        let (done, rest) = block.split_at_mut(n);
        let state = &mut rest[0];
        for previous in done.iter() {
            state.project_out(previous);
        }
        state.normalize();
    }
}

/// Diagonalize `H` within the span of `block` and return its eigenpairs.
fn ritz(h: &Hamiltonian, spectral: &mut Spectral, block: Vec<Wavefunction>) -> Vec<Eigenstate> {
    let n = block.len();
    let grid = h.grid();
    let area = grid.cell_area();
    let mut applied: Vec<Vec<Complex>> = Vec::with_capacity(n);
    for state in &block {
        let mut out = vec![Complex::ZERO; state.as_slice().len()];
        spectral.apply(h, state, &mut out);
        applied.push(out);
    }
    // The projected matrix is Hermitian; the states are real for every Hamiltonian this
    // module builds a search for, so its real part is the whole of it.
    let mut matrix = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            let value: f64 = block[a].as_slice().iter().zip(&applied[b]).map(|(x, y)| (x.conj() * *y).re).sum();
            matrix[a * n + b] = value * area;
        }
    }
    let (values, vectors) = jacobi_eigen(&mut matrix, n);

    let mut states = Vec::with_capacity(n);
    for m in 0..n {
        let mut state = Wavefunction::zeros(grid);
        let mut hpsi = vec![Complex::ZERO; state.as_slice().len()];
        for (a, (basis, applied_a)) in block.iter().zip(&applied).enumerate() {
            let coefficient = vectors[a * n + m];
            for ((s, b), (hs, ha)) in state.as_mut_slice().iter_mut().zip(basis.as_slice()).zip(hpsi.iter_mut().zip(applied_a)) {
                *s += b.scale(coefficient);
                *hs += ha.scale(coefficient);
            }
        }
        let energy = values[m];
        let residual = hpsi
            .iter()
            .zip(state.as_slice())
            .map(|(hp, p)| (*hp - p.scale(energy)).norm_sqr())
            .sum::<f64>();
        let residual = (residual * area).sqrt() / energy.abs().max(f64::MIN_POSITIVE);
        states.push(Eigenstate { energy, state, residual });
    }
    states
}

/// Eigen-decomposition of a small real symmetric matrix by cyclic Jacobi rotations.
/// Returns the eigenvalues and the eigenvectors as columns of a row-major matrix.
fn jacobi_eigen(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * n + j] * a[i * n + j]).sum();
        let scale: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum::<f64>().max(f64::MIN_POSITIVE);
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use lattice_ir::Grid2d;

    use super::*;
    use crate::potential::{Potential, Shape};

    #[test]
    fn jacobi_diagonalizes_a_symmetric_matrix() {
        let mut a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 1.0];
        let original = a.clone();
        let (values, vectors) = jacobi_eigen(&mut a, 3);
        for m in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|k| original[i * 3 + k] * vectors[k * 3 + m]).sum();
                assert!((av - values[m] * vectors[i * 3 + m]).abs() < 1e-12);
            }
        }
        let trace: f64 = values.iter().sum();
        assert!((trace - 8.0).abs() < 1e-12);
    }

    #[test]
    fn the_box_spectrum_is_the_stencils_exactly() {
        // Incommensurate sides, so no level is degenerate.
        let grid = Grid2d::new(20, 13, [1.0, 0.71]);
        let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::FiniteDifference);
        let search = EigenSearch { count: 3, dtau: 2e-3, tolerance: 1e-13, check_every: 20, max_steps: 20_000 };
        let spectrum = eigenstates(&h, &search);
        assert!(spectrum.converged, "{} steps", spectrum.steps);
        // (ħ²/2m)·4sin²(nπ/2N)/Δ² per axis, with ħ = m = 1.
        let level = |n: usize, cells: usize, d: f64| 2.0 * (n as f64 * core::f64::consts::PI / (2.0 * cells as f64)).sin().powi(2) / (d * d);
        let mut expected: Vec<f64> = (1..4)
            .flat_map(|a| (1..4).map(move |b| (a, b)))
            .map(|(a, b)| level(a, grid.nx(), grid.dx()) + level(b, grid.ny(), grid.dy()))
            .collect();
        expected.sort_by(f64::total_cmp);
        for (state, e) in spectrum.states.iter().zip(&expected) {
            assert!((state.energy - e).abs() < 1e-9 * e, "{} vs {e}", state.energy);
            assert!(state.residual < 1e-5, "residual {}", state.residual);
        }
    }

    #[test]
    fn the_oscillator_ground_state_is_half_hbar_omega_per_axis() {
        let grid = Grid2d::with_origin(48, 48, [16.0, 16.0], [-8.0, -8.0]);
        let omega = 1.0;
        let mut potential = Potential::zero(grid);
        potential.add(&Shape::Harmonic { center: [0.0, 0.0], omega, mass: 1.0 });
        let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::Spectral).with_potential(potential);
        let search = EigenSearch { count: 3, dtau: 0.02, tolerance: 1e-12, check_every: 25, max_steps: 40_000 };
        let spectrum = eigenstates(&h, &search);
        assert!(spectrum.converged);
        let expected = [1.0, 2.0, 2.0];
        for (state, e) in spectrum.states.iter().zip(expected) {
            assert!((state.energy - e).abs() < 1e-6, "{} vs {e}", state.energy);
        }
    }
}
