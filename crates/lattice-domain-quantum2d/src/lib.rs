//! `quantum2d`: one particle's wavefunction on a 2D grid.
//!
//! Spec §13.1:
//!
//! > The built-in quantum module should solve controlled low-dimensional problems that
//! > are visually compelling and numerically verifiable. Its central state is a complex
//! > wavefunction on a grid with a user-defined potential. Initial capabilities include
//! > imaginary-time eigenstate search, real-time wave-packet propagation, barriers and
//! > wells, interference, tunneling, expectation values, probability current, and
//! > measurement-inspired sampling. It should not present 2D single-particle quantum
//! > mechanics as a general molecular electronic-structure model.
//!
//! It solves
//!
//! ```text
//!   iħ ∂ψ/∂t = −(ħ²/2m)∇²ψ + Vψ
//! ```
//!
//! by either of the two methods §13.1's table names:
//!
//! | Scheme | Kinetic operator | Edges | Conserves |
//! |---|---|---|---|
//! | [`Scheme::SplitStepFourier`] | spectral, exact per mode | periodic | norm to round-off |
//! | [`Scheme::CrankNicolson`] | five-point finite difference | `ψ = 0` walls | norm and `⟨H⟩` to the solver tolerance |
//!
//! and either can carry a complex absorbing layer ([`Absorber`]) so that a packet
//! leaving the region of interest is removed, with the removed probability accounted
//! for exactly. [`eigen::eigenstates`] finds stationary states by imaginary-time
//! propagation with the same discretization.
//!
//! The FFT the split-step scheme runs on is [`fft`], written here rather than taken
//! from a crate for spec §24.1's reasons: it is the hot loop, and the radix-2 and
//! Bluestein paths that cover every grid size are a few hundred lines.
//!
//! # What this module is not
//!
//! Spec §5.3 draws the line, and every contract repeats it: this is a single particle in
//! a given potential. It visualizes tunneling, interference and quantized levels. It is
//! not electronic structure and says nothing about molecules.
//!
//! # Example
//!
//! ```
//! use lattice_domain_quantum2d::{Hamiltonian, Kinetic, QuantumDomain, Wavefunction};
//! use lattice_ir::{Arena, Domain, Grid2d, StepContext};
//!
//! // Natural units: ħ = m = 1.
//! let grid = Grid2d::with_origin(64, 64, [20.0, 20.0], [-10.0, -10.0]);
//! let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::Spectral);
//! let packet = Wavefunction::gaussian(grid, [-3.0, 0.0], [1.0, 1.0], [2.0, 0.0], 1.0);
//! let mut domain = QuantumDomain::new("q", h).with_state(packet);
//!
//! let mut arena = Arena::with_capacity(0);
//! let mut ctx = StepContext::new(&mut arena);
//! for _ in 0..100 {
//!     domain.advance(0.01, &mut ctx);
//! }
//! // Unitary: the norm holds to round-off.
//! assert!((domain.state().norm() - 1.0).abs() < 1e-12);
//! // Ehrenfest: the centre moves at p/m.
//! let [x, _] = domain.state().mean_position();
//! assert!((x - (-3.0 + 2.0 * 1.0)).abs() < 1e-6);
//! ```

pub mod complex;
mod crank_nicolson;
mod detector;
mod domain;
pub mod eigen;
pub mod fft;
mod hamiltonian;
mod potential;
mod split_step;
mod wavefunction;

pub use complex::Complex;
pub use crank_nicolson::{CrankNicolson, Solve};
pub use detector::Detector;
pub use domain::{QuantumDomain, Scheme, SPLIT_STEP_PHASE};
pub use eigen::{EigenSearch, Eigenstate, Spectrum};
pub use hamiltonian::{Hamiltonian, Kinetic, Moments, Spectral};
pub use potential::{Absorber, Potential, Shape};
pub use split_step::SplitStep;
pub use wavefunction::Wavefunction;

/// The reduced Planck constant, J·s.
pub const HBAR: f64 = lattice_units::constants::value::REDUCED_PLANCK;
