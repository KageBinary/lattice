//! What one step costs, and where it goes: a 2D FFT, a bare split step, and a domain
//! advance, at three grid sizes. Spec §15.6 targets a 512×512 complex grid at
//! interactive or near-interactive rates.
//!
//!   cargo run --release -p lattice-domain-quantum2d --example step_profile
//!
//! Take the median of several runs on an idle machine; see docs/development.md.

use std::time::Instant;
use lattice_domain_quantum2d::*;
use lattice_ir::{Arena, Domain, Grid2d, StepContext};
fn main() {
    for (nx, ny) in [(128usize, 64usize), (256, 256), (512, 512)] {
        let grid = Grid2d::with_origin(nx, ny, [1.0, 1.0], [-0.5, -0.5]);
        let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::Spectral);
        let psi = Wavefunction::gaussian(grid, [0.0, 0.0], [0.1, 0.1], [10.0, 0.0], 1.0);
        let mut raw = psi.clone();
        let mut ssf = SplitStep::new(&h);
        let steps = 20;
        let t = Instant::now();
        for _ in 0..steps { ssf.step(&h, &mut raw, 1e-5); }
        let split = t.elapsed().as_secs_f64() / steps as f64;
        let mut fft = fft::Fft2::new(nx, ny);
        let mut data = psi.as_slice().to_vec();
        let t = Instant::now();
        for _ in 0..steps { fft.forward(&mut data); }
        let one_fft = t.elapsed().as_secs_f64() / steps as f64;
        let mut domain = QuantumDomain::new("q", h).with_state(psi);
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        let t = Instant::now();
        for _ in 0..steps { domain.advance(1e-5, &mut ctx); }
        let full = t.elapsed().as_secs_f64() / steps as f64;
        println!("{nx}x{ny}: one 2D FFT {:.2} ms, split step {:.2} ms, domain advance {:.2} ms", one_fft * 1e3, split * 1e3, full * 1e3);
    }
}
