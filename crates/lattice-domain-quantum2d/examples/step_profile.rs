//! What one step costs, and where it goes: a 2D FFT, a bare split step, and a domain
//! advance, at five grid sizes, on one thread and split across the machine. Spec §15.6
//! targets a 512×512 complex grid at interactive or near-interactive rates.
//!
//!   cargo run --release -p lattice-domain-quantum2d --example step_profile
//!
//! Take the median of several runs on an idle machine; see docs/development.md.

use std::time::Instant;

use lattice_domain_quantum2d::*;
use lattice_ir::{Arena, Domain, Executor, Grid2d, StepContext};

fn time(steps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..steps {
        f();
    }
    t.elapsed().as_secs_f64() / steps as f64 * 1e3
}

fn main() {
    let parallel = Executor::automatic();
    let sequential = Executor::sequential();
    println!("parallel arm: {}", parallel.label());
    for (nx, ny) in [(64usize, 64usize), (128, 128), (256, 256), (512, 512), (768, 512)] {
        let grid = Grid2d::with_origin(nx, ny, [1.0, 1.0], [-0.5, -0.5]);
        let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::Spectral);
        let psi = Wavefunction::gaussian(grid, [0.0, 0.0], [0.1, 0.1], [10.0, 0.0], 1.0);
        let steps = if nx * ny > 100_000 { 20 } else { 200 };
        let mut row = format!("{nx}x{ny}:");
        let mut finals = Vec::new();
        for (label, executor) in [("1 thread", &sequential), ("pool", &parallel)] {
            let mut fft = fft::Fft2::new(nx, ny);
            let mut data = psi.as_slice().to_vec();
            let one_fft = time(steps, || fft.forward_with(&mut data, executor));

            let mut raw = psi.clone();
            let mut ssf = SplitStep::new(&h);
            let split = time(steps, || {
                ssf.step_with(&h, &mut raw, 1e-5, executor);
            });

            let mut domain = QuantumDomain::new("q", h.clone()).with_state(psi.clone());
            let mut arena = Arena::with_capacity(0);
            let mut ctx = StepContext::new(&mut arena).with_executor(executor);
            let full = time(steps, || domain.advance(1e-5, &mut ctx));
            finals.push(domain.state().clone());
            row += &format!("  [{label}] fft {one_fft:.2} ms, split step {split:.2} ms, advance {full:.2} ms");
        }
        let same = finals[0].as_slice().iter().zip(finals[1].as_slice()).all(|(a, b)| {
            a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
        });
        println!("{row}  bit-identical: {same}");
    }
}
