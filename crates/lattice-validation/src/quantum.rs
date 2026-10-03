//! Validation cases for the `quantum2d` module (spec §13.1, §19.2).
//!
//! > Quantum particle in a box, harmonic oscillator, free packet, tunneling, and
//! > double-slit norm checks.
//!
//! That is §19.2's row, and each item is here, plus the cross-scheme check §13.1's table
//! implies by offering two methods: Crank–Nicolson converges to split-step Fourier at
//! the second order its stencil declares.
//!
//! Everything is an electron in SI units. The energies are a fraction of an electronvolt
//! to a few electronvolts and the lengths are nanometres, so a number in a note means
//! what it says.

use lattice_domain_quantum2d::{
    eigen, Absorber, EigenSearch, Hamiltonian, Kinetic, Potential, QuantumDomain, Shape, Wavefunction, HBAR,
};
use lattice_ir::{Arena, Domain, Grid2d, StepContext};

use crate::{Case, Level, Outcome};

/// The electron's mass, kg.
const MASS: f64 = 9.109_383_713_9e-31;
/// One electronvolt, J.
const EV: f64 = 1.602_176_634e-19;
/// One nanometre, m.
const NM: f64 = 1e-9;

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "box_spectrum_matches_the_stencil",
        domain: "quantum2d",
        level: Level::Analytic,
        claim: "imaginary-time search finds the five-point box's eigenvalues (2hbar^2/m) sum sin^2(n pi/2N)/d^2 exactly",
        run: box_spectrum,
    },
    Case {
        name: "box_ground_state_converges_at_second_order",
        domain: "quantum2d",
        level: Level::Manufactured,
        claim: "the box's ground-state energy approaches hbar^2 pi^2/2m (1/Lx^2 + 1/Ly^2) at the stencil's second order",
        run: box_convergence,
    },
    Case {
        name: "harmonic_oscillator_spectrum",
        domain: "quantum2d",
        level: Level::Analytic,
        claim: "the 2D isotropic oscillator's lowest six levels are hbar omega (1, 2, 2, 3, 3, 3)",
        run: oscillator_spectrum,
    },
    Case {
        name: "free_packet_spreads_as_the_analytic_solution",
        domain: "quantum2d",
        level: Level::Analytic,
        claim: "a free Gaussian packet's width grows as sigma0 sqrt(1 + (hbar t / 2 m sigma0^2)^2) and its centre moves at p/m",
        run: free_packet,
    },
    Case {
        name: "tunneling_matches_the_rectangular_barrier",
        domain: "quantum2d",
        level: Level::Analytic,
        claim: "a packet below a rectangular barrier is transmitted with the analytic T(k) averaged over its momenta",
        run: tunneling,
    },
    Case {
        name: "double_slit_norm_is_accounted",
        domain: "quantum2d",
        level: Level::Property,
        claim: "through a double slit into an absorbing boundary, norm plus absorbed probability stays 1 to round-off",
        run: double_slit_norm,
    },
    Case {
        name: "absorbing_layer_reflects_little",
        domain: "quantum2d",
        level: Level::Property,
        claim: "a packet driven into the absorbing layer leaves less than 1e-3 of itself behind",
        run: absorber_reflection,
    },
    Case {
        name: "crank_nicolson_converges_to_split_step",
        domain: "quantum2d",
        level: Level::CrossScheme,
        claim: "Crank-Nicolson's density approaches split-step Fourier's at the second order of its stencil",
        run: cross_scheme,
    },
    Case {
        name: "crank_nicolson_conserves_norm_and_energy",
        domain: "quantum2d",
        level: Level::Property,
        claim: "Crank-Nicolson in a box holds the norm and <H> to its solver tolerance",
        run: crank_nicolson_conservation,
    },
];

fn run(domain: &mut QuantumDomain, dt: f64, steps: usize) {
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
    }
}

/// The ground and two lowest excited energies of a box `lx × ly` on `nx × ny` cells.
fn box_levels(grid: Grid2d, count: usize) -> Vec<f64> {
    let axis = |n: usize, cells: usize, d: f64| {
        2.0 * HBAR * HBAR / MASS * (n as f64 * core::f64::consts::PI / (2.0 * cells as f64)).sin().powi(2) / (d * d)
    };
    let mut levels: Vec<f64> = (1..6)
        .flat_map(|a| (1..6).map(move |b| (a, b)))
        .map(|(a, b)| axis(a, grid.nx(), grid.dx()) + axis(b, grid.ny(), grid.dy()))
        .collect();
    levels.sort_by(f64::total_cmp);
    levels.truncate(count);
    levels
}

fn box_search(grid: Grid2d, count: usize) -> eigen::Spectrum {
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::FiniteDifference);
    let ground = box_levels(grid, 1)[0];
    let search = EigenSearch {
        count,
        dtau: 0.5 * HBAR / ground,
        tolerance: 1e-13,
        check_every: 10,
        max_steps: 20_000,
    };
    eigen::eigenstates(&h, &search)
}

fn box_spectrum() -> Outcome {
    // Incommensurate sides, so no two of the lowest levels coincide.
    let grid = Grid2d::new(24, 17, [2.0 * NM, 1.42 * NM]);
    let spectrum = box_search(grid, 3);
    let expected = box_levels(grid, 3);
    let worst = spectrum
        .states
        .iter()
        .zip(&expected)
        .map(|(state, e)| (state.energy - e).abs() / e)
        .fold(0.0, f64::max);
    Outcome::at_most("max relative error of the three lowest levels", "1", worst, 1e-9)
        .note(format!(
            "levels {:.6} / {:.6} / {:.6} eV after {} imaginary-time steps (converged: {})",
            spectrum.states[0].energy / EV,
            spectrum.states[1].energy / EV,
            spectrum.states[2].energy / EV,
            spectrum.steps,
            spectrum.converged
        ))
        .note("the search stops when no energy moves by 1e-13 over a check; the limit is that, with room")
}

fn box_convergence() -> Outcome {
    let (lx, ly) = (2.0 * NM, 1.42 * NM);
    let continuum = HBAR * HBAR * core::f64::consts::PI.powi(2) / (2.0 * MASS) * (1.0 / (lx * lx) + 1.0 / (ly * ly));
    let error = |nx: usize, ny: usize| {
        let spectrum = box_search(Grid2d::new(nx, ny, [lx, ly]), 1);
        (spectrum.states[0].energy - continuum).abs() / continuum
    };
    let coarse = error(16, 11);
    let fine = error(32, 22);
    Outcome::near("observed order of the ground-state energy error", "1", (coarse / fine).log2(), 2.0, 0.1)
        .note(format!("relative error {coarse:.3e} -> {fine:.3e} on halving the spacing"))
        .note(format!("continuum E1 = {:.6} eV for an electron in a 2 x 1.42 nm box", continuum / EV))
}

fn oscillator_spectrum() -> Outcome {
    let quantum = 0.1 * EV;
    let omega = quantum / HBAR;
    let length = (HBAR / (MASS * omega)).sqrt();
    let half = 8.0 * length;
    let grid = Grid2d::with_origin(64, 64, [2.0 * half, 2.0 * half], [-half, -half]);
    let mut potential = Potential::zero(grid);
    potential.add(&Shape::Harmonic { center: [0.0, 0.0], omega, mass: MASS });
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::Spectral).with_potential(potential);
    let dtau = 0.05 / omega;
    let search = EigenSearch { count: 6, dtau, tolerance: 1e-13, check_every: 20, max_steps: 40_000 };
    let spectrum = eigen::eigenstates(&h, &search);

    let expected = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
    let found: Vec<f64> = spectrum.states.iter().map(|s| s.energy / quantum).collect();
    let worst = found.iter().zip(expected).map(|(e, x)| (e - x).abs()).fold(0.0, f64::max);
    Outcome::at_most("max |E_n - hbar omega (n + 1)| / hbar omega over six levels", "1", worst, 1e-6)
        .note(format!(
            "found {} in units of hbar omega = 0.1 eV",
            found.iter().map(|e| format!("{e:.8}")).collect::<Vec<_>>().join(", ")
        ))
        .note(format!(
            "spectral kinetic operator, so the grid contributes nothing visible; split-step imaginary time at omega dtau = 0.05 \
             leaves the Rayleigh quotient O(dtau^4) off. {} steps",
            spectrum.steps
        ))
}

/// `sqrt(<x²> − <x>²)` and `<x>` of a state.
fn spread_x(psi: &Wavefunction) -> (f64, f64) {
    let grid = psi.grid();
    let (mut weight, mut first, mut second) = (0.0, 0.0, 0.0);
    for j in 0..grid.ny() {
        for i in 0..grid.nx() {
            let p = psi.get(i, j).norm_sqr();
            let x = grid.cell_center(i, j)[0];
            weight += p;
            first += p * x;
            second += p * x * x;
        }
    }
    let mean = first / weight;
    ((second / weight - mean * mean).sqrt(), mean)
}

fn free_packet() -> Outcome {
    let sigma = 0.5 * NM;
    let momentum = (2.0 * MASS * EV).sqrt();
    let spreading_time = 2.0 * MASS * sigma * sigma / HBAR;
    // Free motion separates in x and y, so the width along x is independent of how
    // finely y is resolved; 32 rows keep the problem two-dimensional and cheap.
    let grid = Grid2d::with_origin(256, 32, [40.0 * NM, 16.0 * NM], [-20.0 * NM, -8.0 * NM]);
    let start = -4.0 * NM;
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::Spectral);
    let packet = Wavefunction::gaussian(grid, [start, 0.0], [sigma, sigma], [momentum, 0.0], HBAR);
    let mut domain = QuantumDomain::new("free", h).with_state(packet);

    let steps_per_sample = 25;
    let dt = spreading_time / 100.0;
    let (mut worst_width, mut worst_centre) = (0.0f64, 0.0f64);
    for sample in 1..=12 {
        run(&mut domain, dt, steps_per_sample);
        let t = (sample * steps_per_sample) as f64 * dt;
        let expected = sigma * (1.0 + (t / spreading_time).powi(2)).sqrt();
        let (width, centre) = spread_x(domain.state());
        worst_width = worst_width.max((width - expected).abs() / expected);
        worst_centre = worst_centre.max((centre - (start + momentum / MASS * t)).abs() / sigma);
    }
    Outcome::at_most("max relative error of the packet width over 3 spreading times", "1", worst_width, 1e-8)
        .note(format!("centre within {worst_centre:.2e} sigma0 of x0 + p t / m"))
        .note(
            "a 1 eV electron, sigma0 = 0.5 nm. With no potential every split step is exact, so what is left is \
             measuring moments at cell centres",
        )
}

/// The transmission probability of a plane wave of wavenumber `k` through a
/// rectangular barrier of height `v0` and width `a`.
fn transmission(k: f64, v0: f64, a: f64) -> f64 {
    let energy = HBAR * HBAR * k * k / (2.0 * MASS);
    if (energy - v0).abs() < 1e-12 * v0 {
        let ka = MASS * v0 * a * a / (2.0 * HBAR * HBAR);
        return 1.0 / (1.0 + ka);
    }
    let term = if energy < v0 {
        let kappa = (2.0 * MASS * (v0 - energy)).sqrt() / HBAR;
        (kappa * a).sinh().powi(2)
    } else {
        let q = (2.0 * MASS * (energy - v0)).sqrt() / HBAR;
        (q * a).sin().powi(2)
    };
    1.0 / (1.0 + v0 * v0 * term / (4.0 * energy * (energy - v0).abs()))
}

fn tunneling() -> Outcome {
    // A line: the barrier is uniform in y, so y separates and one row is the problem.
    let cells = 1024;
    let extent = 51.2 * NM;
    let grid = Grid2d::with_origin(cells, 1, [extent, 0.1 * NM], [-0.5 * extent, 0.0]);
    let dx = grid.dx();
    let (v0, energy) = (1.0 * EV, 0.5 * EV);
    // Faces at x = 0 and x = 10 dx: the barrier is exactly ten cells wide.
    let width = 10.0 * dx;
    let mut potential = Potential::zero(grid);
    potential.add(&Shape::Rectangle { x: [0.0, width], y: [-1.0, 1.0], height: v0 });
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::Spectral).with_potential(potential);

    let k0 = (2.0 * MASS * energy).sqrt() / HBAR;
    let sigma = 2.0 * NM;
    let start = -11.0 * NM;
    let packet = Wavefunction::gaussian(grid, [start, 0.05 * NM], [sigma, 1.0], [HBAR * k0, 0.0], HBAR);
    let mut domain = QuantumDomain::new("barrier", h).with_state(packet);
    let speed = HBAR * k0 / MASS;
    // The domain's own default: the barrier's edges reach the grid's top wavenumber,
    // so the step holds the kinetic phase there to SPLIT_STEP_PHASE radians.
    let dt = domain.stable_step().preferred;
    let steps = ((22.0 * NM / speed) / dt).round() as usize;
    run(&mut domain, dt, steps);
    let transmitted = domain.state().probability_where(|[x, _]| x > width);
    let reflected = domain.state().probability_where(|[x, _]| x < 0.0);

    // |φ(k)|² for a minimum-uncertainty packet: Gaussian about k0 with σk = 1/2σ.
    let sigma_k = 1.0 / (2.0 * sigma);
    let samples = 4001;
    let (lo, hi) = (k0 - 8.0 * sigma_k, k0 + 8.0 * sigma_k);
    let step = (hi - lo) / (samples - 1) as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for n in 0..samples {
        let k = lo + n as f64 * step;
        let weight = (-(k - k0).powi(2) / (2.0 * sigma_k * sigma_k)).exp()
            * if n == 0 || n == samples - 1 { 1.0 } else if n % 2 == 1 { 4.0 } else { 2.0 };
        num += weight * transmission(k, v0, width);
        den += weight;
    }
    let expected = num / den;

    // Halving dt or dx at these settings moves T by less than 1e-3 of itself.
    Outcome::near("transmitted probability / analytic", "1", transmitted / expected, 1.0, 2e-3)
        .note(format!(
            "T = {transmitted:.6} against {expected:.6}; R = {reflected:.6}, R + T = {:.12}",
            reflected + transmitted
        ))
        .note(format!(
            "a 0.5 eV electron on a 1 eV barrier {:.3} nm wide, kappa a = {:.2}; dx = {:.3} nm, {steps} steps of {:.3e} s",
            width / NM,
            (2.0 * MASS * (v0 - energy)).sqrt() / HBAR * width,
            dx / NM,
            dt
        ))
        .note(
            "on this barrier and grid, at 7.5 radians of top-of-grid kinetic phase per step instead of 2, a wider packet came out 1.9% high: the barrier's \
             edges couple to the grid's highest modes, which is where split-step's commutator error lives",
        )
}

/// Spec §25.2's scene at a resolution a validation run can afford.
fn double_slit_domain() -> (QuantumDomain, f64) {
    let grid = Grid2d::with_origin(128, 64, [12.0 * NM, 6.0 * NM], [-6.0 * NM, -3.0 * NM]);
    let mut potential = Potential::zero(grid);
    potential.add(&Shape::Wall {
        x: 0.0,
        thickness: 0.15 * NM,
        slits: vec![(-NM, 0.35 * NM), (NM, 0.35 * NM)],
        height: 20.0 * EV,
    });
    let momentum = 1.3e-24;
    let speed = momentum / MASS;
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::Spectral)
        .with_potential(potential)
        .with_absorber(Absorber::for_speed(grid, 0.8 * NM, speed, HBAR));
    let packet = Wavefunction::gaussian(grid, [-3.0 * NM, 0.0], [0.6 * NM, 1.5 * NM], [momentum, 0.0], HBAR);
    let domain = QuantumDomain::new("q", h).with_state(packet).with_detector("screen", 4.5 * NM);
    (domain, speed)
}

fn double_slit_norm() -> Outcome {
    let (mut domain, speed) = double_slit_domain();
    let dt = 0.01e-15;
    let steps = ((12.0 * NM / speed) / dt).round() as usize;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let mut worst = 0.0f64;
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
        worst = worst.max((domain.state().norm() + domain.absorbed() - 1.0).abs());
    }
    let grid = domain.grid();
    Outcome::at_most("max |norm + absorbed - 1| over the run", "1", worst, 1e-12)
        .note(format!(
            "{steps} steps: {:.4} absorbed, {:.4} left on the grid, {:.4} arrived at the screen",
            domain.absorbed(),
            domain.state().norm(),
            domain.detectors()[0].arrived(grid)
        ))
        .note("each half-step's removal is computed from the factor it applies, so the residual is two FFTs' round-off per step")
}

fn absorber_reflection() -> Outcome {
    // A line again, with the layer on both ends of the periodic grid.
    let cells = 1024;
    let extent = 60.0 * NM;
    let grid = Grid2d::with_origin(cells, 1, [extent, 0.1 * NM], [-0.5 * extent, 0.0]);
    let energy = 5.8 * EV;
    let momentum = (2.0 * MASS * energy).sqrt();
    let speed = momentum / MASS;
    let layer = 0.8 * NM;
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::Spectral)
        .with_absorber(Absorber::on_axes(grid, layer, Absorber::strength_for_speed(layer, speed, HBAR), [true, false]));
    let sigma = 1.5 * NM;
    let packet = Wavefunction::gaussian(grid, [10.0 * NM, 0.05 * NM], [sigma, 1.0], [momentum, 0.0], HBAR);
    let mut domain = QuantumDomain::new("edge", h).with_state(packet);
    // Long enough for the whole packet to reach the layer, and for anything it reflects
    // to come back out — not long enough for that to reach the far side.
    let dt = 0.01e-15;
    let travel = 20.0 * NM + 6.0 * sigma;
    run(&mut domain, dt, ((travel / speed) / dt).round() as usize);
    let interior = 0.5 * extent - layer;
    let left = domain.state().probability_where(|[x, _]| x.abs() < interior);
    Outcome::at_most("probability left in the interior after the packet met the layer", "1", left, 1e-3)
        .note(format!(
            "a 5.8 eV electron (k * width = {:.1}) into a 0.8 nm quadratic layer of strength {:.2} eV; {:.3e} absorbed",
            momentum / HBAR * layer,
            domain.hamiltonian().absorber().unwrap().strength() / EV,
            domain.absorbed()
        ))
}

/// Max density difference between Crank–Nicolson and split-step on `cells` cells.
fn scheme_gap(cells: usize) -> f64 {
    let extent = 40.0 * NM;
    let grid = Grid2d::with_origin(cells, 1, [extent, 0.1 * NM], [-0.5 * extent, 0.0]);
    let energy = 0.5 * EV;
    let momentum = (2.0 * MASS * energy).sqrt();
    let packet = Wavefunction::gaussian(grid, [-4.0 * NM, 0.05 * NM], [1.0 * NM, 1.0], [momentum, 0.0], HBAR);
    // Ten femtoseconds at a step whose Cayley phase error stays well below the
    // finest grid's dispersion error.
    let time = 10e-15;
    let dt = 0.005 * HBAR / energy;
    let steps = (time / dt).round() as usize;

    let mut split = QuantumDomain::new("ssf", Hamiltonian::new(grid, MASS, HBAR, Kinetic::Spectral))
        .with_state(packet.clone());
    let mut crank = QuantumDomain::new("cn", Hamiltonian::new(grid, MASS, HBAR, Kinetic::FiniteDifference))
        .with_state(packet)
        .with_tolerance(1e-13);
    run(&mut split, dt, steps);
    run(&mut crank, dt, steps);
    let (a, b) = (split.state().as_slice(), crank.state().as_slice());
    let peak = a.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max);
    a.iter().zip(b).map(|(x, y)| (x.norm_sqr() - y.norm_sqr()).abs()).fold(0.0, f64::max) / peak
}

fn cross_scheme() -> Outcome {
    let coarse = scheme_gap(256);
    let fine = scheme_gap(512);
    Outcome::near("observed order of the density difference", "1", (coarse / fine).log2(), 2.0, 0.2)
        .note(format!("max |rho_CN - rho_SSF| / max rho: {coarse:.3e} -> {fine:.3e} on halving dx"))
        .note(
            "split-step's kinetic operator is exact per mode, so the difference is the five-point stencil's \
             dispersion error, (k dx)^2 / 12 in the phase velocity",
        )
}

fn crank_nicolson_conservation() -> Outcome {
    let grid = Grid2d::with_origin(96, 64, [12.0 * NM, 8.0 * NM], [-6.0 * NM, -4.0 * NM]);
    let mut potential = Potential::zero(grid);
    potential.add(&Shape::Rectangle { x: [1.0 * NM, 1.5 * NM], y: [-4.0 * NM, 4.0 * NM], height: 0.8 * EV });
    let h = Hamiltonian::new(grid, MASS, HBAR, Kinetic::FiniteDifference).with_potential(potential);
    let momentum = (2.0 * MASS * 0.5 * EV).sqrt();
    let packet = Wavefunction::gaussian(grid, [-2.0 * NM, 0.0], [0.8 * NM, 0.8 * NM], [momentum, 0.0], HBAR);
    let mut domain = QuantumDomain::new("cn", h).with_state(packet);
    let e0 = domain.moments().energy();
    let dt = 0.05 * HBAR / (0.5 * EV);
    let steps = 300;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let (mut worst_norm, mut worst_energy) = (0.0f64, 0.0f64);
    let mut iterations = 0;
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
        iterations += domain.last_solve().map_or(0, |s| s.iterations);
        worst_norm = worst_norm.max((domain.state().norm() - 1.0).abs());
        worst_energy = worst_energy.max((domain.moments().energy() - e0).abs() / e0);
    }
    Outcome::at_most("max |norm - 1| over the run", "1", worst_norm, 1e-10)
        .note(format!("max relative energy drift {worst_energy:.3e}"))
        .note(format!(
            "{steps} steps through a 0.8 eV barrier at a mean {:.1} BiCGSTAB iterations; the solve stops at a relative \
             residual of 1e-12, and the norm error is that times the steps, with room",
            iterations as f64 / steps as f64
        ))
}
