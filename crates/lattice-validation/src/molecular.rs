//! Validation cases for the molecular-dynamics module (spec §12.4, §19.2).
//!
//! > Lennard–Jones energy conservation, radial distribution trends, and neighbor-list
//! > consistency.
//!
//! Those three are §19.2's row; the rest are the module's own claims — that a bond
//! vibrates at the frequency its reduced mass sets, that each thermostat does what
//! its contract says, and that the trajectory analyses measure what their names say.
//!
//! The systems are argon in SI units (`ε/k_B = 119.8 K`, `σ = 3.405 Å`), so a
//! temperature in kelvin means what it says. Reduced units appear only in the notes,
//! where they make a number recognisable: `T* = k_B T / ε`, `ρ* = ρ σ²`,
//! `τ = σ √(m/ε)`.

use std::sync::OnceLock;

use lattice_domain_particle::{
    analysis, Angle, Bond, BoundaryBox, HarmonicAngle, HarmonicBond, Integrator, LennardJones,
    ParticleDomain, ParticleSpec, Thermostat, Truncation, BOLTZMANN,
};
use lattice_ir::{Arena, Domain, Pcg32, StepContext};

use crate::{Case, Level, Outcome};

/// Argon's mass, kg.
const MASS: f64 = 39.948 * 1.660_539_066_60e-27;
/// Argon's Lennard-Jones diameter, m.
const SIGMA: f64 = 3.405e-10;
/// Argon's Lennard-Jones well depth, J.
const EPSILON: f64 = 119.8 * BOLTZMANN;

/// The Lennard-Jones time unit `σ √(m/ε)`, about 2.16 ps.
fn tau() -> f64 {
    SIGMA * (MASS / EPSILON).sqrt()
}

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "bond_vibration_uses_the_reduced_mass",
        domain: "molecular2d",
        level: Level::Analytic,
        claim: "a harmonic bond between unequal masses vibrates at omega = sqrt(k / mu), mu = m1 m2 / (m1 + m2)",
        run: bond_vibration,
    },
    Case {
        name: "bonded_chain_energy_is_bounded",
        domain: "molecular2d",
        level: Level::Property,
        claim: "a chain held by bonds and angles conserves energy to velocity Verlet's bound at the step it chooses",
        run: bonded_chain_energy,
    },
    Case {
        name: "force_shifted_lj_energy_error_is_second_order",
        domain: "molecular2d",
        level: Level::Manufactured,
        claim: "with the force shifted to zero at the cutoff, a Lennard-Jones fluid's energy error is velocity Verlet's alone and falls as dt^2",
        run: force_shift_order,
    },
    Case {
        name: "verlet_list_matches_a_fresh_cell_list",
        domain: "molecular2d",
        level: Level::CrossScheme,
        claim: "after a run on a skinned Verlet list, the forces equal those from a cell list rebuilt from scratch",
        run: verlet_list_consistency,
    },
    Case {
        name: "rdf_excludes_the_repulsive_core",
        domain: "molecular2d",
        level: Level::Analytic,
        claim: "a Lennard-Jones liquid has no pairs closer than 0.8 sigma, where the Boltzmann factor is e^-43",
        run: rdf_core,
    },
    Case {
        name: "rdf_first_shell_sits_at_the_potential_minimum",
        domain: "molecular2d",
        level: Level::Property,
        claim: "a Lennard-Jones liquid's first coordination shell peaks at the pair minimum 2^(1/6) sigma, and g(r) returns to 1 beyond it",
        run: rdf_first_shell,
    },
    Case {
        name: "langevin_holds_the_target_temperature",
        domain: "molecular2d",
        level: Level::Analytic,
        claim: "a Langevin bath brings a gas to its target temperature and holds it there within the statistical error of the sample",
        run: langevin_temperature,
    },
    Case {
        name: "langevin_free_diffusion_matches_ornstein_uhlenbeck",
        domain: "molecular2d",
        level: Level::Analytic,
        claim: "a free particle in a Langevin bath has MSD = 4D[t - (1 - e^-gt)/g] with the Einstein D = kT / (m g)",
        run: langevin_diffusion,
    },
    Case {
        name: "velocity_rescaling_relaxes_at_the_berendsen_rate",
        domain: "molecular2d",
        level: Level::Analytic,
        claim: "velocity rescaling moves an ideal gas toward its target as T_n = T0 + (Ti - T0)(1 - dt/tau)^n exactly",
        run: berendsen_relaxation,
    },
];

fn run(domain: &mut ParticleDomain, dt: f64, steps: usize) {
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
    }
}

/// A square lattice of `side × side` argon atoms at reduced density `density` in a
/// periodic box, thermalized to `temperature` kelvin with zero net momentum.
fn lattice_fluid(side: usize, density: f64, temperature: f64, law: LennardJones, seed: u64) -> ParticleDomain {
    let count = side * side;
    let spacing = SIGMA / density.sqrt();
    let box_size = side as f64 * spacing;
    let mut domain = ParticleDomain::new("argon", count)
        .with_integrator(Integrator::VelocityVerlet)
        .with_bounds(BoundaryBox::periodic([0.0, 0.0], [box_size, box_size]))
        .with_force(law)
        .with_seed(seed);
    for index in 0..count {
        let (i, j) = (index % side, index / side);
        domain
            .spawn(ParticleSpec::at([(i as f64 + 0.5) * spacing, (j as f64 + 0.5) * spacing]).with_mass(MASS))
            .unwrap();
    }
    let mut rng = Pcg32::seed_from_u64(seed);
    assert!(analysis::thermalize(domain.store_mut(), temperature, &mut rng));
    domain
}

/// `count` non-interacting argon atoms scattered over an open plane, thermalized to
/// `temperature` kelvin.
fn ideal_gas(count: usize, temperature: f64, seed: u64) -> ParticleDomain {
    let mut domain = ParticleDomain::new("gas", count).with_integrator(Integrator::VelocityVerlet).with_seed(seed);
    let mut rng = Pcg32::seed_from_u64(seed);
    for _ in 0..count {
        domain.spawn(ParticleSpec::at([rng.next_f64() * 1e-8, rng.next_f64() * 1e-8]).with_mass(MASS)).unwrap();
    }
    assert!(analysis::thermalize(domain.store_mut(), temperature, &mut rng));
    domain
}

fn separation(domain: &ParticleDomain, a: usize, b: usize) -> f64 {
    let (x, y) = (domain.store().pos_x(), domain.store().pos_y());
    (x[a] - x[b]).hypot(y[a] - y[b])
}

fn bond_vibration() -> Outcome {
    // Unequal masses, so a law that used either mass alone — or their mean — would be
    // off by tens of percent rather than by integration error.
    let (m1, m2) = (MASS, 3.0 * MASS);
    let mu = m1 * m2 / (m1 + m2);
    let (stiffness, rest) = (300.0, 1.5e-10);
    let amplitude = 0.02 * rest;
    let omega = (stiffness / mu).sqrt();
    let period = core::f64::consts::TAU / omega;

    let mut domain = ParticleDomain::new("dimer", 2).with_integrator(Integrator::VelocityVerlet);
    let a = domain.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(m1)).unwrap();
    let b = domain.spawn(ParticleSpec::at([rest + amplitude, 0.0]).with_mass(m2)).unwrap();
    let mut domain = domain.with_force(HarmonicBond::new(vec![Bond::new(a, b, rest, stiffness)]));
    domain.initialize();

    let steps = 1_000;
    let dt = period / steps as f64;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let mut worst = 0.0f64;
    for n in 1..=steps {
        domain.advance(dt, &mut ctx);
        let expected = rest + amplitude * (omega * dt * n as f64).cos();
        worst = worst.max((separation(&domain, 0, 1) - expected).abs());
    }

    // Velocity Verlet on a harmonic mode is exact up to a frequency shift:
    // cos(ω̃ dt) = 1 − (ω dt)²/2, so ω̃ ≈ ω(1 + (ω dt)²/24) and the phase drifts by
    // (ω dt)²·ωt/24, which at t = T is 2π(ω dt)²/24. The worst position error is that
    // phase times the amplitude.
    let phase = core::f64::consts::TAU * (omega * dt).powi(2) / 24.0;
    let predicted = phase * amplitude;
    Outcome::at_most("max separation error over one period / amplitude", "1", worst / amplitude, 1.5 * phase)
        .note(format!(
            "velocity Verlet's own phase error predicts {:.3e}; the limit is 1.5x that",
            predicted / amplitude
        ))
        .note(format!(
            "a law that used m1 alone would make the period {:.1}% too long",
            100.0 * ((m1 / mu).sqrt() - 1.0)
        ))
}

fn bonded_chain_energy() -> Outcome {
    // A zig-zag chain at its rest geometry: every bond at r0, every angle at 120°.
    let beads = 16;
    let (bond_k, rest) = (300.0, 1.5e-10);
    let (angle_k, angle) = (1e-19, 120f64.to_radians());
    let (dx, dy) = (rest * (angle / 2.0).sin(), rest * (angle / 2.0).cos());

    let mut domain = ParticleDomain::new("chain", beads).with_integrator(Integrator::VelocityVerlet);
    let ids: Vec<_> = (0..beads)
        .map(|i| domain.spawn(ParticleSpec::at([i as f64 * dx, (i % 2) as f64 * dy]).with_mass(MASS)).unwrap())
        .collect();
    let bonds = ids.windows(2).map(|w| Bond::new(w[0], w[1], rest, bond_k)).collect();
    let angles = ids.windows(3).map(|w| Angle::new(w[0], w[1], w[2], angle, angle_k)).collect();
    let mut domain = domain.with_force(HarmonicBond::new(bonds)).with_force(HarmonicAngle::new(angles));
    let mut rng = Pcg32::seed_from_u64(20261003);
    assert!(analysis::thermalize(domain.store_mut(), 300.0, &mut rng));
    domain.initialize();

    let dt = domain.stable_step().preferred;
    let kinetic0 = domain.kinetic_energy();
    let e0 = domain.total_energy();
    let steps = 20_000;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let (mut worst, mut worst_first_tenth) = (0.0f64, 0.0f64);
    for n in 0..steps {
        domain.advance(dt, &mut ctx);
        let deviation = (domain.total_energy() - e0).abs() / kinetic0;
        worst = worst.max(deviation);
        if n < steps / 10 {
            worst_first_tenth = worst_first_tenth.max(deviation);
        }
    }

    // The fastest mode of a chain of springs is the zone-boundary one, ω = 2√(k/m) —
    // √2 above the dimer frequency the step was chosen from. Verlet's energy error
    // oscillates with relative amplitude (ω dt)²/4 of the mode's energy, and that
    // mode cannot hold more than the whole of it.
    let omega_max = 2.0 * (bond_k / MASS).sqrt();
    let bound = (omega_max * dt).powi(2) / 4.0;
    Outcome::at_most("max |E - E0| / K0 over the run", "1", worst, bound)
        .note(format!(
            "bound (omega_max dt)^2 / 4 = {bound:.3e} at the domain's own dt = {dt:.3e} s, {steps} steps"
        ))
        .note(format!(
            "worst over the run is {:.3}x the worst over its first tenth: bounded, not secular",
            worst / worst_first_tenth
        ))
}

/// RMS of the total-energy deviation over `time` seconds at step `dt`, relative to
/// `ε` per particle.
fn energy_error(truncation: Truncation, dt: f64, time: f64) -> f64 {
    let law = LennardJones::with_truncation(EPSILON, SIGMA, 2.5 * SIGMA, truncation);
    let mut domain = lattice_fluid(10, 0.6, 1.0 * EPSILON / BOLTZMANN, law, 7);
    domain.initialize();
    let e0 = domain.total_energy();
    let steps = (time / dt).round() as usize;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let mut sum = 0.0;
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
        sum += (domain.total_energy() - e0).powi(2);
    }
    (sum / steps as f64).sqrt() / (EPSILON * domain.store().len() as f64)
}

fn force_shift_order() -> Outcome {
    let (dt, time) = (0.004 * tau(), 2.0 * tau());
    let coarse = energy_error(Truncation::ForceShift, dt, time);
    let fine = energy_error(Truncation::ForceShift, dt / 2.0, time);
    let order = (coarse / fine).log2();

    let shifted_coarse = energy_error(Truncation::EnergyShift, dt, time);
    let shifted_fine = energy_error(Truncation::EnergyShift, dt / 2.0, time);
    Outcome::near("observed order of the RMS energy error", "1", order, 2.0, 0.3)
        .note(format!("RMS |E - E0| per particle: {coarse:.3e} -> {fine:.3e} eps on halving dt"))
        .note(format!(
            "energy-shifted only, the same runs give {shifted_coarse:.3e} -> {shifted_fine:.3e} eps (order {:.2}): \
             over a run this short the force step at the cutoff is not yet the dominant error",
            (shifted_coarse / shifted_fine).log2()
        ))
        .note("100 atoms at rho* = 0.6 from a lattice at T* = 1, 2 tau, cutoff 2.5 sigma")
}

fn verlet_list_consistency() -> Outcome {
    let law = LennardJones::new(EPSILON, SIGMA, 2.5 * SIGMA);
    let mut skinned = lattice_fluid(20, 0.7, EPSILON / BOLTZMANN, law, 11).with_skin(0.3 * SIGMA);
    skinned.initialize();
    let steps = 1_000;
    run(&mut skinned, 0.005 * tau(), steps);
    let rebuilds = skinned.neighbors().map_or(0, |list| list.rebuilds());

    // The same positions, binned from nothing.
    let store = skinned.store();
    let bounds = skinned.bounds().unwrap();
    let mut fresh = ParticleDomain::new("fresh", store.len())
        .with_integrator(Integrator::VelocityVerlet)
        .with_bounds(bounds)
        .with_force(law);
    for i in 0..store.len() {
        fresh.spawn(ParticleSpec::at([store.pos_x()[i], store.pos_y()[i]]).with_mass(MASS)).unwrap();
    }
    fresh.initialize();

    let (a, b) = (skinned.store(), fresh.store());
    let mut worst = 0.0f64;
    let mut largest = 0.0f64;
    for i in 0..a.len() {
        worst = worst.max((a.force_x()[i] - b.force_x()[i]).hypot(a.force_y()[i] - b.force_y()[i]));
        largest = largest.max(b.force_x()[i].hypot(b.force_y()[i]));
    }
    let agreement = if worst == 0.0 {
        "bit for bit".to_string()
    } else {
        format!("to {:.2e}: the lists visit pairs in different orders", worst / largest)
    };
    Outcome::at_most("max force difference / largest force", "1", worst / largest, 1e-12)
        .note(format!("400 atoms, {steps} steps with a 0.3 sigma skin: {rebuilds} rebuilds"))
        .note(format!(
            "agreement {agreement}; a pair missing from the stale list would differ by a whole pair force"
        ))
}

/// The structure of a Lennard-Jones liquid at `ρ* = 0.7`, `T* = 1`, computed once.
fn liquid_rdf() -> &'static analysis::RadialDistribution {
    static RDF: OnceLock<analysis::RadialDistribution> = OnceLock::new();
    RDF.get_or_init(|| {
        let temperature = EPSILON / BOLTZMANN;
        let law = LennardJones::new(EPSILON, SIGMA, 2.5 * SIGMA);
        let mut domain = lattice_fluid(16, 0.7, temperature, law, 3)
            .with_thermostat(Thermostat::Langevin { temperature, friction: 1.0 / tau() });
        domain.initialize();
        let dt = 0.005 * tau();
        // Melt the lattice and forget it before anything is counted.
        run(&mut domain, dt, 1_500);
        let mut accumulator = {
            let bounds = domain.bounds().unwrap();
            analysis::RdfAccumulator::new(70, 3.5 * SIGMA, bounds.min, bounds.size, bounds.periodic_axes(), 256)
        };
        for _ in 0..200 {
            run(&mut domain, dt, 10);
            accumulator.sample(domain.store());
        }
        accumulator.result()
    })
}

fn rdf_core() -> Outcome {
    let rdf = liquid_rdf();
    let inside = rdf.r.iter().zip(&rdf.g).filter(|(r, _)| **r < 0.8 * SIGMA).map(|(_, g)| *g).fold(0.0, f64::max);
    let boltzmann = (-4.0 * (0.8f64.powi(-12) - 0.8f64.powi(-6))).exp();
    Outcome::at_most("max g(r) below 0.8 sigma", "1", inside, 1e-3)
        .note(format!("exp(-U(0.8 sigma)/kT) = {boltzmann:.1e} at T* = 1"))
        .note(format!("{} frames of 256 atoms at rho* = 0.7", rdf.frames))
}

fn rdf_first_shell() -> Outcome {
    let rdf = liquid_rdf();
    let (r_peak, g_peak) = rdf.peak().unwrap();
    let r_min = 2f64.powf(1.0 / 6.0) * SIGMA;
    let tail: Vec<f64> = rdf.r.iter().zip(&rdf.g).filter(|(r, _)| **r > 2.9 * SIGMA).map(|(_, g)| *g).collect();
    let tail_mean = tail.iter().sum::<f64>() / tail.len() as f64;
    // A packed liquid compresses its first shell a little inside the pair minimum, so
    // the tolerance is that few-percent shift plus one bin.
    Outcome::near("first-peak position / 2^(1/6) sigma", "1", r_peak / r_min, 1.0, 0.06)
        .note(format!("peak g = {g_peak:.2} at {:.3} sigma; bin width {:.3} sigma", r_peak / SIGMA, rdf.bin_width / SIGMA))
        .note(format!("mean g beyond 2.9 sigma = {tail_mean:.3}: the liquid has no long-range order"))
}

fn langevin_temperature() -> Outcome {
    let target = 300.0;
    let friction = 1e12;
    let dt = 0.01 / friction;
    let mut domain = ideal_gas(1_000, 0.5 * target, 5)
        .with_thermostat(Thermostat::Langevin { temperature: target, friction });
    domain.initialize();
    // Five relaxation times to forget the 150 K start.
    run(&mut domain, dt, 500);

    let (samples, every) = (200, 50);
    let mut sum = 0.0;
    for _ in 0..samples {
        run(&mut domain, dt, every);
        sum += domain.temperature().unwrap();
    }
    let mean = sum / samples as f64;

    // Each sample's temperature has relative spread √(2/N_df). Kinetic energy relaxes
    // as e^(−2γt), so successive samples correlate by ρ = e^(−2γ·every·dt), and the
    // mean of M samples with that lag-one correlation is worth M(1 − ρ)/(1 + ρ)
    // independent ones. Four standard errors.
    let dof: f64 = 2.0 * 1_000.0 - 2.0;
    let window = (samples * every) as f64 * dt;
    let rho = (-2.0 * friction * every as f64 * dt).exp();
    let independent = samples as f64 * (1.0 - rho) / (1.0 + rho);
    let standard_error = target * (2.0 / dof).sqrt() / independent.sqrt();
    Outcome::near("time-averaged temperature", "K", mean, target, 4.0 * standard_error)
        .note(format!(
            "1000 free argon atoms from 150 K, gamma dt = 0.01, {samples} samples over {window:.2e} s \
             ({independent:.0} independent); off by {:.1} standard errors",
            (mean - target) / standard_error
        ))
        .note("BAOAB's O step is the exact Ornstein-Uhlenbeck update, so the bias is zero and the tolerance is pure sampling error")
}

fn langevin_diffusion() -> Outcome {
    let temperature = 300.0;
    let friction = 1e12;
    let dt = 0.01 / friction;
    let count = 10_000;
    let mut domain = ideal_gas(count, temperature, 9)
        .with_thermostat(Thermostat::Langevin { temperature, friction });
    domain.initialize();
    let steps = 500;
    run(&mut domain, dt, steps);

    let t = steps as f64 * dt;
    let diffusion = BOLTZMANN * temperature / (MASS * friction);
    let predicted = 4.0 * diffusion * (t - (-(-friction * t).exp_m1()) / friction);
    let observed = domain.mean_squared_displacement();
    // A 2D Gaussian displacement's squared length is exponentially distributed, whose
    // spread equals its mean: the sample mean of N is good to 1/√N. Four of those.
    let tolerance = 4.0 / (count as f64).sqrt();
    Outcome::near("MSD at t = 5/gamma / Ornstein-Uhlenbeck prediction", "1", observed / predicted, 1.0, tolerance)
        .note(format!("predicted {predicted:.4e} m^2, observed {observed:.4e} m^2"))
        .note(format!("D = kT/(m gamma) = {diffusion:.4e} m^2/s; {count} free atoms, gamma dt = 0.01"))
}

fn berendsen_relaxation() -> Outcome {
    let (start, target) = (100.0, 300.0);
    let relaxation = 1e-12;
    let dt = 0.01 * relaxation;
    let mut domain = ideal_gas(500, start, 13).with_thermostat(Thermostat::VelocityRescale { temperature: target, relaxation });
    domain.initialize();

    // With no forces the integrator leaves velocities alone, so each step is exactly
    // T ← T + (dt/τ)(T0 − T): a geometric approach with ratio 1 − dt/τ.
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let mut worst = 0.0f64;
    let steps = 300;
    for n in 1..=steps {
        domain.advance(dt, &mut ctx);
        let predicted = target + (start - target) * (1.0 - dt / relaxation).powi(n);
        worst = worst.max((domain.temperature().unwrap() / predicted - 1.0).abs());
    }
    Outcome::at_most("max relative deviation from the Berendsen recurrence", "1", worst, 1e-12)
        .note(format!(
            "{steps} steps at dt/tau = 0.01 take 100 K to {:.2} K of a 300 K target; worst deviation {worst:.2e}",
            domain.temperature().unwrap()
        ))
        .note(
            "the recurrence is exact, so the residual is round-off; Berendsen suppresses \
             kinetic-energy fluctuations and does not sample the canonical ensemble, which \
             its contract says",
        )
}
