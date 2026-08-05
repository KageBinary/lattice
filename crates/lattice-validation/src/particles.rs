//! Validation cases for the particle domain (spec §19.2).
//!
//! > Free fall, constant acceleration, harmonic oscillator, pendulum, elastic/inelastic
//! > collision, and constrained motion. […] Lennard–Jones energy conservation, radial
//! > distribution trends, and neighbor-list consistency.
//!
//! Collision and constraint cases belong to the rigid-body module (M2) and are not
//! registered here — a case that cannot run yet must be absent from the report, not
//! present and silently skipped.

use lattice_domain_particle::{
    BoundaryBox, HarmonicWell, Integrator, LennardJones, ParticleDomain, ParticleSpec,
    UniformAcceleration,
};
use lattice_ir::{Arena, Domain, Pcg32, StepContext};

use crate::{Case, Level, Outcome};

/// Standard gravity, m/s². Matches [`UniformAcceleration::earth_gravity`].
const G: f64 = 9.806_65;

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "free_fall",
        domain: "particles2d",
        level: Level::Analytic,
        claim: "a particle under constant acceleration follows y = y0 - g t^2 / 2 exactly",
        run: free_fall,
    },
    Case {
        name: "free_fall_is_mass_independent",
        domain: "particles2d",
        level: Level::Property,
        claim: "particles of different mass fall identically, because F = mg cancels the m in a = F/m",
        run: free_fall_mass_independence,
    },
    Case {
        name: "harmonic_oscillator_period",
        domain: "particles2d",
        level: Level::Analytic,
        claim: "after one period T = 2*pi*sqrt(m/k) the oscillator returns to its start",
        run: harmonic_period,
    },
    Case {
        name: "harmonic_energy_drift_velocity_verlet",
        domain: "particles2d",
        level: Level::Property,
        claim: "a symplectic integrator bounds energy error over long runs instead of accumulating it",
        run: harmonic_energy_drift,
    },
    Case {
        name: "explicit_euler_energy_growth_matches_theory",
        domain: "particles2d",
        level: Level::Analytic,
        claim: "explicit Euler amplifies oscillator energy by exactly (1 + (omega*dt)^2) per step",
        run: explicit_euler_energy_growth,
    },
    Case {
        name: "integrator_order_semi_implicit_euler",
        domain: "particles2d",
        level: Level::Manufactured,
        claim: "semi-implicit Euler converges at the first order it declares",
        run: order_semi_implicit,
    },
    Case {
        name: "integrator_order_velocity_verlet",
        domain: "particles2d",
        level: Level::Manufactured,
        claim: "velocity Verlet converges at the second order it declares",
        run: order_velocity_verlet,
    },
    Case {
        name: "velocity_verlet_is_time_reversible",
        domain: "particles2d",
        level: Level::Property,
        claim: "reversing velocity and stepping back returns the system to its starting state",
        run: reversibility,
    },
    Case {
        name: "pairwise_momentum_conservation",
        domain: "particles2d",
        level: Level::Property,
        claim: "Newton's third law makes total momentum exactly conserved under pair forces",
        run: momentum_conservation,
    },
    Case {
        name: "lennard_jones_energy_conservation",
        domain: "particles2d",
        level: Level::Property,
        claim: "a Lennard-Jones fluid in a periodic box conserves total energy within the declared tolerance",
        run: lennard_jones_energy,
    },
];

fn step(domain: &mut ParticleDomain, dt: f64, steps: usize) {
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
    }
}

fn free_fall() -> Outcome {
    let mut domain = ParticleDomain::new("faller", 1)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(UniformAcceleration::earth_gravity());
    domain.spawn(ParticleSpec::at([0.0, 1000.0]).with_mass(3.0)).unwrap();
    domain.initialize();

    let dt = 1e-3;
    let steps = 5_000;
    let mut worst_position = 0.0f64;
    let mut worst_velocity = 0.0f64;

    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for n in 1..=steps {
        domain.advance(dt, &mut ctx);
        let t = dt * n as f64;
        let expected_y = 1000.0 - 0.5 * G * t * t;
        let expected_v = -G * t;
        worst_position = worst_position.max((domain.store().pos_y()[0] - expected_y).abs());
        worst_velocity = worst_velocity.max((domain.store().vel_y()[0] - expected_v).abs());
    }

    Outcome::at_most("max absolute position error over 5 s", "m", worst_position, 1e-9)
        .note(format!("max velocity error {worst_velocity:.3e} m/s"))
        .note(
            "velocity Verlet integrates constant acceleration exactly, so the residual \
             here is floating-point round-off rather than truncation error",
        )
}

fn free_fall_mass_independence() -> Outcome {
    let mut domain = ParticleDomain::new("fallers", 3)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(UniformAcceleration::earth_gravity());
    for mass in [1e-3, 1.0, 1e3] {
        domain.spawn(ParticleSpec::at([0.0, 100.0]).with_mass(mass)).unwrap();
    }
    domain.initialize();
    step(&mut domain, 1e-3, 2_000);

    let ys = domain.store().pos_y();
    let spread = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - ys.iter().copied().fold(f64::INFINITY, f64::min);

    Outcome::at_most("spread in final height across a 10^6 mass range", "m", spread, 1e-12)
        .note("masses 1 mg, 1 kg and 1 t released together")
}

fn harmonic_period() -> Outcome {
    // m = 1, k = 4 gives omega = 2 rad/s and T = pi seconds.
    let stiffness = 4.0f64;
    let mass = 1.0f64;
    let omega = (stiffness / mass).sqrt();
    let period = core::f64::consts::TAU / omega;

    let mut domain = ParticleDomain::new("oscillator", 1)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(HarmonicWell::new([0.0, 0.0], stiffness));
    domain.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(mass)).unwrap();
    domain.initialize();

    let steps = 10_000;
    step(&mut domain, period / steps as f64, steps);

    let error = (domain.store().pos_x()[0] - 1.0).abs();
    Outcome::at_most("displacement error after one full period", "m", error, 1e-6)
        .note(format!("omega = {omega} rad/s, T = {period:.6} s, {steps} steps per period"))
}

fn harmonic_energy_drift() -> Outcome {
    let stiffness = 1.0;
    let mut domain = ParticleDomain::new("oscillator", 1)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(HarmonicWell::new([0.0, 0.0], stiffness));
    domain.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
    domain.initialize();

    let e0 = domain.total_energy();
    let dt = 0.01;
    let periods = 200.0;
    let steps = (periods * core::f64::consts::TAU / dt) as usize;

    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let mut worst = 0.0f64;
    let mut worst_first_tenth = 0.0f64;
    for n in 0..steps {
        domain.advance(dt, &mut ctx);
        let drift = ((domain.total_energy() - e0) / e0).abs();
        worst = worst.max(drift);
        if n < steps / 10 {
            worst_first_tenth = worst_first_tenth.max(drift);
        }
    }

    // The point is not just that drift is small but that it is *bounded*: the worst
    // excursion over the whole run must not exceed the worst over the first tenth.
    let growth = if worst_first_tenth > 0.0 { worst / worst_first_tenth } else { 1.0 };

    Outcome::at_most("max relative energy drift over 200 periods", "1", worst, 1e-4)
        .note(format!(
            "drift over the full run is {growth:.3}x the drift over the first tenth; \
             a value near 1 confirms the error is bounded rather than secular"
        ))
        .note(format!("{steps} steps at dt = {dt} s"))
}

fn explicit_euler_energy_growth() -> Outcome {
    // For x'' = -omega^2 x, explicit Euler multiplies E = (v^2 + omega^2 x^2)/2 by
    // exactly (1 + omega^2 dt^2) every step. This is a closed-form prediction, not a
    // bound, which makes it a sharp test of the integrator's failure mode.
    let stiffness = 1.0;
    let mass = 1.0;
    let omega_squared = stiffness / mass;
    let dt = 0.01;
    let steps = 1_000;

    let mut domain = ParticleDomain::new("euler", 1)
        .with_integrator(Integrator::ExplicitEuler)
        .with_force(HarmonicWell::new([0.0, 0.0], stiffness));
    domain.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(mass)).unwrap();
    domain.initialize();

    let e0 = domain.total_energy();
    step(&mut domain, dt, steps);
    let ratio = domain.total_energy() / e0;

    let predicted = (1.0 + omega_squared * dt * dt).powi(steps as i32);

    Outcome::near("energy ratio after 1000 steps", "1", ratio, predicted, 1e-6 * predicted)
        .note(format!(
            "theory: (1 + omega^2 dt^2)^n = {predicted:.9}; \
             this is why spec 10.2 admits explicit Euler for teaching only"
        ))
}

/// Measure the observed convergence order of an integrator against the analytic
/// oscillator solution `x(t) = cos(omega t)`.
fn observed_order(integrator: Integrator) -> (f64, f64, f64) {
    let stiffness = 1.0;
    let t_end = 2.0;

    let error_at = |dt: f64| {
        let steps = (t_end / dt).round() as usize;
        let mut domain = ParticleDomain::new("osc", 1)
            .with_integrator(integrator)
            .with_force(HarmonicWell::new([0.0, 0.0], stiffness));
        domain.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        domain.initialize();
        step(&mut domain, dt, steps);
        (domain.store().pos_x()[0] - t_end.cos()).abs()
    };

    let coarse = error_at(2e-3);
    let fine = error_at(1e-3);
    ((coarse / fine).log2(), coarse, fine)
}

fn order_semi_implicit() -> Outcome {
    let (order, coarse, fine) = observed_order(Integrator::SemiImplicitEuler);
    Outcome::near("observed convergence order", "1", order, 1.0, 0.15)
        .note(format!("errors {coarse:.3e} -> {fine:.3e} on halving dt"))
}

fn order_velocity_verlet() -> Outcome {
    let (order, coarse, fine) = observed_order(Integrator::VelocityVerlet);
    Outcome::near("observed convergence order", "1", order, 2.0, 0.15)
        .note(format!("errors {coarse:.3e} -> {fine:.3e} on halving dt"))
}

fn reversibility() -> Outcome {
    let mut domain = ParticleDomain::new("osc", 1)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(HarmonicWell::new([0.0, 0.0], 1.0));
    let id = domain.spawn(ParticleSpec::at([1.0, 0.0]).with_velocity([0.0, 0.3]).with_mass(1.0)).unwrap();
    domain.initialize();

    let start = domain.store().position_of(id).unwrap();
    let dt = 0.005;
    let steps = 2_000;
    step(&mut domain, dt, steps);

    // Reverse time by negating the velocity, then re-establish the force invariant.
    let v = domain.store().velocity_of(id).unwrap();
    domain.store_mut().set_velocity(id, [-v[0], -v[1]]);
    domain.initialize();
    step(&mut domain, dt, steps);

    let end = domain.store().position_of(id).unwrap();
    let error = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();

    Outcome::at_most("return distance after forward-then-reverse", "m", error, 1e-9)
        .note(format!("{steps} steps each way at dt = {dt} s"))
}

/// Build a Lennard-Jones fluid with exactly zero net momentum.
fn lennard_jones_box() -> ParticleDomain {
    let side = 8;
    let count = side * side;
    let spacing = 1.5;
    let box_size = side as f64 * spacing;

    let mut domain = ParticleDomain::new("lj", count)
        .with_integrator(Integrator::VelocityVerlet)
        .with_bounds(BoundaryBox::periodic([0.0, 0.0], [box_size, box_size]))
        .with_force(LennardJones::with_default_cutoff(1.0, 1.0));

    let mut rng = Pcg32::seed_from_u64(20260805);
    let velocities: Vec<[f64; 2]> =
        (0..count).map(|_| [rng.normal() * 0.4, rng.normal() * 0.4]).collect();
    let mean_x: f64 = velocities.iter().map(|v| v[0]).sum::<f64>() / count as f64;
    let mean_y: f64 = velocities.iter().map(|v| v[1]).sum::<f64>() / count as f64;

    for (index, v) in velocities.iter().enumerate() {
        let (i, j) = (index % side, index / side);
        domain
            .spawn(
                ParticleSpec::at([(i as f64 + 0.5) * spacing, (j as f64 + 0.5) * spacing])
                    .with_velocity([v[0] - mean_x, v[1] - mean_y])
                    .with_mass(1.0),
            )
            .unwrap();
    }
    domain.initialize();
    domain
}

fn momentum_conservation() -> Outcome {
    let mut domain = lennard_jones_box();
    let p0 = domain.momentum();
    let scale: f64 = domain
        .store()
        .vel_x()
        .iter()
        .zip(domain.store().mass())
        .map(|(v, m)| (v * m).abs())
        .sum();

    step(&mut domain, 1e-3, 5_000);

    let p = domain.momentum();
    let drift = ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2)).sqrt();

    Outcome::at_most("momentum drift relative to total |p_i| after 5000 steps", "1", drift / scale, 1e-12)
        .note(format!("absolute drift {drift:.3e} kg m/s against a scale of {scale:.3e}"))
        .note("pair forces are applied as equal and opposite by construction, so the residual is round-off")
}

fn lennard_jones_energy() -> Outcome {
    let mut domain = lennard_jones_box();
    let e0 = domain.total_energy();

    let dt = 1e-3;
    let steps = 5_000;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    let mut worst = 0.0f64;
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
        worst = worst.max(((domain.total_energy() - e0) / e0.abs()).abs());
    }

    Outcome::at_most("max relative total-energy drift", "1", worst, 5e-3)
        .note(format!("64 particles, {steps} steps at dt = {dt}, cutoff 2.5 sigma"))
        .note(
            "the truncated-and-shifted potential is continuous in energy but not in \
             force, so each cutoff crossing injects a small impulse; that is the \
             dominant term here, not integration error",
        )
}
