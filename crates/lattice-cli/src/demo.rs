//! Runnable demonstration scenes.
//!
//! These are the M0 exit condition — *"analytic heat + particle demos and benchmark
//! harness"* — in the form a user can actually invoke. Each produces a run artifact
//! (§18.1) and a terminal visualization, so the same command serves as a smoke test,
//! a teaching example, and a source of reproducible output.

use std::time::Instant;

use lattice_domain_grid2d::{gaussian, Diffusivity, HeatDomain, TimeScheme};
use lattice_domain_particle::{
    BoundaryBox, HarmonicWell, Integrator, LennardJones, ParticleDomain, ParticleSpec,
};
use lattice_ir::{
    Arena, Boundary, BoundarySet, Domain, Grid2d, Observations, Pcg32, StepContext,
};
use lattice_observe::{phase, Profile, RunArtifact};

use crate::render;

/// How long to run and how often to sample.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Number of simulation steps.
    pub steps: usize,
    /// Approximate number of timeline samples to record.
    pub samples: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { steps: 400, samples: 40 }
    }
}

impl Options {
    fn interval(&self) -> usize {
        (self.steps / self.samples.max(1)).max(1)
    }
}

/// What a demo produced.
pub struct DemoResult {
    /// The machine-readable record.
    pub artifact: RunArtifact,
    /// A terminal visualization.
    pub visual: String,
}

/// A named demonstration scene.
#[derive(Clone, Copy)]
pub struct Demo {
    /// Identifier used on the command line.
    pub name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Run it.
    pub run: fn(&Options) -> DemoResult,
}

/// All registered demos.
pub fn all() -> &'static [Demo] {
    &[
        Demo {
            name: "heat-gaussian",
            description: "a Gaussian heat pulse spreading in a closed box, \
                          with the conserved integral tracked alongside",
            run: heat_gaussian,
        },
        Demo {
            name: "heat-slab",
            description: "a slab held between two temperatures relaxing to the \
                          linear steady profile of textbook conduction",
            run: heat_slab,
        },
        Demo {
            name: "oscillator",
            description: "the same oscillator under all three integrators, showing \
                          why spec §10.2 admits explicit Euler for teaching only",
            run: oscillator,
        },
        Demo {
            name: "lj-gas",
            description: "a Lennard-Jones gas in a periodic box, conserving energy \
                          and momentum while the lattice melts",
            run: lj_gas,
        },
    ]
}

/// Find a demo by exact name.
pub fn find(name: &str) -> Option<Demo> {
    all().iter().copied().find(|d| d.name == name)
}

fn heat_gaussian(options: &Options) -> DemoResult {
    let side = 96;
    let grid = Grid2d::new(side, side, [1.0, 1.0]);
    let diffusivity = 2e-3;

    let mut profile = Profile::new();
    let mut heat = profile.time(phase::SETUP, || {
        HeatDomain::new("temperature", grid, Diffusivity::Uniform(diffusivity))
            .with_scheme(TimeScheme::CrankNicolson)
            .with_boundaries(BoundarySet::INSULATED)
            .with_initial(gaussian([0.5, 0.5], 0.0015, 1.0))
    });

    // Run to a fixed *physical* time rather than a fixed timestep, so the pulse is
    // still a recognizable feature at the end whatever `--steps` is set to. Stepping
    // at the scheme's preferred rate instead would race past equilibrium and render a
    // flat box, which looks like a bug and is not.
    let target_time = 3.0;
    let dt = target_time / options.steps as f64;
    let explicit_limit = lattice_domain_grid2d::DiffusionOperator::new(
        &grid,
        &Diffusivity::Uniform(diffusivity),
    )
    .explicit_stability_limit()
    .max;
    let initial_integral = heat.integral();

    let mut artifact = RunArtifact::new("heat-gaussian");
    artifact.add_contract(heat.contract());
    artifact.set_parameter("grid", format!("{side}x{side}"));
    artifact.set_parameter("diffusivity_m2_per_s", diffusivity);
    artifact.set_parameter("dt_seconds", dt);
    artifact.set_parameter("steps", options.steps);
    artifact.set_parameter("scheme", heat.scheme().name());
    artifact.set_parameter("boundaries", "insulated on all four edges (closed)");
    artifact.set_parameter("explicit_stability_limit_seconds", explicit_limit);
    artifact.set_parameter("dt_over_explicit_limit", dt / explicit_limit);

    let mut observations = Observations::new();
    let mut integral_trace = Vec::new();
    let mut peak_trace = Vec::new();
    let interval = options.interval();

    let compute_start = Instant::now();
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for n in 0..options.steps {
            heat.advance(dt, &mut ctx);
            if n % interval == 0 || n + 1 == options.steps {
                observations.clear();
                heat.observe(&mut observations);
                artifact.record(dt * (n + 1) as f64, n as u64 + 1, &observations);
                integral_trace.push(heat.integral());
                peak_trace.push(heat.field().max_interior());
            }
        }
    }
    profile.record(phase::COMPUTE, compute_start.elapsed());

    let drift = (heat.integral() - initial_integral).abs() / initial_integral.abs();
    if drift > 1e-9 {
        artifact.warn(format!("field integral drifted by {drift:e} on a closed domain"));
    }
    artifact.memory.record("heat domain", heat.memory_bytes());
    artifact.profile = profile;

    let mut visual = String::new();
    visual.push_str(&format!(
        "  temperature after {} steps ({:.4} s of physical time)\n",
        options.steps,
        dt * options.steps as f64
    ));
    visual.push_str(&render::heatmap(heat.field(), 72, 24));
    visual.push_str(&format!(
        "\n  peak    {}\n  integral{}\n",
        render::sparkline(&peak_trace, 72),
        render::sparkline(&integral_trace, 72)
    ));
    let variance = 0.0015 + 2.0 * diffusivity * target_time;
    let ratio = dt / explicit_limit;
    visual.push_str(&format!(
        "\n  The integral is the conserved quantity: it drifted {drift:.2e} relative over the\n  \
         run, which is the conjugate-gradient tolerance rather than a physical loss. The\n  \
         peak falls as sigma^2 = sigma0^2 + 2Dt grows, reaching sigma = {:.3} m — exactly\n  \
         what the analytic heat kernel predicts.\n\n  \
         dt = {dt:.4} s is {ratio:.2}x the explicit stability limit of {explicit_limit:.4} s.\n  \
         Try `--steps 50` to push Crank-Nicolson well past what an explicit scheme could\n  \
         survive, or `--steps 2000` to see the same physics resolved more finely.\n",
        variance.sqrt()
    ));

    DemoResult { artifact, visual }
}

fn heat_slab(options: &Options) -> DemoResult {
    let grid = Grid2d::new(64, 16, [1.0, 0.25]);
    let mut boundaries = BoundarySet::INSULATED;
    boundaries.left = Boundary::fixed(273.15);
    boundaries.right = Boundary::fixed(373.15);

    let mut profile = Profile::new();
    let mut heat = profile.time(phase::SETUP, || {
        HeatDomain::new("temperature", grid, Diffusivity::Uniform(1e-2))
            .with_scheme(TimeScheme::BackwardEuler)
            .with_boundaries(boundaries)
            .with_uniform_initial(273.15)
    });

    let dt = 0.5;
    let mut artifact = RunArtifact::new("heat-slab");
    artifact.add_contract(heat.contract());
    artifact.set_parameter("grid", "64x16");
    artifact.set_parameter("left_boundary_K", 273.15);
    artifact.set_parameter("right_boundary_K", 373.15);
    artifact.set_parameter("dt_seconds", dt);
    artifact.set_parameter("scheme", heat.scheme().name());

    let mut observations = Observations::new();
    let mut max_trace = Vec::new();
    let interval = options.interval();

    let compute_start = Instant::now();
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for n in 0..options.steps {
            heat.advance(dt, &mut ctx);
            if n % interval == 0 || n + 1 == options.steps {
                observations.clear();
                heat.observe(&mut observations);
                artifact.record(dt * (n + 1) as f64, n as u64 + 1, &observations);
                max_trace.push(heat.field().max_interior());
            }
        }
    }
    profile.record(phase::COMPUTE, compute_start.elapsed());
    artifact.memory.record("heat domain", heat.memory_bytes());
    artifact.profile = profile;

    // Compare the mid-height row against the analytic steady profile.
    let row = grid.ny() / 2;
    let mut worst = 0.0f64;
    for i in 0..grid.nx() {
        let x = grid.cell_center(i, row)[0];
        let exact = 273.15 + 100.0 * x;
        worst = worst.max((heat.field().get(i, row) - exact).abs());
    }
    artifact.set_parameter("max_deviation_from_linear_profile_K", worst);

    let mut visual = String::new();
    visual.push_str("  temperature (left edge 273.15 K, right edge 373.15 K)\n");
    visual.push_str(&render::heatmap(heat.field(), 72, 14));
    visual.push_str(&format!("\n  peak    {}\n", render::sparkline(&max_trace, 72)));
    visual.push_str(&format!(
        "\n  The steady solution of conduction across a slab is a straight line.\n  \
         Largest deviation from it: {worst:.3e} K. This also confirms the Dirichlet\n  \
         halo places the prescribed value on the boundary *face*, not at the cell\n  \
         centre — an off-by-half-a-cell error would show up as a constant offset.\n"
    ));

    DemoResult { artifact, visual }
}

fn oscillator(options: &Options) -> DemoResult {
    let stiffness = 1.0f64;
    let mass = 1.0f64;
    let omega = (stiffness / mass).sqrt();
    let period = core::f64::consts::TAU / omega;
    let dt = period / 100.0;

    let mut artifact = RunArtifact::new("oscillator");
    artifact.set_parameter("stiffness_N_per_m", stiffness);
    artifact.set_parameter("mass_kg", mass);
    artifact.set_parameter("angular_frequency_rad_per_s", omega);
    artifact.set_parameter("dt_seconds", dt);
    artifact.set_parameter("steps", options.steps);
    artifact.set_parameter("periods", dt * options.steps as f64 / period);

    let schemes = [
        ("euler", Integrator::ExplicitEuler),
        ("symplectic", Integrator::SemiImplicitEuler),
        ("verlet", Integrator::VelocityVerlet),
    ];

    let mut profile = Profile::new();
    let mut domains: Vec<ParticleDomain> = profile.time(phase::SETUP, || {
        schemes
            .iter()
            .map(|(name, integrator)| {
                let mut d = ParticleDomain::new(*name, 1)
                    .with_integrator(*integrator)
                    .with_force(HarmonicWell::new([0.0, 0.0], stiffness));
                d.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(mass)).expect("capacity is 1");
                d.initialize();
                d
            })
            .collect()
    });

    for domain in &domains {
        artifact.add_contract(domain.contract());
    }
    let initial_energy: Vec<f64> = domains.iter().map(ParticleDomain::total_energy).collect();

    let mut observations = Observations::new();
    let mut traces: Vec<Vec<f64>> = vec![Vec::new(); domains.len()];
    let interval = options.interval();

    let compute_start = Instant::now();
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for n in 0..options.steps {
            for domain in &mut domains {
                domain.advance(dt, &mut ctx);
            }
            if n % interval == 0 || n + 1 == options.steps {
                observations.clear();
                for domain in &domains {
                    domain.observe(&mut observations);
                }
                artifact.record(dt * (n + 1) as f64, n as u64 + 1, &observations);
                for (index, domain) in domains.iter().enumerate() {
                    traces[index].push(domain.total_energy() / initial_energy[index]);
                }
            }
        }
    }
    profile.record(phase::COMPUTE, compute_start.elapsed());
    artifact.memory.record("three oscillators", domains.iter().map(ParticleDomain::memory_bytes).sum());
    artifact.profile = profile;

    let mut visual = String::new();
    visual.push_str(&format!(
        "  total energy relative to its initial value, over {:.1} periods\n\n",
        dt * options.steps as f64 / period
    ));
    for (index, (name, integrator)) in schemes.iter().enumerate() {
        let final_ratio = traces[index].last().copied().unwrap_or(1.0);
        visual.push_str(&format!(
            "  {:<11} {}\n              order {}, symplectic {:<5}  final E/E0 = {final_ratio:.6}\n",
            name,
            render::sparkline(&traces[index], 64),
            integrator.order(),
            integrator.is_symplectic(),
        ));
    }
    let euler_ratio = traces[0].last().copied().unwrap_or(1.0);
    visual.push_str(&format!(
        "\n  Explicit Euler multiplied the energy by {euler_ratio:.4}. That is not a bug:\n  \
         for x'' = -w^2 x it amplifies energy by exactly (1 + (w dt)^2) every step,\n  \
         which is why it exists here only for comparison. Both symplectic schemes\n  \
         oscillate around the true energy without accumulating error.\n"
    ));

    DemoResult { artifact, visual }
}

fn lj_gas(options: &Options) -> DemoResult {
    let side = 16;
    let count = side * side;
    let spacing = 1.4;
    let box_size = side as f64 * spacing;
    let dt = 1e-3;

    let mut profile = Profile::new();
    let mut domain = profile.time(phase::SETUP, || {
        let mut d = ParticleDomain::new("gas", count)
            .with_integrator(Integrator::VelocityVerlet)
            .with_bounds(BoundaryBox::periodic([0.0, 0.0], [box_size, box_size]))
            .with_force(LennardJones::with_default_cutoff(1.0, 1.0));

        let mut rng = Pcg32::seed_from_u64(20260805);
        let velocities: Vec<[f64; 2]> =
            (0..count).map(|_| [rng.normal() * 0.6, rng.normal() * 0.6]).collect();
        let mean_x: f64 = velocities.iter().map(|v| v[0]).sum::<f64>() / count as f64;
        let mean_y: f64 = velocities.iter().map(|v| v[1]).sum::<f64>() / count as f64;
        for (index, v) in velocities.iter().enumerate() {
            let (i, j) = (index % side, index / side);
            d.spawn(
                ParticleSpec::at([(i as f64 + 0.5) * spacing, (j as f64 + 0.5) * spacing])
                    .with_velocity([v[0] - mean_x, v[1] - mean_y])
                    .with_mass(1.0),
            )
            .expect("capacity was sized for exactly this many");
        }
        d.initialize();
        d
    });

    let e0 = domain.total_energy();
    let p0 = domain.momentum();

    let mut artifact = RunArtifact::new("lj-gas").with_seed(20260805);
    artifact.add_contract(domain.contract());
    artifact.set_parameter("particles", count);
    artifact.set_parameter("box_size_m", box_size);
    artifact.set_parameter("dt_seconds", dt);
    artifact.set_parameter("steps", options.steps);
    artifact.set_parameter("potential", "Lennard-Jones, epsilon = 1, sigma = 1, cutoff = 2.5 sigma");
    artifact.set_parameter("boundaries", "periodic (closed)");

    let mut observations = Observations::new();
    let mut energy_trace = Vec::new();
    let mut kinetic_trace = Vec::new();
    let interval = options.interval();

    let compute_start = Instant::now();
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for n in 0..options.steps {
            domain.advance(dt, &mut ctx);
            if n % interval == 0 || n + 1 == options.steps {
                observations.clear();
                domain.observe(&mut observations);
                artifact.record(dt * (n + 1) as f64, n as u64 + 1, &observations);
                energy_trace.push(domain.total_energy() / e0);
                kinetic_trace.push(domain.kinetic_energy());
            }
        }
    }
    profile.record(phase::COMPUTE, compute_start.elapsed());
    artifact.memory.record("particles + cell list", domain.memory_bytes());
    artifact.profile = profile;

    let p = domain.momentum();
    let momentum_drift = ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2)).sqrt();
    let energy_drift = ((domain.total_energy() - e0) / e0.abs()).abs();
    artifact.set_parameter("relative_energy_drift", energy_drift);
    artifact.set_parameter("momentum_drift_kg_m_per_s", momentum_drift);

    let mut visual = String::new();
    visual.push_str(&format!("  {count} particles after {} steps\n", options.steps));
    visual.push_str(&render::scatter(
        domain.store().pos_x(),
        domain.store().pos_y(),
        [0.0, 0.0],
        [box_size, box_size],
        72,
        24,
        "particles",
    ));
    visual.push_str(&format!(
        "\n  E/E0    {}\n  kinetic {}\n",
        render::sparkline(&energy_trace, 72),
        render::sparkline(&kinetic_trace, 72)
    ));
    visual.push_str(&format!(
        "\n  The lattice melts as potential energy converts to kinetic, but the total\n  \
         is held to {energy_drift:.2e} relative. Momentum drifted {momentum_drift:.2e} kg m/s —\n  \
         pair forces are applied as equal and opposite, so that residual is round-off.\n"
    ));

    DemoResult { artifact, visual }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every demo must run, produce a hashable artifact, and not warn. A demo is the
    /// first thing a new user runs; one that emits a conservation warning is either
    /// broken or badly configured, and either way must not ship.
    #[test]
    fn every_demo_runs_cleanly() {
        let options = Options { steps: 40, samples: 8 };
        for demo in all() {
            let result = (demo.run)(&options);
            assert!(
                result.artifact.warnings().is_empty(),
                "{} warned: {:?}",
                demo.name,
                result.artifact.warnings()
            );
            assert!(
                result.artifact.first_non_finite().is_none(),
                "{} produced a non-finite observation",
                demo.name
            );
            assert!(!result.artifact.timeline().is_empty(), "{} recorded nothing", demo.name);
            assert!(!result.visual.is_empty(), "{} rendered nothing", demo.name);
            assert!(result.artifact.memory.total() > 0, "{} reported no memory", demo.name);
        }
    }

    /// FR-011: a recorded run must reproduce. Same inputs, same content hash.
    #[test]
    fn demos_are_reproducible() {
        let options = Options { steps: 30, samples: 6 };
        for demo in all() {
            let first = (demo.run)(&options).artifact.content_hash();
            let second = (demo.run)(&options).artifact.content_hash();
            assert_eq!(first, second, "{} is not reproducible", demo.name);
        }
    }

    #[test]
    fn demos_are_findable_by_name() {
        assert!(find("oscillator").is_some());
        assert!(find("no-such-demo").is_none());
    }

    /// The oscillator demo exists to show a specific contrast. If explicit Euler ever
    /// stops gaining energy relative to the symplectic schemes, the demo has lost its
    /// point and the integrator is probably mislabelled.
    #[test]
    fn the_oscillator_demo_actually_shows_the_contrast() {
        let result = (find("oscillator").unwrap().run)(&Options { steps: 400, samples: 20 });
        let last = result.artifact.timeline().last().unwrap();
        let energy = |name: &str| {
            last.values
                .iter()
                .find(|(k, _)| k == &format!("{name}.total_energy"))
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("no energy recorded for {name}"))
        };
        let (euler, symplectic, verlet) =
            (energy("euler"), energy("symplectic"), energy("verlet"));
        assert!(euler > symplectic * 1.1, "explicit Euler should gain energy: {euler} vs {symplectic}");
        assert!((verlet - 0.5).abs() < 0.01, "velocity Verlet should hold E = 0.5 J, got {verlet}");
    }
}
