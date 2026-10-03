//! The benchmark harness.
//!
//! Spec §19.3 sets the rules, and §15.6 attaches a **correctness condition** to every
//! performance target in the table — "stable contacts and bounded penetration", "no
//! hidden particle dropping", "energy/temperature metrics within profile tolerance",
//! "manufactured-solution convergence". That pairing is the point:
//!
//! > **Optimize after validation.** A faster wrong solver is a regression. Every
//! > optimization must preserve tolerance-based validation suites. — §15.1
//!
//! So a benchmark here does not report a throughput number on its own. It reports
//! throughput *and* the checks that say whether the physics survived, and a run that
//! fails its checks has its throughput marked invalid rather than published.

use std::time::Instant;

use lattice_domain_grid2d::{gaussian, Diffusivity, HeatDomain, TimeScheme};
use lattice_domain_particle::{
    BoundaryBox, Integrator, LennardJones, ParticleDomain, ParticleSpec, UniformAcceleration,
};
use lattice_ir::{Arena, BoundarySet, Domain, Executor, Grid2d, Pcg32, StepContext};
use lattice_observe::{phase, Json, MemoryReport, Profile, Throughput};

/// One correctness condition checked alongside the timing.
#[derive(Clone, Debug)]
pub struct Check {
    /// What was measured.
    pub name: String,
    /// The measured value.
    pub observed: f64,
    /// The bound it must stay under.
    pub limit: f64,
}

impl Check {
    pub fn new(name: impl Into<String>, observed: f64, limit: f64) -> Self {
        Self { name: name.into(), observed, limit }
    }

    /// Whether the physics held.
    pub fn passed(&self) -> bool {
        self.observed.is_finite() && self.observed <= self.limit
    }
}

/// What one benchmark produced.
#[derive(Clone, Debug)]
pub struct BenchOutcome {
    /// Compute-phase throughput.
    pub throughput: Throughput,
    /// Time split by phase.
    pub profile: Profile,
    /// Memory footprint by component.
    pub memory: MemoryReport,
    /// The correctness conditions §15.6 attaches to this target.
    pub checks: Vec<Check>,
}

impl BenchOutcome {
    /// Whether every correctness condition held.
    ///
    /// When this is false the throughput number is meaningless and must not be
    /// published as a result.
    pub fn valid(&self) -> bool {
        self.checks.iter().all(Check::passed)
    }
}

/// What a benchmark is, independent of which backend runs it.
///
/// Split out from [`Benchmark`] so the reporting functions can serve a GPU benchmark too
/// without either backend's benchmark type having to know about the other's.
#[derive(Clone, Copy, Debug)]
pub struct Info {
    /// Identifier, used for filtering.
    pub name: &'static str,
    /// What the scene is.
    pub description: &'static str,
    /// The correctness condition, quoted from or modelled on spec §15.6.
    pub correctness: &'static str,
}

/// How a run was executed.
///
/// §19.3: *"Publish exact hardware, backend, precision, model file, engine revision, and
/// validation tolerance."* Everything in that sentence that varies between backends lives
/// here, so that a throughput figure is never printed without the configuration that
/// produced it.
#[derive(Clone, Debug)]
pub struct Executed {
    /// Backend and its configuration, e.g. `cpu-parallel (20 threads, 4 chunks/thread)`.
    pub label: String,
    /// The §10.5 mode the arithmetic ran in.
    ///
    /// Not a constant any more. The portable GPU backend runs `fast32` because WGSL has no
    /// `f64`, so a benchmark that printed `accurate64` beside a GPU number would be
    /// publishing the wrong precision with the result — exactly what §19.3 exists to stop.
    pub precision: &'static str,
    /// Threads, where that is a meaningful count.
    ///
    /// `None` for a GPU. "Speedup per thread" is not a quantity anyone can interpret when
    /// the divisor would be some product of SM count and occupancy, so efficiency is left
    /// unreported rather than computed against a number chosen to look reasonable.
    pub threads: Option<usize>,
}

impl Executed {
    /// How a CPU run was executed.
    pub fn cpu(executor: &Executor) -> Executed {
        Executed {
            label: executor.label(),
            precision: "accurate64",
            threads: Some(executor.threads()),
        }
    }
}

/// A named benchmark.
#[derive(Clone, Copy)]
pub struct Benchmark {
    /// Identifier, used for filtering.
    pub name: &'static str,
    /// What the scene is.
    pub description: &'static str,
    /// The correctness condition, quoted from or modelled on spec §15.6.
    pub correctness: &'static str,
    /// Run at the given scale factor, splitting solver loops across `executor`.
    pub run: fn(usize, &Executor) -> BenchOutcome,
}

impl Benchmark {
    /// The backend-independent description.
    pub fn info(&self) -> Info {
        Info {
            name: self.name,
            description: self.description,
            correctness: self.correctness,
        }
    }
}

/// All registered benchmarks.
pub fn all() -> &'static [Benchmark] {
    &[
        Benchmark {
            name: "particles-gravity",
            description: "N point particles under uniform gravity, velocity Verlet — \
                          measures raw integrator throughput with no neighbour search",
            correctness: "every particle matches the analytic free-fall trajectory",
            run: bench_particles_gravity,
        },
        Benchmark {
            name: "particles-lj",
            description: "Lennard-Jones fluid in a periodic box with a cell list — \
                          the §15.6 'Lennard-Jones MD' target",
            correctness: "energy and momentum drift within the declared tolerance, \
                          and no particle silently dropped",
            run: bench_particles_lj,
        },
        Benchmark {
            name: "heat-explicit",
            description: "explicit diffusion on a uniform grid — the §15.6 \
                          'heat/diffusion grid' target",
            correctness: "the field integral is conserved on a closed domain",
            run: bench_heat_explicit,
        },
        Benchmark {
            name: "heat-crank-nicolson",
            description: "implicit diffusion with a conjugate-gradient solve at ten \
                          times the explicit stability limit",
            correctness: "the integral is conserved and every linear solve converged",
            run: bench_heat_implicit,
        },
    ]
}

/// Look up benchmarks whose name contains `pattern`.
pub fn matching(pattern: &str) -> Vec<Benchmark> {
    all().iter().copied().filter(|b| b.name.contains(pattern)).collect()
}

// ---------------------------------------------------------------------------
// Particle benchmarks
// ---------------------------------------------------------------------------

fn bench_particles_gravity(scale: usize, executor: &Executor) -> BenchOutcome {
    let count = 16_384 * scale;
    let steps = 400;
    let dt = 1e-3;

    let mut profile = Profile::new();
    let mut domain = profile.time(phase::SETUP, || {
        let mut d = ParticleDomain::new("faller", count)
            .with_integrator(Integrator::VelocityVerlet)
            .with_force(UniformAcceleration::earth_gravity());
        let mut rng = Pcg32::seed_from_u64(1);
        for _ in 0..count {
            d.spawn(
                ParticleSpec::at([rng.range(0.0, 100.0), rng.range(0.0, 100.0)])
                    .with_mass(rng.range(0.5, 2.0)),
            )
            .expect("capacity was sized for exactly this many");
        }
        d.initialize();
        d
    });

    let start_y: Vec<f64> = domain.store().pos_y().to_vec();

    let compute_start = Instant::now();
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena).with_executor(executor);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
        }
    }
    let compute = compute_start.elapsed();
    profile.record(phase::COMPUTE, compute);

    // Correctness: velocity Verlet is exact for constant acceleration, so every
    // particle must sit on the analytic parabola to round-off.
    let t = dt * steps as f64;
    let drop = 0.5 * 9.806_65 * t * t;
    let worst = domain
        .store()
        .pos_y()
        .iter()
        .zip(&start_y)
        .map(|(y, y0)| (y - (y0 - drop)).abs())
        .fold(0.0f64, f64::max);

    let mut memory = MemoryReport::new();
    memory.record("particles", domain.memory_bytes());

    BenchOutcome {
        throughput: Throughput {
            steps: steps as u64,
            simulated_seconds: t,
            wall_clock: compute,
            elements: count as u64,
        },
        profile,
        memory,
        checks: vec![
            Check::new("max deviation from the analytic trajectory (m)", worst, 1e-9),
            Check::new(
                "particles lost",
                (count - domain.store().len()) as f64,
                0.0,
            ),
        ],
    }
}

fn bench_particles_lj(scale: usize, executor: &Executor) -> BenchOutcome {
    let side = 32 * scale;
    let count = side * side;
    let spacing = 1.4;
    let box_size = side as f64 * spacing;
    let steps = 200;
    let dt = 1e-3;

    let mut profile = Profile::new();
    let mut domain = profile.time(phase::SETUP, || {
        let mut d = ParticleDomain::new("lj", count)
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
    let momentum_scale: f64 =
        domain.store().vel_x().iter().zip(domain.store().mass()).map(|(v, m)| (v * m).abs()).sum();
    let pairs = domain
        .neighbors()
        .map(|list| list.pair_count(domain.store().pos_x(), domain.store().pos_y()))
        .unwrap_or(0);

    let compute_start = Instant::now();
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena).with_executor(executor);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
        }
    }
    let compute = compute_start.elapsed();
    profile.record(phase::COMPUTE, compute);

    let energy_drift = ((domain.total_energy() - e0) / e0.abs()).abs();
    let p = domain.momentum();
    let momentum_drift =
        ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2)).sqrt() / momentum_scale.max(1e-30);

    let mut memory = MemoryReport::new();
    memory.record("particles + cell list", domain.memory_bytes());

    BenchOutcome {
        throughput: Throughput {
            steps: steps as u64,
            simulated_seconds: dt * steps as f64,
            wall_clock: compute,
            elements: count as u64,
        },
        profile,
        memory,
        checks: vec![
            Check::new("relative energy drift", energy_drift, 5e-3),
            Check::new("relative momentum drift", momentum_drift, 1e-10),
            Check::new("particles lost", (count - domain.store().len()) as f64, 0.0),
            Check::new(
                "pairs found (must be nonzero, or the benchmark measures nothing)",
                if pairs > 0 { 0.0 } else { 1.0 },
                0.0,
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Grid benchmarks
// ---------------------------------------------------------------------------

fn heat_domain(side: usize, scheme: TimeScheme) -> HeatDomain {
    let grid = Grid2d::new(side, side, [1.0, 1.0]);
    HeatDomain::new("u", grid, Diffusivity::Uniform(1e-2))
        .with_scheme(scheme)
        .with_boundaries(BoundarySet::INSULATED)
        .with_tolerance(1e-10)
        .with_initial(gaussian([0.5, 0.5], 0.004, 1.0))
}

fn run_heat(
    side: usize,
    scheme: TimeScheme,
    steps: usize,
    dt_factor: f64,
    executor: &Executor,
) -> BenchOutcome {
    let mut profile = Profile::new();
    let mut domain = profile.time(phase::SETUP, || heat_domain(side, scheme));

    let explicit_limit = heat_domain(side, TimeScheme::Explicit).stable_step().max;
    let dt = dt_factor * explicit_limit;
    let before = domain.integral();

    let compute_start = Instant::now();
    let mut worst_iterations = 0usize;
    let mut total_iterations = 0usize;
    let mut any_diverged = false;
    {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena).with_executor(executor);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
            if let Some(outcome) = domain.last_solve() {
                worst_iterations = worst_iterations.max(outcome.iterations());
                total_iterations += outcome.iterations();
                any_diverged |= !outcome.is_converged();
            }
        }
    }
    let compute = compute_start.elapsed();
    profile.record(phase::COMPUTE, compute);

    let drift = (domain.integral() - before).abs() / before.abs();
    let non_finite = f64::from(u8::from(domain.field().first_non_finite().is_some()));

    let mut memory = MemoryReport::new();
    memory.record("fields + solver workspace", domain.memory_bytes());

    let mut checks = vec![
        Check::new("relative integral drift on a closed domain", drift, 1e-9),
        Check::new("cells holding a non-finite value", non_finite, 0.0),
    ];
    if scheme.is_implicit() {
        checks.push(Check::new(
            "linear solves that failed to converge",
            f64::from(u8::from(any_diverged)),
            0.0,
        ));
        // Published so that this number is comparable with the GPU benchmark's. A step of an
        // implicit scheme is not a fixed amount of work, and two backends solving to
        // different residual tolerances do different numbers of iterations to reach it — so
        // steps per second alone cannot be compared between them, and §19.3's "publish the
        // conditions" needs this to be one of the conditions.
        checks.push(Check::new(
            "conjugate-gradient iterations per step",
            total_iterations as f64 / steps.max(1) as f64,
            64.0,
        ));
    }

    BenchOutcome {
        throughput: Throughput {
            steps: steps as u64,
            simulated_seconds: dt * steps as f64,
            wall_clock: compute,
            elements: (side * side) as u64,
        },
        profile,
        memory,
        checks,
    }
}

fn bench_heat_explicit(scale: usize, executor: &Executor) -> BenchOutcome {
    // 0.8 of the stability limit: the largest step the scheme actually admits.
    run_heat(256 * scale, TimeScheme::Explicit, 400, 0.8, executor)
}

fn bench_heat_implicit(scale: usize, executor: &Executor) -> BenchOutcome {
    // Ten times the explicit limit, which is the whole reason to pay for a solve.
    run_heat(256 * scale, TimeScheme::CrankNicolson, 100, 10.0, executor)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Render one benchmark's result as text.
pub fn report(
    info: Info,
    scale: usize,
    executed: &Executed,
    outcome: &BenchOutcome,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{} (scale {scale}, {})\n", info.name, executed.label));
    out.push_str(&format!("  {}\n", info.description));
    out.push_str(&format!("  correctness condition: {}\n\n", info.correctness));

    out.push_str(&outcome.throughput.report());
    out.push('\n');
    out.push_str(&outcome.profile.report(None));
    out.push('\n');
    out.push_str(&outcome.memory.report());
    out.push('\n');

    for check in &outcome.checks {
        out.push_str(&format!(
            "  [{}] {:<52} {:>12.4e} (limit {:.1e})\n",
            if check.passed() { "OK  " } else { "FAIL" },
            check.name,
            check.observed,
            check.limit
        ));
    }
    if !outcome.valid() {
        out.push_str(
            "\n  RESULT INVALID: a correctness condition failed, so this throughput \
             number must not be published (spec §15.1).\n",
        );
    }
    out
}

/// Machine-readable form.
pub fn to_json(
    info: Info,
    scale: usize,
    executed: &Executed,
    outcome: &BenchOutcome,
) -> Json {
    let mut checks = Json::array();
    for check in &outcome.checks {
        checks.push(
            Json::object()
                .set("name", check.name.clone())
                .set("observed", check.observed)
                .set("limit", check.limit)
                .set("passed", check.passed()),
        );
    }
    Json::object()
        .set("name", info.name)
        .set("description", info.description)
        .set("correctness_condition", info.correctness)
        .set("scale", scale)
        .set("schedule", executed.label.clone())
        .set("precision", executed.precision)
        .set("threads", executed.threads.unwrap_or(0))
        .set("valid", outcome.valid())
        .set("throughput", outcome.throughput.to_json())
        .set("phases", outcome.profile.to_json())
        .set("memory_bytes", outcome.memory.to_json())
        .set("checks", checks)
}

/// What splitting the work bought, measured rather than assumed.
///
/// Two numbers, because they answer different questions. **Speedup** is what the wall
/// clock did. **Efficiency** is speedup per thread, and it is the one that says whether
/// the kernel is compute-bound: a streaming stencil saturates memory bandwidth long
/// before it saturates cores, so an efficiency well under one is the expected result
/// there and not a defect to chase.
///
/// The correctness conditions are re-checked on both runs, so a speedup is never
/// reported for a configuration that stopped being right (§15.1).
pub fn speedup_report(
    baseline: (&Executed, &BenchOutcome),
    faster: (&Executed, &BenchOutcome),
) -> String {
    let (base_executed, baseline) = baseline;
    let (fast_executed, parallel) = faster;
    let sequential_seconds = baseline.throughput.wall_clock.as_secs_f64();
    let parallel_seconds = parallel.throughput.wall_clock.as_secs_f64();
    if parallel_seconds <= 0.0 || sequential_seconds <= 0.0 {
        return "  speedup: not measurable, the run was too short to time\n".to_string();
    }

    let speedup = sequential_seconds / parallel_seconds;
    // Efficiency only where a divisor exists that a reader can interpret. See
    // `Executed::threads`.
    let per_thread = match fast_executed.threads {
        Some(threads) if threads > 1 => {
            format!("on {threads} threads ({:.0}% of linear; ", 100.0 * speedup / threads as f64)
        }
        _ => "(".to_string(),
    };
    let mut out = format!(
        "\n  speedup {speedup:.2}x {per_thread}{sequential_seconds:.3} s -> \
         {parallel_seconds:.3} s)\n"
    );
    if base_executed.precision != fast_executed.precision {
        out.push_str(&format!(
            "  NOTE: different precisions ({} vs {}), so this ratio compares two answers \
             that are not the same answer (§19.3).\n",
            base_executed.precision, fast_executed.precision
        ));
    }
    if !(baseline.valid() && parallel.valid()) {
        out.push_str(
            "  RESULT INVALID: one of the two runs failed a correctness condition, so \
             this ratio is not a speedup (spec §15.1).\n",
        );
    }
    out
}

/// The build-configuration warning that must accompany any published number.
///
/// Spec §19.3: *"Publish exact hardware, backend, precision, model file, engine
/// revision, and validation tolerance."* A timing taken from an unoptimized build is
/// off by an order of magnitude and is the single easiest way to publish a misleading
/// benchmark, so it is called out rather than left for the reader to infer.
pub fn build_warning() -> Option<String> {
    cfg!(debug_assertions).then(|| {
        "This binary was built without optimizations (debug_assertions is on). \
         Timings below are NOT representative — rebuild with `cargo build --release` \
         before quoting any number."
            .to_string()
    })
}

/// A one-line summary of the environment a benchmark ran in.
///
/// §19.3 asks for the backend to be published with the number, and the thread count is
/// part of the backend as soon as there is more than one of them. A throughput figure
/// with no schedule beside it cannot be compared with anything.
pub fn environment_for(executed: &Executed) -> String {
    format!(
        "engine {}  target {}  profile {}  precision {}  backend {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        if cfg!(debug_assertions) { "debug" } else { "release" },
        executed.precision,
        executed.label,
    )
}

/// [`environment_for`] the sequential executor, for `--version`.
pub fn environment() -> String {
    environment_for(&Executed::cpu(&Executor::sequential()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every benchmark must run at the smallest scale and pass its own correctness
    /// conditions. A benchmark that cannot be trusted at scale 1 cannot be trusted at
    /// scale 8 either.
    #[test]
    fn every_benchmark_runs_and_holds_its_correctness_conditions() {
        let executor = Executor::sequential();
        for benchmark in all() {
            let outcome = (benchmark.run)(1, &executor);
            assert!(
                outcome.valid(),
                "{} failed its correctness conditions:\n{}",
                benchmark.name,
                report(benchmark.info(), 1, &Executed::cpu(&executor), &outcome)
            );
            assert!(outcome.throughput.steps > 0);
            assert!(outcome.memory.total() > 0, "{} reported no memory", benchmark.name);
        }
    }

    /// §15.1: *"a faster wrong solver is a regression."* Every benchmark must still hold
    /// its correctness conditions when the work is split across threads — otherwise the
    /// speedup it reports is measuring a different, broken program.
    #[test]
    fn every_benchmark_still_holds_its_conditions_on_four_threads() {
        let executor = Executor::with_threads(4);
        for benchmark in all() {
            let outcome = (benchmark.run)(1, &executor);
            assert!(
                outcome.valid(),
                "{} failed on four threads:\n{}",
                benchmark.name,
                report(benchmark.info(), 1, &Executed::cpu(&executor), &outcome)
            );
        }
    }

    #[test]
    fn a_failing_check_invalidates_the_result() {
        let executor = Executor::sequential();
        let mut outcome = (all()[0].run)(1, &executor);
        outcome.checks.push(Check::new("deliberate failure", 1.0, 0.0));
        assert!(!outcome.valid());
        assert!(
            report(all()[0].info(), 1, &Executed::cpu(&executor), &outcome)
                .contains("RESULT INVALID")
        );
    }

    #[test]
    fn checks_treat_non_finite_observations_as_failures() {
        assert!(!Check::new("nan", f64::NAN, 1e9).passed());
        assert!(!Check::new("inf", f64::INFINITY, f64::INFINITY).passed());
        assert!(Check::new("fine", 0.5, 1.0).passed());
    }

    #[test]
    fn filtering_selects_benchmarks_by_name() {
        assert_eq!(matching("heat").len(), 2);
        assert_eq!(matching("particles").len(), 2);
        assert!(matching("nonexistent").is_empty());
    }

    #[test]
    fn json_output_carries_the_correctness_conditions_and_the_schedule() {
        let benchmark = &all()[0];
        let executor = Executor::with_threads(3);
        let outcome = (benchmark.run)(1, &executor);
        let json = to_json(benchmark.info(), 1, &Executed::cpu(&executor), &outcome);
        assert!(json.get("correctness_condition").is_some());
        assert_eq!(json.get("valid"), Some(&Json::Bool(true)));
        assert!(json.get("checks").is_some());
        assert!(json.get("throughput").is_some());
        // §19.3: a published number without its backend cannot be compared.
        assert_eq!(json.get("threads"), Some(&Json::Int(3)));
        assert!(format!("{:?}", json.get("schedule")).contains("cpu-parallel"));
    }

    /// §19.3 wants precision published with the number. It used to be the constant
    /// `accurate64`, which stopped being true the moment a `fast32` backend existed.
    #[test]
    fn the_published_precision_comes_from_the_backend() {
        let executor = Executor::sequential();
        let outcome = (all()[0].run)(1, &executor);
        let json = to_json(all()[0].info(), 1, &Executed::cpu(&executor), &outcome);
        assert_eq!(json.get("precision"), Some(&Json::String("accurate64".to_string())));
        assert!(environment_for(&Executed::cpu(&executor)).contains("precision accurate64"));
    }

    /// A ratio between two different precisions is not a speedup at the same answer, and
    /// the report has to say so rather than leaving a reader to notice.
    #[test]
    fn a_speedup_across_precisions_is_flagged_as_one() {
        let executor = Executor::sequential();
        let outcome = (all()[0].run)(1, &executor);
        let cpu = Executed::cpu(&executor);
        let gpu = Executed {
            label: "wgpu (test)".to_string(),
            precision: "fast32",
            threads: None,
        };

        let same = speedup_report((&cpu, &outcome), (&cpu, &outcome));
        assert!(!same.contains("different precisions"), "{same}");

        let across = speedup_report((&cpu, &outcome), (&gpu, &outcome));
        assert!(across.contains("different precisions"), "{across}");
        // Efficiency needs a divisor a reader can interpret, and a GPU has none.
        assert!(!across.contains("of linear"), "{across}");
    }

    #[test]
    fn a_debug_build_warns_about_its_own_timings() {
        // This test runs under `cargo test`, which is a debug build, so the warning
        // must be present. In a release build it must be absent.
        assert_eq!(build_warning().is_some(), cfg!(debug_assertions));
        assert!(environment().contains("cpu-scalar"));
        assert!(environment_for(&Executed::cpu(&Executor::with_threads(4))).contains("4 threads"));
    }
}
