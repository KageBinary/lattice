//! The runtime: clock, timestep negotiation, stepping, and diagnostics.
//!
//! Spec §9.1 sets the problem:
//!
//! > A universal engine must handle domains with dramatically different characteristic
//! > timescales. Lattice therefore cannot require every module to advance with one
//! > fixed global timestep. […] Domains report stability limits and preferred cadence.
//!
//! M1 implements the first of the six time models in that section — **fixed global
//! step** — but implements the *negotiation* that the others build on. Every domain
//! publishes a [`StableStep`]; the runtime takes the tightest
//! and names the domain responsible. Subcycling and adaptive stepping (§9.1) reuse
//! that machinery rather than replacing it.
//!
//! # Refusing to run is a feature
//!
//! NFR-007: *"The engine must expose numerical instability rather than silently clamp
//! or hide it."* Two things follow.
//!
//! A run whose requested timestep exceeds a domain's stability limit **stops before
//! the first step**, naming the domain and the limit. Clamping silently would produce
//! a run that finishes and is wrong.
//!
//! A run that produces a non-finite value **halts at that step**, recording which
//! observation went bad and when. Continuing would fill the artifact with `NaN` and
//! bury the first one.

use std::time::Instant;

mod schedule;

use lattice_ir::{
    Arena, CompiledModel, Domain, Executor, Observations, StabilityReason, StableStep, StepContext,
};
use lattice_observe::{Json, Profile, RunArtifact, Throughput, phase};

/// The simulation clock.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Clock {
    /// Simulation time, seconds.
    pub time: f64,
    /// Steps taken.
    pub step: u64,
}

impl Clock {
    /// Advance by `dt`.
    fn advance(&mut self, dt: f64) {
        self.time += dt;
        self.step += 1;
    }
}

/// How a run should be driven.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct RunConfig {
    /// Physical time to simulate, seconds. `None` means run until another limit.
    pub duration: Option<f64>,
    /// Hard cap on steps, which is what stops a run whose duration is unset.
    pub max_steps: Option<u64>,
    /// Override the negotiated timestep.
    pub timestep: Option<f64>,
    /// Sample observations at most this often, seconds. `None` means every step.
    pub sample_interval: Option<f64>,
    /// Halt on the first non-finite observation.
    pub stop_on_non_finite: bool,
}

impl RunConfig {
    /// A configuration that halts on instability, which is what a user wants unless
    /// they are deliberately studying a blow-up.
    pub fn new() -> RunConfig {
        RunConfig {
            stop_on_non_finite: true,
            ..RunConfig::default()
        }
    }

    /// Take duration and timestep from a compiled model, keeping any override.
    pub fn from_model(model: &CompiledModel) -> RunConfig {
        let sample_interval = model
            .observers
            .iter()
            .filter_map(|observer| observer.interval)
            .fold(None, |best: Option<f64>, interval| {
                Some(best.map_or(interval, |b| b.min(interval)))
            });
        RunConfig {
            duration: model.duration,
            max_steps: None,
            timestep: model.timestep,
            sample_interval,
            stop_on_non_finite: true,
        }
    }

    /// Set the duration.
    pub fn with_duration(mut self, seconds: f64) -> RunConfig {
        self.duration = Some(seconds);
        self
    }

    /// Set the step cap.
    pub fn with_max_steps(mut self, steps: u64) -> RunConfig {
        self.max_steps = Some(steps);
        self
    }

    /// Override the timestep.
    pub fn with_timestep(mut self, dt: f64) -> RunConfig {
        self.timestep = Some(dt);
        self
    }
}

/// Why a run stopped.
#[derive(Clone, PartialEq, Debug)]
pub enum StopReason {
    /// The requested physical time was reached.
    DurationReached,
    /// The step cap was reached.
    StepLimit,
    /// Neither a duration nor a step cap was set, so nothing would have ended it.
    NoStoppingCondition,
    /// An observation went non-finite.
    NonFinite {
        /// Which observation.
        observation: String,
        /// The step it happened on.
        step: u64,
        /// The time it happened at.
        time: f64,
    },
    /// The requested timestep exceeds a domain's stability limit.
    Unstable {
        /// The domain that cannot take this step.
        domain: String,
        /// The step that was asked for.
        requested: f64,
        /// The largest step that domain can survive.
        limit: f64,
        /// What sets the limit.
        reason: StabilityReason,
    },
}

impl StopReason {
    /// True when the run finished as intended.
    pub fn is_success(&self) -> bool {
        matches!(self, StopReason::DurationReached | StopReason::StepLimit)
    }

    /// A message for the user.
    pub fn describe(&self) -> String {
        match self {
            StopReason::DurationReached => "reached the requested duration".to_string(),
            StopReason::StepLimit => "reached the step limit".to_string(),
            StopReason::NoStoppingCondition => {
                "nothing would have stopped this run: set `duration:` in the model, or pass \
                 a step limit"
                    .to_string()
            }
            StopReason::NonFinite {
                observation,
                step,
                time,
            } => format!(
                "HALTED: `{observation}` became non-finite at step {step} (t = {time:e} s). \
                 The run stopped there rather than filling the artifact with NaN"
            ),
            StopReason::Unstable {
                domain,
                requested,
                limit,
                reason,
            } => format!(
                "REFUSED: timestep {requested:e} s exceeds the stability limit {limit:e} s for \
                 domain `{domain}` ({}). Nothing was stepped — a clamped run would have \
                 finished and been wrong",
                reason.describe()
            ),
        }
    }
}

/// What a completed run produced.
#[derive(Debug)]
pub struct RunOutcome {
    /// Why it stopped.
    pub stop: StopReason,
    /// The clock at the end.
    pub clock: Clock,
    /// The timestep used.
    pub timestep: f64,
    /// The self-describing record.
    pub artifact: RunArtifact,
}

impl RunOutcome {
    /// True when the run finished as intended.
    pub fn is_success(&self) -> bool {
        self.stop.is_success()
    }
}

/// A model, its solvers, and the state needed to drive them.
pub struct Simulation {
    model: CompiledModel,
    domains: Vec<Box<dyn Domain>>,
    /// The coupling edges between the domains, and the ledger of what they moved.
    ///
    /// Empty for an uncoupled model, in which case `exchange` is a no-op and costs a
    /// loop over nothing.
    coupler: lattice_coupling::Coupler,
    arena: Arena,
    /// How solvers split their loops (spec §15.3).
    ///
    /// Owned by the simulation rather than by each domain: the thread count is a
    /// property of the run. Sequential unless a caller asks otherwise, and the choice
    /// is recorded in the artifact — parallel execution does not change the numbers
    /// (see `lattice_cpu`), but a reader should not have to take that on trust.
    executor: Executor,
    schedule: Option<schedule::Schedule>,
    clock: Clock,
    observations: Observations,
    profile: Profile,
    /// Coupling faults seen so far, so a mis-wired edge is reported once rather than
    /// once per step.
    coupling_faults: Vec<String>,
}

impl core::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Simulation")
            .field("model", &self.model.name)
            .field("domains", &self.domains.len())
            .field("clock", &self.clock)
            .finish()
    }
}

impl Simulation {
    /// Build a simulation from a compiled model and its instantiated solvers.
    /// Build a simulation with coupling edges between its domains.
    pub fn coupled(
        model: CompiledModel,
        domains: Vec<Box<dyn Domain>>,
        coupler: lattice_coupling::Coupler,
    ) -> Simulation {
        let mut simulation = Simulation::new(model, domains);
        simulation.coupler = coupler;
        simulation
    }

    /// The coupling ledger, for a report or a conservation check.
    pub fn ledger(&self) -> &lattice_ir::ConservationLedger {
        self.coupler.ledger()
    }

    /// Coupling edges that failed, each reported once.
    pub fn coupling_faults(&self) -> &[String] {
        &self.coupling_faults
    }

    pub fn new(model: CompiledModel, domains: Vec<Box<dyn Domain>>) -> Simulation {
        // The buffer plan already decided how much scratch the model needs, which is
        // what keeps NFR-001 true: this is the last allocation before the hot loop.
        let arena = Arena::with_capacity(model.buffers.scratch_elements());
        let schedule = schedule::Schedule::new(
            &model.graph,
            domains.len(),
            model.buffers.scratch_elements(),
        );
        Simulation {
            model,
            domains,
            coupler: lattice_coupling::Coupler::new(),
            arena,
            executor: Executor::sequential(),
            schedule,
            clock: Clock::default(),
            observations: Observations::new(),
            profile: Profile::new(),
            coupling_faults: Vec::new(),
        }
    }

    /// Split every solver's loops across `executor`'s threads.
    ///
    /// Consuming rather than borrowing: the executor owns worker threads, and a
    /// simulation that outlived a borrowed pool would be stepping through a dangling
    /// configuration. One executor per simulation is also the arrangement that keeps
    /// the thread count a property of the run.
    pub fn with_executor(mut self, executor: Executor) -> Simulation {
        if let Some(schedule) = &mut self.schedule {
            schedule.enable_parallel(executor.threads());
        }
        self.executor = executor;
        self
    }

    /// How this simulation splits its work.
    pub fn executor(&self) -> &Executor {
        &self.executor
    }

    /// The compiled model.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    /// The clock.
    pub fn clock(&self) -> Clock {
        self.clock
    }

    /// The most recent observations.
    pub fn observations(&self) -> &Observations {
        &self.observations
    }

    /// The instantiated solvers.
    pub fn domains(&self) -> &[Box<dyn Domain>] {
        &self.domains
    }

    /// Every drawable channel across all domains (spec §7.3).
    pub fn render_channels(&self) -> Vec<lattice_ir::RenderChannel<'_>> {
        self.domains
            .iter()
            .flat_map(|domain| domain.render_channels())
            .collect()
    }

    /// Every curve across all domains, as accumulated so far.
    pub fn curves(&self) -> Vec<lattice_ir::Curve> {
        self.domains.iter().flat_map(|domain| domain.curves()).collect()
    }

    /// The tightest stability constraint across all domains.
    ///
    /// Spec §9.1: domains *"report stability limits and preferred cadence"*, and the
    /// runtime reconciles them. [`StableStep::tightest`] keeps the reason code from
    /// whichever domain binds, so the answer to "why is my timestep so small?" names
    /// a mechanism rather than a number.
    pub fn stability(&self) -> StableStep {
        self.domains.iter().map(|domain| domain.stable_step()).fold(
            StableStep::unconditional(f64::INFINITY),
            StableStep::tightest,
        )
    }

    /// The domain whose stability limit binds, if any does.
    pub fn limiting_domain(&self) -> Option<&str> {
        let overall = self.stability();
        if overall.max.is_infinite() {
            return None;
        }
        self.domains
            .iter()
            .find(|domain| domain.stable_step().max <= overall.max)
            .map(|domain| domain.name())
    }

    /// Decide the timestep for a run, or explain why the requested one is unsafe.
    pub fn resolve_timestep(&self, config: &RunConfig) -> Result<f64, StopReason> {
        let stability = self.stability();
        let requested = config
            .timestep
            .or(self.model.timestep)
            .unwrap_or(stability.preferred);

        // A preferred step of infinity means no domain expressed a preference, which
        // happens only when there are no domains at all.
        let requested = if requested.is_finite() && requested > 0.0 {
            requested
        } else {
            1e-3
        };

        for domain in &self.domains {
            let limit = domain.stable_step();
            if !limit.admits(requested) {
                return Err(StopReason::Unstable {
                    domain: domain.name().to_string(),
                    requested,
                    limit: limit.max,
                    reason: limit.reason,
                });
            }
        }
        Ok(requested)
    }

    /// Take one step at `dt`, without any stability or stopping checks.
    ///
    /// [`Simulation::run`] is the normal entry point; this exists for tests and for
    /// callers driving the clock themselves.
    pub fn step(&mut self, dt: f64) {
        if let Some(schedule) = &self.schedule {
            schedule.step(
                &mut self.domains,
                &mut self.arena,
                &self.executor,
                self.clock.time,
                self.clock.step,
                dt,
            );
        } else {
            let mut context = StepContext {
                time: self.clock.time,
                step: self.clock.step,
                arena: &mut self.arena,
                executor: &self.executor,
            };
            for domain in &mut self.domains {
                domain.prepare(&mut context);
            }
            for domain in &mut self.domains {
                domain.advance(dt, &mut context);
            }
        }

        // Coupling runs *after* the domains, which makes this a loose staggered scheme
        // (§14.2): each domain advances on the latest state it was given, and what it
        // produces reaches its neighbours before the next step. One exchange is always
        // in flight, which is what the ledger's residual measures.
        let report = self
            .coupler
            .exchange(&mut self.domains, self.clock.step, self.clock.time, dt);
        for (edge, fault) in &report.faults {
            let message = format!("coupling edge `{edge}`: {fault}");
            if !self.coupling_faults.contains(&message) {
                self.coupling_faults.push(message);
            }
        }

        self.clock.advance(dt);
    }

    /// Refresh the observation set from every domain.
    pub fn observe(&mut self) {
        self.observations.clear();
        for domain in &self.domains {
            domain.observe(&mut self.observations);
        }
    }

    /// Run to completion.
    pub fn run(&mut self, config: &RunConfig) -> RunOutcome {
        let mut artifact =
            RunArtifact::new(self.model.name.clone()).with_precision(self.model.precision.code());
        for domain in &self.model.domains {
            if let Some(contract) = domain.contract {
                artifact.add_contract(contract);
            }
        }
        artifact.set_parameter("fidelity", self.model.fidelity.code());
        artifact.set_parameter("dimensions", u64::from(self.model.dimensions));

        // Recorded as a *section*, not a parameter, which puts it outside the content
        // hash. §19.3 wants the backend published with every result; FR-011 wants the
        // same model to hash the same. Both hold at once only because how the work was
        // divided is not part of the physics — see `lattice_cpu`, and the validation
        // case `cpu_parallel_matches_scalar` that keeps it true.
        artifact.set_section(
            "execution",
            Json::object()
                .set("backend", "cpu")
                .set("schedule", self.executor.label())
                .set("threads", self.executor.threads()),
        );
        for note in &self.model.notes {
            artifact.warn(note.clone());
        }

        let timestep = match self.resolve_timestep(config) {
            Ok(dt) => dt,
            Err(stop) => {
                artifact.warn(stop.describe());
                return RunOutcome {
                    stop,
                    clock: self.clock,
                    timestep: 0.0,
                    artifact,
                };
            }
        };
        artifact.set_parameter("timestep_seconds", timestep);

        let stability = self.stability();
        artifact.set_parameter("stability_limit_seconds", stability.max);
        artifact.set_parameter("stability_margin", stability.margin(timestep));
        if let Some(domain) = self.limiting_domain() {
            artifact.set_parameter("limiting_domain", domain);
        }

        let max_steps = match (config.duration, config.max_steps) {
            (_, Some(cap)) => cap,
            (Some(duration), None) => (duration / timestep).ceil() as u64,
            (None, None) => {
                let stop = StopReason::NoStoppingCondition;
                artifact.warn(stop.describe());
                return RunOutcome {
                    stop,
                    clock: self.clock,
                    timestep,
                    artifact,
                };
            }
        };

        // Sample the initial state, so the artifact records where the run began.
        self.observe();
        artifact.record(self.clock.time, self.clock.step, &self.observations);
        let mut next_sample = config.sample_interval.unwrap_or(0.0);

        let started = Instant::now();
        let mut stop = StopReason::StepLimit;

        for _ in 0..max_steps {
            self.step(timestep);

            if let Some(duration) = config.duration
                && self.clock.time >= duration - 0.5 * timestep
            {
                stop = StopReason::DurationReached;
            }

            // A sample is due at the step nearest its time, which is the same half-step
            // rule the duration uses. An absolute slack — this was `1e-12` seconds — is
            // wrong at every scale but one: a femtosecond model, stepping 1e-18 s at a
            // time, found every step "due" and sampled two thousand times instead of
            // thirty.
            let due = match config.sample_interval {
                None => true,
                Some(interval) => {
                    if self.clock.time + 0.5 * timestep >= next_sample {
                        while next_sample <= self.clock.time + 0.5 * timestep {
                            next_sample += interval;
                        }
                        true
                    } else {
                        false
                    }
                }
            };
            let last_step = stop == StopReason::DurationReached;

            if due || last_step {
                self.observe();
                artifact.record(self.clock.time, self.clock.step, &self.observations);

                if config.stop_on_non_finite
                    && let Some(observation) = self.observations.first_non_finite()
                {
                    stop = StopReason::NonFinite {
                        observation: observation.name.to_string(),
                        step: self.clock.step,
                        time: self.clock.time,
                    };
                    break;
                }
            }

            if last_step {
                break;
            }
        }

        let elapsed = started.elapsed();
        self.profile.record(phase::COMPUTE, elapsed);
        artifact.profile = core::mem::take(&mut self.profile);
        artifact
            .memory
            .record("planned buffers", self.model.buffers.total_bytes());
        artifact.set_throughput(Throughput {
            steps: self.clock.step,
            simulated_seconds: self.clock.time,
            wall_clock: elapsed,
            elements: self.element_count(),
        });
        if !stop.is_success() {
            artifact.warn(stop.describe());
        }
        // Curves summarize the whole run, so they are taken once, after the last step —
        // including after a stop, where a partial result is still what was measured.
        for curve in self.curves() {
            artifact.add_curve(curve);
        }

        RunOutcome {
            stop,
            clock: self.clock,
            timestep,
            artifact,
        }
    }

    /// Degrees of freedom advanced per step, for the throughput report.
    fn element_count(&self) -> u64 {
        self.model
            .buffers
            .buffers()
            .iter()
            .map(|buffer| match buffer.kind {
                lattice_ir::BufferKind::ScalarField { nx, ny, .. } => (nx * ny) as u64,
                lattice_ir::BufferKind::VectorField { nx, ny, .. } => (2 * nx * ny) as u64,
                lattice_ir::BufferKind::ParticleArrays { capacity } => capacity as u64,
                lattice_ir::BufferKind::RigidBodyArrays { capacity } => capacity as u64,
                lattice_ir::BufferKind::Scratch { .. } => 0,
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{
        BufferKind, BufferPlan, FidelityProfile, Invariant, OperationGraph, Precision,
        SolverContract,
    };

    static CONTRACT: SolverContract = SolverContract {
        name: "test.counter",
        summary: "a domain that counts steps",
        governing_equations: &["dn/dt = 1"],
        discretization: "none",
        integrator: "none",
        assumptions: &["nothing physical happens"],
        valid_regime: "tests only",
        stability: "see the configured limit",
        conserves: &[],
        known_non_conservation: &["everything; this is not a physical model"],
        fidelity: FidelityProfile::Interactive,
        precisions: &[Precision::Accurate64],
        deterministic: true,
        differentiable: false,
        validation_cases: &["none"],
        references: &[],
    };

    /// A domain that counts steps, optionally poisons itself, and reports a
    /// configurable stability limit.
    struct Counter {
        name: String,
        steps: u64,
        limit: f64,
        preferred: f64,
        poison_at: Option<u64>,
    }

    impl Counter {
        fn new(name: &str) -> Counter {
            Counter {
                name: name.to_string(),
                steps: 0,
                limit: f64::INFINITY,
                preferred: 0.1,
                poison_at: None,
            }
        }
    }

    impl Domain for Counter {
        fn name(&self) -> &str {
            &self.name
        }
        fn contract(&self) -> &'static SolverContract {
            &CONTRACT
        }
        fn stable_step(&self) -> StableStep {
            if self.limit.is_infinite() {
                StableStep::unconditional(self.preferred)
            } else {
                StableStep::limited(
                    self.preferred,
                    self.limit,
                    StabilityReason::DiffusionExplicit,
                )
            }
        }
        fn prepare(&mut self, _ctx: &mut StepContext<'_>) {}
        fn advance(&mut self, _dt: f64, _ctx: &mut StepContext<'_>) {
            self.steps += 1;
        }
        fn observe(&self, out: &mut Observations) {
            let value = match self.poison_at {
                Some(at) if self.steps >= at => f64::NAN,
                _ => self.steps as f64,
            };
            out.record_invariant(format!("{}.count", self.name), Invariant::Energy, value);
        }
    }

    fn model_with(name: &str) -> CompiledModel {
        let mut buffers = BufferPlan::new();
        buffers.allocate(
            "state",
            BufferKind::ScalarField {
                nx: 8,
                ny: 8,
                halo: 1,
            },
        );
        buffers.require_scratch(64);
        CompiledModel {
            name: name.to_string(),
            dimensions: 2,
            fidelity: FidelityProfile::Interactive,
            precision: Precision::Accurate64,
            domains: Vec::new(),
            buffers,
            graph: OperationGraph::build(Vec::new()),
            observers: Vec::new(),
            visuals: Vec::new(),
            timestep: None,
            duration: None,
            notes: Vec::new(),
        }
    }

    fn simulation(domains: Vec<Box<dyn Domain>>) -> Simulation {
        Simulation::new(model_with("test"), domains)
    }

    #[test]
    fn a_run_advances_the_clock_and_stops_at_the_duration() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        let config = RunConfig::new().with_duration(1.0).with_timestep(0.1);
        let outcome = simulation.run(&config);

        assert_eq!(outcome.stop, StopReason::DurationReached);
        assert!(outcome.is_success());
        assert_eq!(outcome.clock.step, 10);
        assert!((outcome.clock.time - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_step_cap_stops_a_run_with_no_duration() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        let outcome = simulation.run(&RunConfig::new().with_max_steps(7).with_timestep(0.1));
        assert_eq!(outcome.stop, StopReason::StepLimit);
        assert_eq!(outcome.clock.step, 7);
    }

    /// A run with neither a duration nor a cap would never end. Saying so beats
    /// spinning forever or picking an arbitrary default.
    #[test]
    fn a_run_with_no_stopping_condition_is_refused() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        let outcome = simulation.run(&RunConfig::new().with_timestep(0.1));
        assert_eq!(outcome.stop, StopReason::NoStoppingCondition);
        assert_eq!(outcome.clock.step, 0);
        assert!(outcome.stop.describe().contains("duration:"));
    }

    /// Spec §9.1: the runtime reconciles the domains' stability limits and keeps the
    /// reason from whichever one binds.
    #[test]
    fn the_tightest_domain_sets_the_stability_limit() {
        let mut loose = Counter::new("loose");
        loose.limit = 1.0;
        let mut tight = Counter::new("tight");
        tight.limit = 0.01;

        let simulation = simulation(vec![Box::new(loose), Box::new(tight)]);
        let stability = simulation.stability();
        assert!((stability.max - 0.01).abs() < 1e-12);
        assert_eq!(stability.reason, StabilityReason::DiffusionExplicit);
        assert_eq!(simulation.limiting_domain(), Some("tight"));
    }

    #[test]
    fn a_domain_with_no_limit_reports_none() {
        let simulation = simulation(vec![Box::new(Counter::new("a"))]);
        assert!(simulation.stability().max.is_infinite());
        assert_eq!(simulation.limiting_domain(), None);
    }

    /// NFR-007. A clamped run would finish and be wrong; refusing names the domain,
    /// the limit, and the mechanism.
    #[test]
    fn an_unstable_timestep_is_refused_before_the_first_step() {
        let mut strict = Counter::new("heat");
        strict.limit = 0.001;
        let mut simulation = simulation(vec![Box::new(strict)]);

        let outcome = simulation.run(&RunConfig::new().with_duration(1.0).with_timestep(0.1));
        match &outcome.stop {
            StopReason::Unstable {
                domain,
                requested,
                limit,
                ..
            } => {
                assert_eq!(domain, "heat");
                assert_eq!(*requested, 0.1);
                assert!((*limit - 0.001).abs() < 1e-12);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(outcome.clock.step, 0, "nothing should have been stepped");
        let text = outcome.stop.describe();
        assert!(text.contains("REFUSED"), "{text}");
        assert!(
            text.contains("would have finished and been wrong"),
            "{text}"
        );
    }

    /// NFR-007 again: the first non-finite value halts the run and is recorded.
    #[test]
    fn a_non_finite_observation_halts_the_run() {
        let mut poisoned = Counter::new("blowup");
        poisoned.poison_at = Some(5);
        let mut simulation = simulation(vec![Box::new(poisoned)]);

        let outcome = simulation.run(&RunConfig::new().with_duration(10.0).with_timestep(0.1));
        match &outcome.stop {
            StopReason::NonFinite {
                observation, step, ..
            } => {
                assert_eq!(observation, "blowup.count");
                assert_eq!(*step, 5);
            }
            other => panic!("{other:?}"),
        }
        assert!(!outcome.is_success());
        assert_eq!(
            outcome.clock.step, 5,
            "the run stopped rather than continuing"
        );
        assert!(outcome.artifact.first_non_finite().is_some());
    }

    #[test]
    fn instability_checking_can_be_turned_off_to_study_a_blow_up() {
        let mut poisoned = Counter::new("blowup");
        poisoned.poison_at = Some(3);
        let mut simulation = simulation(vec![Box::new(poisoned)]);

        let mut config = RunConfig::new().with_max_steps(10).with_timestep(0.1);
        config.stop_on_non_finite = false;
        let outcome = simulation.run(&config);

        assert_eq!(outcome.stop, StopReason::StepLimit);
        assert_eq!(outcome.clock.step, 10, "the run continued past the NaN");
        // ...but the artifact still records that it happened.
        assert!(outcome.artifact.first_non_finite().is_some());
    }

    #[test]
    fn every_domain_is_advanced_each_step() {
        let mut simulation = simulation(vec![
            Box::new(Counter::new("a")),
            Box::new(Counter::new("b")),
        ]);
        simulation.run(&RunConfig::new().with_max_steps(4).with_timestep(0.1));
        simulation.observe();
        assert_eq!(simulation.observations().value("a.count"), Some(4.0));
        assert_eq!(simulation.observations().value("b.count"), Some(4.0));
    }

    /// The cadence must not depend on the units' scale: a femtosecond model samples
    /// exactly as often as the same model in seconds.
    #[test]
    fn the_cadence_holds_at_femtosecond_scale() {
        for scale in [1.0, 1e-15] {
            let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
            let mut config = RunConfig::new().with_duration(1.0 * scale).with_timestep(0.01 * scale);
            config.sample_interval = Some(0.1 * scale);
            let samples = simulation.run(&config).artifact.timeline().len();
            assert_eq!(samples, 11, "at scale {scale:e}: the initial state and ten more");
        }
    }

    #[test]
    fn observations_are_sampled_on_the_configured_cadence() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        let mut config = RunConfig::new().with_duration(1.0).with_timestep(0.01);
        config.sample_interval = Some(0.1);
        let outcome = simulation.run(&config);

        // 100 steps, sampled every 0.1 s: about ten samples plus the initial one.
        let samples = outcome.artifact.timeline().len();
        assert!((10..=13).contains(&samples), "got {samples} samples");
        assert_eq!(
            outcome.artifact.timeline()[0].step,
            0,
            "the initial state is recorded"
        );
    }

    #[test]
    fn sampling_every_step_records_every_step() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        let config = RunConfig::new().with_max_steps(5).with_timestep(0.1);
        let outcome = simulation.run(&config);
        assert_eq!(
            outcome.artifact.timeline().len(),
            6,
            "five steps plus the initial state"
        );
    }

    #[test]
    fn the_artifact_records_the_stability_margin_and_solver_contracts() {
        let mut tight = Counter::new("heat");
        tight.limit = 0.5;
        let mut model = model_with("run");
        model.domains.push(lattice_ir::DomainSpec {
            id: lattice_ir::DomainId::from_index(0),
            name: "heat".to_string(),
            family: "test.counter".to_string(),
            summary: "a counter".to_string(),
            buffers: Vec::new(),
            contract: Some(&CONTRACT),
        });

        let mut simulation = Simulation::new(model, vec![Box::new(tight)]);
        let outcome = simulation.run(&RunConfig::new().with_duration(1.0).with_timestep(0.25));

        let json = outcome.artifact.to_json().to_compact_string();
        assert!(json.contains("stability_margin"), "{json}");
        assert!(json.contains("limiting_domain"), "{json}");
        // The contract travels with the results (NFR-003).
        assert!(json.contains("this is not a physical model"), "{json}");
    }

    /// FR-011: the same model and configuration must produce the same artifact hash.
    #[test]
    fn runs_are_reproducible() {
        let hash_of = || {
            let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
            let outcome = simulation.run(&RunConfig::new().with_max_steps(20).with_timestep(0.05));
            outcome.artifact.content_hash()
        };
        assert_eq!(hash_of(), hash_of());
    }

    #[test]
    fn a_model_timestep_is_used_when_the_config_does_not_override_it() {
        let mut model = model_with("m");
        model.timestep = Some(0.02);
        model.duration = Some(0.2);
        let mut simulation = Simulation::new(model, vec![Box::new(Counter::new("a"))]);

        let config = RunConfig::from_model(simulation.model());
        let outcome = simulation.run(&config);
        assert!((outcome.timestep - 0.02).abs() < 1e-12);
        assert_eq!(outcome.clock.step, 10);
    }

    #[test]
    fn a_config_timestep_overrides_the_model() {
        let mut model = model_with("m");
        model.timestep = Some(0.02);
        let mut simulation = Simulation::new(model, vec![Box::new(Counter::new("a"))]);
        let outcome = simulation.run(&RunConfig::new().with_max_steps(1).with_timestep(0.5));
        assert!((outcome.timestep - 0.5).abs() < 1e-12);
    }

    /// With nothing specified, the runtime takes the domains' preferred cadence.
    #[test]
    fn the_preferred_step_is_used_when_nothing_else_is_given() {
        let mut counter = Counter::new("a");
        counter.preferred = 0.004;
        let mut simulation = simulation(vec![Box::new(counter)]);
        let outcome = simulation.run(&RunConfig::new().with_max_steps(3));
        assert!((outcome.timestep - 0.004).abs() < 1e-12);
    }

    #[test]
    fn the_scratch_arena_is_sized_from_the_buffer_plan() {
        let simulation = simulation(vec![Box::new(Counter::new("a"))]);
        assert_eq!(simulation.arena.capacity(), 64);
    }

    #[test]
    fn config_from_a_model_takes_the_tightest_sample_interval() {
        let mut model = model_with("m");
        model.observers.push(lattice_ir::ObserverSpec {
            id: lattice_ir::ObserverId::from_index(0),
            target: "a".to_string(),
            interval: Some(1.0),
        });
        model.observers.push(lattice_ir::ObserverSpec {
            id: lattice_ir::ObserverId::from_index(1),
            target: "b".to_string(),
            interval: Some(0.25),
        });
        let config = RunConfig::from_model(&model);
        assert_eq!(config.sample_interval, Some(0.25));
    }

    #[test]
    fn stop_reasons_read_clearly() {
        assert!(StopReason::DurationReached.is_success());
        assert!(StopReason::StepLimit.is_success());
        assert!(!StopReason::NoStoppingCondition.is_success());
        assert!(
            StopReason::NonFinite {
                observation: "x".to_string(),
                step: 3,
                time: 0.3,
            }
            .describe()
            .contains("HALTED")
        );
    }

    #[test]
    fn graph_schedule_orders_dependencies_and_runs_independent_domains_concurrently() {
        use lattice_ir::{DomainId, Operation, OperationKind};
        use std::sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        };
        struct Probe {
            name: String,
            entered: Arc<AtomicUsize>,
            log: Arc<Mutex<Vec<String>>>,
        }
        impl Domain for Probe {
            fn name(&self) -> &str {
                &self.name
            }
            fn contract(&self) -> &'static SolverContract {
                &CONTRACT
            }
            fn stable_step(&self) -> StableStep {
                StableStep::unconditional(0.1)
            }
            fn prepare(&mut self, _: &mut StepContext<'_>) {
                self.log
                    .lock()
                    .unwrap()
                    .push(format!("prepare {}", self.name));
            }
            fn advance(&mut self, _: f64, ctx: &mut StepContext<'_>) {
                assert_eq!(ctx.executor.threads(), 1, "no nested pool dispatch");
                let deadline = Instant::now() + std::time::Duration::from_secs(10);
                self.entered.fetch_add(1, Ordering::SeqCst);
                while self.entered.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
                    std::thread::yield_now();
                }
                assert_eq!(
                    self.entered.load(Ordering::SeqCst),
                    2,
                    "both independent advances must overlap"
                );
                let mut frame = ctx.arena.frame();
                frame.alloc_zeroed(8)[0] = 42.0;
                self.log
                    .lock()
                    .unwrap()
                    .push(format!("advance {}", self.name));
            }
            fn observe(&self, _: &mut Observations) {}
        }
        let mut model = model_with("dag");
        let mut operations = Vec::new();
        for index in 0..2 {
            for kind in [OperationKind::Prepare, OperationKind::Advance] {
                // No explicit buffers: ownership alone must order prepare before advance.
                operations.push(
                    Operation::new(format!("{kind:?} {index}"), kind)
                        .in_domain(DomainId::from_index(index))
                        .costing(65536.0),
                );
            }
        }
        model.graph = OperationGraph::build(operations);
        let entered = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let domains = (0..2)
            .map(|index| {
                Box::new(Probe {
                    name: index.to_string(),
                    entered: entered.clone(),
                    log: log.clone(),
                }) as Box<dyn Domain>
            })
            .collect();
        let mut simulation =
            Simulation::new(model, domains).with_executor(Executor::with_threads(2));
        simulation.step(0.1);
        let log = log.lock().unwrap();
        assert!(log[..2].iter().all(|s| s.starts_with("prepare")), "{log:?}");
        assert!(log[2..].iter().all(|s| s.starts_with("advance")), "{log:?}");
    }

    #[test]
    fn graph_execution_preserves_the_scalar_artifact_hash() {
        use lattice_ir::{DomainId, Operation, OperationKind};
        let run = |threads| {
            let mut model = model_with("deterministic dag");
            model.graph = OperationGraph::build(
                (0..3)
                    .flat_map(|i| {
                        [OperationKind::Prepare, OperationKind::Advance].map(|kind| {
                            Operation::new(format!("{kind:?}{i}"), kind)
                                .in_domain(DomainId::from_index(i))
                                .costing(65536.0)
                        })
                    })
                    .collect(),
            );
            let domains = (0..3)
                .map(|i| Box::new(Counter::new(&format!("counter{i}"))) as Box<dyn Domain>)
                .collect();
            Simulation::new(model, domains)
                .with_executor(Executor::with_threads(threads))
                .run(&RunConfig::new().with_max_steps(20))
                .artifact
                .content_hash()
        };
        assert_eq!(run(1), run(3));
    }

    #[test]
    fn a_small_cross_domain_dependency_keeps_advance_before_dependent_prepare() {
        use lattice_ir::{BufferId, DomainId, Operation, OperationKind};
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        struct Dependent {
            producer: bool,
            ready: Arc<AtomicBool>,
        }
        impl Domain for Dependent {
            fn name(&self) -> &str {
                if self.producer {
                    "producer"
                } else {
                    "consumer"
                }
            }
            fn contract(&self) -> &'static SolverContract {
                &CONTRACT
            }
            fn stable_step(&self) -> StableStep {
                StableStep::unconditional(0.1)
            }
            fn prepare(&mut self, _: &mut StepContext<'_>) {
                if !self.producer {
                    assert!(self.ready.load(Ordering::SeqCst));
                }
            }
            fn advance(&mut self, _: f64, _: &mut StepContext<'_>) {
                if self.producer {
                    self.ready.store(true, Ordering::SeqCst);
                } else {
                    self.ready.store(false, Ordering::SeqCst);
                }
            }
            fn observe(&self, _: &mut Observations) {}
        }
        let mut model = model_with("cross-domain dependency");
        let a = DomainId::from_index(0);
        let b = DomainId::from_index(1);
        let shared = BufferId::from_index(0);
        model.graph = OperationGraph::build(vec![
            Operation::new("prepare a", OperationKind::Prepare).in_domain(a),
            Operation::new("advance a", OperationKind::Advance)
                .in_domain(a)
                .writing(shared),
            Operation::new("prepare b", OperationKind::Prepare)
                .in_domain(b)
                .reading(shared),
            Operation::new("advance b", OperationKind::Advance).in_domain(b),
        ]);
        let ready = Arc::new(AtomicBool::new(false));
        let domains = [true, false]
            .into_iter()
            .map(|producer| {
                Box::new(Dependent {
                    producer,
                    ready: ready.clone(),
                }) as Box<dyn Domain>
            })
            .collect();
        let mut simulation =
            Simulation::new(model, domains).with_executor(Executor::with_threads(2));
        for _ in 0..3 {
            simulation.step(0.1);
        }
        assert!(!ready.load(Ordering::SeqCst));
    }
}
