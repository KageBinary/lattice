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

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

mod schedule;

use lattice_ir::{
    Arena, CompiledModel, Domain, Executor, ObserveContext, Observations, Observer, ObserverGap,
    StabilityReason, StableStep, StepContext,
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

/// Why an observer was switched off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObserverFaultKind {
    /// It panicked.
    Panic,
    /// It published a row its contract's `outputs` does not declare.
    UndeclaredRow,
    /// It published a row in a unit other than the one `outputs` declares for it.
    WrongUnit,
    /// It published a row of kind `Invariant`, which only an engine solver may claim.
    InvariantRow,
    /// It published the same row twice in one call, which would give the timeline two
    /// columns of one name.
    DuplicateRow,
    /// It returned a curve its contract's `outputs` does not declare.
    UndeclaredCurve,
    /// It returned a curve named like one a domain or observer already returned.
    DuplicateCurve,
}

/// An observer that was switched off (diagnostic `E0951`).
///
/// Only a panic, or a breach of the observer's own contract, is caught. An observer that
/// loops forever, overflows the stack or aborts still hangs or kills the run.
#[derive(Clone, PartialEq, Debug)]
pub struct ObserverFault {
    /// The observer's contract name.
    pub observer: String,
    /// What it did.
    pub kind: ObserverFaultKind,
    /// The step of the sample it failed at.
    pub step: u64,
    /// The time of that sample, seconds.
    pub time: f64,
    /// The panic message, or which row it published without declaring.
    pub message: String,
}

impl ObserverFault {
    /// The diagnostic code.
    pub const CODE: &'static str = "E0951";

    /// A message for the user.
    pub fn describe(&self) -> String {
        let what = match self.kind {
            ObserverFaultKind::Panic => "panicked",
            ObserverFaultKind::UndeclaredRow
            | ObserverFaultKind::WrongUnit
            | ObserverFaultKind::InvariantRow
            | ObserverFaultKind::DuplicateRow
            | ObserverFaultKind::UndeclaredCurve
            | ObserverFaultKind::DuplicateCurve => "broke its contract",
        };
        format!(
            "observer `{}` {what} at step {} and was disabled: {}",
            self.observer, self.step, self.message
        )
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
    /// Every observer disabled on this simulation so far, the same list as
    /// [`Simulation::observer_faults`] and the `E0951` warnings in the artifact, on
    /// every return path. The run carried on without them, so this does not change
    /// [`RunOutcome::is_success`].
    pub observer_faults: Vec<ObserverFault>,
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
    /// Attached observers, in attachment order.
    observers: Vec<ObserverSlot>,
    /// One observer's rows, before they are appended to the sample. Kept apart so the
    /// observer reads the domains' rows through `&` while it writes its own.
    observer_rows: Observations,
    /// Observers that panicked, in the order they did.
    observer_faults: Vec<ObserverFault>,
    /// The timestep of the latest step, which an observer's `due` is told.
    last_dt: f64,
}

/// An attached observer, and whether it is still running.
///
/// What the run needs from the observer's contract is copied here by `add_observer`, and
/// `Observer::contract` is not called again: it is user code, and a panic in it outside
/// the `catch_unwind` around `observe` would take the run down.
struct ObserverSlot {
    observer: Box<dyn Observer>,
    /// The contract's `name`, for the fault record.
    name: String,
    /// The contract's `outputs`: the rows and curves it may publish, with their units.
    outputs: Vec<(std::borrow::Cow<'static, str>, std::borrow::Cow<'static, str>)>,
    /// The domains its contract lists in `reads`, kept here so the context can borrow
    /// them while the observer is borrowed mutably.
    reads: Vec<std::borrow::Cow<'static, str>>,
    disabled: bool,
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
            observers: Vec::new(),
            observer_rows: Observations::new(),
            observer_faults: Vec::new(),
            last_dt: 0.0,
        }
    }

    /// Attach an observer, after auditing its contract.
    ///
    /// Observers run in attachment order, after every domain has published, and only
    /// at the samples their [`Observer::due`] accepts. The attach is refused, with
    /// every gap returned and the observer not attached, for:
    ///
    /// - an empty `name`, `summary`, `reads`, `outputs` or `limitations`, or no backend;
    /// - a `reads` name that is not a domain of this model;
    /// - an `outputs` name declared twice, or already declared by an attached observer
    ///   ([`ObserverGap::OutputTaken`]);
    /// - an `outputs` name a domain already publishes
    ///   ([`ObserverGap::OutputShadowsDomain`]); one scratch sample of the domains
    ///   supplies those names.
    ///
    /// At run time a row outside the declared `outputs`, or in another unit than the one
    /// declared there, or the same row twice in one call, disables the observer
    /// (`E0951`). The observer reaches only the
    /// domains its contract lists in `reads`.
    pub fn add_observer(&mut self, observer: Box<dyn Observer>) -> Result<(), Vec<ObserverGap>> {
        let names: Vec<&str> = self.domains.iter().map(|domain| domain.name()).collect();
        let contract = observer.contract();
        let mut gaps = contract.audit(&names);

        let mut sample = Observations::new();
        for domain in &self.domains {
            domain.observe(&mut sample);
        }
        for (index, (name, _)) in contract.outputs.iter().enumerate() {
            let name = name.as_ref();
            let repeated = contract.outputs[..index].iter().any(|(other, _)| other == name);
            let taken = self
                .observers
                .iter()
                .any(|slot| slot.outputs.iter().any(|(other, _)| other == name));
            if repeated || taken {
                gaps.push(ObserverGap::OutputTaken(name.to_string()));
            }
            if sample.get(name).is_some() {
                gaps.push(ObserverGap::OutputShadowsDomain(name.to_string()));
            }
        }
        if !gaps.is_empty() {
            return Err(gaps);
        }
        let (name, reads, outputs) =
            (contract.name.to_string(), contract.reads.clone(), contract.outputs.clone());
        self.observers.push(ObserverSlot { observer, name, reads, outputs, disabled: false });
        Ok(())
    }

    /// Attach several observers in order, as [`Simulation::add_observer`] does one.
    ///
    /// This is how a compiled model's own observers (its `observe sum(law(set))`
    /// statements) are attached, before any the caller adds. Stops at the first refusal
    /// and says which observer and why; the ones before it stay attached.
    pub fn add_observers(&mut self, observers: Vec<Box<dyn Observer>>) -> Result<(), String> {
        for observer in observers {
            let name = observer.contract().name.to_string();
            self.add_observer(observer).map_err(|gaps| {
                let gaps: Vec<String> = gaps.iter().map(ToString::to_string).collect();
                format!("observer `{name}` was refused: {}", gaps.join("; "))
            })?;
        }
        Ok(())
    }

    /// Explain a halt on `observation` that an attached observer published (`E0950`).
    ///
    /// For a run that stopped as [`StopReason::NonFinite`]: the domains are still in the
    /// state the observer was shown, so each enabled observer that declares the row is
    /// asked to trace its computation against it ([`Observer::explain_non_finite`]),
    /// with no replay. `None` when no observer published the row, or none can say. An
    /// observer that panics while explaining is skipped.
    pub fn explain_observer_halt(&self, observation: &str) -> Option<lattice_ir::LawFault> {
        self.observers.iter().filter(|slot| !slot.disabled).find_map(|slot| {
            if !slot.observer.contract().outputs.iter().any(|(name, _)| name == observation) {
                return None;
            }
            let ctx = ObserveContext::new(self.clock.time, self.clock.step, &self.domains, &self.observations)
                .restricted_to(&slot.reads);
            catch_unwind(AssertUnwindSafe(|| slot.observer.explain_non_finite(&ctx, observation))).ok().flatten()
        })
    }

    /// Observers disabled so far for panicking (`E0951`).
    pub fn observer_faults(&self) -> &[ObserverFault] {
        &self.observer_faults
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

    /// Every curve across all domains and observers, as accumulated so far.
    ///
    /// Each enabled observer's `curves()` is called once per call here, so it is for the
    /// end of a run or a viewer between frames, not an inner loop. An observer that
    /// panics, returns a curve its contract's `outputs` does not declare, or returns one
    /// named like a curve already collected is disabled, its curves from this call are
    /// dropped, and the fault is recorded (`E0951`). A disabled observer is not asked.
    pub fn curves(&mut self) -> Vec<lattice_ir::Curve> {
        let mut curves: Vec<lattice_ir::Curve> =
            self.domains.iter().flat_map(|domain| domain.curves()).collect();
        for slot in &mut self.observers {
            if slot.disabled {
                continue;
            }
            let fault = match catch_unwind(AssertUnwindSafe(|| slot.observer.curves())) {
                Err(payload) => Some((ObserverFaultKind::Panic, panic_message(payload.as_ref()))),
                Ok(found) => {
                    let outputs = &slot.outputs;
                    let mut breach = None;
                    for (index, curve) in found.iter().enumerate() {
                        let name = curve.name.as_str();
                        if !outputs.iter().any(|(declared, _)| declared == name) {
                            breach = Some((
                                ObserverFaultKind::UndeclaredCurve,
                                format!(
                                    "it returned curve `{name}`, which its contract's \
                                     `outputs` does not declare"
                                ),
                            ));
                        } else if curves.iter().chain(&found[..index]).any(|c| c.name == name) {
                            breach = Some((
                                ObserverFaultKind::DuplicateCurve,
                                format!("it returned curve `{name}`, which already exists"),
                            ));
                        }
                        if breach.is_some() {
                            break;
                        }
                    }
                    if breach.is_none() {
                        curves.extend(found);
                    }
                    breach
                }
            };
            if let Some((kind, message)) = fault {
                slot.disabled = true;
                self.observer_faults.push(ObserverFault {
                    observer: slot.name.clone(),
                    kind,
                    step: self.clock.step,
                    time: self.clock.time,
                    message,
                });
            }
        }
        curves
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

        self.last_dt = dt;
        self.clock.advance(dt);
    }

    /// Refresh the observation set from every domain, then from every observer.
    ///
    /// Observers run in attachment order, each at a sample it is due for, and each sees
    /// the rows published before it. One that panics, or publishes a row its contract
    /// does not declare in `outputs`, is disabled, that sample's rows from it are
    /// dropped, and the fault is recorded (`E0951`). A row in another unit than the one
    /// `outputs` declares for it is the same breach, and so is the same row twice in one
    /// call.
    ///
    /// Each call is one sample, and it advances the observers' [`Observer::due`] state,
    /// so call it once per instant: [`Simulation::run`] does, and a caller driving the
    /// clock itself must not call it twice for the same instant.
    pub fn observe(&mut self) {
        self.observations.clear();
        for domain in &self.domains {
            domain.observe(&mut self.observations);
        }
        let (time, step, dt) = (self.clock.time, self.clock.step, self.last_dt);
        for slot in &mut self.observers {
            if slot.disabled {
                continue;
            }
            self.observer_rows.clear();
            let ctx = ObserveContext::new(time, step, &self.domains, &self.observations)
                .restricted_to(&slot.reads);
            let rows = &mut self.observer_rows;
            let observer = &mut slot.observer;
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                if observer.due(time, dt) {
                    observer.observe(&ctx, rows);
                }
            }));
            let fault = match outcome {
                Ok(()) => {
                    let outputs = &slot.outputs;
                    let rows = &self.observer_rows;
                    rows.iter().enumerate().find_map(|(index, row)| {
                        if matches!(row.kind, lattice_ir::ObservationKind::Invariant(_)) {
                            return Some((
                                ObserverFaultKind::InvariantRow,
                                format!(
                                    "it published row `{}` as an invariant, which only an \
                                     engine solver may claim; publish a metric",
                                    row.name
                                ),
                            ));
                        }
                        match outputs.iter().find(|(name, _)| *name == row.name) {
                            None => Some((
                                ObserverFaultKind::UndeclaredRow,
                                format!(
                                    "it published row `{}`, which its contract's `outputs` \
                                     does not declare",
                                    row.name
                                ),
                            )),
                            Some((_, unit)) if *unit != row.unit => Some((
                                ObserverFaultKind::WrongUnit,
                                format!(
                                    "it published row `{}` in `{}`, but its contract's \
                                     `outputs` declares `{}`",
                                    row.name, row.unit, unit
                                ),
                            )),
                            Some(_) if rows.iter().take(index).any(|earlier| earlier.name == row.name) => {
                                Some((
                                    ObserverFaultKind::DuplicateRow,
                                    format!(
                                        "it published row `{}` twice in one call",
                                        row.name
                                    ),
                                ))
                            }
                            Some(_) => None,
                        }
                    })
                }
                Err(payload) => Some((ObserverFaultKind::Panic, panic_message(payload.as_ref()))),
            };
            match fault {
                None => self.observations.append(&mut self.observer_rows),
                Some((kind, message)) => {
                    slot.disabled = true;
                    self.observer_rows.clear();
                    self.observer_faults.push(ObserverFault {
                        observer: slot.name.clone(),
                        kind,
                        step,
                        time,
                        message,
                    });
                }
            }
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
                self.warn_of_observer_faults(&mut artifact);
                return RunOutcome {
                    stop,
                    clock: self.clock,
                    timestep: 0.0,
                    artifact,
                    observer_faults: self.observer_faults.clone(),
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
                self.warn_of_observer_faults(&mut artifact);
                return RunOutcome {
                    stop,
                    clock: self.clock,
                    timestep,
                    artifact,
                    observer_faults: self.observer_faults.clone(),
                };
            }
        };

        // Sample the initial state, so the artifact records where the run began.
        self.last_dt = timestep;
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
        self.warn_of_observer_faults(&mut artifact);

        RunOutcome {
            stop,
            clock: self.clock,
            timestep,
            artifact,
            observer_faults: self.observer_faults.clone(),
        }
    }

    /// One `E0951` warning per observer fault so far.
    fn warn_of_observer_faults(&self, artifact: &mut RunArtifact) {
        for fault in &self.observer_faults {
            artifact.warn(format!("error[{}]: {}", ObserverFault::CODE, fault.describe()));
        }
    }

    /// Replay up to `steps` steps of `dt` with every domain watching its user laws, and
    /// return the first law that produced a non-finite value (NFR-007).
    ///
    /// For a run that [`Simulation::run`] halted as [`StopReason::NonFinite`] at step `N`:
    /// build the same model again, with the same timestep, and replay `N` steps. Runs are
    /// bit-reproducible, so the replay reaches the same states, and nothing is retained
    /// during an ordinary step to make this possible. It stops at the first fault, so the
    /// clock afterwards says which step it was (step 0 is the initial state, before any
    /// step). `None` means the steps all passed without a user law making a non-finite
    /// value: the NaN came from something that is not one.
    ///
    /// Replaying costs about one more run, and the watching re-evaluates each user law
    /// at every step, so it is for the failure path alone. A model with no user law in
    /// any domain is not replayed at all: there is nothing to explain.
    pub fn trace_non_finite(&mut self, dt: f64, steps: u64) -> Option<lattice_ir::LawFault> {
        let mut any_laws = false;
        for domain in &mut self.domains {
            any_laws |= domain.trace_laws(true);
        }
        if !any_laws {
            for domain in &mut self.domains {
                domain.trace_laws(false);
            }
            return None;
        }
        let first = |domains: &[Box<dyn Domain>]| domains.iter().find_map(|domain| domain.law_fault());
        let mut found = first(&self.domains);
        for _ in 0..steps {
            if found.is_some() {
                break;
            }
            self.step(dt);
            found = first(&self.domains);
        }
        for domain in &mut self.domains {
            domain.trace_laws(false);
        }
        found
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

/// The text of a panic payload, which is a `&str` or a `String` unless the panic was
/// raised with `panic_any`.
fn panic_message(payload: &(dyn core::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(no message)".to_string())
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
        /// Whether a replay has asked it to watch its laws.
        watching: bool,
        /// Whether it has any user law to watch.
        has_laws: bool,
        /// A curve it publishes, if any.
        curve: Option<String>,
    }

    impl Counter {
        fn new(name: &str) -> Counter {
            Counter {
                name: name.to_string(),
                steps: 0,
                limit: f64::INFINITY,
                preferred: 0.1,
                poison_at: None,
                watching: false,
                has_laws: true,
                curve: None,
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
        fn curves(&self) -> Vec<lattice_ir::Curve> {
            self.curve
                .iter()
                .map(|name| {
                    lattice_ir::Curve::new(name.clone(), ("x", "1"), ("y", "1"), vec![0.0], vec![1.0])
                })
                .collect()
        }
        fn trace_laws(&mut self, on: bool) -> bool {
            self.watching = on;
            self.has_laws
        }
        fn law_fault(&self) -> Option<lattice_ir::LawFault> {
            // Its "law" goes non-finite on the step it poisons itself, when watched.
            let poisoned = self.poison_at.is_some_and(|at| self.steps >= at);
            (self.watching && poisoned).then(|| lattice_ir::LawFault {
                domain: self.name.clone(),
                law: "count".to_string(),
                site: lattice_ir::FaultSite::Particle(self.steps as usize),
                at: lattice_expr::NonFiniteAt {
                    reg: 0,
                    op: lattice_expr::Op::Const(f64::NAN),
                    origin: None,
                    operands: Vec::new(),
                    value: f64::NAN,
                },
                input: None,
            })
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

    /// A halted run is replayed on a fresh simulation: it stops at the first step a
    /// domain reports a fault, leaves the clock there, and switches the watching off.
    #[test]
    fn a_replay_stops_at_the_first_fault_and_stops_watching() {
        let poisoned = |at| {
            let mut counter = Counter::new("a");
            counter.poison_at = Some(at);
            simulation(vec![Box::new(counter)])
        };
        let mut first = poisoned(4);
        let outcome = first.run(&RunConfig::new().with_max_steps(50).with_timestep(0.1));
        let StopReason::NonFinite { step, .. } = outcome.stop else { panic!("{:?}", outcome.stop) };
        assert_eq!(step, 4);

        let mut replay = poisoned(4);
        let fault = replay.trace_non_finite(0.1, step).expect("the poisoned domain reports");
        assert_eq!((fault.domain.as_str(), fault.law.as_str()), ("a", "count"));
        assert_eq!(replay.clock().step, 4, "the replay stopped where the fault was");
        assert!(replay.domains()[0].law_fault().is_none(), "and stopped watching");

        // A replay too short to reach the fault finds none.
        let mut short = poisoned(4);
        assert!(short.trace_non_finite(0.1, 2).is_none());
        assert_eq!(short.clock().step, 2);
    }

    /// With no user law anywhere there is nothing to explain, so nothing is replayed.
    #[test]
    fn a_model_with_no_user_laws_is_not_replayed() {
        let mut counter = Counter::new("a");
        counter.poison_at = Some(1);
        counter.has_laws = false;
        let mut replay = simulation(vec![Box::new(counter)]);
        assert!(replay.trace_non_finite(0.1, 50).is_none());
        assert_eq!(replay.clock().step, 0, "not a single step");
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
            law: None,
        });
        model.observers.push(lattice_ir::ObserverSpec {
            id: lattice_ir::ObserverId::from_index(1),
            target: "b".to_string(),
            interval: Some(0.25),
            law: None,
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

    /// An observer for the tests: it records `<name>.steps`, the step count of the
    /// `Counter` called `a`, read through the context, and `<name>.calls`.
    struct Probe {
        contract: lattice_ir::ObserverContract,
        calls: u32,
        panic_on_call: Option<u32>,
        /// Publishes a row its contract does not declare, on this call.
        undeclared_on_call: Option<u32>,
        /// Publishes `<name>.steps` in the wrong unit, on this call.
        wrong_unit_on_call: Option<u32>,
        /// Publishes `<name>.steps` twice, on this call.
        duplicate_on_call: Option<u32>,
        /// Once set, `contract()` panics, as a plugin's could after attachment.
        contract_panics: std::sync::Arc<std::sync::atomic::AtomicBool>,
        /// Publishes NaN, on this call.
        nan_on_call: Option<u32>,
        /// Publishes `<name>.saw` as an invariant, on this call.
        invariant_on_call: Option<u32>,
        /// Returns this extra curve name from `curves()`.
        extra_curve: Option<String>,
        panic_in_curves: bool,
        /// How many times `curves()` has been called.
        curve_calls: std::sync::Arc<std::sync::atomic::AtomicU32>,
        /// Only every n-th sample is wanted.
        every: u32,
        asked: u32,
    }

    impl Probe {
        fn new(name: &'static str) -> Probe {
            Probe {
                contract: lattice_ir::ObserverContract {
                    name: name.into(),
                    summary: "a test observer".into(),
                    reads: vec!["a".into()],
                    outputs: ["calls", "steps", "saw", "curve"]
                        .into_iter()
                        .map(|row| (format!("{name}.{row}").into(), "1".into()))
                        .collect(),
                    side_effects: lattice_ir::SideEffects::None,
                    limitations: vec!["exists only to be tested".into()],
                    deterministic: true,
                    backends: lattice_ir::Backends::CPU,
                    differentiable: false,
                },
                calls: 0,
                panic_on_call: None,
                undeclared_on_call: None,
                wrong_unit_on_call: None,
                duplicate_on_call: None,
                contract_panics: Default::default(),
                nan_on_call: None,
                invariant_on_call: None,
                extra_curve: None,
                panic_in_curves: false,
                curve_calls: Default::default(),
                every: 1,
                asked: 0,
            }
        }

        fn name(&self) -> String {
            self.contract.name.to_string()
        }
    }

    impl Observer for Probe {
        fn contract(&self) -> &lattice_ir::ObserverContract {
            if self.contract_panics.load(std::sync::atomic::Ordering::SeqCst) {
                panic!("contract() called during the run");
            }
            &self.contract
        }
        fn observe(&mut self, ctx: &ObserveContext<'_>, out: &mut Observations) {
            self.calls += 1;
            let name = self.name();
            // Written before the panic, to show that a half-finished sample is dropped.
            out.record_metric(format!("{name}.calls"), f64::from(self.calls), "1");
            if self.panic_on_call == Some(self.calls) {
                panic!("probe gave up on call {}", self.calls);
            }
            if self.undeclared_on_call == Some(self.calls) {
                out.record_metric(format!("{name}.extra"), 1.0, "1");
            }
            let counter = ctx.domain::<Counter>("a").expect("domain `a` downcasts to Counter");
            let unit = if self.wrong_unit_on_call == Some(self.calls) { "m" } else { "1" };
            let steps = if self.nan_on_call == Some(self.calls) {
                f64::NAN
            } else {
                counter.steps as f64
            };
            out.record_metric(format!("{name}.steps"), steps, unit);
            if self.duplicate_on_call == Some(self.calls) {
                out.record_metric(format!("{name}.steps"), steps, unit);
            }
            let seen = ctx.observations().value("a.count").expect("domain rows come first");
            if self.invariant_on_call == Some(self.calls) {
                out.record_invariant(format!("{name}.saw"), lattice_ir::Invariant::Energy, seen);
            } else {
                out.record_metric(format!("{name}.saw"), seen, "1");
            }
            assert!(ctx.domain::<String>("a").is_none(), "the wrong type downcasts to nothing");
            assert!(ctx.domain::<Counter>("nope").is_none());
        }
        fn curves(&self) -> Vec<lattice_ir::Curve> {
            self.curve_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.panic_in_curves {
                panic!("no curve today");
            }
            let curve = |name: String| {
                lattice_ir::Curve::new(name, ("x", "1"), ("y", "1"), vec![0.0], vec![1.0])
            };
            let mut curves = vec![curve(format!("{}.curve", self.name()))];
            curves.extend(self.extra_curve.clone().map(curve));
            curves
        }
        fn due(&mut self, _time: f64, _dt: f64) -> bool {
            self.asked += 1;
            (self.asked - 1) % self.every == 0
        }
    }

    fn observed(probes: Vec<Probe>) -> Simulation {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        for probe in probes {
            simulation.add_observer(Box::new(probe)).expect("a complete contract attaches");
        }
        simulation
    }

    fn run5(simulation: &mut Simulation) -> RunOutcome {
        simulation.run(&RunConfig::new().with_max_steps(5).with_timestep(0.1))
    }

    fn column(outcome: &RunOutcome, name: &str) -> Vec<(u64, u64)> {
        outcome
            .artifact
            .timeline()
            .iter()
            .filter_map(|sample| {
                sample.values.iter().find(|(n, _)| n == name).map(|(_, v)| (sample.step, v.to_bits()))
            })
            .collect()
    }

    #[test]
    fn add_observer_refuses_an_incomplete_contract() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);

        let mut no_limits = Probe::new("p");
        no_limits.contract.limitations.clear();
        assert_eq!(
            simulation.add_observer(Box::new(no_limits)).unwrap_err(),
            vec![ObserverGap::NoLimitations]
        );

        let mut unknown = Probe::new("p");
        unknown.contract.reads = vec!["b".into()];
        assert_eq!(
            simulation.add_observer(Box::new(unknown)).unwrap_err(),
            vec![ObserverGap::UnknownRead("b".to_string())]
        );

        let mut no_outputs = Probe::new("p");
        no_outputs.contract.outputs.clear();
        assert_eq!(
            simulation.add_observer(Box::new(no_outputs)).unwrap_err(),
            vec![ObserverGap::NoOutputs]
        );

        // None of the three was attached.
        let outcome = run5(&mut simulation);
        assert!(outcome.artifact.timeline().iter().all(|s| s.values.len() == 1));
    }

    #[test]
    fn observers_run_after_the_domains_in_attachment_order() {
        let mut simulation = observed(vec![Probe::new("first"), Probe::new("second")]);
        let outcome = run5(&mut simulation);
        let names: Vec<&str> =
            outcome.artifact.timeline()[0].values.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "a.count",
                "first.calls",
                "first.steps",
                "first.saw",
                "second.calls",
                "second.steps",
                "second.saw",
            ]
        );
        // One call per sample, the initial state included, and the observer saw the
        // domain's row for the same sample.
        assert_eq!(column(&outcome, "first.calls").len(), 6);
        assert_eq!(column(&outcome, "first.saw"), column(&outcome, "a.count"));
        assert!(outcome.observer_faults.is_empty());
        assert!(outcome.artifact.warnings().is_empty());

        // The series table carries every row's unit and kind, observers' included.
        let json = outcome.artifact.to_json();
        let series = json.get("series").expect("the artifact has a series table");
        let entry = series.get("second.steps").expect("an observer's row is in the table");
        assert_eq!(entry.get("unit"), Some(&Json::from("1")));
        assert_eq!(entry.get("kind"), Some(&Json::from("metric")));
        assert!(series.get("a.count").is_some(), "and so is a domain's");
    }

    #[test]
    fn an_observer_is_called_only_when_due() {
        let mut every_other = Probe::new("slow");
        every_other.every = 2;
        let mut simulation = observed(vec![every_other, Probe::new("fast")]);
        let outcome = run5(&mut simulation);
        let slow: Vec<u64> = column(&outcome, "slow.calls").iter().map(|&(step, _)| step).collect();
        assert_eq!(slow, [0, 2, 4], "samples 1, 3 and 5 of six are due");
        assert_eq!(column(&outcome, "fast.calls").len(), 6);
    }

    #[test]
    fn an_observer_is_told_the_timestep() {
        struct Dt(lattice_ir::ObserverContract, std::sync::Arc<std::sync::Mutex<Vec<(f64, f64)>>>);
        impl Observer for Dt {
            fn contract(&self) -> &lattice_ir::ObserverContract {
                &self.0
            }
            fn observe(&mut self, _: &ObserveContext<'_>, _: &mut Observations) {}
            fn due(&mut self, time: f64, dt: f64) -> bool {
                self.1.lock().unwrap().push((time, dt));
                true
            }
        }
        let log = Default::default();
        let mut simulation = observed(Vec::new());
        simulation.add_observer(Box::new(Dt(Probe::new("dt").contract, std::sync::Arc::clone(&log)))).unwrap();
        run5(&mut simulation);
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 6);
        assert!(log.iter().all(|&(_, dt)| dt == 0.1), "{log:?}");
        assert_eq!(log[0].0, 0.0);
    }

    /// NFR-004: an observer that panics is disabled and the run goes on. Its third call
    /// panics, so rows exist for calls 1 and 2 only, the diagnostic is `E0951`, and
    /// every other row is bit-for-bit what a run without it records.
    #[test]
    fn a_panicking_observer_is_disabled_and_the_run_carries_on() {
        let mut bare = observed(Vec::new());
        let reference = run5(&mut bare);

        let mut bystander = Probe::new("bystander");
        bystander.every = 1;
        let mut crasher = Probe::new("crasher");
        crasher.panic_on_call = Some(3);
        let mut simulation = observed(vec![crasher, bystander]);
        let outcome = run5(&mut simulation);

        assert_eq!(outcome.stop, StopReason::StepLimit);
        assert!(outcome.is_success());
        assert_eq!(outcome.clock.step, 5);

        // Rows stop after call 2: the samples at steps 0 and 1. The third call's
        // half-written `crasher.calls` row is not kept.
        assert_eq!(column(&outcome, "crasher.calls").len(), 2);
        assert_eq!(column(&outcome, "crasher.steps").len(), 2);

        // The diagnostic.
        assert_eq!(outcome.observer_faults.len(), 1);
        let fault = &outcome.observer_faults[0];
        assert_eq!(fault.observer, "crasher");
        assert_eq!(fault.step, 2, "the third sample is at step 2");
        assert_eq!(ObserverFault::CODE, "E0951");
        assert_eq!(
            fault.describe(),
            "observer `crasher` panicked at step 2 and was disabled: probe gave up on call 3"
        );
        assert!(
            outcome.artifact.warnings().iter().any(|w| w.starts_with("error[E0951]: observer `crasher`")),
            "{:?}",
            outcome.artifact.warnings()
        );

        // Everything else is untouched, including an observer attached after it.
        assert_eq!(column(&outcome, "a.count"), column(&reference, "a.count"));
        assert_eq!(column(&outcome, "bystander.calls").len(), 6);
        assert_eq!(simulation.observer_faults(), outcome.observer_faults.as_slice());
    }

    #[test]
    fn a_panic_in_curves_is_recorded_not_propagated() {
        let mut probe = Probe::new("curvy");
        probe.panic_in_curves = true;
        let mut simulation = observed(vec![probe, Probe::new("fine")]);
        let outcome = run5(&mut simulation);
        assert!(outcome.is_success());
        assert_eq!(outcome.observer_faults.len(), 1);
        assert_eq!(outcome.observer_faults[0].observer, "curvy");
        let curves: Vec<&str> = outcome.artifact.curves().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(curves, ["fine.curve"]);
    }

    /// `curves()` is taken once per run, and never from an observer already disabled.
    #[test]
    fn curves_are_taken_once_and_not_from_a_disabled_observer() {
        use std::sync::atomic::Ordering;
        let counted = Probe::new("counted");
        let counted_calls = counted.curve_calls.clone();
        let mut dead = Probe::new("dead");
        dead.panic_on_call = Some(2);
        let dead_calls = dead.curve_calls.clone();
        let mut simulation = observed(vec![counted, dead]);
        let outcome = run5(&mut simulation);

        assert_eq!(outcome.observer_faults.len(), 1);
        assert_eq!(counted_calls.load(Ordering::SeqCst), 1, "once per run");
        assert_eq!(dead_calls.load(Ordering::SeqCst), 0, "disabled in `observe`: never asked");
        let curves: Vec<&str> = outcome.artifact.curves().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(curves, ["counted.curve"]);
        let published: Vec<String> = simulation.curves().into_iter().map(|c| c.name).collect();
        assert_eq!(published, ["counted.curve"], "and `Simulation::curves` agrees");
    }

    #[test]
    fn add_observer_refuses_an_output_another_observer_declares() {
        let mut simulation = observed(vec![Probe::new("p")]);
        // Same output names as the attached one.
        let gaps = simulation.add_observer(Box::new(Probe::new("p"))).unwrap_err();
        assert_eq!(
            gaps,
            vec![
                ObserverGap::OutputTaken("p.calls".to_string()),
                ObserverGap::OutputTaken("p.steps".to_string()),
                ObserverGap::OutputTaken("p.saw".to_string()),
                ObserverGap::OutputTaken("p.curve".to_string()),
            ]
        );
        // A name repeated within one contract is refused too.
        let mut twice = Probe::new("q");
        twice.contract.outputs.push(("q.calls".into(), "1".into()));
        assert_eq!(
            simulation.add_observer(Box::new(twice)).unwrap_err(),
            vec![ObserverGap::OutputTaken("q.calls".to_string())]
        );
        // Neither was attached: only `p` ever publishes.
        let outcome = run5(&mut simulation);
        assert_eq!(column(&outcome, "p.calls").len(), 6);
        assert_eq!(outcome.artifact.timeline()[0].values.len(), 4);
    }

    #[test]
    fn add_observer_refuses_an_output_a_domain_publishes() {
        let mut simulation = simulation(vec![Box::new(Counter::new("a"))]);
        let mut shadow = Probe::new("p");
        shadow.contract.outputs.push(("a.count".into(), "1".into()));
        assert_eq!(
            simulation.add_observer(Box::new(shadow)).unwrap_err(),
            vec![ObserverGap::OutputShadowsDomain("a.count".to_string())]
        );
    }

    /// A row outside `outputs` is a breach: the observer is disabled, that sample's rows
    /// from it are dropped, and `E0951` names the row.
    #[test]
    fn an_undeclared_row_disables_the_observer() {
        let mut breaker = Probe::new("breaker");
        breaker.undeclared_on_call = Some(3);
        let mut simulation = observed(vec![breaker, Probe::new("fine")]);
        let outcome = run5(&mut simulation);

        assert!(outcome.is_success());
        assert_eq!(column(&outcome, "breaker.calls").len(), 2, "calls 1 and 2 only");
        assert!(column(&outcome, "breaker.extra").is_empty(), "the undeclared row is not kept");
        assert_eq!(column(&outcome, "fine.calls").len(), 6);

        assert_eq!(outcome.observer_faults.len(), 1);
        let fault = &outcome.observer_faults[0];
        assert_eq!((fault.observer.as_str(), fault.step), ("breaker", 2));
        assert_eq!(fault.kind, ObserverFaultKind::UndeclaredRow);
        let text = fault.describe();
        assert!(
            text.contains("broke its contract") && text.contains("`breaker.extra`"),
            "{text}"
        );
        assert!(
            outcome.artifact.warnings().iter().any(|w| w.starts_with("error[E0951]:") && w.contains("breaker.extra")),
            "{:?}",
            outcome.artifact.warnings()
        );
    }

    /// The same declared row twice in one call is a breach: the timeline would get two
    /// columns of one name.
    #[test]
    fn a_duplicate_row_disables_the_observer() {
        let mut breaker = Probe::new("breaker");
        breaker.duplicate_on_call = Some(3);
        let mut simulation = observed(vec![breaker, Probe::new("fine")]);
        let outcome = run5(&mut simulation);

        assert!(outcome.is_success());
        assert_eq!(column(&outcome, "breaker.calls").len(), 2, "calls 1 and 2 only");
        assert_eq!(column(&outcome, "breaker.steps").len(), 2, "the bad sample is dropped");
        assert_eq!(column(&outcome, "fine.calls").len(), 6);
        for sample in outcome.artifact.timeline() {
            for (index, (name, _)) in sample.values.iter().enumerate() {
                assert!(
                    sample.values[..index].iter().all(|(other, _)| other != name),
                    "duplicate column `{name}` at step {}",
                    sample.step
                );
            }
        }

        assert_eq!(outcome.observer_faults.len(), 1);
        let fault = &outcome.observer_faults[0];
        assert_eq!((fault.observer.as_str(), fault.step), ("breaker", 2));
        assert_eq!(fault.kind, ObserverFaultKind::DuplicateRow);
        let text = fault.describe();
        assert!(
            text.contains("broke its contract") && text.contains("`breaker.steps`") && text.contains("twice"),
            "{text}"
        );
        assert!(
            outcome.artifact.warnings().iter().any(|w| w.starts_with("error[E0951]:") && w.contains("twice")),
            "{:?}",
            outcome.artifact.warnings()
        );
    }

    /// `contract()` is user code. It is read once, at attachment; a `contract()` that
    /// panics afterwards does not take the run down, whatever else the observer does.
    #[test]
    fn a_panicking_contract_after_attachment_is_never_called() {
        let mut crasher = Probe::new("crasher");
        crasher.panic_on_call = Some(3);
        let mut undeclared = Probe::new("undeclared");
        undeclared.undeclared_on_call = Some(2);
        let mut duplicate = Probe::new("duplicate");
        duplicate.duplicate_on_call = Some(2);
        let mut stray = Probe::new("stray");
        stray.extra_curve = Some("stray.other".to_string());
        let healthy = Probe::new("healthy");

        let armed: Vec<_> = [&crasher, &undeclared, &duplicate, &stray, &healthy]
            .into_iter()
            .map(|probe| std::sync::Arc::clone(&probe.contract_panics))
            .collect();
        let mut simulation = observed(vec![crasher, undeclared, duplicate, stray, healthy]);
        for flag in &armed {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }

        let outcome = run5(&mut simulation);
        assert!(outcome.is_success(), "{:?}", outcome.stop);
        let faults: Vec<(&str, ObserverFaultKind)> =
            outcome.observer_faults.iter().map(|f| (f.observer.as_str(), f.kind)).collect();
        // In the order they happened: call 2 (two observers), call 3, then the curves.
        assert_eq!(
            faults,
            [
                ("undeclared", ObserverFaultKind::UndeclaredRow),
                ("duplicate", ObserverFaultKind::DuplicateRow),
                ("crasher", ObserverFaultKind::Panic),
                ("stray", ObserverFaultKind::UndeclaredCurve),
            ]
        );
        assert_eq!(column(&outcome, "healthy.calls").len(), 6);
        let curves: Vec<&str> = outcome.artifact.curves().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(curves, ["healthy.curve"]);
    }

    /// The outcome carries the accumulated faults on every return path, as the
    /// simulation and the artifact warnings do.
    #[test]
    fn faults_are_reported_on_every_return_path() {
        let mut crasher = Probe::new("crasher");
        crasher.panic_on_call = Some(2);
        let mut simulation = observed(vec![crasher]);
        run5(&mut simulation);

        // Refused for having no stopping condition: nothing stepped, the fault stays.
        let refused = simulation.run(&RunConfig::new().with_timestep(0.1));
        assert_eq!(refused.stop, StopReason::NoStoppingCondition);
        assert_eq!(refused.observer_faults, simulation.observer_faults());
        assert_eq!(refused.observer_faults.len(), 1);
        assert!(refused.artifact.warnings().iter().any(|w| w.starts_with("error[E0951]:")));
    }

    /// A row in another unit than `outputs` declares is a breach like an undeclared row.
    #[test]
    fn a_row_in_the_wrong_unit_disables_the_observer() {
        let mut breaker = Probe::new("breaker");
        breaker.wrong_unit_on_call = Some(3);
        let mut simulation = observed(vec![breaker, Probe::new("fine")]);
        let outcome = run5(&mut simulation);

        assert!(outcome.is_success());
        assert_eq!(column(&outcome, "breaker.calls").len(), 2, "calls 1 and 2 only");
        assert_eq!(column(&outcome, "breaker.steps").len(), 2, "the bad sample is dropped");
        assert_eq!(column(&outcome, "fine.calls").len(), 6);

        assert_eq!(outcome.observer_faults.len(), 1);
        let fault = &outcome.observer_faults[0];
        assert_eq!((fault.observer.as_str(), fault.step), ("breaker", 2));
        assert_eq!(fault.kind, ObserverFaultKind::WrongUnit);
        let text = fault.describe();
        assert!(
            text.contains("`breaker.steps`") && text.contains("`m`") && text.contains("declares `1`"),
            "{text}"
        );
    }

    /// An observer reaches only the domains its contract lists in `reads`.
    #[test]
    fn an_observer_reaches_only_the_domains_it_declares() {
        /// Per sample: reached `a`, reached `b`, the names listed.
        type Seen = Vec<(bool, bool, Vec<String>)>;
        struct Reach {
            contract: lattice_ir::ObserverContract,
            seen: std::sync::Arc<std::sync::Mutex<Seen>>,
        }
        impl Observer for Reach {
            fn contract(&self) -> &lattice_ir::ObserverContract {
                &self.contract
            }
            fn observe(&mut self, ctx: &ObserveContext<'_>, _: &mut Observations) {
                self.seen.lock().unwrap().push((
                    ctx.domain::<Counter>("a").is_some(),
                    ctx.domain::<Counter>("b").is_some(),
                    ctx.domain_names().map(str::to_string).collect(),
                ));
            }
        }
        let mut contract = Probe::new("reach").contract;
        contract.outputs = vec![("reach.none".into(), "1".into())];
        let seen = std::sync::Arc::<std::sync::Mutex<Seen>>::default();
        let mut simulation =
            simulation(vec![Box::new(Counter::new("a")), Box::new(Counter::new("b"))]);
        simulation
            .add_observer(Box::new(Reach { contract, seen: std::sync::Arc::clone(&seen) }))
            .unwrap();
        run5(&mut simulation);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 6);
        // `b` exists and is a Counter, but the contract reads only `a`.
        assert!(seen.iter().all(|(a, b, names)| *a && !*b && names == &["a".to_string()]));
    }

    /// A non-finite observer row halts the run as a domain's does (spec G7).
    #[test]
    fn a_non_finite_observer_row_halts_the_run() {
        let mut poisoner = Probe::new("poisoner");
        poisoner.nan_on_call = Some(3);
        let mut simulation = observed(vec![poisoner]);
        let outcome = run5(&mut simulation);
        let StopReason::NonFinite { observation, step, .. } = &outcome.stop else {
            panic!("{:?}", outcome.stop)
        };
        assert_eq!((observation.as_str(), *step), ("poisoner.steps", 2));
        assert!(outcome.observer_faults.is_empty(), "not a contract breach");
    }

    /// Only the engine's solvers may publish an invariant (spec G2).
    #[test]
    fn an_invariant_row_disables_the_observer() {
        let mut claimant = Probe::new("claimant");
        claimant.invariant_on_call = Some(3);
        let mut simulation = observed(vec![claimant, Probe::new("fine")]);
        let outcome = run5(&mut simulation);

        assert!(outcome.is_success());
        assert_eq!(column(&outcome, "claimant.saw").len(), 2, "the bad sample is dropped");
        assert_eq!(column(&outcome, "fine.calls").len(), 6);
        assert_eq!(outcome.observer_faults.len(), 1);
        let fault = &outcome.observer_faults[0];
        assert_eq!((fault.observer.as_str(), fault.step), ("claimant", 2));
        assert_eq!(fault.kind, ObserverFaultKind::InvariantRow);
        assert!(fault.describe().contains("`claimant.saw`"), "{}", fault.describe());
    }

    /// A curve must be declared in `outputs`.
    #[test]
    fn an_undeclared_curve_disables_the_observer() {
        let mut stray = Probe::new("stray");
        stray.extra_curve = Some("stray.other".to_string());
        let mut simulation = observed(vec![stray, Probe::new("fine")]);
        let outcome = run5(&mut simulation);

        assert!(outcome.is_success());
        assert_eq!(outcome.observer_faults.len(), 1);
        let fault = &outcome.observer_faults[0];
        assert_eq!((fault.observer.as_str(), fault.kind), ("stray", ObserverFaultKind::UndeclaredCurve));
        assert!(fault.describe().contains("`stray.other`"), "{}", fault.describe());
        let curves: Vec<&str> = outcome.artifact.curves().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(curves, ["fine.curve"], "all of the offender's curves are dropped");
    }

    /// A declared curve whose name another curve already has is a clash: with one a
    /// domain returned, or with the observer's own.
    #[test]
    fn a_clashing_curve_disables_the_observer() {
        let mut domain = Counter::new("a");
        domain.curve = Some("a.curve".to_string());
        let mut simulation = Simulation::new(model_with("test"), vec![Box::new(domain)]);

        let mut shadow = Probe::new("shadow");
        shadow.contract.outputs.push(("a.curve".into(), "1".into()));
        shadow.extra_curve = Some("a.curve".to_string());
        let mut twice = Probe::new("twice");
        twice.extra_curve = Some("twice.curve".to_string());
        simulation.add_observer(Box::new(shadow)).unwrap();
        simulation.add_observer(Box::new(twice)).unwrap();
        let outcome = run5(&mut simulation);

        let kinds: Vec<(&str, ObserverFaultKind)> =
            outcome.observer_faults.iter().map(|f| (f.observer.as_str(), f.kind)).collect();
        assert_eq!(
            kinds,
            [("shadow", ObserverFaultKind::DuplicateCurve), ("twice", ObserverFaultKind::DuplicateCurve)]
        );
        let curves: Vec<&str> = outcome.artifact.curves().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(curves, ["a.curve"], "only the domain's curve survives");
    }

    /// `Simulation::curves` records a panic in an observer's `curves()` and disables it,
    /// as the end of a run does.
    #[test]
    fn curves_records_a_panicking_observer() {
        let mut probe = Probe::new("curvy");
        probe.panic_in_curves = true;
        let mut simulation = observed(vec![probe, Probe::new("fine")]);
        let curves: Vec<String> = simulation.curves().into_iter().map(|c| c.name).collect();
        assert_eq!(curves, ["fine.curve"]);
        assert_eq!(simulation.observer_faults().len(), 1);
        assert_eq!(simulation.observer_faults()[0].kind, ObserverFaultKind::Panic);
        // Disabled now: asked no more, and not recorded twice.
        let again: Vec<String> = simulation.curves().into_iter().map(|c| c.name).collect();
        assert_eq!(again, ["fine.curve"]);
        assert_eq!(simulation.observer_faults().len(), 1);
    }
}
