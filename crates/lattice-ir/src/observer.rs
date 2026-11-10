//! Observers: read-only measurements a run takes, written outside the engine.
//!
//! Spec §16.3 puts an *Observer tier* beside the solver tiers. An observer is not a
//! solver. It sees the domains through `&`, so it cannot change the physics, and it
//! keeps its own state between samples (a ring of past velocities, a running mean).
//! It publishes rows into the same [`Observations`] set the domains do, and may publish
//! [`Curve`]s at the end of the run.
//!
//! # The contract
//!
//! P8 asks every extension to state what it reads, what it writes, which backends it
//! runs on and whether it is differentiable; P1 asks it to say what it does *not*
//! measure. [`ObserverContract`] carries all of that, and [`ObserverContract::audit`]
//! refuses an empty one. That is stricter than [`SolverContract::audit`](crate::SolverContract::audit),
//! which a test suite runs: a plugin has no validation suite of ours to fail, so the
//! refusal happens when it is attached.
//!
//! # Reaching a domain's state
//!
//! [`ObserveContext::domain`] downcasts a domain to its concrete type through
//! [`Domain::as_any`](crate::contract::DomainType::as_any), and only for a domain the observer's
//! contract lists in `reads`; any other name gives `None`. That couples the observer to
//! the domain's internals. A read-only view over the domain's data is the better
//! surface, and the SDK is to provide one for particles (it does not exist yet).
//!
//! # What the runtime enforces
//!
//! The contract is checked against behaviour, not just filled in. The observer is
//! disabled and the run records `E0951` if it:
//!
//! - publishes a row not named in `outputs`, or in a unit other than the one declared;
//! - publishes the same row twice in one call, which would give the timeline two columns
//!   of one name;
//! - publishes a row of kind `Invariant`: only the engine's own solvers may claim that a
//!   quantity is conserved (spec M6.2 G2), so an observer publishes metrics;
//! - returns a curve not named in `outputs`, or one whose name another curve already
//!   has.
//!
//! A non-finite row halts the run with `StopReason::NonFinite` in the runtime, exactly as
//! a domain's row does (spec M6.2 G7). So an observer must not publish NaN as a
//! placeholder for "no value yet": leave the row out of that sample instead.
//!
//! # Panics
//!
//! The runtime wraps each call in `catch_unwind`, disables an observer that panics, and
//! carries on with the run (E0951). That isolates panics, and only panics: an infinite
//! loop or a stack overflow in an observer still hangs or kills the run, an abort is not
//! caught, and under `panic = "abort"` (which the workspace profiles do not set, and a
//! downstream build must not) the panic would end the run too.
//!
//! `&` access means the observer cannot corrupt the simulation through the `&` API, but
//! it does not make a panic harmless: a domain with interior mutability (`RefCell`,
//! `Mutex`) can be reached through [`ObserveContext::domain`], and a panic in the middle
//! of a call that touches it can leave that state inconsistent. Reading plain slices
//! avoids this.

use std::borrow::Cow;
use std::path::PathBuf;

use crate::contract::{Domain, DomainType};
use crate::curve::Curve;
use crate::diagnostics::Observations;
use crate::fault::LawFault;

/// Something that measures a run without changing it.
pub trait Observer: Send {
    /// What this observer reads, writes and does not measure.
    fn contract(&self) -> &ObserverContract;

    /// Called at each sample this observer is [due](Observer::due) for, after every
    /// domain has published. `ctx.time()` and `ctx.step()` say when.
    ///
    /// The observer is mutable on itself and read-only on the simulation: `&` access
    /// stops it changing a domain through the `&` API, but not through interior
    /// mutability in a domain reached via [`ObserveContext::domain`]. A panic part-way
    /// through such a call can leave that state inconsistent, and the runtime can only
    /// disable the observer, not repair it.
    ///
    /// Rows go into `out`, which is this observer's alone; the runtime appends it to the
    /// sample afterwards. The domains' rows are read through
    /// [`ObserveContext::observations`] and cannot be altered from here. Each row must be
    /// named in the contract's `outputs` with the unit declared there, and published at
    /// most once per call, or the observer is disabled (`E0951`).
    ///
    /// A non-finite row halts the run with `StopReason::NonFinite`, the same as a
    /// domain's, so never publish NaN as a placeholder: publish nothing for that sample.
    ///
    /// Every call is one sample. See [`Observer::due`].
    fn observe(&mut self, ctx: &ObserveContext<'_>, out: &mut Observations);

    /// Function-valued results, as [`Domain::curves`]. Taken once at the end of a run.
    ///
    /// Each curve's name must be in the contract's `outputs` and must not repeat another
    /// curve's, or the observer is disabled and its curves from that call are dropped
    /// (`E0951`).
    fn curves(&self) -> Vec<Curve> {
        Vec::new()
    }

    /// Whether the sample instant `time` is one this observer wants, given the step
    /// `dt` the run is taking. Default: every sample.
    ///
    /// Asked once per sample, before [`Observer::observe`], and an implementation may
    /// advance its schedule when it answers yes. So the runtime treats each sample as
    /// happening once: `Simulation::observe` must not be called twice for one instant.
    fn due(&mut self, _time: f64, _dt: f64) -> bool {
        true
    }

    /// Explain why the row `row` is non-finite at the state `ctx` shows (diagnostic
    /// `E0950`, spec M6.2 G7). Default: no explanation.
    ///
    /// Called after a run has halted on that row, with the domains still in the state
    /// that was sampled. That state is the observer's own input, so it can trace its
    /// computation against it without replaying the run. An observer that computes
    /// from a program names the first particle, in index order, whose value is not
    /// finite and the operation that made it so.
    fn explain_non_finite(&self, _ctx: &ObserveContext<'_>, _row: &str) -> Option<LawFault> {
        None
    }
}

/// What an observer does besides compute.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum SideEffects {
    /// Nothing outside its own memory.
    #[default]
    None,
    /// It writes these files.
    WritesFiles(Vec<PathBuf>),
    /// Something else, described in words.
    Other(Cow<'static, str>),
}

/// The backends an observer can run on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Backends {
    /// Runs on the CPU.
    pub cpu: bool,
    /// Runs on the GPU.
    pub gpu: bool,
}

impl Backends {
    /// CPU only, which is all an observer can be today.
    pub const CPU: Backends = Backends { cpu: true, gpu: false };

    /// True when at least one backend is named.
    pub fn any(self) -> bool {
        self.cpu || self.gpu
    }
}

/// Everything an observer must say about itself (P8, and P1's honest limitations).
#[derive(Clone, PartialEq, Debug)]
pub struct ObserverContract {
    /// Short identifier, e.g. `"vacf"`.
    pub name: Cow<'static, str>,
    /// One sentence on what it measures.
    pub summary: Cow<'static, str>,
    /// Domain instances it reads, by the name the model gave them.
    pub reads: Vec<Cow<'static, str>>,
    /// The rows and curves it publishes, each with its SI unit (`"1"` when
    /// dimensionless).
    pub outputs: Vec<(Cow<'static, str>, Cow<'static, str>)>,
    /// What it does besides compute.
    pub side_effects: SideEffects,
    /// What it does not measure (P1): sampling, truncation, bias.
    pub limitations: Vec<Cow<'static, str>>,
    /// Whether repeated runs give identical output.
    pub deterministic: bool,
    /// The backends it runs on.
    pub backends: Backends,
    /// Whether gradients can be taken through it.
    pub differentiable: bool,
}

impl ObserverContract {
    /// Check that a contract is filled in, and that what it reads exists.
    ///
    /// `domains` names every domain instance in the model. An empty list is a gap,
    /// not a pass: the contract must say something about what it measures.
    pub fn audit(&self, domains: &[&str]) -> Vec<ObserverGap> {
        let mut gaps = Vec::new();
        if self.name.is_empty() {
            gaps.push(ObserverGap::NoName);
        }
        if self.summary.is_empty() {
            gaps.push(ObserverGap::NoSummary);
        }
        if self.reads.is_empty() {
            gaps.push(ObserverGap::NoReads);
        }
        for read in &self.reads {
            if !domains.contains(&read.as_ref()) {
                gaps.push(ObserverGap::UnknownRead(read.to_string()));
            }
        }
        if self.outputs.is_empty() {
            gaps.push(ObserverGap::NoOutputs);
        }
        if self.limitations.is_empty() {
            gaps.push(ObserverGap::NoLimitations);
        }
        if !self.backends.any() {
            gaps.push(ObserverGap::NoBackends);
        }
        gaps
    }
}

/// A missing or wrong piece of an observer contract.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ObserverGap {
    /// No name.
    NoName,
    /// No summary.
    NoSummary,
    /// Reads nothing, which is not a measurement.
    NoReads,
    /// Reads a domain the model does not have.
    UnknownRead(String),
    /// Publishes no rows or curves.
    NoOutputs,
    /// States no limitations (P1).
    NoLimitations,
    /// Runs on no backend.
    NoBackends,
    /// An `outputs` name another observer, or this one, already declares.
    OutputTaken(String),
    /// An `outputs` name a domain already publishes.
    OutputShadowsDomain(String),
}

impl core::fmt::Display for ObserverGap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ObserverGap::NoName => f.write_str("the contract has no name"),
            ObserverGap::NoSummary => f.write_str("the contract has no summary"),
            ObserverGap::NoReads => f.write_str("the contract names no domain it reads"),
            ObserverGap::UnknownRead(name) => {
                write!(f, "the contract reads `{name}`, which the model does not have")
            }
            ObserverGap::NoOutputs => f.write_str("the contract lists no outputs"),
            ObserverGap::NoLimitations => {
                f.write_str("the contract states no limitations: say what it does not measure")
            }
            ObserverGap::NoBackends => f.write_str("the contract names no backend"),
            ObserverGap::OutputTaken(name) => {
                write!(f, "the output `{name}` is already declared by an observer")
            }
            ObserverGap::OutputShadowsDomain(name) => {
                write!(f, "the output `{name}` is a row a domain already publishes")
            }
        }
    }
}

/// What an observer is shown at a sample: the clock, the domains, the rows so far.
#[derive(Clone, Copy)]
pub struct ObserveContext<'a> {
    time: f64,
    step: u64,
    domains: &'a [Box<dyn Domain>],
    observations: &'a Observations,
    /// The domains the observer may reach; `None` is all of them.
    reads: Option<&'a [Cow<'static, str>]>,
}

impl core::fmt::Debug for ObserveContext<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ObserveContext")
            .field("time", &self.time)
            .field("step", &self.step)
            .field("domains", &self.domains.len())
            .field("observations", &self.observations.len())
            .finish()
    }
}

impl<'a> ObserveContext<'a> {
    /// A context at `time` and `step` over `domains`, with `observations` published.
    ///
    /// Every domain is reachable until [`ObserveContext::restricted_to`] narrows it.
    pub fn new(
        time: f64,
        step: u64,
        domains: &'a [Box<dyn Domain>],
        observations: &'a Observations,
    ) -> Self {
        Self { time, step, domains, observations, reads: None }
    }

    /// The same context, reaching only the domains named in `reads`: an observer's
    /// contract is what it may read.
    pub fn restricted_to(mut self, reads: &'a [Cow<'static, str>]) -> Self {
        self.reads = Some(reads);
        self
    }

    fn allows(&self, name: &str) -> bool {
        self.reads.is_none_or(|reads| reads.iter().any(|read| read == name))
    }

    /// Simulation time of this sample, seconds.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Step counter of this sample.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Every row published so far this sample: the domains', and those of observers
    /// attached before this one.
    ///
    /// This shows all of them whatever `reads` says. Published rows are the shared
    /// channel between solvers and observers; `reads` governs access to domain *state*
    /// through [`ObserveContext::domain`].
    pub fn observations(&self) -> &Observations {
        self.observations
    }

    /// The domain named `name`, as its concrete type, if it is a `T` and the observer's
    /// contract lists it in `reads`. Any other name gives `None`.
    pub fn domain<T: 'static>(&self, name: &str) -> Option<&T> {
        if !self.allows(name) {
            return None;
        }
        // `&**` matters: `Box<dyn Domain>` is itself `Any`, so calling `as_any` on the
        // box would downcast the box rather than the domain inside it.
        let domain: &dyn Domain = &**self.domains.iter().find(|d| d.name() == name)?;
        DomainType::as_any(domain).downcast_ref::<T>()
    }

    /// The names of the domain instances the observer may read, in model order.
    pub fn domain_names(&self) -> impl Iterator<Item = &str> {
        self.domains.iter().map(|d| d.name()).filter(|name| self.allows(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete() -> ObserverContract {
        ObserverContract {
            name: "test.observer".into(),
            summary: "an observer used only by these tests".into(),
            reads: vec!["atoms".into()],
            outputs: vec![("test.value".into(), "1".into())],
            side_effects: SideEffects::None,
            limitations: vec!["measures nothing".into()],
            deterministic: true,
            backends: Backends::CPU,
            differentiable: false,
        }
    }

    #[test]
    fn a_complete_contract_has_no_gaps() {
        assert!(complete().audit(&["atoms", "heat"]).is_empty());
    }

    #[test]
    fn audit_refuses_empty_limitations() {
        let contract = ObserverContract { limitations: Vec::new(), ..complete() };
        assert_eq!(contract.audit(&["atoms"]), vec![ObserverGap::NoLimitations]);
    }

    #[test]
    fn audit_refuses_a_read_the_model_does_not_have() {
        let contract = ObserverContract { reads: vec!["atmos".into()], ..complete() };
        assert_eq!(
            contract.audit(&["atoms"]),
            vec![ObserverGap::UnknownRead("atmos".to_string())]
        );
    }

    #[test]
    fn audit_refuses_empty_outputs() {
        let contract = ObserverContract { outputs: Vec::new(), ..complete() };
        assert_eq!(contract.audit(&["atoms"]), vec![ObserverGap::NoOutputs]);
    }

    #[test]
    fn audit_finds_every_missing_section() {
        let empty = ObserverContract {
            name: "".into(),
            summary: "".into(),
            reads: Vec::new(),
            outputs: Vec::new(),
            limitations: Vec::new(),
            backends: Backends { cpu: false, gpu: false },
            ..complete()
        };
        let gaps = empty.audit(&["atoms"]);
        assert_eq!(
            gaps,
            vec![
                ObserverGap::NoName,
                ObserverGap::NoSummary,
                ObserverGap::NoReads,
                ObserverGap::NoOutputs,
                ObserverGap::NoLimitations,
                ObserverGap::NoBackends,
            ]
        );
    }

    #[test]
    fn a_gap_says_what_to_fix() {
        let text = ObserverGap::UnknownRead("atmos".into()).to_string();
        assert!(text.contains("atmos") && text.contains("does not have"), "{text}");
    }
}
