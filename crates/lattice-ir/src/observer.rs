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
//! [`Domain::as_any`](crate::DomainType::as_any). That couples the observer to the
//! domain's internals, so a plugin should go through a read-only view where one exists
//! (the SDK's `ParticleView`) rather than the store itself.
//!
//! # Panics
//!
//! The runtime wraps each call in `catch_unwind`. Because the observer has only `&`
//! access to the simulation, a panic cannot have corrupted it, and only the observer's
//! own state is suspect, so the observer is disabled and the run continues (NFR-004).
//! This holds for panics, not aborts: it needs `panic = "unwind"` in the profile.

use std::borrow::Cow;
use std::path::PathBuf;

use crate::contract::{Domain, DomainType};
use crate::curve::Curve;
use crate::diagnostics::Observations;

/// Something that measures a run without changing it.
pub trait Observer: Send {
    /// What this observer reads, writes and does not measure.
    fn contract(&self) -> &ObserverContract;

    /// Called at each sample this observer is [due](Observer::due) for, after every
    /// domain has published. `ctx.time()` and `ctx.step()` say when.
    ///
    /// Read-only on the simulation, mutable on itself. Rows go into `out`, which is
    /// this observer's alone: the runtime appends it to the sample afterwards, so the
    /// domains' rows are read through [`ObserveContext::observations`] and cannot be
    /// altered from here.
    fn observe(&mut self, ctx: &ObserveContext<'_>, out: &mut Observations);

    /// Function-valued results, as [`Domain::curves`]. Taken once at the end of a run.
    fn curves(&self) -> Vec<Curve> {
        Vec::new()
    }

    /// Whether the sample instant `time` is one this observer wants, given the step
    /// `dt` the run is taking. Default: every sample.
    fn due(&mut self, _time: f64, _dt: f64) -> bool {
        true
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
    pub fn new(
        time: f64,
        step: u64,
        domains: &'a [Box<dyn Domain>],
        observations: &'a Observations,
    ) -> Self {
        Self { time, step, domains, observations }
    }

    /// Simulation time of this sample, seconds.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Step counter of this sample.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The rows the domains just published, and those of observers attached before
    /// this one.
    pub fn observations(&self) -> &Observations {
        self.observations
    }

    /// The domain named `name`, as its concrete type, if it is a `T`.
    pub fn domain<T: 'static>(&self, name: &str) -> Option<&T> {
        // `&**` matters: `Box<dyn Domain>` is itself `Any`, so calling `as_any` on the
        // box would downcast the box rather than the domain inside it.
        let domain: &dyn Domain = &**self.domains.iter().find(|d| d.name() == name)?;
        DomainType::as_any(domain).downcast_ref::<T>()
    }

    /// The names of the domain instances, in model order.
    pub fn domain_names(&self) -> impl Iterator<Item = &str> {
        self.domains.iter().map(|d| d.name())
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
