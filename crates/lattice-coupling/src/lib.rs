//! Coupling: moving quantities between domains, and accounting for what moved.
//!
//! Spec §14. A **coupling edge** connects a port one domain publishes to a port another
//! consumes, applies a mapping, and — when what crosses is a conserved quantity —
//! leaves an entry in the [ledger].
//!
//! # The mapping is the point
//!
//! §14.1 lists a mapping as part of every edge, and it is easy to read that as
//! interpolation between differently shaped grids. The more important case is
//! **units**. A reaction publishes heat in `W/m²`; a heat solver consumes a source in
//! `K/s`. Those are not the same quantity, and the factor between them is an areal heat
//! capacity `ρ·c·h` that belongs to neither domain — it is a property of the material
//! sitting between them.
//!
//! Wiring the two straight together produces a run that compiles, executes, looks
//! entirely plausible, and is wrong by that factor. So a [`Mapping`] carries it, states
//! the two units it converts between, and the compiler checks them against the ports.
//!
//! # What the ledger is for
//!
//! §14.3: *"This makes it possible to debug whether a coupled result is physically
//! inconsistent because of the model, a mapping error, a timestep issue, or an
//! intentionally open system."*
//!
//! A coupled run's energy drift is otherwise a single number with no way to tell those
//! four apart. Every edge that moves a conserved quantity records how much it claims to
//! have moved; the target's own observation says how much arrived. When those disagree,
//! the discrepancy is a *number attached to a named edge* rather than a mystery.
//!
//! # Strategies
//!
//! §14.2 lists five. Two are implemented:
//!
//! - **One-way**: the source affects the target and the target does not feed back
//!   within the step.
//! - **Loose staggered**: domains advance in sequence, each using the latest state,
//!   with exchanges between them. This is what a bidirectional pair of edges gives.
//!
//! Subcycling, fixed-point iteration and monolithic solves are not here. A fixed-point
//! coupling would need every domain to be re-steppable from a saved state, which is a
//! checkpoint mechanism this runtime does not yet have — and a half-implemented
//! fixed-point iteration that silently does one pass is worse than an honest staggered
//! one, because it claims a convergence it never checked.
//!
//! [ledger]: lattice_ir::ConservationLedger

use lattice_ir::{ConservationLedger, Domain, Grid2d, Invariant, PortData, PortShape, Reconciliation};

/// Which domain and which of its ports.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PortRef {
    /// Index into the runtime's domain list.
    pub domain: usize,
    /// The port's name, without the domain prefix.
    pub port: String,
}

impl PortRef {
    /// A reference to `port` on domain `domain`.
    pub fn new(domain: usize, port: impl Into<String>) -> PortRef {
        PortRef { domain, port: port.into() }
    }
}

/// How a value is transformed on its way across an edge.
#[derive(Clone, PartialEq, Debug)]
pub enum Mapping {
    /// Pass it through unchanged.
    ///
    /// Valid only when the two ports already share a unit, which the compiler checks.
    Direct,
    /// Multiply by a constant, converting between the two ports' units.
    Scale {
        /// The factor.
        factor: f64,
        /// What the factor means, for the model report — e.g. `"1 / (rho c h)"`.
        reason: String,
    },
}

impl Mapping {
    /// A conversion by `factor`, with a one-line explanation of where it came from.
    ///
    /// The explanation is not decoration. A bare number in a coupling edge is the
    /// single most opaque thing in a coupled model: it is the one place where two
    /// domains' units meet, and six months later nobody remembers whether `0.005` was
    /// a heat capacity or a thickness.
    pub fn scale(factor: f64, reason: impl Into<String>) -> Mapping {
        Mapping::Scale { factor, reason: reason.into() }
    }

    /// The factor this mapping applies.
    pub fn factor(&self) -> f64 {
        match self {
            Mapping::Direct => 1.0,
            Mapping::Scale { factor, .. } => *factor,
        }
    }

    /// A one-line description for the model report.
    pub fn describe(&self) -> String {
        match self {
            Mapping::Direct => "unchanged".to_string(),
            Mapping::Scale { factor, reason } => format!("x {factor:.6e}  ({reason})"),
        }
    }
}

/// One coupling edge.
#[derive(Clone, PartialEq, Debug)]
pub struct CouplingEdge {
    /// The name the model gave it, for diagnostics and the ledger.
    pub name: String,
    /// Where the value comes from.
    pub source: PortRef,
    /// Where it goes.
    pub target: PortRef,
    /// How it is transformed on the way.
    pub mapping: Mapping,
    /// Which conserved quantity crosses, if any.
    ///
    /// `None` for an edge that carries a *parameter* rather than a transfer. A
    /// temperature reaching a rate constant changes how fast a reaction goes; it does
    /// not move energy, and recording it in the ledger would put a number in the books
    /// that has nothing to balance against.
    pub quantity: Option<Invariant>,
    /// Exchange every this many steps. One means every step.
    ///
    /// §14.1's *cadence*. A slow domain reading a fast one does not need every value,
    /// and a coupling that runs less often is cheaper — but the value it holds between
    /// exchanges is stale, and this is where that decision is visible.
    pub cadence: usize,
}

impl CouplingEdge {
    /// An edge from one port to another, exchanged every step.
    pub fn new(name: impl Into<String>, source: PortRef, target: PortRef) -> CouplingEdge {
        CouplingEdge {
            name: name.into(),
            source,
            target,
            mapping: Mapping::Direct,
            quantity: None,
            cadence: 1,
        }
    }

    /// Set the mapping.
    pub fn with_mapping(mut self, mapping: Mapping) -> CouplingEdge {
        self.mapping = mapping;
        self
    }

    /// Declare which conserved quantity crosses, so the ledger records it.
    pub fn carrying(mut self, quantity: Invariant) -> CouplingEdge {
        self.quantity = Some(quantity);
        self
    }

    /// Exchange every `cadence` steps.
    pub fn every(mut self, cadence: usize) -> CouplingEdge {
        self.cadence = cadence.max(1);
        self
    }

    /// True when this edge should run on `step`.
    pub fn runs_on(&self, step: u64) -> bool {
        step % self.cadence as u64 == 0
    }

    /// A one-line description for the model report.
    pub fn describe(&self) -> String {
        let mut text = format!(
            "{}: domain {}.{} -> domain {}.{}, {}",
            self.name,
            self.source.domain,
            self.source.port,
            self.target.domain,
            self.target.port,
            self.mapping.describe()
        );
        if let Some(quantity) = self.quantity {
            text.push_str(&format!(", carrying {quantity}"));
        }
        if self.cadence > 1 {
            text.push_str(&format!(", every {} steps", self.cadence));
        }
        text
    }
}

/// Why an exchange did not happen.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EdgeFault {
    /// The source domain index is past the end of the domain list.
    NoSuchSourceDomain,
    /// The target domain index is past the end.
    NoSuchTargetDomain,
    /// The source domain does not publish that port.
    NotPublished,
    /// The target domain does not consume that port.
    NotConsumed,
    /// The two ports disagree about shape — a field cannot be written to a scalar.
    ShapeMismatch,
    /// The value that crossed contained a non-finite number.
    ///
    /// Reported rather than transferred. A NaN travelling along a coupling edge turns
    /// one sick domain into two, and the second one has no way to tell where it came
    /// from (NFR-007).
    NonFinite,
}

impl core::fmt::Display for EdgeFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            EdgeFault::NoSuchSourceDomain => "the source domain does not exist",
            EdgeFault::NoSuchTargetDomain => "the target domain does not exist",
            EdgeFault::NotPublished => "the source domain does not publish that port",
            EdgeFault::NotConsumed => "the target domain does not consume that port",
            EdgeFault::ShapeMismatch => "the two ports have different shapes",
            EdgeFault::NonFinite => "the value crossing the edge was not finite",
        })
    }
}

/// What one exchange achieved.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ExchangeReport {
    /// Edges that ran this step.
    pub exchanged: usize,
    /// Edges that were skipped because their cadence did not fall on this step.
    pub skipped: usize,
    /// Edges that failed, with the reason.
    pub faults: Vec<(String, EdgeFault)>,
}

impl ExchangeReport {
    /// True when every edge that should have run did.
    pub fn is_clean(&self) -> bool {
        self.faults.is_empty()
    }
}

/// Runs the coupling edges and keeps the books.
#[derive(Debug, Default)]
pub struct Coupler {
    edges: Vec<CouplingEdge>,
    ledger: ConservationLedger,
    /// One staging buffer per edge, reused across steps so an exchange allocates
    /// nothing (NFR-001).
    staging: Vec<Option<PortData>>,
}

impl Coupler {
    /// A coupler with no edges.
    pub fn new() -> Coupler {
        Coupler::default()
    }

    /// Add an edge.
    pub fn add(&mut self, edge: CouplingEdge) {
        self.edges.push(edge);
        self.staging.push(None);
    }

    /// The edges, in the order they run.
    ///
    /// Order matters and is the model's to choose: a staggered coupling in which A
    /// feeds B and B feeds A gives a different answer depending on which goes first,
    /// by one step's worth of staleness. Sorting them would hide that.
    pub fn edges(&self) -> &[CouplingEdge] {
        &self.edges
    }

    /// The ledger.
    pub fn ledger(&self) -> &ConservationLedger {
        &self.ledger
    }

    /// Forget every recorded transfer, keeping the edges.
    pub fn clear_ledger(&mut self) {
        self.ledger.clear();
    }

    /// Run every edge whose cadence falls on `step`.
    ///
    /// `dt` is the step just taken, used to turn a per-second rate into the total the
    /// ledger records.
    pub fn exchange(
        &mut self,
        domains: &mut [Box<dyn Domain>],
        step: u64,
        time: f64,
        dt: f64,
    ) -> ExchangeReport {
        let mut report = ExchangeReport::default();

        for index in 0..self.edges.len() {
            if !self.edges[index].runs_on(step) {
                report.skipped += 1;
                continue;
            }
            match self.run_edge(index, domains, time, dt) {
                Ok(()) => report.exchanged += 1,
                Err(fault) => report.faults.push((self.edges[index].name.clone(), fault)),
            }
        }
        report
    }

    fn run_edge(
        &mut self,
        index: usize,
        domains: &mut [Box<dyn Domain>],
        time: f64,
        dt: f64,
    ) -> Result<(), EdgeFault> {
        let edge = &self.edges[index];
        let (from, to) = (edge.source.domain, edge.target.domain);
        if from >= domains.len() {
            return Err(EdgeFault::NoSuchSourceDomain);
        }
        if to >= domains.len() {
            return Err(EdgeFault::NoSuchTargetDomain);
        }

        // Size the staging buffer from the *source* port's shape, once.
        if self.staging[index].is_none() {
            let shape = domains[from]
                .ports()
                .iter()
                .find(|spec| spec.name == edge.source.port)
                .map(|spec| spec.shape)
                .ok_or(EdgeFault::NotPublished)?;
            self.staging[index] = Some(match shape {
                PortShape::Scalar => PortData::scalar(),
                PortShape::Field => {
                    let grid = domains[from].port_grid().ok_or(EdgeFault::ShapeMismatch)?;
                    PortData::field(&grid)
                }
            });
        }
        let buffer = self.staging[index].as_mut().expect("just filled");

        buffer.clear();
        if !domains[from].read_port(&edge.source.port, buffer) {
            return Err(EdgeFault::NotPublished);
        }
        // A NaN travelling along an edge turns one sick domain into two, and the second
        // has no way to tell where it came from. Stopping it here means the fault is
        // reported against the edge that carried it.
        if buffer.has_non_finite() {
            return Err(EdgeFault::NonFinite);
        }

        // The ledger records what left the SOURCE, before the mapping.
        //
        // Which side to measure is not arbitrary. After the mapping the value is in the
        // target's units — for a heat edge, kelvin per second — and integrating that
        // gives a number in K·m² that is off from the energy by exactly the heat
        // capacity the mapping applied. The invariant the edge declares is a physical
        // quantity, and the side that is already in its units is the source.
        //
        // The unit check that makes this sound belongs to the compiler: a source port
        // wired to an edge `carrying(Energy)` has to publish W/m², or the number in the
        // books is a number in the wrong units.
        let total = match edge.quantity {
            None => 0.0,
            Some(_) => {
                let grid = domains[from].port_grid().unwrap_or_else(|| Grid2d::new(1, 1, [1.0, 1.0]));
                buffer.total(&grid) * dt
            }
        };

        buffer.scale(edge.mapping.factor());

        if !domains[to].write_port(&edge.target.port, buffer) {
            return Err(EdgeFault::NotConsumed);
        }

        if let Some(quantity) = edge.quantity {
            let (source_name, target_name) =
                (domains[from].name().to_string(), domains[to].name().to_string());
            self.ledger.record(quantity, source_name, target_name, total, time);
        }
        Ok(())
    }

    /// Check a domain's observed change against what the ledger says it received.
    ///
    /// The §14.3 question in one call.
    pub fn reconcile(
        &self,
        quantity: Invariant,
        domain: &str,
        observed_change: f64,
    ) -> Reconciliation {
        self.ledger.reconcile(quantity, domain, observed_change)
    }

    /// A multi-line description of every edge, for the model report.
    pub fn describe(&self) -> String {
        if self.edges.is_empty() {
            return "no coupling edges".to_string();
        }
        self.edges.iter().map(CouplingEdge::describe).collect::<Vec<_>>().join("\n")
    }
}
