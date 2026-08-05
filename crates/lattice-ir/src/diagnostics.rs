//! Conservation monitoring, coupling ledgers, and solver residuals.
//!
//! Spec P7: *"Visualization is instrumentation... it must display fields, fluxes,
//! constraints, forces, residuals, conservation drift, timestep decisions, and
//! uncertainty."* Everything the viewer or a run artifact shows has to come from
//! somewhere; this module is that somewhere.
//!
//! Three related but distinct things live here:
//!
//! - [`Observations`] — the per-step readings a domain publishes.
//! - [`DriftMonitor`] — how far a supposedly conserved quantity has actually moved.
//! - [`ConservationLedger`] — §14.3's record of what coupling *intended* to transfer,
//!   so that observed drift can be attributed to the model, the mapping, or the
//!   numerics rather than left as an unexplained number.
//!
//! The ledger is the interesting one. Energy leaving a reaction and arriving in a
//! heat field is not drift; it is a transfer, and it should net to zero. Only what
//! *fails* to net out is a numerical problem.

use std::borrow::Cow;
use std::collections::BTreeMap;

use core::fmt;

use lattice_units::Dimension;

/// A physical quantity that a solver may conserve.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Invariant {
    /// Total energy, J.
    Energy,
    /// Kinetic energy, J.
    KineticEnergy,
    /// Potential energy, J.
    PotentialEnergy,
    /// Linear momentum, x component, kg·m/s.
    MomentumX,
    /// Linear momentum, y component, kg·m/s.
    MomentumY,
    /// Angular momentum about the origin, kg·m²/s.
    AngularMomentum,
    /// Total mass, kg.
    Mass,
    /// Total amount of substance, mol.
    Amount,
    /// Total electric charge, C.
    Charge,
    /// Integrated probability density, dimensionless.
    ProbabilityNorm,
    /// Integrated value of a scalar field, in that field's units × m².
    ///
    /// Used for the heat equation, where the conserved quantity on a closed domain is
    /// `∫T dA` (proportional to total thermal energy at constant ρc).
    FieldIntegral,
}

impl Invariant {
    /// Name for reports.
    pub const fn name(self) -> &'static str {
        match self {
            Invariant::Energy => "total energy",
            Invariant::KineticEnergy => "kinetic energy",
            Invariant::PotentialEnergy => "potential energy",
            Invariant::MomentumX => "momentum x",
            Invariant::MomentumY => "momentum y",
            Invariant::AngularMomentum => "angular momentum",
            Invariant::Mass => "mass",
            Invariant::Amount => "amount of substance",
            Invariant::Charge => "charge",
            Invariant::ProbabilityNorm => "probability norm",
            Invariant::FieldIntegral => "field integral",
        }
    }

    /// The physical dimension of this quantity.
    ///
    /// `None` for [`Invariant::FieldIntegral`], whose dimension depends on the field
    /// being integrated and is therefore only known to the solver that owns it.
    /// Returning `None` rather than guessing keeps the dimensional check honest.
    pub const fn dimension(self) -> Option<Dimension> {
        match self {
            Invariant::Energy | Invariant::KineticEnergy | Invariant::PotentialEnergy => {
                Some(Dimension::ENERGY)
            }
            Invariant::MomentumX | Invariant::MomentumY => Some(Dimension::MOMENTUM),
            // Angular momentum shares its dimension with action, kg·m^2/s.
            Invariant::AngularMomentum => Some(Dimension::ACTION),
            Invariant::Mass => Some(Dimension::MASS),
            Invariant::Amount => Some(Dimension::AMOUNT),
            Invariant::Charge => Some(Dimension::CHARGE),
            Invariant::ProbabilityNorm => Some(Dimension::DIMENSIONLESS),
            Invariant::FieldIntegral => None,
        }
    }

    /// Coherent SI unit, for display.
    pub const fn si_unit(self) -> &'static str {
        match self {
            Invariant::Energy | Invariant::KineticEnergy | Invariant::PotentialEnergy => "J",
            Invariant::MomentumX | Invariant::MomentumY => "kg·m/s",
            Invariant::AngularMomentum => "kg·m^2/s",
            Invariant::Mass => "kg",
            Invariant::Amount => "mol",
            Invariant::Charge => "C",
            Invariant::ProbabilityNorm => "1",
            Invariant::FieldIntegral => "(field unit)·m^2",
        }
    }
}

impl fmt::Display for Invariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What kind of reading an [`Observation`] is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObservationKind {
    /// A quantity that should be conserved.
    Invariant(Invariant),
    /// A linear- or nonlinear-solver residual.
    Residual,
    /// A derived measurement (temperature, mean speed, …).
    Metric,
    /// Current timestep as a fraction of the stability limit.
    StabilityMargin,
    /// A count (particles, iterations, rejected steps).
    Count,
}

/// One reading published by a domain.
#[derive(Clone, PartialEq, Debug)]
pub struct Observation {
    /// Dotted name, e.g. `"heat.field_integral"`.
    pub name: Cow<'static, str>,
    /// Value in coherent SI.
    pub value: f64,
    /// SI unit string, for display.
    pub unit: &'static str,
    /// What sort of reading this is.
    pub kind: ObservationKind,
}

/// The set of readings collected during one step.
///
/// Reused across steps: [`Observations::clear`] keeps the allocation, so publishing
/// diagnostics does not allocate in steady state (NFR-001).
#[derive(Clone, Default, Debug)]
pub struct Observations {
    entries: Vec<Observation>,
}

impl Observations {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a reading.
    pub fn record(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        value: f64,
        unit: &'static str,
        kind: ObservationKind,
    ) {
        self.entries.push(Observation { name: name.into(), value, unit, kind });
    }

    /// Record a conserved quantity, taking the unit from the invariant.
    pub fn record_invariant(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        invariant: Invariant,
        value: f64,
    ) {
        self.record(name, value, invariant.si_unit(), ObservationKind::Invariant(invariant));
    }

    /// Record a plain derived metric.
    pub fn record_metric(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        value: f64,
        unit: &'static str,
    ) {
        self.record(name, value, unit, ObservationKind::Metric);
    }

    /// Look up a reading by exact name.
    pub fn get(&self, name: &str) -> Option<&Observation> {
        self.entries.iter().find(|o| o.name == name)
    }

    /// The value of a reading, if present.
    pub fn value(&self, name: &str) -> Option<f64> {
        self.get(name).map(|o| o.value)
    }

    /// All readings.
    pub fn iter(&self) -> impl Iterator<Item = &Observation> {
        self.entries.iter()
    }

    /// Number of readings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop all readings, keeping capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Any reading whose value is not finite.
    ///
    /// NFR-007: the engine must expose instability rather than hide it. The runtime
    /// checks this every step and halts with the offending name attached.
    pub fn first_non_finite(&self) -> Option<&Observation> {
        self.entries.iter().find(|o| !o.value.is_finite())
    }
}

/// Tracks how far a conserved quantity has moved from its reference value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DriftMonitor {
    invariant: Invariant,
    reference: f64,
    latest: f64,
    max_abs_drift: f64,
    samples: u64,
}

impl DriftMonitor {
    /// Start monitoring, taking `reference` as the value that should be preserved.
    pub fn new(invariant: Invariant, reference: f64) -> Self {
        Self { invariant, reference, latest: reference, max_abs_drift: 0.0, samples: 1 }
    }

    /// Which quantity this tracks.
    pub fn invariant(&self) -> Invariant {
        self.invariant
    }

    /// The value drift is measured against.
    pub fn reference(&self) -> f64 {
        self.reference
    }

    /// The most recent sample.
    pub fn latest(&self) -> f64 {
        self.latest
    }

    /// Number of samples taken.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Add a sample.
    pub fn record(&mut self, value: f64) {
        self.latest = value;
        self.samples += 1;
        let drift = (value - self.reference).abs();
        if drift > self.max_abs_drift {
            self.max_abs_drift = drift;
        }
    }

    /// Signed difference from the reference.
    pub fn absolute_drift(&self) -> f64 {
        self.latest - self.reference
    }

    /// Drift relative to the reference magnitude.
    ///
    /// Returns the absolute drift when the reference is zero, since a relative
    /// measure is undefined there and returning infinity would be useless in a plot.
    pub fn relative_drift(&self) -> f64 {
        if self.reference == 0.0 {
            self.absolute_drift()
        } else {
            self.absolute_drift() / self.reference.abs()
        }
    }

    /// Largest absolute drift seen at any sample, not just the latest.
    ///
    /// A symplectic integrator's energy oscillates: sampling only the endpoint can
    /// report near-zero drift while the trajectory swung much further. This is the
    /// number to quote in a validation report.
    pub fn max_absolute_drift(&self) -> f64 {
        self.max_abs_drift
    }

    /// Largest relative drift seen at any sample.
    pub fn max_relative_drift(&self) -> f64 {
        if self.reference == 0.0 {
            self.max_abs_drift
        } else {
            self.max_abs_drift / self.reference.abs()
        }
    }

    /// True if drift has stayed within `tolerance` (relative) at every sample.
    pub fn within(&self, tolerance: f64) -> bool {
        self.max_relative_drift() <= tolerance
    }
}

/// One accounted transfer of a conserved quantity between domains.
#[derive(Clone, PartialEq, Debug)]
pub struct Transfer {
    /// What moved.
    pub quantity: Invariant,
    /// Where it came from, e.g. `"reaction.A_plus_B"`.
    pub from: Cow<'static, str>,
    /// Where it went, e.g. `"heat.source"`.
    pub to: Cow<'static, str>,
    /// How much, in the invariant's SI unit. Always non-negative; direction is
    /// carried by `from`/`to`.
    pub amount: f64,
    /// Simulation time of the transfer, seconds.
    pub time: f64,
}

/// The coupling ledger of spec §14.3.
///
/// Records what each coupling edge claims to have moved, so that a change in a
/// domain's invariant can be checked against the transfers that explain it. Without
/// this, a coupled run's energy drift is a single number with no way to tell a
/// mapping bug from a timestep problem from an intentionally open system.
#[derive(Clone, Default, Debug)]
pub struct ConservationLedger {
    transfers: Vec<Transfer>,
    /// Net amount moved *into* each named sink, keyed by (quantity, sink).
    net_in: BTreeMap<(Invariant, String), f64>,
    /// Net amount moved *out of* each named source.
    net_out: BTreeMap<(Invariant, String), f64>,
}

impl ConservationLedger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a transfer.
    pub fn record(
        &mut self,
        quantity: Invariant,
        from: impl Into<Cow<'static, str>>,
        to: impl Into<Cow<'static, str>>,
        amount: f64,
        time: f64,
    ) {
        let from = from.into();
        let to = to.into();
        *self.net_out.entry((quantity, from.to_string())).or_insert(0.0) += amount;
        *self.net_in.entry((quantity, to.to_string())).or_insert(0.0) += amount;
        self.transfers.push(Transfer { quantity, from, to, amount, time });
    }

    /// Every recorded transfer, in order.
    pub fn transfers(&self) -> &[Transfer] {
        &self.transfers
    }

    /// Total amount of `quantity` delivered to `sink`.
    pub fn received(&self, quantity: Invariant, sink: &str) -> f64 {
        self.net_in.get(&(quantity, sink.to_string())).copied().unwrap_or(0.0)
    }

    /// Total amount of `quantity` taken from `source`.
    pub fn sent(&self, quantity: Invariant, source: &str) -> f64 {
        self.net_out.get(&(quantity, source.to_string())).copied().unwrap_or(0.0)
    }

    /// Compare a domain's observed change against what the ledger says it received.
    ///
    /// This is the question §14.3 exists to answer: *"whether a coupled result is
    /// physically inconsistent because of the model, a mapping error, a timestep
    /// issue, or an intentionally open system."*
    pub fn reconcile(
        &self,
        quantity: Invariant,
        sink: &str,
        observed_change: f64,
    ) -> Reconciliation {
        let expected = self.received(quantity, sink) - self.sent(quantity, sink);
        Reconciliation { quantity, expected, observed: observed_change }
    }

    /// Drop all entries.
    pub fn clear(&mut self) {
        self.transfers.clear();
        self.net_in.clear();
        self.net_out.clear();
    }

    /// Number of recorded transfers.
    pub fn len(&self) -> usize {
        self.transfers.len()
    }

    /// True when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.transfers.is_empty()
    }
}

/// The result of checking observed change against ledgered transfers.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Reconciliation {
    /// Which quantity was checked.
    pub quantity: Invariant,
    /// Net amount the ledger says should have arrived.
    pub expected: f64,
    /// Net amount actually observed to arrive.
    pub observed: f64,
}

impl Reconciliation {
    /// Observed minus expected. Nonzero means unaccounted-for change.
    pub fn discrepancy(&self) -> f64 {
        self.observed - self.expected
    }

    /// Discrepancy relative to the expected transfer magnitude.
    pub fn relative_discrepancy(&self) -> f64 {
        let scale = self.expected.abs().max(self.observed.abs());
        if scale == 0.0 { 0.0 } else { self.discrepancy() / scale }
    }

    /// True if the books balance to within `tolerance` (relative).
    pub fn balances(&self, tolerance: f64) -> bool {
        self.relative_discrepancy().abs() <= tolerance
    }
}

impl fmt::Display for Reconciliation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: expected {:+.6e} {}, observed {:+.6e} {}, unaccounted {:+.3e} ({:.2}%)",
            self.quantity,
            self.expected,
            self.quantity.si_unit(),
            self.observed,
            self.quantity.si_unit(),
            self.discrepancy(),
            self.relative_discrepancy() * 100.0
        )
    }
}

/// The residual trace of one iterative solve.
///
/// Spec §10.3: *"Residual histories are always observable and can stop a run on
/// divergence."*
#[derive(Clone, Default, Debug)]
pub struct ResidualHistory {
    values: Vec<f64>,
}

impl ResidualHistory {
    /// An empty history with room for `capacity` iterations.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { values: Vec::with_capacity(capacity) }
    }

    /// Append an iteration's residual norm.
    pub fn push(&mut self, residual: f64) {
        self.values.push(residual);
    }

    /// Begin a new solve, keeping the allocation.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// The full trace.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Residual before the first iteration.
    pub fn initial(&self) -> Option<f64> {
        self.values.first().copied()
    }

    /// Residual after the last iteration.
    pub fn latest(&self) -> Option<f64> {
        self.values.last().copied()
    }

    /// Iterations performed (one fewer than samples, since the first is the initial
    /// residual).
    pub fn iterations(&self) -> usize {
        self.values.len().saturating_sub(1)
    }

    /// Ratio of final to initial residual. Smaller is better.
    pub fn reduction(&self) -> f64 {
        match (self.initial(), self.latest()) {
            (Some(i), Some(l)) if i != 0.0 => l / i,
            _ => 0.0,
        }
    }

    /// True if the residual has grown or gone non-finite.
    ///
    /// Growth is judged against the *initial* residual rather than the previous
    /// iterate, because Krylov methods legitimately stall or bounce for an iteration
    /// or two without diverging.
    pub fn is_diverging(&self) -> bool {
        match (self.initial(), self.latest()) {
            (Some(i), Some(l)) => !l.is_finite() || (i.is_finite() && l > i * 10.0),
            _ => false,
        }
    }
}

/// How an iterative solve ended.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SolveOutcome {
    /// Residual fell below the tolerance.
    Converged {
        /// Iterations used.
        iterations: usize,
        /// Final residual norm.
        residual: f64,
    },
    /// The iteration cap was reached first.
    ///
    /// Not silently accepted: the caller decides whether a partially converged solve
    /// is usable, and the residual is reported either way (NFR-007).
    MaxIterations {
        /// Iterations used.
        iterations: usize,
        /// Final residual norm.
        residual: f64,
        /// Tolerance that was not met.
        tolerance: f64,
    },
    /// The residual grew or went non-finite.
    Diverged {
        /// Iterations completed before divergence was detected.
        iterations: usize,
        /// The residual that triggered detection.
        residual: f64,
    },
}

impl SolveOutcome {
    /// True only for a clean convergence.
    pub fn is_converged(&self) -> bool {
        matches!(self, SolveOutcome::Converged { .. })
    }

    /// Iterations used, however the solve ended.
    pub fn iterations(&self) -> usize {
        match self {
            SolveOutcome::Converged { iterations, .. }
            | SolveOutcome::MaxIterations { iterations, .. }
            | SolveOutcome::Diverged { iterations, .. } => *iterations,
        }
    }

    /// Final residual, however the solve ended.
    pub fn residual(&self) -> f64 {
        match self {
            SolveOutcome::Converged { residual, .. }
            | SolveOutcome::MaxIterations { residual, .. }
            | SolveOutcome::Diverged { residual, .. } => *residual,
        }
    }
}

impl fmt::Display for SolveOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveOutcome::Converged { iterations, residual } => {
                write!(f, "converged in {iterations} iterations (residual {residual:.3e})")
            }
            SolveOutcome::MaxIterations { iterations, residual, tolerance } => write!(
                f,
                "hit the {iterations}-iteration cap with residual {residual:.3e}, \
                 tolerance {tolerance:.3e}"
            ),
            SolveOutcome::Diverged { iterations, residual } => {
                write!(f, "diverged after {iterations} iterations (residual {residual:.3e})")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The display unit and the declared dimension must describe the same thing.
    /// A mismatch here would mean a plot axis labelled in units the value is not in.
    #[test]
    fn invariant_units_and_dimensions_agree() {
        let cases: &[(Invariant, Dimension)] = &[
            (Invariant::Energy, Dimension::ENERGY),
            (Invariant::KineticEnergy, Dimension::ENERGY),
            (Invariant::MomentumX, Dimension::MOMENTUM),
            (Invariant::AngularMomentum, Dimension::ACTION),
            (Invariant::Mass, Dimension::MASS),
            (Invariant::Amount, Dimension::AMOUNT),
            (Invariant::Charge, Dimension::CHARGE),
            (Invariant::ProbabilityNorm, Dimension::DIMENSIONLESS),
        ];
        for (invariant, expected) in cases {
            assert_eq!(invariant.dimension(), Some(*expected), "{invariant}");
        }
        // A field integral's dimension depends on the field, so it declares nothing.
        assert_eq!(Invariant::FieldIntegral.dimension(), None);
    }

    #[test]
    fn observations_round_trip_by_name() {
        let mut obs = Observations::new();
        obs.record_invariant("particles.energy", Invariant::Energy, 12.5);
        obs.record_metric("particles.mean_speed", 3.0, "m/s");
        assert_eq!(obs.len(), 2);
        assert_eq!(obs.value("particles.energy"), Some(12.5));
        assert_eq!(obs.get("particles.energy").unwrap().unit, "J");
        assert_eq!(obs.value("nope"), None);
    }

    #[test]
    fn observations_detect_non_finite_values() {
        let mut obs = Observations::new();
        obs.record_metric("ok", 1.0, "1");
        assert!(obs.first_non_finite().is_none());
        obs.record_metric("blown_up", f64::NAN, "1");
        assert_eq!(obs.first_non_finite().unwrap().name, "blown_up");
    }

    #[test]
    fn clearing_observations_keeps_capacity() {
        let mut obs = Observations::new();
        for i in 0..100 {
            obs.record_metric(format!("m{i}"), i as f64, "1");
        }
        obs.clear();
        assert!(obs.is_empty());
    }

    #[test]
    fn drift_monitor_reports_signed_and_relative_drift() {
        let mut d = DriftMonitor::new(Invariant::Energy, 100.0);
        d.record(101.0);
        assert!((d.absolute_drift() - 1.0).abs() < 1e-12);
        assert!((d.relative_drift() - 0.01).abs() < 1e-12);
        assert_eq!(d.samples(), 2);
    }

    /// A symplectic integrator's energy oscillates around the true value. Reporting
    /// only the endpoint would hide an excursion; the monitor must remember the worst
    /// sample, which is the number a validation report should quote.
    #[test]
    fn drift_monitor_remembers_the_worst_excursion() {
        let mut d = DriftMonitor::new(Invariant::Energy, 100.0);
        d.record(105.0);
        d.record(95.0);
        d.record(100.0);
        assert!(d.absolute_drift().abs() < 1e-12, "ends where it started");
        assert!((d.max_absolute_drift() - 5.0).abs() < 1e-12, "but swung by 5");
        assert!((d.max_relative_drift() - 0.05).abs() < 1e-12);
        assert!(d.within(0.06));
        assert!(!d.within(0.04));
    }

    #[test]
    fn drift_monitor_handles_a_zero_reference() {
        let mut d = DriftMonitor::new(Invariant::MomentumX, 0.0);
        d.record(1e-9);
        // Relative drift against zero falls back to absolute rather than infinity.
        assert!((d.relative_drift() - 1e-9).abs() < 1e-20);
        assert!(d.relative_drift().is_finite());
    }

    /// The §14.3 scenario: an exothermic reaction hands energy to a heat field. The
    /// ledger explains the heat field's gain, so it is a transfer, not drift.
    #[test]
    fn ledger_explains_a_coupled_energy_transfer() {
        let mut ledger = ConservationLedger::new();
        ledger.record(Invariant::Energy, "reaction.A_plus_B", "heat.source", 25.0, 0.1);
        ledger.record(Invariant::Energy, "reaction.A_plus_B", "heat.source", 15.0, 0.2);

        assert_eq!(ledger.len(), 2);
        assert!((ledger.received(Invariant::Energy, "heat.source") - 40.0).abs() < 1e-12);
        assert!((ledger.sent(Invariant::Energy, "reaction.A_plus_B") - 40.0).abs() < 1e-12);

        // The heat field gained exactly what was sent: the books balance.
        let r = ledger.reconcile(Invariant::Energy, "heat.source", 40.0);
        assert!(r.balances(1e-12));
        assert!(r.discrepancy().abs() < 1e-12);
    }

    /// If the heat field gained more than the ledger accounts for, that gap is the
    /// number worth investigating — and it is exactly what the ledger surfaces.
    #[test]
    fn ledger_surfaces_unaccounted_change() {
        let mut ledger = ConservationLedger::new();
        ledger.record(Invariant::Energy, "reaction", "heat.source", 100.0, 0.0);
        let r = ledger.reconcile(Invariant::Energy, "heat.source", 103.0);
        assert!((r.discrepancy() - 3.0).abs() < 1e-12);
        assert!((r.relative_discrepancy() - 3.0 / 103.0).abs() < 1e-12);
        assert!(!r.balances(1e-3));
        assert!(r.balances(0.03));
        let text = r.to_string();
        assert!(text.contains("unaccounted"), "{text}");
    }

    #[test]
    fn reconciling_an_untouched_sink_reports_no_discrepancy() {
        let ledger = ConservationLedger::new();
        let r = ledger.reconcile(Invariant::Energy, "heat.source", 0.0);
        assert_eq!(r.expected, 0.0);
        assert_eq!(r.relative_discrepancy(), 0.0);
        assert!(r.balances(0.0));
    }

    #[test]
    fn residual_history_tracks_reduction_and_iterations() {
        let mut h = ResidualHistory::with_capacity(8);
        for r in [1.0, 1e-1, 1e-3, 1e-6] {
            h.push(r);
        }
        assert_eq!(h.iterations(), 3);
        assert_eq!(h.initial(), Some(1.0));
        assert_eq!(h.latest(), Some(1e-6));
        assert!((h.reduction() - 1e-6).abs() < 1e-18);
        assert!(!h.is_diverging());
    }

    #[test]
    fn residual_history_detects_divergence_and_nans() {
        let mut growing = ResidualHistory::default();
        growing.push(1.0);
        growing.push(1e6);
        assert!(growing.is_diverging());

        let mut nan = ResidualHistory::default();
        nan.push(1.0);
        nan.push(f64::NAN);
        assert!(nan.is_diverging());

        // A brief stall is not divergence.
        let mut stalling = ResidualHistory::default();
        stalling.push(1.0);
        stalling.push(1.5);
        assert!(!stalling.is_diverging());
    }

    #[test]
    fn solve_outcomes_report_uniformly() {
        let converged = SolveOutcome::Converged { iterations: 12, residual: 1e-9 };
        assert!(converged.is_converged());
        assert_eq!(converged.iterations(), 12);

        let capped =
            SolveOutcome::MaxIterations { iterations: 500, residual: 1e-3, tolerance: 1e-8 };
        assert!(!capped.is_converged());
        assert!(capped.to_string().contains("cap"));

        let diverged = SolveOutcome::Diverged { iterations: 3, residual: f64::INFINITY };
        assert!(!diverged.is_converged());
        assert!(diverged.to_string().contains("diverged"));
    }
}
