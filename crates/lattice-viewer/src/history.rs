//! The observation timeline behind the live plots.
//!
//! # Why series are grouped by unit
//!
//! Total energy is in joules; momentum is in kg·m/s. Plotting them on one pair of axes
//! would require two y-scales, and the alignment between two y-scales is arbitrary —
//! so the chart invents a relationship that is not in the data. A reader sees "energy
//! and momentum move together" when what they are seeing is a choice of scale.
//!
//! So: **one plot per unit, one axis each.** [`History::by_unit`] does the grouping,
//! and the app never draws two units on one plot.
//!
//! # Why decimation halves rather than truncates
//!
//! A long run produces more samples than a plot can show or memory should hold. Two
//! wrong ways to bound it: drop the oldest (the start of the run disappears, which is
//! exactly where the interesting transient is) or stop recording (the plot freezes
//! while the simulation runs on). Halving keeps the full time span at lower
//! resolution, which is the shape a reader is actually looking at.
//!
//! # Why drift is not always relative to the initial value
//!
//! Total momentum in a system set up at rest is conserved *at zero*. Its initial value
//! is not zero but 1e-15 — the round-off left over from summing a few hundred momenta
//! that cancel. Divide by that and a flawless run reports a drift of 70, which reads as
//! a catastrophic bug and is in fact a division by noise.
//!
//! So a domain may publish a companion series naming the scale its quantity should be
//! judged against — for momentum, the sum of the individual momentum magnitudes. When
//! the initial value is negligible against that scale, [`History::drift_of`] divides by
//! the scale instead and says which basis it used. See [`DriftBasis`].

use lattice_ir::{ObservationKind, Observations};

/// One recorded quantity over time.
#[derive(Clone, Debug)]
pub struct Series {
    /// The observation name, e.g. `"temperature.integral"`.
    pub name: String,
    /// Its SI unit, which is what groups it into a plot.
    pub unit: String,
    /// What sort of reading it is, as the *domain* declared it.
    ///
    /// The viewer used to decide "should this be conserved?" by matching the name
    /// against a list — `momentum_x`, `total_energy`, and so on. That guess is wrong
    /// exactly where it matters: a rigid scene with gravity and a floor publishes a
    /// momentum that is *supposed* to change, because gravity injects it and a wall
    /// absorbs it. Reading the domain's own claim instead means the panel reports what
    /// the solver promised rather than what the viewer assumed.
    pub kind: ObservationKind,
    /// `[time, value]` pairs.
    pub points: Vec<[f64; 2]>,
}

impl Series {
    /// True when the domain published this as a quantity it expects to hold.
    pub fn is_invariant(&self) -> bool {
        matches!(self.kind, ObservationKind::Invariant(_))
    }

    /// The first recorded value.
    pub fn initial(&self) -> Option<f64> {
        self.points.first().map(|point| point[1])
    }

    /// The most recent value.
    pub fn latest(&self) -> Option<f64> {
        self.points.last().map(|point| point[1])
    }

    /// Change from the initial value, relative to it.
    ///
    /// Falls back to the absolute change when the initial value is zero, where a
    /// relative measure is undefined and infinity would be useless in a plot.
    pub fn relative_drift(&self) -> Option<f64> {
        let (initial, latest) = (self.initial()?, self.latest()?);
        if initial == 0.0 {
            Some(latest)
        } else {
            Some((latest - initial) / initial.abs())
        }
    }

    /// True when the series never changes — a count, or a genuinely exact invariant.
    pub fn is_constant(&self) -> bool {
        let Some(first) = self.initial() else { return true };
        self.points.iter().all(|point| point[1] == first)
    }

    /// True when the whole series is negligible against an externally supplied scale.
    ///
    /// Distinct from [`Series::is_flat_to_roundoff`], which asks whether a series is
    /// flat *relative to itself*. A net momentum wandering between -1e-13 and +1e-13
    /// spans its own full range many times over, so the self-relative test calls it
    /// varying; against a momentum scale of 200 it is indistinguishable from zero.
    pub fn is_negligible_against(&self, scale: f64) -> bool {
        if !(scale.is_finite() && scale > 0.0) {
            return false;
        }
        self.points.iter().all(|point| point[1].abs() <= 1e-9 * scale)
    }

    /// True when the series varies only in the last couple of digits.
    ///
    /// A plot auto-scales to whatever range it is given, so a conserved quantity
    /// wobbling in its 15th significant digit draws exactly the same dramatic
    /// oscillation as one that doubles — with fifteen-digit axis labels to match.
    /// Beside a series that genuinely moves, that reads as instability where there is
    /// none. Such a series is better reported as a number than drawn as a shape.
    /// The largest magnitude this series has reached.
    ///
    /// The yardstick for "is the current reading distinguishable from zero?" when the
    /// domain published no scale of its own. A quantity that peaked at 4 N·s and now reads
    /// 1e-17 has stopped, whatever the digits say.
    pub fn peak_magnitude(&self) -> f64 {
        self.points
            .iter()
            .map(|point| point[1])
            .filter(|value| value.is_finite())
            .fold(0.0f64, |peak, value| peak.max(value.abs()))
    }

    pub fn is_flat_to_roundoff(&self) -> bool {
        let (Some(low), Some(high)) = (
            self.points.iter().map(|p| p[1]).fold(None, min_finite),
            self.points.iter().map(|p| p[1]).fold(None, max_finite),
        ) else {
            return true;
        };
        let magnitude = high.abs().max(low.abs()).max(f64::MIN_POSITIVE);
        (high - low) <= 1e-12 * magnitude
    }
}

/// The suffix marking a series as the scale for its siblings rather than an
/// observable in its own right.
///
/// `gas.momentum_scale` is the reference for `gas.momentum_x` and `gas.momentum_y`.
pub const SCALE_SUFFIX: &str = "_scale";

/// What a drift figure was divided by.
///
/// Reported alongside the number so a reader is never left guessing which denominator
/// produced it — the difference between "0.1% of the energy" and "0.1% of the total
/// momentum being cancelled" matters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriftBasis {
    /// The quantity's own initial value.
    InitialValue,
    /// A scale the domain published, because the initial value was numerically zero.
    PublishedScale,
    /// Nothing to divide by; the figure is the raw change in the quantity's own unit.
    Absolute,
}

impl DriftBasis {
    /// A phrase completing "…{n} …", naming the denominator.
    pub fn phrase(self) -> &'static str {
        match self {
            DriftBasis::InitialValue => "of the initial value",
            DriftBasis::PublishedScale => "of the published scale",
            DriftBasis::Absolute => "absolute — nothing to compare against",
        }
    }
}

/// How far a quantity has moved, and what that figure is measured against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Drift {
    /// The drift itself. Dimensionless unless `basis` is [`DriftBasis::Absolute`].
    pub value: f64,
    /// What it was divided by.
    pub basis: DriftBasis,
}

/// Recorded observations, in the order the domains published them.
#[derive(Clone, Debug)]
pub struct History {
    series: Vec<Series>,
    capacity: usize,
}

impl Default for History {
    fn default() -> Self {
        History::new(4096)
    }
}

impl History {
    /// A history holding at most `capacity` points per series before decimating.
    pub fn new(capacity: usize) -> History {
        History { series: Vec::new(), capacity: capacity.max(4) }
    }

    /// Append the current observations at `time`.
    pub fn record(&mut self, time: f64, observations: &Observations) {
        for observation in observations.iter() {
            let name = observation.name.as_ref();
            let index = match self.series.iter().position(|s| s.name == name) {
                Some(index) => index,
                None => {
                    self.series.push(Series {
                        name: name.to_string(),
                        unit: observation.unit.to_string(),
                        kind: observation.kind,
                        points: Vec::new(),
                    });
                    self.series.len() - 1
                }
            };
            let series = &mut self.series[index];
            series.points.push([time, observation.value]);
            if series.points.len() > self.capacity {
                decimate(&mut series.points);
            }
        }
    }

    /// Forget everything, keeping the allocation.
    pub fn clear(&mut self) {
        self.series.clear();
    }

    /// All series, in publication order.
    pub fn series(&self) -> &[Series] {
        &self.series
    }

    /// One series by name.
    pub fn get(&self, name: &str) -> Option<&Series> {
        self.series.iter().find(|series| series.name == name)
    }

    /// How many samples have been recorded.
    pub fn sample_count(&self) -> usize {
        self.series.first().map_or(0, |series| series.points.len())
    }

    /// The scale a domain published for judging `name`, if there is one.
    ///
    /// Components of a vector share one scale: `gas.momentum_x` and `gas.momentum_y`
    /// are both judged against `gas.momentum_scale`.
    pub fn scale_for(&self, name: &str) -> Option<f64> {
        self.scale_series_for(name)?.latest()
    }

    /// The companion series itself, so a label can name the yardstick it used.
    ///
    /// Only returned when its current value is a usable denominator.
    pub fn scale_series_for(&self, name: &str) -> Option<&Series> {
        let (prefix, quantity) = name.rsplit_once('.')?;
        let base = ["_x", "_y", "_z"]
            .iter()
            .find_map(|axis| quantity.strip_suffix(axis))
            .unwrap_or(quantity);
        let series = self.get(&format!("{prefix}.{base}{SCALE_SUFFIX}"))?;
        let usable = series.latest().is_some_and(|s| s.is_finite() && s > 0.0);
        usable.then_some(series)
    }

    /// How far `series` has drifted, divided by whichever denominator is meaningful.
    ///
    /// Prefers the quantity's own initial value. Falls back to a published scale when
    /// that initial value is negligible against it — the conserved-at-zero case, where
    /// dividing by the initial value divides by round-off.
    pub fn drift_of(&self, series: &Series) -> Option<Drift> {
        let (initial, latest) = (series.initial()?, series.latest()?);
        let change = latest - initial;

        if let Some(scale) = self.scale_for(&series.name)
            && initial.abs() <= 1e-6 * scale
        {
            return Some(Drift { value: change / scale, basis: DriftBasis::PublishedScale });
        }
        if initial.is_finite() && initial != 0.0 {
            return Some(Drift { value: change / initial.abs(), basis: DriftBasis::InitialValue });
        }
        Some(Drift { value: change, basis: DriftBasis::Absolute })
    }

    /// What a reading of `series` should be judged against before it is printed.
    ///
    /// The domain's published scale if there is one — `momentum_scale` exists for exactly
    /// this — and otherwise the largest magnitude the series has reached in this run.
    ///
    /// This is what stops a stationary body's panel from flickering. A crate resting on
    /// the floor has a net momentum that wanders between `-6.9e-18` and `+1.0e-17`: it is
    /// zero, and every digit *and the sign* change every few steps, so
    /// `format!("{:.3e}")` renders a motionless object as violent instability. The values
    /// table had no scale to judge that against, while the plots — which had this same bug
    /// twice before — already did.
    pub fn display_scale_for(&self, series: &Series) -> f64 {
        self.scale_for(&series.name).unwrap_or_else(|| series.peak_magnitude())
    }

    /// Whether a series carries enough variation to be worth drawing as a curve.
    ///
    /// Three ways to fail: never changing, varying only in the last digits of its own
    /// magnitude, or being negligible against the scale its domain published.
    pub fn is_worth_plotting(&self, series: &Series) -> bool {
        if series.is_constant() || series.is_flat_to_roundoff() {
            return false;
        }
        !self.scale_for(&series.name).is_some_and(|scale| series.is_negligible_against(scale))
    }

    /// Series grouped by unit, units in first-seen order.
    ///
    /// Each group becomes one plot with one y-axis. Nothing here ever puts two units
    /// on one pair of axes.
    pub fn by_unit(&self) -> Vec<(&str, Vec<&Series>)> {
        let mut groups: Vec<(&str, Vec<&Series>)> = Vec::new();
        for series in &self.series {
            match groups.iter_mut().find(|(unit, _)| *unit == series.unit) {
                Some((_, members)) => members.push(series),
                None => groups.push((series.unit.as_str(), vec![series])),
            }
        }
        groups
    }
}

fn min_finite(best: Option<f64>, value: f64) -> Option<f64> {
    if !value.is_finite() {
        return best;
    }
    Some(best.map_or(value, |b: f64| b.min(value)))
}

fn max_finite(best: Option<f64>, value: f64) -> Option<f64> {
    if !value.is_finite() {
        return best;
    }
    Some(best.map_or(value, |b: f64| b.max(value)))
}

/// Keep every other point, preserving the first and last.
fn decimate(points: &mut Vec<[f64; 2]>) {
    let last = points.last().copied();
    let mut kept: Vec<[f64; 2]> = points.iter().step_by(2).copied().collect();
    // The final sample is the current state, so it must survive.
    if let Some(last) = last
        && kept.last() != Some(&last)
    {
        kept.push(last);
    }
    *points = kept;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::Invariant;

    fn observations(pairs: &[(&str, f64)]) -> Observations {
        let mut out = Observations::new();
        for (name, value) in pairs {
            out.record_invariant(name.to_string(), Invariant::Energy, *value);
        }
        out
    }

    /// The numbers a real resting crate publishes, taken from
    /// `what_a_resting_crate_does_to_the_published_numbers` in `lattice-domain-rigid2d`.
    ///
    /// A single body dropped in the sandbox settles completely — kinetic energy `1e-35`,
    /// speed `1e-18` — and its net momentum then wanders across `±1e-17`, flipping sign
    /// every few steps. The physics is exactly right; the panel showed it as every digit
    /// changing constantly, which is what a user reported as "all the numbers fluctuating".
    #[test]
    fn a_settled_body_is_judged_against_a_scale_rather_than_its_own_digits() {
        let mut history = History::default();
        // The falling phase, which is what sets the scale of the quantity.
        for (step, momentum) in [(0, 0.0), (1, -2.4), (2, -4.9), (3, -7.3)] {
            let mut sample = Observations::new();
            sample.record_metric("scene.momentum_x", momentum, "kg·m/s");
            sample.record_metric("scene.kinetic_energy", 0.5 * momentum * momentum, "J");
            history.record(f64::from(step), &sample);
        }
        // Landed, and now at rest to round-off.
        for (step, momentum) in [(4, -6.938_893_903_907_228e-18), (5, 1.040_834_085_586_084e-17)] {
            let mut sample = Observations::new();
            sample.record_metric("scene.momentum_x", momentum, "kg·m/s");
            sample.record_metric("scene.kinetic_energy", 3.957e-35, "J");
            history.record(f64::from(step), &sample);
        }

        let momentum = history.get("scene.momentum_x").unwrap();
        assert_eq!(momentum.peak_magnitude(), 7.3);
        assert_eq!(history.display_scale_for(momentum), 7.3);

        let latest = momentum.latest().unwrap();
        assert_eq!(crate::render::format_value(latest), "1.041e-17", "the old behaviour");
        assert_eq!(
            crate::render::format_value_against(latest, history.display_scale_for(momentum)),
            "0",
            "a resting body must read as stopped"
        );

        let energy = history.get("scene.kinetic_energy").unwrap();
        assert_eq!(
            crate::render::format_value_against(
                energy.latest().unwrap(),
                history.display_scale_for(energy)
            ),
            "0"
        );
    }

    /// The other half of the claim: a genuinely small reading must still be shown.
    ///
    /// A quantity that is small for the whole run has a small peak too, so the ratio test
    /// leaves it alone. Suppressing it would be the same bug in the opposite direction.
    #[test]
    fn a_quantity_that_is_simply_small_is_still_printed() {
        let mut history = History::default();
        for step in 0..4 {
            let mut sample = Observations::new();
            // Penetration: 0.4 mm, constant, and genuinely the value.
            sample.record_metric("scene.penetration", 4.0e-4, "m");
            history.record(f64::from(step), &sample);
        }
        let series = history.get("scene.penetration").unwrap();
        assert_eq!(
            crate::render::format_value_against(
                series.latest().unwrap(),
                history.display_scale_for(series)
            ),
            "4.000e-4"
        );
    }

    /// A published scale outranks the series' own peak, because it is the domain's own
    /// statement about what "small" means for that quantity.
    #[test]
    fn a_published_scale_is_preferred_over_the_observed_peak() {
        let mut history = History::default();
        let mut sample = Observations::new();
        sample.record_metric("gas.momentum_x", 1e-9, "kg·m/s");
        sample.record_metric("gas.momentum_scale", 200.0, "kg·m/s");
        history.record(0.0, &sample);

        let series = history.get("gas.momentum_x").unwrap();
        assert_eq!(series.peak_magnitude(), 1e-9, "its own peak is tiny");
        assert_eq!(history.display_scale_for(series), 200.0, "the domain knows better");
        assert_eq!(
            crate::render::format_value_against(1e-9, history.display_scale_for(series)),
            "0"
        );
    }

    /// The panel asks the domain what it promised rather than guessing from the name.
    /// A rigid scene with gravity publishes a momentum that is supposed to change.
    #[test]
    fn a_series_remembers_the_kind_the_domain_declared() {
        let mut history = History::default();
        let mut sample = Observations::new();
        sample.record_invariant("closed.momentum_x", Invariant::MomentumX, 0.0);
        sample.record_metric("open.momentum_x", 71.0, "kg·m/s");
        history.record(0.0, &sample);

        assert!(history.get("closed.momentum_x").unwrap().is_invariant());
        assert!(
            !history.get("open.momentum_x").unwrap().is_invariant(),
            "same name, different claim — the name was never the right thing to read"
        );
    }

    #[test]
    fn recording_builds_series_in_publication_order() {
        let mut history = History::default();
        history.record(0.0, &observations(&[("b.energy", 1.0), ("a.energy", 2.0)]));
        history.record(0.1, &observations(&[("b.energy", 1.5), ("a.energy", 2.5)]));

        let names: Vec<&str> = history.series().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["b.energy", "a.energy"], "order follows publication, not the alphabet");
        assert_eq!(history.sample_count(), 2);
        assert_eq!(history.get("a.energy").unwrap().points, [[0.0, 2.0], [0.1, 2.5]]);
    }

    #[test]
    fn a_series_that_appears_late_still_gets_recorded() {
        let mut history = History::default();
        history.record(0.0, &observations(&[("a", 1.0)]));
        history.record(0.1, &observations(&[("a", 2.0), ("late", 9.0)]));
        assert_eq!(history.get("late").unwrap().points.len(), 1);
    }

    #[test]
    fn drift_is_relative_to_the_first_sample() {
        let mut history = History::default();
        history.record(0.0, &observations(&[("e", 100.0)]));
        history.record(1.0, &observations(&[("e", 101.0)]));
        let drift = history.get("e").unwrap().relative_drift().unwrap();
        assert!((drift - 0.01).abs() < 1e-12);
    }

    #[test]
    fn drift_against_a_zero_initial_value_falls_back_to_absolute() {
        let mut history = History::default();
        history.record(0.0, &observations(&[("p", 0.0)]));
        history.record(1.0, &observations(&[("p", 1e-9)]));
        let drift = history.get("p").unwrap().relative_drift().unwrap();
        assert!(drift.is_finite());
        assert!((drift - 1e-9).abs() < 1e-20);
    }

    /// The rule that keeps plots honest: never two units on one axis.
    #[test]
    fn series_group_by_unit() {
        let mut history = History::default();
        let mut sample = Observations::new();
        sample.record_invariant("a.energy", Invariant::Energy, 1.0);
        sample.record_invariant("a.momentum_x", Invariant::MomentumX, 2.0);
        sample.record_invariant("b.energy", Invariant::Energy, 3.0);
        history.record(0.0, &sample);

        let groups = history.by_unit();
        assert_eq!(groups.len(), 2, "joules and kg·m/s are different plots");

        let (unit, members) = &groups[0];
        assert_eq!(*unit, "J");
        assert_eq!(members.len(), 2, "both energies share one axis");
        assert_eq!(groups[1].0, "kg·m/s");
        assert_eq!(groups[1].1.len(), 1);
    }

    /// Decimation must preserve the *span* of the run — the transient at the start is
    /// usually the interesting part, so dropping the head would be the worst choice.
    #[test]
    fn decimation_halves_without_losing_the_ends() {
        let mut history = History::new(8);
        for step in 0..40 {
            history.record(f64::from(step), &observations(&[("e", f64::from(step))]));
        }

        let series = history.get("e").unwrap();
        assert!(series.points.len() <= 8, "capacity should bound the series");
        assert_eq!(series.points.first().unwrap()[0], 0.0, "the start of the run must survive");
        assert_eq!(series.points.last().unwrap()[0], 39.0, "the current state must survive");

        // And time still runs forward.
        for pair in series.points.windows(2) {
            assert!(pair[1][0] > pair[0][0]);
        }
    }

    #[test]
    fn constant_series_are_detected() {
        let mut history = History::default();
        for step in 0..5 {
            history.record(f64::from(step), &observations(&[("count", 4.0), ("energy", f64::from(step))]));
        }
        assert!(history.get("count").unwrap().is_constant());
        assert!(!history.get("energy").unwrap().is_constant());
    }

    #[test]
    fn clearing_resets_everything() {
        let mut history = History::default();
        history.record(0.0, &observations(&[("e", 1.0)]));
        history.clear();
        assert!(history.series().is_empty());
        assert_eq!(history.sample_count(), 0);
        assert!(history.get("e").is_none());
    }

    /// The bug this whole mechanism exists for: a perfectly conserved momentum
    /// reported as catastrophically broken because the denominator was round-off.
    #[test]
    fn momentum_conserved_at_zero_is_judged_against_the_published_scale() {
        let mut history = History::default();
        for (step, value) in [1e-15_f64, -9.6e-14].into_iter().enumerate() {
            let mut sample = Observations::new();
            sample.record_invariant("gas.momentum_x", Invariant::MomentumX, value);
            sample.record_metric("gas.momentum_scale", 200.0, "kg·m/s");
            history.record(step as f64, &sample);
        }

        let series = history.get("gas.momentum_x").unwrap();
        let naive = series.relative_drift().unwrap();
        assert!(naive.abs() > 10.0, "dividing by 1e-15 really does explode: {naive}");

        let drift = history.drift_of(series).unwrap();
        assert_eq!(drift.basis, DriftBasis::PublishedScale);
        assert!(drift.value.abs() < 1e-15, "against a scale of 200 this is nothing: {drift:?}");
        assert!(!history.is_worth_plotting(series), "1e-13 of 200 is not a curve");
    }

    /// A published scale must not hijack a quantity that has a scale of its own.
    #[test]
    fn a_meaningful_initial_value_still_wins_over_a_published_scale() {
        let mut history = History::default();
        for (step, value) in [100.0_f64, 101.0].into_iter().enumerate() {
            let mut sample = Observations::new();
            sample.record_invariant("gas.energy_x", Invariant::Energy, value);
            sample.record_metric("gas.energy_scale", 5000.0, "J");
            history.record(step as f64, &sample);
        }
        let drift = history.drift_of(history.get("gas.energy_x").unwrap()).unwrap();
        assert_eq!(drift.basis, DriftBasis::InitialValue);
        assert!((drift.value - 0.01).abs() < 1e-12);
    }

    #[test]
    fn a_vectors_components_share_one_scale() {
        let mut history = History::default();
        let mut sample = Observations::new();
        sample.record_invariant("gas.momentum_x", Invariant::MomentumX, 0.0);
        sample.record_invariant("gas.momentum_y", Invariant::MomentumY, 0.0);
        sample.record_metric("gas.momentum_scale", 7.0, "kg·m/s");
        history.record(0.0, &sample);

        assert_eq!(history.scale_for("gas.momentum_x"), Some(7.0));
        assert_eq!(history.scale_for("gas.momentum_y"), Some(7.0));
        assert_eq!(history.scale_for("gas.energy"), None, "no scale published for energy");
        assert_eq!(history.scale_for("bare"), None, "a name with no domain prefix");
    }

    #[test]
    fn drift_with_nothing_to_divide_by_is_reported_as_absolute() {
        let mut history = History::default();
        history.record(0.0, &observations(&[("p", 0.0)]));
        history.record(1.0, &observations(&[("p", 1e-9)]));
        let drift = history.drift_of(history.get("p").unwrap()).unwrap();
        assert_eq!(drift.basis, DriftBasis::Absolute);
        assert!((drift.value - 1e-9).abs() < 1e-20);
    }

    #[test]
    fn an_empty_history_reports_nothing_rather_than_panicking() {
        let history = History::default();
        assert_eq!(history.sample_count(), 0);
        assert!(history.by_unit().is_empty());
    }
}
