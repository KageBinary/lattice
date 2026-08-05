//! Phase timing, memory accounting, and throughput.
//!
//! Spec §19.3 sets the rules a benchmark report has to follow:
//!
//! > Report simulated steps per second and useful physical time per wall-clock
//! > second, not FPS alone. Separate compute-only, rendering, output, and end-to-end
//! > measurements. Include memory footprint and compilation/startup cost.
//!
//! And §15.1: *"Measure end-to-end: include model compile time, upload/download,
//! solver work, coupling, rendering, and output — not kernel time alone."*
//!
//! The pressure both are pushing against is real: it is easy to publish a number that
//! measures only the inner loop, and easy for that number to be irrelevant because the
//! setup cost dominates. A [`Profile`] therefore accumulates named phases and always
//! reports each as a share of the total.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::json::Json;

/// Conventional phase names, so reports from different runs line up.
pub mod phase {
    /// Parsing and compiling the model.
    pub const COMPILE: &str = "compile";
    /// Allocating state and building acceleration structures.
    pub const SETUP: &str = "setup";
    /// Advancing the simulation.
    pub const COMPUTE: &str = "compute";
    /// Computing diagnostics and observations.
    pub const OBSERVE: &str = "observe";
    /// Writing artifacts to disk.
    pub const OUTPUT: &str = "output";
    /// Drawing.
    pub const RENDER: &str = "render";
}

/// One phase's accumulated cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PhaseStats {
    /// Total time spent.
    pub total: Duration,
    /// Number of times the phase was entered.
    pub calls: u64,
}

impl PhaseStats {
    /// Mean time per call.
    pub fn mean(&self) -> Duration {
        if self.calls == 0 { Duration::ZERO } else { self.total / self.calls as u32 }
    }
}

/// Accumulated wall-clock time per named phase.
///
/// Phases are reported in insertion order, which is the order the run actually
/// executed them — more useful in a report than alphabetical.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    order: Vec<String>,
    stats: BTreeMap<String, PhaseStats>,
}

impl Profile {
    /// An empty profile.
    pub fn new() -> Self {
        Self::default()
    }

    /// Time a closure, attributing its cost to `phase`.
    pub fn time<T>(&mut self, phase: &str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let value = f();
        self.record(phase, start.elapsed());
        value
    }

    /// Attribute an already-measured duration to a phase.
    pub fn record(&mut self, phase: &str, elapsed: Duration) {
        let entry = self.stats.entry(phase.to_string()).or_insert_with(|| {
            self.order.push(phase.to_string());
            PhaseStats::default()
        });
        entry.total += elapsed;
        entry.calls += 1;
    }

    /// Stats for one phase.
    pub fn get(&self, phase: &str) -> Option<PhaseStats> {
        self.stats.get(phase).copied()
    }

    /// Total across all phases.
    ///
    /// Note this is the sum of *measured* phases, which is not the same as wall-clock
    /// time for the whole run — anything not inside a `time` call is unattributed. The
    /// report shows both so the gap is visible rather than hidden.
    pub fn measured_total(&self) -> Duration {
        self.stats.values().map(|s| s.total).sum()
    }

    /// Phases in execution order.
    pub fn phases(&self) -> impl Iterator<Item = (&str, PhaseStats)> {
        self.order.iter().map(|name| (name.as_str(), self.stats[name]))
    }

    /// True when nothing has been measured.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    /// A human-readable table.
    pub fn report(&self, wall_clock: Option<Duration>) -> String {
        let measured = self.measured_total();
        let denominator = wall_clock.unwrap_or(measured);
        let mut out = String::new();
        out.push_str(&format!(
            "{:<14} {:>12} {:>8} {:>12} {:>7}\n",
            "phase", "total", "calls", "mean", "share"
        ));
        for (name, stats) in self.phases() {
            let share = if denominator.is_zero() {
                0.0
            } else {
                stats.total.as_secs_f64() / denominator.as_secs_f64() * 100.0
            };
            out.push_str(&format!(
                "{:<14} {:>12} {:>8} {:>12} {:>6.1}%\n",
                name,
                format_duration(stats.total),
                stats.calls,
                format_duration(stats.mean()),
                share
            ));
        }
        if let Some(wall) = wall_clock {
            let unattributed = wall.saturating_sub(measured);
            out.push_str(&format!(
                "{:<14} {:>12} {:>8} {:>12} {:>6.1}%\n",
                "unattributed",
                format_duration(unattributed),
                "",
                "",
                if wall.is_zero() {
                    0.0
                } else {
                    unattributed.as_secs_f64() / wall.as_secs_f64() * 100.0
                }
            ));
            out.push_str(&format!("{:<14} {:>12}\n", "wall clock", format_duration(wall)));
        }
        out
    }

    /// Machine-readable form for the run artifact.
    pub fn to_json(&self) -> Json {
        let mut root = Json::object();
        for (name, stats) in self.phases() {
            root.insert(
                name,
                Json::object()
                    .set("seconds", stats.total.as_secs_f64())
                    .set("calls", stats.calls)
                    .set("mean_seconds", stats.mean().as_secs_f64()),
            );
        }
        root
    }
}

/// Bytes held by named parts of a run.
///
/// Reported per structure rather than as one total, because "the simulation uses 400
/// MB" is not actionable while "the neighbour list uses 380 MB of it" is.
#[derive(Clone, Debug, Default)]
pub struct MemoryReport {
    entries: Vec<(String, usize)>,
}

impl MemoryReport {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a component's footprint.
    pub fn record(&mut self, name: impl Into<String>, bytes: usize) {
        self.entries.push((name.into(), bytes));
    }

    /// Sum over all components.
    pub fn total(&self) -> usize {
        self.entries.iter().map(|(_, b)| b).sum()
    }

    /// All recorded components.
    pub fn entries(&self) -> &[(String, usize)] {
        &self.entries
    }

    /// True when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A human-readable table.
    pub fn report(&self) -> String {
        let mut out = String::new();
        for (name, bytes) in &self.entries {
            out.push_str(&format!("{:<28} {:>12}\n", name, format_bytes(*bytes)));
        }
        out.push_str(&format!("{:<28} {:>12}\n", "total", format_bytes(self.total())));
        out
    }

    /// Machine-readable form.
    pub fn to_json(&self) -> Json {
        let mut root = Json::object();
        for (name, bytes) in &self.entries {
            root.insert(name.clone(), *bytes);
        }
        root.insert("total", self.total());
        root
    }
}

/// What a run achieved per unit of wall-clock time.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Throughput {
    /// Simulation steps taken.
    pub steps: u64,
    /// Physical time simulated, seconds.
    pub simulated_seconds: f64,
    /// Wall-clock time spent, excluding setup.
    pub wall_clock: Duration,
    /// Degrees of freedom advanced per step (particles, cells, …).
    pub elements: u64,
}

impl Throughput {
    /// Simulation steps per wall-clock second.
    pub fn steps_per_second(&self) -> f64 {
        let wall = self.wall_clock.as_secs_f64();
        if wall <= 0.0 { f64::INFINITY } else { self.steps as f64 / wall }
    }

    /// Simulated seconds per wall-clock second.
    ///
    /// The number §19.3 actually asks for. A simulation running at 10⁶ steps/s is
    /// useless if each step advances a femtosecond; this ratio says whether the run
    /// reaches interesting physical time.
    pub fn realtime_factor(&self) -> f64 {
        let wall = self.wall_clock.as_secs_f64();
        if wall <= 0.0 { f64::INFINITY } else { self.simulated_seconds / wall }
    }

    /// Element-updates per wall-clock second (particles/s, cells/s).
    pub fn element_updates_per_second(&self) -> f64 {
        self.steps_per_second() * self.elements as f64
    }

    /// A human-readable summary.
    pub fn report(&self) -> String {
        format!(
            "{:<28} {:>14}\n{:<28} {:>14}\n{:<28} {:>14}\n{:<28} {:>14}\n{:<28} {:>14}\n",
            "steps",
            self.steps,
            "elements per step",
            self.elements,
            "steps / wall second",
            format!("{:.1}", self.steps_per_second()),
            "element updates / wall second",
            format_count(self.element_updates_per_second()),
            "simulated s / wall s",
            format!("{:.4e}", self.realtime_factor()),
        )
    }

    /// Machine-readable form.
    pub fn to_json(&self) -> Json {
        Json::object()
            .set("steps", self.steps)
            .set("elements_per_step", self.elements)
            .set("simulated_seconds", self.simulated_seconds)
            .set("wall_clock_seconds", self.wall_clock.as_secs_f64())
            .set("steps_per_second", self.steps_per_second())
            .set("element_updates_per_second", self.element_updates_per_second())
            .set("realtime_factor", self.realtime_factor())
    }
}

/// Format a duration with a unit that keeps three or four significant figures.
pub fn format_duration(d: Duration) -> String {
    let seconds = d.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.3} s")
    } else if seconds >= 1e-3 {
        format!("{:.3} ms", seconds * 1e3)
    } else if seconds >= 1e-6 {
        format!("{:.3} µs", seconds * 1e6)
    } else {
        format!("{:.0} ns", seconds * 1e9)
    }
}

/// Format a byte count in binary units.
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{value:.2} {}", UNITS[unit]) }
}

/// Format a large count with an SI-style suffix.
pub fn format_count(value: f64) -> String {
    if value >= 1e9 {
        format!("{:.2}G", value / 1e9)
    } else if value >= 1e6 {
        format!("{:.2}M", value / 1e6)
    } else if value >= 1e3 {
        format!("{:.2}k", value / 1e3)
    } else {
        format!("{value:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_accumulate_across_calls() {
        let mut p = Profile::new();
        p.record(phase::COMPUTE, Duration::from_millis(10));
        p.record(phase::COMPUTE, Duration::from_millis(30));
        p.record(phase::OUTPUT, Duration::from_millis(5));

        let compute = p.get(phase::COMPUTE).unwrap();
        assert_eq!(compute.calls, 2);
        assert_eq!(compute.total, Duration::from_millis(40));
        assert_eq!(compute.mean(), Duration::from_millis(20));
        assert_eq!(p.measured_total(), Duration::from_millis(45));
    }

    #[test]
    fn phases_report_in_execution_order() {
        let mut p = Profile::new();
        p.record("zebra", Duration::from_millis(1));
        p.record("apple", Duration::from_millis(1));
        p.record("zebra", Duration::from_millis(1));
        let names: Vec<&str> = p.phases().map(|(n, _)| n).collect();
        assert_eq!(names, ["zebra", "apple"], "order must follow first use, not the alphabet");
    }

    #[test]
    fn timing_a_closure_returns_its_value() {
        let mut p = Profile::new();
        let answer = p.time(phase::COMPUTE, || 6 * 7);
        assert_eq!(answer, 42);
        assert_eq!(p.get(phase::COMPUTE).unwrap().calls, 1);
    }

    /// Time spent outside any measured phase must be visible, not silently absorbed.
    #[test]
    fn the_report_exposes_unattributed_time() {
        let mut p = Profile::new();
        p.record(phase::COMPUTE, Duration::from_millis(40));
        let text = p.report(Some(Duration::from_millis(100)));
        assert!(text.contains("unattributed"), "{text}");
        assert!(text.contains("60.0%"), "60ms of 100ms is unaccounted for:\n{text}");
    }

    #[test]
    fn an_empty_profile_reports_without_dividing_by_zero() {
        let p = Profile::new();
        let text = p.report(Some(Duration::ZERO));
        assert!(text.contains("phase"));
        assert!(p.is_empty());
    }

    #[test]
    fn memory_report_sums_components() {
        let mut m = MemoryReport::new();
        m.record("particles", 1024);
        m.record("cell list", 512);
        assert_eq!(m.total(), 1536);
        assert!(m.report().contains("1.50 KiB"));
        assert_eq!(m.to_json().get("total"), Some(&Json::Int(1536)));
    }

    /// The distinction §19.3 insists on: steps/s and physical-time ratio are
    /// different numbers, and only the second says whether the run is useful.
    #[test]
    fn throughput_separates_step_rate_from_physical_time() {
        let t = Throughput {
            steps: 100_000,
            simulated_seconds: 0.1,
            wall_clock: Duration::from_secs(2),
            elements: 65_536,
        };
        assert!((t.steps_per_second() - 50_000.0).abs() < 1e-9);
        assert!((t.realtime_factor() - 0.05).abs() < 1e-12);
        assert!((t.element_updates_per_second() - 50_000.0 * 65_536.0).abs() < 1.0);

        let text = t.report();
        assert!(text.contains("steps / wall second"), "{text}");
        assert!(text.contains("simulated s / wall s"), "{text}");
    }

    #[test]
    fn throughput_handles_a_zero_duration() {
        let t = Throughput {
            steps: 1,
            simulated_seconds: 1.0,
            wall_clock: Duration::ZERO,
            elements: 1,
        };
        assert!(t.steps_per_second().is_infinite());
        assert!(t.realtime_factor().is_infinite());
        // And serializes without producing invalid JSON.
        assert!(t.to_json().to_compact_string().contains("Infinity"));
    }

    #[test]
    fn durations_format_with_useful_units() {
        assert_eq!(format_duration(Duration::from_secs(2)), "2.000 s");
        assert_eq!(format_duration(Duration::from_millis(15)), "15.000 ms");
        assert_eq!(format_duration(Duration::from_micros(250)), "250.000 µs");
        assert_eq!(format_duration(Duration::from_nanos(40)), "40 ns");
    }

    #[test]
    fn bytes_format_in_binary_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.00 KiB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.00 MiB");
    }

    #[test]
    fn counts_format_with_si_suffixes() {
        assert_eq!(format_count(999.0), "999");
        assert_eq!(format_count(2_500.0), "2.50k");
        assert_eq!(format_count(3.2e6), "3.20M");
        assert_eq!(format_count(1.5e9), "1.50G");
    }

    #[test]
    fn profile_json_includes_every_phase() {
        let mut p = Profile::new();
        p.record(phase::SETUP, Duration::from_millis(3));
        p.record(phase::COMPUTE, Duration::from_millis(7));
        let json = p.to_json();
        assert!(json.get("setup").is_some());
        assert!(json.get("compute").is_some());
        assert_eq!(json.get("compute").unwrap().get("calls"), Some(&Json::Int(1)));
    }
}
