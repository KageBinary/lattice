//! The validation lab.
//!
//! Spec §19.1 defines a hierarchy of validation levels, and §19.2 lists the canonical
//! cases every release must pass. This crate is where those live — not as `assert!`
//! statements buried in a test module, but as a registry of named cases that each
//! *measure a number and report it*.
//!
//! # Why measured, not just passing
//!
//! A green test says "the error was under the threshold." A validation report says
//! "the observed convergence order was 1.987, the declared order is 2, and the
//! diffusing Gaussian matched the analytic heat kernel to 3.4e-3 peak-relative." The
//! second is what spec P1 asks for — *"every module must state governing equations,
//! discretization, assumptions, valid regimes, error metrics, and known
//! non-conservation"* — and what NFR-003 means by publishing error metrics rather
//! than merely having them.
//!
//! It also catches a failure mode that pass/fail cannot: an error that is still under
//! tolerance but has quietly grown by a factor of ten.
//!
//! # Running
//!
//! ```
//! use lattice_validation::ValidationReport;
//!
//! // Matches both `free_fall` and `free_fall_is_mass_independent`.
//! let report = ValidationReport::run_matching("free_fall");
//! assert_eq!(report.total(), 2);
//! assert!(report.all_passed(), "{}", report.text());
//!
//! // Each case reports the number it measured, not just a verdict.
//! let measured = &report.results()[0].outcome;
//! assert_eq!(measured.unit, "m");
//! assert!(measured.observed < 1e-9, "{}: {}", measured.metric, measured.observed);
//! ```

mod contracts;
mod heat;
mod particles;

use std::time::{Duration, Instant};

use lattice_observe::Json;

/// Where a case sits in the validation hierarchy of spec §19.1.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Level {
    /// Is the local operator implemented correctly?
    Unit,
    /// Does a broad invariant hold — symmetry, positivity, conservation?
    Property,
    /// Does the solver match a known closed-form solution?
    Analytic,
    /// Does the discretization converge at the expected order?
    Manufactured,
    /// Do two independent schemes agree?
    CrossScheme,
    /// Does a coupled model behave and account for its transfers?
    Scenario,
}

impl Level {
    /// Short label for reports.
    pub const fn label(self) -> &'static str {
        match self {
            Level::Unit => "unit",
            Level::Property => "property",
            Level::Analytic => "analytic",
            Level::Manufactured => "manufactured",
            Level::CrossScheme => "cross-scheme",
            Level::Scenario => "scenario",
        }
    }

    /// The question this level answers.
    pub const fn question(self) -> &'static str {
        match self {
            Level::Unit => "is the local operator implemented correctly?",
            Level::Property => "does a broad invariant hold?",
            Level::Analytic => "does the solver match a known solution?",
            Level::Manufactured => "does the discretization converge at the expected order?",
            Level::CrossScheme => "do independent schemes agree?",
            Level::Scenario => "does a coupled model behave and account for transfers?",
        }
    }
}

/// How an observed value is judged.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Criterion {
    /// `|observed − expected| ≤ tolerance`.
    Near {
        /// The value theory predicts.
        expected: f64,
        /// How far from it is acceptable.
        tolerance: f64,
    },
    /// `observed ≤ limit`.
    AtMost {
        /// The upper bound.
        limit: f64,
    },
    /// `observed ≥ floor`.
    AtLeast {
        /// The lower bound.
        floor: f64,
    },
}

impl Criterion {
    fn accepts(&self, observed: f64) -> bool {
        if !observed.is_finite() {
            return false;
        }
        match *self {
            Criterion::Near { expected, tolerance } => (observed - expected).abs() <= tolerance,
            Criterion::AtMost { limit } => observed <= limit,
            Criterion::AtLeast { floor } => observed >= floor,
        }
    }

    fn describe(&self) -> String {
        match *self {
            Criterion::Near { expected, tolerance } => {
                format!("{expected:.6} ± {tolerance:.3e}")
            }
            Criterion::AtMost { limit } => format!("≤ {limit:.3e}"),
            Criterion::AtLeast { floor } => format!("≥ {floor:.3e}"),
        }
    }
}

/// What a case measured.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// What was measured, e.g. `"observed convergence order"`.
    pub metric: String,
    /// The unit of the measurement, or `"1"` for a ratio.
    pub unit: &'static str,
    /// The measured value.
    pub observed: f64,
    /// How it is judged.
    pub criterion: Criterion,
    /// Anything else worth recording.
    pub notes: Vec<String>,
}

impl Outcome {
    /// A measurement judged against a predicted value.
    pub fn near(
        metric: impl Into<String>,
        unit: &'static str,
        observed: f64,
        expected: f64,
        tolerance: f64,
    ) -> Self {
        Self {
            metric: metric.into(),
            unit,
            observed,
            criterion: Criterion::Near { expected, tolerance },
            notes: Vec::new(),
        }
    }

    /// An error measurement judged against an upper bound.
    pub fn at_most(
        metric: impl Into<String>,
        unit: &'static str,
        observed: f64,
        limit: f64,
    ) -> Self {
        Self {
            metric: metric.into(),
            unit,
            observed,
            criterion: Criterion::AtMost { limit },
            notes: Vec::new(),
        }
    }

    /// A measurement judged against a lower bound.
    pub fn at_least(
        metric: impl Into<String>,
        unit: &'static str,
        observed: f64,
        floor: f64,
    ) -> Self {
        Self {
            metric: metric.into(),
            unit,
            observed,
            criterion: Criterion::AtLeast { floor },
            notes: Vec::new(),
        }
    }

    /// Attach a note.
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Whether the measurement satisfies its criterion.
    pub fn passed(&self) -> bool {
        self.criterion.accepts(self.observed)
    }
}

/// One named validation case.
#[derive(Clone, Copy)]
pub struct Case {
    /// Unique identifier, used for filtering.
    pub name: &'static str,
    /// Which module this exercises.
    pub domain: &'static str,
    /// Where it sits in the hierarchy.
    pub level: Level,
    /// What the case establishes, in one sentence.
    pub claim: &'static str,
    /// The measurement.
    pub run: fn() -> Outcome,
}

impl core::fmt::Debug for Case {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Case").field("name", &self.name).field("level", &self.level).finish()
    }
}

/// Every registered case, in a stable order.
pub fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    cases.extend_from_slice(particles::CASES);
    cases.extend_from_slice(heat::CASES);
    cases.extend_from_slice(contracts::CASES);
    cases
}

/// A case together with what it measured and how long it took.
#[derive(Clone, Debug)]
pub struct CaseResult {
    /// The case that ran.
    pub case: Case,
    /// What it measured.
    pub outcome: Outcome,
    /// How long it took.
    pub elapsed: Duration,
}

impl CaseResult {
    /// Whether the case passed.
    pub fn passed(&self) -> bool {
        self.outcome.passed()
    }
}

/// The results of running a set of cases.
#[derive(Clone, Debug)]
pub struct ValidationReport {
    results: Vec<CaseResult>,
}

impl ValidationReport {
    /// Run every registered case.
    pub fn run_all() -> Self {
        Self::run(all_cases())
    }

    /// Run only the cases whose name or domain contains `pattern`.
    pub fn run_matching(pattern: &str) -> Self {
        let cases: Vec<Case> = all_cases()
            .into_iter()
            .filter(|c| c.name.contains(pattern) || c.domain.contains(pattern))
            .collect();
        Self::run(cases)
    }

    /// Run a specific list of cases.
    pub fn run(cases: Vec<Case>) -> Self {
        let results = cases
            .into_iter()
            .map(|case| {
                let start = Instant::now();
                let outcome = (case.run)();
                CaseResult { case, outcome, elapsed: start.elapsed() }
            })
            .collect();
        Self { results }
    }

    /// All results.
    pub fn results(&self) -> &[CaseResult] {
        &self.results
    }

    /// Number of cases run.
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Number that passed.
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.passed()).count()
    }

    /// Number that failed.
    pub fn failed(&self) -> usize {
        self.total() - self.passed()
    }

    /// Whether everything passed.
    pub fn all_passed(&self) -> bool {
        self.failed() == 0
    }

    /// Total time spent running cases.
    pub fn elapsed(&self) -> Duration {
        self.results.iter().map(|r| r.elapsed).sum()
    }

    /// A human-readable report, grouped by validation level.
    pub fn text(&self) -> String {
        let mut out = String::new();
        let mut levels: Vec<Level> = self.results.iter().map(|r| r.case.level).collect();
        levels.sort_unstable();
        levels.dedup();

        for level in levels {
            out.push_str(&format!("\n{} — {}\n", level.label().to_uppercase(), level.question()));
            for result in self.results.iter().filter(|r| r.case.level == level) {
                let mark = if result.passed() { "PASS" } else { "FAIL" };
                out.push_str(&format!(
                    "  [{mark}] {:<42} {:>12.6} {:<10} (want {})\n",
                    result.case.name,
                    result.outcome.observed,
                    result.outcome.unit,
                    result.outcome.criterion.describe()
                ));
                out.push_str(&format!("         {}\n", result.case.claim));
                out.push_str(&format!("         metric: {}\n", result.outcome.metric));
                for note in &result.outcome.notes {
                    out.push_str(&format!("         note:   {note}\n"));
                }
            }
        }

        out.push_str(&format!(
            "\n{} of {} cases passed in {}\n",
            self.passed(),
            self.total(),
            lattice_observe::format_duration(self.elapsed())
        ));
        if !self.all_passed() {
            out.push_str("FAILING CASES:\n");
            for result in self.results.iter().filter(|r| !r.passed()) {
                out.push_str(&format!(
                    "  {} — {} was {} but wanted {}\n",
                    result.case.name,
                    result.outcome.metric,
                    result.outcome.observed,
                    result.outcome.criterion.describe()
                ));
            }
        }
        out
    }

    /// Machine-readable form, for the run artifact and CI.
    pub fn to_json(&self) -> Json {
        let mut cases = Json::array();
        for result in &self.results {
            let mut entry = Json::object()
                .set("name", result.case.name)
                .set("domain", result.case.domain)
                .set("level", result.case.level.label())
                .set("claim", result.case.claim)
                .set("metric", result.outcome.metric.clone())
                .set("unit", result.outcome.unit)
                .set("observed", result.outcome.observed)
                .set("passed", result.passed())
                .set("seconds", result.elapsed.as_secs_f64());
            match result.outcome.criterion {
                Criterion::Near { expected, tolerance } => {
                    entry.insert("expected", expected);
                    entry.insert("tolerance", tolerance);
                }
                Criterion::AtMost { limit } => entry.insert("limit", limit),
                Criterion::AtLeast { floor } => entry.insert("floor", floor),
            }
            if !result.outcome.notes.is_empty() {
                entry.insert("notes", result.outcome.notes.clone());
            }
            cases.push(entry);
        }

        Json::object()
            .set("total", self.total())
            .set("passed", self.passed())
            .set("failed", self.failed())
            .set("all_passed", self.all_passed())
            .set("cases", cases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suite itself. Every canonical case in spec §19.2 must pass, and the
    /// failure message carries the full measured report so a regression says *what*
    /// moved and by how much.
    #[test]
    fn the_full_validation_suite_passes() {
        let report = ValidationReport::run_all();
        assert!(report.all_passed(), "{}", report.text());
    }

    /// Spec §19.2 enumerates the cases a release must carry. This checks the registry
    /// actually covers each validation level rather than being all of one kind.
    #[test]
    fn the_registry_covers_every_validation_level_it_claims() {
        let cases = all_cases();
        assert!(cases.len() >= 15, "only {} cases registered", cases.len());

        for level in [Level::Property, Level::Analytic, Level::Manufactured, Level::CrossScheme] {
            assert!(
                cases.iter().any(|c| c.level == level),
                "no case at the {} level",
                level.label()
            );
        }
        for domain in ["particles2d", "grid2d"] {
            assert!(cases.iter().any(|c| c.domain == domain), "no cases for {domain}");
        }
    }

    #[test]
    fn case_names_are_unique() {
        let cases = all_cases();
        let mut names: Vec<&str> = cases.iter().map(|c| c.name).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "duplicate case names in the registry");
    }

    #[test]
    fn filtering_selects_a_subset() {
        let all = ValidationReport::run_all().total();
        let grid = ValidationReport::run_matching("grid2d").total();
        assert!(grid > 0 && grid < all, "{grid} of {all}");
    }

    #[test]
    fn criteria_reject_non_finite_observations() {
        // A case that produced a NaN must never be reported as passing, whatever
        // the criterion (NFR-007).
        assert!(!Criterion::AtMost { limit: 1.0 }.accepts(f64::NAN));
        assert!(!Criterion::AtLeast { floor: 0.0 }.accepts(f64::NAN));
        assert!(!Criterion::Near { expected: 0.0, tolerance: 1e9 }.accepts(f64::INFINITY));
    }

    #[test]
    fn criteria_judge_as_advertised() {
        assert!(Criterion::Near { expected: 2.0, tolerance: 0.1 }.accepts(1.95));
        assert!(!Criterion::Near { expected: 2.0, tolerance: 0.1 }.accepts(2.2));
        assert!(Criterion::AtMost { limit: 1e-6 }.accepts(1e-9));
        assert!(!Criterion::AtMost { limit: 1e-9 }.accepts(1e-6));
        assert!(Criterion::AtLeast { floor: 100.0 }.accepts(1e6));
    }

    #[test]
    fn the_report_renders_both_ways() {
        let report = ValidationReport::run_matching("free_fall");
        let text = report.text();
        assert!(text.contains("free_fall"), "{text}");
        assert!(text.contains("PASS"), "{text}");

        let json = report.to_json();
        assert_eq!(json.get("all_passed"), Some(&Json::Bool(true)));
        assert!(json.get("cases").is_some());
    }
}
