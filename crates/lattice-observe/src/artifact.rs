//! The self-describing run artifact of spec §18.1.
//!
//! > Every execution produces a self-describing run artifact. The artifact contains
//! > project hash, compiled-model hash, engine version, backend and device, precision,
//! > seeds, solver profiles, timestep history, parameter values, selected state
//! > snapshots, observations, warnings, validation results, and external-job
//! > provenance.
//!
//! # The content hash
//!
//! [`RunArtifact::content_hash`] covers the *physics* — configuration, solver
//! contracts, seeds, and the observation timeline — and deliberately excludes wall
//! clock times, memory figures, and the timestamp. Two runs of the same model on the
//! same build must produce the same hash even though one took longer, which is what
//! makes the hash usable as the regression signal FR-011 asks for. Including timings
//! would make every run unique and the hash worthless.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lattice_ir::{Observations, SolverContract};

use crate::json::Json;
use crate::timing::{MemoryReport, Profile, Throughput};

/// One sampling of the observation set.
#[derive(Clone, Debug)]
pub struct TimelineSample {
    /// Simulation time, seconds.
    pub time: f64,
    /// Step index.
    pub step: u64,
    /// Recorded values, in the order the domains published them.
    pub values: Vec<(String, f64)>,
}

/// A complete record of one execution.
#[derive(Clone, Debug)]
pub struct RunArtifact {
    engine_version: String,
    run_name: String,
    backend: String,
    precision: String,
    seed: Option<u64>,
    created_unix_seconds: Option<u64>,
    contracts: Vec<&'static SolverContract>,
    parameters: Json,
    timeline: Vec<TimelineSample>,
    /// Timing profile. Excluded from the content hash.
    pub profile: Profile,
    /// Memory footprint. Excluded from the content hash.
    pub memory: MemoryReport,
    /// Throughput summary. Excluded from the content hash.
    pub throughput: Option<Throughput>,
    warnings: Vec<String>,
    extra: Json,
}

impl RunArtifact {
    /// Start an artifact for a named run.
    pub fn new(run_name: impl Into<String>) -> Self {
        Self {
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            run_name: run_name.into(),
            backend: "cpu-scalar".to_string(),
            precision: "accurate64".to_string(),
            seed: None,
            created_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs()),
            contracts: Vec::new(),
            parameters: Json::object(),
            timeline: Vec::new(),
            profile: Profile::new(),
            memory: MemoryReport::new(),
            throughput: None,
            warnings: Vec::new(),
            extra: Json::object(),
        }
    }

    /// Record which backend produced this run.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = backend.into();
        self
    }

    /// Record the numeric precision used.
    pub fn with_precision(mut self, precision: impl Into<String>) -> Self {
        self.precision = precision.into();
        self
    }

    /// Record the random seed, without which a stochastic run cannot be replayed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Suppress the timestamp, so the artifact is byte-identical across runs.
    ///
    /// Used by tests and by any workflow that diffs artifacts, where a changing
    /// timestamp is pure noise.
    pub fn without_timestamp(mut self) -> Self {
        self.created_unix_seconds = None;
        self
    }

    /// Attach a solver's published contract.
    ///
    /// NFR-003 requires solver assumptions to travel with results. Embedding the
    /// contract means a stored artifact still states what the numbers assumed, long
    /// after the model file has been edited.
    pub fn add_contract(&mut self, contract: &'static SolverContract) {
        self.contracts.push(contract);
    }

    /// Record a run parameter.
    pub fn set_parameter(&mut self, key: impl Into<String>, value: impl Into<Json>) {
        self.parameters.insert(key, value);
    }

    /// Attach an arbitrary extra section, such as validation results.
    pub fn set_section(&mut self, key: impl Into<String>, value: Json) {
        self.extra.insert(key, value);
    }

    /// Note a warning that should travel with the results.
    pub fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    /// Sample the current observations into the timeline.
    pub fn record(&mut self, time: f64, step: u64, observations: &Observations) {
        self.timeline.push(TimelineSample {
            time,
            step,
            values: observations.iter().map(|o| (o.name.to_string(), o.value)).collect(),
        });
    }

    /// Set the throughput summary.
    pub fn set_throughput(&mut self, throughput: Throughput) {
        self.throughput = Some(throughput);
    }

    /// Samples recorded so far.
    pub fn timeline(&self) -> &[TimelineSample] {
        &self.timeline
    }

    /// Warnings recorded so far.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The first non-finite value anywhere in the timeline, if any.
    ///
    /// NFR-007: instability must be exposed. A run that produced a `NaN` at step
    /// 4,000 of 10,000 should say so at the top of its artifact, not bury it.
    pub fn first_non_finite(&self) -> Option<(&TimelineSample, &str)> {
        self.timeline.iter().find_map(|sample| {
            sample
                .values
                .iter()
                .find(|(_, v)| !v.is_finite())
                .map(|(name, _)| (sample, name.as_str()))
        })
    }

    /// The physics-only part of the artifact, which the content hash covers.
    fn canonical_json(&self) -> Json {
        let mut root = Json::object()
            .set("engine_version", self.engine_version.clone())
            .set("run", self.run_name.clone())
            .set("backend", self.backend.clone())
            .set("precision", self.precision.clone())
            .set("seed", self.seed);

        let mut contracts = Json::array();
        for contract in &self.contracts {
            contracts.push(contract_to_json(contract));
        }
        root.insert("solvers", contracts);
        root.insert("parameters", self.parameters.clone());

        let mut timeline = Json::array();
        for sample in &self.timeline {
            let mut values = Json::object();
            for (name, value) in &sample.values {
                values.insert(name.clone(), *value);
            }
            timeline.push(
                Json::object().set("t", sample.time).set("step", sample.step).set("values", values),
            );
        }
        root.insert("timeline", timeline);
        root.insert("warnings", self.warnings.clone());
        root
    }

    /// A stable 64-bit digest of the run's physics.
    ///
    /// Identical inputs on an identical build produce an identical hash. Timings,
    /// memory figures and the creation timestamp are excluded by construction.
    pub fn content_hash(&self) -> u64 {
        fnv1a_64(self.canonical_json().to_compact_string().as_bytes())
    }

    /// The full artifact, physics and performance together.
    pub fn to_json(&self) -> Json {
        let mut root = self.canonical_json();
        root.insert("content_hash", format!("{:016x}", self.content_hash()));
        if let Some(created) = self.created_unix_seconds {
            root.insert("created_unix_seconds", created);
        }

        let mut performance = Json::object();
        if !self.profile.is_empty() {
            performance.insert("phases", self.profile.to_json());
        }
        if !self.memory.is_empty() {
            performance.insert("memory_bytes", self.memory.to_json());
        }
        if let Some(t) = self.throughput {
            performance.insert("throughput", t.to_json());
        }
        if let Json::Object(entries) = &performance
            && !entries.is_empty()
        {
            root.insert("performance", performance);
        }

        if let Json::Object(entries) = &self.extra {
            for (key, value) in entries {
                root.insert(key.clone(), value.clone());
            }
        }
        root
    }

    /// Write the artifact to a file, creating parent directories as needed.
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut text = self.to_json().to_pretty_string();
        text.push('\n');
        fs::write(path, text)
    }

    /// A short human-readable summary.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "run `{}`  engine {}  backend {}  precision {}\n",
            self.run_name, self.engine_version, self.backend, self.precision
        );
        out.push_str(&format!("content hash: {:016x}\n", self.content_hash()));
        out.push_str(&format!("samples: {}\n", self.timeline.len()));
        if let Some(t) = self.throughput {
            out.push_str(&t.report());
        }
        if !self.warnings.is_empty() {
            out.push_str(&format!("warnings: {}\n", self.warnings.len()));
            for w in &self.warnings {
                out.push_str(&format!("  - {w}\n"));
            }
        }
        if let Some((sample, name)) = self.first_non_finite() {
            out.push_str(&format!(
                "NON-FINITE VALUE: `{name}` at step {} (t = {:e})\n",
                sample.step, sample.time
            ));
        }
        out
    }
}

/// Serialize a solver contract.
pub fn contract_to_json(contract: &SolverContract) -> Json {
    Json::object()
        .set("name", contract.name)
        .set("summary", contract.summary)
        .set("fidelity", contract.fidelity.code())
        .set("fidelity_guarantee", contract.fidelity.guarantee())
        .set("equations", contract.governing_equations.to_vec())
        .set("discretization", contract.discretization)
        .set("integrator", contract.integrator)
        .set("assumptions", contract.assumptions.to_vec())
        .set("valid_regime", contract.valid_regime)
        .set("stability", contract.stability)
        .set(
            "conserves",
            contract.conserves.iter().map(|i| i.name()).collect::<Vec<_>>(),
        )
        .set("known_non_conservation", contract.known_non_conservation.to_vec())
        .set(
            "precisions",
            contract.precisions.iter().map(|p| p.code()).collect::<Vec<_>>(),
        )
        .set("deterministic", contract.deterministic)
        .set("differentiable", contract.differentiable)
        .set("validation_cases", contract.validation_cases.to_vec())
        .set("references", contract.references.to_vec())
}

/// FNV-1a, 64-bit.
///
/// Not cryptographic — it is a change detector, not a tamper detector. Chosen for
/// being short enough to read in full here, so the hash's meaning never depends on a
/// dependency's version.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Convenience: seconds as a `Duration`, for building throughput records.
pub fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(value.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{FidelityProfile, Invariant, Precision};

    static TEST_CONTRACT: SolverContract = SolverContract {
        name: "test.solver",
        summary: "for artifact tests",
        governing_equations: &["du/dt = 0"],
        discretization: "none",
        integrator: "none",
        assumptions: &["nothing happens"],
        valid_regime: "tests",
        stability: "unconditional",
        conserves: &[Invariant::Energy],
        known_non_conservation: &["nothing"],
        fidelity: FidelityProfile::Interactive,
        precisions: &[Precision::Accurate64],
        deterministic: true,
        differentiable: false,
        validation_cases: &["trivial"],
        references: &[],
    };

    fn sample_observations(energy: f64) -> Observations {
        let mut obs = Observations::new();
        obs.record_invariant("test.energy", Invariant::Energy, energy);
        obs.record_metric("test.speed", 1.5, "m/s");
        obs
    }

    fn artifact() -> RunArtifact {
        let mut a = RunArtifact::new("demo").with_seed(7).without_timestamp();
        a.add_contract(&TEST_CONTRACT);
        a.set_parameter("dt", 1e-3);
        a.record(0.0, 0, &sample_observations(10.0));
        a.record(0.1, 100, &sample_observations(10.001));
        a
    }

    #[test]
    fn artifact_records_the_timeline() {
        let a = artifact();
        assert_eq!(a.timeline().len(), 2);
        assert_eq!(a.timeline()[1].step, 100);
        assert_eq!(a.timeline()[0].values[0].0, "test.energy");
    }

    #[test]
    fn json_contains_the_required_sections() {
        let json = artifact().to_json();
        for key in ["engine_version", "run", "backend", "precision", "seed", "solvers", "timeline"]
        {
            assert!(json.get(key).is_some(), "missing `{key}`");
        }
        // The solver contract travels with the results.
        let solvers = json.get("solvers").unwrap();
        assert!(solvers.to_compact_string().contains("nothing happens"));
    }

    /// The hash must be stable across runs of the same physics...
    #[test]
    fn content_hash_is_stable_for_identical_physics() {
        assert_eq!(artifact().content_hash(), artifact().content_hash());
    }

    /// ...and must ignore performance, which varies from run to run.
    #[test]
    fn content_hash_ignores_timings_and_memory() {
        let baseline = artifact().content_hash();

        let mut with_timings = artifact();
        with_timings.profile.record("compute", Duration::from_millis(123));
        with_timings.memory.record("particles", 4096);
        with_timings.set_throughput(Throughput {
            steps: 100,
            simulated_seconds: 0.1,
            wall_clock: Duration::from_millis(500),
            elements: 64,
        });

        assert_eq!(
            with_timings.content_hash(),
            baseline,
            "performance data must not change the physics hash"
        );
        // But it does appear in the serialized artifact.
        assert!(with_timings.to_json().get("performance").is_some());
    }

    /// ...and must change when the physics changes.
    #[test]
    fn content_hash_changes_with_the_results() {
        let mut different = RunArtifact::new("demo").with_seed(7).without_timestamp();
        different.add_contract(&TEST_CONTRACT);
        different.set_parameter("dt", 1e-3);
        different.record(0.0, 0, &sample_observations(10.0));
        different.record(0.1, 100, &sample_observations(99.0)); // different value

        assert_ne!(different.content_hash(), artifact().content_hash());
    }

    #[test]
    fn content_hash_changes_with_the_seed() {
        let mut other = artifact();
        other.seed = Some(8);
        assert_ne!(other.content_hash(), artifact().content_hash());
    }

    #[test]
    fn a_non_finite_value_is_found_and_surfaced() {
        let mut a = artifact();
        a.record(0.2, 200, &sample_observations(f64::NAN));
        let (sample, name) = a.first_non_finite().expect("the NaN must be found");
        assert_eq!(sample.step, 200);
        assert_eq!(name, "test.energy");
        assert!(a.summary().contains("NON-FINITE"), "{}", a.summary());
    }

    #[test]
    fn warnings_travel_with_the_artifact() {
        let mut a = artifact();
        a.warn("timestep exceeded 80% of the stability limit");
        assert_eq!(a.warnings().len(), 1);
        assert!(a.summary().contains("stability limit"));
        assert!(a.to_json().to_compact_string().contains("stability limit"));
    }

    #[test]
    fn artifacts_write_to_disk_and_create_directories() {
        let dir = std::env::temp_dir().join("lattice-artifact-test");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("run.json");

        artifact().write(&path).expect("write should succeed");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"run\": \"demo\""), "{text}");
        assert!(text.ends_with('\n'), "artifacts should end with a newline");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fnv1a_matches_its_published_vectors() {
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn extra_sections_are_merged_at_the_top_level() {
        let mut a = artifact();
        a.set_section("validation", Json::object().set("passed", 12));
        let json = a.to_json();
        assert_eq!(json.get("validation").unwrap().get("passed"), Some(&Json::Int(12)));
    }
}
