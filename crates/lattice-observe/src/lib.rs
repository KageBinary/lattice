//! Metrics, timing, JSON export, and run artifacts.
//!
//! Spec §24 lists this crate as *"metrics, probes, plots, exporters"*. It is the
//! layer that turns what a solver knows into something a user, a CI job, or a future
//! reader of a stored result can act on.
//!
//! | Module | Provides | Spec |
//! |---|---|---|
//! | [`json`] | Dependency-free, order-stable JSON | §18.1 |
//! | [`timing`] | Phase profiles, memory reports, throughput | §19.3, §15.1 |
//! | [`artifact`] | The self-describing run artifact | §18.1, FR-014 |
//!
//! # The theme
//!
//! Spec P7 calls visualization *instrumentation*, not decoration, and NFR-007 requires
//! the engine to *"expose numerical instability rather than silently clamp or hide
//! it"*. Both push the same way, and it shows up in small decisions throughout this
//! crate: a `NaN` survives serialization instead of becoming `null`; unattributed time
//! appears as its own line in the profile instead of vanishing; the content hash
//! covers physics but not timings, so it can actually be compared.
//!
//! # Example
//!
//! ```
//! use lattice_observe::{Json, RunArtifact};
//! use lattice_ir::{Invariant, Observations};
//!
//! let mut artifact = RunArtifact::new("oscillator").with_seed(42).without_timestamp();
//! artifact.set_parameter("dt", 1e-3);
//!
//! let mut obs = Observations::new();
//! obs.record_invariant("osc.energy", Invariant::Energy, 2.0);
//! artifact.record(0.0, 0, &obs);
//!
//! // The same physics always hashes the same way, which is what makes a recorded
//! // run comparable to a replayed one.
//! let hash = artifact.content_hash();
//! assert_eq!(hash, artifact.content_hash());
//!
//! let json = artifact.to_json();
//! assert_eq!(json.get("run"), Some(&Json::from("oscillator")));
//! ```

pub mod artifact;
pub mod json;
pub mod timing;

pub use artifact::{contract_to_json, fnv1a_64, seconds, RunArtifact, TimelineSample};
pub use json::Json;
pub use timing::{
    format_bytes, format_count, format_duration, phase, MemoryReport, PhaseStats, Profile,
    Throughput,
};
