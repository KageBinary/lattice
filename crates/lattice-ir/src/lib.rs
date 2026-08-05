//! Simulation IR: the storage layouts, identifiers, and contracts every Lattice
//! solver is built on.
//!
//! Spec §7.2 draws the line this crate exists to enforce:
//!
//! > The runtime should follow a strict separation between immutable compiled model
//! > state and mutable simulation data. […] expensive parsing and indexing happen
//! > once, while the hot loop operates over preallocated low-level arrays.
//!
//! # What lives here
//!
//! | Module | Provides | Spec |
//! |---|---|---|
//! | [`ids`] | Typed identifiers for every index space | §7.1 |
//! | [`model`] | The compiled model and its buffer plan | §7.2, §8.4 |
//! | [`graph`] | The scheduled operation graph | §9.2 |
//! | [`particles`] | Structure-of-arrays particle storage with stable handles | §11.2, §15.2 |
//! | [`grid`] | Uniform grids, halo'd scalar/vector fields, boundary conditions | §11.3 |
//! | [`arena`] | Preallocated per-step scratch space | NFR-001 |
//! | [`contract`] | The [`Domain`] trait and the [`SolverContract`] every solver publishes | §7.3, NFR-003 |
//! | [`diagnostics`] | Conservation drift, coupling ledger, solver residuals | §14.3, §10.4 |
//! | [`rng`] | Reproducible seeded randomness | P6, FR-011 |
//!
//! # What does not live here
//!
//! No equations. This crate holds no physics — it defines how physics is *stored*,
//! *identified*, and *reported on*. Spec P2: *"the runtime unifies state, units,
//! coupling, scheduling, diagnostics, and authoring — not the equations themselves."*
//! Each solver family brings its own mathematics in its own crate.
//!
//! No units, either, in the hot path. A [`lattice_units::Quantity`] belongs to the
//! compiler; by the time data reaches these structures it is plain `f64` in coherent
//! SI. The one place units reappear is [`diagnostics::Invariant::dimension`], which
//! lets a report state what it is reporting.
//!
//! # Example: a domain that does nothing, honestly
//!
//! ```
//! use lattice_ir::{
//!     Arena, Domain, FidelityProfile, Invariant, Observations, Precision,
//!     SolverContract, StableStep, StepContext,
//! };
//!
//! struct Freefall { velocity: f64 }
//!
//! static CONTRACT: SolverContract = SolverContract {
//!     name: "example.freefall",
//!     summary: "a single particle under constant gravity",
//!     governing_equations: &["dv/dt = -g"],
//!     discretization: "none (single degree of freedom)",
//!     integrator: "explicit Euler",
//!     assumptions: &["gravity is uniform", "no drag"],
//!     valid_regime: "illustrative only",
//!     stability: "unconditionally stable; first-order accurate in dt",
//!     conserves: &[],
//!     known_non_conservation: &["energy, since explicit Euler adds energy each step"],
//!     fidelity: FidelityProfile::Interactive,
//!     precisions: &[Precision::Accurate64],
//!     deterministic: true,
//!     differentiable: true,
//!     validation_cases: &["constant acceleration"],
//!     references: &[],
//! };
//!
//! impl Domain for Freefall {
//!     fn name(&self) -> &str { "freefall" }
//!     fn contract(&self) -> &'static SolverContract { &CONTRACT }
//!     fn stable_step(&self) -> StableStep { StableStep::unconditional(0.01) }
//!     fn prepare(&mut self, _ctx: &mut StepContext<'_>) {}
//!     fn advance(&mut self, dt: f64, _ctx: &mut StepContext<'_>) {
//!         self.velocity -= 9.80665 * dt;
//!     }
//!     fn observe(&self, out: &mut Observations) {
//!         out.record_metric("freefall.velocity", self.velocity, "m/s");
//!     }
//! }
//!
//! let mut arena = Arena::with_capacity(0);
//! let mut ctx = StepContext::new(&mut arena);
//! let mut d = Freefall { velocity: 0.0 };
//! for _ in 0..100 { d.advance(0.01, &mut ctx); }
//!
//! let mut obs = Observations::new();
//! d.observe(&mut obs);
//! assert!((obs.value("freefall.velocity").unwrap() + 9.80665).abs() < 1e-9);
//! // The contract is complete enough to ship.
//! assert!(CONTRACT.audit().is_empty());
//! ```

pub mod arena;
pub mod bodies;
pub mod contract;
pub mod diagnostics;
pub mod graph;
pub mod grid;
pub mod ids;
pub mod model;
pub mod particles;
pub mod port;
pub mod render;
pub mod rng;

pub use arena::{Arena, Frame};
pub use bodies::{BodySpec, Motion, RigidBodyStore};
pub use contract::{
    ContractGap, Domain, FidelityProfile, Precision, SolverContract, StabilityReason, StableStep,
    StepContext,
};
pub use diagnostics::{
    ConservationLedger, DriftMonitor, Invariant, Observation, ObservationKind, Observations,
    Reconciliation, ResidualHistory, SolveOutcome, Transfer,
};
pub use graph::{GraphError, Operation, OperationGraph, OperationKind};
pub use grid::{
    Boundary, BoundaryError, BoundarySet, Grid2d, ScalarField, Side, VectorField,
};
pub use model::{
    BufferKind, BufferPlan, BufferSpec, CompiledModel, DomainSpec, ObserverSpec, VisualSpec,
};
pub use ids::{
    BodyId, BufferId, DomainId, FieldId, MaterialId, ObserverId, OperatorId, ParticleId,
    ParticleKind, PortId, ReactionId, ShapeId, SpeciesId,
};
pub use particles::{Dynamics, ForceAccumulation, ParticleSpec, ParticleStore};
pub use port::{PortData, PortDirection, PortShape, PortSpec};
pub use render::{bounds_of, RenderChannel};
pub use rng::{Pcg32, RngSnapshot};
