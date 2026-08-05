//! 2D rigid-body mechanics.
//!
//! Spec §11.1's MVP row by row: circle, box, convex polygon and segment shapes;
//! translation, rotation, forces, torques and impulses; a broadphase grid, narrowphase
//! contacts, friction and restitution; distance, pin, spring and motor constraints.
//!
//! # What this solver is honest about
//!
//! Contacts are resolved by **sequential impulses** — projected Gauss–Seidel on the
//! velocity constraints, with warm starting. It is the standard interactive method and
//! it is not exact. Three things follow, and the module's [`SolverContract`] says all
//! three out loud rather than leaving them to be discovered:
//!
//! - **It is iterative.** Constraints are satisfied to whatever the iteration count
//!   buys. A tall stack under load will visibly sag at four iterations and not at
//!   twenty. The residual is published every step.
//! - **Restitution moves energy on purpose.** A coefficient below 1 removes kinetic
//!   energy; that is what inelastic means. Energy is not a conserved quantity here and
//!   is not claimed as one.
//! - **Position correction adds energy.** Pushing overlapping bodies apart does work.
//!   The bias is capped and applied to a separate velocity accumulator so it does not
//!   feed back into the momentum, but "does not feed back" is not "is free".
//!
//! Linear and angular momentum *are* conserved exactly, because every impulse is
//! applied equal and opposite to the pair — the same argument that makes the particle
//! module's pair forces exact.
//!
//! [`SolverContract`]: lattice_ir::SolverContract

pub mod broadphase;
pub mod constraint;
pub mod contact;
pub mod domain;
pub mod math;
pub mod narrowphase;
pub mod shape;
pub mod solver;

pub use broadphase::{BroadPhase, CONTACT_MARGIN};
pub use constraint::{Joint, JointSolver, JOINT_SLOP};
pub use domain::RigidDomain;
pub use contact::{BodyPair, Contact, FeatureId, Manifold, ManifoldPoint};
pub use math::{vec2, Rot, Transform, Vec2};
pub use narrowphase::{collide, generate_contacts, LINEAR_SLOP};
pub use solver::{ContactSolver, SolveReport, SolverConfig};
pub use shape::{
    combine_surfaces, Aabb, Collider, ConvexPolygon, MassProperties, Shape, ShapeError,
    MAX_VERTICES,
};
