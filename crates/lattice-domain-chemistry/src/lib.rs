//! Chemistry: reaction networks, kinetics, and reaction-diffusion.
//!
//! Spec §12. The layers this module covers are the first two of §12.1's five:
//!
//! | Layer | What it models | What it does not claim |
//! |---|---|---|
//! | Reaction network | species concentrations and kinetic laws | molecular geometry or electronic mechanism |
//! | Reaction-diffusion | species transport plus local reactions | atomistic solvent or detailed turbulence |
//!
//! The three below them — molecular dynamics, reactive empirical potentials, and
//! external quantum chemistry — are M5 and M8, and nothing here pretends otherwise.
//!
//! # What this module is honest about
//!
//! **Balance is checked, not assumed.** `H2 + O2 -> H2O` integrates perfectly happily
//! and is missing an oxygen. When every species states its composition, every reaction
//! is checked for atom and charge balance and an unbalanced one is refused. When a
//! species does not state its composition the check reports *unknown*, which is not the
//! same as passing. See [`network::Balance`].
//!
//! **Concentrations are per unit area.** `mol/m²`. This is a 2D engine, and a
//! concentration per unit volume would need a thickness nobody declared — the same
//! argument that makes rigid-body density areal. A second-order rate constant is
//! therefore in `m²/(mol·s)`, which is unfamiliar and correct.
//!
//! **A rate constant's units depend on the reaction's order.** Writing a second-order
//! constant where a first-order one belongs is wrong by a factor with the dimensions of
//! a concentration, and the number looks plausible either way. See
//! [`rate::RateLaw::si_unit_for_order`].

pub mod elements;
pub mod kinetics;
pub mod network;
pub mod rate;
pub mod spatial;
pub mod species;

pub use elements::{Category, Element};
pub use kinetics::{Integrator, Kinetics, KineticsReport, DEFAULT_ACCURACY, MAX_SUBSTEPS};
pub use network::{Balance, Reaction, ReactionNetwork, Term};
pub use rate::{mass_action_rate, RateLaw, Temperature, GAS_CONSTANT};
pub use spatial::ReactingMixture;
pub use species::{atomic_mass, Composition, FormulaError, Phase, Species};
