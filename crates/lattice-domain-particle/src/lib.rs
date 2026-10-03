//! 2D particle dynamics.
//!
//! Spec §11.2: *"Particles are a general computational primitive used for gases,
//! granular matter, molecular dynamics, tracers, emitters, and hybrid methods."* This
//! crate provides that primitive: a [`ParticleDomain`] holds a structure-of-arrays
//! store, a list of [`ForceLaw`]s, an [`Integrator`], and a boundary region, and
//! implements the [`Domain`](lattice_ir::Domain) contract so the runtime can schedule
//! it alongside anything else.
//!
//! # A complete simulation
//!
//! ```
//! use lattice_domain_particle::{
//!     BoundaryBox, HarmonicWell, Integrator, ParticleDomain, ParticleSpec,
//! };
//! use lattice_ir::{Arena, Domain, StepContext};
//!
//! // A unit mass on a spring: omega = sqrt(k/m) = 2 rad/s.
//! let mut domain = ParticleDomain::new("oscillator", 1)
//!     .with_integrator(Integrator::VelocityVerlet)
//!     .with_force(HarmonicWell::new([0.0, 0.0], 4.0));
//! domain.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
//! domain.initialize();
//!
//! let energy_before = domain.total_energy();
//! let period = std::f64::consts::TAU / 2.0;
//! let dt = period / 1000.0;
//!
//! let mut arena = Arena::with_capacity(0);
//! let mut ctx = StepContext::new(&mut arena);
//! for _ in 0..1000 {
//!     domain.advance(dt, &mut ctx);
//! }
//!
//! // One full period later, the particle is back where it started...
//! assert!((domain.store().pos_x()[0] - 1.0).abs() < 1e-4);
//! // ...and a symplectic integrator has held the energy.
//! let drift = (domain.total_energy() - energy_before).abs() / energy_before;
//! assert!(drift < 1e-5, "energy drifted {drift:e}");
//! ```
//!
//! # Choosing an integrator
//!
//! | Scheme | Order | Symplectic | Use |
//! |---|---|---|---|
//! | [`Integrator::ExplicitEuler`] | 1 | no | comparison and teaching only |
//! | [`Integrator::SemiImplicitEuler`] | 1 | yes | interactive scenes, cheap and stable |
//! | [`Integrator::VelocityVerlet`] | 2 | yes | conservative systems, molecular dynamics |
//!
//! Explicit Euler is present because spec §10.2 asks for it *"only for
//! teaching/comparison"* — it makes the case for the other two by failing visibly.
//!
//! # Neighbour search
//!
//! Any force law that reports a [`ForceLaw::cutoff`] triggers construction of a
//! [`NeighborList`]: a [`CellList`] whose cells are at least one cutoff wide, so all
//! interacting partners lie in the 3×3 block around a particle's own cell, optionally
//! cached behind a Verlet skin ([`ParticleDomain::with_skin`]). A cutoff-based force
//! with no region declared is a panic at [`ParticleDomain::initialize`] rather than a
//! silent fallback to O(N²).
//!
//! # Molecular dynamics
//!
//! Spec §12.4's module is the same domain with more attached: [`HarmonicBond`] and
//! [`HarmonicAngle`] act on declared topology, [`SoftRepulsion`] and a force-shifted
//! [`LennardJones`] join the pair laws, a [`Thermostat`] couples the system to a
//! temperature, and [`analysis`] measures temperature, pressure, the radial
//! distribution function and mean squared displacement. Every configuration publishes
//! its own contract, so a thermostatted run says what it gave up to hold its
//! temperature.

pub mod analysis;
mod bonded;
mod domain;
mod forces;
mod image;
mod integrator;
mod neighbors;
mod thermostat;
mod verlet;

pub use bonded::{Angle, Bond, HarmonicAngle, HarmonicBond};
pub use domain::{BoundaryBox, ParticleBoundary, ParticleDomain, RdfRequest};
pub use forces::{
    ForceContext, ForceLaw, HarmonicWell, LennardJones, LinearDrag, SoftRepulsion, Truncation,
    UniformAcceleration,
};
pub use image::MinimumImage;
pub use integrator::{Integrator, LangevinBath, PARTICLE_GRAIN};
pub use neighbors::CellList;
pub use thermostat::{ornstein_uhlenbeck_coefficients, rescale_velocities, Thermostat, BOLTZMANN};
pub use verlet::{Exclusions, NeighborList, VerletList};

// Re-exported for convenience: constructing a domain always needs these.
pub use lattice_ir::{ParticleId, ParticleSpec, ParticleStore};
