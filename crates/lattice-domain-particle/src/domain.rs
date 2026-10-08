//! The particle domain: storage, forces, integrator, boundaries, thermostat,
//! neighbour list, analysis, and diagnostics assembled into something the runtime
//! can schedule.

use lattice_ir::{
    Domain, FidelityProfile, Invariant, Observations, ParticleSpec, ParticleStore, Precision,
    SolverContract, StableStep, StepContext,
};

use crate::analysis::{self, RadialDistribution, RdfAccumulator};
use crate::forces::{ForceContext, ForceLaw};
use crate::image::MinimumImage;
use crate::integrator::{Integrator, LangevinBath};
use crate::neighbors::CellList;
use crate::thermostat::{rescale_velocities, Thermostat};
use crate::verlet::{Exclusions, NeighborList};

/// What happens when a particle reaches the edge of the simulation region.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ParticleBoundary {
    /// Wraps to the opposite edge. Pair forces use the minimum-image convention.
    Periodic,
    /// Position mirrors and the normal velocity component flips. Energy-preserving.
    Reflective,
    /// Nothing happens; particles are free to leave.
    #[default]
    Open,
}

/// The rectangular simulation region and its per-axis boundary behaviour.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoundaryBox {
    /// Lower-left corner, m.
    pub min: [f64; 2],
    /// Region size, m.
    pub size: [f64; 2],
    /// Behaviour on the x edges.
    pub x: ParticleBoundary,
    /// Behaviour on the y edges.
    pub y: ParticleBoundary,
}

/// Bound on reflection passes before a runaway particle is clamped instead.
///
/// A particle far outside the box — which only happens after the simulation has
/// already diverged — would otherwise ping-pong here indefinitely. Clamping and
/// letting the NaN/divergence monitors report the real problem is better than
/// hanging (NFR-007).
const MAX_REFLECTIONS: u32 = 64;

impl BoundaryBox {
    /// A region with the same behaviour on both axes.
    pub fn new(min: [f64; 2], size: [f64; 2], behaviour: ParticleBoundary) -> Self {
        Self { min, size, x: behaviour, y: behaviour }
    }

    /// A fully periodic box.
    pub fn periodic(min: [f64; 2], size: [f64; 2]) -> Self {
        Self::new(min, size, ParticleBoundary::Periodic)
    }

    /// A reflecting box.
    pub fn reflective(min: [f64; 2], size: [f64; 2]) -> Self {
        Self::new(min, size, ParticleBoundary::Reflective)
    }

    /// Upper-right corner, m.
    pub fn max(&self) -> [f64; 2] {
        [self.min[0] + self.size[0], self.min[1] + self.size[1]]
    }

    /// Which axes wrap, in the form the cell list wants.
    pub fn periodic_axes(&self) -> [bool; 2] {
        [self.x == ParticleBoundary::Periodic, self.y == ParticleBoundary::Periodic]
    }

    /// The minimum-image rule for this box.
    pub fn image(&self) -> MinimumImage {
        MinimumImage::new(self.size, self.periodic_axes())
    }

    /// Enforce the boundary on every particle.
    pub fn apply(&self, store: &mut ParticleStore) {
        let d = store.dynamics();
        let max = self.max();
        for i in 0..d.len() {
            apply_axis(self.x, &mut d.pos_x[i], &mut d.vel_x[i], self.min[0], max[0], self.size[0]);
            apply_axis(self.y, &mut d.pos_y[i], &mut d.vel_y[i], self.min[1], max[1], self.size[1]);
        }
    }

    /// Enforce the boundary, recording how far each periodic wrap moved a particle.
    ///
    /// `shift` accumulates `position_before − position_after` per particle, which is
    /// a whole number of box lengths on a periodic axis and zero otherwise. The
    /// unwrapped trajectory a mean squared displacement needs is `position + shift`.
    pub fn apply_tracking(&self, store: &mut ParticleStore, shift_x: &mut [f64], shift_y: &mut [f64]) {
        let d = store.dynamics();
        let max = self.max();
        for i in 0..d.len() {
            let before = (d.pos_x[i], d.pos_y[i]);
            apply_axis(self.x, &mut d.pos_x[i], &mut d.vel_x[i], self.min[0], max[0], self.size[0]);
            apply_axis(self.y, &mut d.pos_y[i], &mut d.vel_y[i], self.min[1], max[1], self.size[1]);
            if self.x == ParticleBoundary::Periodic {
                shift_x[i] += before.0 - d.pos_x[i];
            }
            if self.y == ParticleBoundary::Periodic {
                shift_y[i] += before.1 - d.pos_y[i];
            }
        }
    }

    /// True when no particle can leave the region.
    pub fn is_closed(&self) -> bool {
        self.x != ParticleBoundary::Open && self.y != ParticleBoundary::Open
    }

    /// True when both axes wrap — the geometry the virial pressure is defined for.
    pub fn is_fully_periodic(&self) -> bool {
        self.x == ParticleBoundary::Periodic && self.y == ParticleBoundary::Periodic
    }

    /// Area of the region, m².
    pub fn area(&self) -> f64 {
        self.size[0] * self.size[1]
    }
}

fn apply_axis(
    behaviour: ParticleBoundary,
    position: &mut f64,
    velocity: &mut f64,
    lo: f64,
    hi: f64,
    span: f64,
) {
    match behaviour {
        ParticleBoundary::Open => {}
        ParticleBoundary::Periodic => {
            if !position.is_finite() {
                return;
            }
            *position = lo + (*position - lo).rem_euclid(span);
        }
        ParticleBoundary::Reflective => {
            if !position.is_finite() {
                return;
            }
            let mut passes = 0;
            while (*position < lo || *position > hi) && passes < MAX_REFLECTIONS {
                if *position < lo {
                    *position = 2.0 * lo - *position;
                    *velocity = -*velocity;
                }
                if *position > hi {
                    *position = 2.0 * hi - *position;
                    *velocity = -*velocity;
                }
                passes += 1;
            }
            if passes == MAX_REFLECTIONS {
                *position = position.clamp(lo, hi);
            }
        }
    }
}

/// How often, and how finely, a radial distribution function is sampled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RdfRequest {
    /// Histogram bins.
    pub bins: usize,
    /// Outer radius, m. At most half the box on a periodic axis.
    pub range: f64,
    /// Sample every this many steps.
    pub every: u64,
    /// Steps to discard before the first sample, so a run that starts from a lattice
    /// does not count the lattice. Zero samples from the start.
    pub after: u64,
}

/// A 2D particle system.
///
/// Build one with [`ParticleDomain::new`] and the `with_*` methods, then call
/// [`ParticleDomain::initialize`] once before stepping so the force accumulators hold
/// the forces for the initial configuration — the invariant every integrator assumes.
#[derive(Debug)]
pub struct ParticleDomain {
    name: String,
    store: ParticleStore,
    forces: Vec<Box<dyn ForceLaw>>,
    integrator: Integrator,
    neighbors: Option<NeighborList>,
    bounds: Option<BoundaryBox>,
    skin: Option<f64>,
    thermostat: Option<Thermostat>,
    /// Built on first initialization and kept thereafter, so re-initializing a running
    /// domain continues the random stream rather than restarting it.
    bath: Option<LangevinBath>,
    seed: u64,
    preferred_dt: f64,
    steps: u64,
    initialized: bool,
    /// Accumulated periodic wrap offsets per slot; see [`BoundaryBox::apply_tracking`].
    shift_x: Vec<f64>,
    shift_y: Vec<f64>,
    /// Positions at the last initialization, the origin of the displacement.
    reference_x: Vec<f64>,
    reference_y: Vec<f64>,
    rdf_request: Option<RdfRequest>,
    rdf: Option<RdfAccumulator>,
    /// Slot pairs the bonded laws bind, gathered at initialization for drawing.
    bonded: Vec<[u32; 2]>,
}

impl ParticleDomain {
    /// A domain named `name` with room for `capacity` particles.
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            store: ParticleStore::with_capacity(capacity),
            forces: Vec::new(),
            integrator: Integrator::default(),
            neighbors: None,
            bounds: None,
            skin: None,
            thermostat: None,
            bath: None,
            seed: 0,
            preferred_dt: 1e-3,
            steps: 0,
            initialized: false,
            shift_x: vec![0.0; capacity],
            shift_y: vec![0.0; capacity],
            reference_x: vec![0.0; capacity],
            reference_y: vec![0.0; capacity],
            rdf_request: None,
            rdf: None,
            bonded: Vec::new(),
        }
    }

    /// Choose the integrator.
    pub fn with_integrator(mut self, integrator: Integrator) -> Self {
        self.integrator = integrator;
        self
    }

    /// Set the timestep this domain asks for when nothing else constrains it.
    pub fn with_preferred_step(mut self, dt: f64) -> Self {
        self.preferred_dt = dt;
        self
    }

    /// Set the simulation region and boundary behaviour.
    pub fn with_bounds(mut self, bounds: BoundaryBox) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Cache pairs in a Verlet list with the given skin, metres.
    ///
    /// Without a skin the cell list is rebuilt on every step. See
    /// [`crate::verlet`] for what the skin buys and what it changes.
    ///
    /// # Panics
    ///
    /// On a non-positive or non-finite skin.
    pub fn with_skin(mut self, skin: f64) -> Self {
        assert!(skin > 0.0 && skin.is_finite(), "a Verlet skin must be positive and finite, got {skin}");
        self.skin = Some(skin);
        self
    }

    /// Couple the system to a thermostat.
    ///
    /// # Panics
    ///
    /// On unusable thermostat parameters. A Langevin bath additionally requires the
    /// velocity Verlet integrator, checked at [`ParticleDomain::initialize`].
    pub fn with_thermostat(mut self, thermostat: Thermostat) -> Self {
        thermostat.validate();
        self.thermostat = Some(thermostat);
        self
    }

    /// Seed the random kicks of a Langevin bath.
    ///
    /// The bath draws on its own PCG stream, so this may be the same seed the initial
    /// velocities were drawn from without reusing their deviates.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Accumulate a radial distribution function during the run.
    ///
    /// Requires a region; see [`RdfAccumulator::new`] for the range limit.
    pub fn with_rdf(mut self, request: RdfRequest) -> Self {
        assert!(request.every > 0, "an RDF must be sampled at least every step");
        self.rdf_request = Some(request);
        self
    }

    /// Add a force law.
    pub fn with_force(mut self, law: impl ForceLaw + 'static) -> Self {
        self.forces.push(Box::new(law));
        self
    }

    /// Add a particle, returning its handle.
    pub fn spawn(&mut self, spec: ParticleSpec) -> Option<lattice_ir::ParticleId> {
        self.initialized = false;
        self.store.spawn(spec)
    }

    /// Build the neighbour list (if any force needs one), resolve topology, and
    /// evaluate initial forces.
    ///
    /// Also resets the displacement origin used by the mean squared displacement.
    ///
    /// # Panics
    ///
    /// If a force law needs a neighbour list but no region was set with
    /// [`ParticleDomain::with_bounds`] — a cutoff-based force with no region has no
    /// well-defined binning, and silently falling back to O(N²) would turn a
    /// configuration mistake into a mysterious performance cliff. Also if a Langevin
    /// thermostat is paired with an integrator other than velocity Verlet, if a bond
    /// names a particle that is not alive, or if an RDF was requested without a region.
    pub fn initialize(&mut self) {
        for law in &mut self.forces {
            law.prepare(&self.store);
        }

        // Topology: what to draw, and what pair laws must skip.
        self.bonded.clear();
        let mut excluded: Vec<[u32; 2]> = Vec::new();
        for law in &self.forces {
            self.bonded.extend_from_slice(law.bonded_pairs());
            if law.excludes_pair_forces() {
                excluded.extend_from_slice(law.bonded_pairs());
            }
        }

        let cutoff = self
            .forces
            .iter()
            .filter_map(|f| f.cutoff())
            .fold(f64::NEG_INFINITY, f64::max);

        if cutoff.is_finite() && cutoff > 0.0 {
            let bounds = self.bounds.unwrap_or_else(|| {
                panic!(
                    "domain `{}` has a force with a {cutoff} m cutoff but no region; \
                     call with_bounds() so the neighbour list can be built",
                    self.name
                )
            });
            let exclusions = if excluded.is_empty() {
                Exclusions::none()
            } else {
                Exclusions::from_pairs(self.store.len(), &excluded)
            };
            // Rebuilt only when its geometry changed: a running domain that is
            // re-initialized after an edit keeps its list and rebuild statistics.
            let stale = self.neighbors.as_ref().is_none_or(|list| {
                list.cutoff() != cutoff || list.skin() != self.skin || list.exclusions() != &exclusions
            });
            if stale {
                self.neighbors = Some(NeighborList::new(
                    bounds.min,
                    bounds.size,
                    bounds.periodic_axes(),
                    cutoff,
                    self.skin,
                    self.store.capacity(),
                    exclusions,
                ));
            }
        } else {
            self.neighbors = None;
        }

        if let Some(Thermostat::Langevin { temperature, friction }) = self.thermostat {
            assert!(
                matches!(self.integrator, Integrator::VelocityVerlet),
                "domain `{}`: a Langevin thermostat needs the velocity_verlet integrator, not {}",
                self.name,
                self.integrator.name()
            );
            if self.bath.is_none() {
                self.bath = Some(LangevinBath::new(temperature, friction, self.seed));
            }
        }

        if let Some(request) = self.rdf_request
            && self.rdf.is_none()
        {
            let bounds = self.bounds.unwrap_or_else(|| {
                panic!("domain `{}` asks for a radial distribution but has no region", self.name)
            });
            self.rdf = Some(RdfAccumulator::new(
                request.bins,
                request.range,
                bounds.min,
                bounds.size,
                bounds.periodic_axes(),
                self.store.capacity(),
            ));
        }

        self.refresh_forces();

        let n = self.store.len();
        self.reference_x[..n].copy_from_slice(self.store.pos_x());
        self.reference_y[..n].copy_from_slice(self.store.pos_y());
        self.shift_x[..n].fill(0.0);
        self.shift_y[..n].fill(0.0);
        self.initialized = true;
    }

    /// Recompute forces for the current configuration.
    fn refresh_forces(&mut self) {
        let Self { store, forces, neighbors, bounds, shift_x, shift_y, .. } = self;
        Self::evaluate_forces(store, forces, neighbors.as_mut(), bounds.as_ref(), shift_x, shift_y);
    }

    /// Wrap, re-bin, and accumulate every law — the closure every integrator step
    /// calls after moving the particles.
    fn evaluate_forces(
        store: &mut ParticleStore,
        forces: &[Box<dyn ForceLaw>],
        neighbors: Option<&mut NeighborList>,
        bounds: Option<&BoundaryBox>,
        shift_x: &mut [f64],
        shift_y: &mut [f64],
    ) {
        // Positions have just changed: wrap or reflect before binning, so the
        // neighbour list and the minimum-image convention agree on where particles are.
        if let Some(b) = bounds {
            b.apply_tracking(store, shift_x, shift_y);
        }
        let image = bounds.map_or(MinimumImage::open(), BoundaryBox::image);
        let neighbors = neighbors.map(|list| {
            list.update(store);
            &*list
        });
        store.clear_forces();
        let ctx = ForceContext { neighbors, image };
        let mut view = store.force_accumulation();
        for law in forces {
            law.accumulate(&mut view, &ctx);
        }
    }

    /// The force context a law would be evaluated with right now.
    fn force_context(&self) -> ForceContext<'_> {
        ForceContext {
            neighbors: self.neighbors.as_ref(),
            image: self.bounds.map_or(MinimumImage::open(), |b| b.image()),
        }
    }

    /// The particle store.
    pub fn store(&self) -> &ParticleStore {
        &self.store
    }

    /// The particle store, mutably.
    ///
    /// Changing positions or velocities invalidates the force invariant, so call
    /// [`ParticleDomain::initialize`] afterwards.
    pub fn store_mut(&mut self) -> &mut ParticleStore {
        self.initialized = false;
        &mut self.store
    }

    /// The active integrator.
    pub fn integrator(&self) -> Integrator {
        self.integrator
    }

    /// The thermostat, if one is coupled.
    pub fn thermostat(&self) -> Option<Thermostat> {
        self.thermostat
    }

    /// The Verlet skin, if pairs are cached.
    pub fn skin(&self) -> Option<f64> {
        self.skin
    }

    /// The region, if one was declared.
    pub fn bounds(&self) -> Option<BoundaryBox> {
        self.bounds
    }

    /// The neighbour list, if one is in use.
    pub fn neighbors(&self) -> Option<&NeighborList> {
        self.neighbors.as_ref()
    }

    /// The cell list underneath the neighbour list, if one is in use.
    ///
    /// With a skin it is sized for `cutoff + skin`, so count pairs through
    /// [`ParticleDomain::neighbors`] rather than here.
    pub fn cells(&self) -> Option<&CellList> {
        self.neighbors.as_ref().map(NeighborList::cells)
    }

    /// Slot pairs joined by bonded laws, valid after initialization.
    pub fn bonded_pairs(&self) -> &[[u32; 2]] {
        &self.bonded
    }

    /// The accumulated radial distribution function, if one was requested and at
    /// least one frame has been sampled.
    pub fn radial_distribution(&self) -> Option<RadialDistribution> {
        self.rdf.as_ref().filter(|r| r.frames() > 0).map(RdfAccumulator::result)
    }

    /// Names of the active force laws, for the model report.
    pub fn force_names(&self) -> Vec<&'static str> {
        self.forces.iter().map(|f| f.name()).collect()
    }

    /// Steps taken since construction.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Approximate bytes of runtime state, for the memory report (§19.3).
    ///
    /// Counts the preallocated particle arrays, the neighbour list, the displacement
    /// tracking, and any RDF histogram. Excludes the force laws themselves, which are
    /// a handful of scalars each plus their topology.
    pub fn memory_bytes(&self) -> usize {
        let particles = self.store.capacity() * ParticleStore::BYTES_PER_PARTICLE;
        let neighbors = self.neighbors.as_ref().map_or(0, NeighborList::memory_bytes);
        let tracking = 4 * self.store.capacity() * size_of::<f64>();
        let rdf = self.rdf.as_ref().map_or(0, RdfAccumulator::memory_bytes);
        particles + neighbors + tracking + rdf
    }

    /// Total kinetic energy, J.
    ///
    /// Pinned (infinite-mass) particles are skipped: they never move, and `½·∞·0`
    /// would produce a NaN that poisons the whole sum.
    pub fn kinetic_energy(&self) -> f64 {
        let (vx, vy, m) = (self.store.vel_x(), self.store.vel_y(), self.store.mass());
        let mut total = 0.0;
        for i in 0..vx.len() {
            if m[i].is_finite() {
                total += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i]);
            }
        }
        total
    }

    /// Total potential energy from all conservative force laws, J.
    pub fn potential_energy(&self) -> f64 {
        let ctx = self.force_context();
        self.forces
            .iter()
            .filter(|law| law.is_conservative())
            .map(|law| law.potential_energy(&self.store, &ctx))
            .sum()
    }

    /// Kinetic plus potential energy, J.
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// The pair virial `Σ (xᵢ − xⱼ)·Fᵢⱼ` over every law, J.
    pub fn virial(&self) -> f64 {
        let ctx = self.force_context();
        self.forces.iter().map(|law| law.virial(&self.store, &ctx)).sum()
    }

    /// The virial pressure, N/m — defined for a fully periodic box only.
    ///
    /// `P·A = K + ½·W`. With walls the wall forces would enter the virial and are not
    /// accounted, so no number is reported rather than a wrong one.
    pub fn pressure(&self) -> Option<f64> {
        let bounds = self.bounds.filter(BoundaryBox::is_fully_periodic)?;
        Some(analysis::pressure(self.kinetic_energy(), self.virial(), bounds.area()))
    }

    /// Kinetic temperature, K, from the thermal velocities over `2N − 2` degrees of
    /// freedom. `None` for fewer than two mobile particles.
    pub fn temperature(&self) -> Option<f64> {
        analysis::temperature(&self.store)
    }

    /// Mean squared displacement since the last initialization, m², through any
    /// periodic wraps.
    pub fn mean_squared_displacement(&self) -> f64 {
        let n = self.store.len();
        let (xs, ys) = (self.store.pos_x(), self.store.pos_y());
        let mut sum = 0.0;
        for i in 0..n {
            let dx = xs[i] + self.shift_x[i] - self.reference_x[i];
            let dy = ys[i] + self.shift_y[i] - self.reference_y[i];
            sum += dx * dx + dy * dy;
        }
        if n == 0 { 0.0 } else { sum / n as f64 }
    }

    /// Total linear momentum, kg·m/s.
    pub fn momentum(&self) -> [f64; 2] {
        let (vx, vy, m) = (self.store.vel_x(), self.store.vel_y(), self.store.mass());
        let (mut px, mut py) = (0.0, 0.0);
        for i in 0..vx.len() {
            if m[i].is_finite() {
                px += m[i] * vx[i];
                py += m[i] * vy[i];
            }
        }
        [px, py]
    }

    /// The sum of the individual momentum magnitudes, kg·m/s.
    ///
    /// The scale against which a *net* momentum of zero should be judged: a residual
    /// of 1e-13 means nothing on its own, and everything when the parts summing to it
    /// are of order 100.
    pub fn momentum_scale(&self) -> f64 {
        let (vx, vy, m) = (self.store.vel_x(), self.store.vel_y(), self.store.mass());
        (0..vx.len())
            .filter(|&i| m[i].is_finite())
            .map(|i| m[i] * (vx[i] * vx[i] + vy[i] * vy[i]).sqrt())
            .sum()
    }

    /// Fastest particle speed, m/s.
    pub fn max_speed(&self) -> f64 {
        let (vx, vy) = (self.store.vel_x(), self.store.vel_y());
        (0..vx.len()).fold(0.0f64, |acc, i| acc.max((vx[i] * vx[i] + vy[i] * vy[i]).sqrt()))
    }

    /// True when every force law present derives from a potential and no thermostat
    /// exchanges energy with a bath.
    ///
    /// Total energy is only expected to be conserved when this holds *and* the
    /// boundary is closed. The validation suite checks this before asserting on drift,
    /// so that adding drag to a scene does not read as a broken integrator.
    pub fn is_energy_conserving(&self) -> bool {
        self.forces.iter().all(|f| f.is_conservative())
            && self.thermostat.is_none()
            && self.bounds.is_none_or(|b| b.is_closed())
    }

    /// True when nothing external acts on the total momentum.
    ///
    /// Pair and bonded laws conserve it by construction; a Langevin bath kicks every
    /// particle independently and does not.
    pub fn is_momentum_conserving(&self) -> bool {
        !matches!(self.thermostat, Some(Thermostat::Langevin { .. }))
    }
}

/// Which thermostat, for choosing a contract.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Coupling {
    None,
    Langevin,
    VelocityRescale,
}

/// Builds the contract for one integrator and thermostat choice.
///
/// A single static contract cannot be honest here: explicit Euler and velocity Verlet
/// make opposite claims about energy, and a thermostatted run makes a different claim
/// again. Each configuration therefore gets its own, and [`Domain::contract`] returns
/// the one that matches.
const fn contract_for(integrator: Integrator, coupling: Coupling) -> SolverContract {
    SolverContract {
        name: match (integrator, coupling) {
            (Integrator::ExplicitEuler, _) => "particles2d[explicit_euler]",
            (Integrator::SemiImplicitEuler, Coupling::None) => "particles2d[semi_implicit_euler]",
            (Integrator::SemiImplicitEuler, _) => "particles2d[semi_implicit_euler+velocity_rescale]",
            (Integrator::VelocityVerlet, Coupling::None) => "particles2d[velocity_verlet]",
            (Integrator::VelocityVerlet, Coupling::Langevin) => "particles2d[velocity_verlet+langevin]",
            (Integrator::VelocityVerlet, Coupling::VelocityRescale) => {
                "particles2d[velocity_verlet+velocity_rescale]"
            }
        },
        summary: match coupling {
            Coupling::None => "2D point particles advanced under a sum of force laws",
            Coupling::Langevin => {
                "2D point particles under force laws, coupled to a heat bath by Langevin dynamics"
            }
            Coupling::VelocityRescale => {
                "2D point particles under force laws, held near a temperature by velocity rescaling"
            }
        },
        governing_equations: match coupling {
            Coupling::None => &["dx/dt = v", "m dv/dt = sum_k F_k(x, v)"],
            Coupling::Langevin => &[
                "dx/dt = v",
                "m dv = sum_k F_k(x, v) dt - gamma m v dt + sqrt(2 gamma m k_B T) dW",
            ],
            Coupling::VelocityRescale => &[
                "dx/dt = v",
                "m dv/dt = sum_k F_k(x, v)",
                "after each step: v <- v_cm + lambda (v - v_cm), lambda^2 = 1 + (dt/tau)(T0/T - 1)",
            ],
        },
        discretization: "none — particles are discrete degrees of freedom; local pair interactions \
                         are truncated at a cutoff and found through a uniform cell list, \
                         optionally cached behind a Verlet skin; bonds and angles act on declared \
                         topology under the minimum-image convention",
        integrator: match (integrator, coupling) {
            (Integrator::ExplicitEuler, _) => "explicit (forward) Euler, 1st order, not symplectic",
            (Integrator::SemiImplicitEuler, Coupling::None) => {
                "semi-implicit (Euler-Cromer) Euler, 1st order, symplectic"
            }
            (Integrator::SemiImplicitEuler, _) => {
                "semi-implicit (Euler-Cromer) Euler, 1st order, followed by Berendsen velocity rescaling"
            }
            (Integrator::VelocityVerlet, Coupling::None) => {
                "velocity Verlet, 2nd order, symplectic and time-reversible"
            }
            (Integrator::VelocityVerlet, Coupling::Langevin) => {
                "BAOAB splitting (Leimkuhler-Matthews): half kick, half drift, exact \
                 Ornstein-Uhlenbeck velocity update, half drift, half kick; 2nd order weak"
            }
            (Integrator::VelocityVerlet, Coupling::VelocityRescale) => {
                "velocity Verlet, 2nd order, followed by Berendsen velocity rescaling toward T0 \
                 with relaxation time tau"
            }
        },
        assumptions: &[
            "particles are point masses; radius affects only collision queries, not dynamics",
            "pair interactions vanish beyond their declared cutoff",
            "periodic axes use the minimum-image convention, which requires the box to be \
             at least twice the interaction cutoff on that axis",
            "a Verlet skin caches pairs within cutoff + skin and rebuilds when any particle has \
             moved half the skin, so no pair is ever missed",
            "bonded particles are excluded from pair laws unless the bond keeps them",
            "velocity-dependent forces under velocity Verlet are evaluated at the half-step \
             velocity, which is first-order accurate for those terms only",
            "temperature is defined from velocities relative to the centre of mass over 2N-2 \
             degrees of freedom",
            "charges interact by the three-dimensional Coulomb law k q_i q_j / r confined to \
             the plane, not by two-dimensional electrostatics, whose potential is logarithmic",
            "Coulomb interactions are the damped shifted force sum: the real-space Ewald term \
             erfc(alpha r)/r shifted to zero in energy and force at the cutoff, with no \
             reciprocal-space part; this assumes the system is locally neutral on the scale of \
             the cutoff",
        ],
        valid_regime: match (integrator, coupling) {
            (Integrator::ExplicitEuler, _) => {
                "comparison and teaching only; unsuitable for any quantitative result"
            }
            (_, Coupling::None) => "non-relativistic classical dynamics with resolved force timescales",
            (_, Coupling::Langevin) => {
                "canonical (NVT) sampling of classical dynamics; the friction must be small \
                 against the fastest force timescale for the dynamics themselves to be physical"
            }
            (_, Coupling::VelocityRescale) => {
                "equilibration and temperature control; not a canonical ensemble, since the \
                 rescaling suppresses kinetic-energy fluctuations"
            }
        },
        stability: match integrator {
            Integrator::ExplicitEuler => {
                "unstable for oscillatory systems at every timestep: the amplitude grows \
                 by a factor sqrt(1 + (omega*dt)^2) per step"
            }
            _ => {
                "dt < 2/omega_max, where omega_max is the highest vibrational frequency of any \
                 pair, bond or angle at its reduced mass; accuracy typically needs 20-50 steps \
                 per period of the fastest mode"
            }
        },
        conserves: match (integrator, coupling) {
            (Integrator::ExplicitEuler, _) => &[Invariant::MomentumX, Invariant::MomentumY],
            (_, Coupling::None) => &[Invariant::Energy, Invariant::MomentumX, Invariant::MomentumY],
            (_, Coupling::Langevin) => &[],
            (_, Coupling::VelocityRescale) => &[Invariant::MomentumX, Invariant::MomentumY],
        },
        known_non_conservation: match (integrator, coupling) {
            (Integrator::ExplicitEuler, _) => &[
                "energy: not symplectic, so error accumulates secularly rather than \
                 oscillating — energy grows without bound in any oscillatory system",
                "angular momentum: not preserved by the discrete update",
            ],
            (_, Coupling::None) => &[
                "energy: symplectic schemes bound the error but do not eliminate it; \
                 expect a bounded oscillation of order dt^p and no secular drift",
                "energy at a pair-force cutoff: the energy-shifted Lennard-Jones potential \
                 is continuous in energy but not in force, so each crossing of the cutoff \
                 radius injects a small impulse; the force-shifted form removes it",
                "energy under non-conservative laws (drag) and non-closed boundaries \
                 (open edges), both of which are intentional and reported separately",
                "angular momentum under a non-central force law or a non-periodic box",
            ],
            (_, Coupling::Langevin) => &[
                "energy: exchanged with the bath by design; the total is not an invariant \
                 and its drift is not an error",
                "momentum: every particle is kicked independently, so the centre of mass \
                 performs a random walk; net momentum is not conserved",
                "angular momentum, for the same reason",
                "configurational sampling error of order dt^2 in the presence of forces",
            ],
            (_, Coupling::VelocityRescale) => &[
                "energy: removed or added by the rescaling by design; the total is not an \
                 invariant and its drift is not an error",
                "kinetic-energy fluctuations: suppressed, so the ensemble is not canonical",
                "angular momentum: scaled along with the thermal velocities",
            ],
        },
        fidelity: match (integrator, coupling) {
            (Integrator::ExplicitEuler, _) => FidelityProfile::Interactive,
            (_, Coupling::None) => FidelityProfile::Engineering2d,
            _ => FidelityProfile::MolecularKinetic,
        },
        precisions: &[Precision::Accurate64, Precision::Deterministic64],
        deterministic: true,
        differentiable: false,
        validation_cases: match coupling {
            Coupling::None => &[
                "free fall under constant acceleration",
                "harmonic oscillator: period, amplitude, and energy drift",
                "pairwise momentum conservation (Newton's third law)",
                "Lennard-Jones energy conservation in a periodic box",
                "observed convergence order matches the declared order",
                "neighbour-list consistency: a Verlet skin misses no pair",
                "harmonic bond vibration period at the reduced mass",
                "bonded-chain energy conservation with bonds and angles",
                "force-shifted Lennard-Jones energy error converges at second order",
                "damped shifted force Coulomb recovers a 2D rock-salt crystal's Madelung energy",
                "an ionic melt's energy error converges at second order under Coulomb",
            ],
            Coupling::Langevin => &[
                "free particles equilibrate to the bath temperature",
                "free-particle diffusion follows the Einstein relation D = k_B T / (m gamma)",
                "a Lennard-Jones liquid's radial distribution: an empty core and a first shell at the pair minimum",
            ],
            Coupling::VelocityRescale => &[
                "an ideal gas relaxes toward T0 by exactly the Berendsen recurrence",
                "momentum is preserved by the rescaling",
            ],
        },
        references: &[
            "Verlet, L. (1967). Computer experiments on classical fluids. Phys. Rev. 159, 98.",
            "Hairer, Lubich & Wanner (2006). Geometric Numerical Integration, 2nd ed.",
            "Allen & Tildesley (2017). Computer Simulation of Liquids, 2nd ed., ch. 3-5.",
            "Leimkuhler & Matthews (2013). Rational construction of stochastic numerical \
             methods for molecular sampling. Appl. Math. Res. Express 2013, 34.",
            "Berendsen et al. (1984). Molecular dynamics with coupling to an external bath. \
             J. Chem. Phys. 81, 3684.",
            "Fennell & Gezelter (2006). Is the Ewald summation still necessary? Pairwise \
             alternatives to the accepted standard for long-range electrostatics. J. Chem. \
             Phys. 124, 234104.",
            "Wolf, Keblinski, Phillpot & Eggebrecht (1999). Exact method for the simulation of \
             Coulombic systems by spherically truncated, pairwise r^-1 summation. J. Chem. \
             Phys. 110, 8254.",
        ],
    }
}

static EXPLICIT_EULER_CONTRACT: SolverContract = contract_for(Integrator::ExplicitEuler, Coupling::None);
static SEMI_IMPLICIT_CONTRACT: SolverContract = contract_for(Integrator::SemiImplicitEuler, Coupling::None);
static SEMI_IMPLICIT_RESCALE_CONTRACT: SolverContract =
    contract_for(Integrator::SemiImplicitEuler, Coupling::VelocityRescale);
static VELOCITY_VERLET_CONTRACT: SolverContract = contract_for(Integrator::VelocityVerlet, Coupling::None);
static LANGEVIN_CONTRACT: SolverContract = contract_for(Integrator::VelocityVerlet, Coupling::Langevin);
static RESCALE_CONTRACT: SolverContract =
    contract_for(Integrator::VelocityVerlet, Coupling::VelocityRescale);

impl ParticleDomain {
    /// The contract for an integrator and thermostat pairing, without a domain.
    ///
    /// Used by `lattice inspect contracts` to list every configuration that ships.
    pub fn contract_for(integrator: Integrator, thermostat: Option<Thermostat>) -> &'static SolverContract {
        match (integrator, thermostat) {
            (Integrator::ExplicitEuler, _) => &EXPLICIT_EULER_CONTRACT,
            (Integrator::SemiImplicitEuler, None | Some(Thermostat::Langevin { .. })) => &SEMI_IMPLICIT_CONTRACT,
            (Integrator::SemiImplicitEuler, Some(Thermostat::VelocityRescale { .. })) => {
                &SEMI_IMPLICIT_RESCALE_CONTRACT
            }
            (Integrator::VelocityVerlet, None) => &VELOCITY_VERLET_CONTRACT,
            (Integrator::VelocityVerlet, Some(Thermostat::Langevin { .. })) => &LANGEVIN_CONTRACT,
            (Integrator::VelocityVerlet, Some(Thermostat::VelocityRescale { .. })) => &RESCALE_CONTRACT,
        }
    }
}

impl Domain for ParticleDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn contract(&self) -> &'static SolverContract {
        Self::contract_for(self.integrator, self.thermostat)
    }

    fn stable_step(&self) -> StableStep {
        let mut limit = StableStep::unconditional(self.preferred_dt);
        for law in &self.forces {
            if let Some(law_limit) = law.stability_limit(&self.store) {
                limit = limit.tightest(law_limit);
            }
        }
        limit
    }

    fn prepare(&mut self, _ctx: &mut StepContext<'_>) {
        if !self.initialized {
            self.initialize();
        }
    }

    fn advance(&mut self, dt: f64, ctx: &mut StepContext<'_>) {
        debug_assert!(
            self.initialized,
            "ParticleDomain::advance requires initialize() to have established the force invariant"
        );
        let Self { store, forces, neighbors, bounds, integrator, thermostat, bath, shift_x, shift_y, .. } = self;
        let scheme = *integrator;
        // The integrator's per-particle updates are split across threads; the force
        // laws are not. A pair law applies Newton's third law by scattering into *both*
        // particles of a pair, so two threads working on different cells can collide on
        // a shared neighbour. Reformulating it as a gather would fix that and would also
        // change the summation order, which is the one thing this crate promises not to
        // do. See `lattice_cpu`, and `docs/execution.md` for what a GPU backend will
        // have to decide instead.
        let mut eval = |s: &mut ParticleStore| {
            Self::evaluate_forces(s, forces, neighbors.as_mut(), bounds.as_ref(), shift_x, shift_y);
        };
        let coupling = *thermostat;
        match (coupling, bath) {
            (Some(Thermostat::Langevin { .. }), Some(bath)) => {
                scheme.step_langevin(ctx.executor, dt, store, bath, &mut eval);
            }
            _ => scheme.step_with(ctx.executor, dt, store, &mut eval),
        }
        if let Some(Thermostat::VelocityRescale { temperature, relaxation }) = coupling {
            rescale_velocities(ctx.executor, store, dt, temperature, relaxation);
        }
        self.steps += 1;
        if let (Some(request), Some(rdf)) = (self.rdf_request, self.rdf.as_mut())
            && self.steps > request.after
            && self.steps % request.every == 0
        {
            rdf.sample(&self.store);
        }
    }

    fn observe(&self, out: &mut Observations) {
        let prefix = self.name.clone();
        let kinetic = self.kinetic_energy();
        let potential = self.potential_energy();
        let [px, py] = self.momentum();

        out.record(
            format!("{prefix}.count"),
            self.store.len() as f64,
            "1",
            lattice_ir::ObservationKind::Count,
        );
        // The halves are *metrics*: a gas melting out of a lattice converts potential
        // energy into kinetic on purpose, and both are supposed to move. Only the sum
        // is an invariant. Publishing the halves as invariants too would have a healthy
        // run report two conservation failures every step, which is how a reader learns
        // to stop reading the panel.
        let unit = Invariant::Energy.si_unit();
        out.record_metric(format!("{prefix}.kinetic_energy"), kinetic, unit);
        out.record_metric(format!("{prefix}.potential_energy"), potential, unit);
        // Under a thermostat, or with a force that does not derive from a potential
        // (drag), the total is not an invariant and must not be reported as one:
        // energy flows out on purpose, and a panel showing it as a conservation failure
        // every step teaches its reader to ignore the panel. An open boundary deletes
        // nothing, so it leaves the total an invariant and is not part of this test.
        let exchanges_energy =
            self.thermostat.is_some() || self.forces.iter().any(|f| !f.is_conservative());
        if !exchanges_energy {
            out.record_invariant(format!("{prefix}.total_energy"), Invariant::Energy, kinetic + potential);
        } else {
            out.record_metric(format!("{prefix}.total_energy"), kinetic + potential, unit);
        }
        if self.is_momentum_conserving() {
            out.record_invariant(format!("{prefix}.momentum_x"), Invariant::MomentumX, px);
            out.record_invariant(format!("{prefix}.momentum_y"), Invariant::MomentumY, py);
        } else {
            let unit = Invariant::MomentumX.si_unit();
            out.record_metric(format!("{prefix}.momentum_x"), px, unit);
            out.record_metric(format!("{prefix}.momentum_y"), py, unit);
        }

        // Total momentum is conserved *at zero* for a system set up at rest, so
        // "drift relative to the initial value" divides by round-off and reports a
        // perfectly conserved run as catastrophically broken. The meaningful scale is
        // the sum of the individual momenta being cancelled — which only this domain
        // knows. Publishing it lets a reader (or a viewer) judge the residual against
        // something physical.
        out.record_metric(
            format!("{prefix}.momentum_scale"),
            self.momentum_scale(),
            Invariant::MomentumX.si_unit(),
        );
        out.record_metric(format!("{prefix}.max_speed"), self.max_speed(), "m/s");

        // The rows below depend only on how the domain is configured, never on what
        // step it is at, so a table never gains or loses a row mid-run.
        if let Some(temperature) = self.temperature() {
            out.record_metric(format!("{prefix}.temperature"), temperature, "K");
        }
        if let Some(thermostat) = self.thermostat {
            out.record_metric(format!("{prefix}.target_temperature"), thermostat.temperature(), "K");
        }
        if let Some(pressure) = self.pressure() {
            out.record_metric(format!("{prefix}.pressure"), pressure, "N/m");
        }
        out.record_metric(format!("{prefix}.mean_squared_displacement"), self.mean_squared_displacement(), "m^2");
        if let Some(list) = &self.neighbors {
            out.record_metric(format!("{prefix}.neighbor_rebuilds"), list.rebuilds() as f64, "1");
        }
    }

    fn curves(&self) -> Vec<lattice_ir::Curve> {
        let Some(rdf) = &self.rdf else { return Vec::new() };
        let result = rdf.result();
        let mut curve = lattice_ir::Curve::new(
            format!("{}.rdf", self.name),
            ("r", "m"),
            ("g(r)", "1"),
            result.r.clone(),
            result.g.clone(),
        )
        .note(format!("{} frames, bin width {:.4e} m", result.frames, result.bin_width))
        .note(format!("mean number density {:.4e} /m^2", result.density));
        if result.frames > 0
            && let Some((r, g)) = result.peak()
        {
            curve = curve.note(format!("first maximum g = {g:.3} at r = {r:.4e} m"));
        }
        if self.bounds.is_some_and(|b| !b.is_fully_periodic()) {
            curve = curve.note(
                "normalized as for a homogeneous periodic box; near a wall the shells are cut \
                 off and g falls below 1 for geometric rather than physical reasons",
            );
        }
        vec![curve]
    }

    fn render_channels(&self) -> Vec<lattice_ir::RenderChannel<'_>> {
        // With a declared region, draw that. Without one, fit the view to the
        // particles rather than guessing a box and cropping whatever escapes.
        let (origin, extent) = match self.bounds {
            Some(bounds) => (bounds.min, bounds.size),
            None => lattice_ir::bounds_of(self.store.pos_x(), self.store.pos_y()),
        };
        let mut channels = vec![lattice_ir::RenderChannel::Particles {
            name: &self.name,
            x: self.store.pos_x(),
            y: self.store.pos_y(),
            origin,
            extent,
        }];
        if !self.bonded.is_empty() {
            channels.push(lattice_ir::RenderChannel::Bonds {
                name: &self.name,
                x: self.store.pos_x(),
                y: self.store.pos_y(),
                pairs: &self.bonded,
                periodic: self.bounds.map_or([false, false], |b| b.periodic_axes()),
                origin,
                extent,
            });
        }
        channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonded::{Bond, HarmonicBond};
    use crate::forces::{HarmonicWell, LennardJones, LinearDrag, UniformAcceleration};
    use lattice_ir::{Arena, Pcg32};

    fn ctx(arena: &mut Arena) -> StepContext<'_> {
        StepContext::new(arena)
    }

    #[test]
    fn periodic_wrapping_keeps_particles_in_the_box() {
        let b = BoundaryBox::periodic([0.0, 0.0], [10.0, 10.0]);
        let mut store = ParticleStore::with_capacity(3);
        store.spawn(ParticleSpec::at([10.5, -0.5])).unwrap();
        store.spawn(ParticleSpec::at([-25.0, 33.0])).unwrap();
        store.spawn(ParticleSpec::at([5.0, 5.0])).unwrap();
        b.apply(&mut store);

        for i in 0..3 {
            assert!((0.0..10.0).contains(&store.pos_x()[i]), "x {}", store.pos_x()[i]);
            assert!((0.0..10.0).contains(&store.pos_y()[i]), "y {}", store.pos_y()[i]);
        }
        assert!((store.pos_x()[0] - 0.5).abs() < 1e-12);
        assert!((store.pos_y()[0] - 9.5).abs() < 1e-12);
        assert!((store.pos_x()[2] - 5.0).abs() < 1e-12, "an interior particle must not move");
    }

    #[test]
    fn wrap_tracking_records_whole_box_lengths() {
        let b = BoundaryBox::periodic([0.0, 0.0], [10.0, 10.0]);
        let mut store = ParticleStore::with_capacity(2);
        store.spawn(ParticleSpec::at([10.5, -0.5])).unwrap();
        store.spawn(ParticleSpec::at([-25.0, 5.0])).unwrap();
        let mut sx = [0.0; 2];
        let mut sy = [0.0; 2];
        b.apply_tracking(&mut store, &mut sx, &mut sy);
        assert!((sx[0] - 10.0).abs() < 1e-12 && (sy[0] + 10.0).abs() < 1e-12, "{sx:?} {sy:?}");
        assert!((sx[1] + 30.0).abs() < 1e-12 && sy[1] == 0.0);
        // Unwrapped positions are the originals.
        assert!((store.pos_x()[1] + sx[1] + 25.0).abs() < 1e-12);
    }

    #[test]
    fn reflection_mirrors_position_and_flips_velocity() {
        let b = BoundaryBox::reflective([0.0, 0.0], [10.0, 10.0]);
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([10.5, -0.5]).with_velocity([3.0, -4.0])).unwrap();
        b.apply(&mut store);
        assert!((store.pos_x()[0] - 9.5).abs() < 1e-12);
        assert!((store.pos_y()[0] - 0.5).abs() < 1e-12);
        assert!((store.vel_x()[0] + 3.0).abs() < 1e-12);
        assert!((store.vel_y()[0] - 4.0).abs() < 1e-12);
        assert!(!b.is_fully_periodic() && b.is_closed());
    }

    #[test]
    fn reflection_terminates_for_a_runaway_particle() {
        let b = BoundaryBox::reflective([0.0, 0.0], [1.0, 1.0]);
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([1e9, 0.5]).with_velocity([1.0, 0.0])).unwrap();
        b.apply(&mut store);
        // The exact resting place does not matter; not hanging does.
        assert!((0.0..=1.0).contains(&store.pos_x()[0]));
    }

    #[test]
    fn open_boundaries_let_particles_leave() {
        let b = BoundaryBox::new([0.0, 0.0], [1.0, 1.0], ParticleBoundary::Open);
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([5.0, 5.0])).unwrap();
        b.apply(&mut store);
        assert_eq!(store.pos_x()[0], 5.0);
        assert!(!b.is_closed());
    }

    #[test]
    fn free_fall_matches_the_analytic_trajectory() {
        let mut d = ParticleDomain::new("p", 1)
            .with_integrator(Integrator::VelocityVerlet)
            .with_force(UniformAcceleration::earth_gravity());
        d.spawn(ParticleSpec::at([0.0, 100.0]).with_mass(3.0)).unwrap();
        d.initialize();

        let mut arena = Arena::with_capacity(0);
        let dt = 1e-3;
        let steps = 1000;
        for _ in 0..steps {
            d.advance(dt, &mut ctx(&mut arena));
        }
        let t = dt * steps as f64;
        let expected = 100.0 - 0.5 * 9.806_65 * t * t;
        assert!((d.store().pos_y()[0] - expected).abs() < 1e-9);
        // Mass must not affect the trajectory.
        assert!((d.store().vel_y()[0] + 9.806_65 * t).abs() < 1e-9);
        // The displacement tracker sees the fall.
        let fallen = 100.0 - expected;
        assert!((d.mean_squared_displacement() - fallen * fallen).abs() < 1e-9);
    }

    #[test]
    fn harmonic_oscillator_conserves_energy_and_keeps_its_period() {
        let k = 4.0;
        let mut d = ParticleDomain::new("osc", 1)
            .with_integrator(Integrator::VelocityVerlet)
            .with_force(HarmonicWell::new([0.0, 0.0], k));
        d.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        d.initialize();

        assert!(d.is_energy_conserving());
        let e0 = d.total_energy();
        assert!((e0 - 0.5 * k).abs() < 1e-12, "E0 = kx^2/2 = 2 J, got {e0}");

        // omega = 2 rad/s, so one period is pi seconds.
        let period = core::f64::consts::TAU / (k / 1.0f64).sqrt();
        let dt = period / 2000.0;
        let steps = 2000;
        let mut arena = Arena::with_capacity(0);
        let mut worst = 0.0f64;
        for _ in 0..steps {
            d.advance(dt, &mut ctx(&mut arena));
            worst = worst.max(((d.total_energy() - e0) / e0).abs());
        }
        assert!(worst < 1e-5, "energy drifted {worst:e}");
        // After exactly one period the particle is back at its start.
        assert!((d.store().pos_x()[0] - 1.0).abs() < 1e-5, "x = {}", d.store().pos_x()[0]);
    }

    #[test]
    fn drag_is_reported_as_non_conservative() {
        let mut d = ParticleDomain::new("drag", 1)
            .with_force(HarmonicWell::new([0.0, 0.0], 1.0))
            .with_force(LinearDrag::new(0.1));
        d.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        d.initialize();
        assert!(!d.is_energy_conserving(), "a scene with drag must not claim energy conservation");

        let e0 = d.total_energy();
        let mut arena = Arena::with_capacity(0);
        for _ in 0..5_000 {
            d.advance(1e-3, &mut ctx(&mut arena));
        }
        assert!(d.total_energy() < e0, "drag must remove energy");
    }

    /// A Lennard-Jones fluid with exactly zero net momentum.
    fn lennard_jones_gas(count_side: usize, spacing: f64, skin: Option<f64>) -> ParticleDomain {
        let count = count_side * count_side;
        let box_size = count_side as f64 * spacing;
        let mut d = ParticleDomain::new("lj", count)
            .with_integrator(Integrator::VelocityVerlet)
            .with_bounds(BoundaryBox::periodic([0.0, 0.0], [box_size, box_size]))
            .with_force(LennardJones::with_default_cutoff(1.0, 1.0));
        if let Some(skin) = skin {
            d = d.with_skin(skin);
        }
        let mut rng = Pcg32::seed_from_u64(7);
        let mut velocities = Vec::new();
        for _ in 0..count {
            velocities.push([rng.normal() * 0.3, rng.normal() * 0.3]);
        }
        let mean_x: f64 = velocities.iter().map(|v| v[0]).sum::<f64>() / count as f64;
        let mean_y: f64 = velocities.iter().map(|v| v[1]).sum::<f64>() / count as f64;
        for (index, v) in velocities.iter().enumerate() {
            let (i, j) = (index % count_side, index / count_side);
            d.spawn(
                ParticleSpec::at([(i as f64 + 0.5) * spacing, (j as f64 + 0.5) * spacing])
                    .with_velocity([v[0] - mean_x, v[1] - mean_y])
                    .with_mass(1.0),
            )
            .unwrap();
        }
        d.initialize();
        d
    }

    /// Pair forces obey Newton's third law, so total momentum is conserved to
    /// round-off regardless of how chaotic the trajectory becomes.
    #[test]
    fn lennard_jones_conserves_momentum_and_energy_in_a_periodic_box() {
        let mut d = lennard_jones_gas(8, 1.5, None);
        let p0 = d.momentum();
        assert!(p0[0].abs() < 1e-12 && p0[1].abs() < 1e-12, "setup should start at rest: {p0:?}");
        let e0 = d.total_energy();

        let mut arena = Arena::with_capacity(0);
        let dt = 1e-3;
        for _ in 0..5_000 {
            d.advance(dt, &mut ctx(&mut arena));
        }

        let p = d.momentum();
        assert!(p[0].abs() < 1e-9, "momentum x drifted to {}", p[0]);
        assert!(p[1].abs() < 1e-9, "momentum y drifted to {}", p[1]);

        let relative = ((d.total_energy() - e0) / e0.abs()).abs();
        assert!(relative < 5e-3, "LJ energy drifted {relative:e} over 5 ps-equivalent");
    }

    /// The skin is an optimisation, and this is the claim it must keep: the same
    /// pairs, the same conservation, and forces that agree to round-off with the plain
    /// cell list on every configuration along the way.
    #[test]
    fn a_verlet_skin_reproduces_the_cell_list_forces_and_conserves_the_same_things() {
        let mut plain = lennard_jones_gas(8, 1.5, None);
        let mut cached = lennard_jones_gas(8, 1.5, Some(0.4));
        assert_eq!(cached.skin(), Some(0.4));
        let e0 = cached.total_energy();
        assert!((plain.total_energy() - e0).abs() < 1e-12 * e0.abs());

        let mut arena = Arena::with_capacity(0);
        let mut worst_force = 0.0f64;
        for _ in 0..500 {
            cached.advance(1e-3, &mut ctx(&mut arena));
            // Put the plain domain on the cached trajectory and compare forces there.
            {
                let (src_x, src_y) = (cached.store().pos_x().to_vec(), cached.store().pos_y().to_vec());
                let d = plain.store_mut().dynamics();
                d.pos_x.copy_from_slice(&src_x);
                d.pos_y.copy_from_slice(&src_y);
            }
            plain.initialize();
            let (fx, fy) = (cached.store().force_x(), cached.store().force_y());
            let (gx, gy) = (plain.store().force_x(), plain.store().force_y());
            let scale = fx.iter().chain(fy).fold(0.0f64, |m, f| m.max(f.abs())).max(1.0);
            for i in 0..fx.len() {
                worst_force = worst_force.max((fx[i] - gx[i]).abs() / scale).max((fy[i] - gy[i]).abs() / scale);
            }
        }
        // Different summation order over at most a few dozen neighbours: 1e-12 is
        // two orders of magnitude above what that costs at f64.
        assert!(worst_force < 1e-12, "forces disagreed by {worst_force:e} of scale");
        let list = cached.neighbors().unwrap();
        assert!(list.rebuilds() > 1 && list.rebuilds() < list.updates(), "{} of {}", list.rebuilds(), list.updates());
        assert!(((cached.total_energy() - e0) / e0.abs()).abs() < 5e-3);
        let p = cached.momentum();
        assert!(p[0].abs() < 1e-9 && p[1].abs() < 1e-9);
    }

    #[test]
    fn a_langevin_bath_holds_a_gas_at_its_temperature_and_says_what_it_gave_up() {
        let mut d = lennard_jones_gas(10, 2.0, Some(0.3));
        d = d.with_thermostat(Thermostat::Langevin { temperature: 5e22, friction: 2.0 }).with_seed(3);
        d.initialize();
        assert!(!d.is_energy_conserving() && !d.is_momentum_conserving());
        assert!(d.contract().name.contains("langevin"));
        assert!(d.contract().conserves.is_empty());

        let mut arena = Arena::with_capacity(0);
        for _ in 0..3_000 {
            d.advance(2e-3, &mut ctx(&mut arena));
        }
        let mut mean = 0.0;
        let samples = 200;
        for _ in 0..samples {
            for _ in 0..25 {
                d.advance(2e-3, &mut ctx(&mut arena));
            }
            mean += d.temperature().unwrap();
        }
        mean /= samples as f64;
        // Interacting particles: BAOAB's configurational bias is O(dt²) and small at
        // this step; the statistical scatter of the mean dominates.
        assert!((mean / 5e22 - 1.0).abs() < 0.03, "mean temperature {mean:e}");

        let mut obs = Observations::new();
        d.observe(&mut obs);
        assert!(obs.get("lj.total_energy").is_some_and(|o| !matches!(o.kind, lattice_ir::ObservationKind::Invariant(_))));
        assert!(obs.get("lj.momentum_x").is_some_and(|o| !matches!(o.kind, lattice_ir::ObservationKind::Invariant(_))));
        assert_eq!(obs.value("lj.target_temperature"), Some(5e22));
        assert!(obs.value("lj.pressure").is_some());
        assert!(obs.value("lj.neighbor_rebuilds").unwrap() > 0.0);
    }

    #[test]
    fn velocity_rescaling_keeps_momentum_and_reports_energy_as_a_metric() {
        let mut d = lennard_jones_gas(6, 2.0, None)
            .with_thermostat(Thermostat::VelocityRescale { temperature: 1e22, relaxation: 0.05 });
        d.initialize();
        assert!(d.is_momentum_conserving() && !d.is_energy_conserving());
        assert!(d.contract().conserves.contains(&Invariant::MomentumX));

        let mut arena = Arena::with_capacity(0);
        for _ in 0..2_000 {
            d.advance(1e-3, &mut ctx(&mut arena));
        }
        let p = d.momentum();
        assert!(p[0].abs() < 1e-9 && p[1].abs() < 1e-9, "{p:?}");
        assert!((d.temperature().unwrap() / 1e22 - 1.0).abs() < 0.2, "{:?}", d.temperature());
    }

    #[test]
    #[should_panic(expected = "needs the velocity_verlet integrator")]
    fn langevin_with_a_first_order_scheme_is_refused_at_initialization() {
        let mut d = ParticleDomain::new("bad", 2)
            .with_integrator(Integrator::SemiImplicitEuler)
            .with_thermostat(Thermostat::Langevin { temperature: 1.0, friction: 1.0 });
        d.spawn(ParticleSpec::default()).unwrap();
        d.initialize();
    }

    #[test]
    fn a_bonded_chain_conserves_energy_excludes_its_pairs_and_draws_its_bonds() {
        let n = 8;
        let mut d = ParticleDomain::new("chain", n)
            .with_integrator(Integrator::VelocityVerlet)
            .with_bounds(BoundaryBox::periodic([0.0, 0.0], [20.0, 20.0]))
            .with_force(LennardJones::with_default_cutoff(1.0, 1.0));
        let ids: Vec<_> =
            (0..n).map(|i| d.spawn(ParticleSpec::at([2.0 + 1.1 * i as f64, 10.0]).with_mass(1.0)).unwrap()).collect();
        let bonds = (0..n - 1).map(|i| Bond::new(ids[i], ids[i + 1], 1.1, 50.0)).collect();
        d = d.with_force(HarmonicBond::new(bonds));
        {
            let dyn_ = d.store_mut().dynamics();
            dyn_.vel_y[0] = 0.5;
            dyn_.vel_y[n - 1] = -0.5;
        }
        d.initialize();
        assert_eq!(d.bonded_pairs().len(), n - 1);
        assert_eq!(d.neighbors().unwrap().exclusions().len(), n - 1);
        assert!(d.render_channels().iter().any(|c| matches!(c, lattice_ir::RenderChannel::Bonds { .. })));
        assert!(d.stable_step().max < 2.0 / (50.0f64 / 0.5).sqrt() + 1e-12, "the bond binds the step");

        let e0 = d.total_energy();
        let mut arena = Arena::with_capacity(0);
        let mut worst = 0.0f64;
        for _ in 0..4_000 {
            d.advance(1e-3, &mut ctx(&mut arena));
            worst = worst.max(((d.total_energy() - e0) / e0.abs()).abs());
        }
        assert!(worst < 1e-3, "chain energy drifted {worst:e}");
        let p = d.momentum();
        assert!(p[0].abs() < 1e-9 && p[1].abs() < 1e-9);
    }

    #[test]
    fn a_radial_distribution_is_accumulated_on_its_cadence() {
        let mut d = lennard_jones_gas(6, 1.5, None).with_rdf(RdfRequest { bins: 20, range: 4.0, every: 5, after: 0 });
        d.initialize();
        assert!(d.radial_distribution().is_none(), "nothing sampled yet");
        let mut arena = Arena::with_capacity(0);
        for _ in 0..50 {
            d.advance(1e-3, &mut ctx(&mut arena));
        }
        let rdf = d.radial_distribution().unwrap();
        assert_eq!(rdf.frames, 10);
        assert_eq!(rdf.g.len(), 20);
        assert!(rdf.counts.iter().sum::<u64>() > 0);
    }

    #[test]
    fn a_requested_rdf_is_published_as_a_curve_before_and_after_sampling() {
        assert!(lennard_jones_gas(4, 1.5, None).curves().is_empty(), "no request, no curve");

        let mut d = lennard_jones_gas(6, 1.5, None).with_rdf(RdfRequest { bins: 20, range: 4.0, every: 5, after: 0 });
        d.initialize();
        // The set of curves depends on configuration, not on progress.
        let empty = d.curves();
        assert_eq!(empty.len(), 1);
        assert!(empty[0].notes[0].starts_with("0 frames"), "{:?}", empty[0].notes);

        let mut arena = Arena::with_capacity(0);
        for _ in 0..50 {
            d.advance(1e-3, &mut ctx(&mut arena));
        }
        let curve = &d.curves()[0];
        assert_eq!(curve.name, format!("{}.rdf", d.name()));
        assert_eq!((curve.x_label, curve.x_unit, curve.y_label), ("r", "m", "g(r)"));
        assert_eq!(curve.y, d.radial_distribution().unwrap().g);
    }

    #[test]
    fn stable_step_reports_the_binding_force() {
        let mut d = ParticleDomain::new("s", 1)
            .with_preferred_step(1.0)
            .with_force(HarmonicWell::new([0.0, 0.0], 100.0));
        d.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        d.initialize();

        let limit = d.stable_step();
        // omega = 10 rad/s, so the stability limit is 0.2 s.
        assert!((limit.max - 0.2).abs() < 1e-12, "max = {}", limit.max);
        assert_eq!(limit.reason, lattice_ir::StabilityReason::OscillationPeriod);
        assert!(limit.preferred < limit.max);
    }

    #[test]
    fn a_domain_with_no_stiff_forces_is_unconditionally_stable() {
        let mut d = ParticleDomain::new("g", 1)
            .with_preferred_step(0.005)
            .with_force(UniformAcceleration::earth_gravity());
        d.spawn(ParticleSpec::default()).unwrap();
        d.initialize();
        let limit = d.stable_step();
        assert!(limit.max.is_infinite());
        assert_eq!(limit.preferred, 0.005);
    }

    #[test]
    fn observations_are_published_under_the_domain_name() {
        let mut d = ParticleDomain::new("gas", 2)
            .with_force(HarmonicWell::new([0.0, 0.0], 1.0));
        d.spawn(ParticleSpec::at([1.0, 0.0]).with_velocity([0.0, 2.0]).with_mass(1.0)).unwrap();
        d.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(1.0)).unwrap();
        d.initialize();

        let mut obs = Observations::new();
        d.observe(&mut obs);
        assert_eq!(obs.value("gas.count"), Some(2.0));
        assert!((obs.value("gas.kinetic_energy").unwrap() - 2.0).abs() < 1e-12);
        assert!((obs.value("gas.potential_energy").unwrap() - 1.0).abs() < 1e-12);
        assert!((obs.value("gas.total_energy").unwrap() - 3.0).abs() < 1e-12);
        assert!((obs.value("gas.momentum_y").unwrap() - 2.0).abs() < 1e-12);
        // Two particles have two thermal degrees of freedom; one would have none.
        assert!(obs.value("gas.temperature").unwrap() > 0.0);
        assert_eq!(obs.value("gas.mean_squared_displacement"), Some(0.0));
        assert!(obs.value("gas.pressure").is_none(), "no periodic box, no pressure");
        assert!(obs.first_non_finite().is_none());

        let mut single = ParticleDomain::new("one", 1);
        single.spawn(ParticleSpec::default().with_velocity([1.0, 0.0])).unwrap();
        single.initialize();
        let mut obs = Observations::new();
        single.observe(&mut obs);
        assert!(obs.value("one.temperature").is_none(), "a lone particle has no temperature");
    }

    /// The observation must agree with the contract: a scene whose energy flows out on
    /// purpose publishes its total as a metric, never as an invariant that fails every
    /// step. An open boundary deletes nothing, so it keeps the invariant.
    #[test]
    fn total_energy_is_an_invariant_only_where_energy_is_conserved() {
        let kind_of = |d: &ParticleDomain, name: &str| {
            let mut obs = Observations::new();
            d.observe(&mut obs);
            obs.get(&format!("{name}.total_energy")).map(|o| o.kind)
        };
        let scene = |name: &str| {
            let mut d = ParticleDomain::new(name, 2).with_force(HarmonicWell::new([0.0, 0.0], 1.0));
            d.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
            d.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(1.0)).unwrap();
            d
        };

        let mut drag = scene("drag").with_force(LinearDrag::new(0.1));
        drag.initialize();
        assert!(matches!(kind_of(&drag, "drag"), Some(lattice_ir::ObservationKind::Metric)), "drag removes energy on purpose");

        let mut bath = scene("bath").with_thermostat(Thermostat::VelocityRescale { temperature: 1.0, relaxation: 1.0 });
        bath.initialize();
        assert!(matches!(kind_of(&bath, "bath"), Some(lattice_ir::ObservationKind::Metric)), "a bath exchanges energy");

        let mut open = scene("open").with_bounds(BoundaryBox::new([-5.0, -5.0], [5.0, 5.0], ParticleBoundary::Open));
        open.initialize();
        assert!(
            matches!(kind_of(&open, "open"), Some(lattice_ir::ObservationKind::Invariant(Invariant::Energy))),
            "an open boundary deletes nothing, so the total is still conserved"
        );
        // The difference from `is_energy_conserving`, pinned: that test also gives up
        // on an open boundary, and the observation deliberately does not.
        assert!(!open.is_energy_conserving());
    }

    #[test]
    fn contracts_differ_by_configuration_and_are_complete() {
        for integrator in
            [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
        {
            let d = ParticleDomain::new("c", 1).with_integrator(integrator);
            let contract = d.contract();
            assert!(contract.audit().is_empty(), "{} has gaps", contract.name);
            assert!(contract.name.contains(integrator.name()));
        }
        for thermostat in [
            Thermostat::Langevin { temperature: 1.0, friction: 1.0 },
            Thermostat::VelocityRescale { temperature: 1.0, relaxation: 1.0 },
        ] {
            let d = ParticleDomain::new("c", 1)
                .with_integrator(Integrator::VelocityVerlet)
                .with_thermostat(thermostat);
            let contract = d.contract();
            assert!(contract.audit().is_empty(), "{} has gaps", contract.name);
            assert!(contract.name.contains(thermostat.name()), "{}", contract.name);
            assert!(!contract.conserves.contains(&Invariant::Energy));
            assert_eq!(contract.fidelity, FidelityProfile::MolecularKinetic);
        }

        // Explicit Euler must not claim to conserve energy; the others must.
        let euler = ParticleDomain::new("c", 1).with_integrator(Integrator::ExplicitEuler);
        assert!(!euler.contract().conserves.contains(&Invariant::Energy));
        let verlet = ParticleDomain::new("c", 1).with_integrator(Integrator::VelocityVerlet);
        assert!(verlet.contract().conserves.contains(&Invariant::Energy));
        // Six distinct contracts ship.
        let mut names: Vec<&str> = [
            (Integrator::ExplicitEuler, None),
            (Integrator::SemiImplicitEuler, None),
            (Integrator::SemiImplicitEuler, Some(Thermostat::VelocityRescale { temperature: 1.0, relaxation: 1.0 })),
            (Integrator::VelocityVerlet, None),
            (Integrator::VelocityVerlet, Some(Thermostat::Langevin { temperature: 1.0, friction: 1.0 })),
            (Integrator::VelocityVerlet, Some(Thermostat::VelocityRescale { temperature: 1.0, relaxation: 1.0 })),
        ]
        .into_iter()
        .map(|(i, t)| ParticleDomain::contract_for(i, t).name)
        .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 6);
    }

    #[test]
    #[should_panic(expected = "no region")]
    fn a_cutoff_force_without_a_region_is_a_configuration_error() {
        let mut d = ParticleDomain::new("lj", 4)
            .with_force(LennardJones::with_default_cutoff(1.0, 1.0));
        d.spawn(ParticleSpec::default()).unwrap();
        d.initialize();
    }

    #[test]
    fn prepare_initializes_a_domain_that_was_not_initialized() {
        let mut d = ParticleDomain::new("p", 1).with_force(UniformAcceleration::earth_gravity());
        d.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(1.0)).unwrap();
        let mut arena = Arena::with_capacity(0);
        d.prepare(&mut ctx(&mut arena));
        d.advance(0.01, &mut ctx(&mut arena));
        assert!(d.store().pos_y()[0] < 1.0);
    }

    #[test]
    fn re_initializing_keeps_the_bath_stream_and_the_neighbour_list() {
        let mut d = lennard_jones_gas(5, 2.0, Some(0.3))
            .with_thermostat(Thermostat::Langevin { temperature: 1e22, friction: 1.0 });
        d.initialize();
        let mut arena = Arena::with_capacity(0);
        for _ in 0..20 {
            d.advance(1e-3, &mut ctx(&mut arena));
        }
        let rebuilds = d.neighbors().unwrap().rebuilds();
        let state_before = d.bath.as_ref().unwrap().rng.snapshot();
        d.store_mut();
        d.initialize();
        assert_eq!(d.bath.as_ref().unwrap().rng.snapshot(), state_before, "the stream continues");
        assert!(d.neighbors().unwrap().rebuilds() >= rebuilds, "the list is kept, not rebuilt from scratch");
    }
}
