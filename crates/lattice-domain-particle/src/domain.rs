//! The particle domain: storage, forces, integrator, boundaries, and diagnostics
//! assembled into something the runtime can schedule.

use lattice_ir::{
    Domain, FidelityProfile, Invariant, Observations, ParticleSpec, ParticleStore, Precision,
    SolverContract, StableStep, StepContext,
};

use crate::forces::ForceLaw;
use crate::integrator::Integrator;
use crate::neighbors::CellList;

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

    /// Enforce the boundary on every particle.
    pub fn apply(&self, store: &mut ParticleStore) {
        let d = store.dynamics();
        let max = self.max();
        for i in 0..d.len() {
            apply_axis(self.x, &mut d.pos_x[i], &mut d.vel_x[i], self.min[0], max[0], self.size[0]);
            apply_axis(self.y, &mut d.pos_y[i], &mut d.vel_y[i], self.min[1], max[1], self.size[1]);
        }
    }

    /// True when no particle can leave the region.
    pub fn is_closed(&self) -> bool {
        self.x != ParticleBoundary::Open && self.y != ParticleBoundary::Open
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
    cells: Option<CellList>,
    bounds: Option<BoundaryBox>,
    preferred_dt: f64,
    steps: u64,
    initialized: bool,
}

impl ParticleDomain {
    /// A domain named `name` with room for `capacity` particles.
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            store: ParticleStore::with_capacity(capacity),
            forces: Vec::new(),
            integrator: Integrator::default(),
            cells: None,
            bounds: None,
            preferred_dt: 1e-3,
            steps: 0,
            initialized: false,
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

    /// Build the neighbour list (if any force needs one) and evaluate initial forces.
    ///
    /// # Panics
    ///
    /// If a force law needs a neighbour list but no region was set with
    /// [`ParticleDomain::with_bounds`]. A cutoff-based force with no region has no
    /// well-defined binning, and silently falling back to O(N²) would turn a
    /// configuration mistake into a mysterious performance cliff.
    pub fn initialize(&mut self) {
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
            self.cells = Some(CellList::new(
                bounds.min,
                bounds.size,
                bounds.periodic_axes(),
                cutoff,
                self.store.capacity(),
            ));
        } else {
            self.cells = None;
        }

        self.refresh_forces();
        self.initialized = true;
    }

    /// Recompute forces for the current configuration.
    fn refresh_forces(&mut self) {
        let Self { store, forces, cells, bounds, .. } = self;
        if let Some(b) = bounds.as_ref() {
            b.apply(store);
        }
        if let Some(c) = cells.as_mut() {
            c.rebuild(store);
        }
        store.clear_forces();
        let mut view = store.force_accumulation();
        for law in forces.iter() {
            law.accumulate(&mut view, cells.as_ref());
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

    /// The neighbour list, if one is in use.
    pub fn cells(&self) -> Option<&CellList> {
        self.cells.as_ref()
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
    /// Counts the preallocated particle arrays and the neighbour list. Excludes the
    /// force laws themselves, which are a handful of scalars each.
    pub fn memory_bytes(&self) -> usize {
        let particles = self.store.capacity() * ParticleStore::BYTES_PER_PARTICLE;
        let cells = self.cells.as_ref().map_or(0, |c| {
            // CSR offsets, item list, per-cell cursor, and the 3x3 neighbour table.
            let cell_count = c.cell_count();
            (cell_count + 1 + cell_count + cell_count * 9) * size_of::<u32>()
                + self.store.capacity() * size_of::<u32>()
                + cell_count
        });
        particles + cells
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
        self.forces
            .iter()
            .filter(|law| law.is_conservative())
            .map(|law| law.potential_energy(&self.store, self.cells.as_ref()))
            .sum()
    }

    /// Kinetic plus potential energy, J.
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
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

    /// Fastest particle speed, m/s.
    pub fn max_speed(&self) -> f64 {
        let (vx, vy) = (self.store.vel_x(), self.store.vel_y());
        (0..vx.len()).fold(0.0f64, |acc, i| acc.max((vx[i] * vx[i] + vy[i] * vy[i]).sqrt()))
    }

    /// True when every force law present derives from a potential.
    ///
    /// Total energy is only expected to be conserved when this holds *and* the
    /// boundary is closed. The validation suite checks this before asserting on drift,
    /// so that adding drag to a scene does not read as a broken integrator.
    pub fn is_energy_conserving(&self) -> bool {
        self.forces.iter().all(|f| f.is_conservative())
            && self.bounds.is_none_or(|b| b.is_closed())
    }
}

/// Builds the contract for one integrator choice.
///
/// A single static contract cannot be honest here: explicit Euler and velocity Verlet
/// make opposite claims about energy. Each integrator therefore gets its own, and
/// [`Domain::contract`] returns the one that matches the active configuration.
const fn contract_for(integrator: Integrator) -> SolverContract {
    SolverContract {
        name: match integrator {
            Integrator::ExplicitEuler => "particles2d[explicit_euler]",
            Integrator::SemiImplicitEuler => "particles2d[semi_implicit_euler]",
            Integrator::VelocityVerlet => "particles2d[velocity_verlet]",
        },
        summary: "2D point particles advanced under a sum of force laws",
        governing_equations: &[
            "dx/dt = v",
            "m dv/dt = sum_k F_k(x, v)",
        ],
        discretization: "none — particles are discrete degrees of freedom; \
                         local pair interactions are truncated at a cutoff and \
                         accelerated with a uniform cell list",
        integrator: match integrator {
            Integrator::ExplicitEuler => "explicit (forward) Euler, 1st order, not symplectic",
            Integrator::SemiImplicitEuler => "semi-implicit (Euler-Cromer) Euler, 1st order, symplectic",
            Integrator::VelocityVerlet => "velocity Verlet, 2nd order, symplectic and time-reversible",
        },
        assumptions: &[
            "particles are point masses; radius affects only collision queries, not dynamics",
            "pair interactions vanish beyond their declared cutoff",
            "periodic axes use the minimum-image convention, which requires the box to be \
             at least twice the interaction cutoff on that axis",
            "velocity-dependent forces under velocity Verlet are evaluated at the half-step \
             velocity, which is first-order accurate for those terms only",
        ],
        valid_regime: match integrator {
            Integrator::ExplicitEuler => {
                "comparison and teaching only; unsuitable for any quantitative result"
            }
            _ => "non-relativistic classical dynamics with resolved force timescales",
        },
        stability: match integrator {
            Integrator::ExplicitEuler => {
                "unstable for oscillatory systems at every timestep: the amplitude grows \
                 by a factor sqrt(1 + (omega*dt)^2) per step"
            }
            _ => {
                "dt < 2/omega_max, where omega_max is the highest vibrational frequency; \
                 accuracy typically needs 20-50 steps per period of the fastest mode"
            }
        },
        conserves: match integrator {
            Integrator::ExplicitEuler => &[Invariant::MomentumX, Invariant::MomentumY],
            _ => &[Invariant::Energy, Invariant::MomentumX, Invariant::MomentumY],
        },
        known_non_conservation: match integrator {
            Integrator::ExplicitEuler => &[
                "energy: not symplectic, so error accumulates secularly rather than \
                 oscillating — energy grows without bound in any oscillatory system",
                "angular momentum: not preserved by the discrete update",
            ],
            _ => &[
                "energy: symplectic schemes bound the error but do not eliminate it; \
                 expect a bounded oscillation of order dt^p and no secular drift",
                "energy at a pair-force cutoff: the truncated-and-shifted Lennard-Jones \
                 potential is continuous in energy but not in force, so each crossing \
                 of the cutoff radius injects a small impulse",
                "energy under non-conservative laws (drag) and non-closed boundaries \
                 (open edges), both of which are intentional and reported separately",
                "angular momentum under a non-central force law or a non-periodic box",
            ],
        },
        fidelity: match integrator {
            Integrator::ExplicitEuler => FidelityProfile::Interactive,
            _ => FidelityProfile::Engineering2d,
        },
        precisions: &[Precision::Accurate64, Precision::Deterministic64],
        deterministic: true,
        differentiable: false,
        validation_cases: &[
            "free fall under constant acceleration",
            "harmonic oscillator: period, amplitude, and energy drift",
            "pairwise momentum conservation (Newton's third law)",
            "Lennard-Jones energy conservation in a periodic box",
            "observed convergence order matches the declared order",
        ],
        references: &[
            "Verlet, L. (1967). Computer experiments on classical fluids. Phys. Rev. 159, 98.",
            "Hairer, Lubich & Wanner (2006). Geometric Numerical Integration, 2nd ed.",
            "Allen & Tildesley (2017). Computer Simulation of Liquids, 2nd ed., ch. 3-5.",
        ],
    }
}

static EXPLICIT_EULER_CONTRACT: SolverContract = contract_for(Integrator::ExplicitEuler);
static SEMI_IMPLICIT_CONTRACT: SolverContract = contract_for(Integrator::SemiImplicitEuler);
static VELOCITY_VERLET_CONTRACT: SolverContract = contract_for(Integrator::VelocityVerlet);

impl Domain for ParticleDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn contract(&self) -> &'static SolverContract {
        match self.integrator {
            Integrator::ExplicitEuler => &EXPLICIT_EULER_CONTRACT,
            Integrator::SemiImplicitEuler => &SEMI_IMPLICIT_CONTRACT,
            Integrator::VelocityVerlet => &VELOCITY_VERLET_CONTRACT,
        }
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

    fn advance(&mut self, dt: f64, _ctx: &mut StepContext<'_>) {
        debug_assert!(
            self.initialized,
            "ParticleDomain::advance requires initialize() to have established the force invariant"
        );
        let Self { store, forces, cells, bounds, integrator, .. } = self;
        let scheme = *integrator;
        scheme.step(dt, store, |s| {
            // Positions have just changed: wrap or reflect before binning, so the
            // neighbour list and the minimum-image convention agree on where
            // particles are.
            if let Some(b) = bounds.as_ref() {
                b.apply(s);
            }
            if let Some(c) = cells.as_mut() {
                c.rebuild(s);
            }
            s.clear_forces();
            let mut view = s.force_accumulation();
            for law in forces.iter() {
                law.accumulate(&mut view, cells.as_ref());
            }
        });
        self.steps += 1;
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
        out.record_invariant(format!("{prefix}.kinetic_energy"), Invariant::KineticEnergy, kinetic);
        out.record_invariant(
            format!("{prefix}.potential_energy"),
            Invariant::PotentialEnergy,
            potential,
        );
        out.record_invariant(format!("{prefix}.total_energy"), Invariant::Energy, kinetic + potential);
        out.record_invariant(format!("{prefix}.momentum_x"), Invariant::MomentumX, px);
        out.record_invariant(format!("{prefix}.momentum_y"), Invariant::MomentumY, py);
        out.record_metric(format!("{prefix}.max_speed"), self.max_speed(), "m/s");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::{HarmonicWell, LennardJones, LinearDrag, UniformAcceleration};
    use lattice_ir::Arena;

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
    fn reflection_mirrors_position_and_flips_velocity() {
        let b = BoundaryBox::reflective([0.0, 0.0], [10.0, 10.0]);
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([10.5, -0.5]).with_velocity([3.0, -4.0])).unwrap();
        b.apply(&mut store);
        assert!((store.pos_x()[0] - 9.5).abs() < 1e-12);
        assert!((store.pos_y()[0] - 0.5).abs() < 1e-12);
        assert!((store.vel_x()[0] + 3.0).abs() < 1e-12);
        assert!((store.vel_y()[0] - 4.0).abs() < 1e-12);
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

    /// Pair forces obey Newton's third law, so total momentum is conserved to
    /// round-off regardless of how chaotic the trajectory becomes.
    #[test]
    fn lennard_jones_conserves_momentum_and_energy_in_a_periodic_box() {
        let sigma = 1.0;
        let epsilon = 1.0;
        let box_size = 12.0;
        let mut d = ParticleDomain::new("lj", 64)
            .with_integrator(Integrator::VelocityVerlet)
            .with_bounds(BoundaryBox::periodic([0.0, 0.0], [box_size, box_size]))
            .with_force(LennardJones::with_default_cutoff(epsilon, sigma));

        // An 8x8 lattice at 1.5 sigma, with a deterministic velocity spread whose
        // net momentum is exactly zero by construction.
        let mut rng = lattice_ir::Pcg32::seed_from_u64(7);
        let mut velocities = Vec::new();
        for _ in 0..64 {
            velocities.push([rng.normal() * 0.3, rng.normal() * 0.3]);
        }
        let mean_x: f64 = velocities.iter().map(|v| v[0]).sum::<f64>() / 64.0;
        let mean_y: f64 = velocities.iter().map(|v| v[1]).sum::<f64>() / 64.0;
        for (index, v) in velocities.iter().enumerate() {
            let (i, j) = (index % 8, index / 8);
            d.spawn(
                ParticleSpec::at([(i as f64 + 0.5) * 1.5, (j as f64 + 0.5) * 1.5])
                    .with_velocity([v[0] - mean_x, v[1] - mean_y])
                    .with_mass(1.0),
            )
            .unwrap();
        }
        d.initialize();

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
        assert!(obs.first_non_finite().is_none());
    }

    #[test]
    fn contracts_differ_by_integrator_and_are_complete() {
        for integrator in
            [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
        {
            let d = ParticleDomain::new("c", 1).with_integrator(integrator);
            let contract = d.contract();
            assert!(contract.audit().is_empty(), "{} has gaps", contract.name);
            assert!(contract.name.contains(integrator.name()));
        }

        // Explicit Euler must not claim to conserve energy; the others must.
        let euler = ParticleDomain::new("c", 1).with_integrator(Integrator::ExplicitEuler);
        assert!(!euler.contract().conserves.contains(&Invariant::Energy));
        let verlet = ParticleDomain::new("c", 1).with_integrator(Integrator::VelocityVerlet);
        assert!(verlet.contract().conserves.contains(&Invariant::Energy));
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
}
