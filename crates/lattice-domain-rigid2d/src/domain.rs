//! The rigid-body domain: geometry, bodies, joints and the solvers, assembled into
//! something the runtime can schedule.

use lattice_ir::{
    BodyId, BodySpec, Domain, FidelityProfile, Invariant, ObservationKind, Observations, Precision,
    RigidBodyStore, ShapeId, SolverContract, StabilityReason, StableStep, StepContext,
};

use crate::broadphase::{shape_lookup, world_bounds, BroadPhase};
use crate::constraint::{Joint, JointSolver};
use crate::contact::Contact;
use crate::narrowphase::{generate_contacts, LINEAR_SLOP};
use crate::shape::{Collider, MassProperties, Shape, ShapeError};
use crate::solver::{ContactSolver, SolveReport, SolverConfig};

/// A 2D rigid-body world.
///
/// Build one with [`RigidDomain::new`], register the shapes bodies will wear with
/// [`RigidDomain::register`], then spawn bodies and add joints.
///
/// # Example
///
/// ```
/// use lattice_domain_rigid2d::{Collider, RigidDomain, Shape};
/// use lattice_ir::{Arena, BodySpec, Domain, StepContext};
///
/// let mut world = RigidDomain::new("scene", 16);
/// let ground = world.register(Collider::new(Shape::rectangle(10.0, 0.5).unwrap()));
/// let crate_ = world.register(Collider::new(Shape::rectangle(0.5, 0.5).unwrap()));
///
/// world.spawn(BodySpec::statik([0.0, 0.0], ground));
/// world.spawn_with_density(BodySpec::at([0.0, 3.0], crate_), 1.0);
///
/// let mut arena = Arena::with_capacity(1024);
/// for _ in 0..600 {
///     let mut ctx = StepContext::new(&mut arena);
///     world.advance(1.0 / 120.0, &mut ctx);
/// }
///
/// // It landed on the ground rather than falling through it.
/// let y = world.bodies().pos_y()[1];
/// assert!((y - 1.0).abs() < 0.05, "resting at {y}");
/// ```
#[derive(Debug)]
pub struct RigidDomain {
    name: String,
    bodies: RigidBodyStore,
    colliders: Vec<Collider>,
    joints: Vec<Joint>,
    gravity: [f64; 2],
    broad: BroadPhase,
    contacts: Vec<Contact>,
    contact_solver: ContactSolver,
    joint_solver: JointSolver,
    preferred_dt: f64,
    steps: u64,
    last_report: SolveReport,
    last_joint_error: f64,
    /// Cumulative angular momentum injected by position correction, kg·m²/s.
    correction_drift: f64,
    /// Whether [`Domain::prepare`] has run since the last [`Domain::advance`].
    prepared: bool,
}

impl RigidDomain {
    /// A world named `name` with room for `capacity` bodies.
    pub fn new(name: impl Into<String>, capacity: usize) -> RigidDomain {
        RigidDomain {
            name: name.into(),
            bodies: RigidBodyStore::with_capacity(capacity),
            colliders: Vec::new(),
            joints: Vec::new(),
            gravity: [0.0, -9.806_65],
            broad: BroadPhase::with_capacity(capacity),
            contacts: Vec::with_capacity(capacity * 2),
            contact_solver: ContactSolver::new(SolverConfig::default()),
            joint_solver: JointSolver::new(),
            preferred_dt: 1.0 / 120.0,
            steps: 0,
            last_report: SolveReport::default(),
            last_joint_error: 0.0,
            correction_drift: 0.0,
            prepared: false,
        }
    }

    /// Set the uniform acceleration applied to every dynamic body, m/s².
    pub fn with_gravity(mut self, gravity: [f64; 2]) -> RigidDomain {
        self.gravity = gravity;
        self
    }

    /// Set the timestep this domain asks for.
    pub fn with_preferred_step(mut self, dt: f64) -> RigidDomain {
        self.preferred_dt = dt;
        self
    }

    /// Tune the contact solver.
    pub fn with_solver(mut self, config: SolverConfig) -> RigidDomain {
        self.contact_solver.set_config(config);
        self
    }

    /// Register a collider and return the id bodies use to wear it.
    ///
    /// Shapes are shared: a stack of fifty identical crates registers one shape and
    /// fifty bodies referencing it.
    pub fn register(&mut self, collider: Collider) -> ShapeId {
        self.colliders.push(collider);
        ShapeId::from_index((self.colliders.len() - 1) as u32)
    }

    /// Register a collider from a shape, or report why the shape is invalid.
    pub fn register_shape(&mut self, shape: Result<Shape, ShapeError>) -> Result<ShapeId, ShapeError> {
        Ok(self.register(Collider::new(shape?)))
    }

    /// The registered colliders.
    pub fn colliders(&self) -> &[Collider] {
        &self.colliders
    }

    /// Add a body exactly as specified.
    pub fn spawn(&mut self, spec: BodySpec) -> Option<BodyId> {
        self.bodies.spawn(spec)
    }

    /// Add a body, taking its mass and inertia from its shape and an areal density.
    ///
    /// `density` is kg/m² — this is a 2D world, and pretending it has a thickness
    /// would make every printed mass wrong by a factor nobody declared.
    ///
    /// A shape with no area (a segment) gets no mass and is therefore static. Say so
    /// explicitly with [`BodySpec::statik`] if that is what was meant, or give the body
    /// an explicit mass; a massless dynamic body is a division by zero waiting to
    /// happen, and this refuses to create one silently.
    pub fn spawn_with_density(&mut self, spec: BodySpec, density: f64) -> Option<BodyId> {
        let properties = self
            .colliders
            .get(spec.shape.index())
            .map_or(MassProperties::STATIC, |c| c.shape.mass_properties(density));
        self.spawn(BodySpec { mass: properties.mass, inertia: properties.inertia, ..spec })
    }

    /// Add a body with a stated total mass, its inertia scaled from the shape.
    pub fn spawn_with_mass(&mut self, spec: BodySpec, mass: f64) -> Option<BodyId> {
        let properties = self
            .colliders
            .get(spec.shape.index())
            .map_or(MassProperties::STATIC, |c| c.shape.mass_properties(1.0))
            .with_total_mass(mass);
        self.spawn(BodySpec { mass: properties.mass, inertia: properties.inertia, ..spec })
    }

    /// Add a joint. Slots come from [`RigidBodyStore::slot_of`].
    pub fn add_joint(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    /// The joints in this world.
    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// The bodies.
    pub fn bodies(&self) -> &RigidBodyStore {
        &self.bodies
    }

    /// Mutable access to the bodies, for setting up or steering a scene.
    pub fn bodies_mut(&mut self) -> &mut RigidBodyStore {
        &mut self.bodies
    }

    /// The contacts found on the last step.
    ///
    /// Published so a viewer can draw them. §17 asks for constraints and forces to be
    /// visible, and a rigid module whose contacts cannot be seen fails the standard
    /// this milestone is held to.
    pub fn contacts(&self) -> &[Contact] {
        &self.contacts
    }

    /// What the last contact solve achieved.
    pub fn last_solve(&self) -> SolveReport {
        self.last_report
    }

    /// The slot a handle currently resolves to.
    pub fn slot_of(&self, id: BodyId) -> Option<usize> {
        self.bodies.slot_of(id)
    }

    /// Forget contact and joint impulses. Use after teleporting bodies, where last
    /// step's impulses describe geometry that no longer exists.
    pub fn reset_solvers(&mut self) {
        self.contact_solver.reset();
        self.joint_solver.reset();
        self.correction_drift = 0.0;
    }

    /// The world-space bounding box of every body, for a viewer.
    pub fn bounds(&self) -> crate::shape::Aabb {
        world_bounds(&self.bodies, shape_lookup(&self.bodies, &self.colliders))
    }
}

/// The contract, which is the same for every configuration of this solver.
///
/// Only one entry, because unlike the heat module the choice of integrator is not the
/// user's here: a sequential-impulse contact solver assumes semi-implicit Euler, and
/// offering velocity Verlet alongside it would be offering something that does not work.
static CONTRACT: SolverContract = SolverContract {
    name: "rigid2d[sequential_impulse]",
    summary: "2D rigid bodies with contacts, friction, restitution and joints",
    governing_equations: &[
        "m dv/dt = F,  I dw/dt = T",
        "contact:  n . (v_b + w_b x r_b - v_a - w_a x r_a) >= 0,  lambda_n >= 0",
        "friction: |lambda_t| <= mu lambda_n  (Coulomb cone)",
        "restitution: v_after . n = -e (v_before . n)",
    ],
    discretization: "none in space — bodies are rigid. Contacts are found by sweep-and-prune \
                     broadphase and separating-axis narrowphase, giving at most two points \
                     per convex pair",
    integrator: "semi-implicit (symplectic) Euler, with contacts and joints resolved \
                 between the velocity and position updates",
    assumptions: &[
        "bodies are rigid and convex; concave geometry must be built from several bodies",
        "collision detection is discrete, so a body may pass through a thin obstacle in \
         one step if it moves further than the obstacle is thick — there is no \
         continuous collision detection (spec 11.1 lists it under 'later')",
        "the pair coefficient of restitution is the maximum of the two surfaces and the \
         pair friction their geometric mean; neither follows from first principles, and \
         no engine stores the N-by-N table that would",
        "friction is Coulomb with a single coefficient: no distinction between static \
         and kinetic, and no velocity dependence",
        "mass is areal, kg/m-squared; this is a 2D world and giving it a thickness would \
         be a conversion nobody declared (see spec 14.1, where such conversions belong \
         to a port)",
        "a spring joint is a force law, not a constraint, so it imposes a timestep limit \
         where the hard joints do not",
    ],
    valid_regime: "interactive and small-scale quantitative scenes: hundreds to a few \
                   thousand bodies, contacts that persist for several steps, and impact \
                   speeds below (body thickness / timestep). Fast thin projectiles and \
                   large stiff stacks are outside it",
    stability: "the integrator is symplectic and imposes no step limit of its own; a \
                spring joint does, at dt < 1/sqrt(k/m). The practical limit is \
                geometric rather than numerical: a body must not move further in one \
                step than the thinnest obstacle it could pass through",
    conserves: &[Invariant::MomentumX, Invariant::MomentumY],
    known_non_conservation: &[
        "energy is not conserved and is not meant to be: restitution below 1 removes it \
         on purpose, and Coulomb friction removes it as heat that nothing here accounts \
         for. A scene with restitution 1 and no friction does conserve energy, and the \
         validation suite checks that case specifically",
        "angular momentum is exact under the impulse solve — every impulse is applied \
         equal and opposite at a shared point — but position correction perturbs it. \
         Displacing a body without changing its velocity changes its orbital angular \
         momentum m(r x v) about any fixed origin, by dt(J x dv) per correction \
         impulse. No position-correction scheme avoids this; the amount is measured \
         every step and published as `correction_drift`",
        "the contact solve is iterative (projected Gauss-Seidel), so constraints hold to \
         whatever the iteration count buys rather than exactly. A tall stack visibly \
         sags at four iterations and holds at twenty; the residual is published so an \
         under-converged run says so",
        "bodies are allowed to overlap by a millimetre of slop, and deeper under load. \
         Driving overlap to zero makes resting bodies jitter, so the overlap is a \
         deliberate dead band rather than a failure to converge",
        "a rope joint going slack discards its accumulated impulse, so momentum is \
         conserved but the transition is not time-reversible",
    ],
    fidelity: FidelityProfile::Interactive,
    precisions: &[Precision::Accurate64],
    deterministic: true,
    differentiable: false,
    validation_cases: &[
        "elastic head-on collision: velocities exchange, energy and momentum conserved",
        "inelastic collision: the energy lost is exactly what momentum conservation predicts",
        "simple pendulum on a pin joint against the analytic small-angle period",
        "a block on a slope holds when mu > tan(theta) and slides when it does not",
        "linear momentum exact at any iteration count, including one",
        "a box comes to rest on the ground within the slop and stays there",
        "restitution controls rebound height monotonically",
        "a spring oscillates at sqrt(k/m)",
    ],
    references: &[
        "Catto, E. (2006). Sequential Impulses. GDC.",
        "Baraff, D. (1997). An Introduction to Physically Based Modeling: Rigid Body \
         Simulation. SIGGRAPH course notes.",
        "Erleben, K. (2005). Stable, Robust, and Versatile Multibody Dynamics Animation. \
         PhD thesis, University of Copenhagen.",
    ],
};

impl Domain for RigidDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn contract(&self) -> &'static SolverContract {
        &CONTRACT
    }

    fn stable_step(&self) -> StableStep {
        // The integrator itself is unconditional. Springs are not, and a body that
        // moves further in one step than an obstacle is thick will pass through it —
        // a limit that is geometric rather than numerical but no less real.
        let mut limit = StableStep::unconditional(self.preferred_dt);
        for joint in &self.joints {
            let joint_limit = joint.stable_step(&self.bodies);
            if joint_limit.is_finite() {
                limit = limit.tightest(StableStep::limited(
                    self.preferred_dt,
                    joint_limit,
                    StabilityReason::OscillationPeriod,
                ));
            }
        }

        // The tunnelling limit: nothing may cross more than the smallest body's own
        // extent in a step. Reported as a stability limit because that is how it
        // presents — a simulation that quietly leaks bodies through walls.
        if let Some(smallest) = self
            .colliders
            .iter()
            .map(|c| c.shape.circumradius())
            .fold(None, |best: Option<f64>, r| Some(best.map_or(r, |b| b.min(r))))
        {
            let speed = self.bodies.max_speed();
            if speed > 0.0 && smallest.is_finite() && smallest > 0.0 {
                limit = limit.tightest(StableStep::limited(
                    self.preferred_dt,
                    smallest / speed,
                    StabilityReason::NeighborSkin,
                ));
            }
        }
        limit
    }

    fn prepare(&mut self, _ctx: &mut StepContext<'_>) {
        self.bodies.clear_forces();
        for slot in 0..self.bodies.len() {
            let mass = self.bodies.mass()[slot];
            self.bodies.add_force(slot, [self.gravity[0] * mass, self.gravity[1] * mass]);
        }
        self.joint_solver.apply_spring_forces(&mut self.bodies, &self.joints);
        self.prepared = true;
    }

    fn advance(&mut self, dt: f64, ctx: &mut StepContext<'_>) {
        // A caller stepping the domain directly, as the doctest above does, never calls
        // `prepare`, and is still entitled to expect gravity. Guessing from the force
        // accumulators would misfire in a scene whose forces happen to cancel, so the
        // flag says so outright.
        if !self.prepared {
            self.prepare(ctx);
        }
        self.prepared = false;

        self.bodies.integrate_velocities(dt);

        let pairs = self
            .broad
            .find_pairs(&self.bodies, shape_lookup(&self.bodies, &self.colliders))
            .to_vec();
        generate_contacts(&self.bodies, &self.colliders, &pairs, &mut self.contacts);

        // Joints and contacts are solved in ONE interleaved sweep, not one after the
        // other. A pendulum resting against a wall is held by both, and neither is
        // right on its own: solving the joints to convergence and then the contacts
        // just alternates between the two answers. Interleaving lets them negotiate.
        self.contact_solver.prepare(&self.bodies, &self.contacts);
        self.joint_solver.prepare(&self.bodies, &self.joints, true);
        self.contact_solver.warm_start(&mut self.bodies);
        self.joint_solver.warm_start(&mut self.bodies, &self.joints);

        let iterations = self.contact_solver.config().velocity_iterations;
        let mut residual = 0.0;
        for iteration in 0..iterations {
            let contact_residual = self.contact_solver.solve_velocities(&mut self.bodies);
            let joint_residual =
                self.joint_solver.solve_velocities(&mut self.bodies, &self.joints, dt);
            if iteration + 1 == iterations {
                residual = contact_residual.max(joint_residual);
            }
        }

        let (penetration, angular_drift) =
            self.contact_solver.correct_positions(&mut self.bodies, dt);
        let points = self.contact_solver.finish();

        self.bodies.integrate_positions(dt);

        self.last_joint_error = self.joint_solver.max_error(&self.joints);
        self.last_report =
            SolveReport { points, iterations, residual, penetration, angular_drift };
        self.correction_drift += angular_drift;
        self.steps += 1;
    }

    fn observe(&self, out: &mut Observations) {
        let prefix = &self.name;
        let momentum = self.bodies.linear_momentum();

        out.record_metric(format!("{prefix}.count"), self.bodies.len() as f64, "1");
        out.record_invariant(
            format!("{prefix}.kinetic_energy"),
            Invariant::KineticEnergy,
            self.bodies.kinetic_energy(),
        );
        out.record_invariant(format!("{prefix}.momentum_x"), Invariant::MomentumX, momentum[0]);
        out.record_invariant(format!("{prefix}.momentum_y"), Invariant::MomentumY, momentum[1]);
        out.record_metric(
            format!("{prefix}.momentum_scale"),
            self.bodies.momentum_scale(),
            Invariant::MomentumX.si_unit(),
        );
        out.record_invariant(
            format!("{prefix}.angular_momentum"),
            Invariant::AngularMomentum,
            self.bodies.angular_momentum(),
        );
        // The declared non-conservation, measured rather than described. A reader can
        // see how much angular momentum position correction actually spent.
        out.record_metric(
            format!("{prefix}.correction_drift"),
            self.correction_drift,
            Invariant::AngularMomentum.si_unit(),
        );

        out.record_metric(format!("{prefix}.contacts"), self.last_report.points as f64, "1");
        out.record(
            format!("{prefix}.solver_iterations"),
            self.last_report.iterations as f64,
            "1",
            ObservationKind::Count,
        );
        out.record(
            format!("{prefix}.contact_residual"),
            self.last_report.residual,
            "m/s",
            ObservationKind::Residual,
        );
        out.record_metric(format!("{prefix}.penetration"), self.last_report.penetration, "m");
        out.record(
            format!("{prefix}.joint_error"),
            self.last_joint_error,
            "m",
            ObservationKind::Residual,
        );
        out.record_metric(format!("{prefix}.max_speed"), self.bodies.max_speed(), "m/s");
        out.record_metric(
            format!("{prefix}.broadphase_tests"),
            self.broad.tests_performed() as f64,
            "1",
        );
    }
}

/// How much overlap counts as a problem worth reporting, m.
///
/// Ten times the slop: a resting stack sits at one or two slops, so a threshold at the
/// slop itself would flag every healthy scene.
pub const PENETRATION_ALARM: f64 = 10.0 * LINEAR_SLOP;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec2;
    use lattice_ir::Arena;

    const DT: f64 = 1.0 / 240.0;

    fn stepped(world: &mut RigidDomain, steps: usize) {
        let mut arena = Arena::with_capacity(4096);
        for _ in 0..steps {
            let mut ctx = StepContext::new(&mut arena);
            world.prepare(&mut ctx);
            world.advance(DT, &mut ctx);
        }
    }

    fn ground_and_crate() -> RigidDomain {
        let mut world = RigidDomain::new("scene", 8);
        let ground = world.register(Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(0.5));
        let crate_ = world.register(Collider::new(Shape::rectangle(0.5, 0.5).unwrap()).with_friction(0.5));
        world.spawn(BodySpec::statik([0.0, 0.0], ground));
        world.spawn_with_density(BodySpec::at([0.0, 3.0], crate_), 2.0);
        world
    }

    /// The project's standing rule: a solver that claims to conserve everything is a
    /// solver nobody has audited. The validation lab enforces this too, but failing
    /// here points at the module rather than at the lab.
    #[test]
    fn the_contract_is_complete_and_admits_what_it_does_not_conserve() {
        let gaps = CONTRACT.audit();
        assert!(gaps.is_empty(), "{gaps:?}");
        assert!(
            !CONTRACT.known_non_conservation.is_empty(),
            "a contact solver that claims to conserve everything has not been audited"
        );
        assert!(
            !CONTRACT.conserves.contains(&Invariant::Energy),
            "restitution and friction remove energy on purpose; claiming otherwise would be a lie"
        );
        assert!(CONTRACT.conserves.contains(&Invariant::MomentumX));
    }

    /// Density gives mass, and mass gives the inertia the shape implies.
    #[test]
    fn density_produces_the_mass_and_inertia_the_shape_implies() {
        let mut world = RigidDomain::new("scene", 4);
        let boxy = world.register(Collider::new(Shape::rectangle(1.0, 0.5).unwrap()));
        let id = world.spawn_with_density(BodySpec::at([0.0, 0.0], boxy), 3.0).unwrap();
        let slot = world.slot_of(id).unwrap();

        // Area 2 x 1 = 2 m^2 at 3 kg/m^2.
        assert!((world.bodies().mass()[slot] - 6.0).abs() < 1e-12);
        // I = m(a^2 + b^2)/3 with half-extents 1 and 0.5.
        let expected = 6.0 * (1.0 + 0.25) / 3.0;
        assert!((world.bodies().inertia()[slot] - expected).abs() < 1e-12);

        // And an explicit total mass rescales it consistently.
        let light = world.spawn_with_mass(BodySpec::at([5.0, 0.0], boxy), 1.0).unwrap();
        let slot = world.slot_of(light).unwrap();
        assert!((world.bodies().mass()[slot] - 1.0).abs() < 1e-12);
        assert!((world.bodies().inertia()[slot] - expected / 6.0).abs() < 1e-12);
    }

    /// A segment has no area, so density gives it nothing and it comes out static
    /// rather than as a body with zero mass that the solver would divide by.
    #[test]
    fn a_segment_given_a_density_comes_out_static() {
        let mut world = RigidDomain::new("scene", 2);
        let wall = world.register(Collider::new(Shape::segment(5.0).unwrap()));
        let id = world.spawn_with_density(BodySpec::at([0.0, 0.0], wall), 1000.0).unwrap();
        assert!(world.bodies().is_static(world.slot_of(id).unwrap()));
    }

    #[test]
    fn a_crate_falls_and_rests_on_the_ground() {
        let mut world = ground_and_crate();
        stepped(&mut world, 2000);

        let y = world.bodies().pos_y()[1];
        assert!((y - 1.0).abs() < 5.0 * LINEAR_SLOP, "resting at {y}, expected about 1.0");
        assert!(world.last_solve().points > 0, "it should be in contact");
        assert!(world.last_solve().penetration < PENETRATION_ALARM);
    }

    /// Everything the panel needs, present and dimensionally labelled.
    #[test]
    fn the_domain_publishes_what_a_reader_needs_to_judge_it() {
        let mut world = ground_and_crate();
        stepped(&mut world, 1200);

        let mut out = Observations::new();
        world.observe(&mut out);

        for name in [
            "scene.count",
            "scene.kinetic_energy",
            "scene.momentum_x",
            "scene.momentum_y",
            "scene.momentum_scale",
            "scene.angular_momentum",
            "scene.correction_drift",
            "scene.contacts",
            "scene.solver_iterations",
            "scene.contact_residual",
            "scene.penetration",
            "scene.joint_error",
            "scene.max_speed",
            "scene.broadphase_tests",
        ] {
            assert!(out.get(name).is_some(), "missing {name}");
        }

        assert_eq!(out.value("scene.count"), Some(2.0));
        assert_eq!(out.get("scene.momentum_x").unwrap().unit, "kg·m/s");
        assert_eq!(out.get("scene.penetration").unwrap().unit, "m");
        assert_eq!(out.get("scene.contact_residual").unwrap().unit, "m/s");
        // The momentum scale is what makes a net momentum of zero judgeable.
        assert_eq!(out.get("scene.momentum_scale").unwrap().unit, "kg·m/s");
        assert!(matches!(
            out.get("scene.contact_residual").unwrap().kind,
            ObservationKind::Residual
        ));
    }

    /// The declared non-conservation is measured, not described.
    #[test]
    fn the_correction_drift_is_reported_and_starts_at_zero() {
        let mut world = RigidDomain::new("scene", 4);
        let disc = world.register(Collider::new(Shape::circle(0.5).unwrap()));
        world.spawn_with_density(BodySpec::at([0.0, 0.0], disc), 1.0);
        world.spawn_with_density(BodySpec::at([5.0, 0.0], disc), 1.0);

        let mut out = Observations::new();
        world.observe(&mut out);
        assert_eq!(out.value("scene.correction_drift"), Some(0.0), "nothing has overlapped yet");

        stepped(&mut world, 100);
        let mut out = Observations::new();
        world.observe(&mut out);
        assert_eq!(
            out.value("scene.correction_drift"),
            Some(0.0),
            "two bodies in free fall never overlap, so nothing was spent"
        );
    }

    /// A stiff spring imposes a timestep limit, and the domain must surface it rather
    /// than letting the runtime pick a step that explodes.
    #[test]
    fn a_spring_tightens_the_domains_stable_step() {
        let mut world = RigidDomain::new("scene", 4).with_preferred_step(1.0 / 60.0);
        let disc = world.register(Collider::new(Shape::circle(0.2).unwrap()));
        world.spawn(BodySpec::statik([0.0, 0.0], disc));
        world.spawn_with_mass(BodySpec::at([0.0, -1.0], disc), 1.0);

        assert!(world.stable_step().max.is_infinite(), "no joints, no limit");

        world.add_joint(Joint::Spring {
            a: 0,
            b: 1,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            stiffness: 40_000.0,
            damping: 1.0,
        });
        let limit = world.stable_step();
        assert!(limit.max.is_finite(), "a stiff spring has a limit");
        // 1/sqrt(k/m) with k = 40000, m = 1.
        assert!((limit.max - 0.005).abs() < 1e-9, "{}", limit.max);
        assert!(!limit.admits(1.0 / 60.0), "the preferred step is too big for this spring");
        assert_eq!(limit.reason, StabilityReason::OscillationPeriod);
    }

    /// Discrete collision detection means a body must not out-run the geometry. The
    /// domain reports that as a stability limit because that is how it presents: a
    /// simulation that quietly leaks bodies through walls.
    #[test]
    fn a_fast_body_tightens_the_stable_step_before_it_tunnels() {
        let mut world = RigidDomain::new("scene", 4);
        let small = world.register(Collider::new(Shape::circle(0.05).unwrap()));
        world.spawn_with_mass(BodySpec::at([0.0, 0.0], small).with_velocity([100.0, 0.0]), 1.0);

        let limit = world.stable_step();
        // 0.05 m of body at 100 m/s.
        assert!((limit.max - 5e-4).abs() < 1e-12, "{}", limit.max);
        assert_eq!(limit.reason, StabilityReason::NeighborSkin);
    }

    /// A pendulum built from the domain's own pieces swings, and the pin holds.
    #[test]
    fn a_pendulum_assembled_through_the_domain_holds_together() {
        let mut world = RigidDomain::new("scene", 4);
        let pivot = world.register(Collider::new(Shape::circle(0.05).unwrap()));
        let bob = world.register(Collider::new(Shape::circle(0.1).unwrap()));
        world.spawn(BodySpec::statik([0.0, 0.0], pivot));
        world.spawn_with_mass(BodySpec::at([1.0, 0.0], bob), 1.0);
        world.add_joint(Joint::Pin {
            a: 0,
            b: 1,
            local_a: Vec2::ZERO,
            local_b: Vec2::from([-1.0, 0.0]),
        });

        // Sampling the final position would prove nothing: after sixteen seconds a
        // one-metre pendulum has swung seven times and could be anywhere. Track the
        // extremes instead.
        let mut arena = Arena::with_capacity(4096);
        let (mut lowest, mut highest, mut worst_radius) = (0.0f64, -1.0f64, 0.0f64);
        for _ in 0..4000 {
            let mut ctx = StepContext::new(&mut arena);
            world.prepare(&mut ctx);
            world.advance(DT, &mut ctx);
            let (x, y) = (world.bodies().pos_x()[1], world.bodies().pos_y()[1]);
            lowest = lowest.min(y);
            highest = highest.max(y);
            worst_radius = worst_radius.max(((x * x + y * y).sqrt() - 1.0).abs());
        }

        assert!(worst_radius < 1e-2, "the pin let go: worst radius error {worst_radius} m");
        // Released horizontally, it must reach the bottom of its arc...
        assert!(lowest < -0.98, "it only got down to {lowest}");
        // ...and swing back up nearly to where it started, because a pin joint and a
        // symplectic integrator between them lose very little.
        assert!(highest > -0.05, "it only came back up to {highest}");
    }

    #[test]
    fn the_world_bounds_cover_every_body() {
        let world = ground_and_crate();
        let bounds = world.bounds();
        assert!(bounds.min.x <= -10.0 && bounds.max.x >= 10.0);
        assert!(bounds.max.y >= 3.5, "the crate starts at y = 3: {bounds:?}");
    }

    #[test]
    fn an_empty_world_steps_and_reports_without_panicking() {
        let mut world = RigidDomain::new("empty", 4);
        stepped(&mut world, 10);
        let mut out = Observations::new();
        world.observe(&mut out);
        assert_eq!(out.value("empty.count"), Some(0.0));
        assert_eq!(out.value("empty.kinetic_energy"), Some(0.0));
        assert!(world.stable_step().max.is_infinite());
        assert!(world.bounds().is_empty());
    }

    /// FR-011: two identical worlds must stay identical.
    #[test]
    fn a_scene_is_reproducible() {
        let run = || {
            let mut world = RigidDomain::new("scene", 12);
            let ground =
                world.register(Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(0.4));
            let disc = world.register(
                Collider::new(Shape::circle(0.3).unwrap()).with_restitution(0.4).with_friction(0.4),
            );
            world.spawn(BodySpec::statik([0.0, 0.0], ground));
            for i in 0..8 {
                world.spawn_with_density(
                    BodySpec::at([f64::from(i) * 0.31 - 1.0, 1.5 + f64::from(i) * 0.8], disc)
                        .with_velocity([0.3 * f64::from(i % 3) - 0.3, 0.0]),
                    1.0,
                );
            }
            stepped(&mut world, 800);
            (world.bodies().pos_x().to_vec(), world.bodies().pos_y().to_vec())
        };
        assert_eq!(run(), run());
    }

    /// Stepping the domain directly, without the runtime calling `prepare`, must still
    /// apply gravity — and calling `prepare` must not apply it twice.
    #[test]
    fn gravity_is_applied_exactly_once_whether_or_not_prepare_was_called() {
        let fall = |call_prepare: bool| {
            let mut world = RigidDomain::new("scene", 2);
            let disc = world.register(Collider::new(Shape::circle(0.2).unwrap()));
            world.spawn_with_mass(BodySpec::at([0.0, 0.0], disc), 1.0);

            let mut arena = Arena::with_capacity(1024);
            for _ in 0..240 {
                let mut ctx = StepContext::new(&mut arena);
                if call_prepare {
                    world.prepare(&mut ctx);
                }
                world.advance(DT, &mut ctx);
            }
            world.bodies().pos_y()[0]
        };
        let with_prepare = fall(true);
        let without = fall(false);
        assert!((with_prepare - without).abs() < 1e-12, "{with_prepare} vs {without}");
        // One second of free fall from rest is about -4.9 m.
        assert!((with_prepare + 4.9).abs() < 0.1, "fell {with_prepare} m in one second");
    }
}
