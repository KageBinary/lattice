//! The sequential-impulse contact solver.
//!
//! Each contact point carries two constraints: the bodies must not approach along the
//! normal, and they must not slide faster than friction allows along the tangent. The
//! solver visits them in turn, computes the impulse that would satisfy the one it is
//! looking at, applies it immediately, and repeats — projected Gauss–Seidel. Each pass
//! disturbs the constraints already satisfied, so it takes several passes to converge,
//! and it never converges exactly.
//!
//! # The three honest caveats
//!
//! **It is iterative.** [`SolverConfig::velocity_iterations`] buys accuracy. A tall
//! stack visibly sags at four and holds at twenty. [`SolveReport::residual`] publishes
//! the largest constraint violation left when the passes ran out, so a run that needed
//! more says so rather than merely looking wrong.
//!
//! **Restitution removes energy on purpose.** A coefficient below 1 is what "inelastic"
//! means. Energy is not conserved here and is not claimed to be.
//!
//! **Position correction adds energy.** Pushing overlapping bodies apart does work on
//! them. This solver corrects position by [pseudo-velocities]: a separate velocity
//! accumulator that moves the bodies and is then discarded, so the correction never
//! becomes momentum the bodies keep. That is the difference between a stack settling
//! and a stack launching itself — but it is still not free, and the contract says so.
//!
//! # What *is* exact
//!
//! Every impulse is applied equal and opposite to the pair, so **linear momentum is
//! conserved to round-off** regardless of how few iterations run. An under-converged
//! solve looks like bodies sinking into each other; it never looks like momentum
//! appearing from nowhere. That is the same argument the particle module's pair forces
//! rest on, and it is why momentum is the invariant this module publishes with
//! confidence and energy is not.
//!
//! **Angular momentum is exact under the impulse solve** — an equal and opposite pair
//! applied at a shared point changes the total by `P × J + P × (−J) = 0` — but *not*
//! under position correction. Displacing a body without changing its velocity changes
//! its orbital angular momentum `m (r × v)` about any fixed origin, by
//! `dt · (J_p × Δv)` per correction impulse. No position-correction scheme avoids this;
//! it is the price of separating bodies that a finite timestep let overlap. The
//! contract declares it, and [`SolveReport::angular_drift`] measures it every step so a
//! reader can see how much was spent rather than take the word "small" on trust.
//!
//! [pseudo-velocities]: https://box2d.org/files/ErinCatto_SequentialImpulses_GDC2006.pdf

use lattice_ir::RigidBodyStore;

use crate::contact::{BodyPair, Contact, FeatureId, MAX_MANIFOLD_POINTS};
use crate::math::{vec2, Vec2};
use crate::narrowphase::LINEAR_SLOP;

/// How the solver is tuned.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SolverConfig {
    /// Passes over the velocity constraints.
    pub velocity_iterations: usize,
    /// Passes over the position correction.
    pub position_iterations: usize,
    /// Fraction of the excess overlap corrected per position pass, 0 to 1.
    ///
    /// Correcting all of it at once overshoots and makes resting bodies pop. Around a
    /// fifth per pass, over three passes, removes most of the overlap without ringing.
    pub position_correction: f64,
    /// Cap on how fast position correction may push bodies apart, m/s.
    ///
    /// Without it, a body that has somehow ended up deeply overlapped — spawned inside
    /// a wall, say — is fired across the scene. Clamping turns a modelling mistake into
    /// a body that oozes out over a few frames, which is both survivable and visibly
    /// diagnosable.
    pub max_correction_speed: f64,
    /// Approach speed below which restitution is ignored, m/s.
    ///
    /// A resting body is in contact with a tiny approach velocity every step from
    /// gravity. Applying restitution to that makes it hum in place forever. Below this
    /// threshold a collision is treated as inelastic whatever the material says.
    pub restitution_threshold: f64,
    /// Reuse last step's impulses as this step's starting guess.
    ///
    /// The single largest accuracy win available: a stack that needs forty cold
    /// iterations holds at four warm ones, because the previous answer is nearly right.
    pub warm_starting: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            velocity_iterations: 8,
            position_iterations: 3,
            position_correction: 0.2,
            max_correction_speed: 3.0,
            restitution_threshold: 1.0,
            warm_starting: true,
        }
    }
}

impl SolverConfig {
    /// Enough iterations that the remaining error is round-off rather than budget.
    ///
    /// For validation cases, where the question is whether the *physics* is right and
    /// an under-converged solve would be a confounding variable.
    pub fn accurate() -> SolverConfig {
        SolverConfig { velocity_iterations: 64, position_iterations: 12, ..SolverConfig::default() }
    }
}

/// What the solve achieved.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct SolveReport {
    /// Contact points solved.
    pub points: usize,
    /// Velocity passes performed.
    pub iterations: usize,
    /// Largest remaining approach speed across all contacts, m/s.
    ///
    /// Zero means every contact is separating or resting. A positive value means bodies
    /// are still moving into each other when the passes ran out — the honest measure of
    /// whether the iteration budget was enough.
    pub residual: f64,
    /// Deepest overlap remaining after position correction, m.
    pub penetration: f64,
    /// Angular momentum about the world origin injected by position correction,
    /// kg·m²/s.
    ///
    /// Zero whenever no overlap needed correcting, which is the resting case. See the
    /// module documentation for why it cannot be zero in general.
    pub angular_drift: f64,
}

/// One scalar constraint: a direction, a lever arm on each body, and the accumulated
/// impulse along it.
#[derive(Clone, Copy, Debug)]
struct PointConstraint {
    /// Lever arm from body A's centre of mass to the contact point.
    arm_a: Vec2,
    /// Lever arm from body B's centre of mass to the contact point.
    arm_b: Vec2,
    /// `1 / (nᵀ K n)` along the normal — the effective mass the pair presents.
    normal_mass: f64,
    /// The same along the tangent.
    tangent_mass: f64,
    /// Target approach speed, negative for a bounce.
    bounce: f64,
    /// Total normal impulse applied so far this step, N·s. Never negative: a contact
    /// can push, never pull.
    normal_impulse: f64,
    /// Total tangential impulse so far, N·s. Bounded by `friction · normal_impulse`.
    tangent_impulse: f64,
    /// Overlap at the start of the step, m.
    penetration: f64,
    /// Which features produced this point, for warm starting.
    feature: FeatureId,
}

/// A contact prepared for solving.
#[derive(Clone, Copy, Debug)]
struct ContactConstraint {
    pair: BodyPair,
    normal: Vec2,
    tangent: Vec2,
    friction: f64,
    points: [PointConstraint; MAX_MANIFOLD_POINTS],
    count: usize,
}

/// Impulses carried from one step to the next.
///
/// Keyed by `(pair, feature)`: the same corner on the same face keeps its impulse even
/// as the contact point slides, and a contact that changes features starts cold rather
/// than inheriting an impulse computed for different geometry.
#[derive(Clone, Copy, PartialEq, Debug)]
struct CachedImpulse {
    pair: BodyPair,
    feature: FeatureId,
    normal: f64,
    tangent: f64,
}

/// The contact solver and its buffers.
///
/// Holds its working set across steps so a settled scene performs no allocation
/// (NFR-001), and holds the warm-start cache, which is what makes a modest iteration
/// count sufficient.
#[derive(Clone, Default, Debug)]
pub struct ContactSolver {
    config: SolverConfig,
    constraints: Vec<ContactConstraint>,
    /// Last step's impulses, sorted by `(pair, feature)` for a binary search.
    cache: Vec<CachedImpulse>,
    /// This step's impulses, becoming `cache` at the end of the step.
    next_cache: Vec<CachedImpulse>,
    /// Position-correction velocities, discarded after the position passes.
    pseudo_vel: Vec<[f64; 2]>,
    pseudo_omega: Vec<f64>,
}

impl ContactSolver {
    /// A solver with the given configuration.
    pub fn new(config: SolverConfig) -> ContactSolver {
        ContactSolver { config, ..ContactSolver::default() }
    }

    /// The configuration in force.
    pub fn config(&self) -> SolverConfig {
        self.config
    }

    /// Replace the configuration.
    pub fn set_config(&mut self, config: SolverConfig) {
        self.config = config;
    }

    /// Discard the warm-start cache. Use after teleporting bodies or resetting a scene,
    /// where last step's impulses describe geometry that no longer exists.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.next_cache.clear();
        self.constraints.clear();
    }

    /// Resolve `contacts`, modifying the bodies' velocities and positions.
    ///
    /// Call after integrating velocities from forces and *before* integrating
    /// positions — that ordering is what makes the scheme semi-implicit, and it is why
    /// [`RigidBodyStore::integrate`] is split into halves.
    ///
    /// The whole step in one call, for a scene with no joints. A domain that has joints
    /// drives [`ContactSolver::prepare`], [`ContactSolver::warm_start`],
    /// [`ContactSolver::solve_velocities`], [`ContactSolver::correct_positions`] and
    /// [`ContactSolver::finish`] itself, so it can interleave the two constraint sets in
    /// a single sweep instead of alternating between their separate answers.
    pub fn solve(
        &mut self,
        bodies: &mut RigidBodyStore,
        contacts: &[Contact],
        dt: f64,
    ) -> SolveReport {
        self.prepare(bodies, contacts);
        if self.is_empty() {
            return SolveReport::default();
        }
        self.warm_start(bodies);

        let mut residual = 0.0;
        for iteration in 0..self.config.velocity_iterations {
            let worst = self.solve_velocities(bodies);
            // The residual reported is the one left *after* the final pass, which is
            // what a reader wants to know. Earlier passes are expected to be poor.
            if iteration + 1 == self.config.velocity_iterations {
                residual = worst;
            }
        }

        let (penetration, angular_drift) = self.correct_positions(bodies, dt);
        SolveReport {
            points: self.finish(),
            iterations: self.config.velocity_iterations,
            residual,
            penetration,
            angular_drift,
        }
    }

    /// True when there is nothing to solve.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Contact points prepared for this step.
    pub fn point_count(&self) -> usize {
        self.constraints.iter().map(|c| c.count).sum()
    }

    /// Build the per-point constraint data: lever arms, effective masses, and the
    /// restitution target, all of which are constant across the iterations.
    ///
    /// The first phase of a step. Call after integrating velocities from forces.
    pub fn prepare(&mut self, bodies: &RigidBodyStore, contacts: &[Contact]) {
        self.constraints.clear();
        self.next_cache.clear();

        for contact in contacts {
            let (a, b) = (contact.pair.a, contact.pair.b);
            if a >= bodies.len() || b >= bodies.len() {
                continue;
            }
            let normal = contact.manifold.normal;
            let tangent = normal.perp();
            let (inv_ma, inv_mb) = (bodies.inv_mass()[a], bodies.inv_mass()[b]);
            let (inv_ia, inv_ib) = (bodies.inv_inertia()[a], bodies.inv_inertia()[b]);

            let mut constraint = ContactConstraint {
                pair: contact.pair,
                normal,
                tangent,
                friction: contact.friction,
                points: [PointConstraint {
                    arm_a: Vec2::ZERO,
                    arm_b: Vec2::ZERO,
                    normal_mass: 0.0,
                    tangent_mass: 0.0,
                    bounce: 0.0,
                    normal_impulse: 0.0,
                    tangent_impulse: 0.0,
                    penetration: 0.0,
                    feature: FeatureId::SINGLE,
                }; MAX_MANIFOLD_POINTS],
                count: 0,
            };

            for point in contact.manifold.points() {
                let arm_a = point.position - vec2(bodies.pos_x()[a], bodies.pos_y()[a]);
                let arm_b = point.position - vec2(bodies.pos_x()[b], bodies.pos_y()[b]);

                let normal_mass =
                    effective_mass(normal, arm_a, arm_b, inv_ma, inv_mb, inv_ia, inv_ib);
                let tangent_mass =
                    effective_mass(tangent, arm_a, arm_b, inv_ma, inv_mb, inv_ia, inv_ib);
                // Two static bodies, or a pair whose reciprocals are all zero, present
                // infinite mass: no impulse can change anything, so there is nothing to
                // solve and dividing would produce an infinity.
                if normal_mass == 0.0 {
                    continue;
                }

                // Restitution is evaluated on the *approach* speed at the start of the
                // step, before any impulse has been applied — using the current speed
                // inside the loop would let the bounce feed on itself.
                let approach = relative_velocity(bodies, a, b, arm_a, arm_b).dot(normal);
                let bounce = if approach < -self.config.restitution_threshold {
                    -contact.restitution * approach
                } else {
                    0.0
                };

                constraint.points[constraint.count] = PointConstraint {
                    arm_a,
                    arm_b,
                    normal_mass,
                    tangent_mass,
                    bounce,
                    normal_impulse: 0.0,
                    tangent_impulse: 0.0,
                    penetration: point.penetration,
                    feature: point.feature,
                };
                constraint.count += 1;
            }

            if constraint.count > 0 {
                self.constraints.push(constraint);
            }
        }

        if self.config.warm_starting {
            self.seed_from_cache();
        }
    }

    /// Copy last step's impulses onto matching points.
    fn seed_from_cache(&mut self) {
        if self.cache.is_empty() {
            return;
        }
        for constraint in &mut self.constraints {
            for point in &mut constraint.points[..constraint.count] {
                let key = (constraint.pair, point.feature);
                if let Ok(index) = self
                    .cache
                    .binary_search_by(|entry| (entry.pair, entry.feature_key()).cmp(&(key.0, key.1.key())))
                {
                    point.normal_impulse = self.cache[index].normal;
                    point.tangent_impulse = self.cache[index].tangent;
                }
            }
        }
    }

    /// Apply the seeded impulses, so the bodies start the iterations already close to
    /// where they finished last step. A no-op when warm starting is off.
    pub fn warm_start(&mut self, bodies: &mut RigidBodyStore) {
        if !self.config.warm_starting {
            return;
        }
        for constraint in &self.constraints {
            for point in &constraint.points[..constraint.count] {
                let impulse =
                    constraint.normal * point.normal_impulse + constraint.tangent * point.tangent_impulse;
                apply_pair_impulse(bodies, constraint.pair, impulse, point.arm_a, point.arm_b);
            }
        }
    }

    /// One pass over every constraint. Returns the largest remaining approach speed.
    pub fn solve_velocities(&mut self, bodies: &mut RigidBodyStore) -> f64 {
        let mut worst: f64 = 0.0;
        for constraint in &mut self.constraints {
            let (a, b) = (constraint.pair.a, constraint.pair.b);

            for index in 0..constraint.count {
                // Friction first, using the normal impulse accumulated so far. Solving
                // it after the normal impulse in the same pass would bound it by a
                // value from the previous pass, which under-constrains sliding on the
                // first pass and shows up as a box skating before it grips.
                let point = constraint.points[index];
                let relative =
                    relative_velocity(bodies, a, b, point.arm_a, point.arm_b);

                let sliding = relative.dot(constraint.tangent);
                let mut tangent_impulse = -sliding * point.tangent_mass;
                let limit = constraint.friction * point.normal_impulse;
                let clamped = (point.tangent_impulse + tangent_impulse).clamp(-limit, limit);
                tangent_impulse = clamped - point.tangent_impulse;
                constraint.points[index].tangent_impulse = clamped;
                apply_pair_impulse(
                    bodies,
                    constraint.pair,
                    constraint.tangent * tangent_impulse,
                    point.arm_a,
                    point.arm_b,
                );

                // The normal constraint: approach speed must reach the bounce target.
                let point = constraint.points[index];
                let relative = relative_velocity(bodies, a, b, point.arm_a, point.arm_b);
                let approach = relative.dot(constraint.normal);
                worst = worst.max(-(approach + point.bounce));

                let mut normal_impulse = -(approach - point.bounce) * point.normal_mass;
                // A contact pushes; it never pulls. Clamping the *accumulated* impulse
                // rather than the increment is what lets a later pass undo an earlier
                // one's overshoot — the difference between this and a naive
                // apply-and-clamp scheme that cannot relax.
                let total = (point.normal_impulse + normal_impulse).max(0.0);
                normal_impulse = total - point.normal_impulse;
                constraint.points[index].normal_impulse = total;
                apply_pair_impulse(
                    bodies,
                    constraint.pair,
                    constraint.normal * normal_impulse,
                    point.arm_a,
                    point.arm_b,
                );
            }
        }
        worst
    }

    /// Push overlapping bodies apart using velocities that are discarded afterwards.
    ///
    /// The bodies move, but they do not *keep* the motion — so a stack settles instead
    /// of launching. Returns the deepest overlap left and the angular momentum the
    /// correction cost, which is the module's one declared non-conservation.
    pub fn correct_positions(&mut self, bodies: &mut RigidBodyStore, dt: f64) -> (f64, f64) {
        let mut deepest: f64 = 0.0;
        for constraint in &self.constraints {
            for point in &constraint.points[..constraint.count] {
                deepest = deepest.max(point.penetration);
            }
        }
        if deepest <= LINEAR_SLOP || self.config.position_iterations == 0 || dt <= 0.0 {
            return (deepest, 0.0);
        }
        let angular_before = bodies.angular_momentum();

        self.pseudo_vel.clear();
        self.pseudo_vel.resize(bodies.len(), [0.0, 0.0]);
        self.pseudo_omega.clear();
        self.pseudo_omega.resize(bodies.len(), 0.0);

        for _ in 0..self.config.position_iterations {
            for constraint in &self.constraints {
                let (a, b) = (constraint.pair.a, constraint.pair.b);
                for point in &constraint.points[..constraint.count] {
                    // Only the overlap beyond the allowed slop is corrected, and only a
                    // fraction of it per pass.
                    let excess = (point.penetration - LINEAR_SLOP).max(0.0);
                    if excess == 0.0 {
                        continue;
                    }
                    let target = (self.config.position_correction * excess / dt)
                        .min(self.config.max_correction_speed);

                    let separating = pseudo_relative(
                        &self.pseudo_vel,
                        &self.pseudo_omega,
                        a,
                        b,
                        point.arm_a,
                        point.arm_b,
                    )
                    .dot(constraint.normal);
                    let impulse = (target - separating).max(0.0) * point.normal_mass;
                    if impulse == 0.0 {
                        continue;
                    }
                    let push = constraint.normal * impulse;
                    apply_pseudo_impulse(
                        bodies,
                        &mut self.pseudo_vel,
                        &mut self.pseudo_omega,
                        constraint.pair,
                        push,
                        point.arm_a,
                        point.arm_b,
                    );
                }
            }
        }

        // Move the bodies by the pseudo-velocities, then throw them away. This is the
        // step that keeps the correction out of the momentum.
        for slot in 0..bodies.len() {
            let v = self.pseudo_vel[slot];
            let w = self.pseudo_omega[slot];
            if v == [0.0, 0.0] && w == 0.0 {
                continue;
            }
            let position = [bodies.pos_x()[slot] + v[0] * dt, bodies.pos_y()[slot] + v[1] * dt];
            let angle = bodies.rot_sin()[slot].atan2(bodies.rot_cos()[slot]) + w * dt;
            let motion = bodies.motion();
            motion.pos_x[slot] = position[0];
            motion.pos_y[slot] = position[1];
            motion.rot_cos[slot] = angle.cos();
            motion.rot_sin[slot] = angle.sin();
        }

        // Report the overlap the correction was aiming at, less what it removed. The
        // exact remaining depth would need a second narrowphase pass, which is not
        // worth a full collision detection cycle for a diagnostic.
        let remaining = (deepest
            - self.config.position_correction * self.config.position_iterations as f64 * deepest)
            .max(0.0);
        (remaining, bodies.angular_momentum() - angular_before)
    }

    /// Save this step's impulses for next step's warm start, and report how many
    /// contact points were solved. The last phase of a step.
    pub fn finish(&mut self) -> usize {
        let points = self.point_count();
        self.store_impulses();
        points
    }

    /// Save this step's impulses for next step's warm start.
    fn store_impulses(&mut self) {
        self.next_cache.clear();
        for constraint in &self.constraints {
            for point in &constraint.points[..constraint.count] {
                self.next_cache.push(CachedImpulse {
                    pair: constraint.pair,
                    feature: point.feature,
                    normal: point.normal_impulse,
                    tangent: point.tangent_impulse,
                });
            }
        }
        self.next_cache.sort_unstable_by_key(|entry| (entry.pair, entry.feature_key()));
        core::mem::swap(&mut self.cache, &mut self.next_cache);
    }

    /// The impulses that will seed the next step, for tests and diagnostics.
    pub fn cached_impulse_count(&self) -> usize {
        self.cache.len()
    }
}

impl CachedImpulse {
    fn feature_key(&self) -> (u8, u8, bool) {
        self.feature.key()
    }
}

impl FeatureId {
    /// A sortable key, so the warm-start cache can be binary-searched.
    fn key(self) -> (u8, u8, bool) {
        (self.reference, self.incident, self.flipped)
    }
}

/// `1 / (dᵀ K d)` — how much mass the pair presents to an impulse along `d` applied at
/// the given lever arms. Zero when the pair cannot move at all.
fn effective_mass(
    direction: Vec2,
    arm_a: Vec2,
    arm_b: Vec2,
    inv_ma: f64,
    inv_mb: f64,
    inv_ia: f64,
    inv_ib: f64,
) -> f64 {
    let cross_a = arm_a.cross(direction);
    let cross_b = arm_b.cross(direction);
    let total = inv_ma + inv_mb + inv_ia * cross_a * cross_a + inv_ib * cross_b * cross_b;
    if total > 0.0 { 1.0 / total } else { 0.0 }
}

/// Velocity of the contact point on B relative to the same point on A.
fn relative_velocity(
    bodies: &RigidBodyStore,
    a: usize,
    b: usize,
    arm_a: Vec2,
    arm_b: Vec2,
) -> Vec2 {
    let va = vec2(bodies.vel_x()[a], bodies.vel_y()[a]) + arm_a.cross_scalar(bodies.omega()[a]);
    let vb = vec2(bodies.vel_x()[b], bodies.vel_y()[b]) + arm_b.cross_scalar(bodies.omega()[b]);
    vb - va
}

/// Apply `impulse` to B and its negation to A.
///
/// Equal and opposite, in one place, so momentum conservation is a property of the code
/// rather than a coincidence of two call sites agreeing.
fn apply_pair_impulse(
    bodies: &mut RigidBodyStore,
    pair: BodyPair,
    impulse: Vec2,
    arm_a: Vec2,
    arm_b: Vec2,
) {
    bodies.apply_impulse_at_arm(pair.a, (-impulse).to_array(), arm_a.to_array());
    bodies.apply_impulse_at_arm(pair.b, impulse.to_array(), arm_b.to_array());
}

/// The same relative-velocity computation, over the pseudo-velocity arrays.
fn pseudo_relative(
    vel: &[[f64; 2]],
    omega: &[f64],
    a: usize,
    b: usize,
    arm_a: Vec2,
    arm_b: Vec2,
) -> Vec2 {
    let va = Vec2::from(vel[a]) + arm_a.cross_scalar(omega[a]);
    let vb = Vec2::from(vel[b]) + arm_b.cross_scalar(omega[b]);
    vb - va
}

/// Apply a correction impulse to the pseudo-velocity arrays only.
fn apply_pseudo_impulse(
    bodies: &RigidBodyStore,
    vel: &mut [[f64; 2]],
    omega: &mut [f64],
    pair: BodyPair,
    impulse: Vec2,
    arm_a: Vec2,
    arm_b: Vec2,
) {
    let (a, b) = (pair.a, pair.b);
    vel[a][0] -= bodies.inv_mass()[a] * impulse.x;
    vel[a][1] -= bodies.inv_mass()[a] * impulse.y;
    omega[a] -= bodies.inv_inertia()[a] * arm_a.cross(impulse);
    vel[b][0] += bodies.inv_mass()[b] * impulse.x;
    vel[b][1] += bodies.inv_mass()[b] * impulse.y;
    omega[b] += bodies.inv_inertia()[b] * arm_b.cross(impulse);
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{BodySpec, ShapeId};

    use crate::broadphase::{shape_lookup, BroadPhase};
    use crate::contact::Manifold;
    use crate::narrowphase::generate_contacts;
    use crate::shape::{Collider, Shape};

    /// A minimal scene: a few registered colliders and the bodies wearing them.
    struct Scene {
        bodies: RigidBodyStore,
        colliders: Vec<Collider>,
        broad: BroadPhase,
        contacts: Vec<Contact>,
        solver: ContactSolver,
    }

    impl Scene {
        fn new(colliders: Vec<Collider>, capacity: usize, config: SolverConfig) -> Scene {
            Scene {
                bodies: RigidBodyStore::with_capacity(capacity),
                colliders,
                broad: BroadPhase::new(),
                contacts: Vec::new(),
                solver: ContactSolver::new(config),
            }
        }

        fn spawn(&mut self, spec: BodySpec) -> usize {
            let id = self.bodies.spawn(spec).expect("scene sized for this");
            self.bodies.slot_of(id).unwrap()
        }

        /// One full step: forces, velocity integration, contact solve, position
        /// integration. The ordering the semi-implicit scheme requires.
        fn step(&mut self, dt: f64, gravity: [f64; 2]) -> SolveReport {
            self.bodies.clear_forces();
            for slot in 0..self.bodies.len() {
                let mass = self.bodies.mass()[slot];
                self.bodies.add_force(slot, [gravity[0] * mass, gravity[1] * mass]);
            }
            self.bodies.integrate_velocities(dt);

            let pairs = self
                .broad
                .find_pairs(&self.bodies, shape_lookup(&self.bodies, &self.colliders))
                .to_vec();
            generate_contacts(&self.bodies, &self.colliders, &pairs, &mut self.contacts);
            let report = self.solver.solve(&mut self.bodies, &self.contacts, dt);

            self.bodies.integrate_positions(dt);
            report
        }
    }

    fn circles(restitution: f64, friction: f64) -> Vec<Collider> {
        vec![Collider::new(Shape::circle(0.5).unwrap())
            .with_restitution(restitution)
            .with_friction(friction)]
    }

    const SHAPE: ShapeId = ShapeId::from_index(0);
    const DT: f64 = 1.0 / 240.0;
    const G: [f64; 2] = [0.0, -9.80665];

    /// Spec 19.2 elastic collision. Two equal discs meeting head on with restitution 1
    /// must exchange velocities exactly, conserving both momentum and kinetic energy.
    #[test]
    fn an_elastic_head_on_collision_exchanges_velocities() {
        let mut scene = Scene::new(circles(1.0, 0.0), 2, SolverConfig::accurate());
        scene.spawn(BodySpec::at([-1.0, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([2.0, 0.0]));
        scene.spawn(BodySpec::at([1.0, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([-2.0, 0.0]));

        let energy_before = scene.bodies.kinetic_energy();
        let momentum_before = scene.bodies.linear_momentum();

        for _ in 0..500 {
            scene.step(DT, [0.0, 0.0]);
        }

        // They came in at +/-2 and must leave at -/+2.
        assert!((scene.bodies.vel_x()[0] + 2.0).abs() < 1e-6, "{:?}", scene.bodies.vel_x());
        assert!((scene.bodies.vel_x()[1] - 2.0).abs() < 1e-6, "{:?}", scene.bodies.vel_x());

        let energy_after = scene.bodies.kinetic_energy();
        let momentum_after = scene.bodies.linear_momentum();
        assert!(
            (energy_after - energy_before).abs() / energy_before < 1e-6,
            "elastic means energy is kept: {energy_before} -> {energy_after}"
        );
        assert!((momentum_after[0] - momentum_before[0]).abs() < 1e-12);
    }

    /// Spec 19.2 inelastic collision. Restitution 0 means the discs end up moving
    /// together at the centre-of-mass velocity, and the kinetic energy lost is exactly
    /// the amount momentum conservation predicts -- not merely "some".
    #[test]
    fn an_inelastic_collision_loses_exactly_the_predicted_energy() {
        let mut scene = Scene::new(circles(0.0, 0.0), 2, SolverConfig::accurate());
        // Masses 1 and 3, so the centre-of-mass velocity is not zero and a symmetric
        // bug would not pass.
        scene.spawn(BodySpec::at([-1.0, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([4.0, 0.0]));
        scene.spawn(BodySpec::at([1.0, 0.0], SHAPE).with_inertia(3.0, 1.0).with_velocity([0.0, 0.0]));

        let energy_before = scene.bodies.kinetic_energy();
        let momentum_before = scene.bodies.linear_momentum();
        // v_cm = (1*4 + 3*0)/4 = 1 m/s.
        let v_cm = momentum_before[0] / 4.0;
        let energy_expected = 0.5 * 4.0 * v_cm * v_cm;

        for _ in 0..400 {
            scene.step(DT, [0.0, 0.0]);
        }

        assert!((scene.bodies.vel_x()[0] - v_cm).abs() < 1e-4, "{:?}", scene.bodies.vel_x());
        assert!((scene.bodies.vel_x()[1] - v_cm).abs() < 1e-4, "{:?}", scene.bodies.vel_x());
        assert!(
            (scene.bodies.kinetic_energy() - energy_expected).abs() < 1e-4,
            "expected {energy_expected} J left of {energy_before} J, got {}",
            scene.bodies.kinetic_energy()
        );
        assert!(
            (scene.bodies.linear_momentum()[0] - momentum_before[0]).abs() < 1e-12,
            "momentum is exact whatever the restitution"
        );
    }

    /// The invariant this module publishes with confidence: whatever the iteration
    /// count, every impulse is equal and opposite, so linear momentum cannot drift.
    #[test]
    fn linear_momentum_is_conserved_even_when_the_solve_is_starved() {
        for iterations in [1, 2, 8] {
            let config = SolverConfig { velocity_iterations: iterations, ..SolverConfig::default() };
            let mut scene = Scene::new(circles(0.5, 0.4), 6, config);
            // A row of discs closing on each other: contacts are numerous and, at one
            // iteration, badly under-resolved.
            for i in 0..6 {
                let x = f64::from(i) * 1.2 - 3.0;
                scene.spawn(
                    BodySpec::at([x, 0.0], SHAPE)
                        .with_inertia(1.0 + f64::from(i), 0.5)
                        .with_velocity([1.0 - 0.3 * f64::from(i), 0.0]),
                );
            }
            let before = scene.bodies.linear_momentum();

            for _ in 0..400 {
                scene.step(DT, [0.0, 0.0]);
            }

            let after = scene.bodies.linear_momentum();
            let scale = scene.bodies.momentum_scale().max(1.0);
            assert!(
                (after[0] - before[0]).abs() / scale < 1e-12
                    && (after[1] - before[1]).abs() / scale < 1e-12,
                "{iterations} iterations: {before:?} -> {after:?}"
            );
        }
    }

    /// The impulse solve on its own conserves angular momentum exactly: an equal and
    /// opposite pair applied at a shared point contributes P x J + P x (-J) = 0.
    /// Checked with a spinning off-centre collision, where a transposed lever arm would
    /// show up immediately.
    #[test]
    fn the_impulse_solve_alone_conserves_angular_momentum() {
        let mut scene = Scene::new(circles(0.8, 0.5), 2, SolverConfig::accurate());
        // Offset in y so the collision is off-centre and torques are real, and spinning
        // so friction has angular momentum to transfer.
        scene.spawn(
            BodySpec::at([-2.0, -0.3], SHAPE)
                .with_inertia(1.0, 0.125)
                .with_velocity([3.0, 0.0])
                .with_angular_velocity(5.0),
        );
        scene.spawn(
            BodySpec::at([2.0, 0.3], SHAPE)
                .with_inertia(2.0, 0.25)
                .with_velocity([-1.0, 0.0])
                .with_angular_velocity(-2.0),
        );

        let before = scene.bodies.angular_momentum();
        let mut correction_drift = 0.0;
        for _ in 0..600 {
            correction_drift += scene.step(DT, [0.0, 0.0]).angular_drift;
        }
        let after = scene.bodies.angular_momentum();

        // Everything not accounted for by position correction is round-off.
        let unexplained = after - before - correction_drift;
        assert!(
            unexplained.abs() < 1e-10,
            "{before} -> {after}, of which {correction_drift} was position correction; \
             {unexplained} is unexplained"
        );
    }

    /// And the part position correction *does* cost is measured, not waved at. A
    /// pre-overlapped pile is the worst case: the correction runs hard for many steps.
    #[test]
    fn position_correction_costs_angular_momentum_and_reports_how_much() {
        let mut scene = Scene::new(circles(0.0, 0.3), 4, SolverConfig::default());
        for i in 0..4 {
            // Spaced 0.4 apart with radius 0.5: overlapping by 0.6 m each.
            scene.spawn(
                BodySpec::at([f64::from(i) * 0.4, 0.0], SHAPE)
                    .with_inertia(1.0, 0.125)
                    .with_velocity([0.0, 0.5 - 0.3 * f64::from(i)]),
            );
        }
        let before = scene.bodies.angular_momentum();
        let linear_before = scene.bodies.linear_momentum();
        let mut reported = 0.0;
        let mut any_correction = false;
        for _ in 0..300 {
            let report = scene.step(DT, [0.0, 0.0]);
            reported += report.angular_drift;
            any_correction |= report.angular_drift != 0.0;
        }
        let after = scene.bodies.angular_momentum();

        assert!(any_correction, "the pile really was overlapped");
        assert!(
            (after - before - reported).abs() < 1e-10,
            "the reported drift must account for the change: {before} -> {after}, reported {reported}"
        );
        // Linear momentum survives it untouched, which is the point of correcting
        // position through velocities that are then discarded.
        let linear_after = scene.bodies.linear_momentum();
        assert!(
            (linear_after[0] - linear_before[0]).abs() < 1e-12
                && (linear_after[1] - linear_before[1]).abs() < 1e-12,
            "{linear_before:?} -> {linear_after:?}"
        );
    }

    /// A contact pushes. If it can pull, bodies stick as they separate, which is the
    /// most visible way to get an impulse sign wrong.
    #[test]
    fn a_contact_never_pulls_bodies_together() {
        let mut scene = Scene::new(circles(0.0, 0.0), 2, SolverConfig::default());
        // Just touching, and already moving apart.
        scene.spawn(BodySpec::at([-0.5, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([-1.0, 0.0]));
        scene.spawn(BodySpec::at([0.5, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([1.0, 0.0]));

        for _ in 0..50 {
            scene.step(DT, [0.0, 0.0]);
        }
        assert!(scene.bodies.vel_x()[0] <= -1.0 + 1e-9, "{:?}", scene.bodies.vel_x());
        assert!(scene.bodies.vel_x()[1] >= 1.0 - 1e-9, "{:?}", scene.bodies.vel_x());
    }

    /// A box dropped on the ground must come to rest on it, overlapping by no more
    /// than a few slops, and stay there rather than sinking or buzzing.
    #[test]
    fn a_box_comes_to_rest_on_the_ground_and_stays() {
        let colliders = vec![
            Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(0.5),
            Collider::new(Shape::rectangle(0.5, 0.5).unwrap()).with_friction(0.5),
        ];
        let mut scene = Scene::new(colliders, 2, SolverConfig::default());
        scene.spawn(BodySpec::statik([0.0, 0.0], ShapeId::from_index(0)));
        scene.spawn(BodySpec::at([0.0, 2.0], ShapeId::from_index(1)).with_inertia(1.0, 1.0 / 6.0));

        for _ in 0..1200 {
            scene.step(DT, G);
        }

        // Resting height: ground top at 0.5, box half-height 0.5, so y is about 1.0.
        let y = scene.bodies.pos_y()[1];
        assert!((y - 1.0).abs() < 5.0 * LINEAR_SLOP, "resting at {y}, expected about 1.0");
        assert!(scene.bodies.vel_y()[1].abs() < 0.05, "still moving at {}", scene.bodies.vel_y()[1]);

        // And it is still there a thousand steps later -- no slow sinking.
        for _ in 0..1000 {
            scene.step(DT, G);
        }
        let settled = scene.bodies.pos_y()[1];
        assert!((settled - y).abs() < LINEAR_SLOP, "drifted from {y} to {settled}");
    }

    /// Restitution 1 must return a bouncing ball to nearly its drop height; 0 must
    /// leave it on the floor. Anything between is a coefficient, not a bug.
    #[test]
    fn restitution_controls_the_bounce_height() {
        let heights: Vec<f64> = [0.0, 0.5, 0.95]
            .into_iter()
            .map(|restitution| {
                let colliders = vec![
                    Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_restitution(restitution),
                    Collider::new(Shape::circle(0.25).unwrap()).with_restitution(restitution),
                ];
                let mut scene = Scene::new(colliders, 2, SolverConfig::accurate());
                scene.spawn(BodySpec::statik([0.0, 0.0], ShapeId::from_index(0)));
                scene.spawn(BodySpec::at([0.0, 3.0], ShapeId::from_index(1)).with_inertia(1.0, 0.03125));

                let mut peak: f64 = 0.0;
                let mut has_bounced = false;
                for _ in 0..2400 {
                    scene.step(DT, G);
                    if scene.bodies.vel_y()[1] > 0.1 {
                        has_bounced = true;
                    }
                    if has_bounced {
                        peak = peak.max(scene.bodies.pos_y()[1]);
                    }
                }
                peak
            })
            .collect();

        assert!(heights[0] < 0.9, "a dead ball does not bounce: {}", heights[0]);
        assert!(heights[1] > heights[0], "{heights:?}");
        assert!(heights[2] > heights[1], "{heights:?}");
        // Dropping from 3.0 to a resting 0.75 is a fall of 2.25 m. At e = 0.95 the
        // rebound keeps e^2 = 90% of that, so the peak should be near 2.78.
        assert!(heights[2] > 2.4 && heights[2] < 3.0, "e = 0.95 rebound: {}", heights[2]);
    }

    /// Coulomb friction: a block on a slope stays put when mu exceeds tan(theta) and
    /// slides when it does not. That is the definition, and it is the only friction
    /// test that distinguishes a real friction cone from arbitrary damping.
    #[test]
    fn friction_holds_a_block_on_a_slope_exactly_when_coulomb_says_it_should() {
        let angle: f64 = 0.4; // about 23 degrees, tan = 0.42
        let run = |friction: f64| {
            let colliders = vec![
                Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(friction),
                Collider::new(Shape::rectangle(0.4, 0.4).unwrap()).with_friction(friction),
            ];
            let mut scene = Scene::new(colliders, 2, SolverConfig::accurate());
            scene.spawn(BodySpec::statik([0.0, 0.0], ShapeId::from_index(0)).with_angle(angle));
            // Resting on the slope surface, offset along the slope normal.
            let normal = [-angle.sin(), angle.cos()];
            let start = [normal[0] * 0.9, normal[1] * 0.9];
            scene.spawn(
                BodySpec::at(start, ShapeId::from_index(1))
                    .with_angle(angle)
                    .with_inertia(1.0, 0.4 * 0.4 * 2.0 / 3.0),
            );
            let x0 = scene.bodies.pos_x()[1];
            for _ in 0..2400 {
                scene.step(DT, G);
            }
            (scene.bodies.pos_x()[1] - x0).abs()
        };

        let slid_far = run(0.1);
        let slid_little = run(0.9);
        assert!(slid_far > 0.3, "mu = 0.1 < tan 0.4 = 0.42: it must slide, moved {slid_far}");
        assert!(slid_little < 0.05, "mu = 0.9 > tan 0.4: it must hold, moved {slid_little}");
    }

    /// Warm starting is the difference between a stack that holds at three iterations
    /// and one that sags. If it stops working, this is what notices.
    #[test]
    fn warm_starting_holds_a_stack_that_a_cold_solve_lets_sag() {
        let build = |warm: bool| {
            let colliders = vec![
                Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(0.6),
                Collider::new(Shape::rectangle(0.5, 0.5).unwrap()).with_friction(0.6),
            ];
            let config =
                SolverConfig { velocity_iterations: 3, warm_starting: warm, ..SolverConfig::default() };
            let mut scene = Scene::new(colliders, 6, config);
            scene.spawn(BodySpec::statik([0.0, 0.0], ShapeId::from_index(0)));
            for level in 0..5 {
                scene.spawn(
                    BodySpec::at([0.0, 1.0 + f64::from(level) * 1.001], ShapeId::from_index(1))
                        .with_inertia(1.0, 1.0 / 6.0),
                );
            }
            let mut last = SolveReport::default();
            for _ in 0..1500 {
                last = scene.step(DT, G);
            }
            (last, scene.bodies.pos_y()[5])
        };

        let (cold, cold_top) = build(false);
        let (warm, warm_top) = build(true);
        assert!(warm.points > 0, "the stack must actually be in contact");
        assert!(
            warm.residual <= cold.residual + 1e-12,
            "warm {} should not be worse than cold {}",
            warm.residual,
            cold.residual
        );
        // The physical consequence: the warm stack sits higher, having sagged less.
        assert!(warm_top >= cold_top - 1e-9, "warm top {warm_top} vs cold {cold_top}");
    }

    /// Deep overlap must not fire a body across the scene. A model that spawns
    /// something inside a wall should ooze out, not explode.
    ///
    /// The sharp form of the claim: position correction moves the body without ever
    /// becoming velocity it keeps. That is what pseudo-velocities are for, and it is
    /// the difference between a stack settling and a stack launching itself.
    #[test]
    fn correcting_a_deep_overlap_never_becomes_real_velocity() {
        let colliders = vec![
            Collider::new(Shape::rectangle(10.0, 1.0).unwrap()),
            Collider::new(Shape::rectangle(0.5, 0.5).unwrap()),
        ];
        let mut scene = Scene::new(colliders, 2, SolverConfig::default());
        scene.spawn(BodySpec::statik([0.0, 0.0], ShapeId::from_index(0)));
        // A metre inside the ground, but nearer its top face, so "the shortest way out"
        // is unambiguously upward. Starting at the exact centre of a symmetric body is
        // a genuinely ill-posed question and the solver is entitled to answer "down".
        scene.spawn(BodySpec::at([0.0, 0.5], ShapeId::from_index(1)).with_inertia(1.0, 1.0 / 6.0));

        let mut peak: f64 = 0.0;
        for _ in 0..900 {
            scene.step(DT, G);
            peak = peak
                .max((scene.bodies.vel_x()[1].powi(2) + scene.bodies.vel_y()[1].powi(2)).sqrt());
        }
        assert!(
            peak < 1.0,
            "the body was pushed out 1 m but never moved faster than {peak} m/s, \
             so the correction stayed out of the momentum"
        );
        assert!(
            scene.bodies.pos_y()[1] > 1.4,
            "it should be resting on top at about 1.5: {}",
            scene.bodies.pos_y()[1]
        );
    }

    /// A scene with no contacts must cost nothing and report nothing.
    #[test]
    fn an_empty_solve_reports_nothing_rather_than_panicking() {
        let mut scene = Scene::new(circles(0.0, 0.0), 2, SolverConfig::default());
        scene.spawn(BodySpec::at([-10.0, 0.0], SHAPE).with_inertia(1.0, 1.0));
        scene.spawn(BodySpec::at([10.0, 0.0], SHAPE).with_inertia(1.0, 1.0));
        let report = scene.step(DT, [0.0, 0.0]);
        assert_eq!(report.points, 0);
        assert_eq!(report.residual, 0.0);
        assert_eq!(scene.solver.cached_impulse_count(), 0);
    }

    /// A pair of static bodies presents infinite mass; the solver must skip it rather
    /// than divide by zero.
    #[test]
    fn two_static_bodies_in_contact_do_not_divide_by_zero() {
        let mut scene = Scene::new(circles(0.5, 0.5), 2, SolverConfig::default());
        scene.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        scene.spawn(BodySpec::statik([0.5, 0.0], SHAPE));
        // The broadphase already skips static-static pairs, so force the contact.
        let contacts = [Contact {
            pair: BodyPair::new(0, 1),
            manifold: Manifold::single(Vec2::X, vec2(0.25, 0.0), 0.5, FeatureId::SINGLE),
            restitution: 0.5,
            friction: 0.5,
        }];
        let report = scene.solver.solve(&mut scene.bodies, &contacts, DT);
        assert_eq!(report.points, 0, "nothing to solve between two walls");
        assert!(!scene.bodies.has_non_finite());
    }

    /// FR-011: the same scene stepped twice must produce bit-identical results.
    #[test]
    fn a_solve_is_reproducible() {
        let run = || {
            let colliders = vec![
                Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(0.4),
                Collider::new(Shape::circle(0.3).unwrap()).with_restitution(0.4).with_friction(0.4),
            ];
            let mut scene = Scene::new(colliders, 9, SolverConfig::default());
            scene.spawn(BodySpec::statik([0.0, 0.0], ShapeId::from_index(0)));
            for i in 0..8 {
                scene.spawn(
                    BodySpec::at(
                        [f64::from(i) * 0.31 - 1.0, 1.0 + f64::from(i) * 0.7],
                        ShapeId::from_index(1),
                    )
                    .with_inertia(1.0, 0.045)
                    .with_velocity([0.3 * f64::from(i % 3) - 0.3, 0.0]),
                );
            }
            for _ in 0..900 {
                scene.step(DT, G);
            }
            (
                scene.bodies.pos_x().to_vec(),
                scene.bodies.pos_y().to_vec(),
                scene.bodies.omega().to_vec(),
            )
        };
        assert_eq!(run(), run(), "identical inputs must give identical bits");
    }

    #[test]
    fn resetting_clears_the_warm_start_cache() {
        let mut scene = Scene::new(circles(0.0, 0.5), 2, SolverConfig::default());
        scene.spawn(BodySpec::at([-0.4, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([1.0, 0.0]));
        scene.spawn(BodySpec::at([0.4, 0.0], SHAPE).with_inertia(1.0, 1.0).with_velocity([-1.0, 0.0]));
        scene.step(DT, [0.0, 0.0]);
        assert!(scene.solver.cached_impulse_count() > 0);
        scene.solver.reset();
        assert_eq!(scene.solver.cached_impulse_count(), 0);
    }
}
