//! Joints: the §11.1 constraints row — distance, pin, spring, motor.
//!
//! # Why they are solved in the same loop as contacts
//!
//! A pendulum resting against a wall is held by both a pin and a contact, and neither
//! is right on its own: satisfying the pin moves the bob into the wall, satisfying the
//! contact swings it off the pin. Solving one set to convergence and then the other
//! just alternates between the two answers. Interleaving them in a single
//! Gauss–Seidel sweep lets the two constraints negotiate, which is what makes a
//! ragdoll leaning on scenery behave.
//!
//! # Hard constraints and soft ones
//!
//! [`Joint::Distance`], [`Joint::Pin`] and [`Joint::Motor`] are **hard**: they are
//! solved as velocity constraints with an accumulated impulse, exactly like a contact,
//! and their error is corrected positionally. A hard constraint at a sufficient
//! iteration count is satisfied to round-off.
//!
//! [`Joint::Spring`] is **soft** and deliberately different. It is a force law, not a
//! constraint: it applies `-k·x - c·v` and lets the integrator do the rest. Writing a
//! spring as a stiff constraint would make it inextensible, which is not a spring; and
//! writing it as a constraint with a compliance term would reproduce, with more
//! machinery, exactly the force the direct form applies. The cost is that a stiff
//! spring has a timestep limit — see [`Joint::stable_step`] — and the domain reports it
//! rather than silently exploding.

use lattice_ir::RigidBodyStore;

use crate::math::{vec2, Vec2};

/// How far a hard constraint may be violated before position correction acts, m.
///
/// The same argument as the contact slop: driving the error to exactly zero every step
/// makes a jointed assembly buzz, because the correction overshoots and the next step
/// corrects back.
pub const JOINT_SLOP: f64 = 1e-4;

/// A connection between two bodies, or between a body and the world.
///
/// Anchors are given in each body's **local** frame, so a joint follows its bodies as
/// they move and turn. A joint to the world uses a static body as its other end, which
/// keeps one code path rather than two.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Joint {
    /// Hold two anchor points a fixed distance apart.
    ///
    /// The rigid-rod joint: a pendulum arm, a strut, a linkage. Unlike a spring it does
    /// not stretch under load.
    Distance {
        /// Slot of the first body.
        a: usize,
        /// Slot of the second body.
        b: usize,
        /// Anchor on `a`, in its local frame.
        local_a: Vec2,
        /// Anchor on `b`, in its local frame.
        local_b: Vec2,
        /// The distance to hold, m.
        rest_length: f64,
        /// When true the joint may go slack: it pulls at full length but never pushes.
        ///
        /// The difference between a rod and a rope, and a rope is what most models
        /// actually mean.
        rope: bool,
    },
    /// Hold two anchor points coincident, leaving rotation free.
    ///
    /// The revolute joint — a hinge. Two scalar constraints rather than one, since the
    /// anchors must agree in both x and y.
    Pin {
        /// Slot of the first body.
        a: usize,
        /// Slot of the second body.
        b: usize,
        /// Anchor on `a`, in its local frame.
        local_a: Vec2,
        /// Anchor on `b`, in its local frame.
        local_b: Vec2,
    },
    /// A damped linear spring between two anchor points.
    ///
    /// Soft: a force law, not a constraint. See the module documentation.
    Spring {
        /// Slot of the first body.
        a: usize,
        /// Slot of the second body.
        b: usize,
        /// Anchor on `a`, in its local frame.
        local_a: Vec2,
        /// Anchor on `b`, in its local frame.
        local_b: Vec2,
        /// Natural length, m.
        rest_length: f64,
        /// Stiffness, N/m.
        stiffness: f64,
        /// Damping coefficient, N·s/m.
        damping: f64,
    },
    /// Drive the relative angular velocity of two bodies toward a target.
    ///
    /// Hard, but with a torque budget: the motor reaches its target if it can and gives
    /// up gracefully if the load exceeds `max_torque`. A motor with an unbounded budget
    /// would move any load, which is a servo nobody has.
    Motor {
        /// Slot of the driven body.
        a: usize,
        /// Slot of the body it pushes against — often a static frame.
        b: usize,
        /// Target for `ω_a − ω_b`, rad/s.
        target_speed: f64,
        /// Largest torque the motor may apply, N·m.
        max_torque: f64,
    },
}

impl Joint {
    /// The two bodies this joint acts on.
    pub fn bodies(&self) -> (usize, usize) {
        match *self {
            Joint::Distance { a, b, .. }
            | Joint::Pin { a, b, .. }
            | Joint::Spring { a, b, .. }
            | Joint::Motor { a, b, .. } => (a, b),
        }
    }

    /// The largest timestep this joint is stable at, or infinity when it imposes none.
    ///
    /// Only a spring does. It is a harmonic oscillator with `ω = √(k/m)`, and
    /// semi-implicit Euler is stable for `dt < 2/ω`. Half of that is used, because
    /// "stable" at the limit means an oscillation that neither grows nor decays, which
    /// is not the same as accurate — the same distinction spelled out in the heat
    /// module's contract.
    pub fn stable_step(&self, bodies: &RigidBodyStore) -> f64 {
        let Joint::Spring { a, b, stiffness, .. } = *self else {
            return f64::INFINITY;
        };
        if stiffness <= 0.0 || a >= bodies.len() || b >= bodies.len() {
            return f64::INFINITY;
        }
        // The reduced mass of the pair. A spring to a static anchor sees only the
        // moving body's mass, which the zero reciprocal handles.
        let inv_reduced = bodies.inv_mass()[a] + bodies.inv_mass()[b];
        if inv_reduced <= 0.0 {
            return f64::INFINITY;
        }
        let omega = (stiffness * inv_reduced).sqrt();
        if omega > 0.0 { 1.0 / omega } else { f64::INFINITY }
    }

    /// True when both endpoints refer to live slots.
    pub fn is_valid(&self, bodies: &RigidBodyStore) -> bool {
        let (a, b) = self.bodies();
        a < bodies.len() && b < bodies.len() && a != b
    }
}

/// Per-joint state that persists across iterations and steps.
#[derive(Clone, Copy, Debug, Default)]
struct JointState {
    /// Accumulated impulse along the joint axis, or the x component for a pin.
    impulse_x: f64,
    /// Accumulated impulse along y, used only by a pin.
    impulse_y: f64,
    /// Accumulated angular impulse, used only by a motor.
    angular_impulse: f64,
    /// World-space lever arm on body `a`, recomputed each step.
    arm_a: Vec2,
    /// World-space lever arm on body `b`.
    arm_b: Vec2,
    /// Unit vector along the joint, for a distance joint.
    axis: Vec2,
    /// Effective mass along `axis`, for a distance joint.
    mass_axis: f64,
    /// Effective angular mass, for a motor.
    mass_angular: f64,
    /// The inverted 2x2 effective-mass matrix of a pin, as `[m11, m12, m22]`.
    ///
    /// A pin is two coupled constraints, and the coupling is not small: a body pinned
    /// far from its centre of mass presents wildly different resistance along the two
    /// axes. Solving them as independent scalars — Gauss–Seidel within a single joint —
    /// converges at a rate set by that ratio, which for a pendulum bob with small
    /// rotational inertia means it does not converge at all in any sane iteration
    /// count. Inverting the 2x2 solves both exactly in one go.
    pin_inverse: [f64; 3],
    /// Current violation, m for a distance joint, m for a pin.
    error: f64,
    /// Whether this joint contributes anything at all this step.
    active: bool,
}

/// Solves joints alongside contacts.
///
/// Kept separate from [`crate::solver::ContactSolver`] because the two have different
/// data, but stepped in the same loop — see the module documentation.
#[derive(Clone, Default, Debug)]
pub struct JointSolver {
    state: Vec<JointState>,
}

impl JointSolver {
    /// An empty solver.
    pub fn new() -> JointSolver {
        JointSolver::default()
    }

    /// Discard accumulated impulses. Use when a scene is reset.
    pub fn reset(&mut self) {
        self.state.clear();
    }

    /// Apply spring forces. Call before integrating velocities, with the other forces.
    ///
    /// Springs are separated from the constraint solve because they are forces: they
    /// belong in the force accumulation phase, and putting them there is what makes
    /// them compose with gravity and drag the way a reader expects.
    pub fn apply_spring_forces(&self, bodies: &mut RigidBodyStore, joints: &[Joint]) {
        for joint in joints {
            let Joint::Spring { a, b, local_a, local_b, rest_length, stiffness, damping } = *joint
            else {
                continue;
            };
            if !joint.is_valid(bodies) {
                continue;
            }
            let pa = Vec2::from(bodies.to_world_point(a, local_a.to_array()));
            let pb = Vec2::from(bodies.to_world_point(b, local_b.to_array()));
            let delta = pb - pa;
            let Some(axis) = delta.normalize() else { continue };

            let extension = delta.length() - rest_length;
            let va = Vec2::from(bodies.point_velocity(a, local_a.to_array()));
            let vb = Vec2::from(bodies.point_velocity(b, local_b.to_array()));
            let closing = (vb - va).dot(axis);

            // Pulls b toward a when stretched. The damping term opposes the rate of
            // change of length, not the bodies' absolute motion — a spring on two
            // bodies drifting together at constant separation should do nothing.
            let magnitude = -(stiffness * extension + damping * closing);
            let force = axis * magnitude;
            bodies.add_force_at(b, force.to_array(), pb.to_array());
            bodies.add_force_at(a, (-force).to_array(), pa.to_array());
        }
    }

    /// Compute the per-joint constants for this step.
    ///
    /// Call after integrating velocities and before the iteration loop, so lever arms
    /// reflect the poses the constraints will actually be solved at.
    pub fn prepare(&mut self, bodies: &RigidBodyStore, joints: &[Joint], warm_starting: bool) {
        self.state.resize(joints.len(), JointState::default());
        for (index, joint) in joints.iter().enumerate() {
            let state = &mut self.state[index];
            state.active = false;
            if !joint.is_valid(bodies) {
                *state = JointState::default();
                continue;
            }

            match *joint {
                Joint::Spring { .. } => {}
                Joint::Distance { a, b, local_a, local_b, rest_length, rope } => {
                    let pa = Vec2::from(bodies.to_world_point(a, local_a.to_array()));
                    let pb = Vec2::from(bodies.to_world_point(b, local_b.to_array()));
                    let delta = pb - pa;
                    let Some(axis) = delta.normalize() else { continue };
                    let length = delta.length();
                    // A rope that is not taut is not a constraint at all this step.
                    if rope && length <= rest_length {
                        state.impulse_x = 0.0;
                        continue;
                    }
                    state.axis = axis;
                    state.arm_a = pa - vec2(bodies.pos_x()[a], bodies.pos_y()[a]);
                    state.arm_b = pb - vec2(bodies.pos_x()[b], bodies.pos_y()[b]);
                    state.mass_axis = axis_mass(bodies, a, b, state.arm_a, state.arm_b, axis);
                    state.error = length - rest_length;
                    state.active = state.mass_axis > 0.0;
                }
                Joint::Pin { a, b, local_a, local_b } => {
                    let pa = Vec2::from(bodies.to_world_point(a, local_a.to_array()));
                    let pb = Vec2::from(bodies.to_world_point(b, local_b.to_array()));
                    // Both lever arms are taken to the *midpoint* of the two anchors,
                    // not to each body's own anchor. When the joint is satisfied the two
                    // coincide and it makes no difference; when it is not, using the
                    // separate anchors would apply the equal-and-opposite impulses at
                    // two different points, which is a couple — and a couple is angular
                    // momentum created from nothing. The midpoint makes
                    // `mid x J + mid x (-J) = 0` exact, and the position error is
                    // handled by the bias term where it belongs.
                    let mid = (pa + pb) * 0.5;
                    state.arm_a = mid - vec2(bodies.pos_x()[a], bodies.pos_y()[a]);
                    state.arm_b = mid - vec2(bodies.pos_x()[b], bodies.pos_y()[b]);
                    state.pin_inverse = invert_pin_mass(bodies, a, b, state.arm_a, state.arm_b);
                    state.axis = pb - pa;
                    state.error = state.axis.length();
                    state.active = state.pin_inverse != [0.0; 3];
                }
                Joint::Motor { a, b, .. } => {
                    let total = bodies.inv_inertia()[a] + bodies.inv_inertia()[b];
                    state.mass_angular = if total > 0.0 { 1.0 / total } else { 0.0 };
                    state.active = state.mass_angular > 0.0;
                }
            }

            if !warm_starting {
                state.impulse_x = 0.0;
                state.impulse_y = 0.0;
                state.angular_impulse = 0.0;
            }
        }
    }

    /// Re-apply last step's impulses, for the same reason contacts do.
    pub fn warm_start(&self, bodies: &mut RigidBodyStore, joints: &[Joint]) {
        for (index, joint) in joints.iter().enumerate() {
            let state = self.state[index];
            if !state.active {
                continue;
            }
            let (a, b) = joint.bodies();
            match joint {
                Joint::Distance { .. } => {
                    apply(bodies, a, b, state.axis * state.impulse_x, state.arm_a, state.arm_b);
                }
                Joint::Pin { .. } => {
                    let impulse = vec2(state.impulse_x, state.impulse_y);
                    apply(bodies, a, b, impulse, state.arm_a, state.arm_b);
                }
                Joint::Motor { .. } => {
                    apply_angular(bodies, a, b, state.angular_impulse);
                }
                Joint::Spring { .. } => {}
            }
        }
    }

    /// One velocity pass over every joint. Returns the largest remaining violation rate.
    pub fn solve_velocities(&mut self, bodies: &mut RigidBodyStore, joints: &[Joint], dt: f64) -> f64 {
        let mut worst: f64 = 0.0;
        for (index, joint) in joints.iter().enumerate() {
            let state = self.state[index];
            if !state.active {
                continue;
            }
            let (a, b) = joint.bodies();

            match *joint {
                Joint::Spring { .. } => {}
                Joint::Distance { rope, .. } => {
                    let relative = relative_point_velocity(bodies, a, b, state.arm_a, state.arm_b);
                    let rate = relative.dot(state.axis);
                    // Bake a fraction of the position error into the velocity target
                    // (Baumgarte). A hard joint with no positional feedback drifts apart
                    // over thousands of steps, and the drift is invisible until the
                    // linkage visibly comes to pieces.
                    let bias = if dt > 0.0 {
                        0.2 * (state.error.abs() - JOINT_SLOP).max(0.0) * state.error.signum() / dt
                    } else {
                        0.0
                    };
                    worst = worst.max(rate.abs());

                    let mut impulse = -(rate + bias) * state.mass_axis;
                    if rope {
                        // A rope pulls only: the accumulated impulse may not become a push.
                        let total = (self.state[index].impulse_x + impulse).min(0.0);
                        impulse = total - self.state[index].impulse_x;
                        self.state[index].impulse_x = total;
                    } else {
                        self.state[index].impulse_x += impulse;
                    }
                    apply(bodies, a, b, state.axis * impulse, state.arm_a, state.arm_b);
                }
                Joint::Pin { .. } => {
                    let relative = relative_point_velocity(bodies, a, b, state.arm_a, state.arm_b);
                    // Baumgarte: fold a fraction of the position error into the velocity
                    // target. Without it a hard joint drifts apart over thousands of
                    // steps, and the drift is invisible until the linkage comes to pieces.
                    let bias = if dt > 0.0 { 0.2 / dt } else { 0.0 };
                    let target = vec2(
                        relative.x + bias * bias_error(state.axis.x),
                        relative.y + bias * bias_error(state.axis.y),
                    );
                    worst = worst.max(relative.length());

                    let [m11, m12, m22] = state.pin_inverse;
                    let impulse = vec2(
                        -(m11 * target.x + m12 * target.y),
                        -(m12 * target.x + m22 * target.y),
                    );
                    self.state[index].impulse_x += impulse.x;
                    self.state[index].impulse_y += impulse.y;
                    apply(bodies, a, b, impulse, state.arm_a, state.arm_b);
                }
                Joint::Motor { target_speed, max_torque, .. } => {
                    let relative = bodies.omega()[a] - bodies.omega()[b];
                    let error = relative - target_speed;
                    worst = worst.max(error.abs());

                    let mut impulse = -error * state.mass_angular;
                    // The torque budget bounds the *accumulated* impulse, so a motor
                    // stalled against an immovable load applies exactly its maximum
                    // rather than whatever the last pass happened to compute.
                    let limit = max_torque * dt;
                    let total = (self.state[index].angular_impulse + impulse).clamp(-limit, limit);
                    impulse = total - self.state[index].angular_impulse;
                    self.state[index].angular_impulse = total;
                    apply_angular(bodies, a, b, impulse);
                }
            }
        }
        worst
    }

    /// The largest position error across all joints, m. Zero for a motor, which
    /// constrains a rate rather than a configuration.
    pub fn max_error(&self, joints: &[Joint]) -> f64 {
        joints
            .iter()
            .enumerate()
            .filter(|(index, _)| self.state[*index].active)
            .filter(|(_, joint)| !matches!(joint, Joint::Motor { .. }))
            .map(|(index, _)| self.state[index].error.abs())
            .fold(0.0, f64::max)
    }
}

/// Signed error contribution along one axis of a pin, normalized so the bias pushes in
/// the right direction without amplifying a large error.
fn bias_error(component: f64) -> f64 {
    let magnitude = component.abs();
    if magnitude <= JOINT_SLOP {
        0.0
    } else {
        component.signum() * (magnitude - JOINT_SLOP)
    }
}

/// The inverse of a pin's 2x2 effective-mass matrix, as `[m11, m12, m22]`.
///
/// `K` relates an impulse at the anchor to the change in the anchor's relative
/// velocity:
///
/// ```text
///   K_xx = 1/ma + 1/mb + ra.y²/Ia + rb.y²/Ib
///   K_yy = 1/ma + 1/mb + ra.x²/Ia + rb.x²/Ib
///   K_xy = −ra.x·ra.y/Ia − rb.x·rb.y/Ib
/// ```
///
/// Returns zeros when the pair cannot move at all, or when `K` is singular — which
/// happens for two bodies whose rotation is locked and whose masses are infinite.
fn invert_pin_mass(
    bodies: &RigidBodyStore,
    a: usize,
    b: usize,
    ra: Vec2,
    rb: Vec2,
) -> [f64; 3] {
    let (ima, imb) = (bodies.inv_mass()[a], bodies.inv_mass()[b]);
    let (iia, iib) = (bodies.inv_inertia()[a], bodies.inv_inertia()[b]);
    let k11 = ima + imb + iia * ra.y * ra.y + iib * rb.y * rb.y;
    let k22 = ima + imb + iia * ra.x * ra.x + iib * rb.x * rb.x;
    let k12 = -iia * ra.x * ra.y - iib * rb.x * rb.y;

    let determinant = k11 * k22 - k12 * k12;
    if !(determinant.is_finite() && determinant > 0.0) {
        return [0.0; 3];
    }
    [k22 / determinant, -k12 / determinant, k11 / determinant]
}

/// `1 / (dᵀ K d)` for a pair of lever arms — the same quantity the contact solver uses.
fn axis_mass(
    bodies: &RigidBodyStore,
    a: usize,
    b: usize,
    arm_a: Vec2,
    arm_b: Vec2,
    axis: Vec2,
) -> f64 {
    let cross_a = arm_a.cross(axis);
    let cross_b = arm_b.cross(axis);
    let total = bodies.inv_mass()[a]
        + bodies.inv_mass()[b]
        + bodies.inv_inertia()[a] * cross_a * cross_a
        + bodies.inv_inertia()[b] * cross_b * cross_b;
    if total > 0.0 { 1.0 / total } else { 0.0 }
}

fn relative_point_velocity(
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

/// Apply `impulse` to `b` and its negation to `a`, so momentum is conserved by
/// construction rather than by two call sites agreeing.
fn apply(bodies: &mut RigidBodyStore, a: usize, b: usize, impulse: Vec2, arm_a: Vec2, arm_b: Vec2) {
    bodies.apply_impulse_at_arm(a, (-impulse).to_array(), arm_a.to_array());
    bodies.apply_impulse_at_arm(b, impulse.to_array(), arm_b.to_array());
}

/// Apply an angular impulse to `a` and its negation to `b`.
fn apply_angular(bodies: &mut RigidBodyStore, a: usize, b: usize, impulse: f64) {
    let motion = bodies.motion();
    motion.omega[a] += motion.inv_inertia[a] * impulse;
    motion.omega[b] -= motion.inv_inertia[b] * impulse;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{BodySpec, ShapeId};

    const SHAPE: ShapeId = ShapeId::from_index(0);
    const DT: f64 = 1.0 / 480.0;
    const G: f64 = -9.80665;

    /// Bodies plus joints, stepped the way a domain will step them.
    struct Rig {
        bodies: RigidBodyStore,
        joints: Vec<Joint>,
        solver: JointSolver,
        iterations: usize,
    }

    impl Rig {
        fn new(capacity: usize) -> Rig {
            Rig {
                bodies: RigidBodyStore::with_capacity(capacity),
                joints: Vec::new(),
                solver: JointSolver::new(),
                iterations: 20,
            }
        }

        fn spawn(&mut self, spec: BodySpec) -> usize {
            let id = self.bodies.spawn(spec).expect("rig sized for this");
            self.bodies.slot_of(id).unwrap()
        }

        fn step(&mut self, dt: f64, gravity: f64) {
            self.bodies.clear_forces();
            for slot in 0..self.bodies.len() {
                let mass = self.bodies.mass()[slot];
                self.bodies.add_force(slot, [0.0, gravity * mass]);
            }
            self.solver.apply_spring_forces(&mut self.bodies, &self.joints);
            self.bodies.integrate_velocities(dt);

            self.solver.prepare(&self.bodies, &self.joints, true);
            self.solver.warm_start(&mut self.bodies, &self.joints);
            for _ in 0..self.iterations {
                self.solver.solve_velocities(&mut self.bodies, &self.joints, dt);
            }

            self.bodies.integrate_positions(dt);
        }

        fn position(&self, slot: usize) -> Vec2 {
            vec2(self.bodies.pos_x()[slot], self.bodies.pos_y()[slot])
        }
    }

    /// A rod holds its length under load. The whole point of a hard constraint.
    #[test]
    fn a_distance_joint_holds_its_length_under_gravity() {
        let mut rig = Rig::new(2);
        let anchor = rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let bob = rig.spawn(BodySpec::at([2.0, 0.0], SHAPE).with_inertia(3.0, 0.5));
        rig.joints.push(Joint::Distance {
            a: anchor,
            b: bob,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 2.0,
            rope: false,
        });

        for _ in 0..6000 {
            rig.step(DT, G);
        }
        let length = rig.position(bob).length();
        assert!(
            (length - 2.0).abs() < 10.0 * JOINT_SLOP,
            "held at {length} m over 6000 steps, wanted 2.0"
        );
    }

    /// A rope pulls but does not push, which is the difference between a rope and a rod
    /// and is what most models actually mean by "attached by a string".
    #[test]
    fn a_rope_goes_slack_but_a_rod_does_not() {
        let build = |rope: bool| {
            let mut rig = Rig::new(2);
            let anchor = rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
            // Starting well inside the rest length, falling.
            let bob = rig.spawn(BodySpec::at([0.0, -0.5], SHAPE).with_inertia(1.0, 0.5));
            rig.joints.push(Joint::Distance {
                a: anchor,
                b: bob,
                local_a: Vec2::ZERO,
                local_b: Vec2::ZERO,
                rest_length: 2.0,
                rope,
            });
            for _ in 0..300 {
                rig.step(DT, G);
            }
            rig.position(bob).length()
        };

        assert!(build(true) > 0.5, "a rope lets the bob fall until it is taut");
        assert!(
            (build(false) - 2.0).abs() < 0.05,
            "a rod pushes the bob back out to 2.0, got {}",
            build(false)
        );
    }

    /// Spec 19.2 pendulum. A simple pendulum released from a small angle has period
    /// `T = 2*pi*sqrt(L/g)`, and matching it is the check that the pin joint, the
    /// integrator and gravity all agree.
    #[test]
    fn a_pinned_pendulum_has_the_analytic_small_angle_period() {
        let length: f64 = 1.0;
        let amplitude: f64 = 0.05; // radians: small enough that sin(x) ~ x holds
        let mut rig = Rig::new(2);
        let pivot = rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        // A point mass hanging at `length`, displaced by `amplitude`.
        let start = [length * amplitude.sin(), -length * amplitude.cos()];
        // A compact bob: a 4 cm disc, so its own rotational inertia is negligible
        // beside the m·L² the pin gives it and the analytic simple-pendulum period
        // applies. The pin still has to hold a body whose inertia is ten thousand times
        // smaller than the constraint it is enforcing.
        let bob = rig.spawn(BodySpec::at(start, SHAPE).with_inertia(1.0, 1e-4));
        rig.joints.push(Joint::Pin {
            a: pivot,
            b: bob,
            local_a: Vec2::ZERO,
            local_b: vec2(-start[0], -start[1]),
        });

        // Time two zero crossings of x with a positive velocity: one full period.
        let mut previous_x = rig.position(bob).x;
        let mut crossings: Vec<f64> = Vec::new();
        for step in 0..20_000 {
            rig.step(DT, G);
            let x = rig.position(bob).x;
            if previous_x > 0.0 && x <= 0.0 {
                crossings.push(f64::from(step) * DT);
            }
            previous_x = x;
        }

        assert!(crossings.len() >= 3, "expected several swings, saw {}", crossings.len());
        // x crosses zero downward once per full swing, so two crossings span two periods.
        let measured = (crossings[2] - crossings[0]) / 2.0;
        let expected = core::f64::consts::TAU * (length / -G).sqrt();
        assert!(
            (measured - expected).abs() / expected < 0.02,
            "period {measured} s against the analytic {expected} s"
        );
    }

    /// A pin is two constraints, not one: the anchors must agree in both axes. A
    /// version that only constrained the joint direction would let the bob slide along
    /// the perpendicular, and a pendulum test alone would not notice.
    #[test]
    fn a_pin_holds_both_axes() {
        let mut rig = Rig::new(2);
        let frame = rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let arm = rig.spawn(
            BodySpec::at([1.0, 0.0], SHAPE)
                .with_inertia(2.0, 0.7)
                .with_velocity([5.0, -3.0])
                .with_angular_velocity(4.0),
        );
        rig.joints.push(Joint::Pin {
            a: frame,
            b: arm,
            local_a: Vec2::ZERO,
            local_b: vec2(-1.0, 0.0),
        });

        for _ in 0..4000 {
            rig.step(DT, G);
        }
        // Whatever it did, the anchor point must still be at the origin.
        let anchor = Vec2::from(rig.bodies.to_world_point(arm, [-1.0, 0.0]));
        assert!(anchor.length() < 20.0 * JOINT_SLOP, "the pin drifted to {anchor:?}");
    }

    /// A spring is a force law, so it must behave like one: released from rest it
    /// oscillates at `sqrt(k/m)` and damping brings it to the rest length.
    #[test]
    fn a_spring_oscillates_at_its_natural_frequency() {
        let (stiffness, mass) = (100.0, 1.0);
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let bob = rig.spawn(BodySpec::at([0.0, -1.5], SHAPE).with_inertia(mass, 0.1));
        rig.joints.push(Joint::Spring {
            a: 0,
            b: bob,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            stiffness,
            damping: 0.0,
        });

        let mut crossings: Vec<f64> = Vec::new();
        let mut previous = rig.position(bob).y;
        for step in 0..12_000 {
            rig.step(DT, 0.0);
            let y = rig.position(bob).y;
            if previous < -1.0 && y >= -1.0 {
                crossings.push(f64::from(step) * DT);
            }
            previous = y;
        }
        assert!(crossings.len() >= 3, "expected several oscillations");
        // Each upward crossing of the rest length is one full oscillation.
        let measured = crossings[2] - crossings[0];
        let expected = 2.0 * core::f64::consts::TAU / (stiffness / mass).sqrt();
        assert!(
            (measured - expected).abs() / expected < 0.02,
            "period {measured} s against sqrt(k/m) giving {expected} s"
        );
    }

    #[test]
    fn damping_brings_a_spring_to_rest_at_its_rest_length() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let bob = rig.spawn(BodySpec::at([0.0, -1.5], SHAPE).with_inertia(1.0, 0.1));
        rig.joints.push(Joint::Spring {
            a: 0,
            b: bob,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            stiffness: 100.0,
            damping: 8.0,
        });

        for _ in 0..12_000 {
            rig.step(DT, 0.0);
        }
        assert!(
            (rig.position(bob).y + 1.0).abs() < 1e-3,
            "settled at {}, wanted -1.0",
            rig.position(bob).y
        );
        assert!(rig.bodies.vel_y()[bob].abs() < 1e-3);
    }

    /// A spring imposes a timestep limit and must say so, rather than exploding at a
    /// step a caller had no way to know was too big.
    #[test]
    fn a_stiff_spring_reports_a_timestep_limit() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let bob = rig.spawn(BodySpec::at([0.0, -1.0], SHAPE).with_inertia(1.0, 0.1));

        let spring = |stiffness| Joint::Spring {
            a: 0,
            b: bob,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            stiffness,
            damping: 0.0,
        };
        let (soft, stiff) = (spring(100.0), spring(10_000.0));

        let soft_limit = soft.stable_step(&rig.bodies);
        let stiff_limit = stiff.stable_step(&rig.bodies);
        assert!(soft_limit.is_finite() && stiff_limit.is_finite());
        // A hundred times the stiffness is ten times the frequency, so a tenth of the
        // step. That square-root relationship is the whole reason a stiff spring is
        // expensive: making it ten times stiffer only costs three times the steps.
        assert!(
            (soft_limit / stiff_limit - 10.0).abs() < 1e-9,
            "{soft_limit} vs {stiff_limit}"
        );
        // A hard joint imposes none.
        let pin = Joint::Pin { a: 0, b: bob, local_a: Vec2::ZERO, local_b: Vec2::ZERO };
        assert_eq!(pin.stable_step(&rig.bodies), f64::INFINITY);
    }

    /// A motor reaches its target when it can.
    #[test]
    fn a_motor_reaches_its_target_speed() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let wheel = rig.spawn(BodySpec::at([0.0, 0.0], SHAPE).with_inertia(1.0, 0.5));
        rig.joints.push(Joint::Motor { a: wheel, b: 0, target_speed: 3.0, max_torque: 100.0 });

        for _ in 0..2000 {
            rig.step(DT, 0.0);
        }
        assert!(
            (rig.bodies.omega()[wheel] - 3.0).abs() < 1e-6,
            "spun up to {}",
            rig.bodies.omega()[wheel]
        );
    }

    /// And gives up gracefully when it cannot. A motor with an unbounded budget would
    /// move any load, which is a servo nobody has.
    #[test]
    fn a_motor_stalls_at_its_torque_budget() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        // A very heavy wheel, driven by a very weak motor.
        let wheel = rig.spawn(BodySpec::at([0.0, 0.0], SHAPE).with_inertia(1.0, 1000.0));
        rig.joints.push(Joint::Motor { a: wheel, b: 0, target_speed: 50.0, max_torque: 1.0 });

        let steps = 480;
        for _ in 0..steps {
            rig.step(DT, 0.0);
        }
        // At most torque*time/I of angular velocity, whatever the target says.
        let ceiling = 1.0 * (f64::from(steps) * DT) / 1000.0;
        let omega = rig.bodies.omega()[wheel];
        assert!(omega > 0.0, "it should be turning");
        assert!(
            omega <= ceiling * 1.05,
            "{omega} rad/s exceeds the {ceiling} the torque budget allows"
        );
    }

    /// Every joint applies equal and opposite impulses, so a jointed assembly floating
    /// in free space cannot push itself anywhere.
    #[test]
    fn joints_conserve_momentum_in_free_space() {
        let mut rig = Rig::new(3);
        let a = rig.spawn(BodySpec::at([-1.0, 0.0], SHAPE).with_inertia(1.0, 0.3).with_velocity([0.0, 1.0]));
        let b = rig.spawn(BodySpec::at([0.0, 0.0], SHAPE).with_inertia(2.0, 0.4).with_angular_velocity(2.0));
        let c = rig.spawn(BodySpec::at([1.0, 0.0], SHAPE).with_inertia(3.0, 0.5).with_velocity([0.0, -0.5]));
        rig.joints.push(Joint::Pin { a, b, local_a: vec2(1.0, 0.0), local_b: vec2(-1.0, 0.0) });
        rig.joints.push(Joint::Distance {
            a: b,
            b: c,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            rope: false,
        });
        rig.joints.push(Joint::Spring {
            a,
            b: c,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 2.0,
            stiffness: 20.0,
            damping: 0.5,
        });

        let linear_before = rig.bodies.linear_momentum();
        let angular_before = rig.bodies.angular_momentum();
        for _ in 0..3000 {
            rig.step(DT, 0.0);
        }
        let linear_after = rig.bodies.linear_momentum();
        let scale = rig.bodies.momentum_scale().max(1.0);

        assert!(
            (linear_after[0] - linear_before[0]).abs() / scale < 1e-10
                && (linear_after[1] - linear_before[1]).abs() / scale < 1e-10,
            "{linear_before:?} -> {linear_after:?}"
        );
        assert!(
            (rig.bodies.angular_momentum() - angular_before).abs() / scale < 1e-8,
            "angular {angular_before} -> {}",
            rig.bodies.angular_momentum()
        );
    }

    /// A joint whose endpoints do not exist, or which joins a body to itself, must be
    /// ignored rather than indexing out of bounds or dividing by zero.
    #[test]
    fn malformed_joints_are_ignored_rather_than_panicking() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::at([0.0, 0.0], SHAPE).with_inertia(1.0, 1.0));
        rig.joints.push(Joint::Pin { a: 0, b: 99, local_a: Vec2::ZERO, local_b: Vec2::ZERO });
        rig.joints.push(Joint::Pin { a: 0, b: 0, local_a: Vec2::ZERO, local_b: Vec2::ZERO });
        rig.joints.push(Joint::Distance {
            a: 0,
            b: 0,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            rope: false,
        });

        for _ in 0..100 {
            rig.step(DT, G);
        }
        assert!(!rig.bodies.has_non_finite());
        assert_eq!(rig.solver.max_error(&rig.joints), 0.0);
    }

    /// Two static bodies present infinite mass; no impulse can change anything.
    #[test]
    fn a_joint_between_two_static_bodies_does_nothing() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        rig.spawn(BodySpec::statik([5.0, 0.0], SHAPE));
        rig.joints.push(Joint::Distance {
            a: 0,
            b: 1,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            rope: false,
        });
        for _ in 0..100 {
            rig.step(DT, G);
        }
        assert_eq!(rig.position(1), vec2(5.0, 0.0));
        assert!(!rig.bodies.has_non_finite());
    }

    #[test]
    fn joint_error_is_reported() {
        let mut rig = Rig::new(2);
        rig.spawn(BodySpec::statik([0.0, 0.0], SHAPE));
        let bob = rig.spawn(BodySpec::at([3.0, 0.0], SHAPE).with_inertia(1.0, 0.5));
        rig.joints.push(Joint::Distance {
            a: 0,
            b: bob,
            local_a: Vec2::ZERO,
            local_b: Vec2::ZERO,
            rest_length: 1.0,
            rope: false,
        });

        // Prepared but not yet solved: the error is the full 2 m of stretch.
        rig.solver.prepare(&rig.bodies, &rig.joints, false);
        assert!((rig.solver.max_error(&rig.joints) - 2.0).abs() < 1e-12);

        for _ in 0..3000 {
            rig.step(DT, 0.0);
        }
        rig.solver.prepare(&rig.bodies, &rig.joints, false);
        assert!(
            rig.solver.max_error(&rig.joints) < 10.0 * JOINT_SLOP,
            "still {} m out",
            rig.solver.max_error(&rig.joints)
        );
    }
}
