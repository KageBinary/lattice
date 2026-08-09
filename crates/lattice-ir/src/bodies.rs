//! Structure-of-arrays storage for 2D rigid bodies.
//!
//! The rotational counterpart of [`crate::particles`], and the same decisions apply:
//! hot state in parallel `f64` arrays, live bodies packed into `0..len`, and
//! generation-checked handles so a stale [`BodyId`] fails to resolve rather than
//! aliasing whatever now occupies its slot.
//!
//! # Why inverse mass and inverse inertia are stored
//!
//! A sequential-impulse solver divides by mass and inertia several times per contact
//! per iteration. Storing the reciprocals is the usual trade, but the real reason is
//! that it makes *static* bodies fall out of the same arithmetic: a body with zero
//! inverse mass and zero inverse inertia absorbs any impulse without moving, so the
//! solver has no branch for "is this the ground?" — the ground is a body whose
//! reciprocals are zero. A branchless static case is one fewer place for a wall to
//! quietly acquire momentum.
//!
//! # Why orientation is a cosine/sine pair
//!
//! Rotating a body-local contact point into world space happens for every contact of
//! every body on every iteration. Storing the pair means that costs two multiplies
//! rather than a `sin`/`cos`, and composition stays exact instead of accumulating an
//! angle that has to be range-reduced. The cost is one renormalization per body per
//! step, which [`RigidBodyStore::integrate`] does.

use crate::ids::{BodyId, ShapeId};

/// Sentinel for a destroyed body's slot.
const DEAD: u32 = u32::MAX;

/// Indirection entry: where a stable index currently lives.
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// Current slot, or [`DEAD`].
    slot: u32,
    /// Bumped on every destruction so stale handles stop resolving.
    generation: u32,
}

/// Structure-of-arrays storage for 2D rigid bodies.
#[derive(Debug)]
pub struct RigidBodyStore {
    // --- Pose ------------------------------------------------------------------
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    /// Cosine of the orientation angle.
    rot_cos: Vec<f64>,
    /// Sine of the orientation angle.
    rot_sin: Vec<f64>,

    // --- Motion ----------------------------------------------------------------
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    /// Angular velocity about the centre of mass, rad/s.
    omega: Vec<f64>,

    // --- Accumulators, cleared each step ---------------------------------------
    force_x: Vec<f64>,
    force_y: Vec<f64>,
    /// Torque about the centre of mass, N·m.
    torque: Vec<f64>,

    // --- Inertial properties ---------------------------------------------------
    mass: Vec<f64>,
    /// Rotational inertia about the centre of mass, kg·m².
    inertia: Vec<f64>,
    /// `1/mass`, or 0 for a static body.
    inv_mass: Vec<f64>,
    /// `1/inertia`, or 0 for a static body or one pinned against rotation.
    inv_inertia: Vec<f64>,

    // --- Geometry --------------------------------------------------------------
    /// Which registered collider this body wears.
    shape: Vec<ShapeId>,

    // --- Identity --------------------------------------------------------------
    entries: Vec<Entry>,
    stable_of_slot: Vec<u32>,
    free_indices: Vec<u32>,

    len: usize,
    capacity: usize,
}

impl RigidBodyStore {
    /// Allocate storage for exactly `capacity` bodies.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pos_x: vec![0.0; capacity],
            pos_y: vec![0.0; capacity],
            rot_cos: vec![1.0; capacity],
            rot_sin: vec![0.0; capacity],
            vel_x: vec![0.0; capacity],
            vel_y: vec![0.0; capacity],
            omega: vec![0.0; capacity],
            force_x: vec![0.0; capacity],
            force_y: vec![0.0; capacity],
            torque: vec![0.0; capacity],
            mass: vec![0.0; capacity],
            inertia: vec![0.0; capacity],
            inv_mass: vec![0.0; capacity],
            inv_inertia: vec![0.0; capacity],
            shape: vec![ShapeId::from_index(0); capacity],
            entries: Vec::with_capacity(capacity),
            stable_of_slot: vec![0; capacity],
            free_indices: Vec::with_capacity(capacity),
            len: 0,
            capacity,
        }
    }

    /// Bytes of hot state held per body, for the memory report (§19.3).
    pub const BYTES_PER_BODY: usize = 14 * core::mem::size_of::<f64>()
        + core::mem::size_of::<ShapeId>()
        + 2 * core::mem::size_of::<u32>();

    /// Number of live bodies.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no bodies are alive.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of bodies this store can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Create a body. Returns `None` when the store is full.
    pub fn spawn(&mut self, spec: BodySpec) -> Option<BodyId> {
        if self.len == self.capacity {
            return None;
        }
        let slot = self.len as u32;
        let index = match self.free_indices.pop() {
            Some(reused) => {
                self.entries[reused as usize].slot = slot;
                reused
            }
            None => {
                let fresh = self.entries.len() as u32;
                self.entries.push(Entry { slot, generation: 0 });
                fresh
            }
        };

        let s = slot as usize;
        self.pos_x[s] = spec.position[0];
        self.pos_y[s] = spec.position[1];
        self.rot_cos[s] = spec.angle.cos();
        self.rot_sin[s] = spec.angle.sin();
        self.vel_x[s] = spec.velocity[0];
        self.vel_y[s] = spec.velocity[1];
        self.omega[s] = spec.angular_velocity;
        self.force_x[s] = 0.0;
        self.force_y[s] = 0.0;
        self.torque[s] = 0.0;
        self.mass[s] = spec.mass;
        self.inertia[s] = spec.inertia;
        self.inv_mass[s] = reciprocal_or_zero(spec.mass);
        self.inv_inertia[s] = if spec.fixed_rotation { 0.0 } else { reciprocal_or_zero(spec.inertia) };
        self.shape[s] = spec.shape;
        self.stable_of_slot[s] = index;
        self.len += 1;

        Some(BodyId::new(index, self.entries[index as usize].generation))
    }

    /// Destroy a body. Returns false if the handle is stale.
    pub fn despawn(&mut self, id: BodyId) -> bool {
        let Some(slot) = self.slot_of(id) else { return false };
        let last = self.len - 1;
        if slot != last {
            self.move_slot(last, slot);
            let moved_stable = self.stable_of_slot[last];
            self.entries[moved_stable as usize].slot = slot as u32;
            self.stable_of_slot[slot] = moved_stable;
        }
        self.len -= 1;

        let entry = &mut self.entries[id.index() as usize];
        entry.slot = DEAD;
        entry.generation = entry.generation.wrapping_add(1);
        self.free_indices.push(id.index());
        true
    }

    /// Resolve a handle to its current slot, or `None` if the body is gone.
    pub fn slot_of(&self, id: BodyId) -> Option<usize> {
        let entry = *self.entries.get(id.index() as usize)?;
        if entry.slot == DEAD || entry.generation != id.generation() {
            return None;
        }
        Some(entry.slot as usize)
    }

    /// True if the handle still refers to a live body.
    pub fn is_alive(&self, id: BodyId) -> bool {
        self.slot_of(id).is_some()
    }

    /// Remove every body, keeping the allocation.
    pub fn clear(&mut self) {
        self.len = 0;
        self.entries.clear();
        self.free_indices.clear();
    }

    /// Copy every per-body array entry from `src` to `dst`.
    ///
    /// Every channel must be listed here and nowhere else. Adding a channel means
    /// editing this function and the struct — a missing line here is a body that keeps
    /// its position and inherits someone else's velocity when a neighbour is destroyed,
    /// which looks like a physics bug and is not one.
    fn move_slot(&mut self, src: usize, dst: usize) {
        for array in [
            &mut self.pos_x,
            &mut self.pos_y,
            &mut self.rot_cos,
            &mut self.rot_sin,
            &mut self.vel_x,
            &mut self.vel_y,
            &mut self.omega,
            &mut self.force_x,
            &mut self.force_y,
            &mut self.torque,
            &mut self.mass,
            &mut self.inertia,
            &mut self.inv_mass,
            &mut self.inv_inertia,
        ] {
            array[dst] = array[src];
        }
        self.shape[dst] = self.shape[src];
    }

    // --- Read-only views ------------------------------------------------------

    /// Centre-of-mass x coordinates, m.
    pub fn pos_x(&self) -> &[f64] {
        &self.pos_x[..self.len]
    }
    /// Centre-of-mass y coordinates, m.
    pub fn pos_y(&self) -> &[f64] {
        &self.pos_y[..self.len]
    }
    /// Cosines of the orientation angles.
    pub fn rot_cos(&self) -> &[f64] {
        &self.rot_cos[..self.len]
    }
    /// Sines of the orientation angles.
    pub fn rot_sin(&self) -> &[f64] {
        &self.rot_sin[..self.len]
    }
    /// Linear velocity x components, m/s.
    pub fn vel_x(&self) -> &[f64] {
        &self.vel_x[..self.len]
    }
    /// Linear velocity y components, m/s.
    pub fn vel_y(&self) -> &[f64] {
        &self.vel_y[..self.len]
    }
    /// Angular velocities, rad/s.
    pub fn omega(&self) -> &[f64] {
        &self.omega[..self.len]
    }
    /// Masses, kg.
    pub fn mass(&self) -> &[f64] {
        &self.mass[..self.len]
    }
    /// Rotational inertias about the centre of mass, kg·m².
    pub fn inertia(&self) -> &[f64] {
        &self.inertia[..self.len]
    }
    /// Inverse masses, 1/kg. Zero for a static body.
    pub fn inv_mass(&self) -> &[f64] {
        &self.inv_mass[..self.len]
    }
    /// Inverse inertias, 1/(kg·m²). Zero for a static or rotation-locked body.
    pub fn inv_inertia(&self) -> &[f64] {
        &self.inv_inertia[..self.len]
    }
    /// Which collider each body wears.
    pub fn shape(&self) -> &[ShapeId] {
        &self.shape[..self.len]
    }
    /// Accumulated forces, x components, N.
    pub fn force_x(&self) -> &[f64] {
        &self.force_x[..self.len]
    }
    /// Accumulated forces, y components, N.
    pub fn force_y(&self) -> &[f64] {
        &self.force_y[..self.len]
    }
    /// Accumulated torques, N·m.
    pub fn torque(&self) -> &[f64] {
        &self.torque[..self.len]
    }

    /// True when a body cannot be moved by any impulse.
    pub fn is_static(&self, slot: usize) -> bool {
        self.inv_mass[slot] == 0.0 && self.inv_inertia[slot] == 0.0
    }

    /// Position of a body by handle.
    pub fn position_of(&self, id: BodyId) -> Option<[f64; 2]> {
        let slot = self.slot_of(id)?;
        Some([self.pos_x[slot], self.pos_y[slot]])
    }

    /// Orientation of a body by handle, in radians.
    pub fn angle_of(&self, id: BodyId) -> Option<f64> {
        let slot = self.slot_of(id)?;
        Some(self.rot_sin[slot].atan2(self.rot_cos[slot]))
    }

    /// Linear velocity of a body by handle.
    pub fn velocity_of(&self, id: BodyId) -> Option<[f64; 2]> {
        let slot = self.slot_of(id)?;
        Some([self.vel_x[slot], self.vel_y[slot]])
    }

    /// Angular velocity of a body by handle.
    pub fn angular_velocity_of(&self, id: BodyId) -> Option<f64> {
        Some(self.omega[self.slot_of(id)?])
    }

    /// World-space velocity of the body-frame point `local`.
    ///
    /// `v + ω × r` — the quantity every contact constraint is written in terms of.
    pub fn point_velocity(&self, slot: usize, local: [f64; 2]) -> [f64; 2] {
        let r = self.to_world_vector(slot, local);
        let omega = self.omega[slot];
        [self.vel_x[slot] - omega * r[1], self.vel_y[slot] + omega * r[0]]
    }

    /// Rotate a body-local *vector* into world space. No translation.
    pub fn to_world_vector(&self, slot: usize, local: [f64; 2]) -> [f64; 2] {
        let (c, s) = (self.rot_cos[slot], self.rot_sin[slot]);
        [c * local[0] - s * local[1], s * local[0] + c * local[1]]
    }

    /// Transform a body-local *point* into world space.
    pub fn to_world_point(&self, slot: usize, local: [f64; 2]) -> [f64; 2] {
        let v = self.to_world_vector(slot, local);
        [self.pos_x[slot] + v[0], self.pos_y[slot] + v[1]]
    }

    /// Transform a world point into a body's local frame.
    ///
    /// The inverse of [`RigidBodyStore::to_world_point`]. What a click needs: the
    /// question "did the cursor land on this body" is asked in the body's own frame,
    /// where its shape is defined.
    pub fn to_local_point(&self, slot: usize, world: [f64; 2]) -> [f64; 2] {
        let (c, s) = (self.rot_cos[slot], self.rot_sin[slot]);
        let d = [world[0] - self.pos_x[slot], world[1] - self.pos_y[slot]];
        // The transpose, since a rotation is orthogonal.
        [c * d[0] + s * d[1], -s * d[0] + c * d[1]]
    }

    /// The handle of the body currently in `slot`.
    ///
    /// The reverse of [`RigidBodyStore::slot_of`], and the one a picker needs: a click
    /// finds a slot by walking the hot arrays, and everything it wants to *do* with the
    /// result — hold it, delete it — has to survive the next despawn. A slot does not;
    /// a handle does.
    pub fn id_at(&self, slot: usize) -> Option<BodyId> {
        if slot >= self.len {
            return None;
        }
        let stable = self.stable_of_slot[slot];
        let entry = self.entries.get(stable as usize)?;
        Some(BodyId::new(stable, entry.generation))
    }

    // --- Mutation --------------------------------------------------------------

    /// Zero every force and torque accumulator.
    pub fn clear_forces(&mut self) {
        self.force_x[..self.len].fill(0.0);
        self.force_y[..self.len].fill(0.0);
        self.torque[..self.len].fill(0.0);
    }

    /// Add a force through the centre of mass, producing no torque.
    pub fn add_force(&mut self, slot: usize, force: [f64; 2]) {
        self.force_x[slot] += force[0];
        self.force_y[slot] += force[1];
    }

    /// Add a force at a world-space point, producing the torque its lever arm implies.
    pub fn add_force_at(&mut self, slot: usize, force: [f64; 2], world_point: [f64; 2]) {
        self.force_x[slot] += force[0];
        self.force_y[slot] += force[1];
        let r = [world_point[0] - self.pos_x[slot], world_point[1] - self.pos_y[slot]];
        self.torque[slot] += r[0] * force[1] - r[1] * force[0];
    }

    /// Add a pure torque.
    pub fn add_torque(&mut self, slot: usize, torque: f64) {
        self.torque[slot] += torque;
    }

    /// Apply an impulse at a world-space point, changing velocity immediately.
    ///
    /// The primitive the contact and constraint solvers are built from. A static body
    /// has zero reciprocals, so this is a no-op on it without a branch.
    pub fn apply_impulse_at(&mut self, slot: usize, impulse: [f64; 2], world_point: [f64; 2]) {
        let r = [world_point[0] - self.pos_x[slot], world_point[1] - self.pos_y[slot]];
        self.apply_impulse_at_arm(slot, impulse, r);
    }

    /// Apply an impulse whose lever arm from the centre of mass is already known.
    ///
    /// The contact solver computes that arm once per contact and reuses it across
    /// iterations, so it would be wasteful to rederive it from a world point.
    pub fn apply_impulse_at_arm(&mut self, slot: usize, impulse: [f64; 2], arm: [f64; 2]) {
        let inv_m = self.inv_mass[slot];
        self.vel_x[slot] += inv_m * impulse[0];
        self.vel_y[slot] += inv_m * impulse[1];
        self.omega[slot] += self.inv_inertia[slot] * (arm[0] * impulse[1] - arm[1] * impulse[0]);
    }

    /// Place a body, by handle.
    pub fn set_pose(&mut self, id: BodyId, position: [f64; 2], angle: f64) -> bool {
        let Some(slot) = self.slot_of(id) else { return false };
        self.pos_x[slot] = position[0];
        self.pos_y[slot] = position[1];
        self.rot_cos[slot] = angle.cos();
        self.rot_sin[slot] = angle.sin();
        true
    }

    /// Set a body's velocity, by handle.
    pub fn set_velocity(&mut self, id: BodyId, velocity: [f64; 2], angular: f64) -> bool {
        let Some(slot) = self.slot_of(id) else { return false };
        self.vel_x[slot] = velocity[0];
        self.vel_y[slot] = velocity[1];
        self.omega[slot] = angular;
        true
    }

    /// Change a body's mass, scaling its rotational inertia with it.
    ///
    /// Inertia is `∫r²dm`, so for a fixed shape and uniform density it is proportional to
    /// mass. Setting mass alone would leave a body that translates like a feather and spins
    /// like a boulder — physically possible, but never what someone dragging a mass slider
    /// meant, and impossible to get back to by eye. Scaling keeps the shape's mass
    /// distribution and changes only how much of it there is.
    ///
    /// A `mass` of zero or less makes the body **static**: infinite mass, immovable. That
    /// is the same convention [`BodySpec`] uses, and it is why the reciprocals are stored
    /// rather than recomputed — a solver dividing by mass in its inner loop must not
    /// branch on this.
    ///
    /// Returns false if the handle is stale. A body whose rotation was fixed at spawn
    /// keeps its fixed rotation.
    pub fn set_mass(&mut self, id: BodyId, mass: f64) -> bool {
        let Some(slot) = self.slot_of(id) else { return false };
        if !mass.is_finite() {
            return false;
        }

        let previous = self.mass[slot];
        let fixed_rotation = self.inv_inertia[slot] == 0.0 && self.inertia[slot] > 0.0;

        let inertia = if previous > 0.0 && mass > 0.0 {
            self.inertia[slot] * (mass / previous)
        } else if mass > 0.0 {
            // Coming back from static, where the stored inertia is zero and the ratio
            // above would keep it there. Nothing here knows the shape, so the caller gets
            // a body that translates and does not spin until it re-derives an inertia.
            0.0
        } else {
            0.0
        };

        let mass = mass.max(0.0);
        self.mass[slot] = mass;
        self.inertia[slot] = inertia;
        self.inv_mass[slot] = reciprocal_or_zero(mass);
        self.inv_inertia[slot] =
            if fixed_rotation { 0.0 } else { reciprocal_or_zero(inertia) };
        true
    }

    /// Change a body's rotational inertia directly, leaving its mass alone.
    ///
    /// Separate from [`set_mass`](Self::set_mass) because the two are independent
    /// properties once you stop assuming a uniform solid — a flywheel and a disc of the
    /// same mass are different objects.
    pub fn set_inertia(&mut self, id: BodyId, inertia: f64) -> bool {
        let Some(slot) = self.slot_of(id) else { return false };
        if !inertia.is_finite() {
            return false;
        }
        let inertia = inertia.max(0.0);
        self.inertia[slot] = inertia;
        self.inv_inertia[slot] = reciprocal_or_zero(inertia);
        true
    }

    /// Mutable access to velocities and pose, for an integrator or solver.
    pub fn motion(&mut self) -> Motion<'_> {
        let n = self.len;
        Motion {
            pos_x: &mut self.pos_x[..n],
            pos_y: &mut self.pos_y[..n],
            rot_cos: &mut self.rot_cos[..n],
            rot_sin: &mut self.rot_sin[..n],
            vel_x: &mut self.vel_x[..n],
            vel_y: &mut self.vel_y[..n],
            omega: &mut self.omega[..n],
            force_x: &self.force_x[..n],
            force_y: &self.force_y[..n],
            torque: &self.torque[..n],
            inv_mass: &self.inv_mass[..n],
            inv_inertia: &self.inv_inertia[..n],
        }
    }

    /// Advance velocities from the accumulated forces, then positions from the
    /// velocities — semi-implicit (symplectic) Euler.
    ///
    /// Not velocity Verlet, which the particle module uses. A contact solver corrects
    /// velocities *after* the force integration and *before* the position update, and
    /// Verlet's second force evaluation would happen after those corrections had
    /// already been applied — so the scheme's order advantage evaporates and its
    /// bookkeeping does not. Semi-implicit Euler is what the whole family of
    /// sequential-impulse solvers is built around, and it is symplectic, so a body in
    /// free flight still has bounded energy error.
    ///
    /// Call [`RigidBodyStore::integrate_velocities`] and
    /// [`RigidBodyStore::integrate_positions`] separately when a solver needs to run
    /// between the two halves. This is the convenience path for the no-contact case.
    pub fn integrate(&mut self, dt: f64) {
        self.integrate_velocities(dt);
        self.integrate_positions(dt);
    }

    /// The first half-step: `v += F/m · dt`, `ω += τ/I · dt`.
    pub fn integrate_velocities(&mut self, dt: f64) {
        for i in 0..self.len {
            self.vel_x[i] += self.inv_mass[i] * self.force_x[i] * dt;
            self.vel_y[i] += self.inv_mass[i] * self.force_y[i] * dt;
            self.omega[i] += self.inv_inertia[i] * self.torque[i] * dt;
        }
    }

    /// The second half-step: `x += v · dt`, and the orientation by `ω · dt`.
    ///
    /// The rotation is advanced by composing with the exact rotation for the angle
    /// swept, not by adding `ω·dt` to the stored cosine and sine. The additive form
    /// drifts off the unit circle within a few hundred steps, which shows up as a body
    /// that slowly changes size.
    pub fn integrate_positions(&mut self, dt: f64) {
        for i in 0..self.len {
            self.pos_x[i] += self.vel_x[i] * dt;
            self.pos_y[i] += self.vel_y[i] * dt;

            let swept = self.omega[i] * dt;
            if swept != 0.0 {
                let (sin, cos) = swept.sin_cos();
                let (c, s) = (self.rot_cos[i], self.rot_sin[i]);
                let new_cos = c * cos - s * sin;
                let new_sin = s * cos + c * sin;
                // Renormalize anyway: the composition is exact in exact arithmetic, and
                // this bounds the accumulation of rounding in it.
                let magnitude = (new_cos * new_cos + new_sin * new_sin).sqrt();
                if magnitude > 0.0 && magnitude.is_finite() {
                    self.rot_cos[i] = new_cos / magnitude;
                    self.rot_sin[i] = new_sin / magnitude;
                }
            }
        }
    }

    // --- Aggregates -------------------------------------------------------------

    /// Total kinetic energy, `Σ ½mv² + ½Iω²`, in joules.
    pub fn kinetic_energy(&self) -> f64 {
        (0..self.len)
            .map(|i| {
                let v2 = self.vel_x[i] * self.vel_x[i] + self.vel_y[i] * self.vel_y[i];
                0.5 * self.mass[i] * v2 + 0.5 * self.inertia[i] * self.omega[i] * self.omega[i]
            })
            .sum()
    }

    /// Total linear momentum, kg·m/s.
    ///
    /// Static bodies are excluded. A wall has an infinite mass expressed as a zero
    /// reciprocal, and its *stored* mass is zero — so including it would contribute
    /// nothing anyway, but the exclusion is explicit because "the total momentum of a
    /// system containing a wall" is not a conserved quantity and should not be
    /// presented as one.
    pub fn linear_momentum(&self) -> [f64; 2] {
        let mut total = [0.0, 0.0];
        for i in 0..self.len {
            if self.is_static(i) {
                continue;
            }
            total[0] += self.mass[i] * self.vel_x[i];
            total[1] += self.mass[i] * self.vel_y[i];
        }
        total
    }

    /// Total angular momentum about the world origin, kg·m²/s.
    ///
    /// `Σ (I ω + m (r × v))` — the spin of each body plus the orbital term from its
    /// centre of mass moving about the origin. Omitting the orbital half is the
    /// classic way to have a test that passes for a spinning body and fails for a
    /// thrown one.
    pub fn angular_momentum(&self) -> f64 {
        (0..self.len)
            .filter(|&i| !self.is_static(i))
            .map(|i| {
                let orbital =
                    self.mass[i] * (self.pos_x[i] * self.vel_y[i] - self.pos_y[i] * self.vel_x[i]);
                self.inertia[i] * self.omega[i] + orbital
            })
            .sum()
    }

    /// Sum of the individual momentum magnitudes, kg·m/s.
    ///
    /// The scale a *net* momentum of zero should be judged against — see
    /// [`crate::diagnostics`] and the viewer's drift reporting. A residual of 1e-13
    /// means nothing on its own and everything when the parts summing to it are of
    /// order 100.
    pub fn momentum_scale(&self) -> f64 {
        (0..self.len)
            .filter(|&i| !self.is_static(i))
            .map(|i| {
                self.mass[i] * (self.vel_x[i] * self.vel_x[i] + self.vel_y[i] * self.vel_y[i]).sqrt()
            })
            .sum()
    }

    /// True when any body has a non-finite position, velocity or orientation.
    ///
    /// NFR-007: the runtime halts on the first non-finite value rather than producing
    /// confident nonsense for another thousand steps.
    pub fn has_non_finite(&self) -> bool {
        (0..self.len).any(|i| {
            !(self.pos_x[i].is_finite()
                && self.pos_y[i].is_finite()
                && self.vel_x[i].is_finite()
                && self.vel_y[i].is_finite()
                && self.omega[i].is_finite()
                && self.rot_cos[i].is_finite()
                && self.rot_sin[i].is_finite())
        })
    }

    /// Fastest linear speed among live bodies, m/s.
    pub fn max_speed(&self) -> f64 {
        (0..self.len)
            .map(|i| (self.vel_x[i] * self.vel_x[i] + self.vel_y[i] * self.vel_y[i]).sqrt())
            .fold(0.0, f64::max)
    }

    /// Largest angular speed among live bodies, rad/s.
    pub fn max_angular_speed(&self) -> f64 {
        (0..self.len).map(|i| self.omega[i].abs()).fold(0.0, f64::max)
    }
}

/// `1/value`, or zero when the value is not a usable mass.
///
/// Infinity, zero and NaN all mean "does not respond to impulses", which is exactly
/// what a zero reciprocal expresses.
fn reciprocal_or_zero(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        1.0 / value
    } else {
        0.0
    }
}

/// How to create a body.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BodySpec {
    /// Centre of mass, m.
    pub position: [f64; 2],
    /// Orientation, radians.
    pub angle: f64,
    /// Linear velocity, m/s.
    pub velocity: [f64; 2],
    /// Angular velocity, rad/s.
    pub angular_velocity: f64,
    /// Mass, kg. Zero or infinite makes the body static.
    pub mass: f64,
    /// Rotational inertia about the centre of mass, kg·m².
    pub inertia: f64,
    /// Refuse to rotate even under torque, keeping the mass finite.
    ///
    /// Not the same as infinite inertia expressed as a huge number: a large-but-finite
    /// inertia still rotates, just slowly, and a character that slowly tips over is a
    /// harder bug to see than one that never does.
    pub fixed_rotation: bool,
    /// Which registered collider to wear.
    pub shape: ShapeId,
}

impl Default for BodySpec {
    fn default() -> Self {
        BodySpec {
            position: [0.0, 0.0],
            angle: 0.0,
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            mass: 1.0,
            inertia: 1.0,
            fixed_rotation: false,
            shape: ShapeId::from_index(0),
        }
    }
}

impl BodySpec {
    /// A dynamic body at `position` wearing `shape`.
    pub fn at(position: [f64; 2], shape: ShapeId) -> BodySpec {
        BodySpec { position, shape, ..BodySpec::default() }
    }

    /// A body that no impulse can move.
    pub fn statik(position: [f64; 2], shape: ShapeId) -> BodySpec {
        BodySpec { position, shape, mass: 0.0, inertia: 0.0, ..BodySpec::default() }
    }

    /// Set the mass and rotational inertia.
    pub fn with_inertia(mut self, mass: f64, inertia: f64) -> BodySpec {
        self.mass = mass;
        self.inertia = inertia;
        self
    }

    /// Set the orientation, radians.
    pub fn with_angle(mut self, angle: f64) -> BodySpec {
        self.angle = angle;
        self
    }

    /// Set the linear velocity.
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> BodySpec {
        self.velocity = velocity;
        self
    }

    /// Set the angular velocity, rad/s.
    pub fn with_angular_velocity(mut self, angular_velocity: f64) -> BodySpec {
        self.angular_velocity = angular_velocity;
        self
    }

    /// Lock the body against rotation.
    pub fn with_fixed_rotation(mut self) -> BodySpec {
        self.fixed_rotation = true;
        self
    }
}

/// Mutable views of the arrays an integrator or contact solver touches.
///
/// Handing out one struct of disjoint slices rather than a `&mut RigidBodyStore` lets
/// a solver hold velocities and read forces at once without fighting the borrow
/// checker, and makes the set of arrays a hot loop touches visible in its signature.
#[derive(Debug)]
pub struct Motion<'a> {
    /// Centre-of-mass x, m.
    pub pos_x: &'a mut [f64],
    /// Centre-of-mass y, m.
    pub pos_y: &'a mut [f64],
    /// Cosine of the orientation.
    pub rot_cos: &'a mut [f64],
    /// Sine of the orientation.
    pub rot_sin: &'a mut [f64],
    /// Linear velocity x, m/s.
    pub vel_x: &'a mut [f64],
    /// Linear velocity y, m/s.
    pub vel_y: &'a mut [f64],
    /// Angular velocity, rad/s.
    pub omega: &'a mut [f64],
    /// Accumulated force x, N.
    pub force_x: &'a [f64],
    /// Accumulated force y, N.
    pub force_y: &'a [f64],
    /// Accumulated torque, N·m.
    pub torque: &'a [f64],
    /// Inverse masses, 1/kg.
    pub inv_mass: &'a [f64],
    /// Inverse inertias, 1/(kg·m²).
    pub inv_inertia: &'a [f64],
}

impl Motion<'_> {
    /// Number of bodies in the view.
    pub fn len(&self) -> usize {
        self.vel_x.len()
    }

    /// True when the view is empty.
    pub fn is_empty(&self) -> bool {
        self.vel_x.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;
    const SHAPE: ShapeId = ShapeId::from_index(0);

    fn store_with(specs: &[BodySpec]) -> RigidBodyStore {
        let mut store = RigidBodyStore::with_capacity(specs.len().max(1));
        for spec in specs {
            store.spawn(*spec).expect("capacity was sized for these");
        }
        store
    }

    #[test]
    fn a_full_store_refuses_rather_than_reallocating() {
        let mut store = RigidBodyStore::with_capacity(2);
        assert!(store.spawn(BodySpec::at([0.0, 0.0], SHAPE)).is_some());
        assert!(store.spawn(BodySpec::at([1.0, 0.0], SHAPE)).is_some());
        assert!(store.spawn(BodySpec::at([2.0, 0.0], SHAPE)).is_none(), "NFR-001: no growth");
        assert_eq!(store.len(), 2);
    }

    /// The whole reason handles carry a generation.
    #[test]
    fn a_handle_to_a_destroyed_body_stops_resolving() {
        let mut store = RigidBodyStore::with_capacity(4);
        let first = store.spawn(BodySpec::at([1.0, 0.0], SHAPE)).unwrap();
        let second = store.spawn(BodySpec::at([2.0, 0.0], SHAPE)).unwrap();

        assert!(store.despawn(first));
        assert!(!store.is_alive(first));
        assert!(!store.despawn(first), "despawning twice must not succeed");
        assert_eq!(store.position_of(first), None);

        // The survivor moved into the freed slot but kept its identity and its state.
        assert!(store.is_alive(second));
        assert_eq!(store.position_of(second), Some([2.0, 0.0]));

        // A new body reuses the index, and the old handle still does not resolve.
        let third = store.spawn(BodySpec::at([3.0, 0.0], SHAPE)).unwrap();
        assert_eq!(third.index(), first.index(), "index space is reused");
        assert_ne!(third.generation(), first.generation());
        assert_eq!(store.position_of(first), None, "the stale handle must not alias the new body");
    }

    /// A missing line in `move_slot` looks exactly like a physics bug.
    #[test]
    fn a_swap_carries_every_channel_with_it() {
        let mut store = RigidBodyStore::with_capacity(3);
        let doomed = store.spawn(BodySpec::at([0.0, 0.0], SHAPE)).unwrap();
        let survivor = store
            .spawn(
                BodySpec::at([9.0, -3.0], ShapeId::from_index(2))
                    .with_inertia(4.0, 7.0)
                    .with_angle(0.5)
                    .with_velocity([1.5, -2.5])
                    .with_angular_velocity(0.25),
            )
            .unwrap();
        let slot = store.slot_of(survivor).unwrap();
        store.add_force(slot, [11.0, 13.0]);
        store.add_torque(slot, 17.0);

        store.despawn(doomed);
        let slot = store.slot_of(survivor).unwrap();

        assert_eq!(store.position_of(survivor), Some([9.0, -3.0]));
        assert_eq!(store.velocity_of(survivor), Some([1.5, -2.5]));
        assert_eq!(store.angular_velocity_of(survivor), Some(0.25));
        assert!((store.angle_of(survivor).unwrap() - 0.5).abs() < EPSILON);
        assert_eq!(store.mass()[slot], 4.0);
        assert_eq!(store.inertia()[slot], 7.0);
        assert!((store.inv_mass()[slot] - 0.25).abs() < EPSILON);
        assert!((store.inv_inertia()[slot] - 1.0 / 7.0).abs() < EPSILON);
        assert_eq!(store.shape()[slot], ShapeId::from_index(2));
        assert_eq!(store.force_x()[slot], 11.0);
        assert_eq!(store.force_y()[slot], 13.0);
        assert_eq!(store.torque()[slot], 17.0);
    }

    /// Zero, infinite and NaN masses all mean "immovable", and none of them may
    /// produce an infinite or NaN reciprocal that then poisons a velocity.
    #[test]
    fn unusable_masses_become_zero_reciprocals() {
        for mass in [0.0, f64::INFINITY, f64::NAN, -1.0] {
            let store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(mass, mass)]);
            assert_eq!(store.inv_mass()[0], 0.0, "mass {mass}");
            assert_eq!(store.inv_inertia()[0], 0.0, "inertia {mass}");
            assert!(store.is_static(0));
        }
    }

    /// A click finds a slot; everything it then wants to do has to survive a despawn,
    /// and a slot does not.
    #[test]
    fn a_slot_resolves_back_to_the_handle_that_owns_it() {
        let mut store = RigidBodyStore::with_capacity(4);
        let first = store.spawn(BodySpec::at([1.0, 0.0], SHAPE)).unwrap();
        let second = store.spawn(BodySpec::at([2.0, 0.0], SHAPE)).unwrap();

        assert_eq!(store.id_at(0), Some(first));
        assert_eq!(store.id_at(1), Some(second));
        assert_eq!(store.id_at(2), None, "past the live count");

        // After a despawn the survivor has moved, and the slot now names it.
        store.despawn(first);
        assert_eq!(store.id_at(0), Some(second));
        assert_eq!(store.slot_of(second), Some(0));
        assert_eq!(store.id_at(1), None);
    }

    /// Hit-testing happens in the body's own frame, where its shape is defined.
    #[test]
    fn world_and_local_points_round_trip_through_a_rotated_body() {
        let mut store = RigidBodyStore::with_capacity(1);
        store
            .spawn(BodySpec::at([3.0, -2.0], SHAPE).with_angle(0.7))
            .unwrap();

        for local in [[0.0, 0.0], [1.0, 0.0], [-0.5, 0.25]] {
            let world = store.to_world_point(0, local);
            let back = store.to_local_point(0, world);
            assert!((back[0] - local[0]).abs() < EPSILON, "{back:?} vs {local:?}");
            assert!((back[1] - local[1]).abs() < EPSILON, "{back:?} vs {local:?}");
        }
        // The centre of mass is the local origin, whatever the rotation.
        assert_eq!(store.to_local_point(0, [3.0, -2.0]), [0.0, 0.0]);
    }

    #[test]
    fn a_static_body_absorbs_an_impulse_without_moving() {
        let mut store = store_with(&[BodySpec::statik([0.0, 0.0], SHAPE)]);
        store.apply_impulse_at(0, [1000.0, -500.0], [3.0, 4.0]);
        assert_eq!(store.vel_x()[0], 0.0);
        assert_eq!(store.vel_y()[0], 0.0);
        assert_eq!(store.omega()[0], 0.0);
    }

    #[test]
    fn fixed_rotation_locks_spin_without_making_the_body_static() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)
            .with_inertia(2.0, 3.0)
            .with_fixed_rotation()]);
        store.apply_impulse_at(0, [0.0, 1.0], [5.0, 0.0]);
        assert!((store.vel_y()[0] - 0.5).abs() < EPSILON, "it still moves");
        assert_eq!(store.omega()[0], 0.0, "but it does not turn");
        assert!(!store.is_static(0));
    }

    /// An off-centre force must produce the torque its lever arm implies, and the
    /// sign must follow the right-hand rule — a transposed cross product here would
    /// spin every body backwards and conserve momentum while doing it.
    #[test]
    fn an_offset_force_produces_torque_with_the_right_sign() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(1.0, 1.0)]);
        // Push +y at a point on the +x side: counter-clockwise, so positive torque.
        store.add_force_at(0, [0.0, 10.0], [2.0, 0.0]);
        assert_eq!(store.torque()[0], 20.0);
        assert_eq!(store.force_y()[0], 10.0, "the force still acts on the centre of mass");

        store.clear_forces();
        store.add_force_at(0, [0.0, 10.0], [-2.0, 0.0]);
        assert_eq!(store.torque()[0], -20.0, "the other side turns the other way");
    }

    /// `v + ω × r`: the identity every contact constraint is written in.
    #[test]
    fn a_point_on_a_spinning_body_moves_perpendicular_to_its_arm() {
        let store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)
            .with_velocity([1.0, 0.0])
            .with_angular_velocity(2.0)]);
        // A point 3 m along +x, spinning counter-clockwise at 2 rad/s, moves at
        // 6 m/s in +y — plus the body's own 1 m/s in +x.
        let v = store.point_velocity(0, [3.0, 0.0]);
        assert!((v[0] - 1.0).abs() < EPSILON, "{v:?}");
        assert!((v[1] - 6.0).abs() < EPSILON, "{v:?}");
    }

    #[test]
    fn free_flight_is_exact_for_constant_velocity() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_velocity([2.0, -1.0])]);
        for _ in 0..100 {
            store.integrate(0.01);
        }
        assert!((store.pos_x()[0] - 2.0).abs() < 1e-12);
        assert!((store.pos_y()[0] + 1.0).abs() < 1e-12);
    }

    /// Additively integrating cos/sin drifts off the unit circle within a few hundred
    /// steps, which renders as a body that slowly changes size.
    #[test]
    fn a_spinning_body_stays_on_the_unit_circle() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_angular_velocity(7.0)]);
        for _ in 0..100_000 {
            store.integrate_positions(1e-3);
        }
        let magnitude = (store.rot_cos()[0].powi(2) + store.rot_sin()[0].powi(2)).sqrt();
        assert!((magnitude - 1.0).abs() < 1e-12, "drifted to {magnitude}");

        // And after a whole number of turns it is back where it started.
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)
            .with_angular_velocity(core::f64::consts::TAU)]);
        for _ in 0..1000 {
            store.integrate_positions(1e-3);
        }
        assert!(store.angle_of(BodyId::new(0, 0)).unwrap().abs() < 1e-9, "one full turn");
    }

    /// Inertia is `∫r²dm`, so for a fixed shape it is proportional to mass. A mass edit
    /// that left it alone would give a body that translates like a feather and spins like
    /// a boulder.
    #[test]
    fn setting_mass_scales_the_inertia_with_it() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(2.0, 8.0)]);
        let id = BodyId::new(0, 0);

        assert!(store.set_mass(id, 6.0));
        assert!((store.mass()[0] - 6.0).abs() < EPSILON);
        assert!((store.inertia()[0] - 24.0).abs() < EPSILON, "tripled with the mass");
        assert!((store.inv_mass()[0] - 1.0 / 6.0).abs() < EPSILON);
        assert!((store.inv_inertia()[0] - 1.0 / 24.0).abs() < EPSILON);
    }

    /// The two are independent once you stop assuming a uniform solid: a flywheel and a
    /// disc of the same mass are different objects.
    #[test]
    fn inertia_can_be_set_without_disturbing_the_mass() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(2.0, 8.0)]);
        let id = BodyId::new(0, 0);

        assert!(store.set_inertia(id, 1.0));
        assert!((store.inertia()[0] - 1.0).abs() < EPSILON);
        assert!((store.inv_inertia()[0] - 1.0).abs() < EPSILON);
        assert!((store.mass()[0] - 2.0).abs() < EPSILON, "mass must not move");
    }

    /// Zero mass is the store's own spelling of "static", and the reciprocals have to
    /// agree with it — a solver divides by them in its inner loop without branching.
    #[test]
    fn a_mass_of_zero_makes_a_body_static_with_consistent_reciprocals() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(2.0, 8.0)]);
        let id = BodyId::new(0, 0);

        assert!(store.set_mass(id, 0.0));
        assert!(store.is_static(0));
        assert_eq!(store.inv_mass()[0], 0.0);
        assert_eq!(store.inv_inertia()[0], 0.0);
    }

    #[test]
    fn a_non_finite_or_stale_edit_is_refused() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(2.0, 8.0)]);
        let id = BodyId::new(0, 0);

        assert!(!store.set_mass(id, f64::NAN), "NaN mass accepted");
        assert!(!store.set_inertia(id, f64::INFINITY), "infinite inertia accepted");
        assert!((store.mass()[0] - 2.0).abs() < EPSILON, "a refused edit still changed it");

        store.despawn(id);
        assert!(!store.set_mass(id, 1.0), "a stale handle was honoured");
    }

    /// A body spawned with its rotation fixed keeps it. Otherwise a mass edit would
    /// quietly hand it back the spin its author disabled.
    #[test]
    fn a_fixed_rotation_survives_a_mass_change() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)
            .with_inertia(2.0, 8.0)
            .with_fixed_rotation()]);
        let id = BodyId::new(0, 0);
        assert_eq!(store.inv_inertia()[0], 0.0, "it starts fixed");

        assert!(store.set_mass(id, 5.0));
        assert_eq!(store.inv_inertia()[0], 0.0, "the mass edit unfroze the rotation");
    }

    #[test]
    fn forces_and_torques_integrate_into_motion() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE).with_inertia(2.0, 4.0)]);
        store.add_force(0, [4.0, 0.0]);
        store.add_torque(0, 8.0);
        store.integrate_velocities(0.5);
        assert!((store.vel_x()[0] - 1.0).abs() < EPSILON, "a = F/m = 2, v = 1");
        assert!((store.omega()[0] - 1.0).abs() < EPSILON, "α = τ/I = 2, ω = 1");
    }

    /// Angular momentum has an orbital term. Omitting it gives a test that passes for
    /// a spinning body and fails for a thrown one.
    #[test]
    fn angular_momentum_counts_both_spin_and_orbit() {
        // Pure spin at the origin: I·ω only.
        let spinning = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)
            .with_inertia(1.0, 3.0)
            .with_angular_velocity(2.0)]);
        assert!((spinning.angular_momentum() - 6.0).abs() < EPSILON);

        // Pure translation, offset from the origin: m·(r × v) only.
        let thrown = store_with(&[BodySpec::at([0.0, 5.0], SHAPE)
            .with_inertia(2.0, 3.0)
            .with_velocity([4.0, 0.0])]);
        // r × v = (0)(0) − (5)(4) = −20, times m = 2 → −40.
        assert!((thrown.angular_momentum() + 40.0).abs() < EPSILON, "{}", thrown.angular_momentum());
    }

    #[test]
    fn kinetic_energy_counts_both_translation_and_rotation() {
        let store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)
            .with_inertia(2.0, 3.0)
            .with_velocity([3.0, 4.0])
            .with_angular_velocity(2.0)]);
        // ½·2·25 + ½·3·4 = 25 + 6.
        assert!((store.kinetic_energy() - 31.0).abs() < EPSILON);
    }

    /// A wall is not part of the system's momentum budget, and presenting it as if it
    /// were would make every collision with the ground look like a conservation failure.
    #[test]
    fn static_bodies_are_left_out_of_the_momentum_totals() {
        let store = store_with(&[
            BodySpec::statik([0.0, 0.0], SHAPE).with_velocity([100.0, 100.0]),
            BodySpec::at([0.0, 1.0], SHAPE).with_inertia(2.0, 1.0).with_velocity([3.0, 0.0]),
        ]);
        assert_eq!(store.linear_momentum(), [6.0, 0.0]);
        assert!((store.momentum_scale() - 6.0).abs() < EPSILON);
    }

    #[test]
    fn non_finite_state_is_detected() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)]);
        assert!(!store.has_non_finite());
        store.add_force(0, [f64::INFINITY, 0.0]);
        store.integrate(1.0);
        assert!(store.has_non_finite(), "an infinite force must not pass unnoticed");
    }

    #[test]
    fn clearing_resets_everything() {
        let mut store = store_with(&[BodySpec::at([0.0, 0.0], SHAPE)]);
        let id = BodyId::new(0, 0);
        store.clear();
        assert!(store.is_empty());
        assert!(!store.is_alive(id));
        assert_eq!(store.kinetic_energy(), 0.0);
        assert_eq!(store.max_speed(), 0.0);
    }

    #[test]
    fn an_empty_store_reports_zeros_rather_than_nan() {
        let store = RigidBodyStore::with_capacity(4);
        assert_eq!(store.kinetic_energy(), 0.0);
        assert_eq!(store.linear_momentum(), [0.0, 0.0]);
        assert_eq!(store.angular_momentum(), 0.0);
        assert_eq!(store.max_speed(), 0.0);
        assert_eq!(store.max_angular_speed(), 0.0);
        assert!(!store.has_non_finite());
    }
}
