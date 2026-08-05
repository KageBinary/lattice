//! Structure-of-arrays particle storage with stable handles.
//!
//! Spec §11.2: *"Runtime storage should be structure-of-arrays with stable logical
//! IDs and compact active indices."* Those two requirements pull in opposite
//! directions — compact means data moves, stable means handles must not — so the
//! store keeps an indirection table between them.
//!
//! ```text
//!   ParticleId { index, generation }
//!            │
//!            ▼  entries[index] : Entry { slot, generation }
//!          slot ──────────────────────────────┐
//!                                             ▼
//!   pos_x  [ .. .. .. ][slot][ .. .. ]   ← contiguous, indices 0..len
//!   pos_y  [ .. .. .. ][slot][ .. .. ]
//!   vel_x  [ .. .. .. ][slot][ .. .. ]
//! ```
//!
//! Destroying a particle moves the last live particle into the vacated slot
//! (`swap_remove`) and patches the moved particle's entry. Kernels therefore always
//! walk `0..len` with no holes and no liveness test in the inner loop (P4).
//!
//! # Capacity is fixed
//!
//! [`ParticleStore::with_capacity`] allocates once. [`ParticleStore::spawn`] returns
//! `None` when full rather than reallocating, because a reallocation mid-step is
//! exactly the hidden hot-loop allocation NFR-001 forbids. Growing is an explicit
//! planning operation.

use crate::ids::{ParticleId, ParticleKind};

/// Marker stored in an entry whose particle has been destroyed.
const DEAD: u32 = u32::MAX;

/// Where a stable index currently points, and which generation it is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    /// Compact slot holding this particle's data, or [`DEAD`].
    slot: u32,
    /// Bumped on every destruction so stale handles stop resolving.
    generation: u32,
}

/// Structure-of-arrays storage for 2D particles.
#[derive(Debug)]
pub struct ParticleStore {
    // --- Hot state, indexed by slot, valid for 0..len -------------------------
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    force_x: Vec<f64>,
    force_y: Vec<f64>,
    mass: Vec<f64>,
    /// `1/mass`, or 0 for a particle pinned in place. Precomputed because the
    /// integrator divides by mass on every particle on every step.
    inv_mass: Vec<f64>,
    radius: Vec<f64>,
    kind: Vec<ParticleKind>,

    // --- Identity -------------------------------------------------------------
    /// Stable index → current slot and generation.
    entries: Vec<Entry>,
    /// Slot → stable index, so a swap can patch the moved particle's entry.
    stable_of_slot: Vec<u32>,
    /// Stable indices freed by destruction, reused before new ones are minted.
    free_indices: Vec<u32>,

    len: usize,
    capacity: usize,

    /// Reusable buffer for [`ParticleStore::reorder`], allocated once.
    permute_scratch: Vec<f64>,
    permute_scratch_u32: Vec<u32>,
}

impl ParticleStore {
    /// Allocate storage for exactly `capacity` particles.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pos_x: vec![0.0; capacity],
            pos_y: vec![0.0; capacity],
            vel_x: vec![0.0; capacity],
            vel_y: vec![0.0; capacity],
            force_x: vec![0.0; capacity],
            force_y: vec![0.0; capacity],
            mass: vec![0.0; capacity],
            inv_mass: vec![0.0; capacity],
            radius: vec![0.0; capacity],
            kind: vec![ParticleKind::from_index(0); capacity],
            entries: Vec::with_capacity(capacity),
            stable_of_slot: vec![0; capacity],
            free_indices: Vec::with_capacity(capacity),
            len: 0,
            capacity,
            permute_scratch: vec![0.0; capacity],
            permute_scratch_u32: vec![0; capacity],
        }
    }

    /// Number of live particles.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no particles are alive.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of particles this store can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of hot state held per particle, for the memory report (§19.3).
    pub const BYTES_PER_PARTICLE: usize = 9 * core::mem::size_of::<f64>()
        + core::mem::size_of::<ParticleKind>()
        + 2 * core::mem::size_of::<u32>();

    /// Create a particle. Returns `None` when the store is full.
    ///
    /// A mass of `f64::INFINITY` pins the particle: its inverse mass is zero, so
    /// forces accumulate on it but never move it.
    pub fn spawn(&mut self, spec: ParticleSpec) -> Option<ParticleId> {
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
        self.vel_x[s] = spec.velocity[0];
        self.vel_y[s] = spec.velocity[1];
        self.force_x[s] = 0.0;
        self.force_y[s] = 0.0;
        self.mass[s] = spec.mass;
        self.inv_mass[s] = if spec.mass.is_finite() && spec.mass > 0.0 { 1.0 / spec.mass } else { 0.0 };
        self.radius[s] = spec.radius;
        self.kind[s] = spec.kind;
        self.stable_of_slot[s] = index;
        self.len += 1;

        Some(ParticleId::new(index, self.entries[index as usize].generation))
    }

    /// Destroy a particle. Returns false if the handle is stale.
    pub fn despawn(&mut self, id: ParticleId) -> bool {
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

    /// Resolve a handle to its current slot, or `None` if the particle is gone.
    pub fn slot_of(&self, id: ParticleId) -> Option<usize> {
        let entry = *self.entries.get(id.index() as usize)?;
        if entry.slot == DEAD || entry.generation != id.generation() {
            return None;
        }
        Some(entry.slot as usize)
    }

    /// True if the handle still refers to a live particle.
    pub fn is_alive(&self, id: ParticleId) -> bool {
        self.slot_of(id).is_some()
    }

    /// Remove every particle, keeping the allocation.
    pub fn clear(&mut self) {
        self.len = 0;
        self.entries.clear();
        self.free_indices.clear();
    }

    /// Copy every per-particle array entry from `src` to `dst`.
    ///
    /// Every `f64` channel must be listed here, in [`Self::reorder`], and nowhere
    /// else. Adding a channel means editing these two functions and the struct.
    fn move_slot(&mut self, src: usize, dst: usize) {
        for array in [
            &mut self.pos_x,
            &mut self.pos_y,
            &mut self.vel_x,
            &mut self.vel_y,
            &mut self.force_x,
            &mut self.force_y,
            &mut self.mass,
            &mut self.inv_mass,
            &mut self.radius,
        ] {
            array[dst] = array[src];
        }
        self.kind[dst] = self.kind[src];
    }

    /// Permute live particles into a new order, patching all handles.
    ///
    /// `new_order[k]` names the slot that should end up at position `k`. Spec §11.2
    /// calls for sorting particles by spatial cell to improve locality; this is the
    /// operation that applies such a sort. Uses a preallocated scratch buffer, so it
    /// performs no allocation.
    ///
    /// # Panics
    ///
    /// If `new_order` is not a permutation of `0..len`. A malformed permutation would
    /// silently duplicate or drop particles, which is far worse than a panic.
    pub fn reorder(&mut self, new_order: &[u32]) {
        assert_eq!(new_order.len(), self.len, "permutation length must equal the live count");
        debug_assert!(is_permutation(new_order), "reorder requires a permutation of 0..len");

        let len = self.len;
        let scratch = &mut self.permute_scratch;
        for array in [
            &mut self.pos_x,
            &mut self.pos_y,
            &mut self.vel_x,
            &mut self.vel_y,
            &mut self.force_x,
            &mut self.force_y,
            &mut self.mass,
            &mut self.inv_mass,
            &mut self.radius,
        ] {
            for (k, &from) in new_order.iter().enumerate() {
                scratch[k] = array[from as usize];
            }
            array[..len].copy_from_slice(&scratch[..len]);
        }

        // `kind` is not f64, so it gets its own pass using the u32 scratch.
        let kind_scratch = &mut self.permute_scratch_u32;
        for (k, &from) in new_order.iter().enumerate() {
            kind_scratch[k] = self.kind[from as usize].raw();
        }
        for (k, &raw) in kind_scratch[..len].iter().enumerate() {
            self.kind[k] = ParticleKind::from_index(raw);
        }

        // Rebuild the slot→stable map, then repoint every stable entry.
        for (k, &from) in new_order.iter().enumerate() {
            kind_scratch[k] = self.stable_of_slot[from as usize];
        }
        self.stable_of_slot[..len].copy_from_slice(&kind_scratch[..len]);
        for slot in 0..len {
            let stable = self.stable_of_slot[slot];
            self.entries[stable as usize].slot = slot as u32;
        }
    }

    // --- Read-only slices -------------------------------------------------------

    /// Live x positions.
    pub fn pos_x(&self) -> &[f64] {
        &self.pos_x[..self.len]
    }
    /// Live y positions.
    pub fn pos_y(&self) -> &[f64] {
        &self.pos_y[..self.len]
    }
    /// Live x velocities.
    pub fn vel_x(&self) -> &[f64] {
        &self.vel_x[..self.len]
    }
    /// Live y velocities.
    pub fn vel_y(&self) -> &[f64] {
        &self.vel_y[..self.len]
    }
    /// Live x forces.
    pub fn force_x(&self) -> &[f64] {
        &self.force_x[..self.len]
    }
    /// Live y forces.
    pub fn force_y(&self) -> &[f64] {
        &self.force_y[..self.len]
    }
    /// Live masses.
    pub fn mass(&self) -> &[f64] {
        &self.mass[..self.len]
    }
    /// Live inverse masses (0 for pinned particles).
    pub fn inv_mass(&self) -> &[f64] {
        &self.inv_mass[..self.len]
    }
    /// Live radii.
    pub fn radius(&self) -> &[f64] {
        &self.radius[..self.len]
    }
    /// Live particle kinds.
    pub fn kind(&self) -> &[ParticleKind] {
        &self.kind[..self.len]
    }

    /// Position of one particle, by handle.
    pub fn position_of(&self, id: ParticleId) -> Option<[f64; 2]> {
        let s = self.slot_of(id)?;
        Some([self.pos_x[s], self.pos_y[s]])
    }

    /// Velocity of one particle, by handle.
    pub fn velocity_of(&self, id: ParticleId) -> Option<[f64; 2]> {
        let s = self.slot_of(id)?;
        Some([self.vel_x[s], self.vel_y[s]])
    }

    // --- Mutable views ----------------------------------------------------------

    /// Zero every force accumulator. Called at the top of each force evaluation.
    pub fn clear_forces(&mut self) {
        self.force_x[..self.len].fill(0.0);
        self.force_y[..self.len].fill(0.0);
    }

    /// Borrow the arrays an integrator needs, as disjoint mutable slices.
    ///
    /// Handing out one struct of slices rather than a `&mut self` lets an integrator
    /// hold position, velocity and force simultaneously without fighting the borrow
    /// checker or reaching for interior mutability.
    pub fn dynamics(&mut self) -> Dynamics<'_> {
        let n = self.len;
        Dynamics {
            pos_x: &mut self.pos_x[..n],
            pos_y: &mut self.pos_y[..n],
            vel_x: &mut self.vel_x[..n],
            vel_y: &mut self.vel_y[..n],
            force_x: &mut self.force_x[..n],
            force_y: &mut self.force_y[..n],
            inv_mass: &self.inv_mass[..n],
            mass: &self.mass[..n],
        }
    }

    /// Borrow positions and forces, for a force kernel that must not move anything.
    pub fn force_accumulation(&mut self) -> ForceAccumulation<'_> {
        let n = self.len;
        ForceAccumulation {
            pos_x: &self.pos_x[..n],
            pos_y: &self.pos_y[..n],
            vel_x: &self.vel_x[..n],
            vel_y: &self.vel_y[..n],
            force_x: &mut self.force_x[..n],
            force_y: &mut self.force_y[..n],
            mass: &self.mass[..n],
            radius: &self.radius[..n],
            kind: &self.kind[..n],
        }
    }

    /// Set a particle's position, by handle. Returns false for a stale handle.
    pub fn set_position(&mut self, id: ParticleId, position: [f64; 2]) -> bool {
        let Some(s) = self.slot_of(id) else { return false };
        self.pos_x[s] = position[0];
        self.pos_y[s] = position[1];
        true
    }

    /// Set a particle's velocity, by handle. Returns false for a stale handle.
    pub fn set_velocity(&mut self, id: ParticleId, velocity: [f64; 2]) -> bool {
        let Some(s) = self.slot_of(id) else { return false };
        self.vel_x[s] = velocity[0];
        self.vel_y[s] = velocity[1];
        true
    }
}

/// Everything needed to create a particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleSpec {
    /// Position in metres.
    pub position: [f64; 2],
    /// Velocity in m/s.
    pub velocity: [f64; 2],
    /// Mass in kg. `f64::INFINITY` pins the particle in place.
    pub mass: f64,
    /// Collision radius in metres. Zero for point particles.
    pub radius: f64,
    /// Selects material parameters.
    pub kind: ParticleKind,
}

impl Default for ParticleSpec {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            mass: 1.0,
            radius: 0.0,
            kind: ParticleKind::from_index(0),
        }
    }
}

impl ParticleSpec {
    /// A unit-mass point particle at `position`.
    pub fn at(position: [f64; 2]) -> Self {
        Self { position, ..Default::default() }
    }

    /// Set the velocity.
    pub fn with_velocity(mut self, velocity: [f64; 2]) -> Self {
        self.velocity = velocity;
        self
    }

    /// Set the mass.
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    /// Set the collision radius.
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set the particle kind.
    pub fn with_kind(mut self, kind: ParticleKind) -> Self {
        self.kind = kind;
        self
    }
}

/// Disjoint mutable slices for an integrator.
#[derive(Debug)]
pub struct Dynamics<'a> {
    /// x positions, m.
    pub pos_x: &'a mut [f64],
    /// y positions, m.
    pub pos_y: &'a mut [f64],
    /// x velocities, m/s.
    pub vel_x: &'a mut [f64],
    /// y velocities, m/s.
    pub vel_y: &'a mut [f64],
    /// x forces, N.
    pub force_x: &'a mut [f64],
    /// y forces, N.
    pub force_y: &'a mut [f64],
    /// Inverse masses, 1/kg. Zero for pinned particles.
    pub inv_mass: &'a [f64],
    /// Masses, kg.
    pub mass: &'a [f64],
}

impl Dynamics<'_> {
    /// Number of particles in this view.
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }

    /// True when the view is empty.
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }
}

/// Read-only state plus writable force accumulators, for a force kernel.
#[derive(Debug)]
pub struct ForceAccumulation<'a> {
    /// x positions, m.
    pub pos_x: &'a [f64],
    /// y positions, m.
    pub pos_y: &'a [f64],
    /// x velocities, m/s.
    pub vel_x: &'a [f64],
    /// y velocities, m/s.
    pub vel_y: &'a [f64],
    /// x force accumulator, N.
    pub force_x: &'a mut [f64],
    /// y force accumulator, N.
    pub force_y: &'a mut [f64],
    /// Masses, kg.
    pub mass: &'a [f64],
    /// Collision radii, m.
    pub radius: &'a [f64],
    /// Particle kinds.
    pub kind: &'a [crate::ids::ParticleKind],
}

impl ForceAccumulation<'_> {
    /// Number of particles in this view.
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }

    /// True when the view is empty.
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }
}

/// Debug-only check that a slice is a permutation of `0..len`.
fn is_permutation(order: &[u32]) -> bool {
    let mut seen = vec![false; order.len()];
    for &v in order {
        let Some(slot) = seen.get_mut(v as usize) else { return false };
        if *slot {
            return false;
        }
        *slot = true;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(n: usize) -> ParticleStore {
        let mut s = ParticleStore::with_capacity(16);
        for i in 0..n {
            s.spawn(ParticleSpec::at([i as f64, 0.0]).with_mass(1.0 + i as f64)).unwrap();
        }
        s
    }

    #[test]
    fn spawn_and_read_back() {
        let mut s = ParticleStore::with_capacity(4);
        let id = s
            .spawn(ParticleSpec::at([1.0, 2.0]).with_velocity([3.0, 4.0]).with_mass(2.0))
            .unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.position_of(id), Some([1.0, 2.0]));
        assert_eq!(s.velocity_of(id), Some([3.0, 4.0]));
        assert_eq!(s.mass()[0], 2.0);
        assert_eq!(s.inv_mass()[0], 0.5);
    }

    #[test]
    fn spawn_fails_at_capacity_rather_than_reallocating() {
        let mut s = ParticleStore::with_capacity(2);
        assert!(s.spawn(ParticleSpec::default()).is_some());
        assert!(s.spawn(ParticleSpec::default()).is_some());
        assert!(s.spawn(ParticleSpec::default()).is_none(), "must refuse rather than grow");
        assert_eq!(s.capacity(), 2);
    }

    #[test]
    fn infinite_mass_pins_a_particle() {
        let mut s = ParticleStore::with_capacity(2);
        s.spawn(ParticleSpec::default().with_mass(f64::INFINITY)).unwrap();
        assert_eq!(s.inv_mass()[0], 0.0);
    }

    #[test]
    fn despawn_compacts_and_keeps_other_handles_valid() {
        let mut s = ParticleStore::with_capacity(8);
        let a = s.spawn(ParticleSpec::at([0.0, 0.0])).unwrap();
        let b = s.spawn(ParticleSpec::at([1.0, 0.0])).unwrap();
        let c = s.spawn(ParticleSpec::at([2.0, 0.0])).unwrap();

        assert!(s.despawn(b));
        assert_eq!(s.len(), 2);
        // The array must have no hole: both survivors live in 0..2.
        let xs = s.pos_x();
        assert_eq!(xs.len(), 2);
        assert!(xs.contains(&0.0) && xs.contains(&2.0));
        // And both survivors' handles still resolve to their own data.
        assert_eq!(s.position_of(a), Some([0.0, 0.0]));
        assert_eq!(s.position_of(c), Some([2.0, 0.0]));
    }

    #[test]
    fn stale_handles_do_not_resolve_to_recycled_slots() {
        let mut s = ParticleStore::with_capacity(4);
        let a = s.spawn(ParticleSpec::at([1.0, 1.0])).unwrap();
        assert!(s.despawn(a));
        assert!(!s.is_alive(a));
        assert_eq!(s.position_of(a), None);

        // The next spawn reuses a's stable index but must not answer to a's handle.
        let b = s.spawn(ParticleSpec::at([9.0, 9.0])).unwrap();
        assert_eq!(b.index(), a.index(), "index should be recycled");
        assert_ne!(b.generation(), a.generation(), "generation must advance");
        assert_eq!(s.position_of(a), None, "stale handle must not read the new particle");
        assert_eq!(s.position_of(b), Some([9.0, 9.0]));
    }

    #[test]
    fn despawn_of_a_stale_handle_is_a_no_op() {
        let mut s = ParticleStore::with_capacity(4);
        let a = s.spawn(ParticleSpec::default()).unwrap();
        assert!(s.despawn(a));
        assert!(!s.despawn(a), "double despawn must not corrupt the store");
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn despawning_the_last_particle_needs_no_swap() {
        let mut s = store_with(3);
        let ids: Vec<_> = (0..3).map(|i| ParticleId::new(i, 0)).collect();
        assert!(s.despawn(ids[2]));
        assert_eq!(s.len(), 2);
        assert_eq!(s.pos_x(), &[0.0, 1.0]);
    }

    #[test]
    fn churn_does_not_leak_index_space() {
        let mut s = ParticleStore::with_capacity(4);
        for _ in 0..10_000 {
            let id = s.spawn(ParticleSpec::default()).unwrap();
            assert!(s.despawn(id));
        }
        // One stable entry, reused ten thousand times.
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn reorder_permutes_data_and_repoints_handles() {
        let mut s = ParticleStore::with_capacity(8);
        let a = s.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(1.0)).unwrap();
        let b = s.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(2.0)).unwrap();
        let c = s.spawn(ParticleSpec::at([2.0, 0.0]).with_mass(3.0)).unwrap();

        // Reverse the order, as a spatial sort might.
        s.reorder(&[2, 1, 0]);

        assert_eq!(s.pos_x(), &[2.0, 1.0, 0.0]);
        assert_eq!(s.mass(), &[3.0, 2.0, 1.0]);
        // Handles must follow their particles.
        assert_eq!(s.position_of(a), Some([0.0, 0.0]));
        assert_eq!(s.position_of(b), Some([1.0, 0.0]));
        assert_eq!(s.position_of(c), Some([2.0, 0.0]));
        assert_eq!(s.slot_of(a), Some(2));
        assert_eq!(s.slot_of(c), Some(0));
    }

    #[test]
    fn reorder_survives_prior_despawns() {
        let mut s = ParticleStore::with_capacity(8);
        let a = s.spawn(ParticleSpec::at([0.0, 0.0])).unwrap();
        let b = s.spawn(ParticleSpec::at([1.0, 0.0])).unwrap();
        let c = s.spawn(ParticleSpec::at([2.0, 0.0])).unwrap();
        let d = s.spawn(ParticleSpec::at([3.0, 0.0])).unwrap();
        assert!(s.despawn(b));
        // Live set is now {a, d, c} in some order; reverse it.
        let order: Vec<u32> = (0..s.len() as u32).rev().collect();
        s.reorder(&order);
        for (id, expected) in [(a, 0.0), (c, 2.0), (d, 3.0)] {
            assert_eq!(s.position_of(id).map(|p| p[0]), Some(expected));
        }
        assert!(!s.is_alive(b));
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn reorder_rejects_a_wrong_length_permutation() {
        let mut s = store_with(3);
        s.reorder(&[0, 1]);
    }

    #[test]
    fn clear_forces_only_touches_live_particles() {
        let mut s = store_with(3);
        {
            let d = s.dynamics();
            d.force_x.fill(5.0);
            d.force_y.fill(5.0);
        }
        s.clear_forces();
        assert!(s.force_x().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dynamics_view_exposes_disjoint_slices() {
        let mut s = store_with(4);
        let d = s.dynamics();
        assert_eq!(d.len(), 4);
        // All six mutable slices are live simultaneously.
        d.force_x[0] = 1.0;
        d.force_y[0] = 2.0;
        d.pos_x[1] = 3.0;
        d.pos_y[1] = 4.0;
        d.vel_x[2] = 5.0;
        d.vel_y[2] = 6.0;
        assert_eq!((d.force_x[0], d.pos_x[1], d.vel_x[2]), (1.0, 3.0, 5.0));
    }

    #[test]
    fn slices_never_expose_dead_slots() {
        let mut s = store_with(5);
        assert_eq!(s.pos_x().len(), 5);
        let id = ParticleId::new(0, 0);
        s.despawn(id);
        assert_eq!(s.pos_x().len(), 4);
        assert_eq!(s.mass().len(), 4);
        assert_eq!(s.kind().len(), 4);
    }

    #[test]
    fn is_permutation_detects_bad_input() {
        assert!(is_permutation(&[0, 1, 2]));
        assert!(is_permutation(&[2, 0, 1]));
        assert!(!is_permutation(&[0, 0, 1]), "duplicate");
        assert!(!is_permutation(&[0, 1, 3]), "out of range");
    }
}
