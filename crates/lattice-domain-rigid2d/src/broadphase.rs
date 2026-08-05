//! Finding which pairs of bodies are worth testing exactly.
//!
//! Spec §11.1 offers a choice — *"broadphase grid/BVH"* — and this is neither. It is
//! **sweep and prune**: sort the bodies by the left edge of their bounding box, then
//! walk that order, comparing each body only against those whose boxes have not yet
//! ended.
//!
//! # Why not a uniform grid
//!
//! A grid needs a cell size, and there is no good way to choose one for a scene that
//! contains both a 0.1 m pebble and a 50 m ground segment. Size the cells for the
//! pebble and the ground occupies five hundred of them; size them for the ground and
//! every pebble in a pile lands in one cell, which is the O(n²) case the grid was
//! supposed to avoid. The particle module can use a grid because its particles are all
//! the same size — a rigid-body scene's whole point is that they are not.
//!
//! Sweep and prune has no such parameter. It costs a sort, which is `O(n log n)`, and
//! degrades gracefully: the worst case is everything overlapping on the sweep axis,
//! which is also the case where the pairs are real.
//!
//! # Determinism
//!
//! FR-011 requires a reproducible run, so the pair list must not depend on iteration
//! order or on how a sort broke a tie. The sort key is `(min.x, slot)`, which is a
//! total order, and pairs are emitted in canonical `(low, high)` slot order. Two runs
//! of the same model produce the same pairs in the same sequence.

use lattice_ir::RigidBodyStore;

use crate::contact::BodyPair;
use crate::math::{vec2, Rot, Transform};
use crate::shape::{Aabb, Collider, Shape};

/// How far each bounding box is grown before testing for overlap, m.
///
/// A body that will touch next step is worth reporting this step: it lets the solver
/// see the contact one step early and stop the approach rather than resolve an overlap
/// after the fact. It also stops a resting contact being lost and re-found as the
/// bodies settle by a micrometre, which would discard its warm-start impulse each time.
pub const CONTACT_MARGIN: f64 = 2.0 * crate::narrowphase::LINEAR_SLOP;

/// Reusable buffers for pair finding.
///
/// Kept across steps so the broadphase performs no allocation once the scene has
/// reached its size (NFR-001).
#[derive(Clone, Default, Debug)]
pub struct BroadPhase {
    /// Per-slot bounding boxes, grown by [`CONTACT_MARGIN`].
    bounds: Vec<Aabb>,
    /// Slots sorted by the left edge of their box.
    order: Vec<usize>,
    /// The candidate pairs found by the last sweep.
    pairs: Vec<BodyPair>,
    /// How many pairs the last sweep tested, for the diagnostics panel.
    tested: usize,
}

impl BroadPhase {
    /// An empty broadphase.
    pub fn new() -> BroadPhase {
        BroadPhase::default()
    }

    /// Reserve room for `capacity` bodies, so the first step does not allocate either.
    pub fn with_capacity(capacity: usize) -> BroadPhase {
        BroadPhase {
            bounds: Vec::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            pairs: Vec::with_capacity(capacity * 4),
            tested: 0,
        }
    }

    /// Recompute bounding boxes and find every pair worth an exact test.
    ///
    /// `shape_of` supplies the collider a body wears. Static-static pairs are skipped:
    /// two walls never need resolving, and in a scene with a long ground segment they
    /// would otherwise be most of the pairs.
    pub fn find_pairs<'shapes, F>(&mut self, bodies: &RigidBodyStore, shape_of: F) -> &[BodyPair]
    where
        F: Fn(usize) -> Option<&'shapes Shape>,
    {
        self.bounds.clear();
        self.order.clear();
        self.pairs.clear();
        self.tested = 0;

        for slot in 0..bodies.len() {
            let transform = transform_of(bodies, slot);
            let bounds = match shape_of(slot) {
                Some(shape) => shape.aabb(transform).grown(CONTACT_MARGIN),
                // A body with no registered collider still occupies a slot, so the
                // indices stay aligned; an empty box never overlaps anything.
                None => Aabb::EMPTY,
            };
            self.bounds.push(bounds);
            self.order.push(slot);
        }

        // `(min.x, slot)` is a total order, so the sort is deterministic whatever the
        // algorithm does with ties.
        let bounds = &self.bounds;
        self.order.sort_by(|&a, &b| {
            bounds[a].min.x.total_cmp(&bounds[b].min.x).then(a.cmp(&b))
        });

        for i in 0..self.order.len() {
            let a = self.order[i];
            if self.bounds[a].is_empty() {
                continue;
            }
            let reach = self.bounds[a].max.x;
            for j in i + 1..self.order.len() {
                let b = self.order[j];
                // Boxes are sorted by left edge, so once one starts past where `a`
                // ends, so does every box after it.
                if self.bounds[b].min.x > reach {
                    break;
                }
                if self.bounds[b].is_empty() {
                    continue;
                }
                if bodies.is_static(a) && bodies.is_static(b) {
                    continue;
                }
                self.tested += 1;
                if self.bounds[a].overlaps(&self.bounds[b]) {
                    self.pairs.push(BodyPair::new(a, b));
                }
            }
        }

        // Canonical order, so the solver visits contacts identically across runs.
        self.pairs.sort_unstable();
        &self.pairs
    }

    /// The pairs found by the last [`BroadPhase::find_pairs`].
    pub fn pairs(&self) -> &[BodyPair] {
        &self.pairs
    }

    /// Bounding boxes from the last sweep, indexed by slot.
    pub fn bounds(&self) -> &[Aabb] {
        &self.bounds
    }

    /// How many box-overlap tests the last sweep performed.
    ///
    /// Published so a model whose broadphase has degenerated to `n²/2` says so, rather
    /// than merely running slowly for a reason nobody can see (§19.3 asks benchmarks to
    /// report what they actually did).
    pub fn tests_performed(&self) -> usize {
        self.tested
    }

    /// Forget everything, keeping the allocations.
    pub fn clear(&mut self) {
        self.bounds.clear();
        self.order.clear();
        self.pairs.clear();
        self.tested = 0;
    }
}

/// The world transform of a body slot.
pub fn transform_of(bodies: &RigidBodyStore, slot: usize) -> Transform {
    Transform::new(
        vec2(bodies.pos_x()[slot], bodies.pos_y()[slot]),
        Rot { cos: bodies.rot_cos()[slot], sin: bodies.rot_sin()[slot] },
    )
}

/// The world-space bounding box of every body, for a viewer or a query.
pub fn world_bounds<'shapes, F>(bodies: &RigidBodyStore, shape_of: F) -> Aabb
where
    F: Fn(usize) -> Option<&'shapes Shape>,
{
    let mut total = Aabb::EMPTY;
    for slot in 0..bodies.len() {
        if let Some(shape) = shape_of(slot) {
            let box_ = shape.aabb(transform_of(bodies, slot));
            total.expand(box_.min);
            total.expand(box_.max);
        }
    }
    total
}

/// Look up the collider a body wears, given the registered set.
pub fn collider_of<'a>(
    bodies: &RigidBodyStore,
    colliders: &'a [Collider],
    slot: usize,
) -> Option<&'a Collider> {
    colliders.get(bodies.shape()[slot].index())
}

/// Convenience: the shape a body wears, for the closures above.
///
/// Borrowed rather than cloned. A `Shape` holding an eight-vertex polygon is a few
/// hundred bytes, and this is called once per body per step and once per body per pair
/// test — copying it there would be the most expensive thing the broadphase does.
pub fn shape_lookup<'a>(
    bodies: &'a RigidBodyStore,
    colliders: &'a [Collider],
) -> impl Fn(usize) -> Option<&'a Shape> + 'a {
    move |slot| colliders.get(bodies.shape()[slot].index()).map(|c| &c.shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec2;
    use lattice_ir::{BodySpec, ShapeId};

    fn scene(shapes: &[Shape], placements: &[(usize, [f64; 2], bool)]) -> (RigidBodyStore, Vec<Collider>) {
        let colliders: Vec<Collider> = shapes.iter().cloned().map(Collider::new).collect();
        let mut bodies = RigidBodyStore::with_capacity(placements.len().max(1));
        for &(shape, position, is_static) in placements {
            let id = ShapeId::from_index(shape as u32);
            let spec = if is_static {
                BodySpec::statik(position, id)
            } else {
                BodySpec::at(position, id)
            };
            bodies.spawn(spec).unwrap();
        }
        (bodies, colliders)
    }

    fn unit_boxes(placements: &[(usize, [f64; 2], bool)]) -> (RigidBodyStore, Vec<Collider>) {
        scene(&[Shape::rectangle(1.0, 1.0).unwrap()], placements)
    }

    #[test]
    fn overlapping_boxes_are_reported_and_distant_ones_are_not() {
        let (bodies, colliders) =
            unit_boxes(&[(0, [0.0, 0.0], false), (0, [1.5, 0.0], false), (0, [50.0, 0.0], false)]);
        let mut broad = BroadPhase::new();
        let pairs = broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders));

        assert_eq!(pairs, [BodyPair::new(0, 1)], "only the close pair");
    }

    /// The sweep must not stop early and miss a pair. A long body spans many others'
    /// left edges, which is exactly where a `break` in the wrong place hides.
    #[test]
    fn a_long_body_is_paired_with_everything_it_spans() {
        let ground = Shape::rectangle(20.0, 0.5).unwrap();
        let crate_ = Shape::rectangle(0.5, 0.5).unwrap();
        let (bodies, colliders) = scene(
            &[ground, crate_],
            &[
                (0, [0.0, 0.0], true),
                (1, [-10.0, 0.9], false),
                (1, [0.0, 0.9], false),
                (1, [10.0, 0.9], false),
            ],
        );
        let mut broad = BroadPhase::new();
        let pairs = broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders));

        assert!(pairs.contains(&BodyPair::new(0, 1)), "{pairs:?}");
        assert!(pairs.contains(&BodyPair::new(0, 2)), "{pairs:?}");
        assert!(pairs.contains(&BodyPair::new(0, 3)), "{pairs:?}");
        assert_eq!(pairs.len(), 3, "the crates are far apart from each other");
    }

    /// In a scene with a long ground, static-static pairs would be most of the output
    /// and none of them can ever need resolving.
    #[test]
    fn two_static_bodies_are_never_paired() {
        let (bodies, colliders) =
            unit_boxes(&[(0, [0.0, 0.0], true), (0, [1.0, 0.0], true), (0, [1.5, 0.0], false)]);
        let mut broad = BroadPhase::new();
        let pairs = broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders));

        assert!(!pairs.contains(&BodyPair::new(0, 1)), "two walls: {pairs:?}");
        assert!(pairs.contains(&BodyPair::new(0, 2)));
        assert!(pairs.contains(&BodyPair::new(1, 2)));
    }

    /// FR-011: the same scene must produce the same pairs in the same order, whatever
    /// order the bodies happen to sit in.
    #[test]
    fn the_pair_list_is_canonical_and_reproducible() {
        let placements: Vec<(usize, [f64; 2], bool)> =
            (0..12).map(|i| (0, [f64::from(i) * 0.5, 0.0], false)).collect();
        let (bodies, colliders) = unit_boxes(&placements);

        let mut first = BroadPhase::new();
        let a: Vec<BodyPair> = first.find_pairs(&bodies, shape_lookup(&bodies, &colliders)).to_vec();
        let mut second = BroadPhase::with_capacity(64);
        let b: Vec<BodyPair> = second.find_pairs(&bodies, shape_lookup(&bodies, &colliders)).to_vec();

        assert_eq!(a, b);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "sorted and free of duplicates: {a:?}");
        assert!(a.iter().all(|p| p.a < p.b), "canonical order");
    }

    /// Bodies that will touch next step are reported this step, so the solver sees the
    /// contact before the overlap rather than after it.
    #[test]
    fn the_margin_reports_a_pair_just_before_it_touches() {
        let gap = CONTACT_MARGIN;
        let (bodies, colliders) = unit_boxes(&[(0, [0.0, 0.0], false), (0, [2.0 + gap, 0.0], false)]);
        let mut broad = BroadPhase::new();
        assert_eq!(
            broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders)).len(),
            1,
            "within the margin"
        );

        let (bodies, colliders) =
            unit_boxes(&[(0, [0.0, 0.0], false), (0, [2.0 + 4.0 * gap, 0.0], false)]);
        assert!(broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders)).is_empty());
    }

    #[test]
    fn a_body_with_no_registered_collider_pairs_with_nothing() {
        let (bodies, colliders) = scene(
            &[Shape::rectangle(1.0, 1.0).unwrap()],
            &[(0, [0.0, 0.0], false), (7, [0.1, 0.0], false)],
        );
        let mut broad = BroadPhase::new();
        assert!(broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders)).is_empty());
    }

    #[test]
    fn an_empty_scene_finds_nothing_rather_than_panicking() {
        let bodies = RigidBodyStore::with_capacity(4);
        let colliders: Vec<Collider> = Vec::new();
        let mut broad = BroadPhase::new();
        assert!(broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders)).is_empty());
        assert_eq!(broad.tests_performed(), 0);
        assert!(world_bounds(&bodies, shape_lookup(&bodies, &colliders)).is_empty());
    }

    /// The count is published so a scene whose broadphase has degenerated says so.
    #[test]
    fn the_test_count_reflects_how_much_work_the_sweep_did() {
        // Ten boxes all on top of each other: every pair must be tested.
        let placements: Vec<(usize, [f64; 2], bool)> = (0..10).map(|_| (0, [0.0, 0.0], false)).collect();
        let (bodies, colliders) = unit_boxes(&placements);
        let mut broad = BroadPhase::new();
        broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders));
        assert_eq!(broad.tests_performed(), 45, "n(n-1)/2 for a fully overlapping pile");

        // Ten boxes in a line, none touching: the sweep does almost nothing.
        let placements: Vec<(usize, [f64; 2], bool)> =
            (0..10).map(|i| (0, [f64::from(i) * 10.0, 0.0], false)).collect();
        let (bodies, colliders) = unit_boxes(&placements);
        broad.find_pairs(&bodies, shape_lookup(&bodies, &colliders));
        assert_eq!(broad.tests_performed(), 0, "the sweep breaks immediately at every body");
    }

    #[test]
    fn world_bounds_covers_every_body() {
        let (bodies, colliders) =
            unit_boxes(&[(0, [-5.0, 0.0], false), (0, [5.0, 3.0], false)]);
        let bounds = world_bounds(&bodies, shape_lookup(&bodies, &colliders));
        assert_eq!(bounds.min, vec2(-6.0, -1.0));
        assert_eq!(bounds.max, vec2(6.0, 4.0));
    }

    #[test]
    fn a_rotating_body_keeps_a_correct_box() {
        let colliders = [Collider::new(Shape::rectangle(1.0, 0.1).unwrap())];
        let mut bodies = RigidBodyStore::with_capacity(1);
        let id = bodies
            .spawn(BodySpec::at([0.0, 0.0], ShapeId::from_index(0)).with_angle(core::f64::consts::FRAC_PI_2))
            .unwrap();
        let slot = bodies.slot_of(id).unwrap();
        let bounds = colliders[0].shape.aabb(transform_of(&bodies, slot));
        // Turned a quarter turn, the 2 × 0.2 bar is 0.2 wide and 2 tall.
        assert!((bounds.size().x - 0.2).abs() < 1e-9, "{bounds:?}");
        assert!((bounds.size().y - 2.0).abs() < 1e-9, "{bounds:?}");
    }

    #[test]
    fn collider_lookup_resolves_by_shape_id() {
        let (bodies, colliders) = scene(
            &[Shape::circle(1.0).unwrap(), Shape::rectangle(2.0, 2.0).unwrap()],
            &[(1, [0.0, 0.0], false)],
        );
        let collider = collider_of(&bodies, &colliders, 0).unwrap();
        assert!(matches!(collider.shape, Shape::Polygon(_)));
        assert_eq!(Vec2::from([0.0, 0.0]), transform_of(&bodies, 0).position);
    }
}
