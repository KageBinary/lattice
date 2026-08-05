//! Contact manifolds: what the narrowphase produces and the solver consumes.

use crate::math::Vec2;

/// The most contact points two convex 2D shapes can share.
///
/// Two convex polygons overlap along a segment, so two points describe the contact
/// exactly. Any more would be redundant constraints fighting each other.
pub const MAX_MANIFOLD_POINTS: usize = 2;

/// One point of contact.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ManifoldPoint {
    /// Where the bodies touch, in world coordinates.
    ///
    /// Placed at the midpoint of the overlap rather than on either surface, so the
    /// lever arms the solver derives are symmetric between the two bodies.
    pub position: Vec2,
    /// How far the shapes overlap along the normal, m. Positive means overlapping.
    pub penetration: f64,
    /// Which pair of features produced this point.
    ///
    /// Warm starting reuses last step's accumulated impulse as this step's starting
    /// guess, which is what lets four iterations hold a stack that would otherwise
    /// need forty. That only works if this step's points can be matched to last
    /// step's, and position cannot do the matching — the points move. The feature
    /// pair can: as long as the same corner rests on the same face, the id is stable.
    pub feature: FeatureId,
}

/// Identifies the pair of geometric features that produced a contact point.
///
/// Two `u8` indices rather than a hash, so it is comparable, printable, and obviously
/// stable. Distinguishing which shape supplied the reference face matters: a corner of
/// A on a face of B is a different contact from a corner of B on a face of A, and
/// carrying last step's impulse across that switch produces a visible kick.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FeatureId {
    /// Index of the reference face.
    pub reference: u8,
    /// Index of the incident vertex.
    pub incident: u8,
    /// True when the shapes were swapped to make the reference face belong to A.
    pub flipped: bool,
}

impl FeatureId {
    /// A feature id for a contact with no distinguishing features — circles, which
    /// touch at exactly one point however they are arranged.
    pub const SINGLE: FeatureId = FeatureId { reference: 0, incident: 0, flipped: false };
}

/// The contact between two shapes.
///
/// The normal points **from A to B**: pushing the bodies apart means moving B along
/// `normal` and A against it. Every impulse sign in [`crate::solver`] depends on that
/// sentence, so it is stated once here rather than rediscovered per formula.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Manifold {
    /// Unit vector from A toward B.
    pub normal: Vec2,
    /// The contact points, of which the first `count` are valid.
    pub points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
    /// How many points are valid.
    pub count: usize,
}

impl Manifold {
    /// A manifold with a single point.
    pub fn single(normal: Vec2, position: Vec2, penetration: f64, feature: FeatureId) -> Manifold {
        let point = ManifoldPoint { position, penetration, feature };
        Manifold { normal, points: [point; MAX_MANIFOLD_POINTS], count: 1 }
    }

    /// The valid points.
    pub fn points(&self) -> &[ManifoldPoint] {
        &self.points[..self.count]
    }

    /// The deepest overlap in this manifold, m.
    pub fn max_penetration(&self) -> f64 {
        self.points().iter().map(|p| p.penetration).fold(0.0, f64::max)
    }
}

/// A candidate pair from the broadphase, or a resolved contact.
///
/// Slots rather than [`lattice_ir::BodyId`]s: the solver indexes the hot arrays
/// directly, and resolving a handle per contact per iteration would dominate its cost.
/// Pairs are only valid until a body is destroyed, which is why they are rebuilt every
/// step rather than cached across them.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BodyPair {
    /// The lower slot index.
    pub a: usize,
    /// The higher slot index.
    pub b: usize,
}

impl BodyPair {
    /// A pair in canonical order, so `(3, 1)` and `(1, 3)` are the same pair.
    ///
    /// Canonical ordering is what makes the pair usable as a key for warm starting,
    /// and what stops a pair being processed twice with opposite normals.
    pub fn new(first: usize, second: usize) -> BodyPair {
        if first <= second {
            BodyPair { a: first, b: second }
        } else {
            BodyPair { a: second, b: first }
        }
    }
}

/// A contact between two bodies, ready for the solver.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Contact {
    /// Which bodies, in canonical order. The manifold normal points from `a` to `b`.
    pub pair: BodyPair,
    /// Where and how deeply they touch.
    pub manifold: Manifold,
    /// Combined coefficient of restitution for this pair.
    pub restitution: f64,
    /// Combined Coulomb friction coefficient for this pair.
    pub friction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec2;

    #[test]
    fn pairs_are_canonical_so_a_contact_is_never_counted_twice() {
        assert_eq!(BodyPair::new(3, 1), BodyPair::new(1, 3));
        assert_eq!(BodyPair::new(1, 3), BodyPair { a: 1, b: 3 });
        assert_eq!(BodyPair::new(2, 2), BodyPair { a: 2, b: 2 });
    }

    /// Pairs are sorted so the solver visits contacts in a fixed order regardless of
    /// how the broadphase happened to enumerate them (FR-011).
    #[test]
    fn pairs_order_by_first_slot_then_second() {
        let mut pairs = [BodyPair::new(2, 5), BodyPair::new(1, 9), BodyPair::new(2, 3)];
        pairs.sort();
        assert_eq!(pairs, [BodyPair::new(1, 9), BodyPair::new(2, 3), BodyPair::new(2, 5)]);
    }

    #[test]
    fn a_single_point_manifold_exposes_exactly_one_point() {
        let m = Manifold::single(vec2(0.0, 1.0), vec2(3.0, 4.0), 0.25, FeatureId::SINGLE);
        assert_eq!(m.points().len(), 1);
        assert_eq!(m.max_penetration(), 0.25);
        assert_eq!(m.points()[0].position, vec2(3.0, 4.0));
    }

    /// A corner of A on a face of B is a different contact from a corner of B on a
    /// face of A, and carrying an accumulated impulse across that switch kicks.
    #[test]
    fn feature_ids_distinguish_which_shape_supplied_the_reference_face() {
        let a_on_b = FeatureId { reference: 1, incident: 2, flipped: false };
        let b_on_a = FeatureId { reference: 1, incident: 2, flipped: true };
        assert_ne!(a_on_b, b_on_a);
    }
}
