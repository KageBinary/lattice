//! Exact contact generation for a pair of shapes.
//!
//! Circles get closed-form tests. Everything else goes through the **separating-axis
//! theorem** followed by **edge clipping**: SAT finds the axis of least overlap, which
//! is the contact normal, and clipping the incident edge against the reference edge's
//! side planes turns that normal into the one or two points a solver can push at.
//!
//! # Why two points and not one
//!
//! A box resting flat on the ground overlaps along a segment, not at a point. Reduce
//! that to one contact and the box is balanced on a pin: it rocks, and the solver
//! chases the rocking forever. Two points describe the segment exactly, and for a
//! convex pair in 2D there is never a need for a third.
//!
//! # Why the reference face is chosen with hysteresis
//!
//! When two faces are near-parallel their separations are nearly equal, and the
//! arithmetic that picks a winner flips between them on round-off. Each flip changes
//! the feature ids, which discards the warm-start impulses, which lets the stack sag
//! for a step, which changes the geometry — an oscillation driven entirely by a tie.
//! [`REFERENCE_BIAS`] makes A keep the reference face unless B is meaningfully better.

use lattice_ir::{RigidBodyStore, ShapeId};

use crate::broadphase::transform_of;
use crate::contact::{
    BodyPair, Contact, FeatureId, Manifold, ManifoldPoint, MAX_MANIFOLD_POINTS,
};
use crate::math::{vec2, Transform, Vec2};
use crate::shape::{combine_surfaces, Collider, ConvexPolygon, Shape, MAX_VERTICES};

/// Allowed overlap before the solver tries to push bodies apart, m.
///
/// Contacts are not resolved to exactly zero penetration. Driving overlap to zero
/// makes resting bodies jitter: the correction overshoots, they separate, the contact
/// is lost, gravity pulls them back, and the cycle repeats at the frame rate. A small
/// permitted overlap gives the solver a dead band to rest in. One millimetre is the
/// conventional value and is invisible at any sane rendering scale.
pub const LINEAR_SLOP: f64 = 1e-3;

/// How much better B's separation must be before it takes the reference face.
///
/// See the module documentation: this exists to break near-ties in a stable direction
/// rather than to change which face is geometrically correct.
pub const REFERENCE_BIAS: f64 = 0.1 * LINEAR_SLOP;

/// A convex shape reduced to what SAT needs, in body-local coordinates.
///
/// Fixed-size and `Copy`, so the narrowphase allocates nothing (NFR-001).
#[derive(Clone, Copy, Debug)]
struct Facets {
    vertices: [Vec2; MAX_VERTICES],
    normals: [Vec2; MAX_VERTICES],
    count: usize,
}

impl Facets {
    /// The polygonal view of a shape, or `None` for a circle.
    fn of(shape: &Shape) -> Option<Facets> {
        match shape {
            Shape::Circle { .. } => None,
            Shape::Polygon(polygon) => Some(Facets::from_polygon(polygon)),
            // A segment is a slab of zero thickness: two vertices, and two opposite
            // faces. Expressing it this way means every polygon routine below handles
            // segments without a special case.
            Shape::Segment { half_length } => {
                let mut facets =
                    Facets { vertices: [Vec2::ZERO; MAX_VERTICES], normals: [Vec2::ZERO; MAX_VERTICES], count: 2 };
                facets.vertices[0] = vec2(-half_length, 0.0);
                facets.vertices[1] = vec2(*half_length, 0.0);
                facets.normals[0] = vec2(0.0, -1.0);
                facets.normals[1] = vec2(0.0, 1.0);
                Some(facets)
            }
        }
    }

    fn from_polygon(polygon: &ConvexPolygon) -> Facets {
        let mut facets =
            Facets { vertices: [Vec2::ZERO; MAX_VERTICES], normals: [Vec2::ZERO; MAX_VERTICES], count: 0 };
        for (index, (&vertex, &normal)) in
            polygon.vertices().iter().zip(polygon.normals()).enumerate()
        {
            facets.vertices[index] = vertex;
            facets.normals[index] = normal;
            facets.count += 1;
        }
        facets
    }

    fn vertex(&self, index: usize) -> Vec2 {
        self.vertices[index % self.count]
    }

    /// The farthest vertex in `direction`, and its index.
    fn support(&self, direction: Vec2) -> (usize, Vec2) {
        let mut best = 0;
        let mut best_projection = self.vertices[0].dot(direction);
        for index in 1..self.count {
            let projection = self.vertices[index].dot(direction);
            if projection > best_projection {
                best = index;
                best_projection = projection;
            }
        }
        (best, self.vertices[best])
    }
}

/// The colliders a domain has registered, indexed by [`ShapeId`].
pub trait ColliderSet {
    /// Look up a collider. `None` for an id that was never registered.
    fn get(&self, id: ShapeId) -> Option<&Collider>;
}

impl ColliderSet for Vec<Collider> {
    fn get(&self, id: ShapeId) -> Option<&Collider> {
        self.as_slice().get(id.index())
    }
}

/// Compute the contact between two placed shapes, if they touch.
///
/// The returned normal points from `a` toward `b`.
///
/// Shapes that touch *exactly* produce a contact with zero penetration, not `None`.
/// That is the state a resting stack settles into, and a contact that vanished the
/// moment the overlap reached zero would drop the stack for one step, let gravity pull
/// it back, and repeat — visible as a permanent buzz. The solver is written to handle a
/// zero-penetration contact: it stops approach without pushing anything apart.
///
/// Returns `None` when the shapes are apart, and also for two segments: two
/// zero-thickness lines only intersect in configurations of measure zero, and a solver
/// asked to resolve that produces jitter rather than physics. Give one of them a
/// polygon if they need to collide.
pub fn collide(a: &Shape, ta: Transform, b: &Shape, tb: Transform) -> Option<Manifold> {
    match (a, b) {
        (Shape::Circle { radius: ra }, Shape::Circle { radius: rb }) => {
            circle_circle(ta.position, *ra, tb.position, *rb)
        }
        (Shape::Circle { radius }, other) => {
            let facets = Facets::of(other)?;
            // Computed as polygon-vs-circle, then reversed: the normal must point from
            // a (the circle) to b (the polygon).
            polygon_circle(&facets, tb, ta.position, *radius).map(reverse)
        }
        (other, Shape::Circle { radius }) => {
            let facets = Facets::of(other)?;
            polygon_circle(&facets, ta, tb.position, *radius)
        }
        (Shape::Segment { .. }, Shape::Segment { .. }) => None,
        (first, second) => {
            let (fa, fb) = (Facets::of(first)?, Facets::of(second)?);
            polygon_polygon(&fa, ta, &fb, tb)
        }
    }
}

/// Turn broadphase pairs into exact contacts, appending to `out`.
///
/// Pairs that turn out not to touch are dropped. `out` is cleared first and reused
/// across steps, so a settled scene performs no allocation (NFR-001).
pub fn generate_contacts(
    bodies: &RigidBodyStore,
    colliders: &[Collider],
    pairs: &[BodyPair],
    out: &mut Vec<Contact>,
) {
    out.clear();
    for &pair in pairs {
        let (Some(ca), Some(cb)) = (
            colliders.get(bodies.shape()[pair.a].index()),
            colliders.get(bodies.shape()[pair.b].index()),
        ) else {
            continue;
        };
        let manifold = collide(
            &ca.shape,
            transform_of(bodies, pair.a),
            &cb.shape,
            transform_of(bodies, pair.b),
        );
        let Some(manifold) = manifold else { continue };
        let (restitution, friction) = combine_surfaces(ca, cb);
        out.push(Contact { pair, manifold, restitution, friction });
    }
}

/// Flip a manifold so its normal points the other way.
fn reverse(mut manifold: Manifold) -> Manifold {
    manifold.normal = -manifold.normal;
    for point in &mut manifold.points {
        point.feature.flipped = !point.feature.flipped;
    }
    manifold
}

fn circle_circle(ca: Vec2, ra: f64, cb: Vec2, rb: f64) -> Option<Manifold> {
    let delta = cb - ca;
    let distance_squared = delta.length_squared();
    let touching = ra + rb;
    // `>` not `>=`: exactly touching is a contact with zero penetration. See `collide`.
    if distance_squared > touching * touching {
        return None;
    }
    // Exactly concentric circles have no separating direction. Reporting no contact
    // leaves them overlapped, which is visibly wrong — but inventing an axis puts a
    // fabricated direction into an impulse, and the configuration is unreachable from
    // any physical trajectory. Positional drift will separate them next step.
    let normal = delta.normalize()?;
    let distance = distance_squared.sqrt();
    let penetration = touching - distance;
    // The midpoint of the overlap: symmetric lever arms for both bodies.
    let position = ca + normal * (ra - 0.5 * penetration);
    Some(Manifold::single(normal, position, penetration, FeatureId::SINGLE))
}

/// Contact between a polygon (or segment) and a circle. Normal points polygon → circle.
fn polygon_circle(
    facets: &Facets,
    transform: Transform,
    centre: Vec2,
    radius: f64,
) -> Option<Manifold> {
    let local_centre = transform.unapply(centre);

    // Deepest face, measured from the circle's centre.
    let mut best_face = 0;
    let mut best_separation = f64::NEG_INFINITY;
    for index in 0..facets.count {
        let separation = facets.normals[index].dot(local_centre - facets.vertices[index]);
        if separation > best_separation {
            best_separation = separation;
            best_face = index;
        }
    }
    if best_separation > radius {
        return None;
    }

    let feature = FeatureId { reference: best_face as u8, incident: 0, flipped: false };
    let (v0, v1) = (facets.vertex(best_face), facets.vertex(best_face + 1));

    // The centre is inside: the deepest face gives the shortest way out.
    if best_separation < 0.0 {
        let normal = transform.rotation.apply(facets.normals[best_face]);
        let penetration = radius - best_separation;
        let position = centre - normal * (radius - 0.5 * penetration);
        return Some(Manifold::single(normal, position, penetration, feature));
    }

    // Outside: the closest feature of the face is either an endpoint or its interior.
    let edge = v1 - v0;
    let along = (local_centre - v0).dot(edge);
    let closest_local = if along <= 0.0 {
        v0
    } else if along >= edge.length_squared() {
        v1
    } else {
        v0 + edge * (along / edge.length_squared())
    };

    let closest = transform.apply(closest_local);
    let delta = centre - closest;
    let distance_squared = delta.length_squared();
    if distance_squared > radius * radius {
        return None;
    }
    // The centre sits exactly on the boundary; the face normal is the honest answer.
    let normal = delta.normalize().unwrap_or_else(|| transform.rotation.apply(facets.normals[best_face]));
    let penetration = radius - distance_squared.sqrt();
    let position = closest + normal * (0.5 * penetration);
    Some(Manifold::single(normal, position, penetration, feature))
}

/// The face of `reference` whose outward normal is least separated from `incident`,
/// and how far apart they are along it.
///
/// A positive separation means the shapes are apart along that axis, which by the
/// separating-axis theorem means they are apart, full stop.
fn max_separation(
    reference: &Facets,
    reference_tf: Transform,
    incident: &Facets,
    incident_tf: Transform,
) -> (usize, f64) {
    let mut best_face = 0;
    let mut best_separation = f64::NEG_INFINITY;
    for index in 0..reference.count {
        let normal_world = reference_tf.rotation.apply(reference.normals[index]);
        // Farthest point of the incident shape *against* the normal — the point most
        // deeply behind this face.
        let (_, support_local) = incident.support(incident_tf.rotation.unapply(-normal_world));
        let support = incident_tf.apply(support_local);
        let separation = normal_world.dot(support - reference_tf.apply(reference.vertices[index]));
        if separation > best_separation {
            best_separation = separation;
            best_face = index;
        }
    }
    (best_face, best_separation)
}

/// A vertex being clipped, carrying the feature it came from.
#[derive(Clone, Copy, Debug)]
struct ClipVertex {
    position: Vec2,
    incident: u8,
}

fn polygon_polygon(a: &Facets, ta: Transform, b: &Facets, tb: Transform) -> Option<Manifold> {
    let (face_a, separation_a) = max_separation(a, ta, b, tb);
    if separation_a > 0.0 {
        return None;
    }
    let (face_b, separation_b) = max_separation(b, tb, a, ta);
    if separation_b > 0.0 {
        return None;
    }

    // B takes the reference face only if it is meaningfully deeper — see the module
    // documentation on why a near-tie must not be allowed to flip.
    let flipped = separation_b > separation_a + REFERENCE_BIAS;
    let (reference, reference_tf, incident, incident_tf, reference_face) = if flipped {
        (b, tb, a, ta, face_b)
    } else {
        (a, ta, b, tb, face_a)
    };

    let normal = reference_tf.rotation.apply(reference.normals[reference_face]);

    // The incident face is the one most nearly facing the reference normal.
    let normal_in_incident = incident_tf.rotation.unapply(normal);
    let mut incident_face = 0;
    let mut best = f64::INFINITY;
    for index in 0..incident.count {
        let projection = incident.normals[index].dot(normal_in_incident);
        if projection < best {
            best = projection;
            incident_face = index;
        }
    }

    let mut clipped = [
        ClipVertex {
            position: incident_tf.apply(incident.vertex(incident_face)),
            incident: incident_face as u8,
        },
        ClipVertex {
            position: incident_tf.apply(incident.vertex(incident_face + 1)),
            incident: ((incident_face + 1) % incident.count) as u8,
        },
    ];

    // Clip against the reference face's two side planes. A segment has only two
    // vertices, so the reference "face" is the whole segment and the side planes are
    // its endpoints — which the same code handles.
    let v0 = reference_tf.apply(reference.vertex(reference_face));
    let v1 = reference_tf.apply(reference.vertex(reference_face + 1));
    let tangent = (v1 - v0).normalize()?;

    let mut count = clip(&mut clipped, -tangent, -tangent.dot(v0));
    if count < MAX_MANIFOLD_POINTS {
        return None;
    }
    count = clip(&mut clipped, tangent, tangent.dot(v1));
    if count < MAX_MANIFOLD_POINTS {
        return None;
    }

    // Keep only the clipped points that are actually behind the reference face.
    let mut points = [ManifoldPoint {
        position: Vec2::ZERO,
        penetration: 0.0,
        feature: FeatureId::SINGLE,
    }; MAX_MANIFOLD_POINTS];
    let mut kept = 0;
    for vertex in &clipped {
        let separation = normal.dot(vertex.position - v0);
        if separation > 0.0 {
            continue;
        }
        points[kept] = ManifoldPoint {
            // Midway between the reference face and the incident point.
            position: vertex.position - normal * (0.5 * separation),
            penetration: -separation,
            feature: FeatureId {
                reference: reference_face as u8,
                incident: vertex.incident,
                flipped,
            },
        };
        kept += 1;
    }
    if kept == 0 {
        return None;
    }
    if kept == 1 {
        points[1] = points[0];
    }

    // The normal must point from A to B, and it currently points away from whichever
    // shape supplied the reference face.
    let normal = if flipped { -normal } else { normal };
    Some(Manifold { normal, points, count: kept })
}

/// Clip a two-point segment against the half-plane `dot(normal, p) <= offset`.
///
/// Returns how many points survive. A point crossing the plane is moved onto it and
/// keeps the incident feature it came from, so warm starting still recognises it.
fn clip(segment: &mut [ClipVertex; 2], normal: Vec2, offset: f64) -> usize {
    let d0 = normal.dot(segment[0].position) - offset;
    let d1 = normal.dot(segment[1].position) - offset;

    let mut out = [segment[0]; 2];
    let mut count = 0;
    if d0 <= 0.0 {
        out[count] = segment[0];
        count += 1;
    }
    if d1 <= 0.0 {
        out[count] = segment[1];
        count += 1;
    }
    // One in, one out: add the crossing point, inheriting the *outgoing* vertex's
    // feature so the id changes when the geometry does.
    if d0 * d1 < 0.0 && count < 2 {
        let t = d0 / (d0 - d1);
        out[count] = ClipVertex {
            position: segment[0].position + (segment[1].position - segment[0].position) * t,
            incident: if d0 < 0.0 { segment[1].incident } else { segment[0].incident },
        };
        count += 1;
    }

    *segment = out;
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rot;

    const EPSILON: f64 = 1e-9;

    fn at(x: f64, y: f64) -> Transform {
        Transform::new(vec2(x, y), Rot::IDENTITY)
    }

    fn turned(x: f64, y: f64, radians: f64) -> Transform {
        Transform::new(vec2(x, y), Rot::from_angle(radians))
    }

    fn unit_box() -> Shape {
        Shape::rectangle(1.0, 1.0).unwrap()
    }

    #[test]
    fn two_circles_touch_along_the_line_between_their_centres() {
        let a = Shape::circle(1.0).unwrap();
        let b = Shape::circle(2.0).unwrap();

        assert!(collide(&a, at(0.0, 0.0), &b, at(5.0, 0.0)).is_none(), "3 m apart, 5 m away");

        let m = collide(&a, at(0.0, 0.0), &b, at(2.0, 0.0)).unwrap();
        assert!((m.normal - Vec2::X).length() < EPSILON, "normal points from a to b");
        assert!((m.max_penetration() - 1.0).abs() < EPSILON, "radii sum 3, centres 2 apart");
        assert_eq!(m.count, 1);
        // The contact sits midway through the overlap: a's surface is at x = 1,
        // b's is at x = 0, so the midpoint is x = 0.5.
        assert!((m.points()[0].position.x - 0.5).abs() < EPSILON, "{:?}", m.points()[0]);
    }

    #[test]
    fn a_circle_meets_a_box_face_squarely() {
        let boxy = unit_box();
        let circle = Shape::circle(0.5).unwrap();

        // Circle above the box, overlapping by 0.2.
        let m = collide(&boxy, at(0.0, 0.0), &circle, at(0.0, 1.3)).unwrap();
        assert!((m.normal - Vec2::Y).length() < EPSILON, "box → circle is +y, got {:?}", m.normal);
        assert!((m.max_penetration() - 0.2).abs() < EPSILON);

        // And the reverse order gives the reverse normal.
        let reversed = collide(&circle, at(0.0, 1.3), &boxy, at(0.0, 0.0)).unwrap();
        assert!((reversed.normal + Vec2::Y).length() < EPSILON, "{:?}", reversed.normal);
        assert!((reversed.max_penetration() - 0.2).abs() < EPSILON);
    }

    /// The vertex region is the case a face-only test misses: a circle approaching a
    /// corner diagonally is closest to the corner, not to either face.
    #[test]
    fn a_circle_at_a_corner_pushes_along_the_diagonal() {
        let boxy = unit_box();
        let circle = Shape::circle(1.0).unwrap();
        // Centre out past the (1, 1) corner along the diagonal.
        let offset = 1.0 + 0.5 / core::f64::consts::SQRT_2;
        let m = collide(&boxy, at(0.0, 0.0), &circle, at(offset, offset)).unwrap();

        let diagonal = vec2(1.0, 1.0).normalize().unwrap();
        assert!((m.normal - diagonal).length() < 1e-9, "{:?} should be the diagonal", m.normal);
        assert!((m.max_penetration() - 0.5).abs() < 1e-9, "{}", m.max_penetration());
    }

    #[test]
    fn a_circle_whose_centre_is_inside_a_box_takes_the_shortest_way_out() {
        let boxy = unit_box();
        let circle = Shape::circle(0.25).unwrap();
        // Just inside the top face.
        let m = collide(&boxy, at(0.0, 0.0), &circle, at(0.0, 0.9)).unwrap();
        assert!((m.normal - Vec2::Y).length() < EPSILON, "{:?}", m.normal);
        // Separation is −0.1, so penetration is radius + 0.1.
        assert!((m.max_penetration() - 0.35).abs() < EPSILON, "{}", m.max_penetration());
    }

    /// A box resting flat on another must produce *two* points. One point balances it
    /// on a pin and the solver chases the rocking forever.
    #[test]
    fn a_box_resting_flat_produces_two_contact_points() {
        let ground = Shape::rectangle(5.0, 0.5).unwrap();
        let crate_ = unit_box();
        // Box overlapping the ground's top surface by 0.1.
        let m = collide(&ground, at(0.0, 0.0), &crate_, at(0.0, 1.4)).unwrap();

        assert_eq!(m.count, 2, "a flat rest is a segment, not a point");
        assert!((m.normal - Vec2::Y).length() < EPSILON, "ground → crate is +y, got {:?}", m.normal);
        for point in m.points() {
            assert!((point.penetration - 0.1).abs() < EPSILON, "{point:?}");
        }
        // The two points are the crate's two bottom corners, at x = ±1.
        let mut xs: Vec<f64> = m.points().iter().map(|p| p.position.x).collect();
        xs.sort_by(f64::total_cmp);
        assert!((xs[0] + 1.0).abs() < EPSILON && (xs[1] - 1.0).abs() < EPSILON, "{xs:?}");
    }

    /// The same feature pair must keep the same id from step to step, or warm starting
    /// silently stops working and stacks sag.
    #[test]
    fn feature_ids_are_stable_while_the_geometry_is() {
        let ground = Shape::rectangle(5.0, 0.5).unwrap();
        let crate_ = unit_box();
        let first = collide(&ground, at(0.0, 0.0), &crate_, at(0.0, 1.4)).unwrap();
        // Settle a little deeper and drift sideways, still resting on the same face.
        let later = collide(&ground, at(0.0, 0.0), &crate_, at(0.05, 1.39)).unwrap();

        let ids = |m: &Manifold| {
            let mut v: Vec<FeatureId> = m.points().iter().map(|p| p.feature).collect();
            v.sort_by_key(|f| (f.reference, f.incident, f.flipped));
            v
        };
        assert_eq!(ids(&first), ids(&later), "the same corners on the same face");
    }

    #[test]
    fn separated_boxes_report_no_contact() {
        let a = unit_box();
        assert!(collide(&a, at(0.0, 0.0), &a, at(2.5, 0.0)).is_none());
        assert!(collide(&a, at(0.0, 0.0), &a, at(0.0, 2.5)).is_none());

        // The case SAT exists for: a diamond off the square's corner. Their bounding
        // boxes overlap heavily — a broadphase will hand this pair over — but along
        // the diagonal the square reaches 1.41 and the diamond reaches 1, against a
        // centre distance of 2.69, so they are 0.27 m apart.
        assert!(
            collide(&a, at(0.0, 0.0), &a, turned(1.9, 1.9, core::f64::consts::FRAC_PI_4)).is_none(),
            "overlapping bounding boxes, separated shapes"
        );
        // And closing that gap does produce a contact, so the test above is not
        // passing because the arrangement is trivially far apart.
        assert!(collide(&a, at(0.0, 0.0), &a, turned(1.6, 1.6, core::f64::consts::FRAC_PI_4)).is_some());
    }

    /// A contact that vanished the instant the overlap reached zero would drop a
    /// resting stack for one step every step. Touching exactly is a contact.
    #[test]
    fn exactly_touching_is_a_contact_with_no_penetration() {
        let boxy = unit_box();
        let touching = collide(&boxy, at(0.0, 0.0), &boxy, at(2.0, 0.0)).unwrap();
        assert!(touching.max_penetration().abs() < EPSILON, "{}", touching.max_penetration());
        assert!((touching.normal - Vec2::X).length() < EPSILON);

        let circle = Shape::circle(1.0).unwrap();
        let circles = collide(&circle, at(0.0, 0.0), &circle, at(2.0, 0.0)).unwrap();
        assert!(circles.max_penetration().abs() < EPSILON, "circles agree with polygons");

        // A hair further apart and there is nothing.
        assert!(collide(&circle, at(0.0, 0.0), &circle, at(2.0 + 1e-9, 0.0)).is_none());
    }

    /// A corner driven into a face is the other manifold shape: one point, not two.
    #[test]
    fn a_corner_into_a_face_produces_one_point() {
        let ground = Shape::rectangle(5.0, 0.5).unwrap();
        let diamond = unit_box();
        // Turned 45°, so a corner points down. Its lowest point is at −√2.
        let drop = 0.5 + core::f64::consts::SQRT_2 - 0.1;
        let m = collide(
            &ground,
            at(0.0, 0.0),
            &diamond,
            turned(0.0, drop, core::f64::consts::FRAC_PI_4),
        )
        .unwrap();

        assert_eq!(m.count, 1, "a corner touches at a point");
        assert!((m.normal - Vec2::Y).length() < 1e-9, "{:?}", m.normal);
        assert!((m.max_penetration() - 0.1).abs() < 1e-9, "{}", m.max_penetration());
        assert!(m.points()[0].position.x.abs() < 1e-9, "under the corner");
    }

    #[test]
    fn a_segment_is_a_wall_for_a_circle_and_a_box() {
        let ground = Shape::segment(5.0).unwrap();
        let circle = Shape::circle(1.0).unwrap();
        let m = collide(&ground, at(0.0, 0.0), &circle, at(0.0, 0.8)).unwrap();
        assert!((m.normal - Vec2::Y).length() < EPSILON, "{:?}", m.normal);
        assert!((m.max_penetration() - 0.2).abs() < EPSILON);

        // Past the end of the segment, the endpoint is the closest feature.
        let past = collide(&ground, at(0.0, 0.0), &circle, at(5.5, 0.5)).unwrap();
        assert!(past.normal.x > 0.0 && past.normal.y > 0.0, "away from the endpoint: {:?}", past.normal);

        let boxy = unit_box();
        let resting = collide(&ground, at(0.0, 0.0), &boxy, at(0.0, 0.95)).unwrap();
        assert_eq!(resting.count, 2, "a box rests on a segment along a segment");
        assert!((resting.normal - Vec2::Y).length() < EPSILON);
    }

    /// Two zero-thickness lines only intersect in configurations of measure zero, and
    /// a solver asked to resolve that jitters. Saying so beats pretending.
    #[test]
    fn two_segments_do_not_collide() {
        let a = Shape::segment(1.0).unwrap();
        let b = Shape::segment(1.0).unwrap();
        assert!(collide(&a, at(0.0, 0.0), &b, turned(0.0, 0.0, 1.0)).is_none());
    }

    /// Rotating both bodies together must not change the physics — the contact should
    /// come out identical in the rotated frame.
    #[test]
    fn contacts_are_invariant_under_a_shared_rotation() {
        let ground = Shape::rectangle(5.0, 0.5).unwrap();
        let crate_ = unit_box();
        let upright = collide(&ground, at(0.0, 0.0), &crate_, at(0.3, 1.4)).unwrap();

        let angle = 0.6;
        let rot = Rot::from_angle(angle);
        let rotated = collide(
            &ground,
            turned(0.0, 0.0, angle),
            &crate_,
            Transform::new(rot.apply(vec2(0.3, 1.4)), rot),
        )
        .unwrap();

        assert_eq!(rotated.count, upright.count);
        assert!(
            (rotated.normal - rot.apply(upright.normal)).length() < 1e-9,
            "{:?} vs {:?}",
            rotated.normal,
            rot.apply(upright.normal)
        );
        assert!((rotated.max_penetration() - upright.max_penetration()).abs() < 1e-9);
    }

    /// Swapping the argument order must reverse the normal and change nothing else.
    /// A sign error here makes contacts attract, which reads as an exploding solver.
    #[test]
    fn swapping_the_arguments_reverses_the_normal() {
        let ground = Shape::rectangle(5.0, 0.5).unwrap();
        let crate_ = unit_box();
        let forward = collide(&ground, at(0.0, 0.0), &crate_, at(0.0, 1.4)).unwrap();
        let backward = collide(&crate_, at(0.0, 1.4), &ground, at(0.0, 0.0)).unwrap();

        assert!((forward.normal + backward.normal).length() < 1e-9);
        assert_eq!(forward.count, backward.count);
        assert!((forward.max_penetration() - backward.max_penetration()).abs() < 1e-9);
    }

    #[test]
    fn concentric_circles_report_nothing_rather_than_a_fabricated_direction() {
        let a = Shape::circle(1.0).unwrap();
        assert!(collide(&a, at(0.0, 0.0), &a, at(0.0, 0.0)).is_none());
    }

    #[test]
    fn deep_overlap_still_produces_a_sane_normal() {
        let a = Shape::rectangle(1.0, 1.0).unwrap();
        // Almost coincident boxes, offset slightly along x.
        let m = collide(&a, at(0.0, 0.0), &a, at(0.05, 0.0)).unwrap();
        assert!(m.normal.is_finite());
        assert!((m.normal.length() - 1.0).abs() < EPSILON);
        assert!(m.max_penetration() > 0.0 && m.max_penetration().is_finite());
    }
}
