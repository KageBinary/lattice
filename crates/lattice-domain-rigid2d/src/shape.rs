//! Shapes and the mass properties derived from them.
//!
//! Spec §11.1's MVP row: *"circle, box, convex polygon, segment."*
//!
//! # Every shape is expressed about its own centre of mass
//!
//! A body's transform means "where the centre of mass is", not "where the artist put
//! the origin". Every impulse formula in [`crate::solver`] assumes it: `v + ω × r`
//! describes a point on a body only when `ω` is about the centre of mass, and the
//! inertia used to resist a torque has to be about the same point.
//!
//! Rather than carry a centre-of-mass offset through all of that, [`ConvexPolygon::new`]
//! recentres its vertices when it is built. A caller who wants an L-shape hanging off
//! to one side gets it — the *vertices* keep their relative arrangement; only the frame
//! moves — and every downstream formula gets to be the textbook one.

use crate::math::{vec2, Transform, Vec2};

/// Why a shape could not be built.
#[derive(Clone, PartialEq, Debug)]
pub enum ShapeError {
    /// Fewer than three vertices, so there is no area.
    TooFewVertices(usize),
    /// More vertices than the contact code will hold.
    TooManyVertices { given: usize, limit: usize },
    /// The vertices double back — the shape is not convex.
    ///
    /// Carries the offending vertex index so a diagnostic can point at it.
    NotConvex { at: usize },
    /// The vertices are collinear or coincident, so the area is zero.
    Degenerate,
    /// The vertices are wound clockwise.
    ///
    /// Reversing them silently would be safe — the shape is identical either way — but
    /// clockwise winding is usually a symptom rather than a typo: it is what a model
    /// authored for a y-down coordinate system produces, and that model's gravity
    /// vector is about to be wrong too. Better to say so here than to let the shape
    /// through and have the bug surface as physics.
    ClockwiseWinding,
    /// A non-finite coordinate, radius or length.
    NotFinite,
    /// A radius or half-length that is zero or negative.
    NotPositive(f64),
}

impl core::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShapeError::TooFewVertices(n) => {
                write!(f, "a polygon needs at least 3 vertices, got {n}")
            }
            ShapeError::TooManyVertices { given, limit } => {
                write!(f, "a polygon may have at most {limit} vertices, got {given}")
            }
            ShapeError::NotConvex { at } => {
                write!(f, "vertex {at} turns the wrong way; the polygon must be convex")
            }
            ShapeError::Degenerate => f.write_str("the vertices enclose no area"),
            ShapeError::ClockwiseWinding => f.write_str(
                "the vertices are wound clockwise; reverse them. If they came from a \
                 y-down coordinate system, check the sign of gravity too",
            ),
            ShapeError::NotFinite => f.write_str("a coordinate is not finite"),
            ShapeError::NotPositive(value) => write!(f, "expected a positive length, got {value}"),
        }
    }
}

impl core::error::Error for ShapeError {}

/// Vertex limit for a convex polygon.
///
/// Bounded so a manifold, a clipping buffer and a shape can all live in fixed-size
/// storage and the contact loop never allocates (NFR-001). Eight covers boxes,
/// triangles and the polygons a 2D model actually uses; a shape that needs more is
/// better represented as a compound body, which §11.1 lists under "later" anyway.
pub const MAX_VERTICES: usize = 8;

/// A convex polygon in body-local coordinates, wound counter-clockwise, centroid at
/// the origin.
#[derive(Clone, PartialEq, Debug)]
pub struct ConvexPolygon {
    vertices: heapless::Vec<Vec2, MAX_VERTICES>,
    normals: heapless::Vec<Vec2, MAX_VERTICES>,
}

/// A minimal fixed-capacity vector, so a polygon never allocates.
mod heapless {
    /// A `Vec` with its capacity in the type and its storage inline.
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(super) struct Vec<T, const N: usize> {
        items: [T; N],
        len: usize,
    }

    impl<T: Copy + Default, const N: usize> Default for Vec<T, N> {
        fn default() -> Self {
            Vec { items: [T::default(); N], len: 0 }
        }
    }

    impl<T: Copy + Default, const N: usize> Vec<T, N> {
        /// An empty vector.
        pub(super) fn new() -> Self {
            Self::default()
        }

        /// Append, or return the value back when there is no room.
        pub(super) fn push(&mut self, value: T) -> Result<(), T> {
            if self.len == N {
                return Err(value);
            }
            self.items[self.len] = value;
            self.len += 1;
            Ok(())
        }

        /// The occupied prefix. Everything else comes through `Deref` to a slice.
        pub(super) fn as_slice(&self) -> &[T] {
            &self.items[..self.len]
        }
    }

    impl<T: Copy + Default, const N: usize> core::ops::Deref for Vec<T, N> {
        type Target = [T];
        fn deref(&self) -> &[T] {
            self.as_slice()
        }
    }
}

impl ConvexPolygon {
    /// Build a polygon from counter-clockwise vertices.
    ///
    /// The result is recentred on its centroid, so `vertices()` will not generally
    /// equal what was passed in — see the module documentation for why.
    pub fn new(vertices: &[[f64; 2]]) -> Result<ConvexPolygon, ShapeError> {
        if vertices.len() < 3 {
            return Err(ShapeError::TooFewVertices(vertices.len()));
        }
        if vertices.len() > MAX_VERTICES {
            return Err(ShapeError::TooManyVertices {
                given: vertices.len(),
                limit: MAX_VERTICES,
            });
        }
        let points: Vec<Vec2> = vertices.iter().copied().map(Vec2::from).collect();
        if points.iter().any(|p| !p.is_finite()) {
            return Err(ShapeError::NotFinite);
        }

        let (area, centroid) = area_and_centroid(&points);
        // A tolerance relative to the shape's own extent, so a polygon measured in
        // millimetres is not rejected for being small.
        let scale = points.iter().map(|p| p.length()).fold(0.0, f64::max);
        let floor = 1e-12 * scale * scale.max(f64::MIN_POSITIVE);
        if area.abs() <= floor {
            return Err(ShapeError::Degenerate);
        }
        if area < 0.0 {
            return Err(ShapeError::ClockwiseWinding);
        }

        // Convexity, checked on the recentred points so the message indexes the input.
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            let c = points[(i + 2) % points.len()];
            if (b - a).cross(c - b) <= 0.0 {
                return Err(ShapeError::NotConvex { at: (i + 1) % points.len() });
            }
        }

        let mut recentred = heapless::Vec::new();
        for point in &points {
            recentred.push(*point - centroid).expect("length checked above");
        }

        let mut normals = heapless::Vec::new();
        for i in 0..recentred.len() {
            let edge = recentred[(i + 1) % recentred.len()] - recentred[i];
            // Outward for counter-clockwise winding.
            let normal = vec2(edge.y, -edge.x).normalize().ok_or(ShapeError::Degenerate)?;
            normals.push(normal).expect("same length as the vertices");
        }

        Ok(ConvexPolygon { vertices: recentred, normals })
    }

    /// An axis-aligned box with the given half-extents, centred on the origin.
    pub fn rectangle(half_width: f64, half_height: f64) -> Result<ConvexPolygon, ShapeError> {
        for extent in [half_width, half_height] {
            if !extent.is_finite() {
                return Err(ShapeError::NotFinite);
            }
            if extent <= 0.0 {
                return Err(ShapeError::NotPositive(extent));
            }
        }
        ConvexPolygon::new(&[
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ])
    }

    /// A regular polygon with `sides` vertices on a circle of radius `radius`.
    pub fn regular(sides: usize, radius: f64) -> Result<ConvexPolygon, ShapeError> {
        if !radius.is_finite() {
            return Err(ShapeError::NotFinite);
        }
        if radius <= 0.0 {
            return Err(ShapeError::NotPositive(radius));
        }
        if sides < 3 {
            return Err(ShapeError::TooFewVertices(sides));
        }
        if sides > MAX_VERTICES {
            return Err(ShapeError::TooManyVertices { given: sides, limit: MAX_VERTICES });
        }
        let points: Vec<[f64; 2]> = (0..sides)
            .map(|i| {
                let angle = core::f64::consts::TAU * i as f64 / sides as f64;
                [radius * angle.cos(), radius * angle.sin()]
            })
            .collect();
        ConvexPolygon::new(&points)
    }

    /// The vertices, counter-clockwise, relative to the centre of mass.
    pub fn vertices(&self) -> &[Vec2] {
        &self.vertices
    }

    /// Outward unit normal of edge `i`, the edge from vertex `i` to vertex `i + 1`.
    pub fn normals(&self) -> &[Vec2] {
        &self.normals
    }

    /// The farthest vertex in `direction` — the support function SAT and clipping use.
    pub fn support(&self, direction: Vec2) -> Vec2 {
        let mut best = self.vertices[0];
        let mut best_projection = best.dot(direction);
        for &vertex in &self.vertices[1..] {
            let projection = vertex.dot(direction);
            if projection > best_projection {
                best = vertex;
                best_projection = projection;
            }
        }
        best
    }

    /// Index of the edge whose outward normal points most nearly along `direction`.
    pub fn best_edge(&self, direction: Vec2) -> usize {
        let mut best = 0;
        let mut best_projection = f64::NEG_INFINITY;
        for (index, &normal) in self.normals.iter().enumerate() {
            let projection = normal.dot(direction);
            if projection > best_projection {
                best = index;
                best_projection = projection;
            }
        }
        best
    }

    /// Distance from the centroid to the farthest vertex.
    pub fn circumradius(&self) -> f64 {
        self.vertices.iter().map(|v| v.length()).fold(0.0, f64::max)
    }
}

/// Signed area and centroid of a counter-clockwise polygon.
fn area_and_centroid(points: &[Vec2]) -> (f64, Vec2) {
    let mut area2 = 0.0;
    let mut moment = Vec2::ZERO;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let cross = a.cross(b);
        area2 += cross;
        moment += (a + b) * cross;
    }
    let area = 0.5 * area2;
    if area2 == 0.0 {
        return (area, Vec2::ZERO);
    }
    (area, moment / (3.0 * area2))
}

/// A collision shape in body-local coordinates.
///
/// The polygon variant is far larger than the other two — it carries eight vertices and
/// eight normals inline so the narrowphase never allocates. Boxing it to even the
/// variants out would trade that for a pointer chase on every SAT axis, which is the
/// hottest loop in the module. Nothing clones a `Shape` on a hot path; they are
/// registered once and borrowed thereafter.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, PartialEq, Debug)]
pub enum Shape {
    /// A disc centred on the centre of mass.
    Circle {
        /// Radius, m.
        radius: f64,
    },
    /// A convex polygon, already recentred on its centroid.
    Polygon(ConvexPolygon),
    /// A segment along the body-local x axis, from `(-half_length, 0)` to
    /// `(half_length, 0)`.
    ///
    /// Encloses no area, so [`Shape::mass_properties`] gives it no mass at any density
    /// — see that method. Segments are for walls and ground planes, which is what
    /// §11.1 wants them for.
    Segment {
        /// Half the segment's length, m.
        half_length: f64,
    },
}

impl Shape {
    /// A circle of the given radius.
    pub fn circle(radius: f64) -> Result<Shape, ShapeError> {
        if !radius.is_finite() {
            return Err(ShapeError::NotFinite);
        }
        if radius <= 0.0 {
            return Err(ShapeError::NotPositive(radius));
        }
        Ok(Shape::Circle { radius })
    }

    /// An axis-aligned box with the given half-extents.
    pub fn rectangle(half_width: f64, half_height: f64) -> Result<Shape, ShapeError> {
        ConvexPolygon::rectangle(half_width, half_height).map(Shape::Polygon)
    }

    /// A segment along the body-local x axis.
    pub fn segment(half_length: f64) -> Result<Shape, ShapeError> {
        if !half_length.is_finite() {
            return Err(ShapeError::NotFinite);
        }
        if half_length <= 0.0 {
            return Err(ShapeError::NotPositive(half_length));
        }
        Ok(Shape::Segment { half_length })
    }

    /// Enclosed area, m². Zero for a segment.
    pub fn area(&self) -> f64 {
        match self {
            Shape::Circle { radius } => core::f64::consts::PI * radius * radius,
            Shape::Polygon(polygon) => area_and_centroid(polygon.vertices()).0,
            Shape::Segment { .. } => 0.0,
        }
    }

    /// Distance from the centre of mass to the farthest point of the shape.
    ///
    /// The broadphase uses it for a rotation-independent bound, so a body's cell
    /// assignment does not have to be redone from scratch as it spins.
    pub fn circumradius(&self) -> f64 {
        match self {
            Shape::Circle { radius } => *radius,
            Shape::Polygon(polygon) => polygon.circumradius(),
            Shape::Segment { half_length } => *half_length,
        }
    }

    /// Mass and rotational inertia for a uniform body of areal density `density`.
    ///
    /// `density` is a mass per unit area, kg/m² — this is a 2D world, and pretending
    /// it has a thickness would make every printed mass wrong by a factor nobody
    /// declared. §14.1 puts that kind of conversion in a port, not in a solver.
    ///
    /// A [`Shape::Segment`] has no area and therefore no mass. That is not a bug to
    /// paper over: a massless dynamic body is a division by zero waiting to happen, so
    /// a segment must either be static or be given an explicit mass with
    /// [`MassProperties::rod`].
    pub fn mass_properties(&self, density: f64) -> MassProperties {
        match self {
            Shape::Circle { radius } => {
                let mass = density * self.area();
                // I = ½ m r² for a uniform disc about its centre.
                MassProperties { mass, inertia: 0.5 * mass * radius * radius }
            }
            Shape::Polygon(polygon) => {
                let points = polygon.vertices();
                let mut area2 = 0.0;
                let mut second_moment = 0.0;
                for i in 0..points.len() {
                    let a = points[i];
                    let b = points[(i + 1) % points.len()];
                    let cross = a.cross(b);
                    area2 += cross;
                    // The polar second moment of the triangle (origin, a, b).
                    second_moment += cross * (a.dot(a) + a.dot(b) + b.dot(b));
                }
                let mass = density * 0.5 * area2;
                // The polygon is already centred, so the moment about the origin is
                // the moment about the centre of mass and no parallel-axis shift is
                // needed here.
                MassProperties { mass, inertia: density * second_moment / 12.0 }
            }
            Shape::Segment { .. } => MassProperties { mass: 0.0, inertia: 0.0 },
        }
    }

    /// Whether a body-local point is inside this shape.
    ///
    /// `slack` widens the shape outward, in metres. A click needs it: a segment has no
    /// interior at all, and a small circle is a hard target with a mouse. Zero gives the
    /// exact geometric test.
    ///
    /// Not the same question the narrowphase asks. That one is *are these two shapes
    /// overlapping and along which axis*, which is a search for a separating direction.
    /// This is *is this one point inside*, which is a sign test per edge and has no
    /// direction to report.
    pub fn contains(&self, local: Vec2, slack: f64) -> bool {
        if !local.is_finite() {
            return false;
        }
        match self {
            Shape::Circle { radius } => local.length_squared() <= (radius + slack).powi(2),
            Shape::Polygon(polygon) => {
                // Inside a counter-clockwise convex polygon means left of every edge.
                // With slack, "left of" becomes "no further right than `slack`", which
                // grows the shape by that much along each edge normal.
                let vertices = polygon.vertices();
                (0..vertices.len()).all(|i| {
                    let (from, to) = (vertices[i], vertices[(i + 1) % vertices.len()]);
                    (to - from).cross(local - from) >= -slack * (to - from).length()
                })
            }
            Shape::Segment { half_length } => {
                let along = local.x.clamp(-half_length, *half_length);
                (local - vec2(along, 0.0)).length() <= slack
            }
        }
    }

    /// The world-space bounding box of this shape under `transform`.
    pub fn aabb(&self, transform: Transform) -> Aabb {
        match self {
            Shape::Circle { radius } => Aabb {
                min: transform.position - vec2(*radius, *radius),
                max: transform.position + vec2(*radius, *radius),
            },
            Shape::Polygon(polygon) => {
                let mut bounds = Aabb::EMPTY;
                for &vertex in polygon.vertices() {
                    bounds.expand(transform.apply(vertex));
                }
                bounds
            }
            Shape::Segment { half_length } => {
                let mut bounds = Aabb::EMPTY;
                bounds.expand(transform.apply(vec2(-half_length, 0.0)));
                bounds.expand(transform.apply(vec2(*half_length, 0.0)));
                bounds
            }
        }
    }
}

/// The mass and rotational inertia a solver needs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MassProperties {
    /// Mass, kg (2D: really kg per unit depth, see [`Shape::mass_properties`]).
    pub mass: f64,
    /// Rotational inertia about the centre of mass, kg·m².
    pub inertia: f64,
}

impl MassProperties {
    /// A body with no mass and no inertia — immovable.
    pub const STATIC: MassProperties = MassProperties { mass: 0.0, inertia: 0.0 };

    /// A thin rod of `mass` and total length `2 · half_length`, about its centre.
    ///
    /// The escape hatch a [`Shape::Segment`] needs to be dynamic: `I = mL²/12`.
    pub fn rod(mass: f64, half_length: f64) -> MassProperties {
        let length = 2.0 * half_length;
        MassProperties { mass, inertia: mass * length * length / 12.0 }
    }

    /// Shift the inertia to an axis `offset` away from the centre of mass.
    ///
    /// The parallel-axis theorem, `I' = I + m d²`. Used when assembling a compound
    /// body, and by the tests to check the polygon formula against a shape whose
    /// inertia is known by hand.
    pub fn about_offset_axis(self, offset: f64) -> MassProperties {
        MassProperties {
            mass: self.mass,
            inertia: self.inertia + self.mass * offset * offset,
        }
    }

    /// Scale to a total mass, keeping the inertia consistent.
    ///
    /// Lets a model say "this box weighs 2 kg" without computing a density.
    pub fn with_total_mass(self, mass: f64) -> MassProperties {
        if self.mass <= 0.0 || !self.mass.is_finite() {
            return MassProperties { mass, inertia: 0.0 };
        }
        let ratio = mass / self.mass;
        MassProperties { mass, inertia: self.inertia * ratio }
    }
}

/// An axis-aligned bounding box.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Aabb {
    /// Lower corner.
    pub min: Vec2,
    /// Upper corner.
    pub max: Vec2,
}

impl Aabb {
    /// An inverted box that contains nothing, so the first `expand` sets both corners.
    pub const EMPTY: Aabb =
        Aabb { min: vec2(f64::INFINITY, f64::INFINITY), max: vec2(f64::NEG_INFINITY, f64::NEG_INFINITY) };

    /// Grow to include `point`.
    pub fn expand(&mut self, point: Vec2) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Grow outward by `margin` on every side.
    pub fn grown(self, margin: f64) -> Aabb {
        Aabb { min: self.min - vec2(margin, margin), max: self.max + vec2(margin, margin) }
    }

    /// True when the boxes share any point.
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Width and height.
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// True when nothing has been added.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }
}

/// A shape placed on a body: the geometry plus the surface it presents.
#[derive(Clone, PartialEq, Debug)]
pub struct Collider {
    /// The geometry, in body-local coordinates.
    pub shape: Shape,
    /// Coefficient of restitution, 0 (perfectly inelastic) to 1 (perfectly elastic).
    pub restitution: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

impl Collider {
    /// A collider with no bounce and moderate friction.
    pub fn new(shape: Shape) -> Collider {
        Collider { shape, restitution: 0.0, friction: 0.3 }
    }

    /// Set the coefficient of restitution, clamped to `[0, 1]`.
    ///
    /// Above 1 a collision would *create* kinetic energy, which is not a material
    /// property but a bug that takes a while to find. Clamping is the honest choice
    /// only because the alternative — refusing — would reject a model over a typo in a
    /// number the solver can still make sense of.
    pub fn with_restitution(mut self, restitution: f64) -> Collider {
        self.restitution = restitution.clamp(0.0, 1.0);
        self
    }

    /// Set the Coulomb friction coefficient. Negative values clamp to zero.
    pub fn with_friction(mut self, friction: f64) -> Collider {
        self.friction = friction.max(0.0);
        self
    }
}

/// How two colliders' surface properties combine at a contact.
///
/// Restitution takes the maximum and friction the geometric mean. Neither is derivable
/// from first principles — the pair coefficient is a property of the *pair* of
/// materials, and no engine stores an N×N table for it. These are the conventional
/// choices, and they are stated here rather than buried in the solver because they are
/// a modelling assumption, not an implementation detail.
pub fn combine_surfaces(a: &Collider, b: &Collider) -> (f64, f64) {
    let restitution = a.restitution.max(b.restitution);
    let friction = (a.friction * b.friction).sqrt();
    (restitution, friction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rot;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn a_circle_has_the_textbook_area_and_inertia() {
        let circle = Shape::circle(2.0).unwrap();
        assert!((circle.area() - core::f64::consts::PI * 4.0).abs() < EPSILON);

        let m = circle.mass_properties(3.0);
        assert!((m.mass - 3.0 * core::f64::consts::PI * 4.0).abs() < EPSILON);
        // I = ½ m r².
        assert!((m.inertia - 0.5 * m.mass * 4.0).abs() < EPSILON, "{m:?}");
    }

    #[test]
    fn a_box_has_the_textbook_area_and_inertia() {
        let (a, b) = (1.5, 0.5);
        let shape = Shape::rectangle(a, b).unwrap();
        assert!((shape.area() - 4.0 * a * b).abs() < EPSILON);

        let m = shape.mass_properties(2.0);
        assert!((m.mass - 2.0 * 4.0 * a * b).abs() < EPSILON);
        // I = m (w² + h²)/12 = m((2a)² + (2b)²)/12 = m(a² + b²)/3.
        let expected = m.mass * (a * a + b * b) / 3.0;
        assert!((m.inertia - expected).abs() < EPSILON, "{} vs {expected}", m.inertia);
    }

    /// The polygon inertia formula is the one place a sign error hides indefinitely:
    /// it would not disturb momentum, and momentum is what every other test checks.
    /// So check it against a shape whose answer is known independently.
    #[test]
    fn the_polygon_inertia_formula_agrees_with_the_parallel_axis_theorem() {
        // A box, split into two half-boxes side by side. Each half's inertia about the
        // whole box's centre is its own inertia plus m·d², and the two must sum to the
        // whole. If the polygon formula were wrong, the halves and the whole would
        // disagree.
        let whole = Shape::rectangle(2.0, 1.0).unwrap().mass_properties(1.0);
        let half = Shape::rectangle(1.0, 1.0).unwrap().mass_properties(1.0);
        let shifted = half.about_offset_axis(1.0);

        assert!((2.0 * shifted.mass - whole.mass).abs() < EPSILON, "the halves must weigh the whole");
        assert!(
            (2.0 * shifted.inertia - whole.inertia).abs() < EPSILON,
            "two half-boxes about the centre: {} vs the whole {}",
            2.0 * shifted.inertia,
            whole.inertia
        );
    }

    /// A regular polygon approaches the disc it is inscribed in.
    #[test]
    fn a_regular_polygon_converges_on_its_circumscribed_circle() {
        let circle = Shape::circle(1.0).unwrap().mass_properties(1.0);
        let mut previous_error = f64::INFINITY;
        for sides in [3, 4, 5, 6, 8] {
            let polygon =
                Shape::Polygon(ConvexPolygon::regular(sides, 1.0).unwrap()).mass_properties(1.0);
            assert!(polygon.mass < circle.mass, "an inscribed polygon is lighter");
            let error = (polygon.mass - circle.mass).abs();
            assert!(error < previous_error, "{sides} sides should be closer than {previous_error}");
            previous_error = error;
        }
        // An octagon inscribed in a unit circle has area 4·sin(45°) = 2.83 against
        // π = 3.14 — 10% short. Convergence is O(1/n²), so this is the expected gap,
        // not a slack tolerance.
        assert!(previous_error / circle.mass < 0.11, "an octagon should be within 11%");
    }

    #[test]
    fn a_polygon_is_recentred_on_its_centroid() {
        // A box built far from the origin.
        let polygon = ConvexPolygon::new(&[[10.0, 5.0], [12.0, 5.0], [12.0, 6.0], [10.0, 6.0]])
            .unwrap();
        let (_, centroid) = area_and_centroid(polygon.vertices());
        assert!(centroid.length() < EPSILON, "centroid should be the origin, got {centroid:?}");
        // And the shape is unchanged: still a 2 × 1 box.
        let m = Shape::Polygon(polygon).mass_properties(1.0);
        assert!((m.mass - 2.0).abs() < EPSILON);
    }

    #[test]
    fn bad_polygons_are_rejected_with_a_reason() {
        assert_eq!(ConvexPolygon::new(&[[0.0, 0.0], [1.0, 0.0]]), Err(ShapeError::TooFewVertices(2)));
        assert_eq!(
            ConvexPolygon::new(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            Err(ShapeError::Degenerate),
            "collinear points enclose nothing"
        );
        // A clockwise square is a valid convex shape wound the wrong way, and says so
        // rather than being reported as a dent or as enclosing nothing.
        assert_eq!(
            ConvexPolygon::new(&[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
            Err(ShapeError::ClockwiseWinding)
        );
        // A genuine dent.
        assert!(matches!(
            ConvexPolygon::new(&[[0.0, 0.0], [2.0, 0.0], [1.0, 0.5], [2.0, 2.0], [0.0, 2.0]]),
            Err(ShapeError::NotConvex { .. })
        ));
        assert_eq!(ConvexPolygon::new(&[[0.0, 0.0], [1.0, 0.0], [f64::NAN, 1.0]]), Err(ShapeError::NotFinite));
        assert_eq!(Shape::circle(-1.0), Err(ShapeError::NotPositive(-1.0)));
        assert_eq!(Shape::circle(f64::INFINITY), Err(ShapeError::NotFinite));
    }

    #[test]
    fn a_polygon_beyond_the_vertex_limit_is_refused_rather_than_truncated() {
        let many: Vec<[f64; 2]> = (0..MAX_VERTICES + 1)
            .map(|i| {
                let a = core::f64::consts::TAU * i as f64 / (MAX_VERTICES + 1) as f64;
                [a.cos(), a.sin()]
            })
            .collect();
        assert_eq!(
            ConvexPolygon::new(&many),
            Err(ShapeError::TooManyVertices { given: MAX_VERTICES + 1, limit: MAX_VERTICES })
        );
    }

    /// A segment encloses nothing, so density gives it nothing. Silently handing back
    /// a small mass would be worse than zero: it would look like it worked.
    #[test]
    fn a_segment_has_no_mass_from_density_alone() {
        let segment = Shape::segment(3.0).unwrap();
        assert_eq!(segment.area(), 0.0);
        assert_eq!(segment.mass_properties(1000.0), MassProperties::STATIC);

        // The escape hatch: an explicit rod mass, I = mL²/12.
        let rod = MassProperties::rod(6.0, 3.0);
        assert!((rod.inertia - 6.0 * 36.0 / 12.0).abs() < EPSILON, "{rod:?}");
    }

    #[test]
    fn total_mass_scaling_keeps_the_inertia_consistent() {
        let dense = Shape::rectangle(1.0, 2.0).unwrap().mass_properties(7.0);
        let light = dense.with_total_mass(1.0);
        assert!((light.mass - 1.0).abs() < EPSILON);
        // Same geometry, so I/m is unchanged.
        assert!((light.inertia / light.mass - dense.inertia / dense.mass).abs() < EPSILON);
    }

    #[test]
    fn support_and_best_edge_pick_the_extreme_in_a_direction() {
        let square = ConvexPolygon::rectangle(1.0, 1.0).unwrap();
        assert!((square.support(Vec2::X).x - 1.0).abs() < EPSILON);
        assert!((square.support(-Vec2::Y).y + 1.0).abs() < EPSILON);

        let edge = square.best_edge(Vec2::Y);
        assert!((square.normals()[edge].y - 1.0).abs() < EPSILON, "the top edge faces up");
    }

    /// The test a click runs. Interior, exterior, and the boundary itself.
    #[test]
    fn a_point_is_inside_a_shape_or_it_is_not() {
        let circle = Shape::circle(2.0).unwrap();
        assert!(circle.contains(Vec2::ZERO, 0.0), "the centre is inside");
        assert!(circle.contains(vec2(1.999, 0.0), 0.0));
        assert!(!circle.contains(vec2(2.001, 0.0), 0.0));
        // Exactly on the rim counts as inside, so a click on the outline picks it up.
        assert!(circle.contains(vec2(2.0, 0.0), 0.0));

        let boxy = Shape::rectangle(1.0, 0.5).unwrap();
        assert!(boxy.contains(Vec2::ZERO, 0.0));
        assert!(boxy.contains(vec2(0.99, 0.49), 0.0), "just inside a corner");
        assert!(!boxy.contains(vec2(1.01, 0.0), 0.0));
        assert!(!boxy.contains(vec2(0.0, 0.51), 0.0));
        // Outside on the diagonal but inside both axis extents of the bounding box.
        assert!(!boxy.contains(vec2(1.2, 0.6), 0.0));
    }

    /// A segment has no interior, so an exact test can never pick one. Slack is what
    /// makes a wall clickable — and what makes a small circle a fair target with a
    /// mouse.
    #[test]
    fn slack_widens_a_shape_so_a_click_can_reach_it() {
        let segment = Shape::segment(3.0).unwrap();
        assert!(!segment.contains(vec2(0.0, 0.01), 0.0), "no interior, exactly");
        assert!(segment.contains(vec2(0.0, 0.01), 0.05), "but reachable with slack");
        assert!(segment.contains(vec2(3.0, 0.0), 0.05), "including its endpoint");
        assert!(!segment.contains(vec2(3.2, 0.0), 0.05), "and not beyond it");

        let small = Shape::circle(0.05).unwrap();
        assert!(!small.contains(vec2(0.1, 0.0), 0.0));
        assert!(small.contains(vec2(0.1, 0.0), 0.1));

        let boxy = Shape::rectangle(1.0, 1.0).unwrap();
        assert!(!boxy.contains(vec2(1.05, 0.0), 0.0));
        assert!(boxy.contains(vec2(1.05, 0.0), 0.1), "slack grows it along the edge normal");
    }

    #[test]
    fn a_non_finite_point_is_never_inside() {
        let boxy = Shape::rectangle(1.0, 1.0).unwrap();
        assert!(!boxy.contains(vec2(f64::NAN, 0.0), 1e6));
        assert!(!Shape::circle(1.0).unwrap().contains(vec2(0.0, f64::INFINITY), 0.0));
    }

    #[test]
    fn an_aabb_follows_a_rotating_box() {
        let shape = Shape::rectangle(1.0, 1.0).unwrap();
        let upright = shape.aabb(Transform::default());
        assert!((upright.size().x - 2.0).abs() < EPSILON);

        // Turned 45°, the same square is √2 times as wide.
        let tilted = shape.aabb(Transform::new(Vec2::ZERO, Rot::from_angle(core::f64::consts::FRAC_PI_4)));
        assert!((tilted.size().x - 2.0 * core::f64::consts::SQRT_2).abs() < 1e-9, "{tilted:?}");

        // And the circumradius bounds both, which is what the broadphase relies on.
        let bound = shape.circumradius();
        assert!(bound >= tilted.size().x / 2.0 - EPSILON);
    }

    #[test]
    fn aabb_overlap_is_inclusive_and_symmetric() {
        let a = Aabb { min: vec2(0.0, 0.0), max: vec2(1.0, 1.0) };
        let touching = Aabb { min: vec2(1.0, 0.0), max: vec2(2.0, 1.0) };
        let apart = Aabb { min: vec2(1.5, 0.0), max: vec2(2.0, 1.0) };
        assert!(a.overlaps(&touching) && touching.overlaps(&a));
        assert!(!a.overlaps(&apart) && !apart.overlaps(&a));
        assert!(Aabb::EMPTY.is_empty());
    }

    #[test]
    fn surface_properties_combine_by_a_stated_rule() {
        let bouncy = Collider::new(Shape::circle(1.0).unwrap()).with_restitution(0.9).with_friction(0.4);
        let dead = Collider::new(Shape::circle(1.0).unwrap()).with_restitution(0.1).with_friction(0.9);
        let (restitution, friction) = combine_surfaces(&bouncy, &dead);
        assert!((restitution - 0.9).abs() < EPSILON, "the bouncier surface wins");
        assert!((friction - 0.6).abs() < EPSILON, "geometric mean of 0.4 and 0.9");
    }

    /// A restitution above 1 creates energy out of nothing, which looks like an
    /// unstable solver rather than the typo it is.
    #[test]
    fn restitution_cannot_exceed_one() {
        let collider = Collider::new(Shape::circle(1.0).unwrap()).with_restitution(1.5);
        assert_eq!(collider.restitution, 1.0);
        assert_eq!(Collider::new(Shape::circle(1.0).unwrap()).with_friction(-2.0).friction, 0.0);
    }
}
