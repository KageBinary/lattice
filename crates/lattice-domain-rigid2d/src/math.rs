//! Small 2D vector and rotation types.
//!
//! # Why a `Vec2` here when the rest of the engine uses `[f64; 2]`
//!
//! Storage stays as structure-of-arrays `f64` — that decision belongs to
//! [`lattice_ir`] and is not revisited. But contact and constraint maths is written in
//! terms of `r × n`, `v + ω × r`, and `n · (vb − va)`, and spelling those out on raw
//! arrays produces lines nobody can check by eye:
//!
//! ```text
//! let rn_a = ra[0] * n[1] - ra[1] * n[0];
//! ```
//!
//! An impulse solver is exactly the place where a transposed index survives every test
//! that only checks momentum, because momentum is conserved by construction. So the
//! computation gets a readable type and the crate boundary keeps `[f64; 2]`.

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A 2D vector.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Vec2 {
    /// x component.
    pub x: f64,
    /// y component.
    pub y: f64,
}

/// Shorthand constructor.
pub const fn vec2(x: f64, y: f64) -> Vec2 {
    Vec2 { x, y }
}

impl Vec2 {
    /// The origin.
    pub const ZERO: Vec2 = vec2(0.0, 0.0);
    /// The x basis vector.
    pub const X: Vec2 = vec2(1.0, 0.0);
    /// The y basis vector.
    pub const Y: Vec2 = vec2(0.0, 1.0);

    /// Dot product.
    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// The z component of the 3D cross product — the only component two 2D vectors
    /// produce, and the quantity that turns a lever arm and a force into a torque.
    pub fn cross(self, other: Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// `ω × r` for a scalar angular velocity: the velocity a point at `self` picks up
    /// from spinning about the origin at `omega`.
    pub fn cross_scalar(self, omega: f64) -> Vec2 {
        vec2(-omega * self.y, omega * self.x)
    }

    /// Rotated a quarter turn counter-clockwise.
    pub fn perp(self) -> Vec2 {
        vec2(-self.y, self.x)
    }

    /// Squared length. Preferred wherever the square root would be undone anyway.
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    /// Length.
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Unit vector, or `None` when there is no direction to speak of.
    ///
    /// Returns `None` rather than a NaN-filled vector or an arbitrary axis. A contact
    /// normal between two exactly coincident circles is genuinely undefined, and the
    /// caller has to decide what to do about it — silently substituting `(1, 0)` puts
    /// a fabricated direction into an impulse.
    pub fn normalize(self) -> Option<Vec2> {
        let length = self.length();
        (length > 0.0 && length.is_finite()).then(|| self / length)
    }

    /// Componentwise minimum.
    pub fn min(self, other: Vec2) -> Vec2 {
        vec2(self.x.min(other.x), self.y.min(other.y))
    }

    /// Componentwise maximum.
    pub fn max(self, other: Vec2) -> Vec2 {
        vec2(self.x.max(other.x), self.y.max(other.y))
    }

    /// True when both components are finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// As the `[x, y]` array the rest of the engine speaks.
    pub fn to_array(self) -> [f64; 2] {
        [self.x, self.y]
    }
}

impl From<[f64; 2]> for Vec2 {
    fn from([x, y]: [f64; 2]) -> Self {
        vec2(x, y)
    }
}

impl From<Vec2> for [f64; 2] {
    fn from(v: Vec2) -> Self {
        [v.x, v.y]
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        vec2(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        vec2(self.x - other.x, self.y - other.y)
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        vec2(-self.x, -self.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, scale: f64) -> Vec2 {
        vec2(self.x * scale, self.y * scale)
    }
}

impl Mul<Vec2> for f64 {
    type Output = Vec2;
    fn mul(self, v: Vec2) -> Vec2 {
        v * self
    }
}

impl Div<f64> for Vec2 {
    type Output = Vec2;
    fn div(self, divisor: f64) -> Vec2 {
        vec2(self.x / divisor, self.y / divisor)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, other: Vec2) {
        *self = *self + other;
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, other: Vec2) {
        *self = *self - other;
    }
}

/// A rotation, stored as the cosine and sine of its angle.
///
/// Storing the pair rather than the angle means rotating a vector never calls `sin` or
/// `cos` — which matters when every contact point of every body is transformed every
/// step — and it keeps composition exact rather than accumulating an angle that has to
/// be range-reduced.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rot {
    /// Cosine of the angle.
    pub cos: f64,
    /// Sine of the angle.
    pub sin: f64,
}

impl Default for Rot {
    fn default() -> Self {
        Rot::IDENTITY
    }
}

impl Rot {
    /// No rotation.
    pub const IDENTITY: Rot = Rot { cos: 1.0, sin: 0.0 };

    /// From an angle in radians.
    pub fn from_angle(radians: f64) -> Rot {
        Rot { cos: radians.cos(), sin: radians.sin() }
    }

    /// The angle in radians, in `(-π, π]`.
    pub fn angle(self) -> f64 {
        self.sin.atan2(self.cos)
    }

    /// Body-local to world.
    pub fn apply(self, v: Vec2) -> Vec2 {
        vec2(self.cos * v.x - self.sin * v.y, self.sin * v.x + self.cos * v.y)
    }

    /// World to body-local — the transpose, since a rotation is orthogonal.
    pub fn unapply(self, v: Vec2) -> Vec2 {
        vec2(self.cos * v.x + self.sin * v.y, -self.sin * v.x + self.cos * v.y)
    }

    /// Renormalize after accumulating small increments.
    ///
    /// Integrating `cos`/`sin` directly lets the pair drift off the unit circle, which
    /// shows up as a body that slowly grows or shrinks. One division per body per step
    /// is cheaper than the `sin`/`cos` pair that storing an angle would cost anyway.
    pub fn normalized(self) -> Rot {
        let magnitude = (self.cos * self.cos + self.sin * self.sin).sqrt();
        if magnitude > 0.0 && magnitude.is_finite() {
            Rot { cos: self.cos / magnitude, sin: self.sin / magnitude }
        } else {
            Rot::IDENTITY
        }
    }
}

/// A rigid transform: rotation then translation.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Transform {
    /// Origin of the body frame, in world coordinates.
    pub position: Vec2,
    /// Orientation of the body frame.
    pub rotation: Rot,
}

impl Transform {
    /// A transform at `position` with the given rotation.
    pub fn new(position: Vec2, rotation: Rot) -> Transform {
        Transform { position, rotation }
    }

    /// Body-local point to world.
    pub fn apply(self, local: Vec2) -> Vec2 {
        self.position + self.rotation.apply(local)
    }

    /// World point to body-local.
    pub fn unapply(self, world: Vec2) -> Vec2 {
        self.rotation.unapply(world - self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn close(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < EPSILON
    }

    #[test]
    fn cross_products_follow_the_right_hand_rule() {
        assert_eq!(Vec2::X.cross(Vec2::Y), 1.0);
        assert_eq!(Vec2::Y.cross(Vec2::X), -1.0);
        assert_eq!(Vec2::X.cross(Vec2::X), 0.0, "parallel vectors have no cross product");
    }

    /// The identity every impulse formula leans on: a point at `r` on a body spinning
    /// at `ω` moves at `ω × r`, perpendicular to `r` and counter-clockwise.
    #[test]
    fn spinning_moves_a_point_perpendicular_to_its_lever_arm() {
        let r = vec2(2.0, 0.0);
        let velocity = r.cross_scalar(3.0);
        assert!(close(velocity, vec2(0.0, 6.0)), "{velocity:?}");
        assert!(velocity.dot(r).abs() < EPSILON, "must be perpendicular to the arm");
    }

    #[test]
    fn rotations_round_trip_and_compose_with_their_angle() {
        let rot = Rot::from_angle(0.7);
        let v = vec2(3.0, -1.5);
        assert!(close(rot.unapply(rot.apply(v)), v), "unapply must invert apply");
        assert!((rot.angle() - 0.7).abs() < EPSILON);
        assert!(close(Rot::from_angle(core::f64::consts::FRAC_PI_2).apply(Vec2::X), Vec2::Y));
    }

    #[test]
    fn a_rotation_preserves_length() {
        let v = vec2(3.0, 4.0);
        assert!((Rot::from_angle(1.234).apply(v).length() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn renormalizing_pulls_a_drifted_rotation_back_to_the_unit_circle() {
        let drifted = Rot { cos: 1.0, sin: 0.02 };
        assert!(drifted.normalized().apply(Vec2::X).length() - 1.0 < EPSILON);
        // And a rotation that has collapsed entirely does not produce NaN.
        assert_eq!(Rot { cos: 0.0, sin: 0.0 }.normalized(), Rot::IDENTITY);
    }

    /// A contact normal between two coincident bodies is genuinely undefined.
    /// Substituting an arbitrary axis would put a fabricated direction into an impulse.
    #[test]
    fn normalizing_nothing_yields_nothing() {
        assert_eq!(Vec2::ZERO.normalize(), None);
        assert_eq!(vec2(f64::NAN, 0.0).normalize(), None);
        assert!(close(vec2(3.0, 4.0).normalize().unwrap(), vec2(0.6, 0.8)));
    }

    #[test]
    fn transforms_round_trip() {
        let t = Transform::new(vec2(5.0, -2.0), Rot::from_angle(-0.9));
        let local = vec2(1.0, 2.0);
        assert!(close(t.unapply(t.apply(local)), local));
    }
}
