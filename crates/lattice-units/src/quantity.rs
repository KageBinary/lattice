//! Scalar physical quantities: a magnitude in coherent SI units plus a [`Dimension`].
//!
//! # Why magnitudes are always SI
//!
//! A [`Quantity`] stores its value already converted to coherent SI base units. A
//! `Quantity` parsed from `"35 kilojoule/mole"` holds `35_000.0` with dimension
//! `kg·m^2/s^2/mol`. Units are a *surface syntax* concern; once a model is compiled,
//! every number in the runtime is SI and no conversion happens in the hot loop
//! (spec P3: compile models before running).
//!
//! # Why there is no `Add` operator
//!
//! `Mul` and `Div` are total operations on quantities — any two dimensions can be
//! multiplied. Addition is not: `1 meter + 1 second` is meaningless. Implementing
//! `Add` would force a panic on mismatch, and a compiler that panics on invalid user
//! input is a bad compiler. So addition is [`Quantity::try_add`], which returns the
//! structured error FR-002 needs.

use core::fmt;
use core::ops::{Div, Mul, Neg};

use crate::dimension::Dimension;
use crate::error::{DimensionalOp, UnitError};

/// A scalar physical quantity: an SI magnitude tagged with its dimension.
#[derive(Clone, Copy, PartialEq)]
pub struct Quantity {
    value: f64,
    dim: Dimension,
}

impl Quantity {
    /// A quantity from a magnitude already expressed in coherent SI base units.
    pub const fn new(value: f64, dim: Dimension) -> Self {
        Self { value, dim }
    }

    /// A pure number.
    pub const fn dimensionless(value: f64) -> Self {
        Self { value, dim: Dimension::DIMENSIONLESS }
    }

    /// Zero with a given dimension. Useful as an accumulator seed.
    pub const fn zero(dim: Dimension) -> Self {
        Self { value: 0.0, dim }
    }

    /// The magnitude in coherent SI base units.
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// The dimension of this quantity.
    pub const fn dimension(&self) -> Dimension {
        self.dim
    }

    /// True if the magnitude is finite (not NaN or infinite).
    ///
    /// NFR-007 requires the engine to surface non-finite values rather than hide
    /// them; parameter validation calls this before a value reaches a solver.
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    /// Add, requiring identical dimensions.
    pub fn try_add(self, rhs: Self) -> Result<Self, UnitError> {
        self.check_same(rhs, DimensionalOp::Addition)?;
        Ok(Self { value: self.value + rhs.value, dim: self.dim })
    }

    /// Subtract, requiring identical dimensions.
    pub fn try_sub(self, rhs: Self) -> Result<Self, UnitError> {
        self.check_same(rhs, DimensionalOp::Subtraction)?;
        Ok(Self { value: self.value - rhs.value, dim: self.dim })
    }

    /// Order two quantities, requiring identical dimensions.
    ///
    /// Returns `None` only when a magnitude is NaN.
    pub fn try_partial_cmp(self, rhs: Self) -> Result<Option<core::cmp::Ordering>, UnitError> {
        self.check_same(rhs, DimensionalOp::Comparison)?;
        Ok(self.value.partial_cmp(&rhs.value))
    }

    /// Raise to an integer power.
    pub fn try_powi(self, n: i32) -> Result<Self, UnitError> {
        Ok(Self { value: self.value.powi(n), dim: self.dim.try_powi(n)? })
    }

    /// Take the degree-`n` root. Fails if the dimension has a non-divisible exponent.
    pub fn root(self, n: i32) -> Result<Self, UnitError> {
        let dim = self.dim.root(n)?;
        Ok(Self { value: self.value.powf(1.0 / f64::from(n)), dim })
    }

    /// Reciprocal.
    pub fn recip(self) -> Result<Self, UnitError> {
        Ok(Self { value: 1.0 / self.value, dim: self.dim.try_recip()? })
    }

    /// Absolute value, preserving dimension.
    pub fn abs(self) -> Self {
        Self { value: self.value.abs(), dim: self.dim }
    }

    /// The SI magnitude, but only if the dimension is exactly `expected`.
    ///
    /// This is the main entry point for the model compiler: it converts a
    /// user-written quantity into the raw `f64` a solver consumes while proving the
    /// dimension is right. `op` appears in the error message.
    ///
    /// ```
    /// use lattice_units::{Dimension, DimensionalOp, Quantity};
    /// let dt = Quantity::new(0.002, Dimension::TIME);
    /// assert_eq!(dt.require(Dimension::TIME, DimensionalOp::Assignment).unwrap(), 0.002);
    /// assert!(dt.require(Dimension::LENGTH, DimensionalOp::Assignment).is_err());
    /// ```
    pub fn require(&self, expected: Dimension, op: DimensionalOp) -> Result<f64, UnitError> {
        if self.dim == expected {
            Ok(self.value)
        } else {
            Err(UnitError::Mismatch { op, expected, found: self.dim })
        }
    }

    /// The magnitude, but only if this is a pure number.
    pub fn require_dimensionless(&self) -> Result<f64, UnitError> {
        if self.dim.is_dimensionless() {
            Ok(self.value)
        } else {
            Err(UnitError::ExpectedDimensionless { found: self.dim })
        }
    }

    fn check_same(self, rhs: Self, op: DimensionalOp) -> Result<(), UnitError> {
        if self.dim == rhs.dim {
            Ok(())
        } else {
            Err(UnitError::Mismatch { op, expected: self.dim, found: rhs.dim })
        }
    }
}

/// Multiplication is total: any two dimensions compose.
///
/// The only failure mode is exponent overflow, which requires exponents beyond ±127
/// and cannot arise from parsed input (the parser bounds exponents). Rather than make
/// every multiplication fallible, this saturates the exponent — and
/// [`Quantity::checked_mul`] exists for callers that must know.
impl Mul for Quantity {
    type Output = Quantity;

    fn mul(self, rhs: Self) -> Quantity {
        let dim = self.dim.try_mul(rhs.dim).unwrap_or_else(|_| saturating_mul(self.dim, rhs.dim));
        Quantity { value: self.value * rhs.value, dim }
    }
}

impl Div for Quantity {
    type Output = Quantity;

    fn div(self, rhs: Self) -> Quantity {
        let dim = self.dim.try_div(rhs.dim).unwrap_or_else(|_| saturating_div(self.dim, rhs.dim));
        Quantity { value: self.value / rhs.value, dim }
    }
}

impl Mul<f64> for Quantity {
    type Output = Quantity;

    fn mul(self, rhs: f64) -> Quantity {
        Quantity { value: self.value * rhs, dim: self.dim }
    }
}

impl Mul<Quantity> for f64 {
    type Output = Quantity;

    fn mul(self, rhs: Quantity) -> Quantity {
        Quantity { value: self * rhs.value, dim: rhs.dim }
    }
}

impl Div<f64> for Quantity {
    type Output = Quantity;

    fn div(self, rhs: f64) -> Quantity {
        Quantity { value: self.value / rhs, dim: self.dim }
    }
}

impl Neg for Quantity {
    type Output = Quantity;

    fn neg(self) -> Quantity {
        Quantity { value: -self.value, dim: self.dim }
    }
}

impl Quantity {
    /// Multiply, reporting exponent overflow instead of saturating.
    pub fn checked_mul(self, rhs: Self) -> Result<Self, UnitError> {
        Ok(Self { value: self.value * rhs.value, dim: self.dim.try_mul(rhs.dim)? })
    }

    /// Divide, reporting exponent overflow instead of saturating.
    pub fn checked_div(self, rhs: Self) -> Result<Self, UnitError> {
        Ok(Self { value: self.value / rhs.value, dim: self.dim.try_div(rhs.dim)? })
    }
}

fn saturating_mul(a: Dimension, b: Dimension) -> Dimension {
    let (ae, be) = (a.exponents(), b.exponents());
    let mut e = [0i8; crate::dimension::NUM_BASE];
    for (i, slot) in e.iter_mut().enumerate() {
        *slot = ae[i].saturating_add(be[i]);
    }
    Dimension::from_exponents(e)
}

fn saturating_div(a: Dimension, b: Dimension) -> Dimension {
    let (ae, be) = (a.exponents(), b.exponents());
    let mut e = [0i8; crate::dimension::NUM_BASE];
    for (i, slot) in e.iter_mut().enumerate() {
        *slot = ae[i].saturating_sub(be[i]);
    }
    Dimension::from_exponents(e)
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim.is_dimensionless() {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{} {}", self.value, self.dim)
        }
    }
}

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: f64, d: Dimension) -> Quantity {
        Quantity::new(v, d)
    }

    #[test]
    fn multiplication_composes_dimensions() {
        let mass = q(2.0, Dimension::MASS);
        let accel = q(9.81, Dimension::ACCELERATION);
        let force = mass * accel;
        assert_eq!(force.dimension(), Dimension::FORCE);
        assert!((force.value() - 19.62).abs() < 1e-12);
    }

    #[test]
    fn division_composes_dimensions() {
        let distance = q(100.0, Dimension::LENGTH);
        let time = q(4.0, Dimension::TIME);
        let speed = distance / time;
        assert_eq!(speed.dimension(), Dimension::VELOCITY);
        assert_eq!(speed.value(), 25.0);
    }

    #[test]
    fn addition_requires_matching_dimensions() {
        let a = q(1.0, Dimension::LENGTH);
        let b = q(1.0, Dimension::TIME);
        let err = a.try_add(b).unwrap_err();
        assert!(matches!(
            err,
            UnitError::Mismatch { op: DimensionalOp::Addition, .. }
        ));
        // The message must name both dimensions so the user can see the mistake.
        let text = err.to_string();
        assert!(text.contains("length"), "{text}");
        assert!(text.contains("time"), "{text}");
    }

    #[test]
    fn addition_succeeds_for_matching_dimensions() {
        let a = q(1.5, Dimension::ENERGY);
        let b = q(2.5, Dimension::ENERGY);
        assert_eq!(a.try_add(b).unwrap().value(), 4.0);
    }

    #[test]
    fn require_extracts_si_magnitude_when_dimension_matches() {
        let dt = q(0.002, Dimension::TIME);
        assert_eq!(dt.require(Dimension::TIME, DimensionalOp::Assignment).unwrap(), 0.002);
        assert!(dt.require(Dimension::LENGTH, DimensionalOp::Assignment).is_err());
    }

    #[test]
    fn dimensionless_ratio_is_pure_number() {
        let ratio = q(10.0, Dimension::LENGTH) / q(4.0, Dimension::LENGTH);
        assert_eq!(ratio.require_dimensionless().unwrap(), 2.5);
    }

    #[test]
    fn require_dimensionless_rejects_dimensional_values() {
        assert!(matches!(
            q(1.0, Dimension::LENGTH).require_dimensionless(),
            Err(UnitError::ExpectedDimensionless { .. })
        ));
    }

    #[test]
    fn powi_and_root_track_dimensions() {
        let side = q(3.0, Dimension::LENGTH);
        let area = side.try_powi(2).unwrap();
        assert_eq!(area.dimension(), Dimension::AREA);
        assert_eq!(area.value(), 9.0);
        let back = area.root(2).unwrap();
        assert_eq!(back.dimension(), Dimension::LENGTH);
        assert!((back.value() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn root_of_odd_exponent_is_rejected() {
        assert!(q(2.0, Dimension::LENGTH).root(2).is_err());
    }

    #[test]
    fn display_includes_si_units() {
        assert_eq!(q(9.81, Dimension::ACCELERATION).to_string(), "9.81 m/s^2");
        assert_eq!(Quantity::dimensionless(0.5).to_string(), "0.5");
    }

    #[test]
    fn scalar_multiplication_preserves_dimension() {
        let e = q(2.0, Dimension::ENERGY);
        assert_eq!((e * 3.0).value(), 6.0);
        assert_eq!((3.0 * e).dimension(), Dimension::ENERGY);
        assert_eq!((-e).value(), -2.0);
    }

    /// Harmonic-oscillator angular frequency: sqrt(k/m) must come out as 1/s.
    /// This is the canonical case where root() has to succeed.
    #[test]
    fn oscillator_frequency_has_frequency_dimension() {
        let k = q(4.0, Dimension::STIFFNESS);
        let m = q(1.0, Dimension::MASS);
        let omega = (k / m).root(2).unwrap();
        assert_eq!(omega.dimension(), Dimension::FREQUENCY);
        assert!((omega.value() - 2.0).abs() < 1e-12);
    }
}
