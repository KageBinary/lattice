//! Typed ports: what a domain publishes and what it will consume.
//!
//! Spec §14.1's coupling-edge schema names a *source port*, a *target port*, a
//! mapping, a cadence, and a conservation quantity. This module is the first two: the
//! declaration of what a domain offers, and the buffer a value travels in.
//!
//! # Why values are copied rather than borrowed
//!
//! The obvious design hands the target a reference into the source. It does not
//! compile, and the reason it does not compile is a real constraint rather than a
//! borrow-checker inconvenience: the domains live in one `Vec<Box<dyn Domain>>`, and
//! holding a shared borrow of one while mutably borrowing another would require the
//! scheduler to promise an ordering it does not have.
//!
//! Copying through a staging buffer is also where the *mapping* goes. §14.1 lists it
//! as part of the edge for a reason: a heat release in W/m² is not a temperature rate
//! in K/s, and the factor between them is an areal heat capacity that belongs to
//! neither domain. A borrowed reference would have nowhere to put that conversion and
//! would quietly encourage the two sides to agree on units by accident.
//!
//! The buffer is owned by the scheduler and reused, so the copy costs no allocation
//! after the first step (NFR-001).

use crate::grid::{Grid2d, ScalarField};

/// Which way a port faces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PortDirection {
    /// The domain computes this and offers it to others.
    Publishes,
    /// The domain will accept this from another.
    Consumes,
}

/// What shape a port's value has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PortShape {
    /// One value per grid cell.
    Field,
    /// One value for the whole domain.
    Scalar,
}

/// A port a domain offers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PortSpec {
    /// The name an edge refers to it by, without the domain prefix.
    pub name: &'static str,
    /// Which way it faces.
    pub direction: PortDirection,
    /// What shape its value has.
    pub shape: PortShape,
    /// SI unit, for the compiler to check an edge's mapping against.
    ///
    /// The whole reason a coupling edge can be wrong in a way nothing else catches: a
    /// heat release in `W/m^2` connected straight to a source in `K/s` produces a run
    /// that works, looks plausible, and is off by a heat capacity.
    pub unit: &'static str,
    /// One line on what it means, for the model report.
    pub description: &'static str,
}

impl PortSpec {
    /// A field this domain publishes.
    pub const fn publishes_field(
        name: &'static str,
        unit: &'static str,
        description: &'static str,
    ) -> PortSpec {
        PortSpec {
            name,
            direction: PortDirection::Publishes,
            shape: PortShape::Field,
            unit,
            description,
        }
    }

    /// A field this domain will accept.
    pub const fn consumes_field(
        name: &'static str,
        unit: &'static str,
        description: &'static str,
    ) -> PortSpec {
        PortSpec {
            name,
            direction: PortDirection::Consumes,
            shape: PortShape::Field,
            unit,
            description,
        }
    }

    /// A single number this domain publishes.
    pub const fn publishes_scalar(
        name: &'static str,
        unit: &'static str,
        description: &'static str,
    ) -> PortSpec {
        PortSpec {
            name,
            direction: PortDirection::Publishes,
            shape: PortShape::Scalar,
            unit,
            description,
        }
    }

    /// A single number this domain will accept.
    pub const fn consumes_scalar(
        name: &'static str,
        unit: &'static str,
        description: &'static str,
    ) -> PortSpec {
        PortSpec {
            name,
            direction: PortDirection::Consumes,
            shape: PortShape::Scalar,
            unit,
            description,
        }
    }
}

/// A port's value, copied out of one domain on its way to another.
///
/// Owned and reused by the scheduler, so an exchange allocates nothing once the shapes
/// have settled.
#[derive(Clone, PartialEq, Debug)]
pub enum PortData {
    /// A per-cell field.
    Field(ScalarField),
    /// A single number.
    Scalar(f64),
}

impl PortData {
    /// A field-shaped buffer sized for `grid`.
    pub fn field(grid: &Grid2d) -> PortData {
        PortData::Field(ScalarField::new(grid, 1))
    }

    /// A scalar-shaped buffer.
    pub fn scalar() -> PortData {
        PortData::Scalar(0.0)
    }

    /// The shape this buffer holds.
    pub fn shape(&self) -> PortShape {
        match self {
            PortData::Field(_) => PortShape::Field,
            PortData::Scalar(_) => PortShape::Scalar,
        }
    }

    /// The field, if this is one.
    pub fn as_field(&self) -> Option<&ScalarField> {
        match self {
            PortData::Field(field) => Some(field),
            PortData::Scalar(_) => None,
        }
    }

    /// The field, mutably.
    pub fn as_field_mut(&mut self) -> Option<&mut ScalarField> {
        match self {
            PortData::Field(field) => Some(field),
            PortData::Scalar(_) => None,
        }
    }

    /// The number, if this is one.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            PortData::Scalar(value) => Some(*value),
            PortData::Field(_) => None,
        }
    }

    /// Multiply every value by `factor`.
    ///
    /// Where a coupling edge's unit conversion is applied. Kept on the buffer rather
    /// than on either domain because the factor belongs to neither: an areal heat
    /// capacity relating `W/m^2` to `K/s` is a property of the *material between them*.
    pub fn scale(&mut self, factor: f64) {
        match self {
            PortData::Field(field) => {
                for value in field.as_mut_slice() {
                    *value *= factor;
                }
            }
            PortData::Scalar(value) => *value *= factor,
        }
    }

    /// Set everything to zero, keeping the shape and the allocation.
    pub fn clear(&mut self) {
        match self {
            PortData::Field(field) => field.fill_all(0.0),
            PortData::Scalar(value) => *value = 0.0,
        }
    }

    /// The integral of a field over `grid`, or the scalar itself.
    ///
    /// What the ledger records: a per-cell rate integrated over the area it acts on is
    /// the total the edge actually moved.
    pub fn total(&self, grid: &Grid2d) -> f64 {
        match self {
            PortData::Field(field) => field.integrate(grid),
            PortData::Scalar(value) => *value,
        }
    }

    /// True when any value is not finite.
    pub fn has_non_finite(&self) -> bool {
        match self {
            PortData::Field(field) => field.first_non_finite().is_some(),
            PortData::Scalar(value) => !value.is_finite(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid2d {
        Grid2d::new(4, 4, [2.0, 2.0])
    }

    #[test]
    fn a_field_buffer_matches_its_grid() {
        let grid = grid();
        let mut data = PortData::field(&grid);
        assert_eq!(data.shape(), PortShape::Field);
        assert!(data.as_scalar().is_none());

        data.as_field_mut().unwrap().fill_interior(3.0);
        // 16 cells of area 0.25 m² each, at 3 units: 12.
        assert!((data.total(&grid) - 12.0).abs() < 1e-12);
    }

    /// The unit conversion an edge carries lives on the buffer, because the factor
    /// belongs to neither domain.
    #[test]
    fn scaling_applies_the_edges_conversion() {
        let grid = grid();
        let mut data = PortData::field(&grid);
        data.as_field_mut().unwrap().fill_interior(100.0);
        // 100 W/m² through an areal heat capacity of 200 J/(m²·K) is 0.5 K/s.
        data.scale(1.0 / 200.0);
        assert!((data.as_field().unwrap().get(0, 0) - 0.5).abs() < 1e-12);

        let mut scalar = PortData::Scalar(4.0);
        scalar.scale(0.25);
        assert_eq!(scalar.as_scalar(), Some(1.0));
    }

    #[test]
    fn clearing_keeps_the_shape() {
        let grid = grid();
        let mut data = PortData::field(&grid);
        data.as_field_mut().unwrap().fill_all(7.0);
        data.clear();
        assert_eq!(data.shape(), PortShape::Field);
        assert_eq!(data.total(&grid), 0.0);
    }

    #[test]
    fn non_finite_values_are_detected() {
        let grid = grid();
        let mut data = PortData::field(&grid);
        assert!(!data.has_non_finite());
        data.as_field_mut().unwrap().set(1, 1, f64::NAN);
        assert!(data.has_non_finite(), "a NaN crossing an edge must not travel unnoticed");
        assert!(PortData::Scalar(f64::INFINITY).has_non_finite());
    }

    /// A port's unit is what makes a mis-wired edge findable: `W/m^2` connected to
    /// `K/s` produces a run that works and is off by a heat capacity.
    #[test]
    fn port_specs_carry_their_units_and_direction() {
        let out = PortSpec::publishes_field("heat_release", "W/m^2", "heat from reactions");
        let into = PortSpec::consumes_field("source", "K/s", "a temperature source term");
        assert_eq!(out.direction, PortDirection::Publishes);
        assert_eq!(into.direction, PortDirection::Consumes);
        assert_ne!(out.unit, into.unit, "which is exactly why an edge needs a mapping");

        let scalar = PortSpec::publishes_scalar("mean_temperature", "K", "area-weighted mean");
        assert_eq!(scalar.shape, PortShape::Scalar);
        assert_eq!(PortSpec::consumes_scalar("rate", "1/s", "").shape, PortShape::Scalar);
    }
}
