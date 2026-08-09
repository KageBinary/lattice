//! Dimensional analysis, physical quantities, and unit parsing for Project Lattice.
//!
//! This crate is the foundation of spec requirement **FR-002**: *"Validate physical
//! units and dimensional consistency at model compile time wherever expressions are
//! statically known."* It provides three layers:
//!
//! | Layer | Type | Role |
//! |---|---|---|
//! | Dimension | [`Dimension`] | The *kind* of a quantity — `L·T^-2` — as a 7-byte exponent vector. |
//! | Quantity | [`Quantity`] | A magnitude in coherent SI plus its dimension. |
//! | Units | [`UnitRegistry`] | Names → scale/offset/dimension, with SI prefixes and an expression parser. |
//!
//! # The one invariant
//!
//! **Every magnitude in the engine is in coherent SI base units.** Units are a
//! surface-syntax concern. `35 kilojoule/mole` becomes `35000.0` at parse time, and
//! from there the runtime is unit-free `f64` arithmetic. Nothing in the hot loop
//! converts anything (spec P3, P4).
//!
//! The consequence is that `Quantity` belongs to the *compiler*, not the solver.
//! Solvers receive plain `f64` arrays. If you find `Quantity` inside a stepping
//! function, something has gone wrong.
//!
//! # Example
//!
//! ```
//! use lattice_units::{Dimension, DimensionalOp, UnitRegistry};
//!
//! let reg = UnitRegistry::si();
//!
//! // Parse a dimensioned literal from a model file.
//! let ea = reg.parse_quantity("35 kilojoule / mole").unwrap().into_value();
//! assert_eq!(ea.dimension(), Dimension::MOLAR_ENERGY);
//!
//! // Extract the SI magnitude a solver will use, proving the dimension is right.
//! let ea_si = ea.require(Dimension::MOLAR_ENERGY, DimensionalOp::Assignment).unwrap();
//! assert_eq!(ea_si, 35_000.0);
//!
//! // A wrong dimension is a structured error, never a silent conversion.
//! assert!(ea.require(Dimension::ENERGY, DimensionalOp::Assignment).is_err());
//! ```
//!
//! # What this crate deliberately does not do
//!
//! - **Fractional dimensions.** Exponents are integers, so `sqrt(meter)` has no
//!   representation. [`Dimension::root`] reports this rather than rounding.
//! - **Guessing at `a/b*c`.** Left-to-right grouping is applied and a
//!   [`UnitWarning`] is emitted; see [`UnitRegistry::parse_unit`] for why.
//! - **Affine units in expressions.** `joule/celsius` is rejected, because the
//!   Celsius offset has no consistent meaning under division.

pub mod constants;
mod dimension;
mod error;
mod parse;
mod quantity;
mod registry;

pub use dimension::{BaseDim, Dimension, DimensionError, NUM_BASE};
pub use error::{DimensionalOp, UnitError, UnitWarning};
pub use parse::{Parsed, UnitTerm};
pub use quantity::Quantity;
pub use registry::{ResolvedUnit, UnitDef, UnitRegistry};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Every dimensioned literal that appears in the spec's example models (§12.2,
    /// §25.1, §25.2) must parse to the dimension a reader would expect. This is the
    /// crate's acceptance test for FR-002: if the spec's own examples do not compile,
    /// the units layer is not done.
    #[test]
    fn spec_example_literals_parse_with_expected_dimensions() {
        let reg = UnitRegistry::si();
        let cases: &[(&str, Dimension)] = &[
            // §12.2 species and reaction declarations
            ("+1 elementary_charge", Dimension::CHARGE),
            ("9.31e-9 meter^2 / second", Dimension::DIFFUSIVITY),
            ("-57.3 kilojoule / mole", Dimension::MOLAR_ENERGY),
            // §25.1 coupled reaction/diffusion/heat project
            ("298 kelvin", Dimension::TEMPERATURE),
            ("1 mole / meter^2", Dimension::AREAL_CONCENTRATION),
            ("2.0e5 / second", Dimension::FREQUENCY),
            ("35 kilojoule/mole", Dimension::MOLAR_ENERGY),
            ("-25 kilojoule / mole", Dimension::MOLAR_ENERGY),
            ("0.002 second", Dimension::TIME),
            ("2 meter", Dimension::LENGTH),
            // §25.2 quantum double slit
            ("12 nanometer", Dimension::LENGTH),
            ("electron_mass", Dimension::MASS),
            ("0.002 femtosecond", Dimension::TIME),
            ("0.15 nanometer", Dimension::LENGTH),
            ("20 electronvolt", Dimension::ENERGY),
            ("6.5e-24 kilogram*meter/second", Dimension::MOMENTUM),
            ("0.45 nanometer", Dimension::LENGTH),
        ];
        for (src, expected) in cases {
            let parsed = reg
                .parse_quantity(src)
                .unwrap_or_else(|e| panic!("`{src}` failed to parse: {e}"));
            assert_eq!(
                parsed.value.dimension(),
                *expected,
                "`{src}` parsed as {} but should be {}",
                parsed.value.dimension().describe(),
                expected.describe()
            );
            assert!(parsed.is_clean(), "`{src}` produced warnings: {:?}", parsed.warnings);
        }
    }

    /// Spec §20.4 requires that "typed units reject at least ten deliberately invalid
    /// fixture models with source-positioned errors". These are the unit-level half
    /// of that: ten expressions that must fail, each for a stated reason.
    #[test]
    fn deliberately_invalid_expressions_are_rejected() {
        let reg = UnitRegistry::si();
        let cases: &[(&str, &str)] = &[
            ("5 metre_", "misspelled unit"),
            ("5 furlong", "unknown unit"),
            ("5 kilominute", "prefix on a non-prefixable unit"),
            ("5 joule / celsius", "affine unit inside an expression"),
            ("5 meter /", "trailing operator"),
            ("5 (meter", "unclosed parenthesis"),
            ("5 meter^", "missing exponent"),
            ("5 meter^0.5", "fractional exponent"),
            ("5 meter^100", "exponent out of range"),
            ("5 meter @ second", "illegal character"),
            ("5 meter ) second", "unbalanced parenthesis"),
        ];
        assert!(cases.len() >= 10, "spec §20.4 asks for at least ten");
        for (src, reason) in cases {
            let err = reg.parse_quantity(src).unwrap_err();
            // Every rejection must produce a message a user can act on.
            let text = err.to_string();
            assert!(!text.is_empty(), "`{src}` ({reason}) produced an empty message");
        }
    }

    /// A dimensional error in a rate law is the motivating example in FR-002. This
    /// walks the full path: parse two literals, combine them the way a rate
    /// expression would, and confirm the mistake surfaces as a typed error.
    #[test]
    fn a_wrong_rate_law_fails_dimensional_checking() {
        let reg = UnitRegistry::si();
        // Arrhenius prefactor for a first-order reaction: 1/s.
        let a0 = reg.parse_quantity("2.0e5 / second").unwrap().into_value();
        // Concentration on a 2D grid: mol/m^2.
        let c = reg.parse_quantity("1 mole / meter^2").unwrap().into_value();

        // Correct: rate = A0 * c has dimension mol/(m^2 s).
        let rate = a0 * c;
        assert_eq!(rate.dimension(), Dimension::AREAL_REACTION_RATE);

        // Wrong: a second-order rate law written with a first-order prefactor gives
        // mol^2/(m^4 s), which does not match what the solver's source term expects.
        let wrong = a0 * c * c;
        assert_ne!(wrong.dimension(), Dimension::AREAL_REACTION_RATE);
        let err = wrong
            .require(Dimension::AREAL_REACTION_RATE, DimensionalOp::PortTransfer)
            .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("coupling port transfer"), "{text}");
        assert!(text.contains("areal reaction rate"), "{text}");
    }

    /// The engine stores SI internally but users read plots in their own units.
    /// Round-tripping through `value_in` must be lossless to floating-point
    /// precision, including for affine temperature scales.
    #[test]
    fn si_round_trip_through_display_units() {
        let reg = UnitRegistry::si();
        let cases: &[(&str, &str, f64)] = &[
            ("35 kilojoule/mole", "kilojoule/mole", 35.0),
            ("2 hour", "minute", 120.0),
            ("1 bar", "kilopascal", 100.0),
            ("298.15 kelvin", "celsius", 25.0),
            ("1 electronvolt", "joule", 1.602_176_634e-19),
            ("1 nanometer", "angstrom", 10.0),
        ];
        for (literal, display_unit, expected) in cases {
            let q = reg.parse_quantity(literal).unwrap().into_value();
            let shown = reg.value_in(&q, display_unit).unwrap();
            assert!(
                (shown - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{literal} in {display_unit} = {shown}, expected {expected}"
            );
        }
    }
}
