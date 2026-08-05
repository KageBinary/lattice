//! Errors and warnings produced by unit resolution and dimensional checking.
//!
//! These carry enough structure for the model compiler to attach source positions
//! and render the diagnostics FR-002 requires ("invalid force/energy/rate expressions
//! fail with source-positioned diagnostics"). Nothing here knows about source spans —
//! that is the compiler's job; this crate supplies the *reason*.

use core::fmt;

use crate::dimension::{Dimension, DimensionError};

/// The operation that required two dimensions to agree.
///
/// Included in mismatch errors so the message can say what the user was doing rather
/// than only what disagreed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimensionalOp {
    Addition,
    Subtraction,
    Comparison,
    /// Assignment into a declared state variable, parameter, or port.
    Assignment,
    /// A value handed across a coupling port (spec §14.1).
    PortTransfer,
}

impl DimensionalOp {
    const fn describe(self) -> &'static str {
        match self {
            DimensionalOp::Addition => "addition",
            DimensionalOp::Subtraction => "subtraction",
            DimensionalOp::Comparison => "comparison",
            DimensionalOp::Assignment => "assignment",
            DimensionalOp::PortTransfer => "coupling port transfer",
        }
    }
}

/// Everything that can go wrong while resolving units or checking dimensions.
#[derive(Clone, PartialEq, Debug)]
pub enum UnitError {
    /// Two dimensions had to agree and did not.
    Mismatch {
        op: DimensionalOp,
        expected: Dimension,
        found: Dimension,
    },
    /// Dimensional arithmetic failed (overflow, fractional root).
    Arithmetic(DimensionError),
    /// A unit name could not be resolved, with a best-effort spelling suggestion.
    UnknownUnit {
        name: String,
        suggestion: Option<String>,
    },
    /// The unit expression was malformed.
    Syntax {
        message: String,
        /// Byte offset into the unit string where the problem was detected.
        offset: usize,
    },
    /// An offset unit (degree Celsius, degree Fahrenheit) appeared in a compound
    /// expression.
    ///
    /// `celsius` is an affine scale, not a linear one: `20 °C` is not `20 K` scaled,
    /// and `J/°C` is not meaningful the way `J/K` is. Rejecting this outright is more
    /// honest than silently dropping the offset (spec P1).
    OffsetUnitInExpression {
        name: String,
    },
    /// A quantity was required to be a pure number but carried a dimension.
    ExpectedDimensionless {
        found: Dimension,
    },
    /// A unit registration collided with an existing name.
    DuplicateUnit {
        name: String,
    },
}

impl From<DimensionError> for UnitError {
    fn from(e: DimensionError) -> Self {
        UnitError::Arithmetic(e)
    }
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::Mismatch { op, expected, found } => write!(
                f,
                "dimensional mismatch in {}: expected {}, found {}",
                op.describe(),
                expected.describe(),
                found.describe()
            ),
            UnitError::Arithmetic(e) => write!(f, "{e}"),
            UnitError::UnknownUnit { name, suggestion } => match suggestion {
                Some(s) => write!(f, "unknown unit `{name}` (did you mean `{s}`?)"),
                None => write!(f, "unknown unit `{name}`"),
            },
            UnitError::Syntax { message, offset } => {
                write!(f, "malformed unit expression at byte {offset}: {message}")
            }
            UnitError::OffsetUnitInExpression { name } => write!(
                f,
                "`{name}` is an offset temperature scale and cannot be combined with other units; \
                 use `kelvin` for temperature differences and rates"
            ),
            UnitError::ExpectedDimensionless { found } => {
                write!(f, "expected a dimensionless value, found {}", found.describe())
            }
            UnitError::DuplicateUnit { name } => write!(f, "unit `{name}` is already registered"),
        }
    }
}

impl core::error::Error for UnitError {}

/// A non-fatal problem with a unit expression.
///
/// Warnings never block compilation, but the model report (spec §8.4, step 9) prints
/// them so a user can see that their notation was interpreted in a specific way.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum UnitWarning {
    /// A `*` appeared after a `/` at the same parenthesis depth.
    ///
    /// `kJ/mol*K` is read left-to-right as `(kJ/mol)*K`, but many scientific texts
    /// write it meaning `kJ/(mol*K)`. Rather than guess, Lattice applies the
    /// left-to-right rule and says so.
    AmbiguousSlashProduct {
        /// Byte offset of the `*` that triggered the warning.
        offset: usize,
        /// How Lattice actually grouped the expression.
        interpretation: String,
    },
}

impl fmt::Display for UnitWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitWarning::AmbiguousSlashProduct { offset, interpretation } => write!(
                f,
                "multiplication after division at byte {offset} is ambiguous; \
                 Lattice evaluates left-to-right as {interpretation}. \
                 Add parentheses to state the intended grouping."
            ),
        }
    }
}
