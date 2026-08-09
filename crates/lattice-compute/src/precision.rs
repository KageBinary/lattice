//! Numeric precision, as §10.5 defines it.
//!
//! A precision is two decisions, not one: what the *state* is stored as, and what
//! *accumulations* over that state are carried in. `mixed` exists precisely because those
//! two answers can differ, and a type that conflates them cannot express it.
//!
//! # Why this lives at the backend boundary
//!
//! Precision is a property of the *device*, not of the model. A model says `temperature:
//! 300 kelvin`; it does not say `f32`. The narrowing from the host's `f64` to whatever the
//! device stores happens at exactly one place — [`crate::Device::write`] — and that is the
//! whole reason a cross-backend disagreement can be attributed rather than merely
//! measured. See [`crate::Tolerance`].

use core::fmt;

/// A numeric mode from §10.5's table.
///
/// Two of these are implemented. The rest are named because the table names them and
/// because a backend has to be able to *decline* one it does not support — see
/// [`crate::Capabilities::supports`]. A mode that is silently downgraded to one the
/// hardware likes is §15.1's faster wrong solver.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Precision {
    /// 32-bit state, 32-bit accumulation. Interactive scenes, visualization-first work.
    Fast32,
    /// 32-bit state, 64-bit reductions and accumulation. §10.5's default GPU scientific
    /// mode "where supported" — and the qualifier is load-bearing, because the portable
    /// backend does not support it. See [`Capabilities`](crate::Capabilities).
    Mixed,
    /// 64-bit throughout. Validated CPU runs, stiff systems, and the reference every
    /// other mode is compared against.
    Accurate64,
}

impl Precision {
    /// The width of one stored element, in bytes.
    pub const fn state_bytes(self) -> usize {
        match self {
            Precision::Fast32 | Precision::Mixed => 4,
            Precision::Accurate64 => 8,
        }
    }

    /// Machine epsilon of the *state* type: the largest relative gap between adjacent
    /// representable numbers.
    ///
    /// This is the unit the cross-backend tolerance is denominated in. One step of an
    /// elementwise kernel introduces a few of these; how many steps' worth accumulate is
    /// the question [`Tolerance`](crate::Tolerance) exists to make someone answer out
    /// loud.
    pub const fn epsilon(self) -> f64 {
        match self {
            // f32::EPSILON, widened. Written as the literal rather than as a cast so the
            // value is visible at the point where it is being reasoned about.
            Precision::Fast32 | Precision::Mixed => 1.192_092_895_507_812_5e-7,
            Precision::Accurate64 => f64::EPSILON,
        }
    }

    /// Epsilon of the type accumulations are carried in.
    ///
    /// Equal to [`Precision::epsilon`] except under [`Precision::Mixed`], which is the
    /// entire point of that mode: a sum of a million `f32` values loses far more to the
    /// accumulator's rounding than to the summands'.
    pub const fn accumulator_epsilon(self) -> f64 {
        match self {
            Precision::Fast32 => 1.192_092_895_507_812_5e-7,
            Precision::Mixed | Precision::Accurate64 => f64::EPSILON,
        }
    }

    /// True if the state is stored at full `f64` width.
    ///
    /// The reference path, in other words: a backend for which this holds can be compared
    /// bit-for-bit against the scalar CPU implementation, and one for which it does not
    /// needs a tolerance and a reason.
    pub const fn is_reference(self) -> bool {
        matches!(self, Precision::Accurate64)
    }

    /// The §10.5 name, as it appears in a run artifact (§19.3).
    pub const fn name(self) -> &'static str {
        match self {
            Precision::Fast32 => "fast32",
            Precision::Mixed => "mixed",
            Precision::Accurate64 => "accurate64",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast32_epsilon_is_f32s() {
        assert_eq!(Precision::Fast32.epsilon(), f64::from(f32::EPSILON));
    }

    #[test]
    fn mixed_differs_from_fast32_only_in_the_accumulator() {
        assert_eq!(Precision::Mixed.epsilon(), Precision::Fast32.epsilon());
        assert!(Precision::Mixed.accumulator_epsilon() < Precision::Fast32.accumulator_epsilon());
    }

    /// The reference is the thing everything else is measured against, so there must be
    /// exactly one of it.
    #[test]
    fn only_accurate64_is_the_reference() {
        let modes = [Precision::Fast32, Precision::Mixed, Precision::Accurate64];
        let references: Vec<_> = modes.iter().filter(|m| m.is_reference()).collect();
        assert_eq!(references, [&Precision::Accurate64]);
    }

    #[test]
    fn state_width_matches_the_epsilon_it_implies() {
        assert_eq!(Precision::Fast32.state_bytes(), 4);
        assert_eq!(Precision::Accurate64.state_bytes(), 8);
        assert!(Precision::Accurate64.epsilon() < Precision::Fast32.epsilon());
    }
}
