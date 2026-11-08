//! Where a user law first produced a non-finite value (NFR-007, spec M6.1d §4.4).
//!
//! A run that halts as non-finite names the observation that went bad. When a law the
//! model wrote is responsible, the author needs the law, the sub-expression and the
//! values that went into it. A domain that evaluates user laws can find them: the runtime
//! replays the run with [`Domain::trace_laws`](crate::Domain::trace_laws) on, and each such
//! domain reports the first [`LawFault`] it meets through
//! [`Domain::law_fault`](crate::Domain::law_fault). The compiler's source positions reach
//! it through the [`Origin`](lattice_expr::Origin) of each operation, so a front end can
//! put a caret under the failing sub-expression.

use lattice_expr::NonFiniteAt;

/// Which pair, particle or cell evaluated the law when it failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultSite {
    /// A pair of particle slots, `a` then `b`.
    Pair(usize, usize),
    /// One particle slot.
    Particle(usize),
    /// One cell of a grid, `(i, j)`.
    Cell(usize, usize),
}

impl FaultSite {
    /// For a message: `pair (3, 7)`, `particle 4`, `cell (2, 5)`.
    pub fn describe(self) -> String {
        match self {
            FaultSite::Pair(a, b) => format!("pair ({a}, {b})"),
            FaultSite::Particle(k) => format!("particle {k}"),
            FaultSite::Cell(i, j) => format!("cell ({i}, {j})"),
        }
    }
}

/// The first non-finite value a user law produced while a run was replayed.
#[derive(Clone, PartialEq, Debug)]
pub struct LawFault {
    /// The domain whose law it was.
    pub domain: String,
    /// The law's name: a `force` or `potential`, or the reaction whose `rate:` it is.
    pub law: String,
    /// What evaluated it.
    pub site: FaultSite,
    /// The operation, its source range and its operands.
    pub at: NonFiniteAt,
    /// The name of the input or parameter, when that is what arrived non-finite: the
    /// value was already bad on the way in, so no operation of the law is to blame.
    pub input: Option<String>,
}

impl LawFault {
    /// True when the law was handed a non-finite value, rather than making one.
    pub fn is_input(&self) -> bool {
        self.input.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_site_describes_itself() {
        assert_eq!(FaultSite::Pair(3, 7).describe(), "pair (3, 7)");
        assert_eq!(FaultSite::Particle(4).describe(), "particle 4");
        assert_eq!(FaultSite::Cell(2, 5).describe(), "cell (2, 5)");
    }
}
