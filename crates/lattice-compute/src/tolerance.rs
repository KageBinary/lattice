//! Cross-backend agreement budgets, and where each part of one came from.
//!
//! §19.1 asks whether *"CPU and GPU agree within tolerance"*. The number that answers it
//! is the most consequential in M4, because it is the one nobody can check by inspection:
//! a tolerance loose enough to pass hides every bug smaller than itself, and there is no
//! test that fails when it is too loose.
//!
//! So a [`Tolerance`] here is not a float. It is a list of named contributions, each with
//! a mechanism that produces it and a note saying how it was derived. Its total is the sum
//! of its parts, and [`Tolerance::explain`] prints them. A budget nobody can decompose is
//! a budget that was fitted to whatever the code happened to produce.
//!
//! The CPU pair spends none of this. `lattice_cpu`'s scalar and parallel paths agree
//! bit-for-bit, so their validation cases assert equality of IEEE bit patterns and carry
//! no [`Tolerance`] at all. That was a deliberate cost paid in M4.1, and this module is
//! what it bought: every unit of budget spent below belongs to the GPU and can be pointed
//! at.

use core::fmt;

use crate::Precision;

/// A named reason why two backends do not produce identical bits.
///
/// The list is closed on purpose. A disagreement that is not one of these is not a
/// tolerance question — it is a bug, and widening a budget to cover it would be exactly
/// the failure this type exists to prevent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mechanism {
    /// The state is stored in a narrower type than the reference, so every value is
    /// rounded on the way in and every operation rounds to that width.
    ///
    /// This dominates the portable GPU budget, and not by a little: WGSL has no `f64`, so
    /// the baseline backend runs `fast32` against an `accurate64` reference. The gap is
    /// about 10⁹ times `f64::EPSILON` before a single step has run.
    StateRounding,
    /// `a*b + c` evaluated with one rounding instead of two.
    ///
    /// Permitted to GPU compilers and generally not to the reference path. Worth at most a
    /// half-ulp per contracted pair, but it is *per operation* and it is not reproducible
    /// across drivers.
    FmaContraction,
    /// A sum accumulated in a different association order.
    ///
    /// Zero for elementwise and stencil kernels, which is why the first GPU kernel is one.
    /// The moment a reduction runs on the device this term stops being zero and starts
    /// depending on the workgroup size.
    ReductionOrder,
    /// `sin`, `exp` and friends accurate to a different number of ulps.
    ///
    /// WebGPU's accuracy requirements are looser than IEEE-754's recommendations and
    /// looser than the host libm's. Zero for kernels that only add and multiply.
    Transcendental,
}

impl Mechanism {
    /// The short name used in validation output.
    pub const fn name(self) -> &'static str {
        match self {
            Mechanism::StateRounding => "state rounding",
            Mechanism::FmaContraction => "FMA contraction",
            Mechanism::ReductionOrder => "reduction order",
            Mechanism::Transcendental => "transcendental accuracy",
        }
    }
}

impl fmt::Display for Mechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One contribution to a budget.
#[derive(Clone, PartialEq, Debug)]
pub struct Term {
    /// What produces it.
    pub mechanism: Mechanism,
    /// How much relative error it is allowed to contribute.
    pub relative: f64,
    /// How that figure was arrived at. Prose, and it is meant to be read.
    pub note: String,
}

/// A relative-error budget, decomposed into the mechanisms that spend it.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Tolerance {
    terms: Vec<Term>,
}

impl Tolerance {
    /// A budget of zero: the two paths must agree exactly.
    ///
    /// What the CPU pair uses, and what any backend running the reference precision should
    /// be held to until it is shown that it cannot be.
    pub fn exact() -> Tolerance {
        Tolerance { terms: Vec::new() }
    }

    /// Add a contribution.
    pub fn plus(
        mut self,
        mechanism: Mechanism,
        relative: f64,
        note: impl Into<String>,
    ) -> Tolerance {
        self.terms.push(Term { mechanism, relative, note: note.into() });
        self
    }

    /// The budget for `steps` applications of a *contracting* elementwise or stencil
    /// kernel at `precision`.
    ///
    /// Three facts, and the number falls out of them.
    ///
    /// **Each step rounds a few times.** The diffusion stencil is four products and five
    /// sums per cell, then one multiply-add for the update. Ten roundings, each at most
    /// half an ulp, is `5·ε` of fresh relative error per step.
    ///
    /// **The scheme does not amplify what it already has.** An explicit diffusion step
    /// below its stability limit has every eigenvalue of its update operator inside the
    /// unit circle — that is what the stability limit *means* — so a perturbation present
    /// at step *k* is no larger at step *k+1*. Errors accumulate additively rather than
    /// geometrically, which is the difference between a usable bound and a meaningless
    /// one.
    ///
    /// **Additive accumulation over `n` steps is `n` times the per-step figure.** That is
    /// the worst case, in which every step's round-off happens to push the same way. Real
    /// round-off is closer to a random walk and lands near `√n`, so the measured
    /// disagreement should sit far below this bound — and if it ever approaches it, that
    /// is a signal rather than a comfort.
    ///
    /// Deliberately pessimistic, and deliberately derived rather than measured. A bound
    /// fitted to an observation cannot detect the observation getting worse.
    pub fn stepped_kernel(precision: Precision, steps: usize) -> Tolerance {
        if precision.is_reference() {
            return Tolerance::exact();
        }
        let per_step = 5.0 * precision.epsilon();
        let bound = per_step * steps.max(1) as f64;
        Tolerance::exact().plus(
            Mechanism::StateRounding,
            bound,
            format!(
                "{steps} steps x 5 roundings/step at {} epsilon ({:.2e}), accumulated \
                 additively because an explicit diffusion step below its stability limit \
                 does not amplify existing error",
                precision.name(),
                precision.epsilon()
            ),
        )
    }

    /// Allow the compiler to contract multiply-adds.
    ///
    /// `pairs` is how many `a*b + c` forms one step's arithmetic contains per output
    /// element; each saves one rounding of at most half an ulp.
    pub fn allowing_fma(self, precision: Precision, pairs: usize, steps: usize) -> Tolerance {
        let bound = 0.5 * precision.epsilon() * pairs as f64 * steps.max(1) as f64;
        self.plus(
            Mechanism::FmaContraction,
            bound,
            format!(
                "{pairs} contractable multiply-adds per element per step over {steps} steps, \
                 each saving at most half an ulp"
            ),
        )
    }

    /// Total relative error permitted.
    ///
    /// Folded from `0.0` rather than via `Sum`, whose `f64` implementation starts at
    /// `-0.0` and therefore reports an exact budget as `-0.000e0` in every message that
    /// prints one. Numerically identical, and `-0.0` in a report reads as a bug.
    pub fn relative(&self) -> f64 {
        self.terms.iter().fold(0.0, |total, term| total + term.relative)
    }

    /// True if nothing is permitted.
    pub fn is_exact(&self) -> bool {
        self.terms.is_empty()
    }

    /// The contributions, in the order they were added.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// The budget written out, one mechanism per line.
    ///
    /// §19.3 asks for the conditions behind a published number. This is that, for a
    /// tolerance rather than for a timing.
    pub fn explain(&self) -> String {
        if self.terms.is_empty() {
            return "exact: no mechanism permits any disagreement".to_string();
        }
        let mut out = format!("relative budget {:.3e}, spent on:\n", self.relative());
        for term in &self.terms {
            out.push_str(&format!(
                "  {:<24} {:.3e}  {}\n",
                term.mechanism.name(),
                term.relative,
                term.note
            ));
        }
        out
    }

    /// Compare `measured` against `reference` under this budget.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length, which is a wiring error rather than a numerical
    /// one and must not be reported as a tolerance failure.
    pub fn compare(&self, reference: &[f64], measured: &[f64]) -> Comparison {
        assert_eq!(
            reference.len(),
            measured.len(),
            "comparing {} reference values against {} measured ones",
            reference.len(),
            measured.len()
        );

        // The scale is the reference's own magnitude. A quantity that lives near zero has
        // no scale of its own and a relative error against it is meaningless — the same
        // lesson `ParticleDomain::momentum_scale` exists for — so the denominator is
        // stated in the result rather than left for a reader to assume.
        let scale = reference.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
        let denominator = if scale > 0.0 { scale } else { 1.0 };

        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        let mut non_finite = None;
        for (index, (&want, &got)) in reference.iter().zip(measured).enumerate() {
            if !got.is_finite() && non_finite.is_none() {
                non_finite = Some(index);
            }
            let deviation = (got - want).abs() / denominator;
            if deviation > worst {
                worst = deviation;
                worst_at = index;
            }
        }

        Comparison {
            worst_relative: worst,
            worst_at,
            denominator,
            budget: self.relative(),
            non_finite,
        }
    }
}

/// The result of holding two backends' output against a budget.
#[derive(Clone, PartialEq, Debug)]
pub struct Comparison {
    /// Largest relative deviation found.
    pub worst_relative: f64,
    /// Where it was found.
    pub worst_at: usize,
    /// What the relative error was taken against.
    pub denominator: f64,
    /// What was permitted.
    pub budget: f64,
    /// The first non-finite measured value, if any.
    pub non_finite: Option<usize>,
}

impl Comparison {
    /// True if the disagreement fits the budget and nothing became non-finite.
    ///
    /// A `NaN` is not a large error, it is a different kind of event, and no budget
    /// however generous should admit one.
    pub fn within_budget(&self) -> bool {
        self.non_finite.is_none() && self.worst_relative <= self.budget
    }

    /// How much of the budget was actually used, as a fraction.
    ///
    /// The number worth watching over time. A case sitting at 0.001 of its budget is
    /// telling you the bound is loose; one that climbs toward 1.0 across releases is
    /// telling you something is degrading while still passing.
    pub fn budget_used(&self) -> f64 {
        if self.budget > 0.0 {
            self.worst_relative / self.budget
        } else if self.worst_relative > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// A one-line summary for a validation report.
    pub fn summary(&self) -> String {
        if let Some(index) = self.non_finite {
            return format!("non-finite value at index {index}");
        }
        // An exact budget has no fraction to be a percentage of, and printing "inf% of the
        // 0.000e0 budget" says nothing a reader can use.
        if self.budget == 0.0 {
            return format!(
                "worst {:.3e} at index {} (relative to {:.3e}), against an exact budget",
                self.worst_relative, self.worst_at, self.denominator
            );
        }
        format!(
            "worst {:.3e} at index {} (relative to {:.3e}), {:.1}% of the {:.3e} budget",
            self.worst_relative,
            self.worst_at,
            self.denominator,
            100.0 * self.budget_used(),
            self.budget
        )
    }
}

impl fmt::Display for Tolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3e}", self.relative())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f64`'s `Sum` folds from `-0.0`, so the obvious implementation reports an exact
    /// budget as `-0.000e0` everywhere it is printed.
    #[test]
    fn an_exact_budget_is_positive_zero_and_reads_as_exact() {
        let tolerance = Tolerance::exact();
        assert!(tolerance.relative().is_sign_positive(), "an exact budget printed as -0.0");
        assert_eq!(tolerance.to_string(), "0.000e0");

        let summary = tolerance.compare(&[300.0, 300.0], &[300.0, 300.0]).summary();
        assert!(summary.contains("against an exact budget"), "{summary}");
        assert!(!summary.contains("inf"), "{summary}");
    }

    #[test]
    fn an_exact_budget_admits_nothing() {
        let tolerance = Tolerance::exact();
        assert!(tolerance.is_exact());
        assert_eq!(tolerance.relative(), 0.0);
        assert!(tolerance.compare(&[1.0, 2.0], &[1.0, 2.0]).within_budget());
        let nudged = tolerance.compare(&[1.0, 2.0], &[1.0, 2.0 + f64::EPSILON * 4.0]);
        assert!(!nudged.within_budget(), "one ulp must fail an exact budget");
    }

    #[test]
    fn the_reference_precision_needs_no_budget() {
        assert!(Tolerance::stepped_kernel(Precision::Accurate64, 1000).is_exact());
    }

    #[test]
    fn a_budget_is_the_sum_of_its_named_parts() {
        let tolerance = Tolerance::stepped_kernel(Precision::Fast32, 200)
            .allowing_fma(Precision::Fast32, 4, 200);
        assert_eq!(tolerance.terms().len(), 2);
        let total: f64 = tolerance.terms().iter().map(|t| t.relative).sum();
        assert_eq!(tolerance.relative(), total);
        assert_eq!(tolerance.terms()[0].mechanism, Mechanism::StateRounding);
    }

    /// The budget must be dominated by the mechanism that actually dominates, or the
    /// decomposition is decorative.
    #[test]
    fn state_rounding_dominates_the_portable_gpu_budget() {
        let tolerance = Tolerance::stepped_kernel(Precision::Fast32, 200)
            .allowing_fma(Precision::Fast32, 4, 200);
        let rounding = tolerance.terms()[0].relative;
        let fma = tolerance.terms()[1].relative;
        assert!(rounding > fma, "{rounding} should exceed {fma}");
    }

    #[test]
    fn every_term_carries_a_note_explaining_where_it_came_from() {
        let tolerance = Tolerance::stepped_kernel(Precision::Fast32, 200)
            .allowing_fma(Precision::Fast32, 4, 200);
        for term in tolerance.terms() {
            assert!(!term.note.trim().is_empty(), "{:?} has no derivation", term.mechanism);
        }
        assert!(tolerance.explain().contains("state rounding"));
    }

    #[test]
    fn the_budget_grows_with_the_step_count() {
        let short = Tolerance::stepped_kernel(Precision::Fast32, 10).relative();
        let long = Tolerance::stepped_kernel(Precision::Fast32, 1000).relative();
        assert!((long / short - 100.0).abs() < 1e-9, "additive accumulation is linear in steps");
    }

    #[test]
    fn comparison_reports_where_the_worst_deviation_is() {
        let tolerance = Tolerance::exact().plus(Mechanism::StateRounding, 1e-3, "test");
        let reference = [100.0, 100.0, 100.0];
        let measured = [100.0, 100.02, 100.0];
        let comparison = tolerance.compare(&reference, &measured);
        assert_eq!(comparison.worst_at, 1);
        assert_eq!(comparison.denominator, 100.0);
        assert!((comparison.worst_relative - 2e-4).abs() < 1e-12);
        assert!(comparison.within_budget());
        assert!((comparison.budget_used() - 0.2).abs() < 1e-9);
    }

    /// A `NaN` is a different kind of event from a large error, and no budget admits one.
    #[test]
    fn a_non_finite_value_fails_however_large_the_budget() {
        let tolerance = Tolerance::exact().plus(Mechanism::StateRounding, 1e9, "enormous");
        let comparison = tolerance.compare(&[1.0, 1.0], &[1.0, f64::NAN]);
        assert!(!comparison.within_budget());
        assert_eq!(comparison.non_finite, Some(1));
        assert!(comparison.summary().contains("non-finite"));
    }

    /// A field of zeros has no scale of its own; the denominator must not be zero.
    #[test]
    fn a_field_at_zero_falls_back_to_an_absolute_comparison() {
        let tolerance = Tolerance::exact().plus(Mechanism::StateRounding, 1e-6, "test");
        let comparison = tolerance.compare(&[0.0, 0.0], &[0.0, 1e-9]);
        assert_eq!(comparison.denominator, 1.0);
        assert!(comparison.within_budget());
    }
}
