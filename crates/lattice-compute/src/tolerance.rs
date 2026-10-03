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
    /// Two iterative solves stopped at different points, each having satisfied its own
    /// stated residual tolerance.
    ///
    /// Not a rounding mechanism at all, which is why it needed its own name rather than a
    /// wider [`Mechanism::StateRounding`]. Both backends returned an `x` that answers the
    /// question asked of it — `‖b − Ax‖ ≤ τ‖b‖` — and two different answers to that
    /// question are both correct. The disagreement is the *width of the question*, and it
    /// scales with `τ`, not with `ε`.
    ///
    /// It is zero for every explicit scheme, and it dominates every implicit one on a
    /// backend whose precision forces a loose `τ`. See [`ImplicitSolve`].
    SolveTermination,
}

impl Mechanism {
    /// The short name used in validation output.
    pub const fn name(self) -> &'static str {
        match self {
            Mechanism::StateRounding => "state rounding",
            Mechanism::FmaContraction => "FMA contraction",
            Mechanism::ReductionOrder => "reduction order",
            Mechanism::Transcendental => "transcendental accuracy",
            Mechanism::SolveTermination => "solve termination",
        }
    }
}

/// Which norm a budget is stated in, and therefore which norm it is checked in.
///
/// Added in M4.4, and the reason is worth stating because it is easy to get wrong in a way
/// that looks like rigour. Every mechanism up to that milestone bounded the error *per
/// cell*: state rounding, FMA contraction and transcendental accuracy are all statements of
/// the form "each value is off by at most this fraction of itself". A per-cell bound is
/// checked in the max norm, and [`Norm::Max`] is what [`Tolerance::exact`] still gives.
///
/// [`Mechanism::SolveTermination`] is not like that. Conjugate gradient stops on `‖r‖₂`, so
/// everything derivable from its stopping rule is a statement about a *vector*, and the
/// error it permits may sit anywhere in that vector. Converting such a bound into a
/// per-cell one costs a factor of `√N` — 157 on a 24 576-cell grid — which is real but
/// belongs entirely to the change of norm and not to the solver. A budget carrying it would
/// be two orders of magnitude looser than the mechanism it describes, and a budget two
/// orders too loose hides everything smaller.
///
/// So the budget names the norm it was derived in and the comparison is made in the same
/// one. Mixing is the thing being prevented.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Norm {
    /// `max|Δ| / max|reference|`. The norm a per-cell bound is stated in.
    #[default]
    Max,
    /// `‖Δ‖₂ / ‖reference‖₂`. The norm an iterative solver's stopping rule is stated in.
    Euclidean,
}

impl Norm {
    /// The short name used in validation output.
    pub const fn name(self) -> &'static str {
        match self {
            Norm::Max => "max",
            Norm::Euclidean => "euclidean",
        }
    }

    /// Relative deviation of `measured` from `reference`, plus the index that contributed
    /// most to it and the denominator used.
    fn measure(self, reference: &[f64], measured: &[f64]) -> (f64, usize, f64) {
        let mut worst_at = 0;
        let mut worst_gap = 0.0f64;
        for (index, (&want, &got)) in reference.iter().zip(measured).enumerate() {
            let gap = (got - want).abs();
            if gap > worst_gap {
                worst_gap = gap;
                worst_at = index;
            }
        }

        // The scale is the reference's own magnitude. A quantity that lives near zero has
        // no scale of its own and a relative error against it is meaningless — the same
        // lesson `ParticleDomain::momentum_scale` exists for — so the denominator is
        // stated in the result rather than left for a reader to assume.
        let (deviation, scale) = match self {
            Norm::Max => {
                (worst_gap, reference.iter().fold(0.0f64, |acc, value| acc.max(value.abs())))
            }
            Norm::Euclidean => {
                let gaps = reference
                    .iter()
                    .zip(measured)
                    .fold(0.0f64, |acc, (&want, &got)| acc + (got - want) * (got - want));
                let scale = reference.iter().fold(0.0f64, |acc, &value| acc + value * value);
                (gaps.sqrt(), scale.sqrt())
            }
        };
        let denominator = if scale > 0.0 { scale } else { 1.0 };
        (deviation / denominator, worst_at, denominator)
    }
}

impl fmt::Display for Norm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
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
    norm: Norm,
}

impl Tolerance {
    /// A budget of zero: the two paths must agree exactly.
    ///
    /// What the CPU pair uses, and what any backend running the reference precision should
    /// be held to until it is shown that it cannot be.
    ///
    /// Exactness does not depend on the norm — a vector is zero in one exactly when it is
    /// zero in the other — so this is [`Norm::Max`] and any budget built on it stays there
    /// until [`Tolerance::in_norm`] says otherwise.
    pub fn exact() -> Tolerance {
        Tolerance { terms: Vec::new(), norm: Norm::Max }
    }

    /// State which norm every term in this budget was derived in.
    ///
    /// [`Tolerance::compare`] then measures in the same one. See [`Norm`] for why that has
    /// to be a choice rather than a default.
    pub fn in_norm(mut self, norm: Norm) -> Tolerance {
        self.norm = norm;
        self
    }

    /// The norm this budget is stated and checked in.
    pub fn norm(&self) -> Norm {
        self.norm
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
        let mut out =
            format!("relative budget {:.3e} in the {} norm, spent on:\n", self.relative(), self.norm);
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

        // A non-finite measured value is found before any arithmetic is done on it, because
        // in the Euclidean norm one NaN makes the whole sum NaN and the report would lose
        // the index that caused it.
        let non_finite = measured.iter().position(|value| !value.is_finite());

        let (worst_relative, worst_at, denominator) = match non_finite {
            Some(_) => (f64::INFINITY, 0, 1.0),
            None => self.norm.measure(reference, measured),
        };

        Comparison {
            worst_relative,
            worst_at,
            denominator,
            budget: self.relative(),
            norm: self.norm,
            non_finite,
        }
    }
}

/// Everything needed to derive the budget for `steps` steps of an implicit scheme solved
/// iteratively on both backends.
///
/// This is a struct rather than seven arguments because every field is a number somebody
/// has to be able to check, and a positional call would let two of them be swapped without
/// anything noticing.
///
/// # The derivation
///
/// One observation carries the whole thing. Everything that goes wrong in a conjugate
/// gradient solve — `f32` state, a reduction in a different order, a stopping test that
/// fires half an iteration early — reaches the answer through exactly one channel: the
/// residual `b − Ax` that the iteration is checking. And for `A = I − θ·dt·L` with `L`
/// negative semi-definite, every eigenvalue of `A` is at least 1, so
///
/// ```text
/// ‖A⁻¹‖₂ ≤ 1
/// ```
///
/// which turns a bound on the residual into a bound on the solution error *with no
/// amplification at all*. That is unusually generous — the same derivation for a general
/// system would carry `κ(A)` — and it is a property of the heat operator rather than of
/// this code.
///
/// So the per-step budget is a sum of three residual-level terms, each divided by `‖x‖₂` to
/// make it relative:
///
/// | Mechanism | Residual it permits | Why |
/// |---|---|---|
/// | [`Mechanism::SolveTermination`] | `(τ_ref + τ_measured)·‖b‖₂` | each solve stopped when its own criterion was met, and they are different criteria |
/// | [`Mechanism::StateRounding`] | `ε·(1 + ‖A‖₂)·‖b‖₂` | the *true* residual cannot be formed more accurately than this, whatever the iteration thinks it reached |
/// | [`Mechanism::ReductionOrder`] | `τ_measured·(d·ε/2)·‖b‖₂` | the stopping test is a tree-summed `‖r‖²`, so it may fire at a true residual this fraction above the target |
///
/// and the total is multiplied by `steps`, because Crank–Nicolson's amplification factor
/// `(1 + (1−θ)dt·λ)/(1 − θ·dt·λ)` has magnitude at most 1 for `λ ≤ 0` and `θ ≥ ½` — the
/// scheme does not amplify error it already has, so accumulation is additive rather than
/// geometric. The same argument [`Tolerance::stepped_kernel`] makes for the explicit scheme
/// below its stability limit.
///
/// # What the numbers turn out to be, which was the surprise
///
/// `ReductionOrder` was the term M4.2 deferred Crank–Nicolson *for*, and it lands about six
/// orders of magnitude below `SolveTermination`. The reason is structural rather than
/// lucky: a perturbed dot product changes which iterate CG arrives at, and the stopping test
/// then measures that iterate afresh. The reduction moves the path, not the destination.
///
/// This is emphatically **not** a general result about reductions. It holds because CG's
/// termination re-measures what the reduction perturbed. A reduction whose value *is* the
/// answer — a conserved total, an observation, a fixed-iteration solver — has no such
/// backstop, and the term is then the whole disagreement. `gpu_reduction_order_is_the_whole
/// _disagreement_when_nothing_re_measures_it` in the validation suite is that case, and it
/// exists so this term is audited somewhere it can be seen.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ImplicitSolve {
    /// What the measured backend stores state in.
    pub precision: Precision,
    /// Steps taken.
    pub steps: usize,
    /// Relative residual tolerance the *reference* solve stopped at.
    pub reference_tolerance: f64,
    /// Relative residual tolerance the *measured* solve stopped at. On a `fast32` backend
    /// this cannot go below `ε·(1 + ‖A‖₂)`, and it is normally what dominates.
    pub measured_tolerance: f64,
    /// An upper bound on `‖A‖₂`, from Gershgorin over the assembled rows.
    pub operator_norm: f64,
    /// Roundings one term passes through inside the measured backend's dot product.
    pub reduction_depth: usize,
    /// `‖b‖₂ / ‖x‖₂`, measured on the reference run.
    ///
    /// Never below 1, because `‖A⁻¹‖₂ ≤ 1` means `‖x‖₂ ≤ ‖b‖₂`. It is a property of the
    /// problem — how far the right-hand side is from the solution — and taking it from the
    /// reference run is measurement of an *input*, not fitting to the disagreement.
    pub rhs_amplification: f64,
}

impl ImplicitSolve {
    /// The budget, in the Euclidean norm the derivation is stated in.
    ///
    /// Returns [`Tolerance::exact`] at the reference precision with matching tolerances,
    /// because two identical solves of the same system have nothing to disagree about.
    pub fn budget(&self) -> Tolerance {
        let epsilon = self.precision.epsilon();
        let steps = self.steps.max(1) as f64;
        let scale = steps * self.rhs_amplification;

        let termination = scale * (self.reference_tolerance + self.measured_tolerance);
        let reduction =
            scale * self.measured_tolerance * (self.reduction_depth as f64 * epsilon / 2.0);

        // Zero at the reference precision, for the reason `stepped_kernel` gives and by the
        // rule M4.1 established: a backend running the reference precision is held to bit
        // equality until it is shown that it cannot be. The residual floor is real at `f64`
        // too — two *different* `f64` implementations would find it — but claiming it here
        // would hand a reference-precision backend a budget it has not been shown to need,
        // which is the direction this module exists to refuse.
        let rounding = if self.precision.is_reference() {
            0.0
        } else {
            scale * epsilon * (1.0 + self.operator_norm)
        };

        if termination == 0.0 && rounding == 0.0 && reduction == 0.0 {
            return Tolerance::exact();
        }

        Tolerance::exact()
            .in_norm(Norm::Euclidean)
            .plus(
                Mechanism::SolveTermination,
                termination,
                format!(
                    "{} steps x (reference {:.2e} + measured {:.2e}) relative residual, \
                     amplified by ||b||/||x|| = {:.3}; each solve returned an x satisfying \
                     its own criterion and ||A^-1|| <= 1 carries that to the solution \
                     unamplified",
                    self.steps, self.reference_tolerance, self.measured_tolerance,
                    self.rhs_amplification
                ),
            )
            .plus(
                Mechanism::StateRounding,
                rounding,
                format!(
                    "{} steps x {} epsilon ({:.2e}) x (1 + ||A|| <= {:.3}), amplified by \
                     {:.3}: the floor on how accurately the true residual b - Ax can be \
                     formed at all in this precision",
                    self.steps, self.precision.name(), epsilon, self.operator_norm,
                    self.rhs_amplification
                ),
            )
            .plus(
                Mechanism::ReductionOrder,
                reduction,
                format!(
                    "{} steps x measured tolerance {:.2e} x depth {} x epsilon/2: the \
                     stopping test sums ||r||^2 through a tree {} roundings deep, so it may \
                     fire at a true residual this fraction above the target",
                    self.steps, self.measured_tolerance, self.reduction_depth,
                    self.reduction_depth
                ),
            )
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
    /// Which norm both the budget and the deviation are stated in.
    pub norm: Norm,
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
                "{} deviation {:.3e}, worst cell {} (relative to {:.3e}), against an exact budget",
                self.norm, self.worst_relative, self.worst_at, self.denominator
            );
        }
        format!(
            "{} deviation {:.3e}, worst cell {} (relative to {:.3e}), {:.1}% of the {:.3e} budget",
            self.norm,
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

    fn implicit() -> ImplicitSolve {
        ImplicitSolve {
            precision: Precision::Fast32,
            steps: 50,
            reference_tolerance: 1e-10,
            measured_tolerance: 5e-6,
            operator_norm: 21.0,
            reduction_depth: 16,
            rhs_amplification: 1.0,
        }
    }

    /// The result that inverts M4.2's expectation, asserted rather than remembered.
    ///
    /// Reduction order is the mechanism Crank–Nicolson was deferred for, and inside a
    /// residual-checked iteration it is the *smallest* term by orders of magnitude.
    #[test]
    fn solve_termination_dominates_an_implicit_budget_and_reduction_order_does_not() {
        let budget = implicit().budget();
        let of = |mechanism: Mechanism| {
            budget.terms().iter().find(|t| t.mechanism == mechanism).expect("term").relative
        };

        let termination = of(Mechanism::SolveTermination);
        let rounding = of(Mechanism::StateRounding);
        let reduction = of(Mechanism::ReductionOrder);

        assert!(termination > rounding, "{termination:.3e} should exceed {rounding:.3e}");
        assert!(
            reduction < termination / 1e5,
            "reduction order is {reduction:.3e} against termination's {termination:.3e}; if \
             these have come within five orders of each other the claim in ImplicitSolve's \
             documentation needs rewriting, not the assertion loosening"
        );
    }

    /// The budget is checked in the norm it was derived in, and that has to be the
    /// Euclidean one or the `√N` from the change of norm is silently missing.
    #[test]
    fn an_implicit_budget_is_stated_and_checked_in_the_euclidean_norm() {
        let budget = implicit().budget();
        assert_eq!(budget.norm(), Norm::Euclidean);
        assert!(budget.explain().contains("euclidean"), "{}", budget.explain());

        // One cell off by δ in a field of N: the max norm sees δ, the Euclidean norm sees
        // δ/√N. They are different measurements and the report must say which it made.
        let reference = vec![100.0; 400];
        let mut measured = reference.clone();
        measured[7] += 1.0;

        let euclidean = budget.compare(&reference, &measured);
        let max = implicit().budget().in_norm(Norm::Max).compare(&reference, &measured);
        assert_eq!(euclidean.worst_at, 7, "the worst cell is still reported");
        assert_eq!(max.worst_at, 7);
        assert!(
            (max.worst_relative / euclidean.worst_relative - 20.0).abs() < 1e-9,
            "sqrt(400) between the two norms, got {:.3} ",
            max.worst_relative / euclidean.worst_relative
        );
        assert!(euclidean.summary().contains("euclidean"), "{}", euclidean.summary());
    }

    /// M4.1's rule, applied to the implicit budget: a backend running the reference
    /// precision is held to bit equality until it is shown that it cannot be. Two solves
    /// that also stopped at the same criterion have nothing left to disagree about.
    #[test]
    fn an_implicit_budget_at_the_reference_precision_with_matched_tolerances_is_exact() {
        let matched = ImplicitSolve {
            precision: Precision::Accurate64,
            reference_tolerance: 0.0,
            measured_tolerance: 0.0,
            ..implicit()
        };
        assert!(matched.budget().is_exact(), "{}", matched.budget().explain());

        // A reference-precision backend that stops somewhere else still disagrees, and by
        // termination alone.
        let mismatched =
            ImplicitSolve { measured_tolerance: 1e-12, ..matched };
        let budget = mismatched.budget();
        assert!(!budget.is_exact());
        assert_eq!(budget.terms()[0].mechanism, Mechanism::SolveTermination);
        assert_eq!(
            budget.terms().iter().find(|t| t.mechanism == Mechanism::StateRounding).unwrap().relative,
            0.0,
            "nothing narrows the state when the measured backend is at the reference precision"
        );
    }

    /// Every term scales with the step count, for the same additive-accumulation reason the
    /// explicit budget does. A budget that did not would be claiming the scheme forgets.
    #[test]
    fn an_implicit_budget_grows_linearly_with_the_step_count() {
        let short = ImplicitSolve { steps: 10, ..implicit() }.budget().relative();
        let long = ImplicitSolve { steps: 1000, ..implicit() }.budget().relative();
        assert!((long / short - 100.0).abs() < 1e-9, "{long:.3e} over {short:.3e}");
    }

    /// A tighter measured tolerance must buy a tighter budget, or the budget is not
    /// describing the solver.
    #[test]
    fn an_implicit_budget_tightens_when_the_measured_solve_does() {
        let loose = ImplicitSolve { measured_tolerance: 1e-4, ..implicit() }.budget().relative();
        let tight = ImplicitSolve { measured_tolerance: 1e-6, ..implicit() }.budget().relative();
        assert!(tight < loose, "{tight:.3e} should be below {loose:.3e}");
    }

    /// Every mechanism has to be printable and derivable, including the new one.
    #[test]
    fn every_implicit_term_names_a_mechanism_and_explains_itself() {
        let budget = implicit().budget();
        assert_eq!(budget.terms().len(), 3);
        for term in budget.terms() {
            assert!(!term.note.trim().is_empty(), "{:?} has no derivation", term.mechanism);
            assert!(term.relative >= 0.0, "{:?} contributed a negative budget", term.mechanism);
        }
        let explained = budget.explain();
        assert!(explained.contains("solve termination"), "{explained}");
        assert!(explained.contains(Mechanism::SolveTermination.name()));
    }

    /// One NaN in the Euclidean norm makes the whole sum NaN, which would lose the index
    /// that caused it. The non-finite check has to come first, and it has to survive the
    /// change of norm.
    #[test]
    fn a_non_finite_value_is_found_before_the_norm_is_taken() {
        let budget = Tolerance::exact()
            .in_norm(Norm::Euclidean)
            .plus(Mechanism::SolveTermination, 1e9, "enormous");
        let comparison = budget.compare(&[1.0, 1.0, 1.0], &[1.0, f64::NAN, 1.0]);
        assert_eq!(comparison.non_finite, Some(1));
        assert!(!comparison.within_budget());
        assert!(comparison.summary().contains("non-finite"), "{}", comparison.summary());
    }
}
