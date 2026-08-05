//! The solver contract every domain must publish, and the trait every domain must
//! implement.
//!
//! Two spec requirements meet here:
//!
//! - **P1 — scientific honesty over feature count**: *"Every module must state
//!   governing equations, discretization, assumptions, valid regimes, error metrics,
//!   and known non-conservation."*
//! - **NFR-003**: *"Every solver publishes assumptions, supported precision,
//!   stability guidance, and validation cases"* — as *runtime metadata*, not only
//!   prose in a manual.
//!
//! A [`SolverContract`] is that metadata, in a form the CLI can print, the viewer can
//! display next to a running simulation, and a run artifact can embed. If a solver
//! cannot state what it assumes and where it is invalid, it is not ready to ship —
//! and because the contract is a required associated value, the compiler enforces
//! that a solver at least *has* one.
//!
//! The other half of §7.3 is [`Domain`]: the small set of operations the runtime
//! needs in order to schedule a solver it knows nothing else about.

use core::fmt;

use crate::arena::Arena;
use crate::diagnostics::{Invariant, Observations};

/// A named accuracy/intent level (spec §5.1).
///
/// The point of naming these is that a user can see which one is active. "Fast
/// approximations, educational models, engineering models, and external high-fidelity
/// calculations must be named and selectable" — never silently swapped (P5).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum FidelityProfile {
    /// F0 — immediate visual exploration and teaching.
    Interactive,
    /// F1 — quantitative small-scale 2D models with published error metrics.
    Engineering2d,
    /// F2 — particle and stochastic molecular behaviour.
    MolecularKinetic,
    /// F3 — inverse problems and complex coupled workflows.
    ResearchCoupling,
    /// F4 — results delegated to an external electronic-structure backend.
    ExternalQuantum,
}

impl FidelityProfile {
    /// The short code used in project files and reports.
    pub const fn code(self) -> &'static str {
        match self {
            FidelityProfile::Interactive => "F0",
            FidelityProfile::Engineering2d => "F1",
            FidelityProfile::MolecularKinetic => "F2",
            FidelityProfile::ResearchCoupling => "F3",
            FidelityProfile::ExternalQuantum => "F4",
        }
    }

    /// What this profile is for.
    pub const fn purpose(self) -> &'static str {
        match self {
            FidelityProfile::Interactive => "immediate visual exploration and teaching",
            FidelityProfile::Engineering2d => "quantitative small-scale models with validation",
            FidelityProfile::MolecularKinetic => "particle and stochastic molecular behaviour",
            FidelityProfile::ResearchCoupling => "inverse problems and complex coupled workflows",
            FidelityProfile::ExternalQuantum => "electronic structure and reaction-path calculations",
        }
    }

    /// What this profile does — and does not — promise.
    pub const fn guarantee(self) -> &'static str {
        match self {
            FidelityProfile::Interactive => "stable and responsive; not automatically predictive",
            FidelityProfile::Engineering2d => "published benchmark error and conservation metrics",
            FidelityProfile::MolecularKinetic => {
                "correct implementation of the declared model, not ab initio chemistry"
            }
            FidelityProfile::ResearchCoupling => {
                "model-specific validation and explicit convergence controls"
            }
            FidelityProfile::ExternalQuantum => {
                "results inherit the assumptions and validation of the selected backend"
            }
        }
    }
}

impl fmt::Display for FidelityProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.code(), self.purpose())
    }
}

/// A supported numeric mode (spec §10.1).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Precision {
    /// 32-bit state, for interactive GPU scenes.
    Fast32,
    /// 32-bit state with 64-bit reductions.
    Mixed,
    /// 64-bit throughout.
    Accurate64,
    /// 64-bit with restricted operation ordering, for bit-reproducible replay.
    Deterministic64,
    /// 64-bit complex, for wavefunctions.
    Complex64,
}

impl Precision {
    /// Name used in reports and project files.
    pub const fn code(self) -> &'static str {
        match self {
            Precision::Fast32 => "fast32",
            Precision::Mixed => "mixed",
            Precision::Accurate64 => "accurate64",
            Precision::Deterministic64 => "deterministic64",
            Precision::Complex64 => "complex64",
        }
    }
}

/// Why a solver's timestep is limited.
///
/// Reason codes exist so the viewer can answer "why is my timestep so small?" with
/// the responsible mechanism rather than a bare number (spec §7.3, §17.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StabilityReason {
    /// Stable for any timestep — an implicit or unconditionally stable scheme.
    ///
    /// Note that unconditional *stability* is not unconditional *accuracy*: a large
    /// step on an unconditionally stable scheme produces a smooth, plausible, wrong
    /// answer. The preferred step still reflects an accuracy target.
    Unconditional,
    /// Explicit diffusion limit, `dt ≤ 1 / (2D(1/dx² + 1/dy²))`.
    DiffusionExplicit,
    /// Advective CFL limit, `dt ≤ min(dx/|u|, dy/|v|)`.
    AdvectionCfl,
    /// The fastest oscillation in the system must be resolved.
    OscillationPeriod,
    /// A particle must not cross more than the neighbour-list skin per step.
    NeighborSkin,
    /// Fixed by the fidelity profile or the user, not by the numerics.
    Configured,
}

impl StabilityReason {
    /// A one-line explanation for diagnostics.
    pub const fn describe(self) -> &'static str {
        match self {
            StabilityReason::Unconditional => "unconditionally stable; step set by accuracy target",
            StabilityReason::DiffusionExplicit => "explicit diffusion stencil (von Neumann limit)",
            StabilityReason::AdvectionCfl => "advective CFL condition",
            StabilityReason::OscillationPeriod => "fastest oscillation period must be resolved",
            StabilityReason::NeighborSkin => "particles must not outrun the neighbour-list skin",
            StabilityReason::Configured => "fixed by configuration, not by numerics",
        }
    }
}

/// A domain's timestep advice.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StableStep {
    /// The step this domain would choose, balancing accuracy and cost.
    pub preferred: f64,
    /// The largest step that remains stable. `f64::INFINITY` for implicit schemes.
    pub max: f64,
    /// Why `max` is what it is.
    pub reason: StabilityReason,
}

impl StableStep {
    /// A domain with no stability constraint of its own.
    pub const fn unconditional(preferred: f64) -> Self {
        Self { preferred, max: f64::INFINITY, reason: StabilityReason::Unconditional }
    }

    /// An explicitly limited step.
    pub const fn limited(preferred: f64, max: f64, reason: StabilityReason) -> Self {
        Self { preferred, max, reason }
    }

    /// How much headroom a proposed step has before instability, as a ratio.
    ///
    /// Values above 1 are unstable. The viewer plots this so a user can see a run
    /// approaching its limit before it blows up rather than after (NFR-007).
    pub fn margin(&self, dt: f64) -> f64 {
        if self.max.is_infinite() { 0.0 } else { dt / self.max }
    }

    /// True if `dt` is within the stability limit.
    pub fn admits(&self, dt: f64) -> bool {
        dt <= self.max
    }

    /// The tighter of two constraints, for combining domains.
    pub fn tightest(self, other: StableStep) -> StableStep {
        if other.max < self.max {
            StableStep {
                preferred: self.preferred.min(other.preferred),
                max: other.max,
                reason: other.reason,
            }
        } else {
            StableStep {
                preferred: self.preferred.min(other.preferred),
                max: self.max,
                reason: self.reason,
            }
        }
    }
}

/// Everything a solver must publish about itself.
///
/// All fields are `&'static` so a contract is a compile-time constant with no runtime
/// cost. Writing one is meant to be slightly uncomfortable: if the `assumptions` list
/// is empty, either the solver is trivial or its author has not thought about it.
#[derive(Clone, Copy, Debug)]
pub struct SolverContract {
    /// Short identifier, e.g. `"grid2d.heat"`.
    pub name: &'static str,
    /// One sentence on what this solver does.
    pub summary: &'static str,
    /// The continuous equations being solved, in readable notation.
    pub governing_equations: &'static [&'static str],
    /// How those equations are discretized in space.
    pub discretization: &'static str,
    /// How the discrete system is advanced in time.
    pub integrator: &'static str,
    /// What must be true for results to mean anything.
    pub assumptions: &'static [&'static str],
    /// The regime where this solver is appropriate.
    pub valid_regime: &'static str,
    /// Human-readable statement of the stability criterion.
    pub stability: &'static str,
    /// Quantities this solver conserves exactly or to a stated tolerance.
    pub conserves: &'static [Invariant],
    /// Quantities this solver is known *not* to conserve, and why.
    ///
    /// This field is the one that makes the contract honest. A solver that lists
    /// nothing here is claiming exact conservation of everything else it touches.
    pub known_non_conservation: &'static [&'static str],
    /// The fidelity level this solver belongs to.
    pub fidelity: FidelityProfile,
    /// Numeric modes this implementation supports.
    pub precisions: &'static [Precision],
    /// Whether repeated runs with the same seed produce identical results.
    pub deterministic: bool,
    /// Whether gradients can be taken through this solver.
    pub differentiable: bool,
    /// Named reference cases that validate this solver (spec §19.2).
    pub validation_cases: &'static [&'static str],
    /// Textbook or paper references for the method.
    pub references: &'static [&'static str],
}

impl SolverContract {
    /// Render the contract as readable text, for `lattice inspect` and run artifacts.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{} — {}\n", self.name, self.summary));
        out.push_str(&format!("  fidelity:       {}\n", self.fidelity));
        out.push_str(&format!("  guarantee:      {}\n", self.fidelity.guarantee()));
        out.push_str("  equations:\n");
        for eq in self.governing_equations {
            out.push_str(&format!("      {eq}\n"));
        }
        out.push_str(&format!("  discretization: {}\n", self.discretization));
        out.push_str(&format!("  integrator:     {}\n", self.integrator));
        out.push_str(&format!("  valid regime:   {}\n", self.valid_regime));
        out.push_str(&format!("  stability:      {}\n", self.stability));
        out.push_str("  assumptions:\n");
        for a in self.assumptions {
            out.push_str(&format!("      - {a}\n"));
        }
        if self.conserves.is_empty() {
            out.push_str("  conserves:      (none declared)\n");
        } else {
            let names: Vec<&str> = self.conserves.iter().map(|i| i.name()).collect();
            out.push_str(&format!("  conserves:      {}\n", names.join(", ")));
        }
        out.push_str("  does NOT conserve:\n");
        if self.known_non_conservation.is_empty() {
            out.push_str("      (nothing declared — this solver claims exact conservation)\n");
        } else {
            for n in self.known_non_conservation {
                out.push_str(&format!("      - {n}\n"));
            }
        }
        let precisions: Vec<&str> = self.precisions.iter().map(|p| p.code()).collect();
        out.push_str(&format!("  precision:      {}\n", precisions.join(", ")));
        out.push_str(&format!(
            "  deterministic:  {}   differentiable: {}\n",
            self.deterministic, self.differentiable
        ));
        out.push_str("  validated by:\n");
        for c in self.validation_cases {
            out.push_str(&format!("      - {c}\n"));
        }
        if !self.references.is_empty() {
            out.push_str("  references:\n");
            for r in self.references {
                out.push_str(&format!("      - {r}\n"));
            }
        }
        out
    }

    /// Check that a contract is actually filled in.
    ///
    /// Spec §21.1: *"A milestone is not complete because a demo looks compelling."*
    /// The validation suite runs this against every registered solver so an
    /// under-documented module fails CI rather than shipping.
    pub fn audit(&self) -> Vec<ContractGap> {
        let mut gaps = Vec::new();
        if self.governing_equations.is_empty() {
            gaps.push(ContractGap::NoEquations);
        }
        if self.assumptions.is_empty() {
            gaps.push(ContractGap::NoAssumptions);
        }
        if self.validation_cases.is_empty() {
            gaps.push(ContractGap::NoValidationCases);
        }
        if self.precisions.is_empty() {
            gaps.push(ContractGap::NoPrecisions);
        }
        if self.stability.is_empty() {
            gaps.push(ContractGap::NoStabilityStatement);
        }
        gaps
    }
}

/// A missing piece of a solver contract.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContractGap {
    /// No governing equations stated.
    NoEquations,
    /// No assumptions stated.
    NoAssumptions,
    /// No validation cases named.
    NoValidationCases,
    /// No supported precisions listed.
    NoPrecisions,
    /// No stability criterion stated.
    NoStabilityStatement,
}

impl fmt::Display for ContractGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            ContractGap::NoEquations => "no governing equations declared",
            ContractGap::NoAssumptions => "no assumptions declared",
            ContractGap::NoValidationCases => "no validation cases declared",
            ContractGap::NoPrecisions => "no supported precisions declared",
            ContractGap::NoStabilityStatement => "no stability criterion declared",
        };
        f.write_str(text)
    }
}

/// Per-step context handed to a domain.
///
/// Carries the clock and the scratch arena. Everything else a solver needs lives in
/// the solver itself: the runtime deliberately does not hand out a god-object view of
/// global state, because that is how implicit cross-domain coupling creeps in (P2 —
/// coupling is declared through typed ports, not hidden access).
#[derive(Debug)]
pub struct StepContext<'a> {
    /// Simulation time at the start of this step, seconds.
    pub time: f64,
    /// Step counter since the run started.
    pub step: u64,
    /// Preallocated per-step scratch space.
    pub arena: &'a mut Arena,
}

impl<'a> StepContext<'a> {
    /// A context at time zero.
    pub fn new(arena: &'a mut Arena) -> Self {
        Self { time: 0.0, step: 0, arena }
    }
}

/// The operations the runtime needs from a solver in order to schedule it.
///
/// Deliberately small. Spec P2: *"the runtime unifies state, units, coupling,
/// scheduling, diagnostics, and authoring — not the equations themselves."* A domain
/// keeps its own state in whatever layout suits its mathematics; this trait only
/// governs how it is driven and interrogated.
pub trait Domain {
    /// Instance name, as written in the model.
    fn name(&self) -> &str;

    /// This solver's published contract.
    fn contract(&self) -> &'static SolverContract;

    /// The timestep this domain wants and the largest it can survive.
    ///
    /// Called before each step because the limit generally depends on current state
    /// (fastest particle, largest diffusivity).
    fn stable_step(&self) -> StableStep;

    /// Update indices, boundary data, coefficients, and caches before stepping.
    ///
    /// Split out from [`Domain::advance`] so the scheduler can overlap preparation
    /// with other domains' compute (spec §9.2).
    fn prepare(&mut self, ctx: &mut StepContext<'_>);

    /// Advance this domain by `dt` seconds.
    fn advance(&mut self, dt: f64, ctx: &mut StepContext<'_>);

    /// Report metrics, invariants and residuals for this step.
    fn observe(&self, out: &mut Observations);
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPLETE: SolverContract = SolverContract {
        name: "test.solver",
        summary: "a solver used only by these tests",
        governing_equations: &["du/dt = 0"],
        discretization: "none",
        integrator: "none",
        assumptions: &["nothing happens"],
        valid_regime: "always",
        stability: "unconditional",
        conserves: &[Invariant::Energy],
        known_non_conservation: &[],
        fidelity: FidelityProfile::Interactive,
        precisions: &[Precision::Accurate64],
        deterministic: true,
        differentiable: false,
        validation_cases: &["trivial"],
        references: &[],
    };

    #[test]
    fn a_complete_contract_has_no_gaps() {
        assert!(COMPLETE.audit().is_empty());
    }

    #[test]
    fn audit_finds_every_missing_section() {
        let empty = SolverContract {
            governing_equations: &[],
            assumptions: &[],
            validation_cases: &[],
            precisions: &[],
            stability: "",
            ..COMPLETE
        };
        let gaps = empty.audit();
        assert!(gaps.contains(&ContractGap::NoEquations));
        assert!(gaps.contains(&ContractGap::NoAssumptions));
        assert!(gaps.contains(&ContractGap::NoValidationCases));
        assert!(gaps.contains(&ContractGap::NoPrecisions));
        assert!(gaps.contains(&ContractGap::NoStabilityStatement));
        assert_eq!(gaps.len(), 5);
    }

    /// A contract that declares no non-conservation is making a strong claim, and the
    /// report must say so rather than printing an empty list that reads as "n/a".
    #[test]
    fn report_calls_out_an_empty_non_conservation_list() {
        let text = COMPLETE.report();
        assert!(text.contains("claims exact conservation"), "{text}");
    }

    #[test]
    fn report_includes_the_fidelity_guarantee() {
        let text = COMPLETE.report();
        assert!(text.contains("F0"), "{text}");
        assert!(text.contains("not automatically predictive"), "{text}");
    }

    #[test]
    fn stability_margin_flags_an_over_large_step() {
        let s = StableStep::limited(0.001, 0.002, StabilityReason::DiffusionExplicit);
        assert!(s.admits(0.002));
        assert!(!s.admits(0.0021));
        assert!((s.margin(0.001) - 0.5).abs() < 1e-12);
        assert!(s.margin(0.004) > 1.0);
    }

    #[test]
    fn unconditional_steps_have_no_margin_pressure() {
        let s = StableStep::unconditional(0.01);
        assert!(s.admits(1e9));
        assert_eq!(s.margin(1e9), 0.0);
    }

    #[test]
    fn tightest_keeps_the_binding_constraint_and_its_reason() {
        let diffusion = StableStep::limited(0.01, 0.02, StabilityReason::DiffusionExplicit);
        let particles = StableStep::limited(0.005, 0.008, StabilityReason::NeighborSkin);
        let combined = diffusion.tightest(particles);
        assert_eq!(combined.max, 0.008);
        assert_eq!(combined.reason, StabilityReason::NeighborSkin);
        assert_eq!(combined.preferred, 0.005);

        // Order must not matter.
        let other_way = particles.tightest(diffusion);
        assert_eq!(other_way.max, combined.max);
        assert_eq!(other_way.reason, combined.reason);
    }

    #[test]
    fn tightest_against_unconditional_keeps_the_real_limit() {
        let implicit = StableStep::unconditional(0.05);
        let explicit = StableStep::limited(0.01, 0.02, StabilityReason::DiffusionExplicit);
        let combined = implicit.tightest(explicit);
        assert_eq!(combined.max, 0.02);
        assert_eq!(combined.reason, StabilityReason::DiffusionExplicit);
        assert_eq!(combined.preferred, 0.01);
    }

    #[test]
    fn fidelity_profiles_are_ordered_by_rigor() {
        assert!(FidelityProfile::Interactive < FidelityProfile::Engineering2d);
        assert!(FidelityProfile::Engineering2d < FidelityProfile::ExternalQuantum);
    }
}
