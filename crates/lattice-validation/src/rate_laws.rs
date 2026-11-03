//! Validation cases for rate laws (spec §8.3's reaction form, M6.1c).
//!
//! A `rate:` that reads concentrations is compiled to a program and evaluated in place
//! of mass action. The cases hold it against what it should reproduce: the built-in
//! mass action it imitates, an analytic solution of a law mass action cannot express,
//! and the heat its reaction declares. The models are compiled from the language, as a
//! user's would be, so the parser, checker, lowering and domain are all in the loop.

use lattice_compiler::chemistry::rate_law_from_source;
use lattice_compiler::compile_source;
use lattice_domain_chemistry::{Kinetics, Reaction, ReactionNetwork, Species, Term};
use lattice_ir::{Arena, Domain, Observations, RenderChannel, StepContext};
use lattice_syntax::SourceFile;

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "rate_law_reproduces_mass_action_bitwise",
        domain: "laws",
        level: Level::CrossScheme,
        claim: "a rate law written as Arrhenius mass action gives the built-in's reaction-diffusion run to the bit",
        run: mass_action_bitwise,
    },
    Case {
        name: "michaelis_menten_matches_its_closed_form",
        domain: "laws",
        level: Level::Analytic,
        claim: "a Michaelis-Menten rate law integrates to the implicit closed form K ln(S/S0) + S - S0 = -V t within the integrator's own error",
        run: michaelis_menten,
    },
    Case {
        name: "heat_release_is_the_extent_times_the_heat",
        domain: "laws",
        level: Level::Property,
        claim: "spec 8.3's neutralization, compiled as written, publishes exactly heat_release per mole of extent, to the rounding of its sums",
        run: heat_release,
    },
];

/// Compile a model and return its solvers, or panic with the diagnostics: a validation
/// model that does not compile is a broken case, not a measurement.
fn compile(name: &str, source: &str) -> Vec<Box<dyn Domain>> {
    let file = SourceFile::new(name, source);
    let (compiled, diagnostics) = compile_source(&file);
    assert!(!diagnostics.has_errors(), "the validation model compiles:\n{}", diagnostics.render(&file));
    compiled.expect("a model without errors").domains
}

fn step_all(domains: &mut [Box<dyn Domain>], dt: f64, steps: usize) {
    let mut arena = Arena::with_capacity(1 << 16);
    for _ in 0..steps {
        for domain in domains.iter_mut() {
            let mut ctx = StepContext::new(&mut arena);
            domain.prepare(&mut ctx);
            domain.advance(dt, &mut ctx);
        }
    }
}

fn observe(domains: &[Box<dyn Domain>]) -> Observations {
    let mut out = Observations::new();
    for domain in domains {
        domain.observe(&mut out);
    }
    out
}

/// Every scalar field the solvers draw, as bits, halos included.
fn field_bits(domains: &[Box<dyn Domain>]) -> Vec<u64> {
    let mut bits = Vec::new();
    for domain in domains {
        for channel in domain.render_channels() {
            if let RenderChannel::Scalar { field, .. } = channel {
                bits.extend(field.as_slice().iter().map(|v| v.to_bits()));
            }
        }
    }
    bits
}

/// A reacting front: `A` from the left, `B` from the right, meeting in the middle at
/// 340 K, with `reaction` as the reaction's body.
fn front(reaction: &str) -> String {
    format!(
        "project front {{
           duration: 3 second;
           grid g {{ size: [24, 12]; extent: [0.24 meter, 0.12 meter]; }}
           domain chemistry mix {{ grid: g; temperature: 340 kelvin; }}
           species A on g = left_half(2 mole / meter^2) {{ formula: H; diffusivity: 2e-4 meter^2 / second; }}
           species B on g = right_half(1 mole / meter^2) {{ formula: H; diffusivity: 1e-4 meter^2 / second; }}
           species C on g = 0 mole / meter^2 {{ formula: H2; diffusivity: 5e-5 meter^2 / second; }}
           reaction r {{ {reaction} }}
           solve reactions(mix) with strang(dt=0.05 second);
         }}"
    )
}

/// The built-in multiplies `k_f(T)` by the product of the reactants, `[A]·[B]`, after
/// flooring each at zero; `k_f(T) = k · exp(−Ea / (R T))`. The law below says the same
/// in the same order: `c(…)` floors as mass action does, `arrhenius` is the built-in's
/// factor operation for operation, and the parentheses put the concentrations' product
/// where the built-in puts it. Same rates, so the same sub-steps, so the same bits.
fn mass_action_bitwise() -> Outcome {
    let builtin_body = "reactants: A + B; products: C; rate: 30 meter^2 / (mole second); \
                        activation_energy: 8 kilojoule / mole; enthalpy: -50 kilojoule / mole;";
    let law_body = "stoichiometry: A + B -> C; \
                    rate: 30 meter^2 / (mole second) * arrhenius(8 kilojoule / mole) * (c(A) * c(B)); \
                    heat_release: 50 kilojoule / mole;";
    let mut builtin = compile("builtin.lattice", &front(builtin_body));
    let mut law = compile("law.lattice", &front(law_body));

    let (dt, steps) = (0.05, 60);
    step_all(&mut builtin, dt, steps);
    step_all(&mut law, dt, steps);

    let (a, b) = (field_bits(&builtin), field_bits(&law));
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    let heat = |domains: &[Box<dyn Domain>]| observe(domains).value("mix.heat_release").unwrap_or(f64::NAN);
    let produced = observe(&law).value("mix.C.total").unwrap_or(0.0);
    let heat_differs = heat(&builtin).to_bits() != heat(&law).to_bits();

    Outcome::at_most(
        "concentration values differing from the built-in's",
        "values",
        (differing + usize::from(heat_differs)) as f64,
        0.0,
    )
    .note(format!("{} values compared: three species on a 24 x 12 grid with their halos, after {steps} Strang steps of {dt} s", a.len()))
    .note(format!("{produced:.6e} mol of C produced, so the comparison is of a live reaction front"))
    .note(format!(
        "the published heat release agrees {}: {:.6e} W",
        if heat_differs { "only approximately" } else { "to the bit too" },
        heat(&law)
    ))
    .note("the reaction runs at the mixture's `temperature: 340 kelvin`, which the law reads through `arrhenius`")
}

/// Michaelis–Menten, `dS/dt = −V S / (K + S)`, which mass action cannot write: zeroth
/// order while the enzyme is saturated and first order once it is not.
///
/// Its solution is implicit, `F(S) = K ln(S/S₀) + (S − S₀) + V t = 0`, so the error of
/// a computed `S` is measured as `F(S) / F′(S)`, with `F′(S) = K/S + 1` — a Newton step
/// from the computed value to the exact one.
///
/// The bound is the integrator's own, derived rather than tuned. RK4 sub-cycled at an
/// accuracy bound `a` has an error proportional to `a⁴`, so the same run at `a/2`
/// estimates it by Richardson: `e(a) ≈ (16/15) |S(a) − S(a/2)|`. The bound allows twice
/// that, for the estimate's own higher-order terms, plus the rounding of the sub-steps
/// and of the residual: `(8n + 16) ε (S₀ + V t)` for `n` sub-steps. A law that computed
/// the wrong rate would agree with itself at both accuracies and fail by orders of
/// magnitude.
fn michaelis_menten() -> Outcome {
    let (v, k, s0, time) = (1.0, 0.5, 2.0, 3.0);
    let law = rate_law_from_source(&["S", "P"], "1 mole / (meter^2 second) * c(S) / (0.5 mole / meter^2 + c(S))")
        .unwrap_or_else(|rendered| panic!("the validation law compiles:\n{rendered}"));
    let network = || {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("S").with_formula("C").unwrap());
        network.add_species(Species::new("P").with_formula("C").unwrap());
        network.add_reaction(Reaction::with_expression("enzyme", vec![Term::new(0, 1.0)], vec![Term::new(1, 1.0)], law.clone()));
        network
    };
    let run = |accuracy: f64| {
        let mut kinetics = Kinetics::new(network()).with_accuracy(accuracy);
        let mut c = [s0, 0.0];
        let report = kinetics.advance(&mut c, 300.0, time);
        (c, report.substeps)
    };
    let error = |s: f64| (k * (s / s0).ln() + (s - s0) + v * time) / (k / s + 1.0);

    let accuracy = 0.02;
    let (coarse, n) = run(accuracy);
    let (fine, n_fine) = run(accuracy / 2.0);
    let estimate = 16.0 / 15.0 * (coarse[0] - fine[0]).abs();
    let floor = (8.0 * n as f64 + 16.0) * f64::EPSILON * (s0 + v * time);
    let bound = 2.0 * estimate + floor;
    let (e_coarse, e_fine) = (error(coarse[0]).abs(), error(fine[0]).abs());

    Outcome::at_most("|S - S_exact| after 3 s", "mol/m^2", e_coarse, bound)
        .note(format!("S = {:.12} mol/m^2 from S0 = {s0}, V = {v} mol/(m^2 s), K = {k} mol/m^2", coarse[0]))
        .note(format!(
            "Richardson estimate {estimate:.3e} from {n} and {n_fine} sub-steps; rounding floor {floor:.3e}"
        ))
        .note(format!(
            "error at a/2 is {e_fine:.3e}, a ratio of {:.1} where fourth order predicts 16",
            e_coarse / e_fine.max(f64::MIN_POSITIVE)
        ))
        .note(format!("what left S arrived at P: S + P = {:.15}", coarse[0] + coarse[1]))
}

/// §8.3's neutralization as the spec writes it, well mixed so nothing but the reaction
/// moves anything: every cell starts at the same state and nothing diffuses.
const NEUTRALIZATION: &str = "project neutralization {
  duration: 2 second;
  let k = 40 meter^2 / (mole second);
  grid beaker { size: [4, 4]; extent: [0.1 meter, 0.1 meter]; }
  domain chemistry water { grid: beaker; }
  species H_plus on beaker = 1 mole / meter^2 { formula: H; charge: 1; diffusivity: 0 meter^2 / second; }
  species OH_minus on beaker = 1 mole / meter^2 { formula: OH; charge: -1; diffusivity: 0 meter^2 / second; }
  species H2O on beaker = 0 mole / meter^2 { formula: H2O; diffusivity: 0 meter^2 / second; }

  reaction neutralization {
      stoichiometry: H_plus + OH_minus -> H2O;
      rate: k * c(H_plus) * c(OH_minus);
      heat_release: 57.3 kilojoule / mole;
  }

  solve reactions(water) with strang(dt=0.05 second);
}";

/// The energy the mixture publishes against the extent times the declared heat.
///
/// The integrator carries the energy through the same Runge–Kutta stages as the
/// concentrations, so the two agree up to the rounding of their separate sums. Each
/// sub-step rounds the heat of each stage once (`q = 57300 · R`) and each weighted sum
/// of four positive terms about five times, so a sub-step's two increments differ by at
/// most about 13 ε relative; accumulating `n` sub-steps adds at most `2n ε` more. The
/// published field is then divided by `dt` and multiplied back, summed over 16 cells,
/// and over the steps. Sub-steps are bounded analytically: the Jacobian's largest row
/// sum is `k ([H⁺] + [OH⁻]) ≤ 80 /s`, so a half-step of 0.025 s takes at most
/// `⌈0.025 · 80 / 0.1⌉ + 1 = 21` at the default accuracy of 0.1.
fn heat_release() -> Outcome {
    let mut domains = compile("neutralization.lattice", NEUTRALIZATION);
    let (dt, steps) = (0.05, 40);
    let mut arena = Arena::with_capacity(1 << 16);
    let mut energy = 0.0;
    for _ in 0..steps {
        for domain in domains.iter_mut() {
            let mut ctx = StepContext::new(&mut arena);
            domain.prepare(&mut ctx);
            domain.advance(dt, &mut ctx);
        }
        // Average power over the step, W, times the step.
        energy += observe(&domains).value("water.heat_release").unwrap_or(f64::NAN) * dt;
    }
    let after = observe(&domains);
    let extent = after.value("water.H2O.total").unwrap_or(f64::NAN);
    let consumed = 1.0 * 0.01 - after.value("water.H_plus.total").unwrap_or(f64::NAN);
    let expected = extent * 57_300.0;

    let substeps = (steps * 2 * 21) as f64;
    let (cells, epsilon) = (16.0, f64::EPSILON);
    let bound = (13.0 + 2.0 * substeps + 2.0 * cells + 3.0 * steps as f64) * epsilon;
    let relative = (energy - expected).abs() / expected;

    Outcome::at_most("|energy published - extent x heat_release| / energy", "1", relative, bound)
        .note(format!("{energy:.9e} J published; {extent:.9e} mol of H2O formed x 57.3 kJ/mol = {expected:.9e} J"))
        .note(format!("H+ consumed {consumed:.9e} mol, the same extent counted from the other side"))
        .note(format!(
            "equal second-order reactants follow c = c0 / (1 + k c0 t), so the extent should be 0.01 mol x 80/81 = {:.9e}",
            0.01 * 80.0 / 81.0
        ))
        .note(format!("bound {bound:.3e} from at most {substeps} sub-steps per cell, {cells} cells and {steps} steps"))
        .note("heat_release is enthalpy negated: positive is heat out, and the published power is positive")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rate_law_case_passes() {
        for case in CASES {
            let outcome = (case.run)();
            assert!(outcome.passed(), "{}: {} = {} ({:?})\n{:#?}", case.name, outcome.metric, outcome.observed, outcome.criterion, outcome.notes);
        }
    }
}
