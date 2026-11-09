//! `lattice inspect` — show what the engine knows about itself.
//!
//! Spec §17.2 asks that a user be able to *"inspect an entity/cell and see all state
//! values with units and source equations"*, and NFR-003 requires solver metadata to
//! be present at runtime rather than only in a manual. This command is the text-mode
//! surface of that: solver contracts, unit resolution, and the registries of
//! validation cases, benchmarks and demos.

use lattice_compiler::laws::{Ty, TypedLaw};
use lattice_compiler::lower_law::lower;
use lattice_compiler::{compile_source, LawUse};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_domain_particle::user::layout;
use lattice_domain_particle::{Integrator, ParticleDomain, UserLawKind};
use lattice_expr::{Op, MAX_REGISTERS};
use lattice_ir::{Domain, Grid2d, SolverContract};
use lattice_syntax::SourceFile;
use lattice_units::UnitRegistry;

use crate::project::{op_name, show};
use crate::{bench, demo};

/// Every solver configuration the engine ships, in a stable order.
pub fn all_contracts() -> Vec<&'static SolverContract> {
    let grid = Grid2d::new(8, 8, [1.0, 1.0]);
    let mut contracts = Vec::new();
    for integrator in
        [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
    {
        contracts.push(ParticleDomain::new("inspect", 1).with_integrator(integrator).contract());
    }
    for scheme in [TimeScheme::Explicit, TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
        contracts.push(
            HeatDomain::new("inspect", grid, Diffusivity::Uniform(1.0)).with_scheme(scheme).contract(),
        );
    }
    contracts
}

/// Print every solver contract.
pub fn contracts() -> String {
    let contracts = all_contracts();
    let mut out = format!("{} solver configurations\n", contracts.len());
    for contract in contracts {
        out.push('\n');
        out.push_str(&contract.report());
        let gaps = contract.audit();
        if !gaps.is_empty() {
            let names: Vec<String> = gaps.iter().map(ToString::to_string).collect();
            out.push_str(&format!("  INCOMPLETE: {}\n", names.join(", ")));
        }
    }
    out
}

/// Resolve a unit expression or dimensioned literal.
///
/// Prints the SI magnitude, the dimension, and any ambiguity warning — the same
/// information the model compiler would attach to a diagnostic.
pub fn units(expression: Option<&str>) -> Result<String, String> {
    let registry = UnitRegistry::si();

    let Some(expression) = expression else {
        let spellings = registry.spellings();
        let mut out = format!(
            "{} units registered under {} spellings\n\n",
            registry.len(),
            spellings.len()
        );
        out.push_str("Pass an expression to resolve one, for example:\n");
        for example in [
            "9.31e-9 meter^2 / second",
            "-57.3 kilojoule / mole",
            "1 mole / meter^2",
            "20 electronvolt",
            "25 celsius",
        ] {
            out.push_str(&format!("  lattice inspect units \"{example}\"\n"));
        }
        out.push_str("\nRegistered spellings:\n");
        for chunk in spellings.chunks(10) {
            out.push_str(&format!("  {}\n", chunk.join("  ")));
        }
        return Ok(out);
    };

    let parsed = registry
        .parse_quantity(expression)
        .map_err(|e| format!("cannot resolve `{expression}`: {e}"))?;
    let quantity = parsed.value;

    let mut out = format!("  expression   {expression}\n");
    out.push_str(&format!("  SI value     {}\n", quantity.value()));
    out.push_str(&format!("  dimension    {}\n", quantity.dimension().describe()));
    out.push_str(&format!("  exponents    {:?}\n", quantity.dimension().exponents()));
    for warning in &parsed.warnings {
        out.push_str(&format!("  WARNING      {warning}\n"));
    }
    Ok(out)
}

/// List the validation cases.
pub fn cases() -> String {
    let cases = lattice_validation::all_cases();
    let mut out = format!("{} validation cases\n\n", cases.len());
    for case in cases {
        out.push_str(&format!(
            "  {:<44} {:<14} {}\n      {}\n",
            case.name,
            case.level.label(),
            case.domain,
            case.claim
        ));
    }
    out
}

/// List the benchmarks.
pub fn benchmarks() -> String {
    let mut out = format!("{} benchmarks\n\n", bench::all().len());
    for benchmark in bench::all() {
        out.push_str(&format!("  {}\n      {}\n", benchmark.name, benchmark.description));
        out.push_str(&format!("      correctness: {}\n\n", benchmark.correctness));
    }
    out
}

/// List the demos.
pub fn demos() -> String {
    let mut out = format!("{} demos\n\n", demo::all().len());
    for d in demo::all() {
        out.push_str(&format!("  {}\n      {}\n\n", d.name, d.description));
    }
    out
}

/// Why `inspect expr` has nothing to show.
#[derive(Debug)]
pub enum ExprFailure {
    /// The file did not compile; the rendered diagnostics.
    DoesNotCompile(String),
    /// No law has the name asked for; the message lists the ones that exist.
    UnknownLaw(String),
}

/// The most characters of a source expression a row shows.
const SOURCE_WIDTH: usize = 60;

/// Print the compiled form of a file's laws next to the source each operation came from
/// (spec M6.1d §4.5), for the law named or for every law.
pub fn expr(file: &SourceFile, law: Option<&str>) -> Result<String, ExprFailure> {
    let (compiled, diagnostics) = compile_source(file);
    let compiled = match compiled {
        Some(compiled) if !diagnostics.has_errors() => compiled,
        _ => return Err(ExprFailure::DoesNotCompile(diagnostics.render(file))),
    };

    let names: Vec<&str> = compiled.laws.iter().map(|l| l.name.as_str()).collect();
    let selected: Vec<&TypedLaw> = match law {
        Some(wanted) => match compiled.laws.iter().find(|l| l.name == wanted) {
            Some(found) => vec![found],
            None => {
                return Err(ExprFailure::UnknownLaw(if names.is_empty() {
                    format!("no law named `{wanted}`; `{}` declares no laws", file.name())
                } else {
                    format!("no law named `{wanted}`; `{}` declares: {}", file.name(), names.join(", "))
                }));
            }
        },
        None => compiled.laws.iter().collect(),
    };
    if selected.is_empty() {
        return Ok(format!("`{}` declares no laws.\n", file.name()));
    }

    let mut out = String::new();
    for (i, law) in selected.into_iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let uses: Vec<&LawUse> = compiled.law_uses.iter().filter(|u| u.law == law.name).collect();
        out.push_str(&law_listing(file, law, &uses));
    }
    Ok(out)
}

/// One law: header, one row per operation, outputs, and its use sites.
fn law_listing(file: &SourceFile, law: &TypedLaw, uses: &[&LawUse]) -> String {
    let lowered = match lower(law) {
        Ok(lowered) => lowered,
        Err(diagnostic) => {
            return format!("law {}: could not be compiled\n{}", law.name, diagnostic.render(file));
        }
    };
    let program = &lowered.program;
    let kind = lowered.kind;
    let pair = kind.is_pair();

    // What each parameter slot is called: a vector takes two.
    let mut slots: Vec<String> = Vec::new();
    for param in &law.params {
        if matches!(param.ty, Ty::Vec2(_)) {
            slots.push(format!("{}.x", param.name));
            slots.push(format!("{}.y", param.name));
        } else {
            slots.push(param.name.clone());
        }
    }
    let inputs: Vec<String> = (0..kind.inputs() as u32).map(|n| layout::input_name(pair, n)).collect();
    let shown_inputs = if inputs.len() > 8 {
        format!("{} … {}", inputs[..4].join(" "), inputs[inputs.len() - 1])
    } else {
        inputs.join(" ")
    };
    let param_names: Vec<&str> = law.params.iter().map(|p| p.name.as_str()).collect();

    let (label, outputs): (&str, &[&str]) = match kind {
        UserLawKind::PairForce => ("pair force", &["Fx", "Fy"]),
        UserLawKind::BodyForce => ("body force", &["Fx", "Fy"]),
        UserLawKind::PairPotential => ("pair potential", &["U", "dU/dr"]),
        UserLawKind::BodyPotential => ("body potential", &["U", "dU/dx", "dU/dy"]),
        UserLawKind::BodyObserver => ("observer", &["value"]),
    };
    let mut out = format!(
        "law {}: {label}, {} inputs ({shown_inputs}), {} params ({})\n",
        law.name,
        inputs.len(),
        param_names.len(),
        param_names.join(" ")
    );

    // Each row is register, operation, operands, then the source it came from.
    let rows: Vec<(String, String, String)> = program
        .ops()
        .iter()
        .enumerate()
        .map(|(reg, &op)| {
            let (name, operands) = match op {
                Op::Const(value) => ("const".to_string(), show(value)),
                Op::Input(n) => ("input".to_string(), inputs[n as usize].clone()),
                Op::Param(n) => ("param".to_string(), slots[n as usize].clone()),
                _ => {
                    let mut operands: Vec<String> = op.operands().map(|r| format!("r{r}")).collect();
                    if let Op::Powi(_, n) = op {
                        operands.push(n.to_string());
                    }
                    (op_name(op), operands.join(" "))
                }
            };
            let source = program
                .origin(reg as u32)
                .and_then(|o| file.text().get(o.start as usize..o.end as usize))
                .map(one_line)
                .unwrap_or_default();
            (name, operands, source)
        })
        .collect();
    let reg_width = format!("r{}", rows.len().saturating_sub(1)).len();
    let name_width = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    let operand_width = rows.iter().map(|r| r.1.len()).max().unwrap_or(0);
    for (reg, (name, operands, source)) in rows.iter().enumerate() {
        let head = format!("  {:<reg_width$}  {name:<name_width$}  {operands:<operand_width$}", format!("r{reg}"));
        if source.is_empty() {
            out.push_str(head.trim_end());
        } else {
            out.push_str(&format!("{head}  | {source}"));
        }
        out.push('\n');
    }

    let outputs: Vec<String> =
        outputs.iter().zip(program.outputs()).map(|(label, reg)| format!("{label} = r{reg}")).collect();
    let derivative = match &lowered.energy {
        Some(energy) => format!("{} ops (the energy alone: {})", program.ops().len(), energy.ops().len()),
        None => "none".to_string(),
    };
    out.push_str(&format!(
        "  outputs: {}     derivative: {derivative}    registers {} / {}\n",
        outputs.join(", "),
        program.ops().len(),
        MAX_REGISTERS
    ));
    out.push_str(&format!("  fingerprint 0x{:016x}\n", program.fingerprint()));

    if uses.is_empty() {
        out.push_str("  not used by any particle set\n");
    }
    for site in uses {
        let mut at = 0;
        let mut bound = Vec::new();
        for param in &law.params {
            if matches!(param.ty, Ty::Vec2(_)) {
                bound.push(format!("{} = [{}, {}]", param.name, show(site.params[at]), show(site.params[at + 1])));
                at += 2;
            } else {
                bound.push(format!("{} = {}", param.name, show(site.params[at])));
                at += 1;
            }
        }
        if bound.is_empty() {
            out.push_str(&format!("  used by `{}`\n", site.particles));
        } else {
            out.push_str(&format!("  used by `{}` (SI values): {}\n", site.particles, bound.join(", ")));
        }
    }
    out
}

/// Source text on one line: runs of white space become one space, and a long expression
/// is cut with an ellipsis.
fn one_line(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.chars().count() <= SOURCE_WIDTH {
        return joined;
    }
    let cut: String = joined.chars().take(SOURCE_WIDTH - 1).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contracts_are_listed_and_none_are_incomplete() {
        let text = contracts();
        assert!(text.contains("particles2d"), "{text}");
        assert!(text.contains("grid2d.heat"), "{text}");
        assert!(!text.contains("INCOMPLETE"), "a shipped solver has an incomplete contract:\n{text}");
    }

    #[test]
    fn unit_resolution_reports_value_and_dimension() {
        let text = units(Some("9.31e-9 meter^2 / second")).unwrap();
        assert!(text.contains("diffusivity"), "{text}");
        assert!(text.contains("9.31e-9"), "{text}");
    }

    /// An ambiguous expression must resolve *and* say how it was grouped.
    #[test]
    fn unit_resolution_surfaces_ambiguity_warnings() {
        let text = units(Some("8.314 joule / mole * kelvin")).unwrap();
        assert!(text.contains("WARNING"), "{text}");
        assert!(text.contains("left-to-right"), "{text}");
    }

    #[test]
    fn a_bad_unit_expression_is_an_error_not_a_panic() {
        let err = units(Some("5 furlong")).unwrap_err();
        assert!(err.contains("furlong"), "{err}");
    }

    #[test]
    fn listing_units_without_an_expression_shows_examples() {
        let text = units(None).unwrap();
        assert!(text.contains("registered"), "{text}");
        assert!(text.contains("electronvolt"), "{text}");
    }

    fn example(name: &str) -> SourceFile {
        let path = format!("{}/../../examples/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        SourceFile::new(name, text)
    }

    /// The whole listing of the spec's worked example. A change to the lowering, the
    /// origins or the layout shows up here, in a form a person can read.
    #[test]
    fn the_damped_spring_listing_is_stable() {
        let text = expr(&example("damped_spring.lattice"), None).unwrap();
        let expected = "\
law spring: pair force, 15 inputs (d.x d.y r a.position.x … b.charge), 3 params (rest_length stiffness damping)
  r0   input  d.x           | minimum_image(b.position - a.position)
  r1   input  d.y           | minimum_image(b.position - a.position)
  r2   mul    r0 r0         | length(dx)
  r3   mul    r1 r1         | length(dx)
  r4   add    r2 r3         | length(dx)
  r5   sqrt   r4            | length(dx)
  r6   param  rest_length   | rest_length
  r7   sub    r5 r6         | length(dx) - rest_length
  r8   param  stiffness     | stiffness
  r9   mul    r8 r7         | stiffness * extension
  r10  div    r0 r5         | normalize(dx)
  r11  div    r1 r5         | normalize(dx)
  r12  mul    r9 r10        | stiffness * extension * normalize(dx)
  r13  mul    r9 r11        | stiffness * extension * normalize(dx)
  r14  param  damping       | damping
  r15  input  b.velocity.x  | b.velocity
  r16  input  b.velocity.y  | b.velocity
  r17  input  a.velocity.x  | a.velocity
  r18  input  a.velocity.y  | a.velocity
  r19  sub    r15 r17       | b.velocity - a.velocity
  r20  sub    r16 r18       | b.velocity - a.velocity
  r21  mul    r19 r10       | dot(b.velocity - a.velocity, normalize(dx))
  r22  mul    r20 r11       | dot(b.velocity - a.velocity, normalize(dx))
  r23  add    r21 r22       | dot(b.velocity - a.velocity, normalize(dx))
  r24  mul    r14 r23       | damping * dot(b.velocity - a.velocity, normalize(dx))
  r25  mul    r24 r10       | damping * dot(b.velocity - a.velocity, normalize(dx)) * nor…
  r26  mul    r24 r11       | damping * dot(b.velocity - a.velocity, normalize(dx)) * nor…
  r27  add    r12 r25       | stiffness * extension * normalize(dx) + damping * dot(b.vel…
  r28  add    r13 r26       | stiffness * extension * normalize(dx) + damping * dot(b.vel…
  outputs: Fx = r27, Fy = r28     derivative: none    registers 29 / 512
  fingerprint 0x8cd5c476134692ea
  used by `beads` (SI values): rest_length = 1, stiffness = 40, damping = 0.2
";
        assert_eq!(text, expected);
        // Naming the only law prints the same thing.
        assert_eq!(expr(&example("damped_spring.lattice"), Some("spring")).unwrap(), expected);
    }

    #[test]
    fn an_unknown_law_is_an_error_that_lists_the_names() {
        let Err(ExprFailure::UnknownLaw(message)) = expr(&example("damped_spring.lattice"), Some("nope")) else {
            panic!("expected an unknown-law error");
        };
        assert!(message.contains("`nope`"), "{message}");
        assert!(message.contains("declares: spring"), "{message}");
    }

    #[test]
    fn a_file_that_does_not_compile_yields_its_diagnostics() {
        let file = SourceFile::new("bad.lattice", "project bad { particles atoms { count: 4; mass: 1 second; } }");
        let Err(ExprFailure::DoesNotCompile(rendered)) = expr(&file, None) else {
            panic!("expected a compile failure");
        };
        assert!(rendered.contains("error"), "{rendered}");
    }

    #[test]
    fn a_file_without_laws_says_so() {
        let text = expr(&example("lj_gas.lattice"), None).unwrap();
        assert!(text.contains("declares no laws"), "{text}");
    }

    /// A potential lists the program that runs, which holds both the energy and its
    /// derivative, and says how much of it is the energy.
    #[test]
    fn a_potential_shows_its_derivative() {
        let file = SourceFile::new(
            "lj.lattice",
            "project lj {
  dimensions: 2;
  duration: 1 second;
  potential lj(a: particle, b: particle) -> joule {
    param epsilon: joule;
    param sigma: meter;
    let s6 = (sigma / distance(a, b))^6;
    return 4 * epsilon * (s6^2 - s6);
  }
  particles atoms {
    count: 16;
    region: [8 meter, 8 meter];
    spacing: 1.4 meter;
    mass: 1 kilogram;
    boundary: periodic;
    force: lj(epsilon=1 joule, sigma=1 meter, cutoff=2.5 meter);
  }
  solve dynamics(atoms) with velocity_verlet(dt=0.005 second);
}",
        );
        let text = expr(&file, Some("lj")).unwrap();
        assert!(text.starts_with("law lj: pair potential, 15 inputs"), "{text}");
        assert!(text.contains("2 params (epsilon sigma)"), "{text}");
        assert!(text.contains("  | (sigma / distance(a, b))^6"), "{text}");
        assert!(text.contains("outputs: U = r"), "{text}");
        assert!(text.contains(", dU/dr = r"), "{text}");
        assert!(text.contains("(the energy alone: "), "{text}");
        assert!(text.contains("used by `atoms` (SI values): epsilon = 1, sigma = 1"), "{text}");
    }

    #[test]
    fn registries_are_listed() {
        assert!(cases().contains("free_fall"));
        assert!(benchmarks().contains("particles-lj"));
        assert!(benchmarks().contains("correctness:"));
        assert!(demos().contains("oscillator"));
    }
}
