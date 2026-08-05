//! `lattice inspect` — show what the engine knows about itself.
//!
//! Spec §17.2 asks that a user be able to *"inspect an entity/cell and see all state
//! values with units and source equations"*, and NFR-003 requires solver metadata to
//! be present at runtime rather than only in a manual. This command is the text-mode
//! surface of that: solver contracts, unit resolution, and the registries of
//! validation cases, benchmarks and demos.

use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_domain_particle::{Integrator, ParticleDomain};
use lattice_ir::{Domain, Grid2d, SolverContract};
use lattice_units::UnitRegistry;

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

    #[test]
    fn registries_are_listed() {
        assert!(cases().contains("free_fall"));
        assert!(benchmarks().contains("particles-lj"));
        assert!(benchmarks().contains("correctness:"));
        assert!(demos().contains("oscillator"));
    }
}
