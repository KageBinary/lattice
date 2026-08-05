//! Cases that validate the *contracts* rather than the mathematics.
//!
//! Spec §21.1: *"A milestone is not complete because a demo looks compelling. It is
//! complete only when model semantics, validation, diagnostics, data output, and
//! documentation exist."* These cases enforce the parts of that a machine can check,
//! so an under-documented solver fails the validation run rather than shipping.
//!
//! P1 makes the strongest of these requirements: every module must state its *known
//! non-conservation*. A solver whose list is empty is claiming exact conservation of
//! everything it touches, which is almost never true and is exactly the kind of quiet
//! overclaim the spec is written against.

use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_domain_particle::{HarmonicWell, Integrator, ParticleDomain, ParticleSpec};
use lattice_ir::{Domain, Grid2d, SolverContract, StabilityReason};

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "solver_contracts_are_complete",
        domain: "runtime",
        level: Level::Property,
        claim: "every shipped solver states equations, assumptions, stability, precisions and validation cases",
        run: contracts_are_complete,
    },
    Case {
        name: "solvers_declare_what_they_do_not_conserve",
        domain: "runtime",
        level: Level::Property,
        claim: "no solver claims exact conservation of everything by leaving its non-conservation list empty",
        run: non_conservation_is_declared,
    },
    Case {
        name: "stability_limits_carry_a_reason_code",
        domain: "runtime",
        level: Level::Property,
        claim: "a solver that reports a finite timestep limit also reports which mechanism sets it",
        run: stability_limits_have_reasons,
    },
];

/// Every solver configuration the engine currently ships.
fn all_contracts() -> Vec<&'static SolverContract> {
    let grid = Grid2d::new(8, 8, [1.0, 1.0]);
    let mut contracts = Vec::new();

    for integrator in
        [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
    {
        contracts.push(ParticleDomain::new("audit", 1).with_integrator(integrator).contract());
    }
    for scheme in [TimeScheme::Explicit, TimeScheme::CrankNicolson, TimeScheme::BackwardEuler] {
        contracts
            .push(HeatDomain::new("audit", grid, Diffusivity::Uniform(1.0)).with_scheme(scheme).contract());
    }
    contracts
}

fn contracts_are_complete() -> Outcome {
    let contracts = all_contracts();
    let mut gaps = 0usize;
    let mut detail = Vec::new();

    for contract in &contracts {
        let found = contract.audit();
        if !found.is_empty() {
            gaps += found.len();
            let names: Vec<String> = found.iter().map(ToString::to_string).collect();
            detail.push(format!("{}: {}", contract.name, names.join(", ")));
        }
    }

    let mut outcome = Outcome::at_most("incomplete contract sections", "count", gaps as f64, 0.0)
        .note(format!("{} solver configurations audited", contracts.len()));
    for line in detail {
        outcome = outcome.note(line);
    }
    outcome
}

fn non_conservation_is_declared() -> Outcome {
    let contracts = all_contracts();
    let silent: Vec<&str> = contracts
        .iter()
        .filter(|c| c.known_non_conservation.is_empty())
        .map(|c| c.name)
        .collect();

    let mut outcome = Outcome::at_most(
        "solvers with an empty known-non-conservation list",
        "count",
        silent.len() as f64,
        0.0,
    )
    .note(format!("{} solver configurations audited", contracts.len()));

    if silent.is_empty() {
        let total: usize = contracts.iter().map(|c| c.known_non_conservation.len()).sum();
        outcome = outcome.note(format!("{total} distinct limitations declared across all solvers"));
    } else {
        outcome = outcome.note(format!("silent: {}", silent.join(", ")));
    }
    outcome
}

fn stability_limits_have_reasons() -> Outcome {
    let mut contradictions = 0usize;
    let mut detail = Vec::new();

    // A stiff particle domain: the harmonic well must report an oscillation limit.
    let mut stiff = ParticleDomain::new("stiff", 1)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(HarmonicWell::new([0.0, 0.0], 400.0));
    stiff.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
    stiff.initialize();

    // An explicit grid domain: must report a diffusion limit.
    let grid = Grid2d::new(32, 32, [1.0, 1.0]);
    let explicit = HeatDomain::new("explicit", grid, Diffusivity::Uniform(0.5))
        .with_scheme(TimeScheme::Explicit);
    let implicit = HeatDomain::new("implicit", grid, Diffusivity::Uniform(0.5))
        .with_scheme(TimeScheme::CrankNicolson);

    let checks: Vec<(&str, lattice_ir::StableStep)> = vec![
        ("particles2d/harmonic", stiff.stable_step()),
        ("grid2d/explicit", explicit.stable_step()),
        ("grid2d/crank_nicolson", implicit.stable_step()),
    ];

    for (name, limit) in &checks {
        let finite = limit.max.is_finite();
        let claims_unconditional = limit.reason == StabilityReason::Unconditional;
        if finite == claims_unconditional {
            contradictions += 1;
            detail.push(format!(
                "{name}: max = {:e} but reason = {:?}",
                limit.max, limit.reason
            ));
        } else {
            detail.push(format!("{name}: {} ({})", limit.max, limit.reason.describe()));
        }
    }

    let mut outcome = Outcome::at_most(
        "solvers whose stability limit and reason code disagree",
        "count",
        contradictions as f64,
        0.0,
    );
    for line in detail {
        outcome = outcome.note(line);
    }
    outcome
}
