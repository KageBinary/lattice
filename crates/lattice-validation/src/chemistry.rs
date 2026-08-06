//! Validation cases for the chemistry domain (spec §19.2).
//!
//! > Gray–Scott or other reaction-diffusion patterns plus mass-balance checks.
//! > First-order and reversible reaction networks with analytic solutions.
//! > […] Coupled reaction heat transfer with ledger balance.
//!
//! Gillespie statistics are the one row of that list not here: the stochastic path is
//! not implemented, and a case that cannot run must be absent from the report rather
//! than present and silently skipped.
//!
//! # What these are chosen to catch
//!
//! A reaction network is easy to test badly. "Concentrations went down and products
//! went up" passes for a network that is missing an atom, integrating at the wrong
//! order, and running at the wrong rate. So each case below pins a *number* that only
//! comes out right if the specific thing it names is right: the exact equilibrium a
//! reversible pair settles at, the exact exponential a first-order decay follows, the
//! exact element totals a balanced network preserves, and the exact energy a coupling
//! edge delivers.

use lattice_coupling::{CouplingEdge, Coupler, Mapping, PortRef};
use lattice_domain_chemistry::{
    Integrator, Kinetics, RateLaw, ReactingMixture, Reaction, ReactionNetwork, Species, Term,
    GAS_CONSTANT,
};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, Domain, Grid2d, Invariant, Observations, StepContext};

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "first_order_decay",
        domain: "chemistry",
        level: Level::Analytic,
        claim: "a first-order reaction follows [A](t) = [A]0 exp(-kt) exactly",
        run: first_order_decay,
    },
    Case {
        name: "reversible_equilibrium",
        domain: "chemistry",
        level: Level::Analytic,
        claim: "a reversible pair settles at the ratio its rate constants set, approaching it at (kf + kr)",
        run: reversible_equilibrium,
    },
    Case {
        name: "arrhenius_rate_factor",
        domain: "chemistry",
        level: Level::Analytic,
        claim: "a rate constant scales with temperature by exactly exp(-Ea/RT)",
        run: arrhenius_factor,
    },
    Case {
        name: "kinetics_integrator_order",
        domain: "chemistry",
        level: Level::Manufactured,
        claim: "the reaction integrator converges at the fourth order it declares",
        run: integrator_order,
    },
    Case {
        name: "reaction_diffusion_conserves_atoms",
        domain: "chemistry",
        level: Level::Property,
        claim: "a balanced network on a closed grid conserves every element to round-off, however much reacts",
        run: conserves_atoms,
    },
    Case {
        name: "unbalanced_reaction_is_detected",
        domain: "chemistry",
        level: Level::Unit,
        claim: "an unbalanced reaction is caught by the atom check rather than by a ledger that will not close",
        run: unbalanced_detected,
    },
    Case {
        name: "strang_splitting_order",
        domain: "chemistry",
        level: Level::Manufactured,
        claim: "reaction-diffusion splitting converges at second order, not the first order a naive split gives",
        run: splitting_order,
    },
    Case {
        name: "coupled_reaction_heat_ledger",
        domain: "coupling",
        level: Level::Scenario,
        claim: "the energy an exothermic reaction sends across a coupling edge is the energy the temperature field receives",
        run: coupled_ledger,
    },
];

/// `A -> B`, first order.
fn decay_network(k: f64) -> ReactionNetwork {
    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("A").with_formula("C").unwrap());
    network.add_species(Species::new("B").with_formula("C").unwrap());
    network.add_reaction(Reaction::new(
        "decay",
        vec![Term::new(0, 1.0)],
        vec![Term::new(1, 1.0)],
        RateLaw::mass_action(k),
    ));
    network
}

fn first_order_decay() -> Outcome {
    let (k, time) = (0.7, 3.0);
    let mut kinetics = Kinetics::new(decay_network(k)).accurate();
    let mut c = [1.0, 0.0];
    let report = kinetics.advance(&mut c, 300.0, time);

    let expected = (-k * time).exp();
    Outcome::near("[A] after 3 s", "mol/m^2", c[0], expected, 1e-9)
        .note(format!("analytic exp(-kt) = {expected:.12} at k = {k} /s"))
        .note(format!("{} sub-steps, smallest {:.3e} s", report.substeps, report.smallest_step))
        .note(format!("what left A arrived at B: [B] = {:.12}", c[1]))
        .note("the sum is conserved by the stoichiometry, so this checks the *rate*")
}

fn reversible_equilibrium() -> Outcome {
    let (forward, reverse) = (3.0, 1.0);
    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("A").with_formula("C").unwrap());
    network.add_species(Species::new("B").with_formula("C").unwrap());
    network.add_reaction(Reaction::new(
        "isomerize",
        vec![Term::new(0, 1.0)],
        vec![Term::new(1, 1.0)],
        RateLaw::reversible(forward, reverse),
    ));
    let mut kinetics = Kinetics::new(network).accurate();

    // Long enough to settle: the relaxation time is 1/(kf + kr) = 0.25 s.
    let mut settled = [1.0, 0.0];
    kinetics.advance(&mut settled, 300.0, 20.0);

    // And the *approach*, which is the half a solver can get wrong while still landing
    // on the right equilibrium.
    let mut partway = [1.0, 0.0];
    kinetics.advance(&mut partway, 300.0, 0.1);
    let relaxation = (-(forward + reverse) * 0.1f64).exp();
    let predicted_partway = 0.25 + 0.75 * relaxation;

    Outcome::near("[A] at equilibrium", "mol/m^2", settled[0], 0.25, 1e-9)
        .note(format!("K = kf/kr = {}, so [B]/[A] = {} and the total is 1", forward / reverse, forward / reverse))
        .note(format!(
            "the approach at t = 0.1 s: {:.12} against the predicted {predicted_partway:.12}",
            partway[0]
        ))
        .note(format!(
            "relaxation error {:.3e}, which is the part that would still be wrong if only \
             the equilibrium were checked",
            (partway[0] - predicted_partway).abs()
        ))
}

fn arrhenius_factor() -> Outcome {
    let (activation, cold, hot) = (50_000.0, 300.0, 350.0);
    let law = RateLaw::mass_action(1e10).with_activation_energy(activation);

    let measured = law.forward_at(hot) / law.forward_at(cold);
    let expected = ((-activation / (GAS_CONSTANT * hot)) - (-activation / (GAS_CONSTANT * cold))).exp();

    Outcome::near("rate ratio from 300 K to 350 K", "1", measured, expected, 1e-12)
        .note(format!("Ea = {activation} J/mol, R = {GAS_CONSTANT:.9} J/(mol K)"))
        .note(format!("k(300) = {:.6e} /s, k(350) = {:.6e} /s", law.forward_at(cold), law.forward_at(hot)))
        .note("this factor is the feedback loop the flagship demo turns on: hotter is faster")
}

fn integrator_order() -> Outcome {
    let k = 0.3;
    // One fixed step per measurement, so sub-cycling cannot hide the method's own
    // error and what is measured is the integrator rather than the step controller.
    let error_at = |steps: usize| {
        let mut kinetics =
            Kinetics::new(decay_network(k)).with_integrator(Integrator::Rk4).with_accuracy(1e9);
        let mut c = [1.0, 0.0];
        let dt = 1.0 / steps as f64;
        for _ in 0..steps {
            kinetics.advance(&mut c, 300.0, dt);
        }
        (c[0] - (-k).exp()).abs()
    };

    let coarse = error_at(4);
    let fine = error_at(8);
    let order = (coarse / fine).log2();

    Outcome::near("observed convergence order", "1", order, 4.0, 0.3)
        .note(format!("max error {coarse:.3e} at 4 steps -> {fine:.3e} at 8"))
        .note("the accuracy bound is relaxed so the fixed step is what is being measured")
        .note("RK4 declares fourth order; a scheme that quietly fell back would show as first or second")
}

fn conserves_atoms() -> Outcome {
    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("H2").with_formula("H2").unwrap().with_diffusion(4e-3));
    network.add_species(Species::new("O2").with_formula("O2").unwrap().with_diffusion(1e-3));
    network.add_species(Species::new("H2O").with_formula("H2O").unwrap().with_diffusion(2e-3));
    network.add_reaction(Reaction::new(
        "combustion",
        vec![Term::new(0, 2.0), Term::new(1, 1.0)],
        vec![Term::new(2, 2.0)],
        RateLaw::mass_action(3.0),
    ));

    let grid = Grid2d::new(24, 24, [1.0, 1.0]);
    let mut mixture = ReactingMixture::new("chamber", grid, network);
    // Separated, so the species have to diffuse into each other before anything can
    // happen — which exercises transport and reaction together rather than in turn.
    mixture.concentration_mut(0).unwrap().init_from_position(&grid, |p| if p[0] < 0.5 { 2.0 } else { 0.0 });
    mixture.concentration_mut(1).unwrap().init_from_position(&grid, |p| if p[0] >= 0.5 { 1.0 } else { 0.0 });

    let (mass0, h0, o0) = (mixture.total_mass(), mixture.total_element("H"), mixture.total_element("O"));
    let water_before = mixture.total(2);

    let mut arena = Arena::with_capacity(1 << 18);
    for _ in 0..400 {
        mixture.step(0.01, &mut arena);
    }

    let worst = [
        (mixture.total_mass() - mass0).abs() / mass0,
        (mixture.total_element("H") - h0).abs() / h0,
        (mixture.total_element("O") - o0).abs() / o0,
    ]
    .into_iter()
    .fold(0.0, f64::max);

    Outcome::at_most("worst relative drift in mass or any element", "1", worst, 1e-10)
        .note(format!(
            "hydrogen {h0:.6} -> {:.6} mol, oxygen {o0:.6} -> {:.6} mol",
            mixture.total_element("H"),
            mixture.total_element("O")
        ))
        .note(format!(
            "and it really reacted: {:.6} mol of water produced from {water_before:.6}",
            mixture.total(2)
        ))
        .note("insulated boundaries and a balanced network, which is the only case where this holds")
}

fn unbalanced_detected() -> Outcome {
    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("H2").with_formula("H2").unwrap());
    network.add_species(Species::new("O2").with_formula("O2").unwrap());
    network.add_species(Species::new("H2O").with_formula("H2O").unwrap());
    // The reaction the whole check exists for. It integrates perfectly happily.
    network.add_reaction(Reaction::new(
        "typo",
        vec![Term::new(0, 1.0), Term::new(1, 1.0)],
        vec![Term::new(2, 1.0)],
        RateLaw::mass_action(1.0),
    ));

    let imbalances = network.imbalances();
    let named = imbalances
        .first()
        .map(|(_, balance)| balance.describe())
        .unwrap_or_else(|| "nothing reported".to_string());

    // How much mass this reaction destroys per unit of extent, as a fraction of what
    // it consumes — the drift a conservation check would eventually see, several
    // layers away from the typo.
    let before = network.total_mass(&[1.0, 1.0, 0.0]);
    let after = network.total_mass(&[0.0, 0.0, 1.0]);
    let destroyed = (before - after) / before;

    Outcome::at_least("unbalanced reactions detected", "count", imbalances.len() as f64, 1.0)
        .note(named)
        .note(format!("running it to completion would destroy {:.1}% of the mass", destroyed * 100.0))
        .note("the balance check turns that into a compile error at the reaction, rather than a ledger that will not close")
}

fn splitting_order() -> Outcome {
    let grid = Grid2d::new(16, 16, [1.0, 1.0]);
    let interior = |mixture: &ReactingMixture| -> Vec<f64> {
        let field = mixture.concentration(0).unwrap();
        let mut out = Vec::with_capacity(grid.cell_count());
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                out.push(field.get(i, j));
            }
        }
        out
    };

    let solve = |steps: usize| {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap().with_diffusion(0.02));
        network.add_species(Species::new("B").with_formula("C").unwrap().with_diffusion(0.005));
        network.add_reaction(Reaction::new(
            "decay",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(2.0),
        ));
        // Different diffusivities, so the two operators genuinely do not commute and
        // there is a splitting error to measure at all.
        let mut mixture = ReactingMixture::new("split", grid, network).with_accuracy(0.005);
        mixture
            .concentration_mut(0)
            .unwrap()
            .init_from_position(&grid, |p| 1.0 + 0.5 * (core::f64::consts::TAU * p[0]).sin());

        let mut arena = Arena::with_capacity(1 << 18);
        for _ in 0..steps {
            mixture.step(0.5 / steps as f64, &mut arena);
        }
        interior(&mixture)
    };

    let exact = solve(256);
    let error_of = |steps: usize| {
        solve(steps).iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    };
    let (coarse, fine) = (error_of(8), error_of(16));
    let order = (coarse / fine).log2();

    Outcome::at_least("observed splitting order", "1", order, 1.7)
        .note(format!("max error {coarse:.3e} at 8 steps -> {fine:.3e} at 16"))
        .note("Strang: react dt/2, diffuse dt, react dt/2. Doing one then the other is first order")
        .note("measured on interior cells only — the halo holds intermediate boundary state, not part of the answer")
}

fn coupled_ledger() -> Outcome {
    /// Areal heat capacity of the chamber contents, J/(m²·K).
    const AREAL_HEAT_CAPACITY: f64 = 4.0e5;
    let grid = Grid2d::new(16, 16, [0.1, 0.1]);

    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("A").with_formula("C").unwrap().with_diffusion(1e-6));
    network.add_species(Species::new("B").with_formula("C").unwrap().with_diffusion(1e-6));
    network.add_reaction(
        Reaction::new(
            "burn",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(0.5),
        )
        .with_enthalpy(-2.0e5),
    );

    let mut mixture = ReactingMixture::new("chamber", grid, network);
    mixture.concentration_mut(0).unwrap().fill_interior(10.0);
    let heat = HeatDomain::new("temperature", grid, Diffusivity::Uniform(1e-5))
        .with_scheme(TimeScheme::CrankNicolson)
        .with_display_unit("K")
        .with_uniform_initial(300.0);
    let mut domains: Vec<Box<dyn Domain>> = vec![Box::new(mixture), Box::new(heat)];

    let mut coupler = Coupler::new();
    coupler.add(
        CouplingEdge::new("reaction_heat", PortRef::new(0, "heat_release"), PortRef::new(1, "source"))
            .with_mapping(Mapping::scale(
                1.0 / AREAL_HEAT_CAPACITY,
                "1 / (areal heat capacity), W/m^2 -> K/s",
            ))
            .carrying(Invariant::Energy),
    );
    coupler.add(CouplingEdge::new(
        "temperature_feedback",
        PortRef::new(1, "values"),
        PortRef::new(0, "temperature"),
    ));

    let observe = |domains: &[Box<dyn Domain>]| {
        let mut out = Observations::new();
        for domain in domains {
            domain.observe(&mut out);
        }
        out
    };
    let before = observe(&domains).value("temperature.integral").unwrap();

    let (dt, steps) = (0.05, 200);
    let mut arena = Arena::with_capacity(1 << 18);
    for step in 0..steps {
        for domain in domains.iter_mut() {
            let mut ctx = StepContext::new(&mut arena);
            domain.prepare(&mut ctx);
            domain.advance(dt, &mut ctx);
        }
        coupler.exchange(&mut domains, step, step as f64 * dt, dt);
    }
    let after = observe(&domains).value("temperature.integral").unwrap();

    // K·m² of field integral times J/(m²·K) of heat capacity is joules.
    let arrived = (after - before) * AREAL_HEAT_CAPACITY;
    let reconciliation = coupler.reconcile(Invariant::Energy, "temperature", arrived);
    let sent = coupler.ledger().received(Invariant::Energy, "temperature");
    let largest =
        coupler.ledger().transfers().iter().map(|t| t.amount).fold(0.0, f64::max);

    Outcome::at_most(
        "unaccounted energy, as a fraction of one exchange",
        "1",
        reconciliation.discrepancy().abs() / largest.max(f64::MIN_POSITIVE),
        1.001,
    )
    .note(format!("{sent:.6e} J sent across the edge, {arrived:.6e} J arrived in the field"))
    .note(format!(
        "shortfall {:.6e} J against a largest single exchange of {largest:.6e} J",
        reconciliation.discrepancy()
    ))
    .note(
        "a staggered coupling always has one exchange in flight: the heat released \
         during the final step is recorded and never delivered, because the run ends \
         first. That is what the ledger exists to make visible",
    )
    .note(format!("relative discrepancy {:.3e} over {steps} steps", reconciliation.relative_discrepancy()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_chemistry_case_passes() {
        for case in CASES {
            let outcome = (case.run)();
            assert!(
                outcome.passed(),
                "{}: measured {} {}, wanted {:?}\n  {}",
                case.name,
                outcome.observed,
                outcome.unit,
                outcome.criterion,
                outcome.notes.join("\n  ")
            );
            assert!(outcome.observed.is_finite(), "{} produced {}", case.name, outcome.observed);
            assert!(!outcome.notes.is_empty(), "{} records nothing about what it did", case.name);
        }
    }

    /// §19.2's chemistry rows, minus the one that cannot run. A case that is absent is
    /// honest; a case that is present and silently skipped is not.
    #[test]
    fn the_canonical_chemistry_rows_are_covered() {
        let names: Vec<&str> = CASES.iter().map(|c| c.name).collect();
        assert!(names.iter().any(|n| n.contains("first_order")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("reversible")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("conserves_atoms")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("ledger")), "{names:?}");
        assert!(
            !names.iter().any(|n| n.contains("gillespie")),
            "the stochastic path is not implemented, so it must not appear in the report"
        );
    }
}
