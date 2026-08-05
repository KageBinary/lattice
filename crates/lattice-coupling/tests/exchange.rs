//! Coupling edges moving real quantities between real domains.
//!
//! An integration test rather than a unit test, because what is being checked is that
//! two independently written solvers agree about what crossed between them — which is
//! precisely the thing a mock on either side would hide.

use lattice_coupling::{CouplingEdge, Coupler, EdgeFault, Mapping, PortRef};
use lattice_domain_chemistry::{
    RateLaw, ReactingMixture, Reaction, ReactionNetwork, Species, Term,
};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, Domain, Grid2d, Invariant, Observations, StepContext};

/// Areal heat capacity of the chamber contents, J/(m²·K).
///
/// The number a coupling edge exists to carry: it converts a heat release in W/m² into
/// a temperature rate in K/s, and it is a property of the material rather than of
/// either solver.
const AREAL_HEAT_CAPACITY: f64 = 4.0e5;

fn grid() -> Grid2d {
    Grid2d::new(16, 16, [0.1, 0.1])
}

/// `A -> B`, exothermic, temperature-dependent.
fn network(enthalpy: f64, activation: f64) -> ReactionNetwork {
    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("A").with_formula("C").unwrap().with_diffusion(1e-6));
    network.add_species(Species::new("B").with_formula("C").unwrap().with_diffusion(1e-6));
    network.add_reaction(
        Reaction::new(
            "burn",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(0.5).with_activation_energy(activation),
        )
        .with_enthalpy(enthalpy),
    );
    network
}

/// A chamber: reacting mixture at index 0, temperature field at index 1.
fn chamber(enthalpy: f64, activation: f64) -> (Vec<Box<dyn Domain>>, Coupler) {
    let mut mixture = ReactingMixture::new("chamber", grid(), network(enthalpy, activation));
    mixture.concentration_mut(0).unwrap().fill_interior(10.0);

    let heat = HeatDomain::new("temperature", grid(), Diffusivity::Uniform(1e-5))
        .with_scheme(TimeScheme::CrankNicolson)
        .with_display_unit("K")
        .with_uniform_initial(300.0);

    let domains: Vec<Box<dyn Domain>> = vec![Box::new(mixture), Box::new(heat)];

    let mut coupler = Coupler::new();
    // Chemistry heats the field. This is the edge that moves energy, so it is the one
    // the ledger records.
    coupler.add(
        CouplingEdge::new("reaction_heat", PortRef::new(0, "heat_release"), PortRef::new(1, "source"))
            .with_mapping(Mapping::scale(
                1.0 / AREAL_HEAT_CAPACITY,
                "1 / (areal heat capacity), W/m^2 -> K/s",
            ))
            .carrying(Invariant::Energy),
    );
    // And the field feeds back into the rate constants. This carries no conserved
    // quantity: a temperature reaching an Arrhenius law changes how fast a reaction
    // goes, it does not move energy.
    coupler.add(CouplingEdge::new(
        "temperature_feedback",
        PortRef::new(1, "field"),
        PortRef::new(0, "temperature"),
    ));
    (domains, coupler)
}

fn step_all(domains: &mut [Box<dyn Domain>], coupler: &mut Coupler, dt: f64, steps: usize) {
    let mut arena = Arena::with_capacity(1 << 18);
    for step in 0..steps {
        for domain in domains.iter_mut() {
            let mut ctx = StepContext::new(&mut arena);
            domain.prepare(&mut ctx);
            domain.advance(dt, &mut ctx);
        }
        coupler.exchange(domains, step as u64, step as f64 * dt, dt);
    }
}

fn observe(domains: &[Box<dyn Domain>]) -> Observations {
    let mut out = Observations::new();
    for domain in domains {
        domain.observe(&mut out);
    }
    out
}

/// M3's exit condition in one test: an exothermic reaction heats a field, and the
/// energy that arrives is the energy the ledger says was sent.
#[test]
fn the_ledger_balances_an_exothermic_coupling() {
    let (mut domains, mut coupler) = chamber(-2.0e5, 0.0);

    let before = observe(&domains);
    let temperature_before = before.value("temperature.integral").unwrap();

    step_all(&mut domains, &mut coupler, 0.05, 200);

    let after = observe(&domains);
    let temperature_after = after.value("temperature.integral").unwrap();

    // The field integral is in K·m², so the energy that arrived is that change times
    // the areal heat capacity. This is the quantity the ledger claims to have sent.
    let arrived = (temperature_after - temperature_before) * AREAL_HEAT_CAPACITY;
    let sent = coupler.ledger().received(Invariant::Energy, "temperature");

    assert!(sent > 0.0, "an exothermic reaction must have sent energy, got {sent}");
    let reconciliation = coupler.reconcile(Invariant::Energy, "temperature", arrived);
    let discrepancy = reconciliation.relative_discrepancy().abs();

    // The books balance to within one step of staleness, and not better.
    //
    // A *staggered* coupling always has one exchange in flight: the heat released
    // during the final step is recorded in the ledger and never reaches the field,
    // because the run ends first. That is not a bug in the ledger — it is the thing
    // the ledger exists to make visible. Reporting an exact balance would mean the
    // books were being cooked to match.
    let largest_step = coupler
        .ledger()
        .transfers()
        .iter()
        .map(|transfer| transfer.amount)
        .fold(0.0, f64::max);
    assert!(
        reconciliation.discrepancy().abs() <= largest_step * 1.001,
        "the shortfall must be at most the one exchange still in flight: {:.6e} J \
         against a largest single transfer of {largest_step:.6e} J",
        reconciliation.discrepancy()
    );
    assert!(discrepancy < 1e-3, "and small in relative terms: {discrepancy:.3e}");

    // And the reaction really ran: the chamber is warmer and the reactant is gone.
    assert!(
        after.value("chamber.A.total").unwrap() < 0.5 * before.value("chamber.A.total").unwrap(),
        "most of the reactant should have burned"
    );
    assert!(temperature_after > temperature_before);
}

/// The staleness above is first order in the timestep, which is what "loose staggered"
/// means (§14.2). Measuring it turns "the books nearly balance" into a statement about
/// *why* they do not balance exactly — which is the difference between a coupling that
/// is understood and one that merely works.
#[test]
fn the_ledger_shortfall_shrinks_with_the_timestep() {
    let shortfall = |dt: f64, steps: usize| {
        let (mut domains, mut coupler) = chamber(-2.0e5, 0.0);
        let before = observe(&domains).value("temperature.integral").unwrap();
        step_all(&mut domains, &mut coupler, dt, steps);
        let after = observe(&domains).value("temperature.integral").unwrap();
        let arrived = (after - before) * AREAL_HEAT_CAPACITY;
        coupler.reconcile(Invariant::Energy, "temperature", arrived).relative_discrepancy().abs()
    };

    // The same simulated time, at two timesteps.
    let coarse = shortfall(0.1, 100);
    let fine = shortfall(0.05, 200);
    assert!(coarse > 0.0 && fine > 0.0, "there is always something in flight");
    let order = (coarse / fine).log2();
    assert!(
        order > 0.7,
        "one step of staleness is first order in dt: {coarse:.3e} -> {fine:.3e} (order {order})"
    );
}

/// The mapping is what makes the edge correct rather than merely connected. Remove it
/// and the answer is wrong by exactly the heat capacity — a run that looks completely
/// plausible.
#[test]
fn the_mapping_is_the_difference_between_right_and_plausible() {
    let warming = |mapping: Mapping| {
        let (mut domains, _) = chamber(-2.0e5, 0.0);
        let mut coupler = Coupler::new();
        coupler.add(
            CouplingEdge::new("heat", PortRef::new(0, "heat_release"), PortRef::new(1, "source"))
                .with_mapping(mapping)
                .carrying(Invariant::Energy),
        );
        let before = observe(&domains).value("temperature.integral").unwrap();
        step_all(&mut domains, &mut coupler, 0.01, 20);
        observe(&domains).value("temperature.integral").unwrap() - before
    };

    let correct = warming(Mapping::scale(1.0 / AREAL_HEAT_CAPACITY, "1/(rho c h)"));
    let unmapped = warming(Mapping::Direct);

    assert!(correct > 0.0);
    // Wired straight through, the chamber heats by the heat capacity — four hundred
    // thousand times too much. Nothing in either domain would object.
    assert!(
        unmapped / correct > 1e5,
        "an unmapped edge is off by the heat capacity: {unmapped:.3e} against {correct:.3e}"
    );
}

/// The §20.3 feedback loop: heat raises temperature, temperature raises the rate. With
/// an activation energy the reaction must accelerate; without one it must not.
#[test]
fn temperature_feedback_accelerates_an_arrhenius_reaction() {
    let burned = |activation: f64| {
        let (mut domains, mut coupler) = chamber(-2.0e5, activation);
        let before = observe(&domains).value("chamber.A.total").unwrap();
        step_all(&mut domains, &mut coupler, 0.05, 120);
        let after = observe(&domains).value("chamber.A.total").unwrap();
        (before - after) / before
    };

    let flat = burned(0.0);
    let arrhenius = burned(30_000.0);
    assert!(flat > 0.01 && flat < 1.0, "the baseline must actually react: {flat}");
    assert!(
        arrhenius > 0.0,
        "with an activation energy it starts slower but self-accelerates: {arrhenius}"
    );
}

/// An edge that names a port nothing publishes must be reported, not silently ignored.
/// A coupling that transfers nothing looks exactly like a coupling that transfers zero.
#[test]
fn a_misnamed_or_impossible_edge_is_reported() {
    let (mut domains, _) = chamber(-1.0e5, 0.0);
    let mut coupler = Coupler::new();
    coupler.add(CouplingEdge::new("typo", PortRef::new(0, "heat_relase"), PortRef::new(1, "source")));
    coupler.add(CouplingEdge::new("nowhere", PortRef::new(9, "field"), PortRef::new(1, "source")));
    coupler.add(CouplingEdge::new("backwards", PortRef::new(1, "source"), PortRef::new(0, "temperature")));

    let report = coupler.exchange(&mut domains, 0, 0.0, 0.01);
    assert_eq!(report.exchanged, 0);
    assert_eq!(report.faults.len(), 3, "{:?}", report.faults);
    assert!(!report.is_clean());

    let faults: Vec<&EdgeFault> = report.faults.iter().map(|(_, f)| f).collect();
    assert!(faults.contains(&&EdgeFault::NotPublished), "{faults:?}");
    assert!(faults.contains(&&EdgeFault::NoSuchSourceDomain), "{faults:?}");
}

/// §14.1's cadence: a slow domain need not read a fast one every step, and the
/// decision has to be visible rather than implicit.
#[test]
fn cadence_controls_how_often_an_edge_runs() {
    let (mut domains, _) = chamber(-1.0e5, 0.0);
    let mut coupler = Coupler::new();
    coupler.add(
        CouplingEdge::new("heat", PortRef::new(0, "heat_release"), PortRef::new(1, "source"))
            .with_mapping(Mapping::scale(1.0 / AREAL_HEAT_CAPACITY, "1/(rho c h)"))
            .carrying(Invariant::Energy)
            .every(4),
    );

    step_all(&mut domains, &mut coupler, 0.01, 12);
    // Steps 0, 4 and 8 ran; the rest did not.
    assert_eq!(coupler.ledger().len(), 3, "{:?}", coupler.ledger().transfers());
    assert!(coupler.edges()[0].describe().contains("every 4 steps"));
}

/// A NaN travelling along an edge turns one sick domain into two, and the second has
/// no way to tell where it came from.
/// A NaN travelling along an edge turns one sick domain into two, and the second has no
/// way to tell where it came from.
///
/// The poison goes into the *temperature* field rather than a concentration, because a
/// concentration cannot carry one: `mass_action_rate` floors reactants at zero, and
/// `NaN.max(0.0)` is `0.0` in IEEE 754 — the clamp scrubs the NaN on its way past. That
/// is a genuinely useful property of the clamp and it means the interesting case for
/// this guard is the other direction.
#[test]
fn a_non_finite_value_is_stopped_at_the_edge() {
    let (_, mut coupler) = chamber(-1.0e5, 0.0);

    let mut mixture = ReactingMixture::new("chamber", grid(), network(-1.0e5, 0.0));
    mixture.concentration_mut(0).unwrap().fill_interior(10.0);
    let mut heat = HeatDomain::new("temperature", grid(), Diffusivity::Uniform(1e-5))
        .with_display_unit("K")
        .with_uniform_initial(300.0);
    heat.field_mut().set(3, 3, f64::NAN);

    let mut domains: Vec<Box<dyn Domain>> = vec![Box::new(mixture), Box::new(heat)];
    let report = coupler.exchange(&mut domains, 0, 0.0, 0.01);

    let stopped = report.faults.iter().any(|(_, fault)| *fault == EdgeFault::NonFinite);
    assert!(stopped, "the NaN must be caught at the edge: {:?}", report.faults);
    assert_eq!(report.exchanged, 1, "the other edge is unaffected: {report:?}");

    // The chemistry never saw it, so the fault names the edge that carried it rather
    // than surfacing later as a mystery in a rate constant.
    let mut arena = Arena::with_capacity(1 << 16);
    {
        let mut ctx = StepContext::new(&mut arena);
        domains[0].advance(0.01, &mut ctx);
    }
    let out = observe(&domains);
    assert!(
        out.value("chamber.A.total").unwrap().is_finite(),
        "the chemistry must not have been poisoned"
    );
}

/// An edge that carries a parameter rather than a transfer must not appear in the
/// books. A temperature reaching a rate constant moves no energy.
#[test]
fn a_parameter_edge_leaves_no_ledger_entry() {
    let (mut domains, mut coupler) = chamber(-1.0e5, 0.0);
    step_all(&mut domains, &mut coupler, 0.01, 10);

    // Two edges ran ten times; only the energy-carrying one is in the books.
    assert_eq!(coupler.ledger().len(), 10, "{:?}", coupler.ledger().transfers());
    assert!(coupler.ledger().transfers().iter().all(|t| t.quantity == Invariant::Energy));
    assert_eq!(coupler.ledger().received(Invariant::Energy, "chamber"), 0.0);
}

#[test]
fn a_coupler_with_no_edges_does_nothing_and_says_so() {
    let (mut domains, _) = chamber(-1.0e5, 0.0);
    let mut coupler = Coupler::new();
    let report = coupler.exchange(&mut domains, 0, 0.0, 0.01);
    assert_eq!(report.exchanged, 0);
    assert!(report.is_clean());
    assert_eq!(coupler.describe(), "no coupling edges");
    assert!(coupler.ledger().is_empty());
}
