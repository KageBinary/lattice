//! Reactions and the networks they form.
//!
//! Spec §12.2: *"Reactions declare stoichiometry, reversibility, rate law, equilibrium
//! or kinetic parameters, temperature dependence, catalysts, and enthalpy transfer.
//! The compiler verifies dimensional consistency of rate expressions and, when
//! elemental composition is available, atom and charge balance."*
//!
//! # The balance check is the point
//!
//! `H2 + O2 -> H2O` integrates perfectly happily. Concentrations go down on the left
//! and up on the right, the ODE is well-posed, nothing overflows, and the run looks
//! entirely reasonable. It is also missing an oxygen, and the only symptom is a mass
//! ledger that will not close — several layers and a coupling edge away from the typo.
//!
//! So a reaction whose species all state their composition is checked at construction,
//! and [`Balance`] distinguishes the three outcomes that matter: balanced, unbalanced
//! *with the offending element named*, or unknown because a species did not say.

use std::collections::BTreeMap;

use crate::rate::RateLaw;
use crate::species::Species;

/// One side of a reaction: how much of which species.
///
/// Indices into the network's species list rather than names, because the rate
/// evaluation runs per cell per step and resolving a string there would dominate it.
#[derive(Clone, PartialEq, Debug)]
pub struct Term {
    /// Which species.
    pub species: usize,
    /// How many of it, in the reaction as written. Always positive.
    pub coefficient: f64,
}

impl Term {
    /// `coefficient` of species `index`.
    pub fn new(species: usize, coefficient: f64) -> Term {
        Term { species, coefficient }
    }
}

/// Whether a reaction is balanced, and if not, why not.
#[derive(Clone, PartialEq, Debug)]
pub enum Balance {
    /// Every element and the charge balance.
    Balanced,
    /// An element or the charge does not balance.
    Unbalanced {
        /// What is out: an element symbol, or `"charge"`.
        quantity: String,
        /// How much is on the left.
        reactants: f64,
        /// How much is on the right.
        products: f64,
    },
    /// At least one species did not state its composition, so nothing can be checked.
    Unknown {
        /// The first such species, for the diagnostic.
        species: String,
    },
}

impl Balance {
    /// True only for [`Balance::Balanced`] — an unknown balance is not a passing one.
    pub fn is_balanced(&self) -> bool {
        matches!(self, Balance::Balanced)
    }

    /// A one-line explanation.
    pub fn describe(&self) -> String {
        match self {
            Balance::Balanced => "atom and charge balanced".to_string(),
            Balance::Unbalanced { quantity, reactants, products } => format!(
                "{quantity} does not balance: {reactants} on the left, {products} on the right"
            ),
            Balance::Unknown { species } => {
                format!("cannot be checked — `{species}` did not state its composition")
            }
        }
    }
}

/// A chemical reaction.
#[derive(Clone, PartialEq, Debug)]
pub struct Reaction {
    /// The name the model gave it.
    pub name: String,
    /// What is consumed.
    pub reactants: Vec<Term>,
    /// What is produced.
    pub products: Vec<Term>,
    /// How fast it goes.
    pub rate: RateLaw,
    /// Enthalpy of reaction, J per mole of extent.
    ///
    /// Negative is exothermic — the convention chemistry uses and the opposite of what
    /// "releases energy" suggests, so it is stated here rather than left to be assumed.
    /// This is the number the heat coupling consumes.
    pub enthalpy: f64,
}

impl Reaction {
    /// A reaction from its two sides and a rate law.
    pub fn new(
        name: impl Into<String>,
        reactants: Vec<Term>,
        products: Vec<Term>,
        rate: RateLaw,
    ) -> Reaction {
        Reaction { name: name.into(), reactants, products, rate, enthalpy: 0.0 }
    }

    /// Set the enthalpy of reaction, J/mol. Negative is exothermic.
    pub fn with_enthalpy(mut self, enthalpy: f64) -> Reaction {
        self.enthalpy = enthalpy;
        self
    }

    /// The net change in each species per unit of reaction extent.
    ///
    /// Positive for something produced, negative for something consumed. A species
    /// appearing on both sides — a catalyst, or a reagent that is partly regenerated —
    /// gets the difference, which is the whole reason this is computed rather than
    /// read off one side.
    pub fn stoichiometry(&self, species_count: usize) -> Vec<f64> {
        let mut net = vec![0.0; species_count];
        for term in &self.reactants {
            if term.species < species_count {
                net[term.species] -= term.coefficient;
            }
        }
        for term in &self.products {
            if term.species < species_count {
                net[term.species] += term.coefficient;
            }
        }
        net
    }

    /// Check atom and charge balance against the species table.
    pub fn balance(&self, species: &[Species]) -> Balance {
        let mut totals: BTreeMap<String, f64> = BTreeMap::new();
        let mut charge = 0.0;

        for (terms, sign) in [(&self.reactants, -1.0), (&self.products, 1.0)] {
            for term in terms {
                let Some(entry) = species.get(term.species) else { continue };
                if entry.composition.is_unknown() {
                    return Balance::Unknown { species: entry.name.clone() };
                }
                for (element, count) in entry.composition.atoms() {
                    *totals.entry(element.to_string()).or_insert(0.0) +=
                        sign * term.coefficient * f64::from(count);
                }
                charge += sign * term.coefficient * f64::from(entry.charge);
            }
        }

        // A tolerance, because coefficients may be fractional — `H2 + ½O2 -> H2O` is a
        // perfectly ordinary way to write a reaction, and its sums are not exact.
        const TOLERANCE: f64 = 1e-9;

        for (element, net) in &totals {
            if net.abs() > TOLERANCE {
                let (left, right) = self.element_totals(species, element);
                return Balance::Unbalanced {
                    quantity: element.clone(),
                    reactants: left,
                    products: right,
                };
            }
        }
        if charge.abs() > TOLERANCE {
            let (left, right) = self.charge_totals(species);
            return Balance::Unbalanced {
                quantity: "charge".to_string(),
                reactants: left,
                products: right,
            };
        }
        Balance::Balanced
    }

    fn element_totals(&self, species: &[Species], element: &str) -> (f64, f64) {
        let side = |terms: &[Term]| {
            terms
                .iter()
                .filter_map(|term| species.get(term.species).map(|s| (term, s)))
                .map(|(term, s)| term.coefficient * f64::from(s.composition.count(element)))
                .sum()
        };
        (side(&self.reactants), side(&self.products))
    }

    fn charge_totals(&self, species: &[Species]) -> (f64, f64) {
        let side = |terms: &[Term]| {
            terms
                .iter()
                .filter_map(|term| species.get(term.species).map(|s| (term, s)))
                .map(|(term, s)| term.coefficient * f64::from(s.charge))
                .sum()
        };
        (side(&self.reactants), side(&self.products))
    }

    /// The reaction as chemists write it, using the species table for names.
    pub fn equation(&self, species: &[Species]) -> String {
        let side = |terms: &[Term]| {
            if terms.is_empty() {
                return "(nothing)".to_string();
            }
            terms
                .iter()
                .map(|term| {
                    let name = species
                        .get(term.species)
                        .map_or_else(|| format!("#{}", term.species), |s| s.name.clone());
                    if (term.coefficient - 1.0).abs() < 1e-12 {
                        name
                    } else {
                        format!("{} {name}", term.coefficient)
                    }
                })
                .collect::<Vec<_>>()
                .join(" + ")
        };
        let arrow = if self.rate.is_reversible() { "<->" } else { "->" };
        format!("{} {arrow} {}", side(&self.reactants), side(&self.products))
    }

    /// Total order of the forward step — the sum of the reactant coefficients.
    ///
    /// What sets the units of the rate constant: an order-`n` constant is in
    /// `(m²/mol)^(n−1)/s`. A model that writes a second-order constant in `1/s` is
    /// wrong by a factor with the dimensions of a concentration, and the compiler
    /// checks it.
    pub fn order(&self) -> f64 {
        self.reactants.iter().map(|term| term.coefficient).sum()
    }
}

/// A set of species and the reactions among them.
#[derive(Clone, PartialEq, Default, Debug)]
pub struct ReactionNetwork {
    species: Vec<Species>,
    reactions: Vec<Reaction>,
}

impl ReactionNetwork {
    /// An empty network.
    pub fn new() -> ReactionNetwork {
        ReactionNetwork::default()
    }

    /// Add a species, returning the index reactions refer to it by.
    pub fn add_species(&mut self, species: Species) -> usize {
        self.species.push(species);
        self.species.len() - 1
    }

    /// Find a species by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.species.iter().position(|s| s.name == name)
    }

    /// Add a reaction.
    pub fn add_reaction(&mut self, reaction: Reaction) {
        self.reactions.push(reaction);
    }

    /// The species, in the order concentration vectors use.
    pub fn species(&self) -> &[Species] {
        &self.species
    }

    /// The reactions.
    pub fn reactions(&self) -> &[Reaction] {
        &self.reactions
    }

    /// The reactions, mutably — for a caller adjusting a rate constant or an enthalpy
    /// after construction, which is what a parameter sweep does.
    pub fn reactions_mut(&mut self) -> &mut [Reaction] {
        &mut self.reactions
    }

    /// How many species.
    pub fn len(&self) -> usize {
        self.species.len()
    }

    /// True when nothing is declared.
    pub fn is_empty(&self) -> bool {
        self.species.is_empty()
    }

    /// Every reaction that does not balance, with the reason.
    ///
    /// Returned rather than reported, so a compiler can turn each into a diagnostic at
    /// the right source position and a library caller can decide for itself.
    pub fn imbalances(&self) -> Vec<(&Reaction, Balance)> {
        self.reactions
            .iter()
            .map(|reaction| (reaction, reaction.balance(&self.species)))
            .filter(|(_, balance)| !balance.is_balanced())
            .collect()
    }

    /// Total mass of a concentration vector, kg/m².
    ///
    /// The quantity a balanced network conserves exactly and an unbalanced one does
    /// not. Zero when no species states a molar mass, which is why the ledger reports
    /// what it could account for rather than a bare number.
    pub fn total_mass(&self, concentrations: &[f64]) -> f64 {
        self.species
            .iter()
            .zip(concentrations)
            .map(|(species, amount)| species.molar_mass * amount)
            .sum()
    }

    /// Total charge of a concentration vector, in elementary charges per m².
    pub fn total_charge(&self, concentrations: &[f64]) -> f64 {
        self.species
            .iter()
            .zip(concentrations)
            .map(|(species, amount)| f64::from(species.charge) * amount)
            .sum()
    }

    /// Total amount of one element, mol/m².
    ///
    /// The sharper conservation check: mass can balance while atoms do not if two
    /// species happen to have compatible molar masses.
    pub fn total_element(&self, element: &str, concentrations: &[f64]) -> f64 {
        self.species
            .iter()
            .zip(concentrations)
            .map(|(species, amount)| f64::from(species.composition.count(element)) * amount)
            .sum()
    }

    /// Every element any species contains.
    pub fn elements(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for species in &self.species {
            for (element, _) in species.composition.atoms() {
                if !seen.iter().any(|e| e == element) {
                    seen.push(element.to_string());
                }
            }
        }
        seen.sort();
        seen
    }

    /// True when every species states its composition, so the network can be checked.
    pub fn is_fully_specified(&self) -> bool {
        !self.species.is_empty() && self.species.iter().all(|s| !s.composition.is_unknown())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate::RateLaw;

    fn water_network() -> ReactionNetwork {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("H2").with_formula("H2").unwrap());
        network.add_species(Species::new("O2").with_formula("O2").unwrap());
        network.add_species(Species::new("H2O").with_formula("H2O").unwrap());
        network
    }

    #[test]
    fn a_balanced_reaction_is_recognized() {
        let mut network = water_network();
        // 2 H2 + O2 -> 2 H2O.
        network.add_reaction(Reaction::new(
            "combustion",
            vec![Term::new(0, 2.0), Term::new(1, 1.0)],
            vec![Term::new(2, 2.0)],
            RateLaw::mass_action(1.0),
        ));
        assert_eq!(network.reactions()[0].balance(network.species()), Balance::Balanced);
        assert!(network.imbalances().is_empty());
    }

    /// The reaction this whole check exists for: it integrates perfectly happily and
    /// is missing an oxygen.
    #[test]
    fn an_unbalanced_reaction_names_the_element() {
        let mut network = water_network();
        network.add_reaction(Reaction::new(
            "typo",
            vec![Term::new(0, 1.0), Term::new(1, 1.0)],
            vec![Term::new(2, 1.0)],
            RateLaw::mass_action(1.0),
        ));
        match network.reactions()[0].balance(network.species()) {
            Balance::Unbalanced { quantity, reactants, products } => {
                assert_eq!(quantity, "O");
                assert_eq!(reactants, 2.0, "O2 on the left");
                assert_eq!(products, 1.0, "one O in H2O");
            }
            other => panic!("expected an oxygen imbalance, got {other:?}"),
        }
        assert_eq!(network.imbalances().len(), 1);
    }

    #[test]
    fn charge_is_balanced_too() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("H_plus").with_formula("H").unwrap().with_charge(1));
        network.add_species(
            Species::new("OH_minus").with_formula("OH").unwrap().with_charge(-1),
        );
        network.add_species(Species::new("H2O").with_formula("H2O").unwrap());

        // The real reaction balances in atoms and charge.
        network.add_reaction(Reaction::new(
            "neutralization",
            vec![Term::new(0, 1.0), Term::new(1, 1.0)],
            vec![Term::new(2, 1.0)],
            RateLaw::mass_action(1.4e11),
        ));
        assert_eq!(network.reactions()[0].balance(network.species()), Balance::Balanced);

        // Dropping the hydroxide's charge leaves the atoms balanced and the charge not,
        // which is exactly the case an atom-only check would pass.
        let mut broken = network.clone();
        broken.species[1].charge = 0;
        match broken.reactions()[0].balance(broken.species()) {
            Balance::Unbalanced { quantity, reactants, products } => {
                assert_eq!(quantity, "charge");
                assert_eq!((reactants, products), (1.0, 0.0));
            }
            other => panic!("expected a charge imbalance, got {other:?}"),
        }
    }

    /// A species with no composition means the check cannot run, and saying so is
    /// different from saying the reaction is fine.
    #[test]
    fn an_unstated_composition_is_unknown_rather_than_balanced() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A"));
        network.add_species(Species::new("B").with_formula("O2").unwrap());
        network.add_reaction(Reaction::new(
            "vague",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(1.0),
        ));

        let balance = network.reactions()[0].balance(network.species());
        assert!(matches!(balance, Balance::Unknown { .. }));
        assert!(!balance.is_balanced(), "unknown is not a pass");
        assert!(balance.describe().contains("did not state its composition"));
        assert!(!network.is_fully_specified());
        // And it is reported, because silence would read as approval.
        assert_eq!(network.imbalances().len(), 1);
    }

    /// Half-integer coefficients are ordinary chemistry, and their sums are not exact.
    #[test]
    fn fractional_coefficients_balance() {
        let mut network = water_network();
        // H2 + ½O2 -> H2O.
        network.add_reaction(Reaction::new(
            "half",
            vec![Term::new(0, 1.0), Term::new(1, 0.5)],
            vec![Term::new(2, 1.0)],
            RateLaw::mass_action(1.0),
        ));
        assert_eq!(network.reactions()[0].balance(network.species()), Balance::Balanced);
    }

    /// A catalyst appears on both sides and nets to zero, which is what makes it a
    /// catalyst. Reading the stoichiometry off one side would consume it.
    #[test]
    fn a_species_on_both_sides_nets_to_its_difference() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("S").with_formula("C").unwrap());
        network.add_species(Species::new("P").with_formula("C").unwrap());
        network.add_species(Species::new("E").with_formula("Fe").unwrap());

        let reaction = Reaction::new(
            "catalysed",
            vec![Term::new(0, 1.0), Term::new(2, 1.0)],
            vec![Term::new(1, 1.0), Term::new(2, 1.0)],
            RateLaw::mass_action(1.0),
        );
        let net = reaction.stoichiometry(3);
        assert_eq!(net, vec![-1.0, 1.0, 0.0], "the catalyst is not consumed");
        assert_eq!(reaction.balance(network.species()), Balance::Balanced);
        // But it still counts toward the order, because the rate depends on it.
        assert_eq!(reaction.order(), 2.0);
    }

    #[test]
    fn conserved_totals_follow_from_the_stoichiometry() {
        let mut network = water_network();
        network.add_reaction(Reaction::new(
            "combustion",
            vec![Term::new(0, 2.0), Term::new(1, 1.0)],
            vec![Term::new(2, 2.0)],
            RateLaw::mass_action(1.0),
        ));

        // Start with 2 mol/m^2 of H2 and 1 of O2; run the reaction to completion.
        let before = [2.0, 1.0, 0.0];
        let after = [0.0, 0.0, 2.0];

        assert!((network.total_mass(&before) - network.total_mass(&after)).abs() < 1e-12);
        assert!((network.total_element("H", &before) - network.total_element("H", &after)).abs() < 1e-12);
        assert!((network.total_element("O", &before) - network.total_element("O", &after)).abs() < 1e-12);
        assert_eq!(network.elements(), vec!["H".to_string(), "O".to_string()]);
    }

    /// Mass can balance while atoms do not, if two species have compatible molar
    /// masses. The element totals are the sharper check.
    #[test]
    fn element_totals_catch_what_mass_alone_would_miss() {
        let mut network = ReactionNetwork::new();
        // Two species of identical molar mass and different composition: CO and N2 are
        // both 28 g/mol.
        network.add_species(Species::new("CO").with_formula("CO").unwrap());
        network.add_species(Species::new("N2").with_formula("N2").unwrap());

        let before = [1.0, 0.0];
        let after = [0.0, 1.0];
        assert!(
            (network.total_mass(&before) - network.total_mass(&after)).abs() < 1e-3,
            "CO and N2 weigh the same to a tenth of a percent, so mass says nothing"
        );
        assert!(
            (network.total_element("C", &before) - network.total_element("C", &after)).abs() > 0.9,
            "the carbon vanished, and the element total says so"
        );
    }

    #[test]
    fn equations_render_the_way_chemists_write_them() {
        let mut network = water_network();
        network.add_reaction(Reaction::new(
            "combustion",
            vec![Term::new(0, 2.0), Term::new(1, 1.0)],
            vec![Term::new(2, 2.0)],
            RateLaw::mass_action(1.0),
        ));
        assert_eq!(network.reactions()[0].equation(network.species()), "2 H2 + O2 -> 2 H2O");

        network.add_reaction(Reaction::new(
            "reversible",
            vec![Term::new(2, 1.0)],
            vec![Term::new(0, 1.0)],
            RateLaw::reversible(1.0, 2.0),
        ));
        assert!(network.reactions()[1].equation(network.species()).contains("<->"));
    }

    #[test]
    fn an_empty_network_reports_nothing_rather_than_panicking() {
        let network = ReactionNetwork::new();
        assert!(network.is_empty());
        assert!(network.imbalances().is_empty());
        assert!(!network.is_fully_specified(), "nothing to specify is not fully specified");
        assert_eq!(network.total_mass(&[]), 0.0);
        assert!(network.elements().is_empty());
    }

    /// A term pointing past the species table must be ignored rather than panicking:
    /// the compiler reports the unknown name and keeps going to find the rest.
    #[test]
    fn a_dangling_species_index_does_not_panic() {
        let network = water_network();
        let reaction = Reaction::new(
            "dangling",
            vec![Term::new(99, 1.0)],
            vec![Term::new(2, 1.0)],
            RateLaw::mass_action(1.0),
        );
        assert_eq!(reaction.stoichiometry(3), vec![0.0, 0.0, 1.0]);
        // The balance skips what it cannot resolve rather than inventing an answer.
        assert!(!reaction.balance(network.species()).is_balanced());
    }
}
