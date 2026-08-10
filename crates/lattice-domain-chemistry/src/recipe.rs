//! Reaction recipes: a species list, a reaction, and the conditions to run it under.
//!
//! A [`ReactionNetwork`] is what the solver wants. A [`Recipe`] is what a *reader* wants —
//! the same information plus a name, a description, and the starting conditions that make
//! the reaction do something visible. The sandbox builds networks from these, and so can a
//! test.
//!
//! # What a preset is and is not
//!
//! Every reaction in [`presets`] is real: real reactants, real products, real stoichiometry,
//! and an enthalpy of reaction close to the tabulated value. **The rate constants are not.**
//!
//! That is worth being blunt about. A real rate constant depends on phase, pressure,
//! catalyst, surface area and mechanism, and most of these reactions do not proceed by a
//! single elementary step at all — methane does not meet two oxygen molecules and become
//! carbon dioxide, it goes through a radical chain with dozens of steps. What the presets
//! carry is a *plausible mass-action caricature*: the right species, the right balance, the
//! right sign and rough size of heat release, and a rate chosen so the reaction is
//! interesting to watch on a 72-cell grid in a few seconds.
//!
//! [`Recipe::fidelity`] says which of those two things you are looking at, and the sandbox
//! prints it. A number that is illustrative and labelled illustrative is a teaching tool; the
//! same number unlabelled is a lie with a decimal point in it.

use crate::network::{Reaction, ReactionNetwork, Term};
use crate::rate::{RateLaw, GAS_CONSTANT};
use crate::species::Species;

/// The temperature a recipe's [`Recipe::rate`] is quoted at, K.
///
/// Room temperature, and the sandbox's ambient. See [`Recipe::network`] for why a recipe
/// states a rate *at a temperature* rather than a pre-exponential factor.
pub const REFERENCE_TEMPERATURE: f64 = 300.0;

/// The concentration a recipe's [`Recipe::rate`] is quoted at, mol/m².
///
/// A working amount of material — what one stroke of the sandbox's brush lays down. See
/// [`Recipe::network`] for why a rate has to be quoted at a concentration as well as a
/// temperature.
pub const REFERENCE_CONCENTRATION: f64 = 60.0;

/// How much weight a recipe's numbers will bear.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fidelity {
    /// Stoichiometry and enthalpy are real; the rate constant is chosen to be watchable.
    Illustrative,
    /// Everything, including the kinetics, is from a measurement.
    ///
    /// Nothing carries this yet. It exists so that adding a properly sourced reaction is a
    /// change of value rather than a change of design, and so the absence is visible.
    Measured,
}

impl Fidelity {
    /// The sentence the sandbox shows beside a recipe.
    pub const fn caveat(self) -> &'static str {
        match self {
            Fidelity::Illustrative => {
                "balance and heat of reaction are real; the rate constant is chosen to be \
                 watchable, not measured"
            }
            Fidelity::Measured => "kinetics from a cited measurement",
        }
    }
}

/// One species in a recipe, before it becomes an index.
#[derive(Clone, Debug)]
pub struct Ingredient {
    /// Display name, e.g. `"methane"`.
    pub name: String,
    /// Chemical formula, e.g. `"CH4"`.
    pub formula: String,
    /// Diffusion coefficient, m²/s.
    pub diffusion: f64,
    /// Stoichiometric coefficient. Negative consumes, positive produces, zero is a
    /// spectator.
    pub coefficient: f64,
}

impl Ingredient {
    /// A reactant, consumed with the given coefficient.
    pub fn consumes(name: &str, formula: &str, coefficient: f64) -> Ingredient {
        Ingredient {
            name: name.to_string(),
            formula: formula.to_string(),
            diffusion: 4e-4,
            coefficient: -coefficient,
        }
    }

    /// A product, made with the given coefficient.
    pub fn produces(name: &str, formula: &str, coefficient: f64) -> Ingredient {
        Ingredient {
            name: name.to_string(),
            formula: formula.to_string(),
            diffusion: 2e-4,
            coefficient,
        }
    }

    /// Override the diffusion coefficient.
    pub fn with_diffusion(mut self, diffusion: f64) -> Ingredient {
        self.diffusion = diffusion;
        self
    }
}

/// A named reaction with everything needed to run it.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// Short name for a menu.
    pub name: String,
    /// One line on what it is and where it is seen.
    pub description: String,
    /// The species, reactants first.
    pub ingredients: Vec<Ingredient>,
    /// Pseudo-first-order rate at [`REFERENCE_TEMPERATURE`] and
    /// [`REFERENCE_CONCENTRATION`], 1/s. See [`Recipe::network`] for why it is quoted
    /// this way and not as a mass-action constant.
    pub rate: f64,
    /// Arrhenius activation energy, J/mol. Zero for a temperature-independent rate.
    pub activation_energy: f64,
    /// Enthalpy of reaction, J/mol. Negative releases heat.
    pub enthalpy: f64,
    /// How much the numbers can be trusted.
    pub fidelity: Fidelity,
}

impl Recipe {
    /// The species that are consumed, by index into [`Recipe::ingredients`].
    pub fn reactants(&self) -> Vec<usize> {
        self.indices(|coefficient| coefficient < 0.0)
    }

    /// The species that are produced.
    pub fn products(&self) -> Vec<usize> {
        self.indices(|coefficient| coefficient > 0.0)
    }

    fn indices(&self, wanted: impl Fn(f64) -> bool) -> Vec<usize> {
        self.ingredients
            .iter()
            .enumerate()
            .filter(|(_, ingredient)| wanted(ingredient.coefficient))
            .map(|(index, _)| index)
            .collect()
    }

    /// Build the network this recipe describes.
    ///
    /// `rate` and `activation_energy` are passed in rather than read from the recipe, because
    /// the sandbox puts both on sliders and the recipe's own values are only the starting
    /// point. Everything else — which species exist, what they are made of, how many of each
    /// the reaction takes and makes — comes from the recipe and is not adjustable, because
    /// those are what make it *this* reaction rather than another one.
    ///
    /// # `rate` is `k` at [`REFERENCE_TEMPERATURE`], not the pre-exponential factor
    ///
    /// [`RateLaw`] takes the declared constant as `A` in `k(T) = A·exp(−Ea/RT)`, which is
    /// the right primitive and the wrong thing to put in front of a reader. `exp(−Ea/RT)`
    /// at 300 K is `4×10⁻⁵` for an activation energy of 25 kJ/mol and `1×10⁻⁷` for one of
    /// 40 kJ/mol, so two recipes with the same `A` differ in speed by a factor of 400 —
    /// and a menu of reactions specified that way has entries that visibly do nothing,
    /// for a reason invisible in the number. That is not hypothetical: it is what
    /// `every_preset_reacts_visibly_at_its_own_default_rate` caught.
    ///
    /// So a recipe quotes `k` at room temperature — "how fast is this *here*", which is
    /// comparable between reactions and is what a slider should move — and the
    /// pre-exponential is derived: `A = k(T₀)·exp(Ea/RT₀)`.
    ///
    /// The activation energy keeps its real meaning either way. It no longer sets how fast
    /// the reaction goes at room temperature; it sets how sharply it *accelerates* when
    /// heated, which is the question the Arrhenius slider is actually interesting for.
    ///
    /// # …and at [`REFERENCE_CONCENTRATION`], because order is the other trap
    ///
    /// A mass-action constant's units are `(m²/mol)^(order−1)/s`, so the same number means
    /// different speeds for reactions of different order. `4 Fe + 3 O₂ → 2 Fe₂O₃` is
    /// seventh order; at 60 mol/m² its rate carries a factor of `60⁷ ≈ 3×10¹²` that
    /// `2 H₂ + O₂` — third order — does not. Two recipes quoted with the same `k` are then
    /// not remotely comparable, and one of them consumes its reactants in a single step.
    ///
    /// So `rate` is quoted as a **pseudo-first-order constant at the reference
    /// concentration**, in `1/s`: the reaction's initial rate is `rate · c₀`, whatever its
    /// order. The mass-action constant is derived by dividing out the concentration factor
    /// the order implies. `rate = 6` then means the same speed in every recipe, and a
    /// reader comparing two menu entries is comparing something real.
    pub fn network(&self, rate: f64, activation_energy: f64) -> ReactionNetwork {
        let mut network = ReactionNetwork::new();
        for ingredient in &self.ingredients {
            let species = Species::new(ingredient.name.clone())
                .with_formula(&ingredient.formula)
                .unwrap_or_else(|error| {
                    panic!("preset {} has an unparseable formula: {error:?}", self.name)
                })
                .with_diffusion(ingredient.diffusion);
            network.add_species(species);
        }

        let terms = |wanted: fn(f64) -> bool| -> Vec<Term> {
            self.ingredients
                .iter()
                .enumerate()
                .filter(|(_, ingredient)| wanted(ingredient.coefficient))
                .map(|(index, ingredient)| Term::new(index, ingredient.coefficient.abs()))
                .collect()
        };

        // Two normalizations, both explained above. Divide out the concentration factor the
        // reaction's order implies, then undo the Arrhenius suppression at room temperature:
        //
        //     A = rate · c₀^(1−order) · exp(Ea / R T₀)
        let order = self.order();
        let mass_action_constant = rate * REFERENCE_CONCENTRATION.powf(1.0 - order);
        let pre_exponential = mass_action_constant
            * (activation_energy / (GAS_CONSTANT * REFERENCE_TEMPERATURE)).exp();

        network.add_reaction(
            Reaction::new(
                self.name.clone(),
                terms(|c| c < 0.0),
                terms(|c| c > 0.0),
                RateLaw::mass_action(pre_exponential)
                    .with_activation_energy(activation_energy),
            )
            .with_enthalpy(self.enthalpy),
        );
        network
    }

    /// The network at the recipe's own rate and activation energy.
    pub fn default_network(&self) -> ReactionNetwork {
        self.network(self.rate, self.activation_energy)
    }

    /// The reaction's order: the sum of its reactants' stoichiometric coefficients.
    ///
    /// Mass action takes every reactant to the power of its coefficient, so this is the
    /// exponent the concentration enters the rate at — and the reason a rate constant is
    /// not comparable between reactions unless it is normalized. See [`Recipe::network`].
    pub fn order(&self) -> f64 {
        self.ingredients
            .iter()
            .filter(|ingredient| ingredient.coefficient < 0.0)
            .map(|ingredient| -ingredient.coefficient)
            .sum()
    }
}

/// The built-in reactions.
///
/// Chosen for three properties, in this order: every one is a reaction somebody has heard
/// of; between them they cover exothermic and endothermic, fast and slow, and one, two and
/// three products; and every one balances, which is checked rather than asserted.
pub fn presets() -> Vec<Recipe> {
    vec![
        Recipe {
            name: "hydrogen + oxygen".to_string(),
            description: "2 H₂ + O₂ → 2 H₂O — the reaction that lifts rockets, and the \
                          most exothermic thing here per kilogram"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("hydrogen", "H2", 2.0).with_diffusion(8e-4),
                Ingredient::consumes("oxygen", "O2", 1.0),
                Ingredient::produces("water", "H2O", 2.0),
            ],
            rate: 6.0,
            activation_energy: 25_000.0,
            enthalpy: -483_600.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "methane combustion".to_string(),
            description: "CH₄ + 2 O₂ → CO₂ + 2 H₂O — burning natural gas; two products, \
                          so watch both fill in"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("methane", "CH4", 1.0),
                Ingredient::consumes("oxygen", "O2", 2.0),
                Ingredient::produces("carbon dioxide", "CO2", 1.0),
                Ingredient::produces("water", "H2O", 2.0),
            ],
            rate: 3.0,
            activation_energy: 40_000.0,
            enthalpy: -890_300.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "rusting iron".to_string(),
            description: "4 Fe + 3 O₂ → 2 Fe₂O₃ — slow, and the reason the activation \
                          energy slider matters"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("iron", "Fe", 4.0).with_diffusion(2e-5),
                Ingredient::consumes("oxygen", "O2", 3.0),
                Ingredient::produces("iron(III) oxide", "Fe2O3", 2.0).with_diffusion(1e-5),
            ],
            rate: 0.2,
            activation_energy: 60_000.0,
            enthalpy: -824_200.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "table salt".to_string(),
            description: "2 Na + Cl₂ → 2 NaCl — a metal and a poison gas making \
                          something you eat"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("sodium", "Na", 2.0).with_diffusion(1e-4),
                Ingredient::consumes("chlorine", "Cl2", 1.0),
                Ingredient::produces("sodium chloride", "NaCl", 2.0),
            ],
            rate: 12.0,
            activation_energy: 12_000.0,
            enthalpy: -822_000.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "ammonia (Haber)".to_string(),
            description: "N₂ + 3 H₂ → 2 NH₃ — feeds about half the world, and needs \
                          real pressure and a catalyst it does not have here"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("nitrogen", "N2", 1.0),
                Ingredient::consumes("hydrogen", "H2", 3.0).with_diffusion(8e-4),
                Ingredient::produces("ammonia", "NH3", 2.0),
            ],
            rate: 0.8,
            activation_energy: 55_000.0,
            enthalpy: -92_200.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "hydrogen chloride".to_string(),
            description: "H₂ + Cl₂ → 2 HCl — a chain reaction in daylight; here it is \
                          simply fast"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("hydrogen", "H2", 1.0).with_diffusion(8e-4),
                Ingredient::consumes("chlorine", "Cl2", 1.0),
                Ingredient::produces("hydrogen chloride", "HCl", 2.0),
            ],
            rate: 15.0,
            activation_energy: 8_000.0,
            enthalpy: -184_600.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "burning magnesium".to_string(),
            description: "2 Mg + O₂ → 2 MgO — the flare that is hard to look at"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("magnesium", "Mg", 2.0).with_diffusion(3e-5),
                Ingredient::consumes("oxygen", "O2", 1.0),
                Ingredient::produces("magnesium oxide", "MgO", 2.0).with_diffusion(1e-5),
            ],
            rate: 9.0,
            activation_energy: 30_000.0,
            enthalpy: -1_203_600.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "quicklime + water".to_string(),
            description: "CaO + H₂O → Ca(OH)₂ — slaking lime; enough heat to boil the \
                          water you added"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("quicklime", "CaO", 1.0).with_diffusion(2e-5),
                Ingredient::consumes("water", "H2O", 1.0),
                // Ca(OH)₂. Written flat because the formula parser has no parentheses —
                // and `CaOH2` would be Ca-O-H₂, one oxygen short, which is exactly what
                // `every_preset_balances_its_atoms_and_charge` caught.
                Ingredient::produces("slaked lime", "CaO2H2", 1.0).with_diffusion(1e-5),
            ],
            rate: 7.0,
            activation_energy: 15_000.0,
            enthalpy: -63_700.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "iron + sulfur".to_string(),
            description: "Fe + S → FeS — the classic school demonstration of a compound \
                          that is nothing like its ingredients"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("iron", "Fe", 1.0).with_diffusion(2e-5),
                Ingredient::consumes("sulfur", "S", 1.0).with_diffusion(2e-5),
                Ingredient::produces("iron(II) sulfide", "FeS", 1.0).with_diffusion(1e-5),
            ],
            rate: 3.0,
            activation_energy: 45_000.0,
            enthalpy: -100_000.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "thermite".to_string(),
            description: "2 Al + Fe₂O₃ → Al₂O₃ + 2 Fe — hot enough to weld rail; \
                          needs a fierce ignition, then feeds itself"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("aluminium", "Al", 2.0).with_diffusion(2e-5),
                Ingredient::consumes("iron(III) oxide", "Fe2O3", 1.0).with_diffusion(1e-5),
                Ingredient::produces("aluminium oxide", "Al2O3", 1.0).with_diffusion(1e-5),
                Ingredient::produces("iron", "Fe", 2.0).with_diffusion(2e-5),
            ],
            rate: 4.0,
            activation_energy: 60_000.0,
            enthalpy: -851_500.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "nitrogen monoxide".to_string(),
            description: "N₂ + O₂ → 2 NO — endothermic, so it *cools* the chamber; \
                          happens in engines and lightning"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("nitrogen", "N2", 1.0),
                Ingredient::consumes("oxygen", "O2", 1.0),
                Ingredient::produces("nitrogen monoxide", "NO", 2.0),
            ],
            rate: 5.0,
            activation_energy: 50_000.0,
            enthalpy: 180_500.0,
            fidelity: Fidelity::Illustrative,
        },
        Recipe {
            name: "carbon monoxide".to_string(),
            description: "2 C + O₂ → 2 CO — incomplete combustion, and what the \
                          original sandbox reaction actually was"
                .to_string(),
            ingredients: vec![
                Ingredient::consumes("carbon", "C", 2.0).with_diffusion(4e-4),
                Ingredient::consumes("oxygen", "O2", 1.0),
                Ingredient::produces("carbon monoxide", "CO", 2.0),
            ],
            rate: 6.0,
            activation_energy: 25_000.0,
            enthalpy: -221_000.0,
            fidelity: Fidelity::Illustrative,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one property a preset library must have. An unbalanced reaction would create or
    /// destroy atoms, and the sandbox's conservation panel would report the *solver* for it.
    #[test]
    fn every_preset_balances_its_atoms_and_charge() {
        for recipe in presets() {
            let network = recipe.default_network();
            let imbalances = network.imbalances();
            assert!(
                imbalances.is_empty(),
                "{} does not balance: {:?}",
                recipe.name,
                imbalances
                    .iter()
                    .map(|(reaction, balance)| (reaction.name.clone(), balance.describe()))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn every_preset_has_reactants_and_products() {
        for recipe in presets() {
            assert!(!recipe.reactants().is_empty(), "{} consumes nothing", recipe.name);
            assert!(!recipe.products().is_empty(), "{} produces nothing", recipe.name);
            assert!(
                recipe.ingredients.iter().all(|i| i.coefficient != 0.0),
                "{} has a spectator with no coefficient",
                recipe.name
            );
        }
    }

    #[test]
    fn every_preset_is_fully_specified_and_named() {
        for recipe in presets() {
            assert!(!recipe.name.is_empty());
            assert!(!recipe.description.is_empty(), "{} has no description", recipe.name);
            assert!(recipe.rate > 0.0, "{} has a non-positive rate", recipe.name);
            assert!(recipe.activation_energy >= 0.0, "{} has a negative Ea", recipe.name);
            let network = recipe.default_network();
            assert!(
                network.is_fully_specified(),
                "{} has a species the engine cannot weigh",
                recipe.name
            );
        }
    }

    #[test]
    fn preset_names_are_unique() {
        let mut names: Vec<_> = presets().into_iter().map(|r| r.name).collect();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count, "two presets share a name");
    }

    /// The normalization that makes a menu of reactions comparable: two recipes with the
    /// same quoted rate start at the same speed, whatever their order.
    #[test]
    fn the_quoted_rate_means_the_same_thing_at_every_reaction_order() {
        let third = presets().into_iter().find(|r| r.order() == 3.0).expect("a third-order");
        let seventh = presets().into_iter().find(|r| r.order() == 7.0).expect("a seventh-order");

        // Initial rate at reference concentration is k·c0^order for a mass-action law.
        let initial = |recipe: &Recipe| {
            let network = recipe.network(1.0, 0.0);
            let k = network.reactions()[0].rate.forward_at(REFERENCE_TEMPERATURE);
            k * REFERENCE_CONCENTRATION.powf(recipe.order())
        };

        let (a, b) = (initial(&third), initial(&seventh));
        assert!(
            (a / b - 1.0).abs() < 1e-9,
            "order {} started at {a} and order {} at {b}",
            third.order(),
            seventh.order()
        );
        assert!((a - REFERENCE_CONCENTRATION).abs() < 1e-9, "rate 1/s times c0");
    }

    /// The library is meant to span the interesting cases, not just repeat one shape.
    #[test]
    fn the_library_covers_more_than_one_kind_of_reaction() {
        let all = presets();
        assert!(all.len() >= 10, "only {} presets", all.len());
        assert!(
            all.iter().any(|r| r.enthalpy > 0.0),
            "nothing endothermic — a sandbox where every reaction heats up teaches that \
             reactions heat up"
        );
        assert!(all.iter().any(|r| r.enthalpy < 0.0), "nothing exothermic");
        assert!(all.iter().any(|r| r.products().len() > 1), "nothing with two products");
        assert!(all.iter().any(|r| r.ingredients.len() == 3), "nothing simple");
    }

    /// A rate constant nobody measured must not be presented as one somebody did.
    #[test]
    fn illustrative_kinetics_are_labelled_as_such() {
        for recipe in presets() {
            assert_eq!(
                recipe.fidelity,
                Fidelity::Illustrative,
                "{} claims measured kinetics; it needs a citation, not a flag",
                recipe.name
            );
        }
        assert!(Fidelity::Illustrative.caveat().contains("not measured"));
    }

    /// The sliders override the recipe's own numbers; everything else is fixed, because
    /// the rest is what makes it this reaction.
    #[test]
    fn the_sliders_reach_the_rate_and_nothing_else() {
        let recipe = &presets()[0];
        let network = recipe.network(0.5, 1234.0);
        let law = &network.reactions()[0].rate;
        // The slider value is a pseudo-first-order rate at room temperature and reference
        // concentration, not the pre-exponential and not the mass-action constant.
        let order = recipe.order();
        let effective = law.forward_at(REFERENCE_TEMPERATURE)
            * REFERENCE_CONCENTRATION.powf(order - 1.0);
        assert!((effective - 0.5).abs() < 1e-9, "effective rate came back as {effective}");
        assert_eq!(network.species().len(), recipe.ingredients.len());
    }

    /// The parameterization that keeps a menu of reactions comparable: whatever the
    /// activation energy, the quoted rate is the rate you get at room temperature.
    #[test]
    fn the_quoted_rate_is_what_the_reaction_does_at_room_temperature() {
        let recipe = &presets()[0];
        for activation_energy in [0.0, 12_000.0, 40_000.0, 60_000.0] {
            let network = recipe.network(2.5, activation_energy);
            let law = &network.reactions()[0].rate;
            let effective = law.forward_at(REFERENCE_TEMPERATURE)
                * REFERENCE_CONCENTRATION.powf(recipe.order() - 1.0);
            assert!(
                (effective - 2.5).abs() < 1e-9,
                "Ea = {activation_energy} gave an effective rate of {effective} at 300 K"
            );
        }
    }

    /// And the activation energy keeps its real job: how sharply the reaction speeds up
    /// when the chamber is warmed. A larger one must respond more.
    #[test]
    fn a_larger_activation_energy_responds_more_strongly_to_heat() {
        let recipe = &presets()[0];
        let gentle = recipe.network(1.0, 10_000.0);
        let steep = recipe.network(1.0, 50_000.0);

        let warmed = REFERENCE_TEMPERATURE + 200.0;
        let gentle_gain = gentle.reactions()[0].rate.forward_at(warmed)
            / gentle.reactions()[0].rate.forward_at(REFERENCE_TEMPERATURE);
        let steep_gain = steep.reactions()[0].rate.forward_at(warmed)
            / steep.reactions()[0].rate.forward_at(REFERENCE_TEMPERATURE);

        assert!(gentle_gain > 1.0, "warming must speed a reaction up");
        assert!(
            steep_gain > gentle_gain * 5.0,
            "50 kJ/mol gained {steep_gain:.1}x against 10 kJ/mol's {gentle_gain:.1}x"
        );
    }
}
