//! Chemical species and the elements they are made of.
//!
//! Spec §12.2: *"Species are typed entities with names, charge, molar mass, optional
//! elemental composition, phase, diffusion coefficient, heat capacity, and
//! visualization metadata. […] The compiler verifies dimensional consistency of rate
//! expressions and, when elemental composition is available, atom and charge balance."*
//!
//! # Why composition is optional but load-bearing
//!
//! A model can declare a species as a bare name with a diffusion coefficient and get a
//! working reaction-diffusion run. What it gives up is the atom balance check: without
//! knowing that `H2O` is two hydrogens and an oxygen, nothing can tell that
//! `H2 + O2 -> H2O` is missing an oxygen. That reaction integrates perfectly happily
//! and conserves nothing, and the error surfaces as a mass ledger that will not close —
//! several layers away from the typo that caused it.
//!
//! So composition is optional, and [`crate::network::Reaction::balance`] reports
//! exactly which of the three cases a reaction is in: balanced, unbalanced (with the
//! offending element), or unknown because a species did not say.
//!
//! # Concentrations are per unit area
//!
//! `mol/m²`. This is a 2D engine, and a concentration per unit *volume* would need a
//! thickness nobody declared — the same argument that makes rigid-body density areal.
//! A second-order rate constant is therefore in `m²/(mol·s)`, which looks unfamiliar
//! and is the honest consequence.

use std::collections::BTreeMap;

/// The standard atomic weight of an element, in kg/mol.
///
/// Backed by the full table in [`crate::elements`]. This was a 36-entry list of its own
/// until the playground needed a periodic table to point at; keeping two would have meant
/// an element the picker offers and the balance checker rejects.
pub fn atomic_mass(symbol: &str) -> Option<f64> {
    crate::elements::by_symbol(symbol).map(crate::elements::Element::molar_mass)
}

/// Every element this build knows, for a diagnostic's "did you mean" list.
pub fn known_elements() -> impl Iterator<Item = &'static str> {
    crate::elements::all().iter().map(|element| element.symbol)
}

/// Why a formula could not be read.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FormulaError {
    /// A character that cannot start an element symbol.
    Unexpected {
        /// Byte offset into the formula.
        at: usize,
        /// The offending character.
        found: char,
    },
    /// A symbol that is not in this build's element table.
    UnknownElement {
        /// The symbol as written.
        symbol: String,
    },
    /// A subscript of zero, which means the element is not present and should be
    /// omitted rather than written.
    ZeroCount {
        /// Which element.
        symbol: String,
    },
    /// Nothing at all.
    Empty,
}

impl core::fmt::Display for FormulaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FormulaError::Unexpected { at, found } => {
                write!(f, "unexpected `{found}` at position {at}; an element symbol starts with a capital letter")
            }
            FormulaError::UnknownElement { symbol } => {
                write!(f, "`{symbol}` is not an element this build knows")
            }
            FormulaError::ZeroCount { symbol } => {
                write!(f, "`{symbol}0` means the element is absent; leave it out instead")
            }
            FormulaError::Empty => f.write_str("the formula is empty"),
        }
    }
}

impl core::error::Error for FormulaError {}

/// How many of each element a species contains.
///
/// Ordered, so two compositions with the same atoms compare and print identically
/// however they were written.
#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct Composition {
    atoms: BTreeMap<String, i32>,
}

impl Composition {
    /// An empty composition — "not stated", which is different from "no atoms".
    pub fn unknown() -> Composition {
        Composition::default()
    }

    /// Parse a formula such as `H2O`, `CO2`, `C6H12O6` or `Fe2O3`.
    ///
    /// Deliberately does not handle parentheses, hydrates or charge suffixes. A species
    /// whose formula needs them can state its composition directly, and a parser that
    /// half-understood `Ca(OH)2` would be worse than one that says it cannot.
    pub fn parse(formula: &str) -> Result<Composition, FormulaError> {
        let bytes = formula.as_bytes();
        let mut atoms: BTreeMap<String, i32> = BTreeMap::new();
        let mut index = 0usize;

        while index < bytes.len() {
            let start = index;
            if !bytes[index].is_ascii_uppercase() {
                return Err(FormulaError::Unexpected {
                    at: index,
                    found: formula[index..].chars().next().unwrap_or('?'),
                });
            }
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_lowercase() {
                index += 1;
            }
            let symbol = &formula[start..index];

            let digits_from = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            let count: i32 = if digits_from == index {
                1
            } else {
                formula[digits_from..index].parse().unwrap_or(1)
            };

            if atomic_mass(symbol).is_none() {
                return Err(FormulaError::UnknownElement { symbol: symbol.to_string() });
            }
            if count == 0 {
                return Err(FormulaError::ZeroCount { symbol: symbol.to_string() });
            }
            *atoms.entry(symbol.to_string()).or_insert(0) += count;
        }

        if atoms.is_empty() {
            return Err(FormulaError::Empty);
        }
        Ok(Composition { atoms })
    }

    /// Build from counts directly, for a species whose formula the parser cannot read.
    pub fn from_atoms<I, S>(atoms: I) -> Composition
    where
        I: IntoIterator<Item = (S, i32)>,
        S: Into<String>,
    {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        for (symbol, count) in atoms {
            *map.entry(symbol.into()).or_insert(0) += count;
        }
        map.retain(|_, count| *count != 0);
        Composition { atoms: map }
    }

    /// True when nothing was stated. Distinct from a composition of zero atoms, which
    /// cannot be constructed.
    pub fn is_unknown(&self) -> bool {
        self.atoms.is_empty()
    }

    /// How many atoms of `symbol`.
    pub fn count(&self, symbol: &str) -> i32 {
        self.atoms.get(symbol).copied().unwrap_or(0)
    }

    /// Every element and its count, in symbol order.
    pub fn atoms(&self) -> impl Iterator<Item = (&str, i32)> {
        self.atoms.iter().map(|(symbol, count)| (symbol.as_str(), *count))
    }

    /// Molar mass implied by the composition, kg/mol. `None` when unknown.
    pub fn molar_mass(&self) -> Option<f64> {
        if self.is_unknown() {
            return None;
        }
        let mut total = 0.0;
        for (symbol, count) in self.atoms() {
            total += atomic_mass(symbol)? * f64::from(count);
        }
        Some(total)
    }

    /// The formula, written back out in Hill order where carbon is present.
    ///
    /// Hill order — carbon, then hydrogen, then everything else alphabetically — is the
    /// convention chemists read, so a diagnostic that echoes a formula echoes it the
    /// way it was probably written.
    pub fn formula(&self) -> String {
        if self.is_unknown() {
            return "unknown".to_string();
        }
        let mut ordered: Vec<(&str, i32)> = self.atoms().collect();
        if self.count("C") > 0 {
            ordered.sort_by_key(|(symbol, _)| match *symbol {
                "C" => (0, ""),
                "H" => (1, ""),
                other => (2, other),
            });
        }
        ordered
            .iter()
            .map(|(symbol, count)| {
                if *count == 1 {
                    (*symbol).to_string()
                } else {
                    format!("{symbol}{count}")
                }
            })
            .collect()
    }
}

/// What phase a species is in.
///
/// Carried but not yet acted on: nothing in this module changes behaviour with phase.
/// It is recorded because a rate law for a heterogeneous reaction *does* depend on it,
/// and discovering later that the information was thrown away is worse than carrying it.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Phase {
    /// Dissolved in a solvent — the default, and what a concentration usually means.
    #[default]
    Aqueous,
    /// A gas.
    Gas,
    /// A pure liquid.
    Liquid,
    /// A solid.
    Solid,
}

impl Phase {
    /// The one-letter label chemists write.
    pub const fn label(self) -> &'static str {
        match self {
            Phase::Aqueous => "aq",
            Phase::Gas => "g",
            Phase::Liquid => "l",
            Phase::Solid => "s",
        }
    }
}

/// A chemical species.
#[derive(Clone, PartialEq, Debug)]
pub struct Species {
    /// The name the model uses.
    pub name: String,
    /// Elemental composition, or [`Composition::unknown`].
    pub composition: Composition,
    /// Molar mass, kg/mol.
    pub molar_mass: f64,
    /// Charge, in elementary charges. Fractional charges are not chemistry.
    pub charge: i32,
    /// Diffusion coefficient, m²/s. Zero for an immobile species.
    pub diffusion: f64,
    /// Which phase.
    pub phase: Phase,
}

impl Species {
    /// A species with a name and nothing else stated.
    ///
    /// Usable — it will diffuse and react — but it cannot participate in an atom
    /// balance check, and [`Species::describe`] says so.
    pub fn new(name: impl Into<String>) -> Species {
        Species {
            name: name.into(),
            composition: Composition::unknown(),
            molar_mass: 0.0,
            charge: 0,
            diffusion: 0.0,
            phase: Phase::default(),
        }
    }

    /// Set the composition from a formula, taking the molar mass from it unless one was
    /// already given.
    pub fn with_formula(mut self, formula: &str) -> Result<Species, FormulaError> {
        self.composition = Composition::parse(formula)?;
        if self.molar_mass == 0.0 {
            self.molar_mass = self.composition.molar_mass().unwrap_or(0.0);
        }
        Ok(self)
    }

    /// Set the composition directly.
    pub fn with_composition(mut self, composition: Composition) -> Species {
        if self.molar_mass == 0.0 {
            self.molar_mass = composition.molar_mass().unwrap_or(0.0);
        }
        self.composition = composition;
        self
    }

    /// Set the molar mass, kg/mol. Overrides anything the formula implied.
    pub fn with_molar_mass(mut self, molar_mass: f64) -> Species {
        self.molar_mass = molar_mass;
        self
    }

    /// Set the charge, in elementary charges.
    pub fn with_charge(mut self, charge: i32) -> Species {
        self.charge = charge;
        self
    }

    /// Set the diffusion coefficient, m²/s.
    pub fn with_diffusion(mut self, diffusion: f64) -> Species {
        self.diffusion = diffusion;
        self
    }

    /// Set the phase.
    pub fn with_phase(mut self, phase: Phase) -> Species {
        self.phase = phase;
        self
    }

    /// How far the declared molar mass is from what the formula implies, relatively.
    ///
    /// `None` when there is no composition to compare against. A model that writes both
    /// and disagrees has a typo in one of them, and this is what finds it — the numbers
    /// are close enough that neither looks wrong on its own.
    pub fn molar_mass_discrepancy(&self) -> Option<f64> {
        let implied = self.composition.molar_mass()?;
        if implied <= 0.0 || self.molar_mass <= 0.0 {
            return None;
        }
        Some((self.molar_mass - implied).abs() / implied)
    }

    /// A one-line description for the model report.
    pub fn describe(&self) -> String {
        let mut parts = vec![format!("{} ({})", self.composition.formula(), self.phase.label())];
        if self.molar_mass > 0.0 {
            parts.push(format!("{:.4} kg/mol", self.molar_mass));
        }
        if self.charge != 0 {
            parts.push(format!("charge {:+}", self.charge));
        }
        if self.diffusion > 0.0 {
            parts.push(format!("D = {:.3e} m^2/s", self.diffusion));
        } else {
            parts.push("immobile".to_string());
        }
        if self.composition.is_unknown() {
            parts.push("composition not stated, so it cannot be atom-balanced".to_string());
        }
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_formulas_parse() {
        let water = Composition::parse("H2O").unwrap();
        assert_eq!(water.count("H"), 2);
        assert_eq!(water.count("O"), 1);
        assert_eq!(water.count("C"), 0);
        assert_eq!(water.formula(), "H2O");

        let sugar = Composition::parse("C6H12O6").unwrap();
        assert_eq!(sugar.count("C"), 6);
        assert_eq!(sugar.count("H"), 12);
        assert_eq!(sugar.formula(), "C6H12O6", "Hill order puts carbon first");
    }

    /// Two-letter symbols must not be read as two one-letter ones: `Co` is cobalt, not
    /// carbon monoxide, and a parser that got this wrong would balance reactions
    /// against the wrong element without complaining.
    #[test]
    fn two_letter_symbols_are_not_split() {
        let cobalt = Composition::parse("Co").unwrap();
        assert_eq!(cobalt.count("Co"), 1);
        assert_eq!(cobalt.count("C"), 0);
        assert_eq!(cobalt.count("O"), 0);

        let carbon_monoxide = Composition::parse("CO").unwrap();
        assert_eq!(carbon_monoxide.count("C"), 1);
        assert_eq!(carbon_monoxide.count("O"), 1);
        assert_eq!(carbon_monoxide.count("Co"), 0);
    }

    #[test]
    fn repeated_elements_accumulate() {
        // Written the way an equation sometimes is, rather than collected.
        let acetic = Composition::parse("CH3COOH").unwrap();
        assert_eq!(acetic.count("C"), 2);
        assert_eq!(acetic.count("H"), 4);
        assert_eq!(acetic.count("O"), 2);
        assert_eq!(acetic.formula(), "C2H4O2");
    }

    #[test]
    fn bad_formulas_are_rejected_with_a_reason() {
        assert_eq!(Composition::parse(""), Err(FormulaError::Empty));
        assert_eq!(
            Composition::parse("2H"),
            Err(FormulaError::Unexpected { at: 0, found: '2' })
        );
        assert_eq!(
            Composition::parse("H2Xy"),
            Err(FormulaError::UnknownElement { symbol: "Xy".to_string() })
        );
        assert_eq!(
            Composition::parse("H0O"),
            Err(FormulaError::ZeroCount { symbol: "H".to_string() })
        );
        // Parentheses are honestly refused rather than half-understood.
        assert!(matches!(Composition::parse("Ca(OH)2"), Err(FormulaError::Unexpected { .. })));
    }

    #[test]
    fn molar_mass_follows_from_the_formula() {
        let water = Composition::parse("H2O").unwrap();
        // 2 x 1.008 + 15.999 = 18.015 g/mol.
        assert!((water.molar_mass().unwrap() - 0.018_015).abs() < 1e-6);
        assert_eq!(Composition::unknown().molar_mass(), None);
    }

    /// A model that writes both a formula and a molar mass and disagrees has a typo in
    /// one of them. Neither number looks wrong on its own.
    #[test]
    fn a_molar_mass_that_contradicts_the_formula_is_measurable() {
        let honest = Species::new("water").with_formula("H2O").unwrap();
        assert!(honest.molar_mass_discrepancy().unwrap() < 1e-12, "taken from the formula");

        let typo = Species::new("water").with_molar_mass(0.180).with_formula("H2O").unwrap();
        let off_by = typo.molar_mass_discrepancy().unwrap();
        assert!(off_by > 8.0, "180 g/mol against an implied 18: off by {off_by}");

        // Nothing to compare against is not a discrepancy.
        assert_eq!(Species::new("X").with_molar_mass(1.0).molar_mass_discrepancy(), None);
    }

    #[test]
    fn a_species_can_be_declared_with_nothing_but_a_name() {
        let bare = Species::new("A");
        assert!(bare.composition.is_unknown());
        assert!(bare.describe().contains("cannot be atom-balanced"));

        let full = Species::new("hydronium")
            .with_formula("H3O")
            .unwrap()
            .with_charge(1)
            .with_diffusion(9.31e-9)
            .with_phase(Phase::Aqueous);
        assert_eq!(full.charge, 1);
        assert!(full.describe().contains("H3O"));
        assert!(full.describe().contains("charge +1"));
        assert!(full.describe().contains("aq"));
    }

    #[test]
    fn compositions_can_be_built_without_a_formula() {
        let direct = Composition::from_atoms([("Ca", 1), ("O", 2), ("H", 2)]);
        assert_eq!(direct.count("Ca"), 1);
        assert_eq!(direct.count("O"), 2);
        // Cancelling counts leave nothing behind.
        assert!(Composition::from_atoms([("H", 1), ("H", -1)]).is_unknown());
    }

    #[test]
    fn every_tabulated_element_has_a_plausible_weight() {
        for symbol in known_elements() {
            let mass = atomic_mass(symbol).expect("listed elements resolve");
            assert!(mass > 0.0 && mass < 0.3, "{symbol} weighs {mass} kg/mol");
        }
        assert_eq!(atomic_mass("Xy"), None);
        // Hydrogen and lead anchor the range.
        assert!((atomic_mass("H").unwrap() - 0.001_008).abs() < 1e-9);
        assert!((atomic_mass("Pb").unwrap() - 0.2072).abs() < 1e-9);
    }
}
