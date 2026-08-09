//! The periodic table.
//!
//! Standard atomic weights are the IUPAC 2021 abridged values, in g/mol. Where an element
//! has no stable isotopic composition there is no standard atomic weight at all, and the
//! *conventional* value — the mass number of its longest-lived or most-available isotope —
//! is used instead. [`Element::weight_is_conventional`] says which is which, because the
//! difference matters: a standard atomic weight describes terrestrial material and a
//! conventional one describes a single isotope somebody made.
//!
//! # Why the whole table
//!
//! The engine only needs an element's molar mass, and it only needs that for the elements
//! a model actually mentions. The table was 36 entries for exactly that reason. It is 118
//! now because a periodic table is also a *thing to point at*: the playground's species
//! picker is a periodic table, and one with holes in it is a periodic table that has been
//! edited by a programmer's guess about what a reader will want.
//!
//! # What this is not
//!
//! Molar mass and a position in the grid. **Nothing here says what reacts with what.**
//! Predicting products needs thermodynamic data — formation enthalpies and entropies — that
//! this table does not carry and the engine does not have; §5.3 draws that boundary and
//! §13.2 puts it behind external adapters. An element being present here means the engine
//! can weigh it and balance it, not that it knows its chemistry.

/// Where an element sits in the usual classification.
///
/// Used for colouring a picker, and deliberately the *common* scheme rather than a
/// defensible one — the boundaries of "metalloid" are a matter of convention, and a
/// sandbox is not the place to litigate them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Category {
    /// Group 1 except hydrogen.
    AlkaliMetal,
    /// Group 2.
    AlkalineEarthMetal,
    /// The d-block.
    TransitionMetal,
    /// The p-block metals.
    PostTransitionMetal,
    /// B, Si, Ge, As, Sb, Te — by the usual convention.
    Metalloid,
    /// Everything from hydrogen to the halogens that is not a noble gas.
    Nonmetal,
    /// Group 18.
    NobleGas,
    /// The 4f row, La–Lu.
    Lanthanide,
    /// The 5f row, Ac–Lr.
    Actinide,
}

impl Category {
    /// A short label for a legend.
    pub const fn label(self) -> &'static str {
        match self {
            Category::AlkaliMetal => "alkali metal",
            Category::AlkalineEarthMetal => "alkaline earth",
            Category::TransitionMetal => "transition metal",
            Category::PostTransitionMetal => "post-transition metal",
            Category::Metalloid => "metalloid",
            Category::Nonmetal => "nonmetal",
            Category::NobleGas => "noble gas",
            Category::Lanthanide => "lanthanide",
            Category::Actinide => "actinide",
        }
    }
}

/// One element.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Element {
    /// Atomic number, and the table's index + 1.
    pub number: u8,
    /// Chemical symbol, e.g. `"Fe"`.
    pub symbol: &'static str,
    /// English name.
    pub name: &'static str,
    /// Standard (or conventional) atomic weight, g/mol.
    pub weight: f64,
    /// Group 1–18, or `None` for the f-block, which the standard table does not assign one.
    pub group: Option<u8>,
    /// Period 1–7.
    pub period: u8,
    /// The usual classification.
    pub category: Category,
    /// True when [`Element::weight`] is a conventional value for a single isotope rather
    /// than a standard atomic weight for terrestrial material.
    pub weight_is_conventional: bool,
}

impl Element {
    /// Molar mass in kg/mol, which is what the engine works in.
    pub fn molar_mass(&self) -> f64 {
        self.weight / 1000.0
    }

    /// Row and column for drawing the conventional 18-wide table.
    ///
    /// Rows are 1-based periods. The f-block is pulled out below the main body the way
    /// every printed table does it — rows 9 and 10, leaving row 8 blank as the gap — because
    /// inlining it would make the table 32 columns wide and unreadable at any size a panel
    /// has.
    pub fn display_position(&self) -> (u8, u8) {
        match self.category {
            // La is 57 and sits at column 3 of row 9; Ac is 89 at column 3 of row 10.
            Category::Lanthanide => (9, 3 + (self.number - 57)),
            Category::Actinide => (10, 3 + (self.number - 89)),
            _ => (self.period, self.group.expect("only the f-block has no group")),
        }
    }
}

/// Every element, indexed by atomic number minus one.
pub fn all() -> &'static [Element] {
    ELEMENTS
}

/// Look an element up by symbol. Case-sensitive, as chemical symbols are.
pub fn by_symbol(symbol: &str) -> Option<&'static Element> {
    ELEMENTS.iter().find(|element| element.symbol == symbol)
}

/// Look an element up by atomic number.
pub fn by_number(number: u8) -> Option<&'static Element> {
    ELEMENTS.get(number.checked_sub(1)? as usize)
}

const fn e(
    number: u8,
    symbol: &'static str,
    name: &'static str,
    weight: f64,
    group: Option<u8>,
    period: u8,
    category: Category,
    weight_is_conventional: bool,
) -> Element {
    Element { number, symbol, name, weight, group, period, category, weight_is_conventional }
}

use Category::{
    Actinide as Ac, AlkaliMetal as Alk, AlkalineEarthMetal as Ae, Lanthanide as Ln,
    Metalloid as Md, NobleGas as Ng, Nonmetal as Nm, PostTransitionMetal as Ptm,
    TransitionMetal as Tm,
};

/// IUPAC 2021 abridged standard atomic weights, with conventional values in the
/// `weight_is_conventional` rows.
static ELEMENTS: &[Element] = &[
    e(1, "H", "hydrogen", 1.008, Some(1), 1, Nm, false),
    e(2, "He", "helium", 4.0026, Some(18), 1, Ng, false),
    e(3, "Li", "lithium", 6.94, Some(1), 2, Alk, false),
    e(4, "Be", "beryllium", 9.0122, Some(2), 2, Ae, false),
    e(5, "B", "boron", 10.81, Some(13), 2, Md, false),
    e(6, "C", "carbon", 12.011, Some(14), 2, Nm, false),
    e(7, "N", "nitrogen", 14.007, Some(15), 2, Nm, false),
    e(8, "O", "oxygen", 15.999, Some(16), 2, Nm, false),
    e(9, "F", "fluorine", 18.998, Some(17), 2, Nm, false),
    e(10, "Ne", "neon", 20.180, Some(18), 2, Ng, false),
    e(11, "Na", "sodium", 22.990, Some(1), 3, Alk, false),
    e(12, "Mg", "magnesium", 24.305, Some(2), 3, Ae, false),
    e(13, "Al", "aluminium", 26.982, Some(13), 3, Ptm, false),
    e(14, "Si", "silicon", 28.085, Some(14), 3, Md, false),
    e(15, "P", "phosphorus", 30.974, Some(15), 3, Nm, false),
    e(16, "S", "sulfur", 32.06, Some(16), 3, Nm, false),
    e(17, "Cl", "chlorine", 35.45, Some(17), 3, Nm, false),
    e(18, "Ar", "argon", 39.95, Some(18), 3, Ng, false),
    e(19, "K", "potassium", 39.098, Some(1), 4, Alk, false),
    e(20, "Ca", "calcium", 40.078, Some(2), 4, Ae, false),
    e(21, "Sc", "scandium", 44.956, Some(3), 4, Tm, false),
    e(22, "Ti", "titanium", 47.867, Some(4), 4, Tm, false),
    e(23, "V", "vanadium", 50.942, Some(5), 4, Tm, false),
    e(24, "Cr", "chromium", 51.996, Some(6), 4, Tm, false),
    e(25, "Mn", "manganese", 54.938, Some(7), 4, Tm, false),
    e(26, "Fe", "iron", 55.845, Some(8), 4, Tm, false),
    e(27, "Co", "cobalt", 58.933, Some(9), 4, Tm, false),
    e(28, "Ni", "nickel", 58.693, Some(10), 4, Tm, false),
    e(29, "Cu", "copper", 63.546, Some(11), 4, Tm, false),
    e(30, "Zn", "zinc", 65.38, Some(12), 4, Tm, false),
    e(31, "Ga", "gallium", 69.723, Some(13), 4, Ptm, false),
    e(32, "Ge", "germanium", 72.630, Some(14), 4, Md, false),
    e(33, "As", "arsenic", 74.922, Some(15), 4, Md, false),
    e(34, "Se", "selenium", 78.971, Some(16), 4, Nm, false),
    e(35, "Br", "bromine", 79.904, Some(17), 4, Nm, false),
    e(36, "Kr", "krypton", 83.798, Some(18), 4, Ng, false),
    e(37, "Rb", "rubidium", 85.468, Some(1), 5, Alk, false),
    e(38, "Sr", "strontium", 87.62, Some(2), 5, Ae, false),
    e(39, "Y", "yttrium", 88.906, Some(3), 5, Tm, false),
    e(40, "Zr", "zirconium", 91.224, Some(4), 5, Tm, false),
    e(41, "Nb", "niobium", 92.906, Some(5), 5, Tm, false),
    e(42, "Mo", "molybdenum", 95.95, Some(6), 5, Tm, false),
    e(43, "Tc", "technetium", 97.0, Some(7), 5, Tm, true),
    e(44, "Ru", "ruthenium", 101.07, Some(8), 5, Tm, false),
    e(45, "Rh", "rhodium", 102.91, Some(9), 5, Tm, false),
    e(46, "Pd", "palladium", 106.42, Some(10), 5, Tm, false),
    e(47, "Ag", "silver", 107.87, Some(11), 5, Tm, false),
    e(48, "Cd", "cadmium", 112.41, Some(12), 5, Tm, false),
    e(49, "In", "indium", 114.82, Some(13), 5, Ptm, false),
    e(50, "Sn", "tin", 118.71, Some(14), 5, Ptm, false),
    e(51, "Sb", "antimony", 121.76, Some(15), 5, Md, false),
    e(52, "Te", "tellurium", 127.60, Some(16), 5, Md, false),
    e(53, "I", "iodine", 126.90, Some(17), 5, Nm, false),
    e(54, "Xe", "xenon", 131.29, Some(18), 5, Ng, false),
    e(55, "Cs", "caesium", 132.91, Some(1), 6, Alk, false),
    e(56, "Ba", "barium", 137.33, Some(2), 6, Ae, false),
    e(57, "La", "lanthanum", 138.91, None, 6, Ln, false),
    e(58, "Ce", "cerium", 140.12, None, 6, Ln, false),
    e(59, "Pr", "praseodymium", 140.91, None, 6, Ln, false),
    e(60, "Nd", "neodymium", 144.24, None, 6, Ln, false),
    e(61, "Pm", "promethium", 145.0, None, 6, Ln, true),
    e(62, "Sm", "samarium", 150.36, None, 6, Ln, false),
    e(63, "Eu", "europium", 151.96, None, 6, Ln, false),
    e(64, "Gd", "gadolinium", 157.25, None, 6, Ln, false),
    e(65, "Tb", "terbium", 158.93, None, 6, Ln, false),
    e(66, "Dy", "dysprosium", 162.50, None, 6, Ln, false),
    e(67, "Ho", "holmium", 164.93, None, 6, Ln, false),
    e(68, "Er", "erbium", 167.26, None, 6, Ln, false),
    e(69, "Tm", "thulium", 168.93, None, 6, Ln, false),
    e(70, "Yb", "ytterbium", 173.05, None, 6, Ln, false),
    e(71, "Lu", "lutetium", 174.97, None, 6, Ln, false),
    e(72, "Hf", "hafnium", 178.49, Some(4), 6, Tm, false),
    e(73, "Ta", "tantalum", 180.95, Some(5), 6, Tm, false),
    e(74, "W", "tungsten", 183.84, Some(6), 6, Tm, false),
    e(75, "Re", "rhenium", 186.21, Some(7), 6, Tm, false),
    e(76, "Os", "osmium", 190.23, Some(8), 6, Tm, false),
    e(77, "Ir", "iridium", 192.22, Some(9), 6, Tm, false),
    e(78, "Pt", "platinum", 195.08, Some(10), 6, Tm, false),
    e(79, "Au", "gold", 196.97, Some(11), 6, Tm, false),
    e(80, "Hg", "mercury", 200.59, Some(12), 6, Tm, false),
    e(81, "Tl", "thallium", 204.38, Some(13), 6, Ptm, false),
    e(82, "Pb", "lead", 207.2, Some(14), 6, Ptm, false),
    e(83, "Bi", "bismuth", 208.98, Some(15), 6, Ptm, false),
    e(84, "Po", "polonium", 209.0, Some(16), 6, Ptm, true),
    e(85, "At", "astatine", 210.0, Some(17), 6, Md, true),
    e(86, "Rn", "radon", 222.0, Some(18), 6, Ng, true),
    e(87, "Fr", "francium", 223.0, Some(1), 7, Alk, true),
    e(88, "Ra", "radium", 226.0, Some(2), 7, Ae, true),
    e(89, "Ac", "actinium", 227.0, None, 7, Ac, true),
    e(90, "Th", "thorium", 232.04, None, 7, Ac, false),
    e(91, "Pa", "protactinium", 231.04, None, 7, Ac, false),
    e(92, "U", "uranium", 238.03, None, 7, Ac, false),
    e(93, "Np", "neptunium", 237.0, None, 7, Ac, true),
    e(94, "Pu", "plutonium", 244.0, None, 7, Ac, true),
    e(95, "Am", "americium", 243.0, None, 7, Ac, true),
    e(96, "Cm", "curium", 247.0, None, 7, Ac, true),
    e(97, "Bk", "berkelium", 247.0, None, 7, Ac, true),
    e(98, "Cf", "californium", 251.0, None, 7, Ac, true),
    e(99, "Es", "einsteinium", 252.0, None, 7, Ac, true),
    e(100, "Fm", "fermium", 257.0, None, 7, Ac, true),
    e(101, "Md", "mendelevium", 258.0, None, 7, Ac, true),
    e(102, "No", "nobelium", 259.0, None, 7, Ac, true),
    e(103, "Lr", "lawrencium", 266.0, None, 7, Ac, true),
    e(104, "Rf", "rutherfordium", 267.0, Some(4), 7, Tm, true),
    e(105, "Db", "dubnium", 268.0, Some(5), 7, Tm, true),
    e(106, "Sg", "seaborgium", 269.0, Some(6), 7, Tm, true),
    e(107, "Bh", "bohrium", 270.0, Some(7), 7, Tm, true),
    e(108, "Hs", "hassium", 269.0, Some(8), 7, Tm, true),
    e(109, "Mt", "meitnerium", 278.0, Some(9), 7, Tm, true),
    e(110, "Ds", "darmstadtium", 281.0, Some(10), 7, Tm, true),
    e(111, "Rg", "roentgenium", 282.0, Some(11), 7, Tm, true),
    e(112, "Cn", "copernicium", 285.0, Some(12), 7, Tm, true),
    e(113, "Nh", "nihonium", 286.0, Some(13), 7, Ptm, true),
    e(114, "Fl", "flerovium", 289.0, Some(14), 7, Ptm, true),
    e(115, "Mc", "moscovium", 290.0, Some(15), 7, Ptm, true),
    e(116, "Lv", "livermorium", 293.0, Some(16), 7, Ptm, true),
    e(117, "Ts", "tennessine", 294.0, Some(17), 7, Md, true),
    e(118, "Og", "oganesson", 294.0, Some(18), 7, Ng, true),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn the_table_is_complete_and_in_order() {
        assert_eq!(ELEMENTS.len(), 118);
        for (index, element) in ELEMENTS.iter().enumerate() {
            assert_eq!(
                element.number as usize,
                index + 1,
                "{} is out of order",
                element.symbol
            );
        }
    }

    #[test]
    fn symbols_and_names_are_unique() {
        let symbols: HashSet<_> = ELEMENTS.iter().map(|e| e.symbol).collect();
        assert_eq!(symbols.len(), 118, "a symbol is repeated");
        let names: HashSet<_> = ELEMENTS.iter().map(|e| e.name).collect();
        assert_eq!(names.len(), 118, "a name is repeated");
    }

    #[test]
    fn lookups_agree_with_each_other() {
        for element in ELEMENTS {
            assert_eq!(by_symbol(element.symbol).unwrap().number, element.number);
            assert_eq!(by_number(element.number).unwrap().symbol, element.symbol);
        }
        assert!(by_symbol("Xx").is_none());
        assert!(by_number(0).is_none(), "there is no element zero");
        assert!(by_number(119).is_none());
    }

    /// Spot values, against figures that can be checked in any reference.
    #[test]
    fn the_weights_are_right_where_it_is_easy_to_check() {
        for (symbol, weight) in [
            ("H", 1.008),
            ("C", 12.011),
            ("O", 15.999),
            ("Fe", 55.845),
            ("U", 238.03),
        ] {
            let element = by_symbol(symbol).unwrap();
            assert!((element.weight - weight).abs() < 1e-9, "{symbol} is {}", element.weight);
        }
        // And the engine's unit is kg/mol, not g/mol.
        assert!((by_symbol("C").unwrap().molar_mass() - 0.012_011).abs() < 1e-12);
    }

    /// Atomic weight does *not* increase monotonically with atomic number. Four pairs
    /// invert, and a table built by sorting on mass would put them the wrong way round —
    /// which is the mistake Mendeleev's contemporaries made and the reason the table is
    /// ordered by proton count.
    #[test]
    fn the_four_classic_inversions_are_present() {
        for (heavier, lighter) in [("Ar", "K"), ("Co", "Ni"), ("Te", "I"), ("Th", "Pa")] {
            let a = by_symbol(heavier).unwrap();
            let b = by_symbol(lighter).unwrap();
            assert!(a.number < b.number, "{heavier} should come first");
            assert!(
                a.weight > b.weight,
                "{heavier} ({}) should outweigh {lighter} ({})",
                a.weight,
                b.weight
            );
        }
    }

    #[test]
    fn every_element_has_a_place_to_be_drawn() {
        let mut taken: HashSet<(u8, u8)> = HashSet::new();
        for element in ELEMENTS {
            let position = element.display_position();
            assert!(
                (1..=10).contains(&position.0) && (1..=18).contains(&position.1),
                "{} lands at {position:?}",
                element.symbol
            );
            assert!(taken.insert(position), "{} collides at {position:?}", element.symbol);
        }
    }

    /// The f-block is the only part of the table without a group, and it must be exactly
    /// the f-block — a missing group anywhere else would silently panic `display_position`.
    #[test]
    fn only_the_f_block_lacks_a_group() {
        for element in ELEMENTS {
            let f_block = matches!(element.category, Category::Lanthanide | Category::Actinide);
            assert_eq!(
                element.group.is_none(),
                f_block,
                "{} has group {:?} and category {:?}",
                element.symbol,
                element.group,
                element.category
            );
        }
        assert_eq!(ELEMENTS.iter().filter(|e| e.group.is_none()).count(), 30, "15 + 15");
    }

    /// A conventional weight is a statement about one isotope somebody made, and every
    /// element without a stable isotopic composition needs one.
    #[test]
    fn elements_without_stable_isotopes_carry_a_conventional_weight() {
        for symbol in ["Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Np", "Pu", "Og"] {
            assert!(
                by_symbol(symbol).unwrap().weight_is_conventional,
                "{symbol} should be flagged conventional"
            );
        }
        // And the common ones must not be.
        for symbol in ["H", "C", "O", "Fe", "Th", "U"] {
            assert!(
                !by_symbol(symbol).unwrap().weight_is_conventional,
                "{symbol} has a standard atomic weight"
            );
        }
    }

    #[test]
    fn periods_hold_the_number_of_elements_they_should() {
        for (period, count) in [(1, 2), (2, 8), (3, 8), (4, 18), (5, 18), (6, 32), (7, 32)] {
            let actual = ELEMENTS.iter().filter(|e| e.period == period).count();
            assert_eq!(actual, count, "period {period} has {actual} elements");
        }
    }
}
