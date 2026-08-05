//! The runtime unit registry: names → (scale, offset, dimension).
//!
//! Spec §24 lists a "runtime unit registry" as part of this crate, and §16.3 allows
//! plugins to introduce their own vocabulary. So the registry is a constructed,
//! extensible object rather than a static table: [`UnitRegistry::si`] gives the
//! standard set, and [`UnitRegistry::register`] adds domain-specific units.
//!
//! # Resolution order
//!
//! 1. Exact match on a canonical name or alias.
//! 2. Longest matching SI prefix (long form, then symbol form) whose remainder
//!    resolves to a prefixable unit.
//!
//! Exact-match-first is what makes ambiguous symbols behave: `cd` is candela, not
//! centi-day; `min` is minute, not milli-inch; `h` is hour, not hecto-anything.

use std::collections::HashMap;

use crate::dimension::Dimension;
use crate::error::UnitError;

/// One unit's definition. SI value = `raw * scale + offset`.
#[derive(Clone, Debug)]
pub struct UnitDef {
    /// The name used when printing this unit back to a user.
    pub canonical: String,
    /// The dimension the unit measures.
    pub dim: Dimension,
    /// Multiplicative factor to coherent SI base units.
    pub scale: f64,
    /// Additive offset applied after scaling. Nonzero only for affine temperature
    /// scales, which [`UnitRegistry::resolve`] refuses to let into compound
    /// expressions.
    pub offset: f64,
    /// Whether SI prefixes may be attached.
    pub prefixable: bool,
}

impl UnitDef {
    /// True for affine scales (Celsius, Fahrenheit) that cannot be composed.
    pub fn is_affine(&self) -> bool {
        self.offset != 0.0
    }
}

/// A resolved unit reference: a definition plus any prefix factor applied to it.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedUnit<'a> {
    /// The underlying definition.
    pub def: &'a UnitDef,
    /// The SI prefix multiplier (1.0 when no prefix was used).
    pub prefix_factor: f64,
    /// The prefix that was stripped, if any — used in diagnostics.
    pub prefix_name: Option<&'static str>,
}

impl ResolvedUnit<'_> {
    /// Total multiplicative factor to coherent SI base units.
    pub fn scale(&self) -> f64 {
        self.def.scale * self.prefix_factor
    }

    /// Additive offset. Always 0.0 unless the unit is an affine temperature scale.
    pub fn offset(&self) -> f64 {
        self.def.offset
    }

    /// The dimension measured.
    pub fn dim(&self) -> Dimension {
        self.def.dim
    }
}

/// SI decimal prefixes, longest name first so greedy matching is correct.
///
/// `deka`/`deca` and `micro`/`u` both appear because both spellings are in common
/// use and rejecting one would be a pointless papercut.
const PREFIXES: &[(&str, &str, f64)] = &[
    // (long name, symbol, factor)
    ("quetta", "Q", 1e30),
    ("ronna", "R", 1e27),
    ("yotta", "Y", 1e24),
    ("zetta", "Z", 1e21),
    ("exa", "E", 1e18),
    ("peta", "P", 1e15),
    ("tera", "T", 1e12),
    ("giga", "G", 1e9),
    ("mega", "M", 1e6),
    ("kilo", "k", 1e3),
    ("hecto", "h", 1e2),
    ("deka", "da", 1e1),
    ("deca", "da", 1e1),
    ("deci", "d", 1e-1),
    ("centi", "c", 1e-2),
    ("milli", "m", 1e-3),
    ("micro", "\u{b5}", 1e-6),
    ("nano", "n", 1e-9),
    ("pico", "p", 1e-12),
    ("femto", "f", 1e-15),
    ("atto", "a", 1e-18),
    ("zepto", "z", 1e-21),
    ("yocto", "y", 1e-24),
    ("ronto", "r", 1e-27),
    ("quecto", "q", 1e-30),
];

/// Extra micro spellings that are not the canonical symbol.
///
/// `u` is the ASCII stand-in everyone types; `μ` (U+03BC GREEK SMALL LETTER MU) is what
/// most editors insert when you ask for "mu", while SI specifies U+00B5 MICRO SIGN.
/// All three mean the same thing and all three must work.
const MICRO_ALIASES: &[&str] = &["u", "\u{3bc}"];

/// A unit table entry. Split out so [`UnitRegistry::si`] reads as data, not code.
struct UnitSpec {
    canonical: &'static str,
    aliases: &'static [&'static str],
    dim: Dimension,
    scale: f64,
    offset: f64,
    prefixable: bool,
}

const fn u(
    canonical: &'static str,
    aliases: &'static [&'static str],
    dim: Dimension,
    scale: f64,
    prefixable: bool,
) -> UnitSpec {
    UnitSpec { canonical, aliases, dim, scale, offset: 0.0, prefixable }
}

/// Kelvin per degree Fahrenheit.
const F_SCALE: f64 = 5.0 / 9.0;
/// Kelvin at 0 °F: `(0 + 459.67) * 5/9`.
const F_OFFSET: f64 = 459.67 * F_SCALE;

// Atomic-scale unit magnitudes come from `crate::constants::value` so that the unit
// table and the typed constants can never disagree about, say, how many joules an
// electronvolt is.
use crate::constants::value as cv;

const EV_JOULES: f64 = cv::ELECTRONVOLT;
/// Numerically equal to [`EV_JOULES`] by definition of the electronvolt, but a
/// different physical quantity — kept separate so neither can be "fixed" into the
/// other by accident.
const ELEMENTARY_CHARGE_C: f64 = cv::ELEMENTARY_CHARGE;
const AMU_KG: f64 = cv::ATOMIC_MASS_UNIT;
const ELECTRON_MASS_KG: f64 = cv::ELECTRON_MASS;
const PROTON_MASS_KG: f64 = cv::PROTON_MASS;
const BOHR_RADIUS_M: f64 = cv::BOHR_RADIUS;
const HARTREE_J: f64 = cv::HARTREE;

#[rustfmt::skip]
const STANDARD_UNITS: &[UnitSpec] = &[
    // ---- SI base ---------------------------------------------------------------
    u("meter",     &["metre", "m"],            Dimension::LENGTH,      1.0,   true),
    // The SI base unit of mass is the kilogram, but the *prefixable root* is the
    // gram, so `kg` resolves as kilo+gram and `mg` works without a special case.
    u("gram",      &["g"],                     Dimension::MASS,        1e-3,  true),
    u("second",    &["s", "sec"],              Dimension::TIME,        1.0,   true),
    u("ampere",    &["A", "amp"],              Dimension::CURRENT,     1.0,   true),
    u("kelvin",    &["K"],                     Dimension::TEMPERATURE, 1.0,   true),
    u("mole",      &["mol"],                   Dimension::AMOUNT,      1.0,   true),
    u("candela",   &["cd"],                    Dimension::LUMINOUS,    1.0,   true),

    // ---- Coherent derived SI ---------------------------------------------------
    u("newton",    &["N"],                     Dimension::FORCE,       1.0,   true),
    u("joule",     &["J"],                     Dimension::ENERGY,      1.0,   true),
    u("watt",      &["W"],                     Dimension::POWER,       1.0,   true),
    u("pascal",    &["Pa"],                    Dimension::PRESSURE,    1.0,   true),
    u("hertz",     &["Hz"],                    Dimension::FREQUENCY,   1.0,   true),
    u("coulomb",   &["C"],                     Dimension::CHARGE,      1.0,   true),
    u("volt",      &["V"],                     Dimension::VOLTAGE,     1.0,   true),
    u("farad",     &["F"],                     Dimension::CAPACITANCE, 1.0,   true),
    u("ohm",       &["\u{3a9}", "\u{2126}"],   Dimension::RESISTANCE,  1.0,   true),
    u("tesla",     &["T"],                     Dimension::MAGNETIC_FLUX_DENSITY, 1.0, true),
    // Radians and steradians are dimensionless by construction; they exist so that
    // `radian/second` reads as intended and prints as `1/s`.
    u("radian",    &["rad"],                   Dimension::DIMENSIONLESS, 1.0, true),
    u("steradian", &["sr"],                    Dimension::DIMENSIONLESS, 1.0, true),

    // ---- Accepted non-SI -------------------------------------------------------
    u("minute",    &["min"],                   Dimension::TIME,        60.0,  false),
    u("hour",      &["h", "hr"],               Dimension::TIME,        3600.0, false),
    u("day",       &["d"],                     Dimension::TIME,        86400.0, false),
    u("liter",     &["litre", "L"],            Dimension::VOLUME,      1e-3,  true),
    u("tonne",     &["t"],                     Dimension::MASS,        1e3,   true),
    u("angstrom",  &["\u{c5}", "\u{212b}"],    Dimension::LENGTH,      1e-10, false),
    u("degree",    &["deg", "\u{b0}"],         Dimension::DIMENSIONLESS, core::f64::consts::PI / 180.0, false),
    u("percent",   &["%"],                     Dimension::DIMENSIONLESS, 0.01, false),
    u("bar",       &[],                        Dimension::PRESSURE,    1e5,   true),
    u("atmosphere", &["atm"],                  Dimension::PRESSURE,    101_325.0, false),
    u("torr",      &["mmHg"],                  Dimension::PRESSURE,    101_325.0 / 760.0, false),
    u("calorie",   &["cal"],                   Dimension::ENERGY,      4.184, true),
    // 1 molar = 1 mol/L = 1000 mol/m^3. Note this is a *volumetric* concentration;
    // 2D scenes use mol/m^2 instead (spec §25.1).
    u("molar",     &["M"],                     Dimension::CONCENTRATION, 1e3, true),

    // ---- Atomic and molecular scale --------------------------------------------
    u("electronvolt", &["eV"],                 Dimension::ENERGY,      EV_JOULES, true),
    u("dalton",    &["Da", "amu", "atomic_mass_unit"], Dimension::MASS, AMU_KG, true),
    u("elementary_charge", &["e_charge"],      Dimension::CHARGE,      ELEMENTARY_CHARGE_C, false),
    u("electron_mass", &["m_e"],               Dimension::MASS,        ELECTRON_MASS_KG, false),
    u("proton_mass", &["m_p"],                 Dimension::MASS,        PROTON_MASS_KG, false),
    u("bohr",      &["a0", "bohr_radius"],     Dimension::LENGTH,      BOHR_RADIUS_M, false),
    u("hartree",   &["Eh"],                    Dimension::ENERGY,      HARTREE_J, false),

    // ---- Affine temperature scales ---------------------------------------------
    // These are deliberately last: they are the only entries with a nonzero offset,
    // and resolve() rejects them anywhere except as a standalone unit.
    UnitSpec {
        canonical: "celsius",
        aliases: &["degC", "\u{b0}C"],
        dim: Dimension::TEMPERATURE,
        scale: 1.0,
        offset: 273.15,
        prefixable: false,
    },
    UnitSpec {
        canonical: "fahrenheit",
        aliases: &["degF", "\u{b0}F"],
        dim: Dimension::TEMPERATURE,
        scale: F_SCALE,
        offset: F_OFFSET,
        prefixable: false,
    },
];

/// Name → unit lookup with SI prefix handling.
#[derive(Clone, Debug)]
pub struct UnitRegistry {
    defs: Vec<UnitDef>,
    /// Every spelling (canonical name and aliases) mapped to an index into `defs`.
    by_name: HashMap<String, usize>,
}

impl UnitRegistry {
    /// An empty registry. Only useful as a base for a fully custom vocabulary.
    pub fn empty() -> Self {
        Self { defs: Vec::new(), by_name: HashMap::new() }
    }

    /// The standard registry: SI base and derived units, accepted non-SI units,
    /// and atomic-scale units used by the chemistry and quantum modules.
    pub fn si() -> Self {
        let mut reg = Self::empty();
        for spec in STANDARD_UNITS {
            let def = UnitDef {
                canonical: spec.canonical.to_string(),
                dim: spec.dim,
                scale: spec.scale,
                offset: spec.offset,
                prefixable: spec.prefixable,
            };
            let index = reg.defs.len();
            reg.defs.push(def);
            reg.by_name.insert(spec.canonical.to_string(), index);
            for alias in spec.aliases {
                // A collision here is a bug in the table above, not user input.
                debug_assert!(
                    !reg.by_name.contains_key(*alias),
                    "duplicate unit spelling `{alias}` in the standard table"
                );
                reg.by_name.insert((*alias).to_string(), index);
            }
        }
        reg
    }

    /// Add a unit. Fails if any spelling is already taken.
    pub fn register(&mut self, def: UnitDef, aliases: &[&str]) -> Result<(), UnitError> {
        if self.by_name.contains_key(&def.canonical) {
            return Err(UnitError::DuplicateUnit { name: def.canonical });
        }
        for alias in aliases {
            if self.by_name.contains_key(*alias) {
                return Err(UnitError::DuplicateUnit { name: (*alias).to_string() });
            }
        }
        let index = self.defs.len();
        self.by_name.insert(def.canonical.clone(), index);
        for alias in aliases {
            self.by_name.insert((*alias).to_string(), index);
        }
        self.defs.push(def);
        Ok(())
    }

    /// Number of distinct units (not spellings).
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    /// True when no units are registered.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Resolve a unit name, applying an SI prefix if needed.
    ///
    /// ```
    /// use lattice_units::{Dimension, UnitRegistry};
    /// let reg = UnitRegistry::si();
    /// let nm = reg.resolve("nanometer").unwrap();
    /// assert_eq!(nm.dim(), Dimension::LENGTH);
    /// assert!((nm.scale() - 1e-9).abs() < 1e-24);
    /// // Symbol prefixes work too, and exact matches win over prefix decomposition.
    /// assert!((reg.resolve("nm").unwrap().scale() - 1e-9).abs() < 1e-24);
    /// assert_eq!(reg.resolve("cd").unwrap().def.canonical, "candela");
    /// ```
    pub fn resolve<'a>(&'a self, name: &str) -> Result<ResolvedUnit<'a>, UnitError> {
        if let Some(&index) = self.by_name.get(name) {
            return Ok(ResolvedUnit {
                def: &self.defs[index],
                prefix_factor: 1.0,
                prefix_name: None,
            });
        }

        if let Some(resolved) = self.resolve_prefixed(name) {
            return Ok(resolved);
        }

        Err(UnitError::UnknownUnit {
            name: name.to_string(),
            suggestion: self.suggest(name),
        })
    }

    /// Try to split `name` into a known prefix plus a prefixable unit.
    ///
    /// Long prefix names are tried before symbols, and within each family the longest
    /// match wins, so `dam` is deka-meter rather than deci-am-something.
    fn resolve_prefixed<'a>(&'a self, name: &str) -> Option<ResolvedUnit<'a>> {
        let mut candidates: Vec<(&'static str, &str, f64)> = Vec::new();
        for (long, symbol, factor) in PREFIXES {
            candidates.push((long, *long, *factor));
            candidates.push((long, *symbol, *factor));
        }
        for alias in MICRO_ALIASES {
            candidates.push(("micro", alias, 1e-6));
        }
        // Longest spelling first so `deka`/`da` beat `deci`/`d`.
        candidates.sort_by_key(|candidate| core::cmp::Reverse(candidate.1.len()));

        for (prefix_name, spelling, factor) in candidates {
            let Some(rest) = name.strip_prefix(spelling) else { continue };
            if rest.is_empty() {
                continue;
            }
            let Some(&index) = self.by_name.get(rest) else { continue };
            let def = &self.defs[index];
            if !def.prefixable {
                continue;
            }
            return Some(ResolvedUnit { def, prefix_factor: factor, prefix_name: Some(prefix_name) });
        }
        None
    }

    /// The closest known spelling to `name`, if one is close enough to be useful.
    ///
    /// Threshold scales with length so `metr` suggests `meter` but a wholly unrelated
    /// word suggests nothing — a bad suggestion is worse than none.
    fn suggest(&self, name: &str) -> Option<String> {
        let budget = match name.chars().count() {
            0..=3 => 1,
            4..=7 => 2,
            _ => 3,
        };
        let mut best: Option<(usize, &str)> = None;
        for spelling in self.by_name.keys() {
            let d = edit_distance(name, spelling, budget);
            if d <= budget && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, spelling));
            }
        }
        best.map(|(_, s)| s.to_string())
    }

    /// All registered spellings, sorted. Used by `lattice inspect units` and tests.
    pub fn spellings(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.by_name.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::si()
    }
}

/// Levenshtein distance, abandoning early once it exceeds `budget`.
///
/// The early exit matters because [`UnitRegistry::suggest`] runs this against every
/// registered spelling, and it only ever runs on an error path.
fn edit_distance(a: &str, b: &str, budget: usize) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > budget {
        return budget + 1;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        let mut row_min = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            row_min = row_min.min(curr[j]);
        }
        if row_min > budget {
            return budget + 1;
        }
        core::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_units_resolve_to_unit_scale() {
        let reg = UnitRegistry::si();
        for name in ["meter", "second", "ampere", "kelvin", "mole", "candela"] {
            let r = reg.resolve(name).unwrap();
            assert_eq!(r.scale(), 1.0, "{name}");
        }
    }

    #[test]
    fn kilogram_is_the_coherent_mass_unit() {
        let reg = UnitRegistry::si();
        // Resolved via kilo + gram, and must land exactly on 1.0.
        let kg = reg.resolve("kilogram").unwrap();
        assert_eq!(kg.dim(), Dimension::MASS);
        assert!((kg.scale() - 1.0).abs() < 1e-15);
        assert!((reg.resolve("kg").unwrap().scale() - 1.0).abs() < 1e-15);
        assert!((reg.resolve("g").unwrap().scale() - 1e-3).abs() < 1e-18);
    }

    #[test]
    fn long_and_symbol_prefixes_agree() {
        let reg = UnitRegistry::si();
        for (long, short) in [
            ("nanometer", "nm"),
            ("femtosecond", "fs"),
            ("kilojoule", "kJ"),
            ("millimeter", "mm"),
            ("megapascal", "MPa"),
            ("microsecond", "\u{b5}s"),
        ] {
            let a = reg.resolve(long).unwrap();
            let b = reg.resolve(short).unwrap();
            assert_eq!(a.dim(), b.dim(), "{long} vs {short}");
            assert!(
                (a.scale() - b.scale()).abs() <= a.scale().abs() * 1e-12,
                "{long}={} but {short}={}",
                a.scale(),
                b.scale()
            );
        }
    }

    #[test]
    fn ascii_and_greek_micro_are_accepted() {
        let reg = UnitRegistry::si();
        for spelling in ["\u{b5}m", "um", "\u{3bc}m", "micrometer"] {
            let r = reg.resolve(spelling).unwrap();
            assert_eq!(r.dim(), Dimension::LENGTH);
            assert!((r.scale() - 1e-6).abs() < 1e-18, "{spelling}");
        }
    }

    /// Exact match must beat prefix decomposition, or `cd` becomes centi-day.
    #[test]
    fn exact_match_wins_over_prefix_decomposition() {
        let reg = UnitRegistry::si();
        assert_eq!(reg.resolve("cd").unwrap().def.canonical, "candela");
        assert_eq!(reg.resolve("min").unwrap().def.canonical, "minute");
        assert_eq!(reg.resolve("h").unwrap().def.canonical, "hour");
        assert_eq!(reg.resolve("T").unwrap().def.canonical, "tesla");
        assert_eq!(reg.resolve("M").unwrap().def.canonical, "molar");
        assert_eq!(reg.resolve("mol").unwrap().def.canonical, "mole");
    }

    #[test]
    fn longest_prefix_wins() {
        let reg = UnitRegistry::si();
        // `dam` must be deka-meter (10 m), not deci-anything.
        assert!((reg.resolve("dam").unwrap().scale() - 10.0).abs() < 1e-12);
        assert!((reg.resolve("dm").unwrap().scale() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn non_prefixable_units_reject_prefixes() {
        let reg = UnitRegistry::si();
        // There is no such thing as a "kilominute".
        assert!(reg.resolve("kilominute").is_err());
        assert!(reg.resolve("millicelsius").is_err());
    }

    #[test]
    fn unknown_units_suggest_close_spellings() {
        let reg = UnitRegistry::si();
        let err = reg.resolve("metre_").unwrap_err();
        match err {
            UnitError::UnknownUnit { suggestion: Some(s), .. } => {
                assert_eq!(s, "metre");
            }
            other => panic!("expected a suggestion, got {other:?}"),
        }
    }

    #[test]
    fn wildly_unknown_units_suggest_nothing() {
        let reg = UnitRegistry::si();
        match reg.resolve("zzzzzzzzzzzz").unwrap_err() {
            UnitError::UnknownUnit { suggestion, .. } => assert_eq!(suggestion, None),
            other => panic!("expected UnknownUnit, got {other:?}"),
        }
    }

    #[test]
    fn affine_scales_carry_their_offset() {
        let reg = UnitRegistry::si();
        let c = reg.resolve("celsius").unwrap();
        assert_eq!(c.offset(), 273.15);
        assert!(c.def.is_affine());

        let f = reg.resolve("fahrenheit").unwrap();
        // 32 °F is exactly 273.15 K.
        let kelvin = 32.0 * f.scale() + f.offset();
        assert!((kelvin - 273.15).abs() < 1e-10, "got {kelvin}");
        // 212 °F is exactly 373.15 K.
        let boiling = 212.0 * f.scale() + f.offset();
        assert!((boiling - 373.15).abs() < 1e-10, "got {boiling}");
    }

    #[test]
    fn atomic_units_have_expected_magnitudes() {
        let reg = UnitRegistry::si();
        let ev = reg.resolve("electronvolt").unwrap();
        assert_eq!(ev.dim(), Dimension::ENERGY);
        assert!((ev.scale() - 1.602_176_634e-19).abs() < 1e-30);
        // keV/MeV must work by prefix.
        assert!((reg.resolve("keV").unwrap().scale() / ev.scale() - 1e3).abs() < 1e-6);

        let me = reg.resolve("electron_mass").unwrap();
        assert_eq!(me.dim(), Dimension::MASS);
        assert!((me.scale() - 9.109_383_713_9e-31).abs() < 1e-42);
    }

    #[test]
    fn custom_units_can_be_registered() {
        let mut reg = UnitRegistry::si();
        reg.register(
            UnitDef {
                canonical: "lattice_cell".to_string(),
                dim: Dimension::LENGTH,
                scale: 2.5e-3,
                offset: 0.0,
                prefixable: false,
            },
            &["cell"],
        )
        .unwrap();
        assert!((reg.resolve("cell").unwrap().scale() - 2.5e-3).abs() < 1e-18);
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut reg = UnitRegistry::si();
        let def = UnitDef {
            canonical: "meter".to_string(),
            dim: Dimension::LENGTH,
            scale: 1.0,
            offset: 0.0,
            prefixable: false,
        };
        assert!(matches!(reg.register(def, &[]), Err(UnitError::DuplicateUnit { .. })));
    }

    /// Every alias in the standard table must be reachable and point at a unit
    /// whose dimension matches its canonical entry. Guards against typos in the table.
    #[test]
    fn every_standard_spelling_resolves() {
        let reg = UnitRegistry::si();
        for spelling in reg.spellings() {
            let resolved = reg.resolve(spelling).unwrap_or_else(|e| panic!("{spelling}: {e}"));
            assert!(resolved.scale().is_finite(), "{spelling} has non-finite scale");
            assert!(resolved.scale() != 0.0, "{spelling} has zero scale");
        }
    }

    #[test]
    fn edit_distance_is_correct_and_bounded() {
        assert_eq!(edit_distance("meter", "meter", 3), 0);
        assert_eq!(edit_distance("metre", "meter", 3), 2);
        assert_eq!(edit_distance("kitten", "sitting", 5), 3);
        // Exceeding the budget returns budget+1 rather than the true distance.
        assert_eq!(edit_distance("abc", "xyzxyz", 1), 2);
    }
}
