//! Physical constants, in coherent SI base units.
//!
//! Values follow the 2019 SI redefinition (which makes `c`, `h`, `e`, `k_B` and
//! `N_A` exact by definition) and CODATA 2022 for measured quantities. Each entry
//! notes which it is, because "exact" and "measured to 10 digits" are different
//! claims and a simulation engine should not blur them (spec P1).
//!
//! The raw `f64` magnitudes live in [`value`] so the unit registry and the typed
//! [`Quantity`] constants below read from one source. A constant that appears twice
//! with different digits is a bug waiting to happen.

use crate::dimension::Dimension;
use crate::quantity::Quantity;

/// Raw SI magnitudes, for code that wants the number without the dimension.
pub mod value {
    /// Speed of light in vacuum, m/s. Exact by definition of the metre.
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
    /// Planck constant, J·s. Exact by definition of the kilogram.
    pub const PLANCK: f64 = 6.626_070_15e-34;
    /// Reduced Planck constant ħ = h/2π, J·s. Exact (irrational, so truncated).
    pub const REDUCED_PLANCK: f64 = 1.054_571_817_646_156e-34;
    /// Elementary charge, C. Exact by definition of the ampere.
    pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
    /// Boltzmann constant, J/K. Exact by definition of the kelvin.
    pub const BOLTZMANN: f64 = 1.380_649e-23;
    /// Avogadro constant, 1/mol. Exact by definition of the mole.
    pub const AVOGADRO: f64 = 6.022_140_76e23;
    /// Molar gas constant R = N_A·k_B, J/(mol·K). Exact (product of two exact values).
    pub const MOLAR_GAS: f64 = 8.314_462_618_153_24;
    /// Faraday constant F = N_A·e, C/mol. Exact.
    pub const FARADAY: f64 = 96_485.332_123_310_02;
    /// One electronvolt in joules. Exact.
    pub const ELECTRONVOLT: f64 = ELEMENTARY_CHARGE;

    /// Electron rest mass, kg. CODATA 2022, relative uncertainty 3.1e-10.
    pub const ELECTRON_MASS: f64 = 9.109_383_713_9e-31;
    /// Proton rest mass, kg. CODATA 2022.
    pub const PROTON_MASS: f64 = 1.672_621_925_95e-27;
    /// Neutron rest mass, kg. CODATA 2022.
    pub const NEUTRON_MASS: f64 = 1.674_927_500_56e-27;
    /// Unified atomic mass unit (dalton), kg. CODATA 2022.
    pub const ATOMIC_MASS_UNIT: f64 = 1.660_539_068_92e-27;
    /// Bohr radius, m. CODATA 2022.
    pub const BOHR_RADIUS: f64 = 5.291_772_105_44e-11;
    /// Hartree energy, J. CODATA 2022.
    pub const HARTREE: f64 = 4.359_744_722_206_0e-18;
    /// Fine-structure constant, dimensionless. CODATA 2022.
    pub const FINE_STRUCTURE: f64 = 7.297_352_564_3e-3;

    /// Vacuum electric permittivity ε₀, F/m. CODATA 2022 (no longer exact after 2019).
    pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_818_8e-12;
    /// Vacuum magnetic permeability μ₀, N/A². CODATA 2022.
    pub const VACUUM_PERMEABILITY: f64 = 1.256_637_061_27e-6;
    /// Coulomb constant 1/(4πε₀), N·m²/C².
    pub const COULOMB: f64 = 8.987_551_792_3e9;

    /// Newtonian constant of gravitation, m³/(kg·s²). CODATA 2022 — the least
    /// precisely known constant here, relative uncertainty 2.2e-5.
    pub const GRAVITATIONAL: f64 = 6.674_30e-11;
    /// Standard acceleration of gravity, m/s². Exact by convention (not a measurement
    /// of Earth's actual local gravity, which varies by roughly ±0.3%).
    pub const STANDARD_GRAVITY: f64 = 9.806_65;
    /// Stefan–Boltzmann constant, W/(m²·K⁴). Exact (derived from exact constants).
    pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
}

// Dimensions that only constants need. Built inline rather than added to the public
// `Dimension` constant set, which is scoped to what models actually declare.
// Order is [length, mass, time, current, temperature, amount, luminous].

/// 1/mol.
const INVERSE_AMOUNT: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, -1, 0]);
/// C/mol.
const CHARGE_PER_AMOUNT: Dimension = Dimension::from_exponents([0, 0, 1, 1, 0, -1, 0]);
/// F/m = A²·s⁴/(kg·m³).
const PERMITTIVITY: Dimension = Dimension::from_exponents([-3, -1, 4, 2, 0, 0, 0]);
/// N/A² = kg·m/(s²·A²).
const PERMEABILITY: Dimension = Dimension::from_exponents([1, 1, -2, -2, 0, 0, 0]);
/// N·m²/C² = kg·m³/(s⁴·A²).
const COULOMB_CONSTANT: Dimension = Dimension::from_exponents([3, 1, -4, -2, 0, 0, 0]);
/// m³/(kg·s²).
const GRAVITATIONAL_CONSTANT: Dimension = Dimension::from_exponents([3, -1, -2, 0, 0, 0, 0]);
/// W/(m²·K⁴) = kg/(s³·K⁴).
const STEFAN_BOLTZMANN_CONSTANT: Dimension = Dimension::from_exponents([0, 1, -3, 0, -4, 0, 0]);

/// Speed of light in vacuum. Exact.
pub const SPEED_OF_LIGHT: Quantity = Quantity::new(value::SPEED_OF_LIGHT, Dimension::VELOCITY);
/// Planck constant. Exact.
pub const PLANCK: Quantity = Quantity::new(value::PLANCK, Dimension::ACTION);
/// Reduced Planck constant ħ. Exact.
pub const REDUCED_PLANCK: Quantity = Quantity::new(value::REDUCED_PLANCK, Dimension::ACTION);
/// Elementary charge. Exact.
pub const ELEMENTARY_CHARGE: Quantity =
    Quantity::new(value::ELEMENTARY_CHARGE, Dimension::CHARGE);
/// Boltzmann constant. Exact.
pub const BOLTZMANN: Quantity = Quantity::new(value::BOLTZMANN, Dimension::HEAT_CAPACITY);
/// Avogadro constant. Exact.
pub const AVOGADRO: Quantity = Quantity::new(value::AVOGADRO, INVERSE_AMOUNT);
/// Molar gas constant R. Exact.
pub const MOLAR_GAS: Quantity =
    Quantity::new(value::MOLAR_GAS, Dimension::MOLAR_HEAT_CAPACITY);
/// Faraday constant. Exact.
pub const FARADAY: Quantity = Quantity::new(value::FARADAY, CHARGE_PER_AMOUNT);

/// Electron rest mass. CODATA 2022.
pub const ELECTRON_MASS: Quantity = Quantity::new(value::ELECTRON_MASS, Dimension::MASS);
/// Proton rest mass. CODATA 2022.
pub const PROTON_MASS: Quantity = Quantity::new(value::PROTON_MASS, Dimension::MASS);
/// Neutron rest mass. CODATA 2022.
pub const NEUTRON_MASS: Quantity = Quantity::new(value::NEUTRON_MASS, Dimension::MASS);
/// Unified atomic mass unit. CODATA 2022.
pub const ATOMIC_MASS_UNIT: Quantity =
    Quantity::new(value::ATOMIC_MASS_UNIT, Dimension::MASS);
/// Bohr radius. CODATA 2022.
pub const BOHR_RADIUS: Quantity = Quantity::new(value::BOHR_RADIUS, Dimension::LENGTH);
/// Hartree energy. CODATA 2022.
pub const HARTREE: Quantity = Quantity::new(value::HARTREE, Dimension::ENERGY);
/// Fine-structure constant. CODATA 2022.
pub const FINE_STRUCTURE: Quantity =
    Quantity::new(value::FINE_STRUCTURE, Dimension::DIMENSIONLESS);

/// Vacuum electric permittivity ε₀. CODATA 2022.
pub const VACUUM_PERMITTIVITY: Quantity =
    Quantity::new(value::VACUUM_PERMITTIVITY, PERMITTIVITY);
/// Vacuum magnetic permeability μ₀. CODATA 2022.
pub const VACUUM_PERMEABILITY: Quantity =
    Quantity::new(value::VACUUM_PERMEABILITY, PERMEABILITY);
/// Coulomb constant 1/(4πε₀).
pub const COULOMB: Quantity = Quantity::new(value::COULOMB, COULOMB_CONSTANT);

/// Newtonian constant of gravitation. CODATA 2022.
pub const GRAVITATIONAL: Quantity =
    Quantity::new(value::GRAVITATIONAL, GRAVITATIONAL_CONSTANT);
/// Standard acceleration of gravity. Exact by convention.
pub const STANDARD_GRAVITY: Quantity =
    Quantity::new(value::STANDARD_GRAVITY, Dimension::ACCELERATION);
/// Stefan–Boltzmann constant.
pub const STEFAN_BOLTZMANN: Quantity =
    Quantity::new(value::STEFAN_BOLTZMANN, STEFAN_BOLTZMANN_CONSTANT);

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs()
    }

    /// R = N_A · k_B is an identity in the 2019 SI, not an approximation. If these
    /// literals ever drift apart, this test says so.
    #[test]
    fn gas_constant_is_avogadro_times_boltzmann() {
        let derived = AVOGADRO * BOLTZMANN;
        assert_eq!(derived.dimension(), MOLAR_GAS.dimension());
        assert!(
            rel_close(derived.value(), MOLAR_GAS.value(), 1e-14),
            "N_A·k_B = {} but R = {}",
            derived.value(),
            MOLAR_GAS.value()
        );
    }

    /// F = N_A · e, likewise exact.
    #[test]
    fn faraday_is_avogadro_times_elementary_charge() {
        let derived = AVOGADRO * ELEMENTARY_CHARGE;
        assert_eq!(derived.dimension(), FARADAY.dimension());
        assert!(rel_close(derived.value(), FARADAY.value(), 1e-14));
    }

    /// ħ = h / 2π.
    #[test]
    fn reduced_planck_is_planck_over_two_pi() {
        let derived = PLANCK / (2.0 * core::f64::consts::PI);
        assert!(rel_close(derived.value(), REDUCED_PLANCK.value(), 1e-14));
    }

    /// c² = 1/(ε₀·μ₀). This is a real consistency check across three independently
    /// listed values, and it catches a transposed digit in any of them.
    #[test]
    fn permittivity_and_permeability_reproduce_the_speed_of_light() {
        let c_squared = (VACUUM_PERMITTIVITY * VACUUM_PERMEABILITY).recip().unwrap();
        let expected = SPEED_OF_LIGHT.try_powi(2).unwrap();
        assert_eq!(c_squared.dimension(), expected.dimension());
        // CODATA values for ε₀ and μ₀ are measured, so this holds to their precision
        // rather than exactly.
        assert!(
            rel_close(c_squared.value(), expected.value(), 1e-9),
            "1/(e0*u0) = {} but c^2 = {}",
            c_squared.value(),
            expected.value()
        );
    }

    /// k_e = 1/(4πε₀).
    #[test]
    fn coulomb_constant_follows_from_permittivity() {
        let derived = (4.0 * core::f64::consts::PI * VACUUM_PERMITTIVITY).recip().unwrap();
        assert_eq!(derived.dimension(), COULOMB.dimension());
        assert!(rel_close(derived.value(), COULOMB.value(), 1e-9));
    }

    /// α = e²/(4πε₀ħc). Ties the electromagnetic constants to the atomic ones.
    #[test]
    fn fine_structure_follows_from_the_other_constants() {
        let numerator = ELEMENTARY_CHARGE.try_powi(2).unwrap();
        let denominator = 4.0
            * core::f64::consts::PI
            * VACUUM_PERMITTIVITY
            * REDUCED_PLANCK
            * SPEED_OF_LIGHT;
        let derived = numerator / denominator;
        assert!(derived.dimension().is_dimensionless(), "{}", derived.dimension());
        assert!(
            rel_close(derived.value(), FINE_STRUCTURE.value(), 1e-8),
            "derived alpha = {} but listed = {}",
            derived.value(),
            FINE_STRUCTURE.value()
        );
    }

    /// a₀ = ħ/(m_e·c·α), and E_h = α²·m_e·c². Two independent ties between the
    /// atomic-scale constants.
    #[test]
    fn bohr_radius_and_hartree_are_consistent() {
        let a0 = REDUCED_PLANCK / (ELECTRON_MASS * SPEED_OF_LIGHT * FINE_STRUCTURE);
        assert_eq!(a0.dimension(), Dimension::LENGTH);
        assert!(rel_close(a0.value(), BOHR_RADIUS.value(), 1e-8), "a0 = {}", a0.value());

        let e_h = FINE_STRUCTURE.try_powi(2).unwrap()
            * ELECTRON_MASS
            * SPEED_OF_LIGHT.try_powi(2).unwrap();
        assert_eq!(e_h.dimension(), Dimension::ENERGY);
        assert!(rel_close(e_h.value(), HARTREE.value(), 1e-8), "E_h = {}", e_h.value());
    }

    /// Before 2019, the molar mass constant `M_u` was *exactly* 1 g/mol, so the
    /// dalton was exactly `(1 g/mol)/N_A`. The 2019 redefinition fixed `N_A` instead
    /// and demoted `M_u` to a measured quantity, so the two now differ by about one
    /// part in 10^9.
    ///
    /// The tolerance here is set to admit exactly that discrepancy. It is not slack —
    /// tightening it would be asserting a relationship physics no longer guarantees.
    #[test]
    fn atomic_mass_unit_matches_avogadro_to_the_2019_tolerance() {
        let gram_per_mole = Quantity::new(1e-3, Dimension::MOLAR_MASS);
        let per_particle = gram_per_mole / AVOGADRO;
        assert_eq!(per_particle.dimension(), Dimension::MASS);

        let relative = (per_particle.value() - ATOMIC_MASS_UNIT.value()).abs()
            / ATOMIC_MASS_UNIT.value();
        assert!(
            relative < 5e-9,
            "(1 g/mol)/N_A = {} vs dalton = {} (relative {relative:e})",
            per_particle.value(),
            ATOMIC_MASS_UNIT.value()
        );
        // ...and confirm the gap is real rather than an artifact of a typo: the two
        // values must not be bit-identical.
        assert!(relative > 1e-12, "M_u is no longer exactly 1 g/mol; check the literals");
    }

    /// Every constant must have a finite, nonzero magnitude — a smoke test that
    /// catches a mistyped exponent producing 0 or inf.
    #[test]
    fn all_constants_are_finite_and_nonzero() {
        let all: &[(&str, Quantity)] = &[
            ("SPEED_OF_LIGHT", SPEED_OF_LIGHT),
            ("PLANCK", PLANCK),
            ("REDUCED_PLANCK", REDUCED_PLANCK),
            ("ELEMENTARY_CHARGE", ELEMENTARY_CHARGE),
            ("BOLTZMANN", BOLTZMANN),
            ("AVOGADRO", AVOGADRO),
            ("MOLAR_GAS", MOLAR_GAS),
            ("FARADAY", FARADAY),
            ("ELECTRON_MASS", ELECTRON_MASS),
            ("PROTON_MASS", PROTON_MASS),
            ("NEUTRON_MASS", NEUTRON_MASS),
            ("ATOMIC_MASS_UNIT", ATOMIC_MASS_UNIT),
            ("BOHR_RADIUS", BOHR_RADIUS),
            ("HARTREE", HARTREE),
            ("FINE_STRUCTURE", FINE_STRUCTURE),
            ("VACUUM_PERMITTIVITY", VACUUM_PERMITTIVITY),
            ("VACUUM_PERMEABILITY", VACUUM_PERMEABILITY),
            ("COULOMB", COULOMB),
            ("GRAVITATIONAL", GRAVITATIONAL),
            ("STANDARD_GRAVITY", STANDARD_GRAVITY),
            ("STEFAN_BOLTZMANN", STEFAN_BOLTZMANN),
        ];
        for (name, q) in all {
            assert!(q.is_finite(), "{name} is not finite");
            assert!(q.value() != 0.0, "{name} is zero");
        }
    }

    /// The proton is about 1836 times heavier than the electron — a number physicists
    /// know by heart, which makes it a good check that neither mass is off by a
    /// decade.
    #[test]
    fn proton_to_electron_mass_ratio_is_familiar() {
        let ratio = (PROTON_MASS / ELECTRON_MASS).require_dimensionless().unwrap();
        assert!(rel_close(ratio, 1836.152673426, 1e-9), "ratio = {ratio}");
    }
}
