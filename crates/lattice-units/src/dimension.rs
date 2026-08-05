//! Physical dimensions as integer exponent vectors over the seven SI base dimensions.
//!
//! A [`Dimension`] is the *kind* of a physical quantity, independent of the unit it
//! is written in: `9.81 meter/second^2` and `32.2 foot/second^2` share the dimension
//! `L·T^-2`. Spec FR-002 requires the compiler to reject dimensionally inconsistent
//! expressions, and this type is what makes that check cheap enough to run on every
//! node of every expression.
//!
//! # Representation
//!
//! Exponents are `i8`, one per base dimension, in the fixed order
//! `[length, mass, time, current, temperature, amount, luminous]`. The whole struct
//! is 7 bytes, `Copy`, and `Hash` — dimensions are compared and hashed constantly
//! during compilation, so this matters.
//!
//! Exponents are *integers*, not rationals. This is a deliberate limitation: it makes
//! equality exact and the representation compact, at the cost of not being able to
//! express `sqrt(length)` as a dimension. [`Dimension::root`] therefore returns an
//! error rather than silently rounding — see spec P1 (scientific honesty).

use core::fmt;

/// Number of SI base dimensions.
pub const NUM_BASE: usize = 7;

/// The seven SI base dimensions, in canonical storage order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum BaseDim {
    Length = 0,
    Mass = 1,
    Time = 2,
    Current = 3,
    Temperature = 4,
    Amount = 5,
    Luminous = 6,
}

impl BaseDim {
    /// All base dimensions in canonical storage order.
    pub const ALL: [BaseDim; NUM_BASE] = [
        BaseDim::Length,
        BaseDim::Mass,
        BaseDim::Time,
        BaseDim::Current,
        BaseDim::Temperature,
        BaseDim::Amount,
        BaseDim::Luminous,
    ];

    /// The SI symbol of the coherent base *unit* for this dimension (`kg`, not `g`).
    pub const fn si_symbol(self) -> &'static str {
        match self {
            BaseDim::Length => "m",
            BaseDim::Mass => "kg",
            BaseDim::Time => "s",
            BaseDim::Current => "A",
            BaseDim::Temperature => "K",
            BaseDim::Amount => "mol",
            BaseDim::Luminous => "cd",
        }
    }

    /// The name of the dimension itself, for diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            BaseDim::Length => "length",
            BaseDim::Mass => "mass",
            BaseDim::Time => "time",
            BaseDim::Current => "electric current",
            BaseDim::Temperature => "thermodynamic temperature",
            BaseDim::Amount => "amount of substance",
            BaseDim::Luminous => "luminous intensity",
        }
    }

    /// Index into a [`Dimension`]'s exponent array.
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Errors from dimensional arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimensionError {
    /// An exponent left the representable `i8` range.
    ExponentOverflow {
        base: BaseDim,
    },
    /// A root was requested that would produce a fractional exponent.
    ///
    /// Carries the base dimension and exponent that failed to divide evenly, so the
    /// diagnostic can say *why* rather than just "bad root".
    NonIntegerRoot {
        base: BaseDim,
        exponent: i8,
        degree: i32,
    },
    /// A root of degree zero (or negative) was requested.
    InvalidRootDegree {
        degree: i32,
    },
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimensionError::ExponentOverflow { base } => {
                write!(f, "dimension exponent for {} overflowed the representable range (-128..=127)", base.name())
            }
            DimensionError::NonIntegerRoot { base, exponent, degree } => write!(
                f,
                "cannot take the degree-{degree} root: {} has exponent {exponent}, which is not divisible by {degree}",
                base.name()
            ),
            DimensionError::InvalidRootDegree { degree } => {
                write!(f, "root degree must be a positive integer, got {degree}")
            }
        }
    }
}

impl core::error::Error for DimensionError {}

/// The dimension of a physical quantity: an exponent vector over the SI base dimensions.
///
/// Construct from the named constants ([`Dimension::FORCE`], [`Dimension::ENERGY`], …)
/// or from an explicit exponent array via [`Dimension::from_exponents`].
///
/// ```
/// use lattice_units::Dimension;
/// // force = mass × acceleration
/// let force = Dimension::MASS.try_mul(Dimension::ACCELERATION).unwrap();
/// assert_eq!(force, Dimension::FORCE);
/// assert_eq!(force.to_string(), "kg·m/s^2");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dimension {
    e: [i8; NUM_BASE],
}

impl Dimension {
    /// Build a dimension from raw exponents in canonical order.
    ///
    /// Order is `[length, mass, time, current, temperature, amount, luminous]`.
    pub const fn from_exponents(e: [i8; NUM_BASE]) -> Self {
        Self { e }
    }

    /// The raw exponent array in canonical order.
    pub const fn exponents(&self) -> [i8; NUM_BASE] {
        self.e
    }

    /// The exponent of a single base dimension.
    pub const fn exponent(&self, base: BaseDim) -> i8 {
        self.e[base as usize]
    }

    /// True for pure numbers (all exponents zero).
    pub const fn is_dimensionless(&self) -> bool {
        let mut i = 0;
        while i < NUM_BASE {
            if self.e[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Multiply dimensions (add exponents), erroring on overflow.
    pub fn try_mul(self, rhs: Self) -> Result<Self, DimensionError> {
        let mut e = [0i8; NUM_BASE];
        for (i, slot) in e.iter_mut().enumerate() {
            *slot = self.e[i]
                .checked_add(rhs.e[i])
                .ok_or(DimensionError::ExponentOverflow { base: BaseDim::ALL[i] })?;
        }
        Ok(Self { e })
    }

    /// Divide dimensions (subtract exponents), erroring on overflow.
    pub fn try_div(self, rhs: Self) -> Result<Self, DimensionError> {
        let mut e = [0i8; NUM_BASE];
        for (i, slot) in e.iter_mut().enumerate() {
            *slot = self.e[i]
                .checked_sub(rhs.e[i])
                .ok_or(DimensionError::ExponentOverflow { base: BaseDim::ALL[i] })?;
        }
        Ok(Self { e })
    }

    /// Raise to an integer power (multiply exponents), erroring on overflow.
    pub fn try_powi(self, n: i32) -> Result<Self, DimensionError> {
        let mut e = [0i8; NUM_BASE];
        for (i, slot) in e.iter_mut().enumerate() {
            let scaled = i32::from(self.e[i])
                .checked_mul(n)
                .ok_or(DimensionError::ExponentOverflow { base: BaseDim::ALL[i] })?;
            *slot = i8::try_from(scaled)
                .map_err(|_| DimensionError::ExponentOverflow { base: BaseDim::ALL[i] })?;
        }
        Ok(Self { e })
    }

    /// Reciprocal (negate exponents).
    pub fn try_recip(self) -> Result<Self, DimensionError> {
        Dimension::DIMENSIONLESS.try_div(self)
    }

    /// Take the degree-`n` root, erroring if any exponent is not divisible by `n`.
    ///
    /// Integer exponents cannot represent `sqrt(length)`, so this fails loudly rather
    /// than rounding. Callers that genuinely need a half-power (rare outside of
    /// dimensional-analysis scaling arguments) must restructure the expression.
    pub fn root(self, n: i32) -> Result<Self, DimensionError> {
        if n <= 0 {
            return Err(DimensionError::InvalidRootDegree { degree: n });
        }
        let mut e = [0i8; NUM_BASE];
        for (i, slot) in e.iter_mut().enumerate() {
            let exponent = self.e[i];
            if i32::from(exponent) % n != 0 {
                return Err(DimensionError::NonIntegerRoot {
                    base: BaseDim::ALL[i],
                    exponent,
                    degree: n,
                });
            }
            *slot = (i32::from(exponent) / n) as i8;
        }
        Ok(Self { e })
    }

    /// The conventional name of this dimension, if it has one.
    ///
    /// Used by diagnostics so an error can read "expected force, found energy"
    /// instead of only showing exponent vectors.
    pub fn name(&self) -> Option<&'static str> {
        NAMED_DIMENSIONS
            .iter()
            .find(|(dim, _)| dim.e == self.e)
            .map(|(_, name)| *name)
    }

    /// A human-readable description combining name and SI form, for diagnostics.
    ///
    /// `"force (kg·m/s^2)"`, or just `"kg·m^2/s^3/K"` when the dimension is unnamed.
    pub fn describe(&self) -> String {
        match self.name() {
            Some(name) => format!("{name} ({self})"),
            None => self.to_string(),
        }
    }

    // ---- Base dimensions ------------------------------------------------------

    pub const DIMENSIONLESS: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 0, 0]);
    pub const LENGTH: Dimension = Dimension::from_exponents([1, 0, 0, 0, 0, 0, 0]);
    pub const MASS: Dimension = Dimension::from_exponents([0, 1, 0, 0, 0, 0, 0]);
    pub const TIME: Dimension = Dimension::from_exponents([0, 0, 1, 0, 0, 0, 0]);
    pub const CURRENT: Dimension = Dimension::from_exponents([0, 0, 0, 1, 0, 0, 0]);
    pub const TEMPERATURE: Dimension = Dimension::from_exponents([0, 0, 0, 0, 1, 0, 0]);
    pub const AMOUNT: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 1, 0]);
    pub const LUMINOUS: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 0, 1]);

    // ---- Geometry and kinematics ----------------------------------------------

    pub const AREA: Dimension = Dimension::from_exponents([2, 0, 0, 0, 0, 0, 0]);
    pub const VOLUME: Dimension = Dimension::from_exponents([3, 0, 0, 0, 0, 0, 0]);
    pub const VELOCITY: Dimension = Dimension::from_exponents([1, 0, -1, 0, 0, 0, 0]);
    pub const ACCELERATION: Dimension = Dimension::from_exponents([1, 0, -2, 0, 0, 0, 0]);
    pub const FREQUENCY: Dimension = Dimension::from_exponents([0, 0, -1, 0, 0, 0, 0]);
    pub const ANGULAR_VELOCITY: Dimension = Dimension::FREQUENCY;

    // ---- Mechanics -------------------------------------------------------------

    pub const FORCE: Dimension = Dimension::from_exponents([1, 1, -2, 0, 0, 0, 0]);
    pub const ENERGY: Dimension = Dimension::from_exponents([2, 1, -2, 0, 0, 0, 0]);
    pub const POWER: Dimension = Dimension::from_exponents([2, 1, -3, 0, 0, 0, 0]);
    pub const PRESSURE: Dimension = Dimension::from_exponents([-1, 1, -2, 0, 0, 0, 0]);
    pub const MOMENTUM: Dimension = Dimension::from_exponents([1, 1, -1, 0, 0, 0, 0]);
    pub const ACTION: Dimension = Dimension::from_exponents([2, 1, -1, 0, 0, 0, 0]);
    pub const MASS_DENSITY: Dimension = Dimension::from_exponents([-3, 1, 0, 0, 0, 0, 0]);
    /// Mass per unit area — the 2D analogue of density, which is what a 2D-first
    /// engine actually uses (spec P9).
    pub const AREAL_MASS_DENSITY: Dimension = Dimension::from_exponents([-2, 1, 0, 0, 0, 0, 0]);
    pub const DYNAMIC_VISCOSITY: Dimension = Dimension::from_exponents([-1, 1, -1, 0, 0, 0, 0]);
    /// Also kinematic viscosity and thermal diffusivity: `L^2·T^-1`.
    pub const DIFFUSIVITY: Dimension = Dimension::from_exponents([2, 0, -1, 0, 0, 0, 0]);
    pub const STIFFNESS: Dimension = Dimension::from_exponents([0, 1, -2, 0, 0, 0, 0]);
    /// A linear damping coefficient, `N·s/m` — the `c` in `F = −c·v`.
    pub const DAMPING: Dimension = Dimension::from_exponents([0, 1, -1, 0, 0, 0, 0]);
    /// Torque, `N·m`.
    ///
    /// Dimensionally identical to [`Dimension::ENERGY`], which is not a mistake: SI
    /// genuinely cannot tell a newton-metre of torque from a joule of work, because
    /// angle is dimensionless. Naming it separately documents intent at the call site
    /// without pretending the units differ.
    pub const TORQUE: Dimension = Dimension::ENERGY;

    // ---- Electromagnetism ------------------------------------------------------

    pub const CHARGE: Dimension = Dimension::from_exponents([0, 0, 1, 1, 0, 0, 0]);
    pub const VOLTAGE: Dimension = Dimension::from_exponents([2, 1, -3, -1, 0, 0, 0]);
    pub const ELECTRIC_FIELD: Dimension = Dimension::from_exponents([1, 1, -3, -1, 0, 0, 0]);
    pub const CAPACITANCE: Dimension = Dimension::from_exponents([-2, -1, 4, 2, 0, 0, 0]);
    pub const RESISTANCE: Dimension = Dimension::from_exponents([2, 1, -3, -2, 0, 0, 0]);
    pub const MAGNETIC_FLUX_DENSITY: Dimension = Dimension::from_exponents([0, 1, -2, -1, 0, 0, 0]);

    // ---- Thermal ---------------------------------------------------------------

    pub const HEAT_CAPACITY: Dimension = Dimension::from_exponents([2, 1, -2, 0, -1, 0, 0]);
    pub const SPECIFIC_HEAT_CAPACITY: Dimension = Dimension::from_exponents([2, 0, -2, 0, -1, 0, 0]);
    pub const THERMAL_CONDUCTIVITY: Dimension = Dimension::from_exponents([1, 1, -3, 0, -1, 0, 0]);
    pub const ENTROPY: Dimension = Dimension::HEAT_CAPACITY;

    // ---- Chemistry -------------------------------------------------------------

    pub const MOLAR_MASS: Dimension = Dimension::from_exponents([0, 1, 0, 0, 0, -1, 0]);
    pub const MOLAR_ENERGY: Dimension = Dimension::from_exponents([2, 1, -2, 0, 0, -1, 0]);
    pub const MOLAR_HEAT_CAPACITY: Dimension = Dimension::from_exponents([2, 1, -2, 0, -1, -1, 0]);
    pub const CONCENTRATION: Dimension = Dimension::from_exponents([-3, 0, 0, 0, 0, 1, 0]);
    /// Amount per unit area — how species are stored on a 2D grid (spec §25.1 uses
    /// `1 mole / meter^2`).
    pub const AREAL_CONCENTRATION: Dimension = Dimension::from_exponents([-2, 0, 0, 0, 0, 1, 0]);
    /// Rate of change of areal concentration, the unit of a 2D reaction source term.
    pub const AREAL_REACTION_RATE: Dimension = Dimension::from_exponents([-2, 0, -1, 0, 0, 1, 0]);
    pub const CATALYTIC_ACTIVITY: Dimension = Dimension::from_exponents([0, 0, -1, 0, 0, 1, 0]);

    // ---- Field source terms ----------------------------------------------------

    /// Power per unit area — the source term of the 2D heat equation.
    pub const AREAL_POWER_DENSITY: Dimension = Dimension::from_exponents([0, 1, -3, 0, 0, 0, 0]);
    /// Power per unit volume — the 3D form, kept for dimension-agnostic code (P9).
    pub const VOLUMETRIC_POWER_DENSITY: Dimension = Dimension::from_exponents([-1, 1, -3, 0, 0, 0, 0]);
}

/// Conventional names for dimensions, searched by [`Dimension::name`].
///
/// First match wins, so aliases (entropy = heat capacity) resolve to the first
/// listed name. This is a diagnostic aid only — it never affects type checking.
static NAMED_DIMENSIONS: &[(Dimension, &str)] = &[
    (Dimension::DIMENSIONLESS, "dimensionless"),
    (Dimension::LENGTH, "length"),
    (Dimension::MASS, "mass"),
    (Dimension::TIME, "time"),
    (Dimension::CURRENT, "electric current"),
    (Dimension::TEMPERATURE, "temperature"),
    (Dimension::AMOUNT, "amount of substance"),
    (Dimension::LUMINOUS, "luminous intensity"),
    (Dimension::AREA, "area"),
    (Dimension::VOLUME, "volume"),
    (Dimension::VELOCITY, "velocity"),
    (Dimension::ACCELERATION, "acceleration"),
    (Dimension::FREQUENCY, "frequency"),
    (Dimension::FORCE, "force"),
    (Dimension::ENERGY, "energy"),
    (Dimension::POWER, "power"),
    (Dimension::PRESSURE, "pressure"),
    (Dimension::MOMENTUM, "momentum"),
    (Dimension::ACTION, "action"),
    (Dimension::MASS_DENSITY, "mass density"),
    (Dimension::AREAL_MASS_DENSITY, "areal mass density"),
    (Dimension::DYNAMIC_VISCOSITY, "dynamic viscosity"),
    (Dimension::DIFFUSIVITY, "diffusivity"),
    (Dimension::STIFFNESS, "stiffness"),
    (Dimension::CHARGE, "electric charge"),
    (Dimension::VOLTAGE, "electric potential"),
    (Dimension::ELECTRIC_FIELD, "electric field"),
    (Dimension::CAPACITANCE, "capacitance"),
    (Dimension::RESISTANCE, "electric resistance"),
    (Dimension::MAGNETIC_FLUX_DENSITY, "magnetic flux density"),
    (Dimension::HEAT_CAPACITY, "heat capacity"),
    (Dimension::SPECIFIC_HEAT_CAPACITY, "specific heat capacity"),
    (Dimension::THERMAL_CONDUCTIVITY, "thermal conductivity"),
    (Dimension::MOLAR_MASS, "molar mass"),
    (Dimension::MOLAR_ENERGY, "molar energy"),
    (Dimension::MOLAR_HEAT_CAPACITY, "molar heat capacity"),
    (Dimension::CONCENTRATION, "concentration"),
    (Dimension::AREAL_CONCENTRATION, "areal concentration"),
    (Dimension::AREAL_REACTION_RATE, "areal reaction rate"),
    (Dimension::CATALYTIC_ACTIVITY, "catalytic activity"),
    (Dimension::AREAL_POWER_DENSITY, "areal power density"),
    (Dimension::VOLUMETRIC_POWER_DENSITY, "volumetric power density"),
];

/// Order used when *printing* a dimension, which is not the storage order.
///
/// SI convention writes the newton as `kg·m/s^2`, mass before length. Storage order
/// starts with length because that is the conventional order for the base dimensions
/// themselves. Keeping the two separate means neither has to be bent to suit the
/// other.
const DISPLAY_ORDER: [BaseDim; NUM_BASE] = [
    BaseDim::Mass,
    BaseDim::Length,
    BaseDim::Time,
    BaseDim::Current,
    BaseDim::Temperature,
    BaseDim::Amount,
    BaseDim::Luminous,
];

/// Renders the coherent SI form: `kg·m/s^2`, `mol/m^2`, `1` for dimensionless.
///
/// Positive exponents come first, then a single `/` and the negative exponents. There
/// is at most one `/`, so the output is unambiguous read left-to-right — the same
/// property the unit *parser* warns about when user input lacks it.
impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return f.write_str("1");
        }

        let mut wrote_numerator = false;
        for base in DISPLAY_ORDER {
            let exp = self.exponent(base);
            if exp > 0 {
                if wrote_numerator {
                    f.write_str("·")?;
                }
                write_factor(f, base.si_symbol(), exp)?;
                wrote_numerator = true;
            }
        }
        if !wrote_numerator {
            f.write_str("1")?;
        }

        let mut wrote_denominator = false;
        for base in DISPLAY_ORDER {
            let exp = self.exponent(base);
            if exp < 0 {
                f.write_str(if wrote_denominator { "·" } else { "/" })?;
                write_factor(f, base.si_symbol(), -exp)?;
                wrote_denominator = true;
            }
        }
        Ok(())
    }
}

fn write_factor(f: &mut fmt::Formatter<'_>, symbol: &str, exp: i8) -> fmt::Result {
    if exp == 1 {
        f.write_str(symbol)
    } else {
        write!(f, "{symbol}^{exp}")
    }
}

/// Shows the SI form rather than the raw array, because that is what a developer
/// staring at a failed assertion actually needs.
impl fmt::Debug for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dimension({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_is_mass_times_acceleration() {
        assert_eq!(Dimension::MASS.try_mul(Dimension::ACCELERATION).unwrap(), Dimension::FORCE);
    }

    #[test]
    fn energy_is_force_times_length() {
        assert_eq!(Dimension::FORCE.try_mul(Dimension::LENGTH).unwrap(), Dimension::ENERGY);
    }

    #[test]
    fn power_is_energy_over_time() {
        assert_eq!(Dimension::ENERGY.try_div(Dimension::TIME).unwrap(), Dimension::POWER);
    }

    #[test]
    fn pressure_is_force_over_area() {
        assert_eq!(Dimension::FORCE.try_div(Dimension::AREA).unwrap(), Dimension::PRESSURE);
    }

    /// The 2D heat equation's source term is power per unit area, not per volume.
    /// Getting this wrong is exactly the class of bug FR-002 exists to catch.
    #[test]
    fn areal_power_density_is_power_over_area() {
        assert_eq!(
            Dimension::POWER.try_div(Dimension::AREA).unwrap(),
            Dimension::AREAL_POWER_DENSITY
        );
    }

    #[test]
    fn diffusivity_is_area_over_time() {
        assert_eq!(Dimension::AREA.try_div(Dimension::TIME).unwrap(), Dimension::DIFFUSIVITY);
    }

    #[test]
    fn voltage_is_energy_per_charge() {
        assert_eq!(Dimension::ENERGY.try_div(Dimension::CHARGE).unwrap(), Dimension::VOLTAGE);
    }

    #[test]
    fn powi_and_root_round_trip() {
        let squared = Dimension::LENGTH.try_powi(2).unwrap();
        assert_eq!(squared, Dimension::AREA);
        assert_eq!(squared.root(2).unwrap(), Dimension::LENGTH);
    }

    #[test]
    fn root_rejects_fractional_exponents() {
        let err = Dimension::LENGTH.root(2).unwrap_err();
        assert!(matches!(
            err,
            DimensionError::NonIntegerRoot { base: BaseDim::Length, exponent: 1, degree: 2 }
        ));
    }

    #[test]
    fn root_rejects_zero_degree() {
        assert!(matches!(
            Dimension::AREA.root(0),
            Err(DimensionError::InvalidRootDegree { degree: 0 })
        ));
    }

    #[test]
    fn exponent_overflow_is_reported_not_wrapped() {
        let big = Dimension::from_exponents([120, 0, 0, 0, 0, 0, 0]);
        assert!(matches!(
            big.try_mul(big),
            Err(DimensionError::ExponentOverflow { base: BaseDim::Length })
        ));
    }

    #[test]
    fn powi_overflow_is_reported() {
        let big = Dimension::from_exponents([100, 0, 0, 0, 0, 0, 0]);
        assert!(matches!(
            big.try_powi(3),
            Err(DimensionError::ExponentOverflow { base: BaseDim::Length })
        ));
    }

    #[test]
    fn display_uses_coherent_si_form() {
        assert_eq!(Dimension::DIMENSIONLESS.to_string(), "1");
        assert_eq!(Dimension::FORCE.to_string(), "kg·m/s^2");
        assert_eq!(Dimension::ENERGY.to_string(), "kg·m^2/s^2");
        assert_eq!(Dimension::FREQUENCY.to_string(), "1/s");
        assert_eq!(Dimension::AREAL_CONCENTRATION.to_string(), "mol/m^2");
        assert_eq!(Dimension::PRESSURE.to_string(), "kg/m·s^2");
    }

    #[test]
    fn describe_names_known_dimensions() {
        assert_eq!(Dimension::FORCE.describe(), "force (kg·m/s^2)");
        // An unnamed dimension still describes itself in SI form.
        let odd = Dimension::from_exponents([5, 0, 0, 0, 0, 0, 0]);
        assert_eq!(odd.describe(), "m^5");
    }

    #[test]
    fn dimensionless_detection() {
        assert!(Dimension::DIMENSIONLESS.is_dimensionless());
        assert!(!Dimension::LENGTH.is_dimensionless());
        assert!(Dimension::VELOCITY.try_div(Dimension::VELOCITY).unwrap().is_dimensionless());
    }

    #[test]
    fn dimension_is_seven_bytes() {
        // Dimensions are compared and hashed on every expression node; keeping them
        // pointer-free and small is a load-bearing property, not an accident.
        assert_eq!(core::mem::size_of::<Dimension>(), 7);
    }
}
