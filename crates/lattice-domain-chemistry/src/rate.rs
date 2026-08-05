//! How fast a reaction goes.
//!
//! Spec §12.2 asks for *"rate law, equilibrium or kinetic parameters, temperature
//! dependence"*. This module has mass action, reversibility, and Arrhenius — the three
//! that the flagship demo needs and that a validation case can check against a closed
//! form.
//!
//! # Why the rate constant is not just a number
//!
//! A rate constant's units depend on the reaction's order: an order-`n` constant is in
//! `(m²/mol)^(n−1)/s`. A first-order constant is `1/s`, a second-order one is
//! `m²/(mol·s)`, and writing the second where the first belongs is wrong by a factor
//! with the dimensions of a concentration. The number looks perfectly plausible either
//! way. [`RateLaw::si_unit_for_order`] gives the compiler what it needs to reject that.
//!
//! # Why temperature dependence lives here rather than in the coupling
//!
//! §12.5 lists *"temperature changes rate constants through declared models such as
//! Arrhenius relationships"* as a coupling concern, and it is — the temperature comes
//! from another domain. But the *relationship* is a property of the reaction, so it is
//! declared with the reaction and evaluated where the rate is. The coupling edge
//! carries the temperature; it does not carry the chemistry.

/// The molar gas constant, J/(mol·K).
///
/// Exact since the 2019 SI redefinition: `R = N_A · k_B`, both of which are now defined
/// constants.
pub const GAS_CONSTANT: f64 = 8.314_462_618_153_24;

/// How a rate constant depends on temperature.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Temperature {
    /// It does not. The constant is whatever was declared, at any temperature.
    #[default]
    Independent,
    /// Arrhenius: `k(T) = A · exp(−Ea / R T)`.
    ///
    /// The declared constant is taken as `A`, the pre-exponential factor, so a reaction
    /// with `activation_energy: 0` behaves exactly as if no temperature model were
    /// declared. That continuity is deliberate: adding an activation energy of zero to
    /// a working model must not change its answer.
    Arrhenius {
        /// Activation energy, J/mol.
        activation_energy: f64,
    },
}

impl Temperature {
    /// The factor the declared constant is multiplied by at `temperature` kelvin.
    ///
    /// Returns 1 for [`Temperature::Independent`], and for a non-physical temperature:
    /// `exp(−Ea/RT)` at `T ≤ 0` is either an overflow or a division by zero, and a
    /// rate constant of infinity turns a model that had one bad cell into a model that
    /// is entirely NaN one step later. The caller's own non-finite monitor is a better
    /// place to notice it (NFR-007).
    pub fn factor(self, temperature: f64) -> f64 {
        match self {
            Temperature::Independent => 1.0,
            Temperature::Arrhenius { activation_energy } => {
                if !(temperature.is_finite() && temperature > 0.0) {
                    return 1.0;
                }
                (-activation_energy / (GAS_CONSTANT * temperature)).exp()
            }
        }
    }

    /// The temperature at which the factor is 1 — where `A` *is* `k`.
    ///
    /// There is none: `exp(−Ea/RT)` reaches 1 only as `T → ∞`. Stated as a method
    /// returning `None` because it is a question readers ask, and the answer explains
    /// why an Arrhenius rate always runs slower than its pre-exponential factor.
    pub fn reference_temperature(self) -> Option<f64> {
        match self {
            Temperature::Independent => Some(f64::INFINITY),
            Temperature::Arrhenius { activation_energy: 0.0 } => {
                Some(f64::INFINITY)
            }
            Temperature::Arrhenius { .. } => None,
        }
    }

    /// A one-line description for the model report.
    pub fn describe(self) -> String {
        match self {
            Temperature::Independent => "temperature-independent".to_string(),
            Temperature::Arrhenius { activation_energy } => {
                format!("Arrhenius, Ea = {:.4e} J/mol", activation_energy)
            }
        }
    }
}

/// How a reaction's rate is computed from concentrations.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RateLaw {
    /// Forward rate constant, in `(m²/mol)^(order−1)/s`.
    pub forward: f64,
    /// Reverse rate constant, or zero for an irreversible reaction.
    pub reverse: f64,
    /// How both constants depend on temperature.
    ///
    /// One model for both directions. A reaction whose forward and reverse steps have
    /// different activation energies is really two reactions, and writing it as two is
    /// clearer than a rate law with four parameters — the equilibrium constant then
    /// falls out of the ratio rather than being a fifth thing to keep consistent.
    pub temperature: Temperature,
}

impl RateLaw {
    /// Irreversible mass action at the given forward constant.
    pub fn mass_action(forward: f64) -> RateLaw {
        RateLaw { forward, reverse: 0.0, temperature: Temperature::Independent }
    }

    /// Reversible mass action.
    pub fn reversible(forward: f64, reverse: f64) -> RateLaw {
        RateLaw { forward, reverse, temperature: Temperature::Independent }
    }

    /// Add an Arrhenius temperature dependence with the given activation energy, J/mol.
    pub fn with_activation_energy(mut self, activation_energy: f64) -> RateLaw {
        self.temperature = Temperature::Arrhenius { activation_energy };
        self
    }

    /// True when the reverse step is present.
    pub fn is_reversible(&self) -> bool {
        self.reverse != 0.0
    }

    /// The equilibrium constant, `k_f / k_r`.
    ///
    /// `None` for an irreversible reaction, which has no equilibrium — it runs to
    /// completion. Temperature-independent even under Arrhenius, because both
    /// directions share one model and the factor cancels; a reaction whose equilibrium
    /// shifts with temperature is two reactions with different activation energies.
    pub fn equilibrium_constant(&self) -> Option<f64> {
        (self.reverse != 0.0).then(|| self.forward / self.reverse)
    }

    /// The forward constant at `temperature` kelvin.
    pub fn forward_at(&self, temperature: f64) -> f64 {
        self.forward * self.temperature.factor(temperature)
    }

    /// The reverse constant at `temperature` kelvin.
    pub fn reverse_at(&self, temperature: f64) -> f64 {
        self.reverse * self.temperature.factor(temperature)
    }

    /// The SI unit of a rate constant for a reaction of total order `order`.
    ///
    /// What the compiler checks a declared constant against. Concentrations are per
    /// unit *area* here, so the familiar `(m³/mol)^(n−1)/s` becomes `(m²/mol)^(n−1)/s`.
    pub fn si_unit_for_order(order: f64) -> String {
        let exponent = order - 1.0;
        if exponent.abs() < 1e-12 {
            return "1/s".to_string();
        }
        if (exponent - 1.0).abs() < 1e-12 {
            return "m^2/(mol·s)".to_string();
        }
        format!("(m^2/mol)^{exponent}/s")
    }

    /// A one-line description for the model report.
    pub fn describe(&self, order: f64) -> String {
        let unit = RateLaw::si_unit_for_order(order);
        let mut text = format!("k = {:.4e} {unit}", self.forward);
        if let Some(equilibrium) = self.equilibrium_constant() {
            text.push_str(&format!(", k_rev = {:.4e}, K = {equilibrium:.4e}", self.reverse));
        }
        if self.temperature != Temperature::Independent {
            text.push_str(&format!(", {}", self.temperature.describe()));
        }
        text
    }
}

/// The rate of a mass-action reaction, mol/(m²·s).
///
/// `Π[R]^ν − (k_r/k_f)·Π[P]^ν`, scaled by the forward constant at this temperature.
/// Negative when the reverse step dominates, which is what lets one number drive both
/// directions of an equilibrium.
///
/// Concentrations below zero are treated as zero. They should not occur — a mass-action
/// rate slows to nothing as a reactant is exhausted — but a timestep past the stability
/// limit produces them, and a negative concentration raised to a fractional power is a
/// NaN that spreads through the whole field before anything notices.
pub fn mass_action_rate(
    law: &RateLaw,
    reactants: &[(usize, f64)],
    products: &[(usize, f64)],
    concentrations: &[f64],
    temperature: f64,
) -> f64 {
    let product_of = |terms: &[(usize, f64)]| -> f64 {
        terms
            .iter()
            .map(|&(index, coefficient)| {
                let amount = concentrations.get(index).copied().unwrap_or(0.0).max(0.0);
                if (coefficient - 1.0).abs() < 1e-12 {
                    amount
                } else {
                    amount.powf(coefficient)
                }
            })
            .product()
    };

    let forward = law.forward_at(temperature) * product_of(reactants);
    if !law.is_reversible() {
        return forward;
    }
    forward - law.reverse_at(temperature) * product_of(products)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn mass_action_multiplies_the_reactants() {
        let law = RateLaw::mass_action(2.0);
        // A + B with [A] = 3, [B] = 5: rate = 2 x 3 x 5.
        let rate = mass_action_rate(&law, &[(0, 1.0), (1, 1.0)], &[], &[3.0, 5.0], 300.0);
        assert!((rate - 30.0).abs() < EPSILON);

        // 2A: rate = k [A]^2, not k [A] x 2.
        let squared = mass_action_rate(&law, &[(0, 2.0)], &[], &[3.0], 300.0);
        assert!((squared - 18.0).abs() < EPSILON, "{squared}");
    }

    #[test]
    fn a_reversible_reaction_stops_at_its_equilibrium() {
        let law = RateLaw::reversible(2.0, 4.0);
        assert_eq!(law.equilibrium_constant(), Some(0.5));

        // A <-> B. At equilibrium k_f[A] = k_r[B], so [B]/[A] = K = 0.5.
        let rate = mass_action_rate(&law, &[(0, 1.0)], &[(1, 1.0)], &[2.0, 1.0], 300.0);
        assert!(rate.abs() < EPSILON, "net rate at equilibrium should vanish: {rate}");

        // Past it, the net rate reverses.
        let overshoot = mass_action_rate(&law, &[(0, 1.0)], &[(1, 1.0)], &[1.0, 2.0], 300.0);
        assert!(overshoot < 0.0, "{overshoot}");
    }

    #[test]
    fn an_irreversible_reaction_has_no_equilibrium() {
        let law = RateLaw::mass_action(3.0);
        assert_eq!(law.equilibrium_constant(), None);
        assert!(!law.is_reversible());
        // The products do not slow it down, however much of them there is.
        let rate = mass_action_rate(&law, &[(0, 1.0)], &[(1, 1.0)], &[1.0, 1e6], 300.0);
        assert!((rate - 3.0).abs() < EPSILON);
    }

    /// The relationship the flagship demo turns on: hotter is faster, by the factor
    /// Arrhenius says and not some other one.
    #[test]
    fn arrhenius_speeds_a_reaction_up_with_temperature() {
        // Ea = 50 kJ/mol, a typical activation energy.
        let law = RateLaw::mass_action(1e10).with_activation_energy(50_000.0);

        let cold = law.forward_at(300.0);
        let hot = law.forward_at(310.0);
        assert!(hot > cold, "{cold} -> {hot}");

        // The textbook rule of thumb: near room temperature a 10 K rise roughly doubles
        // a reaction with Ea around 50 kJ/mol. Checked against the closed form, not the
        // rule — the rule is why this value of Ea was chosen for the test.
        let expected = (-50_000.0f64 / (GAS_CONSTANT * 310.0)).exp()
            / (-50_000.0f64 / (GAS_CONSTANT * 300.0)).exp();
        assert!((hot / cold - expected).abs() < 1e-9);
        assert!((hot / cold - 1.9).abs() < 0.2, "roughly a doubling: {}", hot / cold);
    }

    /// Adding an activation energy of zero must not change a working model's answer.
    #[test]
    fn a_zero_activation_energy_is_the_same_as_no_temperature_model() {
        let plain = RateLaw::mass_action(7.0);
        let annotated = RateLaw::mass_action(7.0).with_activation_energy(0.0);
        for temperature in [1.0, 300.0, 5000.0] {
            assert!((plain.forward_at(temperature) - annotated.forward_at(temperature)).abs() < EPSILON);
        }
        assert_eq!(annotated.temperature.reference_temperature(), Some(f64::INFINITY));
    }

    /// An Arrhenius rate never reaches its pre-exponential factor, which is the usual
    /// surprise when someone reads a rate constant off a table.
    #[test]
    fn an_arrhenius_rate_is_always_slower_than_its_prefactor() {
        let law = RateLaw::mass_action(1e10).with_activation_energy(50_000.0);
        for temperature in [200.0, 300.0, 1000.0, 1e6] {
            assert!(law.forward_at(temperature) < law.forward, "at {temperature} K");
        }
        assert_eq!(law.temperature.reference_temperature(), None);
    }

    /// A rate constant of infinity turns one bad cell into a whole NaN field one step
    /// later. The non-finite monitor is a better place to notice a bad temperature.
    #[test]
    fn a_non_physical_temperature_does_not_produce_an_infinite_rate() {
        let law = RateLaw::mass_action(1.0).with_activation_energy(50_000.0);
        for temperature in [0.0, -300.0, f64::NAN, f64::INFINITY] {
            let k = law.forward_at(temperature);
            assert!(k.is_finite(), "{temperature} K gave k = {k}");
        }
    }

    /// A negative concentration raised to a fractional power is a NaN that spreads
    /// through a whole field before anything notices.
    #[test]
    fn negative_concentrations_are_clamped_rather_than_raised_to_a_power() {
        let law = RateLaw::mass_action(1.0);
        let rate = mass_action_rate(&law, &[(0, 0.5)], &[], &[-1e-18], 300.0);
        assert!(rate.is_finite() && rate == 0.0, "{rate}");
    }

    /// An order-2 constant written in 1/s is wrong by a factor with the dimensions of a
    /// concentration, and the number looks plausible either way.
    #[test]
    fn rate_constant_units_follow_the_order() {
        assert_eq!(RateLaw::si_unit_for_order(1.0), "1/s");
        assert_eq!(RateLaw::si_unit_for_order(2.0), "m^2/(mol·s)");
        assert!(RateLaw::si_unit_for_order(3.0).contains("^2/s"));
        // Zero order: the rate does not depend on concentration at all.
        assert!(RateLaw::si_unit_for_order(0.0).contains("^-1/s"));
    }

    #[test]
    fn descriptions_name_what_the_reader_needs() {
        let law = RateLaw::reversible(2.0, 4.0).with_activation_energy(50_000.0);
        let text = law.describe(2.0);
        assert!(text.contains("m^2/(mol·s)"), "{text}");
        assert!(text.contains("K = 5.0000e-1"), "{text}");
        assert!(text.contains("Arrhenius"), "{text}");
    }

    #[test]
    fn an_index_past_the_concentration_vector_reads_as_zero() {
        let law = RateLaw::mass_action(5.0);
        assert_eq!(mass_action_rate(&law, &[(9, 1.0)], &[], &[1.0], 300.0), 0.0);
    }
}
