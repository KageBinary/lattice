//! Thermostats: the two spec §12.4 names, *"Langevin and simple velocity-rescaling"*.
//!
//! A thermostat replaces the microcanonical dynamics of velocity Verlet with
//! something that samples a temperature. Both here change what is conserved, and each
//! configuration therefore gets its own solver contract — an energy monitor reading a
//! thermostatted run as a broken integrator would be exactly the confusion P1 exists
//! to prevent.
//!
//! # Temperature
//!
//! Temperature is defined from the *thermal* kinetic energy — velocities relative to
//! the centre-of-mass velocity — over `N_df = 2N − 2` degrees of freedom, because a
//! system whose net momentum is fixed (by pair forces, or by being set to zero at the
//! start) has two fewer independent velocity components than it has coordinates:
//!
//! ```text
//!   K_thermal = Σ ½·mᵢ·|vᵢ − v_cm|²        T = 2·K_thermal / (N_df·k_B)
//! ```
//!
//! See [`crate::analysis::thermal`] for the exact rule, including pinned particles.
//!
//! # Langevin: the BAOAB splitting
//!
//! The Langevin equation adds friction and a matched random force to every particle:
//!
//! ```text
//!   m dv = F dt − γ m v dt + sqrt(2 γ m k_B T) dW
//! ```
//!
//! It is integrated by the BAOAB splitting of Leimkuhler and Matthews: half kick (B),
//! half drift (A), an *exact* Ornstein–Uhlenbeck update of the velocity (O), half
//! drift, forces, half kick. The O step is
//!
//! ```text
//!   v ← c₁·v + c₂·sqrt(k_B T / m)·ξ      c₁ = e^(−γ dt),  c₂ = sqrt(1 − c₁²)
//! ```
//!
//! with `ξ` a standard normal per component. Because the O step is exact for any
//! `dt`, a free particle samples the Maxwell distribution at `T` exactly — that is the
//! analytic check the validation suite makes. With interactions the configurational
//! sampling error is `O(dt²)`, and BAOAB's is the smallest of the standard splittings.
//! With `γ → 0` the scheme reduces to velocity Verlet with the drift taken in two
//! halves, which agrees with the one-piece drift to round-off.
//!
//! The random numbers come from the domain's own [`Pcg32`](lattice_ir::Pcg32), drawn
//! in particle order on one thread. A run is reproducible from its seed and does not
//! depend on the thread count, which keeps the promise in `docs/execution.md`.
//!
//! # Velocity rescaling: Berendsen weak coupling
//!
//! After every step the thermal velocities are scaled by
//!
//! ```text
//!   λ = sqrt(1 + (dt/τ)·(T₀/T − 1))
//! ```
//!
//! which drives `T` toward `T₀` with relaxation time `τ`: for an ideal gas, exactly
//! `T_{n+1} = T_n + (dt/τ)(T₀ − T_n)`, the discrete form of `dT/dt = (T₀ − T)/τ`. At
//! `dt ≥ τ` the scaling is isokinetic. Scaling is applied to `v − v_cm`, so the
//! centre-of-mass velocity — and with it the total momentum — is untouched. This
//! thermostat is deterministic, cheap, and does *not* sample the canonical ensemble:
//! it suppresses kinetic-energy fluctuations. It is the "simple profile" the spec
//! asks for, and its contract says what it is.

use lattice_ir::{Executor, ParticleStore};

use crate::analysis::{thermal, Thermal};
use crate::integrator::PARTICLE_GRAIN;

/// Boltzmann's constant, J/K.
pub const BOLTZMANN: f64 = lattice_units::constants::value::BOLTZMANN;

/// A temperature-control scheme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Thermostat {
    /// Langevin dynamics integrated by BAOAB.
    Langevin {
        /// Target temperature, kelvin.
        temperature: f64,
        /// Friction `γ`, per second. Its inverse is the velocity correlation time.
        friction: f64,
    },
    /// Berendsen weak-coupling velocity rescaling.
    VelocityRescale {
        /// Target temperature, kelvin.
        temperature: f64,
        /// Relaxation time `τ`, seconds.
        relaxation: f64,
    },
}

impl Thermostat {
    /// Short name for reports.
    pub const fn name(self) -> &'static str {
        match self {
            Thermostat::Langevin { .. } => "langevin",
            Thermostat::VelocityRescale { .. } => "velocity_rescale",
        }
    }

    /// The target temperature, kelvin.
    pub const fn temperature(self) -> f64 {
        match self {
            Thermostat::Langevin { temperature, .. } | Thermostat::VelocityRescale { temperature, .. } => {
                temperature
            }
        }
    }

    /// Check the parameters are usable.
    ///
    /// # Panics
    ///
    /// On a non-positive or non-finite temperature, friction or relaxation time. A
    /// thermostat at 0 K or with zero friction is a request for something else.
    pub fn validate(self) {
        let positive = |value: f64, what: &str| {
            assert!(value > 0.0 && value.is_finite(), "{} needs a positive finite {what}, got {value}", self.name());
        };
        match self {
            Thermostat::Langevin { temperature, friction } => {
                positive(temperature, "temperature");
                positive(friction, "friction");
            }
            Thermostat::VelocityRescale { temperature, relaxation } => {
                positive(temperature, "temperature");
                positive(relaxation, "relaxation time");
            }
        }
    }

    /// One line for the model report.
    pub fn describe(self) -> String {
        match self {
            Thermostat::Langevin { temperature, friction } => {
                format!("Langevin (BAOAB) at {temperature} K, friction {friction:.4e} /s")
            }
            Thermostat::VelocityRescale { temperature, relaxation } => {
                format!("velocity rescaling toward {temperature} K, relaxation {relaxation:.4e} s")
            }
        }
    }
}

/// The exact Ornstein–Uhlenbeck coefficients for one step of friction `γ`.
///
/// `c₁ = e^(−γ dt)` damps the velocity; `c₂ = sqrt(1 − c₁²)` scales the fresh noise so
/// the stationary variance is exactly `k_B T / m`.
pub fn ornstein_uhlenbeck_coefficients(friction: f64, dt: f64) -> (f64, f64) {
    let c1 = (-friction * dt).exp();
    // `1 − c₁²` in the form that keeps precision when γ dt is tiny.
    let c2 = (-(-2.0 * friction * dt).exp_m1()).sqrt();
    (c1, c2)
}

/// Apply Berendsen rescaling toward `target` with relaxation time `tau`.
///
/// Returns the scale factor applied to the thermal velocities, or 1 when the system
/// has no thermal motion to scale — a gas at rest cannot be heated by multiplying
/// zero, and the thermostat says so by leaving it alone rather than dividing by it.
pub fn rescale_velocities(
    executor: &Executor,
    store: &mut ParticleStore,
    dt: f64,
    target: f64,
    tau: f64,
) -> f64 {
    let Thermal { temperature, com_velocity, .. } = thermal(store);
    let Some(current) = temperature else { return 1.0 };
    if current <= 0.0 || !current.is_finite() {
        return 1.0;
    }
    // At dt ≥ τ the coupling is isokinetic: rescale exactly onto the target.
    let weight = (dt / tau).min(1.0);
    let lambda = (1.0 + weight * (target / current - 1.0)).sqrt();

    let d = store.dynamics();
    let [ux, uy] = com_velocity;
    let inv_mass = d.inv_mass;
    // A pinned particle has no thermal velocity and is left exactly where it is.
    executor.for_each_chunk_mut(d.vel_x, PARTICLE_GRAIN, |start, chunk| {
        for (offset, v) in chunk.iter_mut().enumerate() {
            if inv_mass[start + offset] > 0.0 {
                *v = ux + lambda * (*v - ux);
            }
        }
    });
    executor.for_each_chunk_mut(d.vel_y, PARTICLE_GRAIN, |start, chunk| {
        for (offset, v) in chunk.iter_mut().enumerate() {
            if inv_mass[start + offset] > 0.0 {
                *v = uy + lambda * (*v - uy);
            }
        }
    });
    lambda
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::temperature;
    use lattice_ir::{Pcg32, ParticleSpec};

    fn gas(n: usize, speed: f64, seed: u64) -> ParticleStore {
        let mut store = ParticleStore::with_capacity(n);
        let mut rng = Pcg32::seed_from_u64(seed);
        for i in 0..n {
            store
                .spawn(
                    ParticleSpec::at([i as f64, 0.0])
                        .with_velocity([rng.normal() * speed, rng.normal() * speed])
                        .with_mass(1.0 + (i % 3) as f64),
                )
                .unwrap();
        }
        store
    }

    #[test]
    fn ou_coefficients_preserve_the_stationary_variance() {
        for (gamma, dt) in [(1.0, 0.01), (100.0, 0.5), (1e-9, 1.0)] {
            let (c1, c2) = ornstein_uhlenbeck_coefficients(gamma, dt);
            assert!((c1 * c1 + c2 * c2 - 1.0).abs() < 1e-14, "γ={gamma} dt={dt}: {c1} {c2}");
            assert!(c1 > 0.0 && c1 <= 1.0 && c2 >= 0.0);
        }
        let (c1, c2) = ornstein_uhlenbeck_coefficients(0.0, 1.0);
        assert_eq!((c1, c2), (1.0, 0.0), "zero friction is velocity Verlet");
    }

    #[test]
    fn rescaling_moves_the_temperature_by_the_berendsen_rule() {
        let mut store = gas(200, 3.0, 1);
        let before = temperature(&store).unwrap();
        let target = 0.25 * before;
        let (dt, tau) = (0.1, 1.0);
        let lambda = rescale_velocities(Executor::shared_sequential(), &mut store, dt, target, tau);
        let after = temperature(&store).unwrap();
        let expected = before + (dt / tau) * (target - before);
        assert!((after - expected).abs() < 1e-9 * before, "after {after}, expected {expected}");
        assert!(lambda < 1.0);
    }

    #[test]
    fn rescaling_preserves_the_centre_of_mass_velocity() {
        let mut store = gas(50, 2.0, 2);
        {
            let d = store.dynamics();
            for v in d.vel_x.iter_mut() {
                *v += 5.0;
            }
        }
        let state = thermal(&store);
        let target = 0.25 * state.temperature.unwrap();
        rescale_velocities(Executor::shared_sequential(), &mut store, 1.0, target, 1.0);
        let after = thermal(&store);
        let drift = (state.com_velocity[0] - after.com_velocity[0]).abs().max((state.com_velocity[1] - after.com_velocity[1]).abs());
        assert!(drift < 1e-12 * state.com_velocity[0].abs(), "centre of mass moved by {drift}");
        assert!((after.temperature.unwrap() / target - 1.0).abs() < 1e-12, "dt = τ is isokinetic: {:?}", after.temperature);
    }

    #[test]
    fn rescaling_is_bit_identical_across_thread_counts() {
        let run = |executor: &Executor| {
            let mut store = gas(70_000, 1.0, 3);
            rescale_velocities(executor, &mut store, 0.01, 300.0, 0.1);
            store.vel_x().iter().chain(store.vel_y()).map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let reference = run(Executor::shared_sequential());
        assert_eq!(run(&Executor::with_threads(4)), reference);
    }

    #[test]
    fn a_gas_at_rest_is_left_at_rest() {
        let mut store = gas(10, 0.0, 4);
        let lambda = rescale_velocities(Executor::shared_sequential(), &mut store, 0.1, 300.0, 1.0);
        assert_eq!(lambda, 1.0);
        assert!(store.vel_x().iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "positive finite friction")]
    fn zero_friction_is_rejected() {
        Thermostat::Langevin { temperature: 1.0, friction: 0.0 }.validate();
    }

    #[test]
    fn descriptions_name_the_scheme() {
        let t = Thermostat::VelocityRescale { temperature: 2.0, relaxation: 0.5 };
        assert_eq!(t.name(), "velocity_rescale");
        assert_eq!(t.temperature(), 2.0);
        assert!(t.describe().contains("2 K"));
        t.validate();
    }
}
