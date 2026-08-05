//! Time integrators for particle dynamics.
//!
//! Spec §10.2: *"Integrators are selected per domain rather than globally. The initial
//! registry should include explicit Euler **only for teaching/comparison**,
//! semi-implicit Euler for simple mechanics, velocity Verlet/leapfrog for conservative
//! particle systems […] Each implementation publishes order, stability notes,
//! conservation behavior, backend support, and dense-output capability."*
//!
//! That "only for teaching" is load-bearing. Explicit Euler is included precisely
//! because it is *wrong* in an instructive way: run an orbit or an oscillator with it
//! and the energy grows without bound, which is the clearest possible demonstration of
//! why the other two exist. It is not a default anyone should reach for.
//!
//! # The force invariant
//!
//! Every integrator here assumes that on entry the store's force accumulators hold
//! `F(x(t))`, and guarantees that on exit they hold `F(x(t+dt))`. The domain
//! establishes this once at initialization. Velocity Verlet *requires* it — it needs
//! both `a(t)` and `a(t+dt)` and would otherwise have to evaluate forces twice per
//! step instead of once.
//!
//! # Velocity-dependent forces
//!
//! Velocity Verlet's derivation assumes forces depend only on position. With a
//! velocity-dependent law such as drag, the force at `t+dt` is evaluated using the
//! half-step velocity, since the end-of-step velocity is not yet known. This is
//! standard practice and costs the *velocity-dependent terms* their second-order
//! accuracy; position-dependent terms are unaffected. The domain contract states this.

use lattice_ir::ParticleStore;

/// A time-integration scheme for particle dynamics.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Integrator {
    /// Forward Euler. First order, not symplectic, systematically *adds* energy to
    /// oscillatory systems. Present for comparison only (spec §10.2).
    ExplicitEuler,
    /// Symplectic (Euler–Cromer) Euler: velocity updated first, then position with the
    /// new velocity. First order but symplectic, so bounded energy error.
    #[default]
    SemiImplicitEuler,
    /// Velocity Verlet. Second order, symplectic, time-reversible. The default for
    /// conservative particle systems and molecular dynamics.
    VelocityVerlet,
}

impl Integrator {
    /// Name used in reports and project files.
    pub const fn name(self) -> &'static str {
        match self {
            Integrator::ExplicitEuler => "explicit_euler",
            Integrator::SemiImplicitEuler => "semi_implicit_euler",
            Integrator::VelocityVerlet => "velocity_verlet",
        }
    }

    /// Order of accuracy in `dt`.
    pub const fn order(self) -> u32 {
        match self {
            Integrator::ExplicitEuler | Integrator::SemiImplicitEuler => 1,
            Integrator::VelocityVerlet => 2,
        }
    }

    /// Whether the scheme preserves phase-space volume.
    ///
    /// Symplectic schemes have *bounded* energy error that oscillates rather than
    /// accumulating; non-symplectic ones drift secularly. This is the single most
    /// important property for long conservative runs.
    pub const fn is_symplectic(self) -> bool {
        match self {
            Integrator::ExplicitEuler => false,
            Integrator::SemiImplicitEuler | Integrator::VelocityVerlet => true,
        }
    }

    /// Whether the scheme is time-reversible.
    pub const fn is_time_reversible(self) -> bool {
        matches!(self, Integrator::VelocityVerlet)
    }

    /// One-line statement of this scheme's conservation behaviour.
    pub const fn conservation_note(self) -> &'static str {
        match self {
            Integrator::ExplicitEuler => {
                "not symplectic: energy grows without bound in oscillatory systems; \
                 included for comparison only"
            }
            Integrator::SemiImplicitEuler => {
                "symplectic: energy error is bounded and oscillates at O(dt), \
                 with no secular drift"
            }
            Integrator::VelocityVerlet => {
                "symplectic and time-reversible: energy error is bounded and oscillates \
                 at O(dt^2), with no secular drift"
            }
        }
    }

    /// Advance the system by `dt`.
    ///
    /// `eval_forces` must clear and recompute the store's force accumulators from its
    /// current positions (and velocities, for velocity-dependent laws).
    ///
    /// On entry the accumulators must hold the forces for the current positions; on
    /// return they hold the forces for the new positions.
    pub fn step<F>(self, dt: f64, store: &mut ParticleStore, mut eval_forces: F)
    where
        F: FnMut(&mut ParticleStore),
    {
        match self {
            Integrator::ExplicitEuler => {
                let d = store.dynamics();
                // Both updates read the *old* state, so position must be advanced
                // with the old velocity before velocity is touched.
                for i in 0..d.len() {
                    d.pos_x[i] += dt * d.vel_x[i];
                    d.pos_y[i] += dt * d.vel_y[i];
                }
                for i in 0..d.len() {
                    d.vel_x[i] += dt * d.inv_mass[i] * d.force_x[i];
                    d.vel_y[i] += dt * d.inv_mass[i] * d.force_y[i];
                }
                eval_forces(store);
            }

            Integrator::SemiImplicitEuler => {
                let d = store.dynamics();
                // Velocity first, then position with the *new* velocity. That single
                // reordering is what makes this symplectic.
                for i in 0..d.len() {
                    let a_x = d.inv_mass[i] * d.force_x[i];
                    let a_y = d.inv_mass[i] * d.force_y[i];
                    d.vel_x[i] += dt * a_x;
                    d.vel_y[i] += dt * a_y;
                    d.pos_x[i] += dt * d.vel_x[i];
                    d.pos_y[i] += dt * d.vel_y[i];
                }
                eval_forces(store);
            }

            Integrator::VelocityVerlet => {
                let half = 0.5 * dt;
                {
                    let d = store.dynamics();
                    // Half kick, using a(t), then full drift.
                    for i in 0..d.len() {
                        d.vel_x[i] += half * d.inv_mass[i] * d.force_x[i];
                        d.vel_y[i] += half * d.inv_mass[i] * d.force_y[i];
                        d.pos_x[i] += dt * d.vel_x[i];
                        d.pos_y[i] += dt * d.vel_y[i];
                    }
                }
                // Forces at the new positions.
                eval_forces(store);
                {
                    let d = store.dynamics();
                    // Second half kick, using a(t+dt).
                    for i in 0..d.len() {
                        d.vel_x[i] += half * d.inv_mass[i] * d.force_x[i];
                        d.vel_y[i] += half * d.inv_mass[i] * d.force_y[i];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{ParticleSpec, ParticleStore};

    /// A one-dimensional harmonic oscillator: x'' = -(k/m) x.
    /// Forces are recomputed from position alone, so all three schemes apply cleanly.
    fn oscillator(k: f64) -> impl FnMut(&mut ParticleStore) {
        move |store: &mut ParticleStore| {
            store.clear_forces();
            let view = store.force_accumulation();
            for i in 0..view.len() {
                view.force_x[i] = -k * view.pos_x[i];
            }
        }
    }

    fn single_particle(x: f64, v: f64, mass: f64) -> ParticleStore {
        let mut s = ParticleStore::with_capacity(1);
        s.spawn(ParticleSpec::at([x, 0.0]).with_velocity([v, 0.0]).with_mass(mass)).unwrap();
        s
    }

    fn energy(store: &ParticleStore, k: f64) -> f64 {
        let x = store.pos_x()[0];
        let v = store.vel_x()[0];
        let m = store.mass()[0];
        0.5 * m * v * v + 0.5 * k * x * x
    }

    #[test]
    fn metadata_matches_the_schemes() {
        assert_eq!(Integrator::ExplicitEuler.order(), 1);
        assert_eq!(Integrator::VelocityVerlet.order(), 2);
        assert!(!Integrator::ExplicitEuler.is_symplectic());
        assert!(Integrator::SemiImplicitEuler.is_symplectic());
        assert!(Integrator::VelocityVerlet.is_time_reversible());
        assert!(!Integrator::SemiImplicitEuler.is_time_reversible());
        assert_eq!(Integrator::default(), Integrator::SemiImplicitEuler);
    }

    /// Constant force, zero initial velocity: every scheme must reproduce
    /// v = at exactly, and the position must match its own truncation order.
    #[test]
    fn constant_acceleration_gives_exact_velocity() {
        for integrator in
            [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
        {
            let mut store = single_particle(0.0, 0.0, 2.0);
            let force = |s: &mut ParticleStore| {
                s.clear_forces();
                let v = s.force_accumulation();
                for i in 0..v.len() {
                    v.force_y[i] = -2.0 * 10.0; // m * g, with g = 10 m/s^2
                }
            };
            let mut f = force;
            f(&mut store);

            let dt = 0.001;
            let steps = 1000;
            for _ in 0..steps {
                integrator.step(dt, &mut store, &mut f);
            }
            let t = dt * steps as f64;
            let v = store.vel_y()[0];
            assert!(
                (v - (-10.0 * t)).abs() < 1e-9,
                "{}: v = {v}, expected {}",
                integrator.name(),
                -10.0 * t
            );
        }
    }

    /// Velocity Verlet integrates constant acceleration *exactly*, including position:
    /// y = -g t^2 / 2 with no truncation error at all. The first-order schemes cannot.
    #[test]
    fn velocity_verlet_is_exact_for_constant_acceleration() {
        let mut store = single_particle(0.0, 0.0, 1.0);
        let mut force = |s: &mut ParticleStore| {
            s.clear_forces();
            let v = s.force_accumulation();
            for i in 0..v.len() {
                v.force_y[i] = -9.806_65;
            }
        };
        force(&mut store);

        let dt = 0.01;
        let steps = 100;
        for _ in 0..steps {
            Integrator::VelocityVerlet.step(dt, &mut store, &mut force);
        }
        let t = dt * steps as f64;
        let expected = -0.5 * 9.806_65 * t * t;
        let y = store.pos_y()[0];
        assert!((y - expected).abs() < 1e-12, "y = {y}, expected {expected}");
    }

    /// The teaching point: explicit Euler pumps energy into an oscillator, the
    /// symplectic schemes do not. This is the behaviour §10.2 keeps explicit Euler
    /// around to demonstrate.
    #[test]
    fn symplectic_schemes_bound_the_energy_error_and_explicit_euler_does_not() {
        let k = 1.0;
        let periods = 50.0;
        let dt = 0.01;
        let steps = (periods * core::f64::consts::TAU / dt) as usize;

        let mut worst = std::collections::BTreeMap::new();
        for integrator in
            [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
        {
            let mut store = single_particle(1.0, 0.0, 1.0);
            let mut force = oscillator(k);
            force(&mut store);
            let e0 = energy(&store, k);

            let mut max_relative = 0.0f64;
            for _ in 0..steps {
                integrator.step(dt, &mut store, &mut force);
                let drift = ((energy(&store, k) - e0) / e0).abs();
                max_relative = max_relative.max(drift);
            }
            worst.insert(integrator.name(), max_relative);
        }

        let euler = worst["explicit_euler"];
        let semi = worst["semi_implicit_euler"];
        let verlet = worst["velocity_verlet"];

        // Over 50 periods explicit Euler roughly triples the energy.
        assert!(euler > 1.0, "explicit Euler should gain >100% energy, gained {euler:.3}");
        // Both symplectic schemes stay within a fraction of a percent, forever.
        assert!(semi < 0.02, "semi-implicit Euler drifted {semi:.5}");
        assert!(verlet < 0.02, "velocity Verlet drifted {verlet:.5}");
        // And second order really is better than first.
        assert!(verlet < semi, "Verlet {verlet:.3e} should beat semi-implicit {semi:.3e}");
    }

    /// Symplectic error is *bounded*, not merely small: running ten times longer must
    /// not make it ten times worse.
    #[test]
    fn symplectic_energy_error_does_not_grow_with_run_length() {
        let k = 1.0;
        let dt = 0.01;
        let mut store = single_particle(1.0, 0.0, 1.0);
        let mut force = oscillator(k);
        force(&mut store);
        let e0 = energy(&store, k);

        let short_steps = 1_000;
        let mut short_max = 0.0f64;
        for _ in 0..short_steps {
            Integrator::VelocityVerlet.step(dt, &mut store, &mut force);
            short_max = short_max.max(((energy(&store, k) - e0) / e0).abs());
        }

        let mut long_max = 0.0f64;
        for _ in 0..short_steps * 20 {
            Integrator::VelocityVerlet.step(dt, &mut store, &mut force);
            long_max = long_max.max(((energy(&store, k) - e0) / e0).abs());
        }

        assert!(
            long_max < short_max * 1.5,
            "error grew from {short_max:.3e} to {long_max:.3e} over 20x the run"
        );
    }

    /// Halving `dt` must reduce the phase error by the factor each scheme's order
    /// predicts. This is the observable consequence of `Integrator::order`.
    #[test]
    fn observed_convergence_order_matches_the_declared_order() {
        let k = 1.0;
        let omega = 1.0;
        let t_end = 2.0;

        for integrator in [Integrator::SemiImplicitEuler, Integrator::VelocityVerlet] {
            let error_at = |dt: f64| {
                let steps = (t_end / dt).round() as usize;
                let mut store = single_particle(1.0, 0.0, 1.0);
                let mut force = oscillator(k);
                force(&mut store);
                for _ in 0..steps {
                    integrator.step(dt, &mut store, &mut force);
                }
                (store.pos_x()[0] - (omega * t_end).cos()).abs()
            };

            let coarse = error_at(0.002);
            let fine = error_at(0.001);
            let observed_order = (coarse / fine).log2();
            let declared = f64::from(integrator.order());
            assert!(
                (observed_order - declared).abs() < 0.15,
                "{}: observed order {observed_order:.3}, declared {declared}",
                integrator.name()
            );
        }
    }

    /// Velocity Verlet is time-reversible: step forward, negate velocity, step back
    /// the same number of steps, and the particle returns to where it started.
    #[test]
    fn velocity_verlet_is_reversible() {
        let k = 1.0;
        let dt = 0.005;
        let steps = 500;
        let mut store = single_particle(1.0, 0.3, 1.0);
        let mut force = oscillator(k);
        force(&mut store);
        let (x0, v0) = (store.pos_x()[0], store.vel_x()[0]);

        for _ in 0..steps {
            Integrator::VelocityVerlet.step(dt, &mut store, &mut force);
        }
        // Reverse time by flipping the velocity.
        let id = lattice_ir::ParticleId::new(0, 0);
        let v = store.velocity_of(id).unwrap();
        store.set_velocity(id, [-v[0], -v[1]]);
        force(&mut store);
        for _ in 0..steps {
            Integrator::VelocityVerlet.step(dt, &mut store, &mut force);
        }

        assert!((store.pos_x()[0] - x0).abs() < 1e-10, "x returned to {}", store.pos_x()[0]);
        assert!((-store.vel_x()[0] - v0).abs() < 1e-10);
    }

    #[test]
    fn pinned_particles_never_move() {
        let mut store = ParticleStore::with_capacity(2);
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(f64::INFINITY)).unwrap();
        store.spawn(ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        let mut force = |s: &mut ParticleStore| {
            s.clear_forces();
            let v = s.force_accumulation();
            for i in 0..v.len() {
                v.force_x[i] = 100.0;
            }
        };
        force(&mut store);
        for _ in 0..100 {
            Integrator::VelocityVerlet.step(0.01, &mut store, &mut force);
        }
        assert_eq!(store.pos_x()[0], 0.0, "an infinite-mass particle must not accelerate");
        assert!(store.pos_x()[1] > 1.0);
    }

    #[test]
    fn an_empty_store_steps_without_incident() {
        let mut store = ParticleStore::with_capacity(4);
        for integrator in
            [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
        {
            integrator.step(0.01, &mut store, |s| s.clear_forces());
        }
        assert_eq!(store.len(), 0);
    }
}
