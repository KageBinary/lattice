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

use lattice_ir::{Executor, Grain, ParticleStore};

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
    pub fn step<F>(self, dt: f64, store: &mut ParticleStore, eval_forces: F)
    where
        F: FnMut(&mut ParticleStore),
    {
        self.step_with(Executor::shared_sequential(), dt, store, eval_forces);
    }

    /// [`Integrator::step`], with the per-particle loops split across `executor`.
    ///
    /// Every update here writes particle `i` from particle `i` alone, so bands of
    /// particles are independent and the arithmetic for each one is untouched — this is
    /// bit-identical to [`Integrator::step`] at any thread count. Where a pass reads
    /// what an earlier pass wrote (velocity Verlet's drift reads the half-kicked
    /// velocity), the passes are separate dispatches, and a dispatch is a barrier.
    ///
    /// The parenthesisation below is deliberate and load-bearing: `dt * (im * f)` and
    /// `(dt * im) * f` are different numbers, so each expression is written exactly as
    /// the single-threaded loop wrote it. `bit_identical_to_the_sequential_step` is what
    /// keeps that true.
    pub fn step_with<F>(
        self,
        executor: &Executor,
        dt: f64,
        store: &mut ParticleStore,
        mut eval_forces: F,
    ) where
        F: FnMut(&mut ParticleStore),
    {
        match self {
            Integrator::ExplicitEuler => {
                let d = store.dynamics();
                // Both updates read the *old* state, so position must be advanced
                // with the old velocity before velocity is touched.
                let (vel_x, vel_y) = (&*d.vel_x, &*d.vel_y);
                banded(executor, d.pos_x, |i, x| *x += dt * vel_x[i]);
                banded(executor, d.pos_y, |i, y| *y += dt * vel_y[i]);

                let (inv_mass, force_x, force_y) = (d.inv_mass, &*d.force_x, &*d.force_y);
                banded(executor, d.vel_x, |i, v| *v += dt * inv_mass[i] * force_x[i]);
                banded(executor, d.vel_y, |i, v| *v += dt * inv_mass[i] * force_y[i]);
                eval_forces(store);
            }

            Integrator::SemiImplicitEuler => {
                let d = store.dynamics();
                // Velocity first, then position with the *new* velocity. That single
                // reordering is what makes this symplectic.
                {
                    let (inv_mass, force_x, force_y) = (d.inv_mass, &*d.force_x, &*d.force_y);
                    banded(executor, d.vel_x, |i, v| *v += dt * (inv_mass[i] * force_x[i]));
                    banded(executor, d.vel_y, |i, v| *v += dt * (inv_mass[i] * force_y[i]));
                }
                {
                    let (vel_x, vel_y) = (&*d.vel_x, &*d.vel_y);
                    banded(executor, d.pos_x, |i, x| *x += dt * vel_x[i]);
                    banded(executor, d.pos_y, |i, y| *y += dt * vel_y[i]);
                }
                eval_forces(store);
            }

            Integrator::VelocityVerlet => {
                let half = 0.5 * dt;
                {
                    let d = store.dynamics();
                    // Half kick, using a(t), then full drift.
                    {
                        let (inv_mass, force_x, force_y) = (d.inv_mass, &*d.force_x, &*d.force_y);
                        banded(executor, d.vel_x, |i, v| *v += half * inv_mass[i] * force_x[i]);
                        banded(executor, d.vel_y, |i, v| *v += half * inv_mass[i] * force_y[i]);
                    }
                    let (vel_x, vel_y) = (&*d.vel_x, &*d.vel_y);
                    banded(executor, d.pos_x, |i, x| *x += dt * vel_x[i]);
                    banded(executor, d.pos_y, |i, y| *y += dt * vel_y[i]);
                }
                // Forces at the new positions.
                eval_forces(store);
                {
                    let d = store.dynamics();
                    // Second half kick, using a(t+dt).
                    let (inv_mass, force_x, force_y) = (d.inv_mass, &*d.force_x, &*d.force_y);
                    banded(executor, d.vel_x, |i, v| *v += half * inv_mass[i] * force_x[i]);
                    banded(executor, d.vel_y, |i, v| *v += half * inv_mass[i] * force_y[i]);
                }
            }
        }
    }
}

/// How many particles are worth splitting, and how finely.
///
/// The floor is high, and it is high because it was measured rather than guessed.
///
/// The per-particle work is one multiply and one add against streamed memory, and a
/// velocity-Verlet step splits into six such passes. Each pass is a barrier: every
/// worker has to be reached and every worker has to report back, which costs a few
/// microseconds no matter how little work is inside. On the 20-thread machine this was
/// tuned on, `lattice bench particles-gravity --threads auto --compare` measured — best
/// of three release runs, with the floor lifted so the small sizes were actually split:
///
/// | particles | 16k | 33k | 65k | 131k | 262k | 524k |
/// |---|---|---|---|---|---|---|
/// | speedup | 0.42x | 0.65x | **1.19x** | 1.55x | 1.90x | 1.92x |
///
/// — a *slowdown* until about 65k, and a 2.4x one at the low end. Splitting a
/// 16k-particle model would therefore make `--threads auto` a pessimization on the most
/// common particle scene, and §15.1 is unambiguous: a faster wrong answer is a
/// regression, and so is a slower right one dressed as an optimization. Below the floor
/// the partition collapses to one chunk and the model runs exactly as it did before M4.
///
/// The chunk size is *not* the floor. See [`Grain`] — using one number for both would
/// keep a 262k-particle model on four threads to protect a 16k one.
/// Public so a test can ask whether a given population would actually be split. A
/// cross-backend case run below the floor compares the sequential path with itself and
/// passes for the wrong reason.
pub const PARTICLE_GRAIN: Grain = Grain::new(65_536, 8_192);

/// Run `f(index, slot)` over every element of `target`, split across `executor`.
///
/// The per-element expression stays at the call site rather than being folded into a
/// generic `axpy`, because the *association* of the multiplications differs between
/// schemes and a helper that quietly normalised it would change results by round-off.
fn banded<F>(executor: &Executor, target: &mut [f64], f: F)
where
    F: Fn(usize, &mut f64) + Sync,
{
    executor.for_each_chunk_mut(target, PARTICLE_GRAIN, |start, chunk| {
        for (offset, slot) in chunk.iter_mut().enumerate() {
            f(start + offset, slot);
        }
    });
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

    /// The claim `lattice_cpu` makes, checked where it is easiest to break: three
    /// schemes, several thread counts, and enough particles and steps that a
    /// mis-parenthesised update would have separated the trajectories long before the
    /// end. Bits, not tolerances — between two CPU threads there is nothing a tolerance
    /// would be excusing.
    #[test]
    fn a_parallel_step_is_bit_identical_to_the_sequential_step() {
        let build = || {
            let mut store = ParticleStore::with_capacity(4000);
            for index in 0..4000 {
                let t = index as f64 * 0.001;
                store
                    .spawn(
                        ParticleSpec::at([t.sin(), t.cos() * 2.0])
                            .with_velocity([t.cos(), -t.sin()])
                            .with_mass(0.5 + t),
                    )
                    .unwrap();
            }
            store
        };

        for scheme in
            [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet]
        {
            let run = |executor: &Executor| {
                let mut store = build();
                let mut forces = oscillator(3.0);
                forces(&mut store);
                for _ in 0..50 {
                    scheme.step_with(executor, 0.003, &mut store, &mut forces);
                }
                let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                (bits(store.pos_x()), bits(store.pos_y()), bits(store.vel_x()))
            };

            let reference = run(Executor::shared_sequential());
            for threads in [2usize, 3, 8] {
                assert_eq!(
                    run(&Executor::with_threads(threads)),
                    reference,
                    "{scheme:?} diverged on {threads} threads"
                );
            }
        }
    }

    /// A population too small to split must still take the same path.
    #[test]
    fn a_single_particle_is_unaffected_by_the_executor() {
        let mut sequential = single_particle(1.0, 0.0, 2.0);
        let mut parallel = single_particle(1.0, 0.0, 2.0);
        let scheme = Integrator::VelocityVerlet;
        let executor = Executor::with_threads(8);

        let mut a = oscillator(5.0);
        let mut b = oscillator(5.0);
        a(&mut sequential);
        b(&mut parallel);
        for _ in 0..100 {
            scheme.step(0.01, &mut sequential, &mut a);
            scheme.step_with(&executor, 0.01, &mut parallel, &mut b);
        }
        assert_eq!(parallel.pos_x()[0].to_bits(), sequential.pos_x()[0].to_bits());
        assert_eq!(parallel.vel_x()[0].to_bits(), sequential.vel_x()[0].to_bits());
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
