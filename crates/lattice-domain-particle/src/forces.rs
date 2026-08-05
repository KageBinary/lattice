//! Force laws.
//!
//! A [`ForceLaw`] adds its contribution to the particle force accumulators. Laws are
//! dynamically dispatched, but dispatch happens *once per law per step*, not once per
//! particle — the inner loops inside each law are fully monomorphic. That is the
//! trade spec §16.3 implies: extension points at the operator level (Tier "Operator"),
//! not inside the hot loop.
//!
//! # Every law states whether it conserves energy
//!
//! [`ForceLaw::potential_energy`] exists so the runtime can compute total energy and
//! monitor drift (§10.4). A dissipative law returns zero and reports
//! [`ForceLaw::is_conservative`] as false, so the energy monitor knows not to treat
//! the resulting decline as numerical error. Without that distinction, adding drag to
//! a scene would look identical to an unstable integrator.
//!
//! # Every law states what it does to the timestep
//!
//! [`ForceLaw::stability_limit`] lets a stiff law report the step it needs. The
//! domain takes the tightest limit across its laws, so the reason code the viewer
//! shows names the force actually responsible (§7.3, §17.3).

use core::fmt;

use lattice_ir::{ForceAccumulation, ParticleStore, StabilityReason, StableStep};

use crate::neighbors::CellList;

/// A contribution to the forces on a particle system.
pub trait ForceLaw: fmt::Debug + Send + Sync {
    /// Short name, used in diagnostics and the model report.
    fn name(&self) -> &'static str;

    /// True if this law derives from a potential and therefore conserves energy.
    fn is_conservative(&self) -> bool;

    /// The interaction cutoff, if this law needs a neighbour list.
    fn cutoff(&self) -> Option<f64> {
        None
    }

    /// Add this law's forces to the accumulators.
    ///
    /// `neighbors` is `Some` exactly when [`ForceLaw::cutoff`] returned `Some` and
    /// the domain has built a list.
    fn accumulate(&self, view: &mut ForceAccumulation<'_>, neighbors: Option<&CellList>);

    /// Potential energy of the current configuration, joules.
    ///
    /// Zero for non-conservative laws.
    fn potential_energy(&self, store: &ParticleStore, neighbors: Option<&CellList>) -> f64;

    /// The timestep this law needs, if it imposes one.
    fn stability_limit(&self, _store: &ParticleStore) -> Option<StableStep> {
        None
    }
}

/// Uniform acceleration, e.g. gravity.
///
/// `F_i = m_i · g`. Because the force is proportional to mass, every particle
/// accelerates identically regardless of mass — the check the free-fall validation
/// case makes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformAcceleration {
    /// Acceleration vector, m/s².
    pub acceleration: [f64; 2],
    /// Height datum for the potential energy, m. Only affects the constant offset.
    pub reference: [f64; 2],
}

impl UniformAcceleration {
    /// Earth-surface gravity pointing in −y.
    pub fn earth_gravity() -> Self {
        Self {
            acceleration: [0.0, -lattice_units::constants::value::STANDARD_GRAVITY],
            reference: [0.0, 0.0],
        }
    }

    /// A custom uniform acceleration.
    pub fn new(acceleration: [f64; 2]) -> Self {
        Self { acceleration, reference: [0.0, 0.0] }
    }
}

impl ForceLaw for UniformAcceleration {
    fn name(&self) -> &'static str {
        "uniform_acceleration"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, _neighbors: Option<&CellList>) {
        let [gx, gy] = self.acceleration;
        for i in 0..view.len() {
            let m = view.mass[i];
            view.force_x[i] += m * gx;
            view.force_y[i] += m * gy;
        }
    }

    fn potential_energy(&self, store: &ParticleStore, _neighbors: Option<&CellList>) -> f64 {
        // U = -m g·(x - x_ref); for g = (0, -9.81) this is the familiar +mgh.
        let [gx, gy] = self.acceleration;
        let [rx, ry] = self.reference;
        let (xs, ys, ms) = (store.pos_x(), store.pos_y(), store.mass());
        let mut total = 0.0;
        for i in 0..xs.len() {
            if ms[i].is_finite() {
                total -= ms[i] * (gx * (xs[i] - rx) + gy * (ys[i] - ry));
            }
        }
        total
    }
}

/// Linear (Stokes) drag, `F = −c·v`.
///
/// Dissipative: it removes energy on purpose. The energy monitor must not read that
/// decline as integration error, which is why [`ForceLaw::is_conservative`] is false.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearDrag {
    /// Drag coefficient `c`, in kg/s.
    pub coefficient: f64,
}

impl LinearDrag {
    /// Drag with the given coefficient.
    pub fn new(coefficient: f64) -> Self {
        Self { coefficient }
    }
}

impl ForceLaw for LinearDrag {
    fn name(&self) -> &'static str {
        "linear_drag"
    }

    fn is_conservative(&self) -> bool {
        false
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, _neighbors: Option<&CellList>) {
        let c = self.coefficient;
        for i in 0..view.len() {
            view.force_x[i] -= c * view.vel_x[i];
            view.force_y[i] -= c * view.vel_y[i];
        }
    }

    fn potential_energy(&self, _store: &ParticleStore, _neighbors: Option<&CellList>) -> f64 {
        0.0
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        // Explicit integration of `dv/dt = -(c/m) v` is stable while dt < 2m/c.
        // The lightest particle binds.
        let min_mass = store
            .mass()
            .iter()
            .copied()
            .filter(|m| m.is_finite() && *m > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !min_mass.is_finite() || self.coefficient <= 0.0 {
            return None;
        }
        let max = 2.0 * min_mass / self.coefficient;
        Some(StableStep::limited(0.1 * max, max, StabilityReason::OscillationPeriod))
    }
}

/// An isotropic harmonic well, `F = −k·(x − centre)`.
///
/// The canonical validation case: the analytic solution is known exactly, and energy
/// drift under different integrators is the clearest demonstration of why symplectic
/// integrators matter (§19.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarmonicWell {
    /// Well centre, m.
    pub center: [f64; 2],
    /// Spring constant, N/m.
    pub stiffness: f64,
}

impl HarmonicWell {
    /// A well of the given stiffness centred at `center`.
    pub fn new(center: [f64; 2], stiffness: f64) -> Self {
        Self { center, stiffness }
    }

    /// Angular frequency for a particle of mass `m`, rad/s.
    pub fn angular_frequency(&self, mass: f64) -> f64 {
        (self.stiffness / mass).sqrt()
    }
}

impl ForceLaw for HarmonicWell {
    fn name(&self) -> &'static str {
        "harmonic_well"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, _neighbors: Option<&CellList>) {
        let [cx, cy] = self.center;
        let k = self.stiffness;
        for i in 0..view.len() {
            view.force_x[i] -= k * (view.pos_x[i] - cx);
            view.force_y[i] -= k * (view.pos_y[i] - cy);
        }
    }

    fn potential_energy(&self, store: &ParticleStore, _neighbors: Option<&CellList>) -> f64 {
        let [cx, cy] = self.center;
        let (xs, ys) = (store.pos_x(), store.pos_y());
        let mut total = 0.0;
        for i in 0..xs.len() {
            let (dx, dy) = (xs[i] - cx, ys[i] - cy);
            total += 0.5 * self.stiffness * (dx * dx + dy * dy);
        }
        total
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        let min_mass = store
            .mass()
            .iter()
            .copied()
            .filter(|m| m.is_finite() && *m > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !min_mass.is_finite() || self.stiffness <= 0.0 {
            return None;
        }
        let omega = self.angular_frequency(min_mass);
        // Symplectic Euler and velocity Verlet are both stable for dt < 2/omega.
        // Accuracy needs far less: ~50 steps per oscillation is the usual guidance.
        let max = 2.0 / omega;
        let period = core::f64::consts::TAU / omega;
        Some(StableStep::limited(period / 50.0, max, StabilityReason::OscillationPeriod))
    }
}

/// Truncated-and-shifted Lennard-Jones pair potential.
///
/// ```text
///   U(r) = 4ε[(σ/r)^12 − (σ/r)^6] − U(r_c)     for r < r_c
///        = 0                                    otherwise
/// ```
///
/// The `−U(r_c)` shift makes the *energy* continuous at the cutoff, which is what
/// energy-conservation tests measure. The *force* is still discontinuous there — a
/// particle crossing the cutoff radius feels a small step in force — so this law
/// declares that residual non-conservation rather than pretending it away. Force
/// shifting or a smooth switching function would remove it at the cost of changing
/// the potential everywhere inside the cutoff; that belongs to the MD module (§12.4),
/// where the choice can be a named option.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LennardJones {
    /// Well depth ε, joules.
    pub epsilon: f64,
    /// Length scale σ, metres. The potential crosses zero at `r = σ`.
    pub sigma: f64,
    /// Interaction cutoff, metres. Conventionally `2.5σ`.
    pub cutoff: f64,
    /// Precomputed `U_unshifted(cutoff)`, subtracted to make the energy continuous.
    energy_shift: f64,
}

impl LennardJones {
    /// A Lennard-Jones interaction with an explicit cutoff.
    ///
    /// # Panics
    ///
    /// If the cutoff is not greater than zero, or ε or σ are non-positive.
    pub fn new(epsilon: f64, sigma: f64, cutoff: f64) -> Self {
        assert!(epsilon > 0.0 && sigma > 0.0, "LJ requires positive epsilon and sigma");
        assert!(cutoff > 0.0, "LJ cutoff must be positive");
        let s6 = (sigma / cutoff).powi(6);
        Self { epsilon, sigma, cutoff, energy_shift: 4.0 * epsilon * (s6 * s6 - s6) }
    }

    /// A Lennard-Jones interaction with the conventional `2.5σ` cutoff.
    pub fn with_default_cutoff(epsilon: f64, sigma: f64) -> Self {
        Self::new(epsilon, sigma, 2.5 * sigma)
    }

    /// Separation at the potential minimum, `2^(1/6)·σ`.
    pub fn r_min(&self) -> f64 {
        2f64.powf(1.0 / 6.0) * self.sigma
    }

    /// Pair energy at separation `r`, joules. Zero beyond the cutoff.
    pub fn pair_energy(&self, r: f64) -> f64 {
        if r >= self.cutoff {
            return 0.0;
        }
        let s6 = (self.sigma / r).powi(6);
        4.0 * self.epsilon * (s6 * s6 - s6) - self.energy_shift
    }

    /// Curvature at the potential minimum, `U''(r_min) = 72·2^(−1/3)·ε/σ²`.
    ///
    /// The effective spring constant of a bonded pair, which sets the highest
    /// vibrational frequency in the system and therefore the timestep.
    pub fn curvature_at_minimum(&self) -> f64 {
        72.0 * 2f64.powf(-1.0 / 3.0) * self.epsilon / (self.sigma * self.sigma)
    }
}

impl ForceLaw for LennardJones {
    fn name(&self) -> &'static str {
        "lennard_jones"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn cutoff(&self) -> Option<f64> {
        Some(self.cutoff)
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, neighbors: Option<&CellList>) {
        let Some(cells) = neighbors else {
            debug_assert!(false, "Lennard-Jones requires a neighbour list");
            return;
        };
        let sigma2 = self.sigma * self.sigma;
        let twenty_four_eps = 24.0 * self.epsilon;

        // `for_each_pair` needs to read positions while the force slices are borrowed
        // mutably, so the position slices are reborrowed here as shared.
        let (pos_x, pos_y) = (view.pos_x, view.pos_y);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);

        cells.for_each_pair(pos_x, pos_y, |i, j, dx, dy, r2| {
            let inv_r2 = 1.0 / r2;
            let s6 = (sigma2 * inv_r2).powi(3);
            let s12 = s6 * s6;
            // Positive coefficient means repulsion, which pushes i away from j.
            let coefficient = twenty_four_eps * inv_r2 * (2.0 * s12 - s6);
            let (fx, fy) = (coefficient * dx, coefficient * dy);
            // Newton's third law, applied by construction: whatever i loses, j gains.
            // This is what makes total momentum conserved to round-off.
            force_x[i] -= fx;
            force_y[i] -= fy;
            force_x[j] += fx;
            force_y[j] += fy;
        });
    }

    fn potential_energy(&self, store: &ParticleStore, neighbors: Option<&CellList>) -> f64 {
        let Some(cells) = neighbors else { return 0.0 };
        let sigma2 = self.sigma * self.sigma;
        let mut total = 0.0;
        cells.for_each_pair(store.pos_x(), store.pos_y(), |_, _, _, _, r2| {
            let s6 = (sigma2 / r2).powi(3);
            total += 4.0 * self.epsilon * (s6 * s6 - s6) - self.energy_shift;
        });
        total
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        let min_mass = store
            .mass()
            .iter()
            .copied()
            .filter(|m| m.is_finite() && *m > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !min_mass.is_finite() {
            return None;
        }
        // Highest vibrational frequency, from the curvature at the potential minimum.
        let omega = (self.curvature_at_minimum() / min_mass).sqrt();
        let max = 2.0 / omega;
        let period = core::f64::consts::TAU / omega;
        // MD practice is 20-50 steps per period of the fastest mode; take the
        // conservative end, since a dense phase samples steeper parts of the curve
        // than the minimum.
        Some(StableStep::limited(period / 50.0, max, StabilityReason::OscillationPeriod))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{ParticleSpec, ParticleStore};

    fn store_of(positions: &[[f64; 2]], mass: f64) -> ParticleStore {
        let mut s = ParticleStore::with_capacity(positions.len().max(1));
        for &p in positions {
            s.spawn(ParticleSpec::at(p).with_mass(mass)).unwrap();
        }
        s
    }

    #[test]
    fn uniform_acceleration_is_mass_proportional() {
        let mut store = ParticleStore::with_capacity(2);
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(1.0)).unwrap();
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(7.0)).unwrap();

        let g = UniformAcceleration::earth_gravity();
        let mut view = store.force_accumulation();
        g.accumulate(&mut view, None);

        // Force scales with mass, so acceleration does not.
        assert!((view.force_y[0] / view.mass[0] - view.force_y[1] / view.mass[1]).abs() < 1e-15);
        assert!((view.force_y[0] + 9.806_65).abs() < 1e-12);
    }

    #[test]
    fn gravitational_potential_is_mgh() {
        let store = store_of(&[[0.0, 3.0]], 2.0);
        let g = UniformAcceleration::earth_gravity();
        let u = g.potential_energy(&store, None);
        assert!((u - 2.0 * 9.806_65 * 3.0).abs() < 1e-9, "U = {u}");
    }

    #[test]
    fn pinned_particles_are_excluded_from_gravitational_potential() {
        let mut store = ParticleStore::with_capacity(2);
        store.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(1.0)).unwrap();
        store.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(f64::INFINITY)).unwrap();
        let u = UniformAcceleration::earth_gravity().potential_energy(&store, None);
        assert!(u.is_finite(), "an infinite-mass particle must not poison the total");
        assert!((u - 9.806_65).abs() < 1e-9);
    }

    #[test]
    fn drag_opposes_velocity_and_is_dissipative() {
        let mut store = ParticleStore::with_capacity(1);
        store
            .spawn(ParticleSpec::at([0.0, 0.0]).with_velocity([2.0, -3.0]).with_mass(1.0))
            .unwrap();
        let drag = LinearDrag::new(0.5);
        assert!(!drag.is_conservative());

        let mut view = store.force_accumulation();
        drag.accumulate(&mut view, None);
        assert!((view.force_x[0] + 1.0).abs() < 1e-15);
        assert!((view.force_y[0] - 1.5).abs() < 1e-15);
    }

    #[test]
    fn drag_stability_limit_scales_with_mass_over_coefficient() {
        let store = store_of(&[[0.0, 0.0]], 4.0);
        let limit = LinearDrag::new(2.0).stability_limit(&store).unwrap();
        assert!((limit.max - 4.0).abs() < 1e-12, "expected 2m/c = 4 s, got {}", limit.max);
    }

    #[test]
    fn harmonic_well_restores_toward_the_centre() {
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([2.0, 0.0]).with_mass(1.0)).unwrap();
        let well = HarmonicWell::new([0.0, 0.0], 3.0);

        let mut view = store.force_accumulation();
        well.accumulate(&mut view, None);
        assert!((view.force_x[0] + 6.0).abs() < 1e-15, "F = -kx = -6 N");
        assert_eq!(view.force_y[0], 0.0);
    }

    #[test]
    fn harmonic_potential_is_half_k_x_squared() {
        let store = store_of(&[[3.0, 4.0]], 1.0);
        let well = HarmonicWell::new([0.0, 0.0], 2.0);
        // r = 5, so U = 0.5 * 2 * 25 = 25 J.
        assert!((well.potential_energy(&store, None) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn harmonic_stability_limit_matches_two_over_omega() {
        let store = store_of(&[[1.0, 0.0]], 1.0);
        let well = HarmonicWell::new([0.0, 0.0], 4.0); // omega = 2 rad/s
        let limit = well.stability_limit(&store).unwrap();
        assert!((limit.max - 1.0).abs() < 1e-12, "2/omega = 1 s, got {}", limit.max);
        assert!(limit.preferred < limit.max, "the preferred step must resolve the oscillation");
    }

    #[test]
    fn lennard_jones_crosses_zero_at_sigma_and_bottoms_at_r_min() {
        let lj = LennardJones::with_default_cutoff(1.0, 1.0);
        // The shift moves the zero crossing slightly, so compare the unshifted form.
        let unshifted = |r: f64| {
            let s6 = (1.0f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        assert!(unshifted(1.0).abs() < 1e-12, "U(sigma) must vanish");
        let r_min = lj.r_min();
        assert!((r_min - 1.122_462_048_3).abs() < 1e-9, "r_min = {r_min}");
        assert!((unshifted(r_min) + 1.0).abs() < 1e-9, "the well depth must be -epsilon");

        // The minimum really is a minimum.
        assert!(unshifted(r_min) < unshifted(r_min * 0.99));
        assert!(unshifted(r_min) < unshifted(r_min * 1.01));
    }

    /// The whole point of the energy shift: no step in energy as a particle crosses
    /// the cutoff.
    #[test]
    fn energy_is_continuous_at_the_cutoff() {
        let lj = LennardJones::new(1.5, 0.8, 2.0);
        let inside = lj.pair_energy(2.0 - 1e-9);
        let outside = lj.pair_energy(2.0 + 1e-9);
        assert!(inside.abs() < 1e-8, "energy just inside the cutoff should be ~0, got {inside}");
        assert_eq!(outside, 0.0);
    }

    #[test]
    fn lennard_jones_curvature_matches_the_analytic_value() {
        let lj = LennardJones::with_default_cutoff(2.0, 3.0);
        // Numerically differentiate the unshifted potential twice at r_min.
        let u = |r: f64| {
            let s6 = (3.0f64 / r).powi(6);
            4.0 * 2.0 * (s6 * s6 - s6)
        };
        let r = lj.r_min();
        let h = r * 1e-5;
        let numeric = (u(r + h) - 2.0 * u(r) + u(r - h)) / (h * h);
        let analytic = lj.curvature_at_minimum();
        assert!(
            (numeric - analytic).abs() < 1e-4 * analytic.abs(),
            "numeric {numeric} vs analytic {analytic}"
        );
    }

    /// Newton's third law is what makes pairwise forces conserve momentum exactly.
    /// If this ever fails, every momentum-conservation test downstream is meaningless.
    #[test]
    fn pair_forces_sum_to_zero() {
        let positions = [[0.0, 0.0], [1.0, 0.2], [0.5, 0.9], [1.4, 1.1], [0.1, 1.3]];
        let mut store = store_of(&positions, 1.0);
        let lj = LennardJones::with_default_cutoff(1.0, 1.0);
        let mut cells =
            CellList::new([0.0, 0.0], [4.0, 4.0], [false, false], lj.cutoff, positions.len());
        cells.rebuild(&store);

        let mut view = store.force_accumulation();
        lj.accumulate(&mut view, Some(&cells));

        let sum_x: f64 = view.force_x.iter().sum();
        let sum_y: f64 = view.force_y.iter().sum();

        // The cancellation is exact in real arithmetic, so all that survives is the
        // round-off of summing terms of this magnitude. Judge against the force scale
        // rather than an absolute number: a genuine third-law violation would leave a
        // residual comparable to `scale`, not 10^-17 of it.
        let scale = view
            .force_x
            .iter()
            .chain(view.force_y.iter())
            .fold(0.0f64, |m, f| m.max(f.abs()));
        assert!(scale > 1.0, "the configuration must actually produce forces, got {scale}");
        assert!(sum_x.abs() < 1e-12 * scale, "net x force {sum_x} against scale {scale}");
        assert!(sum_y.abs() < 1e-12 * scale, "net y force {sum_y} against scale {scale}");
    }

    /// The force must be the negative gradient of the potential this law reports, or
    /// the energy monitor is measuring something the integrator is not solving.
    #[test]
    fn force_is_the_negative_gradient_of_the_reported_potential() {
        let lj = LennardJones::with_default_cutoff(1.0, 1.0);
        let cells = CellList::new([-5.0, -5.0], [10.0, 10.0], [false, false], lj.cutoff, 2);

        let energy_at = |separation: f64| {
            let mut s = store_of(&[[0.0, 0.0], [separation, 0.0]], 1.0);
            let mut c = cells.clone();
            c.rebuild(&s);
            let u = lj.potential_energy(&s, Some(&c));
            s.clear_forces();
            u
        };

        for r in [0.95, 1.05, 1.2, 1.6] {
            let h = 1e-6;
            let numeric_force = -(energy_at(r + h) - energy_at(r - h)) / (2.0 * h);

            let mut s = store_of(&[[0.0, 0.0], [r, 0.0]], 1.0);
            let mut c = cells.clone();
            c.rebuild(&s);
            let mut view = s.force_accumulation();
            lj.accumulate(&mut view, Some(&c));
            // Force on the second particle along +x.
            let analytic_force = view.force_x[1];

            assert!(
                (numeric_force - analytic_force).abs() < 1e-4 * analytic_force.abs().max(1.0),
                "at r={r}: numeric {numeric_force} vs analytic {analytic_force}"
            );
        }
    }

    #[test]
    fn lennard_jones_requires_and_reports_a_cutoff() {
        let lj = LennardJones::with_default_cutoff(1.0, 2.0);
        assert_eq!(lj.cutoff(), Some(5.0));
        assert!(lj.is_conservative());
        assert_eq!(UniformAcceleration::earth_gravity().cutoff(), None);
    }

    #[test]
    #[should_panic(expected = "positive epsilon and sigma")]
    fn degenerate_lennard_jones_parameters_are_rejected() {
        LennardJones::new(0.0, 1.0, 2.5);
    }
}
