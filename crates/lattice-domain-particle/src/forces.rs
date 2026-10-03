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
//!
//! # Every pair law reports its virial
//!
//! [`ForceLaw::virial`] is `Σ (xᵢ − xⱼ)·Fᵢⱼ` over the interactions a law produces,
//! the quantity the pressure estimate of spec §12.4 needs. External fields — gravity,
//! a harmonic well — are not pair interactions and contribute nothing to it, which is
//! the default.
//!
//! # The fastest pair mode is a *pair* mode
//!
//! Two equal masses `m` joined by a spring `k` oscillate at `ω = sqrt(k/μ)` with the
//! reduced mass `μ = m/2`, not at `sqrt(k/m)`. Every pair law here reports its limit
//! from the reduced mass of the two lightest particles, which is the highest
//! frequency the configuration can contain. (Until M5.1 the Lennard-Jones limit used
//! `m` alone and so overstated its stability limit by `√2`.)

use core::fmt;

use lattice_ir::{ForceAccumulation, ParticleStore, StabilityReason, StableStep};

use crate::image::MinimumImage;
use crate::verlet::NeighborList;

/// What a force law is handed alongside the particle arrays.
#[derive(Clone, Copy, Debug)]
pub struct ForceContext<'a> {
    /// The neighbour list, present exactly when some law reported a cutoff.
    pub neighbors: Option<&'a NeighborList>,
    /// How to measure a separation in this domain's box. Bonded laws need it even
    /// when no pair law is present.
    pub image: MinimumImage,
}

impl ForceContext<'_> {
    /// Open space, no neighbour list — what a law sees in a domain without a region.
    pub const fn open() -> Self {
        Self { neighbors: None, image: MinimumImage::open() }
    }
}

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

    /// Resolve handles and cache whatever the hot loop needs.
    ///
    /// Called by the domain at initialization and again whenever the store may have
    /// been touched. Laws without topology have nothing to do.
    fn prepare(&mut self, _store: &ParticleStore) {}

    /// Add this law's forces to the accumulators.
    ///
    /// `ctx.neighbors` is `Some` exactly when [`ForceLaw::cutoff`] returned `Some` for
    /// some law in the domain and the domain has built a list.
    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>);

    /// Potential energy of the current configuration, joules.
    ///
    /// Zero for non-conservative laws.
    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64;

    /// The pair virial `Σ (xᵢ − xⱼ)·Fᵢⱼ`, joules. Zero for external fields.
    fn virial(&self, _store: &ParticleStore, _ctx: &ForceContext<'_>) -> f64 {
        0.0
    }

    /// The timestep this law needs, if it imposes one.
    fn stability_limit(&self, _store: &ParticleStore) -> Option<StableStep> {
        None
    }

    /// Pairs of particle slots this law binds together, valid after
    /// [`ForceLaw::prepare`]. Drawn by viewers, and excluded from pair laws when
    /// [`ForceLaw::excludes_pair_forces`] says so.
    fn bonded_pairs(&self) -> &[[u32; 2]] {
        &[]
    }

    /// Whether pair laws must skip the pairs in [`ForceLaw::bonded_pairs`].
    fn excludes_pair_forces(&self) -> bool {
        false
    }
}

/// The lightest mobile mass in a store, or `None` when nothing can move.
fn lightest_mass(store: &ParticleStore) -> Option<f64> {
    let min = store
        .mass()
        .iter()
        .copied()
        .filter(|m| m.is_finite() && *m > 0.0)
        .fold(f64::INFINITY, f64::min);
    min.is_finite().then_some(min)
}

/// The stability limit of the fastest pair mode for a pair stiffness `k`.
///
/// The reduced mass of the two lightest particles is `m_min / 2`; symplectic Euler
/// and velocity Verlet are stable below `2/ω`, and MD practice resolves the fastest
/// period with 20–50 steps. The conservative end is taken, since a dense phase
/// samples steeper parts of a potential than its minimum.
fn pair_mode_limit(store: &ParticleStore, stiffness: f64) -> Option<StableStep> {
    let m_min = lightest_mass(store)?;
    if stiffness <= 0.0 {
        return None;
    }
    let omega = (stiffness / (0.5 * m_min)).sqrt();
    let period = core::f64::consts::TAU / omega;
    Some(StableStep::limited(period / 50.0, 2.0 / omega, StabilityReason::OscillationPeriod))
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

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, _ctx: &ForceContext<'_>) {
        let [gx, gy] = self.acceleration;
        for i in 0..view.len() {
            let m = view.mass[i];
            view.force_x[i] += m * gx;
            view.force_y[i] += m * gy;
        }
    }

    fn potential_energy(&self, store: &ParticleStore, _ctx: &ForceContext<'_>) -> f64 {
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

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, _ctx: &ForceContext<'_>) {
        let c = self.coefficient;
        for i in 0..view.len() {
            view.force_x[i] -= c * view.vel_x[i];
            view.force_y[i] -= c * view.vel_y[i];
        }
    }

    fn potential_energy(&self, _store: &ParticleStore, _ctx: &ForceContext<'_>) -> f64 {
        0.0
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        // Explicit integration of `dv/dt = -(c/m) v` is stable while dt < 2m/c.
        // The lightest particle binds.
        let min_mass = lightest_mass(store)?;
        if self.coefficient <= 0.0 {
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

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, _ctx: &ForceContext<'_>) {
        let [cx, cy] = self.center;
        let k = self.stiffness;
        for i in 0..view.len() {
            view.force_x[i] -= k * (view.pos_x[i] - cx);
            view.force_y[i] -= k * (view.pos_y[i] - cy);
        }
    }

    fn potential_energy(&self, store: &ParticleStore, _ctx: &ForceContext<'_>) -> f64 {
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
        let min_mass = lightest_mass(store)?;
        if self.stiffness <= 0.0 {
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

/// How a pair potential is made to vanish at its cutoff.
///
/// A bare truncation leaves a step in the energy at `r_c`, which an energy monitor
/// sees as a jump on every crossing. The two options here remove that step in
/// different ways, and each is a *different potential* — a model must say which it
/// means, because the two do not give the same pressure or the same equation of
/// state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Truncation {
    /// Subtract `U(r_c)`: the energy is continuous at the cutoff, the force is not.
    ///
    /// The conventional choice, and the one the GPU kernel implements. A particle
    /// crossing `r_c` feels a small step in force, so energy conservation holds only
    /// to the size of that step — which the contract declares.
    #[default]
    EnergyShift,
    /// Subtract `U(r_c) + (r − r_c)·U′(r_c)`: both energy and force are continuous.
    ///
    /// The force `F(r) − F(r_c)` differs from the true Lennard-Jones force everywhere
    /// inside the cutoff by the constant `F(r_c)`, so the minimum moves slightly and
    /// the well is a little shallower. Energy is then conserved to integration error
    /// alone, which is what a long microcanonical run wants.
    ForceShift,
}

impl Truncation {
    /// Name used in reports and project files.
    pub const fn name(self) -> &'static str {
        match self {
            Truncation::EnergyShift => "energy_shift",
            Truncation::ForceShift => "force_shift",
        }
    }
}

/// Truncated Lennard-Jones pair potential.
///
/// ```text
///   U(r) = 4ε[(σ/r)^12 − (σ/r)^6] − U_c − (r − r_c)·U′_c     for r < r_c
///        = 0                                                 otherwise
/// ```
///
/// with the `(r − r_c)·U′_c` term present only under [`Truncation::ForceShift`]. The
/// `−U_c` shift makes the *energy* continuous at the cutoff, which is what
/// energy-conservation tests measure; under an energy shift alone the *force* is
/// still discontinuous there, and this law declares that residual non-conservation
/// rather than pretending it away.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LennardJones {
    /// Well depth ε, joules.
    pub epsilon: f64,
    /// Length scale σ, metres. The potential crosses zero at `r = σ`.
    pub sigma: f64,
    /// Interaction cutoff, metres. Conventionally `2.5σ`.
    pub cutoff: f64,
    /// How the potential is brought to zero at the cutoff.
    pub truncation: Truncation,
    /// Precomputed `U_unshifted(cutoff)`, subtracted to make the energy continuous.
    energy_shift: f64,
    /// Precomputed `F_unshifted(cutoff)` (the radial force magnitude, positive for
    /// repulsion), subtracted under a force shift. Zero under an energy shift.
    force_shift: f64,
}

impl LennardJones {
    /// A Lennard-Jones interaction with an explicit cutoff and the conventional
    /// energy shift.
    ///
    /// # Panics
    ///
    /// If the cutoff is not greater than zero, or ε or σ are non-positive.
    pub fn new(epsilon: f64, sigma: f64, cutoff: f64) -> Self {
        Self::with_truncation(epsilon, sigma, cutoff, Truncation::EnergyShift)
    }

    /// A Lennard-Jones interaction with the given truncation.
    ///
    /// # Panics
    ///
    /// As [`LennardJones::new`].
    pub fn with_truncation(epsilon: f64, sigma: f64, cutoff: f64, truncation: Truncation) -> Self {
        assert!(epsilon > 0.0 && sigma > 0.0, "LJ requires positive epsilon and sigma");
        assert!(cutoff > 0.0, "LJ cutoff must be positive");
        let s6 = (sigma / cutoff).powi(6);
        let energy_shift = 4.0 * epsilon * (s6 * s6 - s6);
        let force_shift = match truncation {
            Truncation::EnergyShift => 0.0,
            // F(r) = 24ε/r · (2 s12 − s6), evaluated at the cutoff.
            Truncation::ForceShift => 24.0 * epsilon / cutoff * (2.0 * s6 * s6 - s6),
        };
        Self { epsilon, sigma, cutoff, truncation, energy_shift, force_shift }
    }

    /// A Lennard-Jones interaction with the conventional `2.5σ` cutoff.
    pub fn with_default_cutoff(epsilon: f64, sigma: f64) -> Self {
        Self::new(epsilon, sigma, 2.5 * sigma)
    }

    /// Separation at the minimum of the *untruncated* potential, `2^(1/6)·σ`.
    pub fn r_min(&self) -> f64 {
        2f64.powf(1.0 / 6.0) * self.sigma
    }

    /// Pair energy at separation `r`, joules. Zero beyond the cutoff.
    pub fn pair_energy(&self, r: f64) -> f64 {
        if r >= self.cutoff {
            return 0.0;
        }
        let s6 = (self.sigma / r).powi(6);
        4.0 * self.epsilon * (s6 * s6 - s6) - self.energy_shift + (r - self.cutoff) * self.force_shift
    }

    /// Radial pair force at separation `r`, newtons — positive for repulsion. Zero
    /// beyond the cutoff.
    pub fn pair_force(&self, r: f64) -> f64 {
        if r >= self.cutoff {
            return 0.0;
        }
        let s6 = (self.sigma / r).powi(6);
        24.0 * self.epsilon / r * (2.0 * s6 * s6 - s6) - self.force_shift
    }

    /// Curvature at the potential minimum, `U''(r_min) = 72·2^(−1/3)·ε/σ²`.
    ///
    /// The effective spring constant of a bonded pair, which sets the highest
    /// vibrational frequency in the system and therefore the timestep. A force shift
    /// adds a constant to the force and so leaves the curvature unchanged.
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

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>) {
        let Some(list) = ctx.neighbors else {
            debug_assert!(false, "Lennard-Jones requires a neighbour list");
            return;
        };
        let sigma2 = self.sigma * self.sigma;
        let twenty_four_eps = 24.0 * self.epsilon;
        let force_shift = self.force_shift;

        // `for_each_pair` needs to read positions while the force slices are borrowed
        // mutably, so the position slices are reborrowed here as shared.
        let (pos_x, pos_y) = (view.pos_x, view.pos_y);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);

        // Two loops rather than a branch per pair: the energy-shifted form needs no
        // square root, and that is the form the benchmarks and the GPU kernel share.
        if force_shift == 0.0 {
            list.for_each_pair(pos_x, pos_y, |i, j, dx, dy, r2| {
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
        } else {
            list.for_each_pair(pos_x, pos_y, |i, j, dx, dy, r2| {
                let inv_r2 = 1.0 / r2;
                let s6 = (sigma2 * inv_r2).powi(3);
                let s12 = s6 * s6;
                // F(r)/r − F(r_c)/r: the shift is a constant radial force.
                let coefficient = twenty_four_eps * inv_r2 * (2.0 * s12 - s6) - force_shift * inv_r2.sqrt();
                let (fx, fy) = (coefficient * dx, coefficient * dy);
                force_x[i] -= fx;
                force_y[i] -= fy;
                force_x[j] += fx;
                force_y[j] += fy;
            });
        }
    }

    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let Some(list) = ctx.neighbors else { return 0.0 };
        let sigma2 = self.sigma * self.sigma;
        let mut total = 0.0;
        if self.force_shift == 0.0 {
            list.for_each_pair(store.pos_x(), store.pos_y(), |_, _, _, _, r2| {
                let s6 = (sigma2 / r2).powi(3);
                total += 4.0 * self.epsilon * (s6 * s6 - s6) - self.energy_shift;
            });
        } else {
            list.for_each_pair(store.pos_x(), store.pos_y(), |_, _, _, _, r2| {
                let s6 = (sigma2 / r2).powi(3);
                total += 4.0 * self.epsilon * (s6 * s6 - s6) - self.energy_shift
                    + (r2.sqrt() - self.cutoff) * self.force_shift;
            });
        }
        total
    }

    fn virial(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let Some(list) = ctx.neighbors else { return 0.0 };
        let sigma2 = self.sigma * self.sigma;
        let twenty_four_eps = 24.0 * self.epsilon;
        let force_shift = self.force_shift;
        let mut total = 0.0;
        // (xᵢ − xⱼ)·Fᵢⱼ = coefficient · r², since Fᵢ = coefficient·(xᵢ − xⱼ).
        list.for_each_pair(store.pos_x(), store.pos_y(), |_, _, _, _, r2| {
            let inv_r2 = 1.0 / r2;
            let s6 = (sigma2 * inv_r2).powi(3);
            let coefficient = twenty_four_eps * inv_r2 * (2.0 * s6 * s6 - s6) - force_shift * inv_r2.sqrt();
            total += coefficient * r2;
        });
        total
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        // Highest vibrational frequency, from the curvature at the potential minimum
        // and the reduced mass of the lightest pair.
        pair_mode_limit(store, self.curvature_at_minimum())
    }
}

/// A soft repulsive disc, `U = ½·k·(d − r)²` for `r < d`.
///
/// The simplest excluded-volume interaction: a spring that only pushes. Both energy
/// and force go to zero continuously at `r = d`, so it needs no truncation shift and
/// conserves energy to integration error alone. Spec §12.4 lists it as *"soft
/// repulsion"* — the potential of choice for a granular or crowd-like scene where
/// the `r⁻¹²` wall of Lennard-Jones would force an unhelpfully small timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftRepulsion {
    /// Stiffness `k`, N/m.
    pub stiffness: f64,
    /// Range `d`, metres: the separation below which particles push apart.
    pub range: f64,
}

impl SoftRepulsion {
    /// A repulsion of the given stiffness acting inside `range`.
    ///
    /// # Panics
    ///
    /// On a non-positive stiffness or range.
    pub fn new(stiffness: f64, range: f64) -> Self {
        assert!(stiffness > 0.0 && range > 0.0, "soft repulsion needs positive stiffness and range");
        Self { stiffness, range }
    }

    /// Pair energy at separation `r`, joules.
    pub fn pair_energy(&self, r: f64) -> f64 {
        if r >= self.range {
            0.0
        } else {
            let overlap = self.range - r;
            0.5 * self.stiffness * overlap * overlap
        }
    }
}

impl ForceLaw for SoftRepulsion {
    fn name(&self) -> &'static str {
        "soft_repulsion"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn cutoff(&self) -> Option<f64> {
        Some(self.range)
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>) {
        let Some(list) = ctx.neighbors else {
            debug_assert!(false, "soft repulsion requires a neighbour list");
            return;
        };
        let (k, d) = (self.stiffness, self.range);
        let (pos_x, pos_y) = (view.pos_x, view.pos_y);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);
        list.for_each_pair(pos_x, pos_y, |i, j, dx, dy, r2| {
            let r = r2.sqrt();
            // F = k(d − r) along r̂, pushing i away from j.
            let coefficient = k * (d - r) / r;
            let (fx, fy) = (coefficient * dx, coefficient * dy);
            force_x[i] -= fx;
            force_y[i] -= fy;
            force_x[j] += fx;
            force_y[j] += fy;
        });
    }

    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let Some(list) = ctx.neighbors else { return 0.0 };
        let mut total = 0.0;
        list.for_each_pair(store.pos_x(), store.pos_y(), |_, _, _, _, r2| {
            total += self.pair_energy(r2.sqrt());
        });
        total
    }

    fn virial(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let Some(list) = ctx.neighbors else { return 0.0 };
        let (k, d) = (self.stiffness, self.range);
        let mut total = 0.0;
        list.for_each_pair(store.pos_x(), store.pos_y(), |_, _, _, _, r2| {
            let r = r2.sqrt();
            total += k * (d - r) * r;
        });
        total
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        pair_mode_limit(store, self.stiffness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verlet::Exclusions;
    use lattice_ir::{ParticleSpec, ParticleStore};

    fn store_of(positions: &[[f64; 2]], mass: f64) -> ParticleStore {
        let mut s = ParticleStore::with_capacity(positions.len().max(1));
        for &p in positions {
            s.spawn(ParticleSpec::at(p).with_mass(mass)).unwrap();
        }
        s
    }

    fn open_list(extent: f64, cutoff: f64, capacity: usize) -> NeighborList {
        NeighborList::new([-extent, -extent], [2.0 * extent; 2], [false; 2], cutoff, None, capacity, Exclusions::none())
    }

    #[test]
    fn uniform_acceleration_is_mass_proportional() {
        let mut store = ParticleStore::with_capacity(2);
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(1.0)).unwrap();
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(7.0)).unwrap();

        let g = UniformAcceleration::earth_gravity();
        let mut view = store.force_accumulation();
        g.accumulate(&mut view, &ForceContext::open());

        // Force scales with mass, so acceleration does not.
        assert!((view.force_y[0] / view.mass[0] - view.force_y[1] / view.mass[1]).abs() < 1e-15);
        assert!((view.force_y[0] + 9.806_65).abs() < 1e-12);
        assert_eq!(g.virial(&store, &ForceContext::open()), 0.0, "an external field has no pair virial");
    }

    #[test]
    fn gravitational_potential_is_mgh() {
        let store = store_of(&[[0.0, 3.0]], 2.0);
        let g = UniformAcceleration::earth_gravity();
        let u = g.potential_energy(&store, &ForceContext::open());
        assert!((u - 2.0 * 9.806_65 * 3.0).abs() < 1e-9, "U = {u}");
    }

    #[test]
    fn pinned_particles_are_excluded_from_gravitational_potential() {
        let mut store = ParticleStore::with_capacity(2);
        store.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(1.0)).unwrap();
        store.spawn(ParticleSpec::at([0.0, 1.0]).with_mass(f64::INFINITY)).unwrap();
        let u = UniformAcceleration::earth_gravity().potential_energy(&store, &ForceContext::open());
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
        drag.accumulate(&mut view, &ForceContext::open());
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
        well.accumulate(&mut view, &ForceContext::open());
        assert!((view.force_x[0] + 6.0).abs() < 1e-15, "F = -kx = -6 N");
        assert_eq!(view.force_y[0], 0.0);
    }

    #[test]
    fn harmonic_potential_is_half_k_x_squared() {
        let store = store_of(&[[3.0, 4.0]], 1.0);
        let well = HarmonicWell::new([0.0, 0.0], 2.0);
        // r = 5, so U = 0.5 * 2 * 25 = 25 J.
        assert!((well.potential_energy(&store, &ForceContext::open()) - 25.0).abs() < 1e-12);
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
        // But the force is not: the declared residual non-conservation.
        assert!(lj.pair_force(2.0 - 1e-9).abs() > 1e-3);
    }

    /// The force shift removes the step in force as well, at the cost of a slightly
    /// different potential inside the cutoff.
    #[test]
    fn a_force_shift_makes_the_force_continuous_too() {
        let lj = LennardJones::with_truncation(1.5, 0.8, 2.0, Truncation::ForceShift);
        assert!(lj.pair_energy(2.0 - 1e-9).abs() < 1e-8);
        assert!(lj.pair_force(2.0 - 1e-9).abs() < 1e-8, "force at the cutoff {}", lj.pair_force(2.0 - 1e-9));
        assert_eq!(lj.pair_force(2.5), 0.0);
        assert_eq!(lj.truncation.name(), "force_shift");

        // Inside the cutoff the shifted force differs from the plain one by exactly
        // the constant F(r_c).
        let plain = LennardJones::new(1.5, 0.8, 2.0);
        let f_c = plain.pair_force(2.0 - 1e-12);
        for r in [0.9, 1.1, 1.5, 1.9] {
            assert!((plain.pair_force(r) - lj.pair_force(r) - f_c).abs() < 1e-9 * f_c.abs().max(1.0));
        }
        // And the force is still the negative gradient of the energy.
        for r in [0.95, 1.2, 1.7] {
            let h = 1e-6;
            let numeric = -(lj.pair_energy(r + h) - lj.pair_energy(r - h)) / (2.0 * h);
            assert!((numeric - lj.pair_force(r)).abs() < 1e-5 * lj.pair_force(r).abs().max(1.0));
        }
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
        for truncation in [Truncation::EnergyShift, Truncation::ForceShift] {
            let lj = LennardJones::with_truncation(1.0, 1.0, 2.5, truncation);
            let mut list = open_list(4.0, lj.cutoff, positions.len());
            list.update(&store);
            let ctx = ForceContext { neighbors: Some(&list), image: list.image() };

            store.clear_forces();
            let mut view = store.force_accumulation();
            lj.accumulate(&mut view, &ctx);

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
    }

    /// The force must be the negative gradient of the potential this law reports, or
    /// the energy monitor is measuring something the integrator is not solving.
    #[test]
    fn force_is_the_negative_gradient_of_the_reported_potential() {
        for truncation in [Truncation::EnergyShift, Truncation::ForceShift] {
            let lj = LennardJones::with_truncation(1.0, 1.0, 2.5, truncation);

            let energy_at = |separation: f64| {
                let s = store_of(&[[0.0, 0.0], [separation, 0.0]], 1.0);
                let mut list = open_list(5.0, lj.cutoff, 2);
                list.update(&s);
                let ctx = ForceContext { neighbors: Some(&list), image: list.image() };
                lj.potential_energy(&s, &ctx)
            };

            for r in [0.95, 1.05, 1.2, 1.6, 2.4] {
                let h = 1e-6;
                let numeric_force = -(energy_at(r + h) - energy_at(r - h)) / (2.0 * h);

                let mut s = store_of(&[[0.0, 0.0], [r, 0.0]], 1.0);
                let mut list = open_list(5.0, lj.cutoff, 2);
                list.update(&s);
                let ctx = ForceContext { neighbors: Some(&list), image: list.image() };
                let mut view = s.force_accumulation();
                lj.accumulate(&mut view, &ctx);
                // Force on the second particle along +x.
                let analytic_force = view.force_x[1];

                assert!(
                    (numeric_force - analytic_force).abs() < 1e-4 * analytic_force.abs().max(1.0),
                    "{truncation:?} at r={r}: numeric {numeric_force} vs analytic {analytic_force}"
                );
                assert!((analytic_force - lj.pair_force(r)).abs() < 1e-9 * lj.pair_force(r).abs().max(1.0));
            }
        }
    }

    /// The virial of one pair is r·F(r): repulsive inside r_min, attractive outside.
    #[test]
    fn lennard_jones_virial_is_r_times_the_pair_force() {
        let lj = LennardJones::with_default_cutoff(1.0, 1.0);
        for r in [1.0, 1.3, 2.0] {
            let s = store_of(&[[0.0, 0.0], [r, 0.0]], 1.0);
            let mut list = open_list(5.0, lj.cutoff, 2);
            list.update(&s);
            let ctx = ForceContext { neighbors: Some(&list), image: list.image() };
            let expected = r * lj.pair_force(r);
            let virial = lj.virial(&s, &ctx);
            assert!((virial - expected).abs() < 1e-12 * expected.abs().max(1.0), "r={r}: {virial} vs {expected}");
        }
    }

    #[test]
    fn lennard_jones_requires_and_reports_a_cutoff() {
        let lj = LennardJones::with_default_cutoff(1.0, 2.0);
        assert_eq!(lj.cutoff(), Some(5.0));
        assert!(lj.is_conservative());
        assert_eq!(UniformAcceleration::earth_gravity().cutoff(), None);
    }

    /// The fastest mode of a pair is at the reduced mass, `√2` above the single-mass
    /// estimate for equal masses.
    #[test]
    fn pair_stability_uses_the_reduced_mass() {
        let store = store_of(&[[0.0, 0.0], [1.5, 0.0]], 2.0);
        let lj = LennardJones::with_default_cutoff(1.0, 1.0);
        let limit = lj.stability_limit(&store).unwrap();
        let omega = (lj.curvature_at_minimum() / 1.0).sqrt(); // μ = 2/2 = 1
        assert!((limit.max - 2.0 / omega).abs() < 1e-12, "{} vs {}", limit.max, 2.0 / omega);
        assert_eq!(limit.reason, StabilityReason::OscillationPeriod);

        let soft = SoftRepulsion::new(9.0, 1.0);
        let limit = soft.stability_limit(&store).unwrap();
        assert!((limit.max - 2.0 / 3.0).abs() < 1e-12, "ω = sqrt(9/1) = 3: {}", limit.max);
    }

    #[test]
    #[should_panic(expected = "positive epsilon and sigma")]
    fn degenerate_lennard_jones_parameters_are_rejected() {
        LennardJones::new(0.0, 1.0, 2.5);
    }

    #[test]
    fn soft_repulsion_pushes_apart_inside_its_range_and_vanishes_smoothly() {
        let soft = SoftRepulsion::new(10.0, 1.0);
        assert_eq!(soft.cutoff(), Some(1.0));
        assert_eq!(soft.pair_energy(1.0), 0.0);
        assert!((soft.pair_energy(0.5) - 1.25).abs() < 1e-12, "½·10·0.5² = 1.25 J");

        let mut s = store_of(&[[0.0, 0.0], [0.6, 0.0]], 1.0);
        let mut list = open_list(3.0, 1.0, 2);
        list.update(&s);
        let ctx = ForceContext { neighbors: Some(&list), image: list.image() };
        let mut view = s.force_accumulation();
        soft.accumulate(&mut view, &ctx);
        // k(d − r) = 10·0.4 = 4 N, pushing 1 toward +x and 0 toward −x.
        assert!((view.force_x[1] - 4.0).abs() < 1e-12, "{}", view.force_x[1]);
        assert!((view.force_x[0] + 4.0).abs() < 1e-12);
        assert!((soft.virial(&s, &ctx) - 0.6 * 4.0).abs() < 1e-12);

        // Gradient check across the range boundary.
        for r in [0.3, 0.7, 0.99] {
            let energy_at = |r: f64| {
                let s = store_of(&[[0.0, 0.0], [r, 0.0]], 1.0);
                let mut list = open_list(3.0, 1.0, 2);
                list.update(&s);
                soft.potential_energy(&s, &ForceContext { neighbors: Some(&list), image: list.image() })
            };
            let h = 1e-6;
            let numeric = -(energy_at(r + h) - energy_at(r - h)) / (2.0 * h);
            assert!((numeric - 10.0 * (1.0 - r)).abs() < 1e-5, "r={r}: {numeric}");
        }
    }

    #[test]
    #[should_panic(expected = "positive stiffness and range")]
    fn degenerate_soft_repulsion_is_rejected() {
        SoftRepulsion::new(1.0, 0.0);
    }
}
