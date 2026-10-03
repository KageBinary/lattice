//! Bonded interactions: harmonic bonds and harmonic angles.
//!
//! Spec §12.4 lists *"harmonic bonds/angles"* among the potentials the molecular
//! module should support, and §20.2 scopes the MVP to *"2D LJ and harmonic bonds"*.
//! A bond is a *topological* interaction — it acts between two named particles for
//! the whole run, whatever their separation — which is what distinguishes it from a
//! pair law found through the neighbour list.
//!
//! # Handles, not slots
//!
//! Topology is declared in terms of [`ParticleId`] handles and resolved to storage
//! slots by [`ForceLaw::prepare`], which the domain calls whenever the store may
//! have been touched. The hot loop then reads plain slot pairs. A handle that no
//! longer resolves is a configuration error reported at preparation, not a silently
//! skipped bond.
//!
//! # Every bonded law reports its virial
//!
//! Pressure is `P·A = K + ½·Σ rᵢⱼ·Fᵢⱼ` (2D), and a bond contributes to the sum like
//! any other pair interaction. An angle does not: the bending energy depends only on
//! an angle, which is invariant under a uniform scaling of all positions, so its
//! virial is identically zero. [`HarmonicAngle::virial`] returns zero *because of that
//! identity*, not because it was left unimplemented.

use lattice_ir::{ForceAccumulation, ParticleId, ParticleStore, StabilityReason, StableStep};

use crate::forces::{ForceContext, ForceLaw};

/// Steps per period of the fastest bonded mode that the preferred timestep resolves.
///
/// The same guidance the pair laws use: 20–50 steps per period is MD practice, and
/// the conservative end is taken because a bond samples its full curvature at every
/// stretch rather than only near the minimum.
const STEPS_PER_PERIOD: f64 = 50.0;

/// One harmonic bond, `U = ½·k·(r − r₀)²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bond {
    /// One end.
    pub a: ParticleId,
    /// The other end.
    pub b: ParticleId,
    /// Rest length `r₀`, metres.
    pub length: f64,
    /// Stiffness `k`, newtons per metre.
    pub stiffness: f64,
}

impl Bond {
    /// A bond between two particles.
    pub fn new(a: ParticleId, b: ParticleId, length: f64, stiffness: f64) -> Self {
        Self { a, b, length, stiffness }
    }
}

/// A set of harmonic bonds.
///
/// ```text
///   U = Σ_bonds ½·k·(|x_b − x_a| − r₀)²
///   F_a = k·(r − r₀)·(x_b − x_a)/r        F_b = −F_a
/// ```
///
/// with `x_b − x_a` taken under the minimum-image convention, so a bond may span a
/// periodic seam. A bond of zero current length has no defined direction and exerts
/// no force that step; the energy `½·k·r₀²` is still counted.
#[derive(Clone, Debug)]
pub struct HarmonicBond {
    bonds: Vec<Bond>,
    /// Resolved `[slot_a, slot_b]` per bond, refreshed by [`ForceLaw::prepare`].
    slots: Vec<[u32; 2]>,
    exclude_pair_forces: bool,
}

impl HarmonicBond {
    /// Bonds that also switch off pair laws between their ends — the usual choice.
    pub fn new(bonds: Vec<Bond>) -> Self {
        Self { bonds, slots: Vec::new(), exclude_pair_forces: true }
    }

    /// Whether pair laws (Lennard-Jones, soft repulsion) still act across a bond.
    ///
    /// Off by default: the bond *is* the interaction. A bead–spring polymer that wants
    /// its beads to keep repelling each other switches this on.
    pub fn with_pair_forces(mut self, keep: bool) -> Self {
        self.exclude_pair_forces = !keep;
        self
    }

    /// The bonds, as declared.
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }

    /// Current bond lengths, in declaration order. Empty until prepared.
    pub fn lengths(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> Vec<f64> {
        let (xs, ys) = (store.pos_x(), store.pos_y());
        self.slots
            .iter()
            .map(|&[a, b]| {
                let (a, b) = (a as usize, b as usize);
                let (dx, dy) = ctx.image.separation(xs[b] - xs[a], ys[b] - ys[a]);
                (dx * dx + dy * dy).sqrt()
            })
            .collect()
    }
}

impl ForceLaw for HarmonicBond {
    fn name(&self) -> &'static str {
        "harmonic_bond"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn prepare(&mut self, store: &ParticleStore) {
        self.slots.clear();
        for (index, bond) in self.bonds.iter().enumerate() {
            let a = store.slot_of(bond.a).unwrap_or_else(|| {
                panic!("bond {index}: particle {:?} is not alive in the store", bond.a)
            });
            let b = store.slot_of(bond.b).unwrap_or_else(|| {
                panic!("bond {index}: particle {:?} is not alive in the store", bond.b)
            });
            assert!(a != b, "bond {index} joins a particle to itself");
            self.slots.push([a as u32, b as u32]);
        }
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>) {
        debug_assert_eq!(self.slots.len(), self.bonds.len(), "prepare() must run before accumulate()");
        let (pos_x, pos_y) = (view.pos_x, view.pos_y);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);
        for (bond, &[a, b]) in self.bonds.iter().zip(&self.slots) {
            let (a, b) = (a as usize, b as usize);
            let (dx, dy) = ctx.image.separation(pos_x[b] - pos_x[a], pos_y[b] - pos_y[a]);
            let r = (dx * dx + dy * dy).sqrt();
            if r == 0.0 {
                continue;
            }
            // Positive when stretched: pulls a toward b, and b toward a.
            let tension = bond.stiffness * (r - bond.length) / r;
            let (fx, fy) = (tension * dx, tension * dy);
            force_x[a] += fx;
            force_y[a] += fy;
            force_x[b] -= fx;
            force_y[b] -= fy;
        }
    }

    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let (xs, ys) = (store.pos_x(), store.pos_y());
        let mut total = 0.0;
        for (bond, &[a, b]) in self.bonds.iter().zip(&self.slots) {
            let (a, b) = (a as usize, b as usize);
            let (dx, dy) = ctx.image.separation(xs[b] - xs[a], ys[b] - ys[a]);
            let stretch = (dx * dx + dy * dy).sqrt() - bond.length;
            total += 0.5 * bond.stiffness * stretch * stretch;
        }
        total
    }

    fn virial(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        // Σ (x_a − x_b)·F_a = Σ −tension·r², negative for a stretched bond, which is
        // the sign an attractive interaction contributes to the pressure.
        let (xs, ys) = (store.pos_x(), store.pos_y());
        let mut total = 0.0;
        for (bond, &[a, b]) in self.bonds.iter().zip(&self.slots) {
            let (a, b) = (a as usize, b as usize);
            let (dx, dy) = ctx.image.separation(xs[b] - xs[a], ys[b] - ys[a]);
            let r2 = dx * dx + dy * dy;
            let r = r2.sqrt();
            if r == 0.0 {
                continue;
            }
            total -= bond.stiffness * (r - bond.length) * r;
        }
        total
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        // The fastest bond vibration: ω² = k/μ with μ the reduced mass. A pinned end
        // (infinite mass) leaves the other end's mass as μ.
        let masses = store.mass();
        let mut omega_max: f64 = 0.0;
        for (bond, &[a, b]) in self.bonds.iter().zip(&self.slots) {
            let (ma, mb) = (masses[a as usize], masses[b as usize]);
            let reduced = reduced_mass(ma, mb);
            if reduced.is_finite() && reduced > 0.0 && bond.stiffness > 0.0 {
                omega_max = omega_max.max((bond.stiffness / reduced).sqrt());
            }
        }
        if omega_max == 0.0 {
            return None;
        }
        let period = core::f64::consts::TAU / omega_max;
        Some(StableStep::limited(period / STEPS_PER_PERIOD, 2.0 / omega_max, StabilityReason::OscillationPeriod))
    }

    fn bonded_pairs(&self) -> &[[u32; 2]] {
        &self.slots
    }

    fn excludes_pair_forces(&self) -> bool {
        self.exclude_pair_forces
    }
}

/// The reduced mass of two particles, either of which may be pinned.
fn reduced_mass(ma: f64, mb: f64) -> f64 {
    match (ma.is_finite(), mb.is_finite()) {
        (true, true) => ma * mb / (ma + mb),
        (true, false) => ma,
        (false, true) => mb,
        (false, false) => f64::INFINITY,
    }
}

/// One harmonic angle at vertex `b`, `U = ½·k_θ·(θ − θ₀)²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle {
    /// First arm.
    pub a: ParticleId,
    /// The vertex.
    pub b: ParticleId,
    /// Second arm.
    pub c: ParticleId,
    /// Rest angle `θ₀`, radians, in `[0, π]`.
    pub angle: f64,
    /// Stiffness `k_θ`, joules per radian squared.
    pub stiffness: f64,
}

impl Angle {
    /// An angle `a–b–c` with its vertex at `b`.
    pub fn new(a: ParticleId, b: ParticleId, c: ParticleId, angle: f64, stiffness: f64) -> Self {
        Self { a, b, c, angle, stiffness }
    }
}

/// A set of harmonic angles.
///
/// ```text
///   u = x_a − x_b,  v = x_c − x_b   (minimum image)
///   θ = |atan2(u × v, u · v)|       ∈ [0, π]
///   U = Σ ½·k_θ·(θ − θ₀)²
/// ```
///
/// The angle is taken from `atan2` of the cross and dot products rather than from
/// `acos` of the cosine, because the gradient of `atan2` is smooth everywhere the
/// arms have length — `acos` loses precision near 0 and π and its gradient carries a
/// `1/sin θ` that a straight chain divides by zero. The gradients are
///
/// ```text
///   ∂θ/∂u = ( (u·v)·v_y − (u×v)·v_x ,  −(u·v)·v_x − (u×v)·v_y ) / (|u|²|v|²)
///   ∂θ/∂v = ( −(u·v)·u_y − (u×v)·u_x ,  (u·v)·u_x − (u×v)·u_y ) / (|u|²|v|²)
/// ```
///
/// with `F_a = −k_θ(θ−θ₀)·sign·∂θ/∂u`, `F_c` likewise, and `F_b = −(F_a + F_c)` so
/// the three forces sum to zero and momentum is conserved by construction. At an
/// exactly straight configuration the unsigned angle has a cusp; the force there is
/// whatever the signed branch gives, which is zero when `θ₀ = π` and finite otherwise.
#[derive(Clone, Debug)]
pub struct HarmonicAngle {
    angles: Vec<Angle>,
    slots: Vec<[u32; 3]>,
}

impl HarmonicAngle {
    /// A set of angles.
    pub fn new(angles: Vec<Angle>) -> Self {
        Self { angles, slots: Vec::new() }
    }

    /// The angles, as declared.
    pub fn angles(&self) -> &[Angle] {
        &self.angles
    }

    /// Current angles in radians, in declaration order. Empty until prepared.
    pub fn values(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> Vec<f64> {
        let (xs, ys) = (store.pos_x(), store.pos_y());
        self.slots
            .iter()
            .map(|&[a, b, c]| {
                let (a, b, c) = (a as usize, b as usize, c as usize);
                let (ux, uy) = ctx.image.separation(xs[a] - xs[b], ys[a] - ys[b]);
                let (vx, vy) = ctx.image.separation(xs[c] - xs[b], ys[c] - ys[b]);
                (ux * vy - uy * vx).atan2(ux * vx + uy * vy).abs()
            })
            .collect()
    }
}

impl ForceLaw for HarmonicAngle {
    fn name(&self) -> &'static str {
        "harmonic_angle"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn prepare(&mut self, store: &ParticleStore) {
        self.slots.clear();
        for (index, angle) in self.angles.iter().enumerate() {
            let resolve = |id: ParticleId| {
                store.slot_of(id).unwrap_or_else(|| {
                    panic!("angle {index}: particle {id:?} is not alive in the store")
                }) as u32
            };
            let slots = [resolve(angle.a), resolve(angle.b), resolve(angle.c)];
            assert!(
                slots[0] != slots[1] && slots[1] != slots[2] && slots[0] != slots[2],
                "angle {index} names the same particle twice"
            );
            self.slots.push(slots);
        }
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>) {
        debug_assert_eq!(self.slots.len(), self.angles.len(), "prepare() must run before accumulate()");
        let (pos_x, pos_y) = (view.pos_x, view.pos_y);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);
        for (angle, &[a, b, c]) in self.angles.iter().zip(&self.slots) {
            let (a, b, c) = (a as usize, b as usize, c as usize);
            let (ux, uy) = ctx.image.separation(pos_x[a] - pos_x[b], pos_y[a] - pos_y[b]);
            let (vx, vy) = ctx.image.separation(pos_x[c] - pos_x[b], pos_y[c] - pos_y[b]);
            let dot = ux * vx + uy * vy;
            let cross = ux * vy - uy * vx;
            let denominator = (ux * ux + uy * uy) * (vx * vx + vy * vy);
            if denominator == 0.0 {
                continue;
            }
            let signed = cross.atan2(dot);
            let sign = if signed < 0.0 { -1.0 } else { 1.0 };
            // −dU/dθ, carried onto the signed branch.
            let torque = -angle.stiffness * (signed.abs() - angle.angle) * sign;

            let gu = ((dot * vy - cross * vx) / denominator, (-dot * vx - cross * vy) / denominator);
            let gv = ((-dot * uy - cross * ux) / denominator, (dot * ux - cross * uy) / denominator);
            let (fax, fay) = (torque * gu.0, torque * gu.1);
            let (fcx, fcy) = (torque * gv.0, torque * gv.1);
            force_x[a] += fax;
            force_y[a] += fay;
            force_x[c] += fcx;
            force_y[c] += fcy;
            force_x[b] -= fax + fcx;
            force_y[b] -= fay + fcy;
        }
    }

    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        self.values(store, ctx)
            .iter()
            .zip(&self.angles)
            .map(|(&theta, angle)| {
                let bend = theta - angle.angle;
                0.5 * angle.stiffness * bend * bend
            })
            .sum()
    }

    fn virial(&self, _store: &ParticleStore, _ctx: &ForceContext<'_>) -> f64 {
        // θ is invariant under x → λx, so Σ xᵢ·Fᵢ = −Σ xᵢ·∂U/∂xᵢ = −dU/dλ|₁ = 0.
        0.0
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        // The bending mode of a linear triatomic with stiff arms of length r:
        // ω² = (k_θ / r²)·(1/m_a + 4/m_b + 1/m_c). Used as an estimate of the fastest
        // bending frequency; the current arm lengths stand in for the rest lengths.
        let (xs, ys, masses) = (store.pos_x(), store.pos_y(), store.mass());
        let inverse = |m: f64| if m.is_finite() && m > 0.0 { 1.0 / m } else { 0.0 };
        let mut omega_max: f64 = 0.0;
        for (angle, &[a, b, c]) in self.angles.iter().zip(&self.slots) {
            let (a, b, c) = (a as usize, b as usize, c as usize);
            let arm = |p: usize| ((xs[p] - xs[b]).powi(2) + (ys[p] - ys[b]).powi(2)).sqrt();
            let r = arm(a).min(arm(c));
            let inertia = inverse(masses[a]) + 4.0 * inverse(masses[b]) + inverse(masses[c]);
            if r > 0.0 && r.is_finite() && inertia > 0.0 && angle.stiffness > 0.0 {
                omega_max = omega_max.max((angle.stiffness * inertia / (r * r)).sqrt());
            }
        }
        if omega_max == 0.0 {
            return None;
        }
        let period = core::f64::consts::TAU / omega_max;
        Some(StableStep::limited(period / STEPS_PER_PERIOD, 2.0 / omega_max, StabilityReason::OscillationPeriod))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::MinimumImage;
    use lattice_ir::ParticleSpec;

    fn store_of(positions: &[[f64; 2]], mass: f64) -> (ParticleStore, Vec<ParticleId>) {
        let mut store = ParticleStore::with_capacity(positions.len());
        let ids = positions.iter().map(|&p| store.spawn(ParticleSpec::at(p).with_mass(mass)).unwrap()).collect();
        (store, ids)
    }

    fn open() -> ForceContext<'static> {
        ForceContext { neighbors: None, image: MinimumImage::open() }
    }

    fn forces_on(law: &dyn ForceLaw, store: &mut ParticleStore, ctx: &ForceContext<'_>) -> Vec<[f64; 2]> {
        store.clear_forces();
        let mut view = store.force_accumulation();
        law.accumulate(&mut view, ctx);
        (0..view.len()).map(|i| [view.force_x[i], view.force_y[i]]).collect()
    }

    /// Central finite-difference gradient of a law's potential, one particle and axis
    /// at a time. The reference every force implementation is checked against.
    fn numeric_forces(law: &dyn ForceLaw, store: &mut ParticleStore, ctx: &ForceContext<'_>, h: f64) -> Vec<[f64; 2]> {
        (0..store.len())
            .map(|i| {
                [0usize, 1].map(|axis| {
                    let mut shifted = |delta: f64| {
                        {
                            let d = store.dynamics();
                            if axis == 0 { d.pos_x[i] += delta } else { d.pos_y[i] += delta }
                        }
                        let u = law.potential_energy(store, ctx);
                        {
                            let d = store.dynamics();
                            if axis == 0 { d.pos_x[i] -= delta } else { d.pos_y[i] -= delta }
                        }
                        u
                    };
                    -(shifted(h) - shifted(-h)) / (2.0 * h)
                })
            })
            .collect()
    }

    #[test]
    fn a_bond_at_rest_length_exerts_no_force() {
        let (mut store, ids) = store_of(&[[0.0, 0.0], [1.5, 0.0]], 1.0);
        let mut law = HarmonicBond::new(vec![Bond::new(ids[0], ids[1], 1.5, 10.0)]);
        law.prepare(&store);
        let f = forces_on(&law, &mut store, &open());
        assert_eq!(f, vec![[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(law.potential_energy(&store, &open()), 0.0);
    }

    #[test]
    fn a_stretched_bond_pulls_its_ends_together() {
        let (mut store, ids) = store_of(&[[0.0, 0.0], [2.0, 0.0]], 1.0);
        let mut law = HarmonicBond::new(vec![Bond::new(ids[0], ids[1], 1.5, 10.0)]);
        law.prepare(&store);
        let f = forces_on(&law, &mut store, &open());
        // k·(r − r0) = 10·0.5 = 5 N, toward each other.
        assert!((f[0][0] - 5.0).abs() < 1e-12, "{f:?}");
        assert!((f[1][0] + 5.0).abs() < 1e-12, "{f:?}");
        assert!((law.potential_energy(&store, &open()) - 1.25).abs() < 1e-12);
        // Stretched: the virial is −k(r−r0)·r = −10.
        assert!((law.virial(&store, &open()) + 10.0).abs() < 1e-12);
    }

    #[test]
    fn bond_forces_are_the_negative_gradient_of_the_energy() {
        let (mut store, ids) = store_of(&[[0.1, 0.2], [1.3, 0.9], [0.4, 1.7], [2.2, 2.1]], 1.0);
        let mut law = HarmonicBond::new(vec![
            Bond::new(ids[0], ids[1], 1.0, 7.0),
            Bond::new(ids[1], ids[2], 1.2, 3.0),
            Bond::new(ids[2], ids[3], 0.8, 11.0),
            Bond::new(ids[3], ids[0], 1.5, 5.0),
        ]);
        law.prepare(&store);
        let analytic = forces_on(&law, &mut store, &open());
        let numeric = numeric_forces(&law, &mut store, &open(), 1e-6);
        for (a, n) in analytic.iter().zip(&numeric) {
            for axis in 0..2 {
                assert!((a[axis] - n[axis]).abs() < 1e-6, "analytic {a:?} vs numeric {n:?}");
            }
        }
        let net: [f64; 2] = analytic.iter().fold([0.0; 2], |s, f| [s[0] + f[0], s[1] + f[1]]);
        assert!(net[0].abs() < 1e-12 && net[1].abs() < 1e-12, "bonds must conserve momentum: {net:?}");
    }

    #[test]
    fn a_bond_spans_a_periodic_seam() {
        let (mut store, ids) = store_of(&[[0.2, 5.0], [9.9, 5.0]], 1.0);
        let mut law = HarmonicBond::new(vec![Bond::new(ids[0], ids[1], 0.3, 10.0)]);
        law.prepare(&store);
        let ctx = ForceContext { neighbors: None, image: MinimumImage::new([10.0, 10.0], [true, true]) };
        // 0.3 apart through the wall, exactly at rest length: no force.
        let f = forces_on(&law, &mut store, &ctx);
        assert!(f[0][0].abs() < 1e-12 && f[1][0].abs() < 1e-12, "{f:?}");
        assert_eq!(law.lengths(&store, &ctx).len(), 1);
        assert!((law.lengths(&store, &ctx)[0] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn bond_stability_uses_the_reduced_mass() {
        let (store, ids) = store_of(&[[0.0, 0.0], [1.0, 0.0]], 2.0);
        let mut law = HarmonicBond::new(vec![Bond::new(ids[0], ids[1], 1.0, 8.0)]);
        law.prepare(&store);
        // μ = 1, ω = sqrt(8), max = 2/ω.
        let limit = law.stability_limit(&store).unwrap();
        assert!((limit.max - 2.0 / 8f64.sqrt()).abs() < 1e-12, "{}", limit.max);
        assert_eq!(limit.reason, StabilityReason::OscillationPeriod);
        assert_eq!(reduced_mass(3.0, f64::INFINITY), 3.0);
        assert!(reduced_mass(f64::INFINITY, f64::INFINITY).is_infinite());
    }

    #[test]
    fn bonds_exclude_pair_forces_unless_asked_not_to() {
        let (store, ids) = store_of(&[[0.0, 0.0], [1.0, 0.0]], 1.0);
        let mut law = HarmonicBond::new(vec![Bond::new(ids[0], ids[1], 1.0, 1.0)]);
        law.prepare(&store);
        assert!(law.excludes_pair_forces());
        assert_eq!(law.bonded_pairs(), &[[0, 1]]);
        let kept = HarmonicBond::new(vec![]).with_pair_forces(true);
        assert!(!kept.excludes_pair_forces());
    }

    #[test]
    #[should_panic(expected = "is not alive")]
    fn a_bond_to_a_dead_particle_is_a_configuration_error() {
        let (mut store, ids) = store_of(&[[0.0, 0.0], [1.0, 0.0]], 1.0);
        store.despawn(ids[1]);
        let mut law = HarmonicBond::new(vec![Bond::new(ids[0], ids[1], 1.0, 1.0)]);
        law.prepare(&store);
    }

    #[test]
    fn a_right_angle_at_rest_exerts_no_force_and_a_bent_one_restores() {
        let (mut store, ids) = store_of(&[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], 1.0);
        let right = core::f64::consts::FRAC_PI_2;
        let mut law = HarmonicAngle::new(vec![Angle::new(ids[0], ids[1], ids[2], right, 2.0)]);
        law.prepare(&store);
        let f = forces_on(&law, &mut store, &open());
        for force in &f {
            assert!(force[0].abs() < 1e-12 && force[1].abs() < 1e-12, "{f:?}");
        }
        assert!((law.values(&store, &open())[0] - right).abs() < 1e-12);

        // Open the angle to 120°: the arms are pushed back toward 90°.
        let mut law = HarmonicAngle::new(vec![Angle::new(ids[0], ids[1], ids[2], right, 2.0)]);
        {
            let d = store.dynamics();
            d.pos_x[2] = -0.5;
            d.pos_y[2] = 3f64.sqrt() / 2.0;
        }
        law.prepare(&store);
        let f = forces_on(&law, &mut store, &open());
        // Particle a at (1,0): closing the angle means pushing it toward +y.
        assert!(f[0][1] > 0.0, "{f:?}");
        assert!(f[2][0] > 0.0, "c must be pushed toward +x: {f:?}");
        let expected = 0.5 * 2.0 * (core::f64::consts::PI / 6.0).powi(2);
        assert!((law.potential_energy(&store, &open()) - expected).abs() < 1e-12);
    }

    #[test]
    fn angle_forces_are_the_negative_gradient_of_the_energy() {
        let (mut store, ids) = store_of(&[[0.3, 0.1], [1.2, 0.7], [1.9, 1.8], [0.8, 2.4], [2.9, 0.2]], 1.0);
        let mut law = HarmonicAngle::new(vec![
            Angle::new(ids[0], ids[1], ids[2], 1.9, 4.0),
            Angle::new(ids[1], ids[2], ids[3], 2.6, 1.5),
            Angle::new(ids[2], ids[1], ids[4], 0.7, 2.5),
        ]);
        law.prepare(&store);
        let analytic = forces_on(&law, &mut store, &open());
        let numeric = numeric_forces(&law, &mut store, &open(), 1e-6);
        for (a, n) in analytic.iter().zip(&numeric) {
            for axis in 0..2 {
                assert!((a[axis] - n[axis]).abs() < 1e-6, "analytic {a:?} vs numeric {n:?}");
            }
        }
        let net: [f64; 2] = analytic.iter().fold([0.0; 2], |s, f| [s[0] + f[0], s[1] + f[1]]);
        assert!(net[0].abs() < 1e-12 && net[1].abs() < 1e-12, "angles must conserve momentum: {net:?}");
        // Bending forces produce no torque about any point either: angular momentum.
        let (xs, ys) = (store.pos_x(), store.pos_y());
        let torque: f64 = analytic.iter().enumerate().map(|(i, f)| xs[i] * f[1] - ys[i] * f[0]).sum();
        assert!(torque.abs() < 1e-12, "net torque {torque}");
        assert_eq!(law.virial(&store, &open()), 0.0);
    }

    #[test]
    fn a_straight_chain_at_its_rest_angle_is_force_free() {
        let (mut store, ids) = store_of(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], 1.0);
        let mut law = HarmonicAngle::new(vec![Angle::new(ids[0], ids[1], ids[2], core::f64::consts::PI, 3.0)]);
        law.prepare(&store);
        let f = forces_on(&law, &mut store, &open());
        for force in &f {
            assert!(force[0].is_finite() && force[1].is_finite());
            assert!(force[0].abs() < 1e-12 && force[1].abs() < 1e-12, "{f:?}");
        }
        assert!((law.values(&store, &open())[0] - core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn angle_stability_matches_the_triatomic_bending_estimate() {
        let (store, ids) = store_of(&[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], 1.0);
        let mut law = HarmonicAngle::new(vec![Angle::new(ids[0], ids[1], ids[2], core::f64::consts::PI, 2.0)]);
        law.prepare(&store);
        let limit = law.stability_limit(&store).unwrap();
        // ω² = (k/r²)(1 + 4 + 1) = 12.
        assert!((limit.max - 2.0 / 12f64.sqrt()).abs() < 1e-12, "{}", limit.max);
    }
}
