//! User-defined force laws (spec §8.3): a compiled [`Program`] in the force loop.
//!
//! The compiler checks a `force` or `potential` declaration, lowers it to a program
//! whose inputs follow [`layout`], and binds its parameters where the law is used. This
//! module runs it: once per pair inside the cutoff for a law between two particles, once
//! per particle for a law on one.
//!
//! # A force is applied in equal and opposite pairs
//!
//! A pair law returns the force on `a`, and the engine adds its negation to `b`, so total
//! momentum is conserved by construction whatever the author wrote. For a pair `force`
//! that is all the engine can promise: it may read velocities and owes nothing to an
//! energy. A pair `potential` returns `U(r)` and nothing else, and the force is the
//! derivative the compiler took, `F_a = U′(r) d̂` with `d = x_b − x_a`. That force is
//! central, so angular momentum is conserved too, and it conserves energy wherever `U`
//! is continuous.
//!
//! # Same arithmetic, same bits
//!
//! The separation `d` and `r² = d·d` are the neighbour list's own, and `r` is `√r²`.
//! The force on `a` is added and its negation subtracted from `b`, which in IEEE
//! arithmetic is exactly what the built-in laws do when they subtract a force from `i`
//! and add it to `j`. A law written in a built-in's operation order therefore
//! reproduces that built-in to the last bit, which is what the validation suite checks.

use lattice_expr::{Program, MAX_REGISTERS};
use lattice_ir::{ForceAccumulation, ParticleStore, StableStep};

use crate::forces::{pair_mode_limit, ForceContext, ForceLaw, Truncation};

/// Where a program finds its inputs.
///
/// A law between two particles reads the separation `d = x_b − x_a` (through the
/// nearest image), its length `r`, and then each particle's members. A law on one
/// particle reads that particle's members alone.
pub mod layout {
    /// `d.x`, m: the separation from `a` to `b` through the nearest image.
    pub const DX: u32 = 0;
    /// `d.y`, m.
    pub const DY: u32 = 1;
    /// `r = |d|`, m.
    pub const R: u32 = 2;
    /// Where particle `a`'s members start, in a pair law.
    pub const PAIR_A: u32 = 3;
    /// Where particle `b`'s members start, in a pair law.
    pub const PAIR_B: u32 = PAIR_A + FIELDS;
    /// Inputs to a pair law.
    pub const PAIR_INPUTS: u32 = PAIR_B + FIELDS;
    /// Where the particle's members start, in a law on one particle.
    pub const BODY_A: u32 = 0;
    /// Inputs to a law on one particle.
    pub const BODY_INPUTS: u32 = FIELDS;

    /// `x`, m, as an offset within a particle's members.
    pub const PX: u32 = 0;
    /// `y`, m.
    pub const PY: u32 = 1;
    /// `vₓ`, m/s.
    pub const VX: u32 = 2;
    /// `v_y`, m/s.
    pub const VY: u32 = 3;
    /// Mass, kg.
    pub const MASS: u32 = 4;
    /// Charge, C.
    pub const CHARGE: u32 = 5;
    /// Members per particle.
    pub const FIELDS: u32 = 6;
}

/// What a user law computes, which decides what the engine does with its outputs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UserLawKind {
    /// `force f(a, b) -> vec2<newton>`: outputs `(Fx, Fy)` on `a`.
    PairForce,
    /// `potential u(a, b) -> joule`: outputs `(U, dU/dr)`.
    PairPotential,
    /// `force f(a) -> vec2<newton>`: outputs `(Fx, Fy)`.
    BodyForce,
    /// `potential u(a) -> joule`: outputs `(U, ∂U/∂x, ∂U/∂y)`.
    BodyPotential,
}

impl UserLawKind {
    /// True for a law between two particles.
    pub fn is_pair(self) -> bool {
        matches!(self, UserLawKind::PairForce | UserLawKind::PairPotential)
    }

    /// True for a law derived from an energy.
    pub fn is_potential(self) -> bool {
        matches!(self, UserLawKind::PairPotential | UserLawKind::BodyPotential)
    }

    /// How many outputs its program must have.
    pub fn outputs(self) -> usize {
        match self {
            UserLawKind::PairForce | UserLawKind::BodyForce | UserLawKind::PairPotential => 2,
            UserLawKind::BodyPotential => 3,
        }
    }

    /// How many inputs its program reads.
    pub fn inputs(self) -> usize {
        if self.is_pair() { layout::PAIR_INPUTS as usize } else { layout::BODY_INPUTS as usize }
    }
}

/// A compiled user law with its parameters bound.
#[derive(Clone, PartialEq, Debug)]
pub struct UserLaw {
    name: String,
    kind: UserLawKind,
    program: Program,
    params: Vec<f64>,
    cutoff: Option<f64>,
    truncation: Option<Truncation>,
    /// `U(r_c)`, for a pair potential.
    energy_at_cutoff: f64,
    /// `U′(r_c)`, for a pair potential.
    slope_at_cutoff: f64,
    /// The pair stiffness the timestep is limited by, when the compiler could find one.
    stiffness: Option<f64>,
    /// True when the law reads a mass or a charge, so `U(r_c)` and `U′(r_c)` differ from
    /// pair to pair and are evaluated for each pair rather than once.
    member_dependent: bool,
}

impl UserLaw {
    /// A user law.
    ///
    /// `program` must already return what [`UserLawKind::outputs`] says — for a
    /// potential, the compiler's derivative program — and read the inputs of
    /// [`layout`]. A pair law needs a `cutoff`; a pair potential may be truncated.
    ///
    /// # Panics
    ///
    /// If the program's shape does not match the kind, it does not fit
    /// [`MAX_REGISTERS`], or a pair law has no positive cutoff: all compiler bugs.
    pub fn new(
        name: impl Into<String>,
        kind: UserLawKind,
        program: Program,
        params: Vec<f64>,
        cutoff: Option<f64>,
        truncation: Option<Truncation>,
    ) -> UserLaw {
        assert_eq!(program.output_count(), kind.outputs(), "{kind:?} program has the wrong outputs");
        assert_eq!(program.input_count(), kind.inputs(), "{kind:?} program has the wrong inputs");
        assert_eq!(program.param_count(), params.len(), "unbound parameters");
        assert!(program.fits(), "a program larger than the register file");
        if kind.is_pair() {
            assert!(cutoff.is_some_and(|c| c > 0.0), "a pair law needs a positive cutoff");
        }
        let mut law = UserLaw {
            name: name.into(),
            kind,
            program,
            params,
            cutoff,
            truncation: if kind == UserLawKind::PairPotential { truncation } else { None },
            energy_at_cutoff: 0.0,
            slope_at_cutoff: 0.0,
            stiffness: None,
            member_dependent: false,
        };
        if let (UserLawKind::PairPotential, Some(rc)) = (kind, cutoff) {
            let at = law.pair_potential(rc);
            law.energy_at_cutoff = at.0;
            law.slope_at_cutoff = at.1;
        }
        law
    }

    /// Declare the pair stiffness `k` whose mode bounds the step, as for a built-in
    /// pair potential: the curvature at the potential's minimum.
    pub fn with_stiffness(mut self, stiffness: f64) -> UserLaw {
        self.stiffness = (stiffness.is_finite() && stiffness > 0.0).then_some(stiffness);
        self
    }

    /// Declare that the law reads a mass or a charge. A truncated pair potential then
    /// evaluates its energy and slope at the cutoff for each pair, with that pair's own
    /// members, instead of once: a charged potential's `U(r_c)` is `q_a q_b`-dependent.
    pub fn with_member_dependence(mut self, member_dependent: bool) -> UserLaw {
        self.member_dependent = member_dependent;
        self
    }

    /// What it computes.
    pub fn kind(&self) -> UserLawKind {
        self.kind
    }

    /// The cutoff of a pair law, m; `None` for a law on one particle.
    pub fn pair_cutoff(&self) -> Option<f64> {
        if self.kind.is_pair() { self.cutoff } else { None }
    }

    /// The pair stiffness bounding the step, N/m, when one was found.
    pub fn stiffness(&self) -> Option<f64> {
        self.stiffness
    }

    /// How a pair potential is brought to zero at its cutoff, if it is.
    pub fn truncation(&self) -> Option<Truncation> {
        self.truncation
    }

    /// The program's raw outputs at the given inputs, laid out as [`layout`] says: for
    /// the compiler's checks of a bound law, never a hot loop.
    pub fn outputs_at(&self, inputs: &[f64]) -> Vec<f64> {
        self.program.eval_to_vec(inputs, &self.params)
    }

    /// `U(r)` and `U′(r)` of a pair potential between uncharged particles of zero mass,
    /// as written: no truncation applied. For a law that reads no member that is the law.
    pub fn pair_potential(&self, r: f64) -> (f64, f64) {
        self.pair_potential_between(r, [0.0, 0.0], [0.0, 0.0])
    }

    /// `U(r)` and `U′(r)` of a pair potential between particles with the given
    /// `[mass, charge]`, as written: no truncation applied.
    pub fn pair_potential_between(&self, r: f64, a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
        let mut inputs = [0.0; layout::PAIR_INPUTS as usize];
        inputs[layout::R as usize] = r;
        inputs[layout::DX as usize] = r;
        for (base, [mass, charge]) in [(layout::PAIR_A, a), (layout::PAIR_B, b)] {
            inputs[(base + layout::MASS) as usize] = mass;
            inputs[(base + layout::CHARGE) as usize] = charge;
        }
        let out = self.program.eval_to_vec(&inputs, &self.params);
        (out[0], out[1])
    }

    /// The truncated energy and the slope the force uses, at separation `r`, given the
    /// energy and slope at the cutoff for this pair.
    fn truncated(&self, u: f64, du: f64, r: f64, at_cutoff: (f64, f64)) -> (f64, f64) {
        let (u_c, du_c) = at_cutoff;
        match self.truncation {
            None => (u, du),
            Some(Truncation::EnergyShift) => (u - u_c, du),
            Some(Truncation::ForceShift) => {
                let rc = self.cutoff.unwrap_or(r);
                (u - u_c - (r - rc) * du_c, du - du_c)
            }
        }
    }

    /// Visit every pair inside the cutoff with the force on `a`, `(i, j, dx, dy, fx, fy)`,
    /// and its energy. The one place a pair law is evaluated, so the force loop, the
    /// energy and the virial cannot disagree about what the law is.
    #[allow(clippy::too_many_arguments)]
    fn for_each_pair_force(
        &self,
        list: &crate::verlet::NeighborList,
        pos: (&[f64], &[f64]),
        vel: (&[f64], &[f64]),
        mass: &[f64],
        charge: &[f64],
        mut visit: impl FnMut(usize, usize, f64, f64, f64, f64, f64),
    ) {
        let cutoff = self.cutoff.unwrap_or(0.0);
        let cutoff2 = cutoff * cutoff;
        let mut regs = [0.0; MAX_REGISTERS];
        let mut inputs = [0.0; layout::PAIR_INPUTS as usize];
        let mut out = [0.0; 2];
        let mut at_cutoff = [0.0; 2];
        let per_pair = self.member_dependent && self.truncation.is_some();
        list.for_each_pair(pos.0, pos.1, |i, j, dx, dy, r2| {
            if r2 >= cutoff2 {
                return;
            }
            let r = r2.sqrt();
            inputs[layout::DX as usize] = dx;
            inputs[layout::DY as usize] = dy;
            inputs[layout::R as usize] = r;
            for (base, k) in [(layout::PAIR_A, i), (layout::PAIR_B, j)] {
                let base = base as usize;
                inputs[base + layout::PX as usize] = pos.0[k];
                inputs[base + layout::PY as usize] = pos.1[k];
                inputs[base + layout::VX as usize] = vel.0[k];
                inputs[base + layout::VY as usize] = vel.1[k];
                inputs[base + layout::MASS as usize] = mass[k];
                inputs[base + layout::CHARGE as usize] = charge[k];
            }
            self.program.eval(&inputs, &self.params, &mut regs, &mut out);
            match self.kind {
                UserLawKind::PairForce => visit(i, j, dx, dy, out[0], out[1], 0.0),
                _ => {
                    // A potential that reads a member has its own U(r_c) for each pair:
                    // evaluate it again at the cutoff with this pair's members. A
                    // potential reads positions only through r, so only r changes.
                    let shift = if per_pair {
                        inputs[layout::R as usize] = cutoff;
                        self.program.eval(&inputs, &self.params, &mut regs, &mut at_cutoff);
                        inputs[layout::R as usize] = r;
                        (at_cutoff[0], at_cutoff[1])
                    } else {
                        (self.energy_at_cutoff, self.slope_at_cutoff)
                    };
                    // F_a = −∂U/∂x_a = U′(r) d/r, since ∂r/∂x_a = −d/r.
                    let (u, du) = self.truncated(out[0], out[1], r, shift);
                    let coefficient = du / r;
                    visit(i, j, dx, dy, coefficient * dx, coefficient * dy, u);
                }
            }
        });
    }

    /// Visit every particle with the force on it and its energy.
    fn for_each_body_force(
        &self,
        pos: (&[f64], &[f64]),
        vel: (&[f64], &[f64]),
        mass: &[f64],
        charge: &[f64],
        mut visit: impl FnMut(usize, f64, f64, f64),
    ) {
        let mut regs = [0.0; MAX_REGISTERS];
        let mut inputs = [0.0; layout::BODY_INPUTS as usize];
        let mut out = [0.0; 3];
        for k in 0..pos.0.len() {
            inputs[layout::PX as usize] = pos.0[k];
            inputs[layout::PY as usize] = pos.1[k];
            inputs[layout::VX as usize] = vel.0[k];
            inputs[layout::VY as usize] = vel.1[k];
            inputs[layout::MASS as usize] = mass[k];
            inputs[layout::CHARGE as usize] = charge[k];
            self.program.eval(&inputs, &self.params, &mut regs, &mut out);
            match self.kind {
                UserLawKind::BodyForce => visit(k, out[0], out[1], 0.0),
                _ => visit(k, -out[1], -out[2], out[0]),
            }
        }
    }
}

impl ForceLaw for UserLaw {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_conservative(&self) -> bool {
        self.kind.is_potential()
    }

    fn conserves_momentum(&self) -> bool {
        // A pair law is applied equal and opposite; a law on one particle is a field.
        self.kind.is_pair()
    }

    fn cutoff(&self) -> Option<f64> {
        if self.kind.is_pair() { self.cutoff } else { None }
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>) {
        let pos = (view.pos_x, view.pos_y);
        let vel = (view.vel_x, view.vel_y);
        let (mass, charge) = (view.mass, view.charge);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);
        if self.kind.is_pair() {
            let Some(list) = ctx.neighbors else {
                debug_assert!(false, "a pair law requires a neighbour list");
                return;
            };
            self.for_each_pair_force(list, pos, vel, mass, charge, |i, j, _, _, fx, fy, _| {
                // The force on `a` is added and its negation taken from `b`: equal and
                // opposite by construction.
                force_x[i] += fx;
                force_y[i] += fy;
                force_x[j] -= fx;
                force_y[j] -= fy;
            });
        } else {
            self.for_each_body_force(pos, vel, mass, charge, |k, fx, fy, _| {
                force_x[k] += fx;
                force_y[k] += fy;
            });
        }
    }

    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        if !self.kind.is_potential() {
            return 0.0;
        }
        let pos = (store.pos_x(), store.pos_y());
        let vel = (store.vel_x(), store.vel_y());
        let mut total = 0.0;
        if self.kind.is_pair() {
            let Some(list) = ctx.neighbors else { return 0.0 };
            self.for_each_pair_force(list, pos, vel, store.mass(), store.charge(), |_, _, _, _, _, _, u| total += u);
        } else {
            self.for_each_body_force(pos, vel, store.mass(), store.charge(), |_, _, _, u| total += u);
        }
        total
    }

    fn virial(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        if !self.kind.is_pair() {
            return 0.0;
        }
        let Some(list) = ctx.neighbors else { return 0.0 };
        let mut total = 0.0;
        // (xᵢ − xⱼ)·Fᵢ = −d·F_a.
        self.for_each_pair_force(
            list,
            (store.pos_x(), store.pos_y()),
            (store.vel_x(), store.vel_y()),
            store.mass(),
            store.charge(),
            |_, _, dx, dy, fx, fy, _| total -= dx * fx + dy * fy,
        );
        total
    }

    fn stability_limit(&self, store: &ParticleStore) -> Option<StableStep> {
        pair_mode_limit(store, self.stiffness?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_expr::{Builder, Op};

    /// `U(r) = ½k(r − r₀)²` between two particles, as the compiler would lower it.
    fn spring_potential(k: f64, r0: f64) -> UserLaw {
        let mut b = Builder::new(layout::PAIR_INPUTS, 2);
        let r = b.push(Op::Input(layout::R));
        let stiffness = b.push(Op::Param(0));
        let rest = b.push(Op::Param(1));
        let stretch = b.push(Op::Sub(r, rest));
        let square = b.push(Op::Mul(stretch, stretch));
        let half = b.constant(0.5);
        let half_k = b.push(Op::Mul(half, stiffness));
        let u = b.push(Op::Mul(half_k, square));
        let program = b.finish(vec![u]).derivative(layout::R);
        UserLaw::new("spring", UserLawKind::PairPotential, program, vec![k, r0], Some(5.0), None)
    }

    #[test]
    fn a_pair_potential_reports_its_energy_and_slope() {
        let law = spring_potential(3.0, 1.0);
        let (u, du) = law.pair_potential(1.5);
        assert!((u - 0.375).abs() < 1e-15 && (du - 1.5).abs() < 1e-15, "{u}, {du}");
        assert!(law.is_conservative());
    }

    #[test]
    fn an_energy_shift_zeroes_the_energy_at_the_cutoff() {
        let mut law = spring_potential(3.0, 1.0);
        law.truncation = Some(Truncation::EnergyShift);
        let (u, du) = law.pair_potential(5.0);
        assert_eq!(law.truncated(u, du, 5.0, (law.energy_at_cutoff, law.slope_at_cutoff)).0, 0.0);
    }
}
