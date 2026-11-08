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
//!
//! # Evaluated a batch at a time
//!
//! The loops gather up to [`LANES`] pairs (or particles), evaluate them in one
//! [`Program::eval_batch`] pass, and then visit them in the order they were gathered.
//! Each lane is bit-identical to the scalar evaluation, and the visit sequence is the
//! scalar one, so forces, energy and virial accumulate to the same bits whichever
//! [`Evaluation`] a law uses. That holds for every non-NaN result: `eval_batch` does
//! not pin a NaN's payload, so a NaN may differ in its bits (not in being NaN). The
//! scalar path stays as the reference and for the benchmark's comparison.

use lattice_expr::{NonFiniteAt, Op, Program, LANES, MAX_REGISTERS};
use lattice_ir::{FaultSite, ForceAccumulation, LawFault, ParticleStore, StableStep};

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

    /// What input `n` is called in a message: `d.x`, `r`, `a.position.x`, `b.mass`. The
    /// two particles of a pair law are `a` and `b`; a law on one particle has `a` alone.
    pub fn input_name(pair: bool, n: u32) -> String {
        let member = |offset: u32| match offset {
            PX => "position.x",
            PY => "position.y",
            VX => "velocity.x",
            VY => "velocity.y",
            MASS => "mass",
            _ => "charge",
        };
        if !pair {
            return format!("a.{}", member(n % FIELDS));
        }
        match n {
            DX => "d.x".to_string(),
            DY => "d.y".to_string(),
            R => "r".to_string(),
            _ => {
                let particle = if n < PAIR_B { "a" } else { "b" };
                let base = if n < PAIR_B { PAIR_A } else { PAIR_B };
                format!("{particle}.{}", member(n - base))
            }
        }
    }
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

/// How the force loops run a law's program.
///
/// Both give the same bits for every non-NaN result (a NaN's payload is not pinned);
/// they differ in speed alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Evaluation {
    /// One pair or particle at a time, through [`Program::eval`].
    Scalar,
    /// [`LANES`] at a time, through [`Program::eval_batch`].
    #[default]
    Batched,
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
    /// Whether the loops evaluate one at a time or a batch at a time.
    evaluation: Evaluation,
}

/// Up to [`LANES`] pairs gathered for one pass of the program, with their scratch. A
/// stack value: the step allocates nothing (NFR-001).
struct PairBatch {
    len: usize,
    ij: [(usize, usize); LANES],
    d: [(f64, f64); LANES],
    r: [f64; LANES],
    /// `inputs[k * LANES + lane]`, as [`Program::eval_batch`] reads them.
    inputs: [f64; layout::PAIR_INPUTS as usize * LANES],
    regs: [f64; MAX_REGISTERS * LANES],
    out: [f64; 2 * LANES],
    at_cutoff: [f64; 2 * LANES],
}

/// Up to [`LANES`] particles gathered for one pass of the program.
struct BodyBatch {
    len: usize,
    index: [usize; LANES],
    inputs: [f64; layout::BODY_INPUTS as usize * LANES],
    regs: [f64; MAX_REGISTERS * LANES],
    out: [f64; 3 * LANES],
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
            evaluation: Evaluation::default(),
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

    /// True when the law reads a mass or a charge (see [`Self::with_member_dependence`]).
    pub fn is_member_dependent(&self) -> bool {
        self.member_dependent
    }

    /// Choose how the force loops run the program. The default is
    /// [`Evaluation::Batched`]; the answer is the same either way.
    pub fn with_evaluation(mut self, evaluation: Evaluation) -> UserLaw {
        self.evaluation = evaluation;
        self
    }

    /// How the force loops run the program.
    pub fn evaluation(&self) -> Evaluation {
        self.evaluation
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

    /// The first operation of the program to make a non-finite value out of finite ones,
    /// when the program's outputs at `inputs` are not all finite.
    fn non_finite_at(&self, inputs: &[f64]) -> Option<NonFiniteAt> {
        let mut regs = [0.0; MAX_REGISTERS];
        let mut out = [0.0; 3];
        self.program.eval(inputs, &self.params, &mut regs, &mut out);
        if out[..self.kind.outputs()].iter().all(|v| v.is_finite()) {
            return None;
        }
        self.program.trace_outputs(inputs, &self.params, 0..self.kind.outputs())
    }

    /// A fault at `site`, naming the input when that is what arrived non-finite.
    fn fault(&self, site: FaultSite, at: NonFiniteAt) -> LawFault {
        let input = match at.op {
            Op::Input(n) => Some(layout::input_name(self.kind.is_pair(), n)),
            Op::Param(n) => Some(format!("parameter {n}")),
            _ => None,
        };
        LawFault { domain: String::new(), law: self.name.clone(), site, at, input }
    }

    /// The first pair or particle, in visit order, whose evaluation is not finite.
    fn first_fault(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> Option<LawFault> {
        let (pos, vel) = ((store.pos_x(), store.pos_y()), (store.vel_x(), store.vel_y()));
        let (mass, charge) = (store.mass(), store.charge());
        if !self.kind.is_pair() {
            let mut inputs = [0.0; layout::BODY_INPUTS as usize];
            for k in 0..pos.0.len() {
                inputs[layout::PX as usize] = pos.0[k];
                inputs[layout::PY as usize] = pos.1[k];
                inputs[layout::VX as usize] = vel.0[k];
                inputs[layout::VY as usize] = vel.1[k];
                inputs[layout::MASS as usize] = mass[k];
                inputs[layout::CHARGE as usize] = charge[k];
                if let Some(at) = self.non_finite_at(&inputs) {
                    return Some(self.fault(FaultSite::Particle(k), at));
                }
            }
            return None;
        }
        let list = ctx.neighbors?;
        let cutoff = self.cutoff.unwrap_or(0.0);
        let cutoff2 = cutoff * cutoff;
        let per_pair = self.member_dependent && self.truncation.is_some();
        let mut inputs = [0.0; layout::PAIR_INPUTS as usize];
        let mut found: Option<(usize, usize, NonFiniteAt)> = None;
        list.for_each_pair(pos.0, pos.1, |i, j, dx, dy, r2| {
            if found.is_some() || r2 >= cutoff2 {
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
            let mut at = self.non_finite_at(&inputs);
            if at.is_none() && per_pair && self.kind != UserLawKind::PairForce {
                // The energy and slope at the cutoff that this pair's truncation uses.
                inputs[layout::R as usize] = cutoff;
                at = self.non_finite_at(&inputs);
            }
            found = at.map(|at| (i, j, at));
        });
        found.map(|(i, j, at)| self.fault(FaultSite::Pair(i, j), at))
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
    ///
    /// The visits come in the neighbour list's order whichever [`Evaluation`] is in use.
    #[allow(clippy::too_many_arguments)]
    fn for_each_pair_force(
        &self,
        list: &crate::verlet::NeighborList,
        pos: (&[f64], &[f64]),
        vel: (&[f64], &[f64]),
        mass: &[f64],
        charge: &[f64],
        visit: impl FnMut(usize, usize, f64, f64, f64, f64, f64),
    ) {
        match self.evaluation {
            Evaluation::Scalar => self.pair_forces_scalar(list, pos, vel, mass, charge, visit),
            Evaluation::Batched => self.pair_forces_batched(list, pos, vel, mass, charge, visit),
        }
    }

    /// [`Self::for_each_pair_force`] a batch at a time: gather [`LANES`] pairs, evaluate
    /// them in one pass, and visit them in the order they were gathered.
    #[allow(clippy::too_many_arguments)]
    fn pair_forces_batched(
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
        let per_pair = self.member_dependent && self.truncation.is_some();
        let mut batch = PairBatch {
            len: 0,
            ij: [(0, 0); LANES],
            d: [(0.0, 0.0); LANES],
            r: [0.0; LANES],
            inputs: [0.0; layout::PAIR_INPUTS as usize * LANES],
            regs: [0.0; MAX_REGISTERS * LANES],
            out: [0.0; 2 * LANES],
            at_cutoff: [0.0; 2 * LANES],
        };
        list.for_each_pair(pos.0, pos.1, |i, j, dx, dy, r2| {
            if r2 >= cutoff2 {
                return;
            }
            let r = r2.sqrt();
            let lane = batch.len;
            batch.ij[lane] = (i, j);
            batch.d[lane] = (dx, dy);
            batch.r[lane] = r;
            let mut put = |input: u32, value: f64| batch.inputs[input as usize * LANES + lane] = value;
            put(layout::DX, dx);
            put(layout::DY, dy);
            put(layout::R, r);
            for (base, k) in [(layout::PAIR_A, i), (layout::PAIR_B, j)] {
                put(base + layout::PX, pos.0[k]);
                put(base + layout::PY, pos.1[k]);
                put(base + layout::VX, vel.0[k]);
                put(base + layout::VY, vel.1[k]);
                put(base + layout::MASS, mass[k]);
                put(base + layout::CHARGE, charge[k]);
            }
            batch.len += 1;
            if batch.len == LANES {
                self.flush_pairs(&mut batch, cutoff, per_pair, &mut visit);
            }
        });
        self.flush_pairs(&mut batch, cutoff, per_pair, &mut visit);
    }

    /// Evaluate the gathered pairs and visit each in gather order, then empty the batch.
    fn flush_pairs(
        &self,
        batch: &mut PairBatch,
        cutoff: f64,
        per_pair: bool,
        visit: &mut impl FnMut(usize, usize, f64, f64, f64, f64, f64),
    ) {
        let n = batch.len;
        if n == 0 {
            return;
        }
        self.program.eval_batch(n, &batch.inputs, &self.params, &mut batch.regs, &mut batch.out);
        let potential = self.kind != UserLawKind::PairForce;
        if potential && per_pair {
            // A potential that reads a member has its own U(r_c) for each pair: evaluate
            // the batch again with `r` set to the cutoff, as the scalar path does pair
            // by pair. A potential reads positions only through r, so only r changes.
            let row = layout::R as usize * LANES;
            let mut saved = [0.0; LANES];
            saved.copy_from_slice(&batch.inputs[row..row + LANES]);
            batch.inputs[row..row + LANES].fill(cutoff);
            self.program.eval_batch(n, &batch.inputs, &self.params, &mut batch.regs, &mut batch.at_cutoff);
            batch.inputs[row..row + LANES].copy_from_slice(&saved);
        }
        for lane in 0..n {
            let (i, j) = batch.ij[lane];
            let (dx, dy) = batch.d[lane];
            let (first, second) = (batch.out[lane], batch.out[LANES + lane]);
            if !potential {
                visit(i, j, dx, dy, first, second, 0.0);
                continue;
            }
            let shift = if per_pair {
                (batch.at_cutoff[lane], batch.at_cutoff[LANES + lane])
            } else {
                (self.energy_at_cutoff, self.slope_at_cutoff)
            };
            // F_a = −∂U/∂x_a = U′(r) d/r, since ∂r/∂x_a = −d/r.
            let r = batch.r[lane];
            let (u, du) = self.truncated(first, second, r, shift);
            let coefficient = du / r;
            visit(i, j, dx, dy, coefficient * dx, coefficient * dy, u);
        }
        batch.len = 0;
    }

    /// [`Self::for_each_pair_force`] one pair at a time: the reference the batched path
    /// is held against.
    #[allow(clippy::too_many_arguments)]
    fn pair_forces_scalar(
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
        visit: impl FnMut(usize, f64, f64, f64),
    ) {
        match self.evaluation {
            Evaluation::Scalar => self.body_forces_scalar(pos, vel, mass, charge, visit),
            Evaluation::Batched => self.body_forces_batched(pos, vel, mass, charge, visit),
        }
    }

    /// [`Self::for_each_body_force`] a batch of particles at a time, visited in index order.
    fn body_forces_batched(
        &self,
        pos: (&[f64], &[f64]),
        vel: (&[f64], &[f64]),
        mass: &[f64],
        charge: &[f64],
        mut visit: impl FnMut(usize, f64, f64, f64),
    ) {
        let mut batch = BodyBatch {
            len: 0,
            index: [0; LANES],
            inputs: [0.0; layout::BODY_INPUTS as usize * LANES],
            regs: [0.0; MAX_REGISTERS * LANES],
            out: [0.0; 3 * LANES],
        };
        for k in 0..pos.0.len() {
            let lane = batch.len;
            batch.index[lane] = k;
            let mut put = |input: u32, value: f64| batch.inputs[input as usize * LANES + lane] = value;
            put(layout::PX, pos.0[k]);
            put(layout::PY, pos.1[k]);
            put(layout::VX, vel.0[k]);
            put(layout::VY, vel.1[k]);
            put(layout::MASS, mass[k]);
            put(layout::CHARGE, charge[k]);
            batch.len += 1;
            if batch.len == LANES {
                self.flush_bodies(&mut batch, &mut visit);
            }
        }
        self.flush_bodies(&mut batch, &mut visit);
    }

    /// Evaluate the gathered particles and visit each in index order, then empty the batch.
    fn flush_bodies(&self, batch: &mut BodyBatch, visit: &mut impl FnMut(usize, f64, f64, f64)) {
        let n = batch.len;
        if n == 0 {
            return;
        }
        self.program.eval_batch(n, &batch.inputs, &self.params, &mut batch.regs, &mut batch.out);
        for lane in 0..n {
            let k = batch.index[lane];
            let (a, b, c) = (batch.out[lane], batch.out[LANES + lane], batch.out[2 * LANES + lane]);
            match self.kind {
                UserLawKind::BodyForce => visit(k, a, b, 0.0),
                _ => visit(k, -b, -c, a),
            }
        }
        batch.len = 0;
    }

    /// [`Self::for_each_body_force`] one particle at a time: the batched path's reference.
    fn body_forces_scalar(
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

    fn is_user_law(&self) -> bool {
        true
    }

    fn explain_non_finite(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> Option<LawFault> {
        self.first_fault(store, ctx)
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

    fn origin(start: u32, end: u32) -> Option<lattice_expr::Origin> {
        Some(lattice_expr::Origin { start, end })
    }

    /// `U(r) = 1 / (r − 1)`, each operation at the byte range it would have in the source.
    fn pole_potential() -> UserLaw {
        let mut b = Builder::new(layout::PAIR_INPUTS, 0);
        b.at(origin(20, 21));
        let r = b.push(Op::Input(layout::R));
        b.at(origin(24, 25));
        let one = b.constant(1.0);
        b.at(origin(20, 25));
        let gap = b.push(Op::Sub(r, one));
        b.at(origin(10, 26));
        let u = b.push(Op::Div(one, gap));
        let program = b.finish(vec![u]).derivative(layout::R);
        UserLaw::new("pole", UserLawKind::PairPotential, program, vec![], Some(3.0), None)
    }

    fn store_of(positions: &[[f64; 2]]) -> ParticleStore {
        let mut store = ParticleStore::with_capacity(positions.len());
        for &p in positions {
            store.spawn(lattice_ir::ParticleSpec::at(p).with_mass(1.0)).unwrap();
        }
        store
    }

    fn list_for(store: &ParticleStore) -> crate::verlet::NeighborList {
        let mut list = crate::verlet::NeighborList::new(
            [-10.0, -10.0],
            [20.0, 20.0],
            [false; 2],
            3.0,
            None,
            store.len(),
            crate::verlet::Exclusions::none(),
        );
        list.update(store);
        list
    }

    #[test]
    fn a_pair_law_is_traced_to_the_pair_and_the_operation() {
        let law = pole_potential();
        // 0 and 2 are 2 apart; 1 and 2 are exactly 1 apart.
        let store = store_of(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        let list = list_for(&store);
        let ctx = ForceContext { neighbors: Some(&list), image: list.image() };
        let fault = law.explain_non_finite(&store, &ctx).expect("a pole at r = 1");
        assert_eq!(fault.law, "pole");
        assert!(matches!(fault.site, FaultSite::Pair(..)), "{fault:?}");
        let FaultSite::Pair(a, b) = fault.site else { unreachable!() };
        assert_eq!((a as i32 - b as i32).abs(), 1, "an adjacent pair, not 0 and 2");
        assert_eq!(fault.at.op, Op::Div(1, 2));
        assert_eq!(fault.at.origin, origin(10, 26));
        assert_eq!(fault.at.operands, [1.0, 0.0]);
        assert!(!fault.is_input());

        // Move the pair off the pole and nothing is reported.
        let healthy = store_of(&[[0.0, 0.0], [1.5, 0.0]]);
        let list = list_for(&healthy);
        let ctx = ForceContext { neighbors: Some(&list), image: list.image() };
        assert!(law.explain_non_finite(&healthy, &ctx).is_none());
    }

    #[test]
    fn a_body_law_is_traced_to_the_particle_and_a_bad_input_blames_the_input() {
        // F = (ln(x), 0).
        let mut b = Builder::new(layout::BODY_INPUTS, 0);
        b.at(origin(40, 45));
        let x = b.push(Op::Input(layout::PX));
        b.at(origin(37, 46));
        let ln = b.push(Op::Ln(x));
        let zero = b.constant(0.0);
        let law = UserLaw::new("pull", UserLawKind::BodyForce, b.finish(vec![ln, zero]), vec![], None, None);

        let mut store = ParticleStore::with_capacity(2);
        let first = store.spawn(lattice_ir::ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        store.spawn(lattice_ir::ParticleSpec::at([-2.0, 0.0]).with_mass(1.0)).unwrap();
        let ctx = ForceContext::open();
        let fault = law.explain_non_finite(&store, &ctx).unwrap();
        assert_eq!(fault.site, FaultSite::Particle(1));
        assert_eq!((fault.at.op, fault.at.origin), (Op::Ln(0), origin(37, 46)));
        assert_eq!(fault.at.operands, [-2.0]);
        assert!(!fault.is_input());

        // A position that is NaN already: the input is to blame, and no operation.
        assert!(store.set_position(first, [f64::NAN, 0.0]));
        let fault = law.explain_non_finite(&store, &ctx).unwrap();
        assert_eq!(fault.site, FaultSite::Particle(0));
        assert_eq!(fault.at.op, Op::Input(layout::PX));
        assert_eq!(fault.at.origin, origin(40, 45));
        assert!(fault.at.operands.is_empty());
        assert_eq!(fault.input.as_deref(), Some("a.position.x"));
    }

    /// A domain whose initial state is already NaN: asked to watch its laws, it explains
    /// the initial forces, blames the input, and reports the domain's own name. A domain
    /// that is told to stop watching forgets.
    #[test]
    fn a_domain_with_a_nan_initial_state_blames_the_input_when_asked_to_watch() {
        use lattice_ir::Domain;
        let mut b = Builder::new(layout::BODY_INPUTS, 0);
        b.at(origin(40, 50));
        let x = b.push(Op::Input(layout::PX));
        let zero = b.constant(0.0);
        let law = UserLaw::new("pull", UserLawKind::BodyForce, b.finish(vec![x, zero]), vec![], None, None);
        let mut domain = crate::ParticleDomain::new("bead", 2).with_force(law);
        domain.spawn(lattice_ir::ParticleSpec::at([1.0, 0.0]).with_mass(1.0)).unwrap();
        domain.spawn(lattice_ir::ParticleSpec::at([f64::NAN, 0.0]).with_mass(1.0)).unwrap();
        domain.initialize();
        assert!(domain.law_fault().is_none(), "not watching");

        assert!(domain.trace_laws(true), "it has a user law");
        let fault = domain.law_fault().expect("the second particle's position is NaN");
        assert_eq!(fault.domain, "bead");
        assert_eq!(fault.site, FaultSite::Particle(1));
        assert_eq!((fault.at.op, fault.at.origin), (Op::Input(layout::PX), origin(40, 50)));
        assert_eq!(fault.input.as_deref(), Some("a.position.x"));
        assert!(fault.at.operands.is_empty(), "no operation is blamed");

        domain.trace_laws(false);
        assert!(domain.law_fault().is_none());
    }

    #[test]
    fn layout_names_every_input() {
        assert_eq!(layout::input_name(true, layout::R), "r");
        assert_eq!(layout::input_name(true, layout::DY), "d.y");
        assert_eq!(layout::input_name(true, layout::PAIR_A + layout::VX), "a.velocity.x");
        assert_eq!(layout::input_name(true, layout::PAIR_B + layout::CHARGE), "b.charge");
        assert_eq!(layout::input_name(false, layout::MASS), "a.mass");
    }

    #[test]
    fn an_energy_shift_zeroes_the_energy_at_the_cutoff() {
        let mut law = spring_potential(3.0, 1.0);
        law.truncation = Some(Truncation::EnergyShift);
        let (u, du) = law.pair_potential(5.0);
        assert_eq!(law.truncated(u, du, 5.0, (law.energy_at_cutoff, law.slope_at_cutoff)).0, 0.0);
    }
}
