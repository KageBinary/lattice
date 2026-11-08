//! Compiled user-defined laws (spec §8.3): a scalar program, its interpreter, and its
//! derivative.
//!
//! The compiler checks a law's types and units and then lowers it to a [`Program`]: a
//! list of `f64` operations in single-assignment form, each reading earlier results.
//! Vectors have become pairs of scalars, truth values have become `0.0` and `1.0`, and
//! every unit has become a constant. The program knows nothing about physics. A domain
//! fills its inputs (a separation, two particles' members) and its parameters (the
//! values a use site bound) and reads its outputs (a force, an energy).
//!
//! # Optimizations never change a value
//!
//! The [`Builder`] removes repeated subexpressions and folds constant operands, and
//! that is all. It does not reassociate, does not fuse a multiply into an add, and does
//! not replace `x / y` with `x * (1/y)`. Each of those would change the last bits of a
//! result, and a law written in the same order as a built-in force must reproduce it
//! bit for bit. Removing a repeat is safe because the same operation on the same
//! operands gives the same bits. Folding is safe because it runs the very function the
//! interpreter would.
//!
//! # Derivatives are taken on the program
//!
//! [`Program::derivative`] is forward-mode differentiation as a program transform:
//! every operation gets a second operation computing its tangent, and the result is a
//! program returning both the value and its derivative with respect to one input. A
//! pair potential `U(r)` becomes a program for `U` and `dU/dr` together, so a force is
//! the exact derivative of the energy the author wrote, never a finite difference of it.
//! Comparisons have no derivative; `if` takes the derivative of the arm it chose,
//! which is right everywhere except at the switch itself.
//!
//! # Every operation remembers where it came from
//!
//! A [`Builder`] can be told the [`Origin`] of what it pushes next, a byte range in the
//! source the compiler read. A [`Program`] keeps one per operation, so a failure deep in
//! a law can be pointed at the sub-expression that caused it. Origins are metadata: they
//! take no part in what a program computes, so `==` and [`Program::fingerprint`]
//! ignore them, and two laws that compute the same thing are the same program wherever
//! they were written.
//!
//! # A non-finite value can be traced to the operation that made it
//!
//! [`Program::trace`] re-runs a program on one set of inputs and follows a non-finite
//! output back, through the values that carried it, to the operation that turned finite
//! operands into an infinity or a NaN, returning the operation, its [`Origin`] and the
//! operands it read. When two independent faults meet, the one computed first is reported. A
//! non-finite value a `Select` discarded is not blamed. The domains call it, after a run
//! has halted, on the inputs that produced the failure.
//!
//! # No allocation while evaluating
//!
//! [`Program::eval`] runs on a register array the caller owns. Laws are small, and
//! [`MAX_REGISTERS`] bounds them so a stack array always fits: a pair loop evaluates a
//! law once per pair, and spec NFR-001 allows no allocation there.
//!
//! # A batch is the same arithmetic, [`LANES`] at a time
//!
//! [`Program::eval_batch`] runs [`LANES`] independent sets of inputs through one pass of
//! the operations, so the dispatch on each operation is paid once for all of them. It
//! computes nothing differently: one macro supplies every operation's expression to both
//! the scalar and the batched interpreter, so each lane is bit-identical to
//! [`Program::eval`] on that lane's inputs. The one exception is a NaN's payload, which
//! Rust does not pin: where the scalar result is NaN the lane's is NaN too, and the
//! tests compare no more than that.

use std::collections::HashMap;

/// A register: the index of the operation that wrote it.
pub type Reg = u32;

/// The most operations a program may have, so its registers fit a stack array.
pub const MAX_REGISTERS: usize = 512;

/// How many sets of inputs [`Program::eval_batch`] evaluates in one pass: the register
/// file of a full program is `MAX_REGISTERS × LANES × 8 B = 32 KiB`, the size of a typical
/// L1 data cache, before the inputs, parameters and outputs. The choice is unmeasured here;
/// the particle-loop benchmark is what checks it.
pub const LANES: usize = 8;

/// A byte range in the law's source file, for the operation that came from it.
///
/// This crate stays independent of the syntax crate, so it holds the two offsets and no
/// more.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Origin {
    /// First byte.
    pub start: u32,
    /// One past the last byte.
    pub end: u32,
}

/// The first operation of a run that made a value non-finite: what [`Program::trace`]
/// finds.
#[derive(Clone, PartialEq, Debug)]
pub struct NonFiniteAt {
    /// The register the operation wrote.
    pub reg: Reg,
    /// The operation.
    pub op: Op,
    /// Where in the source it came from, when the compiler said.
    pub origin: Option<Origin>,
    /// The values it read, in the order [`Op::operands`] gives the registers. All finite;
    /// empty for an input, a parameter or a constant.
    pub operands: Vec<f64>,
    /// What it wrote: infinite or NaN.
    pub value: f64,
}

/// One operation. Each writes the register numbered by its position.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Op {
    /// A constant.
    Const(f64),
    /// The `n`th input.
    Input(u32),
    /// The `n`th parameter.
    Param(u32),
    /// `-x`
    Neg(Reg),
    /// `1` when `x` is `0`, else `0`.
    Not(Reg),
    /// `√x`
    Sqrt(Reg),
    /// `|x|`
    Abs(Reg),
    /// `eˣ`
    Exp(Reg),
    /// `ln x`
    Ln(Reg),
    /// `sin x`
    Sin(Reg),
    /// `cos x`
    Cos(Reg),
    /// `tan x`
    Tan(Reg),
    /// `erfc x`
    Erfc(Reg),
    /// `xⁿ` for a whole `n`.
    Powi(Reg, i32),
    /// `a + b`
    Add(Reg, Reg),
    /// `a − b`
    Sub(Reg, Reg),
    /// `a × b`
    Mul(Reg, Reg),
    /// `a ÷ b`
    Div(Reg, Reg),
    /// The smaller.
    Min(Reg, Reg),
    /// The larger.
    Max(Reg, Reg),
    /// `atan2(y, x)`.
    Atan2(Reg, Reg),
    /// `aᵇ` for a real `b`.
    Pow(Reg, Reg),
    /// `1` when `a < b`, else `0`.
    Lt(Reg, Reg),
    /// `1` when `a ≤ b`, else `0`.
    Le(Reg, Reg),
    /// `1` when `a = b`, else `0`.
    Eq(Reg, Reg),
    /// `1` when `a ≠ b`, else `0`.
    Ne(Reg, Reg),
    /// `1` when both are non-zero, else `0`.
    And(Reg, Reg),
    /// `1` when either is non-zero, else `0`.
    Or(Reg, Reg),
    /// `a` when `c` is non-zero, else `b`.
    Select(Reg, Reg, Reg),
}

impl Op {
    /// The registers this operation reads, in order.
    pub fn operands(self) -> impl Iterator<Item = Reg> {
        use Op::*;
        let (a, b, c) = match self {
            Const(_) | Input(_) | Param(_) => (None, None, None),
            Neg(x) | Not(x) | Sqrt(x) | Abs(x) | Exp(x) | Ln(x) | Sin(x) | Cos(x) | Tan(x) | Erfc(x)
            | Powi(x, _) => (Some(x), None, None),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Min(a, b) | Max(a, b) | Atan2(a, b) | Pow(a, b)
            | Lt(a, b) | Le(a, b) | Eq(a, b) | Ne(a, b) | And(a, b) | Or(a, b) => (Some(a), Some(b), None),
            Select(c, a, b) => (Some(c), Some(a), Some(b)),
        };
        [a, b, c].into_iter().flatten()
    }

    /// The same operation reading renumbered registers.
    fn remap(self, map: impl Fn(Reg) -> Reg) -> Op {
        use Op::*;
        match self {
            Const(_) | Input(_) | Param(_) => self,
            Neg(x) => Neg(map(x)),
            Not(x) => Not(map(x)),
            Sqrt(x) => Sqrt(map(x)),
            Abs(x) => Abs(map(x)),
            Exp(x) => Exp(map(x)),
            Ln(x) => Ln(map(x)),
            Sin(x) => Sin(map(x)),
            Cos(x) => Cos(map(x)),
            Tan(x) => Tan(map(x)),
            Erfc(x) => Erfc(map(x)),
            Powi(x, n) => Powi(map(x), n),
            Add(a, b) => Add(map(a), map(b)),
            Sub(a, b) => Sub(map(a), map(b)),
            Mul(a, b) => Mul(map(a), map(b)),
            Div(a, b) => Div(map(a), map(b)),
            Min(a, b) => Min(map(a), map(b)),
            Max(a, b) => Max(map(a), map(b)),
            Atan2(a, b) => Atan2(map(a), map(b)),
            Pow(a, b) => Pow(map(a), map(b)),
            Lt(a, b) => Lt(map(a), map(b)),
            Le(a, b) => Le(map(a), map(b)),
            Eq(a, b) => Eq(map(a), map(b)),
            Ne(a, b) => Ne(map(a), map(b)),
            And(a, b) => And(map(a), map(b)),
            Or(a, b) => Or(map(a), map(b)),
            Select(c, a, b) => Select(map(c), map(a), map(b)),
        }
    }

    /// A hashable identity: two operations with equal keys compute equal bits.
    fn key(self) -> (u8, u64, u64, u64) {
        use Op::*;
        let r = |x: Reg| u64::from(x);
        match self {
            Const(v) => (0, v.to_bits(), 0, 0),
            Input(n) => (1, u64::from(n), 0, 0),
            Param(n) => (2, u64::from(n), 0, 0),
            Neg(x) => (3, r(x), 0, 0),
            Not(x) => (4, r(x), 0, 0),
            Sqrt(x) => (5, r(x), 0, 0),
            Abs(x) => (6, r(x), 0, 0),
            Exp(x) => (7, r(x), 0, 0),
            Ln(x) => (8, r(x), 0, 0),
            Sin(x) => (9, r(x), 0, 0),
            Cos(x) => (10, r(x), 0, 0),
            Tan(x) => (11, r(x), 0, 0),
            Erfc(x) => (12, r(x), 0, 0),
            Powi(x, n) => (13, r(x), n as u32 as u64, 0),
            Add(a, b) => (14, r(a), r(b), 0),
            Sub(a, b) => (15, r(a), r(b), 0),
            Mul(a, b) => (16, r(a), r(b), 0),
            Div(a, b) => (17, r(a), r(b), 0),
            Min(a, b) => (18, r(a), r(b), 0),
            Max(a, b) => (19, r(a), r(b), 0),
            Atan2(a, b) => (20, r(a), r(b), 0),
            Pow(a, b) => (21, r(a), r(b), 0),
            Lt(a, b) => (22, r(a), r(b), 0),
            Le(a, b) => (23, r(a), r(b), 0),
            Eq(a, b) => (24, r(a), r(b), 0),
            Ne(a, b) => (25, r(a), r(b), 0),
            And(a, b) => (26, r(a), r(b), 0),
            Or(a, b) => (27, r(a), r(b), 0),
            Select(c, a, b) => (28, r(c), r(a), r(b)),
        }
    }
}

/// What every operation computes, written once.
///
/// The scalar interpreter ([`apply`]) and the batched one ([`Program::eval_batch`]) both
/// take their arms from here, so an operation cannot be added to one and not the other
/// (spec M6.1d R1), and a lane computes the exact expression the scalar does: the same
/// function on the same operands, so the same bits.
///
/// `$wrap` names a macro that receives `$g $input $param => <expression>` for each arm.
/// `$g(r)` reads register `r`, `$input(n)` input `n`, `$param(n)` parameter `n`; the
/// caller's macro decides what those read (one value, or one lane of a batch) and what
/// to do with the expression's value. The names come from the caller so that its
/// bindings and the expression's uses are the same identifiers.
macro_rules! op_match {
    ($op:expr, $wrap:ident, $g:ident, $input:ident, $param:ident) => {{
        use Op::*;
        match $op {
            Const(v) => $wrap! { $g $input $param => v },
            Input(n) => $wrap! { $g $input $param => $input(n) },
            Param(n) => $wrap! { $g $input $param => $param(n) },
            Neg(x) => $wrap! { $g $input $param => -$g(x) },
            Not(x) => $wrap! { $g $input $param => truth($g(x) == 0.0) },
            Sqrt(x) => $wrap! { $g $input $param => $g(x).sqrt() },
            Abs(x) => $wrap! { $g $input $param => $g(x).abs() },
            Exp(x) => $wrap! { $g $input $param => $g(x).exp() },
            Ln(x) => $wrap! { $g $input $param => $g(x).ln() },
            Sin(x) => $wrap! { $g $input $param => $g(x).sin() },
            Cos(x) => $wrap! { $g $input $param => $g(x).cos() },
            Tan(x) => $wrap! { $g $input $param => $g(x).tan() },
            Erfc(x) => $wrap! { $g $input $param => erfc($g(x)) },
            Powi(x, n) => $wrap! { $g $input $param => $g(x).powi(n) },
            Add(a, b) => $wrap! { $g $input $param => $g(a) + $g(b) },
            Sub(a, b) => $wrap! { $g $input $param => $g(a) - $g(b) },
            Mul(a, b) => $wrap! { $g $input $param => $g(a) * $g(b) },
            Div(a, b) => $wrap! { $g $input $param => $g(a) / $g(b) },
            Min(a, b) => $wrap! { $g $input $param => min($g(a), $g(b)) },
            Max(a, b) => $wrap! { $g $input $param => max($g(a), $g(b)) },
            Atan2(y, x) => $wrap! { $g $input $param => $g(y).atan2($g(x)) },
            Pow(a, b) => $wrap! { $g $input $param => $g(a).powf($g(b)) },
            Lt(a, b) => $wrap! { $g $input $param => truth($g(a) < $g(b)) },
            Le(a, b) => $wrap! { $g $input $param => truth($g(a) <= $g(b)) },
            Eq(a, b) => $wrap! { $g $input $param => truth($g(a) == $g(b)) },
            Ne(a, b) => $wrap! { $g $input $param => truth($g(a) != $g(b)) },
            And(a, b) => $wrap! { $g $input $param => truth($g(a) != 0.0 && $g(b) != 0.0) },
            Or(a, b) => $wrap! { $g $input $param => truth($g(a) != 0.0 || $g(b) != 0.0) },
            Select(c, a, b) => $wrap! { $g $input $param => if $g(c) != 0.0 { $g(a) } else { $g(b) } },
        }
    }};
}

/// The smaller of two values, with `-0.0` below `+0.0`.
///
/// `f64::min` may return either zero when the two compare equal, and the scalar and batched
/// interpreters are separate code, so the choice could differ between them. Here equal
/// operands are picked by sign, which pins the bits. Anything else is `f64::min`, so a NaN
/// operand is handled as it always was.
#[inline]
fn min(a: f64, b: f64) -> f64 {
    if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else {
        a.min(b)
    }
}

/// The larger of two values, with `+0.0` above `-0.0`: [`min`]'s counterpart.
#[inline]
fn max(a: f64, b: f64) -> f64 {
    if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else {
        a.max(b)
    }
}

/// A truth value as the interpreter holds it.
#[inline]
fn truth(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Apply one operation to register values: the single definition of what each
/// operation computes, shared by the interpreter and constant folding.
#[inline]
fn apply(op: Op, regs: &[f64], inputs: &[f64], params: &[f64]) -> f64 {
    let g = |x: Reg| regs[x as usize];
    let input = |n: u32| inputs[n as usize];
    let param = |n: u32| params[n as usize];
    macro_rules! value {
        ($g:ident $input:ident $param:ident => $e:expr) => {
            $e
        };
    }
    op_match!(op, value, g, input, param)
}

/// Builds a [`Program`], removing repeated operations and folding constants.
#[derive(Debug)]
pub struct Builder {
    ops: Vec<Op>,
    /// Where each operation came from, parallel to `ops`.
    origins: Vec<Option<Origin>>,
    /// The origin given to what is pushed next.
    origin: Option<Origin>,
    /// The constant value of each register, when it has one.
    constant: Vec<Option<f64>>,
    seen: HashMap<(u8, u64, u64, u64), Reg>,
    inputs: u32,
    params: u32,
}

impl Builder {
    /// A builder for a program of `inputs` inputs and `params` parameters.
    pub fn new(inputs: u32, params: u32) -> Builder {
        Builder {
            ops: Vec::new(),
            origins: Vec::new(),
            origin: None,
            constant: Vec::new(),
            seen: HashMap::new(),
            inputs,
            params,
        }
    }

    /// Set the origin given to operations pushed from now on.
    ///
    /// An operation removed as a repeat keeps the origin of the first, and a folded
    /// constant has the origin of the operation it replaced.
    pub fn at(&mut self, origin: Option<Origin>) {
        self.origin = origin;
    }

    /// Append an operation, or return the register that already holds the same value.
    ///
    /// An operation whose operands are all constants is evaluated now, by the same
    /// function the interpreter uses, and becomes a constant.
    ///
    /// # Panics
    ///
    /// If an input or parameter is out of range, or an operand names a register not yet
    /// written: both are compiler bugs, not user errors.
    pub fn push(&mut self, op: Op) -> Reg {
        match op {
            Op::Input(n) => assert!(n < self.inputs, "input {n} out of range"),
            Op::Param(n) => assert!(n < self.params, "parameter {n} out of range"),
            _ => {}
        }
        for operand in op.operands() {
            assert!((operand as usize) < self.ops.len(), "operand {operand} not yet written");
        }
        let foldable = !matches!(op, Op::Const(_) | Op::Input(_) | Op::Param(_))
            && op.operands().all(|r| self.constant[r as usize].is_some());
        let op = if foldable {
            let values: Vec<f64> = self.constant.iter().map(|c| c.unwrap_or(0.0)).collect();
            Op::Const(apply(op, &values, &[], &[]))
        } else {
            op
        };
        if let Some(&existing) = self.seen.get(&op.key()) {
            return existing;
        }
        let reg = self.ops.len() as Reg;
        self.ops.push(op);
        self.origins.push(self.origin);
        self.constant.push(match op {
            Op::Const(v) => Some(v),
            _ => None,
        });
        self.seen.insert(op.key(), reg);
        reg
    }

    /// A constant.
    pub fn constant(&mut self, value: f64) -> Reg {
        self.push(Op::Const(value))
    }

    /// The constant value a register holds, if it is one.
    pub fn constant_value(&self, reg: Reg) -> Option<f64> {
        self.constant.get(reg as usize).copied().flatten()
    }

    /// Finish with the given outputs, dropping every operation no output depends on.
    pub fn finish(self, outputs: Vec<Reg>) -> Program {
        let mut live = vec![false; self.ops.len()];
        for &out in &outputs {
            live[out as usize] = true;
        }
        for index in (0..self.ops.len()).rev() {
            if live[index] {
                for operand in self.ops[index].operands() {
                    live[operand as usize] = true;
                }
            }
        }
        let mut renumber = vec![Reg::MAX; self.ops.len()];
        let mut ops = Vec::new();
        let mut origins = Vec::new();
        for (index, op) in self.ops.iter().enumerate() {
            if live[index] {
                renumber[index] = ops.len() as Reg;
                ops.push(op.remap(|r| renumber[r as usize]));
                origins.push(self.origins[index]);
            }
        }
        let outputs = outputs.iter().map(|&r| renumber[r as usize]).collect();
        Program { ops, origins, outputs, inputs: self.inputs, params: self.params }
    }
}

/// A compiled law: operations in single-assignment form and the registers it returns.
#[derive(Clone, Debug)]
pub struct Program {
    ops: Vec<Op>,
    /// Where each operation came from, parallel to `ops`.
    origins: Vec<Option<Origin>>,
    outputs: Vec<Reg>,
    inputs: u32,
    params: u32,
}

/// Programs are equal when they compute the same thing; where each operation was
/// written is not part of that.
impl PartialEq for Program {
    fn eq(&self, other: &Program) -> bool {
        self.ops == other.ops
            && self.outputs == other.outputs
            && self.inputs == other.inputs
            && self.params == other.params
    }
}

impl Program {
    /// The operations, in order.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The source range operation `reg` came from, when the compiler said.
    pub fn origin(&self, reg: Reg) -> Option<Origin> {
        self.origins.get(reg as usize).copied().flatten()
    }

    /// The registers it returns, in order.
    pub fn outputs(&self) -> &[Reg] {
        &self.outputs
    }

    /// How many values it returns.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// How many inputs it reads.
    pub fn input_count(&self) -> usize {
        self.inputs as usize
    }

    /// How many parameters it reads.
    pub fn param_count(&self) -> usize {
        self.params as usize
    }

    /// True when it fits a stack register file of [`MAX_REGISTERS`].
    pub fn fits(&self) -> bool {
        self.ops.len() <= MAX_REGISTERS
    }

    /// Evaluate into `out`, using `regs` as scratch.
    ///
    /// # Panics
    ///
    /// If `regs` is shorter than the program, `out` shorter than its outputs, or the
    /// inputs and parameters fewer than it reads.
    #[inline]
    pub fn eval(&self, inputs: &[f64], params: &[f64], regs: &mut [f64], out: &mut [f64]) {
        for (index, op) in self.ops.iter().enumerate() {
            regs[index] = apply(*op, regs, inputs, params);
        }
        for (slot, &reg) in out.iter_mut().zip(&self.outputs) {
            *slot = regs[reg as usize];
        }
    }

    /// Evaluate [`LANES`] independent sets of inputs in one pass.
    ///
    /// `inputs[k * LANES + l]` is input `k` of lane `l`, and `out[o * LANES + l]` is output
    /// `o` of lane `l`. Every lane shares `params`. `regs` is scratch of at least
    /// `ops.len() * LANES`. Lanes from `active` up are computed but their outputs are
    /// meaningless: each reads the last active lane's inputs, so that padding cannot
    /// raise anything the real lanes would not, and the caller discards them.
    ///
    /// The op loop is outside and the lane loop inside, so one `match` covers `LANES`
    /// values. Each lane runs the expression [`apply`] runs, from the same macro, so
    /// lane `l` is bit-identical to [`Program::eval`] on lane `l`'s inputs, except that a
    /// NaN's payload is not pinned by Rust: where the scalar result is NaN, so is the lane's.
    ///
    /// # Panics
    ///
    /// If `active` is not in `1..=LANES`, or a slice is shorter than the program needs.
    #[inline]
    pub fn eval_batch(&self, active: usize, inputs: &[f64], params: &[f64], regs: &mut [f64], out: &mut [f64]) {
        assert!((1..=LANES).contains(&active), "active lanes {active} not in 1..={LANES}");
        assert!(
            out.len() >= self.outputs.len() * LANES,
            "out holds {} values, and {} outputs of {LANES} lanes need {}",
            out.len(),
            self.outputs.len(),
            self.outputs.len() * LANES
        );
        for (index, op) in self.ops.iter().enumerate() {
            let base = index * LANES;
            macro_rules! lanes {
                ($g:ident $input:ident $param:ident => $e:expr) => {
                    for lane in 0..LANES {
                        // An arm reads some of these and not others.
                        #[allow(unused_variables)]
                        let $g = |x: Reg| regs[x as usize * LANES + lane];
                        #[allow(unused_variables)]
                        let $input = |n: u32| inputs[n as usize * LANES + lane.min(active - 1)];
                        #[allow(unused_variables)]
                        let $param = |n: u32| params[n as usize];
                        let value: f64 = $e;
                        regs[base + lane] = value;
                    }
                };
            }
            op_match!(*op, lanes, g, input, param);
        }
        for (slot, &reg) in out.chunks_exact_mut(LANES).zip(&self.outputs) {
            slot.copy_from_slice(&regs[reg as usize * LANES..][..LANES]);
        }
    }

    /// Evaluate on one set of inputs and find the operation that made an output
    /// non-finite: [`Program::trace_outputs`] over every output.
    pub fn trace(&self, inputs: &[f64], params: &[f64]) -> Option<NonFiniteAt> {
        self.trace_outputs(inputs, params, 0..self.outputs.len())
    }

    /// Evaluate on one set of inputs and find the operation that made one of the outputs
    /// in `outputs` (indices into the program's outputs) non-finite.
    ///
    /// Every value is computed first. The trace then takes the *cone* of each non-finite
    /// output: the operations whose values carried the badness to it. At an operation
    /// that is every non-finite operand; at a `Select`, its condition if that is
    /// non-finite and the arm it took, so a non-finite value on the arm it discarded is
    /// not in the cone. The culprits are the operations in the cone whose operands were
    /// all finite, and the one with the lowest register is reported: when two
    /// independent faults meet at an operation, the one the program computed first. An
    /// input that arrives non-finite is a culprit too, reported as itself, an `Op::Input`
    /// with no operands, with no operation blamed for it. `None` means none of
    /// `outputs` is non-finite.
    ///
    /// It allocates and keeps every operand: it is for explaining a failure after the
    /// fact, never for a loop.
    ///
    /// # Panics
    ///
    /// As [`Program::eval`] does, if the inputs or parameters are fewer than it reads;
    /// and if `outputs` reaches past the program's outputs.
    pub fn trace_outputs(
        &self,
        inputs: &[f64],
        params: &[f64],
        outputs: core::ops::Range<usize>,
    ) -> Option<NonFiniteAt> {
        let mut regs: Vec<f64> = Vec::with_capacity(self.ops.len());
        for &op in &self.ops {
            // `apply` reads only registers before this one, which `regs` holds.
            let value = apply(op, &regs, inputs, params);
            regs.push(value);
        }
        let bad = |r: Reg| !regs[r as usize].is_finite();
        let mut pending: Vec<Reg> = self.outputs[outputs].iter().copied().filter(|&r| bad(r)).collect();
        let mut seen = vec![false; self.ops.len()];
        let mut culprit: Option<Reg> = None;
        while let Some(reg) = pending.pop() {
            if std::mem::replace(&mut seen[reg as usize], true) {
                continue;
            }
            let op = self.ops[reg as usize];
            let before = pending.len();
            match op {
                Op::Select(c, a, b) => {
                    if bad(c) {
                        pending.push(c);
                    }
                    let taken = if regs[c as usize] != 0.0 { a } else { b };
                    if bad(taken) {
                        pending.push(taken);
                    }
                }
                _ => pending.extend(op.operands().filter(|&r| bad(r))),
            }
            if pending.len() == before {
                culprit = Some(culprit.map_or(reg, |lowest| lowest.min(reg)));
            }
        }
        let reg = culprit?;
        let op = self.ops[reg as usize];
        let operands = op.operands().map(|r| regs[r as usize]).collect();
        Some(NonFiniteAt { reg, op, origin: self.origin(reg), operands, value: regs[reg as usize] })
    }

    /// Evaluate, allocating: for compile-time sampling and tests, never a hot loop.
    pub fn eval_to_vec(&self, inputs: &[f64], params: &[f64]) -> Vec<f64> {
        let mut regs = vec![0.0; self.ops.len()];
        let mut out = vec![0.0; self.outputs.len()];
        self.eval(inputs, params, &mut regs, &mut out);
        out
    }

    /// A program returning this one's outputs followed by each output's derivative
    /// with respect to input `wrt`.
    pub fn derivative(&self, wrt: u32) -> Program {
        self.gradient(&[wrt])
    }

    /// A program returning this one's outputs, then their derivatives with respect to
    /// `wrt[0]`, then with respect to `wrt[1]`, and so on. The values are computed once
    /// and shared by every derivative.
    pub fn gradient(&self, wrt: &[u32]) -> Program {
        let mut b = Builder::new(self.inputs, self.params);
        let mut value: Vec<Reg> = Vec::with_capacity(self.ops.len());
        let mut tangents: Vec<Vec<Tangent>> = vec![Vec::with_capacity(self.ops.len()); wrt.len()];
        for (index, &op) in self.ops.iter().enumerate() {
            // Everything this operation's derivative needs comes from the same source.
            b.at(self.origins[index]);
            let mapped = op.remap(|r| value[r as usize]);
            let reg = b.push(mapped);
            for (tangent, &input) in tangents.iter_mut().zip(wrt) {
                let t = tangent_of(&mut b, op, reg, &value, tangent, input);
                tangent.push(t);
            }
            value.push(reg);
        }
        let mut outputs: Vec<Reg> = self.outputs.iter().map(|&r| value[r as usize]).collect();
        for tangent in &tangents {
            for &r in &self.outputs {
                b.at(self.origins[r as usize]);
                outputs.push(tangent[r as usize].materialize(&mut b));
            }
        }
        b.finish(outputs)
    }

    /// A stable fingerprint of what the program computes, for caches and artifacts.
    /// Origins are not part of it.
    ///
    /// Two programs with the same operations in the same order have the same hash on
    /// every machine: FNV-1a over the operations' keys, not `std`'s randomized hasher.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            for byte in x.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        mix(u64::from(self.inputs));
        mix(u64::from(self.params));
        for op in &self.ops {
            let (tag, a, b, c) = op.key();
            mix(u64::from(tag));
            mix(a);
            mix(b);
            mix(c);
        }
        for &out in &self.outputs {
            mix(u64::from(out));
        }
        h
    }
}

/// A derivative during differentiation: known to be zero or one, or in a register.
/// Keeping the trivial cases symbolic means a constant never produces a `× 0` or a
/// `× 1` that the program would then have to compute.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Tangent {
    Zero,
    One,
    In(Reg),
}

impl Tangent {
    fn materialize(self, b: &mut Builder) -> Reg {
        match self {
            Tangent::Zero => b.constant(0.0),
            Tangent::One => b.constant(1.0),
            Tangent::In(r) => r,
        }
    }

    /// `self × x`.
    fn times(self, b: &mut Builder, x: Reg) -> Tangent {
        match self {
            Tangent::Zero => Tangent::Zero,
            Tangent::One => Tangent::In(x),
            Tangent::In(r) => Tangent::In(b.push(Op::Mul(r, x))),
        }
    }

    /// `self ÷ x`.
    fn over(self, b: &mut Builder, x: Reg) -> Tangent {
        match self {
            Tangent::Zero => Tangent::Zero,
            Tangent::One => {
                let one = b.constant(1.0);
                Tangent::In(b.push(Op::Div(one, x)))
            }
            Tangent::In(r) => Tangent::In(b.push(Op::Div(r, x))),
        }
    }

    fn neg(self, b: &mut Builder) -> Tangent {
        match self {
            Tangent::Zero => Tangent::Zero,
            other => {
                let r = other.materialize(b);
                Tangent::In(b.push(Op::Neg(r)))
            }
        }
    }

    fn plus(self, b: &mut Builder, other: Tangent) -> Tangent {
        match (self, other) {
            (Tangent::Zero, t) | (t, Tangent::Zero) => t,
            (x, y) => {
                let (x, y) = (x.materialize(b), y.materialize(b));
                Tangent::In(b.push(Op::Add(x, y)))
            }
        }
    }

    fn minus(self, b: &mut Builder, other: Tangent) -> Tangent {
        match (self, other) {
            (t, Tangent::Zero) => t,
            (Tangent::Zero, t) => t.neg(b),
            (x, y) => {
                let (x, y) = (x.materialize(b), y.materialize(b));
                Tangent::In(b.push(Op::Sub(x, y)))
            }
        }
    }

    /// `if c { self } else { other }`.
    fn select(self, b: &mut Builder, c: Reg, other: Tangent) -> Tangent {
        if self == Tangent::Zero && other == Tangent::Zero {
            return Tangent::Zero;
        }
        let (x, y) = (self.materialize(b), other.materialize(b));
        Tangent::In(b.push(Op::Select(c, x, y)))
    }
}

/// The tangent of one operation, given the tangents of its operands. `reg` is the
/// operation's own value in the new program; `value` maps old registers to new.
fn tangent_of(b: &mut Builder, op: Op, reg: Reg, value: &[Reg], tangent: &[Tangent], wrt: u32) -> Tangent {
    use Op::*;
    let v = |x: Reg| value[x as usize];
    let t = |x: Reg| tangent[x as usize];
    match op {
        Const(_) | Param(_) => Tangent::Zero,
        Input(n) => {
            if n == wrt {
                Tangent::One
            } else {
                Tangent::Zero
            }
        }
        Not(_) | Lt(..) | Le(..) | Eq(..) | Ne(..) | And(..) | Or(..) => Tangent::Zero,
        Neg(x) => t(x).neg(b),
        Add(x, y) => t(x).plus(b, t(y)),
        Sub(x, y) => t(x).minus(b, t(y)),
        // d(xy) = x'y + xy'.
        Mul(x, y) => {
            let left = t(x).times(b, v(y));
            let right = t(y).times(b, v(x));
            left.plus(b, right)
        }
        // d(x/y) = (x' − (x/y) y') / y.
        Div(x, y) => {
            let scaled = t(y).times(b, reg);
            let numerator = t(x).minus(b, scaled);
            numerator.over(b, v(y))
        }
        // d(xⁿ) = n xⁿ⁻¹ x'.
        Powi(x, n) => match n {
            0 => Tangent::Zero,
            1 => t(x),
            _ => {
                if t(x) == Tangent::Zero {
                    return Tangent::Zero;
                }
                let n_reg = b.constant(f64::from(n));
                let lower = b.push(Powi(v(x), n - 1));
                let factor = b.push(Mul(n_reg, lower));
                t(x).times(b, factor)
            }
        },
        // d√x = x' / (2√x).
        Sqrt(x) => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let two = b.constant(2.0);
            let twice = b.push(Mul(two, reg));
            t(x).over(b, twice)
        }
        Abs(x) => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let zero = b.constant(0.0);
            let negative = b.push(Lt(v(x), zero));
            let flipped = t(x).neg(b);
            flipped.select(b, negative, t(x))
        }
        // `min` returns the first operand when it is the smaller.
        Min(x, y) => {
            let first = b.push(Lt(v(x), v(y)));
            t(x).select(b, first, t(y))
        }
        Max(x, y) => {
            let first = b.push(Lt(v(y), v(x)));
            t(x).select(b, first, t(y))
        }
        Exp(x) => t(x).times(b, reg),
        Ln(x) => t(x).over(b, v(x)),
        Sin(x) => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let cos = b.push(Cos(v(x)));
            t(x).times(b, cos)
        }
        Cos(x) => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let sin = b.push(Sin(v(x)));
            t(x).times(b, sin).neg(b)
        }
        // d tan x = x' (1 + tan² x).
        Tan(x) => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let one = b.constant(1.0);
            let square = b.push(Mul(reg, reg));
            let factor = b.push(Add(one, square));
            t(x).times(b, factor)
        }
        // d atan2(y, x) = (x y' − y x') / (x² + y²).
        Atan2(y, x) => {
            if t(x) == Tangent::Zero && t(y) == Tangent::Zero {
                return Tangent::Zero;
            }
            let first = t(y).times(b, v(x));
            let second = t(x).times(b, v(y));
            let numerator = first.minus(b, second);
            let xx = b.push(Mul(v(x), v(x)));
            let yy = b.push(Mul(v(y), v(y)));
            let denominator = b.push(Add(xx, yy));
            numerator.over(b, denominator)
        }
        // d erfc x = −(2/√π) e^{−x²} x'.
        Erfc(x) => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let square = b.push(Mul(v(x), v(x)));
            let negative = b.push(Neg(square));
            let gauss = b.push(Exp(negative));
            let scale = b.constant(-core::f64::consts::FRAC_2_SQRT_PI);
            let factor = b.push(Mul(scale, gauss));
            t(x).times(b, factor)
        }
        // A constant exponent: d(xʸ) = y xʸ⁻¹ x', which is finite at x = 0 for y ≥ 1,
        // where the general form below divides by x and returns ∞ · 0.
        Pow(x, y) if t(y) == Tangent::Zero => {
            if t(x) == Tangent::Zero {
                return Tangent::Zero;
            }
            let one = b.constant(1.0);
            let lower_exponent = b.push(Sub(v(y), one));
            let lower = b.push(Pow(v(x), lower_exponent));
            let factor = b.push(Mul(v(y), lower));
            t(x).times(b, factor)
        }
        // d(xʸ) = xʸ (y' ln x + y x'/x).
        Pow(x, y) => {
            let from_exponent = if t(y) == Tangent::Zero {
                Tangent::Zero
            } else {
                let ln = b.push(Ln(v(x)));
                t(y).times(b, ln)
            };
            let from_base = if t(x) == Tangent::Zero {
                Tangent::Zero
            } else {
                let ratio = b.push(Div(v(y), v(x)));
                t(x).times(b, ratio)
            };
            from_exponent.plus(b, from_base).times(b, reg)
        }
        Select(c, x, y) => t(x).select(b, v(c), t(y)),
    }
}

/// `2/√π`.
const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;

/// The complementary error function, `1 − erf(x)`, to a relative error of a few parts
/// in 10¹³ over the whole real line.
///
/// Written here because the standard library has none and §24.1 keeps dependencies
/// out of the hot loop. Two forms, each where it is accurate:
///
/// - `x < 2`: `erf(x) = (2/√π) e^{−x²} Σ 2ⁿ x^{2n+1} / (1·3·…·(2n+1))`, whose terms are
///   all positive, so nothing cancels until the final `1 − erf`, which at `x = 2` costs
///   about two digits;
/// - `x ≥ 2`: the continued fraction `erfc(x) = (e^{−x²}/√π) / (x + ½/(x + 1/(x + 3/2/(x + …))))`,
///   evaluated from a fixed depth backwards, which at `x = 2` has converged to
///   round-off by sixty levels.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        let x2 = x * x;
        let (mut term, mut sum) = (x, x);
        let mut n = 0u32;
        while term > 1e-17 * sum {
            n += 1;
            term *= 2.0 * x2 / f64::from(2 * n + 1);
            sum += term;
        }
        return 1.0 - TWO_OVER_SQRT_PI * (-x2).exp() * sum;
    }
    if x > 27.0 {
        // e^{−x²} underflows.
        return 0.0;
    }
    let mut tail = x;
    for n in (1..=60u32).rev() {
        tail = x + 0.5 * f64::from(n) / tail;
    }
    (-x * x).exp() / (core::f64::consts::PI.sqrt() * tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `U(r) = 4ε((σ/r)¹² − (σ/r)⁶)` with ε and σ as parameters and r as input 0.
    fn lennard_jones() -> Program {
        let mut b = Builder::new(1, 2);
        let r = b.push(Op::Input(0));
        let eps = b.push(Op::Param(0));
        let sigma = b.push(Op::Param(1));
        let s = b.push(Op::Div(sigma, r));
        let s6 = b.push(Op::Powi(s, 6));
        let s12 = b.push(Op::Mul(s6, s6));
        let diff = b.push(Op::Sub(s12, s6));
        let four = b.constant(4.0);
        let k = b.push(Op::Mul(four, eps));
        let u = b.push(Op::Mul(k, diff));
        b.finish(vec![u])
    }

    #[test]
    fn a_program_evaluates_what_was_written() {
        let program = lennard_jones();
        let u = program.eval_to_vec(&[2f64.powf(1.0 / 6.0)], &[1.0, 1.0])[0];
        assert!((u + 1.0).abs() < 1e-15, "the minimum is −ε, found {u}");
    }

    /// The derivative program is exact, up to round-off, against the analytic
    /// `U′(r) = (24ε/r)(s⁶ − 2s¹²)` at points across the well.
    #[test]
    fn the_derivative_is_the_analytic_one() {
        let d = lennard_jones().derivative(0);
        assert_eq!(d.output_count(), 2);
        for r in [0.9, 1.0, 1.12, 1.5, 2.5] {
            let out = d.eval_to_vec(&[r], &[1.3, 0.8]);
            let s6 = (0.8f64 / r).powi(6);
            let exact = 24.0 * 1.3 / r * (s6 - 2.0 * s6 * s6);
            assert!(((out[1] - exact) / exact).abs() < 1e-13, "r = {r}: {} against {exact}", out[1]);
        }
    }

    /// Every operation's derivative against a central difference, whose error is
    /// O(h²) with h = 1e-5: a relative agreement of 1e-7 leaves room for that and
    /// nothing else.
    #[test]
    fn every_operation_differentiates_correctly() {
        let unary: [fn(&mut Builder, Reg) -> Reg; 10] = [
            |b, x| b.push(Op::Neg(x)),
            |b, x| b.push(Op::Sqrt(x)),
            |b, x| b.push(Op::Abs(x)),
            |b, x| b.push(Op::Exp(x)),
            |b, x| b.push(Op::Ln(x)),
            |b, x| b.push(Op::Sin(x)),
            |b, x| b.push(Op::Cos(x)),
            |b, x| b.push(Op::Tan(x)),
            |b, x| b.push(Op::Erfc(x)),
            |b, x| b.push(Op::Powi(x, -3)),
        ];
        let binary: [fn(&mut Builder, Reg, Reg) -> Reg; 7] = [
            |b, x, y| b.push(Op::Mul(x, y)),
            |b, x, y| b.push(Op::Div(x, y)),
            |b, x, y| b.push(Op::Min(x, y)),
            |b, x, y| b.push(Op::Max(x, y)),
            |b, x, y| b.push(Op::Atan2(x, y)),
            |b, x, y| b.push(Op::Pow(x, y)),
            |b, x, y| {
                let c = b.push(Op::Lt(x, y));
                b.push(Op::Select(c, x, y))
            },
        ];
        let check = |program: &Program, x: f64, other: f64| {
            let d = program.derivative(0);
            let at = |x: f64| program.eval_to_vec(&[x, other], &[])[0];
            let h = 1e-5;
            let numeric = (at(x + h) - at(x - h)) / (2.0 * h);
            let exact = d.eval_to_vec(&[x, other], &[])[1];
            assert!((exact - numeric).abs() <= 1e-7 * (1.0 + numeric.abs()), "{program:?}: {exact} against {numeric}");
        };
        for f in unary {
            let mut b = Builder::new(2, 0);
            let x = b.push(Op::Input(0));
            let y = f(&mut b, x);
            check(&b.finish(vec![y]), 0.7, 0.0);
        }
        for f in binary {
            for (x, other) in [(0.7, 1.9), (1.9, 0.7)] {
                let mut b = Builder::new(2, 0);
                let (xr, yr) = (b.push(Op::Input(0)), b.push(Op::Input(1)));
                let z = f(&mut b, xr, yr);
                check(&b.finish(vec![z]), x, other);
                // And with respect to the second operand, through the first input.
                let mut b = Builder::new(2, 0);
                let (xr, yr) = (b.push(Op::Input(0)), b.push(Op::Input(1)));
                let z = f(&mut b, yr, xr);
                check(&b.finish(vec![z]), x, other);
            }
        }
    }

    /// `pow(x, 2.5)` at `x = 0` has derivative 0, not the `∞ · 0` the general formula
    /// would give.
    #[test]
    fn a_real_power_differentiates_at_zero() {
        let mut b = Builder::new(1, 0);
        let x = b.push(Op::Input(0));
        let exponent = b.constant(2.5);
        let y = b.push(Op::Pow(x, exponent));
        let d = b.finish(vec![y]).derivative(0);
        assert_eq!(d.eval_to_vec(&[0.0], &[]), [0.0, 0.0]);
        let at = d.eval_to_vec(&[4.0], &[]);
        assert!((at[1] - 2.5 * 4f64.powf(1.5)).abs() < 1e-12, "{at:?}");
    }

    #[test]
    fn repeats_are_computed_once_and_constants_fold() {
        let mut b = Builder::new(1, 0);
        let x = b.push(Op::Input(0));
        let n1 = b.push(Op::Sqrt(x));
        let n2 = b.push(Op::Sqrt(x));
        assert_eq!(n1, n2, "the same operation on the same operand is one register");
        let two = b.constant(2.0);
        let three = b.constant(3.0);
        let six = b.push(Op::Mul(two, three));
        assert_eq!(b.constant_value(six), Some(6.0));
        let out = b.push(Op::Mul(n1, six));
        let program = b.finish(vec![out]);
        // Input, sqrt, the folded 6 and the product; 2 and 3 are dead.
        assert_eq!(program.ops().len(), 4, "{:?}", program.ops());
    }

    /// Folding and repeat removal must not change a bit: the optimized program and a
    /// direct evaluation of the same arithmetic agree exactly.
    #[test]
    fn optimization_changes_no_bits() {
        let program = lennard_jones();
        for r in [0.95, 1.07, 1.3, 2.2] {
            let (eps, sigma): (f64, f64) = (0.997, 3.405);
            let s6 = (sigma / r).powi(6);
            let direct = 4.0 * eps * (s6 * s6 - s6);
            let interpreted = program.eval_to_vec(&[r], &[eps, sigma])[0];
            assert_eq!(interpreted.to_bits(), direct.to_bits(), "r = {r}");
        }
    }

    #[test]
    fn select_and_comparisons_choose_the_right_arm() {
        let mut b = Builder::new(1, 0);
        let x = b.push(Op::Input(0));
        let one = b.constant(1.0);
        let small = b.push(Op::Lt(x, one));
        let doubled = b.push(Op::Add(x, x));
        let out = b.push(Op::Select(small, doubled, x));
        let program = b.finish(vec![out]);
        assert_eq!(program.eval_to_vec(&[0.25], &[])[0], 0.5);
        assert_eq!(program.eval_to_vec(&[3.0], &[])[0], 3.0);
    }

    /// `f64::min` and `max` may return either zero for `±0.0`; the interpreters may not.
    #[test]
    fn min_and_max_of_the_two_zeros_are_pinned_in_both_interpreters() {
        let mut b = Builder::new(2, 0);
        let (x, y) = (b.push(Op::Input(0)), b.push(Op::Input(1)));
        let lo = b.push(Op::Min(x, y));
        let hi = b.push(Op::Max(x, y));
        let program = b.finish(vec![lo, hi]);
        for (a, c) in [(0.0f64, -0.0f64), (-0.0, 0.0), (0.0, 0.0), (-0.0, -0.0)] {
            let want_lo = if a.is_sign_negative() || c.is_sign_negative() { -0.0f64 } else { 0.0 };
            let want_hi = if a.is_sign_negative() && c.is_sign_negative() { -0.0f64 } else { 0.0 };
            let scalar = program.eval_to_vec(&[a, c], &[]);
            assert_eq!(scalar[0].to_bits(), want_lo.to_bits(), "min({a:?}, {c:?})");
            assert_eq!(scalar[1].to_bits(), want_hi.to_bits(), "max({a:?}, {c:?})");
            let mut inputs = [0.0; 2 * LANES];
            for lane in 0..LANES {
                inputs[lane] = a;
                inputs[LANES + lane] = c;
            }
            let (mut regs, mut out) = ([0.0; 4 * LANES], [0.0; 2 * LANES]);
            program.eval_batch(LANES, &inputs, &[], &mut regs, &mut out);
            for lane in 0..LANES {
                assert_eq!(out[lane].to_bits(), want_lo.to_bits(), "batched min({a:?}, {c:?})");
                assert_eq!(out[LANES + lane].to_bits(), want_hi.to_bits(), "batched max({a:?}, {c:?})");
            }
        }
        // Unequal operands and NaN are as `f64::min` and `max` have them.
        assert_eq!(program.eval_to_vec(&[1.0, 2.0], &[]), [1.0, 2.0]);
        assert_eq!(program.eval_to_vec(&[f64::NAN, 2.0], &[]), [2.0, 2.0]);
    }

    #[test]
    fn the_fingerprint_is_stable_and_discriminating() {
        let a = lennard_jones();
        assert_eq!(a.fingerprint(), lennard_jones().fingerprint());
        assert_ne!(a.fingerprint(), a.derivative(0).fingerprint());
    }

    const fn origin(start: u32, end: u32) -> Option<Origin> {
        Some(Origin { start, end })
    }

    #[test]
    fn a_repeat_keeps_the_first_origin() {
        let mut b = Builder::new(1, 0);
        b.at(origin(0, 1));
        let x = b.push(Op::Input(0));
        b.at(origin(4, 11));
        let first = b.push(Op::Sqrt(x));
        b.at(origin(20, 27));
        let second = b.push(Op::Sqrt(x));
        assert_eq!(first, second);
        let program = b.finish(vec![first]);
        assert_eq!(program.origin(0), origin(0, 1));
        assert_eq!(program.origin(1), origin(4, 11), "the repeat did not replace the first origin");
    }

    #[test]
    fn a_folded_constant_has_the_origin_of_its_expression() {
        let mut b = Builder::new(1, 0);
        b.at(origin(0, 1));
        let x = b.push(Op::Input(0));
        b.at(origin(5, 6));
        let two = b.constant(2.0);
        b.at(origin(9, 10));
        let three = b.constant(3.0);
        b.at(origin(5, 10));
        let six = b.push(Op::Mul(two, three));
        assert_eq!(b.constant_value(six), Some(6.0));
        b.at(origin(0, 10));
        let out = b.push(Op::Mul(x, six));
        let program = b.finish(vec![out]);
        // Input, the folded 6, the product: 2 and 3 are dead.
        assert_eq!(program.ops().len(), 3);
        assert_eq!(program.origin(0), origin(0, 1));
        assert_eq!(program.origin(1), origin(5, 10), "the folded 6 came from `2 * 3`");
        assert_eq!(program.origin(2), origin(0, 10));
    }

    #[test]
    fn origins_survive_dropping_dead_operations() {
        let mut b = Builder::new(2, 0);
        b.at(origin(0, 1));
        let dead = b.push(Op::Input(0));
        b.at(origin(2, 3));
        let x = b.push(Op::Input(1));
        let _ = dead;
        b.at(origin(4, 9));
        let out = b.push(Op::Neg(x));
        let program = b.finish(vec![out]);
        assert_eq!(program.ops(), [Op::Input(1), Op::Neg(0)]);
        assert_eq!(program.origin(0), origin(2, 3));
        assert_eq!(program.origin(1), origin(4, 9));
        assert_eq!(program.origin(2), None, "past the end is no origin");
    }

    #[test]
    fn an_operation_pushed_without_an_origin_has_none() {
        let mut b = Builder::new(1, 0);
        let x = b.push(Op::Input(0));
        b.at(origin(3, 4));
        let n = b.push(Op::Neg(x));
        b.at(None);
        let m = b.push(Op::Abs(n));
        let program = b.finish(vec![m]);
        assert_eq!(program.origin(0), None);
        assert_eq!(program.origin(1), origin(3, 4));
        assert_eq!(program.origin(2), None);
    }

    #[test]
    fn a_derivative_inherits_the_origins_of_what_it_differentiates() {
        // `x * x` and `ln(x)`, with each written at its own place.
        let mut b = Builder::new(1, 0);
        b.at(origin(0, 1));
        let x = b.push(Op::Input(0));
        b.at(origin(0, 5));
        let square = b.push(Op::Mul(x, x));
        b.at(origin(8, 13));
        let ln = b.push(Op::Ln(square));
        let program = b.finish(vec![ln]);
        let d = program.derivative(0);
        // d(x·x) = x + x, then d ln = (x + x) / (x·x): every operation of the derivative
        // is at its primal's origin, and none is left without one.
        let at = |kind: fn(&Op) -> bool| -> Vec<Option<Origin>> {
            d.ops().iter().enumerate().filter(|(_, op)| kind(op)).map(|(i, _)| d.origin(i as Reg)).collect()
        };
        assert_eq!(at(|op| matches!(op, Op::Input(_))), [origin(0, 1)]);
        assert_eq!(at(|op| matches!(op, Op::Mul(..))), [origin(0, 5)]);
        assert_eq!(at(|op| matches!(op, Op::Add(..))), [origin(0, 5)], "the tangent of the product");
        assert_eq!(at(|op| matches!(op, Op::Ln(..))), [origin(8, 13)]);
        assert_eq!(at(|op| matches!(op, Op::Div(..))), [origin(8, 13)], "the tangent of the log");
        for index in 0..d.ops().len() {
            assert!(d.origin(index as Reg).is_some(), "operation {index} of the derivative has an origin");
        }
    }

    #[test]
    fn a_gradient_output_constant_takes_the_origin_of_its_primal() {
        // A tangent that is zero is materialized as a constant at the end.
        let mut b = Builder::new(2, 0);
        b.at(origin(0, 1));
        let _ = b.push(Op::Input(0));
        b.at(origin(2, 3));
        let y = b.push(Op::Input(1));
        let g = b.finish(vec![y]).gradient(&[0]);
        // The input, the value, and the tangent 0 of `y` with respect to input 0.
        assert_eq!(g.output_count(), 2);
        let zero = g.ops().iter().position(|op| *op == Op::Const(0.0)).expect("a zero");
        assert_eq!(g.origin(zero as Reg), origin(2, 3));
    }

    #[test]
    fn equality_and_the_fingerprint_ignore_origins() {
        let build = |shift: u32| {
            let mut b = Builder::new(1, 2);
            let at = |start: u32, end: u32, b: &mut Builder| b.at(origin(start + shift, end + shift));
            at(0, 1, &mut b);
            let r = b.push(Op::Input(0));
            at(2, 5, &mut b);
            let eps = b.push(Op::Param(0));
            let sigma = b.push(Op::Param(1));
            at(6, 9, &mut b);
            let s = b.push(Op::Div(sigma, r));
            let out = b.push(Op::Mul(eps, s));
            b.finish(vec![out])
        };
        let (here, there) = (build(0), build(100));
        assert_ne!(here.origin(0), there.origin(0), "the two really do differ in origin");
        assert_eq!(here, there);
        assert_eq!(here.fingerprint(), there.fingerprint());
        // And a program with no origins at all is the same program.
        let mut plain = Builder::new(1, 2);
        let r = plain.push(Op::Input(0));
        let eps = plain.push(Op::Param(0));
        let sigma = plain.push(Op::Param(1));
        let s = plain.push(Op::Div(sigma, r));
        let out = plain.push(Op::Mul(eps, s));
        let plain = plain.finish(vec![out]);
        assert_eq!(here, plain);
        assert_eq!(here.fingerprint(), plain.fingerprint());
        // Equality still tells different programs apart.
        assert_ne!(here, lennard_jones());
    }

    /// A seeded generator (SplitMix64): the test needs no dependency and must replay.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// A number in `0..n`.
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// A value for an input: usually plain, often one that stresses an operation.
        fn value(&mut self) -> f64 {
            const SPECIAL: [f64; 22] = [
                0.0,
                -0.0,
                1.0,
                -1.0,
                0.5,
                2.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::MIN_POSITIVE,
                -f64::MIN_POSITIVE,
                f64::MIN_POSITIVE / 2.0,
                5e-324,
                -5e-324,
                f64::MAX,
                -f64::MAX,
                f64::MAX / 2.0,
                1.0e154,
                -1.0e154,
                1.0e308,
                f64::EPSILON,
                core::f64::consts::PI,
            ];
            match self.below(4) {
                0 | 1 => SPECIAL[self.below(SPECIAL.len())],
                2 => (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0,
                _ => f64::from_bits(self.next()),
            }
        }
    }

    /// A random program that stays mostly live, built through the [`Builder`] so it is
    /// one the compiler could have produced.
    fn random_program(rng: &mut Rng) -> Program {
        let inputs = 1 + rng.below(4) as u32;
        let params = rng.below(4) as u32;
        let mut b = Builder::new(inputs, params);
        let mut regs: Vec<Reg> = (0..inputs).map(|n| b.push(Op::Input(n))).collect();
        regs.extend((0..params).map(|n| b.push(Op::Param(n))));
        let target = 1 + rng.below(200);
        while b.ops.len() < target {
            // Mostly recent results, so the chain stays live and deep.
            let pick = |rng: &mut Rng| {
                if rng.below(2) == 0 {
                    regs[regs.len() - 1 - rng.below(regs.len().min(3))]
                } else {
                    regs[rng.below(regs.len())]
                }
            };
            let (x, y, z) = (pick(rng), pick(rng), pick(rng));
            let op = match rng.below(29) {
                0 => Op::Const(rng.value()),
                1 => Op::Input(rng.below(inputs as usize) as u32),
                2 if params > 0 => Op::Param(rng.below(params as usize) as u32),
                3 => Op::Neg(x),
                4 => Op::Not(x),
                5 => Op::Sqrt(x),
                6 => Op::Abs(x),
                7 => Op::Exp(x),
                8 => Op::Ln(x),
                9 => Op::Sin(x),
                10 => Op::Cos(x),
                11 => Op::Tan(x),
                12 => Op::Erfc(x),
                13 => Op::Powi(x, [-3, -2, -1, 0, 1, 2, 3, 6, 12, i32::MAX, i32::MIN][rng.below(11)]),
                14 => Op::Add(x, y),
                15 => Op::Sub(x, y),
                16 => Op::Mul(x, y),
                17 => Op::Div(x, y),
                18 => Op::Min(x, y),
                19 => Op::Max(x, y),
                20 => Op::Atan2(x, y),
                21 => Op::Pow(x, y),
                22 => Op::Lt(x, y),
                23 => Op::Le(x, y),
                24 => Op::Eq(x, y),
                25 => Op::Ne(x, y),
                26 => Op::And(x, y),
                27 => Op::Or(x, y),
                28 => Op::Select(x, y, z),
                _ => continue,
            };
            regs.push(b.push(op));
        }
        let mut outputs = vec![*regs.last().expect("a program has registers")];
        for _ in 0..rng.below(4) {
            outputs.push(regs[rng.below(regs.len())]);
        }
        b.finish(outputs)
    }

    /// Two results are the same when they have the same bits, or are both NaN, whose
    /// payload Rust does not pin.
    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// Spec §7: `eval_batch` is `eval`, lane by lane, bit for bit, on 10,000 random
    /// programs of up to 200 operations, every operation among them, inputs that include
    /// ±0, ±∞, NaN, subnormals and values near overflow, and every `active`.
    #[test]
    fn eval_batch_is_eval_lane_by_lane() {
        let mut rng = Rng(0x1a77_1ce0_0d06_f00d);
        let mut seen = [false; 29];
        let mut longest = 0;
        for case in 0..10_000 {
            let program = random_program(&mut rng);
            longest = longest.max(program.ops().len());
            for op in program.ops() {
                seen[usize::from(op.key().0)] = true;
            }
            let (n_in, n_param, n_out) = (program.input_count(), program.param_count(), program.output_count());
            let params: Vec<f64> = (0..n_param).map(|_| rng.value()).collect();
            let lane_inputs: Vec<Vec<f64>> = (0..LANES).map(|_| (0..n_in).map(|_| rng.value()).collect()).collect();
            let expected: Vec<Vec<f64>> = lane_inputs.iter().map(|inputs| program.eval_to_vec(inputs, &params)).collect();
            let mut regs = vec![f64::NAN; program.ops().len() * LANES];
            let mut out = vec![0.0; n_out * LANES];
            for active in 1..=LANES {
                // Lanes at and past `active` hold arbitrary values that must not matter.
                let mut inputs = vec![0.0; n_in * LANES];
                for (lane, values) in lane_inputs.iter().enumerate() {
                    for (k, &v) in values.iter().enumerate() {
                        inputs[k * LANES + lane] = if lane < active { v } else { rng.value() };
                    }
                }
                program.eval_batch(active, &inputs, &params, &mut regs, &mut out);
                for lane in 0..active {
                    for o in 0..n_out {
                        let (batch, scalar) = (out[o * LANES + lane], expected[lane][o]);
                        assert!(
                            same(batch, scalar),
                            "program {case}, active {active}, lane {lane}, output {o}: {batch:e} ({:#x}) against {scalar:e} ({:#x})\n{:?}",
                            batch.to_bits(),
                            scalar.to_bits(),
                            program.ops(),
                        );
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "the corpus missed an operation: {seen:?}");
        assert!(longest > 100, "the corpus has no long program (longest {longest})");
    }

    #[test]
    #[should_panic(expected = "active lanes 0")]
    fn a_batch_needs_an_active_lane() {
        let p = lennard_jones();
        p.eval_batch(0, &[1.0; LANES], &[1.0, 1.0], &mut [0.0; 16 * LANES], &mut [0.0; LANES]);
    }

    #[test]
    #[should_panic(expected = "not in 1..")]
    fn a_batch_has_at_most_lanes_active() {
        let p = lennard_jones();
        p.eval_batch(LANES + 1, &[1.0; LANES], &[1.0, 1.0], &mut [0.0; 16 * LANES], &mut [0.0; LANES]);
    }

    #[test]
    #[should_panic(expected = "out holds")]
    fn a_batch_does_not_truncate_a_short_output() {
        let p = lennard_jones();
        let inputs = vec![1.0; p.input_count() * LANES];
        let mut regs = vec![0.0; p.ops().len() * LANES];
        let mut out = vec![0.0; p.output_count() * LANES - 1];
        p.eval_batch(LANES, &inputs, &[1.0, 1.0], &mut regs, &mut out);
    }

    /// `1 / (r - 1)`, each operation at the place it would be written.
    fn pole_at_one() -> Program {
        let mut b = Builder::new(1, 0);
        b.at(origin(10, 11));
        let r = b.push(Op::Input(0));
        b.at(origin(14, 15));
        let one = b.constant(1.0);
        b.at(origin(10, 15));
        let gap = b.push(Op::Sub(r, one));
        b.at(origin(5, 16));
        let u = b.push(Op::Div(one, gap));
        b.finish(vec![u])
    }

    #[test]
    fn a_trace_finds_the_operation_that_made_the_value_and_what_it_read() {
        let program = pole_at_one();
        assert_eq!(program.trace(&[2.0], &[]), None, "finite everywhere");
        let at = program.trace(&[1.0], &[]).expect("a division by zero");
        // Input, the constant 1, the difference, the quotient.
        assert_eq!(at.reg, 3);
        assert_eq!(at.op, Op::Div(1, 2));
        assert_eq!(at.origin, origin(5, 16));
        assert_eq!(at.operands, [1.0, 0.0]);
        assert_eq!(at.value, f64::INFINITY);
    }

    #[test]
    fn a_trace_skips_what_only_propagated_a_non_finite_value() {
        // `sqrt(1/(r-1))` after `1/(r-1)` is infinite: the sqrt is infinite too, but it
        // read an infinity, so the division is blamed and not the sqrt.
        let mut b = Builder::new(1, 0);
        let r = b.push(Op::Input(0));
        let one = b.constant(1.0);
        let gap = b.push(Op::Sub(r, one));
        let q = b.push(Op::Div(one, gap));
        let root = b.push(Op::Sqrt(q));
        let at = b.finish(vec![root]).trace(&[1.0], &[]).unwrap();
        assert!(matches!(at.op, Op::Div(..)), "{at:?}");
    }

    #[test]
    fn a_nan_made_of_finite_operands_is_found() {
        let mut b = Builder::new(1, 0);
        b.at(origin(3, 8));
        let x = b.push(Op::Input(0));
        b.at(origin(0, 9));
        let ln = b.push(Op::Ln(x));
        let at = b.finish(vec![ln]).trace(&[-2.0], &[]).unwrap();
        assert_eq!((at.op, at.operands.as_slice(), at.origin), (Op::Ln(0), &[-2.0][..], origin(0, 9)));
        assert!(at.value.is_nan());
    }

    #[test]
    fn a_non_finite_input_is_blamed_as_an_input_and_no_operation_is() {
        // `1 / (r - 1)` turns an infinite `r` into a finite zero, so only NaN reaches its
        // output; `r - 1` carries an infinity through.
        let mut b = Builder::new(1, 0);
        b.at(origin(10, 11));
        let r = b.push(Op::Input(0));
        let one = b.constant(1.0);
        let gap = b.push(Op::Sub(r, one));
        let shifted = b.finish(vec![gap]);
        for (program, bad) in [
            (pole_at_one(), f64::NAN),
            (shifted.clone(), f64::NAN),
            (shifted.clone(), f64::INFINITY),
            (shifted, f64::NEG_INFINITY),
        ] {
            let at = program.trace(&[bad], &[]).unwrap();
            assert_eq!((at.reg, at.op), (0, Op::Input(0)), "the input, not the subtraction after it");
            assert!(at.operands.is_empty());
            assert_eq!(at.origin, origin(10, 11));
            assert_eq!(at.value.to_bits(), bad.to_bits());
        }
    }

    #[test]
    fn a_non_finite_parameter_is_blamed_as_a_parameter() {
        let mut b = Builder::new(1, 1);
        let x = b.push(Op::Input(0));
        let k = b.push(Op::Param(0));
        let y = b.push(Op::Mul(k, x));
        let at = b.finish(vec![y]).trace(&[1.0], &[f64::NAN]).unwrap();
        assert_eq!(at.op, Op::Param(0));
    }

    /// `(if x == 0 { 0 } else { 1/x }) + ln(y)` at `x = 0`, `y = −1`: the guarded division
    /// is infinite but discarded, and the NaN is the log's. The trace blames the log.
    #[test]
    fn a_trace_blames_the_log_and_not_the_division_a_select_discarded() {
        let mut b = Builder::new(2, 0);
        let (x, y) = (b.push(Op::Input(0)), b.push(Op::Input(1)));
        let zero = b.constant(0.0);
        let one = b.constant(1.0);
        b.at(origin(10, 13));
        let pole = b.push(Op::Div(one, x));
        let is_zero = b.push(Op::Eq(x, zero));
        let guarded = b.push(Op::Select(is_zero, zero, pole));
        b.at(origin(30, 35));
        let ln = b.push(Op::Ln(y));
        let sum = b.push(Op::Add(guarded, ln));
        let program = b.finish(vec![sum]);

        let at = program.trace(&[0.0, -1.0], &[]).unwrap();
        assert_eq!((at.op, at.origin, at.operands.as_slice()), (Op::Ln(1), origin(30, 35), &[-1.0][..]));

        // With the log fine the output is finite, and so is the whole trace, although the
        // division at x = 0 is infinite in a register.
        assert_eq!(program.eval_to_vec(&[0.0, 2.0], &[])[0], 2f64.ln());
        assert_eq!(program.trace(&[0.0, 2.0], &[]), None);
        // A guard that discards the division leaves a finite output: nothing to trace.
        let mut b = Builder::new(1, 0);
        let x = b.push(Op::Input(0));
        let (zero, one) = (b.constant(0.0), b.constant(1.0));
        let pole = b.push(Op::Div(one, x));
        let nonzero = b.push(Op::Ne(x, zero));
        let taken = b.push(Op::Select(nonzero, pole, zero));
        let p = b.finish(vec![taken]);
        assert_eq!(p.trace(&[0.0], &[]), None);
        // A select whose chosen arm is the bad one passes the blame to that arm's op.
        let mut b = Builder::new(1, 0);
        let x = b.push(Op::Input(0));
        let (zero, one) = (b.constant(0.0), b.constant(1.0));
        let pole = b.push(Op::Div(one, x));
        let is_zero = b.push(Op::Eq(x, zero));
        let chosen = b.push(Op::Select(is_zero, pole, zero));
        let at = b.finish(vec![chosen]).trace(&[0.0], &[]).unwrap();
        assert!(matches!(at.op, Op::Div(..)));
    }

    /// `let t = ln(y); 1/x + t` at `x = 0`, `y = -1`: two independent faults meet at the
    /// sum. The log is the first computed, so it is the one reported, although the
    /// division is the sum's first operand.
    #[test]
    fn when_two_faults_meet_the_first_computed_is_blamed() {
        let mut b = Builder::new(2, 0);
        let (x, y) = (b.push(Op::Input(0)), b.push(Op::Input(1)));
        b.at(origin(5, 10));
        let t = b.push(Op::Ln(y));
        let one = b.constant(1.0);
        b.at(origin(20, 23));
        let pole = b.push(Op::Div(one, x));
        let sum = b.push(Op::Add(pole, t));
        let program = b.finish(vec![sum]);
        let at = program.trace(&[0.0, -1.0], &[]).unwrap();
        assert_eq!((at.op, at.origin), (Op::Ln(1), origin(5, 10)));
        // Either fault alone is reported as itself.
        assert!(matches!(program.trace(&[0.0, 2.0], &[]).unwrap().op, Op::Div(..)));
        assert!(matches!(program.trace(&[1.0, -1.0], &[]).unwrap().op, Op::Ln(_)));
    }

    /// The trace starts from the outputs the caller checked, not any output of the program.
    #[test]
    fn a_trace_looks_only_at_the_outputs_asked_for() {
        let mut b = Builder::new(2, 0);
        let (x, y) = (b.push(Op::Input(0)), b.push(Op::Input(1)));
        let ln = b.push(Op::Ln(y));
        let neg = b.push(Op::Neg(x));
        let program = b.finish(vec![neg, ln]);
        assert_eq!(program.trace_outputs(&[1.0, -1.0], &[], 0..1), None, "the first output is fine");
        assert!(matches!(program.trace_outputs(&[1.0, -1.0], &[], 0..2).unwrap().op, Op::Ln(_)));
        assert!(matches!(program.trace(&[1.0, -1.0], &[]).unwrap().op, Op::Ln(_)));
    }

    #[test]
    fn erfc_matches_reference_values() {
        for (x, expected) in [(0.0, 1.0), (0.5, 0.479_500_122_186_953_5), (2.5, 4.069_520_174_449_59e-4), (-1.0, 1.842_700_792_949_715)] {
            assert!(((erfc(x) - expected) / expected).abs() < 1e-12, "erfc({x})");
        }
    }
}
