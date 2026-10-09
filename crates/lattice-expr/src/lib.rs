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
//! # No allocation while evaluating
//!
//! [`Program::eval`] runs on a register array the caller owns. Laws are small, and
//! [`MAX_REGISTERS`] bounds them so a stack array always fits: a pair loop evaluates a
//! law once per pair, and spec NFR-001 allows no allocation there.

use std::collections::HashMap;

/// A register: the index of the operation that wrote it.
pub type Reg = u32;

/// The most operations a program may have, so its registers fit a stack array.
pub const MAX_REGISTERS: usize = 512;

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

/// Apply one operation to register values: the single definition of what each
/// operation computes, shared by the interpreter and constant folding.
#[inline]
fn apply(op: Op, regs: &[f64], inputs: &[f64], params: &[f64]) -> f64 {
    use Op::*;
    let g = |x: Reg| regs[x as usize];
    let truth = |b: bool| if b { 1.0 } else { 0.0 };
    match op {
        Const(v) => v,
        Input(n) => inputs[n as usize],
        Param(n) => params[n as usize],
        Neg(x) => -g(x),
        Not(x) => truth(g(x) == 0.0),
        Sqrt(x) => g(x).sqrt(),
        Abs(x) => g(x).abs(),
        Exp(x) => g(x).exp(),
        Ln(x) => g(x).ln(),
        Sin(x) => g(x).sin(),
        Cos(x) => g(x).cos(),
        Tan(x) => g(x).tan(),
        Erfc(x) => erfc(g(x)),
        Powi(x, n) => g(x).powi(n),
        Add(a, b) => g(a) + g(b),
        Sub(a, b) => g(a) - g(b),
        Mul(a, b) => g(a) * g(b),
        Div(a, b) => g(a) / g(b),
        Min(a, b) => g(a).min(g(b)),
        Max(a, b) => g(a).max(g(b)),
        Atan2(y, x) => g(y).atan2(g(x)),
        Pow(a, b) => g(a).powf(g(b)),
        Lt(a, b) => truth(g(a) < g(b)),
        Le(a, b) => truth(g(a) <= g(b)),
        Eq(a, b) => truth(g(a) == g(b)),
        Ne(a, b) => truth(g(a) != g(b)),
        And(a, b) => truth(g(a) != 0.0 && g(b) != 0.0),
        Or(a, b) => truth(g(a) != 0.0 || g(b) != 0.0),
        Select(c, a, b) => {
            if g(c) != 0.0 {
                g(a)
            } else {
                g(b)
            }
        }
    }
}

/// Builds a [`Program`], removing repeated operations and folding constants.
#[derive(Debug)]
pub struct Builder {
    ops: Vec<Op>,
    /// The constant value of each register, when it has one.
    constant: Vec<Option<f64>>,
    seen: HashMap<(u8, u64, u64, u64), Reg>,
    inputs: u32,
    params: u32,
}

impl Builder {
    /// A builder for a program of `inputs` inputs and `params` parameters.
    pub fn new(inputs: u32, params: u32) -> Builder {
        Builder { ops: Vec::new(), constant: Vec::new(), seen: HashMap::new(), inputs, params }
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
        for (index, op) in self.ops.iter().enumerate() {
            if live[index] {
                renumber[index] = ops.len() as Reg;
                ops.push(op.remap(|r| renumber[r as usize]));
            }
        }
        let outputs = outputs.iter().map(|&r| renumber[r as usize]).collect();
        Program { ops, outputs, inputs: self.inputs, params: self.params }
    }
}

/// A compiled law: operations in single-assignment form and the registers it returns.
#[derive(Clone, PartialEq, Debug)]
pub struct Program {
    ops: Vec<Op>,
    outputs: Vec<Reg>,
    inputs: u32,
    params: u32,
}

impl Program {
    /// The operations, in order.
    pub fn ops(&self) -> &[Op] {
        &self.ops
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
        for &op in &self.ops {
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
                outputs.push(tangent[r as usize].materialize(&mut b));
            }
        }
        b.finish(outputs)
    }

    /// A stable fingerprint of what the program computes, for caches and artifacts.
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

    #[test]
    fn the_fingerprint_is_stable_and_discriminating() {
        let a = lennard_jones();
        assert_eq!(a.fingerprint(), lennard_jones().fingerprint());
        assert_ne!(a.fingerprint(), a.derivative(0).fingerprint());
    }

    #[test]
    fn erfc_matches_reference_values() {
        for (x, expected) in [(0.0, 1.0), (0.5, 0.479_500_122_186_953_5), (2.5, 4.069_520_174_449_59e-4), (-1.0, 1.842_700_792_949_715)] {
            assert!(((erfc(x) - expected) / expected).abs() < 1e-12, "erfc({x})");
        }
    }
}
