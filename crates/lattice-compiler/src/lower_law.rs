//! Lowering a checked law into a [`Program`] (spec §8.3, §8.4 "normalize expressions
//! and lower high-level constructs").
//!
//! A [`TypedLaw`] has vectors, truth values and particles; a program has only `f64`
//! registers. Lowering splits every vector into its two components, makes a truth value
//! `0.0` or `1.0`, and replaces each particle member with an input of the particle
//! domain's [`layout`]. It keeps the author's operation order exactly: `a * b * c` is
//! `(a × b) × c`, a vector times a scalar is `x × s` and a scalar times a vector is
//! `s × x`, so a law written as a built-in is written computes the built-in's bits.

use lattice_domain_particle::user::{layout, UserLawKind};
use lattice_expr::{Builder, Op, Program, Reg};
use lattice_syntax::{BinaryOp, Diagnostic};

use crate::laws::{Builtin, LawKind, Member, TExpr, TExprKind, Ty, TypedLaw};

/// A law ready for a domain: its program and what the program's outputs mean.
#[derive(Clone, Debug)]
pub struct LoweredLaw {
    /// How the domain applies it.
    pub kind: UserLawKind,
    /// The program, returning what [`UserLawKind::outputs`] says.
    pub program: Program,
    /// The program before differentiation, for a potential: it returns `U` alone, and is
    /// what the compiler samples to find the potential's well.
    pub energy: Option<Program>,
    /// How many parameter slots it reads: a scalar `param` takes one, a vector two.
    pub param_slots: usize,
    /// True when the law branches (`if`, `min`, `max`, `abs`, a comparison), so a
    /// potential may be discontinuous and its energy conserved only piecewise.
    pub branches: bool,
}

#[derive(Clone, Copy)]
enum Value {
    Scalar(Reg),
    Vector(Reg, Reg),
}

impl Value {
    fn scalar(self) -> Reg {
        match self {
            Value::Scalar(r) => r,
            Value::Vector(x, _) => x,
        }
    }

    fn vector(self) -> (Reg, Reg) {
        match self {
            Value::Vector(x, y) => (x, y),
            Value::Scalar(s) => (s, s),
        }
    }
}

/// Lower a checked law. Fails only for a construct the checker accepts that M6.1b does
/// not lower — today, a `minimum_image` of anything but the pair's own separation.
pub fn lower(law: &TypedLaw) -> Result<LoweredLaw, Box<Diagnostic>> {
    let kind = match (law.kind, law.is_pair()) {
        (LawKind::Force, true) => UserLawKind::PairForce,
        (LawKind::Force, false) => UserLawKind::BodyForce,
        (LawKind::Potential, true) => UserLawKind::PairPotential,
        (LawKind::Potential, false) => UserLawKind::BodyPotential,
    };
    let mut offsets = Vec::with_capacity(law.params.len());
    let mut slots = 0u32;
    for param in &law.params {
        offsets.push(slots);
        slots += if matches!(param.ty, Ty::Vec2(_)) { 2 } else { 1 };
    }
    let inputs = kind.inputs() as u32;
    let mut lowering = Lowering { b: Builder::new(inputs, slots), law, offsets, locals: Vec::new(), branches: false };
    for local in &law.locals {
        let value = lowering.expr(&local.value)?;
        lowering.locals.push(value);
    }
    let result = lowering.expr(&law.result)?;
    let branches = lowering.branches;
    let outputs = match result {
        Value::Vector(x, y) => vec![x, y],
        Value::Scalar(u) => vec![u],
    };
    let written = lowering.b.finish(outputs);
    let (program, energy) = match kind {
        UserLawKind::PairPotential => (written.derivative(layout::R), Some(written)),
        UserLawKind::BodyPotential => {
            let a = layout::BODY_A;
            (written.gradient(&[a + layout::PX, a + layout::PY]), Some(written))
        }
        UserLawKind::PairForce | UserLawKind::BodyForce => (written, None),
    };
    if !program.fits() {
        return Err(Box::new(Diagnostic::error(format!("`{}` is too large to compile", law.name))
            .with_code("E0413")
            .at(law.span, format!("{} operations", program.ops().len()))
            .note(format!(
                "a law runs on a fixed register file of {} values; split it into smaller laws",
                lattice_expr::MAX_REGISTERS
            ))));
    }
    Ok(LoweredLaw { kind, program, energy, param_slots: slots as usize, branches })
}

struct Lowering<'a> {
    b: Builder,
    law: &'a TypedLaw,
    offsets: Vec<u32>,
    locals: Vec<Value>,
    branches: bool,
}

impl Lowering<'_> {
    fn input(&mut self, n: u32) -> Reg {
        self.b.push(Op::Input(n))
    }

    fn particle_base(&self, particle: usize) -> u32 {
        if self.law.is_pair() {
            if particle == 0 { layout::PAIR_A } else { layout::PAIR_B }
        } else {
            layout::BODY_A
        }
    }

    fn expr(&mut self, e: &TExpr) -> Result<Value, Box<Diagnostic>> {
        Ok(match &e.kind {
            TExprKind::Const(v) => Value::Scalar(self.b.constant(*v)),
            TExprKind::Local(i) => self.locals[*i],
            TExprKind::Param(i) => {
                let at = self.offsets[*i];
                match self.law.params[*i].ty {
                    Ty::Vec2(_) => Value::Vector(self.b.push(Op::Param(at)), self.b.push(Op::Param(at + 1))),
                    _ => Value::Scalar(self.b.push(Op::Param(at))),
                }
            }
            TExprKind::Field { particle, member } => {
                let base = self.particle_base(*particle);
                match member {
                    Member::Position => Value::Vector(self.input(base + layout::PX), self.input(base + layout::PY)),
                    Member::Velocity => Value::Vector(self.input(base + layout::VX), self.input(base + layout::VY)),
                    Member::Mass => Value::Scalar(self.input(base + layout::MASS)),
                    Member::Charge => Value::Scalar(self.input(base + layout::CHARGE)),
                }
            }
            TExprKind::Distance => Value::Scalar(self.input(layout::R)),
            TExprKind::Component(v, k) => {
                let (x, y) = self.expr(v)?.vector();
                Value::Scalar(if *k == 0 { x } else { y })
            }
            TExprKind::Neg(v) => match self.expr(v)? {
                Value::Scalar(s) => Value::Scalar(self.b.push(Op::Neg(s))),
                Value::Vector(x, y) => Value::Vector(self.b.push(Op::Neg(x)), self.b.push(Op::Neg(y))),
            },
            TExprKind::Not(v) => {
                let s = self.expr(v)?.scalar();
                Value::Scalar(self.b.push(Op::Not(s)))
            }
            TExprKind::Power(v, n) => {
                let s = self.expr(v)?.scalar();
                Value::Scalar(self.b.push(Op::Powi(s, *n)))
            }
            TExprKind::Binary(op, l, r) => self.binary(*op, l, r)?,
            TExprKind::Select(c, a, b) => {
                self.branches = true;
                let c = self.expr(c)?.scalar();
                match (self.expr(a)?, self.expr(b)?) {
                    (Value::Vector(ax, ay), Value::Vector(bx, by)) => {
                        Value::Vector(self.b.push(Op::Select(c, ax, bx)), self.b.push(Op::Select(c, ay, by)))
                    }
                    (a, b) => Value::Scalar(self.b.push(Op::Select(c, a.scalar(), b.scalar()))),
                }
            }
            TExprKind::Call(builtin, args) => self.call(*builtin, args, e)?,
        })
    }

    fn binary(&mut self, op: BinaryOp, l: &TExpr, r: &TExpr) -> Result<Value, Box<Diagnostic>> {
        let (lv, rv) = (self.expr(l)?, self.expr(r)?);
        let scalar = |b: &mut Builder, op: fn(Reg, Reg) -> Op| Value::Scalar(b.push(op(lv.scalar(), rv.scalar())));
        Ok(match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let f: fn(Reg, Reg) -> Op = if op == BinaryOp::Add { Op::Add } else { Op::Sub };
                match (lv, rv) {
                    (Value::Vector(ax, ay), Value::Vector(bx, by)) => Value::Vector(self.b.push(f(ax, bx)), self.b.push(f(ay, by))),
                    _ => scalar(&mut self.b, f),
                }
            }
            BinaryOp::Mul => match (lv, rv) {
                (Value::Scalar(s), Value::Vector(x, y)) => Value::Vector(self.b.push(Op::Mul(s, x)), self.b.push(Op::Mul(s, y))),
                (Value::Vector(x, y), Value::Scalar(s)) => Value::Vector(self.b.push(Op::Mul(x, s)), self.b.push(Op::Mul(y, s))),
                _ => scalar(&mut self.b, Op::Mul),
            },
            BinaryOp::Div => match (lv, rv) {
                (Value::Vector(x, y), Value::Scalar(s)) => Value::Vector(self.b.push(Op::Div(x, s)), self.b.push(Op::Div(y, s))),
                _ => scalar(&mut self.b, Op::Div),
            },
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne => {
                self.branches = true;
                let (a, b) = (lv.scalar(), rv.scalar());
                Value::Scalar(self.b.push(match op {
                    BinaryOp::Lt => Op::Lt(a, b),
                    BinaryOp::Le => Op::Le(a, b),
                    BinaryOp::Gt => Op::Lt(b, a),
                    BinaryOp::Ge => Op::Le(b, a),
                    BinaryOp::Eq => Op::Eq(a, b),
                    _ => Op::Ne(a, b),
                }))
            }
            BinaryOp::And => scalar(&mut self.b, Op::And),
            BinaryOp::Or => scalar(&mut self.b, Op::Or),
        })
    }

    fn length(&mut self, x: Reg, y: Reg) -> Reg {
        // `x² + y²` in the neighbour list's order, so `length(minimum_image(…))` is the
        // pair loop's own `r`.
        let xx = self.b.push(Op::Mul(x, x));
        let yy = self.b.push(Op::Mul(y, y));
        let sum = self.b.push(Op::Add(xx, yy));
        self.b.push(Op::Sqrt(sum))
    }

    fn call(&mut self, builtin: Builtin, args: &[TExpr], at: &TExpr) -> Result<Value, Box<Diagnostic>> {
        use Builtin as B;
        if builtin == B::MinimumImage {
            return self.minimum_image(&args[0], at);
        }
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            values.push(self.expr(arg)?);
        }
        let s = |i: usize| values[i].scalar();
        let unary = |b: &mut Builder, op: fn(Reg) -> Op| Value::Scalar(b.push(op(s(0))));
        Ok(match builtin {
            B::Sqrt => unary(&mut self.b, Op::Sqrt),
            B::Abs => {
                self.branches = true;
                unary(&mut self.b, Op::Abs)
            }
            B::Exp => unary(&mut self.b, Op::Exp),
            B::Ln => unary(&mut self.b, Op::Ln),
            B::Sin => unary(&mut self.b, Op::Sin),
            B::Cos => unary(&mut self.b, Op::Cos),
            B::Tan => unary(&mut self.b, Op::Tan),
            B::Erfc => unary(&mut self.b, Op::Erfc),
            B::Pow => Value::Scalar(self.b.push(Op::Pow(s(0), s(1)))),
            B::Atan2 => Value::Scalar(self.b.push(Op::Atan2(s(0), s(1)))),
            B::Min | B::Max | B::Clamp => {
                self.branches = true;
                match builtin {
                    B::Min => Value::Scalar(self.b.push(Op::Min(s(0), s(1)))),
                    B::Max => Value::Scalar(self.b.push(Op::Max(s(0), s(1)))),
                    _ => {
                        let low = self.b.push(Op::Max(s(0), s(1)));
                        Value::Scalar(self.b.push(Op::Min(low, s(2))))
                    }
                }
            }
            B::Vec2 => Value::Vector(s(0), s(1)),
            B::Length => {
                let (x, y) = values[0].vector();
                Value::Scalar(self.length(x, y))
            }
            B::Dot => {
                let ((ax, ay), (bx, by)) = (values[0].vector(), values[1].vector());
                let xx = self.b.push(Op::Mul(ax, bx));
                let yy = self.b.push(Op::Mul(ay, by));
                Value::Scalar(self.b.push(Op::Add(xx, yy)))
            }
            B::Cross => {
                let ((ax, ay), (bx, by)) = (values[0].vector(), values[1].vector());
                let xy = self.b.push(Op::Mul(ax, by));
                let yx = self.b.push(Op::Mul(ay, bx));
                Value::Scalar(self.b.push(Op::Sub(xy, yx)))
            }
            B::Normalize => {
                let (x, y) = values[0].vector();
                let length = self.length(x, y);
                Value::Vector(self.b.push(Op::Div(x, length)), self.b.push(Op::Div(y, length)))
            }
            B::MinimumImage => unreachable!("handled above"),
        })
    }

    /// `minimum_image(b.position − a.position)` is the pair loop's own separation `d`,
    /// and `minimum_image(a.position − b.position)` is `−d`. A general minimum image of
    /// an arbitrary vector needs the box at run time, which M6.1b does not pass to a law.
    fn minimum_image(&mut self, arg: &TExpr, at: &TExpr) -> Result<Value, Box<Diagnostic>> {
        let position = |e: &TExpr| match e.kind {
            TExprKind::Field { particle, member: Member::Position } => Some(particle),
            _ => None,
        };
        if let TExprKind::Binary(BinaryOp::Sub, l, r) = &arg.kind {
            match (position(l), position(r)) {
                (Some(1), Some(0)) => {
                    return Ok(Value::Vector(self.input(layout::DX), self.input(layout::DY)));
                }
                (Some(0), Some(1)) => {
                    let (dx, dy) = (self.input(layout::DX), self.input(layout::DY));
                    return Ok(Value::Vector(self.b.push(Op::Neg(dx)), self.b.push(Op::Neg(dy))));
                }
                _ => {}
            }
        }
        let (a, b) = (&self.law.particles[0], &self.law.particles[1]);
        Err(Box::new(Diagnostic::error("`minimum_image` takes the separation of the law's two particles")
            .with_code("E0412")
            .at(at.span, "not a separation of the pair")
            .help(format!("write `minimum_image({b}.position - {a}.position)`"))
            .note("the image of an arbitrary vector needs the box, which a law is not given")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Evaluator;
    use crate::laws::check_laws;
    use lattice_syntax::{parse, Diagnostics, SourceFile};
    use lattice_units::UnitRegistry;

    fn lowered(source: &str) -> LoweredLaw {
        let file = SourceFile::new("t.lattice", source);
        let (project, _) = parse(&file);
        let units = UnitRegistry::si();
        let mut diagnostics = Diagnostics::new();
        let laws = check_laws(&project.unwrap(), &Evaluator::new(&file, &units), &mut diagnostics);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        lower(&laws[0]).unwrap()
    }

    fn pair_inputs(dx: f64, dy: f64) -> Vec<f64> {
        let mut inputs = vec![0.0; layout::PAIR_INPUTS as usize];
        inputs[layout::DX as usize] = dx;
        inputs[layout::DY as usize] = dy;
        inputs[layout::R as usize] = (dx * dx + dy * dy).sqrt();
        inputs[(layout::PAIR_B + layout::PX) as usize] = dx;
        inputs[(layout::PAIR_B + layout::PY) as usize] = dy;
        inputs
    }

    /// A Lennard-Jones force written in the built-in's operation order computes the
    /// built-in's coefficient to the bit.
    #[test]
    fn a_force_in_builtin_order_computes_builtin_bits() {
        let law = lowered(
            "project p { force lj(a: particle, b: particle) -> vec2<newton> {
                param epsilon: joule;
                param sigma: meter;
                let d = minimum_image(b.position - a.position);
                let inv_r2 = 1 / dot(d, d);
                let s6 = (sigma * sigma * inv_r2)^3;
                let s12 = s6 * s6;
                let coefficient = 24 * epsilon * inv_r2 * (2 * s12 - s6);
                return -(coefficient * d);
            } }",
        );
        assert_eq!(law.kind, UserLawKind::PairForce);
        let (epsilon, sigma) = (1.65e-21_f64, 3.4e-10_f64);
        let (dx, dy) = (3.1e-10_f64, -1.7e-10_f64);
        let out = law.program.eval_to_vec(&pair_inputs(dx, dy), &[epsilon, sigma]);
        // The built-in, `forces.rs`: coefficient = 24ε · inv_r2 · (2 s12 − s6), F_i = −c d.
        let (sigma2, r2) = (sigma * sigma, dx * dx + dy * dy);
        let inv_r2 = 1.0 / r2;
        let s6 = (sigma2 * inv_r2).powi(3);
        let s12 = s6 * s6;
        let coefficient = 24.0 * epsilon * inv_r2 * (2.0 * s12 - s6);
        assert_eq!(out[0].to_bits(), (-(coefficient * dx)).to_bits());
        assert_eq!(out[1].to_bits(), (-(coefficient * dy)).to_bits());
    }

    #[test]
    fn a_pair_potential_lowers_to_its_energy_and_derivative() {
        let law = lowered(
            "project p { potential well(a: particle, b: particle) -> joule {
                param k: newton / meter;
                return 0.5 * k * (distance(a, b) - 1 meter)^2;
            } }",
        );
        assert_eq!(law.kind, UserLawKind::PairPotential);
        assert!(!law.branches);
        let out = law.program.eval_to_vec(&pair_inputs(1.5, 0.0), &[4.0]);
        assert!((out[0] - 0.5).abs() < 1e-15 && (out[1] - 2.0).abs() < 1e-15, "{out:?}");
    }

    #[test]
    fn a_body_potential_lowers_to_its_gradient() {
        let law = lowered(
            "project p { potential trap(a: particle) -> joule {
                return 2 newton / meter * dot(a.position, a.position);
            } }",
        );
        let mut inputs = vec![0.0; layout::BODY_INPUTS as usize];
        inputs[layout::PX as usize] = 0.5;
        inputs[layout::PY as usize] = -1.0;
        let out = law.program.eval_to_vec(&inputs, &[]);
        assert_eq!(out, [2.5, 2.0, -4.0]);
    }

    #[test]
    fn branches_are_noticed() {
        let law = lowered(
            "project p { potential step(a: particle, b: particle) -> joule {
                return if distance(a, b) < 1 meter { 1 joule } else { 0 joule };
            } }",
        );
        assert!(law.branches);
    }

    #[test]
    fn minimum_image_takes_only_the_pair_separation() {
        let file = SourceFile::new(
            "t.lattice",
            "project p { force f(a: particle, b: particle) -> vec2<newton> {
                return 1 newton / meter * minimum_image(a.position + b.position);
            } }",
        );
        let (project, _) = parse(&file);
        let units = UnitRegistry::si();
        let mut diagnostics = Diagnostics::new();
        let laws = check_laws(&project.unwrap(), &Evaluator::new(&file, &units), &mut diagnostics);
        let error = lower(&laws[0]).unwrap_err();
        assert_eq!(error.code.as_deref(), Some("E0412"));
    }
}
