//! User-defined laws (spec §8.3): type and unit checking.
//!
//! Spec §8.3 asks for *"a restricted, typed, side-effect-free language for scalar/vector
//! algebra, … pair interactions, conditions, … and selected mathematical functions"*,
//! compiled rather than interpreted. This module is the front half of that: it reads
//! each `force` or `potential` declaration, gives every sub-expression a type — a
//! scalar or a two-component vector of some physical dimension, a truth value, or a
//! particle — and reports anything that does not add up, at the character that caused
//! it. What comes out is a [`TypedLaw`]: the law with every name resolved, every unit
//! folded into an SI constant, and nothing left that needs the source to understand.
//!
//! # Dimensions are checked here and nowhere else
//!
//! The runtime that evaluates a law is plain `f64`, as the rest of the engine is. Units
//! cost nothing at run time because they are finished with by the time this module
//! returns: `4 * epsilon` is a multiplication of two numbers, and the fact that the
//! product is an energy was established once, at compile time.
//!
//! # What a law kind promises
//!
//! The kinds differ in what they may read, and the restriction is the point. A pair
//! `potential` sees the two particles only through `distance(a, b)`, so the force it
//! implies is central and equal and opposite: momentum is conserved by construction,
//! whatever the author wrote, and angular momentum too in open space (a periodic seam
//! breaks it, as it does for every central force). Energy is conserved by the
//! integrator only where the energy is smooth. The language admits `if`, `abs`, `min`
//! and `max`, so a law can jump or kink, and at a jump the derived force is silent and
//! the integrator crosses it with a step change in energy. A solver contract for a
//! potential must therefore say "energy, where U is continuous", and M6.1b's will. A
//! `force` may read anything, including velocities, and is promised nothing about
//! energy. Making the restrictions type errors rather than conventions is what lets the
//! parts of a contract that are claimed be true by construction (P1).

use lattice_syntax::{
    BinaryOp, Diagnostic, Diagnostics, Expr, ExprKind, Ident, LawDecl, Project, Span, StmtKind,
    TypeExpr, TypeKind, UnaryOp,
};
use lattice_units::Dimension;

use crate::eval::Evaluator;

/// The type of a value inside a law.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    /// A number of some physical dimension.
    Scalar(Dimension),
    /// A two-component vector whose components share a dimension.
    Vec2(Dimension),
    /// A truth value, from a comparison.
    Bool,
    /// A particle, readable only through its members and `distance`.
    Particle,
}

impl Ty {
    /// How the type reads in a diagnostic.
    pub fn describe(self) -> String {
        match self {
            Ty::Scalar(d) => d.describe(),
            Ty::Vec2(d) => format!("a vector of {}", d.describe()),
            Ty::Bool => "a truth value".to_string(),
            Ty::Particle => "a particle".to_string(),
        }
    }

    fn shape(self) -> &'static str {
        match self {
            Ty::Scalar(_) => "a scalar",
            Ty::Vec2(_) => "a vector",
            Ty::Bool => "a truth value",
            Ty::Particle => "a particle",
        }
    }
}

/// The kinds of law the compiler knows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LawKind {
    /// `force f(a) -> vec2<newton>` or `force f(a, b) -> vec2<newton>`.
    Force,
    /// `potential u(a) -> joule` or `potential u(a, b) -> joule`.
    Potential,
    /// A reaction's `rate:` written as a law (spec §8.3), `k * c(A) * c(B)`: no header
    /// and no body, just the expression. It reads concentrations through `c(…)` and the
    /// cell's `temperature`, and returns a rate in mol/(m²·s).
    Rate,
    /// `observer o(a: particle) -> unit`: a per-particle quantity of any scalar
    /// dimension, which an `observe` reduces over a particle set (spec M6.2 §4.1). It
    /// reads one particle's members and nothing of the chemistry.
    Observer,
}

impl LawKind {
    fn parse(text: &str) -> Option<LawKind> {
        match text {
            "force" => Some(LawKind::Force),
            "potential" => Some(LawKind::Potential),
            "observer" => Some(LawKind::Observer),
            _ => None,
        }
    }

    /// The type a law of this kind returns, or `None` for an observer, which returns a
    /// scalar of whatever dimension its header declares.
    pub fn returns(self) -> Option<Ty> {
        match self {
            LawKind::Force => Some(Ty::Vec2(Dimension::FORCE)),
            LawKind::Potential => Some(Ty::Scalar(Dimension::ENERGY)),
            LawKind::Rate => Some(Ty::Scalar(Dimension::AREAL_REACTION_RATE)),
            LawKind::Observer => None,
        }
    }

    fn written_return(self) -> &'static str {
        match self {
            LawKind::Force => "vec2<newton>",
            LawKind::Potential => "joule",
            LawKind::Rate => "mole / (meter^2 second)",
            LawKind::Observer => "joule",
        }
    }
}

/// Law kinds the specification names whose turn has not come, as `(kind, milestone)`.
const PLANNED_KINDS: &[(&str, &str)] = &[];

/// What a law reads from a particle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Member {
    /// `.position`, m.
    Position,
    /// `.velocity`, m/s.
    Velocity,
    /// `.mass`, kg.
    Mass,
    /// `.charge`, C.
    Charge,
}

impl Member {
    const ALL: [Member; 4] = [Member::Position, Member::Velocity, Member::Mass, Member::Charge];

    fn name(self) -> &'static str {
        match self {
            Member::Position => "position",
            Member::Velocity => "velocity",
            Member::Mass => "mass",
            Member::Charge => "charge",
        }
    }

    fn ty(self) -> Ty {
        match self {
            Member::Position => Ty::Vec2(Dimension::LENGTH),
            Member::Velocity => Ty::Vec2(Dimension::VELOCITY),
            Member::Mass => Ty::Scalar(Dimension::MASS),
            Member::Charge => Ty::Scalar(Dimension::CHARGE),
        }
    }
}

/// The functions a law may call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Builtin {
    /// `sqrt(x)`, of any dimension whose exponents are even.
    Sqrt,
    /// `abs(x)`
    Abs,
    /// `min(x, y)`
    Min,
    /// `max(x, y)`
    Max,
    /// `clamp(x, lo, hi)`
    Clamp,
    /// `exp(x)`, dimensionless.
    Exp,
    /// `ln(x)`, dimensionless.
    Ln,
    /// `sin(x)`, dimensionless (radians).
    Sin,
    /// `cos(x)`
    Cos,
    /// `tan(x)`
    Tan,
    /// `atan2(y, x)`, any shared dimension; the result is an angle.
    Atan2,
    /// `erfc(x)`, dimensionless.
    Erfc,
    /// `pow(x, y)`, both dimensionless — a real exponent has no dimension to give.
    Pow,
    /// `vec2(x, y)`
    Vec2,
    /// `length(v)`
    Length,
    /// `dot(u, v)`
    Dot,
    /// `cross(u, v)`, the scalar `u.x v.y − u.y v.x`.
    Cross,
    /// `normalize(v)`, dimensionless.
    Normalize,
    /// `minimum_image(dx)`: the separation through the nearest periodic image.
    MinimumImage,
}

impl Builtin {
    const ALL: [Builtin; 19] = [
        Builtin::Sqrt,
        Builtin::Abs,
        Builtin::Min,
        Builtin::Max,
        Builtin::Clamp,
        Builtin::Exp,
        Builtin::Ln,
        Builtin::Sin,
        Builtin::Cos,
        Builtin::Tan,
        Builtin::Atan2,
        Builtin::Erfc,
        Builtin::Pow,
        Builtin::Vec2,
        Builtin::Length,
        Builtin::Dot,
        Builtin::Cross,
        Builtin::Normalize,
        Builtin::MinimumImage,
    ];

    /// The name a law calls it by.
    pub fn name(self) -> &'static str {
        match self {
            Builtin::Sqrt => "sqrt",
            Builtin::Abs => "abs",
            Builtin::Min => "min",
            Builtin::Max => "max",
            Builtin::Clamp => "clamp",
            Builtin::Exp => "exp",
            Builtin::Ln => "ln",
            Builtin::Sin => "sin",
            Builtin::Cos => "cos",
            Builtin::Tan => "tan",
            Builtin::Atan2 => "atan2",
            Builtin::Erfc => "erfc",
            Builtin::Pow => "pow",
            Builtin::Vec2 => "vec2",
            Builtin::Length => "length",
            Builtin::Dot => "dot",
            Builtin::Cross => "cross",
            Builtin::Normalize => "normalize",
            Builtin::MinimumImage => "minimum_image",
        }
    }

    fn parse(name: &str) -> Option<Builtin> {
        Builtin::ALL.into_iter().find(|b| b.name() == name)
    }

    fn arity(self) -> usize {
        match self {
            Builtin::Clamp => 3,
            Builtin::Min | Builtin::Max | Builtin::Atan2 | Builtin::Pow | Builtin::Vec2 | Builtin::Dot | Builtin::Cross => 2,
            _ => 1,
        }
    }
}

/// A checked expression: what it computes, its type, and where it was written.
#[derive(Clone, PartialEq, Debug)]
pub struct TExpr {
    /// The operation.
    pub kind: TExprKind,
    /// Its type.
    pub ty: Ty,
    /// The source it came from, for diagnostics at run time (§17.3).
    pub span: Span,
}

/// The operations of a checked law. Every unit has been folded into a [`TExprKind::Const`]
/// in SI, so nothing here knows about units.
#[derive(Clone, PartialEq, Debug)]
pub enum TExprKind {
    /// An SI number.
    Const(f64),
    /// The `n`th `let` of the body.
    Local(usize),
    /// The `n`th `param`, bound where the law is used.
    Param(usize),
    /// A member of the `n`th particle in the header.
    Field {
        /// Which particle.
        particle: usize,
        /// Which member.
        member: Member,
    },
    /// The separation of the header's two particles, through the nearest image.
    Distance,
    /// In a rate law, `c(X)`: the `n`th species' concentration, floored at zero.
    Concentration(usize),
    /// In a rate law, the cell's temperature.
    Temperature,
    /// In a rate law, `arrhenius(Ea)`: `exp(−Ea / R T)` at the cell's temperature, or
    /// `1` where the temperature is not a positive finite number, as the built-in
    /// Arrhenius factor does.
    Arrhenius(Box<TExpr>),
    /// `.x` (0) or `.y` (1) of a vector.
    Component(Box<TExpr>, usize),
    /// `-x`
    Neg(Box<TExpr>),
    /// `!x`
    Not(Box<TExpr>),
    /// An arithmetic, comparison or logical operator.
    Binary(BinaryOp, Box<TExpr>, Box<TExpr>),
    /// `x^n`
    Power(Box<TExpr>, i32),
    /// A call to a built-in function.
    Call(Builtin, Vec<TExpr>),
    /// `if c { a } else { b }`
    Select(Box<TExpr>, Box<TExpr>, Box<TExpr>),
}

/// A `param` of a law: a value bound where the law is used.
#[derive(Clone, PartialEq, Debug)]
pub struct LawParamInfo {
    /// Its name.
    pub name: String,
    /// Its type: a scalar or vector of the declared dimension.
    pub ty: Ty,
    /// The SI value used when the use site gives none.
    pub default: Option<Vec<f64>>,
    /// The declaration.
    pub span: Span,
}

/// A `let` inside a law.
#[derive(Clone, PartialEq, Debug)]
pub struct LawLocal {
    /// Its name.
    pub name: String,
    /// Its value.
    pub value: TExpr,
}

/// A law that has been checked: everything later stages need, and no source.
#[derive(Clone, PartialEq, Debug)]
pub struct TypedLaw {
    /// What kind of law.
    pub kind: LawKind,
    /// Its name.
    pub name: String,
    /// The particles in its header, in order.
    pub particles: Vec<String>,
    /// Its `param`s, in declaration order.
    pub params: Vec<LawParamInfo>,
    /// Its `let`s, in order; each may use those before it.
    pub locals: Vec<LawLocal>,
    /// The returned value.
    pub result: TExpr,
    /// True when anything reads a velocity — which rules out energy conservation.
    pub reads_velocity: bool,
    /// True when anything reads a mass or a charge, so the law differs from pair to pair
    /// and nothing about it can be sampled without the particles' own values.
    pub reads_members: bool,
    /// True when a rate law reads the cell's temperature, directly or through
    /// `arrhenius`.
    pub reads_temperature: bool,
    /// The unit the header declared when it is a bare name (`joule`, `meter`), for labelling
    /// what an observer law records; `None` for a compound unit or a law with no header.
    pub declared_unit: Option<String>,
    /// The declaration.
    pub span: Span,
}

impl TypedLaw {
    /// True for a law between two particles.
    pub fn is_pair(&self) -> bool {
        self.particles.len() == 2
    }
}

/// Check every law in a project.
///
/// A law that fails is left out of the result after its problems are reported; the
/// others are still checked, so one compile reports every problem.
pub fn check_laws(project: &Project, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Vec<TypedLaw> {
    let mut checked: Vec<TypedLaw> = Vec::new();
    let mut seen: Vec<(&str, Span)> = Vec::new();
    for law in project.laws() {
        if let Some((_, previous)) = seen.iter().find(|(name, _)| *name == law.name.text) {
            diagnostics.push(
                Diagnostic::error(format!("law `{}` is declared twice", law.name.text))
                    .with_code("E0201")
                    .at(law.name.span, "declared again here")
                    .also(*previous, "first declared here"),
            );
            continue;
        }
        seen.push((&law.name.text, law.name.span));
        if let Some(typed) = check_law(law, evaluator, diagnostics) {
            checked.push(typed);
        }
    }
    checked
}

fn check_law(law: &LawDecl, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<TypedLaw> {
    let Some(kind) = LawKind::parse(&law.kind.text) else {
        let diagnostic = match PLANNED_KINDS.iter().find(|(name, _)| *name == law.kind.text) {
            Some((name, milestone)) => Diagnostic::error(format!("`{name}` laws are not implemented yet"))
                .with_code("E0900")
                .at(law.kind.span, format!("arrives in {milestone}"))
                .note(format!("this is milestone {milestone}")),
            None => Diagnostic::error(format!("`{}` is not a kind of law", law.kind.text))
                .with_code("E0205")
                .at(law.kind.span, "unknown law kind")
                .help("the kinds of law are `force`, `potential` and `observer`"),
        };
        diagnostics.push(diagnostic);
        return None;
    };

    let mut checker = Checker::new(evaluator, diagnostics, kind, &[]);
    checker.header(law);
    let result = checker.body(law);
    let returns = checker.declared_return(law);

    let (Some(mut result), Some(returns)) = (result, returns) else { return None };
    checker.coerce_zero(&mut result, returns);
    if result.ty != returns {
        checker.diagnostics.push(
            Diagnostic::error(format!("`{}` returns the wrong type", law.name.text))
                .with_code("E0411")
                .at(result.span, format!("this is {}", result.ty.describe()))
                .also(law.returns.as_ref().map_or(law.name.span, |r| r.span), format!("declared {}", returns.describe()))
                .note("the returned value's dimension is worked out from its units, and must be the declared one"),
        );
        return None;
    }
    if checker.failed || law.recovered {
        return None;
    }
    Some(TypedLaw {
        kind,
        name: law.name.text.clone(),
        particles: checker.particles.iter().map(|p| p.text.clone()).collect(),
        params: checker.params,
        locals: checker.locals,
        result,
        reads_velocity: checker.reads_velocity,
        reads_members: checker.reads_members,
        reads_temperature: false,
        declared_unit: law.returns.as_ref().and_then(|r| match &r.kind {
            TypeKind::Plain(expr) => expr.as_name().map(str::to_string),
            TypeKind::Vec2(_) => None,
        }),
        span: law.span,
    })
}

/// Check a reaction's `rate:` written as a law (spec §8.3): `k * c(H_plus) * c(OH_minus)`.
///
/// `species` names the mixture's species in the network's order, which is the order
/// `c(…)` indexes. The law must come out in mol/(m²·s), a reaction rate per unit area.
/// Returns a [`TypedLaw`] of kind [`LawKind::Rate`] with no particles, `param`s or
/// `let`s, named after the reaction.
pub fn check_rate(
    reaction: &Ident,
    value: &Expr,
    species: &[String],
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<TypedLaw> {
    let mut checker = Checker::new(evaluator, diagnostics, LawKind::Rate, species);
    let result = checker.expr(value);
    let expected = LawKind::Rate.returns().expect("a rate law returns a rate");
    let mut result = result?;
    checker.coerce_zero(&mut result, expected);
    if result.ty != expected {
        checker.error(
            Diagnostic::error(format!("the rate of `{}` is not a reaction rate", reaction.text))
                .with_code("E0400")
                .at(result.span, format!("this is {}", result.ty.describe()))
                .note(
                    "a `rate:` that reads concentrations is the reaction's rate itself, in \
                     mol/(m^2 s): how many moles of extent per square metre per second",
                )
                .help("a second-order law is `k * c(A) * c(B)` with `k` in meter^2 / (mole second)"),
        );
    }
    if checker.failed {
        return None;
    }
    Some(TypedLaw {
        kind: LawKind::Rate,
        name: reaction.text.clone(),
        particles: Vec::new(),
        params: Vec::new(),
        locals: Vec::new(),
        result,
        reads_velocity: false,
        reads_members: false,
        reads_temperature: checker.reads_temperature,
        declared_unit: None,
        span: value.span,
    })
}

struct Checker<'e, 'a, 'd> {
    evaluator: &'e Evaluator<'a>,
    diagnostics: &'d mut Diagnostics,
    kind: LawKind,
    particles: Vec<Ident>,
    params: Vec<LawParamInfo>,
    locals: Vec<LawLocal>,
    /// A rate law's species, in the order `c(…)` indexes.
    species: Vec<String>,
    reads_velocity: bool,
    reads_members: bool,
    reads_temperature: bool,
    /// Set by any error, so a law that reported one is not returned half-checked.
    failed: bool,
}

impl<'e, 'a, 'd> Checker<'e, 'a, 'd> {
    fn new(evaluator: &'e Evaluator<'a>, diagnostics: &'d mut Diagnostics, kind: LawKind, species: &[String]) -> Self {
        Checker {
            evaluator,
            diagnostics,
            kind,
            particles: Vec::new(),
            params: Vec::new(),
            locals: Vec::new(),
            species: species.to_vec(),
            reads_velocity: false,
            reads_members: false,
            reads_temperature: false,
            failed: false,
        }
    }
}

impl Checker<'_, '_, '_> {
    fn error(&mut self, diagnostic: Diagnostic) {
        self.failed = true;
        self.diagnostics.push(diagnostic);
    }

    fn is_pair(&self) -> bool {
        self.particles.len() == 2
    }

    // --- the header ----------------------------------------------------------

    fn header(&mut self, law: &LawDecl) {
        let article = if self.kind == LawKind::Observer { "an" } else { "a" };
        for param in &law.params {
            let is_particle =
                matches!(&param.ty.kind, TypeKind::Plain(expr) if expr.as_name() == Some("particle"));
            if !is_particle {
                self.error(
                    Diagnostic::error(format!("{article} {} takes particles", law.kind.text))
                        .with_code("E0411")
                        .at(param.ty.span, "expected `particle`")
                        .help("values the law needs from its use site are declared in the body with `param`"),
                );
                continue;
            }
            if self.particles.iter().any(|p| p.text == param.name.text) {
                self.error(
                    Diagnostic::error(format!("`{}` names two parameters", param.name.text))
                        .with_code("E0201")
                        .span(param.name.span),
                );
                continue;
            }
            self.particles.push(param.name.clone());
        }
        if self.kind == LawKind::Observer {
            // Pair observers need the neighbour list and the cutoff semantics of pair
            // laws; they are the next step, so the second particle is refused by name.
            if self.particles.len() >= 2 {
                self.error(
                    Diagnostic::error("observer laws take one particle; pair observers are not implemented")
                        .with_code("E0416")
                        .at(self.particles[1].span, "a second particle")
                        .note("an observer is evaluated once per particle of a set, and reduced over the set")
                        .help(format!(
                            "write `observer {}({}: particle)`",
                            law.name.text, self.particles[0].text
                        )),
                );
            } else if law.params.is_empty() {
                self.error(
                    Diagnostic::error("an observer acts on one particle")
                        .with_code("E0411")
                        .at(law.name.span, "no parameters")
                        .help(format!("write `observer {}(a: particle) -> joule`", law.name.text)),
                );
            }
            return;
        }
        if !(1..=2).contains(&law.params.len()) {
            let found = law.params.len();
            self.error(
                Diagnostic::error(format!("a {} acts on one particle or between two", law.kind.text))
                    .with_code("E0411")
                    .at(law.name.span, format!("{found} parameters"))
                    .help(format!(
                        "write `{} {}(a: particle)` for a force on each particle, or `(a: particle, b: particle)` for a pair",
                        law.kind.text, law.name.text
                    )),
            );
        }
    }

    fn declared_return(&mut self, law: &LawDecl) -> Option<Ty> {
        let Some(written) = &law.returns else {
            self.error(
                Diagnostic::error(format!("`{}` does not say what it returns", law.name.text))
                    .with_code("E0411")
                    .at(law.name.span, "no return type")
                    .help(if self.kind == LawKind::Observer {
                        "an observer returns a scalar and names its unit: `-> joule`, or `-> 1` for a plain number"
                            .to_string()
                    } else {
                        format!("a {} returns `-> {}`", law.kind.text, self.kind.written_return())
                    }),
            );
            return None;
        };
        let ty = self.type_of(written)?;
        let Some(expected) = self.kind.returns() else {
            // An observer: any scalar, in the unit it declares.
            return match ty {
                Ty::Scalar(_) => Some(ty),
                Ty::Vec2(_) => {
                    self.error(
                        Diagnostic::error("an observer returns a scalar, not a vector")
                            .with_code("E0417")
                            .at(written.span, format!("declared {}", ty.describe()))
                            .note("it is reduced over a particle set, and a reduction of vectors is not defined")
                            .help("observe each component separately: one observer returning `v.x`, another `v.y`"),
                    );
                    None
                }
                Ty::Bool | Ty::Particle => {
                    self.error(
                        Diagnostic::error(format!("an observer returns a scalar quantity, not {}", ty.describe()))
                            .with_code("E0411")
                            .at(written.span, format!("declared {}", ty.describe()))
                            .help("write a unit, such as `-> joule`, or `-> 1` for a plain number"),
                    );
                    None
                }
            };
        };
        if ty != expected {
            self.error(
                Diagnostic::error(format!("a {} returns {}", law.kind.text, expected.describe()))
                    .with_code("E0411")
                    .at(written.span, format!("declared {}", ty.describe()))
                    .help(format!("write `-> {}`", self.kind.written_return())),
            );
            return None;
        }
        Some(ty)
    }

    /// A written type: `particle`, `bool`, a unit expression, or `vec2<unit>`.
    fn type_of(&mut self, written: &TypeExpr) -> Option<Ty> {
        match &written.kind {
            TypeKind::Plain(expr) if expr.as_name() == Some("particle") => Some(Ty::Particle),
            TypeKind::Plain(expr) if expr.as_name() == Some("bool") => Some(Ty::Bool),
            TypeKind::Plain(expr) => {
                let value = self.evaluator.quantity(expr, self.diagnostics)?;
                Some(Ty::Scalar(value.dimension()))
            }
            TypeKind::Vec2(expr) => {
                let value = self.evaluator.quantity(expr, self.diagnostics)?;
                Some(Ty::Vec2(value.dimension()))
            }
        }
    }

    // --- the body ------------------------------------------------------------

    fn body(&mut self, law: &LawDecl) -> Option<TExpr> {
        let mut result = None;
        for stmt in &law.body {
            if result.is_some() {
                self.error(
                    Diagnostic::error("nothing can follow `return`")
                        .with_code("E0415")
                        .at(stmt.span, "after the return")
                        .help("a law's `return` is its last statement"),
                );
                break;
            }
            match &stmt.kind {
                StmtKind::Let { name, value } => {
                    self.check_new_name(name);
                    let Some(value) = self.expr(value) else { continue };
                    if value.ty == Ty::Particle {
                        self.error(
                            Diagnostic::error("a particle cannot be given another name")
                                .with_code("E0412")
                                .span(value.span),
                        );
                        continue;
                    }
                    self.locals.push(LawLocal { name: name.text.clone(), value });
                }
                StmtKind::Param { name, ty, default } => {
                    // A pair law's use site binds `cutoff=` and `truncation=` itself; a
                    // `param` of either name would capture the argument and leave the law
                    // with no cutoff, unusable.
                    if self.is_pair() && matches!(name.text.as_str(), "cutoff" | "truncation") {
                        self.error(
                            Diagnostic::error(format!("`{}` is reserved in a pair law", name.text))
                                .with_code("E0201")
                                .at(name.span, "bound by the use site")
                                .note("a pair law's use site writes `cutoff=` and `truncation=` for the engine")
                                .help("give the parameter another name"),
                        );
                        continue;
                    }
                    self.check_new_name(name);
                    // The evaluator reports a bad unit itself, but only the checker knows
                    // the law has failed: a law with an unreadable `param` is not one
                    // that "checks".
                    let Some(ty) = self.type_of(ty) else {
                        self.failed = true;
                        continue;
                    };
                    if !matches!(ty, Ty::Scalar(_) | Ty::Vec2(_)) {
                        self.error(
                            Diagnostic::error("a `param` is a quantity")
                                .with_code("E0411")
                                .at(stmt.span, format!("declared {}", ty.describe())),
                        );
                        continue;
                    }
                    // A default that fails is a failed law, not a `param` without one.
                    // The `param` is still declared, so its uses do not cascade into
                    // errors of their own.
                    let default = match default {
                        Some(value) => {
                            let default = self.default_of(value, ty);
                            self.failed |= default.is_none();
                            default
                        }
                        None => None,
                    };
                    self.params.push(LawParamInfo { name: name.text.clone(), ty, default, span: name.span });
                }
                // A return that failed was reported already; stop here rather than
                // also saying the return is missing.
                StmtKind::Return(value) => result = Some(self.expr(value)?),
            }
        }
        if result.is_none() && !self.failed && !law.recovered {
            self.error(
                Diagnostic::error(format!("`{}` never returns a value", law.name.text))
                    .with_code("E0415")
                    .at(law.name.span, "no `return`")
                    .help("end the body with `return <value>;`"),
            );
        }
        result
    }

    /// A `param`'s default, folded to SI. A vector default is written the way a
    /// vector is everywhere else in a law, `vec2(x, y)`.
    fn default_of(&mut self, value: &Expr, ty: Ty) -> Option<Vec<f64>> {
        match ty {
            Ty::Scalar(d) => Some(vec![self.evaluator.require(value, d, "the default", self.diagnostics)?]),
            Ty::Vec2(d) => {
                let components = match &value.kind {
                    ExprKind::Call(callee, arguments)
                        if callee.as_name() == Some("vec2")
                            && arguments.len() == 2
                            && arguments.iter().all(|a| a.name.is_none()) =>
                    {
                        [&arguments[0].value, &arguments[1].value]
                    }
                    _ => {
                        self.diagnostics.push(
                            Diagnostic::error("a vector default is written `vec2(x, y)`")
                                .with_code("E0410")
                                .at(value.span, format!("the `param` is {}", ty.describe())),
                        );
                        return None;
                    }
                };
                let x = self.evaluator.require(components[0], d, "the default (x)", self.diagnostics);
                let y = self.evaluator.require(components[1], d, "the default (y)", self.diagnostics);
                Some(vec![x?, y?])
            }
            _ => None,
        }
    }

    /// A `let` or `param` name must be new within the law; shadowing a unit is legal
    /// but warned about, because every later use of the unit in this law changes
    /// meaning.
    fn check_new_name(&mut self, name: &Ident) {
        let taken = self.locals.iter().any(|l| l.name == name.text)
            || self.params.iter().any(|p| p.name == name.text)
            || self.particles.iter().any(|p| p.text == name.text);
        if taken {
            self.error(
                Diagnostic::error(format!("`{}` is already defined in this law", name.text))
                    .with_code("E0201")
                    .span(name.span)
                    .note("a law's names cannot be redefined: each `let` names a new value"),
            );
            return;
        }
        if self.evaluator.units().resolve(&name.text).is_ok() {
            self.diagnostics.push(shadow_warning(name));
        }
    }

    // --- expressions ---------------------------------------------------------

    fn expr(&mut self, expr: &Expr) -> Option<TExpr> {
        let span = expr.span;
        let typed = |kind, ty| Some(TExpr { kind, ty, span });
        match &expr.kind {
            ExprKind::Number(value) => typed(TExprKind::Const(*value), Ty::Scalar(Dimension::DIMENSIONLESS)),
            ExprKind::Bool(value) => typed(TExprKind::Const(f64::from(u8::from(*value))), Ty::Bool),
            ExprKind::Name(name) => self.name(name, span),
            ExprKind::Member(receiver, member) => self.member(receiver, member, span),
            ExprKind::Call(callee, arguments) => self.call(callee, arguments, span),
            ExprKind::Unary(op, inner) => {
                let inner = self.expr(inner)?;
                match (op, inner.ty) {
                    (UnaryOp::Pos, Ty::Scalar(_) | Ty::Vec2(_)) => Some(TExpr { span, ..inner }),
                    (UnaryOp::Neg, Ty::Scalar(_) | Ty::Vec2(_)) => typed(TExprKind::Neg(Box::new(inner.clone())), inner.ty),
                    (UnaryOp::Not, Ty::Bool) => typed(TExprKind::Not(Box::new(inner)), Ty::Bool),
                    (UnaryOp::Not, other) => {
                        self.not_a_condition(&inner, other);
                        None
                    }
                    (_, other) => {
                        self.error(
                            Diagnostic::error(format!("cannot negate {}", other.describe()))
                                .with_code("E0410")
                                .span(inner.span),
                        );
                        None
                    }
                }
            }
            ExprKind::Binary(op, left, right) => {
                // Both sides are checked even when one fails, so two problems are two
                // diagnostics.
                let l = self.expr(left);
                let r = self.expr(right);
                let (mut l, mut r) = (l?, r?);
                self.binary(*op, &mut l, &mut r, (left, right), span)
            }
            ExprKind::Power(base, exponent) => {
                let base = self.expr(base)?;
                let Ty::Scalar(d) = base.ty else {
                    self.error(
                        Diagnostic::error(format!("only a scalar can be raised to a power, not {}", base.ty.shape()))
                            .with_code("E0410")
                            .at(base.span, base.ty.describe())
                            .help("for a vector's squared length, write `dot(v, v)`"),
                    );
                    return None;
                };
                let d = self.dimension(d.try_powi(*exponent), span)?;
                typed(TExprKind::Power(Box::new(base), *exponent), Ty::Scalar(d))
            }
            ExprKind::If(condition, then, otherwise) => {
                let c = self.expr(condition);
                let a = self.expr(then);
                let b = self.expr(otherwise);
                let (c, mut a, mut b) = (c?, a?, b?);
                if c.ty != Ty::Bool {
                    self.not_a_condition(&c, c.ty);
                    return None;
                }
                self.unify_zero(&mut a, &mut b, (then, otherwise));
                if a.ty != b.ty {
                    self.mismatch("the two arms of an `if` must have the same type", &a, &b, span);
                    return None;
                }
                let ty = a.ty;
                typed(TExprKind::Select(Box::new(c), Box::new(a), Box::new(b)), ty)
            }
            ExprKind::Str(_) | ExprKind::List(_) | ExprKind::Tuple(_) | ExprKind::Yields(..) => {
                let what = match &expr.kind {
                    ExprKind::Str(_) => "a string",
                    ExprKind::Yields(..) => "a reaction equation",
                    _ => "a list",
                };
                self.error(
                    Diagnostic::error(format!("{what} has no meaning inside a law"))
                        .with_code("E0410")
                        .span(span)
                        .help("a vector is written `vec2(x, y)`"),
                );
                None
            }
        }
    }

    fn name(&mut self, name: &str, span: Span) -> Option<TExpr> {
        if let Some(index) = self.locals.iter().rposition(|l| l.name == name) {
            let ty = self.locals[index].value.ty;
            return Some(TExpr { kind: TExprKind::Local(index), ty, span });
        }
        if let Some(index) = self.params.iter().position(|p| p.name == name) {
            return Some(TExpr { kind: TExprKind::Param(index), ty: self.params[index].ty, span });
        }
        if self.kind == LawKind::Observer && name == "temperature" && !self.evaluator.is_constant(name) {
            self.error(observer_reads_particles_only(name, span));
            return None;
        }
        if self.kind == LawKind::Rate && name == "temperature" {
            if self.evaluator.is_constant(name) {
                // The model's own constant wins, as declared names do everywhere; but a
                // rate law that reads concentrations probably meant the cell's.
                self.diagnostics.push(
                    Diagnostic::warning("`temperature` here is the model's constant, not the cell's temperature")
                        .with_code("W0319")
                        .at(span, "the `let temperature` constant")
                        .note("a declared constant hides the state of the same name, in a rate law as anywhere else")
                        .help("rename the constant if the law should follow the cell's temperature"),
                );
            } else {
                self.reads_temperature = true;
                return Some(TExpr { kind: TExprKind::Temperature, ty: Ty::Scalar(Dimension::TEMPERATURE), span });
            }
        }
        if self.kind == LawKind::Rate && self.species.iter().any(|s| s == name) && !self.evaluator.is_constant(name) {
            self.error(
                Diagnostic::error(format!("`{name}` is a species, not a value"))
                    .with_code("E0412")
                    .at(span, "a species")
                    .help(format!("its concentration is `c({name})`")),
            );
            return None;
        }
        if self.particles.iter().any(|p| p.text == name) {
            self.error(
                Diagnostic::error(format!("`{name}` is a particle, not a value"))
                    .with_code("E0412")
                    .at(span, "a particle")
                    .help(format!(
                        "read one of its members, such as `{name}.position`, or pass both particles to `distance`"
                    )),
            );
            return None;
        }
        // A project constant or a unit: the evaluator knows both, and folds the name
        // to an SI magnitude.
        let quantity = self.evaluator.quantity(&Expr::new(ExprKind::Name(name.to_string()), span), self.diagnostics);
        let Some(quantity) = quantity else {
            self.failed = true;
            return None;
        };
        Some(TExpr { kind: TExprKind::Const(quantity.value()), ty: Ty::Scalar(quantity.dimension()), span })
    }

    fn particle_index(&self, expr: &Expr) -> Option<usize> {
        let name = expr.as_name()?;
        if self.locals.iter().any(|l| l.name == name) || self.params.iter().any(|p| p.name == name) {
            return None;
        }
        self.particles.iter().position(|p| p.text == name)
    }

    fn member(&mut self, receiver: &Expr, member: &Ident, span: Span) -> Option<TExpr> {
        if let Some(particle) = self.particle_index(receiver) {
            let Some(found) = Member::ALL.into_iter().find(|m| m.name() == member.text) else {
                let known: Vec<&str> = Member::ALL.iter().map(|m| m.name()).collect();
                self.error(
                    Diagnostic::error(format!("a particle has no `{}`", member.text))
                        .with_code("E0412")
                        .span(member.span)
                        .help(format!("a particle's members are {}", known.join(", "))),
                );
                return None;
            };
            if found == Member::Velocity {
                if self.kind == LawKind::Potential {
                    self.error(
                        Diagnostic::error("a potential cannot read a velocity")
                            .with_code("E0412")
                            .at(span, "velocity")
                            .note("an energy that depends on velocity does not define a conservative force")
                            .help("a velocity-dependent law is a `force`"),
                    );
                    return None;
                }
                self.reads_velocity = true;
            }
            if matches!(found, Member::Mass | Member::Charge) {
                self.reads_members = true;
            }
            if found == Member::Position && self.kind == LawKind::Potential && self.is_pair() {
                self.error(
                    Diagnostic::error("a pair potential reads positions only through `distance(a, b)`")
                        .with_code("E0412")
                        .at(span, "a position")
                        .note(
                            "an energy of the separation alone gives a central force that conserves \
                             energy, momentum and angular momentum; one of the positions themselves \
                             promises none of that",
                        )
                        .help(format!("write `distance({}, {})`", self.particles[0].text, self.particles[1].text)),
                );
                return None;
            }
            return Some(TExpr { kind: TExprKind::Field { particle, member: found }, ty: found.ty(), span });
        }

        let value = self.expr(receiver)?;
        let component = match member.text.as_str() {
            "x" => 0,
            "y" => 1,
            other => {
                self.error(
                    Diagnostic::error(format!("`.{other}` is not something a value has"))
                        .with_code("E0412")
                        .span(member.span)
                        .help("a vector's components are `.x` and `.y`"),
                );
                return None;
            }
        };
        let Ty::Vec2(d) = value.ty else {
            self.error(
                Diagnostic::error(format!("`.{}` reads a vector's component, and this is {}", member.text, value.ty.shape()))
                    .with_code("E0410")
                    .span(value.span),
            );
            return None;
        };
        Some(TExpr { kind: TExprKind::Component(Box::new(value), component), ty: Ty::Scalar(d), span })
    }

    fn call(&mut self, callee: &Expr, arguments: &[lattice_syntax::Argument], span: Span) -> Option<TExpr> {
        let Some(name) = callee.as_name() else {
            self.error(Diagnostic::error("only a named function can be called").with_code("E0413").span(callee.span));
            return None;
        };

        // An observer reads particles and has no chemistry context. A `let` or `param`
        // that happens to be called `c` is a value, and is told so below instead.
        if self.kind == LawKind::Observer && matches!(name, "c" | "arrhenius") {
            let is_value = self.locals.iter().any(|l| l.name == name) || self.params.iter().any(|p| p.name == name);
            if !is_value {
                self.error(observer_reads_particles_only(name, callee.span));
                return None;
            }
        }

        // A rate law's two readers of the state, which no other law has.
        match (name, self.kind == LawKind::Rate) {
            ("c", true) => return self.concentration(arguments, span),
            ("arrhenius", true) => return self.arrhenius(arguments, span),
            ("c" | "arrhenius", false) if self.kind != LawKind::Observer => {
                self.error(
                    Diagnostic::error(format!("`{name}` belongs to a reaction's rate law"))
                        .with_code("E0413")
                        .at(callee.span, format!("in a {}", if self.kind == LawKind::Force { "force" } else { "potential" }))
                        .note("concentrations and the temperature are read in a reaction's `rate:`, not on particles"),
                );
                return None;
            }
            ("distance" | "minimum_image", true) => {
                self.error(
                    Diagnostic::error(format!("`{name}` reads particles, and a rate law has none"))
                        .with_code("E0412")
                        .span(callee.span)
                        .note("a rate law reads concentrations through `c(…)` and the cell's `temperature`"),
                );
                return None;
            }
            _ => {}
        }

        // `distance(a, b)` takes the header's particles themselves.
        if name == "distance" {
            return self.distance(arguments, span);
        }

        let Some(builtin) = Builtin::parse(name) else {
            let is_value = self.locals.iter().any(|l| l.name == name) || self.params.iter().any(|p| p.name == name);
            let diagnostic = if is_value {
                Diagnostic::error(format!("`{name}` is a value, not a function"))
                    .with_code("E0413")
                    .at(callee.span, "followed by `(`")
                    .note("a name followed by `(` is a call, even with a space between them")
                    .help(format!("to multiply, write `{name} * (…)`"))
            } else {
                let mut known: Vec<&str> = Builtin::ALL
                    .iter()
                    .filter(|b| self.kind != LawKind::Rate || **b != Builtin::MinimumImage)
                    .map(|b| b.name())
                    .collect();
                if self.kind == LawKind::Rate {
                    known.extend(["c", "arrhenius"]);
                } else if self.kind != LawKind::Observer {
                    known.push("distance");
                }
                Diagnostic::error(format!("`{name}` is not a function a law can call"))
                    .with_code("E0413")
                    .span(callee.span)
                    .note(format!("the functions are {}", known.join(", ")))
            };
            self.error(diagnostic);
            return None;
        };

        if let Some(named) = arguments.iter().find_map(|a| a.name.as_ref()) {
            self.error(
                Diagnostic::error(format!("`{}` takes its arguments in order, not by name", builtin.name()))
                    .with_code("E0413")
                    .span(named.span),
            );
            return None;
        }
        if arguments.len() != builtin.arity() {
            self.error(
                Diagnostic::error(format!(
                    "`{}` takes {} argument{}, found {}",
                    builtin.name(),
                    builtin.arity(),
                    if builtin.arity() == 1 { "" } else { "s" },
                    arguments.len()
                ))
                .with_code("E0413")
                .span(span),
            );
            return None;
        }
        let mut args = Vec::with_capacity(arguments.len());
        for argument in arguments {
            args.push(self.expr(&argument.value));
        }
        let mut args: Vec<TExpr> = args.into_iter().collect::<Option<_>>()?;
        let ty = self.builtin_type(builtin, &mut args, arguments, span)?;
        Some(TExpr { kind: TExprKind::Call(builtin, args), ty, span })
    }

    /// `c(X)`: the concentration of species `X` in this cell.
    fn concentration(&mut self, arguments: &[lattice_syntax::Argument], span: Span) -> Option<TExpr> {
        let name = match arguments {
            [argument] if argument.name.is_none() => argument.value.as_name(),
            _ => None,
        };
        let Some(name) = name else {
            self.error(
                Diagnostic::error("`c` takes one species")
                    .with_code("E0413")
                    .span(span)
                    .help("write `c(A)` for the concentration of species `A`"),
            );
            return None;
        };
        let Some(index) = self.species.iter().position(|s| s == name) else {
            self.error(
                Diagnostic::error(format!("there is no species called `{name}` in this mixture"))
                    .with_code("E0202")
                    .at(arguments[0].value.span, "unknown species")
                    .help(if self.species.is_empty() {
                        "the mixture has no species".to_string()
                    } else {
                        format!("its species are {}", self.species.join(", "))
                    }),
            );
            return None;
        };
        Some(TExpr { kind: TExprKind::Concentration(index), ty: Ty::Scalar(Dimension::AREAL_CONCENTRATION), span })
    }

    /// `arrhenius(Ea)`: the Arrhenius factor at the cell's temperature.
    fn arrhenius(&mut self, arguments: &[lattice_syntax::Argument], span: Span) -> Option<TExpr> {
        let [argument] = arguments else {
            self.error(
                Diagnostic::error(format!("`arrhenius` takes one argument, found {}", arguments.len()))
                    .with_code("E0413")
                    .span(span)
                    .help("write `arrhenius(50 kilojoule / mole)` for an activation energy"),
            );
            return None;
        };
        if let Some(named) = &argument.name {
            self.error(
                Diagnostic::error("`arrhenius` takes its activation energy in order, not by name")
                    .with_code("E0413")
                    .span(named.span),
            );
            return None;
        }
        let energy = self.expr(&argument.value)?;
        let per_mole = Dimension::ENERGY.try_div(Dimension::AMOUNT).expect("energy per amount is representable");
        if energy.ty != Ty::Scalar(per_mole) {
            self.error(
                Diagnostic::error("`arrhenius` takes an activation energy per mole")
                    .with_code("E0400")
                    .at(energy.span, format!("this is {}", energy.ty.describe()))
                    .help("write it in joule / mole or kilojoule / mole"),
            );
            return None;
        }
        self.reads_temperature = true;
        Some(TExpr {
            kind: TExprKind::Arrhenius(Box::new(energy)),
            ty: Ty::Scalar(Dimension::DIMENSIONLESS),
            span,
        })
    }

    fn distance(&mut self, arguments: &[lattice_syntax::Argument], span: Span) -> Option<TExpr> {
        if !self.is_pair() {
            self.error(
                Diagnostic::error("`distance` is the separation of a pair, and this law acts on one particle")
                    .with_code("E0412")
                    .span(span),
            );
            return None;
        }
        let indices: Vec<Option<usize>> = arguments.iter().map(|a| self.particle_index(&a.value)).collect();
        if indices != [Some(0), Some(1)] && indices != [Some(1), Some(0)] {
            self.error(
                Diagnostic::error("`distance` takes the law's two particles")
                    .with_code("E0413")
                    .span(span)
                    .help(format!("write `distance({}, {})`", self.particles[0].text, self.particles[1].text)),
            );
            return None;
        }
        Some(TExpr { kind: TExprKind::Distance, ty: Ty::Scalar(Dimension::LENGTH), span })
    }

    fn builtin_type(
        &mut self,
        builtin: Builtin,
        args: &mut [TExpr],
        written: &[lattice_syntax::Argument],
        span: Span,
    ) -> Option<Ty> {
        use Builtin as B;
        let dimensionless = Ty::Scalar(Dimension::DIMENSIONLESS);
        match builtin {
            B::Exp | B::Ln | B::Sin | B::Cos | B::Tan | B::Erfc | B::Pow => {
                for (arg, value) in args.iter().zip(written) {
                    if arg.ty != dimensionless {
                        self.error(
                            Diagnostic::error(format!("`{}` takes a plain number", builtin.name()))
                                .with_code(if matches!(arg.ty, Ty::Scalar(_)) { "E0400" } else { "E0410" })
                                .at(value.value.span, format!("this is {}", arg.ty.describe()))
                                .note("a transcendental function of a dimensioned quantity has no unit to give its result")
                                .help("divide by a reference value to make the argument dimensionless"),
                        );
                        return None;
                    }
                }
                Some(dimensionless)
            }
            B::Sqrt => {
                let d = self.scalar_arg(builtin, &args[0])?;
                let root = self.dimension(d.root(2), span)?;
                Some(Ty::Scalar(root))
            }
            B::Abs => Some(Ty::Scalar(self.scalar_arg(builtin, &args[0])?)),
            B::Min | B::Max | B::Clamp | B::Atan2 => {
                let (first, rest) = args.split_first_mut().expect("arity checked");
                for (arg, value) in rest.iter_mut().zip(&written[1..]) {
                    self.unify_zero(first, arg, (&written[0].value, &value.value));
                }
                let d = self.scalar_arg(builtin, &args[0])?;
                for arg in &args[1..] {
                    if arg.ty != Ty::Scalar(d) {
                        self.mismatch(&format!("the arguments of `{}` must have the same dimension", builtin.name()), &args[0], arg, span);
                        return None;
                    }
                }
                Some(if builtin == B::Atan2 { dimensionless } else { Ty::Scalar(d) })
            }
            B::Vec2 => {
                let (x, y) = args.split_at_mut(1);
                self.unify_zero(&mut x[0], &mut y[0], (&written[0].value, &written[1].value));
                let d = self.scalar_arg(builtin, &args[0])?;
                if args[1].ty != Ty::Scalar(d) {
                    self.mismatch("both components of a vector must have the same dimension", &args[0], &args[1], span);
                    return None;
                }
                Some(Ty::Vec2(d))
            }
            B::Length => Some(Ty::Scalar(self.vector_arg(builtin, &args[0])?)),
            B::Normalize => {
                self.vector_arg(builtin, &args[0])?;
                Some(Ty::Vec2(Dimension::DIMENSIONLESS))
            }
            B::Dot | B::Cross => {
                let a = self.vector_arg(builtin, &args[0])?;
                let b = self.vector_arg(builtin, &args[1])?;
                Some(Ty::Scalar(self.dimension(a.try_mul(b), span)?))
            }
            B::MinimumImage => {
                if !self.is_pair() {
                    self.error(
                        Diagnostic::error("`minimum_image` belongs to a law between two particles")
                            .with_code("E0412")
                            .span(span),
                    );
                    return None;
                }
                let d = self.vector_arg(builtin, &args[0])?;
                if d != Dimension::LENGTH {
                    self.error(
                        Diagnostic::error("`minimum_image` takes a separation")
                            .with_code("E0400")
                            .at(args[0].span, format!("this is {}", args[0].ty.describe()))
                            .help("write `minimum_image(b.position - a.position)`"),
                    );
                    return None;
                }
                Some(Ty::Vec2(Dimension::LENGTH))
            }
        }
    }

    fn scalar_arg(&mut self, builtin: Builtin, arg: &TExpr) -> Option<Dimension> {
        if let Ty::Scalar(d) = arg.ty {
            return Some(d);
        }
        self.error(
            Diagnostic::error(format!("`{}` takes a scalar, not {}", builtin.name(), arg.ty.shape()))
                .with_code("E0410")
                .span(arg.span)
                .help("`length(v)` is a vector's magnitude"),
        );
        None
    }

    fn vector_arg(&mut self, builtin: Builtin, arg: &TExpr) -> Option<Dimension> {
        if let Ty::Vec2(d) = arg.ty {
            return Some(d);
        }
        self.error(
            Diagnostic::error(format!("`{}` takes a vector, not {}", builtin.name(), arg.ty.shape()))
                .with_code("E0410")
                .span(arg.span),
        );
        None
    }

    fn binary(&mut self, op: BinaryOp, l: &mut TExpr, r: &mut TExpr, written: (&Expr, &Expr), span: Span) -> Option<TExpr> {
        let ty = self.binary_type(op, l, r, written, span)?;
        Some(TExpr { kind: TExprKind::Binary(op, Box::new(l.clone()), Box::new(r.clone())), ty, span })
    }

    fn binary_type(&mut self, op: BinaryOp, l: &mut TExpr, r: &mut TExpr, written: (&Expr, &Expr), span: Span) -> Option<Ty> {
        let result = Some;
        match op {
            BinaryOp::And | BinaryOp::Or => {
                for side in [&*l, &*r] {
                    if side.ty != Ty::Bool {
                        self.not_a_condition(side, side.ty);
                        return None;
                    }
                }
                result(Ty::Bool)
            }
            BinaryOp::Add | BinaryOp::Sub => {
                self.unify_zero(l, r, written);
                match (l.ty, r.ty) {
                    (Ty::Scalar(a), Ty::Scalar(b)) | (Ty::Vec2(a), Ty::Vec2(b)) if a == b => result(l.ty),
                    (Ty::Scalar(_), Ty::Scalar(_)) | (Ty::Vec2(_), Ty::Vec2(_)) => {
                        let verb = if op == BinaryOp::Add { "add" } else { "subtract" };
                        self.mismatch(&format!("cannot {verb} quantities with different dimensions"), l, r, span);
                        None
                    }
                    _ => {
                        let verb = if op == BinaryOp::Add { "add" } else { "subtract" };
                        self.shape_error(&format!("cannot {verb} {} and {}", l.ty.shape(), r.ty.shape()), l, r, span);
                        None
                    }
                }
            }
            BinaryOp::Mul => match (l.ty, r.ty) {
                (Ty::Scalar(a), Ty::Scalar(b)) => result(Ty::Scalar(self.dimension(a.try_mul(b), span)?)),
                (Ty::Scalar(a), Ty::Vec2(b)) | (Ty::Vec2(a), Ty::Scalar(b)) => {
                    result(Ty::Vec2(self.dimension(a.try_mul(b), span)?))
                }
                (Ty::Vec2(_), Ty::Vec2(_)) => {
                    self.error(
                        Diagnostic::error("two vectors have no single product")
                            .with_code("E0410")
                            .at(span, "vector times vector")
                            .help("write `dot(u, v)` for the scalar product or `cross(u, v)` for the perpendicular one"),
                    );
                    None
                }
                _ => {
                    self.shape_error(&format!("cannot multiply {} and {}", l.ty.shape(), r.ty.shape()), l, r, span);
                    None
                }
            },
            BinaryOp::Div => match (l.ty, r.ty) {
                (Ty::Scalar(a), Ty::Scalar(b)) => result(Ty::Scalar(self.dimension(a.try_div(b), span)?)),
                (Ty::Vec2(a), Ty::Scalar(b)) => result(Ty::Vec2(self.dimension(a.try_div(b), span)?)),
                _ => {
                    self.shape_error(&format!("cannot divide {} by {}", l.ty.shape(), r.ty.shape()), l, r, span);
                    None
                }
            },
            _ => {
                // A comparison.
                self.unify_zero(l, r, written);
                match (l.ty, r.ty) {
                    (Ty::Scalar(a), Ty::Scalar(b)) if a == b => result(Ty::Bool),
                    (Ty::Scalar(_), Ty::Scalar(_)) => {
                        self.mismatch("cannot compare quantities with different dimensions", l, r, span);
                        None
                    }
                    _ => {
                        self.shape_error(
                            &format!("`{}` compares scalars, not {} and {}", op.symbol(), l.ty.shape(), r.ty.shape()),
                            l,
                            r,
                            span,
                        );
                        None
                    }
                }
            }
        }
    }

    /// A literal `0` takes the dimension of whatever it meets, as it does in a
    /// setting: `if r < 0 { … }` and `max(x, 0)` mean what they say.
    fn unify_zero(&self, a: &mut TExpr, b: &mut TExpr, written: (&Expr, &Expr)) {
        if is_literal_zero(written.0) {
            self.coerce_zero(a, b.ty);
        } else if is_literal_zero(written.1) {
            self.coerce_zero(b, a.ty);
        }
    }

    fn coerce_zero(&self, value: &mut TExpr, to: Ty) {
        let is_zero = matches!(value.kind, TExprKind::Const(v) if v == 0.0)
            || matches!(&value.kind, TExprKind::Neg(inner) if matches!(inner.kind, TExprKind::Const(v) if v == 0.0));
        if is_zero && value.ty == Ty::Scalar(Dimension::DIMENSIONLESS) && matches!(to, Ty::Scalar(_)) {
            value.ty = to;
        }
    }

    fn dimension<E: core::fmt::Display>(&mut self, result: Result<Dimension, E>, span: Span) -> Option<Dimension> {
        match result {
            Ok(d) => Some(d),
            Err(error) => {
                self.error(Diagnostic::error(error.to_string()).with_code("E0400").span(span));
                None
            }
        }
    }

    fn mismatch(&mut self, message: &str, a: &TExpr, b: &TExpr, span: Span) {
        self.error(
            Diagnostic::error(message)
                .with_code("E0400")
                .span(span)
                .also(a.span, format!("this is {}", a.ty.describe()))
                .also(b.span, format!("this is {}", b.ty.describe())),
        );
    }

    fn shape_error(&mut self, message: &str, a: &TExpr, b: &TExpr, span: Span) {
        self.error(
            Diagnostic::error(message)
                .with_code("E0410")
                .span(span)
                .also(a.span, format!("this is {}", a.ty.describe()))
                .also(b.span, format!("this is {}", b.ty.describe()))
                .note(
                    "a value in a law is a scalar or a `vec2`; vectors add to vectors, scale by \
                     scalars, and meet each other only through `dot` and `cross`",
                ),
        );
    }

    fn not_a_condition(&mut self, value: &TExpr, ty: Ty) {
        self.error(
            Diagnostic::error(format!("a condition must be a truth value, and this is {}", ty.describe()))
                .with_code("E0414")
                .span(value.span)
                .help("compare it with something, as in `r < 2.5 sigma`"),
        );
    }
}

/// Whether an expression is the number zero as written.
fn is_literal_zero(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Number(value) => *value == 0.0,
        ExprKind::Unary(_, inner) => is_literal_zero(inner),
        _ => false,
    }
}

/// `c(…)`, `temperature` and `arrhenius(…)` in an observer law: it runs on particles, in
/// no cell, so there is no concentration or temperature to read.
fn observer_reads_particles_only(name: &str, span: Span) -> Diagnostic {
    Diagnostic::error("observer laws read particles only")
        .with_code("E0418")
        .at(span, format!("`{name}` reads a cell's chemistry"))
        .note("an observer is evaluated per particle, with no concentrations or temperature in its context")
        .help("read `a.position`, `a.velocity`, `a.mass` or `a.charge`, or a `param`")
}

/// The warning for a `let` or `param` whose name is also a unit.
pub(crate) fn shadow_warning(name: &Ident) -> Diagnostic {
    Diagnostic::warning(format!("`{}` is also a unit, and this hides it", name.text))
        .with_code("W0313")
        .at(name.span, "shadows a unit")
        .note(format!(
            "after this, `{0}` means this value wherever it is in scope, including in a unit \
             expression such as `3 {0}`",
            name.text
        ))
        .help("choose a name that is not a unit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_syntax::{parse, SourceFile};
    use lattice_units::UnitRegistry;

    fn check(source: &str) -> (Vec<TypedLaw>, Diagnostics) {
        let file = SourceFile::new("t.lattice", source);
        let (project, mut diagnostics) = parse(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let laws = check_laws(&project.unwrap(), &evaluator, &mut diagnostics);
        (laws, diagnostics)
    }

    fn ok(source: &str) -> TypedLaw {
        let (mut laws, diagnostics) = check(source);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&SourceFile::new("t.lattice", source)));
        laws.remove(0)
    }

    fn codes(source: &str) -> Vec<String> {
        let (laws, diagnostics) = check(source);
        assert!(laws.is_empty(), "the law should have been rejected");
        diagnostics.codes().into_iter().map(String::from).collect()
    }

    /// Spec §8.3's spring, with the parameters it leaves unbound declared.
    const SPRING: &str = "project p {
        force spring(a: particle, b: particle) -> vec2<newton> {
            param rest_length: meter;
            param stiffness: newton / meter;
            param damping: kilogram / second;
            let dx = minimum_image(b.position - a.position);
            let extension = length(dx) - rest_length;
            return stiffness * extension * normalize(dx)
                 - damping * dot(b.velocity - a.velocity, normalize(dx)) * normalize(dx);
        }
    }";

    #[test]
    fn the_spec_spring_checks() {
        let law = ok(SPRING);
        assert_eq!(law.kind, LawKind::Force);
        assert_eq!(law.particles, ["a", "b"]);
        assert_eq!(law.params.len(), 3);
        assert_eq!(law.locals.len(), 2);
        assert_eq!(law.result.ty, Ty::Vec2(Dimension::FORCE));
        assert!(law.reads_velocity, "the damping term reads velocities");
    }

    #[test]
    fn units_fold_to_si_constants() {
        let law = ok("project p { potential u(a: particle) -> joule {
            return 2 kilojoule;
        } }");
        let TExprKind::Binary(BinaryOp::Mul, two, kilojoule) = &law.result.kind else { panic!("{:?}", law.result) };
        assert_eq!(two.kind, TExprKind::Const(2.0));
        assert_eq!(kilojoule.kind, TExprKind::Const(1000.0));
    }

    #[test]
    fn a_lennard_jones_potential_checks() {
        let law = ok("project p {
            potential lj(a: particle, b: particle) -> joule {
                param epsilon: joule;
                param sigma: meter = 3.4 angstrom;
                let s6 = (sigma / distance(a, b))^6;
                return 4 * epsilon * (s6^2 - s6);
            }
        }");
        assert!(law.is_pair() && !law.reads_velocity);
        assert_eq!(law.params[1].default, Some(vec![3.4e-10]));
    }

    #[test]
    fn a_dimension_error_points_at_the_subexpression() {
        let source = "project p { force f(a: particle) -> vec2<newton> {
            return a.position + a.velocity;
        } }";
        assert_eq!(codes(source), ["E0400"]);
    }

    #[test]
    fn the_wrong_return_type_is_reported() {
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { return a.mass * a.velocity; } }"),
            ["E0411"],
            "momentum is not a force"
        );
        assert_eq!(codes("project p { force f(a: particle) -> joule { return 1 joule; } }"), ["E0411"]);
        assert_eq!(codes("project p { force f(a: particle) { return vec2(0 newton, 0 newton); } }"), ["E0411"]);
    }

    #[test]
    fn shapes_must_agree() {
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { return a.position * a.position; } }"),
            ["E0410"],
        );
        assert_eq!(codes("project p { potential u(a: particle) -> joule { return a.position + 1 joule; } }"), ["E0410"]);
    }

    #[test]
    fn a_pair_potential_cannot_read_positions_or_velocities() {
        assert_eq!(
            codes("project p { potential u(a: particle, b: particle) -> joule {
                return 1 joule * length(b.position - a.position) / meter; } }"),
            ["E0412", "E0412"],
            "both positions are refused"
        );
        assert_eq!(
            codes("project p { potential u(a: particle) -> joule { return a.mass * dot(a.velocity, a.velocity); } }"),
            ["E0412", "E0412"]
        );
    }

    #[test]
    fn calls_are_checked() {
        assert_eq!(codes("project p { potential u(a: particle) -> joule { return exp(a.mass) * 1 joule; } }"), ["E0400"]);
        assert_eq!(codes("project p { potential u(a: particle) -> joule { return wobble(1) * 1 joule; } }"), ["E0413"]);
        assert_eq!(codes("project p { potential u(a: particle) -> joule { return max(1 joule) ; } }"), ["E0413"]);
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { return minimum_image(a.position) * 1 newton / meter; } }"),
            ["E0412"],
            "minimum_image needs a pair"
        );
    }

    /// `4 epsilon (s6^2 - s6)` reads as a call of `epsilon`; the diagnostic says so.
    #[test]
    fn a_value_followed_by_a_paren_is_explained() {
        let source = "project p { potential u(a: particle, b: particle) -> joule {
            param epsilon: joule;
            let s = distance(a, b) / meter;
            return 4 epsilon (s - 1);
        } }";
        let (_, diagnostics) = check(source);
        let rendered = diagnostics.render(&SourceFile::new("t.lattice", source));
        assert!(rendered.contains("is a value, not a function") && rendered.contains("epsilon * (…)"), "{rendered}");
    }

    #[test]
    fn conditions_must_be_truth_values() {
        assert_eq!(
            codes("project p { potential u(a: particle) -> joule { return if a.mass { 1 joule } else { 0 joule }; } }"),
            ["E0414"]
        );
        let law = ok("project p { potential u(a: particle) -> joule {
            return if a.mass > 1 kilogram && !(a.charge == 0) { 1 joule } else { 0 };
        } }");
        assert!(matches!(law.result.kind, TExprKind::Select(..)));
    }

    #[test]
    fn a_body_must_end_in_one_return() {
        assert_eq!(codes("project p { force f(a: particle) -> vec2<newton> { let x = 1; } }"), ["E0415"]);
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { return vec2(0 newton, 0 newton); let x = 1; } }"),
            ["E0415"]
        );
    }

    #[test]
    fn names_are_defined_once_and_a_unit_shadow_is_warned_about() {
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { let x = 1; let x = 2; return vec2(0 newton, 0 newton); } }"),
            ["E0201"]
        );
        let (laws, diagnostics) = check("project p { potential u(a: particle) -> joule { let m = a.mass; return m * 1 meter^2 / second^2; } }");
        assert_eq!(laws.len(), 1);
        assert_eq!(diagnostics.codes(), ["W0313"]);
    }

    /// A `param` whose unit or default fails is a failed law — never one that "checks",
    /// and never a `param` that silently has no default.
    #[test]
    fn a_bad_param_fails_the_law() {
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { param k: furlongs_per_fortnight; return vec2(0 newton, 0 newton); } }"),
            ["E0200"]
        );
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> { param k: meter = 3 second; return vec2(0 newton, 0 newton); } }"),
            ["E0400"]
        );
    }

    #[test]
    fn a_vector_default_is_written_as_a_vector() {
        let law = ok("project p { force f(a: particle) -> vec2<newton> {
            param accel: vec2<meter / second^2> = vec2(0, -9.81 meter / second^2);
            return a.mass * accel;
        } }");
        assert_eq!(law.params[0].default, Some(vec![0.0, -9.81]));
        assert_eq!(
            codes("project p { force f(a: particle) -> vec2<newton> {
                param accel: vec2<meter / second^2> = [0 meter / second^2, 1 meter / second^2];
                return a.mass * accel; } }"),
            ["E0410"],
            "a list is not a vector inside a law"
        );
    }

    /// A pair law's use site binds `cutoff=` and `truncation=`; a `param` of either name
    /// would capture them.
    #[test]
    fn a_pair_law_cannot_declare_the_use_site_names() {
        for name in ["cutoff", "truncation"] {
            assert_eq!(
                codes(&format!(
                    "project p {{ potential u(a: particle, b: particle) -> joule {{
                        param {name}: meter;
                        return 1 joule;
                    }} }}"
                )),
                ["E0201"],
                "{name}"
            );
        }
        // On one particle there is no cutoff to collide with.
        ok("project p { force f(a: particle) -> vec2<newton> { param cutoff: newton; return vec2(cutoff, 0 newton); } }");
    }

    #[test]
    fn law_kinds_are_known() {
        assert_eq!(codes("project p { torque t(a: particle) -> joule { return 1 joule; } }"), ["E0205"]);
        assert!(PLANNED_KINDS.is_empty(), "every named kind of law is implemented");
    }

    #[test]
    fn a_valid_observer_law_checks() {
        let law = ok("project p { observer kinetic_x(a: particle) -> joule {
            return 0.5 * a.mass * a.velocity.x * a.velocity.x;
        } }");
        assert_eq!(law.kind, LawKind::Observer);
        assert_eq!(law.particles, ["a"]);
        assert_eq!(law.result.ty, Ty::Scalar(Dimension::ENERGY));
        assert!(law.reads_velocity && law.reads_members);
        assert!(!law.reads_temperature);

        // The unit is the law's own, whatever dimension: a plain number is `-> 1`, and
        // a `param` and an `if` are as welcome as in any body law.
        let law = ok("project p { observer right_half(a: particle) -> 1 {
            param x0: meter = 0 meter;
            return if a.position.x > x0 { 1 } else { 0 };
        } }");
        assert_eq!(law.result.ty, Ty::Scalar(Dimension::DIMENSIONLESS));
        assert_eq!(law.params.len(), 1);
        ok("project p { observer speed(a: particle) -> meter / second { return length(a.velocity); } }");
        ok("project p { observer q(a: particle) -> coulomb { return a.charge; } }");
    }

    #[test]
    fn an_observer_is_checked_against_its_declared_unit() {
        assert_eq!(codes("project p { observer o(a: particle) -> joule { return a.mass; } }"), ["E0411"]);
        assert_eq!(codes("project p { observer o(a: particle) { return a.mass; } }"), ["E0411"]);
        assert_eq!(codes("project p { observer o(a: particle) -> bool { return a.mass > 1 kilogram; } }"), ["E0411"]);
    }

    #[test]
    fn an_observer_takes_one_particle() {
        let body = "{ return 1 meter; }";
        assert_eq!(codes(&format!("project p {{ observer o(a: particle, b: particle) -> meter {body} }}")), ["E0416"]);
        assert_eq!(codes(&format!("project p {{ observer o() -> meter {body} }}")), ["E0411"]);
        assert_eq!(codes(&format!("project p {{ observer o(a: meter) -> meter {body} }}")), ["E0411"]);
    }

    #[test]
    fn an_observer_returns_a_scalar() {
        assert_eq!(
            codes("project p { observer o(a: particle) -> vec2<meter / second> { return a.velocity; } }"),
            ["E0417"]
        );
    }

    #[test]
    fn an_observer_reads_particles_only() {
        for body in [
            "return c(A) * 1 meter^2 / mole;",
            "return arrhenius(50 kilojoule / mole) * 1 joule;",
            "return temperature;",
            "return temperature * 1 joule / kelvin;",
        ] {
            let source = format!("project p {{ observer o(a: particle) -> joule {{ {body} }} }}");
            assert_eq!(codes(&source), ["E0418"], "{body}");
        }
        // A `let` or `param` named `c` is a value, and calling it is the accurate error.
        for declaration in ["let c = 1 meter;", "param c: meter = 1 meter;"] {
            let source = format!("project p {{ observer o(a: particle) -> meter {{ {declaration} return c(a); }} }}");
            let (laws, diagnostics) = check(&source);
            assert!(laws.is_empty());
            assert_eq!(diagnostics.codes(), ["E0413"], "{declaration}");
        }
    }

    #[test]
    fn the_header_takes_one_or_two_particles() {
        assert_eq!(codes("project p { force f() -> vec2<newton> { return vec2(0 newton, 0 newton); } }"), ["E0411"]);
        assert_eq!(codes("project p { force f(a: joule) -> vec2<newton> { return vec2(0 newton, 0 newton); } }"), ["E0411"]);
    }
}
