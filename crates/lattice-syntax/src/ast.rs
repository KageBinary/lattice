//! The abstract syntax tree.
//!
//! Shapes follow the example models in spec §12.2, §25.1 and §25.2. Every node
//! carries a [`Span`], because a dimensional error found in the compiler's third pass
//! still has to point at the character that caused it (FR-002).
//!
//! The tree is deliberately *unresolved*. `Setting { key: "extent", value: … }` does
//! not know that `extent` is meaningful for a grid, and `ExprKind::Name("kilojoule")`
//! does not know it is a unit. Keeping those questions out of the parser means a
//! malformed model still parses far enough to report several problems at once,
//! instead of stopping at the first.

use core::fmt;

use crate::source::Span;

/// A name with its source position.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ident {
    /// The text.
    pub text: String,
    /// Where it appeared.
    pub span: Span,
}

impl Ident {
    /// Build an identifier.
    pub fn new(text: impl Into<String>, span: Span) -> Ident {
        Ident { text: text.into(), span }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// A dotted path, such as `A_plus_B.heat_release`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Path {
    /// Dot-separated segments, at least one.
    pub segments: Vec<Ident>,
    /// The whole path.
    pub span: Span,
}

impl Path {
    /// The first segment, which names the item being referred to.
    pub fn root(&self) -> &Ident {
        &self.segments[0]
    }

    /// Everything after the root, which names a port or member.
    pub fn tail(&self) -> &[Ident] {
        &self.segments[1..]
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined: Vec<&str> = self.segments.iter().map(|s| s.text.as_str()).collect();
        f.write_str(&joined.join("."))
    }
}

/// A `key: value;` pair inside a block.
#[derive(Clone, PartialEq, Debug)]
pub struct Setting {
    /// The key.
    pub key: Ident,
    /// The value.
    pub value: Expr,
    /// The whole setting including its terminator.
    pub span: Span,
}

/// A named argument, as in `dt=0.002 second` or a positional `A`.
#[derive(Clone, PartialEq, Debug)]
pub struct Argument {
    /// The parameter name, if given.
    pub name: Option<Ident>,
    /// The value.
    pub value: Expr,
    /// The whole argument.
    pub span: Span,
}

/// Unary operators.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `+x`
    Pos,
}

/// Binary operators.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`, including juxtaposition such as `35 kilojoule`.
    Mul,
    /// `a / b`
    Div,
}

impl BinaryOp {
    /// The spelling, for diagnostics.
    pub const fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        }
    }
}

/// An expression.
#[derive(Clone, PartialEq, Debug)]
pub struct Expr {
    /// What kind it is.
    pub kind: ExprKind,
    /// Where it is.
    pub span: Span,
}

impl Expr {
    /// Build an expression node.
    pub fn new(kind: ExprKind, span: Span) -> Expr {
        Expr { kind, span }
    }

    /// The identifier this expression is, if it is a bare name.
    pub fn as_name(&self) -> Option<&str> {
        match &self.kind {
            ExprKind::Name(name) => Some(name),
            _ => None,
        }
    }

    /// The numeric literal this expression is, if it is one.
    pub fn as_number(&self) -> Option<f64> {
        match self.kind {
            ExprKind::Number(value) => Some(value),
            _ => None,
        }
    }

    /// The list elements, if this is a list.
    pub fn as_list(&self) -> Option<&[Expr]> {
        match &self.kind {
            ExprKind::List(items) => Some(items),
            _ => None,
        }
    }
}

/// The kinds of expression.
#[derive(Clone, PartialEq, Debug)]
pub enum ExprKind {
    /// A numeric literal.
    Number(f64),
    /// A string literal, already unescaped.
    Str(String),
    /// `true` or `false`.
    Bool(bool),
    /// A bare name. Could be a declared item, a parameter, or a unit — the compiler
    /// decides, in that order.
    Name(String),
    /// `receiver.member`
    Member(Box<Expr>, Ident),
    /// `callee(args)`
    Call(Box<Expr>, Vec<Argument>),
    /// `base^exponent`, with an integer exponent so dimensions stay representable.
    Power(Box<Expr>, i32),
    /// A prefix operator.
    Unary(UnaryOp, Box<Expr>),
    /// An infix operator.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// `[a, b, c]`
    List(Vec<Expr>),
    /// `(a, b)`
    Tuple(Vec<Expr>),
}

/// A declaration of the general form `<kind> <name> { … }` or `<kind> <name> … ;`.
///
/// Covers `grid chamber { … }`, `reaction acid_base { … }`, `potential barrier { … }`,
/// `wavepacket initial { … }`, and `detector screen at x=4.5 nanometer;`. The parser
/// does not know which kinds are meaningful — that is the compiler's decision, which
/// is what lets a new solver family arrive without a grammar change.
#[derive(Clone, PartialEq, Debug)]
pub struct Decl {
    /// What sort of thing this declares: `grid`, `reaction`, `detector`, …
    pub kind: Ident,
    /// The declared name.
    pub name: Ident,
    /// A trailing modifier word, such as the `at` in `detector screen at x=…`.
    pub modifier: Option<Ident>,
    /// Arguments following the modifier.
    pub arguments: Vec<Argument>,
    /// Settings inside the braces, when written in block form.
    pub settings: Vec<Setting>,
    /// The whole declaration.
    pub span: Span,
}

impl Decl {
    /// Look up a setting by key.
    pub fn setting(&self, key: &str) -> Option<&Setting> {
        self.settings.iter().find(|s| s.key.text == key)
    }

    /// Look up an argument by name.
    pub fn argument(&self, name: &str) -> Option<&Argument> {
        self.arguments.iter().find(|a| a.name.as_ref().is_some_and(|n| n.text == name))
    }
}

/// `field temperature on chamber = 298 kelvin;` or the block form.
#[derive(Clone, PartialEq, Debug)]
pub struct FieldDecl {
    /// The declared name.
    pub name: Ident,
    /// The grid it lives on, from `on <grid>`.
    pub grid: Option<Ident>,
    /// The initial value, from `= <expr>`.
    pub initial: Option<Expr>,
    /// Settings, when written in block form.
    pub settings: Vec<Setting>,
    /// The whole declaration.
    pub span: Span,
}

impl FieldDecl {
    /// Look up a setting by key.
    pub fn setting(&self, key: &str) -> Option<&Setting> {
        self.settings.iter().find(|s| s.key.text == key)
    }
}

/// `domain quantum2d q { … }` — a solver family plus an instance name.
#[derive(Clone, PartialEq, Debug)]
pub struct DomainDecl {
    /// The solver family, e.g. `quantum2d` or `particles2d`.
    pub family: Ident,
    /// The instance name.
    pub name: Ident,
    /// Settings inside the braces.
    pub settings: Vec<Setting>,
    /// The whole declaration.
    pub span: Span,
}

impl DomainDecl {
    /// Look up a setting by key.
    pub fn setting(&self, key: &str) -> Option<&Setting> {
        self.settings.iter().find(|s| s.key.text == key)
    }
}

/// `solve diffusion(A, B, C) with crank_nicolson(dt=0.002 second);`
#[derive(Clone, PartialEq, Debug)]
pub struct SolveStmt {
    /// What to solve, e.g. `diffusion` or `heat`.
    pub solver: Ident,
    /// The state it acts on.
    pub targets: Vec<Argument>,
    /// The method, e.g. `crank_nicolson`.
    pub method: Ident,
    /// Method parameters.
    pub parameters: Vec<Argument>,
    /// The whole statement.
    pub span: Span,
}

impl SolveStmt {
    /// Look up a method parameter by name.
    pub fn parameter(&self, name: &str) -> Option<&Argument> {
        self.parameters.iter().find(|a| a.name.as_ref().is_some_and(|n| n.text == name))
    }
}

/// `couple A_plus_B.heat_release -> temperature.source conserve energy;`
#[derive(Clone, PartialEq, Debug)]
pub struct CoupleStmt {
    /// The publishing port.
    pub source: Path,
    /// The consuming port.
    pub target: Path,
    /// The quantity whose transfer must balance.
    pub conserve: Option<Ident>,
    /// The whole statement.
    pub span: Span,
}

/// `observe total_energy every 0.1 second;`
#[derive(Clone, PartialEq, Debug)]
pub struct ObserveStmt {
    /// What to observe.
    pub target: Expr,
    /// The sampling interval.
    pub every: Option<Expr>,
    /// The whole statement.
    pub span: Span,
}

/// `visualize temperature as heatmap;` — the style is optional, since spec §25.2
/// writes `visualize probability_density;` and lets the domain choose.
#[derive(Clone, PartialEq, Debug)]
pub struct VisualizeStmt {
    /// What to draw.
    pub target: Expr,
    /// How to draw it, if stated.
    pub style: Option<Ident>,
    /// The whole statement.
    pub span: Span,
}

/// One top-level entry inside a project.
#[derive(Clone, PartialEq, Debug)]
pub enum Item {
    /// `dimensions: 2;`
    Setting(Setting),
    /// `grid chamber { … }`, `reaction r { … }`, `detector screen at x=…;` — any
    /// declaration of the general `<kind> <name>` form.
    Decl(Decl),
    /// `field temperature on chamber = …;`
    Field(FieldDecl),
    /// `species A on chamber = …;` or `species H_plus { … }`
    Species(FieldDecl),
    /// `domain quantum2d q { … }`
    Domain(DomainDecl),
    /// `solve … with …;`
    Solve(SolveStmt),
    /// `couple … -> …;`
    Couple(CoupleStmt),
    /// `observe … every …;`
    Observe(ObserveStmt),
    /// `visualize … as …;`
    Visualize(VisualizeStmt),
}

impl Item {
    /// The declared name, for items that declare one.
    pub fn name(&self) -> Option<&Ident> {
        match self {
            Item::Setting(s) => Some(&s.key),
            Item::Decl(d) => Some(&d.name),
            Item::Field(d) | Item::Species(d) => Some(&d.name),
            Item::Domain(d) => Some(&d.name),
            Item::Solve(_) | Item::Couple(_) | Item::Observe(_) | Item::Visualize(_) => None,
        }
    }

    /// What sort of item this is, for diagnostics.
    pub fn kind_name(&self) -> &str {
        match self {
            Item::Setting(_) => "setting",
            Item::Decl(d) => &d.kind.text,
            Item::Field(_) => "field",
            Item::Species(_) => "species",
            Item::Domain(_) => "domain",
            Item::Solve(_) => "solve",
            Item::Couple(_) => "couple",
            Item::Observe(_) => "observe",
            Item::Visualize(_) => "visualize",
        }
    }

    /// The span of the whole item.
    pub fn span(&self) -> Span {
        match self {
            Item::Setting(s) => s.span,
            Item::Decl(d) => d.span,
            Item::Field(d) | Item::Species(d) => d.span,
            Item::Domain(d) => d.span,
            Item::Solve(s) => s.span,
            Item::Couple(s) => s.span,
            Item::Observe(s) => s.span,
            Item::Visualize(s) => s.span,
        }
    }
}

/// A parsed project.
#[derive(Clone, PartialEq, Debug)]
pub struct Project {
    /// The project name.
    pub name: Ident,
    /// Everything declared inside it.
    pub items: Vec<Item>,
    /// The whole `project … { … }`.
    pub span: Span,
}

impl Project {
    /// Top-level settings such as `dimensions:` and `fidelity:`.
    pub fn settings(&self) -> impl Iterator<Item = &Setting> {
        self.items.iter().filter_map(|item| match item {
            Item::Setting(setting) => Some(setting),
            _ => None,
        })
    }

    /// Look up a top-level setting by key.
    pub fn setting(&self, key: &str) -> Option<&Setting> {
        self.settings().find(|s| s.key.text == key)
    }

    /// Every general-form declaration, of any kind.
    pub fn declarations(&self) -> impl Iterator<Item = &Decl> {
        self.items.iter().filter_map(|item| match item {
            Item::Decl(decl) => Some(decl),
            _ => None,
        })
    }

    /// Declarations of one kind, e.g. `"grid"` or `"reaction"`.
    pub fn declarations_of<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Decl> + 'a {
        self.declarations().filter(move |decl| decl.kind.text == kind)
    }

    /// Look up a declaration by kind and name.
    pub fn declaration(&self, kind: &str, name: &str) -> Option<&Decl> {
        self.declarations().find(|d| d.kind.text == kind && d.name.text == name)
    }

    /// All grid declarations.
    pub fn grids(&self) -> impl Iterator<Item = &Decl> {
        self.declarations_of("grid")
    }

    /// All field declarations.
    pub fn fields(&self) -> impl Iterator<Item = &FieldDecl> {
        self.items.iter().filter_map(|item| match item {
            Item::Field(decl) => Some(decl),
            _ => None,
        })
    }

    /// All species declarations.
    pub fn species(&self) -> impl Iterator<Item = &FieldDecl> {
        self.items.iter().filter_map(|item| match item {
            Item::Species(decl) => Some(decl),
            _ => None,
        })
    }

    /// All `solve` statements.
    pub fn solves(&self) -> impl Iterator<Item = &SolveStmt> {
        self.items.iter().filter_map(|item| match item {
            Item::Solve(stmt) => Some(stmt),
            _ => None,
        })
    }

    /// All `couple` statements.
    pub fn couples(&self) -> impl Iterator<Item = &CoupleStmt> {
        self.items.iter().filter_map(|item| match item {
            Item::Couple(stmt) => Some(stmt),
            _ => None,
        })
    }

    /// All `observe` statements.
    pub fn observes(&self) -> impl Iterator<Item = &ObserveStmt> {
        self.items.iter().filter_map(|item| match item {
            Item::Observe(stmt) => Some(stmt),
            _ => None,
        })
    }

    /// All `visualize` statements.
    pub fn visualizes(&self) -> impl Iterator<Item = &VisualizeStmt> {
        self.items.iter().filter_map(|item| match item {
            Item::Visualize(stmt) => Some(stmt),
            _ => None,
        })
    }

    /// All reaction declarations.
    pub fn reactions(&self) -> impl Iterator<Item = &Decl> {
        self.declarations_of("reaction")
    }

    /// All explicit `domain` declarations.
    pub fn domains(&self) -> impl Iterator<Item = &DomainDecl> {
        self.items.iter().filter_map(|item| match item {
            Item::Domain(decl) => Some(decl),
            _ => None,
        })
    }
}
