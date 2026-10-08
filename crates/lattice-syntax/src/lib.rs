//! Lexer, AST, parser, and diagnostics for the `.lattice` project language.
//!
//! Spec §16.1 lists the project DSL as the *"declarative canonical model"* — the
//! surface every other authoring route (Python, Rust, the editor) ultimately produces.
//! §16.2 sets the requirements: readable declarative syntax with explicit units,
//! source-positioned diagnostics, no hidden global state.
//!
//! This crate turns text into an [`ast::Project`]. It resolves nothing: whether
//! `kilojoule` is a unit or a typo, and whether `298 second` is dimensionally valid,
//! are questions for `lattice-compiler`. The split matters because a parser that
//! knows about units cannot report two independent errors in one pass.

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod parser;
pub mod source;

pub use ast::{
    Argument, BinaryOp, CoupleStmt, Decl, DomainDecl, Expr, ExprKind, FieldDecl, Ident, Item,
    LawDecl, LawParam, LetDecl, ObserveStmt, Path, Project, Setting, SolveStmt, Stmt, StmtKind,
    TypeExpr, TypeKind, UnaryOp, VisualizeStmt,
};
pub use diagnostic::{Diagnostic, Diagnostics, Label, Severity};
pub use lexer::{tokenize, Keyword, Token, TokenKind};
pub use parser::parse;
pub use source::{Location, SourceFile, Span};
