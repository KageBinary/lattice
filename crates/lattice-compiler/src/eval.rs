//! Constant expression evaluation with dimensional checking.
//!
//! This is where spec **FR-002** is actually satisfied:
//!
//! > Validate physical units and dimensional consistency at model compile time
//! > wherever expressions are statically known. *Acceptance: invalid force/energy/rate
//! > expressions fail with source-positioned diagnostics.*
//!
//! Every setting value passes through here and comes out as a [`Quantity`] — an SI
//! magnitude with a dimension — or as a diagnostic pointing at the offending
//! character.
//!
//! # Name resolution
//!
//! A bare identifier is looked up in this order:
//!
//! 1. A declared parameter, if the caller supplied a scope.
//! 2. The unit registry.
//!
//! Declared names win, which is what makes `100 / dt` a division by a parameter while
//! `100 / second` is a frequency. An identifier that is neither gets a diagnostic
//! naming both possibilities, because "unknown unit `dt`" would be actively
//! misleading when the user meant a parameter they forgot to declare.
//!
//! # What is deliberately not here
//!
//! No functions, no field access, no control flow. Those belong to the expression
//! compiler of spec §8.3, which compiles user-defined force and rate laws to CPU and
//! GPU kernels — milestone M6. Call forms that *are* meaningful (field initializers,
//! boundary conditions, integrator methods) are recognized structurally by
//! [`crate::builtins`] rather than evaluated, so the evaluator stays a pure
//! constant folder.

use lattice_syntax::{BinaryOp, Diagnostic, Diagnostics, Expr, ExprKind, SourceFile, Span, UnaryOp};
use lattice_units::{Dimension, Quantity, UnitError, UnitRegistry};

/// Evaluates constant expressions to dimensioned quantities.
pub struct Evaluator<'a> {
    file: &'a SourceFile,
    units: &'a UnitRegistry,
    /// Named constants the model has declared, checked before the unit registry.
    scope: Vec<(String, Quantity)>,
}

impl core::fmt::Debug for Evaluator<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Evaluator").field("scope", &self.scope.len()).finish()
    }
}

impl<'a> Evaluator<'a> {
    /// An evaluator over a source file and unit registry.
    pub fn new(file: &'a SourceFile, units: &'a UnitRegistry) -> Evaluator<'a> {
        Evaluator { file, units, scope: Vec::new() }
    }

    /// Declare a named constant, visible to later expressions.
    pub fn define(&mut self, name: impl Into<String>, value: Quantity) {
        self.scope.push((name.into(), value));
    }

    /// The source file being compiled.
    pub fn file(&self) -> &SourceFile {
        self.file
    }

    /// The unit registry.
    pub fn units(&self) -> &UnitRegistry {
        self.units
    }

    /// Evaluate an expression to a quantity.
    ///
    /// Returns `None` after reporting; the caller continues with other settings so
    /// one compile reports every problem.
    pub fn quantity(&self, expr: &Expr, diagnostics: &mut Diagnostics) -> Option<Quantity> {
        self.warn_if_ambiguous(expr, diagnostics);
        self.eval(expr, diagnostics)
    }

    /// Evaluate and require a specific dimension.
    pub fn require(
        &self,
        expr: &Expr,
        expected: Dimension,
        what: &str,
        diagnostics: &mut Diagnostics,
    ) -> Option<f64> {
        let value = self.quantity(expr, diagnostics)?;
        if value.dimension() == expected {
            return Some(value.value());
        }
        // A bare zero is zero in every unit, so spec §25.2's `x=0` and `[-4 nanometer, 0]`
        // mean what they say. Temperature is the exception: zero kelvin and zero celsius
        // are different temperatures, and a bare 0 cannot say which.
        if is_literal_zero(expr) && expected != Dimension::TEMPERATURE {
            return Some(0.0);
        }
        diagnostics.push(
            Diagnostic::error(format!("{what} has the wrong dimension"))
                .with_code("E0400")
                .at(
                    expr.span,
                    format!(
                        "expected {}, found {}",
                        expected.describe(),
                        value.dimension().describe()
                    ),
                )
                .note(format!("{what} must be measurable in {expected}")),
        );
        None
    }

    /// Evaluate and require a dimensionless value.
    pub fn dimensionless(&self, expr: &Expr, what: &str, diagnostics: &mut Diagnostics) -> Option<f64> {
        let value = self.quantity(expr, diagnostics)?;
        match value.require_dimensionless() {
            Ok(raw) => Some(raw),
            Err(_) => {
                diagnostics.push(
                    Diagnostic::error(format!("{what} must be a plain number"))
                        .with_code("E0402")
                        .at(expr.span, format!("found {}", value.dimension().describe()))
                        .help("remove the unit, or divide by it to make the value dimensionless"),
                );
                None
            }
        }
    }

    /// Evaluate and require a non-negative whole number.
    pub fn count(&self, expr: &Expr, what: &str, diagnostics: &mut Diagnostics) -> Option<usize> {
        let raw = self.dimensionless(expr, what, diagnostics)?;
        if raw.fract() != 0.0 || raw < 0.0 || !raw.is_finite() {
            diagnostics.push(
                Diagnostic::error(format!("{what} must be a whole number of at least zero"))
                    .with_code("E0404")
                    .at(expr.span, format!("found {raw}")),
            );
            return None;
        }
        Some(raw as usize)
    }

    /// Evaluate a list of expressions, requiring exactly `n` elements.
    pub fn fixed_list<'e>(
        &self,
        expr: &'e Expr,
        n: usize,
        what: &str,
        diagnostics: &mut Diagnostics,
    ) -> Option<Vec<&'e Expr>> {
        let (ExprKind::List(items) | ExprKind::Tuple(items)) = &expr.kind else {
            diagnostics.push(
                Diagnostic::error(format!("{what} must be a list of {n} values"))
                    .with_code("E0401")
                    .at(expr.span, "expected a list")
                    .help(format!("write it as `[a, b]` with {n} entries")),
            );
            return None;
        };
        if items.len() != n {
            diagnostics.push(
                Diagnostic::error(format!("{what} must have exactly {n} values"))
                    .with_code("E0401")
                    .at(expr.span, format!("found {}", items.len())),
            );
            return None;
        }
        Some(items.iter().collect())
    }

    /// Evaluate a two-element list into a pair of magnitudes with a required
    /// dimension. Grid sizes and extents are both this shape.
    pub fn pair(
        &self,
        expr: &Expr,
        expected: Dimension,
        what: &str,
        diagnostics: &mut Diagnostics,
    ) -> Option<[f64; 2]> {
        let items = self.fixed_list(expr, 2, what, diagnostics)?;
        let x = self.require(items[0], expected, &format!("{what} (x)"), diagnostics)?;
        let y = self.require(items[1], expected, &format!("{what} (y)"), diagnostics)?;
        Some([x, y])
    }

    /// The bare identifier an expression is, if it is one.
    pub fn as_name<'e>(&self, expr: &'e Expr) -> Option<&'e str> {
        expr.as_name()
    }

    fn eval(&self, expr: &Expr, diagnostics: &mut Diagnostics) -> Option<Quantity> {
        match &expr.kind {
            ExprKind::Number(value) => Some(Quantity::dimensionless(*value)),

            ExprKind::Name(name) => self.resolve_name(name, expr.span, diagnostics),

            ExprKind::Unary(UnaryOp::Neg, inner) => Some(-self.eval(inner, diagnostics)?),
            ExprKind::Unary(UnaryOp::Pos, inner) => self.eval(inner, diagnostics),

            // Conditions are values inside a law, where they choose between two
            // quantities. A setting is one quantity, decided once.
            ExprKind::Unary(UnaryOp::Not, _) | ExprKind::If(..) => {
                diagnostics.push(
                    Diagnostic::error("a condition cannot be a setting's value")
                        .with_code("E0414")
                        .span(expr.span)
                        .note("`if`, `!` and comparisons choose between values inside a law")
                        .help("a value that depends on the state belongs in a `force` or `potential`"),
                );
                None
            }
            ExprKind::Binary(op, _, _) if op.is_comparison() || op.is_logical() => {
                diagnostics.push(
                    Diagnostic::error(format!("`{}` gives a truth value, where a quantity is required", op.symbol()))
                        .with_code("E0414")
                        .span(expr.span)
                        .note("comparisons choose between values inside a law"),
                );
                None
            }
            ExprKind::Yields(..) => {
                diagnostics.push(
                    Diagnostic::error("a reaction equation cannot be a quantity")
                        .with_code("E0401")
                        .span(expr.span)
                        .note("`reactants -> products` is the value of a reaction's `stoichiometry:`"),
                );
                None
            }

            ExprKind::Binary(op, left, right) => {
                // Both sides are evaluated even when the first fails, so a model with
                // two bad units reports both instead of one per compile.
                let a = self.eval(left, diagnostics);
                let b = self.eval(right, diagnostics);
                let (a, b) = (a?, b?);
                match op {
                    BinaryOp::Mul => Some(a * b),
                    BinaryOp::Div => Some(a / b),
                    // Reported by the arm above, before either side was evaluated.
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::And
                    | BinaryOp::Or => None,
                    BinaryOp::Add | BinaryOp::Sub => {
                        let result = if *op == BinaryOp::Add { a.try_add(b) } else { a.try_sub(b) };
                        match result {
                            Ok(value) => Some(value),
                            Err(UnitError::Mismatch { .. }) => {
                                diagnostics.push(
                                    Diagnostic::error(format!(
                                        "cannot {} quantities with different dimensions",
                                        if *op == BinaryOp::Add { "add" } else { "subtract" }
                                    ))
                                    .with_code("E0400")
                                    .at(expr.span, format!("`{}` here", op.symbol()))
                                    .also(left.span, format!("this is {}", a.dimension().describe()))
                                    .also(right.span, format!("this is {}", b.dimension().describe()))
                                    .help(
                                        "quantities can only be added when they measure the \
                                         same thing; check the units on each side, and that \
                                         the expression groups the way you intended",
                                    ),
                                );
                                None
                            }
                            Err(other) => {
                                diagnostics.push(
                                    Diagnostic::error(other.to_string())
                                        .with_code("E0400")
                                        .span(expr.span),
                                );
                                None
                            }
                        }
                    }
                }
            }

            ExprKind::Power(base, exponent) => {
                let value = self.eval(base, diagnostics)?;
                match value.try_powi(*exponent) {
                    Ok(result) => Some(result),
                    Err(error) => {
                        diagnostics.push(
                            Diagnostic::error(error.to_string()).with_code("E0400").span(expr.span),
                        );
                        None
                    }
                }
            }

            ExprKind::Call(callee, _) => {
                let name = callee.as_name().unwrap_or("this");
                diagnostics.push(
                    Diagnostic::error(format!("`{name}` cannot be used as a value here"))
                        .with_code("E0210")
                        .at(expr.span, "a function call, where a quantity is required")
                        .note(
                            "function calls are recognized only in the places that expect \
                             them: field initializers, boundary conditions, and integrator \
                             methods",
                        )
                        .help(
                            "a law that computes a value from the state is declared with \
                             `force` or `potential` and used by name; see docs/language.md",
                        ),
                );
                None
            }

            ExprKind::List(_) | ExprKind::Tuple(_) => {
                diagnostics.push(
                    Diagnostic::error("expected a single quantity, found a list")
                        .with_code("E0401")
                        .span(expr.span),
                );
                None
            }

            ExprKind::Str(_) => {
                diagnostics.push(
                    Diagnostic::error("expected a quantity, found a string")
                        .with_code("E0401")
                        .span(expr.span),
                );
                None
            }

            ExprKind::Member(_, member) => {
                diagnostics.push(
                    Diagnostic::error(format!("`.{member}` cannot be read at compile time"))
                        .with_code("E0210")
                        .at(expr.span, "member access")
                        .help(
                            "members such as `a.position` are runtime values, read inside a \
                             `force` or `potential`; rate laws that read concentrations arrive \
                             in milestone M6.1c",
                        ),
                );
                None
            }

            ExprKind::Bool(_) => {
                diagnostics.push(
                    Diagnostic::error("expected a quantity, found a boolean")
                        .with_code("E0401")
                        .span(expr.span),
                );
                None
            }
        }
    }

    fn resolve_name(&self, name: &str, span: Span, diagnostics: &mut Diagnostics) -> Option<Quantity> {
        // Declared names beat units, so a parameter called `k` is not silently a kelvin.
        if let Some((_, value)) = self.scope.iter().rev().find(|(key, _)| key == name) {
            return Some(*value);
        }

        match self.units.resolve(name) {
            Ok(resolved) => {
                if resolved.def.is_affine() {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "`{name}` is an offset temperature scale and cannot appear in an expression"
                        ))
                        .with_code("E0403")
                        .at(span, "offset scale")
                        .note(
                            "0 °C is 273.15 K, so scaling or dividing by it has no consistent \
                             meaning",
                        )
                        .help("use `kelvin` for temperature differences, rates and coefficients"),
                    );
                    return None;
                }
                Some(Quantity::new(resolved.scale(), resolved.dim()))
            }
            Err(UnitError::UnknownUnit { suggestion, .. }) => {
                let mut diagnostic = Diagnostic::error(format!("cannot resolve `{name}`"))
                    .with_code("E0200")
                    .at(span, "not a declared name or a known unit");
                if let Some(suggestion) = suggestion {
                    diagnostic = diagnostic.help(format!("did you mean the unit `{suggestion}`?"));
                } else if !self.scope.is_empty() {
                    let declared: Vec<&str> =
                        self.scope.iter().map(|(key, _)| key.as_str()).take(6).collect();
                    diagnostic =
                        diagnostic.note(format!("declared names in scope: {}", declared.join(", ")));
                }
                diagnostics.push(diagnostic);
                None
            }
            Err(other) => {
                diagnostics.push(
                    Diagnostic::error(other.to_string()).with_code("E0200").span(span),
                );
                None
            }
        }
    }

    /// Warn about `a / b * c`, which scientific notation and programming languages
    /// read differently.
    ///
    /// `kilojoule / mole * kelvin` means `kJ/(mol·K)` to a chemist and `(kJ/mol)·K` to
    /// every parser ever written — a difference of `K²`. Lattice applies the
    /// left-to-right rule and says so rather than guessing, which is the same choice
    /// the unit-expression parser makes.
    fn warn_if_ambiguous(&self, expr: &Expr, diagnostics: &mut Diagnostics) {
        let ExprKind::Binary(BinaryOp::Mul, left, right) = &expr.kind else { return };
        let ExprKind::Binary(BinaryOp::Div, _, _) = &left.kind else { return };

        // Only warn when the multiplicand is a bare unit. `rate / volume * count` is
        // ordinary arithmetic on declared names and needs no advice.
        let Some(name) = right.as_name() else { return };
        let is_unit = !self.scope.iter().any(|(key, _)| key == name)
            && self.units.resolve(name).is_ok();
        if !is_unit {
            return;
        }

        let source = self.file.slice(expr.span);
        diagnostics.push(
            Diagnostic::warning("multiplication after division is grouped left to right")
                .with_code("W0300")
                .at(expr.span, "read as `(a / b) * c`")
                .note(format!(
                    "`{source}` is evaluated as `({}) * {name}`",
                    self.file.slice(left.span)
                ))
                .help("add parentheses to state the intended grouping"),
        );
    }
}

/// Whether an expression is the number zero as written, `0` or `-0.0`, with no unit.
fn is_literal_zero(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Number(value) => *value == 0.0,
        ExprKind::Unary(_, inner) => is_literal_zero(inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_syntax::{parse, Item};

    #[test]
    fn a_bare_zero_is_zero_of_any_dimension_but_temperature() {
        let file = SourceFile::new("t.lattice", "project p { a: 0; b: -0.0; c: 2; }");
        let (project, _) = parse(&file);
        let project = project.unwrap();
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let value = |k: usize| match &project.items[k] {
            Item::Setting(s) => s.value.clone(),
            _ => panic!(),
        };
        let mut diagnostics = Diagnostics::default();
        assert_eq!(evaluator.require(&value(0), Dimension::LENGTH, "x", &mut diagnostics), Some(0.0));
        assert_eq!(evaluator.require(&value(1), Dimension::MOMENTUM, "p", &mut diagnostics), Some(0.0));
        assert!(!diagnostics.has_errors());
        assert_eq!(evaluator.require(&value(2), Dimension::LENGTH, "x", &mut diagnostics), None, "2 is not 2 m");
        let mut diagnostics = Diagnostics::default();
        assert_eq!(evaluator.require(&value(0), Dimension::TEMPERATURE, "T", &mut diagnostics), None);
        assert!(diagnostics.codes().contains(&"E0400"), "0 K or 0 °C? A bare zero cannot say");
    }

    /// Evaluate the value of `x: <source>;` in a throwaway project.
    fn eval_setting(source: &str) -> (Option<Quantity>, Diagnostics, SourceFile) {
        let file = SourceFile::new("t.lattice", format!("project p {{ x: {source}; }}"));
        let (project, mut diagnostics) = parse(&file);
        let project = project.expect("should parse");
        let Item::Setting(setting) = &project.items[0] else { panic!("expected a setting") };

        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let value = evaluator.quantity(&setting.value, &mut diagnostics);
        (value, diagnostics, file)
    }

    fn ok(source: &str) -> Quantity {
        let (value, diagnostics, file) = eval_setting(source);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        value.expect("should evaluate")
    }

    fn err(source: &str) -> (String, Vec<String>) {
        let (_, diagnostics, file) = eval_setting(source);
        assert!(diagnostics.has_errors(), "expected an error from `{source}`");
        let codes = diagnostics.codes().into_iter().map(String::from).collect();
        (diagnostics.render(&file), codes)
    }

    #[test]
    fn plain_numbers_are_dimensionless() {
        let value = ok("42");
        assert_eq!(value.value(), 42.0);
        assert!(value.dimension().is_dimensionless());
    }

    /// Every dimensioned literal from the spec's example models.
    #[test]
    fn spec_literals_evaluate_with_the_right_dimensions() {
        let cases: &[(&str, Dimension, f64)] = &[
            ("298 kelvin", Dimension::TEMPERATURE, 298.0),
            ("2 meter", Dimension::LENGTH, 2.0),
            ("0.002 second", Dimension::TIME, 0.002),
            ("9.31e-9 meter^2 / second", Dimension::DIFFUSIVITY, 9.31e-9),
            ("-57.3 kilojoule / mole", Dimension::MOLAR_ENERGY, -57_300.0),
            ("-25 kilojoule / mole", Dimension::MOLAR_ENERGY, -25_000.0),
            ("1 mole / meter^2", Dimension::AREAL_CONCENTRATION, 1.0),
            ("2.0e5 / second", Dimension::FREQUENCY, 2.0e5),
            ("20 electronvolt", Dimension::ENERGY, 20.0 * 1.602_176_634e-19),
            ("12 nanometer", Dimension::LENGTH, 12e-9),
            ("0.002 femtosecond", Dimension::TIME, 2e-18),
            ("6.5e-24 kilogram*meter/second", Dimension::MOMENTUM, 6.5e-24),
            ("electron_mass", Dimension::MASS, 9.109_383_713_9e-31),
        ];
        for (source, dimension, expected) in cases {
            let value = ok(source);
            assert_eq!(value.dimension(), *dimension, "`{source}`");
            assert!(
                (value.value() - expected).abs() <= 1e-9 * expected.abs().max(1e-30),
                "`{source}` = {} but expected {expected}",
                value.value()
            );
        }
    }

    #[test]
    fn arithmetic_composes_dimensions() {
        assert_eq!(ok("2 meter * 3 meter").dimension(), Dimension::AREA);
        assert_eq!(ok("10 meter / 2 second").dimension(), Dimension::VELOCITY);
        assert_eq!(ok("(2 meter)^2").dimension(), Dimension::AREA);
        assert!((ok("2 meter * 3 meter").value() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn addition_of_matching_dimensions_works() {
        let value = ok("1 meter + 50 centimeter");
        assert_eq!(value.dimension(), Dimension::LENGTH);
        assert!((value.value() - 1.5).abs() < 1e-12);
    }

    /// The motivating FR-002 case, and the diagnostic must name both sides.
    #[test]
    fn adding_incompatible_dimensions_is_a_positioned_error() {
        let (text, codes) = err("1 meter + 2 second");
        assert!(codes.contains(&"E0400".to_string()));
        assert!(text.contains("cannot add"), "{text}");
        assert!(text.contains("this is length"), "{text}");
        assert!(text.contains("this is time"), "{text}");
        assert!(text.contains('^'), "the error should point at the source:\n{text}");
    }

    #[test]
    fn an_unknown_name_suggests_a_unit() {
        let (text, codes) = err("5 metre_");
        assert!(codes.contains(&"E0200".to_string()));
        assert!(text.contains("cannot resolve `metre_`"), "{text}");
        assert!(text.contains("did you mean the unit `metre`"), "{text}");
    }

    #[test]
    fn an_unknown_name_with_no_close_match_still_reports_clearly() {
        let (text, codes) = err("5 zzzzzzzzzz");
        assert!(codes.contains(&"E0200".to_string()));
        assert!(text.contains("not a declared name or a known unit"), "{text}");
    }

    /// Declared names beat units, so a parameter named `k` is not a kelvin.
    #[test]
    fn declared_names_take_priority_over_units() {
        let file = SourceFile::new("t.lattice", "project p { x: 3 * k; }");
        let (project, mut diagnostics) = parse(&file);
        let project = project.unwrap();
        let Item::Setting(setting) = &project.items[0] else { panic!() };

        let units = UnitRegistry::si();
        let mut evaluator = Evaluator::new(&file, &units);
        evaluator.define("k", Quantity::new(7.0, Dimension::FREQUENCY));

        let value = evaluator.quantity(&setting.value, &mut diagnostics).unwrap();
        assert!(!diagnostics.has_errors());
        assert_eq!(value.dimension(), Dimension::FREQUENCY);
        assert!((value.value() - 21.0).abs() < 1e-12);
    }

    #[test]
    fn an_affine_temperature_scale_is_rejected_in_an_expression() {
        let (text, codes) = err("5 joule / celsius");
        assert!(codes.contains(&"E0403".to_string()));
        assert!(text.contains("offset temperature scale"), "{text}");
        assert!(text.contains("use `kelvin`"), "{text}");
    }

    /// The `a/b*c` ambiguity, warned rather than guessed at.
    #[test]
    fn multiplication_after_division_warns() {
        let (_, diagnostics, file) = eval_setting("8.314 joule / mole * kelvin");
        assert!(!diagnostics.has_errors(), "it is a warning, not an error");
        assert_eq!(diagnostics.warning_count(), 1);
        let text = diagnostics.render(&file);
        assert!(text.contains("left to right"), "{text}");
        assert!(text.contains("parentheses"), "{text}");
    }

    #[test]
    fn parenthesized_grouping_does_not_warn() {
        let (_, diagnostics, _) = eval_setting("8.314 joule / (mole * kelvin)");
        assert_eq!(diagnostics.warning_count(), 0);
    }

    /// Arithmetic on declared names is not unit notation and needs no advice.
    #[test]
    fn division_then_multiplication_by_a_declared_name_does_not_warn() {
        let file = SourceFile::new("t.lattice", "project p { x: 1 second / rate * count; }");
        let (project, mut diagnostics) = parse(&file);
        let Item::Setting(setting) = &project.unwrap().items[0] else { panic!() };

        let units = UnitRegistry::si();
        let mut evaluator = Evaluator::new(&file, &units);
        evaluator.define("rate", Quantity::new(2.0, Dimension::FREQUENCY));
        evaluator.define("count", Quantity::dimensionless(3.0));

        evaluator.quantity(&setting.value, &mut diagnostics);
        assert_eq!(diagnostics.warning_count(), 0);
    }

    #[test]
    fn a_call_where_a_quantity_belongs_explains_itself() {
        let (text, codes) = err("gaussian(1, 2)");
        assert!(codes.contains(&"E0210".to_string()));
        assert!(text.contains("field initializers"), "{text}");
        assert!(text.contains("`force` or `potential`"), "the message should say where a computed value goes:\n{text}");
    }

    #[test]
    fn member_access_points_at_laws_and_the_rate_law_milestone() {
        let (text, codes) = err("A.concentration");
        assert!(codes.contains(&"E0210".to_string()));
        assert!(text.contains("inside a") && text.contains("M6.1c"), "{text}");
    }

    #[test]
    fn a_condition_is_not_a_setting_value() {
        for source in ["1 meter < 2 meter", "if true { 1 } else { 2 }", "!true"] {
            let (_, codes) = err(source);
            assert_eq!(codes, ["E0414"], "{source}");
        }
    }

    #[test]
    fn a_list_where_a_scalar_belongs_is_reported() {
        let (_, codes) = err("[1, 2]");
        assert!(codes.contains(&"E0401".to_string()));
    }

    /// Both halves of a binary expression are checked, so one compile reports both
    /// bad units rather than one per run.
    #[test]
    fn both_sides_of_an_expression_are_reported() {
        let (_, diagnostics, _) = eval_setting("1 blorp + 2 zorp");
        assert_eq!(diagnostics.error_count(), 2, "each unknown unit should be reported");
    }

    // --- the typed helpers --------------------------------------------------

    fn helper_setup(source: &str) -> (SourceFile, Expr, Diagnostics) {
        let file = SourceFile::new("t.lattice", format!("project p {{ x: {source}; }}"));
        let (project, diagnostics) = parse(&file);
        let Item::Setting(setting) = &project.unwrap().items[0] else { panic!() };
        (file, setting.value.clone(), diagnostics)
    }

    #[test]
    fn require_extracts_the_magnitude_when_the_dimension_matches() {
        let (file, expr, mut diagnostics) = helper_setup("5 second");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        assert_eq!(evaluator.require(&expr, Dimension::TIME, "the timestep", &mut diagnostics), Some(5.0));
        assert!(!diagnostics.has_errors());
    }

    #[test]
    fn require_names_both_dimensions_when_it_fails() {
        let (file, expr, mut diagnostics) = helper_setup("5 second");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        assert_eq!(evaluator.require(&expr, Dimension::TEMPERATURE, "the initial value", &mut diagnostics), None);

        let text = diagnostics.render(&file);
        assert!(text.contains("expected temperature"), "{text}");
        assert!(text.contains("found time"), "{text}");
        assert!(text.contains("the initial value"), "{text}");
    }

    #[test]
    fn count_requires_a_whole_non_negative_number() {
        let units = UnitRegistry::si();
        for (source, expected) in [("512", Some(512usize)), ("0", Some(0))] {
            let (file, expr, mut diagnostics) = helper_setup(source);
            let evaluator = Evaluator::new(&file, &units);
            assert_eq!(evaluator.count(&expr, "the cell count", &mut diagnostics), expected);
        }
        for source in ["2.5", "-1", "5 meter"] {
            let (file, expr, mut diagnostics) = helper_setup(source);
            let evaluator = Evaluator::new(&file, &units);
            assert_eq!(evaluator.count(&expr, "the cell count", &mut diagnostics), None, "{source}");
            assert!(diagnostics.has_errors(), "{source}");
        }
    }

    #[test]
    fn pair_reads_a_two_element_list_with_units() {
        let (file, expr, mut diagnostics) = helper_setup("[2 meter, 1 meter]");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let extent = evaluator.pair(&expr, Dimension::LENGTH, "the extent", &mut diagnostics);
        assert_eq!(extent, Some([2.0, 1.0]));
        assert!(!diagnostics.has_errors());
    }

    #[test]
    fn pair_reports_a_wrong_length_and_a_wrong_shape() {
        let units = UnitRegistry::si();
        for source in ["[1 meter]", "[1 meter, 2 meter, 3 meter]", "5 meter"] {
            let (file, expr, mut diagnostics) = helper_setup(source);
            let evaluator = Evaluator::new(&file, &units);
            assert_eq!(
                evaluator.pair(&expr, Dimension::LENGTH, "the extent", &mut diagnostics),
                None,
                "{source}"
            );
            assert!(diagnostics.codes().contains(&"E0401"), "{source}");
        }
    }

    #[test]
    fn a_pair_with_a_wrong_dimension_names_the_offending_component() {
        let (file, expr, mut diagnostics) = helper_setup("[2 meter, 1 second]");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        assert_eq!(evaluator.pair(&expr, Dimension::LENGTH, "the extent", &mut diagnostics), None);
        let text = diagnostics.render(&file);
        assert!(text.contains("the extent (y)"), "{text}");
    }
}
