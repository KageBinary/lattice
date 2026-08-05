//! Parsing of unit expressions and dimensioned literals.
//!
//! # Grammar
//!
//! ```text
//! expr := term (('*' | '·' | '/') term | term)*     // juxtaposition = multiplication
//! term := atom ('^' signed_int)?
//! atom := number | identifier | '(' expr ')'
//! ```
//!
//! `*` and `/` have equal precedence and associate left-to-right, and juxtaposition
//! (`kilogram meter`) means multiplication. Everything the spec writes in §12.2 and
//! §25 parses: `9.31e-9 meter^2 / second`, `-57.3 kilojoule / mole`,
//! `2.0e5 / second`, `6.5e-24 kilogram*meter/second`, `20 electronvolt`.
//!
//! # The `a/b*c` problem
//!
//! `kilojoule/mole*kelvin` is written by working scientists to mean
//! `kilojoule/(mole*kelvin)`, but read by every programming language as
//! `(kilojoule/mole)*kelvin`. Those differ by `kelvin^2`. Lattice applies the
//! left-to-right rule — guessing at intent in a units system is exactly the kind of
//! silent reinterpretation P1 forbids — and emits a [`UnitWarning`] naming the
//! grouping it used, so the model report shows the user what happened.
//!
//! # Affine temperature scales
//!
//! `celsius` and `fahrenheit` are affine, not linear: `20 °C` is `293.15 K`, but
//! `20 °C` is not `20 K` times anything. They are accepted only as a standalone unit
//! (`298 kelvin`, `25 celsius`) and rejected inside compound expressions, where the
//! offset has no consistent meaning.

use crate::dimension::Dimension;
use crate::error::{UnitError, UnitWarning};
use crate::quantity::Quantity;
use crate::registry::UnitRegistry;

/// Largest absolute exponent accepted in a unit expression.
///
/// Bounds dimension exponents well inside `i8`, so downstream dimensional arithmetic
/// cannot overflow from parsed input.
const MAX_EXPONENT: i32 = 32;

/// A parse result plus any non-fatal warnings.
#[derive(Clone, Debug, PartialEq)]
pub struct Parsed<T> {
    /// The parsed value.
    pub value: T,
    /// Warnings that should be surfaced in the model report but do not block use.
    pub warnings: Vec<UnitWarning>,
}

impl<T> Parsed<T> {
    /// Discard warnings and take the value.
    pub fn into_value(self) -> T {
        self.value
    }

    /// True when the input parsed without any ambiguity warnings.
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

/// A resolved unit expression: SI value = `raw * scale + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitTerm {
    /// Multiplicative factor to coherent SI base units.
    pub scale: f64,
    /// Additive offset; nonzero only for a standalone affine temperature scale.
    pub offset: f64,
    /// The dimension the expression measures.
    pub dim: Dimension,
}

impl UnitTerm {
    /// A dimensionless factor of 1.
    pub const IDENTITY: UnitTerm =
        UnitTerm { scale: 1.0, offset: 0.0, dim: Dimension::DIMENSIONLESS };

    fn mul(self, rhs: UnitTerm) -> Result<UnitTerm, UnitError> {
        Ok(UnitTerm {
            scale: self.scale * rhs.scale,
            offset: 0.0,
            dim: self.dim.try_mul(rhs.dim)?,
        })
    }

    fn div(self, rhs: UnitTerm) -> Result<UnitTerm, UnitError> {
        Ok(UnitTerm {
            scale: self.scale / rhs.scale,
            offset: 0.0,
            dim: self.dim.try_div(rhs.dim)?,
        })
    }

    fn powi(self, n: i32) -> Result<UnitTerm, UnitError> {
        Ok(UnitTerm { scale: self.scale.powi(n), offset: 0.0, dim: self.dim.try_powi(n)? })
    }
}

impl UnitRegistry {
    /// Parse a unit expression such as `"meter^2 / second"`.
    ///
    /// The returned [`UnitTerm::scale`] folds in any numeric literals, so
    /// `"2.0e5 / second"` yields `scale = 2e5` with dimension `1/s`.
    pub fn parse_unit(&self, src: &str) -> Result<Parsed<UnitTerm>, UnitError> {
        let tokens = tokenize(src)?;
        if tokens.is_empty() {
            return Ok(Parsed { value: UnitTerm::IDENTITY, warnings: Vec::new() });
        }

        // Affine scales are legal only as `<unit>` or `<number> <unit>`.
        if let Some(term) = self.try_parse_affine(src, &tokens)? {
            return Ok(Parsed { value: term, warnings: Vec::new() });
        }

        let mut parser = Parser { src, tokens: &tokens, pos: 0, reg: self, warnings: Vec::new() };
        let term = parser.parse_expr()?;
        parser.expect_end()?;
        Ok(Parsed { value: term, warnings: parser.warnings })
    }

    /// Parse a dimensioned literal such as `"-57.3 kilojoule / mole"` into a
    /// [`Quantity`] whose magnitude is in coherent SI base units.
    ///
    /// ```
    /// use lattice_units::{Dimension, UnitRegistry};
    /// let reg = UnitRegistry::si();
    /// let q = reg.parse_quantity("9.31e-9 meter^2 / second").unwrap().into_value();
    /// assert_eq!(q.dimension(), Dimension::DIFFUSIVITY);
    /// assert!((q.value() - 9.31e-9).abs() < 1e-24);
    /// ```
    pub fn parse_quantity(&self, src: &str) -> Result<Parsed<Quantity>, UnitError> {
        let Parsed { value: term, warnings } = self.parse_unit(src)?;
        Ok(Parsed { value: Quantity::new(term.scale + term.offset, term.dim), warnings })
    }

    /// Express a quantity in a target unit, returning the numeric magnitude.
    ///
    /// Used by exporters and the viewer: the runtime is all SI, but a user reading a
    /// plot wants `kJ/mol`. Fails if the target unit has a different dimension.
    ///
    /// ```
    /// use lattice_units::{DimensionalOp, UnitRegistry};
    /// let reg = UnitRegistry::si();
    /// let q = reg.parse_quantity("298.15 kelvin").unwrap().into_value();
    /// assert!((reg.value_in(&q, "celsius").unwrap() - 25.0).abs() < 1e-9);
    /// ```
    pub fn value_in(&self, q: &Quantity, unit_src: &str) -> Result<f64, UnitError> {
        let term = self.parse_unit(unit_src)?.into_value();
        if term.dim != q.dimension() {
            return Err(UnitError::Mismatch {
                op: crate::error::DimensionalOp::Comparison,
                expected: term.dim,
                found: q.dimension(),
            });
        }
        Ok((q.value() - term.offset) / term.scale)
    }

    /// Handle the two shapes an affine temperature may take, or return `None` so the
    /// general expression parser runs.
    fn try_parse_affine(&self, src: &str, tokens: &[Token]) -> Result<Option<UnitTerm>, UnitError> {
        let (magnitude, ident) = match tokens {
            [t] if t.kind == TokKind::Ident => (1.0, *t),
            [n, t] if matches!(n.kind, TokKind::Number(_)) && t.kind == TokKind::Ident => {
                let TokKind::Number(v) = n.kind else { unreachable!() };
                (v, *t)
            }
            _ => return Ok(None),
        };
        let name = &src[ident.start..ident.end];
        // A name that does not resolve at all is not this function's problem; let the
        // general parser produce the UnknownUnit error with its suggestion.
        let Ok(resolved) = self.resolve(name) else { return Ok(None) };
        if !resolved.def.is_affine() {
            return Ok(None);
        }
        Ok(Some(UnitTerm {
            scale: magnitude * resolved.scale(),
            offset: resolved.offset(),
            dim: resolved.dim(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum TokKind {
    Number(f64),
    Ident,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

#[derive(Clone, Copy, Debug)]
struct Token {
    kind: TokKind,
    start: usize,
    end: usize,
}

impl Token {
    /// True when this token can begin an atom, which is how juxtaposition
    /// (`kilogram meter`) is detected as implicit multiplication.
    fn starts_atom(&self) -> bool {
        matches!(self.kind, TokKind::Number(_) | TokKind::Ident | TokKind::LParen)
    }
}

/// Characters that may appear in a unit identifier besides letters, digits and `_`.
///
/// These are the unit symbols that are not ASCII letters: micro sign and Greek mu,
/// ohm sign and capital omega, angstrom sign and A-with-ring, degree sign, percent.
fn is_unit_symbol(c: char) -> bool {
    matches!(c, '%' | '\u{b0}' | '\u{b5}' | '\u{3bc}' | '\u{3a9}' | '\u{2126}' | '\u{c5}' | '\u{212b}')
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || is_unit_symbol(c)
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || is_unit_symbol(c)
}

fn tokenize(src: &str) -> Result<Vec<Token>, UnitError> {
    let bytes = src.as_bytes();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0usize;

    while i < src.len() {
        let c = src[i..].chars().next().expect("index is on a char boundary");
        let clen = c.len_utf8();

        if c.is_whitespace() {
            i += clen;
            continue;
        }

        // A sign is meaningful in exactly two places: on the leading magnitude
        // (`-57.3 kilojoule/mole`) and on an exponent (`second^-2`). Unit expressions
        // have no subtraction, so a `-` anywhere else is a typo, not an operator.
        let sign_position =
            tokens.is_empty() || tokens.last().is_some_and(|t| t.kind == TokKind::Caret);
        if (c == '-' || c == '+') && sign_position {
            let digits_at = i + clen;
            if !bytes.get(digits_at).is_some_and(u8::is_ascii_digit) {
                return Err(UnitError::Syntax {
                    message: format!("expected digits after `{c}`"),
                    offset: i,
                });
            }
            let (magnitude, end) = lex_number(src, digits_at)?;
            let value = if c == '-' { -magnitude } else { magnitude };
            tokens.push(Token { kind: TokKind::Number(value), start: i, end });
            i = end;
            continue;
        }

        // A digit, or a '.' that is immediately followed by a digit, starts a number.
        let starts_number = c.is_ascii_digit()
            || (c == '.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit));
        if starts_number {
            let (value, end) = lex_number(src, i)?;
            tokens.push(Token { kind: TokKind::Number(value), start: i, end });
            i = end;
            continue;
        }

        if is_ident_start(c) {
            let mut end = i + clen;
            while let Some(next) = src[end..].chars().next() {
                if is_ident_continue(next) {
                    end += next.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(Token { kind: TokKind::Ident, start: i, end });
            i = end;
            continue;
        }

        let kind = match c {
            // `·` U+00B7 MIDDLE DOT and `⋅` U+22C5 DOT OPERATOR both mean product.
            '*' | '\u{b7}' | '\u{22c5}' => TokKind::Star,
            '/' => TokKind::Slash,
            '^' => TokKind::Caret,
            '(' => TokKind::LParen,
            ')' => TokKind::RParen,
            other => {
                return Err(UnitError::Syntax {
                    message: format!("unexpected character `{other}`"),
                    offset: i,
                });
            }
        };
        // `**` is an accepted spelling of `^`.
        let mut end = i + clen;
        if kind == TokKind::Star && src[end..].starts_with('*') {
            end += 1;
            tokens.push(Token { kind: TokKind::Caret, start: i, end });
            i = end;
            continue;
        }
        tokens.push(Token { kind, start: i, end });
        i = end;
    }

    Ok(tokens)
}

/// Lex a decimal literal with optional fraction and exponent, returning its value and
/// end offset. The exponent is consumed only if it is well-formed, so `2e` lexes as
/// the number `2` followed by the identifier `e`.
fn lex_number(src: &str, start: usize) -> Result<(f64, usize), UnitError> {
    let bytes = src.as_bytes();
    let mut end = start;

    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    if bytes.get(end) == Some(&b'.') {
        let mut frac = end + 1;
        while bytes.get(frac).is_some_and(u8::is_ascii_digit) {
            frac += 1;
        }
        // A trailing '.' with no digits (`2.`) is still a valid f64 literal.
        end = frac;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let mut exp = end + 1;
        if matches!(bytes.get(exp), Some(b'+' | b'-')) {
            exp += 1;
        }
        if bytes.get(exp).is_some_and(u8::is_ascii_digit) {
            while bytes.get(exp).is_some_and(u8::is_ascii_digit) {
                exp += 1;
            }
            end = exp;
        }
    }

    let text = &src[start..end];
    let value = text.parse::<f64>().map_err(|_| UnitError::Syntax {
        message: format!("`{text}` is not a valid number"),
        offset: start,
    })?;
    Ok((value, end))
}

// ---------------------------------------------------------------------------
// Recursive-descent parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    src: &'a str,
    tokens: &'a [Token],
    pos: usize,
    reg: &'a UnitRegistry,
    warnings: Vec<UnitWarning>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<Token> {
        let t = self.peek();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect_end(&self) -> Result<(), UnitError> {
        match self.peek() {
            None => Ok(()),
            Some(t) => Err(UnitError::Syntax {
                message: format!("unexpected `{}`", &self.src[t.start..t.end]),
                offset: t.start,
            }),
        }
    }

    fn parse_expr(&mut self) -> Result<UnitTerm, UnitError> {
        let expr_start = self.peek().map_or(0, |t| t.start);
        let mut acc = self.parse_term()?;
        // Byte offset of the first `/` at this depth, cleared once we have warned so
        // that a single expression produces at most one ambiguity warning.
        let mut pending_slash: Option<usize> = None;

        while let Some(t) = self.peek() {
            let (is_product, op_offset) = match t.kind {
                TokKind::Star => {
                    self.advance();
                    (true, t.start)
                }
                TokKind::Slash => {
                    self.advance();
                    (false, t.start)
                }
                // Juxtaposition: `kilogram meter` is a product with no operator.
                _ if t.starts_atom() => (true, t.start),
                _ => break,
            };

            let rhs_start = self.peek().map_or(self.src.len(), |t| t.start);
            let rhs = self.parse_term()?;
            let rhs_end = self.tokens[self.pos - 1].end;

            if is_product {
                if pending_slash.is_some() {
                    self.warnings.push(UnitWarning::AmbiguousSlashProduct {
                        offset: op_offset,
                        interpretation: format!(
                            "({}) * {}",
                            self.src[expr_start..op_offset].trim(),
                            self.src[rhs_start..rhs_end].trim()
                        ),
                    });
                    pending_slash = None;
                }
                acc = acc.mul(rhs)?;
            } else {
                acc = acc.div(rhs)?;
                pending_slash.get_or_insert(op_offset);
            }
        }
        Ok(acc)
    }

    fn parse_term(&mut self) -> Result<UnitTerm, UnitError> {
        let atom = self.parse_atom()?;
        if self.peek().is_some_and(|t| t.kind == TokKind::Caret) {
            self.advance();
            let exponent = self.parse_exponent()?;
            return atom.powi(exponent);
        }
        Ok(atom)
    }

    fn parse_exponent(&mut self) -> Result<i32, UnitError> {
        // The tokenizer emits `-2` as a number, so an exponent arrives either as a
        // Number token (possibly negative) or as an identifier-free numeric literal.
        let Some(t) = self.advance() else {
            return Err(UnitError::Syntax {
                message: "expected an exponent after `^`".to_string(),
                offset: self.src.len(),
            });
        };
        // The tokenizer folds a sign directly after `^` into the number, so a
        // well-formed exponent always arrives as a single Number token.
        let TokKind::Number(value) = t.kind else {
            return Err(UnitError::Syntax {
                message: format!(
                    "expected an exponent after `^`, found `{}`",
                    &self.src[t.start..t.end]
                ),
                offset: t.start,
            });
        };
        if value.fract() != 0.0 {
            return Err(UnitError::Syntax {
                message: format!(
                    "exponent must be a whole number, found {value}; \
                     fractional dimensions are not representable"
                ),
                offset: t.start,
            });
        }
        let exponent = value as i32;
        if exponent.abs() > MAX_EXPONENT {
            return Err(UnitError::Syntax {
                message: format!("exponent {exponent} exceeds the supported range ±{MAX_EXPONENT}"),
                offset: t.start,
            });
        }
        Ok(exponent)
    }

    fn parse_atom(&mut self) -> Result<UnitTerm, UnitError> {
        let Some(t) = self.advance() else {
            return Err(UnitError::Syntax {
                message: "expected a unit or number".to_string(),
                offset: self.src.len(),
            });
        };
        match t.kind {
            TokKind::Number(v) => {
                Ok(UnitTerm { scale: v, offset: 0.0, dim: Dimension::DIMENSIONLESS })
            }
            TokKind::Ident => {
                let name = &self.src[t.start..t.end];
                let resolved = self.reg.resolve(name)?;
                if resolved.def.is_affine() {
                    return Err(UnitError::OffsetUnitInExpression { name: name.to_string() });
                }
                Ok(UnitTerm { scale: resolved.scale(), offset: 0.0, dim: resolved.dim() })
            }
            TokKind::LParen => {
                let inner = self.parse_expr()?;
                match self.advance() {
                    Some(Token { kind: TokKind::RParen, .. }) => Ok(inner),
                    Some(other) => Err(UnitError::Syntax {
                        message: format!("expected `)`, found `{}`", &self.src[other.start..other.end]),
                        offset: other.start,
                    }),
                    None => Err(UnitError::Syntax {
                        message: "unclosed `(`".to_string(),
                        offset: t.start,
                    }),
                }
            }
            TokKind::Slash => Err(UnitError::Syntax {
                message: "expected a unit or number before `/`".to_string(),
                offset: t.start,
            }),
            other => Err(UnitError::Syntax {
                message: format!("unexpected {other:?}"),
                offset: t.start,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Dimension;

    fn reg() -> UnitRegistry {
        UnitRegistry::si()
    }

    fn q(src: &str) -> Quantity {
        reg().parse_quantity(src).unwrap_or_else(|e| panic!("{src}: {e}")).into_value()
    }

    fn approx(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1.0)
    }

    #[test]
    fn plain_numbers_are_dimensionless() {
        let v = q("1.5");
        assert!(v.dimension().is_dimensionless());
        assert_eq!(v.value(), 1.5);
    }

    #[test]
    fn simple_dimensioned_literal() {
        let v = q("298 kelvin");
        assert_eq!(v.dimension(), Dimension::TEMPERATURE);
        assert_eq!(v.value(), 298.0);
    }

    /// Straight from spec §12.2.
    #[test]
    fn diffusion_coefficient_from_spec() {
        let v = q("9.31e-9 meter^2 / second");
        assert_eq!(v.dimension(), Dimension::DIFFUSIVITY);
        assert!(approx(v.value(), 9.31e-9, 1e-12));
    }

    /// Straight from spec §12.2 — negative enthalpy in kJ/mol.
    #[test]
    fn molar_enthalpy_from_spec() {
        let v = q("-57.3 kilojoule / mole");
        assert_eq!(v.dimension(), Dimension::MOLAR_ENERGY);
        assert!(approx(v.value(), -57_300.0, 1e-12));
    }

    /// Straight from spec §25.1 — an Arrhenius prefactor with no explicit magnitude.
    #[test]
    fn reciprocal_time_with_leading_one() {
        let v = q("2.0e5 / second");
        assert_eq!(v.dimension(), Dimension::FREQUENCY);
        assert!(approx(v.value(), 2.0e5, 1e-12));

        let one_over = q("1 / second");
        assert_eq!(one_over.dimension(), Dimension::FREQUENCY);
        assert_eq!(one_over.value(), 1.0);
    }

    /// Straight from spec §25.2 — a momentum written with explicit `*`.
    #[test]
    fn momentum_from_spec() {
        let v = q("6.5e-24 kilogram*meter/second");
        assert_eq!(v.dimension(), Dimension::MOMENTUM);
        assert!(approx(v.value(), 6.5e-24, 1e-12));
    }

    /// Straight from spec §25.1 — areal concentration on a 2D grid.
    #[test]
    fn areal_concentration_from_spec() {
        let v = q("1 mole / meter^2");
        assert_eq!(v.dimension(), Dimension::AREAL_CONCENTRATION);
        assert_eq!(v.value(), 1.0);
    }

    /// Straight from spec §25.2 — barrier height in electronvolts.
    #[test]
    fn electronvolt_barrier_from_spec() {
        let v = q("20 electronvolt");
        assert_eq!(v.dimension(), Dimension::ENERGY);
        assert!(approx(v.value(), 20.0 * 1.602_176_634e-19, 1e-12));
    }

    #[test]
    fn bare_unit_means_one_of_it() {
        let v = q("electron_mass");
        assert_eq!(v.dimension(), Dimension::MASS);
        assert!(approx(v.value(), 9.109_383_713_9e-31, 1e-12));
    }

    #[test]
    fn juxtaposition_is_multiplication() {
        let a = q("9.81 meter second^-2");
        let b = q("9.81 meter / second^2");
        assert_eq!(a.dimension(), Dimension::ACCELERATION);
        assert_eq!(a.dimension(), b.dimension());
        assert!(approx(a.value(), b.value(), 1e-15));
    }

    #[test]
    fn middle_dot_and_double_star_are_accepted() {
        let a = q("1 kilogram\u{b7}meter/second**2");
        assert_eq!(a.dimension(), Dimension::FORCE);
        assert!(approx(a.value(), 1.0, 1e-15));
    }

    #[test]
    fn parentheses_group_denominators() {
        let v = q("8.314 joule / (mole * kelvin)");
        assert_eq!(v.dimension(), Dimension::MOLAR_HEAT_CAPACITY);
        assert!(approx(v.value(), 8.314, 1e-12));
    }

    /// The `a/b*c` case: Lattice groups left-to-right and says so.
    #[test]
    fn slash_then_star_warns_and_groups_left_to_right() {
        let parsed = reg().parse_quantity("8.314 joule / mole * kelvin").unwrap();
        // Left-to-right gives J·K/mol, NOT the J/(mol·K) a chemist would intend.
        let joule_kelvin_per_mole =
            Dimension::MOLAR_ENERGY.try_mul(Dimension::TEMPERATURE).unwrap();
        assert_eq!(parsed.value.dimension(), joule_kelvin_per_mole);
        assert_ne!(parsed.value.dimension(), Dimension::MOLAR_HEAT_CAPACITY);
        assert!(!parsed.is_clean(), "expected an ambiguity warning");
        let text = parsed.warnings[0].to_string();
        assert!(text.contains("left-to-right"), "{text}");
        assert!(text.contains("parentheses"), "{text}");
    }

    #[test]
    fn unambiguous_expressions_produce_no_warnings() {
        for src in [
            "9.31e-9 meter^2 / second",
            "6.5e-24 kilogram*meter/second",
            "8.314 joule / (mole * kelvin)",
            "1 newton",
        ] {
            assert!(reg().parse_quantity(src).unwrap().is_clean(), "{src} warned unexpectedly");
        }
    }

    #[test]
    fn celsius_converts_to_kelvin() {
        let v = q("25 celsius");
        assert_eq!(v.dimension(), Dimension::TEMPERATURE);
        assert!(approx(v.value(), 298.15, 1e-12));
        assert!(approx(q("0 celsius").value(), 273.15, 1e-12));
        assert!(approx(q("\u{b0}C").value(), 274.15, 1e-12));
    }

    #[test]
    fn fahrenheit_converts_to_kelvin() {
        assert!(approx(q("32 fahrenheit").value(), 273.15, 1e-12));
        assert!(approx(q("212 fahrenheit").value(), 373.15, 1e-12));
        assert!(approx(q("-40 degF").value(), q("-40 celsius").value(), 1e-12));
    }

    #[test]
    fn celsius_in_a_compound_expression_is_rejected() {
        let err = reg().parse_quantity("5 joule / celsius").unwrap_err();
        assert!(matches!(err, UnitError::OffsetUnitInExpression { .. }));
        let text = err.to_string();
        assert!(text.contains("kelvin"), "error should point at the fix: {text}");
    }

    #[test]
    fn value_in_converts_back_out_of_si() {
        let r = reg();
        let energy = r.parse_quantity("35 kilojoule / mole").unwrap().into_value();
        assert!(approx(r.value_in(&energy, "kilojoule/mole").unwrap(), 35.0, 1e-12));
        assert!(approx(r.value_in(&energy, "joule/mole").unwrap(), 35_000.0, 1e-12));

        let t = r.parse_quantity("298.15 kelvin").unwrap().into_value();
        assert!(approx(r.value_in(&t, "celsius").unwrap(), 25.0, 1e-9));
    }

    #[test]
    fn value_in_rejects_mismatched_dimensions() {
        let r = reg();
        let energy = r.parse_quantity("35 kilojoule").unwrap().into_value();
        assert!(matches!(r.value_in(&energy, "second"), Err(UnitError::Mismatch { .. })));
    }

    #[test]
    fn unknown_unit_reports_position_and_suggestion() {
        let err = reg().parse_quantity("5 metre_").unwrap_err();
        match err {
            UnitError::UnknownUnit { name, suggestion } => {
                assert_eq!(name, "metre_");
                assert_eq!(suggestion.as_deref(), Some("metre"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn malformed_expressions_are_syntax_errors() {
        for src in [
            "5 meter /",
            "5 (meter",
            "5 meter^",
            "5 meter ) second",
            "5 meter @ second",
            "5 meter^-",
            "5 - meter",
        ] {
            let err = reg().parse_quantity(src).unwrap_err();
            assert!(
                matches!(err, UnitError::Syntax { .. }),
                "{src} should be a syntax error, got {err:?}"
            );
        }
    }

    #[test]
    fn negative_exponents_parse() {
        let v = q("9.81 meter second^-2");
        assert_eq!(v.dimension(), Dimension::ACCELERATION);
        assert!(approx(v.value(), 9.81, 1e-12));

        let inverse = q("1 second^-1");
        assert_eq!(inverse.dimension(), Dimension::FREQUENCY);

        // An explicit `+` is accepted and means the same as no sign.
        assert_eq!(q("1 meter^+2").dimension(), Dimension::AREA);
    }

    #[test]
    fn fractional_exponents_are_rejected_not_rounded() {
        let err = reg().parse_quantity("1 meter^0.5").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("whole number"), "{text}");
    }

    #[test]
    fn huge_exponents_are_rejected() {
        assert!(reg().parse_quantity("1 meter^100").is_err());
    }

    #[test]
    fn number_lexing_handles_exponent_edge_cases() {
        assert_eq!(lex_number("2e", 0).unwrap(), (2.0, 1));
        assert_eq!(lex_number("2e5", 0).unwrap(), (2e5, 3));
        assert_eq!(lex_number("2e+5", 0).unwrap(), (2e5, 4));
        assert_eq!(lex_number("9.31e-9", 0).unwrap().0, 9.31e-9);
        assert_eq!(lex_number("2.", 0).unwrap(), (2.0, 2));
    }

    #[test]
    fn empty_input_is_dimensionless_one() {
        let v = q("");
        assert!(v.dimension().is_dimensionless());
        assert_eq!(v.value(), 1.0);
    }
}
