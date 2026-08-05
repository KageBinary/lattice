//! Tokenizer for the `.lattice` project language.
//!
//! # Units are not a lexical concern
//!
//! The obvious way to lex `35 kilojoule / mole` is to notice the number and switch
//! into a "unit mode" that consumes what follows. That approach then has to decide
//! what `100 / dt` means, where `dt` is a declared parameter and not a unit — and it
//! cannot, because the lexer does not know what has been declared.
//!
//! So there is no unit mode. `kilojoule` lexes as a plain identifier, and *name
//! resolution* decides what it is: declared names win, and anything left over is
//! looked up in the unit registry. `35 kilojoule / mole` becomes
//! `35 × kilojoule ÷ mole` where the last two resolve to quantities; `100 / dt`
//! becomes a division by a parameter. Both fall out of one rule, and an identifier
//! that is neither gets a diagnostic naming both possibilities.
//!
//! The one thing this costs is juxtaposition: `35 kilojoule` has no operator between
//! its terms. The parser treats adjacency as multiplication, which is what the
//! notation means in every physics text ever written.

use crate::diagnostic::{Diagnostic, Diagnostics};
use crate::source::{SourceFile, Span};

/// A reserved word.
///
/// The list is deliberately short. `grid`, `reaction`, `potential`, `wavepacket` and
/// `detector` are *not* reserved: they are ordinary declaration kinds parsed by the
/// generic `<kind> <name> { … }` form, and the compiler decides which ones it knows.
/// Two things fall out of that. New solver families need no grammar change, and
/// spec §25.2's `grid: [768, 384];` — where `grid` is a *setting key* inside a domain
/// block — parses without a special case.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Keyword {
    /// `project`
    Project,
    /// `field`
    Field,
    /// `species`
    Species,
    /// `domain`
    Domain,
    /// `solve`
    Solve,
    /// `couple`
    Couple,
    /// `observe`
    Observe,
    /// `visualize`
    Visualize,
    /// `import`
    Import,
    /// `on`
    On,
    /// `with`
    With,
    /// `as`
    As,
    /// `every`
    Every,
    /// `conserve`
    Conserve,
    /// `true`
    True,
    /// `false`
    False,
}

impl Keyword {
    /// Every reserved word.
    pub const ALL: &'static [Keyword] = &[
        Keyword::Project,
        Keyword::Field,
        Keyword::Species,
        Keyword::Domain,
        Keyword::Solve,
        Keyword::Couple,
        Keyword::Observe,
        Keyword::Visualize,
        Keyword::Import,
        Keyword::On,
        Keyword::With,
        Keyword::As,
        Keyword::Every,
        Keyword::Conserve,
        Keyword::True,
        Keyword::False,
    ];

    /// The spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Keyword::Project => "project",
            Keyword::Field => "field",
            Keyword::Species => "species",
            Keyword::Domain => "domain",
            Keyword::Solve => "solve",
            Keyword::Couple => "couple",
            Keyword::Observe => "observe",
            Keyword::Visualize => "visualize",
            Keyword::Import => "import",
            Keyword::On => "on",
            Keyword::With => "with",
            Keyword::As => "as",
            Keyword::Every => "every",
            Keyword::Conserve => "conserve",
            Keyword::True => "true",
            Keyword::False => "false",
        }
    }

    /// Recognize a reserved word.
    ///
    /// Named `parse` rather than `from_str` so it is not mistaken for the standard
    /// `FromStr` trait method, which has different semantics (`Result`, not `Option`).
    pub fn parse(text: &str) -> Option<Keyword> {
        Keyword::ALL.iter().copied().find(|k| k.as_str() == text)
    }
}

/// What a token is.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TokenKind {
    /// An identifier, or a unit name — the distinction is made later.
    Ident,
    /// A numeric literal, already parsed.
    Number(f64),
    /// A string literal. The span includes the quotes.
    Str,
    /// A reserved word.
    Keyword(Keyword),
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `->`
    Arrow,
    /// `=`
    Equals,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `^`
    Caret,
    /// End of input.
    Eof,
}

impl TokenKind {
    /// A human-readable name, for "expected X, found Y" messages.
    pub fn describe(self) -> String {
        match self {
            TokenKind::Ident => "an identifier".to_string(),
            TokenKind::Number(_) => "a number".to_string(),
            TokenKind::Str => "a string".to_string(),
            TokenKind::Keyword(k) => format!("`{}`", k.as_str()),
            TokenKind::Eof => "end of file".to_string(),
            other => format!("`{}`", other.symbol()),
        }
    }

    /// The punctuation spelling, or `""` for tokens that carry text.
    pub const fn symbol(self) -> &'static str {
        match self {
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::Semicolon => ";",
            TokenKind::Colon => ":",
            TokenKind::Comma => ",",
            TokenKind::Dot => ".",
            TokenKind::Arrow => "->",
            TokenKind::Equals => "=",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Caret => "^",
            _ => "",
        }
    }
}

/// A lexed token.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Token {
    /// What it is.
    pub kind: TokenKind,
    /// Where it is.
    pub span: Span,
}

impl Token {
    /// True if this token is the given keyword.
    pub fn is_keyword(&self, keyword: Keyword) -> bool {
        self.kind == TokenKind::Keyword(keyword)
    }
}

/// Characters that may appear in an identifier besides letters, digits and `_`.
///
/// `°` is the only unit symbol that is not already `char::is_alphabetic` — the micro
/// sign, Greek mu, ohm sign and angstrom sign all are.
fn is_extra_ident_char(c: char) -> bool {
    c == '\u{b0}'
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || is_extra_ident_char(c)
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || is_extra_ident_char(c)
}

/// Tokenize a source file.
///
/// Lexing recovers from bad input rather than stopping at the first problem: an
/// unexpected character is reported and skipped, so a single run reports every
/// lexical error instead of one per compile.
pub fn tokenize(file: &SourceFile) -> (Vec<Token>, Diagnostics) {
    let text = file.text();
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut diagnostics = Diagnostics::new();
    let mut i = 0usize;

    while i < text.len() {
        let c = text[i..].chars().next().expect("index is on a char boundary");
        let clen = c.len_utf8();

        if c.is_whitespace() {
            i += clen;
            continue;
        }

        // Line comment.
        if text[i..].starts_with("//") {
            i = text[i..].find('\n').map_or(text.len(), |offset| i + offset);
            continue;
        }

        // Block comment, nesting.
        if text[i..].starts_with("/*") {
            let start = i;
            let mut depth = 0usize;
            loop {
                if text[i..].starts_with("/*") {
                    depth += 1;
                    i += 2;
                } else if text[i..].starts_with("*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else if i >= text.len() {
                    diagnostics.push(
                        Diagnostic::error("unterminated block comment")
                            .with_code("E0002")
                            .at(Span::new(start as u32, (start + 2) as u32), "opened here")
                            .help("add `*/` to close it"),
                    );
                    break;
                } else {
                    i += text[i..].chars().next().map_or(1, char::len_utf8);
                }
            }
            continue;
        }

        // Number.
        if c.is_ascii_digit() {
            let (value, end) = lex_number(text, i);
            match value {
                Some(value) => {
                    tokens.push(Token { kind: TokenKind::Number(value), span: Span::new(i as u32, end as u32) })
                }
                None => diagnostics.push(
                    Diagnostic::error(format!("`{}` is not a valid number", &text[i..end]))
                        .with_code("E0003")
                        .span(Span::new(i as u32, end as u32)),
                ),
            }
            i = end;
            continue;
        }

        // Identifier or keyword.
        if is_ident_start(c) {
            let start = i;
            i += clen;
            while let Some(next) = text[i..].chars().next() {
                if is_ident_continue(next) {
                    i += next.len_utf8();
                } else {
                    break;
                }
            }
            let span = Span::new(start as u32, i as u32);
            let kind = match Keyword::parse(&text[start..i]) {
                Some(keyword) => TokenKind::Keyword(keyword),
                None => TokenKind::Ident,
            };
            tokens.push(Token { kind, span });
            continue;
        }

        // String literal.
        if c == '"' {
            let start = i;
            i += 1;
            let mut terminated = false;
            while i < text.len() {
                let next = text[i..].chars().next().expect("on a boundary");
                i += next.len_utf8();
                if next == '\\' && i < text.len() {
                    i += text[i..].chars().next().map_or(1, char::len_utf8);
                } else if next == '"' {
                    terminated = true;
                    break;
                } else if next == '\n' {
                    break;
                }
            }
            let span = Span::new(start as u32, i as u32);
            if terminated {
                tokens.push(Token { kind: TokenKind::Str, span });
            } else {
                diagnostics.push(
                    Diagnostic::error("unterminated string literal")
                        .with_code("E0004")
                        .at(span, "started here")
                        .help("string literals may not span lines"),
                );
            }
            continue;
        }

        // Punctuation.
        let two = if bytes.len() >= i + 2 { &text[i..i + 2] } else { "" };
        let (kind, width) = match two {
            "->" => (TokenKind::Arrow, 2),
            _ => {
                let single = match c {
                    '{' => TokenKind::LBrace,
                    '}' => TokenKind::RBrace,
                    '(' => TokenKind::LParen,
                    ')' => TokenKind::RParen,
                    '[' => TokenKind::LBracket,
                    ']' => TokenKind::RBracket,
                    ';' => TokenKind::Semicolon,
                    ':' => TokenKind::Colon,
                    ',' => TokenKind::Comma,
                    '.' => TokenKind::Dot,
                    '=' => TokenKind::Equals,
                    '+' => TokenKind::Plus,
                    '-' => TokenKind::Minus,
                    // `·` and `⋅` are accepted as multiplication so unit expressions
                    // can be written the way they are printed.
                    '*' | '\u{b7}' | '\u{22c5}' => TokenKind::Star,
                    '/' => TokenKind::Slash,
                    '^' => TokenKind::Caret,
                    other => {
                        diagnostics.push(
                            Diagnostic::error(format!("unexpected character `{other}`"))
                                .with_code("E0001")
                                .span(Span::new(i as u32, (i + clen) as u32)),
                        );
                        i += clen;
                        continue;
                    }
                };
                (single, clen)
            }
        };
        tokens.push(Token { kind, span: Span::new(i as u32, (i + width) as u32) });
        i += width;
    }

    tokens.push(Token { kind: TokenKind::Eof, span: Span::new(text.len() as u32, text.len() as u32) });
    (tokens, diagnostics)
}

/// Lex a decimal literal, returning its value and end offset.
///
/// The exponent is consumed only when well-formed, so `2e` lexes as `2` followed by
/// the identifier `e` rather than as a malformed number.
fn lex_number(text: &str, start: usize) -> (Option<f64>, usize) {
    let bytes = text.as_bytes();
    let mut end = start;

    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    // A `.` is part of the number only when a digit follows, so `4.field` stays a
    // number followed by a member access.
    if bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
        end += 1;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let mut exponent = end + 1;
        if matches!(bytes.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        if bytes.get(exponent).is_some_and(u8::is_ascii_digit) {
            while bytes.get(exponent).is_some_and(u8::is_ascii_digit) {
                exponent += 1;
            }
            end = exponent;
        }
    }

    (text[start..end].parse::<f64>().ok(), end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> (Vec<TokenKind>, Diagnostics) {
        let file = SourceFile::new("t.lattice", source);
        let (tokens, diagnostics) = tokenize(&file);
        (tokens.into_iter().map(|t| t.kind).collect(), diagnostics)
    }

    fn kinds(source: &str) -> Vec<TokenKind> {
        let (kinds, diagnostics) = lex(source);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&SourceFile::new("t", source)));
        kinds
    }

    #[test]
    fn keywords_are_distinguished_from_identifiers() {
        let k = kinds("project field chamber");
        assert_eq!(k[0], TokenKind::Keyword(Keyword::Project));
        assert_eq!(k[1], TokenKind::Keyword(Keyword::Field));
        assert_eq!(k[2], TokenKind::Ident, "`chamber` is not reserved");
    }

    /// Declaration kinds are ordinary identifiers, which is what lets `grid` be both
    /// a declaration kind and a setting key (spec §25.2 uses it as both).
    #[test]
    fn declaration_kinds_are_not_reserved() {
        for word in ["grid", "reaction", "material", "potential", "particles", "wavepacket", "detector"] {
            assert_eq!(kinds(word)[0], TokenKind::Ident, "`{word}` should not be reserved");
        }
    }

    #[test]
    fn every_reserved_word_round_trips() {
        for keyword in Keyword::ALL {
            assert_eq!(
                kinds(keyword.as_str())[0],
                TokenKind::Keyword(*keyword),
                "{}",
                keyword.as_str()
            );
        }
    }

    #[test]
    fn punctuation_lexes_including_the_arrow() {
        let k = kinds("{ } ( ) [ ] ; : , . -> = + - * / ^");
        let expected = [
            TokenKind::LBrace,
            TokenKind::RBrace,
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::LBracket,
            TokenKind::RBracket,
            TokenKind::Semicolon,
            TokenKind::Colon,
            TokenKind::Comma,
            TokenKind::Dot,
            TokenKind::Arrow,
            TokenKind::Equals,
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Caret,
            TokenKind::Eof,
        ];
        assert_eq!(k, expected);
    }

    /// `->` must not lex as `-` followed by `>`, or every coupling statement breaks.
    #[test]
    fn the_arrow_beats_a_bare_minus() {
        assert_eq!(kinds("a->b")[1], TokenKind::Arrow);
        assert_eq!(kinds("a - b")[1], TokenKind::Minus);
    }

    #[test]
    fn numbers_cover_the_forms_the_spec_uses() {
        for (source, expected) in [
            ("512", 512.0),
            ("0.002", 0.002),
            ("9.31e-9", 9.31e-9),
            ("2.0e5", 2.0e5),
            ("6.5e-24", 6.5e-24),
            ("298", 298.0),
        ] {
            match kinds(source)[0] {
                TokenKind::Number(value) => assert_eq!(value, expected, "{source}"),
                other => panic!("{source} lexed as {other:?}"),
            }
        }
    }

    /// `2e` is a number followed by an identifier, not a malformed number — the
    /// exponent is only consumed when it is well-formed.
    #[test]
    fn a_bare_exponent_marker_is_not_consumed() {
        let k = kinds("2e");
        assert_eq!(k[0], TokenKind::Number(2.0));
        assert_eq!(k[1], TokenKind::Ident);
    }

    /// `4.field` must be a number and a member access, not `4.` and `field`.
    #[test]
    fn a_dot_without_a_following_digit_is_member_access() {
        let k = kinds("4.concentration");
        assert_eq!(k[0], TokenKind::Number(4.0));
        assert_eq!(k[1], TokenKind::Dot);
        assert_eq!(k[2], TokenKind::Ident);
    }

    /// The central design decision: unit names are plain identifiers.
    #[test]
    fn units_lex_as_identifiers() {
        let k = kinds("35 kilojoule / mole");
        assert_eq!(k[0], TokenKind::Number(35.0));
        assert_eq!(k[1], TokenKind::Ident, "`kilojoule` is just an identifier here");
        assert_eq!(k[2], TokenKind::Slash);
        assert_eq!(k[3], TokenKind::Ident);
    }

    #[test]
    fn unit_symbols_lex_as_identifiers() {
        for source in ["µm", "μs", "Ω", "Å", "°C", "kelvin"] {
            let k = kinds(source);
            assert_eq!(k[0], TokenKind::Ident, "{source} should be an identifier");
            assert_eq!(k[1], TokenKind::Eof, "{source} should be one token");
        }
    }

    #[test]
    fn middle_dot_is_multiplication() {
        assert_eq!(kinds("kg\u{b7}m")[1], TokenKind::Star);
    }

    #[test]
    fn line_and_block_comments_are_skipped() {
        let k = kinds("a // comment\nb /* block */ c");
        assert_eq!(k, [TokenKind::Ident, TokenKind::Ident, TokenKind::Ident, TokenKind::Eof]);
    }

    #[test]
    fn block_comments_nest() {
        let k = kinds("a /* outer /* inner */ still outer */ b");
        assert_eq!(k, [TokenKind::Ident, TokenKind::Ident, TokenKind::Eof]);
    }

    #[test]
    fn an_unterminated_block_comment_is_reported() {
        let (_, diagnostics) = lex("a /* never closed");
        assert!(diagnostics.has_errors());
        assert_eq!(diagnostics.codes(), ["E0002"]);
    }

    #[test]
    fn strings_lex_with_escapes() {
        let k = kinds(r#"basis: "def2-SVP";"#);
        assert_eq!(k[2], TokenKind::Str);
        assert_eq!(kinds(r#""a\"b""#)[0], TokenKind::Str);
    }

    #[test]
    fn an_unterminated_string_is_reported() {
        let (_, diagnostics) = lex("\"open\n");
        assert_eq!(diagnostics.codes(), ["E0004"]);
    }

    /// Lexing recovers so one run reports every problem, not just the first.
    #[test]
    fn lexing_recovers_and_reports_every_bad_character() {
        let (kinds, diagnostics) = lex("a @ b # c");
        assert_eq!(diagnostics.error_count(), 2, "both `@` and `#` should be reported");
        assert_eq!(
            kinds.iter().filter(|k| **k == TokenKind::Ident).count(),
            3,
            "the identifiers around them still lex"
        );
    }

    #[test]
    fn spans_point_at_the_right_text() {
        let file = SourceFile::new("t.lattice", "field temperature on chamber;");
        let (tokens, _) = tokenize(&file);
        assert_eq!(file.slice(tokens[0].span), "field");
        assert_eq!(file.slice(tokens[1].span), "temperature");
        assert_eq!(file.slice(tokens[3].span), "chamber");
    }

    #[test]
    fn an_empty_file_yields_just_eof() {
        assert_eq!(kinds(""), [TokenKind::Eof]);
        assert_eq!(kinds("   \n\t  // nothing\n"), [TokenKind::Eof]);
    }

    /// A spec example, lexed end to end.
    #[test]
    fn the_spec_grid_declaration_lexes() {
        let k = kinds("grid chamber { size: [512, 256]; extent: [2 meter, 1 meter]; }");
        assert_eq!(k[0], TokenKind::Ident, "`grid` is a declaration kind, not a keyword");
        assert_eq!(k[1], TokenKind::Ident);
        assert_eq!(k[2], TokenKind::LBrace);
        assert!(k.contains(&TokenKind::Number(512.0)));
        assert_eq!(k[k.len() - 1], TokenKind::Eof);
    }

    #[test]
    fn token_descriptions_read_well_in_messages() {
        assert_eq!(TokenKind::Ident.describe(), "an identifier");
        assert_eq!(TokenKind::Semicolon.describe(), "`;`");
        assert_eq!(TokenKind::Keyword(Keyword::On).describe(), "`on`");
        assert_eq!(TokenKind::Eof.describe(), "end of file");
    }
}
