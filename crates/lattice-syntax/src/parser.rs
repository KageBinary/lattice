//! Recursive-descent parser for the `.lattice` project language.
//!
//! # Grammar
//!
//! ```text
//! project  := 'project' IDENT '{' item* '}'
//! item     := setting | grid | field | species | block | domain | law | let
//!           | solve | couple | observe | visualize
//! setting  := IDENT ':' expr ('->' expr)? ';'           -- `->` for stoichiometry
//! grid     := 'grid' IDENT '{' setting* '}'
//! field    := ('field' | 'species') IDENT ('on' IDENT)? ('=' expr)? (';' | '{' setting* '}')
//! block    := ('reaction' | 'material' | 'potential' | 'particles') IDENT '{' setting* '}'
//! domain   := 'domain' IDENT IDENT '{' setting* '}'
//! law      := IDENT IDENT '(' (IDENT ':' type),* ')' ('->' type)? '{' stmt* '}'
//! stmt     := 'let' IDENT '=' expr ';' | 'param' IDENT ':' type ('=' expr)? ';'
//!           | 'return' expr ';'
//! type     := 'vec2' '<' add '>' | add
//! let      := 'let' IDENT '=' expr ';'
//! solve    := 'solve' IDENT '(' args ')' 'with' IDENT '(' args ')' ';'
//! couple   := 'couple' path '->' path ('conserve' IDENT)? ';'
//! observe  := 'observe' expr ('every' expr)? ';'
//! visualize:= 'visualize' expr 'as' IDENT ';'
//!
//! expr     := or
//! or       := and ('||' and)*
//! and      := cmp ('&&' cmp)*
//! cmp      := add (('<' | '<=' | '>' | '>=' | '==' | '!=') add)?   -- never chained
//! add      := mul (('+' | '-') mul)*
//! mul      := unary (('*' | '/') unary | IDENT)*        -- adjacency is multiplication
//! unary    := ('-' | '+' | '!')* power
//! power    := postfix ('^' SIGNED_INT)?
//! postfix  := primary ('.' IDENT | '(' args ')')*
//! primary  := NUMBER | STRING | 'true' | 'false' | IDENT | if
//!           | '(' expr (',' expr)* ')' | '[' (expr (',' expr)*)? ']'
//! if       := 'if' expr '{' expr '}' 'else' (if | '{' expr '}')
//! ```
//!
//! # Laws are values, not programs
//!
//! A law body is a sequence of `let` bindings ending in a `return` — spec §8.3's
//! *"restricted, typed, side-effect-free language"*. There is no assignment, no loop,
//! and `if` is an expression with both arms required, so every law computes exactly one
//! value on every path. The parser accepts any statement order; the compiler checks
//! that the body ends in its `return`.
//!
//! # Adjacency is multiplication
//!
//! `35 kilojoule / mole` has no operator between `35` and `kilojoule`. That is how
//! every physics text writes it, and since [units are ordinary
//! identifiers](crate::lexer), the parser only has to know that an identifier
//! immediately after an expression continues a product. The rule is narrow — only a
//! bare `IDENT` triggers it, never `(`, which stays a call — so it cannot swallow
//! syntax that means something else.
//!
//! # Recovery
//!
//! A malformed item does not abort the parse. The parser skips to the next `;` or the
//! end of the enclosing block, tracking brace depth, and continues. One run therefore
//! reports every problem in a file rather than the first, which matters when a user is
//! fixing a model they just typed.

use crate::ast::*;
use crate::diagnostic::{Diagnostic, Diagnostics};
use crate::lexer::{tokenize, Keyword, Token, TokenKind};
use crate::source::{SourceFile, Span};

/// Stop reporting after this many errors, to avoid burying the real one in cascades.
const MAX_ERRORS: usize = 100;

/// Largest permitted exponent, matching the bound in the unit parser so dimensional
/// arithmetic downstream cannot overflow.
const MAX_EXPONENT: i32 = 32;

/// Parse a source file into a project.
///
/// Returns `None` for the project only when nothing parseable was found; a project
/// with recovered errors is still returned so later passes can report more.
pub fn parse(file: &SourceFile) -> (Option<Project>, Diagnostics) {
    let (tokens, mut diagnostics) = tokenize(file);
    let mut parser = Parser { file, tokens, pos: 0, diagnostics: Diagnostics::new() };
    let project = parser.parse_project();
    diagnostics.extend(parser.diagnostics);
    (project, diagnostics)
}

struct Parser<'a> {
    file: &'a SourceFile,
    tokens: Vec<Token>,
    pos: usize,
    diagnostics: Diagnostics,
}

impl<'a> Parser<'a> {
    // --- token access ------------------------------------------------------

    fn peek(&self) -> Token {
        self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_at(&self, offset: usize) -> Token {
        self.tokens[(self.pos + offset).min(self.tokens.len() - 1)]
    }

    fn at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }

    fn advance(&mut self) -> Token {
        let token = self.peek();
        if !self.at_end() {
            self.pos += 1;
        }
        token
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn eat(&mut self, kind: TokenKind) -> Option<Token> {
        if self.check(kind) { Some(self.advance()) } else { None }
    }

    fn text(&self, span: Span) -> &str {
        self.file.slice(span)
    }

    // --- diagnostics -------------------------------------------------------

    fn error_labelled(
        &mut self,
        message: impl Into<String>,
        span: Span,
        label: impl Into<String>,
        code: &str,
    ) {
        if self.diagnostics.error_count() < MAX_ERRORS {
            self.diagnostics.push(Diagnostic::error(message).with_code(code).at(span, label));
        }
    }

    fn expect(&mut self, kind: TokenKind, context: &str) -> Result<Token, ()> {
        if self.check(kind) {
            return Ok(self.advance());
        }
        let found = self.peek();
        self.error_labelled(
            format!("expected {} {context}, found {}", kind.describe(), found.kind.describe()),
            found.span,
            format!("expected {}", kind.describe()),
            "E0101",
        );
        Err(())
    }

    fn expect_ident(&mut self, context: &str) -> Result<Ident, ()> {
        let token = self.peek();
        if token.kind == TokenKind::Ident {
            self.advance();
            return Ok(Ident::new(self.text(token.span), token.span));
        }
        // A keyword where a name belongs is worth calling out specifically: it is the
        // most common way a model breaks after the reserved-word list grows.
        if let TokenKind::Keyword(keyword) = token.kind {
            self.advance();
            self.diagnostics.push(
                Diagnostic::error(format!("`{}` is a reserved word and cannot be used as {context}", keyword.as_str()))
                    .with_code("E0102")
                    .at(token.span, "reserved word")
                    .help("choose a different name"),
            );
            return Err(());
        }
        self.error_labelled(
            format!("expected {context}, found {}", token.kind.describe()),
            token.span,
            "expected a name",
            "E0101",
        );
        Err(())
    }

    /// Skip forward to a point where parsing can restart.
    ///
    /// Consumes through the next `;` at the current brace depth, or stops before the
    /// `}` that closes the enclosing block. Tracking depth is what keeps a bad
    /// statement inside a nested block from terminating the whole item.
    fn recover(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.peek().kind {
                TokenKind::Eof => return,
                TokenKind::LBrace => {
                    depth += 1;
                    self.advance();
                }
                TokenKind::RBrace => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.advance();
                }
                TokenKind::Semicolon => {
                    self.advance();
                    if depth == 0 {
                        return;
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    // --- items -------------------------------------------------------------

    fn parse_project(&mut self) -> Option<Project> {
        // Skip anything before `project` so a stray token does not hide the whole file.
        if !self.peek().is_keyword(Keyword::Project) {
            let found = self.peek();
            if found.kind == TokenKind::Eof {
                self.diagnostics.push(
                    Diagnostic::error("this file declares no project")
                        .with_code("E0100")
                        .help("a model file starts with `project <name> { … }`"),
                );
                return None;
            }
            self.error_labelled(
                format!("expected `project`, found {}", found.kind.describe()),
                found.span,
                "a model file must start here with `project`",
                "E0100",
            );
            return None;
        }

        let start = self.advance().span;
        let name = self.expect_ident("a project name").ok()?;
        self.expect(TokenKind::LBrace, "after the project name").ok()?;

        let mut items = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_end() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(()) => self.recover(),
            }
        }

        let end = match self.expect(TokenKind::RBrace, "to close the project") {
            Ok(token) => token.span,
            Err(()) => self.peek().span,
        };
        Some(Project { name, items, span: start.merge(end) })
    }

    fn parse_item(&mut self) -> Result<Item, ()> {
        let token = self.peek();

        // `key: value;` is checked before the keyword dispatch, so a reserved word
        // can still be a setting key. Spec §25.2 needs this: `grid: [768, 384];`
        // inside a domain block would otherwise collide with the declaration form.
        if self.peek_at(1).kind == TokenKind::Colon
            && matches!(token.kind, TokenKind::Ident | TokenKind::Keyword(_))
        {
            return Ok(Item::Setting(self.parse_setting()?));
        }

        match token.kind {
            TokenKind::Keyword(Keyword::Field) => Ok(Item::Field(self.parse_field_decl()?)),
            TokenKind::Keyword(Keyword::Species) => Ok(Item::Species(self.parse_field_decl()?)),
            TokenKind::Keyword(Keyword::Domain) => Ok(Item::Domain(self.parse_domain_decl()?)),
            TokenKind::Keyword(Keyword::Solve) => Ok(Item::Solve(self.parse_solve()?)),
            TokenKind::Keyword(Keyword::Couple) => Ok(Item::Couple(self.parse_couple()?)),
            TokenKind::Keyword(Keyword::Observe) => Ok(Item::Observe(self.parse_observe()?)),
            TokenKind::Keyword(Keyword::Visualize) => Ok(Item::Visualize(self.parse_visualize()?)),
            TokenKind::Keyword(Keyword::Let) => Ok(Item::Let(self.parse_let_item()?)),
            // `<kind> <name>(…) -> type { … }` — a user-defined law. The `(` after the
            // name is what tells it from a block declaration.
            TokenKind::Ident
                if self.peek_at(1).kind == TokenKind::Ident && self.peek_at(2).kind == TokenKind::LParen =>
            {
                Ok(Item::Law(self.parse_law()?))
            }
            // `<kind> <name> …` — grid, reaction, potential, detector, and anything
            // a future solver family introduces. Which kinds are meaningful is the
            // compiler's question, not the grammar's.
            TokenKind::Ident if self.peek_at(1).kind == TokenKind::Ident => {
                Ok(Item::Decl(self.parse_decl()?))
            }
            _ => {
                self.error_labelled(
                    format!("expected a declaration, found {}", token.kind.describe()),
                    token.span,
                    "not a declaration",
                    "E0103",
                );
                self.diagnostics.push(Diagnostic::new(
                    crate::diagnostic::Severity::Note,
                    "a project contains `name: value;` settings, `<kind> <name> { … }` \
                     declarations, `<kind> <name>(…) { … }` laws, `let` constants, and the \
                     statements `field`, `species`, `domain`, `solve`, `couple`, `observe`, \
                     `visualize`",
                ));
                Err(())
            }
        }
    }

    /// A setting key, which may be a reserved word when followed by `:`.
    fn parse_setting_key(&mut self) -> Result<Ident, ()> {
        let token = self.peek();
        let acceptable = matches!(token.kind, TokenKind::Ident)
            || (matches!(token.kind, TokenKind::Keyword(_))
                && self.peek_at(1).kind == TokenKind::Colon);
        if acceptable {
            self.advance();
            return Ok(Ident::new(self.text(token.span), token.span));
        }
        self.error_labelled(
            format!("expected a setting name, found {}", token.kind.describe()),
            token.span,
            "expected a name",
            "E0101",
        );
        Err(())
    }

    fn parse_setting(&mut self) -> Result<Setting, ()> {
        let key = self.parse_setting_key()?;
        self.expect(TokenKind::Colon, "after a setting name")?;
        let mut value = self.parse_expr()?;
        // `reactants -> products`, at the lowest precedence and only as a whole value,
        // so `->` keeps exactly one meaning wherever else it appears.
        if self.eat(TokenKind::Arrow).is_some() {
            let products = self.parse_expr()?;
            let span = value.span.merge(products.span);
            value = Expr::new(ExprKind::Yields(Box::new(value), Box::new(products)), span);
        }
        let end = self.expect(TokenKind::Semicolon, "after a setting value")?;
        let span = key.span.merge(end.span);
        Ok(Setting { key, value, span })
    }

    fn parse_settings_block(&mut self) -> Result<(Vec<Setting>, Span), ()> {
        let open = self.expect(TokenKind::LBrace, "to open a block")?;
        let mut settings = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_end() {
            match self.parse_setting() {
                Ok(setting) => settings.push(setting),
                Err(()) => {
                    self.recover();
                    if self.check(TokenKind::RBrace) {
                        break;
                    }
                }
            }
        }
        let close = self.expect(TokenKind::RBrace, "to close the block")?;
        Ok((settings, open.span.merge(close.span)))
    }

    /// `<kind> <name> { … }` or `<kind> <name> <modifier> <args>;`
    fn parse_decl(&mut self) -> Result<Decl, ()> {
        let kind = self.expect_ident("a declaration kind")?;
        let start = kind.span;
        let name = self.expect_ident("a name")?;

        if self.check(TokenKind::LBrace) {
            let (settings, block_span) = self.parse_settings_block()?;
            return Ok(Decl {
                kind,
                name,
                modifier: None,
                arguments: Vec::new(),
                settings,
                span: start.merge(block_span),
            });
        }

        // Statement form, as in spec §25.2's `detector screen at x=4.5 nanometer;`.
        let modifier =
            if self.check(TokenKind::Ident) { Some(self.expect_ident("a modifier")?) } else { None };
        let arguments = if modifier.is_some() {
            self.parse_arguments(TokenKind::Semicolon)?
        } else {
            Vec::new()
        };
        let end = self.expect(TokenKind::Semicolon, "to end the declaration")?;
        Ok(Decl { kind, name, modifier, arguments, settings: Vec::new(), span: start.merge(end.span) })
    }

    /// `let name = value;` at the top level of a project.
    fn parse_let_item(&mut self) -> Result<LetDecl, ()> {
        let start = self.advance().span;
        let name = self.expect_ident("a name after `let`")?;
        self.expect(TokenKind::Equals, "after the name in a `let`")?;
        let value = self.parse_expr()?;
        let end = self.expect(TokenKind::Semicolon, "to end the `let`")?;
        Ok(LetDecl { name, value, span: start.merge(end.span) })
    }

    /// `<kind> <name>(<params>) -> <type> { <stmts> }`
    fn parse_law(&mut self) -> Result<LawDecl, ()> {
        let kind = self.expect_ident("a law kind")?;
        let name = self.expect_ident("a law name")?;
        self.expect(TokenKind::LParen, "to open the law's parameters")?;
        let mut params = Vec::new();
        while !self.check(TokenKind::RParen) {
            let param_name = self.expect_ident("a parameter name")?;
            self.expect(TokenKind::Colon, "after a parameter name")?;
            let ty = self.parse_type()?;
            let span = param_name.span.merge(ty.span);
            params.push(LawParam { name: param_name, ty, span });
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }
        self.expect(TokenKind::RParen, "to close the law's parameters")?;
        let returns = if self.eat(TokenKind::Arrow).is_some() { Some(self.parse_type()?) } else { None };

        self.expect(TokenKind::LBrace, "to open the law's body")?;
        // Where the body ends, found before parsing it: a statement that fails inside
        // an `if` arm leaves the parser between braces, and only the matching brace
        // says which `}` closes the law rather than the arm.
        let close_at = self.matching_brace(self.pos - 1);
        let mut body = Vec::new();
        let mut recovered = false;
        loop {
            let at_close = close_at.map_or_else(|| self.check(TokenKind::RBrace), |end| self.pos >= end);
            if at_close || self.at_end() {
                break;
            }
            let start = self.pos;
            match self.parse_stmt() {
                Ok(stmt) => body.push(stmt),
                Err(()) => {
                    recovered = true;
                    self.pos = start;
                    self.skip_statement(close_at);
                }
            }
        }
        let close = self.expect(TokenKind::RBrace, "to close the law's body")?;
        Ok(LawDecl { span: kind.span.merge(close.span), kind, name, params, returns, body, recovered })
    }

    /// The index of the `}` matching the `{` at `open`, if the braces balance.
    fn matching_brace(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for (index, token) in self.tokens.iter().enumerate().skip(open) {
            match token.kind {
                TokenKind::LBrace => depth += 1,
                TokenKind::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(index);
                    }
                }
                TokenKind::Eof => return None,
                _ => {}
            }
        }
        None
    }

    /// From the start of a statement, skip past its `;`, counting braces from the
    /// statement's own start so a `;` inside an `if` arm does not end it, and never
    /// past the law's closing brace.
    fn skip_statement(&mut self, close_at: Option<usize>) {
        let mut depth = 0usize;
        loop {
            if close_at.is_some_and(|end| self.pos >= end) || self.at_end() {
                return;
            }
            match self.peek().kind {
                TokenKind::LBrace => depth += 1,
                TokenKind::RBrace if depth == 0 => return,
                TokenKind::RBrace => depth -= 1,
                TokenKind::Semicolon if depth == 0 => {
                    self.advance();
                    return;
                }
                _ => {}
            }
            self.advance();
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ()> {
        let token = self.peek();
        let kind = match token.kind {
            TokenKind::Keyword(Keyword::Let) => {
                self.advance();
                let name = self.expect_ident("a name after `let`")?;
                self.expect(TokenKind::Equals, "after the name in a `let`")?;
                StmtKind::Let { name, value: self.parse_expr()? }
            }
            TokenKind::Keyword(Keyword::Return) => {
                self.advance();
                StmtKind::Return(self.parse_expr()?)
            }
            // `param` is not reserved: it opens a statement only here, where nothing
            // else could stand.
            TokenKind::Ident if self.text(token.span) == "param" && self.peek_at(1).kind == TokenKind::Ident => {
                self.advance();
                let name = self.expect_ident("a parameter name")?;
                self.expect(TokenKind::Colon, "after the parameter name; a `param` declares its unit")?;
                let ty = self.parse_type()?;
                let default = if self.eat(TokenKind::Equals).is_some() { Some(self.parse_expr()?) } else { None };
                StmtKind::Param { name, ty, default }
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error(format!("expected `let`, `param` or `return`, found {}", token.kind.describe()))
                        .with_code("E0112")
                        .at(token.span, "not a statement")
                        .note(
                            "a law's body names values with `let`, declares what its use site binds \
                             with `param`, and ends with `return`; there is no assignment and no loop",
                        ),
                );
                return Err(());
            }
        };
        let end = self.expect(TokenKind::Semicolon, "to end the statement")?;
        Ok(Stmt { kind, span: token.span.merge(end.span) })
    }

    /// `vec2<unit>`, or a plain name or unit expression.
    ///
    /// The unit inside the brackets is parsed above the comparison level, so the `>`
    /// that closes it is never read as a comparison.
    fn parse_type(&mut self) -> Result<TypeExpr, ()> {
        let token = self.peek();
        if token.kind == TokenKind::Ident && self.text(token.span) == "vec2" && self.peek_at(1).kind == TokenKind::Lt {
            self.advance();
            self.advance();
            let unit = self.parse_additive()?;
            let close = self.expect(TokenKind::Gt, "to close `vec2<…>`")?;
            return Ok(TypeExpr { kind: TypeKind::Vec2(unit), span: token.span.merge(close.span) });
        }
        let ty = self.parse_additive()?;
        Ok(TypeExpr { span: ty.span, kind: TypeKind::Plain(ty) })
    }

    fn parse_field_decl(&mut self) -> Result<FieldDecl, ()> {
        let start = self.advance().span;
        let name = self.expect_ident("a name")?;

        let grid = if self.peek().is_keyword(Keyword::On) {
            self.advance();
            Some(self.expect_ident("a grid name after `on`")?)
        } else {
            None
        };

        let mut initial = None;
        if self.eat(TokenKind::Equals).is_some() {
            initial = Some(self.parse_expr()?);
        }

        if self.check(TokenKind::LBrace) {
            let (settings, block_span) = self.parse_settings_block()?;
            return Ok(FieldDecl { name, grid, initial, settings, span: start.merge(block_span) });
        }

        let end = self.expect(TokenKind::Semicolon, "to end the declaration")?;
        Ok(FieldDecl { name, grid, initial, settings: Vec::new(), span: start.merge(end.span) })
    }

    fn parse_domain_decl(&mut self) -> Result<DomainDecl, ()> {
        let start = self.advance().span;
        let family = self.expect_ident("a solver family")?;
        let name = self.expect_ident("an instance name")?;
        let (settings, block_span) = self.parse_settings_block()?;
        Ok(DomainDecl { family, name, settings, span: start.merge(block_span) })
    }

    fn parse_solve(&mut self) -> Result<SolveStmt, ()> {
        let start = self.advance().span;
        let solver = self.expect_ident("a solver name")?;
        self.expect(TokenKind::LParen, "after the solver name")?;
        let targets = self.parse_arguments(TokenKind::RParen)?;
        self.expect(TokenKind::RParen, "to close the solver arguments")?;

        if !self.peek().is_keyword(Keyword::With) {
            let found = self.peek();
            self.error_labelled(
                format!("expected `with` to name the method, found {}", found.kind.describe()),
                found.span,
                "expected `with`",
                "E0104",
            );
            return Err(());
        }
        self.advance();

        let method = self.expect_ident("a method name after `with`")?;
        let mut parameters = Vec::new();
        if self.eat(TokenKind::LParen).is_some() {
            parameters = self.parse_arguments(TokenKind::RParen)?;
            self.expect(TokenKind::RParen, "to close the method parameters")?;
        }
        let end = self.expect(TokenKind::Semicolon, "to end the solve statement")?;
        Ok(SolveStmt { solver, targets, method, parameters, span: start.merge(end.span) })
    }

    fn parse_couple(&mut self) -> Result<CoupleStmt, ()> {
        let start = self.advance().span;
        let source = self.parse_path()?;
        if self.eat(TokenKind::Arrow).is_none() {
            let found = self.peek();
            self.error_labelled(
                format!("expected `->` between the coupled ports, found {}", found.kind.describe()),
                found.span,
                "expected `->`",
                "E0105",
            );
            return Err(());
        }
        let target = self.parse_path()?;

        let conserve = if self.peek().is_keyword(Keyword::Conserve) {
            self.advance();
            Some(self.expect_ident("a conserved quantity after `conserve`")?)
        } else {
            None
        };
        let end = self.expect(TokenKind::Semicolon, "to end the couple statement")?;
        Ok(CoupleStmt { source, target, conserve, span: start.merge(end.span) })
    }

    fn parse_observe(&mut self) -> Result<ObserveStmt, ()> {
        let start = self.advance().span;
        let target = self.parse_expr()?;
        let every = if self.peek().is_keyword(Keyword::Every) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let end = self.expect(TokenKind::Semicolon, "to end the observe statement")?;
        Ok(ObserveStmt { target, every, span: start.merge(end.span) })
    }

    fn parse_visualize(&mut self) -> Result<VisualizeStmt, ()> {
        let start = self.advance().span;
        let target = self.parse_expr()?;
        // The style is optional: spec §25.2 writes `visualize probability_density;`
        // and lets the domain pick a default encoding.
        let style = if self.peek().is_keyword(Keyword::As) {
            self.advance();
            Some(self.expect_ident("a visual style after `as`")?)
        } else {
            None
        };
        let end = self.expect(TokenKind::Semicolon, "to end the visualize statement")?;
        Ok(VisualizeStmt { target, style, span: start.merge(end.span) })
    }

    fn parse_path(&mut self) -> Result<Path, ()> {
        let first = self.expect_ident("a port path")?;
        let mut span = first.span;
        let mut segments = vec![first];
        while self.eat(TokenKind::Dot).is_some() {
            let segment = self.expect_ident("a path segment after `.`")?;
            span = span.merge(segment.span);
            segments.push(segment);
        }
        Ok(Path { segments, span })
    }

    // --- expressions -------------------------------------------------------

    fn parse_arguments(&mut self, terminator: TokenKind) -> Result<Vec<Argument>, ()> {
        let mut arguments = Vec::new();
        if self.check(terminator) {
            return Ok(arguments);
        }
        loop {
            arguments.push(self.parse_argument()?);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
            // Tolerate a trailing comma before the terminator.
            if self.check(terminator) {
                break;
            }
        }
        Ok(arguments)
    }

    fn parse_argument(&mut self) -> Result<Argument, ()> {
        // `name = value` when an identifier is immediately followed by `=`. A reserved
        // word is accepted as the name too: `rdf(every=10)` reads the way `observe … every`
        // does, and no expression can begin with a keyword followed by `=`, so it cannot
        // be mistaken for anything else.
        let token = self.peek();
        let named = matches!(token.kind, TokenKind::Ident | TokenKind::Keyword(_))
            && self.peek_at(1).kind == TokenKind::Equals;
        if named {
            self.advance();
            let name = Ident::new(self.text(token.span), token.span);
            self.advance();
            let value = self.parse_expr()?;
            let span = name.span.merge(value.span);
            return Ok(Argument { name: Some(name), value, span });
        }
        let value = self.parse_expr()?;
        let span = value.span;
        Ok(Argument { name: None, value, span })
    }

    /// The entry point for expressions.
    pub(crate) fn parse_expr(&mut self) -> Result<Expr, ()> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_and()?;
        while self.eat(TokenKind::OrOr).is_some() {
            let right = self.parse_and()?;
            let span = left.span.merge(right.span);
            left = Expr::new(ExprKind::Binary(BinaryOp::Or, Box::new(left), Box::new(right)), span);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_comparison()?;
        while self.eat(TokenKind::AndAnd).is_some() {
            let right = self.parse_comparison()?;
            let span = left.span.merge(right.span);
            left = Expr::new(ExprKind::Binary(BinaryOp::And, Box::new(left), Box::new(right)), span);
        }
        Ok(left)
    }

    fn comparison_op(kind: TokenKind) -> Option<BinaryOp> {
        Some(match kind {
            TokenKind::Lt => BinaryOp::Lt,
            TokenKind::Le => BinaryOp::Le,
            TokenKind::Gt => BinaryOp::Gt,
            TokenKind::Ge => BinaryOp::Ge,
            TokenKind::EqEq => BinaryOp::Eq,
            TokenKind::NotEq => BinaryOp::Ne,
            _ => return None,
        })
    }

    /// One comparison at most. `a < b < c` reads as a range in mathematics and as
    /// `(a < b) < c` — a truth value compared with a quantity — in every grammar that
    /// allows it, so it is refused rather than given either meaning.
    fn parse_comparison(&mut self) -> Result<Expr, ()> {
        let left = self.parse_additive()?;
        let Some(op) = Self::comparison_op(self.peek().kind) else { return Ok(left) };
        self.advance();
        let right = self.parse_additive()?;
        let span = left.span.merge(right.span);
        if let Some(second) = Self::comparison_op(self.peek().kind) {
            let token = self.peek();
            self.diagnostics.push(
                Diagnostic::error(format!("comparisons do not chain: `{}` follows `{}`", second.symbol(), op.symbol()))
                    .with_code("E0111")
                    .at(token.span, "second comparison")
                    .help("write a range as two comparisons joined by `&&`, as in `a < b && b < c`"),
            );
            return Err(());
        }
        Ok(Expr::new(ExprKind::Binary(op, Box::new(left), Box::new(right)), span))
    }

    /// `if condition { value } else { value }`, with `else if` chaining.
    fn parse_if(&mut self) -> Result<Expr, ()> {
        let start = self.advance().span;
        let condition = self.parse_expr()?;
        let then = self.parse_braced_expr("the `if` value")?;
        if !self.peek().is_keyword(Keyword::Else) {
            let found = self.peek();
            self.diagnostics.push(
                Diagnostic::error(format!("an `if` needs an `else`, found {}", found.kind.describe()))
                    .with_code("E0113")
                    .at(found.span, "expected `else`")
                    .note("`if` chooses between two values, so a law computes one on every path"),
            );
            return Err(());
        }
        self.advance();
        let otherwise = if self.peek().is_keyword(Keyword::If) {
            self.parse_if()?
        } else {
            self.parse_braced_expr("the `else` value")?
        };
        let span = start.merge(otherwise.span);
        Ok(Expr::new(ExprKind::If(Box::new(condition), Box::new(then), Box::new(otherwise)), span))
    }

    fn parse_braced_expr(&mut self, what: &str) -> Result<Expr, ()> {
        let open = self.expect(TokenKind::LBrace, &format!("to open {what}"))?;
        let value = self.parse_expr()?;
        let close = self.expect(TokenKind::RBrace, &format!("to close {what}"))?;
        Ok(Expr::new(value.kind, open.span.merge(close.span)))
    }

    fn parse_additive(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek().kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            let span = left.span.merge(right.span);
            left = Expr::new(ExprKind::Binary(op, Box::new(left), Box::new(right)), span);
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_juxtaposed()?;
        loop {
            let op = match self.peek().kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                _ => break,
            };
            self.advance();
            let right = self.parse_juxtaposed()?;
            let span = left.span.merge(right.span);
            left = Expr::new(ExprKind::Binary(op, Box::new(left), Box::new(right)), span);
        }
        Ok(left)
    }

    /// Adjacency: `35 kilojoule`, `9.81 meter second^-2`.
    ///
    /// This binds *tighter* than `*` and `/`, which is the difference between
    /// `10 meter / 2 second` meaning 5 m/s and meaning 5 m·s. A dimensioned literal is
    /// one thing, and treating it as one atom is what makes division behave the way
    /// every physics text assumes.
    ///
    /// Only a bare identifier continues a juxtaposition, so `f(x)` stays a call and a
    /// following `[` stays a list.
    fn parse_juxtaposed(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_unary()?;
        while self.check(TokenKind::Ident) {
            let right = self.parse_power()?;
            let span = left.span.merge(right.span);
            left = Expr::new(
                ExprKind::Binary(BinaryOp::Mul, Box::new(left), Box::new(right)),
                span,
            );
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ()> {
        let token = self.peek();
        let op = match token.kind {
            TokenKind::Minus => UnaryOp::Neg,
            TokenKind::Plus => UnaryOp::Pos,
            TokenKind::Bang => UnaryOp::Not,
            _ => return self.parse_power(),
        };
        self.advance();
        let operand = self.parse_unary()?;
        let span = token.span.merge(operand.span);
        Ok(Expr::new(ExprKind::Unary(op, Box::new(operand)), span))
    }

    fn parse_power(&mut self) -> Result<Expr, ()> {
        let base = self.parse_postfix()?;
        if self.eat(TokenKind::Caret).is_none() {
            return Ok(base);
        }
        let (exponent, span) = self.parse_integer_exponent()?;
        let whole = base.span.merge(span);
        Ok(Expr::new(ExprKind::Power(Box::new(base), exponent), whole))
    }

    fn parse_integer_exponent(&mut self) -> Result<(i32, Span), ()> {
        let mut negative = false;
        let mut span = self.peek().span;
        if let Some(token) = self.eat(TokenKind::Minus) {
            negative = true;
            span = token.span;
        } else if let Some(token) = self.eat(TokenKind::Plus) {
            span = token.span;
        }

        let token = self.peek();
        let TokenKind::Number(value) = token.kind else {
            self.error_labelled(
                format!("expected a whole-number exponent, found {}", token.kind.describe()),
                token.span,
                "expected a number",
                "E0107",
            );
            return Err(());
        };
        self.advance();
        span = span.merge(token.span);

        if value.fract() != 0.0 {
            self.error_labelled(
                format!("exponent must be a whole number, found {value}"),
                span,
                "not a whole number",
                "E0108",
            );
            self.diagnostics.push(Diagnostic::new(
                crate::diagnostic::Severity::Note,
                "dimensions are tracked as integer exponents, so `meter^0.5` has no representation",
            ));
            return Err(());
        }
        let exponent = if negative { -(value as i32) } else { value as i32 };
        if exponent.abs() > MAX_EXPONENT {
            self.error_labelled(
                format!("exponent {exponent} exceeds the supported range ±{MAX_EXPONENT}"),
                span,
                "too large",
                "E0109",
            );
            return Err(());
        }
        Ok((exponent, span))
    }

    fn parse_postfix(&mut self) -> Result<Expr, ()> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.eat(TokenKind::Dot).is_some() {
                let member = self.expect_ident("a member name after `.`")?;
                let span = expr.span.merge(member.span);
                expr = Expr::new(ExprKind::Member(Box::new(expr), member), span);
                continue;
            }
            if self.check(TokenKind::LParen) {
                self.advance();
                let arguments = self.parse_arguments(TokenKind::RParen)?;
                let close = self.expect(TokenKind::RParen, "to close the call")?;
                let span = expr.span.merge(close.span);
                expr = Expr::new(ExprKind::Call(Box::new(expr), arguments), span);
                continue;
            }
            break;
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ()> {
        let token = self.peek();
        match token.kind {
            TokenKind::Number(value) => {
                self.advance();
                Ok(Expr::new(ExprKind::Number(value), token.span))
            }
            TokenKind::Str => {
                self.advance();
                let raw = self.text(token.span);
                Ok(Expr::new(ExprKind::Str(unescape(raw)), token.span))
            }
            TokenKind::Keyword(Keyword::True) => {
                self.advance();
                Ok(Expr::new(ExprKind::Bool(true), token.span))
            }
            TokenKind::Keyword(Keyword::False) => {
                self.advance();
                Ok(Expr::new(ExprKind::Bool(false), token.span))
            }
            TokenKind::Ident => {
                self.advance();
                Ok(Expr::new(ExprKind::Name(self.text(token.span).to_string()), token.span))
            }
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::LParen => {
                self.advance();
                let first = self.parse_expr()?;
                if self.check(TokenKind::Comma) {
                    let mut items = vec![first];
                    while self.eat(TokenKind::Comma).is_some() {
                        if self.check(TokenKind::RParen) {
                            break;
                        }
                        items.push(self.parse_expr()?);
                    }
                    let close = self.expect(TokenKind::RParen, "to close the tuple")?;
                    let span = token.span.merge(close.span);
                    return Ok(Expr::new(ExprKind::Tuple(items), span));
                }
                let close = self.expect(TokenKind::RParen, "to close the group")?;
                // A parenthesized expression keeps the wider span so diagnostics
                // underline the parentheses the user actually wrote.
                Ok(Expr::new(first.kind, token.span.merge(close.span)))
            }
            TokenKind::LBracket => {
                self.advance();
                let mut items = Vec::new();
                if !self.check(TokenKind::RBracket) {
                    loop {
                        items.push(self.parse_expr()?);
                        if self.eat(TokenKind::Comma).is_none() {
                            break;
                        }
                        if self.check(TokenKind::RBracket) {
                            break;
                        }
                    }
                }
                let close = self.expect(TokenKind::RBracket, "to close the list")?;
                let span = token.span.merge(close.span);
                Ok(Expr::new(ExprKind::List(items), span))
            }
            _ => {
                self.error_labelled(
                    format!("expected a value, found {}", token.kind.describe()),
                    token.span,
                    "expected a value",
                    "E0110",
                );
                Err(())
            }
        }
    }
}

/// Strip the surrounding quotes and resolve escapes.
fn unescape(raw: &str) -> String {
    let inner = raw.strip_prefix('"').unwrap_or(raw).strip_suffix('"').unwrap_or(raw);
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            // An unknown escape keeps both characters rather than silently dropping
            // the backslash, so a Windows path such as "C:\data" survives intact.
            // (`\t` really is a tab, so "C:\temp" does not — quote paths with `/`
            // or double the backslash.)
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(source: &str) -> Project {
        let file = SourceFile::new("t.lattice", source);
        let (project, diagnostics) = parse(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        project.expect("a project should have parsed")
    }

    fn parse_err(source: &str) -> (Option<Project>, String, Vec<String>) {
        let file = SourceFile::new("t.lattice", source);
        let (project, diagnostics) = parse(&file);
        assert!(diagnostics.has_errors(), "expected an error from:\n{source}");
        let codes = diagnostics.codes().into_iter().map(String::from).collect();
        (project, diagnostics.render(&file), codes)
    }

    fn expr_of(source: &str) -> Expr {
        let project = parse_ok(&format!("project p {{ x: {source}; }}"));
        match &project.items[0] {
            Item::Setting(setting) => setting.value.clone(),
            other => panic!("expected a setting, got {other:?}"),
        }
    }

    // --- structure ---------------------------------------------------------

    #[test]
    fn an_empty_project_parses() {
        let project = parse_ok("project empty { }");
        assert_eq!(project.name.text, "empty");
        assert!(project.items.is_empty());
    }

    #[test]
    fn top_level_settings_parse() {
        let project = parse_ok("project p { dimensions: 2; precision: mixed; fidelity: engineering_2d; }");
        assert_eq!(project.settings().count(), 3);
        assert_eq!(project.setting("precision").unwrap().value.as_name(), Some("mixed"));
        assert_eq!(project.setting("dimensions").unwrap().value.as_number(), Some(2.0));
    }

    #[test]
    fn a_grid_declaration_parses() {
        let project = parse_ok("project p { grid chamber { size: [512, 256]; extent: [2 meter, 1 meter]; } }");
        let grid = project.grids().next().unwrap();
        assert_eq!(grid.name.text, "chamber");
        assert_eq!(grid.settings.len(), 2);
        let size = grid.setting("size").unwrap().value.as_list().unwrap();
        assert_eq!(size[0].as_number(), Some(512.0));
    }

    #[test]
    fn field_and_species_declarations_parse() {
        let project = parse_ok(
            "project p {
               field temperature on chamber = 298 kelvin;
               species A on chamber = left_half(1 mole / meter^2);
               species H_plus { charge: 1 elementary_charge; }
             }",
        );
        let field = project.fields().next().unwrap();
        assert_eq!(field.name.text, "temperature");
        assert_eq!(field.grid.as_ref().unwrap().text, "chamber");
        assert!(field.initial.is_some());

        let species: Vec<&FieldDecl> = project.species().collect();
        assert_eq!(species.len(), 2);
        assert!(species[0].initial.is_some(), "A has an initial value");
        assert_eq!(species[1].settings.len(), 1, "H_plus uses the block form");
    }

    #[test]
    fn a_solve_statement_parses() {
        let project = parse_ok("project p { solve diffusion(A, B, C) with crank_nicolson(dt=0.002 second); }");
        let solve = project.solves().next().unwrap();
        assert_eq!(solve.solver.text, "diffusion");
        assert_eq!(solve.targets.len(), 3);
        assert_eq!(solve.method.text, "crank_nicolson");
        assert!(solve.parameter("dt").is_some());
        assert!(solve.parameter("nope").is_none());
    }

    #[test]
    fn a_couple_statement_parses() {
        let project = parse_ok("project p { couple A_plus_B.heat_release -> temperature.source conserve energy; }");
        let couple = project.couples().next().unwrap();
        assert_eq!(couple.source.to_string(), "A_plus_B.heat_release");
        assert_eq!(couple.target.to_string(), "temperature.source");
        assert_eq!(couple.source.root().text, "A_plus_B");
        assert_eq!(couple.source.tail().len(), 1);
        assert_eq!(couple.conserve.as_ref().unwrap().text, "energy");
    }

    #[test]
    fn couple_without_conserve_parses() {
        let project = parse_ok("project p { couple a.b -> c.d; }");
        assert!(project.couples().next().unwrap().conserve.is_none());
    }

    #[test]
    fn observe_and_visualize_parse() {
        let project = parse_ok(
            "project p {
               observe total_energy every 0.1 second;
               observe probability_norm;
               visualize temperature as heatmap;
               visualize [A, B, C] as rgb_mix;
             }",
        );
        let observes: Vec<&ObserveStmt> = project.observes().collect();
        assert!(observes[0].every.is_some());
        assert!(observes[1].every.is_none());

        let visuals: Vec<&VisualizeStmt> = project.visualizes().collect();
        assert_eq!(visuals[0].style.as_ref().unwrap().text, "heatmap");
        assert_eq!(visuals[1].target.as_list().unwrap().len(), 3);
    }

    #[test]
    fn visualize_without_a_style_parses() {
        let project = parse_ok("project p { visualize probability_density; }");
        assert!(project.visualizes().next().unwrap().style.is_none());
    }

    /// Declaration kinds are open. `grid`, `reaction` and a hypothetical future
    /// `wavepacket` all take the same path, so a new solver family needs no grammar
    /// change — only a compiler that knows the kind.
    #[test]
    fn declarations_of_any_kind_parse_uniformly() {
        let project = parse_ok(
            "project p {
               grid chamber { size: [4, 4]; }
               reaction r { rate: k; }
               potential barrier { height: 20 electronvolt; }
               wavepacket initial { sigma: 0.45 nanometer; }
             }",
        );
        assert_eq!(project.declarations().count(), 4);
        assert_eq!(project.declarations_of("grid").count(), 1);
        assert_eq!(project.declarations_of("wavepacket").count(), 1);
        assert_eq!(project.declaration("potential", "barrier").unwrap().settings.len(), 1);
        assert!(project.declaration("grid", "missing").is_none());
    }

    /// Spec §25.2's `detector screen at x=4.5 nanometer;` — the statement form of a
    /// declaration, with a modifier word and arguments.
    #[test]
    fn a_declaration_with_a_modifier_and_arguments_parses() {
        let project = parse_ok("project p { detector screen at x=4.5 nanometer; }");
        let decl = project.declarations_of("detector").next().unwrap();
        assert_eq!(decl.name.text, "screen");
        assert_eq!(decl.modifier.as_ref().unwrap().text, "at");
        assert_eq!(decl.arguments.len(), 1);
        assert!(decl.argument("x").is_some());
    }

    /// A reserved word is acceptable as a setting key, because before a `:` it cannot
    /// be anything else. Spec §25.2 relies on this for `grid:` inside a domain block.
    #[test]
    fn reserved_words_work_as_setting_keys() {
        let project = parse_ok(
            "project p { domain quantum2d q { grid: [768, 384]; field: psi; on: true; } }",
        );
        let domain = project.domains().next().unwrap();
        assert_eq!(domain.settings.len(), 3);
        assert!(domain.setting("grid").is_some());
        assert!(domain.setting("field").is_some());
    }

    #[test]
    fn a_domain_declaration_parses() {
        let project = parse_ok(
            "project p { domain quantum2d q { grid: [768, 384]; mass: electron_mass; } }",
        );
        let domain = project.domains().next().unwrap();
        assert_eq!(domain.family.text, "quantum2d");
        assert_eq!(domain.name.text, "q");
        assert_eq!(domain.setting("mass").unwrap().value.as_name(), Some("electron_mass"));
    }

    // --- expressions -------------------------------------------------------

    /// Adjacency means multiplication, which is what makes `35 kilojoule` work.
    #[test]
    fn adjacency_parses_as_multiplication() {
        let expr = expr_of("35 kilojoule");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Mul, left, right) => {
                assert_eq!(left.as_number(), Some(35.0));
                assert_eq!(right.as_name(), Some("kilojoule"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn multiplicative_operators_associate_left_to_right() {
        // `35 kilojoule / mole` must group as `(35 * kilojoule) / mole`.
        let expr = expr_of("35 kilojoule / mole");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Div, left, right) => {
                assert!(matches!(left.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
                assert_eq!(right.as_name(), Some("mole"));
            }
            other => panic!("{other:?}"),
        }
    }

    /// Adjacency binds tighter than `/`, so `10 meter / 2 second` is a velocity and
    /// not `m·s`. A dimensioned literal is one atom.
    #[test]
    fn adjacency_binds_tighter_than_division() {
        let expr = expr_of("10 meter / 2 second");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Div, left, right) => {
                // Both sides must themselves be `number × unit` products.
                assert!(
                    matches!(left.kind, ExprKind::Binary(BinaryOp::Mul, _, _)),
                    "left should be `10 meter`: {left:?}"
                );
                assert!(
                    matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)),
                    "right should be `2 second`, not just `2`: {right:?}"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn repeated_adjacency_chains_left_to_right() {
        // `9.81 meter second^-2` is `((9.81 * meter) * second^-2)`.
        let expr = expr_of("9.81 meter second^-2");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Mul, left, right) => {
                assert!(matches!(left.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
                assert!(matches!(right.kind, ExprKind::Power(_, -2)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn powers_bind_tighter_than_products() {
        // `meter^2 / second` is `(meter^2) / second`.
        let expr = expr_of("meter^2 / second");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Div, left, _) => {
                assert!(matches!(left.kind, ExprKind::Power(_, 2)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn negative_exponents_parse() {
        assert!(matches!(expr_of("second^-2").kind, ExprKind::Power(_, -2)));
        assert!(matches!(expr_of("meter^+2").kind, ExprKind::Power(_, 2)));
    }

    #[test]
    fn unary_minus_applies_to_the_magnitude() {
        // `-57.3 kilojoule / mole` is `((-57.3) * kilojoule) / mole`.
        let expr = expr_of("-57.3 kilojoule / mole");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Div, left, _) => match left.kind {
                ExprKind::Binary(BinaryOp::Mul, magnitude, _) => {
                    assert!(matches!(magnitude.kind, ExprKind::Unary(UnaryOp::Neg, _)));
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn addition_binds_loosest() {
        let expr = expr_of("a * b + c");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Add, left, right) => {
                assert!(matches!(left.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
                assert_eq!(right.as_name(), Some("c"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parentheses_override_precedence() {
        let expr = expr_of("joule / (mole * kelvin)");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Div, _, right) => {
                assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn member_access_and_calls_chain() {
        let expr = expr_of("arrhenius(A0=2.0e5 / second, Ea=35 kilojoule/mole) * A.concentration");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Mul, left, right) => {
                match left.kind {
                    ExprKind::Call(callee, args) => {
                        assert_eq!(callee.as_name(), Some("arrhenius"));
                        assert_eq!(args.len(), 2);
                        assert_eq!(args[0].name.as_ref().unwrap().text, "A0");
                        assert_eq!(args[1].name.as_ref().unwrap().text, "Ea");
                    }
                    other => panic!("{other:?}"),
                }
                assert!(matches!(right.kind, ExprKind::Member(_, _)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn positional_and_named_arguments_mix() {
        let expr = expr_of("f(a, b=2, c)");
        match expr.kind {
            ExprKind::Call(_, args) => {
                assert!(args[0].name.is_none());
                assert_eq!(args[1].name.as_ref().unwrap().text, "b");
                assert!(args[2].name.is_none());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_reserved_word_can_name_a_parameter() {
        let expr = expr_of("rdf(bins=10, every=5)");
        match expr.kind {
            ExprKind::Call(_, args) => {
                assert_eq!(args[1].name.as_ref().unwrap().text, "every");
                assert_eq!(args[1].value.as_number(), Some(5.0));
            }
            other => panic!("{other:?}"),
        }
        // Only before `=`: a bare keyword is still not a value.
        let file = SourceFile::new("t.lattice", "project p { x: f(every); }");
        let (_, diagnostics) = parse(&file);
        assert!(diagnostics.has_errors());
    }

    #[test]
    fn lists_and_tuples_parse() {
        assert_eq!(expr_of("[1, 2, 3]").as_list().unwrap().len(), 3);
        assert_eq!(expr_of("[]").as_list().unwrap().len(), 0);
        assert!(matches!(expr_of("(1, 2)").kind, ExprKind::Tuple(_)));
        // A single parenthesized value is just that value, not a one-tuple.
        assert_eq!(expr_of("(5)").as_number(), Some(5.0));
    }

    /// From spec §25.2: a list of tuples, scaled by a trailing unit.
    #[test]
    fn a_list_followed_by_a_unit_parses() {
        let expr = expr_of("[(-1.0, 0.35), (1.0, 0.35)] nanometer");
        match expr.kind {
            ExprKind::Binary(BinaryOp::Mul, left, right) => {
                assert_eq!(left.as_list().unwrap().len(), 2);
                assert_eq!(right.as_name(), Some("nanometer"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn trailing_commas_are_tolerated() {
        assert_eq!(expr_of("[1, 2,]").as_list().unwrap().len(), 2);
        match expr_of("f(a, b,)").kind {
            ExprKind::Call(_, args) => assert_eq!(args.len(), 2),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn strings_parse_and_unescape() {
        match expr_of(r#""def2-SVP""#).kind {
            ExprKind::Str(text) => assert_eq!(text, "def2-SVP"),
            other => panic!("{other:?}"),
        }
        match expr_of(r#""a\nb\\c\"d""#).kind {
            ExprKind::Str(text) => assert_eq!(text, "a\nb\\c\"d"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_unknown_escape_keeps_both_characters() {
        // `\d` means nothing, so both characters survive.
        assert_eq!(unescape(r#""C:\data""#), r"C:\data");
        // ...but `\t` is a real escape and does become a tab.
        assert_eq!(unescape(r#""a\tb""#), "a\tb");
    }

    #[test]
    fn booleans_parse() {
        assert!(matches!(expr_of("true").kind, ExprKind::Bool(true)));
        assert!(matches!(expr_of("false").kind, ExprKind::Bool(false)));
    }

    // --- errors and recovery -----------------------------------------------

    #[test]
    fn a_file_without_a_project_is_reported() {
        let (project, text, codes) = parse_err("grid g { }");
        assert!(project.is_none());
        assert!(codes.contains(&"E0100".to_string()));
        assert!(text.contains("must start here with `project`"), "{text}");
    }

    #[test]
    fn an_empty_file_is_reported() {
        let (_, text, codes) = parse_err("");
        assert!(codes.contains(&"E0100".to_string()));
        assert!(text.contains("declares no project"), "{text}");
    }

    #[test]
    fn a_missing_semicolon_is_reported_with_position() {
        let (_, text, codes) = parse_err("project p { dimensions: 2 }");
        assert!(codes.contains(&"E0101".to_string()));
        assert!(text.contains("expected `;`"), "{text}");
    }

    /// One run must report every problem, not stop at the first.
    #[test]
    fn recovery_reports_multiple_errors_in_one_pass() {
        let (project, text, _) = parse_err(
            "project p {
               dimensions: ;
               grid g { size: [4, 4]; }
               precision: ;
               fidelity: interactive;
             }",
        );
        let errors = text.matches("error").count();
        assert!(errors >= 2, "expected several errors, got:\n{text}");
        // And the good declarations between them still made it into the tree.
        let project = project.expect("recovery should still yield a project");
        assert_eq!(project.grids().count(), 1);
        assert!(project.setting("fidelity").is_some(), "parsing continued past the errors");
    }

    #[test]
    fn a_broken_setting_inside_a_block_does_not_end_the_block() {
        let (project, _, _) = parse_err("project p { grid g { size: ; extent: [1 meter, 1 meter]; } }");
        let project = project.unwrap();
        let grid = project.grids().next().expect("the grid should still be declared");
        assert!(grid.setting("extent").is_some(), "the later setting should still parse");
    }

    #[test]
    fn a_reserved_word_used_as_a_name_says_so() {
        let (_, text, codes) = parse_err("project p { field solve on chamber; }");
        assert!(codes.contains(&"E0102".to_string()), "{text}");
        assert!(text.contains("reserved word"), "{text}");
    }

    #[test]
    fn a_fractional_exponent_is_rejected_with_an_explanation() {
        let (_, text, codes) = parse_err("project p { x: meter^0.5; }");
        assert!(codes.contains(&"E0108".to_string()));
        assert!(text.contains("whole number"), "{text}");
        assert!(text.contains("integer exponents"), "{text}");
    }

    #[test]
    fn an_oversized_exponent_is_rejected() {
        let (_, _, codes) = parse_err("project p { x: meter^99; }");
        assert!(codes.contains(&"E0109".to_string()));
    }

    #[test]
    fn a_missing_with_in_solve_is_reported() {
        let (_, text, codes) = parse_err("project p { solve heat(t) crank_nicolson(); }");
        assert!(codes.contains(&"E0104".to_string()));
        assert!(text.contains("expected `with`"), "{text}");
    }

    #[test]
    fn a_missing_arrow_in_couple_is_reported() {
        let (_, text, codes) = parse_err("project p { couple a.b c.d; }");
        assert!(codes.contains(&"E0105".to_string()));
        assert!(text.contains("expected `->`"), "{text}");
    }

    #[test]
    fn something_that_is_not_a_declaration_at_all_is_reported() {
        let (_, text, codes) = parse_err("project p { 42 }");
        assert!(codes.contains(&"E0103".to_string()), "{text}");
        assert!(text.contains("not a declaration"), "{text}");
        assert!(text.contains("`name: value;` settings"), "{text}");
    }

    #[test]
    fn an_unclosed_brace_is_reported_rather_than_looping() {
        let (_, text, _) = parse_err("project p { grid g { size: [1,1];");
        assert!(text.contains("expected `}`"), "{text}");
    }

    // --- the specification's own models ------------------------------------

    /// Spec §25.1, verbatim. If the specification's flagship example does not parse,
    /// the grammar is wrong.
    #[test]
    fn the_spec_hot_reaction_project_parses() {
        let project = parse_ok(
            r#"
project hot_reaction {
  dimensions: 2;
  precision: mixed;
  fidelity: engineering_2d;

  grid chamber { size: [512, 256]; extent: [2 meter, 1 meter]; }

  field temperature on chamber = 298 kelvin;
  species A on chamber = left_half(1 mole / meter^2);
  species B on chamber = right_half(1 mole / meter^2);

  reaction A_plus_B {
    reactants: A + B;
    products: C;
    rate: arrhenius(A0=2.0e5 / second, Ea=35 kilojoule/mole)
          * A.concentration * B.concentration;
    enthalpy: -25 kilojoule / mole;
  }

  solve diffusion(A, B, C) with crank_nicolson(dt=0.002 second);
  solve heat(temperature) with crank_nicolson(dt=0.01 second);
  couple A_plus_B.heat_release -> temperature.source conserve energy;

  observe total_species every 0.1 second;
  observe total_energy every 0.1 second;
  visualize temperature as heatmap;
  visualize [A, B, C] as rgb_mix;
}
"#,
        );
        assert_eq!(project.name.text, "hot_reaction");
        assert_eq!(project.grids().count(), 1);
        assert_eq!(project.fields().count(), 1);
        assert_eq!(project.species().count(), 2);
        assert_eq!(project.reactions().count(), 1);
        assert_eq!(project.solves().count(), 2);
        assert_eq!(project.couples().count(), 1);
        assert_eq!(project.observes().count(), 2);
        assert_eq!(project.visualizes().count(), 2);
    }

    /// Spec §25.2, verbatim.
    #[test]
    fn the_spec_double_slit_project_parses() {
        let project = parse_ok(
            r#"
project double_slit {
  domain quantum2d q {
    grid: [768, 384];
    extent: [12 nanometer, 6 nanometer];
    mass: electron_mass;
    boundary: absorbing(width=0.8 nanometer);
    integrator: split_step_fourier(dt=0.002 femtosecond);
  }

  potential barrier {
    shape: vertical_wall(x=0, thickness=0.15 nanometer);
    slits: [(-1.0, 0.35), (1.0, 0.35)] nanometer;
    height: 20 electronvolt;
  }

  wavepacket initial {
    center: [-4 nanometer, 0];
    momentum: [6.5e-24 kilogram*meter/second, 0];
    sigma: 0.45 nanometer;
  }

  detector screen at x=4.5 nanometer;
  observe probability_norm every step;
  visualize probability_density;
  visualize phase;
}
"#,
        );
        assert_eq!(project.name.text, "double_slit");
        assert_eq!(project.domains().count(), 1);
        assert_eq!(project.observes().count(), 1);
    }

    /// Spec §12.2's species and reaction declarations.
    #[test]
    fn the_spec_species_and_reaction_declarations_parse() {
        let project = parse_ok(
            r#"
project chem {
  species H_plus {
    formula: H;
    charge: 1 elementary_charge;
    diffusion: 9.31e-9 meter^2 / second;
  }

  reaction acid_base {
    reactants: 1 H_plus + 1 OH_minus;
    products: 1 H2O;
    rate: k_forward * H_plus.concentration * OH_minus.concentration;
    enthalpy: -57.3 kilojoule / mole;
  }
}
"#,
        );
        let species = project.species().next().unwrap();
        assert_eq!(species.settings.len(), 3);
        assert_eq!(project.reactions().next().unwrap().settings.len(), 4);
    }

    #[test]
    fn comments_are_ignored_everywhere() {
        let project = parse_ok(
            "// leading
             project p { // trailing
               /* block */ dimensions: 2;
             }",
        );
        assert_eq!(project.settings().count(), 1);
    }

    // --- laws (spec §8.3) ---------------------------------------------------

    /// Spec §8.3's two examples, character for character, inside a project.
    #[test]
    fn the_spec_expression_examples_parse_verbatim() {
        let project = parse_ok(
            r#"project p {
force spring(a: particle, b: particle) -> vec2<newton> {
    let dx = minimum_image(b.position - a.position);
    let extension = length(dx) - rest_length;
    return stiffness * extension * normalize(dx)
         - damping * dot(b.velocity - a.velocity, normalize(dx)) * normalize(dx);
}

reaction neutralization {
    stoichiometry: H_plus + OH_minus -> H2O;
    rate: k * c(H_plus) * c(OH_minus);
    heat_release: 57.3 kilojoule / mole;
}
}"#,
        );
        let law = project.laws().next().expect("the force parses as a law");
        assert_eq!((law.kind.text.as_str(), law.name.text.as_str()), ("force", "spring"));
        assert_eq!(law.params.len(), 2);
        assert_eq!(law.params[1].name.text, "b");
        assert!(matches!(law.returns.as_ref().unwrap().kind, TypeKind::Vec2(_)));
        assert_eq!(law.body.len(), 3);
        assert!(matches!(law.body[2].kind, StmtKind::Return(_)));

        let reaction = project.reactions().next().unwrap();
        let stoichiometry = &reaction.setting("stoichiometry").unwrap().value;
        assert!(matches!(stoichiometry.kind, ExprKind::Yields(_, _)), "{stoichiometry:?}");
    }

    #[test]
    fn a_law_takes_params_defaults_and_a_scalar_return_type() {
        let project = parse_ok(
            "project p {
               potential lj(a: particle, b: particle) -> joule {
                 param epsilon: joule;
                 param sigma: meter = 1 angstrom;
                 let s6 = (sigma / distance(a, b))^6;
                 return 4 * epsilon * (s6^2 - s6);
               }
             }",
        );
        let law = project.laws().next().unwrap();
        let StmtKind::Param { name, default, .. } = &law.body[1].kind else { panic!() };
        assert_eq!(name.text, "sigma");
        assert!(default.is_some());
        assert!(matches!(law.returns.as_ref().unwrap().kind, TypeKind::Plain(_)));
    }

    #[test]
    fn a_law_without_a_return_type_or_parameters_parses() {
        let project = parse_ok("project p { force nothing() { return 0; } }");
        let law = project.laws().next().unwrap();
        assert!(law.params.is_empty() && law.returns.is_none());
    }

    /// A declaration and a law share `<kind> <name>`; the `(` is what tells them apart,
    /// so the quantum module's `potential barrier { … }` is untouched.
    #[test]
    fn a_block_declaration_is_not_mistaken_for_a_law() {
        let project = parse_ok("project p { potential barrier { height: 1 electronvolt; } }");
        assert_eq!(project.declarations().count(), 1);
        assert_eq!(project.laws().count(), 0);
    }

    #[test]
    fn top_level_let_parses() {
        let project = parse_ok("project p { let k = 1.4e11 meter^2 / (mole second); rate: k; }");
        let constant = project.lets().next().unwrap();
        assert_eq!(constant.name.text, "k");
    }

    #[test]
    fn comparison_and_logic_bind_below_arithmetic() {
        // `a + b < c && d` is `((a + b) < c) && d`.
        let ExprKind::Binary(BinaryOp::And, left, _) = expr_of("a + b < c && d").kind else { panic!() };
        let ExprKind::Binary(BinaryOp::Lt, sum, _) = &left.kind else { panic!("{left:?}") };
        assert!(matches!(sum.kind, ExprKind::Binary(BinaryOp::Add, _, _)));
        // `||` binds below `&&`.
        assert!(matches!(expr_of("a || b && c").kind, ExprKind::Binary(BinaryOp::Or, _, _)));
        assert!(matches!(expr_of("!a").kind, ExprKind::Unary(UnaryOp::Not, _)));
    }

    #[test]
    fn if_is_an_expression_with_else_if_chaining() {
        let ExprKind::If(_, _, otherwise) = expr_of("if r < 1 meter { 1 } else if r < 2 meter { 2 } else { 3 }").kind
        else {
            panic!()
        };
        assert!(matches!(otherwise.kind, ExprKind::If(_, _, _)));
    }

    #[test]
    fn law_syntax_errors_have_their_own_codes() {
        let (_, _, codes) = parse_err("project p { x: a < b < c; }");
        assert!(codes.contains(&"E0111".to_string()), "{codes:?}");
        let (_, _, codes) = parse_err("project p { x: if a { 1 }; }");
        assert!(codes.contains(&"E0113".to_string()), "{codes:?}");
        let (_, _, codes) = parse_err("project p { force f() { x = 1; return 1; } }");
        assert!(codes.contains(&"E0112".to_string()), "{codes:?}");
    }

    /// A bad statement does not take the rest of the law, or the project, with it.
    #[test]
    fn a_bad_statement_recovers_inside_the_law() {
        let (project, _, codes) =
            parse_err("project p { force f(a: particle) -> newton { x = 1; return 1 newton; } dimensions: 2; }");
        assert_eq!(codes, ["E0112"]);
        let project = project.unwrap();
        assert_eq!(project.laws().next().unwrap().body.len(), 1, "the `return` survives");
        assert!(project.setting("dimensions").is_some(), "the item after the law survives");
    }

    /// A statement that fails inside an `if` arm stops the parser between braces. The
    /// law must still close at its own `}`, not the arm's, or the rest of the arm is
    /// read as project items and reported as errors that are not there.
    #[test]
    fn a_failure_inside_an_if_arm_does_not_close_the_law() {
        let (project, _, codes) = parse_err(
            "project p {
               potential u(a: particle) -> joule {
                 return if a.mass > 1 kilogram { 1 joule + ; } else { 2 joule };
               }
               dimensions: 2;
             }",
        );
        assert_eq!(codes, ["E0110"], "one error, not cascades");
        let project = project.unwrap();
        assert!(project.laws().next().unwrap().recovered);
        assert!(project.setting("dimensions").is_some());
    }
}
