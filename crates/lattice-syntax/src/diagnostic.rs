//! Diagnostics: structured errors that point at source.
//!
//! Spec FR-002 does not merely require invalid models to be rejected — it requires
//! them to *"fail with source-positioned diagnostics"*, and §20.4 makes ten such
//! rejections an acceptance criterion. The difference matters: "dimensional mismatch"
//! sends a user hunting, while
//!
//! ```text
//! error[E0402]: dimensional mismatch in assignment
//!   --> hot_reaction.lattice:14:22
//!    |
//! 14 |   field temperature on chamber = 298 second;
//!    |                                  ^^^^^^^^^^ expected temperature (K), found time (s)
//!    |
//!    = help: `field temperature` was declared with dimension K
//! ```
//!
//! ends the search. Every diagnostic here carries a span, and the compiler threads
//! spans through every expression node so a dimensional error found three passes
//! later still points at the character that caused it.

use core::fmt;

use crate::source::{SourceFile, Span};

/// How serious a diagnostic is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Severity {
    /// Compilation cannot continue meaningfully.
    Error,
    /// Compilation continues, but the model report shows this.
    Warning,
    /// Supporting context attached to another diagnostic.
    Note,
}

impl Severity {
    /// The word used in rendered output.
    pub const fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A span with an explanation, underlined in the rendered output.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Label {
    /// What to underline.
    pub span: Span,
    /// What to say about it. May be empty for a bare underline.
    pub message: String,
}

impl Label {
    /// A labelled span.
    pub fn new(span: Span, message: impl Into<String>) -> Label {
        Label { span, message: message.into() }
    }
}

/// One structured problem with a model.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnostic {
    /// How serious it is.
    pub severity: Severity,
    /// A stable code, so a specific error can be looked up and tested for.
    pub code: Option<String>,
    /// The headline.
    pub message: String,
    /// The span most responsible, underlined with `^`.
    pub primary: Option<Label>,
    /// Related spans, underlined with `-`.
    pub secondary: Vec<Label>,
    /// Free-standing explanatory lines.
    pub notes: Vec<String>,
    /// A suggested fix.
    pub help: Option<String>,
}

impl Diagnostic {
    /// Start an error.
    pub fn error(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Error, message)
    }

    /// Start a warning.
    pub fn warning(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Warning, message)
    }

    /// Start a diagnostic of any severity.
    pub fn new(severity: Severity, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            primary: None,
            secondary: Vec::new(),
            notes: Vec::new(),
            help: None,
        }
    }

    /// Attach a stable diagnostic code.
    pub fn with_code(mut self, code: impl Into<String>) -> Diagnostic {
        self.code = Some(code.into());
        self
    }

    /// Set the primary span and its label.
    pub fn at(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.primary = Some(Label::new(span, message));
        self
    }

    /// Set the primary span with no label of its own.
    pub fn span(mut self, span: Span) -> Diagnostic {
        self.primary = Some(Label::new(span, String::new()));
        self
    }

    /// Add a related span.
    pub fn also(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.secondary.push(Label::new(span, message));
        self
    }

    /// Add an explanatory line.
    pub fn note(mut self, note: impl Into<String>) -> Diagnostic {
        self.notes.push(note.into());
        self
    }

    /// Suggest a fix.
    pub fn help(mut self, help: impl Into<String>) -> Diagnostic {
        self.help = Some(help.into());
        self
    }

    /// The primary span, or `Span::NONE`.
    pub fn primary_span(&self) -> Span {
        self.primary.as_ref().map_or(Span::NONE, |label| label.span)
    }

    /// Render against a source file.
    pub fn render(&self, file: &SourceFile) -> String {
        let mut out = String::new();

        match &self.code {
            Some(code) => out.push_str(&format!("{}[{}]: {}\n", self.severity, code, self.message)),
            None => out.push_str(&format!("{}: {}\n", self.severity, self.message)),
        }

        // Width of the gutter, sized to the largest line number shown.
        let gutter = self
            .primary
            .iter()
            .chain(&self.secondary)
            .map(|label| file.location(label.span.start).line)
            .max()
            .map_or(1, |line| line.to_string().len());

        if let Some(primary) = &self.primary {
            let location = file.location(primary.span.start);
            out.push_str(&format!(
                "{:>width$}--> {}:{}\n",
                "",
                file.name(),
                location,
                width = gutter
            ));
            out.push_str(&render_label(file, primary, '^', gutter));
        }

        for label in &self.secondary {
            let location = file.location(label.span.start);
            out.push_str(&format!(
                "{:>width$}--> {}:{}\n",
                "",
                file.name(),
                location,
                width = gutter
            ));
            out.push_str(&render_label(file, label, '-', gutter));
        }

        for note in &self.notes {
            out.push_str(&format!("{:>width$} = note: {note}\n", "", width = gutter));
        }
        if let Some(help) = &self.help {
            out.push_str(&format!("{:>width$} = help: {help}\n", "", width = gutter));
        }
        out
    }
}

/// Render one labelled span as a source excerpt with an underline.
fn render_label(file: &SourceFile, label: &Label, marker: char, gutter: usize) -> String {
    let location = file.location(label.span.start);
    let (line_text, caret_column) = file.rendered_line(location.line, label.span.start);
    let width = file.rendered_width(label.span);

    let mut out = String::new();
    out.push_str(&format!("{:>gutter$} |\n", ""));
    out.push_str(&format!("{:>gutter$} | {line_text}\n", location.line));
    out.push_str(&format!(
        "{:>gutter$} | {}{}",
        "",
        " ".repeat(caret_column),
        marker.to_string().repeat(width)
    ));
    if label.message.is_empty() {
        out.push('\n');
    } else {
        out.push_str(&format!(" {}\n", label.message));
    }

    // A span that runs past its first line has been clipped; say so rather than
    // letting the underline silently under-report what is wrong.
    let end_line = file.location(label.span.end.saturating_sub(1)).line;
    if end_line > location.line {
        out.push_str(&format!(
            "{:>gutter$} | ... continues to line {end_line}\n",
            ""
        ));
    }
    out
}

/// A collection of diagnostics, in the order they were reported.
#[derive(Clone, Default, Debug)]
pub struct Diagnostics {
    entries: Vec<Diagnostic>,
}

impl Diagnostics {
    /// An empty collection.
    pub fn new() -> Diagnostics {
        Diagnostics::default()
    }

    /// Record a diagnostic.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.entries.push(diagnostic);
    }

    /// All diagnostics.
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.entries.iter()
    }

    /// Number recorded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of errors.
    pub fn error_count(&self) -> usize {
        self.entries.iter().filter(|d| d.severity == Severity::Error).count()
    }

    /// Number of warnings.
    pub fn warning_count(&self) -> usize {
        self.entries.iter().filter(|d| d.severity == Severity::Warning).count()
    }

    /// True if any diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.error_count() > 0
    }

    /// The codes present, for tests that assert on a specific failure.
    pub fn codes(&self) -> Vec<&str> {
        self.entries.iter().filter_map(|d| d.code.as_deref()).collect()
    }

    /// Merge another collection in.
    pub fn extend(&mut self, other: Diagnostics) {
        self.entries.extend(other.entries);
    }

    /// Sort by source position, so a run of errors reads top to bottom.
    pub fn sort_by_position(&mut self) {
        self.entries.sort_by_key(|d| d.primary_span().start);
    }

    /// Render every diagnostic, followed by a summary line.
    pub fn render(&self, file: &SourceFile) -> String {
        let mut out = String::new();
        for diagnostic in &self.entries {
            out.push_str(&diagnostic.render(file));
            out.push('\n');
        }
        let (errors, warnings) = (self.error_count(), self.warning_count());
        if errors > 0 || warnings > 0 {
            out.push_str(&format!(
                "{errors} error{}, {warnings} warning{}\n",
                if errors == 1 { "" } else { "s" },
                if warnings == 1 { "" } else { "s" }
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> SourceFile {
        SourceFile::new(
            "hot_reaction.lattice",
            "project hot_reaction {\n  field temperature on chamber = 298 second;\n}\n",
        )
    }

    fn span_of(file: &SourceFile, needle: &str) -> Span {
        let start = file.text().find(needle).expect("needle should be present") as u32;
        Span::new(start, start + needle.len() as u32)
    }

    #[test]
    fn a_rendered_error_points_at_the_source() {
        let f = file();
        let diagnostic = Diagnostic::error("dimensional mismatch in assignment")
            .with_code("E0402")
            .at(span_of(&f, "298 second"), "expected temperature (K), found time (s)")
            .help("`field temperature` was declared with dimension K");

        let text = diagnostic.render(&f);
        assert!(text.starts_with("error[E0402]: dimensional mismatch in assignment\n"), "{text}");
        assert!(text.contains("--> hot_reaction.lattice:2:34"), "{text}");
        assert!(text.contains("field temperature on chamber = 298 second;"), "{text}");
        assert!(text.contains("^^^^^^^^^^ expected temperature (K), found time (s)"), "{text}");
        assert!(text.contains("= help:"), "{text}");
    }

    /// The caret must sit under the span, not near it. This is the whole value of a
    /// source-positioned diagnostic.
    #[test]
    fn the_caret_aligns_with_the_span() {
        let f = file();
        let span = span_of(&f, "298 second");
        let text = Diagnostic::error("nope").span(span).render(&f);

        let lines: Vec<&str> = text.lines().collect();
        let source_line = lines.iter().find(|l| l.contains("field temperature")).unwrap();
        let caret_line = lines.iter().find(|l| l.contains('^')).unwrap();

        // Strip the identical `N | ` gutter from both and compare offsets.
        let source_body = source_line.split_once("| ").unwrap().1;
        let caret_body = caret_line.split_once("| ").unwrap().1;
        let caret_start = caret_body.find('^').unwrap();
        assert_eq!(&source_body[caret_start..caret_start + 10], "298 second", "{text}");
        assert_eq!(caret_body.matches('^').count(), 10);
    }

    #[test]
    fn secondary_labels_use_a_different_marker() {
        let f = file();
        let text = Diagnostic::error("conflict")
            .at(span_of(&f, "298 second"), "here")
            .also(span_of(&f, "temperature"), "declared here")
            .render(&f);
        assert!(text.contains('^'), "{text}");
        assert!(text.contains("- declared here"), "{text}");
    }

    #[test]
    fn notes_and_help_render_after_the_excerpt() {
        let f = file();
        let text = Diagnostic::error("x")
            .span(span_of(&f, "298"))
            .note("first note")
            .note("second note")
            .help("try this")
            .render(&f);
        assert!(text.contains("= note: first note"), "{text}");
        assert!(text.contains("= note: second note"), "{text}");
        assert!(text.contains("= help: try this"), "{text}");
    }

    #[test]
    fn a_diagnostic_without_a_span_still_renders() {
        let f = file();
        let text = Diagnostic::error("no project declaration found").render(&f);
        assert_eq!(text, "error: no project declaration found\n");
    }

    #[test]
    fn a_multi_line_span_says_it_continues() {
        let f = SourceFile::new("m.lattice", "reaction r {\n  rate: a\n     * b;\n}\n");
        let start = f.text().find("a\n").unwrap() as u32;
        let end = f.text().find("b;").unwrap() as u32 + 1;
        let text = Diagnostic::error("bad rate").span(Span::new(start, end)).render(&f);
        assert!(text.contains("continues to line 3"), "{text}");
    }

    #[test]
    fn the_gutter_widens_for_large_line_numbers() {
        let mut source = String::new();
        for _ in 0..120 {
            source.push_str("// filler\n");
        }
        source.push_str("field x on g;\n");
        let f = SourceFile::new("big.lattice", source);
        let span = span_of(&f, "field x");
        let text = Diagnostic::error("x").span(span).render(&f);
        // Line 121 needs a three-wide gutter, and the `-->` must line up with it.
        assert!(text.contains("121 | field x on g;"), "{text}");
        assert!(text.contains("   --> big.lattice:121:1"), "{text}");
    }

    #[test]
    fn collections_count_and_summarize() {
        let f = file();
        let mut diagnostics = Diagnostics::new();
        assert!(diagnostics.is_empty());
        assert!(!diagnostics.has_errors());

        diagnostics.push(Diagnostic::error("one").with_code("E0001").span(span_of(&f, "298")));
        diagnostics.push(Diagnostic::warning("two").with_code("W0001").span(span_of(&f, "field")));

        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics.error_count(), 1);
        assert_eq!(diagnostics.warning_count(), 1);
        assert!(diagnostics.has_errors());
        assert_eq!(diagnostics.codes(), ["E0001", "W0001"]);

        let text = diagnostics.render(&f);
        assert!(text.contains("1 error, 1 warning"), "{text}");
    }

    #[test]
    fn sorting_orders_diagnostics_by_position() {
        let f = file();
        let mut diagnostics = Diagnostics::new();
        diagnostics.push(Diagnostic::error("later").span(span_of(&f, "298")));
        diagnostics.push(Diagnostic::error("earlier").span(span_of(&f, "project")));
        diagnostics.sort_by_position();
        assert_eq!(diagnostics.iter().next().unwrap().message, "earlier");
    }

    /// A caret must land correctly after multi-byte characters, or diagnostics become
    /// actively misleading in exactly the models that use proper unit symbols.
    #[test]
    fn carets_survive_multibyte_source() {
        let f = SourceFile::new("u.lattice", "  diffusion: 1 µm²/second;\n");
        let span = span_of(&f, "second");
        let text = Diagnostic::error("x").at(span, "here").render(&f);

        let caret_line = text.lines().find(|l| l.contains('^')).unwrap();
        let source_line = text.lines().find(|l| l.contains("diffusion")).unwrap();
        let caret_body = caret_line.split_once("| ").unwrap().1;
        let source_body: Vec<char> = source_line.split_once("| ").unwrap().1.chars().collect();
        let caret_start = caret_body.find('^').unwrap();
        let under: String = source_body[caret_start..caret_start + 6].iter().collect();
        assert_eq!(under, "second", "{text}");
    }
}
