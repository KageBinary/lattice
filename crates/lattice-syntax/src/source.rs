//! Source files, byte spans, and the mapping from offsets to line/column.
//!
//! Spec FR-002 requires invalid expressions to *"fail with source-positioned
//! diagnostics"*, and §16.2 asks for *"source-positioned diagnostics and canonical
//! formatting"*. Everything downstream — lexer, parser, compiler — carries a [`Span`]
//! on every node so that a dimensional error found three passes later can still point
//! at the character that caused it.
//!
//! # Byte offsets, character columns
//!
//! Spans are byte offsets, because that is what slicing a `&str` needs. Columns are
//! *character* counts, because that is what lines up a caret under the right symbol.
//! A model that writes `9.31e-9 meter²/second` must not have its caret drift by the
//! two extra bytes of `²`.

use core::fmt;
use core::ops::Range;

/// A half-open byte range within a [`SourceFile`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Span {
    /// First byte.
    pub start: u32,
    /// One past the last byte.
    pub end: u32,
}

impl Span {
    /// An empty span at the start of the file, for synthesized nodes.
    pub const NONE: Span = Span { start: 0, end: 0 };

    /// A span covering `[start, end)`.
    pub const fn new(start: u32, end: u32) -> Span {
        Span { start, end }
    }

    /// A span covering a byte range.
    pub fn from_range(range: Range<usize>) -> Span {
        Span { start: range.start as u32, end: range.end as u32 }
    }

    /// The smallest span containing both.
    pub fn merge(self, other: Span) -> Span {
        if self == Span::NONE {
            return other;
        }
        if other == Span::NONE {
            return self;
        }
        Span { start: self.start.min(other.start), end: self.end.max(other.end) }
    }

    /// Byte length.
    pub const fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    /// True when the span covers nothing.
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// As a `Range<usize>`, for slicing.
    pub const fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A one-based line and column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Location {
    /// One-based line number.
    pub line: u32,
    /// One-based column, counted in characters.
    pub column: u32,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A named piece of source text with a precomputed line index.
#[derive(Clone, Debug)]
pub struct SourceFile {
    name: String,
    text: String,
    /// Byte offset of the first character of each line.
    line_starts: Vec<u32>,
}

/// Spaces a tab expands to when rendering a diagnostic.
///
/// Rendering must expand tabs *and* shift the caret by the same amount, or the caret
/// lands in the wrong place in exactly the files where indentation is inconsistent.
const TAB_WIDTH: usize = 4;

impl SourceFile {
    /// Index a source file.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> SourceFile {
        let text = text.into();
        let mut line_starts = vec![0u32];
        for (offset, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(offset as u32 + 1);
            }
        }
        SourceFile { name: name.into(), text, line_starts }
    }

    /// The file's name, as it appears in diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The whole text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Number of lines.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// The text a span covers.
    pub fn slice(&self, span: Span) -> &str {
        let end = (span.end as usize).min(self.text.len());
        let start = (span.start as usize).min(end);
        &self.text[start..end]
    }

    /// The line and character column of a byte offset.
    pub fn location(&self, offset: u32) -> Location {
        let offset = offset.min(self.text.len() as u32);
        // The line whose start is the greatest offset <= `offset`.
        let line_index = match self.line_starts.binary_search(&offset) {
            Ok(index) => index,
            Err(index) => index - 1,
        };
        let line_start = self.line_starts[line_index] as usize;
        let column = self.text[line_start..offset as usize].chars().count() + 1;
        Location { line: line_index as u32 + 1, column: column as u32 }
    }

    /// The text of a one-based line, without its terminator.
    pub fn line_text(&self, line: u32) -> &str {
        let index = (line.saturating_sub(1)) as usize;
        let Some(&start) = self.line_starts.get(index) else { return "" };
        let end = self
            .line_starts
            .get(index + 1)
            .map_or(self.text.len(), |&next| next as usize);
        self.text[start as usize..end].trim_end_matches(['\n', '\r'])
    }

    /// Render a line with tabs expanded, and the column a byte offset maps to in that
    /// rendering.
    pub(crate) fn rendered_line(&self, line: u32, offset: u32) -> (String, usize) {
        let raw = self.line_text(line);
        let line_start = self.line_starts[(line.saturating_sub(1)) as usize];
        let within = (offset.saturating_sub(line_start)) as usize;

        let mut rendered = String::with_capacity(raw.len());
        let mut caret_column = None;
        let mut byte = 0usize;
        for c in raw.chars() {
            if byte == within {
                caret_column = Some(rendered.chars().count());
            }
            if c == '\t' {
                let pad = TAB_WIDTH - (rendered.chars().count() % TAB_WIDTH);
                rendered.push_str(&" ".repeat(pad));
            } else {
                rendered.push(c);
            }
            byte += c.len_utf8();
        }
        // An offset at or past the end of the line puts the caret just after it,
        // which is what an "unexpected end of input" diagnostic wants.
        let caret = caret_column.unwrap_or_else(|| rendered.chars().count());
        (rendered, caret)
    }

    /// How many characters wide a span is once tabs are expanded, clamped to the line
    /// it starts on.
    pub(crate) fn rendered_width(&self, span: Span) -> usize {
        let start_line = self.location(span.start).line;
        let line_start = self.line_starts[(start_line.saturating_sub(1)) as usize];
        let line_end = line_start as usize + self.line_text(start_line).len();
        let end = (span.end as usize).min(line_end).max(span.start as usize);

        let mut width = 0usize;
        let mut column = self.rendered_line(start_line, span.start).1;
        for c in self.text[span.start as usize..end].chars() {
            if c == '\t' {
                let pad = TAB_WIDTH - (column % TAB_WIDTH);
                width += pad;
                column += pad;
            } else {
                width += 1;
                column += 1;
            }
        }
        width.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> SourceFile {
        SourceFile::new("test.lattice", "project p {\n  grid g { size: 4; }\n}\n")
    }

    #[test]
    fn line_starts_are_indexed() {
        let f = file();
        assert_eq!(f.line_count(), 4, "three lines plus the empty one after the final newline");
        assert_eq!(f.line_text(1), "project p {");
        assert_eq!(f.line_text(2), "  grid g { size: 4; }");
        assert_eq!(f.line_text(3), "}");
        assert_eq!(f.line_text(99), "", "out of range lines are empty, not a panic");
    }

    #[test]
    fn offsets_map_to_one_based_locations() {
        let f = file();
        assert_eq!(f.location(0), Location { line: 1, column: 1 });
        assert_eq!(f.location(8), Location { line: 1, column: 9 });
        // First character of line 2.
        assert_eq!(f.location(12), Location { line: 2, column: 1 });
        // `grid` starts two spaces in.
        assert_eq!(f.location(14), Location { line: 2, column: 3 });
    }

    #[test]
    fn an_offset_past_the_end_clamps() {
        let f = file();
        let location = f.location(9_999);
        assert!(location.line <= f.line_count() as u32);
    }

    /// Columns count characters, not bytes, so a caret lands under the right symbol
    /// even after multi-byte text.
    #[test]
    fn columns_count_characters_not_bytes() {
        let f = SourceFile::new("u.lattice", "diffusion: 1 µm²/second;");
        // `µ` is two bytes and `²` is three; the column must not drift.
        let micro = f.text().find('µ').unwrap() as u32;
        assert_eq!(f.location(micro).column, 14);
        let squared = f.text().find('²').unwrap() as u32;
        assert_eq!(f.location(squared).column, 16, "µ and m each advance one column");
    }

    #[test]
    fn spans_slice_and_merge() {
        let f = file();
        let span = Span::new(0, 7);
        assert_eq!(f.slice(span), "project");
        assert_eq!(span.len(), 7);
        assert!(!span.is_empty());

        let other = Span::new(8, 9);
        assert_eq!(span.merge(other), Span::new(0, 9));
        assert_eq!(span.merge(Span::NONE), span, "merging with NONE is identity");
        assert_eq!(Span::NONE.merge(other), other);
    }

    #[test]
    fn slicing_past_the_end_is_clamped_not_a_panic() {
        let f = file();
        assert_eq!(f.slice(Span::new(0, 9_999)), f.text());
        assert_eq!(f.slice(Span::new(9_999, 10_000)), "");
    }

    /// Tabs must expand consistently in the rendered line and in the caret offset, or
    /// the caret drifts in exactly the files with mixed indentation.
    #[test]
    fn tabs_expand_consistently_for_line_and_caret() {
        let f = SourceFile::new("t.lattice", "\tfield x on g;\n");
        let x_offset = f.text().find('x').unwrap() as u32;
        let (rendered, caret) = f.rendered_line(1, x_offset);
        assert_eq!(rendered, "    field x on g;");
        assert_eq!(rendered.chars().nth(caret), Some('x'), "caret should land on `x`");
    }

    #[test]
    fn rendered_width_covers_the_span_and_never_reaches_zero() {
        let f = file();
        assert_eq!(f.rendered_width(Span::new(0, 7)), 7);
        // A zero-length span still needs one column so the caret is visible.
        assert_eq!(f.rendered_width(Span::new(3, 3)), 1);
    }

    #[test]
    fn a_span_crossing_lines_is_clamped_to_its_first_line() {
        let f = file();
        let whole = Span::new(0, f.text().len() as u32);
        // Line one is 11 characters; the underline must not run past it.
        assert_eq!(f.rendered_width(whole), 11);
    }

    #[test]
    fn an_empty_file_is_handled() {
        let f = SourceFile::new("empty.lattice", "");
        assert_eq!(f.line_count(), 1);
        assert_eq!(f.location(0), Location { line: 1, column: 1 });
        assert_eq!(f.line_text(1), "");
    }
}
