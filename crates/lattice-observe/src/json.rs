//! A minimal, dependency-free JSON writer.
//!
//! Spec §18.1 requires every run to produce a self-describing artifact whose
//! *"metadata remains readable without loading all arrays"*, and FR-014 requires
//! *"machine-readable diagnostics"*. That needs a serializer.
//!
//! # Why not serde
//!
//! Two reasons, both about the artifact rather than about convenience.
//!
//! **Byte-for-byte reproducibility.** Objects here are insertion-ordered `Vec`s, not
//! hash maps, so the same run produces the same bytes and artifacts can be diffed and
//! hashed. FR-011 asks a recorded run to *"reproduce hashes within documented
//! floating-point constraints"*; a serializer that reorders keys makes that
//! impossible.
//!
//! **Non-finite values.** JSON has no `NaN` or `Infinity`. Most serializers either
//! refuse to write them or silently emit `null`. For a diagnostics artifact that is
//! backwards: a `NaN` is the single most interesting thing a run can produce
//! (NFR-007), and it must survive the trip to disk. [`Json::Number`] writes
//! non-finite values as the strings `"NaN"`, `"Infinity"` and `"-Infinity"`, which is
//! valid JSON, lossless, and obvious to a reader.

use core::fmt::{self, Write as _};

/// A JSON value.
#[derive(Clone, PartialEq, Debug)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A floating-point number. Non-finite values serialize as strings.
    Number(f64),
    /// An integer, written without a decimal point.
    Int(i64),
    /// A string, escaped on output.
    String(String),
    /// An ordered list.
    Array(Vec<Json>),
    /// An ordered set of key/value pairs. Order is preserved exactly as inserted.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// An empty object, ready for [`Json::set`].
    pub fn object() -> Json {
        Json::Object(Vec::new())
    }

    /// An empty array.
    pub fn array() -> Json {
        Json::Array(Vec::new())
    }

    /// Insert or replace a key. Returns `self` for chaining.
    ///
    /// # Panics
    ///
    /// If called on a value that is not an object.
    pub fn set(mut self, key: impl Into<String>, value: impl Into<Json>) -> Json {
        self.insert(key, value);
        self
    }

    /// Insert or replace a key in place.
    ///
    /// # Panics
    ///
    /// If called on a value that is not an object.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Json>) {
        let Json::Object(entries) = self else {
            panic!("insert() called on a JSON {}, not an object", self.kind());
        };
        let key = key.into();
        let value = value.into();
        match entries.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => entries.push((key, value)),
        }
    }

    /// Append to an array.
    ///
    /// # Panics
    ///
    /// If called on a value that is not an array.
    pub fn push(&mut self, value: impl Into<Json>) {
        let Json::Array(items) = self else {
            panic!("push() called on a JSON {}, not an array", self.kind());
        };
        items.push(value.into());
    }

    /// Look up a key in an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The type name, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "bool",
            Json::Number(_) | Json::Int(_) => "number",
            Json::String(_) => "string",
            Json::Array(_) => "array",
            Json::Object(_) => "object",
        }
    }

    /// Serialize with two-space indentation.
    pub fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0, true).expect("writing to a String cannot fail");
        out
    }

    /// Serialize with no whitespace.
    pub fn to_compact_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0, false).expect("writing to a String cannot fail");
        out
    }

    fn write(&self, out: &mut String, depth: usize, pretty: bool) -> fmt::Result {
        match self {
            Json::Null => out.write_str("null"),
            Json::Bool(true) => out.write_str("true"),
            Json::Bool(false) => out.write_str("false"),
            Json::Int(v) => write!(out, "{v}"),
            Json::Number(v) => write_number(out, *v),
            Json::String(s) => write_string(out, s),

            Json::Array(items) if items.is_empty() => out.write_str("[]"),
            Json::Array(items) => {
                out.write_char('[')?;
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.write_char(',')?;
                    }
                    newline_indent(out, depth + 1, pretty)?;
                    item.write(out, depth + 1, pretty)?;
                }
                newline_indent(out, depth, pretty)?;
                out.write_char(']')
            }

            Json::Object(entries) if entries.is_empty() => out.write_str("{}"),
            Json::Object(entries) => {
                out.write_char('{')?;
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index > 0 {
                        out.write_char(',')?;
                    }
                    newline_indent(out, depth + 1, pretty)?;
                    write_string(out, key)?;
                    out.write_char(':')?;
                    if pretty {
                        out.write_char(' ')?;
                    }
                    value.write(out, depth + 1, pretty)?;
                }
                newline_indent(out, depth, pretty)?;
                out.write_char('}')
            }
        }
    }
}

fn newline_indent(out: &mut String, depth: usize, pretty: bool) -> fmt::Result {
    if pretty {
        out.write_char('\n')?;
        for _ in 0..depth {
            out.write_str("  ")?;
        }
    }
    Ok(())
}

/// Write a number, preserving non-finite values as strings.
fn write_number(out: &mut String, value: f64) -> fmt::Result {
    if value.is_nan() {
        return out.write_str("\"NaN\"");
    }
    if value.is_infinite() {
        return out.write_str(if value > 0.0 { "\"Infinity\"" } else { "\"-Infinity\"" });
    }
    // `{:?}` on f64 emits the shortest representation that round-trips exactly,
    // which is what a reproducible artifact needs. It also always includes a
    // decimal point or exponent, so the value never reads back as an integer.
    write!(out, "{value:?}")
}

fn write_string(out: &mut String, s: &str) -> fmt::Result {
    out.write_char('"')?;
    for c in s.chars() {
        match c {
            '"' => out.write_str("\\\"")?,
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            '\u{8}' => out.write_str("\\b")?,
            '\u{c}' => out.write_str("\\f")?,
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32)?,
            c => out.write_char(c)?,
        }
    }
    out.write_char('"')
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_pretty_string())
    }
}

impl From<f64> for Json {
    fn from(v: f64) -> Json {
        Json::Number(v)
    }
}

impl From<f32> for Json {
    fn from(v: f32) -> Json {
        Json::Number(f64::from(v))
    }
}

impl From<i64> for Json {
    fn from(v: i64) -> Json {
        Json::Int(v)
    }
}

impl From<i32> for Json {
    fn from(v: i32) -> Json {
        Json::Int(i64::from(v))
    }
}

impl From<u64> for Json {
    fn from(v: u64) -> Json {
        // Values beyond i64 cannot occur for step counts or byte totals, but rather
        // than silently wrap, they widen to a float.
        i64::try_from(v).map_or(Json::Number(v as f64), Json::Int)
    }
}

impl From<usize> for Json {
    fn from(v: usize) -> Json {
        Json::from(v as u64)
    }
}

impl From<bool> for Json {
    fn from(v: bool) -> Json {
        Json::Bool(v)
    }
}

impl From<&str> for Json {
    fn from(v: &str) -> Json {
        Json::String(v.to_string())
    }
}

impl From<String> for Json {
    fn from(v: String) -> Json {
        Json::String(v)
    }
}

impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Json {
        Json::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Json {
        v.map_or(Json::Null, Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_serialize_as_expected() {
        assert_eq!(Json::Null.to_compact_string(), "null");
        assert_eq!(Json::Bool(true).to_compact_string(), "true");
        assert_eq!(Json::Int(-42).to_compact_string(), "-42");
        assert_eq!(Json::Number(1.5).to_compact_string(), "1.5");
        assert_eq!(Json::from("hi").to_compact_string(), "\"hi\"");
    }

    /// Numbers must round-trip exactly, or a replayed run will not match its artifact.
    #[test]
    fn floats_round_trip_exactly() {
        for value in [0.1, 1e-300, 1.7976931348623157e308, -2.5e-9, 1.0 / 3.0] {
            let text = Json::Number(value).to_compact_string();
            let parsed: f64 = text.parse().unwrap_or_else(|_| panic!("cannot reparse {text}"));
            assert_eq!(parsed, value, "{text}");
        }
    }

    /// A NaN in a diagnostics artifact is the finding, not an inconvenience. It must
    /// survive serialization rather than becoming `null`.
    #[test]
    fn non_finite_values_survive_as_strings() {
        assert_eq!(Json::Number(f64::NAN).to_compact_string(), "\"NaN\"");
        assert_eq!(Json::Number(f64::INFINITY).to_compact_string(), "\"Infinity\"");
        assert_eq!(Json::Number(f64::NEG_INFINITY).to_compact_string(), "\"-Infinity\"");
    }

    #[test]
    fn strings_are_escaped() {
        // Quote, backslash and the named control escapes, plus a bare control
        // character with no short form, which must become a \u sequence.
        let s = Json::from("a\"b\\c\nd\te\u{1}");
        assert_eq!(s.to_compact_string(), "\"a\\\"b\\\\c\\nd\\te\\u0001\"");
    }

    #[test]
    fn unicode_passes_through_unescaped() {
        assert_eq!(Json::from("µm · Å").to_compact_string(), "\"µm · Å\"");
    }

    /// Key order is preserved so artifacts are diffable and hashable.
    #[test]
    fn object_key_order_is_insertion_order() {
        let doc = Json::object().set("zebra", 1).set("apple", 2).set("mango", 3);
        assert_eq!(doc.to_compact_string(), r#"{"zebra":1,"apple":2,"mango":3}"#);
    }

    #[test]
    fn setting_an_existing_key_replaces_it_in_place() {
        let doc = Json::object().set("a", 1).set("b", 2).set("a", 99);
        assert_eq!(doc.to_compact_string(), r#"{"a":99,"b":2}"#);
    }

    #[test]
    fn nested_structures_serialize() {
        let doc = Json::object()
            .set("name", "run")
            .set("steps", 100u64)
            .set("metrics", Json::object().set("energy", 1.5).set("drift", -2e-9))
            .set("tags", vec!["a", "b"]);
        let compact = doc.to_compact_string();
        assert_eq!(
            compact,
            r#"{"name":"run","steps":100,"metrics":{"energy":1.5,"drift":-2e-9},"tags":["a","b"]}"#
        );
    }

    #[test]
    fn pretty_printing_indents_consistently() {
        let doc = Json::object().set("a", Json::object().set("b", 1));
        assert_eq!(doc.to_pretty_string(), "{\n  \"a\": {\n    \"b\": 1\n  }\n}");
    }

    #[test]
    fn empty_containers_stay_on_one_line() {
        assert_eq!(Json::object().to_pretty_string(), "{}");
        assert_eq!(Json::array().to_pretty_string(), "[]");
    }

    #[test]
    fn lookup_finds_nested_values() {
        let doc = Json::object().set("outer", Json::object().set("inner", 7));
        assert_eq!(doc.get("outer").and_then(|o| o.get("inner")), Some(&Json::Int(7)));
        assert_eq!(doc.get("missing"), None);
        assert_eq!(Json::Int(1).get("anything"), None);
    }

    #[test]
    fn arrays_can_be_built_incrementally() {
        let mut a = Json::array();
        a.push(1);
        a.push("two");
        a.push(3.5);
        assert_eq!(a.to_compact_string(), r#"[1,"two",3.5]"#);
    }

    #[test]
    fn options_become_null_or_the_value() {
        assert_eq!(Json::from(None::<f64>), Json::Null);
        assert_eq!(Json::from(Some(2.0)), Json::Number(2.0));
    }

    #[test]
    #[should_panic(expected = "not an object")]
    fn inserting_into_a_non_object_is_a_bug() {
        Json::array().set("k", 1);
    }

    /// The same content must always produce the same bytes.
    #[test]
    fn serialization_is_deterministic() {
        let build = || {
            Json::object()
                .set("b", 2)
                .set("a", Json::array())
                .set("c", vec![1.0, 2.0, 3.0])
        };
        assert_eq!(build().to_pretty_string(), build().to_pretty_string());
    }
}
