//! A small argument parser.
//!
//! The CLI takes a subcommand, some positional arguments, and a handful of long
//! flags. `clap` would do this well, but the whole workspace is deliberately
//! dependency-free at this milestone (spec §24.1 limits dependencies to those that do
//! *not* define core semantics, and a first build that needs no network is worth
//! more right now than nicer help text). Roughly a hundred lines covers what the
//! commands actually use.

use std::collections::BTreeMap;
use std::str::FromStr;

/// Parsed command line.
#[derive(Clone, Debug, Default)]
pub struct Args {
    /// The subcommand, if one was given.
    pub command: Option<String>,
    /// Positional arguments after the subcommand.
    pub positional: Vec<String>,
    /// Long flags. A flag with no value maps to `None`.
    flags: BTreeMap<String, Option<String>>,
}

impl Args {
    /// Parse an iterator of arguments, excluding the program name.
    ///
    /// Accepts `--flag`, `--key value`, `--key=value`, and the short aliases `-h`
    /// and `-V`. A `--` terminator sends everything after it to positionals.
    pub fn parse(argv: impl IntoIterator<Item = String>) -> Args {
        let mut args = Args::default();
        let mut iter = argv.into_iter().peekable();
        let mut only_positional = false;

        while let Some(token) = iter.next() {
            if only_positional {
                args.push_positional(token);
                continue;
            }
            if token == "--" {
                only_positional = true;
                continue;
            }

            if let Some(rest) = token.strip_prefix("--") {
                match rest.split_once('=') {
                    Some((key, value)) => {
                        args.flags.insert(key.to_string(), Some(value.to_string()));
                    }
                    None => {
                        // A following token is this flag's value unless it is itself
                        // a flag. `--json --quiet` therefore reads as two flags, not
                        // as `--json="--quiet"`.
                        let takes_value =
                            iter.peek().is_some_and(|next| !next.starts_with('-') || next == "-");
                        let value = takes_value.then(|| iter.next().expect("peeked"));
                        args.flags.insert(rest.to_string(), value);
                    }
                }
                continue;
            }

            if let Some(short) = token.strip_prefix('-')
                && token.len() == 2
            {
                let expanded = match short {
                    "h" => "help",
                    "V" => "version",
                    "q" => "quiet",
                    other => other,
                };
                args.flags.insert(expanded.to_string(), None);
                continue;
            }

            args.push_positional(token);
        }
        args
    }

    /// Read the process arguments.
    pub fn from_env() -> Args {
        Args::parse(std::env::args().skip(1))
    }

    fn push_positional(&mut self, token: String) {
        if self.command.is_none() {
            self.command = Some(token);
        } else {
            self.positional.push(token);
        }
    }

    /// Whether a flag was present, with or without a value.
    pub fn has(&self, name: &str) -> bool {
        self.flags.contains_key(name)
    }

    /// The value attached to a flag.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.flags.get(name).and_then(|v| v.as_deref())
    }

    /// Parse a flag's value, erroring if it is present but malformed.
    ///
    /// A typo'd number should stop the run rather than silently fall back to a
    /// default and produce a benchmark nobody can reproduce.
    pub fn parsed<T: FromStr>(&self, name: &str) -> Result<Option<T>, String>
    where
        T::Err: std::fmt::Display,
    {
        match self.value(name) {
            None => Ok(None),
            Some(raw) => raw
                .parse::<T>()
                .map(Some)
                .map_err(|e| format!("--{name}: `{raw}` is not valid ({e})")),
        }
    }

    /// Parse a flag's value, or fall back to a default.
    pub fn parsed_or<T: FromStr>(&self, name: &str, default: T) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        Ok(self.parsed(name)?.unwrap_or(default))
    }

    /// Flag names that were given, for detecting typos.
    pub fn flag_names(&self) -> impl Iterator<Item = &str> {
        self.flags.keys().map(String::as_str)
    }

    /// Any flag not in `known`, so an unrecognized option is reported rather than
    /// silently ignored.
    pub fn unknown_flags(&self, known: &[&str]) -> Vec<&str> {
        self.flag_names().filter(|name| !known.contains(name)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Args {
        Args::parse(line.split_whitespace().map(String::from))
    }

    #[test]
    fn command_and_positionals_are_separated() {
        let a = parse("demo heat-gaussian extra");
        assert_eq!(a.command.as_deref(), Some("demo"));
        assert_eq!(a.positional, ["heat-gaussian", "extra"]);
    }

    #[test]
    fn flags_accept_both_spellings() {
        let a = parse("bench --scale 4 --json=out.json --quiet");
        assert_eq!(a.value("scale"), Some("4"));
        assert_eq!(a.value("json"), Some("out.json"));
        assert!(a.has("quiet"));
        assert_eq!(a.value("quiet"), None);
    }

    /// `--json --quiet` must read as two flags. Treating `--quiet` as the value of
    /// `--json` would write the artifact to a file named `--quiet`.
    #[test]
    fn a_flag_does_not_swallow_the_next_flag() {
        let a = parse("bench --json --quiet");
        assert!(a.has("json"));
        assert_eq!(a.value("json"), None);
        assert!(a.has("quiet"));
    }

    #[test]
    fn short_aliases_expand() {
        let a = parse("-h");
        assert!(a.has("help"));
        assert!(parse("-V").has("version"));
    }

    #[test]
    fn double_dash_forces_positionals() {
        let a = parse("run -- --not-a-flag");
        assert_eq!(a.positional, ["--not-a-flag"]);
        assert!(!a.has("not-a-flag"));
    }

    #[test]
    fn numeric_flags_parse_and_report_errors() {
        let a = parse("bench --scale 8");
        assert_eq!(a.parsed::<usize>("scale").unwrap(), Some(8));
        assert_eq!(a.parsed_or::<usize>("missing", 3).unwrap(), 3);

        let bad = parse("bench --scale eight");
        let err = bad.parsed::<usize>("scale").unwrap_err();
        assert!(err.contains("--scale"), "{err}");
        assert!(err.contains("eight"), "{err}");
    }

    /// A mistyped flag must be reported. Silently ignoring `--jsno out.json` would
    /// produce a run with no artifact and no explanation.
    #[test]
    fn unknown_flags_are_detectable() {
        let a = parse("bench --scale 2 --jsno out.json");
        let unknown = a.unknown_flags(&["scale", "json"]);
        assert_eq!(unknown, ["jsno"]);
    }

    #[test]
    fn an_empty_command_line_parses_to_nothing() {
        let a = Args::parse(Vec::<String>::new());
        assert!(a.command.is_none());
        assert!(a.positional.is_empty());
        assert!(a.unknown_flags(&[]).is_empty());
    }
}
