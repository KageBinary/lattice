//! The molecular-dynamics vocabulary: thermostats, bonded topology, and analyses.
//!
//! Spec §12.4's module adds three kinds of setting to a `particles` block beyond the
//! force laws in [`crate::builtins`]:
//!
//! ```text
//!   thermostat: langevin(temperature=94 kelvin, friction=1 / picosecond);
//!   bonds:      chain(stiffness=300 newton/meter, length=1.5 angstrom);
//!   angles:     chain(stiffness=1e-19 joule, angle=180 degree);
//!   analysis:   rdf(bins=200, range=1 nanometer, every=10);
//! ```
//!
//! Topology is written in terms of *indices into the particle set in placement order*,
//! either implicitly (`chain`, `ring`) or explicitly (`pairs([[0, 1], …])`,
//! `triples([[0, 1, 2], …])`). The compiler expands the topology once, checks every
//! index against the count, and hands the domain resolved handles.

use lattice_domain_particle::{RdfRequest, Thermostat};
use lattice_syntax::{Diagnostic, Diagnostics, Expr, ExprKind, Span};
use lattice_units::Dimension;

use crate::builtins::{unknown, Call};
use crate::eval::Evaluator;

/// Which particles a bonded law joins.
#[derive(Clone, PartialEq, Debug)]
pub enum Topology {
    /// Consecutive particles `(i, i+1)`, or `(i, i+1, i+2)` for angles.
    Chain,
    /// A chain closed back onto its first particle.
    Ring,
    /// Explicit index tuples, as written.
    Explicit(Vec<Vec<usize>>),
}

impl Topology {
    fn name(&self) -> &'static str {
        match self {
            Topology::Chain => "chain",
            Topology::Ring => "ring",
            Topology::Explicit(_) => "explicit",
        }
    }
}

/// A `bonds:` setting.
#[derive(Clone, PartialEq, Debug)]
pub struct BondSpec {
    /// Which pairs.
    pub topology: Topology,
    /// Stiffness, N/m.
    pub stiffness: f64,
    /// Rest length, m. `None` means "the placement spacing", resolved by the compiler.
    pub length: Option<f64>,
    /// Whether pair laws still act across these bonds.
    pub keep_pair_forces: bool,
    /// Where it was written.
    pub span: Span,
}

/// An `angles:` setting.
#[derive(Clone, PartialEq, Debug)]
pub struct AngleSpec {
    /// Which triples.
    pub topology: Topology,
    /// Stiffness, J/rad².
    pub stiffness: f64,
    /// Rest angle, radians.
    pub angle: f64,
    /// Where it was written.
    pub span: Span,
}

const THERMOSTATS: &[&str] = &["langevin", "velocity_rescale", "berendsen"];
const BOND_TOPOLOGIES: &[&str] = &["chain", "ring", "pairs"];
const ANGLE_TOPOLOGIES: &[&str] = &["chain", "ring", "triples"];
const ANALYSES: &[&str] = &["rdf"];

/// Report a non-positive value where only a positive one makes sense.
fn require_positive(value: f64, what: &str, span: Span, diagnostics: &mut Diagnostics) -> Option<f64> {
    if value > 0.0 && value.is_finite() {
        Some(value)
    } else {
        diagnostics.push(
            Diagnostic::error(format!("{what} must be positive"))
                .with_code("E0405")
                .at(span, format!("{what} is {value}")),
        );
        None
    }
}

/// Parse a `thermostat:` setting.
pub fn thermostat(expr: &Expr, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<Thermostat> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected a thermostat")
                .with_code("E0210")
                .at(expr.span, "not a thermostat")
                .note(format!("available: {}", THERMOSTATS.join(", "))),
        );
        return None;
    };
    let temperature = |call: &Call<'_>, diagnostics: &mut Diagnostics| {
        let expr = call.require("temperature", 0, diagnostics)?;
        let value = evaluator.require(expr, Dimension::TEMPERATURE, "the target temperature", diagnostics)?;
        require_positive(value, "the target temperature", expr.span, diagnostics)
    };
    match call.name {
        "langevin" => {
            call.reject_unknown(&["temperature", "friction"], diagnostics);
            let temperature = temperature(&call, diagnostics);
            let friction_expr = call.require("friction", 1, diagnostics);
            let friction = friction_expr.and_then(|expr| {
                let value = evaluator.require(expr, Dimension::FREQUENCY, "the friction", diagnostics)?;
                require_positive(value, "the friction", expr.span, diagnostics)
            });
            Some(Thermostat::Langevin { temperature: temperature?, friction: friction? })
        }
        "velocity_rescale" | "berendsen" => {
            call.reject_unknown(&["temperature", "relaxation"], diagnostics);
            let temperature = temperature(&call, diagnostics);
            let relaxation_expr = call.require("relaxation", 1, diagnostics);
            let relaxation = relaxation_expr.and_then(|expr| {
                let value = evaluator.require(expr, Dimension::TIME, "the relaxation time", diagnostics)?;
                require_positive(value, "the relaxation time", expr.span, diagnostics)
            });
            Some(Thermostat::VelocityRescale { temperature: temperature?, relaxation: relaxation? })
        }
        other => {
            unknown("thermostat", other, call.span, THERMOSTATS, diagnostics);
            None
        }
    }
}

/// Parse a keyword-valued argument such as `pair_forces=excluded`.
fn keyword<'a>(
    call: &Call<'_>,
    key: &str,
    choices: &[&'a str],
    diagnostics: &mut Diagnostics,
) -> Option<Option<&'a str>> {
    let Some(expr) = call.named(key) else { return Some(None) };
    match expr.as_name().and_then(|name| choices.iter().find(|c| **c == name)) {
        Some(choice) => Some(Some(choice)),
        None => {
            diagnostics.push(
                Diagnostic::error(format!("`{key}` must be one of: {}", choices.join(", ")))
                    .with_code("E0208")
                    .at(expr.span, "unknown keyword value"),
            );
            None
        }
    }
}

/// Parse an explicit list of index tuples, each of `arity` entries.
fn index_tuples(
    expr: &Expr,
    arity: usize,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Vec<Vec<usize>>> {
    let ExprKind::List(items) = &expr.kind else {
        diagnostics.push(
            Diagnostic::error(format!("expected a list of {arity}-element index lists"))
                .with_code("E0401")
                .at(expr.span, "not a list")
                .help(format!("write it as `[[0, 1], [1, 2]]` with {arity} indices per entry")),
        );
        return None;
    };
    let mut tuples = Vec::with_capacity(items.len());
    for item in items {
        let entries = evaluator.fixed_list(item, arity, "an index tuple", diagnostics)?;
        let mut tuple = Vec::with_capacity(arity);
        for entry in entries {
            tuple.push(evaluator.count(entry, "a particle index", diagnostics)?);
        }
        tuples.push(tuple);
    }
    Some(tuples)
}

/// Parse a `bonds:` setting.
pub fn bonds(expr: &Expr, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<BondSpec> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected a bond topology")
                .with_code("E0210")
                .at(expr.span, "not a bond topology")
                .note(format!("available: {}", BOND_TOPOLOGIES.join(", "))),
        );
        return None;
    };
    let (topology, first_named) = match call.name {
        "chain" => (Topology::Chain, 0),
        "ring" => (Topology::Ring, 0),
        "pairs" => {
            let list = call.positional(0).or_else(|| call.named("pairs"));
            let Some(list) = list else {
                diagnostics.push(
                    Diagnostic::error("`pairs` needs a list of index pairs")
                        .with_code("E0203")
                        .at(call.span, "missing the pair list")
                        .help("write `pairs([[0, 1], [2, 3]], stiffness=…, length=…)`"),
                );
                return None;
            };
            (Topology::Explicit(index_tuples(list, 2, evaluator, diagnostics)?), 1)
        }
        other => {
            unknown("bond topology", other, call.span, BOND_TOPOLOGIES, diagnostics);
            return None;
        }
    };
    call.reject_unknown(&["stiffness", "length", "pair_forces", "pairs"], diagnostics);

    let stiffness = call.require("stiffness", first_named, diagnostics).and_then(|expr| {
        let value = evaluator.require(expr, Dimension::STIFFNESS, "the bond stiffness", diagnostics)?;
        require_positive(value, "the bond stiffness", expr.span, diagnostics)
    });
    let length = match call.get("length", first_named + 1) {
        Some(expr) => {
            let value = evaluator.require(expr, Dimension::LENGTH, "the bond rest length", diagnostics)?;
            Some(require_positive(value, "the bond rest length", expr.span, diagnostics)?)
        }
        None => None,
    };
    let keep_pair_forces = keyword(&call, "pair_forces", &["excluded", "included"], diagnostics)? == Some("included");

    Some(BondSpec { topology, stiffness: stiffness?, length, keep_pair_forces, span: call.span })
}

/// Parse an `angles:` setting.
pub fn angles(expr: &Expr, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<AngleSpec> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected an angle topology")
                .with_code("E0210")
                .at(expr.span, "not an angle topology")
                .note(format!("available: {}", ANGLE_TOPOLOGIES.join(", "))),
        );
        return None;
    };
    let (topology, first_named) = match call.name {
        "chain" => (Topology::Chain, 0),
        "ring" => (Topology::Ring, 0),
        "triples" => {
            let list = call.positional(0).or_else(|| call.named("triples"));
            let Some(list) = list else {
                diagnostics.push(
                    Diagnostic::error("`triples` needs a list of index triples")
                        .with_code("E0203")
                        .at(call.span, "missing the triple list")
                        .help("write `triples([[0, 1, 2]], stiffness=…, angle=…)`"),
                );
                return None;
            };
            (Topology::Explicit(index_tuples(list, 3, evaluator, diagnostics)?), 1)
        }
        other => {
            unknown("angle topology", other, call.span, ANGLE_TOPOLOGIES, diagnostics);
            return None;
        }
    };
    call.reject_unknown(&["stiffness", "angle", "triples"], diagnostics);

    let stiffness = call.require("stiffness", first_named, diagnostics).and_then(|expr| {
        // Energy per radian squared; a radian is dimensionless, so this is an energy.
        let value = evaluator.require(expr, Dimension::ENERGY, "the angle stiffness", diagnostics)?;
        require_positive(value, "the angle stiffness", expr.span, diagnostics)
    });
    let angle = call.require("angle", first_named + 1, diagnostics).and_then(|expr| {
        let value = evaluator.dimensionless(expr, "the rest angle", diagnostics)?;
        if (0.0..=core::f64::consts::PI + 1e-12).contains(&value) {
            Some(value)
        } else {
            diagnostics.push(
                Diagnostic::error("the rest angle must lie between 0 and 180 degrees")
                    .with_code("E0405")
                    .at(expr.span, format!("{value} rad"))
                    .help("write it as `angle=120 degree` or in radians"),
            );
            None
        }
    });

    Some(AngleSpec { topology, stiffness: stiffness?, angle: angle?, span: call.span })
}

/// Parse an `analysis:` setting.
pub fn analysis(expr: &Expr, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<RdfRequest> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected an analysis")
                .with_code("E0210")
                .at(expr.span, "not an analysis")
                .note(format!("available: {}", ANALYSES.join(", "))),
        );
        return None;
    };
    match call.name {
        "rdf" => {
            call.reject_unknown(&["bins", "range", "every"], diagnostics);
            let bins = call.require("bins", 0, diagnostics).and_then(|expr| {
                let bins = evaluator.count(expr, "the bin count", diagnostics)?;
                if bins == 0 {
                    diagnostics.push(
                        Diagnostic::error("an RDF needs at least one bin").with_code("E0405").at(expr.span, "zero bins"),
                    );
                    return None;
                }
                Some(bins)
            });
            let range = call.require("range", 1, diagnostics).and_then(|expr| {
                let value = evaluator.require(expr, Dimension::LENGTH, "the RDF range", diagnostics)?;
                require_positive(value, "the RDF range", expr.span, diagnostics)
            });
            let every = call.require("every", 2, diagnostics).and_then(|expr| {
                let every = evaluator.count(expr, "the sampling interval in steps", diagnostics)?;
                if every == 0 {
                    diagnostics.push(
                        Diagnostic::error("the sampling interval must be at least one step")
                            .with_code("E0405")
                            .at(expr.span, "every 0 steps"),
                    );
                    return None;
                }
                Some(every as u64)
            });
            Some(RdfRequest { bins: bins?, range: range?, every: every? })
        }
        other => {
            unknown("analysis", other, call.span, ANALYSES, diagnostics);
            None
        }
    }
}

/// Expand a topology into index tuples of `arity` over `count` particles.
///
/// Every index is checked against the count and every tuple against itself, so the
/// domain never sees a bond to a particle that does not exist or to itself.
pub fn expand(
    topology: &Topology,
    arity: usize,
    count: usize,
    what: &str,
    span: Span,
    diagnostics: &mut Diagnostics,
) -> Option<Vec<Vec<usize>>> {
    let tuples: Vec<Vec<usize>> = match topology {
        Topology::Chain => {
            if count < arity {
                diagnostics.push(
                    Diagnostic::error(format!("a {what} chain needs at least {arity} particles"))
                        .with_code("E0405")
                        .at(span, format!("the set has {count}")),
                );
                return None;
            }
            (0..=count - arity).map(|i| (i..i + arity).collect()).collect()
        }
        Topology::Ring => {
            if count < arity + 1 {
                diagnostics.push(
                    Diagnostic::error(format!("a {what} ring needs at least {} particles", arity + 1))
                        .with_code("E0405")
                        .at(span, format!("the set has {count}")),
                );
                return None;
            }
            (0..count).map(|i| (i..i + arity).map(|j| j % count).collect()).collect()
        }
        Topology::Explicit(tuples) => tuples.clone(),
    };

    let mut ok = true;
    for tuple in &tuples {
        for &index in tuple {
            if index >= count {
                diagnostics.push(
                    Diagnostic::error(format!("{what} names particle {index}, but the set has {count}"))
                        .with_code("E0405")
                        .at(span, "index out of range")
                        .note("particle indices count from 0 in placement order"),
                );
                ok = false;
            }
        }
        let mut sorted = tuple.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != tuple.len() {
            diagnostics.push(
                Diagnostic::error(format!("{what} {tuple:?} names the same particle twice"))
                    .with_code("E0405")
                    .at(span, "degenerate topology"),
            );
            ok = false;
        }
    }
    ok.then_some(tuples)
}

impl BondSpec {
    /// A one-line description for the model report.
    pub fn describe(&self, count: usize) -> String {
        let pairs = match &self.topology {
            Topology::Chain => count.saturating_sub(1),
            Topology::Ring => count,
            Topology::Explicit(t) => t.len(),
        };
        format!(
            "{pairs} harmonic bonds ({}), k = {:.4e} N/m{}",
            self.topology.name(),
            self.stiffness,
            match self.length {
                Some(length) => format!(", r0 = {length:.4e} m"),
                None => ", r0 = placement spacing".to_string(),
            }
        )
    }
}

impl AngleSpec {
    /// A one-line description for the model report.
    pub fn describe(&self, count: usize) -> String {
        let triples = match &self.topology {
            Topology::Chain => count.saturating_sub(2),
            Topology::Ring => count,
            Topology::Explicit(t) => t.len(),
        };
        format!(
            "{triples} harmonic angles ({}), k = {:.4e} J/rad^2, theta0 = {:.4} rad",
            self.topology.name(),
            self.stiffness,
            self.angle
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_syntax::{parse, Item, SourceFile};
    use lattice_units::UnitRegistry;

    fn parse_setting(source: &str) -> (SourceFile, Expr) {
        let file = SourceFile::new("t.lattice", format!("project p {{ x: {source}; }}"));
        let (project, _) = parse(&file);
        let project = project.expect("should parse");
        let Item::Setting(setting) = &project.items[0] else { panic!("expected a setting") };
        (file, setting.value.clone())
    }

    fn with<T>(source: &str, f: impl FnOnce(&Expr, &Evaluator<'_>, &mut Diagnostics) -> Option<T>) -> (Option<T>, Diagnostics, SourceFile) {
        let (file, expr) = parse_setting(source);
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let mut diagnostics = Diagnostics::default();
        let result = f(&expr, &evaluator, &mut diagnostics);
        (result, diagnostics, file)
    }

    #[test]
    fn thermostats_parse_with_their_dimensions() {
        let (t, d, f) = with("langevin(temperature=94 kelvin, friction=1 / picosecond)", thermostat);
        assert!(!d.has_errors(), "{}", d.render(&f));
        assert_eq!(t, Some(Thermostat::Langevin { temperature: 94.0, friction: 1e12 }));

        let (t, d, _) = with("berendsen(temperature=300 kelvin, relaxation=0.1 picosecond)", thermostat);
        assert!(!d.has_errors());
        assert_eq!(t, Some(Thermostat::VelocityRescale { temperature: 300.0, relaxation: 1e-13 }));

        let (t, d, _) = with("langevin(temperature=94 kelvin, friction=1 second)", thermostat);
        assert!(t.is_none() && d.codes().contains(&"E0400"), "seconds are not a friction");
        let (t, d, _) = with("langevin(temperature=0 kelvin, friction=1 hertz)", thermostat);
        assert!(t.is_none() && d.codes().contains(&"E0405"), "0 K is not a target");
        let (t, d, _) = with("nose_hoover(temperature=1 kelvin)", thermostat);
        assert!(t.is_none() && d.codes().contains(&"E0210"));
    }

    #[test]
    fn bond_topologies_expand_and_check_their_indices() {
        let (b, d, f) = with("chain(stiffness=10 newton/meter, length=1 meter)", bonds);
        assert!(!d.has_errors(), "{}", d.render(&f));
        let b = b.unwrap();
        assert_eq!(b.topology, Topology::Chain);
        assert!(!b.keep_pair_forces);
        let mut diagnostics = Diagnostics::default();
        let pairs = expand(&b.topology, 2, 4, "bond", b.span, &mut diagnostics).unwrap();
        assert_eq!(pairs, vec![vec![0, 1], vec![1, 2], vec![2, 3]]);

        let (r, _, _) = with("ring(stiffness=10 newton/meter, pair_forces=included)", bonds);
        let r = r.unwrap();
        assert!(r.keep_pair_forces && r.length.is_none());
        let ring = expand(&r.topology, 2, 3, "bond", r.span, &mut diagnostics).unwrap();
        assert_eq!(ring, vec![vec![0, 1], vec![1, 2], vec![2, 0]]);

        let (p, d, f) = with("pairs([[0, 3], [1, 2]], stiffness=5 newton/meter, length=2 meter)", bonds);
        assert!(!d.has_errors(), "{}", d.render(&f));
        let p = p.unwrap();
        assert_eq!(p.topology, Topology::Explicit(vec![vec![0, 3], vec![1, 2]]));
        assert!(expand(&p.topology, 2, 3, "bond", p.span, &mut diagnostics).is_none(), "index 3 of 3");
        assert!(diagnostics.codes().contains(&"E0405"));

        let (p, _, _) = with("pairs([[1, 1]], stiffness=5 newton/meter)", bonds);
        let mut diagnostics = Diagnostics::default();
        assert!(expand(&p.unwrap().topology, 2, 3, "bond", Span::default(), &mut diagnostics).is_none());

        let (b, d, _) = with("chain(stiffness=10 newton)", bonds);
        assert!(b.is_none() && d.codes().contains(&"E0400"));
        let (b, d, _) = with("chain(stiffness=10 newton/meter, pair_forces=sometimes)", bonds);
        assert!(b.is_none() && d.codes().contains(&"E0208"));
    }

    #[test]
    fn angles_take_degrees_or_radians() {
        let (a, d, f) = with("chain(stiffness=2e-19 joule, angle=120 degree)", angles);
        assert!(!d.has_errors(), "{}", d.render(&f));
        let a = a.unwrap();
        assert!((a.angle - 2.0 * core::f64::consts::PI / 3.0).abs() < 1e-12);
        let mut diagnostics = Diagnostics::default();
        let triples = expand(&a.topology, 3, 4, "angle", a.span, &mut diagnostics).unwrap();
        assert_eq!(triples, vec![vec![0, 1, 2], vec![1, 2, 3]]);
        assert!(expand(&Topology::Chain, 3, 2, "angle", a.span, &mut diagnostics).is_none());

        let (a, d, _) = with("chain(stiffness=2e-19 joule, angle=200 degree)", angles);
        assert!(a.is_none() && d.codes().contains(&"E0405"));
        let (a, d, _) = with("triples([[0, 1, 2]], stiffness=1 joule, angle=3)", angles);
        assert!(a.is_some() && !d.has_errors());
    }

    #[test]
    fn analyses_parse() {
        let (r, d, f) = with("rdf(bins=100, range=2 nanometer, every=5)", analysis);
        assert!(!d.has_errors(), "{}", d.render(&f));
        assert_eq!(r, Some(RdfRequest { bins: 100, range: 2e-9, every: 5 }));
        let (r, d, _) = with("rdf(bins=100, range=2 nanometer, every=0)", analysis);
        assert!(r.is_none() && d.codes().contains(&"E0405"));
        let (r, d, _) = with("msd(every=1)", analysis);
        assert!(r.is_none() && d.codes().contains(&"E0210"));
    }

    #[test]
    fn descriptions_count_their_topology() {
        let (b, _, _) = with("chain(stiffness=10 newton/meter)", bonds);
        assert!(b.unwrap().describe(8).starts_with("7 harmonic bonds (chain)"));
        let (a, _, _) = with("ring(stiffness=1 joule, angle=90 degree)", angles);
        assert!(a.unwrap().describe(8).starts_with("8 harmonic angles (ring)"));
    }
}
