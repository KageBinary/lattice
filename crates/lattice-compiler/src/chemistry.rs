//! The chemistry front end: `reaction` declarations and the species attributes they
//! need.
//!
//! Spec §12.2's example, which this parses:
//!
//! ```text
//! species H_plus on chamber = 0 mole / meter^2 {
//!   formula:   H;
//!   charge:    1;
//!   diffusion: 9.31e-9 meter^2 / second;
//! }
//!
//! reaction acid_base {
//!   reactants: H_plus + OH_minus;
//!   products:  H2O;
//!   rate:      1.4e11 meter^2 / (mole second);
//!   enthalpy:  -57.3 kilojoule / mole;
//! }
//! ```
//!
//! # Why the rate constant's unit is checked against the reaction's order
//!
//! A rate constant is in `(m²/mol)^(n−1)/s` where `n` is the total order. A first-order
//! constant is `1/s`; a second-order one is `m²/(mol·s)`. Writing the first where the
//! second belongs is wrong by a factor with the dimensions of a concentration, and the
//! *number* looks entirely plausible either way — rate constants span twenty orders of
//! magnitude, so nothing about `1.4e11` says which it is.
//!
//! The compiler knows the order, because it just read the reactants. So it computes the
//! expected dimension and checks the declared constant against it, which turns a class
//! of silent error into a message naming both units.
//!
//! # Why an unbalanced reaction is an error rather than a warning
//!
//! `H2 + O2 -> H2O` integrates perfectly happily and destroys 47% of the mass it
//! touches. The only symptom is a conservation check failing several layers away, in a
//! ledger, long after the typo. When every species states its composition there is no
//! reason to let that through.
//!
//! A species that does *not* state a composition makes the check impossible, and that
//! is reported as a warning rather than an error: it is a real gap in what can be
//! verified, but it is also a legitimate way to write an abstract `A -> B` model.
//!
//! # Spec §8.3's form
//!
//! ```text
//! reaction neutralization {
//!     stoichiometry: H_plus + OH_minus -> H2O;
//!     rate: k * c(H_plus) * c(OH_minus);
//!     heat_release: 57.3 kilojoule / mole;
//! }
//! ```
//!
//! `stoichiometry:` is `reactants:` and `products:` in one line, and `heat_release:` is
//! `enthalpy:` with the sign a reader expects: positive releases heat. A `rate:` that
//! reads the state — `c(X)`, `temperature`, `arrhenius(Ea)` — is not a constant for mass
//! action to multiply but the reaction's rate itself, in mol/(m²·s), checked by
//! [`laws::check_rate`] and compiled to a [`RateExpression`]. One that reads nothing is
//! a rate constant, as before, so the two forms mix freely in one network.

use std::collections::BTreeMap;

use lattice_domain_chemistry::{
    Composition, FormulaError, RateExpression, RateLaw, Reaction, Species, Term,
};
use lattice_syntax::{Decl, Diagnostic, Diagnostics, Expr, ExprKind, Span};
use lattice_units::Dimension;

use crate::eval::Evaluator;
use crate::laws;
use crate::lower_law::lower_rate;

/// Settings a `reaction` declaration accepts.
pub const REACTION_SETTINGS: &[&str] = &[
    "reactants",
    "products",
    "rate",
    "reverse_rate",
    "activation_energy",
    "enthalpy",
    // Spec §8.3's form.
    "stoichiometry",
    "heat_release",
];

/// True when a `rate:` reads the state, and so is a rate law rather than a constant: it
/// calls something (`c(H_plus)`, `arrhenius(…)`), reads a member, or names `temperature`
/// or a species. The model's constants are not the state, so a constant called
/// `temperature` leaves a constant rate a constant.
pub fn is_rate_law(expr: &Expr, evaluator: &Evaluator<'_>, species: &[String]) -> bool {
    match &expr.kind {
        ExprKind::Call(..) | ExprKind::Member(..) | ExprKind::If(..) => true,
        ExprKind::Name(name) => {
            (name == "temperature" || species.contains(name)) && !evaluator.is_constant(name)
        }
        ExprKind::Unary(_, inner) | ExprKind::Power(inner, _) => is_rate_law(inner, evaluator, species),
        ExprKind::Binary(_, a, b) => is_rate_law(a, evaluator, species) || is_rate_law(b, evaluator, species),
        _ => false,
    }
}

/// Chemistry settings a `species` declaration accepts, on top of the field ones.
pub const SPECIES_SETTINGS: &[&str] = &["formula", "charge", "molar_mass"];

/// One side of a reaction, as written.
#[derive(Clone, PartialEq, Debug)]
pub struct SideTerm {
    /// The species name.
    pub name: String,
    /// Its stoichiometric coefficient.
    pub coefficient: f64,
    /// Where it appeared, for a diagnostic.
    pub span: Span,
}

/// Read `2 A + B` into its terms.
///
/// The grammar already parses this: `+` is an infix operator and `2 A` is a
/// juxtaposition, which binds tighter — the same rule that makes `10 meter / 2 second`
/// a velocity rather than a metre-second. So no new syntax was needed for stoichiometry;
/// walking the expression the parser already built is enough.
pub fn side(
    expr: &Expr,
    _evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Vec<SideTerm>> {
    let mut terms = Vec::new();
    collect_side(expr, &mut terms, diagnostics)?;
    if terms.is_empty() {
        diagnostics.push(
            Diagnostic::error("a reaction side needs at least one species")
                .with_code("E0203")
                .at(expr.span, "nothing here")
                .help("write `reactants: 2 fuel + oxidizer;`"),
        );
        return None;
    }
    Some(terms)
}

fn collect_side(expr: &Expr, out: &mut Vec<SideTerm>, diagnostics: &mut Diagnostics) -> Option<()> {
    use lattice_syntax::BinaryOp;
    match &expr.kind {
        // `a + b`: both sides are terms.
        ExprKind::Binary(BinaryOp::Add, left, right) => {
            collect_side(left, out, diagnostics)?;
            collect_side(right, out, diagnostics)
        }
        // `2 A`: a coefficient times a species. Juxtaposition parses as a product.
        ExprKind::Binary(BinaryOp::Mul, left, right) => {
            let (Some(coefficient), Some(name)) = (constant(left), right.as_name()) else {
                diagnostics.push(
                    Diagnostic::error("a reaction term is a species, optionally with a count")
                        .with_code("E0401")
                        .at(expr.span, "not `<number> <species>`")
                        .help("write `2 H2` or just `H2`"),
                );
                return None;
            };
            out.push(SideTerm { name: name.to_string(), coefficient, span: expr.span });
            Some(())
        }
        ExprKind::Name(name) => {
            out.push(SideTerm { name: name.clone(), coefficient: 1.0, span: expr.span });
            Some(())
        }
        _ => {
            diagnostics.push(
                Diagnostic::error("a reaction side is a sum of species")
                    .with_code("E0401")
                    .at(expr.span, "not a species or a sum of them")
                    .help("write `reactants: 2 fuel + oxidizer;`"),
            );
            None
        }
    }
}

/// A bare numeric literal, with no unit — a stoichiometric coefficient.
fn constant(expr: &Expr) -> Option<f64> {
    match &expr.kind {
        ExprKind::Number(value) => Some(*value),
        _ => None,
    }
}

/// Read a `reaction` declaration into a solver reaction.
///
/// `index_of` resolves a species name to its slot in the network, and `species` lists
/// the names in slot order, which is the order a rate law's `c(…)` indexes.
pub fn reaction(
    decl: &Decl,
    index_of: &BTreeMap<String, usize>,
    species: &[String],
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Reaction> {
    let (reactants, products) = sides(decl, evaluator, diagnostics)?;

    let resolve = |terms: &[SideTerm], diagnostics: &mut Diagnostics| -> Option<Vec<Term>> {
        terms
            .iter()
            .map(|term| match index_of.get(&term.name) {
                Some(index) => Some(Term::new(*index, term.coefficient)),
                None => {
                    let known: Vec<&str> = index_of.keys().map(String::as_str).collect();
                    diagnostics.push(
                        Diagnostic::error(format!("there is no species called `{}`", term.name))
                            .with_code("E0202")
                            .at(term.span, "unknown species")
                            .help(if known.is_empty() {
                                "declare one with `species A on <grid> = …;`".to_string()
                            } else {
                                format!("declared species: {}", known.join(", "))
                            }),
                    );
                    None
                }
            })
            .collect()
    };

    let reactant_terms = resolve(&reactants, diagnostics)?;
    let product_terms = match &products {
        Some(terms) => resolve(terms, diagnostics)?,
        None => Vec::new(),
    };

    let enthalpy = enthalpy(decl, evaluator, diagnostics);

    // A rate law is the rate itself; it brings its own temperature dependence and its
    // own reverse step, so the settings for mass action's have nothing to modify.
    if let Some(rate) = decl.setting("rate").filter(|rate| is_rate_law(&rate.value, evaluator, species)) {
        for key in ["reverse_rate", "activation_energy"] {
            if let Some(setting) = decl.setting(key) {
                diagnostics.push(
                    Diagnostic::error(format!("`{key}` modifies a rate constant, and this rate is a law"))
                        .with_code("E0209")
                        .at(setting.key.span, "for mass action")
                        .also(rate.value.span, "a rate law, which says its own")
                        .help(if key == "reverse_rate" {
                            "subtract the reverse rate inside the law: `kf * c(A) - kr * c(B)`"
                        } else {
                            "write the temperature dependence into the law: `k * arrhenius(50 kilojoule / mole) * c(A)`"
                        }),
                );
            }
        }
        let typed = laws::check_rate(&decl.name, &rate.value, species, evaluator, diagnostics)?;
        let program = match lower_rate(&typed, species.len()) {
            Ok(program) => program,
            Err(diagnostic) => {
                diagnostics.push(*diagnostic);
                return None;
            }
        };
        if decl.setting("reverse_rate").is_some() || decl.setting("activation_energy").is_some() {
            return None;
        }
        let text = evaluator.file().slice(rate.value.span).to_string();
        let law = RateExpression::new(program, text);
        return Some(
            Reaction::with_expression(decl.name.text.clone(), reactant_terms, product_terms, law)
                .with_enthalpy(enthalpy?),
        );
    }

    // The order sets the rate constant's dimension, so it has to be known before the
    // constant can be checked — which is why the sides are read first.
    let order: f64 = reactants.iter().map(|term| term.coefficient).sum();
    let forward = rate_constant(decl, "rate", order, evaluator, diagnostics)?;

    let reverse_order: f64 = products.as_ref().map_or(0.0, |terms| {
        terms.iter().map(|term| term.coefficient).sum()
    });
    let reverse = decl
        .setting("reverse_rate")
        .and_then(|_| rate_constant(decl, "reverse_rate", reverse_order, evaluator, diagnostics))
        .unwrap_or(0.0);

    let mut law = RateLaw { forward, reverse, temperature: Default::default() };
    if let Some(setting) = decl.setting("activation_energy") {
        let expected = Dimension::ENERGY
            .try_div(Dimension::AMOUNT)
            .expect("energy per amount is representable");
        if let Some(activation) =
            evaluator.require(&setting.value, expected, "`activation_energy`", diagnostics)
        {
            if activation < 0.0 {
                diagnostics.push(
                    Diagnostic::error("an activation energy cannot be negative")
                        .with_code("E0405")
                        .at(setting.value.span, format!("this is {activation} J/mol"))
                        .note(
                            "a negative barrier would make the reaction run slower as it gets \
                             hotter, which is not what Arrhenius describes",
                        ),
                );
            } else {
                law = law.with_activation_energy(activation);
            }
        }
    }

    Some(
        Reaction::new(decl.name.text.clone(), reactant_terms, product_terms, law)
            .with_enthalpy(enthalpy?),
    )
}

/// The two sides, from `reactants:` and `products:` or from spec §8.3's
/// `stoichiometry: A + B -> C;`, which says both at once and so excludes them.
fn sides(
    decl: &Decl,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<(Vec<SideTerm>, Option<Vec<SideTerm>>)> {
    if let Some(stoichiometry) = decl.setting("stoichiometry") {
        let mut conflict = false;
        for key in ["reactants", "products"] {
            if let Some(setting) = decl.setting(key) {
                diagnostics.push(
                    Diagnostic::error(format!("`stoichiometry` and `{key}` both say what reacts"))
                        .with_code("E0209")
                        .at(setting.key.span, format!("sets the {key}"))
                        .also(stoichiometry.key.span, "sets both sides")
                        .help("keep one: `stoichiometry: A + B -> C;`, or `reactants:` with `products:`"),
                );
                conflict = true;
            }
        }
        let ExprKind::Yields(left, right) = &stoichiometry.value.kind else {
            diagnostics.push(
                Diagnostic::error("a stoichiometry is written `reactants -> products`")
                    .with_code("E0401")
                    .at(stoichiometry.value.span, "no `->`")
                    .help("write `stoichiometry: H_plus + OH_minus -> H2O;`"),
            );
            return None;
        };
        let reactants = side(left, evaluator, diagnostics);
        let products = side(right, evaluator, diagnostics);
        return if conflict { None } else { Some((reactants?, Some(products?))) };
    }

    let Some(setting) = decl.setting("reactants") else {
        diagnostics.push(
            Diagnostic::error(format!("reaction `{}` needs `reactants`", decl.name.text))
                .with_code("E0203")
                .at(decl.name.span, "missing `reactants`")
                .help("add `reactants: 2 fuel + oxidizer;`, or `stoichiometry: 2 fuel + oxidizer -> exhaust;`"),
        );
        return None;
    };
    let reactants = side(&setting.value, evaluator, diagnostics)?;
    // Products may be empty — a species leaving the system to something the model does
    // not track is a legitimate sink, and `products: (nothing)` has no syntax.
    let products = decl.setting("products").and_then(|s| side(&s.value, evaluator, diagnostics));
    Some((reactants, products))
}

/// The enthalpy of reaction, J/mol, from `enthalpy:` or from spec §8.3's
/// `heat_release:`, its negation: chemistry writes an exothermic enthalpy as negative,
/// and a heat *release* reads the way a reader expects, positive for heat out. `None`
/// after reporting a problem; zero when neither is written.
fn enthalpy(decl: &Decl, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<f64> {
    let expected = Dimension::ENERGY.try_div(Dimension::AMOUNT).expect("energy per amount is representable");
    match (decl.setting("enthalpy"), decl.setting("heat_release")) {
        (Some(enthalpy), Some(release)) => {
            diagnostics.push(
                Diagnostic::error("`enthalpy` and `heat_release` both say how much heat the reaction moves")
                    .with_code("E0209")
                    .at(enthalpy.key.span, "negative for heat out")
                    .also(release.key.span, "positive for heat out")
                    .help("keep one; `heat_release: 57.3 kilojoule / mole` is `enthalpy: -57.3 kilojoule / mole`"),
            );
            None
        }
        (Some(enthalpy), None) => evaluator.require(&enthalpy.value, expected, "`enthalpy`", diagnostics),
        (None, Some(release)) => {
            evaluator.require(&release.value, expected, "`heat_release`", diagnostics).map(|heat| -heat)
        }
        (None, None) => Some(0.0),
    }
}

/// Compile a rate law written alone, as a reaction's `rate:` would compile it, for a
/// mixture whose species are `species` in slot order. For validation and tests, which
/// hold a law against a closed form without building a whole model; the text is
/// the law's own, so it may use units but no model constants.
///
/// Returns the rendered diagnostics if the law does not compile.
pub fn rate_law_from_source(species: &[&str], rate: &str) -> Result<RateExpression, String> {
    let file = lattice_syntax::SourceFile::new("rate.lattice", format!("project law {{ rate: {rate}; }}"));
    let (project, mut diagnostics) = lattice_syntax::parse(&file);
    let Some(project) = project.filter(|_| !diagnostics.has_errors()) else {
        return Err(diagnostics.render(&file));
    };
    let Some(setting) = project.setting("rate") else { return Err(diagnostics.render(&file)) };
    let units = lattice_units::UnitRegistry::si();
    let evaluator = Evaluator::new(&file, &units);
    let names: Vec<String> = species.iter().map(|s| s.to_string()).collect();
    let typed = laws::check_rate(&setting.key, &setting.value, &names, &evaluator, &mut diagnostics);
    let program = typed.as_ref().map(|typed| lower_rate(typed, names.len()));
    match program {
        Some(Ok(program)) if !diagnostics.has_errors() => {
            Ok(RateExpression::new(program, file.slice(setting.value.span).to_string()))
        }
        Some(Err(diagnostic)) => {
            diagnostics.push(*diagnostic);
            Err(diagnostics.render(&file))
        }
        _ => Err(diagnostics.render(&file)),
    }
}

/// Read a rate constant, checking its unit against the reaction's order.
fn rate_constant(
    decl: &Decl,
    key: &str,
    order: f64,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<f64> {
    let Some(setting) = decl.setting(key) else {
        if key == "rate" {
            diagnostics.push(
                Diagnostic::error(format!("reaction `{}` needs a `rate`", decl.name.text))
                    .with_code("E0203")
                    .at(decl.name.span, "missing `rate`")
                    .help(format!(
                        "for a reaction of order {order}, that is a constant in {}",
                        RateLaw::si_unit_for_order(order)
                    )),
            );
        }
        return None;
    };

    let expected = rate_dimension(order)?;
    // A rate where a rate constant belongs: the author meant a law, and a constant law
    // is zeroth order, which keeps consuming a reactant that has run out. Said plainly
    // rather than as a bare unit mismatch.
    let mut quiet = Diagnostics::new();
    if expected != Dimension::AREAL_REACTION_RATE
        && evaluator.quantity(&setting.value, &mut quiet).is_some_and(|q| q.dimension() == Dimension::AREAL_REACTION_RATE)
    {
        diagnostics.push(
            Diagnostic::error(format!("`{key}` is a rate, and a constant `{key}:` is a rate constant"))
                .with_code("E0400")
                .at(setting.value.span, format!("mol/(m^2 s), where order {order} needs {}", RateLaw::si_unit_for_order(order)))
                .note(
                    "a `rate:` that reads no concentration is mass action's rate constant; one that \
                     reads them with `c(…)` is the rate itself. A constant rate would be zeroth \
                     order, and would keep consuming a reactant after it ran out",
                )
                .help("write the law, such as `k * c(A)`, or give the rate constant in its own unit"),
        );
        return None;
    }
    let value = evaluator.require(
        &setting.value,
        expected,
        &format!("`{key}` for a reaction of order {order}"),
        diagnostics,
    )?;
    if value < 0.0 {
        diagnostics.push(
            Diagnostic::error("a rate constant cannot be negative")
                .with_code("E0405")
                .at(setting.value.span, format!("this is {value}"))
                .note("a negative constant would run the reaction backwards without an equilibrium"),
        );
        return None;
    }
    Some(value)
}

/// The dimension of a rate constant for a reaction of total order `order`.
///
/// `(m²/mol)^(order−1) / s`. Areal, not volumetric — this is a 2D engine, and the
/// familiar `(m³/mol)^(n−1)/s` would be off by a length per order above the first.
fn rate_dimension(order: f64) -> Option<Dimension> {
    let exponent = order - 1.0;
    if exponent.fract() != 0.0 {
        // A fractional order is legitimate chemistry — empirical rate laws have them —
        // but its constant has a fractional dimension, which the seven-exponent
        // representation cannot hold. Refusing to *check* is better than checking
        // against the wrong thing.
        return None;
    }
    let per_concentration = Dimension::AREA.try_div(Dimension::AMOUNT).ok()?;
    let scaled = per_concentration.try_powi(exponent as i32).ok()?;
    scaled.try_div(Dimension::TIME).ok()
}

/// Read the chemistry attributes of a `species` declaration.
///
/// `diffusion` comes from the field's own `diffusivity` setting, so it is not read here
/// — the species and the field are the same declaration wearing two hats.
pub fn species(
    name: &str,
    settings: &[lattice_syntax::Setting],
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Species {
    let mut species = Species::new(name);
    let get = |key: &str| settings.iter().find(|s| s.key.text == key);

    if let Some(setting) = get("formula") {
        // A formula is written as a bare name — `H2O` lexes as an identifier — so it
        // arrives as an expression rather than a string.
        let text = evaluator
            .as_name(&setting.value)
            .map(str::to_string)
            .or_else(|| match &setting.value.kind {
                ExprKind::Str(text) => Some(text.clone()),
                _ => None,
            });
        match text {
            Some(text) => match Composition::parse(&text) {
                Ok(composition) => species = species.with_composition(composition),
                Err(error) => diagnostics.push(formula_diagnostic(&error, setting.value.span)),
            },
            None => diagnostics.push(
                Diagnostic::error("a formula is a chemical formula such as `H2O`")
                    .with_code("E0401")
                    .at(setting.value.span, "not a formula"),
            ),
        }
    }

    if let Some(setting) = get("charge")
        && let Some(charge) = evaluator.dimensionless(&setting.value, "`charge`", diagnostics)
    {
        if charge.fract() == 0.0 {
            species = species.with_charge(charge as i32);
        } else {
            diagnostics.push(
                Diagnostic::error("a charge is a whole number of elementary charges")
                    .with_code("E0405")
                    .at(setting.value.span, format!("this is {charge}"))
                    .note("fractional charges are not chemistry"),
            );
        }
    }

    if let Some(setting) = get("molar_mass") {
        let expected = Dimension::MASS
            .try_div(Dimension::AMOUNT)
            .expect("mass per amount is representable");
        if let Some(mass) = evaluator.require(&setting.value, expected, "`molar_mass`", diagnostics)
        {
            species = species.with_molar_mass(mass);
        }
    }

    species
}

/// Turn a formula error into a diagnostic that says how to fix it.
fn formula_diagnostic(error: &FormulaError, span: Span) -> Diagnostic {
    let base = Diagnostic::error(format!("this formula cannot be read: {error}"))
        .with_code("E0211")
        .at(span, "invalid formula");
    match error {
        FormulaError::UnknownElement { .. } => base.help(
            "element symbols are capitalized, one or two letters — `Co` is cobalt, `CO` \
             is carbon monoxide",
        ),
        FormulaError::Unexpected { .. } => base.help(
            "parentheses and hydrates are not parsed; write the collected formula, or \
             leave `formula` out and give a `molar_mass` instead",
        ),
        _ => base,
    }
}

/// Warn when a species' declared molar mass disagrees with its formula.
///
/// The two numbers are close enough that neither looks wrong alone, and a model that
/// writes both has a typo in one of them.
pub fn check_molar_mass(species: &Species, span: Span, diagnostics: &mut Diagnostics) {
    let Some(discrepancy) = species.molar_mass_discrepancy() else { return };
    if discrepancy <= 1e-3 {
        return;
    }
    let implied = species.composition.molar_mass().unwrap_or(0.0);
    diagnostics.push(
        Diagnostic::warning(format!(
            "`{}` declares a molar mass that its formula does not support",
            species.name
        ))
        .with_code("W0304")
        .at(span, format!("declared {:.6} kg/mol", species.molar_mass))
        .note(format!(
            "{} implies {implied:.6} kg/mol, which is {:.1}% away",
            species.composition.formula(),
            discrepancy * 100.0
        ))
        .help("remove one of them; the formula alone is enough"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_syntax::SourceFile;
    use lattice_units::UnitRegistry;

    fn parse_expr(text: &str) -> (SourceFile, Expr) {
        let source = format!("project p {{ reaction r {{ reactants: {text}; }} }}");
        let file = SourceFile::new("t.lattice", &source);
        let (project, diagnostics) = lattice_syntax::parse(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        let decl = project.unwrap().declarations_of("reaction").next().unwrap().clone();
        let expr = decl.setting("reactants").unwrap().value.clone();
        (file, expr)
    }

    fn terms_of(text: &str) -> Vec<(String, f64)> {
        let (file, expr) = parse_expr(text);
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let mut diagnostics = Diagnostics::new();
        let terms = side(&expr, &evaluator, &mut diagnostics).expect("should parse");
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        terms.into_iter().map(|t| (t.name, t.coefficient)).collect()
    }

    /// Stoichiometry needed no new syntax: `+` is already an operator and `2 A` is
    /// already a juxtaposition, which binds tighter than `+`.
    #[test]
    fn stoichiometry_reuses_the_expression_grammar() {
        assert_eq!(terms_of("A"), [("A".to_string(), 1.0)]);
        assert_eq!(terms_of("A + B"), [("A".to_string(), 1.0), ("B".to_string(), 1.0)]);
        assert_eq!(
            terms_of("2 H2 + O2"),
            [("H2".to_string(), 2.0), ("O2".to_string(), 1.0)]
        );
        // Three terms, and a fractional coefficient, which is ordinary chemistry.
        assert_eq!(
            terms_of("H2 + 0.5 O2 + catalyst"),
            [
                ("H2".to_string(), 1.0),
                ("O2".to_string(), 0.5),
                ("catalyst".to_string(), 1.0)
            ]
        );
    }

    /// A rate constant's dimension depends on the order, and getting it wrong is
    /// invisible in the number.
    #[test]
    fn rate_constant_dimensions_follow_the_order() {
        // First order: 1/s.
        assert_eq!(rate_dimension(1.0), Some(Dimension::FREQUENCY));
        // Second order: m²/(mol·s).
        let second = rate_dimension(2.0).unwrap();
        let expected = Dimension::AREA
            .try_div(Dimension::AMOUNT)
            .unwrap()
            .try_div(Dimension::TIME)
            .unwrap();
        assert_eq!(second, expected);
        assert_ne!(second, Dimension::FREQUENCY, "which is the whole point of checking");

        // A fractional order has a fractional dimension the representation cannot hold,
        // and refusing to check beats checking against the wrong thing.
        assert_eq!(rate_dimension(1.5), None);
    }

    #[test]
    fn a_malformed_side_is_rejected_with_a_reason() {
        let (file, expr) = parse_expr("2 + 3");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&file, &units);
        let mut diagnostics = Diagnostics::new();
        assert!(side(&expr, &evaluator, &mut diagnostics).is_none());
        assert!(diagnostics.has_errors());
        let text = diagnostics.render(&file);
        assert!(text.contains("a sum of species"), "{text}");
    }
}
