//! The built-in vocabulary: field initializers, boundary conditions, integrator
//! methods, and force laws.
//!
//! These are recognized *structurally* — by matching a call form in the AST — rather
//! than evaluated, because their arguments are not all quantities. `fixed(273 kelvin)`
//! is a boundary condition, not a number, and `gaussian(center=[…], sigma=…)` is a
//! function of position.
//!
//! The set is closed and small. Spec §8.3 describes a real expression language that
//! compiles user-written force and rate laws to CPU and GPU kernels; that is milestone
//! M6. Until then, an unrecognized function is an error that *lists what is
//! available*, which is more useful than a generic "unknown function" and honest about
//! the boundary of what the engine can do.
//!
//! # Dimensions are checked against context
//!
//! Every value argument is checked against the dimension the surrounding declaration
//! implies. `field temperature on chamber = uniform(5 second);` fails because a
//! temperature field's initializer must produce kelvin — the initializer does not have
//! an intrinsic dimension of its own.

use lattice_domain_particle::Truncation;
use lattice_ir::{Boundary, Grid2d};
use lattice_syntax::{Argument, Diagnostic, Diagnostics, Expr, ExprKind, Span};
use lattice_units::Dimension;

use crate::eval::Evaluator;

/// A field initializer: a function of position within a grid.
#[derive(Clone, PartialEq, Debug)]
pub enum Initializer {
    /// The same value everywhere.
    Uniform(f64),
    /// A value on one side of the domain, zero on the other.
    Half {
        /// Which half is filled.
        side: HalfSide,
        /// The filled value.
        value: f64,
    },
    /// A Gaussian bump with a given peak value.
    Gaussian {
        /// Centre, m.
        center: [f64; 2],
        /// Standard deviation, m.
        sigma: f64,
        /// Peak value, in the field's units.
        peak: f64,
    },
    /// A filled disc.
    Disc {
        /// Centre, m.
        center: [f64; 2],
        /// Radius, m.
        radius: f64,
        /// Value inside.
        value: f64,
    },
    /// A linear ramp across the domain.
    Ramp {
        /// Which axis the ramp runs along.
        axis: Axis,
        /// Value at the low edge.
        from: f64,
        /// Value at the high edge.
        to: f64,
    },
}

/// Which half of the domain an initializer fills.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HalfSide {
    /// Low x.
    Left,
    /// High x.
    Right,
    /// Low y.
    Bottom,
    /// High y.
    Top,
}

/// A coordinate axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    /// The x axis.
    X,
    /// The y axis.
    Y,
}

impl Initializer {
    /// The value at a position, in the field's SI units.
    pub fn sample(&self, position: [f64; 2], grid: &Grid2d) -> f64 {
        let [x, y] = position;
        let origin = grid.origin();
        let extent = grid.extent();
        let midpoint = [origin[0] + 0.5 * extent[0], origin[1] + 0.5 * extent[1]];

        match *self {
            Initializer::Uniform(value) => value,

            Initializer::Half { side, value } => {
                let filled = match side {
                    HalfSide::Left => x < midpoint[0],
                    HalfSide::Right => x >= midpoint[0],
                    HalfSide::Bottom => y < midpoint[1],
                    HalfSide::Top => y >= midpoint[1],
                };
                if filled { value } else { 0.0 }
            }

            Initializer::Gaussian { center, sigma, peak } => {
                let r2 = (x - center[0]).powi(2) + (y - center[1]).powi(2);
                peak * (-r2 / (2.0 * sigma * sigma)).exp()
            }

            Initializer::Disc { center, radius, value } => {
                let r2 = (x - center[0]).powi(2) + (y - center[1]).powi(2);
                if r2 <= radius * radius { value } else { 0.0 }
            }

            Initializer::Ramp { axis, from, to } => {
                let (coordinate, low, span) = match axis {
                    Axis::X => (x, origin[0], extent[0]),
                    Axis::Y => (y, origin[1], extent[1]),
                };
                let t = if span > 0.0 { ((coordinate - low) / span).clamp(0.0, 1.0) } else { 0.0 };
                from + (to - from) * t
            }
        }
    }

    /// A one-line description for the model report.
    pub fn describe(&self) -> String {
        match self {
            Initializer::Uniform(v) => format!("uniform {v:.6}"),
            Initializer::Half { side, value } => format!("{side:?} half at {value:.6}"),
            Initializer::Gaussian { center, sigma, peak } => {
                format!("gaussian peak {peak:.6} at {center:?}, sigma {sigma:.4} m")
            }
            Initializer::Disc { center, radius, value } => {
                format!("disc {value:.6} at {center:?}, radius {radius:.4} m")
            }
            Initializer::Ramp { axis, from, to } => {
                format!("{axis:?} ramp {from:.6} -> {to:.6}")
            }
        }
    }
}

/// Every initializer name, for the "did you mean" list.
const INITIALIZERS: &[&str] =
    &["uniform", "left_half", "right_half", "bottom_half", "top_half", "gaussian", "disc", "ramp_x", "ramp_y"];

/// Every boundary name.
const BOUNDARIES: &[&str] = &["periodic", "insulated", "fixed", "flux", "robin"];

/// A helper over a call's arguments.
pub(crate) struct Call<'a> {
    pub(crate) name: &'a str,
    pub(crate) span: Span,
    pub(crate) arguments: &'a [Argument],
}

impl<'a> Call<'a> {
    /// Match `name(args)` or a bare `name`, which is treated as a call with none.
    pub(crate) fn match_expr(expr: &'a Expr) -> Option<Call<'a>> {
        match &expr.kind {
            ExprKind::Call(callee, arguments) => {
                let name = callee.as_name()?;
                Some(Call { name, span: expr.span, arguments })
            }
            ExprKind::Name(name) => Some(Call { name, span: expr.span, arguments: &[] }),
            _ => None,
        }
    }

    pub(crate) fn named(&self, key: &str) -> Option<&'a Expr> {
        self.arguments
            .iter()
            .find(|a| a.name.as_ref().is_some_and(|n| n.text == key))
            .map(|a| &a.value)
    }

    pub(crate) fn positional(&self, index: usize) -> Option<&'a Expr> {
        self.arguments.iter().filter(|a| a.name.is_none()).nth(index).map(|a| &a.value)
    }

    /// A named argument, falling back to a positional one at `index`.
    pub(crate) fn get(&self, key: &str, index: usize) -> Option<&'a Expr> {
        self.named(key).or_else(|| self.positional(index))
    }

    pub(crate) fn require(&self, key: &str, index: usize, diagnostics: &mut Diagnostics) -> Option<&'a Expr> {
        match self.get(key, index) {
            Some(expr) => Some(expr),
            None => {
                diagnostics.push(
                    Diagnostic::error(format!("`{}` needs a `{key}` argument", self.name))
                        .with_code("E0203")
                        .at(self.span, format!("missing `{key}`")),
                );
                None
            }
        }
    }

    /// Report any argument name the callee does not understand.
    ///
    /// A silently ignored `sigmaa=0.1` produces a model that compiles, runs, and is
    /// wrong — the worst of the three outcomes.
    pub(crate) fn reject_unknown(&self, allowed: &[&str], diagnostics: &mut Diagnostics) {
        for argument in self.arguments {
            let Some(name) = &argument.name else { continue };
            if allowed.contains(&name.text.as_str()) {
                continue;
            }
            diagnostics.push(
                Diagnostic::error(format!("`{}` has no argument called `{}`", self.name, name.text))
                    .with_code("E0204")
                    .at(name.span, "unknown argument")
                    .help(format!("`{}` accepts: {}", self.name, allowed.join(", "))),
            );
        }
    }
}

/// Report an unrecognized builtin, listing what is available.
pub(crate) fn unknown(
    what: &str,
    name: &str,
    span: Span,
    available: &[&str],
    diagnostics: &mut Diagnostics,
) {
    let suggestion = closest(name, available);
    let mut diagnostic = Diagnostic::error(format!("`{name}` is not a known {what}"))
        .with_code("E0210")
        .at(span, format!("unknown {what}"))
        .note(format!("available: {}", available.join(", ")));
    if let Some(suggestion) = suggestion {
        diagnostic = diagnostic.help(format!("did you mean `{suggestion}`?"));
    }
    diagnostics.push(diagnostic);
}

/// The closest name by edit distance, if one is close enough to be worth suggesting.
fn closest<'a>(name: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let budget = if name.len() <= 4 { 1 } else { 2 };
    candidates
        .iter()
        .map(|candidate| (edit_distance(name, candidate), *candidate))
        .filter(|(distance, _)| *distance <= budget)
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, candidate)| candidate)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        current[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            current[j] = (previous[j] + 1).min(current[j - 1] + 1).min(previous[j - 1] + cost);
        }
        core::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// Parse a field initializer, checking every value against `dimension`.
pub fn initializer(
    expr: &Expr,
    dimension: Dimension,
    what: &str,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Initializer> {
    let Some(call) = Call::match_expr(expr) else {
        // Not a call form, so it must be a plain quantity meaning "uniform".
        let value = evaluator.require(expr, dimension, what, diagnostics)?;
        return Some(Initializer::Uniform(value));
    };

    // A bare name that is a unit is a quantity, not an initializer: `= 1 kelvin`
    // parses as a product whose right operand is a Name.
    if call.arguments.is_empty() && !INITIALIZERS.contains(&call.name) {
        let value = evaluator.require(expr, dimension, what, diagnostics)?;
        return Some(Initializer::Uniform(value));
    }

    let value_of = |key: &str, index: usize, diagnostics: &mut Diagnostics| {
        let argument = call.require(key, index, diagnostics)?;
        evaluator.require(argument, dimension, &format!("`{}` {key}", call.name), diagnostics)
    };
    let length_of = |key: &str, index: usize, diagnostics: &mut Diagnostics| {
        let argument = call.require(key, index, diagnostics)?;
        evaluator.require(argument, Dimension::LENGTH, &format!("`{}` {key}", call.name), diagnostics)
    };
    let point_of = |key: &str, index: usize, diagnostics: &mut Diagnostics| {
        let argument = call.require(key, index, diagnostics)?;
        evaluator.pair(argument, Dimension::LENGTH, &format!("`{}` {key}", call.name), diagnostics)
    };

    match call.name {
        "uniform" => {
            call.reject_unknown(&["value"], diagnostics);
            Some(Initializer::Uniform(value_of("value", 0, diagnostics)?))
        }
        "left_half" | "right_half" | "bottom_half" | "top_half" => {
            call.reject_unknown(&["value"], diagnostics);
            let side = match call.name {
                "left_half" => HalfSide::Left,
                "right_half" => HalfSide::Right,
                "bottom_half" => HalfSide::Bottom,
                _ => HalfSide::Top,
            };
            Some(Initializer::Half { side, value: value_of("value", 0, diagnostics)? })
        }
        "gaussian" => {
            call.reject_unknown(&["center", "sigma", "peak"], diagnostics);
            let center = point_of("center", 0, diagnostics);
            let sigma = length_of("sigma", 1, diagnostics);
            let peak = value_of("peak", 2, diagnostics);
            let (center, sigma, peak) = (center?, sigma?, peak?);
            if sigma <= 0.0 {
                diagnostics.push(
                    Diagnostic::error("`gaussian` needs a positive sigma")
                        .with_code("E0405")
                        .at(call.span, format!("sigma is {sigma}")),
                );
                return None;
            }
            Some(Initializer::Gaussian { center, sigma, peak })
        }
        "disc" => {
            call.reject_unknown(&["center", "radius", "value"], diagnostics);
            let center = point_of("center", 0, diagnostics);
            let radius = length_of("radius", 1, diagnostics);
            let value = value_of("value", 2, diagnostics);
            Some(Initializer::Disc { center: center?, radius: radius?, value: value? })
        }
        "ramp_x" | "ramp_y" => {
            call.reject_unknown(&["from", "to"], diagnostics);
            let axis = if call.name == "ramp_x" { Axis::X } else { Axis::Y };
            let from = value_of("from", 0, diagnostics);
            let to = value_of("to", 1, diagnostics);
            Some(Initializer::Ramp { axis, from: from?, to: to? })
        }
        other => {
            unknown("field initializer", other, call.span, INITIALIZERS, diagnostics);
            None
        }
    }
}

/// Infer the dimension a field initializer produces, reporting nothing.
///
/// This is what lets `field temperature on chamber = 298 kelvin;` become a
/// temperature field without the user declaring a dimension separately. The value
/// argument is evaluated quietly; the real [`initializer`] call that follows
/// re-evaluates it against the inferred dimension and reports anything wrong, so no
/// diagnostic is lost by discarding them here.
pub fn infer_initializer_dimension(expr: &Expr, evaluator: &Evaluator<'_>) -> Option<Dimension> {
    let mut quiet = Diagnostics::new();

    let Some(call) = Call::match_expr(expr) else {
        return evaluator.quantity(expr, &mut quiet).map(|q| q.dimension());
    };
    // A bare name that is not an initializer is a unit, so the whole expression is
    // the quantity: `= 298 kelvin` parses as a product ending in a Name.
    if call.arguments.is_empty() && !INITIALIZERS.contains(&call.name) {
        return evaluator.quantity(expr, &mut quiet).map(|q| q.dimension());
    }

    // Whichever argument carries the field's own units.
    let value = match call.name {
        "uniform" | "left_half" | "right_half" | "bottom_half" | "top_half" => call.get("value", 0),
        "gaussian" => call.get("peak", 2),
        "disc" => call.get("value", 2),
        "ramp_x" | "ramp_y" => call.get("from", 0),
        _ => None,
    }?;
    evaluator.quantity(value, &mut quiet).map(|q| q.dimension())
}

/// Parse a boundary condition for a field of the given dimension.
pub fn boundary(
    expr: &Expr,
    dimension: Dimension,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Boundary> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected a boundary condition")
                .with_code("E0210")
                .at(expr.span, "not a boundary condition")
                .note(format!("available: {}", BOUNDARIES.join(", "))),
        );
        return None;
    };

    match call.name {
        "periodic" => {
            call.reject_unknown(&[], diagnostics);
            Some(Boundary::Periodic)
        }
        "insulated" => {
            call.reject_unknown(&[], diagnostics);
            Some(Boundary::INSULATED)
        }
        "fixed" | "dirichlet" => {
            call.reject_unknown(&["value"], diagnostics);
            let argument = call.require("value", 0, diagnostics)?;
            let value = evaluator.require(argument, dimension, "the boundary value", diagnostics)?;
            Some(Boundary::Dirichlet { value })
        }
        "flux" | "neumann" => {
            call.reject_unknown(&["gradient"], diagnostics);
            let argument = call.require("gradient", 0, diagnostics)?;
            // An outward derivative is field units per metre.
            let expected = dimension.try_div(Dimension::LENGTH).ok()?;
            let gradient =
                evaluator.require(argument, expected, "the boundary gradient", diagnostics)?;
            Some(Boundary::Neumann { gradient })
        }
        "robin" => {
            call.reject_unknown(&["coefficient", "ambient"], diagnostics);
            let coefficient_expr = call.require("coefficient", 0, diagnostics);
            let ambient_expr = call.require("ambient", 1, diagnostics);
            // The exchange coefficient is h/k, which is 1/length.
            let inverse_length = Dimension::LENGTH.try_recip().ok()?;
            let coefficient = evaluator.require(
                coefficient_expr?,
                inverse_length,
                "the Robin coefficient",
                diagnostics,
            );
            let ambient =
                evaluator.require(ambient_expr?, dimension, "the ambient value", diagnostics);
            Some(Boundary::Robin { coefficient: coefficient?, ambient: ambient? })
        }
        other => {
            unknown("boundary condition", other, call.span, BOUNDARIES, diagnostics);
            None
        }
    }
}

/// A time-integration method named by a `solve … with <method>(…)` clause.
#[derive(Clone, PartialEq, Debug)]
pub struct Method {
    /// The method name as written.
    pub name: String,
    /// The timestep it was given, seconds.
    pub timestep: Option<f64>,
    /// Where it appeared.
    pub span: Span,
}

/// Parse the method and parameters of a `solve` statement.
pub fn method(
    name: &lattice_syntax::Ident,
    parameters: &[Argument],
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Method {
    let call = Call { name: &name.text, span: name.span, arguments: parameters };
    call.reject_unknown(&["dt", "tolerance", "max_iterations"], diagnostics);

    let timestep = call
        .get("dt", 0)
        .and_then(|expr| evaluator.require(expr, Dimension::TIME, "the timestep `dt`", diagnostics));

    if let Some(dt) = timestep
        && dt <= 0.0
    {
        diagnostics.push(
            Diagnostic::error("the timestep must be positive")
                .with_code("E0405")
                .at(name.span, format!("dt is {dt} s")),
        );
    }

    Method { name: name.text.clone(), timestep, span: name.span }
}

/// A force law declared in a particle domain.
#[derive(Clone, PartialEq, Debug)]
pub enum ForceSpec {
    /// Uniform acceleration, m/s².
    Gravity {
        /// The acceleration vector.
        acceleration: [f64; 2],
    },
    /// Linear drag with coefficient in kg/s.
    Drag {
        /// The drag coefficient.
        coefficient: f64,
    },
    /// An isotropic harmonic well.
    HarmonicWell {
        /// Centre, m.
        center: [f64; 2],
        /// Stiffness, N/m.
        stiffness: f64,
    },
    /// Truncated Lennard-Jones.
    LennardJones {
        /// Well depth, J.
        epsilon: f64,
        /// Length scale, m.
        sigma: f64,
        /// Cutoff, m.
        cutoff: f64,
        /// How the potential is brought to zero at the cutoff.
        truncation: Truncation,
    },
    /// A soft repulsive disc.
    SoftRepulsion {
        /// Stiffness, N/m.
        stiffness: f64,
        /// Range, m.
        range: f64,
    },
}

/// Every force name.
const FORCES: &[&str] = &["gravity", "drag", "harmonic_well", "lennard_jones", "soft_repulsion"];

impl ForceSpec {
    /// A one-line description for the model report.
    pub fn describe(&self) -> String {
        match self {
            ForceSpec::Gravity { acceleration } => {
                format!("gravity {acceleration:?} m/s^2")
            }
            ForceSpec::Drag { coefficient } => format!("linear drag c = {coefficient} kg/s"),
            ForceSpec::HarmonicWell { center, stiffness } => {
                format!("harmonic well k = {stiffness} N/m at {center:?}")
            }
            ForceSpec::LennardJones { epsilon, sigma, cutoff, truncation } => {
                format!(
                    "Lennard-Jones eps = {epsilon:.4e} J, sigma = {sigma:.4e} m, cutoff = {cutoff:.4e} m ({})",
                    truncation.name()
                )
            }
            ForceSpec::SoftRepulsion { stiffness, range } => {
                format!("soft repulsion k = {stiffness:.4e} N/m inside {range:.4e} m")
            }
        }
    }

    /// True for a pair law, which needs a region to bin particles into.
    pub fn needs_region(&self) -> bool {
        self.cutoff().is_some()
    }

    /// The interaction cutoff of a pair law, m.
    pub fn cutoff(&self) -> Option<f64> {
        match self {
            ForceSpec::LennardJones { cutoff, .. } => Some(*cutoff),
            ForceSpec::SoftRepulsion { range, .. } => Some(*range),
            _ => None,
        }
    }
}

/// Parse one force law.
pub fn force(
    expr: &Expr,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<ForceSpec> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected a force law")
                .with_code("E0210")
                .at(expr.span, "not a force law")
                .note(format!("available: {}", FORCES.join(", "))),
        );
        return None;
    };

    match call.name {
        "gravity" => {
            call.reject_unknown(&["acceleration", "g"], diagnostics);
            match call.get("acceleration", 0).or_else(|| call.named("g")) {
                Some(argument) => {
                    // Accept either a vector or a downward magnitude.
                    if matches!(argument.kind, ExprKind::List(_) | ExprKind::Tuple(_)) {
                        let acceleration = evaluator.pair(
                            argument,
                            Dimension::ACCELERATION,
                            "the gravity vector",
                            diagnostics,
                        )?;
                        Some(ForceSpec::Gravity { acceleration })
                    } else {
                        let magnitude = evaluator.require(
                            argument,
                            Dimension::ACCELERATION,
                            "the gravitational acceleration",
                            diagnostics,
                        )?;
                        Some(ForceSpec::Gravity { acceleration: [0.0, -magnitude] })
                    }
                }
                None => Some(ForceSpec::Gravity {
                    acceleration: [0.0, -lattice_units::constants::value::STANDARD_GRAVITY],
                }),
            }
        }

        "drag" => {
            call.reject_unknown(&["coefficient"], diagnostics);
            let argument = call.require("coefficient", 0, diagnostics)?;
            // Drag coefficient c in F = -c v has units of mass per time.
            let expected = Dimension::MASS.try_div(Dimension::TIME).ok()?;
            let coefficient =
                evaluator.require(argument, expected, "the drag coefficient", diagnostics)?;
            Some(ForceSpec::Drag { coefficient })
        }

        "harmonic_well" => {
            call.reject_unknown(&["center", "stiffness"], diagnostics);
            let center = match call.get("center", 0) {
                Some(argument) => {
                    evaluator.pair(argument, Dimension::LENGTH, "the well centre", diagnostics)?
                }
                None => [0.0, 0.0],
            };
            let stiffness_expr = call.require("stiffness", 1, diagnostics)?;
            let stiffness = evaluator.require(
                stiffness_expr,
                Dimension::STIFFNESS,
                "the well stiffness",
                diagnostics,
            )?;
            Some(ForceSpec::HarmonicWell { center, stiffness })
        }

        "lennard_jones" => {
            call.reject_unknown(&["epsilon", "sigma", "cutoff", "truncation"], diagnostics);
            let epsilon_expr = call.require("epsilon", 0, diagnostics);
            let sigma_expr = call.require("sigma", 1, diagnostics);
            let epsilon =
                evaluator.require(epsilon_expr?, Dimension::ENERGY, "the LJ well depth", diagnostics);
            let sigma =
                evaluator.require(sigma_expr?, Dimension::LENGTH, "the LJ length scale", diagnostics);
            let (epsilon, sigma) = (epsilon?, sigma?);
            let cutoff = match call.get("cutoff", 2) {
                Some(argument) => {
                    evaluator.require(argument, Dimension::LENGTH, "the LJ cutoff", diagnostics)?
                }
                // The conventional default.
                None => 2.5 * sigma,
            };
            if epsilon <= 0.0 || sigma <= 0.0 || cutoff <= 0.0 {
                diagnostics.push(
                    Diagnostic::error("Lennard-Jones needs positive epsilon, sigma and cutoff")
                        .with_code("E0405")
                        .at(call.span, format!("eps = {epsilon}, sigma = {sigma}, cutoff = {cutoff}")),
                );
                return None;
            }
            // The two truncations are different potentials; the conventional energy
            // shift is the default and the one the GPU kernel implements.
            let truncation = match call.named("truncation") {
                None => Truncation::EnergyShift,
                Some(argument) => match argument.as_name() {
                    Some("energy_shift") => Truncation::EnergyShift,
                    Some("force_shift") => Truncation::ForceShift,
                    _ => {
                        diagnostics.push(
                            Diagnostic::error("`truncation` must be `energy_shift` or `force_shift`")
                                .with_code("E0208")
                                .at(argument.span, "unknown keyword value"),
                        );
                        return None;
                    }
                },
            };
            Some(ForceSpec::LennardJones { epsilon, sigma, cutoff, truncation })
        }

        "soft_repulsion" => {
            call.reject_unknown(&["stiffness", "range"], diagnostics);
            let stiffness_expr = call.require("stiffness", 0, diagnostics);
            let range_expr = call.require("range", 1, diagnostics);
            let stiffness = evaluator.require(
                stiffness_expr?,
                Dimension::STIFFNESS,
                "the repulsion stiffness",
                diagnostics,
            );
            let range = evaluator.require(range_expr?, Dimension::LENGTH, "the repulsion range", diagnostics);
            let (stiffness, range) = (stiffness?, range?);
            if stiffness <= 0.0 || range <= 0.0 {
                diagnostics.push(
                    Diagnostic::error("soft repulsion needs a positive stiffness and range")
                        .with_code("E0405")
                        .at(call.span, format!("k = {stiffness}, range = {range}")),
                );
                return None;
            }
            Some(ForceSpec::SoftRepulsion { stiffness, range })
        }

        other => {
            unknown("force law", other, call.span, FORCES, diagnostics);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_syntax::{parse, Item, SourceFile};
    use lattice_units::UnitRegistry;

    struct Harness {
        file: SourceFile,
        expr: Expr,
        diagnostics: Diagnostics,
    }

    fn harness(source: &str) -> Harness {
        let file = SourceFile::new("t.lattice", format!("project p {{ x: {source}; }}"));
        let (project, diagnostics) = parse(&file);
        let project = project.expect("should parse");
        let Item::Setting(setting) = &project.items[0] else { panic!("expected a setting") };
        Harness { expr: setting.value.clone(), file, diagnostics }
    }

    fn init(source: &str, dimension: Dimension) -> Initializer {
        let mut h = harness(source);
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        let result = initializer(&h.expr, dimension, "the initial value", &evaluator, &mut h.diagnostics);
        assert!(!h.diagnostics.has_errors(), "{}", h.diagnostics.render(&h.file));
        result.expect("should parse an initializer")
    }

    fn init_err(source: &str, dimension: Dimension) -> (String, Vec<String>) {
        let mut h = harness(source);
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        initializer(&h.expr, dimension, "the initial value", &evaluator, &mut h.diagnostics);
        assert!(h.diagnostics.has_errors(), "expected an error from `{source}`");
        (h.diagnostics.render(&h.file), h.diagnostics.codes().into_iter().map(String::from).collect())
    }

    fn grid() -> Grid2d {
        Grid2d::new(10, 10, [1.0, 1.0])
    }

    #[test]
    fn a_bare_quantity_is_a_uniform_initializer() {
        assert_eq!(init("298 kelvin", Dimension::TEMPERATURE), Initializer::Uniform(298.0));
        // An explicit `uniform(...)` means the same thing.
        assert_eq!(init("uniform(298 kelvin)", Dimension::TEMPERATURE), Initializer::Uniform(298.0));
    }

    /// From spec §25.1: `species A on chamber = left_half(1 mole / meter^2);`
    #[test]
    fn the_spec_half_initializers_work() {
        let left = init("left_half(1 mole / meter^2)", Dimension::AREAL_CONCENTRATION);
        assert_eq!(left, Initializer::Half { side: HalfSide::Left, value: 1.0 });

        let g = grid();
        assert_eq!(left.sample([0.25, 0.5], &g), 1.0, "the left half is filled");
        assert_eq!(left.sample([0.75, 0.5], &g), 0.0, "the right half is not");

        let right = init("right_half(2 mole / meter^2)", Dimension::AREAL_CONCENTRATION);
        assert_eq!(right.sample([0.75, 0.5], &g), 2.0);
        assert_eq!(right.sample([0.25, 0.5], &g), 0.0);
    }

    #[test]
    fn vertical_halves_split_on_y() {
        let g = grid();
        let bottom = init("bottom_half(5 kelvin)", Dimension::TEMPERATURE);
        assert_eq!(bottom.sample([0.5, 0.25], &g), 5.0);
        assert_eq!(bottom.sample([0.5, 0.75], &g), 0.0);
        let top = init("top_half(5 kelvin)", Dimension::TEMPERATURE);
        assert_eq!(top.sample([0.5, 0.75], &g), 5.0);
    }

    #[test]
    fn a_gaussian_peaks_at_its_centre_and_decays() {
        let gaussian =
            init("gaussian(center=[0.5 meter, 0.5 meter], sigma=0.1 meter, peak=100 kelvin)", Dimension::TEMPERATURE);
        let g = grid();
        assert!((gaussian.sample([0.5, 0.5], &g) - 100.0).abs() < 1e-12);
        // One sigma out is exp(-1/2) of the peak.
        let one_sigma = gaussian.sample([0.6, 0.5], &g);
        assert!((one_sigma - 100.0 * (-0.5f64).exp()).abs() < 1e-9, "{one_sigma}");
        assert!(gaussian.sample([0.0, 0.0], &g) < 1e-6, "the tail should be negligible");
    }

    #[test]
    fn positional_arguments_work_too() {
        let a = init("gaussian([0.5 meter, 0.5 meter], 0.1 meter, 100 kelvin)", Dimension::TEMPERATURE);
        let b = init("gaussian(center=[0.5 meter, 0.5 meter], sigma=0.1 meter, peak=100 kelvin)", Dimension::TEMPERATURE);
        assert_eq!(a, b);
    }

    #[test]
    fn a_disc_is_sharp_at_its_radius() {
        let disc = init("disc(center=[0.5 meter, 0.5 meter], radius=0.2 meter, value=7 kelvin)", Dimension::TEMPERATURE);
        let g = grid();
        assert_eq!(disc.sample([0.5, 0.5], &g), 7.0);
        assert_eq!(disc.sample([0.65, 0.5], &g), 7.0);
        assert_eq!(disc.sample([0.75, 0.5], &g), 0.0);
    }

    #[test]
    fn a_ramp_interpolates_across_the_domain() {
        let ramp = init("ramp_x(from=0 kelvin, to=100 kelvin)", Dimension::TEMPERATURE);
        let g = grid();
        assert!((ramp.sample([0.0, 0.5], &g) - 0.0).abs() < 1e-12);
        assert!((ramp.sample([0.5, 0.5], &g) - 50.0).abs() < 1e-12);
        assert!((ramp.sample([1.0, 0.5], &g) - 100.0).abs() < 1e-12);
    }

    /// An initializer has no intrinsic dimension: it must produce what the field
    /// declared. This is the FR-002 check applied to initial conditions.
    #[test]
    fn an_initializer_value_is_checked_against_the_field_dimension() {
        let (text, codes) = init_err("left_half(5 second)", Dimension::TEMPERATURE);
        assert!(codes.contains(&"E0400".to_string()));
        assert!(text.contains("expected temperature"), "{text}");
        assert!(text.contains("found time"), "{text}");
    }

    #[test]
    fn an_unknown_initializer_lists_what_exists() {
        let (text, codes) = init_err("triangle(1 kelvin)", Dimension::TEMPERATURE);
        assert!(codes.contains(&"E0210".to_string()));
        assert!(text.contains("available:"), "{text}");
        assert!(text.contains("gaussian"), "{text}");
    }

    #[test]
    fn a_near_miss_initializer_name_is_suggested() {
        let (text, _) = init_err("gausian(center=[0 meter,0 meter], sigma=1 meter, peak=1 kelvin)", Dimension::TEMPERATURE);
        assert!(text.contains("did you mean `gaussian`"), "{text}");
    }

    /// A mistyped argument name must be an error. Ignoring it would produce a model
    /// that compiles, runs, and is quietly wrong.
    #[test]
    fn a_mistyped_argument_name_is_rejected() {
        let (text, codes) = init_err(
            "gaussian(center=[0.5 meter,0.5 meter], sigmaa=0.1 meter, peak=1 kelvin)",
            Dimension::TEMPERATURE,
        );
        assert!(codes.contains(&"E0204".to_string()), "{text}");
        assert!(text.contains("no argument called `sigmaa`"), "{text}");
        assert!(text.contains("accepts: center, sigma, peak"), "{text}");
    }

    #[test]
    fn a_missing_required_argument_is_reported() {
        let (text, codes) = init_err("gaussian(center=[0.5 meter,0.5 meter])", Dimension::TEMPERATURE);
        assert!(codes.contains(&"E0203".to_string()), "{text}");
        assert!(text.contains("needs a `sigma` argument"), "{text}");
    }

    #[test]
    fn a_non_positive_sigma_is_rejected() {
        let (_, codes) = init_err(
            "gaussian(center=[0.5 meter,0.5 meter], sigma=0 meter, peak=1 kelvin)",
            Dimension::TEMPERATURE,
        );
        assert!(codes.contains(&"E0405".to_string()));
    }

    // --- boundaries ---------------------------------------------------------

    fn bound(source: &str, dimension: Dimension) -> Boundary {
        let mut h = harness(source);
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        let result = boundary(&h.expr, dimension, &evaluator, &mut h.diagnostics);
        assert!(!h.diagnostics.has_errors(), "{}", h.diagnostics.render(&h.file));
        result.expect("should parse a boundary")
    }

    #[test]
    fn simple_boundaries_parse_as_bare_names() {
        assert_eq!(bound("periodic", Dimension::TEMPERATURE), Boundary::Periodic);
        assert_eq!(bound("insulated", Dimension::TEMPERATURE), Boundary::INSULATED);
    }

    #[test]
    fn a_dirichlet_value_is_checked_against_the_field() {
        assert_eq!(
            bound("fixed(273.15 kelvin)", Dimension::TEMPERATURE),
            Boundary::Dirichlet { value: 273.15 }
        );

        let mut h = harness("fixed(5 second)");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        boundary(&h.expr, Dimension::TEMPERATURE, &evaluator, &mut h.diagnostics);
        assert!(h.diagnostics.has_errors(), "a time-valued temperature boundary must fail");
    }

    /// A Neumann gradient is field units *per metre* — a dimension the compiler
    /// derives rather than the user restating.
    #[test]
    fn a_neumann_gradient_carries_the_derived_dimension() {
        let b = bound("flux(gradient=10 kelvin / meter)", Dimension::TEMPERATURE);
        assert_eq!(b, Boundary::Neumann { gradient: 10.0 });

        let mut h = harness("flux(gradient=10 kelvin)");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        boundary(&h.expr, Dimension::TEMPERATURE, &evaluator, &mut h.diagnostics);
        assert!(h.diagnostics.has_errors(), "a gradient without a length is wrong");
    }

    #[test]
    fn a_robin_boundary_checks_both_arguments() {
        let b = bound("robin(coefficient=5 / meter, ambient=300 kelvin)", Dimension::TEMPERATURE);
        assert_eq!(b, Boundary::Robin { coefficient: 5.0, ambient: 300.0 });
    }

    #[test]
    fn an_unknown_boundary_lists_what_exists() {
        let mut h = harness("sticky");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        boundary(&h.expr, Dimension::TEMPERATURE, &evaluator, &mut h.diagnostics);
        let text = h.diagnostics.render(&h.file);
        assert!(text.contains("periodic"), "{text}");
        assert!(text.contains("insulated"), "{text}");
    }

    // --- forces -------------------------------------------------------------

    fn force_of(source: &str) -> ForceSpec {
        let mut h = harness(source);
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        let result = force(&h.expr, &evaluator, &mut h.diagnostics);
        assert!(!h.diagnostics.has_errors(), "{}", h.diagnostics.render(&h.file));
        result.expect("should parse a force")
    }

    #[test]
    fn gravity_defaults_to_earth_and_accepts_a_magnitude_or_vector() {
        assert_eq!(
            force_of("gravity"),
            ForceSpec::Gravity { acceleration: [0.0, -9.806_65] }
        );
        assert_eq!(
            force_of("gravity(1.62 meter/second^2)"),
            ForceSpec::Gravity { acceleration: [0.0, -1.62] }
        );
        assert_eq!(
            force_of("gravity([1 meter/second^2, -2 meter/second^2])"),
            ForceSpec::Gravity { acceleration: [1.0, -2.0] }
        );
    }

    #[test]
    fn lennard_jones_defaults_its_cutoff_to_two_and_a_half_sigma() {
        let ForceSpec::LennardJones { sigma, cutoff, truncation, .. } =
            force_of("lennard_jones(epsilon=1 joule, sigma=2 meter)")
        else {
            panic!("wrong variant")
        };
        assert_eq!(sigma, 2.0);
        assert_eq!(cutoff, 5.0);
        assert_eq!(truncation, Truncation::EnergyShift, "the conventional default");
    }

    #[test]
    fn lennard_jones_names_its_truncation() {
        let ForceSpec::LennardJones { truncation, .. } =
            force_of("lennard_jones(epsilon=1 joule, sigma=1 meter, truncation=force_shift)")
        else {
            panic!("wrong variant")
        };
        assert_eq!(truncation, Truncation::ForceShift);

        let mut h = harness("lennard_jones(epsilon=1 joule, sigma=1 meter, truncation=smooth)");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        assert!(force(&h.expr, &evaluator, &mut h.diagnostics).is_none());
        assert!(h.diagnostics.codes().contains(&"E0208"));
    }

    #[test]
    fn soft_repulsion_checks_its_dimensions() {
        assert_eq!(
            force_of("soft_repulsion(stiffness=10 newton/meter, range=0.5 meter)"),
            ForceSpec::SoftRepulsion { stiffness: 10.0, range: 0.5 }
        );
        assert!(ForceSpec::SoftRepulsion { stiffness: 1.0, range: 1.0 }.needs_region());
        assert!(!ForceSpec::Gravity { acceleration: [0.0, -1.0] }.needs_region());
        let mut h = harness("soft_repulsion(stiffness=10 newton/meter, range=0.5 second)");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        assert!(force(&h.expr, &evaluator, &mut h.diagnostics).is_none());
        assert!(h.diagnostics.codes().contains(&"E0400"));
    }

    #[test]
    fn a_harmonic_well_checks_its_stiffness_dimension() {
        let well = force_of("harmonic_well(center=[0 meter, 0 meter], stiffness=4 newton/meter)");
        assert_eq!(well, ForceSpec::HarmonicWell { center: [0.0, 0.0], stiffness: 4.0 });

        let mut h = harness("harmonic_well(stiffness=4 newton)");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        force(&h.expr, &evaluator, &mut h.diagnostics);
        assert!(h.diagnostics.has_errors(), "newtons are not a stiffness");
    }

    #[test]
    fn drag_requires_mass_per_time() {
        assert_eq!(
            force_of("drag(coefficient=0.5 kilogram/second)"),
            ForceSpec::Drag { coefficient: 0.5 }
        );
    }

    #[test]
    fn an_unknown_force_lists_what_exists() {
        let mut h = harness("magnetism(1 tesla)");
        let units = UnitRegistry::si();
        let evaluator = Evaluator::new(&h.file, &units);
        force(&h.expr, &evaluator, &mut h.diagnostics);
        let text = h.diagnostics.render(&h.file);
        assert!(text.contains("not a known force law"), "{text}");
        assert!(text.contains("lennard_jones"), "{text}");
    }

    #[test]
    fn edit_distance_backs_the_suggestions() {
        assert_eq!(edit_distance("gausian", "gaussian"), 1);
        assert_eq!(closest("gausian", INITIALIZERS), Some("gaussian"));
        assert_eq!(closest("completely_different", INITIALIZERS), None);
    }
}
