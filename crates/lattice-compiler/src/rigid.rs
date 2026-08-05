//! The rigid-body front end: `material`, `body` and `joint` declarations.
//!
//! # Why a body is one declaration rather than a count
//!
//! A particle set is written as `particles atoms { count: 256; … }` because its members
//! are interchangeable — the model means "256 of these", and naming them individually
//! would be noise. Rigid bodies are the opposite: a scene is *the ground*, *the ramp*,
//! *the crate*, each with its own shape, place and role, and half of them are referred
//! to by name from a joint. So each gets a declaration:
//!
//! ```text
//! material steel {
//!   density:     7850 kilogram / meter^2;
//!   restitution: 0.4;
//!   friction:    0.6;
//! }
//!
//! body ground {
//!   shape:    box(20 meter, 0.5 meter);
//!   at:       [0 meter, 0 meter];
//!   material: steel;
//!   motion:   static;
//! }
//! ```
//!
//! No grammar change was needed for any of it: `<kind> <name> { … }` already parses,
//! which is exactly what spec §8.4's generic declaration form was for.
//!
//! # Why density is per unit *area*
//!
//! `kilogram / meter^2`, not `kilogram / meter^3`. This is a 2D world. Giving it an
//! implied thickness would make every printed mass wrong by a factor nobody declared,
//! and §14.1 puts that kind of conversion in a port rather than hiding it in a solver.
//! The dimension check enforces it: a model that writes `7850 kilogram / meter^3` is
//! told what the engine expects and why.

use std::collections::BTreeMap;

use lattice_domain_rigid2d::{ConvexPolygon, Joint, Shape, ShapeError, Vec2};
use lattice_syntax::{Decl, Diagnostic, Diagnostics, Expr, ExprKind, Ident, Span};
use lattice_units::Dimension;

use crate::builtins::Call;
use crate::eval::Evaluator;

/// Settings a `material` declaration accepts.
pub const MATERIAL_SETTINGS: &[&str] = &["density", "restitution", "friction"];

/// Settings a `body` declaration accepts.
pub const BODY_SETTINGS: &[&str] =
    &["shape", "at", "angle", "material", "motion", "velocity", "spin", "mass"];

/// Settings a `joint` declaration accepts.
pub const JOINT_SETTINGS: &[&str] = &[
    "kind",
    "bodies",
    "anchor_a",
    "anchor_b",
    "length",
    "stiffness",
    "damping",
    "speed",
    "torque",
];

/// The shapes a model may name.
const SHAPES: &[&str] = &["circle", "box", "polygon", "segment"];

/// The joint kinds a model may name.
const JOINT_KINDS: &[&str] = &["distance", "rope", "pin", "spring", "motor"];

/// A material's bulk and surface properties.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Material {
    /// Areal density, kg/m².
    pub density: f64,
    /// Coefficient of restitution, 0 to 1.
    pub restitution: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Where it was declared.
    pub span: Span,
}

impl Default for Material {
    fn default() -> Self {
        // A dense, dead, moderately grippy default: something that falls, lands, and
        // stays put. A model that never declares a material still behaves sensibly.
        Material { density: 1000.0, restitution: 0.0, friction: 0.4, span: Span::new(0, 0) }
    }
}

/// A body as the model declared it, before it becomes a slot.
#[derive(Clone, PartialEq, Debug)]
pub struct BodyPlan {
    /// The declared name, used by joints.
    pub name: String,
    /// Its geometry.
    pub shape: Shape,
    /// Centre of mass, m.
    pub at: [f64; 2],
    /// Orientation, radians.
    pub angle: f64,
    /// Its surface and density.
    pub material: Material,
    /// True when no impulse may move it.
    pub is_static: bool,
    /// An explicit total mass, overriding the density.
    pub mass: Option<f64>,
    /// Initial linear velocity, m/s.
    pub velocity: [f64; 2],
    /// Initial angular velocity, rad/s.
    pub spin: f64,
    /// Where it was declared.
    pub span: Span,
}

/// A joint as the model declared it, naming bodies rather than slots.
#[derive(Clone, PartialEq, Debug)]
pub struct JointPlan {
    /// The declared name, for diagnostics.
    pub name: String,
    /// The two bodies it connects, by name.
    pub bodies: [String; 2],
    /// Which kind, already resolved except for the slot indices.
    pub build: JointShape,
    /// Where it was declared.
    pub span: Span,
}

/// A joint's kind and parameters, with slots still to be filled in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum JointShape {
    /// A rod or a rope.
    Distance {
        /// Anchor on the first body, local, m.
        anchor_a: Vec2,
        /// Anchor on the second body, local, m.
        anchor_b: Vec2,
        /// The length to hold, or `None` to take the current separation.
        length: Option<f64>,
        /// True for a rope, which pulls but never pushes.
        rope: bool,
    },
    /// A hinge.
    Pin {
        /// Anchor on the first body, local, m.
        anchor_a: Vec2,
        /// Anchor on the second body, local, m.
        anchor_b: Vec2,
    },
    /// A damped linear spring.
    Spring {
        /// Anchor on the first body, local, m.
        anchor_a: Vec2,
        /// Anchor on the second body, local, m.
        anchor_b: Vec2,
        /// Natural length, or `None` to take the current separation.
        length: Option<f64>,
        /// Stiffness, N/m.
        stiffness: f64,
        /// Damping, N·s/m.
        damping: f64,
    },
    /// A torque source.
    Motor {
        /// Target relative angular velocity, rad/s.
        speed: f64,
        /// Torque budget, N·m.
        torque: f64,
    },
}

impl JointPlan {
    /// Turn the plan into a solver joint, given the slots the bodies ended up in.
    ///
    /// `separation` is the current distance between the two anchors, used when the
    /// model did not state a length — "however far apart they are now" is what a reader
    /// means by leaving it out, and computing it here keeps that inference in one place.
    pub fn build(&self, a: usize, b: usize, separation: f64) -> Joint {
        match self.build {
            JointShape::Distance { anchor_a, anchor_b, length, rope } => Joint::Distance {
                a,
                b,
                local_a: anchor_a,
                local_b: anchor_b,
                rest_length: length.unwrap_or(separation),
                rope,
            },
            JointShape::Pin { anchor_a, anchor_b } => {
                Joint::Pin { a, b, local_a: anchor_a, local_b: anchor_b }
            }
            JointShape::Spring { anchor_a, anchor_b, length, stiffness, damping } => Joint::Spring {
                a,
                b,
                local_a: anchor_a,
                local_b: anchor_b,
                rest_length: length.unwrap_or(separation),
                stiffness,
                damping,
            },
            JointShape::Motor { speed, torque } => Joint::Motor {
                a,
                b,
                target_speed: speed,
                max_torque: torque,
            },
        }
    }

    /// A one-line description for the model report.
    pub fn describe(&self) -> String {
        let (from, to) = (&self.bodies[0], &self.bodies[1]);
        match self.build {
            JointShape::Distance { length, rope, .. } => format!(
                "{} `{from}`–`{to}` at {}",
                if rope { "rope" } else { "rod" },
                length.map_or_else(|| "their initial separation".to_string(), |l| format!("{l} m"))
            ),
            JointShape::Pin { .. } => format!("pin `{from}`–`{to}`"),
            JointShape::Spring { stiffness, damping, .. } => {
                format!("spring `{from}`–`{to}`, k = {stiffness} N/m, c = {damping} N.s/m")
            }
            JointShape::Motor { speed, torque } => {
                format!("motor `{from}`–`{to}` at {speed} rad/s, up to {torque} N.m")
            }
        }
    }
}

/// Read a `material` declaration.
pub fn material(
    decl: &Decl,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Material {
    let mut material = Material { span: decl.name.span, ..Material::default() };

    // Areal density: this is a 2D world, and the dimension check is where a model
    // written for a 3D engine finds out.
    if let Some(setting) = decl.setting("density") {
        if let Some(value) = evaluator.require(
            &setting.value,
            Dimension::AREAL_MASS_DENSITY,
            "`density`",
            diagnostics,
        ) {
            if value > 0.0 && value.is_finite() {
                material.density = value;
            } else {
                diagnostics.push(
                    Diagnostic::error("a density must be positive")
                        .with_code("E0405")
                        .at(setting.value.span, format!("this is {value}"))
                        .note("a body with no mass cannot respond to a force"),
                );
            }
        }
    }

    material.restitution =
        fraction(decl, "restitution", material.restitution, evaluator, diagnostics);
    if let Some(value) = decl
        .setting("friction")
        .and_then(|s| evaluator.dimensionless(&s.value, "`friction`", diagnostics))
    {
        if value >= 0.0 {
            material.friction = value;
        } else {
            diagnostics.push(
                Diagnostic::error("a friction coefficient cannot be negative")
                    .with_code("E0405")
                    .at(decl.setting("friction").unwrap().value.span, format!("this is {value}"))
                    .note("negative friction would accelerate a sliding body"),
            );
        }
    }
    material
}

/// Read a setting that must lie in `[0, 1]`.
fn fraction(
    decl: &Decl,
    key: &str,
    fallback: f64,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> f64 {
    let Some(setting) = decl.setting(key) else { return fallback };
    let Some(value) = evaluator.dimensionless(&setting.value, &format!("`{key}`"), diagnostics)
    else {
        return fallback;
    };
    if (0.0..=1.0).contains(&value) {
        return value;
    }
    diagnostics.push(
        Diagnostic::error(format!("`{key}` must be between 0 and 1"))
            .with_code("E0405")
            .at(setting.value.span, format!("this is {value}"))
            .note(
                "a coefficient of restitution above 1 would return more energy than the \
                 collision had, which is a typo rather than a material",
            ),
    );
    fallback.clamp(0.0, 1.0)
}

/// Read a `shape:` setting.
pub fn shape(
    expr: &Expr,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Shape> {
    let Some(call) = Call::match_expr(expr) else {
        diagnostics.push(
            Diagnostic::error("expected a shape")
                .with_code("E0210")
                .at(expr.span, "not a shape")
                .note(format!("available: {}", SHAPES.join(", ")))
                .help("for example `shape: box(1 meter, 0.5 meter);`"),
        );
        return None;
    };

    let built = match call.name {
        "circle" => {
            call.reject_unknown(&["radius"], diagnostics);
            let argument = call.require("radius", 0, diagnostics)?;
            let radius =
                evaluator.require(argument, Dimension::LENGTH, "the circle radius", diagnostics)?;
            Shape::circle(radius)
        }
        "box" => {
            call.reject_unknown(&["half_width", "half_height"], diagnostics);
            let width = call.require("half_width", 0, diagnostics)?;
            let height = call.require("half_height", 1, diagnostics)?;
            let half_width =
                evaluator.require(width, Dimension::LENGTH, "the box half-width", diagnostics)?;
            let half_height =
                evaluator.require(height, Dimension::LENGTH, "the box half-height", diagnostics)?;
            Shape::rectangle(half_width, half_height)
        }
        "segment" => {
            call.reject_unknown(&["half_length"], diagnostics);
            let argument = call.require("half_length", 0, diagnostics)?;
            let half_length = evaluator.require(
                argument,
                Dimension::LENGTH,
                "the segment half-length",
                diagnostics,
            )?;
            Shape::segment(half_length)
        }
        "polygon" => return polygon(&call, expr.span, evaluator, diagnostics),
        other => {
            diagnostics.push(
                Diagnostic::error(format!("`{other}` is not a shape"))
                    .with_code("E0210")
                    .at(call.span, "unknown shape")
                    .help(format!("available: {}", SHAPES.join(", "))),
            );
            return None;
        }
    };

    report_shape(built, expr.span, diagnostics)
}

/// `polygon([x, y], [x, y], …)` — a list of vertices, counter-clockwise.
fn polygon(
    call: &Call<'_>,
    span: Span,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<Shape> {
    let mut vertices = Vec::new();
    for argument in call.arguments {
        if let Some(name) = &argument.name {
            diagnostics.push(
                Diagnostic::error("a polygon's vertices are positional")
                    .with_code("E0210")
                    .at(name.span, "unexpected name")
                    .help("write `polygon([0 m, 0 m], [1 m, 0 m], [0 m, 1 m])`"),
            );
            return None;
        }
        vertices.push(evaluator.pair(
            &argument.value,
            Dimension::LENGTH,
            "a polygon vertex",
            diagnostics,
        )?);
    }
    report_shape(ConvexPolygon::new(&vertices).map(Shape::Polygon), span, diagnostics)
}

/// Turn a geometry error into a diagnostic that says how to fix it.
fn report_shape(
    result: Result<Shape, ShapeError>,
    span: Span,
    diagnostics: &mut Diagnostics,
) -> Option<Shape> {
    match result {
        Ok(shape) => Some(shape),
        Err(error) => {
            let diagnostic = Diagnostic::error(format!("this shape is not valid: {error}"))
                .with_code("E0211")
                .at(span, "invalid geometry");
            let diagnostic = match error {
                ShapeError::ClockwiseWinding => diagnostic.help(
                    "list the vertices counter-clockwise. If they came from a y-down \
                     coordinate system, check the sign of gravity too",
                ),
                ShapeError::NotConvex { at } => diagnostic
                    .note(format!("vertex {at} turns back on itself"))
                    .help("build a concave outline from several convex bodies"),
                ShapeError::TooManyVertices { limit, .. } => diagnostic.help(format!(
                    "a convex polygon may have at most {limit} vertices; use several bodies"
                )),
                _ => diagnostic,
            };
            diagnostics.push(diagnostic);
            None
        }
    }
}

/// Read a `body` declaration.
pub fn body(
    decl: &Decl,
    materials: &BTreeMap<String, Material>,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<BodyPlan> {
    let Some(setting) = decl.setting("shape") else {
        diagnostics.push(
            Diagnostic::error(format!("body `{}` needs a `shape`", decl.name.text))
                .with_code("E0203")
                .at(decl.name.span, "missing `shape`")
                .help("add `shape: box(1 meter, 0.5 meter);` inside the braces"),
        );
        return None;
    };
    let shape = shape(&setting.value, evaluator, diagnostics)?;

    let material = match decl.setting("material") {
        None => Material::default(),
        Some(setting) => {
            let Some(name) = evaluator.as_name(&setting.value) else {
                diagnostics.push(
                    Diagnostic::error("a `material` must name a declared material")
                        .with_code("E0401")
                        .at(setting.value.span, "not a name"),
                );
                return None;
            };
            match materials.get(name) {
                Some(material) => *material,
                None => {
                    let known: Vec<&str> = materials.keys().map(String::as_str).collect();
                    diagnostics.push(
                        Diagnostic::error(format!("there is no material called `{name}`"))
                            .with_code("E0202")
                            .at(setting.value.span, "unknown material")
                            .help(if known.is_empty() {
                                "declare one with `material steel { density: … }`".to_string()
                            } else {
                                format!("declared materials: {}", known.join(", "))
                            }),
                    );
                    return None;
                }
            }
        }
    };

    let at = decl
        .setting("at")
        .and_then(|s| evaluator.pair(&s.value, Dimension::LENGTH, "`at`", diagnostics))
        .unwrap_or([0.0, 0.0]);
    // Angle is dimensionless in SI, so `angle: 0.4;` and `angle: 0.4 radian;` are the
    // same number — which is the honest answer, not a gap in the checking.
    let angle = decl
        .setting("angle")
        .and_then(|s| evaluator.dimensionless(&s.value, "`angle`", diagnostics))
        .unwrap_or(0.0);
    let velocity = decl
        .setting("velocity")
        .and_then(|s| evaluator.pair(&s.value, Dimension::VELOCITY, "`velocity`", diagnostics))
        .unwrap_or([0.0, 0.0]);
    let spin = decl
        .setting("spin")
        .and_then(|s| {
            evaluator.require(&s.value, Dimension::ANGULAR_VELOCITY, "`spin`", diagnostics)
        })
        .unwrap_or(0.0);
    let mass = decl
        .setting("mass")
        .and_then(|s| evaluator.require(&s.value, Dimension::MASS, "`mass`", diagnostics));

    let is_static = match decl.setting("motion").map(|s| (&s.value, evaluator.as_name(&s.value))) {
        None => false,
        Some((_, Some("static"))) => true,
        Some((_, Some("dynamic"))) => false,
        Some((expr, _)) => {
            diagnostics.push(
                Diagnostic::error("`motion` must be `static` or `dynamic`")
                    .with_code("E0208")
                    .at(expr.span, "unknown motion")
                    .note(
                        "a static body has infinite mass: forces accumulate on it and never \
                         move it, which is how a wall is expressed",
                    ),
            );
            false
        }
    };

    // A segment encloses no area, so density gives it no mass. That is not a value to
    // paper over: a massless dynamic body is a division by zero waiting to happen.
    if !is_static && mass.is_none() && matches!(shape, Shape::Segment { .. }) {
        diagnostics.push(
            Diagnostic::error(format!("body `{}` would have no mass", decl.name.text))
                .with_code("E0405")
                .at(setting.value.span, "a segment encloses no area")
                .note("density times zero area is zero mass, and nothing can move a body with none")
                .help("add `motion: static;` for a wall, or give it an explicit `mass:`"),
        );
        return None;
    }

    Some(BodyPlan {
        name: decl.name.text.clone(),
        shape,
        at,
        angle,
        material,
        is_static,
        mass,
        velocity,
        spin,
        span: decl.name.span,
    })
}

/// Read a `joint` declaration.
pub fn joint(
    decl: &Decl,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<JointPlan> {
    let bodies = joint_bodies(decl, evaluator, diagnostics)?;

    let kind_setting = decl.setting("kind");
    let kind = match kind_setting.and_then(|s| evaluator.as_name(&s.value)) {
        Some(name) => name,
        None => {
            let span = kind_setting.map_or(decl.name.span, |s| s.value.span);
            diagnostics.push(
                Diagnostic::error(format!("joint `{}` needs a `kind`", decl.name.text))
                    .with_code("E0203")
                    .at(span, "missing or unreadable `kind`")
                    .help(format!("one of: {}", JOINT_KINDS.join(", "))),
            );
            return None;
        }
    };

    let anchor = |key: &str, diagnostics: &mut Diagnostics| -> Vec2 {
        decl.setting(key)
            .and_then(|s| evaluator.pair(&s.value, Dimension::LENGTH, &format!("`{key}`"), diagnostics))
            .map_or(Vec2::ZERO, Vec2::from)
    };
    let length = decl
        .setting("length")
        .and_then(|s| evaluator.require(&s.value, Dimension::LENGTH, "`length`", diagnostics));

    let build = match kind {
        "distance" | "rope" => JointShape::Distance {
            anchor_a: anchor("anchor_a", diagnostics),
            anchor_b: anchor("anchor_b", diagnostics),
            length,
            rope: kind == "rope",
        },
        "pin" => JointShape::Pin {
            anchor_a: anchor("anchor_a", diagnostics),
            anchor_b: anchor("anchor_b", diagnostics),
        },
        "spring" => {
            let Some(stiffness) = decl.setting("stiffness").and_then(|s| {
                evaluator.require(&s.value, Dimension::STIFFNESS, "`stiffness`", diagnostics)
            }) else {
                diagnostics.push(
                    Diagnostic::error(format!("spring `{}` needs a `stiffness`", decl.name.text))
                        .with_code("E0203")
                        .at(decl.name.span, "missing `stiffness`")
                        .help("add `stiffness: 500 newton / meter;`"),
                );
                return None;
            };
            let damping = decl
                .setting("damping")
                .and_then(|s| {
                    evaluator.require(&s.value, Dimension::DAMPING, "`damping`", diagnostics)
                })
                .unwrap_or(0.0);
            JointShape::Spring {
                anchor_a: anchor("anchor_a", diagnostics),
                anchor_b: anchor("anchor_b", diagnostics),
                length,
                stiffness,
                damping,
            }
        }
        "motor" => {
            let speed = decl
                .setting("speed")
                .and_then(|s| {
                    evaluator.require(&s.value, Dimension::ANGULAR_VELOCITY, "`speed`", diagnostics)
                })
                .unwrap_or(0.0);
            let Some(torque) = decl.setting("torque").and_then(|s| {
                evaluator.require(&s.value, Dimension::TORQUE, "`torque`", diagnostics)
            }) else {
                diagnostics.push(
                    Diagnostic::error(format!("motor `{}` needs a `torque` budget", decl.name.text))
                        .with_code("E0203")
                        .at(decl.name.span, "missing `torque`")
                        .note(
                            "a motor with an unbounded budget would move any load, which is a \
                             servo nobody has",
                        )
                        .help("add `torque: 50 newton meter;`"),
                );
                return None;
            };
            JointShape::Motor { speed, torque }
        }
        other => {
            diagnostics.push(
                Diagnostic::error(format!("`{other}` is not a joint kind"))
                    .with_code("E0208")
                    .at(kind_setting.unwrap().value.span, "unknown kind")
                    .help(format!("one of: {}", JOINT_KINDS.join(", "))),
            );
            return None;
        }
    };

    Some(JointPlan { name: decl.name.text.clone(), bodies, build, span: decl.name.span })
}

/// Read the `bodies: [a, b];` setting.
fn joint_bodies(
    decl: &Decl,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<[String; 2]> {
    let Some(setting) = decl.setting("bodies") else {
        diagnostics.push(
            Diagnostic::error(format!("joint `{}` needs two `bodies`", decl.name.text))
                .with_code("E0203")
                .at(decl.name.span, "missing `bodies`")
                .help("add `bodies: [ground, crate];`"),
        );
        return None;
    };
    let items = match &setting.value.kind {
        ExprKind::List(items) | ExprKind::Tuple(items) => items,
        _ => {
            diagnostics.push(
                Diagnostic::error("`bodies` must be a list of two names")
                    .with_code("E0401")
                    .at(setting.value.span, "not a list")
                    .help("write `bodies: [ground, crate];`"),
            );
            return None;
        }
    };
    if items.len() != 2 {
        diagnostics.push(
            Diagnostic::error(format!("a joint connects two bodies, not {}", items.len()))
                .with_code("E0401")
                .at(setting.value.span, format!("{} named", items.len()))
                .note("a joint between three bodies is two joints"),
        );
        return None;
    }
    let mut names = Vec::new();
    for item in items {
        let Some(name) = evaluator.as_name(item) else {
            diagnostics.push(
                Diagnostic::error("a joint endpoint must be a body name")
                    .with_code("E0401")
                    .at(item.span, "not a name"),
            );
            return None;
        };
        names.push(name.to_string());
    }
    if names[0] == names[1] {
        diagnostics.push(
            Diagnostic::error(format!("joint `{}` connects a body to itself", decl.name.text))
                .with_code("E0401")
                .at(setting.value.span, "both endpoints are the same body")
                .note("a self-joint constrains nothing, and its effective mass is infinite"),
        );
        return None;
    }
    Some([names[0].clone(), names[1].clone()])
}

/// Resolve a joint's body names to slots, reporting any that do not exist.
pub fn resolve(
    plan: &JointPlan,
    slots: &BTreeMap<String, usize>,
    diagnostics: &mut Diagnostics,
) -> Option<(usize, usize)> {
    let mut resolved = [0usize; 2];
    for (index, name) in plan.bodies.iter().enumerate() {
        match slots.get(name) {
            Some(slot) => resolved[index] = *slot,
            None => {
                let known: Vec<&str> = slots.keys().map(String::as_str).collect();
                diagnostics.push(
                    Diagnostic::error(format!("there is no body called `{name}`"))
                        .with_code("E0202")
                        .at(plan.span, "unknown body")
                        .help(if known.is_empty() {
                            "declare one with `body ground { shape: … }`".to_string()
                        } else {
                            format!("declared bodies: {}", known.join(", "))
                        }),
                );
                return None;
            }
        }
    }
    Some((resolved[0], resolved[1]))
}

/// A body's one-line entry in the model report.
pub fn describe(plan: &BodyPlan) -> String {
    let geometry = match &plan.shape {
        Shape::Circle { radius } => format!("circle r = {radius} m"),
        Shape::Polygon(polygon) => format!("{}-gon", polygon.vertices().len()),
        Shape::Segment { half_length } => format!("segment {} m long", 2.0 * half_length),
    };
    let motion = if plan.is_static { "static" } else { "dynamic" };
    format!(
        "{motion} {geometry} at [{:.3}, {:.3}] m, e = {}, mu = {}",
        plan.at[0], plan.at[1], plan.material.restitution, plan.material.friction
    )
}

/// A body's name and where it was declared, for a duplicate check.
pub fn identity(decl: &Decl) -> (&str, &Ident) {
    (decl.name.text.as_str(), &decl.name)
}
