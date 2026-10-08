//! The `quantum2d` vocabulary: the domain block, potentials, wave packets, detectors.
//!
//! Spec §25.2, which this module compiles as written:
//!
//! ```text
//!   domain quantum2d q {
//!     grid: [768, 384];
//!     extent: [12 nanometer, 6 nanometer];
//!     mass: electron_mass;
//!     boundary: absorbing(width=0.8 nanometer);
//!     integrator: split_step_fourier(dt=0.002 femtosecond);
//!   }
//!   potential barrier {
//!     shape: vertical_wall(x=0, thickness=0.15 nanometer);
//!     slits: [(-1.0, 0.35), (1.0, 0.35)] nanometer;
//!     height: 20 electronvolt;
//!   }
//!   wavepacket initial {
//!     center: [-4 nanometer, 0];
//!     momentum: [6.5e-24 kilogram*meter/second, 0];
//!     sigma: 0.45 nanometer;
//!   }
//!   detector screen at x=4.5 nanometer;
//! ```
//!
//! A detector can also count single arrivals — `detector screen at x=4.5 nanometer,
//! clicks=4000, seed=7;` fires 4000 particles and records where each one lands, drawn
//! from the current through the screen (see `lattice_domain_quantum2d::sampling`).
//!
//! The grid is centred on the origin unless an `origin:` says otherwise — §25.2 puts
//! its wall at `x = 0` and its packet at `x = −4 nm` on a 12 nm grid, which only makes
//! sense centred. Several wave packets superpose; their sum is normalized.
//!
//! Three things are checked here that nothing else would catch, as warnings because
//! each is a legitimate experiment done on purpose and a mistake done by accident:
//!
//! - a packet whose kinetic energy is above a barrier's height, which then mostly
//!   transmits rather than tunnels or diffracts (`W0308`);
//! - a packet whose momenta reach the grid's Nyquist wavenumber, which then alias
//!   (`W0309`);
//! - a split-step step too long for a potential with sharp edges, measured to bias
//!   transmission by percents (`W0310`).

use lattice_domain_quantum2d::{
    Absorber, Complex, Hamiltonian, Kinetic, Potential, QuantumDomain, Scheme, Shape, Wavefunction, HBAR,
    SPLIT_STEP_PHASE,
};
use lattice_ir::Grid2d;
use lattice_syntax::{BinaryOp, Decl, Diagnostic, Diagnostics, DomainDecl, Expr, ExprKind, Span};
use lattice_units::Dimension;

use crate::builtins::{unknown, Call};
use crate::eval::Evaluator;

const SETTINGS: &[&str] = &["grid", "extent", "origin", "mass", "boundary", "integrator"];
const INTEGRATORS: &[&str] = &["split_step_fourier", "crank_nicolson"];
const BOUNDARIES: &[&str] = &["periodic", "walls", "absorbing"];
const SHAPES: &[&str] = &["vertical_wall", "rectangle", "harmonic"];

/// One electronvolt, J, for the messages.
const EV: f64 = lattice_units::constants::value::ELECTRONVOLT;

/// What the grid's edges do.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Edges {
    /// Split-step's own: what leaves one side enters the other.
    Periodic,
    /// Crank–Nicolson's own: `ψ = 0` on the outer faces.
    Walls,
    /// A complex absorbing layer in front of whichever of those the scheme has.
    Absorbing {
        /// Depth, m.
        width: f64,
        /// Peak strength, J; derived from the wave packets when not given.
        strength: Option<f64>,
    },
}

/// A `domain quantum2d` block, read.
#[derive(Clone, PartialEq, Debug)]
pub struct QuantumPlan {
    /// The domain's name.
    pub name: String,
    /// Where it was declared.
    pub span: Span,
    /// The grid.
    pub grid: Grid2d,
    /// The particle's mass, kg.
    pub mass: f64,
    /// The integrator.
    pub scheme: Scheme,
    /// The integrator's `dt`, if given.
    pub dt: Option<f64>,
    /// Crank–Nicolson's solver tolerance, if given.
    pub tolerance: Option<f64>,
    /// The edges.
    pub edges: Edges,
}

/// A `wavepacket` declaration, read.
#[derive(Clone, PartialEq, Debug)]
pub struct PacketPlan {
    /// Its name.
    pub name: String,
    /// Centre, m.
    pub center: [f64; 2],
    /// Mean momentum, kg·m/s.
    pub momentum: [f64; 2],
    /// Position spread per axis, m.
    pub sigma: [f64; 2],
    /// Where it was declared.
    pub span: Span,
}

/// A `potential` declaration, read.
#[derive(Clone, PartialEq, Debug)]
pub struct PotentialPlan {
    /// Its name.
    pub name: String,
    /// What it is.
    pub shape: Shape,
    /// Where it was declared.
    pub span: Span,
}

fn require_positive(value: f64, what: &str, span: Span, diagnostics: &mut Diagnostics) -> Option<f64> {
    if value > 0.0 && value.is_finite() {
        Some(value)
    } else {
        diagnostics.push(
            Diagnostic::error(format!("{what} must be positive"))
                .with_code("E0405")
                .at(span, format!("found {value:e}")),
        );
        None
    }
}

fn missing(what: &str, setting: &str, example: &str, span: Span, diagnostics: &mut Diagnostics) {
    diagnostics.push(
        Diagnostic::error(format!("{what} needs a `{setting}`"))
            .with_code("E0203")
            .at(span, format!("missing `{setting}`"))
            .help(format!("add `{setting}: {example};`")),
    );
}

/// Read a `domain quantum2d` block.
pub fn domain(decl: &DomainDecl, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<QuantumPlan> {
    for setting in &decl.settings {
        if !SETTINGS.contains(&setting.key.text.as_str()) {
            diagnostics.push(
                Diagnostic::error(format!("a quantum2d domain has no setting called `{}`", setting.key.text))
                    .with_code("E0204")
                    .at(setting.key.span, "unknown setting")
                    .help(format!("quantum2d accepts: {}", SETTINGS.join(", "))),
            );
        }
    }
    let what = format!("quantum2d domain `{}`", decl.name.text);

    let cells = match decl.setting("grid") {
        Some(setting) => evaluator.fixed_list(&setting.value, 2, "`grid`", diagnostics).and_then(|items| {
            let nx = evaluator.count(items[0], "the cell count along x", diagnostics)?;
            let ny = evaluator.count(items[1], "the cell count along y", diagnostics)?;
            if nx < 2 || ny < 1 {
                diagnostics.push(
                    Diagnostic::error("a quantum grid needs at least two cells along x and one along y")
                        .with_code("E0405")
                        .at(setting.value.span, format!("{nx} x {ny}")),
                );
                return None;
            }
            Some([nx, ny])
        }),
        None => {
            missing(&what, "grid", "[256, 128]", decl.name.span, diagnostics);
            None
        }
    };
    let extent = match decl.setting("extent") {
        Some(setting) => evaluator.pair(&setting.value, Dimension::LENGTH, "`extent`", diagnostics).and_then(|e| {
            Some([
                require_positive(e[0], "the extent", setting.value.span, diagnostics)?,
                require_positive(e[1], "the extent", setting.value.span, diagnostics)?,
            ])
        }),
        None => {
            missing(&what, "extent", "[12 nanometer, 6 nanometer]", decl.name.span, diagnostics);
            None
        }
    };
    let origin = decl
        .setting("origin")
        .map(|setting| evaluator.pair(&setting.value, Dimension::LENGTH, "`origin`", diagnostics));
    let mass = match decl.setting("mass") {
        Some(setting) => evaluator
            .require(&setting.value, Dimension::MASS, "`mass`", diagnostics)
            .and_then(|m| require_positive(m, "the mass", setting.value.span, diagnostics)),
        None => {
            missing(&what, "mass", "electron_mass", decl.name.span, diagnostics);
            None
        }
    };

    let (scheme, dt, tolerance) = match decl.setting("integrator") {
        None => (Scheme::SplitStepFourier, None, None),
        Some(setting) => integrator(&setting.value, evaluator, diagnostics)?,
    };
    let edges = match decl.setting("boundary") {
        None => match scheme {
            Scheme::SplitStepFourier => Edges::Periodic,
            Scheme::CrankNicolson => Edges::Walls,
        },
        Some(setting) => boundary(&setting.value, scheme, evaluator, diagnostics)?,
    };

    let [nx, ny] = cells?;
    let extent = extent?;
    let origin = match origin {
        Some(pair) => pair?,
        None => [-0.5 * extent[0], -0.5 * extent[1]],
    };
    let grid = Grid2d::with_origin(nx, ny, extent, origin);
    if let Edges::Absorbing { width, .. } = edges {
        let short = [(nx, extent[0]), (ny, extent[1])].iter().any(|&(n, e)| n > 1 && 2.0 * width >= e);
        if short {
            diagnostics.push(
                Diagnostic::error("the absorbing layers on opposite sides would meet")
                    .with_code("E0405")
                    .at(decl.setting("boundary").map_or(decl.name.span, |s| s.value.span), format!("{width:e} m deep"))
                    .note(format!("the grid is {:e} x {:e} m", extent[0], extent[1])),
            );
            return None;
        }
    }
    Some(QuantumPlan { name: decl.name.text.clone(), span: decl.name.span, grid, mass: mass?, scheme, dt, tolerance, edges })
}

fn integrator(
    expr: &Expr,
    evaluator: &Evaluator<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<(Scheme, Option<f64>, Option<f64>)> {
    let (name, span, call) = match Call::match_expr(expr) {
        Some(call) => (call.name, call.span, Some(call)),
        None => (expr.as_name().unwrap_or(""), expr.span, None),
    };
    let scheme = match name {
        "split_step_fourier" => Scheme::SplitStepFourier,
        "crank_nicolson" => Scheme::CrankNicolson,
        other => {
            unknown("quantum integrator", other, span, INTEGRATORS, diagnostics);
            return None;
        }
    };
    let Some(call) = call else { return Some((scheme, None, None)) };
    let allowed: &[&str] = match scheme {
        Scheme::SplitStepFourier => &["dt"],
        Scheme::CrankNicolson => &["dt", "tolerance"],
    };
    call.reject_unknown(allowed, diagnostics);
    let dt = call.get("dt", 0).and_then(|expr| {
        let dt = evaluator.require(expr, Dimension::TIME, "the timestep `dt`", diagnostics)?;
        require_positive(dt, "the timestep", expr.span, diagnostics)
    });
    let tolerance = call.named("tolerance").and_then(|expr| {
        let value = evaluator.dimensionless(expr, "the solver tolerance", diagnostics)?;
        require_positive(value, "the solver tolerance", expr.span, diagnostics)
    });
    Some((scheme, dt, tolerance))
}

fn boundary(expr: &Expr, scheme: Scheme, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<Edges> {
    let (name, span, call) = match Call::match_expr(expr) {
        Some(call) => (call.name, call.span, Some(call)),
        None => (expr.as_name().unwrap_or(""), expr.span, None),
    };
    let edges = match name {
        "periodic" => Edges::Periodic,
        "walls" => Edges::Walls,
        "absorbing" => {
            let Some(call) = call else {
                diagnostics.push(
                    Diagnostic::error("an absorbing boundary needs its depth")
                        .with_code("E0203")
                        .at(span, "no width")
                        .help("write `absorbing(width=0.8 nanometer)`"),
                );
                return None;
            };
            call.reject_unknown(&["width", "strength"], diagnostics);
            let width = call.require("width", 0, diagnostics).and_then(|expr| {
                let w = evaluator.require(expr, Dimension::LENGTH, "the absorbing width", diagnostics)?;
                require_positive(w, "the absorbing width", expr.span, diagnostics)
            })?;
            let strength = match call.named("strength") {
                Some(expr) => Some(
                    evaluator
                        .require(expr, Dimension::ENERGY, "the absorbing strength", diagnostics)
                        .and_then(|s| require_positive(s, "the absorbing strength", expr.span, diagnostics))?,
                ),
                None => None,
            };
            Edges::Absorbing { width, strength }
        }
        other => {
            unknown("quantum boundary", other, span, BOUNDARIES, diagnostics);
            return None;
        }
    };
    let conflict = match (edges, scheme) {
        (Edges::Periodic, Scheme::CrankNicolson) => Some("Crank-Nicolson here solves a box with walls; periodic edges need `split_step_fourier`"),
        (Edges::Walls, Scheme::SplitStepFourier) => Some("split-step Fourier is periodic by construction; walls need `crank_nicolson`"),
        _ => None,
    };
    if let Some(reason) = conflict {
        diagnostics.push(
            Diagnostic::error(format!("`{name}` edges and the `{}` integrator do not go together", scheme.name()))
                .with_code("E0208")
                .at(span, "conflicts with the integrator")
                .note(reason),
        );
        return None;
    }
    Some(edges)
}

/// `[(a, b), (c, d)] nanometer` or `[(a nm, b nm), …]`: pairs of lengths.
fn length_pairs(expr: &Expr, what: &str, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<Vec<(f64, f64)>> {
    if let ExprKind::Binary(BinaryOp::Mul, list, unit) = &expr.kind
        && let ExprKind::List(items) = &list.kind
    {
        // A unit written once after the list applies to every number in it.
        let scale = evaluator.require(unit, Dimension::LENGTH, &format!("the unit of {what}"), diagnostics)?;
        let mut pairs = Vec::with_capacity(items.len());
        for item in items {
            let parts = evaluator.fixed_list(item, 2, what, diagnostics)?;
            let a = evaluator.dimensionless(parts[0], what, diagnostics)?;
            let b = evaluator.dimensionless(parts[1], what, diagnostics)?;
            pairs.push((a * scale, b * scale));
        }
        return Some(pairs);
    }
    let ExprKind::List(items) = &expr.kind else {
        diagnostics.push(
            Diagnostic::error(format!("{what} must be a list of (centre, width) pairs"))
                .with_code("E0401")
                .at(expr.span, "expected a list")
                .help("write `[(-1.0, 0.35), (1.0, 0.35)] nanometer`"),
        );
        return None;
    };
    items
        .iter()
        .map(|item| evaluator.pair(item, Dimension::LENGTH, what, diagnostics).map(|[a, b]| (a, b)))
        .collect()
}

/// Read a `potential` declaration.
pub fn potential(decl: &Decl, mass: f64, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<PotentialPlan> {
    for setting in &decl.settings {
        if !["shape", "slits", "height"].contains(&setting.key.text.as_str()) {
            diagnostics.push(
                Diagnostic::error(format!("a potential has no setting called `{}`", setting.key.text))
                    .with_code("E0204")
                    .at(setting.key.span, "unknown setting")
                    .help("potentials accept: shape, slits, height"),
            );
        }
    }
    let what = format!("potential `{}`", decl.name.text);
    let Some(shape_setting) = decl.setting("shape") else {
        missing(&what, "shape", "vertical_wall(x=0, thickness=0.15 nanometer)", decl.name.span, diagnostics);
        return None;
    };
    let Some(call) = Call::match_expr(&shape_setting.value) else {
        diagnostics.push(
            Diagnostic::error("a potential's shape is written as a call")
                .with_code("E0210")
                .at(shape_setting.value.span, "not a shape")
                .note(format!("available: {}", SHAPES.join(", "))),
        );
        return None;
    };
    let height = |diagnostics: &mut Diagnostics| match decl.setting("height") {
        Some(setting) => evaluator.require(&setting.value, Dimension::ENERGY, "the potential's `height`", diagnostics),
        None => {
            missing(&what, "height", "20 electronvolt", decl.name.span, diagnostics);
            None
        }
    };
    let length = |name: &str, index: usize, diagnostics: &mut Diagnostics| {
        call.require(name, index, diagnostics)
            .and_then(|expr| evaluator.require(expr, Dimension::LENGTH, &format!("`{name}`"), diagnostics))
    };
    let refuse_slits = |shape: &str, diagnostics: &mut Diagnostics| {
        if let Some(setting) = decl.setting("slits") {
            diagnostics.push(
                Diagnostic::error(format!("only a wall has slits, not a {shape}"))
                    .with_code("E0204")
                    .at(setting.key.span, "slits on a shape without them"),
            );
        }
    };

    let shape = match call.name {
        "vertical_wall" => {
            call.reject_unknown(&["x", "thickness"], diagnostics);
            let x = length("x", 0, diagnostics);
            let thickness = length("thickness", 1, diagnostics)
                .and_then(|t| require_positive(t, "the wall's thickness", shape_setting.value.span, diagnostics));
            let slits = match decl.setting("slits") {
                Some(setting) => length_pairs(&setting.value, "`slits`", evaluator, diagnostics)?,
                None => Vec::new(),
            };
            Shape::Wall { x: x?, thickness: thickness?, slits, height: height(diagnostics)? }
        }
        "rectangle" => {
            call.reject_unknown(&["x", "y"], diagnostics);
            refuse_slits("rectangle", diagnostics);
            let range = |name: &str, index: usize, diagnostics: &mut Diagnostics| {
                call.require(name, index, diagnostics)
                    .and_then(|expr| evaluator.pair(expr, Dimension::LENGTH, &format!("`{name}`"), diagnostics))
            };
            let (x, y) = (range("x", 0, diagnostics), range("y", 1, diagnostics));
            Shape::Rectangle { x: x?, y: y?, height: height(diagnostics)? }
        }
        "harmonic" => {
            call.reject_unknown(&["center", "omega"], diagnostics);
            refuse_slits("harmonic trap", diagnostics);
            if let Some(setting) = decl.setting("height") {
                diagnostics.push(
                    Diagnostic::error("a harmonic trap is set by its frequency, not a height")
                        .with_code("E0204")
                        .at(setting.key.span, "height on a harmonic trap"),
                );
            }
            let center = match call.named("center") {
                Some(expr) => evaluator.pair(expr, Dimension::LENGTH, "`center`", diagnostics)?,
                None => [0.0, 0.0],
            };
            let omega = call.require("omega", 0, diagnostics).and_then(|expr| {
                let w = evaluator.require(expr, Dimension::FREQUENCY, "the trap's `omega`", diagnostics)?;
                require_positive(w, "the trap's frequency", expr.span, diagnostics)
            })?;
            Shape::Harmonic { center, omega, mass }
        }
        other => {
            unknown("potential shape", other, call.span, SHAPES, diagnostics);
            return None;
        }
    };
    Some(PotentialPlan { name: decl.name.text.clone(), shape, span: decl.name.span })
}

/// Read a `wavepacket` declaration.
pub fn wavepacket(decl: &Decl, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<PacketPlan> {
    for setting in &decl.settings {
        if !["center", "momentum", "sigma"].contains(&setting.key.text.as_str()) {
            diagnostics.push(
                Diagnostic::error(format!("a wave packet has no setting called `{}`", setting.key.text))
                    .with_code("E0204")
                    .at(setting.key.span, "unknown setting")
                    .help("wave packets accept: center, momentum, sigma"),
            );
        }
    }
    let what = format!("wave packet `{}`", decl.name.text);
    let center = match decl.setting("center") {
        Some(setting) => evaluator.pair(&setting.value, Dimension::LENGTH, "`center`", diagnostics),
        None => {
            missing(&what, "center", "[-4 nanometer, 0]", decl.name.span, diagnostics);
            None
        }
    };
    let momentum = match decl.setting("momentum") {
        Some(setting) => evaluator.pair(&setting.value, Dimension::MOMENTUM, "`momentum`", diagnostics),
        None => Some([0.0, 0.0]),
    };
    let sigma = match decl.setting("sigma") {
        Some(setting) => {
            let pair = matches!(setting.value.kind, ExprKind::List(_) | ExprKind::Tuple(_));
            let sigma = if pair {
                evaluator.pair(&setting.value, Dimension::LENGTH, "`sigma`", diagnostics)
            } else {
                evaluator.require(&setting.value, Dimension::LENGTH, "`sigma`", diagnostics).map(|s| [s, s])
            };
            sigma.and_then(|[a, b]| {
                Some([
                    require_positive(a, "a packet's width", setting.value.span, diagnostics)?,
                    require_positive(b, "a packet's width", setting.value.span, diagnostics)?,
                ])
            })
        }
        None => {
            missing(&what, "sigma", "0.45 nanometer", decl.name.span, diagnostics);
            None
        }
    };
    Some(PacketPlan { name: decl.name.text.clone(), center: center?, momentum: momentum?, sigma: sigma?, span: decl.name.span })
}

/// A detector as declared.
#[derive(Clone, Debug)]
pub struct DetectorPlan {
    /// Its name.
    pub name: String,
    /// Where it was asked to be, m.
    pub x: f64,
    /// Particles fired and the seed their arrivals are drawn with, when it counts them.
    pub clicks: Option<(u64, u64)>,
    /// The declaration's name, for diagnostics.
    pub span: Span,
}

/// Read `detector <name> at x=…[, clicks=…, seed=…];`.
pub fn detector(decl: &Decl, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<DetectorPlan> {
    let mut clicks = None;
    let x = if decl.modifier.as_ref().is_some_and(|m| m.text == "at") {
        let call = Call { name: "at", span: decl.span, arguments: &decl.arguments };
        call.reject_unknown(&["x", "clicks", "seed"], diagnostics);
        if let Some(fired) = call.named("clicks") {
            let fired = evaluator.count(fired, "the detector's `clicks`", diagnostics)? as u64;
            let seed = match call.named("seed") {
                Some(seed) => evaluator.count(seed, "the detector's `seed`", diagnostics)? as u64,
                None => 0,
            };
            clicks = Some((fired, seed));
        } else if let Some(seed) = call.named("seed") {
            diagnostics.push(
                Diagnostic::error("a detector's `seed` draws its clicks, and it has none")
                    .with_code("E0204")
                    .at(seed.span, "nothing to seed")
                    .help(format!("count arrivals with `detector {} at x=…, clicks=4000, seed=…;`", decl.name.text)),
            );
            return None;
        }
        call.require("x", 0, diagnostics)
            .and_then(|expr| evaluator.require(expr, Dimension::LENGTH, "the detector's `x`", diagnostics))
    } else {
        diagnostics.push(
            Diagnostic::error("a detector is placed with `at x=…`")
                .with_code("E0203")
                .at(decl.name.span, "no position")
                .help(format!("write `detector {} at x=4.5 nanometer;`", decl.name.text)),
        );
        None
    };
    Some(DetectorPlan { name: decl.name.text.clone(), x: x?, clicks, span: decl.name.span })
}

/// Everything a quantum domain is built from.
#[derive(Clone, Debug)]
pub struct Assembly<'a> {
    /// The domain block.
    pub plan: &'a QuantumPlan,
    /// Its potentials.
    pub potentials: &'a [PotentialPlan],
    /// Its wave packets.
    pub packets: &'a [PacketPlan],
    /// Its detectors.
    pub detectors: &'a [DetectorPlan],
}

impl Assembly<'_> {
    /// The mean kinetic energy of a packet, J.
    fn energy(&self, packet: &PacketPlan) -> f64 {
        (packet.momentum[0].powi(2) + packet.momentum[1].powi(2)) / (2.0 * self.plan.mass)
    }

    /// One line for the model report.
    pub fn describe(&self) -> String {
        let plan = self.plan;
        let [lx, ly] = plan.grid.extent();
        let mut parts = vec![format!(
            "{}x{} over {} x {} m, mass {} kg, {}",
            plan.grid.nx(),
            plan.grid.ny(),
            lattice_ir::format_number(lx),
            lattice_ir::format_number(ly),
            lattice_ir::format_number(plan.mass),
            plan.scheme.name()
        )];
        parts.push(match plan.edges {
            Edges::Periodic => "periodic edges".to_string(),
            Edges::Walls => "walls on the outer faces".to_string(),
            Edges::Absorbing { width, .. } => format!("absorbing layer {width:.4e} m deep"),
        });
        for potential in self.potentials {
            parts.push(format!("potential `{}`: {}", potential.name, potential.shape.describe()));
        }
        for packet in self.packets {
            parts.push(format!(
                "packet `{}` at ({:.4e}, {:.4e}) m, |p| = {:.4e} kg m/s ({:.4} eV)",
                packet.name,
                packet.center[0],
                packet.center[1],
                packet.momentum[0].hypot(packet.momentum[1]),
                self.energy(packet) / EV
            ));
        }
        for DetectorPlan { name, x, clicks, .. } in self.detectors {
            parts.push(match clicks {
                Some((fired, seed)) => format!("detector `{name}` at x = {x:.4e} m counting {fired} arrivals (seed {seed})"),
                None => format!("detector `{name}` at x = {x:.4e} m"),
            });
        }
        parts.join("; ")
    }

    /// Build the domain, reporting what it cannot build and warning about what it can
    /// but probably should not.
    pub fn build(&self, diagnostics: &mut Diagnostics) -> Option<QuantumDomain> {
        let plan = self.plan;
        let grid = plan.grid;
        if self.packets.is_empty() {
            diagnostics.push(
                Diagnostic::error(format!("quantum2d domain `{}` has no wave packet to evolve", plan.name))
                    .with_code("E0203")
                    .at(plan.span, "no initial state")
                    .help("declare `wavepacket initial { center: [...]; momentum: [...]; sigma: ...; }`"),
            );
            return None;
        }

        let mut potential = Potential::zero(grid);
        for p in self.potentials {
            potential.add(&p.shape);
        }
        let kinetic = match plan.scheme {
            Scheme::SplitStepFourier => Kinetic::Spectral,
            Scheme::CrankNicolson => Kinetic::FiniteDifference,
        };
        let mut h = Hamiltonian::new(grid, plan.mass, HBAR, kinetic).with_potential(potential);
        if let Edges::Absorbing { width, strength } = plan.edges {
            // Sized for the fastest thing that will reach it: a packet's group velocity
            // plus three of its velocity spreads.
            let strength = strength.unwrap_or_else(|| {
                let speed = self
                    .packets
                    .iter()
                    .map(|p| {
                        let spread = HBAR / (2.0 * p.sigma[0].min(p.sigma[1]));
                        (p.momentum[0].hypot(p.momentum[1]) + 3.0 * spread) / plan.mass
                    })
                    .fold(0.0, f64::max);
                Absorber::strength_for_speed(width, speed, HBAR)
            });
            let axes = [grid.nx() > 1, grid.ny() > 1];
            h = h.with_absorber(Absorber::on_axes(grid, width, strength, axes));
        }

        // Superpose the packets, then normalize the sum.
        let mut psi = Wavefunction::zeros(grid);
        for packet in self.packets {
            let one = Wavefunction::gaussian(grid, packet.center, packet.sigma, packet.momentum, HBAR);
            for (z, add) in psi.as_mut_slice().iter_mut().zip(one.as_slice()) {
                *z += *add;
            }
        }
        if psi.normalize() == 0.0 {
            diagnostics.push(
                Diagnostic::error("the wave packets cancel, leaving no particle")
                    .with_code("E0405")
                    .at(plan.span, "zero norm"),
            );
            return None;
        }
        debug_assert!(psi.as_slice().iter().all(|z: &Complex| z.is_finite()));

        self.warn(&h, diagnostics);

        let mut domain = QuantumDomain::new(plan.name.clone(), h).with_state(psi);
        if let Some(dt) = plan.dt {
            domain = domain.with_preferred_step(dt);
        }
        if let Some(tolerance) = plan.tolerance {
            domain = domain.with_tolerance(tolerance);
        }
        let first = grid.cell_center(0, 0)[0];
        let last = grid.cell_center(grid.nx() - 1, 0)[0];
        for DetectorPlan { name, x, clicks, span } in self.detectors {
            if !(first..=last).contains(x) {
                diagnostics.push(
                    Diagnostic::error(format!("detector `{name}` is off the grid"))
                        .with_code("E0405")
                        .at(*span, format!("x = {x:e} m"))
                        .note(format!("cell centres run from {first:e} to {last:e} m")),
                );
                return None;
            }
            domain = match clicks {
                Some((fired, seed)) => domain.with_clicking_detector(name.clone(), *x, *fired, *seed),
                None => domain.with_detector(name.clone(), *x),
            };
        }
        Some(domain)
    }

    fn warn(&self, h: &Hamiltonian, diagnostics: &mut Diagnostics) {
        let plan = self.plan;
        let grid = plan.grid;
        // W0308: above the barrier.
        let barrier = self
            .potentials
            .iter()
            .filter_map(|p| match &p.shape {
                Shape::Wall { height, .. } | Shape::Rectangle { height, .. } if *height > 0.0 => Some((p, *height)),
                _ => None,
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((potential, height)) = barrier {
            for packet in self.packets {
                let energy = self.energy(packet);
                if energy > height {
                    diagnostics.push(
                        Diagnostic::warning(format!(
                            "wave packet `{}` carries more energy than potential `{}` is high",
                            packet.name, potential.name
                        ))
                        .with_code("W0308")
                        .at(packet.span, format!("{:.3} eV", energy / EV))
                        .also(potential.span, format!("{:.3} eV high", height / EV))
                        .note("a packet above a barrier mostly passes over it, rather than tunnelling or diffracting")
                        .help("lower the momentum or raise the height if the barrier is meant to stop it"),
                    );
                }
            }
        }
        // W0309: the grid cannot represent the packet.
        let nyquist = [core::f64::consts::PI / grid.dx(), core::f64::consts::PI / grid.dy()];
        for packet in self.packets {
            for axis in 0..2 {
                if [grid.nx(), grid.ny()][axis] < 2 {
                    continue;
                }
                let reach = packet.momentum[axis].abs() / HBAR + 3.0 / (2.0 * packet.sigma[axis]);
                if reach > nyquist[axis] {
                    diagnostics.push(
                        Diagnostic::warning(format!("the grid cannot represent wave packet `{}`", packet.name))
                            .with_code("W0309")
                            .at(packet.span, format!("wavenumbers reach {reach:.3e} rad/m"))
                            .note(format!(
                                "the grid's Nyquist wavenumber along {} is {:.3e} rad/m; beyond it momenta alias \
                                 onto the opposite direction",
                                ["x", "y"][axis],
                                nyquist[axis]
                            ))
                            .help("refine the grid or lower the momentum"),
                    );
                    break;
                }
            }
        }
        // W0310: split-step's step against a sharp potential.
        let sharp = self.potentials.iter().any(|p| matches!(p.shape, Shape::Wall { .. } | Shape::Rectangle { .. }));
        if let (Scheme::SplitStepFourier, Some(dt), true) = (plan.scheme, plan.dt, sharp) {
            let phase = h.max_kinetic() * dt / HBAR;
            if phase > SPLIT_STEP_PHASE {
                let suggested = SPLIT_STEP_PHASE * HBAR / h.max_kinetic();
                diagnostics.push(
                    Diagnostic::warning("this step is too long for a potential with sharp edges")
                        .with_code("W0310")
                        .at(plan.span, format!("{phase:.1} radians of kinetic phase per step at the grid's top wavenumber"))
                        .note(
                            "a wall or rectangle reaches the grid's highest wavenumbers, where split-step's \
                             splitting error lives; at 7.5 radians a validated tunnelling case came out 1.9% wrong",
                        )
                        .help(format!("use dt <= {suggested:.3e} s, or leave dt out to get that")),
                );
            }
        }
    }
}
