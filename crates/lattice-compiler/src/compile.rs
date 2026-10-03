//! The compilation pass.
//!
//! Spec §8.4 lists nine steps; this implements the ones M1 can honour:
//!
//! 1. Parse the project into an AST — `lattice-syntax`.
//! 2. **Resolve** names, dimensions, units, grids.
//! 3. **Normalize** declarations into domain configurations.
//! 4. Construct read/write sets and dependency edges.
//! 5. **Select solver implementations** from the fidelity profile and the method named.
//! 6. **Plan buffers** and size the scratch arena.
//! 7. *(Kernel compilation and caching — M4.)*
//! 8. **Validate**: dimensional analysis, topology, boundary completeness.
//! 9. **Emit** the immutable [`CompiledModel`] plus a model report.
//!
//! # Constructs that do not exist yet
//!
//! A `reaction` or `couple` declaration parses and is checked as far as it can be,
//! then produces an error naming the milestone that will implement it. It is an
//! *error*, not a warning: a model whose chemistry is silently dropped would run and
//! produce confident, wrong numbers, which is precisely what spec P1 exists to
//! prevent. `lattice check` still prints the report for everything that did compile.

use std::collections::BTreeMap;

use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_domain_chemistry::{ReactingMixture, ReactionNetwork, Species as ChemSpecies};
use lattice_domain_rigid2d::{Collider, RigidDomain, SolverConfig};
use lattice_coupling::{Coupler, CouplingEdge, Mapping, PortRef};

use crate::{chemistry, molecular, quantum, rigid};
use lattice_domain_quantum2d::Scheme as QuantumScheme;
use lattice_domain_particle::{
    analysis, Angle, Bond, BoundaryBox, HarmonicAngle, HarmonicBond, HarmonicWell, Integrator,
    LennardJones, LinearDrag, ParticleBoundary, ParticleDomain, ParticleId, ParticleSpec,
    RdfRequest, SoftRepulsion, Thermostat, UniformAcceleration,
};
use lattice_ir::{
    BodySpec, BoundarySet, BufferKind, BufferPlan, CompiledModel, Domain, DomainId, DomainSpec,
    Invariant,
    FidelityProfile, Grid2d, ObserverId, ObserverSpec, Operation, OperationGraph, OperationKind,
    Pcg32, Precision, Side, VisualSpec,
};
use lattice_syntax::{
    Decl, Diagnostic, Diagnostics, Expr, ExprKind, FieldDecl, Ident, Project, SolveStmt, SourceFile,
    Span,
};
use lattice_units::{Dimension, UnitRegistry};

use crate::builtins::{self, ForceSpec, Initializer};
use crate::eval::Evaluator;
use crate::molecular::{AngleSpec, BondSpec};

/// How a particle set is laid out at the start.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Layout {
    /// Row by row, every row left to right.
    #[default]
    Lattice,
    /// Row by row, alternating direction, so consecutive indices are always one
    /// spacing apart — what a bonded chain needs to start unstrained.
    Serpentine,
}

/// A compiled, runnable project.
pub struct Compiled {
    /// The immutable description.
    pub model: CompiledModel,
    /// The instantiated solvers, parallel to `model.domains`.
    pub domains: Vec<Box<dyn Domain>>,
    /// The coupling edges between them, and the ledger that will account for what
    /// they move (spec §14).
    pub coupler: Coupler,
}

impl core::fmt::Debug for Compiled {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Compiled")
            .field("model", &self.model.name)
            .field("domains", &self.domains.len())
            .finish()
    }
}

/// Compile a parsed project.
///
/// Always returns a best-effort [`Compiled`] alongside the diagnostics, so
/// `lattice check` can print what it understood even when something failed. Callers
/// that intend to *run* the model must check [`Diagnostics::has_errors`] first.
pub fn compile(file: &SourceFile, project: &Project) -> (Compiled, Diagnostics) {
    // The registry lives outside the compiler so that `Compiler::evaluator` can hand
    // back an `Evaluator` borrowing *it* rather than borrowing `self`. Without that,
    // every `evaluator.require(…, &mut self.diagnostics)` would be a borrow conflict.
    let units = UnitRegistry::si();
    let mut compiler = Compiler::new(file, &units);
    let compiled = compiler.run(project);
    (compiled, compiler.diagnostics)
}

/// The collections a lowering pass appends to.
///
/// Bundled rather than passed as four parallel `&mut` parameters: every `lower_*`
/// method needs all of them, and threading them separately made the signatures long
/// enough to obscure which arguments were inputs and which were outputs.
#[derive(Default)]
struct Lowering {
    domains: Vec<Box<dyn Domain>>,
    specs: Vec<DomainSpec>,
    buffers: BufferPlan,
    operations: Vec<Operation>,
}

/// What the compiler learned about a declared grid.
#[derive(Clone, Debug)]
struct GridInfo {
    grid: Grid2d,
    span: Span,
}

/// What the compiler learned about a declared field or species.
#[derive(Clone, Debug)]
struct FieldInfo {
    name: String,
    grid: String,
    dimension: Dimension,
    initializer: Initializer,
    diffusivity: Option<f64>,
    boundaries: BoundarySet,
    is_species: bool,
    span: Span,
    /// Set once a `solve` claims this field, so an unsolved field can be warned about.
    solved: bool,
}

/// What the compiler learned about a declared particle set.
#[derive(Clone, Debug)]
struct ParticlesInfo {
    name: String,
    count: usize,
    origin: [f64; 2],
    region: Option<[f64; 2]>,
    mass: f64,
    radius: f64,
    boundary: ParticleBoundary,
    spacing: Option<f64>,
    speed: f64,
    /// An initial temperature, an alternative to `speed`.
    temperature: Option<f64>,
    seed: u64,
    forces: Vec<ForceSpec>,
    skin: Option<f64>,
    thermostat: Option<Thermostat>,
    bonds: Vec<BondSpec>,
    angles: Vec<AngleSpec>,
    analyses: Vec<RdfRequest>,
    layout: Layout,
    span: Span,
    solved: bool,
}

struct Compiler<'a> {
    file: &'a SourceFile,
    units: &'a UnitRegistry,
    diagnostics: Diagnostics,
    grids: BTreeMap<String, GridInfo>,
    fields: BTreeMap<String, FieldInfo>,
    particles: BTreeMap<String, ParticlesInfo>,
    materials: BTreeMap<String, rigid::Material>,
    /// Bodies in declaration order, which is the order they take slots in.
    bodies: Vec<rigid::BodyPlan>,
    /// How many `body` declarations were *seen*, including those that failed to build.
    ///
    /// Kept separately so a scene whose bodies were all rejected reports the rejections
    /// and not a second, misleading "this world has no bodies" on top of them.
    declared_bodies: usize,
    joints: Vec<rigid::JointPlan>,
    /// Settings from an optional `domain rigid2d <name> { … }` block.
    worlds: BTreeMap<String, RigidWorld>,
    /// True once a `solve rigid(…)` has consumed the bodies.
    bodies_solved: bool,
    /// The chemistry attributes of each declared species, by name.
    chemistry: BTreeMap<String, ChemSpecies>,
    /// Reacting mixtures declared with `domain chemistry <name> { grid: … }`.
    mixtures: BTreeMap<String, MixtureInfo>,
    /// Every `reaction` declaration, kept because the network can only be built once
    /// the species list it indexes into is known — and that belongs to a mixture.
    ///
    /// Consumed when a mixture is solved, which is how an unsolved reaction is noticed.
    reaction_decls: Vec<Decl>,
    /// Every reaction's name, kept whether or not it was consumed.
    ///
    /// A coupling edge naming a reaction is the commonest wrong answer — a reaction is
    /// the thing that releases the heat — and the diagnostic can only say so if the
    /// names outlive the declarations.
    reaction_names: Vec<String>,
    /// Where each domain ended up in the runtime's list, by name — what a coupling
    /// edge resolves a path's first segment to.
    domain_index: BTreeMap<String, usize>,
    /// The areal heat capacity a field declared, J/(m²·K), for deriving a coupling
    /// factor.
    heat_capacity: BTreeMap<String, f64>,
    notes: Vec<String>,
}

/// A reacting mixture declared with `domain chemistry <name> { … }`.
#[derive(Clone, PartialEq, Debug)]
struct MixtureInfo {
    grid: String,
    span: Span,
    solved: bool,
}

/// Settings for a rigid world, from `domain rigid2d <name> { … }`.
#[derive(Clone, Copy, PartialEq, Debug)]
struct RigidWorld {
    gravity: [f64; 2],
    iterations: usize,
    span: Span,
}

impl Default for RigidWorld {
    fn default() -> Self {
        RigidWorld {
            gravity: [0.0, -lattice_units::constants::value::STANDARD_GRAVITY],
            iterations: SolverConfig::default().velocity_iterations,
            span: Span::new(0, 0),
        }
    }
}

impl<'a> Compiler<'a> {
    fn new(file: &'a SourceFile, units: &'a UnitRegistry) -> Compiler<'a> {
        Compiler {
            file,
            units,
            diagnostics: Diagnostics::new(),
            grids: BTreeMap::new(),
            fields: BTreeMap::new(),
            particles: BTreeMap::new(),
            materials: BTreeMap::new(),
            bodies: Vec::new(),
            declared_bodies: 0,
            joints: Vec::new(),
            worlds: BTreeMap::new(),
            bodies_solved: false,
            chemistry: BTreeMap::new(),
            mixtures: BTreeMap::new(),
            reaction_decls: Vec::new(),
            reaction_names: Vec::new(),
            domain_index: BTreeMap::new(),
            heat_capacity: BTreeMap::new(),
            notes: Vec::new(),
        }
    }

    /// An evaluator borrowing the source and registry — deliberately *not* `self`,
    /// so it can coexist with `&mut self.diagnostics`.
    fn evaluator(&self) -> Evaluator<'a> {
        Evaluator::new(self.file, self.units)
    }

    fn error(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    fn run(&mut self, project: &Project) -> Compiled {
        let settings = self.project_settings(project);

        // Grids first: fields refer to them by name.
        for decl in project.declarations_of("grid") {
            self.declare_grid(decl);
        }
        for decl in project.declarations_of("particles") {
            self.declare_particles(decl);
        }
        // Materials, then bodies, then joints: each refers to the one before it by
        // name, and resolving forward references would buy nothing but the ability to
        // write a scene in a confusing order.
        for domain in project.domains().filter(|d| d.family.text == "rigid2d") {
            self.declare_rigid_world(domain);
        }
        for domain in project.domains().filter(|d| d.family.text == "chemistry") {
            self.declare_mixture(domain);
        }
        self.reaction_decls = project.declarations_of("reaction").cloned().collect();
        self.reaction_names =
            self.reaction_decls.iter().map(|decl| decl.name.text.clone()).collect();
        for decl in project.declarations_of("material") {
            self.declare_material(decl);
        }
        for decl in project.declarations_of("body") {
            self.declare_body(decl);
        }
        for decl in project.declarations_of("joint") {
            self.declare_joint(decl);
        }
        for field in project.fields() {
            self.declare_field(field, false);
        }
        for species in project.species() {
            self.declare_field(species, true);
        }

        self.reject_unsupported(project);

        // Lower `solve` statements into solver instances.
        let mut lowering = Lowering::default();
        for solve in project.solves() {
            self.lower_solve(solve, &mut lowering);
        }
        // A quantum domain carries its integrator in its own block (spec §25.2), so
        // there is no `solve` statement to lower it from.
        self.lower_quantum(project, &mut lowering);

        self.warn_about_unsolved_state();

        // An observer reads every buffer, which is what places it after all the
        // advances in the schedule.
        let observers = self.lower_observers(project);
        let all_buffers: Vec<_> = lowering.buffers.buffers().iter().map(|b| b.id).collect();
        for observer in &observers {
            let mut operation =
                Operation::new(format!("observe {}", observer.target), OperationKind::Observe)
                    .costing(0.1);
            for buffer in &all_buffers {
                operation = operation.reading(*buffer);
            }
            lowering.operations.push(operation);
        }

        let Lowering { domains, specs, buffers, operations } = lowering;
        let visuals = self.lower_visuals(project);
        let graph = OperationGraph::build(operations);

        let model = CompiledModel {
            name: project.name.text.clone(),
            dimensions: settings.dimensions,
            fidelity: settings.fidelity,
            precision: settings.precision,
            domains: specs,
            buffers,
            graph,
            observers,
            visuals,
            timestep: settings.timestep,
            duration: settings.duration,
            notes: core::mem::take(&mut self.notes),
        };

        let coupler = self.lower_couples(project);
        Compiled { model, domains, coupler }
    }

    // --- project settings ---------------------------------------------------

    fn project_settings(&mut self, project: &Project) -> ProjectSettings {
        let mut settings = ProjectSettings::default();

        if let Some(setting) = project.setting("dimensions") {
            let evaluator = self.evaluator();
            if let Some(value) =
                evaluator.count(&setting.value, "`dimensions`", &mut self.diagnostics)
            {
                if value != 2 {
                    self.error(
                        Diagnostic::error("only two-dimensional models are supported")
                            .with_code("E0900")
                            .at(setting.value.span, format!("this model declares {value}"))
                            .note(
                                "spec P9 makes 2D the first target and keeps the architecture \
                                 dimension-agnostic; a 3D backend is not part of any current \
                                 milestone",
                            ),
                    );
                }
                settings.dimensions = value as u32;
            }
        }

        if let Some(setting) = project.setting("fidelity") {
            match setting.value.as_name() {
                Some("interactive") => settings.fidelity = FidelityProfile::Interactive,
                Some("engineering_2d") => settings.fidelity = FidelityProfile::Engineering2d,
                Some("molecular") | Some("kinetic") => {
                    settings.fidelity = FidelityProfile::MolecularKinetic
                }
                Some("research") => settings.fidelity = FidelityProfile::ResearchCoupling,
                Some("external_quantum") => {
                    self.error(
                        Diagnostic::error("the external quantum fidelity profile has no backend yet")
                            .with_code("E0900")
                            .at(setting.value.span, "F4 requires an external solver adapter")
                            .note("external quantum-chemistry adapters arrive in milestone M8"),
                    );
                }
                other => self.error(
                    Diagnostic::error("unknown fidelity profile")
                        .with_code("E0204")
                        .at(setting.value.span, format!("`{}`", other.unwrap_or("?")))
                        .help(
                            "one of: interactive, engineering_2d, molecular, research, \
                             external_quantum",
                        ),
                ),
            }
        }

        if let Some(setting) = project.setting("precision") {
            match setting.value.as_name() {
                Some("accurate64") | Some("mixed") => {
                    // `mixed` means 32-bit state with 64-bit reductions, which needs a
                    // GPU backend to be meaningful. On the CPU path it is 64-bit.
                    if setting.value.as_name() == Some("mixed") {
                        self.notes.push(
                            "precision `mixed` was compiled as `accurate64`: the mixed mode \
                             needs the portable GPU backend from milestone M4"
                                .to_string(),
                        );
                    }
                    settings.precision = Precision::Accurate64;
                }
                Some("deterministic64") => settings.precision = Precision::Deterministic64,
                Some("fast32") => {
                    self.error(
                        Diagnostic::error("32-bit execution needs a GPU backend")
                            .with_code("E0900")
                            .at(setting.value.span, "`fast32` is not available on the CPU path")
                            .note("portable GPU execution arrives in milestone M4")
                            .help("use `accurate64` for now"),
                    );
                }
                other => self.error(
                    Diagnostic::error("unknown precision mode")
                        .with_code("E0204")
                        .at(setting.value.span, format!("`{}`", other.unwrap_or("?")))
                        .help("one of: accurate64, deterministic64, mixed, fast32"),
                ),
            }
        }

        for key in ["duration", "run"] {
            if let Some(setting) = project.setting(key) {
                let evaluator = self.evaluator();
                settings.duration = evaluator.require(
                    &setting.value,
                    Dimension::TIME,
                    &format!("`{key}`"),
                    &mut self.diagnostics,
                );
            }
        }

        for key in ["timestep", "dt"] {
            if let Some(setting) = project.setting(key) {
                let evaluator = self.evaluator();
                settings.timestep = evaluator.require(
                    &setting.value,
                    Dimension::TIME,
                    &format!("`{key}`"),
                    &mut self.diagnostics,
                );
            }
        }

        // Anything else at the top level is a typo worth reporting.
        const KNOWN: &[&str] =
            &["dimensions", "fidelity", "precision", "duration", "run", "timestep", "dt", "seed"];
        for setting in project.settings() {
            if !KNOWN.contains(&setting.key.text.as_str()) {
                self.error(
                    Diagnostic::warning(format!("unknown project setting `{}`", setting.key.text))
                        .with_code("W0301")
                        .at(setting.key.span, "ignored")
                        .help(format!("known settings: {}", KNOWN.join(", "))),
                );
            }
        }

        settings
    }

    // --- declarations -------------------------------------------------------

    fn declare_grid(&mut self, decl: &Decl) {
        if let Some(existing) = self.grids.get(&decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("grid `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here")
                    .help("give one of them a different name, or remove the duplicate"),
            );
            return;
        }

        self.check_settings(decl, &["size", "extent", "origin"]);

        let evaluator = self.evaluator();
        let size = decl.setting("size").and_then(|setting| {
            let pair = evaluator.fixed_list(
                &setting.value,
                2,
                "a grid `size`",
                &mut self.diagnostics,
            )?;
            let nx = evaluator.count(pair[0], "the cell count (x)", &mut self.diagnostics)?;
            let ny = evaluator.count(pair[1], "the cell count (y)", &mut self.diagnostics)?;
            Some([nx, ny])
        });
        let extent = decl.setting("extent").and_then(|setting| {
            evaluator.pair(&setting.value, Dimension::LENGTH, "a grid `extent`", &mut self.diagnostics)
        });
        let origin = decl
            .setting("origin")
            .and_then(|setting| {
                evaluator.pair(&setting.value, Dimension::LENGTH, "a grid `origin`", &mut self.diagnostics)
            })
            .unwrap_or([0.0, 0.0]);

        let (Some(size), Some(extent)) = (size, extent) else {
            for (key, what) in [("size", "cell counts"), ("extent", "physical size")] {
                if decl.setting(key).is_none() {
                    self.error(
                        Diagnostic::error(format!("grid `{}` needs a `{key}` setting", decl.name.text))
                            .with_code("E0203")
                            .at(decl.name.span, format!("missing the {what}"))
                            .help(format!("add `{key}: […];` inside the braces")),
                    );
                }
            }
            return;
        };

        if size[0] == 0 || size[1] == 0 {
            self.error(
                Diagnostic::error("a grid must have at least one cell on each axis")
                    .with_code("E0405")
                    .at(decl.name.span, format!("size is {}x{}", size[0], size[1])),
            );
            return;
        }
        if !(extent[0] > 0.0 && extent[1] > 0.0) {
            self.error(
                Diagnostic::error("a grid extent must be positive on each axis")
                    .with_code("E0405")
                    .at(decl.name.span, format!("extent is {extent:?} m")),
            );
            return;
        }

        self.grids.insert(
            decl.name.text.clone(),
            GridInfo {
                grid: Grid2d::with_origin(size[0], size[1], extent, origin),
                span: decl.name.span,
            },
        );
    }

    fn declare_field(&mut self, decl: &FieldDecl, is_species: bool) {
        let what = if is_species { "species" } else { "field" };

        if let Some(existing) = self.fields.get(&decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("`{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here")
                    .help("fields and species share one namespace; rename one of them"),
            );
            return;
        }

        let Some(grid_name) = &decl.grid else {
            self.error(
                Diagnostic::error(format!("{what} `{}` must name a grid", decl.name.text))
                    .with_code("E0203")
                    .at(decl.name.span, "no grid given")
                    .help(format!("write `{what} {} on <grid> = …;`", decl.name.text)),
            );
            return;
        };
        if !self.grids.contains_key(&grid_name.text) {
            let known: Vec<&str> = self.grids.keys().map(String::as_str).collect();
            let mut diagnostic =
                Diagnostic::error(format!("there is no grid called `{}`", grid_name.text))
                    .with_code("E0202")
                    .at(grid_name.span, "unknown grid");
            if !known.is_empty() {
                diagnostic = diagnostic.help(format!("declared grids: {}", known.join(", ")));
            } else {
                diagnostic = diagnostic.help("declare one with `grid <name> { size: …; extent: …; }`");
            }
            self.error(diagnostic);
            return;
        }

        let Some(initial) = &decl.initial else {
            self.error(
                Diagnostic::error(format!("{what} `{}` needs an initial value", decl.name.text))
                    .with_code("E0203")
                    .at(decl.name.span, "no initial value")
                    .help("add `= <value>` — the field's dimension is inferred from it"),
            );
            return;
        };

        self.check_field_settings(decl);

        // The dimension comes from the initializer, so `= 298 kelvin` makes a
        // temperature field without the user restating it.
        let evaluator = self.evaluator();
        let Some(dimension) = builtins::infer_initializer_dimension(initial, &evaluator) else {
            // Re-run with diagnostics so the user sees why it could not be read.
            builtins::initializer(
                initial,
                Dimension::DIMENSIONLESS,
                "the initial value",
                &evaluator,
                &mut self.diagnostics,
            );
            if !self.diagnostics.has_errors() {
                self.error(
                    Diagnostic::error(format!(
                        "cannot infer the dimension of {what} `{}`",
                        decl.name.text
                    ))
                    .with_code("E0400")
                    .at(initial.span, "the initial value has no clear dimension"),
                );
            }
            return;
        };

        let Some(initializer) = builtins::initializer(
            initial,
            dimension,
            "the initial value",
            &evaluator,
            &mut self.diagnostics,
        ) else {
            return;
        };

        let diffusivity = decl.setting("diffusivity").and_then(|setting| {
            evaluator.require(
                &setting.value,
                Dimension::DIFFUSIVITY,
                "`diffusivity`",
                &mut self.diagnostics,
            )
        });
        if let Some(value) = diffusivity
            && value < 0.0
        {
            self.error(
                Diagnostic::error("diffusivity must not be negative")
                    .with_code("E0405")
                    .at(decl.setting("diffusivity").unwrap().value.span, format!("{value} m^2/s"))
                    .note(
                        "a negative diffusivity runs the heat equation backwards, which \
                         amplifies every short-wavelength mode without bound",
                    ),
            );
        }

        let boundaries = self.field_boundaries(decl, dimension);

        // A material property the solver never reads, kept so a coupling edge can
        // derive its own conversion rather than taking a number on trust.
        if let Some(setting) = decl.setting("heat_capacity") {
            let expected = Dimension::ENERGY
                .try_div(Dimension::AREA)
                .and_then(|d| d.try_div(Dimension::TEMPERATURE))
                .expect("energy per area per kelvin is representable");
            let evaluator = self.evaluator();
            if let Some(capacity) =
                evaluator.require(&setting.value, expected, "`heat_capacity`", &mut self.diagnostics)
            {
                if capacity > 0.0 {
                    self.heat_capacity.insert(decl.name.text.clone(), capacity);
                } else {
                    self.error(
                        Diagnostic::error("a heat capacity must be positive")
                            .with_code("E0405")
                            .at(setting.value.span, format!("this is {capacity}"))
                            .note("a negative one would make heating cool the field"),
                    );
                }
            }
        }

        if is_species {
            let evaluator = self.evaluator();
            let chemistry = chemistry::species(
                &decl.name.text,
                &decl.settings,
                &evaluator,
                &mut self.diagnostics,
            );
            chemistry::check_molar_mass(&chemistry, decl.name.span, &mut self.diagnostics);
            self.chemistry.insert(decl.name.text.clone(), chemistry);
        }

        self.fields.insert(
            decl.name.text.clone(),
            FieldInfo {
                name: decl.name.text.clone(),
                grid: grid_name.text.clone(),
                dimension,
                initializer,
                diffusivity,
                boundaries,
                is_species,
                span: decl.name.span,
                solved: false,
            },
        );
    }

    fn field_boundaries(&mut self, decl: &FieldDecl, dimension: Dimension) -> BoundarySet {
        let evaluator = self.evaluator();
        let mut set = BoundarySet::INSULATED;

        if let Some(setting) = decl.setting("boundary")
            && let Some(boundary) =
                builtins::boundary(&setting.value, dimension, &evaluator, &mut self.diagnostics)
        {
            set = BoundarySet::uniform(boundary);
        }

        for (key, side) in [
            ("boundary_left", Side::Left),
            ("boundary_right", Side::Right),
            ("boundary_bottom", Side::Bottom),
            ("boundary_top", Side::Top),
        ] {
            if let Some(setting) = decl.setting(key)
                && let Some(boundary) =
                    builtins::boundary(&setting.value, dimension, &evaluator, &mut self.diagnostics)
            {
                set.set(side, boundary);
            }
        }

        // Spec §8.4 step 8 asks for boundary completeness to be validated. An axis
        // that is periodic on one edge and not the other has no meaning.
        if let Err(error) = set.validate() {
            let span = decl
                .setting("boundary")
                .map_or(decl.name.span, |setting| setting.value.span);
            self.error(
                Diagnostic::error(error.to_string())
                    .with_code("E0405")
                    .at(span, "inconsistent boundaries")
                    .help("declare `periodic` on both edges of an axis, or neither"),
            );
            // Fall back to something constructible. The error above already stops the
            // run; letting an invalid set reach `HeatDomain::with_boundaries` would
            // turn a reported diagnostic into a panic.
            return BoundarySet::INSULATED;
        }
        set
    }

    fn declare_particles(&mut self, decl: &Decl) {
        if let Some(existing) = self.particles.get(&decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("particle set `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here")
                    .help("give one of them a different name, or remove the duplicate"),
            );
            return;
        }

        self.check_settings(
            decl,
            &[
                "count", "region", "origin", "mass", "radius", "boundary", "spacing", "speed",
                "temperature", "seed", "force", "skin", "thermostat", "bonds", "angles",
                "analysis", "layout",
            ],
        );

        let evaluator = self.evaluator();
        let Some(count) = decl.setting("count").and_then(|setting| {
            evaluator.count(&setting.value, "`count`", &mut self.diagnostics)
        }) else {
            self.error(
                Diagnostic::error(format!("particle set `{}` needs a `count`", decl.name.text))
                    .with_code("E0203")
                    .at(decl.name.span, "missing `count`")
                    .help("add `count: 256;` inside the braces"),
            );
            return;
        };

        let region = decl.setting("region").and_then(|setting| {
            evaluator.pair(&setting.value, Dimension::LENGTH, "`region`", &mut self.diagnostics)
        });
        let origin = decl
            .setting("origin")
            .and_then(|setting| {
                evaluator.pair(&setting.value, Dimension::LENGTH, "`origin`", &mut self.diagnostics)
            })
            .unwrap_or([0.0, 0.0]);
        let mass = decl
            .setting("mass")
            .and_then(|setting| {
                evaluator.require(&setting.value, Dimension::MASS, "`mass`", &mut self.diagnostics)
            })
            .unwrap_or(1.0);
        let radius = decl
            .setting("radius")
            .and_then(|setting| {
                evaluator.require(&setting.value, Dimension::LENGTH, "`radius`", &mut self.diagnostics)
            })
            .unwrap_or(0.0);
        let spacing = decl.setting("spacing").and_then(|setting| {
            evaluator.require(&setting.value, Dimension::LENGTH, "`spacing`", &mut self.diagnostics)
        });
        let speed = decl
            .setting("speed")
            .and_then(|setting| {
                evaluator.require(&setting.value, Dimension::VELOCITY, "`speed`", &mut self.diagnostics)
            })
            .unwrap_or(0.0);
        let seed = decl
            .setting("seed")
            .and_then(|setting| evaluator.count(&setting.value, "`seed`", &mut self.diagnostics))
            .unwrap_or(0) as u64;

        let boundary = match decl.setting("boundary").and_then(|s| s.value.as_name()) {
            None => ParticleBoundary::Open,
            Some("periodic") => ParticleBoundary::Periodic,
            Some("reflective") => ParticleBoundary::Reflective,
            Some("open") => ParticleBoundary::Open,
            Some(other) => {
                let span = decl.setting("boundary").unwrap().value.span;
                self.error(
                    Diagnostic::error(format!("`{other}` is not a particle boundary"))
                        .with_code("E0204")
                        .at(span, "unknown boundary")
                        .help("one of: periodic, reflective, open"),
                );
                ParticleBoundary::Open
            }
        };

        // `force:` may appear more than once, which is how a scene combines gravity
        // with drag.
        let mut forces = Vec::new();
        for setting in decl.settings.iter().filter(|s| s.key.text == "force") {
            if let Some(force) = builtins::force(&setting.value, &evaluator, &mut self.diagnostics) {
                forces.push(force);
            }
        }

        if mass <= 0.0 {
            self.error(
                Diagnostic::error("particle mass must be positive")
                    .with_code("E0405")
                    .at(decl.name.span, format!("mass is {mass} kg")),
            );
        }

        // --- the molecular settings (spec §12.4) ---
        let temperature = decl.setting("temperature").and_then(|setting| {
            let value = evaluator.require(
                &setting.value,
                Dimension::TEMPERATURE,
                "the initial temperature",
                &mut self.diagnostics,
            )?;
            if value < 0.0 || !value.is_finite() {
                self.error(
                    Diagnostic::error("the initial temperature cannot be negative")
                        .with_code("E0405")
                        .at(setting.value.span, format!("{value} K")),
                );
                return None;
            }
            Some(value)
        });
        if let (Some(temperature), Some(speed_setting)) = (decl.setting("temperature"), decl.setting("speed")) {
            self.error(
                Diagnostic::error("`temperature` and `speed` both set the initial velocities")
                    .with_code("E0209")
                    .at(temperature.key.span, "sets a Maxwell distribution")
                    .also(speed_setting.key.span, "sets a velocity spread")
                    .help("keep one: `temperature:` for a thermal start, `speed:` for a plain spread"),
            );
        }
        if let (Some(_), Some(setting)) = (temperature, decl.setting("temperature"))
            && count < 2
        {
            self.error(
                Diagnostic::error("a temperature needs at least two particles")
                    .with_code("E0405")
                    .at(setting.value.span, format!("the set has {count}"))
                    .note("temperature is defined over 2N - 2 thermal degrees of freedom, and one \
                           particle has none"),
            );
        }
        let skin = decl.setting("skin").and_then(|setting| {
            let value =
                evaluator.require(&setting.value, Dimension::LENGTH, "the Verlet skin", &mut self.diagnostics)?;
            if value <= 0.0 || !value.is_finite() {
                self.error(
                    Diagnostic::error("the Verlet skin must be positive")
                        .with_code("E0405")
                        .at(setting.value.span, format!("{value} m"))
                        .help("leave `skin` out to rebuild the cell list every step"),
                );
                return None;
            }
            Some(value)
        });
        let thermostat = decl
            .setting("thermostat")
            .and_then(|setting| molecular::thermostat(&setting.value, &evaluator, &mut self.diagnostics));
        let mut bonds = Vec::new();
        for setting in decl.settings.iter().filter(|s| s.key.text == "bonds") {
            if let Some(bond) = molecular::bonds(&setting.value, &evaluator, &mut self.diagnostics) {
                bonds.push(bond);
            }
        }
        let mut angles = Vec::new();
        for setting in decl.settings.iter().filter(|s| s.key.text == "angles") {
            if let Some(angle) = molecular::angles(&setting.value, &evaluator, &mut self.diagnostics) {
                angles.push(angle);
            }
        }
        let mut analyses = Vec::new();
        for setting in decl.settings.iter().filter(|s| s.key.text == "analysis") {
            if let Some(request) = molecular::analysis(&setting.value, &evaluator, &mut self.diagnostics) {
                analyses.push(request);
            }
        }
        let layout = match decl.setting("layout").and_then(|s| s.value.as_name()) {
            None | Some("lattice") => Layout::Lattice,
            Some("serpentine") => Layout::Serpentine,
            Some(other) => {
                let span = decl.setting("layout").unwrap().value.span;
                self.error(
                    Diagnostic::error(format!("`{other}` is not a particle layout"))
                        .with_code("E0208")
                        .at(span, "unknown layout")
                        .help("one of: lattice, serpentine"),
                );
                Layout::Lattice
            }
        };
        if bonds.iter().any(|b| b.length.is_none()) && spacing.is_none() {
            let span = decl.settings.iter().find(|s| s.key.text == "bonds").map_or(decl.name.span, |s| s.value.span);
            self.error(
                Diagnostic::error("a bond without a `length` takes the placement `spacing`, and none is declared")
                    .with_code("E0203")
                    .at(span, "no rest length")
                    .help("add `length=… meter` to the bond, or a `spacing:` to the set"),
            );
        }

        self.particles.insert(
            decl.name.text.clone(),
            ParticlesInfo {
                name: decl.name.text.clone(),
                count,
                origin,
                region,
                mass,
                radius,
                boundary,
                spacing,
                speed,
                temperature,
                seed,
                forces,
                skin,
                thermostat,
                bonds,
                angles,
                analyses,
                layout,
                span: decl.name.span,
                solved: false,
            },
        );
    }

    /// Report settings a declaration does not understand.
    fn check_settings(&mut self, decl: &Decl, allowed: &[&str]) {
        for setting in &decl.settings {
            if !allowed.contains(&setting.key.text.as_str()) {
                self.error(
                    Diagnostic::error(format!(
                        "`{}` has no setting called `{}`",
                        decl.kind.text, setting.key.text
                    ))
                    .with_code("E0204")
                    .at(setting.key.span, "unknown setting")
                    .help(format!("`{}` accepts: {}", decl.kind.text, allowed.join(", "))),
                );
            }
        }
    }

    fn check_field_settings(&mut self, decl: &FieldDecl) {
        const ALLOWED: &[&str] = &[
            "diffusivity",
            "boundary",
            "boundary_left",
            "boundary_right",
            "boundary_bottom",
            "boundary_top",
            // A material property rather than a solver setting: the heat solver never
            // reads it. It is here so a coupling edge can derive its own conversion
            // factor instead of taking a magic number (spec §14.1).
            "heat_capacity",
            // Chemistry, meaningful on a `species`.
            "formula",
            "charge",
            "molar_mass",
        ];
        for setting in &decl.settings {
            if !ALLOWED.contains(&setting.key.text.as_str()) {
                self.error(
                    Diagnostic::error(format!("a field has no setting called `{}`", setting.key.text))
                        .with_code("E0204")
                        .at(setting.key.span, "unknown setting")
                        .help(format!("fields accept: {}", ALLOWED.join(", "))),
                );
            }
        }
    }

    // --- unsupported constructs ---------------------------------------------

    fn reject_unsupported(&mut self, project: &Project) {
        const KNOWN_KINDS: &[&str] = &[
            "grid", "particles", "material", "body", "joint", "reaction", "potential", "wavepacket",
            "detector",
        ];
        // Declaration kinds the spec names whose module does not exist yet, as
        // `(kind, milestone, what)`. Empty since M5; kept so the next one has a place.
        const PLANNED: &[(&str, &str, &str)] = &[];

        for decl in project.declarations() {
            let kind = decl.kind.text.as_str();
            if KNOWN_KINDS.contains(&kind) {
                continue;
            }
            match PLANNED.iter().find(|(name, _, _)| *name == kind) {
                Some((_, milestone, what)) => self.error(
                    Diagnostic::error(format!("{what} are not implemented yet"))
                        .with_code("E0900")
                        .at(decl.kind.span, format!("`{kind}` needs the {milestone} module"))
                        .note(format!(
                            "this is milestone {milestone}; the declaration parsed and its \
                             settings were checked, but nothing can execute it"
                        )),
                ),
                None => self.error(
                    Diagnostic::error(format!("`{kind}` is not a known declaration"))
                        .with_code("E0205")
                        .at(decl.kind.span, "unknown declaration kind")
                        .help(format!("available now: {}", KNOWN_KINDS.join(", "))),
                ),
            }
        }

        for domain in project.domains() {
            // These were handled by their own declaration passes.
            if matches!(domain.family.text.as_str(), "rigid2d" | "chemistry" | "quantum2d") {
                continue;
            }
            let milestone = match domain.family.text.as_str() {
                "fluid2d" => "M2",
                _ => {
                    self.error(
                        Diagnostic::error(format!(
                            "`{}` is not a known solver family",
                            domain.family.text
                        ))
                        .with_code("E0205")
                        .at(domain.family.span, "unknown family")
                        .help(
                            "declare grids and particle sets directly; explicit `domain` blocks \
                             are for solver families that need their own configuration",
                        ),
                    );
                    continue;
                }
            };
            self.error(
                Diagnostic::error(format!(
                    "the `{}` solver family is not implemented yet",
                    domain.family.text
                ))
                .with_code("E0900")
                .at(domain.family.span, format!("needs the {milestone} module"))
                .note(format!("this is milestone {milestone}")),
            );
        }
    }

    fn warn_about_unsolved_state(&mut self) {
        let unsolved_fields: Vec<(String, Span)> = self
            .fields
            .values()
            .filter(|field| !field.solved)
            .map(|field| (field.name.clone(), field.span))
            .collect();
        for (name, span) in unsolved_fields {
            self.error(
                Diagnostic::warning(format!("`{name}` is declared but never solved"))
                    .with_code("W0302")
                    .at(span, "no `solve` statement names this")
                    .note("it will hold its initial value for the whole run")
                    .help(format!("add `solve diffusion({name}) with crank_nicolson(dt=…);`")),
            );
        }

        if !self.reaction_decls.is_empty() {
            let span = self.reaction_decls[0].name.span;
            let count = self.reaction_decls.len();
            self.error(
                Diagnostic::warning(format!(
                    "{count} reactions are declared but no `solve reactions` names a mixture"
                ))
                .with_code("W0306")
                .at(span, "declared here")
                .note("nothing will react; the species will only diffuse")
                .help(
                    "add `domain chemistry mixture { grid: … }` and \
                     `solve reactions(mixture) with strang(dt = …);`",
                ),
            );
        }

        if !self.bodies.is_empty() && !self.bodies_solved {
            let span = self.bodies[0].span;
            let count = self.bodies.len();
            self.error(
                Diagnostic::warning(format!(
                    "{count} bodies are declared but no `solve rigid` names a world"
                ))
                .with_code("W0303")
                .at(span, "declared here")
                .note("they will sit where they were placed for the whole run")
                .help("add `solve rigid(world) with sequential_impulse(dt = 0.008 second);`"),
            );
        }

        let unsolved_particles: Vec<(String, Span)> = self
            .particles
            .values()
            .filter(|set| !set.solved)
            .map(|set| (set.name.clone(), set.span))
            .collect();
        for (name, span) in unsolved_particles {
            self.error(
                Diagnostic::warning(format!("particle set `{name}` is declared but never solved"))
                    .with_code("W0302")
                    .at(span, "no `solve` statement names this")
                    .help(format!("add `solve dynamics({name}) with velocity_verlet(dt=…);`")),
            );
        }
    }

    // --- chemistry -----------------------------------------------------------

    fn declare_mixture(&mut self, decl: &lattice_syntax::DomainDecl) {
        if let Some(existing) = self.mixtures.get(&decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("mixture `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here"),
            );
            return;
        }

        const ALLOWED: &[&str] = &["grid", "temperature"];
        for setting in &decl.settings {
            if !ALLOWED.contains(&setting.key.text.as_str()) {
                self.error(
                    Diagnostic::error(format!(
                        "a chemistry domain has no setting called `{}`",
                        setting.key.text
                    ))
                    .with_code("E0204")
                    .at(setting.key.span, "unknown setting")
                    .help(format!("chemistry accepts: {}", ALLOWED.join(", "))),
                );
            }
        }

        let evaluator = self.evaluator();
        let Some(grid) = decl.setting("grid").and_then(|s| evaluator.as_name(&s.value)) else {
            self.error(
                Diagnostic::error(format!("mixture `{}` needs a `grid`", decl.name.text))
                    .with_code("E0203")
                    .at(decl.name.span, "missing `grid`")
                    .note("every species in a mixture reacts cell by cell on one grid")
                    .help("add `grid: chamber;`"),
            );
            return;
        };
        if !self.grids.contains_key(grid) {
            let known: Vec<&str> = self.grids.keys().map(String::as_str).collect();
            self.error(
                Diagnostic::error(format!("there is no grid called `{grid}`"))
                    .with_code("E0202")
                    .at(decl.setting("grid").unwrap().value.span, "unknown grid")
                    .help(format!("declared grids: {}", known.join(", "))),
            );
            return;
        }

        self.mixtures.insert(
            decl.name.text.clone(),
            MixtureInfo { grid: grid.to_string(), span: decl.name.span, solved: false },
        );
    }

    /// Build the reaction network from every `reaction` declaration.
    ///
    /// Returned rather than stored, because a network is only meaningful alongside the
    /// species list it indexes into — and that list is the mixture's.
    fn build_network(&mut self, grid: &str) -> (ReactionNetwork, Vec<String>) {
        let mut network = ReactionNetwork::new();
        let mut names = Vec::new();
        let mut index_of: BTreeMap<String, usize> = BTreeMap::new();

        // Species join in declaration order, so the report and the concentration
        // vectors can be read against the source by counting down the file.
        // `self.fields` is a BTreeMap, so this is already in name order — which is a
        // total order and therefore reproducible, which is what FR-011 needs.
        let species: Vec<&FieldInfo> =
            self.fields.values().filter(|f| f.is_species && f.grid == grid).collect();
        for field in species {
            let chemistry = self
                .chemistry
                .get(&field.name)
                .cloned()
                .unwrap_or_else(|| ChemSpecies::new(field.name.clone()));
            let index = network
                .add_species(chemistry.with_diffusion(field.diffusivity.unwrap_or(0.0)));
            index_of.insert(field.name.clone(), index);
            names.push(field.name.clone());
        }

        let evaluator = self.evaluator();
        let declarations = core::mem::take(&mut self.reaction_decls);
        for decl in &declarations {
            self.check_settings(decl, chemistry::REACTION_SETTINGS);
            if let Some(reaction) =
                chemistry::reaction(decl, &index_of, &evaluator, &mut self.diagnostics)
            {
                network.add_reaction(reaction);
            }
        }

        // An unbalanced reaction destroys atoms in every cell, every step, and the only
        // symptom is a conservation check failing several layers away. When every
        // species states its composition there is no reason to let it through.
        for (reaction, balance) in network.imbalances() {
            let span = declarations
                .iter()
                .find(|d| d.name.text == reaction.name)
                .map_or_else(|| Span::new(0, 0), |d| d.name.span);
            match balance {
                lattice_domain_chemistry::Balance::Unknown { .. } => self.error(
                    Diagnostic::warning(format!(
                        "reaction `{}` cannot be checked for balance",
                        reaction.name
                    ))
                    .with_code("W0305")
                    .at(span, balance.describe())
                    .note(
                        "a reaction whose atoms cannot be counted may destroy mass without \
                         anything noticing until a conservation check fails",
                    )
                    .help("give every species a `formula:`"),
                ),
                _ => self.error(
                    Diagnostic::error(format!("reaction `{}` does not balance", reaction.name))
                        .with_code("E0212")
                        .at(span, balance.describe())
                        .note(
                            "an unbalanced reaction integrates perfectly happily and destroys \
                             mass every step; the only symptom is a ledger that will not close",
                        )
                        .help("check the stoichiometric coefficients"),
                ),
            }
        }

        (network, names)
    }

    fn lower_reactions(&mut self, solve: &SolveStmt, method: &builtins::Method, out: &mut Lowering) {
        let Some(target) = solve.targets.first() else { return };
        let Some(name) = target.value.as_name() else {
            self.error(
                Diagnostic::error("a chemistry solver target must be a mixture name")
                    .with_code("E0401")
                    .at(target.value.span, "not a name")
                    .help("write `solve reactions(mixture) with strang(dt = 0.5 second);`"),
            );
            return;
        };
        match method.name.as_str() {
            "strang" | "splitting" | "operator_splitting" => {}
            other => {
                self.error(
                    Diagnostic::error(format!("`{other}` is not a method for reactions"))
                        .with_code("E0207")
                        .at(method.span, "unknown method")
                        .help("use `strang(dt = …)`"),
                );
                return;
            }
        }

        let Some(mixture) = self.mixtures.get(name).cloned() else {
            let known: Vec<&str> = self.mixtures.keys().map(String::as_str).collect();
            self.error(
                Diagnostic::error(format!("there is no mixture called `{name}`"))
                    .with_code("E0202")
                    .at(target.value.span, "unknown mixture")
                    .help(if known.is_empty() {
                        "declare one with `domain chemistry mixture { grid: chamber; }`".to_string()
                    } else {
                        format!("declared mixtures: {}", known.join(", "))
                    }),
            );
            return;
        };
        if let Some(entry) = self.mixtures.get_mut(name) {
            entry.solved = true;
        }

        let (network, species_names) = self.build_network(&mixture.grid);
        if network.is_empty() {
            self.error(
                Diagnostic::error(format!("mixture `{name}` has no species"))
                    .with_code("E0203")
                    .at(mixture.span, "nothing to solve")
                    .help(format!("declare one with `species A on {} = …;`", mixture.grid)),
            );
            return;
        }

        let grid = self.grids[&mixture.grid].grid;
        let mut domain = ReactingMixture::new(name.to_string(), grid, network);
        for (index, species) in species_names.iter().enumerate() {
            let field = self.fields[species].clone();
            if let Some(entry) = self.fields.get_mut(species) {
                entry.solved = true;
            }
            domain.set_boundaries(index, field.boundaries);
            let initializer = field.initializer.clone();
            if let Some(concentration) = domain.concentration_mut(index) {
                concentration.init_from_position(&grid, |p| initializer.sample(p, &grid));
            }
        }

        let buffer = out.buffers.allocate(
            name.to_string(),
            BufferKind::ScalarField { nx: grid.nx(), ny: grid.ny(), halo: 1 },
        );
        out.buffers.require_scratch(grid.cell_count());

        let id = DomainId::from_index(out.specs.len() as u32);
        out.operations.push(
            Operation::new(format!("prepare {name}"), OperationKind::Prepare)
                .in_domain(id)
                .writing(buffer)
                .costing(0.1),
        );
        out.operations.push(
            Operation::new(format!("advance {name}"), OperationKind::Advance)
                .in_domain(id)
                .reading(buffer)
                .writing(buffer)
                // A reacting cell costs a Runge-Kutta solve per species, and a diffusing
                // one costs a linear solve; both scale with the cell count.
                .costing(grid.cell_count() as f64 * (20.0 + 4.0 * species_names.len() as f64)),
        );

        let reactions = domain.network().reactions().len();
        let equations: Vec<String> = domain
            .network()
            .reactions()
            .iter()
            .map(|reaction| reaction.equation(domain.network().species()))
            .collect();
        for (reaction, equation) in domain.network().reactions().iter().zip(&equations) {
            self.notes.push(format!(
                "reaction `{}`: {equation}, {}",
                reaction.name,
                reaction.rate.describe(reaction.order())
            ));
        }

        self.domain_index.insert(name.to_string(), out.specs.len());
        out.specs.push(DomainSpec {
            id,
            name: name.to_string(),
            family: "chemistry.reaction_diffusion[strang]".to_string(),
            summary: format!(
                "{} species on `{}`, {reactions} reactions: {}",
                species_names.len(),
                mixture.grid,
                if equations.is_empty() { "none".to_string() } else { equations.join("; ") }
            ),
            buffers: vec![buffer],
            contract: Some(domain.contract()),
        });
        if let Some(dt) = method.timestep {
            let _ = dt;
        }
        out.domains.push(Box::new(domain));
    }

    /// Resolve every `couple` statement into an edge, deriving its unit conversion.
    ///
    /// # Why the compiler computes the factor rather than reading one
    ///
    /// A coupling edge's mapping is where two domains' units meet, and it is the one
    /// number in a coupled model that nothing else checks. Writing it by hand means
    /// writing `0.0000025` in a model file and hoping; six months later nobody knows
    /// whether it was a heat capacity or a thickness.
    ///
    /// The compiler already knows both sides. A reaction publishes `W/m²`; a
    /// temperature field consumes `K/s`. The ratio of those has the dimensions of a
    /// reciprocal areal heat capacity, and the field can *declare* that as a material
    /// property. So the model says what the material is, and the arithmetic that turns
    /// it into a coupling factor happens where it can be dimension-checked.
    fn lower_couples(&mut self, project: &Project) -> Coupler {
        let mut coupler = Coupler::new();
        for couple in project.couples() {
            let Some(edge) = self.lower_couple(couple) else { continue };
            self.notes.push(format!("couple {}", edge.describe()));
            coupler.add(edge);
        }
        coupler
    }

    fn lower_couple(&mut self, couple: &lattice_syntax::CoupleStmt) -> Option<CouplingEdge> {
        let source = self.resolve_port(&couple.source, "source")?;
        let target = self.resolve_port(&couple.target, "target")?;

        let name = format!("{}_to_{}", couple.source.root().text, couple.target.root().text);
        let mut edge =
            CouplingEdge::new(name, source.reference.clone(), target.reference.clone());

        // The two units either match, in which case nothing is needed, or they do not,
        // in which case the model has to have said what converts them.
        if source.dimension == target.dimension {
            edge = edge.with_mapping(Mapping::Direct);
        } else {
            edge = edge.with_mapping(self.conversion(&source, &target, couple.span)?);
        }

        if let Some(conserve) = &couple.conserve {
            match invariant_named(&conserve.text) {
                Some(quantity) => edge = edge.carrying(quantity),
                None => self.error(
                    Diagnostic::error(format!("`{}` is not a conserved quantity", conserve.text))
                        .with_code("E0208")
                        .at(conserve.span, "unknown quantity")
                        .help("one of: energy, mass, momentum_x, momentum_y, charge, amount"),
                ),
            }
        }
        Some(edge)
    }

    /// Resolve `domain.port` to an index, a port name, and the dimension it carries.
    fn resolve_port(&mut self, path: &lattice_syntax::Path, role: &str) -> Option<ResolvedPort> {
        let root = path.root();
        let Some(&index) = self.domain_index.get(&root.text) else {
            let known: Vec<&str> = self.domain_index.keys().map(String::as_str).collect();
            // A reaction is the commonest wrong answer here, because it is the thing
            // that releases the heat. It is not a domain: the mixture that contains it
            // is, and that is what publishes the port.
            let is_reaction = self.reaction_names.contains(&root.text);
            let diagnostic = Diagnostic::error(format!(
                "there is no domain called `{}`",
                root.text
            ))
            .with_code("E0202")
            .at(root.span, format!("unknown {role} domain"));
            let diagnostic = if is_reaction {
                diagnostic
                    .note(format!(
                        "`{}` is a reaction, and a reaction is not a domain — the mixture \
                         containing it is what publishes `heat_release`",
                        root.text
                    ))
                    .help(if known.is_empty() {
                        "declare one with `domain chemistry mixture { grid: … }` and solve it"
                            .to_string()
                    } else {
                        format!("couple from one of: {}", known.join(", "))
                    })
            } else {
                diagnostic.help(if known.is_empty() {
                    "a coupling connects two solved domains; nothing is solved yet".to_string()
                } else {
                    format!("solved domains: {}", known.join(", "))
                })
            };
            self.error(diagnostic);
            return None;
        };
        let Some(port) = path.tail().first() else {
            self.error(
                Diagnostic::error("a coupling endpoint names a domain and a port")
                    .with_code("E0401")
                    .at(path.span, "no port named")
                    .help("write `chamber.heat_release -> temperature.source`"),
            );
            return None;
        };

        // The dimension a port carries. Known for the ports this milestone ships;
        // anything else is refused rather than guessed, because a guess here is a
        // silent factor in a coupled model.
        let dimension = match port.text.as_str() {
            "heat_release" => Dimension::POWER.try_div(Dimension::AREA).ok(),
            "temperature" => Some(Dimension::TEMPERATURE),
            "values" => self.field_dimension_of(&root.text),
            "source" => self
                .field_dimension_of(&root.text)
                .and_then(|d| d.try_div(Dimension::TIME).ok()),
            _ => None,
        };
        let Some(dimension) = dimension else {
            self.error(
                Diagnostic::error(format!("`{}` is not a coupling port", port.text))
                    .with_code("E0202")
                    .at(port.span, "unknown port")
                    .help("available: values, source, heat_release, temperature"),
            );
            return None;
        };

        Some(ResolvedPort {
            reference: PortRef::new(index, port.text.clone()),
            dimension,
            owner: root.text.clone(),
        })
    }

    /// The dimension of the field a domain solves, when it solves one.
    fn field_dimension_of(&self, domain: &str) -> Option<Dimension> {
        self.fields.get(domain).map(|field| field.dimension)
    }

    /// Find the declared material property that converts `source` into `target`.
    fn conversion(
        &mut self,
        source: &ResolvedPort,
        target: &ResolvedPort,
        span: Span,
    ) -> Option<Mapping> {
        let needed = target.dimension.try_div(source.dimension).ok();

        // The only conversion this milestone knows: an areal heat capacity, declared on
        // the field that receives the heat.
        if let Some(&capacity) = self.heat_capacity.get(&target.owner)
            && capacity > 0.0
        {
            let reciprocal = Dimension::AREA
                .try_mul(Dimension::TEMPERATURE)
                .ok()
                .and_then(|d| d.try_div(Dimension::ENERGY).ok());
            if needed.is_some() && needed == reciprocal {
                return Some(Mapping::scale(
                    1.0 / capacity,
                    format!(
                        "1 / heat_capacity of `{}` ({capacity:.4e} J/(m^2 K))",
                        target.owner
                    ),
                ));
            }
        }

        self.error(
            Diagnostic::error("this coupling needs a unit conversion the model has not declared")
                .with_code("E0400")
                .at(span, format!(
                    "{} does not convert to {}",
                    source.dimension.describe(),
                    target.dimension.describe()
                ))
                .note(format!(
                    "the factor between them would have to be in {}",
                    needed.map_or_else(
                        || "a dimension outside the representable range".to_string(),
                        |d| d.describe()
                    )
                ))
                .help(format!(
                    "for a heat coupling, add `heat_capacity: 4e5 joule / (meter^2 kelvin);` \
                     to `{}`",
                    target.owner
                )),
        );
        None
    }

    // --- rigid bodies --------------------------------------------------------

    fn declare_rigid_world(&mut self, decl: &lattice_syntax::DomainDecl) {
        if let Some(existing) = self.worlds.get(&decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("rigid world `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here"),
            );
            return;
        }

        const ALLOWED: &[&str] = &["gravity", "iterations"];
        for setting in &decl.settings {
            if !ALLOWED.contains(&setting.key.text.as_str()) {
                self.error(
                    Diagnostic::error(format!(
                        "a rigid2d domain has no setting called `{}`",
                        setting.key.text
                    ))
                    .with_code("E0204")
                    .at(setting.key.span, "unknown setting")
                    .help(format!("rigid2d accepts: {}", ALLOWED.join(", "))),
                );
            }
        }

        let evaluator = self.evaluator();
        let mut world = RigidWorld { span: decl.name.span, ..RigidWorld::default() };

        if let Some(setting) = decl.setting("gravity") {
            // Either a vector or a downward magnitude, matching how the particle
            // module's `gravity(…)` force reads its argument.
            world.gravity = if matches!(setting.value.kind, ExprKind::List(_) | ExprKind::Tuple(_)) {
                evaluator
                    .pair(&setting.value, Dimension::ACCELERATION, "`gravity`", &mut self.diagnostics)
                    .unwrap_or(world.gravity)
            } else {
                evaluator
                    .require(&setting.value, Dimension::ACCELERATION, "`gravity`", &mut self.diagnostics)
                    .map_or(world.gravity, |magnitude| [0.0, -magnitude])
            };
        }
        if let Some(setting) = decl.setting("iterations") {
            if let Some(count) =
                evaluator.count(&setting.value, "`iterations`", &mut self.diagnostics)
            {
                if count == 0 {
                    self.error(
                        Diagnostic::error("a contact solve needs at least one iteration")
                            .with_code("E0405")
                            .at(setting.value.span, "zero iterations")
                            .note("with none, contacts are found and then ignored"),
                    );
                } else {
                    world.iterations = count;
                }
            }
        }

        self.worlds.insert(decl.name.text.clone(), world);
    }

    fn declare_material(&mut self, decl: &Decl) {
        if let Some(existing) = self.materials.get(&decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("material `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here")
                    .help("give one of them a different name, or remove the duplicate"),
            );
            return;
        }
        self.check_settings(decl, rigid::MATERIAL_SETTINGS);
        let evaluator = self.evaluator();
        let material = rigid::material(decl, &evaluator, &mut self.diagnostics);
        self.materials.insert(decl.name.text.clone(), material);
    }

    fn declare_body(&mut self, decl: &Decl) {
        if let Some(existing) = self.bodies.iter().find(|b| b.name == decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("body `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here")
                    .help("give one of them a different name, or remove the duplicate"),
            );
            return;
        }
        self.check_settings(decl, rigid::BODY_SETTINGS);
        self.declared_bodies += 1;
        let evaluator = self.evaluator();
        if let Some(plan) = rigid::body(decl, &self.materials, &evaluator, &mut self.diagnostics) {
            self.bodies.push(plan);
        }
    }

    fn declare_joint(&mut self, decl: &Decl) {
        if let Some(existing) = self.joints.iter().find(|j| j.name == decl.name.text) {
            let previous = existing.span;
            self.error(
                Diagnostic::error(format!("joint `{}` is declared twice", decl.name.text))
                    .with_code("E0201")
                    .at(decl.name.span, "redeclared here")
                    .also(previous, "first declared here"),
            );
            return;
        }
        self.check_settings(decl, rigid::JOINT_SETTINGS);
        let evaluator = self.evaluator();
        if let Some(plan) = rigid::joint(decl, &evaluator, &mut self.diagnostics) {
            self.joints.push(plan);
        }
    }

    // --- lowering -----------------------------------------------------------

    fn lower_solve(&mut self, solve: &SolveStmt, out: &mut Lowering) {
        let evaluator = self.evaluator();
        let method = builtins::method(&solve.method, &solve.parameters, &evaluator, &mut self.diagnostics);

        match solve.solver.text.as_str() {
            "heat" | "diffusion" | "transport" => {
                for target in &solve.targets {
                    self.lower_heat(solve, target.value.clone(), &method, out);
                }
            }
            "dynamics" | "particles" => {
                for target in &solve.targets {
                    self.lower_dynamics(&target.value, &method, out);
                }
            }
            "rigid" | "bodies" | "contacts" => self.lower_rigid(solve, &method, out),
            "reactions" | "kinetics" | "chemistry" => self.lower_reactions(solve, &method, out),
            "schrodinger" | "quantum" => self.error(
                Diagnostic::error("a quantum domain is not lowered from a `solve` statement")
                    .with_code("E0206")
                    .at(solve.solver.span, "not a solve")
                    .note("its integrator is part of its block, as in spec §25.2")
                    .help("write `domain quantum2d q { ...; integrator: split_step_fourier(dt=...); }`"),
            ),
            other => {
                let planned = match other {
                    "flow" | "fluid" => Some("M2"),
                    _ => None,
                };
                match planned {
                    Some(milestone) => self.error(
                        Diagnostic::error(format!("the `{other}` solver is not implemented yet"))
                            .with_code("E0900")
                            .at(solve.solver.span, format!("needs the {milestone} module")),
                    ),
                    None => self.error(
                        Diagnostic::error(format!("`{other}` is not a known solver"))
                            .with_code("E0206")
                            .at(solve.solver.span, "unknown solver")
                            .help(
                                "available now: heat, diffusion, transport, dynamics, \
                                 rigid, reactions",
                            ),
                    ),
                }
            }
        }
    }

    fn lower_heat(
        &mut self,
        solve: &SolveStmt,
        target: Expr,
        method: &builtins::Method,
        out: &mut Lowering,
    ) {
        let Some(name) = target.as_name() else {
            self.error(
                Diagnostic::error("a solver target must be a field name")
                    .with_code("E0401")
                    .at(target.span, "not a name"),
            );
            return;
        };

        let Some(field) = self.fields.get(name).cloned() else {
            let known: Vec<&str> = self.fields.keys().map(String::as_str).collect();
            self.error(
                Diagnostic::error(format!("there is no field called `{name}`"))
                    .with_code("E0202")
                    .at(target.span, "unknown field")
                    .help(if known.is_empty() {
                        "declare one with `field <name> on <grid> = <value>;`".to_string()
                    } else {
                        format!("declared fields: {}", known.join(", "))
                    }),
            );
            return;
        };
        if let Some(entry) = self.fields.get_mut(name) {
            entry.solved = true;
        }

        let scheme = match method.name.as_str() {
            "explicit" | "forward_euler" | "ftcs" => TimeScheme::Explicit,
            "crank_nicolson" => TimeScheme::CrankNicolson,
            "backward_euler" | "implicit" => TimeScheme::BackwardEuler,
            other => {
                self.error(
                    Diagnostic::error(format!("`{other}` is not a time scheme for `{}`", solve.solver.text))
                        .with_code("E0207")
                        .at(method.span, "unknown method")
                        .help("one of: explicit, crank_nicolson, backward_euler"),
                );
                return;
            }
        };

        let Some(diffusivity) = field.diffusivity else {
            self.error(
                Diagnostic::error(format!("`{name}` needs a `diffusivity` to be solved"))
                    .with_code("E0203")
                    .at(target.span, "no diffusivity declared")
                    .also(field.span, "declared here")
                    .help(format!(
                        "add a block: `field {name} on {} = … {{ diffusivity: 1e-5 meter^2/second; }}`",
                        field.grid
                    )),
            );
            return;
        };

        if !(diffusivity.is_finite() && diffusivity >= 0.0) {
            // Already reported where it was declared. Constructing the operator with
            // this value would panic, so lowering stops here.
            return;
        }

        let grid = self.grids[&field.grid].grid;
        let mut heat = HeatDomain::new(field.name.clone(), grid, Diffusivity::Uniform(diffusivity))
            .with_scheme(scheme)
            .with_boundaries(field.boundaries)
            // The compiler inferred the field's dimension; pass it on so plots are
            // labelled `K` rather than "field units".
            .with_display_unit(field.dimension.to_string());
        if let Some(dt) = method.timestep {
            heat = heat.with_preferred_step(dt);
        }
        let initializer = field.initializer.clone();
        heat.field_mut().init_from_position(&grid, |position| initializer.sample(position, &grid));

        // The explicit scheme has a hard stability limit; a model that asks for more
        // must be told before it runs, not after it fills with NaN (NFR-007).
        if scheme == TimeScheme::Explicit
            && let Some(dt) = method.timestep
        {
            let limit = heat.stable_step().max;
            if dt > limit {
                self.error(
                    Diagnostic::error(format!(
                        "timestep {dt} s exceeds the explicit stability limit for `{name}`"
                    ))
                    .with_code("E0405")
                    .at(method.span, format!("the limit is {limit:.6e} s"))
                    .note(
                        "an explicit diffusion stencil amplifies the shortest-wavelength mode \
                         every step past this limit; the field is NaN within a few dozen steps",
                    )
                    .help("use `crank_nicolson`, or reduce dt below the limit"),
                );
            }
        }

        let buffer = out.buffers.allocate(
            field.name.clone(),
            BufferKind::ScalarField { nx: grid.nx(), ny: grid.ny(), halo: 1 },
        );
        // Crank-Nicolson and backward Euler carry three assembly fields plus a
        // three-vector CG workspace, all sized with the grid.
        out.buffers.require_scratch(grid.cell_count());

        let id = DomainId::from_index(out.specs.len() as u32);
        out.operations.push(
            Operation::new(format!("prepare {}", field.name), OperationKind::Prepare)
                .in_domain(id)
                .writing(buffer)
                .costing(0.1),
        );
        out.operations.push(
            Operation::new(format!("advance {}", field.name), OperationKind::Advance)
                .in_domain(id)
                .reading(buffer)
                .writing(buffer)
                .costing(grid.cell_count() as f64 * if scheme.is_implicit() { 20.0 } else { 1.0 }),
        );

        self.domain_index.insert(field.name.clone(), out.specs.len());
        out.specs.push(DomainSpec {
            id,
            name: field.name.clone(),
            family: format!("grid2d.heat[{}]", scheme.name()),
            summary: format!(
                "{}x{} on `{}`, values in {}, D = {diffusivity:.4e} m^2/s, {}, initial {}",
                grid.nx(),
                grid.ny(),
                field.grid,
                field.dimension.describe(),
                describe_boundaries(&field.boundaries),
                field.initializer.describe()
            ),
            buffers: vec![buffer],
            contract: Some(heat.contract()),
        });
        if field.is_species {
            self.notes.push(format!(
                "species `{}` was compiled as a diffusing scalar field; reaction terms need \
                 milestone M3",
                field.name
            ));
        }
        out.domains.push(Box::new(heat));
    }

    fn lower_quantum(&mut self, project: &Project, out: &mut Lowering) {
        let blocks: Vec<_> = project.domains().filter(|d| d.family.text == "quantum2d").collect();
        let furnishings: Vec<&Decl> = ["potential", "wavepacket", "detector"]
            .iter()
            .flat_map(|kind| project.declarations_of(kind))
            .collect();
        let Some(block) = blocks.first() else {
            if let Some(decl) = furnishings.first() {
                self.error(
                    Diagnostic::error(format!("a `{}` needs a quantum domain to belong to", decl.kind.text))
                        .with_code("E0203")
                        .at(decl.kind.span, "no `domain quantum2d` in this project")
                        .help("declare `domain quantum2d q { grid: ...; extent: ...; mass: ...; }`"),
                );
            }
            return;
        };
        for extra in &blocks[1..] {
            self.error(
                Diagnostic::error("a project holds one quantum2d domain")
                    .with_code("E0209")
                    .at(extra.name.span, "a second one")
                    .also(block.name.span, "the first")
                    .note(
                        "potentials, wave packets and detectors are not addressed to a domain, so a \
                         second would be ambiguous",
                    ),
            );
        }

        let evaluator = self.evaluator();
        let Some(plan) = quantum::domain(block, &evaluator, &mut self.diagnostics) else { return };
        let potentials: Vec<_> = project
            .declarations_of("potential")
            .filter_map(|decl| quantum::potential(decl, plan.mass, &evaluator, &mut self.diagnostics))
            .collect();
        let packets: Vec<_> = project
            .declarations_of("wavepacket")
            .filter_map(|decl| quantum::wavepacket(decl, &evaluator, &mut self.diagnostics))
            .collect();
        let detectors: Vec<_> = project
            .declarations_of("detector")
            .filter_map(|decl| {
                quantum::detector(decl, &evaluator, &mut self.diagnostics).map(|(name, x)| (name, x, decl.name.span))
            })
            .collect();
        let assembly =
            quantum::Assembly { plan: &plan, potentials: &potentials, packets: &packets, detectors: &detectors };
        let Some(domain) = assembly.build(&mut self.diagnostics) else { return };

        let (nx, ny) = (plan.grid.nx(), plan.grid.ny());
        let cells = nx * ny;
        let state = out.buffers.allocate(plan.name.clone(), BufferKind::VectorField { nx, ny, halo: 0 });
        // Split-step keeps two phase tables, a loss table and an observation transform;
        // Crank-Nicolson its Krylov vectors. In f64 elements, a complex value being two.
        let (workspace, per_cell) = match plan.scheme {
            QuantumScheme::SplitStepFourier => (2 * 3 * cells + cells, 2.0 * (cells as f64).log2() + 6.0),
            QuantumScheme::CrankNicolson => (2 * 12 * cells, 60.0),
        };
        let scratch =
            out.buffers.allocate(format!("{} workspace", plan.name), BufferKind::Scratch { elements: workspace });

        let id = DomainId::from_index(out.specs.len() as u32);
        out.operations.push(
            Operation::new(format!("prepare {}", plan.name), OperationKind::Prepare)
                .in_domain(id)
                .writing(state)
                .costing(0.1),
        );
        out.operations.push(
            Operation::new(format!("advance {}", plan.name), OperationKind::Advance)
                .in_domain(id)
                .reading(state)
                .writing(state)
                .writing(scratch)
                .costing(cells as f64 * per_cell),
        );
        self.domain_index.insert(plan.name.clone(), out.specs.len());
        out.specs.push(DomainSpec {
            id,
            name: plan.name.clone(),
            family: domain.contract().name.to_string(),
            summary: assembly.describe(),
            buffers: vec![state, scratch],
            contract: Some(domain.contract()),
        });
        out.domains.push(Box::new(domain));
    }

    fn lower_dynamics(&mut self, target: &Expr, method: &builtins::Method, out: &mut Lowering) {
        let Some(name) = target.as_name() else {
            self.error(
                Diagnostic::error("a solver target must be a particle set name")
                    .with_code("E0401")
                    .at(target.span, "not a name"),
            );
            return;
        };

        let Some(set) = self.particles.get(name).cloned() else {
            let known: Vec<&str> = self.particles.keys().map(String::as_str).collect();
            self.error(
                Diagnostic::error(format!("there is no particle set called `{name}`"))
                    .with_code("E0202")
                    .at(target.span, "unknown particle set")
                    .help(if known.is_empty() {
                        "declare one with `particles <name> { count: …; }`".to_string()
                    } else {
                        format!("declared particle sets: {}", known.join(", "))
                    }),
            );
            return;
        };
        if let Some(entry) = self.particles.get_mut(name) {
            entry.solved = true;
        }

        let integrator = match method.name.as_str() {
            "velocity_verlet" | "verlet" => Integrator::VelocityVerlet,
            "semi_implicit_euler" | "symplectic_euler" => Integrator::SemiImplicitEuler,
            "explicit_euler" | "forward_euler" => Integrator::ExplicitEuler,
            other => {
                self.error(
                    Diagnostic::error(format!("`{other}` is not an integrator for particles"))
                        .with_code("E0207")
                        .at(method.span, "unknown integrator")
                        .help("one of: velocity_verlet, semi_implicit_euler, explicit_euler"),
                );
                return;
            }
        };

        if matches!(set.thermostat, Some(Thermostat::Langevin { .. })) && integrator != Integrator::VelocityVerlet {
            self.error(
                Diagnostic::error("a Langevin thermostat needs the velocity_verlet integrator")
                    .with_code("E0207")
                    .at(method.span, format!("`{}` has no Langevin form", method.name))
                    .note(
                        "the BAOAB splitting is built on velocity Verlet's kick-drift structure; \
                         the first-order schemes have no such pairing",
                    )
                    .help("solve with `velocity_verlet(dt=…)`, or use `velocity_rescale`"),
            );
            return;
        }

        let needs_region = set.forces.iter().any(ForceSpec::needs_region)
            || set.boundary != ParticleBoundary::Open
            || !set.analyses.is_empty();
        if needs_region && set.region.is_none() {
            self.error(
                Diagnostic::error(format!("particle set `{name}` needs a `region`"))
                    .with_code("E0203")
                    .at(set.span, "no region declared")
                    .note(
                        "a pair force with a cutoff needs a region to bin particles into, a \
                         periodic or reflective boundary needs one to wrap or bounce against, \
                         and a radial distribution needs one to normalize by",
                    )
                    .help("add `region: [10 meter, 10 meter];`"),
            );
            return;
        }
        if let (Some(region), Some(skin)) = (set.region, set.skin)
            && let Some(cutoff) = set.forces.iter().filter_map(ForceSpec::cutoff).fold(None, |m: Option<f64>, c| Some(m.map_or(c, |m| m.max(c))))
            && set.boundary == ParticleBoundary::Periodic
            && 2.0 * (cutoff + skin) > region[0].min(region[1])
        {
            self.error(
                Diagnostic::error("the cutoff plus skin exceeds half the periodic region")
                    .with_code("E0405")
                    .at(set.span, format!("cutoff {cutoff} m + skin {skin} m against a region of {region:?} m"))
                    .note("the minimum-image convention needs the box to be at least twice the list radius"),
            );
            return;
        }
        for request in &set.analyses
        {
            if let Some(region) = set.region
                && set.boundary == ParticleBoundary::Periodic
                && request.range > 0.5 * region[0].min(region[1])
            {
                self.error(
                    Diagnostic::error("the RDF range exceeds half the periodic region")
                        .with_code("E0405")
                        .at(set.span, format!("range {} m against a region of {region:?} m", request.range))
                        .note("beyond half the box a pair and its image are both counted"),
                );
                return;
            }
        }

        let mut domain = ParticleDomain::new(set.name.clone(), set.count.max(1))
            .with_integrator(integrator)
            .with_seed(set.seed);
        if let Some(dt) = method.timestep {
            domain = domain.with_preferred_step(dt);
        }
        if let Some(skin) = set.skin {
            domain = domain.with_skin(skin);
        }
        if let Some(thermostat) = set.thermostat {
            domain = domain.with_thermostat(thermostat);
        }
        for request in &set.analyses {
            domain = domain.with_rdf(*request);
        }
        // Attach the region only when something needs it. An `open` boundary does
        // nothing, and binding the domain to a box anyway would crop the viewer's
        // window to a region particles are entitled to leave.
        if needs_region
            && let Some(region) = set.region
        {
            domain = domain.with_bounds(BoundaryBox {
                min: set.origin,
                size: region,
                x: set.boundary,
                y: set.boundary,
            });
        }
        for force in &set.forces {
            domain = match *force {
                ForceSpec::Gravity { acceleration } => {
                    domain.with_force(UniformAcceleration::new(acceleration))
                }
                ForceSpec::Drag { coefficient } => domain.with_force(LinearDrag::new(coefficient)),
                ForceSpec::HarmonicWell { center, stiffness } => {
                    domain.with_force(HarmonicWell::new(center, stiffness))
                }
                ForceSpec::LennardJones { epsilon, sigma, cutoff, truncation } => {
                    domain.with_force(LennardJones::with_truncation(epsilon, sigma, cutoff, truncation))
                }
                ForceSpec::SoftRepulsion { stiffness, range } => {
                    domain.with_force(SoftRepulsion::new(stiffness, range))
                }
            };
        }

        let ids = self.populate(&set, &mut domain);
        if ids.len() != set.count {
            return;
        }
        let Some(domain) = self.attach_topology(&set, domain, &ids) else { return };
        let mut domain = domain;
        domain.initialize();
        self.warn_about_strained_bonds(&set, &domain);

        let buffer = out
            .buffers
            .allocate(set.name.clone(), BufferKind::ParticleArrays { capacity: set.count.max(1) });

        let id = DomainId::from_index(out.specs.len() as u32);
        out.operations.push(
            Operation::new(format!("prepare {}", set.name), OperationKind::Prepare)
                .in_domain(id)
                .writing(buffer)
                .costing(0.1),
        );
        out.operations.push(
            Operation::new(format!("advance {}", set.name), OperationKind::Advance)
                .in_domain(id)
                .reading(buffer)
                .writing(buffer)
                .costing(set.count as f64),
        );

        let mut parts: Vec<String> = set.forces.iter().map(ForceSpec::describe).collect();
        parts.extend(set.bonds.iter().map(|b| b.describe(set.count)));
        parts.extend(set.angles.iter().map(|a| a.describe(set.count)));
        if let Some(thermostat) = set.thermostat {
            parts.push(thermostat.describe());
        }
        if let Some(skin) = set.skin {
            parts.push(format!("Verlet skin {skin:.4e} m"));
        }
        if let Some(temperature) = set.temperature {
            parts.push(format!("started at {temperature} K"));
        }
        for request in &set.analyses {
            let after = if request.after > 0 { format!(" after step {}", request.after) } else { String::new() };
            parts.push(format!(
                "RDF: {} bins to {:.4e} m every {} steps{after}",
                request.bins, request.range, request.every
            ));
        }
        let forces = if parts.is_empty() { "no forces".to_string() } else { parts.join("; ") };
        self.domain_index.insert(set.name.clone(), out.specs.len());
        out.specs.push(DomainSpec {
            id,
            name: set.name.clone(),
            family: domain.contract().name.to_string(),
            summary: format!(
                "{} particles, mass {} kg, {:?} boundary, {forces}",
                set.count, lattice_ir::format_number(set.mass), set.boundary
            ),
            buffers: vec![buffer],
            contract: Some(domain.contract()),
        });
        out.domains.push(Box::new(domain));
    }

    /// Expand the declared bonds and angles into laws over the spawned handles.
    ///
    /// Returns `None` after reporting when a topology does not fit the set.
    fn attach_topology(
        &mut self,
        set: &ParticlesInfo,
        mut domain: ParticleDomain,
        ids: &[ParticleId],
    ) -> Option<ParticleDomain> {
        let count = ids.len();
        for spec in &set.bonds {
            let pairs = molecular::expand(&spec.topology, 2, count, "bond", spec.span, &mut self.diagnostics)?;
            let length = spec.length.or(set.spacing)?;
            let bonds = pairs
                .iter()
                .map(|pair| Bond::new(ids[pair[0]], ids[pair[1]], length, spec.stiffness))
                .collect();
            domain = domain.with_force(HarmonicBond::new(bonds).with_pair_forces(spec.keep_pair_forces));
        }
        for spec in &set.angles {
            let triples = molecular::expand(&spec.topology, 3, count, "angle", spec.span, &mut self.diagnostics)?;
            let angles = triples
                .iter()
                .map(|t| Angle::new(ids[t[0]], ids[t[1]], ids[t[2]], spec.angle, spec.stiffness))
                .collect();
            domain = domain.with_force(HarmonicAngle::new(angles));
        }
        Some(domain)
    }

    /// A bond that starts far from its rest length stores a lot of energy the model
    /// never asked for. That is a legitimate thing to want and an easy thing to do by
    /// accident — a `chain` over a square lattice jumps a row every `sqrt(count)`
    /// particles — so it is a warning that names the bond, not an error.
    fn warn_about_strained_bonds(&mut self, set: &ParticlesInfo, domain: &ParticleDomain) {
        let (xs, ys) = (domain.store().pos_x(), domain.store().pos_y());
        let image = domain.bounds().map_or(lattice_domain_particle::MinimumImage::open(), |b| b.image());
        let mut worst: Option<(usize, f64, f64)> = None;
        for (index, &[a, b]) in domain.bonded_pairs().iter().enumerate() {
            let (a, b) = (a as usize, b as usize);
            let (dx, dy) = image.separation(xs[b] - xs[a], ys[b] - ys[a]);
            let length = (dx * dx + dy * dy).sqrt();
            // Every bond spec contributes its own rest length; find the one this
            // pair came from by walking the specs in the order they were attached.
            let mut offset = 0;
            let mut rest = None;
            for spec in &set.bonds {
                let n = match &spec.topology {
                    molecular::Topology::Chain => set.count.saturating_sub(1),
                    molecular::Topology::Ring => set.count,
                    molecular::Topology::Explicit(t) => t.len(),
                };
                if index < offset + n {
                    rest = spec.length.or(set.spacing);
                    break;
                }
                offset += n;
            }
            let Some(rest) = rest else { continue };
            let strain = (length - rest).abs() / rest;
            if strain > 0.5 && worst.is_none_or(|(_, _, s)| strain > s) {
                worst = Some((index, length, strain));
            }
        }
        if let Some((index, length, strain)) = worst {
            self.error(
                Diagnostic::warning(format!(
                    "bond {index} of `{}` starts at {:.3e} m, {:.0}% away from its rest length",
                    set.name,
                    length,
                    strain * 100.0
                ))
                .with_code("W0307")
                .at(set.span, "strained at the start")
                .note("the stored spring energy is released into the dynamics on the first step")
                .help("use `layout: serpentine;` so consecutive particles start one spacing apart"),
            );
        }
    }

    /// Turn every `body` and `joint` declaration into one rigid world.
    ///
    /// All bodies in a project join a single world. Two independent rigid worlds in one
    /// model would need bodies to say which they belong to, and nothing yet wants that
    /// — spec §14 puts interaction between domains in a `couple`, not in a shared body
    /// list.
    fn lower_rigid(&mut self, solve: &SolveStmt, method: &builtins::Method, out: &mut Lowering) {
        let Some(target) = solve.targets.first() else {
            return;
        };
        let Some(name) = target.value.as_name() else {
            self.error(
                Diagnostic::error("a rigid solver target must be a world name")
                    .with_code("E0401")
                    .at(target.value.span, "not a name")
                    .help("write `solve rigid(world) with sequential_impulse(dt = 0.008 second);`"),
            );
            return;
        };
        if solve.targets.len() > 1 {
            self.error(
                Diagnostic::error("a rigid solve names one world")
                    .with_code("E0401")
                    .at(solve.targets[1].value.span, "extra target")
                    .note("every declared body joins a single world"),
            );
        }

        match method.name.as_str() {
            "sequential_impulse" | "impulse" | "contacts" => {}
            other => {
                self.error(
                    Diagnostic::error(format!("`{other}` is not a method for rigid bodies"))
                        .with_code("E0207")
                        .at(method.span, "unknown method")
                        .note(
                            "a sequential-impulse solver assumes semi-implicit Euler; offering \
                             an alternative would be offering something that does not work",
                        )
                        .help("use `sequential_impulse(dt = …)`"),
                );
                return;
            }
        }

        if self.bodies.is_empty() {
            // Only when there were none to begin with. If every body was rejected, the
            // reader already has a reason for each, and a second error saying the world
            // is empty is a consequence rather than a cause.
            if self.declared_bodies == 0 {
                self.error(
                    Diagnostic::error(format!("rigid world `{name}` has no bodies"))
                        .with_code("E0203")
                        .at(target.value.span, "nothing to solve")
                        .help(
                            "declare one with `body ground { shape: box(10 meter, 0.5 meter); }`",
                        ),
                );
            }
            return;
        }

        // A `domain rigid2d` block is optional; without one the world takes standard
        // gravity and the solver's own defaults.
        let settings = match self.worlds.get(name) {
            Some(world) => *world,
            None => {
                if !self.worlds.is_empty() {
                    let known: Vec<&str> = self.worlds.keys().map(String::as_str).collect();
                    self.error(
                        Diagnostic::error(format!("there is no rigid world called `{name}`"))
                            .with_code("E0202")
                            .at(target.value.span, "unknown world")
                            .help(format!("declared worlds: {}", known.join(", "))),
                    );
                    return;
                }
                self.notes.push(format!(
                    "rigid world `{name}` uses standard gravity; add \
                     `domain rigid2d {name} {{ gravity: …; }}` to change it"
                ));
                RigidWorld::default()
            }
        };

        let mut world = RigidDomain::new(name.to_string(), self.bodies.len())
            .with_gravity(settings.gravity)
            .with_solver(SolverConfig {
                velocity_iterations: settings.iterations,
                ..SolverConfig::default()
            });
        if let Some(dt) = method.timestep {
            world = world.with_preferred_step(dt);
        }

        // Bodies take slots in declaration order, so a reader can match the report to
        // the source by counting down the file.
        let mut slots: BTreeMap<String, usize> = BTreeMap::new();
        let plans = std::mem::take(&mut self.bodies);
        for plan in &plans {
            let shape = world.register(
                Collider::new(plan.shape.clone())
                    .with_restitution(plan.material.restitution)
                    .with_friction(plan.material.friction),
            );
            let spec = BodySpec {
                position: plan.at,
                angle: plan.angle,
                velocity: plan.velocity,
                angular_velocity: plan.spin,
                shape,
                ..BodySpec::default()
            };
            let id = if plan.is_static {
                world.spawn(BodySpec { mass: 0.0, inertia: 0.0, ..spec })
            } else if let Some(mass) = plan.mass {
                world.spawn_with_mass(spec, mass)
            } else {
                world.spawn_with_density(spec, plan.material.density)
            };
            let Some(id) = id else {
                self.error(
                    Diagnostic::error(format!("body `{}` did not fit in the world", plan.name))
                        .with_code("E0405")
                        .at(plan.span, "no room"),
                );
                continue;
            };
            let slot = world.slot_of(id).expect("just spawned");

            // A dynamic body with no mass cannot be moved, and the model almost
            // certainly meant something else.
            if !plan.is_static && world.bodies().is_static(slot) {
                self.error(
                    Diagnostic::error(format!("body `{}` came out with no mass", plan.name))
                        .with_code("E0405")
                        .at(plan.span, "mass is zero")
                        .note("density times area, or the explicit mass, evaluated to zero")
                        .help("add `motion: static;` if it was meant to be immovable"),
                );
            }
            slots.insert(plan.name.clone(), slot);
        }

        let joint_plans = std::mem::take(&mut self.joints);
        for plan in &joint_plans {
            let Some((a, b)) = rigid::resolve(plan, &slots, &mut self.diagnostics) else {
                continue;
            };
            // "However far apart they are now" is what leaving the length out means.
            let anchors = joint_anchors(&world, plan, a, b);
            world.add_joint(plan.build(a, b, anchors));
        }

        self.bodies_solved = true;

        let buffer = out
            .buffers
            .allocate(name.to_string(), BufferKind::RigidBodyArrays { capacity: plans.len() });
        let id = DomainId::from_index(out.specs.len() as u32);
        out.operations.push(
            Operation::new(format!("prepare {name}"), OperationKind::Prepare)
                .in_domain(id)
                .writing(buffer)
                .costing(0.1),
        );
        out.operations.push(
            Operation::new(format!("advance {name}"), OperationKind::Advance)
                .in_domain(id)
                .reading(buffer)
                .writing(buffer)
                // Broadphase is n log n and the contact solve is per-point per-iteration;
                // a body is worth several particles.
                .costing(plans.len() as f64 * settings.iterations as f64),
        );

        let statics = plans.iter().filter(|p| p.is_static).count();
        let mut summary = format!(
            "{} bodies ({statics} static), gravity [{:.3}, {:.3}] m/s^2, {} contact iterations",
            plans.len(),
            settings.gravity[0],
            settings.gravity[1],
            settings.iterations
        );
        if !joint_plans.is_empty() {
            summary.push_str(&format!(", {} joints", joint_plans.len()));
        }
        for plan in &plans {
            self.notes.push(format!("body `{}`: {}", plan.name, rigid::describe(plan)));
        }
        for plan in &joint_plans {
            self.notes.push(format!("joint `{}`: {}", plan.name, plan.describe()));
        }

        self.domain_index.insert(name.to_string(), out.specs.len());
        out.specs.push(DomainSpec {
            id,
            name: name.to_string(),
            family: "rigid2d[sequential_impulse]".to_string(),
            summary,
            buffers: vec![buffer],
            contract: Some(world.contract()),
        });
        out.domains.push(Box::new(world));
    }

    /// Place particles and give them initial velocities, returning their handles in
    /// placement order.
    ///
    /// The velocity distribution is shifted so total momentum is exactly zero. Without
    /// that, a "conserved" momentum starts at some arbitrary value and the diagnostic
    /// is far less useful. A `temperature` draws from the Maxwell distribution instead
    /// and rescales so the instantaneous temperature is exactly the one written.
    fn populate(&mut self, set: &ParticlesInfo, domain: &mut ParticleDomain) -> Vec<ParticleId> {
        let region = set.region.unwrap_or([1.0, 1.0]);
        let mut rng = Pcg32::seed_from_u64(set.seed);

        let velocities: Vec<[f64; 2]> = (0..set.count)
            .map(|_| {
                if set.speed > 0.0 {
                    [rng.normal() * set.speed, rng.normal() * set.speed]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let count = set.count.max(1) as f64;
        let mean = [
            velocities.iter().map(|v| v[0]).sum::<f64>() / count,
            velocities.iter().map(|v| v[1]).sum::<f64>() / count,
        ];

        // A square lattice that fits the requested count, so a default spacing fills
        // the region evenly.
        let per_side = (set.count as f64).sqrt().ceil().max(1.0) as usize;
        let spacing = set
            .spacing
            .unwrap_or_else(|| (region[0] / per_side as f64).min(region[1] / per_side as f64));

        let mut ids = Vec::with_capacity(set.count);
        for (index, velocity) in velocities.iter().enumerate() {
            let (column, row) = (index % per_side, index / per_side);
            // A serpentine fill reverses every other row, so index `i + 1` is always
            // one spacing from index `i` — including across the row boundary.
            let i = match set.layout {
                Layout::Lattice => column,
                Layout::Serpentine if row % 2 == 1 => per_side - 1 - column,
                Layout::Serpentine => column,
            };
            let position = [
                set.origin[0] + (i as f64 + 0.5) * spacing,
                set.origin[1] + (row as f64 + 0.5) * spacing,
            ];
            let velocity = [velocity[0] - mean[0], velocity[1] - mean[1]];
            let spawned = domain.spawn(
                ParticleSpec::at(position)
                    .with_velocity(velocity)
                    .with_mass(set.mass)
                    .with_radius(set.radius),
            );
            match spawned {
                Some(id) => ids.push(id),
                None => {
                    self.error(
                        Diagnostic::error(format!(
                            "particle set `{}` could not be filled to {} particles",
                            set.name, set.count
                        ))
                        .with_code("E0405")
                        .at(set.span, "capacity exhausted"),
                    );
                    return ids;
                }
            }
        }
        if let Some(temperature) = set.temperature {
            analysis::thermalize(domain.store_mut(), temperature, &mut rng);
        }
        ids
    }

    fn lower_observers(&mut self, project: &Project) -> Vec<ObserverSpec> {
        let mut observers = Vec::new();
        for statement in project.observes() {
            let target = match statement.target.as_name() {
                Some(name) => name.to_string(),
                None => self.file.slice(statement.target.span).to_string(),
            };
            let interval = statement.every.as_ref().and_then(|expr| {
                // `every step` is the one non-quantity form the spec uses.
                if expr.as_name() == Some("step") {
                    return None;
                }
                let evaluator = self.evaluator();
                evaluator.require(expr, Dimension::TIME, "an observe interval", &mut self.diagnostics)
            });
            observers.push(ObserverSpec {
                id: ObserverId::from_index(observers.len() as u32),
                target,
                interval,
            });
        }
        observers
    }

    fn lower_visuals(&mut self, project: &Project) -> Vec<VisualSpec> {
        project
            .visualizes()
            .map(|statement| VisualSpec {
                target: statement
                    .target
                    .as_name()
                    .map_or_else(|| self.file.slice(statement.target.span).to_string(), String::from),
                style: statement.style.as_ref().map(|ident: &Ident| ident.text.clone()),
            })
            .collect()
    }
}

/// Top-level settings with their defaults.
struct ProjectSettings {
    dimensions: u32,
    fidelity: FidelityProfile,
    precision: Precision,
    duration: Option<f64>,
    timestep: Option<f64>,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        ProjectSettings {
            dimensions: 2,
            fidelity: FidelityProfile::Interactive,
            precision: Precision::Accurate64,
            duration: None,
            timestep: None,
        }
    }
}

fn describe_boundaries(set: &BoundarySet) -> String {
    let sides = [
        ("left", set.left),
        ("right", set.right),
        ("bottom", set.bottom),
        ("top", set.top),
    ];
    if sides.iter().all(|(_, b)| *b == set.left) {
        return format!("{} on all sides", set.left.kind_name());
    }
    sides
        .iter()
        .map(|(name, boundary)| format!("{name} {}", boundary.kind_name()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A coupling endpoint, resolved.
#[derive(Clone, PartialEq, Debug)]
struct ResolvedPort {
    reference: PortRef,
    dimension: Dimension,
    /// The domain the port belongs to, for a diagnostic and for looking up the material
    /// property that converts it.
    owner: String,
}

/// The conserved quantity a `conserve` clause names.
fn invariant_named(name: &str) -> Option<Invariant> {
    Some(match name {
        "energy" => Invariant::Energy,
        "mass" => Invariant::Mass,
        "momentum_x" => Invariant::MomentumX,
        "momentum_y" => Invariant::MomentumY,
        "charge" => Invariant::Charge,
        "amount" | "moles" => Invariant::Amount,
        _ => return None,
    })
}

/// The current world distance between a joint's two anchor points.
///
/// Used when the model left the length out: "however far apart they are now" is what a
/// reader means by omitting it, and computing it from the placed bodies is the only way
/// to honour that.
fn joint_anchors(world: &RigidDomain, plan: &rigid::JointPlan, a: usize, b: usize) -> f64 {
    let (local_a, local_b) = match plan.build {
        rigid::JointShape::Distance { anchor_a, anchor_b, .. }
        | rigid::JointShape::Pin { anchor_a, anchor_b }
        | rigid::JointShape::Spring { anchor_a, anchor_b, .. } => (anchor_a, anchor_b),
        rigid::JointShape::Motor { .. } => return 0.0,
    };
    let pa = world.bodies().to_world_point(a, local_a.to_array());
    let pb = world.bodies().to_world_point(b, local_b.to_array());
    (pb[0] - pa[0]).hypot(pb[1] - pa[1])
}
