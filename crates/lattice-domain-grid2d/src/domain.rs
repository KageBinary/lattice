//! The heat / diffusion domain.

use lattice_ir::{
    Boundary, BoundarySet, Domain, FidelityProfile, Grid2d, Invariant, ObservationKind,
    Observations, Precision, ResidualHistory, ScalarField, SolveOutcome, SolverContract,
    StableStep, StepContext,
};

use crate::boundary::{apply_boundaries, HaloMode};
use crate::operator::{DiffusionOperator, Diffusivity};
use crate::solver::{conjugate_gradient, CgWorkspace};

/// How the diffusion term is advanced in time.
///
/// All three are the `θ`-method `u^{n+1} = u^n + dt·[(1−θ)L(u^n) + θL(u^{n+1})]` at
/// different `θ`. They differ in cost and in what they are honest about.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TimeScheme {
    /// `θ = 0`. One stencil pass per step, no linear solve — and a hard stability
    /// limit that tightens as `1/h²`. Doubling the resolution quarters the timestep.
    Explicit,
    /// `θ = ½`. Second-order accurate and unconditionally stable.
    ///
    /// Unconditionally *stable* is not unconditionally *accurate*: at large `dt`
    /// Crank–Nicolson does not blow up, but it rings — a sharp initial profile
    /// produces decaying oscillations rather than smooth decay. The amplification
    /// factor approaches −1 for the shortest modes instead of 0.
    #[default]
    CrankNicolson,
    /// `θ = 1`. First order, unconditionally stable, and strongly damping.
    ///
    /// Preferred over Crank–Nicolson when the initial condition has a discontinuity,
    /// because its monotone damping suppresses the ringing rather than propagating it.
    BackwardEuler,
}

impl TimeScheme {
    /// The `θ` weight on the new time level.
    pub const fn theta(self) -> f64 {
        match self {
            TimeScheme::Explicit => 0.0,
            TimeScheme::CrankNicolson => 0.5,
            TimeScheme::BackwardEuler => 1.0,
        }
    }

    /// Name used in reports and project files.
    pub const fn name(self) -> &'static str {
        match self {
            TimeScheme::Explicit => "explicit",
            TimeScheme::CrankNicolson => "crank_nicolson",
            TimeScheme::BackwardEuler => "backward_euler",
        }
    }

    /// Order of accuracy in `dt`.
    pub const fn order(self) -> u32 {
        match self {
            TimeScheme::Explicit | TimeScheme::BackwardEuler => 1,
            TimeScheme::CrankNicolson => 2,
        }
    }

    /// Whether a linear solve is required each step.
    pub const fn is_implicit(self) -> bool {
        !matches!(self, TimeScheme::Explicit)
    }
}

/// A scalar transport domain solving `∂u/∂t = ∇·(D∇u) + S`.
///
/// Named for its most common use — temperature — but the equation is the same for
/// species concentration, and the chemistry module reuses it directly.
#[derive(Debug)]
pub struct HeatDomain {
    name: String,
    grid: Grid2d,
    field: ScalarField,
    operator: DiffusionOperator,
    boundaries: BoundarySet,
    scheme: TimeScheme,
    source: Option<ScalarField>,

    // Scratch, allocated once at construction (NFR-001).
    rhs: ScalarField,
    work: ScalarField,
    bc_constant: ScalarField,
    cg: CgWorkspace,

    history: ResidualHistory,
    tolerance: f64,
    max_iterations: usize,
    last_outcome: Option<SolveOutcome>,
    preferred_dt: f64,
    steps: u64,
    /// What the values are measured in, for labelling plots and observations.
    ///
    /// The solver does not need this — it computes in unit-free SI either way — but a
    /// reader of a heatmap does. The compiler knows the field's dimension and sets it.
    display_unit: String,
}

/// The coupling ports this solver offers (spec §14.1).
static PORTS: [lattice_ir::PortSpec; 2] = [
    lattice_ir::PortSpec::publishes_field(
        "field",
        "field units",
        "the solved field itself, cell by cell",
    ),
    lattice_ir::PortSpec::consumes_field(
        "source",
        "field units / s",
        "a source term added to every cell. For temperature this is K/s, NOT a power \
         density: converting one to the other needs an areal heat capacity, which is a \
         property of neither domain and which spec 14.1 puts on the coupling edge",
    ),
];

/// Halo width. One cell is all a 5-point stencil reads.
const HALO: usize = 1;

/// What the values are called before anyone says otherwise.
///
/// A solver run straight from Rust may genuinely not know; a model compiled from
/// `.lattice` source always does, because the compiler read the field's dimension.
const UNLABELLED: &str = "field units";

impl HeatDomain {
    /// Build a domain on `grid` with the given diffusivity.
    ///
    /// Defaults: insulated on all sides, Crank–Nicolson, relative tolerance `1e-10`,
    /// at most 500 CG iterations.
    pub fn new(name: impl Into<String>, grid: Grid2d, diffusivity: Diffusivity) -> Self {
        let operator = DiffusionOperator::new(&grid, &diffusivity);
        // An implicit scheme has no stability limit, so the default step is set by
        // usefulness rather than necessity: ten times what an explicit scheme could
        // manage is a visible win without being recklessly inaccurate.
        let explicit_limit = operator.explicit_stability_limit().max;
        let preferred_dt = if explicit_limit.is_finite() { 10.0 * explicit_limit } else { 1.0 };

        Self {
            name: name.into(),
            field: ScalarField::new(&grid, HALO),
            rhs: ScalarField::new(&grid, HALO),
            work: ScalarField::new(&grid, HALO),
            bc_constant: ScalarField::new(&grid, HALO),
            cg: CgWorkspace::new(&grid, HALO),
            operator,
            boundaries: BoundarySet::INSULATED,
            scheme: TimeScheme::default(),
            source: None,
            grid,
            history: ResidualHistory::with_capacity(64),
            tolerance: 1e-10,
            max_iterations: 500,
            last_outcome: None,
            preferred_dt,
            steps: 0,
            display_unit: UNLABELLED.to_string(),
        }
    }

    /// Label the values with a unit, for plots and observations.
    pub fn with_display_unit(mut self, unit: impl Into<String>) -> Self {
        self.display_unit = unit.into();
        self
    }

    /// The unit the values are labelled with.
    pub fn display_unit(&self) -> &str {
        &self.display_unit
    }

    /// The unit of `∫u dA`.
    ///
    /// [`Invariant::FieldIntegral`] carries a placeholder unit because the IR cannot
    /// know what is being integrated — that is the honest answer at that layer. This
    /// solver does know: the area integral of a field in K is in K·m^2. Publishing the
    /// placeholder next to a scale bar that says "K" leaves a reader deciding which of
    /// the two labels to believe.
    pub fn integral_unit(&self) -> String {
        if self.display_unit == UNLABELLED {
            Invariant::FieldIntegral.si_unit().to_string()
        } else {
            format!("{}·m^2", self.display_unit)
        }
    }

    /// Choose the time scheme.
    pub fn with_scheme(mut self, scheme: TimeScheme) -> Self {
        self.scheme = scheme;
        if scheme == TimeScheme::Explicit {
            self.preferred_dt = self.operator.explicit_stability_limit().preferred;
        }
        self
    }

    /// Set the boundary conditions.
    ///
    /// # Panics
    ///
    /// If periodicity is declared on one edge of an axis but not the other. Spec §8.4
    /// step 8 asks for boundary completeness to be validated at compile time; this is
    /// the earliest point that check can run.
    pub fn with_boundaries(mut self, boundaries: BoundarySet) -> Self {
        if let Err(e) = boundaries.validate() {
            panic!("invalid boundary conditions for domain `{}`: {e}", self.name);
        }
        self.boundaries = boundaries;
        self
    }

    /// Set the relative residual tolerance for the implicit solve.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the iteration cap for the implicit solve.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the timestep this domain asks the scheduler for.
    pub fn with_preferred_step(mut self, dt: f64) -> Self {
        self.preferred_dt = dt;
        self
    }

    /// Initialize the field from cell-centre positions.
    pub fn with_initial(mut self, f: impl FnMut([f64; 2]) -> f64) -> Self {
        self.field.init_from_position(&self.grid, f);
        self
    }

    /// Set a uniform initial value.
    pub fn with_uniform_initial(mut self, value: f64) -> Self {
        self.field.fill_interior(value);
        self
    }

    /// Set a source term, in field units per second.
    ///
    /// For a temperature field this is K/s, *not* W/m². Converting a power density to
    /// a temperature rate needs the areal heat capacity `ρ·c·thickness`, which is a
    /// property of the coupling edge rather than of this solver — see §14.1, where
    /// unit conversion is part of the port declaration.
    pub fn set_source(&mut self, source: ScalarField) {
        assert_eq!(
            (source.nx(), source.ny()),
            (self.grid.nx(), self.grid.ny()),
            "source field must match the grid"
        );
        self.source = Some(source);
    }

    /// A zeroed source field sized for this grid, ready to fill.
    pub fn make_source(&self) -> ScalarField {
        ScalarField::new(&self.grid, HALO)
    }

    /// The grid.
    pub fn grid(&self) -> &Grid2d {
        &self.grid
    }

    /// The solution field.
    pub fn field(&self) -> &ScalarField {
        &self.field
    }

    /// The solution field, mutably.
    pub fn field_mut(&mut self) -> &mut ScalarField {
        &mut self.field
    }

    /// The active time scheme.
    pub fn scheme(&self) -> TimeScheme {
        self.scheme
    }

    /// The boundary conditions.
    pub fn boundaries(&self) -> &BoundarySet {
        &self.boundaries
    }

    /// Steps taken.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Approximate bytes of runtime state, for the memory report (§19.3).
    ///
    /// Counts the solution field, the three assembly scratch fields, the optional
    /// source, the conjugate-gradient workspace, and the precomputed face
    /// diffusivities.
    pub fn memory_bytes(&self) -> usize {
        let field_bytes = self.field.len() * size_of::<f64>();
        let fields = 4 + usize::from(self.source.is_some());
        let faces = ((self.grid.nx() + 1) * self.grid.ny()
            + self.grid.nx() * (self.grid.ny() + 1))
            * size_of::<f64>();
        fields * field_bytes + self.cg.bytes() + faces
    }

    /// How the last implicit solve ended, if there has been one.
    pub fn last_solve(&self) -> Option<SolveOutcome> {
        self.last_outcome
    }

    /// The residual trace of the last implicit solve.
    pub fn residual_history(&self) -> &ResidualHistory {
        &self.history
    }

    /// `∫u dA` over the domain.
    ///
    /// Conserved exactly (to round-off) when the boundary is closed and there is no
    /// source — the property the finite-volume discretization was chosen for.
    pub fn integral(&self) -> f64 {
        self.field.integrate(&self.grid)
    }

    /// True when nothing can enter or leave: closed boundaries and no source.
    ///
    /// The validation suite checks this before asserting that the integral is
    /// conserved, so that a Dirichlet edge does not read as a broken solver.
    pub fn is_closed(&self) -> bool {
        self.boundaries.is_closed() && self.source.is_none()
    }

    /// Replace the diffusivity and rebuild the face coefficients.
    pub fn set_diffusivity(&mut self, diffusivity: &Diffusivity) {
        self.operator.recompute_faces(diffusivity);
    }

    /// Explicit update: one stencil pass, no solve.
    fn step_explicit(&mut self, dt: f64) {
        let (dx, dy) = (self.grid.dx(), self.grid.dy());
        let Self { field, work, operator, boundaries, source, .. } = self;

        apply_boundaries(field, boundaries, dx, dy, HaloMode::Inhomogeneous);
        operator.apply(field, work);

        let nx = field.nx();
        for j in 0..field.ny() {
            let start = field.index(0, j);
            let laplacian = work.row(j);
            let source_row = source.as_ref().map(|s| s.row(j));
            let target = &mut field.as_mut_slice()[start..start + nx];
            for i in 0..nx {
                let s = source_row.map_or(0.0, |r| r[i]);
                target[i] += dt * (laplacian[i] + s);
            }
        }
    }

    /// Implicit update: assemble the right-hand side, then solve.
    ///
    /// The affine boundary contribution is split out explicitly. `L` with declared
    /// boundary conditions is affine — `L(u) = L_hom(u) + c` — and conjugate gradient
    /// needs a *linear* operator. So `c` is measured by applying `L` to a zero field
    /// with the real boundary conditions, moved to the right-hand side, and the
    /// iteration runs against `L_hom` alone.
    fn step_implicit(&mut self, dt: f64) {
        let theta = self.scheme.theta();
        let (dx, dy) = (self.grid.dx(), self.grid.dy());
        let Self {
            field,
            rhs,
            work,
            bc_constant,
            cg,
            operator,
            boundaries,
            history,
            source,
            tolerance,
            max_iterations,
            last_outcome,
            ..
        } = self;

        // c = L(0) under the declared boundary conditions.
        work.fill_interior(0.0);
        apply_boundaries(work, boundaries, dx, dy, HaloMode::Inhomogeneous);
        operator.apply(work, bc_constant);

        // L(u^n) under the declared boundary conditions, which is L_hom(u^n) + c.
        apply_boundaries(field, boundaries, dx, dy, HaloMode::Inhomogeneous);
        operator.apply(field, work);

        // rhs = u^n + dt·[(1−θ)·L(u^n) + θ·c + S]
        let nx = field.nx();
        for j in 0..field.ny() {
            let start = rhs.index(0, j);
            let current = field.row(j);
            let laplacian = work.row(j);
            let constant = bc_constant.row(j);
            let source_row = source.as_ref().map(|s| s.row(j));
            let target = &mut rhs.as_mut_slice()[start..start + nx];
            for i in 0..nx {
                let s = source_row.map_or(0.0, |r| r[i]);
                target[i] =
                    current[i] + dt * ((1.0 - theta) * laplacian[i] + theta * constant[i] + s);
            }
        }

        // Solve (I − θ·dt·L_hom)·u^{n+1} = rhs, warm-started from u^n.
        let coefficient = theta * dt;
        let outcome = conjugate_gradient(
            field,
            rhs,
            |input, output| {
                apply_boundaries(input, boundaries, dx, dy, HaloMode::Homogeneous);
                operator.apply(input, output);
                let width = output.nx();
                for j in 0..output.ny() {
                    let start = output.index(0, j);
                    let source_row = input.row(j);
                    let target = &mut output.as_mut_slice()[start..start + width];
                    for i in 0..width {
                        target[i] = source_row[i] - coefficient * target[i];
                    }
                }
            },
            cg,
            *tolerance,
            *max_iterations,
            history,
        );
        *last_outcome = Some(outcome);
    }
}

/// The contract for one time scheme.
///
/// Split by scheme for the same reason the particle domain splits by integrator: an
/// explicit scheme and an unconditionally stable one make different promises, and a
/// single contract could only be honest about one of them.
const fn contract_for(scheme: TimeScheme) -> SolverContract {
    SolverContract {
        name: match scheme {
            TimeScheme::Explicit => "grid2d.heat[explicit]",
            TimeScheme::CrankNicolson => "grid2d.heat[crank_nicolson]",
            TimeScheme::BackwardEuler => "grid2d.heat[backward_euler]",
        },
        summary: "scalar transport by diffusion on a uniform 2D grid",
        governing_equations: &["du/dt = div(D grad u) + S"],
        discretization: "cell-centred finite volume; fluxes evaluated on cell faces \
                         with harmonic-mean face diffusivities; boundary conditions \
                         applied through a one-cell halo",
        integrator: match scheme {
            TimeScheme::Explicit => "forward Euler (theta = 0), 1st order",
            TimeScheme::CrankNicolson => "Crank-Nicolson (theta = 1/2), 2nd order",
            TimeScheme::BackwardEuler => "backward Euler (theta = 1), 1st order",
        },
        assumptions: &[
            "the grid is uniform and axis-aligned",
            "D is isotropic; anisotropic conductivity would need a tensor, not a scalar",
            "the source term is expressed in field units per second, not as a power density",
            "material properties are constant over a step; a temperature-dependent D is \
             lagged by one step unless the caller recomputes it",
            "for temperature, the volumetric heat capacity is uniform, which is what \
             lets the equation be written in terms of a single diffusivity",
        ],
        valid_regime: match scheme {
            TimeScheme::Explicit => {
                "smooth fields at timesteps well inside the stability limit; \
                 interactive use where a small dt is acceptable"
            }
            TimeScheme::CrankNicolson => {
                "smooth initial data at large timesteps; \
                 not recommended for discontinuous initial conditions, which ring"
            }
            TimeScheme::BackwardEuler => {
                "discontinuous or stiff initial data where monotone damping matters \
                 more than second-order accuracy"
            }
        },
        stability: match scheme {
            TimeScheme::Explicit => {
                "conditionally stable: dt <= 1/(2*D_max*(1/dx^2 + 1/dy^2)). \
                 Exceeding it amplifies the shortest-wavelength mode every step"
            }
            _ => {
                "unconditionally stable for any dt. Stability is not accuracy: \
                 Crank-Nicolson rings on sharp data at large dt, and backward Euler \
                 over-damps"
            }
        },
        conserves: &[Invariant::FieldIntegral],
        known_non_conservation: &[
            "the field integral is conserved only on a closed boundary with no source; \
             Dirichlet and Robin edges are sources or sinks by construction and their \
             flux is reported separately",
            "the implicit solve is iterative, so conservation holds to the solver \
             tolerance rather than to round-off; tighten the tolerance if the \
             conservation budget matters more than speed",
            "harmonic-mean face diffusivities are exact for a layered medium but only \
             first-order accurate across a diagonal material interface",
        ],
        fidelity: FidelityProfile::Engineering2d,
        precisions: &[Precision::Accurate64, Precision::Deterministic64],
        deterministic: true,
        differentiable: false,
        validation_cases: &[
            "diffusing Gaussian against the analytic heat kernel",
            "manufactured solution: observed convergence order in space and time",
            "steady state under Dirichlet boundaries",
            "field integral conserved on a closed domain",
            "cross-scheme agreement between explicit and implicit at small dt",
        ],
        references: &[
            "Patankar, S. (1980). Numerical Heat Transfer and Fluid Flow, ch. 4.",
            "LeVeque, R. (2007). Finite Difference Methods for ODEs and PDEs, ch. 9.",
            "Crank, J. & Nicolson, P. (1947). Proc. Camb. Phil. Soc. 43, 50-67.",
        ],
    }
}

static EXPLICIT_CONTRACT: SolverContract = contract_for(TimeScheme::Explicit);
static CRANK_NICOLSON_CONTRACT: SolverContract = contract_for(TimeScheme::CrankNicolson);
static BACKWARD_EULER_CONTRACT: SolverContract = contract_for(TimeScheme::BackwardEuler);

impl Domain for HeatDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn contract(&self) -> &'static SolverContract {
        match self.scheme {
            TimeScheme::Explicit => &EXPLICIT_CONTRACT,
            TimeScheme::CrankNicolson => &CRANK_NICOLSON_CONTRACT,
            TimeScheme::BackwardEuler => &BACKWARD_EULER_CONTRACT,
        }
    }

    fn stable_step(&self) -> StableStep {
        match self.scheme {
            TimeScheme::Explicit => self.operator.explicit_stability_limit(),
            _ => StableStep::unconditional(self.preferred_dt),
        }
    }

    fn prepare(&mut self, _ctx: &mut StepContext<'_>) {}

    fn advance(&mut self, dt: f64, _ctx: &mut StepContext<'_>) {
        if self.scheme.is_implicit() {
            self.step_implicit(dt);
        } else {
            self.step_explicit(dt);
        }
        self.steps += 1;
    }

    fn observe(&self, out: &mut Observations) {
        let prefix = &self.name;
        out.record(
            format!("{prefix}.integral"),
            self.integral(),
            self.integral_unit(),
            ObservationKind::Invariant(Invariant::FieldIntegral),
        );
        out.record_metric(format!("{prefix}.min"), self.field.min_interior(), self.display_unit.clone());
        out.record_metric(format!("{prefix}.max"), self.field.max_interior(), self.display_unit.clone());

        if let Some(outcome) = self.last_outcome {
            out.record(
                format!("{prefix}.solver_iterations"),
                outcome.iterations() as f64,
                "1",
                ObservationKind::Count,
            );
            out.record(
                format!("{prefix}.solver_residual"),
                outcome.residual(),
                self.display_unit.clone(),
                ObservationKind::Residual,
            );
        }
    }

    fn ports(&self) -> &'static [lattice_ir::PortSpec] {
        &PORTS
    }

    fn port_grid(&self) -> Option<Grid2d> {
        Some(self.grid)
    }

    fn read_port(&self, name: &str, out: &mut lattice_ir::PortData) -> bool {
        match (name, out.as_field_mut()) {
            ("field", Some(buffer)) => {
                buffer.copy_interior_from(&self.field);
                true
            }
            _ => false,
        }
    }

    fn write_port(&mut self, name: &str, value: &lattice_ir::PortData) -> bool {
        match (name, value.as_field()) {
            ("source", Some(field)) => {
                let mut source = ScalarField::new(&self.grid, HALO);
                source.copy_interior_from(field);
                self.set_source(source);
                true
            }
            _ => false,
        }
    }

    fn render_channels(&self) -> Vec<lattice_ir::RenderChannel<'_>> {
        vec![lattice_ir::RenderChannel::Scalar {
            name: &self.name,
            field: &self.field,
            grid: self.grid,
            unit: &self.display_unit,
        }]
    }
}

/// A rectangular region helper for building initial conditions.
///
/// Spec §25.1 writes `species A on chamber = left_half(1 mole / meter^2)`; this is the
/// primitive such helpers are built from.
pub fn left_half(grid: &Grid2d, value: f64) -> impl Fn([f64; 2]) -> f64 + '_ {
    let midpoint = grid.origin()[0] + 0.5 * grid.extent()[0];
    move |[x, _]| if x < midpoint { value } else { 0.0 }
}

/// A Gaussian bump, the analytic solution of the heat equation at `t = t0`.
///
/// `∫ G dA = amplitude`, so a Gaussian built this way starts with a known integral —
/// which is what makes it a conservation test as well as a shape test.
pub fn gaussian(center: [f64; 2], variance: f64, amplitude: f64) -> impl Fn([f64; 2]) -> f64 {
    let normalization = amplitude / (core::f64::consts::TAU * variance);
    move |[x, y]| {
        let r2 = (x - center[0]).powi(2) + (y - center[1]).powi(2);
        normalization * (-r2 / (2.0 * variance)).exp()
    }
}

/// Set a Dirichlet value on one side and insulate the rest.
pub fn heated_edge(side: lattice_ir::Side, value: f64) -> BoundarySet {
    let mut bs = BoundarySet::INSULATED;
    bs.set(side, Boundary::fixed(value));
    bs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(name: &str) -> HeatDomain {
        let grid = Grid2d::new(16, 16, [1.0, 1.0]);
        HeatDomain::new(name, grid, Diffusivity::Uniform(1.0))
    }

    /// The viewer draws a scale bar labelled "K" beside a table row reading
    /// "(field unit)·m^2", and a reader has to decide which one is lying.
    #[test]
    fn the_integral_carries_the_unit_of_the_field_it_integrated() {
        assert_eq!(
            domain("h").integral_unit(),
            "(field unit)·m^2",
            "with nothing declared, the IR placeholder is still the honest answer"
        );

        let labelled = domain("h").with_display_unit("K");
        assert_eq!(labelled.integral_unit(), "K·m^2", "an area integral of kelvin is K·m^2");

        // And it survives into the observation, which is where a reader sees it.
        let mut out = Observations::new();
        labelled.observe(&mut out);
        assert_eq!(out.get("h.integral").unwrap().unit, "K·m^2");
        assert_eq!(out.get("h.max").unwrap().unit, "K", "and agrees with the scale bar");
        assert!(
            matches!(
                out.get("h.integral").unwrap().kind,
                ObservationKind::Invariant(Invariant::FieldIntegral)
            ),
            "relabelling the unit must not stop it counting as a conserved quantity"
        );
    }
}
