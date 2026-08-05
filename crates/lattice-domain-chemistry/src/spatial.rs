//! Reaction-diffusion: species transported across a grid and reacting where they meet.
//!
//! Spec §12.3: *"Spatial reaction-diffusion through operator splitting or coupled
//! solvers."* This is the splitting path, and the split is **Strang**:
//!
//! ```text
//!   react dt/2  →  diffuse dt  →  react dt/2
//! ```
//!
//! # Why Strang rather than the obvious thing
//!
//! Doing one operator then the other — Lie–Trotter splitting — is first order in `dt`,
//! and its error has a *systematic sign*: it consistently lags or leads, so a
//! reaction-diffusion front travels at slightly the wrong speed and keeps doing so.
//! Halving the reaction step and putting the diffusion in the middle cancels the
//! leading error term, at the cost of one extra reaction pass. The reaction half-steps
//! of consecutive timesteps can be merged into one full step in the interior of a run,
//! which recovers most of that cost, and this implementation does not bother — it is
//! one cheap pass against a linear solve.
//!
//! # Why each species gets its own diffusion domain
//!
//! Rather than reimplementing `∇·(D∇u)`, each species carries a [`HeatDomain`]. That
//! solver is already validated against the analytic heat kernel and against
//! manufactured solutions, already handles Dirichlet, Neumann, Robin and periodic
//! boundaries, and already offers an implicit scheme with no stability limit. Species
//! diffuse at different rates and may have different boundaries, so one domain each is
//! also the natural shape.
//!
//! # What is conserved and what is not
//!
//! **Mass and every element are conserved exactly**, to round-off, on a closed
//! boundary with a balanced network. Diffusion conserves each species' integral
//! separately; a balanced reaction conserves atoms in each cell independently. Neither
//! half can move mass across the boundary of the other's guarantee.
//!
//! **Energy is not tracked here.** An exothermic reaction's heat is *published* as a
//! field, and what happens to it is the coupling's business — §14.1 puts the
//! conversion from a heat release in W/m² to a temperature rate in K/s on the coupling
//! edge, because it needs an areal heat capacity that is a property of neither domain.

use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{
    Arena, BoundarySet, Domain, FidelityProfile, Grid2d, Invariant, ObservationKind, Observations,
    Precision, ScalarField, SolverContract, StabilityReason, StableStep, StepContext,
};

use crate::kinetics::{Kinetics, KineticsReport};
use crate::network::ReactionNetwork;

/// Species diffusing and reacting on a shared grid.
#[derive(Debug)]
pub struct ReactingMixture {
    name: String,
    grid: Grid2d,
    kinetics: Kinetics,
    /// One diffusion solver per species, in the network's species order.
    transport: Vec<HeatDomain>,
    /// Temperature every cell reacts at, K. Uniform unless a coupling supplies a field.
    uniform_temperature: f64,
    /// Per-cell temperature, when a coupling has supplied one.
    temperature: Option<ScalarField>,
    /// Heat released per cell over the last step, W/m². Published for the coupling.
    heat: ScalarField,
    /// Gathered concentrations for one cell, reused so the reaction loop never
    /// allocates (NFR-001).
    cell: Vec<f64>,
    steps: u64,
    last_report: KineticsReport,
    /// Cells whose reaction ran out of sub-steps last step.
    exhausted_cells: usize,
    /// The stiffest ratio seen anywhere on the grid last step.
    worst_stiffness: f64,
}

impl ReactingMixture {
    /// A mixture on `grid`, reacting by `network`.
    ///
    /// Each species gets a diffusion solver at its own declared coefficient. A species
    /// with `diffusion: 0` still gets one — an immobile species is a diffusion problem
    /// with `D = 0`, which the operator handles and which keeps the species list and
    /// the transport list the same length.
    pub fn new(name: impl Into<String>, grid: Grid2d, network: ReactionNetwork) -> ReactingMixture {
        let name = name.into();
        let transport = network
            .species()
            .iter()
            .map(|species| {
                HeatDomain::new(
                    format!("{name}.{}", species.name),
                    grid,
                    Diffusivity::Uniform(species.diffusion.max(0.0)),
                )
                .with_scheme(TimeScheme::CrankNicolson)
                .with_display_unit("mol/m^2")
            })
            .collect();

        let count = network.len();
        ReactingMixture {
            name,
            grid,
            kinetics: Kinetics::new(network),
            transport,
            uniform_temperature: 298.15,
            temperature: None,
            heat: ScalarField::new(&grid, 1),
            cell: vec![0.0; count],
            steps: 0,
            last_report: KineticsReport::default(),
            exhausted_cells: 0,
            worst_stiffness: 0.0,
        }
    }

    /// Set the temperature every cell reacts at, K.
    pub fn with_temperature(mut self, temperature: f64) -> ReactingMixture {
        self.uniform_temperature = temperature;
        self
    }

    /// Tighten the reaction integration. See [`Kinetics::with_accuracy`].
    pub fn with_accuracy(mut self, accuracy: f64) -> ReactingMixture {
        self.kinetics = self.kinetics.clone().with_accuracy(accuracy);
        self
    }

    /// Set one species' boundary conditions.
    ///
    /// # Panics
    ///
    /// If `index` is not a declared species. A boundary set on a species that does not
    /// exist is a modelling error whose only symptom would be a boundary that quietly
    /// did nothing.
    pub fn set_boundaries(&mut self, index: usize, boundaries: BoundarySet) {
        let name = self.kinetics.network().species()[index].name.clone();
        let domain = std::mem::replace(
            &mut self.transport[index],
            HeatDomain::new("placeholder", self.grid, Diffusivity::Uniform(0.0)),
        );
        self.transport[index] = domain.with_boundaries(boundaries);
        debug_assert!(!name.is_empty());
    }

    /// The concentration field of one species, mol/m².
    pub fn concentration(&self, index: usize) -> Option<&ScalarField> {
        self.transport.get(index).map(HeatDomain::field)
    }

    /// The concentration field of one species, mutably — for setting up an initial
    /// condition.
    pub fn concentration_mut(&mut self, index: usize) -> Option<&mut ScalarField> {
        self.transport.get_mut(index).map(HeatDomain::field_mut)
    }

    /// Look up a species by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.kinetics.network().index_of(name)
    }

    /// The reaction network.
    pub fn network(&self) -> &ReactionNetwork {
        self.kinetics.network()
    }

    /// The grid every species lives on.
    pub fn grid(&self) -> Grid2d {
        self.grid
    }

    /// Heat released per cell over the last step, W/m². Positive warms.
    ///
    /// What the coupling consumes. Converting it to a temperature *rate* needs an areal
    /// heat capacity, which is a property of neither domain — §14.1 puts that on the
    /// coupling edge, and so does this.
    pub fn heat_release(&self) -> &ScalarField {
        &self.heat
    }

    /// Supply a per-cell temperature field, K.
    ///
    /// Copied rather than borrowed, so the mixture does not hold a reference into
    /// another domain across a step — the runtime is entitled to step the two in either
    /// order, and a borrow would fix that order by accident.
    pub fn set_temperature_field(&mut self, temperature: &ScalarField) {
        let target = self.temperature.get_or_insert_with(|| ScalarField::new(&self.grid, 1));
        target.copy_interior_from(temperature);
    }

    /// The temperature of one cell, K.
    fn temperature_at(&self, i: usize, j: usize) -> f64 {
        self.temperature.as_ref().map_or(self.uniform_temperature, |field| field.get(i, j))
    }

    /// Total amount of one species over the whole grid, mol.
    pub fn total(&self, index: usize) -> f64 {
        self.concentration(index).map_or(0.0, |field| field.integrate(&self.grid))
    }

    /// Total mass over the whole grid, kg.
    ///
    /// The quantity a balanced network on a closed boundary conserves exactly, and the
    /// one M3's exit condition checks.
    pub fn total_mass(&self) -> f64 {
        self.kinetics
            .network()
            .species()
            .iter()
            .enumerate()
            .map(|(index, species)| species.molar_mass * self.total(index))
            .sum()
    }

    /// Total amount of one element over the whole grid, mol.
    pub fn total_element(&self, element: &str) -> f64 {
        self.kinetics
            .network()
            .species()
            .iter()
            .enumerate()
            .map(|(index, species)| {
                f64::from(species.composition.count(element)) * self.total(index)
            })
            .sum()
    }

    /// Run the reaction in every cell for `dt`, accumulating the energy released.
    ///
    /// Both Strang half-steps accumulate into the same buffer, so what comes out covers
    /// the whole step exactly once. `reset` clears it first, which the earlier half
    /// does and the later one does not.
    fn react(&mut self, dt: f64, reset: bool) {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let species = self.kinetics.len();
        if species == 0 || dt <= 0.0 {
            return;
        }
        if reset {
            self.heat.fill_all(0.0);
        }

        for j in 0..ny {
            for i in 0..nx {
                for index in 0..species {
                    self.cell[index] = self.transport[index].field().get(i, j);
                }
                let temperature = self.temperature_at(i, j);

                let report = self.kinetics.advance(&mut self.cell, temperature, dt);
                // The *energy* the reaction released over this half-step, J/m². The
                // integrator accumulated it alongside the concentrations, so it is the
                // integral of the power rather than a sample of it — which matters
                // exactly when the reaction is fast enough to slow down during the step.
                self.heat.set(i, j, self.heat.get(i, j) + report.energy);
                if report.exhausted {
                    self.exhausted_cells += 1;
                }
                if let Some(stiffness) = report.stiffness {
                    self.worst_stiffness = self.worst_stiffness.max(stiffness);
                }
                self.last_report = report;

                for index in 0..species {
                    self.transport[index].field_mut().set(i, j, self.cell[index]);
                }
            }
        }
    }
}

/// The coupling ports this domain offers (spec §14.1).
static PORTS: [lattice_ir::PortSpec; 2] = [
    lattice_ir::PortSpec::publishes_field(
        "heat_release",
        "W/m^2",
        "heat released by reaction, averaged over the step just taken",
    ),
    lattice_ir::PortSpec::consumes_field(
        "temperature",
        "K",
        "the temperature every cell reacts at, which Arrhenius rate laws read",
    ),
];

/// The contract. One entry, because the splitting and the schemes it composes are not
/// separately selectable — offering a first-order split would be offering a worse
/// answer for no saving worth having.
static CONTRACT: SolverContract = SolverContract {
    name: "chemistry.reaction_diffusion[strang]",
    summary: "species transported by diffusion and reacting locally, split symmetrically",
    governing_equations: &[
        "dc_i/dt = div(D_i grad c_i) + sum_r nu_ir R_r(c, T)",
        "R_r = k_f(T) prod c^nu - k_r(T) prod c^nu   (mass action)",
        "k(T) = A exp(-Ea / R T)                     (Arrhenius)",
        "q = -sum_r dH_r R_r                          (heat released, W/m^2)",
    ],
    discretization: "finite volume for the transport, on the same cell-centred grid the \
                     heat module uses; the reaction is a local ODE per cell",
    integrator: "Strang splitting: react dt/2, diffuse dt, react dt/2. The transport \
                 half is Crank-Nicolson; the reaction half is RK4, sub-cycled to the \
                 network's own stability and accuracy limits",
    assumptions: &[
        "concentrations are per unit AREA, mol/m^2; this is a 2D engine and a \
         concentration per unit volume would need a thickness nobody declared",
        "the mixture is dilute: species diffuse independently, with no cross-diffusion \
         and no effect of one species' gradient on another's flux",
        "reactions are mass-action in the declared species, with no explicit solvent, \
         activity coefficients, or ionic-strength corrections",
        "a rate constant's units follow the reaction's total order; a second-order \
         constant written where a first-order one belongs is wrong by a factor with \
         the dimensions of a concentration, and the compiler checks it",
        "one temperature model per reaction covers both directions, so an equilibrium \
         constant does not shift with temperature; a reaction whose does is two \
         reactions",
        "the heat released is published in W/m^2 and converted to a temperature rate \
         by the coupling edge, because that conversion needs an areal heat capacity \
         which is a property of neither domain (spec 14.1)",
    ],
    valid_regime: "well-populated concentrations where a continuum description holds, \
                   and networks whose fastest and slowest timescales are within about \
                   three orders of magnitude of each other. Low-copy systems want the \
                   stochastic path; stiffer networks want an implicit solver, which \
                   spec 12.3 defers to an established library",
    stability: "the transport half is unconditionally stable (Crank-Nicolson). The \
                reaction half sub-cycles to its own Jacobian, so it is stable at any \
                requested step — but a step much longer than the reaction timescale is \
                resolved by taking many sub-steps, and the splitting error grows as \
                dt^2 whatever the sub-steps do",
    conserves: &[Invariant::Mass, Invariant::Amount],
    known_non_conservation: &[
        "mass and every element are conserved exactly ONLY when the boundary is closed \
         and the network is balanced. A Dirichlet or Robin edge is a source or a sink \
         by construction, and an unbalanced reaction destroys atoms every step — which \
         is why an unbalanced reaction is a compile error rather than a warning",
        "energy is not tracked. The heat a reaction releases is published as a field \
         and accounted for by the coupling ledger; nothing here checks that it arrives",
        "Strang splitting is second order in dt, so a reaction-diffusion front's speed \
         carries an O(dt^2) error that no amount of sub-cycling inside either half \
         removes. Halving the step is the only fix",
        "the reaction integration is explicit and sub-cycled, so a stiff network \
         exhausts its sub-step budget rather than converging. The count of cells that \
         did is published every step",
        "round-off can drive an exhausted reactant a few ulps below zero; that much is \
         clamped and summed into a published total, because a silent clamp is how a \
         mass check comes to fail for no visible reason",
    ],
    fidelity: FidelityProfile::Engineering2d,
    precisions: &[Precision::Accurate64],
    deterministic: true,
    differentiable: false,
    validation_cases: &[
        "a balanced network on a closed grid conserves mass and every element to round-off",
        "first-order decay against its analytic exponential, in every cell",
        "a reversible reaction settles at its equilibrium constant",
        "diffusion with no reaction reproduces the heat module's analytic kernel",
        "Strang splitting converges at second order in dt",
        "Gray-Scott produces patterns and conserves what it should",
    ],
    references: &[
        "Strang, G. (1968). On the construction and comparison of difference schemes. \
         SIAM J. Numer. Anal. 5, 506-517.",
        "Pearson, J. (1993). Complex patterns in a simple system. Science 261, 189-192.",
        "LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems, ch. 17.",
    ],
};

impl Domain for ReactingMixture {
    fn name(&self) -> &str {
        &self.name
    }

    fn contract(&self) -> &'static SolverContract {
        &CONTRACT
    }

    fn stable_step(&self) -> StableStep {
        // The transport half is unconditionally stable and the reaction half sub-cycles,
        // so nothing here *forbids* a step. What a large step costs is splitting
        // accuracy, and the preferred step reflects that rather than a limit.
        let preferred = self
            .transport
            .iter()
            .map(|domain| domain.stable_step().preferred)
            .fold(f64::INFINITY, f64::min);
        if preferred.is_finite() {
            StableStep::limited(preferred, f64::INFINITY, StabilityReason::Unconditional)
        } else {
            StableStep::unconditional(1.0)
        }
    }

    fn prepare(&mut self, _ctx: &mut StepContext<'_>) {
        self.exhausted_cells = 0;
        self.worst_stiffness = 0.0;
    }

    fn advance(&mut self, dt: f64, ctx: &mut StepContext<'_>) {
        if dt <= 0.0 {
            return;
        }
        // Strang: half a reaction, a whole diffusion, half a reaction.
        self.react(0.5 * dt, true);

        for domain in &mut self.transport {
            domain.prepare(ctx);
            domain.advance(dt, ctx);
        }

        self.react(0.5 * dt, false);

        // `heat` has accumulated joules per square metre over both half-steps. What the
        // coupling wants is a rate, because it will multiply by its own step — so the
        // published field is the *average* power over this step. Publishing an
        // instantaneous power instead would be wrong by however much the reaction
        // changed speed during it.
        let inverse = 1.0 / dt;
        for value in self.heat.as_mut_slice() {
            *value *= inverse;
        }
        self.steps += 1;
    }

    fn observe(&self, out: &mut Observations) {
        let prefix = &self.name;
        let network = self.kinetics.network();

        for (index, species) in network.species().iter().enumerate() {
            let field = self.transport[index].field();
            out.record_metric(
                format!("{prefix}.{}.total", species.name),
                self.total(index),
                "mol",
            );
            out.record_metric(format!("{prefix}.{}.max", species.name), field.max_interior(), "mol/m^2");
            out.record_metric(format!("{prefix}.{}.min", species.name), field.min_interior(), "mol/m^2");
        }

        // Mass and atoms are invariants only when nothing can leave and the network
        // balances. Claiming them otherwise would fire on every model with an open
        // boundary — the mistake the rigid module made with momentum.
        let closed = self.is_conservative();
        let mass = self.total_mass();
        if closed {
            out.record_invariant(format!("{prefix}.total_mass"), Invariant::Mass, mass);
        } else {
            out.record_metric(format!("{prefix}.total_mass"), mass, "kg");
        }
        for element in network.elements() {
            let amount = self.total_element(&element);
            let name = format!("{prefix}.element_{element}");
            if closed {
                out.record_invariant(name, Invariant::Amount, amount);
            } else {
                out.record_metric(name, amount, "mol");
            }
        }

        out.record_metric(format!("{prefix}.heat_release"), self.heat.integrate(&self.grid), "W");
        out.record_metric(
            format!("{prefix}.peak_heat"),
            self.heat.max_abs_interior(),
            "W/m^2",
        );
        out.record(
            format!("{prefix}.reaction_substeps"),
            self.last_report.substeps as f64,
            "1",
            ObservationKind::Count,
        );
        out.record(
            format!("{prefix}.exhausted_cells"),
            self.exhausted_cells as f64,
            "1",
            ObservationKind::Count,
        );
        out.record_metric(format!("{prefix}.stiffness"), self.worst_stiffness, "1");
    }

    fn ports(&self) -> &'static [lattice_ir::PortSpec] {
        &PORTS
    }

    fn port_grid(&self) -> Option<Grid2d> {
        Some(self.grid)
    }

    fn read_port(&self, name: &str, out: &mut lattice_ir::PortData) -> bool {
        match (name, out.as_field_mut()) {
            ("heat_release", Some(buffer)) => {
                buffer.copy_interior_from(&self.heat);
                true
            }
            _ => false,
        }
    }

    fn write_port(&mut self, name: &str, value: &lattice_ir::PortData) -> bool {
        match (name, value.as_field()) {
            ("temperature", Some(field)) => {
                self.set_temperature_field(field);
                true
            }
            _ => false,
        }
    }

    fn render_channels(&self) -> Vec<lattice_ir::RenderChannel<'_>> {
        self.kinetics
            .network()
            .species()
            .iter()
            .enumerate()
            .map(|(index, species)| lattice_ir::RenderChannel::Scalar {
                name: &species.name,
                field: self.transport[index].field(),
                grid: self.grid,
                unit: "mol/m^2",
            })
            .collect()
    }
}

impl ReactingMixture {
    /// True when mass and atoms are genuinely conserved: a closed boundary on every
    /// species, and a network that balances.
    ///
    /// Both halves matter. A Dirichlet edge is a source by construction, and an
    /// unbalanced reaction destroys atoms in every cell — claiming conservation in
    /// either case would fire an alarm on a model that is behaving exactly as written.
    pub fn is_conservative(&self) -> bool {
        self.transport.iter().all(HeatDomain::is_closed)
            && self.kinetics.network().is_fully_specified()
            && self.kinetics.network().imbalances().is_empty()
    }

    /// How many cells ran out of reaction sub-steps last step.
    pub fn exhausted_cells(&self) -> usize {
        self.exhausted_cells
    }

    /// Step the mixture without a runtime, for tests and examples.
    pub fn step(&mut self, dt: f64, arena: &mut Arena) {
        let mut ctx = StepContext::new(arena);
        self.prepare(&mut ctx);
        self.advance(dt, &mut ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{Reaction, Term};
    use crate::rate::RateLaw;
    use crate::species::Species;

    fn grid() -> Grid2d {
        Grid2d::new(24, 24, [1.0, 1.0])
    }

    /// `A -> B`, first order, both species mobile.
    fn decay_network(k: f64, diffusion: f64) -> ReactionNetwork {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap().with_diffusion(diffusion));
        network.add_species(Species::new("B").with_formula("C").unwrap().with_diffusion(diffusion));
        network.add_reaction(Reaction::new(
            "decay",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(k),
        ));
        network
    }

    fn run(mixture: &mut ReactingMixture, dt: f64, steps: usize) {
        let mut arena = Arena::with_capacity(1 << 16);
        for _ in 0..steps {
            mixture.step(dt, &mut arena);
        }
    }

    /// The claim M3's exit condition rests on: a balanced network on a closed grid
    /// conserves mass and every element to round-off.
    #[test]
    fn a_balanced_network_on_a_closed_grid_conserves_mass_and_atoms() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("H2").with_formula("H2").unwrap().with_diffusion(4e-3));
        network.add_species(Species::new("O2").with_formula("O2").unwrap().with_diffusion(1e-3));
        network.add_species(Species::new("H2O").with_formula("H2O").unwrap().with_diffusion(2e-3));
        network.add_reaction(Reaction::new(
            "combustion",
            vec![Term::new(0, 2.0), Term::new(1, 1.0)],
            vec![Term::new(2, 2.0)],
            RateLaw::mass_action(3.0),
        ));

        let mut mixture = ReactingMixture::new("chamber", grid(), network);
        assert!(mixture.is_conservative(), "insulated by default and balanced");

        // Two species on opposite sides, so they have to diffuse into each other before
        // anything happens — which exercises transport and reaction together.
        let g = mixture.grid();
        mixture.concentration_mut(0).unwrap().init_from_position(&g, |p| if p[0] < 0.5 { 2.0 } else { 0.0 });
        mixture.concentration_mut(1).unwrap().init_from_position(&g, |p| if p[0] >= 0.5 { 1.0 } else { 0.0 });

        let mass_before = mixture.total_mass();
        let h_before = mixture.total_element("H");
        let o_before = mixture.total_element("O");
        assert!(mass_before > 0.0);

        run(&mut mixture, 0.01, 400);

        assert!(
            (mixture.total_mass() - mass_before).abs() / mass_before < 1e-10,
            "{mass_before} -> {}",
            mixture.total_mass()
        );
        assert!((mixture.total_element("H") - h_before).abs() / h_before < 1e-10);
        assert!((mixture.total_element("O") - o_before).abs() / o_before < 1e-10);
        // And it actually reacted, or the conservation check proves nothing.
        assert!(mixture.total(2) > 0.1 * o_before, "water was produced: {}", mixture.total(2));
    }

    /// With no transport, every cell is an independent well-mixed reactor and must
    /// match the closed form exactly.
    #[test]
    fn without_diffusion_every_cell_follows_the_analytic_decay() {
        let k = 0.8;
        let mut mixture =
            ReactingMixture::new("still", grid(), decay_network(k, 0.0)).with_accuracy(0.02);
        mixture.concentration_mut(0).unwrap().fill_interior(1.0);

        run(&mut mixture, 0.05, 40);

        let expected = (-k * 2.0f64).exp();
        let field = mixture.concentration(0).unwrap();
        assert!((field.max_interior() - expected).abs() < 1e-8, "{} vs {expected}", field.max_interior());
        assert!((field.min_interior() - expected).abs() < 1e-8, "uniform, so min equals max");
    }

    /// With no reaction, the mixture must reproduce the heat module's diffusion
    /// exactly — it is the same solver, and any difference would be the splitting
    /// interfering with a half-step that should be a no-op.
    #[test]
    fn without_reaction_it_is_exactly_diffusion() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap().with_diffusion(0.01));
        let mut mixture = ReactingMixture::new("spread", grid(), network);

        let g = mixture.grid();
        mixture.concentration_mut(0).unwrap().init_from_position(&g, |p| {
            let r2 = (p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2);
            (-r2 / 0.01).exp()
        });
        let total_before = mixture.total(0);

        run(&mut mixture, 0.01, 200);

        // A closed boundary conserves the integral to round-off.
        assert!(
            (mixture.total(0) - total_before).abs() / total_before < 1e-12,
            "{total_before} -> {}",
            mixture.total(0)
        );
        // And it actually spread: the peak fell and the minimum rose.
        let field = mixture.concentration(0).unwrap();
        assert!(field.max_interior() < 0.9, "the peak spread out: {}", field.max_interior());
        assert!(field.min_interior() > 1e-6, "and reached the corners: {}", field.min_interior());
    }

    /// Strang splitting is second order. Lie–Trotter — one operator then the other —
    /// would be first, and the difference is invisible except in a convergence study.
    #[test]
    fn the_splitting_converges_at_second_order() {
        // A reaction and a diffusion on comparable timescales, so neither dominates and
        // the splitting error is actually what is being measured.
        let reference = |steps: usize| {
            let mut mixture =
                ReactingMixture::new("split", Grid2d::new(16, 16, [1.0, 1.0]), decay_network(2.0, 0.02))
                    .with_accuracy(0.005);
            let g = mixture.grid();
            mixture.concentration_mut(0).unwrap().init_from_position(&g, |p| {
                1.0 + 0.5 * (core::f64::consts::TAU * p[0]).sin()
            });
            run(&mut mixture, 0.5 / steps as f64, steps);
            interior(mixture.concentration(0).unwrap(), &g)
        };

        // Richardson: the error at h against the error at h/2, both measured against a
        // much finer run.
        let exact = reference(256);
        let error_of = |steps: usize| {
            let coarse = reference(steps);
            coarse
                .iter()
                .zip(&exact)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        };

        let coarse = error_of(8);
        let fine = error_of(16);
        let order = (coarse / fine).log2();
        assert!(coarse > 0.0, "there must be an error to measure");
        assert!(
            order > 1.7,
            "Strang should be second order, measured {order} ({coarse} -> {fine})"
        );
    }

    /// An exothermic reaction must publish heat where it happened, and the total must
    /// be the reaction extent times the enthalpy — the number the coupling ledger will
    /// check against.
    #[test]
    fn heat_is_published_where_the_reaction_happened() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(
            Reaction::new(
                "burn",
                vec![Term::new(0, 1.0)],
                vec![Term::new(1, 1.0)],
                RateLaw::mass_action(0.5),
            )
            .with_enthalpy(-100_000.0),
        );

        let mut mixture = ReactingMixture::new("chamber", grid(), network);
        // Fuel in the left half only, and nothing moves, so the heat must appear there.
        let g = mixture.grid();
        mixture.concentration_mut(0).unwrap().init_from_position(&g, |p| if p[0] < 0.5 { 1.0 } else { 0.0 });

        let mut arena = Arena::with_capacity(1 << 16);
        mixture.step(0.1, &mut arena);

        let heat = mixture.heat_release();
        let (nx, ny) = (g.nx(), g.ny());
        assert!(heat.get(nx / 4, ny / 2) > 0.0, "the fuelled half is warming");
        assert_eq!(heat.get(3 * nx / 4, ny / 2), 0.0, "the empty half is not");

        // Energy accounting: the heat released over the step must equal the extent
        // times the enthalpy, *exactly*. This is the identity the coupling ledger will
        // balance, and it is exact because the integrator accumulated the energy
        // alongside the concentrations rather than sampling a power.
        let extent = 0.5 * g.area() - mixture.total(0);
        let energy = mixture.heat_release().integrate(&g) * 0.1;
        let expected = extent * 100_000.0;
        assert!(
            (energy - expected).abs() / expected < 1e-6,
            "{energy} J against {expected} J of extent"
        );
    }

    /// The feedback loop the flagship demo turns on: a hotter cell reacts faster.
    #[test]
    fn a_temperature_field_changes_the_rate_cell_by_cell() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(Reaction::new(
            "arrhenius",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(1e6).with_activation_energy(50_000.0),
        ));

        let mut mixture = ReactingMixture::new("chamber", grid(), network);
        mixture.concentration_mut(0).unwrap().fill_interior(1.0);

        // A temperature ramp across the chamber.
        let g = mixture.grid();
        let mut temperature = ScalarField::new(&g, 1);
        temperature.init_from_position(&g, |p| 300.0 + 60.0 * p[0]);
        mixture.set_temperature_field(&temperature);

        let mut arena = Arena::with_capacity(1 << 16);
        mixture.step(1e-3, &mut arena);

        let field = mixture.concentration(1).unwrap();
        let cold = field.get(1, g.ny() / 2);
        let hot = field.get(g.nx() - 2, g.ny() / 2);
        assert!(hot > cold * 3.0, "the hot end converted much more: {cold} vs {hot}");
    }

    /// An open boundary or an unbalanced network means mass is not conserved, and the
    /// domain must not claim it is — the mistake the rigid module made with momentum.
    #[test]
    fn conservation_is_claimed_only_where_it_holds() {
        let closed = ReactingMixture::new("closed", grid(), decay_network(1.0, 0.01));
        assert!(closed.is_conservative());
        let mut out = Observations::new();
        closed.observe(&mut out);
        assert!(matches!(
            out.get("closed.total_mass").unwrap().kind,
            ObservationKind::Invariant(Invariant::Mass)
        ));

        // A Dirichlet edge is a source by construction.
        let mut open = ReactingMixture::new("open", grid(), decay_network(1.0, 0.01));
        open.set_boundaries(0, lattice_domain_grid2d::heated_edge(lattice_ir::Side::Left, 1.0));
        assert!(!open.is_conservative());
        let mut out = Observations::new();
        open.observe(&mut out);
        assert!(matches!(out.get("open.total_mass").unwrap().kind, ObservationKind::Metric));

        // So is a species that never said what it is made of: nothing can check it.
        let mut vague = ReactionNetwork::new();
        vague.add_species(Species::new("X").with_diffusion(0.01));
        let unchecked = ReactingMixture::new("vague", grid(), vague);
        assert!(!unchecked.is_conservative(), "an unbalanceable network claims nothing");
    }

    #[test]
    fn the_contract_is_complete_and_admits_what_it_does_not_conserve() {
        let gaps = CONTRACT.audit();
        assert!(gaps.is_empty(), "{gaps:?}");
        assert!(!CONTRACT.known_non_conservation.is_empty());
        assert!(
            !CONTRACT.conserves.contains(&Invariant::Energy),
            "the heat goes to the coupling; nothing here tracks energy"
        );
    }

    #[test]
    fn the_domain_publishes_what_a_reader_needs() {
        let mut mixture = ReactingMixture::new("chamber", grid(), decay_network(1.0, 0.01));
        mixture.concentration_mut(0).unwrap().fill_interior(1.0);
        run(&mut mixture, 0.01, 5);

        let mut out = Observations::new();
        mixture.observe(&mut out);
        for name in [
            "chamber.A.total",
            "chamber.A.max",
            "chamber.B.total",
            "chamber.total_mass",
            "chamber.element_C",
            "chamber.heat_release",
            "chamber.reaction_substeps",
            "chamber.exhausted_cells",
            "chamber.stiffness",
        ] {
            assert!(out.get(name).is_some(), "missing {name}");
        }
        assert_eq!(out.get("chamber.A.total").unwrap().unit, "mol");
        assert_eq!(out.get("chamber.A.max").unwrap().unit, "mol/m^2");
        assert_eq!(out.value("chamber.exhausted_cells"), Some(0.0), "nothing stiff here");

        // Every species is drawable, or a reader cannot see what the model is doing.
        assert_eq!(mixture.render_channels().len(), 2);
    }


    /// Interior cells only.
    ///
    /// `ScalarField::as_slice` hands back the whole buffer *including the halo*, and
    /// the halo holds whatever the boundary pass last wrote — an intermediate state,
    /// not part of the answer. Comparing two runs' whole buffers therefore compares
    /// their scratch space, which is how a perfectly second-order scheme comes to
    /// measure as first order.
    fn interior(field: &ScalarField, grid: &Grid2d) -> Vec<f64> {
        let mut out = Vec::with_capacity(grid.cell_count());
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                out.push(field.get(i, j));
            }
        }
        out
    }

    #[test]
    fn an_empty_mixture_steps_and_reports_without_panicking() {
        let mut mixture = ReactingMixture::new("nothing", grid(), ReactionNetwork::new());
        run(&mut mixture, 0.1, 5);
        let mut out = Observations::new();
        mixture.observe(&mut out);
        assert_eq!(out.value("nothing.total_mass"), Some(0.0));
        assert!(mixture.render_channels().is_empty());
        assert_eq!(mixture.total(0), 0.0);
        assert_eq!(mixture.total_element("H"), 0.0);
    }

    /// FR-011: the same mixture stepped twice must give identical bits.
    #[test]
    fn a_run_is_reproducible() {
        let once = || {
            let mut mixture =
                ReactingMixture::new("chamber", grid(), decay_network(1.5, 0.02));
            let g = mixture.grid();
            mixture.concentration_mut(0).unwrap().init_from_position(&g, |p| p[0] * p[1]);
            let g = mixture.grid();
            run(&mut mixture, 0.01, 50);
            interior(mixture.concentration(1).unwrap(), &g)
        };
        assert_eq!(once(), once());
    }
}
