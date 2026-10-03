//! The `quantum2d` domain: a wavefunction, its Hamiltonian, and what is observed of it.

use std::sync::{Mutex, OnceLock, PoisonError};

use lattice_ir::{
    Curve, Domain, FidelityProfile, Grid2d, Invariant, ObservationKind, Observations, Precision,
    ScalarField, SolverContract, StableStep, StepContext,
};

use crate::complex::Complex;
use crate::crank_nicolson::{CrankNicolson, Solve};
use crate::detector::Detector;
use crate::hamiltonian::{Hamiltonian, Kinetic, Moments, Spectral};
use crate::split_step::SplitStep;
use crate::wavefunction::Wavefunction;

/// The time integrator, which also fixes what the grid's edges are.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Scheme {
    /// Split-step Fourier: spectral kinetic operator, periodic edges.
    SplitStepFourier,
    /// Crank–Nicolson: finite-difference kinetic operator, `ψ = 0` walls.
    CrankNicolson,
}

impl Scheme {
    /// Name used in reports and project files.
    pub const fn name(self) -> &'static str {
        match self {
            Scheme::SplitStepFourier => "split_step_fourier",
            Scheme::CrankNicolson => "crank_nicolson",
        }
    }

    /// The kinetic discretization this scheme applies.
    pub const fn kinetic(self) -> Kinetic {
        match self {
            Scheme::SplitStepFourier => Kinetic::Spectral,
            Scheme::CrankNicolson => Kinetic::FiniteDifference,
        }
    }
}

/// The largest kinetic phase per step, `E_max Δt / ħ` at the grid's Nyquist corner,
/// that split-step's default step allows.
///
/// A potential with a sharp edge — a wall, a rectangular barrier — has Fourier content
/// all the way to the grid's highest wavenumber, and that is where the neglected
/// commutator `[T, V]` lives. Measured: a 0.5 eV electron through a 1 eV, 0.5 nm barrier
/// on a 0.05 nm grid, where `E_max` is 150 eV, was transmitted 1.9% too often at 7.5
/// radians per step and 0.06% too often at 1.9. Two radians it is.
pub const SPLIT_STEP_PHASE: f64 = 2.0;

/// A drawable field computed on demand and kept until the state changes.
///
/// Drawing is rare next to stepping — a viewer draws once a frame, a headless run once
/// at the end — and the phase costs an `atan2` per cell, which on a 512² grid is as long
/// as the step itself. So a step only marks the field stale, and the first draw after
/// it recomputes it, reusing the previous allocation.
#[derive(Debug, Default)]
struct LazyField {
    current: OnceLock<ScalarField>,
    spare: Mutex<Option<ScalarField>>,
}

impl LazyField {
    fn invalidate(&mut self) {
        if let Some(field) = self.current.take() {
            *self.spare.get_mut().unwrap_or_else(PoisonError::into_inner) = Some(field);
        }
    }

    fn get(&self, grid: Grid2d, fill: impl FnOnce(&mut ScalarField)) -> &ScalarField {
        self.current.get_or_init(|| {
            let spare = self.spare.lock().unwrap_or_else(PoisonError::into_inner).take();
            let mut field = spare.unwrap_or_else(|| ScalarField::new(&grid, 0));
            fill(&mut field);
            field
        })
    }
}

#[derive(Debug)]
enum Propagator {
    Split(Box<SplitStep>),
    Crank(Box<CrankNicolson>),
}

/// A single particle's wavefunction evolving on a 2D grid.
///
/// Build with [`QuantumDomain::new`], set the state with
/// [`QuantumDomain::with_state`], then step it like any other domain.
#[derive(Debug)]
pub struct QuantumDomain {
    name: String,
    scheme: Scheme,
    hamiltonian: Hamiltonian,
    psi: Wavefunction,
    propagator: Propagator,
    preferred_dt: f64,
    steps: u64,
    /// Probability removed by the absorbing layer so far.
    absorbed: f64,
    last_solve: Option<Solve>,
    detectors: Vec<Detector>,
    /// Observations need a Fourier transform, and `observe` takes `&self`.
    spectral: Mutex<Spectral>,
    density: LazyField,
    phase: LazyField,
    potential_field: ScalarField,
    channel_names: [String; 3],
}

impl QuantumDomain {
    /// A domain named `name` whose Hamiltonian is `hamiltonian`, starting from no
    /// particle at all until [`QuantumDomain::with_state`] gives it one.
    ///
    /// The scheme follows from the Hamiltonian's kinetic operator: spectral means
    /// split-step Fourier, finite difference means Crank–Nicolson.
    pub fn new(name: impl Into<String>, hamiltonian: Hamiltonian) -> Self {
        let name = name.into();
        let grid = hamiltonian.grid();
        let scheme = match hamiltonian.kinetic() {
            Kinetic::Spectral => Scheme::SplitStepFourier,
            Kinetic::FiniteDifference => Scheme::CrankNicolson,
        };
        let propagator = match scheme {
            Scheme::SplitStepFourier => Propagator::Split(Box::new(SplitStep::new(&hamiltonian))),
            Scheme::CrankNicolson => Propagator::Crank(Box::new(CrankNicolson::new(&hamiltonian))),
        };
        let mut potential_field = ScalarField::new(&grid, 0);
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                potential_field.set(i, j, hamiltonian.potential().values()[j * grid.nx() + i]);
            }
        }
        // A step that resolves the fastest phase the state is likely to hold: a tenth
        // of a radian of the larger of the potential's depth and a modest fraction of
        // the grid's top kinetic energy. Split-step is held tighter still — see
        // `SPLIT_STEP_PHASE`. Models set their own.
        let hbar = hamiltonian.hbar();
        let scale = hamiltonian.potential().max_abs().max(0.01 * hamiltonian.max_kinetic());
        let mut preferred_dt = 0.1 * hbar / scale;
        if scheme == Scheme::SplitStepFourier {
            preferred_dt = preferred_dt.min(SPLIT_STEP_PHASE * hbar / hamiltonian.max_kinetic());
        }
        Self {
            channel_names: [
                format!("{name}.probability_density"),
                format!("{name}.phase"),
                format!("{name}.potential"),
            ],
            name,
            scheme,
            psi: Wavefunction::zeros(grid),
            propagator,
            preferred_dt,
            steps: 0,
            absorbed: 0.0,
            last_solve: None,
            detectors: Vec::new(),
            spectral: Mutex::new(Spectral::new(grid)),
            density: LazyField::default(),
            phase: LazyField::default(),
            potential_field,
            hamiltonian,
        }
    }

    /// Set the state.
    ///
    /// # Panics
    ///
    /// If it lives on a different grid.
    pub fn with_state(mut self, psi: Wavefunction) -> Self {
        assert_eq!(psi.grid(), self.hamiltonian.grid(), "the state must live on the domain's grid");
        self.psi = psi;
        self.restart();
        self
    }

    /// Set the step this domain asks for.
    pub fn with_preferred_step(mut self, dt: f64) -> Self {
        assert!(dt > 0.0 && dt.is_finite(), "a timestep must be positive, got {dt}");
        self.preferred_dt = dt;
        self
    }

    /// Set Crank–Nicolson's solver tolerance. Ignored by split-step.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        if let Propagator::Crank(propagator) = self.propagator {
            self.propagator = Propagator::Crank(Box::new(propagator.with_tolerance(tolerance)));
        }
        self
    }

    /// Add a screen at `x` metres.
    pub fn with_detector(mut self, name: impl Into<String>, x: f64) -> Self {
        let mut detector = Detector::new(name, self.hamiltonian.grid(), x);
        detector.reset(&self.hamiltonian, &self.psi);
        self.detectors.push(detector);
        self
    }

    /// Clear accumulated records and redraw, after the state changed.
    fn restart(&mut self) {
        self.absorbed = 0.0;
        self.steps = 0;
        for detector in &mut self.detectors {
            detector.reset(&self.hamiltonian, &self.psi);
        }
        self.refresh_channels();
    }

    /// Mark the drawable fields stale; they are recomputed when next drawn.
    fn refresh_channels(&mut self) {
        self.density.invalidate();
        self.phase.invalidate();
    }

    /// `|ψ|²` or `arg ψ`, row by row. The fields have no halo, so a row of the field and
    /// a row of `ψ` line up element for element.
    fn draw<'a>(&'a self, field: &'a LazyField, f: fn(Complex) -> f64) -> &'a ScalarField {
        let grid = self.hamiltonian.grid();
        field.get(grid, |out| {
            for (j, amplitudes) in self.psi.as_slice().chunks_exact(grid.nx()).enumerate() {
                for (cell, z) in out.row_mut(j).iter_mut().zip(amplitudes) {
                    *cell = f(*z);
                }
            }
        })
    }

    /// The scheme.
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// The Hamiltonian.
    pub fn hamiltonian(&self) -> &Hamiltonian {
        &self.hamiltonian
    }

    /// The state.
    pub fn state(&self) -> &Wavefunction {
        &self.psi
    }

    /// The grid.
    pub fn grid(&self) -> Grid2d {
        self.hamiltonian.grid()
    }

    /// Steps taken.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Probability removed by the absorbing layer so far.
    pub fn absorbed(&self) -> f64 {
        self.absorbed
    }

    /// The detectors.
    pub fn detectors(&self) -> &[Detector] {
        &self.detectors
    }

    /// How the last Crank–Nicolson solve went.
    pub fn last_solve(&self) -> Option<Solve> {
        self.last_solve
    }

    /// Norm, momentum and energy of the current state.
    pub fn moments(&self) -> Moments {
        let mut spectral = self.spectral.lock().unwrap_or_else(PoisonError::into_inner);
        spectral.measure(&self.hamiltonian, &self.psi)
    }

    /// The contract for a scheme, with or without an absorbing layer.
    pub fn contract_for(scheme: Scheme, absorbing: bool) -> &'static SolverContract {
        match (scheme, absorbing) {
            (Scheme::SplitStepFourier, false) => &SPLIT_STEP_CONTRACT,
            (Scheme::SplitStepFourier, true) => &SPLIT_STEP_ABSORBING_CONTRACT,
            (Scheme::CrankNicolson, false) => &CRANK_NICOLSON_CONTRACT,
            (Scheme::CrankNicolson, true) => &CRANK_NICOLSON_ABSORBING_CONTRACT,
        }
    }
}

const fn contract_for(scheme: Scheme, absorbing: bool) -> SolverContract {
    SolverContract {
        name: match (scheme, absorbing) {
            (Scheme::SplitStepFourier, false) => "quantum2d[split_step_fourier]",
            (Scheme::SplitStepFourier, true) => "quantum2d[split_step_fourier+absorbing]",
            (Scheme::CrankNicolson, false) => "quantum2d[crank_nicolson]",
            (Scheme::CrankNicolson, true) => "quantum2d[crank_nicolson+absorbing]",
        },
        summary: "one non-relativistic particle's wavefunction on a 2D grid, by the \
                  time-dependent Schrödinger equation",
        governing_equations: if absorbing {
            &["i hbar dpsi/dt = -(hbar^2 / 2m) laplacian(psi) + V psi - i W psi"]
        } else {
            &["i hbar dpsi/dt = -(hbar^2 / 2m) laplacian(psi) + V psi"]
        },
        discretization: match scheme {
            Scheme::SplitStepFourier => {
                "cell-centred uniform grid; kinetic operator applied exactly in Fourier space \
                 (spectral), so every representable mode has its exact dispersion; the grid \
                 is periodic"
            }
            Scheme::CrankNicolson => {
                "cell-centred uniform grid; five-point finite-difference Laplacian, second \
                 order in space, with psi = 0 on the grid's outer faces"
            }
        },
        integrator: match scheme {
            Scheme::SplitStepFourier => {
                "Strang splitting: half potential step, exact kinetic step by FFT, half \
                 potential step; second order; each factor unitary"
            }
            Scheme::CrankNicolson => {
                "Crank-Nicolson (Cayley form), second order, unitary in exact arithmetic; the \
                 positive-real system is solved by Jacobi-preconditioned BiCGSTAB"
            }
        },
        assumptions: &[
            "a single particle with no spin, no relativity, and no field quantization",
            "a static potential; the particle does not act back on anything",
            "the grid resolves the state: its momentum content lies well inside the Nyquist \
             wavenumber pi/dx, or it aliases",
            "this is a model of one particle in a given potential, not electronic structure: \
             it does not describe molecules, bonds, or chemistry (spec §5.3)",
        ],
        valid_regime: "states whose momentum content the grid resolves, under potentials that \
                       vary slowly on the scale of one step's phase",
        stability: match scheme {
            Scheme::SplitStepFourier => {
                "unconditionally stable (unitary for any dt); accuracy needs V dt/hbar small \
                 and, for a potential with sharp edges, E_max dt/hbar below about 2, where \
                 E_max is the grid's top kinetic energy: the edges reach the highest \
                 wavenumbers, and that is where the splitting commutator is large"
            }
            Scheme::CrankNicolson => {
                "unconditionally stable (unitary for any dt); accuracy needs E dt/hbar small \
                 for the energies present, since Cayley lags the true phase by (E dt/hbar)^3/12"
            }
        },
        conserves: if absorbing { &[Invariant::ProbabilityNorm] } else { &[Invariant::ProbabilityNorm, Invariant::Energy] },
        known_non_conservation: match (scheme, absorbing) {
            (Scheme::SplitStepFourier, false) => &[
                "probability norm only to the round-off of two FFTs per step",
                "energy: the splitting conserves a nearby Hamiltonian, so <H> oscillates at \
                 O(dt^2) without secular drift",
                "nothing leaves the grid: a packet crossing one edge re-enters at the opposite \
                 one",
            ],
            (Scheme::SplitStepFourier, true) => &[
                "probability norm: the absorbing layer removes probability by design; the \
                 norm plus the absorbed total is the invariant, to round-off",
                "energy: removed with the absorbed probability, and splitting error of O(dt^2) \
                 besides",
                "the absorbing layer reflects a small fraction of what reaches it, more for \
                 slow and long-wavelength components",
            ],
            (Scheme::CrankNicolson, false) => &[
                "probability norm and <H> only to the linear solver's tolerance",
                "phase: Cayley's propagator lags the exact phase by (E dt / hbar)^3 / 12 per \
                 step, so fast components run slow",
                "the walls reflect perfectly; nothing leaves the box",
            ],
            (Scheme::CrankNicolson, true) => &[
                "probability norm: removed by the absorbing layer by design; the norm plus the \
                 absorbed total is the invariant, to the solver tolerance",
                "energy: removed with the absorbed probability",
                "the absorbing layer reflects a small fraction of what reaches it, and the walls \
                 behind it reflect whatever it lets through",
            ],
        },
        fidelity: FidelityProfile::Engineering2d,
        precisions: &[Precision::Accurate64, Precision::Deterministic64],
        deterministic: true,
        differentiable: false,
        validation_cases: match scheme {
            Scheme::SplitStepFourier => &[
                "harmonic oscillator spectrum by imaginary-time propagation",
                "free Gaussian packet: centre and spreading against the analytic solution",
                "tunneling through a rectangular barrier against the analytic transmission",
                "double slit: norm plus absorbed probability conserved",
                "absorbing layer: measured reflection",
            ],
            Scheme::CrankNicolson => &[
                "particle in a box: discrete spectrum exactly, continuum spectrum at second order",
                "norm and energy conserved to the solver tolerance",
                "agreement with split-step Fourier on a packet that never reaches the walls",
            ],
        },
        references: &[
            "Feit, Fleck & Steiger (1982). Solution of the Schrödinger equation by a spectral \
             method. J. Comput. Phys. 47, 412.",
            "Goldberg, Schey & Schwartz (1967). Computer-generated motion pictures of \
             one-dimensional quantum-mechanical transmission and reflection phenomena. \
             Am. J. Phys. 35, 177.",
            "van der Vorst (1992). Bi-CGSTAB: a fast and smoothly converging variant of Bi-CG for \
             the solution of nonsymmetric linear systems. SIAM J. Sci. Stat. Comput. 13, 631.",
            "Muga, Palao, Navarro & Egusquiza (2004). Complex absorbing potentials. Phys. Rep. \
             395, 357.",
            "Kosloff & Tal-Ezer (1986). A direct relaxation method for calculating eigenfunctions \
             and eigenvalues of the Schrödinger equation on a grid. Chem. Phys. Lett. 127, 223.",
        ],
    }
}

static SPLIT_STEP_CONTRACT: SolverContract = contract_for(Scheme::SplitStepFourier, false);
static SPLIT_STEP_ABSORBING_CONTRACT: SolverContract = contract_for(Scheme::SplitStepFourier, true);
static CRANK_NICOLSON_CONTRACT: SolverContract = contract_for(Scheme::CrankNicolson, false);
static CRANK_NICOLSON_ABSORBING_CONTRACT: SolverContract = contract_for(Scheme::CrankNicolson, true);

impl Domain for QuantumDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn contract(&self) -> &'static SolverContract {
        Self::contract_for(self.scheme, self.hamiltonian.absorber().is_some())
    }

    fn stable_step(&self) -> StableStep {
        StableStep::unconditional(self.preferred_dt)
    }

    fn prepare(&mut self, _ctx: &mut StepContext<'_>) {}

    fn advance(&mut self, dt: f64, _ctx: &mut StepContext<'_>) {
        match &mut self.propagator {
            Propagator::Split(propagator) => {
                self.absorbed += propagator.step(&self.hamiltonian, &mut self.psi, dt);
            }
            Propagator::Crank(propagator) => {
                let (absorbed, solve) = propagator.step(&self.hamiltonian, &mut self.psi, dt);
                self.absorbed += absorbed;
                self.last_solve = Some(solve);
            }
        }
        for detector in &mut self.detectors {
            detector.record(&self.hamiltonian, &self.psi, dt);
        }
        self.steps += 1;
        self.refresh_channels();
    }

    fn observe(&self, out: &mut Observations) {
        let prefix = &self.name;
        let moments = self.moments();
        let absorbing = self.hamiltonian.absorber().is_some();
        let energy_unit = Invariant::Energy.si_unit();

        // With an absorber the norm falls on purpose; what must not change is the norm
        // plus what was absorbed. Publishing the norm as an invariant there would report
        // every packet that reaches the edge as a conservation failure.
        if absorbing {
            out.record_metric(format!("{prefix}.probability_norm"), moments.norm, "1");
            out.record_metric(format!("{prefix}.probability_absorbed"), self.absorbed, "1");
            out.record_invariant(
                format!("{prefix}.probability_accounted"),
                Invariant::ProbabilityNorm,
                moments.norm + self.absorbed,
            );
            out.record_metric(format!("{prefix}.energy"), moments.energy(), energy_unit);
        } else {
            out.record_invariant(format!("{prefix}.probability_norm"), Invariant::ProbabilityNorm, moments.norm);
            out.record_invariant(format!("{prefix}.energy"), Invariant::Energy, moments.energy());
        }
        out.record_metric(format!("{prefix}.kinetic_energy"), moments.kinetic, energy_unit);
        out.record_metric(format!("{prefix}.potential_energy"), moments.potential, energy_unit);

        let [x, y] = self.psi.mean_position();
        out.record_metric(format!("{prefix}.mean_x"), x, "m");
        out.record_metric(format!("{prefix}.mean_y"), y, "m");
        out.record_metric(format!("{prefix}.mean_px"), moments.momentum[0], "kg·m/s");
        out.record_metric(format!("{prefix}.mean_py"), moments.momentum[1], "kg·m/s");

        let grid = self.hamiltonian.grid();
        for detector in &self.detectors {
            out.record_metric(format!("{prefix}.{}.arrived", detector.name()), detector.arrived(grid), "1");
        }

        if self.scheme == Scheme::CrankNicolson {
            let (iterations, residual) = self.last_solve.map_or((0.0, 0.0), |s| (s.iterations as f64, s.residual));
            out.record(format!("{prefix}.solver_iterations"), iterations, "1", ObservationKind::Count);
            out.record(format!("{prefix}.solver_residual"), residual, "1", ObservationKind::Residual);
        }
    }

    fn curves(&self) -> Vec<Curve> {
        let grid = self.hamiltonian.grid();
        self.detectors
            .iter()
            .map(|detector| {
                let (y, density) = detector.pattern(grid);
                Curve::new(format!("{}.{}", self.name, detector.name()), ("y", "m"), ("arrivals", "1/m"), y, density)
                    .note(format!(
                        "time-integrated probability current through x = {:.4e} m (the cell face nearest \
                         the requested {:.4e} m)",
                        detector.face_x(grid),
                        detector.x()
                    ))
                    .note(format!("{:.6} of the probability has arrived", detector.arrived(grid)))
            })
            .collect()
    }

    fn render_channels(&self) -> Vec<lattice_ir::RenderChannel<'_>> {
        let grid = self.hamiltonian.grid();
        vec![
            lattice_ir::RenderChannel::Scalar {
                name: &self.channel_names[0],
                field: self.draw(&self.density, Complex::norm_sqr),
                grid,
                unit: "1/m^2",
            },
            lattice_ir::RenderChannel::Scalar {
                name: &self.channel_names[1],
                field: self.draw(&self.phase, Complex::arg),
                grid,
                unit: "rad",
            },
            lattice_ir::RenderChannel::Scalar {
                name: &self.channel_names[2],
                field: &self.potential_field,
                grid,
                unit: "J",
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use lattice_ir::Arena;

    use super::*;
    use crate::potential::Absorber;

    const HBAR: f64 = 1.0;

    fn grid() -> Grid2d {
        Grid2d::with_origin(64, 32, [16.0, 8.0], [-8.0, -4.0])
    }

    fn run(domain: &mut QuantumDomain, dt: f64, steps: usize) {
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
        }
    }

    #[test]
    fn every_contract_is_complete() {
        for scheme in [Scheme::SplitStepFourier, Scheme::CrankNicolson] {
            for absorbing in [false, true] {
                let contract = QuantumDomain::contract_for(scheme, absorbing);
                assert!(contract.audit().is_empty(), "{}: {:?}", contract.name, contract.audit());
            }
        }
    }

    #[test]
    fn an_absorbing_run_publishes_the_norm_as_a_metric_and_the_total_as_the_invariant() {
        let h = Hamiltonian::new(grid(), 1.0, HBAR, Kinetic::Spectral)
            .with_absorber(Absorber::for_speed(grid(), 2.0, 2.0, HBAR));
        let psi = Wavefunction::gaussian(grid(), [2.0, 0.0], [1.0, 1.0], [2.0, 0.0], HBAR);
        let mut domain = QuantumDomain::new("q", h).with_state(psi).with_detector("screen", 4.0);
        run(&mut domain, 0.01, 400);

        let mut out = Observations::new();
        domain.observe(&mut out);
        let find = |name: &str| out.iter().find(|o| o.name == name).unwrap_or_else(|| panic!("no {name}")).clone();
        assert!(matches!(find("q.probability_norm").kind, ObservationKind::Metric));
        let total = find("q.probability_accounted");
        assert!(matches!(total.kind, ObservationKind::Invariant(Invariant::ProbabilityNorm)));
        assert!((total.value - 1.0).abs() < 1e-12, "{:e}", total.value - 1.0);
        assert!(domain.absorbed() > 0.5);

        let arrived = find("q.screen.arrived").value;
        assert!(arrived > 0.5 && arrived < 1.0 + 1e-6, "{arrived}");
        let curves = domain.curves();
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].name, "q.screen");
    }

    #[test]
    fn render_channels_show_density_phase_and_potential() {
        let h = Hamiltonian::new(grid(), 1.0, HBAR, Kinetic::FiniteDifference);
        let psi = Wavefunction::gaussian(grid(), [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], HBAR);
        let domain = QuantumDomain::new("q", h).with_state(psi);
        let channels = domain.render_channels();
        let names: Vec<&str> = channels.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["q.probability_density", "q.phase", "q.potential"]);
        let lattice_ir::RenderChannel::Scalar { field, .. } = &channels[0] else { panic!() };
        let integral = field.sum_interior() * grid().cell_area();
        assert!((integral - 1.0).abs() < 1e-12);
    }

    #[test]
    fn crank_nicolson_reports_its_solve() {
        let h = Hamiltonian::new(grid(), 1.0, HBAR, Kinetic::FiniteDifference);
        let psi = Wavefunction::gaussian(grid(), [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], HBAR);
        let mut domain = QuantumDomain::new("q", h).with_state(psi);
        run(&mut domain, 0.02, 10);
        let solve = domain.last_solve().unwrap();
        assert!(solve.converged && solve.iterations > 0, "{solve:?}");
        let mut out = Observations::new();
        domain.observe(&mut out);
        assert!(out.iter().any(|o| o.name == "q.solver_iterations"));
        let norm = out.iter().find(|o| o.name == "q.probability_norm").unwrap();
        assert!(matches!(norm.kind, ObservationKind::Invariant(Invariant::ProbabilityNorm)));
    }
}
