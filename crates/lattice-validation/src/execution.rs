//! Cross-backend validation (spec §19.1).
//!
//! > | Cross-backend | Do CPU and GPU agree within tolerance? | all released kernels |
//!
//! There is no GPU yet. There are two CPU backends — the scalar path and the parallel
//! one — and the level exists now rather than waiting for `wgpu` because the harness is
//! the hard part, and because the answer here is the reference the GPU will be judged
//! against.
//!
//! # Why these cases assert zero and not a tolerance
//!
//! Between two CPU threads there is nothing for a tolerance to excuse. Both evaluate
//! the same expression on the same inputs with the same instructions; the only thing
//! the executor changes is *who* runs which band. So the criterion is exact equality of
//! the IEEE bit patterns, and any drift at all is a bug rather than a rounding budget
//! being spent.
//!
//! That is deliberately not what the GPU cases will look like. A GPU differs in fused
//! multiply-add contraction, transcendental accuracy, and reduction order, so its
//! agreement genuinely will be tolerance-based. Keeping the CPU pair exact means the
//! tolerance introduced later belongs to the GPU and can be attributed to it, rather
//! than being a number that has always been there and that nobody can account for.
//!
//! # What is deliberately not covered
//!
//! Lennard-Jones pair forces run on one thread (see `lattice_domain_particle`), so
//! `particle_trajectories_match_across_thread_counts` exercises per-particle laws only.
//! Reporting that as a passing cross-backend case would overclaim. When the pair loop
//! gains a parallel form, its agreement will not be exact, and it will need a case of
//! its own that says so.

use lattice_domain_grid2d::{gaussian, Diffusivity, HeatDomain, TimeScheme, BAND_GRAIN};
use lattice_domain_quantum2d::fft::FFT_GRAIN;
use lattice_domain_quantum2d::{Absorber, Hamiltonian, Kinetic, Potential, QuantumDomain, Shape, Wavefunction};
use lattice_domain_particle::{
    HarmonicWell, Integrator, ParticleDomain, UniformAcceleration, PARTICLE_GRAIN,
};
use lattice_ir::{Arena, BoundarySet, Domain, Executor, Grid2d, ParticleSpec, StepContext};
use lattice_runtime::{RunConfig, Simulation};

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "heat_field_matches_across_thread_counts",
        domain: "cpu",
        level: Level::CrossBackend,
        claim: "the diffusion stencil and its elementwise update give bit-identical fields on 1, 2, 3 and 8 threads",
        run: heat_field_matches,
    },
    Case {
        name: "implicit_solve_matches_across_thread_counts",
        domain: "cpu",
        level: Level::CrossBackend,
        claim: "Crank-Nicolson takes the same conjugate-gradient iterations and reaches the same field on 1, 2, 3 and 8 threads",
        run: implicit_solve_matches,
    },
    Case {
        name: "particle_trajectories_match_across_thread_counts",
        domain: "cpu",
        level: Level::CrossBackend,
        claim: "per-particle force laws and every integrator give bit-identical trajectories on 1, 2, 3 and 8 threads",
        run: particle_trajectories_match,
    },
    Case {
        name: "quantum_split_step_matches_across_thread_counts",
        domain: "cpu",
        level: Level::CrossBackend,
        claim: "split-step Fourier, transforms and phase factors split across the pool, gives a bit-identical wavefunction and absorbed probability on 1, 2, 3 and 8 threads",
        run: quantum_split_step_matches,
    },
    Case {
        name: "run_artifacts_hash_identically_across_thread_counts",
        domain: "cpu",
        level: Level::CrossBackend,
        claim: "FR-011's reproducibility hash is unchanged by the thread count, so a parallel run is a valid regression baseline for a sequential one",
        run: artifact_hash_is_thread_independent,
    },
];

/// Thread counts every case sweeps. One is the reference; the rest must match it.
///
/// Eight is included even on a machine with fewer cores. Oversubscribing does not change
/// the partition — it is computed from the *configured* thread count — so the case still
/// measures what it claims to, just with more contention.
const THREAD_COUNTS: [usize; 3] = [2, 3, 8];

/// The grid these cases run on.
///
/// Above [`BAND_GRAIN`]'s floor, and deliberately not square or power-of-two: a case
/// whose problem falls below the floor compares the sequential path with itself and
/// passes for the wrong reason, and a case whose bands all divide evenly never exercises
/// the short final band. `the_cases_actually_split_their_work` holds both properties.
const GRID: (usize, usize) = (257, 149);

/// Particles these cases run, above [`PARTICLE_GRAIN`]'s floor for the same reason.
const PARTICLE_COUNT: usize = 80_000;

/// A stiff-ish diffusion problem with a source and mixed boundaries.
///
/// Deliberately not symmetric: a uniform field or a symmetric one can survive a
/// mis-indexed band by luck, and this must not.
fn heat_model() -> HeatDomain {
    let grid = Grid2d::new(GRID.0, GRID.1, [1.0, 0.6]);
    let hot_spot = gaussian([0.3, 0.2], 0.006, 40.0);

    let mut diffusivity = lattice_ir::ScalarField::new(&grid, 1);
    diffusivity.init_from_position(&grid, |[x, y]| 1e-4 * (1.0 + 3.0 * x * x + y));

    HeatDomain::new("plate", grid, Diffusivity::Variable(diffusivity))
        .with_initial(move |[x, y]| 300.0 + hot_spot([x, y]) + 7.0 * (13.0 * x).sin() * y)
        .with_boundaries(BoundarySet::INSULATED)
        .with_scheme(TimeScheme::Explicit)
}

fn field_bits(domain: &HeatDomain) -> Vec<u64> {
    domain.field().as_slice().iter().map(|value| value.to_bits()).collect()
}

fn run_heat(executor: &Executor, steps: usize) -> Vec<u64> {
    let mut domain = heat_model();
    let mut arena = Arena::with_capacity(0);
    let dt = domain.stable_step().preferred;
    {
        let mut ctx = StepContext::new(&mut arena).with_executor(executor);
        domain.prepare(&mut ctx);
        for _ in 0..steps {
            domain.advance(dt, &mut ctx);
        }
    }
    field_bits(&domain)
}

fn heat_field_matches() -> Outcome {
    let steps = 200;
    let reference = run_heat(Executor::shared_sequential(), steps);

    let mut worst = 0usize;
    let mut notes = vec![format!(
        "{}x{} grid, split into {} bands on 2 threads",
        GRID.0,
        GRID.1,
        band_count(2)
    )];
    for threads in THREAD_COUNTS {
        let executor = Executor::with_threads(threads);
        let observed = run_heat(&executor, steps);
        let differing = observed.iter().zip(&reference).filter(|(a, b)| a != b).count();
        worst = worst.max(differing);
        notes.push(format!("{} cells differ on {}", differing, executor.label()));
    }

    let mut outcome = Outcome::at_most("differing cells after 200 steps", "cells", worst as f64, 0.0)
        .note(format!("{} cells compared per thread count", reference.len()));
    for note in notes {
        outcome = outcome.note(note);
    }
    outcome
}

/// Bands the heat grid is split into at a given thread count.
///
/// One band means the executor decided the problem was not worth splitting, and a case
/// running on one band is comparing the sequential path with itself.
fn band_count(threads: usize) -> usize {
    Executor::with_threads(threads).partition(GRID.1, BAND_GRAIN.per_row(GRID.0)).count()
}

/// Chunks the particle population is split into at a given thread count. See
/// [`band_count`].
fn particle_chunk_count(threads: usize) -> usize {
    Executor::with_threads(threads).partition(PARTICLE_COUNT, PARTICLE_GRAIN).count()
}

/// The implicit path, which is the sharpest instrument available here.
///
/// Conjugate gradient stops on `‖r‖ ≤ tol·‖b‖`. A single-ULP difference anywhere in the
/// stencil or the right-hand side moves the residual, and sooner or later moves it
/// across the threshold on some iteration — at which point the two runs take a different
/// number of iterations and diverge visibly. The iteration count is therefore a
/// discrete amplifier of a difference too small to see in the field itself, which is why
/// this case reports it as well as the field.
fn run_implicit_heat(executor: &Executor, steps: usize) -> (Vec<u64>, usize) {
    let mut domain = heat_model().with_scheme(TimeScheme::CrankNicolson);
    let mut arena = Arena::with_capacity(0);
    let mut iterations = 0;
    {
        let mut ctx = StepContext::new(&mut arena).with_executor(executor);
        domain.prepare(&mut ctx);
        for _ in 0..steps {
            domain.advance(0.5, &mut ctx);
            iterations += domain.last_solve().map_or(0, |outcome| outcome.iterations());
        }
    }
    (field_bits(&domain), iterations)
}

fn implicit_solve_matches() -> Outcome {
    let steps = 25;
    let (reference, reference_iterations) = run_implicit_heat(Executor::shared_sequential(), steps);

    let mut worst_cells = 0usize;
    let mut worst_iteration_gap = 0usize;
    let mut notes = vec![format!("{reference_iterations} CG iterations over {steps} steps, sequential")];
    for threads in THREAD_COUNTS {
        let (observed, iterations) = run_implicit_heat(&Executor::with_threads(threads), steps);
        worst_cells = worst_cells.max(observed.iter().zip(&reference).filter(|(a, b)| a != b).count());
        worst_iteration_gap =
            worst_iteration_gap.max(iterations.abs_diff(reference_iterations));
        notes.push(format!("{threads} threads: {iterations} iterations"));
    }

    let mut outcome = Outcome::at_most(
        "differing cells, plus any change in the iteration count",
        "cells",
        (worst_cells + worst_iteration_gap) as f64,
        0.0,
    );
    for note in notes {
        outcome = outcome.note(note);
    }
    outcome
}

fn particle_domain() -> ParticleDomain {
    let mut domain = ParticleDomain::new("cloud", PARTICLE_COUNT)
        .with_integrator(Integrator::VelocityVerlet)
        .with_force(UniformAcceleration::earth_gravity())
        .with_force(HarmonicWell::new([0.5, 0.5], 3.0));
    for index in 0..PARTICLE_COUNT {
        let t = index as f64 * 7.0e-4;
        domain
            .spawn(
                ParticleSpec::at([0.5 + t.cos() * 0.3, 0.5 + (2.0 * t).sin() * 0.25])
                    .with_velocity([(3.0 * t).sin(), t.cos() * 0.5])
                    .with_mass(0.2 + 0.8 * t.sin().abs()),
            )
            .expect("the store was sized for these particles");
    }
    domain
}

fn run_particles(executor: &Executor, integrator: Integrator, steps: usize) -> Vec<u64> {
    let mut domain = particle_domain().with_integrator(integrator);
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena).with_executor(executor);
    domain.prepare(&mut ctx);
    for _ in 0..steps {
        domain.advance(1e-3, &mut ctx);
    }
    let store = domain.store();
    store
        .pos_x()
        .iter()
        .chain(store.pos_y())
        .chain(store.vel_x())
        .chain(store.vel_y())
        .map(|value| value.to_bits())
        .collect()
}

fn particle_trajectories_match() -> Outcome {
    let steps = 120;
    let schemes =
        [Integrator::ExplicitEuler, Integrator::SemiImplicitEuler, Integrator::VelocityVerlet];

    let mut worst = 0usize;
    let mut compared = 0usize;
    let mut notes = vec![format!(
        "{PARTICLE_COUNT} particles, split into {} chunks on 2 threads",
        particle_chunk_count(2)
    )];
    for scheme in schemes {
        let reference = run_particles(Executor::shared_sequential(), scheme, steps);
        compared = reference.len();
        let mut scheme_worst = 0usize;
        for threads in THREAD_COUNTS {
            let observed = run_particles(&Executor::with_threads(threads), scheme, steps);
            scheme_worst =
                scheme_worst.max(observed.iter().zip(&reference).filter(|(a, b)| a != b).count());
        }
        worst = worst.max(scheme_worst);
        notes.push(format!("{}: {} values differ", scheme.name(), scheme_worst));
    }

    let mut outcome = Outcome::at_most(
        format!("differing state values after {steps} steps"),
        "values",
        worst as f64,
        0.0,
    )
    .note(format!("{compared} values compared per scheme per thread count"));
    for note in notes {
        outcome = outcome.note(note);
    }
    outcome
}

/// The quantum grid: above [`FFT_GRAIN`]'s floor in both passes, and neither side a
/// power of two, so the Bluestein path — the one with per-thread scratch — runs in both.
const QUANTUM_GRID: (usize, usize) = (96, 75);

/// FFT bands at a given thread count, for the row pass and the column pass.
fn fft_band_counts(threads: usize) -> (usize, usize) {
    let executor = Executor::with_threads(threads);
    let (nx, ny) = QUANTUM_GRID;
    (executor.partition(ny, FFT_GRAIN.per_row(nx)).count(), executor.partition(nx, FFT_GRAIN.per_row(ny)).count())
}

/// A packet through a barrier into an absorbing layer, so the sequential absorbed sum is
/// exercised beside the split transforms.
fn run_quantum(executor: &Executor, steps: usize) -> (Vec<u64>, u64) {
    let (nx, ny) = QUANTUM_GRID;
    let grid = Grid2d::with_origin(nx, ny, [24.0, 18.0], [-12.0, -9.0]);
    let mut potential = Potential::zero(grid);
    potential.add(&Shape::Rectangle { x: [1.0, 2.0], y: [-9.0, 9.0], height: 2.0 });
    let h = Hamiltonian::new(grid, 1.0, 1.0, Kinetic::Spectral)
        .with_potential(potential)
        .with_absorber(Absorber::for_speed(grid, 3.0, 2.0, 1.0));
    let psi = Wavefunction::gaussian(grid, [-3.0, 0.5], [1.2, 1.0], [2.0, 0.3], 1.0);
    let mut domain = QuantumDomain::new("q", h).with_state(psi);
    let dt = domain.stable_step().preferred;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena).with_executor(executor);
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
    }
    let bits = domain.state().as_slice().iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect();
    (bits, domain.absorbed().to_bits())
}

fn quantum_split_step_matches() -> Outcome {
    let steps = 150;
    let (reference, reference_absorbed) = run_quantum(Executor::shared_sequential(), steps);
    let (rows, columns) = fft_band_counts(2);
    let mut worst = 0usize;
    let mut notes = vec![format!(
        "{}x{} grid (Bluestein both ways), split into {rows} row bands and {columns} column bands on 2 threads",
        QUANTUM_GRID.0, QUANTUM_GRID.1
    )];
    for threads in THREAD_COUNTS {
        let (observed, absorbed) = run_quantum(&Executor::with_threads(threads), steps);
        let differing = observed.iter().zip(&reference).filter(|(a, b)| a != b).count()
            + usize::from(absorbed != reference_absorbed);
        worst = worst.max(differing);
        notes.push(format!("{threads} threads: {differing} values differ"));
    }
    let mut outcome = Outcome::at_most(
        format!("differing amplitudes, plus the absorbed total, after {steps} steps"),
        "values",
        worst as f64,
        0.0,
    )
    .note(format!(
        "absorbed probability {:.6} — the layer is doing work",
        f64::from_bits(reference_absorbed)
    ));
    for note in notes {
        outcome = outcome.note(note);
    }
    outcome
}

/// The end-to-end statement: FR-011's hash does not move.
///
/// This is what makes the promise usable rather than merely true. A regression baseline
/// recorded on a build machine with sixteen cores has to be comparable with a run on a
/// laptop with four, or the hash is not a regression signal at all.
fn artifact_hash_is_thread_independent() -> Outcome {
    let hash_with = |executor: Executor| {
        let domain = heat_model();
        let model = lattice_ir::CompiledModel {
            name: "cross_backend_plate".to_string(),
            dimensions: 2,
            fidelity: lattice_ir::FidelityProfile::Engineering2d,
            precision: lattice_ir::Precision::Accurate64,
            domains: Vec::new(),
            buffers: lattice_ir::BufferPlan::new(),
            graph: lattice_ir::OperationGraph::build(Vec::new()),
            observers: Vec::new(),
            visuals: Vec::new(),
            timestep: None,
            duration: None,
            notes: Vec::new(),
        };
        let mut simulation =
            Simulation::new(model, vec![Box::new(domain)]).with_executor(executor);
        let outcome =
            simulation.run(&RunConfig::new().with_max_steps(300).with_timestep(1.0e-3));
        assert!(outcome.is_success(), "{}", outcome.stop.describe());
        (outcome.artifact.content_hash(), outcome.artifact.to_json().to_compact_string())
    };

    let (reference_hash, reference_json) = hash_with(Executor::sequential());
    let mut mismatches = 0usize;
    let mut notes = vec![format!("sequential hash {reference_hash:016x}")];
    for threads in THREAD_COUNTS {
        let (hash, json) = hash_with(Executor::with_threads(threads));
        if hash != reference_hash {
            mismatches += 1;
        }
        notes.push(format!("{threads} threads: {hash:016x}"));
        // The execution schedule *is* recorded — it is simply outside the hash.
        debug_assert!(json.contains("cpu-parallel"), "the artifact should say how it ran");
    }
    let _ = reference_json;

    let mut outcome =
        Outcome::at_most("thread counts whose artifact hash differs", "runs", mismatches as f64, 0.0);
    for note in notes {
        outcome = outcome.note(note);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard that keeps these cases from quietly becoming vacuous.
    ///
    /// Every kernel here declines to split a problem below its grain's floor. That is
    /// correct — a barrier is not free — but it means a cross-backend case run on a
    /// small problem compares the sequential path with *itself* and passes for a reason
    /// that has nothing to do with what it claims. Raising a floor is a legitimate
    /// tuning decision that would silently do exactly that, so the case sizes are pinned
    /// to the floors here rather than to a comment.
    #[test]
    fn the_cases_actually_split_their_work() {
        for threads in THREAD_COUNTS {
            assert!(
                band_count(threads) > 1,
                "the {}x{} grid is not split on {threads} threads (grain floor is {} rows)",
                GRID.0,
                GRID.1,
                BAND_GRAIN.per_row(GRID.0).floor
            );
            let (rows, columns) = fft_band_counts(threads);
            assert!(
                rows > 1 && columns > 1,
                "the {:?} quantum grid is not split on {threads} threads: {rows} row bands, \
                 {columns} column bands",
                QUANTUM_GRID
            );
            assert!(
                particle_chunk_count(threads) > 1,
                "{PARTICLE_COUNT} particles are not split on {threads} threads \
                 (grain floor is {})",
                PARTICLE_GRAIN.floor
            );
        }
    }

    /// A partition whose chunks all divide evenly never exercises the short final chunk,
    /// which is where an off-by-one in a band's bounds would live.
    #[test]
    fn the_cases_produce_a_short_final_chunk() {
        let rows_per_band =
            Executor::with_threads(3).partition(GRID.1, BAND_GRAIN.per_row(GRID.0)).units_per_chunk();
        assert_ne!(GRID.1 % rows_per_band, 0, "{} rows divide evenly by {rows_per_band}", GRID.1);
    }

    /// The implicit case has to actually iterate, or it proves nothing: a solve that
    /// converges in zero iterations would agree across thread counts trivially.
    #[test]
    fn the_implicit_case_does_real_work() {
        let (_, iterations) = run_implicit_heat(Executor::shared_sequential(), 25);
        assert!(iterations >= 25, "only {iterations} CG iterations over 25 steps");
    }

    #[test]
    fn every_registered_case_passes() {
        for case in CASES {
            let outcome = (case.run)();
            assert!(outcome.passed(), "{}: {:?}", case.name, outcome);
        }
    }
}
