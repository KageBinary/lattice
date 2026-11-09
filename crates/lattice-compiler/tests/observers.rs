//! Observer laws through a whole run: compile a model, attach what it declares, run it.

use lattice_compiler::compile_source;
use lattice_ir::Executor;
use lattice_runtime::{RunConfig, RunOutcome, Simulation, StopReason};
use lattice_syntax::SourceFile;

/// A free gas of `count` atoms, `law` declared and `observes` as its tail.
fn gas(count: usize, extra: &str, observes: &str) -> String {
    format!(
        "project gas {{
           dimensions: 2;
           duration: 0.9 second;
           {extra}
           particles atoms {{
             count: {count};
             region: [10 meter, 10 meter];
             spacing: 1 meter;
             mass: 2 kilogram;
             speed: 1 meter / second;
             seed: 5;
           }}
           solve dynamics(atoms) with velocity_verlet(dt=0.01 second);
           {observes}
         }}"
    )
}

const KINETIC: &str = "observer kinetic(a: particle) -> joule {
    return 0.5 * a.mass * dot(a.velocity, a.velocity);
}";

fn run_source(source: &str, threads: usize) -> (SourceFile, Simulation, RunOutcome) {
    let file = SourceFile::new("gas.lattice", source);
    let (compiled, diagnostics) = compile_source(&file);
    assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
    let compiled = compiled.expect("a model");
    let config = RunConfig::from_model(&compiled.model);
    let mut simulation = Simulation::coupled(compiled.model, compiled.domains, compiled.coupler)
        .with_executor(Executor::with_threads(threads));
    simulation.add_observers(compiled.observers).expect("the model's observers attach");
    let outcome = simulation.run(&config);
    (file, simulation, outcome)
}

/// `(time, value)` of every sample that recorded `row`.
fn series(outcome: &RunOutcome, row: &str) -> Vec<(f64, f64)> {
    outcome
        .artifact
        .timeline()
        .iter()
        .filter_map(|s| s.values.iter().find(|(name, _)| name == row).map(|(_, v)| (s.time, *v)))
        .collect()
}

/// Spec M6.2 §7, cadence: two law observers at `every 0.1` and `every 0.3` record in a
/// 3:1 ratio, and the slower one's rows sit at the expected times.
#[test]
fn two_law_observers_sample_at_their_own_cadence() {
    let (_, _, outcome) = run_source(
        &gas(
            4,
            KINETIC,
            "observe sum(kinetic(atoms)) every 0.1 second;
             observe max(kinetic(atoms)) every 0.3 second;",
        ),
        1,
    );
    assert!(outcome.is_success(), "{}", outcome.stop.describe());
    let fast = series(&outcome, "atoms.kinetic.sum");
    let slow = series(&outcome, "atoms.kinetic.max");
    // Both record the initial state; after it, nine samples of the first to three of the
    // second.
    assert_eq!((fast.len() - 1, slow.len() - 1), (9, 3));
    let times: Vec<f64> = slow.iter().map(|(t, _)| *t).collect();
    for (got, want) in times.iter().zip([0.0, 0.3, 0.6, 0.9]) {
        assert!((got - want).abs() < 1e-9, "{times:?}");
    }
    // Every row of the slower one is also a sample of the faster, at the same time.
    for (time, _) in &slow {
        assert!(fast.iter().any(|(t, _)| t == time), "no fast sample at {time}");
    }
}

/// Law-observer rows are the same bits whatever the thread count (spec M6.2 §4.3: the CPU
/// pool never runs an observer, and the state it reads is the same on every count).
#[test]
fn law_observer_rows_do_not_depend_on_the_thread_count() {
    let source = gas(
        64,
        KINETIC,
        "observe sum(kinetic(atoms)) every 0.1 second;
         observe mean(kinetic(atoms)) every 0.1 second;
         observe min(kinetic(atoms)) every 0.3 second;
         observe max(kinetic(atoms)) every 0.1 second;",
    );
    let bits = |outcome: &RunOutcome| -> Vec<(u64, u64)> {
        ["sum", "mean", "min", "max"]
            .iter()
            .flat_map(|reduction| series(outcome, &format!("atoms.kinetic.{reduction}")))
            .map(|(t, v)| (t.to_bits(), v.to_bits()))
            .collect()
    };
    let (_, _, reference) = run_source(&source, 1);
    assert!(reference.is_success());
    let reference = bits(&reference);
    assert_eq!(reference.len(), 10 + 10 + 4 + 10);
    for threads in [2, 3, 8] {
        let (_, _, outcome) = run_source(&source, threads);
        assert_eq!(bits(&outcome), reference, "{threads} threads");
    }
}

/// Spec M6.2 §7, non-finite: `ln(a.position.x)` with an atom at negative x halts the run
/// on the observer's row, and the sampled state explains it: the operation is `ln`, and
/// its operand is the position it was handed.
#[test]
fn a_non_finite_observer_row_halts_the_run_and_is_explained_at_the_operation() {
    let source = gas(
        4,
        "observer log_x(a: particle) -> 1 { return ln(a.position.x / 1 meter); }",
        "observe sum(log_x(atoms)) every 0.1 second;",
    )
    .replace("spacing: 1 meter;", "spacing: 1 meter; origin: [-1.5 meter, 0 meter];");
    let (file, simulation, outcome) = run_source(&source, 1);
    let StopReason::NonFinite { observation, .. } = &outcome.stop else {
        panic!("expected a halt, got {:?}", outcome.stop)
    };
    assert_eq!(observation, "atoms.log_x.sum");
    let fault = simulation.explain_observer_halt(observation).expect("the observer explains its row");
    assert_eq!((fault.domain.as_str(), fault.law.as_str()), ("atoms", "log_x"));
    assert_eq!(fault.site, lattice_ir::FaultSite::Particle(0));
    assert!(fault.input.is_none(), "an operation made it, not an input");
    assert!(matches!(fault.at.op, lattice_expr::Op::Ln(_)), "{:?}", fault.at.op);
    let origin = fault.at.origin.expect("a source range");
    assert_eq!(&file.text()[origin.start as usize..origin.end as usize], "ln(a.position.x / 1 meter)");
    assert_eq!(fault.at.operands.len(), 1);
    assert!(fault.at.operands[0] < 0.0 && fault.at.operands[0].is_finite());
    // A row nothing published is not explained.
    assert!(simulation.explain_observer_halt("atoms.count").is_none());
}
