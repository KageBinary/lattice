//! Four independent heat fields, comparing the previous driver with DAG scheduling.
use lattice_domain_grid2d::HeatDomain;
use lattice_ir::{Executor, OperationGraph};
use lattice_runtime::Simulation;
use std::time::Instant;

fn run(side: usize, scheduled: bool) -> (std::time::Duration, Vec<Vec<u64>>) {
    let mut text = format!(
        "project independent {{ dimensions: 2; fidelity: engineering_2d; grid box {{ size: [{side}, {side}]; extent: [1 meter, 1 meter]; }}"
    );
    for index in 0..4 {
        text.push_str(&format!("field t{index} on box = gaussian(center=[0.4 meter, 0.6 meter], sigma=0.08 meter, peak=100 kelvin) {{ diffusivity: 1e-4 meter^2 / second; boundary: insulated; }} solve heat(t{index}) with explicit;"));
    }
    text.push('}');
    let file = lattice_syntax::SourceFile::new("schedule-bench", text);
    let (compiled, diagnostics) = lattice_compiler::compile_source(&file);
    assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
    let mut compiled = compiled.unwrap();
    if !scheduled {
        compiled.model.graph = OperationGraph::build(Vec::new());
    }
    let dt = compiled
        .domains
        .iter()
        .map(|d| d.stable_step().preferred)
        .fold(f64::INFINITY, f64::min);
    let mut sim = Simulation::coupled(compiled.model, compiled.domains, compiled.coupler)
        .with_executor(Executor::with_threads(4));
    for _ in 0..5 {
        sim.step(dt);
    }
    let start = Instant::now();
    for _ in 0..200 {
        sim.step(dt);
    }
    let elapsed = start.elapsed();
    let fields = sim
        .domains()
        .iter()
        .map(|d| {
            d.as_ref()
                .as_any()
                .downcast_ref::<HeatDomain>()
                .unwrap()
                .field()
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect();
    (elapsed, fields)
}
fn main() {
    for side in [16, 256, 512] {
        for attempt in 0..3 {
            let (old, a) = run(side, false);
            let (new, b) = run(side, true);
            assert_eq!(a, b, "scheduling changed field arithmetic");
            println!(
                "4 x {side}x{side}, 200 steps, attempt {attempt}: previous {old:?}, scheduled {new:?}, {:.2}x; fields bit-identical",
                old.as_secs_f64() / new.as_secs_f64()
            );
        }
    }
}
