//! Compile the domain calls in the operation DAG once, outside the stepping loop.
//! Coupling remains a serial end-of-step exchange; observations retain their cadence.
use lattice_ir::{Arena, Domain, Executor, Grain, OperationGraph, OperationKind, StepContext};
use std::sync::Mutex;

#[derive(Debug)]
pub(crate) struct Schedule {
    levels: Vec<Level>,
    scratch: Vec<Mutex<Arena>>,
    scratch_elements: usize,
    domains: usize,
}

#[derive(Debug)]
struct Level {
    calls: Vec<Option<OperationKind>>,
    order: Vec<usize>,
    parallel: bool,
}

impl Schedule {
    pub(crate) fn new(graph: &OperationGraph, domains: usize, scratch: usize) -> Option<Self> {
        if graph.is_empty() {
            return None;
        }
        let mut counts = vec![(0, 0); domains];
        let mut levels = Vec::new();
        for nodes in graph.levels() {
            let mut calls = vec![None; domains];
            let mut order = Vec::new();
            let mut cost = 0.0;
            for id in nodes {
                let op = graph.operation(id);
                if op.kind == OperationKind::Observe {
                    continue;
                }
                // Coupling is a dedicated, globally ordered phase in this runtime.
                // Refuse a graph claiming an unsupported in-step exchange.
                assert_ne!(
                    op.kind,
                    OperationKind::Couple,
                    "in-step coupling needs an iterative execution group"
                );
                let index = op
                    .domain
                    .expect("a domain operation needs an owner")
                    .index();
                assert!(index < domains, "operation refers to an unknown domain");
                assert!(
                    calls[index].is_none(),
                    "concurrent exclusive calls on one domain"
                );
                match op.kind {
                    OperationKind::Prepare => counts[index].0 += 1,
                    OperationKind::Advance => counts[index].1 += 1,
                    _ => unreachable!(),
                }
                calls[index] = Some(op.kind);
                order.push(index);
                cost += op.estimated_cost;
            }
            if !order.is_empty() {
                // Small scenes keep their inner-loop executor and pay no worker barrier.
                let parallel = order.len() > 1 && cost >= 32_768.0;
                levels.push(Level {
                    calls,
                    order,
                    parallel,
                });
            }
        }
        assert!(
            counts.iter().all(|&c| c == (1, 1)),
            "each domain must prepare and advance exactly once per step"
        );
        // The common small-scene graph is exactly the original two-phase driver.
        // Use that driver when no level can benefit from domain parallelism. Do
        // not erase graphs with cross-domain ordering that changes these phases.
        if levels.len() == 2
            && levels.iter().all(|level| !level.parallel)
            && levels[0]
                .calls
                .iter()
                .all(|&kind| kind == Some(OperationKind::Prepare))
            && levels[1]
                .calls
                .iter()
                .all(|&kind| kind == Some(OperationKind::Advance))
        {
            return None;
        }
        Some(Self {
            levels,
            scratch: Vec::new(),
            scratch_elements: scratch,
            domains,
        })
    }

    pub(crate) fn enable_parallel(&mut self, threads: usize) {
        if threads > 1
            && self.scratch.is_empty()
            && self
                .levels
                .iter()
                .any(|l| l.parallel && l.order.len() >= threads)
        {
            self.scratch = (0..self.domains)
                .map(|_| Mutex::new(Arena::with_capacity(self.scratch_elements)))
                .collect();
        }
    }

    pub(crate) fn step(
        &self,
        domains: &mut [Box<dyn Domain>],
        arena: &mut Arena,
        executor: &Executor,
        time: f64,
        step: u64,
        dt: f64,
    ) {
        for level in &self.levels {
            if level.parallel && executor.threads() > 1 && level.order.len() >= executor.threads() {
                // The existing pool lends disjoint domain slots. Each has its own arena;
                // inner kernels stay scalar to avoid nested dispatch and oversubscription.
                executor.for_each_chunk_mut(domains, Grain::new(0, 1), |offset, chunk| {
                    for (local, domain) in chunk.iter_mut().enumerate() {
                        let index = offset + local;
                        if let Some(kind) = level.calls[index] {
                            let mut scratch =
                                self.scratch[index].lock().expect("domain scratch poisoned");
                            let mut ctx = StepContext {
                                time,
                                step,
                                arena: &mut scratch,
                                executor: Executor::shared_sequential(),
                            };
                            call(domain.as_mut(), kind, dt, &mut ctx);
                        }
                    }
                });
            } else {
                let mut ctx = StepContext {
                    time,
                    step,
                    arena,
                    executor,
                };
                for &index in &level.order {
                    call(
                        domains[index].as_mut(),
                        level.calls[index].unwrap(),
                        dt,
                        &mut ctx,
                    );
                }
            }
        }
    }
}

fn call(domain: &mut dyn Domain, kind: OperationKind, dt: f64, ctx: &mut StepContext<'_>) {
    match kind {
        OperationKind::Prepare => domain.prepare(ctx),
        OperationKind::Advance => domain.advance(dt, ctx),
        _ => unreachable!("only domain calls are compiled into this schedule"),
    }
}
