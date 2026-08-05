//! The compiled operation graph.
//!
//! Spec §9.2:
//!
//! > The compiled operation graph is a directed acyclic graph within a simulation
//! > phase. Nodes declare buffer reads/writes, backend, estimated cost, and
//! > synchronization requirements. The scheduler can run independent CPU operations
//! > in parallel […] Cyclic multiphysics dependencies are represented as explicit
//! > iterative coupling groups with convergence tests, **not hidden scheduler
//! > cycles**.
//!
//! # Where the edges come from
//!
//! Operations do not declare their dependencies. They declare what they *read* and
//! what they *write*, and the edges follow:
//!
//! | Hazard | Condition | Why it orders |
//! |---|---|---|
//! | read-after-write | `i` writes what `j` reads | `j` needs `i`'s output |
//! | write-after-write | both write the same buffer | last writer must be last |
//! | write-after-read | `i` reads what `j` writes | `j` must not clobber `i`'s input |
//!
//! Direction comes from declaration order, so a graph built this way is acyclic by
//! construction. That is the point: a user cannot accidentally create a scheduler
//! cycle by writing a model. Cycles can only appear through
//! [`OperationGraph::with_dependencies`], which is how an *explicit* iterative
//! coupling group would be expressed — and that is exactly where the spec wants the
//! cycle to be visible and named.

use core::fmt;

use crate::ids::{BufferId, DomainId, OperatorId};

/// What an operation does.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum OperationKind {
    /// Update spatial indices, boundary data, coefficients and caches.
    Prepare,
    /// Advance a domain by one step.
    Advance,
    /// Transfer a quantity across a coupling edge.
    Couple,
    /// Take a read-only measurement.
    Observe,
}

impl OperationKind {
    /// Name used in the model report.
    pub const fn label(self) -> &'static str {
        match self {
            OperationKind::Prepare => "prepare",
            OperationKind::Advance => "advance",
            OperationKind::Couple => "couple",
            OperationKind::Observe => "observe",
        }
    }

    /// True for operations that only read state.
    ///
    /// Observers can always be reordered or skipped without changing the physics,
    /// which is what lets the runtime sample them on a cadence.
    pub const fn is_read_only(self) -> bool {
        matches!(self, OperationKind::Observe)
    }
}

/// One node of the graph.
#[derive(Clone, PartialEq, Debug)]
pub struct Operation {
    /// A name for reports and traces, e.g. `"advance heat"`.
    pub name: String,
    /// The domain this belongs to, if any.
    pub domain: Option<DomainId>,
    /// What it does.
    pub kind: OperationKind,
    /// Buffers read.
    pub reads: Vec<BufferId>,
    /// Buffers written.
    pub writes: Vec<BufferId>,
    /// Relative cost, used to order parallel work longest-first.
    pub estimated_cost: f64,
}

impl Operation {
    /// A new operation with no dependencies declared.
    pub fn new(name: impl Into<String>, kind: OperationKind) -> Operation {
        Operation {
            name: name.into(),
            domain: None,
            kind,
            reads: Vec::new(),
            writes: Vec::new(),
            estimated_cost: 1.0,
        }
    }

    /// Attach the owning domain.
    pub fn in_domain(mut self, domain: DomainId) -> Operation {
        self.domain = Some(domain);
        self
    }

    /// Declare a buffer this operation reads.
    pub fn reading(mut self, buffer: BufferId) -> Operation {
        self.reads.push(buffer);
        self
    }

    /// Declare a buffer this operation writes.
    pub fn writing(mut self, buffer: BufferId) -> Operation {
        self.writes.push(buffer);
        self
    }

    /// Set the relative cost estimate.
    pub fn costing(mut self, cost: f64) -> Operation {
        self.estimated_cost = cost;
        self
    }

    /// True if this operation and `other` touch a buffer in a way that orders them.
    fn conflicts_with(&self, other: &Operation) -> bool {
        let raw = self.writes.iter().any(|b| other.reads.contains(b));
        let waw = self.writes.iter().any(|b| other.writes.contains(b));
        let war = self.reads.iter().any(|b| other.writes.contains(b));
        raw || waw || war
    }
}

/// Why a graph could not be built.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GraphError {
    /// An explicit dependency named an operation that does not exist.
    UnknownOperation {
        /// The out-of-range index.
        index: usize,
        /// How many operations exist.
        count: usize,
    },
    /// The explicit dependencies form a cycle.
    ///
    /// Carries the operation names around the loop, in order, so the message can
    /// show the loop rather than just assert one exists.
    Cycle {
        /// Names of the operations forming the cycle.
        names: Vec<String>,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownOperation { index, count } => {
                write!(f, "dependency refers to operation #{index}, but only {count} exist")
            }
            GraphError::Cycle { names } => write!(
                f,
                "the operation graph contains a cycle: {} -> {}. \
                 Cyclic dependencies must be declared as an iterative coupling group \
                 with a convergence test, not left for the scheduler to resolve",
                names.join(" -> "),
                names.first().map_or("?", String::as_str)
            ),
        }
    }
}

impl core::error::Error for GraphError {}

/// A scheduled, acyclic set of operations.
#[derive(Clone, Debug)]
pub struct OperationGraph {
    operations: Vec<Operation>,
    /// `successors[i]` lists operations that must run after `i`.
    successors: Vec<Vec<usize>>,
    /// Number of operations each one must wait for.
    predecessor_counts: Vec<usize>,
    /// A valid execution order.
    order: Vec<usize>,
    /// Operations grouped so that everything in a level may run concurrently.
    levels: Vec<Vec<usize>>,
}

impl OperationGraph {
    /// Build a graph, deriving dependencies from read/write sets.
    ///
    /// Never fails: edges follow declaration order, so the result is acyclic.
    pub fn build(operations: Vec<Operation>) -> OperationGraph {
        Self::with_dependencies(operations, &[])
            .expect("edges derived from declaration order cannot cycle")
    }

    /// Build a graph with additional explicit `(before, after)` dependencies.
    ///
    /// Explicit edges are the only way to introduce a cycle, and a cycle is an error
    /// naming the loop.
    pub fn with_dependencies(
        operations: Vec<Operation>,
        extra: &[(usize, usize)],
    ) -> Result<OperationGraph, GraphError> {
        let count = operations.len();
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); count];

        // Hazard edges, always pointing forward in declaration order.
        for later in 0..count {
            for earlier in 0..later {
                if operations[earlier].conflicts_with(&operations[later]) {
                    successors[earlier].push(later);
                }
            }
        }

        for &(before, after) in extra {
            for index in [before, after] {
                if index >= count {
                    return Err(GraphError::UnknownOperation { index, count });
                }
            }
            if !successors[before].contains(&after) {
                successors[before].push(after);
            }
        }

        let mut predecessor_counts = vec![0usize; count];
        for edges in &successors {
            for &target in edges {
                predecessor_counts[target] += 1;
            }
        }

        // Kahn's algorithm, taking a whole level at a time so the levels fall out.
        let mut remaining = predecessor_counts.clone();
        let mut order = Vec::with_capacity(count);
        let mut levels: Vec<Vec<usize>> = Vec::new();
        let mut ready: Vec<usize> = (0..count).filter(|&i| remaining[i] == 0).collect();

        while !ready.is_empty() {
            // Longest-first within a level: with a fixed number of workers, starting
            // the most expensive operation first is what keeps the tail short.
            ready.sort_by(|&a, &b| {
                operations[b]
                    .estimated_cost
                    .partial_cmp(&operations[a].estimated_cost)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            order.extend_from_slice(&ready);
            levels.push(ready.clone());

            let mut next = Vec::new();
            for &node in &ready {
                for &target in &successors[node] {
                    remaining[target] -= 1;
                    if remaining[target] == 0 {
                        next.push(target);
                    }
                }
            }
            ready = next;
        }

        if order.len() != count {
            let unscheduled: Vec<usize> = (0..count).filter(|i| remaining[*i] > 0).collect();
            let names = find_cycle(&successors, &unscheduled)
                .into_iter()
                .map(|index| operations[index].name.clone())
                .collect();
            return Err(GraphError::Cycle { names });
        }

        Ok(OperationGraph { operations, successors, predecessor_counts, order, levels })
    }

    /// All operations, in declaration order.
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// One operation.
    pub fn operation(&self, id: OperatorId) -> &Operation {
        &self.operations[id.index()]
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// True when the graph has no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// A valid sequential execution order.
    pub fn order(&self) -> impl Iterator<Item = OperatorId> + '_ {
        self.order.iter().map(|&index| OperatorId::from_index(index as u32))
    }

    /// Operations grouped so everything within a group may run concurrently.
    ///
    /// Spec §9.2: *"The scheduler can run independent CPU operations in parallel."*
    /// The number of levels is the critical path length; the widest level bounds how
    /// much parallelism there is to exploit.
    pub fn levels(&self) -> Vec<Vec<OperatorId>> {
        self.levels
            .iter()
            .map(|level| level.iter().map(|&i| OperatorId::from_index(i as u32)).collect())
            .collect()
    }

    /// Operations that must run after this one.
    pub fn successors(&self, id: OperatorId) -> impl Iterator<Item = OperatorId> + '_ {
        self.successors[id.index()].iter().map(|&i| OperatorId::from_index(i as u32))
    }

    /// How many operations this one waits for.
    pub fn predecessor_count(&self, id: OperatorId) -> usize {
        self.predecessor_counts[id.index()]
    }

    /// Length of the critical path, in levels.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// The most operations that could run at once.
    pub fn max_width(&self) -> usize {
        self.levels.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Total estimated cost of every operation.
    pub fn total_cost(&self) -> f64 {
        self.operations.iter().map(|op| op.estimated_cost).sum()
    }

    /// Estimated cost along the critical path, assuming unlimited workers.
    ///
    /// The ratio of [`OperationGraph::total_cost`] to this is the best speedup
    /// parallel execution could achieve — useful for deciding whether it is worth
    /// building before it is built.
    pub fn critical_cost(&self) -> f64 {
        self.levels
            .iter()
            .map(|level| {
                level
                    .iter()
                    .map(|&i| self.operations[i].estimated_cost)
                    .fold(0.0f64, f64::max)
            })
            .sum()
    }

    /// A readable listing, for `lattice check` and the model report.
    pub fn report(&self) -> String {
        let mut out = format!(
            "{} operations in {} levels (max width {})\n",
            self.len(),
            self.depth(),
            self.max_width()
        );
        for (index, level) in self.levels.iter().enumerate() {
            out.push_str(&format!("  level {index}:\n"));
            for &node in level {
                let op = &self.operations[node];
                out.push_str(&format!("    [{}] {}", op.kind.label(), op.name));
                if !op.reads.is_empty() || !op.writes.is_empty() {
                    out.push_str(&format!(
                        "  reads {} writes {}",
                        op.reads.len(),
                        op.writes.len()
                    ));
                }
                out.push('\n');
            }
        }
        let speedup =
            if self.critical_cost() > 0.0 { self.total_cost() / self.critical_cost() } else { 1.0 };
        out.push_str(&format!(
            "  total cost {:.1}, critical path {:.1} (ideal speedup {speedup:.2}x)\n",
            self.total_cost(),
            self.critical_cost()
        ));
        out
    }
}

/// Find one cycle among the operations that could not be scheduled.
///
/// Kahn's algorithm proves a cycle exists but does not say where it is. A depth-first
/// walk over what is left recovers the actual loop, so the error can print it.
fn find_cycle(successors: &[Vec<usize>], candidates: &[usize]) -> Vec<usize> {
    let mut on_stack = vec![false; successors.len()];
    let mut visited = vec![false; successors.len()];
    let mut stack: Vec<usize> = Vec::new();

    fn walk(
        node: usize,
        successors: &[Vec<usize>],
        candidates: &[usize],
        visited: &mut [bool],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        visited[node] = true;
        on_stack[node] = true;
        stack.push(node);

        for &next in &successors[node] {
            if !candidates.contains(&next) {
                continue;
            }
            if on_stack[next] {
                let start = stack.iter().position(|&n| n == next).unwrap_or(0);
                return Some(stack[start..].to_vec());
            }
            if !visited[next]
                && let Some(cycle) = walk(next, successors, candidates, visited, on_stack, stack)
            {
                return Some(cycle);
            }
        }

        stack.pop();
        on_stack[node] = false;
        None
    }

    for &start in candidates {
        if !visited[start]
            && let Some(cycle) =
                walk(start, successors, candidates, &mut visited, &mut on_stack, &mut stack)
        {
            return cycle;
        }
    }
    candidates.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(index: u32) -> BufferId {
        BufferId::from_index(index)
    }

    fn op(name: &str, reads: &[u32], writes: &[u32]) -> Operation {
        let mut operation = Operation::new(name, OperationKind::Advance);
        for &r in reads {
            operation = operation.reading(buffer(r));
        }
        for &w in writes {
            operation = operation.writing(buffer(w));
        }
        operation
    }

    #[test]
    fn independent_operations_share_one_level() {
        let graph = OperationGraph::build(vec![
            op("a", &[0], &[1]),
            op("b", &[2], &[3]),
            op("c", &[4], &[5]),
        ]);
        assert_eq!(graph.depth(), 1, "nothing conflicts, so everything runs at once");
        assert_eq!(graph.max_width(), 3);
        assert!((graph.critical_cost() - 1.0).abs() < 1e-12);
    }

    /// Read-after-write: the consumer must wait for the producer.
    #[test]
    fn read_after_write_creates_an_edge() {
        let graph = OperationGraph::build(vec![
            op("produce", &[], &[7]),
            op("consume", &[7], &[8]),
        ]);
        assert_eq!(graph.depth(), 2);
        let order: Vec<OperatorId> = graph.order().collect();
        assert_eq!(graph.operation(order[0]).name, "produce");
        assert_eq!(graph.operation(order[1]).name, "consume");
    }

    /// Write-after-write: two writers of the same buffer must be ordered, or the
    /// result depends on which finishes first.
    #[test]
    fn write_after_write_creates_an_edge() {
        let graph = OperationGraph::build(vec![op("first", &[], &[1]), op("second", &[], &[1])]);
        assert_eq!(graph.depth(), 2);
        assert_eq!(graph.predecessor_count(OperatorId::from_index(1)), 1);
    }

    /// Write-after-read: a writer must not clobber a buffer an earlier operation is
    /// still reading.
    #[test]
    fn write_after_read_creates_an_edge() {
        let graph = OperationGraph::build(vec![op("reader", &[3], &[]), op("writer", &[], &[3])]);
        assert_eq!(graph.depth(), 2);
    }

    #[test]
    fn a_diamond_schedules_in_three_levels() {
        // a writes 1; b and c both read 1 and write their own; d reads both.
        let graph = OperationGraph::build(vec![
            op("a", &[], &[1]),
            op("b", &[1], &[2]),
            op("c", &[1], &[3]),
            op("d", &[2, 3], &[4]),
        ]);
        assert_eq!(graph.depth(), 3);
        assert_eq!(graph.max_width(), 2, "b and c are independent");
        let levels = graph.levels();
        assert_eq!(levels[1].len(), 2);
    }

    /// A graph built from read/write sets alone cannot cycle, because every edge
    /// points forward in declaration order. A user cannot write a model that
    /// deadlocks the scheduler.
    #[test]
    fn derived_edges_are_acyclic_even_for_mutually_dependent_operations() {
        // Both read and write the same two buffers — maximally conflicting.
        let graph = OperationGraph::build(vec![
            op("a", &[1, 2], &[1, 2]),
            op("b", &[1, 2], &[1, 2]),
            op("c", &[1, 2], &[1, 2]),
        ]);
        assert_eq!(graph.depth(), 3, "fully serialized, but scheduled");
        assert_eq!(graph.len(), 3);
    }

    /// Explicit dependencies are the only way to create a cycle, and the error names
    /// the loop rather than merely reporting that one exists (§9.2).
    #[test]
    fn an_explicit_cycle_is_reported_with_the_loop() {
        let operations = vec![op("heat", &[], &[1]), op("reaction", &[], &[2])];
        // heat -> reaction is derived from nothing, so add both directions explicitly.
        let error = OperationGraph::with_dependencies(operations, &[(0, 1), (1, 0)]).unwrap_err();
        match &error {
            GraphError::Cycle { names } => {
                assert_eq!(names.len(), 2);
                assert!(names.contains(&"heat".to_string()));
                assert!(names.contains(&"reaction".to_string()));
            }
            other => panic!("{other:?}"),
        }
        let text = error.to_string();
        assert!(text.contains("iterative coupling group"), "{text}");
        assert!(text.contains("->"), "the loop should be shown: {text}");
    }

    #[test]
    fn a_longer_cycle_is_found_whole() {
        let operations = vec![op("a", &[], &[1]), op("b", &[], &[2]), op("c", &[], &[3])];
        let error =
            OperationGraph::with_dependencies(operations, &[(0, 1), (1, 2), (2, 0)]).unwrap_err();
        match error {
            GraphError::Cycle { names } => assert_eq!(names.len(), 3, "{names:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_out_of_range_dependency_is_reported() {
        let operations = vec![op("a", &[], &[1])];
        let error = OperationGraph::with_dependencies(operations, &[(0, 5)]).unwrap_err();
        assert_eq!(error, GraphError::UnknownOperation { index: 5, count: 1 });
    }

    #[test]
    fn explicit_dependencies_add_ordering_without_hazards() {
        // No buffer conflict at all, but the model insists on an order.
        let graph =
            OperationGraph::with_dependencies(vec![op("a", &[], &[]), op("b", &[], &[])], &[(0, 1)])
                .unwrap();
        assert_eq!(graph.depth(), 2);
    }

    /// The most expensive operation in a level starts first, which is what shortens
    /// the tail when workers are limited.
    #[test]
    fn levels_order_expensive_operations_first() {
        let graph = OperationGraph::build(vec![
            op("cheap", &[0], &[1]).costing(1.0),
            op("expensive", &[2], &[3]).costing(100.0),
            op("medium", &[4], &[5]).costing(10.0),
        ]);
        let level = &graph.levels()[0];
        assert_eq!(graph.operation(level[0]).name, "expensive");
        assert_eq!(graph.operation(level[1]).name, "medium");
        assert_eq!(graph.operation(level[2]).name, "cheap");
    }

    #[test]
    fn cost_analysis_estimates_the_available_speedup() {
        let graph = OperationGraph::build(vec![
            op("a", &[0], &[1]).costing(10.0),
            op("b", &[2], &[3]).costing(10.0),
            op("c", &[4], &[5]).costing(10.0),
            op("join", &[1, 3, 5], &[6]).costing(5.0),
        ]);
        assert!((graph.total_cost() - 35.0).abs() < 1e-12);
        // Three parallel 10s then a 5: critical path is 15.
        assert!((graph.critical_cost() - 15.0).abs() < 1e-12);
        let text = graph.report();
        assert!(text.contains("ideal speedup 2.33x"), "{text}");
    }

    #[test]
    fn an_empty_graph_is_valid() {
        let graph = OperationGraph::build(Vec::new());
        assert!(graph.is_empty());
        assert_eq!(graph.depth(), 0);
        assert_eq!(graph.max_width(), 0);
        assert_eq!(graph.order().count(), 0);
    }

    #[test]
    fn the_report_lists_every_operation() {
        let graph = OperationGraph::build(vec![
            Operation::new("prepare cells", OperationKind::Prepare).writing(buffer(0)),
            Operation::new("advance particles", OperationKind::Advance).reading(buffer(0)),
            Operation::new("energy", OperationKind::Observe).reading(buffer(0)),
        ]);
        let text = graph.report();
        assert!(text.contains("prepare cells"), "{text}");
        assert!(text.contains("[advance]"), "{text}");
        assert!(text.contains("[observe]"), "{text}");
        // One `level N:` heading per level. Counting bare "level" would also match
        // the "N levels" in the summary line.
        let headings = text.lines().filter(|line| line.trim_start().starts_with("level ")).count();
        assert_eq!(headings, graph.depth(), "{text}");
    }

    #[test]
    fn observers_are_marked_read_only() {
        assert!(OperationKind::Observe.is_read_only());
        assert!(!OperationKind::Advance.is_read_only());
        assert!(!OperationKind::Couple.is_read_only());
    }

    /// A realistic frame: prepare, advance two domains, couple them, observe.
    /// This is the shape spec Figure 1 describes.
    #[test]
    fn a_realistic_frame_schedules_sensibly() {
        let (heat_field, species_field, source) = (buffer(0), buffer(1), buffer(2));
        let graph = OperationGraph::build(vec![
            Operation::new("prepare heat", OperationKind::Prepare).writing(heat_field).costing(1.0),
            Operation::new("prepare species", OperationKind::Prepare)
                .writing(species_field)
                .costing(1.0),
            Operation::new("advance species", OperationKind::Advance)
                .reading(species_field)
                .writing(species_field)
                .costing(20.0),
            Operation::new("reaction -> heat source", OperationKind::Couple)
                .reading(species_field)
                .writing(source)
                .costing(2.0),
            Operation::new("advance heat", OperationKind::Advance)
                .reading(heat_field)
                .reading(source)
                .writing(heat_field)
                .costing(30.0),
            Operation::new("total energy", OperationKind::Observe)
                .reading(heat_field)
                .reading(species_field)
                .costing(0.5),
        ]);

        // The two prepares are independent and go first together.
        assert_eq!(graph.levels()[0].len(), 2);
        // The observer must come last: it reads what the advances write.
        let order: Vec<String> =
            graph.order().map(|id| graph.operation(id).name.clone()).collect();
        assert_eq!(order.last().unwrap(), "total energy");
        // And the coupling sits between the species advance and the heat advance.
        let position = |name: &str| order.iter().position(|n| n == name).unwrap();
        assert!(position("advance species") < position("reaction -> heat source"));
        assert!(position("reaction -> heat source") < position("advance heat"));
    }
}
