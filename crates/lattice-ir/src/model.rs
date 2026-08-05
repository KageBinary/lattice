//! The compiled model: the immutable half of spec §7.2's split.
//!
//! > The compiled model contains topology, entity schemas, expression bytecode or
//! > kernels, material parameters, solver configuration, indices, buffer plans, and
//! > dependency information. Runtime data contains current positions, velocities,
//! > field values […] This mirrors a proven performance pattern in simulation
//! > engines: expensive parsing and indexing happen once, while the hot loop operates
//! > over preallocated low-level arrays.
//!
//! # How complete this is
//!
//! A [`CompiledModel`] holds everything about a model that does not change while it
//! runs: what domains exist, how much memory they need, the operation graph, the
//! observers, and a human-readable report of the equations and assumptions involved.
//!
//! It does **not** yet hold the solvers themselves. Today a `HeatDomain` owns both its
//! configuration and its field values, so the runtime instantiates domains from a
//! compiled model and keeps them alongside it. Splitting each solver into a schema and
//! a state block is the rest of §7.2 and is deferred rather than faked — the
//! architectural line is drawn here, and the solvers cross it one at a time.

use core::fmt;

use crate::contract::{FidelityProfile, Precision, SolverContract};
use crate::graph::OperationGraph;
use crate::ids::{BufferId, DomainId, ObserverId};

/// What a planned buffer holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BufferKind {
    /// A cell-centred scalar field with a halo.
    ScalarField {
        /// Interior cells along x.
        nx: usize,
        /// Interior cells along y.
        ny: usize,
        /// Ghost cells per side.
        halo: usize,
    },
    /// Two collocated scalar components.
    VectorField {
        /// Interior cells along x.
        nx: usize,
        /// Interior cells along y.
        ny: usize,
        /// Ghost cells per side.
        halo: usize,
    },
    /// Structure-of-arrays particle state.
    ParticleArrays {
        /// Maximum particles.
        capacity: usize,
    },
    /// A plain run of `f64` values.
    Scratch {
        /// Element count.
        elements: usize,
    },
}

impl BufferKind {
    /// Bytes this buffer occupies.
    pub fn bytes(&self) -> usize {
        const F64: usize = core::mem::size_of::<f64>();
        match *self {
            BufferKind::ScalarField { nx, ny, halo } => (nx + 2 * halo) * (ny + 2 * halo) * F64,
            BufferKind::VectorField { nx, ny, halo } => {
                2 * (nx + 2 * halo) * (ny + 2 * halo) * F64
            }
            BufferKind::ParticleArrays { capacity } => {
                capacity * crate::particles::ParticleStore::BYTES_PER_PARTICLE
            }
            BufferKind::Scratch { elements } => elements * F64,
        }
    }

    /// A short description for the plan report.
    pub fn describe(&self) -> String {
        match *self {
            BufferKind::ScalarField { nx, ny, halo } => format!("scalar {nx}x{ny} halo {halo}"),
            BufferKind::VectorField { nx, ny, halo } => format!("vector {nx}x{ny} halo {halo}"),
            BufferKind::ParticleArrays { capacity } => format!("particles cap {capacity}"),
            BufferKind::Scratch { elements } => format!("scratch {elements}"),
        }
    }
}

/// One planned buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferSpec {
    /// Its identifier.
    pub id: BufferId,
    /// A name for reports, e.g. `"temperature"`.
    pub name: String,
    /// What it holds.
    pub kind: BufferKind,
}

impl BufferSpec {
    /// Bytes this buffer occupies.
    pub fn bytes(&self) -> usize {
        self.kind.bytes()
    }
}

/// The memory the model needs, decided once at compile time.
///
/// Spec §8.4 step 6: *"Plan buffers, alignment, structure-of-arrays layouts, scratch
/// arenas, and transfer paths."* Knowing the footprint before the first step is what
/// lets NFR-001 hold — the arena is sized here, not grown in the hot loop.
#[derive(Clone, Default, Debug)]
pub struct BufferPlan {
    buffers: Vec<BufferSpec>,
    scratch_elements: usize,
}

impl BufferPlan {
    /// An empty plan.
    pub fn new() -> BufferPlan {
        BufferPlan::default()
    }

    /// Reserve a buffer and return its identifier.
    pub fn allocate(&mut self, name: impl Into<String>, kind: BufferKind) -> BufferId {
        let id = BufferId::from_index(self.buffers.len() as u32);
        self.buffers.push(BufferSpec { id, name: name.into(), kind });
        id
    }

    /// Raise the scratch requirement to at least `elements`.
    ///
    /// Callers state what they need; the plan keeps the maximum, because scratch is
    /// reused between operations rather than accumulated.
    pub fn require_scratch(&mut self, elements: usize) {
        self.scratch_elements = self.scratch_elements.max(elements);
    }

    /// Elements the scratch arena must hold.
    pub fn scratch_elements(&self) -> usize {
        self.scratch_elements
    }

    /// All planned buffers.
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.buffers
    }

    /// One buffer.
    pub fn get(&self, id: BufferId) -> Option<&BufferSpec> {
        self.buffers.get(id.index())
    }

    /// Number of buffers.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// True when nothing is planned.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Total bytes, including the scratch arena.
    pub fn total_bytes(&self) -> usize {
        self.buffers.iter().map(BufferSpec::bytes).sum::<usize>()
            + self.scratch_elements * core::mem::size_of::<f64>()
    }

    /// A readable listing.
    pub fn report(&self) -> String {
        let mut out = String::new();
        for buffer in &self.buffers {
            out.push_str(&format!(
                "  {:<28} {:<26} {:>12}\n",
                buffer.name,
                buffer.kind.describe(),
                format_bytes(buffer.bytes())
            ));
        }
        if self.scratch_elements > 0 {
            out.push_str(&format!(
                "  {:<28} {:<26} {:>12}\n",
                "(scratch arena)",
                format!("{} elements", self.scratch_elements),
                format_bytes(self.scratch_elements * core::mem::size_of::<f64>())
            ));
        }
        out.push_str(&format!("  {:<28} {:<26} {:>12}\n", "total", "", format_bytes(self.total_bytes())));
        out
    }
}

/// One solver instance in the compiled model.
#[derive(Clone, Debug)]
pub struct DomainSpec {
    /// Its identifier.
    pub id: DomainId,
    /// The instance name from the model.
    pub name: String,
    /// The solver family, e.g. `"grid2d.heat"` or `"particles2d"`.
    pub family: String,
    /// A one-line summary of the configuration.
    pub summary: String,
    /// Buffers this domain owns.
    pub buffers: Vec<BufferId>,
    /// The published contract, once the solver has been instantiated.
    pub contract: Option<&'static SolverContract>,
}

/// A declared measurement.
#[derive(Clone, Debug)]
pub struct ObserverSpec {
    /// Its identifier.
    pub id: ObserverId,
    /// What is observed, as written in the model.
    pub target: String,
    /// Sampling interval in seconds; `None` means every step.
    pub interval: Option<f64>,
}

impl ObserverSpec {
    /// How often this observer samples, for the report.
    pub fn cadence(&self) -> String {
        match self.interval {
            Some(seconds) => format!("every {seconds} s"),
            None => "every step".to_string(),
        }
    }
}

/// A declared visual encoding. Carried through compilation so a headless run records
/// what an interactive one would have drawn (spec FR-013).
#[derive(Clone, Debug)]
pub struct VisualSpec {
    /// What is drawn.
    pub target: String,
    /// How, if the model said.
    pub style: Option<String>,
}

/// A compiled, runnable model.
#[derive(Clone, Debug)]
pub struct CompiledModel {
    /// The project name.
    pub name: String,
    /// Spatial dimensionality. Always 2 for now (spec P9).
    pub dimensions: u32,
    /// The declared fidelity profile.
    pub fidelity: FidelityProfile,
    /// The declared numeric precision.
    pub precision: Precision,
    /// Solver instances.
    pub domains: Vec<DomainSpec>,
    /// Planned memory.
    pub buffers: BufferPlan,
    /// The scheduled operations.
    pub graph: OperationGraph,
    /// Declared measurements.
    pub observers: Vec<ObserverSpec>,
    /// Declared visual encodings.
    pub visuals: Vec<VisualSpec>,
    /// The timestep the model asked for, if it named one.
    pub timestep: Option<f64>,
    /// How long to run, if the model said.
    pub duration: Option<f64>,
    /// Anything the compiler wants the reader to know: approximations applied,
    /// defaults chosen, features deferred to a later milestone.
    pub notes: Vec<String>,
}

impl CompiledModel {
    /// The model report of spec §8.4 step 9.
    ///
    /// > Emit the immutable CompiledModel plus a model report describing equations
    /// > and assumptions.
    ///
    /// This is what `lattice check` prints. It exists so that a user can see what the
    /// compiler decided *before* running anything — which solver was selected, what
    /// it assumes, what memory it will take, and what the compiler had to approximate.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("project {}\n", self.name));
        out.push_str(&format!("  dimensions   {}\n", self.dimensions));
        out.push_str(&format!("  fidelity     {}\n", self.fidelity));
        out.push_str(&format!("  guarantee    {}\n", self.fidelity.guarantee()));
        out.push_str(&format!("  precision    {}\n", self.precision.code()));
        if let Some(dt) = self.timestep {
            out.push_str(&format!("  timestep     {dt} s\n"));
        }
        if let Some(duration) = self.duration {
            out.push_str(&format!("  duration     {duration} s\n"));
        }

        out.push_str(&format!("\ndomains ({})\n", self.domains.len()));
        for domain in &self.domains {
            out.push_str(&format!("  {} : {}\n      {}\n", domain.name, domain.family, domain.summary));
        }

        if !self.observers.is_empty() {
            out.push_str(&format!("\nobservers ({})\n", self.observers.len()));
            for observer in &self.observers {
                out.push_str(&format!("  {:<32} {}\n", observer.target, observer.cadence()));
            }
        }

        if !self.visuals.is_empty() {
            out.push_str(&format!("\nvisuals ({})\n", self.visuals.len()));
            for visual in &self.visuals {
                match &visual.style {
                    Some(style) => out.push_str(&format!("  {:<32} as {style}\n", visual.target)),
                    None => out.push_str(&format!("  {}\n", visual.target)),
                }
            }
        }

        out.push_str("\nbuffer plan\n");
        out.push_str(&self.buffers.report());

        out.push_str("\noperation graph\n");
        out.push_str(&self.graph.report());

        // Contracts last: they are the longest section, and P1 wants them present
        // rather than prominent.
        for domain in &self.domains {
            if let Some(contract) = domain.contract {
                out.push_str(&format!("\nsolver contract for `{}`\n", domain.name));
                out.push_str(&contract.report());
            }
        }

        if !self.notes.is_empty() {
            out.push_str("\ncompiler notes\n");
            for note in &self.notes {
                out.push_str(&format!("  - {note}\n"));
            }
        }
        out
    }

    /// Total planned memory.
    pub fn memory_bytes(&self) -> usize {
        self.buffers.total_bytes()
    }
}

impl fmt::Display for CompiledModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report())
    }
}

fn format_bytes(bytes: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{value:.2} {}", UNITS[unit]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Operation, OperationKind};

    #[test]
    fn buffer_sizes_account_for_the_halo() {
        // A 10x10 interior with one ghost cell per side is 12x12 stored.
        let kind = BufferKind::ScalarField { nx: 10, ny: 10, halo: 1 };
        assert_eq!(kind.bytes(), 12 * 12 * 8);
        // A vector field is two of them.
        assert_eq!(BufferKind::VectorField { nx: 10, ny: 10, halo: 1 }.bytes(), 2 * 12 * 12 * 8);
        assert_eq!(BufferKind::Scratch { elements: 100 }.bytes(), 800);
    }

    #[test]
    fn a_plan_totals_its_buffers_and_scratch() {
        let mut plan = BufferPlan::new();
        let temperature = plan.allocate("temperature", BufferKind::ScalarField { nx: 4, ny: 4, halo: 1 });
        plan.allocate("species_a", BufferKind::ScalarField { nx: 4, ny: 4, halo: 1 });
        plan.require_scratch(100);

        assert_eq!(plan.len(), 2);
        assert_eq!(plan.get(temperature).unwrap().name, "temperature");
        assert_eq!(plan.total_bytes(), 2 * 6 * 6 * 8 + 100 * 8);
    }

    /// Scratch is reused between operations, so the plan keeps the maximum request
    /// rather than the sum. Summing would over-allocate by the number of solvers.
    #[test]
    fn scratch_takes_the_maximum_not_the_sum() {
        let mut plan = BufferPlan::new();
        plan.require_scratch(100);
        plan.require_scratch(500);
        plan.require_scratch(50);
        assert_eq!(plan.scratch_elements(), 500);
    }

    #[test]
    fn buffer_ids_are_dense_and_in_order() {
        let mut plan = BufferPlan::new();
        for index in 0..5u32 {
            let id = plan.allocate(format!("b{index}"), BufferKind::Scratch { elements: 1 });
            assert_eq!(id.raw(), index);
        }
        assert!(plan.get(BufferId::from_index(99)).is_none());
    }

    fn sample_model() -> CompiledModel {
        let mut buffers = BufferPlan::new();
        let temperature = buffers.allocate("temperature", BufferKind::ScalarField { nx: 64, ny: 64, halo: 1 });
        buffers.require_scratch(4096);

        let graph = OperationGraph::build(vec![
            Operation::new("advance heat", OperationKind::Advance)
                .reading(temperature)
                .writing(temperature),
            Operation::new("total energy", OperationKind::Observe).reading(temperature),
        ]);

        CompiledModel {
            name: "demo".to_string(),
            dimensions: 2,
            fidelity: FidelityProfile::Engineering2d,
            precision: Precision::Accurate64,
            domains: vec![DomainSpec {
                id: DomainId::from_index(0),
                name: "temperature".to_string(),
                family: "grid2d.heat".to_string(),
                summary: "64x64 Crank-Nicolson, insulated".to_string(),
                buffers: vec![temperature],
                contract: None,
            }],
            buffers,
            graph,
            observers: vec![ObserverSpec {
                id: ObserverId::from_index(0),
                target: "total_energy".to_string(),
                interval: Some(0.1),
            }],
            visuals: vec![VisualSpec {
                target: "temperature".to_string(),
                style: Some("heatmap".to_string()),
            }],
            timestep: Some(0.01),
            duration: Some(10.0),
            notes: vec!["chemistry is deferred to milestone M3".to_string()],
        }
    }

    /// The report is what `lattice check` prints, and it has to answer "what did the
    /// compiler decide?" without the user running anything.
    #[test]
    fn the_model_report_covers_every_section() {
        let text = sample_model().report();
        for expected in [
            "project demo",
            "fidelity     F1",
            "guarantee",
            "domains (1)",
            "grid2d.heat",
            "observers (1)",
            "every 0.1 s",
            "visuals (1)",
            "as heatmap",
            "buffer plan",
            "temperature",
            "(scratch arena)",
            "operation graph",
            "advance heat",
            "compiler notes",
            "milestone M3",
        ] {
            assert!(text.contains(expected), "report is missing `{expected}`:\n{text}");
        }
    }

    #[test]
    fn observers_describe_their_cadence() {
        let every_step = ObserverSpec {
            id: ObserverId::from_index(0),
            target: "norm".to_string(),
            interval: None,
        };
        assert_eq!(every_step.cadence(), "every step");
        let periodic = ObserverSpec { interval: Some(0.25), ..every_step };
        assert_eq!(periodic.cadence(), "every 0.25 s");
    }

    #[test]
    fn memory_is_reported_in_readable_units() {
        let model = sample_model();
        assert!(model.memory_bytes() > 0);
        let text = model.report();
        assert!(text.contains("KiB") || text.contains("MiB"), "{text}");
    }

    #[test]
    fn byte_formatting_scales() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.00 KiB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.00 MiB");
    }
}
