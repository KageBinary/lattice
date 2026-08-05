PROJECT LATTICE
A High-Performance, Expressive 2D Multiphysics, Chemistry, and Quantum Simulation Runtime
[IMAGE]
Working architecture vision — specialized solvers unified by one typed simulation runtime.
Technical & Product Specification
Version 0.1  |  Concept specification  |  August 2026
Status: ambitious design proposal; implementation choices and numerical claims require benchmarking and validation.

# Document Control

<TABLE>
| Field | Value |
| Working title | Project Lattice |
| Document type | Technical architecture, product requirements, and implementation roadmap |
| Primary author/owner | Solo founder / research-engineering project |
| Initial dimensionality | 2D-first; architecture must remain dimension-agnostic where practical |
| Primary implementation stance | Software-only, commodity desktop hardware, portable CPU/GPU execution |
| Long-term ambition | A general substrate for interactive physics, chemistry, molecular, and externally coupled quantum simulations |
</TABLE>

<TABLE>
| The central thesis / Lattice should not attempt to create one magical numerical method that solves every physical regime. It should create a common language, runtime, coupling model, optimization layer, and debugging experience in which specialized solvers can coexist, exchange state, and declare their assumptions explicitly. |
</TABLE>

# Contents

1. Executive Summary
2. Product Definition and Problem Statement
3. Design Principles
4. Users and Representative Workflows
5. Scope, Fidelity, and Scientific Boundaries
6. Product and System Requirements
7. System Architecture
8. Simulation Intermediate Representation
9. Runtime, Time, and Scheduling
10. Numerical Foundations
11. Physics Domain Modules
12. Chemistry and Molecular Simulation
13. Quantum Mechanics and Quantum-Chemistry Integration
14. Multiphysics Coupling
15. Performance and Optimization Architecture
16. Expressiveness, DSL, APIs, and Plugins
17. Visualization and Scientific Debugging
18. Data, Reproducibility, and Interchange
19. Validation, Testing, and Benchmarking
20. Solo-Developer MVP
21. Milestone Roadmap
22. Product and Business Paths
23. Risks and Failure Modes
24. Recommended Repository Structure
25. Example Models
26. Open Research Questions
27. Glossary
28. References

# 1. Executive Summary

Project Lattice is a proposed high-performance simulation engine and authoring environment for constructing, running, inspecting, and coupling physical models. The first implementation is intentionally two-dimensional so that a solo developer can build a coherent, visible, and testable system without the geometry, rendering, and computational burden of a full 3D engine. The architecture, however, treats dimensionality as a property of a domain and data layout rather than a permanent product limitation.
The long-term vision spans ordinary mechanics, particles, fluids, heat transfer, waves, electromagnetism, chemical kinetics, molecular dynamics, and selected quantum workflows. These regimes cannot be made scientifically honest by forcing them into one solver. Lattice instead provides a universal simulation substrate: a typed model, units and dimensions, domain-specific solver contracts, an explicit coupling graph, a compiled execution plan, portable compute backends, deterministic replay, diagnostics, and a lightweight visual environment.
The most important product distinction is not merely breadth. Existing engines are often either fast but domain-specific, expressive but difficult to optimize, educational but physically shallow, or scientifically powerful but inaccessible and visually fragmented. Lattice aims to make the relationship between model meaning and execution explicit: users describe physical state and laws; the compiler selects and schedules kernels; solvers publish stability and validity constraints; and the runtime continuously displays numerical error, conservation drift, and fidelity assumptions.
<TABLE>
| First release definition / The MVP is a 2D desktop application and Python/Rust SDK that supports particles, rigid bodies, scalar/vector fields, heat and diffusion PDEs, reaction-diffusion, chemical reaction networks, Lennard-Jones molecular dynamics, and a small educational 2D Schrödinger solver. It includes a typed scene format, CPU execution, one portable GPU backend, basic visualizations, deterministic recording, and a benchmark/validation suite. |
</TABLE>

## 1.1 What “universal” means

One shared model and runtime can host multiple solver domains.
Domains communicate through typed ports with explicit units, cadence, interpolation, and conservation semantics.
A user can replace force laws, integrators, constitutive models, reactions, boundary conditions, and visual encodings.
The engine can trade accuracy for speed only through named fidelity profiles, never silently.
Advanced external solvers can be treated as backends or co-simulation components instead of being reimplemented poorly.

## 1.2 What “optimized” means

Optimization is a whole-system property. The design therefore includes data-oriented storage, model compilation, memory preallocation, spatial acceleration structures, kernel fusion, backend specialization, adaptive update rates, asynchronous output, and profiler-guided execution. It does not define success as a single impressive particle count detached from numerical error or model usefulness.
[IMAGE]
Figure 1. A frame is compiled into a dependency-aware sequence of domain and coupling operations.

# 2. Product Definition and Problem Statement

## 2.1 Problem statement

Scientific and educational simulation software is fragmented by domain. Rigid-body engines do not naturally express reaction kinetics. molecular simulators do not provide accessible interactive field editing. quantum-chemistry packages provide highly specialized electronic-structure calculations but are not general visual physics environments. multiphysics tools can be extremely capable but often demand specialized meshing, proprietary workflows, or heavyweight configuration. Users who want to mix domains typically glue together multiple programs and manually reconcile units, timesteps, geometry, state, and output.
Lattice addresses this fragmentation with a common substrate while preserving specialized numerical methods. Its target is not to surpass every mature solver in its home domain. Its target is to make domain composition, experimentation, customization, visualization, and deployment substantially easier while preserving a path to high performance and external validation.

## 2.2 Product statement

<TABLE>
| Product statement / For students, educators, researchers, and technical builders who need to explore interacting physical and chemical systems, Lattice is a 2D-first simulation engine that combines an expressive typed model with portable high-performance execution and visible numerical diagnostics. Unlike a game engine or a single-domain scientific package, Lattice treats solver assumptions, coupling, fidelity, units, and error as first-class parts of the model. |
</TABLE>

## 2.3 Competitive design precedents—not direct product copies

<TABLE>
| System | Relevant precedent | Lesson adopted | Boundary |
| MuJoCo | Compiled model separated from dynamic data; preallocated runtime structures; rich scene language | Compile human-readable models into low-level runtime layouts | Lattice covers fields, chemistry, and multiple solver families rather than articulated mechanics alone. |
| LAMMPS | Extensible particle simulation, neighbor lists, domain decomposition, CPU/GPU parallelism | Use domain-native acceleration structures and extension points | Lattice does not initially target distributed billion-particle scale. |
| OpenMM | Library/application split and custom force expressions compiled for hardware | Separate low-level engine from approachable authoring and allow custom kernels | Lattice includes non-molecular domains and an explicit multiphysics coupling graph. |
| Taichi / NVIDIA Warp | JIT-compiled numerical kernels and portable accelerated execution | Compile high-level numerical definitions into specialized kernels | Lattice should own its model semantics and validation rather than expose arbitrary kernels as the primary product. |
| ModelingToolkit / FMI | Composable equation-based models and standardized co-simulation interfaces | Represent model structure explicitly and support external components | Lattice begins with an opinionated interactive runtime rather than a general symbolic language alone. |
| PySCF / Quantum ESPRESSO | Mature modular electronic-structure methods | Integrate serious quantum calculations rather than reimplementing them superficially | The built-in quantum module remains small and educational until external adapters mature. |
</TABLE>

# 3. Design Principles

P1 — Scientific honesty over feature count: Every module must state governing equations, discretization, assumptions, valid regimes, error metrics, and known non-conservation. A visually convincing result is not sufficient.
P2 — Specialize solvers; unify orchestration: Rigid dynamics, fluids, molecular dynamics, and quantum mechanics use different numerical structures. The runtime unifies state, units, coupling, scheduling, diagnostics, and authoring—not the equations themselves.
P3 — Compile models before running: User-friendly declarations should become validated, cross-indexed, preallocated runtime structures. Expensive interpretation, graph construction, and memory planning should occur before the hot loop.
P4 — Data-oriented hot paths: Runtime data should be structured for contiguous access, vectorization, coalesced GPU access, and minimal allocation. Object-oriented APIs may exist above the runtime but should not dictate memory layout.
P5 — Fidelity is explicit: Fast approximations, educational models, engineering models, and external high-fidelity calculations must be named and selectable. Switching fidelity should preserve the conceptual model where possible.
P6 — Determinism is a feature: CPU deterministic mode, seeded stochasticity, event logs, checkpoints, and replay should make regressions and scientific comparison possible.
P7 — Visualization is instrumentation: The renderer is not decoration. It must display fields, fluxes, constraints, forces, residuals, conservation drift, timestep decisions, and uncertainty.
P8 — Customization through typed extension points: Users may add equations, forces, reactions, materials, emitters, observers, and solvers, but extensions declare units, read/write sets, side effects, backend support, and differentiability.
P9 — Start 2D, preserve generality: The first engine optimizes 2D execution and interaction. Core concepts should avoid assumptions that make a future 3D backend impossible.
P10 — External excellence beats internal imitation: When a mature package already solves a hard domain well, Lattice should integrate through adapters, co-simulation, and data exchange before attempting replacement.

# 4. Users and Representative Workflows

<TABLE>
| Persona | Primary need | First useful workflow | Long-term value |
| Student / learner | See how equations produce behavior and test counterfactuals | Build a 2D gas, change temperature and particle mass, inspect energy | A programmable laboratory for mechanics, chemistry, fields, and quantum concepts. |
| Educator | Create interactive demonstrations without writing a full engine | Author a reaction-diffusion or electric-field scene and publish it | Reusable lessons, assignments, parameter sweeps, and embedded simulations. |
| Technical hobbyist | Construct custom physical systems and unusual rules | Define custom forces and materials in a script | A “modding platform” for physics and chemistry. |
| Research engineer | Prototype coupled models before moving to heavyweight solvers | Couple heat, flow, and reaction kinetics in 2D | Rapid hypothesis testing, inverse problems, and external-solver orchestration. |
| AI/optimization researcher | Differentiate through simulations or search parameter spaces | Fit parameters to observed trajectories | A controllable environment for system identification and scientific ML. |
| Product developer | Embed a simulation in another app | Run headless from Rust/Python and stream state | SDK licensing, specialized vertical products, and cloud execution. |
</TABLE>

## 4.1 Representative “hero” scenarios

Coupled exothermic reaction: Two species diffuse through a chamber, react according to a configurable rate law, release heat, and change local diffusivity and fluid buoyancy.
Molecular gas to continuum comparison: Run a 2D Lennard-Jones gas beside a continuum pressure/temperature field and compare macroscopic observables.
Electric separation experiment: Charged particles move through a fluid under an electric field with drag, collisions, and optional Brownian motion.
Quantum double-slit: Propagate a 2D wave packet through configurable barriers and compare probability density with classical particles.
Reaction pathway workbench: Define a reaction network, fit rate constants, sweep temperature, and optionally request external electronic-structure energies for selected molecular geometries.
Physics sandbox: Create rigid and soft obstacles, particle emitters, fields, sensors, plots, and scripted events through the editor or Python API.

# 5. Scope, Fidelity, and Scientific Boundaries

[IMAGE]
Figure 2. Lattice exposes a ladder of fidelity rather than implying that all simulations are equally predictive.

## 5.1 Fidelity profiles

<TABLE>
| Profile | Purpose | Typical methods | Guarantee |
| F0 Interactive | Immediate visual exploration and teaching | semi-implicit Euler, simplified collision response, low-resolution grids, qualitative chemistry | Stable and responsive; not automatically predictive. |
| F1 Engineering 2D | Quantitative small-scale models with validation | symplectic integrators, finite difference/volume PDEs, iterative linear solvers, convergence tests | Published benchmark error and conservation metrics. |
| F2 Molecular / kinetic | Particle and stochastic molecular behavior | Verlet integration, neighbor lists, thermostats, force fields, SSA/tau-leaping | Correct implementation of declared model, not ab initio chemistry. |
| F3 Research coupling | Inverse problems and complex coupled workflows | adaptive timesteps, operator splitting, iterative coupling, autodiff where supported | Model-specific validation and explicit convergence controls. |
| F4 External quantum | Electronic structure and reaction-path calculations | PySCF, Psi4, Quantum ESPRESSO, other plugins | Results inherit the assumptions and validation of the selected backend. |
</TABLE>

## 5.2 Explicit non-goals for the first major version

A full replacement for mature 3D finite-element, CFD, molecular-dynamics, or electronic-structure packages.
Automatic prediction of arbitrary chemical reactions from molecular structures without validated models.
Production-grade ab initio reaction dynamics implemented from scratch.
Photorealistic rendering or a game-engine asset pipeline.
Distributed supercomputer execution in the initial architecture.
Claims of real-world predictive accuracy without convergence studies and experimental validation.
One-click conversion of any equation into a stable high-performance solver.

## 5.3 Scientific boundary for “quantum chemistry”

Lattice should distinguish three meanings that are frequently conflated. First, a built-in grid-based Schrödinger module can visualize wave packets, potentials, tunneling, eigenstates, and interference in one or two dimensions. Second, molecular mechanics and reactive force-field modules can model atoms using parameterized classical potentials. Third, real electronic-structure calculations—Hartree–Fock, density-functional theory, correlated wavefunction methods, reaction pathways, and periodic solids—should initially run through external quantum-chemistry backends. This separation prevents an educational wavefunction visualizer from being marketed as a general chemical reaction predictor.

# 6. Product and System Requirements

## 6.1 Functional requirements

<TABLE>
| ID | Priority | Requirement | Acceptance evidence |
| FR-001 | Must | Load a human-readable simulation project containing domains, entities, materials, fields, equations, reactions, boundaries, observers, and visual layers. | Reference project loads with zero ambiguity and produces a canonical compiled model. |
| FR-002 | Must | Validate physical units and dimensional consistency at model compile time wherever expressions are statically known. | Invalid force/energy/rate expressions fail with source-positioned diagnostics. |
| FR-003 | Must | Support particles and rigid 2D bodies with forces, impulses, constraints, collisions, and sensors. | Validation suite reproduces analytic free-fall, oscillator, momentum, and collision cases. |
| FR-004 | Must | Support scalar and vector fields on uniform 2D grids. | Users can create, initialize, sample, render, and export fields. |
| FR-005 | Must | Support heat/diffusion, reaction-diffusion, and configurable boundary conditions. | Convergence tests match analytic or manufactured solutions. |
| FR-006 | Must | Support chemical reaction networks with deterministic ODE and stochastic execution modes. | Mass-action reference cases match independent calculations. |
| FR-007 | Must | Support 2D classical molecular dynamics with pair potentials and neighbor lists. | Lennard-Jones benchmarks conserve energy within declared tolerance. |
| FR-008 | Should | Support an incompressible 2D fluid solver and particle/field coupling. | Canonical cavity/flow tests and visual mixing scenario pass. |
| FR-009 | Should | Support a built-in 2D Schrödinger wave-packet module. | Norm conservation and known eigenstate tests pass. |
| FR-010 | Must | Compile model declarations into a scheduled runtime graph with read/write dependencies. | Execution plan is inspectable and stable for identical input. |
| FR-011 | Must | Provide deterministic CPU replay and seeded stochastic runs. | Recorded run can reproduce hashes within documented floating-point constraints. |
| FR-012 | Must | Expose Rust and Python APIs plus a command-line runner. | Same model executes through all three surfaces. |
| FR-013 | Must | Provide a basic interactive 2D viewer with pause, step, scrub, inspect, and parameter adjustment. | A user can modify allowed parameters and compare runs without code changes. |
| FR-014 | Must | Record metrics, events, solver residuals, conservation diagnostics, and performance traces. | Run artifact contains machine-readable diagnostics and visual timeline. |
| FR-015 | Should | Allow custom force, field, reaction, observer, and visualization plugins. | Example plugin builds without modifying engine source. |
| FR-016 | Should | Support parameter sweeps and batch headless execution. | CLI can expand a parameter grid and aggregate results. |
| FR-017 | Could | Support external quantum-chemistry jobs through adapters. | At least one adapter submits a geometry/energy calculation and imports results with provenance. |
| FR-018 | Could | Support differentiable execution for a subset of solvers. | Gradient check passes against finite differences for supported models. |
</TABLE>

## 6.2 Nonfunctional requirements

<TABLE>
| ID | Priority | Requirement | Acceptance evidence |
| NFR-001 | Must | No heap allocation in validated hot loops except explicitly profiled scratch arenas. | Allocation instrumentation reports zero steady-state allocations. |
| NFR-002 | Must | The core runtime must be memory safe at public API boundaries. | Rust ownership/validation prevents dangling user-visible references. |
| NFR-003 | Must | Every solver publishes assumptions, supported precision, stability guidance, and validation cases. | Documentation and runtime metadata are present before module release. |
| NFR-004 | Must | Crashes in optional scripts or plugins must not corrupt saved project state. | Sandboxed process/plugin failure test leaves canonical project intact. |
| NFR-005 | Should | CPU and portable GPU backends should share semantics and validation tolerances. | Cross-backend result comparison stays within module-specific thresholds. |
| NFR-006 | Should | Projects remain readable and version-migratable. | Old fixtures migrate forward without silent semantic change. |
| NFR-007 | Must | The engine must expose numerical instability rather than silently clamp or hide it. | NaN, divergence, CFL, residual, and conservation monitors create explicit diagnostics. |
| NFR-008 | Should | Basic simulations should start quickly on commodity laptops. | Reference small scene reaches first rendered frame without external service dependencies. |
| NFR-009 | Could | The viewer and selected compute workloads should run in a browser using WebGPU/WebAssembly. | Browser demo runs a defined subset with feature flags. |
</TABLE>

# 7. System Architecture

[IMAGE]
Figure 3. The architecture separates human-facing model semantics from specialized execution backends.

## 7.1 Major subsystems

<TABLE>
| Subsystem | Responsibility | Initial implementation |
| Project loader | Read project files, imports, assets, parameters, and version metadata | Rust parser + serde; canonical AST. |
| Model compiler | Resolve names/units, construct domains and coupling graph, choose layouts and kernels | Rust compiler pipeline with diagnostics. |
| Simulation IR | Immutable compiled model plus mutable runtime state schemas | Typed IDs, SoA layouts, operation graph. |
| Runtime scheduler | Advance time, run substeps, schedule events, execute coupling, checkpoint | Single-process dependency scheduler. |
| Domain modules | Implement mechanics, fields, fluids, chemistry, MD, and quantum solvers | Independent crates implementing solver contracts. |
| Compute abstraction | Dispatch CPU scalar/SIMD and GPU kernels | CPU first; wgpu/WebGPU portable GPU next; native accelerators later. |
| Visualization | Render particles, geometry, fields, vectors, bonds, contours, plots, and diagnostics | wgpu renderer + egui inspector. |
| Data system | Stream trajectories, metrics, checkpoints, and metadata | Chunked binary arrays + JSON metadata; exporters. |
| SDKs | Rust library, Python package, command line, optional web surface | pyo3 bindings and thin CLI. |
| Validation lab | Analytic tests, reference datasets, benchmark harness, cross-backend comparisons | Dedicated fixtures and CI profiles. |
</TABLE>

## 7.2 Model/data separation

The runtime should follow a strict separation between immutable compiled model state and mutable simulation data. The compiled model contains topology, entity schemas, expression bytecode or kernels, material parameters, solver configuration, indices, buffer plans, and dependency information. Runtime data contains current positions, velocities, field values, reaction extents, wavefunctions, scratch buffers, events, and diagnostics. This mirrors a proven performance pattern in simulation engines: expensive parsing and indexing happen once, while the hot loop operates over preallocated low-level arrays.
<TABLE>
| CompiledModel (immutable)
  domains[]
  materials[]
  topology[]
  operation_graph[]
  buffer_plan[]
  kernel_handles[]
  validation_contracts[]

SimulationState (mutable)
  time
  particle_arrays
  rigid_body_arrays
  field_buffers
  species_concentrations
  wavefunction_buffers
  scratch_arenas
  diagnostics |
</TABLE>

## 7.3 Domain contract

Every solver domain implements the same conceptual contract even though its internal mathematics differs. The contract allows the runtime to compose modules without pretending that their internal state is interchangeable.
<TABLE>
| Contract element | Meaning |
| Schema | Static model declarations and runtime state layout. |
| Compile | Validation, unit checking, topology construction, kernel generation, buffer requests. |
| Prepare step | Update spatial indices, boundaries, coefficients, caches, and external inputs. |
| Stable-step estimate | Return preferred or maximum timestep with reason codes. |
| Advance | Execute one domain step or substep. |
| Ports | Publish typed inputs/outputs for coupling. |
| Observe | Expose metrics, residuals, invariants, and render channels. |
| Checkpoint | Serialize sufficient state for restart/replay. |
| Capabilities | Backend, precision, deterministic, differentiable, and dimensionality support. |
</TABLE>

# 8. Simulation Intermediate Representation

## 8.1 Why an IR is central

The simulation IR is the most important potentially novel component. A scene file should not be executed directly. It should compile into a normalized representation that contains the physical meaning of the model, the operations required to advance it, and the constraints required to execute it correctly. This allows the same project to target a deterministic CPU backend, a portable GPU backend, batch execution, a web viewer, or an external co-simulation environment.

## 8.2 Core IR concepts

<TABLE>
| Concept | Examples | Key metadata |
| Domain | rigid2d, particles2d, grid2d, chemistry, quantum2d | solver family, precision, cadence, fidelity |
| State variable | position, velocity, temperature, concentration, potential | shape, unit, location, mutability, storage class |
| Parameter | mass, viscosity, rate constant, charge | unit, bounds, default, compile-time/runtime |
| Operator | force evaluation, divergence, collision solve, reaction update | reads, writes, backend kernel, stability rule |
| Boundary | periodic, reflective, no-slip, Dirichlet, Neumann | geometry, target variables, time dependence |
| Material | density, elasticity, conductivity, force-field type | properties, temperature dependence, references |
| Port | heat_flux, force_density, species_source, electric_field | unit, topology, interpolation, cadence |
| Observer | energy, region average, trajectory, detector, histogram | sampling rate, reduction, output channel |
| Event | collision, threshold crossing, scheduled parameter change | trigger, priority, deterministic ordering |
| Fidelity declaration | interactive, validated, external | method, tolerances, assumptions, known limitations |
</TABLE>

## 8.3 Expression system

Users need high expressiveness without forcing every custom law through a slow interpreter. The proposed expression system is a restricted, typed, side-effect-free language for scalar/vector algebra, local field sampling, pair interactions, conditions, material lookup, and selected mathematical functions. Expressions compile to CPU bytecode or native code and to GPU shader kernels. Complex algorithms remain plugins; the expression language is for local constitutive laws and force/reaction definitions.
<TABLE>
| force spring(a: particle, b: particle) -> vec2<newton> {
    let dx = minimum_image(b.position - a.position);
    let extension = length(dx) - rest_length;
    return stiffness * extension * normalize(dx)
         - damping * dot(b.velocity - a.velocity, normalize(dx)) * normalize(dx);
}

reaction neutralization {
    stoichiometry: H_plus + OH_minus -> H2O;
    rate: k * c(H_plus) * c(OH_minus);
    heat_release: 57.3 kilojoule / mole;
} |
</TABLE>

## 8.4 Compile pipeline

Parse project and imported modules into a source AST.
Resolve names, types, dimensions, units, species, materials, and geometry references.
Normalize expressions and lower high-level constructs into domain operations.
Construct read/write sets, dependency edges, coupling ports, and event ordering.
Select solver implementations and precision based on fidelity profile and backend capabilities.
Plan buffers, alignment, structure-of-arrays layouts, scratch arenas, and transfer paths.
Compile/cache kernels and create a stable operation graph.
Run static validation: dimensional analysis, topology, boundary completeness, unsupported coupling, and parameter bounds.
Emit the immutable CompiledModel plus a model report describing equations and assumptions.

# 9. Runtime, Time, and Scheduling

## 9.1 Time model

A universal engine must handle domains with dramatically different characteristic timescales. Lattice therefore cannot require every module to advance with one fixed global timestep. The runtime maintains a master simulation clock and schedules domains at integer or rational substeps where possible. Domains report stability limits and preferred cadence. Coupling edges declare whether values are held, interpolated, extrapolated, or iterated between synchronization points.
<TABLE>
| Mode | Use | Behavior |
| Fixed global step | simple interactive scenes | all domains advance at one dt; easiest to reproduce and debug |
| Subcycling | fast particles coupled to slower fields | a domain advances N substeps before synchronization |
| Adaptive local step | stiff ODEs, wave propagation, error-controlled integration | solver chooses dt within runtime constraints and emits acceptance/rejection metrics |
| Event-driven | collisions, threshold reactions, scheduled changes | events ordered by time and deterministic priority |
| Quasi-static solve | electrostatics, equilibrium, steady diffusion | domain solves to tolerance when inputs change rather than every frame |
| External co-simulation | quantum job, third-party solver | runtime exchanges state at declared communication points |
</TABLE>

## 9.2 Operation graph scheduling

The compiled operation graph is a directed acyclic graph within a simulation phase. Nodes declare buffer reads/writes, backend, estimated cost, and synchronization requirements. The scheduler can run independent CPU operations in parallel, fuse compatible local kernels, overlap GPU compute with CPU preparation, and skip operations whose inputs have not changed. Cyclic multiphysics dependencies are represented as explicit iterative coupling groups with convergence tests, not hidden scheduler cycles.

## 9.3 Events and discontinuities

Events use stable ordering: time, domain priority, event class, source ID, sequence number.
Discontinuous parameter changes create checkpoints and invalidate dependent caches.
Collision/contact events may remain solver-internal but publish summary events for observers.
Reaction threshold or phase-change events must define conservation transfers explicitly.
A run can choose strict deterministic event ordering or higher-throughput relaxed GPU ordering.

# 10. Numerical Foundations

## 10.1 Numeric types and precision

<TABLE>
| Mode | Representation | Use |
| fast32 | 32-bit floating point | interactive GPU scenes, visualization-first workloads |
| mixed | 32-bit state with selected 64-bit reductions/accumulation | default GPU scientific mode where supported |
| accurate64 | 64-bit floating point | validated CPU runs, stiff systems, reference comparisons |
| deterministic32/64 | restricted operations and ordering | replay, regression, and cross-run comparison |
| complex32/64 | complex fields | quantum wavefunctions and frequency-domain modules |
</TABLE>

## 10.2 Integrator registry

Integrators are selected per domain rather than globally. The initial registry should include explicit Euler only for teaching/comparison, semi-implicit Euler for simple mechanics, velocity Verlet/leapfrog for conservative particle systems, Runge–Kutta methods for reaction networks, implicit or Crank–Nicolson variants for diffusion, and split-step Fourier or Crank–Nicolson methods for the Schrödinger module. Each implementation publishes order, stability notes, conservation behavior, backend support, and dense-output capability.

## 10.3 Linear and nonlinear solvers

Conjugate gradient for symmetric positive-definite systems.
BiCGSTAB or GMRES for general sparse systems, added only with benchmark justification.
Jacobi/Gauss–Seidel and multigrid-oriented smoothers for grid workloads.
Newton and damped Newton methods for nonlinear local/coupled solves.
Preconditioner interface separated from solver selection.
Residual histories are always observable and can stop a run on divergence.

## 10.4 Conservation and error budgets

Each run maintains an error budget with module-specific monitors: total energy, linear/angular momentum, species mass, charge, probability norm, divergence, flux balance, constraint error, linear-solver residual, and timestep rejection. Coupling operations must account for transferred quantities so that the diagnostic system can distinguish physical dissipation from numerical loss and intentional external forcing.

# 11. Physics Domain Modules

## 11.1 Rigid-body mechanics

<TABLE>
| Capability | MVP | Later |
| Shapes | circle, box, convex polygon, segment | compound bodies, signed-distance geometry |
| Dynamics | translation, rotation, forces, torques, impulses | continuous collision detection and articulated systems |
| Contacts | broadphase grid/BVH, narrowphase, friction, restitution | optimization-based or complementarity contact variants |
| Constraints | distance, pin, spring, motor | general joints and constraint graph optimization |
| Sensors | position, velocity, force, collision events | virtual IMU and control interfaces |
</TABLE>

## 11.2 Particle systems

Particles are a general computational primitive used for gases, granular matter, molecular dynamics, tracers, emitters, and hybrid methods. Runtime storage should be structure-of-arrays with stable logical IDs and compact active indices. The default local-interaction accelerator is a cell-linked list / uniform spatial hash in 2D, with a Verlet-style neighbor cache for molecular workloads. Sorting by cell improves locality on CPU and GPU.

## 11.3 Field and PDE domain

<TABLE>
| Field type | Initial equations/use | Discretization direction |
| Scalar | temperature, concentration, pressure, potential | cell-centered finite difference/finite volume |
| Vector | velocity, force density, electric field | staggered or collocated layouts by solver |
| Tensor | stress/strain later | explicitly deferred until material module |
| Mask/region | obstacles, materials, boundary tags | compact integer/bitfield grid |
</TABLE>

## 11.4 Heat and diffusion

Explicit scheme for interactive low-stiffness cases with CFL diagnostics.
Implicit or Crank–Nicolson scheme for larger timesteps and validated runs.
Spatially varying conductivity, diffusivity, heat capacity, and sources.
Dirichlet, Neumann, Robin, periodic, and geometry-derived boundaries.
Coupling ports for reaction heat, frictional heating, radiation approximation, and temperature-dependent material properties.

## 11.5 Fluids

The first fluid module should be deliberately narrow: incompressible 2D flow for interactive and educational use, with advection, viscosity, pressure projection, obstacles, inflow/outflow, scalar transport, and particle tracers. The implementation may begin with a stable semi-Lagrangian method for usability, followed by a more conservative finite-volume profile. The UI must label which profile is active because visual stability and conservation accuracy are different objectives.

## 11.6 Waves and optics

Scalar wave equation on a 2D grid with reflective, absorbing, and periodic boundaries.
Sources, interference, diffraction barriers, and detectors.
Optional geometric-ray module as a separate domain, not conflated with wave optics.
Frequency and phase visualizations; energy-flux diagnostics where supported.

## 11.7 Electromagnetism

Initial electromagnetism should begin with electrostatics: charge density, potential solve, electric field derivation, and charged-particle coupling. A later finite-difference time-domain module can model 2D electromagnetic waves under explicitly defined polarization assumptions. Magnetostatic and full Maxwell coupling should not enter the MVP unless supported by rigorous validation cases.

## 11.8 Soft matter and deformables

Deformable bodies are a later module. Candidate methods include mass-spring systems for interactive use, position-based dynamics for stable visual applications, and material point or finite-element approaches for more quantitative work. These methods should not share one misleading “soft body” fidelity label; each must declare its constitutive meaning.

# 12. Chemistry and Molecular Simulation

## 12.1 Chemistry model layers

<TABLE>
| Layer | What it models | What it does not claim |
| Reaction network | species counts/concentrations and kinetic laws | molecular geometry or electronic mechanism |
| Reaction-diffusion | species transport plus local reactions | atomistic solvent or detailed turbulence |
| Classical molecular dynamics | particles/atoms under parameterized force fields | electronic bond rearrangement unless reactive potential added |
| Reactive empirical model | bond-order or learned potential approximations | universal ab initio accuracy |
| External quantum chemistry | electronic energies, forces, properties, paths | real-time large-scale interactive dynamics by default |
</TABLE>

## 12.2 Chemical species and reaction networks

Species are typed entities with names, charge, molar mass, optional elemental composition, phase, diffusion coefficient, heat capacity, and visualization metadata. Reactions declare stoichiometry, reversibility, rate law, equilibrium or kinetic parameters, temperature dependence, catalysts, and enthalpy transfer. The compiler verifies dimensional consistency of rate expressions and, when elemental composition is available, atom and charge balance.
<TABLE>
| species H_plus {
  formula: H;
  charge: +1 elementary_charge;
  diffusion: 9.31e-9 meter^2 / second;
}

reaction acid_base {
  reactants: 1 H_plus + 1 OH_minus;
  products: 1 H2O;
  rate: k_forward * H_plus.concentration * OH_minus.concentration;
  enthalpy: -57.3 kilojoule / mole;
} |
</TABLE>

## 12.3 Deterministic and stochastic kinetics

Deterministic ODE integration for well-mixed, sufficiently populated systems.
Gillespie stochastic simulation algorithm for discrete low-copy systems.
Tau-leaping as a later accelerated stochastic mode.
Spatial reaction-diffusion through operator splitting or coupled solvers.
Stiff solver integration through a plugin or established numerical library rather than a weak custom method.

## 12.4 Classical molecular dynamics

The built-in 2D molecular-dynamics module is a platform for learning, algorithm development, and coupled experiments—not an immediate competitor to LAMMPS or OpenMM. It should support pair potentials such as Lennard–Jones, Coulomb with declared cutoff/approximation, harmonic bonds/angles, periodic boundaries, thermostats, barostat research later, and trajectory analysis. Users may define custom local potential expressions that compile to CPU/GPU kernels.
<TABLE>
| Component | Initial choice | Reason |
| Integrator | velocity Verlet | simple, symplectic, common baseline |
| Neighbor search | cell list + optional skin distance | linear-ish local interaction scaling in uniform systems |
| Boundary | periodic/reflective/open | covers canonical tests |
| Thermostat | Langevin and simple velocity-rescaling profile | stochastic and educational control |
| Potentials | LJ, harmonic bond, soft repulsion, Coulomb cutoff | small validated foundation |
| Analysis | energy, RDF, MSD, temperature, pressure estimate | basic scientific observability |
</TABLE>

## 12.5 Chemistry-to-continuum coupling

Reaction rates consume/produce species concentration fields.
Reaction enthalpy contributes to the heat equation.
Temperature changes rate constants through declared models such as Arrhenius relationships.
Fluid velocity advects species and temperature fields.
Particle populations can deposit density, charge, or species sources onto grids and sample fields back to particles.
Mass, charge, and energy transfer are accounted for in a coupling ledger.

# 13. Quantum Mechanics and Quantum-Chemistry Integration

## 13.1 Built-in quantum2d module

The built-in quantum module should solve controlled low-dimensional problems that are visually compelling and numerically verifiable. Its central state is a complex wavefunction on a grid with a user-defined potential. Initial capabilities include imaginary-time eigenstate search, real-time wave-packet propagation, barriers and wells, interference, tunneling, expectation values, probability current, and measurement-inspired sampling. It should not present 2D single-particle quantum mechanics as a general molecular electronic-structure model.
<TABLE>
| Capability | MVP method candidate | Validation |
| Time evolution | split-step Fourier where boundaries permit; Crank–Nicolson alternative | norm, free packet, harmonic oscillator, barrier transmission |
| Eigenstates | imaginary-time propagation / sparse eigensolver plugin | box and harmonic oscillator spectra |
| Potential | analytic expression, image/field, coupled electrostatic potential | unit and grid consistency tests |
| Observables | norm, position/momentum expectation, energy, detector regions | analytic comparisons |
| Visualization | probability density, phase, real/imaginary parts, current | consistent color/scale and normalization |
</TABLE>

## 13.2 External quantum-chemistry adapters

Mature packages already implement broad electronic-structure methods. PySCF exposes modular molecular and periodic methods through Python; Psi4 provides a Python-friendly quantum-chemistry environment; Quantum ESPRESSO provides plane-wave DFT and related solid-state tools. Lattice should define a provider-neutral job contract, then implement adapters that export geometry and method settings, execute locally or remotely, and import energies, forces, charge distributions, orbitals, or reaction-path data with full provenance.
<TABLE>
| quantum_job optimize_water {
  provider: pyscf;
  method: dft("B3LYP");
  basis: "def2-SVP";
  geometry: molecule("water");
  outputs: [optimized_geometry, total_energy, forces, partial_charges];
  cache_key: content_hash;
} |
</TABLE>

## 13.3 QM/MM and multiscale direction

A future hybrid mode can define a quantum region embedded in a classical environment. The first practical version should orchestrate established libraries rather than invent the coupling mathematics. The Lattice contribution would be region selection, state conversion, workflow management, visualization, caching, parameter sweeps, and explicit provenance—not a new electronic-structure solver unless later research produces one.

# 14. Multiphysics Coupling

[IMAGE]
Figure 4. Coupling is declared through typed ports rather than hidden cross-module access.

## 14.1 Coupling edge schema

<TABLE>
| Field | Purpose | Example |
| source port | published quantity and topology | reaction.heat_release on grid cells |
| target port | consumed quantity | heat.source |
| mapping | spatial transfer/interpolation | particle-to-grid bilinear deposition |
| cadence | exchange frequency | every fluid step |
| temporal policy | hold/interpolate/extrapolate | linear interpolate field samples |
| conservation | quantity bookkeeping | energy transfer must balance |
| iteration | one-way, staggered, fixed-point, Newton-like | temperature ↔ reaction-rate fixed-point |
| tolerance | coupling convergence | relative residual < 1e-5 |
</TABLE>

## 14.2 Coupling strategies

One-way coupling: source affects target, but target does not feed back during the step.
Loose staggered coupling: domains advance sequentially using the latest available state.
Subcycled coupling: a fast domain performs multiple steps between exchanges.
Iterative fixed-point coupling: repeat domain solves until port changes converge.
Monolithic solve: deferred to specialized modules; the runtime should permit a plugin to own the coupled system.

## 14.3 Coupling ledger

Transfers of conserved quantities should create ledger entries. For example, an exothermic reaction removes reactants, creates products, and adds heat. The ledger records expected mass/element/charge/energy changes, external work, boundary fluxes, and numerical discrepancy. This makes it possible to debug whether a coupled result is physically inconsistent because of the model, a mapping error, a timestep issue, or an intentionally open system.

# 15. Performance and Optimization Architecture

## 15.1 Performance principles

Measure end-to-end: Include model compile time, upload/download, solver work, coupling, rendering, and output—not kernel time alone.
Separate fast and deterministic modes: A high-throughput GPU path may use atomics and relaxed ordering; regression mode may use slower stable reductions.
Exploit structure: Use grids for local neighbors, sparsity for active fields, cached topology for bonds, and change detection for quasi-static domains.
Minimize movement: Keep simulation and rendering buffers on the same GPU when possible; stream only selected observations to the CPU.
Amortize compilation: Cache kernels by normalized expression, backend, precision, and hardware capabilities.
Optimize after validation: A faster wrong solver is a regression. Every optimization must preserve tolerance-based validation suites.

## 15.2 Data layout

Structure-of-arrays for particle and body hot state; array-of-structures only in authoring APIs or cold metadata.
Compact active arrays with stable external IDs mapped through indirection tables.
Aligned buffers and backend-specific padding hidden behind typed views.
Double/triple buffering for asynchronous compute, rendering, and observation readback where useful.
Scratch arenas sized during model compilation; per-thread or per-domain partitions to avoid contention.
Sparse/tiled field representation added only after uniform-grid baselines are correct and profiled.

## 15.3 CPU execution

Rust core with explicit parallel iterators/task pools for independent operations.
Portable SIMD for local vectorizable kernels where stable support permits; scalar reference path retained.
Cache-aware cell sorting, blocked grid operations, and reduction trees.
Avoid virtual dispatch and bounds checks inside validated hot kernels where safe abstractions can compile away.

## 15.4 GPU execution

A portable WebGPU/wgpu backend is attractive for one code path across Windows, macOS, Linux, and potentially browsers. WebGPU exposes both rendering and general GPU computation, allowing simulation buffers to feed the viewer directly. Because scientific workloads may require features or precision beyond the portable baseline, the architecture should permit later native backends such as CUDA without forcing the simulation IR to change. The portable backend is the product baseline; native accelerators are performance plugins.

## 15.5 Kernel compilation and fusion

Normalize custom expressions and hash the semantic form for cache reuse.
Fuse consecutive elementwise field operations when read/write dependencies permit.
Specialize constants, material types, boundary modes, and dimensions at compile time.
Avoid fusion across operations that need independent residuals, synchronization, or debugging visibility unless an optimized profile is selected.
Retain a traceable mapping from generated kernels back to source expressions.

## 15.6 Performance targets (engineering goals, not promises)

<TABLE>
| Benchmark scene | Baseline target | Stretch target | Correctness condition |
| 2D rigid bodies | 1,000 simple bodies interactive on laptop CPU | 5,000 simple bodies with GPU/optimized broadphase | stable contacts and bounded penetration |
| Local particles | 250k visual particles at interactive rate on mainstream GPU | 1M simple particles | declared force model and no hidden particle dropping |
| Lennard–Jones MD | 25k particles with neighbor lists at useful analysis rate | 100k particles on GPU | energy/temperature metrics within profile tolerance |
| Heat/diffusion grid | 512×512 grid comfortably interactive | 1024×1024 grid interactive on GPU | manufactured-solution convergence |
| Reaction-diffusion | 512×512, 2–4 species interactive | 1024×1024 with fused kernels | species bounds and convergence checks |
| Quantum wave packet | 512×512 complex grid interactive/near-interactive | 1024×1024 accelerated | probability norm drift below declared limit |
</TABLE>

# 16. Expressiveness, DSL, APIs, and Plugins

## 16.1 Authoring surfaces

<TABLE>
| Surface | Audience | Role |
| Project DSL | users, educators, reusable examples | declarative canonical model and version control |
| Python API | scientists, AI/optimization, notebooks | programmatic construction, sweeps, analysis, custom callbacks |
| Rust API | engine integrators and high-performance extensions | embedded execution, custom domains, native applications |
| Visual editor | learners and interactive builders | scene construction, parameters, inspection, plots |
| CLI | automation and CI | headless run, validate, benchmark, convert, inspect |
</TABLE>

## 16.2 DSL principles

Readable declarative syntax with explicit units and defaults.
Composable modules and parameterized templates without unrestricted metaprogramming in v1.
Source-positioned diagnostics and canonical formatting.
Clear separation between compile-time topology and runtime-adjustable parameters.
No hidden global state; all external inputs are declared.
Model migrations must be semantic and versioned.

## 16.3 Plugin tiers

<TABLE>
| Tier | Extension | Safety/performance |
| Expression | local force, source, rate, material law | validated, side-effect-free, compilable to supported backends |
| Observer | metrics, reductions, exports, custom visuals | read-only state access; asynchronous output permitted |
| Operator | custom update kernel within an existing domain | declares reads/writes, units, backend, stability |
| Domain | entire solver family and state schema | native Rust ABI/API; strongest review/validation burden |
| External provider | third-party solver or service adapter | process boundary, explicit serialization and provenance |
</TABLE>

## 16.4 Differentiability and optimization

Differentiable simulation should be a capability flag, not a universal promise. Supported operators declare forward and reverse rules or participate in a backend that can differentiate them. The runtime records discontinuities, contacts, stochastic operations, and adaptive decisions that may invalidate simple gradients. Where gradients are unreliable, Lattice should provide finite differences, ensemble methods, derivative-free optimization, and explicit warnings.

# 17. Visualization and Scientific Debugging

## 17.1 Viewer capabilities

<TABLE>
| Layer | Examples |
| Geometry | bodies, boundaries, meshes, obstacles, sensors |
| Particles | position, velocity trails, species, charge, bonds, neighbor cells |
| Scalar fields | heatmaps, contours, isolines, logarithmic/linear scales |
| Vector fields | arrows, streamlines, glyph sampling, divergence/curl overlays |
| Quantum | probability density, phase wheel, real/imaginary components, probability current |
| Diagnostics | forces, contacts, constraints, residuals, CFL, energy/charge/mass drift |
| Plots | time series, histograms, phase space, spectra, region probes |
| Comparison | side-by-side runs, difference field, synchronized scrub, parameter branch |
</TABLE>

## 17.2 Interaction model

Pause, single-step, run-to-event, reset, checkpoint, and replay.
Inspect an entity/cell and see all state values with units and source equations.
Apply a temporary force, heat source, species injection, or potential edit and observe consequences.
Adjust runtime parameters within declared safe bounds; topology-changing edits trigger recompilation.
Pin probes and compare observed values against analytic/reference curves.
Toggle fidelity and solver profile while retaining a visible explanation of what changed.

## 17.3 Numerical debugger

<TABLE>
| A defining feature / The debugger should answer “why did this simulation become unstable?” It should show the first NaN or invariant violation, the operation that produced it, relevant inputs, timestep/stability estimates, coupling residuals, and the source model declaration. Scientific debugging is a product surface, not only a developer log. |
</TABLE>

# 18. Data, Reproducibility, and Interchange

## 18.1 Run artifact

Every execution produces a self-describing run artifact. The artifact contains project hash, compiled-model hash, engine version, backend and device, precision, seeds, solver profiles, timestep history, parameter values, selected state snapshots, observations, warnings, validation results, and external-job provenance. Large trajectories are chunked and independently streamable; metadata remains readable without loading all arrays.
<TABLE>
| Data class | Candidate representation | Notes |
| Project source | text DSL + JSON-compatible metadata | human-readable and version controlled |
| Compiled model cache | engine-private binary | invalidated by semantic version, backend, and capability hash |
| Dense trajectories/fields | chunked array format such as Zarr/HDF5-compatible export | compression and partial reads |
| Tabular metrics | Arrow/Parquet or CSV export | easy analysis |
| Molecular structures | XYZ/PDB and common adapters | retain units and atom IDs |
| External simulation | FMI/FMUs or provider-specific adapters | co-simulation and model exchange |
| Images/video | PNG and standard video export | never the sole scientific output |
</TABLE>

## 18.2 Checkpoints and replay

Full checkpoints contain all dynamic state and random-generator state required for restart.
Delta checkpoints are optional after correctness of full checkpoints is established.
The event log records user interventions, parameter edits, external inputs, and adaptive-step decisions.
Replay mode verifies operation hashes and reports divergence rather than silently continuing.

## 18.3 Interchange strategy

Lattice should import/export common domain formats but retain its own canonical model because external formats rarely encode all coupling, fidelity, and diagnostics semantics. FMI is a useful long-term bridge for model exchange, co-simulation, and scheduled execution. Molecular and quantum providers require format-specific adapters. Every import should produce a conversion report listing unsupported or approximated concepts.

# 19. Validation, Testing, and Benchmarking

## 19.1 Validation hierarchy

<TABLE>
| Level | Question | Examples |
| Unit | Is the local operator implemented correctly? | force expression, stencil, boundary update, unit conversion |
| Property | Does a broad invariant hold? | symmetry, positivity, conservation, monotonicity |
| Analytic | Does the solver match a known solution? | projectile, oscillator, diffusion kernel, box eigenstates |
| Manufactured solution | Does discretization converge at expected order? | forced PDE with chosen exact solution |
| Reference implementation | Does it agree with an independent trusted code? | small MD/kinetics/quantum cases |
| Cross-backend | Do CPU and GPU agree within tolerance? | all released kernels |
| Scenario | Does a coupled model behave and account for transfers? | exothermic reaction + heat + flow |
| Performance regression | Did throughput/memory change? | fixed hardware CI or recorded benchmark lab |
</TABLE>

## 19.2 Required canonical tests

Free fall, constant acceleration, harmonic oscillator, pendulum, elastic/inelastic collision, and constrained motion.
Heat equation on periodic and fixed boundaries; diffusion Gaussian; convergence under grid refinement.
Poisson/electrostatic potential for known charge and boundary configurations.
Lid-driven cavity or similarly recognized fluid baseline once fluid module is quantitative.
Gray–Scott or other reaction-diffusion patterns plus mass-balance checks.
First-order and reversible reaction networks with analytic solutions.
Gillespie statistics against known distributions.
Lennard–Jones energy conservation, radial distribution trends, and neighbor-list consistency.
Quantum particle in a box, harmonic oscillator, free packet, tunneling, and double-slit norm checks.
Coupled reaction heat transfer with ledger balance.

## 19.3 Benchmark rules

Publish exact hardware, backend, precision, model file, engine revision, and validation tolerance.
Report simulated steps per second and useful physical time per wall-clock second, not FPS alone.
Separate compute-only, rendering, output, and end-to-end measurements.
Include memory footprint and compilation/startup cost.
Never compare against another engine using materially different physical assumptions without saying so.

# 20. Solo-Developer MVP

## 20.1 MVP objective

<TABLE>
| MVP objective / Demonstrate that one typed 2D project can combine particles, fields, reaction kinetics, heat, and a lightweight visual debugger while running efficiently on commodity hardware. The MVP proves the architecture and authoring model; it does not prove universal scientific coverage. |
</TABLE>

## 20.2 Included modules

<TABLE>
| Area | MVP scope | Deferred |
| Core | project parser, units, compiler, operation graph, checkpoints, deterministic CPU | distributed execution and live collaboration |
| Mechanics | particles and simple rigid circles/boxes | articulated systems and complex polygons |
| Fields | uniform scalar/vector grids and sampling | adaptive mesh refinement |
| PDE | heat/diffusion and reaction-diffusion | general symbolic PDE compiler |
| Chemistry | reaction networks and spatial species fields | automatic reaction prediction |
| Molecular | 2D LJ and harmonic bonds | broad production force-field ecosystem |
| Quantum | 2D wave-packet propagation | many-electron solver |
| GPU | portable compute for selected particle/grid kernels | native CUDA specialization |
| Viewer | 2D layers, probes, plots, diagnostics, parameter controls | full node editor and polished authoring suite |
| SDK | Rust, Python, CLI | browser authoring and remote cloud service |
</TABLE>

## 20.3 MVP demonstration

The flagship demo is a 2D chamber containing two diffusing species. A barrier is removed; species mix and react; reaction heat raises local temperature; temperature changes the rate constant; an optional velocity field transports species; particles sample the field and change color with local temperature. The user can pause, inspect cells and particles, alter a rate constant, switch solver profile, display conservation ledgers, and replay the run. A second tab displays a quantum wave packet through a double slit to demonstrate that the same project/runtime can host a radically different solver domain without conflating their equations.

## 20.4 MVP acceptance criteria

One command validates and runs a project; one application opens it interactively.
Typed units reject at least ten deliberately invalid fixture models with source-positioned errors.
All released solvers include equations, assumptions, reference cases, and machine-readable diagnostics.
The flagship coupled demo runs in CPU mode and on at least one portable GPU backend.
No steady-state allocation occurs in core particle/grid hot loops.
CPU deterministic replay reproduces recorded reference runs within the documented mode.
At least one custom force and one custom reaction are defined without modifying engine source.
Benchmark harness reports compute, render, I/O, and memory separately.
Viewer exposes timestep, residual, conservation, and performance panels.
Python API performs parameter sweeps and returns structured results.

# 21. Milestone Roadmap

<TABLE>
| Milestone | Core result | Exit condition |
| M0 — Numerical kernel spike | Rust arrays, units prototype, CPU particle/grid kernels, tiny viewer | analytic heat + particle demos and benchmark harness |
| M1 — Compiled model | DSL/AST, unit checking, immutable model, mutable state, operation graph | same project executes headless and interactively |
| M2 — Mechanics and fields | particles, simple rigid bodies, fields, diffusion, observers | canonical validation suite passes |
| M3 — Chemistry | reaction networks, reaction-diffusion, heat coupling, ledger | flagship exothermic reaction demo passes conservation checks |
| M4 — Portable GPU | wgpu compute backend, kernel cache, zero-copy rendering paths | selected CPU/GPU cross-validation and performance goals |
| M5 — Molecular and quantum | LJ MD, bonds, quantum2d | energy/norm tests and visual examples |
| M6 — Extensibility | expression compiler, Python API, plugin SDK, batch sweeps | external user can author custom law and observer |
| M7 — Scientific productization | project packages, report export, reproducibility artifacts, docs | public alpha with validated example library |
| M8 — External solvers | first quantum/FMI adapter and job provenance | round-trip external result integrated into a Lattice scene |
</TABLE>

## 21.1 Scope-control rule

A milestone is not complete because a demo looks compelling. It is complete only when model semantics, validation, diagnostics, data output, and documentation exist. New domains should not be added while the current domain lacks a reference test or cannot explain its stability limits.

# 22. Product and Business Paths

The project can begin as an open technical engine and later choose a commercial wedge. The engine itself may be valuable, but a solo founder generally reaches users faster through a specific experience built on top of the engine. The architecture should preserve both possibilities.
<TABLE>
| Path | Customer | Paid product | Why Lattice matters |
| Education studio | college/high-school STEM | interactive lessons, assignments, instructor authoring, LMS integration | cross-domain simulations with visible reasoning and error |
| Virtual lab/instrument training | universities and technical programs | specific chemistry/physics laboratory modules and assessment | same runtime models procedure, physical state, instruments, and mistakes |
| Scientific prototyping SDK | research groups and startups | commercial support, private plugins, deployment, collaboration | rapid coupled models without assembling multiple stacks |
| Simulation embedding | software/product companies | runtime license and custom domain modules | portable high-performance engine with controllable visuals |
| Optimization and inverse design | engineering/research teams | parameter fitting, sweeps, differentiable/ensemble workflows | one model can run interactively and headlessly |
| Cloud simulation workspace | teams and classrooms | hosted execution, sharing, dataset storage, reproducible reports | removes installation and hardware friction |
</TABLE>

## 22.1 Suggested commercial sequencing

1. Build the core and examples in public enough to attract technical users and validate the architecture.
2. Choose one paid vertical only after observing repeated demand—likely interactive chemistry/physics education or a specialized scientific prototyping tool.
3. Keep the canonical model and local engine usable without cloud dependency; sell polished authoring, collaboration, specialized modules, deployment, support, and hosted compute.
4. Avoid promising laboratory-grade prediction until a narrow domain has external validation and expert partners.

# 23. Risks and Failure Modes

<TABLE>
| Risk | Why it is dangerous | Mitigation |
| Scope explosion | “all physics and chemistry” can prevent any module from becoming trustworthy | 2D-first, milestone gates, fidelity ladder, narrow MVP demo |
| False universality | one abstraction may erase domain-specific needs | domain contracts and specialized state layouts |
| Numerical instability | customization can create invalid models | unit checks, stability estimates, residual/invariant monitors, safe profiles |
| Performance portability | portable GPU abstractions may leave performance on the table | baseline wgpu plus native backend plugin boundary |
| Quantum overclaim | users may assume built-in wave simulation predicts chemistry | strict naming, documentation, and external backend separation |
| Validation burden | each solver multiplies testing work | release fewer modules with canonical validation packages |
| Plugin ABI churn | early native plugins can become impossible to maintain | versioned high-level plugin API; unstable internal ABI until mature |
| Solo maintenance | UI, compilers, numerics, GPU, and docs are each large projects | keep viewer basic, reuse libraries, prioritize engine differentiator |
| Crowded ecosystem | many excellent domain tools already exist | focus on coupling, authoring, diagnostics, and accessible 2D workflows |
| No clear buyer | a general engine may gain users but no revenue | develop vertical applications and interview users before commercial build-out |
</TABLE>

## 23.1 Kill criteria

The typed model cannot express the flagship coupled scenario without domain-specific hacks.
The compiled runtime is not materially faster or easier to inspect than a straightforward Python prototype.
Users consistently prefer existing single-domain tools and do not value coupling or diagnostics.
Portable GPU execution requires so many semantic compromises that CPU/GPU models behave inconsistently.
Validation work becomes impossible to sustain at the planned module count; in that case narrow the engine to the strongest domain combination.

# 24. Recommended Repository Structure

<TABLE>
| lattice/
  Cargo.toml
  crates/
    lattice-units/          # dimensions, quantities, runtime unit registry
    lattice-syntax/         # parser, AST, formatter, source diagnostics
    lattice-ir/             # normalized simulation IR and schemas
    lattice-compiler/       # validation, lowering, buffer and operation planning
    lattice-runtime/        # clock, scheduler, events, checkpoints, replay
    lattice-compute/        # backend traits, buffers, kernel cache
    lattice-cpu/            # scalar/SIMD CPU execution
    lattice-wgpu/           # portable GPU kernels and renderer interop
    lattice-domain-particle/
    lattice-domain-rigid2d/
    lattice-domain-grid2d/
    lattice-domain-chemistry/
    lattice-domain-md/
    lattice-domain-quantum2d/
    lattice-coupling/       # ports, mappings, convergence, conservation ledger
    lattice-observe/        # metrics, probes, plots, exporters
    lattice-viewer/         # desktop UI and 2D renderer
    lattice-cli/
    lattice-python/         # pyo3 bindings
    lattice-validation/     # reference cases and comparison tools
  examples/
  benchmarks/
  docs/
  schemas/
  python/
  web/                      # optional later WASM/WebGPU surface |
</TABLE>

## 24.1 Dependency policy

Use mature libraries for windowing/UI, parsing support, linear algebra primitives, FFTs, serialization, and Python bindings where they do not define the core semantics.
Keep solver-domain crates independently testable and avoid circular dependencies through the IR/runtime contracts.
The scalar CPU reference implementation is the executable specification for accelerated kernels.
Generated code and shader sources must be cacheable and inspectable in debug builds.

# 25. Example Models

## 25.1 Coupled reaction, diffusion, and heat

<TABLE>
| project hot_reaction {
  dimensions: 2;
  precision: mixed;
  fidelity: engineering_2d;

  grid chamber { size: [512, 256]; extent: [2 meter, 1 meter]; }

  field temperature on chamber = 298 kelvin;
  species A on chamber = left_half(1 mole / meter^2);
  species B on chamber = right_half(1 mole / meter^2);

  reaction A_plus_B {
    reactants: A + B;
    products: C;
    rate: arrhenius(A0=2.0e5 / second, Ea=35 kilojoule/mole)
          * A.concentration * B.concentration;
    enthalpy: -25 kilojoule / mole;
  }

  solve diffusion(A, B, C) with crank_nicolson(dt=0.002 second);
  solve heat(temperature) with crank_nicolson(dt=0.01 second);
  couple A_plus_B.heat_release -> temperature.source conserve energy;

  observe total_species every 0.1 second;
  observe total_energy every 0.1 second;
  visualize temperature as heatmap;
  visualize [A, B, C] as rgb_mix;
} |
</TABLE>

## 25.2 Quantum double slit

<TABLE>
| project double_slit {
  domain quantum2d q {
    grid: [768, 384];
    extent: [12 nanometer, 6 nanometer];
    mass: electron_mass;
    boundary: absorbing(width=0.8 nanometer);
    integrator: split_step_fourier(dt=0.002 femtosecond);
  }

  potential barrier {
    shape: vertical_wall(x=0, thickness=0.15 nanometer);
    slits: [(-1.0, 0.35), (1.0, 0.35)] nanometer;
    height: 20 electronvolt;
  }

  wavepacket initial {
    center: [-4 nanometer, 0];
    momentum: [6.5e-24 kilogram*meter/second, 0];
    sigma: 0.45 nanometer;
  }

  detector screen at x=4.5 nanometer;
  observe probability_norm every step;
  visualize probability_density;
  visualize phase;
} |
</TABLE>

## 25.3 Python parameter sweep

<TABLE>
| from lattice import Project, Sweep

model = Project.load("examples/hot_reaction.lattice")
results = Sweep(model).grid({
    "reaction.A_plus_B.Ea": [25, 30, 35, 40],   # kJ/mol
    "initial.temperature": [290, 300, 310],     # K
}).run(backend="gpu", until="15 s")

results.plot("peak_temperature", x="Ea", group="initial.temperature")
results.export("runs/reaction_sweep.parquet") |
</TABLE>

# 26. Open Research Questions

Can the simulation IR be expressive enough for different domains while remaining compilable and optimizable?
What is the smallest set of coupling semantics that covers most useful multiphysics workflows without becoming a full Modelica-like language?
Can solver validity and stability constraints be represented in machine-readable form and surfaced interactively?
How should the runtime schedule domains with adaptive and stochastic timesteps while retaining replayability?
Which custom-expression subset provides useful scientific flexibility while compiling efficiently to CPU and WebGPU?
Can conservation ledgers diagnose coupling errors automatically rather than merely report drift?
What data layout and kernel strategy gives acceptable performance across integrated and discrete GPUs?
How can the engine compare fidelity profiles and communicate the effect of approximations to non-experts?
Where can automatic differentiation be trusted across contacts, adaptive timesteps, reactions, and stochastic events?
Can an external quantum calculation be cached and interpolated safely inside an interactive classical simulation?
Would users value one cross-domain environment more than best-in-class single-domain tools, and for which workflows?
Which narrow commercial application produces the strongest feedback loop for the underlying engine?

# 27. Glossary

<TABLE>
| Term | Definition |
| Backend | A concrete CPU/GPU/external implementation of compiled operations. |
| Coupling graph | Explicit connections that transfer typed quantities between domains. |
| Domain | A solver family with its own state, equations, timestep behavior, and validation contract. |
| Fidelity profile | A named combination of methods and tolerances that communicates intended accuracy/use. |
| Hot loop | Performance-critical repeated execution after model compilation. |
| IR | Intermediate representation produced by compiling user models. |
| Manufactured solution | A chosen exact solution used to construct a forcing term and test PDE convergence. |
| Observer | A read-only measurement, reduction, plot, detector, or export operation. |
| Operator splitting | Advancing parts of a coupled equation separately in a controlled sequence. |
| Port | A typed and unit-bearing quantity published or consumed by a domain. |
| Solver contract | Metadata and functions required for a domain to participate in the runtime. |
| Structure of arrays | Data layout storing each property contiguously rather than storing whole objects together. |
</TABLE>

# 28. References and Design Sources

[1] MuJoCo Documentation — Overview and Programming. https://mujoco.readthedocs.io/en/stable/overview.html
[2] LAMMPS Documentation — Overview, Features, and Extension Model. https://docs.lammps.org/Intro_overview.html
[3] OpenMM — High-performance, customizable molecular simulation and User Guide. https://openmm.org/
[4] Taichi — High-performance parallel programming and numerical simulation. https://www.taichi-lang.org/
[5] NVIDIA Warp Documentation — GPU-accelerated and differentiable simulation kernels. https://nvidia.github.io/warp/stable/index.html
[6] W3C WebGPU Specification — portable GPU rendering and computation API. https://gpuweb.github.io/gpuweb/
[7] PySCF User Guide — modular electronic-structure methods and integrations. https://pyscf.org/user/
[8] Quantum ESPRESSO Documentation — plane-wave DFT, molecular dynamics, reaction paths, and related packages. https://www.quantum-espresso.org/documentation/
[9] Psi4 Documentation — Python-accessible quantum chemistry methods. https://psicode.org/psi4manual/master/index.html
[10] Functional Mock-up Interface 3.0.2 Specification — model exchange, co-simulation, and scheduled execution. https://fmi-standard.org/docs/3.0.2/
[11] ModelingToolkit Documentation — symbolic-numeric, equation-based, composable modeling. https://docs.sciml.ai/ModelingToolkit/stable/
These sources are architecture precedents and integration targets, not evidence that the proposed product is already implemented or that its performance targets have been achieved. Numerical methods must additionally cite domain-specific textbooks and papers during implementation.
END OF SPECIFICATION