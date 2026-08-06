# Roadmap

The milestones are the spec's (§21). This records what M0 actually delivered and what
M1 needs, so the gap between plan and reality stays visible.

## M0 — Numerical kernel spike ✅

**Spec exit condition:** *"analytic heat + particle demos and benchmark harness."*

Delivered:

- `lattice-units` — dimensional analysis, SI registry, unit-expression parser with
  ambiguity warnings, CODATA constants with cross-checks.
- `lattice-ir` — typed IDs, structure-of-arrays particle storage with generation-checked
  handles, halo'd grid fields, boundary conditions, bump arenas, reproducible PCG32,
  the `Domain` contract, conservation ledger and residual histories.
- `lattice-domain-particle` — three integrators, four force laws, uniform cell list,
  periodic/reflective/open boundaries.
- `lattice-domain-grid2d` — finite-volume diffusion with variable coefficients, three
  time schemes, matrix-free conjugate gradient, four boundary types.
- `lattice-observe` — order-stable JSON, phase profiles, memory reports, throughput,
  run artifacts with physics-only content hashes.
- `lattice-validation` — 24 cases across five validation levels, each reporting a
  measured value.
- `lattice-cli` — `validate`, `bench`, `demo`, `inspect`, plus a terminal field viewer.

365 tests. Four demos. Four benchmarks, each paired with its §15.6 correctness
condition.

### What M0 taught us

- **Convergence tests are worth more than tolerance tests.** Several early assertions
  compared two schemes at one timestep and needed arbitrary tolerances. Rewriting them
  to measure the *order* at which the gap closes made them both sharper and
  self-justifying.
- **The affine/linear boundary split is the easiest thing to get quietly wrong.** An
  implicit solve with the constant term left inside the operator converges happily to
  the wrong boundary values. Only the analytic slab case catches it.
- **Contracts change behaviour, not just documentation.** Splitting the contract per
  integrator forced an explicit answer to "does this scheme conserve energy?" for each
  one, which is a question a single shared document lets you avoid.

## M1 — Compiled model ✅

**Spec exit condition:** *"same project executes headless and interactively."*

Delivered:

- **`lattice-syntax`** — source map with spans, tokenizer, AST, recursive-descent
  parser with error recovery, and diagnostics that render a caret under the offending
  characters. Parses spec §25.1, §25.2 and §12.2 verbatim.
- **`lattice-compiler`** — name and unit resolution, dimensional checking of every
  expression (FR-002), field-dimension inference from initial values, solver
  selection, buffer planning, operation-graph construction, and the model report.
- **`lattice-runtime`** — clock, timestep negotiation across domains, observers on a
  cadence, and run artifacts.
- **`lattice-ir` additions** — `CompiledModel`, `BufferPlan`, the operation graph with
  hazard-derived edges and parallel levels, and render channels.
- **`lattice check` and `lattice run`** — the headless half of M1's exit condition.
- **`lattice-viewer` / `lattice-view`** — the interactive half: a window with field
  heatmaps on a perceptually uniform ramp, particle scatter, play/pause/step/reset and
  speed, live plots grouped one-per-unit, conservation drift with a verdict, the
  timestep against its stability limit, and the solver contract. Documented in
  [viewer.md](viewer.md). The project's first external dependencies
  (`eframe`, `egui_plot`), in a crate the other ten do not build.
- **Five example models and sixteen invalid fixtures**, each fixture declaring the
  diagnostic code it must produce.

636 tests. Clippy clean.

### What M1 taught us

- **Units cannot be a lexical concern.** The first design gave the lexer a "unit mode"
  after a number. It could not answer what `100 / dt` meant, because the lexer does not
  know what has been declared. Making units *ordinary identifiers*, resolved after
  declared names, removed the whole problem and made the grammar smaller.
- **Adjacency has to bind tighter than division.** `10 meter / 2 second` parsed as
  `m·s` until juxtaposition was given its own precedence level. A dimensioned literal
  is one atom, and that is not negotiable notation.
- **Declaration kinds do not belong in the grammar.** `grid` was a keyword until
  spec §25.2's `grid: [768, 384];` — a *setting key* — showed why that fails. Parsing
  `<kind> <name> { … }` generically and letting the compiler decide which kinds it knows
  also means a new solver family needs no grammar change.
- **Diagnostics are a design surface.** The fixture suite asserts the *specific* code
  each invalid model produces, and that every rejection carries a source position and
  either a fix or the rule it enforces. A test that only checks "this failed" passes
  just as happily when the model is rejected for the wrong reason.
- **A green test suite does not mean the window is right.** Every viewer defect found
  so far was found by screenshotting the running program: round-off plotted as a
  dramatic oscillation, a values table saying "field units" beside a scale bar saying
  "K", a legend box parked on the curve it labelled, and a perfectly conserved momentum
  reported as `✖ not conserved` because the denominator was `1e-15`. None of them were
  reachable from a unit test that did not already know to look. They are covered by
  tests now — written after the screenshot, not before.
- **Relative error needs a scale, and sometimes only the domain has it.** A quantity
  conserved *at zero* has no scale of its own. `ParticleDomain` publishes
  `momentum_scale = Σ|mᵢvᵢ|` for exactly this, and the viewer states which denominator
  it used rather than leaving a reader to guess.

## M2 — Mechanics and fields ✅

**Spec exit condition:** *"canonical validation suite passes"* for particles, simple
rigid bodies, fields, diffusion and observers. 33 of 33 cases pass.

Delivered:

- **`lattice-domain-rigid2d`** — circles, boxes, convex polygons and segments; forces,
  torques and impulses; sweep-and-prune broadphase, separating-axis narrowphase,
  Coulomb friction and restitution; distance, rope, pin, spring and motor joints
  (§11.1's MVP row, complete). Contacts are resolved by sequential impulses with warm
  starting, and joints are solved in the same interleaved sweep so the two sets
  negotiate rather than alternate.
- **`lattice-ir` additions** — `RigidBodyStore`, the rotational counterpart of
  `ParticleStore`; a `runtime_id!` macro so `BodyId` and `ParticleId` share one
  generation-checked implementation; `RenderChannel::Bodies` and
  `RenderChannel::Contacts`.
- **Collision validation** — §19.2's elastic and inelastic collision and constrained
  motion, the rows M0 had to leave out. Elastic gives exactly ∓2.000000 m/s; inelastic
  leaves exactly the 2.000000 J momentum conservation predicts; the pendulum period is
  2.006250 s against an analytic 2.006409 s; a block on a 0.4 rad slope slides 24.03 m
  at μ = 0.338 and 0.0000 m at μ = 0.507.
- **`material`, `body` and `joint` declarations** — the §8.2 material concept, which
  rigid bodies are the first real consumer of. No grammar change was needed: the
  generic `<kind> <name> { … }` form already parsed them.
- **Viewer support** — body outlines, static bodies drawn as scenery, contact points
  and normals drawn over the geometry they belong to.

757 tests. Clippy clean.

### What M2 taught us

- **A contract that overclaims is worse than no contract.** The rigid module first
  declared momentum conserved, full stop. It is — under the impulse solve. It is not
  under gravity, which injects momentum every step, or against a static body, which is
  an infinite sink for it. Every realistic scene tripped the alarm, and an alarm that
  fires on every correct run is one a reader learns to ignore. The domain now publishes
  momentum as an invariant *only* in a world with no gravity and no static bodies, and
  says so in the panel when it does not.
- **The same trap caught the particle module.** Kinetic and potential energy were both
  published as invariants; only their sum is one. A gas melting out of a lattice
  converts one into the other *on purpose*. Fixed in both, and the reason is now
  written on `ObservationKind` where the next domain will read it.
- **The viewer must read the claim, not guess it.** It decided "should this be
  conserved?" by matching names — `momentum_x`, `total_energy`. That guess cannot
  distinguish a closed world from an open one. It now reads the `ObservationKind` the
  domain published, which is the domain's own promise.
- **Position correction is not free, and the amount is measurable.** Moving a body
  without changing its velocity changes its angular momentum about any fixed origin by
  `dt·(J × Δv)`. No scheme avoids it. It is measured every step and published as
  `correction_drift`, which is a better answer than the word "small".
- **A 2×2 matters.** The pin joint first solved x and y as independent scalars.
  Convergence is set by the ratio of the two effective masses, and for a pendulum bob
  that ratio is enormous — the period came out eight times too fast. Inverting the 2×2
  fixed it in one iteration.

## M3 — Chemistry ✅

**Spec exit condition:** *"flagship exothermic reaction demo passes conservation
checks."* [`examples/chamber.lattice`](../examples/chamber.lattice) does, and 41 of 41
validation cases pass.

Delivered:

- **`lattice-domain-chemistry`** — species with formulas, charges, molar masses and
  diffusion coefficients; reaction networks with **atom and charge balance checking**;
  mass-action kinetics with reversibility and Arrhenius temperature dependence;
  reaction-diffusion by Strang splitting over the existing, already-validated diffusion
  solver.
- **`lattice-coupling`** — typed ports, coupling edges with mappings and cadence, the
  one-way and loose-staggered strategies of §14.2, and the `ConservationLedger` that had
  existed unused in `lattice-ir` since M0.
- **The language** — `reaction` declarations (whose stoichiometry needed no new syntax:
  `2 H2 + O2` already parses, because juxtaposition binds tighter than `+`), chemistry
  attributes on `species`, `domain chemistry`, `solve reactions(…)`, and `couple`.
- **Chemistry validation** — §19.2's first-order, reversible, reaction-diffusion and
  coupled-ledger rows. Gillespie statistics are *absent* rather than skipped, because
  the stochastic path is not implemented.

833 tests. Clippy clean.

### What M3 taught us

- **A coupling edge's unit conversion should be derived, not written.** A reaction
  publishes `W/m²`; a heat solver consumes `K/s`. Wire them straight together and the
  run compiles, executes, looks entirely plausible, and is wrong by an areal heat
  capacity. So the model declares `heat_capacity` as a *material property of the field*,
  and the compiler does the dimensional algebra — which means a mis-wired edge is a
  message naming both units rather than a number nobody can check.
- **Which side of a mapping the ledger measures is not arbitrary.** Recording the
  post-mapping value gives a number in the target's units, off from the energy by
  exactly the factor the mapping applied. The ledger records what left the *source*.
- **The books should not balance exactly, and that is the point.** A staggered coupling
  always has one exchange in flight. The validation case asserts that the shortfall is
  at most one transfer *and* that it is first order in `dt`, which turns "nearly
  balances" into a statement about why.
- **Stiffness is not the same as "many steps".** A budget can run out because the
  network is stiff or because the interval asked for is enormous relative to the
  timescale, and the two need different fixes. `KineticsReport` reports `exhausted` and
  `stiffness` separately.
- **Energy is a state variable.** The heat released was first sampled as an
  instantaneous power; for a reaction that half-finishes in one step that is wrong by a
  factor of `e`. Carrying it through the same Runge–Kutta stages as the concentrations
  makes it fourth-order accurate and costs nothing, because the heat rate falls out of
  rate evaluations the stages already perform.
- **`as_slice()` includes the halo.** A perfectly second-order splitting measured as
  first order, for a whole afternoon, because the test compared whole buffers and the
  halo holds intermediate boundary state rather than part of the answer.

## M4 — Portable GPU (next)

**Spec exit condition:** *"selected CPU/GPU cross-validation and performance goals."*

Needed:

1. **A `wgpu` compute backend** — the operation graph already computes which operations
   are independent and reports the ideal speedup; nothing consumes that yet.
2. **Kernel cache and zero-copy rendering** — §15.5's normalized expression hashing, and
   the viewer drawing from simulation buffers rather than a CPU texture upload.
3. **CPU/GPU cross-validation** — §19.1 lists it as its own level, and it is the whole
   reason the scalar CPU path is described as the executable specification.
4. **CPU parallelism** — §15.3's parallel iterators, deferred twice now. A coupled model
   finally has independent operations to run across, and `examples/chamber.lattice` is
   the first model whose graph has a width above one.

## Outside the milestones: the playground

`lattice-play` is not in the specification. It exists because the spec describes a
scientific instrument — write a model, compile it, run it, measure it — and someone who
opens the project reasonably expects to be able to *touch* the physics before they learn
a language to describe it. So the playground is a second front door onto the same
solvers, built by clicking rather than by compiling.

It is not a shortcut around any of the standing constraints. It publishes the same
observations, its panel applies the same conservation rules, and it added the same kind
of test the rest of the engine has: fifty tests driving `Playground::pointer` directly,
plus one conformance exercise every mode has to survive so a fourth cannot be added
below the bar of the three that exist.

Three engine bugs came out of building it, which is the argument for having built it:

- `RigidDomain::render_channels` published pose arrays at their new length while the
  per-slot tables were still at the old one, so a body spawned between steps was an
  out-of-bounds index in the drawing code. Interactive spawning is what found it; no
  model file can spawn a body mid-run.
- `HeatDomain::observe` gated its solver rows on whether a step had happened, so a table
  gained two rows after the first step and lost them again on a reset. The set of rows a
  domain publishes now depends only on how it is configured.
- A sequential ramp normalized between the data's own min and max renders an all-zero
  concentration field as a solid mid-tone — an empty vessel that reads as uniformly full.
  `field_to_image_above` floors the ramp where the quantity's zero actually is.

## Later milestones

| Milestone | Result | Blocked on |
|---|---|---|
| M5 — Molecular and quantum | LJ MD proper, bonds, `quantum2d` | M2, done |
| M6 — Extensibility | expression compiler (§8.3), Python API, plugin SDK | M1, done |
| M7 — Productization | packages, report export, reproducibility artifacts | M3, done |
| M8 — External solvers | quantum/FMI adapters with provenance | M7 |

## Standing constraints

From spec §21.1, and they apply to every milestone above:

> A milestone is not complete because a demo looks compelling. It is complete only when
> model semantics, validation, diagnostics, data output, and documentation exist. New
> domains should not be added while the current domain lacks a reference test or cannot
> explain its stability limits.

And the kill criteria of §23.1 are worth rereading before M4, particularly:

> The compiled runtime is not materially faster or easier to inspect than a
> straightforward Python prototype.

M0's numbers (774M cell-updates/s, 427M particle-updates/s, single-threaded) clear the
"materially faster" half. The "easier to inspect" half is what `lattice inspect
contracts` and the measured validation report are for, and it stays an open question
until someone outside the project uses them.
