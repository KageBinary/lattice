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

## M4 — Portable GPU ✅

**Spec exit condition:** *"selected CPU/GPU cross-validation and performance goals."*

Current implementation: resident field rendering, domain-DAG scheduling, GPU gravity
and Lennard–Jones, implicit Dirichlet boundaries, and pipeline caching are implemented.
See [M4 engineering report](m4-engineering-report.md) for the final test and performance
record, supported scope and remaining formatting debt. All 56 validation cases pass,
including eleven GPU cases; both shared-device viewer smoke paths pass. The final
review also fixed explicit timestep selection, small-graph overhead, redundant
gravity dispatch and incorrect backend/precision metadata in exported reports.
M4.1–M4.4 below retain their historical measurements and decisions.

The state at the end of M4.4 was:

1. **A `wgpu` compute backend** ✅ — the portable baseline, running explicit diffusion and
   now implicit Crank–Nicolson device-resident. The operation graph's parallel levels are
   still unconsumed.
2. **Kernel cache** ✅ / **zero-copy rendering** — §15.5's cache is built and keyed on all
   four of the inputs that section names; the viewer still uploads a CPU texture.
3. **CPU/GPU cross-validation** ✅ — eight GPU rows at §19.1's `cross-backend` level, with
   tolerances derived rather than fitted, and `lattice bench --backend gpu` for the
   performance half.
4. **CPU parallelism** ✅ — §15.3's parallel iterators, deferred twice before this.

### M4.1 — Parallel CPU execution ✅

Delivered, and documented in [execution.md](execution.md):

- **`lattice-cpu`** — the crate §24 calls `lattice-cpu`. A persistent worker pool with
  lock-free claiming and a spin-before-park window, and an `Executor` that decides how a
  loop is split. It holds the only `unsafe` in the workspace, behind two invariants that
  `dispatch` enforces rather than documents.
- **Parallel kernels** — the diffusion stencil, the explicit update, the Crank–Nicolson
  right-hand side and operator application, and all three integrators' per-particle
  passes. Measured on 20 threads: **4.8× at 512², 3.4× at 256², 1.9× on 262k particles**,
  and nothing anywhere is slower than it was.
- **A cross-backend validation level** — §19.1's own row, with four cases asserting that
  parallel and scalar agree *exactly*: every cell, every particle, the CG iteration
  count, and FR-011's artifact hash.
- **`--threads <n|auto>` and `--compare`** — the second runs a benchmark both ways and
  reports the measured speedup with its efficiency, refusing to report either if a
  §15.6 correctness condition failed.

936 tests. Clippy clean.

### What M4.1 taught us

- **The promise is worth more than the speed.** *Parallel execution changes the schedule,
  never the numbers* is a stronger claim than §19.1 asks for between backends, and it
  costs real performance — Crank–Nicolson gains only 1.6× because its inner products stay
  sequential. It buys two things worth more: a regression baseline recorded on a
  sixteen-core machine is comparable with a run on a four-core one, and the tolerance the
  GPU will eventually need becomes *attributable to the GPU* rather than a number that
  has always been there.
- **A grain is two numbers, not one.** "How much work is worth splitting at all" and "how
  small may a piece get" are different questions. Using one number for both looks tidy
  and quietly costs: a floor high enough to keep 16k particles sequential, reused as a
  minimum chunk size, pinned a 262k-particle model to four threads and dropped it from
  1.90× to 1.34×.
- **An optimization that is sometimes a pessimization is a regression.** The first
  working version made `particles-gravity` **2.4× slower** on twenty threads at 16k
  particles, because six barriers per step cost more than the arithmetic between them.
  §15.1 says a faster wrong solver is a regression; a slower right one is too. The fix
  was half pool engineering and half admitting where the crossover is and refusing to
  split below it.
- **A cross-backend case can pass for the wrong reason.** Every kernel declines to split a
  problem below its floor — so a case sized under that floor compares the sequential path
  with *itself* and passes without testing anything. Raising a floor is an ordinary
  tuning decision that would silently do this. The case sizes are now pinned to the
  published grains, with a test that asserts the work is actually split.
- **A shared counter that resets is a use-after-free waiting for load.** A worker that has
  just run the last task of one dispatch probes once more for work. If the next dispatch
  has already published, that probe either runs its task against the *previous* dispatch's
  freed closure pointer or silently eats an index the new dispatch is waiting for. A
  monotonic counter with per-job ranges, probed by compare-exchange rather than
  fetch-add, makes the stale probe land harmlessly out of range.
- **Timings lie at ±20%.** A "0.94× regression" that justified a design change turned out
  to be nothing; three repetitions made the curve monotonic and the real crossover
  obvious. Nothing in the table above is a single measurement.

### M4.2 — The portable GPU backend ✅

Delivered, and documented in [backends.md](backends.md):

- **`lattice-compute`** — the crate §24 calls *"backend traits, buffers, kernel cache"*,
  with no dependencies at all, because everything in it is a boundary. `Device`, `Buffer`,
  §10.5's `Precision`, `Capabilities`, §15.5's `KernelCache`, and `Tolerance`.
- **`lattice-wgpu`** — §15.4's portable baseline. `GpuDevice` behind `Device`, and
  `GpuDiffusion`, an explicit finite-volume run that uploads once, steps on the device, and
  reads back once. Eight tests of its own properties.
- **`CpuDevice`** — `lattice-cpu` became an implementation *behind* `lattice-compute` rather
  than staying beside it, and can store `f32` as well as the reference `f64`.
- **Three GPU cross-backend rows**, and a `gpu` feature that keeps `wgpu` out of the CLI.
  `lattice validate` runs 45 cases; `--features gpu` runs 48.

983 tests. Clippy clean.

### What M4.2 taught us

- **The tolerance question was a precision question.** The table below predicted the GPU
  budget would be spent on FMA contraction, transcendental accuracy and reduction order.
  All three are real and the first is present — but **WGSL has no `f64` at all**, so the
  product baseline cannot execute the reference precision even in principle, and state
  rounding dominates every budget by roughly eight orders of magnitude. §23's *"portable
  GPU abstractions may leave performance on the table"* understates it for scientific work:
  what the portable abstraction leaves on the table is precision.
- **`wgpu::Features::SHADER_F64` exists and is a trap.** It enables 64-bit floats in
  *SPIR-V* shaders on Vulkan; the portable path compiles *WGSL*, whose validator gates the
  type behind a capability `wgpu` never grants. This adapter reports the feature. Reading
  the flag alone would have produced a backend advertising `accurate64` that then failed to
  compile — so the claim is a test that compiles the shader and fails if it ever succeeds.
- **`Device` is not `Executor` with more implementations**, and finding that out early
  saved the design. `Executor` answers "how do I split this loop across threads sharing
  memory"; there is no `&mut [f64]` to hand out when the data is on a device, and no loop to
  split when the kernel *is* the loop body. They coexist — `CpuDevice` uses an `Executor`
  internally — rather than one nesting inside the other.
- **A budget nobody can decompose is a budget that was fitted.** A cross-backend tolerance
  is the one number in a validation suite that nothing checks: too loose and it passes
  forever, hiding every defect smaller than itself. Making `Tolerance` a list of named
  mechanisms with their derivations, and reporting *fraction of budget used* rather than the
  raw error, turns it into something a reader can audit. The headline case sits at 0.3%.
- **The attribution needs two numbers, not one.** 4.3 `f32` ulps of disagreement is
  consistent with `f32` storage; it is also 2.3×10⁹ `f64` ulps. Reporting both is what makes
  the mechanism a measurement rather than an assertion — a transposed index would blow the
  first bound, and a backend not running the precision it claims would fail the second.
- **A cross-backend suite needs a case where the tolerance is not needed.** On a field with
  no gradient every flux is a difference of equal numbers, so the two backends agree *to the
  bit despite different precisions*. If that case ever fails while the budgeted ones pass,
  the budget is covering a real defect.

**Four questions M4.1 answered for the CPU and deferred for the GPU.** Two are now
answered; two are still open and are what M4.3 runs into.

| Question | CPU answer | GPU answer |
|---|---|---|
| How exact is cross-backend agreement? | bit-identical | a derived budget, dominated by `f32` state rounding |
| Precision | `accurate64` throughout | `fast32` only — WGSL has no `f64` |
| Reductions | sequential | a fixed tree of stated depth — answered in M4.4, and it turned out not to matter |
| Lennard-Jones pair forces | left sequential | **open** — a gather or a colouring becomes mandatory |

### M4.3 — `lattice bench --backend gpu` ✅

The GPU was reachable only from the validation suite; now it is a backend the benchmark
harness can select, which is what a performance claim needed before it could be made.

- **`--backend cpu|gpu`**, with the reporting layer made backend-agnostic: `Executed`
  carries label, precision and thread count, so §19.3's "publish precision with the number"
  stopped being the constant `accurate64` the moment a `fast32` backend existed.
- **Efficiency is not printed for the GPU.** Speedup per thread needs a divisor a reader
  can interpret, and "SM count × occupancy" is not one.
- **A cross-precision ratio is flagged in the output**, because a speedup between two
  different answers is not a speedup.
- **A derived drift limit.** The §15.6 correctness condition is the same check at a
  different epsilon: `n·ε/2` for `n` steps, from the one rounding per cell per step that
  the update performs. At `f64` the CPU's flat `1e-9` is looser than the formula; it is left
  alone as an established baseline rather than tightened in the same change.

985 tests. Clippy clean.

### What M4.3 taught us

- **The kernel wins by two orders of magnitude and the program loses.** 130× on the stencil
  at 1024², and the whole GPU *process* is 4.1× slower than the CPU's at 256², because
  opening a device costs 0.72–0.82 s once. §15.1's "measure end-to-end" is not a
  formality — it inverts the conclusion at two of the three sizes measured. Both numbers
  are printed; neither is honest alone.
- **Half the speedup is precision.** `f32` against `f64` on a bandwidth-bound kernel moves
  half the bytes before any architectural advantage applies. The harness now says so in the
  output rather than leaving it to a reader.
- **A reproducible wrong number is not more trustworthy than a noisy one.** A 500 ms
  readback at 256² was consistent across six runs, was size-inverted in a way that suggested
  a real effect, and survived two rounds of plausible hardware hypotheses. It was `647.700
  µs` being parsed as milliseconds by the measuring script. Consistency measures the
  measurement, not the thing.
- **Lazy initialization lands wherever you first touch it.** The first buffer round trip on
  a fresh device costs ~56 ms against a ~160 µs steady state, and it silently became "the
  readback is slow". `GpuDevice::open` pays it on purpose now, which is also where §15.1
  would put it.

### M4.4 — Crank–Nicolson on the device ✅

The implicit scheme, solved by conjugate gradient with the dot products running on the
device. Documented in [backends.md](backends.md).

- **`lattice-wgpu::reduction`** — `GpuDot`, a dot product whose association order is fixed
  at construction and *written down*: contiguous blocks per workgroup, a contiguous serial
  run per invocation, two fixed binary trees. `Interior::depth` reports the resulting
  accumulation depth — **16** at 24 576 cells, against 24 575 for a sequential sum — so the
  budget is handed the real number rather than a constant somebody chose.
- **`GpuCrankNicolson`** — thirteen kernels and three reductions, with `α` and `β` staying
  on the device and exactly **one eight-byte readback per iteration**, which is what §10.3's
  stopping rule costs. Two backends' worth of divergence semantics kept identical by moving
  the curvature test *into* the update kernels.
- **A tolerance floor that is enforced, not documented.** `ε·(1 + ‖A‖₂)`, computed at setup
  from the face coefficients. `HeatDomain`'s default `1e-10` is 26 000× below it and is
  refused with the floor and the `‖A‖` it came from.
- **`Mechanism::SolveTermination` and `Norm`** — a mechanism that scales with `τ` rather
  than `ε`, and the first budget in the project that had to say which norm it was derived in.
- **Five more cross-backend rows** (`lattice validate` runs 45 cases, `--features gpu` runs
  53) and **`lattice bench heat-crank-nicolson --backend gpu`**.

1052 tests. Clippy and rustdoc clean.

### What M4.4 taught us

- **The reduction was the wrong thing to be afraid of.** M4.2 deferred this whole piece of
  work because a reduction is where `Mechanism::ReductionOrder` stops being zero, and the
  reasoning was explicit: CG's inner products decide when the iteration stops, so a change
  in reduction order changes the iteration count and through it the answer. All of that is
  true and the term lands **six orders of magnitude below** the budget's largest. A perturbed
  dot product changes *which iterate* CG arrives at, and the stopping test then measures that
  iterate afresh — the reduction moves the path, not the destination. What actually dominates
  is a mechanism that did not exist in the explicit case at all.
- **A reduction inside a solver audits nothing.** The corollary, and the reason there is a
  case comparing a bare `GpuDot` against a sequential sum. A term worth a billionth of a
  budget is a term the budget does not test. Reductions whose value *is* the answer — a
  conserved total, an observation, a fixed-iteration solver — have no re-measurement to hide
  behind, and that is where the mechanism is real.
- **A precision has a floor on what question you may ask it.** Not on the answer's accuracy —
  on the *question*. `ε·(1 + ‖A‖₂)` is what forming `b − Ax` costs even given an exact `x`,
  so a solver asked for less iterates to its cap and reports as slow hardware. This is the
  first place in the project where a backend refuses a *parameter* rather than a precision,
  and it belongs in the same family as `Capabilities::supports`: refuse rather than
  substitute, and say what the limit was.
- **A budget and its measurement must be in the same norm.** Every mechanism before this one
  bounded error per cell, so the max norm was correct without anyone choosing it. A stopping
  rule on `‖r‖₂` is a statement about a vector, and checking it per cell costs `√N` — 157
  here — belonging entirely to the change of norm. Two orders of magnitude of looseness that
  describe nothing is exactly the failure `Tolerance` exists to prevent, so the norm is now
  part of the budget.
- **The stall is the program.** The stencil is 27× the CPU at 256²; a CG iteration is
  **1.8×**. The arithmetic per iteration is the same kind the stencil does — the difference
  is that an iteration contains a fence, because §10.3 requires the residual to be able to
  stop the run. Fusing the scalar copy onto the iteration's own submission, and reusing one
  staging buffer, was worth **1.6×** on its own, which says how much of an implicit device
  solve is queue latency rather than work. (That 1.6× rests on a single before-measurement,
  which is the mistake M4.3 warned about; it is flagged in backends.md and wants re-taking.)
- **Steps per second is not a unit when the step is a solve.** The GPU looked 4.0× the CPU
  and is 1.8×; the gap is entirely that the two stop at different residual tolerances and
  therefore run 4.57 against 9.5 iterations per step. Both benchmarks now publish iterations
  per step, because §19.3's "publish the conditions" has to include the thing that makes two
  numbers comparable.

### Where M4.5 started (historical checklist)

- **Zero-copy rendering** is now the largest unbuilt item in M4. The buffers live on the
  device; the viewer still uploads a CPU texture. The measured readback cost (0.4–5.9 ms per
  frame at these sizes, plus a device open) is what it would remove.
- **`OperationGraph::levels` is *still* unconsumed.** It has computed which operations are
  independent since M1. A backend that schedules across *operations* rather than within one
  is what would use it, and none of M4.2, M4.3 or M4.4 became that.
- **The fence per CG iteration is the implicit path's whole performance story**, and there
  are two ways at it that do not touch §10.3. A preconditioner cuts iterations, and each
  iteration carries a fence, so it cuts wall-clock more than arithmetic — but it changes the
  iteration count and therefore the answer, which makes it a change to the *reference*.
  Speculative dispatch of `k` iterations with a batched residual readback does not change
  the answer, but needs `x` checkpointed to unwind past the converged one. Neither has been
  attempted.
- **Device open is 0.8–1.5 s and nothing has tried to reduce it.** Unchanged from M4.3, and
  it is plausibly mostly adapter enumeration.
- **The particle benchmarks still have no GPU kernels.** §15.6's "local particles" and
  "Lennard-Jones MD" targets are GPU targets, and the pair-force question — a gather or a
  colouring — is still open. It is now the only one of M4.2's four open questions left.
- **Dirichlet on the implicit path** is where `HaloMode`'s homogeneous/inhomogeneous
  distinction finally needs a device-side counterpart, and where `assemble_rhs` stops being
  able to omit the affine term `θ·c`.

**Before starting, reread §23.1's kill criteria**, particularly *"the compiled runtime is
not materially faster or easier to inspect than a straightforward Python prototype."* M4.1
improved the first half. M4.2 arguably improved the second — a validation report that
prints where its tolerance went is an inspectability claim — and did nothing measured for
the first. M4.4 is the first milestone to move the first half *backwards* in an honest
direction: it found a case where the GPU is worth 1.9× rather than 27×, and said so.

### M4.5

M4.5 implementation and remaining limitations are recorded in
[the engineering report](m4-engineering-report.md). In particular, GPU rendering is
available for uncoupled heat fields, and GPU particle execution is available through
the Rust backend API and benchmark harness. Arbitrary mixed-domain GPU execution,
GPU rigid bodies, and GPU chemistry remain outside the released backend scope.

## M5 — Molecular and quantum ✅

**Spec exit condition:** *"energy/norm tests and visual examples."* Two halves: the
molecular module of §12.4, whose tests are energy tests, and `quantum2d` of §13.1, whose
tests are norm tests. Both are done; what each leaves out is listed under it.

### M5.1 — Classical molecular dynamics ✅

§12.4's table, row by row:

| Component | Spec's initial choice | Built |
|---|---|---|
| Integrator | velocity Verlet | velocity Verlet; BAOAB when a Langevin bath is attached |
| Neighbour search | cell list + optional skin | `NeighborList`: a cell list, optionally cached behind a Verlet skin with bonded exclusions |
| Boundary | periodic/reflective/open | all three, with minimum-image geometry that bonds and angles respect across a periodic seam |
| Thermostat | Langevin and simple velocity-rescaling | both, each with its own contract — a thermostatted run does not claim energy conservation |
| Potentials | LJ, harmonic bond, soft repulsion, Coulomb cutoff | LJ (energy- or force-shifted), harmonic bonds and angles, soft repulsion. **Coulomb is not built** |
| Analysis | energy, RDF, MSD, temperature, pressure | all five; the RDF is published as a curve |

In the language: `temperature:`, `layout: serpentine`, `bonds:`, `angles:`,
`thermostat:`, `skin:` and `analysis: rdf(…)` on a `particles` block, documented in
[language.md](language.md). [`examples/argon.lattice`](../examples/argon.lattice) is
liquid argon in a bath with its structure measured;
[`examples/polymer.lattice`](../examples/polymer.lattice) is a microcanonical bead-spring
chain.

Nine validation cases (`lattice validate --filter molecular2d`): §19.2's *"Lennard–Jones
energy conservation, radial distribution trends, and neighbor-list consistency"*, plus
the reduced-mass bond period, bonded-chain energy, both thermostats against their exact
results, and Ornstein–Uhlenbeck diffusion. Every tolerance is derived — from Verlet's
phase error, from the chain's fastest mode, or from the counting statistics of the
sample — and every case prints what it measured against what the derivation predicted.

A new IR type came with it: `Curve`, a result that is a function rather than a number.
`Domain::curves` publishes them, the run artifact carries them (inside the content hash),
and `lattice run` plots them.

### What M5.1 taught us

- **A result nobody can read is not a result.** The RDF was accumulated correctly from the
  first draft — and then discarded at the end of every run, because the artifact had a
  timeline of numbers and nowhere to put a function. §21.1 lists *data output* beside
  validation for a reason: the unit tests passed, the analysis existed, and a user who
  asked for it got nothing back.
- **Reserved words collide with arguments.** `rdf(every=10)` did not parse, because
  `every` is the keyword `observe … every` uses. A keyword before `=` in an argument list
  is now a parameter name, since nothing else can stand there.
- **A statistical tolerance has to count the right thing.** The first Langevin case
  assumed `2γt` independent samples where the kinetic energy's correlation time gives
  `γt`, and the tolerance came out √2 too tight — tight enough that an honest run sat at
  2.8 standard errors. Five seeds now average −0.6.
- **Not every theoretical error is visible at every length.** The energy-shifted cutoff
  injects an impulse at every crossing, and the force-shifted form exists to remove it —
  but over two τ of a 100-atom fluid the energy-shifted error still converges at order
  1.89. The case reports that number rather than the story it was expected to tell.

### M5.2 — `quantum2d` ✅

A new crate, `lattice-domain-quantum2d`, named as §24 names it: one particle's
wavefunction on a 2D grid, by both of §13.1's methods.

| §13.1 capability | Built | Validated by |
|---|---|---|
| Time evolution | split-step Fourier (spectral, periodic) and Crank–Nicolson (five-point, walls); a quadratic complex absorbing layer for either | free packet spreading to 1e-13; tunnelling within 0.09% of the momentum-averaged analytic T; Crank–Nicolson converging to split-step at order 2.02; norm and `⟨H⟩` to the solver tolerance |
| Eigenstates | imaginary-time propagation of a block with Rayleigh–Ritz | the box's discrete spectrum to 4e-15 and its continuum limit at order 2.00; the 2D oscillator's six lowest levels to 2e-7 ħω |
| Potential | walls with slits, rectangles, harmonic traps; anything else through `Potential::from_fn` | — |
| Observables | norm, absorbed probability, position, momentum and energy; detectors integrating the probability current | the double slit's norm plus absorbed probability to 1.5e-14; absorber reflection 1.4e-5 |
| Visualization | probability density, phase and potential as render channels; the detector's arrival pattern as a curve | the viewer loads, steps and draws the double slit |

The FFT is written here — iterative radix-2, plus Bluestein's chirp-z for any other
length, because §25.2's grid is 768 wide and refusing the spec's own example for an
implementation reason would be the wrong trade. In the language: `domain quantum2d`,
`potential`, `wavepacket` and `detector`, documented in [language.md](language.md).
§25.2 compiles as written, and [`examples/double_slit.lattice`](../examples/double_slit.lattice)
is the same scene with numbers that work.

### What M5.2 taught us

- **The spec's own example is wrong, and the compiler can say how.** §25.2's electron
  has 145 eV; its wall is 20 eV high. It would sail over the barrier rather than pass
  through the slits. Its step puts 9.4 radians of phase per step on the grid's top mode
  against a wall's sharp edges. Both are now warnings (`W0308`, `W0310`), and the
  example that ships uses 5.8 eV and the domain's own step.
- **Sharp edges set split-step's step, not the physics.** A rectangular barrier came out
  1.9% too transmissive at a step that resolved every phase the packet itself had. The
  barrier's edges have Fourier content up to the grid's Nyquist wavenumber, and the
  splitting commutator lives there. Holding the top mode's phase to 2 radians per step
  brought it within 0.06%; that is the default now.
- **A method that is stable in real time can be backwards in imaginary time.**
  Crank–Nicolson multiplies an eigencomponent by `(1 − x)/(1 + x)`, which tends to −1 as
  `x` grows: at a useful step the grid's *highest* states decayed slowest, and the first
  eigenstate search converged onto 43 eV in a box whose ground state is 0.28 eV.
  Imaginary time is backward Euler, whose factor falls monotonically.
- **COCG breaks down; the system did not need it.** The Crank–Nicolson matrix is complex
  symmetric, the textbook case for COCG, and COCG blew up on the 23rd step of a free
  packet when its unconjugated inner product passed through zero. The matrix is also
  positive real — every eigenvalue at least 1 from the origin — so restarted BiCGSTAB
  converges in two to eight iterations and has not failed since.
- **An absolute tolerance is a unit choice in disguise.** The runtime's sampling test
  added 1e-12 *seconds* of slack, harmless at the scale of seconds. A femtosecond model
  found every step due and sampled two thousand times instead of thirty. The slack is
  half a step now, the rule the duration already used.
- **Drawing cost as much as stepping.** The phase channel is an `atan2` per cell, and on
  a 512² grid that took as long as the two FFTs. The channels are now computed when
  drawn rather than when stepped.

### What M5 leaves out

- **Coulomb with a declared cutoff** — the one §12.4 potential not built. A plain cutoff
  on `1/r` is a poor approximation the contract would have to say a great deal about;
  doing it properly is Ewald-shaped work.
- **Measurement-inspired sampling and the probability current as a picture** — §13.1 and
  §17.1. The current is computed (the detectors integrate it) but not drawn, and there is
  no sampling of detection events.
- **A phase wheel** — §17.1 asks for one. The viewer draws phase with its ordinary
  colourmaps, which are monotone and therefore wrong for a cyclic quantity; and it does
  not draw curves yet, so the interference pattern is in `lattice run` and the artifact
  but not the window.
- **Speed.** A 512² split step is 12 ms on one core — about 80 steps a second, under
  §15.6's "interactive/near-interactive" for that size. The row and column transforms
  are independent and would split across the worker pool without changing a bit; that
  has not been done, nor a GPU path.

### Where M5.2 started (historical)

- **`quantum2d`** — §13.1: a complex wavefunction on a grid, split-step Fourier where the
  boundaries permit and Crank–Nicolson where they do not, imaginary-time eigenstates,
  absorbing boundaries, and the norm, expectation and detector observables. §19.2's
  quantum row is its validation list, and §25.2's double slit is its example.
- **Coulomb with a declared cutoff** is the one §12.4 potential not built. A plain cutoff
  on `1/r` is a poor approximation the contract would have to say a great deal about;
  doing it properly is Ewald-shaped work.

## Outside the milestones: the playground

`lattice-play` is not in the specification. It exists because the spec describes a
scientific instrument — write a model, compile it, run it, measure it — and someone who
opens the project reasonably expects to be able to *touch* the physics before they learn
a language to describe it. So the playground is a second front door onto the same
solvers, built by clicking rather than by compiling.

It is not a shortcut around any of the standing constraints. It publishes the same
observations, its panel applies the same conservation rules, and it added the same kind
of test the rest of the engine has: sixty-two tests driving `Playground::pointer` directly,
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
| M6 — Extensibility | expression compiler (§8.3), Python API, plugin SDK | M1, done |
| M7 — Productization | packages, report export, reproducibility artifacts | M3, done |
| M8 — External solvers | quantum/FMI adapters with provenance | M7 |

## Standing constraints

From spec §21.1, and they apply to every milestone above:

> A milestone is not complete because a demo looks compelling. It is complete only when
> model semantics, validation, diagnostics, data output, and documentation exist. New
> domains should not be added while the current domain lacks a reference test or cannot
> explain its stability limits.

And the kill criteria of §23.1 are worth rereading before the rest of M4, particularly:

> The compiled runtime is not materially faster or easier to inspect than a
> straightforward Python prototype.

M0's numbers (774M cell-updates/s, 427M particle-updates/s, single-threaded) clear the
"materially faster" half, and M4.1 multiplies the grid figure by another 4.8×. The
"easier to inspect" half is what `lattice inspect contracts` and the measured validation
report are for, and it stays an open question until someone outside the project uses them.

§23's *"performance portability — portable GPU abstractions may leave performance on the
table"* is the risk M4's remaining three items run into, and M4.1 is a small preview of
it: the parallel CPU backend leaves plenty on the table (5× on twenty threads) and the
reason is memory bandwidth, which no amount of scheduling recovers. Naming the wall
matters more than the ratio.
