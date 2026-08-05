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
- **`lattice check` and `lattice run`** — M1's exit condition, minus a window.
- **Five example models and sixteen invalid fixtures**, each fixture declaring the
  diagnostic code it must produce.

591 tests. Clippy clean. Still zero external dependencies.

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

## M2 — Mechanics and fields (next)

**Spec exit condition:** *"canonical validation suite passes"* for particles, simple
rigid bodies, fields, diffusion and observers.

Needed:

1. **`lattice-domain-rigid2d`** — circles, boxes and convex polygons; forces, torques
   and impulses; a broadphase grid or BVH, narrowphase contacts, friction and
   restitution; distance, pin, spring and motor constraints (§11.1).
2. **Collision validation** — §19.2's elastic and inelastic collision cases, which M0
   deliberately left out because the rigid module did not exist.
3. **`material` declarations** — the §8.2 material concept, which rigid bodies are the
   first real consumer of.
4. **CPU parallelism** — §15.3 asks for parallel iterators over independent operations.
   The operation graph already computes which operations those are and reports the
   ideal speedup; nothing consumes that yet.

## Later milestones

| Milestone | Result | Blocked on |
|---|---|---|
| M3 — Chemistry | reaction networks, reaction-diffusion, heat coupling, ledger | M2; the ledger and coupling diagnostics are already built |
| M4 — Portable GPU | `wgpu` compute, kernel cache, zero-copy rendering | the operation graph, which exists |
| M5 — Molecular and quantum | LJ MD proper, bonds, `quantum2d` | M2 |
| M6 — Extensibility | expression compiler (§8.3), Python API, plugin SDK | M1, done |
| M7 — Productization | packages, report export, reproducibility artifacts | M3 |
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
