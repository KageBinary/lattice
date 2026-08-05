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

## M1 — Compiled model (next)

**Spec exit condition:** *"same project executes headless and interactively."*

Needed:

1. **`lattice-syntax`** — lexer, parser, AST, and source-positioned diagnostics for the
   `.lattice` DSL of §25. The unit parser already produces the error shape this needs;
   the remaining work is spans and a project grammar.
2. **`lattice-compiler`** — name and type resolution, dimensional checking of every
   expression (FR-002), lowering to domain operations, read/write set construction, and
   the buffer plan that sizes the arenas M0 currently sizes by hand.
3. **Operation graph** — the DAG §9.2 describes, with the scheduler that walks it. This
   is the point at which `lattice-ir` grows the "immutable `CompiledModel`" half of the
   §7.2 split that M0 only half-implements.
4. **Ten invalid fixture models** — §20.4 asks for at least ten deliberately invalid
   models rejected with source-positioned errors. The unit layer already carries eleven
   of these at expression level; they need to be lifted to whole models.

The acceptance test is spec §25.1's `hot_reaction` project parsing, compiling, and
reporting its dimensional errors — even before the chemistry it describes exists.

## Later milestones

| Milestone | Result | Blocked on |
|---|---|---|
| M2 — Mechanics and fields | rigid bodies, contacts, constraints, observers | M1 for scene description |
| M3 — Chemistry | reaction networks, reaction-diffusion, heat coupling, ledger | M1; the ledger itself is already built |
| M4 — Portable GPU | `wgpu` compute, kernel cache, zero-copy rendering | M1's operation graph |
| M5 — Molecular and quantum | LJ MD proper, bonds, `quantum2d` | M2 |
| M6 — Extensibility | expression compiler, Python API, plugin SDK | M1 |
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
