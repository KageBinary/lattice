# Project Lattice

A 2D-first multiphysics, chemistry, and quantum simulation runtime.

> For students, educators, researchers, and technical builders who need to explore
> interacting physical and chemical systems, Lattice is a 2D-first simulation engine
> that combines an expressive typed model with portable high-performance execution and
> visible numerical diagnostics. Unlike a game engine or a single-domain scientific
> package, Lattice treats solver assumptions, coupling, fidelity, units, and error as
> first-class parts of the model.
>
> — [Technical & Product Specification](docs/spec/project-lattice-spec-v0.1.md) §2.2

---

## Status: milestones M0 and M1 complete

The spec lays out nine milestones, M0 through M8.

- **M0 — Numerical kernel spike.** Exit condition: *"analytic heat + particle demos
  and benchmark harness."*
- **M1 — Compiled model.** Exit condition: *"same project executes headless and
  interactively."*

Being specific about that, in the spirit of design principle **P1 — scientific
honesty over feature count**:

### What is real

| Area | Implemented | Validated by |
|---|---|---|
| **Units** | 7-dimension analysis, SI registry with prefixes, unit-expression parser, CODATA constants | 83 tests; every dimensioned literal in the spec's example models parses to the right dimension |
| **Language** | The `.lattice` DSL — lexer, parser, AST, source-positioned diagnostics with carets | Parses spec §25.1, §25.2 and §12.2 verbatim |
| **Compiler** | Name and unit resolution, dimensional checking of every expression, solver selection, buffer planning, operation graph, model report | 16 invalid fixtures each rejected for its declared reason |
| **Runtime** | Clock, timestep negotiation across domains, observers on a cadence, run artifacts | Refuses unstable steps before running; halts on the first non-finite value |
| **Storage** | Structure-of-arrays particles with stable handles, halo'd grid fields, bump arenas, reproducible RNG | 112 tests; no allocation in stepping loops |
| **Particles** | Explicit Euler, semi-implicit Euler, velocity Verlet; gravity, drag, harmonic wells, Lennard-Jones; uniform cell list | Free fall, oscillator period, energy drift, convergence order, momentum conservation |
| **Heat / diffusion** | Finite-volume `∇·(D∇u)`, explicit / Crank–Nicolson / backward Euler, matrix-free conjugate gradient, Dirichlet / Neumann / Robin / periodic boundaries, variable diffusivity | Analytic heat kernel, manufactured solutions, convergence orders, conservation, series conduction |
| **Diagnostics** | Solver contracts, conservation drift monitors, coupling ledger, residual histories, render channels | Every solver publishes equations, assumptions, and what it does *not* conserve |
| **Tooling** | `lattice check`, `run`, `validate`, `bench`, `demo`, `inspect`; JSON run artifacts with reproducible content hashes; terminal viewer | 591 tests across 10 crates |

### What is not built yet

Rigid bodies, fluids, waves, electromagnetism, chemistry, molecular dynamics beyond
Lennard-Jones, the quantum module, the coupling scheduler, GPU execution, the
graphical viewer, and the Python SDK. Those are M2–M8.

Constructs the language accepts but cannot execute — `reaction`, `couple`,
`domain quantum2d` — are **compile errors that name the milestone that will implement
them**. Not warnings: a model whose chemistry was silently dropped would run and
produce confident wrong numbers.

### What this is not

Not a replacement for LAMMPS, OpenMM, or Quantum ESPRESSO. Not validated against
experiment. The heat solver reproduces analytic solutions at the expected order; that
is a statement about the discretization, not about any real material.

---

## Quick start

Requires Rust 1.85 or newer. On Windows without the MSVC C++ build tools, use the GNU
toolchain — see [docs/development.md](docs/development.md).

```console
$ cargo build --release
$ ./target/release/lattice check examples/slab.lattice
$ ./target/release/lattice run   examples/slab.lattice
$ ./target/release/lattice validate
```

### A model

Models are written in the `.lattice` language — see
[docs/language.md](docs/language.md) for the reference, and `examples/` for five
working scenes.

```
project slab {
  fidelity: engineering_2d;
  duration: 400 second;

  grid bar { size: [64, 8]; extent: [1 meter, 0.125 meter]; }

  // The field's dimension comes from its initial value: this is a temperature field
  // because 273.15 kelvin is a temperature. Every boundary below is checked against
  // that, so `fixed(5 second)` would be a compile error.
  field temperature on bar = 273.15 kelvin {
    diffusivity: 1e-2 meter^2 / second;
    boundary:       insulated;
    boundary_left:  fixed(273.15 kelvin);
    boundary_right: fixed(373.15 kelvin);
  }

  solve heat(temperature) with crank_nicolson(dt=0.5 second);
  observe temperature every 20 second;
  visualize temperature as heatmap;
}
```

`lattice run` on that reaches the linear steady profile every conduction textbook
opens with:

```text
  temperature — 64x8 scalar field in K
  |     ........::::::::--------=======++++++++********########%%%%%%%%@@@@|
  |     ........::::::::--------=======++++++++********########%%%%%%%%@@@@|
  |     ........::::::::--------=======++++++++********########%%%%%%%%@@@@|
  scale: 273.931250  .:-=+*#%@ 372.368750   span 9.844e1
```

273.93 K and 372.37 K are the first and last *cell centres* — half a cell in from the
273.15 and 373.15 faces, which is exactly where they should be.

### `lattice check`

Compiles without running, and prints the model report spec §8.4 step 9 calls for:
which solver was selected, what it assumes, what it does not conserve, how much memory
it will take, and what the compiler had to approximate.

```text
domains (1)
  temperature : grid2d.heat[crank_nicolson]
      64x8 on `bar`, values in temperature (K), D = 1.0000e-2 m^2/s,
      left Dirichlet, right Dirichlet, bottom Neumann, top Neumann

operation graph
3 operations in 3 levels (max width 1)
  ...
  total cost 10240.2, critical path 10240.2 (ideal speedup 1.00x)
```

### `lattice validate`

Runs the validation suite and reports the number each case measured, not just a
verdict. An error that is still under tolerance but has grown tenfold is a regression
worth seeing.

```text
MANUFACTURED — does the discretization converge at the expected order?
  [PASS] integrator_order_velocity_verlet       2.000000 1   (want 2.000000 ± 1.500e-1)
         velocity Verlet converges at the second order it declares
         metric: observed convergence order
         note:   errors 3.031e-7 -> 7.577e-8 on halving dt
  [PASS] heat_temporal_convergence_crank_nicolson  1.999406 1   (want 2.000000 ± 2.000e-1)
  [PASS] heat_spatial_convergence               1.955115 1   (want 2.000000 ± 1.500e-1)

24 of 24 cases passed in 314.771 ms
```

### `lattice demo oscillator`

The same oscillator under all three integrators, which is why spec §10.2 keeps
explicit Euler around *"only for teaching/comparison"*:

```text
  euler                    ..............::::::::------   [1.003948e0 .. 1.063384e1]
              order 1, symplectic false  final E/E0 = 10.633843
  symplectic  ==  %## ++@..::@== %%#  +@@..:@@=  %##  +   [9.698779e-1 .. 1.032105e0]
              order 1, symplectic true   final E/E0 = 0.999610
  verlet      @@--.%%+  *%%..=@@-..%++ **%%.==@--.%%++    [9.990170e-1 .. 1.000000e0]
              order 2, symplectic true   final E/E0 = 1.000000
```

Explicit Euler multiplies the energy by exactly `(1 + (ωΔt)²)` per step. That is not a
bug — it is the closed-form failure mode, and the validation suite asserts it to six
digits.

### `lattice bench`

Every performance number is reported next to the correctness condition spec §15.6
attaches to it, because *"a faster wrong solver is a regression"* (§15.1). Measured on
one desktop CPU core, release build:

| Benchmark | Size | Throughput | Correctness |
|---|---|---|---|
| `particles-gravity` | 32,768 particles | 427M particle-updates/s | matches the analytic trajectory to 2.8e-14 m |
| `particles-lj` | 4,096 particles, cell list | 5.98M particle-updates/s | energy drift 7.4e-7, momentum drift 7.1e-17 |
| `heat-explicit` | 512×512 grid | 774M cell-updates/s | integral conserved to 4.4e-16 |
| `heat-crank-nicolson` | 512×512 grid, 10× the explicit limit | 34.5M cell-updates/s | integral conserved, every solve converged |

A benchmark whose correctness condition fails has its throughput marked
`RESULT INVALID` rather than published.

### `lattice inspect contracts`

Every solver publishes its equations, assumptions, valid regime, and — the part that
matters most — what it is known *not* to conserve:

```text
grid2d.heat[crank_nicolson] — scalar transport by diffusion on a uniform 2D grid
  fidelity:       F1 (quantitative small-scale models with validation)
  equations:      du/dt = div(D grad u) + S
  stability:      unconditionally stable for any dt. Stability is not accuracy:
                  Crank-Nicolson rings on sharp data at large dt...
  does NOT conserve:
      - the field integral is conserved only on a closed boundary with no source...
      - the implicit solve is iterative, so conservation holds to the solver tolerance...
      - harmonic-mean face diffusivities are exact for a layered medium but only
        first-order accurate across a diagonal material interface
```

---

## Repository layout

Follows spec §24, with crates added as each milestone lands.

```
lattice/
  crates/
    lattice-units/            dimensions, quantities, unit registry and parser
    lattice-syntax/           lexer, AST, parser, source-positioned diagnostics
    lattice-ir/               typed IDs, SoA storage, grids, arenas, solver contracts,
                              compiled model, operation graph, render channels
    lattice-compiler/         name and unit resolution, dimensional checking, lowering
    lattice-runtime/          clock, timestep negotiation, stepping, run artifacts
    lattice-domain-particle/  integrators, force laws, neighbour search
    lattice-domain-grid2d/    diffusion operator, boundaries, conjugate gradient
    lattice-observe/          JSON, timing profiles, run artifacts
    lattice-validation/       the validation lab
    lattice-cli/              the `lattice` binary
  examples/                   working .lattice models
  tests/invalid/              models that must be rejected, each declaring why
  docs/
    architecture.md           how the code maps onto the specification
    language.md               the .lattice language reference
    development.md            toolchain setup and conventions
    roadmap.md                what each milestone delivered
    spec/                     the source specification
```

## Dependencies

There are none. Every crate builds from `std` alone — including, somewhat to my own
surprise, the whole M1 compiler and its diagnostics.

This will not survive contact with M4 (`wgpu`) or M6 (`pyo3`), and external crates are
approved for those. Spec §24.1 permits mature libraries *"where they do not define the
core semantics"*, and the line has held so far because units, dimensional analysis,
the IR, and the reproducibility guarantee all *are* core semantics:

- The ~40-line PCG generator, rather than `rand`, buys a reproducibility promise that
  outlives any dependency's major version — which FR-011 needs.
- The JSON writer exists because artifacts must be byte-stable and must preserve
  `NaN`, and general-purpose serializers do neither.
- The parser is hand-written because FR-002's diagnostics are a *product surface*, not
  an implementation detail, and a generator's error messages are nobody's design.

## Design principles in practice

The spec's ten principles are not decoration; each one shows up as a specific
decision:

- **P1 — scientific honesty.** Every solver's contract has a `known_non_conservation`
  field, and a validation case fails the build if any solver leaves it empty.
- **P3 — compile before running.** Units exist only at model-construction time; the
  hot loop is unit-free `f64` in coherent SI.
- **P4 — data-oriented hot paths.** Structure-of-arrays storage, compacted on removal,
  with an indirection table so handles stay stable.
- **P5 — fidelity is explicit.** Each integrator and time scheme gets its *own*
  contract, because explicit Euler and velocity Verlet make opposite claims about
  energy and one document cannot be honest about both.
- **P6 — determinism is a feature.** The RNG is reproducible across platforms and
  versions by construction, and its state — including the cached normal spare —
  round-trips through a checkpoint.
- **NFR-007 — expose instability.** `NaN` survives serialization instead of becoming
  `null`; the terminal viewer marks non-finite cells with `!`; a conserved quantity
  wobbling at round-off renders flat rather than as a dramatic oscillation.

## Licence

Apache-2.0 OR MIT.
