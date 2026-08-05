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

## Status: milestones M0, M1 and M2 complete

The spec lays out nine milestones, M0 through M8.

- **M0 — Numerical kernel spike.** Exit condition: *"analytic heat + particle demos
  and benchmark harness."*
- **M1 — Compiled model.** Exit condition: *"same project executes headless and
  interactively."* Both halves: `lattice run` headless, `lattice-view` in a window.
- **M2 — Mechanics and fields.** Exit condition: *"canonical validation suite passes."*
  33 of 33 cases, including §19.2's elastic and inelastic collision and constrained
  motion, which M0 had to leave out because the rigid module did not exist.

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
| **Rigid bodies** | Circles, boxes, convex polygons, segments; sweep-and-prune broadphase, SAT narrowphase, friction and restitution; distance, rope, pin, spring and motor joints; sequential-impulse solver with warm starting | Elastic collision exchanges velocities exactly; inelastic loses exactly the predicted energy; pendulum period within 0.008% of analytic; Coulomb friction threshold to the digit |
| **Heat / diffusion** | Finite-volume `∇·(D∇u)`, explicit / Crank–Nicolson / backward Euler, matrix-free conjugate gradient, Dirichlet / Neumann / Robin / periodic boundaries, variable diffusivity | Analytic heat kernel, manufactured solutions, convergence orders, conservation, series conduction |
| **Diagnostics** | Solver contracts, conservation drift monitors, coupling ledger, residual histories, render channels | Every solver publishes equations, assumptions, and what it does *not* conserve |
| **Tooling** | `lattice check`, `run`, `validate`, `bench`, `demo`, `inspect`; JSON run artifacts with reproducible content hashes; terminal viewer | 757 tests across 12 crates |
| **Viewer** | `lattice-view` — a window with field heatmaps, particle scatter, rigid-body outlines, contact normals, transport controls, live plots, conservation drift and the solver's contract | Perceptually uniform ramps asserted single-hue and monotone in lightness; flat fields and round-off never drawn as structure |

### What is not built yet

Fluids, waves, electromagnetism, chemistry, molecular dynamics beyond Lennard-Jones,
the quantum module, the coupling scheduler, GPU execution, and the Python SDK. Those
are M3–M8. The viewer draws through a CPU texture upload, which is fine at 64×64 and
will not be at 768×384; GPU rendering is M4. Rigid-body collision detection is
discrete, so a fast thin projectile can pass through a thin wall — continuous
collision detection is what §11.1 lists under "later".

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

To watch one run instead, build the viewer — a separate binary, because its GPU stack
is a few hundred crates and the core CLI stays dependency-free without it:

```console
$ cargo build --release -p lattice-viewer
$ ./target/release/lattice-view examples/diffusing_pulse.lattice --play
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

### `lattice-view`

A window on a running model: the field or the particles at full size, and beside them
the numbers that say whether to believe the picture — every observed quantity, how far
each conserved one has drifted and against what, the timestep as a fraction of its
stability limit, and the solver's published contract.

```text
stability
  ✔ stable        dt is 2% of the 0.2646 s limit
  set by domain `atoms`
conservation
  total_energy
  ! drifting slowly          -2.530e-4 of the initial value
  momentum_x
  ✔ conserved to round-off   -1.454e-16 of momentum_scale
```

That last line is the whole idea. Total momentum in a system set up at rest is
conserved *at zero*, so its initial value is round-off; dividing by it reported a
flawless run as `✖ not conserved -7.288e1`. The particle domain now publishes
`momentum_scale = Σ|mᵢvᵢ|` — a scale only it can know — and the viewer says which
denominator it used. See [docs/viewer.md](docs/viewer.md), which is mostly a list of
ways a picture can assert something the data does not say.

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
    lattice-domain-rigid2d/   shapes, broadphase, contacts, joints, impulse solver
    lattice-observe/          JSON, timing profiles, run artifacts
    lattice-validation/       the validation lab
    lattice-cli/              the `lattice` binary
    lattice-viewer/           the `lattice-view` window: colourmaps, heatmaps,
                              plots, the diagnostics panel
  examples/                   working .lattice models
  tests/invalid/              models that must be rejected, each declaring why
  docs/
    architecture.md           how the code maps onto the specification
    language.md               the .lattice language reference
    development.md            toolchain setup and conventions
    roadmap.md                what each milestone delivered
    viewer.md                 what the window shows and the rules it draws by
    spec/                     the source specification
```

## Dependencies

Ten of the eleven crates have none. Everything from units through the compiler to the
validation lab builds from `std` alone — including, somewhat to my own surprise, the
whole M1 compiler and its diagnostics.

The exception is `lattice-viewer`, which needs `eframe`/`egui` and `egui_plot` to have
a window at all. Spec §24.1 permits mature libraries *"where they do not define the
core semantics"*, and an immediate-mode widget set does not: the colourmaps, the drift
arithmetic, and every rule in [docs/viewer.md](docs/viewer.md) are ours and are tested
here. `cargo test` on the other ten crates does not build it.

Elsewhere the line has held because units, dimensional analysis, the IR, and the
reproducibility guarantee all *are* core semantics:

- The ~40-line PCG generator, rather than `rand`, buys a reproducibility promise that
  outlives any dependency's major version — which FR-011 needs.
- The JSON writer exists because artifacts must be byte-stable and must preserve
  `NaN`, and general-purpose serializers do neither.
- The parser is hand-written because FR-002's diagnostics are a *product surface*, not
  an implementation detail, and a generator's error messages are nobody's design.

M4 (`wgpu`) and M6 (`pyo3`) will add more.

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
