# Architecture

How the code maps onto the [specification](spec/project-lattice-spec-v0.1.md), and why
the non-obvious decisions were made that way.

## Layering

```
                  ┌───────────────────────┐  ┌──────────────────┐
   authoring      │ lattice-cli           │  │ lattice-viewer   │
                  │ check/run/validate    │  │ `lattice-view`   │
                  │ bench/demo/inspect    │  │ the window       │
                  └───────────┬───────────┘  └────────┬─────────┘
                              └────────┬──────────────┘
                                       │
     ┌─────────────────┬───────────────┼───────────────┬──────────────────┐
     │                 │               │               │                  │
┌────▼─────────┐ ┌─────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐ ┌───────▼───────┐
│ lattice-     │ │ lattice-   │ │ lattice-     │ │ lattice-     │ │ lattice-      │
│ validation   │ │ compiler   │ │ runtime      │ │ domain-*     │ │ observe       │
│ (the lab)    │ │ resolve,   │ │ clock, dt    │ │ particle,    │ │ JSON, timing, │
│              │ │ check,     │ │ negotiation, │ │ grid2d,      │ │ artifacts     │
│              │ │ lower      │ │ stepping     │ │ rigid2d,     │ │               │
│              │ │            │ │ + coupling   │ │ chemistry    │ │               │
└────┬─────────┘ └──┬──────┬──┘ └──────┬───────┘ └─────┬────────┘ └───────┬───────┘
     │              │      │           │               │                  │
     │       ┌──────▼───┐  └───────────┼───────────────┼──────────────────┘
     │       │ lattice- │              │               │
     │       │ syntax   │              │               │
     │       │ lex,     │              │               │
     │       │ parse,   │              │               │
     │       │ diagnose │              │               │
     │       └──────┬───┘              │               │
     │              │                  │               │
     └──────────────┴──────────────────┴───────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  lattice-ir             │
                          │  IDs, SoA storage,      │
                          │  grids, arenas,         │
                          │  Domain contract,       │
                          │  CompiledModel, graph,  │
                          │  diagnostics, render    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
             ┌────────────▼────────────┐ ┌──────────▼──────────────┐
             │  lattice-units          │ │  lattice-cpu            │
             │  dimensions, quantities │ │  worker pool, Executor, │
             │                         │ │  Grain, Partition       │
             └─────────────────────────┘ └─────────────────────────┘
```

Dependencies point downward only. `lattice-ir` holds no physics; the domain crates
hold no storage layout decisions; `lattice-runtime` does not depend on the compiler —
it takes a `CompiledModel` and a `Vec<Box<dyn Domain>>`, and the CLI wires the two
together.

`lattice-cpu` is at the bottom beside `lattice-units` because it knows nothing about
simulation: it splits slices and runs closures. It is *below* `lattice-ir` rather than
beside the domains because `StepContext` carries an `Executor` — a solver is handed the
schedule the same way it is handed the clock, and for the same reason. Spec §24 names
this crate `lattice-cpu`; the `lattice-compute` backend traits and `lattice-wgpu` it
names alongside are M4's remaining work.

`lattice-viewer` sits beside the CLI rather than under it: both are consumers of the
same compile-then-run path, and neither is on the other's path. Nothing below the
authoring layer knows a window exists — a domain publishes `Observations` and
`RenderChannel`s and cannot tell whether they are going to a terminal, a JSON artifact,
or a texture. See [viewer.md](viewer.md).

## The compilation pipeline

Spec §8.4 lists nine steps. Text becomes a runnable model like this:

```
  .lattice text
       │
       ▼  lattice-syntax
  tokens ──► AST                    units are ordinary identifiers here;
       │                            nothing is resolved yet
       ▼  lattice-compiler
  ┌────────────────────────────────────────────────────────┐
  │ resolve   grids, fields, particle sets, unit names      │
  │ check     every expression, dimensionally  (FR-002)     │
  │ select    which solver implements each `solve`          │
  │ plan      buffers, scratch arena size                   │
  │ schedule  operation graph from read/write sets          │
  └────────────────────────────────────────────────────────┘
       │
       ├──► CompiledModel   immutable: what the model *is*
       └──► Vec<Box<dyn Domain>>   the solvers that will run it
```

Two decisions in that pipeline shaped everything downstream.

### Units are not a lexical concern

The first design gave the lexer a "unit mode" entered after a number. It could not
decide what `100 / dt` meant, because the lexer does not know what has been declared.

So there is no unit mode. `kilojoule` lexes as an identifier, and name resolution
decides: declared names first, then the unit registry. `100 / dt` is a division by a
parameter; `100 / second` is a frequency; both are one rule. An identifier that is
neither gets a diagnostic naming both possibilities, rather than "unknown unit `dt`"
when the user meant a parameter they forgot to declare.

The cost is that `35 kilojoule` has no operator between its terms — and adjacency has
to bind *tighter* than `*` and `/`, or `10 meter / 2 second` groups as `((10 m)/2)·s`
and yields m·s. A dimensioned literal is one atom.

### Declaration kinds are not in the grammar

`grid` was a keyword until spec §25.2's `grid: [768, 384];` — where `grid` is a
*setting key* inside a domain block — showed why that fails. Now `grid chamber { … }`,
`reaction r { … }` and `wavepacket initial { … }` all take one generic
`<kind> <name> { … }` path, and the *compiler* decides which kinds it knows.

A new solver family therefore needs no grammar change, and an unknown kind gets a
diagnostic that distinguishes "not a thing" from "not implemented until M5".

### A field's dimension comes from its initial value

`field temperature on bar = 273.15 kelvin;` is a temperature field because its initial
value is a temperature. Every boundary, source and solver parameter is then checked
against that.

The alternative — a separate `dimension:` declaration — is one more thing to keep in
sync, and the failure mode when it drifts is a model that compiles and is wrong.

## The one invariant that shapes everything

**Every magnitude in the running engine is in coherent SI base units.**

Units are surface syntax. `35 kilojoule/mole` becomes `35000.0` the moment it is
parsed, and from there the runtime is unit-free `f64` arithmetic. This is spec P3
(*"expensive interpretation… should occur before the hot loop"*) taken literally: a
`Quantity` is a compiler-time type, and finding one inside a stepping function means
something has gone wrong.

The consequence is that dimensional checking is *free at runtime*, because it has
already happened. `Quantity::require(dimension, op)` is the gate: it hands back the raw
`f64` a solver wants, but only after proving the dimension matches.

## Model / data separation (§7.2)

The spec asks for a strict split between immutable compiled model state and mutable
simulation data. At M0 there is no compiler yet, so the split appears in a smaller
form: solver *configuration* is immutable after `initialize()`, solver *state* is the
preallocated arrays.

Two properties follow, and both are enforced by tests:

- **Capacity is fixed.** `ParticleStore::spawn` returns `None` at capacity rather than
  reallocating. A reallocation mid-step is exactly the hidden hot-loop allocation
  NFR-001 forbids, and refusing is more honest than growing silently.
- **Scratch is preallocated.** `Arena` hands out disjoint `&mut [f64]` slices by
  repeatedly `split_at_mut`ing one buffer. Multiple slices can be live at once with no
  `unsafe`, and a whole run performs one allocation.

## The solver contract (§7.3, NFR-003)

Every domain implements `Domain` and publishes a `SolverContract`:

```rust
pub trait Domain {
    fn name(&self) -> &str;
    fn contract(&self) -> &'static SolverContract;
    fn stable_step(&self) -> StableStep;
    fn prepare(&mut self, ctx: &mut StepContext<'_>);
    fn advance(&mut self, dt: f64, ctx: &mut StepContext<'_>);
    fn observe(&self, out: &mut Observations);
}
```

The contract is `&'static`, so it is a compile-time constant with no runtime cost, and
carries governing equations, discretization, assumptions, valid regime, stability
criterion, conserved quantities, **known non-conservation**, fidelity profile,
supported precisions, and validation cases.

`known_non_conservation` is the field that makes the contract honest. A solver that
leaves it empty is claiming exact conservation of everything it touches — so
`SolverContract::audit()` and a validation case both check for it, and the build fails
if any shipped solver is silent.

### One contract per *configuration*, not per module

`ParticleDomain` returns a different contract for each integrator, and `HeatDomain`
for each time scheme. This is not over-engineering. Explicit Euler and velocity Verlet
make opposite claims:

| | explicit Euler | velocity Verlet |
|---|---|---|
| conserves | momentum only | energy and momentum |
| stability | unstable for oscillatory systems at every `dt` | `dt < 2/ω_max` |
| fidelity | F0, teaching only | F1, engineering |

A single contract could only be honest about one of them.

## Stability reasons, not just numbers

`StableStep` carries a `StabilityReason` alongside the limit:

```rust
StableStep::limited(preferred, max, StabilityReason::DiffusionExplicit)
```

so the viewer can answer *"why is my timestep so small?"* with the responsible
mechanism rather than a bare number (§7.3, §17.3). `StableStep::tightest` combines
constraints across domains and keeps the reason from whichever one binds.

## The boundary-condition split

This is the subtlest piece of the heat solver and worth reading before touching it.

The implicit update solves `(I − θ·dt·L)u = b`. But `L` with declared boundary
conditions is **affine**, not linear: a Dirichlet ghost cell is `u_ghost = 2V − u_edge`,
and that `2V` is a constant. Conjugate gradient requires a linear operator.

So the affine part is measured and moved to the right-hand side:

1. `c = L(0)` — apply the operator to a *zero* field carrying the real boundary
   conditions. What comes back is the constant.
2. `b = u^n + dt·[(1−θ)·L(u^n) + θ·c + S]`.
3. Iterate against `L_hom` alone, which is `L` with every prescribed constant set to
   zero.

`HaloMode::Inhomogeneous` and `HaloMode::Homogeneous` make the choice explicit at
every call site, because getting it wrong is quiet: the solve converges, the answer
looks plausible, and the boundary values are wrong. The `dirichlet_slab_linear_profile`
validation case is what catches it — a half-cell error in the halo shows up as a
constant offset from the analytic straight line.

## Why the diffusion operator is finite-volume

The stencil is assembled from face fluxes rather than by differencing `u` twice:

```
L[u]_ij = [ D_{i+½,j}(u_{i+1,j} − u_ij) − D_{i−½,j}(u_ij − u_{i−1,j}) ] / dx² + (y terms)
```

Whatever leaves one cell through a face enters its neighbour through the *same* face
with the *same* coefficient, so interior fluxes cancel exactly in the sum over all
cells. `∫u dA` is therefore conserved to round-off — the benchmark measures 4.4e-16 on
a 512×512 grid — rather than merely to truncation error. For uniform `D` this reduces
to the familiar 5-point stencil, so nothing is lost in the common case.

Face diffusivities use the **harmonic** mean `2D₁D₂/(D₁+D₂)`, which is the exact
effective conductivity of two cells in series. The arithmetic mean over-predicts
transport across a sharp contrast, which is a well-known way to get plausible-looking
but wrong answers at material interfaces. The `two_material_series_conduction` case
measures the gradient ratio across a 4:1 conductivity step and gets exactly 4.

## Pair forces and momentum

`CellList::for_each_pair` visits each distinct pair once and hands the force law
`(i, j, dx, dy, r²)` with the minimum-image convention already applied. Force laws
then apply the force as equal and opposite by construction:

```rust
force_x[i] -= fx;   force_x[j] += fx;
```

Total momentum is therefore conserved to floating-point round-off regardless of how
chaotic the trajectory becomes — the benchmark measures a relative drift of 7e-17 over
200 steps. Applying forces independently to each particle would lose this, and no
amount of integrator accuracy would recover it.

One wrinkle: on a periodic axis with fewer than three cells, several of the nine 3×3
offsets alias the *same* cell. Neighbour cells are deduplicated per cell at
construction, or each pair would be reported three times.

## An invariant is a claim, not a label

`ObservationKind::Invariant` asserts that a quantity should not change, and everything
downstream believes it: the viewer's conservation panel flags any drift, the run report
calls it out. So publishing something as an invariant that is not one does not merely
mislabel it — it fires an alarm on every correct run, and an alarm that always fires is
one a reader learns to ignore. That is a worse outcome than saying nothing.

Two versions of this mistake shipped and had to be fixed:

**Kinetic energy.** Both the particle and rigid modules published `kinetic_energy` and
`potential_energy` as invariants. Neither is conserved on its own in any system where
anything happens — a Lennard-Jones gas melting out of a lattice converts one into the
other *on purpose*. Only the sum is an invariant. The halves are metrics now.

**Momentum in an open world.** The rigid module declared momentum conserved, which is
true of the impulse solve and false of almost any scene. Gravity injects momentum every
step. A static body is an infinite sink for it: a ball bouncing off the ground changes
the system's total by twice its own. `RigidDomain::observe` therefore publishes momentum
as an invariant *only* when `is_closed()` — no gravity, no static bodies — and as a
metric otherwise. The number and its unit are identical either way; what changes is
whether anything claims it should hold still. When there is nothing to claim, the panel
says so in as many words rather than showing an empty heading.

The rule this leaves behind, written on `ObservationKind` where the next domain will
read it: **publish everything, claim only what you can defend.**

A corollary for the viewer: it used to decide "should this be conserved?" by matching
observation names against a list — `momentum_x`, `total_energy`, and so on. That guess
cannot tell a closed world from an open one, because they publish the same names. It
now reads the kind the domain declared, which is the domain's own promise rather than
the viewer's assumption.

## Rigid bodies: what is exact and what is not

The contact solver is projected Gauss–Seidel on the velocity constraints — sequential
impulses. Three things follow, and `crates/lattice-domain-rigid2d/src/lib.rs` states
all three rather than leaving them to be discovered:

**Linear momentum is exact at any iteration count.** Every impulse goes through one
function that applies it equal and opposite to the pair, so a starved solve looks like
bodies sinking into each other and never like momentum appearing from nowhere. A
validation case runs the solver at *one* iteration and asserts a relative drift below
1e-12. It is the same argument the particle module's pair forces rest on.

**Angular momentum is exact under the impulses and not under position correction.**
Displacing a body without changing its velocity changes its orbital angular momentum
`m(r × v)` about any fixed origin, by `dt·(J_p × Δv)` per correction impulse. No
position-correction scheme avoids this. Rather than describe the amount as small, the
solver measures it every step and publishes it as `correction_drift`.

**Energy is not conserved and is not claimed to be.** Restitution below 1 removes it on
purpose; Coulomb friction removes it as heat that nothing here accounts for. The one
configuration where nothing removes it — restitution 1, no friction — is a validation
case, and it holds to 8.000000000 J exactly.

### Why joints and contacts share one sweep

A pendulum resting against a wall is held by both a pin and a contact, and neither is
right on its own: satisfying the pin pushes the bob into the wall, satisfying the
contact swings it off the pin. Solving one set to convergence and then the other simply
alternates between the two answers. `RigidDomain::advance` interleaves them in a single
Gauss–Seidel sweep, which is what lets them negotiate.

### Why the pin joint inverts a 2×2

A pin is two coupled scalar constraints, and the coupling is not small: a body pinned
far from its centre of mass presents wildly different resistance along the two axes.
Solving them as independent scalars converges at a rate set by that ratio — which for a
pendulum bob means it does not converge at all in any sane iteration count. The first
implementation did exactly that and reported a period eight times too fast. Inverting
the 2×2 effective-mass matrix solves both exactly in one go.

The same joint also applied its equal-and-opposite impulses at each body's *own* anchor.
Those are different points whenever the joint is violated, and two opposite impulses at
two different points are a couple — angular momentum created from nothing. Both impulses
now act at the anchors' midpoint, which is exact when the joint is satisfied and honest
when it is not.

## Chemistry and coupling: where the units meet

The single most opaque thing in a coupled model is the number that converts one
domain's units into another's. A reaction publishes heat in `W/m²`; a heat solver
consumes a source in `K/s`. Wire them straight together and the run compiles, executes,
looks entirely plausible, and is wrong by an areal heat capacity — four hundred
thousand, in `examples/chamber.lattice`.

Three decisions follow from that.

**The conversion is a material property, declared where the material is.** The model
writes `heat_capacity: 4.0e5 joule / (meter^2 kelvin);` on the temperature field. The
heat solver never reads it. The coupling edge does, and the compiler checks by
dimensional algebra that it is the right *kind* of quantity to bridge the two ports.
A model with no way to convert gets a message naming both units and what is missing.

**The ledger records the source side.** After the mapping the value is in the target's
units, and integrating that gives a number off from the energy by exactly the factor
the mapping applied. Getting this wrong was the first thing the coupling tests caught.

**The books do not balance exactly, and the residual is meaningful.** A staggered
coupling always has one exchange in flight: the heat released during the final step is
recorded and never delivered. `crates/lattice-coupling/tests/exchange.rs` asserts that
the shortfall is at most one transfer *and* that it is first order in `dt`, which turns
"nearly balances" into a statement about why.

### Why the reaction module measures its own stiffness

Spec §12.3 says stiff systems belong to *"an established numerical library rather than
a weak custom method"*. That is a constraint on what this module may pretend to be. A
half-good BDF would work on the test cases and fail quietly on a real network.

So the reaction integrator is explicit, sub-cycles to the network's own Jacobian —
bounded by both stability and accuracy — and publishes what it found. `exhausted` says
the sub-step budget ran out; `stiffness` says whether stiffness was why. They are
separate fields because they need different fixes: a budget can run out because a
network is stiff, or simply because the interval asked for is enormous relative to the
timescale.

### Energy is a state variable, not a sample

The heat a reaction releases was first computed by sampling the instantaneous power at
the start of a step. For a reaction that half-finishes during that step, that is wrong
by a factor of `e`. Energy is now carried through the same Runge–Kutta stages as the
concentrations, with the same weights, which makes it fourth-order accurate and costs
nothing — the heat rate falls out of rate evaluations the stages already perform.

It is the number the coupling ledger balances, which is why it is worth being exact
about.

## Diagnostics and the coupling ledger

`ConservationLedger` implements §14.3. The distinction it draws is between *transfer*
and *drift*:

- Energy leaving a reaction and arriving in a heat field is a transfer. It should net
  to zero, and the ledger records both ends.
- Whatever fails to net out is a numerical problem.

`Reconciliation` compares a domain's observed change against what the ledger says it
received, so a coupled run's energy imbalance becomes an attributable number rather
than an unexplained one. Nothing couples yet — this is M3 machinery, built now because
the diagnostics have to exist before the coupling does, not after.

## Reproducibility

`RunArtifact::content_hash` covers the physics — configuration, solver contracts,
seeds, and the observation timeline — and deliberately **excludes** wall-clock times,
memory figures, and the creation timestamp. Two runs of the same model on the same
build produce the same hash even though one took longer, which is what makes the hash
usable as the regression signal FR-011 asks for. Including timings would make every run
unique and the hash worthless.

The RNG is a hand-written PCG32 with pinned test vectors. `rand` promises
reproducibility within a major version; a checkpoint recorded today must replay in five
years.

**The thread count is not part of the physics.** A run's execution schedule is recorded
in the artifact's `execution` *section*, which is outside the content hash, so a model
run on twenty threads hashes identically to the same model run on one. §19.3 wants the
backend published with every result and FR-011 wants the same model to hash the same;
both hold at once only because parallel execution is built to change the schedule and
never the numbers. That is a promise with teeth and a price — see
[execution.md](execution.md) for what it costs and what stays sequential to keep it.

## Where the executor lives, and why it is not in the solver

A solver receives an `Executor` through `StepContext` rather than owning one. The
alternative — each domain constructing its own pool — is worse in two ways that only
show up in a coupled model: it starts a thread pool per domain, all competing for the
same cores, and it leaves a caller no way to ask the whole simulation to run on one
thread. How many threads to use is a property of the *run*, not of the physics, so it
travels with the clock and the arena.

The partitioning rule sits with the executor and the *grain* sits with the kernel. That
split matters: the executor knows how many threads there are, and only the kernel knows
how much work a unit is. A `Grain` is a solver's statement about its own cost, measured
and recorded next to the constant, in the same spirit as a `SolverContract` being a
solver's statement about its own error.

## How complete the §7.2 split is

Spec §7.2 wants each solver separated into an immutable schema and a mutable state
block. `CompiledModel` is the immutable half at the *model* level: domains, buffer
plan, operation graph, observers, notes. But a `HeatDomain` still owns both its
configuration and its field values, so the runtime instantiates domains from a compiled
model and keeps them alongside it.

That is stated rather than papered over. The architectural line is drawn — the runtime
never reaches into a solver's internals, and everything it needs to schedule comes from
the model — and solvers cross it one at a time.

## Deliberate omissions

- **No parallelism.** Everything is single-threaded scalar CPU. §15.3 wants parallel
  iterators and SIMD; §15.1 says *"optimize after validation"*. The operation graph
  already computes which operations are independent and reports the ideal speedup, so
  the information is there when the execution is.
- **No coupling.** `couple a.b -> c.d` parses, type-checks as far as it can, and then
  errors with "milestone M3". The conservation ledger that will account for those
  transfers is already built and tested — diagnostics have to exist before the thing
  they diagnose, not after.
- **No user-defined expressions.** Spec §8.3's expression language, which compiles
  custom force and rate laws to CPU and GPU kernels, is M6. Until then the builtin
  vocabulary is a closed set, and an unrecognized function is an error that *lists what
  is available*.
- **No GPU.** M4.

Each of these is a case of spec §21.1: *"New domains should not be added while the
current domain lacks a reference test or cannot explain its stability limits."*
