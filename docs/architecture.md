# Architecture

How the code maps onto the [specification](spec/project-lattice-spec-v0.1.md), and why
the non-obvious decisions were made that way.

## Layering

```
                    ┌─────────────────────────────────────┐
   authoring        │  lattice-cli   (validate/bench/demo) │
                    └──────────────────┬──────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐          ┌──────────▼──────────┐        ┌──────────▼────────┐
│ lattice-       │          │ lattice-domain-     │        │ lattice-observe   │
│ validation     │          │ particle, -grid2d   │        │ (JSON, timing,    │
│ (the lab)      │          │ (the physics)       │        │  run artifacts)   │
└───────┬────────┘          └──────────┬──────────┘        └──────────┬────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  lattice-ir             │
                          │  IDs, SoA storage,      │
                          │  grids, arenas,         │
                          │  Domain contract,       │
                          │  diagnostics            │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  lattice-units          │
                          │  dimensions, quantities │
                          └─────────────────────────┘
```

Dependencies point downward only. `lattice-ir` holds no physics; the domain crates
hold no storage layout decisions.

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

## Deliberate omissions at M0

- **No model file format.** Models are built through the Rust API. The `.lattice` DSL
  of spec §25 is M1, and building it before the runtime it targets exists would be
  designing against a guess.
- **No operation graph.** Spec §8 makes the IR the central novel component, and it is —
  but a scheduler with nothing to schedule is speculative. It arrives with the compiler
  that produces it.
- **No parallelism.** Everything is single-threaded scalar CPU. §15.3 wants parallel
  iterators and SIMD; §15.1 says *"optimize after validation"*, and the validation
  suite is what makes an optimization safe to attempt.
- **No GPU.** M4.

Each of these is a case of spec §21.1: *"New domains should not be added while the
current domain lacks a reference test or cannot explain its stability limits."*
