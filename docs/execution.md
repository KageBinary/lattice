# Execution

How a model's work is divided, and what the division is allowed to change.

Spec §15.3 asks for *"explicit parallel iterators/task pools for independent
operations."* This is the first half of M4 — the CPU half. The GPU backend §15.4
describes is not built yet, and several decisions below exist to make its arrival
legible rather than to anticipate it.

## The promise

**Parallel execution changes the schedule, never the numbers.**

The same model on eight threads produces the same `f64` bit patterns, and therefore the
same artifact content hash (FR-011), as the same model on one. Not "agrees to
1e-12" — the same bits.

```console
$ lattice run examples/diffusing_pulse.lattice --threads 1 --json a.json
$ lattice run examples/diffusing_pulse.lattice --threads auto --json b.json
```

`a.json` and `b.json` differ in their `execution` section and their timings. Their
`content_hash` is identical.

Four validation cases hold this in place, at `lattice validate --filter cpu`:

| Case | What it compares |
|---|---|
| `heat_field_matches_across_thread_counts` | every cell of a variable-diffusivity field after 200 explicit steps |
| `implicit_solve_matches_across_thread_counts` | the field *and* the conjugate-gradient iteration count after 40 Crank–Nicolson steps |
| `particle_trajectories_match_across_thread_counts` | position and velocity for 3,000 particles under three integrators |
| `run_artifacts_hash_identically_across_thread_counts` | the FR-011 reproducibility hash, end to end |

The implicit case is the sharp one. CG stops on `‖r‖ ≤ tol·‖b‖`, so a difference of a
single ULP anywhere in the stencil eventually lands on the wrong side of that threshold
on some iteration, and the two runs take different numbers of iterations from then on.
The iteration count is a discrete amplifier for a difference too small to see in the
field itself.

### Why the promise is worth its cost

§19.1 only asks that backends *"agree within tolerance"*. Insisting on exact agreement
between the two CPU paths costs real performance — see [what stays
sequential](#what-stays-sequential) — and buys two things.

A regression baseline recorded on a sixteen-core build machine is comparable with a run
on a four-core laptop. Without that, the content hash is not a regression signal, it is
a fingerprint of the hardware.

And when the GPU backend arrives with a genuine tolerance, that tolerance will be
*attributable*. A GPU differs in FMA contraction, transcendental accuracy, and reduction
order, and its disagreement can be traced to those. If the CPU pair had already been
tolerance-based, nobody could later tell which part of the budget belonged to which
cause.

## How it holds

Two facts, and everything else follows.

**The partition does not depend on the threads.** `Partition::new(units, chunks_wanted,
grain)` is a pure function of the work size, the *configured* thread count, and the
caller's declared grain. It never asks which worker is free. A kernel written against it
sees the same chunk boundaries whether eight threads are racing or one thread is walking
them in order — and the sequential path deliberately walks the same partition rather
than taking the whole slice as one chunk, which would be faster and would break this.

**Every parallel kernel is elementwise or a stencil.** Output cell *k* is computed by one
expression over inputs that no other chunk writes. Partitioning cannot reassociate
anything, because there is nothing to reassociate.

The parenthesisation is load-bearing and is written out at each call site rather than
folded into a shared helper:

```rust
// Semi-implicit Euler                      Velocity Verlet
*v += dt * (inv_mass[i] * force_x[i]);      *v += half * inv_mass[i] * force_x[i];
//         ^ dt·(m⁻¹F)                      //  ^ (half·m⁻¹)·F
```

These are different numbers. A generic `axpy` that normalised them would have been
tidier and would have silently changed both schemes' results.

## What stays sequential

Three things, each for a stated reason. None of them is an oversight, and each is a
decision the GPU backend will have to make differently.

**Reductions.** A sum split into chunks and added back is not the sum added in order.
The difference is round-off, but round-off that would vary with the core count of
whichever machine ran it. §15.3's reduction trees need a partition fixed *independently
of the hardware* before they can be added without giving up the promise. Until then
`sum_interior`, `max_abs_interior` and friends run on one thread.

**Conjugate gradient's inner products.** The same reason, with teeth: the residual
decides when the iteration stops, so a round-off change there changes the iteration
count and through it the answer. CG's *stencil applications* are parallel — they are the
expensive part — and only its dot products are not.

**Lennard-Jones pair forces.** A pair law applies Newton's third law by scattering into
both particles of a pair, which is what makes momentum conserved to round-off. Two
threads working on different cells can therefore collide on a shared neighbour.
Reformulating it as a gather — each particle summing over its own neighbours, visiting
each pair twice — would fix the collision and would also change the summation order.
That is the one thing this design does not do.

The honest summary: the MD workload of §15.6 gets its integrator parallelised and its
force loop not, and the force loop is where its time goes. A parallel pair loop is a
real piece of work with a real decision in it, and it belongs with the GPU backend,
where the tolerance question is being answered anyway.

## What is parallel

| Kernel | Split by | Grain |
|---|---|---|
| `DiffusionOperator::apply_with` | rows of the output field | `BAND_GRAIN` |
| explicit heat update | rows | `BAND_GRAIN` |
| Crank–Nicolson right-hand side and `I − θ·dt·L` | rows | `BAND_GRAIN` |
| all three integrators' per-particle updates | particles | `PARTICLE_GRAIN` |

Rows are the natural split for a stencil. Cell `(i, j)` reads rows `j−1`, `j`, `j+1` of
the *input* and writes only row `j` of the *output*, so bands of rows write disjoint
memory while reading freely across the boundary between them. Nothing is copied and no
band needs its neighbour's output — which is exactly why `ScalarField::row_span_mut`
exists.

## The grain, and why it has two numbers

`Grain { floor, chunk }` is where a kernel declares how much work is worth a barrier.
Both numbers were measured, and the reason there are two of them was measured too.

**`floor` — is this worth splitting at all?** A dispatch has to reach every worker and
hear back from every worker. That costs a few microseconds whatever is inside it, and a
velocity-Verlet step contains six of them. Lifting the particle floor and measuring what
happens below it:

| particles | 16k | 33k | 65k | 131k | 262k | 524k |
|---|---|---|---|---|---|---|
| speedup | 0.42x | 0.65x | **1.19x** | 1.55x | 1.90x | 1.92x |

A 16k-particle model split across twenty threads runs **2.4× slower** than on one. That
is not a disappointing optimization, it is a regression — §15.1 draws no distinction
between a faster wrong answer and a slower right one — so the floor sits at the
crossover and smaller models take exactly the path they took before M4.

**`chunk` — how small may a piece get?** This is not the same number, and using one for
both is a trap that costs real performance. A floor high enough to keep 16k particles
sequential, used as a minimum chunk size, also keeps a 262k-particle model on four
threads: measured, that dropped it from 1.90× to 1.34×.

Grid grains are stated in *cells* and converted with `Grain::per_row`, because a row is
not a fixed amount of work — ten rows of a 512-wide grid and ten of a 32-wide one differ
by a factor of sixteen.

Two consequences worth knowing. A small interactive scene is never split, so `--threads
auto` cannot make it slower. And a validation case has to be sized *above* the floor or
it compares the sequential path with itself and passes for the wrong reason — which is
why the cross-backend cases pin their problem sizes to the published grains and assert
that they actually split.

## The pool

`lattice-cpu` is two layers. `ThreadPool` owns worker threads and hands out chunk
indices. `Executor` decides how a loop is split. Everything at the `Executor` layer and
above is safe code.

`ThreadPool` contains the only `unsafe` in the workspace, resting on two invariants that
`dispatch` enforces rather than documents:

1. **The closure pointer outlives every use.** `dispatch` publishes a type-erased pointer
   to the caller's work, and does not return until `finished == total` and the job slot
   has been cleared under the lock. A panicking task still counts as finished; its
   payload is re-raised on the calling thread afterwards.
2. **Each task index is claimed exactly once.** Claiming increments a counter under the
   state lock. That is what lets chunk *k* be produced from index *k* as a `&mut`
   sub-slice with no aliasing.

Threads are persistent. `std::thread::scope` would give the same guarantees with no
`unsafe` and was the first implementation; spawning and joining on every dispatch costs
tens of microseconds, and a 256² diffusion step is around a hundred, so per-call spawning
turned the intended speedup into a slowdown at exactly the sizes §15.6's interactive
targets care about.

Two further latency costs showed up in measurement rather than in design, and both are
addressed in the pool rather than papered over with a larger grain:

**Workers spin before parking.** A solver's next dispatch is usually only hundreds of
nanoseconds away — six passes in one velocity-Verlet step — and a condition-variable
wakeup costs far more than that gap. Workers poll a lock-free generation counter for a
short while before parking, which turns the common case into a few hundred nanoseconds
of polling. Past the spin budget they yield, and past that they park, because an idle
pool must not keep a machine's fans on.

**Claiming is a compare-exchange, not a lock.** Twenty threads contending on one mutex,
twice per task, is its own bottleneck. The claim counter is global, monotonic, and never
reset, with each job owning a half-open range carved from it. That is not a
micro-optimization but a correctness requirement: with a per-dispatch counter reset to
zero, a worker finishing the last task of dispatch *N* probes once more for work, and if
*N+1* has already published, that probe either runs an *N+1* task against *N*'s freed
closure pointer or silently swallows an index *N+1* is waiting for. A load-then-compare-
exchange lands past the worker's own range and consumes nothing.

Together these moved the particle crossover from about 150k to 65k.

Two footguns are closed rather than documented. A task that asks the pool for more work
runs inline instead of deadlocking on the driver lock. A second thread that drives the
pool concurrently waits rather than corrupting the single job slot.

## What it bought

Best of three release runs on a 20-thread desktop. §19.3 asks for the hardware, backend
and model to be published with any number, and `lattice bench` prints all three; these
are reproduced here as a rough expectation, not a promise.

| Benchmark | Size | Speedup |
|---|---|---|
| `heat-explicit` | 256² | 3.4x |
| `heat-explicit` | 512² | 4.8x |
| `heat-explicit` | 1024² | 4.7x |
| `heat-crank-nicolson` | 256² | 1.6x |
| `particles-gravity` | 262k | 1.9x |
| `particles-lj` | 1k | 1.0x |
| `quantum2d` domain advance | 512² | 5.6x |
| `quantum2d` domain advance | 768×512 | 5.7x |

The two `quantum2d` rows are not from `lattice bench` and are not best of three: they
are the `advance` column of
`cargo run --release -p lattice-domain-quantum2d --example step_profile`, median of three
runs on 2026-10-08 with no game running. That is 9.28 ms on one thread against 1.67 ms on
the pool at 512² (about 600 steps a second), and 46.9 ms against 8.17 ms at §25.2's
768×512, whose rows need Bluestein's transform. Every run reported the pool's
wavefunction bit-identical to the single thread's. The transforms, transposes and phase
multiplies all run on the pool; what stays on the calling thread is the absorbed
probability, a sum over the grid folded twice a step in a fixed order so its bits do not
depend on the machine, and the domain's bookkeeping around the step. Which of those
bounds the 5.6x has not been measured.

Three things in that table are worth reading rather than skimming.

**Nothing regresses.** `particles-lj` is 1.0x because its pair loop is sequential and its
population is under the integrator's floor — so `--threads auto` costs it nothing. That
is the design working, not an omission.

**Crank–Nicolson gains least**, and it is the scheme doing the most work. Its stencils
are parallel and its inner products are not, so Amdahl's law caps it. That is the price
of the exactness promise, paid where it is visible.

**Five times on twenty threads is not a disappointment.** The stencil is six loads and a
handful of flops per cell; it saturates memory bandwidth long before it saturates cores.
The flattening between 512² and 1024² is the working set outgrowing cache. Efficiency —
speedup per thread, which `--compare` prints — is the number that says which wall you
have hit.

## Measuring it

```console
$ cargo build --release
$ ./target/release/lattice bench heat-explicit --threads auto --compare
```

`--compare` runs the benchmark sequentially first and then at the requested thread count.
The baseline runs first deliberately, so the parallel run has the warmer cache and the
ratio is biased *down*.

Timings here are noisy at the ±20% level, and the first version of this document quoted a
0.94x that turned out to be nothing at all. Take the best of three before believing a
number, especially near a grain's floor, where the effect being measured is smallest.

Both runs are checked against their §15.6 correctness conditions, and a speedup is
marked invalid rather than reported if either one failed. §15.1: *"a faster wrong solver
is a regression."*

## Choosing a thread count

The default is one. That is deliberate and is the opposite of what most tools do: §24.1
calls the scalar CPU path *"the executable specification for accelerated kernels"*, and a
run that has not asked for anything else should be running the specification.

- `--threads auto` sizes to the machine.
- `--threads <n>` pins it, which is what a benchmark should do.
- `Simulation::with_executor` is the same choice from Rust; `StepContext::with_executor`
  is the same choice for a solver driven directly.

The choice is recorded in the run artifact's `execution` section — as a *section*, not a
parameter, which puts it outside the content hash. §19.3 wants the backend published with
every result and FR-011 wants the same model to hash the same. Both hold at once only
because how the work was divided is not part of the physics.

## What M4 still owes

This document covers item 4 of the four M4 needs in [the roadmap](roadmap.md). The three
things named here as needing revisiting when a GPU arrived are now answered in
[backends.md](backends.md), and the answers are recorded rather than deleted because two of
them came out differently than expected:

- **The cross-backend cases assert exact equality; the GPU ones do not.** Both kinds now
  exist side by side, and they read as different kinds of claim: the cases above compare
  IEEE bit patterns and carry no tolerance at all, while the GPU rows carry a budget
  decomposed into named mechanisms with their derivations. The exactness promise above is
  what makes that budget attributable — every unit of it belongs to the GPU.
- **The pair-force loop and the reductions are still live, and still deferred.** The first
  GPU kernel is a stencil precisely because a stencil has no reduction in it. Crank–Nicolson
  is not on the device for the same reason: CG's inner products are where reduction order
  stops being a zero term, and a GPU has no sequential fallback to retreat to.
- **`Executor` stayed a CPU concept, and that turned out to be right.** `lattice-compute`'s
  `Device` is not `Executor` with more implementations — there is no `&mut [f64]` to hand
  out when the data is in device memory, and no loop to split when the kernel *is* the loop
  body. The two coexist; `CpuDevice` uses an `Executor` internally.
  `OperationGraph::levels` is still unconsumed.

One thing this document did not anticipate. It assumed the GPU's disagreement would be
spent on FMA contraction, transcendental accuracy and reduction order. Those are real, and
the first is present — but WGSL has no `f64` at all, so the portable backend runs `fast32`
against an `accurate64` reference, and *state rounding* dominates every budget by roughly
eight orders of magnitude. The tolerance question turned out to be a precision question.
