# Backends

## Current M4 capabilities

The backend runs explicit insulated diffusion, Crank–Nicolson/backward Euler with
mixed insulated and Dirichlet faces, and velocity-Verlet gravity/Lennard–Jones
particles. Its pipeline cache retains matching layouts as well as shaders.
`GpuDevice::from_shared` uses an application's existing device and queue; field
presentation through `GpuFieldImage` has no CPU pixel path. See the
[engineering report](m4-engineering-report.md) for measurements and limitations.

The discussion below records the M4.2–M4.4 baseline. Statements about unbuilt
Dirichlet, particle, graph scheduling and rendering paths describe that historical
baseline, not the current implementation.

Who runs the arithmetic, at what precision, and what the answer is allowed to differ by.

[execution.md](execution.md) covers how one backend divides work across threads. This
covers the boundary between backends, and the number that separates them.

## The one fact that reorders the milestone

**WGSL has no `f64`, so the portable GPU backend cannot run the reference precision.**

§15.4 makes the portable WebGPU backend *"the product baseline"*. §10.5 makes `accurate64`
the mode validated runs use. §24.1 makes the scalar CPU path *"the executable specification
for accelerated kernels"*. Those three sentences are individually reasonable and jointly
impossible: the product baseline is structurally unable to execute the specification's own
precision, because the shading language it compiles has no 64-bit float type.

This is easy to get wrong in the optimistic direction. `wgpu::Features::SHADER_F64` exists,
and the RTX 4070 this was developed on reports it. Reading the feature flag alone would
produce a backend that advertises `accurate64` and then fails to compile, because that
feature enables 64-bit floats in **SPIR-V** shaders on Vulkan, while the portable path
compiles **WGSL**. `naga`'s WGSL front end will *parse* `f64`; its validator gates the type
behind a capability `wgpu` never grants. What comes back is:

```
Shader validation error: Entry point main at Compute is invalid
  ┌─ f64-probe:6:40
  │
6 │     let widened: f64 = f64(data[0]);
  │                        ^^^^^^^^^^^^ naga::ir::Expression [3]
  │
  = Expression [3] is invalid
```

Because that is a claim about a toolchain rather than about this code, it is tested rather
than asserted. `GpuDevice::probe_wgsl_f64` compiles exactly that shader, and
`wgsl_f64_is_rejected_by_the_portable_backend` **fails if it ever succeeds** — which is the
right way round, since the fix for that failure is to offer more precisions.

The consequence for §23's *"portable GPU abstractions may leave performance on the table"*:
for scientific work the portable abstraction leaves **precision** on the table, not
throughput, and §15.4's native-backend plugin boundary is what a run needing `accurate64`
on a GPU would eventually go through.

## The budget, and why it is a structure

A cross-backend tolerance is the one number in a validation suite that nothing checks. Too
tight and it fails visibly; too loose and it passes forever while hiding every defect
smaller than itself. There is no test that fails when a tolerance is too generous.

So `Tolerance` is not a float. It is a list of named `Mechanism`s, each carrying the
derivation that produced its share, and its total is the sum of its parts:

```
relative budget 1.669e-4, spent on:
  state rounding    1.192e-4  200 steps x 5 roundings/step at fast32 epsilon (1.19e-7),
                              accumulated additively because an explicit diffusion step
                              below its stability limit does not amplify existing error
  FMA contraction   4.768e-5  4 contractable multiply-adds per element per step over
                              200 steps, each saving at most half an ulp
```

The four mechanisms are closed on purpose. A disagreement that is not one of them is not a
tolerance question — it is a bug, and widening a budget to cover it is the failure the type
exists to prevent.

| Mechanism | Where it comes from | Present today |
|---|---|---|
| state rounding | the state is stored narrower than the reference | dominant |
| FMA contraction | `a*b + c` rounded once instead of twice | yes |
| reduction order | a sum accumulated in a different association | zero — nothing reduces on the device yet |
| transcendental accuracy | `sin`/`exp` accurate to a different ulp count | zero — the stencil only adds and multiplies |

### How the state-rounding term is derived

Three facts, and the number falls out.

**Each step rounds a few times.** The stencil is four products and five sums per cell, then
one multiply-add for the update. Ten roundings at half an ulp each is `5·ε` of fresh
relative error per step.

**The scheme does not amplify what it already has.** An explicit diffusion step below its
stability limit has every eigenvalue of its update operator inside the unit circle — that
is what the stability limit *means* — so a perturbation present at step *k* is no larger at
step *k+1*. Errors accumulate additively rather than geometrically, which is the difference
between a usable bound and a meaningless one.

**Additive accumulation over `n` steps is `n` times the per-step figure.** That is the worst
case, in which every step's round-off pushes the same way. Real round-off behaves like a
random walk and lands near `√n`.

Deliberately pessimistic, and deliberately derived *before* the measurement. A bound fitted
to an observation cannot detect the observation getting worse.

## What was measured

`lattice validate --filter gpu`, on an RTX 4070 Laptop GPU through Vulkan, 200 explicit
steps of variable-diffusivity diffusion on a 192×128 grid:

| Case | Measured | Bound |
|---|---|---|
| `gpu_diffusion_matches_the_cpu_reference` | **0.3%** of the derived budget | ≤ 100% |
| `gpu_disagreement_is_attributable_to_f32_storage` | **4.3** `f32` ulps | ≤ 1000 |
| `gpu_and_cpu_agree_exactly_where_no_rounding_occurs` | **0** cells differ | 0 |

Three things in that table are worth reading rather than skimming.

**0.3% is the number to watch, not 5.129e-7.** A case sitting at a few percent of its
budget says the bound is loose and honest. One climbing toward 100% across releases says
something is degrading while still passing, and no pass/fail line would have caught it.

**4.3 `f32` ulps is also 2.3×10⁹ `f64` ulps**, and reporting both is what makes the
attribution a measurement rather than an argument. A transposed index would blow the first
bound; a backend secretly running at a different precision than it claims would fail the
second. Together they say the disagreement is precision and nothing else.

**The third case is the control.** On a field with no gradient every flux term is
`coefficient × (u − u)`, which is exactly zero in any precision, so the update adds exactly
zero and the two backends agree *to the bit despite running at different precisions*. A
suite where the first two cases passed and this one did not would be a suite whose budget
was covering a real defect.

## The implicit path, and the mechanism it added

M4.4 put Crank–Nicolson on the device: a linear system per step, solved by conjugate
gradient, with the dot products running as a device reduction. It changed the shape of the
budget more than it changed the code.

### An `f32` solve cannot be asked for `1e-10`

`HeatDomain` defaults to a relative residual tolerance of `1e-10`. That is an unremarkable
ask of an `f64` solver. On the portable backend it is not merely slow — it is impossible,
and the reason is short enough to check.

Conjugate gradient stops on a residual it maintains recursively, but what a caller is
promised is a statement about the true one, `b − Ax`. Even given a *perfectly exact* `x`,
forming that difference at unit round-off `ε` leaves an error of about `ε·(‖b‖ + ‖A‖·‖x‖)`.
For `A = I − θ·dt·L` with `L` negative semi-definite every eigenvalue of `A` is at least 1,
so `‖A⁻¹‖₂ ≤ 1` and therefore `‖x‖ ≤ ‖b‖`. Dividing through:

```text
floor = ε · (1 + ‖A‖₂)
```

`‖A‖₂` is bounded by Gershgorin over the assembled rows, so the whole number is computable
at setup from the face coefficients and `θ·dt` — nothing fitted, nothing measured. On the
validation problem at twenty times the explicit stability limit it is **2.6×10⁻⁶**, about
26 000 times what the CPU asks for.

So `GpuCrankNicolson::new` **refuses**, and names the floor and the `‖A‖` it came from. The
failure being prevented is not a wrong answer: a solver that accepted `1e-10` would iterate
to its cap every step, report `MaxIterations`, and look exactly like a slow GPU. Refusing at
setup is the difference between a numerical fact and a performance mystery.

### The budget is dominated by a mechanism that did not exist before

Two conjugate-gradient solves of the same system, each stopping when its own criterion is
met, return two different correct answers. `Mechanism::SolveTermination` is that, and it
scales with `τ` rather than with `ε`.

Everything else reaches the answer through the same channel — the residual — and
`‖A⁻¹‖₂ ≤ 1` carries a residual bound to a solution bound with no amplification at all.
That is what makes the decomposition possible:

| Mechanism | Per step | Measured value |
|---|---|---|
| solve termination | `(τ_ref + τ_gpu)·‖b‖₂` | **1.45×10⁻³** |
| state rounding | `ε·(1 + ‖A‖₂)·‖b‖₂` | 1.45×10⁻⁴ |
| reduction order | `τ_gpu·(d·ε/2)·‖b‖₂` | 1.38×10⁻⁹ |

**The reduction is the smallest term by six orders of magnitude**, and it is the term M4.2
deferred this entire piece of work for. The reason is structural rather than lucky: a
perturbed dot product changes *which iterate* CG arrives at, and the stopping test then
measures that iterate afresh. The reduction moves the path, not the destination.

That is emphatically not a general result about reductions. It holds because CG's
termination re-measures what the reduction perturbed. A reduction whose value *is* the
answer — a conserved total, an observation, a fixed-iteration solver — has no such backstop.
`gpu_reduction_order_is_the_whole_disagreement_when_nothing_re_measures_it` is that case,
and it exists so the term is audited somewhere it can be seen: a bare device dot product
against a sequential sum, at **1.5%** of a budget that is `d·ε` and nothing else.

### The reduction's order is written down, not discovered

`reduction.wgsl` fixes the association: workgroup `g` owns a contiguous block, invocation
`t` sums a contiguous run of it serially, and two fixed binary trees combine the rest.
Nothing is an atomic and nothing depends on the order workgroups finish in, so the same
input gives the same sum on every run — which `the_device_reduction_is_bit_reproducible`
checks. WGSL has no atomic add for floats, so the tempting non-deterministic shortcut is
closed at the language level rather than by discipline.

The payoff is that the accumulation depth is a *number*: `per_thread − 1 + 2·log₂(256)`. At
24 576 cells that is **16**, against **24 575** for a sequential sum of the same values. A
reduction that merely differs from the reference is a mechanism you can bound; one that is
unpredictable is not, and no amount of tolerance fixes it.

### The budget and the measurement have to be in the same norm

CG stops on `‖r‖₂`, so everything derivable from its stopping rule is a statement about a
*vector*. Checking such a bound per cell costs a factor of `√N` — 157 on this grid —
belonging entirely to the change of norm and not to the solver. A budget carrying it would
be two orders of magnitude looser than the mechanism it describes, and a budget two orders
too loose hides everything smaller than itself.

So `Tolerance` now names the norm it was derived in and `compare` measures in the same one.
The per-cell mechanisms stay in the max norm, where they were always stated. Mixing is the
thing being prevented, and until M4.4 nothing could mix because every mechanism was
per-cell.

### What the implicit rows measure

`lattice validate --filter gpu`, same machine, 50 Crank–Nicolson steps at twenty times the
explicit stability limit on a 192×128 grid:

| Case | Measured | Bound |
|---|---|---|
| `gpu_crank_nicolson_matches_the_cpu_reference` | **11.3%** of the derived budget | ≤ 100% |
| `gpu_implicit_disagreement_is_dominated_by_solve_termination` | **1.05×10⁶** separation | ≥ 5.2×10⁵ |
| `gpu_reduction_order_is_the_whole_disagreement…` | **1.5%** of the derived budget | ≤ 100% |
| `gpu_refuses_a_residual_tolerance_it_cannot_reach` | refused | must refuse |
| `gpu_and_cpu_implicit_agree_exactly_on_a_field_with_no_gradient` | **0** cells differ | 0 |

The second row deserves a note on what it does *not* measure. The ratio of solve termination
to state rounding is exactly the floor multiple the case runs at — a constant chosen in that
file — so reporting it as a finding would be reporting our own input back. The ratio that is
measured is termination over *reduction order*, whose predicted value `2/(d·ε)` depends on
the reduction's depth and the precision and on nothing chosen anywhere.

The sharpest evidence is a note rather than a metric: the measured disagreement is
**1.795×10⁻⁴** against a random-walk prediction of `√steps · (τ_ref + τ_gpu) · ‖b‖/‖x‖` =
2.049×10⁻⁴, or **0.88×**. A run landing near the walk rather than near the budget's
worst case of `steps ×` the same figure is what says the identified mechanism is the real
one.

### What it costs: a fence per iteration

The explicit solver touches host memory twice — once up, once down. An implicit one cannot,
because *stopping* is a host decision and the quantity it depends on lives on the device.
§10.3 requires that residual histories can stop a run, so the stopping test is not
negotiable and something has to come home.

What comes home is eight bytes per iteration. `α` and `β` stay on the device, computed by
one-invocation kernels reading the same scalar buffer the reductions write. Getting it to
*one* readback rather than two took a trick: the CPU checks `pᵀAp > 0` before applying the
step, and the device cannot ask the host mid-command-buffer, so the update kernels carry the
test themselves. A whole iteration is dispatched at once, and a non-positive curvature means
the updates have already declined to run — the host learns one readback later and finds `x`
and `r` exactly where the CPU would have left them.

Two measured facts about what remains:

**Fusing the copy onto the iteration's own submission was worth about 1.6×.** Appending
`copy_buffer_to_buffer` to the encoder that just recorded the iteration, and reusing one
staging buffer instead of allocating per iteration, took `heat-crank-nicolson --backend gpu`
from **467** to **~750 steps/s**. The removed cost was one submission and one fence per
iteration — a round trip to a device that was already idle.

*The 467 is a single measurement and should be re-taken.* M4.3's own lesson was that
timings lie at ±20% and that nothing should rest on one run; the figure after the change is
a median of six, the figure before it is not.

**Per iteration, the GPU is worth 1.8×, against 27× for the stencil.** On the same device,
grid and precision:

| Benchmark | CPU | GPU | Ratio |
|---|---|---|---|
| `heat-explicit`, steps/s at 256² | 3 354 | 90 705 | **27×** |
| `heat-crank-nicolson`, steps/s | 200 | ~750 | 3.8× |
| `heat-crank-nicolson`, CG iterations/s | 1 900 | ~3 430 | **1.8×** |

The steps-per-second ratio flatters the GPU, because the two backends solve to different
residual tolerances and therefore do different numbers of iterations — 9.5 per step against
4.57. Both benchmarks now publish iterations per step for exactly that reason; a step of an
implicit scheme is not a fixed amount of work, and §19.3's "publish the conditions" needs
that to be one of the conditions.

Normalizing for it leaves **1.8×**. That is what a fence per iteration costs on a kernel
whose stencil form runs 27× faster. Amdahl's law with a device queue in the serial part.

**Measurement conditions, because §19.3 asks for them.** The implicit figures are medians of
six (GPU) and three (CPU) runs on an otherwise idle machine; the `heat-explicit` row is one
run of each and its 27× should be read as an order of magnitude rather than a figure. A
later attempt to re-measure on the same machine while a game held the GPU produced GPU
numbers between 54 and 505 and CPU numbers around 85 — a reminder that on a shared consumer
device the benchmark measures the machine at least as much as the code.

## What the CPU pair bought

M4.1 held the scalar and parallel CPU paths to *bit-identical* agreement, at a real cost in
performance — Crank–Nicolson gains only 1.6× because its inner products stay sequential.
This is what that bought.

Every unit of budget above belongs to the GPU and can be pointed at. Had the CPU pair been
tolerance-based, there would now be a single number covering thread scheduling and `f32`
storage together, and nobody could say which part belonged to which cause. The two levels
of claim stay visibly different kinds of statement rather than one weakened into the other:
`execution.md`'s cases assert equality of IEEE bit patterns and carry no tolerance at all.

## What it costs, measured

`lattice bench heat-explicit --backend gpu`, 400 explicit steps, best of three runs in
separate processes on an RTX 4070 Laptop GPU through Vulkan and a 20-thread desktop CPU.
§19.3 wants the hardware, backend and precision beside any number, and `lattice bench`
prints all three.

| Grid | cpu-scalar (f64) | cpu, 20 threads (f64) | wgpu (f32) | vs scalar | vs 20 threads |
|---|---|---|---|---|---|
| 256² | 102.8 ms | 37.5 ms | **3.87 ms** | 26.6× | 9.7× |
| 512² | 403.3 ms | 95.5 ms | **5.86 ms** | 68.8× | 16.3× |
| 1024² | 1971 ms | 424.3 ms | **15.1 ms** | 130.3× | 28.0× |

**Half of that ratio is precision, not hardware.** The GPU stores `f32` and the CPU `f64`,
and both are bandwidth-bound, so the GPU moves half the bytes per cell before any
architectural advantage applies. `lattice bench --compare` prints a note saying exactly
this whenever the two sides ran at different precisions, because a speedup between two
different answers is not a speedup.

At 1024² the kernel does 27.7 G element-updates/s, which at a minimum of 8 bytes of DRAM
traffic per cell per step is about 222 GB/s against the adapter's ~256 GB/s. That is the
wall, and it is the same wall §15.6's CPU numbers hit — [execution.md](execution.md)'s
"five times on twenty threads is not a disappointment" applies here for the same reason.

### The number that matters more

§15.1: *"Measure end-to-end: include model compile time, upload/download, solver work […]
not kernel time alone."* Total process wall time, best of three:

| Grid | CPU process | GPU process | |
|---|---|---|---|
| 256² | **235 ms** | 958 ms | GPU 4.1× *slower* |
| 512² | **569 ms** | 974 ms | GPU 1.7× *slower* |
| 1024² | 2269 ms | **1192 ms** | GPU 1.9× faster |

Opening a device costs **0.72–0.82 s** on this machine — enumerating adapters, creating a
device, and forcing the driver's lazy initialization — and it is paid once per process
whatever the grid size. For a 400-step benchmark that dominates everything: the GPU wins
the kernel by two orders of magnitude and loses the *program* at 256² and 512².

Neither number is the honest one on its own, so both are printed. A real simulation runs
far more than 400 steps, and the break-even follows directly: at 1024² the GPU saves about
1.96 s per 400 steps, so it repays a 0.8 s device open in roughly **165 steps**. At 256² it
saves 99 ms per 400 steps and needs about **3,200**.

### What the measurement got wrong first

Three times, and each one is the reason a number above is trustworthy now.

**A 25 ms readback that was driver initialization.** The first buffer round trip on a fresh
device costs ~56 ms; every subsequent one of the same size costs ~160 µs. Attributing that
to whichever operation happened to go first made the first operation's measurement a lie,
so `GpuDevice::open` now pays it deliberately and the benchmark warms each problem shape
before timing it.

**A 500 ms readback at 256² that was a unit-parsing bug in the measuring script**, which
read `647.700 µs` as milliseconds. It was reproducible, it was size-inverted — 256² "slower"
than 1024² — and it survived two rounds of plausible hypotheses about power states and
dispatch overhead before an isolated harness showed the same operation taking 612 µs.
A reproducible wrong number is not more trustworthy than a noisy one.

**A speedup quoted from `--compare`.** Running both backends back to back in one process
gave 40× at 1024² where separate best-of-three runs give 130×, because the CPU baseline
leaves the machine in a different state. `--compare` is for a quick check; a published
figure needs separate processes and best of three, which is what
[execution.md](execution.md) already said and what this table does.

## The shape of the boundary

`lattice-compute` has no dependencies, because everything in it is a boundary.

**`Device` is not `Executor` with more implementations.** `Executor` answers "how do I split
this loop across threads that share memory". There is no `&mut [f64]` to hand out when the
data is in device memory, there is no loop because the kernel *is* the loop body, and the
split is not the caller's to choose. `CpuDevice` uses an `Executor` internally, the way a
wgpu device uses a command encoder.

**The host always speaks `f64`.** `Device::write` takes `&[f64]` and `Device::read` fills
`&mut [f64]`, whatever the device stores. That costs a conversion pass a device-native API
would avoid, and it buys the property the milestone needs: every narrowing happens at
exactly one place, so the disagreement has one entrance and can be attributed to it.

**A device refuses a precision it does not have.** `GpuDevice::open(Accurate64)` is an
error, not a silent downgrade to `fast32`. §15.1 draws no distinction between a faster
wrong solver and a quieter one.

**The buffer type is associated, not boxed.** Runtime backend selection happens once per
run, not once per dispatch, so the polymorphism lives at the top where the choice is
actually made rather than behind a vtable in the hot path.

**Face coefficients are uploaded, not recomputed.** The GPU takes
`DiffusionOperator::face_x` and `face_y` from the CPU operator that built them, so a
disagreement between backends is a disagreement about *arithmetic* and never about what the
coefficients were. §24.1 again: the CPU path is the executable specification, and a GPU
that re-derived the harmonic means would be a second implementation of a rule the
specification already fixes.

## The kernel cache

§15.5 asks for kernels cached *"by normalized expression, backend, precision, and hardware
capabilities."* All four are in `KernelKey`, and all four have to be. Dropping the backend
shares a WGSL module with a CPU closure; dropping the precision hands an `f32` kernel to an
`f64` run; dropping the capabilities reuses a module specialized for one device's workgroup
limit on a device with a smaller one — the one that would survive testing on a single
machine and fail in the field.

Normalization today is *lexical*: comments stripped, whitespace runs collapsed. §15.5's
normalization is semantic and belongs to §8.3's expression compiler, which does not exist
yet. The direction the weaker version is weak in matters: lexical normalization produces
**false misses** (`a+b` and `a + b` compile twice), which costs compilation time and
nothing else. It cannot produce a false *hit*, and a false hit is the only failure mode a
cache has that is worse than not caching.

## Running it

The GPU backend is off by default. §24.1's dependency policy and §15.4's portable baseline
pull in opposite directions — the GPU backend is the product baseline, and it is also a few
hundred crates, while `lattice check`, `run` and `validate` have no dependencies at all.

```console
$ cargo run -p lattice-wgpu --example probe          # what this machine offers
$ cargo run -p lattice-cli --features gpu -- validate --filter gpu
$ cargo test -p lattice-wgpu                          # the backend's own properties

$ cargo build --release -p lattice-cli --features gpu
$ ./target/release/lattice bench heat-explicit --backend gpu --scale 4
$ ./target/release/lattice bench heat-explicit --backend gpu --compare
$ ./target/release/lattice bench heat-crank-nicolson --backend gpu
```

`--backend gpu` runs only the benchmarks that have GPU kernels — today `heat-explicit` and
`heat-crank-nicolson` — and says so rather than falling back to the CPU for the others. A
GPU number produced by the CPU would be the most misleading thing the harness could print.
`particles-gravity` and `particles-lj` still have no GPU kernels; §15.6's "local particles"
and "Lennard-Jones MD" are GPU targets and the pair-force question is still open.

Without `--features gpu` the GPU rows are **absent** from `lattice validate` — 54 cases
rather than 65 — rather than reported as skipped-and-passing, which is how M3 handles the
Gillespie statistics and for the same reason. A machine with no adapter produces no rows
either: `GpuDevice::open` returns `Unavailable`, and `gpu::cases()` returns an empty list. A
case that could not run must not report as passing.

## Remaining limits after M4

Each of these is a decision with a reason, not an oversight.

**Explicit diffusion requires insulated boundaries.** Other modes are refused by the
GPU viewer. The implicit path supports mixed insulated and Dirichlet faces.

**Robin, periodic and nonzero-flux implicit boundaries.** These still require their own
GPU implementation and validation. Dirichlet now uses homogeneous ghosts in every CG
matrix product and the affine `c=L(0)` term in the right-hand side. CPU/GPU validation
covers mixed faces, corners and single-cell axes with variable coefficients and sources.

**Preconditioning.** The GPU CG is unpreconditioned, like the CPU's. A Jacobi or incomplete
Cholesky preconditioner would cut the iteration count, and since every iteration carries a
fence it would cut wall-clock more than it cuts arithmetic. It would also change the
iteration count and therefore the answer, which makes it a change to the *reference* and not
a backend optimization.

**Corner ghosts are not filled.** The five-point stencil reads `(i±1, j)` and `(i, j±1)`
and nothing diagonal, so no corner ghost is ever an input. That is a property of the CPU
operator, so it is asserted there rather than assumed here:
`the_five_point_stencil_never_reads_a_corner_ghost` poisons all four corners and requires
the output to be unchanged. If the operator ever gains a nine-point form, that test fails
and names the backend that has to change with it.

**General GPU model execution and rendering.** `lattice-view --gpu` presents a single
resident heat field directly. Coupled models and other visual channels still use the
CPU viewer. Diagnostics require explicit readback; drawing the GPU field does not.

**Fine-grained graph dispatch.** The runtime consumes `OperationGraph::levels` for
independent domain execution on CPU. GPU solvers accept caller-owned command streams,
but the general compiled-model driver does not yet schedule GPU domains or coupling.
