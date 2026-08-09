# Backends

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

## What the CPU pair bought

M4.1 held the scalar and parallel CPU paths to *bit-identical* agreement, at a real cost in
performance — Crank–Nicolson gains only 1.6× because its inner products stay sequential.
This is what that bought.

Every unit of budget above belongs to the GPU and can be pointed at. Had the CPU pair been
tolerance-based, there would now be a single number covering thread scheduling and `f32`
storage together, and nobody could say which part belonged to which cause. The two levels
of claim stay visibly different kinds of statement rather than one weakened into the other:
`execution.md`'s cases assert equality of IEEE bit patterns and carry no tolerance at all.

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
```

Without `--features gpu` the GPU rows are **absent** from `lattice validate` — 45 cases
rather than 48 — rather than reported as skipped-and-passing, which is how M3 handles the
Gillespie statistics and for the same reason. A machine with no adapter produces no rows
either: `GpuDevice::open` returns `Unavailable`, and `gpu::cases()` returns an empty list. A
case that could not run must not report as passing.

## What is deliberately not built

Each of these is a decision with a reason, not an oversight.

**Only insulated boundaries.** Zero-gradient Neumann is a copy. Dirichlet, Robin and
periodic each need their own halo kernel and their own test, and a boundary condition
quietly replaced by a different one produces a run that looks entirely plausible and is
wrong. Asking for one is an error.

**Only the explicit scheme.** Crank–Nicolson needs conjugate gradient, CG needs inner
products, and a reduction is where `Mechanism::ReductionOrder` stops being zero. On the CPU
that term was avoided by keeping reductions sequential; a GPU has no sequential fallback to
retreat to, so it is a real decision and it belongs with the work that makes it.

**Corner ghosts are not filled.** The five-point stencil reads `(i±1, j)` and `(i, j±1)`
and nothing diagonal, so no corner ghost is ever an input. That is a property of the CPU
operator, so it is asserted there rather than assumed here:
`the_five_point_stencil_never_reads_a_corner_ghost` poisons all four corners and requires
the output to be unchanged. If the operator ever gains a nine-point form, that test fails
and names the backend that has to change with it.

**Zero-copy rendering.** §15.5 and §15.2 both want simulation buffers feeding the viewer
directly. The buffers now exist on the device; the viewer still uploads a CPU texture.

**The operation graph is still unconsumed.** `OperationGraph::levels` has computed which
operations are independent since M1, and nothing reads it. A scheduler *across* operations
rather than within one is what would use it, and it remains the scheduler-shaped hole
described in the roadmap.
