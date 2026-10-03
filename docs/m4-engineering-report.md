# M4 engineering record

Status: M4's scoped execution, rendering, validation and performance checklist is
complete, including the additional engineering review. Full-workspace formatting
debt remains as recorded below; this is not a claim that every future GPU path exists.

## Delivered paths

- `lattice-runtime`: graph-derived domain scheduling on the existing worker pool.
  Same-domain state and RAW/WAR/WAW hazards are ordered. Small/narrow levels retain
  the scalar or kernel-parallel driver. Per-domain scratch is allocated only when
  inter-domain parallelism will be used. No nested pool dispatch occurs.
- `lattice-wgpu`: velocity Verlet with uniform gravitational acceleration and
  Lennard–Jones, using SoA device storage and sorted spatial bins. Periodic and open
  boxes are supported. Coincident pairs are skipped, matching the CPU reference.
- Implicit diffusion: prescribed face values are represented by homogeneous
  Dirichlet ghosts plus the affine right-hand side. Both Crank–Nicolson and backward
  Euler use the existing CPU coefficients and numerical error budgets.
- `GpuFieldImage`: caller-owned palette and persistent texture, with prebuilt bind
  groups for both diffusion buffers. Compute precedes texture generation on one queue.
- `lattice-view --gpu`: a `.lattice` heat field on the renderer's device, without a
  second adapter/device open. Diagnostic snapshots are explicit and timestamped.
  The CPU viewer also now preserves compiled coupling edges when loading a model.
- Shader, pipeline and layout caching; deferred upload submission; removal of waits
  before mapping readback buffers. The implicit solver still observes every CG
  iteration and uses its persistent scalar staging buffer.
- Review fixes: explicit heat honors its requested preferred timestep without
  changing its stability limit; small canonical graphs use the original driver;
  uniform gravity reuses its constant force buffer; invalid rendering geometries
  fail before resource creation. Viewer diagnostic time tracks completed substeps.
  Benchmark artifact headers now identify the actual backend/precision, and mixed
  CPU/GPU validation artifacts identify both precisions instead of inheriting CPU defaults.

## Numerical checks

The CPU remains f64 reference arithmetic and the portable GPU remains f32. Existing
tolerances were not loosened. New gravity checks use accumulated rounding bounds.
Lennard–Jones trajectory checks bound rounding and short-time amplification on a
dilute lattice, with separate momentum, shifted-potential energy, minimum-image,
coincident-pair and exact repeatability checks.

Dirichlet validation covers 30 layouts (1x1, 1x9, 13x1, 17x11, 96x64; mixed faces and
four prescribed faces; two implicit schemes), with variable diffusivity and a source.
Worst disagreement used 13.98% of the existing derived solve budget. The original
implicit and explicit cases used 11.27% and 0.307% respectively. Gravity used 5.39%
of its rounding budget and LJ used 1.70% of its short-trajectory budget.

## Verification

The broad `cargo test --workspace --all-features --offline --no-fail-fast` run passed
1,060 tests including doctests. Subsequent review changes passed targeted compiler,
grid, runtime, validation and GPU suites, including two additional regression tests
(1,062 tests now registered). The release validation command passed all 56 cases,
including all eleven GPU rows on a real Vulkan adapter. Workspace Clippy with all
features/targets and warnings denied passed; rustdoc with warnings denied passed.

Tests added cover concurrent independent advances, dependent prepare ordering,
deterministic artifact hashes, explicit timestep selection, GPU pair forces and
invalid inputs, tiny periodic cell lists, resident texture pixels and orientation,
pipeline reuse with independent state, and compiled coupling in the CPU viewer.

The desktop smoke checks rendered both the Dirichlet slab and insulated explicit
pulse through the shared device and exited successfully. The first PowerShell
launcher lost its process exit status; direct invocation confirmed both exit codes.
The final explicit smoke reached exactly 0.16 s after 16 requested 0.01 s steps,
with integral change 4.8e-9. The slab reached 8 s after 16 half-second steps.

The initial workspace formatting check emitted approximately 1 MB of formatting
diffs, including untouched files. Formatting is applied and checked on changed Rust
files; unrelated formatting is intentionally kept out of this implementation.

## Measurement conditions and baseline

Windows, Rust 1.97.1 GNU toolchain, NVIDIA GeForce RTX 4070 Laptop GPU, Vulkan, release
profile. Compare separate processes on an otherwise idle machine. No CPU/GPU ratio
implies equal precision. Implicit comparisons also report iterations per step.

Before changes, three separate `heat-crank-nicolson --backend gpu` runs at 256x256
and 100 timesteps reported compute times 135.747, 142.597, 105.832 ms; setup times
164.688, 155.121, 177.477 ms; device opens 1.079, 1.069, 1.002 s. Median compute was
135.747 ms and median setup 164.688 ms. Every run passed correctness checks and used
4.57 CG iterations per step. Removing the extra map waits alone did not establish a
speedup above timing noise; final pipeline-cache measurements follow below.

## Final measurements

Compute excludes setup and observation. CPU particle comparisons use one thread
and f64, GPU uses f32. Three separate processes per particle/backend/size; table
entries are median milliseconds. All 36 particle runs passed their correctness
conditions. These are kernel-throughput comparisons, not process-startup speedups.

| Workload | Particles | Steps | CPU ms | GPU ms | CPU/GPU |
|---|---:|---:|---:|---:|---:|
| Uniform gravity | 16,384 | 400 | 18.076 | 2.887 | 6.26x |
| Uniform gravity | 65,536 | 400 | 128.385 | 6.385 | 20.11x |
| Uniform gravity | 262,144 | 400 | 801.606 | 9.188 | 87.25x |
| Lennard–Jones | 1,024 | 200 | 91.888 | 11.399 | 8.06x |
| Lennard–Jones | 4,096 | 200 | 371.397 | 6.739 | 55.11x |
| Lennard–Jones | 25,600 | 200 | 1,058.176 | 9.534 | 111.00x |

Small GPU workloads varied substantially: the 1,024-particle LJ runs took
9.99–13.61 ms, 4,096 took 6.38–9.39 ms, and 25,600 took 7.41–12.49 ms. These results
do not establish monotonic scaling or portable crossover sizes. Maximum LJ energy
drift was 1.061e-6, momentum drift 5.830e-8, and gravity position error 6.239e-5 m.
No particle state was non-finite. The LJ energy limit remains the CPU benchmark's
5e-3; the momentum and trajectory budgets account for GPU rounding.

| GPU implicit phase, 256x256 / 100 steps | Before median | After median |
|---|---:|---:|
| Compute | 135.747 ms | 77.604 ms |
| Benchmark setup, after warm-up | 164.688 ms | 6.978 ms |
| Device open | 1.069 s | 0.565 s |
| CG iterations/step | 4.57 | 4.57 |

After compute repeats: 80.266, 77.604, 77.143 ms. Setup: 7.006, 6.978, 6.841 ms.
The observed compute improvement is 1.75x. Device discovery was not changed; its
lower observed time is not attributed to the implementation. Opening a device still
dominates short processes. Reusing a renderer's device avoids a second open.

The dedicated setup example measured 103.411 ms for the first implicit solver and
1.231–1.398 ms (median 1.296 ms) for five subsequent solvers on the same device.
Two pipeline families remained cached and each independent constant field stayed
exact. This isolates the benefit of pipeline/layout reuse from adapter discovery.
The implicit loop still has one scalar readback per iteration plus one per step.
The CPU implicit baseline took 500.009 ms but 9.5 iterations/step at its tighter
tolerance, so the meaningful iteration-throughput ratio is about 3.10x, rather than
the 6.44x ratio of whole-step times.

Explicit heat smoke benchmarks passed at 256x256 (CPU 107.428 ms, GPU 3.667 ms) and
1024x1024 (CPU 1,933.577 ms, GPU 15.074 ms), each for 400 steps. These single-run
checks found no gross regression relative to the earlier explicit path; they are
not a new statistically controlled baseline.

Resident presentation measures 120 compute-and-color-conversion updates, draining
the queue at the end. It compares GPU → CPU field readback, CPU palette conversion
and texture upload against conversion directly into a resident texture. It excludes
window presentation/vsync and is not a desktop FPS claim. Median times across three
processes are below; final simulation fields were bit-identical in every comparison.

| Field | Round-trip ms | Resident ms | Ratio of medians |
|---|---:|---:|---:|
| 256x256 | 105.470 | 6.373 | 16.55x |
| 512x512 | 317.406 | 4.920 | 64.51x |
| 1024x1024 | 1,209.482 | 10.305 | 117.37x |

Graph scheduling compares the former two-phase driver with the scheduled driver,
both using four workers, four independent fields and 200 steps. Every field remained
bit-identical. Median paired times: 16x16, 0.867 → 0.892 ms; 256x256,
76.218 → 67.076 ms; 512x512, 350.183 → 331.645 ms. Small canonical graphs now execute
the same driver; their observed difference is 0.12 microseconds/step. The 512x512
paired ratios ranged from 0.77x to 1.14x, so there is no robust speedup claim for that
size. Ready-level scheduling is correct; larger-scene load balancing remains useful
follow-up work. Do not infer a universal parallel speedup from these measurements.

Local raw logs/JSON are under `runs/m4-*`. The particle matrix and first heat timings
were recorded before the artifact-header correction; their per-benchmark rows have
the correct backend/precision. Final artifact smoke checks separately verify the
corrected headers. Existing raw measurement files were not rewritten.

The final exported artifact check passed all four GPU benchmarks, a two-thread CPU
benchmark, and all 56 validation cases. Headers were parsed and checked against the
execution: `wgpu`/`fast32`, the CPU worker configuration/`accurate64`, and
`cpu+wgpu`/`accurate64+fast32` respectively. Artifacts are
`runs/m4-final-gpu-artifact.json`, `runs/m4-final-cpu-artifact.json`, and
`runs/m4-final-validation.json`.

## Reproduce

```powershell
cargo test --workspace --all-features --offline --no-fail-fast
cargo clippy --workspace --all-features --all-targets --offline -- -D warnings
cargo build --release -p lattice-cli --features gpu -p lattice-viewer
./target/release/lattice validate
./target/release/lattice bench --backend gpu
./target/release/lattice bench particles-lj --backend gpu --scale 5
cargo run --release -p lattice-wgpu --example render_bench
cargo run --release -p lattice-wgpu --example startup_bench
cargo run --release -p lattice-cli --example schedule_bench
./target/release/lattice-view examples/slab.lattice --gpu --play
./target/release/lattice-view examples/gpu_explicit_pulse.lattice --gpu --smoke-frames=4
```

Substantially changed crates: `lattice-wgpu` (solvers, resources, rendering and
bench examples), `lattice-runtime` / `lattice-ir` (scheduling and domain ownership),
`lattice-validation` (CPU/GPU comparisons), `lattice-cli` (particle benchmarks),
`lattice-viewer` (shared-device view and coupling), and `lattice-domain-grid2d`
(backend setup accessors and timestep correction). The existing uncommitted work in
`lattice-compute`, `lattice-cpu`, implicit kernels and validation remains preserved.

`Domain` now requires `Send` for exclusive parallel ownership and gains type inspection
through the blanket `DomainType`/`Any` implementation. Existing method names are
preserved; external domains must own thread-sendable, `'static` state.

## Scope and known limits

- `lattice run` is still the CPU model driver. GPU particle solvers are exposed in
  Rust and through benchmarks; the GPU viewer accepts one uncoupled heat field.
- GPU particle integration currently implements velocity Verlet, uniform gravity,
  and cutoff LJ. It does not implement all-pairs Newtonian gravity, rigid bodies,
  reflective boxes, user force laws or GPU chemistry.
- GPU explicit diffusion requires insulated faces. Implicit diffusion supports
  insulated/Dirichlet mixtures; Robin, periodic and nonzero-flux GPU faces remain
  unsupported.
- The graph scheduler uses ready levels, not a work-stealing graph executor. It
  avoids inter-domain dispatch when there are fewer ready domains than worker
  threads. Coupling is a synchronization phase after all advances.
- Per-cell particle sorting is quadratic in cell occupancy. Typical dilute LJ
  workloads have bounded small occupancy; collapsed configurations can be slow.
- GPU diagnostics still involve readback; implicit stopping requires one scalar
  readback per iteration. Field drawing itself requires none. GPU viewer colors
  use a labelled fixed range and a 256-entry lookup table.
- Hardware validation here covers Windows/Vulkan on one NVIDIA adapter. Other
  drivers, operating systems and adapters still need CI/hardware coverage.

M5+ follow-up: molecular bonds and quantum validation; general backend selection for
compiled models; GPU coupling/observation reductions; improved graph load balancing;
larger/denser particle benchmarks and checkpoint-based solver batching.
