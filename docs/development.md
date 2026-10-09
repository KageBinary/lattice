# Development

## Toolchain

Rust 1.85 or newer (edition 2024). No other tools are needed. Seventeen of the twenty
crates have no external dependencies at all: the two that open a window, and the portable
GPU backend, are the exceptions.

### Windows: use the GNU toolchain unless you have MSVC C++ build tools

Rustup installs `stable-x86_64-pc-windows-msvc` by default, which needs the linker
from Visual Studio's *"Desktop development with C++"* workload. Installing Visual
Studio without that workload leaves you with a `link.exe` that is not the MSVC linker —
usually GNU coreutils' `link`, which produces this:

```
error: linking with `link.exe` failed: exit code: 1
  = note: link: extra operand '…rcgu.o'
          Try 'link --help' for more information.
```

Two fixes. Either install the C++ build tools, or switch to the GNU toolchain, which
bundles its own linker and needs nothing else:

```console
$ rustup toolchain install stable-x86_64-pc-windows-gnu
$ rustup default stable-x86_64-pc-windows-gnu
```

There is deliberately no `rust-toolchain.toml`. Pinning `channel = "stable"` would
resolve to the *host default* triple, which is `msvc` on Windows — reintroducing the
problem for exactly the people this note is for.

## Building and testing

```console
$ cargo test --workspace --all-features # 1258 tests including doctests across 20 crates
$ cargo test -p lattice-units     # one crate
$ cargo build --release           # the `lattice` binary
$ cargo build --release -p lattice-viewer   # the `lattice-view` window
```

Tests run at `opt-level = 2` (see the root `Cargo.toml`). The validation suite runs
convergence studies over thousands of steps; at `opt-level = 0` they take minutes
instead of a second.

`lattice-viewer`, `lattice-playground` and `lattice-wgpu` are the only crates with
external dependencies, and a cold build of their GPU stack takes several minutes. Nothing
else depends on any of them by default, so building or testing another crate by name never
pays for it. Changing anything a domain *publishes* — `Observations`, `RenderChannel` —
does reach them, so run the whole workspace before committing.

### The GPU backend

Off by default. §19.1's GPU cross-backend rows are *absent* from `lattice validate` without
the feature — 63 cases rather than 74 — rather than reported as skipped-and-passing.

```console
$ cargo run -p lattice-wgpu --example probe    # what this machine's adapter offers
$ cargo test -p lattice-wgpu                   # the backend's own properties
$ cargo run -p lattice-cli --features gpu -- validate --filter gpu

$ cargo build --release -p lattice-cli --features gpu
$ ./target/release/lattice bench heat-explicit --backend gpu --scale 4
$ ./target/release/lattice bench particles-lj --backend gpu --scale 5
$ cargo run --release -p lattice-wgpu --example render_bench
$ cargo run --release -p lattice-wgpu --example startup_bench
$ cargo run --release -p lattice-cli --example schedule_bench
```

**Publishing a GPU timing needs more care than a CPU one.** Opening a device costs about
0.8 s once per process and is printed separately for that reason; the first buffer round
trip on a fresh device costs ~56 ms against a ~160 µs steady state; and `--compare` runs
both backends in one process, which is fine for a check and wrong for a published figure.
Separate processes, report all repeats and their median, and read [backends.md](backends.md)'s "what the
measurement got wrong first" before trusting a surprising number — all three mistakes it
records were reproducible.

Every GPU test skips with a printed reason when no adapter opens, so a machine without one
still passes the suite without silently claiming to have exercised it. Read
[backends.md](backends.md) before changing anything about precision or tolerance — in
particular, the portable backend runs `fast32` because **WGSL has no `f64`**, and
`wgsl_f64_is_rejected_by_the_portable_backend` is written to fail if that ever stops being
true.

**A green test suite does not mean the window is right.** Every defect found in the
viewer so far was found by screenshotting the running program, not by a test: round-off
plotted as a dramatic oscillation, a values table saying "field units" beside a scale
bar saying "K", a legend box parked on the curve it labelled, and a perfectly conserved
momentum reported as `not conserved` because the denominator was `1e-15`. Build it, run
it on `examples/lj_gas.lattice` and `examples/diffusing_pulse.lattice`, and look at it.

**Always benchmark a release build.** `lattice bench` prints a loud warning when
`debug_assertions` is on, because a timing from an unoptimized build is off by an order
of magnitude and is the easiest way to publish a misleading number.

## Adding a parallel kernel

Read [execution.md](execution.md) first — particularly the promise, which is that
parallel execution changes the schedule and never the numbers. Three rules follow, and
all three have already been broken once:

- **Write the arithmetic at the call site.** `dt * (m⁻¹F)` and `(dt · m⁻¹) · F` are
  different numbers. A shared `axpy` helper that normalised them would be tidier and
  would silently change two integrators' results.
- **Declare a `Grain`, and measure it.** `Grain { floor, chunk }` says how much work is
  worth a barrier and how small a piece may get. They are different numbers; see
  execution.md for what conflating them cost. Measure with `lattice bench <name>
  --threads auto --compare`, best of three — the noise floor is around ±20%, and a
  "regression" that justified a redesign here turned out to be nothing.
- **No reductions on the CPU.** A sum split into chunks is not the sum added in order, and
  the difference would vary with the machine's core count. If a CPU kernel needs one, it
  stays sequential and says so.

  The GPU has no sequential path to retreat to, so the rule there is different and stricter
  in its own way: a device reduction must have an association order that is **fixed and
  written down**, not merely deterministic by accident. `reduction.wgsl` states its order in
  the file header and `Interior::depth` turns it into the number a budget is handed. A
  reduction that merely *differs* from the reference is a mechanism you can bound; one whose
  order is unpredictable is not, and no amount of tolerance fixes it.

Then add a case to `crates/lattice-validation/src/execution.rs`, sized **above** the
grain's floor. A cross-backend case on a problem too small to be split compares the
sequential path with itself and passes without testing anything;
`the_cases_actually_split_their_work` is the guard against that, and it needs the new
grain added to it.

## Testing a window without a window

`lattice-view` and `lattice-play` are the only parts of the engine that cannot be tested
by calling them. The rule that has worked: **drive the logic, not the pixels.**

Everything a mode does in response to the pointer goes through `Playground::pointer`,
which takes a plain `Pointer` struct and needs no window, no GPU and no event loop. So
`a_throw_leaves_the_body_moving_at_the_hands_speed` constructs the press and the release
directly and asserts on the resulting velocity. Sixty-two tests cover the click paths this
way, and they run in the ordinary `cargo test`.

Only `draw` needs a real `egui::Painter`, and it is deliberately thin — it reads state
and emits shapes, with no decisions in it that a test would want to make claims about.

Automated capture of the running window was tried and is not reliable on this machine.
For the record, so nobody spends the afternoon again:

- `Process.MainWindowHandle` is cached at first look and goes stale once the app has been
  interacted with. Re-resolve by enumerating the process's own top-level windows.
- `PrintWindow` renders the whole window including the frame; sizing the bitmap from
  `GetClientRect` silently crops that many pixels off the bottom, which looks exactly
  like the app clipping its own controls.
- `SetForegroundWindow` is refused from a process that is not already in front, so
  synthetic clicks land on whatever window actually is.
- In PowerShell, aliases resolve before functions, and `Move` is `Move-Item`. A helper
  named `Move` renames files instead of moving the pointer — and because the button
  events still post, the window sees a click with no cursor position behind it and
  correctly ignores it.

Even with all four fixed, the input that arrives is not reproducible enough to assert on.
Screenshots are worth taking to *look* at; they are not worth building a suite on.

## Conventions

### Tests state what they establish

Test names are claims, not labels: `symplectic_schemes_bound_the_energy_error_and_explicit_euler_does_not`,
not `test_integrators`. When a test encodes a physical fact or a subtle invariant, the
doc comment says *why it matters*, so a future reader knows whether a failure is a
regression or a changed requirement.

### Tolerances are justified, not tuned

A tolerance is a claim about the numerics. Two rules:

- **Judge against the right scale.** Pair forces cancel exactly in real arithmetic, so
  the residual is round-off *of the force magnitude*. Asserting `< 1e-20` on forces of
  10⁵ N fails for no reason; asserting `< 1e-12 × scale` is the actual claim.
- **Prefer convergence to magnitude.** "These two schemes agree to 1e-3" is arbitrary.
  "The gap between them closes at first order as `dt` shrinks" is the property that
  distinguishes a truncation error from a bug, and it is what several validation cases
  measure.

Where a tolerance admits a real physical effect, say so. `atomic_mass_unit_matches_avogadro_to_the_2019_tolerance`
allows 5e-9 because the 2019 SI redefinition demoted the molar mass constant from
exactly 1 g/mol to a measured quantity — and the same test asserts the gap is *larger*
than 1e-12, so a future edit that makes the two literals agree exactly fails.

### Every solver publishes a contract

Adding a domain means writing a `SolverContract`. `audit()` checks the sections are
filled in and a validation case fails the build if `known_non_conservation` is empty.
If you cannot name something your solver does not conserve, look harder — cutoff
truncation, iterative solver tolerance, and non-closed boundaries are almost always
candidates.

## Adding a validation case

Cases live in `crates/lattice-validation/src/`. A case *measures a number*:

```rust
Case {
    name: "heat_spatial_convergence",
    domain: "grid2d",
    level: Level::Manufactured,
    claim: "refining the grid reduces the error against the continuous solution at second order",
    run: spatial_convergence,
}

fn spatial_convergence() -> Outcome {
    let coarse = error_at(16);
    let fine = error_at(32);
    Outcome::near("observed spatial convergence order", "1", (coarse / fine).log2(), 2.0, 0.15)
        .note(format!("max error {coarse:.3e} at 16^2 -> {fine:.3e} at 32^2"))
}
```

The measured value is reported whether the case passes or fails, which is the point: an
error still under tolerance but ten times larger than last week is a regression that
pass/fail cannot show.

Registering it in the module's `CASES` slice is enough — `lattice validate` and the
suite test pick it up automatically, and `case_names_are_unique` guards against
collisions.

## Adding a benchmark

Benchmarks live in `crates/lattice-cli/src/bench.rs`. Every one carries the correctness
condition spec §15.6 attaches to its performance target, and a benchmark that fails its
checks has its throughput marked `RESULT INVALID` rather than reported. That is §15.1
in code: *"a faster wrong solver is a regression."*

## Repository conventions

- Crate names follow spec §24: `lattice-<component>` or `lattice-domain-<family>`.
- New crates are added to `[workspace.dependencies]` in the root `Cargo.toml` when they
  land, so the workspace always builds.
- `crates/lattice-cli` does not inherit workspace lints: `unreachable_pub` fires on
  every item in a binary crate and says nothing useful.
