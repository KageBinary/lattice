# Development

## Toolchain

Rust 1.85 or newer (edition 2024). No other tools are needed — the workspace has zero
external dependencies.

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
$ cargo test                      # 833 tests across 14 crates
$ cargo test -p lattice-units     # one crate
$ cargo build --release           # the `lattice` binary
$ cargo build --release -p lattice-viewer   # the `lattice-view` window
```

Tests run at `opt-level = 2` (see the root `Cargo.toml`). The validation suite runs
convergence studies over thousands of steps; at `opt-level = 0` they take minutes
instead of a second.

`lattice-viewer` is the only crate with external dependencies, and a cold build of its
GPU stack takes several minutes. Nothing else depends on it, so building or testing any
other crate by name never pays for it. Changing anything a domain *publishes* —
`Observations`, `RenderChannel` — does reach it, so run the whole workspace before
committing.

**A green test suite does not mean the window is right.** Every defect found in the
viewer so far was found by screenshotting the running program, not by a test: round-off
plotted as a dramatic oscillation, a values table saying "field units" beside a scale
bar saying "K", a legend box parked on the curve it labelled, and a perfectly conserved
momentum reported as `not conserved` because the denominator was `1e-15`. Build it, run
it on `examples/lj_gas.lattice` and `examples/diffusing_pulse.lattice`, and look at it.

**Always benchmark a release build.** `lattice bench` prints a loud warning when
`debug_assertions` is on, because a timing from an unoptimized build is off by an order
of magnitude and is the easiest way to publish a misleading number.

## Testing a window without a window

`lattice-view` and `lattice-play` are the only parts of the engine that cannot be tested
by calling them. The rule that has worked: **drive the logic, not the pixels.**

Everything a mode does in response to the pointer goes through `Playground::pointer`,
which takes a plain `Pointer` struct and needs no window, no GPU and no event loop. So
`a_throw_leaves_the_body_moving_at_the_hands_speed` constructs the press and the release
directly and asserts on the resulting velocity. Fifty tests cover the click paths this
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
