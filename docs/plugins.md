# Writing an observer plugin

An *observer* measures a run without changing it. The language can express a measurement
of one particle at a time ([observer laws](language.md#observer-laws)); anything that
needs memory between samples, a neighbour search, a file or a curve is a plugin: a Rust
crate that links `lattice-sdk`, implements `Observer`, and is attached to a compiled
simulation. Spec §16.3 calls this the Observer tier, and FR-015 asks for an example that
*"builds without modifying engine source"*. [`plugins/vacf`](../plugins/vacf) is that
example: a velocity autocorrelation, in a crate outside the engine's workspace.

A plugin is **not** a solver. It sees the domains through `&`, so it cannot change the
physics, and a model's results are the same whether or not one is attached (a test
compares every other row bit for bit). It is also not dynamically loaded: there is no
`dylib`, no `cdylib` and no stable binary interface (below).

## The SDK surface

`lattice-sdk` ([`crates/lattice-sdk`](../crates/lattice-sdk)) is a facade. A plugin names
no other engine crate; everything it touches is re-exported from here, so the engine can
move code between crates without breaking it.

| Item | What it is |
|---|---|
| `Observer` | the trait: `contract`, `observe`, and by default `curves`, `due` and `explain_non_finite` |
| `ObserverContract`, `SideEffects`, `Backends`, `ObserverGap` | what an observer says about itself, and what is wrong with a contract that says too little |
| `ObserveContext` | what `observe` is given: `time()`, `step()`, `observations()` (the rows the domains just published), `domain::<T>(name)` |
| `Observations`, `Observation`, `ObservationKind`, `Curve` | what it publishes: rows (`record_metric`) and, at the end of a run, curves |
| `ParticleView` | **the supported way to read particles** |
| `compile_file`, `compile_source`, `compile`, `Compiled`, `Simulation`, `RunConfig`, `RunOutcome`, `StopReason`, `ObserverFault` | build a model and run it |
| `Diagnostic`, `Diagnostics`, `Severity`, `SourceFile` | what the compiler says when it cannot |
| `SDK_VERSION` | `(0, 1)`: the revision of this surface |

`ParticleView` wraps a particle domain and gives read-only slices: positions, velocities,
masses and charges, indexed by particle slot, and nothing that mutates. It exists so that
the particle store can change without breaking plugins. `ObserveContext::domain::<T>` can
downcast a domain to its concrete type, because every domain can be reached as `Any`, and
a plugin that does so reads the engine's internals and is coupled to them. That is
allowed and unsupported. The view is the surface the engine keeps stable (as far as
anything here is stable; see below), and it is the only one `plugins/vacf` uses.

Only particle domains have a view. A grid, a wavefunction or a rigid-body world has
no view yet, and a plugin that wants one has to go through `domain::<T>` and the engine's
types, which is exactly the coupling the view avoids.

### `compile_file`

```rust
let mut simulation = lattice_sdk::compile_file(Path::new("model.lattice"))?;
simulation.add_observer(Box::new(my_observer))?;
let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
```

`compile_file` returns the diagnostics of a model that does not compile (the warnings
with the errors), and attaches the model's own law observers first, in source order, so
that a plugin's rows sit beside theirs. The warnings of a model that does compile are not
returned; call `compile_source` for those. The file is the model's source text and
nothing else: the SDK does not run the plugin on its own, and a plugin ships its own
binary (`plugins/vacf/src/bin/vacf.rs` is the pattern).

### The contract, and its audit

P8 asks every extension to state what it reads and writes; P1 asks what it does *not*
measure. An `ObserverContract` carries both: `name`, `summary`, `reads` (domain instance
names), `outputs` (each row and curve, with its unit, `"1"` for a plain number),
`side_effects`, `limitations`, `deterministic`, `backends` and `differentiable`.

`Simulation::add_observer` audits it and returns `Err` with every gap if:

- `name`, `summary`, `reads`, `outputs` or `limitations` is empty, or no backend is named;
- a `reads` name is not a domain of the model;
- an output is declared twice, is already declared by an observer that is attached, or is
  a row a domain already publishes.

That is stricter than a solver's contract, whose audit a test suite runs. It is deliberate: a
plugin has no validation suite of ours to fail, so the refusal happens when it is
attached. It checks that the contract is *filled in*, not that it is *true*: an
observer that lists a limitation it does not have, or leaves out one it does, passes. The
engine's own coverage of this is partial too (spec M6.2 §6): the validation lab audits the
particle, heat and rigid contracts, and `lattice inspect contracts` the particle and heat
ones.

At run time the contract is enforced as behaviour. An observer is switched off (`E0951`)
if it publishes a row it did not declare, or in another unit than it declared, or the same
row twice in one call, or a row of kind `Invariant` (only the engine's own solvers may
claim that a quantity is conserved, so an observer publishes metrics), or a curve it did
not declare. `ObserveContext::domain` returns `None` for a domain outside `reads`.

### Cadence, and what a sample is

The run samples at the smallest `every` of the model's `observe` statements. `observe`
is called at the samples the observer's `due(time, dt)` accepts, after every domain has
published, in the order the observers were attached (law observers first). The default
`due` accepts every sample. `due` may advance the observer's schedule when it says yes, so
a sample is a one-time event: it is asked once per sample, and `Simulation::observe`
must not be called twice for an instant. A run's final step is also sampled whether or
not it falls on the cadence, so `due` should say no to an instant it does not want.

A non-finite row halts the run, as a domain's does (NFR-007). So an observer must not
publish NaN for "no value yet": publish nothing that sample. (`plugins/vacf` does this for
its row, which is absent until the second sample.)

A model's own `observe sum(law(set))` law observers refuse an `every` that is zero,
negative or not finite (`E0405` in the compiler, and `LawObserver::new` returns an error),
because such an interval would silently sample at every step. An observer written
against the SDK reads its own interval and should do the same.

## What happens when a plugin panics

The runtime wraps each call in `catch_unwind`. An observer that panics is **disabled**,
the half-written row of that call is dropped, the run carries on, and `lattice run`
prints `error[E0951]: observer `vacf` panicked at step N and was disabled: …`. The other
rows are bit-identical to a run without the observer, and its own rows stop at the last
call that finished (a unit test in `lattice-runtime` checks this with an observer that
panics on its third call). NFR-004 (*a failing extension does not take the run down*) is met for a panic.

It is **not** met for anything that is not a panic:

- `panic = "abort"` in the profile that builds the plugin turns the panic into the end of
  the process. The workspace sets no `panic` key and the default is `unwind`; a plugin
  with its own `[profile]` must keep it. `plugins/vacf/Cargo.toml` says so beside its
  profile.
- An observer that loops forever hangs the run, and one that overflows the stack ends it.
- A panic while another is unwinding aborts, and so does a panic in a destructor.
- `&` access means a plugin cannot change a domain through the `&` API. A domain with
  interior mutability (a `RefCell`, a `Mutex`) reached through `domain::<T>` is a way
  around that, and a panic in the middle of touching one can leave it inconsistent. The
  runtime can disable the observer; it cannot repair the domain. Plain slices, which is
  all `ParticleView` hands out, avoid this.
- `E0951` is printed by `lattice run`. The viewer does not print it, and it attaches no
  plugin at all (it runs a model file and the model's own law observers).
- `contract()` is read when the observer is attached, outside `catch_unwind`, so a
  `contract()` that panics fails `add_observer` for the caller. The runtime caches the
  name, `reads` and `outputs` then and does not call it during a run. One call is
  left: when a run halts on a non-finite row an observer declared, `explain_observer_halt`
  reads that observer's contract again, unguarded (listed in the handoff).

## No stable ABI

A plugin is compiled with the engine version it names, as a Rust crate with a path (or
version) dependency on `lattice-sdk`. There is no dynamic loading, and none is planned:
spec §23 calls the plugin ABI a risk and prescribes *"unstable internal ABI until
mature"*. `SDK_VERSION` is the revision of the source surface. While the major number is
0, it can change in a minor, and a plugin should compare it with the value it was
written for and refuse to run on a mismatch, as `plugins/vacf` does at startup.
Nothing here promises that a plugin built against one engine commit links against
another.

## Building the example, and how it is tested

```console
$ cargo build --manifest-path plugins/vacf/Cargo.toml
$ cargo run --release --manifest-path plugins/vacf/Cargo.toml -- examples/argon.lattice atoms --lags 50
```

`plugins/vacf` has its own `[workspace]` and a path dependency on `crates/lattice-sdk`, so
it is not a member of the engine's workspace: `cargo test --workspace` does not build it.
The validation case `plugin_vacf_ideal_gas` does, and that is the standing test that the
example builds and works against the engine as it is. It has cargo build and run the
plugin's binary on a force-free gas in a Langevin bath, and requires `C(τ)` to match
`exp(−γτ)` within five standard errors derived from the particle and origin counts
(the derivation is in `crates/lattice-validation/src/plugin.rs`; it is not tuned).

Two things follow from how it is run, and both matter on a machine that is not this one:

- **`lattice validate` needs cargo and a source checkout.** The case finds the manifest
  through the path compiled into the validation crate (`CARGO_MANIFEST_DIR`), and runs
  `cargo run --release` on it (the `CARGO` environment variable if set, otherwise `cargo` on the
  `PATH`). A copy of the `lattice` binary on a machine with no cargo, or with the source
  tree gone or moved, reports the case as failed (unmeasured, with the reason), not
  skipped. Every other case still runs.
- **The first run builds the engine a second time.** The plugin is its own workspace, so it
  gets its own target directory (`plugins/vacf/target`, or `$CARGO_TARGET_DIR/plugin-vacf`
  when that is set) and builds `lattice-sdk` and what it pulls in, in release, from
  scratch. That is minutes, once; later runs reuse it.

## What the example shows

`Vacf` keeps the last `lags + 1` velocity snapshots in a fixed ring that `new` allocates,
adds one dot product per lag and particle at each sample, and publishes `C(τ)` as a curve
(`SET.vacf`) and the running estimate at one lag as a row (`SET.vacf.lag1`). Its contract
lists what it does not measure: lags are multiples of the sample interval; origins
overlap, so there is no error bar; particles are followed by slot, so a changed count
disables it; the centre-of-mass drift is not subtracted; it reads two components only;
and `exp(−γτ)` holds only for free particles in a Langevin bath, not in a dense fluid.
The ring and the sums are never reallocated; the one allocation left per sample is
the row's name, cloned by `record_metric`.

## Not built

These came up while writing the observers and are left for later; the roadmap's
[M6.2 handoff](roadmap.md#handoff-from-m62) has the list.

- Pair observers (sums over neighbour pairs, such as a coordination number), and observer
  laws over grids, wavefunctions and rigid bodies. Only the particle domain has a law
  input layout.
- A `histogram` reduction. It produces a curve and not a scalar, and how to accumulate
  across samples is its own design question. A plugin can publish a curve today, as
  `Vacf` does.
- Filtering by target: the plain `observe TARGET` still records everything.
- Observers on the GPU: `backends` can name one, and nothing runs it.
- Dynamic loading, a stable ABI, and the Python API (M6.5, whose observer hook should
  wrap this trait).
