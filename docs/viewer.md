# The viewer

`lattice-view` opens a `.lattice` model in a window and runs it.

```console
$ cargo build --release -p lattice-viewer
$ ./target/release/lattice-view examples/diffusing_pulse.lattice --play
```

It is a separate binary from `lattice`. The GPU stack it needs is a few hundred crates
and several minutes of cold build; keeping it out of the core CLI means `lattice
validate` stays dependency-free and quick to compile.

---

## What the window shows

```
┌──────────────────────────────────────────────┬──────────────────────────┐
│ ⏸ pause  ⏭ step  ↺ reset   t  step  dt       │ diagnostics              │
│ speed ▁▁▁█▁▁▁  4 steps/frame        ☀ light  │  stability               │
├──────────────────────────────────────────────┤   ✔ stable — dt is 2% of │
│ temperature   64×64 scalar field in K        │     the 0.2646 s limit   │
│ ┌──────────────────────────────────────────┐ │     set by domain `atoms`│
│ │                                          │ │                          │
│ │              the field, or               │ │  conservation            │
│ │           the particle scatter           │ │   total_energy           │
│ │                                          │ │   ! drifting slowly      │
│ │                                          │ │     -2.5e-4 of the       │
│ └──────────────────────────────────────────┘ │     initial value        │
│ ▁▂▃▄▅▆▇█  0.0324 K              10.3193 K    │                          │
└──────────────────────────────────────────────┤  current values          │
                                               │   integral 2.2619 K·m^2  │
                                               │   min      0.0324 K      │
                                               │   …                      │
                                               │                          │
                                               │  over time               │
                                               │   ┌────────────────────┐ │
                                               │   │ one plot per unit  │ │
                                               │   └────────────────────┘ │
                                               │   ● min ● max            │
                                               │                          │
                                               │  ☐ solver contracts      │
                                               │  ☐ model report          │
                                               └──────────────────────────┘
```

The picture is the largest thing on screen and it never appears alone. Beside it sit
the exact numbers behind it, how far each conserved quantity has drifted, how close the
timestep is to its stability limit, and what the solver admits it does not conserve.

That layout is a reading of design principle **P7**:

> The renderer is not decoration. It must display fields, fluxes, constraints, forces,
> residuals, conservation drift, timestep decisions, and uncertainty.

A viewer that showed only the pretty field would be the thing **P1** warns against —
*"a visually convincing result is not sufficient."*

A model that does not compile still opens, and shows its diagnostics with the same
carets `lattice check` prints. Fix the file and press reset.

---

## The rules the rendering obeys

Each of these exists because breaking it makes a picture assert something the data does
not say.

### Sequential colour means one hue

A rainbow ramp invents visual boundaries — the eye reads the green-to-yellow transition
as a step change, because it is one, in the palette. The reader then sees structure that
belongs to the colourmap rather than to the field.

So the sequential map is a single hue, light to dark, and `palette.rs` has a test that
fails the build if it ever stops being one:

```rust
assert!(delta < 0.26, "hue drifted by {delta} radians — that is a rainbow");
```

Monotone lightness is asserted too, so the ramp still orders correctly in greyscale, in
print, and for a reader with any form of colour vision deficiency. The diverging map is
two hues with a neutral grey midpoint, its arms checked for perceptual symmetry so that
"+3" and "−3" look equally far from zero. All of it is computed in OKLab; sRGB
interpolation would put a dark muddy band in the middle of every ramp.

### One unit per plot, one axis per plot

Energy is in joules and momentum is in kg·m/s. Drawing them on one pair of axes needs
two y-scales, and the alignment between two y-scales is a free parameter — so the chart
manufactures a correlation the reader did not ask for and cannot check.

`History::by_unit` groups series by unit and the app draws one plot per group. There is
no code path that puts two units on one axis.

### Colour never carries a value alone

Every heatmap ships a scale legend and the exact min / mean / max. Every status ships an
icon *and* a word — `✔ conserved`, `! drifting slowly`, `✖ not conserved` — never a
coloured dot on its own. Every plot ships a legend and a values table. Nothing in the
window requires distinguishing two colours to be read correctly.

### Never let the picture imply structure the data does not have

The recurring one, and the source of most of the bugs found by actually looking at the
window rather than at the tests.

**Flat fields.** A heatmap normalizes to its own range, so a field that is uniform to
round-off renders as full-contrast noise. `FieldImage::is_flat` detects it and the
viewer paints a flat surface with a note instead.

**Round-off drawn as oscillation.** A plot auto-scales to whatever range it is given,
so a conserved quantity wobbling in its fifteenth significant digit draws exactly the
same dramatic waveform as one that doubles — with fifteen-digit axis labels to match.
Beside a series that genuinely moves, that reads as instability. Such series are
excluded from plots, reported as numbers, and listed under *"constant to round-off, not
plotted"* so a missing curve is a stated decision rather than a mystery.

**Nearest-neighbour sampling, deliberately.** Smoothing makes a coarse grid look like a
fine one. Blocky cells are the honest picture: the reader can see the resolution they
are actually getting.

---

## Conservation drift, and the momentum problem

The panel's most-used line is *how far has this drifted*, and getting it right took two
corrections that only became visible in a screenshot.

### Dividing by zero, slowly

Total momentum in a system set up at rest is conserved **at zero**. Its initial value is
not `0.0` but `1e-15` — the round-off left over from summing a few hundred momenta that
cancel. Divide by that and a flawless run reports:

```
momentum_x   ✖ not conserved   -7.288e1 relative to the initial value
```

while the values table two inches below reads `-9.576e-14 kg·m/s`. The number was a
division by noise, and it was the loudest thing in the window.

The fix is that a domain may publish the scale its quantity should be judged against.
`ParticleDomain` publishes `momentum_scale` — the sum of the individual momentum
magnitudes, `Σ|mᵢvᵢ|`, which only the domain knows. `History::drift_of` uses the
quantity's own initial value when that value means something, and falls back to the
published scale when it does not, reporting which basis it used:

```
momentum_x   ✔ conserved to round-off   -2.044e-16 of momentum_scale
```

`momentum_scale` appears in the values table, so the label points at a number the reader
can see rather than describing one they cannot.

The same lookup decides plotting. A net momentum wandering between ±1e-13 spans its own
range many times over, so a self-relative flatness test calls it varying and draws it;
against a scale of 250 kg·m/s it is indistinguishable from zero and is reported as a
number instead.

### One threshold cannot fit two solvers

A finite-volume diffusion integral on a closed domain conserves to round-off, so `1e-9`
there is a real problem. A Lennard-Jones fluid with a truncated potential exchanges a
little energy every time a particle crosses the cutoff, so `1e-4` there is expected —
and the solver's contract says so, in `known_non_conservation`.

The first thresholds called that `✖ not conserved`, which is how a reader learns to
ignore an indicator. The bands now describe the magnitude rather than pronouncing on
it, and only a genuine runaway is critical:

| Relative drift | Verdict |
|---|---|
| ≤ 1e-9 | ✔ conserved to round-off |
| ≤ 1e-6 | ✔ conserved to 1e-6 |
| ≤ 1e-3 | ! drifting slowly |
| ≤ 1e-1 | ▲ drifting |
| above, or non-finite | ✖ not conserved |

---

## Smaller decisions worth stating

**The legend goes below the plot, not inside it.** A floating legend has to land
somewhere in the plot area, and on a live chart there is often no free corner: the three
energy series of a Lennard-Jones run span +156 J to −368 J and reach every edge. An
earlier version tried to pick the emptiest corner by counting points; it only chose
which curve to hide. One row of height underneath can never hide anything.

**Text wears text tokens, never the series colour.** The swatch beside a legend entry
carries the identity; the label stays in secondary ink. Colouring the label to match its
series would make the legend the only place the mapping is stated *and* make it
unreadable to the reader who needs it most.

**Eight fixed categorical slots, never cycled.** A ninth series would repeat a colour
and two series would claim the same identity, so it goes to the values table with a
count of what was left out.

**Increasing y draws upward.** Image rows run top-to-bottom and grid rows run
bottom-to-top; the flip is in `field_to_image` with a test that asserts it.

**Non-finite cells are painted in the critical status colour and counted.** A NaN that
renders as "some colour" is a NaN that ships.

**Dark mode is selected, not flipped.** It has its own steps from the same ramps,
validated against the dark surface. The sequential ramp reverses its anchor — dark-on-
light becomes light-on-dark — because "more ink means more" inverts with the background.

---

## Layout

| File | Contents |
|---|---|
| `palette.rs` | OKLab conversion, the validated tokens, colourmap sampling, the tests that keep the ramps honest |
| `render.rs` | Field-to-image, particle scatter, colour scale, value formatting, status verdicts |
| `history.rs` | The observation timeline, unit grouping, decimation, drift bases |
| `app.rs` | Window layout, transport controls, the diagnostics panel |
| `lib.rs` / `main.rs` | `run(path)` and the `lattice-view` binary |

`eframe` / `egui` and `egui_plot` are the only external dependencies, and they are the
first in the project — spec §24.1 permits mature libraries *"where they do not define
the core semantics."* A window and an immediate-mode widget set do not; the colourmaps,
the drift arithmetic, and every rule above are ours and are tested here.

## What it does not do yet

No GPU field rendering (M4 — this draws through a CPU texture upload, which is fine at
64×64 and will not be at 768×384), no vector-field or streamline rendering, no particle
colouring by a per-particle quantity, no probe or measurement tools, no export of the
figure. §17's flux arrows, constraint visualization, and uncertainty bands arrive with
the domains that produce them.
