# The playground

`lattice-play` is the other half of the interface. `lattice-view` opens a model somebody
wrote and reports whether to believe it; the playground has no model at all — the scene
is built by clicking, and the physics underneath is the same validated code either way.
A ball dropped here obeys the contact solver whose elastic collision is measured against
theory in the validation suite. A reaction painted here balances its atoms because an
unbalanced one would not have compiled.

```powershell
cargo run --release -p lattice-playground                    # opens on physics
cargo run --release -p lattice-playground -- --mode reactions
cargo run --release -p lattice-playground -- --mode heat
```

Or, once built, `target\release\lattice-play.exe --mode heat`.

## The three modes

| mode | what you do | what it demonstrates |
|---|---|---|
| `physics` | drop boxes, balls and wedges; drag them; throw them | contacts, friction, restitution, stacking |
| `reactions` | paint two reactants together and warm them | mass action, Arrhenius, reaction–diffusion, the temperature coupling |
| `heat` | paint hot and cold spots | diffusion, and what "conserved" is a claim *about* |

Left-drag always acts with the selected tool. Right-click always removes — a body, a
brush stroke. That is the same in every mode, because a sandbox where the buttons mean
different things per mode is one nobody can put down and pick up again.

## Adding a mode

One file implementing [`Playground`](../crates/lattice-playground/src/mode.rs). The shell
needs no changes: it owns the window, the clock, the transport controls, the world/screen
mapping and the diagnostics panel, and knows nothing about rigid bodies or reactions. A
mode says what tools it offers, what knobs and checkboxes it wants, what timestep it
would like, what to draw, and what to publish.

Two things are worth getting right in a new mode:

- **`preferred_step`** — the shell drives the clock, so `pause`, `step` and the speed
  slider mean the same thing everywhere. A mode that ran at its own pace would break that.
- **`disturbances`** — bump it whenever the reader's pointer adds or removes a conserved
  quantity. See below.

## The inspector

Click a body and the panel shows what it is: mass, velocity, momentum, angular velocity,
inertia, pose, and — derived from those — speed and kinetic energy. Everything but the
derived pair can be edited while the scene runs.

**Selecting is not a tool.** Touching a thing is already how you say which thing you mean,
and a sandbox that made you switch modes to read a number would be asking you to plan
before poking. Clicking a body selects it; spawning one selects what was just made, so a
new block's numbers are on screen without a second click.

**Fields are drag-and-type boxes, not sliders.** The two are not interchangeable. A slider
is for sweeping a range to see what happens, which is what the scene-wide knobs above are
for. The inspector exists to set a value *exactly* — 3 kg, not 2.97 kg because that is
where the pixel landed.

Three rules the fields follow, each of which is a decision rather than an accident:

- **Mass and inertia move together.** Inertia is `∫r²dm`, so for a fixed shape it is
  proportional to mass. Editing mass alone would give a body that translates like a feather
  and spins like a boulder — reachable by typing, and not recoverable by eye. Inertia is
  separately editable for when that *is* what you want, because a flywheel and a disc of
  the same mass are genuinely different objects.
- **Momentum is a way of writing velocity.** It is not stored; setting it to `p` sets the
  velocity to `p/m` and leaves the mass alone. The alternative — changing mass to suit a
  requested momentum — would make one box silently move another.
- **Static bodies are shown and not editable.** A wall reads "static — infinite mass" with
  every field greyed. Making one dynamic by typing into a mass box would drop the floor out
  of the world, and the inertia it would then need is not recoverable from what is stored.

An edit counts as a [disturbance](#what-the-panel-will-and-will-not-claim), because reaching
in and setting a velocity moves the quantities the panel is watching. That is the reader's
hand, not solver drift.

The selection is a `BodyId`, never a slot. Slots move — `despawn` fills the hole by swapping
the last body into it — so a stored slot would quietly start describing a different object.
Generation checking catches a handle outliving its body; it does **not** catch a handle
outliving the whole store, so `reset` clears the selection explicitly. A fresh store
restarts its generation counters at zero, and without that line a handle held across a reset
resolves cleanly against an unrelated new body. `a_reset_clears_the_selection` found that
one.

## What the panel will and will not claim

Same rules as the model viewer. The panel reports what a domain *declared* as an
invariant and says plainly when it declared none:

> **conservation** — nothing here is conserved — a sandbox with gravity and walls has an
> external force and an infinite sink, so this mode claims none

That is the physics sandbox being honest rather than modest. Gravity is an external
force and the walls are static bodies with infinite mass; momentum is genuinely not
conserved, and a panel that claimed otherwise would be wrong on purpose.

The heat mode is the interesting one. An insulated plate conserves its heat integral
exactly, and the panel shows it holding to round-off. Tick **open the left edge** and the
plate becomes an open system — the same solver, the same field, one boundary changed —
and the panel stops calling it conserved. Being able to flip that with one checkbox is
the clearest demonstration in the engine of what the word is a claim about.

### Why painting does not count as drift

Painting heat onto a plate adds energy. Dropping a body adds mass. Neither is a solver
error, but both move the integral the panel is watching. A panel that reported
`drifting +2e-2` because someone used the brush would be blaming the solver for the
reader's own hand — which is the exact mistake the panel exists to catch, committed by
the panel itself.

So a mode counts its `disturbances`, and the shell throws the history away when that
number changes. Not a *correction* to the drift figure: a refusal to compute one across
the disturbance. Every drift number in the playground measures an undisturbed stretch of
simulation, or it is not shown.

## Drawing rules the modes inherit

The playground draws through the same `lattice-viewer` code as the model window, so the
same rules apply: a sequential ramp is one hue, a diverging ramp is two hues around a
neutral grey, a non-finite cell is painted in the reserved critical colour rather than
normalized away, and colour never carries a value that is not also written down.

Three decisions specific to these modes:

- **Concentration is floored at zero, not normalized to its own range.** An empty
  chamber has `min == max == 0`; a ramp with no range picks the middle, and the vessel
  renders as a solid mid-tone that reads as *uniformly full of the product it does not
  contain*. `field_to_image_above` starts the ramp at zero so empty looks empty.
- **Temperature is diverging about ambient, not about zero.** Anchored on zero, a plate
  at 300 K is one colour and the interesting variation is a rounding error against 300.
- **One field at a time.** The reactions mode draws the product *or* the temperature,
  never both. Two colour fields over the same cells is a picture where neither value can
  be read off.

Contact normals are available in the physics mode but **off by default** — a change of
audience rather than of opinion. The model viewer draws them because someone reading a
run wants to see what the solver is doing; a resting stack of nine boxes has twenty-eight
of them, and a reader who came to throw things at a wall did not ask the question they
answer.

## How the grab works

Not a joint. Joints hold *slot indices*, and a sandbox where things are constantly
created and destroyed permutes those slots underneath them. The grab instead applies a
spring force directly, every step, from a `BodyId` resolved fresh — a handle survives a
despawn and refuses to resolve if the body is gone, so an object deleted under the cursor
simply stops being held.

The spring is critically damped and **scaled to the body's mass** (`k = mω²`, `c = 2mω`),
so a heavy crate and a light ball follow the cursor at the same speed. Without the mass
scaling a sandbox feels wrong in a way that is hard to name: light things snap to the
cursor and heavy things lag, when what a hand does is the opposite. There is a test for
exactly this — `a_heavy_body_and_a_light_one_reach_the_cursor_at_the_same_rate` — because
it is the kind of property that is obvious when broken and invisible in a diff.

A throw uses the **pointer's** velocity, not the body's. A spring-held body lags the
cursor by roughly one time constant, and releasing at its own speed feels weak.

## Known gaps

- No keyboard shortcuts. Space for pause and `R` for reset are the obvious two.
- **Only the physics mode has an inspector.** `inspection` defaults to `None` on the trait,
  so heat and reactions simply do not offer one. A cell of the heat field and a species in
  the vessel are both selectable things with editable properties, and neither is wired up.
- **Friction and restitution are still scene-wide, not per body.** They live on the
  collider, and colliders are shared by shape and size, so a per-body surface needs a
  private collider per body and an exemption from `refresh_surfaces`. Worth doing; not done.
- The reaction mode's heat capacity is a constant, not a slider. Deliberate for now: a
  playground where the heat capacity is adjustable is one where "why did that not get
  hot" has two answers instead of one.
- No way to save or load a scene. Everything is built by clicking, every time.
- The interaction is covered by 60 unit tests driving `Playground::pointer` directly.
  Automated *end-to-end* capture of the real window is not reliable on this machine — see
  the note in `docs/development.md`.
