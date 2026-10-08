# The `.lattice` language

A reference for the project DSL. Spec §16.1 calls it the *"declarative canonical
model"* — the surface every other authoring route eventually produces.

```
project slab {
  fidelity: engineering_2d;
  duration: 400 second;

  grid bar { size: [64, 8]; extent: [1 meter, 0.125 meter]; }

  field temperature on bar = 273.15 kelvin {
    diffusivity: 1e-2 meter^2 / second;
    boundary:       insulated;
    boundary_left:  fixed(273.15 kelvin);
    boundary_right: fixed(373.15 kelvin);
  }

  solve heat(temperature) with crank_nicolson(dt=0.5 second);
  observe temperature every 20 second;
  visualize temperature as heatmap;
}
```

---

## Units are ordinary identifiers

This is the one design decision worth understanding before anything else.

There is no "unit mode" in the lexer. `kelvin` is an identifier, exactly like
`temperature` is. Name resolution decides what it means, in this order:

1. a declared name,
2. a unit from the registry.

Everything else follows. `100 / dt` divides by a parameter; `100 / second` is a
frequency. Both are the same grammar. An identifier that is neither gets a diagnostic
naming both possibilities, rather than "unknown unit `dt`" when the user meant a
parameter they forgot to declare.

The one thing this costs is that `35 kilojoule` has no operator between its terms.
Adjacency means multiplication, and it **binds tighter than `*` and `/`**:

| Written | Grouped as | Value |
|---|---|---|
| `35 kilojoule / mole` | `(35 × kJ) / mol` | 35 000 J/mol |
| `10 meter / 2 second` | `(10 × m) / (2 × s)` | 5 m/s |
| `9.81 meter second^-2` | `((9.81 × m) × s⁻²)` | 9.81 m/s² |

If adjacency bound *loosely*, that middle row would be `((10 m)/2)·s` = 5 m·s. A
dimensioned literal is one atom.

### `a / b * c` is ambiguous, and says so

`joule / mole * kelvin` means `J/(mol·K)` to a chemist and `(J/mol)·K` to every parser
ever written. They differ by K². Lattice applies the left-to-right rule and emits a
warning naming the grouping it used:

```
warning[W0300]: multiplication after division is grouped left to right
  --> model.lattice:7:12
   |
 7 |   value: 8.314 joule / mole * kelvin;
   |          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ read as `(a / b) * c`
   |
   = help: add parentheses to state the intended grouping
```

It warns only when the multiplicand is a bare unit. `rate / volume * count` on declared
names is ordinary arithmetic and needs no advice.

---

### A bare zero has every dimension but one

`0` with no unit is accepted wherever a quantity is expected, because zero is zero in
every unit: §25.2 writes `x=0` for a wall and `[-4 nanometer, 0]` for a centre.
Temperature is the exception — `0` could be 0 K or 0 °C, which differ by 273.15 — so a
bare zero temperature is still a dimension error.

## Grammar

```text
project  := 'project' IDENT '{' item* '}'
item     := setting | decl | field | domain | solve | couple | observe | visualize
setting  := IDENT ':' expr ';'
decl     := IDENT IDENT ( '{' setting* '}' | IDENT args? ';' )
field    := ('field' | 'species') IDENT ('on' IDENT)? ('=' expr)? (';' | '{' setting* '}')
solve    := 'solve' IDENT '(' args ')' 'with' IDENT '(' args ')' ';'
couple   := 'couple' path '->' path ('conserve' IDENT)? ';'
observe  := 'observe' expr ('every' expr)? ';'
visualize:= 'visualize' expr ('as' IDENT)? ';'
```

Declaration *kinds* are not reserved words. `grid`, `reaction`, `potential` and
`wavepacket` all take the `IDENT IDENT { … }` path, and the compiler decides which it
knows. Two things fall out: a new solver family needs no grammar change, and `grid:`
works as a setting key (spec §25.2 uses it as both).

Reserved words are only: `project`, `field`, `species`, `domain`, `solve`, `couple`,
`observe`, `visualize`, `import`, `on`, `with`, `as`, `every`, `conserve`, `true`,
`false` — and even those are accepted as setting keys, where they cannot be anything
else.

---

## Project settings

| Setting | Meaning |
|---|---|
| `dimensions:` | Must be 2. Spec P9 makes 2D the first target. |
| `fidelity:` | `interactive`, `engineering_2d`, `molecular`, `research` (§5.1). |
| `precision:` | `accurate64`, `deterministic64`. `mixed` compiles as `accurate64` with a note; `fast32` needs the GPU backend (M4). |
| `duration:` / `run:` | Physical time to simulate. `lattice run` needs this or `--duration`. |
| `timestep:` / `dt:` | Override the negotiated step. |

## Grids

```
grid chamber { size: [512, 256]; extent: [2 meter, 1 meter]; origin: [0 meter, 0 meter]; }
```

`size` is dimensionless cell counts; `extent` and `origin` are lengths.

## Fields and species

```
field temperature on bar = 273.15 kelvin { diffusivity: 1e-2 meter^2 / second; }
species A on chamber = left_half(1 mole / meter^2) { diffusivity: 2e-4 meter^2/second; }
```

**A field's dimension comes from its initial value.** `= 273.15 kelvin` makes a
temperature field, and every boundary, source and solver parameter is then checked
against that. There is no separate dimension declaration to keep in sync.

`species` is `field` with an intent; both compile to the same transport solver today.
The distinction starts to matter when reactions can consume them (M3).

### Initializers

| Form | Arguments |
|---|---|
| a bare quantity | the value, uniform everywhere |
| `uniform(v)` | same |
| `left_half(v)`, `right_half(v)`, `bottom_half(v)`, `top_half(v)` | the filled value; zero elsewhere |
| `gaussian(center=, sigma=, peak=)` | centre and sigma are lengths, peak is in field units |
| `disc(center=, radius=, value=)` | value inside, zero outside |
| `ramp_x(from=, to=)`, `ramp_y(from=, to=)` | linear across the domain |

### Boundaries

`boundary:` sets all four sides; `boundary_left`, `boundary_right`, `boundary_bottom`
and `boundary_top` override individually.

| Form | Meaning |
|---|---|
| `periodic` | wraps; must be declared on **both** edges of an axis |
| `insulated` | zero outward gradient |
| `fixed(v)` | prescribed value at the boundary *face* |
| `flux(gradient=)` | prescribed outward derivative, in field units per metre |
| `robin(coefficient=, ambient=)` | convective exchange; coefficient is 1/length |

Signs follow the outward normal, so the same positive gradient describes opposite
physical fluxes on opposite edges.

## Particles

```
particles atoms {
  count:   256;
  region:  [22.4 meter, 22.4 meter];
  spacing: 1.4 meter;
  mass:    1 kilogram;
  speed:   0.4 meter / second;
  seed:    20260805;
  boundary: periodic;
  force: lennard_jones(epsilon=1 joule, sigma=1 meter);
}
```

Particles are placed on a square lattice at `origin + (i + 0.5) × spacing`, with
velocities drawn from a normal distribution of width `speed` and then shifted so total
momentum is exactly zero — which is what makes the momentum diagnostic meaningful.

`force:` may appear more than once; the contributions add.

| Force | Arguments |
|---|---|
| `gravity` | a magnitude (downward) or a `[ax, ay]` vector; defaults to Earth |
| `drag(coefficient=)` | mass per time |
| `harmonic_well(center=, stiffness=)` | stiffness in N/m |
| `lennard_jones(epsilon=, sigma=, cutoff=, truncation=)` | cutoff defaults to 2.5σ; `truncation` is `energy_shift` (default) or `force_shift` |
| `soft_repulsion(stiffness=, range=)` | `U = ½k(d − r)²` inside `range`; stiffness in N/m |
| `coulomb(cutoff=, damping=)` | `k qᵢqⱼ/r` by the damped shifted force sum; needs `charge:`. `damping` is an inverse length and defaults to `2.4 / cutoff` |

`boundary:` is `periodic`, `reflective` or `open` (the default). A `region` is required
for a periodic or reflective boundary, and for any force with a cutoff.

### Molecular settings

Spec §12.4's molecular module is the same `particles` block with more attached. A
complete one is [`examples/argon.lattice`](../examples/argon.lattice); a bonded chain is
[`examples/polymer.lattice`](../examples/polymer.lattice).

```
particles chain {
  count:   64;
  spacing: 3.8 angstrom;
  layout:  serpentine;
  mass:    100 dalton;
  temperature: 300 kelvin;
  seed:    64;

  bonds:  chain(stiffness=100 newton/meter);
  angles: chain(stiffness=2e-20 joule, angle=180 degree);
  thermostat: langevin(temperature=300 kelvin, friction=1 / picosecond);
  skin: 1 angstrom;
  analysis: rdf(bins=100, range=1.5 nanometer, every=10, after=1000);
}
```

| Setting | Meaning |
|---|---|
| `temperature:` | initial velocities from a Maxwell distribution, rescaled so the instantaneous temperature is exactly this with zero net momentum. Excludes `speed:` |
| `layout:` | `lattice` (default) fills rows left to right; `serpentine` reverses every other row, so consecutive particles are always one `spacing` apart — what a chain needs to start unstrained |
| `bonds:` | harmonic bonds. `chain(…)`, `ring(…)` or `pairs([[i, j], …], …)` over particle indices in placement order; `stiffness=` in N/m, `length=` defaults to `spacing`. Bonded pairs are excluded from pair forces unless `pair_forces=included` |
| `angles:` | harmonic angles. `chain(…)`, `ring(…)` or `triples([[i, j, k], …], …)`; `stiffness=` in J/rad², `angle=` the rest angle at the middle particle |
| `thermostat:` | `langevin(temperature=, friction=)` — BAOAB, samples the canonical ensemble, needs `velocity_verlet` — or `velocity_rescale(temperature=, relaxation=)` (alias `berendsen`), which does not |
| `skin:` | cache pairs in a Verlet list out to cutoff + skin, rebuilt when any particle has moved half the skin. Results agree with rebuilding every step to round-off |
| `analysis:` | `rdf(bins=, range=, every=, after=)` — a radial distribution sampled every `every` steps once `after` steps have passed. `range` may not exceed half a periodic box |
| `charge:` | a charge on every particle, or `alternating(q)` — `+q` and `−q` on alternate lattice sites, the 2D rock-salt checkerboard. Only `coulomb` reads it |

A thermostatted run publishes total energy as a metric, not an invariant: energy flows
through the bath on purpose, and reporting its drift as a conservation failure would be
wrong. The radial distribution is published as a *curve* — `g(r)` against `r` — in the
run artifact and at the end of `lattice run`.

### Coulomb

A plain cutoff ruins `1/r`: the neglected tail is not small, and a truncated sum over a
neutral crystal oscillates with whichever shell of charges the cutoff cuts through.
`coulomb` is Fennell and Gezelter's damped shifted force sum instead — the real-space
half of an Ewald sum, `erfc(αr)/r`, shifted so that its energy and force both vanish at
the cutoff. Nothing jumps when a pair crosses it, so a run without a thermostat conserves
energy to the integrator's error alone. What it leaves out is Ewald's reciprocal-space
half, which assumes the system is neutral on the scale of the cutoff; the default
`αR = 2.4` puts the real-space truncation near `erfc(2.4) ≈ 7e-4` of a pair's energy.

The charges are point charges in the plane interacting by the three-dimensional law —
ions on a surface — not two-dimensional electrostatics, where line charges interact by
`−ln r`. A complete model is [`examples/salt.lattice`](../examples/salt.lattice).

Three combinations compile and run and give a result nobody asked for, so they warn:
a `coulomb` force with no charges (`W0311`), a set that is not neutral — a uniform charge,
or an odd count on the checkerboard (`W0311`) — and alternating charges with no
`lennard_jones` or `soft_repulsion` core to keep opposite charges apart (`W0312`).

## Rigid bodies

A particle set is written as one declaration with a `count`, because its members are
interchangeable. Rigid bodies are the opposite: a scene is *the ground*, *the ramp*,
*the crate*, each with its own shape and role, and half of them are named by a joint.
So each gets its own declaration.

```
domain rigid2d yard {
  gravity:    9.80665 meter / second^2;   // or a [gx, gy] vector
  iterations: 10;                          // contact solver passes per step
}

material concrete {
  density:     240 kilogram / meter^2;
  restitution: 0.1;
  friction:    0.8;
}

body ground {
  shape:    box(10 meter, 0.3 meter);
  at:       [0 meter, -3 meter];
  material: concrete;
  motion:   static;
}

body crate {
  shape:    box(0.4 meter, 0.4 meter);
  at:       [-6.5 meter, 2 meter];
  angle:    0.35;
  material: ice;
}

joint arm {
  kind:     pin;
  bodies:   [hook, bob];
  anchor_b: [-1.5 meter, 0 meter];
}

solve rigid(yard) with sequential_impulse(dt = 0.004 second);
```

The `domain rigid2d` block is optional; without one the world takes standard gravity
and the solver's defaults. Every declared body joins a single world.

### Density is per unit **area**

`kilogram / meter^2`, not `kilogram / meter^3`. This is a 2D world, and giving it an
implied thickness would make every mass wrong by a factor nobody declared — §14.1 puts
that kind of conversion in a port rather than hiding it in a solver. A model that
writes `7850 kilogram / meter^3` is told which dimension was expected and which it
gave. The familiar 3D densities times a 10 cm slab are usually what a reader means.

### Shapes

| Shape | Arguments |
|---|---|
| `circle(radius)` | one length |
| `box(half_width, half_height)` | half-extents, so `box(1 m, 1 m)` is 2 m square |
| `polygon([x, y], …)` | 3 to 8 vertices, **counter-clockwise**, convex |
| `segment(half_length)` | along the body's local x axis; has no area |

A polygon is recentred on its own centroid, because every impulse formula assumes the
body's transform *is* its centre of mass. Clockwise winding is refused rather than
silently reversed: it is what a model authored for a y-down coordinate system produces,
and that model's gravity is about to be wrong too. A segment encloses no area, so
density gives it no mass — it must be `motion: static;` or carry an explicit `mass:`.

### Body settings

| Setting | Meaning |
|---|---|
| `shape` | required |
| `at` | centre of mass, m |
| `angle` | radians (dimensionless in SI, so `0.35` and `0.35 radian` are the same) |
| `material` | a declared material; without one, a sensible default |
| `motion` | `static` or `dynamic` (the default) |
| `mass` | an explicit total mass, overriding the density |
| `velocity`, `spin` | initial linear and angular velocity |

### Joints

| `kind` | Holds | Extra settings |
|---|---|---|
| `distance` | two anchors a fixed distance apart — a rod | `length` (defaults to the initial separation) |
| `rope` | the same, but pulls only | `length` |
| `pin` | two anchors coincident — a hinge | — |
| `spring` | a damped force, not a constraint | `stiffness`, `damping`, `length` |
| `motor` | a target relative angular velocity | `speed`, `torque` (required) |

`anchor_a` and `anchor_b` are in each body's **local** frame, so a joint follows its
bodies as they move and turn. A motor's `torque` is required: an unbounded budget would
move any load, which is a servo nobody has.

## Chemistry

```
domain chemistry chamber {
  grid: vessel;            // every species in a mixture reacts on one grid
}

species fuel on vessel = left_half(40 mole / meter^2) {
  formula:     CH4;
  diffusivity: 2e-5 meter^2 / second;
  boundary:    insulated;
}

reaction combustion {
  reactants:         fuel + 2 oxidiser;
  products:          product + 2 water;
  rate:              30 meter^4 / (mole^2 second);
  activation_energy: 30 kilojoule / mole;
  enthalpy:          -8.0e2 kilojoule / mole;     // negative is exothermic
}

solve reactions(chamber) with strang(dt = 0.02 second);
```

Stoichiometry needed no new syntax. `2 H2 + O2` already parses: `+` is an infix
operator and `2 H2` is a juxtaposition, which binds tighter — the same rule that makes
`10 meter / 2 second` a velocity rather than a metre-second.

### Concentrations are per unit **area**

`mole / meter^2`. This is a 2D engine, and a concentration per unit volume would need a
thickness nobody declared. A second-order rate constant is therefore in `m^2/(mol s)`,
which is unfamiliar and correct.

### Two things the compiler checks that nothing else would

**Atom and charge balance.** `H2 + O2 -> H2O` integrates perfectly happily and destroys
47% of the mass it touches; the only symptom is a conservation check failing several
layers away. When every species states a `formula` the reaction is checked, and the
diagnostic names the element:

```text
error[E0212]: reaction `typo` does not balance
26 |   reaction typo {
   |            ^^^^ O does not balance: 2 on the left, 1 on the right
```

A species with no `formula` makes the check impossible, which is a warning rather than
an error — an abstract `A -> B` model is a legitimate thing to write.

**The rate constant's unit against the reaction's order.** An order-`n` constant is in
`(m^2/mol)^(n-1)/s`. Rate constants span twenty orders of magnitude, so nothing about
the *number* says which order was meant. The compiler works the order out from the
reactants:

```text
error[E0400]: `rate` for a reaction of order 2 has the wrong dimension
30 |     rate:      1e3 / second;
   |                ^^^^^^^^^^^^ expected m^2/s·mol, found frequency (1/s)
```

### Species settings

| Setting | Meaning |
|---|---|
| `formula` | a chemical formula such as `H2O`; gives the molar mass and enables balance checking |
| `charge` | whole elementary charges |
| `molar_mass` | overrides what the formula implies — a warning if the two disagree |
| `diffusivity` | as for any field; zero for an immobile species |
| `boundary` | as for any field |

### Reaction settings

| Setting | Meaning |
|---|---|
| `reactants` | required; a sum of species with optional coefficients |
| `products` | a sum of species, or omitted for a sink |
| `rate` | required; in `(m^2/mol)^(order-1)/s` |
| `reverse_rate` | makes the reaction reversible; the equilibrium constant is the ratio |
| `activation_energy` | Arrhenius: `k(T) = A exp(-Ea/RT)`, with the declared `rate` as `A` |
| `enthalpy` | J/mol of extent. **Negative is exothermic** — the chemistry convention |

## Coupling

```
field temperature on vessel = 300 kelvin {
  diffusivity:   1.2e-5 meter^2 / second;
  heat_capacity: 4.0e5 joule / (meter^2 kelvin);
}

couple chamber.heat_release -> temperature.source conserve energy;
couple temperature.values   -> chamber.temperature;
```

An edge connects a port one domain publishes to a port another consumes. `conserve`
names the quantity that crosses, which puts every transfer in the ledger; leave it out
for an edge that carries a *parameter* rather than a transfer — a temperature reaching a
rate constant changes how fast a reaction goes, it does not move energy.

### The conversion is derived, not written

A reaction publishes `W/m^2`; a heat source is in `K/s`. Those are not the same
quantity, and the factor between them is an areal heat capacity that belongs to neither
domain — it is a property of the material between them. Wire them straight together and
the run compiles, executes, looks entirely plausible, and is wrong by that factor.

So the model declares the material property and the compiler does the algebra. A
coupling with no way to convert says so, and says what is missing:

```text
error[E0400]: this coupling needs a unit conversion the model has not declared
36 |   couple mixture.heat_release -> temperature.source conserve energy;
   |   ^^^ areal power density (kg/s^3) does not convert to K/s
   = note: the factor between them would have to be in s^2·K/kg
   = help: for a heat coupling, add `heat_capacity: 4e5 joule / (meter^2 kelvin);`
```

### Ports

| Port | Domain | Direction | Unit |
|---|---|---|---|
| `values` | any field solver | publishes | the field's own unit |
| `source` | any field solver | consumes | field units per second |
| `heat_release` | a chemistry mixture | publishes | `W/m^2` |
| `temperature` | a chemistry mixture | consumes | `K` |

`values`, not `field` — the grammar reserves `field`, and a port a model cannot name is
a port that does not exist.

### What the ledger reports

```text
coupling ledger
  chamber -> temperature      1.774271e5 total energy over 2000 exchanges
  a staggered coupling always has one exchange in flight, so the last of this has
  not landed
```

That last line is not an apology. A staggered coupling (§14.2) advances each domain on
the latest state it was given, so the final step's transfer is recorded and never
delivered. The shortfall is first order in the timestep, and the validation suite
measures it rather than tolerating it.

## Quantum

Spec §13.1's built-in module: one particle's wavefunction on a 2D grid, under
`iħ ∂ψ/∂t = −(ħ²/2m)∇²ψ + Vψ`. A complete scene is
[`examples/double_slit.lattice`](../examples/double_slit.lattice); spec §25.2's own
version compiles as written.

```
domain quantum2d q {
  grid: [256, 128];
  extent: [12 nanometer, 6 nanometer];
  mass: electron_mass;
  boundary: absorbing(width=0.8 nanometer);
  integrator: split_step_fourier;
}

potential barrier {
  shape: vertical_wall(x=0, thickness=0.15 nanometer);
  slits: [(-1.0, 0.35), (1.0, 0.35)] nanometer;
  height: 20 electronvolt;
}

wavepacket initial {
  center: [-3 nanometer, 0];
  momentum: [1.3e-24 kilogram*meter/second, 0];
  sigma: [0.6 nanometer, 1.5 nanometer];
}

detector screen at x=4.5 nanometer;
```

The grid is **centred on the origin** unless an `origin:` is given — §25.2 puts its
wall at `x = 0` and its packet at `x = −4 nm` on a 12 nm grid. A quantum domain carries
its integrator in its own block, so there is no `solve` statement for it, and a project
holds at most one: potentials, packets and detectors belong to it without naming it.

| Domain setting | Meaning |
|---|---|
| `grid:` | cells along x and y. One row (`[n, 1]`) is a line, and the y direction drops out |
| `extent:`, `origin:` | size, and the lower-left corner (default: centred) |
| `mass:` | the particle's mass |
| `integrator:` | `split_step_fourier(dt=…)` — spectral, periodic, the default — or `crank_nicolson(dt=…, tolerance=…)` — finite difference, a box with `ψ = 0` walls. Leave `dt` out to get one sized for the grid |
| `boundary:` | `periodic` (split-step's own), `walls` (Crank–Nicolson's own), or `absorbing(width=…, strength=…)` in front of either. The strength defaults to `10ħv/width`, with `v` a packet's group velocity plus three of its velocity spreads |

| Potential `shape:` | Settings |
|---|---|
| `vertical_wall(x=, thickness=)` | `height:`; optional `slits:` as `(centre, width)` pairs |
| `rectangle(x=[a, b], y=[c, d])` | `height:` — positive for a barrier, negative for a well |
| `harmonic(center=[…], omega=…)` | none; the stiffness is `m ω²` for the domain's mass |

Potentials add. A `wavepacket` is a minimum-uncertainty Gaussian with position spread
`sigma` (one value or `[σx, σy]`) and mean `momentum`; several superpose, and their sum
is normalized. A `detector` is a screen parallel to y that integrates the probability
current through it; `lattice run` plots what it collected — the interference pattern —
and the run artifact stores it as a curve.

A detector can also count single arrivals, which is how §13.1's *measurement-inspired
sampling* reaches the language:

```
detector screen at x=4.5 nanometer, clicks=4000, seed=1;
```

fires 4000 particles and records where each one lands, drawn from the forward current
through the screen; `seed` (default 0) makes the draw reproducible, and is an error
without `clicks`. The histogram is a second curve, `<domain>.<detector>.clicks`, on the
same axes as the exact pattern, and the count detected so far is an observation. The
screen samples the flux without collapsing the state: the arrivals have the right
statistics, and the wavefunction evolves as one undisturbed particle.

| Render channel | What `lattice-view` draws |
|---|---|
| `probability_density` | the density on a sequential ramp |
| `phase` | `arg ψ` on a phase wheel — fixed lightness and chroma, the hue turning once around — faded toward the background by the amplitude, so a phase where there is no wavefunction recedes; the wheel is its legend |
| `probability_current` | the current's magnitude, with arrows for its direction |
| `potential` | `V` |

The terminal viewer draws the current's magnitude and says what it cannot draw: a
character ramp has two ends, and a phase does not.

With an absorbing boundary the norm falls on purpose. The domain publishes
`probability_norm` and `probability_absorbed` as metrics and their sum,
`probability_accounted`, as the invariant.

### Three warnings

Each of these describes an experiment someone might run on purpose, so they warn
rather than refuse. Spec §25.2's own example triggers two of them.

| Code | When | Why it matters |
|---|---|---|
| `W0308` | a packet's mean kinetic energy is above a barrier's height | it mostly passes over, rather than tunnelling or diffracting |
| `W0309` | a packet's wavenumbers reach the grid's Nyquist limit `π/Δ` | those momenta alias onto the opposite direction |
| `W0310` | split-step's `dt` puts more than 2 radians of kinetic phase per step on the grid's top mode, with a wall or rectangle present | sharp edges reach the top modes, where the splitting error lives; a validated tunnelling case was 1.9% wrong at 7.5 radians |

## Solving

```
solve heat(temperature) with crank_nicolson(dt=0.5 second);
solve diffusion(A, B)   with crank_nicolson(dt=0.5 second);
solve dynamics(atoms)   with velocity_verlet(dt=0.005 second);
```

| Solver | Targets | Methods |
|---|---|---|
| `heat`, `diffusion`, `transport` | fields and species | `explicit`, `crank_nicolson`, `backward_euler` |
| `dynamics` | particle sets | `velocity_verlet`, `semi_implicit_euler`, `explicit_euler` |
| `rigid` | a rigid world | `sequential_impulse` |
| `reactions` | a chemistry mixture | `strang` |

`rigid` offers one method because a sequential-impulse contact solver assumes
semi-implicit Euler; offering velocity Verlet alongside it would be offering something
that does not work.

An explicit scheme past its stability limit is **refused at compile time**, with the
limit and the fix in the message. Clamping silently would produce a run that finishes
and is wrong.

## Observing and visualizing

```
observe temperature every 20 second;
observe probability_norm;              // every step
visualize temperature as heatmap;
visualize probability_density;         // the domain picks an encoding
```

---

## Diagnostic codes

| Range | Meaning |
|---|---|
| `E00xx` | lexical: bad character, unterminated comment or string |
| `E01xx` | syntax: unexpected token, reserved word as a name, bad exponent |
| `E02xx` | resolution: unknown name, duplicate declaration, unknown setting |
| `E02xx` | also: unknown keyword value (`E0208`) |
| `E021x` | geometry and chemistry: unknown builtin (`E0210`), invalid shape or formula (`E0211`), unbalanced reaction (`E0212`) |
| `E04xx` | units: dimensional mismatch, affine scale misuse, value out of range |
| `E09xx` | not implemented yet — the message names the milestone |
| `W03xx` | warnings: ambiguous grouping, unknown setting, unsolved state, the quantum module's physics checks (`W0308`–`W0310`), and charges that do not suit a Coulomb law (`W0311`–`W0312`) |

Every rejection carries a source position and either a suggested fix or the rule it
enforces; `crates/lattice-compiler/tests/fixtures.rs` asserts both across the
twenty-six models in `tests/invalid/`.

---

## Not implemented yet

These parse and are checked as far as possible, then produce an error naming the
milestone that will implement them. They are errors rather than warnings because a
model whose chemistry was silently dropped would run and produce confident wrong
numbers.

| Construct | Milestone |
|---|---|
| `domain fluid2d` | M4 |
| user-defined expressions and force laws (spec §8.3) | M6 |

Stochastic kinetics (Gillespie) is not implemented either, but it has no syntax of its
own — a `solve reactions(…) with gillespie(…)` would be the way in, and it reports an
unknown method.
