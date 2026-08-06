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
| `lennard_jones(epsilon=, sigma=, cutoff=)` | cutoff defaults to 2.5σ |

`boundary:` is `periodic`, `reflective` or `open` (the default). A `region` is required
for a periodic or reflective boundary, and for any force with a cutoff.

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
| `W03xx` | warnings: ambiguous grouping, unknown setting, unsolved state |

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
| `domain quantum2d`, `potential`, `wavepacket`, `detector` | M5 |

Stochastic kinetics (Gillespie) is not implemented either, but it has no syntax of its
own — a `solve reactions(…) with gillespie(…)` would be the way in, and it reports an
unknown method.
| user-defined expressions and force laws (spec §8.3) | M6 |
