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
| `E04xx` | units: dimensional mismatch, affine scale misuse, value out of range |
| `E09xx` | not implemented yet — the message names the milestone |
| `W03xx` | warnings: ambiguous grouping, unknown setting, unsolved state |

Every rejection carries a source position and either a suggested fix or the rule it
enforces; `crates/lattice-compiler/tests/fixtures.rs` asserts both across the sixteen
models in `tests/invalid/`.

---

## Not implemented yet

These parse and are checked as far as possible, then produce an error naming the
milestone that will implement them. They are errors rather than warnings because a
model whose chemistry was silently dropped would run and produce confident wrong
numbers.

| Construct | Milestone |
|---|---|
| `reaction` blocks | M3 |
| `couple … -> …` | M3 |
| `material` blocks | M2 |
| `domain rigid2d`, `domain fluid2d` | M2 |
| `domain quantum2d`, `potential`, `wavepacket`, `detector` | M5 |
| user-defined expressions and force laws (spec §8.3) | M6 |
