//! The reaction sandbox: paint chemicals and watch them find each other.
//!
//! Two species, `A` and `B`, react where they overlap to make `C`, releasing heat.
//! The heat raises the local temperature and the temperature raises the rate through
//! Arrhenius, so a patch of mixture that gets going gets going faster — which is what
//! makes a reaction front a front rather than a smudge.
//!
//! # Why the temperature is a field the mixture reads
//!
//! It would have been simpler to hold one temperature for the whole sandbox. It would
//! also have been the wrong physics: a hot spot that instantly warmed the entire
//! chamber could never produce a front, and the most interesting thing this mode does
//! would be missing. The temperature is a real diffusing field, and the mixture reads
//! it cell by cell — the same arrangement `examples/chamber.lattice` uses, minus the
//! model file.
//!
//! # The conversion is still a material property
//!
//! Heat comes out of the reaction in `W/m²` and has to reach the field in `K/s`. The
//! factor is an areal heat capacity, and it is [`HEAT_CAPACITY`] here rather than
//! anything the sandbox derives — a slider would be fun, but a playground where the
//! heat capacity is adjustable is a playground where "why did that not get hot" has two
//! answers instead of one.

use eframe::egui;
use lattice_domain_chemistry::{
    RateLaw, ReactingMixture, Reaction, ReactionNetwork, Species, Term,
};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, Domain, Grid2d, Observations, ScalarField, StepContext};
use lattice_viewer::render::WorldView;
use lattice_viewer::{render, Colormap, Palette};

use crate::mode::{Knob, Playground, Pointer, Toggle, Tool};

/// Areal heat capacity of the chamber contents, J/(m²·K).
const HEAT_CAPACITY: f64 = 6.0e4;

/// Grid resolution. Enough to see a front, small enough that a reaction step across the
/// whole grid stays inside a frame.
const CELLS: usize = 72;

/// The temperature the chamber starts at, K — and the midpoint of the diverging map
/// when the temperature is the field on screen.
const AMBIENT: f64 = 300.0;

/// The chamber, m.
const EXTENT: [f64; 2] = [1.0, 1.0];

/// How much a single click deposits, mol/m².
const BRUSH_STRENGTH: f64 = 60.0;

/// What the brush paints.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Brush {
    ReactantA,
    ReactantB,
    Warmth,
}

impl Brush {
    const ALL: [Brush; 3] = [Brush::ReactantA, Brush::ReactantB, Brush::Warmth];

    fn label(self) -> &'static str {
        match self {
            Brush::ReactantA => "paint A",
            Brush::ReactantB => "paint B",
            Brush::Warmth => "warm it up",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Brush::ReactantA => "drag to lay down reactant A; it reacts wherever it meets B",
            Brush::ReactantB => "drag to lay down reactant B; right-click erases",
            Brush::Warmth => "drag to add heat — hotter mixture reacts faster (Arrhenius)",
        }
    }
}

/// The reaction sandbox.
#[derive(Debug)]
pub struct ReactionPlayground {
    mixture: ReactingMixture,
    heat: HeatDomain,
    arena: Arena,
    grid: Grid2d,
    /// Staging buffers for the two coupling directions, reused every step.
    heat_buffer: ScalarField,
    temperature_buffer: ScalarField,
    brush: Brush,
    brush_radius: f64,
    rate: f64,
    activation: f64,
    hover: Option<[f64; 2]>,
    /// Draw the temperature field instead of the product.
    ///
    /// One or the other, never both. Two colour fields over the same cells is a picture
    /// where neither value can be read off, which is the rainbow mistake with extra
    /// steps — and the temperature is the more interesting of the two exactly when the
    /// question is *why* a front is moving.
    show_temperature: bool,
    /// Bumped every time the brush adds material or heat; see
    /// [`Playground::disturbances`].
    disturbances: u64,
    painted: f64,
}

impl Default for ReactionPlayground {
    fn default() -> Self {
        ReactionPlayground::new()
    }
}

impl ReactionPlayground {
    /// An empty chamber at room temperature.
    pub fn new() -> ReactionPlayground {
        let grid = Grid2d::new(CELLS, CELLS, EXTENT);
        let mut playground = ReactionPlayground {
            mixture: ReactingMixture::new("chamber", grid, network(4.0, 25_000.0)),
            heat: room_temperature(grid),
            arena: Arena::with_capacity(1 << 17),
            grid,
            heat_buffer: ScalarField::new(&grid, 1),
            temperature_buffer: ScalarField::new(&grid, 1),
            brush: Brush::ReactantA,
            brush_radius: 0.08,
            rate: 4.0,
            activation: 25_000.0,
            hover: None,
            show_temperature: false,
            disturbances: 0,
            painted: 0.0,
        };
        playground.rebuild();
        playground
    }

    /// Rebuild the mixture after a rate change.
    ///
    /// A rate constant is baked into the network at construction, so changing one means
    /// a new network. The concentrations are carried across, because a slider that
    /// wiped the scene would be a slider nobody dares touch.
    fn rebuild(&mut self) {
        let mut fresh = ReactingMixture::new("chamber", self.grid, network(self.rate, self.activation));
        for index in 0..3 {
            let (Some(from), Some(into)) =
                (self.mixture.concentration(index), fresh.concentration_mut(index))
            else {
                continue;
            };
            into.copy_interior_from(from);
        }
        self.mixture = fresh;
    }

    /// Add `amount` in a soft disc around `at`.
    ///
    /// Feathered rather than a hard circle: a hard edge in a concentration field is a
    /// discontinuity the diffusion solver then spends several steps smoothing, and the
    /// visible result is a brush stroke that blurs *after* you draw it.
    fn paint(&mut self, at: [f64; 2], amount: f64) {
        self.disturbances += 1;
        let radius = self.brush_radius;
        let grid = self.grid;
        let field = match self.brush {
            Brush::ReactantA => self.mixture.concentration_mut(0),
            Brush::ReactantB => self.mixture.concentration_mut(1),
            Brush::Warmth => Some(self.heat.field_mut()),
        };
        let Some(field) = field else { return };

        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let centre = grid.cell_center(i, j);
                let distance = (centre[0] - at[0]).hypot(centre[1] - at[1]);
                if distance > radius {
                    continue;
                }
                // Cosine falloff: one at the middle, zero and flat at the rim.
                let falloff = 0.5 * (1.0 + (core::f64::consts::PI * distance / radius).cos());
                let value = field.get(i, j) + amount * falloff;
                field.set(i, j, value.max(0.0));
            }
        }
    }

    /// Move heat into the temperature field and temperature back into the mixture.
    ///
    /// The same two edges `examples/chamber.lattice` declares, applied directly. A
    /// `Coupler` would work here too, and would carry the ledger — but a sandbox has no
    /// run to reconcile at the end of, and the indirection would buy nothing a reader
    /// could see.
    fn exchange(&mut self, dt: f64) {
        self.heat_buffer.copy_interior_from(self.mixture.heat_release());
        for value in self.heat_buffer.as_mut_slice() {
            *value /= HEAT_CAPACITY;
        }
        let mut source = ScalarField::new(&self.grid, 1);
        source.copy_interior_from(&self.heat_buffer);
        self.heat.set_source(source);

        self.temperature_buffer.copy_interior_from(self.heat.field());
        self.mixture.set_temperature_field(&self.temperature_buffer);
        let _ = dt;
    }
}

/// `A + B -> C`, exothermic and temperature-dependent.
fn network(rate: f64, activation: f64) -> ReactionNetwork {
    let mut network = ReactionNetwork::new();
    network.add_species(Species::new("A").with_formula("C").unwrap().with_diffusion(4e-4));
    network.add_species(Species::new("B").with_formula("O").unwrap().with_diffusion(4e-4));
    network.add_species(Species::new("C").with_formula("CO").unwrap().with_diffusion(2e-4));
    network.add_reaction(
        Reaction::new(
            "combine",
            vec![Term::new(0, 1.0), Term::new(1, 1.0)],
            vec![Term::new(2, 1.0)],
            RateLaw::mass_action(rate).with_activation_energy(activation),
        )
        .with_enthalpy(-2.0e5),
    );
    network
}

fn room_temperature(grid: Grid2d) -> HeatDomain {
    HeatDomain::new("temperature", grid, Diffusivity::Uniform(2e-4))
        .with_scheme(TimeScheme::CrankNicolson)
        .with_display_unit("K")
        .with_uniform_initial(AMBIENT)
}

impl Playground for ReactionPlayground {
    fn name(&self) -> &'static str {
        "reactions"
    }

    fn description(&self) -> &'static str {
        "paint two chemicals together and watch them react and heat"
    }

    fn tools(&self) -> Vec<Tool> {
        Brush::ALL.iter().map(|brush| Tool::new(brush.label(), brush.hint())).collect()
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![
            Knob::new("brush", self.brush_radius, 0.02, 0.3, "m"),
            Knob::logarithmic("rate", self.rate, 0.05, 200.0, "m^2/(mol s)"),
            Knob::new("activation", self.activation / 1000.0, 0.0, 60.0, "kJ/mol"),
        ]
    }

    fn set_knob(&mut self, index: usize, value: f64) {
        match index {
            0 => self.brush_radius = value,
            1 => {
                self.rate = value;
                self.rebuild();
            }
            2 => {
                self.activation = value * 1000.0;
                self.rebuild();
            }
            _ => {}
        }
    }

    fn toggles(&self) -> Vec<Toggle> {
        vec![Toggle::new(
            "show temperature",
            self.show_temperature,
            "draw the temperature field instead of the product — the field the rate law \
             is reading, and the reason a front is a front",
        )]
    }

    fn set_toggle(&mut self, index: usize, value: bool) {
        if index == 0 {
            self.show_temperature = value;
        }
    }

    fn preferred_step(&self) -> f64 {
        0.02
    }

    fn step(&mut self, dt: f64) {
        let mut ctx = StepContext::new(&mut self.arena);
        self.mixture.prepare(&mut ctx);
        self.mixture.advance(dt, &mut ctx);
        self.heat.prepare(&mut ctx);
        self.heat.advance(dt, &mut ctx);
        drop(ctx);
        self.exchange(dt);
    }

    fn pointer(&mut self, pointer: Pointer, tool: usize) {
        self.hover = Some(pointer.world);
        self.brush = Brush::ALL[tool.min(Brush::ALL.len() - 1)];

        // Painting on press *and* while held, so a click deposits and a drag draws.
        if pointer.pressed || pointer.held {
            let amount = match self.brush {
                Brush::Warmth => 12.0,
                _ => BRUSH_STRENGTH,
            };
            self.paint(pointer.world, amount);
            if self.brush != Brush::Warmth {
                self.painted += amount;
            }
        }
        if pointer.secondary {
            // Erase: the same brush with a large negative amount, clamped at zero by
            // `paint`. For warmth, back toward room temperature rather than to nothing —
            // a chamber at 0 K is not a thing anyone meant to draw.
            let amount = match self.brush {
                Brush::Warmth => -12.0,
                _ => -BRUSH_STRENGTH * 4.0,
            };
            self.paint(pointer.world, amount);
        }
    }

    fn pointer_left(&mut self) {
        self.hover = None;
    }

    fn draw(&self, painter: &egui::Painter, view: WorldView, palette: &Palette) {
        // One field at a time. The product on a one-hue ramp, because concentration has
        // a floor and no meaningful middle; the temperature diverging about ambient,
        // because the question there is which way a cell is *off* room temperature and
        // a one-hue ramp cannot answer it.
        let rendered = if self.show_temperature {
            render::field_to_image_about(
                self.heat.field(),
                Colormap::Diverging,
                palette.mode,
                palette,
                AMBIENT,
            )
        } else {
            // Floored at zero, not normalized to the data's own range: an empty chamber
            // has min == max == 0, and a ramp with no range to speak of would paint the
            // whole thing mid-tone — a picture that says the vessel is uniformly full of
            // the product it does not contain.
            render::field_to_image_above(
                self.mixture.concentration(2).expect("three species"),
                Colormap::Sequential,
                palette.mode,
                palette,
                0.0,
            )
        };
        let rect = view.rect();
        let handle = painter.ctx().load_texture(
            "reaction",
            rendered.image.clone(),
            egui::TextureOptions::NEAREST,
        );
        painter.image(
            handle.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // The brush, so its size means something before anything is painted.
        if let Some(at) = self.hover
            && let Some(centre) = view.project(at)
        {
            let radius = self.brush_radius as f32 * view.scale();
            painter.circle_stroke(
                centre,
                radius,
                egui::Stroke::new(1.0, palette.text_muted),
            );
        }
    }

    fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        (self.grid.origin(), EXTENT)
    }

    fn observe(&self, out: &mut Observations) {
        self.mixture.observe(out);
        self.heat.observe(out);
    }

    fn disturbances(&self) -> u64 {
        self.disturbances
    }

    fn reset(&mut self) {
        self.mixture = ReactingMixture::new("chamber", self.grid, network(self.rate, self.activation));
        self.heat = room_temperature(self.grid);
        self.painted = 0.0;
    }

    fn status(&self) -> String {
        let hottest = self.heat.field().max_interior();
        format!(
            "{:.1} mol of product, hottest cell {:.1} K",
            self.mixture.total(2),
            hottest
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hover(world: [f64; 2]) -> Pointer {
        Pointer {
            world,
            velocity: [0.0, 0.0],
            pressed: false,
            held: false,
            released: false,
            secondary: false,
        }
    }

    fn press(world: [f64; 2]) -> Pointer {
        Pointer { pressed: true, held: true, ..hover(world) }
    }

    fn run(playground: &mut ReactionPlayground, seconds: f64) {
        let dt = playground.preferred_step();
        for _ in 0..(seconds / dt).round() as usize {
            playground.step(dt);
        }
    }

    /// Totals of A, B and C.
    fn totals(playground: &ReactionPlayground) -> [f64; 3] {
        [
            playground.mixture.total(0),
            playground.mixture.total(1),
            playground.mixture.total(2),
        ]
    }

    #[test]
    fn a_fresh_chamber_is_empty_and_at_room_temperature() {
        let playground = ReactionPlayground::new();
        assert_eq!(totals(&playground), [0.0, 0.0, 0.0]);
        assert!((playground.heat.field().max_interior() - AMBIENT).abs() < 1e-9);
    }

    /// A alone has nothing to react with. If this ever produces C the rate law is
    /// reading the wrong term.
    #[test]
    fn one_reactant_alone_makes_nothing() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        run(&mut playground, 4.0);

        let [a, b, c] = totals(&playground);
        assert!(a > 0.0, "A was painted");
        assert_eq!(b, 0.0);
        assert!(c < 1e-12, "made {c} of product out of one reactant");
        assert!(
            (playground.heat.field().max_interior() - 300.0).abs() < 1e-6,
            "and released no heat"
        );
    }

    #[test]
    fn two_reactants_in_the_same_place_react_and_heat() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        playground.pointer(press([0.5, 0.5]), 1);
        run(&mut playground, 4.0);

        let [_, _, c] = totals(&playground);
        assert!(c > 0.0, "no product");
        assert!(
            playground.heat.field().max_interior() > 300.5,
            "exothermic, so the chamber should be warmer than {:.2} K",
            playground.heat.field().max_interior()
        );
        assert!(playground.status().contains("product"));
    }

    /// `A + B -> C` with unit coefficients: every mole of A consumed is a mole of B
    /// consumed and a mole of C made. Diffusion moves material about but cannot create
    /// or destroy it, so this holds over the whole grid.
    #[test]
    fn the_stoichiometry_balances() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        playground.pointer(press([0.5, 0.5]), 1);
        let before = totals(&playground);
        run(&mut playground, 3.0);
        let after = totals(&playground);

        let (used_a, used_b, made_c) = (
            before[0] - after[0],
            before[1] - after[1],
            after[2] - before[2],
        );
        assert!(made_c > 1e-3, "nothing happened to measure");
        let scale = made_c.abs();
        assert!(
            (used_a - made_c).abs() / scale < 1e-6,
            "consumed {used_a} of A to make {made_c} of C"
        );
        assert!(
            (used_b - made_c).abs() / scale < 1e-6,
            "consumed {used_b} of B to make {made_c} of C"
        );
    }

    /// Arrhenius, which is the reason the temperature is a field the mixture reads
    /// rather than a number. A warmed patch must react faster than a cold one.
    #[test]
    fn a_warmer_mixture_reacts_faster() {
        fn product_after(warm: bool) -> f64 {
            let mut playground = ReactionPlayground::new();
            playground.pointer(press([0.5, 0.5]), 0);
            playground.pointer(press([0.5, 0.5]), 1);
            if warm {
                for _ in 0..20 {
                    playground.pointer(press([0.5, 0.5]), 2);
                }
            }
            run(&mut playground, 1.0);
            playground.mixture.total(2)
        }

        let (cold, warm) = (product_after(false), product_after(true));
        assert!(cold > 0.0, "the cold case should still react a little");
        assert!(
            warm > cold * 1.2,
            "warm made {warm:.4} and cold {cold:.4} — Arrhenius is not being felt"
        );
    }

    #[test]
    fn a_right_click_erases_what_was_painted() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        let painted = playground.mixture.total(0);
        assert!(painted > 0.0);

        playground.pointer(Pointer { secondary: true, ..hover([0.5, 0.5]) }, 0);
        assert!(
            playground.mixture.total(0) < painted * 0.01,
            "erasing left {} of {painted}",
            playground.mixture.total(0)
        );
    }

    /// A negative concentration is not a thing, and it would make the colour scale
    /// meaningless for everything else on screen.
    #[test]
    fn the_brush_never_drives_a_concentration_negative() {
        let mut playground = ReactionPlayground::new();
        for _ in 0..10 {
            playground.pointer(Pointer { secondary: true, ..hover([0.5, 0.5]) }, 0);
        }
        for index in 0..3 {
            let field = playground.mixture.concentration(index).expect("three species");
            assert!(field.min_interior() >= 0.0, "species {index} went negative");
        }
    }

    /// A slider that wiped the scene is a slider nobody dares touch.
    #[test]
    fn changing_the_rate_keeps_what_was_painted() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        let before = playground.mixture.total(0);

        playground.set_knob(1, 80.0);
        let after = playground.mixture.total(0);
        assert!((after - before).abs() / before < 1e-9, "{before} became {after}");

        playground.set_knob(2, 5.0);
        assert!((playground.mixture.total(0) - before).abs() / before < 1e-9);
    }

    #[test]
    fn reset_empties_the_chamber_and_it_still_works() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        playground.pointer(press([0.5, 0.5]), 1);
        run(&mut playground, 1.0);
        playground.reset();

        assert_eq!(totals(&playground), [0.0, 0.0, 0.0]);
        assert!((playground.heat.field().max_interior() - AMBIENT).abs() < 1e-9);

        playground.pointer(press([0.5, 0.5]), 0);
        playground.pointer(press([0.5, 0.5]), 1);
        run(&mut playground, 1.0);
        assert!(playground.mixture.total(2) > 0.0);
    }

    /// The temperature view swaps which field is drawn; it does not lay a second colour
    /// field over the first. Nothing observable about the simulation may change with it.
    #[test]
    fn the_temperature_view_changes_only_what_is_drawn() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        playground.pointer(press([0.5, 0.5]), 1);
        run(&mut playground, 2.0);

        let before = totals(&playground);
        let heat = playground.heat.field().max_interior();
        playground.set_toggle(0, true);
        assert!(playground.show_temperature);
        assert_eq!(totals(&playground), before);
        assert_eq!(playground.heat.field().max_interior(), heat);

        playground.set_toggle(0, false);
        assert!(!playground.show_temperature);
        assert_eq!(totals(&playground), before);
    }

    #[test]
    fn a_long_run_stays_finite() {
        let mut playground = ReactionPlayground::new();
        for step in 0..8 {
            let x = 0.2 + f64::from(step) * 0.08;
            playground.pointer(press([x, 0.5]), 0);
            playground.pointer(press([x, 0.5]), 1);
            playground.pointer(press([x, 0.5]), 2);
        }
        run(&mut playground, 20.0);

        for index in 0..3 {
            let field = playground.mixture.concentration(index).expect("three species");
            assert!(field.first_non_finite().is_none(), "species {index} went non-finite");
        }
        assert!(playground.heat.field().first_non_finite().is_none());
    }
}
