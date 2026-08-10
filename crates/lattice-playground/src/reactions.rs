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
//! factor is an areal heat capacity, and it is the fixed `HEAT_CAPACITY` here rather than
//! anything the sandbox derives — a slider would be fun, but a playground where the
//! heat capacity is adjustable is a playground where "why did that not get hot" has two
//! answers instead of one.

use eframe::egui;
use lattice_domain_chemistry::{presets, ReactingMixture, Recipe};
use lattice_domain_grid2d::{Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, Domain, Grid2d, Observations, ScalarField, StepContext};
use lattice_viewer::render::WorldView;
use lattice_viewer::{render, Colormap, Palette};

use crate::mode::{Choice, Knob, Playground, Pointer, Toggle, Tool};

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
/// What the brush paints.
///
/// One variant per *reactant* of whichever recipe is loaded, plus heat. Indexed rather
/// than named, because the species are now the recipe's and a fixed `A`/`B` pair could
/// only ever describe one reaction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Brush {
    /// Lay down the species at this index into the recipe's ingredients.
    Paint(usize),
    /// Add heat.
    Warmth,
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
    /// The loaded reaction, and its index into [`presets`].
    recipe: Recipe,
    recipe_index: usize,
    /// Which species the canvas draws when it is not drawing temperature.
    ///
    /// Defaults to the recipe's first product, which is what someone who just picked a
    /// reaction wants to watch appear.
    display: usize,
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
        let recipe = presets().swap_remove(0);
        let display = recipe.products().first().copied().unwrap_or(0);
        let (rate, activation) = (recipe.rate, recipe.activation_energy);
        let mut playground = ReactionPlayground {
            mixture: ReactingMixture::new("chamber", grid, recipe.network(rate, activation)),
            heat: room_temperature(grid),
            arena: Arena::with_capacity(1 << 17),
            grid,
            heat_buffer: ScalarField::new(&grid, 1),
            temperature_buffer: ScalarField::new(&grid, 1),
            recipe,
            recipe_index: 0,
            display,
            brush: Brush::Paint(0),
            brush_radius: 0.08,
            rate,
            activation,
            hover: None,
            show_temperature: false,
            disturbances: 0,
            painted: 0.0,
        };
        playground.rebuild();
        playground
    }

    /// Load preset `index`, emptying the chamber.
    ///
    /// The concentrations are *not* carried across, unlike a rate change. Species mean
    /// different things between recipes — index 1 is oxygen in one and iron(III) oxide in
    /// the next — so copying them over would silently relabel whatever was on screen as
    /// something it is not. Changing the reaction is changing the experiment.
    fn load(&mut self, index: usize) {
        let mut recipes = presets();
        if index >= recipes.len() {
            return;
        }
        let recipe = recipes.swap_remove(index);
        self.display = recipe.products().first().copied().unwrap_or(0);
        self.rate = recipe.rate;
        self.activation = recipe.activation_energy;
        self.recipe = recipe;
        self.recipe_index = index;
        self.brush = Brush::Paint(self.recipe.reactants().first().copied().unwrap_or(0));

        self.mixture = ReactingMixture::new(
            "chamber",
            self.grid,
            self.recipe.network(self.rate, self.activation),
        );
        self.heat = room_temperature(self.grid);
        self.painted = 0.0;
        self.disturbances += 1;
    }

    /// Rebuild the mixture after a rate change.
    ///
    /// A rate constant is baked into the network at construction, so changing one means
    /// a new network. The concentrations are carried across, because a slider that
    /// wiped the scene would be a slider nobody dares touch.
    fn rebuild(&mut self) {
        let mut fresh = ReactingMixture::new(
            "chamber",
            self.grid,
            self.recipe.network(self.rate, self.activation),
        );
        for index in 0..self.recipe.ingredients.len() {
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
            Brush::Paint(index) => self.mixture.concentration_mut(index),
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
        let mut tools: Vec<Tool> = self
            .recipe
            .reactants()
            .into_iter()
            .map(|index| {
                let ingredient = &self.recipe.ingredients[index];
                Tool::new(
                    format!("paint {}", ingredient.formula),
                    format!(
                        "drag to lay down {} ({}); right-click erases",
                        ingredient.name, ingredient.formula
                    ),
                )
            })
            .collect();
        tools.push(Tool::new(
            "warm it up",
            "drag to add heat — a hotter mixture reacts faster (Arrhenius)",
        ));
        tools
    }

    fn choices(&self) -> Vec<Choice> {
        let recipes = presets();
        let mut out = vec![
            Choice::new(
                "reaction",
                recipes.iter().map(|recipe| recipe.name.clone()).collect(),
                self.recipe_index,
            )
            .with_detail(format!(
                "{}
{}",
                self.recipe.description,
                self.recipe.fidelity.caveat()
            )),
        ];
        out.push(
            Choice::new(
                "draw",
                self.recipe.ingredients.iter().map(|i| i.formula.clone()).collect(),
                self.display,
            )
            .with_detail(format!(
                "{} — the canvas shows this species unless 'show temperature' is on",
                self.recipe.ingredients[self.display].name
            )),
        );
        out
    }

    fn set_choice(&mut self, index: usize, option: usize) {
        match index {
            0 => self.load(option),
            1 if option < self.recipe.ingredients.len() => self.display = option,
            _ => {}
        }
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
        // Scoped, not dropped: the block is what ends the borrow of `self.arena` so
        // `exchange` can take `&mut self` below.
        {
            let mut ctx = StepContext::new(&mut self.arena);
            self.mixture.prepare(&mut ctx);
            self.mixture.advance(dt, &mut ctx);
            self.heat.prepare(&mut ctx);
            self.heat.advance(dt, &mut ctx);
        }
        self.exchange(dt);
    }

    fn pointer(&mut self, pointer: Pointer, tool: usize) {
        self.hover = Some(pointer.world);
        let reactants = self.recipe.reactants();
        // The last tool is always warmth; everything before it is a reactant.
        self.brush = match reactants.get(tool) {
            Some(&species) => Brush::Paint(species),
            None => Brush::Warmth,
        };

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
                self.mixture.concentration(self.display).expect("a loaded species"),
                Colormap::Sequential,
                palette.mode,
                palette,
                0.0,
            )
        };
        let rect = view.rect();
        let handle = painter.ctx().load_texture(
            "reaction",
            rendered.image,
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
        self.mixture = ReactingMixture::new(
            "chamber",
            self.grid,
            self.recipe.network(self.rate, self.activation),
        );
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

    /// Totals of every species in the loaded recipe, in ingredient order.
    fn totals(playground: &ReactionPlayground) -> Vec<f64> {
        (0..playground.recipe.ingredients.len())
            .map(|index| playground.mixture.total(index))
            .collect()
    }

    /// Paint every reactant of the loaded recipe into the same spot.
    fn paint_all_reactants(playground: &mut ReactionPlayground, at: [f64; 2]) {
        for tool in 0..playground.recipe.reactants().len() {
            playground.pointer(press(at), tool);
        }
    }

    #[test]
    fn a_fresh_chamber_is_empty_and_at_room_temperature() {
        let playground = ReactionPlayground::new();
        assert!(totals(&playground).iter().all(|total| *total == 0.0));
        assert!((playground.heat.field().max_interior() - AMBIENT).abs() < 1e-9);
    }

    /// A alone has nothing to react with. If this ever produces C the rate law is
    /// reading the wrong term.
    #[test]
    fn one_reactant_alone_makes_nothing() {
        let mut playground = ReactionPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        run(&mut playground, 4.0);

        let totals = totals(&playground);
        let reactants = playground.recipe.reactants();
        assert!(totals[reactants[0]] > 0.0, "the first reactant was painted");
        for &other in &reactants[1..] {
            assert_eq!(totals[other], 0.0, "nothing else was painted");
        }
        for product in playground.recipe.products() {
            assert!(totals[product] < 1e-12, "made {} of product", totals[product]);
        }
        assert!(
            (playground.heat.field().max_interior() - 300.0).abs() < 1e-6,
            "and released no heat"
        );
    }

    #[test]
    fn two_reactants_in_the_same_place_react_and_heat() {
        let mut playground = ReactionPlayground::new();
        paint_all_reactants(&mut playground, [0.5, 0.5]);
        run(&mut playground, 4.0);

        let totals = totals(&playground);
        let product = playground.recipe.products()[0];
        assert!(totals[product] > 0.0, "no product");
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
        paint_all_reactants(&mut playground, [0.5, 0.5]);
        let before = totals(&playground);
        run(&mut playground, 3.0);
        let after = totals(&playground);

        // Every species change divided by its stoichiometric coefficient must be the
        // same number: that is what a balanced equation *means*. The old form compared
        // raw totals, which only worked because every coefficient happened to be one.
        let recipe = &playground.recipe;
        let extents: Vec<f64> = recipe
            .ingredients
            .iter()
            .enumerate()
            .map(|(index, ingredient)| (after[index] - before[index]) / ingredient.coefficient)
            .collect();

        let extent = extents[0];
        assert!(extent > 1e-3, "nothing happened to measure");
        for (index, measured) in extents.iter().enumerate() {
            assert!(
                (measured - extent).abs() / extent < 1e-6,
                "{} moved by {measured} of reaction extent, not {extent}",
                recipe.ingredients[index].formula
            );
        }
    }

    /// Every preset must actually do something when its reactants meet, at its own
    /// default rate. A recipe whose rate constant is off by three decades is a menu entry
    /// that does nothing when picked, and nothing else in the suite would notice.
    #[test]
    fn every_preset_reacts_visibly_at_its_own_default_rate() {
        for (index, recipe) in presets().into_iter().enumerate() {
            let mut playground = ReactionPlayground::new();
            playground.set_choice(0, index);
            paint_all_reactants(&mut playground, [0.5, 0.5]);

            let before = totals(&playground);
            run(&mut playground, 4.0);
            let after = totals(&playground);

            let product = recipe.products()[0];
            let made = after[product] - before[product];
            assert!(
                made > 1e-3,
                "{} made only {made:.3e} of {} in four seconds",
                recipe.name,
                recipe.ingredients[product].formula
            );
            assert!(
                after.iter().all(|total| total.is_finite()),
                "{} went non-finite",
                recipe.name
            );
        }
    }

    /// The sandbox sliders have to reach every preset own numbers, or picking one leaves
    /// a control showing a value it cannot represent.
    #[test]
    fn every_preset_sits_inside_the_sandbox_sliders() {
        let playground = ReactionPlayground::new();
        let knobs = playground.knobs();
        let rate = knobs.iter().find(|k| k.name == "rate").expect("a rate knob");
        let activation = knobs.iter().find(|k| k.name == "activation").expect("an Ea knob");

        for recipe in presets() {
            assert!(
                (rate.min..=rate.max).contains(&recipe.rate),
                "rate {} of {} is outside the slider {}..{}",
                recipe.rate,
                recipe.name,
                rate.min,
                rate.max
            );
            let kilojoules = recipe.activation_energy / 1000.0;
            assert!(
                (activation.min..=activation.max).contains(&kilojoules),
                "activation {kilojoules} kJ/mol of {} is outside {}..{}",
                recipe.name,
                activation.min,
                activation.max
            );
        }
    }

    /// Arrhenius, which is the reason the temperature is a field the mixture reads
    /// rather than a number. A warmed patch must react faster than a cold one.
    ///
    /// Measured on a *slow, high-activation-energy* recipe rather than whatever happens to
    /// be loaded first. Both halves of that matter. A high activation energy is where
    /// Arrhenius has something to say — that is what the exponent multiplies — and a slow
    /// reaction is where the answer is not pinned at "all of it reacted either way". On the
    /// default recipe this same test measures an 18% gain, not because the physics is
    /// weaker but because the cold case has already consumed most of its reactants.
    #[test]
    fn a_warmer_mixture_reacts_faster() {
        let steep = presets()
            .iter()
            .position(|recipe| recipe.name == "rusting iron")
            .expect("a slow, high-activation-energy preset");

        fn product_after(preset: usize, warm: bool) -> f64 {
            let mut playground = ReactionPlayground::new();
            playground.set_choice(0, preset);
            paint_all_reactants(&mut playground, [0.5, 0.5]);
            if warm {
                // The warmth brush is always the tool after the last reactant.
                let warmth = playground.recipe.reactants().len();
                for _ in 0..20 {
                    playground.pointer(press([0.5, 0.5]), warmth);
                }
            }
            run(&mut playground, 1.0);
            let product = playground.recipe.products()[0];
            playground.mixture.total(product)
        }

        let (cold, warm) = (product_after(steep, false), product_after(steep, true));
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

        assert!(totals(&playground).iter().all(|total| *total == 0.0));
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
