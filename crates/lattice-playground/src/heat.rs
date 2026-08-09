//! The diffusion sandbox: paint hot and cold, watch it even out.
//!
//! The simplest of the three modes, and the one where the *diagnostics* are the point.
//! A closed box conserves its heat integral exactly — the panel shows it holding to
//! round-off — and opening one edge makes it a sink, which the panel then reports by
//! declining to call it conserved at all. Being able to flip that with one control is
//! the clearest demonstration in the whole engine of what "conserved" is actually a
//! claim about.

use eframe::egui;
use lattice_domain_grid2d::{heated_edge, Diffusivity, HeatDomain, TimeScheme};
use lattice_ir::{Arena, BoundarySet, Domain, Grid2d, Observations, Side, StepContext};
use lattice_viewer::render::WorldView;
use lattice_viewer::{render, Colormap, Palette};

use crate::mode::{Knob, Playground, Pointer, Toggle, Tool};

/// Grid resolution.
const CELLS: usize = 96;

/// The plate, m.
const EXTENT: [f64; 2] = [1.0, 1.0];

/// Ambient temperature, K.
const AMBIENT: f64 = 300.0;

/// What the brush does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Brush {
    Hot,
    Cold,
}

impl Brush {
    const ALL: [Brush; 2] = [Brush::Hot, Brush::Cold];

    fn label(self) -> &'static str {
        match self {
            Brush::Hot => "heat",
            Brush::Cold => "chill",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Brush::Hot => "drag to warm the plate; right-click cools instead",
            Brush::Cold => "drag to cool the plate; right-click warms instead",
        }
    }

    fn amount(self) -> f64 {
        match self {
            Brush::Hot => 40.0,
            Brush::Cold => -40.0,
        }
    }
}

/// The diffusion sandbox.
#[derive(Debug)]
pub struct HeatPlayground {
    field: HeatDomain,
    arena: Arena,
    grid: Grid2d,
    brush: Brush,
    brush_radius: f64,
    diffusivity: f64,
    /// True when the left edge is held at ambient — which makes the plate an open
    /// system and the heat integral no longer a conserved quantity.
    open_edge: bool,
    hover: Option<[f64; 2]>,
    /// Bumped every time the brush puts energy into the plate; see
    /// [`Playground::disturbances`].
    disturbances: u64,
}

impl Default for HeatPlayground {
    fn default() -> Self {
        HeatPlayground::new()
    }
}

impl HeatPlayground {
    /// A plate at ambient temperature, insulated all round.
    pub fn new() -> HeatPlayground {
        let grid = Grid2d::new(CELLS, CELLS, EXTENT);
        HeatPlayground {
            field: plate(grid, 2e-4, false),
            arena: Arena::with_capacity(1 << 17),
            grid,
            brush: Brush::Hot,
            brush_radius: 0.08,
            diffusivity: 2e-4,
            open_edge: false,
            hover: None,
            disturbances: 0,
        }
    }

    /// Rebuild the solver after a boundary or diffusivity change, keeping the field.
    fn rebuild(&mut self) {
        let mut fresh = plate(self.grid, self.diffusivity, self.open_edge);
        fresh.field_mut().copy_interior_from(self.field.field());
        self.field = fresh;
    }

    /// Add `amount` kelvin in a soft disc around `at`.
    fn paint(&mut self, at: [f64; 2], amount: f64) {
        self.disturbances += 1;
        let (radius, grid) = (self.brush_radius, self.grid);
        let field = self.field.field_mut();
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let centre = grid.cell_center(i, j);
                let distance = (centre[0] - at[0]).hypot(centre[1] - at[1]);
                if distance > radius {
                    continue;
                }
                let falloff = 0.5 * (1.0 + (core::f64::consts::PI * distance / radius).cos());
                // Floored well above absolute zero: a plate at 0 K is not something
                // anyone meant to draw, and a negative temperature would make the
                // colour scale meaningless for everything else on screen.
                let value = (field.get(i, j) + amount * falloff).max(1.0);
                field.set(i, j, value);
            }
        }
    }
}

fn plate(grid: Grid2d, diffusivity: f64, open_edge: bool) -> HeatDomain {
    let boundaries = if open_edge {
        heated_edge(Side::Left, AMBIENT)
    } else {
        BoundarySet::INSULATED
    };
    HeatDomain::new("plate", grid, Diffusivity::Uniform(diffusivity))
        .with_scheme(TimeScheme::CrankNicolson)
        .with_display_unit("K")
        .with_boundaries(boundaries)
        .with_uniform_initial(AMBIENT)
}

impl Playground for HeatPlayground {
    fn name(&self) -> &'static str {
        "heat"
    }

    fn description(&self) -> &'static str {
        "paint hot and cold spots and watch them even out"
    }

    fn tools(&self) -> Vec<Tool> {
        Brush::ALL.iter().map(|brush| Tool::new(brush.label(), brush.hint())).collect()
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![
            Knob::new("brush", self.brush_radius, 0.02, 0.3, "m"),
            Knob::logarithmic("diffusivity", self.diffusivity, 1e-5, 5e-3, "m^2/s"),
        ]
    }

    fn toggles(&self) -> Vec<Toggle> {
        vec![Toggle::new(
            "open the left edge",
            self.open_edge,
            "hold the left edge at ambient, which makes the plate an open system — \
             watch the panel stop calling the heat integral conserved",
        )]
    }

    fn set_toggle(&mut self, index: usize, value: bool) {
        if index == 0 && value != self.open_edge {
            self.open_edge = value;
            self.rebuild();
        }
    }

    fn set_knob(&mut self, index: usize, value: f64) {
        match index {
            0 => self.brush_radius = value,
            1 => {
                self.diffusivity = value;
                self.rebuild();
            }
            _ => {}
        }
    }

    fn preferred_step(&self) -> f64 {
        0.02
    }

    fn step(&mut self, dt: f64) {
        let mut ctx = StepContext::new(&mut self.arena);
        self.field.prepare(&mut ctx);
        self.field.advance(dt, &mut ctx);
    }

    fn pointer(&mut self, pointer: Pointer, tool: usize) {
        self.hover = Some(pointer.world);
        self.brush = Brush::ALL[tool.min(Brush::ALL.len() - 1)];
        if pointer.pressed || pointer.held {
            self.paint(pointer.world, self.brush.amount());
        }
        if pointer.secondary {
            self.paint(pointer.world, -self.brush.amount());
        }
    }

    fn pointer_left(&mut self) {
        self.hover = None;
    }

    fn draw(&self, painter: &egui::Painter, view: WorldView, palette: &Palette) {
        // Diverging, anchored on ambient: the interesting question here is which way a
        // spot is *off* room temperature, and a one-hue ramp cannot answer it. This is
        // the case a diverging map exists for.
        let rendered = render::field_to_image_about(
            self.field.field(),
            Colormap::Diverging,
            palette.mode,
            palette,
            AMBIENT,
        );
        let rect = view.rect();
        let handle = painter.ctx().load_texture(
            "plate",
            rendered.image,
            egui::TextureOptions::NEAREST,
        );
        painter.image(
            handle.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        if let Some(at) = self.hover
            && let Some(centre) = view.project(at)
        {
            painter.circle_stroke(
                centre,
                self.brush_radius as f32 * view.scale(),
                egui::Stroke::new(1.0, palette.text_muted),
            );
        }
    }

    fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        (self.grid.origin(), EXTENT)
    }

    fn observe(&self, out: &mut Observations) {
        self.field.observe(out);
    }

    fn disturbances(&self) -> u64 {
        self.disturbances
    }

    fn reset(&mut self) {
        self.field = plate(self.grid, self.diffusivity, self.open_edge);
    }

    fn status(&self) -> String {
        let field = self.field.field();
        let edge = if self.open_edge {
            "left edge held at ambient, so heat leaves"
        } else {
            "insulated, so the heat integral is conserved"
        };
        format!(
            "{:.1} K to {:.1} K — {edge}",
            field.min_interior(),
            field.max_interior()
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

    fn run(playground: &mut HeatPlayground, seconds: f64) {
        let dt = playground.preferred_step();
        for _ in 0..(seconds / dt).round() as usize {
            playground.step(dt);
        }
    }

    /// Kelvin times square metres — the quantity an insulated plate conserves.
    fn heat_integral(playground: &HeatPlayground) -> f64 {
        playground.field.field().integrate(&playground.grid)
    }

    #[test]
    fn a_fresh_plate_is_uniformly_ambient() {
        let playground = HeatPlayground::new();
        assert!((playground.field.field().min_interior() - AMBIENT).abs() < 1e-9);
        assert!((playground.field.field().max_interior() - AMBIENT).abs() < 1e-9);
    }

    #[test]
    fn painting_makes_a_hot_spot_and_diffusion_spreads_it() {
        let mut playground = HeatPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        let peak = playground.field.field().max_interior();
        assert!(peak > AMBIENT + 30.0, "the brush deposited {peak:.2} K");

        run(&mut playground, 30.0);
        let spread = playground.field.field().max_interior();
        assert!(spread < peak, "the peak should fall as it spreads: {peak} -> {spread}");
        assert!(
            playground.field.field().min_interior() > AMBIENT + 1e-9,
            "and the cold corners should have come up"
        );
    }

    #[test]
    fn the_chill_brush_cools() {
        let mut playground = HeatPlayground::new();
        playground.pointer(press([0.5, 0.5]), 1);
        assert!(playground.field.field().min_interior() < AMBIENT - 30.0);
        assert_eq!(playground.brush, Brush::Cold);
    }

    /// The clearest statement in the engine of what "conserved" is a claim about: the
    /// same solver, the same field, one boundary changed, and the answer flips.
    #[test]
    fn an_insulated_plate_holds_its_heat_integral() {
        let mut playground = HeatPlayground::new();
        playground.pointer(press([0.3, 0.7]), 0);
        playground.pointer(press([0.7, 0.3]), 1);
        let before = heat_integral(&playground);
        run(&mut playground, 40.0);
        let after = heat_integral(&playground);

        assert!(
            (after - before).abs() / before < 1e-10,
            "{before:.9} became {after:.9}"
        );
        assert!(playground.status().contains("conserved"));
    }

    #[test]
    fn an_open_edge_is_a_sink_and_the_status_says_so() {
        let mut playground = HeatPlayground::new();
        playground.set_toggle(0, true);
        assert!(playground.open_edge);
        playground.pointer(press([0.1, 0.5]), 0);
        let before = heat_integral(&playground);
        run(&mut playground, 40.0);
        let after = heat_integral(&playground);

        assert!(after < before, "heat should leave: {before:.6} -> {after:.6}");
        assert!(!playground.status().contains("conserved"));
        assert!(playground.status().contains("heat leaves"));
    }

    /// Closing the edge again has to restore the claim, or the control is a one-way door.
    #[test]
    fn closing_the_edge_makes_it_conserved_again() {
        let mut playground = HeatPlayground::new();
        playground.set_toggle(0, true);
        playground.pointer(press([0.5, 0.5]), 0);
        run(&mut playground, 5.0);

        playground.set_toggle(0, false);
        assert!(!playground.open_edge);
        let before = heat_integral(&playground);
        run(&mut playground, 20.0);
        assert!((heat_integral(&playground) - before).abs() / before < 1e-10);
        assert!(playground.status().contains("conserved"));
    }

    #[test]
    fn changing_the_diffusivity_keeps_the_picture() {
        let mut playground = HeatPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        let before = heat_integral(&playground);

        playground.set_knob(1, 1e-3);
        let after = heat_integral(&playground);
        assert!((after - before).abs() / before < 1e-12, "{before} became {after}");
        assert!((playground.diffusivity - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn a_faster_plate_evens_out_sooner() {
        fn range_after(diffusivity: f64) -> f64 {
            let mut playground = HeatPlayground::new();
            playground.set_knob(1, diffusivity);
            playground.pointer(press([0.5, 0.5]), 0);
            run(&mut playground, 20.0);
            let field = playground.field.field();
            field.max_interior() - field.min_interior()
        }

        let (slow, fast) = (range_after(5e-5), range_after(3e-3));
        assert!(fast < slow, "fast plate spread {fast:.4} K, slow one {slow:.4} K");
    }

    /// A plate at 0 K is not something anyone meant to draw, and a negative temperature
    /// would make the diverging colour scale meaningless for everything else on screen.
    #[test]
    fn the_chill_brush_cannot_reach_absolute_zero() {
        let mut playground = HeatPlayground::new();
        for _ in 0..200 {
            playground.pointer(press([0.5, 0.5]), 1);
        }
        assert!(playground.field.field().min_interior() > 0.0);
    }

    #[test]
    fn reset_returns_the_plate_to_ambient() {
        let mut playground = HeatPlayground::new();
        playground.pointer(press([0.5, 0.5]), 0);
        run(&mut playground, 2.0);
        playground.reset();
        assert!((playground.field.field().max_interior() - AMBIENT).abs() < 1e-9);
        assert!((playground.field.field().min_interior() - AMBIENT).abs() < 1e-9);
    }
}
