//! The shell: window, mode tabs, canvas, palette, and the diagnostics panel.
//!
//! Everything here is mode-agnostic. The shell knows how to run a clock, turn a cursor
//! position into a world position, draw a palette of tools and a rack of sliders, and
//! show what a mode publishes. It knows nothing about rigid bodies or reactions.
//!
//! # Why the panel is still here
//!
//! A sandbox is a toy, and it would have been easy to drop the diagnostics. But the
//! most interesting thing about a physics playground is the moment something behaves in
//! a way you did not expect, and that is exactly when "is momentum still conserved?" is
//! the question worth being able to answer. The panel is smaller and quieter than the
//! model viewer's, and it reports the same quantities by the same rules — including
//! saying nothing at all when a mode has no conserved quantity to claim.

use std::time::Instant;

use eframe::egui::{self, containers};
use lattice_ir::{ObservationKind, Observations};
use lattice_viewer::render::WorldView;
use lattice_viewer::{render, History, Mode, Palette, Status};

use crate::mode::{Playground, Pointer};

/// How many steps one frame may take, whatever the clock says.
///
/// A window that has been dragged, minimised, or blocked by a breakpoint comes back
/// with a large elapsed time, and a sandbox that faithfully simulated all of it would
/// freeze for a second and then show a scene that had already happened. Capping means
/// a stalled frame loses time rather than the interaction.
const MAX_STEPS_PER_FRAME: usize = 12;

/// The playground.
///
/// `Debug` is hand-written because a boxed trait object cannot derive it, and the
/// alternative — requiring every mode to be `Debug` — would put a bound on the trait
/// for the sake of a line of output nobody reads.
pub struct PlaygroundApp {
    modes: Vec<Box<dyn Playground>>,
    active: usize,
    tool: usize,
    playing: bool,
    /// Multiplier on real time, so the same scene can be watched slowly.
    speed: f64,
    mode_theme: Mode,
    history: History,
    observations: Observations,
    /// Wall-clock time of the last frame, for a real-time clock.
    last_frame: Instant,
    /// Simulated time since the mode was reset.
    elapsed: f64,
    /// Cursor position last frame, for the pointer velocity a throw needs.
    last_cursor: Option<[f64; 2]>,
    steps: u64,
    /// The active mode's disturbance count as of the last frame.
    disturbed: u64,
    show_panel: bool,
}

impl core::fmt::Debug for PlaygroundApp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlaygroundApp")
            .field("mode", &self.current().name())
            .field("playing", &self.playing)
            .field("steps", &self.steps)
            .finish()
    }
}

impl PlaygroundApp {
    /// A playground over the given modes.
    pub fn new(modes: Vec<Box<dyn Playground>>) -> PlaygroundApp {
        PlaygroundApp {
            modes,
            active: 0,
            tool: 0,
            playing: true,
            speed: 1.0,
            mode_theme: Mode::Dark,
            history: History::new(2048),
            observations: Observations::new(),
            last_frame: Instant::now(),
            elapsed: 0.0,
            last_cursor: None,
            steps: 0,
            disturbed: 0,
            show_panel: true,
        }
    }

    /// Open on a particular mode, before the window is shown.
    pub fn select(&mut self, index: usize) {
        self.active = index.min(self.modes.len().saturating_sub(1));
        self.tool = 0;
    }

    fn current(&self) -> &dyn Playground {
        self.modes[self.active].as_ref()
    }

    fn current_mut(&mut self) -> &mut dyn Playground {
        self.modes[self.active].as_mut()
    }

    /// Switch modes, starting the new one's history clean.
    ///
    /// Keeping the old series would put two modes' quantities on one plot, which is the
    /// dual-axis mistake wearing a different hat: the numbers would share a chart and
    /// have nothing to do with each other.
    fn activate(&mut self, index: usize) {
        if index == self.active {
            return;
        }
        self.active = index.min(self.modes.len() - 1);
        self.tool = 0;
        self.history.clear();
        self.elapsed = 0.0;
        self.steps = 0;
        self.disturbed = self.current().disturbances();
        self.record();
    }

    fn reset(&mut self) {
        self.current_mut().reset();
        self.history.clear();
        self.elapsed = 0.0;
        self.steps = 0;
        self.disturbed = self.current().disturbances();
        self.record();
    }

    /// Sample the current mode into the history.
    fn record(&mut self) {
        self.observations.clear();
        self.modes[self.active].observe(&mut self.observations);
        let time = self.elapsed;
        self.history.record(time, &self.observations);
    }

    /// Drop the history if the reader has reached into the scene since the last frame.
    ///
    /// Not a correction to the drift figure — a *refusal to compute one* across the
    /// disturbance. Painting heat in and then reporting the resulting jump as drift
    /// would be the panel's own version of the mistake it exists to catch.
    fn rebaseline_if_disturbed(&mut self) {
        let now = self.current().disturbances();
        if now != self.disturbed {
            self.disturbed = now;
            self.history.clear();
            self.record();
        }
    }

    /// Advance by however much real time has passed, bounded.
    fn advance(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let real = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;
        if !self.playing {
            return;
        }

        let dt = self.current().preferred_step();
        let wanted = (real * self.speed / dt).round() as usize;
        let steps = wanted.min(MAX_STEPS_PER_FRAME);
        for _ in 0..steps {
            self.current_mut().step(dt);
            self.elapsed += dt;
            self.steps += 1;
        }
        if steps > 0 {
            self.record();
        }
        // A sandbox is only alive while it is being repainted.
        ctx.request_repaint();
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        ui.horizontal(|ui| {
            // Modes first: the question "what kind of playground is this" comes before
            // anything you might do inside one.
            for index in 0..self.modes.len() {
                let name = self.modes[index].name();
                if ui.selectable_label(index == self.active, name).clicked() {
                    self.activate(index);
                }
            }

            ui.separator();
            if ui.button(if self.playing { "⏸ pause" } else { "▶ play" }).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("⏭ step").clicked() {
                let dt = self.current().preferred_step();
                self.playing = false;
                self.current_mut().step(dt);
                self.elapsed += dt;
                self.steps += 1;
                self.record();
            }
            if ui.button("↺ reset").clicked() {
                self.reset();
            }

            ui.separator();
            ui.colored_label(palette.text_muted, "speed");
            ui.add(
                egui::Slider::new(&mut self.speed, 0.05..=2.0)
                    .logarithmic(true)
                    .show_value(false),
            );
            ui.colored_label(palette.text_secondary, format!("{:.2}x", self.speed));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (label, next) = match self.mode_theme {
                    Mode::Dark => ("☀ light", Mode::Light),
                    Mode::Light => ("🌙 dark", Mode::Dark),
                };
                if ui.button(label).clicked() {
                    self.mode_theme = next;
                }
                ui.checkbox(&mut self.show_panel, "diagnostics");
            });
        });
    }

    /// The tool palette and the mode's sliders.
    fn controls(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        let tools = self.current().tools();
        if !tools.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for (index, tool) in tools.iter().enumerate() {
                    if ui.selectable_label(index == self.tool, &tool.name).clicked() {
                        self.tool = index;
                    }
                }
            });
            let hint = tools[self.tool.min(tools.len() - 1)].hint.clone();
            ui.colored_label(palette.text_muted, hint);
        }

        let toggles = self.current().toggles();
        if !toggles.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for (index, toggle) in toggles.iter().enumerate() {
                    let mut value = toggle.value;
                    if ui
                        .checkbox(&mut value, &toggle.name)
                        .on_hover_text(&toggle.hint)
                        .changed()
                    {
                        self.current_mut().set_toggle(index, value);
                    }
                }
            });
        }

        let knobs = self.current().knobs();
        for (index, knob) in knobs.iter().enumerate() {
            let mut value = knob.value;
            ui.horizontal(|ui| {
                ui.colored_label(palette.text_secondary, &knob.name);
                let slider = egui::Slider::new(&mut value, knob.min..=knob.max)
                    .logarithmic(knob.logarithmic)
                    .show_value(false);
                if ui.add(slider).changed() {
                    self.current_mut().set_knob(index, value);
                }
                let unit = if knob.unit.is_empty() {
                    String::new()
                } else {
                    format!(" {}", knob.unit)
                };
                ui.colored_label(
                    palette.text_secondary,
                    format!("{}{unit}", render::format_value(value)),
                );
            });
        }
    }

    /// How many leading dotted segments every published name has in common.
    ///
    /// One domain on screen means the prefix is noise on every row, and dropping it buys
    /// a narrower panel for nothing. Two domains — a mixture and the temperature field it
    /// reads — mean the prefix is the only thing saying which `min` is whose, and
    /// dropping it turns the table into a list of numbers with ambiguous labels.
    fn shared_prefix(&self) -> usize {
        let series = self.history.series();
        let Some(first) = series.first() else { return 0 };
        let Some((head, _)) = first.name.split_once('.') else { return 0 };
        let shared = series.iter().all(|other| {
            other.name.split_once('.').is_some_and(|(other_head, _)| other_head == head)
        });
        usize::from(shared)
    }

    /// How tall the control rack needs to be, in this style, for the busiest mode.
    ///
    /// Measured from egui's own spacing rather than written down as a pixel count, so it
    /// still fits if the font or the theme changes.
    fn controls_height(&self, ui: &egui::Ui) -> f32 {
        let row = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
        let rows = self
            .modes
            .iter()
            .map(|mode| {
                // The tool row, its hint, one row of checkboxes, and the sliders.
                let extras = usize::from(!mode.tools().is_empty()) * 2
                    + usize::from(!mode.toggles().is_empty());
                extras + mode.knobs().len()
            })
            .max()
            .unwrap_or(0);
        16.0 + row * rows as f32
    }

    /// The canvas, and the pointer handling that makes it a playground.
    fn canvas(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        let (origin, extent) = self.current().bounds();
        let aspect = (extent[0] / extent[1]) as f32;
        // Reserve the status line's row first. Without this the canvas takes the whole
        // panel and the line under it is clipped by the controls — and the status line is
        // where a square arena says how many bodies are in it.
        let status_height = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
        let available = ui.available_size() - egui::Vec2::new(0.0, status_height);
        // Fit while preserving the aspect ratio: a square arena drawn as a rectangle
        // makes every length read off the screen wrong.
        let width = available.x.min(available.y * aspect).max(120.0);
        let size = egui::Vec2::new(width, width / aspect);

        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        ui.painter().rect_filled(rect, 2.0, palette.plane);
        let view = WorldView::new(rect, origin, extent);

        self.handle_pointer(&response, view, ui.input(|i| i.stable_dt));

        self.current().draw(ui.painter(), view, palette);
        ui.painter().rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, palette.border),
            egui::StrokeKind::Inside,
        );
        ui.colored_label(palette.text_muted, self.current().status());
    }

    fn handle_pointer(&mut self, response: &egui::Response, view: WorldView, frame_time: f32) {
        let Some(screen) = response.hover_pos().or_else(|| response.interact_pointer_pos()) else {
            // The cursor left, or the window lost focus. A mode holding something has to
            // let go, or an object stays stuck to a cursor that is no longer there.
            self.current_mut().pointer_left();
            self.last_cursor = None;
            return;
        };
        let world = view.unproject(screen);

        // Velocity from the *displacement over the frame*, not from egui's pointer
        // delta, because the two disagree whenever a frame is long — and a throw whose
        // strength depends on the frame rate is the worst bug a sandbox can have.
        let dt = f64::from(frame_time).max(1.0 / 240.0);
        let velocity = match self.last_cursor {
            Some(previous) => [(world[0] - previous[0]) / dt, (world[1] - previous[1]) / dt],
            None => [0.0, 0.0],
        };
        self.last_cursor = Some(world);

        let pointer = Pointer {
            world,
            velocity,
            pressed: response.drag_started() || response.clicked(),
            held: response.dragged(),
            released: response.drag_stopped(),
            secondary: response.secondary_clicked(),
        };
        let tool = self.tool;
        self.current_mut().pointer(pointer, tool);
        self.rebaseline_if_disturbed();
    }

    /// The diagnostics panel: what this mode claims and what it measures.
    fn panel(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        ui.heading("diagnostics");
        ui.colored_label(palette.text_muted, self.current().description());

        ui.add_space(8.0);
        ui.colored_label(palette.text_secondary, "clock");
        ui.colored_label(
            palette.text_primary,
            format!("t {:.3} s   step {}", self.elapsed, self.steps),
        );

        // Conservation, by the same rule the model viewer uses: report what the domain
        // *declared* as an invariant, and say plainly when it declared none.
        ui.add_space(8.0);
        ui.colored_label(palette.text_secondary, "conservation");
        let mut any = false;
        for series in self.history.series() {
            if !series.is_invariant() || series.is_constant() {
                continue;
            }
            let Some(drift) = self.history.drift_of(series) else { continue };
            any = true;
            ui.colored_label(palette.text_primary, trim(&series.name, self.shared_prefix()));
            let mut verdict = render::drift_verdict(drift.value);
            verdict.detail = format!("{:+.2e} {}", drift.value, drift.basis.phrase());
            render::status_line(ui, palette, &verdict);
        }
        if !any {
            ui.colored_label(
                palette.text_muted,
                if self.history.sample_count() < 2 {
                    "nothing has moved yet"
                } else {
                    "nothing here is conserved — a sandbox with gravity and walls has an \
                     external force and an infinite sink, so this mode claims none"
                },
            );
        }

        ui.add_space(8.0);
        ui.colored_label(palette.text_secondary, "current values");
        let strip = self.shared_prefix();
        egui::Grid::new("values").num_columns(2).striped(true).show(ui, |ui| {
            for series in self.history.series() {
                let Some(latest) = series.latest() else { continue };
                if matches!(series.kind, ObservationKind::Count) && latest == 0.0 {
                    continue;
                }
                ui.colored_label(palette.text_secondary, trim(&series.name, strip));
                let colour = if latest.is_finite() {
                    palette.text_primary
                } else {
                    palette.status(Status::Critical)
                };
                let unit = if series.unit == "1" { "" } else { &series.unit };
                let scale = self.history.display_scale_for(series);
                ui.colored_label(
                    colour,
                    format!("{} {unit}", render::format_value_against(latest, scale)),
                );
                ui.end_row();
            }
        });
    }
}

impl eframe::App for PlaygroundApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let palette = Palette::for_mode(self.mode_theme);
        render::apply_theme(ui.ctx(), &palette);
        self.advance(ui.ctx());

        containers::Panel::top("bar").show(ui, |ui| {
            ui.add_space(2.0);
            self.top_bar(ui, &palette);
            ui.add_space(2.0);
        });

        if self.show_panel {
            containers::Panel::right("diagnostics")
                .resizable(true)
                .default_size(320.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| self.panel(ui, &palette));
                });
        }

        // Tall enough for the rack of whichever mode has the most controls, not just the
        // one on screen. Two reasons: a shrink-wrapped panel clips its last slider when
        // the window is short, and a slider you cannot see is a control that does not
        // exist; and a panel that resized per mode would move the canvas under the
        // cursor every time a tab was clicked.
        containers::Panel::bottom("controls")
            .exact_size(self.controls_height(ui))
            .show(ui, |ui| {
                ui.add_space(4.0);
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, &palette));
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.canvas(ui, &palette);
        });
    }
}

/// Drop `count` leading dotted segments from an observation name.
fn trim(name: &str, count: usize) -> &str {
    let mut rest = name;
    for _ in 0..count {
        match rest.split_once('.') {
            Some((_, tail)) if !tail.is_empty() => rest = tail,
            _ => return rest,
        }
    }
    rest
}
