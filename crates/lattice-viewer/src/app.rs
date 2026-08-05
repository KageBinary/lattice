//! The viewer application.
//!
//! Spec P7 — *visualization is instrumentation*:
//!
//! > The renderer is not decoration. It must display fields, fluxes, constraints,
//! > forces, residuals, conservation drift, timestep decisions, and uncertainty.
//!
//! So the window is laid out around that claim. The picture is the largest thing on
//! screen, but it never appears alone: beside it sit the exact numbers behind it, how
//! far each conserved quantity has drifted, how close the timestep is to its stability
//! limit, and what the solver admits it does not conserve.
//!
//! A viewer that showed only the pretty field would be the thing the spec warns
//! against — *"a visually convincing result is not sufficient"* (P1).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use eframe::egui::{self, containers, TextureHandle, TextureOptions};
use egui_plot::{Line, Plot, PlotPoints};
use lattice_compiler::compile_source;
use lattice_ir::{Observations, RenderChannel};
use lattice_runtime::Simulation;
use lattice_syntax::SourceFile;

use crate::history::{DriftBasis, History, Series};
use crate::palette::{Colormap, Mode, Palette, Status};
use crate::render;

/// Nearest-neighbour sampling, deliberately.
///
/// Smoothing a field makes a coarse grid look like a fine one. Blocky cells are the
/// honest picture: the reader can see the resolution they are actually getting.
const TEXTURE: TextureOptions = TextureOptions::NEAREST;

/// A loaded, runnable model.
struct Loaded {
    simulation: Simulation,
    report: String,
    /// Compiler warnings, if any survived a successful compile.
    warnings: String,
    timestep: f64,
    stability_limit: f64,
    limiting_domain: Option<String>,
}

/// Whether the file compiled.
enum State {
    Ready(Box<Loaded>),
    Failed(String),
}

/// The viewer.
pub struct ViewerApp {
    path: PathBuf,
    state: State,
    playing: bool,
    steps_per_frame: usize,
    colormap: Colormap,
    mode: Mode,
    history: History,
    textures: BTreeMap<String, TextureHandle>,
    show_report: bool,
    show_contracts: bool,
}

impl std::fmt::Debug for ViewerApp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewerApp").field("path", &self.path).field("playing", &self.playing).finish()
    }
}

impl ViewerApp {
    /// Load a model file into a viewer.
    pub fn new(path: impl Into<PathBuf>) -> ViewerApp {
        ViewerApp::with_options(path, false)
    }

    /// Load a model, optionally starting it immediately.
    ///
    /// Running on open is off by default — a viewer that starts moving before you have
    /// read the initial condition is hard to reason about — but it is what makes an
    /// unattended screenshot or a demo useful.
    pub fn with_options(path: impl Into<PathBuf>, play: bool) -> ViewerApp {
        let path = path.into();
        let state = load(&path);
        let mut app = ViewerApp {
            path,
            state,
            playing: play,
            steps_per_frame: 4,
            colormap: Colormap::Sequential,
            mode: Mode::Dark,
            history: History::default(),
            textures: BTreeMap::new(),
            show_report: false,
            show_contracts: false,
        };
        app.record_initial_state();
        app
    }

    /// Recompile from disk and start over.
    fn reset(&mut self) {
        self.state = load(&self.path);
        self.history.clear();
        self.textures.clear();
        self.playing = false;
        self.record_initial_state();
    }

    /// Sample the model before it has taken a step, so the values table and the plots
    /// have their `t = 0` point — which is also the baseline every drift is measured
    /// against.
    fn record_initial_state(&mut self) {
        if let State::Ready(loaded) = &mut self.state {
            let mut observations = Observations::new();
            for domain in loaded.simulation.domains() {
                domain.observe(&mut observations);
            }
            self.history.record(0.0, &observations);
        }
    }
}

/// Compile a file into a runnable simulation, or a rendered error.
fn load(path: &Path) -> State {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => return State::Failed(format!("cannot read {}: {error}", path.display())),
    };
    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    let file = SourceFile::new(name, text);

    let (compiled, diagnostics) = compile_source(&file);
    if diagnostics.has_errors() {
        return State::Failed(diagnostics.render(&file));
    }
    let Some(compiled) = compiled else {
        return State::Failed("the file produced no model".to_string());
    };
    if compiled.domains.is_empty() {
        return State::Failed(
            "this model compiles but declares no solvers, so there is nothing to run.\n\
             Add a `solve …` statement."
                .to_string(),
        );
    }

    let warnings = if diagnostics.is_empty() { String::new() } else { diagnostics.render(&file) };
    let report = compiled.model.report();
    let simulation = Simulation::new(compiled.model, compiled.domains);

    let stability = simulation.stability();
    let timestep = simulation
        .model()
        .timestep
        .filter(|dt| dt.is_finite() && *dt > 0.0)
        .unwrap_or(stability.preferred)
        .min(stability.max);
    let limiting_domain = simulation.limiting_domain().map(str::to_string);

    State::Ready(Box::new(Loaded {
        simulation,
        report,
        warnings,
        timestep,
        stability_limit: stability.max,
        limiting_domain,
    }))
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let palette = Palette::for_mode(self.mode);
        render::apply_theme(&ctx, &palette);

        // Advance before drawing, so the picture and the numbers describe the same
        // instant. Recording once per frame rather than once per step is also what
        // bounds the history without decimating on every step.
        if self.playing
            && let State::Ready(loaded) = &mut self.state
        {
            let dt = loaded.timestep;
            for _ in 0..self.steps_per_frame {
                loaded.simulation.step(dt);
            }
            loaded.simulation.observe();
            let time = loaded.simulation.clock().time;
            self.history.record(time, loaded.simulation.observations());

            // Stop at the first non-finite value rather than filling the plots with
            // NaN and burying it (NFR-007).
            if loaded.simulation.observations().first_non_finite().is_some() {
                self.playing = false;
            }
            ctx.request_repaint();
        }

        self.top_bar(ui, &palette);
        self.side_panel(ui, &palette);
        self.central(ui, &palette);
    }
}

impl ViewerApp {
    fn top_bar(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        containers::Panel::top("transport").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let ready = matches!(self.state, State::Ready(_));

                ui.add_enabled_ui(ready, |ui| {
                    let label = if self.playing { "⏸  pause" } else { "▶  play" };
                    if ui.button(label).clicked() {
                        self.playing = !self.playing;
                    }
                    if ui.button("⏭  step").clicked() {
                        self.playing = false;
                        if let State::Ready(loaded) = &mut self.state {
                            let dt = loaded.timestep;
                            loaded.simulation.step(dt);
                            loaded.simulation.observe();
                            let time = loaded.simulation.clock().time;
                            self.history.record(time, loaded.simulation.observations());
                        }
                    }
                });
                if ui.button("↺  reset").clicked() {
                    self.reset();
                }

                ui.separator();

                if let State::Ready(loaded) = &self.state {
                    let clock = loaded.simulation.clock();
                    ui.colored_label(palette.text_secondary, "t");
                    ui.colored_label(
                        palette.text_primary,
                        format!("{} s", render::format_value(clock.time)),
                    );
                    ui.colored_label(palette.text_secondary, "step");
                    ui.colored_label(palette.text_primary, format!("{}", clock.step));
                    ui.colored_label(palette.text_secondary, "dt");
                    ui.colored_label(
                        palette.text_primary,
                        format!("{} s", render::format_value(loaded.timestep)),
                    );
                }

                ui.separator();
                ui.colored_label(palette.text_secondary, "speed");
                ui.add(
                    egui::Slider::new(&mut self.steps_per_frame, 1..=200)
                        .logarithmic(true)
                        .suffix(" steps/frame"),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let theme = if self.mode == Mode::Dark { "☀ light" } else { "🌙 dark" };
                    if ui.button(theme).clicked() {
                        self.mode = if self.mode == Mode::Dark { Mode::Light } else { Mode::Dark };
                    }
                    ui.colored_label(
                        palette.text_muted,
                        self.path.file_name().unwrap_or_default().to_string_lossy(),
                    );
                });
            });
            ui.add_space(4.0);
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        containers::Panel::right("diagnostics").default_size(380.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let State::Ready(loaded) = &self.state else {
                    ui.colored_label(palette.status(Status::Critical), "this model did not compile");
                    return;
                };

                ui.add_space(6.0);
                ui.heading("diagnostics");

                // Timestep against the stability limit. Spec §17.3 wants the viewer to
                // answer "why is my timestep so small?" with the mechanism, so the
                // limiting domain is named.
                ui.add_space(4.0);
                ui.colored_label(palette.text_secondary, "stability");
                let verdict = render::stability_verdict(loaded.timestep, loaded.stability_limit);
                render::status_line(ui, palette, &verdict);
                if let Some(domain) = &loaded.limiting_domain {
                    ui.colored_label(palette.text_muted, format!("set by domain `{domain}`"));
                }

                // Conservation drift, one line per quantity that ought to be conserved.
                ui.add_space(8.0);
                ui.colored_label(palette.text_secondary, "conservation");
                let mut any = false;
                for series in self.history.series() {
                    if !series.is_invariant() || series.is_constant() {
                        continue;
                    }
                    let Some(drift) = self.history.drift_of(series) else { continue };
                    any = true;
                    ui.colored_label(palette.text_primary, short_name(&series.name));
                    let mut verdict = render::drift_verdict(drift.value);
                    // Name the yardstick rather than describing it. "of momentum_scale"
                    // points at a row of the values table two inches below; "of the
                    // scale being cancelled" leaves the reader to guess which scale.
                    let basis = match drift.basis {
                        DriftBasis::PublishedScale => self
                            .history
                            .scale_series_for(&series.name)
                            .map_or_else(|| drift.basis.phrase().to_string(), |scale| {
                                format!("of {}", short_name(&scale.name))
                            }),
                        other => other.phrase().to_string(),
                    };
                    verdict.detail = format!("{:+.3e} {basis}", drift.value);
                    render::status_line(ui, palette, &verdict);
                }
                if !any {
                    // Distinguish "not started" from "this scene has no invariant to
                    // report". A rigid world with gravity and a floor genuinely has
                    // none, and saying so beats an empty heading.
                    let claims_any = self.history.series().iter().any(Series::is_invariant);
                    ui.colored_label(
                        palette.text_muted,
                        if claims_any || self.history.sample_count() < 2 {
                            "nothing has drifted yet — press play"
                        } else {
                            "no conserved quantity here — gravity adds momentum and a static \
                             body absorbs it, so this domain publishes none"
                        },
                    );
                }

                // The values table. This is the relief channel that makes the plots
                // legible without relying on colour, and it is where a reader gets an
                // exact number rather than a position on an axis.
                ui.add_space(8.0);
                ui.colored_label(palette.text_secondary, "current values");
                egui::Grid::new("values").num_columns(2).striped(true).show(ui, |ui| {
                    for series in self.history.series() {
                        let Some(latest) = series.latest() else { continue };
                        ui.colored_label(palette.text_secondary, short_name(&series.name));
                        let color = if latest.is_finite() {
                            palette.text_primary
                        } else {
                            palette.status(Status::Critical)
                        };
                        ui.colored_label(
                            color,
                            format!("{} {}", render::format_value(latest), unit_label(&series.unit)),
                        );
                        ui.end_row();
                    }
                });

                // One plot per unit. Never two units on one pair of axes.
                let groups = self.history.by_unit();
                let plottable = groups
                    .iter()
                    .any(|(_, members)| members.iter().any(|s| self.history.is_worth_plotting(s)));
                if plottable {
                    ui.add_space(10.0);
                    ui.colored_label(palette.text_secondary, "over time");
                    for (unit, members) in &groups {
                        plot_group(ui, palette, &self.history, unit, members);
                    }
                }

                ui.add_space(10.0);
                ui.checkbox(&mut self.show_contracts, "solver contracts");
                if self.show_contracts {
                    for spec in &loaded.simulation.model().domains {
                        if let Some(contract) = spec.contract {
                            ui.add_space(4.0);
                            ui.monospace(contract.report());
                        }
                    }
                }
                ui.checkbox(&mut self.show_report, "model report");
                if self.show_report {
                    ui.monospace(&loaded.report);
                }
                if !loaded.warnings.is_empty() {
                    ui.add_space(6.0);
                    ui.colored_label(palette.status(Status::Warning), "compiler warnings");
                    ui.monospace(&loaded.warnings);
                }
                ui.add_space(12.0);
            });
        });
    }

    fn central(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        containers::CentralPanel::default().show(ui, |ui| {
            let State::Ready(loaded) = &self.state else {
                let State::Failed(message) = &self.state else { return };
                ui.add_space(12.0);
                ui.colored_label(palette.status(Status::Critical), "this model did not compile");
                ui.add_space(6.0);
                ui.monospace(message);
                ui.add_space(6.0);
                ui.colored_label(palette.text_muted, "fix the file and press reset");
                return;
            };

            let channels = loaded.simulation.render_channels();
            if channels.is_empty() {
                ui.colored_label(palette.text_muted, "this model exposes nothing to draw");
                return;
            }

            // Contacts are drawn over the bodies rather than in a panel of their own,
            // so they must not claim a share of the height — otherwise a rigid scene
            // gets half the picture it should and nothing says why.
            let drawn = channels
                .iter()
                .filter(|c| !matches!(c, RenderChannel::Contacts { .. }))
                .count()
                .max(1);
            let available = ui.available_height();
            let per_channel = (available / drawn as f32).max(160.0);

            // Contacts belong on top of the bodies they were found between, not in a
            // panel of their own where a reader would have to align two pictures by eye.
            let contacts =
                channels.iter().find(|c| matches!(c, RenderChannel::Contacts { .. }));

            egui::ScrollArea::vertical().show(ui, |ui| {
                for channel in &channels {
                    draw_channel(
                        ui,
                        channel,
                        contacts,
                        per_channel,
                        self.colormap,
                        self.mode,
                        palette,
                        &mut self.textures,
                    );
                    ui.add_space(10.0);
                }

                // Colourmap controls belong to fields. A rigid scene has no continuous
                // encoding to configure, and offering the choice anyway invites a reader
                // to look for the field it applies to.
                if channels.iter().any(|c| matches!(c, RenderChannel::Scalar { .. })) {
                    ui.horizontal(|ui| {
                        ui.colored_label(palette.text_secondary, "colour");
                        for map in [Colormap::Sequential, Colormap::Diverging] {
                            ui.radio_value(&mut self.colormap, map, map.label());
                        }
                    });
                    ui.colored_label(
                        palette.text_muted,
                        "cells are drawn unsmoothed, so the grid you see is the grid being solved",
                    );
                }
            });
        });
    }
}

/// Draw one render channel and its scale.
#[allow(clippy::too_many_arguments)]
fn draw_channel(
    ui: &mut egui::Ui,
    channel: &RenderChannel<'_>,
    // The contacts published alongside a body channel, drawn into the same rectangle.
    contacts: Option<&RenderChannel<'_>>,
    height: f32,
    colormap: Colormap,
    mode: Mode,
    palette: &Palette,
    textures: &mut BTreeMap<String, TextureHandle>,
) {
    // Contacts are drawn over the bodies, so a heading of their own would sit above
    // nothing.
    if matches!(channel, RenderChannel::Contacts { .. }) {
        return;
    }

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.colored_label(palette.text_primary, channel.name());
        ui.colored_label(palette.text_muted, channel.describe());
    });

    match channel {
        RenderChannel::Scalar { name, field, grid, unit } => {
            let rendered = render::field_to_image(field, colormap, mode, palette);

            let handle = match textures.get_mut(*name) {
                Some(handle) => {
                    handle.set(rendered.image.clone(), TEXTURE);
                    handle.clone()
                }
                None => {
                    let handle = ui.ctx().load_texture(*name, rendered.image.clone(), TEXTURE);
                    textures.insert((*name).to_string(), handle.clone());
                    handle
                }
            };

            // Preserve the domain's aspect ratio: a square chamber must not be drawn
            // as a rectangle, or every length read off the picture is wrong.
            let extent = grid.extent();
            let aspect = (extent[0] / extent[1]) as f32;
            let plot_height = (height - 60.0).max(120.0);
            let width = (plot_height * aspect).min(ui.available_width() - 8.0);
            let size = egui::Vec2::new(width, width / aspect);
            ui.add(egui::Image::new(&handle).fit_to_exact_size(size));

            if rendered.is_flat() {
                ui.colored_label(
                    palette.text_muted,
                    format!(
                        "flat to within round-off at {} {} — drawn as one colour rather than \
                         amplifying the noise",
                        render::format_value(rendered.mean),
                        unit_label(unit)
                    ),
                );
            } else {
                render::color_scale(
                    ui,
                    colormap,
                    mode,
                    palette,
                    rendered.min,
                    rendered.max,
                    unit_label(unit),
                );
            }
            ui.colored_label(
                palette.text_muted,
                format!(
                    "min {}   mean {}   max {}",
                    render::format_value(rendered.min),
                    render::format_value(rendered.mean),
                    render::format_value(rendered.max)
                ),
            );
            if rendered.non_finite > 0 {
                render::status_line(
                    ui,
                    palette,
                    &render::Verdict {
                        status: Status::Critical,
                        label: "non-finite cells".to_string(),
                        detail: format!("{} of them, drawn in red", rendered.non_finite),
                    },
                );
            }
        }

        RenderChannel::Bodies { x, origin, extent, .. } => {
            let aspect = (extent[0] / extent[1]) as f32;
            let plot_height = (height - 40.0).max(120.0);
            let width = (plot_height * aspect).min(ui.available_width() - 8.0);
            let size = egui::Vec2::new(width, width / aspect);

            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, palette.plane);
            let view = render::WorldView::new(rect, *origin, *extent);
            let outside = render::draw_bodies(ui.painter(), view, channel, palette);
            // Contacts go into the same rectangle, on top. Giving them their own panel
            // would put a set of points beside the geometry they belong to and leave
            // the reader to align the two by eye.
            if let Some(contacts) = contacts {
                render::draw_contacts(ui.painter(), view, contacts, palette);
            }
            ui.painter().rect_stroke(
                rect,
                2.0,
                egui::Stroke::new(1.0, palette.border),
                egui::StrokeKind::Inside,
            );

            let mut caption = format!(
                "{} bodies over {} × {} m",
                x.len(),
                render::format_value(extent[0]),
                render::format_value(extent[1])
            );
            if let Some(RenderChannel::Contacts { x: cx, depth, .. }) = contacts {
                let deepest = depth.iter().copied().fold(0.0, f64::max);
                caption.push_str(&format!(
                    ", {} contacts, deepest {} m",
                    cx.len(),
                    render::format_value(deepest)
                ));
            }
            if outside > 0 {
                caption.push_str(&format!(", {outside} outside the view"));
            }
            ui.colored_label(palette.text_muted, caption);
        }

        // Drawn over the bodies rather than on its own, so it never appears here.
        RenderChannel::Contacts { .. } => {}

        RenderChannel::Particles { x, y, origin, extent, .. } => {
            let aspect = (extent[0] / extent[1]) as f32;
            let plot_height = (height - 40.0).max(120.0);
            let width = (plot_height * aspect).min(ui.available_width() - 8.0);
            let size = egui::Vec2::new(width, width / aspect);

            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, palette.plane);
            let outside = render::draw_particles(ui.painter(), rect, x, y, *origin, *extent, palette);
            ui.painter().rect_stroke(
                rect,
                2.0,
                egui::Stroke::new(1.0, palette.border),
                egui::StrokeKind::Inside,
            );

            let mut caption = format!(
                "{} particles over {} × {} m",
                x.len(),
                render::format_value(extent[0]),
                render::format_value(extent[1])
            );
            if outside > 0 {
                caption.push_str(&format!(", {outside} outside the view"));
            }
            ui.colored_label(palette.text_muted, caption);
        }
    }
}

/// Plot every series sharing a unit, on one axis.
fn plot_group(
    ui: &mut egui::Ui,
    palette: &Palette,
    history: &History,
    unit: &str,
    members: &[&crate::history::Series],
) {
    // Two kinds of series are excluded. One that never changes is a straight line
    // that tells the reader nothing and takes a colour slot from one that would. One
    // that varies only in its last couple of digits is worse: the plot auto-scales, so
    // round-off gets drawn as a dramatic oscillation with fifteen-digit axis labels,
    // and beside a series that genuinely moves it reads as instability. Both are
    // reported as numbers in the values table instead, which is where an exact figure
    // belongs anyway. A third case joins them: a quantity that is nothing against the
    // scale its domain published — a net momentum of 1e-13 where the momenta being
    // cancelled are of order 100.
    let varying: Vec<&crate::history::Series> =
        members.iter().copied().filter(|series| history.is_worth_plotting(series)).collect();
    if varying.is_empty() {
        return;
    }

    ui.add_space(6.0);
    ui.colored_label(palette.text_muted, unit_label(unit));

    // The palette validates eight adjacent slots; past that, colours would repeat and
    // two series would claim the same identity.
    let shown = varying.len().min(Palette::SERIES_SLOTS);
    // Leave room for the outer half of the rightmost tick label, which egui centres
    // on a tick sitting exactly at the plot's right edge and would otherwise clip
    // against the panel — a time axis ending in "20" instead of "200".
    let width = (ui.available_width() - 14.0).max(120.0);
    let plot = Plot::new(format!("plot-{unit}"))
        .height(150.0)
        .width(width)
        .allow_scroll(false)
        .x_axis_label("t (s)");

    plot.show(ui, |plot_ui| {
        for (slot, series) in varying.iter().take(shown).enumerate() {
            let points: PlotPoints = series.points.clone().into();
            plot_ui.line(
                Line::new(short_name(&series.name), points)
                    .color(palette.series(slot))
                    .width(2.0),
            );
        }
    });

    let names: Vec<&str> =
        varying.iter().take(shown).map(|series| short_name(&series.name)).collect();
    render::legend_row(ui, palette, &names);

    if varying.len() > shown {
        ui.colored_label(
            palette.text_muted,
            format!("{} more in the values table above", varying.len() - shown),
        );
    }

    // Say what was left out, so a missing curve is a stated decision rather than a
    // reader wondering where their conserved quantity went.
    let flat: Vec<&str> = members
        .iter()
        .filter(|series| !history.is_worth_plotting(series))
        .map(|series| short_name(&series.name))
        .collect();
    if !flat.is_empty() {
        ui.colored_label(
            palette.text_muted,
            format!("constant to round-off, not plotted: {}", flat.join(", ")),
        );
    }
}

/// Drop the domain prefix from an observation name, so legends stay readable.
fn short_name(name: &str) -> &str {
    name.split_once('.').map_or(name, |(_, rest)| rest)
}

/// A unit string fit for a label.
fn unit_label(unit: &str) -> &str {
    match unit {
        "1" => "",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_drop_the_domain_prefix() {
        assert_eq!(short_name("temperature.integral"), "integral");
        assert_eq!(short_name("bare"), "bare");
        assert_eq!(short_name("a.b.c"), "b.c");
    }

    #[test]
    fn dimensionless_units_render_as_no_label() {
        assert_eq!(unit_label("1"), "");
        assert_eq!(unit_label("J"), "J");
    }

    #[test]
    fn a_missing_file_reports_rather_than_panicking() {
        match load(Path::new("definitely-not-here.lattice")) {
            State::Failed(message) => assert!(message.contains("cannot read"), "{message}"),
            State::Ready(_) => panic!("a missing file should not load"),
        }
    }

    #[test]
    fn a_model_that_does_not_compile_reports_its_diagnostics() {
        let dir = std::env::temp_dir().join("lattice-viewer-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("broken.lattice");
        std::fs::write(&path, "project p { grid g { size: [4, 4]; } }").unwrap();

        match load(&path) {
            State::Failed(message) => {
                assert!(message.contains("extent"), "{message}");
                assert!(message.contains("-->"), "diagnostics should carry a position: {message}");
            }
            State::Ready(_) => panic!("this model should not compile"),
        }
    }

    /// A model with no solvers compiles but cannot run; saying so beats an empty window.
    #[test]
    fn a_model_with_no_solvers_explains_itself() {
        let dir = std::env::temp_dir().join("lattice-viewer-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.lattice");
        std::fs::write(&path, "project p { dimensions: 2; }").unwrap();

        match load(&path) {
            State::Failed(message) => assert!(message.contains("no solvers"), "{message}"),
            State::Ready(_) => panic!("a model with no solvers should not be runnable"),
        }
    }

    #[test]
    fn a_valid_model_loads_with_a_usable_timestep() {
        let dir = std::env::temp_dir().join("lattice-viewer-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("good.lattice");
        std::fs::write(
            &path,
            r#"
project good {
  duration: 10 second;
  grid g { size: [16, 16]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin {
    diffusivity: 1e-4 meter^2/second;
    boundary: insulated;
  }
  solve heat(t) with crank_nicolson(dt=0.1 second);
}
"#,
        )
        .unwrap();

        match load(&path) {
            State::Ready(loaded) => {
                assert!((loaded.timestep - 0.1).abs() < 1e-12);
                assert!(loaded.report.contains("grid2d.heat"));
                assert_eq!(loaded.simulation.render_channels().len(), 1);
            }
            State::Failed(message) => panic!("{message}"),
        }
    }
}
