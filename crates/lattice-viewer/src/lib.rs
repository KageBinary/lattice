//! The desktop viewer.
//!
//! Spec §17 makes visualization a product surface rather than a convenience, and P7
//! states the reason: *"The renderer is not decoration. It must display fields, fluxes,
//! constraints, forces, residuals, conservation drift, timestep decisions, and
//! uncertainty."*
//!
//! This crate opens a window on a `.lattice` model: field heatmaps, particle scatter,
//! transport controls, live plots of every observed quantity, conservation drift with
//! a status verdict, the timestep against its stability limit, and the solver's own
//! contract.
//!
//! # Three rules the rendering obeys
//!
//! **Sequential colour means one hue.** Rainbow ramps invent visual boundaries the
//! data does not have, so a reader sees structure that belongs to the palette. See
//! [`palette`].
//!
//! **One unit per plot, one axis per plot.** Energy and momentum never share a pair of
//! axes: the alignment between two y-scales is arbitrary, so a dual-axis chart
//! manufactures a correlation. See [`history::History::by_unit`].
//!
//! **Colour never carries a value alone.** Every heatmap ships a scale legend and the
//! exact min/mean/max; every status ships an icon and a label; every plot ships a
//! legend and a values table.
//!
//! # Running
//!
//! ```no_run
//! lattice_viewer::run("examples/slab.lattice").unwrap();
//! ```

pub mod app;
pub mod gpu;
pub mod history;
pub mod palette;
pub mod render;

pub use app::ViewerApp;
pub use history::{History, Series};
pub use palette::{Colormap, Mode, Palette, Status};

use std::path::Path;

/// Open the viewer on a model file.
///
/// Blocks until the window closes.
pub fn run(path: impl AsRef<Path>) -> Result<(), String> {
    run_with(path, false)
}

/// Open the viewer, optionally running the model immediately.
pub fn run_with(path: impl AsRef<Path>, play: bool) -> Result<(), String> {
    let path = path.as_ref().to_path_buf();
    let title = format!(
        "lattice — {}",
        path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned()
        )
    );

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([720.0, 480.0])
            .with_title(&title),
        ..Default::default()
    };

    eframe::run_native(
        &title,
        options,
        Box::new(move |_cc| Ok(Box::new(ViewerApp::with_options(path, play)))),
    )
    .map_err(|error| format!("cannot open a window: {error}"))
}
