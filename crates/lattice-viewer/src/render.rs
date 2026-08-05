//! Turning domain state into pixels.
//!
//! Two marks, both governed by the rule that a picture must not imply structure the
//! data does not have:
//!
//! - A **scalar field** becomes an image through a one-hue sequential ramp (or a
//!   diverging one when the field has a signed zero), always accompanied by a scale
//!   legend and exact numbers in the panel. Colour alone never carries a value.
//! - **Particles** become a scatter, drawn with a surface-coloured ring so overlapping
//!   marks stay countable without a border.
//!
//! Non-finite cells are painted in the reserved *critical* status colour rather than
//! being clamped into the ramp. A `NaN` is the single most important thing a field can
//! contain, and normalizing it away is how a broken run comes to look fine
//! (spec NFR-007).

use eframe::egui::{self, ColorImage, Pos2, Rect, Stroke, Vec2};
use lattice_ir::{RenderChannel, ScalarField};

use crate::palette::{self, Colormap, Mode, Palette, Status};

/// A rendered field plus the numbers a reader needs to interpret it.
#[derive(Clone, Debug)]
pub struct FieldImage {
    /// The pixels.
    pub image: ColorImage,
    /// Smallest finite interior value.
    pub min: f64,
    /// Largest finite interior value.
    pub max: f64,
    /// Mean of the finite interior values.
    pub mean: f64,
    /// How many cells held a non-finite value.
    pub non_finite: usize,
}

impl FieldImage {
    /// The span of the data range.
    pub fn span(&self) -> f64 {
        self.max - self.min
    }

    /// True when every cell holds effectively the same value.
    ///
    /// A flat field normalized to the full ramp shows dramatic structure built
    /// entirely from round-off, so the caller renders it flat and says so instead.
    pub fn is_flat(&self) -> bool {
        let magnitude = self.max.abs().max(self.min.abs()).max(f64::MIN_POSITIVE);
        self.span() <= 1e-12 * magnitude
    }
}

/// Render a scalar field.
///
/// `symmetric` centres a diverging map on zero so that equal magnitudes either side
/// of it are equally far from the neutral midpoint.
pub fn field_to_image(
    field: &ScalarField,
    map: Colormap,
    mode: Mode,
    palette: &Palette,
) -> FieldImage {
    let (nx, ny) = (field.nx(), field.ny());

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut finite = 0usize;
    let mut non_finite = 0usize;
    for j in 0..ny {
        for &value in field.row(j) {
            if value.is_finite() {
                min = min.min(value);
                max = max.max(value);
                sum += value;
                finite += 1;
            } else {
                non_finite += 1;
            }
        }
    }
    if finite == 0 {
        min = 0.0;
        max = 0.0;
    }
    let mean = if finite > 0 { sum / finite as f64 } else { 0.0 };

    // A diverging map is anchored on zero, not on the data's own midpoint —
    // otherwise "the neutral colour" would drift with the data and stop meaning
    // "nothing".
    let extreme = max.abs().max(min.abs()).max(f64::MIN_POSITIVE);
    let span = max - min;
    let flat = span <= 1e-12 * extreme;

    let critical = palette.status(Status::Critical);
    let mut pixels = Vec::with_capacity(nx * ny);
    // Image space runs top-down; grid space runs bottom-up. Increasing y must appear
    // higher on screen, or every buoyancy and gravity scene reads upside down.
    for row in 0..ny {
        let j = ny - 1 - row;
        for &value in field.row(j) {
            if !value.is_finite() {
                pixels.push(critical);
                continue;
            }
            let t = match map {
                Colormap::Sequential => {
                    if flat { 0.5 } else { (value - min) / span }
                }
                Colormap::Diverging => value / extreme,
            };
            pixels.push(palette::sample(map, mode, t));
        }
    }

    FieldImage { image: ColorImage::new([nx, ny], pixels), min, max, mean, non_finite }
}

/// Draw particles into `rect`, mapping world coordinates through `origin`/`extent`.
///
/// Marks get a surface-coloured ring rather than a dark border: a border around every
/// mark reads as a grid, while a ring in the surface colour just separates neighbours.
pub fn draw_particles(
    painter: &egui::Painter,
    rect: Rect,
    x: &[f64],
    y: &[f64],
    origin: [f64; 2],
    extent: [f64; 2],
    palette: &Palette,
) -> usize {
    let fill = palette.series(0);
    let critical = palette.status(Status::Critical);

    // Enough to see, small enough to stay thin at a few thousand marks.
    let radius = (rect.width().min(rect.height()) / 160.0).clamp(1.5, 4.0);
    let ring = Stroke::new(1.0, palette.surface);

    let mut outside = 0usize;
    for (&wx, &wy) in x.iter().zip(y) {
        if !wx.is_finite() || !wy.is_finite() {
            // A particle at NaN has no position to draw, so mark the corner rather
            // than dropping it silently.
            painter.circle_filled(rect.left_top() + Vec2::new(6.0, 6.0), radius * 1.5, critical);
            outside += 1;
            continue;
        }
        let fx = (wx - origin[0]) / extent[0];
        let fy = (wy - origin[1]) / extent[1];
        if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
            outside += 1;
            continue;
        }
        let position = Pos2::new(
            rect.left() + fx as f32 * rect.width(),
            // Flip y so increasing y draws upward.
            rect.bottom() - fy as f32 * rect.height(),
        );
        painter.circle(position, radius, fill, ring);
    }
    outside
}

/// Maps world coordinates onto a screen rectangle, with y increasing upward.
///
/// Shared by the body and contact renderers so a contact point lands exactly where the
/// surfaces it belongs to are drawn. Two copies of this arithmetic would drift apart,
/// and the drift would be invisible until someone tried to debug a contact.
#[derive(Clone, Copy, Debug)]
pub struct WorldView {
    rect: Rect,
    origin: [f64; 2],
    extent: [f64; 2],
}

impl WorldView {
    /// A view of `origin`..`origin + extent` drawn into `rect`.
    pub fn new(rect: Rect, origin: [f64; 2], extent: [f64; 2]) -> WorldView {
        WorldView { rect, origin, extent }
    }

    /// World point to screen position. `None` for a non-finite coordinate.
    pub fn project(&self, world: [f64; 2]) -> Option<Pos2> {
        if !(world[0].is_finite() && world[1].is_finite()) {
            return None;
        }
        let fx = (world[0] - self.origin[0]) / self.extent[0];
        let fy = (world[1] - self.origin[1]) / self.extent[1];
        Some(Pos2::new(
            self.rect.left() + fx as f32 * self.rect.width(),
            // Flip y so increasing y draws upward.
            self.rect.bottom() - fy as f32 * self.rect.height(),
        ))
    }

    /// Screen pixels per world metre, on the x axis.
    pub fn scale(&self) -> f32 {
        if self.extent[0] > 0.0 {
            self.rect.width() / self.extent[0] as f32
        } else {
            1.0
        }
    }
}

/// Draw rigid bodies as filled outlines.
///
/// Static bodies get the surface's own muted ink and dynamic ones a series colour, so
/// the two questions a rigid scene provokes — "why is the ground falling" and "why is
/// the crate not" — are answered by looking at it. That distinction is carried by
/// lightness as well as hue, so it survives greyscale and colour vision deficiency.
///
/// Returns how many bodies fell outside the view.
pub fn draw_bodies(
    painter: &egui::Painter,
    view: WorldView,
    channel: &RenderChannel<'_>,
    palette: &Palette,
) -> usize {
    let RenderChannel::Bodies { x, is_static, .. } = channel else {
        return 0;
    };

    let dynamic = palette.series(0);
    let critical = palette.status(Status::Critical);
    let mut world = Vec::with_capacity(32);
    let mut screen: Vec<Pos2> = Vec::with_capacity(32);
    let mut outside = 0usize;

    for index in 0..x.len() {
        channel.body_outline(index, &mut world);
        if world.is_empty() {
            continue;
        }
        screen.clear();
        let mut lost = false;
        for point in &world {
            match view.project(*point) {
                Some(position) => screen.push(position),
                None => lost = true,
            }
        }
        if lost || screen.len() < 2 {
            // A body at NaN has no position to draw, so mark the corner rather than
            // dropping it silently (NFR-007).
            painter.circle_filled(view.rect.left_top() + Vec2::new(6.0, 6.0), 5.0, critical);
            outside += 1;
            continue;
        }
        if screen.iter().all(|p| !view.rect.contains(*p)) {
            outside += 1;
        }

        // Scenery in neutral grey, movers in a saturated hue: the difference is in
        // lightness as well as chroma, so it survives greyscale and colour vision
        // deficiency without needing a legend.
        let is_wall = is_static[index];
        let fill = if is_wall { palette.axis } else { dynamic };
        let stroke = Stroke::new(1.5, if is_wall { palette.border } else { palette.text_primary });

        // A two-point outline is a segment: it has no interior, and filling one would
        // draw a wall that is not there.
        if screen.len() > 2 {
            painter.add(egui::Shape::convex_polygon(screen.clone(), fill, stroke));
        } else {
            painter.line_segment([screen[0], screen[1]], Stroke::new(2.5, palette.border));
        }
    }
    outside
}

/// Draw contact points and their normals.
///
/// The arrow points the way the solver is pushing, which is the single most useful
/// thing to see when a stack is misbehaving: jitter, sinking and sticking are
/// indistinguishable from outside and obvious once the normals are on screen. Depth is
/// shown by the marker's size rather than by colour, so the scale bar a colour encoding
/// would demand is not needed.
pub fn draw_contacts(
    painter: &egui::Painter,
    view: WorldView,
    channel: &RenderChannel<'_>,
    palette: &Palette,
) {
    let RenderChannel::Contacts { x, y, normal_x, normal_y, depth, .. } = channel else {
        return;
    };

    // A fixed screen length, so a normal stays legible at any zoom. It is a direction,
    // not a magnitude — drawing it proportional to anything would invite a reader to
    // measure it.
    let arrow = (view.rect.width() * 0.02).clamp(6.0, 18.0);
    let color = palette.status(Status::Warning);

    for index in 0..x.len() {
        let Some(at) = view.project([x[index], y[index]]) else { continue };
        if !view.rect.contains(at) {
            continue;
        }
        let normal = Vec2::new(normal_x[index] as f32, -normal_y[index] as f32);
        let length = normal.length();
        if length > 0.0 {
            let tip = at + normal / length * arrow;
            painter.line_segment([at, tip], Stroke::new(1.5, color));
        }
        // Deeper overlap draws a larger dot, bounded so a badly overlapped scene stays
        // readable rather than becoming one enormous blob.
        let size = (2.0_f32 + (depth[index] * f64::from(view.scale())) as f32).clamp(2.0, 6.0);
        painter.circle_filled(at, size, color);
    }
}

/// Draw the colour-scale legend for a continuous map.
///
/// Required, not optional: a continuous encoding with no scale is unreadable, and the
/// numbers at each end are what let a viewer tell "hot" from "0.3 K above ambient".
pub fn color_scale(
    ui: &mut egui::Ui,
    map: Colormap,
    mode: Mode,
    palette: &Palette,
    min: f64,
    max: f64,
    unit: &str,
) {
    let height = 12.0;
    let width = ui.available_width().min(320.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), egui::Sense::hover());
    let painter = ui.painter();

    // One thin vertical strip per pixel column, so the gradient is the ramp itself
    // rather than a two-stop approximation of it.
    let steps = rect.width().max(1.0) as usize;
    for step in 0..steps {
        let fraction = step as f32 / steps.max(1) as f32;
        let t = match map {
            Colormap::Sequential => f64::from(fraction),
            Colormap::Diverging => f64::from(fraction) * 2.0 - 1.0,
        };
        let x = rect.left() + fraction * rect.width();
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(1.5, height)),
            0.0,
            palette::sample(map, mode, t),
        );
    }
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0, palette.border),
        egui::StrokeKind::Inside,
    );

    // Labels wear text tokens, never the series or ramp colour.
    let (low, high) = match map {
        Colormap::Sequential => (min, max),
        Colormap::Diverging => {
            let extreme = max.abs().max(min.abs());
            (-extreme, extreme)
        }
    };
    ui.horizontal(|ui| {
        ui.colored_label(palette.text_muted, format_value(low));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.colored_label(palette.text_muted, format!("{} {unit}", format_value(high)));
        });
    });
}

/// Format a number for a label: fixed where readable, scientific where not.
pub fn format_value(value: f64) -> String {
    if !value.is_finite() {
        return format!("{value}");
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs();
    if (1e-3..1e6).contains(&magnitude) {
        // Enough digits to distinguish neighbouring cells, few enough to read.
        format!("{value:.4}")
    } else {
        format!("{value:.3e}")
    }
}

/// A status verdict plus the words that carry it without colour.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// Which status band.
    pub status: Status,
    /// The short label shown beside the icon.
    pub label: String,
    /// The supporting detail.
    pub detail: String,
}

/// Classify a timestep against a domain's stability limit.
pub fn stability_verdict(dt: f64, limit: f64) -> Verdict {
    if !limit.is_finite() {
        return Verdict {
            status: Status::Good,
            label: "unconditional".to_string(),
            // Not "an implicit scheme": the rigid module is explicit and symplectic and
            // still has no step limit of its own. What the two share is that nothing in
            // the *integrator* bounds dt — which is what this line is reporting.
            detail: "the integrator imposes no step limit".to_string(),
        };
    }
    let margin = dt / limit;
    let (status, label) = if margin >= 1.0 {
        (Status::Critical, "unstable")
    } else if margin >= 0.9 {
        (Status::Serious, "at the limit")
    } else if margin >= 0.5 {
        (Status::Warning, "close to the limit")
    } else {
        (Status::Good, "stable")
    };
    Verdict {
        status,
        label: label.to_string(),
        detail: format!("dt is {:.0}% of the {} s limit", margin * 100.0, format_value(limit)),
    }
}

/// Classify how far a conserved quantity has drifted, relative to a meaningful scale.
///
/// The bands are set by what good solvers actually deliver, not by round numbers. A
/// finite-volume diffusion integral on a closed domain conserves to round-off, so
/// 1e-9 there is a real problem. A Lennard-Jones fluid with a truncated potential
/// exchanges a little energy every time a particle crosses the cutoff, so 1e-4 there
/// is expected and the module's contract says so. One threshold cannot be strict
/// enough for the first and fair to the second, so the wording states the magnitude
/// rather than pronouncing on it, and only a genuine runaway is called critical.
pub fn drift_verdict(relative: f64) -> Verdict {
    let magnitude = relative.abs();
    let (status, label) = if !relative.is_finite() {
        (Status::Critical, "non-finite")
    } else if magnitude > 1e-1 {
        (Status::Critical, "not conserved")
    } else if magnitude > 1e-3 {
        (Status::Serious, "drifting")
    } else if magnitude > 1e-6 {
        (Status::Warning, "drifting slowly")
    } else if magnitude > 1e-9 {
        (Status::Good, "conserved to 1e-6")
    } else {
        (Status::Good, "conserved to round-off")
    };
    Verdict { status, label: label.to_string(), detail: format!("{relative:+.3e} relative") }
}

/// Draw a plot's legend beneath it rather than inside it.
///
/// A floating legend has to land somewhere in the plot area, and on a live chart there
/// is often no free corner: the three energy series of a Lennard-Jones run span +156 J
/// to -368 J and reach every edge, so wherever the box goes it covers a curve. Trying
/// to pick the emptiest corner only chooses which curve gets hidden. One row of height
/// underneath can never hide anything.
///
/// The swatch carries the identity; the text stays in secondary ink. Colouring the
/// label to match its series would make the legend the only place the mapping is
/// stated *and* make it unreadable to the reader who needs it most.
pub fn legend_row(ui: &mut egui::Ui, palette: &Palette, names: &[&str]) {
    /// Diameter of the swatch. The dataviz floor for a marker is 8px.
    const SWATCH: f32 = 9.0;

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (index, name) in names.iter().enumerate() {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(SWATCH, SWATCH), egui::Sense::hover());
            ui.painter().circle_filled(rect.center(), SWATCH / 2.0, palette.series(index));
            ui.colored_label(palette.text_secondary, *name);
            ui.add_space(8.0);
        }
    });
}

/// Draw a status line: icon, label, then detail — so the meaning survives without
/// colour, in print, and under forced-colours.
///
/// Wraps rather than clipping. The detail is the part that explains the verdict, and a
/// panel narrow enough to cut off "of the initial value" would leave a reader with a
/// number and no idea what it is a fraction of.
pub fn status_line(ui: &mut egui::Ui, palette: &Palette, verdict: &Verdict) {
    ui.horizontal_wrapped(|ui| {
        let color = palette.status(verdict.status);
        ui.colored_label(color, verdict.status.icon());
        ui.colored_label(color, &verdict.label);
        ui.colored_label(palette.text_muted, &verdict.detail);
    });
}

/// The surface colour for an `egui` visuals override.
pub fn apply_theme(ctx: &egui::Context, palette: &Palette) {
    let mut visuals = match palette.mode {
        Mode::Light => egui::Visuals::light(),
        Mode::Dark => egui::Visuals::dark(),
    };
    visuals.panel_fill = palette.surface;
    visuals.window_fill = palette.surface;
    visuals.extreme_bg_color = palette.plane;
    visuals.override_text_color = Some(palette.text_primary);
    ctx.set_visuals(visuals);
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::Color32;
    use lattice_ir::Grid2d;

    fn field_with(nx: usize, ny: usize, f: impl Fn(usize, usize) -> f64) -> ScalarField {
        let grid = Grid2d::new(nx, ny, [1.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        for j in 0..ny {
            for i in 0..nx {
                field.set(i, j, f(i, j));
            }
        }
        field
    }

    fn palette() -> Palette {
        Palette::for_mode(Mode::Light)
    }

    #[test]
    fn a_field_image_has_one_pixel_per_interior_cell() {
        let field = field_with(16, 9, |i, _| f64::from(i as u32));
        let rendered = field_to_image(&field, Colormap::Sequential, Mode::Light, &palette());
        assert_eq!(rendered.image.size, [16, 9]);
        assert_eq!(rendered.image.pixels.len(), 16 * 9);
    }

    #[test]
    fn statistics_ignore_the_halo_and_report_the_range() {
        let field = field_with(4, 4, |i, j| f64::from((i + j) as u32));
        let rendered = field_to_image(&field, Colormap::Sequential, Mode::Light, &palette());
        assert_eq!(rendered.min, 0.0);
        assert_eq!(rendered.max, 6.0);
        assert!((rendered.mean - 3.0).abs() < 1e-12);
        assert_eq!(rendered.non_finite, 0);
    }

    /// Increasing y must draw upward. Getting this backwards makes every gravity and
    /// buoyancy scene look wrong in a way that is easy to mistake for a physics bug.
    #[test]
    fn increasing_y_renders_upward() {
        let field = field_with(2, 2, |_, j| f64::from(j as u32));
        let rendered = field_to_image(&field, Colormap::Sequential, Mode::Light, &palette());

        // In light mode the ramp darkens with magnitude, so the high-y row (drawn
        // first, at the top) must be darker than the low-y row.
        let top = rendered.image.pixels[0];
        let bottom = rendered.image.pixels[2];
        let luminance = |c: Color32| u32::from(c.r()) + u32::from(c.g()) + u32::from(c.b());
        assert!(luminance(top) < luminance(bottom), "high y should be at the top and darker");
    }

    /// A NaN is painted in the reserved critical colour, not clamped into the ramp.
    #[test]
    fn non_finite_cells_are_marked_and_counted() {
        let mut field = field_with(4, 4, |_, _| 1.0);
        field.set(1, 1, f64::NAN);
        field.set(2, 2, f64::INFINITY);
        let rendered = field_to_image(&field, Colormap::Sequential, Mode::Light, &palette());

        assert_eq!(rendered.non_finite, 2);
        let critical = palette().status(Status::Critical);
        assert!(rendered.image.pixels.contains(&critical), "a NaN must be visible");
        // ...and the finite statistics are unaffected by them.
        assert_eq!(rendered.min, 1.0);
        assert_eq!(rendered.max, 1.0);
    }

    /// A field whose values differ only by round-off must not be stretched across the
    /// whole ramp — that manufactures structure out of floating-point noise.
    #[test]
    fn a_flat_field_renders_flat() {
        let field = field_with(8, 8, |i, _| 300.0 + f64::from(i as u32) * 1e-15);
        let rendered = field_to_image(&field, Colormap::Sequential, Mode::Light, &palette());
        assert!(rendered.is_flat());

        let first = rendered.image.pixels[0];
        assert!(
            rendered.image.pixels.iter().all(|p| *p == first),
            "a field flat to round-off should render as one colour"
        );
    }

    /// A diverging map is anchored on zero, so the neutral colour always means
    /// "nothing" rather than "the middle of whatever happens to be on screen".
    #[test]
    fn a_diverging_map_is_anchored_on_zero() {
        // Values from -1 to +3: zero is *not* the data midpoint.
        let field = field_with(5, 1, |i, _| f64::from(i as u32) - 1.0);
        let rendered = field_to_image(&field, Colormap::Diverging, Mode::Light, &palette());

        // The cell holding exactly zero must get the neutral midpoint colour.
        let neutral = palette::sample(Colormap::Diverging, Mode::Light, 0.0);
        assert_eq!(rendered.image.pixels[1], neutral, "the zero cell should be neutral");
        // And the negative cell must not.
        assert_ne!(rendered.image.pixels[0], neutral);
    }

    #[test]
    fn an_all_nan_field_does_not_divide_by_zero() {
        let field = field_with(3, 3, |_, _| f64::NAN);
        let rendered = field_to_image(&field, Colormap::Sequential, Mode::Light, &palette());
        assert_eq!(rendered.non_finite, 9);
        assert_eq!(rendered.min, 0.0);
        assert!(rendered.mean.is_finite());
    }

    #[test]
    fn stability_verdicts_escalate_with_the_margin() {
        assert_eq!(stability_verdict(0.1, f64::INFINITY).status, Status::Good);
        assert_eq!(stability_verdict(0.1, 1.0).status, Status::Good);
        assert_eq!(stability_verdict(0.6, 1.0).status, Status::Warning);
        assert_eq!(stability_verdict(0.95, 1.0).status, Status::Serious);
        assert_eq!(stability_verdict(1.5, 1.0).status, Status::Critical);
    }

    #[test]
    fn drift_verdicts_escalate_with_the_magnitude() {
        assert_eq!(drift_verdict(1e-16).status, Status::Good);
        assert_eq!(drift_verdict(1e-7).status, Status::Good);
        assert_eq!(drift_verdict(1e-4).status, Status::Warning);
        assert_eq!(drift_verdict(1e-2).status, Status::Serious);
        assert_eq!(drift_verdict(0.5).status, Status::Critical);
        assert_eq!(drift_verdict(f64::NAN).status, Status::Critical);
        // Sign does not change the severity.
        assert_eq!(drift_verdict(-0.5).status, Status::Critical);
    }

    /// A Lennard-Jones fluid's energy drifts by ~1e-4 because of cutoff crossings,
    /// which its contract declares. Calling that "not conserved" would train a reader
    /// to ignore the indicator.
    #[test]
    fn a_normal_molecular_dynamics_drift_is_not_alarming() {
        let verdict = drift_verdict(2.3e-4);
        assert_eq!(verdict.status, Status::Warning);
        assert!(verdict.label.contains("slowly"), "{}", verdict.label);
    }

    /// Every verdict must carry words, so meaning never rests on the colour.
    #[test]
    fn verdicts_carry_a_label_and_an_icon() {
        for verdict in [stability_verdict(1.5, 1.0), drift_verdict(1e-4)] {
            assert!(!verdict.label.is_empty());
            assert!(!verdict.detail.is_empty());
            assert!(!verdict.status.icon().is_empty());
        }
    }

    #[test]
    fn values_format_readably_across_magnitudes() {
        assert_eq!(format_value(0.0), "0");
        assert_eq!(format_value(273.15), "273.1500");
        assert!(format_value(1e-9).contains('e'));
        assert!(format_value(1e12).contains('e'));
        assert_eq!(format_value(f64::NAN), "NaN");
    }
}
