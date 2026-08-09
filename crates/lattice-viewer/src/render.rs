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
    field_to_image_about(field, map, mode, palette, 0.0)
}

/// The same, with a diverging map anchored on `neutral` rather than on zero.
///
/// # Why an anchor is worth a second entry point
///
/// A diverging map's whole job is to say which side of *something* a value falls on,
/// and the grey midpoint is where that something is. Zero is the right anchor for a
/// signed quantity — a velocity, a charge, a departure from equilibrium.
///
/// It is the wrong anchor for a temperature. A plate at 300 K painted with a diverging
/// map anchored on zero is uniformly one colour, because every value is on the same
/// side of zero and the interesting variation is a rounding error against 300. Anchored
/// on ambient, the same field says at a glance which spots are warmer and which are
/// cooler than the room — which is the question a reader actually has.
///
/// Sequential maps ignore `neutral`: they run from the data's own minimum to its
/// maximum, and have no midpoint to place.
pub fn field_to_image_about(
    field: &ScalarField,
    map: Colormap,
    mode: Mode,
    palette: &Palette,
    neutral: f64,
) -> FieldImage {
    field_to_image_inner(field, map, mode, palette, neutral, None)
}

/// A sequential ramp that starts at `floor` instead of at the data's own minimum.
///
/// # Why the default is wrong for some quantities
///
/// Normalizing a sequential map between the data's min and max is right for a
/// temperature: a plate between 300 K and 310 K should use the whole ramp, because the
/// ten kelvin is the story and zero kelvin is not on the table.
///
/// It is wrong for a concentration. A chamber holding nothing has min == max == 0, so
/// there is no range to normalize and the flat case picks the middle of the ramp — an
/// empty chamber rendered as a solid mid-tone, which reads as *uniformly full*. That is
/// the worst kind of chart error: not a picture that is hard to read, but one that
/// confidently says something false. Flooring the ramp at zero makes empty look empty
/// and keeps the shading comparable from one frame to the next.
///
/// Diverging maps ignore `floor` and keep their anchor, since their midpoint already
/// carries the meaning a floor would.
pub fn field_to_image_above(
    field: &ScalarField,
    map: Colormap,
    mode: Mode,
    palette: &Palette,
    floor: f64,
) -> FieldImage {
    field_to_image_inner(field, map, mode, palette, floor, Some(floor))
}

fn field_to_image_inner(
    field: &ScalarField,
    map: Colormap,
    mode: Mode,
    palette: &Palette,
    neutral: f64,
    floor: Option<f64>,
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

    // A diverging map is anchored on `neutral`, not on the data's own midpoint —
    // otherwise "the neutral colour" would drift with the data and stop meaning
    // anything in particular. The scale is the larger departure from the anchor, so
    // both arms cover the same range and equal departures read as equally far.
    let extreme = (max - neutral).abs().max((min - neutral).abs()).max(f64::MIN_POSITIVE);
    // A floored sequential ramp runs from the floor to the data's largest value, so an
    // all-zero field is at the bottom of the ramp rather than in the flat case's middle.
    let (low, span) = match floor {
        Some(base) => (base, (max - base).max(f64::MIN_POSITIVE)),
        None => (min, max - min),
    };
    let flat = floor.is_none() && span <= 1e-12 * extreme;

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
                    if flat { 0.5 } else { ((value - low) / span).clamp(0.0, 1.0) }
                }
                Colormap::Diverging => (value - neutral) / extreme,
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

    /// Screen position back to a world point.
    ///
    /// The exact inverse of [`WorldView::project`], including the y flip. A playground
    /// needs it for every click, and having the two directions in one place is what
    /// stops them drifting apart — a picked body that is not the one under the cursor
    /// is a bug nobody can see the cause of.
    pub fn unproject(&self, screen: Pos2) -> [f64; 2] {
        let fx = f64::from((screen.x - self.rect.left()) / self.rect.width());
        let fy = f64::from((self.rect.bottom() - screen.y) / self.rect.height());
        [self.origin[0] + fx * self.extent[0], self.origin[1] + fy * self.extent[1]]
    }

    /// The rectangle this view draws into.
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// The world region this view shows, as `(origin, extent)`.
    pub fn region(&self) -> ([f64; 2], [f64; 2]) {
        (self.origin, self.extent)
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
        // Defensive: the channel promises these are the same length, and a caller
        // that got it wrong should draw a body oddly rather than take the process down.
        let is_wall = is_static.get(index).copied().unwrap_or(false);
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
/// [`format_value`], with anything indistinguishable from zero at `scale` shown as `0`.
///
/// Use this wherever a *live* reading is printed. `format_value` alone is honest about the
/// bits and dishonest about the physics: a resting body's net momentum wanders across
/// `±1e-17`, and printing `-6.939e-18` then `+1.041e-17` a few steps later shows a
/// motionless object as though every digit of it were unstable. The threshold matches
/// [`Series::is_negligible_against`](crate::history::Series::is_negligible_against).
///
/// A scale of zero or one that is not finite falls through to the plain formatter, because
/// there is then nothing to judge smallness against and inventing one would hide real
/// values.
pub fn format_value_against(value: f64, scale: f64) -> String {
    if value != 0.0 && scale.is_finite() && scale > 0.0 && value.abs() <= 1e-9 * scale {
        return "0".to_string();
    }
    format_value(value)
}

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
    /// A plate at 300 K on a zero-anchored diverging map is one flat colour, because
    /// every value is on the same side of zero and the variation is a rounding error
    /// against 300. Anchoring on ambient is what makes it readable.
    #[test]
    fn a_diverging_map_can_be_anchored_somewhere_other_than_zero() {
        let grid = Grid2d::new(3, 1, [3.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        field.set(0, 0, 280.0);
        field.set(1, 0, 300.0);
        field.set(2, 0, 320.0);
        let palette = Palette::for_mode(Mode::Dark);

        let anchored = field_to_image_about(&field, Colormap::Diverging, Mode::Dark, &palette, 300.0);
        let pixels = anchored.image.as_raw();
        let at = |i: usize| [pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]];
        let (cold, middle, hot) = (at(0), at(1), at(2));

        assert_ne!(cold, hot, "either side of ambient must be distinguishable");
        assert_ne!(cold, middle);
        assert_ne!(hot, middle);
        // The anchor lands on the neutral midpoint, which is the low-chroma grey.
        let chroma = |c: [u8; 3]| {
            let m = f64::from(c.iter().copied().max().unwrap());
            let n = f64::from(c.iter().copied().min().unwrap());
            m - n
        };
        assert!(chroma(middle) < chroma(cold), "{middle:?} against {cold:?}");
        assert!(chroma(middle) < chroma(hot), "{middle:?} against {hot:?}");

        // And that is the whole point: on the default anchor of zero, 280 K and 320 K sit
        // at 0.875 and 1.0 of the way up the *same arm* of the ramp, so the plate reads
        // as one colour with a hint of shading; anchored on 300 they land on opposite
        // ends with the grey midpoint between them.
        //
        // Measured as the largest distance between any two pixels, not as a spread in
        // total brightness. A diverging map is built to hold lightness roughly equal at
        // both ends and vary the hue, so summing the channels is very nearly blind to
        // the one thing this test is about. The claim is comparative rather than a
        // threshold, which would only record what today's ramp happens to do.
        let spread = |rendered: &FieldImage| {
            let pixels: Vec<[f64; 3]> = rendered
                .image
                .as_raw()
                .chunks(4)
                .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
                .collect();
            let mut worst: f64 = 0.0;
            for (index, a) in pixels.iter().enumerate() {
                for b in &pixels[index + 1..] {
                    let distance = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
                    worst = worst.max(distance);
                }
            }
            worst
        };
        let zeroed = field_to_image(&field, Colormap::Diverging, Mode::Dark, &palette);
        let (flat, spanned) = (spread(&zeroed), spread(&anchored));
        assert!(
            spanned > flat * 3.0,
            "anchoring on ambient separated the plate by {spanned:.1} against {flat:.1} on \
             zero, which is not enough of a difference to be worth the parameter"
        );
    }

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
    /// A picked body that is not the one under the cursor is a bug with no visible
    /// cause, so the two directions of the mapping are checked against each other.
    #[test]
    fn projecting_and_unprojecting_are_inverses() {
        let view = WorldView::new(
            Rect::from_min_size(Pos2::new(10.0, 20.0), egui::vec2(400.0, 200.0)),
            [-2.0, 5.0],
            [8.0, 4.0],
        );
        for world in [[-2.0, 5.0], [6.0, 9.0], [0.0, 7.0], [-1.5, 6.25]] {
            let screen = view.project(world).expect("finite");
            let back = view.unproject(screen);
            assert!((back[0] - world[0]).abs() < 1e-9, "{back:?} vs {world:?}");
            assert!((back[1] - world[1]).abs() < 1e-9, "{back:?} vs {world:?}");
        }

        // And y really is flipped: the world origin is at the *bottom* left.
        let bottom_left = view.project([-2.0, 5.0]).unwrap();
        assert!((bottom_left.y - view.rect().bottom()).abs() < 1e-3, "{bottom_left:?}");
        let top_right = view.project([6.0, 9.0]).unwrap();
        assert!((top_right.y - view.rect().top()).abs() < 1e-3, "{top_right:?}");
    }

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
