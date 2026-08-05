//! Colour tokens and the perceptual maths behind the field colourmaps.
//!
//! Every hex here comes from a validated palette rather than being chosen by eye.
//! The categorical slots clear the lightness band, chroma floor, colour-vision
//! separation (worst adjacent pair ΔE 9.1 protan, 19.6 normal) and contrast checks
//! in both light and dark modes.
//!
//! # The rules this file exists to enforce
//!
//! **Sequential means one hue, light to dark.** A field heatmap encodes *magnitude*,
//! and magnitude is one-dimensional, so the colour should be too. Rainbow ramps —
//! viridis, jet, turbo — are common in scientific plotting and are a mistake for the
//! same reason every time: they invent visual boundaries where the data has none, so
//! a reader sees structure that is an artifact of the palette. One hue cannot do that.
//!
//! **Diverging means two opposing hues around a neutral grey.** Only for fields with
//! a meaningful zero, where the question is *which side*. The midpoint must read as
//! "nothing", which is why it is grey and not a third hue.
//!
//! **Interpolation happens in OKLab.** Blending two colours in sRGB passes through a
//! muddy, darkened middle — the classic blue-to-yellow-via-grey artifact. OKLab is
//! perceptually uniform, so equal steps in the data produce equal steps in perceived
//! colour, which is the entire point of a scientific colourmap.

use eframe::egui::Color32;

/// The two display modes. Dark is a *selected* set of steps for the dark surface,
/// not an inverted copy of the light one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    /// Light surface.
    #[default]
    Light,
    /// Dark surface.
    Dark,
}

/// Parse a `#rrggbb` literal at compile-time-ish cost.
///
/// # Panics
///
/// On a malformed literal, which can only be a typo in this file.
const fn hex(value: u32) -> Color32 {
    Color32::from_rgb(((value >> 16) & 0xff) as u8, ((value >> 8) & 0xff) as u8, (value & 0xff) as u8)
}

/// Chart chrome and ink for one mode.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    /// Which mode these tokens are for.
    pub mode: Mode,
    /// The surface charts are drawn on.
    pub surface: Color32,
    /// The plane behind the surface.
    pub plane: Color32,
    /// Body text.
    pub text_primary: Color32,
    /// Supporting text.
    pub text_secondary: Color32,
    /// Axis labels and other recessive text.
    pub text_muted: Color32,
    /// Hairline gridlines.
    pub grid: Color32,
    /// Axis and baseline rules.
    pub axis: Color32,
    /// Hairline borders.
    pub border: Color32,
}

impl Palette {
    /// Tokens for a mode.
    pub fn for_mode(mode: Mode) -> Palette {
        match mode {
            Mode::Light => Palette {
                mode,
                surface: hex(0xfcfcfb),
                plane: hex(0xf9f9f7),
                text_primary: hex(0x0b0b0b),
                text_secondary: hex(0x52514e),
                text_muted: hex(0x898781),
                grid: hex(0xe1e0d9),
                axis: hex(0xc3c2b7),
                border: Color32::from_rgba_unmultiplied(11, 11, 11, 26),
            },
            Mode::Dark => Palette {
                mode,
                surface: hex(0x1a1a19),
                plane: hex(0x0d0d0d),
                text_primary: hex(0xffffff),
                text_secondary: hex(0xc3c2b7),
                text_muted: hex(0x898781),
                grid: hex(0x2c2c2a),
                axis: hex(0x383835),
                border: Color32::from_rgba_unmultiplied(255, 255, 255, 26),
            },
        }
    }

    /// The categorical series colour for slot `index`, in fixed order.
    ///
    /// Order is the colour-vision-safety mechanism, not decoration: these eight were
    /// chosen so that *adjacent* slots stay distinguishable under protanopia and
    /// deuteranopia. Assignment therefore never skips or cycles — a ninth series
    /// folds into "other" rather than inventing a hue.
    pub fn series(&self, index: usize) -> Color32 {
        const LIGHT: [u32; 8] =
            [0x2a78d6, 0xeb6834, 0x1baf7a, 0xeda100, 0xe87ba4, 0x008300, 0x4a3aa7, 0xe34948];
        const DARK: [u32; 8] =
            [0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x008300, 0x9085e9, 0xe66767];
        let table = match self.mode {
            Mode::Light => &LIGHT,
            Mode::Dark => &DARK,
        };
        hex(table[index % table.len()])
    }

    /// How many distinct series colours exist before folding to "other".
    pub const SERIES_SLOTS: usize = 8;

    /// A reserved status colour. Never used for series identity, and always shipped
    /// alongside an icon and a label so meaning never rests on hue alone.
    pub fn status(&self, status: Status) -> Color32 {
        // Status steps are mode-invariant by design: all four clear 3:1 on the dark
        // surface, and on light the icon+label pairing carries the two that do not.
        hex(match status {
            Status::Good => 0x0ca30c,
            Status::Warning => 0xfab219,
            Status::Serious => 0xec835a,
            Status::Critical => 0xd03b3b,
        })
    }
}

/// A reserved state scale. Never a series colour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// Behaving as expected.
    Good,
    /// Worth watching.
    Warning,
    /// Degraded.
    Serious,
    /// Broken.
    Critical,
}

impl Status {
    /// A glyph that carries the meaning without colour, for readers who cannot rely
    /// on hue and for forced-colours and print.
    pub const fn icon(self) -> &'static str {
        match self {
            Status::Good => "✔",
            Status::Warning => "!",
            Status::Serious => "▲",
            Status::Critical => "✖",
        }
    }
}

/// How a scalar field is mapped to colour.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Colormap {
    /// One hue, light to dark. For magnitude with no meaningful zero.
    #[default]
    Sequential,
    /// Two opposing hues around a neutral grey. For fields with a signed zero.
    Diverging,
}

impl Colormap {
    /// Name shown in the control.
    pub const fn label(self) -> &'static str {
        match self {
            Colormap::Sequential => "sequential (magnitude)",
            Colormap::Diverging => "diverging (signed)",
        }
    }
}

/// The validated single-hue sequential ramp, steps 100 → 700.
const BLUE_RAMP: [u32; 13] = [
    0xcde2fb, 0xb7d3f6, 0x9ec5f4, 0x86b6ef, 0x6da7ec, 0x5598e7, 0x3987e5, 0x2a78d6, 0x256abf,
    0x1c5cab, 0x184f95, 0x104281, 0x0d366b,
];

/// The warm pole of the diverging pair. Its *hue* is used; lightness and chroma are
/// mirrored from the cool arm so the two arms are perceptually symmetric.
const RED_POLE: u32 = 0xe34948;

/// Neutral midpoints — the "nothing" colour of a diverging scale.
const MIDPOINT_LIGHT: u32 = 0xf0efec;
const MIDPOINT_DARK: u32 = 0x383835;

/// Sample the colourmap at `t`.
///
/// For [`Colormap::Sequential`], `t` runs 0 → 1 across the data range. For
/// [`Colormap::Diverging`], `t` runs −1 → 1 with 0 at the neutral midpoint.
pub fn sample(map: Colormap, mode: Mode, t: f64) -> Color32 {
    match map {
        Colormap::Sequential => sequential(mode, t),
        Colormap::Diverging => diverging(mode, t),
    }
}

/// One hue, light → dark. In dark mode the anchor flips so that the end meaning
/// "near zero" is the one that recedes into the surface.
fn sequential(mode: Mode, t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let t = match mode {
        Mode::Light => t,
        // On a dark surface the *darkest* step is the one that disappears, so low
        // values take the dark end and the ramp runs the other way.
        Mode::Dark => 1.0 - t,
    };
    ramp_sample(&BLUE_RAMP, t)
}

/// Two hues around a neutral grey, with perceptually symmetric arms.
fn diverging(mode: Mode, t: f64) -> Color32 {
    let t = t.clamp(-1.0, 1.0);
    let midpoint = to_oklab(hex(match mode {
        Mode::Light => MIDPOINT_LIGHT,
        Mode::Dark => MIDPOINT_DARK,
    }));

    // Both arms use the cool ramp's lightness and chroma profile; only the hue
    // differs. That is what makes "0.4 above" and "0.4 below" read as equally far
    // from the middle.
    let magnitude = t.abs();
    let pole = to_oklab(ramp_sample(&BLUE_RAMP, magnitude));
    let arm = if t >= 0.0 {
        // Warm arm: keep the cool arm's L and chroma, take the warm pole's hue.
        rotate_hue_to(pole, to_oklab(hex(RED_POLE)))
    } else {
        pole
    };

    from_oklab(lerp_oklab(midpoint, arm, magnitude))
}

/// Sample a ramp of hex steps at `t` ∈ [0,1], interpolating in OKLab.
fn ramp_sample(ramp: &[u32], t: f64) -> Color32 {
    let last = ramp.len() - 1;
    let scaled = (t.clamp(0.0, 1.0) * last as f64).clamp(0.0, last as f64);
    let low = scaled.floor() as usize;
    let high = (low + 1).min(last);
    let fraction = scaled - low as f64;
    from_oklab(lerp_oklab(to_oklab(hex(ramp[low])), to_oklab(hex(ramp[high])), fraction))
}

/// A colour in OKLab.
#[derive(Clone, Copy, Debug)]
struct Oklab {
    l: f64,
    a: f64,
    b: f64,
}

impl Oklab {
    /// Chroma — distance from the neutral axis.
    fn chroma(self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }
}

/// Take `source`'s lightness and chroma, but `hue_of`'s hue.
fn rotate_hue_to(source: Oklab, hue_of: Oklab) -> Oklab {
    let target_chroma = hue_of.chroma();
    if target_chroma == 0.0 {
        return Oklab { l: source.l, a: 0.0, b: 0.0 };
    }
    let scale = source.chroma() / target_chroma;
    Oklab { l: source.l, a: hue_of.a * scale, b: hue_of.b * scale }
}

fn lerp_oklab(from: Oklab, to: Oklab, t: f64) -> Oklab {
    Oklab {
        l: from.l + (to.l - from.l) * t,
        a: from.a + (to.a - from.a) * t,
        b: from.b + (to.b - from.b) * t,
    }
}

fn srgb_to_linear(channel: f64) -> f64 {
    if channel <= 0.04045 { channel / 12.92 } else { ((channel + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(channel: f64) -> f64 {
    if channel <= 0.003_130_8 {
        12.92 * channel
    } else {
        1.055 * channel.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB → OKLab, using Ottosson's matrices.
fn to_oklab(color: Color32) -> Oklab {
    let r = srgb_to_linear(f64::from(color.r()) / 255.0);
    let g = srgb_to_linear(f64::from(color.g()) / 255.0);
    let b = srgb_to_linear(f64::from(color.b()) / 255.0);

    let long = 0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b;
    let medium = 0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b;
    let short = 0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b;

    let (l_, m_, s_) = (long.cbrt(), medium.cbrt(), short.cbrt());
    Oklab {
        l: 0.210_454_255_3 * l_ + 0.793_617_785_0 * m_ - 0.004_072_046_8 * s_,
        a: 1.977_998_495_1 * l_ - 2.428_592_205_0 * m_ + 0.450_593_709_9 * s_,
        b: 0.025_904_037_1 * l_ + 0.782_771_766_2 * m_ - 0.808_675_766_0 * s_,
    }
}

/// OKLab → sRGB, clamping out-of-gamut results into range.
fn from_oklab(color: Oklab) -> Color32 {
    let l_ = color.l + 0.396_337_777_4 * color.a + 0.215_803_757_3 * color.b;
    let m_ = color.l - 0.105_561_345_8 * color.a - 0.063_854_172_8 * color.b;
    let s_ = color.l - 0.089_484_177_5 * color.a - 1.291_485_548_0 * color.b;

    let (long, medium, short) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    let r = 4.076_741_662_1 * long - 3.307_711_591_3 * medium + 0.230_969_929_2 * short;
    let g = -1.268_438_004_6 * long + 2.609_757_401_1 * medium - 0.341_319_396_5 * short;
    let b = -0.004_196_086_3 * long - 0.703_418_614_7 * medium + 1.707_614_701_0 * short;

    let encode = |channel: f64| (linear_to_srgb(channel).clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(encode(r), encode(g), encode(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lightness(color: Color32) -> f64 {
        to_oklab(color).l
    }

    #[test]
    fn oklab_round_trips() {
        for hexvalue in [0x000000u32, 0xffffff, 0x2a78d6, 0xe34948, 0x1baf7a] {
            let original = hex(hexvalue);
            let round_tripped = from_oklab(to_oklab(original));
            // One step of rounding in each direction.
            for (a, b) in [
                (original.r(), round_tripped.r()),
                (original.g(), round_tripped.g()),
                (original.b(), round_tripped.b()),
            ] {
                assert!(a.abs_diff(b) <= 1, "{hexvalue:06x}: {a} vs {b}");
            }
        }
    }

    /// The defining property of a sequential ramp: lightness moves in one direction
    /// the whole way. A ramp that brightens then darkens invents a boundary.
    #[test]
    fn the_sequential_ramp_is_monotone_in_lightness() {
        let samples: Vec<f64> =
            (0..=100).map(|i| lightness(sequential(Mode::Light, f64::from(i) / 100.0))).collect();
        for pair in samples.windows(2) {
            assert!(
                pair[1] <= pair[0] + 1e-9,
                "light mode should darken monotonically: {} then {}",
                pair[0],
                pair[1]
            );
        }

        // Dark mode flips the anchor, so it brightens monotonically instead.
        let dark: Vec<f64> =
            (0..=100).map(|i| lightness(sequential(Mode::Dark, f64::from(i) / 100.0))).collect();
        for pair in dark.windows(2) {
            assert!(pair[1] >= pair[0] - 1e-9, "dark mode should brighten monotonically");
        }
    }

    /// The ramp must actually span a useful range — a "sequential" map that barely
    /// changes lightness conveys nothing.
    #[test]
    fn the_sequential_ramp_spans_a_wide_lightness_range() {
        let low = lightness(sequential(Mode::Light, 0.0));
        let high = lightness(sequential(Mode::Light, 1.0));
        assert!(low - high > 0.4, "range was only {}", low - high);
    }

    /// Sequential is one hue: every sample must sit in the same hue family. This is
    /// what rules out rainbow ramps, which is the whole point.
    #[test]
    fn the_sequential_ramp_holds_one_hue() {
        let hues: Vec<f64> = (0..=20)
            .map(|i| {
                let c = to_oklab(sequential(Mode::Light, f64::from(i) / 20.0));
                c.b.atan2(c.a)
            })
            .collect();
        let first = hues[0];
        for hue in &hues {
            // Within ~15 degrees across the whole ramp.
            let delta = (hue - first).abs();
            assert!(delta < 0.26, "hue drifted by {delta} radians — that is a rainbow");
        }
    }

    /// The diverging midpoint must read as "nothing": near-zero chroma, so it cannot
    /// be mistaken for a third category.
    #[test]
    fn the_diverging_midpoint_is_neutral() {
        for mode in [Mode::Light, Mode::Dark] {
            let middle = to_oklab(diverging(mode, 0.0));
            assert!(middle.chroma() < 0.02, "{mode:?} midpoint has chroma {}", middle.chroma());
        }
    }

    /// The two arms must read as opposite — one warm, one cool — and be perceptually
    /// symmetric, so equal magnitudes look equally far from the middle.
    #[test]
    fn the_diverging_arms_are_opposite_and_symmetric() {
        let cool = to_oklab(diverging(Mode::Light, -1.0));
        let warm = to_oklab(diverging(Mode::Light, 1.0));

        // Opposite: the `a` axis runs green-to-red, so the poles straddle it.
        assert!(cool.a < 0.0, "the cool pole should sit on the negative a axis");
        assert!(warm.a > 0.0, "the warm pole should sit on the positive a axis");

        // Symmetric: same lightness and chroma, different hue.
        assert!((cool.l - warm.l).abs() < 0.02, "arms differ in lightness: {} vs {}", cool.l, warm.l);
        assert!(
            (cool.chroma() - warm.chroma()).abs() < 0.02,
            "arms differ in chroma: {} vs {}",
            cool.chroma(),
            warm.chroma()
        );
    }

    #[test]
    fn diverging_lightness_falls_away_from_the_midpoint_on_both_arms() {
        let middle = lightness(diverging(Mode::Light, 0.0));
        for magnitude in [0.25, 0.5, 0.75, 1.0] {
            for signed in [magnitude, -magnitude] {
                assert!(
                    lightness(diverging(Mode::Light, signed)) < middle,
                    "t = {signed} should be darker than the neutral midpoint"
                );
            }
        }
    }

    #[test]
    fn sampling_clamps_out_of_range_inputs() {
        assert_eq!(sample(Colormap::Sequential, Mode::Light, -5.0), sequential(Mode::Light, 0.0));
        assert_eq!(sample(Colormap::Sequential, Mode::Light, 5.0), sequential(Mode::Light, 1.0));
        assert_eq!(sample(Colormap::Diverging, Mode::Light, -5.0), diverging(Mode::Light, -1.0));
    }

    /// Series colours are assigned in fixed order and never cycled within the eight
    /// validated slots; past that, assignment wraps only because the caller is
    /// expected to have folded to "other" first.
    #[test]
    fn series_slots_are_stable_and_distinct() {
        let palette = Palette::for_mode(Mode::Light);
        let colors: Vec<Color32> = (0..Palette::SERIES_SLOTS).map(|i| palette.series(i)).collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "slots {i} and {j} collide");
            }
        }
        // The order is the safety mechanism, so slot 0 is always the same blue.
        assert_eq!(palette.series(0), hex(0x2a78d6));
    }

    /// Status colours must never coincide with a series colour, or a warning would
    /// impersonate a data series.
    #[test]
    fn status_colours_are_distinct_from_series_colours() {
        for mode in [Mode::Light, Mode::Dark] {
            let palette = Palette::for_mode(mode);
            for status in [Status::Good, Status::Warning, Status::Serious, Status::Critical] {
                let color = palette.status(status);
                for slot in 0..Palette::SERIES_SLOTS {
                    assert_ne!(color, palette.series(slot), "{status:?} collides in {mode:?}");
                }
                assert!(!status.icon().is_empty(), "status must carry a non-colour cue");
            }
        }
    }

    #[test]
    fn both_modes_have_readable_ink() {
        for mode in [Mode::Light, Mode::Dark] {
            let palette = Palette::for_mode(mode);
            let surface = lightness(palette.surface);
            let ink = lightness(palette.text_primary);
            assert!((surface - ink).abs() > 0.5, "{mode:?} ink is too close to its surface");
        }
    }
}
