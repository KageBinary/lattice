//! Render channels: the read-only views a viewer may draw.
//!
//! Spec §7.3 lists "Observe — expose metrics, residuals, invariants, and **render
//! channels**" as part of the domain contract, and P7 insists the renderer is
//! instrumentation rather than decoration:
//!
//! > It must display fields, fluxes, constraints, forces, residuals, conservation
//! > drift, timestep decisions, and uncertainty.
//!
//! A channel is a *borrowed* view of state a domain already holds. Nothing is copied
//! or converted, so exposing one costs nothing and a domain has no reason to withhold
//! it. That matters: a solver whose internals are invisible cannot be debugged when
//! it misbehaves, and P7's whole argument is that this is when visualization earns
//! its place.

use crate::grid::{Grid2d, ScalarField};

/// A drawable view of some domain state.
#[derive(Clone, Copy, Debug)]
pub enum RenderChannel<'a> {
    /// A scalar field on a grid.
    Scalar {
        /// Channel name, e.g. `"temperature"`.
        name: &'a str,
        /// The values.
        field: &'a ScalarField,
        /// The geometry they live on.
        grid: Grid2d,
        /// The SI unit of the values, for the scale label.
        ///
        /// Borrowed rather than `'static` so a domain can report the unit the
        /// *compiler* inferred for it. A heatmap labelled "field units" tells a
        /// reader nothing; one labelled "K" tells them whether 400 is alarming.
        unit: &'a str,
    },
    /// Particle positions.
    Particles {
        /// Channel name.
        name: &'a str,
        /// x positions, m.
        x: &'a [f64],
        /// y positions, m.
        y: &'a [f64],
        /// Lower-left corner of the view, m.
        origin: [f64; 2],
        /// View size, m.
        extent: [f64; 2],
    },
}

impl RenderChannel<'_> {
    /// The channel's name.
    pub fn name(&self) -> &str {
        match self {
            RenderChannel::Scalar { name, .. } | RenderChannel::Particles { name, .. } => name,
        }
    }

    /// A short description of what this channel holds.
    pub fn describe(&self) -> String {
        match self {
            RenderChannel::Scalar { field, unit, .. } => {
                format!("{}x{} scalar field in {unit}", field.nx(), field.ny())
            }
            RenderChannel::Particles { x, .. } => format!("{} particle positions", x.len()),
        }
    }

    /// True when the channel contains a non-finite value.
    ///
    /// The viewer marks these rather than letting them normalize the whole image
    /// away (NFR-007).
    pub fn has_non_finite(&self) -> bool {
        match self {
            RenderChannel::Scalar { field, .. } => field.first_non_finite().is_some(),
            RenderChannel::Particles { x, y, .. } => {
                x.iter().chain(y.iter()).any(|v| !v.is_finite())
            }
        }
    }
}

/// Compute a view region that contains every finite point, with a small margin.
///
/// Used when a particle domain has no declared region: the alternative is drawing
/// into a guessed box and silently cropping whatever falls outside it.
pub fn bounds_of(x: &[f64], y: &[f64]) -> ([f64; 2], [f64; 2]) {
    let finite = |values: &[f64]| {
        values.iter().copied().filter(|v| v.is_finite()).fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(lo, hi), v| (lo.min(v), hi.max(v)),
        )
    };
    let (min_x, max_x) = finite(x);
    let (min_y, max_y) = finite(y);

    if !min_x.is_finite() || !min_y.is_finite() {
        // Nothing finite to bound; return a unit box so the caller can still draw.
        return ([0.0, 0.0], [1.0, 1.0]);
    }

    // A margin of 5%, and a floor so a single point or a degenerate line still has
    // a drawable extent.
    let width = (max_x - min_x).max(1e-12);
    let height = (max_y - min_y).max(1e-12);
    let margin = [width * 0.05, height * 0.05];
    (
        [min_x - margin[0], min_y - margin[1]],
        [width + 2.0 * margin[0], height + 2.0 * margin[1]],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scalar_channel_describes_itself() {
        let grid = Grid2d::new(16, 8, [1.0, 1.0]);
        let field = ScalarField::new(&grid, 1);
        let channel = RenderChannel::Scalar { name: "temperature", field: &field, grid, unit: "K" };
        assert_eq!(channel.name(), "temperature");
        assert_eq!(channel.describe(), "16x8 scalar field in K");
        assert!(!channel.has_non_finite());
    }

    #[test]
    fn a_non_finite_field_is_flagged() {
        let grid = Grid2d::new(4, 4, [1.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        field.set(2, 2, f64::NAN);
        let channel = RenderChannel::Scalar { name: "t", field: &field, grid, unit: "K" };
        assert!(channel.has_non_finite());
    }

    #[test]
    fn a_particle_channel_describes_itself() {
        let (x, y) = (vec![0.0, 1.0], vec![0.0, 1.0]);
        let channel = RenderChannel::Particles {
            name: "gas",
            x: &x,
            y: &y,
            origin: [0.0, 0.0],
            extent: [1.0, 1.0],
        };
        assert_eq!(channel.describe(), "2 particle positions");
        assert!(!channel.has_non_finite());
    }

    #[test]
    fn bounds_contain_every_point_with_a_margin() {
        let (origin, extent) = bounds_of(&[0.0, 10.0], &[-5.0, 5.0]);
        assert!(origin[0] < 0.0 && origin[1] < -5.0);
        assert!(origin[0] + extent[0] > 10.0);
        assert!(origin[1] + extent[1] > 5.0);
    }

    /// A single point, or points on a line, must still produce a drawable box rather
    /// than a zero-width one that divides by zero downstream.
    #[test]
    fn degenerate_point_sets_still_get_an_extent() {
        let (_, extent) = bounds_of(&[3.0], &[3.0]);
        assert!(extent[0] > 0.0 && extent[1] > 0.0);

        let (_, line) = bounds_of(&[0.0, 1.0, 2.0], &[5.0, 5.0, 5.0]);
        assert!(line[1] > 0.0, "a horizontal line still needs height");
    }

    #[test]
    fn non_finite_positions_do_not_poison_the_bounds() {
        let (origin, extent) = bounds_of(&[0.0, f64::NAN, 2.0], &[0.0, 1.0, f64::INFINITY]);
        assert!(origin.iter().all(|v| v.is_finite()));
        assert!(extent.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn an_entirely_non_finite_set_falls_back_to_a_unit_box() {
        let (origin, extent) = bounds_of(&[f64::NAN], &[f64::NAN]);
        assert_eq!(origin, [0.0, 0.0]);
        assert_eq!(extent, [1.0, 1.0]);
    }
}
