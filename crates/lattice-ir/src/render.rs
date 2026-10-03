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
    /// Rigid bodies: poses plus the closed outlines they wear.
    ///
    /// # Why an outline table rather than a shape
    ///
    /// This crate holds no physics, so it cannot name a circle or a convex polygon —
    /// those types live in the domain crate, which depends on this one and not the
    /// reverse. Handing over a flat table of body-local vertices keeps that direction
    /// intact and costs nothing: the table is built once when a shape is registered
    /// and borrowed unchanged thereafter, while the poses come straight out of the
    /// store's arrays. A circle arrives already tessellated, which is what a renderer
    /// would have done with it anyway.
    Bodies {
        /// Channel name.
        name: &'a str,
        /// Centre-of-mass x, m.
        x: &'a [f64],
        /// Centre-of-mass y, m.
        y: &'a [f64],
        /// Cosine of each body's orientation.
        cos: &'a [f64],
        /// Sine of each body's orientation.
        sin: &'a [f64],
        /// Which outline each body wears, as an index into `starts`.
        outline: &'a [u32],
        /// Every outline's vertices, body-local and concatenated.
        vertices: &'a [[f64; 2]],
        /// Where each outline begins in `vertices`, with a final end sentinel.
        ///
        /// Outline `i` occupies `vertices[starts[i] .. starts[i + 1]]`, so `starts` has
        /// one more entry than there are outlines.
        starts: &'a [u32],
        /// True for each body that no impulse can move.
        ///
        /// Drawn differently, because "why is the ground falling" and "why is the crate
        /// not falling" are the two questions a rigid scene provokes, and both are
        /// answered by seeing which bodies are static.
        is_static: &'a [bool],
        /// Lower-left corner of the view, m.
        origin: [f64; 2],
        /// View size, m.
        extent: [f64; 2],
    },
    /// Bonds between particles: pairs of indices into a particle channel's arrays.
    ///
    /// Published alongside [`RenderChannel::Particles`] by a domain with bonded
    /// topology, and drawn over it. A pair whose separation exceeds half the box on a
    /// periodic axis is bonded through the wall; a viewer draws nothing for it rather
    /// than a line across the whole picture.
    Bonds {
        /// Channel name.
        name: &'a str,
        /// x positions, m.
        x: &'a [f64],
        /// y positions, m.
        y: &'a [f64],
        /// Bonded slot pairs.
        pairs: &'a [[u32; 2]],
        /// Which axes of the region wrap.
        periodic: [bool; 2],
        /// Lower-left corner of the view, m.
        origin: [f64; 2],
        /// View size, m.
        extent: [f64; 2],
    },
    /// Contact points and their normals.
    ///
    /// §17 asks for constraints and forces to be visible. A contact solver whose
    /// contacts cannot be seen is the hardest kind of code to debug: every symptom —
    /// jitter, sinking, sticking — looks the same from outside, and looks completely
    /// different once the normals are on screen.
    Contacts {
        /// Channel name.
        name: &'a str,
        /// Contact point x, m.
        x: &'a [f64],
        /// Contact point y, m.
        y: &'a [f64],
        /// Normal x component, unit.
        normal_x: &'a [f64],
        /// Normal y component, unit.
        normal_y: &'a [f64],
        /// Overlap along the normal, m. Zero for a resting contact.
        depth: &'a [f64],
    },
}

impl RenderChannel<'_> {
    /// The channel's name.
    pub fn name(&self) -> &str {
        match self {
            RenderChannel::Scalar { name, .. }
            | RenderChannel::Particles { name, .. }
            | RenderChannel::Bodies { name, .. }
            | RenderChannel::Bonds { name, .. }
            | RenderChannel::Contacts { name, .. } => name,
        }
    }

    /// A short description of what this channel holds.
    pub fn describe(&self) -> String {
        match self {
            RenderChannel::Scalar { field, unit, .. } => {
                format!("{}x{} scalar field in {unit}", field.nx(), field.ny())
            }
            RenderChannel::Particles { x, .. } => format!("{} particle positions", x.len()),
            RenderChannel::Bodies { x, is_static, .. } => {
                let statics = is_static.iter().filter(|s| **s).count();
                format!("{} rigid bodies ({statics} static)", x.len())
            }
            RenderChannel::Bonds { pairs, .. } => format!("{} bonds", pairs.len()),
            RenderChannel::Contacts { x, .. } => format!("{} contact points", x.len()),
        }
    }

    /// True when the channel contains a non-finite value.
    ///
    /// The viewer marks these rather than letting them normalize the whole image
    /// away (NFR-007).
    pub fn has_non_finite(&self) -> bool {
        match self {
            RenderChannel::Scalar { field, .. } => field.first_non_finite().is_some(),
            RenderChannel::Particles { x, y, .. } | RenderChannel::Bonds { x, y, .. } => {
                x.iter().chain(y.iter()).any(|v| !v.is_finite())
            }
            RenderChannel::Bodies { x, y, cos, sin, .. } => {
                x.iter().chain(*y).chain(*cos).chain(*sin).any(|v| !v.is_finite())
            }
            RenderChannel::Contacts { x, y, normal_x, normal_y, depth, .. } => x
                .iter()
                .chain(*y)
                .chain(*normal_x)
                .chain(*normal_y)
                .chain(*depth)
                .any(|v| !v.is_finite()),
        }
    }

    /// The two endpoints of one bond, in world coordinates, or `None` when the bond
    /// crosses a periodic seam (its minimum-image length exceeds half the box) or the
    /// channel is not [`RenderChannel::Bonds`].
    pub fn bond_endpoints(&self, index: usize) -> Option<[[f64; 2]; 2]> {
        let RenderChannel::Bonds { x, y, pairs, periodic, extent, .. } = self else {
            return None;
        };
        let &[a, b] = pairs.get(index)?;
        let (a, b) = (a as usize, b as usize);
        let from = [*x.get(a)?, *y.get(a)?];
        let to = [*x.get(b)?, *y.get(b)?];
        for axis in 0..2 {
            if periodic[axis] && (to[axis] - from[axis]).abs() > 0.5 * extent[axis] {
                return None;
            }
        }
        Some([from, to])
    }

    /// The vertices of one body's outline, in world coordinates.
    ///
    /// Returns nothing for a channel that is not [`RenderChannel::Bodies`], or for an
    /// index past the end — a viewer iterating bodies should not have to bounds-check
    /// a table it did not build.
    pub fn body_outline(&self, index: usize, out: &mut Vec<[f64; 2]>) {
        out.clear();
        let RenderChannel::Bodies { x, y, cos, sin, outline, vertices, starts, .. } = self else {
            return;
        };
        if index >= x.len() {
            return;
        }
        let Some(&slot) = outline.get(index) else { return };
        let (Some(&from), Some(&to)) =
            (starts.get(slot as usize), starts.get(slot as usize + 1))
        else {
            return;
        };
        let (c, s) = (cos[index], sin[index]);
        for &[vx, vy] in &vertices[from as usize..to as usize] {
            out.push([x[index] + c * vx - s * vy, y[index] + s * vx + c * vy]);
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
    fn a_bond_channel_hides_bonds_through_a_periodic_wall() {
        let (x, y) = (vec![0.5, 9.5, 1.0], vec![5.0, 5.0, 5.0]);
        let pairs = [[0u32, 1u32], [0, 2], [2, 9]];
        let channel = RenderChannel::Bonds {
            name: "chain",
            x: &x,
            y: &y,
            pairs: &pairs,
            periodic: [true, false],
            origin: [0.0, 0.0],
            extent: [10.0, 10.0],
        };
        assert_eq!(channel.describe(), "3 bonds");
        assert_eq!(channel.bond_endpoints(0), None, "bonded through the wall");
        assert_eq!(channel.bond_endpoints(1), Some([[0.5, 5.0], [1.0, 5.0]]));
        assert_eq!(channel.bond_endpoints(2), None, "an index past the arrays draws nothing");
        assert_eq!(channel.bond_endpoints(7), None);
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
