//! The minimum-image convention.
//!
//! In a periodic box the separation between two particles is not `x_j − x_i`: the
//! nearest copy of `j` may be through a wall. Every pair law and every bonded law has
//! to measure separations the same way, so the rule lives here once rather than being
//! remembered separately by each of them.

/// How to measure a separation in a box whose axes may wrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimumImage {
    extent: [f64; 2],
    periodic: [bool; 2],
}

impl MinimumImage {
    /// Open space: a separation is a plain difference.
    pub const fn open() -> Self {
        Self { extent: [0.0, 0.0], periodic: [false, false] }
    }

    /// A box of the given size, wrapping on the axes marked periodic.
    ///
    /// # Panics
    ///
    /// If a periodic axis has a non-positive or non-finite extent — there is no
    /// "nearest image" in a box of zero width.
    pub fn new(extent: [f64; 2], periodic: [bool; 2]) -> Self {
        for axis in 0..2 {
            assert!(
                !periodic[axis] || (extent[axis] > 0.0 && extent[axis].is_finite()),
                "a periodic axis needs a positive finite extent, got {extent:?}"
            );
        }
        Self { extent, periodic }
    }

    /// Box size, metres. Meaningful only on periodic axes.
    pub fn extent(&self) -> [f64; 2] {
        self.extent
    }

    /// Which axes wrap.
    pub fn periodic(&self) -> [bool; 2] {
        self.periodic
    }

    /// True when at least one axis wraps.
    pub fn is_periodic(&self) -> bool {
        self.periodic[0] || self.periodic[1]
    }

    /// The shortest separation equivalent to `(dx, dy)`.
    ///
    /// `round` rather than `floor`, so the result lies in `[−L/2, L/2]` on each
    /// periodic axis.
    #[inline]
    pub fn separation(&self, mut dx: f64, mut dy: f64) -> (f64, f64) {
        if self.periodic[0] {
            dx -= self.extent[0] * (dx / self.extent[0]).round();
        }
        if self.periodic[1] {
            dy -= self.extent[1] * (dy / self.extent[1]).round();
        }
        (dx, dy)
    }

    /// The largest separation the convention can represent unambiguously: half the
    /// box on the shortest periodic axis, or infinity in open space.
    ///
    /// An interaction reaching further than this sees a particle *and* its own image,
    /// which is why the domain contract requires the box to be at least twice the
    /// cutoff.
    pub fn max_representable(&self) -> f64 {
        let mut limit = f64::INFINITY;
        for axis in 0..2 {
            if self.periodic[axis] {
                limit = limit.min(0.5 * self.extent[axis]);
            }
        }
        limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_space_is_a_plain_difference() {
        let image = MinimumImage::open();
        assert_eq!(image.separation(7.5, -3.0), (7.5, -3.0));
        assert!(!image.is_periodic());
        assert!(image.max_representable().is_infinite());
    }

    #[test]
    fn a_periodic_axis_takes_the_short_way_round() {
        let image = MinimumImage::new([10.0, 10.0], [true, false]);
        let (dx, dy) = image.separation(9.8, 9.8);
        assert!((dx + 0.2).abs() < 1e-12, "x wraps: {dx}");
        assert!((dy - 9.8).abs() < 1e-12, "y does not: {dy}");
        assert_eq!(image.max_representable(), 5.0);
    }

    #[test]
    fn separations_land_in_the_half_open_box() {
        let image = MinimumImage::new([4.0, 6.0], [true, true]);
        for raw in [-13.0, -2.1, -2.0, 0.0, 1.9, 2.0, 2.1, 17.0] {
            let (dx, dy) = image.separation(raw, raw);
            assert!(dx.abs() <= 2.0 + 1e-12, "{raw} -> {dx}");
            assert!(dy.abs() <= 3.0 + 1e-12, "{raw} -> {dy}");
        }
    }

    #[test]
    #[should_panic(expected = "positive finite extent")]
    fn a_periodic_axis_without_an_extent_is_rejected() {
        MinimumImage::new([0.0, 1.0], [true, false]);
    }
}
