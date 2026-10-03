//! Curves: results that are a function rather than a number.
//!
//! An [`Observation`](crate::Observation) is one value at one time, and the timeline
//! of a run is a list of them. Some results do not fit that shape. A radial
//! distribution function is `g(r)` over a range of `r`, accumulated over the whole run;
//! a spectrum is a value per frequency. Spec §12.4 lists the RDF among the analyses
//! the molecular module owes, and §18.1 asks the run artifact to carry *"selected state
//! snapshots"* beside the observations — a curve is the smallest such snapshot that is
//! still a scientific result.
//!
//! A curve is published once, at the end of a run (or whenever a viewer asks), by
//! [`Domain::curves`](crate::Domain::curves). It states its axes and their units, so a
//! stored artifact says what its numbers were without the model file beside it.

/// A sampled function `y(x)` a domain publishes as a result.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    /// Qualified name, `domain.quantity`, matching the observation naming scheme.
    pub name: String,
    /// What the horizontal axis is, e.g. `"r"`.
    pub x_label: &'static str,
    /// Its SI unit.
    pub x_unit: &'static str,
    /// What the vertical axis is, e.g. `"g(r)"`.
    pub y_label: &'static str,
    /// Its SI unit, `"1"` when dimensionless.
    pub y_unit: &'static str,
    /// Abscissae, increasing.
    pub x: Vec<f64>,
    /// Ordinates, one per abscissa.
    pub y: Vec<f64>,
    /// How the curve was obtained — sample counts, normalization — so a reader can
    /// judge its noise without rerunning it.
    pub notes: Vec<String>,
}

impl Curve {
    /// A curve with the given axes.
    ///
    /// # Panics
    ///
    /// If `x` and `y` differ in length.
    pub fn new(
        name: impl Into<String>,
        (x_label, x_unit): (&'static str, &'static str),
        (y_label, y_unit): (&'static str, &'static str),
        x: Vec<f64>,
        y: Vec<f64>,
    ) -> Self {
        assert_eq!(x.len(), y.len(), "a curve needs one ordinate per abscissa");
        Self { name: name.into(), x_label, x_unit, y_label, y_unit, x, y, notes: Vec::new() }
    }

    /// Attach a note.
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// The point with the largest finite ordinate.
    pub fn peak(&self) -> Option<(f64, f64)> {
        self.x
            .iter()
            .zip(&self.y)
            .filter(|(_, y)| y.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(x, y)| (*x, *y))
    }

    /// Whether any ordinate is NaN or infinite.
    pub fn has_non_finite(&self) -> bool {
        self.y.iter().any(|y| !y.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_peak_ignores_non_finite_points() {
        let curve = Curve::new("d.g", ("r", "m"), ("g(r)", "1"), vec![1.0, 2.0, 3.0], vec![0.5, f64::NAN, 2.0]);
        assert_eq!(curve.peak(), Some((3.0, 2.0)));
        assert!(curve.has_non_finite());
    }

    #[test]
    #[should_panic(expected = "one ordinate per abscissa")]
    fn mismatched_axes_are_refused() {
        let _ = Curve::new("d.g", ("r", "m"), ("g", "1"), vec![1.0], vec![]);
    }
}
