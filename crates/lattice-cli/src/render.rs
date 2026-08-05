//! A terminal field viewer.
//!
//! Spec M0 asks for a *"tiny viewer"*, and P7 insists that visualization is
//! instrumentation rather than decoration: *"it must display fields, fluxes,
//! constraints, forces, residuals, conservation drift, timestep decisions, and
//! uncertainty."* A wgpu renderer is M4 work. What a terminal can do today — and what
//! matters most for catching a wrong sign or an inverted boundary — is show the field
//! and label its range.
//!
//! The scale bar is the load-bearing part. An unlabelled heatmap normalizes away
//! exactly the information you need: a field that has quietly gone negative and one
//! that is behaving look identical once each is rescaled to its own extremes.

use lattice_ir::ScalarField;

/// Density ramp from empty to full.
const RAMP: &[u8] = b" .:-=+*#%@";

/// Render a field as an ASCII heatmap with a labelled scale.
///
/// The field is box-sampled down to `columns × rows`. Terminal cells are about twice
/// as tall as they are wide, so a square domain is rendered with roughly twice as
/// many columns as rows to keep it looking square.
pub fn heatmap(field: &ScalarField, columns: usize, rows: usize) -> String {
    let (min, max) = (field.min_interior(), field.max_interior());
    let span = max - min;

    let mut out = String::new();
    for row in 0..rows {
        // Row 0 is the top of the image, which is the *high* j end of the grid.
        let j0 = (rows - 1 - row) * field.ny() / rows;
        let j1 = ((rows - row) * field.ny() / rows).max(j0 + 1);
        out.push_str("  |");
        for column in 0..columns {
            let i0 = column * field.nx() / columns;
            let i1 = ((column + 1) * field.nx() / columns).max(i0 + 1);

            // Box average, so a downsampled view cannot miss a one-cell spike by
            // landing between samples.
            let mut sum = 0.0;
            let mut count = 0.0;
            for j in j0..j1.min(field.ny()) {
                for i in i0..i1.min(field.nx()) {
                    sum += field.get(i, j);
                    count += 1.0;
                }
            }
            let value = if count > 0.0 { sum / count } else { min };

            let symbol = if !value.is_finite() {
                b'!'
            } else if span <= 0.0 {
                RAMP[0]
            } else {
                let normalized = ((value - min) / span).clamp(0.0, 1.0);
                RAMP[((normalized * (RAMP.len() - 1) as f64).round() as usize).min(RAMP.len() - 1)]
            };
            out.push(symbol as char);
        }
        out.push_str("|\n");
    }

    // The span is printed explicitly because min and max can round to the same text
    // when the field has nearly equilibrated — and "nearly flat" versus "flat" is
    // exactly the distinction the picture alone destroys.
    out.push_str(&format!(
        "  scale: {} {} {}   span {span:.3e}   [{} = min, {} = max, ! = non-finite]\n",
        format_value(min),
        String::from_utf8_lossy(RAMP),
        format_value(max),
        RAMP[0] as char,
        RAMP[RAMP.len() - 1] as char,
    ));
    if !span.is_finite() || field.first_non_finite().is_some() {
        out.push_str("  WARNING: the field contains non-finite values\n");
    }
    out
}

/// Render particle positions as a scatter plot over a rectangular region.
pub fn scatter(
    xs: &[f64],
    ys: &[f64],
    origin: [f64; 2],
    extent: [f64; 2],
    columns: usize,
    rows: usize,
) -> String {
    let mut counts = vec![0u32; columns * rows];
    let mut outside = 0usize;

    for (&x, &y) in xs.iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() {
            outside += 1;
            continue;
        }
        let fx = (x - origin[0]) / extent[0];
        let fy = (y - origin[1]) / extent[1];
        if !(0.0..1.0).contains(&fx) || !(0.0..1.0).contains(&fy) {
            outside += 1;
            continue;
        }
        let column = ((fx * columns as f64) as usize).min(columns - 1);
        // Flip so that increasing y appears higher on screen.
        let row = rows - 1 - ((fy * rows as f64) as usize).min(rows - 1);
        counts[row * columns + column] += 1;
    }

    let busiest = counts.iter().copied().max().unwrap_or(0).max(1);
    let mut out = String::new();
    for row in 0..rows {
        out.push_str("  |");
        for column in 0..columns {
            let count = counts[row * columns + column];
            let symbol = if count == 0 {
                RAMP[0]
            } else {
                let normalized = count as f64 / busiest as f64;
                RAMP[((normalized * (RAMP.len() - 1) as f64).ceil() as usize).min(RAMP.len() - 1)]
            };
            out.push(symbol as char);
        }
        out.push_str("|\n");
    }
    out.push_str(&format!(
        "  {} particles, densest cell holds {busiest}",
        xs.len()
    ));
    if outside > 0 {
        out.push_str(&format!(", {outside} outside the view"));
    }
    out.push('\n');
    out
}

/// Render a time series as a sparkline with its range labelled.
pub fn sparkline(values: &[f64], width: usize) -> String {
    if values.is_empty() {
        return "(no samples)".to_string();
    }
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let (min, max) = finite.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    let span = max - min;

    // A conserved quantity varies only by round-off, and normalizing that to the full
    // ramp draws a wildly oscillating line from noise at the 16th digit. Rendering it
    // flat — and saying so — is the honest picture, since the range label alone would
    // print the same number twice and explain nothing.
    //
    // The shortcut is skipped when any sample is non-finite: a `NaN` must always reach
    // the plot as a `!` (NFR-007), and summarizing the trace as "constant" would be
    // the exact kind of hiding the spec forbids.
    let has_non_finite = values.len() != finite.len();
    let magnitude = max.abs().max(min.abs()).max(f64::MIN_POSITIVE);
    if !has_non_finite && span <= 1e-12 * magnitude {
        let flat = (RAMP[RAMP.len() / 2] as char).to_string().repeat(width.min(values.len()).max(1));
        return format!("{flat}   [constant at {min:.6e}, varying by {span:.1e}]");
    }

    // One column per sample when the trace is shorter than the plot, and an even
    // stride through the samples when it is longer. Striding by `width` in both cases
    // would silently compress a short trace into the first third of the plot.
    let columns = width.min(values.len()).max(1);
    let mut out = String::new();
    for column in 0..columns {
        let index = (column * values.len() / columns).min(values.len() - 1);
        let value = values[index];
        let symbol = if !value.is_finite() {
            b'!'
        } else if span <= 0.0 {
            RAMP[RAMP.len() / 2]
        } else {
            let normalized = ((value - min) / span).clamp(0.0, 1.0);
            RAMP[((normalized * (RAMP.len() - 1) as f64).round() as usize).min(RAMP.len() - 1)]
        };
        out.push(symbol as char);
    }

    // A sparkline always fills its full range, so a quantity varying by one part in
    // 40,000 draws the same dramatic shape as one that doubles. Beside a series that
    // genuinely swings, that reads as instability where there is none — so when the
    // variation is small relative to the value, say so.
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let relative = if mean == 0.0 { f64::INFINITY } else { span / mean.abs() };
    let note = if relative < 1e-3 {
        format!("  ({relative:.1e} relative, drawn at full scale)")
    } else {
        String::new()
    };

    format!("{out}   [{min:.6e} .. {max:.6e}]{note}")
}

fn format_value(v: f64) -> String {
    if !v.is_finite() {
        format!("{v}")
    } else if v == 0.0 || (1e-3..1e5).contains(&v.abs()) {
        format!("{v:.6}")
    } else {
        format!("{v:.3e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::Grid2d;

    fn ramp_field() -> (Grid2d, ScalarField) {
        let grid = Grid2d::new(16, 16, [1.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        field.init_from_position(&grid, |[x, _]| x);
        (grid, field)
    }

    #[test]
    fn heatmap_has_the_requested_shape() {
        let (_, field) = ramp_field();
        let text = heatmap(&field, 32, 8);
        let rows: Vec<&str> = text.lines().filter(|l| l.starts_with("  |")).collect();
        assert_eq!(rows.len(), 8);
        for row in rows {
            // Two leading spaces, a bar, the cells, and a closing bar.
            assert_eq!(row.chars().count(), 3 + 32 + 1, "{row}");
        }
    }

    /// The scale bar is what makes a normalized image honest.
    #[test]
    fn heatmap_labels_its_range() {
        let (_, field) = ramp_field();
        let text = heatmap(&field, 20, 5);
        assert!(text.contains("scale:"), "{text}");
        assert!(text.contains("min"), "{text}");
        assert!(text.contains("max"), "{text}");
    }

    #[test]
    fn a_horizontal_ramp_renders_left_to_right() {
        let (_, field) = ramp_field();
        let text = heatmap(&field, 10, 3);
        let first_row = text.lines().next().unwrap();
        let cells: Vec<char> = first_row.chars().skip(3).take(10).collect();
        // The leftmost cell must be lighter than the rightmost.
        let index = |c: char| RAMP.iter().position(|&r| r == c as u8).unwrap();
        assert!(index(cells[0]) < index(cells[9]), "{cells:?}");
    }

    /// Increasing y must appear higher on screen, not lower. Getting this backwards
    /// makes every buoyancy and gravity scene look wrong in a way that is easy to
    /// mistake for a physics bug.
    #[test]
    fn increasing_y_renders_upward() {
        let grid = Grid2d::new(8, 8, [1.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        field.init_from_position(&grid, |[_, y]| y);
        let text = heatmap(&field, 8, 4);
        let rows: Vec<&str> = text.lines().filter(|l| l.starts_with("  |")).collect();

        let index = |line: &str| {
            let c = line.chars().nth(3).unwrap();
            RAMP.iter().position(|&r| r == c as u8).unwrap()
        };
        assert!(index(rows[0]) > index(rows[3]), "the top row should hold the high-y values");
    }

    #[test]
    fn a_uniform_field_renders_without_dividing_by_zero() {
        let grid = Grid2d::new(8, 8, [1.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        field.fill_interior(300.0);
        let text = heatmap(&field, 8, 4);
        assert!(text.contains("300.0000"), "{text}");
        assert!(!text.contains("NaN"), "{text}");
    }

    #[test]
    fn non_finite_cells_are_marked_and_warned_about() {
        let grid = Grid2d::new(8, 8, [1.0, 1.0]);
        let mut field = ScalarField::new(&grid, 1);
        field.fill_interior(1.0);
        field.set(4, 4, f64::NAN);
        let text = heatmap(&field, 8, 8);
        assert!(text.contains('!'), "the NaN cell must be visible:\n{text}");
        assert!(text.contains("WARNING"), "{text}");
    }

    #[test]
    fn scatter_places_particles_and_counts_strays() {
        let xs = [0.1, 0.9, 5.0];
        let ys = [0.1, 0.9, 5.0];
        let text = scatter(&xs, &ys, [0.0, 0.0], [1.0, 1.0], 10, 5);
        assert!(text.contains("3 particles"), "{text}");
        assert!(text.contains("1 outside the view"), "{text}");
    }

    #[test]
    fn sparkline_reports_its_range() {
        let values: Vec<f64> = (0..50).map(|i| (i as f64 / 10.0).sin()).collect();
        let line = sparkline(&values, 20);
        assert!(line.contains(".."), "{line}");
        assert_eq!(line.chars().take(20).count(), 20);
    }

    /// A trace shorter than the plot width must use one column per sample and span
    /// its whole range. Striding by the width instead compresses it into the first
    /// fraction of the plot and silently drops the tail.
    #[test]
    fn a_short_trace_spans_its_samples_not_a_fraction_of_them() {
        // A monotone ramp of exactly as many samples as the ramp has levels must
        // render as the ramp itself: one column per sample, spanning min to max.
        let values: Vec<f64> = (0..10).map(f64::from).collect();
        let line = sparkline(&values, 40);
        let plot: String = line.chars().take(RAMP.len()).collect();
        assert_eq!(plot, String::from_utf8_lossy(RAMP), "{line}");
    }

    #[test]
    fn sparkline_handles_degenerate_input() {
        assert_eq!(sparkline(&[], 10), "(no samples)");
        assert!(sparkline(&[5.0, 5.0, 5.0], 3).contains("5.000000e0"));
        assert!(sparkline(&[1.0, f64::NAN], 2).contains('!'));
    }

    /// A conserved quantity wobbling at the 16th digit must render flat, not as a
    /// dramatic oscillation. The naive normalization turns round-off into a picture
    /// of instability that is not there.
    #[test]
    fn a_quantity_conserved_to_round_off_renders_flat() {
        let values = [1.0, 1.0 + 2e-16, 1.0 - 1e-16, 1.0, 1.0 + 1e-16];
        let line = sparkline(&values, 40);
        assert!(line.contains("constant at"), "{line}");
        assert!(line.contains("varying by"), "{line}");

        let cells: Vec<char> = line.chars().take(5).collect();
        assert!(cells.iter().all(|c| *c == cells[0]), "every cell should match: {line}");

        // A genuine variation of the same absolute size but on a small magnitude is
        // still real and must still be drawn.
        let real = [0.0, 2e-16, 1e-16];
        assert!(!sparkline(&real, 10).contains("constant at"));

        // And a NaN must never be summarized away, even in an otherwise flat trace.
        let with_nan = [1.0, 1.0, f64::NAN, 1.0];
        let nan_line = sparkline(&with_nan, 10);
        assert!(nan_line.contains('!'), "{nan_line}");
        assert!(!nan_line.contains("constant at"), "{nan_line}");
    }

    /// A symplectic integrator's energy oscillates by a tiny fraction of its value.
    /// Drawn at full scale beside a series that genuinely swings, that reads as
    /// instability — so the plot states how small the variation really is.
    #[test]
    fn a_nearly_constant_series_says_how_nearly() {
        // Total energy wobbling by 1 part in 40,000, as velocity Verlet produces.
        let values: Vec<f64> =
            (0..40).map(|i| 4.0 + 1e-4 * (i as f64 / 4.0).sin()).collect();
        let line = sparkline(&values, 40);
        assert!(line.contains("relative, drawn at full scale"), "{line}");

        // A series that genuinely doubles needs no such caveat.
        let swinging: Vec<f64> = (0..40).map(|i| 1.0 + i as f64 / 40.0).collect();
        assert!(!sparkline(&swinging, 40).contains("drawn at full scale"));
    }
}
