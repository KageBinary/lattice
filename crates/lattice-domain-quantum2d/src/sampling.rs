//! Measurement-inspired sampling: drawing definite outcomes from a wavefunction.
//!
//! Spec §13.1 lists *"measurement-inspired sampling"* among the module's capabilities.
//! The wavefunction is a probability amplitude, and an experiment never sees it — it
//! sees outcomes, one at a time, distributed as the amplitude says. The double slit is
//! the canonical case: Tonomura's electrons arrive as single dots, and the fringes are
//! only visible once thousands of them have landed.
//!
//! Two samplers, both seeded and deterministic (P6):
//!
//! - [`Wavefunction::sample_positions`] — the Born rule. Draw positions from `|ψ|²` as
//!   the state is *now*: a position measurement repeated on many identical copies.
//! - [`Clicks`] — a detector screen's arrivals. Of `fired` particles, each one is
//!   detected when probability current carries it across the screen, at a `y` drawn
//!   from where that current crossed.
//!
//! # What "measurement-inspired" leaves out
//!
//! Nothing here collapses the state. A real screen absorbs what it detects; this one
//! samples the flux and lets the wavefunction carry on, which is the same thing for the
//! statistics of the arrivals and different for anything that happens afterwards. It is
//! the honest reading of "inspired": the outcomes have the right distribution, and the
//! evolution is still that of one undisturbed particle.
//!
//! # How the clicks are counted
//!
//! Each step, the screen sees the forward probability `ΔP⁺ = Σ_y max(0, J_x Δt) Δy`
//! cross it. Of the `M` particles not yet detected, each is detected this step with
//! probability `ΔP⁺ / (1 − P⁺)`, where `P⁺` is what has crossed before — so the number
//! this step is `Binomial(M, ΔP⁺/(1 − P⁺))`. That thinning makes the total over a run
//! exactly `Binomial(fired, P⁺_total)` and the expected count at each `y` exactly
//! `fired · ∫J⁺ dt Δy`, with the shot noise of a real counter rather than a rounded
//! fraction. Backflow — current crossing in `−x` — clicks nothing and un-clicks
//! nothing.

use lattice_ir::{Grid2d, Pcg32};

use crate::wavefunction::Wavefunction;

impl Wavefunction {
    /// `count` positions drawn from `|ψ|²`, metres.
    ///
    /// A cell is chosen with probability `|ψ|²ΔA / norm`, and the point is placed
    /// uniformly within it: the grid knows nothing finer than a cell, and putting every
    /// sample on a centre would draw a lattice that is not in the physics. The state is
    /// renormalized for the draw, so a partly absorbed packet samples where the
    /// remaining probability is.
    ///
    /// # Panics
    ///
    /// If the state is zero everywhere: there is nowhere to find the particle.
    pub fn sample_positions(&self, rng: &mut Pcg32, count: usize) -> Vec<[f64; 2]> {
        let grid = self.grid();
        let cumulative: Vec<f64> = self
            .as_slice()
            .iter()
            .scan(0.0, |total, z| {
                *total += z.norm_sqr();
                Some(*total)
            })
            .collect();
        let total = cumulative.last().copied().unwrap_or(0.0);
        assert!(total > 0.0, "a state that is zero everywhere cannot be sampled");
        let nx = grid.nx();
        (0..count)
            .map(|_| {
                let k = pick(&cumulative, rng.next_f64() * total);
                let [x, y] = grid.cell_center(k % nx, k / nx);
                [x + (rng.next_f64() - 0.5) * grid.dx(), y + (rng.next_f64() - 0.5) * grid.dy()]
            })
            .collect()
    }
}

/// The first index whose cumulative weight exceeds `target`.
fn pick(cumulative: &[f64], target: f64) -> usize {
    cumulative.partition_point(|&c| c <= target).min(cumulative.len() - 1)
}

/// A draw from `Binomial(trials, p)`.
///
/// Inversion — walking the probability mass function up from zero — costs one step
/// per unit of the result, and its starting term `(1 − p)ⁿ` underflows once `np` reaches
/// a few hundred. So a large draw is split into pieces with mean at most 64 and their
/// results added, which is exact: binomials with one `p` sum to a binomial. Above
/// `p = ½` the complement is drawn instead, which keeps `(1 − p)ⁿ` well away from
/// underflow in every piece.
pub fn binomial(rng: &mut Pcg32, trials: u64, p: f64) -> u64 {
    if trials == 0 || p.is_nan() || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return trials;
    }
    if p > 0.5 {
        return trials - binomial(rng, trials, 1.0 - p);
    }
    let piece = ((64.0 / p).floor() as u64).max(1);
    let mut remaining = trials;
    let mut total = 0;
    while remaining > 0 {
        let n = remaining.min(piece);
        total += invert(rng, n, p);
        remaining -= n;
    }
    total
}

/// Binomial inversion for `n·p ≤ 64` and `p ≤ ½`.
fn invert(rng: &mut Pcg32, n: u64, p: f64) -> u64 {
    let ratio = p / (1.0 - p);
    let mut mass = (1.0 - p).powf(n as f64);
    let mut cumulative = mass;
    let u = rng.next_f64();
    let mut k = 0;
    while u > cumulative && k < n {
        mass *= ratio * (n - k) as f64 / (k + 1) as f64;
        k += 1;
        cumulative += mass;
    }
    k
}

/// A detector screen's individual arrivals: of `fired` particles, which have been
/// detected, and where.
///
/// See the [module documentation](self) for how they are drawn.
#[derive(Clone, Debug)]
pub struct Clicks {
    fired: u64,
    seed: u64,
    rng: Pcg32,
    /// Particles not detected yet.
    remaining: u64,
    /// Forward probability that has crossed so far, `P⁺`.
    crossed: f64,
    /// `∫J⁺ dt Δy` per row: the expected fraction of `fired` detected in that row.
    forward: Vec<f64>,
    /// Where each detection landed, m, in the order they happened.
    y: Vec<f64>,
    /// This step's forward probability per row, as a running sum for drawing a row.
    step: Vec<f64>,
}

impl Clicks {
    /// A counter for `fired` particles over a screen of `rows` rows.
    pub fn new(fired: u64, seed: u64, rows: usize) -> Self {
        Self {
            fired,
            seed,
            rng: Pcg32::seed_from_u64(seed),
            remaining: fired,
            crossed: 0.0,
            forward: vec![0.0; rows],
            y: Vec::new(),
            step: vec![0.0; rows],
        }
    }

    /// Start over: nothing detected, the generator reseeded.
    pub fn reset(&mut self) {
        *self = Self::new(self.fired, self.seed, self.forward.len());
    }

    /// Particles fired.
    pub fn fired(&self) -> u64 {
        self.fired
    }

    /// The seed the draws come from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Where each detection landed, m, in order.
    pub fn positions(&self) -> &[f64] {
        &self.y
    }

    /// `∫J⁺ dt Δy` per row: the probability of a click in that row, of which the
    /// counts are a sample.
    pub fn expected(&self) -> &[f64] {
        &self.forward
    }

    /// Detections per row.
    pub fn counts(&self, grid: Grid2d) -> Vec<u64> {
        let mut counts = vec![0u64; grid.ny()];
        let bottom = grid.origin()[1];
        for y in &self.y {
            let row = (((y - bottom) / grid.dy()).floor().max(0.0) as usize).min(grid.ny() - 1);
            counts[row] += 1;
        }
        counts
    }

    /// Record one step, given the forward probability that crossed each row in it.
    pub(crate) fn record(&mut self, grid: Grid2d, crossed: impl Iterator<Item = f64>) {
        let mut total = 0.0;
        for ((slot, forward), probability) in self.step.iter_mut().zip(&mut self.forward).zip(crossed) {
            let probability = probability.max(0.0);
            *forward += probability;
            total += probability;
            *slot = total;
        }
        if total <= 0.0 || self.remaining == 0 {
            self.crossed += total;
            return;
        }
        let undetected = 1.0 - self.crossed;
        let p = if undetected > 0.0 { (total / undetected).min(1.0) } else { 1.0 };
        let detected = binomial(&mut self.rng, self.remaining, p);
        self.remaining -= detected;
        self.crossed += total;
        let bottom = grid.cell_center(0, 0)[1];
        for _ in 0..detected {
            let row = pick(&self.step, self.rng.next_f64() * total);
            let y = bottom + row as f64 * grid.dy() + (self.rng.next_f64() - 0.5) * grid.dy();
            self.y.push(y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

    #[test]
    fn binomial_draws_have_the_right_mean_and_variance() {
        let mut rng = Pcg32::seed_from_u64(11);
        for (n, p) in [(20u64, 0.3), (5_000, 0.002), (100_000, 0.4), (40, 0.9)] {
            let draws: Vec<f64> = (0..4_000).map(|_| binomial(&mut rng, n, p) as f64).collect();
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            let variance = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64;
            let (expected_mean, expected_variance) = (n as f64 * p, n as f64 * p * (1.0 - p));
            // Five standard errors of each estimator.
            let mean_error = 5.0 * (expected_variance / draws.len() as f64).sqrt();
            let variance_error = 5.0 * expected_variance * (2.0 / draws.len() as f64).sqrt();
            assert!((mean - expected_mean).abs() < mean_error, "n {n} p {p}: mean {mean} vs {expected_mean}");
            assert!(
                (variance - expected_variance).abs() < variance_error,
                "n {n} p {p}: variance {variance} vs {expected_variance}"
            );
        }
        assert_eq!(binomial(&mut rng, 10, 0.0), 0);
        assert_eq!(binomial(&mut rng, 10, 1.0), 10);
        assert_eq!(binomial(&mut rng, 0, 0.5), 0);
    }

    #[test]
    fn born_samples_land_where_the_density_is() {
        let grid = Grid2d::with_origin(64, 64, [8.0, 8.0], [-4.0, -4.0]);
        // Everything in two cells, 3:1.
        let psi = Wavefunction::from_fn(grid, |[x, y]| {
            if (x - 1.0625).abs() < 0.01 && (y - 0.0625).abs() < 0.01 {
                Complex::real(3f64.sqrt())
            } else if (x + 2.0625).abs() < 0.01 && (y + 1.0625).abs() < 0.01 {
                Complex::real(1.0)
            } else {
                Complex::ZERO
            }
        });
        let mut rng = Pcg32::seed_from_u64(3);
        let samples = psi.sample_positions(&mut rng, 8_000);
        let first = samples.iter().filter(|p| (p[0] - 1.0625).abs() <= 0.0625 && (p[1] - 0.0625).abs() <= 0.0625).count();
        let second = samples.iter().filter(|p| (p[0] + 2.0625).abs() <= 0.0625 && (p[1] + 1.0625).abs() <= 0.0625).count();
        assert_eq!(first + second, samples.len(), "every sample lies in a cell holding probability");
        // 3/4 of 8000, with five binomial standard errors (≈ 39 each).
        assert!((first as f64 - 6_000.0).abs() < 5.0 * (8_000.0f64 * 0.75 * 0.25).sqrt(), "{first}");
        // Seeded: the same seed gives the same draws.
        assert_eq!(samples, psi.sample_positions(&mut Pcg32::seed_from_u64(3), 8_000));
    }

    #[test]
    fn clicks_are_thinned_so_nobody_is_detected_twice() {
        let grid = Grid2d::new(4, 8, [1.0, 1.0]);
        let mut clicks = Clicks::new(10_000, 5, grid.ny());
        // A screen that sees forward probability 0.1 a step, uniformly, for twelve
        // steps: 1.2 in all, more than there is. Thinning caps detection at everyone.
        for _ in 0..12 {
            clicks.record(grid, std::iter::repeat_n(0.1 / 8.0, 8));
        }
        assert_eq!(clicks.positions().len(), 10_000);
        // A row's clicks fall inside that row.
        let counts = clicks.counts(grid);
        assert_eq!(counts.iter().sum::<u64>(), 10_000);
        assert!(counts.iter().all(|&c| c > 1_000), "{counts:?}");
        // Backflow clicks nothing.
        let mut quiet = Clicks::new(100, 5, grid.ny());
        quiet.record(grid, std::iter::repeat_n(-0.1, 8));
        assert!(quiet.positions().is_empty());
        assert!(quiet.expected().iter().all(|&e| e == 0.0));
    }
}
