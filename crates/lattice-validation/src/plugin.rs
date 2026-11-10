//! Validation of the example plugin (spec M6.2 §7, FR-015).
//!
//! `plugins/vacf` is a crate outside the engine's workspace that builds against
//! `lattice-sdk` alone. Its test is that it builds with
//! `cargo build --manifest-path plugins/vacf/Cargo.toml` against the engine as it
//! stands, runs, and measures something with a known answer. This case does exactly
//! that: it has cargo build and run the plugin's binary, as a user's script would, and
//! reads the curve it prints.
//!
//! # The physics
//!
//! A force-free gas in a Langevin bath of friction `γ` moves by BAOAB, whose O step
//! is the exact Ornstein-Uhlenbeck update `v ← c₁v + c₂ξ`, `c₁ = exp(−γΔt)`. With no
//! force the B and A steps leave the velocity alone, so the velocity at the samples is
//! exactly an AR(1) process with `ρ = exp(−γΔ)` per sample interval `Δ`, whatever the
//! timestep, and `⟨v(t)·v(t+τ)⟩ = exp(−γτ)⟨v(t)·v(t)⟩`. So the normalised velocity
//! autocorrelation is `exp(−γτ)` in expectation, with no discretisation error to allow
//! for. What remains is statistical, and the tolerance is the statistics.
//!
//! # The tolerance
//!
//! Not tuned: it is the estimator's own standard error, computed from the number of
//! particles `N` and of time origins `M`, and the pass limit is a fixed number of those.
//!
//! The plugin's estimate at lag `k` is a ratio, `Ĉ_k = Σ v(t)·v(t+kΔ) / Σ v(t)·v(t)`,
//! summed over the `M_k = M − k` origins that have lag `k` and over the `N` particles.
//! The lag-0 value is exactly 1 and carries no error. For `k ≥ 1`, per velocity
//! component of unit variance, write `x_j` for the sample at origin `j`; the error of
//! the ratio is, to first order, the mean of
//!
//! ```text
//! Z_j = x_j x_{j+k} − ρᵏ x_j²           (mean zero, because ⟨x_j x_{j+k}⟩ = ρᵏ⟨x_j²⟩)
//! ```
//!
//! The components are Gaussian with `⟨x_j x_j'⟩ = r(j−j') = ρ^|j−j'|`, so Isserlis'
//! theorem gives the covariance of `Z` at origin separation `δ = j − j'` exactly:
//!
//! ```text
//! cov(δ) = r(δ)² + r(δ−k) r(δ+k) − 2ρᵏ r(δ) [r(δ+k) + r(δ−k)] + 2ρ²ᵏ r(δ)²
//! ```
//!
//! Origins overlap (every sample is one), so `Z_j` are correlated and `M_k` origins are
//! worth fewer than `M_k` independent ones; summing `cov` over all pairs of origins
//! accounts for that. The `N` particles and the 2 components of each are independent
//! (each has its own noise), so
//!
//! ```text
//! σ_k² = (1 / (2 N M_k²)) Σ_{δ=−(M_k−1)}^{M_k−1} (M_k − |δ|) cov(δ)
//! ```
//!
//! With uncorrelated origins this is `1/(2 N M_k)`, the standard error of a mean over
//! `2 N M_k` unit-variance products. The estimator's bias is `O(1/(N M))`, far under `σ`.
//!
//! The case measures `max_k |Ĉ_k − exp(−γ τ_k)| / σ_k` over lags `1..=40`, and passes at
//! or under 5. Five standard errors is a two-sided Gaussian tail of 5.7e-7 per lag, so a
//! correct plugin fails with probability under 2.3e-5 over 40 lags (the lags are
//! correlated, which only lowers that). It was chosen before the run and is not adjusted
//! to it. A plugin or an engine that got the decay rate wrong by 10% is far outside it
//! (a unit test checks this).
//!
//! The spec's sketch gives `√(2/(N·M))` at lag 0. The estimator here is normalised by
//! its own lag-0 sum, which makes lag 0 exact and the error at other lags the one above.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[Case {
    name: "plugin_vacf_ideal_gas",
    domain: "plugin",
    level: Level::Analytic,
    claim: "plugins/vacf, built outside the workspace against lattice-sdk only, measures C(tau) = exp(-gamma tau) for a force-free gas in a Langevin bath, within 5 standard errors derived from the particle and origin counts",
    run: vacf_ideal_gas,
}];

/// Friction, per second. The model below states it as `1 / picosecond`.
const GAMMA: f64 = 1e12;
/// The longest lag the plugin keeps, in sample intervals (4 ps, where `exp(−γτ) ≈ 0.018`).
const LAGS: usize = 40;
/// Pass limit, in standard errors. See the module comment.
const Z_LIMIT: f64 = 5.0;
/// Velocity components per particle.
const COMPONENTS: f64 = 2.0;

/// 400 atoms, no forces, in a Langevin bath of 1/ps. 10 000 steps of 10 fs sampled every
/// 0.1 ps give 1001 samples, each an origin, against a correlation time of 1 ps.
const MODEL: &str = "project ideal_gas {
  dimensions: 2;
  fidelity: molecular;
  duration: 100 picosecond;

  particles atoms {
    count:   400;
    region:  [8 nanometer, 8 nanometer];
    spacing: 0.4 nanometer;
    mass:    39.948 dalton;
    temperature: 94 kelvin;
    seed:    20261109;
    boundary: periodic;
    thermostat: langevin(temperature=94 kelvin, friction=1 / picosecond);
  }

  solve dynamics(atoms) with velocity_verlet(dt=10 femtosecond);

  observe atoms every 0.1 picosecond;
}
";

/// What the plugin printed.
struct Measured {
    particles: usize,
    samples: usize,
    /// `(τ in seconds, C)` for lags 0, 1, …
    curve: Vec<(f64, f64)>,
}

/// The plugin's manifest, found from this crate's own location.
fn manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/vacf/Cargo.toml")
}

/// Build the plugin with cargo and run it on `MODEL`.
///
/// Its own target directory, under `CARGO_TARGET_DIR` when that is set and beside the
/// plugin otherwise: the plugin is a separate workspace, and sharing the engine's
/// directory would have the two cargo runs wait on each other's lock.
fn run_plugin() -> Measured {
    let manifest = manifest();
    assert!(manifest.exists(), "the plugin's manifest is missing: {}", manifest.display());
    let target = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir).join("plugin-vacf"),
        None => manifest.with_file_name("target"),
    };
    let model = std::env::temp_dir().join(format!("lattice-validation-vacf-{}.lattice", std::process::id()));
    std::fs::write(&model, MODEL).expect("the temporary model is written");

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["run", "--release", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target)
        .arg("--")
        .arg(&model)
        .args(["atoms", "--lags", &LAGS.to_string()])
        .output()
        .expect("cargo runs");
    std::fs::remove_file(&model).ok();
    assert!(
        output.status.success(),
        "`cargo run --manifest-path plugins/vacf/Cargo.toml` failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse(&String::from_utf8_lossy(&output.stdout))
}

/// Read the plugin's output: a `# vacf …` header, then `lag tau C` lines.
fn parse(text: &str) -> Measured {
    let header = text.lines().find(|line| line.starts_with("# vacf ")).expect("the plugin prints a header");
    let field = |key: &str| -> &str {
        header
            .split_whitespace()
            .find_map(|word| word.strip_prefix(key))
            .unwrap_or_else(|| panic!("the header has no `{key}`: {header}"))
    };
    let particles = field("particles=").parse().expect("a particle count");
    let samples = field("samples=").parse().expect("a sample count");
    let curve = text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let _lag = parts.next();
            let tau = parts.next().and_then(|s| s.parse().ok()).expect("tau");
            let c = parts.next().and_then(|s| s.parse().ok()).expect("C");
            (tau, c)
        })
        .collect();
    Measured { particles, samples, curve }
}

/// The standard error of `Ĉ_k` for an AR(1) velocity with correlation `rho` per sample,
/// from `particles` independent particles and `origins` overlapping origins (see the
/// module comment for the derivation).
fn standard_error(rho: f64, lag: usize, origins: usize, particles: usize) -> f64 {
    let r = |m: i64| rho.powi(m.unsigned_abs() as i32);
    let k = lag as i64;
    let rho_k = rho.powi(lag as i32);
    let m = origins as i64;
    let mut sum = 0.0;
    for delta in -(m - 1)..m {
        let cov = r(delta).powi(2) + r(delta - k) * r(delta + k)
            - 2.0 * rho_k * r(delta) * (r(delta + k) + r(delta - k))
            + 2.0 * rho_k * rho_k * r(delta).powi(2);
        sum += (m - delta.abs()) as f64 * cov;
    }
    (sum / (COMPONENTS * particles as f64 * (origins * origins) as f64)).sqrt()
}

/// How far a measured curve is from `exp(−γτ)`.
struct Judgement {
    /// The largest `|Ĉ − exp(−γτ)| / σ` over lags `1..`.
    worst_z: f64,
    worst_lag: usize,
    /// The largest `|Ĉ − exp(−γτ)|`.
    worst_deviation: f64,
    /// `σ` at lag 1 and at the last lag.
    first_sigma: f64,
    last_sigma: f64,
}

/// Judge `curve` (`(τ, C)` from lag 0) against `exp(−γτ)`, with `samples` origins in all
/// and `particles` particles.
fn judge(curve: &[(f64, f64)], gamma: f64, samples: usize, particles: usize) -> Judgement {
    assert!(curve.len() >= 2, "a curve needs a lag besides 0");
    assert!((curve[0].1 - 1.0).abs() < 1e-12, "C(0) is 1 by construction, found {}", curve[0].1);
    let rho = (-gamma * curve[1].0).exp();
    let mut result =
        Judgement { worst_z: 0.0, worst_lag: 0, worst_deviation: 0.0, first_sigma: 0.0, last_sigma: 0.0 };
    for (lag, &(tau, c)) in curve.iter().enumerate().skip(1) {
        let sigma = standard_error(rho, lag, samples - lag, particles);
        let deviation = (c - (-gamma * tau).exp()).abs();
        let z = deviation / sigma;
        if z > result.worst_z || !z.is_finite() {
            result.worst_z = z;
            result.worst_lag = lag;
        }
        result.worst_deviation = result.worst_deviation.max(deviation);
        if lag == 1 {
            result.first_sigma = sigma;
        }
        result.last_sigma = sigma;
    }
    result
}

fn vacf_ideal_gas() -> Outcome {
    let measured = run_plugin();
    assert_eq!(measured.curve.len(), LAGS + 1, "one point per lag, 0 to {LAGS}");
    let judgement = judge(&measured.curve, GAMMA, measured.samples, measured.particles);
    Outcome::at_most(
        "largest |C(tau) - exp(-gamma tau)| over lags 1..=40, in standard errors of the estimate",
        "sigma",
        judgement.worst_z,
        Z_LIMIT,
    )
    .note(format!(
        "{} particles, {} samples, every one a time origin, gamma = 1/ps, sample interval 0.1 ps; \
         sigma is derived from those counts, not tuned",
        measured.particles, measured.samples
    ))
    .note(format!(
        "worst at lag {}; largest absolute deviation {:.2e}; sigma {:.2e} at lag 1 and {:.2e} at lag {LAGS}",
        judgement.worst_lag, judgement.worst_deviation, judgement.first_sigma, judgement.last_sigma
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::Pcg32;

    fn exact(gamma: f64) -> Vec<(f64, f64)> {
        (0..=LAGS).map(|k| (k as f64 * 1e-13, (-gamma * k as f64 * 1e-13).exp())).collect()
    }

    /// The judge has teeth: the exact curve is at zero standard errors, and a decay rate
    /// 10% off is far beyond the limit.
    #[test]
    fn a_wrong_decay_rate_is_judged_a_failure() {
        let right = judge(&exact(GAMMA), GAMMA, 1001, 400);
        assert!(right.worst_z < 1e-6, "{}", right.worst_z);
        let wrong = judge(&exact(1.1 * GAMMA), GAMMA, 1001, 400);
        assert!(wrong.worst_z > 3.0 * Z_LIMIT, "{}", wrong.worst_z);
    }

    /// With origins far apart the formula is the textbook `1 / (2 N M)`.
    #[test]
    fn uncorrelated_origins_give_the_plain_standard_error() {
        let sigma = standard_error(1e-9, 1, 1000, 400);
        let plain = (1.0 / (2.0 * 400.0 * 1000.0_f64)).sqrt();
        assert!((sigma / plain - 1.0).abs() < 1e-6, "{sigma} vs {plain}");
    }

    /// The derived standard error is the scatter the estimator really has: simulate the
    /// AR(1) velocities the plugin sees, estimate `Ĉ_k` as the plugin does, repeat, and
    /// compare the spread of the estimates with `standard_error`.
    #[test]
    fn the_standard_error_is_the_estimators_scatter() {
        let (rho, particles, samples, repeats) = (0.9_f64, 40usize, 200usize, 600usize);
        let drive = (1.0 - rho * rho).sqrt();
        let lags = [1usize, 5, 20];
        let mut estimates = vec![Vec::new(); lags.len()];
        let mut rng = Pcg32::seed_with_stream(11, 3);
        for _ in 0..repeats {
            let mut sums = vec![(0.0, 0.0); lags.len()];
            for _ in 0..particles {
                for _ in 0..COMPONENTS as usize {
                    let mut x = Vec::with_capacity(samples);
                    let mut value = rng.normal();
                    for _ in 0..samples {
                        x.push(value);
                        value = rho * value + drive * rng.normal();
                    }
                    for (slot, &lag) in lags.iter().enumerate() {
                        for j in 0..samples - lag {
                            sums[slot].0 += x[j] * x[j + lag];
                            sums[slot].1 += x[j] * x[j];
                        }
                    }
                }
            }
            for (slot, (cross, norm)) in sums.into_iter().enumerate() {
                estimates[slot].push(cross / norm);
            }
        }
        for (slot, &lag) in lags.iter().enumerate() {
            let n = estimates[slot].len() as f64;
            let mean = estimates[slot].iter().sum::<f64>() / n;
            let spread = (estimates[slot].iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            let derived = standard_error(rho, lag, samples - lag, particles);
            // The spread of 600 estimates is known to about 3% (1/sqrt(2·599)); 12% is four of those.
            assert!(
                (spread / derived - 1.0).abs() < 0.12,
                "lag {lag}: simulated scatter {spread:.4e} against derived {derived:.4e}"
            );
            assert!((mean - rho.powi(lag as i32)).abs() < 5.0 * derived / (repeats as f64).sqrt(), "lag {lag}: biased, mean {mean}");
        }
    }

    #[test]
    fn the_plugins_output_is_read() {
        let text = "# vacf model=m set=atoms particles=400 samples=1001 interval=1e-13\n# note\n# lag tau_seconds C\n0 0e0 1e0\n1 1e-13 9.0e-1\n";
        let measured = parse(text);
        assert_eq!((measured.particles, measured.samples), (400, 1001));
        assert_eq!(measured.curve, [(0.0, 1.0), (1e-13, 0.9)]);
    }
}
