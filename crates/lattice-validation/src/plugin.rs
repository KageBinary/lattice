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

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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
#[derive(Debug)]
struct Measured {
    particles: usize,
    samples: usize,
    /// `(τ in seconds, C)` for lags 0, 1, …
    curve: Vec<(f64, f64)>,
}

/// A path under the temporary directory that no other call in any thread of any process
/// uses: the process id separates processes, and a counter separates the calls within one.
/// The registry runs cases on several threads when tests call `run_all` in parallel, and a
/// path shared between them would be removed by one while another's plugin still read it.
fn unique_temp(stem: &str, extension: &str) -> PathBuf {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}-{}-{call}.{extension}", std::process::id()))
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
///
/// Every way this can go wrong is an `Err` saying why, never a panic: a case that
/// panics aborts the whole validation run, and the cases after it would not run.
fn run_plugin() -> Result<Measured, String> {
    let manifest = manifest();
    let target = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir).join("plugin-vacf"),
        None => manifest.with_file_name("target"),
    };
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    run_plugin_with(&cargo, &manifest, &target, MODEL)
}

/// [`run_plugin`] with the cargo program, the manifest, the target directory and the
/// model given.
fn run_plugin_with(cargo: &OsStr, manifest: &Path, target: &Path, model_text: &str) -> Result<Measured, String> {
    if !manifest.exists() {
        return Err(format!("the plugin's manifest is missing: {}", manifest.display()));
    }
    let model = unique_temp("lattice-validation-vacf", "lattice");
    std::fs::write(&model, model_text)
        .map_err(|error| format!("cannot write the temporary model {}: {error}", model.display()))?;
    let output = Command::new(cargo)
        .args(["run", "--release", "--quiet", "--manifest-path"])
        .arg(manifest)
        .arg("--target-dir")
        .arg(target)
        .arg("--")
        .arg(&model)
        .args(["atoms", "--lags", &LAGS.to_string()])
        .output();
    std::fs::remove_file(&model).ok();
    let output = output.map_err(|error| format!("cannot run `{}`: {error}", cargo.to_string_lossy()))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo run --manifest-path {}` failed ({}):\n{}{}",
            manifest.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse(&String::from_utf8_lossy(&output.stdout))
}

/// Read the plugin's output: a `# vacf …` header, then `lag tau C` lines.
fn parse(text: &str) -> Result<Measured, String> {
    let header = text
        .lines()
        .find(|line| line.starts_with("# vacf "))
        .ok_or("the plugin printed no `# vacf` header")?;
    let field = |key: &str| -> Result<usize, String> {
        header
            .split_whitespace()
            .find_map(|word| word.strip_prefix(key))
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| format!("the header has no usable `{key}`: {header}"))
    };
    let (particles, samples) = (field("particles=")?, field("samples=")?);
    let mut curve = Vec::new();
    for line in text.lines().filter(|line| !line.starts_with('#') && !line.trim().is_empty()) {
        let mut parts = line.split_whitespace();
        let _lag = parts.next();
        let mut number = |what: &str| -> Result<f64, String> {
            parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("no {what} in the line `{line}`"))
        };
        curve.push((number("tau")?, number("C")?));
    }
    Ok(Measured { particles, samples, curve })
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
fn judge(curve: &[(f64, f64)], gamma: f64, samples: usize, particles: usize) -> Result<Judgement, String> {
    if curve.len() < 2 {
        return Err(format!("the curve has {} point(s); it needs a lag besides 0", curve.len()));
    }
    if (curve[0].1 - 1.0).abs() >= 1e-12 {
        return Err(format!("C(0) is 1 by construction, but the plugin reported {}", curve[0].1));
    }
    if samples <= curve.len() {
        return Err(format!("{samples} samples cannot give {} lags", curve.len()));
    }
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
    Ok(result)
}

/// A report of a case that could not be measured: the observed value is NaN, which no
/// criterion accepts, and the note says why.
fn unmeasured(reason: String) -> Outcome {
    Outcome::at_most("the case could not measure the plugin", "sigma", f64::NAN, Z_LIMIT).note(reason)
}

fn vacf_ideal_gas() -> Outcome {
    let measured = match run_plugin() {
        Ok(measured) => measured,
        Err(reason) => return unmeasured(reason),
    };
    if measured.curve.len() != LAGS + 1 {
        return unmeasured(format!(
            "expected one point per lag, 0 to {LAGS}; the plugin printed {}",
            measured.curve.len()
        ));
    }
    let judgement = match judge(&measured.curve, GAMMA, measured.samples, measured.particles) {
        Ok(judgement) => judgement,
        Err(reason) => return unmeasured(reason),
    };
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
    use crate::Criterion;
    use lattice_ir::Pcg32;

    fn exact(gamma: f64) -> Vec<(f64, f64)> {
        (0..=LAGS).map(|k| (k as f64 * 1e-13, (-gamma * k as f64 * 1e-13).exp())).collect()
    }

    /// The judge has teeth: the exact curve is at zero standard errors, and a decay rate
    /// 10% off is far beyond the limit.
    #[test]
    fn a_wrong_decay_rate_is_judged_a_failure() {
        let right = judge(&exact(GAMMA), GAMMA, 1001, 400).unwrap();
        assert!(right.worst_z < 1e-6, "{}", right.worst_z);
        let wrong = judge(&exact(1.1 * GAMMA), GAMMA, 1001, 400).unwrap();
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

    /// A plugin that cannot be run is a failed report with the reason, not a panic that
    /// would abort the whole validation run and the cases after this one.
    #[test]
    fn a_missing_manifest_is_a_reason_not_a_panic() {
        let missing = Path::new("no-such-directory").join("Cargo.toml");
        let error = run_plugin_with(OsStr::new("cargo"), &missing, Path::new("target"), MODEL).expect_err("a missing manifest is an error");
        assert!(error.contains("manifest is missing") && error.contains("no-such-directory"), "{error}");
    }

    #[test]
    fn a_missing_cargo_is_a_reason_not_a_panic() {
        let error = run_plugin_with(OsStr::new("no-such-cargo-program"), &manifest(), Path::new("target"), MODEL).expect_err("a missing cargo is an error");
        assert!(error.contains("cannot run `no-such-cargo-program`"), "{error}");
    }

    /// A build that fails is reported with its exit status. The real cargo is run on a
    /// manifest that exists but is not valid TOML, which cargo certainly rejects.
    #[test]
    fn a_failing_build_is_a_reason_not_a_panic() {
        let directory = unique_temp("lattice-validation-broken-plugin", "d");
        std::fs::create_dir_all(&directory).unwrap();
        let manifest = directory.join("Cargo.toml");
        std::fs::write(&manifest, "this is [not toml").unwrap();
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let result = run_plugin_with(&cargo, &manifest, &directory.join("target"), MODEL);
        std::fs::remove_dir_all(&directory).ok();
        let error = result.expect_err("a manifest that is not TOML cannot build");
        assert!(error.contains("failed") && error.contains("Cargo.toml"), "{error}");
    }

    /// Two calls never share a temporary path.
    #[test]
    fn temporary_paths_are_unique_per_call() {
        let (a, b) = (unique_temp("x", "lattice"), unique_temp("x", "lattice"));
        assert_ne!(a, b);
    }

    /// Whatever goes wrong, the case reports a failure and still returns.
    #[test]
    fn an_unmeasured_case_fails() {
        let outcome = unmeasured("because".to_string());
        assert!(!outcome.passed());
        assert_eq!(outcome.notes, ["because"]);
    }

    #[test]
    fn a_malformed_output_is_a_reason_not_a_panic() {
        assert!(parse("nothing useful").expect_err("no header").contains("header"));
        assert!(parse("# vacf particles=4 samples=x
").is_err());
        assert!(parse("# vacf particles=4 samples=9
0 0e0
").is_err());
        assert!(judge(&[(0.0, 1.0)], GAMMA, 1001, 400).is_err());
        assert!(judge(&[(0.0, 0.5), (1e-13, 0.9)], GAMMA, 1001, 400).is_err());
    }

    /// A NaN lag in the plugin's curve is a failure, not a pass.
    #[test]
    fn a_nan_lag_is_judged_a_failure() {
        let mut curve = exact(GAMMA);
        curve[3].1 = f64::NAN;
        let judgement = judge(&curve, GAMMA, 1001, 400).unwrap();
        assert!(!Criterion::AtMost { limit: Z_LIMIT }.accepts(judgement.worst_z));
    }

    #[test]
    fn the_plugins_output_is_read() {
        let text = "# vacf model=m set=atoms particles=400 samples=1001 interval=1e-13\n# note\n# lag tau_seconds C\n0 0e0 1e0\n1 1e-13 9.0e-1\n";
        let measured = parse(text).unwrap();
        assert_eq!((measured.particles, measured.samples), (400, 1001));
        assert_eq!(measured.curve, [(0.0, 1.0), (1e-13, 0.9)]);
    }
}
