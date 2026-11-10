//! A velocity-autocorrelation observer, written against `lattice-sdk` and nothing else.
//!
//! ```text
//! C(τ) = ⟨ v(t)·v(t+τ) ⟩ / ⟨ v(t)·v(t) ⟩
//! ```
//!
//! The average runs over the particles of one domain and over every time origin `t` the
//! run has offered: every sample is an origin. The lags `τ` are whole numbers of sample
//! intervals, `0` to `lags`. For a free particle in a Langevin bath of friction `γ`, the
//! velocity is an Ornstein-Uhlenbeck process and `C(τ) = exp(−γτ)`; in a fluid with
//! forces it is not, and the curve is the measurement.
//!
//! [`Vacf`] publishes `C(τ)` as a curve named `SET.vacf` (τ in seconds), and the running
//! estimate of `C` at one lag as the row `SET.vacf.lag1` of the timeline.
//!
//! # Memory
//!
//! The last `lags + 1` velocity snapshots live in a ring that [`Vacf::new`] allocates,
//! together with the running sums. [`Observer::observe`] allocates nothing: at each
//! sample it overwrites the oldest snapshot and adds one dot product per lag and
//! particle.
//!
//! # What it cannot say
//!
//! The contract lists the limitations, and they are the honest ones: lags are multiples
//! of the sample interval, origins overlap so there is no error bar, particles are
//! tracked by slot, and the longest lag is the ring's length.

use std::borrow::Cow;

use lattice_sdk::{
    Backends, Curve, ObserveContext, Observations, Observer, ObserverContract, ParticleView,
    SideEffects, SDK_VERSION,
};

/// The revision of `lattice-sdk` this plugin was written for.
pub const WRITTEN_FOR_SDK: (u32, u32) = (0, 1);

/// The velocity autocorrelation of one particle domain.
#[derive(Debug)]
pub struct Vacf {
    contract: ObserverContract,
    set: String,
    row: String,
    curve: String,
    particles: usize,
    lags: usize,
    /// `lags + 1` snapshots of `2 * particles` numbers each: all `vx`, then all `vy`.
    ring: Vec<f64>,
    /// `Σ v(t)·v(t+kΔ)` over the origins that have lag `k`, for `k = 0..=lags`.
    numerator: Vec<f64>,
    /// `Σ v(t)·v(t)` over those same origins.
    denominator: Vec<f64>,
    /// How many origins have lag `k`.
    origins: Vec<u64>,
    samples: u64,
    first_time: f64,
    /// The sample interval, seconds, from the first two samples.
    interval: Option<f64>,
}

impl Vacf {
    /// An observer of the particle domain `set`, which holds `particles` particles,
    /// keeping lags `1..=lags` as well as lag 0.
    ///
    /// # Errors
    ///
    /// If `set` is empty, or `particles` or `lags` is 0: there would be nothing to
    /// correlate, or no lag to publish.
    pub fn new(set: impl Into<String>, particles: usize, lags: usize) -> Result<Vacf, String> {
        let set = set.into();
        if set.is_empty() {
            return Err("vacf: the particle domain's name is empty".to_string());
        }
        if particles == 0 {
            return Err(format!("vacf: `{set}` has no particles to correlate"));
        }
        if lags == 0 {
            return Err("vacf: it needs at least one lag".to_string());
        }
        let row = format!("{set}.vacf.lag1");
        let curve = format!("{set}.vacf");
        let contract = ObserverContract {
            name: Cow::Borrowed("vacf"),
            summary: Cow::Owned(format!(
                "velocity autocorrelation C(tau) = <v(t).v(t+tau)> / <v(t).v(t)> of the particles of `{set}`, over time origins at every sample"
            )),
            reads: vec![Cow::Owned(set.clone())],
            // Units: C is a ratio of two velocity products, so it is dimensionless. The
            // curve's own axes (tau in seconds) are on the curve.
            outputs: vec![
                (Cow::Owned(row.clone()), Cow::Borrowed("1")),
                (Cow::Owned(curve.clone()), Cow::Borrowed("1")),
            ],
            // Reads and writes: it reads `v` of `set` through `ParticleView` and writes
            // only its own memory and the rows and curve above.
            side_effects: SideEffects::None,
            limitations: vec![
                Cow::Borrowed("lags are multiples of the sample interval, which is taken to be constant and is measured from the first two samples"),
                Cow::Borrowed("origins overlap (every sample is an origin), so neighbouring origins are correlated; there is no error bar, and the number of independent origins is smaller than the count"),
                Cow::Borrowed("particles are followed by slot: if one is spawned, destroyed or reordered, or the count changes, the correlation is wrong, and a changed count disables the observer"),
                Cow::Borrowed("lags stop at the ring's length, and the longer lags have fewer origins"),
                Cow::Borrowed("the row is the running estimate over the origins so far, not the final curve's value, and is absent until the second sample"),
                Cow::Borrowed("exp(-gamma tau) holds only for free particles in a Langevin bath; with forces the curve is not that"),
            ],
            deterministic: true,
            backends: Backends::CPU,
            differentiable: false,
        };
        let slots = lags + 1;
        Ok(Vacf {
            contract,
            set,
            row,
            curve,
            particles,
            lags,
            ring: vec![0.0; slots * 2 * particles],
            numerator: vec![0.0; slots],
            denominator: vec![0.0; slots],
            origins: vec![0; slots],
            samples: 0,
            first_time: 0.0,
            interval: None,
        })
    }

    /// The SDK revision this plugin was written for, and the one it is running against.
    pub fn sdk_versions() -> ((u32, u32), (u32, u32)) {
        (WRITTEN_FOR_SDK, SDK_VERSION)
    }

    /// The domain it reads.
    pub fn set(&self) -> &str {
        &self.set
    }

    /// The row it records: `C` at one lag, as a running estimate.
    pub fn row(&self) -> &str {
        &self.row
    }

    /// The curve it publishes: `C(τ)`.
    pub fn curve_name(&self) -> &str {
        &self.curve
    }

    /// The number of samples taken so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// The longest lag, in sample intervals.
    pub fn lags(&self) -> usize {
        self.lags
    }
}

impl Observer for Vacf {
    fn contract(&self) -> &ObserverContract {
        &self.contract
    }

    fn observe(&mut self, ctx: &ObserveContext<'_>, out: &mut Observations) {
        let Some(view) = ParticleView::from_context(ctx, &self.set) else {
            panic!(
                "vacf could not reach `{}` as a particle domain: it is missing, is not a particle domain, or is outside the contract's `reads`",
                self.set
            )
        };
        let n = self.particles;
        assert_eq!(
            view.len(),
            n,
            "vacf was built for {n} particles in `{}` and now sees {}",
            self.set,
            view.len()
        );

        let slots = self.lags + 1;
        let stride = 2 * n;
        let now = (self.samples % slots as u64) as usize;
        let snapshot = &mut self.ring[now * stride..(now + 1) * stride];
        snapshot[..n].copy_from_slice(view.vel_x());
        snapshot[n..].copy_from_slice(view.vel_y());

        // Every earlier snapshot still in the ring, and this one (lag 0), is an origin.
        let reach = self.samples.min(self.lags as u64) as usize;
        let current = &self.ring[now * stride..(now + 1) * stride];
        for lag in 0..=reach {
            let origin = ((self.samples - lag as u64) % slots as u64) as usize;
            let then = &self.ring[origin * stride..(origin + 1) * stride];
            let (mut cross, mut norm) = (0.0, 0.0);
            for (a, b) in then.iter().zip(current) {
                cross += a * b;
                norm += a * a;
            }
            self.numerator[lag] += cross;
            self.denominator[lag] += norm;
            self.origins[lag] += 1;
        }

        match self.samples {
            0 => self.first_time = ctx.time(),
            1 => self.interval = Some(ctx.time() - self.first_time),
            _ => {}
        }
        self.samples += 1;

        // No row until a lag-1 pair exists, and none when nothing moves: C is 0/0 there,
        // and a NaN row would halt the run.
        if self.samples >= 2 && self.denominator[1] > 0.0 {
            out.record_metric(self.row.clone(), self.numerator[1] / self.denominator[1], "1");
        }
    }

    fn curves(&self) -> Vec<Curve> {
        let Some(interval) = self.interval.filter(|dt| *dt > 0.0) else { return Vec::new() };
        let (mut tau, mut c) = (Vec::new(), Vec::new());
        for lag in 0..=self.lags {
            if self.origins[lag] > 0 && self.denominator[lag] > 0.0 {
                tau.push(lag as f64 * interval);
                c.push(self.numerator[lag] / self.denominator[lag]);
            }
        }
        let last = self.lags.min(self.samples.saturating_sub(1) as usize);
        vec![
            Curve::new(self.curve.clone(), ("tau", "s"), ("C", "1"), tau, c).note(format!(
                "{} samples of {} particles, sample interval {interval:e} s, lags 0..={}; \
                 origins per lag fall from {} at lag 0 to {} at lag {last}; \
                 origins overlap and there is no error bar",
                self.samples, self.particles, self.lags, self.origins[0], self.origins[last]
            )),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_sdk::{compile_file, RunConfig, StopReason};
    use std::path::PathBuf;

    /// A gas with no forces and, unless `thermostat` says so, no bath: each particle
    /// keeps its velocity.
    fn model(thermostat: &str) -> String {
        format!(
            "project gas {{
  dimensions: 2;
  fidelity: molecular;
  duration: 5 picosecond;

  particles atoms {{
    count:   25;
    region:  [5 nanometer, 5 nanometer];
    spacing: 1 nanometer;
    mass:    39.948 dalton;
    temperature: 94 kelvin;
    seed:    7;
    boundary: periodic;
    {thermostat}
  }}

  solve dynamics(atoms) with velocity_verlet(dt=10 femtosecond);

  observe atoms every 0.1 picosecond;
}}
"
        )
    }

    fn write_model(tag: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("vacf-{}-{tag}.lattice", std::process::id()));
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn it_refuses_nothing_to_correlate() {
        assert!(Vacf::new("", 10, 4).is_err());
        assert!(Vacf::new("atoms", 0, 4).is_err());
        assert!(Vacf::new("atoms", 10, 0).is_err());
        assert!(Vacf::new("atoms", 10, 4).is_ok());
    }

    #[test]
    fn its_contract_passes_the_audit_and_declares_its_limits() {
        let vacf = Vacf::new("atoms", 10, 4).unwrap();
        let contract = vacf.contract();
        assert!(contract.audit(&["atoms"]).is_empty());
        assert_eq!(contract.reads, ["atoms"]);
        assert_eq!(contract.outputs.len(), 2);
        assert!(contract.limitations.iter().any(|l| l.contains("multiples of the sample interval")));
        assert!(contract.limitations.iter().any(|l| l.contains("no error bar")));
        assert_eq!(contract.side_effects, SideEffects::None);
        assert!(contract.backends.cpu && !contract.backends.gpu && !contract.differentiable);
        assert_eq!(Vacf::sdk_versions(), ((0, 1), (0, 1)), "written for the SDK it runs against");
    }

    /// With no bath and no forces every velocity is constant, so C(τ) is 1 at every lag.
    #[test]
    fn free_flight_correlates_perfectly() {
        let path = write_model("flight", &model(""));
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        simulation.add_observer(Box::new(Vacf::new("atoms", 25, 6).unwrap())).expect("attaches");
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        assert!(outcome.is_success(), "{}", outcome.stop.describe());
        assert!(outcome.observer_faults.is_empty());

        let curves = simulation.curves();
        let curve = curves.iter().find(|c| c.name == "atoms.vacf").expect("the curve");
        assert_eq!(curve.y.len(), 7);
        for (lag, (tau, c)) in curve.x.iter().zip(&curve.y).enumerate() {
            assert!((c - 1.0).abs() < 1e-12, "C({tau} s) = {c}");
            assert!((tau - lag as f64 * 1e-13).abs() < 1e-24, "lag {lag} is at {tau} s");
        }
        let last = outcome.artifact.timeline().last().unwrap();
        let lag1 = last.values.iter().find(|(name, _)| name == "atoms.vacf.lag1").map(|(_, v)| *v);
        assert!((lag1.expect("the row is recorded") - 1.0).abs() < 1e-12);
        let first = &outcome.artifact.timeline()[0];
        assert!(first.values.iter().all(|(name, _)| name != "atoms.vacf.lag1"), "no lag-1 pair at the first sample");
    }

    /// A bath decorrelates the velocities, so C falls with the lag and stays below 1.
    #[test]
    fn a_bath_decorrelates() {
        let thermostat = "thermostat: langevin(temperature=94 kelvin, friction=1 / picosecond);";
        let path = write_model("bath", &model(thermostat));
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        simulation.add_observer(Box::new(Vacf::new("atoms", 25, 6).unwrap())).expect("attaches");
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        assert!(outcome.is_success());
        let curves = simulation.curves();
        let y = &curves.iter().find(|c| c.name == "atoms.vacf").unwrap().y;
        assert_eq!(y[0], 1.0);
        assert!(y[1] < 1.0 && y[6] < y[1], "{y:?}");
    }

    /// A plugin that panics is disabled and the run goes on: here the particle count it
    /// was built for is wrong.
    #[test]
    fn a_wrong_particle_count_disables_it_without_ending_the_run() {
        let path = write_model("wrong", &model(""));
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        simulation.add_observer(Box::new(Vacf::new("atoms", 24, 6).unwrap())).expect("attaches");
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        assert!(matches!(outcome.stop, StopReason::DurationReached), "{}", outcome.stop.describe());
        assert_eq!(outcome.observer_faults.len(), 1);
        assert!(outcome.observer_faults[0].message.contains("built for 24 particles"));
    }
}
