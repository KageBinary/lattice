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
//! together with the running sums. The ring and the sums are never reallocated: at each
//! sample [`Observer::observe`] overwrites the oldest snapshot and adds one dot product
//! per lag and particle. The one allocation left is the row's name, cloned each time the
//! row is recorded, because the name is built from the domain's and `record_metric`
//! wants an owned or `'static` string.
//!
//! # What it cannot say
//!
//! The contract lists the limitations, and they are the honest ones: lags are multiples
//! of the sample interval, origins overlap so there is no error bar, particles are
//! tracked by slot, and the longest lag is the ring's length.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
    /// The same count, readable after the observer has been handed to the simulation.
    shared_samples: Arc<AtomicU64>,
    first_time: f64,
    /// The time of the latest sample taken.
    last_time: f64,
    /// The sample interval, seconds, from the first two samples.
    interval: Option<f64>,
}

/// How far apart two sample spacings can be through the clock's round-off alone.
///
/// The simulation time at `step` is the sum of `step` timesteps, and a sum of `n` positive
/// terms in `f64` is off by at most `n · ε/2` of its value. A spacing is the difference of
/// two such times, and the interval it is compared with is one too, measured early; so the
/// two disagree by at most twice `step · ε` of the later time. Anything larger is a real
/// change of spacing, about a timestep at least. Derived, not tuned.
fn clock_roundoff(time: f64, step: u64) -> f64 {
    2.0 * step as f64 * f64::EPSILON * time
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
                Cow::Borrowed("lags are multiples of the sample interval, which must be constant and is measured from the first two samples; a later sample spaced differently disables the observer (E0951) rather than being mixed in, as happens when `every` is not a multiple of the timestep"),
                Cow::Borrowed("one run only: a second run on the same simulation samples its start at the instant the first ended, and the observer refuses it (disables itself, E0951) rather than store the same state twice; take the curve between runs"),
                Cow::Borrowed("a sample closer to the previous one than a full interval (the run's final sample, when the duration is not a multiple of the interval) is skipped"),
                Cow::Borrowed("origins overlap (every sample is an origin), so neighbouring origins are correlated; there is no error bar, and the number of independent origins is smaller than the count"),
                Cow::Borrowed("particles are followed by slot: if one is spawned, destroyed or reordered, or the count changes, the correlation is wrong, and a changed count disables the observer"),
                Cow::Borrowed("lags stop at the ring's length, and the longer lags have fewer origins"),
                Cow::Borrowed("the row is the running estimate over the origins so far, not the final curve's value, and is absent until the second sample"),
                Cow::Borrowed("velocities are used as they are: the centre-of-mass drift is not subtracted, so a net drift of the particles adds to C"),
                Cow::Borrowed("2D only: it reads vel_x and vel_y of the particle domain and nothing else"),
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
            shared_samples: Arc::new(AtomicU64::new(0)),
            first_time: 0.0,
            last_time: 0.0,
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

    /// A handle on the number of samples taken, which is the number of time origins at
    /// lag 0. It stays readable after the observer is attached to a simulation.
    pub fn sample_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.shared_samples)
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

        // Refuse, rather than reset or mix in, a sample that does not follow the previous
        // one by one regular interval. Both checks panic, which the runtime turns into
        // E0951 and disables the observer; they come before any state is touched. A
        // reset on a restart would throw away a finished run's data, and a tolerant
        // average over uneven spacing would put lags at the wrong τ without saying so.
        let time = ctx.time();
        if self.samples > 0 {
            if time <= self.last_time || time.is_nan() {
                panic!(
                    "vacf was sampled at t = {time:e} s, not after the previous sample at {:e} s: the simulation was run again, which this observer does not support (take the curve between runs)",
                    self.last_time
                );
            }
            if let Some(interval) = self.interval {
                let spacing = time - self.last_time;
                if (spacing - interval).abs() > clock_roundoff(time, ctx.step()) {
                    panic!(
                        "vacf sample spacing changed from {interval:e} s to {spacing:e} s at t = {time:e} s: it needs evenly spaced samples, so `every` must be a multiple of the timestep"
                    );
                }
            }
        }

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
            0 => self.first_time = time,
            1 => self.interval = Some(time - self.first_time),
            _ => {}
        }
        self.last_time = time;
        self.samples += 1;
        self.shared_samples.store(self.samples, Ordering::Relaxed);

        // No row until a lag-1 pair exists, and none when nothing moves: C is 0/0 there,
        // and a NaN row would halt the run.
        if self.samples >= 2 && self.denominator[1] > 0.0 {
            out.record_metric(self.row.clone(), self.numerator[1] / self.denominator[1], "1");
        }
    }

    /// Every sample, except one that comes sooner than a full interval after the last:
    /// that is the run's final sample, taken wherever the duration ends, and it is off
    /// the grid. The runtime's own rule puts a sample on the grid within half a step, so
    /// "sooner than an interval by more than half a step" cannot be a grid sample. A
    /// restart (a time not after the last) is let through, so `observe` refuses it.
    fn due(&mut self, time: f64, dt: f64) -> bool {
        match self.interval {
            Some(interval) if self.samples > 0 && time > self.last_time => {
                time - self.last_time >= interval - 0.5 * dt
            }
            _ => true,
        }
    }

    fn curves(&self) -> Vec<Curve> {
        let Some(interval) = self.interval.filter(|dt| *dt > 0.0) else { return Vec::new() };
        // Every lag that has an origin is a point, at its true τ, so the point's index is
        // its lag. A lag whose origins all have zero speed has no defined C (0/0): it is
        // NaN, and the note says so, rather than leaving a gap that shifts the later lags.
        let last = self.lags.min(self.samples.saturating_sub(1) as usize);
        let tau: Vec<f64> = (0..=last).map(|lag| lag as f64 * interval).collect();
        let c: Vec<f64> = (0..=last)
            .map(|lag| {
                if self.denominator[lag] > 0.0 {
                    self.numerator[lag] / self.denominator[lag]
                } else {
                    f64::NAN
                }
            })
            .collect();
        let undefined = c.iter().filter(|c| c.is_nan()).count();
        let mut curve = Curve::new(self.curve.clone(), ("tau", "s"), ("C", "1"), tau, c).note(format!(
            "{} samples of {} particles, sample interval {interval:e} s, lags 0..={last} of at most {}; \
             origins per lag fall from {} at lag 0 to {} at lag {last}; \
             origins overlap and there is no error bar",
            self.samples, self.particles, self.lags, self.origins[0], self.origins[last]
        ));
        if undefined > 0 {
            curve = curve.note(format!(
                "{undefined} lag(s) are NaN: every origin of those lags had zero speed, so C is 0/0"
            ));
        }
        vec![curve]
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

    /// The curve's tau axis is `lag x interval`, computed the same way in the curve, so it is
    /// exact against that product; and the interval is the 10 timesteps of 10 fs between
    /// samples, off 1e-13 s only by the clock's round-off: ten additions of a timestep that
    /// is itself the literal `10 femtosecond` rounded, each at most eps/2 of the running sum,
    /// so 20 eps of the value bounds it with room.
    fn assert_tau_axis(x: &[f64]) {
        assert!((x[1] - 1e-13).abs() <= 20.0 * f64::EPSILON * 1e-13, "the sample interval is {} s", x[1]);
        for (lag, tau) in x.iter().enumerate() {
            assert_eq!(*tau, lag as f64 * x[1], "lag {lag} is at {tau} s");
        }
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
        // With no force and no bath each velocity is the same bit pattern at every sample,
        // so at every lag the numerator is the denominator's sum of a*a term for term:
        // C is exactly 1, not near it.
        assert_tau_axis(&curve.x);
        for (lag, c) in curve.y.iter().enumerate() {
            assert_eq!(*c, 1.0, "C at lag {lag}");
        }
        let last = outcome.artifact.timeline().last().unwrap();
        let lag1 = last.values.iter().find(|(name, _)| name == "atoms.vacf.lag1").map(|(_, v)| *v);
        assert_eq!(lag1.expect("the row is recorded"), 1.0);
        let first = &outcome.artifact.timeline()[0];
        assert!(first.values.iter().all(|(name, _)| name != "atoms.vacf.lag1"), "no lag-1 pair at the first sample");
    }

    /// Particles that never move have no defined C: the lags stay in the curve at their
    /// true τ, as NaN with a note, instead of being dropped and shifting the later lags.
    /// No row is recorded, since a NaN row would halt the run.
    #[test]
    fn particles_at_rest_leave_nan_lags_in_place() {
        let path = write_model("rest", &model("").replace("temperature: 94 kelvin;", ""));
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        let vacf = Vacf::new("atoms", 25, 6).unwrap();
        let samples = vacf.sample_counter();
        simulation.add_observer(Box::new(vacf)).expect("attaches");
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        assert!(outcome.is_success(), "{}", outcome.stop.describe());
        assert!(outcome.observer_faults.is_empty());
        assert_eq!(samples.load(std::sync::atomic::Ordering::Relaxed) as usize, outcome.artifact.timeline().len());

        let curves = simulation.curves();
        let curve = curves.iter().find(|c| c.name == "atoms.vacf").expect("the curve");
        assert_eq!(curve.x.len(), 7);
        assert_tau_axis(&curve.x);
        for (lag, c) in curve.y.iter().enumerate() {
            assert!(c.is_nan(), "C at lag {lag} is {c}");
        }
        assert!(curve.notes.iter().any(|n| n.contains("7 lag(s) are NaN")), "{:?}", curve.notes);
        assert!(outcome.artifact.timeline().iter().all(|s| s.values.iter().all(|(n, _)| n != "atoms.vacf.lag1")));
    }

    /// A second run on the same simulation samples its start at the instant the first
    /// ended. The observer refuses that sample (it would store one state twice with a
    /// spacing of zero) by disabling itself, and the curve of the first run, taken between
    /// the runs, is intact.
    #[test]
    fn a_second_run_is_refused_not_double_counted() {
        let path = write_model("twice", &model(""));
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        let vacf = Vacf::new("atoms", 25, 6).unwrap();
        let samples = vacf.sample_counter();
        simulation.add_observer(Box::new(vacf)).expect("attaches");
        let config = RunConfig::from_model(simulation.model());

        let first = simulation.run(&config);
        assert!(first.is_success() && first.observer_faults.is_empty());
        let taken = samples.load(std::sync::atomic::Ordering::Relaxed);
        let between = simulation.curves();
        assert_eq!(between.iter().find(|c| c.name == "atoms.vacf").expect("the curve").y.len(), 7);

        let second = simulation.run(&config);
        assert!(second.is_success(), "{}", second.stop.describe());
        assert_eq!(second.observer_faults.len(), 1);
        assert!(second.observer_faults[0].message.contains("was run again"), "{}", second.observer_faults[0].message);
        assert_eq!(samples.load(std::sync::atomic::Ordering::Relaxed), taken, "nothing was stored from the second run");
    }

    /// `every` that is not a multiple of the timestep makes the half-step rule space
    /// samples unevenly (10 and 11 timesteps here). The observer says so and stops rather
    /// than put lags at the wrong tau.
    #[test]
    fn uneven_sample_spacing_is_refused() {
        let uneven = model("").replace("every 0.1 picosecond", "every 0.105 picosecond");
        let path = write_model("uneven", &uneven);
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        simulation.add_observer(Box::new(Vacf::new("atoms", 25, 6).unwrap())).expect("attaches");
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        assert!(outcome.is_success(), "{}", outcome.stop.describe());
        assert_eq!(outcome.observer_faults.len(), 1);
        assert!(outcome.observer_faults[0].message.contains("spacing changed"), "{}", outcome.observer_faults[0].message);
    }

    /// The run's final sample is taken where the duration ends. Here that is 0.2 ps after
    /// the last grid sample of `every 0.3 picosecond`; the observer skips it, rather than
    /// being disabled by an off-grid spacing, and keeps its regular grid.
    #[test]
    fn the_final_off_grid_sample_is_skipped_not_fatal() {
        let ragged = model("").replace("every 0.1 picosecond", "every 0.3 picosecond");
        let path = write_model("ragged", &ragged);
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        std::fs::remove_file(&path).ok();
        let vacf = Vacf::new("atoms", 25, 6).unwrap();
        let samples = vacf.sample_counter();
        simulation.add_observer(Box::new(vacf)).expect("attaches");
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        assert!(outcome.is_success() && outcome.observer_faults.is_empty(), "{:?}", outcome.observer_faults);
        // Grid samples at 0, 0.3, ..., 4.8 ps are 17; the run also samples at 5 ps.
        assert_eq!(outcome.artifact.timeline().len(), 18);
        assert_eq!(samples.load(std::sync::atomic::Ordering::Relaxed), 17);
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
