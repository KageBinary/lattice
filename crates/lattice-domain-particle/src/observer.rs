//! Observer laws at their use site: `observe sum(kinetic(atoms)) every 0.1 picosecond;`
//! (spec M6.2 §4.2).
//!
//! A [`LawObserver`] is the engine's own [`Observer`]. It evaluates an observer law's
//! program once per particle of one particle domain, reduces the values, and records the
//! result as one row. It goes through the same trait, contract and runtime checks a
//! plugin does, so the trait is not an interface only an example uses.
//!
//! # Evaluated a batch at a time, reduced in index order
//!
//! Particles are gathered [`LANES`] at a time and run through one
//! [`Program::eval_batch`] pass; the values are then folded in the order they were
//! gathered, which is particle index order. Each lane is bit-identical to a scalar
//! [`Program::eval`], and the fold is the scalar fold, so the result does not depend on
//! the batch width. Nothing here runs on the CPU pool: a reduction in parallel would
//! change the order of the additions and so the bits, and a sampled observer is not the
//! hot loop. The thread count therefore cannot change a reported value.
//!
//! # The reductions
//!
//! `sum` is a left fold starting at `+0.0`; `mean` is that sum divided by the count. `min`
//! and `max` pick equal operands by sign (`-0.0` is below `+0.0`), as the expression
//! interpreters do. A NaN value makes a `min` or `max` NaN, rather than being skipped as
//! `f64::min` would: a reduction that hid a bad particle would hide the fault the run
//! halts on. On an empty set `sum` is `0` and the others are NaN, since an empty `mean`
//! has no value; the compiler warns (`W0321`) when a set can start empty.

use std::borrow::Cow;

use lattice_expr::{Program, LANES, MAX_REGISTERS};
use lattice_ir::{
    Backends, FaultSite, LawFault, ObserveContext, Observations, Observer, ObserverContract,
    ObserverLawUse, ObserverSpec, ParticleStore, Reduction, SideEffects,
};

use crate::domain::ParticleDomain;
use crate::user::{faulty_input, layout};

/// An observer law reduced over a particle domain, as an [`Observer`].
#[derive(Debug)]
pub struct LawObserver {
    contract: ObserverContract,
    law: String,
    set: String,
    reduction: Reduction,
    program: Program,
    params: Vec<f64>,
    row: String,
    unit: String,
    interval: Option<f64>,
    /// The next instant this observer is due at, seconds.
    next_due: f64,
    /// The instant of the latest sample asked about, to notice a run starting over.
    last_time: Option<f64>,
}

impl LawObserver {
    /// The observer for `law`, due every `interval` seconds, or at every sample when
    /// `None`.
    ///
    /// # Errors
    ///
    /// If `interval` is `Some` of a value that is not finite and positive (`0`, a
    /// negative number, NaN, infinity). Such an interval would make the observer sample at
    /// every step while claiming a cadence, and a caller that builds one directly (the SDK)
    /// does not pass through the compiler's `E0405`, so the refusal is here.
    ///
    /// # Panics
    ///
    /// If the program does not have the shape of an observer law (one output, the body
    /// inputs of [`layout`], as many parameters as `law.params`, and a size that fits
    /// [`MAX_REGISTERS`]): a compiler bug.
    pub fn new(law: &ObserverLawUse, interval: Option<f64>) -> Result<LawObserver, String> {
        if let Some(seconds) = interval.filter(|seconds| !(*seconds > 0.0 && seconds.is_finite())) {
            return Err(format!(
                "observer `{}`: the interval must be a positive, finite time in seconds, found {seconds}",
                law.row
            ));
        }
        assert_eq!(law.program.output_count(), 1, "an observer law has one output");
        assert_eq!(law.program.input_count(), layout::BODY_INPUTS as usize, "an observer law reads the body inputs");
        assert_eq!(law.program.param_count(), law.params.len(), "unbound parameters");
        assert!(law.program.fits(), "a program larger than the register file");
        let contract = ObserverContract {
            name: Cow::Owned(law.row.clone()),
            summary: Cow::Owned(format!(
                "{} of the observer law `{}` over the particles of `{}`, in {}",
                law.reduction.name(),
                law.law,
                law.domain,
                law.unit
            )),
            reads: vec![Cow::Owned(law.domain.clone())],
            outputs: vec![(Cow::Owned(law.row.clone()), Cow::Owned(law.unit.clone()))],
            side_effects: SideEffects::None,
            limitations: vec![
                Cow::Borrowed("sampled at observation instants only"),
                Cow::Borrowed(
                    "sum and mean depend on particle index order in their last bits: floating-point addition is not associative, so relabelling the particles can change them",
                ),
                Cow::Borrowed(
                    "the row is a metric, never an invariant: the engine cannot know whether the quantity is conserved",
                ),
                Cow::Borrowed(
                    "mean, min and max of an empty set are NaN, which halts the run",
                ),
            ],
            deterministic: true,
            backends: Backends::CPU,
            differentiable: false,
        };
        Ok(LawObserver {
            contract,
            law: law.law.clone(),
            set: law.domain.clone(),
            reduction: law.reduction,
            program: law.program.clone(),
            params: law.params.clone(),
            row: law.row.clone(),
            unit: law.unit.clone(),
            interval,
            next_due: 0.0,
            last_time: None,
        })
    }

    /// The observer a compiled model's `observe` statement declared, if it is a law form.
    /// The inner `Err` is [`LawObserver::new`]'s refusal of the interval.
    pub fn from_spec(spec: &ObserverSpec) -> Option<Result<LawObserver, String>> {
        spec.law.as_ref().map(|law| LawObserver::new(law, spec.interval))
    }

    /// The row it records.
    pub fn row(&self) -> &str {
        &self.row
    }
}

impl Observer for LawObserver {
    fn contract(&self) -> &ObserverContract {
        &self.contract
    }

    fn observe(&mut self, ctx: &ObserveContext<'_>, out: &mut Observations) {
        // The contract lists the set and the model audit checked it is a particle domain, so
        // not finding it is a bug; the runtime turns the panic into `E0951` and disables
        // this observer, which is better than recording nothing for the rest of the run.
        let Some(domain) = ctx.domain::<ParticleDomain>(&self.set) else {
            panic!(
                "observer `{}` could not reach `{}` as a particle domain: it is missing from the simulation, is not a particle domain, or is outside the observer's `reads`",
                self.row, self.set
            )
        };
        let value = reduce_law(&self.program, &self.params, self.reduction, domain.store());
        out.record_metric(self.row.clone(), value, self.unit.clone());
    }

    /// Due at `0`, then every `interval` by the half-step rule the run's own sample
    /// cadence uses: the first sample at or past the instant, to within half a step (a
    /// sample half a step early counts). The run samples at the smallest interval of all
    /// its observers, so a slower one is due at some samples and not others, and is never
    /// due between them.
    ///
    /// A sample at or before the previous one's instant starts the schedule over: a
    /// second `run` on the same simulation samples its initial state at the instant the
    /// first ended, and that sample must be taken.
    fn due(&mut self, time: f64, dt: f64) -> bool {
        let Some(interval) = self.interval else { return true };
        if self.last_time.is_some_and(|last| time <= last) {
            self.next_due = time;
        }
        self.last_time = Some(time);
        let reach = time + 0.5 * dt;
        if reach < self.next_due {
            return false;
        }
        while self.next_due <= reach {
            self.next_due += interval;
        }
        true
    }

    fn explain_non_finite(&self, ctx: &ObserveContext<'_>, row: &str) -> Option<LawFault> {
        if row != self.row {
            return None;
        }
        let domain = ctx.domain::<ParticleDomain>(&self.set)?;
        let store = domain.store();
        let mut inputs = [0.0; layout::BODY_INPUTS as usize];
        let mut regs = [0.0; MAX_REGISTERS];
        let mut out = [0.0; 1];
        for k in 0..store.len() {
            fill_inputs(&mut inputs, store, k);
            self.program.eval(&inputs, &self.params, &mut regs, &mut out);
            if !out[0].is_finite() {
                let at = self.program.trace(&inputs, &self.params)?;
                let input = faulty_input(at.op, false);
                return Some(LawFault {
                    domain: self.set.clone(),
                    law: self.law.clone(),
                    site: FaultSite::Particle(k),
                    at,
                    input,
                });
            }
        }
        None
    }
}

/// One particle's members, laid out as [`layout`] says.
fn fill_inputs(inputs: &mut [f64], store: &ParticleStore, k: usize) {
    inputs[layout::PX as usize] = store.pos_x()[k];
    inputs[layout::PY as usize] = store.pos_y()[k];
    inputs[layout::VX as usize] = store.vel_x()[k];
    inputs[layout::VY as usize] = store.vel_y()[k];
    inputs[layout::MASS as usize] = store.mass()[k];
    inputs[layout::CHARGE as usize] = store.charge()[k];
}

/// The smaller of two values: `-0.0` below `+0.0`, and NaN if either is NaN.
fn smaller(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() { a } else { b }
    } else {
        a.min(b)
    }
}

/// The larger of two values: `+0.0` above `-0.0`, and NaN if either is NaN.
fn larger(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() { b } else { a }
    } else {
        a.max(b)
    }
}

/// A reduction in progress, fed values in index order.
struct Accumulator {
    reduction: Reduction,
    value: f64,
    count: usize,
}

impl Accumulator {
    fn new(reduction: Reduction) -> Accumulator {
        Accumulator { reduction, value: 0.0, count: 0 }
    }

    fn push(&mut self, v: f64) {
        self.value = match (self.reduction, self.count) {
            (Reduction::Sum | Reduction::Mean, _) => self.value + v,
            (_, 0) => v,
            (Reduction::Min, _) => smaller(self.value, v),
            (Reduction::Max, _) => larger(self.value, v),
        };
        self.count += 1;
    }

    fn finish(self) -> f64 {
        match (self.reduction, self.count) {
            (Reduction::Sum, _) => self.value,
            (_, 0) => f64::NAN,
            (Reduction::Mean, n) => self.value / n as f64,
            (Reduction::Min | Reduction::Max, _) => self.value,
        }
    }
}

/// `reduction` of `program` over every particle of `store`, in index order.
///
/// The program is evaluated [`LANES`] particles at a time. The result is bit-identical to
/// evaluating one particle at a time and folding the same way.
pub fn reduce_law(program: &Program, params: &[f64], reduction: Reduction, store: &ParticleStore) -> f64 {
    let n = store.len();
    let mut acc = Accumulator::new(reduction);
    let mut inputs = [0.0; layout::BODY_INPUTS as usize * LANES];
    let mut regs = [0.0; MAX_REGISTERS * LANES];
    let mut out = [0.0; LANES];
    let (px, py) = (store.pos_x(), store.pos_y());
    let (vx, vy) = (store.vel_x(), store.vel_y());
    let (mass, charge) = (store.mass(), store.charge());
    let mut first = 0;
    while first < n {
        let active = (n - first).min(LANES);
        for lane in 0..active {
            let k = first + lane;
            let mut put = |input: u32, value: f64| inputs[input as usize * LANES + lane] = value;
            put(layout::PX, px[k]);
            put(layout::PY, py[k]);
            put(layout::VX, vx[k]);
            put(layout::VY, vy[k]);
            put(layout::MASS, mass[k]);
            put(layout::CHARGE, charge[k]);
        }
        program.eval_batch(active, &inputs, params, &mut regs, &mut out);
        for &value in &out[..active] {
            acc.push(value);
        }
        first += active;
    }
    acc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_expr::{Builder, Op};
    use lattice_ir::{Domain, ParticleSpec, Pcg32};

    /// `0.5 m |v|² − p0 · sin(x)`: reads positions, velocities and masses, takes both
    /// signs, and has a parameter.
    fn program() -> Program {
        let mut b = Builder::new(layout::BODY_INPUTS, 1);
        let half = b.constant(0.5);
        let m = b.push(Op::Input(layout::MASS));
        let vx = b.push(Op::Input(layout::VX));
        let vy = b.push(Op::Input(layout::VY));
        let hm = b.push(Op::Mul(half, m));
        let vxx = b.push(Op::Mul(vx, vx));
        let vyy = b.push(Op::Mul(vy, vy));
        let v2 = b.push(Op::Add(vxx, vyy));
        let ke = b.push(Op::Mul(hm, v2));
        let x = b.push(Op::Input(layout::PX));
        let sx = b.push(Op::Sin(x));
        let p = b.push(Op::Param(0));
        let ps = b.push(Op::Mul(p, sx));
        let out = b.push(Op::Sub(ke, ps));
        b.finish(vec![out])
    }

    fn store(n: usize) -> ParticleStore {
        let mut store = ParticleStore::with_capacity(n.max(1));
        let mut rng = Pcg32::seed_from_u64(7 + n as u64);
        for _ in 0..n {
            let mut next = || rng.next_f64() * 2.0 - 1.0;
            store
                .spawn(
                    ParticleSpec::at([next() * 3.0, next()])
                        .with_velocity([next(), next()])
                        .with_mass(0.5 + next().abs()),
                )
                .unwrap();
        }
        store
    }

    /// The reference: one particle at a time through `eval`, folded by the plainest code.
    fn scalar(program: &Program, params: &[f64], reduction: Reduction, store: &ParticleStore) -> f64 {
        let mut values = Vec::new();
        for k in 0..store.len() {
            let mut inputs = [0.0; layout::BODY_INPUTS as usize];
            fill_inputs(&mut inputs, store, k);
            values.push(program.eval_to_vec(&inputs, params)[0]);
        }
        match reduction {
            Reduction::Sum => values.iter().fold(0.0, |total, v| total + v),
            Reduction::Mean => values.iter().fold(0.0, |total, v| total + v) / values.len() as f64,
            Reduction::Min => values.iter().copied().reduce(smaller).unwrap_or(f64::NAN),
            Reduction::Max => values.iter().copied().reduce(larger).unwrap_or(f64::NAN),
        }
    }

    /// Spec M6.2 §7: for 1, 7, 8, 9 and 1,000 particles, batched evaluation plus the
    /// index-order reduction equals the scalar loop under `to_bits`, for every reduction.
    #[test]
    fn the_batched_reduction_is_bit_identical_to_the_scalar_loop() {
        let program = program();
        let params = [1.7];
        for n in [1, 7, 8, 9, 1000] {
            let store = store(n);
            for reduction in Reduction::ALL {
                let batched = reduce_law(&program, &params, reduction, &store);
                let reference = scalar(&program, &params, reduction, &store);
                assert!(batched.is_finite());
                assert_eq!(
                    batched.to_bits(),
                    reference.to_bits(),
                    "{} over {n} particles: {batched:e} against {reference:e}",
                    reduction.name()
                );
            }
        }
    }

    #[test]
    fn an_empty_set_sums_to_zero_and_has_no_mean_min_or_max() {
        let program = program();
        let empty = store(0);
        assert_eq!(reduce_law(&program, &[1.0], Reduction::Sum, &empty).to_bits(), 0.0f64.to_bits());
        for reduction in [Reduction::Mean, Reduction::Min, Reduction::Max] {
            assert!(reduce_law(&program, &[1.0], reduction, &empty).is_nan(), "{}", reduction.name());
        }
    }

    #[test]
    fn min_and_max_pick_a_tie_by_sign_and_never_skip_a_nan() {
        assert_eq!(smaller(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(smaller(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(larger(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(larger(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert!(smaller(f64::NAN, 1.0).is_nan() && smaller(1.0, f64::NAN).is_nan());
        assert!(larger(f64::NAN, 1.0).is_nan() && larger(1.0, f64::NAN).is_nan());
        assert_eq!(smaller(2.0, 1.0), 1.0);
        assert_eq!(larger(2.0, 1.0), 2.0);
    }

    fn law_use(reduction: Reduction) -> ObserverLawUse {
        ObserverLawUse {
            law: "wave".to_string(),
            domain: "gas".to_string(),
            reduction,
            program: program(),
            params: vec![1.7],
            unit: "J".to_string(),
            row: format!("gas.wave.{}", reduction.name()),
        }
    }

    /// The generated contract passes the audit, names the set it reads, and is honest
    /// about its limits.
    #[test]
    fn the_generated_contract_is_complete() {
        let observer = LawObserver::new(&law_use(Reduction::Max), None).unwrap();
        let contract = observer.contract();
        assert!(contract.audit(&["gas"]).is_empty());
        assert_eq!(contract.reads, ["gas"]);
        assert_eq!(contract.outputs, [("gas.wave.max".into(), "J".into())]);
        assert_eq!(
            contract.limitations,
            [
                "sampled at observation instants only",
                "sum and mean depend on particle index order in their last bits: floating-point addition is not associative, so relabelling the particles can change them",
                "the row is a metric, never an invariant: the engine cannot know whether the quantity is conserved",
                "mean, min and max of an empty set are NaN, which halts the run",
            ]
        );
        assert!(contract.deterministic && !contract.differentiable);
        assert_eq!(observer.row(), "gas.wave.max");
    }

    #[test]
    fn it_records_one_metric_row_through_the_context() {
        let mut domain = ParticleDomain::new("gas", 9);
        for n in 0..9 {
            let k = f64::from(n);
            domain.spawn(ParticleSpec::at([k, 0.0]).with_velocity([k, 0.0]).with_mass(2.0)).unwrap();
        }
        let domains: Vec<Box<dyn Domain>> = vec![Box::new(domain)];
        let observations = Observations::new();
        let reads = [Cow::Borrowed("gas")];
        let ctx = ObserveContext::new(0.0, 0, &domains, &observations).restricted_to(&reads);
        let mut observer = LawObserver::new(&law_use(Reduction::Sum), None).unwrap();
        let mut out = Observations::new();
        observer.observe(&ctx, &mut out);
        let row = out.get("gas.wave.sum").expect("the row");
        assert_eq!(row.unit, "J");
        assert!(matches!(row.kind, lattice_ir::ObservationKind::Metric));
        let store = domains[0].as_any().downcast_ref::<ParticleDomain>().unwrap().store();
        assert_eq!(row.value.to_bits(), scalar(&program(), &[1.7], Reduction::Sum, store).to_bits());
    }

    /// Not finding its set is a bug the runtime reports as `E0951`, not a quiet gap in the
    /// series.
    #[test]
    #[should_panic(
        expected = "observer `gas.wave.sum` could not reach `gas` as a particle domain: it is missing from the simulation, is not a particle domain, or is outside the observer's `reads`"
    )]
    fn a_missing_set_is_a_panic_not_silence() {
        let domains: Vec<Box<dyn Domain>> = Vec::new();
        let observations = Observations::new();
        let ctx = ObserveContext::new(0.0, 0, &domains, &observations);
        LawObserver::new(&law_use(Reduction::Sum), None).unwrap().observe(&ctx, &mut Observations::new());
    }

    /// A run that starts over at the instant the last one ended, or earlier, is sampled
    /// at once rather than waiting for the old schedule.
    #[test]
    fn a_second_run_is_sampled_at_its_start() {
        let dt = 0.05;
        let mut observer = LawObserver::new(&law_use(Reduction::Sum), Some(0.3)).unwrap();
        assert!(observer.due(0.9, dt));
        assert!(!observer.due(1.0, dt), "the schedule continues from the restart");
        assert!(observer.due(0.9, dt), "the same instant starts over");
        assert!(observer.due(0.0, dt), "and so does an earlier one");
        assert!(!observer.due(0.1, dt));
        assert!(observer.due(0.3, dt));
    }

    /// An interval that is not a positive, finite time is refused rather than quietly
    /// sampling at every step; the SDK reaches `new` without the compiler's `E0405`.
    #[test]
    fn a_bad_interval_is_refused() {
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let refused = LawObserver::new(&law_use(Reduction::Sum), Some(bad));
            let message = refused.expect_err("an interval of that kind must be refused");
            assert!(message.contains("interval") && message.contains("gas.wave.sum"), "{message}");
        }
        assert!(LawObserver::new(&law_use(Reduction::Sum), Some(1e-30)).is_ok());
    }

    /// Due at 0, then by the half-step rule: a 0.3 observer on a 0.1 sample grid is due
    /// every third sample, and never between samples.
    #[test]
    fn it_is_due_at_its_own_cadence() {
        let dt = 0.05;
        let mut slow = LawObserver::new(&law_use(Reduction::Sum), Some(0.3)).unwrap();
        let mut fast = LawObserver::new(&law_use(Reduction::Sum), Some(0.1)).unwrap();
        let mut every = LawObserver::new(&law_use(Reduction::Sum), None).unwrap();
        let (mut slow_times, mut fast_count) = (Vec::new(), 0);
        // The run's samples: the initial state, then every 0.1 s by the same accumulation.
        for k in 0..=9 {
            let time = f64::from(k) * 0.1;
            if slow.due(time, dt) {
                slow_times.push(k);
            }
            fast_count += usize::from(fast.due(time, dt));
            assert!(every.due(time, dt));
        }
        assert_eq!(slow_times, [0, 3, 6, 9]);
        assert_eq!(fast_count, 10);
    }
}
