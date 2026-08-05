//! Deterministic integration of a reaction network.
//!
//! Spec §12.3: *"Deterministic ODE integration for well-mixed, sufficiently populated
//! systems. […] Stiff solver integration through a plugin or established numerical
//! library rather than a weak custom method."*
//!
//! That last clause is a constraint on this file. Chemical kinetics is the canonical
//! stiff problem: a fast equilibrium alongside a slow conversion produces a system
//! whose eigenvalues differ by ten orders of magnitude, and an explicit method on it
//! either takes ten billion steps or explodes. Writing a half-good BDF here would
//! produce something that works on the test cases and fails quietly on a real network.
//!
//! So this module integrates explicitly, **measures its own stiffness**, and says when
//! the answer is out of reach:
//!
//! - Each call sub-cycles at the timestep the network's own Jacobian allows, so a
//!   caller asking for one second of a fast reaction gets it correctly rather than
//!   getting nonsense.
//! - [`MAX_SUBSTEPS`] bounds the work. Hitting it sets [`KineticsReport::stiff`], and
//!   the domain reports that rather than silently spending a minute per step.
//! - [`Kinetics::stiffness_ratio`] is published, so "this needs an implicit solver" is
//!   a number a reader can see rather than a conclusion they have to reach.
//!
//! # Why RK4 rather than something adaptive per step
//!
//! Embedded Runge–Kutta pairs estimate the local error and adapt to it, which is the
//! right tool when accuracy is the constraint. Here it usually is not: the constraint
//! is *stability*, and the stability limit of an explicit method is known in closed
//! form from the Jacobian. Sub-cycling to that limit is cheaper than estimating an
//! error at every step, and it is the same argument the heat module makes for using a
//! von Neumann limit rather than a step controller.

use crate::network::ReactionNetwork;
use crate::rate::mass_action_rate;

/// Most sub-steps one [`Kinetics::advance`] call will take.
///
/// A bound on *work*, not a judgement about the network. Asking for one second of a
/// reaction whose timescale is a fifth of a millisecond genuinely needs fifty thousand
/// sub-steps, and taking them is the correct answer rather than a symptom — a small
/// network's rate evaluation is a handful of multiplies, so that costs milliseconds.
///
/// What the bound protects against is the genuinely stiff case, where the count is not
/// fifty thousand but a billion. Running out sets [`KineticsReport::exhausted`], and
/// [`KineticsReport::stiffness`] says whether stiffness was the reason.
pub const MAX_SUBSTEPS: usize = 100_000;

/// Fraction of the stability limit each sub-step takes.
///
/// RK4's stability region reaches to about `2.8/|λ|` on the negative real axis. Running
/// at the edge means an oscillation that neither grows nor decays, which is not the
/// same as accurate — the distinction the heat module's contract also spells out. Half
/// of it leaves room for the Jacobian changing during the step.
const SAFETY: f64 = 1.4;

/// Default bound on `|λ|·h` per sub-step, where `λ` is the fastest Jacobian eigenvalue.
///
/// **Stability is not accuracy.** A step at the stability limit does not blow up; it
/// simply gets the answer wrong, and RK4's local error there is `(|λ|h)^5/120` — about
/// 1% at `|λ|h = 2`. That is fine for a picture and not for a validation case.
///
/// A caller's `dt` is a coupling or output cadence, not a statement about accuracy, so
/// deciding how finely to sub-cycle inside it is this module's job rather than theirs.
/// At `0.1` the local error is around `8e-8` per sub-step, which is below what the
/// analytic validation cases measure and costs a handful of extra evaluations.
/// [`Kinetics::with_accuracy`] relaxes it for a run that wants speed over digits.
pub const DEFAULT_ACCURACY: f64 = 0.1;

/// How the network is advanced.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Integrator {
    /// Classical fourth-order Runge–Kutta. Four rate evaluations per sub-step.
    #[default]
    Rk4,
    /// Forward Euler. One evaluation, first order.
    ///
    /// Present because it is the reference the convergence-order validation case
    /// measures RK4 against, not because anyone should choose it.
    ExplicitEuler,
}

impl Integrator {
    /// The name for the solver contract.
    pub const fn name(self) -> &'static str {
        match self {
            Integrator::Rk4 => "rk4",
            Integrator::ExplicitEuler => "explicit_euler",
        }
    }

    /// The order of accuracy this method claims.
    pub const fn order(self) -> u32 {
        match self {
            Integrator::Rk4 => 4,
            Integrator::ExplicitEuler => 1,
        }
    }

    /// How far the stability region reaches along the negative real axis.
    const fn stability_reach(self) -> f64 {
        match self {
            Integrator::Rk4 => 2.785,
            Integrator::ExplicitEuler => 2.0,
        }
    }
}

/// What one [`Kinetics::advance`] achieved.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct KineticsReport {
    /// Sub-steps taken.
    pub substeps: usize,
    /// The smallest sub-step the stability limit forced, seconds.
    pub smallest_step: f64,
    /// True when [`MAX_SUBSTEPS`] ran out before the requested interval was covered.
    ///
    /// Not the same as "stiff", which is why it is not called that. A budget can run
    /// out because the network is stiff, or simply because the interval asked for is
    /// enormous compared with the timescale — the two need different fixes, and
    /// [`KineticsReport::stiffness`] is what distinguishes them. The state is left at
    /// however far it got, and [`KineticsReport::covered`] says how far that was.
    pub exhausted: bool,
    /// The ratio of the fastest reaction timescale to the slowest, at the start.
    ///
    /// `None` when nothing was happening. Above roughly 1000 an explicit method is
    /// spending most of its steps resolving a process that reached equilibrium long
    /// ago, and §12.3's "established numerical library" is the right answer — which
    /// this module does not pretend to be.
    pub stiffness: Option<f64>,
    /// How much of the requested interval was actually integrated, seconds.
    pub covered: f64,
    /// Energy released over the interval, J/m². Positive is exothermic.
    ///
    /// Carried through the integrator's own stages as an extra state variable, so it is
    /// as accurate as the concentrations and costs no extra rate evaluations. Sampling
    /// the instantaneous power instead would be wrong by however much the reaction
    /// changed speed during the step — which for a reaction that half-finishes in one
    /// step is a factor of `e`. It is the number the coupling ledger balances.
    pub energy: f64,
    /// Total round-off clamped back to zero, mol/m².
    ///
    /// Mass action cannot drive a concentration negative in exact arithmetic, so
    /// anything here is floating-point noise near an exhausted reactant. It is summed
    /// rather than ignored because a *large* value means the sub-stepping failed, and a
    /// silent clamp is how a conservation check comes to fail for no visible reason.
    pub clamped: f64,
}

/// The network, flattened into the arrays the inner loop indexes.
///
/// Separate from the scratch buffers so a Runge–Kutta stage can read the plan while
/// writing a slope vector. Nesting both in one struct compiles only with a borrow
/// dance, and the dance would obscure what is a genuinely simple split: this half never
/// changes during a step, the other half is nothing but working space.
#[derive(Clone, Debug)]
struct Plan {
    network: ReactionNetwork,
    /// Net stoichiometry per reaction, flattened: `[r * species + i]`.
    stoichiometry: Vec<f64>,
    /// Reactant `(index, coefficient)` pairs, concatenated with per-reaction offsets.
    reactants: Vec<(usize, f64)>,
    reactant_starts: Vec<usize>,
    products: Vec<(usize, f64)>,
    product_starts: Vec<usize>,
}

impl Plan {
    fn build(network: ReactionNetwork) -> Plan {
        let species = network.len();
        let mut stoichiometry = Vec::with_capacity(network.reactions().len() * species);
        let mut reactants = Vec::new();
        let mut reactant_starts = vec![0usize];
        let mut products = Vec::new();
        let mut product_starts = vec![0usize];

        for reaction in network.reactions() {
            stoichiometry.extend(reaction.stoichiometry(species));
            reactants.extend(reaction.reactants.iter().map(|t| (t.species, t.coefficient)));
            reactant_starts.push(reactants.len());
            products.extend(reaction.products.iter().map(|t| (t.species, t.coefficient)));
            product_starts.push(products.len());
        }
        Plan { network, stoichiometry, reactants, reactant_starts, products, product_starts }
    }

    fn reaction_rate(&self, index: usize, concentrations: &[f64], temperature: f64) -> f64 {
        let Some(reaction) = self.network.reactions().get(index) else { return 0.0 };
        mass_action_rate(
            &reaction.rate,
            &self.reactants[self.reactant_starts[index]..self.reactant_starts[index + 1]],
            &self.products[self.product_starts[index]..self.product_starts[index + 1]],
            concentrations,
            temperature,
        )
    }

    /// `dc/dt` into `out`, returning the heat released at the same state, W/m².
    ///
    /// The heat comes back from the same pass because it is built from the same
    /// reaction rates. Energy is then just another state variable — `dE/dt = q` — and
    /// integrating it with the *same* Runge–Kutta stages makes it as accurate as the
    /// concentrations for no extra rate evaluations at all. Computing it separately
    /// would either cost a second pass or settle for a trapezoid, and the energy is
    /// what the coupling ledger balances.
    fn derivative(&self, concentrations: &[f64], temperature: f64, out: &mut [f64]) -> f64 {
        out.fill(0.0);
        let species = self.network.len();
        let mut heat = 0.0;
        for index in 0..self.network.reactions().len() {
            let rate = self.reaction_rate(index, concentrations, temperature);
            if rate == 0.0 {
                continue;
            }
            heat -= self.network.reactions()[index].enthalpy * rate;
            let row = &self.stoichiometry[index * species..(index + 1) * species];
            for (slot, coefficient) in row.iter().enumerate() {
                if *coefficient != 0.0 {
                    out[slot] += coefficient * rate;
                }
            }
        }
        heat
    }

    /// The largest reactant concentration of one reaction, mol/m².
    fn reactant_scale(&self, index: usize, concentrations: &[f64]) -> f64 {
        self.reactants[self.reactant_starts[index]..self.reactant_starts[index + 1]]
            .iter()
            .map(|&(slot, _)| concentrations.get(slot).copied().unwrap_or(0.0).abs())
            .fold(0.0, f64::max)
    }

    /// The pseudo-first-order rate of each direction of one reaction, 1/s.
    ///
    /// Measured *without* the subtraction that gives the net rate. A fast equilibrium
    /// sitting at equilibrium has a net rate of exactly zero and a timescale of
    /// nanoseconds — that combination is what stiffness *is*, and a measure built on
    /// the net rate reports the stiffest system in the module as perfectly tame.
    fn one_way_rates(&self, index: usize, concentrations: &[f64], temperature: f64) -> (f64, f64) {
        let Some(reaction) = self.network.reactions().get(index) else { return (0.0, 0.0) };
        let product_of = |terms: &[(usize, f64)]| -> f64 {
            terms
                .iter()
                .map(|&(slot, coefficient)| {
                    let amount = concentrations.get(slot).copied().unwrap_or(0.0).max(0.0);
                    if (coefficient - 1.0).abs() < 1e-12 { amount } else { amount.powf(coefficient) }
                })
                .product()
        };
        let forward = reaction.rate.forward_at(temperature)
            * product_of(&self.reactants[self.reactant_starts[index]..self.reactant_starts[index + 1]]);
        let reverse = reaction.rate.reverse_at(temperature)
            * product_of(&self.products[self.product_starts[index]..self.product_starts[index + 1]]);
        (forward.abs(), reverse.abs())
    }
}

/// Working space for a Runge–Kutta step, allocated once.
#[derive(Clone, Debug, Default)]
struct Scratch {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    work: Vec<f64>,
    /// Buffers for the finite-difference Jacobian, so `stable_step` does not allocate
    /// once per cell per step in a spatial run.
    base: Vec<f64>,
    perturbed: Vec<f64>,
    probe: Vec<f64>,
    rows: Vec<f64>,
}

impl Scratch {
    fn sized(species: usize) -> Scratch {
        Scratch {
            k1: vec![0.0; species],
            k2: vec![0.0; species],
            k3: vec![0.0; species],
            k4: vec![0.0; species],
            work: vec![0.0; species],
            base: vec![0.0; species],
            perturbed: vec![0.0; species],
            probe: vec![0.0; species],
            rows: vec![0.0; species],
        }
    }
}

/// A reaction network and the buffers needed to integrate it without allocating.
#[derive(Clone, Debug)]
pub struct Kinetics {
    plan: Plan,
    integrator: Integrator,
    accuracy: f64,
    scratch: Scratch,
}

impl Kinetics {
    /// Prepare a network for integration.
    ///
    /// Flattens the stoichiometry and term lists once, so the inner loop indexes arrays
    /// rather than walking nested `Vec`s — the same reason the particle module stores
    /// structure-of-arrays.
    pub fn new(network: ReactionNetwork) -> Kinetics {
        let species = network.len();
        Kinetics {
            plan: Plan::build(network),
            integrator: Integrator::default(),
            accuracy: DEFAULT_ACCURACY,
            scratch: Scratch::sized(species),
        }
    }

    /// Choose the integrator.
    pub fn with_integrator(mut self, integrator: Integrator) -> Kinetics {
        self.integrator = integrator;
        self
    }

    /// Tighten the accuracy bound until the remaining error is round-off rather than
    /// budget.
    ///
    /// For validation cases, where the question is whether the *chemistry* is right and
    /// an under-resolved integration would be a confounding variable. Costs about five
    /// times the sub-steps of the default.
    pub fn accurate(self) -> Kinetics {
        self.with_accuracy(0.02)
    }

    /// Bound `|λ|·h` per sub-step. See [`DEFAULT_ACCURACY`].
    ///
    /// Larger is faster and less accurate. Values above the integrator's stability
    /// reach have no effect, because the stability limit binds first.
    pub fn with_accuracy(mut self, accuracy: f64) -> Kinetics {
        self.accuracy = accuracy.max(f64::MIN_POSITIVE);
        self
    }

    /// The network being integrated.
    pub fn network(&self) -> &ReactionNetwork {
        &self.plan.network
    }

    /// Which integrator is in use.
    pub fn integrator(&self) -> Integrator {
        self.integrator
    }

    /// How many species the concentration vector must hold.
    pub fn len(&self) -> usize {
        self.plan.network.len()
    }

    /// True when there is nothing to integrate.
    pub fn is_empty(&self) -> bool {
        self.plan.network.reactions().is_empty()
    }

    /// The rate of reaction `index`, mol/(m²·s).
    pub fn reaction_rate(&self, index: usize, concentrations: &[f64], temperature: f64) -> f64 {
        self.plan.reaction_rate(index, concentrations, temperature)
    }

    /// `dc/dt` for every species, mol/(m²·s).
    pub fn derivative(&self, concentrations: &[f64], temperature: f64, out: &mut [f64]) {
        self.plan.derivative(concentrations, temperature, out);
    }

    /// `dc/dt` and the heat released at the same state, in one pass.
    pub fn derivative_and_heat(
        &self,
        concentrations: &[f64],
        temperature: f64,
        out: &mut [f64],
    ) -> f64 {
        self.plan.derivative(concentrations, temperature, out)
    }

    /// Heat released by the network, W/m². Positive warms the surroundings.
    ///
    /// `−Σ ΔH_r · rate_r`. The sign flip is because chemistry writes an exothermic
    /// enthalpy as negative — the reaction loses energy, the surroundings gain it — and
    /// a heat *source* is positive. Getting this backwards makes an exothermic reaction
    /// refrigerate, which is the sort of thing that looks like a coupling bug for a
    /// long time.
    pub fn heat_release(&self, concentrations: &[f64], temperature: f64) -> f64 {
        (0..self.plan.network.reactions().len())
            .map(|index| {
                let enthalpy = self.plan.network.reactions()[index].enthalpy;
                -enthalpy * self.plan.reaction_rate(index, concentrations, temperature)
            })
            .sum()
    }

    /// The largest timestep this network is stable at, seconds.
    ///
    /// From the Jacobian's spectral radius, bounded by Gershgorin's theorem — the
    /// largest absolute row sum. Computed by finite differences rather than
    /// analytically, so it is correct for any rate law rather than only for the ones
    /// whose derivative was worked out by hand.
    ///
    /// Infinite when nothing is happening: a network at equilibrium, or with no
    /// reactions, has no timescale of its own.
    pub fn stable_step(&mut self, concentrations: &[f64], temperature: f64) -> f64 {
        let radius = self.spectral_radius(concentrations, temperature);
        if radius <= 0.0 || !radius.is_finite() {
            return f64::INFINITY;
        }
        self.integrator.stability_reach() / (SAFETY * radius)
    }

    /// The sub-step [`Kinetics::advance`] will actually take — the smaller of the
    /// stability limit and the accuracy bound.
    pub fn substep(&mut self, concentrations: &[f64], temperature: f64) -> f64 {
        let radius = self.spectral_radius(concentrations, temperature);
        if radius <= 0.0 || !radius.is_finite() {
            return f64::INFINITY;
        }
        let stability = self.integrator.stability_reach() / (SAFETY * radius);
        let accuracy = self.accuracy / radius;
        stability.min(accuracy)
    }

    /// The ratio of the fastest timescale to the slowest — how stiff this is.
    ///
    /// Above roughly 1000, an explicit method is spending most of its steps resolving a
    /// process that reached equilibrium long ago, and §12.3's "established numerical
    /// library" is the right answer. Published rather than acted on, because deciding
    /// for the user which solver they need is not this module's call.
    ///
    /// `None` when nothing is happening.
    pub fn stiffness_ratio(&self, concentrations: &[f64], temperature: f64) -> Option<f64> {
        let mut fastest: f64 = 0.0;
        let mut slowest = f64::INFINITY;
        for index in 0..self.plan.network.reactions().len() {
            // Each *direction* separately. A fast equilibrium has a net rate of zero
            // and a timescale of nanoseconds, and that combination is precisely what
            // makes a network stiff — measuring the net rate would report the stiffest
            // system here as perfectly tame.
            let (forward, reverse) = self.plan.one_way_rates(index, concentrations, temperature);
            let scale = self.plan.reactant_scale(index, concentrations).max(f64::MIN_POSITIVE);
            // A rate is mol/(m²·s); dividing by a concentration gives a 1/s timescale.
            let inverse_time = forward.max(reverse) / scale;
            if inverse_time > 0.0 {
                fastest = fastest.max(inverse_time);
                slowest = slowest.min(inverse_time);
            }
        }
        (fastest > 0.0 && slowest.is_finite() && slowest > 0.0).then(|| fastest / slowest)
    }

    /// Advance `concentrations` by `dt`, sub-cycling at the stability limit.
    pub fn advance(
        &mut self,
        concentrations: &mut [f64],
        temperature: f64,
        dt: f64,
    ) -> KineticsReport {
        let mut report = KineticsReport { smallest_step: dt, ..KineticsReport::default() };
        if dt <= 0.0 || self.is_empty() || concentrations.len() != self.plan.network.len() {
            report.covered = dt.max(0.0);
            return report;
        }
        report.stiffness = self.stiffness_ratio(concentrations, temperature);

        let mut remaining = dt;
        while remaining > 0.0 {
            if report.substeps >= MAX_SUBSTEPS {
                report.exhausted = true;
                break;
            }
            let limit = self.substep(concentrations, temperature);
            let step = if limit.is_finite() { remaining.min(limit) } else { remaining };
            // A limit that has collapsed to nothing would loop forever making no
            // progress. Reporting stiffness is the honest end to that.
            if step.is_nan() || step <= 0.0 || step < dt * 1e-12 {
                report.exhausted = true;
                break;
            }

            report.energy += match self.integrator {
                Integrator::Rk4 => self.rk4(concentrations, temperature, step),
                Integrator::ExplicitEuler => self.euler(concentrations, temperature, step),
            };
            report.clamped += clamp_round_off(concentrations);

            report.smallest_step = report.smallest_step.min(step);
            report.covered += step;
            remaining -= step;
            report.substeps += 1;
        }
        report
    }

    /// Gershgorin bound on the Jacobian's spectral radius, 1/s.
    fn spectral_radius(&mut self, concentrations: &[f64], temperature: f64) -> f64 {
        let species = self.plan.network.len();
        if species == 0 {
            return 0.0;
        }
        let scratch = &mut self.scratch;
        scratch.probe.copy_from_slice(concentrations);
        scratch.rows.fill(0.0);
        self.plan.derivative(concentrations, temperature, &mut scratch.base);

        // Row sums of |∂f_i/∂c_j|, accumulated column by column.
        for (column, &centre) in concentrations.iter().enumerate() {
            // A step relative to the concentration, with an absolute floor so an
            // exhausted species still contributes its derivative rather than dividing
            // by zero.
            let delta = centre.abs().max(1e-12) * 1e-7;
            scratch.probe[column] = centre + delta;
            self.plan.derivative(&scratch.probe, temperature, &mut scratch.perturbed);
            scratch.probe[column] = centre;

            for row in 0..species {
                scratch.rows[row] +=
                    ((scratch.perturbed[row] - scratch.base[row]) / delta).abs();
            }
        }
        scratch.rows.iter().copied().fold(0.0, f64::max)
    }

    /// One forward-Euler step. Returns the energy released, J/m².
    fn euler(&mut self, concentrations: &mut [f64], temperature: f64, dt: f64) -> f64 {
        let scratch = &mut self.scratch;
        let heat = self.plan.derivative(concentrations, temperature, &mut scratch.k1);
        for (value, slope) in concentrations.iter_mut().zip(&scratch.k1) {
            *value += dt * slope;
        }
        heat * dt
    }

    /// One classical RK4 step. Returns the energy released, J/m².
    ///
    /// The energy is carried through the same four stages as the concentrations, with
    /// the same weights, so it is fourth-order accurate too. It costs nothing: the heat
    /// rate falls out of the rate evaluations the stages already perform.
    fn rk4(&mut self, concentrations: &mut [f64], temperature: f64, dt: f64) -> f64 {
        let half = 0.5 * dt;
        let scratch = &mut self.scratch;

        let q1 = self.plan.derivative(concentrations, temperature, &mut scratch.k1);

        step_into(&mut scratch.work, concentrations, &scratch.k1, half);
        let q2 = self.plan.derivative(&scratch.work, temperature, &mut scratch.k2);

        step_into(&mut scratch.work, concentrations, &scratch.k2, half);
        let q3 = self.plan.derivative(&scratch.work, temperature, &mut scratch.k3);

        step_into(&mut scratch.work, concentrations, &scratch.k3, dt);
        let q4 = self.plan.derivative(&scratch.work, temperature, &mut scratch.k4);

        for (index, value) in concentrations.iter_mut().enumerate() {
            *value += dt / 6.0
                * (scratch.k1[index]
                    + 2.0 * scratch.k2[index]
                    + 2.0 * scratch.k3[index]
                    + scratch.k4[index]);
        }
        dt / 6.0 * (q1 + 2.0 * q2 + 2.0 * q3 + q4)
    }

}

/// `out = base + slope * dt`.
fn step_into(out: &mut [f64], base: &[f64], slope: &[f64], dt: f64) {
    for (index, value) in out.iter_mut().enumerate() {
        *value = base[index] + dt * slope[index];
    }
}

/// Clamp negative concentrations that are round-off, returning how much was clamped.
///
/// Only the round-off case: a value more negative than this is a real overshoot, and
/// leaving it visible is what lets the sub-stepping and the caller's monitors notice.
/// Silently clamping a large negative is how a conservation check comes to fail for no
/// reason anyone can find.
fn clamp_round_off(concentrations: &mut [f64]) -> f64 {
    let scale = concentrations.iter().fold(0.0f64, |best, c| best.max(c.abs()));
    let floor = -1e-12 * scale.max(f64::MIN_POSITIVE);
    let mut clamped = 0.0;
    for value in concentrations.iter_mut() {
        if *value < 0.0 && *value >= floor {
            clamped += -*value;
            *value = 0.0;
        }
    }
    clamped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{Reaction, Term};
    use crate::rate::RateLaw;
    use crate::species::Species;

    /// `A -> B`, first order, so `[A](t) = [A]₀ e^(−kt)`.
    fn decay(k: f64) -> Kinetics {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(Reaction::new(
            "decay",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(k),
        ));
        Kinetics::new(network)
    }

    /// `A <-> B`, which settles at `[B]/[A] = k_f/k_r`.
    fn equilibrium(forward: f64, reverse: f64) -> Kinetics {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(Reaction::new(
            "isomerize",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::reversible(forward, reverse),
        ));
        Kinetics::new(network)
    }

    /// §19.2's first-order network against its analytic solution.
    #[test]
    fn first_order_decay_matches_the_exponential() {
        let k = 0.7;
        let mut kinetics = decay(k).accurate();
        let mut c = [1.0, 0.0];
        let report = kinetics.advance(&mut c, 300.0, 3.0);

        let expected = (-k * 3.0f64).exp();
        assert!((c[0] - expected).abs() < 1e-9, "{} against {expected}", c[0]);
        assert!((c[1] - (1.0 - expected)).abs() < 1e-9, "what left A arrived at B");
        assert!(!report.exhausted);
        assert!((report.covered - 3.0).abs() < 1e-12);

        // At the interactive default the answer is a few digits worse and many times
        // cheaper, which is the trade `with_accuracy` exists to expose.
        let mut quick = decay(k);
        let mut c = [1.0, 0.0];
        let cheap = quick.advance(&mut c, 300.0, 3.0);
        assert!((c[0] - expected).abs() < 1e-6, "{} against {expected}", c[0]);
        assert!(cheap.substeps * 4 < report.substeps, "{} against {}", cheap.substeps, report.substeps);
    }

    /// §19.2's reversible network. The equilibrium is set by the ratio of the
    /// constants, not by either one alone.
    #[test]
    fn a_reversible_reaction_settles_at_its_equilibrium_constant() {
        let mut kinetics = equilibrium(3.0, 1.0).accurate();
        let mut c = [1.0, 0.0];
        kinetics.advance(&mut c, 300.0, 20.0);

        // K = 3, so [B]/[A] = 3 and the total is 1: [A] = 0.25, [B] = 0.75.
        assert!((c[0] - 0.25).abs() < 1e-9, "{c:?}");
        assert!((c[1] - 0.75).abs() < 1e-9, "{c:?}");

        // And the approach is exponential at rate (k_f + k_r), which is the other half
        // of the closed form and the part a solver can get wrong while still landing on
        // the right equilibrium.
        let mut c = [1.0, 0.0];
        kinetics.advance(&mut c, 300.0, 0.1);
        let relaxation = (-(3.0 + 1.0) * 0.1f64).exp();
        let expected = 0.25 + 0.75 * relaxation;
        assert!((c[0] - expected).abs() < 1e-9, "{} against {expected}", c[0]);
    }

    /// A balanced network must conserve every element exactly, which is what the
    /// flagship demo's ledger check rests on.
    #[test]
    fn a_balanced_network_conserves_mass_and_atoms() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("H2").with_formula("H2").unwrap());
        network.add_species(Species::new("O2").with_formula("O2").unwrap());
        network.add_species(Species::new("H2O").with_formula("H2O").unwrap());
        network.add_reaction(Reaction::new(
            "combustion",
            vec![Term::new(0, 2.0), Term::new(1, 1.0)],
            vec![Term::new(2, 2.0)],
            RateLaw::mass_action(5.0),
        ));
        let mut kinetics = Kinetics::new(network);

        let mut c = [2.0, 1.5, 0.0];
        let mass_before = kinetics.network().total_mass(&c);
        let h_before = kinetics.network().total_element("H", &c);
        let o_before = kinetics.network().total_element("O", &c);

        kinetics.advance(&mut c, 300.0, 5.0);

        assert!((kinetics.network().total_mass(&c) - mass_before).abs() / mass_before < 1e-12);
        assert!((kinetics.network().total_element("H", &c) - h_before).abs() < 1e-12);
        assert!((kinetics.network().total_element("O", &c) - o_before).abs() < 1e-12);

        // Hydrogen is the limiting reactant, but it approaches zero rather than
        // reaching it: the rate goes as [H2]², so the last of it takes forever. What
        // must hold exactly is the *ratio* — every 2 mol of H2 consumed takes 1 mol of
        // O2 with it.
        assert!(c[0] < 0.1, "most of the hydrogen is gone: {c:?}");
        let consumed_h2 = 2.0 - c[0];
        let consumed_o2 = 1.5 - c[1];
        assert!(
            (consumed_h2 / consumed_o2 - 2.0).abs() < 1e-9,
            "2 H2 per O2: {consumed_h2} to {consumed_o2}"
        );
        assert!((c[2] - consumed_h2).abs() < 1e-9, "and every H2 became an H2O: {c:?}");
    }

    /// RK4 must actually be fourth order, or the name is a claim nobody checked.
    #[test]
    fn the_integrators_converge_at_the_order_they_declare() {
        for (integrator, expected) in
            [(Integrator::ExplicitEuler, 1.0), (Integrator::Rk4, 4.0)]
        {
            // A single fixed step per measurement, so sub-cycling cannot hide the
            // method's own error. `k` is small enough that one step is stable.
            let k = 0.3;
            let error_at = |steps: usize| {
                let mut kinetics = decay(k).with_integrator(integrator);
                let mut c = [1.0, 0.0];
                let dt = 1.0 / steps as f64;
                for _ in 0..steps {
                    match integrator {
                        Integrator::Rk4 => kinetics.rk4(&mut c, 300.0, dt),
                        Integrator::ExplicitEuler => kinetics.euler(&mut c, 300.0, dt),
                    };
                }
                (c[0] - (-k).exp()).abs()
            };

            let coarse = error_at(8);
            let fine = error_at(16);
            let order = (coarse / fine).log2();
            assert!(
                (order - expected).abs() < 0.3,
                "{} measured order {order}, declared {expected}",
                integrator.name()
            );
        }
    }

    /// The stability limit has to be a real bound, not a decoration: stepping past it
    /// with a fixed-step method must actually diverge.
    #[test]
    fn the_stability_limit_is_where_a_fixed_step_starts_to_fail() {
        let mut kinetics = decay(1000.0);
        let c = [1.0, 0.0];
        let limit = kinetics.stable_step(&c, 300.0);
        assert!(limit.is_finite() && limit > 0.0, "{limit}");

        // Comfortably inside the limit: bounded and correct.
        let mut inside = c;
        for _ in 0..200 {
            let _ = kinetics.rk4(&mut inside, 300.0, limit * 0.5);
        }
        assert!(inside[0] >= 0.0 && inside[0] <= 1.0, "{inside:?}");

        // Well past it, the answer leaves the physical range. It does not run away to
        // infinity, because `mass_action_rate` floors a negative concentration at zero
        // — which stops a NaN spreading and is exactly why "did it diverge?" is the
        // wrong question to ask. "Is the answer still a concentration?" is the right
        // one, and past the limit it is not.
        let mut outside = c;
        for _ in 0..200 {
            let _ = kinetics.rk4(&mut outside, 300.0, limit * 4.0);
        }
        assert!(
            outside[0] < -0.1 || outside[0] > 1.1,
            "past the limit the answer must leave [0, 1], got {outside:?}"
        );
        assert!(
            (outside[0] + outside[1] - 1.0).abs() < 1e-9,
            "and the error is in the split, not in the total: {outside:?}"
        );
    }

    /// And `advance` respects it automatically, so a caller asking for a whole second
    /// of a fast reaction gets the right answer rather than an explosion.
    #[test]
    fn advancing_a_fast_reaction_sub_cycles_rather_than_exploding() {
        let k = 5000.0;
        let mut kinetics = decay(k);
        let mut c = [1.0, 0.0];
        let report = kinetics.advance(&mut c, 300.0, 1.0);

        assert!(report.substeps > 100, "it should have sub-cycled: {report:?}");
        assert!(!report.exhausted, "fast is not the same as stiff: {report:?}");
        assert!(report.stiffness.unwrap() < 2.0, "one reaction, one timescale");
        // Fully converted after 5000 e-foldings.
        assert!(c[0] < 1e-12 && (c[1] - 1.0).abs() < 1e-9, "{c:?}");
    }

    /// §12.3 defers stiff systems to an established library rather than a weak custom
    /// method. What this module owes is to *say* when it has met one.
    #[test]
    fn a_stiff_network_is_reported_rather_than_silently_ground_through() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_species(Species::new("C").with_formula("C").unwrap());
        // A fast equilibrium beside a slow conversion: the canonical stiff pair.
        network.add_reaction(Reaction::new(
            "fast",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::reversible(1e8, 1e8),
        ));
        network.add_reaction(Reaction::new(
            "slow",
            vec![Term::new(1, 1.0)],
            vec![Term::new(2, 1.0)],
            RateLaw::mass_action(1e-2),
        ));
        let mut kinetics = Kinetics::new(network);

        let c = [1.0, 1.0, 0.0];
        let ratio = kinetics.stiffness_ratio(&c, 300.0).unwrap();
        assert!(ratio > 1e6, "ten orders of magnitude apart: {ratio}");

        // Asking for a second of it exhausts the substep bound and says so, rather
        // than taking a billion steps — and reports the stiffness that explains why.
        let mut state = c;
        let report = kinetics.advance(&mut state, 300.0, 1.0);
        assert!(report.exhausted, "{report:?}");
        assert!(report.stiffness.unwrap() > 1e6, "and says why: {report:?}");
        assert_eq!(report.substeps, MAX_SUBSTEPS);
        assert!(report.covered < 1.0, "it did not finish, and says how far it got");
        assert!(state.iter().all(|c| c.is_finite()), "what it did do is still sane: {state:?}");
    }

    /// A network at equilibrium has a net rate of zero and a timescale all the same:
    /// perturb it and it relaxes at `k_f + k_r`. A stability limit read off the *net*
    /// rate would be infinite, and a step taken on that basis would be unstable.
    #[test]
    fn equilibrium_does_not_mean_the_stability_limit_disappears() {
        let mut kinetics = equilibrium(3.0, 1.0);
        let settled = [0.25, 0.75];

        let limit = kinetics.stable_step(&settled, 300.0);
        assert!(limit.is_finite(), "an equilibrium still relaxes");
        // The Jacobian's spectral radius here is k_f + k_r = 4.
        let expected = Integrator::Rk4.stability_reach() / (SAFETY * 4.0);
        assert!((limit - expected).abs() / expected < 1e-6, "{limit} against {expected}");

        // A network with nothing in it is the case that genuinely has no timescale.
        let mut nothing = Kinetics::new(ReactionNetwork::new());
        assert_eq!(nothing.stable_step(&[], 300.0), f64::INFINITY);
        assert_eq!(nothing.stiffness_ratio(&[], 300.0), None);
    }

    /// Sub-stepping is bounded by accuracy as well as stability, because a caller's
    /// `dt` is an output cadence rather than a statement about how many digits they
    /// want.
    #[test]
    fn the_substep_is_bounded_by_accuracy_as_well_as_stability() {
        let mut accurate = decay(4.0);
        let mut fast = decay(4.0).with_accuracy(100.0);
        let c = [1.0, 0.0];

        let stability = accurate.stable_step(&c, 300.0);
        assert!(accurate.substep(&c, 300.0) < stability, "accuracy binds first by default");
        assert!(
            (fast.substep(&c, 300.0) - stability).abs() < 1e-12,
            "a relaxed accuracy target leaves stability in charge"
        );

        // And it shows in the answer: the same interval, integrated more finely, is
        // closer to the closed form.
        let error_with = |kinetics: &mut Kinetics| {
            let mut state = [1.0, 0.0];
            kinetics.advance(&mut state, 300.0, 1.0);
            (state[0] - (-4.0f64).exp()).abs()
        };
        assert!(error_with(&mut accurate) < error_with(&mut fast) * 0.1);
    }

    /// Heat release must warm for an exothermic reaction. Getting the sign backwards
    /// makes it refrigerate, which looks like a coupling bug for a long time.
    #[test]
    fn an_exothermic_reaction_releases_heat() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(
            Reaction::new(
                "exothermic",
                vec![Term::new(0, 1.0)],
                vec![Term::new(1, 1.0)],
                RateLaw::mass_action(2.0),
            )
            // Chemistry writes an exothermic enthalpy as negative.
            .with_enthalpy(-50_000.0),
        );
        let kinetics = Kinetics::new(network.clone());

        // rate = 2 x 1 = 2 mol/(m^2 s), so 2 x 50 kJ/mol = 100 kW/m^2 of heating.
        let heat = kinetics.heat_release(&[1.0, 0.0], 300.0);
        assert!((heat - 100_000.0).abs() < 1e-6, "{heat}");

        // And an endothermic one cools.
        let mut cold = network;
        cold.reactions_mut()[0].enthalpy = 50_000.0;
        assert!(Kinetics::new(cold).heat_release(&[1.0, 0.0], 300.0) < 0.0);
    }

    /// Temperature reaching the rate is what the flagship demo's feedback loop is.
    /// The energy released has to be the integral of the power, not a sample of it.
    /// For a reaction that half-finishes during the step those differ by that factor,
    /// and it is the number the coupling ledger balances.
    #[test]
    fn the_energy_released_is_integrated_rather_than_sampled() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(
            Reaction::new(
                "burn",
                vec![Term::new(0, 1.0)],
                vec![Term::new(1, 1.0)],
                RateLaw::mass_action(1.0),
            )
            .with_enthalpy(-100_000.0),
        );
        let mut kinetics = Kinetics::new(network).accurate();

        // One full e-folding: the reaction slows by a factor of e across it, so
        // sampling the power at the start would overstate the energy by 58%.
        let mut c = [1.0, 0.0];
        let start_power = kinetics.heat_release(&c, 300.0);
        let report = kinetics.advance(&mut c, 300.0, 1.0);

        // Energy = extent x |enthalpy|, and the extent is what actually converted.
        let expected = c[1] * 100_000.0;
        assert!(
            (report.energy - expected).abs() / expected < 1e-6,
            "{} against {expected}",
            report.energy
        );
        assert!(
            start_power * 1.0 > report.energy * 1.5,
            "sampling the initial power would have been badly wrong: {start_power} vs {}",
            report.energy
        );

        // An endothermic reaction takes energy in.
        let mut cold = kinetics.clone();
        cold.plan.network.reactions_mut()[0].enthalpy = 100_000.0;
        let mut c = [1.0, 0.0];
        assert!(cold.advance(&mut c, 300.0, 1.0).energy < 0.0);
    }

    #[test]
    fn temperature_changes_the_rate_through_the_network() {
        let mut network = ReactionNetwork::new();
        network.add_species(Species::new("A").with_formula("C").unwrap());
        network.add_species(Species::new("B").with_formula("C").unwrap());
        network.add_reaction(Reaction::new(
            "arrhenius",
            vec![Term::new(0, 1.0)],
            vec![Term::new(1, 1.0)],
            RateLaw::mass_action(1e6).with_activation_energy(50_000.0),
        ));
        let mut kinetics = Kinetics::new(network);

        let converted = |temperature: f64| {
            let mut c = [1.0, 0.0];
            let mut k = kinetics.clone();
            k.advance(&mut c, temperature, 1e-3);
            c[1]
        };
        let cold = converted(300.0);
        let hot = converted(350.0);
        assert!(hot > cold * 5.0, "50 K should be a large factor: {cold} -> {hot}");
        // Sanity: it is the same reaction, so both convert something.
        assert!(cold > 0.0 && hot < 1.0 + 1e-12);
        let _ = kinetics.advance(&mut [1.0, 0.0], 300.0, 0.0);
    }

    #[test]
    fn an_empty_or_degenerate_call_reports_rather_than_panicking() {
        let mut kinetics = Kinetics::new(ReactionNetwork::new());
        assert!(kinetics.is_empty());
        let report = kinetics.advance(&mut [], 300.0, 1.0);
        assert_eq!(report.substeps, 0);
        assert!(!report.exhausted);

        // A zero or negative interval does nothing and says nothing is wrong.
        let mut decay = decay(1.0);
        let mut c = [1.0, 0.0];
        assert_eq!(decay.advance(&mut c, 300.0, 0.0).substeps, 0);
        assert_eq!(c, [1.0, 0.0]);

        // A concentration vector of the wrong length is refused rather than indexed.
        let mut wrong = [1.0];
        decay.advance(&mut wrong, 300.0, 1.0);
        assert_eq!(wrong, [1.0]);
    }

    #[test]
    fn round_off_below_zero_is_clamped_and_counted() {
        let mut values = [1.0, -1e-18, -0.5];
        let clamped = clamp_round_off(&mut values);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0, "round-off is clamped");
        assert_eq!(values[2], -0.5, "a real overshoot is left visible");
        assert!((clamped - 1e-18).abs() < 1e-30);
    }
}
