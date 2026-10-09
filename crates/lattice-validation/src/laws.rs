//! Validation cases for user-defined laws (spec §8.3, M6.1b).
//!
//! A law is written in the language, checked, lowered to a program and bound exactly
//! as a model's `force:` line would bind it; the cases then hold it against what it
//! should reproduce. Two are against the built-in it imitates, one is against the
//! energy it was derived from, and one is against an analytic solution.

use lattice_compiler::user_force::law_from_source;
use lattice_domain_particle::{
    analysis, BoundaryBox, Integrator, LennardJones, ParticleDomain, ParticleSpec, UserLaw, BOLTZMANN,
};
use lattice_ir::{Arena, Domain, Pcg32, StepContext};

use crate::{Case, Level, Outcome};

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "user_lj_force_reproduces_the_builtin_bitwise",
        domain: "laws",
        level: Level::CrossScheme,
        claim: "a Lennard-Jones force written in the language in the built-in's operation order gives a bit-identical trajectory",
        run: user_force_bitwise,
    },
    Case {
        name: "user_lj_potential_matches_the_builtin_force",
        domain: "laws",
        level: Level::CrossScheme,
        claim: "the force derived from a natural-form Lennard-Jones potential matches the built-in's to the rounding of the two formulas",
        run: user_potential_matches,
    },
    Case {
        name: "derived_force_is_the_energy_gradient",
        domain: "laws",
        level: Level::Manufactured,
        claim: "a potential's compiled derivative agrees with a central difference of its energy within the difference's own error bound",
        run: derivative_is_the_gradient,
    },
    Case {
        name: "damped_spring_decays_at_the_analytic_rate",
        domain: "laws",
        level: Level::Analytic,
        claim: "spec 8.3's damped spring (with its damping sign corrected) decays at c/2mu: the Richardson-extrapolated rate agrees to second order",
        run: damped_spring,
    },
];

/// Argon in SI, written so the language parses the same numbers Rust does.
const EPSILON: f64 = 1.654e-21;
const SIGMA: f64 = 3.405e-10;
const CUTOFF: f64 = 8.5125e-10;
const MASS: f64 = 6.634e-26;

/// Bind the law `law_source` declares, as the use site `use_site` would.
pub(crate) fn user_law(law_source: &str, use_site: &str) -> UserLaw {
    law_from_source(law_source, use_site).unwrap_or_else(|rendered| panic!("the validation law compiles:\n{rendered}"))
}

/// A harmonic trap written as a potential on one particle, with a vector parameter:
/// the user law the thread-count case adds to its cloud.
pub(crate) fn user_trap() -> UserLaw {
    user_law(
        "potential trap(a: particle) -> joule {
            param center: vec2<meter>;
            param stiffness: newton / meter;
            let offset = a.position - center;
            return 0.5 * stiffness * dot(offset, offset);
        }",
        "trap(center=[0.5 meter, 0.5 meter], stiffness=1.5 newton/meter)",
    )
}

const LJ_FORCE: &str = "force lj(a: particle, b: particle) -> vec2<newton> {
    param epsilon: joule;
    param sigma: meter;
    let d = minimum_image(b.position - a.position);
    let inv_r2 = 1 / dot(d, d);
    let s6 = (sigma * sigma * inv_r2)^3;
    let s12 = s6 * s6;
    return -(24 * epsilon * inv_r2 * (2 * s12 - s6) * d);
}";

const LJ_POTENTIAL: &str = "potential lj(a: particle, b: particle) -> joule {
    param epsilon: joule;
    param sigma: meter;
    let s6 = (sigma / distance(a, b))^6;
    return 4 * epsilon * (s6^2 - s6);
}";

fn lj_use_site() -> String {
    format!("lj(epsilon={EPSILON:e} joule, sigma={SIGMA:e} meter, cutoff={CUTOFF:e} meter)")
}

/// A 10×10 argon fluid at T* ≈ 0.6, the same for both arms of a comparison.
fn fluid(name: &str, law: impl lattice_domain_particle::ForceLaw + 'static) -> ParticleDomain {
    let side = 10;
    let spacing = 1.12 * SIGMA;
    let size = side as f64 * spacing;
    let mut domain = ParticleDomain::new(name, side * side)
        .with_integrator(Integrator::VelocityVerlet)
        .with_bounds(BoundaryBox::periodic([0.0, 0.0], [size, size]))
        .with_force(law)
        .with_seed(11);
    for index in 0..side * side {
        let (i, j) = (index % side, index / side);
        domain
            .spawn(ParticleSpec::at([(i as f64 + 0.5) * spacing, (j as f64 + 0.5) * spacing]).with_mass(MASS))
            .expect("sized for these particles");
    }
    let mut rng = Pcg32::seed_from_u64(11);
    assert!(analysis::thermalize(domain.store_mut(), 0.6 * EPSILON / BOLTZMANN, &mut rng));
    domain.initialize();
    domain
}

fn state_bits(domain: &ParticleDomain) -> Vec<u64> {
    let store = domain.store();
    [store.pos_x(), store.pos_y(), store.vel_x(), store.vel_y()]
        .into_iter()
        .flat_map(|values| values.iter().map(|v| v.to_bits()))
        .collect()
}

fn user_force_bitwise() -> Outcome {
    let steps = 1000;
    let dt = 2e-15;
    let mut builtin = fluid("builtin", LennardJones::new(EPSILON, SIGMA, CUTOFF));
    let mut user = fluid("user", user_law(LJ_FORCE, &lj_use_site()));
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for _ in 0..steps {
        builtin.advance(dt, &mut ctx);
        user.advance(dt, &mut ctx);
    }
    let (a, b) = (state_bits(&builtin), state_bits(&user));
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let moved = builtin.store().pos_x().iter().zip(fluid("start", LennardJones::new(EPSILON, SIGMA, CUTOFF)).store().pos_x()).filter(|(x, y)| x != y).count();
    Outcome::at_most("positions and velocities differing from the built-in's", "values", differing as f64, 0.0)
        .note(format!("100 argon atoms at T* = 0.6, {steps} velocity Verlet steps of {dt:e} s"))
        .note(format!("{moved} of 100 atoms moved from their lattice sites, so the comparison is of a live fluid"))
        .note("the law reads `minimum_image(b.position - a.position)`, the neighbour list's own separation, and its arithmetic in the built-in's order")
}

/// The radial force `−U′(r)` against the built-in's, across the well and the repulsive
/// wall.
///
/// The two compute the same function by different formulas: the built-in from `1/r²`
/// with no square root, the potential from `σ/r` differentiated by the compiler. Each
/// carries the rounding of its own route — `s = σ/r` is rounded once, `s⁶` multiplies
/// that error by six and adds three roundings of its own, `s¹²` doubles it again — so
/// each term is good to about 25 ε_mach of its own size, and the difference of two
/// such paths to about 50. The bound is 64 ε_mach times the larger term,
/// `24ε/r · (2s¹² + s⁶)`, since near the minimum the force itself is near zero and a
/// relative error would be meaningless there.
fn user_potential_matches() -> Outcome {
    let law = user_law(LJ_POTENTIAL, &lj_use_site());
    let builtin = LennardJones::new(EPSILON, SIGMA, CUTOFF);
    let samples = 2000;
    let mut worst: f64 = 0.0;
    let mut worst_at = 0.0;
    for k in 0..samples {
        let r = SIGMA * (0.88 + 1.6 * k as f64 / samples as f64);
        let (_, du) = law.pair_potential(r);
        let user_force = -du;
        let s6 = (SIGMA / r).powi(6);
        let scale = 24.0 * EPSILON / r * (2.0 * s6 * s6 + s6);
        let gap = (user_force - builtin.pair_force(r)).abs() / (f64::EPSILON * scale);
        if gap > worst {
            worst = gap;
            worst_at = r / SIGMA;
        }
    }
    Outcome::at_most("worst |F_user - F_builtin| in eps_mach x |force terms|", "1", worst, 64.0)
        .note(format!("{samples} separations from 0.88 sigma to 2.48 sigma; the worst at r = {worst_at:.3} sigma"))
        .note("the user potential is `4 epsilon (s^12 - s^6)` with s = sigma / distance(a, b); its force is the compiler's exact derivative")
}

/// The compiled `U′` against the central difference `(U(r₊) − U(r₋)) / (r₊ − r₋)`.
///
/// `r₊ = r + h` and `r₋ = r − h` are rounded when they are formed, so the difference is
/// taken over the points actually evaluated: `r₊ − r₋` is exact (Sterbenz), and the
/// quotient's only errors are then its truncation and the rounding of the energies.
///
/// - Truncation: `h²|U‴|/6`, with `U‴` from the analytic third derivative.
/// - Rounding: `U = 4ε(s¹² − s⁶)` with `s = σ/r`, unit roundoff `u = ε_mach/2`. `s`
///   carries `u`. `powi(s, 6)` forms `s²`, then `s⁴ = s²·s²`, then `s⁶ = s⁴·s²`, and each
///   product carries its factors' accumulated errors plus its own: `s²` has `3u`,
///   `s⁴` `7u`, `s⁶` `11u`. Then `s¹² = s⁶·s⁶` has `23u`. The subtraction and the multiply
///   by `4ε` (exact, a power of two times ε, then one rounding) add `2u` to each term. So
///   one energy is within `25u (|s¹²| + |s⁶|) 4ε = 12.5 ε_mach × |energy terms|`, and two
///   within 25.
///
///   The first version of this bound had two slips: it counted `s⁶` as `9u`, ignoring
///   the error its factors already carried, and it took `U‴`'s `s¹²` coefficient as
///   11·12·13 instead of 12·13·14. It failed at 1.05. The second slip was the binding
///   one, because the worst point, on the repulsive wall, is truncation-dominated, as the
///   last note shows. The bound, not the compiler, was wrong.
///
/// The case measures the worst ratio of the disagreement to that sum, which must be at
/// most 1.
fn derivative_is_the_gradient() -> Outcome {
    let law = user_law(LJ_POTENTIAL, &lj_use_site());
    let h = 1e-5 * SIGMA;
    let mut worst: f64 = 0.0;
    let mut at_worst = (0.0, 0.0, 0.0, 0.0);
    let samples = 400;
    for k in 0..samples {
        let r = SIGMA * (0.9 + 1.5 * k as f64 / samples as f64);
        let (_, du) = law.pair_potential(r);
        let (plus, minus) = (r + h, r - h);
        let width = plus - minus;
        let numeric = (law.pair_potential(plus).0 - law.pair_potential(minus).0) / width;
        let s6 = (SIGMA / r).powi(6);
        let energy_terms = 4.0 * EPSILON * (s6 * s6 + s6);
        // U‴ for 4ε(s¹² − s⁶) is 4ε(−12·13·14 s¹² + 6·7·8 s⁶)/r³ = 4ε(−2184 s¹² + 336 s⁶)/r³;
        // its magnitude is bounded by the sum of the terms' magnitudes.
        let third = 4.0 * EPSILON * (2184.0 * s6 * s6 + 336.0 * s6) / (r * r * r);
        let half = width / 2.0;
        let truncation = half * half * third / 6.0;
        let rounding = 25.0 * f64::EPSILON * energy_terms / width;
        let ratio = (du - numeric).abs() / (truncation + rounding);
        if ratio > worst {
            worst = ratio;
            at_worst = (r / SIGMA, (du - numeric).abs(), truncation, rounding);
        }
    }
    Outcome::at_most("worst |U'_compiled - central difference| / its error bound", "1", worst, 1.0)
        .note(format!("{samples} separations from 0.9 sigma to 2.4 sigma, h = 1e-5 sigma"))
        .note("the bound is the difference's truncation h^2 |U'''| / 6 plus 25 eps_mach of the energy terms over the step, derived from the operations in 4 epsilon (s^12 - s^6)")
        .note("the step is taken over the rounded points r+h and r-h themselves, whose difference is exact")
        .note(format!(
            "worst at r = {:.4} sigma: disagreement {:.3e} N against truncation {:.3e} N + rounding {:.3e} N",
            at_worst.0, at_worst.1, at_worst.2, at_worst.3
        ))
}

/// Spec §8.3's damped spring, with the damping sign that makes it a damper.
const SPRING: &str = "force spring(a: particle, b: particle) -> vec2<newton> {
    param rest_length: meter;
    param stiffness: newton / meter;
    param damping: kilogram / second;
    let dx = minimum_image(b.position - a.position);
    let extension = length(dx) - rest_length;
    return stiffness * extension * normalize(dx)
         + damping * dot(b.velocity - a.velocity, normalize(dx)) * normalize(dx);
}";

const SPRING_K: f64 = 40.0;
const SPRING_C: f64 = 2.0;
const SPRING_MU: f64 = 0.5;
const SPRING_REST: f64 = 1.0;

/// The decay rate measured from a run at `dt`.
///
/// For `μẍ + cẋ + kx = 0`, `E = ½μẋ² + ½kx² + γμxẋ` with `γ = c/2μ` decays as exactly
/// `e^{−2γt}` — no ripple at twice the frequency, unlike the plain energy — so the rate
/// is `−ln(E(T)/E(0)) / 2T` from two samples.
fn spring_decay(dt: f64) -> f64 {
    let law = user_law(
        SPRING,
        &format!(
            "spring(rest_length={SPRING_REST} meter, stiffness={SPRING_K} newton/meter, damping={SPRING_C} kilogram/second, cutoff=3 meter)"
        ),
    );
    let mut domain = ParticleDomain::new("pair", 2)
        .with_integrator(Integrator::VelocityVerlet)
        .with_bounds(BoundaryBox::new([-5.0, -5.0], [10.0, 10.0], lattice_domain_particle::ParticleBoundary::Open))
        .with_force(law);
    domain.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(1.0)).expect("room for two");
    domain.spawn(ParticleSpec::at([SPRING_REST + 0.2, 0.0]).with_mass(1.0)).expect("room for two");
    domain.initialize();
    let gamma = SPRING_C / (2.0 * SPRING_MU);
    let invariant = |d: &ParticleDomain| {
        let store = d.store();
        let x = store.pos_x()[1] - store.pos_x()[0] - SPRING_REST;
        let v = store.vel_x()[1] - store.vel_x()[0];
        0.5 * SPRING_MU * v * v + 0.5 * SPRING_K * x * x + gamma * SPRING_MU * x * v
    };
    let e0 = invariant(&domain);
    let duration = 2.0;
    let steps = (duration / dt).round() as usize;
    let mut arena = Arena::with_capacity(0);
    let mut ctx = StepContext::new(&mut arena);
    for _ in 0..steps {
        domain.advance(dt, &mut ctx);
    }
    -(invariant(&domain) / e0).ln() / (2.0 * steps as f64 * dt)
}

/// Does the law decay at the analytic `γ = c/2μ`?
///
/// Velocity Verlet hands a force the half-kicked velocity `v(t + dt/2)`, so the damping
/// term lags by half a step and the measured rate carries a first-order error in `dt`.
/// Richardson extrapolation, `2γ(dt/2) − γ(dt)`, cancels that term whatever its
/// coefficient, and what remains is second order. Its natural scale is `(ω₀ dt)²` with
/// `ω₀ = √(k/μ)`, velocity Verlet's own relative error being `(ω₀ dt)²/24` in the
/// frequency. The case asserts the extrapolated rate is within `(ω₀ dt)²` of `c/2μ`, which
/// is an order-of-magnitude bound: it holds unless the second-order coefficient exceeds
/// one.
///
/// The linearized lag, `μẍ ≈ −kx − cẋ + (c dt/2) ẍ`, predicts a mass of `μ − c dt/2` and a
/// first-order error of `c/2(μ − c dt/2) − c/2μ`. The measured first-order error is about
/// 12% below that. The case prints both, and does not assert the coefficient: the gap
/// is not explained here.
fn damped_spring() -> Outcome {
    let gamma = SPRING_C / (2.0 * SPRING_MU);
    let (coarse_dt, fine_dt) = (4e-3, 2e-3);
    let (coarse, fine) = (spring_decay(coarse_dt), spring_decay(fine_dt));
    let extrapolated = 2.0 * fine - coarse;
    let relative = ((extrapolated - gamma) / gamma).abs();
    let omega0 = (SPRING_K / SPRING_MU).sqrt();
    let bound = (omega0 * coarse_dt).powi(2);
    let (e_coarse, e_fine) = ((coarse - gamma).abs(), (fine - gamma).abs());
    let order = (e_coarse / e_fine).log2();
    let predicted = |dt: f64| SPRING_C / (2.0 * (SPRING_MU - SPRING_C * dt / 2.0)) - gamma;
    Outcome::at_most("|Richardson-extrapolated decay rate - c/2mu| / (c/2mu)", "1", relative, bound)
        .note(format!(
            "analytic gamma = c/2mu = {gamma}; measured {coarse:.6} at dt = {coarse_dt}, {fine:.6} at dt = {fine_dt}; extrapolated {extrapolated:.6}"
        ))
        .note(format!("the bound is (omega0 dt)^2 = {bound:.3e} at the coarse step, omega0 = sqrt(k/mu)"))
        .note(format!(
            "first-order error {e_coarse:.3e} -> {e_fine:.3e} (order {order:.3}); the half-step lag's linearization predicts {:.3e} -> {:.3e}",
            predicted(coarse_dt),
            predicted(fine_dt)
        ))
        .note("k = 40 N/m, c = 2 kg/s, two 1 kg beads (mu = 0.5 kg), released 0.2 m stretched; spec 8.3 writes `- damping`, which feeds energy in (W0317)")
}
