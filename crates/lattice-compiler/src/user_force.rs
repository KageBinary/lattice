//! A user-defined law at its use site: `force: spring(stiffness=40 newton/meter,
//! cutoff=1.5 meter);` (spec §8.3).
//!
//! The law was checked where it was declared. Here its `param`s are bound to values,
//! its cutoff and truncation are read, its program is lowered, and what the engine can
//! promise about it is worked out: the contract that goes into the model report.

use lattice_domain_particle::user::layout;
use lattice_domain_particle::{Truncation, UserLaw, UserLawKind};
use lattice_expr::Program;
use lattice_syntax::{Argument, Diagnostic, Diagnostics, Expr, ExprKind};
use lattice_units::Dimension;

use crate::eval::Evaluator;
use crate::laws::{Ty, TypedLaw};
use crate::lower_law::{lower, LoweredLaw};

/// A bound user law, and what the report says about it.
#[derive(Clone, PartialEq, Debug)]
pub struct UserForce {
    /// The law, ready for the domain.
    pub law: UserLaw,
    /// The report line: the law, its bindings, and its contract.
    pub description: String,
}

/// Bind a law where a particle set uses it. `sample` is that set's particles, which a
/// law reading masses or charges is sampled with. Reports and returns `None` on any
/// problem.
pub fn bind(
    law: &TypedLaw,
    value: &Expr,
    evaluator: &Evaluator<'_>,
    sample: &PairSample,
    diagnostics: &mut Diagnostics,
) -> Option<UserForce> {
    let arguments: &[Argument] = match &value.kind {
        ExprKind::Call(_, arguments) => arguments,
        _ => &[],
    };
    let lowered = match lower(law) {
        Ok(lowered) => lowered,
        Err(diagnostic) => {
            diagnostics.push(*diagnostic);
            return None;
        }
    };
    let is_pair = lowered.kind.is_pair();
    let is_pair_potential = lowered.kind == UserLawKind::PairPotential;

    let mut known: Vec<&str> = law.params.iter().map(|p| p.name.as_str()).collect();
    if is_pair {
        known.push("cutoff");
    }
    if is_pair_potential {
        known.push("truncation");
    }
    let mut failed = false;
    let mut bound: Vec<Option<Vec<f64>>> = vec![None; law.params.len()];
    let mut cutoff = None;
    let mut truncation = is_pair_potential.then_some(Truncation::EnergyShift);
    let mut seen: Vec<&str> = Vec::new();
    for argument in arguments {
        let Some(name) = &argument.name else {
            diagnostics.push(
                Diagnostic::error(format!("`{}` binds its parameters by name", law.name))
                    .with_code("E0204")
                    .at(argument.span, "a value without a name")
                    .help(format!("write it as `name=value`; `{}` takes {}", law.name, known.join(", "))),
            );
            failed = true;
            continue;
        };
        if seen.contains(&name.text.as_str()) {
            diagnostics.push(
                Diagnostic::error(format!("`{}` is bound twice", name.text)).with_code("E0201").span(name.span),
            );
            failed = true;
            continue;
        }
        seen.push(&name.text);
        if let Some(index) = law.params.iter().position(|p| p.name == name.text) {
            match bind_value(&argument.value, law.params[index].ty, &name.text, evaluator, diagnostics) {
                Some(values) => bound[index] = Some(values),
                None => failed = true,
            }
        } else if name.text == "cutoff" && is_pair {
            match evaluator.require(&argument.value, Dimension::LENGTH, "the cutoff", diagnostics) {
                Some(rc) if rc > 0.0 && rc.is_finite() => cutoff = Some(rc),
                Some(rc) => {
                    diagnostics.push(
                        Diagnostic::error("the cutoff must be a positive length")
                            .with_code("E0405")
                            .at(argument.value.span, format!("found {rc} m")),
                    );
                    failed = true;
                }
                None => failed = true,
            }
        } else if name.text == "truncation" && is_pair_potential {
            truncation = match argument.value.as_name() {
                Some("energy_shift") => Some(Truncation::EnergyShift),
                Some("force_shift") => Some(Truncation::ForceShift),
                Some("none") => None,
                _ => {
                    diagnostics.push(
                        Diagnostic::error("`truncation` is `energy_shift`, `force_shift` or `none`")
                            .with_code("E0208")
                            .span(argument.value.span),
                    );
                    failed = true;
                    None
                }
            };
        } else {
            diagnostics.push(
                Diagnostic::error(format!("`{}` has no parameter `{}`", law.name, name.text))
                    .with_code("E0204")
                    .at(name.span, "unknown parameter")
                    .help(format!("`{}` takes {}", law.name, known.join(", "))),
            );
            failed = true;
        }
    }

    let mut params = Vec::with_capacity(lowered.param_slots);
    for (param, binding) in law.params.iter().zip(&bound) {
        match binding.as_ref().or(param.default.as_ref()) {
            Some(values) => params.extend_from_slice(values),
            None => {
                diagnostics.push(
                    Diagnostic::error(format!("`{}` needs `{}`", law.name, param.name))
                        .with_code("E0203")
                        .at(value.span, format!("`{}` is not bound", param.name))
                        .help(format!("add `{}=…` here, or give the `param` a default", param.name)),
                );
                failed = true;
            }
        }
    }
    if is_pair && cutoff.is_none() && !failed {
        diagnostics.push(
            Diagnostic::error(format!("`{}` is a pair law and needs a `cutoff`", law.name))
                .with_code("E0203")
                .at(value.span, "no cutoff")
                .note("pairs are found through a neighbour list, which needs to know how far to look")
                .help("add `cutoff=2.5 meter`"),
        );
        failed = true;
    }
    if failed {
        return None;
    }

    let LoweredLaw { kind, program, energy, branches, .. } = lowered;
    let user = UserLaw::new(law.name.clone(), kind, program, params.clone(), cutoff, truncation)
        .with_member_dependence(law.reads_members);
    let mut user = user;

    // A law that reads a mass or a charge is sampled with this set's own members, one
    // pairing of charges at a time; one that reads neither is the same for every pair.
    let pairings = sample.pairings(law.reads_members);
    if let (UserLawKind::PairPotential, Some(rc)) = (kind, cutoff) {
        for &(a, b) in &pairings {
            let (u_c, du_c) = user.pair_potential_between(rc, a, b);
            if !(u_c.is_finite() && du_c.is_finite()) {
                diagnostics.push(
                    Diagnostic::error(format!("`{}` is not finite at its cutoff for this set's particles", law.name))
                        .with_code("E0405")
                        .span(value.span)
                        .note(format!(
                            "between masses {:.3e} and {:.3e} kg with charges {:.3e} and {:.3e} C, U(r_c) = {u_c:e} J \
                             and U'(r_c) = {du_c:e} N",
                            a[0], b[0], a[1], b[1]
                        ))
                        .help("a law that divides by a member needs particles for which that member is not zero"),
                );
                return None;
            }
        }
    }
    if let (UserLawKind::PairPotential, Some(energy), Some(rc)) = (kind, &energy, cutoff) {
        let stiffest = pairings
            .iter()
            .filter_map(|&(a, b)| well_curvature(energy, &params, rc, a, b))
            .fold(None, |most: Option<f64>, k| Some(most.map_or(k, |m| m.max(k))));
        if let Some(stiffness) = stiffest {
            user = user.with_stiffness(stiffness);
        }
    }
    if user.stiffness().is_none() {
        diagnostics.push(
            Diagnostic::warning(format!("`{}` does not limit the timestep", law.name))
                .with_code("W0314")
                .span(value.span)
                .note(match kind {
                    UserLawKind::PairPotential => "the potential has no well inside its cutoff for this set's particles, so there is no curvature to set a vibration period from",
                    UserLawKind::BodyPotential => "a potential on one particle sets no pair vibration, and its curvature is not sampled, so the step is the solve's `dt`, unchecked",
                    _ => "a force's stiffness cannot be read off its formula, so the step is the solve's `dt`, unchecked",
                })
                .help("give the solve an explicit `dt` small enough for the stiffest motion this law causes"),
        );
    }

    if let (UserLawKind::PairPotential, None, Some(rc)) = (kind, truncation, cutoff) {
        let largest = pairings
            .iter()
            .map(|&(a, b)| user.pair_potential_between(rc, a, b).0)
            .fold(0.0f64, |most, u| if u.abs() > most.abs() { u } else { most });
        if largest != 0.0 {
            diagnostics.push(
                Diagnostic::warning(format!("`{}` is untruncated, and its energy at the cutoff is not zero", law.name))
                    .with_code("W0315")
                    .span(value.span)
                    .note(format!(
                        "U(r_c) = {largest:.3e} J for this set's particles: every pair that crosses the cutoff \
                         changes the total energy by that much"
                    ))
                    .help("use `truncation=energy_shift` (the default) or `force_shift`"),
            );
        }
    }

    if kind == UserLawKind::PairForce && law.reads_velocity
        && let Some(power) = pairings.iter().filter_map(|&(a, b)| velocity_power(&user, a, b)).reduce(f64::max)
    {
        diagnostics.push(
            Diagnostic::warning(format!("`{}` feeds energy in through its velocity terms", law.name))
                .with_code("W0317")
                .span(value.span)
                .note(format!(
                    "with b moving at 1 m/s relative to a, along each axis in each direction, the \
                     velocity-dependent part of the force does up to {power:.3e} W of work on the pair; \
                     a damper must do negative work"
                ))
                .note(
                    "a pair force returns the force on `a`, with `minimum_image(b.position - a.position)` \
                     pointing from a to b: damping on `a` is `+c (v_rel . d) d` with `v_rel = b.velocity - a.velocity`",
                )
                .help("check the sign of the velocity term"),
        );
    }

    let description = describe(law, &user, &params, branches, &pairings);
    Some(UserForce { law: user, description })
}

/// The particles a law will act between, for sampling it at compile time: the set's
/// mass, and each pairing of charges the set contains.
#[derive(Clone, PartialEq, Debug)]
pub struct PairSample {
    /// The set's particle mass, kg.
    pub mass: f64,
    /// Each distinct `(q_a, q_b)` the set contains, C.
    pub charges: Vec<(f64, f64)>,
}

impl PairSample {
    /// Uncharged particles of one mass.
    pub fn uncharged(mass: f64) -> PairSample {
        PairSample { mass, charges: vec![(0.0, 0.0)] }
    }

    /// The `[mass, charge]` pairs to sample at. A law that reads no member is the same
    /// for every pair, so one pairing stands for all.
    fn pairings(&self, reads_members: bool) -> Vec<([f64; 2], [f64; 2])> {
        let all: Vec<_> = self.charges.iter().map(|&(qa, qb)| ([self.mass, qa], [self.mass, qb])).collect();
        if reads_members || all.is_empty() { all.into_iter().take(8).collect() } else { all.into_iter().take(1).collect() }
    }
}

/// Compile one law and bind it as the use site `use_site` would, outside any model: for
/// benchmarks, validation and tests that want a law without a scene around it.
///
/// `law_source` is the law's declaration; `use_site` is what follows `force:`. Returns
/// the rendered diagnostics when either fails.
pub fn law_from_source(law_source: &str, use_site: &str) -> Result<UserLaw, String> {
    let text = format!("project law {{ {law_source} x: {use_site}; }}");
    let file = lattice_syntax::SourceFile::new("law.lattice", text);
    let (project, mut diagnostics) = lattice_syntax::parse(&file);
    let Some(project) = project.filter(|_| !diagnostics.has_errors()) else {
        return Err(diagnostics.render(&file));
    };
    let units = lattice_units::UnitRegistry::si();
    let evaluator = Evaluator::new(&file, &units);
    let laws = crate::laws::check_laws(&project, &evaluator, &mut diagnostics);
    let (Some(law), Some(setting)) = (laws.first(), project.setting("x")) else {
        return Err(diagnostics.render(&file));
    };
    match bind(law, &setting.value, &evaluator, &PairSample::uncharged(1.0), &mut diagnostics) {
        Some(bound) if !diagnostics.has_errors() => Ok(bound.law),
        _ => Err(diagnostics.render(&file)),
    }
}

/// The arguments a `vec2` or scalar `param` is bound to.
fn bind_value(value: &Expr, ty: Ty, name: &str, evaluator: &Evaluator<'_>, diagnostics: &mut Diagnostics) -> Option<Vec<f64>> {
    match ty {
        Ty::Scalar(d) => Some(vec![evaluator.require(value, d, &format!("`{name}`"), diagnostics)?]),
        Ty::Vec2(d) => {
            // A setting writes a vector as a list, as `region:` does; `vec2(x, y)`, as a
            // law writes it, is accepted too.
            if let ExprKind::Call(callee, arguments) = &value.kind
                && callee.as_name() == Some("vec2")
                && arguments.len() == 2
            {
                let x = evaluator.require(&arguments[0].value, d, &format!("`{name}` (x)"), diagnostics);
                let y = evaluator.require(&arguments[1].value, d, &format!("`{name}` (y)"), diagnostics);
                return Some(vec![x?, y?]);
            }
            evaluator.pair(value, d, &format!("`{name}`"), diagnostics).map(|p| p.to_vec())
        }
        Ty::Bool | Ty::Particle => None,
    }
}

/// The curvature at a pair potential's well inside its cutoff, `U″(r_min)`, which sets
/// the fastest vibration and so the step, as for the built-in pair potentials.
///
/// The potential is sampled at 4096 separations up to the cutoff; the lowest finite
/// energy, if it is not at either end, is the well, and the curvature there comes from
/// differentiating the program twice — exactly, not by differences. A potential with
/// no interior well (purely repulsive or attractive) has no such mode and returns
/// `None`. `a` and `b` are the pair's `[mass, charge]`.
fn well_curvature(energy: &Program, params: &[f64], cutoff: f64, a: [f64; 2], b: [f64; 2]) -> Option<f64> {
    const SAMPLES: usize = 4096;
    let second = energy.derivative(layout::R).derivative(layout::R);
    let mut inputs = vec![0.0; layout::PAIR_INPUTS as usize];
    for (base, [mass, charge]) in [(layout::PAIR_A, a), (layout::PAIR_B, b)] {
        inputs[(base + layout::MASS) as usize] = mass;
        inputs[(base + layout::CHARGE) as usize] = charge;
    }
    let mut best: Option<(usize, f64)> = None;
    for k in 0..SAMPLES {
        let r = cutoff * (k + 1) as f64 / SAMPLES as f64;
        inputs[layout::R as usize] = r;
        let u = energy.eval_to_vec(&inputs, params)[0];
        if u.is_finite() && best.is_none_or(|(_, lowest)| u < lowest) {
            best = Some((k, u));
        }
    }
    let (k, _) = best?;
    if k == 0 || k == SAMPLES - 1 {
        return None;
    }
    inputs[layout::R as usize] = cutoff * (k + 1) as f64 / SAMPLES as f64;
    // Outputs: U, U′ (twice, once from each differentiation), U″.
    let curvature = second.eval_to_vec(&inputs, params)[3];
    (curvature.is_finite() && curvature > 0.0).then_some(curvature)
}

/// The most power the velocity-dependent part of a pair force puts into the pair, W,
/// when it is positive: a sampled test for a damper written with the wrong sign.
///
/// The pair is placed at half the cutoff along x, `a` at rest, with the members `a` and
/// `b` (`[mass, charge]`). For `b` moving at 1 m/s along each axis in each direction,
/// the force on `a` is compared with its value when nothing moves; the difference is the
/// velocity part, and its power on the pair is `ΔF_a · (v_a − v_b)`. A force across the
/// relative velocity, as a magnetic one is, does no work and passes.
fn velocity_power(user: &UserLaw, a: [f64; 2], b: [f64; 2]) -> Option<f64> {
    let rc = user.pair_cutoff()?;
    let r = 0.5 * rc;
    let base = |vbx: f64, vby: f64| {
        let mut inputs = vec![0.0; layout::PAIR_INPUTS as usize];
        inputs[layout::DX as usize] = r;
        inputs[layout::R as usize] = r;
        inputs[(layout::PAIR_B + layout::PX) as usize] = r;
        inputs[(layout::PAIR_A + layout::MASS) as usize] = a[0];
        inputs[(layout::PAIR_A + layout::CHARGE) as usize] = a[1];
        inputs[(layout::PAIR_B + layout::MASS) as usize] = b[0];
        inputs[(layout::PAIR_B + layout::CHARGE) as usize] = b[1];
        inputs[(layout::PAIR_B + layout::VX) as usize] = vbx;
        inputs[(layout::PAIR_B + layout::VY) as usize] = vby;
        user.outputs_at(&inputs)
    };
    let still = base(0.0, 0.0);
    let mut worst: f64 = 0.0;
    for (vx, vy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let moving = base(vx, vy);
        let (dfx, dfy) = (moving[0] - still[0], moving[1] - still[1]);
        // v_a − v_b = −v_b.
        let power = -(dfx * vx + dfy * vy);
        let scale = dfx.hypot(dfy);
        if power.is_finite() && power > 1e-9 * scale {
            worst = worst.max(power);
        }
    }
    (worst > 0.0).then_some(worst)
}

/// The report line: the law, what it was bound to, and what the law's own forces
/// promise (P1). Boundaries are the domain's business: a reflecting wall changes
/// momentum whatever the law, and a periodic seam breaks angular momentum.
fn describe(
    law: &TypedLaw,
    user: &UserLaw,
    params: &[f64],
    branches: bool,
    pairings: &[([f64; 2], [f64; 2])],
) -> String {
    let mut bindings = Vec::new();
    let mut at = 0;
    for param in &law.params {
        match param.ty {
            Ty::Vec2(_) => {
                bindings.push(format!("{} = [{:.4e}, {:.4e}]", param.name, params[at], params[at + 1]));
                at += 2;
            }
            _ => {
                bindings.push(format!("{} = {:.4e}", param.name, params[at]));
                at += 1;
            }
        }
    }
    let mut parts = vec![format!("user {} `{}`", if user.kind().is_potential() { "potential" } else { "force" }, law.name)];
    if !bindings.is_empty() {
        parts.push(bindings.join(", "));
    }
    if let Some(rc) = user.pair_cutoff() {
        let truncation = user.truncation().map_or("untruncated", Truncation::name);
        parts.push(format!("cutoff {rc:.4e} m ({truncation})"));
    }
    let promise = match user.kind() {
        UserLawKind::PairPotential => {
            // The largest energy and slope at the cutoff over the set's pairings; for a
            // law that reads no member there is one pairing and these are the law's own.
            let largest = |pick: fn((f64, f64)) -> f64| {
                user.pair_cutoff().map_or(0.0, |rc| {
                    pairings
                        .iter()
                        .map(|&(a, b)| pick(user.pair_potential_between(rc, a, b)))
                        .fold(0.0f64, |most, v| if v.abs() > most.abs() { v } else { most })
                })
            };
            let (u_c, du_c) = (largest(|p| p.0), largest(|p| p.1));
            let per_pair = if law.reads_members { " (the largest over this set's pairs; shifted pair by pair)" } else { "" };
            let energy = match user.truncation() {
                Some(Truncation::ForceShift) => "energy to integration error".to_string(),
                Some(Truncation::EnergyShift) => {
                    format!("energy up to the force step at the cutoff, U'(r_c) = {du_c:.3e} N{per_pair}")
                }
                None => format!("energy except at the cutoff, where it jumps by U(r_c) = {u_c:.3e} J{per_pair}"),
            };
            let qualifier = if branches { "; it branches, so only where U is continuous" } else { "" };
            return format!(
                "{}; its forces conserve momentum, angular momentum in open space (not across a periodic seam), and {energy}{qualifier}",
                parts.join("; ")
            );
        }
        UserLawKind::BodyPotential if branches => "conserves energy only where U is continuous — it branches",
        UserLawKind::BodyPotential => "conserves energy",
        UserLawKind::PairForce => "conserves momentum (equal and opposite by construction); energy not claimed",
        UserLawKind::BodyForce => "conserves nothing by construction; energy not claimed",
    };
    parts.push(promise.to_string());
    parts.join("; ")
}
