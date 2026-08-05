//! Validation cases for the rigid-body domain (spec §19.2).
//!
//! > Free fall, constant acceleration, harmonic oscillator, pendulum, **elastic/inelastic
//! > collision**, and **constrained motion**.
//!
//! The two in bold are the cases M0 deliberately left out of the report because the
//! rigid module did not exist. A case that cannot run must be absent, not present and
//! silently skipped — so this file is the other half of that promise.
//!
//! # What each case is really testing
//!
//! A collision test that only checks momentum proves almost nothing: momentum is
//! conserved *by construction* here, because every impulse is applied equal and
//! opposite in one function. The cases below are chosen to fail if the physics is
//! wrong in ways momentum would not notice — the exact energy an inelastic collision
//! loses, the analytic period of a pendulum, and the precise friction coefficient at
//! which a block stops sliding.

use lattice_domain_rigid2d::{
    Collider, Joint, RigidDomain, Shape, SolverConfig, Vec2, LINEAR_SLOP,
};
use lattice_ir::{Arena, BodySpec, Domain, StepContext};

use crate::{Case, Level, Outcome};

/// Standard gravity, m/s².
const G: f64 = 9.806_65;

/// The step every case runs at unless it says otherwise.
///
/// Small enough that the *integrator* is not the limiting error in any case here, so a
/// failure points at the physics rather than at the timestep.
const DT: f64 = 1.0 / 480.0;

pub(crate) static CASES: &[Case] = &[
    Case {
        name: "elastic_collision_exchanges_velocities",
        domain: "rigid2d",
        level: Level::Analytic,
        claim: "two equal bodies meeting head on with restitution 1 exchange velocities exactly",
        run: elastic_collision,
    },
    Case {
        name: "elastic_collision_conserves_energy",
        domain: "rigid2d",
        level: Level::Property,
        claim: "with restitution 1 and no friction there is nothing left to remove energy, so it is conserved",
        run: elastic_energy,
    },
    Case {
        name: "inelastic_collision_loses_the_predicted_energy",
        domain: "rigid2d",
        level: Level::Analytic,
        claim: "a perfectly inelastic collision loses exactly the energy momentum conservation predicts, not merely some",
        run: inelastic_energy,
    },
    Case {
        name: "collision_momentum_survives_one_iteration",
        domain: "rigid2d",
        level: Level::Property,
        claim: "linear momentum is exact even at a single solver iteration, because impulses are equal and opposite",
        run: momentum_at_one_iteration,
    },
    Case {
        name: "pendulum_period",
        domain: "rigid2d",
        level: Level::Analytic,
        claim: "a pin-jointed pendulum swings at the analytic small-angle period T = 2*pi*sqrt(L/g)",
        run: pendulum_period,
    },
    Case {
        name: "constrained_motion_holds_its_radius",
        domain: "rigid2d",
        level: Level::Property,
        claim: "a body on a distance joint stays on its circle over thousands of steps rather than spiralling out",
        run: constrained_radius,
    },
    Case {
        name: "coulomb_friction_threshold",
        domain: "rigid2d",
        level: Level::Analytic,
        claim: "a block on a slope holds when mu > tan(theta) and slides when it does not, at the analytic threshold",
        run: friction_threshold,
    },
    Case {
        name: "resting_penetration_is_bounded",
        domain: "rigid2d",
        level: Level::Property,
        claim: "a settled stack overlaps by a bounded multiple of the slop rather than sinking without limit",
        run: resting_penetration,
    },
    Case {
        name: "restitution_rebound_height",
        domain: "rigid2d",
        level: Level::Analytic,
        claim: "a ball dropped from h rebounds to e^2*h, which is what the coefficient of restitution means",
        run: rebound_height,
    },
];

/// Two discs of the given masses and speeds, collided head on.
fn head_on(restitution: f64, mass_a: f64, mass_b: f64, speed: f64, iterations: usize) -> RigidDomain {
    let mut world = RigidDomain::new("collision", 4)
        .with_gravity([0.0, 0.0])
        .with_solver(SolverConfig { velocity_iterations: iterations, ..SolverConfig::accurate() });
    let disc = world.register(
        Collider::new(Shape::circle(0.5).unwrap()).with_restitution(restitution).with_friction(0.0),
    );
    world.spawn_with_mass(BodySpec::at([-2.0, 0.0], disc).with_velocity([speed, 0.0]), mass_a);
    world.spawn_with_mass(BodySpec::at([2.0, 0.0], disc).with_velocity([-speed, 0.0]), mass_b);
    world
}

fn run(world: &mut RigidDomain, steps: usize) {
    let mut arena = Arena::with_capacity(8192);
    for _ in 0..steps {
        let mut ctx = StepContext::new(&mut arena);
        world.prepare(&mut ctx);
        world.advance(DT, &mut ctx);
    }
}

fn elastic_collision() -> Outcome {
    let mut world = head_on(1.0, 1.0, 1.0, 2.0, 64);
    run(&mut world, 3000);

    let (va, vb) = (world.bodies().vel_x()[0], world.bodies().vel_x()[1]);
    // Equal masses, restitution 1: they swap. The first was going +2 and must leave
    // at −2.
    Outcome::near("velocity of the first body after impact", "m/s", va, -2.0, 1e-6)
        .note(format!("the second leaves at {vb:+.9} m/s"))
        .note("equal masses with e = 1 exchange velocities; this is the textbook case")
}

fn elastic_energy() -> Outcome {
    let mut world = head_on(1.0, 1.0, 3.0, 2.0, 64);
    let before = world.bodies().kinetic_energy();
    run(&mut world, 3000);
    let after = world.bodies().kinetic_energy();

    Outcome::near("relative kinetic energy change", "1", (after - before) / before, 0.0, 1e-6)
        .note(format!("{before:.9} J before, {after:.9} J after"))
        .note("unequal masses, so a symmetric bug would not pass this")
        .note(
            "energy is NOT generally conserved by this module — restitution and friction \
             both remove it. This case isolates the one configuration where nothing does",
        )
}

fn inelastic_energy() -> Outcome {
    let (mass_a, mass_b, speed) = (1.0, 3.0, 4.0);
    let mut world = RigidDomain::new("collision", 4)
        .with_gravity([0.0, 0.0])
        .with_solver(SolverConfig::accurate());
    let disc =
        world.register(Collider::new(Shape::circle(0.5).unwrap()).with_restitution(0.0).with_friction(0.0));
    world.spawn_with_mass(BodySpec::at([-2.0, 0.0], disc).with_velocity([speed, 0.0]), mass_a);
    world.spawn_with_mass(BodySpec::at([2.0, 0.0], disc), mass_b);

    let before = world.bodies().kinetic_energy();
    run(&mut world, 2500);
    let after = world.bodies().kinetic_energy();

    // Everything ends up at the centre-of-mass velocity, so the energy left is
    // ½(ma+mb)·v_cm², and the loss is the rest. This is a *number*, not an inequality:
    // a solver that removed the wrong amount would still pass "energy decreased".
    let v_cm = mass_a * speed / (mass_a + mass_b);
    let expected = 0.5 * (mass_a + mass_b) * v_cm * v_cm;

    Outcome::near("kinetic energy remaining", "J", after, expected, 1e-4)
        .note(format!("{before:.6} J before; theory leaves {expected:.6} J at v_cm = {v_cm:.6} m/s"))
        .note(format!(
            "the bodies end at {:+.6} and {:+.6} m/s",
            world.bodies().vel_x()[0],
            world.bodies().vel_x()[1]
        ))
        .note("the energy lost is fixed by momentum conservation, so this checks the amount")
}

fn momentum_at_one_iteration() -> Outcome {
    // A messy multi-body pile-up at the worst iteration count the solver offers.
    let mut world = RigidDomain::new("pileup", 8)
        .with_gravity([0.0, 0.0])
        .with_solver(SolverConfig { velocity_iterations: 1, ..SolverConfig::default() });
    let disc =
        world.register(Collider::new(Shape::circle(0.5).unwrap()).with_restitution(0.4).with_friction(0.3));
    for i in 0..6 {
        world.spawn_with_mass(
            BodySpec::at([f64::from(i) * 1.3 - 3.0, 0.1 * f64::from(i)], disc)
                .with_velocity([1.5 - 0.5 * f64::from(i), 0.2])
                .with_angular_velocity(f64::from(i) - 2.5),
            1.0 + f64::from(i),
        );
    }

    let before = world.bodies().linear_momentum();
    run(&mut world, 2000);
    let after = world.bodies().linear_momentum();
    let scale = world.bodies().momentum_scale().max(1e-30);
    let drift = ((after[0] - before[0]).hypot(after[1] - before[1])) / scale;

    Outcome::at_most("relative linear momentum drift", "1", drift, 1e-12)
        .note("one velocity iteration: the constraints are badly under-resolved on purpose")
        .note(format!("{before:?} -> {after:?}, against a momentum scale of {scale:.6}"))
        .note(
            "an under-converged solve looks like bodies sinking into each other; it must \
             never look like momentum appearing from nowhere",
        )
}

fn pendulum_period() -> Outcome {
    let length = 1.0;
    let amplitude: f64 = 0.05;

    let mut world = RigidDomain::new("pendulum", 4).with_solver(SolverConfig::accurate());
    let pivot = world.register(Collider::new(Shape::circle(0.02).unwrap()));
    let bob = world.register(Collider::new(Shape::circle(0.02).unwrap()));
    world.spawn(BodySpec::statik([0.0, 0.0], pivot));
    let start = [length * amplitude.sin(), -length * amplitude.cos()];
    world.spawn_with_mass(BodySpec::at(start, bob), 1.0);
    world.add_joint(Joint::Pin {
        a: 0,
        b: 1,
        local_a: Vec2::ZERO,
        local_b: Vec2::from([-start[0], -start[1]]),
    });

    // Time the swings by watching x cross zero on the way down: once per period.
    let mut arena = Arena::with_capacity(8192);
    let mut previous = world.bodies().pos_x()[1];
    let mut crossings: Vec<f64> = Vec::new();
    for step in 0..30_000 {
        let mut ctx = StepContext::new(&mut arena);
        world.prepare(&mut ctx);
        world.advance(DT, &mut ctx);
        let x = world.bodies().pos_x()[1];
        if previous > 0.0 && x <= 0.0 {
            crossings.push(f64::from(step) * DT);
        }
        previous = x;
    }

    let expected = core::f64::consts::TAU * (length / G).sqrt();
    let measured = if crossings.len() >= 3 {
        (crossings[2] - crossings[0]) / 2.0
    } else {
        f64::NAN
    };

    Outcome::near("pendulum period", "s", measured, expected, 0.02 * expected)
        .note(format!("analytic small-angle period 2*pi*sqrt(L/g) = {expected:.6} s"))
        .note(format!("{} swings observed over {:.1} s", crossings.len(), 30_000.0 * DT))
        .note("the amplitude is 0.05 rad, small enough that sin(x) ~ x holds to 0.04%")
}

fn constrained_radius() -> Outcome {
    let mut world = RigidDomain::new("orbit", 4).with_solver(SolverConfig::accurate());
    let pivot = world.register(Collider::new(Shape::circle(0.02).unwrap()));
    let bob = world.register(Collider::new(Shape::circle(0.02).unwrap()));
    world.spawn(BodySpec::statik([0.0, 0.0], pivot));
    // Given a sideways kick, so it swings right round rather than oscillating.
    world.spawn_with_mass(BodySpec::at([2.0, 0.0], bob).with_velocity([0.0, 8.0]), 1.0);
    world.add_joint(Joint::Distance {
        a: 0,
        b: 1,
        local_a: Vec2::ZERO,
        local_b: Vec2::ZERO,
        rest_length: 2.0,
        rope: false,
    });

    let mut arena = Arena::with_capacity(8192);
    let mut worst: f64 = 0.0;
    for _ in 0..20_000 {
        let mut ctx = StepContext::new(&mut arena);
        world.prepare(&mut ctx);
        world.advance(DT, &mut ctx);
        let (x, y) = (world.bodies().pos_x()[1], world.bodies().pos_y()[1]);
        worst = worst.max(((x * x + y * y).sqrt() - 2.0).abs());
    }

    Outcome::at_most("worst radius error", "m", worst, 1e-3)
        .note("20,000 steps of a bob swinging right round its pivot under gravity")
        .note("a hard constraint without positional feedback drifts, and the drift is invisible until the linkage comes apart")
}

/// Distance a block slides down a slope of `angle` with friction `mu`, in metres.
fn slide_distance(angle: f64, friction: f64) -> f64 {
    let mut world = RigidDomain::new("slope", 4).with_solver(SolverConfig::accurate());
    let slope =
        world.register(Collider::new(Shape::rectangle(20.0, 0.5).unwrap()).with_friction(friction));
    let block =
        world.register(Collider::new(Shape::rectangle(0.4, 0.4).unwrap()).with_friction(friction));
    world.spawn(BodySpec::statik([0.0, 0.0], slope).with_angle(angle));

    let normal = [-angle.sin(), angle.cos()];
    world.spawn_with_mass(
        BodySpec::at([normal[0] * 0.9, normal[1] * 0.9], block).with_angle(angle),
        1.0,
    );

    let x0 = world.bodies().pos_x()[1];
    run(&mut world, 4000);
    (world.bodies().pos_x()[1] - x0).abs()
}

fn friction_threshold() -> Outcome {
    let angle: f64 = 0.4;
    let critical = angle.tan();
    // Twenty percent either side of the analytic threshold: close enough that a solver
    // whose friction cone is the wrong shape fails, far enough that round-off does not
    // decide the answer.
    let below = slide_distance(angle, critical * 0.8);
    let above = slide_distance(angle, critical * 1.2);

    // The ratio is the measurement: sliding freely below the threshold and holding
    // above it. A solver with no friction at all, or with friction that never releases,
    // fails this in opposite directions.
    let ratio = if below > 0.0 { above / below } else { f64::INFINITY };

    Outcome::at_most("slip above the threshold, as a fraction of slip below it", "1", ratio, 0.05)
        .note(format!("slope {angle:.3} rad, so Coulomb predicts slipping below mu = {critical:.4}"))
        .note(format!("mu = {:.4} slid {below:.4} m over {:.1} s", critical * 0.8, 4000.0 * DT))
        .note(format!("mu = {:.4} slid {above:.4} m", critical * 1.2))
}

fn resting_penetration() -> Outcome {
    let mut world = RigidDomain::new("stack", 8);
    let ground =
        world.register(Collider::new(Shape::rectangle(10.0, 0.5).unwrap()).with_friction(0.6));
    let crate_ =
        world.register(Collider::new(Shape::rectangle(0.5, 0.5).unwrap()).with_friction(0.6));
    world.spawn(BodySpec::statik([0.0, 0.0], ground));
    for level in 0..5 {
        world.spawn_with_mass(
            BodySpec::at([0.0, 1.0 + f64::from(level) * 1.002], crate_),
            1.0,
        );
    }

    run(&mut world, 6000);

    // The whole stack should still be a stack. The ground's top face is at 0.5 and each
    // crate is 1 m tall, so the centres belong at 1, 2, 3, 4 and 5 and the highest sits
    // at 5.0. Anything below that is the stack settling into itself.
    let resting_height = 0.5 + 0.5 + 4.0;
    let top = world.bodies().pos_y()[5];
    let sag = (resting_height - top).max(0.0);

    Outcome::at_most("total sag of a five-crate stack", "m", sag, 10.0 * LINEAR_SLOP)
        .note(format!(
            "top crate at {top:.6} m after {:.1} s, against a resting height of {resting_height}",
            6000.0 * DT
        ))
        .note(format!("the allowed overlap is {LINEAR_SLOP} m per contact, and there are five"))
        .note("a sinking stack is what an under-converged or non-warm-started solver looks like")
}

fn rebound_height() -> Outcome {
    let restitution = 0.8;
    let mut world = RigidDomain::new("bounce", 4).with_solver(SolverConfig::accurate());
    let ground = world.register(
        Collider::new(Shape::rectangle(10.0, 0.5).unwrap())
            .with_restitution(restitution)
            .with_friction(0.0),
    );
    let ball = world.register(
        Collider::new(Shape::circle(0.25).unwrap()).with_restitution(restitution).with_friction(0.0),
    );
    world.spawn(BodySpec::statik([0.0, 0.0], ground));
    // Resting height is 0.5 + 0.25 = 0.75; released from 3.0, so it falls 2.25 m.
    world.spawn_with_mass(BodySpec::at([0.0, 3.0], ball), 1.0);

    let mut arena = Arena::with_capacity(8192);
    let mut peak: f64 = 0.0;
    let mut bounced = false;
    for _ in 0..6000 {
        let mut ctx = StepContext::new(&mut arena);
        world.prepare(&mut ctx);
        world.advance(DT, &mut ctx);
        if world.bodies().vel_y()[1] > 0.1 {
            bounced = true;
        }
        if bounced {
            peak = peak.max(world.bodies().pos_y()[1]);
        }
    }

    let fall = 3.0 - 0.75;
    let expected = 0.75 + restitution * restitution * fall;
    Outcome::near("height of the first rebound", "m", peak, expected, 0.05 * fall)
        .note(format!("dropped from 3.0 m onto a surface at 0.75 m, so it falls {fall} m"))
        .note(format!("e = {restitution} keeps e^2 = {:.2} of the height", restitution * restitution))
        .note("this is what the coefficient of restitution *means*, so it is the case that would catch it being applied to the wrong quantity")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every case must pass, and every case must be *judged* — a case with an
    /// unreachable criterion is a case that proves nothing.
    #[test]
    fn every_rigid_case_passes() {
        for case in CASES {
            let outcome = (case.run)();
            assert!(
                outcome.passed(),
                "{}: measured {} {}, wanted {:?}\n  {}",
                case.name,
                outcome.observed,
                outcome.unit,
                outcome.criterion,
                outcome.notes.join("\n  ")
            );
            assert!(outcome.observed.is_finite(), "{} produced {}", case.name, outcome.observed);
            assert!(!outcome.notes.is_empty(), "{} records nothing about what it did", case.name);
        }
    }

    /// The two §19.2 rows M0 could not run must now be present under their own names.
    #[test]
    fn the_missing_canonical_cases_are_now_registered() {
        let names: Vec<&str> = CASES.iter().map(|c| c.name).collect();
        assert!(names.iter().any(|n| n.contains("elastic_collision")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("inelastic_collision")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("constrained_motion")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("pendulum")), "{names:?}");
    }
}
