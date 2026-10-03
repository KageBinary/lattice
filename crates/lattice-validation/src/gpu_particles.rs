use crate::Outcome;
use lattice_domain_particle::{
    BoundaryBox, Integrator, LennardJones, ParticleDomain, UniformAcceleration,
};
use lattice_ir::{Arena, Domain, ParticleSpec, StepContext};
use lattice_wgpu::{GpuParticles, ParticleSetup};

pub(super) fn gravity() -> Outcome {
    let device = super::device().expect("device");
    let mut worst = 0.0f64;
    for n in [0, 1, 65, 16384] {
        let positions: Vec<_> = (0..n)
            .map(|i| [0.01 * i as f64, 3.0 + 0.1 * (i % 7) as f64])
            .collect();
        let velocities = vec![[0.3, -0.2]; n];
        let masses: Vec<_> = (0..n).map(|i| 0.5 + 0.1 * (i % 11) as f64).collect();
        let acceleration = [0.0, -9.80665];
        let dt = 1e-3;
        let mut gpu = GpuParticles::new(
            device,
            ParticleSetup {
                positions: &positions,
                velocities: &velocities,
                masses: &masses,
                gravity: acceleration,
                lennard_jones: None,
                origin: [0.0; 2],
                extent: [1.0; 2],
                periodic: false,
                dt,
            },
        )
        .expect("GPU gravity setup");
        let mut cpu = ParticleDomain::new("gravity", n.max(1))
            .with_integrator(Integrator::VelocityVerlet)
            .with_force(UniformAcceleration::new(acceleration));
        for i in 0..n {
            cpu.spawn(
                ParticleSpec::at(positions[i])
                    .with_velocity(velocities[i])
                    .with_mass(masses[i]),
            );
        }
        cpu.initialize();
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for _ in 0..100 {
            cpu.advance(dt, &mut ctx);
        }
        gpu.run(device, 100).expect("GPU gravity run");
        let state = gpu.snapshot(device).expect("readback");
        // Additive rounding over 100 steps, at most 8 arithmetic roundings per component.
        for (i, position) in positions.iter().enumerate() {
            for (axis, coordinate) in position.iter().enumerate() {
                let p = [cpu.store().pos_x()[i], cpu.store().pos_y()[i]][axis];
                let v = [cpu.store().vel_x()[i], cpu.store().vel_y()[i]][axis];
                let bound = 100.0 * 8.0 * f32::EPSILON as f64 * coordinate.abs().max(4.0);
                let error = (state.positions[i][axis] - p)
                    .abs()
                    .max((state.velocities[i][axis] - v).abs());
                assert!(
                    error.is_finite() && error <= bound,
                    "gravity {i}: {error} > {bound}"
                );
                worst = worst.max(error / bound);
            }
        }
    }
    Outcome::at_most("fraction of accumulated rounding bound", "1", worst, 1.0)
}

pub(super) fn lennard_jones() -> Outcome {
    let device = super::device().expect("device");
    let mut worst = 0.0f64;
    for side in [2, 3, 8, 32] {
        let n = side * side;
        let extent = [side as f64 * 1.4; 2];
        let positions: Vec<_> = (0..n)
            .map(|i| [(i % side) as f64 * 1.4 + 0.7, (i / side) as f64 * 1.4 + 0.7])
            .collect();
        let velocities: Vec<_> = (0..n)
            .map(|i| [0.02 * (i as f64).sin(), 0.02 * (i as f64).cos()])
            .collect();
        let masses = vec![1.0; n];
        let dt = 1e-4;
        let setup = ParticleSetup {
            positions: &positions,
            velocities: &velocities,
            masses: &masses,
            gravity: [0.0; 2],
            lennard_jones: Some([1.0, 1.0, 2.5]),
            origin: [0.0; 2],
            extent,
            periodic: true,
            dt,
        };
        let mut gpu = GpuParticles::new(device, setup).expect("GPU LJ setup");
        let mut cpu = ParticleDomain::new("lj", n)
            .with_integrator(Integrator::VelocityVerlet)
            .with_bounds(BoundaryBox::periodic([0.0; 2], extent))
            .with_force(LennardJones::with_default_cutoff(1.0, 1.0));
        for i in 0..n {
            cpu.spawn(ParticleSpec::at(positions[i]).with_velocity(velocities[i]));
        }
        cpu.initialize();
        let initial_energy = cpu.total_energy();
        let initial_momentum = cpu.momentum();
        let mut arena = Arena::with_capacity(0);
        let mut ctx = StepContext::new(&mut arena);
        for _ in 0..40 {
            cpu.advance(dt, &mut ctx);
        }
        gpu.run(device, 40).expect("GPU LJ run");
        let result = gpu.snapshot(device).expect("readback");
        let mut repeated = GpuParticles::new(device, setup).expect("repeat setup");
        repeated.run(device, 40).expect("repeat run");
        let repeat = repeated.snapshot(device).expect("repeat readback");
        assert_eq!(
            result.positions, repeat.positions,
            "cell insertion schedule must not change the sum"
        );
        assert_eq!(result.velocities, repeat.velocities);
        // On this dilute lattice r remains >1.3 sigma. The absolute derivative of
        // each LJ pair force is <72 epsilon/sigma^2, with at most 24 neighbours.
        // The short interval has exp(sqrt(24*72)*t)<1.2 amplification. Budget
        // 32 arithmetic roundings per neighbour per step, scaled by coordinates.
        let bound = 1.2 * 40.0 * 24.0 * 32.0 * f32::EPSILON as f64 * extent[0];
        for i in 0..n {
            for axis in 0..2 {
                let p = [cpu.store().pos_x()[i], cpu.store().pos_y()[i]][axis];
                let v = [cpu.store().vel_x()[i], cpu.store().vel_y()[i]][axis];
                let error = (p - result.positions[i][axis])
                    .abs()
                    .max((v - result.velocities[i][axis]).abs());
                assert!(
                    error.is_finite() && error < bound,
                    "LJ {side} {i}: {error} > {bound}"
                );
                worst = worst.max(error / bound);
            }
        }
        // Reuse the reference energy definition on the GPU snapshot (shifted cutoff).
        let d = cpu.store_mut().dynamics();
        for i in 0..n {
            d.pos_x[i] = result.positions[i][0];
            d.pos_y[i] = result.positions[i][1];
            d.vel_x[i] = result.velocities[i][0];
            d.vel_y[i] = result.velocities[i][1];
        }
        cpu.initialize();
        let energy_error = (cpu.total_energy() - initial_energy).abs() / initial_energy.abs();
        let momentum_error = (cpu.momentum()[0] - initial_momentum[0])
            .hypot(cpu.momentum()[1] - initial_momentum[1])
            / n as f64;
        assert!(energy_error < 1e-3, "energy {energy_error}");
        assert!(
            momentum_error < 40.0 * 32.0 * f32::EPSILON as f64,
            "momentum {momentum_error}"
        );
    }
    Outcome::at_most("fraction of short-trajectory rounding bound", "1",worst,1.0)
        .note("2x2 through 32x32 particles; periodic alias deduplication; bit-identical repeated GPU runs; energy drift <1e-3")
}
