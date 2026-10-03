use crate::bench::{BenchOutcome, Check};
use lattice_compute::{Device, DeviceError};
use lattice_domain_particle::{
    BoundaryBox, Integrator, LennardJones, ParticleDomain, ParticleSpec, UniformAcceleration,
};
use lattice_ir::Pcg32;
use lattice_observe::{MemoryReport, Profile, Throughput, phase};
use lattice_wgpu::{GpuDevice, GpuParticles, ParticleSetup};
use std::time::Instant;

pub(super) fn gravity(scale: usize, device: &GpuDevice) -> Result<BenchOutcome, DeviceError> {
    run(scale, device, false)
}
pub(super) fn lj(scale: usize, device: &GpuDevice) -> Result<BenchOutcome, DeviceError> {
    run(scale, device, true)
}

fn run(scale: usize, device: &GpuDevice, lj: bool) -> Result<BenchOutcome, DeviceError> {
    let start = Instant::now();
    let side = 32 * scale;
    let n = if lj { side * side } else { 16384 * scale };
    let extent = if lj {
        [side as f64 * 1.4; 2]
    } else {
        [100.0; 2]
    };
    let mut cpu = ParticleDomain::new("particles", n).with_integrator(Integrator::VelocityVerlet);
    if lj {
        cpu = cpu
            .with_bounds(BoundaryBox::periodic([0.0; 2], extent))
            .with_force(LennardJones::with_default_cutoff(1.0, 1.0));
        let mut rng = Pcg32::seed_from_u64(20260805);
        let velocities: Vec<_> = (0..n)
            .map(|_| [rng.normal() * 0.4, rng.normal() * 0.4])
            .collect();
        let mean = [
            velocities.iter().map(|v| v[0]).sum::<f64>() / n as f64,
            velocities.iter().map(|v| v[1]).sum::<f64>() / n as f64,
        ];
        for (i, v) in velocities.iter().enumerate() {
            cpu.spawn(
                ParticleSpec::at([
                    ((i % side) as f64 + 0.5) * 1.4,
                    ((i / side) as f64 + 0.5) * 1.4,
                ])
                .with_velocity([v[0] - mean[0], v[1] - mean[1]]),
            );
        }
    } else {
        cpu = cpu.with_force(UniformAcceleration::earth_gravity());
        let mut rng = Pcg32::seed_from_u64(1);
        for _ in 0..n {
            cpu.spawn(
                ParticleSpec::at([rng.range(0.0, 100.0), rng.range(0.0, 100.0)])
                    .with_mass(rng.range(0.5, 2.0)),
            );
        }
    }
    cpu.initialize();
    let initial_energy = cpu.total_energy();
    let initial_momentum = cpu.momentum();
    let momentum_scale = cpu.momentum_scale();
    let store = cpu.store();
    let positions: Vec<_> = store
        .pos_x()
        .iter()
        .zip(store.pos_y())
        .map(|(&x, &y)| [x, y])
        .collect();
    let velocities: Vec<_> = store
        .vel_x()
        .iter()
        .zip(store.vel_y())
        .map(|(&x, &y)| [x, y])
        .collect();
    let masses = store.mass().to_vec();
    let steps = if lj { 200 } else { 400 };
    let dt = 1e-3;
    let setup = ParticleSetup {
        positions: &positions,
        velocities: &velocities,
        masses: &masses,
        gravity: if lj { [0.0; 2] } else { [0.0, -9.80665] },
        lennard_jones: lj.then_some([1.0, 1.0, 2.5]),
        origin: [0.0; 2],
        extent,
        periodic: lj,
        dt,
    };
    let mut warm = GpuParticles::new(device, setup)?;
    warm.run(device, 2)?;
    drop(warm);
    let mut solver = GpuParticles::new(device, setup)?;
    device.finish()?;
    let mut profile = Profile::new();
    profile.record(phase::SETUP, start.elapsed());
    let compute_start = Instant::now();
    solver.run(device, steps)?;
    let compute = compute_start.elapsed();
    profile.record(phase::COMPUTE, compute);
    let observe_start = Instant::now();
    let result = solver.snapshot(device)?;
    let nonfinite = result
        .positions
        .iter()
        .chain(&result.velocities)
        .chain(&result.forces)
        .flatten()
        .any(|v| !v.is_finite());
    let mut checks = vec![Check::new(
        "non-finite particle state",
        f64::from(u8::from(nonfinite)),
        0.0,
    )];
    if lj {
        let d = cpu.store_mut().dynamics();
        for i in 0..n {
            d.pos_x[i] = result.positions[i][0];
            d.pos_y[i] = result.positions[i][1];
            d.vel_x[i] = result.velocities[i][0];
            d.vel_y[i] = result.velocities[i][1];
        }
        cpu.initialize();
        checks.push(Check::new(
            "relative shifted-potential energy drift",
            ((cpu.total_energy() - initial_energy) / initial_energy).abs(),
            5e-3,
        ));
        checks.push(Check::new(
            "relative momentum drift",
            (cpu.momentum()[0] - initial_momentum[0])
                .hypot(cpu.momentum()[1] - initial_momentum[1])
                / momentum_scale.max(1e-30),
            steps as f64 * 32.0 * f32::EPSILON as f64,
        ));
    } else {
        let drop = 0.5 * 9.80665 * (dt * steps as f64).powi(2);
        let worst = positions
            .iter()
            .zip(&result.positions)
            .map(|(before, after)| (after[1] - (before[1] - drop)).abs())
            .fold(0.0f64, f64::max);
        checks.push(Check::new(
            "max deviation from analytic trajectory (m)",
            worst,
            100.0 * steps as f64 * 8.0 * f32::EPSILON as f64,
        ));
    }
    profile.record(phase::OBSERVE, observe_start.elapsed());
    let mut memory = MemoryReport::new();
    memory.record(
        "resident particles and sorted cell list",
        solver.memory_bytes(),
    );
    Ok(BenchOutcome {
        throughput: Throughput {
            steps: steps as u64,
            simulated_seconds: dt * steps as f64,
            wall_clock: compute,
            elements: n as u64,
        },
        profile,
        memory,
        checks,
    })
}
