use lattice_wgpu::{GpuDevice, GpuParticles, ParticleSetup};
use std::sync::OnceLock;
fn device() -> Option<&'static GpuDevice> {
    static DEVICE: OnceLock<Option<GpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| match GpuDevice::open_default() {
            Ok(d) => Some(d),
            Err(lattice_compute::DeviceError::Unavailable { detail, .. }) => {
                eprintln!("SKIPPED GPU: {detail}");
                None
            }
            Err(e) => panic!("{e}"),
        })
        .as_ref()
}
#[test]
fn pair_forces_use_minimum_image_and_skip_coincident_particles() {
    let Some(device) = device() else { return };
    for positions in [
        [[0.1, 0.5], [2.9, 0.5]],
        [[0.1, 0.5], [0.1, 0.5]],
        [[0.1, 0.5], [1.6, 0.5]],
    ] {
        let masses = [1.0, 2.0];
        let velocities = [[0.0; 2]; 2];
        let setup = ParticleSetup {
            positions: &positions,
            velocities: &velocities,
            masses: &masses,
            gravity: [0.0; 2],
            lennard_jones: Some([1.0, 0.15, 0.4]),
            origin: [0.0; 2],
            extent: [3.0; 2],
            periodic: true,
            dt: 1e-5,
        };
        let gpu = GpuParticles::new(device, setup).unwrap();
        let state = gpu.snapshot(device).unwrap();
        let dx = positions[1][0] - positions[0][0];
        let dx = dx - 3.0 * (dx / 3.0).round();
        let r2 = dx * dx;
        let expected = if r2 > 0.0 && r2 < 0.16 {
            let inv = 1.0 / r2;
            let s6 = (0.15 * 0.15 * inv).powi(3);
            -24.0 * inv * (2.0 * s6 * s6 - s6) * dx
        } else {
            0.0
        };
        // Position narrowing perturbs separation by at most eps*box width;
        // r^-13 force sensitivity bounds the resulting force error.
        let bound = expected.abs() * 32.0 * f32::EPSILON as f64 * 3.0 / r2.sqrt().max(0.15) + 1e-6;
        assert!(
            (state.forces[0][0] - expected).abs() <= bound,
            "{:?} expected {expected}",
            state.forces
        );
        assert_eq!(state.forces[0][0], -state.forces[1][0]);
    }
}
#[test]
fn invalid_particle_inputs_return_errors_before_gpu_dispatch() {
    let Some(device) = device() else { return };
    let positions = [[0.0; 2]];
    let velocities = [[0.0; 2]];
    let masses = [1.0];
    let setup = ParticleSetup {
        positions: &positions,
        velocities: &velocities,
        masses: &masses,
        gravity: [0.0; 2],
        lennard_jones: None,
        origin: [0.0; 2],
        extent: [1.0; 2],
        periodic: false,
        dt: 0.01,
    };
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300] {
        assert!(GpuParticles::new(device, ParticleSetup { dt, ..setup }).is_err());
    }
    assert!(
        GpuParticles::new(
            device,
            ParticleSetup {
                velocities: &[],
                ..setup
            }
        )
        .is_err()
    );
    assert!(
        GpuParticles::new(
            device,
            ParticleSetup {
                lennard_jones: Some([1.0, 1e-30, 1.0]),
                ..setup
            }
        )
        .is_err()
    );
}
