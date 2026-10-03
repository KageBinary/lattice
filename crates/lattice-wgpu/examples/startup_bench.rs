//! Separate device discovery, first pipeline setup and reuse on the same device.
use lattice_compute::{Device, Precision};
use lattice_wgpu::{CrankNicolsonSetup, GpuCrankNicolson, GpuDevice};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let device = GpuDevice::open_default()?;
    println!("{}: device open {:?}", device.label(), start.elapsed());
    let n = 256;
    let field = vec![300.0; (n + 2) * (n + 2)];
    let faces = vec![0.01; n * (n + 1)];
    let mut setup = CrankNicolsonSetup {
        nx: n,
        ny: n,
        halo: 1,
        stride: n + 2,
        inv_dx2: (n * n) as f64,
        inv_dy2: (n * n) as f64,
        theta: 0.5,
        dt: 0.001,
        field: &field,
        face_x: &faces,
        face_y: &faces,
        source: None,
        tolerance: 1.0,
        max_iterations: 500,
    };
    setup.tolerance = 10.0 * GpuCrankNicolson::floor_for(Precision::Fast32, &setup);
    for attempt in 0..6 {
        let start = Instant::now();
        let mut solver = GpuCrankNicolson::new(&device, setup)?;
        device.finish()?;
        let elapsed = start.elapsed();
        assert!(solver.step(&device)?.is_converged());
        assert!(solver.interior(&device)?.iter().all(|&v| v == 300.0));
        println!(
            "setup {attempt}: {elapsed:?}, {} cached pipeline families; constant field exact",
            device.pipeline_cache_len()
        );
    }
    Ok(())
}
