//! What this machine's portable backend actually offers.
//!
//! `cargo run -p lattice-wgpu --example probe`

use lattice_compute::{Buffer, Device, Precision, Usage};
use lattice_wgpu::GpuDevice;

fn main() {
    let device = match GpuDevice::open_default() {
        Ok(device) => device,
        Err(error) => {
            println!("no portable GPU backend here: {error}");
            return;
        }
    };

    let info = device.adapter_info();
    println!("adapter    : {}", info.name);
    println!("backend    : {:?}", info.backend);
    println!("device type: {:?}", info.device_type);
    println!("driver     : {} {}", info.driver, info.driver_info);
    println!("label      : {}", device.label());
    println!("precisions : {:?}", device.capabilities().precisions);
    println!("max wg     : {:?}", device.capabilities().max_workgroup);
    println!("max buffer : {} MiB", device.capabilities().max_buffer_bytes / (1 << 20));

    println!("\n-- does WGSL accept f64 through wgpu? --");
    match device.probe_wgsl_f64() {
        Ok(()) => println!("  COMPILED. The backend could offer accurate64; revisit the docs."),
        Err(message) => {
            println!("  rejected, as expected:");
            for line in message.lines().take(12) {
                println!("    {line}");
            }
        }
    }

    println!("\n-- readback latency: is it bandwidth or per-call cost? --");
    // The benchmark harness sees ~25 ms per readback and sees it at every buffer size,
    // which would make it a fixed sync cost rather than a transfer. Worth knowing which,
    // because the two have completely different fixes: a faster copy, or fewer copies.
    for elements in [1_024usize, 65_536, 1_048_576] {
        let data = vec![1.0f64; elements];
        let buffer = device.upload(&data, Usage::Readback).unwrap();
        let mut out = vec![0.0; elements];

        // First call separately: it pays for whatever the driver initializes lazily.
        let start = std::time::Instant::now();
        device.read(&buffer, &mut out).unwrap();
        let first = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..10 {
            device.read(&buffer, &mut out).unwrap();
        }
        let steady = start.elapsed() / 10;

        println!(
            "  {:>9} elements ({:>7.1} KiB): first {:>8.3?}, steady {:>8.3?}",
            elements,
            (elements * 4) as f64 / 1024.0,
            first,
            steady
        );
    }

    println!("\n-- what f32 storage costs before any arithmetic --");
    let values: Vec<f64> = (0..8).map(|k| 300.0 + f64::from(k) / 3.0).collect();
    let buffer = device.upload(&values, Usage::Resident).unwrap();
    let back = device.download(&buffer).unwrap();
    let worst = values
        .iter()
        .zip(&back)
        .map(|(want, got)| (got - want).abs() / want.abs())
        .fold(0.0f64, f64::max);
    println!("  round-trip worst relative error : {worst:.3e}");
    println!("  f32 epsilon                     : {:.3e}", Precision::Fast32.epsilon());
    println!("  f64 epsilon                     : {:.3e}", Precision::Accurate64.epsilon());
    println!("  buffer bytes for 8 elements     : {}", buffer.bytes());
}
