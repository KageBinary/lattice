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
