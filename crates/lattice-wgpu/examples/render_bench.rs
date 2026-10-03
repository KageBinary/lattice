//! End-to-end field presentation comparison, including transfers and color conversion.
//! Run in release mode on an otherwise idle GPU. Both paths compute identical fields.
use lattice_compute::Device;
use lattice_wgpu::{DiffusionSetup, GpuDevice, GpuDiffusion, GpuFieldImage, Interior, wgpu};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::open_default()?;
    println!(
        "{}; {}",
        device.label(),
        if cfg!(debug_assertions) {
            "DEBUG timings are not publishable"
        } else {
            "release"
        }
    );
    for side in [256, 512, 1024] {
        let frames = 120;
        let stride = side + 2;
        let mut field = vec![0.0; stride * stride];
        for j in 0..side {
            for i in 0..side {
                field[(j + 1) * stride + i + 1] = (i + j) as f64 / (2 * side) as f64;
            }
        }
        let faces = vec![0.01; (side + 1) * side];
        let setup = DiffusionSetup {
            nx: side,
            ny: side,
            halo: 1,
            stride,
            inv_dx2: (side * side) as f64,
            inv_dy2: (side * side) as f64,
            field: &field,
            face_x: &faces,
            face_y: &faces,
            source: None,
        };
        let dt = 0.2 / (0.01 * (side * side) as f64);
        let mut cpu_copy = GpuDiffusion::new(&device, setup)?;
        let mut resident = GpuDiffusion::new(&device, setup)?;
        let colors =
            std::array::from_fn(|i| [i as f32 / 255.0, i as f32 / 255.0, i as f32 / 255.0, 1.0]);
        let image = GpuFieldImage::new(
            &device,
            &resident.buffers(),
            Interior {
                nx: side,
                ny: side,
                stride,
                origin: stride + 1,
            },
            &colors,
            [0.0, 1.0],
        )?;
        let upload = device
            .raw_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("CPU-uploaded presentation"),
                size: wgpu::Extent3d {
                    width: side as u32,
                    height: side as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        // Warm both compute pipelines before either timing.
        cpu_copy.run(&device, dt, 1)?;
        resident.run(&device, dt, 1)?;
        device.finish()?;
        let mut pixels = vec![0u8; side * side * 4];
        let start = Instant::now();
        for _ in 0..frames {
            cpu_copy.run(&device, dt, 1)?;
            let values = cpu_copy.interior(&device)?;
            for j in 0..side {
                for i in 0..side {
                    let v = (values[j * side + i] * 255.0).round().clamp(0.0, 255.0) as u8;
                    let index = ((side - 1 - j) * side + i) * 4;
                    pixels[index..index + 4].copy_from_slice(&[v, v, v, 255]);
                }
            }
            device.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &upload,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side as u32 * 4),
                    rows_per_image: Some(side as u32),
                },
                wgpu::Extent3d {
                    width: side as u32,
                    height: side as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        device.finish()?;
        let roundtrip = start.elapsed();
        let start = Instant::now();
        for _ in 0..frames {
            let mut encoder = device
                .raw_device()
                .create_command_encoder(&Default::default());
            resident.encode(&device, &mut encoder, dt, 1);
            image.encode(&mut encoder, resident.parity());
            device.queue().submit(Some(encoder.finish()));
        }
        device.finish()?;
        let direct = start.elapsed();
        assert_eq!(cpu_copy.interior(&device)?, resident.interior(&device)?);
        println!(
            "{side}x{side}: {frames} frames, roundtrip {roundtrip:?}, resident {direct:?}, {:.2}x; final fields bit-identical",
            roundtrip.as_secs_f64() / direct.as_secs_f64()
        );
    }
    Ok(())
}
