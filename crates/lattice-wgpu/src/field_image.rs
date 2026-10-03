//! A GPU field becomes a reusable display texture without crossing host memory.
//! The caller supplies its palette and owns submission order; no UI dependency.
use crate::{GpuDevice, Interior};
use lattice_compute::{Device, DeviceError, KernelKey, KernelSource, Usage};

#[derive(Debug)]
pub struct GpuFieldImage {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    pipeline: wgpu::ComputePipeline,
    groups: Vec<wgpu::BindGroup>,
    layout: Interior,
}

impl GpuFieldImage {
    /// Palette entries are gamma-encoded RGBA values in `[0,1]`. Range is fixed and
    /// labelled by the caller, so changing data cannot silently rescale the picture.
    pub fn new(
        device: &GpuDevice,
        fields: &[&wgpu::Buffer],
        layout: Interior,
        colors: &[[f32; 4]; 256],
        range: [f64; 2],
    ) -> Result<Self, DeviceError> {
        let limits = device.device.limits();
        if layout.nx == 0
            || layout.ny == 0
            || layout.nx > limits.max_texture_dimension_2d as usize
            || layout.ny > limits.max_texture_dimension_2d as usize
            || layout.stride < layout.nx
            || fields.is_empty()
            || range
                .iter()
                .any(|v| !v.is_finite() || !(*v as f32).is_finite())
            || range[1] < range[0]
            || !((range[1] as f32) - (range[0] as f32)).is_finite()
            || colors
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(DeviceError::Backend(
                "invalid field image geometry or range".into(),
            ));
        }
        let required = (layout.ny - 1)
            .checked_mul(layout.stride)
            .and_then(|n| n.checked_add(layout.origin))
            .and_then(|n| n.checked_add(layout.nx))
            .ok_or_else(|| DeviceError::Backend("field image size overflow".into()))?;
        if required > u32::MAX as usize || fields.iter().any(|b| b.size() / 4 < required as u64) {
            return Err(DeviceError::Backend(
                "field image buffer is smaller than its geometry".into(),
            ));
        }
        let texture = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resident field image"),
            size: wgpu::Extent3d {
                width: layout.nx as u32,
                height: layout.ny as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let uniform = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field image params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut params = [0u8; 32];
        for (i, v) in [layout.nx, layout.ny, layout.stride, layout.origin]
            .into_iter()
            .enumerate()
        {
            params[i * 4..i * 4 + 4].copy_from_slice(&(v as u32).to_le_bytes());
        }
        params[16..20].copy_from_slice(&(range[0] as f32).to_le_bytes());
        params[20..24].copy_from_slice(&(range[1] as f32).to_le_bytes());
        device.queue.write_buffer(&uniform, 0, &params);
        let palette: Vec<f64> = colors.iter().flatten().map(|&v| f64::from(v)).collect();
        let palette = device.upload(&palette, Usage::Upload)?;
        let source = KernelSource::new("field.image", include_str!("field_image.wgsl"));
        let key = KernelKey::new(
            &source,
            device.backend(),
            device.precision(),
            device.capabilities(),
        );
        let kernels = device.pipeline_set(
            key,
            &source,
            &[
                crate::bindings::uniform(0),
                crate::bindings::storage(1, true),
                crate::bindings::storage(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            &["image"],
        )?;
        let group_layout = kernels.layout;
        let pipeline = kernels.pipelines[0].clone();
        let groups = fields
            .iter()
            .map(|field| {
                device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("field image"),
                    layout: &group_layout,
                    entries: &[
                        crate::bindings::entry(0, &uniform),
                        crate::bindings::entry(1, field),
                        crate::bindings::entry(2, &palette.buffer),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                    ],
                })
            })
            .collect();
        Ok(Self {
            texture,
            view,
            pipeline,
            groups,
            layout,
        })
    }
    /// Must follow the writes to this field. No host fence or readback is required.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, field: usize) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[field], &[]);
        pass.dispatch_workgroups(
            (self.layout.nx as u32).div_ceil(8),
            (self.layout.ny as u32).div_ceil(8),
            1,
        );
    }
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offscreen_pixels_follow_the_palette_and_flip_y_without_reading_halos() {
        let device = match GpuDevice::open_default() {
            Ok(d) => d,
            Err(DeviceError::Unavailable { detail, .. }) => {
                eprintln!("SKIPPED GPU: {detail}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        let layout = Interior {
            nx: 3,
            ny: 2,
            stride: 5,
            origin: 6,
        };
        let mut data = vec![f64::NAN; 20];
        data[6..9].copy_from_slice(&[0.0, 0.5, 1.0]);
        data[11..14].copy_from_slice(&[1.0, 0.25, 0.75]);
        let field = device.upload(&data, Usage::Upload).unwrap();
        let colors = std::array::from_fn(|i| [i as f32 / 255.0, 0.0, 0.0, 1.0]);
        for invalid in [
            Interior {
                stride: usize::MAX,
                ..layout
            },
            Interior {
                origin: usize::MAX,
                ..layout
            },
            Interior {
                nx: usize::MAX,
                ..layout
            },
        ] {
            assert!(
                GpuFieldImage::new(&device, &[&field.buffer], invalid, &colors, [0.0, 1.0])
                    .is_err()
            );
        }
        assert!(
            GpuFieldImage::new(
                &device,
                &[&field.buffer],
                layout,
                &colors,
                [-(f32::MAX as f64), f32::MAX as f64]
            )
            .is_err()
        );
        let image =
            GpuFieldImage::new(&device, &[&field.buffer], layout, &colors, [0.0, 1.0]).unwrap();
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 512,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.device.create_command_encoder(&Default::default());
        image.encode(&mut encoder, 0);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: image.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 3,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        device.queue.submit(Some(encoder.finish()));
        let (send, recv) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
        device.wait().unwrap();
        recv.recv().unwrap().unwrap();
        let pixels = staging.slice(..).get_mapped_range().unwrap();
        assert_eq!([pixels[0], pixels[4], pixels[8]], [255, 64, 191]);
        assert_eq!([pixels[256], pixels[260], pixels[264]], [0, 128, 255]);
        assert_eq!([pixels[3], pixels[7], pixels[11]], [255; 3]);
        drop(pixels);
        staging.unmap();
    }
}
