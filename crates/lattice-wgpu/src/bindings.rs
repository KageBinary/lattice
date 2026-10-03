//! The three bind-group descriptors every solver in this crate writes out.
//!
//! Not an abstraction — a deduplication. Each of these is one struct literal with one
//! interesting field, repeated once per binding, and three solvers spelling them out
//! separately is thirty lines of noise in which a `read_only: true` can be wrong without
//! looking wrong.

/// A uniform buffer binding, visible to compute.
pub(crate) fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A storage buffer binding. `read_only` must match the shader's `read` or `read_write`.
pub(crate) fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Bind a whole buffer to `binding`.
pub(crate) fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}
