//! Shared checks before host-size geometry is narrowed to WGSL's u32 indices.
use crate::GpuDevice;
pub(crate) fn grid(
    device: &GpuDevice,
    nx: usize,
    ny: usize,
    halo: usize,
    stride: usize,
) -> Result<usize, String> {
    if nx == 0 || ny == 0 {
        return Err("a field needs at least one cell".into());
    }
    if halo == 0 {
        return Err("the five-point stencil requires halo >= 1".into());
    }
    let padding = halo.checked_mul(2).ok_or("halo size overflow")?;
    let width = nx.checked_add(padding).ok_or("field width overflow")?;
    let height = ny.checked_add(padding).ok_or("field height overflow")?;
    if stride != width {
        return Err(format!(
            "stride {stride} does not match nx {nx} with halo {halo}"
        ));
    }
    let total = stride.checked_mul(height).ok_or("field size overflow")?;
    let limits = device.raw_device().limits();
    let axis = limits.max_compute_workgroups_per_dimension as usize * 8;
    if total > u32::MAX as usize || nx > axis || ny > axis {
        return Err("field exceeds GPU index or dispatch limits".into());
    }
    if total as u64 * 4 > limits.max_storage_buffer_binding_size {
        return Err("field exceeds GPU storage binding limit".into());
    }
    Ok(total)
}
pub(crate) fn coefficients(
    inv_dx2: f64,
    inv_dy2: f64,
    face_x: &[f64],
    face_y: &[f64],
) -> Result<(), String> {
    if [inv_dx2, inv_dy2]
        .iter()
        .any(|&v| !v.is_finite() || !(v as f32).is_finite() || v as f32 <= 0.0)
        || face_x
            .iter()
            .chain(face_y)
            .any(|&v| !v.is_finite() || !(v as f32).is_finite() || v < 0.0)
    {
        return Err("diffusion requires positive finite inverse spacing and nonnegative finite face coefficients representable in f32".into());
    }
    Ok(())
}
