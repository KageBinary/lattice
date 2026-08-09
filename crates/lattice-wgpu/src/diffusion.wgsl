// Explicit finite-volume diffusion, one step per `step` dispatch.
//
// This is a transcription of `DiffusionOperator::apply_with` and `HeatDomain::
// step_explicit` from `lattice-domain-grid2d`, which §24.1 makes the executable
// specification this file is judged against. The parenthesisation below is copied from
// there deliberately and must not be "simplified": `(east + west) * inv_dx2 + (north +
// south) * inv_dy2` and `center + dt * (lap + s)` are specific floating-point expressions,
// and reassociating them changes the numbers.
//
// Everything is f32 because WGSL has no f64. That is the whole of the cross-backend
// budget's dominant term; see `lattice_compute::Mechanism::StateRounding`.

struct Params {
    nx: u32,
    ny: u32,
    stride: u32,
    // Flat index of interior cell (0, 0): halo * stride + halo.
    origin: u32,
    inv_dx2: f32,
    inv_dy2: f32,
    dt: f32,
    // Keeps the struct a multiple of 16 bytes for the uniform address space.
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> face_x: array<f32>;
@group(0) @binding(4) var<storage, read> face_y: array<f32>;
@group(0) @binding(5) var<storage, read> source: array<f32>;

// Zero-gradient Neumann: the ghost cell takes the value of the interior cell it faces, so
// the flux across that boundary face is exactly zero. `ghost_value` in
// `lattice-domain-grid2d/src/boundary.rs` computes `data[edge] + g * spacing` with g = 0.
//
// Corner ghosts are deliberately not written. The five-point stencil below reads only
// (i±1, j) and (i, j±1), so no corner ghost is ever an input, and
// `the_five_point_stencil_never_reads_a_corner_ghost` in the grid2d crate holds that
// property on the CPU side rather than leaving it as an assumption here.
@compute @workgroup_size(64)
fn halo_insulated(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nx = params.nx;
    let ny = params.ny;
    let stride = params.stride;
    let origin = params.origin;

    if (idx < nx) {
        let i = idx;
        src[origin - stride + i] = src[origin + i];
        return;
    }
    if (idx < 2u * nx) {
        let i = idx - nx;
        src[origin + ny * stride + i] = src[origin + (ny - 1u) * stride + i];
        return;
    }
    if (idx < 2u * nx + ny) {
        let j = idx - 2u * nx;
        src[origin + j * stride - 1u] = src[origin + j * stride];
        return;
    }
    if (idx < 2u * nx + 2u * ny) {
        let j = idx - 2u * nx - ny;
        src[origin + j * stride + nx] = src[origin + j * stride + nx - 1u];
        return;
    }
}

@compute @workgroup_size(8, 8)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= params.nx || j >= params.ny) {
        return;
    }

    let stride = params.stride;
    let k = params.origin + j * stride + i;
    let center = src[k];

    // Face coefficients are harmonic means of adjacent diffusivities, precomputed on the
    // host. face_x has (nx + 1) entries per row; face_y has nx entries per row boundary.
    let fx_row = j * (params.nx + 1u);
    let fy_row = j * params.nx;
    let fy_next = (j + 1u) * params.nx;

    let west = face_x[fx_row + i] * (src[k - 1u] - center);
    let east = face_x[fx_row + i + 1u] * (src[k + 1u] - center);
    let south = face_y[fy_row + i] * (src[k - stride] - center);
    let north = face_y[fy_next + i] * (src[k + stride] - center);

    let lap = (east + west) * params.inv_dx2 + (north + south) * params.inv_dy2;
    dst[k] = center + params.dt * (lap + source[k]);
}
