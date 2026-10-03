// Crank-Nicolson diffusion and its conjugate-gradient solve, resident on the device.
//
// This is a transcription of `HeatDomain::step_implicit` and `conjugate_gradient` from
// `lattice-domain-grid2d`, which §24.1 makes the executable specification this file is
// judged against. Every arithmetic expression below is copied from there with its
// parenthesisation intact. Reassociating one changes the numbers, and inside an iterative
// solver it changes them twice over: a different residual stops the iteration somewhere
// else, so the answer is a different approximation of the same solution.
//
// Everything is f32 because WGSL has no f64. See `lattice_compute::Mechanism`.
//
// The dot products live in `reduction.wgsl`, because the reduction is the piece M4.2
// deferred and it deserved to be a thing you can point at on its own.

struct Params {
    nx: u32,
    ny: u32,
    stride: u32,
    // Flat index of interior cell (0, 0): halo * stride + halo.
    origin: u32,

    inv_dx2: f32,
    inv_dy2: f32,
    // theta * dt, the coefficient in A = I - theta*dt*L.
    coefficient: f32,
    // 1 - theta, the explicit weight in the right-hand side. Its own field rather than
    // folded into dt, because `dt * ((1 - theta) * lap + s)` is the expression
    // `step_implicit` evaluates and the factored form rounds differently.
    one_minus_theta: f32,

    // Element offsets into `coeffs`, which packs face_x ++ face_y ++ source. Three separate
    // storage buffers would put this shader at nine bindings, and WebGPU's portable floor
    // is eight per stage.
    face_y_off: u32,
    source_off: u32,
    dt: f32,
    theta: f32,
    // Left, right, bottom, top. Homogeneous ghost = sign * edge.
    boundary_sign: vec4<f32>,
    // Inhomogeneous ghost adds 2 * prescribed face value.
    boundary_constant: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> p: array<f32>;
@group(0) @binding(5) var<storage, read_write> ap: array<f32>;
@group(0) @binding(6) var<storage, read> coeffs: array<f32>;
@group(0) @binding(7) var<storage, read_write> scalars: array<f32>;

// Slots in `scalars`. Kept in step with `Scalar` in crank_nicolson.rs.
const RS_OLD: u32 = 0u;
const RS_NEW: u32 = 1u;
const CURVATURE: u32 = 2u;
const ALPHA: u32 = 3u;
const BETA: u32 = 4u;

// True while the iteration is still meaningful.
//
// `pAp <= 0` means the operator is not the symmetric positive-definite one CG assumes, and
// `conjugate_gradient` reports that as divergence *without* having applied the step. The
// device makes the same decision one readback before the host learns of it, and that is
// what lets a whole iteration be dispatched as a single command buffer and still leave `x`
// and `r` exactly where the CPU would have left them.
//
// The upper test is not decoration. `conjugate_gradient` rejects a curvature that is not
// *finite* as well as one that is not positive, and WGSL has no `isInf`; comparing against
// the largest finite f32 is the portable spelling of the same test. NaN fails both
// comparisons, which is the third case and needs no separate handling.
const F32_MAX: f32 = 3.4028235e38;

fn iterating() -> bool {
    let curvature = scalars[CURVATURE];
    return curvature > 0.0 && curvature <= F32_MAX;
}

// L[u] = div(D grad u) at interior cell (i, j), read from `p`.
//
// Copied from `DiffusionOperator::apply_with`, expression for expression. `p` is the only
// field any stencil here reads, which is why it is also the only one with a halo kernel.
fn laplacian_p(i: u32, j: u32) -> f32 {
    let stride = params.stride;
    let k = params.origin + j * stride + i;
    let center = p[k];

    let fx_row = j * (params.nx + 1u);
    let fy_row = params.face_y_off + j * params.nx;
    let fy_next = params.face_y_off + (j + 1u) * params.nx;

    let west = coeffs[fx_row + i] * (p[k - 1u] - center);
    let east = coeffs[fx_row + i + 1u] * (p[k + 1u] - center);
    let south = coeffs[fy_row + i] * (p[k - stride] - center);
    let north = coeffs[fy_next + i] * (p[k + stride] - center);

    return (east + west) * params.inv_dx2 + (north + south) * params.inv_dy2;
}

// Zero-gradient Neumann on `p`, the only field an operator application reads.
//
// Insulated is the one boundary this solver implements, and it is also the one where the
// homogeneous and inhomogeneous halos coincide: `ghost = edge + g*h` with `g = 0` is the
// same write either way. `HaloMode` exists on the CPU because Dirichlet, Robin and periodic
// do *not* have that property; adding any of them here means adding the distinction too.
// `CrankNicolsonSetup` refuses them rather than substituting this one.
//
// Corner ghosts are deliberately not written, for the reason `diffusion.wgsl` gives.
@compute @workgroup_size(64)
fn halo_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nx = params.nx;
    let ny = params.ny;
    let stride = params.stride;
    let origin = params.origin;

    if (idx < nx) {
        p[origin - stride + idx] = params.boundary_sign.z * p[origin + idx];
        return;
    }
    if (idx < 2u * nx) {
        let i = idx - nx;
        p[origin + ny * stride + i] = params.boundary_sign.w * p[origin + (ny - 1u) * stride + i];
        return;
    }
    if (idx < 2u * nx + ny) {
        let j = idx - 2u * nx;
        p[origin + j * stride - 1u] = params.boundary_sign.x * p[origin + j * stride];
        return;
    }
    if (idx < 2u * nx + 2u * ny) {
        let j = idx - 2u * nx - ny;
        p[origin + j * stride + nx] = params.boundary_sign.y * p[origin + j * stride + nx - 1u];
        return;
    }
}

// ---------------------------------------------------------------------------
// Field kernels. Each writes one field's interior, and each is named for the
// line of solver.rs or domain.rs it transcribes.
// ---------------------------------------------------------------------------

fn outside(gid: vec3<u32>) -> bool {
    return gid.x >= params.nx || gid.y >= params.ny;
}

fn cell_of(gid: vec3<u32>) -> u32 {
    return params.origin + gid.y * params.stride + gid.x;
}

@compute @workgroup_size(8, 8)
fn copy_x_to_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid)) { return; }
    let k = cell_of(gid);
    p[k] = x[k];
}

@compute @workgroup_size(8, 8)
fn copy_r_to_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid)) { return; }
    let k = cell_of(gid);
    p[k] = r[k];
}

// `x.fill_interior(0.0)` — the exact solution when the right-hand side is the zero vector.
@compute @workgroup_size(8, 8)
fn zero_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid)) { return; }
    x[cell_of(gid)] = 0.0;
}

// rhs = u^n + dt*[(1-theta)*L(u^n) + theta*c + S], with u^n staged into `p` and its halo
// already filled.
//
// `lap` uses homogeneous ghosts. Add c=L(0), the prescribed-face contribution,
// to obtain L(u^n). Insulated faces contribute zero; corner cells include both
// incident faces. The matrix-vector products keep homogeneous ghosts throughout CG.
@compute @workgroup_size(8, 8)
fn assemble_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid)) { return; }
    let k = cell_of(gid);
    let lap = laplacian_p(gid.x, gid.y);
    let s = coeffs[params.source_off + k];
    let i = gid.x;
    let j = gid.y;
    let fx = j * (params.nx + 1u);
    let fy = params.face_y_off + j * params.nx;
    var west = 0.0;
    var east = 0.0;
    var south = 0.0;
    var north = 0.0;
    if (i == 0u) { west = coeffs[fx] * params.boundary_constant.x; }
    if (i + 1u == params.nx) { east = coeffs[fx + params.nx] * params.boundary_constant.y; }
    if (j == 0u) { south = coeffs[fy + i] * params.boundary_constant.z; }
    if (j + 1u == params.ny) { north = coeffs[fy + params.nx + i] * params.boundary_constant.w; }
    let c = (east + west) * params.inv_dx2 + (north + south) * params.inv_dy2;
    b[k] = x[k] + params.dt * (params.one_minus_theta * (lap + c) + params.theta * c + s);
}

// ap = A*p = p - theta*dt*L_hom(p), the operator CG iterates against.
@compute @workgroup_size(8, 8)
fn apply_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid)) { return; }
    let k = cell_of(gid);
    ap[k] = p[k] - params.coefficient * laplacian_p(gid.x, gid.y);
}

// r = b - ap.
@compute @workgroup_size(8, 8)
fn sub_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid)) { return; }
    let k = cell_of(gid);
    r[k] = b[k] - ap[k];
}

// x += alpha*p.
@compute @workgroup_size(8, 8)
fn axpy_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid) || !iterating()) { return; }
    let k = cell_of(gid);
    x[k] = x[k] + scalars[ALPHA] * p[k];
}

// r += (-alpha)*ap. Written the way `interior_axpy(residual, -alpha, a_direction)` is:
// the negation is on the scalar, not on the product.
@compute @workgroup_size(8, 8)
fn axpy_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid) || !iterating()) { return; }
    let k = cell_of(gid);
    r[k] = r[k] + (-scalars[ALPHA]) * ap[k];
}

// p = r + beta*p.
@compute @workgroup_size(8, 8)
fn xpby_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (outside(gid) || !iterating()) { return; }
    let k = cell_of(gid);
    p[k] = r[k] + scalars[BETA] * p[k];
}

// ---------------------------------------------------------------------------
// Scalar kernels. One invocation each. They exist so that alpha and beta never
// travel to the host and back, which would be two more stalls per iteration
// than the one the stopping test genuinely needs.
// ---------------------------------------------------------------------------

// After the opening `r . r`, before the first iteration. Nothing seeds the curvature,
// because every guarded kernel runs after that iteration's `p . Ap` has been written.
@compute @workgroup_size(1)
fn seed_rs_old() {
    scalars[RS_OLD] = scalars[RS_NEW];
}

@compute @workgroup_size(1)
fn compute_alpha() {
    scalars[ALPHA] = scalars[RS_OLD] / scalars[CURVATURE];
}

@compute @workgroup_size(1)
fn compute_beta() {
    if (!iterating()) { return; }
    scalars[BETA] = scalars[RS_NEW] / scalars[RS_OLD];
    scalars[RS_OLD] = scalars[RS_NEW];
}
