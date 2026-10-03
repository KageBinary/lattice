// A dot product over the interior of two haloed fields, with a stated association order.
//
// The explicit stencil in `diffusion.wgsl` has no reduction, which is why
// `lattice_compute::Mechanism::ReductionOrder` was exactly zero for M4.2. This file is
// where it stops being zero, so the order is fixed and written down rather than left to
// whatever the scheduler does:
//
//   - Workgroup `g` owns the contiguous interior range `[g*block, (g+1)*block)`, where
//     `block = WG * per_thread`.
//   - Within it, invocation `t` sums the contiguous run
//     `[g*block + t*per_thread, g*block + (t+1)*per_thread)` serially, in index order.
//   - The WG partials are combined by a fixed binary tree in workgroup memory.
//   - The `groups` workgroup results are combined by the same tree in one final workgroup,
//     which is why `groups` is capped at WG on the host.
//
// Nothing here is an atomic and nothing depends on the order workgroups finish in, so the
// same input gives the same sum on the same device on every run. That is the property a
// budget needs. A reduction whose order merely *differs* from the reference is a mechanism
// you can bound; one whose order is unpredictable is not, and no amount of tolerance fixes
// it.
//
// WGSL has no atomic add for floats, so the tempting non-deterministic shortcut is closed
// at the language level rather than by discipline.
//
// # The accumulation depth, which is the whole error bound
//
// Every input value passes through `per_thread - 1` serial additions, then `log2(WG)` tree
// levels, then `log2(WG)` more in the final stage. Call that total `depth`. The standard
// bound for a summation of that shape is `|computed - exact| <= depth * eps * sum|x_i|`,
// against `n - 1` for a sequential sum. `GpuDot::depth` reports it and
// `lattice_compute::Tolerance` is handed it rather than a constant.
//
// For a dot product of a vector with itself every term is non-negative, so `sum|x_i|`
// equals `|sum x_i|` and the relative bound is exactly `depth * eps`, with no cancellation
// factor. `p . Ap` has mixed signs and does carry one — a property of the problem, not of
// the reduction.

struct DotParams {
    nx: u32,
    ny: u32,
    stride: u32,
    // Flat index of interior cell (0, 0): halo * stride + halo.
    origin: u32,

    // Interior cells: nx * ny.
    n: u32,
    // Interior cells summed serially by one invocation of stage one.
    per_thread: u32,
    // Workgroups stage one dispatches, and partials stage two reads. Never exceeds WG.
    groups: u32,
    // Where in `out` the result lands, so several dots can share one scalar buffer.
    out_index: u32,
};

@group(0) @binding(0) var<uniform> params: DotParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// Invocations per workgroup. 256 is WebGPU's floor for
// `maxComputeInvocationsPerWorkgroup`, so a kernel built around it runs anywhere §15.4's
// portable baseline is supposed to run.
const WG: u32 = 256u;

var<workgroup> scratch: array<f32, WG>;

// Interior cell `t` in row-major order, as an index into a haloed field.
fn interior_index(t: u32) -> u32 {
    let j = t / params.nx;
    let i = t - j * params.nx;
    return params.origin + j * params.stride + i;
}

// Combine `scratch` into `scratch[0]` by a fixed binary tree.
//
// `span` takes the same sequence of values in every invocation, so `workgroupBarrier` is
// reached in uniform control flow, which WGSL requires. The `t < span` test that is not
// uniform sits between two barriers rather than around one.
fn tree_sum(t: u32) -> f32 {
    var span = WG / 2u;
    loop {
        if (span == 0u) {
            break;
        }
        workgroupBarrier();
        if (t < span) {
            scratch[t] = scratch[t] + scratch[t + span];
        }
        span = span / 2u;
    }
    workgroupBarrier();
    return scratch[0];
}

@compute @workgroup_size(WG)
fn stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = lid.x;
    let start = wid.x * WG * params.per_thread + t * params.per_thread;
    var end = start + params.per_thread;
    if (end > params.n) {
        end = params.n;
    }

    var acc = 0.0;
    // `start >= end` for any invocation whose run lies past the last interior cell, which
    // is the only way the tail workgroup is handled. No masking, no partial tree.
    for (var c = start; c < end; c = c + 1u) {
        let k = interior_index(c);
        acc = acc + a[k] * b[k];
    }

    scratch[t] = acc;
    let total = tree_sum(t);
    if (t == 0u) {
        partials[wid.x] = total;
    }
}

@compute @workgroup_size(WG)
fn stage2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var value = 0.0;
    if (t < params.groups) {
        value = partials[t];
    }
    scratch[t] = value;
    let total = tree_sum(t);
    if (t == 0u) {
        out[params.out_index] = total;
    }
}
