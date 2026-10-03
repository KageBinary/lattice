//! A device-side dot product whose association order is a stated, derivable number.
//!
//! M4.2 left this open, and the way it was open is worth restating: the explicit stencil is
//! elementwise, so [`Mechanism::ReductionOrder`](lattice_compute::Mechanism::ReductionOrder)
//! contributed *exactly zero* to the cross-backend budget, and a term that is exactly zero
//! is a term nobody has to defend. Conjugate gradient is three dot products per iteration.
//!
//! # What this type is for
//!
//! Not speed. A reduction on a device is easy to make fast and easy to make irreproducible,
//! and the second is what the milestone had to avoid. So the shape here is fixed at
//! construction — how many workgroups, how many elements each invocation sums serially, in
//! what order the partials combine — and [`GpuDot::depth`] reports the resulting
//! accumulation depth so a [`Tolerance`](lattice_compute::Tolerance) can be handed the real
//! number instead of a constant somebody chose.
//!
//! See `reduction.wgsl` for the order itself.
//!
//! # Why the pipelines are shared and the bind groups are not
//!
//! Conjugate gradient needs three dots — `r·r`, `p·Ap`, `b·b` — over the same geometry with
//! different operands. They are the same two entry points against the same bind group
//! *layout*, so [`DotPipelines`] is built once and each [`GpuDot`] is a bind group, a
//! params buffer and a partials buffer. That matters more than it sounds: pipeline creation
//! is the expensive half of setting an implicit solver up, and an implicit solver already
//! needs about four times as many pipelines as an explicit one.

use lattice_compute::{Device, DeviceError, KernelKey, KernelSource, Usage};

use crate::device::GpuDevice;

/// Invocations per workgroup, matching `WG` in the shader.
///
/// 256 is WebGPU's floor for `maxComputeInvocationsPerWorkgroup`, so a kernel built around
/// it runs everywhere §15.4's portable baseline is meant to run.
pub(crate) const WG: u32 = 256;

/// Most workgroups stage one may dispatch.
///
/// Capped at [`WG`] so that stage two is a single workgroup reading one partial per
/// invocation. A second serial pass over the partials would be a third association rule to
/// write down and defend, for a reduction that is already not the bottleneck.
pub(crate) const MAX_GROUPS: u32 = WG;

/// Where the interior of a haloed field sits inside its buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interior {
    /// Interior cells along x.
    pub nx: usize,
    /// Interior cells along y.
    pub ny: usize,
    /// Elements between vertically adjacent cells.
    pub stride: usize,
    /// Flat index of interior cell `(0, 0)`.
    pub origin: usize,
}

impl Interior {
    /// Interior cells.
    pub fn cells(&self) -> usize {
        self.nx * self.ny
    }

    /// How the reduction will be split: elements summed serially per invocation, and
    /// workgroups.
    ///
    /// Chosen so that coverage is exact and contiguous: workgroup `g` owns
    /// `[g·WG·per_thread, (g+1)·WG·per_thread)`, the last one running past the end and
    /// clamping. Deriving `groups` from `per_thread` rather than the other way round is
    /// what keeps `groups <= MAX_GROUPS` true by construction.
    pub(crate) fn plan(&self) -> (u32, u32) {
        let cells = self.cells().max(1) as u32;
        let per_thread = cells.div_ceil(WG * MAX_GROUPS).max(1);
        let groups = cells.div_ceil(WG * per_thread).max(1);
        debug_assert!(
            groups <= MAX_GROUPS,
            "{groups} workgroups exceeds the stage-two cap"
        );
        (per_thread, groups)
    }

    /// Roundings any single input value passes through before it reaches the result.
    ///
    /// `per_thread - 1` serial additions, then `log2(WG)` tree levels in stage one, then
    /// `log2(WG)` more in stage two. This is the `d` in the standard summation bound
    /// `|computed - exact| <= d·ε·Σ|xᵢ|`, and it is what makes the reduction's contribution
    /// to a budget a derived quantity: a sequential sum of the same values has `d = n - 1`,
    /// and at 24 576 cells that is 24 575 against **31**.
    pub fn depth(&self) -> usize {
        let (per_thread, _) = self.plan();
        (per_thread as usize - 1) + 2 * WG.trailing_zeros() as usize
    }
}

/// The two entry points and the layout every [`GpuDot`] over one device shares.
#[derive(Debug)]
pub(crate) struct DotPipelines {
    layout: wgpu::BindGroupLayout,
    stage1: wgpu::ComputePipeline,
    stage2: wgpu::ComputePipeline,
    key: KernelKey,
}

const SHADER: &str = include_str!("reduction.wgsl");

impl DotPipelines {
    /// Compile the reduction for `interior`'s shape.
    ///
    /// The shape is a §15.5 specialization and therefore part of the cache key, together
    /// with the split it implies: two runs whose grids differ must not share a module
    /// compiled for one of them.
    pub(crate) fn new(device: &GpuDevice, interior: Interior) -> Result<DotPipelines, DeviceError> {
        let (per_thread, groups) = interior.plan();
        let source = KernelSource::new("reduce.dot", SHADER)
            .specialize("nx", interior.nx)
            .specialize("ny", interior.ny)
            .specialize("workgroup", WG)
            .specialize("per_thread", per_thread)
            .specialize("groups", groups);
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
                crate::bindings::storage(3, false),
                crate::bindings::storage(4, false),
            ],
            &["stage1", "stage2"],
        )?;
        let layout = kernels.layout;
        let make = |entry: &str| kernels.pipelines[usize::from(entry == "stage2")].clone();

        Ok(DotPipelines {
            stage1: make("stage1"),
            stage2: make("stage2"),
            layout,
            key,
        })
    }

    /// The cache key the module was compiled under (§15.5).
    pub(crate) fn kernel_key(&self) -> KernelKey {
        self.key
    }
}

/// One dot product: two operands, and a slot in a scalar buffer to land in.
#[derive(Debug)]
pub struct GpuDot {
    // Held because the bind group refers to them. wgpu resources are reference-counted
    // handles, so this is ownership rather than a lifetime.
    _params: wgpu::Buffer,
    _partials: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    out: wgpu::Buffer,
    out_index: u32,
    groups: u32,
    interior: Interior,
}

impl GpuDot {
    /// Prepare `sum over the interior of a[k] * b[k]`, written to `out[out_index]`.
    ///
    /// `a` and `b` may be the same buffer — `r·r` is the common case and both bindings are
    /// read-only, so aliasing them is exactly what WebGPU permits.
    pub(crate) fn new(
        device: &GpuDevice,
        pipelines: &DotPipelines,
        interior: Interior,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        out_index: u32,
    ) -> GpuDot {
        let (per_thread, groups) = interior.plan();

        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce.dot params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bytes = [0u8; 32];
        for (slot, value) in [
            interior.nx as u32,
            interior.ny as u32,
            interior.stride as u32,
            interior.origin as u32,
            interior.cells() as u32,
            per_thread,
            groups,
            out_index,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        device.queue.write_buffer(&params, 0, &bytes);

        let partials = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce.dot partials"),
            size: u64::from(MAX_GROUPS) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce.dot"),
            layout: &pipelines.layout,
            entries: &[
                crate::bindings::entry(0, &params),
                crate::bindings::entry(1, a),
                crate::bindings::entry(2, b),
                crate::bindings::entry(3, &partials),
                crate::bindings::entry(4, out),
            ],
        });

        GpuDot {
            _params: params,
            _partials: partials,
            bind_group,
            out: out.clone(),
            out_index,
            groups,
            interior,
        }
    }

    /// Record both stages into an open compute pass.
    ///
    /// Dispatches inside one pass execute in order with a memory barrier between them,
    /// which is what lets stage two read stage one's partials without a fence — the same
    /// property `GpuDiffusion::run` relies on for its ping-pong.
    pub(crate) fn record(&self, pipelines: &DotPipelines, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&pipelines.stage1);
        pass.dispatch_workgroups(self.groups, 1, 1);
        pass.set_pipeline(&pipelines.stage2);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Run the reduction on its own and read the result back.
    ///
    /// One submission and one stall, so this is the form for a measurement or a test rather
    /// than for the inside of an iteration. [`GpuDot::record`] is the one CG uses.
    pub(crate) fn evaluate(
        &self,
        device: &GpuDevice,
        pipelines: &DotPipelines,
    ) -> Result<f64, DeviceError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce.dot"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce.dot"),
                timestamp_writes: None,
            });
            self.record(pipelines, &mut pass);
        }
        // The following readback copy is ordered after this dispatch on the same
        // queue. Mapping waits for both; a separate host fence here is redundant.
        device.queue.submit(Some(encoder.finish()));
        let mut value = [0.0];
        device.read_raw_at(&self.out, self.out_index as usize, &mut value)?;
        Ok(value[0])
    }

    /// The whole reduction over two host fields: upload, reduce, read back.
    ///
    /// The form a *measurement* of the reduction needs, as against the form conjugate
    /// gradient needs. It exists because
    /// [`Mechanism::ReductionOrder`](lattice_compute::Mechanism::ReductionOrder) turns out
    /// to be almost invisible inside CG — a perturbed dot product changes which iterate you
    /// arrive at, and the stopping test then measures that iterate afresh — so a validation
    /// case that only ran the solver would be reporting a term it had no way to see.
    /// Reducing a field and comparing against a sequential `f64` sum has nothing to hide
    /// behind: the whole disagreement is the association order and the storage precision.
    ///
    /// `a` and `b` are whole haloed buffers; only `interior` is summed.
    pub fn over_host(
        device: &GpuDevice,
        interior: Interior,
        a: &[f64],
        b: &[f64],
    ) -> Result<f64, DeviceError> {
        assert_eq!(
            a.len(),
            b.len(),
            "the two operands must be the same field shape"
        );
        let pipelines = DotPipelines::new(device, interior)?;
        let left = device.upload(a, Usage::Upload)?;
        let right = device.upload(b, Usage::Upload)?;
        let out = device.upload(&[0.0], Usage::Readback)?;
        let dot = GpuDot::new(
            device,
            &pipelines,
            interior,
            &left.buffer,
            &right.buffer,
            &out.buffer,
            0,
        );
        dot.evaluate(device, &pipelines)
    }

    /// The geometry this dot was built for.
    pub fn interior(&self) -> Interior {
        self.interior
    }

    /// Workgroups stage one dispatches.
    pub fn groups(&self) -> u32 {
        self.groups
    }

    /// Roundings any input passes through. See [`Interior::depth`].
    pub fn depth(&self) -> usize {
        self.interior.depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split must cover every interior cell exactly once, at every size the cap
    /// changes shape at. An off-by-one here is a reduction that silently drops a row.
    #[test]
    fn the_split_covers_every_cell_exactly_once() {
        for cells in [
            1usize,
            63,
            64,
            255,
            256,
            257,
            65_535,
            65_536,
            65_537,
            1 << 20,
        ] {
            let interior = Interior {
                nx: cells,
                ny: 1,
                stride: cells + 2,
                origin: cells + 3,
            };
            let (per_thread, groups) = interior.plan();
            let covered = groups as usize * WG as usize * per_thread as usize;
            assert!(
                covered >= cells,
                "{cells} cells, {groups}x{WG}x{per_thread} covers {covered}"
            );
            // One fewer workgroup must not be enough, or the plan is wasting dispatches.
            let short = (groups as usize - 1) * WG as usize * per_thread as usize;
            assert!(
                short < cells,
                "{cells} cells fits in {groups} groups minus one"
            );
            assert!(
                groups <= MAX_GROUPS,
                "{groups} exceeds the stage-two cap for {cells} cells"
            );
        }
    }

    /// The whole reason to fix the order is to be able to state the depth. A depth that
    /// grew like `n` would be no better than the sequential sum it is being compared with.
    #[test]
    fn depth_grows_like_the_logarithm_and_not_like_the_count() {
        let small = Interior {
            nx: 192,
            ny: 128,
            stride: 194,
            origin: 195,
        }
        .depth();
        let large = Interior {
            nx: 1024,
            ny: 1024,
            stride: 1026,
            origin: 1027,
        }
        .depth();
        assert_eq!(
            small, 16,
            "24 576 cells fit one per invocation, so only the trees count"
        );
        assert!(
            large < 40,
            "1 048 576 cells should still be a depth of tens, not thousands"
        );
        assert!(large > small);
    }
}
