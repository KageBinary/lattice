//! A [`Device`] backed by wgpu.
//!
//! §15.4 makes this the product baseline: *"a portable WebGPU/wgpu backend is attractive
//! for one code path across Windows, macOS, Linux, and potentially browsers."* Everything
//! here is written against the portable feature set — no native extensions, no
//! vendor-specific paths — because a baseline that needs a particular adapter is not one.

use std::sync::Mutex;

use lattice_compute::{
    Backend, Buffer, Capabilities, Device, DeviceError, KernelCache, KernelKey, KernelSource,
    Precision, Usage, require_precision,
};

/// Elements per workgroup in one dimension. 8x8 = 64 invocations, which is two AMD waves
/// or two NVIDIA warps and is within every adapter's 256-invocation floor.
pub(crate) const WORKGROUP: u32 = 8;

/// A buffer in device memory.
#[derive(Debug)]
pub struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    len: usize,
    precision: Precision,
    usage: Usage,
}

impl GpuBuffer {
    /// Elements this buffer holds.
    pub fn elements(&self) -> usize {
        self.len
    }
}

impl Buffer for GpuBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn usage(&self) -> Usage {
        self.usage
    }
}

/// The portable GPU backend.
#[derive(Debug)]
pub struct GpuDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    precision: Precision,
    capabilities: Capabilities,
    adapter_info: wgpu::AdapterInfo,
    /// §15.5's *"amortize compilation"*. Behind a mutex because [`Device`] is `Sync` and a
    /// cache is the one piece of a device that is genuinely mutable; the lock is taken
    /// once per pipeline construction, never inside a dispatch.
    shaders: Mutex<KernelCache<wgpu::ShaderModule>>,
}

impl GpuDevice {
    /// Open the default adapter at `precision`.
    ///
    /// # Errors
    ///
    /// [`DeviceError::Unavailable`] when there is no adapter — a headless CI machine, a
    /// container without a GPU, a driver that will not initialize. Callers in the
    /// validation suite treat that as *absent* rather than as a failure, the way M3 treats
    /// the Gillespie rows: a case that could not run must not report as passing.
    ///
    /// [`DeviceError::UnsupportedPrecision`] for anything but [`Precision::Fast32`]. See
    /// [`GpuDevice::capabilities`].
    pub fn open(precision: Precision) -> Result<GpuDevice, DeviceError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|error| DeviceError::Unavailable {
            backend: Backend::Wgpu,
            detail: error.to_string(),
        })?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("lattice"),
            // Nothing. §15.4's portable baseline is the whole point, and a required
            // feature here would be a device that some supported platform cannot open.
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|error| DeviceError::Unavailable {
            backend: Backend::Wgpu,
            detail: error.to_string(),
        })?;

        let capabilities = Capabilities {
            adapter: format!("{} ({:?})", adapter_info.name, adapter_info.backend),
            // `Fast32` alone, and this is the constraint that shapes M4. See the crate
            // docs: WGSL has no `f64`, so the portable backend cannot store or compute
            // the reference precision. `wgsl_f64_is_rejected_by_the_portable_backend`
            // holds this claim against the real toolchain rather than against a memory of
            // the specification.
            precisions: vec![Precision::Fast32],
            max_workgroup: Some(limits.max_compute_invocations_per_workgroup),
            max_buffer_bytes: limits.max_buffer_size,
        };

        require_precision(Backend::Wgpu, &capabilities, precision)?;

        let opened = GpuDevice {
            device,
            queue,
            precision,
            capabilities,
            adapter_info,
            shaders: Mutex::new(KernelCache::new()),
        };
        opened.warm_up()?;
        Ok(opened)
    }

    /// Force the driver to initialize its transfer path before anyone measures anything.
    ///
    /// A freshly opened device is not actually ready to use. The first buffer round trip on
    /// this machine costs about **56 ms**; every subsequent one of the same size costs
    /// **160 µs**. That cost is real and someone has to pay it, but attributing it to
    /// whichever operation happened to go first makes that operation's measurement a lie —
    /// the benchmark harness initially reported a 25 ms *readback* which was almost entirely
    /// this, and the same latency would otherwise land in the first validation case and be
    /// read as GPU slowness.
    ///
    /// So it is paid here, where it belongs and where §15.1's "measure end-to-end" already
    /// accounts for it: opening a device is a setup cost, and this is part of opening a
    /// device.
    fn warm_up(&self) -> Result<(), DeviceError> {
        let mut scratch = self.alloc(1, Usage::Readback)?;
        self.write(&mut scratch, &[0.0])?;
        let mut out = [0.0];
        self.read(&scratch, &mut out)?;
        Ok(())
    }

    /// The shader module for `source`, compiling it if this is the first request.
    ///
    /// §15.5: *"amortize compilation … cache kernels by normalized expression, backend,
    /// precision, and hardware capabilities."* All four are in `key`; see
    /// [`KernelKey::new`].
    pub(crate) fn shader_module(
        &self,
        key: KernelKey,
        source: &KernelSource,
    ) -> Result<wgpu::ShaderModule, DeviceError> {
        let mut cache =
            self.shaders.lock().map_err(|error| DeviceError::Backend(error.to_string()))?;
        let module = cache.get_or_compile(key, source, |source| {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(source.name()),
                source: wgpu::ShaderSource::Wgsl(source.body().into()),
            });
            match pollster::block_on(scope.pop()) {
                Some(error) => Err(DeviceError::Backend(error.to_string())),
                None => Ok(module),
            }
        })?;
        Ok(module.clone())
    }

    /// How many shader modules have been compiled, and how many requests reused one.
    pub fn kernel_cache_stats(&self) -> (usize, usize) {
        match self.shaders.lock() {
            Ok(cache) => (cache.hits(), cache.misses()),
            Err(_) => (0, 0),
        }
    }

    /// Read any device buffer of `out.len()` `f32`s back into `f64`.
    ///
    /// The solver owns its ping-pong buffers as raw wgpu handles rather than as
    /// [`GpuBuffer`]s, so readback needs a form that does not go through the wrapper.
    pub(crate) fn read_raw(
        &self,
        buffer: &wgpu::Buffer,
        out: &mut [f64],
    ) -> Result<(), DeviceError> {
        if out.is_empty() {
            return Ok(());
        }
        let bytes = (out.len() * self.precision.state_bytes()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.submit_and_wait(encoder)?;

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait()?;
        receiver
            .recv()
            .map_err(|error| DeviceError::Backend(error.to_string()))?
            .map_err(|error| DeviceError::Backend(error.to_string()))?;

        {
            let view = slice
                .get_mapped_range()
                .map_err(|error| DeviceError::Backend(error.to_string()))?;
            for (slot, chunk) in out.iter_mut().zip(view.chunks_exact(4)) {
                let bits = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                *slot = f64::from(f32::from_bits(bits));
            }
        }
        staging.unmap();
        Ok(())
    }

    /// Open at [`Precision::Fast32`], the only precision this backend has.
    pub fn open_default() -> Result<GpuDevice, DeviceError> {
        GpuDevice::open(Precision::Fast32)
    }

    /// The adapter wgpu selected, for §19.3's *"publish exact hardware"*.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Try to compile a WGSL shader declaring an `f64`, and report what the toolchain
    /// says.
    ///
    /// This exists because the claim it tests is the single most consequential fact in
    /// M4 and is *not* obvious from the outside. `wgpu::Features::SHADER_F64` exists, and
    /// reading only that would suggest the reference precision is reachable on hardware
    /// that supports it. It is not: that feature enables 64-bit floats in **SPIR-V**
    /// shaders on Vulkan, and the portable path compiles **WGSL**, whose validator gates
    /// `f64` behind a capability `wgpu` never grants.
    ///
    /// Returns `Ok(())` if such a shader compiles — which would mean this comment has gone
    /// stale and the backend should offer more than `fast32` — and the validation message
    /// otherwise.
    pub fn probe_wgsl_f64(&self) -> Result<(), String> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("f64-probe"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var<storage, read_write> data: array<f32>;

                @compute @workgroup_size(1)
                fn main() {
                    let widened: f64 = f64(data[0]);
                    data[0] = f32(widened * 2.0lf);
                }
                "#
                .into(),
            ),
        });
        match pollster::block_on(scope.pop()) {
            Some(error) => Err(error.to_string()),
            None => Ok(()),
        }
    }

    /// Submit any pending work and block until the queue has drained.
    pub(crate) fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) -> Result<(), DeviceError> {
        self.queue.submit(Some(encoder.finish()));
        self.wait()
    }

    pub(crate) fn wait(&self) -> Result<(), DeviceError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| DeviceError::Backend(error.to_string()))
    }
}

impl Device for GpuDevice {
    type Buffer = GpuBuffer;

    fn backend(&self) -> Backend {
        Backend::Wgpu
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn alloc(&self, elements: usize, usage: Usage) -> Result<GpuBuffer, DeviceError> {
        let bytes = (elements * self.precision.state_bytes()) as u64;
        if bytes > self.capabilities.max_buffer_bytes {
            return Err(DeviceError::AllocationTooLarge {
                requested: bytes,
                limit: self.capabilities.max_buffer_bytes,
            });
        }

        // Every buffer is COPY_SRC | COPY_DST | STORAGE. A tighter set would save nothing
        // measurable here and would make `Usage` a correctness trap rather than the
        // scheduling hint §15.2 wants it to be.
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuBuffer { buffer, len: elements, precision: self.precision, usage })
    }

    fn write(&self, buffer: &mut GpuBuffer, data: &[f64]) -> Result<(), DeviceError> {
        if buffer.len != data.len() {
            return Err(DeviceError::LengthMismatch { buffer: buffer.len, host: data.len() });
        }
        if data.is_empty() {
            return Ok(());
        }

        // The narrowing. One place, and the reason `Mechanism::StateRounding` can be
        // named as the source of the disagreement this backend produces.
        let narrowed: Vec<f32> = data.iter().map(|&value| value as f32).collect();
        self.queue.write_buffer(&buffer.buffer, 0, bytes_of_f32(&narrowed));
        self.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
        Ok(())
    }

    fn read(&self, buffer: &GpuBuffer, out: &mut [f64]) -> Result<(), DeviceError> {
        if buffer.len != out.len() {
            return Err(DeviceError::LengthMismatch { buffer: buffer.len, host: out.len() });
        }
        self.read_raw(&buffer.buffer, out)
    }

    fn finish(&self) -> Result<(), DeviceError> {
        self.wait()
    }
}

/// Reinterpret `f32`s as the bytes wgpu wants, without a dependency on `bytemuck`.
///
/// Sound because `f32` has no padding and no invalid bit patterns, and the resulting slice
/// borrows the input for its whole lifetime.
fn bytes_of_f32(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy`, has no uninitialized padding, and every bit pattern of
    // `[u8; 4]` is a valid `f32`. The lifetime is tied to `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}
