//! The portable GPU backend: §15.4's product baseline.
//!
//! > A portable WebGPU/wgpu backend is attractive for one code path across Windows, macOS,
//! > Linux, and potentially browsers. […] The portable backend is the product baseline;
//! > native accelerators are performance plugins.
//!
//! This crate implements [`lattice_compute::Device`] over `wgpu`, and one solver on top of
//! it: [`GpuDiffusion`], an explicit finite-volume diffusion run that lives on the device
//! from upload to readback.
//!
//! # The constraint worth knowing before reading anything else
//!
//! **WGSL has no `f64`, so this backend cannot run the reference precision.**
//!
//! That is not a `wgpu` limitation to be worked around; it is the WebGPU shading language,
//! whose scalar float types are `f32` and (by extension) `f16`. §10.5 makes `accurate64`
//! the mode validated runs use and §24.1 makes the scalar CPU path the executable
//! specification, so the product baseline is structurally unable to execute the
//! specification's own precision.
//!
//! The claim is easy to get wrong in the optimistic direction, because
//! `wgpu::Features::SHADER_F64` exists and an adapter may well report it. It enables
//! 64-bit floats in **SPIR-V** shaders on Vulkan; the portable path compiles **WGSL**,
//! and although `naga`'s WGSL front end will *parse* `f64`, its validator gates the type
//! behind a capability `wgpu` never grants. Reading the feature flag alone would have
//! produced a backend that advertised `accurate64` and then failed to compile.
//!
//! Because that is a claim about a toolchain rather than about this code, it is tested
//! rather than asserted: [`GpuDevice::probe_wgsl_f64`] compiles such a shader and reports
//! what comes back, and `wgsl_f64_is_rejected_by_the_portable_backend` fails if the
//! situation ever improves — which is the right way round, since the fix for that failure
//! is to offer more precisions.
//!
//! # What this means for M4
//!
//! The roadmap expected the GPU's disagreement with the CPU to be spent on FMA
//! contraction, transcendental accuracy and reduction order. Those are real, and the first
//! is present here. But they are worth a few `f32` ulps between them, while storing state
//! in `f32` at all is worth about 10⁹ times `f64::EPSILON` before a single step has run.
//! [`lattice_compute::Mechanism::StateRounding`] dominates, and the validation cases say
//! so explicitly rather than reporting one undifferentiated number.
//!
//! §23's *"portable GPU abstractions may leave performance on the table"* therefore
//! understates the situation for scientific work: what the portable abstraction leaves on
//! the table is **precision**, and that is what §15.4's native-backend plugin boundary
//! would eventually be for.
//!
//! # Running without a GPU
//!
//! [`GpuDevice::open`] returns [`DeviceError::Unavailable`](lattice_compute::DeviceError)
//! on a machine with no usable adapter. Callers should treat that as *absent* rather than
//! as a failure — the validation suite reports such cases the way M3 reports the Gillespie
//! rows, because a case that could not run must not report as passing.

mod bindings;
mod crank_nicolson;
mod device;
mod diffusion;
mod field_image;
mod geometry;
mod particles;
mod reduction;
pub use field_image::GpuFieldImage;
pub use wgpu;

pub use crank_nicolson::{CgOutcome, CrankNicolsonSetup, GpuCrankNicolson, ImplicitError};
pub use device::{GpuBuffer, GpuDevice};
pub use diffusion::{DiffusionError, DiffusionSetup, GpuDiffusion, SOLVER_PRECISION};
pub use particles::{GpuParticles, ParticleSetup, ParticleSnapshot};
pub use reduction::{GpuDot, Interior};
