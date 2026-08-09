//! The backend boundary: backend traits, device buffers, precision modes, the kernel
//! cache, and the tolerance budgets that hold two backends against each other.
//!
//! §24 names this crate *"backend traits, buffers, kernel cache"* and puts it beside
//! `lattice-cpu` and `lattice-wgpu` rather than above them. It sits at the bottom of the
//! workspace with no dependencies at all, because everything it describes is a boundary,
//! and a boundary that needed to import something would be describing the thing on one
//! side of it.
//!
//! # The four pieces
//!
//! [`Device`] and [`Buffer`] are who owns memory. The host always speaks `f64` across this
//! boundary — §24.1 makes the scalar CPU path *"the executable specification for
//! accelerated kernels"*, so the reference representation is the one the specification is
//! written in — and a backend narrows on the way in and widens on the way out.
//!
//! [`Precision`] is §10.5's table. It is a property of the device, not of the model, and a
//! device that cannot run a mode [refuses](Capabilities::supports) rather than substituting
//! one it likes.
//!
//! [`KernelCache`] is §15.5, keyed on all four of the inputs that section names.
//!
//! [`Tolerance`] is the answer to §19.1's *"do CPU and GPU agree within tolerance?"*, and
//! it is deliberately not a float. See below.
//!
//! # Why the tolerance is a structure and not a number
//!
//! A cross-backend tolerance is the one number in a validation suite that nothing checks.
//! A bound that is too tight fails visibly; a bound that is too loose passes forever while
//! hiding every defect smaller than itself. So [`Tolerance`] is a list of named
//! [`Mechanism`]s, each with the derivation that produced its share, and its total is the
//! sum of its parts.
//!
//! M4.1 paid for this in advance. The scalar and parallel CPU paths agree *bit-for-bit*,
//! at a real cost in performance, so that every unit of budget spent here belongs to the
//! GPU and can be pointed at rather than inherited.
//!
//! # The constraint that shapes M4
//!
//! WGSL has no 64-bit float type. Not "slow", not "optional" — absent, and absent from the
//! WebGPU specification rather than from any one driver. §15.4 makes the portable backend
//! the product baseline, and §10.5 makes `accurate64` the mode validated runs use, so the
//! product baseline cannot execute the reference precision even in principle.
//!
//! That single fact reorders M4's difficulties. The roadmap expected the GPU budget to be
//! spent on FMA contraction, transcendental accuracy and reduction order; those are real
//! and they are present, but together they are worth a handful of `f32` ulps, while
//! storing state in `f32` at all costs about 10⁹ times `f64::EPSILON` before a single step
//! has run. [`Mechanism::StateRounding`] dominates every budget the portable backend will
//! ever need, and a test asserts that it does — a decomposition in which the dominant term
//! is not dominant has stopped describing the system.
//!
//! The consequence for §23's *"portable GPU abstractions may leave performance on the
//! table"*: what the portable abstraction leaves on the table here is not throughput but
//! **precision**, and the native-backend plugin boundary §15.4 asks for is what a run
//! needing `accurate64` on a GPU would eventually go through.
//!
//! # Using it
//!
//! ```
//! use lattice_compute::{Mechanism, Precision, Tolerance};
//!
//! // 200 explicit diffusion steps against an f64 reference.
//! let budget = Tolerance::stepped_kernel(Precision::Fast32, 200);
//!
//! assert_eq!(budget.terms().len(), 1);
//! assert_eq!(budget.terms()[0].mechanism, Mechanism::StateRounding);
//! assert!(budget.explain().contains("does not amplify existing error"));
//!
//! let reference = [300.0, 310.0, 305.0];
//! let measured = [300.0, 310.000_01, 305.0];
//! let comparison = budget.compare(&reference, &measured);
//! assert!(comparison.within_budget());
//! ```

mod cache;
mod device;
mod precision;
mod tolerance;

pub use cache::{KernelCache, KernelKey, KernelSource};
pub use device::{
    Backend, Buffer, Capabilities, Device, DeviceError, Usage, require_precision,
};
pub use precision::Precision;
pub use tolerance::{Comparison, Mechanism, Term, Tolerance};
