//! Parallel CPU execution.
//!
//! Spec §24 names this crate *"scalar/SIMD CPU execution"* and §15.3 says what it owes:
//!
//! > Rust core with explicit parallel iterators/task pools for independent operations.
//! > […] Cache-aware cell sorting, blocked grid operations, and reduction trees.
//!
//! The crate is two layers. [`ThreadPool`] owns worker threads and hands out chunk
//! indices; it contains the only `unsafe` in the workspace, confined to two documented
//! invariants. [`Executor`] sits on top and decides *how* a loop is split — everything
//! at that level and above is safe code.
//!
//! # What it promises
//!
//! **Parallel execution changes the schedule, never the numbers.** Running a model on
//! eight threads produces the same bits, and therefore the same artifact content hash
//! (FR-011), as running it on one.
//!
//! It holds because [`Partition`] is a pure function of the work size, the *configured*
//! thread count, and the [`Grain`] — never of which worker happened to be free — and
//! because every kernel written against it is elementwise or a stencil, so each output
//! cell is computed by the same expression over the same inputs whatever the partition
//! is.
//!
//! This is deliberately a stronger promise than §19.1 asks for between backends, where
//! agreement is tolerance-based. It is available here because a CPU thread and another
//! CPU thread evaluate the same expression identically; it will not be available from a
//! GPU, and the contrast is the point. A cross-backend disagreement on the CPU is a bug,
//! full stop, with no tolerance to hide behind.
//!
//! # What it does not do
//!
//! No reductions. A blocked sum is a different number from a sequential sum, and the
//! blocking would depend on the core count of whichever machine ran it. §15.3's
//! reduction trees need a partition fixed independently of the hardware before they can
//! be added without giving up the promise above.
//!
//! # Using it
//!
//! ```
//! use lattice_cpu::{Executor, Grain};
//!
//! let executor = Executor::with_threads(4);
//! let mut field = vec![1.0f64; 65_536];
//!
//! // "Worth splitting above 4096 elements, in chunks of at least 512."
//! executor.for_each_chunk_mut(&mut field, Grain::new(4096, 512), |offset, chunk| {
//!     for (index, value) in chunk.iter_mut().enumerate() {
//!         *value = (offset + index) as f64 * 0.5;
//!     }
//! });
//!
//! assert_eq!(field[100], 50.0);
//! ```
//!
//! [`Grain`] is where a kernel says how much work is worth a barrier, and it is worth
//! reading before adding one: getting it wrong turns `--threads auto` into a
//! pessimization, which §15.1 treats as a regression like any other.

mod device;
mod exec;
mod pool;

pub use device::{CpuBuffer, CpuDevice};
pub use exec::{Executor, Grain, Partition};
pub use pool::ThreadPool;
