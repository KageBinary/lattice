//! The executor: how a loop is split, and who runs the pieces.
//!
//! A domain never talks to [`crate::ThreadPool`] directly. It asks an [`Executor`] to
//! run a kernel over a partition, and the executor decides — from its configured thread
//! count and the caller's declared grain — how many pieces there are. Which thread runs
//! which piece is the pool's business and nobody else's.
//!
//! # The determinism claim
//!
//! **Parallel execution changes the schedule, never the numbers.**
//!
//! That is a stronger promise than the tolerance-based agreement §19.1 asks of a
//! cross-backend comparison, and it is worth the constraint it imposes. It holds
//! because [`Partition`] is a pure function of the work size, the *configured* thread
//! count, and the grain — never of which worker happened to be free — and because every
//! kernel written against it is elementwise or a stencil, so each output cell is
//! computed by the same expression over the same inputs whatever the partition is.
//!
//! The constraint: **no reductions run here**. A sum split into chunks and added back
//! together is not the sum computed in order — the difference is round-off, but
//! round-off that would vary with the machine's core count, and FR-011 asks for the
//! same content hash from the same model and configuration. Reductions therefore stay
//! sequential until there is a partition for them that does not depend on the hardware.
//! [`Executor::label`] records the configuration in the run artifact either way, so a
//! result can always be traced to how it was executed.

use crate::pool::ThreadPool;

/// A kernel's statement of how much work is worth splitting.
///
/// Two numbers, because they answer two questions that a single one gets wrong.
///
/// **`floor`** — is this problem worth splitting *at all*? A dispatch is a barrier:
/// every worker has to be reached and every worker has to report back, which costs a few
/// microseconds however little work is inside. Below the floor the partition collapses
/// to one chunk and the kernel runs exactly as it did before, on the calling thread.
///
/// **`chunk`** — once it is worth splitting, how small may a piece get? This guards the
/// other end: a problem just above the floor should not be cut into fifty slivers whose
/// per-chunk overhead exceeds their contents.
///
/// Conflating the two is a real trap and was measured, not imagined. Setting one large
/// grain high enough to protect small problems also caps a medium problem at a handful
/// of chunks — a 262k-particle model dropped from 1.99x to 1.34x that way, because a
/// grain that keeps 16k particles sequential also keeps 262k on four threads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Grain {
    /// Below this many units, do not split at all.
    pub floor: usize,
    /// Never produce a chunk smaller than this.
    pub chunk: usize,
}

impl Grain {
    /// A grain that splits above `floor` units, into chunks of at least `chunk`.
    pub const fn new(floor: usize, chunk: usize) -> Grain {
        Grain { floor, chunk }
    }

    /// Split above `floor`, with chunks no smaller than an eighth of it.
    ///
    /// The ratio is a default, not a law: it means a problem right at the floor becomes
    /// at most eight chunks, which is enough to use a machine without cutting the
    /// smallest worthwhile problem into dust.
    pub const fn above(floor: usize) -> Grain {
        Grain { floor, chunk: if floor / 8 > 1 { floor / 8 } else { 1 } }
    }

    /// The same grain measured in rows of a `width`-cell grid.
    ///
    /// A row is not a fixed amount of work — ten rows of a 512-wide grid and ten of a
    /// 32-wide one differ by a factor of sixteen — so a kernel that splits by rows
    /// states its grain in cells and converts here.
    pub const fn per_row(self, width: usize) -> Grain {
        let width = if width == 0 { 1 } else { width };
        Grain { floor: self.floor.div_ceil(width), chunk: self.chunk.div_ceil(width) }
    }
}

/// How a run of work units is split into chunks.
///
/// A "unit" is whatever the caller counts: an element, a row, a cell. The partition is
/// computed once and is reproducible from its inputs — never from which worker happened
/// to be free — and that is what the crate's determinism promise rests on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Partition {
    units: usize,
    per_chunk: usize,
}

impl Partition {
    /// Split `units` into about `chunks_wanted` chunks, subject to `grain`.
    pub fn new(units: usize, chunks_wanted: usize, grain: Grain) -> Partition {
        if units < grain.floor {
            return Partition { units, per_chunk: units.max(1) };
        }
        let chunks_wanted = chunks_wanted.max(1);
        let per_chunk = units.div_ceil(chunks_wanted).max(grain.chunk.max(1));
        Partition { units, per_chunk }
    }

    /// Units in every chunk but possibly the last.
    pub fn units_per_chunk(&self) -> usize {
        self.per_chunk
    }

    /// How many chunks the work becomes.
    pub fn count(&self) -> usize {
        self.units.div_ceil(self.per_chunk.max(1))
    }

    /// Total units.
    pub fn units(&self) -> usize {
        self.units
    }
}

/// Decides how kernels are split and runs them.
///
/// Cheap to pass around by reference, expensive to construct — building one starts
/// threads. Construct it once per process and lend it out through `StepContext`.
#[derive(Debug)]
pub struct Executor {
    pool: Option<ThreadPool>,
    chunks_per_thread: usize,
}

/// The executor a `StepContext` uses when nobody supplies one.
static SEQUENTIAL: Executor = Executor::sequential();

impl Executor {
    /// An executor that runs everything on the calling thread.
    ///
    /// `const` so it can back a `static`: the sequential path must not need a
    /// constructor call, because it is what every test and every embedding that has not
    /// asked for threads gets.
    pub const fn sequential() -> Executor {
        Executor { pool: None, chunks_per_thread: 1 }
    }

    /// The shared sequential executor.
    pub fn shared_sequential() -> &'static Executor {
        &SEQUENTIAL
    }

    /// An executor using `threads` threads in total, including the caller's.
    ///
    /// `threads <= 1` is the sequential executor. Anything larger starts `threads - 1`
    /// workers, because the thread that calls a kernel runs a share of it.
    pub fn with_threads(threads: usize) -> Executor {
        if threads <= 1 {
            return Executor::sequential();
        }
        Executor { pool: Some(ThreadPool::new(threads - 1)), chunks_per_thread: 4 }
    }

    /// An executor sized to the machine.
    ///
    /// Falls back to sequential if the platform will not say how many cores it has,
    /// which is the right failure: a wrong guess about core count is a performance bug
    /// that hides, and sequential is at least honest.
    pub fn automatic() -> Executor {
        match std::thread::available_parallelism() {
            Ok(count) => Executor::with_threads(count.get()),
            Err(_) => Executor::sequential(),
        }
    }

    /// Set how many chunks each thread should be offered.
    ///
    /// One chunk per thread is optimal when every chunk costs the same, and worst when
    /// they do not: one slow chunk then holds up every other thread. Several chunks per
    /// thread lets a thread that finishes early take more. The default is four, which
    /// is enough to absorb the unevenness a cell list produces without making the
    /// per-chunk overhead visible.
    pub fn with_chunks_per_thread(mut self, chunks: usize) -> Executor {
        self.chunks_per_thread = chunks.max(1);
        self
    }

    /// Threads available to run kernels, including the caller's.
    pub fn threads(&self) -> usize {
        self.pool.as_ref().map_or(1, ThreadPool::threads)
    }

    /// True when there is nothing to run work on but the calling thread.
    pub fn is_sequential(&self) -> bool {
        self.pool.is_none()
    }

    /// How this executor should be described in a run artifact (§19.3: *"publish exact
    /// hardware, backend, precision…"*).
    pub fn label(&self) -> String {
        match &self.pool {
            None => "cpu-scalar (1 thread)".to_string(),
            Some(pool) => format!(
                "cpu-parallel ({} threads, {} chunks/thread)",
                pool.threads(),
                self.chunks_per_thread
            ),
        }
    }

    /// The partition this executor would choose for `units` work units.
    ///
    /// Exposed so a caller can report it, and so tests can assert on it without running
    /// a kernel.
    pub fn partition(&self, units: usize, grain: Grain) -> Partition {
        Partition::new(units, self.threads() * self.chunks_per_thread, grain)
    }

    /// Run `f(offset, chunk)` over disjoint chunks of `items`.
    ///
    /// `offset` is the chunk's start index in `items`. See [`Grain`] for what the
    /// kernel is declaring.
    ///
    /// # Panics
    ///
    /// Re-raises the first panic from any chunk, on the calling thread.
    pub fn for_each_chunk_mut<T, F>(&self, items: &mut [T], grain: Grain, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        if items.is_empty() {
            return;
        }
        let per_chunk = self.partition(items.len(), grain).units_per_chunk();
        match &self.pool {
            Some(pool) => pool.for_each_chunk_mut(items, per_chunk, f),
            None => {
                for (index, chunk) in items.chunks_mut(per_chunk).enumerate() {
                    f(index * per_chunk, chunk);
                }
            }
        }
    }

    /// Run `f(first_row, band)` over disjoint bands of whole rows.
    ///
    /// `rows` is a row-major buffer whose rows are `stride` elements apart — a padded
    /// `ScalarField` row span, in practice. Splitting on row boundaries is what keeps a
    /// stencil's writes disjoint while its reads may reach into the neighbouring band:
    /// the halo rows a band reads belong to another band's *input*, and no band writes
    /// another's rows.
    ///
    /// `grain` is measured in *rows* — see [`Grain::per_row`] for converting a
    /// cell-denominated grain to one.
    ///
    /// # Panics
    ///
    /// If `stride` is zero, or `rows.len()` is not a multiple of `stride`. Both mean
    /// the caller has mis-described the buffer, and a silently truncated last row is a
    /// far worse outcome than a message.
    pub fn for_each_row_band_mut<T, F>(&self, rows: &mut [T], stride: usize, grain: Grain, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        assert!(stride > 0, "a row band needs a positive stride");
        assert_eq!(rows.len() % stride, 0, "the row span must hold whole rows");
        if rows.is_empty() {
            return;
        }

        let row_count = rows.len() / stride;
        let rows_per_band = self.partition(row_count, grain).units_per_chunk();
        let per_chunk = rows_per_band * stride;

        match &self.pool {
            Some(pool) => {
                pool.for_each_chunk_mut(rows, per_chunk, |offset, band| f(offset / stride, band));
            }
            None => {
                for (index, band) in rows.chunks_mut(per_chunk).enumerate() {
                    f(index * rows_per_band, band);
                }
            }
        }
    }
}

impl Default for Executor {
    /// Sequential, not automatic. A library that starts threads because it was
    /// default-constructed is a surprise; asking for [`Executor::automatic`] is not.
    fn default() -> Executor {
        Executor::sequential()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partition_covers_every_unit_exactly_once() {
        for units in [0usize, 1, 7, 64, 1000] {
            for wanted in [1usize, 3, 8] {
                let part = Partition::new(units, wanted, Grain::new(0, 1));
                let covered: usize = (0..part.count())
                    .map(|k| {
                        let start = k * part.units_per_chunk();
                        (start + part.units_per_chunk()).min(units) - start
                    })
                    .sum();
                assert_eq!(covered, units, "units={units} wanted={wanted}");
            }
        }
    }

    #[test]
    fn the_floor_stops_a_small_problem_from_being_split_at_all() {
        assert_eq!(Partition::new(999, 16, Grain::new(1000, 8)).count(), 1);
        assert_eq!(Partition::new(1000, 16, Grain::new(1000, 8)).count(), 16);
    }

    /// The trap the two numbers exist to avoid: a floor high enough to protect a small
    /// problem must not also cap a large one at a handful of chunks.
    #[test]
    fn a_high_floor_does_not_cap_a_large_problem() {
        let grain = Grain::new(65_536, 8_192);
        assert_eq!(Partition::new(16_384, 80, grain).count(), 1, "below the floor");
        assert_eq!(Partition::new(262_144, 80, grain).count(), 32, "well above it");
        // A single grain used for both jobs would have given only four chunks here.
        assert_eq!(Partition::new(262_144, 80, Grain::new(0, 65_536)).count(), 4);
    }

    #[test]
    fn the_chunk_size_stops_a_problem_from_being_cut_into_slivers() {
        let part = Partition::new(100, 16, Grain::new(0, 64));
        assert_eq!(part.count(), 2, "at most two chunks of 64 fit in 100");
    }

    #[test]
    fn a_large_problem_uses_every_chunk_it_was_offered() {
        let part = Partition::new(10_000, 8, Grain::new(0, 16));
        assert_eq!(part.count(), 8);
        assert_eq!(part.units_per_chunk(), 1250);
    }

    #[test]
    fn a_row_grain_converts_from_cells() {
        let cells = Grain::new(16_384, 4_096);
        assert_eq!(cells.per_row(256), Grain::new(64, 16));
        assert_eq!(cells.per_row(512), Grain::new(32, 8));
        // A 64-wide grid needs 256 rows before splitting is worth it, which is exactly
        // the point: a 64x64 grid holds 4096 cells and is not worth a barrier.
        assert_eq!(cells.per_row(64), Grain::new(256, 64));
    }

    #[test]
    fn the_default_ratio_keeps_a_problem_at_the_floor_useful() {
        assert_eq!(Grain::above(65_536), Grain::new(65_536, 8_192));
        assert_eq!(Partition::new(65_536, 80, Grain::above(65_536)).count(), 8);
    }

    #[test]
    fn a_sequential_executor_starts_no_threads() {
        let exec = Executor::sequential();
        assert!(exec.is_sequential());
        assert_eq!(exec.threads(), 1);
        assert_eq!(exec.label(), "cpu-scalar (1 thread)");
    }

    #[test]
    fn one_thread_is_the_sequential_executor() {
        assert!(Executor::with_threads(1).is_sequential());
        assert!(Executor::with_threads(0).is_sequential());
    }

    #[test]
    fn a_parallel_executor_reports_its_configuration() {
        let exec = Executor::with_threads(4).with_chunks_per_thread(2);
        assert_eq!(exec.threads(), 4);
        assert_eq!(exec.label(), "cpu-parallel (4 threads, 2 chunks/thread)");
    }

    /// The point of the whole crate, stated as a test: the same kernel over the same
    /// data gives the same bits at any thread count.
    #[test]
    fn an_elementwise_kernel_is_bit_identical_at_every_thread_count() {
        let kernel = |threads: usize| {
            let exec = Executor::with_threads(threads);
            let mut data: Vec<f64> = (0..5000).map(|i| (i as f64) * 0.001).collect();
            exec.for_each_chunk_mut(&mut data, Grain::new(0, 32), |_, chunk| {
                for value in chunk.iter_mut() {
                    *value = value.sin().exp() / 3.0 + 1e-17;
                }
            });
            data
        };
        let reference = kernel(1);
        for threads in [2usize, 3, 5, 8] {
            let bits: Vec<u64> = kernel(threads).iter().map(|v| v.to_bits()).collect();
            let expected: Vec<u64> = reference.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, expected, "{threads} threads changed the result");
        }
    }

    #[test]
    fn row_bands_split_on_row_boundaries() {
        let exec = Executor::with_threads(4);
        let (stride, rows) = (10usize, 40usize);
        let mut buffer = vec![0.0f64; stride * rows];
        exec.for_each_row_band_mut(&mut buffer, stride, Grain::new(0, 1), |first_row, band| {
            assert_eq!(band.len() % stride, 0, "a band must hold whole rows");
            for (offset, slot) in band.iter_mut().enumerate() {
                *slot = (first_row + offset / stride) as f64;
            }
        });
        for j in 0..rows {
            assert_eq!(buffer[j * stride], j as f64);
            assert_eq!(buffer[j * stride + stride - 1], j as f64);
        }
    }

    #[test]
    fn row_bands_are_the_same_at_every_thread_count() {
        let spans = |threads: usize| {
            let exec = Executor::with_threads(threads);
            let (stride, rows) = (8usize, 33usize);
            let mut buffer = vec![0.0f64; stride * rows];
            let seen = std::sync::Mutex::new(Vec::new());
            exec.for_each_row_band_mut(&mut buffer, stride, Grain::new(0, 1), |first_row, band| {
                seen.lock().unwrap().push((first_row, band.len() / stride));
            });
            let mut spans = seen.into_inner().unwrap();
            spans.sort_unstable();
            spans
        };
        // Four threads, four chunks each: 33 rows over 16 bands is 3 rows per band.
        assert_eq!(spans(4), vec![(0, 3), (3, 3), (6, 3), (9, 3), (12, 3), (15, 3), (18, 3), (21, 3), (24, 3), (27, 3), (30, 3)]);
        // A different thread count gives a different partition — but the same partition
        // whether or not a pool is behind it, which is what the kernels rely on.
        assert_eq!(spans(1), vec![(0, 33)]);
    }

    #[test]
    #[should_panic(expected = "whole rows")]
    fn a_ragged_row_span_is_rejected() {
        let exec = Executor::sequential();
        let mut buffer = vec![0.0f64; 25];
        exec.for_each_row_band_mut(&mut buffer, 10, Grain::new(0, 1), |_, _| {});
    }

    #[test]
    fn an_empty_kernel_is_a_no_op() {
        let exec = Executor::with_threads(4);
        let mut nothing: Vec<f64> = Vec::new();
        exec.for_each_chunk_mut(&mut nothing, Grain::new(0, 8), |_, _| panic!("should not run"));
        exec.for_each_row_band_mut(&mut nothing, 4, Grain::new(0, 1), |_, _| panic!("should not run"));
    }

    #[test]
    fn the_shared_sequential_executor_is_usable_from_anywhere() {
        let exec = Executor::shared_sequential();
        assert!(exec.is_sequential());
        let mut data = vec![1.0f64; 16];
        exec.for_each_chunk_mut(&mut data, Grain::new(0, 4), |_, chunk| chunk.fill(2.0));
        assert!(data.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn the_default_executor_does_not_start_threads() {
        assert!(Executor::default().is_sequential());
    }
}
