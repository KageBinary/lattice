//! A persistent worker pool.
//!
//! Spec §15.3: *"Rust core with explicit parallel iterators/task pools for independent
//! operations."* The emphasis is on **explicit**. Nothing here decides on its own that
//! a loop should be parallel; a caller asks for a partition and gets it.
//!
//! # What the design is fighting
//!
//! Not throughput — *latency*. A solver dispatches many times per step: velocity Verlet
//! alone splits into six passes over the particle arrays, and a coupled model steps
//! hundreds of times a second. Each of those dispatches has to reach every worker and
//! come back. Two costs dominate, and both were measured before they were designed
//! around:
//!
//! | Cost | Why it hurts | What is done about it |
//! |---|---|---|
//! | thread spawn | tens of microseconds, per dispatch | workers are persistent |
//! | waking a parked thread | ~10 µs on this platform, times the worker count | workers spin before parking |
//! | contending on a claim lock | tens of task claims per dispatch, all on one mutex | claiming is a compare-exchange, not a lock |
//!
//! [`std::thread::scope`] would give the same borrowing guarantees with no `unsafe` at
//! all, and was the first implementation. It pays the first row of that table on every
//! call. A 256² diffusion step is around a hundred microseconds, so per-call spawning
//! turned the intended speedup into a slowdown at exactly the sizes the spec's
//! interactive targets care about (§15.6).
//!
//! # The one unsafe invariant
//!
//! Persistent workers cannot borrow the caller's stack through the type system, so the
//! pool passes a type-erased pointer to the caller's closure and re-establishes the
//! borrow inside a shim. Exactly two facts make that sound, and both are enforced by
//! [`ThreadPool::dispatch`] rather than by convention:
//!
//! 1. **The pointer outlives every use.** `dispatch` publishes the job, and does not
//!    return until every task has been counted in `finished` and the job slot has been
//!    cleared. No worker can reach the pointer afterwards — including when a task
//!    panics, since a panicking task still counts as finished and its payload is
//!    re-raised on the calling thread.
//! 2. **Each task index is claimed exactly once.** Claiming compare-exchanges a global
//!    monotonic counter, so two workers can never receive the same index. That is what
//!    lets [`ThreadPool::for_each_chunk_mut`] hand out `&mut` sub-slices without
//!    aliasing: chunk *k* is produced from index *k*, and the ranges are disjoint by
//!    construction.
//!
//! ## Why the claim counter is never reset
//!
//! It is tempting to reset the counter to zero at each dispatch and have workers
//! `fetch_add` it. That is wrong in a way that only shows up under load. A worker that
//! has just finished the last task of dispatch *N* does one more claim to discover
//! there is no more work; if dispatch *N+1* has already published by then, that probe
//! either **runs a task of *N+1* against *N*'s stale closure pointer** — a
//! use-after-free — or silently **consumes an index of *N+1***, so *N+1* waits forever
//! for a task nobody will run.
//!
//! So the counter is global and monotonic, each job owns the half-open range
//! `base..end` carved from it, and the probe is a *load-then-compare-exchange* rather
//! than a `fetch_add`. A stale worker's load lands past its own `end`, it breaks
//! without consuming anything, and the next job's range is untouched.
//!
//! Everything in [`crate::exec`] is safe code written against these guarantees.

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// How long a worker looks for new work before parking.
///
/// A solver dispatches several times per step with almost no gap between them, so a
/// worker that parks the instant it runs dry pays a condition-variable wakeup — the most
/// expensive thing in the table above — for every one of those gaps. Spinning across the
/// gap turns the common case into a few hundred nanoseconds of polling.
///
/// The two phases matter. [`core::hint::spin_loop`] is for the sub-microsecond gap
/// between passes of one step; [`std::thread::yield_now`] is for the longer gap between
/// steps, and it hands the core back rather than burning it. Past both, the pool is
/// genuinely idle and parking is right — an idle pool must not keep a machine's fans on.
const SPIN_ROUNDS: u32 = 256;
/// Yields after [`SPIN_ROUNDS`] before parking. See there.
const YIELD_ROUNDS: u32 = 32;

thread_local! {
    /// True while this thread is running a pool task.
    ///
    /// A task that asks the pool for more work would deadlock — the driver lock is
    /// already held, and the single job slot is occupied. Rather than document that as
    /// a rule and wait for someone to break it, a nested request runs inline on the
    /// calling thread. The work still happens; only the parallelism is lost.
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

/// True if the calling thread is already executing pool work.
pub(crate) fn is_worker() -> bool {
    IN_POOL.with(Cell::get)
}

/// Sets [`IN_POOL`] for a scope and restores it even if the scope unwinds.
struct InPoolGuard(bool);

impl InPoolGuard {
    fn enter() -> InPoolGuard {
        InPoolGuard(IN_POOL.with(|flag| flag.replace(true)))
    }
}

impl Drop for InPoolGuard {
    fn drop(&mut self) {
        IN_POOL.with(|flag| flag.set(self.0));
    }
}

/// A type-erased pointer to the caller's work, plus the shim that restores its type.
///
/// A thin data pointer and a function pointer, rather than a `*const dyn Trait`, so no
/// fat-pointer lifetime transmute is involved. The lifetime is carried by
/// [`ThreadPool::dispatch`]'s signature and enforced by its blocking behaviour.
#[derive(Clone, Copy)]
struct TaskPtr {
    data: *const (),
    run: unsafe fn(*const (), usize),
}

// SAFETY: `data` points at a `W: Sync` value owned by the thread inside `dispatch`,
// which does not return until every task that could observe the pointer has finished.
// Sharing a `&W` across threads is exactly what `Sync` permits.
unsafe impl Send for TaskPtr {}
unsafe impl Sync for TaskPtr {}

/// Restore `W` from an erased pointer and run one task index.
///
/// # Safety
///
/// `data` must have come from `&W` and must still be live.
unsafe fn run_shim<W: Task>(data: *const (), index: usize) {
    // SAFETY: the caller guarantees the provenance and liveness of `data`.
    let work: &W = unsafe { &*data.cast::<W>() };
    work.run(index);
}

/// Work that can be split into independently executable indices.
///
/// `Sync` because every worker calls [`Task::run`] through a shared reference.
trait Task: Sync {
    /// Run one task. Called exactly once for each index in `0..total`.
    fn run(&self, index: usize);
}

/// One dispatch, as a worker sees it.
///
/// `Copy` so a worker can take it out from under the lock once and then work entirely
/// against atomics.
#[derive(Clone, Copy)]
struct Job {
    task: TaskPtr,
    /// First value of the global claim counter belonging to this job.
    base: usize,
    /// One past the last. `end - base` is the task count.
    end: usize,
}

struct State {
    job: Option<Job>,
    /// Bumped on every publish. Duplicated into [`Shared::generation`] so a spinning
    /// worker can notice new work without taking the lock.
    generation: u64,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    /// Signalled when a job is published or shutdown is requested.
    wake: Condvar,
    /// Signalled when the last task of a job finishes.
    done: Condvar,
    /// Global monotonic claim counter. Never reset — see the module docs.
    claimed: AtomicUsize,
    /// Global monotonic completion counter, in the same units as `claimed`.
    finished: AtomicUsize,
    /// Lock-free mirror of `State::generation`.
    generation: AtomicU64,
    /// The first panic payload of the current dispatch, re-raised by `dispatch`.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

/// Lock without caring about poisoning.
///
/// A task's panic is caught before it can escape into a locked region, so the only way
/// to poison these mutexes is a bug in the pool itself. Recovering the guard keeps that
/// bug from turning into a second, less informative panic on every other thread.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A fixed set of worker threads.
///
/// Construct one per process — or per [`crate::Executor`] — and keep it. Creating one
/// per step would reintroduce the spawn cost the pool exists to avoid.
pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Held for the whole of a dispatch. The pool has one job slot, so two threads
    /// driving it at once would corrupt each other's bookkeeping; this makes the second
    /// one wait instead.
    driver: Mutex<()>,
}

impl core::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ThreadPool").field("threads", &self.threads()).finish()
    }
}

impl ThreadPool {
    /// Start a pool with `extra` worker threads.
    ///
    /// The calling thread also runs tasks, so the total number of threads doing work is
    /// `extra + 1`. `extra == 0` is legal and means everything runs on the caller.
    pub fn new(extra: usize) -> ThreadPool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, generation: 0, shutdown: false }),
            wake: Condvar::new(),
            done: Condvar::new(),
            claimed: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            panic: Mutex::new(None),
        });

        let workers = (0..extra)
            .map(|index| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("lattice-cpu-{index}"))
                    .spawn(move || worker_loop(&shared))
                    .expect("the operating system refused a worker thread")
            })
            .collect();

        ThreadPool { shared, workers, driver: Mutex::new(()) }
    }

    /// Threads that execute tasks, counting the calling thread.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Split `items` into contiguous chunks of `chunk_len` and run `f` on each.
    ///
    /// `f` receives the chunk's starting index in `items` and the chunk itself. The
    /// final chunk is short when `chunk_len` does not divide the length.
    ///
    /// The partition is a pure function of `items.len()` and `chunk_len` — it does not
    /// depend on how many threads happen to be free — so an elementwise or stencil
    /// kernel written against it produces bit-identical output at any thread count.
    /// That is the property [`crate::Executor`] builds its determinism claim on.
    ///
    /// # Panics
    ///
    /// Re-raises the first panic from any task, on the calling thread, after every
    /// other task has finished. If `chunk_len` is zero.
    pub fn for_each_chunk_mut<T, F>(&self, items: &mut [T], chunk_len: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        assert!(chunk_len > 0, "chunk length must be positive");
        let len = items.len();
        if len == 0 {
            return;
        }

        let chunks = len.div_ceil(chunk_len);
        if chunks == 1 || self.workers.is_empty() || is_worker() {
            // The same partition, walked in order. Running the whole slice as one chunk
            // would be faster here and would break the promise above: a reduction that
            // summed per chunk would then depend on whether the pool had workers.
            for (index, chunk) in items.chunks_mut(chunk_len).enumerate() {
                f(index * chunk_len, chunk);
            }
            return;
        }

        let work = ChunkWork { base: items.as_mut_ptr(), len, chunk_len, f };
        self.dispatch(chunks, &work);
    }

    /// Publish `total` tasks and run them to completion.
    fn dispatch<W: Task>(&self, total: usize, work: &W) {
        let _driver = lock(&self.driver);
        let task = TaskPtr { data: (work as *const W).cast::<()>(), run: run_shim::<W> };

        // The previous dispatch left the counter somewhere at or past its own end; this
        // one starts from wherever that was. Monotonic, never reset — see the module
        // docs for what resetting would break.
        let base = self.shared.claimed.load(Ordering::Acquire);
        let job = Job { task, base, end: base + total };
        self.shared.finished.store(base, Ordering::Release);
        {
            let mut state = lock(&self.shared.state);
            state.job = Some(job);
            state.generation += 1;
            self.shared.generation.store(state.generation, Ordering::Release);
        }
        self.shared.wake.notify_all();

        // The calling thread is a worker too. With `extra + 1` threads and `extra + 1`
        // chunks, leaving the caller idle would waste a whole core and cap the speedup
        // one short.
        let _in_pool = InPoolGuard::enter();
        run_claims(&self.shared, &job);

        let payload = {
            let mut state = lock(&self.shared.state);
            while self.shared.finished.load(Ordering::Acquire) < job.end {
                state = self
                    .shared
                    .done
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            // Clearing the slot is what ends the borrow: after this, no worker can
            // reach `task.data` again.
            state.job = None;
            drop(state);
            lock(&self.shared.panic).take()
        };

        drop(_in_pool);
        if let Some(payload) = payload {
            panic::resume_unwind(payload);
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut state = lock(&self.shared.state);
            state.shutdown = true;
        }
        self.shared.wake.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Claim and run tasks until this job is exhausted.
///
/// The probe is a load followed by a compare-exchange rather than a `fetch_add`, so the
/// final probe — the one that discovers there is no work left — consumes nothing. A
/// worker still in this loop when the *next* dispatch publishes therefore cannot
/// puncture that dispatch's index range, and cannot run one of its tasks against this
/// job's stale closure pointer.
fn run_claims(shared: &Shared, job: &Job) {
    loop {
        let current = shared.claimed.load(Ordering::Relaxed);
        if current >= job.end {
            return;
        }
        if shared
            .claimed
            .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            continue;
        }

        // SAFETY: `dispatch` published this job and cannot return until this call has
        // been counted in `finished`, so the pointee is still live. The index was won
        // by exactly one compare-exchange, so no other thread is running it.
        // `AssertUnwindSafe` is justified because the payload is re-raised on the
        // calling thread rather than swallowed: the caller never observes half-updated
        // state without also seeing the panic that produced it.
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            (job.task.run)(job.task.data, current - job.base)
        }));

        if let Err(payload) = outcome {
            let mut slot = lock(&shared.panic);
            if slot.is_none() {
                *slot = Some(payload);
            }
        }

        if shared.finished.fetch_add(1, Ordering::AcqRel) + 1 == job.end {
            // Taking the state lock before notifying is what closes the window between
            // the waiter's predicate check and its wait — both happen while it holds
            // this lock, so the notification cannot be lost.
            drop(lock(&shared.state));
            shared.done.notify_all();
        }
    }
}

fn worker_loop(shared: &Arc<Shared>) {
    let _in_pool = InPoolGuard::enter();
    let mut seen = 0u64;

    loop {
        // Look for new work without touching the lock. In a stepping solver the next
        // dispatch is usually only hundreds of nanoseconds away, and paying a condvar
        // wakeup for that gap is what made the first version of this pool slower than
        // no pool at all on the particle benchmarks.
        let mut spun = 0u32;
        while shared.generation.load(Ordering::Acquire) == seen {
            if spun < SPIN_ROUNDS {
                core::hint::spin_loop();
            } else if spun < SPIN_ROUNDS + YIELD_ROUNDS {
                std::thread::yield_now();
            } else {
                break;
            }
            spun += 1;
        }

        let job = {
            let mut state = lock(&shared.state);
            loop {
                if state.shutdown {
                    return;
                }
                if state.generation != seen {
                    seen = state.generation;
                    if let Some(job) = state.job {
                        break job;
                    }
                }
                state = shared.wake.wait(state).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };
        run_claims(shared, &job);
    }
}

/// The only [`Task`] implementation: chunk `k` of a mutable slice.
struct ChunkWork<T, F> {
    base: *mut T,
    len: usize,
    chunk_len: usize,
    f: F,
}

// SAFETY: `base` is derived from the `&mut [T]` borrowed by `for_each_chunk_mut`, which
// outlives the dispatch. `T: Send` because a chunk is used from another thread, and
// `F: Sync` because every worker calls it through a shared reference.
unsafe impl<T: Send, F: Sync> Sync for ChunkWork<T, F> {}

impl<T: Send, F: Fn(usize, &mut [T]) + Sync> Task for ChunkWork<T, F> {
    fn run(&self, index: usize) {
        let start = index * self.chunk_len;
        let end = (start + self.chunk_len).min(self.len);
        // SAFETY: `run_claims` calls this exactly once per index, so no other thread is
        // producing this range; ranges for different indices are disjoint by
        // construction; and `start < end <= len` because `dispatch` was given
        // `len.div_ceil(chunk_len)` indices.
        let chunk = unsafe { core::slice::from_raw_parts_mut(self.base.add(start), end - start) };
        (self.f)(start, chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn every_element_is_visited_exactly_once() {
        let pool = ThreadPool::new(3);
        let mut data = vec![0usize; 1000];
        pool.for_each_chunk_mut(&mut data, 64, |start, chunk| {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = start + offset;
            }
        });
        assert!(data.iter().enumerate().all(|(i, &v)| i == v));
    }

    /// The chunk boundaries follow from the length and the chunk size alone, which is
    /// what makes a parallel kernel's output independent of the thread count.
    #[test]
    fn the_partition_does_not_depend_on_the_thread_count() {
        let record = |threads: usize| {
            let pool = ThreadPool::new(threads);
            let mut data = vec![0usize; 250];
            let seen = Mutex::new(Vec::new());
            pool.for_each_chunk_mut(&mut data, 32, |start, chunk| {
                lock(&seen).push((start, chunk.len()));
            });
            let mut spans = seen.into_inner().unwrap();
            spans.sort_unstable();
            spans
        };
        let expected =
            vec![(0, 32), (32, 32), (64, 32), (96, 32), (128, 32), (160, 32), (192, 32), (224, 26)];
        assert_eq!(record(0), expected);
        assert_eq!(record(1), expected);
        assert_eq!(record(7), expected);
    }

    #[test]
    fn a_short_slice_runs_as_one_chunk() {
        let pool = ThreadPool::new(4);
        let mut data = vec![1.0f64; 10];
        let calls = AtomicUsize::new(0);
        pool.for_each_chunk_mut(&mut data, 64, |start, chunk| {
            calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(start, 0);
            assert_eq!(chunk.len(), 10);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn an_empty_slice_does_nothing() {
        let pool = ThreadPool::new(2);
        let mut data: Vec<f64> = Vec::new();
        pool.for_each_chunk_mut(&mut data, 8, |_, _| panic!("should not be called"));
    }

    #[test]
    fn work_is_actually_spread_across_threads() {
        let pool = ThreadPool::new(3);
        assert_eq!(pool.threads(), 4);
        let mut data = vec![0u64; 4096];
        let threads = Mutex::new(std::collections::HashSet::new());
        pool.for_each_chunk_mut(&mut data, 64, |_, chunk| {
            lock(&threads).insert(std::thread::current().id());
            // Enough work that the first worker cannot plausibly take every chunk.
            for slot in chunk.iter_mut() {
                *slot = (0..2000u64).sum();
            }
        });
        assert!(threads.into_inner().unwrap().len() > 1, "only one thread ran any chunk");
    }

    /// A panicking task must not strand the dispatch, and must reach the caller intact.
    #[test]
    fn a_panicking_task_is_re_raised_on_the_caller() {
        let pool = ThreadPool::new(3);
        let mut data = vec![0u32; 512];
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            pool.for_each_chunk_mut(&mut data, 32, |start, _| {
                assert!(start != 128, "deliberate failure in chunk 4");
            });
        }));
        let payload = result.expect_err("the panic should have propagated");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("");
        assert!(message.contains("deliberate failure"), "payload was {message:?}");
    }

    /// ...and the pool is still usable afterwards, because the job slot and the panic
    /// slot were both cleared.
    #[test]
    fn the_pool_survives_a_panicking_task() {
        let pool = ThreadPool::new(2);
        let mut data = vec![0u32; 256];
        let _ = panic::catch_unwind(AssertUnwindSafe(|| {
            pool.for_each_chunk_mut(&mut data, 16, |_, _| panic!("boom"));
        }));
        let mut again = vec![0u32; 256];
        pool.for_each_chunk_mut(&mut again, 16, |start, chunk| {
            chunk.fill(start as u32);
        });
        assert_eq!(again[0], 0);
        assert_eq!(again[16], 16);
    }

    /// Nesting would deadlock on the driver lock. It runs inline instead.
    #[test]
    fn a_nested_dispatch_runs_inline_rather_than_deadlocking() {
        let pool = ThreadPool::new(3);
        let mut outer = vec![0usize; 128];
        pool.for_each_chunk_mut(&mut outer, 16, |start, chunk| {
            assert!(is_worker(), "a task runs with the in-pool flag set");
            let mut inner = vec![0usize; chunk.len()];
            pool.for_each_chunk_mut(&mut inner, 1, |k, slot| slot[0] = start + k);
            chunk.copy_from_slice(&inner);
        });
        assert!(outer.iter().enumerate().all(|(i, &v)| i == v));
        assert!(!is_worker(), "the flag is cleared once the dispatch returns");
    }

    #[test]
    fn a_pool_with_no_workers_still_runs_everything() {
        let pool = ThreadPool::new(0);
        assert_eq!(pool.threads(), 1);
        let mut data = vec![0usize; 100];
        pool.for_each_chunk_mut(&mut data, 8, |start, chunk| {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = start + offset;
            }
        });
        assert!(data.iter().enumerate().all(|(i, &v)| i == v));
    }

    /// The back-to-back case the whole design is for, and the one that would expose the
    /// stale-claim bug the module docs describe: many short dispatches in a row, with
    /// workers still winding down from the previous one as the next is published. A
    /// consumed-but-unrun index would hang this test; a stale pointer would corrupt it.
    #[test]
    fn many_short_dispatches_in_a_row_stay_correct() {
        let pool = ThreadPool::new(7);
        let mut data = vec![0u64; 777];
        for round in 0..2000u64 {
            pool.for_each_chunk_mut(&mut data, 16, |_, chunk| {
                for slot in chunk.iter_mut() {
                    *slot += 1;
                }
            });
            assert_eq!(data[0], round + 1);
        }
        assert!(data.iter().all(|&v| v == 2000));
    }

    /// Dispatches separated by long enough that every worker has parked. The wakeup
    /// path is the one the spin phase skips, so it needs its own exercise.
    #[test]
    fn a_dispatch_after_the_workers_have_parked_still_runs() {
        let pool = ThreadPool::new(4);
        for _ in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let mut data = vec![0usize; 512];
            pool.for_each_chunk_mut(&mut data, 32, |start, chunk| {
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    *slot = start + offset;
                }
            });
            assert!(data.iter().enumerate().all(|(i, &v)| i == v));
        }
    }

    #[test]
    #[should_panic(expected = "chunk length must be positive")]
    fn a_zero_chunk_length_is_rejected() {
        let pool = ThreadPool::new(1);
        let mut data = vec![0u8; 4];
        pool.for_each_chunk_mut(&mut data, 0, |_, _| {});
    }
}
