//! Scratch arenas: preallocated per-step temporary storage.
//!
//! **NFR-001**: *"No heap allocation in validated hot loops except explicitly
//! profiled scratch arenas."* Solvers constantly need temporaries — a residual
//! vector, a force accumulator, a neighbour count — and allocating them each step
//! puts the allocator in the hot path and makes step timing depend on heap state.
//!
//! An [`Arena`] owns one `Vec<f64>` sized during model compilation (spec §8.4 step 6,
//! "plan buffers, alignment, structure-of-arrays layouts, scratch arenas"). Each step
//! opens a [`Frame`], bump-allocates disjoint slices from it, and drops the frame,
//! which resets the cursor. No allocator call happens in between.
//!
//! # How multiple live mutable slices are safe
//!
//! [`Frame::alloc`] repeatedly `split_at_mut`s the remaining buffer and hands back the
//! head, keeping the tail. Because the splits are disjoint by construction, every
//! returned slice can borrow for the frame's full lifetime simultaneously. No `unsafe`
//! is involved.
//!
//! ```
//! use lattice_ir::Arena;
//!
//! let mut arena = Arena::with_capacity(1024);
//! {
//!     let mut frame = arena.frame();
//!     let forces = frame.alloc_zeroed(128);
//!     let residual = frame.alloc_zeroed(128);
//!     forces[0] = 1.0;
//!     residual[0] = 2.0;              // both slices live at once
//!     assert_eq!(forces[0] + residual[0], 3.0);
//! }
//! assert_eq!(arena.high_water(), 256); // frame reset, peak usage recorded
//! ```

use core::fmt;

/// A preallocated block of `f64` scratch space.
#[derive(Debug)]
pub struct Arena {
    buffer: Vec<f64>,
    /// Largest number of elements handed out by any single frame so far.
    ///
    /// The benchmark harness reports this so an over-sized arena shows up as waste
    /// and an under-sized one shows up before it panics in production.
    high_water: usize,
}

impl Arena {
    /// Allocate an arena holding `capacity` `f64` values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { buffer: vec![0.0; capacity], high_water: 0 }
    }

    /// Total capacity in `f64` elements.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Peak elements used by a single frame since construction (or the last
    /// [`Arena::reset_high_water`]).
    pub fn high_water(&self) -> usize {
        self.high_water
    }

    /// Peak usage as a fraction of capacity, for the benchmark report.
    pub fn utilization(&self) -> f64 {
        if self.buffer.is_empty() {
            0.0
        } else {
            self.high_water as f64 / self.buffer.len() as f64
        }
    }

    /// Forget the recorded peak.
    pub fn reset_high_water(&mut self) {
        self.high_water = 0;
    }

    /// Open a frame. Dropping it releases everything allocated from it.
    pub fn frame(&mut self) -> Frame<'_> {
        Frame { remaining: &mut self.buffer, used: 0, high_water: &mut self.high_water }
    }

    /// Grow the arena to hold at least `capacity` elements.
    ///
    /// This is a *planning* operation, not something a solver may call mid-step. It
    /// exists so the compiler can size the arena once it knows the model, and so a
    /// benchmark can right-size it after observing the high-water mark.
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.buffer.len() {
            self.buffer.resize(capacity, 0.0);
        }
    }
}

/// A bump cursor over an [`Arena`], live for the duration of one step.
pub struct Frame<'a> {
    remaining: &'a mut [f64],
    used: usize,
    high_water: &'a mut usize,
}

impl<'a> Frame<'a> {
    /// Carve off `n` elements. Contents are whatever the previous frame left.
    ///
    /// # Panics
    ///
    /// If the arena does not have `n` elements left. Arena size is a compile-time
    /// decision, so exhaustion means the buffer plan is wrong — a bug to fix, not a
    /// condition to handle. The message reports both numbers so the fix is obvious.
    /// Use [`Frame::try_alloc`] where exhaustion is genuinely recoverable.
    pub fn alloc(&mut self, n: usize) -> &'a mut [f64] {
        match self.try_alloc(n) {
            Some(slice) => slice,
            None => panic!(
                "scratch arena exhausted: requested {n} more elements with {} of {} used; \
                 increase the arena in the buffer plan",
                self.used,
                self.used + self.remaining.len()
            ),
        }
    }

    /// Carve off `n` elements and zero them.
    pub fn alloc_zeroed(&mut self, n: usize) -> &'a mut [f64] {
        let slice = self.alloc(n);
        slice.fill(0.0);
        slice
    }

    /// Carve off `n` elements and fill with `value`.
    pub fn alloc_filled(&mut self, n: usize, value: f64) -> &'a mut [f64] {
        let slice = self.alloc(n);
        slice.fill(value);
        slice
    }

    /// Carve off `n` elements, or `None` if the arena is exhausted.
    pub fn try_alloc(&mut self, n: usize) -> Option<&'a mut [f64]> {
        if n > self.remaining.len() {
            return None;
        }
        // `mem::take` leaves an empty slice behind, which lets the head escape with
        // the frame's lifetime while the tail stays owned by the cursor.
        let remaining = core::mem::take(&mut self.remaining);
        let (head, tail) = remaining.split_at_mut(n);
        self.remaining = tail;
        self.used += n;
        if self.used > *self.high_water {
            *self.high_water = self.used;
        }
        Some(head)
    }

    /// Elements handed out by this frame so far.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Elements still available.
    pub fn available(&self) -> usize {
        self.remaining.len()
    }
}

impl fmt::Debug for Frame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("used", &self.used)
            .field("available", &self.remaining.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_slices_are_disjoint_and_simultaneously_live() {
        let mut arena = Arena::with_capacity(64);
        let mut frame = arena.frame();
        let a = frame.alloc_zeroed(16);
        let b = frame.alloc_zeroed(16);
        let c = frame.alloc_filled(16, 7.0);

        a.fill(1.0);
        b.fill(2.0);
        assert_eq!(a[0], 1.0);
        assert_eq!(b[0], 2.0);
        assert_eq!(c[0], 7.0);
        assert_eq!(frame.used(), 48);
        assert_eq!(frame.available(), 16);
    }

    #[test]
    fn dropping_a_frame_releases_everything() {
        let mut arena = Arena::with_capacity(32);
        for _ in 0..1000 {
            let mut frame = arena.frame();
            let s = frame.alloc(32);
            s[0] = 1.0;
        }
        // A thousand steps, one allocation total.
        assert_eq!(arena.capacity(), 32);
        assert_eq!(arena.high_water(), 32);
    }

    #[test]
    fn high_water_tracks_the_largest_frame() {
        let mut arena = Arena::with_capacity(100);
        {
            let mut f = arena.frame();
            f.alloc(10);
        }
        assert_eq!(arena.high_water(), 10);
        {
            let mut f = arena.frame();
            f.alloc(60);
        }
        assert_eq!(arena.high_water(), 60);
        {
            let mut f = arena.frame();
            f.alloc(20);
        }
        assert_eq!(arena.high_water(), 60, "high water must not shrink");
        assert!((arena.utilization() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn try_alloc_reports_exhaustion() {
        let mut arena = Arena::with_capacity(8);
        let mut frame = arena.frame();
        assert!(frame.try_alloc(8).is_some());
        assert!(frame.try_alloc(1).is_none());
    }

    #[test]
    #[should_panic(expected = "scratch arena exhausted")]
    fn alloc_panics_with_an_actionable_message() {
        let mut arena = Arena::with_capacity(4);
        let mut frame = arena.frame();
        frame.alloc(8);
    }

    #[test]
    fn zeroed_allocation_clears_previous_contents() {
        let mut arena = Arena::with_capacity(16);
        {
            let mut f = arena.frame();
            f.alloc_filled(16, 42.0);
        }
        {
            let mut f = arena.frame();
            let s = f.alloc_zeroed(16);
            assert!(s.iter().all(|&v| v == 0.0), "stale data leaked into a zeroed frame");
        }
    }

    #[test]
    fn ensure_capacity_grows_but_never_shrinks() {
        let mut arena = Arena::with_capacity(10);
        arena.ensure_capacity(50);
        assert_eq!(arena.capacity(), 50);
        arena.ensure_capacity(20);
        assert_eq!(arena.capacity(), 50);
    }
}
