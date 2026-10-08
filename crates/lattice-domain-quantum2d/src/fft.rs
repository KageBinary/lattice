//! Fast Fourier transforms, one- and two-dimensional.
//!
//! Split-step Fourier propagation is two FFTs per step, so this is the inner loop of the
//! whole module. Two algorithms cover every length:
//!
//! - **radix-2**, iterative and in place, for powers of two — the fast path, and the
//!   one the grid sizes in this module's examples are chosen for;
//! - **Bluestein's chirp-z**, which rewrites a transform of any length `n` as a
//!   circular convolution of length `m ≥ 2n − 1`, a power of two. Spec §25.2's grid is
//!   768 wide, and a module that could only run powers of two would be refusing the
//!   spec's own example for an implementation reason.
//!
//! The forward transform is unnormalized, `X_k = Σ_j x_j e^{−2πi jk/n}`; the inverse
//! carries the `1/n`, so `inverse(forward(x)) = x`.
//!
//! # Accuracy
//!
//! Twiddle factors are computed directly as `e^{−2πik/n}` rather than by repeated
//! multiplication, so their error is one rounding each rather than one per stage.
//! Bluestein's chirp angle `πj²/n` is reduced modulo `2π` *in integers* — as
//! `j² mod 2n` — before the floating-point multiply, because at `j = 700` the
//! unreduced angle is already ~2000 radians and would carry eleven bits of error into
//! every chirp.

use lattice_ir::{Executor, Grain};

use crate::complex::Complex;

/// A one-dimensional transform plan for a fixed length.
#[derive(Clone, Debug)]
pub struct Fft {
    n: usize,
    kind: Kind,
}

#[derive(Clone, Debug)]
enum Kind {
    /// Lengths 0 and 1, which are their own transforms.
    Trivial,
    Radix2 {
        /// Each stage's twiddles laid end to end: for the stage that combines
        /// half-length `h`, the `h` values `e^{−2πik/2h}`, starting at offset `h − 1`.
        /// The same values a single `n/2` table indexed at stride `n/2h` holds — so the
        /// same bits — read contiguously instead of a cache line per butterfly.
        forward: Vec<Complex>,
        /// `forward`, conjugated.
        inverse: Vec<Complex>,
        /// The bit-reversal permutation.
        reversed: Vec<u32>,
    },
    Bluestein {
        /// `w_j = e^{−iπj²/n}` for `j < n`.
        chirp: Vec<Complex>,
        /// The forward transform of the convolution kernel `conj(w)`, wrapped to `m`.
        kernel: Vec<Complex>,
        /// The power-of-two transform the convolution runs on.
        inner: Box<Fft>,
    },
}

impl Fft {
    /// A plan for transforms of length `n`.
    pub fn new(n: usize) -> Self {
        let kind = if n <= 1 {
            Kind::Trivial
        } else if n.is_power_of_two() {
            let twiddles: Vec<Complex> = (0..n / 2)
                .map(|k| Complex::cis(-core::f64::consts::TAU * k as f64 / n as f64))
                .collect();
            let mut forward = Vec::with_capacity(n - 1);
            let mut half = 1;
            while half < n {
                let stride = n / (2 * half);
                forward.extend((0..half).map(|k| twiddles[k * stride]));
                half *= 2;
            }
            let inverse = forward.iter().map(|w| w.conj()).collect();
            let bits = n.trailing_zeros();
            let reversed = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
            Kind::Radix2 { forward, inverse, reversed }
        } else {
            let m = (2 * n - 1).next_power_of_two();
            let modulus = 2 * n as u64;
            let chirp: Vec<Complex> = (0..n as u64)
                .map(|j| {
                    let reduced = (j * j) % modulus;
                    Complex::cis(-core::f64::consts::PI * reduced as f64 / n as f64)
                })
                .collect();
            let inner = Fft::new(m);
            let mut kernel = vec![Complex::ZERO; m];
            kernel[0] = chirp[0].conj();
            for j in 1..n {
                kernel[j] = chirp[j].conj();
                kernel[m - j] = chirp[j].conj();
            }
            inner.forward(&mut kernel, &mut []);
            Kind::Bluestein { chirp, kernel, inner: Box::new(inner) }
        };
        Self { n, kind }
    }

    /// The transform length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the length is zero.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Scratch the transforms need, in elements. Zero for a power of two.
    pub fn scratch_len(&self) -> usize {
        match &self.kind {
            Kind::Bluestein { inner, .. } => inner.len(),
            _ => 0,
        }
    }

    /// Forward transform in place, unnormalized.
    ///
    /// # Panics
    ///
    /// If `data` is not exactly the plan's length, or `scratch` is shorter than
    /// [`Fft::scratch_len`].
    pub fn forward(&self, data: &mut [Complex], scratch: &mut [Complex]) {
        assert_eq!(data.len(), self.n, "an FFT plan for {} was handed {} values", self.n, data.len());
        match &self.kind {
            Kind::Trivial => {}
            Kind::Radix2 { forward, reversed, .. } => radix2(data, forward, reversed),
            Kind::Bluestein { chirp, kernel, inner } => bluestein(data, chirp, kernel, inner, scratch),
        }
    }

    /// Inverse transform in place, including the `1/n`.
    ///
    /// # Panics
    ///
    /// As [`Fft::forward`].
    pub fn inverse(&self, data: &mut [Complex], scratch: &mut [Complex]) {
        assert_eq!(data.len(), self.n, "an FFT plan for {} was handed {} values", self.n, data.len());
        match &self.kind {
            Kind::Trivial => return,
            Kind::Radix2 { inverse, reversed, .. } => radix2(data, inverse, reversed),
            Kind::Bluestein { chirp, kernel, inner } => {
                // inverse(x) = conj(forward(conj(x))) / n
                for z in data.iter_mut() {
                    *z = z.conj();
                }
                bluestein(data, chirp, kernel, inner, scratch);
                for z in data.iter_mut() {
                    *z = z.conj();
                }
            }
        }
        let scale = 1.0 / self.n as f64;
        for z in data.iter_mut() {
            *z = z.scale(scale);
        }
    }
}

/// The iterative Cooley–Tukey butterfly network over per-stage twiddle tables — the
/// conjugated ones for the inverse, which leaves the scaling to the caller.
fn radix2(data: &mut [Complex], stages: &[Complex], reversed: &[u32]) {
    let n = data.len();
    for (i, &r) in reversed.iter().enumerate() {
        let r = r as usize;
        if i < r {
            data.swap(i, r);
        }
    }
    let mut half = 1;
    while half < n {
        let twiddles = &stages[half - 1..2 * half - 1];
        for block in data.chunks_exact_mut(2 * half) {
            let (low, high) = block.split_at_mut(half);
            for ((a, b), w) in low.iter_mut().zip(high.iter_mut()).zip(twiddles) {
                let top = *a;
                let bottom = *b * *w;
                *a = top + bottom;
                *b = top - bottom;
            }
        }
        half *= 2;
    }
}

/// `X_k = w_k · Σ_j (x_j w_j) · conj(w_{k−j})`, the convolution done by a power-of-two
/// transform of length `m`.
fn bluestein(data: &mut [Complex], chirp: &[Complex], kernel: &[Complex], inner: &Fft, scratch: &mut [Complex]) {
    let m = inner.len();
    assert!(scratch.len() >= m, "Bluestein needs {m} scratch values, was handed {}", scratch.len());
    let work = &mut scratch[..m];
    for (slot, (x, w)) in work.iter_mut().zip(data.iter().zip(chirp)) {
        *slot = *x * *w;
    }
    work[data.len()..].fill(Complex::ZERO);
    inner.forward(work, &mut []);
    for (a, b) in work.iter_mut().zip(kernel) {
        *a *= *b;
    }
    inner.inverse(work, &mut []);
    for ((x, w), c) in data.iter_mut().zip(chirp).zip(work.iter()) {
        *x = *c * *w;
    }
}

/// How much transform work is worth a dispatch, in cells.
///
/// A row transform is `O(n log n)` flops rather than a stencil's handful per cell, so the
/// floor sits well below the grid crate's `BAND_GRAIN`. Measured with no floor at all on a
/// 20-thread laptop: a 32² transform (1k cells) ran a quarter slower split across the
/// pool, a 64² one (4k cells) 1.5× faster, and 128² 1.6×.
pub const FFT_GRAIN: Grain = Grain::new(4_096, 512);

/// A two-dimensional transform over a row-major `nx × ny` array.
///
/// Rows are transformed in place. Columns are transposed into a second buffer, where
/// they are rows, transformed there, and transposed back — two copies of the array per
/// transform, the same traffic the column-at-a-time gather it replaced paid, and it
/// turns the column pass into independent contiguous rows.
///
/// # Parallelism
///
/// [`Fft2::forward_with`] splits both passes, and both transposes, into bands of rows
/// across an [`Executor`]. Every row's transform is the same arithmetic on the same
/// inputs whichever thread runs it and however the rows are banded, and a transpose
/// is a copy, so the result is bit-for-bit the sequential one (`docs/execution.md`).
/// There is no reduction anywhere in an FFT to make that promise expensive.
#[derive(Clone, Debug)]
pub struct Fft2 {
    nx: usize,
    ny: usize,
    rows: Fft,
    columns: Fft,
    /// The array column-major, `ny`-long rows one per column, during the column pass.
    transposed: Vec<Complex>,
}

thread_local! {
    /// Bluestein's convolution buffer, one per thread that runs transforms, so a band
    /// running on a worker needs no allocation once that worker has run one.
    static SCRATCH: core::cell::RefCell<Vec<Complex>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with at least `len` elements of this thread's scratch.
fn with_scratch(len: usize, f: impl FnOnce(&mut [Complex])) {
    if len == 0 {
        return f(&mut []);
    }
    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < len {
            scratch.resize(len, Complex::ZERO);
        }
        f(&mut scratch[..len]);
    });
}

impl Fft2 {
    /// A plan for `nx × ny` arrays.
    pub fn new(nx: usize, ny: usize) -> Self {
        Self { nx, ny, rows: Fft::new(nx), columns: Fft::new(ny), transposed: vec![Complex::ZERO; nx * ny] }
    }

    /// The array shape, `[nx, ny]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.nx, self.ny]
    }

    /// Forward transform in place, on the calling thread.
    ///
    /// # Panics
    ///
    /// If `data` is not `nx × ny` long.
    pub fn forward(&mut self, data: &mut [Complex]) {
        self.transform(data, false, Executor::shared_sequential());
    }

    /// Inverse transform in place, including the `1/(nx·ny)`, on the calling thread.
    ///
    /// # Panics
    ///
    /// If `data` is not `nx × ny` long.
    pub fn inverse(&mut self, data: &mut [Complex]) {
        self.transform(data, true, Executor::shared_sequential());
    }

    /// [`Fft2::forward`], split across `executor`. The same bits either way.
    pub fn forward_with(&mut self, data: &mut [Complex], executor: &Executor) {
        self.transform(data, false, executor);
    }

    /// [`Fft2::inverse`], split across `executor`. The same bits either way.
    pub fn inverse_with(&mut self, data: &mut [Complex], executor: &Executor) {
        self.transform(data, true, executor);
    }

    fn transform(&mut self, data: &mut [Complex], inverse: bool, executor: &Executor) {
        let (nx, ny) = (self.nx, self.ny);
        assert_eq!(data.len(), nx * ny, "a {nx}x{ny} FFT was handed {} values", data.len());
        if data.is_empty() {
            return;
        }
        rows_in_place(&self.rows, data, inverse, executor);
        transpose(data, nx, &mut self.transposed, executor);
        rows_in_place(&self.columns, &mut self.transposed, inverse, executor);
        transpose(&self.transposed, ny, data, executor);
    }
}

/// Transform every `plan.len()`-long row of `data`, in bands across `executor`.
fn rows_in_place(plan: &Fft, data: &mut [Complex], inverse: bool, executor: &Executor) {
    let width = plan.len();
    executor.for_each_row_band_mut(data, width, FFT_GRAIN.per_row(width), |_, band| {
        with_scratch(plan.scratch_len(), |scratch| {
            for row in band.chunks_exact_mut(width) {
                if inverse {
                    plan.inverse(row, scratch);
                } else {
                    plan.forward(row, scratch);
                }
            }
        });
    });
}

/// `out = sourceᵀ`, where `source` is row-major with rows `width` long; `out`'s rows
/// are therefore `source.len() / width` long. Split by bands of `out`'s rows, each of
/// which reads a strip of `source`'s columns, and done in square tiles so that both
/// sides are touched a cache line at a time: measured at 512², 0.64 ms against 1.5 ms
/// for the untiled loop.
fn transpose(source: &[Complex], width: usize, out: &mut [Complex], executor: &Executor) {
    const BLOCK: usize = 16;
    let height = source.len() / width;
    executor.for_each_row_band_mut(out, height, FFT_GRAIN.per_row(height), |first, band| {
        let rows = band.len() / height;
        for tile_r in (0..rows).step_by(BLOCK) {
            for tile_j in (0..height).step_by(BLOCK) {
                let j_end = (tile_j + BLOCK).min(height);
                for r in tile_r..(tile_r + BLOCK).min(rows) {
                    let column = first + r;
                    let target = &mut band[r * height + tile_j..r * height + j_end];
                    for (j, slot) in (tile_j..j_end).zip(target) {
                        *slot = source[j * width + column];
                    }
                }
            }
        }
    });
}

/// The angular wavenumbers of an `n`-point transform over length `extent`, in FFT
/// order: `0, 1, …, ⌈n/2⌉−1, −⌊n/2⌋, …, −1` times `2π/extent`.
pub fn wavenumbers(n: usize, extent: f64) -> Vec<f64> {
    let base = core::f64::consts::TAU / extent;
    (0..n)
        .map(|k| {
            let signed = if k < n.div_ceil(2) { k as f64 } else { k as f64 - n as f64 };
            signed * base
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(data: &[Complex], sign: f64) -> Vec<Complex> {
        let n = data.len();
        (0..n)
            .map(|k| {
                data.iter().enumerate().fold(Complex::ZERO, |acc, (j, x)| {
                    let angle = sign * core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                    acc + *x * Complex::cis(angle)
                })
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<Complex> {
        (0..n).map(|j| Complex::new((j as f64 * 0.37).sin() + 0.1 * j as f64, (j as f64 * 1.3).cos())).collect()
    }

    fn worst(a: &[Complex], b: &[Complex]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn every_length_matches_the_naive_dft() {
        for n in [1, 2, 3, 4, 5, 7, 8, 12, 16, 31, 64, 96, 100, 128] {
            let x = signal(n);
            let plan = Fft::new(n);
            let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
            let mut y = x.clone();
            plan.forward(&mut y, &mut scratch);
            let expected = naive(&x, -1.0);
            let scale = expected.iter().map(|z| z.abs()).fold(1.0, f64::max);
            assert!(worst(&y, &expected) < 1e-12 * scale, "n = {n}: {:e}", worst(&y, &expected));

            plan.inverse(&mut y, &mut scratch);
            assert!(worst(&y, &x) < 1e-12, "n = {n} round trip: {:e}", worst(&y, &x));
        }
    }

    #[test]
    fn a_bluestein_transform_of_the_specs_width_is_accurate() {
        // 768 = 3·256, spec §25.2's grid width.
        let n = 768;
        let x = signal(n);
        let plan = Fft::new(n);
        assert_eq!(plan.scratch_len(), 2048);
        let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
        let mut y = x.clone();
        plan.forward(&mut y, &mut scratch);
        let expected = naive(&x, -1.0);
        let scale = expected.iter().map(|z| z.abs()).fold(1.0, f64::max);
        assert!(worst(&y, &expected) < 1e-11 * scale, "{:e}", worst(&y, &expected) / scale);
    }

    #[test]
    fn parseval_holds() {
        let n = 256;
        let x = signal(n);
        let mut y = x.clone();
        Fft::new(n).forward(&mut y, &mut []);
        let time: f64 = x.iter().map(|z| z.norm_sqr()).sum();
        let frequency: f64 = y.iter().map(|z| z.norm_sqr()).sum::<f64>() / n as f64;
        assert!((time - frequency).abs() < 1e-11 * time);
    }

    #[test]
    fn a_2d_transform_of_a_plane_wave_is_one_spike() {
        let (nx, ny) = (16, 12);
        let (kx, ky) = (3, 5);
        let mut data: Vec<Complex> = (0..nx * ny)
            .map(|index| {
                let (i, j) = (index % nx, index / nx);
                let phase = core::f64::consts::TAU * (kx * i) as f64 / nx as f64
                    + core::f64::consts::TAU * (ky * j) as f64 / ny as f64;
                Complex::cis(phase)
            })
            .collect();
        let original = data.clone();
        let mut plan = Fft2::new(nx, ny);
        plan.forward(&mut data);
        for (index, z) in data.iter().enumerate() {
            let expected = if index == ky * nx + kx { (nx * ny) as f64 } else { 0.0 };
            assert!((z.abs() - expected).abs() < 1e-10, "bin {index}: {z:?}");
        }
        plan.inverse(&mut data);
        assert!(worst(&data, &original) < 1e-13);
    }

    #[test]
    fn a_split_transform_has_the_sequential_transforms_bits() {
        let bits = |data: &[Complex]| data.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect::<Vec<_>>();
        let pool = Executor::with_threads(4);
        // Radix-2, and Bluestein both ways with a short final band.
        for (nx, ny) in [(128, 64), (96, 75)] {
            assert!(pool.partition(ny, FFT_GRAIN.per_row(nx)).count() > 1, "{nx}x{ny} rows are not split");
            assert!(pool.partition(nx, FFT_GRAIN.per_row(ny)).count() > 1, "{nx}x{ny} columns are not split");
            let mut sequential = signal(nx * ny);
            let mut split = sequential.clone();
            let (mut a, mut b) = (Fft2::new(nx, ny), Fft2::new(nx, ny));
            a.forward(&mut sequential);
            b.forward_with(&mut split, &pool);
            assert_eq!(bits(&sequential), bits(&split), "{nx}x{ny} forward");
            a.inverse(&mut sequential);
            b.inverse_with(&mut split, &pool);
            assert_eq!(bits(&sequential), bits(&split), "{nx}x{ny} inverse");
            let original = signal(nx * ny);
            let scale = original.iter().map(|z| z.abs()).fold(1.0, f64::max);
            assert!(worst(&sequential, &original) < 1e-12 * scale, "{nx}x{ny} round trip");
        }
    }

    #[test]
    fn wavenumbers_follow_fft_order() {
        let k = wavenumbers(4, core::f64::consts::TAU);
        assert_eq!(k, vec![0.0, 1.0, -2.0, -1.0]);
        let k = wavenumbers(5, core::f64::consts::TAU);
        assert_eq!(k, vec![0.0, 1.0, 2.0, -2.0, -1.0]);
    }
}
