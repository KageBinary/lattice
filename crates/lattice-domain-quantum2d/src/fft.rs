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
        /// `e^{−2πik/n}` for `k < n/2`.
        twiddles: Vec<Complex>,
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
            let twiddles = (0..n / 2)
                .map(|k| Complex::cis(-core::f64::consts::TAU * k as f64 / n as f64))
                .collect();
            let bits = n.trailing_zeros();
            let reversed = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
            Kind::Radix2 { twiddles, reversed }
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
            Kind::Radix2 { twiddles, reversed } => radix2(data, twiddles, reversed, false),
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
            Kind::Radix2 { twiddles, reversed } => radix2(data, twiddles, reversed, true),
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

/// The iterative Cooley–Tukey butterfly network. `inverse` conjugates the twiddles and
/// leaves the scaling to the caller.
fn radix2(data: &mut [Complex], twiddles: &[Complex], reversed: &[u32], inverse: bool) {
    let n = data.len();
    for (i, &r) in reversed.iter().enumerate() {
        let r = r as usize;
        if i < r {
            data.swap(i, r);
        }
    }
    let mut half = 1;
    while half < n {
        let stride = n / (2 * half);
        for start in (0..n).step_by(2 * half) {
            for k in 0..half {
                let w = twiddles[k * stride];
                let w = if inverse { w.conj() } else { w };
                let a = data[start + k];
                let b = data[start + k + half] * w;
                data[start + k] = a + b;
                data[start + k + half] = a - b;
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

/// A two-dimensional transform over a row-major `nx × ny` array.
///
/// Rows are transformed in place; columns are gathered into a contiguous buffer,
/// transformed and scattered back, which keeps the inner transform cache-friendly at
/// the cost of one copy per column.
#[derive(Clone, Debug)]
pub struct Fft2 {
    nx: usize,
    ny: usize,
    rows: Fft,
    columns: Fft,
    column: Vec<Complex>,
    scratch: Vec<Complex>,
}

impl Fft2 {
    /// A plan for `nx × ny` arrays.
    pub fn new(nx: usize, ny: usize) -> Self {
        let rows = Fft::new(nx);
        let columns = Fft::new(ny);
        let scratch = vec![Complex::ZERO; rows.scratch_len().max(columns.scratch_len())];
        Self { nx, ny, rows, columns, column: vec![Complex::ZERO; ny], scratch }
    }

    /// The array shape, `[nx, ny]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.nx, self.ny]
    }

    /// Forward transform in place.
    ///
    /// # Panics
    ///
    /// If `data` is not `nx × ny` long.
    pub fn forward(&mut self, data: &mut [Complex]) {
        self.transform(data, false);
    }

    /// Inverse transform in place, including the `1/(nx·ny)`.
    ///
    /// # Panics
    ///
    /// If `data` is not `nx × ny` long.
    pub fn inverse(&mut self, data: &mut [Complex]) {
        self.transform(data, true);
    }

    fn transform(&mut self, data: &mut [Complex], inverse: bool) {
        let (nx, ny) = (self.nx, self.ny);
        assert_eq!(data.len(), nx * ny, "a {nx}x{ny} FFT was handed {} values", data.len());
        for row in data.chunks_exact_mut(nx) {
            if inverse {
                self.rows.inverse(row, &mut self.scratch);
            } else {
                self.rows.forward(row, &mut self.scratch);
            }
        }
        for i in 0..nx {
            for (j, slot) in self.column.iter_mut().enumerate() {
                *slot = data[j * nx + i];
            }
            if inverse {
                self.columns.inverse(&mut self.column, &mut self.scratch);
            } else {
                self.columns.forward(&mut self.column, &mut self.scratch);
            }
            for (j, value) in self.column.iter().enumerate() {
                data[j * nx + i] = *value;
            }
        }
    }
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
    fn wavenumbers_follow_fft_order() {
        let k = wavenumbers(4, core::f64::consts::TAU);
        assert_eq!(k, vec![0.0, 1.0, -2.0, -1.0]);
        let k = wavenumbers(5, core::f64::consts::TAU);
        assert_eq!(k, vec![0.0, 1.0, 2.0, -2.0, -1.0]);
    }
}
