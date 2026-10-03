//! A complex number, as small as a wavefunction needs.
//!
//! Spec §10.1 lists `complex32/64` as a numeric type *"for quantum wavefunctions and
//! frequency-domain modules"*. This is the 64-bit one. It is written out rather than
//! taken from a crate for the same reason the rest of the workspace avoids
//! dependencies (§24.1): it is a dozen operators, and its layout — two `f64`s, real
//! first — is part of what a GPU backend would one day have to match.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// `re + i·im`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Zero.
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    /// One.
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    /// The imaginary unit.
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    /// `re + i·im`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// A real number.
    #[inline]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// `e^{iθ}`, the unit phasor at angle `theta`.
    #[inline]
    pub fn cis(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self { re: c, im: s }
    }

    /// `r·e^{iθ}`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::cis(theta).scale(r)
    }

    /// The complex conjugate.
    #[inline]
    pub const fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    /// `|z|²`, which for a wavefunction is a probability density.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// `|z|`.
    #[inline]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The argument, in `(−π, π]`.
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Multiply by a real number.
    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self { re: self.re * factor, im: self.im * factor }
    }

    /// Multiply by `i`.
    #[inline]
    pub fn mul_i(self) -> Self {
        Self { re: -self.im, im: self.re }
    }

    /// Whether both parts are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, rhs: Complex) -> Complex {
        Complex { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Complex) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, rhs: Complex) -> Complex {
        Complex { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: Complex) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, rhs: Complex) {
        *self = *self * rhs;
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: f64) -> Complex {
        self.scale(rhs)
    }
}

impl Div for Complex {
    type Output = Complex;
    #[inline]
    fn div(self, rhs: Complex) -> Complex {
        let d = rhs.norm_sqr();
        Complex {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl Neg for Complex {
    type Output = Complex;
    #[inline]
    fn neg(self) -> Complex {
        Complex { re: -self.re, im: -self.im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_the_textbook() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        assert_eq!(a * b, Complex::new(5.0, 5.0));
        assert_eq!(a + b, Complex::new(4.0, 1.0));
        assert_eq!(a - b, Complex::new(-2.0, 3.0));
        let q = (a * b) / b;
        assert!((q - a).abs() < 1e-15);
        assert_eq!(a.conj(), Complex::new(1.0, -2.0));
        assert_eq!(a.mul_i(), a * Complex::I);
        assert_eq!(a.norm_sqr(), 5.0);
    }

    #[test]
    fn cis_is_a_unit_phasor() {
        for k in 0..16 {
            let theta = f64::from(k) * 0.4 - 3.0;
            let z = Complex::cis(theta);
            assert!((z.abs() - 1.0).abs() < 1e-15);
            assert!((z.arg() - theta).abs() < 1e-12);
        }
    }
}
