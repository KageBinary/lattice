//! Coulomb interactions with a declared cutoff: the damped shifted force sum.
//!
//! Spec §12.4 asks for *"Coulomb with declared cutoff/approximation"*. Coulomb's `1/r`
//! is the one pair law in the table that a plain cutoff ruins: it decays too slowly
//! for the neglected tail to be small, and a truncated sum over a neutral crystal does
//! not even converge to the right number as the cutoff grows — it oscillates with
//! whichever shell of charges the cutoff happens to cut through.
//!
//! The approximation here is Fennell and Gezelter's *damped shifted force* (DSF),
//!
//! ```text
//!   U(r) = k qᵢqⱼ [ erfc(αr)/r − erfc(αR)/R + (erfc(αR)/R² + (2α/√π) e^{−α²R²}/R)(r − R) ]   r < R
//!   U_self = −k (erfc(αR)/2R + α/√π) Σ qᵢ²
//! ```
//!
//! It is the real-space half of an Ewald sum, `erfc(αr)/r`, made to vanish with its
//! derivative at the cutoff `R`, plus the Ewald self term. Two things make it work:
//!
//! - the damping `α` hands the long-range part of each interaction to a smooth
//!   neutralizing background, so what is left decays like `erfc(αr)` and a cutoff a few
//!   `1/α` out loses almost nothing;
//! - shifting both the energy and the force to zero at `R` means a pair crossing the
//!   cutoff feels no impulse, so the dynamics conserve this potential's energy to the
//!   integrator's error alone, as with the force-shifted Lennard-Jones.
//!
//! What it leaves out is the reciprocal-space half of Ewald. For a crystal that is a
//! sum of terms like `exp(−k²/4α²)` over its reciprocal lattice, which a modest `α`
//! makes negligible; in a disordered system it is the long-wavelength part of the
//! electrostatics, and the approximation assumes the system is locally neutral at the
//! scale `R` so that part is small. The validation suite measures both halves against
//! a 2D rock-salt crystal's Madelung energy.
//!
//! # Which Coulomb law
//!
//! Point charges confined to the plane, interacting by the three-dimensional law
//! `k qᵢqⱼ / r` with `k = 1/4πε₀` — ions adsorbed on a surface, a Wigner crystal, a dusty
//! plasma layer. Not two-dimensional electrostatics, where line charges interact by
//! `−ln r`; that is a different physical system and is not built.

use lattice_ir::{ForceAccumulation, ParticleStore};

use crate::forces::{ForceContext, ForceLaw};

/// `2/√π`.
const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;

/// The complementary error function, shared with the expression compiler so a law's
/// `erfc` and the built-in Coulomb sum compute the same bits.
pub use lattice_expr::erfc;

/// `erfcx(x) = e^{x²} erfc(x)` on `[0, top]` as a Chebyshev series, fitted once.
///
/// [`erfc`] is exact enough but slow — a continued fraction sixty levels deep — and the
/// pair loop needs it for every pair inside the cutoff on every step. `erfcx` is smooth
/// and slowly varying (from 1 down to about `1/(x√π)`), so a Chebyshev series converges
/// fast: measured against Python's `math.erfc`, 32 terms hold it to 2e-14 relative out
/// to `x = 5`, 48 to 1e-13 out to 10, and 64 to 2e-13 out to 20. Past `top` the exact
/// function is used, which only happens for a pair a few `1/α` inside a very wide cutoff.
#[derive(Clone, Debug, PartialEq)]
struct Erfcx {
    top: f64,
    coefficients: Vec<f64>,
}

impl Erfcx {
    fn fit(top: f64) -> Self {
        let top = top.clamp(1.0, 20.0);
        let n: usize = if top <= 5.0 {
            32
        } else if top <= 10.0 {
            48
        } else {
            64
        };
        let nodes: Vec<f64> = (0..n)
            .map(|k| {
                let y = (core::f64::consts::PI * (k as f64 + 0.5) / n as f64).cos();
                let x = 0.5 * top * (y + 1.0);
                erfc(x) * (x * x).exp()
            })
            .collect();
        let coefficients = (0..n)
            .map(|j| {
                let sum: f64 = nodes
                    .iter()
                    .enumerate()
                    .map(|(k, f)| f * (core::f64::consts::PI * j as f64 * (k as f64 + 0.5) / n as f64).cos())
                    .sum();
                2.0 * sum / n as f64
            })
            .collect();
        Self { top, coefficients }
    }

    /// `(erfc(x), e^{−x²})` for `x ≥ 0`.
    #[inline]
    fn eval(&self, x: f64) -> (f64, f64) {
        let gaussian = (-x * x).exp();
        if x > self.top {
            return (erfc(x), gaussian);
        }
        // Clenshaw's recurrence on y ∈ [−1, 1].
        let y = 2.0 * x / self.top - 1.0;
        let (mut b1, mut b2) = (0.0, 0.0);
        for &c in self.coefficients[1..].iter().rev() {
            (b1, b2) = (2.0 * y * b1 - b2 + c, b1);
        }
        let erfcx = y * b1 - b2 + 0.5 * self.coefficients[0];
        (erfcx * gaussian, gaussian)
    }
}

/// Coulomb's law between charged particles by the damped shifted force sum.
///
/// See the [module documentation](self) for the potential, what it approximates, and
/// which Coulomb law it is.
#[derive(Clone, Debug, PartialEq)]
pub struct Coulomb {
    /// The cutoff `R`, m.
    pub cutoff: f64,
    /// The damping `α`, 1/m. Zero is the undamped shifted-force sum.
    pub damping: f64,
    /// The Coulomb constant `k`, N·m²/C²: `1/4πε₀` in SI, or 1 in reduced units.
    pub constant: f64,
    /// `erfc(αR)/R`.
    energy_shift: f64,
    /// `erfc(αR)/R² + (2α/√π) e^{−α²R²}/R`, the radial force per `k qᵢqⱼ` at `R`.
    force_shift: f64,
    /// `erfc` over the range of `αr` the cutoff allows.
    table: Erfcx,
}

impl Coulomb {
    /// The damped shifted force sum with cutoff `R` and damping `α`, in SI units.
    ///
    /// # Panics
    ///
    /// On a non-positive cutoff or a negative damping.
    pub fn damped_shifted_force(cutoff: f64, damping: f64) -> Self {
        Self::with_constant(cutoff, damping, lattice_units::constants::value::COULOMB)
    }

    /// The same with an explicit Coulomb constant — 1 for reduced units.
    ///
    /// # Panics
    ///
    /// As [`Coulomb::damped_shifted_force`], or on a non-positive constant.
    pub fn with_constant(cutoff: f64, damping: f64, constant: f64) -> Self {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "a Coulomb cutoff must be positive, got {cutoff}");
        assert!(damping >= 0.0 && damping.is_finite(), "a Coulomb damping cannot be negative, got {damping}");
        assert!(constant > 0.0 && constant.is_finite(), "the Coulomb constant must be positive, got {constant}");
        let erfc_r = erfc(damping * cutoff);
        let gaussian = TWO_OVER_SQRT_PI * damping * (-(damping * cutoff).powi(2)).exp();
        Self {
            cutoff,
            damping,
            constant,
            energy_shift: erfc_r / cutoff,
            force_shift: erfc_r / (cutoff * cutoff) + gaussian / cutoff,
            table: Erfcx::fit(damping * cutoff),
        }
    }

    /// The pair energy at separation `r` for charges whose product is `qq`, J.
    pub fn pair_energy(&self, qq: f64, r: f64) -> f64 {
        if r >= self.cutoff {
            return 0.0;
        }
        let (erfc_r, _) = self.table.eval(self.damping * r);
        self.constant * qq * (erfc_r / r - self.energy_shift + self.force_shift * (r - self.cutoff))
    }

    /// The radial pair force at separation `r`, N — positive for repulsion.
    pub fn pair_force(&self, qq: f64, r: f64) -> f64 {
        if r >= self.cutoff {
            return 0.0;
        }
        self.constant * qq * self.radial(r)
    }

    /// The radial force per `k qᵢqⱼ`, 1/m².
    fn radial(&self, r: f64) -> f64 {
        let a = self.damping;
        let (erfc_r, gaussian) = self.table.eval(a * r);
        erfc_r / (r * r) + TWO_OVER_SQRT_PI * a * gaussian / r - self.force_shift
    }

    /// The self term `−k (erfc(αR)/2R + α/√π) Σ qᵢ²`, J — a constant for a fixed set of
    /// charges, without which a crystal's energy does not converge to its Madelung
    /// value.
    pub fn self_energy(&self, charges: &[f64]) -> f64 {
        let sum: f64 = charges.iter().map(|q| q * q).sum();
        -self.constant * (0.5 * self.energy_shift + self.damping / core::f64::consts::PI.sqrt()) * sum
    }
}

impl ForceLaw for Coulomb {
    fn name(&self) -> &'static str {
        "coulomb"
    }

    fn is_conservative(&self) -> bool {
        true
    }

    fn cutoff(&self) -> Option<f64> {
        Some(self.cutoff)
    }

    fn accumulate(&self, view: &mut ForceAccumulation<'_>, ctx: &ForceContext<'_>) {
        let Some(list) = ctx.neighbors else {
            debug_assert!(false, "Coulomb requires a neighbour list");
            return;
        };
        let (pos_x, pos_y, charge) = (view.pos_x, view.pos_y, view.charge);
        let (force_x, force_y) = (&mut *view.force_x, &mut *view.force_y);
        let cutoff2 = self.cutoff * self.cutoff;
        list.for_each_pair(pos_x, pos_y, |i, j, dx, dy, r2| {
            let qq = charge[i] * charge[j];
            if qq == 0.0 || r2 >= cutoff2 {
                return;
            }
            let r = r2.sqrt();
            // Fᵢ = F(r) r̂ with r̂ from j to i, and (dx, dy) = xⱼ − xᵢ.
            let coefficient = self.constant * qq * self.radial(r) / r;
            let (fx, fy) = (coefficient * dx, coefficient * dy);
            force_x[i] -= fx;
            force_y[i] -= fy;
            force_x[j] += fx;
            force_y[j] += fy;
        });
    }

    fn potential_energy(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let Some(list) = ctx.neighbors else { return 0.0 };
        let charge = store.charge();
        let mut total = 0.0;
        list.for_each_pair(store.pos_x(), store.pos_y(), |i, j, _, _, r2| {
            let qq = charge[i] * charge[j];
            if qq != 0.0 {
                total += self.pair_energy(qq, r2.sqrt());
            }
        });
        total + self.self_energy(charge)
    }

    fn virial(&self, store: &ParticleStore, ctx: &ForceContext<'_>) -> f64 {
        let Some(list) = ctx.neighbors else { return 0.0 };
        let charge = store.charge();
        let mut total = 0.0;
        // (xᵢ − xⱼ)·Fᵢⱼ = F(r)·r.
        list.for_each_pair(store.pos_x(), store.pos_y(), |i, j, _, _, r2| {
            let qq = charge[i] * charge[j];
            if qq != 0.0 {
                let r = r2.sqrt();
                total += self.pair_force(qq, r) * r;
            }
        });
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from Python's `math.erfc`, which is correctly rounded to within
    /// an ulp or two.
    #[test]
    fn erfc_matches_reference_values() {
        let reference = [
            (0.0, 1.0),
            (1e-8, 0.999_999_988_716_208_3),
            (0.3, 0.671_373_240_540_872_6),
            (1.0, 0.157_299_207_050_285_1),
            (1.999, 4.698_443_348_629_487e-3),
            (2.0, 4.677_734_981_047_265e-3),
            (2.6, 2.360_344_165_293_490_8e-4),
            (3.4, 1.521_993_362_862_286e-6),
            (5.0, 1.537_459_794_428_035_1e-12),
            (9.0, 4.137_031_746_513_81e-37),
            (-1.0, 1.842_700_792_949_715),
        ];
        for (x, expected) in reference {
            let got = erfc(x);
            assert!(((got - expected) / expected).abs() < 5e-13, "erfc({x}) = {got:e}, want {expected:e}");
        }
        assert_eq!(erfc(30.0), 0.0);
        assert!(erfc(f64::NAN).is_nan());
    }

    #[test]
    fn the_fitted_erfc_matches_the_exact_one_across_its_range() {
        for top in [0.5, 2.4, 5.0, 9.0, 18.0, 40.0] {
            let table = Erfcx::fit(top);
            for k in 0..=500 {
                let x = top * f64::from(k) / 500.0;
                let (fitted, gaussian) = table.eval(x);
                let exact = erfc(x);
                if exact == 0.0 {
                    // Past x = 27 both underflow.
                    assert_eq!(fitted, 0.0);
                    continue;
                }
                assert!(((fitted - exact) / exact).abs() < 5e-13, "top {top}, x {x}: {fitted:e} vs {exact:e}");
                assert_eq!(gaussian, (-x * x).exp());
            }
        }
    }

    #[test]
    fn the_force_is_minus_the_derivative_of_the_energy_and_both_vanish_at_the_cutoff() {
        let law = Coulomb::with_constant(6.0, 0.4, 1.0);
        for r in [0.3, 0.9, 1.7, 3.0, 5.2, 5.99] {
            let h = 1e-5 * r;
            let numeric = -(law.pair_energy(1.0, r + h) - law.pair_energy(1.0, r - h)) / (2.0 * h);
            let analytic = law.pair_force(1.0, r);
            assert!((numeric - analytic).abs() < 1e-7 * analytic.abs().max(1e-3), "r = {r}: {numeric} vs {analytic}");
        }
        // Both continuous at the cutoff, which is what lets a crossing pair conserve energy.
        let below = 6.0 * (1.0 - 1e-9);
        assert!(law.pair_energy(1.0, below).abs() < 1e-9 && law.pair_force(1.0, below).abs() < 1e-9);
        assert_eq!(law.pair_energy(1.0, 6.0), 0.0);
    }

    #[test]
    fn undamped_with_a_far_cutoff_it_is_coulombs_law_near_the_origin() {
        let law = Coulomb::with_constant(1e6, 0.0, 1.0);
        // U = 1/r − 1/R + (r − R)/R², which near r = 1 is 1/r to a part in 10⁶.
        assert!((law.pair_energy(1.0, 1.0) - 1.0).abs() < 3e-6);
        assert!((law.pair_force(-2.0, 0.5) + 8.0).abs() < 1e-5, "attraction between opposite charges");
    }
}
