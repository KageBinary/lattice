//! Trajectory analysis: temperature, pressure, the radial distribution function, and
//! mean squared displacement.
//!
//! Spec §12.4 lists *"energy, RDF, MSD, temperature, pressure estimate"* as the basic
//! scientific observability the molecular module owes. Energy lives on the domain;
//! the rest is here, as plain functions over the store so a validation case can call
//! them on any configuration without a domain around it.

use lattice_ir::ParticleStore;

use crate::image::MinimumImage;
use crate::neighbors::CellList;
use crate::thermostat::BOLTZMANN;

/// The thermal state of a particle set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thermal {
    /// Kinetic energy of motion relative to the centre of mass, joules.
    pub kinetic: f64,
    /// Degrees of freedom the temperature is defined over.
    pub degrees_of_freedom: usize,
    /// Temperature, kelvin, or `None` when there are no degrees of freedom to define
    /// one — a single free particle has a speed, not a temperature.
    pub temperature: Option<f64>,
    /// Centre-of-mass velocity of the mobile particles, m/s.
    pub com_velocity: [f64; 2],
}

/// Compute the thermal state.
///
/// The centre-of-mass velocity is removed and two degrees of freedom with it, because
/// pair forces conserve momentum and a drifting box is not hotter than a stationary
/// one. Pinned (infinite-mass) particles are excluded from both sums. When any
/// particle is pinned the mobile ones do *not* conserve momentum — the pinned ones
/// absorb it — so no degrees of freedom are subtracted and no centre-of-mass motion
/// is removed.
pub fn thermal(store: &ParticleStore) -> Thermal {
    let (vx, vy, masses) = (store.vel_x(), store.vel_y(), store.mass());
    let mut total_mass = 0.0;
    let mut px = 0.0;
    let mut py = 0.0;
    let mut mobile = 0usize;
    let mut pinned = false;
    for i in 0..vx.len() {
        if masses[i].is_finite() {
            total_mass += masses[i];
            px += masses[i] * vx[i];
            py += masses[i] * vy[i];
            mobile += 1;
        } else {
            pinned = true;
        }
    }
    let com_velocity = if pinned || total_mass <= 0.0 { [0.0, 0.0] } else { [px / total_mass, py / total_mass] };
    let degrees_of_freedom = if pinned { 2 * mobile } else { (2 * mobile).saturating_sub(2) };

    let mut kinetic = 0.0;
    for i in 0..vx.len() {
        if masses[i].is_finite() {
            let (ux, uy) = (vx[i] - com_velocity[0], vy[i] - com_velocity[1]);
            kinetic += 0.5 * masses[i] * (ux * ux + uy * uy);
        }
    }
    let temperature = if degrees_of_freedom == 0 {
        None
    } else {
        Some(2.0 * kinetic / (degrees_of_freedom as f64 * BOLTZMANN))
    };
    Thermal { kinetic, degrees_of_freedom, temperature, com_velocity }
}

/// The temperature alone, kelvin.
pub fn temperature(store: &ParticleStore) -> Option<f64> {
    thermal(store).temperature
}

/// Draw velocities for a target temperature.
///
/// Each component is normal with variance `k_B T / m`; the centre-of-mass velocity is
/// then removed and the result rescaled so the instantaneous temperature is `T`
/// *exactly*, not merely in expectation. That makes "started at 300 K" a statement
/// about the run rather than about the generator. Pinned particles keep zero velocity.
///
/// Returns false, leaving velocities untouched, if the set has no degrees of freedom
/// to carry a temperature.
pub fn thermalize(store: &mut ParticleStore, temperature: f64, rng: &mut lattice_ir::Pcg32) -> bool {
    assert!(temperature >= 0.0 && temperature.is_finite(), "temperature must be non-negative, got {temperature}");
    if thermal(store).degrees_of_freedom == 0 {
        return false;
    }
    let n = store.len();
    {
        let d = store.dynamics();
        for i in 0..n {
            let inv_mass = d.inv_mass[i];
            if inv_mass > 0.0 {
                let sigma = (BOLTZMANN * temperature * inv_mass).sqrt();
                d.vel_x[i] = sigma * rng.normal();
                d.vel_y[i] = sigma * rng.normal();
            } else {
                d.vel_x[i] = 0.0;
                d.vel_y[i] = 0.0;
            }
        }
    }
    let state = thermal(store);
    let Some(current) = state.temperature else { return false };
    let [ux, uy] = state.com_velocity;
    let scale = if current > 0.0 { (temperature / current).sqrt() } else { 0.0 };
    let d = store.dynamics();
    for i in 0..n {
        if d.inv_mass[i] > 0.0 {
            d.vel_x[i] = scale * (d.vel_x[i] - ux);
            d.vel_y[i] = scale * (d.vel_y[i] - uy);
        }
    }
    true
}

/// The pressure of a two-dimensional system from the virial theorem.
///
/// ```text
///   P·A = K + ½·W        W = Σ_pairs (xᵢ − xⱼ)·Fᵢⱼ
/// ```
///
/// `K` is the full kinetic energy — the mechanical theorem does not know about
/// centre-of-mass motion — and `W` the pair virial summed over every interaction.
/// In 2D a pressure is a force per unit length, N/m.
pub fn pressure(kinetic_energy: f64, virial: f64, area: f64) -> f64 {
    (kinetic_energy + 0.5 * virial) / area
}

/// A radial distribution function `g(r)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialDistribution {
    /// Bin width, metres.
    pub bin_width: f64,
    /// Bin centres, metres.
    pub r: Vec<f64>,
    /// `g(r)` at each centre. Unity for an ideal gas.
    pub g: Vec<f64>,
    /// Raw pair counts per bin, summed over frames.
    pub counts: Vec<u64>,
    /// How many configurations were accumulated.
    pub frames: u64,
    /// Mean number density over the frames, per square metre.
    pub density: f64,
}

impl RadialDistribution {
    /// `g(r)` linearly interpolated at `r`, or `None` outside the histogram.
    pub fn at(&self, r: f64) -> Option<f64> {
        if self.r.is_empty() || r < self.r[0] || r > *self.r.last()? {
            return None;
        }
        let f = (r - self.r[0]) / self.bin_width;
        let k = (f.floor() as usize).min(self.r.len() - 2);
        let t = (f - k as f64).clamp(0.0, 1.0);
        Some(self.g[k] + t * (self.g[k + 1] - self.g[k]))
    }

    /// The position and height of the maximum of `g`.
    ///
    /// For a simple liquid this is the first-neighbour shell. Whether it is
    /// *structure* is the caller's judgement against the counting statistics: an
    /// ideal gas has a maximum too, a little above 1 by chance.
    pub fn peak(&self) -> Option<(f64, f64)> {
        let (k, &g) = self
            .g
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))?;
        Some((self.r[k], g))
    }
}

/// Accumulates pair separations into a histogram over many configurations.
///
/// Normalized so that an ideal gas gives `g(r) = 1` at every `r`: the expected number
/// of unordered pairs in a shell of area `S(r)` is `N(N−1)/2 · S(r)/A`, and `g` is the
/// observed count divided by that expectation. The normalization assumes a
/// homogeneous periodic box; in a box with walls the shells near an edge are cut off
/// and `g` falls below 1 there for geometric rather than physical reasons.
#[derive(Clone, Debug)]
pub struct RdfAccumulator {
    range: f64,
    bin_width: f64,
    counts: Vec<u64>,
    frames: u64,
    /// Σ over frames of N(N−1)/(2A), so the normalization follows the population.
    pair_density_sum: f64,
    number_density_sum: f64,
    area: f64,
    image: MinimumImage,
    cells: CellList,
}

impl RdfAccumulator {
    /// A histogram of `bins` bins out to `range`, over a box of `extent` from
    /// `origin` wrapping on the axes marked periodic.
    ///
    /// # Panics
    ///
    /// If `range` exceeds half the box on a periodic axis, where the minimum image is
    /// ambiguous and a histogram would count some pairs twice; or on a non-positive
    /// range or zero bins.
    pub fn new(bins: usize, range: f64, origin: [f64; 2], extent: [f64; 2], periodic: [bool; 2], capacity: usize) -> Self {
        assert!(bins > 0, "an RDF needs at least one bin");
        assert!(range > 0.0 && range.is_finite(), "RDF range must be positive, got {range}");
        let image = MinimumImage::new(extent, periodic);
        assert!(
            range <= image.max_representable(),
            "RDF range {range} exceeds half the periodic box ({})",
            image.max_representable()
        );
        Self {
            range,
            bin_width: range / bins as f64,
            counts: vec![0; bins],
            frames: 0,
            pair_density_sum: 0.0,
            number_density_sum: 0.0,
            area: extent[0] * extent[1],
            image,
            cells: CellList::new(origin, extent, periodic, range, capacity),
        }
    }

    /// The outer radius, metres.
    pub fn range(&self) -> f64 {
        self.range
    }

    /// Frames accumulated so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Add one configuration.
    pub fn sample(&mut self, store: &ParticleStore) {
        self.sample_positions(store.pos_x(), store.pos_y());
    }

    /// Add one configuration from raw position slices.
    pub fn sample_positions(&mut self, pos_x: &[f64], pos_y: &[f64]) {
        let n = pos_x.len();
        self.cells.rebuild_from(pos_x, pos_y);
        let (counts, width) = (&mut self.counts, self.bin_width);
        let bins = counts.len();
        self.cells.for_each_pair(pos_x, pos_y, |_, _, _, _, r2| {
            let k = (r2.sqrt() / width) as usize;
            if k < bins {
                counts[k] += 1;
            }
        });
        self.frames += 1;
        self.pair_density_sum += (n as f64) * (n as f64 - 1.0) / (2.0 * self.area);
        self.number_density_sum += n as f64 / self.area;
    }

    /// The normalized distribution.
    pub fn result(&self) -> RadialDistribution {
        let bins = self.counts.len();
        let mut r = Vec::with_capacity(bins);
        let mut g = Vec::with_capacity(bins);
        for k in 0..bins {
            let inner = k as f64 * self.bin_width;
            let outer = inner + self.bin_width;
            let shell = core::f64::consts::PI * (outer * outer - inner * inner);
            let expected = self.pair_density_sum * shell;
            r.push(0.5 * (inner + outer));
            g.push(if expected > 0.0 { self.counts[k] as f64 / expected } else { 0.0 });
        }
        RadialDistribution {
            bin_width: self.bin_width,
            r,
            g,
            counts: self.counts.clone(),
            frames: self.frames,
            density: if self.frames > 0 { self.number_density_sum / self.frames as f64 } else { 0.0 },
        }
    }

    /// The minimum-image rule in use.
    pub fn image(&self) -> MinimumImage {
        self.image
    }

    /// Bytes held.
    pub fn memory_bytes(&self) -> usize {
        self.counts.len() * size_of::<u64>() + self.cells.memory_bytes()
    }
}

/// Mean squared displacement of a set of accumulated displacements, m².
pub fn mean_squared_displacement(dx: &[f64], dy: &[f64]) -> f64 {
    if dx.is_empty() {
        return 0.0;
    }
    let sum: f64 = dx.iter().zip(dy).map(|(x, y)| x * x + y * y).sum();
    sum / dx.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::{Pcg32, ParticleSpec};

    #[test]
    fn temperature_uses_thermal_velocities_over_2n_minus_2_degrees_of_freedom() {
        let mut store = ParticleStore::with_capacity(2);
        // Two unit masses moving oppositely at 1 m/s: K = 1 J, N_df = 2, T = 1/k_B.
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_velocity([1.0, 0.0]).with_mass(1.0)).unwrap();
        store.spawn(ParticleSpec::at([1.0, 0.0]).with_velocity([-1.0, 0.0]).with_mass(1.0)).unwrap();
        let state = thermal(&store);
        assert_eq!(state.degrees_of_freedom, 2);
        assert!((state.kinetic - 1.0).abs() < 1e-15);
        assert!((state.temperature.unwrap() - 1.0 / BOLTZMANN).abs() < 1e-3 / BOLTZMANN);

        // Adding a common drift changes nothing thermal.
        {
            let d = store.dynamics();
            d.vel_x[0] += 7.0;
            d.vel_x[1] += 7.0;
        }
        let drifting = thermal(&store);
        assert!((drifting.kinetic - 1.0).abs() < 1e-12, "{}", drifting.kinetic);
        assert!((drifting.com_velocity[0] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn a_single_particle_has_no_temperature() {
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_velocity([3.0, 4.0])).unwrap();
        assert_eq!(temperature(&store), None);
        assert_eq!(thermal(&store).degrees_of_freedom, 0);
        assert_eq!(temperature(&ParticleStore::with_capacity(0)), None);
    }

    #[test]
    fn pinned_particles_keep_every_mobile_degree_of_freedom() {
        let mut store = ParticleStore::with_capacity(3);
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_mass(f64::INFINITY)).unwrap();
        store.spawn(ParticleSpec::at([1.0, 0.0]).with_velocity([2.0, 0.0]).with_mass(1.0)).unwrap();
        store.spawn(ParticleSpec::at([2.0, 0.0]).with_velocity([2.0, 0.0]).with_mass(1.0)).unwrap();
        let state = thermal(&store);
        assert_eq!(state.degrees_of_freedom, 4, "a wall absorbs momentum, so nothing is subtracted");
        assert_eq!(state.com_velocity, [0.0, 0.0]);
        assert!((state.kinetic - 4.0).abs() < 1e-12);
    }

    #[test]
    fn thermalize_hits_the_target_exactly_with_zero_momentum() {
        let mut store = ParticleStore::with_capacity(500);
        for i in 0..500 {
            store.spawn(ParticleSpec::at([i as f64, 0.0]).with_mass(1.0 + (i % 4) as f64)).unwrap();
        }
        let mut rng = Pcg32::seed_from_u64(9);
        assert!(thermalize(&mut store, 300.0, &mut rng));
        let state = thermal(&store);
        assert!((state.temperature.unwrap() - 300.0).abs() < 1e-9, "{:?}", state.temperature);
        assert!(state.com_velocity[0].abs() < 1e-12 && state.com_velocity[1].abs() < 1e-12);

        // The x and y components are drawn independently, so the equipartition
        // between them holds to the usual sqrt(2/N) statistical scatter.
        let kx: f64 = store.vel_x().iter().zip(store.mass()).map(|(v, m)| 0.5 * m * v * v).sum();
        let ky: f64 = store.vel_y().iter().zip(store.mass()).map(|(v, m)| 0.5 * m * v * v).sum();
        assert!((kx / ky - 1.0).abs() < 5.0 * (2.0f64 / 500.0).sqrt(), "kx {kx} ky {ky}");
    }

    #[test]
    fn thermalize_refuses_a_single_particle() {
        let mut store = ParticleStore::with_capacity(1);
        store.spawn(ParticleSpec::at([0.0, 0.0]).with_velocity([1.0, 1.0])).unwrap();
        let mut rng = Pcg32::seed_from_u64(1);
        assert!(!thermalize(&mut store, 10.0, &mut rng));
        assert_eq!(store.vel_x()[0], 1.0, "velocities are left untouched");
    }

    #[test]
    fn ideal_gas_pressure_is_kinetic_only() {
        // P·A = K with no virial; area 4 and K = 8 gives P = 2 N/m.
        assert_eq!(pressure(8.0, 0.0, 4.0), 2.0);
        assert_eq!(pressure(8.0, -4.0, 4.0), 1.5, "an attractive virial lowers it");
    }

    /// An ideal gas — uniformly random positions — must give g(r) = 1 everywhere, to
    /// within counting statistics. This is the normalization check.
    #[test]
    fn an_ideal_gas_has_a_flat_distribution() {
        let n = 2000;
        let extent = 40.0;
        let mut rng = Pcg32::seed_from_u64(31);
        let mut rdf = RdfAccumulator::new(20, 5.0, [0.0; 2], [extent; 2], [true; 2], n);
        let mut xs = vec![0.0; n];
        let mut ys = vec![0.0; n];
        for _ in 0..30 {
            for i in 0..n {
                xs[i] = rng.range(0.0, extent);
                ys[i] = rng.range(0.0, extent);
            }
            rdf.sample_positions(&xs, &ys);
        }
        let result = rdf.result();
        assert_eq!(result.frames, 30);
        assert!((result.density - n as f64 / (extent * extent)).abs() < 1e-12);
        for (k, (&g, &count)) in result.g.iter().zip(&result.counts).enumerate() {
            // Poisson counting: relative scatter 1/sqrt(count); allow five of them.
            let scatter = 5.0 / (count as f64).sqrt();
            assert!((g - 1.0).abs() < scatter, "bin {k}: g = {g} from {count} pairs");
        }
        let (_, peak) = result.peak().unwrap();
        let sparsest = result.counts.iter().copied().min().unwrap() as f64;
        assert!(peak < 1.0 + 5.0 / sparsest.sqrt(), "no structure in an ideal gas, got a peak of {peak}");
        assert!(result.at(2.5).is_some() && result.at(9.0).is_none());
    }

    #[test]
    fn a_lattice_puts_all_its_pairs_at_the_lattice_spacing() {
        let side = 10;
        let spacing = 1.0;
        let extent = side as f64 * spacing;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for j in 0..side {
            for i in 0..side {
                xs.push((i as f64 + 0.5) * spacing);
                ys.push((j as f64 + 0.5) * spacing);
            }
        }
        let mut rdf = RdfAccumulator::new(30, 1.5, [0.0; 2], [extent; 2], [true; 2], xs.len());
        rdf.sample_positions(&xs, &ys);
        let result = rdf.result();
        // The first-neighbour shell at r = 1 holds exactly 4 neighbours per particle:
        // 200 unordered pairs on a periodic 10x10 lattice.
        let k = (1.0 / result.bin_width) as usize;
        assert_eq!(result.counts[k], 200, "{:?}", result.counts);
        assert_eq!(result.counts[..k].iter().sum::<u64>(), 0, "nothing closer than the spacing");
        let (peak_r, peak_g) = result.peak().expect("a lattice has a peak");
        assert!((peak_r - 1.025).abs() < 1e-12 && peak_g > 1.0, "peak at {peak_r}, g = {peak_g}");
    }

    #[test]
    #[should_panic(expected = "exceeds half the periodic box")]
    fn an_rdf_reaching_past_the_minimum_image_is_rejected() {
        RdfAccumulator::new(10, 6.0, [0.0; 2], [10.0; 2], [true; 2], 1);
    }

    #[test]
    fn msd_averages_squared_displacements() {
        assert_eq!(mean_squared_displacement(&[], &[]), 0.0);
        assert!((mean_squared_displacement(&[3.0, 0.0], &[4.0, 2.0]) - 14.5).abs() < 1e-12);
    }
}
