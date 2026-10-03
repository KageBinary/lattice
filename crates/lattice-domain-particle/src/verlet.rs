//! The neighbour list a force law sees: a cell list, optionally cached behind a
//! Verlet skin, with bonded pairs excluded.
//!
//! Spec §11.2 asks for *"a cell-linked list / uniform spatial hash in 2D, with a
//! Verlet-style neighbor cache for molecular workloads"*, and §12.4 makes the cache
//! optional: *"cell list + optional skin distance"*. Both are here, behind one
//! [`NeighborList`] so a force law does not care which is in use.
//!
//! # Why a skin helps
//!
//! A cell list with cells one cutoff wide scans the 3×3 block around every particle,
//! which is nine cutoff-squares of area for an interaction circle of area `π·r_c²` —
//! about 35% of the candidates are inside the cutoff. A Verlet list built once with
//! radius `r_c + s` keeps only pairs that can come within `r_c` before it is rebuilt,
//! so per step the candidate fraction rises to `r_c² / (r_c + s)²` (69% at
//! `s = 0.2·r_c`) and the cell list itself is rebuilt only when something has moved
//! far enough to matter. The list is exact, not approximate: the rebuild rule below is
//! sufficient for no pair to be missed.
//!
//! # The rebuild rule
//!
//! Every pair now within `r_c` was within `r_c + s` at the last build provided no
//! particle has moved more than `s / 2` since then — two particles approaching each
//! other can close the gap by at most `s` between them. [`VerletList::needs_rebuild`]
//! checks exactly that, on the current positions, before every force evaluation. The
//! check is what makes the list correct at any timestep: a step so large that a
//! particle crosses half the skin simply triggers a rebuild, it does not miss a pair.
//! What a large step costs is a rebuild every step, which the domain reports as a
//! rebuild count so a reader can see the skin was too small.
//!
//! # What changes, and what does not
//!
//! A Verlet list visits the same pairs as the cell list, in a different order. Forces
//! are therefore the same set of terms summed differently, and agree with the plain
//! cell list to floating-point round-off rather than to the bit. The
//! `neighbor_list_consistency` validation case measures both facts: no pair missed,
//! and the force disagreement against a bound derived from the neighbour count.
//! Without a skin the list *is* the cell list and nothing changes at all.
//!
//! # Exclusions
//!
//! A pair law does not act between particles that a bond holds together — the bond
//! *is* their interaction, and adding a Lennard-Jones repulsion on top would put the
//! equilibrium length somewhere other than where the bond says. Excluded pairs are
//! removed here, at the list, so no law has to know about topology.

use lattice_ir::ParticleStore;

use crate::image::MinimumImage;
use crate::neighbors::CellList;

/// Pairs of particle slots that pair laws must skip.
///
/// Stored as a per-particle adjacency table in CSR form, so membership is a scan of
/// a particle's (few) excluded partners. Particles with no bonds cost one comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exclusions {
    /// CSR offsets, length `count + 1`.
    starts: Vec<u32>,
    /// Excluded partners, grouped by particle and sorted within each group.
    partners: Vec<u32>,
}

impl Exclusions {
    /// No exclusions at all.
    pub fn none() -> Self {
        Self { starts: vec![0], partners: Vec::new() }
    }

    /// Exclude each listed pair, in both directions, for a store of `count` slots.
    ///
    /// Duplicate pairs and self-pairs are dropped rather than rejected: a topology
    /// that lists a bond twice still means one exclusion.
    ///
    /// # Panics
    ///
    /// If a pair names a slot at or beyond `count`.
    pub fn from_pairs(count: usize, pairs: &[[u32; 2]]) -> Self {
        let mut degree = vec![0u32; count];
        for &[a, b] in pairs {
            assert!(
                (a as usize) < count && (b as usize) < count,
                "exclusion [{a}, {b}] names a slot beyond the {count} particles"
            );
            if a != b {
                degree[a as usize] += 1;
                degree[b as usize] += 1;
            }
        }

        let mut starts = vec![0u32; count + 1];
        for i in 0..count {
            starts[i + 1] = starts[i] + degree[i];
        }
        let mut partners = vec![0u32; starts[count] as usize];
        let mut cursor: Vec<u32> = starts[..count].to_vec();
        for &[a, b] in pairs {
            if a == b {
                continue;
            }
            for (from, to) in [(a, b), (b, a)] {
                partners[cursor[from as usize] as usize] = to;
                cursor[from as usize] += 1;
            }
        }

        // Sort each group and drop repeats, then compact.
        let mut compact = Vec::with_capacity(partners.len());
        let mut new_starts = vec![0u32; count + 1];
        for i in 0..count {
            let group = &mut partners[starts[i] as usize..starts[i + 1] as usize];
            group.sort_unstable();
            let before = compact.len();
            for &p in group.iter() {
                if compact.len() == before || compact[compact.len() - 1] != p {
                    compact.push(p);
                }
            }
            new_starts[i + 1] = compact.len() as u32;
        }
        Self { starts: new_starts, partners: compact }
    }

    /// True when nothing is excluded.
    pub fn is_empty(&self) -> bool {
        self.partners.is_empty()
    }

    /// Number of excluded (unordered) pairs.
    pub fn len(&self) -> usize {
        self.partners.len() / 2
    }

    /// Whether the pair `(i, j)` is excluded.
    #[inline]
    pub fn contains(&self, i: usize, j: usize) -> bool {
        let Some(&start) = self.starts.get(i) else { return false };
        let Some(&end) = self.starts.get(i + 1) else { return false };
        self.partners[start as usize..end as usize].contains(&(j as u32))
    }
}

/// A cached pair list with a skin.
#[derive(Clone, Debug)]
pub struct VerletList {
    /// The interaction cutoff — pairs are *reported* within this radius.
    cutoff: f64,
    /// The skin — pairs are *stored* within `cutoff + skin`.
    skin: f64,
    /// CSR offsets by first particle, length `capacity + 1`.
    starts: Vec<u32>,
    /// Partners `j > i`, grouped by `i`.
    partners: Vec<u32>,
    /// Positions at the last build, for the displacement test.
    ref_x: Vec<f64>,
    ref_y: Vec<f64>,
    /// How many particles the last build covered.
    built_for: usize,
    /// Whether a build has happened at all.
    built: bool,
    image: MinimumImage,
    rebuilds: u64,
    /// Per-particle write cursor for the scatter pass, kept so a rebuild allocates
    /// nothing.
    cursor: Vec<u32>,
}

impl VerletList {
    fn new(cutoff: f64, skin: f64, capacity: usize, image: MinimumImage) -> Self {
        assert!(skin > 0.0 && skin.is_finite(), "a Verlet skin must be positive and finite, got {skin}");
        Self {
            cutoff,
            skin,
            starts: vec![0; capacity + 1],
            partners: Vec::new(),
            ref_x: vec![0.0; capacity],
            ref_y: vec![0.0; capacity],
            built_for: 0,
            built: false,
            image,
            rebuilds: 0,
            cursor: vec![0; capacity],
        }
    }

    /// The skin distance, metres.
    pub fn skin(&self) -> f64 {
        self.skin
    }

    /// How many times the list has been built.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// Number of stored candidate pairs.
    pub fn stored_pairs(&self) -> usize {
        self.partners.len()
    }

    /// True if some particle has moved more than half the skin since the last build,
    /// or if the list has never been built or the particle count changed.
    pub fn needs_rebuild(&self, pos_x: &[f64], pos_y: &[f64]) -> bool {
        if !self.built || pos_x.len() != self.built_for {
            return true;
        }
        let limit = 0.25 * self.skin * self.skin;
        for i in 0..pos_x.len() {
            let (dx, dy) = self.image.separation(pos_x[i] - self.ref_x[i], pos_y[i] - self.ref_y[i]);
            let moved = dx * dx + dy * dy;
            // A NaN position compares false and is left to the non-finite monitor.
            if moved > limit {
                return true;
            }
        }
        false
    }

    /// Rebuild from a cell list that has just been binned on the same positions and
    /// was sized for `cutoff + skin`.
    ///
    /// Two passes over the cell list — count, then fill — so the CSR layout is built
    /// without a temporary pair buffer. `partners` grows only when a build finds more
    /// pairs than any earlier one, which is the single allocation this list makes
    /// after construction; a fluid at steady density reaches that ceiling on its first
    /// few builds.
    fn rebuild(&mut self, cells: &CellList, pos_x: &[f64], pos_y: &[f64], exclusions: &Exclusions) {
        let n = pos_x.len();
        assert!(
            n < self.starts.len(),
            "Verlet list built for {} particles, asked for {n}",
            self.starts.len() - 1
        );
        let Self { starts, partners, cursor, ref_x, ref_y, .. } = self;

        // Pass 1: how many partners each particle has.
        starts[..=n].fill(0);
        cells.for_each_pair(pos_x, pos_y, |i, j, _, _, _| {
            if !exclusions.contains(i, j) {
                starts[i + 1] += 1;
            }
        });
        for i in 0..n {
            starts[i + 1] += starts[i];
        }
        let total = starts[n] as usize;
        if partners.len() < total {
            partners.resize(total, 0);
        }

        // Pass 2: scatter, with a per-particle write head.
        cursor[..n].copy_from_slice(&starts[..n]);
        cells.for_each_pair(pos_x, pos_y, |i, j, _, _, _| {
            if !exclusions.contains(i, j) {
                partners[cursor[i] as usize] = j as u32;
                cursor[i] += 1;
            }
        });

        ref_x[..n].copy_from_slice(pos_x);
        ref_y[..n].copy_from_slice(pos_y);
        self.built_for = n;
        self.built = true;
        self.rebuilds += 1;
    }

    /// Visit every stored pair currently closer than the cutoff.
    ///
    /// Same contract as [`CellList::for_each_pair`]: `i < j`, minimum-image
    /// separation, coincident pairs skipped.
    pub fn for_each_pair(
        &self,
        pos_x: &[f64],
        pos_y: &[f64],
        mut visit: impl FnMut(usize, usize, f64, f64, f64),
    ) {
        debug_assert!(self.built && pos_x.len() == self.built_for, "list must be rebuilt before use");
        let cutoff2 = self.cutoff * self.cutoff;
        let n = self.built_for.min(pos_x.len());
        for i in 0..n {
            let (xi, yi) = (pos_x[i], pos_y[i]);
            let start = self.starts[i] as usize;
            let end = self.starts[i + 1] as usize;
            for &pj in &self.partners[start..end] {
                let j = pj as usize;
                let (dx, dy) = self.image.separation(pos_x[j] - xi, pos_y[j] - yi);
                let r2 = dx * dx + dy * dy;
                if r2 < cutoff2 && r2 > 0.0 {
                    visit(i, j, dx, dy, r2);
                }
            }
        }
    }
}

/// The neighbour structure a domain owns and a force law reads.
#[derive(Clone, Debug)]
pub struct NeighborList {
    /// Sized for `cutoff` alone, or for `cutoff + skin` when a Verlet list is cached.
    cells: CellList,
    verlet: Option<VerletList>,
    exclusions: Exclusions,
    cutoff: f64,
    image: MinimumImage,
    updates: u64,
}

impl NeighborList {
    /// Build over `extent` from `origin`, for interactions reaching `cutoff`.
    ///
    /// With `skin` set, pairs are cached in a Verlet list and the cell list is sized
    /// for `cutoff + skin`; without it, the cell list is rebuilt on every update.
    ///
    /// # Panics
    ///
    /// On a non-positive cutoff or skin, or a degenerate extent — the same conditions
    /// [`CellList::new`] refuses.
    pub fn new(
        origin: [f64; 2],
        extent: [f64; 2],
        periodic: [bool; 2],
        cutoff: f64,
        skin: Option<f64>,
        capacity: usize,
        exclusions: Exclusions,
    ) -> Self {
        let image = MinimumImage::new(extent, periodic);
        let list_radius = cutoff + skin.unwrap_or(0.0);
        let cells = CellList::new(origin, extent, periodic, list_radius, capacity);
        let verlet = skin.map(|s| VerletList::new(cutoff, s, capacity, image));
        Self { cells, verlet, exclusions, cutoff, image, updates: 0 }
    }

    /// Bring the list up to date with the store's positions.
    ///
    /// Returns true when the cell list was re-binned — every call without a skin, and
    /// only on a skin crossing with one.
    pub fn update(&mut self, store: &ParticleStore) -> bool {
        self.update_from(store.pos_x(), store.pos_y())
    }

    /// [`NeighborList::update`] from raw position slices.
    pub fn update_from(&mut self, pos_x: &[f64], pos_y: &[f64]) -> bool {
        self.updates += 1;
        match self.verlet.as_mut() {
            Some(verlet) => {
                if verlet.needs_rebuild(pos_x, pos_y) {
                    self.cells.rebuild_from(pos_x, pos_y);
                    verlet.rebuild(&self.cells, pos_x, pos_y, &self.exclusions);
                    true
                } else {
                    false
                }
            }
            None => {
                self.cells.rebuild_from(pos_x, pos_y);
                true
            }
        }
    }

    /// Visit every interacting pair closer than the cutoff, excluded pairs omitted.
    ///
    /// `(i, j, dx, dy, r2)` with `i < j`, `dx = x[j] − x[i]` under the minimum image.
    pub fn for_each_pair(
        &self,
        pos_x: &[f64],
        pos_y: &[f64],
        mut visit: impl FnMut(usize, usize, f64, f64, f64),
    ) {
        match &self.verlet {
            Some(verlet) => verlet.for_each_pair(pos_x, pos_y, visit),
            None if self.exclusions.is_empty() => self.cells.for_each_pair(pos_x, pos_y, visit),
            None => {
                let exclusions = &self.exclusions;
                self.cells.for_each_pair(pos_x, pos_y, |i, j, dx, dy, r2| {
                    if !exclusions.contains(i, j) {
                        visit(i, j, dx, dy, r2);
                    }
                });
            }
        }
    }

    /// Number of interacting pairs at the given positions.
    pub fn pair_count(&self, pos_x: &[f64], pos_y: &[f64]) -> usize {
        let mut count = 0;
        self.for_each_pair(pos_x, pos_y, |_, _, _, _, _| count += 1);
        count
    }

    /// The interaction cutoff, metres.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// The Verlet skin, if one is in use.
    pub fn skin(&self) -> Option<f64> {
        self.verlet.as_ref().map(VerletList::skin)
    }

    /// The minimum-image rule this list measures with.
    pub fn image(&self) -> MinimumImage {
        self.image
    }

    /// The underlying cell list.
    pub fn cells(&self) -> &CellList {
        &self.cells
    }

    /// The excluded pairs.
    pub fn exclusions(&self) -> &Exclusions {
        &self.exclusions
    }

    /// How many times the cell list has been re-binned.
    ///
    /// Without a skin this equals [`NeighborList::updates`]; with one, the ratio is the
    /// average number of steps a build survives — the number that says whether the
    /// skin was worth its candidates.
    pub fn rebuilds(&self) -> u64 {
        match &self.verlet {
            Some(verlet) => verlet.rebuilds(),
            None => self.updates,
        }
    }

    /// How many times [`NeighborList::update`] has been called.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Bytes held, for the memory report.
    pub fn memory_bytes(&self) -> usize {
        let cell_count = self.cells.cell_count();
        let capacity = self.cells.capacity();
        // CSR offsets, item list, per-cell cursor, and the 3×3 neighbour table.
        let cells = (cell_count + 1 + cell_count + cell_count * 9) * size_of::<u32>()
            + capacity * size_of::<u32>()
            + cell_count;
        let verlet = self.verlet.as_ref().map_or(0, |v| {
            (v.starts.len() + v.partners.capacity() + v.cursor.capacity()) * size_of::<u32>()
                + (v.ref_x.len() + v.ref_y.len()) * size_of::<f64>()
        });
        let exclusions = (self.exclusions.starts.len() + self.exclusions.partners.len()) * size_of::<u32>();
        cells + verlet + exclusions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::Pcg32;

    fn brute_force_pairs(
        pos_x: &[f64],
        pos_y: &[f64],
        cutoff: f64,
        image: MinimumImage,
        exclusions: &Exclusions,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let cutoff2 = cutoff * cutoff;
        for i in 0..pos_x.len() {
            for j in (i + 1)..pos_x.len() {
                let (dx, dy) = image.separation(pos_x[j] - pos_x[i], pos_y[j] - pos_y[i]);
                let r2 = dx * dx + dy * dy;
                if r2 < cutoff2 && r2 > 0.0 && !exclusions.contains(i, j) {
                    out.push((i, j));
                }
            }
        }
        out
    }

    fn sorted_pairs(list: &NeighborList, pos_x: &[f64], pos_y: &[f64]) -> Vec<(usize, usize)> {
        let mut found = Vec::new();
        list.for_each_pair(pos_x, pos_y, |i, j, _, _, _| found.push((i, j)));
        found.sort_unstable();
        found
    }

    fn random_gas(n: usize, extent: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Pcg32::seed_from_u64(seed);
        let xs = (0..n).map(|_| rng.range(0.0, extent)).collect();
        let ys = (0..n).map(|_| rng.range(0.0, extent)).collect();
        (xs, ys)
    }

    #[test]
    fn exclusions_are_symmetric_and_deduplicated() {
        let ex = Exclusions::from_pairs(5, &[[0, 1], [1, 0], [3, 4], [2, 2]]);
        assert_eq!(ex.len(), 2, "one duplicate and one self-pair must collapse");
        assert!(ex.contains(0, 1) && ex.contains(1, 0));
        assert!(ex.contains(3, 4) && ex.contains(4, 3));
        assert!(!ex.contains(2, 2));
        assert!(!ex.contains(0, 4));
        assert!(!ex.contains(9, 0), "slots beyond the table are simply not excluded");
        assert!(Exclusions::none().is_empty());
    }

    #[test]
    #[should_panic(expected = "beyond the 3 particles")]
    fn an_exclusion_outside_the_store_is_rejected() {
        Exclusions::from_pairs(3, &[[0, 3]]);
    }

    #[test]
    fn without_a_skin_the_list_is_the_cell_list() {
        let (xs, ys) = random_gas(300, 20.0, 1);
        let image = MinimumImage::new([20.0, 20.0], [true, true]);
        let mut list = NeighborList::new([0.0; 2], [20.0; 2], [true; 2], 2.5, None, 300, Exclusions::none());
        assert!(list.update_from(&xs, &ys), "a plain cell list re-bins on every update");
        assert_eq!(sorted_pairs(&list, &xs, &ys), brute_force_pairs(&xs, &ys, 2.5, image, &Exclusions::none()));
        assert_eq!(list.rebuilds(), 1);
        assert!(list.update_from(&xs, &ys));
        assert_eq!(list.rebuilds(), 2);
    }

    #[test]
    fn a_verlet_list_reports_exactly_the_cell_list_pairs() {
        let (xs, ys) = random_gas(400, 25.0, 2);
        let image = MinimumImage::new([25.0, 25.0], [true, true]);
        let mut list = NeighborList::new([0.0; 2], [25.0; 2], [true; 2], 2.5, Some(0.4), 400, Exclusions::none());
        assert!(list.update_from(&xs, &ys), "the first update must build");
        assert_eq!(sorted_pairs(&list, &xs, &ys), brute_force_pairs(&xs, &ys, 2.5, image, &Exclusions::none()));
        assert!(!list.update_from(&xs, &ys), "nothing moved, so nothing rebuilds");
        assert_eq!(list.rebuilds(), 1);
        assert_eq!(list.updates(), 2);
    }

    /// The correctness argument: move every particle by a random step below half
    /// the skin, many times, and the cached list must still report exactly the pairs
    /// a fresh search would. Then move one particle past the threshold and the list
    /// must rebuild.
    #[test]
    fn a_verlet_list_never_misses_a_pair_while_displacements_stay_under_half_the_skin() {
        let n = 250;
        let extent = 18.0;
        let skin = 0.5;
        let (mut xs, mut ys) = random_gas(n, extent, 3);
        let image = MinimumImage::new([extent; 2], [true; 2]);
        let mut list = NeighborList::new([0.0; 2], [extent; 2], [true; 2], 2.0, Some(skin), n, Exclusions::none());
        list.update_from(&xs, &ys);

        let mut rng = Pcg32::seed_from_u64(11);
        let mut rebuilds_seen = 0;
        for _ in 0..40 {
            for i in 0..n {
                // Each particle drifts by at most 0.2·skin per move, so a rebuild
                // becomes necessary only after a few moves accumulate.
                xs[i] = (xs[i] + rng.range(-0.2 * skin, 0.2 * skin)).rem_euclid(extent);
                ys[i] = (ys[i] + rng.range(-0.2 * skin, 0.2 * skin)).rem_euclid(extent);
            }
            if list.update_from(&xs, &ys) {
                rebuilds_seen += 1;
            }
            assert_eq!(
                sorted_pairs(&list, &xs, &ys),
                brute_force_pairs(&xs, &ys, 2.0, image, &Exclusions::none()),
                "cached list disagreed with a fresh search"
            );
        }
        assert!(rebuilds_seen > 0 && rebuilds_seen < 40, "expected occasional rebuilds, saw {rebuilds_seen}");

        // One particle jumping half the skin plus a hair forces a rebuild.
        let before = list.rebuilds();
        list.update_from(&xs, &ys);
        assert_eq!(list.rebuilds(), before, "no movement, no rebuild");
        xs[0] = (xs[0] + 0.5 * skin + 1e-9).rem_euclid(extent);
        assert!(list.update_from(&xs, &ys));
        assert_eq!(list.rebuilds(), before + 1);
    }

    #[test]
    fn excluded_pairs_are_omitted_on_both_paths() {
        let (xs, ys) = random_gas(120, 10.0, 4);
        let image = MinimumImage::new([10.0; 2], [true; 2]);
        let pairs: Vec<[u32; 2]> = (0..119).map(|i| [i, i + 1]).collect();
        let exclusions = Exclusions::from_pairs(120, &pairs);
        let expected = brute_force_pairs(&xs, &ys, 3.0, image, &exclusions);
        let without = brute_force_pairs(&xs, &ys, 3.0, image, &Exclusions::none());
        assert!(expected.len() < without.len(), "the chain must exclude something");

        for skin in [None, Some(0.3)] {
            let mut list = NeighborList::new([0.0; 2], [10.0; 2], [true; 2], 3.0, skin, 120, exclusions.clone());
            list.update_from(&xs, &ys);
            assert_eq!(sorted_pairs(&list, &xs, &ys), expected, "skin {skin:?}");
        }
    }

    #[test]
    fn pair_separations_agree_between_the_two_paths() {
        let (xs, ys) = random_gas(200, 12.0, 5);
        let mut plain = NeighborList::new([0.0; 2], [12.0; 2], [true; 2], 2.5, None, 200, Exclusions::none());
        let mut cached = NeighborList::new([0.0; 2], [12.0; 2], [true; 2], 2.5, Some(0.5), 200, Exclusions::none());
        plain.update_from(&xs, &ys);
        cached.update_from(&xs, &ys);
        let collect = |list: &NeighborList| {
            let mut v = Vec::new();
            list.for_each_pair(&xs, &ys, |i, j, dx, dy, r2| v.push((i, j, dx.to_bits(), dy.to_bits(), r2.to_bits())));
            v.sort_unstable();
            v
        };
        // The same minimum-image arithmetic runs on both paths, so per-pair values
        // are identical to the bit; only the visiting order differs.
        assert_eq!(collect(&plain), collect(&cached));
    }

    #[test]
    fn a_shrinking_population_is_handled() {
        let (xs, ys) = random_gas(50, 8.0, 6);
        let mut list = NeighborList::new([0.0; 2], [8.0; 2], [true; 2], 2.0, Some(0.3), 50, Exclusions::none());
        list.update_from(&xs, &ys);
        assert!(list.update_from(&xs[..30], &ys[..30]), "a different count must rebuild");
        let image = MinimumImage::new([8.0; 2], [true; 2]);
        assert_eq!(
            sorted_pairs(&list, &xs[..30], &ys[..30]),
            brute_force_pairs(&xs[..30], &ys[..30], 2.0, image, &Exclusions::none())
        );
    }

    #[test]
    #[should_panic(expected = "positive and finite")]
    fn a_zero_skin_is_rejected() {
        NeighborList::new([0.0; 2], [1.0; 2], [false; 2], 1.0, Some(0.0), 1, Exclusions::none());
    }
}
