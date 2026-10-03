//! Uniform cell list for local pair interactions.
//!
//! Spec §11.2: *"The default local-interaction accelerator is a cell-linked list /
//! uniform spatial hash in 2D."* Evaluating every pair is O(N²); binning particles
//! into cells at least as wide as the interaction cutoff makes it O(N) for a
//! roughly uniform system, because every partner within the cutoff must lie in the
//! 3×3 block of cells around the particle's own.
//!
//! # Layout
//!
//! Particles are stored in CSR form — one flat `items` array grouped by cell, plus a
//! `starts` offset array — built with a counting sort. This keeps a cell's members
//! contiguous, which matters more for the inner loop than the O(N) build does:
//!
//! ```text
//!   starts: [0, 2, 2, 5, ...]        items: [7, 3 | | 1, 9, 4 | ...]
//!            ^cell0 ^cell1(empty)            cell0    cell2
//! ```
//!
//! # Each pair exactly once
//!
//! The 3×3 scan would visit every pair twice, so pairs are filtered by `j > i`. That
//! is the whole deduplication rule — with one wrinkle: on a periodic axis with fewer
//! than three cells, several of the nine offsets alias the *same* cell, which would
//! reintroduce duplicates. Neighbour cells are therefore deduplicated per cell before
//! the particle loop rather than assumed distinct.

use lattice_ir::ParticleStore;

use crate::image::MinimumImage;

/// Uniform spatial binning over a rectangular region.
#[derive(Clone, Debug)]
pub struct CellList {
    nx: usize,
    ny: usize,
    cell_dx: f64,
    cell_dy: f64,
    origin: [f64; 2],
    extent: [f64; 2],
    periodic: [bool; 2],
    /// The minimum-image rule for this box, applied to every reported separation.
    image: MinimumImage,
    /// CSR offsets, length `nx*ny + 1`.
    starts: Vec<u32>,
    /// Particle slots grouped by cell.
    items: Vec<u32>,
    /// Scratch: per-cell counts during build, then per-cell write cursors.
    cursor: Vec<u32>,
    /// Deduplicated neighbour cells for each cell, `9` slots each.
    neighbor_cells: Vec<u32>,
    /// How many of each cell's nine slots are populated.
    neighbor_counts: Vec<u8>,
    /// The cutoff this list was sized for.
    cutoff: f64,
}

impl CellList {
    /// Build a cell list covering `extent` from `origin`, with cells no smaller than
    /// `cutoff`, sized for up to `capacity` particles.
    ///
    /// # Panics
    ///
    /// If `cutoff` is not positive and finite, or the extent is degenerate. A
    /// zero cutoff would ask for infinitely many cells.
    pub fn new(
        origin: [f64; 2],
        extent: [f64; 2],
        periodic: [bool; 2],
        cutoff: f64,
        capacity: usize,
    ) -> Self {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive and finite, got {cutoff}");
        assert!(
            extent[0] > 0.0 && extent[1] > 0.0 && extent[0].is_finite() && extent[1].is_finite(),
            "cell list extent must be positive and finite, got {extent:?}"
        );

        // Cells must be at least `cutoff` wide, so at least one full cutoff radius is
        // covered by the 3x3 block. `floor` rather than `round` guarantees that.
        let nx = ((extent[0] / cutoff).floor() as usize).max(1);
        let ny = ((extent[1] / cutoff).floor() as usize).max(1);
        let cells = nx * ny;

        let mut list = Self {
            nx,
            ny,
            cell_dx: extent[0] / nx as f64,
            cell_dy: extent[1] / ny as f64,
            origin,
            extent,
            periodic,
            image: MinimumImage::new(extent, periodic),
            starts: vec![0; cells + 1],
            items: vec![0; capacity],
            cursor: vec![0; cells],
            neighbor_cells: vec![0; cells * 9],
            neighbor_counts: vec![0; cells],
            cutoff,
        };
        list.build_neighbor_table();
        list
    }

    /// Cells along x.
    pub fn nx(&self) -> usize {
        self.nx
    }
    /// Cells along y.
    pub fn ny(&self) -> usize {
        self.ny
    }
    /// Total cells.
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }
    /// The interaction cutoff this list was built for.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }
    /// Region size.
    pub fn extent(&self) -> [f64; 2] {
        self.extent
    }
    /// Which axes wrap.
    pub fn periodic(&self) -> [bool; 2] {
        self.periodic
    }
    /// How many particles the list was sized for.
    pub fn capacity(&self) -> usize {
        self.items.len()
    }
    /// The minimum-image rule this list measures separations with.
    pub fn image(&self) -> MinimumImage {
        self.image
    }
    /// Bytes held, for the memory report (§19.3).
    pub fn memory_bytes(&self) -> usize {
        // CSR offsets, item list, per-cell cursor, and the 3×3 neighbour table.
        (self.starts.len() + self.items.len() + self.cursor.len() + self.neighbor_cells.len())
            * size_of::<u32>()
            + self.neighbor_counts.len()
    }

    /// Precompute each cell's deduplicated 3×3 neighbourhood.
    ///
    /// Done once at construction rather than per rebuild: the topology depends only
    /// on the grid, not on where the particles are.
    fn build_neighbor_table(&mut self) {
        for cy in 0..self.ny {
            for cx in 0..self.nx {
                let cell = cy * self.nx + cx;
                let mut found: [u32; 9] = [0; 9];
                let mut count = 0usize;
                for oy in -1i64..=1 {
                    for ox in -1i64..=1 {
                        let Some(n) = self.offset_cell(cx, cy, ox, oy) else { continue };
                        let n = n as u32;
                        // A small linear scan: at most nine entries, and this runs once.
                        if !found[..count].contains(&n) {
                            found[count] = n;
                            count += 1;
                        }
                    }
                }
                self.neighbor_cells[cell * 9..cell * 9 + count].copy_from_slice(&found[..count]);
                self.neighbor_counts[cell] = count as u8;
            }
        }
    }

    /// The cell at `(cx + ox, cy + oy)`, wrapping on periodic axes and returning
    /// `None` when the offset leaves a non-periodic domain.
    fn offset_cell(&self, cx: usize, cy: usize, ox: i64, oy: i64) -> Option<usize> {
        let i = cx as i64 + ox;
        let j = cy as i64 + oy;
        let i = if self.periodic[0] {
            i.rem_euclid(self.nx as i64)
        } else if i < 0 || i >= self.nx as i64 {
            return None;
        } else {
            i
        };
        let j = if self.periodic[1] {
            j.rem_euclid(self.ny as i64)
        } else if j < 0 || j >= self.ny as i64 {
            return None;
        } else {
            j
        };
        Some(j as usize * self.nx + i as usize)
    }

    /// The cell containing a point.
    ///
    /// Points outside a non-periodic axis clamp to the edge cell rather than being
    /// dropped: a particle that has drifted just outside the region should still
    /// interact with its neighbours instead of silently losing all its forces.
    fn cell_of(&self, x: f64, y: f64) -> usize {
        let fx = (x - self.origin[0]) / self.cell_dx;
        let fy = (y - self.origin[1]) / self.cell_dy;
        let i = Self::axis_index(fx, self.nx, self.periodic[0]);
        let j = Self::axis_index(fy, self.ny, self.periodic[1]);
        j * self.nx + i
    }

    fn axis_index(f: f64, n: usize, periodic: bool) -> usize {
        if !f.is_finite() {
            // A NaN position is a bug elsewhere; binning it at 0 keeps this function
            // total so the caller's NaN monitor is what reports the real problem.
            return 0;
        }
        let k = f.floor() as i64;
        if periodic {
            k.rem_euclid(n as i64) as usize
        } else {
            k.clamp(0, n as i64 - 1) as usize
        }
    }

    /// Re-bin every live particle. Call once per step, after positions change.
    pub fn rebuild(&mut self, store: &ParticleStore) {
        self.rebuild_from(store.pos_x(), store.pos_y());
    }

    /// Re-bin from raw position slices.
    ///
    /// # Panics
    ///
    /// If there are more particles than the capacity the list was built for.
    pub fn rebuild_from(&mut self, pos_x: &[f64], pos_y: &[f64]) {
        let n = pos_x.len();
        assert!(
            n <= self.items.len(),
            "cell list built for {} particles but asked to bin {n}",
            self.items.len()
        );

        let cells = self.cursor.len();

        // Counting sort, pass 1: how many land in each cell.
        self.cursor.fill(0);
        for k in 0..n {
            let c = self.cell_of(pos_x[k], pos_y[k]);
            self.cursor[c] += 1;
        }

        // Prefix sum into CSR offsets.
        let mut running = 0u32;
        for (c, start) in self.starts.iter_mut().enumerate().take(cells) {
            *start = running;
            running += self.cursor[c];
        }
        self.starts[cells] = running;

        // Pass 2: scatter, reusing `cursor` as the per-cell write head.
        self.cursor.copy_from_slice(&self.starts[..cells]);
        for k in 0..n {
            let c = self.cell_of(pos_x[k], pos_y[k]);
            let slot = self.cursor[c] as usize;
            self.items[slot] = k as u32;
            self.cursor[c] += 1;
        }
    }

    /// Particle slots currently binned in `cell`.
    pub fn cell_members(&self, cell: usize) -> &[u32] {
        let start = self.starts[cell] as usize;
        let end = self.starts[cell + 1] as usize;
        &self.items[start..end]
    }

    /// Shortest separation between two coordinates under the periodic convention.
    ///
    /// On a periodic axis the shortest image may be through the boundary, so a raw
    /// difference is wrong for any particle near an edge. Applied here once, rather
    /// than remembered separately by every force law — see [`MinimumImage`].
    #[inline]
    fn minimum_image(&self, dx: f64, dy: f64) -> (f64, f64) {
        self.image.separation(dx, dy)
    }

    /// Visit every distinct pair closer than the cutoff.
    ///
    /// The callback receives `(i, j, dx, dy, r2)` where `i < j`, `dx = x[j] - x[i]`
    /// under the minimum-image convention, and `r2 = dx² + dy²`. Coincident particles
    /// (`r2 == 0`) are skipped, since every pair force divides by `r`.
    pub fn for_each_pair(
        &self,
        pos_x: &[f64],
        pos_y: &[f64],
        mut visit: impl FnMut(usize, usize, f64, f64, f64),
    ) {
        let cutoff2 = self.cutoff * self.cutoff;
        for cell in 0..self.cell_count() {
            let members = self.cell_members(cell);
            if members.is_empty() {
                continue;
            }
            let neighbor_start = cell * 9;
            let neighbor_end = neighbor_start + self.neighbor_counts[cell] as usize;
            let neighbors = &self.neighbor_cells[neighbor_start..neighbor_end];

            for &pi in members {
                let i = pi as usize;
                let (xi, yi) = (pos_x[i], pos_y[i]);
                for &nc in neighbors {
                    for &pj in self.cell_members(nc as usize) {
                        let j = pj as usize;
                        if j <= i {
                            continue;
                        }
                        let (dx, dy) = self.minimum_image(pos_x[j] - xi, pos_y[j] - yi);
                        let r2 = dx * dx + dy * dy;
                        if r2 < cutoff2 && r2 > 0.0 {
                            visit(i, j, dx, dy, r2);
                        }
                    }
                }
            }
        }
    }

    /// Number of pairs the cell list currently reports.
    ///
    /// Used by tests and by the benchmark harness to report neighbour-list density.
    pub fn pair_count(&self, pos_x: &[f64], pos_y: &[f64]) -> usize {
        let mut count = 0;
        self.for_each_pair(pos_x, pos_y, |_, _, _, _, _| count += 1);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::ParticleSpec;

    /// Reference implementation: every pair, no acceleration structure. The cell list
    /// must agree with this exactly — same pairs, same separations.
    fn brute_force_pairs(
        pos_x: &[f64],
        pos_y: &[f64],
        cutoff: f64,
        extent: [f64; 2],
        periodic: [bool; 2],
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let cutoff2 = cutoff * cutoff;
        for i in 0..pos_x.len() {
            for j in (i + 1)..pos_x.len() {
                let mut dx = pos_x[j] - pos_x[i];
                let mut dy = pos_y[j] - pos_y[i];
                if periodic[0] {
                    dx -= extent[0] * (dx / extent[0]).round();
                }
                if periodic[1] {
                    dy -= extent[1] * (dy / extent[1]).round();
                }
                let r2 = dx * dx + dy * dy;
                if r2 < cutoff2 && r2 > 0.0 {
                    out.push((i, j));
                }
            }
        }
        out.sort_unstable();
        out
    }

    fn lattice_positions(n: usize, spacing: f64) -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for j in 0..n {
            for i in 0..n {
                xs.push((i as f64 + 0.5) * spacing);
                ys.push((j as f64 + 0.5) * spacing);
            }
        }
        (xs, ys)
    }

    #[test]
    fn cells_are_at_least_as_wide_as_the_cutoff() {
        let cl = CellList::new([0.0, 0.0], [10.0, 10.0], [false, false], 1.5, 100);
        assert!(cl.cell_dx >= 1.5, "cell width {} < cutoff", cl.cell_dx);
        assert!(cl.cell_dy >= 1.5);
        assert_eq!(cl.nx(), 6);
    }

    #[test]
    fn a_region_smaller_than_the_cutoff_becomes_one_cell() {
        let cl = CellList::new([0.0, 0.0], [1.0, 1.0], [false, false], 5.0, 10);
        assert_eq!(cl.cell_count(), 1);
    }

    #[test]
    fn agrees_with_brute_force_on_an_open_domain() {
        let (xs, ys) = lattice_positions(12, 1.0);
        let cutoff = 2.2;
        let mut cl = CellList::new([0.0, 0.0], [12.0, 12.0], [false, false], cutoff, xs.len());
        cl.rebuild_from(&xs, &ys);

        let mut found = Vec::new();
        cl.for_each_pair(&xs, &ys, |i, j, _, _, _| found.push((i, j)));
        found.sort_unstable();

        let expected = brute_force_pairs(&xs, &ys, cutoff, [12.0, 12.0], [false, false]);
        assert_eq!(found, expected);
        assert!(!expected.is_empty(), "test would be vacuous with no pairs");
    }

    #[test]
    fn agrees_with_brute_force_on_a_periodic_domain() {
        let (xs, ys) = lattice_positions(10, 1.0);
        let cutoff = 2.5;
        let extent = [10.0, 10.0];
        let mut cl = CellList::new([0.0, 0.0], extent, [true, true], cutoff, xs.len());
        cl.rebuild_from(&xs, &ys);

        let mut found = Vec::new();
        cl.for_each_pair(&xs, &ys, |i, j, _, _, _| found.push((i, j)));
        found.sort_unstable();

        let expected = brute_force_pairs(&xs, &ys, cutoff, extent, [true, true]);
        assert_eq!(found, expected);
    }

    /// The aliasing case: with a periodic axis of only two cells, several of the nine
    /// offsets land on the same cell. Without deduplication this reports each pair
    /// three times.
    #[test]
    fn periodic_domains_with_few_cells_do_not_duplicate_pairs() {
        for cells_per_axis in [1usize, 2, 3] {
            let cutoff = 1.0;
            let extent = [cells_per_axis as f64, cells_per_axis as f64];
            let xs = vec![0.2, 0.6, 0.4];
            let ys = vec![0.2, 0.3, 0.7];
            let mut cl = CellList::new([0.0, 0.0], extent, [true, true], cutoff, xs.len());
            assert_eq!(cl.nx(), cells_per_axis);
            cl.rebuild_from(&xs, &ys);

            let mut found = Vec::new();
            cl.for_each_pair(&xs, &ys, |i, j, _, _, _| found.push((i, j)));
            let mut sorted = found.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                found.len(),
                sorted.len(),
                "duplicate pairs at {cells_per_axis} cells per axis"
            );

            let expected = brute_force_pairs(&xs, &ys, cutoff, extent, [true, true]);
            found.sort_unstable();
            assert_eq!(found, expected, "at {cells_per_axis} cells per axis");
        }
    }

    /// Two particles either side of a periodic seam are neighbours, and their
    /// separation must be the short way round.
    #[test]
    fn minimum_image_wraps_across_the_seam() {
        let xs = vec![0.1, 9.9];
        let ys = vec![5.0, 5.0];
        let extent = [10.0, 10.0];
        let mut cl = CellList::new([0.0, 0.0], extent, [true, true], 1.0, 2);
        cl.rebuild_from(&xs, &ys);

        let mut pairs = Vec::new();
        cl.for_each_pair(&xs, &ys, |i, j, dx, dy, r2| pairs.push((i, j, dx, dy, r2)));
        assert_eq!(pairs.len(), 1, "the seam pair must be found");
        let (_, _, dx, _, r2) = pairs[0];
        // 0.2 apart through the boundary, not 9.8 across the domain.
        assert!((r2.sqrt() - 0.2).abs() < 1e-12, "separation {}", r2.sqrt());
        assert!(dx < 0.0, "the short path goes in -x from particle 0");
    }

    #[test]
    fn non_periodic_seams_are_not_neighbours() {
        let xs = vec![0.1, 9.9];
        let ys = vec![5.0, 5.0];
        let mut cl = CellList::new([0.0, 0.0], [10.0, 10.0], [false, false], 1.0, 2);
        cl.rebuild_from(&xs, &ys);
        assert_eq!(cl.pair_count(&xs, &ys), 0);
    }

    #[test]
    fn coincident_particles_are_skipped() {
        let xs = vec![1.0, 1.0];
        let ys = vec![1.0, 1.0];
        let mut cl = CellList::new([0.0, 0.0], [10.0, 10.0], [false, false], 2.0, 2);
        cl.rebuild_from(&xs, &ys);
        assert_eq!(cl.pair_count(&xs, &ys), 0, "r=0 would divide by zero in every force law");
    }

    #[test]
    fn particles_outside_an_open_domain_still_interact() {
        // A particle that has drifted past the edge clamps into the edge cell rather
        // than vanishing from the neighbour list.
        let xs = vec![-0.5, 0.2];
        let ys = vec![0.5, 0.5];
        let mut cl = CellList::new([0.0, 0.0], [10.0, 10.0], [false, false], 1.0, 2);
        cl.rebuild_from(&xs, &ys);
        assert_eq!(cl.pair_count(&xs, &ys), 1);
    }

    #[test]
    fn csr_layout_partitions_every_particle_exactly_once() {
        let (xs, ys) = lattice_positions(8, 1.1);
        let mut cl = CellList::new([0.0, 0.0], [10.0, 10.0], [false, false], 1.0, xs.len());
        cl.rebuild_from(&xs, &ys);

        let mut seen = vec![0u32; xs.len()];
        for cell in 0..cl.cell_count() {
            for &p in cl.cell_members(cell) {
                seen[p as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "every particle must appear in exactly one cell");
    }

    #[test]
    fn rebuild_is_idempotent() {
        let (xs, ys) = lattice_positions(6, 1.0);
        let mut cl = CellList::new([0.0, 0.0], [6.0, 6.0], [true, true], 1.5, xs.len());
        cl.rebuild_from(&xs, &ys);
        let first = cl.pair_count(&xs, &ys);
        cl.rebuild_from(&xs, &ys);
        assert_eq!(cl.pair_count(&xs, &ys), first);
    }

    #[test]
    fn rebuild_accepts_a_particle_store() {
        let mut store = ParticleStore::with_capacity(4);
        for i in 0..4 {
            store.spawn(ParticleSpec::at([i as f64 * 0.5, 0.5])).unwrap();
        }
        let mut cl = CellList::new([0.0, 0.0], [4.0, 4.0], [false, false], 1.0, 4);
        cl.rebuild(&store);
        assert!(cl.pair_count(store.pos_x(), store.pos_y()) > 0);
    }

    #[test]
    #[should_panic(expected = "cutoff must be positive")]
    fn zero_cutoff_is_rejected() {
        CellList::new([0.0, 0.0], [1.0, 1.0], [false, false], 0.0, 1);
    }

    #[test]
    #[should_panic(expected = "asked to bin")]
    fn exceeding_capacity_is_reported() {
        let mut cl = CellList::new([0.0, 0.0], [1.0, 1.0], [false, false], 1.0, 2);
        cl.rebuild_from(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
    }
}
