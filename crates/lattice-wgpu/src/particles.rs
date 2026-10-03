//! Resident gravity and Lennard–Jones particles. CPU force laws remain the reference.
//! A sorted cell list makes gather order deterministic without floating atomics.
use crate::GpuDevice;
use lattice_compute::{Device, DeviceError, KernelKey, KernelSource, Usage};

/// Immutable data and force parameters for a velocity-Verlet run, in SI units.
#[derive(Clone, Copy, Debug)]
pub struct ParticleSetup<'a> {
    pub positions: &'a [[f64; 2]],
    pub velocities: &'a [[f64; 2]],
    pub masses: &'a [f64],
    pub gravity: [f64; 2],
    /// epsilon, sigma, cutoff; absent for local gravity alone.
    pub lennard_jones: Option<[f64; 3]>,
    pub origin: [f64; 2],
    pub extent: [f64; 2],
    /// Both axes periodic, or both open. Other boundary modes require a separate kernel.
    pub periodic: bool,
    pub dt: f64,
}

/// A snapshot requested explicitly for observations or export, never for stepping.
#[derive(Debug)]
pub struct ParticleSnapshot {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub forces: Vec<[f64; 2]>,
}

#[derive(Debug)]
pub struct GpuParticles {
    particles: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipelines: Vec<wgpu::ComputePipeline>,
    count: usize,
    cells: usize,
    lj: bool,
    steps: usize,
}

impl GpuParticles {
    pub fn new(device: &GpuDevice, setup: ParticleSetup<'_>) -> Result<Self, DeviceError> {
        let n = setup.positions.len();
        let bad = |s: &str| DeviceError::Backend(s.to_string());
        if n != setup.velocities.len() || n != setup.masses.len() || n > u32::MAX as usize {
            return Err(bad("particle arrays must have equal representable lengths"));
        }
        let finite = |v: f64| v.is_finite() && (v as f32).is_finite();
        if !finite(setup.dt)
            || setup.dt as f32 <= 0.0
            || setup.extent.iter().any(|&x| !finite(x) || x as f32 <= 0.0)
            || setup
                .origin
                .iter()
                .chain(&setup.gravity)
                .any(|&x| !finite(x))
            || setup
                .positions
                .iter()
                .chain(setup.velocities)
                .flatten()
                .any(|&x| !finite(x))
            || setup
                .masses
                .iter()
                .any(|&m| !finite(m) || m as f32 <= 0.0 || !finite(1.0 / m))
        {
            return Err(bad(
                "particle setup requires finite f32 geometry, positive timestep and positive finite masses",
            ));
        }
        let lj = setup.lennard_jones;
        if lj.is_some_and(|p| {
            p.iter().any(|&v| !finite(v) || v as f32 <= 0.0)
                || !finite(24.0 * p[0])
                || !finite(p[1] * p[1])
                || !finite(p[2] * p[2])
                || ((p[1] * p[1]) as f32) < f32::MIN_POSITIVE
                || ((p[2] * p[2]) as f32) < f32::MIN_POSITIVE
        }) {
            return Err(bad(
                "Lennard-Jones parameters and their squares must be positive finite f32 values",
            ));
        }
        let dims = lj.map_or([1, 1], |p| {
            setup.extent.map(|v| (v / p[2]).floor().max(1.0) as usize)
        });
        let cells = dims[0]
            .checked_mul(dims[1])
            .ok_or_else(|| bad("cell count overflow"))?;
        if cells > 1_000_000 {
            return Err(bad("particle cell list exceeds one million cells"));
        }
        let dispatch_limit =
            device.device.limits().max_compute_workgroups_per_dimension as usize * 64;
        if n > dispatch_limit || cells > dispatch_limit {
            return Err(bad(
                "particle dispatch exceeds this device's workgroup limit",
            ));
        }
        let mut packed = vec![0.0; n.max(1) * 8];
        for i in 0..n {
            for (lane, value) in [
                setup.positions[i][0],
                setup.positions[i][1],
                setup.velocities[i][0],
                setup.velocities[i][1],
                0.0,
                0.0,
                setup.masses[i],
                1.0 / setup.masses[i],
            ]
            .into_iter()
            .enumerate()
            {
                packed[lane * n + i] = value;
            }
        }
        let particles = device.upload(&packed, Usage::Readback)?.buffer;
        let heads = device.alloc(cells, Usage::Resident)?.buffer;
        let links = device.alloc(n.max(1), Usage::Resident)?.buffer;
        let neighbors = topology(dims, setup.periodic);
        let neighbor_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle neighbor cells"),
            size: (neighbors.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bytes: Vec<u8> = neighbors.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.queue.write_buffer(&neighbor_buffer, 0, &bytes);
        let mut params = [0u8; 64];
        for (slot, v) in [
            (0, n as u32),
            (1, dims[0] as u32),
            (2, dims[1] as u32),
            (3, u32::from(setup.periodic)),
            (14, u32::from(lj.is_some())),
        ] {
            params[slot * 4..slot * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let [epsilon, sigma, cutoff] = lj.unwrap_or([0.0, 1.0, 1.0]);
        for (slot, v) in [
            (4, setup.gravity[0]),
            (5, setup.gravity[1]),
            (6, 24.0 * epsilon),
            (7, sigma * sigma),
            (8, setup.origin[0]),
            (9, setup.origin[1]),
            (10, setup.extent[0]),
            (11, setup.extent[1]),
            (12, setup.dt),
            (13, cutoff * cutoff),
        ] {
            params[slot * 4..slot * 4 + 4].copy_from_slice(&(v as f32).to_le_bytes());
        }
        let uniform = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle parameters"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&uniform, 0, &params);
        let source = KernelSource::new("particles.verlet", include_str!("particles.wgsl"));
        let key = KernelKey::new(
            &source,
            device.backend(),
            device.precision(),
            device.capabilities(),
        );
        let kernels = device.pipeline_set(
            key,
            &source,
            &[
                crate::bindings::uniform(0),
                crate::bindings::storage(1, false),
                crate::bindings::storage(2, false),
                crate::bindings::storage(3, false),
                crate::bindings::storage(4, true),
            ],
            &["clear", "bin", "sort", "forces", "drift", "kick"],
        )?;
        let layout = kernels.layout;
        let pipelines = kernels.pipelines;
        let group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particles"),
            layout: &layout,
            entries: &[
                crate::bindings::entry(0, &uniform),
                crate::bindings::entry(1, &particles),
                crate::bindings::entry(2, &heads),
                crate::bindings::entry(3, &links),
                crate::bindings::entry(4, &neighbor_buffer),
            ],
        });
        let solver = Self {
            particles,
            group,
            pipelines,
            count: n,
            cells,
            lj: lj.is_some(),
            steps: 0,
        };
        let mut encoder = device.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            solver.refresh(&mut pass);
        }
        device.submit_and_wait(encoder)?;
        Ok(solver)
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, kernel: usize, count: usize) {
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_pipeline(&self.pipelines[kernel]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    fn refresh(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.lj {
            self.dispatch(pass, 0, self.cells);
            self.dispatch(pass, 1, self.count);
            self.dispatch(pass, 2, self.cells);
        }
        self.dispatch(pass, 3, self.count);
    }
    /// Append work to the caller's command stream. The caller owns submission and fences.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder, steps: usize) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for _ in 0..steps {
            self.dispatch(&mut pass, 4, self.count);
            // Uniform gravity and masses are immutable: the initial force buffer
            // remains exact for every step. Only LJ requires a force refresh.
            if self.lj {
                self.refresh(&mut pass);
            }
            self.dispatch(&mut pass, 5, self.count);
        }
        self.steps += steps;
    }
    pub fn run(&mut self, device: &GpuDevice, steps: usize) -> Result<(), DeviceError> {
        let mut encoder = device.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, steps);
        device.submit_and_wait(encoder)
    }
    pub fn snapshot(&self, device: &GpuDevice) -> Result<ParticleSnapshot, DeviceError> {
        let mut data = vec![0.0; self.count * 8];
        device.read_raw(&self.particles, &mut data)?;
        let mut out = ParticleSnapshot {
            positions: Vec::with_capacity(self.count),
            velocities: Vec::with_capacity(self.count),
            forces: Vec::with_capacity(self.count),
        };
        for i in 0..self.count {
            let n = self.count;
            out.positions.push([data[i], data[n + i]]);
            out.velocities.push([data[2 * n + i], data[3 * n + i]]);
            out.forces.push([data[4 * n + i], data[5 * n + i]]);
        }
        Ok(out)
    }
    /// Read-only SoA rendering input: x, y, vx, vy, fx, fy, mass, inverse mass,
    /// each `len()` scalars. Submit compute before drawing on the same device/queue.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.particles
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    pub fn memory_bytes(&self) -> usize {
        self.count * 36 + self.cells * 40 + 64
    }
}

fn topology([nx, ny]: [usize; 2], periodic: bool) -> Vec<u32> {
    let mut result = vec![u32::MAX; nx * ny * 9];
    for y in 0..ny {
        for x in 0..nx {
            let mut adjacent = Vec::with_capacity(9);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (mut i, mut j) = (x as i64 + dx, y as i64 + dy);
                    if periodic {
                        i = i.rem_euclid(nx as i64);
                        j = j.rem_euclid(ny as i64);
                    }
                    if i >= 0 && j >= 0 && i < nx as i64 && j < ny as i64 {
                        adjacent.push(j as u32 * nx as u32 + i as u32);
                    }
                }
            }
            adjacent.sort_unstable();
            adjacent.dedup();
            let start = (y * nx + x) * 9;
            result[start..start + adjacent.len()].copy_from_slice(&adjacent);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn tiny_periodic_topologies_do_not_double_count_cells() {
        for dims in [[1, 1], [1, 2], [2, 2], [3, 3]] {
            let topology = super::topology(dims, true);
            for neighbors in topology.chunks_exact(9) {
                let cells: Vec<_> = neighbors
                    .iter()
                    .copied()
                    .filter(|&n| n != u32::MAX)
                    .collect();
                assert_eq!(cells.len(), dims[0] * dims[1]);
                assert!(cells.windows(2).all(|p| p[0] < p[1]));
            }
        }
    }
}
