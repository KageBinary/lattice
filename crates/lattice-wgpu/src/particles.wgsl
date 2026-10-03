// Velocity Verlet. Each invocation owns one particle's output. Neighbour cells
// and particle indices are traversed in fixed ascending order after bin sorting.
struct Params {
    n: u32, cx: u32, cy: u32, periodic: u32,
    gravity: vec2<f32>, epsilon24: f32, sigma2: f32,
    origin: vec2<f32>, extent: vec2<f32>,
    dt: f32, cutoff2: f32, lj: u32, pad: u32,
};
@group(0) @binding(0) var<uniform> cfg: Params;
// Eight SoA lanes: x, y, vx, vy, fx, fy, mass, inverse mass. Adjacent
// invocations access adjacent scalars, including in the force gather.
@group(0) @binding(1) var<storage, read_write> particles: array<f32>;
@group(0) @binding(2) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> links: array<u32>;
@group(0) @binding(4) var<storage, read> neighbors: array<u32>;
const END: u32 = 0xffffffffu;

fn position(i:u32)->vec2<f32> {return vec2<f32>(particles[i],particles[cfg.n+i]);}
fn velocity(i:u32)->vec2<f32> {return vec2<f32>(particles[2u*cfg.n+i],particles[3u*cfg.n+i]);}
fn force_at(i:u32)->vec2<f32> {return vec2<f32>(particles[4u*cfg.n+i],particles[5u*cfg.n+i]);}

fn wrap(p: vec2<f32>) -> vec2<f32> {
    if (cfg.periodic == 0u) { return p; }
    let v = p - cfg.origin;
    return cfg.origin + v - floor(v / cfg.extent) * cfg.extent;
}
fn cell(p: vec2<f32>) -> u32 {
    let c = vec2<u32>(clamp(floor((p - cfg.origin) / cfg.extent * vec2<f32>(f32(cfg.cx), f32(cfg.cy))), vec2<f32>(0.0), vec2<f32>(f32(cfg.cx-1u), f32(cfg.cy-1u))));
    return c.y * cfg.cx + c.x;
}
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < cfg.cx*cfg.cy) { atomicStore(&heads[id.x], END); }
}
@compute @workgroup_size(64)
fn bin(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.n) { return; }
    let p = wrap(position(id.x));
    particles[id.x] = p.x;
    particles[cfg.n+id.x] = p.y;
    links[id.x] = atomicExchange(&heads[cell(p)], id.x);
}
// One invocation owns a cell and its disjoint links. Sort the atomic insertion
// order away before gathering forces. Cost is quadratic in cell occupancy only.
@compute @workgroup_size(64)
fn sort(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.cx*cfg.cy) { return; }
    var item = atomicLoad(&heads[id.x]);
    var sorted = END;
    while (item != END) {
        let next = links[item];
        if (item < sorted) {
            links[item] = sorted;
            sorted = item;
        } else {
            var cursor = sorted;
            while (links[cursor] < item) { cursor = links[cursor]; }
            links[item] = links[cursor];
            links[cursor] = item;
        }
        item = next;
    }
    atomicStore(&heads[id.x], sorted);
}
@compute @workgroup_size(64)
fn forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.n) { return; }
    var force = particles[6u*cfg.n+i] * cfg.gravity;
    if (cfg.lj != 0u) {
        let c = cell(position(i));
        for (var offset = 0u; offset < 9u; offset++) {
            let neighbor = neighbors[c*9u + offset];
            if (neighbor == END) { break; }
            var j = atomicLoad(&heads[neighbor]);
            while (j != END) {
                var delta = position(j) - position(i);
                if (cfg.periodic != 0u) {
                    // Rust f64::round rounds half ties away from zero.
                    let ratio = delta / cfg.extent;
                    delta -= cfg.extent * sign(ratio) * floor(abs(ratio) + vec2<f32>(0.5));
                }
                let r2 = delta.x*delta.x + delta.y*delta.y;
                if (r2 > 0.0 && r2 < cfg.cutoff2 && i != j) {
                    let inv = 1.0 / r2;
                    let q = cfg.sigma2 * inv;
                    let s6 = q*q*q;
                    let coeff = cfg.epsilon24 * inv * (2.0*s6*s6 - s6);
                    force -= coeff * delta;
                }
                j = links[j];
            }
        }
    }
    particles[4u*cfg.n+i] = force.x;
    particles[5u*cfg.n+i] = force.y;
}
@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.n) { return; }
    let v = velocity(i) + ((0.5*cfg.dt)*particles[7u*cfg.n+i])*force_at(i);
    let p = wrap(position(i) + cfg.dt * v);
    particles[i]=p.x;particles[cfg.n+i]=p.y;
    particles[2u*cfg.n+i]=v.x;particles[3u*cfg.n+i]=v.y;
}
@compute @workgroup_size(64)
fn kick(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.n) { return; }
    let v = velocity(i) + ((0.5*cfg.dt)*particles[7u*cfg.n+i])*force_at(i);
    particles[2u*cfg.n+i] = v.x;
    particles[3u*cfg.n+i] = v.y;
}
