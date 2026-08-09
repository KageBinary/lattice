//! The portable backend against properties that hold whatever precision it runs at.
//!
//! Every test here skips when there is no adapter. §19.1's cross-backend level is where
//! the GPU is compared against the CPU reference; these are the properties the GPU must
//! have on its own, and they are the ones that would catch a transposed index or a
//! mis-sized dispatch before a comparison ever ran.

use lattice_compute::{Device, DeviceError, Precision};
use lattice_wgpu::{DiffusionSetup, GpuDevice, GpuDiffusion};

/// The grid every test here runs on. Not square, so a transposed index cannot pass.
const NX: usize = 96;
const NY: usize = 64;
const HALO: usize = 1;
const STRIDE: usize = NX + 2 * HALO;
const DIFFUSIVITY: f64 = 1e-3;

/// Open a device, or explain why the test is being skipped.
///
/// A machine without a GPU must not fail this suite, and must not silently pass it either
/// — the message is the difference.
fn device() -> Option<GpuDevice> {
    match GpuDevice::open_default() {
        Ok(device) => Some(device),
        Err(DeviceError::Unavailable { detail, .. }) => {
            eprintln!("SKIPPED: no portable GPU adapter available ({detail})");
            None
        }
        Err(error) => panic!("unexpected device error: {error}"),
    }
}

struct Problem {
    field: Vec<f64>,
    face_x: Vec<f64>,
    face_y: Vec<f64>,
    inv_dx2: f64,
    inv_dy2: f64,
    dt: f64,
}

impl Problem {
    /// A pulse on a uniform-diffusivity plate, at 80% of the explicit stability limit.
    fn new(initial: impl Fn(usize, usize) -> f64) -> Problem {
        let (dx, dy) = (1.0 / NX as f64, 0.75 / NY as f64);
        let (inv_dx2, inv_dy2) = (1.0 / (dx * dx), 1.0 / (dy * dy));

        let mut field = vec![0.0; STRIDE * (NY + 2 * HALO)];
        for j in 0..NY {
            for i in 0..NX {
                field[(j + HALO) * STRIDE + i + HALO] = initial(i, j);
            }
        }

        Problem {
            field,
            face_x: vec![DIFFUSIVITY; (NX + 1) * NY],
            face_y: vec![DIFFUSIVITY; NX * (NY + 1)],
            inv_dx2,
            inv_dy2,
            dt: 0.8 / (2.0 * DIFFUSIVITY * (inv_dx2 + inv_dy2)),
        }
    }

    fn setup(&self) -> DiffusionSetup<'_> {
        DiffusionSetup {
            nx: NX,
            ny: NY,
            halo: HALO,
            stride: STRIDE,
            inv_dx2: self.inv_dx2,
            inv_dy2: self.inv_dy2,
            field: &self.field,
            face_x: &self.face_x,
            face_y: &self.face_y,
            source: None,
        }
    }
}

fn pulse(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / NX as f64, j as f64 / NY as f64);
    let r2 = (x - 0.35).powi(2) + (y - 0.6).powi(2);
    300.0 + 60.0 * (-r2 / 0.01).exp()
}

/// The strongest property an insulated diffusion run has: it moves heat around and never
/// loses any. A mis-sized dispatch that skipped the last row would fail this, and so would
/// a halo kernel that wrote the wrong ghost.
#[test]
fn insulated_diffusion_conserves_heat() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse);
    let mut solver = GpuDiffusion::new(&device, problem.setup()).unwrap();

    let before: f64 = solver.interior(&device).unwrap().iter().sum();
    solver.run(&device, problem.dt, 400).unwrap();
    let after: f64 = solver.interior(&device).unwrap().iter().sum();

    let drift = (after - before).abs() / before.abs();
    eprintln!("heat drift over 400 steps: {drift:.3e} (f32 eps {:.3e})", Precision::Fast32.epsilon());
    assert!(drift < 1e-5, "insulated boundaries must not leak heat, drift was {drift:.3e}");
    assert_eq!(solver.steps(), 400);
}

/// The discrete maximum principle: below its stability limit, an explicit diffusion step
/// is a weighted average of a cell and its neighbours, so no cell can leave the range the
/// field started in. The peak falls, and nothing overshoots either end.
///
/// The trough is asserted *non-decreasing* rather than rising. The initial field is
/// `300 + 60·exp(-r²/0.01)`, so its minimum sits in a corner many cells from the pulse; in
/// 200 steps the heat that reaches there is smaller than an `f32` ulp of 300 and the
/// corner is still exactly 300. Asserting a strict rise would be asserting something about
/// the step count rather than about diffusion, and it failed for exactly that reason.
#[test]
fn diffusion_smooths_without_overshooting() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse);
    let mut solver = GpuDiffusion::new(&device, problem.setup()).unwrap();

    let start = solver.interior(&device).unwrap();
    let (low, high) = extremes(&start);
    solver.run(&device, problem.dt, 200).unwrap();
    let end = solver.interior(&device).unwrap();
    let (new_low, new_high) = extremes(&end);

    assert!(new_high < high, "the peak must fall: {new_high} vs {high}");
    assert!(new_high <= high, "no overshoot above the initial maximum");
    assert!(new_low >= low, "no undershoot below the initial minimum");
    assert!(end.iter().all(|value| value.is_finite()), "the run went non-finite");
}

/// A field with no gradient has nothing to diffuse. Under insulated boundaries it must
/// come back *exactly* — this is the one GPU result that can be asserted at zero
/// tolerance, because every flux term is a difference of equal numbers.
#[test]
fn a_uniform_field_is_a_fixed_point() {
    let Some(device) = device() else { return };
    let problem = Problem::new(|_, _| 300.0);
    let mut solver = GpuDiffusion::new(&device, problem.setup()).unwrap();

    solver.run(&device, problem.dt, 50).unwrap();

    for (index, value) in solver.interior(&device).unwrap().iter().enumerate() {
        assert_eq!(*value, 300.0, "cell {index} drifted from a uniform field");
    }
}

/// Symmetry the grid itself has: a pulse centred in x must stay centred in x. This is what
/// catches an off-by-one in the face-coefficient row indexing, which conservation alone
/// would not.
#[test]
fn a_symmetric_pulse_stays_symmetric() {
    let Some(device) = device() else { return };
    let problem = Problem::new(|i, j| {
        let x = (i as f64 + 0.5) / NX as f64;
        let y = (j as f64 + 0.5) / NY as f64;
        300.0 + 50.0 * (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.02).exp()
    });
    let mut solver = GpuDiffusion::new(&device, problem.setup()).unwrap();
    solver.run(&device, problem.dt, 150).unwrap();
    let field = solver.interior(&device).unwrap();

    let mut worst = 0.0f64;
    for j in 0..NY {
        for i in 0..NX / 2 {
            let left = field[j * NX + i];
            let right = field[j * NX + (NX - 1 - i)];
            worst = worst.max((left - right).abs());
        }
    }
    eprintln!("worst left/right asymmetry: {worst:.3e}");
    assert!(worst < 1e-3, "the pulse lost its mirror symmetry by {worst:.3e}");
}

/// §15.5's amortization, observed rather than assumed: two solvers with the same shape
/// compile one module, and a differently shaped one compiles a second.
#[test]
fn identical_solvers_share_a_compiled_module() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse);

    let first = GpuDiffusion::new(&device, problem.setup()).unwrap();
    let (_, misses_after_first) = device.kernel_cache_stats();
    let second = GpuDiffusion::new(&device, problem.setup()).unwrap();
    let (hits, misses) = device.kernel_cache_stats();

    assert_eq!(first.kernel_key(), second.kernel_key(), "same shape, same key");
    assert_eq!(misses, misses_after_first, "the second solver recompiled");
    assert!(hits >= 1, "the second solver did not hit the cache");
}

/// The claim the crate documentation rests on, held against the real toolchain.
///
/// If this ever fails, the portable backend has gained `f64` and should advertise
/// `accurate64` — so the failure is an instruction, not a defect.
#[test]
fn wgsl_f64_is_rejected_by_the_portable_backend() {
    let Some(device) = device() else { return };

    let outcome = device.probe_wgsl_f64();
    assert!(
        outcome.is_err(),
        "WGSL now accepts f64 through wgpu; lattice-wgpu should offer accurate64 and the \
         crate docs, the capability list and the tolerance rationale all need revisiting"
    );

    assert_eq!(device.capabilities().precisions, vec![Precision::Fast32]);
    assert!(!device.capabilities().supports(Precision::Accurate64));
}

/// Asking for a precision the backend does not have must fail at `open`, before any run
/// has been set up. Silent substitution here would make every downstream tolerance a lie.
#[test]
fn opening_at_the_reference_precision_is_refused() {
    if device().is_none() {
        return;
    }
    match GpuDevice::open(Precision::Accurate64) {
        Err(DeviceError::UnsupportedPrecision { requested, .. }) => {
            assert_eq!(requested, Precision::Accurate64);
        }
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("the portable backend must not claim accurate64"),
    }
}

/// Geometry that does not describe a runnable problem is refused with the reason, rather
/// than producing a plausible field from a mis-sized buffer.
#[test]
fn mismatched_buffers_are_refused_with_the_reason() {
    let Some(device) = device() else { return };
    let problem = Problem::new(pulse);

    let mut setup = problem.setup();
    let short = vec![DIFFUSIVITY; 4];
    setup.face_x = &short;
    let error = GpuDiffusion::new(&device, setup).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("face_x"), "{message}");
    assert!(message.contains(&((NX + 1) * NY).to_string()), "{message}");

    let mut setup = problem.setup();
    setup.halo = 0;
    setup.stride = NX;
    assert!(GpuDiffusion::new(&device, setup).is_err(), "a stencil needs a ghost cell");
}

fn extremes(values: &[f64]) -> (f64, f64) {
    values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}
