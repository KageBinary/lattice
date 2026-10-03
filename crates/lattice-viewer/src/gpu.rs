//! Resident field viewer. The compute backend has no dependency on egui; this
//! adapter shares egui's device and registers a persistent native texture.
use crate::palette::{self, Colormap, Mode};
use eframe::egui;
use lattice_domain_grid2d::{HeatDomain, TimeScheme};
use lattice_ir::{Boundary, Domain};
use lattice_wgpu::{
    CrankNicolsonSetup, DiffusionSetup, GpuCrankNicolson, GpuDevice, GpuDiffusion, GpuFieldImage,
    Interior, wgpu,
};
use std::path::Path;

enum Solver {
    Explicit(GpuDiffusion),
    Implicit(Box<GpuCrankNicolson>),
}
impl Solver {
    fn interior(&self, device: &GpuDevice) -> Result<Vec<f64>, String> {
        match self {
            Self::Explicit(s) => s.interior(device).map_err(|e| e.to_string()),
            Self::Implicit(s) => s.interior(device).map_err(|e| e.to_string()),
        }
    }
}

pub fn run(path: impl AsRef<Path>, play: bool) -> Result<(), String> {
    run_with_smoke(path, play, None)
}

/// Exercise real device sharing, compute, texture registration and presentation,
/// then close the window. Used by the release smoke check.
pub fn run_with_smoke(
    path: impl AsRef<Path>,
    play: bool,
    frames: Option<u64>,
) -> Result<(), String> {
    let path = path.as_ref().to_path_buf();
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Lattice — GPU field",
        options,
        Box::new(move |cc| {
            GpuViewer::new(cc, &path, play)
                .map(|mut app| {
                    app.smoke_frames = frames;
                    Box::new(app) as Box<dyn eframe::App>
                })
                .map_err(|message| {
                    Box::new(std::io::Error::other(message))
                        as Box<dyn std::error::Error + Send + Sync>
                })
        }),
    )
    .map_err(|e| e.to_string())
}

struct GpuViewer {
    device: GpuDevice,
    solver: Solver,
    image: GpuFieldImage,
    texture: egui::TextureId,
    renderer: eframe::egui_wgpu::RenderState,
    name: String,
    unit: String,
    report: String,
    range: [f64; 2],
    extent: [f64; 2],
    dt: f64,
    steps: u64,
    playing: bool,
    speed: usize,
    error: Option<String>,
    sample: String,
    frames: u64,
    initial_integral: f64,
    cell_area: f64,
    closed: bool,
    smoke_frames: Option<u64>,
}
impl GpuViewer {
    fn new(cc: &eframe::CreationContext<'_>, path: &Path, playing: bool) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let file = lattice_syntax::SourceFile::new(path.display().to_string(), text);
        let (compiled, diagnostics) = lattice_compiler::compile_source(&file);
        if diagnostics.has_errors() {
            return Err(diagnostics.render(&file));
        }
        let compiled = compiled.ok_or("no compiled model")?;
        if compiled.domains.len() != 1 || !compiled.coupler.edges().is_empty() {
            return Err(
                "GPU field viewer currently requires one uncoupled heat/diffusion domain".into(),
            );
        }
        let domain = compiled.domains[0]
            .as_ref()
            .as_any()
            .downcast_ref::<HeatDomain>()
            .ok_or("GPU field viewer requires a heat/diffusion domain")?;
        let renderer = cc
            .wgpu_render_state
            .clone()
            .ok_or("viewer requires the wgpu renderer")?;
        let device = GpuDevice::from_shared(
            renderer.device.clone(),
            renderer.queue.clone(),
            renderer.adapter.get_info(),
        );
        let dt = compiled
            .model
            .timestep
            .unwrap_or(domain.stable_step().preferred);
        if !dt.is_finite() || dt <= 0.0 || !domain.stable_step().admits(dt) {
            return Err("invalid or unstable timestep".into());
        }
        let boundaries = domain.boundaries();
        let mut values = [None; 4];
        for (i, b) in [
            boundaries.left,
            boundaries.right,
            boundaries.bottom,
            boundaries.top,
        ]
        .into_iter()
        .enumerate()
        {
            values[i] = match b {
                Boundary::Dirichlet { value } => Some(value),
                Boundary::Neumann { gradient: 0.0 } => None,
                _ => {
                    return Err(
                        "GPU field viewer supports insulated and implicit Dirichlet faces".into(),
                    );
                }
            };
        }
        let grid = domain.grid();
        let field = domain.field();
        let op = domain.operator();
        let layout = Interior {
            nx: grid.nx(),
            ny: grid.ny(),
            stride: field.stride(),
            origin: field.halo() * field.stride() + field.halo(),
        };
        let source = domain.source().map(|s| s.as_slice());
        let mut report = compiled.model.report();
        let solver = if domain.scheme() == TimeScheme::Explicit {
            if values.iter().any(Option::is_some) {
                return Err("explicit GPU fields currently require insulated faces".into());
            }
            Solver::Explicit(
                GpuDiffusion::new(
                    &device,
                    DiffusionSetup {
                        nx: layout.nx,
                        ny: layout.ny,
                        stride: layout.stride,
                        halo: field.halo(),
                        inv_dx2: op.inv_dx2(),
                        inv_dy2: op.inv_dy2(),
                        field: field.as_slice(),
                        face_x: op.face_x(),
                        face_y: op.face_y(),
                        source,
                    },
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            let mut setup = CrankNicolsonSetup {
                nx: layout.nx,
                ny: layout.ny,
                stride: layout.stride,
                halo: field.halo(),
                inv_dx2: op.inv_dx2(),
                inv_dy2: op.inv_dy2(),
                theta: domain.scheme().theta(),
                dt,
                field: field.as_slice(),
                face_x: op.face_x(),
                face_y: op.face_y(),
                source,
                tolerance: 1.0,
                max_iterations: 500,
            };
            setup.tolerance =
                10.0 * GpuCrankNicolson::floor_for(lattice_wgpu::SOLVER_PRECISION, &setup);
            report.push_str(&format!("\nGPU fast32 relative residual tolerance: {:.6e} (10 × computed precision floor)\n",setup.tolerance));
            Solver::Implicit(Box::new(
                GpuCrankNicolson::with_dirichlet(&device, setup, values)
                    .map_err(|e| e.to_string())?,
            ))
        };
        let initial: Vec<_> = (0..grid.ny())
            .flat_map(|j| field.row(j).iter().copied())
            .collect();
        let low = initial
            .iter()
            .chain(values.iter().flatten())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let high = initial
            .iter()
            .chain(values.iter().flatten())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = [low, high];
        let colors = std::array::from_fn(|i| {
            palette::sample(Colormap::Sequential, Mode::Dark, i as f64 / 255.0)
                .to_array()
                .map(|c| f32::from(c) / 255.0)
        });
        let buffers = match &solver {
            Solver::Explicit(s) => s.buffers().to_vec(),
            Solver::Implicit(s) => vec![s.buffer()],
        };
        let image = GpuFieldImage::new(&device, &buffers, layout, &colors, range)
            .map_err(|e| e.to_string())?;
        let texture = renderer.renderer.write().register_native_texture(
            &renderer.device,
            image.view(),
            wgpu::FilterMode::Nearest,
        );
        let mut app = Self {
            device,
            solver,
            image,
            texture,
            renderer,
            name: domain.name().into(),
            unit: domain.display_unit().into(),
            report,
            range,
            extent: grid.extent(),
            dt,
            steps: 0,
            playing,
            speed: 4,
            error: None,
            sample: String::new(),
            frames: 0,
            initial_integral: initial.iter().sum::<f64>() * grid.cell_area(),
            cell_area: grid.cell_area(),
            closed: values.iter().all(Option::is_none) && source.is_none(),
            smoke_frames: None,
        };
        app.advance(0)?;
        app.sample()?;
        Ok(app)
    }
    fn advance(&mut self, steps: usize) -> Result<(), String> {
        let mut encoder = self
            .device
            .raw_device()
            .create_command_encoder(&Default::default());
        let parity = match &mut self.solver {
            Solver::Explicit(s) => {
                s.encode(&self.device, &mut encoder, self.dt, steps);
                self.steps += steps as u64;
                s.parity()
            }
            Solver::Implicit(s) => {
                for _ in 0..steps {
                    let outcome = s.step(&self.device).map_err(|e| e.to_string())?;
                    self.steps += 1;
                    if !outcome.is_converged() {
                        return Err(outcome.to_string());
                    }
                }
                0
            }
        };
        self.image.encode(&mut encoder, parity);
        self.device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
    fn sample(&mut self) -> Result<(), String> {
        let values = self.solver.interior(&self.device)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("non-finite field; simulation halted".into());
        }
        let integral = values.iter().sum::<f64>() * self.cell_area;
        self.sample = format!(
            "Sample at t={:.6} s: integral={:.8e} {}·m²",
            self.steps as f64 * self.dt,
            integral,
            self.unit
        );
        if self.closed {
            self.sample.push_str(&format!(
                "; change={:.3e}",
                integral - self.initial_integral
            ));
        }
        Ok(())
    }
}
impl eframe::App for GpuViewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(frames) = self.smoke_frames {
            assert!(
                self.error.is_none(),
                "GPU viewer smoke failure: {:?}",
                self.error
            );
            if self.frames >= frames {
                self.sample().expect("GPU smoke diagnostics");
                eprintln!(
                    "GPU VIEWER SMOKE PASSED: {} frames, {} steps; {}",
                    self.frames, self.steps, self.sample
                );
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
        }
        ui.horizontal(|ui| {
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Step").clicked() {
                self.playing = false;
                if let Err(e) = self.advance(1) {
                    self.error = Some(e);
                }
            }
            ui.add(egui::Slider::new(&mut self.speed, 1..=100).text("steps/frame"));
            if ui.button("Sample diagnostics").clicked()
                && let Err(e) = self.sample()
            {
                self.error = Some(e);
            }
        });
        if self.playing && self.error.is_none() {
            if let Err(e) = self.advance(self.speed) {
                self.error = Some(e);
                self.playing = false;
            }
            self.frames += 1;
            // Data readback is for an explicitly timestamped diagnostic snapshot only.
            // Drawing itself never downloads or uploads the field.
            if self.frames % 60 == 0
                && let Err(e) = self.sample()
            {
                self.error = Some(e);
                self.playing = false;
            }
            ui.ctx().request_repaint();
        }
        ui.label(format!(
            "{} — GPU fast32 — t={:.6} s, dt={:.6} s",
            self.name,
            self.steps as f64 * self.dt,
            self.dt
        ));
        ui.label(format!(
            "Fixed color range: {:.6} to {:.6} {}",
            self.range[0], self.range[1], self.unit
        ));
        ui.label(&self.sample);
        if let Solver::Implicit(s) = &self.solver {
            ui.label(format!(
                "Last solve: {:?}; relative tolerance {:.3e}",
                s.last_outcome(),
                s.tolerance()
            ));
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::RED, e);
        }
        ui.collapsing("Model and solver contract", |ui| {
            ui.monospace(&self.report);
        });
        let size = ui.available_size();
        let aspect = (self.extent[0] / self.extent[1]) as f32;
        let height = size.y.min(size.x / aspect).max(1.0);
        ui.image((self.texture, egui::vec2(height * aspect, height)));
    }
}
impl Drop for GpuViewer {
    fn drop(&mut self) {
        self.renderer.renderer.write().free_texture(&self.texture);
    }
}
