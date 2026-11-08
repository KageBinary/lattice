//! `lattice check` and `lattice run` — the commands that take a `.lattice` file.
//!
//! Spec §20.4's first acceptance criterion is *"One command validates and runs a
//! project"*, and M1's exit condition is that *"the same project executes headless and
//! interactively"*. These two commands are that, minus a window.
//!
//! # Why `check` prints a report and not just "ok"
//!
//! Spec §8.4 step 9 asks the compiler to emit *"the immutable CompiledModel plus a
//! model report describing equations and assumptions"*. A user who runs `check`
//! should learn which solver was selected, what it assumes, what it does not
//! conserve, how much memory it will take, and what the compiler had to approximate —
//! before spending any time running it. A bare "ok" would answer none of that.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use lattice_compiler::compile_source;
use lattice_expr::Op;
use lattice_ir::{Executor, LawFault, RenderChannel};
use lattice_observe::phase;
use lattice_runtime::{RunConfig, Simulation};
use lattice_syntax::{Diagnostic, SourceFile, Span};

use crate::args::Args;
use crate::render;

/// Exit code for "ran, but a check failed".
const CHECK_FAILED: u8 = 1;

/// Read a model file into a `SourceFile`.
fn read(path: &str) -> Result<SourceFile, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    // Show just the file name in diagnostics; the full path is already on screen.
    let name = Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned());
    Ok(SourceFile::new(name, text))
}

/// `lattice check <file>` — compile and report, without running.
pub fn check(args: &Args) -> Result<ExitCode, String> {
    let path = args
        .positional
        .first()
        .ok_or("`check` needs a model file: `lattice check model.lattice`")?;
    let file = read(path)?;

    let started = Instant::now();
    let (compiled, diagnostics) = compile_source(&file);
    let elapsed = started.elapsed();

    if !diagnostics.is_empty() {
        print!("{}", diagnostics.render(&file));
    }

    if let Some(compiled) = &compiled
        && !args.has("quiet")
    {
        println!("{}", compiled.model.report());
    }

    if diagnostics.has_errors() {
        println!("`{path}` did not compile.");
        return Ok(ExitCode::from(CHECK_FAILED));
    }

    println!(
        "`{path}` compiled in {} with {} warning{}.",
        lattice_observe::format_duration(elapsed),
        diagnostics.warning_count(),
        if diagnostics.warning_count() == 1 { "" } else { "s" }
    );
    Ok(ExitCode::SUCCESS)
}

/// `lattice run <file>` — compile, execute, and report.
pub fn run(args: &Args) -> Result<ExitCode, String> {
    let path = args
        .positional
        .first()
        .ok_or("`run` needs a model file: `lattice run model.lattice`")?;
    let file = read(path)?;

    let mut profile = lattice_observe::Profile::new();
    let (compiled, diagnostics) =
        profile.time(phase::COMPILE, || compile_source(&file));

    if !diagnostics.is_empty() {
        print!("{}", diagnostics.render(&file));
    }
    if diagnostics.has_errors() {
        println!("`{path}` did not compile; nothing was run.");
        return Ok(ExitCode::from(CHECK_FAILED));
    }
    let compiled = compiled.ok_or_else(|| format!("`{path}` produced no model"))?;

    // The model's own settings are the defaults; command-line flags override them,
    // so a model can be re-run for longer without editing it.
    let mut config = RunConfig::from_model(&compiled.model);
    if let Some(duration) = args.parsed::<f64>("duration")? {
        config.duration = Some(duration);
    }
    if let Some(steps) = args.parsed::<u64>("steps")? {
        config.max_steps = Some(steps);
    }
    if let Some(dt) = args.parsed::<f64>("timestep")? {
        config.timestep = Some(dt);
    }
    if config.duration.is_none() && config.max_steps.is_none() {
        return Err(format!(
            "`{path}` sets no `duration:`, so `run` needs one: \
             pass --duration <seconds> or --steps <n>"
        ));
    }

    let executor = executor_from(args)?;
    let schedule = executor.label();
    let mut simulation = Simulation::coupled(compiled.model, compiled.domains, compiled.coupler)
        .with_executor(executor);
    let outcome = simulation.run(&config);

    if !args.has("quiet") {
        println!();
        println!("execution: {schedule}");
        for channel in simulation.render_channels() {
            print!("{}", render_channel(&channel));
        }
        print!("{}", timeline_plots(&outcome.artifact));
        for curve in outcome.artifact.curves() {
            print!("{}", render_curve(curve));
        }
        println!();
        println!("{}", outcome.stop.describe());
        if let lattice_runtime::StopReason::NonFinite { step, .. } = outcome.stop
            && let Some(fault) = explain_halt(&file, executor_from(args)?, outcome.timestep, step)
        {
            println!();
            print!("{}", fault.render(&file));
        }
        println!();
        print!("{}", coupling_report(&simulation));
        print!("{}", outcome.artifact.summary());
        print!("{}", outcome.artifact.profile.report(None));
    }

    if let Some(json) = args.value("json") {
        let mut artifact = outcome.artifact;
        artifact.profile.record(phase::COMPILE, profile.get(phase::COMPILE).unwrap_or_default().total);
        artifact.write(json).map_err(|e| format!("cannot write {json}: {e}"))?;
        println!("wrote {json}");
        return Ok(exit_code(&outcome.stop));
    }

    Ok(exit_code(&outcome.stop))
}

/// Replay a run that halted as non-finite and, if a law the model wrote is responsible,
/// say which operation of it and where (diagnostic `E0950`, spec M6.1d §4.4).
///
/// Runs are bit-reproducible, so the failure is reached again by compiling the model
/// afresh and stepping it `step` times at `timestep` with every domain watching its user
/// laws. Nothing is kept during an ordinary run to make this possible, which is why it
/// costs one extra run, and only when a run has already failed. `None` means no user law
/// is to blame: the model has none, or the first non-finite value came from elsewhere.
pub fn explain_halt(
    file: &SourceFile,
    executor: Executor,
    timestep: f64,
    step: u64,
) -> Option<Diagnostic> {
    let (compiled, diagnostics) = compile_source(file);
    if diagnostics.has_errors() {
        return None;
    }
    let compiled = compiled?;
    let mut replay =
        Simulation::coupled(compiled.model, compiled.domains, compiled.coupler).with_executor(executor);
    let fault = replay.trace_non_finite(timestep, step)?;
    let clock = replay.clock();
    Some(fault_diagnostic(&fault, clock.step, clock.time))
}

/// A number as a message shows it: plain when it reads easily, else in exponent form.
fn show(value: f64) -> String {
    if value == 0.0 || !value.is_finite() || (1e-3..1e6).contains(&value.abs()) {
        format!("{value}")
    } else {
        format!("{value:e}")
    }
}

/// The name of an operation as a message says it: `div`, `ln`, `powi`.
fn op_name(op: Op) -> String {
    let debug = format!("{op:?}");
    debug.split('(').next().unwrap_or(&debug).to_lowercase()
}

/// `E0950`: the first non-finite value a user law produced, as a diagnostic whose caret
/// is under the sub-expression that produced it.
fn fault_diagnostic(fault: &LawFault, step: u64, time: f64) -> Diagnostic {
    let at = &fault.at;
    let (message, label) = match &fault.input {
        Some(input) => (
            format!("law `{}` was handed a non-finite value", fault.law),
            format!("`{input}` was already {}", show(at.value)),
        ),
        None => (
            format!("law `{}` produced a non-finite value", fault.law),
            format!("{} gave {}", op_name(at.op), show(at.value)),
        ),
    };
    let mut diagnostic = Diagnostic::error(message).with_code("E0950");
    diagnostic = match at.origin {
        Some(origin) => diagnostic.at(Span::new(origin.start, origin.end), label),
        None => diagnostic.note(label),
    };
    diagnostic = if fault.is_input() {
        diagnostic.note(
            "the law was given this value, so none of its operations produced it: look at what \
             computed the input",
        )
    } else if at.operands.is_empty() {
        diagnostic.note("it is a constant the compiler folded from this expression")
    } else {
        let operands: Vec<String> = at.operands.iter().map(|&v| show(v)).collect();
        diagnostic.note(format!(
            "operands: {} (every one finite; this is the first operation to go wrong)",
            operands.join(", ")
        ))
    };
    let when = if step == 0 {
        "in the initial state, before any step".to_string()
    } else {
        format!("at step {step} (t = {time:e} s)")
    };
    diagnostic.note(format!("evaluated for {} of `{}`, {when}", fault.site.describe(), fault.domain))
}

/// Build the executor `--threads` asks for.
///
/// Sequential is the default rather than [`Executor::automatic`], which is the opposite
/// of what a general-purpose tool usually does. The reason is §24.1: the scalar CPU path
/// is *"the executable specification for accelerated kernels"*, and a run that has not
/// asked for anything else should be running the specification. `--threads auto` is one
/// word away for anyone who wants the machine.
pub fn executor_from(args: &Args) -> Result<Executor, String> {
    match args.value("threads") {
        None => Ok(Executor::sequential()),
        Some("auto") => Ok(Executor::automatic()),
        Some(raw) => {
            let threads: usize = raw.parse().map_err(|_| {
                format!("--threads: `{raw}` is not a thread count; pass a number or `auto`")
            })?;
            if threads == 0 {
                return Err("--threads: 0 threads would run nothing; 1 is sequential".to_string());
            }
            Ok(Executor::with_threads(threads))
        }
    }
}

fn exit_code(stop: &lattice_runtime::StopReason) -> ExitCode {
    if stop.is_success() { ExitCode::SUCCESS } else { ExitCode::from(CHECK_FAILED) }
}

/// What the coupling edges moved, and whether any of them failed.
///
/// §14.3's whole argument: *"This makes it possible to debug whether a coupled result
/// is physically inconsistent because of the model, a mapping error, a timestep issue,
/// or an intentionally open system."* A coupled run's drift is a single number until
/// the transfers that were supposed to explain it are written down beside it.
///
/// Empty for an uncoupled model, which is most of them.
fn coupling_report(simulation: &Simulation) -> String {
    let ledger = simulation.ledger();
    let faults = simulation.coupling_faults();
    if ledger.is_empty() && faults.is_empty() {
        return String::new();
    }

    let mut out = String::from("coupling ledger\n");
    if ledger.is_empty() {
        out.push_str("  nothing was transferred\n");
    } else {
        // Totals per (quantity, source, target), which is what a reader wants — the
        // per-step entries are in the run artifact for anyone who needs them.
        let mut totals: BTreeMap<(String, String, String), (f64, usize)> = BTreeMap::new();
        for transfer in ledger.transfers() {
            let key = (
                transfer.quantity.to_string(),
                transfer.from.to_string(),
                transfer.to.to_string(),
            );
            let entry = totals.entry(key).or_insert((0.0, 0));
            entry.0 += transfer.amount;
            entry.1 += 1;
        }
        for ((quantity, from, to), (amount, count)) in totals {
            out.push_str(&format!(
                "  {from} -> {to:<24} {amount:>14.6e} {quantity} over {count} exchanges\n"
            ));
        }
        out.push_str(
            "  a staggered coupling always has one exchange in flight, so the last of \
             this has not landed\n",
        );
    }

    for fault in faults {
        out.push_str(&format!("  FAULT {fault}\n"));
    }
    out.push('\n');
    out
}

/// Most observation series to plot, so a scene with many domains stays readable.
const MAX_SERIES: usize = 10;

/// Plot each observed quantity over time.
///
/// A snapshot of the final state answers "where did it end up"; the timeline answers
/// "did it get there sensibly". For a particle scene the snapshot is a scatter of
/// dots and the *energy trace* is the whole story — P7's point that visualization is
/// instrumentation, not decoration.
fn timeline_plots(artifact: &lattice_observe::RunArtifact) -> String {
    let samples = artifact.timeline();
    if samples.len() < 2 {
        return String::new();
    }

    // Series in the order the domains published them, so related quantities stay
    // adjacent rather than being scattered alphabetically.
    let mut names: Vec<&str> = Vec::new();
    for sample in samples {
        for (name, _) in &sample.values {
            if !names.contains(&name.as_str()) {
                names.push(name);
            }
        }
    }

    let width = names.iter().map(|n| n.len()).max().unwrap_or(0).min(28);
    let mut out = format!("\n  observations over {} samples\n", samples.len());
    for name in names.iter().take(MAX_SERIES) {
        let series: Vec<f64> = samples
            .iter()
            .filter_map(|sample| {
                sample.values.iter().find(|(key, _)| key == name).map(|(_, value)| *value)
            })
            .collect();
        out.push_str(&format!("  {name:<width$} {}\n", render::sparkline(&series, 56)));
    }
    if names.len() > MAX_SERIES {
        out.push_str(&format!(
            "  ({} more series in the run artifact)\n",
            names.len() - MAX_SERIES
        ));
    }
    out
}

/// Draw one curve, with its axes and how it was obtained.
fn render_curve(curve: &lattice_ir::Curve) -> String {
    let mut out = format!(
        "\n  {} — {} [{}] against {} [{}]\n",
        curve.name, curve.y_label, curve.y_unit, curve.x_label, curve.x_unit
    );
    out.push_str(&render::plot(&curve.x, &curve.y, 64, 12));
    for note in &curve.notes {
        out.push_str(&format!("  {note}\n"));
    }
    if curve.has_non_finite() {
        out.push_str("  WARNING: this curve contains non-finite values\n");
    }
    out
}

/// Draw one render channel in the terminal.
fn render_channel(channel: &RenderChannel<'_>) -> String {
    let mut out = format!("  {} — {}\n", channel.name(), channel.describe());
    match channel {
        RenderChannel::Scalar { field, .. } => {
            out.push_str(&render::heatmap(field, 72, 20));
        }
        RenderChannel::Phase { .. } => {
            // A character ramp has two ends, and an angle does not: drawn on one, a phase
            // shows a hard edge at every wrap, and box-averaging cells across a wrap gives
            // an angle that is in neither. `lattice-view` draws it on a cyclic wheel.
            out.push_str("  an angle — drawn by `lattice-view` on a phase wheel, not on a two-ended ramp\n");
        }
        RenderChannel::Vector { x, y, grid, unit, .. } => {
            // The terminal shows the magnitude; the arrows are `lattice-view`'s.
            let mut magnitude = lattice_ir::ScalarField::new(grid, 0);
            let mut net = [0.0, 0.0];
            for j in 0..grid.ny() {
                for i in 0..grid.nx() {
                    let (vx, vy) = (x.get(i, j), y.get(i, j));
                    magnitude.set(i, j, vx.hypot(vy));
                    net[0] += vx * grid.cell_area();
                    net[1] += vy * grid.cell_area();
                }
            }
            out.push_str(&render::heatmap(&magnitude, 72, 20));
            out.push_str(&format!(
                "  magnitude shown; largest {:.3e} {unit}, integral over the grid ({:.3e}, {:.3e}) {unit}·m^2\n",
                magnitude.max_interior(),
                net[0],
                net[1]
            ));
        }
        RenderChannel::Particles { x, y, origin, extent, .. } => {
            out.push_str(&render::scatter(x, y, *origin, *extent, 72, 20, "particles"));
        }
        RenderChannel::Bodies { x, origin, extent, .. } => {
            // A terminal cannot fill a polygon, so the outlines are plotted as the
            // points they are made of. The silhouette is recognizable and nothing is
            // implied that the data does not have — `lattice-view` draws them properly.
            let mut outline = Vec::new();
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for index in 0..x.len() {
                channel.body_outline(index, &mut outline);
                // Walk every edge including the closing one, interpolating along it so
                // a large body does not appear as four disconnected corners.
                const STEPS: u32 = 12;
                for edge in 0..outline.len() {
                    let from = outline[edge];
                    let to = outline[(edge + 1) % outline.len()];
                    for step in 0..STEPS {
                        let t = f64::from(step) / f64::from(STEPS);
                        xs.push(from[0] + t * (to[0] - from[0]));
                        ys.push(from[1] + t * (to[1] - from[1]));
                    }
                }
            }
            out.push_str(&render::scatter(&xs, &ys, *origin, *extent, 72, 20, "outline points"));
        }
        RenderChannel::Bonds { pairs, .. } => {
            // The scatter above already shows where the particles are; a terminal
            // cannot draw a line between two of them without covering others. The
            // count is what it can show honestly.
            let through_walls = (0..pairs.len()).filter(|&i| channel.bond_endpoints(i).is_none()).count();
            out.push_str(&format!("  {} bonds, {through_walls} through a periodic wall\n", pairs.len()));
        }
        RenderChannel::Contacts { x, depth, .. } => {
            // Overlaying these on the ASCII silhouette would put two marks in one cell
            // and hide both. The numbers are what a terminal can show honestly.
            let deepest = depth.iter().copied().fold(0.0, f64::max);
            out.push_str(&format!(
                "  {} contact points, deepest overlap {deepest:.3e} m
",
                x.len()
            ));
        }
    }
    if channel.has_non_finite() {
        out.push_str("  WARNING: this channel contains non-finite values\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_model(name: &str, source: &str) -> String {
        let dir = std::env::temp_dir().join("lattice-cli-tests");
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        fs::write(&path, source).expect("write model");
        path.to_string_lossy().into_owned()
    }

    fn args(line: &str) -> Args {
        Args::parse(line.split_whitespace().map(String::from))
    }

    const SLAB: &str = r#"
project slab {
  fidelity: engineering_2d;
  duration: 5 second;

  grid bar { size: [32, 8]; extent: [1 meter, 0.25 meter]; }

  field temperature on bar = 273.15 kelvin {
    diffusivity: 1e-3 meter^2 / second;
    boundary_left:  fixed(273.15 kelvin);
    boundary_right: fixed(373.15 kelvin);
  }

  solve heat(temperature) with crank_nicolson(dt=0.05 second);
  observe temperature every 0.5 second;
}
"#;

    #[test]
    fn check_accepts_a_valid_model() {
        let path = write_model("slab_check.lattice", SLAB);
        let mut a = args("check --quiet");
        a.positional.push(path);
        assert_eq!(check(&a).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn check_rejects_a_model_with_errors() {
        let path = write_model(
            "broken_check.lattice",
            "project p { grid g { size: [4, 4]; } }",
        );
        let mut a = args("check --quiet");
        a.positional.push(path);
        // Missing `extent`, so it must not compile.
        assert_eq!(check(&a).unwrap(), ExitCode::from(CHECK_FAILED));
    }

    #[test]
    fn check_without_a_file_explains_itself() {
        let error = check(&args("check")).unwrap_err();
        assert!(error.contains("needs a model file"), "{error}");
    }

    #[test]
    fn a_missing_file_is_reported_clearly() {
        let mut a = args("check");
        a.positional.push("definitely-not-here.lattice".to_string());
        let error = check(&a).unwrap_err();
        assert!(error.contains("cannot read"), "{error}");
    }

    #[test]
    fn run_executes_a_valid_model() {
        let path = write_model("slab_run.lattice", SLAB);
        let mut a = args("run --quiet");
        a.positional.push(path);
        assert_eq!(run(&a).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn run_refuses_a_model_that_does_not_compile() {
        let path = write_model(
            "broken_run.lattice",
            "project p { grid g { size: [4, 4]; extent: [1 meter, 1 meter]; }\n\
             field t on g = 1 kelvin;\n solve heat(t) with crank_nicolson(dt=1 second); }",
        );
        let mut a = args("run --quiet");
        a.positional.push(path);
        // No diffusivity, so it cannot be solved.
        assert_eq!(run(&a).unwrap(), ExitCode::from(CHECK_FAILED));
    }

    /// A model with no duration and no override would never stop; saying so beats
    /// running forever.
    #[test]
    fn run_needs_a_duration_from_somewhere() {
        let path = write_model(
            "endless.lattice",
            r#"
project endless {
  grid g { size: [8, 8]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin { diffusivity: 1e-4 meter^2/second; }
  solve heat(t) with crank_nicolson(dt=0.1 second);
}
"#,
        );
        let mut a = args("run --quiet");
        a.positional.push(path.clone());
        let error = run(&a).unwrap_err();
        assert!(error.contains("--duration"), "{error}");

        // ...and the override supplies it.
        let mut a = args("run --quiet --steps 5");
        a.positional.push(path);
        assert_eq!(run(&a).unwrap(), ExitCode::SUCCESS);
    }

    /// Compile and run `source` until it halts, and explain the halt: the step it halted
    /// at, and the diagnostic the replay produced.
    fn halted(source: &str) -> (SourceFile, u64, Option<Diagnostic>) {
        let file = SourceFile::new("fixture.lattice", source);
        let (compiled, diagnostics) = compile_source(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        let compiled = compiled.expect("a model");
        let config = RunConfig::from_model(&compiled.model);
        let mut simulation = Simulation::coupled(compiled.model, compiled.domains, compiled.coupler);
        let outcome = simulation.run(&config);
        let lattice_runtime::StopReason::NonFinite { step, .. } = outcome.stop else {
            panic!("expected a halt, got {:?}", outcome.stop)
        };
        let diagnostic = explain_halt(&file, Executor::sequential(), outcome.timestep, step);
        (file, step, diagnostic)
    }

    /// The source text under the diagnostic's caret.
    fn underlined<'a>(file: &'a SourceFile, diagnostic: &Diagnostic) -> &'a str {
        let span = diagnostic.primary_span();
        &file.text()[span.start as usize..span.end as usize]
    }

    fn note_with<'a>(diagnostic: &'a Diagnostic, needle: &str) -> &'a str {
        diagnostic
            .notes
            .iter()
            .find(|note| note.contains(needle))
            .unwrap_or_else(|| panic!("no note with `{needle}` in {:?}", diagnostic.notes))
    }

    /// Spec M6.1d §7, the pair potential: `1 / (r − 1)` with the atoms at distance 1.
    #[test]
    fn a_pair_potential_dividing_by_zero_is_explained() {
        let (file, _, diagnostic) = halted(
            "project pole {
  dimensions: 2;
  duration: 1 second;
  potential pole(a: particle, b: particle) -> joule {
      return 1 joule * meter / (distance(a, b) - 1 meter);
  }
  particles pair {
    count: 2;
    region: [4 meter, 4 meter];
    spacing: 1 meter;
    mass: 1 kilogram;
    force: pole(cutoff=2 meter);
  }
  solve dynamics(pair) with velocity_verlet(dt=0.01 second);
  observe pair every 0.05 second;
}",
        );
        let diagnostic = diagnostic.expect("a user law is to blame");
        assert_eq!(diagnostic.code.as_deref(), Some("E0950"));
        assert_eq!(diagnostic.message, "law `pole` produced a non-finite value");
        // The caret is under the whole quotient, the operation that divided by zero.
        let start = file.text().find("1 joule * meter /").unwrap() as u32;
        let expression = "1 joule * meter / (distance(a, b) - 1 meter)";
        assert_eq!(diagnostic.primary_span(), Span::new(start, start + expression.len() as u32));
        assert_eq!(underlined(&file, &diagnostic), expression);
        assert_eq!(diagnostic.primary.as_ref().unwrap().message, "div gave inf");
        assert!(note_with(&diagnostic, "operands").contains("operands: 1, 0 "), "{:?}", diagnostic.notes);
        assert!(note_with(&diagnostic, "pair (0, 1) of `pair`").contains("initial state"));
        let rendered = diagnostic.render(&file);
        assert!(rendered.starts_with("error[E0950]"), "{rendered}");
    }

    /// The body force: `ln(px)` with the bead crossing to a negative position, found at a
    /// step before the one the run halted at, since observations are sampled sparsely.
    #[test]
    fn a_body_force_taking_the_log_of_a_negative_position_is_explained() {
        let (file, halt_step, diagnostic) = halted(
            "project lnx {
  dimensions: 2;
  duration: 5 second;
  force pull(a: particle) -> vec2<newton> {
      return vec2(1 newton * ln(a.position.x / 1 meter), 0 newton);
  }
  particles bead {
    count: 1;
    region: [4 meter, 4 meter];
    spacing: 1 meter;
    mass: 1 kilogram;
    force: pull();
  }
  solve dynamics(bead) with velocity_verlet(dt=0.01 second);
  observe bead every 0.05 second;
}",
        );
        let diagnostic = diagnostic.expect("a user law is to blame");
        assert_eq!(diagnostic.code.as_deref(), Some("E0950"));
        assert_eq!(underlined(&file, &diagnostic), "ln(a.position.x / 1 meter)");
        assert_eq!(diagnostic.primary.as_ref().unwrap().message, "ln gave NaN");
        let operands = note_with(&diagnostic, "operands:");
        let value: f64 = operands
            .trim_start_matches("operands: ")
            .split(' ')
            .next()
            .unwrap()
            .parse()
            .expect("one operand");
        assert!(value < 0.0 && value.is_finite(), "{operands}");
        let when = note_with(&diagnostic, "particle 0 of `bead`");
        let step: u64 = when.split("at step ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!(step <= halt_step, "the fault is at or before the halt: {step} against {halt_step}");
        assert!(step % 5 != 0 || step == halt_step, "found between samples, not only at one: {step}");
    }

    /// The rate law: `k2 / c(A)` once `A` has run out.
    #[test]
    fn a_rate_law_dividing_by_an_exhausted_reactant_is_explained() {
        let (file, _, diagnostic) = halted(
            "project runout {
  dimensions: 2;
  fidelity: engineering_2d;
  duration: 5 second;
  let vmax = 5 mole / (meter^2 second);
  let km = 0.1 mole / meter^2;
  let k2 = 0.001 mole^2 / (meter^4 second);
  grid dish { size: [2, 2]; extent: [1 meter, 1 meter]; }
  domain chemistry mix { grid: dish; temperature: 298.15 kelvin; }
  species A on dish = 1 mole / meter^2 { formula: C; diffusivity: 0 meter^2 / second; boundary: insulated; }
  species B on dish = 0 mole / meter^2 { formula: C; diffusivity: 0 meter^2 / second; boundary: insulated; }
  species C on dish = 0 mole / meter^2 { formula: C; diffusivity: 0 meter^2 / second; boundary: insulated; }
  reaction consume {
    stoichiometry: A -> B;
    rate: vmax * c(A) / (km + c(A));
  }
  reaction leak {
    stoichiometry: B -> C;
    rate: k2 / c(A);
  }
  solve reactions(mix) with strang(dt = 0.1 second);
  observe mix every 0.5 second;
}",
        );
        let diagnostic = diagnostic.expect("a rate law is to blame");
        assert_eq!(diagnostic.code.as_deref(), Some("E0950"));
        assert_eq!(diagnostic.message, "law `leak` produced a non-finite value");
        assert_eq!(underlined(&file, &diagnostic), "k2 / c(A)");
        assert_eq!(diagnostic.primary.as_ref().unwrap().message, "div gave inf");
        assert!(note_with(&diagnostic, "operands").contains("operands: 0.001, 0 "), "{:?}", diagnostic.notes);
        assert!(note_with(&diagnostic, "cell (").contains("of `mix`"));
    }

    /// A non-finite value that is already in the state a law reads is the state's fault,
    /// not the law's: the diagnostic blames the input and no operation. Here a built-in
    /// Lennard-Jones force between atoms 1e-60 m apart overflows, positions go NaN, and the
    /// tether law, perfectly well-behaved, is handed them.
    #[test]
    fn a_law_handed_a_nan_blames_the_input_and_no_operation() {
        let (file, _, diagnostic) = halted(
            "project inp {
  dimensions: 2;
  duration: 1 second;
  force tether(a: particle) -> vec2<newton> {
      param k: newton / meter;
      return vec2(-1 * k * a.position.x, 0 newton);
  }
  particles pair {
    count: 2;
    region: [4 meter, 4 meter];
    spacing: 1e-60 meter;
    mass: 1 kilogram;
    force: lennard_jones(epsilon=1 joule, sigma=1 meter, cutoff=2.5 meter);
    force: tether(k=1 newton / meter);
  }
  solve dynamics(pair) with velocity_verlet(dt=0.01 second);
  observe pair every 0.01 second;
}",
        );
        let diagnostic = diagnostic.expect("the tether was handed a NaN");
        assert_eq!(diagnostic.code.as_deref(), Some("E0950"));
        assert_eq!(diagnostic.message, "law `tether` was handed a non-finite value");
        // The input was pushed where the member `a.position` was read; `.x` selects from it.
        assert_eq!(underlined(&file, &diagnostic), "a.position");
        assert_eq!(diagnostic.primary.as_ref().unwrap().message, "`a.position.x` was already NaN");
        assert!(note_with(&diagnostic, "none of its operations produced it").contains("input"));
        assert!(
            diagnostic.notes.iter().all(|note| !note.contains("operands")),
            "no operation is blamed: {:?}",
            diagnostic.notes
        );
    }

    /// A model with no user law has nothing to explain, and is not replayed.
    #[test]
    fn a_halt_with_no_user_law_has_no_explanation() {
        let (_, _, diagnostic) = halted(
            "project blowup {
  dimensions: 2;
  duration: 1 second;
  particles pair {
    count: 2;
    region: [4 meter, 4 meter];
    spacing: 1e-60 meter;
    mass: 1 kilogram;
    force: lennard_jones(epsilon=1 joule, sigma=1 meter, cutoff=2.5 meter);
  }
  solve dynamics(pair) with velocity_verlet(dt=0.01 second);
  observe pair every 0.01 second;
}",
        );
        assert!(diagnostic.is_none());
    }

    #[test]
    fn run_writes_an_artifact_when_asked() {
        let path = write_model("slab_artifact.lattice", SLAB);
        let out = std::env::temp_dir().join("lattice-cli-tests").join("run.json");
        let _ = fs::remove_file(&out);

        let mut a = args(&format!("run --quiet --json {}", out.to_string_lossy()));
        a.positional.clear();
        a.positional.push(path);
        assert_eq!(run(&a).unwrap(), ExitCode::SUCCESS);

        let text = fs::read_to_string(&out).expect("artifact should exist");
        assert!(text.contains("\"run\": \"slab\""), "{text}");
        assert!(text.contains("timeline"), "{text}");
        assert!(text.contains("content_hash"), "{text}");
        let _ = fs::remove_file(&out);
    }

    /// The command-line duration overrides the model's, so a scene can be re-run for
    /// longer without editing it.
    #[test]
    fn command_line_overrides_beat_the_model() {
        let path = write_model("slab_override.lattice", SLAB);
        let mut a = args("run --quiet --steps 3");
        a.positional.push(path);
        assert_eq!(run(&a).unwrap(), ExitCode::SUCCESS);
    }
}
