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
use lattice_ir::{Executor, RenderChannel};
use lattice_observe::phase;
use lattice_runtime::{RunConfig, Simulation};
use lattice_syntax::SourceFile;

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
        println!();
        println!("{}", outcome.stop.describe());
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

/// Draw one render channel in the terminal.
fn render_channel(channel: &RenderChannel<'_>) -> String {
    let mut out = format!("  {} — {}\n", channel.name(), channel.describe());
    match channel {
        RenderChannel::Scalar { field, .. } => {
            out.push_str(&render::heatmap(field, 72, 20));
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
