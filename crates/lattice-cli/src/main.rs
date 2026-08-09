//! `lattice` — the command-line runner.
//!
//! Spec FR-012 requires the same model to be reachable from Rust, Python and a
//! command line; this is the third of those, and at M0 it is the one that exists.
//! Spec §20.4 sets the acceptance bar it has to clear: *"One command validates and
//! runs a project"*, and *"Benchmark harness reports compute, render, I/O, and memory
//! separately."*
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! |---|---|
//! | 0 | everything requested succeeded |
//! | 1 | it ran, and something failed a check — a validation case, a benchmark's correctness condition |
//! | 2 | the command line was wrong and nothing ran |
//!
//! Separating 1 from 2 matters for CI: a failed validation case is a result worth
//! recording, while a mistyped flag is not.

mod args;
mod bench;
#[cfg(feature = "gpu")]
mod bench_gpu;
mod demo;
mod inspect;
mod project;
mod render;

use std::process::ExitCode;

use args::Args;
use lattice_observe::RunArtifact;
use lattice_validation::ValidationReport;

/// Exit code for "ran, but a check failed".
const CHECK_FAILED: u8 = 1;
/// Exit code for "the command line was wrong".
const USAGE_ERROR: u8 = 2;

fn main() -> ExitCode {
    let args = Args::from_env();
    match run(&args) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {message}\n");
            eprintln!("{}", usage());
            ExitCode::from(USAGE_ERROR)
        }
    }
}

fn run(args: &Args) -> Result<ExitCode, String> {
    if args.has("version") {
        println!("lattice {}", env!("CARGO_PKG_VERSION"));
        println!("{}", bench::environment());
        return Ok(ExitCode::SUCCESS);
    }
    if args.has("help") || args.command.is_none() {
        println!("{}", usage());
        return Ok(ExitCode::SUCCESS);
    }

    match args.command.as_deref().expect("checked above") {
        "check" => {
            check_flags(args, &["quiet"])?;
            project::check(args)
        }
        "run" => {
            check_flags(args, &["duration", "steps", "timestep", "threads", "json", "quiet"])?;
            project::run(args)
        }
        "validate" => cmd_validate(args),
        "bench" => cmd_bench(args),
        "demo" => cmd_demo(args),
        "inspect" => cmd_inspect(args),
        "help" => {
            println!("{}", usage());
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!(
            "unknown command `{other}`; expected check, run, validate, bench, demo, or inspect"
        )),
    }
}

/// Reject unrecognized flags rather than ignoring them.
///
/// A silently ignored `--jsno out.json` produces a run with no artifact and no
/// explanation, which is a worse outcome than refusing to start.
fn check_flags(args: &Args, known: &[&str]) -> Result<(), String> {
    let mut allowed = known.to_vec();
    allowed.extend_from_slice(&["help", "version"]);
    let unknown = args.unknown_flags(&allowed);
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unknown flag{} {}; this command accepts {}",
            if unknown.len() > 1 { "s" } else { "" },
            unknown.iter().map(|f| format!("--{f}")).collect::<Vec<_>>().join(", "),
            known.iter().map(|f| format!("--{f}")).collect::<Vec<_>>().join(", ")
        ))
    }
}

fn cmd_validate(args: &Args) -> Result<ExitCode, String> {
    check_flags(args, &["filter", "json", "quiet"])?;

    let report = match args.value("filter") {
        Some(pattern) => ValidationReport::run_matching(pattern),
        None => ValidationReport::run_all(),
    };

    if report.total() == 0 {
        return Err(format!(
            "no validation case matches `{}`; try `lattice inspect cases`",
            args.value("filter").unwrap_or("")
        ));
    }

    if !args.has("quiet") {
        print!("{}", report.text());
    }

    if let Some(path) = args.value("json") {
        let mut artifact = RunArtifact::new("validation");
        artifact.set_parameter("filter", args.value("filter").unwrap_or("(all)"));
        artifact.set_section("validation", report.to_json());
        for contract in inspect::all_contracts() {
            artifact.add_contract(contract);
        }
        artifact.write(path).map_err(|e| format!("cannot write {path}: {e}"))?;
        println!("wrote {path}");
    }

    Ok(if report.all_passed() { ExitCode::SUCCESS } else { ExitCode::from(CHECK_FAILED) })
}

fn cmd_bench(args: &Args) -> Result<ExitCode, String> {
    check_flags(args, &["scale", "threads", "backend", "compare", "json", "quiet"])?;

    let scale: usize = args.parsed_or("scale", 1)?;
    if scale == 0 {
        return Err("--scale must be at least 1".to_string());
    }

    match args.value("backend").unwrap_or("cpu") {
        "cpu" => cmd_bench_cpu(args, scale),
        "gpu" => cmd_bench_gpu(args, scale),
        other => Err(format!("--backend: `{other}` is not a backend; pass `cpu` or `gpu`")),
    }
}

/// `lattice bench --backend gpu`.
///
/// Behind a feature flag, and absent rather than silently falling back to the CPU when it
/// is not compiled in: a GPU number produced by the CPU would be the most misleading thing
/// this harness could print.
#[cfg(not(feature = "gpu"))]
fn cmd_bench_gpu(_args: &Args, _scale: usize) -> Result<ExitCode, String> {
    Err("this binary was built without the GPU backend; rebuild with `--features gpu`"
        .to_string())
}

#[cfg(feature = "gpu")]
fn cmd_bench_gpu(args: &Args, scale: usize) -> Result<ExitCode, String> {
    if args.value("threads").is_some() {
        return Err("--threads applies to the CPU backend; the GPU chooses its own schedule"
            .to_string());
    }

    // Timed and printed because §15.1 says measure end-to-end, and this is the largest
    // single cost in a short GPU run — enumerating adapters, creating a device, and
    // forcing the driver's lazy initialization. It is paid once per process rather than
    // once per step, which is exactly why it must be shown separately instead of folded
    // into a throughput figure that would then depend on how long the run happened to be.
    let open_start = std::time::Instant::now();
    let device = lattice_wgpu::GpuDevice::open_default()
        .map_err(|error| format!("could not open a GPU device: {error}"))?;
    let device_open = open_start.elapsed();
    let executed = bench_gpu::executed(&device);

    let selected = match args.positional.first() {
        Some(pattern) => bench_gpu::matching(pattern),
        None => bench_gpu::all().to_vec(),
    };
    if selected.is_empty() {
        let available: Vec<_> = bench_gpu::all().iter().map(|b| b.name).collect();
        return Err(format!(
            "no GPU benchmark matches `{}`; the GPU backend implements: {}",
            args.positional.first().map(String::as_str).unwrap_or(""),
            available.join(", ")
        ));
    }

    println!("{}", bench::environment_for(&executed));
    println!(
        "device open {device_open:.3?} (once per process, not included in the phase tables below)"
    );
    if let Some(warning) = bench::build_warning() {
        println!("\nWARNING: {warning}");
    }

    let compare = args.has("compare");
    let sequential = lattice_ir::Executor::sequential();
    let cpu_executed = bench::Executed::cpu(&sequential);

    let mut artifact = RunArtifact::new("benchmark");
    artifact.set_parameter("scale", scale);
    artifact.set_parameter("backend", "gpu");
    let mut results = lattice_observe::Json::array();
    let mut all_valid = true;
    let mut valid_count = 0usize;

    for benchmark in &selected {
        let info = benchmark.info();
        // The CPU baseline runs first for the same reason it does on the CPU path: the
        // measured run gets the warm cache, which biases the ratio down.
        let baseline = compare
            .then(|| bench::matching(info.name).first().map(|cpu| (cpu.run)(scale, &sequential)))
            .flatten();
        let outcome = (benchmark.run)(scale, &device)
            .map_err(|error| format!("{} failed on the GPU: {error}", info.name))?;

        all_valid &= outcome.valid();
        if let Some(baseline) = &baseline {
            all_valid &= baseline.valid();
        }

        if !args.has("quiet") {
            if let Some(baseline) = &baseline {
                print!("{}", bench::report(info, scale, &cpu_executed, baseline));
            }
            print!("{}", bench::report(info, scale, &executed, &outcome));
            if let Some(baseline) = &baseline {
                print!(
                    "{}",
                    bench::speedup_report((&cpu_executed, baseline), (&executed, &outcome))
                );
            }
        }

        valid_count += usize::from(outcome.valid());
        let mut json = bench::to_json(info, scale, &executed, &outcome);
        if let Some(baseline) = &baseline {
            json = json.set(
                "cpu_baseline",
                bench::to_json(info, scale, &cpu_executed, baseline),
            );
        }
        results.push(json);
    }
    artifact.set_section("benchmarks", results);

    finish_bench(args, artifact, all_valid, valid_count, selected.len())
}

fn cmd_bench_cpu(args: &Args, scale: usize) -> Result<ExitCode, String> {
    let executor = project::executor_from(args)?;
    let compare = args.has("compare");
    if compare && executor.is_sequential() {
        return Err(
            "--compare needs something to compare against: pass --threads <n> or --threads auto"
                .to_string(),
        );
    }

    let selected = match args.positional.first() {
        Some(pattern) => bench::matching(pattern),
        None => bench::all().to_vec(),
    };
    if selected.is_empty() {
        return Err(format!(
            "no benchmark matches `{}`; try `lattice inspect benchmarks`",
            args.positional.first().map(String::as_str).unwrap_or("")
        ));
    }

    let executed = bench::Executed::cpu(&executor);
    println!("{}", bench::environment_for(&executed));
    if let Some(warning) = bench::build_warning() {
        println!("\nWARNING: {warning}");
    }

    let mut artifact = RunArtifact::new("benchmark");
    artifact.set_parameter("scale", scale);
    artifact.set_parameter("backend", "cpu");
    let mut results = lattice_observe::Json::array();
    let mut all_valid = true;
    let mut valid_count = 0usize;

    let sequential = lattice_ir::Executor::sequential();
    let sequential_executed = bench::Executed::cpu(&sequential);
    for benchmark in &selected {
        let info = benchmark.info();
        // The baseline runs first when comparing, so the parallel run is the one with a
        // warm cache. That biases the speedup *down*, which is the direction a
        // performance claim should be biased.
        let baseline = compare.then(|| (benchmark.run)(scale, &sequential));
        let outcome = (benchmark.run)(scale, &executor);
        all_valid &= outcome.valid();
        if let Some(baseline) = &baseline {
            all_valid &= baseline.valid();
        }

        if !args.has("quiet") {
            if let Some(baseline) = &baseline {
                print!("{}", bench::report(info, scale, &sequential_executed, baseline));
            }
            print!("{}", bench::report(info, scale, &executed, &outcome));
            if let Some(baseline) = &baseline {
                print!(
                    "{}",
                    bench::speedup_report(
                        (&sequential_executed, baseline),
                        (&executed, &outcome)
                    )
                );
            }
        }

        valid_count += usize::from(outcome.valid());
        let mut json = bench::to_json(info, scale, &executed, &outcome);
        if let Some(baseline) = &baseline {
            json = json.set(
                "sequential_baseline",
                bench::to_json(info, scale, &sequential_executed, baseline),
            );
        }
        results.push(json);
    }
    artifact.set_section("benchmarks", results);

    finish_bench(args, artifact, all_valid, valid_count, selected.len())
}

/// The tail every `bench` path shares: the count, the artifact, and the exit code.
fn finish_bench(
    args: &Args,
    artifact: RunArtifact,
    all_valid: bool,
    valid: usize,
    total: usize,
) -> Result<ExitCode, String> {
    println!("\n{valid} of {total} benchmarks met their correctness conditions");

    if let Some(path) = args.value("json") {
        artifact.write(path).map_err(|e| format!("cannot write {path}: {e}"))?;
        println!("wrote {path}");
    }

    Ok(if all_valid { ExitCode::SUCCESS } else { ExitCode::from(CHECK_FAILED) })
}

fn cmd_demo(args: &Args) -> Result<ExitCode, String> {
    check_flags(args, &["steps", "samples", "json", "quiet"])?;

    let name = args
        .positional
        .first()
        .ok_or("`demo` needs a scene name; run `lattice inspect demos` to see them")?;
    let scene = demo::find(name)
        .ok_or_else(|| format!("no demo named `{name}`; try `lattice inspect demos`"))?;

    let options = demo::Options {
        steps: args.parsed_or("steps", 400usize)?,
        samples: args.parsed_or("samples", 40usize)?,
    };
    if options.steps == 0 {
        return Err("--steps must be at least 1".to_string());
    }

    let result = (scene.run)(&options);

    if !args.has("quiet") {
        println!("\n{} — {}\n", scene.name, scene.description);
        print!("{}", result.visual);
        println!("\n{}", result.artifact.summary());
        print!("{}", result.artifact.profile.report(None));
        println!();
        print!("{}", result.artifact.memory.report());
    }

    if let Some(path) = args.value("json") {
        result.artifact.write(path).map_err(|e| format!("cannot write {path}: {e}"))?;
        println!("\nwrote {path}");
    }

    // A demo that produced a non-finite value has failed, however pretty it looked.
    let healthy =
        result.artifact.first_non_finite().is_none() && result.artifact.warnings().is_empty();
    Ok(if healthy { ExitCode::SUCCESS } else { ExitCode::from(CHECK_FAILED) })
}

fn cmd_inspect(args: &Args) -> Result<ExitCode, String> {
    check_flags(args, &[])?;

    let subject = args.positional.first().map(String::as_str);
    let text = match subject {
        Some("contracts") => inspect::contracts(),
        Some("units") => {
            // Join the rest so an unquoted expression still works.
            let expression = args.positional[1..].join(" ");
            inspect::units((!expression.is_empty()).then_some(expression.as_str()))?
        }
        Some("cases") => inspect::cases(),
        Some("benchmarks") => inspect::benchmarks(),
        Some("demos") => inspect::demos(),
        None => {
            "usage: lattice inspect <subject>\n\n\
             subjects:\n  \
             contracts    every solver's equations, assumptions, and known limitations\n  \
             units        resolve a unit expression, or list the registry\n  \
             cases        the validation cases and what each establishes\n  \
             benchmarks   the benchmarks and their correctness conditions\n  \
             demos        the demonstration scenes\n"
                .to_string()
        }
        Some(other) => {
            return Err(format!(
                "cannot inspect `{other}`; expected contracts, units, cases, benchmarks, or demos"
            ));
        }
    };
    print!("{text}");
    Ok(ExitCode::SUCCESS)
}

fn usage() -> String {
    format!(
        "\
lattice {version} — a 2D multiphysics simulation runtime

USAGE
  lattice <command> [options]

COMMANDS
  check <file.lattice>     compile a model and print its report without running it
    --quiet                  suppress the model report, keep the diagnostics

  run <file.lattice>       compile and execute a model
    --duration <seconds>     override the model's `duration:`
    --steps <n>              stop after this many steps
    --timestep <seconds>     override the negotiated timestep
    --threads <n|auto>       split solver loops across n threads (default 1)
    --json <path>            write the run artifact
    --quiet                  suppress the visualization

  validate                 run the validation suite and report measured error metrics
    --filter <pattern>       run only cases whose name or domain contains <pattern>
    --json <path>            write a machine-readable report
    --quiet                  suppress the text report

  bench [<pattern>]        run benchmarks, reporting throughput and correctness together
    --scale <n>              problem-size multiplier (default 1)
    --threads <n|auto>       split solver loops across n threads (default 1)
    --backend <cpu|gpu>      which backend runs the kernels (default cpu; gpu needs
                             a build with --features gpu, and runs fast32)
    --compare                also run the sequential CPU baseline and report the speedup
    --json <path>            write a machine-readable report
    --quiet                  suppress the text report

  demo <name>              run a demonstration scene
    --steps <n>              simulation steps (default 400)
    --samples <n>            timeline samples to record (default 40)
    --json <path>            write the run artifact
    --quiet                  suppress the visualization

  inspect <subject>        show contracts, units, cases, benchmarks, or demos

  help, --help, -h         this message
  --version, -V            version and build configuration

EXAMPLES
  lattice check examples/slab.lattice
  lattice run examples/slab.lattice --json runs/slab.json
  lattice run examples/chamber.lattice --threads auto
  lattice validate
  lattice validate --filter grid2d --json runs/validation.json
  lattice bench heat --scale 2
  lattice bench heat-explicit --threads auto --compare
  lattice bench heat-explicit --backend gpu --compare
  lattice demo oscillator
  lattice demo heat-gaussian --steps 800 --json runs/heat.json
  lattice inspect contracts
  lattice inspect units \"9.31e-9 meter^2 / second\"

EXIT CODES
  0  everything requested succeeded
  1  ran, but a check failed
  2  the command line was wrong and nothing ran
",
        version = env!("CARGO_PKG_VERSION")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Args {
        Args::parse(line.split_whitespace().map(String::from))
    }

    #[test]
    fn help_and_version_succeed() {
        assert_eq!(run(&parse("--help")).unwrap(), ExitCode::SUCCESS);
        assert_eq!(run(&parse("--version")).unwrap(), ExitCode::SUCCESS);
        assert_eq!(run(&parse("help")).unwrap(), ExitCode::SUCCESS);
        // No arguments at all prints help rather than erroring.
        assert_eq!(run(&parse("")).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn an_unknown_command_is_a_usage_error() {
        let err = run(&parse("frobnicate")).unwrap_err();
        assert!(err.contains("frobnicate"), "{err}");
        assert!(err.contains("validate"), "the message should list valid commands: {err}");
    }

    /// A mistyped flag must stop the run. Ignoring it would silently drop the
    /// artifact the user asked for.
    #[test]
    fn a_mistyped_flag_is_rejected() {
        let err = run(&parse("validate --jsno out.json")).unwrap_err();
        assert!(err.contains("--jsno"), "{err}");
        assert!(err.contains("--json"), "the message should show what was expected: {err}");
    }

    #[test]
    fn a_malformed_numeric_flag_is_rejected() {
        let err = run(&parse("bench --scale zero")).unwrap_err();
        assert!(err.contains("--scale"), "{err}");
    }

    #[test]
    fn a_zero_scale_is_rejected() {
        assert!(run(&parse("bench --scale 0")).is_err());
        assert!(run(&parse("demo oscillator --steps 0")).is_err());
    }

    #[test]
    fn validate_runs_and_a_filter_narrows_it() {
        assert_eq!(run(&parse("validate --filter free_fall --quiet")).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn a_filter_matching_nothing_is_an_error_not_a_silent_pass() {
        let err = run(&parse("validate --filter nonexistent --quiet")).unwrap_err();
        assert!(err.contains("no validation case"), "{err}");
    }

    #[test]
    fn demo_requires_a_known_name() {
        let err = run(&parse("demo")).unwrap_err();
        assert!(err.contains("scene name"), "{err}");
        let err = run(&parse("demo not-a-demo")).unwrap_err();
        assert!(err.contains("not-a-demo"), "{err}");
    }

    #[test]
    fn a_demo_runs_and_succeeds() {
        assert_eq!(
            run(&parse("demo oscillator --steps 20 --samples 4 --quiet")).unwrap(),
            ExitCode::SUCCESS
        );
    }

    #[test]
    fn inspect_subjects_all_work() {
        for subject in ["contracts", "cases", "benchmarks", "demos", "units"] {
            assert_eq!(
                run(&parse(&format!("inspect {subject}"))).unwrap(),
                ExitCode::SUCCESS,
                "inspect {subject}"
            );
        }
        // With no subject, the command lists them.
        assert_eq!(run(&parse("inspect")).unwrap(), ExitCode::SUCCESS);
        assert!(run(&parse("inspect nonsense")).is_err());
    }

    #[test]
    fn inspect_units_accepts_an_unquoted_expression() {
        assert_eq!(run(&parse("inspect units 9.31e-9 meter^2 / second")).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn usage_text_documents_every_command() {
        let text = usage();
        for command in ["check", "run", "validate", "bench", "demo", "inspect"] {
            assert!(text.contains(command), "usage is missing `{command}`");
        }
        assert!(text.contains("EXIT CODES"), "exit codes must be documented");
    }

    #[test]
    fn check_and_run_require_a_file() {
        assert!(run(&parse("check")).is_err());
        assert!(run(&parse("run")).is_err());
    }

    #[test]
    fn check_and_run_reject_unknown_flags() {
        let error = run(&parse("check model.lattice --verbse")).unwrap_err();
        assert!(error.contains("--verbse"), "{error}");
    }
}
