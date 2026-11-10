//! `vacf MODEL SET [--lags N] [--steps N]`: compile a model, attach the velocity
//! autocorrelation observer to the particle domain `SET`, run, and print `C(τ)`.
//!
//! The output is plain text, one lag per line, so a script can read it:
//!
//! ```text
//! # vacf model=gas.lattice set=atoms particles=400 samples=601 interval=1e-13
//! # lag tau_seconds C
//! 0 0e0 1e0
//! 1 1e-13 9.04e-1
//! ```

use std::path::Path;
use std::process::ExitCode;

use lattice_sdk::{compile_file, ParticleView, RunConfig, SourceFile, SDK_VERSION};
use vacf::{Vacf, WRITTEN_FOR_SDK};

const USAGE: &str = "usage: vacf MODEL.lattice SET [--lags N] [--steps N]";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    if SDK_VERSION != WRITTEN_FOR_SDK {
        return Err(format!(
            "this plugin was written for lattice-sdk {WRITTEN_FOR_SDK:?} and is linked against {SDK_VERSION:?}"
        ));
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut positional, mut lags, mut steps) = (Vec::new(), 50usize, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--lags" => lags = number(rest.next(), "--lags")?,
            "--steps" => steps = Some(number(rest.next(), "--steps")?),
            flag if flag.starts_with("--") => return Err(format!("unknown flag `{flag}`\n{USAGE}")),
            _ => positional.push(arg.as_str()),
        }
    }
    let [model, set] = positional[..] else { return Err(USAGE.to_string()) };

    let mut simulation = compile_file(Path::new(model)).map_err(|diagnostics| {
        // The file was readable if it produced diagnostics other than the read error.
        let file = std::fs::read_to_string(model)
            .map(|text| SourceFile::new(model, text))
            .ok();
        let rendered: Vec<String> = diagnostics
            .iter()
            .map(|d| file.as_ref().map_or_else(|| d.message.clone(), |f| d.render(f)))
            .collect();
        format!("{}\n`{model}` did not compile; nothing was run.", rendered.join("\n"))
    })?;

    let particles = simulation
        .domains()
        .iter()
        .find(|domain| domain.name() == set)
        .and_then(|domain| ParticleView::from_domain(&**domain))
        .map(|view| view.len())
        .ok_or_else(|| format!("`{model}` has no particle domain `{set}`"))?;

    let observer = Vacf::new(set, particles, lags)?;
    let curve_name = observer.curve_name().to_string();
    simulation
        .add_observer(Box::new(observer))
        .map_err(|gaps| format!("the observer was refused: {gaps:?}"))?;

    let mut config = RunConfig::from_model(simulation.model());
    if let Some(steps) = steps {
        config.max_steps = Some(steps as u64);
        config.duration = None;
    }
    let outcome = simulation.run(&config);
    for fault in &outcome.observer_faults {
        eprintln!("{}", fault.describe());
    }
    if !outcome.is_success() {
        return Err(format!("the run stopped early: {}", outcome.stop.describe()));
    }

    let curves = simulation.curves();
    let curve = curves
        .iter()
        .find(|curve| curve.name == curve_name)
        .ok_or_else(|| format!("the observer published no curve `{curve_name}`; it needs at least two samples"))?;
    let samples = outcome.artifact.timeline().len();
    let interval = curve.x.get(1).copied().unwrap_or(0.0);
    println!("# vacf model={model} set={set} particles={particles} samples={samples} interval={interval:e}");
    for note in &curve.notes {
        println!("# {note}");
    }
    println!("# lag tau_seconds C");
    for (lag, (tau, c)) in curve.x.iter().zip(&curve.y).enumerate() {
        println!("{lag} {tau:e} {c:e}");
    }
    Ok(())
}

fn number(value: Option<&String>, flag: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{flag} needs a number\n{USAGE}"))?
        .parse()
        .map_err(|_| format!("{flag} needs a whole number\n{USAGE}"))
}
