//! `lattice-view` — open a model in a window.
//!
//! A separate binary from `lattice`. The GPU stack it needs is a few hundred crates
//! and several minutes of cold build; keeping it out of the core CLI means
//! `lattice validate` stays dependency-free and quick to compile.

use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let play = arguments.iter().any(|a| a == "--play");
    let Some(path) = arguments.iter().find(|a| !a.starts_with('-')).cloned() else {
        let asked_for_help = arguments.iter().any(|a| a == "--help" || a == "-h");
        if asked_for_help {
            println!("{}", usage());
            return ExitCode::SUCCESS;
        }
        eprintln!("{}", usage());
        return ExitCode::from(2);
    };
    if path == "--help" || path == "-h" {
        println!("{}", usage());
        return ExitCode::SUCCESS;
    }
    if path == "--version" || path == "-V" {
        println!("lattice-view {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    let result = if arguments.iter().any(|a| a == "--gpu") {
        let smoke = arguments
            .iter()
            .find_map(|a| a.strip_prefix("--smoke-frames="))
            .map(str::parse::<u64>)
            .transpose();
        let smoke = match smoke {
            Ok(n) => n,
            Err(e) => {
                eprintln!("invalid smoke frame count: {e}");
                return ExitCode::from(2);
            }
        };
        lattice_viewer::gpu::run_with_smoke(&path, play || smoke.is_some(), smoke)
    } else {
        lattice_viewer::run_with(&path, play)
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn usage() -> String {
    format!(
        "\
lattice-view {version} — open a .lattice model in a window

USAGE
  lattice-view <file.lattice> [--play] [--gpu]

  --play    start running immediately instead of opening paused
  --gpu     resident heat field with direct GPU texture rendering

The window shows the model's fields and particles, transport controls, live plots of
every observed quantity, conservation drift, the timestep against its stability limit,
and the solver's published contract.

A model that does not compile opens anyway and shows its diagnostics; fix the file and
press reset.

EXAMPLES
  lattice-view examples/slab.lattice
  lattice-view examples/lj_gas.lattice

See also `lattice check` and `lattice run` for the headless equivalents.
",
        version = env!("CARGO_PKG_VERSION")
    )
}
