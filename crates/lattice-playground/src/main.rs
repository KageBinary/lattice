//! `lattice-play` — an interactive sandbox.
//!
//! Separate from `lattice-view` because they answer different questions. The viewer
//! opens a `.lattice` model you wrote and tells you whether to believe it. The
//! playground has no model at all: you build the scene by clicking, and the physics
//! underneath is the same validated solver either way.

use std::process::ExitCode;

use lattice_playground::{run, Mode};

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", usage());
        return ExitCode::SUCCESS;
    }
    if arguments.iter().any(|a| a == "--version" || a == "-V") {
        println!("lattice-play {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    // An unrecognized mode is refused rather than silently ignored: someone who typed
    // `--mode chemisty` wants to know, not to get the physics sandbox.
    let start = match arguments.iter().position(|a| a == "--mode") {
        None => Mode::default(),
        Some(index) => match arguments.get(index + 1).map(String::as_str).and_then(Mode::parse) {
            Some(mode) => mode,
            None => {
                eprintln!(
                    "error: --mode takes one of: {}",
                    Mode::ALL.iter().map(Mode::name).collect::<Vec<_>>().join(", ")
                );
                return ExitCode::from(2);
            }
        },
    };

    if let Err(error) = run(start) {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn usage() -> String {
    format!(
        "\
lattice-play {version} — an interactive physics and chemistry sandbox

USAGE
  lattice-play [--mode <name>]

  --mode <name>   which sandbox to open: {modes}

CONTROLS
  left click      use the selected tool — drop a shape, paint, or grab
  left drag       hold something and move it; release while moving to throw
  right click     remove what is under the cursor
  play / pause    the scene keeps running while you interact with it
  reset           start the mode over

Nothing here loads a model file. To open a written `.lattice` model with its full
diagnostics instead, use `lattice-view <file.lattice>`.
",
        version = env!("CARGO_PKG_VERSION"),
        modes = Mode::ALL.iter().map(Mode::name).collect::<Vec<_>>().join(", "),
    )
}
