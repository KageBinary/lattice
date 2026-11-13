//! The `vacf` binary, run as a script would run it.

use std::path::PathBuf;
use std::process::Command;

fn model(every: &str) -> String {
    format!(
        "project gas {{
  dimensions: 2;
  fidelity: molecular;
  duration: 5 picosecond;

  particles atoms {{
    count:   25;
    region:  [5 nanometer, 5 nanometer];
    spacing: 1 nanometer;
    mass:    39.948 dalton;
    temperature: 94 kelvin;
    seed:    7;
    boundary: periodic;
  }}

  solve dynamics(atoms) with velocity_verlet(dt=10 femtosecond);

  observe atoms every {every};
}}
"
    )
}

fn write_model(tag: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("vacf-bin-{}-{tag}.lattice", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

/// A healthy run prints the curve and exits 0.
#[test]
fn a_healthy_run_prints_a_curve() {
    let path = write_model("healthy", &model("0.1 picosecond"));
    let output = Command::new(env!("CARGO_BIN_EXE_vacf")).arg(&path).args(["atoms", "--lags", "6"]).output().unwrap();
    std::fs::remove_file(&path).ok();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("# vacf ") && stdout.contains("# lag tau_seconds C"), "{stdout}");
}

/// An observer that disables itself partway (here, `every` not a multiple of the timestep,
/// so the sample spacing changes) leaves a truncated sample set. The binary fails and
/// prints no curve, rather than exit 0 with a curve from the samples taken before the fault.
#[test]
fn an_observer_fault_fails_the_binary_and_prints_no_curve() {
    let path = write_model("uneven", &model("0.105 picosecond"));
    let output = Command::new(env!("CARGO_BIN_EXE_vacf")).arg(&path).args(["atoms", "--lags", "6"]).output().unwrap();
    std::fs::remove_file(&path).ok();
    assert!(!output.status.success(), "exit {}: {}", output.status, String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("disabled") && stderr.contains("spacing changed"), "{stderr}");
    assert!(output.stdout.is_empty(), "no curve is printed: {}", String::from_utf8_lossy(&output.stdout));
}
