//! Fixture tests over the repository's `.lattice` files.
//!
//! Spec §20.4 sets two acceptance criteria this covers:
//!
//! > One command validates and runs a project.
//! >
//! > Typed units reject at least ten deliberately invalid fixture models with
//! > source-positioned errors.
//!
//! Every file in `examples/` must compile *and run*. Every file in `tests/invalid/`
//! must be rejected with the specific diagnostic it declares in its own header:
//!
//! ```text
//! // expect: E0400
//! ```
//!
//! Naming the expected code matters. A test that only asserts "this failed" passes
//! just as happily when the model is rejected for the wrong reason, which is how
//! error quality quietly rots.

use std::fs;
use std::path::{Path, PathBuf};

use lattice_compiler::compile_source;
use lattice_runtime::{RunConfig, Simulation};
use lattice_syntax::SourceFile;

/// The repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> should have two ancestors")
        .to_path_buf()
}

/// Every `.lattice` file in a directory, sorted for a stable test order.
fn lattice_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(directory)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", directory.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "lattice"))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> SourceFile {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    SourceFile::new(name, text)
}

/// The diagnostic codes a fixture declares it should produce.
fn expected_codes(source: &str) -> Vec<String> {
    source
        .lines()
        .take(10)
        .filter_map(|line| line.trim().strip_prefix("// expect:"))
        .flat_map(|list| {
            list.split(',').map(|code| code.trim().to_string()).filter(|code| !code.is_empty())
        })
        .collect()
}

/// Every example must compile without errors.
#[test]
fn every_example_compiles() {
    let directory = repo_root().join("examples");
    let files = lattice_files(&directory);
    assert!(!files.is_empty(), "no examples found in {}", directory.display());

    for path in &files {
        let file = load(path);
        let (compiled, diagnostics) = compile_source(&file);
        assert!(
            !diagnostics.has_errors(),
            "{} should compile:\n{}",
            path.display(),
            diagnostics.render(&file)
        );
        let compiled = compiled.expect("a model without errors should be produced");
        assert!(
            !compiled.domains.is_empty(),
            "{} compiled to no solvers, so it would do nothing",
            path.display()
        );
    }
}

/// Every example must also *run*. Compiling is not the same as being steppable: a
/// model can pass every static check and still fail to negotiate a timestep.
#[test]
fn every_example_runs() {
    for path in lattice_files(&repo_root().join("examples")) {
        let file = load(path.as_path());
        let (compiled, diagnostics) = compile_source(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        let compiled = compiled.unwrap();

        // A short run: enough to exercise stepping and the observers without making
        // the test suite slow.
        let mut config = RunConfig::from_model(&compiled.model);
        config.duration = None;
        config.max_steps = Some(20);

        let mut simulation = Simulation::new(compiled.model, compiled.domains);
        let outcome = simulation.run(&config);

        assert!(
            outcome.is_success(),
            "{} failed to run: {}",
            path.display(),
            outcome.stop.describe()
        );
        assert!(
            outcome.artifact.first_non_finite().is_none(),
            "{} produced a non-finite observation",
            path.display()
        );
        assert_eq!(outcome.clock.step, 20, "{} did not take every step", path.display());
    }
}

/// Every example must declare a duration, so `lattice run <file>` works with no
/// arguments. An example that needs a flag to do anything is a bad example.
#[test]
fn every_example_declares_a_duration() {
    for path in lattice_files(&repo_root().join("examples")) {
        let file = load(path.as_path());
        let (compiled, _) = compile_source(&file);
        let compiled = compiled.unwrap();
        assert!(
            compiled.model.duration.is_some(),
            "{} has no `duration:`, so `lattice run` on it would refuse to start",
            path.display()
        );
    }
}

/// Spec §20.4: at least ten invalid models, each rejected with a source-positioned
/// error, and each for the *specific* reason it declares.
#[test]
fn every_invalid_fixture_is_rejected_for_its_declared_reason() {
    let directory = repo_root().join("tests").join("invalid");
    let files = lattice_files(&directory);
    assert!(
        files.len() >= 10,
        "spec §20.4 asks for at least ten invalid fixtures, found {}",
        files.len()
    );

    for path in &files {
        let file = load(path);
        let expected = expected_codes(file.text());
        assert!(
            !expected.is_empty(),
            "{} declares no expected diagnostic; add `// expect: E0000` to its header",
            path.display()
        );

        let (_, diagnostics) = compile_source(&file);
        let rendered = diagnostics.render(&file);

        assert!(
            diagnostics.has_errors(),
            "{} should have been rejected but compiled cleanly",
            path.display()
        );

        let produced = diagnostics.codes();
        for code in &expected {
            assert!(
                produced.contains(&code.as_str()),
                "{} should produce {code} but produced {produced:?}\n{rendered}",
                path.display()
            );
        }

        // FR-002 asks for *source-positioned* diagnostics, so every rejection must
        // point at a line and underline something.
        assert!(
            rendered.contains("-->") && rendered.contains('^'),
            "{} was rejected without a source position:\n{rendered}",
            path.display()
        );
    }
}

/// Diagnostics must be actionable, not merely correct. Every rejection should either
/// suggest a fix or explain the rule it enforces.
#[test]
fn every_rejection_offers_help_or_explanation() {
    for path in lattice_files(&repo_root().join("tests").join("invalid")) {
        let file = load(path.as_path());
        let (_, diagnostics) = compile_source(&file);
        let rendered = diagnostics.render(&file);
        assert!(
            rendered.contains("= help:") || rendered.contains("= note:"),
            "{} produced an error with no help or note:\n{rendered}",
            path.display()
        );
    }
}

/// The fixture set must cover a spread of failure kinds, not fifteen variations of
/// one. Each of these classes is a different part of the compiler.
#[test]
fn the_fixture_set_covers_every_diagnostic_class() {
    let mut seen: Vec<String> = Vec::new();
    for path in lattice_files(&repo_root().join("tests").join("invalid")) {
        let file = load(path.as_path());
        seen.extend(expected_codes(file.text()));
    }

    let classes: &[(&str, &str)] = &[
        ("E0108", "parser: fractional exponent"),
        ("E0200", "resolution: unknown name"),
        ("E0201", "resolution: duplicate declaration"),
        ("E0202", "resolution: unknown reference"),
        ("E0203", "resolution: missing required setting"),
        ("E0204", "resolution: unknown setting or argument"),
        ("E0206", "lowering: unknown solver"),
        ("E0210", "builtins: unknown function"),
        ("E0400", "units: dimensional mismatch"),
        ("E0403", "units: affine scale in an expression"),
        ("E0405", "validation: value out of range"),
        ("E0900", "not yet implemented"),
    ];
    for (code, description) in classes {
        assert!(
            seen.iter().any(|s| s == code),
            "no fixture covers {code} ({description})"
        );
    }
}

/// Independent problems must all be reported from one compile, not one per run.
#[test]
fn multiple_independent_errors_are_reported_together() {
    let file = SourceFile::new(
        "multi.lattice",
        r#"
project multi {
  grid bad { size: [8, 8]; extent: [1 metre_, 1 meter]; }
  grid good { size: [4, 4]; extent: [1 meter, 1 meter]; }
  grid good { size: [2, 2]; extent: [1 meter, 1 meter]; }
  field t on nowhere = 1 kelvin;
}
"#,
    );
    let (_, diagnostics) = compile_source(&file);
    let codes = diagnostics.codes();

    assert!(codes.contains(&"E0200"), "the unknown unit should be reported: {codes:?}");
    assert!(codes.contains(&"E0201"), "the duplicate grid should be reported: {codes:?}");
    assert!(codes.contains(&"E0202"), "the unknown grid should be reported: {codes:?}");
}

/// A declaration that failed is *not* registered, so later references to it report
/// "unknown" rather than compounding on a half-built entry.
///
/// This is a deliberate choice, not an accident. The alternative — registering a
/// placeholder — would suppress the follow-on error but let a 1×1 stand-in grid reach
/// the model report, where it reads as a real configuration the user did not write.
/// One clear error plus one honest follow-on beats one error and a fictional report.
#[test]
fn a_failed_declaration_is_not_registered() {
    let file = SourceFile::new(
        "cascade.lattice",
        r#"
project cascade {
  grid g { size: [8, 8]; }
  field t on g = 300 kelvin { diffusivity: 1e-5 meter^2/second; }
  solve heat(t) with crank_nicolson(dt=0.1 second);
}
"#,
    );
    let (compiled, diagnostics) = compile_source(&file);
    let codes = diagnostics.codes();

    // The grid is missing its extent...
    assert!(codes.contains(&"E0203"), "{codes:?}");
    // ...and the field that referred to it says so rather than silently succeeding.
    assert!(codes.contains(&"E0202"), "{codes:?}");
    // Nothing was lowered, so no fictional 1x1 grid appears in the report.
    let compiled = compiled.unwrap();
    assert!(compiled.domains.is_empty());
    assert!(!compiled.model.report().contains("1x1"));
}

/// Diagnostics are sorted by position, so a run of errors reads top to bottom the way
/// the file does.
#[test]
fn diagnostics_are_ordered_by_position() {
    let file = SourceFile::new(
        "ordered.lattice",
        r#"
project ordered {
  grid a { size: [8, 8]; extent: [1 zzzz, 1 meter]; }
  grid b { size: [8, 8]; extent: [1 meter, 1 yyyy]; }
}
"#,
    );
    let (_, diagnostics) = compile_source(&file);
    let positions: Vec<u32> = diagnostics.iter().map(|d| d.primary_span().start).collect();
    let mut sorted = positions.clone();
    sorted.sort_unstable();
    assert_eq!(positions, sorted, "diagnostics should read in source order");
}
