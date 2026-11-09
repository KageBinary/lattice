//! The model compiler: text in, runnable model out.
//!
//! Spec P3 — *compile models before running*:
//!
//! > User-friendly declarations should become validated, cross-indexed, preallocated
//! > runtime structures. Expensive interpretation, graph construction, and memory
//! > planning should occur before the hot loop.
//!
//! [`compile_source`] takes a `.lattice` file and produces a [`Compiled`]: an
//! immutable [`CompiledModel`](lattice_ir::CompiledModel) describing what the model
//! *is*, plus the instantiated solvers that will run it.
//!
//! # Example
//!
//! ```
//! use lattice_compiler::compile_source;
//! use lattice_syntax::SourceFile;
//!
//! let file = SourceFile::new("slab.lattice", r#"
//! project slab {
//!   fidelity: engineering_2d;
//!   duration: 20 second;
//!
//!   grid bar { size: [64, 8]; extent: [1 meter, 0.125 meter]; }
//!
//!   field temperature on bar = 273.15 kelvin {
//!     diffusivity: 1e-4 meter^2 / second;
//!     boundary_left:  fixed(273.15 kelvin);
//!     boundary_right: fixed(373.15 kelvin);
//!   }
//!
//!   solve heat(temperature) with crank_nicolson(dt=0.05 second);
//!   observe temperature every 1 second;
//! }
//! "#);
//!
//! let (compiled, diagnostics) = compile_source(&file);
//! assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
//!
//! let compiled = compiled.unwrap();
//! assert_eq!(compiled.domains.len(), 1);
//! assert_eq!(compiled.model.duration, Some(20.0));
//! // The field's dimension came from its initial value, not a separate declaration.
//! assert!(compiled.model.report().contains("grid2d.heat[crank_nicolson]"));
//! ```
//!
//! # Three passes, one report
//!
//! | Pass | Module | Answers |
//! |---|---|---|
//! | Evaluate | [`eval`] | what quantity is this expression, and does it type-check? |
//! | Recognize | [`builtins`] | is this a field initializer, a boundary, a method, a force? |
//! | Lower | [`compile`](mod@compile) | which solver implements this, with what memory and schedule? |
//!
//! Everything they learn ends up in the model report, which is what `lattice check`
//! prints. Spec §8.4 step 9 asks for exactly that: the compiled model *plus a report
//! describing equations and assumptions*, so a user can see what the compiler decided
//! before running anything.

pub mod builtins;
pub mod chemistry;
pub mod compile;
pub mod eval;
pub mod laws;
pub mod lower_law;
pub mod user_force;
pub mod molecular;
pub mod observe;
pub mod quantum;
pub mod rigid;

pub use compile::{compile, Compiled, LawUse};
pub use eval::Evaluator;

use lattice_syntax::{Diagnostics, SourceFile};

/// Parse and compile a source file.
///
/// Returns `None` only when the file could not be parsed at all. A file that parsed
/// but failed to compile still yields a best-effort model alongside the diagnostics,
/// so `lattice check` can show what it understood.
pub fn compile_source(file: &SourceFile) -> (Option<Compiled>, Diagnostics) {
    let (project, mut diagnostics) = lattice_syntax::parse(file);
    let Some(project) = project else {
        return (None, diagnostics);
    };
    let (compiled, compile_diagnostics) = compile(file, &project);
    diagnostics.extend(compile_diagnostics);
    diagnostics.sort_by_position();
    (Some(compiled), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(source: &str) -> (Option<Compiled>, Diagnostics, SourceFile) {
        let file = SourceFile::new("t.lattice", source);
        let (compiled, diagnostics) = compile_source(&file);
        (compiled, diagnostics, file)
    }

    fn ok(source: &str) -> Compiled {
        let (compiled, diagnostics, file) = build(source);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        compiled.expect("should compile")
    }

    fn err(source: &str) -> (String, Vec<String>) {
        let (_, diagnostics, file) = build(source);
        assert!(diagnostics.has_errors(), "expected an error from:\n{source}");
        (diagnostics.render(&file), diagnostics.codes().into_iter().map(String::from).collect())
    }

    const SLAB: &str = r#"
project slab {
  fidelity: engineering_2d;
  duration: 20 second;

  grid bar { size: [64, 8]; extent: [1 meter, 0.125 meter]; }

  field temperature on bar = 273.15 kelvin {
    diffusivity: 1e-4 meter^2 / second;
    boundary_left:  fixed(273.15 kelvin);
    boundary_right: fixed(373.15 kelvin);
  }

  solve heat(temperature) with crank_nicolson(dt=0.05 second);
  observe temperature every 1 second;
  visualize temperature as heatmap;
}
"#;

    #[test]
    fn a_heat_project_compiles_to_a_runnable_domain() {
        let compiled = ok(SLAB);
        assert_eq!(compiled.model.name, "slab");
        assert_eq!(compiled.domains.len(), 1);
        assert_eq!(compiled.model.duration, Some(20.0));
        assert_eq!(compiled.model.observers.len(), 1);
        assert_eq!(compiled.model.observers[0].interval, Some(1.0));
        assert_eq!(compiled.model.visuals[0].style.as_deref(), Some("heatmap"));

        // Two operations per domain: prepare and advance, plus the observer.
        assert_eq!(compiled.model.graph.len(), 3);
        assert!(compiled.model.memory_bytes() > 0);
    }

    /// A field's dimension is inferred from its initial value, so a temperature field
    /// needs no separate declaration — and a mismatched boundary is caught against it.
    #[test]
    fn a_field_dimension_is_inferred_and_then_enforced() {
        let compiled = ok(SLAB);
        assert!(compiled.model.report().contains("D = 1.0000e-4"));

        let (text, codes) = err(&SLAB.replace("fixed(373.15 kelvin)", "fixed(5 second)"));
        assert!(codes.contains(&"E0400".to_string()));
        assert!(text.contains("expected temperature"), "{text}");
        assert!(text.contains("found time"), "{text}");
    }

    #[test]
    fn the_model_report_names_the_solver_and_its_contract() {
        let text = ok(SLAB).model.report();
        assert!(text.contains("grid2d.heat[crank_nicolson]"), "{text}");
        assert!(text.contains("Dirichlet"), "{text}");
        assert!(text.contains("does NOT conserve"), "the contract must travel with the model");
        assert!(text.contains("operation graph"), "{text}");
    }

    #[test]
    fn a_particle_project_compiles() {
        let compiled = ok(r#"
project gas {
  fidelity: molecular;
  duration: 1 second;

  particles atoms {
    count: 64;
    region: [12 meter, 12 meter];
    spacing: 1.4 meter;
    mass: 1 kilogram;
    speed: 0.4 meter/second;
    boundary: periodic;
    seed: 7;
    force: lennard_jones(epsilon=1 joule, sigma=1 meter);
  }

  solve dynamics(atoms) with velocity_verlet(dt=0.001 second);
  observe atoms every 0.1 second;
}
"#);
        assert_eq!(compiled.domains.len(), 1);
        let report = compiled.model.report();
        assert!(report.contains("particles2d[velocity_verlet]"), "{report}");
        assert!(report.contains("Lennard-Jones"), "{report}");
        assert!(report.contains("64 particles"), "{report}");
    }

    /// Multiple `force:` settings combine, which is how a scene gets gravity and drag.
    #[test]
    fn multiple_forces_combine() {
        let compiled = ok(r#"
project falling {
  particles drops {
    count: 4;
    mass: 2 kilogram;
    force: gravity;
    force: drag(coefficient=0.1 kilogram/second);
  }
  solve dynamics(drops) with velocity_verlet(dt=0.001 second);
}
"#);
        let report = compiled.model.report();
        assert!(report.contains("gravity"), "{report}");
        assert!(report.contains("drag"), "{report}");
    }

    // --- errors -------------------------------------------------------------

    #[test]
    fn an_unknown_grid_lists_the_declared_ones() {
        let (text, codes) = err(r#"
project p {
  grid chamber { size: [4, 4]; extent: [1 meter, 1 meter]; }
  field t on chambre = 1 kelvin { diffusivity: 1 meter^2/second; }
  solve heat(t) with crank_nicolson(dt=1 second);
}
"#);
        assert!(codes.contains(&"E0202".to_string()));
        assert!(text.contains("no grid called `chambre`"), "{text}");
        assert!(text.contains("declared grids: chamber"), "{text}");
    }

    #[test]
    fn a_duplicate_declaration_points_at_both() {
        let (text, codes) = err(r#"
project p {
  grid g { size: [4, 4]; extent: [1 meter, 1 meter]; }
  grid g { size: [8, 8]; extent: [1 meter, 1 meter]; }
}
"#);
        assert!(codes.contains(&"E0201".to_string()));
        assert!(text.contains("declared twice"), "{text}");
        assert!(text.contains("first declared here"), "{text}");
    }

    #[test]
    fn a_field_without_a_diffusivity_cannot_be_solved() {
        let (text, codes) = err(r#"
project p {
  grid g { size: [4, 4]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin;
  solve heat(t) with crank_nicolson(dt=1 second);
}
"#);
        assert!(codes.contains(&"E0203".to_string()));
        assert!(text.contains("needs a `diffusivity`"), "{text}");
    }

    /// An explicit scheme past its stability limit must be refused at compile time,
    /// not discovered when the field fills with NaN (NFR-007).
    #[test]
    fn an_unstable_explicit_timestep_is_refused_before_running() {
        let (text, codes) = err(r#"
project p {
  grid g { size: [64, 64]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin { diffusivity: 1 meter^2/second; }
  solve heat(t) with explicit(dt=1 second);
}
"#);
        assert!(codes.contains(&"E0405".to_string()));
        assert!(text.contains("exceeds the explicit stability limit"), "{text}");
        assert!(text.contains("crank_nicolson"), "the fix should be suggested: {text}");
    }

    #[test]
    fn an_unknown_setting_is_reported_with_the_alternatives() {
        let (text, codes) = err(r#"
project p {
  grid g { size: [4, 4]; extent: [1 meter, 1 meter]; diffusivity: 1 meter^2/second; }
}
"#);
        assert!(codes.contains(&"E0204".to_string()));
        assert!(text.contains("`grid` has no setting called `diffusivity`"), "{text}");
        assert!(text.contains("accepts: size, extent, origin"), "{text}");
    }

    #[test]
    fn an_unpaired_periodic_boundary_is_rejected() {
        let (text, codes) = err(r#"
project p {
  grid g { size: [4, 4]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin {
    diffusivity: 1e-6 meter^2/second;
    boundary_left: periodic;
  }
  solve heat(t) with crank_nicolson(dt=1 second);
}
"#);
        assert!(codes.contains(&"E0405".to_string()));
        assert!(text.contains("both edges"), "{text}");
    }

    #[test]
    fn a_negative_diffusivity_explains_why_it_is_wrong() {
        let (text, _) = err(r#"
project p {
  grid g { size: [4, 4]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin { diffusivity: -1 meter^2/second; }
  solve heat(t) with crank_nicolson(dt=1 second);
}
"#);
        assert!(text.contains("backwards"), "{text}");
    }

    #[test]
    fn a_declared_but_unsolved_field_warns() {
        let (_, diagnostics, file) = build(r#"
project p {
  grid g { size: [4, 4]; extent: [1 meter, 1 meter]; }
  field t on g = 300 kelvin { diffusivity: 1e-6 meter^2/second; }
}
"#);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        assert_eq!(diagnostics.warning_count(), 1);
        assert!(diagnostics.render(&file).contains("never solved"));
    }

    // --- constructs that do not exist yet ------------------------------------

    /// Spec §25.1's flagship project. Everything the engine understands must compile;
    /// what it cannot do must say which milestone brings it, rather than being
    /// silently dropped.
    #[test]
    fn the_spec_flagship_project_reports_exactly_what_is_missing() {
        let (compiled, diagnostics, file) = build(
            r#"
project hot_reaction {
  dimensions: 2;
  precision: mixed;
  fidelity: engineering_2d;

  grid chamber { size: [512, 256]; extent: [2 meter, 1 meter]; }

  field temperature on chamber = 298 kelvin;
  species A on chamber = left_half(1 mole / meter^2);
  species B on chamber = right_half(1 mole / meter^2);

  reaction A_plus_B {
    reactants: A + B;
    products: C;
    enthalpy: -25 kilojoule / mole;
  }

  couple A_plus_B.heat_release -> temperature.source conserve energy;

  observe total_species every 0.1 second;
  visualize temperature as heatmap;
}
"#,
        );
        let text = diagnostics.render(&file);

        // The parts that exist were understood.
        let compiled = compiled.expect("a best-effort model is still produced");
        assert_eq!(compiled.model.name, "hot_reaction");
        assert_eq!(compiled.model.observers.len(), 1);
        assert_eq!(compiled.model.visuals.len(), 1);

        // As of M3 the reaction and the coupling are both implemented, so what this
        // model is missing is no longer a milestone — it is two declarations. The
        // diagnostics say which, and the one that would be easiest to get wrong gets a
        // note of its own: a reaction is not a domain, so it cannot be coupled from.
        assert!(text.contains("no `solve reactions` names a mixture"), "{text}");
        assert!(text.contains("domain chemistry"), "{text}");
        assert!(
            text.contains("a reaction is not a domain"),
            "the couple should explain which thing publishes the port: {text}"
        );
        // And the grid, fields and species were still parsed and dimension-checked.
        assert!(!text.contains("mole / meter^2` is not"), "the species should type-check: {text}");
    }

    #[test]
    fn a_fluid_domain_names_its_milestone() {
        let (text, codes) = err(r#"
project f {
  domain fluid2d air { grid: [64, 64]; }
}
"#);
        assert!(codes.contains(&"E0900".to_string()), "{text}");
        assert!(text.contains("not implemented yet"), "{text}");
    }

    /// Spec §25.2, exactly as the spec writes it.
    const SPEC_DOUBLE_SLIT: &str = r#"
project double_slit {
  domain quantum2d q {
    grid: [768, 384];
    extent: [12 nanometer, 6 nanometer];
    mass: electron_mass;
    boundary: absorbing(width=0.8 nanometer);
    integrator: split_step_fourier(dt=0.002 femtosecond);
  }

  potential barrier {
    shape: vertical_wall(x=0, thickness=0.15 nanometer);
    slits: [(-1.0, 0.35), (1.0, 0.35)] nanometer;
    height: 20 electronvolt;
  }

  wavepacket initial {
    center: [-4 nanometer, 0];
    momentum: [6.5e-24 kilogram*meter/second, 0];
    sigma: 0.45 nanometer;
  }

  detector screen at x=4.5 nanometer;
  observe probability_norm every step;
  visualize probability_density;
  visualize phase;
}
"#;

    #[test]
    fn the_spec_double_slit_compiles_and_says_what_is_wrong_with_it() {
        let file = SourceFile::new("double_slit.lattice", SPEC_DOUBLE_SLIT);
        let (compiled, diagnostics) = compile_source(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        let compiled = compiled.unwrap();
        assert_eq!(compiled.domains.len(), 1);
        assert_eq!(compiled.model.domains[0].family, "quantum2d[split_step_fourier+absorbing]");
        let codes = diagnostics.codes();
        // 145 eV against a 20 eV wall, and 9.4 radians per step against sharp edges.
        assert!(codes.contains(&"W0308"), "{}", diagnostics.render(&file));
        assert!(codes.contains(&"W0310"), "{}", diagnostics.render(&file));
        assert!(!codes.contains(&"W0309"), "its momentum is well inside the grid's Nyquist");
        let report = compiled.model.report();
        assert!(report.contains("2 slits") && report.contains("detector `screen`"), "{report}");
    }

    /// A rock-salt set, with `{charge}` and `{forces}` filled in by each test.
    fn salt(charge: &str, forces: &str) -> String {
        format!(
            r#"
project salt {{
  duration: 1 picosecond;
  particles ions {{
    count: 16; region: [1.12 nanometer, 1.12 nanometer]; spacing: 2.8 angstrom;
    mass: 29 dalton; boundary: periodic; {charge}
    {forces}
  }}
  solve dynamics(ions) with velocity_verlet(dt=1 femtosecond);
}}
"#
        )
    }

    const CORE: &str = "force: lennard_jones(epsilon=0.5 electronvolt, sigma=2.5 angstrom, cutoff=5 angstrom);";

    #[test]
    fn charges_and_coulomb_compile_into_the_domain() {
        let source = salt(
            "charge: alternating(1 elementary_charge);",
            &format!("{CORE} force: coulomb(cutoff=5.5 angstrom);"),
        );
        let (compiled, diagnostics, file) = build(&source);
        assert!(diagnostics.is_empty(), "{}", diagnostics.render(&file));
        let report = compiled.unwrap().model.report();
        // The default damping is alpha R = 2.4.
        assert!(report.contains("Coulomb (damped shifted force) cutoff = 5.5000e-10 m, alpha = 4.3636e9 1/m"), "{report}");
    }

    #[test]
    fn coulomb_without_charges_or_with_a_net_charge_or_no_core_warns() {
        for (charge, forces, code, words) in [
            ("", format!("{CORE} force: coulomb(cutoff=5 angstrom);"), "W0311", "no charges"),
            ("charge: 1 elementary_charge;", format!("{CORE} force: coulomb(cutoff=5 angstrom);"), "W0311", "net charge"),
            ("charge: alternating(1 elementary_charge);", "force: coulomb(cutoff=5 angstrom);".to_string(), "W0312", "nothing keeping them apart"),
        ] {
            let source = salt(charge, &forces);
            let (_, diagnostics, file) = build(&source);
            let text = diagnostics.render(&file);
            assert!(!diagnostics.has_errors(), "{text}");
            assert!(diagnostics.codes().contains(&code), "{charge} / {forces}: {text}");
            assert!(text.contains(words), "{charge} / {forces}: {text}");
        }
    }

    #[test]
    fn a_coulomb_damping_must_be_an_inverse_length() {
        let source = salt(
            "charge: alternating(1 elementary_charge);",
            &format!("{CORE} force: coulomb(cutoff=5 angstrom, damping=2 angstrom);"),
        );
        let (_, diagnostics, file) = build(&source);
        assert!(diagnostics.has_errors(), "{}", diagnostics.render(&file));
        assert!(diagnostics.render(&file).contains("Coulomb damping"), "{}", diagnostics.render(&file));
    }

    #[test]
    fn a_detector_can_count_clicks_and_says_so() {
        let source = SPEC_DOUBLE_SLIT.replace(
            "detector screen at x=4.5 nanometer;",
            "detector screen at x=4.5 nanometer, clicks=2500, seed=9;",
        );
        let file = SourceFile::new("clicks.lattice", source);
        let (compiled, diagnostics) = compile_source(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        let report = compiled.unwrap().model.report();
        assert!(report.contains("counting 2500 arrivals (seed 9)"), "{report}");

        for (written, code, words) in [
            ("detector screen at x=4.5 nanometer, seed=9;", "E0204", "has none"),
            ("detector screen at x=4.5 nanometer, clicks=2.5;", "E0404", "whole number"),
            ("detector screen at x=4.5 nanometer, click=10;", "E0204", "no argument called `click`"),
        ] {
            let source = SPEC_DOUBLE_SLIT.replace("detector screen at x=4.5 nanometer;", written);
            let file = SourceFile::new("clicks.lattice", source);
            let (_, diagnostics) = compile_source(&file);
            let text = diagnostics.render(&file);
            assert!(diagnostics.codes().contains(&code), "{written}: {text}");
            assert!(text.contains(words), "{written}: {text}");
        }
    }

    #[test]
    fn quantum_declarations_without_a_domain_are_rejected() {
        let (text, codes) = err("project p { wavepacket w { center: [0, 0]; sigma: 1 nanometer; } }");
        assert!(codes.contains(&"E0203".to_string()), "{text}");
        assert!(text.contains("needs a quantum domain"), "{text}");
    }

    #[test]
    fn a_quantum_domain_needs_a_state_and_its_edges_must_suit_its_integrator() {
        let (text, codes) = err(r#"
project p {
  domain quantum2d q { grid: [16, 16]; extent: [1 nanometer, 1 nanometer]; mass: electron_mass; }
}
"#);
        assert!(codes.contains(&"E0203".to_string()) && text.contains("no wave packet"), "{text}");

        let (text, codes) = err(r#"
project p {
  domain quantum2d q {
    grid: [16, 16]; extent: [1 nanometer, 1 nanometer]; mass: electron_mass;
    boundary: periodic; integrator: crank_nicolson(dt=1 attosecond);
  }
  wavepacket w { center: [0, 0]; sigma: 0.1 nanometer; }
}
"#);
        assert!(codes.contains(&"E0208".to_string()), "{text}");
    }

    #[test]
    fn a_harmonic_trap_and_a_box_compile_under_crank_nicolson() {
        let file = SourceFile::new(
            "trap.lattice",
            r#"
project trap {
  duration: 1 femtosecond;
  domain quantum2d q {
    grid: [32, 32]; extent: [4 nanometer, 4 nanometer]; mass: electron_mass;
    integrator: crank_nicolson(dt=0.01 femtosecond, tolerance=1e-11);
  }
  potential well { shape: harmonic(omega=1.5e14 / second); }
  wavepacket w { center: [0.3 nanometer, 0]; sigma: [0.4 nanometer, 0.3 nanometer]; }
}
"#,
        );
        let (compiled, diagnostics) = compile_source(&file);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        assert_eq!(compiled.unwrap().model.domains[0].family, "quantum2d[crank_nicolson]");
    }

    #[test]
    fn an_unknown_declaration_kind_is_distinguished_from_a_planned_one() {
        let (text, codes) = err("project p { frobnicator x { } }");
        assert!(codes.contains(&"E0205".to_string()), "{text}");
        assert!(text.contains("not a known declaration"), "{text}");
        assert!(text.contains("available now: grid, particles"), "{text}");
    }

    #[test]
    fn fast32_precision_points_at_the_gpu_milestone() {
        let (text, codes) = err("project p { precision: fast32; }");
        assert!(codes.contains(&"E0900".to_string()));
        assert!(text.contains("M4"), "{text}");
    }

    /// `mixed` is accepted but noted, because on the CPU path it is simply 64-bit.
    /// Silently treating it as equivalent would misrepresent what ran.
    #[test]
    fn mixed_precision_is_accepted_with_a_note() {
        let compiled = ok("project p { precision: mixed; }");
        assert!(
            compiled.model.notes.iter().any(|note| note.contains("mixed")),
            "{:?}",
            compiled.model.notes
        );
    }

    #[test]
    fn a_three_dimensional_model_is_refused() {
        let (text, codes) = err("project p { dimensions: 3; }");
        assert!(codes.contains(&"E0900".to_string()));
        assert!(text.contains("P9"), "{text}");
    }

    // --- the compiled model actually runs ------------------------------------

    /// The end of the pipeline: a compiled model steps, and its physics is right.
    #[test]
    fn a_compiled_model_steps_and_conserves() {
        let mut compiled = ok(r#"
project closed {
  grid box { size: [32, 32]; extent: [1 meter, 1 meter]; }
  field t on box = gaussian(center=[0.5 meter, 0.5 meter], sigma=0.1 meter, peak=100 kelvin) {
    diffusivity: 1e-4 meter^2 / second;
    boundary: insulated;
  }
  solve heat(t) with crank_nicolson(dt=0.05 second);
}
"#);
        let mut arena = lattice_ir::Arena::with_capacity(compiled.model.buffers.scratch_elements());
        let mut ctx = lattice_ir::StepContext::new(&mut arena);

        let mut observations = lattice_ir::Observations::new();
        compiled.domains[0].observe(&mut observations);
        let before = observations.value("t.integral").expect("the integral is observed");

        for _ in 0..100 {
            compiled.domains[0].advance(0.05, &mut ctx);
        }

        observations.clear();
        compiled.domains[0].observe(&mut observations);
        let after = observations.value("t.integral").unwrap();

        let drift = (after - before).abs() / before.abs();
        assert!(drift < 1e-9, "a closed domain drifted {drift:e}");
        // And the initial condition really was the Gaussian, not a uniform fill.
        assert!(before > 0.0);
    }

    #[test]
    fn a_compiled_particle_model_steps() {
        let mut compiled = ok(r#"
project orbit {
  particles p {
    count: 1;
    mass: 1 kilogram;
    force: harmonic_well(center=[0 meter, 0 meter], stiffness=4 newton/meter);
  }
  solve dynamics(p) with velocity_verlet(dt=0.001 second);
}
"#);
        let mut arena = lattice_ir::Arena::with_capacity(0);
        let mut ctx = lattice_ir::StepContext::new(&mut arena);
        for _ in 0..1000 {
            compiled.domains[0].advance(0.001, &mut ctx);
        }
        let mut observations = lattice_ir::Observations::new();
        compiled.domains[0].observe(&mut observations);
        assert!(observations.first_non_finite().is_none());
    }

    // --- `let` constants and law names, end to end -------------------------------

    /// A constant is usable in any setting, wherever the setting is written, and only
    /// the constants above it in another `let`.
    #[test]
    fn constants_reach_every_setting() {
        // Both settings come before the constants they read.
        let compiled = ok(&SLAB.replace(
            "duration: 20 second;",
            "duration: run_for; timestep: step; let run_for = 20 second; let step = run_for / 400;",
        ));
        assert_eq!(compiled.model.duration, Some(20.0));
        assert_eq!(compiled.model.timestep, Some(0.05));
    }

    #[test]
    fn a_constant_uses_only_the_ones_above_it() {
        let (_, codes) = err("project p { let a = b; let b = 1 meter; }");
        assert_eq!(codes, ["E0200"], "`b` is not defined yet where `a` reads it");
    }

    #[test]
    fn constants_are_declared_once_and_a_unit_shadow_is_warned_about() {
        let (_, codes) = err("project p { let k = 1; let k = 2; }");
        assert_eq!(codes, ["E0201"]);
        let (_, diagnostics, _) = build("project p { let m = 2 kilogram; }");
        assert_eq!(diagnostics.codes(), ["W0313"]);
    }

    #[test]
    fn a_law_cannot_take_a_built_in_force_name() {
        let (_, codes) = err("project p { force drag(a: particle) -> vec2<newton> { return vec2(0 newton, 0 newton); } }");
        assert!(codes.contains(&"E0201".to_string()), "{codes:?}");
    }

    /// A law used by name compiles into the particle set, and the report carries its
    /// bindings and its contract.
    #[test]
    fn a_law_used_by_name_is_bound_and_described() {
        let compiled = ok("project p {
            force push(a: particle) -> vec2<newton> {
                param strength: newton;
                return vec2(strength, 0 newton);
            }
            particles s { count: 1; mass: 1 kilogram; force: push(strength=2 newton); }
            solve dynamics(s) with velocity_verlet(dt=0.01 second);
        }");
        let summary = &compiled.model.domains[0].summary;
        assert!(summary.contains("user force `push`") && summary.contains("strength = 2.0000e0"), "{summary}");
        assert!(summary.contains("conserves nothing by construction"), "{summary}");
    }

    /// An observer law checks and compiles, and is not a force: naming one in `force:` is
    /// refused rather than run as one. One that nothing observes draws an
    /// observer-specific `W0316`, not the force-oriented one.
    #[test]
    fn an_observer_law_is_not_a_force() {
        let observer = "observer kinetic(a: particle) -> joule {
            return 0.5 * a.mass * dot(a.velocity, a.velocity);
        }";
        let model = |force: &str| {
            format!(
                "project p {{ {observer}
                   particles s {{ count: 1; mass: 1 kilogram; {force} }}
                   solve dynamics(s) with velocity_verlet(dt=0.01 second);
                 }}"
            )
        };
        let (_, diagnostics, file) = build(&model(""));
        let text = diagnostics.render(&file);
        assert_eq!(diagnostics.codes(), ["W0316"], "{text}");
        assert!(
            text.contains("nothing observes it") && !text.contains("force:") && !text.contains("not implemented yet"),
            "{text}"
        );
        let (_, codes) = err(&model("force: kinetic;"));
        assert_eq!(codes, ["E0420"]);

        // A declared constant named `temperature` is the model's own, as it is everywhere.
        let (_, diagnostics, _) = build(
            "project p { let temperature = 300 kelvin;
               observer o(a: particle) -> kelvin { return temperature; } }",
        );
        assert!(!diagnostics.has_errors(), "{:?}", diagnostics.codes());
    }

    const KINETIC: &str = "observer kinetic(a: particle) -> joule {
        return 0.5 * a.mass * dot(a.velocity, a.velocity);
    }";

    /// A project with `KINETIC`, a solved set `s` of `count` particles and `tail`.
    fn observed(count: usize, tail: &str) -> String {
        format!(
            "project p {{ duration: 1 second; {KINETIC}
               grid g {{ size: [4, 4]; extent: [1 meter, 1 meter]; }}
               field t on g = 300 kelvin {{ diffusivity: 1e-4 meter^2/second; }}
               particles s {{ count: {count}; speed: 1 meter/second; mass: 1 kilogram; }}
               solve dynamics(s) with velocity_verlet(dt=0.01 second);
               solve diffusion(t) with crank_nicolson(dt=0.05 second);
               {tail} }}"
        )
    }

    /// The rendered diagnostics and the error codes alone: the laws these models declare
    /// and do not observe draw warnings that are not the point.
    fn errors(source: &str) -> (String, Vec<String>) {
        let (text, codes) = err(source);
        (text, codes.into_iter().filter(|code| code.starts_with('E')).collect())
    }

    /// `observe sum(law(set))` lowers to a law observer with its own row, unit, cadence
    /// and bound parameters, and uses the law, so it is no longer unused.
    #[test]
    fn an_observer_law_is_reduced_at_its_use_site() {
        let source = observed(
            4,
            "observer right_half(a: particle) -> 1 {
                 param x0: meter = 0 meter;
                 return if a.position.x > x0 { 1 } else { 0 };
             }
             observe sum(kinetic(s)) every 0.1 second;
             observe mean(right_half(s, x0 = 2 meter));
             observe s;",
        );
        let (compiled, diagnostics, file) = build(&source);
        let rendered = diagnostics.render(&file);
        assert!(diagnostics.is_empty(), "{rendered}");
        let compiled = compiled.unwrap();
        let [kinetic, right, plain] = &compiled.model.observers[..] else { panic!("three observers") };
        let law = kinetic.law.as_ref().expect("a law observer");
        assert_eq!((law.row.as_str(), law.unit.as_str(), law.domain.as_str()), ("s.kinetic.sum", "J", "s"));
        assert_eq!(law.reduction, lattice_ir::Reduction::Sum);
        assert_eq!(kinetic.interval, Some(0.1));
        let law = right.law.as_ref().unwrap();
        assert_eq!((law.row.as_str(), law.unit.as_str()), ("s.right_half.mean", "1"));
        assert_eq!(law.params, [2.0]);
        assert_eq!(right.interval, None);
        assert!(plain.law.is_none(), "the plain form filters nothing");
        // Attached in source order, one per law form.
        let names: Vec<String> = compiled.observers.iter().map(|o| o.contract().name.to_string()).collect();
        assert_eq!(names, ["s.kinetic.sum", "s.right_half.mean"]);
        assert!(compiled.model.report().contains("records `s.kinetic.sum` in J"));
    }

    /// A row's unit is the law's: `J` for an energy declared as one, and the coherent SI
    /// form for anything else, including a torque, which has an energy's dimension.
    #[test]
    fn a_row_is_labelled_with_the_unit_the_law_declared() {
        let unit_of = |law: &str, name: &str| {
            let compiled = ok(&observed(2, &format!("{law} observe sum({name}(s));")));
            compiled.model.observers[0].law.as_ref().unwrap().unit.clone()
        };
        let torque = "observer torque(a: particle) -> newton * meter { return a.mass * 1 meter / second^2 * a.position.x; }";
        assert_eq!(unit_of(torque, "torque"), "kg·m^2/s^2");
        let energy = "observer work(a: particle) -> joule { return a.mass * 1 meter^2 / second^2; }";
        assert_eq!(unit_of(energy, "work"), "J");
        let speed = "observer speed(a: particle) -> meter / second { return length(a.velocity); }";
        assert_eq!(unit_of(speed, "speed"), "m/s");
    }

    /// A pinned particle (infinite mass) makes `0.5 m v²` NaN; the guard shown in
    /// docs/language.md keeps it finite, because only the arm taken counts.
    #[test]
    fn a_guard_on_the_mass_keeps_a_pinned_particle_finite() {
        use lattice_domain_particle::user::layout;
        let program_of = |law: &str, name: &str| {
            let compiled = ok(&observed(2, &format!("{law} observe sum({name}(s));")));
            compiled.model.observers[0].law.as_ref().unwrap().program.clone()
        };
        let mut inputs = vec![0.0; layout::BODY_INPUTS as usize];
        inputs[layout::MASS as usize] = f64::INFINITY;
        let bare = program_of(
            "observer bare(a: particle) -> joule { return 0.5 * a.mass * dot(a.velocity, a.velocity); }",
            "bare",
        );
        assert!(bare.eval_to_vec(&inputs, &[])[0].is_nan());
        let guarded = program_of(
            "observer guarded(a: particle) -> joule {
               return if a.mass < 1e300 kilogram { 0.5 * a.mass * dot(a.velocity, a.velocity) } else { 0 joule };
             }",
            "guarded",
        );
        assert_eq!(guarded.eval_to_vec(&inputs, &[])[0].to_bits(), 0.0f64.to_bits());
        inputs[layout::MASS as usize] = 2.0;
        inputs[layout::VX as usize] = 3.0;
        assert_eq!(guarded.eval_to_vec(&inputs, &[])[0], 9.0);
    }

    /// The codes `observe` raises that are not its own: an argument that is not a law call
    /// or a law that does not exist is `E0202`, and the table in docs/language.md says so.
    #[test]
    fn an_observe_that_reduces_no_law_is_a_resolution_error() {
        for tail in ["observe sum(s);", "observe sum(nolaw(s));"] {
            let (text, codes) = errors(&observed(2, tail));
            assert_eq!(codes, ["E0202"], "{tail}: {text}");
        }
    }

    /// `E0419`: the outer call must be one of the four reductions.
    #[test]
    fn an_observe_without_a_reduction_is_refused() {
        for tail in ["observe average(kinetic(s));", "observe kinetic(s);"] {
            let (text, codes) = errors(&observed(2, tail));
            assert_eq!(codes, ["E0419"], "{tail}: {text}");
            assert!(text.contains("sum") && text.contains("mean"), "{text}");
        }
    }

    /// `E0420`: the inner callee must be an observer law.
    #[test]
    fn an_observe_of_a_force_is_refused() {
        let (text, codes) = errors(&observed(
            2,
            "force pull(a: particle) -> vec2<newton> { return vec2(0 newton, 0 newton); }
             observe sum(pull(s));",
        ));
        assert_eq!(codes, ["E0420"], "{text}");
        assert!(text.contains("`pull` is a force"), "{text}");
    }

    /// `E0421`: the set must be a solved particle domain, and a near miss is suggested.
    #[test]
    fn an_observe_over_something_that_is_not_a_particle_domain_is_refused() {
        for (set, expect) in [
            ("t", "it is a field"),
            ("g", "it is a grid"),
            ("ss", "did you mean `s`?"),
            ("nothing_like_it", "nothing is called that"),
        ] {
            let (text, codes) = errors(&observed(2, &format!("observe sum(kinetic({set}));")));
            assert_eq!(codes, ["E0421"], "{set}: {text}");
            assert!(text.contains(expect), "{set}: {text}");
        }
        let (text, codes) = errors(&observed(2, "observe sum(kinetic());"));
        assert_eq!(codes, ["E0421"], "{text}");
        // A set that was declared and never solved has no domain.
        let (text, codes) = err(
            "project p { observer kinetic(a: particle) -> joule { return 0.5 * a.mass * dot(a.velocity, a.velocity); }
               particles s { count: 2; mass: 1 kilogram; }
               observe sum(kinetic(s)); }",
        );
        assert!(codes.contains(&"E0421".to_string()) && text.contains("never solved"), "{codes:?} {text}");
    }

    /// Arguments bind as they do for a force: unknown, missing, unnamed and duplicated
    /// parameters are the same errors.
    #[test]
    fn observer_parameters_bind_like_a_forces() {
        let law = "observer above(a: particle) -> 1 { param x0: meter; return if a.position.x > x0 { 1 } else { 0 }; }";
        let cases = [
            ("above(s)", "E0203"),
            ("above(s, y0 = 1 meter)", "E0204"),
            ("above(s, 1 meter)", "E0204"),
            ("above(s, x0 = 1 meter, x0 = 2 meter)", "E0201"),
            ("above(s, x0 = 1 second)", "E0400"),
        ];
        for (call, code) in cases {
            let (text, codes) = errors(&observed(2, &format!("{law} observe sum({call});")));
            assert!(codes.contains(&code.to_string()), "{call}: {codes:?} {text}");
        }
        let compiled = ok(&observed(2, &format!("{law} observe sum(above(s, x0 = 1 meter));")));
        assert_eq!(compiled.model.observers[0].law.as_ref().unwrap().params, [1.0]);
    }

    /// `E0413` applies where an observer law is lowered, which is its `observe`.
    #[test]
    fn an_observer_law_too_large_for_the_register_file_is_refused() {
        // Distinct terms: the builder shares identical sub-expressions.
        let terms = (1..=300).map(|k| format!("{k} * a.mass * a.position.x")).collect::<Vec<_>>().join(" + ");
        let law = format!("observer big(a: particle) -> kilogram * meter {{ return {terms}; }}");
        let (text, codes) = errors(&observed(2, &format!("{law} observe sum(big(s));")));
        assert_eq!(codes, ["E0413"], "{text}");
    }

    #[test]
    fn the_same_row_cannot_be_observed_twice() {
        let (text, codes) = errors(&observed(
            2,
            "observe sum(kinetic(s)) every 0.1 second; observe sum(kinetic(s)) every 0.2 second;",
        ));
        assert_eq!(codes, ["E0201"], "{text}");
        // Another reduction of the same law is another row.
        ok(&observed(2, "observe sum(kinetic(s)); observe max(kinetic(s));"));
        let (text, codes) = errors(&observed(2, "observe sum(kinetic(s)) every 0 second;"));
        assert_eq!(codes, ["E0405"], "{text}");
    }

    /// `W0320`: a plain `observe` target must name a domain, a species or field, or an
    /// observation; a near miss is suggested. `W0321`: `mean`, `min` and `max` of a set
    /// that starts empty are NaN, which halts a run.
    #[test]
    fn plain_targets_are_resolved_and_empty_sets_are_warned_about() {
        for target in ["s", "t", "total_energy", "s.kinetic_energy", "kinetic_energy"] {
            let (_, diagnostics, file) = build(&observed(2, &format!("observe {target};")));
            assert_eq!(diagnostics.codes(), ["W0316"], "{target}: {}", diagnostics.render(&file));
        }
        let (_, diagnostics, file) = build(&observed(2, "observe ss every 1 second; observe kinetic_energi;"));
        let text = diagnostics.render(&file);
        assert_eq!(diagnostics.codes(), ["W0316", "W0320", "W0320"], "{text}");
        assert!(text.contains("observe target `ss` names nothing") && text.contains("did you mean `s`?"), "{text}");
        assert!(text.contains("did you mean `kinetic_energy`?"), "{text}");

        for (reduction, warns) in [("sum", false), ("mean", true), ("min", true), ("max", true)] {
            let (compiled, diagnostics, file) = build(&observed(0, &format!("observe {reduction}(kinetic(s));")));
            let text = diagnostics.render(&file);
            assert!(!diagnostics.has_errors(), "{text}");
            assert_eq!(diagnostics.codes().contains(&"W0321"), warns, "{reduction}: {text}");
            assert!(compiled.is_some());
        }
    }

    /// A rate that reads `temperature` is the state, and so a rate law, which must come
    /// out in mol/(m² s) — unless `temperature` is a constant, when the rate is one too.
    #[test]
    fn a_constant_named_temperature_is_not_the_state() {
        let mixture = |extra: &str, rate: &str| {
            format!(
                "project p {{
                   {extra}
                   grid g {{ size: [4, 4]; extent: [1 meter, 1 meter]; }}
                   species A on g = 1 mole / meter^2 {{ diffusivity: 0 meter^2 / second; }}
                   species B on g = 0 mole / meter^2 {{ diffusivity: 0 meter^2 / second; }}
                   domain chemistry mix {{ grid: g; }}
                   reaction r {{ reactants: A; products: B; rate: {rate}; }}
                   solve reactions(mix) with strang(dt=0.1 second);
                   duration: 1 second;
                 }}"
            )
        };
        // A law, and 1/s is a rate constant's unit, not a rate's.
        let (text, codes) = err(&mixture("", "temperature / (300 kelvin second)"));
        assert!(codes.contains(&"E0400".to_string()) && text.contains("not a reaction rate"), "{text}");
        ok(&mixture("", "temperature / (300 kelvin second) * c(A)"));
        // A constant: a first-order rate constant, as before M6.1c.
        ok(&mixture("let temperature = 300 kelvin;", "temperature / (300 kelvin second)"));
        // And a law that reads concentrations sees the constant too, and is told so.
        let (_, diagnostics, _) = build(&mixture("let temperature = 300 kelvin;", "temperature / (300 kelvin second) * c(A)"));
        assert!(!diagnostics.has_errors() && diagnostics.codes().contains(&"W0319"), "{:?}", diagnostics.codes());
    }

    // --- spec §8.3's reaction form (M6.1c) ---------------------------------------

    /// A mixture of `A`, `B` and `C` on a 4×4 grid, `A` and `B` starting at 1 mol/m² and
    /// `C` empty, holding one reaction with the given body.
    fn reaction_model(extra: &str, body: &str) -> String {
        format!(
            "project p {{
               duration: 1 second;
               {extra}
               grid g {{ size: [4, 4]; extent: [1 meter, 1 meter]; }}
               species A on g = 1 mole / meter^2 {{ formula: H; diffusivity: 0 meter^2 / second; }}
               species B on g = 1 mole / meter^2 {{ formula: H; diffusivity: 0 meter^2 / second; }}
               species C on g = 0 mole / meter^2 {{ formula: H2; diffusivity: 0 meter^2 / second; }}
               domain chemistry mix {{ grid: g; }}
               reaction r {{ {body} }}
               solve reactions(mix) with strang(dt=0.1 second);
             }}"
        )
    }

    fn reaction_codes(extra: &str, body: &str) -> (String, Vec<String>) {
        let (_, diagnostics, file) = build(&reaction_model(extra, body));
        (diagnostics.render(&file), diagnostics.codes().into_iter().map(String::from).collect())
    }

    /// Spec §8.3's three settings compile, and `heat_release` is `enthalpy` negated.
    #[test]
    fn the_spec_reaction_form_compiles() {
        let compiled = ok(&reaction_model(
            "let k = 2 meter^2 / (mole second);",
            "stoichiometry: A + B -> C; rate: k * c(A) * c(B); heat_release: 57.3 kilojoule / mole;",
        ));
        let report = compiled.model.report();
        assert!(report.contains("A + B -> C, rate law `k * c(A) * c(B)`"), "{report}");
        assert!(report.contains("releases 5.7300e4 J/mol"), "{report}");
        assert!(report.contains("reads A, B and not the temperature"), "{report}");

        // `heat_release: x` and `enthalpy: -x` describe the same reaction.
        let enthalpy = ok(&reaction_model(
            "",
            "reactants: A + B; products: C; rate: 2 meter^2 / (mole second); enthalpy: -57.3 kilojoule / mole;",
        ));
        assert!(enthalpy.model.report().contains("releases 5.7300e4 J/mol"));
    }

    /// The §8.3 settings exclude the ones that say the same thing another way, and a
    /// rate law excludes the settings that modify a rate constant.
    #[test]
    fn the_spec_reaction_form_refuses_double_statements() {
        for body in [
            "stoichiometry: A + B -> C; reactants: A + B; rate: 1 meter^2 / (mole second);",
            "stoichiometry: A + B -> C; rate: 1 meter^2 / (mole second); enthalpy: -1 joule / mole; heat_release: 1 joule / mole;",
            "stoichiometry: A + B -> C; rate: 1 meter^2 / (mole second) * c(A) * c(B); reverse_rate: 1 / second;",
            "stoichiometry: A + B -> C; rate: 1 meter^2 / (mole second) * c(A) * c(B); activation_energy: 1 joule / mole;",
        ] {
            let (text, codes) = reaction_codes("", body);
            assert!(codes.contains(&"E0209".to_string()), "{body}\n{text}");
        }
        let (text, codes) = reaction_codes("", "stoichiometry: A + B; rate: 1 meter^2 / (mole second);");
        assert!(codes.contains(&"E0401".to_string()) && text.contains("reactants -> products"), "{text}");
    }

    /// What a rate law may read, and what it is told when it reaches for something else.
    #[test]
    fn a_rate_law_reads_concentrations_and_temperature() {
        let k = "let k = 1 meter^2 / (mole second);";
        for (body, code, phrase) in [
            ("stoichiometry: A + B -> C; rate: k * c(A) * c(D);", "E0202", "its species are A, B, C"),
            ("stoichiometry: A + B -> C; rate: k * A * c(B);", "E0412", "its concentration is `c(A)`"),
            ("stoichiometry: A + B -> C; rate: k * A.concentration * c(B);", "E0412", "`c(A)`"),
            ("stoichiometry: A + B -> C; rate: k * c(A);", "E0400", "not a reaction rate"),
            ("stoichiometry: A + B -> C; rate: k * c(A) * c(B) * arrhenius(2 joule);", "E0400", "per mole"),
            ("stoichiometry: A + B -> C; rate: k * c(A) * c(B) * distance(A, B);", "E0412", "a rate law has none"),
        ] {
            let (text, codes) = reaction_codes(k, body);
            assert!(codes.contains(&code.to_string()) && text.contains(phrase), "{body}: expected {code}\n{text}");
        }
        // Temperature, directly and through `arrhenius`, and every helper a law has.
        for rate in [
            "k * arrhenius(40 kilojoule / mole) * c(A) * c(B)",
            "k * c(A) * c(B) * temperature / (300 kelvin)",
            "k * max(c(A), 0 mole / meter^2) * c(B) * exp(-(1 joule / mole) / (8.314 joule / (mole kelvin) * temperature))",
        ] {
            let (text, codes) = reaction_codes(k, &format!("stoichiometry: A + B -> C; rate: {rate};"));
            assert!(!codes.iter().any(|c| c.starts_with('E')), "{rate}\n{text}");
        }
        // `c` belongs to rate laws, not to forces.
        let (text, codes) = err(
            "project p { force f(a: particle) -> vec2<newton> { return c(A) * vec2(1 newton, 0 newton); } }",
        );
        assert!(codes.contains(&"E0413".to_string()) && text.contains("rate law"), "{text}");
    }

    /// A law that keeps consuming a species that has run out is warned about; mass
    /// action, and a law shaped like it, is not.
    #[test]
    fn a_rate_law_that_can_go_negative_is_warned_about() {
        let k = "let k = 1 meter^2 / (mole second); let v = 1 mole / (meter^2 second);";
        let warned = |rate: &str| {
            let (text, codes) = reaction_codes(k, &format!("stoichiometry: A + B -> C; rate: {rate};"));
            assert!(!codes.iter().any(|c| c.starts_with('E')), "{rate}\n{text}");
            codes.contains(&"W0318".to_string())
        };
        assert!(warned("v + 0 * k * c(A) * c(B)"), "a zeroth-order rate consumes reactants that are gone");
        assert!(warned("k * c(A) * (1 mole / meter^2)"), "it does not vanish with B");
        assert!(warned("-k * c(A) * c(B)"), "it runs backwards and consumes C, which starts empty");
        assert!(!warned("k * c(A) * c(B)"));
        // Reversible inside the law: each direction vanishes with what it consumes.
        assert!(!warned("k * c(A) * c(B) - 1 / second * c(C)"));
        // Michaelis–Menten in A, first order in B.
        assert!(!warned("v * c(A) / (0.5 mole / meter^2 + c(A)) * c(B) / (1 mole / meter^2)"));

        // A constant in the unit of a rate is not taken for a zeroth-order law: a
        // constant `rate:` is a rate constant, and the message says which was meant.
        let (text, codes) = reaction_codes(k, "stoichiometry: A + B -> C; rate: v;");
        assert!(codes.contains(&"E0400".to_string()) && text.contains("a constant `rate:` is a rate constant"), "{text}");
    }

    /// A rate law that is not finite where the mixture starts is refused.
    #[test]
    fn a_rate_law_that_is_not_finite_is_refused() {
        let (text, codes) = reaction_codes(
            "let k = 1 meter^2 / (mole second);",
            "stoichiometry: A + B -> C; rate: k * c(A) * c(B) * (1 mole / meter^2) / c(C);",
        );
        // C starts empty, so the law divides by zero on the first step: refused, not
        // warned about. (A blind review found this once passed with only W0318.)
        assert!(codes.contains(&"E0405".to_string()) && text.contains("not finite"), "{text}");
        let (text, codes) = reaction_codes(
            "let k = 1 meter^2 / (mole second); let none = 0 mole / meter^2;",
            "stoichiometry: A + B -> C; rate: k * c(A) * c(B) * (1 mole / meter^2) / none;",
        );
        assert!(codes.contains(&"E0405".to_string()) && text.contains("not finite"), "{text}");
    }

    /// A mixture's `temperature:` reaches the domain; it was once accepted and dropped.
    #[test]
    fn a_mixture_temperature_is_read() {
        let compiled = ok(&reaction_model("", "stoichiometry: A + B -> C; rate: 1 meter^2 / (mole second);")
            .replace("domain chemistry mix { grid: g; }", "domain chemistry mix { grid: g; temperature: 350 kelvin; }"));
        let summary = &compiled.model.domains[0].summary;
        assert!(summary.contains("at 350 K"), "{summary}");
        let default = ok(&reaction_model("", "stoichiometry: A + B -> C; rate: 1 meter^2 / (mole second);"));
        assert!(default.model.domains[0].summary.contains("at 298.15 K"), "{}", default.model.domains[0].summary);
        let source = reaction_model("", "stoichiometry: A + B -> C; rate: 1 meter^2 / (mole second);")
            .replace("domain chemistry mix { grid: g; }", "domain chemistry mix { grid: g; temperature: -4 kelvin; }");
        let (text, codes) = err(&source);
        assert!(codes.contains(&"E0405".to_string()), "{text}");
    }

    // --- user laws at their use site (M6.1b) -------------------------------------

    /// Two particles one metre apart in a set using `law` through `force_line`.
    fn pair_model(law: &str, force_line: &str, charge: &str) -> String {
        format!(
            "project p {{
               duration: 1 second;
               {law}
               particles s {{
                 count: 2; region: [4 meter, 4 meter]; spacing: 1 meter; mass: 1 kilogram;
                 {charge}
                 force: {force_line};
               }}
               solve dynamics(s) with velocity_verlet(dt=0.001 second);
             }}"
        )
    }

    fn warnings(source: &str) -> Vec<String> {
        let (_, diagnostics, file) = build(source);
        assert!(!diagnostics.has_errors(), "{}", diagnostics.render(&file));
        diagnostics.codes().into_iter().map(String::from).collect()
    }

    const SPEC_SPRING: &str = "force spring(a: particle, b: particle) -> vec2<newton> {
        param rest_length: meter;
        param stiffness: newton / meter;
        param damping: kilogram / second;
        let dx = minimum_image(b.position - a.position);
        let extension = length(dx) - rest_length;
        return stiffness * extension * normalize(dx)
             - damping * dot(b.velocity - a.velocity, normalize(dx)) * normalize(dx);
    }";

    const SPRING_USE: &str =
        "spring(rest_length=1 meter, stiffness=40 newton/meter, damping=0.2 kilogram/second, cutoff=1.9 meter)";

    /// Spec §8.3's spring as written feeds energy in, and is told so; with the sign that
    /// makes it a damper it is not. Either way a force does not limit the step.
    #[test]
    fn the_spec_spring_draws_the_sign_warning_and_the_corrected_one_does_not() {
        let spec = warnings(&pair_model(SPEC_SPRING, SPRING_USE, ""));
        assert!(spec.contains(&"W0317".to_string()) && spec.contains(&"W0314".to_string()), "{spec:?}");
        let corrected = warnings(&pair_model(&SPEC_SPRING.replace("- damping", "+ damping"), SPRING_USE, ""));
        assert!(!corrected.contains(&"W0317".to_string()), "{corrected:?}");
    }

    /// A potential with a well limits the step; an untruncated one with energy at its
    /// cutoff is warned about; an unused law is pointed out.
    #[test]
    fn potential_warnings_fire_when_they_should() {
        let well = "potential well(a: particle, b: particle) -> joule {
            param k: newton / meter;
            return 0.5 * k * (distance(a, b) - 1 meter)^2;
        }";
        let shifted = warnings(&pair_model(well, "well(k=10 newton/meter, cutoff=2 meter)", ""));
        assert!(!shifted.contains(&"W0314".to_string()), "a well sets the step: {shifted:?}");
        assert!(!shifted.contains(&"W0315".to_string()), "energy_shift is the default: {shifted:?}");
        let bare = warnings(&pair_model(well, "well(k=10 newton/meter, cutoff=2 meter, truncation=none)", ""));
        assert!(bare.contains(&"W0315".to_string()), "{bare:?}");
        let (_, diagnostics, _) = build(&format!("project p {{ dimensions: 2; {well} }}"));
        assert_eq!(diagnostics.codes(), ["W0316"]);
    }

    /// A potential that reads charges is shifted at its cutoff pair by pair, with each
    /// pair's own charges. Two opposite charges at 1 m with a 2 m cutoff:
    /// `U − U(r_c) = k q_a q_b (1/r − 1/r_c)`.
    #[test]
    fn a_charged_potential_is_shifted_pair_by_pair() {
        let law = "potential coulomb_like(a: particle, b: particle) -> joule {
            param k: newton * meter^2 / coulomb^2;
            return k * a.charge * b.charge / distance(a, b);
        }";
        let compiled = ok(&pair_model(law, "coulomb_like(k=2 newton meter^2 / coulomb^2, cutoff=2 meter)", "charge: alternating(3 coulomb);"));
        assert!(compiled.model.domains[0].summary.contains("shifted pair by pair"), "{}", compiled.model.domains[0].summary);
        let mut observations = lattice_ir::Observations::new();
        compiled.domains[0].observe(&mut observations);
        let energy = observations.value("s.potential_energy").unwrap();
        let expected = 2.0 * 3.0 * -3.0 * (1.0 / 1.0 - 1.0 / 2.0);
        assert!((energy - expected).abs() < 1e-12, "{energy} against {expected}");
    }

    /// Both truncations of a user potential give the built-in's pair energy: the two
    /// particles of `pair_model` sit 1 m apart, inside a 2.5 m cutoff. The formulas
    /// differ (`σ/r` against `1/r²`), so the agreement is to their rounding, 64 ε_mach of
    /// the energy's terms, as the validation suite derives.
    #[test]
    fn user_truncations_match_the_builtin_energies() {
        use lattice_domain_particle::{LennardJones, Truncation};
        let law = "potential lj(a: particle, b: particle) -> joule {
            param epsilon: joule;
            param sigma: meter;
            let s6 = (sigma / distance(a, b))^6;
            return 4 * epsilon * (s6^2 - s6);
        }";
        let (epsilon, sigma, cutoff, r) = (1.0, 0.95, 2.5, 1.0);
        for (written, truncation) in [("energy_shift", Truncation::EnergyShift), ("force_shift", Truncation::ForceShift)] {
            let compiled = ok(&pair_model(
                law,
                &format!("lj(epsilon={epsilon} joule, sigma={sigma} meter, cutoff={cutoff} meter, truncation={written})"),
                "",
            ));
            let mut observations = lattice_ir::Observations::new();
            compiled.domains[0].observe(&mut observations);
            let user = observations.value("s.potential_energy").unwrap();
            let builtin = LennardJones::with_truncation(epsilon, sigma, cutoff, truncation).pair_energy(r);
            let s6 = (sigma / r).powi(6);
            let terms = 4.0 * epsilon * (s6 * s6 + s6);
            assert!((user - builtin).abs() <= 64.0 * f64::EPSILON * terms, "{written}: {user} against {builtin}");
        }
    }

    /// A potential that divides by a member the set gives as zero is refused, rather
    /// than shifting every pair by an infinity.
    #[test]
    fn a_potential_infinite_at_its_cutoff_is_refused() {
        let law = "potential bad(a: particle, b: particle) -> joule {
            return 1 joule * coulomb^2 / (a.charge * b.charge) * meter / distance(a, b);
        }";
        let (_, codes) = err(&pair_model(law, "bad(cutoff=2 meter)", ""));
        assert!(codes.contains(&"E0405".to_string()), "{codes:?}");
    }
}
