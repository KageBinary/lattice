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
pub mod molecular;
pub mod quantum;
pub mod rigid;

pub use compile::{compile, Compiled};
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
}
