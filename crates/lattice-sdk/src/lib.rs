//! The public surface for code written outside the engine (spec M6.2 §4.4).
//!
//! A plugin is a Rust crate that links this one, implements [`Observer`], and attaches
//! itself to a compiled [`Simulation`]. It does not need to name any other engine crate:
//! everything it touches is re-exported here, so the engine can move code between its
//! crates without breaking it. `plugins/vacf` in this repository is the worked example,
//! and builds against an unmodified engine.
//!
//! # What is supported
//!
//! - [`Observer`], [`ObserverContract`] and the types it is built from, [`ObserveContext`],
//!   [`Observations`] and [`Curve`]: the trait, and what it is given and returns.
//! - [`ParticleView`]: read-only slices of a particle domain's state. This is the
//!   supported way to read particles. [`ObserveContext::domain`] can reach a concrete
//!   domain type too, but a plugin that does so couples itself to the engine's internals,
//!   which change without notice.
//! - [`compile_file`], [`Simulation`], [`RunConfig`]: build a model and run it.
//! - [`SDK_VERSION`].
//!
//! There is no stable binary interface and no dynamic loading. A plugin is compiled with
//! the engine version it names, and [`SDK_VERSION`] says which revision of this surface
//! that was.
//!
//! # Panics
//!
//! The runtime catches a panic in an observer, disables the observer and records `E0951`;
//! the run carries on. That holds for panics only, and only while the build uses
//! `panic = "unwind"`, which is Cargo's default. An abort, an infinite loop or a stack
//! overflow in a plugin still ends or hangs the run.

use std::path::Path;

use lattice_domain_particle::ParticleDomain;
use lattice_ir::contract::DomainType;

pub use lattice_compiler::{compile, compile_source, Compiled};
pub use lattice_ir::{
    Backends, Curve, Domain, Observation, ObservationKind, Observations, ObserveContext, Observer,
    ObserverContract, ObserverGap, SideEffects,
};
pub use lattice_runtime::{
    Clock, ObserverFault, ObserverFaultKind, RunConfig, RunOutcome, Simulation, StopReason,
};
pub use lattice_syntax::{Diagnostic, Diagnostics, Severity, SourceFile};

/// The revision of this public surface, `(major, minor)`.
///
/// While the major number is 0 the surface can change between minors; a plugin should
/// compare it against the value it was written for and refuse to run on a mismatch.
pub const SDK_VERSION: (u32, u32) = (0, 1);

/// Compile the model in the file at `path` into a simulation ready to run.
///
/// The model's own observers (its `observe sum(law(set))` statements) are attached, in
/// source order, so a caller's [`Simulation::add_observer`] comes after them and the rows
/// of both are recorded.
///
/// # Errors
///
/// The diagnostics when the file cannot be read, does not compile (every diagnostic, the
/// warnings with the errors), or declares an observer the simulation refuses to attach.
/// Warnings of a model that does compile are not returned: they are not part of the
/// result. To show them, call [`compile_source`] instead.
pub fn compile_file(path: &Path) -> Result<Simulation, Vec<Diagnostic>> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        vec![Diagnostic::error(format!("cannot read `{}`: {error}", path.display()))]
    })?;
    let file = SourceFile::new(path.display().to_string(), text);
    let (compiled, diagnostics) = compile_source(&file);
    let compiled = match compiled {
        Some(compiled) if !diagnostics.has_errors() => compiled,
        _ => {
            let mut found: Vec<Diagnostic> = diagnostics.iter().cloned().collect();
            if !found.iter().any(|d| d.severity == Severity::Error) {
                found.push(Diagnostic::error(format!("`{}` produced no model", path.display())));
            }
            return Err(found);
        }
    };
    let mut simulation = Simulation::coupled(compiled.model, compiled.domains, compiled.coupler);
    simulation
        .add_observers(compiled.observers)
        .map_err(|refused| vec![Diagnostic::error(refused)])?;
    Ok(simulation)
}

/// A read-only view of a particle domain: slices of its state and nothing that mutates.
///
/// The slices are indexed by particle slot, `0..len()`, and are as long as the number
/// of live particles. A slot is not a particle's identity: the engine may spawn,
/// destroy or reorder particles, which moves them between slots. An observer that
/// follows individual particles over time (a velocity autocorrelation, a mean squared
/// displacement) must say that it assumes they do not.
///
/// Positions are in metres, velocities in metres per second, masses in kilograms and
/// charges in coulombs.
#[derive(Clone, Copy, Debug)]
pub struct ParticleView<'a> {
    domain: &'a ParticleDomain,
}

impl<'a> ParticleView<'a> {
    /// The particle domain named `name`, as an observer sees it.
    ///
    /// `None` when there is no such domain, when it is not a particle domain, or when
    /// the observer's contract does not list it in `reads`.
    pub fn from_context(ctx: &'a ObserveContext<'_>, name: &str) -> Option<ParticleView<'a>> {
        ctx.domain::<ParticleDomain>(name).map(|domain| ParticleView { domain })
    }

    /// The same view of a domain found some other way, for example in
    /// [`Simulation::domains`] before a run starts. `None` if it is not a particle domain.
    pub fn from_domain(domain: &'a dyn Domain) -> Option<ParticleView<'a>> {
        DomainType::as_any(domain).downcast_ref::<ParticleDomain>().map(|domain| ParticleView { domain })
    }

    /// The domain's name, as the model wrote it.
    pub fn name(&self) -> &'a str {
        self.domain.name()
    }

    /// The number of live particles.
    pub fn len(&self) -> usize {
        self.domain.store().len()
    }

    /// True when there are no particles.
    pub fn is_empty(&self) -> bool {
        self.domain.store().is_empty()
    }

    /// X positions, metres.
    pub fn pos_x(&self) -> &'a [f64] {
        self.domain.store().pos_x()
    }

    /// Y positions, metres.
    pub fn pos_y(&self) -> &'a [f64] {
        self.domain.store().pos_y()
    }

    /// X velocities, metres per second.
    pub fn vel_x(&self) -> &'a [f64] {
        self.domain.store().vel_x()
    }

    /// Y velocities, metres per second.
    pub fn vel_y(&self) -> &'a [f64] {
        self.domain.store().vel_y()
    }

    /// Masses, kilograms.
    pub fn mass(&self) -> &'a [f64] {
        self.domain.store().mass()
    }

    /// Electric charges, coulombs. Zero unless the model gives one.
    pub fn charge(&self) -> &'a [f64] {
        self.domain.store().charge()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_domain_particle::ParticleSpec;
    use std::path::PathBuf;

    fn gas() -> ParticleDomain {
        let mut domain = ParticleDomain::new("gas", 4);
        for k in 0..3 {
            let k = f64::from(k);
            domain
                .spawn(
                    ParticleSpec::at([k, 2.0 * k])
                        .with_velocity([10.0 + k, -k])
                        .with_mass(3.0 + k)
                        .with_charge(0.5 * k),
                )
                .unwrap();
        }
        domain
    }

    #[test]
    fn the_version_is_0_1() {
        assert_eq!(SDK_VERSION, (0, 1));
    }

    #[test]
    fn a_view_shows_the_domains_state_as_slices() {
        let domain = gas();
        let view = ParticleView::from_domain(&domain).expect("a particle domain");
        assert_eq!(view.name(), "gas");
        assert_eq!((view.len(), view.is_empty()), (3, false));
        assert_eq!(view.pos_x(), [0.0, 1.0, 2.0]);
        assert_eq!(view.pos_y(), [0.0, 2.0, 4.0]);
        assert_eq!(view.vel_x(), [10.0, 11.0, 12.0]);
        assert_eq!(view.vel_y(), [0.0, -1.0, -2.0]);
        assert_eq!(view.mass(), [3.0, 4.0, 5.0]);
        assert_eq!(view.charge(), [0.0, 0.5, 1.0]);
    }

    #[test]
    fn a_view_follows_the_context_reads() {
        let boxed: Vec<Box<dyn Domain>> = vec![Box::new(gas())];
        let observations = Observations::new();
        let ctx = ObserveContext::new(0.0, 0, &boxed, &observations);
        assert_eq!(ParticleView::from_context(&ctx, "gas").map(|v| v.len()), Some(3));
        assert!(ParticleView::from_context(&ctx, "nothing").is_none());
        let reads = [std::borrow::Cow::Borrowed("other")];
        let narrowed = ctx.restricted_to(&reads);
        assert!(
            ParticleView::from_context(&narrowed, "gas").is_none(),
            "a domain outside the contract's `reads` is out of reach"
        );
    }

    /// Write `text` to a file of its own under the temporary directory.
    fn model_file(tag: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("lattice-sdk-{}-{tag}.lattice", std::process::id()));
        std::fs::write(&path, text).unwrap();
        path
    }

    const GAS: &str = "project gas {
  dimensions: 2;
  fidelity: molecular;
  duration: 1 picosecond;

  observer vx(a: particle) -> meter / second {
    return a.velocity.x;
  }

  particles atoms {
    count:   16;
    region:  [4 nanometer, 4 nanometer];
    spacing: 1 nanometer;
    mass:    39.948 dalton;
    temperature: 94 kelvin;
    seed:    1;
    boundary: periodic;
  }

  solve dynamics(atoms) with velocity_verlet(dt=10 femtosecond);

  observe atoms every 0.1 picosecond;
  observe max(vx(atoms)) every 0.1 picosecond;
}
";

    #[test]
    fn compile_file_attaches_the_models_own_observers() {
        let path = model_file("ok", GAS);
        let mut simulation = compile_file(&path).unwrap_or_else(|d| panic!("{d:?}"));
        let outcome = simulation.run(&RunConfig::from_model(simulation.model()));
        std::fs::remove_file(&path).ok();
        assert!(outcome.is_success(), "{}", outcome.stop.describe());
        let last = outcome.artifact.timeline().last().expect("samples");
        assert!(
            last.values.iter().any(|(name, _)| name == "atoms.vx.max"),
            "the model's observer law row is recorded: {:?}",
            last.values.iter().map(|(name, _)| name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn compile_file_reports_a_missing_file() {
        let path = std::env::temp_dir().join("lattice-sdk-no-such-model.lattice");
        let Err(errors) = compile_file(&path) else { panic!("a missing file is an error") };
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("cannot read"), "{}", errors[0].message);
    }

    #[test]
    fn compile_file_returns_the_compilers_diagnostics() {
        let path = model_file("bad", &GAS.replace("velocity_verlet(dt=10 femtosecond)", "no_such_method()"));
        let Err(errors) = compile_file(&path) else { panic!("a model that does not compile is an error") };
        std::fs::remove_file(&path).ok();
        assert!(errors.iter().any(|d| d.severity == Severity::Error), "{errors:?}");
    }
}
