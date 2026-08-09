//! An interactive sandbox over the same solvers the model runner uses.
//!
//! `lattice-view` opens a `.lattice` model somebody wrote and reports whether to
//! believe it. This crate has no model: the scene is built by clicking, and the physics
//! underneath is the same validated code either way. A ball dropped here obeys the
//! contact solver whose elastic collision is measured against theory in the validation
//! suite; a reaction painted here balances its atoms because an unbalanced one would
//! not have compiled.
//!
//! # Modes
//!
//! A playground is not one thing. [`Mode`] picks which sandbox to open, and each is a
//! single file implementing [`mode::Playground`] — the shell needs no changes to gain
//! another.

pub mod app;
pub mod heat;
pub mod mode;
pub mod physics;
pub mod reactions;

pub use app::PlaygroundApp;
pub use mode::{Knob, Playground, Pointer, Toggle, Tool};

/// Which sandbox to open.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Mode {
    /// Rigid bodies: drop shapes, pick them up, throw them.
    #[default]
    Physics,
    /// Reaction-diffusion: paint reactants and watch them react and heat.
    Reactions,
    /// Diffusion alone: paint hot and cold and watch it even out.
    Heat,
}

impl Mode {
    /// Every mode, in the order the tabs appear.
    pub const ALL: [Mode; 3] = [Mode::Physics, Mode::Reactions, Mode::Heat];

    /// The name used on the tab and on the command line.
    pub const fn name(&self) -> &'static str {
        match self {
            Mode::Physics => "physics",
            Mode::Reactions => "reactions",
            Mode::Heat => "heat",
        }
    }

    /// Parse a mode name, or `None`.
    pub fn parse(name: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|mode| mode.name() == name)
    }
}

/// Every sandbox, in the order [`Mode::ALL`] names them.
///
/// One list, so the tabs across the top and the names on the command line cannot drift
/// apart — the shell addresses modes by index, and two hand-maintained orderings would
/// eventually disagree in a way that opens the wrong tab and looks like nothing at all.
pub fn modes() -> Vec<Box<dyn Playground>> {
    vec![
        Box::new(physics::PhysicsPlayground::new()),
        Box::new(reactions::ReactionPlayground::new()),
        Box::new(heat::HeatPlayground::new()),
    ]
}

/// Open the playground, starting on `mode`.
///
/// Blocks until the window closes.
pub fn run(mode: Mode) -> Result<(), String> {
    let modes = modes();
    let start = Mode::ALL.iter().position(|m| *m == mode).unwrap_or(0);

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1240.0, 820.0])
            .with_min_inner_size([760.0, 520.0])
            .with_title("lattice — playground"),
        ..Default::default()
    };

    let mut app = PlaygroundApp::new(modes);
    app.select(start);
    eframe::run_native("lattice — playground", options, Box::new(move |_cc| Ok(Box::new(app))))
        .map_err(|error| format!("cannot open a window: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ir::Observations;

    #[test]
    fn every_mode_name_round_trips() {
        for mode in Mode::ALL {
            assert_eq!(Mode::parse(mode.name()), Some(mode));
        }
        assert_eq!(Mode::parse("Physics"), None, "names are exact");
        assert_eq!(Mode::parse(""), None);
        assert_eq!(Mode::default(), Mode::Physics);
    }

    /// The shell addresses modes by index. If these two lists ever disagree, `--mode
    /// heat` opens the reactions tab and nothing in the output says so.
    #[test]
    fn the_tab_order_matches_the_mode_enum() {
        let built = modes();
        assert_eq!(built.len(), Mode::ALL.len());
        for (index, mode) in Mode::ALL.iter().enumerate() {
            assert_eq!(built[index].name(), mode.name());
        }
    }

    /// One exercise every sandbox has to survive, so a fourth mode cannot be added
    /// without meeting the same bar as the three that exist.
    #[test]
    fn every_mode_survives_the_same_exercise() {
        for mut mode in modes() {
            let name = mode.name();
            assert!(!mode.description().is_empty(), "{name} has no description");
            assert!(!mode.tools().is_empty(), "{name} offers no tools");
            assert!(mode.preferred_step() > 0.0, "{name} wants a non-positive step");

            let (_, extent) = mode.bounds();
            assert!(extent[0] > 0.0 && extent[1] > 0.0, "{name} has an empty view");

            for tool in mode.tools() {
                assert!(!tool.name.is_empty() && !tool.hint.is_empty(), "{name}: bare tool");
            }
            for knob in mode.knobs() {
                assert!(knob.min < knob.max, "{name}: knob {} has an empty range", knob.name);
                assert!(
                    (knob.min..=knob.max).contains(&knob.value),
                    "{name}: knob {} opens at {} outside {}..{}",
                    knob.name,
                    knob.value,
                    knob.min,
                    knob.max
                );
                assert!(
                    !knob.logarithmic || knob.min > 0.0,
                    "{name}: knob {} is logarithmic through zero",
                    knob.name
                );
            }

            // Every knob to both ends and back, every toggle both ways, every tool
            // clicked — in a mode that has just been reset, with no idea what any of it
            // means. Nothing here may panic.
            let dt = mode.preferred_step();
            for index in 0..mode.knobs().len() {
                let knob = mode.knobs()[index].clone();
                for value in [knob.min, knob.max, knob.value] {
                    mode.set_knob(index, value);
                    mode.step(dt);
                }
            }
            for index in 0..mode.toggles().len() {
                for value in [true, false, true] {
                    mode.set_toggle(index, value);
                    mode.step(dt);
                }
                assert!(mode.toggles()[index].value, "{name}: toggle did not stick");
                mode.set_toggle(index, false);
            }
            mode.set_knob(999, 1.0);
            mode.set_toggle(999, true);

            let ([x, y], [w, h]) = mode.bounds();
            for tool in 0..mode.tools().len() + 1 {
                for step in 0..6 {
                    let at = [
                        x + w * (0.2 + 0.12 * f64::from(step)),
                        y + h * (0.3 + 0.08 * f64::from(step)),
                    ];
                    mode.pointer(press(at), tool);
                    mode.pointer(drag(at, [1.0, -1.0]), tool);
                    mode.pointer(lift(at, [3.0, 2.0]), tool);
                    mode.pointer(Pointer { secondary: true, ..press(at) }, tool);
                }
            }
            mode.pointer_left();

            // The inspector, if this mode has one. Whatever the clicking above selected,
            // every field must be self-consistent and every one must survive being written
            // to at both ends of its range — including the derived ones, which the shell
            // does not offer but nothing stops a caller from trying.
            if let Some(inspection) = mode.inspection() {
                assert!(!inspection.title.is_empty(), "{name}: a selection with no title");
                assert!(!inspection.fields.is_empty(), "{name}: a selection with no fields");
                for field in &inspection.fields {
                    assert!(!field.name.is_empty(), "{name}: an unnamed field");
                    assert!(field.value.is_finite(), "{name}: field {} is non-finite", field.name);
                    if field.editable {
                        assert!(
                            field.min < field.max,
                            "{name}: field {} has an empty range",
                            field.name
                        );
                        assert!(
                            (field.min..=field.max).contains(&field.value),
                            "{name}: field {} opens at {} outside {}..{}",
                            field.name,
                            field.value,
                            field.min,
                            field.max
                        );
                    }
                }

                for index in 0..inspection.fields.len() {
                    let field = mode.inspection().expect("still selected").fields[index].clone();
                    for value in [field.min, field.max, field.value] {
                        mode.set_field(index, value);
                        mode.step(dt);
                    }
                }
                mode.set_field(999, 1.0);
                mode.set_field(0, f64::NAN);
                mode.step(dt);

                mode.clear_selection();
                assert!(
                    mode.inspection().is_none(),
                    "{name}: clearing the selection left one behind"
                );
            }

            for _ in 0..200 {
                mode.step(dt);
            }

            let mut observations = Observations::new();
            mode.observe(&mut observations);
            assert!(!observations.is_empty(), "{name} publishes nothing");
            for observation in observations.iter() {
                assert!(
                    observation.value.is_finite(),
                    "{name}: {} went non-finite",
                    observation.name
                );
            }
            assert!(!mode.status().is_empty(), "{name} has no status line");

            mode.reset();
            let mut after = Observations::new();
            mode.observe(&mut after);
            let missing: Vec<String> = observations
                .iter()
                .filter(|o| after.get(&o.name).is_none())
                .map(|o| o.name.to_string())
                .collect();
            assert!(missing.is_empty(), "{name} stops publishing {missing:?} after a reset");
        }
    }

    /// A window that is dragged off screen and back hands the shell a huge elapsed time.
    /// Whatever it does with that, a mode must not be asked for a step it cannot take.
    #[test]
    fn no_mode_is_destabilised_by_its_own_preferred_step() {
        for mut mode in modes() {
            let dt = mode.preferred_step();
            for _ in 0..2_000 {
                mode.step(dt);
            }
            let mut observations = Observations::new();
            mode.observe(&mut observations);
            for observation in observations.iter() {
                assert!(
                    observation.value.is_finite(),
                    "{}: {} went non-finite over 2000 steps",
                    mode.name(),
                    observation.name
                );
            }
        }
    }

    fn press(world: [f64; 2]) -> Pointer {
        Pointer {
            world,
            velocity: [0.0, 0.0],
            pressed: true,
            held: true,
            released: false,
            secondary: false,
        }
    }

    fn drag(world: [f64; 2], velocity: [f64; 2]) -> Pointer {
        Pointer { pressed: false, velocity, ..press(world) }
    }

    fn lift(world: [f64; 2], velocity: [f64; 2]) -> Pointer {
        Pointer { pressed: false, held: false, released: true, velocity, ..press(world) }
    }
}
