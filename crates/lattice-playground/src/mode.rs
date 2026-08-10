//! What a playground mode is.
//!
//! The shell owns the window, the canvas, the transport controls and the diagnostics
//! panel. A mode owns a simulation and decides what a click means. Adding one is a
//! single file implementing [`Playground`] — the shell needs no changes.
//!
//! # Why the shell drives the clock rather than the mode
//!
//! A mode is asked to advance by a timestep it does not choose. That is deliberate: a
//! sandbox that ran each mode at its own pace would make "pause" and "speed" mean
//! different things in each, and the first thing anyone does in a playground is pause
//! it to look at something. The mode says what step it *wants* through
//! [`Playground::preferred_step`], and the shell honours it unless the user has said
//! otherwise.

use eframe::egui;
use lattice_ir::Observations;
use lattice_viewer::render::WorldView;
use lattice_viewer::Palette;

/// A thing the cursor can do.
///
/// Each mode publishes its own list, and the shell draws them as a palette. A tool is
/// not a mouse *button*: left-drag always means "act with the current tool" and
/// right-click always means "remove", in every mode, because a sandbox where the
/// buttons mean different things per mode is a sandbox nobody can put down and pick up.
#[derive(Clone, PartialEq, Debug)]
pub struct Tool {
    /// The label on the button.
    pub name: String,
    /// One line on what it does, shown under the palette.
    pub hint: String,
}

impl Tool {
    /// A tool with a name and a hint.
    pub fn new(name: impl Into<String>, hint: impl Into<String>) -> Tool {
        Tool { name: name.into(), hint: hint.into() }
    }
}

/// A numeric knob a mode exposes.
///
/// The shell draws these as sliders and writes changes back through
/// [`Playground::set_knob`]. Every one carries its unit, because a slider labelled
/// "gravity 9.8" and one labelled "gravity 9.8 m/s²" are different amounts of
/// information and the second costs nothing.
#[derive(Clone, PartialEq, Debug)]
pub struct Knob {
    /// The label.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Smallest allowed.
    pub min: f64,
    /// Largest allowed.
    pub max: f64,
    /// SI unit, or `""` for a dimensionless one.
    pub unit: String,
    /// True when the range is better swept logarithmically — a rate constant spanning
    /// four decades is unusable on a linear slider.
    pub logarithmic: bool,
}

impl Knob {
    /// A linear knob.
    pub fn new(name: impl Into<String>, value: f64, min: f64, max: f64, unit: impl Into<String>) -> Knob {
        Knob {
            name: name.into(),
            value,
            min,
            max,
            unit: unit.into(),
            logarithmic: false,
        }
    }

    /// A knob whose range spans decades.
    pub fn logarithmic(
        name: impl Into<String>,
        value: f64,
        min: f64,
        max: f64,
        unit: impl Into<String>,
    ) -> Knob {
        Knob { logarithmic: true, ..Knob::new(name, value, min, max, unit) }
    }
}

/// A yes-or-no control a mode exposes.
///
/// Separate from [`Knob`] because a boolean on a slider is a lie about the value space:
/// a control that reads "open left edge 0.3400" invites a reader to wonder what a third
/// of an open edge is, and the answer is that there is no such thing. Anything with two
/// states gets a checkbox.
#[derive(Clone, PartialEq, Debug)]
pub struct Toggle {
    /// The label beside the box.
    pub name: String,
    /// Whether it is on.
    pub value: bool,
    /// One line on what turning it on does, shown on hover.
    pub hint: String,
}

impl Toggle {
    /// A toggle with a name, a state and a hint.
    pub fn new(name: impl Into<String>, value: bool, hint: impl Into<String>) -> Toggle {
        Toggle { name: name.into(), value, hint: hint.into() }
    }
}

/// A one-of-many choice a mode exposes.
///
/// The third control type, after [`Knob`] and [`Toggle`], and it exists for the same reason
/// [`Toggle`] does: a dozen reactions on a slider would be a lie about the value space. There
/// is no reaction halfway between burning magnesium and making ammonia, and a control that
/// implies one is a control that has to be explained away.
#[derive(Clone, PartialEq, Debug)]
pub struct Choice {
    /// The label.
    pub name: String,
    /// What can be picked, in display order.
    pub options: Vec<String>,
    /// Index into [`Choice::options`].
    pub selected: usize,
    /// One line about the current selection, shown underneath.
    pub detail: String,
}

impl Choice {
    /// A choice among `options`, currently on `selected`.
    pub fn new(name: impl Into<String>, options: Vec<String>, selected: usize) -> Choice {
        Choice {
            name: name.into(),
            options,
            selected,
            detail: String::new(),
        }
    }

    /// Attach the line shown under the control.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Choice {
        self.detail = detail.into();
        self
    }
}

/// One property of whatever is currently selected.
///
/// The per-object counterpart of [`Knob`]. A knob is a setting of the *scene* — gravity,
/// the size the next spawn will have — and there is one of it. A field belongs to the one
/// object under the cursor, and there is a different set of them for every selection.
///
/// Some are derived rather than stored: a body's speed and kinetic energy follow from its
/// velocity and mass, so they are shown and not edited. Marking that in the data rather
/// than by convention means the shell can render them differently without knowing what any
/// of them mean.
#[derive(Clone, PartialEq, Debug)]
pub struct Field {
    /// The label.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// SI unit, or `""` for a dimensionless one.
    pub unit: String,
    /// Smallest and largest sensible values, for the drag widget's range.
    pub min: f64,
    /// See [`Field::min`].
    pub max: f64,
    /// False for a quantity computed from the others.
    pub editable: bool,
    /// One line shown on hover, for anything whose meaning is not obvious from the name.
    pub hint: String,
}

impl Field {
    /// An editable field.
    pub fn new(
        name: impl Into<String>,
        value: f64,
        min: f64,
        max: f64,
        unit: impl Into<String>,
    ) -> Field {
        Field {
            name: name.into(),
            value,
            unit: unit.into(),
            min,
            max,
            editable: true,
            hint: String::new(),
        }
    }

    /// A field computed from the others, shown but not editable.
    pub fn derived(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Field {
        Field { editable: false, ..Field::new(name, value, f64::MIN, f64::MAX, unit) }
    }

    /// Attach the hover text.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Field {
        self.hint = hint.into();
        self
    }
}

/// What the mode has selected, and what can be changed about it.
#[derive(Clone, PartialEq, Debug)]
pub struct Inspection {
    /// A heading naming the selection, e.g. `"box · body 4"`.
    pub title: String,
    /// One line on what this object is.
    pub subtitle: String,
    /// Its properties, in the order they should be shown.
    pub fields: Vec<Field>,
}

/// Where the pointer is, and what it is doing.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Pointer {
    /// Cursor position in world coordinates.
    pub world: [f64; 2],
    /// How far it moved since the last frame, in world units per second.
    ///
    /// What a throw needs. Reading the position difference inside a mode would give it
    /// the wrong answer whenever the frame rate changed, and a throw whose strength
    /// depends on the frame rate is the most frustrating possible bug in a sandbox.
    pub velocity: [f64; 2],
    /// True on the frame the primary button went down.
    pub pressed: bool,
    /// True while the primary button is held.
    pub held: bool,
    /// True on the frame the primary button came up.
    pub released: bool,
    /// True on the frame the secondary button went down.
    pub secondary: bool,
}

/// A sandbox the shell can host.
pub trait Playground {
    /// The name on the tab.
    fn name(&self) -> &'static str;

    /// One line under the title on what this mode is for.
    fn description(&self) -> &'static str;

    /// What the cursor can do here.
    fn tools(&self) -> Vec<Tool>;

    /// The knobs to draw as sliders.
    fn knobs(&self) -> Vec<Knob>;

    /// Apply a change to knob `index`.
    fn set_knob(&mut self, index: usize, value: f64);

    /// The dropdowns to draw. Most modes have none.
    fn choices(&self) -> Vec<Choice> {
        Vec::new()
    }

    /// Apply a change to choice `index`, selecting its `option`.
    fn set_choice(&mut self, index: usize, option: usize) {
        let _ = (index, option);
    }

    /// The checkboxes to draw. Most modes have none.
    fn toggles(&self) -> Vec<Toggle> {
        Vec::new()
    }

    /// Apply a change to toggle `index`.
    fn set_toggle(&mut self, index: usize, value: bool) {
        let _ = (index, value);
    }

    /// What is selected right now, if anything, and what can be changed about it.
    ///
    /// Read fresh every frame rather than cached, so the numbers move while the simulation
    /// runs — the point of an inspector on a live scene is watching a value change, not
    /// taking a snapshot of it.
    fn inspection(&self) -> Option<Inspection> {
        None
    }

    /// Apply a change to field `index` of the current selection.
    fn set_field(&mut self, index: usize, value: f64) {
        let _ = (index, value);
    }

    /// Drop the current selection.
    fn clear_selection(&mut self) {}

    /// The timestep this mode wants, in seconds.
    fn preferred_step(&self) -> f64;

    /// Advance by `dt`.
    fn step(&mut self, dt: f64);

    /// Handle the pointer, with `tool` naming the selected tool.
    ///
    /// Called every frame while the cursor is over the canvas, whether or not anything
    /// is pressed — a mode that wants to preview what a click would do needs to know
    /// where the cursor is when it is not clicking.
    fn pointer(&mut self, pointer: Pointer, tool: usize);

    /// Called when the cursor leaves the canvas or the window loses focus.
    ///
    /// A mode holding something must let go. Without this, dragging out of the window
    /// and releasing leaves an object stuck to a cursor that is no longer there.
    fn pointer_left(&mut self);

    /// Draw into `view`.
    fn draw(&self, painter: &egui::Painter, view: WorldView, palette: &Palette);

    /// The world region to show.
    fn bounds(&self) -> ([f64; 2], [f64; 2]);

    /// Publish what a reader should be able to see.
    fn observe(&self, out: &mut Observations);

    /// How many times the reader has reached in and changed a conserved quantity.
    ///
    /// Painting heat onto a plate adds energy; dropping a body adds mass. Neither is a
    /// solver error, but both move the integral the panel is watching — and a panel that
    /// reported "drifting +2e-2" because someone used the brush would be blaming the
    /// solver for the reader's own hand. The shell re-baselines whenever this changes,
    /// so a drift figure always measures an undisturbed stretch of simulation.
    ///
    /// A counter rather than a flag because the shell samples once a frame, and several
    /// disturbances inside one frame still have to be noticed.
    fn disturbances(&self) -> u64 {
        0
    }

    /// Start over.
    fn reset(&mut self);

    /// A line of text under the canvas describing the current state.
    fn status(&self) -> String;
}
