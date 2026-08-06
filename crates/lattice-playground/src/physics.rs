//! The rigid-body sandbox: drop things, pick them up, throw them.
//!
//! # How the grab works
//!
//! Not a joint. The obvious implementation attaches a spring joint between the grabbed
//! body and an invisible body at the cursor, and it works — but joints hold *slot
//! indices*, and a sandbox where things are constantly created and destroyed permutes
//! those slots underneath it. Creating and destroying a joint on every click is asking
//! for exactly the class of bug [`RigidDomain::despawn`] had to be careful about.
//!
//! Instead the grab applies a spring force directly, every step, from a [`BodyId`]
//! resolved fresh. A handle survives a despawn and refuses to resolve if the body is
//! gone, so a grabbed object that is deleted underneath the cursor simply stops being
//! held.
//!
//! The spring is **critically damped and scaled to the body's mass**:
//!
//! ```text
//!   k = m ω²      c = 2 m ω
//! ```
//!
//! so a heavy crate and a light ball follow the cursor at the same speed. Without the
//! mass scaling a sandbox feels wrong in a way that is hard to name: light things snap
//! to the cursor and heavy things lag, when what a hand does is the opposite.
//!
//! # Why a throw uses the pointer's velocity and not the body's
//!
//! Releasing a spring-held body leaves it moving at whatever the spring last dragged it
//! to, which lags the cursor by roughly one time constant. Throwing that way feels
//! weak. The release instead sets the body's velocity from the *pointer's*, which is
//! what the hand actually did.

use eframe::egui;
use lattice_domain_rigid2d::{Collider, RigidDomain, Shape, SolverConfig, Vec2};
use lattice_ir::{Arena, BodyId, BodySpec, Domain, Observations, ShapeId, StepContext};
use lattice_viewer::render::WorldView;
use lattice_viewer::{render, Palette, Status};

use crate::mode::{Knob, Playground, Pointer, Toggle, Tool};

/// How fast a grabbed body chases the cursor, rad/s.
///
/// About 1.6 Hz. Faster feels more responsive and starts to fight the contact solver:
/// a body held stiffly against the floor is a constraint the spring and the contacts
/// disagree about, and the visible result is a buzz. This is the fastest that stays
/// quiet at the step below.
const GRAB_FREQUENCY: f64 = 10.0;

/// Largest force a grab may apply, per kilogram of the body it holds, N/kg.
///
/// Without it, grabbing a body and dragging it through a wall produces an unbounded
/// force and a solver fight that ends with something across the room. Capping turns
/// that into a body that slides along the wall and lets go of the cursor, which is
/// what a hand would do.
const GRAB_MAX_ACCELERATION: f64 = 400.0;

/// Half-extent of the arena, m.
const ARENA: [f64; 2] = [6.0, 3.5];

/// The shapes the palette offers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Spawn {
    Box,
    Ball,
    Triangle,
}

impl Spawn {
    const ALL: [Spawn; 3] = [Spawn::Box, Spawn::Ball, Spawn::Triangle];

    fn label(self) -> &'static str {
        match self {
            Spawn::Box => "box",
            Spawn::Ball => "ball",
            Spawn::Triangle => "wedge",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Spawn::Box => "click empty space to drop a box; drag one to throw it",
            Spawn::Ball => "click empty space to drop a ball; drag one to throw it",
            Spawn::Triangle => "click empty space to drop a wedge; drag one to throw it",
        }
    }

    /// The shape, `size` metres across.
    ///
    /// Across, not half-across. [`Shape::rectangle`] and friends take half-extents, and
    /// passing the slider straight through would make a control reading "size 0.25 m"
    /// produce a box half a metre wide. Whatever a slider labelled in metres says should
    /// be what a ruler held to the screen measures.
    fn shape(self, size: f64) -> Shape {
        let half = size / 2.0;
        match self {
            Spawn::Box => Shape::rectangle(half, half).expect("a positive size"),
            Spawn::Ball => Shape::circle(half).expect("a positive size"),
            Spawn::Triangle => Shape::Polygon(
                lattice_domain_rigid2d::ConvexPolygon::new(&[
                    [-half, -half * 0.8],
                    [half, -half * 0.8],
                    [0.0, half],
                ])
                .expect("a valid wedge"),
            ),
        }
    }
}

/// What the cursor is currently holding.
#[derive(Clone, Copy, Debug)]
struct Grab {
    body: BodyId,
    /// Where on the body it was grabbed, in that body's local frame — so a crate picked
    /// up by its corner hangs from its corner rather than snapping to its centre.
    local: Vec2,
}

/// The rigid-body sandbox.
#[derive(Debug)]
pub struct PhysicsPlayground {
    world: RigidDomain,
    arena: Arena,
    /// One registered collider per (shape, size) the palette can make. Shapes are
    /// registered once and shared, which is what the domain's shape table is for.
    palette: Vec<(Spawn, u32, ShapeId)>,
    selected: Spawn,
    size: f64,
    gravity: f64,
    friction: f64,
    restitution: f64,
    grab: Option<Grab>,
    /// The pointer's world velocity, kept so a release can throw with it.
    pointer_velocity: [f64; 2],
    /// Where the cursor is, for drawing the spawn preview.
    hover: Option<[f64; 2]>,
    /// Whether to draw the contact normals.
    ///
    /// Off by default, and that is a change of audience rather than of opinion. In the
    /// model viewer they are on, because someone reading a run wants to see what the
    /// solver is doing. A resting stack of nine boxes has twenty-eight of them, and a
    /// reader who came to throw things at a wall did not ask a question they answer.
    show_contacts: bool,
    /// Bumped whenever a body is added, removed or thrown; see
    /// [`Playground::disturbances`].
    ///
    /// This mode claims nothing is conserved, so nothing is currently re-baselined by
    /// it. It is counted anyway: the day someone adds a zero-gravity mode with closed
    /// walls, the claim becomes real and the counter has to already be right.
    disturbances: u64,
    spawned: usize,
}

impl Default for PhysicsPlayground {
    fn default() -> Self {
        PhysicsPlayground::new()
    }
}

impl PhysicsPlayground {
    /// A sandbox with a floor, two walls, and an opening scene.
    pub fn new() -> PhysicsPlayground {
        let mut playground = PhysicsPlayground {
            world: world(),
            arena: Arena::with_capacity(1 << 16),
            palette: Vec::new(),
            selected: Spawn::Box,
            size: 0.5,
            gravity: 9.806_65,
            friction: 0.4,
            restitution: 0.2,
            grab: None,
            pointer_velocity: [0.0, 0.0],
            hover: None,
            show_contacts: false,
            disturbances: 0,
            spawned: 0,
        };
        playground.rebuild(true);
        playground
    }

    /// Throw the scene away and build it again, with or without the opening stack.
    ///
    /// A whole new [`RigidDomain`], not a `clear_bodies` — the shape table is not part
    /// of what `clear_bodies` clears, and rebuilding the arena into the same world would
    /// register the walls again every time. Four dead colliders per reset is not a
    /// visible bug, but [`refresh_surfaces`](Self::refresh_surfaces) walks that table on
    /// every touch of the friction slider, so it is a leak that gets *felt* before it
    /// gets noticed.
    fn rebuild(&mut self, seeded: bool) {
        self.world = world();
        self.palette.clear();
        self.grab = None;
        self.spawned = 0;
        self.build_arena();
        if seeded {
            self.seed();
        }
        self.refresh_surfaces();
        self.world.refresh_render_tables();
    }

    /// A stack and two balls, so the first frame is a scene rather than an empty box.
    ///
    /// An empty canvas with a hint line under it asks the reader to take on faith that
    /// clicking will do something. A pyramid already resting on the floor has answered
    /// the question before it was asked: things here have weight, they stack, and the
    /// contact solver is holding them up. [`Playground::reset`] returns to this rather
    /// than to nothing, because "reset" means back to how it started.
    fn seed(&mut self) {
        let brick = self.collider_for(Spawn::Box, 0.5);
        let ball = self.collider_for(Spawn::Ball, 0.6);
        let floor = -ARENA[1] + 0.15;

        // Three rows, each one shorter than the one below and offset by half a brick.
        for row in 0..3 {
            let count = 4 - row;
            for column in 0..count {
                let x = (f64::from(column) - f64::from(count - 1) / 2.0) * 0.52 - 2.2;
                let y = floor + 0.25 + f64::from(row) * 0.5;
                if self.world.spawn_with_density(BodySpec::at([x, y], brick), 800.0).is_some() {
                    self.spawned += 1;
                }
            }
        }

        // Resting on the floor, not hovering above it: the opening frame should be a
        // scene at rest, so the first thing that moves is the first thing the reader did.
        for x in [1.6_f64, 2.9] {
            if self
                .world
                .spawn_with_density(BodySpec::at([x, floor + 0.3], ball), 800.0)
                .is_some()
            {
                self.spawned += 1;
            }
        }
    }

    /// Floor, walls and a ceiling, so nothing can leave.
    ///
    /// An open sandbox looks fine for a few seconds and then quietly becomes an empty
    /// screen, which reads as a broken simulation and is not one.
    fn build_arena(&mut self) {
        let wall = Collider::new(Shape::rectangle(ARENA[0] + 0.3, 0.15).expect("positive"))
            .with_friction(0.5);
        let side = Collider::new(Shape::rectangle(0.15, ARENA[1]).expect("positive"))
            .with_friction(0.5);
        let floor = self.world.register(wall.clone());
        let post = self.world.register(side);

        self.world.spawn(BodySpec::statik([0.0, -ARENA[1]], floor));
        self.world.spawn(BodySpec::statik([0.0, ARENA[1]], floor));
        self.world.spawn(BodySpec::statik([-ARENA[0], 0.0], post));
        self.world.spawn(BodySpec::statik([ARENA[0], 0.0], post));
        self.world.set_gravity([0.0, -self.gravity]);
    }

    /// How many bodies the user has added.
    fn walls(&self) -> usize {
        4
    }

    /// The collider for the current shape and size, registered on first use.
    ///
    /// Keyed by size to the nearest millimetre, so dragging the size slider does not
    /// register a new shape per pixel of slider travel.
    fn collider_for(&mut self, spawn: Spawn, size: f64) -> ShapeId {
        let key = (size * 1000.0).round() as u32;
        if let Some((_, _, id)) = self
            .palette
            .iter()
            .find(|(kind, stored, _)| *kind == spawn && *stored == key)
        {
            return *id;
        }
        let id = self.world.register(
            Collider::new(spawn.shape(f64::from(key) / 1000.0))
                .with_friction(self.friction)
                .with_restitution(self.restitution),
        );
        self.palette.push((spawn, key, id));
        id
    }

    /// Refresh every registered collider's surface from the sliders.
    ///
    /// Surfaces live on the collider, and colliders are shared, so changing a slider
    /// changes every body wearing that shape — including the ones already on screen.
    /// That is what a reader expects from a slider labelled "friction": the whole scene
    /// responds, not just the next thing they drop.
    fn refresh_surfaces(&mut self) {
        let (friction, restitution) = (self.friction, self.restitution);
        self.world.colliders_mut().iter_mut().for_each(|collider| {
            collider.friction = friction.max(0.0);
            collider.restitution = restitution.clamp(0.0, 1.0);
        });
    }

    /// The topmost body under `world`, if any.
    ///
    /// Searched from the end, so the most recently added body wins an overlap. In a
    /// sandbox the thing you just dropped is the thing you meant to grab.
    fn pick(&self, world: [f64; 2]) -> Option<BodyId> {
        let point = Vec2::from(world);
        // A few pixels' worth of forgiveness, so a small ball is a fair target.
        let slack = 0.04;
        for slot in (0..self.world.bodies().len()).rev() {
            let Some(collider) = self.world.colliders().get(self.world.bodies().shape()[slot].index())
            else {
                continue;
            };
            let transform = lattice_domain_rigid2d::broadphase::transform_of(self.world.bodies(), slot);
            if collider.shape.contains(transform.unapply(point), slack) {
                return self.world.body_at(slot);
            }
        }
        None
    }

    /// Pull the grabbed body toward the cursor.
    fn apply_grab(&mut self, target: [f64; 2]) {
        let Some(grab) = self.grab else { return };
        let Some(slot) = self.world.slot_of(grab.body) else {
            // The body was removed while held. Letting go is the only sensible answer.
            self.grab = None;
            return;
        };

        let bodies = self.world.bodies();
        let mass = bodies.mass()[slot];
        if mass <= 0.0 {
            return;
        }
        let held = Vec2::from(bodies.to_world_point(slot, grab.local.to_array()));
        let velocity = Vec2::from(bodies.point_velocity(slot, grab.local.to_array()));

        // Critically damped, and scaled to the mass so heavy and light follow alike.
        let omega = GRAB_FREQUENCY;
        let displacement = Vec2::from(target) - held;
        let force = displacement * (mass * omega * omega) - velocity * (2.0 * mass * omega);

        // Capped per unit mass, so dragging something into a wall makes it slide rather
        // than fight the contact solver and lose.
        let limit = mass * GRAB_MAX_ACCELERATION;
        let force = if force.length() > limit {
            force.normalize().map_or(Vec2::ZERO, |unit| unit * limit)
        } else {
            force
        };

        self.world
            .bodies_mut()
            .add_force_at(slot, force.to_array(), held.to_array());
    }
}

impl Playground for PhysicsPlayground {
    fn name(&self) -> &'static str {
        "physics"
    }

    fn description(&self) -> &'static str {
        "drop shapes, pick them up, throw them at each other"
    }

    fn tools(&self) -> Vec<Tool> {
        Spawn::ALL.iter().map(|kind| Tool::new(kind.label(), kind.hint())).collect()
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![
            Knob::new("size across", self.size, 0.1, 1.6, "m"),
            Knob::new("gravity", self.gravity, -20.0, 20.0, "m/s^2"),
            Knob::new("friction", self.friction, 0.0, 1.5, ""),
            Knob::new("bounciness", self.restitution, 0.0, 1.0, ""),
        ]
    }

    fn set_knob(&mut self, index: usize, value: f64) {
        match index {
            0 => self.size = value,
            1 => {
                self.gravity = value;
                self.world.set_gravity([0.0, -value]);
            }
            2 => {
                self.friction = value;
                self.refresh_surfaces();
            }
            3 => {
                self.restitution = value;
                self.refresh_surfaces();
            }
            _ => {}
        }
    }

    fn toggles(&self) -> Vec<Toggle> {
        vec![Toggle::new(
            "contact normals",
            self.show_contacts,
            "draw an arrow at every contact, pointing the way the solver is pushing — \
             what to turn on when a stack is jittering or sinking",
        )]
    }

    fn set_toggle(&mut self, index: usize, value: bool) {
        if index == 0 {
            self.show_contacts = value;
        }
    }

    fn preferred_step(&self) -> f64 {
        // Fast enough that a 10 rad/s grab spring is nowhere near its stability limit,
        // and that a thrown ball cannot cross the 0.3 m walls in one step.
        1.0 / 240.0
    }

    fn step(&mut self, dt: f64) {
        let mut ctx = StepContext::new(&mut self.arena);
        self.world.prepare(&mut ctx);
        drop(ctx);

        // After `prepare` (which cleared the accumulators and applied gravity) and
        // before `advance`, which is the only window where an external force lands on
        // this step rather than the next.
        if let Some(target) = self.hover {
            self.apply_grab(target);
        }

        let mut ctx = StepContext::new(&mut self.arena);
        self.world.advance(dt, &mut ctx);
    }

    fn pointer(&mut self, pointer: Pointer, tool: usize) {
        self.hover = Some(pointer.world);
        self.pointer_velocity = pointer.velocity;
        self.selected = Spawn::ALL[tool.min(Spawn::ALL.len() - 1)];

        if pointer.secondary {
            if let Some(body) = self.pick(pointer.world) {
                // Walls are scenery. Deleting one would leave an open sandbox that
                // quietly empties itself, which reads as a broken simulation.
                if self.world.slot_of(body).is_some_and(|slot| slot >= self.walls()) {
                    self.world.despawn(body);
                    self.spawned = self.spawned.saturating_sub(1);
                    self.disturbances += 1;
                    self.world.refresh_render_tables();
                }
            }
            return;
        }

        if pointer.pressed {
            match self.pick(pointer.world) {
                Some(body) => {
                    let slot = self.world.slot_of(body).expect("just picked");
                    // A static body has infinite mass and cannot be dragged; grabbing
                    // one would apply a force that does nothing and feel broken.
                    if !self.world.bodies().is_static(slot) {
                        let local = Vec2::from(
                            self.world.bodies().to_local_point(slot, pointer.world),
                        );
                        self.grab = Some(Grab { body, local });
                    }
                }
                None => {
                    let size = self.size;
                    let selected = self.selected;
                    let shape = self.collider_for(selected, size);
                    if self
                        .world
                        .spawn_with_density(BodySpec::at(pointer.world, shape), 800.0)
                        .is_some()
                    {
                        self.spawned += 1;
                        self.disturbances += 1;
                        // So a click lands visibly even while paused.
                        self.world.refresh_render_tables();
                    }
                }
            }
        }

        if pointer.released {
            if let Some(grab) = self.grab.take()
                && let Some(slot) = self.world.slot_of(grab.body)
            {
                // Throw with the *hand's* velocity, not the body's — a spring-held body
                // lags the cursor, and releasing at its own speed feels weak.
                let spin = self.world.bodies().omega()[slot];
                self.world.bodies_mut().set_velocity(grab.body, self.pointer_velocity, spin);
                self.disturbances += 1;
            }
        }
    }

    fn pointer_left(&mut self) {
        self.grab = None;
        self.hover = None;
    }

    fn draw(&self, painter: &egui::Painter, view: WorldView, palette: &Palette) {
        for channel in self.world.render_channels() {
            match channel {
                lattice_ir::RenderChannel::Bodies { .. } => {
                    render::draw_bodies(painter, view, &channel, palette);
                }
                lattice_ir::RenderChannel::Contacts { .. } if self.show_contacts => {
                    render::draw_contacts(painter, view, &channel, palette);
                }
                _ => {}
            }
        }

        // The line from the cursor to the point it is holding, so it is obvious *where*
        // a body is held — a crate hanging from its corner behaves differently from one
        // held at its centre, and this is what says which is happening.
        if let (Some(grab), Some(target)) = (self.grab, self.hover)
            && let Some(slot) = self.world.slot_of(grab.body)
        {
            let held = self.world.bodies().to_world_point(slot, grab.local.to_array());
            if let (Some(from), Some(to)) = (view.project(held), view.project(target)) {
                painter.line_segment(
                    [from, to],
                    egui::Stroke::new(1.5, palette.status(Status::Warning)),
                );
                painter.circle_filled(from, 4.0, palette.status(Status::Warning));
            }
        }

        // A ghost of what a click would drop, so the size slider means something before
        // anything is on screen.
        if self.grab.is_none()
            && let Some(at) = self.hover
            && self.pick(at).is_none()
        {
            let outline = self.selected.shape(self.size);
            let points: Vec<egui::Pos2> = preview_points(&outline)
                .into_iter()
                .filter_map(|local| view.project([at[0] + local[0], at[1] + local[1]]))
                .collect();
            if points.len() > 2 {
                painter.add(egui::Shape::closed_line(
                    points,
                    egui::Stroke::new(1.0, palette.text_muted),
                ));
            }
        }
    }

    fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        // Fixed, not fitted to the contents. A view that resized as things were added
        // would make everything on screen jump whenever a body was dropped near an edge.
        (
            [-ARENA[0] - 0.4, -ARENA[1] - 0.4],
            [2.0 * (ARENA[0] + 0.4), 2.0 * (ARENA[1] + 0.4)],
        )
    }

    fn observe(&self, out: &mut Observations) {
        self.world.observe(out);
    }

    fn disturbances(&self) -> u64 {
        self.disturbances
    }

    fn reset(&mut self) {
        self.rebuild(true);
    }

    fn status(&self) -> String {
        let report = self.world.last_solve();
        let held = if self.grab.is_some() { ", holding one" } else { "" };
        format!(
            "{} bodies, {} contacts{held}",
            self.spawned, report.points
        )
    }
}

/// An empty world, configured the way this sandbox wants it.
fn world() -> RigidDomain {
    RigidDomain::new("sandbox", 512)
        .with_solver(SolverConfig { velocity_iterations: 8, ..SolverConfig::default() })
}

/// The outline of a shape, for the spawn ghost.
fn preview_points(shape: &Shape) -> Vec<[f64; 2]> {
    match shape {
        Shape::Circle { radius } => (0..24)
            .map(|step| {
                let angle = core::f64::consts::TAU * f64::from(step) / 24.0;
                [radius * angle.cos(), radius * angle.sin()]
            })
            .collect(),
        Shape::Polygon(polygon) => polygon.vertices().iter().map(|v| v.to_array()).collect(),
        Shape::Segment { half_length } => vec![[-half_length, 0.0], [*half_length, 0.0]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pointer hovering at `world` with nothing pressed.
    fn hover(world: [f64; 2]) -> Pointer {
        Pointer {
            world,
            velocity: [0.0, 0.0],
            pressed: false,
            held: false,
            released: false,
            secondary: false,
        }
    }

    fn press(world: [f64; 2]) -> Pointer {
        Pointer { pressed: true, held: true, ..hover(world) }
    }

    fn release(world: [f64; 2], velocity: [f64; 2]) -> Pointer {
        Pointer { released: true, velocity, ..hover(world) }
    }

    /// Press and let go without moving — what dropping something is.
    fn click(playground: &mut PhysicsPlayground, world: [f64; 2]) {
        playground.pointer(press(world), 0);
        playground.pointer(release(world, [0.0, 0.0]), 0);
    }

    /// Advance for `seconds` at the mode's own step.
    fn settle(playground: &mut PhysicsPlayground, seconds: f64) {
        let dt = playground.preferred_step();
        for _ in 0..(seconds / dt).round() as usize {
            playground.step(dt);
        }
    }

    /// A sandbox with the starter scene cleared away, which is what most of these
    /// tests want: one thing happening, against nothing else.
    fn empty() -> PhysicsPlayground {
        let mut playground = PhysicsPlayground::new();
        playground.rebuild(false);
        playground
    }

    /// Bodies the user added, as (position, velocity).
    fn loose(playground: &PhysicsPlayground) -> Vec<([f64; 2], [f64; 2])> {
        let bodies = playground.world.bodies();
        (playground.walls()..bodies.len())
            .map(|slot| {
                (
                    [bodies.pos_x()[slot], bodies.pos_y()[slot]],
                    [bodies.vel_x()[slot], bodies.vel_y()[slot]],
                )
            })
            .collect()
    }

    #[test]
    fn a_cleared_sandbox_is_walls_and_nothing_else() {
        let playground = empty();
        assert_eq!(playground.world.bodies().len(), playground.walls());
        assert_eq!(loose(&playground).len(), 0);
        assert!(playground.status().starts_with("0 bodies"));
    }

    /// Opening the app shows a scene, not an empty box with a hint under it.
    #[test]
    fn the_sandbox_opens_onto_a_standing_stack() {
        let mut playground = PhysicsPlayground::new();
        let placed = loose(&playground);
        assert_eq!(placed.len(), 11, "three rows of bricks and two balls");
        assert!(playground.status().starts_with("11 bodies"));

        // And it is *standing*: two seconds of gravity should not move it much, which
        // is only true if the contacts are holding.
        let before: Vec<[f64; 2]> = placed.iter().map(|(at, _)| *at).collect();
        settle(&mut playground, 2.0);
        for (index, ([x, y], _)) in loose(&playground).iter().enumerate() {
            let drop = before[index][1] - y;
            assert!(
                drop < 0.06 && (x - before[index][0]).abs() < 0.06,
                "body {index} moved from {:?} to ({x:.3}, {y:.3}) — the stack is collapsing",
                before[index]
            );
        }
    }

    #[test]
    fn reset_returns_to_the_scene_it_opened_with() {
        let mut playground = PhysicsPlayground::new();
        let opened = loose(&playground);
        settle(&mut playground, 1.0);
        for step in 0..4 {
            click(&mut playground, [f64::from(step) - 1.0, 2.5]);
        }
        playground.pointer(press([0.0, 2.5]), 0);
        playground.reset();

        let after = loose(&playground);
        assert_eq!(after.len(), opened.len());
        for (index, (at, velocity)) in after.iter().enumerate() {
            assert_eq!(*at, opened[index].0, "body {index} came back somewhere else");
            assert_eq!(*velocity, [0.0, 0.0], "and at rest");
        }
        assert!(playground.grab.is_none());
    }

    #[test]
    fn a_click_on_empty_space_drops_a_body() {
        let mut playground = empty();
        click(&mut playground, [0.0, 2.0]);
        assert_eq!(loose(&playground).len(), 1);
        assert_eq!(loose(&playground)[0].0, [0.0, 2.0], "it lands where you clicked");
    }

    /// The whole point of the mode. If this ever regresses the sandbox is decoration.
    #[test]
    fn a_dropped_body_falls_and_comes_to_rest_on_the_floor() {
        let mut playground = empty();
        click(&mut playground, [0.0, 2.0]);
        settle(&mut playground, 4.0);

        let ([_, y], [_, vy]) = loose(&playground)[0];
        // The floor's centre is at -ARENA[1] and its half-height is 0.15, so its top is
        // 0.15 above that; the default 0.5 m box adds its own 0.25 m half-height.
        let expected = -ARENA[1] + 0.15 + 0.25;
        assert!(
            (y - expected).abs() < 0.02,
            "resting at {y:.4} m, expected about {expected:.4} m"
        );
        assert!(vy.abs() < 0.01, "still moving at {vy:.4} m/s");
    }

    /// Clicking something already there means "pick this up", not "stack another on it".
    /// Getting this backwards makes the sandbox unusable within about four clicks.
    #[test]
    fn a_click_on_a_body_grabs_it_rather_than_dropping_another() {
        let mut playground = empty();
        click(&mut playground, [0.0, 2.0]);
        playground.pointer(press([0.0, 2.0]), 0);

        assert_eq!(loose(&playground).len(), 1, "no second body");
        assert!(playground.grab.is_some(), "holding the first");
        assert!(playground.status().ends_with(", holding one"));
    }

    #[test]
    fn a_grabbed_body_follows_the_cursor() {
        let mut playground = empty();
        click(&mut playground, [0.0, 0.0]);
        playground.pointer(press([0.0, 0.0]), 0);

        // Drag to the right and hold there.
        let target = [2.0, 0.0];
        for _ in 0..240 {
            playground.pointer(Pointer { held: true, ..hover(target) }, 0);
            playground.step(playground.preferred_step());
        }

        let ([x, _], _) = loose(&playground)[0];
        assert!(
            (x - target[0]).abs() < 0.1,
            "chased to {x:.3} m, wanted {:.3} m",
            target[0]
        );
    }

    /// Documented in the module header: the mass scaling exists so a heavy crate and a
    /// light ball reach the cursor together. Without it this ratio is far from one.
    #[test]
    fn a_heavy_body_and_a_light_one_reach_the_cursor_at_the_same_rate() {
        fn chase(size: f64) -> f64 {
            let mut playground = empty();
            playground.set_knob(0, size);
            // No gravity, so the only force acting is the grab.
            playground.set_knob(1, 0.0);
            click(&mut playground, [0.0, 0.0]);
            playground.pointer(press([0.0, 0.0]), 0);
            for _ in 0..60 {
                playground.pointer(Pointer { held: true, ..hover([2.0, 0.0]) }, 0);
                playground.step(playground.preferred_step());
            }
            loose(&playground)[0].0[0]
        }

        // 0.3 m and 1.4 m boxes at the same density: a mass ratio of about 22.
        let (light, heavy) = (chase(0.3), chase(1.4));
        assert!(light > 0.1 && heavy > 0.1, "both actually moved: {light} {heavy}");
        let ratio = light / heavy;
        assert!(
            (0.9..1.1).contains(&ratio),
            "light reached {light:.4} m and heavy {heavy:.4} m — ratio {ratio:.3}, \
             so the grab is not mass-scaled"
        );
    }

    #[test]
    fn a_throw_leaves_the_body_moving_at_the_hands_speed() {
        let mut playground = empty();
        click(&mut playground, [0.0, 0.0]);
        playground.pointer(press([0.0, 0.0]), 0);
        playground.pointer(release([0.1, 0.0], [7.0, 3.0]), 0);

        let (_, velocity) = loose(&playground)[0];
        assert!((velocity[0] - 7.0).abs() < 1e-9, "vx was {}", velocity[0]);
        assert!((velocity[1] - 3.0).abs() < 1e-9, "vy was {}", velocity[1]);
        assert!(playground.grab.is_none(), "let go");
    }

    #[test]
    fn a_right_click_removes_a_body() {
        let mut playground = empty();
        click(&mut playground, [0.0, 1.0]);
        click(&mut playground, [2.0, 1.0]);
        assert_eq!(loose(&playground).len(), 2);

        playground.pointer(Pointer { secondary: true, ..hover([2.0, 1.0]) }, 0);
        assert_eq!(loose(&playground).len(), 1);
        assert!(playground.status().starts_with("1 bodies"));
        assert_eq!(loose(&playground)[0].0, [0.0, 1.0], "the other one survived");
    }

    /// A sandbox with a hole in the floor quietly empties itself, which reads as a broken
    /// simulation and is not one.
    #[test]
    fn a_right_click_never_removes_a_wall() {
        let mut playground = empty();
        for at in [[0.0, -ARENA[1]], [0.0, ARENA[1]], [-ARENA[0], 0.0], [ARENA[0], 0.0]] {
            playground.pointer(Pointer { secondary: true, ..hover(at) }, 0);
        }
        assert_eq!(playground.world.bodies().len(), playground.walls());
    }

    #[test]
    fn a_wall_cannot_be_grabbed() {
        let mut playground = empty();
        playground.pointer(press([0.0, -ARENA[1]]), 0);
        assert!(playground.grab.is_none(), "a static body has no finite mass to pull on");
        assert_eq!(loose(&playground).len(), 0, "and clicking one does not spawn either");
    }

    /// A handle survives its body's removal and refuses to resolve, which is the whole
    /// reason the grab is a `BodyId` and not a slot.
    #[test]
    fn deleting_the_held_body_lets_go_rather_than_panicking() {
        let mut playground = empty();
        click(&mut playground, [0.0, 1.0]);
        playground.pointer(press([0.0, 1.0]), 0);
        assert!(playground.grab.is_some());

        playground.pointer(Pointer { secondary: true, ..hover([0.0, 1.0]) }, 0);
        settle(&mut playground, 0.1);
        assert!(playground.grab.is_none(), "let go of what is no longer there");
        assert_eq!(loose(&playground).len(), 0);
    }

    #[test]
    fn the_pointer_leaving_the_canvas_drops_what_it_was_holding() {
        let mut playground = empty();
        click(&mut playground, [0.0, 1.0]);
        playground.pointer(press([0.0, 1.0]), 0);
        playground.pointer_left();
        assert!(playground.grab.is_none());
        assert!(playground.hover.is_none(), "and no ghost is left on the canvas");
    }

    /// Documented in [`GRAB_MAX_ACCELERATION`]: dragging into a wall must not produce an
    /// unbounded force. The failure mode without the cap is something across the room.
    #[test]
    fn dragging_a_body_through_a_wall_does_not_launch_it() {
        let mut playground = empty();
        click(&mut playground, [4.0, 0.0]);
        playground.pointer(press([4.0, 0.0]), 0);

        // Haul the cursor far outside the arena and hold it there.
        for _ in 0..480 {
            playground.pointer(Pointer { held: true, ..hover([40.0, 0.0]) }, 0);
            playground.step(playground.preferred_step());
        }
        playground.pointer(release([40.0, 0.0], [0.0, 0.0]), 0);
        settle(&mut playground, 2.0);

        let ([x, y], _) = loose(&playground)[0];
        assert!(
            x.abs() < ARENA[0] && y.abs() < ARENA[1],
            "ended up at ({x:.3}, {y:.3}), outside the arena"
        );
    }

    #[test]
    fn nothing_escapes_the_arena() {
        let mut playground = empty();
        for step in 0..24 {
            let x = -4.0 + f64::from(step % 8) * 1.1;
            let y = 1.0 + f64::from(step / 8) * 0.8;
            click(&mut playground, [x, y]);
        }
        // Then fling one as hard as the hand allows.
        playground.pointer(press([0.0, 1.0]), 0);
        playground.pointer(release([0.0, 1.0], [60.0, 45.0]), 0);
        settle(&mut playground, 6.0);

        for (index, ([x, y], _)) in loose(&playground).iter().enumerate() {
            assert!(
                x.abs() < ARENA[0] && y.abs() < ARENA[1],
                "body {index} escaped to ({x:.3}, {y:.3})"
            );
            assert!(x.is_finite() && y.is_finite(), "body {index} went non-finite");
        }
    }

    #[test]
    fn negative_gravity_floats_things_upward() {
        let mut playground = empty();
        click(&mut playground, [0.0, 0.0]);
        playground.set_knob(1, -9.806_65);
        settle(&mut playground, 3.0);

        let ([_, y], _) = loose(&playground)[0];
        assert!(y > 2.0, "should be against the ceiling, is at {y:.3} m");
    }

    #[test]
    fn the_friction_slider_changes_bodies_already_on_screen() {
        let mut playground = empty();
        click(&mut playground, [0.0, 0.0]);
        playground.set_knob(2, 1.25);
        assert!(
            playground.world.colliders().iter().all(|c| (c.friction - 1.25).abs() < 1e-12),
            "a slider labelled friction should mean the whole scene, not the next drop"
        );
    }

    #[test]
    fn the_sandbox_still_works_after_a_reset() {
        let mut playground = empty();
        for step in 0..6 {
            click(&mut playground, [f64::from(step) - 3.0, 1.0]);
        }
        playground.pointer(press([0.0, 1.0]), 0);
        playground.reset();
        assert!(playground.grab.is_none());

        let restored = loose(&playground).len();
        click(&mut playground, [0.0, 3.0]);
        settle(&mut playground, 1.0);
        assert_eq!(loose(&playground).len(), restored + 1);
    }

    #[test]
    fn every_tool_drops_the_shape_it_names() {
        for tool in 0..Spawn::ALL.len() {
            let mut playground = empty();
            playground.pointer(press([0.0, 1.0]), tool);
            playground.pointer(release([0.0, 1.0], [0.0, 0.0]), tool);
            assert_eq!(loose(&playground).len(), 1, "tool {tool} dropped nothing");
            assert_eq!(playground.selected, Spawn::ALL[tool]);
            settle(&mut playground, 3.0);
            let ([x, y], _) = loose(&playground)[0];
            assert!(y > -ARENA[1] && y < 0.0 && x.abs() < 1.0, "tool {tool} settled oddly");
        }
    }

    /// A tool index past the end is the shell's business to get right, but a mode that
    /// panicked on one would take the window with it.
    #[test]
    fn an_out_of_range_tool_is_clamped_rather_than_a_panic() {
        let mut playground = empty();
        playground.pointer(press([0.0, 1.0]), 999);
        assert_eq!(playground.selected, Spawn::Triangle);
    }

    /// A slider labelled in metres has to be the number a ruler held to the screen
    /// would read. The shape constructors take half-extents, so this is one `/ 2.0`
    /// away from being quietly wrong in a way nothing else would catch.
    #[test]
    fn the_size_slider_is_the_width_across_not_the_half_width() {
        let mut playground = empty();
        playground.set_knob(0, 0.9);

        // A ball 0.9 m across dropped on the floor rests 0.45 m above it.
        playground.pointer(press([0.0, 1.0]), 1);
        playground.pointer(release([0.0, 1.0], [0.0, 0.0]), 1);
        settle(&mut playground, 4.0);

        let ([_, y], _) = loose(&playground)[0];
        let expected = -ARENA[1] + 0.15 + 0.45;
        assert!(
            (y - expected).abs() < 0.02,
            "a 0.9 m ball rests at {y:.4} m; if the slider were half-extents it would \
             be at {:.4} m",
            -ARENA[1] + 0.15 + 0.9
        );

        // And the ghost outline drawn before the click agrees with what lands.
        let reach = preview_points(&Spawn::Ball.shape(0.9))
            .into_iter()
            .map(|[x, y]| x.hypot(y))
            .fold(0.0, f64::max);
        assert!((reach - 0.45).abs() < 1e-12, "the preview draws {reach:.4} m");
    }

    /// `clear_bodies` does not clear the shape table, so a reset that reused the world
    /// would register the walls and the opening scene again every time — and
    /// `refresh_surfaces` walks that table on every touch of the friction slider.
    #[test]
    fn resetting_repeatedly_does_not_grow_the_shape_table() {
        let mut playground = PhysicsPlayground::new();
        let opened = playground.world.colliders().len();
        for _ in 0..25 {
            click(&mut playground, [0.0, 3.0]);
            playground.reset();
        }
        assert_eq!(playground.world.colliders().len(), opened);
    }

    /// The model viewer draws contact normals by default because someone reading a run
    /// wants them. A reader who came to throw things at a wall did not ask the question
    /// they answer, and a resting stack of nine boxes has twenty-eight of them.
    #[test]
    fn contact_normals_are_off_until_they_are_asked_for() {
        let mut playground = PhysicsPlayground::new();
        assert!(!playground.toggles()[0].value);
        assert!(!playground.show_contacts);

        playground.set_toggle(0, true);
        assert!(playground.toggles()[0].value);
        playground.set_toggle(0, false);
        assert!(!playground.toggles()[0].value);
    }

    #[test]
    fn the_shape_table_is_shared_rather_than_grown_per_click() {
        let mut playground = empty();
        for step in 0..20 {
            click(&mut playground, [f64::from(step) * 0.3 - 3.0, 2.0]);
        }
        // Two walls plus one box: twenty identical clicks register one collider.
        assert_eq!(playground.world.colliders().len(), 3);
    }
}
