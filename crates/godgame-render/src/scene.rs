//! Putting a world into a requested starting state, for something that is not
//! a person.
//!
//! # Why this is in the library and not the binary
//!
//! It was in the binary, because `--at` and `--edit` are command-line flags and
//! that is where flags are parsed. The result was a scenario tooling that TESTS
//! COULD NOT USE: an integration test boots the plugin group and inserts
//! resources by hand, so reaching a situation meant reimplementing this, and
//! reimplementing this meant getting three orderings right again that took three
//! attempts the first time (see [`arrange_the_scene`]).
//!
//! So a scenario proved by hand at a terminal could never become a gate without
//! somebody rewriting it — and the whole point of being able to reach a
//! situation is to be able to ASSERT something about it afterwards. The flags
//! stay in the binary; the meaning of them lives here, and
//! `tests/scenarios.rs` uses exactly the same code path the command line does.
//!
//! # What a scenario cannot say yet
//!
//! Worth stating, because the gaps decide which questions are still unanswerable
//! without new Rust:
//!
//!   - **one brush stroke, not a list.** A scene needing two materials needs two
//!     runs sharing a save directory.
//!   - **no mob placement**, so combat is not reachable this way.
//!
//! [`StartWith`] closed the third of these — an inventory could not be seeded,
//! so nothing could start holding a lantern and `scenarios/README.md` records
//! there being no scripted route to a lit cave. It is also what makes crafting
//! testable at all: a recipe needs ingredients, and mining them takes longer
//! than any scenario wants to run.

use bevy::prelude::*;
use godgame_core::config::cell_at;
use godgame_core::items::registry::ItemCode;
use godgame_core::sim::edits::{EditMode, apply_brush};
use godgame_core::sim::materials::CellId;

use crate::items::Pack;
use crate::player::PlayerBody;
use crate::world::{SimWorld, WorldFocus, place_body};

/// Put the body here once the world has streamed to it.
///
/// World px, sim convention (+y down). This is what turns "photograph a
/// lava-lit cave" or "check the sky at dusk" from a hand-built Rust scene into
/// one line — `crate::lit_scene`'s header records being built and thrown away
/// three times because the default spawn is a snowy surface in daylight where
/// most of the lighting stack is invisible.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct StartAt {
    /// World px, +x right.
    pub x: f32,
    /// World px, +y DOWN.
    pub y: f32,
}

/// One brush stroke stamped at load.
///
/// The brush is a mouse verb and a headless run has no mouse. This is the same
/// stroke `crate::input` would emit, taken from a resource instead: it is what
/// lets an agent or CI show that digging changes the world and that the automata
/// reacts to the hole, rather than assert it from a test that never drew a pixel.
///
/// `cx`/`cy` are cells RELATIVE TO THE VIEW CENTRE, because that is the only
/// coordinate a caller knows without first reading the spawn point out of the
/// worldgen — and, with [`StartAt`], the centre is wherever the caller asked to
/// be.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct StartupEdit {
    /// Dig or place.
    pub mode: EditMode,
    /// Material to place. Ignored when digging.
    pub mat: CellId,
    /// Cells right of the view centre.
    pub cx: i32,
    /// Cells below the view centre.
    pub cy: i32,
    /// Disc radius in cells.
    pub r: i32,
}

/// Items to put in the pack before anything else happens.
///
/// The starting kit is a pick, a sword and three bandages, which is the right
/// thing for a player and useless for a scenario: every question about crafting
/// needs INGREDIENTS, and mining them takes longer than any scenario wants to
/// run. Without this, "does a workbench recipe work" is unanswerable without
/// writing Rust — which is the whole thing scenarios exist to avoid.
///
/// Added on top of the starting kit rather than replacing it, because a scenario
/// asking for three logs is asking for three logs and not for an empty pack.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct StartWith(pub Vec<(ItemCode, u32)>);

/// Hand the pack what [`StartWith`] asked for, once.
///
/// Through `Inventory::add`, which is the game's own placement policy — merge
/// into an existing stack, then first free slot, respecting stack limits. NOT
/// `put_at`, which the save loader uses: a scenario says *what* the player has
/// and has no business deciding *where*, and going through `add` means a
/// scenario cannot express an inventory the game could not itself produce.
fn give_at_start(mut commands: Commands, want: Res<StartWith>, pack: Option<ResMut<Pack>>) {
    let Some(mut pack) = pack else {
        // No inventory in this app at all. Not an error — `UiPlugin` and the
        // scene machine run without one — but worth saying, because a scenario
        // that silently got nothing would look like a broken recipe.
        warn!("scene: --give had nothing to give to; this app has no inventory");
        commands.remove_resource::<StartWith>();
        return;
    };
    for (code, n) in &want.0 {
        let left = pack.0.add(*code, *n);
        if left > 0 {
            warn!("scene: --give could not fit {left} of item code {code}");
        }
    }
    info!("scene: gave {} stack(s)", want.0.len());
    commands.remove_resource::<StartWith>();
}

/// Put the scene in the state [`StartAt`] and [`StartupEdit`] asked for.
///
/// ONE system, and not three chained ones, because the steps are not independent
/// and an ordering alone was not enough. Three versions looked right and were
/// not, each found only with a state dump — a screenshot showed "a cave" in all
/// three:
///
/// 1. Place the body, then carve. The body is standing in solid rock and the
///    collision resolver ejects it before the carve that was meant to make room
///    lands — 260 px of drift, into terrain nobody asked to see.
/// 2. Carve, then place. [`StartupEdit`]'s coordinates are cells from the VIEW
///    CENTRE, which is still the spawn, so a stroke aimed 900 px down carved at
///    cell (-127, 32) — a thousand cells from the cave it was meant to make.
/// 3. Aim the camera, carve, place, all on one frame. The camera moves instantly
///    and the WORLD does not: the streaming window still covered the spawn, the
///    brush was clipped away entirely, and the "cave" was solid stone with the
///    body ejected out of the top of it.
///
/// So it waits. The focus moves on the first frame, `world::stream_window`
/// brings the world to it over the next few, and nothing is stamped until the
/// grid actually holds the cell the stroke is aimed at. Then the carve and the
/// placement happen together, in that order, on one frame, and both resources
/// are removed — a placement that ran every frame would pin the body and no
/// scenario could ever walk away from it.
///
/// **If you add a flag that touches the world at startup, put it in here rather
/// than beside it.** Anything ordered against this from outside is a fourth
/// version of the bug above.
pub fn arrange_the_scene(
    mut commands: Commands,
    at: Option<Res<StartAt>>,
    edit: Option<Res<StartupEdit>>,
    mut focus: ResMut<WorldFocus>,
    mut world: ResMut<SimWorld>,
    body: Option<ResMut<PlayerBody>>,
) {
    // The camera first, and every frame until this system retires, so the
    // streamer has somewhere to go.
    if let Some(at) = &at {
        place_body(**at, &mut focus, None);
    }

    if let Some(edit) = &edit {
        let cx = cell_at(focus.x) + edit.cx;
        let cy = cell_at(focus.y) + edit.cy;
        let grid = &world.level.grid;
        let (gx, gy) = (cx - grid.origin_cell_x(), cy - grid.origin_cell_y());
        if gx < 0 || gy < 0 || gx >= grid.cols() || gy >= grid.rows() {
            // Still streaming. Next frame. Stamping now would clip the stroke to
            // nothing, which looks exactly like a brush that did not work.
            return;
        }
        apply_brush(&mut world.level.grid, edit.mode, cx, cy, edit.r, edit.mat);
        info!("scene: {:?} at cell ({cx}, {cy}) r={}", edit.mode, edit.r);
        commands.remove_resource::<StartupEdit>();
    }

    if let Some(at) = at {
        let at = *at;
        place_body(at, &mut focus, body.map(ResMut::into_inner));
        info!("scene: body at ({}, {})", at.x, at.y);
        commands.remove_resource::<StartAt>();
    }
}

/// Frames a caller should allow for [`arrange_the_scene`] to finish.
///
/// Not a guess. The stroke is stamped on the first frame the streaming window
/// covers its target, and a `StartAt` far from the spawn needs the window to
/// shift there first. Measured: a target 900 px down was covered on frame 2, and
/// `scenarios/README.md` records that `--warmup 0` and `1` produce untouched
/// rock — a silent failure that looks exactly like a no-op brush.
///
/// This is the floor, not a settling time. Anything wanting the automata to
/// settle, chunks to finish arriving or creatures to spawn needs its own,
/// larger, number on top.
pub const ARRANGE_FRAMES: u32 = 8;

/// [`arrange_the_scene`], registered.
///
/// Deliberately NOT in `GodGameRenderPlugin`: a scene request is scaffolding for
/// a capture, a scenario or a test, and a game nobody is instrumenting should
/// not carry a system that looks for resources it will never have.
pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                arrange_the_scene.run_if(resource_exists::<SimWorld>),
                // After `glue::start_a_run` has handed out the starting kit,
                // which `Update` already is: the kit goes out on the state
                // transition into `Playing`, a schedule earlier.
                give_at_start.run_if(resource_exists::<StartWith>),
            ),
        );
    }
}
