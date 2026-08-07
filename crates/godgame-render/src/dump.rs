//! `--dump-state out.json`: the whole game, as text, for something that is not
//! a person.
//!
//! # Why this exists
//!
//! There are two ways to find out whether a change worked, and until now the
//! tree had only the first of each pair. You can take a SCREENSHOT and look at
//! it, which needs eyes and answers "does it look right"; or you can write a
//! bespoke Rust test, which needs a compile and answers exactly one question
//! forever. Neither answers "did the inventory actually gain three stone, and is
//! the body where I think it is" without somebody doing work proportional to the
//! question.
//!
//! This is the third instrument. It runs the real binary, with the real
//! renderer, and writes what the simulation believes — player, inventory,
//! creatures, clock, and a window of cells around the body — as JSON that `jq`,
//! a diff, or a script can read. Paired with `--script` it turns "does this
//! work" into a committed artefact, and paired with itself it turns "did this
//! change anything" into a diff.
//!
//! # It is a view, never a source
//!
//! Every field is read from a live resource at the moment of the dump. Nothing
//! here computes a value, caches one, or keeps its own copy, for
//! `crate::debug`'s reason: a dump that derived its own answer would agree with
//! itself and disagree with the game, and it would do so most convincingly
//! exactly when the game was wrong.
//!
//! # What it deliberately leaves out
//!
//! The world. A dump of 352x256 live cells is 90 000 numbers that change every
//! tick and diff uselessly; [`CELL_WINDOW`] takes a small square around the body
//! instead, which is the part any question is actually about. The full world is
//! already reproducible from `(seed, chunk)` — that is what `worldgen_purity`
//! proves — so a dump that carried it would be storing something derivable.

use std::path::PathBuf;

use bevy::prelude::*;
use godgame_core::config::{CELL_SIZE, CHUNK_CELLS};
use godgame_core::items::inventory::{HOTBAR, SLOT_COUNT};
use godgame_core::json::Json;
use godgame_core::sim::coords::WorldCell;

use crate::daynight::WorldClock;
use crate::items::{GroundItems, Pack};
use crate::mobs::Creatures;
use crate::player::PlayerBody;
use crate::world::{SimWorld, WorldFocus};

/// Half-width, in cells, of the square of terrain the dump carries.
///
/// 10 gives a 21x21 block centred on the body — 441 numbers, which is small
/// enough to read and to diff line by line, and wide enough to cover the reach
/// of any tool (`reach 4` on the starting pick) plus the ground under the feet
/// and the ceiling overhead. A whole streaming window would be 90 000 numbers
/// and would bury the fields anybody came for.
const CELL_WINDOW: i32 = 10;

/// Where to write, and whether it has been written.
///
/// A resource rather than a plain flag because the binary inserts it from a CLI
/// argument and the system that acts on it lives here — the same shape
/// `--screenshot` uses.
#[derive(Resource, Clone, Debug)]
pub struct DumpState {
    /// The file to write.
    pub path: PathBuf,
    /// Frames still to render before the dump is taken.
    pub frames_left: u32,
}

/// Everything the dump reads. Bundled so the system stays inside clippy's
/// argument budget, as `light`'s solve and `debug`'s gather both do.
#[derive(bevy::ecs::system::SystemParam)]
pub struct DumpSources<'w> {
    world: Option<Res<'w, SimWorld>>,
    focus: Res<'w, WorldFocus>,
    clock: Option<Res<'w, WorldClock>>,
    body: Option<Res<'w, PlayerBody>>,
    pack: Option<Res<'w, Pack>>,
    creatures: Option<Res<'w, Creatures>>,
    ground: Option<Res<'w, GroundItems>>,
}

/// Count down, then write once and drop the resource.
///
/// The countdown is the same mechanic `--screenshot` uses and for the same
/// reason: chunks stream, creatures spawn over a few hundred frames, and a dump
/// taken on frame 1 describes a world that has not finished existing.
pub fn dump_when_ready(mut commands: Commands, mut state: ResMut<DumpState>, src: DumpSources) {
    if state.frames_left > 0 {
        state.frames_left -= 1;
        return;
    }
    let text = render(&src);
    match std::fs::write(&state.path, &text) {
        Ok(()) => info!("--dump-state wrote {}", state.path.display()),
        // A warning and not a panic: the dump is an observation, and losing it
        // should not take down a run somebody may also be screenshotting.
        Err(e) => warn!("--dump-state could not write {}: {e}", state.path.display()),
    }
    commands.remove_resource::<DumpState>();
}

/// The document.
///
/// Split out from [`dump_when_ready`] so it is a function of its inputs: a test
/// can build the sources and compare the text without a filesystem.
pub fn render(src: &DumpSources) -> String {
    let mut j = Json::new();
    j.object(|j| {
        j.field("focus", |j| {
            j.object(|j| {
                j.field("x", |j| j.float(f64::from(src.focus.x)));
                j.field("y", |j| j.float(f64::from(src.focus.y)));
            });
        });

        let Some(world) = src.world.as_ref() else {
            // A run with no world is a real state — the menu, or a boot that
            // failed — and saying so beats emitting a plausible skeleton of
            // zeroes. `crate::debug`'s panel makes the same choice.
            j.field("world", |j| j.null());
            return;
        };
        j.field("seed", |j| j.uint(u64::from(world.seed)));
        j.field("window", |j| {
            j.object(|j| {
                let g = &world.level.grid;
                j.field("origin_cell_x", |j| j.int(i64::from(g.origin_cell_x())));
                j.field("origin_cell_y", |j| j.int(i64::from(g.origin_cell_y())));
                j.field("cols", |j| j.int(i64::from(g.cols())));
                j.field("rows", |j| j.int(i64::from(g.rows())));
            });
        });

        j.field("clock", |j| match src.clock.as_ref() {
            Some(c) => j.object(|j| {
                j.field("t", |j| j.float(f64::from(c.0.t())));
                j.field("day", |j| j.float(f64::from(c.0.phase().day)));
            }),
            None => j.null(),
        });

        j.field("player", |j| match src.body.as_ref() {
            Some(b) => j.object(|j| {
                let p = &b.0;
                j.field("x", |j| j.float(f64::from(p.x)));
                j.field("y", |j| j.float(f64::from(p.y)));
                j.field("vx", |j| j.float(f64::from(p.vx)));
                j.field("vy", |j| j.float(f64::from(p.vy)));
                j.field("facing", |j| j.float(f64::from(p.facing)));
                j.field("health", |j| j.float(f64::from(p.health)));
                j.field("untouchable", |j| j.bool(p.untouchable));
            }),
            // Free camera. Not an error, and not a body at the origin.
            None => j.null(),
        });

        j.field("inventory", |j| match src.pack.as_ref() {
            Some(pack) => j.object(|j| {
                let inv = &pack.0;
                j.field("selected", |j| j.uint(inv.selected() as u64));
                j.field("hotbar_slots", |j| j.uint(HOTBAR as u64));
                // Only the OCCUPIED slots, each carrying its index. A fixed
                // 30-element array of mostly nulls would diff badly: adding one
                // item to slot 0 would shift nothing, but reading the file to
                // find what changed would mean counting nulls.
                j.field("slots", |j| {
                    j.array(|j| {
                        for slot in 0..SLOT_COUNT {
                            let Some((code, count)) = inv.stack_at(slot) else {
                                continue;
                            };
                            j.object(|j| {
                                j.field("slot", |j| j.uint(slot as u64));
                                j.field("item", |j| j.str(item_id(code)));
                                j.field("count", |j| j.uint(u64::from(count)));
                            });
                        }
                    });
                });
            }),
            None => j.null(),
        });

        j.field("mobs", |j| {
            j.array(|j| {
                let Some(c) = src.creatures.as_ref() else {
                    return;
                };
                for m in c.0.mobs().iter().filter(|m| m.active) {
                    j.object(|j| {
                        j.field("id", |j| j.str(m.def.id));
                        j.field("x", |j| j.float(f64::from(m.body.x)));
                        j.field("y", |j| j.float(f64::from(m.body.y)));
                        j.field("vx", |j| j.float(f64::from(m.vx)));
                        j.field("vy", |j| j.float(f64::from(m.vy)));
                        j.field("health", |j| j.float(f64::from(m.health)));
                    });
                }
            });
        });
        j.field("xp_banked", |j| {
            j.int(i64::from(
                src.creatures.as_ref().map_or(0, |c| c.0.xp_banked()),
            ));
        });
        j.field("drops", |j| {
            j.uint(src.ground.as_ref().map_or(0, |g| g.0.active()) as u64);
        });

        // The terrain the body is standing in. Two planes, because the wall
        // plane is invisible in a screenshot wherever the front plane covers it
        // and this is the only instrument that can show it at all.
        let centre = WorldCell::new(
            (src.focus.x / CELL_SIZE as f32).floor() as i32,
            (src.focus.y / CELL_SIZE as f32).floor() as i32,
        );
        j.field("cells", |j| {
            j.object(|j| {
                j.field("centre", |j| {
                    j.array(|j| {
                        j.int(i64::from(centre.x));
                        j.int(i64::from(centre.y));
                    });
                });
                j.field("radius", |j| j.int(i64::from(CELL_WINDOW)));
                j.field("chunk", |j| {
                    j.array(|j| {
                        j.int(i64::from(centre.x.div_euclid(CHUNK_CELLS)));
                        j.int(i64::from(centre.y.div_euclid(CHUNK_CELLS)));
                    });
                });
                // One array per row, so a diff points at a row rather than at an
                // offset into a flat run of 441 numbers.
                for (name, back) in [("front", false), ("wall", true)] {
                    j.field(name, |j| {
                        j.array(|j| {
                            for dy in -CELL_WINDOW..=CELL_WINDOW {
                                j.array(|j| {
                                    for dx in -CELL_WINDOW..=CELL_WINDOW {
                                        let at = WorldCell::new(centre.x + dx, centre.y + dy);
                                        let id = if back {
                                            world.level.grid.get_back_world(at)
                                        } else {
                                            world.level.grid.get_world(at)
                                        };
                                        j.uint(u64::from(id));
                                    }
                                });
                            }
                        });
                    });
                }
            });
        });
    });
    j.finish()
}

/// An item code's authoring id, or a placeholder.
///
/// Ids rather than display names, because this file is read by tools: `"coal"`
/// is what a `content/` file and a recipe both call it, and "Coal Lump" is a
/// string that changes when somebody rewrites a label.
fn item_id(code: godgame_core::items::registry::ItemCode) -> &'static str {
    godgame_data::items::ITEMS
        .get(code as usize)
        .map_or("?", |d| d.id)
}
