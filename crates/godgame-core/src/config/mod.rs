//! The configuration tier — every cross-module tuned number in the game.
//!
//! ```ignore
//! use godgame_core::config::{CELL_SIZE, GRAVITY};
//! ```
//!
//! # What lives here, and what does not
//!
//! A number belongs in `config` when more than one module has to agree on it, or
//! when it is a knob a designer would reach for. A coefficient private to one
//! algorithm — an fBm octave count in the cave generator, a gradient stop in the
//! sky — stays next to the code that owns it. Hoisting those here would recreate
//! the 483-line monolith this replaced.
//!
//! A number belongs in `content/` instead of here when it describes one THING
//! rather than the whole game: a block's hardness, a mob's speed, a weapon's
//! damage. Those are compiled by `contentc` into `godgame-data` and the config
//! tier never restates them.
//!
//! Every `pub` item in this module tree must carry a doc comment — `cargo xtask
//! tuning` fails the build otherwise. An undocumented number is one nobody can
//! safely turn.
//!
//! # The domains
//!
//! | Module | Owns |
//! |---|---|
//! | [`world`] | cell/chunk/window geometry, sim rate. The units everything else uses. Depends on nothing. |
//! | [`view`] | logical buffer resolution and zoom |
//! | [`worldgen`] | where the ground, sea and depth bands sit |
//! | [`physics`] | the player's body and how it moves; owns `PHYS_SCALE` / `scaled` |
//! | [`combat`] | bare-handed melee and projectile facts weapons cannot override |
//! | [`mechanics`] | abilities (dash, wall jump) and surface reactions (ice, bounce) |
//! | [`render`] | render knobs shared across passes |
//! | [`interact`] | the dig/place brush |

pub mod combat;
pub mod interact;
pub mod mechanics;
pub mod physics;
pub mod render;
pub mod view;
pub mod world;
pub mod worldgen;

pub use combat::*;
pub use interact::*;
pub use mechanics::*;
pub use physics::*;
pub use render::*;
pub use view::View;
pub use world::*;
pub use worldgen::*;
