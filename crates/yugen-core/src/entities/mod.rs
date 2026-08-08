//! The things that move under their own power.
//!
//! Entities own velocity, gravity and decisions; [`crate::physics`] owns "where
//! does that put me" and [`crate::sim`] owns the cells. The arrow points one
//! way: an entity reads the grid and the collider, and neither of them has ever
//! heard of an entity.
//!
//! # What the port dropped on the way in
//!
//! `Player.draw` and `Player.drawGlow` did not come across. They were Canvas2D —
//! `ctx.save`, `ctx.transform`, `PLAYER_SPRITE.draw` — and this crate carries no
//! renderer. Everything they READ is published here as an accessor instead
//! (`squash`, `accel_x`, `step_up_visual`, `dash_dir`, `anim`, `anim_t`,
//! `run_phase`), so the drawing can be reassembled in `yugen-render` without
//! the simulation gaining a pixel of knowledge about how it looks.
//!
//! The tuning that ONLY `draw` read went with it and is not restated here:
//! `DASH_SMEAR_X`/`_Y`, `DASH_GHOST_STEP`, `RISE_STRETCH_DIV`/`_MIN`/`_MAX`,
//! `LEAN_DIV` and `LEAN_MAX`. They are squash, smear and shear — the shape of a
//! drawn sprite, not of a body — and the tuning tier says a number describing
//! one algorithm lives next to that algorithm. `ACCEL_SMOOTH` is the exception
//! that proves it: the lean's low-pass is applied inside `update_anim`, which
//! runs on the fixed step, so the constant stays with the code that uses it.
//!
//! [`projectiles`] lost its `draw`/`drawGlow` the same way and answers the same
//! way: [`ProjectileSystem::shots`](projectiles::ProjectileSystem::shots) plus
//! the two style tables is everything the two passes read.

pub mod mobs;
pub mod player;
pub mod projectiles;

pub use player::{
    AmmoSource, AnimState, Loadout, NoProjectiles, Player, PlayerEvent, PlayerWeapon, Projectiles,
};
pub use projectiles::{
    HitFn, MAX_SHOTS, NoTargets, ProjectileSystem, SHOT_STYLE_ARROW, Shot, ShotSpec, ShotWorld,
};
