//! The bestiary: what a creature IS, how one thinks, and who owns the crowd.
//!
//! Three files, one per question, and the arrow points one way:
//!
//! | Module | Owns |
//! |---|---|
//! | [`defs`] | the facade over the compiled bestiary — the load-time scaling, the geometry split, the habitat vocabulary |
//! | [`brain`] | one creature: the five brains, the shared physics, and the creatures' own RNG |
//! | [`system`] | the fixed pool, spawning, culling, projectiles and combat |
//!
//! A brain reads a [`MobDef`] and never a table; the system reads both and is
//! the only thing that knows a player exists.
//!
//! # What the port dropped on the way in
//!
//! `MobSystem.draw`, `drawTell`, `drawGlow` and `drawShots`, and the `Sprite`
//! the facade constructed per creature. They were Canvas2D, and this crate
//! carries no renderer. Everything they READ is published here instead — the art
//! rect and its padding on [`MobDef`], the pose and the animation clock on
//! [`Mob`], the tell countdown ([`TELL_TIME`] and `Mob::tell_t`), the colour and
//! glow on [`Shot`] — so the drawing can be reassembled in `yugen-render`
//! without the simulation gaining a pixel of knowledge about how it looks.
//!
//! The tuning that ONLY the draw code read went with it: the tell's alpha ramp
//! and per-cell jitter, and the projectile's `lighter` composite. They describe
//! the shape of a drawn thing, not of a body, and the tuning tier says a number
//! describing one algorithm lives next to that algorithm.

pub mod brain;
pub mod defs;
pub mod system;

pub use brain::{Mob, MobClock, MobRng, TELL_TIME, step_mob};
pub use defs::{
    BAND_SURFACE_MAX, Band, Dmg, MOB_COUNT, MOB_DEFS, MobBand, MobBrain, MobDef, MobDrop, MobPose,
    MobProjectile, MobRanged, Mobf, VARIANT_COUNT, band_at_depth, band_bit, def_by_id, def_index,
};
pub use system::{
    MAX_MOBS, MobEvent, MobEventKind, MobLoot, MobSystem, MobTarget, ProjectileSpec, Shot,
    SpawnRects,
};
