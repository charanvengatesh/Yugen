//! One schema per content kind.
//!
//! **Adding a field to the game is adding a row here.** Each module returns a
//! `Schema` describing one kind: its fields in emission order, the flat arrays
//! its hot fields produce, its bit-constant sets, its matrices, and the
//! placeholder used for a tombstoned id.
//!
//! Ordering is significant twice over: `default` callbacks see the def built so
//! far (so `collides` can key off `state`), and the emitted struct follows this
//! order. Never reorder a field list to tidy it.

use crate::schema::Schema;

pub mod block;
pub mod feature;
pub mod item;
pub mod mob;
pub mod sprite;
pub mod structure;

/// One registered kind: its schema, where its records live, and what it emits.
///
/// This table is the whole registration surface — six rows. A seventh kind is a
/// seventh row plus a module.
pub struct Kind {
    pub schema: Schema,
    /// Directory under `content/`.
    pub dir: &'static str,
    /// Output module name under `yugen-data/src/`.
    pub out: &'static str,
}

/// Every kind, in a fixed order so code assignment and emission are
/// deterministic across runs.
pub fn kinds() -> Vec<Kind> {
    vec![
        Kind {
            schema: block::schema(),
            dir: "blocks",
            out: "blocks",
        },
        Kind {
            schema: item::schema(),
            dir: "items",
            out: "items",
        },
        Kind {
            schema: mob::schema(),
            dir: "mobs",
            out: "mobs",
        },
        Kind {
            schema: sprite::schema(),
            dir: "sprites",
            out: "sprites",
        },
        Kind {
            schema: structure::schema(),
            dir: "structures",
            out: "structs",
        },
        Kind {
            schema: feature::schema(),
            dir: "worldgen",
            out: "worldgen",
        },
    ]
}
