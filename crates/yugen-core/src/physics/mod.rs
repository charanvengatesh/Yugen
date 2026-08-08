//! How bodies move through the cell world.
//!
//! Everything here is a free function over a `&CellGrid` and an [`Aabb`]: the
//! collider owns no state, knows nothing about the player, the mobs or the
//! items, and is therefore the one piece every moving thing in the game can
//! share. The entities on top of it own the velocity, the gravity and the
//! decisions; this layer only answers "where does that put me".

pub mod collision;

pub use collision::{
    Aabb, NO_ONE_WAY, ResolveHits, ResolveResult, StepResult, box_overlaps_solid,
    for_each_overlapped_cell, is_solid_cell, move_horizontal_stepped, one_way_under_feet,
    resolve_axis,
};
