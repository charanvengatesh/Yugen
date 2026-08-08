//! Player interaction with the world — the dig/place tool.
//!
//! Paired with `crate::interact`. The brush is a radius in CELLS, not px,
//! because what the player is editing is the cell grid; converting to px at the
//! edit site keeps the number meaningful when the zoom changes.

/// Starting brush radius, in cells.
pub const BRUSH_RADIUS: i32 = 4;

/// Smallest brush radius the player may scroll to, in cells.
pub const BRUSH_MIN: i32 = 1;

/// Largest brush radius the player may scroll to, in cells.
pub const BRUSH_MAX: i32 = 20;
