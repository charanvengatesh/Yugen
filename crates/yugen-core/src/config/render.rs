//! Render tunables that cross module boundaries.
//!
//! Only knobs more than one pass has to agree on live here. A coefficient
//! private to one pass (a sky gradient stop, a particle's fade curve) stays in
//! that pass's own tuning block — hoisting it would make this file a second
//! dumping ground, which is the thing the split exists to prevent.

/// Cells per dynamic-light sample.
///
/// ONE sample per cell, so a light texel covers exactly the [`CELL_SIZE`] px of
/// the cell under it and the light grid shares the art's lattice. That is what
/// lets the grid be sampled NEAREST: the glow lands on block boundaries instead
/// of straddling them, and a lit block reads as a block rather than as an
/// airbrushed smudge over one.
///
/// This was 4 — a 20 px texel, four times the art's, which had to be sampled
/// linearly because nearest at that size lays a foreign lattice over the frame.
/// The smooth upscale that bought is the soft wash the underground was reported
/// as having, and it was blamed on the bloom twice before anyone measured it.
///
/// **Every quantity measured in light cells must be DERIVED from this, never
/// re-tuned by hand.** A light cell is a unit of DISTANCE, so changing this
/// silently changes what a per-light-cell decay, splat reach or blur width
/// means in the world. `yugen_render::light` states each of them per SIM CELL
/// and raises to this; see its `OPEN_DECAY` for the pattern.
///
/// Read by the light solver, the ambience layer's hot scan, and the benches
/// that size a scratch grid — hence shared rather than private.
///
/// [`CELL_SIZE`]: super::CELL_SIZE
pub const LIGHT_DOWNSCALE: i32 = 1;
