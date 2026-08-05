//! Render tunables that cross module boundaries.
//!
//! Only knobs more than one pass has to agree on live here. A coefficient
//! private to one pass (a sky gradient stop, a particle's fade curve) stays in
//! that pass's own tuning block — hoisting it would make this file a second
//! dumping ground, which is the thing the split exists to prevent.

/// Cells per dynamic-light sample.
///
/// One light sample per 4 sim cells (20px). The small light grid is upscaled
/// smoothly at draw time. Read by the light solver, the cell pass that samples
/// it, and the benches that size a scratch grid — hence shared rather than
/// private.
pub const LIGHT_DOWNSCALE: i32 = 4;
