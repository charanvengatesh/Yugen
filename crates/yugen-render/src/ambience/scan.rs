//! Where to look, and what was found there.
//!
//! [`super`]'s "The view rectangle" and "The hot list". A mote is spawned from a
//! CELL, and scanning every cell in view every frame would be the one part of
//! this system with a cost worth caring about — so the rectangle is computed
//! once and the emitting cells inside it are collected into a list the spawner
//! samples from.
//!
//! No Bevy: this is arithmetic over a `CellGrid` and a `View`.

use yugen_core::config::{CELL_SIZE, View};
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, EMPTY, mat_by_code};

use super::spawn::*;

// ---------------------------------------------------------------------------
// The view rectangle
// ---------------------------------------------------------------------------

/// The rectangle of world an emitter may spawn into, in world px, +y DOWN.
///
/// Spawn points are rejection-sampled inside this and nowhere else. The streamed
/// window is a good deal larger, and sampling it would spend most of the tries
/// on cells nobody can see.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRect {
    /// Left edge, world px.
    pub x: f32,
    /// Top edge, world px, +y DOWN.
    pub y: f32,
    /// Width, world px.
    pub w: f32,
    /// Height, world px.
    pub h: f32,
}

impl ViewRect {
    /// The rect a `view`-sized viewport centred on world px `(cx, cy)` covers.
    ///
    /// [`WorldFocus`] is the CENTRE and the TypeScript camera's `x`/`y` were the
    /// top-left, which is the whole of the difference between this and the
    /// original's `cam.viewX`.
    pub fn centred_on(cx: f32, cy: f32, view: View) -> ViewRect {
        let w = view.w as f32;
        let h = view.h as f32;
        ViewRect {
            x: cx - w * 0.5,
            y: cy - h * 0.5,
            w,
            h,
        }
    }

    /// Centre column, in cells and fractional — where the climate is sampled.
    #[inline]
    pub fn centre_col(&self) -> f32 {
        (self.x + self.w * 0.5) / CELL_SIZE as f32
    }

    /// Centre row, in cells and fractional.
    #[inline]
    pub fn centre_row(&self) -> f32 {
        (self.y + self.h * 0.5) / CELL_SIZE as f32
    }
}

/// Everything outside the model that one frame of ambience reads.
///
/// Bundled rather than passed as three more arguments: they travel together,
/// they all answer "what does the world look like right now", and a `&mut self`
/// method taking six references is where an argument gets silently swapped.
pub struct Around<'a> {
    /// The streamed cells. Only cells inside [`Around::view`] are sampled.
    pub grid: &'a CellGrid,
    /// The view rectangle, world px, +y DOWN.
    pub view: ViewRect,
    /// Emissive cells found in view this frame — see [`scan_hot`].
    pub hot: &'a [WorldCell],
}

// ---------------------------------------------------------------------------
// The hot list
// ---------------------------------------------------------------------------

/// The light pass's emit level for a material, 0 when it is not an emitter.
///
/// Reproduces the TypeScript's `EMIT_LEVEL` table exactly: a declared
/// `light_emit` normalised by [`EMIT_MAX_LEVEL`] wins, and a material with none
/// falls back to its renderer `emissive` at [`EMIT_FALLBACK_GAIN`]. Air is never
/// an emitter — the TypeScript's table loop started at id 1 for the same reason.
pub fn emit_level(id: CellId) -> f32 {
    if id == EMPTY {
        return 0.0;
    }
    let def = mat_by_code(id);
    let declared = def.light_emit as f32 / EMIT_MAX_LEVEL;
    if declared > 0.0 {
        declared
    } else {
        def.emissive * EMIT_FALLBACK_GAIN
    }
}

/// Cells between hot-list samples.
///
/// **Deliberately its own number, not [`LIGHT_DOWNSCALE`].** It was that
/// constant, from when this stood in for a lighting pass that did not exist yet
/// and had to find the same cells the light grid would. Both facts have since
/// stopped being true: the lighting pass exists, and it does not sample on a
/// stride any more — `LIGHT_DOWNSCALE` went to 1 so light could sit on the art's
/// own grid.
///
/// Following it there cost 16x for nothing. This scan does not want a faithful
/// mirror of the solver; it wants a sparse "roughly where is it hot" for an
/// EMBER RATE, and it is capped at [`HOT_MAX`] anyway. Measured, the coupled
/// version went 1.26 us -> 17.69 us per frame, and worse than the cost: the cap
/// then covered a sixteenth of the area, so ember spawns bunched toward the
/// top-left of the view instead of spreading across it.
///
/// 4 is the value this scan was tuned at and is what keeps the ember rate where
/// it has always been.
pub(super) const HOT_STRIDE_CELLS: i32 = 4;

/// Fill `out` with the emissive cells in view, capped at [`HOT_MAX`].
///
/// Strided by [`HOT_STRIDE_CELLS`] and sampling the stride's centre cell. A lava
/// pool one cell wide can therefore be missed, and it will be missed
/// CONSISTENTLY — the lattice is anchored to the world and not to the camera, so
/// a hot cell does not blink as the view scrolls a pixel. That consistency is
/// the property that matters here, not completeness: this feeds how many embers
/// drift up, not what is lit.
pub fn scan_hot(grid: &CellGrid, view: ViewRect, out: &mut Vec<WorldCell>) {
    out.clear();

    let step = HOT_STRIDE_CELLS;
    let half = step / 2;
    let cell = CELL_SIZE as f32;
    let lx0 = div_floor((view.x / cell).floor() as i32, step);
    let lx1 = div_floor(((view.x + view.w) / cell).ceil() as i32, step);
    let ly0 = div_floor((view.y / cell).floor() as i32, step);
    let ly1 = div_floor(((view.y + view.h) / cell).ceil() as i32, step);

    for ly in ly0..=ly1 {
        for lx in lx0..=lx1 {
            if out.len() >= HOT_MAX {
                return;
            }
            let at = WorldCell::new(lx * step + half, ly * step + half);
            if !grid.is_loaded_world(at) {
                continue;
            }
            if emit_level(grid.get_world(at)) > 0.0 {
                out.push(at);
            }
        }
    }
}

/// Floor division, so the light lattice is continuous west and north of the
/// world origin instead of folding at it.
#[inline]
pub(super) fn div_floor(a: i32, b: i32) -> i32 {
    let q = a / b;
    if a % b != 0 && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}
