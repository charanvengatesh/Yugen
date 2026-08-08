//! Viewport and zoom.
//!
//! In the TypeScript build this module read `window.innerWidth` at import time
//! and froze `VIEW_W`/`VIEW_H` as module constants — which meant it was the one
//! config module that touched the DOM, and that resizing the window did nothing
//! until reload. Native has a real window that can be dragged, so the same
//! arithmetic is a function of the surface size instead, and this module depends
//! on nothing.

use super::world::CELL_SIZE;

/// Baseline magnification, so the world reads at a decent size on a small
/// display rather than being a field of 5px specks.
const ZOOM_MIN: f32 = 2.0;

/// Cap on how much world is on screen at once, horizontally, in cells.
///
/// The streaming window is `WINDOW_COLS x WINDOW_ROWS` (352x256) cells and
/// recenters only once the player drifts a chunk from the middle, so the visible
/// rect has to stay inside the window even at that worst-case drift. 160x100
/// visible cells leaves ample margin at full drift; exceeding it would let the
/// camera see past the generated edge.
const MAX_VIEW_CELLS_W: i32 = 160;

/// Cap on how much world is on screen at once, vertically, in cells.
const MAX_VIEW_CELLS_H: i32 = 100;

const MAX_VIEW_W: f32 = (MAX_VIEW_CELLS_W * CELL_SIZE) as f32; // 800 logical px
const MAX_VIEW_H: f32 = (MAX_VIEW_CELLS_H * CELL_SIZE) as f32; // 500 logical px

/// The logical drawing buffer: what every render pass and HUD lays out against.
///
/// The game renders into a buffer of this size and it is then upscaled to the
/// window with nearest-neighbour filtering, so the logical size divided into the
/// physical size is the zoom factor: a smaller logical buffer shows less world
/// across the same screen, i.e. zooms in. Zooming this way leaves `CELL_SIZE`
/// (and therefore all sim, tile and collision math) completely untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Physical pixels per logical pixel.
    pub zoom: f32,
    /// Logical buffer width, in world px.
    pub w: i32,
    /// Logical buffer height, in world px.
    pub h: i32,
}

impl View {
    /// Fit the logical buffer to a physical window.
    ///
    /// On a display too large to satisfy the cell cap at `ZOOM_MIN` we zoom in
    /// further rather than showing unstreamed world.
    ///
    /// The buffer is rounded to whole px because it is upscaled with hard pixel
    /// edges, and a fractional buffer edge would resample the whole frame.
    pub fn for_screen(screen_w: u32, screen_h: u32) -> View {
        let sw = screen_w.max(1) as f32;
        let sh = screen_h.max(1) as f32;
        let zoom = ZOOM_MIN.max(sw / MAX_VIEW_W).max(sh / MAX_VIEW_H);
        View {
            zoom,
            w: (sw / zoom).round() as i32,
            h: (sh / zoom).round() as i32,
        }
    }

    /// Visible width in cells, rounded up, plus the sub-cell scroll margin.
    pub fn cells_w(&self) -> i32 {
        (self.w + CELL_SIZE - 1) / CELL_SIZE + 2
    }

    /// Visible height in cells, rounded up, plus the sub-cell scroll margin.
    pub fn cells_h(&self) -> i32 {
        (self.h + CELL_SIZE - 1) / CELL_SIZE + 2
    }
}

impl Default for View {
    /// The size the TypeScript build fell back to when there was no `window` —
    /// kept as the headless default so benches and tests have a defined viewport.
    fn default() -> Self {
        View::for_screen(1000, 500)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::world::{WINDOW_COLS, WINDOW_ROWS};

    #[test]
    fn the_visible_rect_never_escapes_the_streaming_window() {
        // Even at a full chunk of drift from the window centre, the camera must
        // not see past the generated edge.
        for (w, h) in [
            (1000, 500),
            (1440, 900),
            (2560, 1440),
            (3840, 2160),
            (640, 480),
        ] {
            let v = View::for_screen(w, h);
            assert!(
                v.cells_w() <= MAX_VIEW_CELLS_W + 2,
                "{w}x{h} too wide: {}",
                v.cells_w()
            );
            assert!(
                v.cells_h() <= MAX_VIEW_CELLS_H + 2,
                "{w}x{h} too tall: {}",
                v.cells_h()
            );
            assert!(v.cells_w() < WINDOW_COLS);
            assert!(v.cells_h() < WINDOW_ROWS);
            assert!(v.zoom >= ZOOM_MIN);
        }
    }

    #[test]
    fn the_headless_default_matches_the_old_fallback() {
        let v = View::default();
        assert_eq!(v.zoom, 2.0);
        assert_eq!((v.w, v.h), (500, 250));
    }
}
