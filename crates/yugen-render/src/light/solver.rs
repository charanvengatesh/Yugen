//! The solver: the light grid, the blur that spreads it, and the scan that
//! feeds it.
//!
//! [`super`]'s "The solver" and "The census scan". This is the hot half of the
//! module — a downscaled grid, a separable blur, and a per-cell walk that runs
//! every frame — and it is deliberately plain arithmetic over buffers.
//!
//! No Bevy, which is what makes `light_blur_matches_cpu` and
//! `shader_matches_cpu` legible as CPU-oracle tests rather than as tests that
//! happen to link a renderer: the oracle they compare the GPU against is this
//! file, and it can be called directly.

use yugen_core::config::{CELL_SIZE, LIGHT_DOWNSCALE, SURFACE_ANCHOR_Y, View, WorldScale};
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, EMPTY, MAT_COLLIDE, MAT_LIGHT, Tag, has_tags};
use yugen_core::sim::noise::Noise;
use yugen_core::sim::worldgen::heightmap::Heightmap;
use yugen_core::sim::worldgen::world_noise;

use super::model::*;

// --- The solver --------------------------------------------------------------

/// The frame-varying inputs to one solve.
///
/// Bundled rather than passed loose because every one of them is read by two or
/// three of the passes and the alternative is the same four arguments threaded
/// through four signatures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightFrame {
    /// Light-cell x of the grid's left edge, absolute.
    pub ox: i32,
    /// Light-cell y of the grid's top edge, absolute.
    pub oy: i32,
    /// The world clock's daylight weight, 0 night to 1 noon.
    ///
    /// It scales the open-sky flood ONLY: caves are already dark and should not
    /// get darker, but the surface must actually go dim at night for the cycle
    /// to read.
    pub day: f32,
    /// Seconds, for the flicker phase. Frame-constant.
    pub t: f32,
}

/// The light solver: four grids of floats over the visible world.
///
/// Bevy-free by construction. Everything the frame touches is allocated once in
/// [`LightGrid::new`]; `solve` refills the buffers and allocates nothing.
pub struct LightGrid {
    /// Grid width in light cells (view plus a one-cell margin each side).
    pub(super) lw: i32,
    /// Grid height in light cells.
    pub(super) lh: i32,

    /// Scalar light per light cell, 0..1.
    pub(super) light: Vec<f32>,
    /// Ping-pong scratch for the separable blur.
    pub(super) scratch: Vec<f32>,
    /// One running column sum per grid column, for the blur's vertical pass.
    pub(super) acc: Vec<f32>,

    /// COLOURED light, accumulated separately from the scalar grid.
    ///
    /// The scalar grid drives the darkness MULTIPLY (how much of the scene
    /// survives at all); these three drive an ADDITIVE pass on top of it.
    /// Keeping them apart is what makes a torch-lit cave look different from a
    /// daylit surface rather than merely brighter: skylight contributes only to
    /// the scalar grid, so open ground is revealed in its own colours, while an
    /// emitter contributes to both, so rock near lava is revealed AND washed
    /// orange. One grid could not express that difference.
    pub(super) lr: Vec<f32>,
    pub(super) lg: Vec<f32>,
    pub(super) lb: Vec<f32>,

    /// Whether anything wrote into the colour grids this frame.
    ///
    /// Residual HEAT casts colour without being an emitter, so the hot list
    /// alone would miss a wall that a fire has been licking; this flag is the
    /// honest answer to "is the coloured pass worth blurring and uploading".
    pub(super) colour_dirty: bool,

    /// Per-light-cell dedup mask for the census splat, cleared per frame.
    pub(super) seen: Vec<bool>,

    /// Emissive cells the last solve found, in absolute cells, capped at
    /// [`HOT_MAX`].
    ///
    /// The ambience layer seeds its embers from this instead of scanning the
    /// grid again: this pass already visits every light cell, so the list is a
    /// by-product that costs one push per hit. One frame stale, which nobody can
    /// see.
    pub(super) hot: Vec<[i32; 2]>,

    /// The surface heightmap, for seeding a column that starts underground.
    pub(super) heights: Heightmap,
    /// The noise the heightmap is evaluated against.
    pub(super) noise: Noise,
    /// The world seed [`LightGrid::heights`] and [`LightGrid::noise`] describe.
    /// See [`LightGrid::follow_seed`].
    pub(super) seed: u32,
}

impl LightGrid {
    /// A grid sized for `view`, reading the surface line of world `seed`.
    pub fn new(view: View, seed: u32) -> LightGrid {
        let (lw, lh) = grid_size(view);
        let n = (lw * lh) as usize;
        LightGrid {
            lw,
            lh,
            light: vec![0.0; n],
            scratch: vec![0.0; n],
            acc: vec![0.0; lw.max(1) as usize],
            lr: vec![0.0; n],
            lg: vec![0.0; n],
            lb: vec![0.0; n],
            colour_dirty: false,
            seen: vec![false; n],
            hot: Vec::with_capacity(HOT_MAX),
            heights: Heightmap::new(),
            noise: world_noise(seed),
            seed,
        }
    }

    /// Rebuild the climate fields if the world underneath has changed.
    ///
    /// # Why this exists when nothing can currently call it usefully
    ///
    /// This grid keeps its own `Heightmap` and `Noise` so the skylight flood can
    /// seed a column that starts underground. `crate::ambience` keeps a second
    /// copy for its own reasons, and has always guarded itself this way; this one
    /// did not. It was built from the compile-time `SEED` at both production
    /// sites and never read `SimWorld::seed`.
    ///
    /// Nothing is wrong on screen today, because `build_world` is only ever
    /// called with that same constant. But it takes a seed — the signature is an
    /// open invitation — and the failure mode on the day someone adds a `--seed`
    /// flag or a new-world menu is silent: the light solver would flood sky down
    /// to one surface line while the terrain sat at another, and no test would
    /// notice. That is precisely the shape of the three bugs in `HANDOFF.md`
    /// §7.1, all of which degraded to plausible output.
    ///
    /// Only the seeded fields are rebuilt; the buffers do not depend on the seed.
    ///
    /// # The heightmap is REPLACED, and that is not belt-and-braces
    ///
    /// `Heightmap` invalidates its 4096-slot memo by comparing the ADDRESS of the
    /// `Noise` it is handed against the one its live entries belong to
    /// (`retire_if_new_seed`) — an O(1) trick that is exactly right for the
    /// worldgen path, where a new world means a new `ChunkGen` and so a new
    /// `Noise` at a new address.
    ///
    /// It is a trap here. `self.noise = world_noise(seed)` overwrites a field in
    /// place, so the address does not move, so the memo would go on serving the
    /// previous world's surface rows forever. Handing the new noise to a fresh
    /// `Heightmap` is what makes the invalidation fire.
    pub fn follow_seed(&mut self, seed: u32) {
        if self.seed != seed {
            self.seed = seed;
            self.noise = world_noise(seed);
            self.heights = Heightmap::new();
        }
    }

    /// The world seed the climate fields were built from.
    #[inline]
    pub const fn seed(&self) -> u32 {
        self.seed
    }

    /// Grid width in light cells.
    #[inline]
    pub const fn cols(&self) -> i32 {
        self.lw
    }

    /// Grid height in light cells.
    #[inline]
    pub const fn rows(&self) -> i32 {
        self.lh
    }

    /// Scalar light at a light cell, or 0 outside the grid.
    #[inline]
    pub fn light_at(&self, lx: i32, ly: i32) -> f32 {
        self.index(lx, ly).map_or(0.0, |i| self.light[i])
    }

    /// Coloured light at a light cell, or black outside the grid.
    #[inline]
    pub fn colour_at(&self, lx: i32, ly: i32) -> [f32; 3] {
        self.index(lx, ly)
            .map_or([0.0; 3], |i| [self.lr[i], self.lg[i], self.lb[i]])
    }

    /// Whether anything emitted into the colour grids on the last solve.
    #[inline]
    pub const fn colour_dirty(&self) -> bool {
        self.colour_dirty
    }

    /// Absolute cells of the emitters the last solve found.
    #[inline]
    pub fn hot(&self) -> &[[i32; 2]] {
        &self.hot
    }

    #[inline]
    fn index(&self, lx: i32, ly: i32) -> Option<usize> {
        if lx < 0 || ly < 0 || lx >= self.lw || ly >= self.lh {
            None
        } else {
            Some((ly * self.lw + lx) as usize)
        }
    }

    /// One frame's light: skylight, then emitters, then the census, then blur.
    ///
    /// The order is not arbitrary. Skylight WRITES the scalar grid — every cell,
    /// unconditionally, which is what makes it the only pass that needs no
    /// clear. The two emissive passes ADD to it. The blur is last because it is
    /// what turns a cross-shaped splat into a glow.
    ///
    /// `census_x` / `census_y` are the absolute cells of every emitter in view,
    /// shaped to take [`crate::cells::EmitterCensus`]'s two accessors directly.
    /// Pass empty slices to solve without one — the result is correct for bulk
    /// emitters and blind to single-cell ones, which is the whole reason the
    /// census exists.
    pub fn solve(
        &mut self,
        grid: &CellGrid,
        frame: LightFrame,
        census_x: &[i32],
        census_y: &[i32],
    ) {
        self.compute_skylight(grid, frame);
        self.add_emissive(grid, frame);
        self.add_census_emitters(grid, census_x, census_y, frame);
        self.blur();
    }

    /// Top-down skylight flood.
    ///
    /// Each column starts at full brightness above the world (open sky) and
    /// carries a running light value downward: open cells keep most of it, solid
    /// cells swallow it fast, so light attenuates behind and beneath rock and
    /// caves fall dark. One pass, column-major.
    pub fn compute_skylight(&mut self, grid: &CellGrid, frame: LightFrame) {
        let (lw, lh) = (self.lw, self.lh);
        let step = LIGHT_DOWNSCALE;
        let half = step / 2;
        let sky = SKY_NIGHT + SKY_DAY_GAIN * frame.day;
        let top_cy = frame.oy * step + half;

        // Resolve the window origin once and index the material plane directly:
        // `get_world` is a bounds test plus a translation per sample, and this
        // loop samples every light cell in the viewport.
        let gx0 = grid.origin_cell_x();
        let gy0 = grid.origin_cell_y();
        let cols = grid.cols();
        let rows = grid.rows();

        for lx in 0..lw {
            let cx = (frame.ox + lx) * step + half; // sample cell centre
            let below_surface = top_cy
                - self
                    .heights
                    .surface_row_at(&self.noise, cx, None, WorldScale::LIVE);
            let mut carry = sky
                * if below_surface <= 0 {
                    1.0
                } else {
                    SEED_DECAY.powi(below_surface).min(1.0)
                };

            let gx = cx - gx0;
            let in_col = gx >= 0 && gx < cols;
            for ly in 0..lh {
                let gy = (frame.oy + ly) * step + half - gy0;
                // Unloaded cells read as air, exactly as `get_world` reports them.
                let loaded = in_col && gy >= 0 && gy < rows;
                let at = if loaded { (gy * cols + gx) as usize } else { 0 };
                let solid = loaded && MAT_COLLIDE[grid.material[at] as usize] == 1;
                // Open air with a wall behind it. Only asked where the front is
                // not solid, because rock in front of a wall is just rock.
                //
                // FLORA behind you is not a wall. A share of trees are drawn into
                // the background plane so the player can walk through them, and
                // counting their canopies as rock put three quarters of the sky
                // cells in a wood under `WALL_DECAY` — a forest lit like a cave,
                // in broad daylight. A trunk you can see daylight past does not
                // occlude, which is the same reason leaves do not collide.
                let behind = if loaded { grid.back[at] } else { EMPTY };
                let walled = loaded && !solid && behind != EMPTY && !has_tags(behind, Tag::FLORA);

                // STORE, THEN DECAY. A cell is lit by the light that REACHES it;
                // the occlusion it causes applies to what is behind it, not to
                // itself. Decaying first meant the topmost solid sample in every
                // column — the sunlit ground you are standing on — was already
                // darkened by 45% before it was ever written, so at noon the
                // surface came out the same value as rock four cells under it
                // and the whole world read as overcast dusk. The lit face of the
                // terrain is the thing the player looks at most; it has to be
                // the brightest solid in its column, and this is the ordering
                // that makes it so.
                self.light[(ly * lw + lx) as usize] = carry;
                carry *= if solid {
                    SOLID_DECAY
                } else if walled {
                    WALL_DECAY
                } else {
                    OPEN_DECAY
                };
            }
        }
    }

    /// Smear emissive cells into the light grid so their pockets read as lit.
    ///
    /// One additive splat per emissive light cell with a small cross spread; the
    /// blur afterwards rounds it off. Two things make the glow LIVE rather than
    /// sit flat:
    ///
    ///   - a per-cell flicker on a phase hashed from the cell's absolute coords,
    ///     so neighbouring pockets breathe out of step instead of pulsing as one
    ///     slab;
    ///   - the heat field: a cell that is merely HOT (rock next to lava, a wall
    ///     a fire has been licking) contributes a weak glow of its own, so heat
    ///     visibly spreads into the surroundings and fades as it dissipates.
    ///
    /// The heat read is free: this loop already resolves the cell's flat index,
    /// so the temperature is one extra load per light cell.
    ///
    /// This is also the pass that CLEARS the colour grids and the hot list, so
    /// it must run before the census pass and not after.
    pub fn add_emissive(&mut self, grid: &CellGrid, frame: LightFrame) {
        let (lw, lh) = (self.lw, self.lh);
        let step = LIGHT_DOWNSCALE;
        let half = step / 2;

        self.lr.fill(0.0);
        self.lg.fill(0.0);
        self.lb.fill(0.0);
        self.colour_dirty = false;
        self.hot.clear();

        let gx0 = grid.origin_cell_x();
        let gy0 = grid.origin_cell_y();
        let cols = grid.cols();
        let rows = grid.rows();

        for ly in 0..lh {
            let cy = (frame.oy + ly) * step + half;
            let gy = cy - gy0;
            if gy < 0 || gy >= rows {
                continue;
            }
            let row = gy * cols;

            for lx in 0..lw {
                let cx = (frame.ox + lx) * step + half;
                let gx = cx - gx0;
                if gx < 0 || gx >= cols {
                    continue;
                }

                let gi = (row + gx) as usize;
                let e = emitter(grid.material[gi]);
                if e.emits() {
                    self.splat_emitter(e, (lx, ly), (cx, cy), frame.t, &FAR_RING);
                    continue;
                }

                let heat = f32::from(grid.temp[gi]);
                if heat > HEAT_THRESHOLD {
                    let v = ((heat - HEAT_THRESHOLD) / HEAT_RANGE) * HEAT_GAIN;
                    splat(&mut self.light, lw, lh, lx, ly, v, v * HEAT_EDGE);
                    self.splat_rgb(lx, ly, [v * HEAT_RGB[0], v * HEAT_RGB[1], v * HEAT_RGB[2]]);
                }
            }
        }
    }

    /// Splat the emitters the census found, deduplicated per light cell.
    ///
    /// A downscaled sampling loop catches BULK emitters — a lava lake fills
    /// every block it is sampled in — but by construction cannot see a one-cell
    /// torch, because it only looks at one cell in `LIGHT_DOWNSCALE` squared.
    /// This pass covers that gap. At a downscale of 1 there is no gap and no
    /// census is taken at all; see [`CENSUS_NEEDED`] for why feeding this one
    /// anyway would double every emitter rather than add nothing.
    ///
    /// Deduplicated per LIGHT CELL: several census entries commonly land in the
    /// same one (a 6-cell campfire, a torch beside a lantern), and splatting
    /// each of them would stack the same glow three or four times and blow out
    /// to white.
    ///
    /// The census reports WHERE, not WHAT, so the material is read back out of
    /// `grid`. A cell that stopped emitting between the scan and the splat
    /// therefore contributes nothing rather than a ghost.
    pub fn add_census_emitters(
        &mut self,
        grid: &CellGrid,
        census_x: &[i32],
        census_y: &[i32],
        frame: LightFrame,
    ) {
        let n = census_x.len().min(census_y.len());
        if n == 0 {
            return;
        }
        let step = LIGHT_DOWNSCALE;
        self.seen.fill(false);

        for i in 0..n {
            let (cx, cy) = (census_x[i], census_y[i]);
            let lx = cx.div_euclid(step) - frame.ox;
            let ly = cy.div_euclid(step) - frame.oy;
            let Some(li) = self.index(lx, ly) else {
                continue;
            };
            if self.seen[li] {
                continue;
            }
            self.seen[li] = true;

            let e = emitter(cell_at_world(grid, cx, cy));
            if !e.emits() {
                continue;
            }
            self.splat_emitter(e, (lx, ly), (cx, cy), frame.t, &FAR_RING_AXIAL);
        }
    }

    /// The scalar and coloured splat one emitter makes, plus its far ring.
    ///
    /// `ring` is what the two callers differ by and nothing else — see
    /// [`FAR_RING_AXIAL`] for why they differ at all.
    fn splat_emitter(
        &mut self,
        e: Emitter,
        (lx, ly): (i32, i32),
        (cx, cy): (i32, i32),
        t: f32,
        ring: &[(i32, i32)],
    ) {
        let (lw, lh) = (self.lw, self.lh);
        let v = e.level * flicker(t, cx, cy);
        splat(&mut self.light, lw, lh, lx, ly, v, v * SPLAT_EDGE);
        if e.level > FAR_SPLAT_LEVEL {
            let far = v * FAR_SPLAT_GAIN;
            for (dx, dy) in ring {
                add(&mut self.light, lw, lh, lx + dx, ly + dy, far);
            }
        }
        let cv = v * COLOUR_GAIN;
        self.splat_rgb(lx, ly, [e.rgb[0] * cv, e.rgb[1] * cv, e.rgb[2] * cv]);
        if self.hot.len() < HOT_MAX {
            self.hot.push([cx, cy]);
        }
    }

    /// Additive centre plus 4-neighbour spread into the three colour grids.
    fn splat_rgb(&mut self, lx: i32, ly: i32, rgb: [f32; 3]) {
        let (lw, lh) = (self.lw, self.lh);
        self.colour_dirty = true;
        for (plane, c) in [&mut self.lr, &mut self.lg, &mut self.lb]
            .into_iter()
            .zip(rgb)
        {
            splat(plane, lw, lh, lx, ly, c, c * COLOUR_EDGE);
        }
    }

    /// Two separable passes over the scalar grid and the three colour grids.
    ///
    /// The colour grids are only blurred when something actually emitted this
    /// frame. Standing on the surface in daylight — the common case — that is a
    /// single `fill`-cleared scan away from free, and the three extra blurs
    /// never run at all.
    pub fn blur(&mut self) {
        let (lw, lh) = (self.lw, self.lh);
        blur_one(&mut self.light, &mut self.scratch, &mut self.acc, lw, lh);
        if self.colour_dirty {
            blur_one(&mut self.lr, &mut self.scratch, &mut self.acc, lw, lh);
            blur_one(&mut self.lg, &mut self.scratch, &mut self.acc, lw, lh);
            blur_one(&mut self.lb, &mut self.scratch, &mut self.acc, lw, lh);
        }
    }

    /// Bake the darkness pass into RGBA multiply factors.
    ///
    /// # The algebra
    ///
    /// Canvas2D drew an RGBA whose alpha was the darkness with
    /// `globalCompositeOperation = "multiply"`, which is a source-over composite
    /// of a multiply blend:
    ///
    /// ```text
    /// out = (1 - a) * dst + a * (shadow * dst)
    ///     = dst * (1 - a + a * shadow)
    /// ```
    ///
    /// The bracket depends only on this module's own numbers, so it is evaluated
    /// here, per light cell, and written into RGB. The GPU then needs nothing
    /// but `dst * src` — see [`BLEND_MULTIPLY`] — and the texture is a plain
    /// non-sRGB `Rgba8Unorm` because these bytes are MULTIPLIERS and must not be
    /// transfer-function-decoded on the way in.
    ///
    /// Alpha is held at 255 so the destination's own alpha survives: the low-res
    /// buffer is sampled by [`crate::lowres`]'s blit sprite, and a pass that
    /// zeroed its alpha would erase the frame.
    ///
    /// `out` must be `cols() * rows() * 4` bytes.
    pub fn bake_shadow(&self, depth: f32, day: f32, out: &mut [u8]) {
        let shadow = shadow_tint(depth, day);
        let floor = ambient_floor(depth);
        for (px, &l) in out.chunks_exact_mut(4).zip(self.light.iter()) {
            let a = 1.0 - l.max(floor).min(1.0);
            for (dst, s) in px.iter_mut().zip(shadow) {
                *dst = unit_byte(1.0 - a + a * s);
            }
            px[3] = u8::MAX;
        }
    }

    /// Bake the coloured pass into RGBA additive increments.
    ///
    /// `"lighter"` at full alpha is `dst + src`, so the increment IS the texel
    /// and there is no algebra to do — only the clamp the original did on its
    /// way into an 8-bit `ImageData`.
    ///
    /// `out` must be `cols() * rows() * 4` bytes.
    pub fn bake_colour(&self, out: &mut [u8]) {
        for (i, px) in out.chunks_exact_mut(4).enumerate() {
            px[0] = unit_byte(self.lr[i]);
            px[1] = unit_byte(self.lg[i]);
            px[2] = unit_byte(self.lb[i]);
            // Additive blending leaves the destination alpha alone (see
            // `BLEND_ADD`), so this byte is never read. Held at 255 anyway so a
            // debug view of the texture is not an invisible rectangle.
            px[3] = u8::MAX;
        }
    }
}

/// Light-grid size for a view: the visible rect, rounded up, plus a one-cell
/// margin each side.
///
/// The margin is what covers the camera sitting BETWEEN two light cells: the
/// grid's origin is floored to the lattice, so the view can hang up to one light
/// cell off each far edge, and without the spare column there the composite quad
/// would stop short of the screen. It also gives the upscale a value to
/// interpolate toward rather than clamping at the frame edge — a second reason
/// that used to be the stated one, and that now covers [`LIGHT_SOFTNESS`] of a
/// cell rather than a whole 20 px texel.
pub fn grid_size(view: View) -> (i32, i32) {
    let stride = light_stride_px();
    let ceil = |px: i32| (px.max(0) + stride - 1) / stride + 2;
    (ceil(view.w), ceil(view.h))
}

/// World px covered by one light cell on each axis.
pub const fn light_stride_px() -> i32 {
    CELL_SIZE * LIGHT_DOWNSCALE
}

/// Additive centre plus four arms at [`SPLAT_REACH`], clamped to 1.
///
/// Five writes whatever the stride: the arms move further out in light cells as
/// the grid gets finer, but there are still four of them. What fills the gap
/// between the centre and an arm is [`blur_one`], whose radius is the same
/// distance — that pairing is why a cross of five deltas reads as a glow and not
/// as five dots, and it is why the two reaches have to move together.
pub(super) fn splat(buf: &mut [f32], lw: i32, lh: i32, lx: i32, ly: i32, core: f32, edge: f32) {
    add(buf, lw, lh, lx, ly, core);
    add(buf, lw, lh, lx - SPLAT_REACH, ly, edge);
    add(buf, lw, lh, lx + SPLAT_REACH, ly, edge);
    add(buf, lw, lh, lx, ly - SPLAT_REACH, edge);
    add(buf, lw, lh, lx, ly + SPLAT_REACH, edge);
}

/// Clamped additive write into a light cell. Out of range is a no-op.
pub(super) fn add(buf: &mut [f32], lw: i32, lh: i32, x: i32, y: i32, v: f32) {
    if x < 0 || y < 0 || x >= lw || y >= lh {
        return;
    }
    let i = (y * lw + x) as usize;
    buf[i] = (buf[i] + v).min(1.0);
}

/// A triangle blur of radius [`BLUR_REACH`], edges clamped to themselves.
///
/// # Four box passes, and why not nine taps
///
/// A triangle kernel IS a box convolved with a box, so the whole blur is four
/// runs of a sliding window: two along the rows and two down the columns. At
/// radius 1 that is `[1, 2, 1] / 4` exactly, the kernel this pass has always
/// run; at radius 4 it is `[1..5..1] / 25`. The uneven `BLUR_BACK`/`BLUR_FWD`
/// split is what keeps the pair CENTRED when the box has an even width — the
/// first pass leans forward by half a sample and the second leans back by the
/// same half, and the two cancel. Getting that wrong shifts the entire light
/// field half a cell against the art, which is the one error this arrangement
/// can make and the reason the offsets are named rather than inlined.
///
/// The obvious implementation is `2 * BLUR_REACH + 1` taps per cell per axis,
/// and it was written that way first. It costs the reach: nine taps at radius 4
/// measured **171 µs** for one frame's four grids, 80% of the entire light
/// solve, and it would have got worse the moment anyone widened the glow. A
/// sliding window is O(1) in the radius — three float ops per cell per pass
/// whatever the reach — so the look and the cost are no longer the same dial.
///
/// `scratch` is the caller's ping-pong buffer and `acc` its column accumulator,
/// one entry per grid column. Both come in dirty and go out dirty, which is the
/// whole point of them being owned by the grid rather than allocated per pass:
/// the vertical window has to carry a running sum per column, and walking the
/// grid column by column to avoid that would read every cache line `h` times.
///
/// # This is the shipping path AND an oracle, which is unusual and deliberate
///
/// `lightblur.wgsl` is a GPU implementation of exactly this kernel, and
/// `tests/light_blur_matches_cpu.rs` compiles it on a headless adapter and
/// compares the two texel for texel — agreement measured at 9.1e-4, under a
/// quarter of an 8-bit step. **The game does not run the shader.** It was built,
/// verified, wired in and measured, and wiring it in cost 237 µs of whole-frame
/// time to remove 97.9 µs of CPU work that a whole-frame measurement says is
/// worth 1 µs; `docs/PERF.md` §8.6 has the table.
///
/// So this function stays where it is, on the frame path, and the shader stays
/// beside it in the position `scan_emitters` occupies: kept, tested, correct and
/// not run. `pub` because the harness needs it — the same trade `crate::cells`'
/// `paint_cells` makes beside `cells.wgsl`, run the other way round.
pub fn blur_one(a: &mut [f32], scratch: &mut [f32], acc: &mut [f32], lw: i32, lh: i32) {
    let (w, h) = (lw as usize, lh as usize);
    box_rows(a, scratch, w, h, BLUR_BACK, BLUR_FWD);
    box_rows(scratch, a, w, h, BLUR_FWD, BLUR_BACK);
    box_cols(a, scratch, acc, w, h, BLUR_BACK, BLUR_FWD);
    box_cols(scratch, a, acc, w, h, BLUR_FWD, BLUR_BACK);
}

/// The blur's window as `(back, fwd)`, for whatever has to restate it.
///
/// Two things do: the uniform `lightblur.wgsl` reads its loop bounds from, and
/// the harness that diffs that shader against [`blur_one`]. Both get the numbers
/// from here rather than from a literal, so re-tuning [`BLUR_REACH_CELLS`] moves
/// the CPU pass, the GPU pass and the test together or moves none of them.
pub const fn blur_window() -> (u32, u32) {
    (BLUR_BACK as u32, BLUR_FWD as u32)
}

/// One box pass along the rows, window `[x - back, x + fwd]`, edges clamped.
///
/// The output is clamped to 0..1 as well as the window. A running sum adds and
/// subtracts the same values back out over a whole row, so it drifts by a few
/// ulps where a fresh sum of taps would not — enough to leave a `-1e-8` in a
/// grid every other pass in this module is entitled to assume is a fraction.
/// The clamp is two ops per cell and buys back an invariant.
pub(super) fn box_rows(src: &[f32], dst: &mut [f32], w: usize, h: usize, back: usize, fwd: usize) {
    let inv = 1.0 / (back + fwd + 1) as f32;
    let last = w as isize - 1;
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        let at = |x: isize| row[x.clamp(0, last) as usize];

        let mut sum = 0.0;
        for d in 0..=(back + fwd) {
            sum += at(d as isize - back as isize);
        }
        for (x, out) in dst[y * w..y * w + w].iter_mut().enumerate() {
            *out = (sum * inv).clamp(0.0, 1.0);
            sum += at((x + fwd + 1) as isize) - at(x as isize - back as isize);
        }
    }
}

/// One box pass down the columns, window `[y - back, y + fwd]`, edges clamped.
///
/// Row-major throughout: `acc` holds one running sum per column, so the walk is
/// still linear over both buffers and the vertical pass costs what the
/// horizontal one does.
pub(super) fn box_cols(
    src: &[f32],
    dst: &mut [f32],
    acc: &mut [f32],
    w: usize,
    h: usize,
    back: usize,
    fwd: usize,
) {
    let inv = 1.0 / (back + fwd + 1) as f32;
    let last = h as isize - 1;
    let row_at = |y: isize| (y.clamp(0, last) as usize) * w;

    acc[..w].fill(0.0);
    for d in 0..=(back + fwd) {
        let r = row_at(d as isize - back as isize);
        for (a, s) in acc[..w].iter_mut().zip(&src[r..r + w]) {
            *a += *s;
        }
    }
    for y in 0..h {
        let add = row_at((y + fwd + 1) as isize);
        let sub = row_at(y as isize - back as isize);
        let out = &mut dst[y * w..y * w + w];
        for (((o, a), plus), minus) in out
            .iter_mut()
            .zip(acc[..w].iter_mut())
            .zip(&src[add..add + w])
            .zip(&src[sub..sub + w])
        {
            *o = (*a * inv).clamp(0.0, 1.0);
            *a += *plus - *minus;
        }
    }
}

/// The per-cell emissive flicker, 0.76..1.0.
#[inline]
pub(super) fn flicker(t: f32, cx: i32, cy: i32) -> f32 {
    FLICKER_BASE + FLICKER_SWING * (t * FLICKER_RATE + hash_phase(cx, cy)).sin()
}

/// Cell coords to a stable phase in `[0, TAU)`.
///
/// An integer hash: no trig, no table — the point is only that two adjacent
/// pockets get decorrelated phases.
///
/// The first mix reproduces the original exactly. JavaScript's `*` on these
/// magnitudes is an exact `f64` product and `>>> 0` is a wrap to `u32`, which is
/// what an `i64` product cast to `u32` does. The SECOND mix does not, and cannot
/// — `(h ^ (h >>> 13)) * 1274126177` overflows `f64`'s 53-bit mantissa, so the
/// original was quietly rounding before it wrapped. `wrapping_mul` is the
/// operation that line was written to express, and the value it produces is a
/// decorrelation phase that nothing compares against anything.
#[inline]
pub(super) fn hash_phase(x: i32, y: i32) -> f32 {
    let mut h = (i64::from(x) * 374_761_393 + i64::from(y) * 668_265_263) as u32;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (f32::from((h & 0xffff) as u16) / 65536.0) * std::f32::consts::TAU
}

/// The colour an unlit surface falls toward, 0..1 per channel.
///
/// The shadow's HUE moves with depth and time of day, and that single change
/// does most of the work of making the underground feel like a different place.
/// Near the surface an unlit face is in shadow but still lit by the sky, so its
/// shadow is blue; five hundred cells down there is no sky to bounce, so the
/// shadow goes to a dead neutral that swallows colour. At night the surface
/// shadow cools and deepens toward the same place.
pub fn shadow_tint(depth: f32, day: f32) -> [f32; 3] {
    let night = 1.0 - day;
    let mut out = [0.0f32; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let v =
            SHADOW_SKY[i] + (SHADOW_DEEP[i] - SHADOW_SKY[i]) * depth - night * SHADOW_NIGHT_COOL[i];
        // Floored to a whole 0..255 byte before use, as the original did: the
        // shadow hue is an 8-bit colour, and carrying more precision than that
        // into the multiply would put a value in it that no pixel could hold.
        *o = v.max(0.0).trunc() / 255.0;
    }
    out
}

/// The light a cave keeps however unlit it is.
pub fn ambient_floor(depth: f32) -> f32 {
    AMBIENT_FLOOR - depth * AMBIENT_DEPTH_FALL
}

/// Depth below the surface anchor, 0 at the surface to 1 deep.
///
/// Measured in ABSOLUTE cells from the view's centre so it does not jump as the
/// streaming window scrolls. The original derived the centre from the view's top
/// edge plus half its height; [`WorldFocus`] is already the centre, so the
/// arithmetic is one term shorter and says the same thing.
pub fn depth_at(centre_y_px: f32) -> f32 {
    let cells = centre_y_px / CELL_SIZE as f32 - SURFACE_ANCHOR_Y as f32;
    (cells / DEPTH_RANGE_CELLS).clamp(0.0, 1.0)
}

/// A 0..1 float as a 0..255 byte, clamped and truncated.
#[inline]
pub(super) fn unit_byte(v: f32) -> u8 {
    if v >= 1.0 {
        u8::MAX
    } else if v <= 0.0 {
        0
    } else {
        (v * 255.0) as u8
    }
}

/// Material at an absolute cell, or air if the window does not hold it.
pub(super) fn cell_at_world(grid: &CellGrid, cx: i32, cy: i32) -> CellId {
    let (gx, gy) = (cx - grid.origin_cell_x(), cy - grid.origin_cell_y());
    if gx < 0 || gy < 0 || gx >= grid.cols() || gy >= grid.rows() {
        EMPTY
    } else {
        grid.material[(gy * grid.cols() + gx) as usize]
    }
}

// --- The census scan ---------------------------------------------------------

/// Absolute cells of every emitting cell in a rect, deduplicated and capped.
///
/// See the module header for why this walk is paid here rather than taken as a
/// by-product of the cell blit. `x` and `y` are parallel and always the same
/// length, so they hand to [`LightGrid::add_census_emitters`] in exactly the
/// shape [`crate::cells::EmitterCensus`] publishes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitterScan {
    x: Vec<i32>,
    y: Vec<i32>,
}

impl EmitterScan {
    /// Absolute cell X of each emitter found.
    pub fn x(&self) -> &[i32] {
        &self.x
    }
    /// Absolute cell Y of each emitter found.
    pub fn y(&self) -> &[i32] {
        &self.y
    }
    /// How many were found, at most [`EMIT_CENSUS_MAX`].
    pub fn len(&self) -> usize {
        self.x.len()
    }
    /// Whether the rect contained no emitters at all.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Walk every cell of the rect whose top-left is absolute cell `(ox, oy)` and
/// record the emitters, at most one per [`CENSUS_GROUP`] cells per row.
///
/// Refills `out` rather than returning a fresh one: this runs every frame over
/// up to ~16 000 cells, and those two vectors are the only allocation the whole
/// light pass would otherwise make.
pub fn scan_emitters(grid: &CellGrid, ox: i32, oy: i32, w: i32, h: i32, out: &mut EmitterScan) {
    out.x.clear();
    out.y.clear();

    let gx0 = grid.origin_cell_x();
    let gy0 = grid.origin_cell_y();
    let (cols, rows) = (grid.cols(), grid.rows());

    // Clip to the loaded window once, per axis, rather than testing every cell:
    // outside it there is nothing to find.
    let x0 = ox.max(gx0);
    let x1 = (ox + w).min(gx0 + cols);
    let y0 = oy.max(gy0);
    let y1 = (oy + h).min(gy0 + rows);

    for cy in y0..y1 {
        let row = (cy - gy0) * cols;
        let mut last_group = i32::MIN;
        for cx in x0..x1 {
            if MAT_LIGHT[grid.material[(row + cx - gx0) as usize] as usize] == 0 {
                continue;
            }
            let group = cx.div_euclid(CENSUS_GROUP);
            if group == last_group {
                continue;
            }
            last_group = group;
            out.x.push(cx);
            out.y.push(cy);
            if out.x.len() >= EMIT_CENSUS_MAX {
                return;
            }
        }
    }
}
