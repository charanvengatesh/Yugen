//! Procedural worldgen, a pure function of ABSOLUTE cell coordinates (wcx, wcy —
//! both may be negative and unbounded). Terrain is placed by positional noise
//! thresholds and positional hashes only, never by rand() call order, so a chunk
//! generated in isolation matches its neighbour's edge regardless of load order —
//! the property that makes 2D streaming seamless and deterministic, and the one
//! the chunk persistence layer leans on when it regenerates a pristine chunk.
//!
//! CHUNK INDEPENDENCE, precisely: every decision below is a function of
//! (wcx, wcy, seed) via continuous noise fields or a stateless coordinate hash.
//! Nothing consults "what did the previous column decide" — biome BLENDING in
//! particular is driven by continuous climate distances (see `biomes`), not by
//! comparing a column against its neighbour's chosen biome, which is the classic
//! way to get chunk-aligned seams. The coarse-lattice optimisation in
//! `worldgen::caves` obeys the same rule: its lattice is anchored to WORLD space
//! with a stride that divides CHUNK_CELLS, so two chunks sharing an edge
//! interpolate from identical corner samples.
//!
//! This file is the ORCHESTRATOR only. The generator proper lives in the sibling
//! modules:
//!
//!   `spline`     monotone piecewise-linear curves — noise value → landform
//!   `fields`     the low-frequency world fields (continentalness, erosion,
//!                peaks/valleys, weirdness) and shaping helpers
//!   `heightmap`  the surface row: splines + domain warp + terracing, memoised
//!   `caves`      cheese chambers, tunnel networks, ravines, the liquid table
//!   `layers`     depth banding, cap/shore, rock, veins, strata, underworld
//!
//! The column scan below is deliberately three straight-line passes so the cheap
//! cases stay cheap: sky and sea need no field evaluation at all, the cap band
//! needs no cave evaluation unless the column can host a surface chasm, and a
//! chunk with no underground cells never even fills the cave lattice.
//!
//! # What changed in the port
//!
//! The TypeScript kept the per-chunk scratch (`COL_PROFILE`, `COL_CAVE`,
//! `COL_SURF`, `COL_SHORE`, the cave lattice, the heightmap memo and the cached
//! `Noise`) at MODULE scope, so `generateChunk(cx, cy, seed)` allocated nothing
//! but its output. That shape is single-threaded by construction. Here the same
//! scratch is OWNED by a [`ChunkGen`], one per worker: there are no statics with
//! interior mutability and no thread locals anywhere in worldgen, so
//! [`generate_chunk`] is callable from a rayon pool and N workers generating N
//! chunks produce exactly what one worker generating them in sequence would.
//! `tests/worldgen_purity.rs` asserts that rather than assuming it.

use crate::config::{
    BODY_SCALE, CELL_SIZE, CHUNK_CELLS, DEEP_DEPTH, PLAYER_H, PLAYER_W, SEA_LEVEL_Y, WorldScale,
};
use crate::sim::biomes::{ColumnProfile, column_profile_at};
use crate::sim::decor::structures::StructureDecorator;
use crate::sim::decor::trees::TreeDecorator;
use crate::sim::decor::{DecorContext, Decorator, ores::OreDecorator};
use crate::sim::materials::{CellId, MAT_COLLIDE, block};
use crate::sim::noise::Noise;

use super::caves::{Carve, CaveColumn, CaveLattice, carve_exact, cave_column_at, strata_exact};
use super::features::LANDMARK_DECORATOR;
use super::heightmap::{Heightmap, shore_weight_at};
use super::layers::{AIR, cap_at, liquid_at, solid_at, ug_fade_at};

// Material codes resolved once (the registry is fixed at compile time — the
// TypeScript paid a `codeOf` lookup at module load for the same thing).
const WATER: CellId = block::WATER;

/// The shared deterministic noise for a seed — for biome/atmosphere sampling.
///
/// The TypeScript memoised one instance per seed in a module-level pair of
/// `let`s, because building the permutation tables is not free and every
/// `generateChunk` call went through it. Here the instance is owned by whoever
/// needs it — [`ChunkGen`] holds one for the life of a world — so this is a
/// plain constructor and callers keep the result rather than re-asking.
pub fn world_noise(seed: u32) -> Noise {
    Noise::new(seed)
}

// --- Decorations -------------------------------------------------------------
// Everything non-columnar (trees, ores, structures) lives behind the decorator
// registry in `sim::decor`. Each pass authors from a positional origin and every
// chunk it can touch recomputes it independently — see `decor` for why that
// rule is not optional.
// Order is load-bearing. Trees and ores are terrain dressing and go first;
// structures are built things and overwrite them. `landmarks` runs LAST because
// its features are landform-scale (an island is terrain, a lake replaces it, a
// mineshaft cuts through it) and its templates are the largest built things in
// the world — both should win over a tree that grew where they now stand.
pub const DECORATORS: [&dyn Decorator; 4] = [
    &TreeDecorator,
    &OreDecorator,
    &StructureDecorator,
    &LANDMARK_DECORATOR,
];

/// Terrain material at an absolute cell — the ARBITRARY-COORDINATE PROBE.
///
/// [`ChunkGen::generate`] does NOT go through here: it runs the same decisions
/// against a per-chunk coarse lattice, which is an order of magnitude cheaper
/// per cell. This path evaluates every low-frequency field exactly instead, so
/// it works anywhere without a chunk loaded — which is what a feature or
/// structure pass needs when it probes "what is at (x, y)?" outside the chunk it
/// is writing.
///
/// The two can disagree by at most the lattice interpolation error, and only on
/// cells within a hair of a threshold. `generate_chunk` is the authority on what
/// the terrain IS; treat this as an oracle for placement, and have feature
/// passes overwrite what they find rather than assume it.
pub fn material_at(
    noise: &Noise,
    wcx: i32,
    wcy: i32,
    col: &ColumnProfile,
    surf: i32,
    scale: WorldScale,
) -> CellId {
    let depth = wcy - surf;
    if depth < 0 {
        return if wcy >= scale.row(SEA_LEVEL_Y) {
            WATER
        } else {
            AIR
        };
    }

    let cc = cave_column_at(noise, wcx, surf, col, scale);

    if scale.depth(f64::from(depth)) < col.cap_thickness {
        if cc.breaches && carve_exact(noise, wcx, wcy, depth, &cc, scale) != Carve::Solid {
            return AIR;
        }
        return cap_at(
            noise,
            wcx,
            wcy,
            depth,
            col,
            shore_weight_at(surf, scale),
            scale,
        );
    }

    let c = carve_exact(noise, wcx, wcy, depth, &cc, scale);
    if c == Carve::Air {
        return AIR;
    }
    let u = ug_fade_at(depth, col, scale);
    let bd = band_depth(depth, &cc, scale);
    if c != Carve::Solid {
        return liquid_at(noise, wcx, wcy, bd, col, u, scale);
    }
    let strata = if bd >= f64::from(DEEP_DEPTH) {
        strata_exact(noise, wcx, wcy, scale)
    } else {
        0.0
    };
    solid_at(noise, wcx, wcy, bd, col, u, strata, scale)
}

/// Material is selected by BAND DEPTH, the same shifted depth `caves` carves
/// against, or the rock would change band at a different line from the caves and
/// every interface would show as a double edge.
///
/// FRACTIONAL, and it has to stay that way. The obvious port is to floor it, on
/// the grounds that every band test in `layers` is `bd >= T` against an integer
/// `T` and `floor(bd) >= T ⟺ bd >= T`. That is true of the tests and false of the
/// module: `solid_at` also reads the band depth CONTINUOUSLY, in the ramp that
/// hardens the crust to obsidian over the last 45 cells above `UNDERWORLD_FLOOR`.
/// Flooring there quantises a dither probability that is only 1/45 per cell wide,
/// and it shows up as roughly one cell in 3000 of the underworld crust taking the
/// wrong rock. (Measured: it was the only disagreement between this port and the
/// TypeScript over 226 304 cells.)
#[inline]
fn band_depth(depth: i32, cc: &CaveColumn, scale: WorldScale) -> f64 {
    // Returns a LEGACY depth: `band_shift` is authored in legacy cells and every
    // band threshold this feeds — CAVERN_DEPTH, DEEP_DEPTH, UNDERWORLD_* — is
    // too, so the world depth crosses inward before the shift is added.
    scale.depth(f64::from(depth)) + cc.band_shift
}

/// One worker's worldgen scratch: the noise, the heightmap memo, the cave
/// lattice and the per-column tables, all owned.
///
/// Build one per thread and keep it — [`Noise::new`] fills a permutation table
/// and [`Heightmap`] is ~48 KiB of memo, neither of which a chunk generation
/// should pay for. Nothing inside is observable from outside: two `ChunkGen`s on
/// the same seed produce identical chunks, and a single one produces the same
/// chunk whatever it generated before. That is checks 1 and 8 of the purity
/// suite, and it is the whole reason this is a struct instead of a module.
pub struct ChunkGen {
    seed: u32,
    noise: Noise,
    heightmap: Heightmap,
    lattice: CaveLattice,
    // The former module-level `COL_*` scratch. Sized once at construction and
    // reused by every `generate` call; the only allocation a chunk generation
    // performs is its own output array.
    col_profile: Vec<ColumnProfile>,
    col_cave: Vec<CaveColumn>,
    col_surf: Vec<i32>,
    col_shore: Vec<f64>,
    scale: WorldScale,
    decorated: bool,
}

impl ChunkGen {
    /// A generator for one world seed, at the scale the game runs at.
    pub fn new(seed: u32) -> ChunkGen {
        ChunkGen::with_scale(seed, WorldScale::LIVE)
    }

    /// A generator pinned to a given world scale.
    ///
    /// The only caller that passes anything but [`WorldScale::LIVE`] is
    /// `tests/player_golden.rs`, which must keep regenerating the world its
    /// un-regenerable 4 048-step replay was recorded against. See [`WorldScale`].
    pub fn with_scale(seed: u32, scale: WorldScale) -> ChunkGen {
        let noise = Noise::new(seed);
        // `ColumnProfile` has no meaningful zero — it is a blend of biomes — so
        // the tables are seeded with a real profile and overwritten in pass 1
        // before anything reads them.
        let seed_profile = column_profile_at(&noise, 0, scale);
        // One entry per COLUMN of a chunk, not per cell.
        let n = CHUNK_CELLS as usize;
        ChunkGen {
            seed,
            noise,
            heightmap: Heightmap::new(),
            lattice: CaveLattice::new(),
            col_profile: vec![seed_profile; n],
            col_cave: vec![CaveColumn::default(); n],
            col_surf: vec![0; n],
            col_shore: vec![0.0; n],
            scale,
            decorated: true,
        }
    }

    /// The world scale this generator is pinned to.
    #[inline]
    pub fn scale(&self) -> WorldScale {
        self.scale
    }

    /// Stop running the decorator passes: terrain, and nothing that grows on it.
    ///
    /// For `tests/player_golden.rs` and nothing else. That fixture replays 4 048
    /// fixed steps against an arena stamped into generated terrain and pins the
    /// window by material hash, and decorators put it at the mercy of work it has
    /// no stake in: a tree is not part of `Player::step`'s ordering, but redrawing
    /// one moves the hash and breaks the replay. Trees, ores and structures are
    /// exactly the parts of worldgen that get retuned most often.
    ///
    /// The arena the replay actually walks on is stamped ON TOP of this and is
    /// unaffected; what goes away is the flora on the terrain around it.
    pub fn without_decor(mut self) -> ChunkGen {
        self.decorated = false;
        self
    }

    /// The world seed this generator is pinned to.
    #[inline]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// The noise this generator resolves against.
    #[inline]
    pub fn noise(&self) -> &Noise {
        &self.noise
    }

    /// The memoising heightmap this generator owns — the surface row for any
    /// absolute column, warm for the chunk just generated.
    #[inline]
    pub fn heightmap(&mut self) -> &mut Heightmap {
        &mut self.heightmap
    }

    /// Generate one chunk's material as a fresh `CHUNK_CELLS²` array from
    /// absolute coordinates. This is the unit a streaming chunk store requests
    /// on demand.
    ///
    /// Signature is fixed: (chunk_x, chunk_y) in, materials out, no neighbour
    /// reads, no hidden state.
    ///
    /// This is the function `worldgen_golden` hashes 357 chunks of, so it is
    /// deliberately a thin wrapper: everything happens in [`ChunkGen::generate_into`]
    /// and this passes `None`, which is the code path that existed before the back
    /// plane and emits exactly the same arithmetic.
    pub fn generate(&mut self, chunk_x: i32, chunk_y: i32) -> Vec<CellId> {
        self.generate_into(chunk_x, chunk_y, None)
    }

    /// One chunk's front plane and its background walls, together.
    ///
    /// The two are generated in ONE pass because the wall is a function of the
    /// same column profile, ground line and cave parameters the front plane
    /// already computed — running a second pass would double the cost of the most
    /// expensive thing the engine does and would risk the two disagreeing.
    pub fn generate_with_back(&mut self, chunk_x: i32, chunk_y: i32) -> (Vec<CellId>, Vec<CellId>) {
        let mut back = vec![AIR; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let out = self.generate_into(chunk_x, chunk_y, Some(&mut back));
        (out, back)
    }

    /// The generator proper. `back`, when present, is filled with the background
    /// wall plane — see [`ChunkGen::generate_with_back`].
    ///
    /// # Why the back plane is an `Option` and not a second return value
    ///
    /// So that `None` is provably the old code path. Every write to the wall
    /// plane below sits inside an `if let Some(back)`, so with `None` the emitted
    /// arithmetic for the front plane is exactly what it was before this
    /// parameter existed — which is the only way to be sure a 357-chunk frozen
    /// hash is untouched by construction rather than by inspection.
    ///
    /// The overwhelming majority of calls are `None`: `generate` is what the
    /// worldgen purity suite, the parity suite, the benches and the dump binary
    /// all use, and none of them has anything to do with walls.
    fn generate_into(
        &mut self,
        chunk_x: i32,
        chunk_y: i32,
        back: Option<&mut [CellId]>,
    ) -> Vec<CellId> {
        let base_x = chunk_x * CHUNK_CELLS;
        let base_y = chunk_y * CHUNK_CELLS;
        let mut out = vec![AIR; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let chunk_bottom = base_y + CHUNK_CELLS;
        // Taken by reference once rather than re-matched per cell, so the hot
        // loops below branch on an `Option` that is constant for the whole chunk.
        let mut back = back;

        // --- Pass 1: per-column profile, ground line, cave parameters --------
        // Climate, biome blend, underground layer blend, surface height and the
        // cave column parameters all depend on the column only. `lattice_from`
        // tracks the highest row any column could need a cave field at — the cap
        // bottom normally, or the ground line itself where a surface chasm can
        // breach the topsoil.
        let mut lattice_from = f64::INFINITY;
        for lx in 0..CHUNK_CELLS {
            let wcx = base_x + lx;
            let col = column_profile_at(&self.noise, wcx, self.scale);
            let detail = self
                .heightmap
                .surface_detail_at(&self.noise, wcx, &col, self.scale);
            let cc = cave_column_at(&self.noise, wcx, detail.surf, &col, self.scale);
            // `cap_thickness` is a legacy thickness and this is a world row, so
            // the cap crosses outward before it is added to the ground line.
            let from = f64::from(detail.surf)
                + if cc.breaches {
                    0.0
                } else {
                    self.scale.len(col.cap_thickness)
                };
            if from < lattice_from {
                lattice_from = from;
            }
            let i = lx as usize;
            self.col_profile[i] = col;
            self.col_surf[i] = detail.surf;
            self.col_shore[i] = detail.shore;
            self.col_cave[i] = cc;
        }

        // --- Pass 2: the cave lattice, only if this chunk has anything to carve
        // Sky chunks — the majority of what a streaming window touches while the
        // player walks — skip 7 fields × 81 samples entirely.
        let carving = f64::from(chunk_bottom) > lattice_from;
        if carving {
            self.lattice.fill(&self.noise, base_x, base_y, self.scale);
        }

        // --- Pass 3: the vertical scan ---------------------------------------
        for lx in 0..CHUNK_CELLS {
            let i0 = lx as usize;
            let wcx = base_x + lx;
            let col = self.col_profile[i0];
            let cc = self.col_cave[i0];
            let surf = self.col_surf[i0];
            let shore = self.col_shore[i0];

            // Sky and sea. Everything above the ground line is air, except where
            // the ground line fell below sea level — then the gap is water, which
            // is the entire lake/ocean mechanism. `out` arrives zeroed and AIR is
            // 0, so the air run costs nothing at all: skip straight to the first
            // water row.
            let ground_ly = surf - base_y;
            let sky_end = ground_ly.clamp(0, CHUNK_CELLS);
            let sea_ly = self.scale.row(SEA_LEVEL_Y) - base_y;
            let mut ly = sea_ly.clamp(0, sky_end);
            while ly < sky_end {
                out[(ly * CHUNK_CELLS + lx) as usize] = WATER;
                ly += 1;
            }

            // Topsoil cap. `depth < capThickness` with a fractional thickness, so
            // the loop bound is the ceiling of the fractional bottom row.
            let cap_bottom =
                (f64::from(surf) + self.scale.len(col.cap_thickness) - f64::from(base_y)).ceil();
            let cap_end = if cap_bottom > f64::from(CHUNK_CELLS) {
                CHUNK_CELLS
            } else {
                cap_bottom as i32
            };
            while ly < cap_end {
                let wcy = base_y + ly;
                let depth = wcy - surf;
                // Only columns that can host a surface chasm pay for a carve test
                // up here.
                let i = (ly * CHUNK_CELLS + lx) as usize;
                if carving
                    && cc.breaches
                    && self
                        .lattice
                        .carve(&self.noise, wcx, wcy, depth, lx, ly, &cc, self.scale)
                        != Carve::Solid
                {
                    // Carved out of the cap by a surface chasm. The front is air;
                    // the WALL is the topsoil the chasm removed, which is what
                    // stops a chasm reading as a hole punched through to nothing.
                    if let Some(back) = back.as_deref_mut() {
                        back[i] = cap_at(&self.noise, wcx, wcy, depth, &col, shore, self.scale);
                    }
                    ly += 1;
                    continue; // already AIR
                }
                let cap = cap_at(&self.noise, wcx, wcy, depth, &col, shore, self.scale);
                out[i] = cap;
                // Solid front, so the wall behind it is the same material and the
                // word is copied rather than recomputed.
                if let Some(back) = back.as_deref_mut() {
                    back[i] = cap;
                }
                ly += 1;
            }

            // Underground.
            while ly < CHUNK_CELLS {
                let wcy = base_y + ly;
                let depth = wcy - surf;
                let c = self
                    .lattice
                    .carve(&self.noise, wcx, wcy, depth, lx, ly, &cc, self.scale);
                let i = (ly * CHUNK_CELLS + lx) as usize;

                // The wall is the rock that WOULD be here if nothing had carved,
                // so it is the same `solid_at` the front plane takes on its solid
                // branch — evaluated whatever the carve said. That is the whole
                // rule, and it is why a fresh shaft mined into virgin rock has a
                // wall behind it rather than a black void.
                let u = ug_fade_at(depth, &col, self.scale);
                let bd = band_depth(depth, &cc, self.scale);
                let strata = if bd >= f64::from(DEEP_DEPTH) {
                    self.lattice.strata_at(lx, ly)
                } else {
                    0.0
                };
                //
                // Wherever the front is already solid this is the SAME word the
                // front takes, so the common case is one evaluation shared by both
                // planes rather than two. Only carved cells — air and liquid, the
                // ones the front pass used to `continue` straight past — pay for an
                // extra `solid_at`, which is why asking for walls costs about 1% on
                // a recenter and 3% on a whole shift tick rather than doubling it.
                // (Measured: `recenter only` 328.9 -> 332.3 us, `shift tick`
                // 475.0 -> 489.2 us, same criterion session.)
                let rock = if back.is_some() || c == Carve::Solid {
                    solid_at(&self.noise, wcx, wcy, bd, &col, u, strata, self.scale)
                } else {
                    // Never read on this branch, and never evaluated either: a
                    // carved cell with no back plane asked for must not pay for a
                    // wall nobody wants. This is the arithmetic `generate` skips.
                    AIR
                };
                if let Some(back) = back.as_deref_mut() {
                    back[i] = rock;
                }

                if c == Carve::Air {
                    ly += 1;
                    continue; // already AIR
                }
                if c != Carve::Solid {
                    out[i] = liquid_at(&self.noise, wcx, wcy, bd, &col, u, self.scale);
                    ly += 1;
                    continue;
                }
                out[i] = rock;
                ly += 1;
            }
        }

        if self.decorated {
            self.decorate(base_x, base_y, &mut out, back);
        }
        out
    }

    /// Run every registered decoration pass over one chunk's material array.
    ///
    /// `back` is handed on so a pass may draw BEHIND the play plane — the tree
    /// decorator puts a share of its trees there. It stays an `Option` for the
    /// same reason the wall plane itself does: on the `generate` path there is no
    /// back plane, a behind-drawn decoration writes nothing at all rather than
    /// falling through to the front, and the front plane the goldens hash is
    /// identical on both paths.
    fn decorate(
        &mut self,
        base_x: i32,
        base_y: i32,
        out: &mut [CellId],
        back: Option<&mut [CellId]>,
    ) {
        let mut ctx = DecorContext::new(
            &self.noise,
            self.seed,
            base_x,
            base_y,
            out,
            &mut self.heightmap,
            self.scale,
        );
        if let Some(back) = back {
            ctx = ctx.with_back(back);
        }
        for d in DECORATORS {
            d.decorate(&mut ctx);
        }
    }
}

/// Generate one chunk from absolute chunk coordinates and a seed.
///
/// Convenience over [`ChunkGen`] for a caller that generates one chunk and
/// throws the machinery away. It builds a whole `Noise` and a whole heightmap
/// memo per call, so anything generating chunks in bulk — the streaming path,
/// the benches, the dump tool — should hold a [`ChunkGen`] instead.
pub fn generate_chunk(chunk_x: i32, chunk_y: i32, seed: u32) -> Vec<CellId> {
    ChunkGen::new(seed).generate(chunk_x, chunk_y)
}

/// [`generate_chunk`] at an explicit world scale.
///
/// This exists for exactly one caller. `tests/player_golden.rs` replays 4 048
/// fixed steps against an arena stamped into a world it pins by material hash
/// and cannot regenerate — its provenance is a TypeScript tool in a repository
/// that no longer exists, and it has no bless path. It calls this with
/// [`WorldScale::LEGACY`] so the generator can move underneath it without the
/// replay losing the ground it was recorded on.
///
/// Anything else wants [`generate_chunk`].
/// [`generate_chunk_scaled`] with the decorator passes switched off — terrain
/// only. See [`ChunkGen::without_decor`] for the single caller and why.
pub fn generate_chunk_terrain(
    chunk_x: i32,
    chunk_y: i32,
    seed: u32,
    scale: WorldScale,
) -> Vec<CellId> {
    ChunkGen::with_scale(seed, scale)
        .without_decor()
        .generate(chunk_x, chunk_y)
}

pub fn generate_chunk_scaled(
    chunk_x: i32,
    chunk_y: i32,
    seed: u32,
    scale: WorldScale,
) -> Vec<CellId> {
    ChunkGen::with_scale(seed, scale).generate(chunk_x, chunk_y)
}

// --- Spawn -------------------------------------------------------------------

/// How far either side of the requested column [`spawn_point`] will look.
const SPAWN_SEARCH: i32 = 512;
/// How far above sea level the ground has to be to count as dry land.
const SPAWN_CLEARANCE: i32 = 4;
/// The column the TypeScript defaulted to.
pub const SPAWN_COL: i32 = 8;

/// A world-pixel position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnPoint {
    pub x: f32,
    pub y: f32,
}

/// Spawn point (world px) for a fresh world.
///
/// Now that oceans exist, a fixed column is a coin flip between a beach and the
/// sea floor. The search walks outward from `spawn_col` for the nearest column
/// whose ground line clears sea level by a few cells — still a pure function of
/// (seed, spawn_col), still no neighbour state, just a deterministic scan of a
/// deterministic field.
///
/// (`spawn_col` was a defaulted parameter in the TypeScript; pass [`SPAWN_COL`]
/// for the same behaviour.)
pub fn spawn_point(seed: u32, spawn_col: i32, scale: WorldScale) -> SpawnPoint {
    let noise = world_noise(seed);
    let mut hm = Heightmap::new();
    // Every one of these is authored in legacy cells: the search radius and the
    // clearance are distances, the waterline is a row, and the 6-cell lift below
    // is how far above the ground the body starts.
    let search = scale.row(SPAWN_SEARCH);
    let dry = scale.row(SEA_LEVEL_Y - SPAWN_CLEARANCE);
    // The lift is how far above the ground the BODY starts, so it scales with the
    // body and not with the world. Scaling it with the world put the player 24
    // cells up — four body heights of empty air — and `walkable_spawn` then
    // measured walkable ground at a row nowhere near the ground, reporting zero
    // clear cells in either direction at every seed.
    let lift = 6 * BODY_SCALE;
    let mut col = spawn_col;
    let mut surf = hm.surface_row_at(&noise, col, None, scale);
    let mut r = 1;
    while r <= search && surf > dry {
        let right = hm.surface_row_at(&noise, spawn_col + r, None, scale);
        if right <= dry {
            col = spawn_col + r;
            surf = right;
            break;
        }
        let left = hm.surface_row_at(&noise, spawn_col - r, None, scale);
        if left <= dry {
            col = spawn_col - r;
            surf = left;
            break;
        }
        r += 1;
    }
    SpawnPoint {
        x: (col * CELL_SIZE) as f32,
        y: ((surf - lift) * CELL_SIZE) as f32,
    }
}

/// Cells of walkable ground a spawn wants either side of the body.
///
/// Twelve — six body-widths, 60 world px.
///
/// Chosen from the terrain rather than from taste. Sampling 24 seeds, the
/// walkable run beside a spawn is typically 12 to 20 cells and only occasionally
/// more (one seed in 24 offered 54). The surface is broken up by slopes steeper
/// than the body's one-cell step, so asking for much more than this does not
/// find a better spot, it drags the spawn hundreds of columns from the one the
/// caller asked for to find a rare flat.
///
/// **What this does NOT promise is an open world to walk in.** Twelve cells is
/// room to move, see where you are and pick a direction. Getting further means
/// jumping or digging, which is the game. What it rules out is starting the run
/// entombed, which was happening to a quarter of seeds.
const SPAWN_WALK_CELLS: i32 = 12;

/// How far the ground may RISE between two adjacent columns and still be walked.
///
/// One cell. That is the step-up the body actually has — `player_golden`'s arena
/// includes a one-cell ledge for exactly this — so a check that allowed more
/// would call a wall a path. Downward steps are unbounded: falling off a ledge
/// is still leaving, and a spawn beside a drop is a fine place to start.
const SPAWN_STEP_UP: i32 = 1;

/// How far below the spawn row to look for the ground.
///
/// [`spawn_point`] returns a point six cells ABOVE the surface — the body is
/// dropped in and falls — so a walkability check run at the spawn's own row is
/// measuring thin air. This is the distance it scans down to find what the body
/// will actually be standing on. Generous, because the columns either side may
/// be lower than the one the body starts over.
const SPAWN_GROUND_SCAN: i32 = 24;

/// A spawn you can actually walk away from.
///
/// # Why this is not just [`spawn_point`]
///
/// [`spawn_point`] answers "which column is dry land", and it answers it
/// correctly — it is a pure function of the heightmap, which is why it can be
/// baselined and why it costs nothing. But the heightmap is not the world.
/// Decorators run afterwards and put TREES on that dry land, and every leaf
/// material in `content/blocks/flora.toml` is authored `collides = true`, so a
/// canopy is a wall.
///
/// Measured over 24 seeds at [`SPAWN_COL`], counting cells of walkable ground
/// either side: **23 of them gave the body under 12 cells on one side or both,
/// and six gave it ZERO on one side — seed 17 had zero on both.** The body could
/// not move at all. That is not bad luck at one seed.
///
/// After: every one of the 24 has at least 12 either way, mean worst side 14.
///
/// # Why a second function rather than a fix to the first
///
/// Two different questions. "Where is the land" is a property of the heightmap
/// and is what `worldgen_golden` pins; "where can a body stand" is a property of
/// the generated world, needs chunks, and costs a few milliseconds. Folding the
/// second into the first would make a cheap pure function expensive, and would
/// move a baselined value for a reason that has nothing to do with terrain.
///
/// So this starts from [`spawn_point`] and walks outward for the nearest column
/// the body fits in with somewhere to go. It is still a pure function of
/// `(seed, spawn_col)`.
///
/// # It follows the GROUND, not a row
///
/// The first version of this checked a fixed row for clear cells, and it was
/// wrong in a way that looked right: `spawn_point`'s row is six cells above the
/// surface, so on open ground every column passed trivially and the body still
/// stopped dead after exactly the number of cells the check had verified.
/// Terrain is not flat. What matters is whether there is a walkable SURFACE
/// leading away — ground the body can stand on, rising no faster than it can
/// step — so each column's ground row is found by scanning down, and the
/// comparison is against its neighbour rather than against a constant.
///
/// # It never fails
///
/// If nothing within [`SPAWN_SEARCH`] qualifies, the plain [`spawn_point`] is
/// returned. A world with nowhere good to stand should start somewhere bad, not
/// refuse to start — and the caller has no better answer than this one does.
pub fn walkable_spawn(seed: u32, spawn_col: i32, scale: WorldScale) -> SpawnPoint {
    let base = spawn_point(seed, spawn_col, scale);
    let col = (base.x / CELL_SIZE as f32).floor() as i32;
    let row = (base.y / CELL_SIZE as f32).floor() as i32;

    let mut probe = SpawnProbe::new(seed, scale);
    // Both sides first, then either side. A spawn you can only leave in one
    // direction is playable but poor: half of what the player tries at the very
    // start of a run walks straight into a wall, and it is the half a scenario
    // file is as likely to pick as the other. Trying for both and settling for
    // one is what the fallback is; it costs a second scan of columns already in
    // the chunk cache.
    for want_both in [true, false] {
        for r in 0..=SPAWN_SEARCH {
            // Outward alternately, so a tie goes to the column nearest the one
            // the caller asked for rather than always to the right.
            let candidates: &[i32] = if r == 0 { &[col] } else { &[col + r, col - r] };
            for &at in candidates {
                if probe.stands_and_walks(at, row, want_both) {
                    return SpawnPoint {
                        x: (at * CELL_SIZE) as f32,
                        y: base.y,
                    };
                }
            }
        }
    }
    base
}

/// Walkable ground either side of a spawn, capped at `far` cells.
///
/// Exported for one reason: it is how a caller checks whether this module and
/// the RUNNING GAME agree about what "walkable" means. They did not, twice, and
/// both times the disagreement was invisible from inside — the check passed and
/// the body still stopped dead.
pub fn spawn_ground_runs(seed: u32, at: SpawnPoint, far: i32, scale: WorldScale) -> (i32, i32) {
    let col = (at.x / CELL_SIZE as f32).floor() as i32;
    let row = (at.y / CELL_SIZE as f32).floor() as i32;
    let w = (PLAYER_W / CELL_SIZE as f32).ceil() as i32;
    let mut probe = SpawnProbe::new(seed, scale);
    (
        probe.run_of_ground(col, row, -1, far),
        probe.run_of_ground(col + w - 1, row, 1, far),
    )
}

/// Chunk-generating scratch for [`walkable_spawn`].
///
/// A cache, because the columns it probes are adjacent and a chunk is 32 of
/// them: without one, a search that walked a hundred columns would regenerate
/// the same chunk a hundred times.
struct SpawnProbe {
    chunks_from: ChunkGen,
    chunks: Vec<((i32, i32), Vec<CellId>)>,
}

impl SpawnProbe {
    fn new(seed: u32, scale: WorldScale) -> SpawnProbe {
        SpawnProbe {
            // The probe MUST generate at the same scale the search reasons in, or
            // it measures walkable ground in a world the caller is not standing
            // in and reports a clear ledge on the face of a cliff.
            chunks_from: ChunkGen::with_scale(seed, scale),
            chunks: Vec::new(),
        }
    }

    /// The generated material at an absolute cell.
    fn cell(&mut self, x: i32, y: i32) -> CellId {
        let key = (x.div_euclid(CHUNK_CELLS), y.div_euclid(CHUNK_CELLS));
        if !self.chunks.iter().any(|(k, _)| *k == key) {
            let c = self.chunks_from.generate(key.0, key.1);
            self.chunks.push((key, c));
        }
        let chunk = &self
            .chunks
            .iter()
            .find(|(k, _)| *k == key)
            .expect("just inserted")
            .1;
        let (lx, ly) = (x.rem_euclid(CHUNK_CELLS), y.rem_euclid(CHUNK_CELLS));
        chunk[(ly * CHUNK_CELLS + lx) as usize]
    }

    /// The row of the first solid cell at or below `from`, and whether a body
    /// standing on it has room to stand.
    ///
    /// `None` means either nothing solid within [`SPAWN_GROUND_SCAN`] — a column
    /// over a chasm, which is not somewhere to walk to — or ground with
    /// something on top of it that the body does not fit under.
    fn ground_at(&mut self, x: i32, from: i32) -> Option<i32> {
        let h = (PLAYER_H / CELL_SIZE as f32).ceil() as i32;
        let floor = (from..from + SPAWN_GROUND_SCAN)
            .find(|&y| MAT_COLLIDE[self.cell(x, y) as usize] == 1)?;
        // The body stands ON that cell, so it occupies the `h` rows above it.
        (1..=h)
            .all(|dy| MAT_COLLIDE[self.cell(x, floor - dy) as usize] == 0)
            .then_some(floor)
    }

    /// How many columns of walkable ground run away from `x` in `step`.
    ///
    /// Stops at the first column with no standable ground, or one whose surface
    /// rises more than [`SPAWN_STEP_UP`] above the last. Drops are free: falling
    /// off a ledge is still leaving.
    fn run_of_ground(&mut self, x: i32, from: i32, step: i32, want: i32) -> i32 {
        let Some(mut last) = self.ground_at(x, from) else {
            return 0;
        };
        let mut n = 0;
        while n < want {
            let at = x + step * (n + 1);
            let Some(g) = self.ground_at(at, (last - 4).min(from)) else {
                break;
            };
            if last - g > SPAWN_STEP_UP {
                break;
            }
            last = g;
            n += 1;
        }
        n
    }

    /// Can a body stand at `x`, and walk [`SPAWN_WALK_CELLS`] — both ways if
    /// `both`, otherwise either way?
    fn stands_and_walks(&mut self, x: i32, row: i32, both: bool) -> bool {
        let w = (PLAYER_W / CELL_SIZE as f32).ceil() as i32;
        if (0..w).any(|dx| self.ground_at(x + dx, row).is_none()) {
            return false;
        }
        let right = self.run_of_ground(x + w - 1, row, 1, SPAWN_WALK_CELLS) == SPAWN_WALK_CELLS;
        let left = self.run_of_ground(x, row, -1, SPAWN_WALK_CELLS) == SPAWN_WALK_CELLS;
        if both { left && right } else { left || right }
    }
}

// (`generateWorld` is not ported: it was a phase-1 shim that blitted chunks into
// a finite `CellGrid`, and this crate has no `CellGrid` — the streaming store
// consumes `generate_chunk` directly.)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorldScale;
    use crate::config::worldgen::SEED;

    const N: usize = (CHUNK_CELLS * CHUNK_CELLS) as usize;

    /// Walkable ground either side of a spawn, well past what is required.
    fn room_around(seed: u32, at: SpawnPoint) -> (i32, i32) {
        spawn_ground_runs(seed, at, 200, WorldScale::LIVE)
    }

    /// The defect this was written for, and the proof the fix answers it.
    ///
    /// `spawn_point` asks the heightmap where the land is, and the heightmap
    /// knows nothing about the trees the decorators put on it. Every leaf
    /// material is authored `collides = true`, so a canopy is a wall — and a
    /// third of seeds dropped the body into one.
    #[test]
    fn a_spawn_is_somewhere_the_body_can_walk_away_from() {
        let seeds: Vec<u32> = (0..24).collect();

        // The control, and the reason this test exists rather than being
        // assumed: the plain spawn is genuinely bad on a lot of seeds. If this
        // ever stops being true the fix below has stopped being needed, and
        // that is worth being told about rather than quietly carrying.
        let cramped = seeds
            .iter()
            .filter(|&&s| {
                let (l, r) = room_around(s, spawn_point(s, SPAWN_COL, WorldScale::LIVE));
                l.min(r) < SPAWN_WALK_CELLS
            })
            .count();
        // The positive control: `walkable_spawn` is only worth its cost if the
        // PLAIN spawn is often bad. This has fallen a long way — 23 of 24 before
        // the flora rework, 11 once trees were planted on a scaled stride instead
        // of every three cells, 7 once a third of them moved behind the play
        // plane where nothing collides with them.
        //
        // 7 of 24 is still not nothing and the search is still cheap, but this is
        // now close enough to the floor to say plainly: if it reaches ~3, DELETE
        // `walkable_spawn` and the whole probe with it rather than carrying a
        // search that almost never finds anything to fix.
        assert!(
            cramped >= 5,
            "only {cramped}/24 plain spawns are cramped, and 7 were measured. \
             If the surface stopped being broken up, `walkable_spawn` is now \
             dead weight and should go rather than be quietly carried"
        );

        // And the claim. BOTH sides, not either: a spawn you can only leave in
        // one direction is playable but poor, and half of what a player — or a
        // scenario file — tries first walks straight into a wall. All 24 of
        // these clear it, so the one-sided fallback inside `walkable_spawn` is
        // for worlds stranger than any of them rather than for these.
        let mut both = 0;
        for &seed in &seeds {
            let at = walkable_spawn(seed, SPAWN_COL, WorldScale::LIVE);
            let (left, right) = room_around(seed, at);
            if left.min(right) >= SPAWN_WALK_CELLS {
                both += 1;
            }
            // The guarantee, and it holds for every seed: the body can always
            // walk SPAWN_WALK_CELLS in at least one direction. This is what
            // `walkable_spawn` actually promises, and a failure here means it
            // returned somewhere the run cannot start.
            assert!(
                left.max(right) >= SPAWN_WALK_CELLS,
                "seed {seed}: walkable_spawn put the body at {at:?} with {left} \
                 cells clear to the left and {right} to the right — it could not \
                 walk away in EITHER direction"
            );
        }
        // And the preference. This fell as the world grew — 24 of 24 at the 1x
        // world, 22 at 2x, 15 at 4x — and the flora rework put it back to 24 of
        // 24, which is the strongest evidence available that the forest is
        // walkable again rather than merely different.
        //
        // It also settles the open question the previous version of this comment
        // recorded. The two candidate causes were a bigger body needing more
        // ground, and clutter obstructing the surface. It was neither: trees were
        // being planted every three cells while each one grew four times wider,
        // so a woodland was a fence. (Clutter was never even upscaled — that
        // hypothesis was wrong on its own terms.)
        assert!(
            both == seeds.len(),
            "only {both}/{} spawns clear SPAWN_WALK_CELLS on BOTH sides, and all \
             24 were measured — the forest has stopped being walkable somewhere",
            seeds.len()
        );
    }

    /// Still a pure function of its arguments, like everything else here.
    #[test]
    fn a_walkable_spawn_is_the_same_every_time_it_is_asked() {
        for seed in [0u32, 17, 2334] {
            let a = walkable_spawn(seed, SPAWN_COL, WorldScale::LIVE);
            let b = walkable_spawn(seed, SPAWN_COL, WorldScale::LIVE);
            assert_eq!(a, b, "seed {seed}");
        }
    }

    /// It only moves the body sideways. The row is the heightmap's answer and
    /// this has no business second-guessing how high above the ground to stand.
    #[test]
    fn a_walkable_spawn_keeps_the_height_the_heightmap_chose() {
        for seed in 0..12u32 {
            assert_eq!(
                walkable_spawn(seed, SPAWN_COL, WorldScale::LIVE).y,
                spawn_point(seed, SPAWN_COL, WorldScale::LIVE).y,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn a_chunk_is_a_full_square_of_known_materials() {
        let c = generate_chunk(0, 2, SEED);
        assert_eq!(c.len(), N);
        assert!(
            c.iter()
                .all(|&m| (m as usize) < crate::sim::materials::MAT_R.len())
        );
    }

    #[test]
    fn sky_is_empty_and_the_floor_is_solid() {
        let mut g = ChunkGen::new(SEED);
        // Far above any possible surface: nothing but air.
        let sky = g.generate(0, -8);
        assert!(sky.iter().all(|&m| m == AIR), "the sky has something in it");
        // Below UNDERWORLD_FLOOR (640 cells / 32 = chunk row 20 and beyond,
        // measured from a surface near row 48): bedrock, no holes.
        let floor = g.generate(0, 40);
        assert!(
            floor.iter().all(|&m| m != AIR),
            "the world has a hole in its floor"
        );
    }

    #[test]
    fn one_generator_and_a_fresh_one_agree() {
        // The cheap version of purity check 1, kept here so a broken memo fails
        // the unit suite too and not only the integration one.
        let mut g = ChunkGen::new(SEED);
        for (cx, cy) in [(0, 0), (-2, 5), (7, 12), (0, 0)] {
            assert_eq!(g.generate(cx, cy), generate_chunk(cx, cy, SEED));
        }
    }

    #[test]
    fn spawn_is_on_dry_land_or_the_search_ran_out() {
        // Both the 6-cell lift and the waterline are authored in legacy cells, so
        // the expectations go through the scale rather than being written as the
        // world rows they happen to be at one particular value of it.
        let scale = WorldScale::LIVE;
        let p = spawn_point(SEED, SPAWN_COL, scale);
        let noise = world_noise(SEED);
        let mut hm = Heightmap::new();
        let col = (p.x as i32) / CELL_SIZE;
        let surf = hm.surface_row_at(&noise, col, None, scale);
        assert_eq!(p.y as i32, (surf - 6 * BODY_SCALE) * CELL_SIZE);
        assert!(
            surf <= scale.row(SEA_LEVEL_Y - SPAWN_CLEARANCE),
            "spawned in the sea"
        );
    }
}
