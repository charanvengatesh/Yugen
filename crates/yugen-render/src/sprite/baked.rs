//! Rasterising the authored characters into one strip of pixels.
//!
//! [`super`]'s "The baked sprite" section. This is where an authored grid of
//! palette digits becomes RGBA8: one tile per distinct frame across every sequence and
//! variant, laid out as a single horizontal strip, with the palette resolved and
//! the variant tints already multiplied in.
//!
//! The rasterising itself is pure — a palette, some digits, a `Vec<u8>` — and
//! that is what lets the bake be tested for exact pixels without a device.
//! Bevy appears only at the far end, where [`BakedSprite::image`] and
//! [`BakedSprite::layout`] hand the finished strip over as an `Image` and a
//! `TextureAtlasLayout`. Those two methods are the whole of the dependency, and
//! they are here rather than in [`super`] because the tile geometry they
//! describe is this file's.

use std::collections::HashMap;

use yugen_core::config::CELL_SIZE;

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, TextureAtlasLayout};
use bevy::math::{UVec2, Vec4};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::anim::pick_frame_index;
use super::vocab::{
    Frame, POSE_COUNT, Pose, SeqSpec, SpriteArt, SpriteClock, SpriteError, VARIANT_COUNT,
    VARIANT_TINT,
};

// ---------------------------------------------------------------------------
// The baked sprite
// ---------------------------------------------------------------------------

/// A resolved sequence index, with `fallback` already applied.
///
/// Opaque, and only obtainable from [`BakedSprite::state_id`] on the sprite it
/// indexes — which is what let the TypeScript's `state < 0 || state >= len` guards
/// go. A `StateId` from one sprite used on another is a bug the way any index
/// confusion is; keep the one your facade resolved at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateId(u8);

impl StateId {
    /// The dense sequence index. For diagnostics, and for a facade that wants to
    /// store the whole resolved set as an array.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Tile index per `[variant][state][frame]`. Entries repeat where poses do.
///
/// A named alias because it is a field type, a local and a constructor result,
/// and three spellings of `Vec<Vec<Vec<u32>>>` is three chances to get the
/// nesting order wrong.
pub(super) type TileGrid = Vec<Vec<Vec<u32>>>;

/// One rasterised sprite: the atlas pixels, the tile index per pose and frame,
/// and the cadence metadata that picks between them.
///
/// Constructed once. Nothing here rasterises, allocates or hashes after
/// [`BakedSprite::new`] returns — see the module header.
#[derive(Clone, Debug)]
pub struct BakedSprite {
    /// Art grid width in cells. `grain` texels per cell; see the module header.
    pub cells_w: u32,
    /// Art grid height in cells.
    pub cells_h: u32,
    /// Art pixels per cell, per axis. 1 for everything except a finer-grain
    /// experiment; the TEXEL dimensions of every tile are `cells * grain`.
    pub grain: u32,
    /// Art rect width in world px. For a mob this is also exactly the collision
    /// box.
    pub w_px: f32,
    /// Art rect height in world px.
    pub h_px: f32,
    /// How many tinted copies actually exist.
    pub variants: usize,
    /// Distinct tiles actually rasterised, and the atlas's tile count.
    ///
    /// Diagnostic as well as structural — it exists so the pose-sharing claim on
    /// [`BakedSprite::new`] is a measurable fact rather than an assertion, and so
    /// a regression that starts double-baking every sprite is visible instead of
    /// merely slow.
    pub bake_count: usize,

    /// RGBA8, `bake_count * cells_w` by `cells_h`. See [`BakedSprite::image`].
    pixels: Vec<u8>,
    tiles: TileGrid,
    /// Parallel to `tiles[v]` — cadence metadata, one copy for all variants.
    specs: Vec<SeqSpec>,
    /// Dense, in authoring order. Diagnostic; the lookup is `ids`.
    poses: Vec<Pose>,
    /// [`Pose`] -> dense state index, with `fallback` already applied.
    ids: [Option<StateId>; POSE_COUNT],
}

impl BakedSprite {
    /// Rasterise every pose of every variant, once.
    ///
    /// # The bake cache
    ///
    /// A pose repeated across sequences bakes ONCE, keyed on the frame's own text.
    /// The TypeScript's hand-written `PlayerSprite` got this by hand — a human
    /// noticed that run's passing pose is the idle exhale and assigned the same
    /// canvas to both. Content cannot do that: an author writes the same six
    /// characters in two bodies and has no way to say "this is the same bitmap".
    /// Keying on the text recovers the property automatically, for every sprite,
    /// without content knowing the cache exists. Construction-time only — nothing
    /// consults it at draw.
    ///
    /// The key includes the variant, because two tints of the same pose are two
    /// different bitmaps.
    ///
    /// # Validation
    ///
    /// Shape and every character are checked and anything illegal is an error.
    /// See [`SpriteError`] for why that is not negotiable.
    pub fn new(art: &SpriteArt, who: &str) -> Result<BakedSprite, SpriteError> {
        if art.cells_w == 0 || art.cells_h == 0 {
            return Err(SpriteError::Grid {
                who: who.to_string(),
                cells_w: art.cells_w as i32,
                cells_h: art.cells_h as i32,
            });
        }
        if art.variants < 1 || art.variants > VARIANT_COUNT {
            return Err(SpriteError::VariantCount {
                who: who.to_string(),
                asked: art.variants,
            });
        }
        debug_assert_eq!(
            art.poses.len(),
            art.seqs.len(),
            "a SpriteArt carries parallel pose and sequence lists"
        );

        let ids = build_ids(art, who)?;

        // Index 0 is the transparent slot and is never painted, so it is never
        // parsed either — content is free to write "." or "transparent" there, as
        // every existing art table does.
        let base: Vec<[u8; 3]> = art
            .pal
            .iter()
            .enumerate()
            .map(|(i, hex)| {
                if i == 0 {
                    Ok([0, 0, 0])
                } else {
                    parse_hex(hex, who)
                }
            })
            .collect::<Result<_, _>>()?;

        // `(variant, frame text) -> tile index`, and the tile pixels in tile
        // order. A `Frame` is a list of `&'static str`, so the key clones
        // pointers and not characters.
        let mut cache: HashMap<(usize, Frame), u32> = HashMap::new();
        let mut tile_pixels: Vec<Vec<u8>> = Vec::new();
        let mut tiles: TileGrid = Vec::with_capacity(art.variants);

        for (v, tint) in VARIANT_TINT.iter().enumerate().take(art.variants) {
            let pal: Vec<[u8; 3]> = base.iter().map(|c| tinted(*c, *tint)).collect();
            let mut per_state = Vec::with_capacity(art.seqs.len());
            for (s, seq) in art.seqs.iter().enumerate() {
                let mut out = Vec::with_capacity(seq.frames.len());
                for (f, frame) in seq.frames.iter().enumerate() {
                    let key = (v, frame.clone());
                    let tile = match cache.get(&key) {
                        Some(t) => *t,
                        None => {
                            let name = format!("{who}.{:?}[{f}]", art.poses[s]);
                            let px = bake_frame(
                                frame,
                                &pal,
                                art.cells_w * art.grain,
                                art.cells_h * art.grain,
                                &name,
                            )?;
                            let t = tile_pixels.len() as u32;
                            tile_pixels.push(px);
                            cache.insert(key, t);
                            t
                        }
                    };
                    out.push(tile);
                }
                per_state.push(out);
            }
            tiles.push(per_state);
        }

        Ok(BakedSprite {
            cells_w: art.cells_w,
            cells_h: art.cells_h,
            grain: art.grain,
            // The DRAWN rect is cells, not texels: a finer grain packs more
            // texels into the same world rectangle, which is the entire point.
            w_px: (art.cells_w * CELL_SIZE as u32) as f32,
            h_px: (art.cells_h * CELL_SIZE as u32) as f32,
            variants: art.variants,
            bake_count: tile_pixels.len(),
            pixels: assemble_strip(
                &tile_pixels,
                art.cells_w * art.grain,
                art.cells_h * art.grain,
            ),
            tiles,
            specs: art.seqs.clone(),
            poses: art.poses.clone(),
            ids,
        })
    }

    /// The dense sequence index for a pose, following `fallback`. `None` = absent.
    ///
    /// CALL THIS ONCE, AT LOAD, and pass the [`StateId`] to [`BakedSprite::tile`]
    /// forever after. The fallback map was applied at construction, so
    /// `state_id(Pose::Air)` on a sprite with no air sequence already returns
    /// `Move`'s index — there is no per-frame `art.air ?? art.move` left anywhere
    /// in the system.
    pub fn state_id(&self, pose: Pose) -> Option<StateId> {
        self.ids[pose.index()]
    }

    /// Which frame of a sequence a clock is showing. Pure; allocates nothing.
    pub fn frame_index(&self, state: StateId, clock: &SpriteClock) -> usize {
        pick_frame_index(&self.specs[state.index()], clock)
    }

    /// The atlas tile a clock is showing — what the TypeScript's `draw` resolved
    /// before its single `drawImage`.
    ///
    /// Feed it to [`SpriteAtlas::sprite_at`], or straight to a
    /// `TextureAtlas::index`. Facing is the caller's `Sprite::flip_x`: mirroring
    /// about the rect's own vertical axis is what the TypeScript's
    /// `translate(x + w, y); scale(-1, 1)` did by hand, and it is what makes the
    /// figure turn while staying exactly where it was — flipping about the origin
    /// would fling it across the screen.
    pub fn tile(&self, state: StateId, variant: usize, clock: &SpriteClock) -> usize {
        let f = self.frame_index(state, clock);
        self.tiles[self.wrap_variant(variant)][state.index()][f] as usize
    }

    /// Frame 0 of a state, no clock — item icons in a hotbar, bestiary portraits.
    ///
    /// A still caller has no clock to own and no business inventing one, and
    /// frame 0 is the authored neutral pose of every sequence in the game by
    /// convention. The TypeScript's `drawStill` was also always unflipped, because
    /// UI has no facing; here that is the caller leaving `flip_x` false.
    pub fn still_tile(&self, state: StateId, variant: usize) -> usize {
        self.tiles[self.wrap_variant(variant)][state.index()][0] as usize
    }

    /// A tile's sub-rectangle in the strip, as `(u0, v0, du, dv)`.
    ///
    /// The same tile [`BakedSprite::tile`] returns, addressed the other way. A
    /// [`Sprite`] takes an atlas INDEX and the engine does this arithmetic; a
    /// mesh takes uv, and the additive glow pass is a mesh because a sprite has
    /// no per-sprite blend state to make additive. Both draws must land on the
    /// same texels or the glow shows as a fringe, so the strip's layout is
    /// published here rather than recomputed by the one caller that needs it.
    ///
    /// `flip_x` mirrors by walking the tile backwards — a negative `du` from the
    /// far edge — which is the same mirror `Sprite::flip_x` performs and, like
    /// it, is about the tile's own axis rather than the texture's.
    pub fn tile_uv(&self, tile: usize, flip_x: bool) -> Vec4 {
        // A strip is one row, so a tile's width is its share of the whole and
        // `v` always spans it. `bake_count` cannot be zero — a sprite with no
        // sequences is a construction error — but a division is not worth the
        // asymmetry with the rest of this type's total functions.
        let du = 1.0 / self.bake_count.max(1) as f32;
        let u0 = tile as f32 * du;
        if flip_x {
            Vec4::new(u0 + du, 0.0, -du, 1.0)
        } else {
            Vec4::new(u0, 0.0, du, 1.0)
        }
    }

    /// The first authored sequence. What an icon with exactly one pose wants, and
    /// what `drawStill`'s `state = 0` default meant.
    pub fn first_state(&self) -> StateId {
        StateId(0)
    }

    /// The poses this sprite actually authored, dense and in authoring order.
    /// `poses()[id.index()]` names the sequence `id` resolves to.
    pub fn poses(&self) -> &[Pose] {
        &self.poses
    }

    /// The cadence metadata for a sequence.
    pub fn spec(&self, state: StateId) -> &SeqSpec {
        &self.specs[state.index()]
    }

    /// The atlas pixels: RGBA8, `bake_count * cells_w` by `cells_h`.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The atlas texture size in texels.
    pub fn atlas_size(&self) -> UVec2 {
        UVec2::new(
            self.cells_w * self.grain * self.bake_count as u32,
            self.cells_h * self.grain,
        )
    }

    /// Variants are assigned from entity ids and hashes, so wrap rather than
    /// trust.
    fn wrap_variant(&self, variant: usize) -> usize {
        variant % self.variants
    }

    // -- The Bevy half ------------------------------------------------------

    /// The atlas as an image asset.
    ///
    /// # Why one strip and not one texture per pose
    ///
    /// One texture per pose is one bind per pose. A strip is one bind for the
    /// whole creature and turns "which frame" into an integer the engine already
    /// knows how to index. It stays a single ROW rather than a square grid because
    /// that makes a tile's x offset `tile * cells_w` and nothing else — the widest
    /// sprite in the game is four cells and the busiest has under forty tiles, so
    /// the strip is a couple of hundred texels against an 8192-texel limit on
    /// every target this runs on.
    ///
    /// `Rgba8UnormSrgb`, because the palette is authored as sRGB hex and a
    /// `Sprite`'s colour tint multiplies in linear — the same format
    /// [`crate::effects`] bakes its wash ramp in, for the same reason.
    ///
    /// `RENDER_WORLD` only: the bake happens once and nothing reads the pixels
    /// back, so the CPU copy is dropped after upload. That is the whole
    /// allocation story, and the contrast is `crate::cellmap`'s id texture, which
    /// is `MAIN_WORLD | RENDER_WORLD` precisely because it is rewritten every
    /// frame.
    ///
    /// Sampling is nearest, from `ImagePlugin::default_nearest()` in the binary.
    /// It has to be: adjacent tiles share an edge in the strip, and any filtering
    /// wider than a texel would bleed one pose into the next. It also has to be
    /// for the reason every sampler in this game is nearest — the art is pixel
    /// art, upscaled with hard edges.
    pub fn image(&self) -> Image {
        let size = self.atlas_size();
        Image::new(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            self.pixels.clone(),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    /// The atlas layout: `bake_count` tiles across, one down.
    pub fn layout(&self) -> TextureAtlasLayout {
        TextureAtlasLayout::from_grid(
            UVec2::new(self.cells_w * self.grain, self.cells_h * self.grain),
            self.bake_count as u32,
            1,
            None,
            None,
        )
    }
}

/// Rasterise one frame into a fresh RGBA8 tile. The dimensions arrive in
/// TEXELS — `cells * grain` — and a frame's text must match them exactly.
pub(super) fn bake_frame(
    frame: &Frame,
    pal: &[[u8; 3]],
    cells_w: u32,
    cells_h: u32,
    who: &str,
) -> Result<Vec<u8>, SpriteError> {
    if frame.len() != cells_h as usize {
        return Err(SpriteError::FrameRows {
            who: who.to_string(),
            frame: 0,
            rows: frame.len(),
            expected: cells_h,
        });
    }
    let mut out = vec![0u8; (cells_w * cells_h * 4) as usize];

    for (row, line) in frame.iter().enumerate() {
        // Counted rather than measured with `len()`, which is bytes: content is
        // ASCII so the two agree today, and this keeps agreeing if it ever is not.
        let chars = line.chars().count();
        if chars != cells_w as usize {
            return Err(SpriteError::RowWidth {
                who: who.to_string(),
                row,
                chars,
                expected: cells_w,
            });
        }
        for (col, ch) in line.chars().enumerate() {
            if ch == '.' || ch == '0' {
                continue;
            }
            // '0'..'9' -> 0..9. Any other character lands outside the range and is
            // rejected rather than painted with whatever colour was last set,
            // which is exactly the bug the original's header describes.
            let idx = (ch as u32).wrapping_sub('0' as u32) as usize;
            if idx < 1 || idx >= pal.len() {
                return Err(SpriteError::PaletteIndex {
                    who: who.to_string(),
                    row,
                    col,
                    ch,
                });
            }
            let c = pal[idx];
            let px = ((row * cells_w as usize) + col) * 4;
            out[px] = c[0];
            out[px + 1] = c[1];
            out[px + 2] = c[2];
            out[px + 3] = 255;
        }
    }
    Ok(out)
}

/// Lay the baked tiles out left to right into one RGBA8 strip.
pub(super) fn assemble_strip(tiles: &[Vec<u8>], cells_w: u32, cells_h: u32) -> Vec<u8> {
    let strip_w = cells_w as usize * tiles.len();
    let mut out = vec![0u8; strip_w * cells_h as usize * 4];
    for (t, tile) in tiles.iter().enumerate() {
        for row in 0..cells_h as usize {
            let src = row * cells_w as usize * 4;
            let dst = (row * strip_w + t * cells_w as usize) * 4;
            let n = cells_w as usize * 4;
            out[dst..dst + n].copy_from_slice(&tile[src..src + n]);
        }
    }
    out
}

/// `#rrggbb` and nothing else.
pub(super) fn parse_hex(hex: &str, who: &str) -> Result<[u8; 3], SpriteError> {
    let bad = || SpriteError::Palette {
        who: who.to_string(),
        entry: hex.to_string(),
    };
    let b = hex.as_bytes();
    if b.len() != 7 || b[0] != b'#' {
        return Err(bad());
    }
    let mut out = [0u8; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let s = &hex[1 + i * 2..3 + i * 2];
        *slot = u8::from_str_radix(s, 16).map_err(|_| bad())?;
    }
    Ok(out)
}

/// One palette entry through one variant tilt.
///
/// Saturating, not wrapping: `0xff * 1.12` is 285, and letting that wrap would
/// turn the brightest highlight in a palette into a dark smear on exactly the
/// variant meant to be the warmest.
pub(super) fn tinted(rgb: [u8; 3], mul: [f32; 3]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (rgb[i] as f32 * mul[i]).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Build the pose -> index table, resolving `fallback` chains at construction.
///
/// Authored poses always win: a fallback entry for a pose that genuinely exists is
/// ignored rather than shadowing it. A fallback that leads nowhere is an error —
/// it is a facade bug, and the alternative (returning `None` and drawing nothing)
/// is an invisible creature, which is the single hardest rendering bug to trace
/// back to its cause.
pub(super) fn build_ids(
    art: &SpriteArt,
    who: &str,
) -> Result<[Option<StateId>; POSE_COUNT], SpriteError> {
    let mut ids: [Option<StateId>; POSE_COUNT] = [None; POSE_COUNT];
    for (i, pose) in art.poses.iter().enumerate() {
        if ids[pose.index()].is_some() {
            return Err(SpriteError::DuplicatePose {
                who: who.to_string(),
                pose: *pose,
            });
        }
        ids[pose.index()] = Some(StateId(i as u8));
    }

    for (from, first) in art.fallback.iter().copied() {
        if ids[from.index()].is_some() {
            continue; // authored wins
        }
        // Chase the chain until it lands on a real sequence. Bounded by a seen
        // mask so `[(A, B), (B, A)]` is an error instead of a hang at load. Only
        // fourteen poses exist, so the mask is a `u16` and the walk allocates
        // nothing.
        let mut seen: u16 = 1 << from.index();
        let mut cur = from;
        let mut next = Some(first);
        while let Some(step) = next {
            cur = step;
            if ids[step.index()].is_some() {
                break;
            }
            if seen & (1 << step.index()) != 0 {
                return Err(SpriteError::FallbackCycle {
                    who: who.to_string(),
                    from,
                });
            }
            seen |= 1 << step.index();
            next = art
                .fallback
                .iter()
                .find(|(f, _)| *f == step)
                .map(|(_, t)| *t);
        }
        match ids[cur.index()] {
            Some(id) => ids[from.index()] = Some(id),
            None => {
                return Err(SpriteError::FallbackDangling {
                    who: who.to_string(),
                    from,
                    to: first,
                });
            }
        }
    }
    Ok(ids)
}
