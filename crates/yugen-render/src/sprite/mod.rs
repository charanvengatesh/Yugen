//! The one baked sprite. The player, item icons and every creature draw through
//! this.
//!
//! A 1:1 port of three TypeScript files, merged the way they were always meant
//! to be read together:
//!
//! | TypeScript | Here |
//! |---|---|
//! | `src/sprite/SpriteArt.ts` | [`Pose`], [`PlayMode`], [`SpriteClock`], [`SeqSpec`], [`SpriteArt`], [`split_frames`], [`pick_frame_index`] |
//! | `src/sprite/Sprite.ts` | [`VARIANT_TINT`], [`BakedSprite`] |
//! | `src/sprite/fromContent.ts` | [`ContentArt`], [`ContentSeq`], [`FromContentOpts`], [`sprite_art_from_content`] |
//!
//! Their headers are the design document for this module and most of what they
//! say is reproduced below, next to the code it explains. Three things are worth
//! reading before calling anything here.
//!
//! # ONE SPRITE PIXEL IS ONE WORLD CELL
//!
//! The invariant the whole system exists to protect. A frame is `cells_h` strings
//! of `cells_w` characters; it bakes to exactly `cells_w x cells_h` texels and is
//! drawn, nearest-neighbour, into a rect of `cells_w * CELL_SIZE` by
//! `cells_h * CELL_SIZE` world px. That is why creatures, the player and item
//! icons read at the same grain as the sand and stone they sit on instead of
//! looking like smooth stickers pasted onto a chunky world.
//!
//! Every size here is therefore a whole number of CELLS, never a scale factor: a
//! 1.1x mob would put sprite pixels at 5.5px against 5px cells and reintroduce
//! exactly the mismatch this system was built to kill. Cosmetic variation is
//! carried by TINT, which has no geometry — see [`VARIANT_TINT`].
//!
//! # RESOLVE POSES ONCE, AT LOAD
//!
//! [`BakedSprite::state_id`] maps a [`Pose`] to the dense sequence index, with
//! `fallback` already applied. Call it once when you build your facade and keep
//! the [`StateId`]; everything on the draw path takes the resolved id. The
//! TypeScript said this loudly because there `stateId` was a `Map<string, …>`
//! lookup and a per-frame call would have put a hash in the hottest loop the
//! renderer has. Here it is an array index into a 14-slot table, so the cost
//! argument is gone — but the CORRECTNESS argument is not: resolving at load is
//! what turns "content has no wall-slide sequence" into a startup failure instead
//! of a character that renders nothing on one wall, six months later.
//!
//! # BAKE ONCE
//!
//! The TypeScript rasterised every pose of every variant into a `<canvas>` in the
//! constructor, at module init, and `draw` was a single `drawImage`. Nothing was
//! allocated after construction. The same promise is kept here, in the shape Bevy
//! wants: [`SpritePlugin`] rasterises the whole sprite table once in `PreStartup`
//! and publishes [`SpriteAtlases`] — one `Handle<Image>` and one
//! `Handle<TextureAtlasLayout>` per sprite. A draw is then a tile index. Nothing
//! rasterises per frame, ever.
//!
//! # What the port changed
//!
//! - **Strings became enums.** The TypeScript keyed sequences by name; the
//!   compiler emits the same vocabulary as two closed enums (`SpriteSeqState`,
//!   twelve members, for the sprite table; `MobSpecSeqState`, three, for a mob's
//!   `art` group). [`Pose`] is their union, declared once here, and the two
//!   `From` impls are exhaustive matches — so a state added to a schema stops
//!   compiling until code decides what the thing has to be doing to reach it.
//!   That is the "red squiggle rather than an invisible player" property
//!   `PlayerArt.ts` wanted and could only approximate with a `satisfies` witness.
//! - **Throws became `Result`.** Every `throw new Error` is a [`SpriteError`].
//!   The failure POLICY is unchanged, and it is the point: malformed art is loud
//!   at load, never degraded, because a frame that bakes once bakes forever and a
//!   silent skip ships as a hole in a creature. [`SpritePlugin`] panics on one,
//!   which is what a module-init throw did.
//! - **`fromContent`'s mode check is gone**, because it cannot fail. It existed
//!   to narrow a generated `string` to the `PlayMode` union; the Rust tables
//!   already carry a closed enum, so the check is `impl From`.
//! - **`Sprite`'s runtime bounds checks are gone.** `draw` guarded
//!   `state < 0 || state >= specs.length` because it took a raw number. A
//!   [`StateId`] is opaque and only obtainable from the sprite that issued it,
//!   so the guard has nothing left to catch.
//! - **The bitmaps are one atlas per sprite, not one canvas per pose.** Same
//!   pixels, same dedup, one texture bind. See [`BakedSprite::image`].
//! - **`draw` / `drawGlow` / `drawStill` did not come across.** They were
//!   Canvas2D — `ctx.drawImage` plus a `"lighter"` composite for the damage
//!   flash and self-luminance. Here the caller spawns a Bevy entity and the
//!   engine draws it; what those three methods COMPUTED (which tile, flipped or
//!   not) is published as [`BakedSprite::tile`], [`BakedSprite::still_tile`] and
//!   [`BakedSprite::tile_uv`]. The additive passes belong to whoever owns the
//!   drawn thing, not here: `flash` is a tint on the same sprite, and the glow
//!   is a second entity above the light composite on an additive material — see
//!   [`crate::mobs`], which does both for creatures.
//!
//! # Naming
//!
//! The class is [`BakedSprite`] and not `Sprite`, because `bevy::prelude::Sprite`
//! is a component every consumer of this module also imports. The rename is
//! purely to keep `use` lists honest; it is the TypeScript's `Sprite` class.

mod anim;
mod baked;
mod content;
mod vocab;

pub use anim::*;
pub use baked::*;
pub use content::*;
pub use vocab::*;

use bevy::image::{TextureAtlas, TextureAtlasLayout};
use bevy::prelude::*;

use yugen_data::sprites::{SPRITE_COUNT, SPRITES};

// ---------------------------------------------------------------------------
// The Bevy plumbing
// ---------------------------------------------------------------------------

/// One sprite's baked art plus the two handles a `Sprite` component needs.
#[derive(Clone, Debug)]
pub struct SpriteAtlas {
    /// The strip. One bind for the whole sprite.
    pub image: Handle<Image>,
    /// `bake_count` tiles across, one down.
    pub layout: Handle<TextureAtlasLayout>,
    /// The rasteriser's own output: tile lookup, cadence, geometry.
    pub baked: BakedSprite,
}

impl SpriteAtlas {
    /// A [`TextureAtlas`] pointing at one tile.
    pub fn atlas_at(&self, tile: usize) -> TextureAtlas {
        TextureAtlas {
            layout: self.layout.clone(),
            index: tile,
        }
    }

    /// A ready `Sprite` component showing one tile at one texel per world cell.
    ///
    /// `custom_size` is the art rect in world px — `cells * CELL_SIZE` — which is
    /// what keeps the blit at exactly one sprite pixel per world cell. A caller
    /// that wants squash-stretch (the player's landing squash, its rise stretch)
    /// overwrites `custom_size` afterwards; that is what the TypeScript's explicit
    /// `w`/`h` arguments to `draw` were for, and why they were not just `wPx`/`hPx`.
    pub fn sprite_at(&self, tile: usize, flip_x: bool) -> Sprite {
        Sprite {
            image: self.image.clone(),
            texture_atlas: Some(self.atlas_at(tile)),
            flip_x,
            custom_size: Some(Vec2::new(self.baked.w_px, self.baked.h_px)),
            ..default()
        }
    }

    /// Frame 0 of the first authored sequence, unflipped — the still an item icon
    /// or a bestiary portrait wants. The TypeScript's `drawStill` with both its
    /// defaults.
    pub fn still(&self) -> Sprite {
        self.sprite_at(self.baked.still_tile(self.baked.first_state(), 0), false)
    }
}

/// Every sprite in the table, rasterised and uploaded.
///
/// Indexed by sprite code, because that is what the table is: `SPRITES` is
/// index == code and so is this. [`SpriteAtlases::get`] takes an id for the callers
/// that hold one — `ITEM_ICONS` publishes ids, not codes — and is the linear scan
/// [`sprite_code_of`] describes, which belongs at load and not in a draw loop.
///
/// Every slot is `Some` in practice; the `Option` exists so a sprite that failed to
/// bake could in principle be reported without shifting every code after it. Today
/// one that fails to bake panics — see [`SpritePlugin`].
///
/// # Why this is `Clone`
///
/// So a consumer that cannot borrow the world can hold one — see [`crate::glue`],
/// where the HUD's icon source is a `Box<dyn IconAtlas>` with no access to `Res`.
/// The clone happens once, in `Startup`.
///
/// It costs **1744 bytes**, measured in `docs/PERF.md` rather than estimated.
/// This comment previously claimed "a few megabytes", which was wrong by roughly
/// a thousand times: it assumed the clone duplicated the baked CPU pixels, but by
/// `Startup` those are in `Assets<Image>` and a [`SpriteAtlas`] holds two handles
/// and six numbers. Threading an `Arc` through the plugin to save under 2 KB, once,
/// would be the worse trade — but for the opposite reason to the one recorded here
/// before.
#[derive(Resource, Default, Clone)]
pub struct SpriteAtlases {
    by_code: Vec<Option<SpriteAtlas>>,
}

impl SpriteAtlases {
    /// The atlas for a sprite code.
    pub fn by_code(&self, code: u16) -> Option<&SpriteAtlas> {
        self.by_code.get(code as usize)?.as_ref()
    }

    /// The atlas for an authoring id, or `None` if the table has no such sprite.
    ///
    /// Resolve once and keep the [`SpriteAtlas`] or the code — see
    /// [`sprite_code_of`].
    pub fn get(&self, id: &str) -> Option<&SpriteAtlas> {
        self.by_code(sprite_code_of(id)?)
    }

    /// How many sprites are loaded. Zero before [`SpritePlugin`]'s `PreStartup`
    /// bake, which is the state a test that never ran the schedule sees.
    pub fn count(&self) -> usize {
        self.by_code.iter().filter(|s| s.is_some()).count()
    }
}

/// Bakes the whole sprite table once, before anything asks for it.
///
/// # Why `PreStartup`
///
/// So that every facade built in `Startup` — `crate::ui`'s hotbar icons,
/// `crate::player_art`'s body — can take `Res<SpriteAtlases>` and find it
/// populated, with no `.after()` between them to get wrong. The bake is one pass
/// over 51 records of a few hundred texels each; it is not worth a loading state
/// and it is emphatically not worth doing lazily on a draw path.
///
/// # Why a panic is the right failure
///
/// Malformed art is a content bug the compiler let through. The TypeScript threw
/// from a module-level `new Sprite(...)`, which took the page down at load; this
/// panics out of `PreStartup`, which takes the app down at load. Both are the same
/// deliberate choice: a hole in a creature that ships is far more expensive than a
/// startup that refuses.
///
/// # Assets
///
/// `Assets<Image>` and `Assets<TextureAtlasLayout>` come from `ImagePlugin`, and
/// [`SpritePlugin::build`] registers them only if nothing already has. The guard
/// matters both ways: `init_asset` REPLACES the store rather than skipping, so
/// calling it unconditionally after `DefaultPlugins` would drop every image the app
/// had already loaded — and without it this plugin could not be driven by a
/// headless test, which is how the plumbing below is checked.
pub struct SpritePlugin;

impl Plugin for SpritePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Assets<Image>>() {
            app.init_asset::<Image>();
        }
        if !app
            .world()
            .contains_resource::<Assets<TextureAtlasLayout>>()
        {
            app.init_asset::<TextureAtlasLayout>();
        }
        app.init_resource::<SpriteAtlases>()
            .add_systems(PreStartup, bake_sprite_table);
    }
}

/// Rasterise and upload every record in `yugen_data::sprites::SPRITES`.
///
/// No `fallback`, no `rate_scale` and no `variants` override for anything in this
/// table: it holds the player, who authors every state it can reach and plays each
/// at the rate content asked for, and 50 item icons, which have one pose and no
/// crowd. Those are all decisions the CALLER makes, so a facade that wants
/// different ones calls [`sprite_art_from_content`] itself — this system is the
/// default, not the only door. Mob art wants all three and so is not baked here:
/// [`crate::mobs`] bakes the bestiary in the same `PreStartup`, through this same
/// bridge, supplying the three from `yugen-core` — see [`MobAtlases`].
///
/// [`MobAtlases`]: crate::mobs::MobAtlases
fn bake_sprite_table(
    mut atlases: ResMut<SpriteAtlases>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<TextureAtlasLayout>>,
) {
    let opts = FromContentOpts::default();
    let mut out = Vec::with_capacity(SPRITE_COUNT);
    for def in SPRITES.iter() {
        let art = sprite_art_from_content(def, def.id, &opts)
            .unwrap_or_else(|e| panic!("content/sprites: {e}"));
        let baked =
            BakedSprite::new(&art, def.id).unwrap_or_else(|e| panic!("content/sprites: {e}"));
        out.push(Some(SpriteAtlas {
            image: images.add(baked.image()),
            layout: layouts.add(baked.layout()),
            baked,
        }));
    }
    atlases.by_code = out;
}

/// THE ONE AXIS FLIP. Sim `(left, top)` and an extent, to a Bevy translation.
///
/// The simulation, the collision boxes, the cell grid and every art row are +y
/// DOWN — row 0 is the top of a creature. Bevy is +y UP. The conversion is a single
/// negation, and this is the single place in the sprite system that performs it, so
/// that no facade has to remember which way its own numbers point.
///
/// Nothing flips in the PIXELS. A texture's row 0 is already drawn at the top of a
/// quad, so a frame authored top-down arrives on screen top-down with no work;
/// inverting the atlas rows and then inverting the transform back would be two bugs
/// that cancel until one of them is fixed.
///
/// The top-left is rounded and the half-extents added afterwards, rather than the
/// centre being rounded, so an odd extent does not land the sprite's edges on half
/// pixels — the rule `crate::player::place_body` and `crate::mobs::place_mobs`
/// already snap with. A fractional edge is the one thing the low-res target cannot
/// forgive: see `crate::lowres`.
pub fn world_translation(left: f32, top: f32, w: f32, h: f32, z: f32) -> Vec3 {
    let left = left.round();
    let top = top.round();
    Vec3::new(left + w * 0.5, -(top + h * 0.5), z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::TextureFormat;
    use yugen_core::config::CELL_SIZE;
    use yugen_core::items::ITEM_ICONS;
    use yugen_data::mobs::MobSpecArt;
    use yugen_data::sprites::sprite;
    use yugen_data::sprites::{SpriteSeqMode, SpriteSeqState};

    /// A 2x2 art table with a two-colour palette, for the rasteriser tests.
    fn art(frames: Vec<Frame>, mode: PlayMode) -> SpriteArt {
        SpriteArt {
            cells_w: 2,
            cells_h: 2,
            grain: 1,
            pal: &[".", "#102030", "#ff8000"],
            variants: 1,
            poses: vec![Pose::Idle],
            seqs: vec![SeqSpec {
                frames,
                mode,
                fps: 10.0,
                blink_every: 0.0,
                blink_for: 0.1,
            }],
            fallback: Vec::new(),
        }
    }

    fn one_frame(rows: [&'static str; 2]) -> Frame {
        rows.to_vec()
    }

    /// The four RGBA bytes of one texel of a baked strip.
    fn texel(b: &BakedSprite, tile: usize, x: u32, y: u32) -> [u8; 4] {
        let w = b.atlas_size().x as usize;
        let i = ((y as usize * w) + tile * b.cells_w as usize + x as usize) * 4;
        [
            b.pixels()[i],
            b.pixels()[i + 1],
            b.pixels()[i + 2],
            b.pixels()[i + 3],
        ]
    }

    // -- Rasterising --------------------------------------------------------

    #[test]
    fn a_frame_bakes_one_texel_per_cell_at_the_authored_palette_colour() {
        let a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();

        assert_eq!(b.atlas_size(), UVec2::new(2, 2), "one texel per cell");
        assert_eq!(texel(&b, 0, 0, 0), [0x10, 0x20, 0x30, 255]);
        assert_eq!(texel(&b, 0, 1, 0), [0xff, 0x80, 0x00, 255]);
        assert_eq!(texel(&b, 0, 0, 1), [0xff, 0x80, 0x00, 255]);
        assert_eq!(texel(&b, 0, 1, 1), [0x10, 0x20, 0x30, 255]);
    }

    #[test]
    fn the_art_rect_is_a_whole_number_of_cells_in_world_px() {
        // The invariant the module header is about: geometry is always cells times
        // CELL_SIZE, never a scale factor.
        let a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.w_px, (2 * CELL_SIZE) as f32);
        assert_eq!(b.h_px, (2 * CELL_SIZE) as f32);
    }

    #[test]
    fn both_spellings_of_transparent_are_left_unpainted_and_fully_clear() {
        // '.' and '0' mean the same thing, and index 0 of the palette is never
        // parsed — which is why content is free to write "." there.
        let a = art(vec![one_frame([".0", "0."])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(
                texel(&b, 0, x, y),
                [0, 0, 0, 0],
                "({x},{y}) should be clear"
            );
        }
    }

    #[test]
    fn the_transparent_slot_is_never_parsed_as_a_colour() {
        // Slot 0 of every palette in the game is ".", which is not #rrggbb. If the
        // rasteriser parsed it, nothing in the game would bake at all.
        let a = art(vec![one_frame(["11", "11"])], PlayMode::Hold);
        assert!(BakedSprite::new(&a, "t").is_ok());
    }

    #[test]
    fn a_row_of_the_wrong_width_is_a_construction_error() {
        let a = art(vec![one_frame(["123", "12"])], PlayMode::Hold);
        assert_eq!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::RowWidth {
                who: "t.Idle[0]".into(),
                row: 0,
                chars: 3,
                expected: 2
            }
        );
    }

    #[test]
    fn a_frame_with_the_wrong_number_of_rows_is_a_construction_error() {
        let a = art(vec![vec!["12"]], PlayMode::Hold);
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::FrameRows {
                rows: 1,
                expected: 2,
                ..
            }
        ));
    }

    #[test]
    fn a_character_outside_the_palette_is_a_construction_error() {
        // The bug the original's header describes: 'a' is not a digit, and silently
        // skipping it would leave a hole in a creature that ships. '9' IS a digit
        // and is still out of range for a two-colour palette.
        for bad in ["a2", "92"] {
            let a = art(vec![one_frame([bad, "12"])], PlayMode::Hold);
            assert!(
                matches!(
                    BakedSprite::new(&a, "t").unwrap_err(),
                    SpriteError::PaletteIndex { row: 0, col: 0, .. }
                ),
                "{bad:?} should not bake"
            );
        }
    }

    #[test]
    fn a_palette_entry_that_is_not_six_hex_digits_is_a_construction_error() {
        for pal in [
            &[".", "#12345", "#ff8000"],
            &[".", "102030", "#ff8000"],
            &[".", "#nothex", "#ff8000"],
            &[".", "#1234567", "#ff8000"],
        ] {
            let mut a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
            a.pal = pal.as_slice();
            assert!(
                matches!(
                    BakedSprite::new(&a, "t").unwrap_err(),
                    SpriteError::Palette { .. }
                ),
                "{:?} should not parse",
                pal[1]
            );
        }
    }

    #[test]
    fn a_tiles_uv_rect_is_its_share_of_the_strip_and_mirrors_in_place() {
        // Three distinct frames, so the strip is three tiles wide.
        let mut a = art(vec![one_frame(["12", "21"])], PlayMode::Loop);
        a.seqs[0].frames = vec![
            one_frame(["12", "21"]),
            one_frame(["11", "22"]),
            one_frame(["22", "11"]),
        ];
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.bake_count, 3);

        // The middle tile is the middle third, and v always spans the strip.
        assert_eq!(
            b.tile_uv(1, false),
            Vec4::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0)
        );
        // Flipped, it starts at the far edge and walks back — the same texels in
        // the other order, so the figure turns without moving.
        let flipped = b.tile_uv(1, true);
        assert_eq!(flipped, Vec4::new(2.0 / 3.0, 0.0, -1.0 / 3.0, 1.0));
        let plain = b.tile_uv(1, false);
        assert_eq!(flipped.x + flipped.z, plain.x, "same span, other end");

        // Tiles tile: every one starts where the last one ended.
        assert_eq!(b.tile_uv(0, false).x, 0.0);
        assert_eq!(b.tile_uv(2, false).x + b.tile_uv(2, false).z, 1.0);
    }

    // -- Pose sharing and variants -----------------------------------------

    #[test]
    fn a_pose_repeated_across_sequences_bakes_once() {
        // The bake cache, and the reason `bake_count` is published: content has no
        // way to say "this is the same bitmap", so keying on the frame's own text
        // recovers the sharing PlayerSprite only had because a human hand-assigned
        // the same canvas twice.
        let neutral = one_frame(["12", "21"]);
        let other = one_frame(["11", "22"]);
        let mut a = art(vec![neutral.clone()], PlayMode::Loop);
        a.poses = vec![Pose::Idle, Pose::Run];
        a.seqs = vec![
            SeqSpec {
                frames: vec![neutral.clone(), other.clone()],
                mode: PlayMode::Loop,
                fps: 8.0,
                blink_every: 0.0,
                blink_for: 0.0,
            },
            // Run reuses idle's two poses, in the other order.
            SeqSpec {
                frames: vec![other, neutral],
                mode: PlayMode::Loop,
                fps: 8.0,
                blink_every: 0.0,
                blink_for: 0.0,
            },
        ];
        let b = BakedSprite::new(&a, "t").unwrap();

        assert_eq!(b.bake_count, 2, "four frames, two distinct poses");
        let idle = b.state_id(Pose::Idle).unwrap();
        let run = b.state_id(Pose::Run).unwrap();
        // Idle's frame 0 and run's frame 1 are the same bitmap, so the same tile.
        let late_run = SpriteClock {
            state_t: 0.2,
            ..default()
        };
        assert_eq!(b.frame_index(run, &late_run), 1);
        assert_eq!(b.still_tile(idle, 0), b.tile(run, 0, &late_run));
    }

    #[test]
    fn each_variant_is_the_same_pose_through_its_own_tint() {
        let mut a = art(vec![one_frame(["11", "11"])], PlayMode::Hold);
        a.variants = 3;
        let b = BakedSprite::new(&a, "t").unwrap();

        assert_eq!(b.bake_count, 3, "one pose, three tints, three tiles");
        let s = b.first_state();
        let v0 = texel(&b, b.still_tile(s, 0), 0, 0);
        let v1 = texel(&b, b.still_tile(s, 1), 0, 0);
        // Variant 0 is the identity tilt, so it is the authored colour exactly.
        assert_eq!(v0, [0x10, 0x20, 0x30, 255]);
        // Variant 1 is warmer: more red, less blue. That is the whole point of the
        // table — two of a kind must not be pixel-identical.
        assert_ne!(v0, v1);
        assert!(v1[0] > v0[0] && v1[2] < v0[2]);
    }

    #[test]
    fn a_variant_tilt_saturates_rather_than_wrapping_round_to_black() {
        // 0xff * 1.12 is 285 and 0xff * 1.02 is 260. Wrapping either would turn the
        // brightest highlight in a palette into a dark smear on exactly the variant
        // meant to be the warmest.
        let mut a = art(vec![one_frame(["22", "22"])], PlayMode::Hold);
        a.pal = &[".", "#102030", "#ffffff"];
        a.variants = 2;
        let b = BakedSprite::new(&a, "t").unwrap();
        let warm = texel(&b, b.still_tile(b.first_state(), 1), 0, 0);
        assert_eq!(
            [warm[0], warm[1]],
            [255, 255],
            "both channels this tilt brightens clamp at white"
        );
        // Blue is tilted DOWN by the same tint, so it is the one channel with
        // somewhere left to go — which is what makes the variant read warm at all.
        assert_eq!(warm[2], (255.0f32 * 0.9).round() as u8);
        assert_eq!(warm[3], 255);
    }

    #[test]
    fn asking_for_more_variants_than_the_tint_table_has_is_a_construction_error() {
        let mut a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        a.variants = VARIANT_COUNT + 1;
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::VariantCount { .. }
        ));
        a.variants = 0;
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::VariantCount { .. }
        ));
    }

    #[test]
    fn a_variant_index_from_an_entity_id_wraps_instead_of_indexing_out() {
        // Variants are assigned from entity ids and hashes, so the sprite wraps
        // rather than trusts.
        let mut a = art(vec![one_frame(["11", "11"])], PlayMode::Hold);
        a.variants = 3;
        let b = BakedSprite::new(&a, "t").unwrap();
        let s = b.first_state();
        assert_eq!(b.still_tile(s, 7), b.still_tile(s, 1));
        assert_eq!(b.still_tile(s, 300), b.still_tile(s, 0));
    }

    // -- The frame picker ---------------------------------------------------

    fn strip(n: usize) -> SeqSpec {
        SeqSpec {
            frames: (0..n).map(|_| one_frame(["12", "21"])).collect(),
            mode: PlayMode::Loop,
            fps: 10.0,
            blink_every: 0.0,
            blink_for: 0.1,
        }
    }

    #[test]
    fn a_single_frame_sequence_is_frame_zero_in_every_mode() {
        for mode in [
            PlayMode::Hold,
            PlayMode::Loop,
            PlayMode::Once,
            PlayMode::Phase,
            PlayMode::Ambient,
        ] {
            let mut s = strip(1);
            s.mode = mode;
            let clock = SpriteClock {
                state_t: 99.0,
                clock_t: 99.0,
                phase: 0.9,
            };
            assert_eq!(pick_frame_index(&s, &clock), 0, "{mode:?}");
        }
    }

    #[test]
    fn hold_ignores_the_clock_entirely() {
        let mut s = strip(4);
        s.mode = PlayMode::Hold;
        let clock = SpriteClock {
            state_t: 99.0,
            clock_t: 99.0,
            phase: 0.75,
        };
        assert_eq!(pick_frame_index(&s, &clock), 0);
    }

    #[test]
    fn loop_advances_with_the_state_clock_and_wraps() {
        let s = strip(4); // 10 fps
        for (t, want) in [
            (0.0, 0),
            (0.05, 0),
            (0.1, 1),
            (0.35, 3),
            (0.4, 0),
            (0.45, 0),
        ] {
            let clock = SpriteClock {
                state_t: t,
                ..default()
            };
            assert_eq!(pick_frame_index(&s, &clock), want, "t={t}");
        }
    }

    #[test]
    fn once_holds_its_last_frame_forever() {
        // A one-shot that wrapped would replay its own impact forever, which is
        // precisely wrong for land, punch and double-jump.
        let mut s = strip(3);
        s.mode = PlayMode::Once;
        for (t, want) in [(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 2), (100.0, 2)] {
            let clock = SpriteClock {
                state_t: t,
                ..default()
            };
            assert_eq!(pick_frame_index(&s, &clock), want, "t={t}");
        }
    }

    #[test]
    fn phase_indexes_off_distance_travelled_rather_than_time() {
        // The run cycle. A half-speed run plays its contacts at half rate with no
        // extra state, because the clock this reads is advanced by real horizontal
        // speed and not by dt — which is what stops the legs sliding.
        let mut s = strip(4);
        s.mode = PlayMode::Phase;
        for (p, want) in [(0.0, 0), (0.24, 0), (0.25, 1), (0.5, 2), (0.99, 3)] {
            let clock = SpriteClock {
                phase: p,
                state_t: 99.0,
                clock_t: 99.0,
            };
            assert_eq!(pick_frame_index(&s, &clock), want, "phase={p}");
        }
    }

    #[test]
    fn a_negative_clock_wraps_back_into_range_instead_of_blanking_the_sprite() {
        // One frame of negative dt — a window regaining focus, a debug scrub — must
        // not index out of the array. It reads as a flicker and looks like a GPU
        // problem.
        let mut s = strip(4);
        let clock = SpriteClock {
            state_t: -1.0,
            ..default()
        };
        assert_eq!(pick_frame_index(&s, &clock), 0, "loop clamps state_t at 0");

        s.mode = PlayMode::Phase;
        let back = SpriteClock {
            phase: -0.3,
            ..default()
        };
        // floor(-0.3 * 4) is -2, wrapped into 0..4 is 2.
        assert_eq!(pick_frame_index(&s, &back), 2);

        s.mode = PlayMode::Once;
        let neg = SpriteClock {
            state_t: -5.0,
            ..default()
        };
        assert_eq!(pick_frame_index(&s, &neg), 0);
    }

    #[test]
    fn a_non_finite_clock_falls_back_to_the_neutral_pose() {
        // The TypeScript produced a NaN index here, indexed `undefined`, and drew
        // nothing for one frame. Frame 0 is the authored neutral pose of every
        // sequence, so this is strictly the better of the two — see `wrap_index`.
        let mut s = strip(4);
        for mode in [
            PlayMode::Loop,
            PlayMode::Phase,
            PlayMode::Once,
            PlayMode::Ambient,
        ] {
            s.mode = mode;
            let clock = SpriteClock {
                state_t: f32::NAN,
                clock_t: f32::NAN,
                phase: f32::NAN,
            };
            assert_eq!(pick_frame_index(&s, &clock), 0, "{mode:?}");
        }
    }

    #[test]
    fn an_ambient_sequence_with_no_blink_period_loops_over_every_frame() {
        // Leaving a permanently-invisible last frame would be a silent art bug, so
        // the mode gives it back rather than hiding it.
        let mut s = strip(3);
        s.mode = PlayMode::Ambient;
        s.fps = 2.0; // beat 0.5s
        s.blink_every = 0.0;
        let seen: Vec<usize> = [0.0, 0.5, 1.0, 1.5]
            .iter()
            .map(|t| {
                pick_frame_index(
                    &s,
                    &SpriteClock {
                        clock_t: *t,
                        ..default()
                    },
                )
            })
            .collect();
        assert_eq!(seen, vec![0, 1, 2, 0]);
    }

    #[test]
    fn an_ambient_sequence_reserves_its_last_frame_for_the_blink() {
        // The blink is the LAST frame by convention, so adding one to an existing
        // loop is an append rather than an index rewrite.
        let mut s = strip(3);
        s.mode = PlayMode::Ambient;
        s.fps = 2.0; // beat 0.5s
        s.blink_every = 4.0;
        s.blink_for = 0.2;

        // Inside the blink window at the top of every period.
        for t in [0.0, 0.1, 4.0, 4.1] {
            let i = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t,
                    ..default()
                },
            );
            assert_eq!(i, 2, "t={t} is inside the blink");
        }
        // Outside it, the breath cycles over frames 0..n-2 only.
        for t in [0.3, 0.9, 1.4, 2.6, 3.9] {
            let i = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t,
                    ..default()
                },
            );
            assert!(
                i < 2,
                "the breath must not show the blink frame, t={t} gave {i}"
            );
        }
    }

    #[test]
    fn the_ambient_breath_restarts_in_step_with_every_blink() {
        // Both cycles read the SAME wrapped t, so the idle reads as one repeating
        // gesture rather than two unrelated ones beating against each other.
        let mut s = strip(3);
        s.mode = PlayMode::Ambient;
        s.fps = 2.0;
        s.blink_every = 4.0;
        s.blink_for = 0.2;
        for t in [0.3, 0.9, 1.4, 3.9] {
            let a = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t,
                    ..default()
                },
            );
            let b = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t + 4.0,
                    ..default()
                },
            );
            assert_eq!(a, b, "the breath should repeat with the period, t={t}");
        }
    }

    #[test]
    fn the_ambient_beat_divides_where_every_other_mode_multiplies() {
        // The old animation divided by a period of 0.62s, so the facade authors
        // `fps: 1/0.62` and this mode divides by `1/fps` to make the two
        // equivalent. In f64 that round trip was exact; the compiler emits f32, so
        // it is exact to about one part in 1e7 — the one place the port loses the
        // bit-for-bit equivalence the original had, worth eight frames an hour at
        // 60fps.
        let authored = SPRITES[sprite::PLAYER as usize].seq.unwrap()[0];
        assert_eq!(authored.state, SpriteSeqState::Idle);
        assert_eq!(authored.mode, SpriteSeqMode::Ambient);
        let beat = 1.0 / authored.fps;
        assert!(
            (beat - 0.62).abs() < 1e-6,
            "the player's idle beat should round-trip to 0.62s, was {beat}"
        );
    }

    // -- split_frames -------------------------------------------------------

    #[test]
    fn split_frames_reads_a_body_as_a_filmstrip_separated_by_blank_lines() {
        // Two blanks in a row, and a trailing one: a run of blank lines separates,
        // it does not add an empty frame, and neither does a body that ends on one.
        static BODY: [&str; 7] = ["12", "21", "", "", "11", "22", ""];
        let f = split_frames(&BODY, 2, "t").unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], vec!["12", "21"]);
        assert_eq!(f[1], vec!["11", "22"]);
    }

    #[test]
    fn split_frames_rejects_a_frame_whose_row_count_disagrees_with_the_art_grid() {
        // The one error in a sprite table that renders as a plausible-looking
        // creature instead of a crash: the art and the collision box disagree.
        static BODY: [&str; 5] = ["12", "21", "", "11", "22"];
        assert!(matches!(
            split_frames(&BODY, 3, "t").unwrap_err(),
            SpriteError::FrameRows {
                frame: 0,
                rows: 2,
                expected: 3,
                ..
            }
        ));
    }

    #[test]
    fn split_frames_of_an_empty_body_is_no_frames_rather_than_one_empty_one() {
        static NONE: [&str; 0] = [];
        assert!(split_frames(&NONE, 2, "t").unwrap().is_empty());
    }

    // -- Fallback -----------------------------------------------------------

    fn two_state_art(poses: Vec<Pose>, fallback: Vec<(Pose, Pose)>) -> SpriteArt {
        let seq = SeqSpec {
            frames: vec![one_frame(["12", "21"])],
            mode: PlayMode::Hold,
            fps: 8.0,
            blink_every: 0.0,
            blink_for: 0.0,
        };
        SpriteArt {
            grain: 1,
            cells_w: 2,
            cells_h: 2,
            pal: &[".", "#102030", "#ff8000"],
            variants: 1,
            seqs: vec![seq; poses.len()],
            poses,
            fallback,
        }
    }

    #[test]
    fn an_unauthored_pose_borrows_the_sequence_the_renderer_names() {
        // "A mob with no air pose uses its walk cycle" — a rule about how the game
        // animates creatures, not a fact about any one creature.
        let a = two_state_art(vec![Pose::Idle, Pose::Move], vec![(Pose::Air, Pose::Move)]);
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Air), b.state_id(Pose::Move));
        assert_eq!(b.state_id(Pose::Air).unwrap().index(), 1);
    }

    #[test]
    fn a_fallback_chain_is_chased_until_it_lands_on_a_real_sequence() {
        let a = two_state_art(
            vec![Pose::Idle],
            vec![(Pose::Air, Pose::Move), (Pose::Move, Pose::Idle)],
        );
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Air), b.state_id(Pose::Idle));
        assert_eq!(b.state_id(Pose::Move), b.state_id(Pose::Idle));
    }

    #[test]
    fn an_authored_pose_beats_a_fallback_that_names_it() {
        let a = two_state_art(vec![Pose::Idle, Pose::Air], vec![(Pose::Air, Pose::Idle)]);
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Air).unwrap().index(), 1, "authored wins");
    }

    #[test]
    fn a_fallback_cycle_is_a_construction_error_rather_than_a_hang() {
        let a = two_state_art(
            vec![Pose::Idle],
            vec![(Pose::Air, Pose::Move), (Pose::Move, Pose::Air)],
        );
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::FallbackCycle {
                from: Pose::Air,
                ..
            }
        ));
    }

    #[test]
    fn a_fallback_that_names_no_sequence_is_a_construction_error() {
        // The alternative is returning None and drawing nothing, which is an
        // invisible creature — the single hardest rendering bug to trace.
        let a = two_state_art(vec![Pose::Idle], vec![(Pose::Air, Pose::Move)]);
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::FallbackDangling {
                from: Pose::Air,
                to: Pose::Move,
                ..
            }
        ));
    }

    #[test]
    fn a_pose_declared_twice_is_a_construction_error() {
        let a = two_state_art(vec![Pose::Idle, Pose::Idle], Vec::new());
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::DuplicatePose {
                pose: Pose::Idle,
                ..
            }
        ));
    }

    #[test]
    fn a_pose_the_sprite_never_authored_and_never_borrowed_is_absent() {
        let a = two_state_art(vec![Pose::Idle], Vec::new());
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Swim), None);
    }

    // -- The content bridge -------------------------------------------------

    #[test]
    fn a_sequence_fps_of_zero_inherits_the_sprite_wide_rate() {
        // 0 is "unset", not "deliberately 0" — otherwise retuning a sprite's base
        // rate would silently skip every sequence written at the old default.
        let def = &SPRITES[sprite::PLAYER as usize];
        let a = sprite_art_from_content(def, "player", &FromContentOpts::default()).unwrap();
        let authored = def.seq.unwrap();
        for (i, s) in authored.iter().enumerate() {
            let want = if s.fps == 0.0 { def.fps } else { s.fps };
            assert_eq!(a.seqs[i].fps, want, "{:?}", s.state);
        }
        assert!(
            authored.iter().any(|s| s.fps == 0.0),
            "the player record should exercise the inheritance path"
        );
    }

    #[test]
    fn the_rate_scale_multiplies_whichever_rate_won() {
        // A multiplier rather than an override, so a sequence that DID set its own
        // rate is scaled rather than overwritten — a mob's idle is halved either
        // way.
        let def = &SPRITES[sprite::PLAYER as usize];
        let plain = sprite_art_from_content(def, "player", &FromContentOpts::default()).unwrap();
        let scaled = sprite_art_from_content(
            def,
            "player",
            &FromContentOpts {
                rate_scale: &[(Pose::Idle, 0.5)],
                ..default()
            },
        )
        .unwrap();
        let i = plain.poses.iter().position(|p| *p == Pose::Idle).unwrap();
        assert_eq!(scaled.seqs[i].fps, plain.seqs[i].fps * 0.5);
        let r = plain.poses.iter().position(|p| *p == Pose::Run).unwrap();
        assert_eq!(
            scaled.seqs[r].fps, plain.seqs[r].fps,
            "unnamed poses are untouched"
        );
    }

    #[test]
    fn the_variant_override_beats_what_content_declared() {
        let def = &SPRITES[sprite::PLAYER as usize];
        assert_eq!(def.variants, 1, "the protagonist is not a crowd");
        let a = sprite_art_from_content(
            def,
            "player",
            &FromContentOpts {
                variants: Some(VARIANT_COUNT),
                ..default()
            },
        )
        .unwrap();
        assert_eq!(a.variants, VARIANT_COUNT);
    }

    #[test]
    fn every_sprite_in_the_compiled_table_bakes() {
        // The regression that matters: the whole shipping art set through the whole
        // rasteriser, where every failure mode above is loud.
        assert_eq!(SPRITES.len(), SPRITE_COUNT);
        for def in SPRITES.iter() {
            let art = sprite_art_from_content(def, def.id, &FromContentOpts::default())
                .unwrap_or_else(|e| panic!("{e}"));
            let baked = BakedSprite::new(&art, def.id).unwrap_or_else(|e| panic!("{e}"));
            assert!(baked.bake_count > 0, "{} baked nothing", def.id);
            assert_eq!(baked.cells_w, def.cells_w as u32);
            assert_eq!(baked.cells_h, def.cells_h as u32);
            assert_eq!(
                baked.pixels().len(),
                (baked.atlas_size().x * baked.atlas_size().y * 4) as usize
            );
        }
    }

    #[test]
    fn a_finer_grain_packs_more_texels_into_the_same_world_rect() {
        // Both halves of the invariant grain bends are asserted: the TEXELS
        // double per axis, and the DRAWN rect does not move a pixel — the whole
        // point is more resolution inside the same silhouette, and a grain that
        // changed `w_px` would be a resize wearing an experiment's name.
        //
        // Built here rather than read out of shipped content. It used to sample
        // the frostmite, which was the game's one grain-2 record; every body
        // sprite is grain 1 now, because BODY_SCALE doubled the cell box of each
        // one and halving the grain is what spreads the SAME characters over
        // twice the cells. A guard that can be switched off by a content edit is
        // not guarding the mechanism, so this constructs both grains itself and
        // pins the relationship between them.
        let coarse = art(vec![one_frame(["1.", ".2"])], PlayMode::Loop);

        let mut fine = art(vec![vec!["1..2", ".12.", ".21.", "2..1"]], PlayMode::Loop);
        fine.grain = 2;

        let baked_coarse = BakedSprite::new(&coarse, "coarse").expect("grain 1 bakes");
        let baked_fine = BakedSprite::new(&fine, "fine").expect("grain 2 bakes");

        assert_eq!(
            baked_fine.atlas_size().y,
            baked_coarse.atlas_size().y * 2,
            "texel height is cells * grain"
        );
        assert_eq!(
            baked_fine.atlas_size().x,
            baked_coarse.atlas_size().x * 2,
            "texel width is cells * grain"
        );
        assert_eq!(
            (baked_fine.w_px, baked_fine.h_px),
            (baked_coarse.w_px, baked_coarse.h_px),
            "the drawn rect is still cells * CELL_SIZE -- grain must never touch it"
        );
    }

    #[test]
    fn mob_art_bakes_through_the_same_bridge_as_the_sprite_table() {
        // The reason `ContentArt` is a trait: the identical schema fragment is
        // printed under two names, and binding the bridge to one would make it
        // unusable by the other host.
        let mut seen = 0;
        for spec in yugen_data::mobs::MOBS.iter() {
            let Some(art) = spec.art.as_ref() else {
                continue;
            };
            let a = sprite_art_from_content(
                art,
                spec.id,
                &FromContentOpts {
                    fallback: &[(Pose::Air, Pose::Move)],
                    rate_scale: &[(Pose::Idle, 0.5)],
                    variants: Some(VARIANT_COUNT),
                },
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let baked = BakedSprite::new(&a, spec.id).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(baked.variants, VARIANT_COUNT);
            // Whatever a creature authored, the renderer's borrowing rule means it
            // always has something to draw in the air.
            assert!(
                baked.state_id(Pose::Air).is_some(),
                "{} has no air pose",
                spec.id
            );
            seen += 1;
        }
        assert!(seen > 0, "no mob in the table carries art");
    }

    #[test]
    fn a_record_with_no_sequences_is_a_construction_error() {
        // `MobSpecArt` makes `seq` optional, so this is a shape the tables can
        // actually hold — and a sprite with no sequences has nothing to draw.
        let art = MobSpecArt {
            grain: None,
            cells_w: 2,
            cells_h: 2,
            pal: &[".", "#102030"],
            fps: None,
            variants: None,
            seq: None,
        };
        assert!(matches!(
            sprite_art_from_content(&art, "t", &FromContentOpts::default()).unwrap_err(),
            SpriteError::NoSequences { .. }
        ));
    }

    // -- The id lookup ------------------------------------------------------

    #[test]
    fn every_item_icon_id_names_a_sprite_the_table_has() {
        // `ITEM_ICONS` validated its ids against this same table at compile time,
        // so a miss here means the two generated modules are out of step.
        let mut found = 0;
        for icon in ITEM_ICONS.iter().flatten() {
            assert!(sprite_code_of(icon).is_some(), "{icon} names no sprite");
            found += 1;
        }
        assert!(found > 0, "no item carries an icon");
    }

    #[test]
    fn a_sprite_id_the_table_does_not_have_degrades_to_no_icon() {
        assert_eq!(sprite_code_of("icon_not_a_thing"), None);
        assert_eq!(sprite_code_of(""), None);
        // And the resource says the same thing, so a dangling reference reaches the
        // flat-colour path rather than the renderer.
        let empty = SpriteAtlases::default();
        assert!(empty.get("icon_not_a_thing").is_none());
        assert!(empty.by_code(9999).is_none());
    }

    #[test]
    fn a_sprite_id_the_table_does_have_resolves_to_its_own_code() {
        assert_eq!(sprite_code_of("player"), Some(sprite::PLAYER));
        assert_eq!(sprite_code_of("icon_torch"), Some(sprite::ICON_TORCH));
        for def in SPRITES.iter() {
            assert_eq!(sprite_code_of(def.id), Some(def.code));
        }
    }

    // -- The atlas and the flip --------------------------------------------

    #[test]
    fn the_atlas_is_one_row_of_tiles_at_grain_texels_per_cell() {
        // The subject is the player, which is now a grain-2 sprite — so this
        // test is ALSO the check that a finer grain strides the atlas in
        // texels. It used to be called "...at_one_texel_per_cell" and assert
        // tile rects in cells; the day the player took `grain = 2` it went red,
        // which is exactly what it was for. Tile n starts n * (cells * grain)
        // texels across, and the strip is one tile tall — the property that
        // makes the offset arithmetic `tile * tile_w`.
        let def = &SPRITES[sprite::PLAYER as usize];
        let art = sprite_art_from_content(def, "player", &FromContentOpts::default()).unwrap();
        // The player is grain 1 under BODY_SCALE: doubling its cell box and
        // halving its grain is what spread the same characters over twice the
        // cells. The tile arithmetic below is the claim, and it holds at any
        // grain — so it is read off the record rather than pinned to a value.
        let g = art.grain;
        assert!(g >= 1, "grain is a positive texel rate");
        let b = BakedSprite::new(&art, "player").unwrap();
        let layout = b.layout();
        assert_eq!(layout.textures.len(), b.bake_count);
        assert_eq!(layout.size, b.atlas_size());
        let (tw, th) = (b.cells_w * b.grain, b.cells_h * b.grain);
        for (i, rect) in layout.textures.iter().enumerate() {
            assert_eq!(rect.min, UVec2::new(i as u32 * tw, 0));
            assert_eq!(rect.max - rect.min, UVec2::new(tw, th));
        }
    }

    #[test]
    fn the_image_asset_is_srgb_and_matches_the_baked_strip() {
        let a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();
        let img = b.image();
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!(img.texture_descriptor.size.width, b.atlas_size().x);
        assert_eq!(img.texture_descriptor.size.height, b.atlas_size().y);
    }

    #[test]
    fn the_one_axis_flip_puts_a_sim_top_left_rect_where_bevy_wants_its_centre() {
        // Sim +y is DOWN, Bevy +y is UP, and this is the only place in the sprite
        // system that knows it.
        let t = world_translation(100.0, 40.0, 10.0, 15.0, 0.5);
        assert_eq!(t.x, 105.0);
        assert_eq!(
            t.y, -47.5,
            "further down the sim is further down the screen"
        );
        assert_eq!(t.z, 0.5);
    }

    #[test]
    fn the_flip_snaps_the_corner_and_not_the_centre() {
        // An odd extent must not land the sprite's edges on half pixels; a
        // fractional edge is the one thing the low-res target cannot forgive.
        let t = world_translation(100.4, 40.6, 5.0, 15.0, 0.0);
        assert_eq!(t.x, 102.5, "corner rounded to 100, then half the extent");
        assert_eq!(t.y, -48.5);
    }

    // -- The plugin ---------------------------------------------------------

    /// A headless app with the sprite plugin and nothing else that draws.
    fn baked_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .add_plugins(SpritePlugin);
        app.update();
        app
    }

    #[test]
    fn the_plugin_bakes_the_whole_table_before_startup_runs() {
        // Headless: no window, no renderer, no GPU. `SpritePlugin` registers the
        // two asset stores itself when nothing else has, which is what makes this
        // possible — and the guard is what stops it wiping them when something has.
        let app = baked_app();

        let atlases = app.world().resource::<SpriteAtlases>();
        assert_eq!(atlases.count(), SPRITE_COUNT);

        let player = atlases
            .get("player")
            .expect("the player sprite is in the table");
        // 8, not 2: the character was redrawn at 4x5 and then BODY_SCALE spread
        // the same art over 8x10 cells. The number is asserted rather than
        // derived so that a bake reading the WRONG record still fails here —
        // `by_code` and `get` are two different lookups and this is the one place
        // both are checked against the same expectation.
        assert_eq!(player.baked.cells_w, 8);
        assert!(player.baked.state_id(Pose::WallSlide).is_some());
        assert_eq!(atlases.by_code(sprite::PLAYER).unwrap().baked.cells_w, 8);

        // The images really landed in the store, one per sprite and no more.
        assert_eq!(app.world().resource::<Assets<Image>>().len(), SPRITE_COUNT);
    }

    #[test]
    fn a_still_from_the_atlas_is_frame_zero_of_the_first_sequence_unflipped() {
        let app = baked_app();
        let atlases = app.world().resource::<SpriteAtlases>();
        let icon = atlases
            .get("icon_torch")
            .expect("icon_torch is in the table");

        let s = icon.still();
        assert!(!s.flip_x, "UI has no facing");
        assert_eq!(
            s.texture_atlas.as_ref().unwrap().index,
            icon.baked.still_tile(icon.baked.first_state(), 0)
        );
        assert_eq!(
            s.custom_size,
            Some(Vec2::new(icon.baked.w_px, icon.baked.h_px)),
            "one sprite pixel per world cell"
        );
        assert_eq!(s.image, icon.image);
    }

    #[test]
    fn a_facing_sprite_flips_about_its_own_axis_rather_than_the_origin() {
        // `flip_x` mirrors within the quad, which is what the TypeScript's
        // `translate(x + w, y); scale(-1, 1)` did by hand. The transform is
        // untouched, so the figure turns while staying exactly where it was.
        let app = baked_app();
        let atlases = app.world().resource::<SpriteAtlases>();
        let player = atlases.get("player").unwrap();
        let facing_left = player.sprite_at(0, true);
        let facing_right = player.sprite_at(0, false);
        assert!(facing_left.flip_x);
        assert!(!facing_right.flip_x);
        assert_eq!(facing_left.custom_size, facing_right.custom_size);
    }
}
