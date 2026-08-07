//! The screen-space overlay: the bitmap font, the hotbar, the health bar, and
//! the two full-screen cards.
//!
//! Ported from `src/ui/BuildHud.ts`, `src/ui/Hud.ts` and `src/ui/Screens.ts`.
//! Those three were one idiom — raw `fillRect` and `fillText` into the same
//! small canvas the world had just been drawn into, no DOM, no retained objects
//! — and they are one module here for the same reason: they share a coordinate
//! space, a paint order, and a font.
//!
//! # There was no bitmap font to port
//!
//! The font was said to live in these three files. It did not. All three reach
//! for the platform:
//!
//! ```text
//! ctx.font = "13px Segoe UI, Tahoma, sans-serif";
//! ctx.fillText(name, x, textY);
//! cx += ctx.measureText(name).width;
//! ```
//!
//! Every glyph in the TypeScript game came out of a system font rasteriser, at
//! whatever subpixel position and hinting the browser chose, and the original's
//! own header admits what that cost: *"Text still upscales soft — that is the
//! deal the whole game makes — but the chrome around it is exact."*
//!
//! That deal is not available here and should not be. There is no `fillText` in
//! Bevy, there is no font asset in this repository, and this crate's own
//! `Cargo.toml` already describes itself as owning a "bitmap font". So the font
//! below is **authored, not ported**. What IS ported faithfully is everything
//! the font is in service of: which tier of type each label is drawn at, what it
//! is measured with, where measurement advances the cursor to, and the exact
//! pixel geometry of every plate, well, frame, tab and pip around it.
//!
//! The one number that pins the new font to the old one is [`digits_w`]. The
//! TypeScript refused to call `measureText` for a stack count and hard-coded the
//! answer — *"the face is a fixed 10px sans, where digits advance at ~6px"*.
//! [`Face::Regular`] is a 5px cell with a 1px gap: a 6px advance, exactly. So
//! `TS 10px sans == Regular @ 1`, by construction and not by taste, and
//! `digits_w(n) == COUNT.measure(&n.to_string())` is a test below.
//!
//! # The size ladder
//!
//! The TypeScript asked for nine face sizes: 8, 10, 11, 12, 13, 14, 18, 52 and
//! 56 px. A bitmap face has one size and integer multiples of it, so those nine
//! requests collapse onto two faces and a scale by [`TextStyle::for_px`]:
//!
//! | TS px | Here | Cap height |
//! |---|---|---|
//! | 8 | [`Face::Small`] @ 1 | 5 px |
//! | 10, 11, 12, 13, 14 | [`Face::Regular`] @ 1 | 7 px |
//! | 18 | `Regular` @ 2 | 14 px |
//! | 52 | `Regular` @ 5 | 35 px |
//! | 56 | `Regular` @ 6 | 42 px |
//!
//! **The 10-to-14 band collapsing to one size is the biggest judgement call in
//! this file.** The alternative is authoring four more faces one pixel apart,
//! which at these sizes is four faces that differ by nothing a player can read.
//! The hierarchy the label row actually depends on survives intact, because the
//! TypeScript never leant on size to carry it: its own comment says *"Labels
//! dim, values bright"* — the separation is in ALPHA, and alpha ports exactly.
//! The one place size was doing real work is the 8px key tab versus the 13px
//! item name, and that distinction is kept: `Small` is a genuinely different,
//! genuinely smaller face.
//!
//! [`Face::Small`] carries digits and two marks and nothing else, because the
//! 8px tier in all three files draws exactly one thing: a single hotbar key
//! digit. Anything else asked of it renders as a solid block — see [`Face`].
//!
//! # Whole pixels, everywhere, by construction
//!
//! The buffer this draws into is upscaled with a nearest sampler (see
//! [`crate::lowres`]), so a glyph at a fractional position is not a slightly
//! soft glyph, it is a doubled column of pixels next to a dropped one. Four
//! things together make that unrepresentable rather than merely avoided:
//!
//!   1. Every field of [`UiPrim`] is an `i32` in buffer pixels. There is no way
//!      to spell a half-pixel rectangle or a half-pixel glyph run.
//!   2. Alignment is resolved into a whole-pixel left edge *when the prim is
//!      built*, by [`UiPrim::text`], not at paint time. Centring is exact and
//!      not merely rounded: every advance is even (6 or 4 px times a whole
//!      scale), so a measured run is even, so `measure / 2` has no remainder.
//!      `a_centred_run_never_lands_a_glyph_on_a_half_pixel` asserts it.
//!   3. Vertical alignment goes through [`TextStyle::baseline_from_middle`] and
//!      [`TextStyle::baseline_from_top`], which return `i32`.
//!   4. [`quad_centre`] turns a whole-pixel rect into a sprite centre that lands
//!      on the buffer's own texel grid for either buffer parity — see its doc
//!      comment for the arithmetic, and
//!      `a_quad_lands_on_texel_boundaries_for_either_buffer_parity` for the
//!      proof.
//!
//! # Which camera, and which layer
//!
//! [`crate::lowres::WORLD_LAYERS`], as a child of
//! [`crate::lowres::WorldCamera`].
//!
//! That reads backwards for a UI and is the only correct answer. `WORLD_LAYERS`
//! is not "things in the world" — it is *everything that goes into the low-res
//! buffer*. [`crate::lowres::CANVAS_LAYERS`] holds one sprite, the upscale blit
//! itself, and `lowres`'s own header says nothing else is ever on it. A HUD on
//! the canvas layer would be drawn at physical window resolution over the top of
//! the blit: crisp, hairline-thin, and completely outside the look the whole
//! game is built around. So the UI goes in the buffer with everything else and
//! gets upscaled with it.
//!
//! Screen-space-ness comes from `ChildOf(camera)`, which is the idiom
//! [`crate::sky`] and [`crate::weather`] already use for their full-view quads:
//! the world camera pans, and a child of it does not. It also inherits the
//! camera's half-pixel snap from [`crate::lowres`], which is what keeps point 4
//! above true on an odd buffer axis.
//!
//! z is [`UI_Z`] and up — above [`crate::effects`]'s hit flash at 0.9, which is
//! the highest thing any other pass draws.
//!
//! # +y
//!
//! The sim, the TypeScript and every number in this file are +y DOWN with the
//! origin at the buffer's top-left corner. Bevy is +y UP. **The flip happens in
//! exactly one function, [`quad_centre`], and nowhere else.** Every layout
//! builder, every constant, and every test below is in the TypeScript's
//! coordinates and can be read straight against the original.
//!
//! # Item icons, and where the sprite join lives
//!
//! [`IconAtlas`] is the whole of it, and it is deliberately two methods:
//!
//! ```ignore
//! fn icon_cells(&self, id: &str) -> Option<(i32, i32)>;
//! fn icon_sprite(&self, id: &str) -> Option<Sprite>;
//! ```
//!
//! `id` is what [`ITEM_ICONS`] already stores — a validated sprite id, the same
//! `&'static str` for every item naming it. `icon_cells` answers in ICON PIXELS
//! (the authored art's own grid, not screen px); the layout needs it to run
//! [`icon_scale`] and centre the result in the well, and it has to be answered
//! before anything can be placed. `icon_sprite` is the still frame, and it is a
//! whole `Sprite` rather than a `Handle<Image>` because a baked sprite is a tile
//! in a strip, not a texture of its own — the seam has no business knowing
//! whether the art arrives as an atlas index, a sub-rect or a lone image.
//! [`paint`] overwrites `custom_size` and `color` and leaves every other field
//! of it alone.
//!
//! **The join is `impl IconAtlas for SpriteAtlases`, and it lives in
//! [`crate::glue`].** `SpriteAtlases` had both halves already —
//! `baked.cells_w`/`cells_h`, and `SpriteAtlas::still()`, whose own doc comment
//! calls itself "the still an item icon or a bestiary portrait wants" — so the
//! impl is a forward with no judgement in it. It is in `glue.rs` rather than in
//! either neighbour because this module must never name a sprite type and
//! `sprite.rs` has never heard of a hotbar: a file for the joins is what stops
//! two modules that need each other from importing each other.
//!
//! [`Icons`] is what carries the atlas in, and `glue`'s `install_icons` writes it
//! at `Startup`. An app that mounts [`UiPlugin`] WITHOUT
//! [`crate::glue::GluePlugin`] leaves it `None`, and then every swatch takes the
//! flat-colour path — which is not a degraded mode but *the state the TypeScript
//! hotbar shipped in* before any item art was authored, and the state most items
//! are still in. Nothing here has to know which of the two it got.
//!
//! # What the port changed
//!
//! - **The dash pip is rasterised, not stroked.** `ctx.arc(...); ctx.fill()`
//!   antialiases its rim; magnified by the blit that is a grey halo on the one
//!   element whose whole job is to read as on or off at a glance. [`disc`]
//!   emits one whole-pixel span per scanline instead, which is what a circle in
//!   a pixel buffer is.
//! - **The health bar's fill is rounded to a whole pixel.** `barW * frac` is
//!   fractional for all but a handful of values of `health`, and the original
//!   relied on the canvas antialiasing its right edge — the one place its own
//!   "everything is integers" rule was broken.
//! - **Paint order is z, not call order.** A 2D sprite phase sorts by z and
//!   nothing else, so [`paint`] gives quad `i` a z of `UI_Z + i * UI_Z_STEP`.
//!   The display list is still built in the original's exact call order; that
//!   order is now carried explicitly instead of implicitly.
//! - **Alpha blends in linear space, not sRGB.** Canvas2D composites in sRGB;
//!   the buffer here is `Bgra8UnormSrgb` and blending happens after the transfer
//!   function. A `rgba(0,0,0,0.6)` plate over bright sky is a shade lighter than
//!   it was. Matching the original exactly would mean compositing the UI on the
//!   CPU, which is the wrong trade for a difference no side-by-side would find.
//! - **`measureText` is a multiplication.** The original called it seven times a
//!   frame to advance the label row's cursor and once more inside `note()`, and
//!   went out of its way to avoid an eighth (`digitsW`). A fixed-advance face
//!   makes all eight the same one-line function, so the special case is gone and
//!   [`digits_w`] survives only as the parity anchor described above.
//! - **The screen cards are a resource, not a call site.** `Game.ts` chose
//!   between `drawMenu`, `drawGameOver` and the playing HUD. That state machine
//!   is `scenes.rs`'s; [`UiScreen`] is the seam, and it defaults to
//!   [`UiScreen::Playing`] so this plugin is useful on its own.
//!
//! # What the port dropped on the way in
//!
//! - **Bold.** `drawMenu` and `drawGameOver` asked for `bold 56px` and
//!   `bold 52px`. There is one weight. At scale 5 and 6 the stems are five and
//!   six pixels wide already, which is heavier than the browser's bold ever was
//!   at that size.
//! - **Descenders.** `g j p q y` sit on the baseline rather than below it, so
//!   the cell is a clean 7 rows and every line-height number is exact. At a 7px
//!   cap a true descender is one pixel, and one pixel of tail costs an eighth
//!   row on every glyph in the face.
//! - **The TypeScript's `ITEM_STACK` and `MATERIALS` tables.** Both are lookups
//!   through the generated registry here — `item_by_code(..).stack` and
//!   `mat_by_code(..).name` — because that is where this port already put them.
//! - **`drawCreative`'s reliance on `slots.length`.** [`BuildTool::slots`]
//!   returns a fixed `[CellId; PALETTE_SLOTS]`, so the strip width is a constant
//!   rather than a length read off a list that could not vary anyway.

use std::borrow::Cow;

use bevy::asset::RenderAssetUsages;
use bevy::color::Srgba;
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use godgame_core::config::{MAX_HEALTH, View};
use godgame_core::interact::{BuildTool, PALETTE_SLOTS};
use godgame_core::items::registry::{ITEM_ICONS, ItemCode, item_by_code, item_for_block};
use godgame_core::items::{HOTBAR, Inventory};
use godgame_core::sim::materials::{CellId, mat_by_code};

use crate::input::Tool;
use crate::items::Pack;
use crate::lowres::{LowResTarget, WORLD_LAYERS, WorldCamera};
use crate::player::PlayerBody;

// ---------------------------------------------------------------------------
// The font
// ---------------------------------------------------------------------------

/// Widest a glyph cell gets, in font pixels. [`Face::Regular`]'s width.
///
/// Five is the narrowest cell that holds a legible `M`, `W` and `%` without them
/// collapsing into each other, and it is what nearly every pixel font since the
/// LCD character generator has settled on for the same reason.
const CELL_W: i32 = 5;

/// Tallest a glyph cell gets, in font pixels. [`Face::Regular`]'s height.
///
/// Seven rows is what a 5-wide cell needs to give lowercase a 5-row x-height
/// with a 2-row ascender. Six would put `b` and `h` at the same height as `o`.
const CELL_H: i32 = 7;

/// Blank columns between one glyph cell and the next, in font pixels.
///
/// One. It is what makes [`Face::Regular`]'s advance 6 px, which is the number
/// the TypeScript hard-coded for its 10px face in `digitsW` — see the module
/// header. It is also the minimum: at zero, `ll` and `rn` stop being two glyphs.
const GLYPH_GAP: i32 = 1;

/// Blank rows between one baseline's cell and the next, in font pixels.
///
/// Two, so [`Face::Regular`] lines are 9 px apart. Only used by
/// [`TextStyle::line_h`]; every multi-line site in the original hard-coded its
/// own leading, and those numbers are ported as they were.
const LINE_GAP: i32 = 2;

/// Which of the two authored faces a run of text is set in.
///
/// Two, and not one or five. See the size-ladder table in the module header for
/// why the 10-to-14 px band is one face, and why 8 px is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// 3x5, 4 px advance. Digits, space and a hyphen — nothing else.
    ///
    /// The 8px tier in all three ported files draws exactly one thing: a hotbar
    /// slot's key digit. Authoring 92 glyphs for a face with one caller would be
    /// 90 glyphs nobody ever looks at, and each one is a judgement call about
    /// legibility at 3 px wide that no test can make.
    ///
    /// A character this face does not have renders as a SOLID block, not the
    /// hollow box [`Face::Regular`] uses: a hollow 3x5 box is the digit zero,
    /// and a missing-glyph marker that reads as a valid character is worse than
    /// no marker at all.
    Small,
    /// 5x7, 6 px advance. Everything else.
    Regular,
}

impl Face {
    /// Cell width in font pixels.
    #[inline]
    pub const fn cell_w(self) -> i32 {
        match self {
            Face::Small => 3,
            Face::Regular => CELL_W,
        }
    }

    /// Cell height in font pixels.
    ///
    /// Also the cap height: the baseline is the bottom edge of the cell, because
    /// this font has no descenders.
    #[inline]
    pub const fn cell_h(self) -> i32 {
        match self {
            Face::Small => 5,
            Face::Regular => CELL_H,
        }
    }

    /// This face's glyph table: the characters it covers, and their rows.
    ///
    /// The two are index-parallel; `a_face_has_one_row_set_per_character` is the
    /// test that keeps them that way, and it is the only thing standing between
    /// a mistyped table and every glyph after it being the wrong shape.
    #[inline]
    const fn table(self) -> (&'static str, &'static [[u8; CELL_H as usize]]) {
        match self {
            Face::Small => (SMALL_CHARS, &SMALL_ROWS),
            Face::Regular => (REGULAR_CHARS, &REGULAR_ROWS),
        }
    }

    /// Index of `ch` in this face's table, or `None` for a character it has no
    /// glyph for.
    ///
    /// A linear scan, and deliberately: the tables are 92 and 12 entries, this
    /// runs once per DRAWN glyph and never once per measured one (measurement is
    /// fixed-advance and needs no lookup at all), and a few hundred glyphs a
    /// frame against a 92-char scan is not a number worth a hash map, a build
    /// step, or a `OnceLock` that would have to be threaded through every test
    /// of the pure layer.
    #[inline]
    fn index_of(self, ch: char) -> Option<usize> {
        self.table().0.chars().position(|c| c == ch)
    }

    /// The rows of `ch`, or this face's missing-glyph marker.
    ///
    /// Never `None`. A character with no glyph must still occupy its advance and
    /// still be visible — text that silently shortens itself is a bug that hides
    /// until someone authors an item name with an accent in it.
    #[inline]
    pub fn rows(self, ch: char) -> [u8; CELL_H as usize] {
        let (_, rows) = self.table();
        match self.index_of(ch) {
            Some(i) => rows[i],
            None => match self {
                Face::Small => SMALL_TOFU,
                Face::Regular => REGULAR_TOFU,
            },
        }
    }

    /// Every character this face has a glyph for, in table order.
    #[inline]
    pub fn chars(self) -> impl Iterator<Item = char> {
        self.table().0.chars()
    }

    /// How many glyphs this face has, missing-glyph marker excluded.
    #[inline]
    pub fn glyph_count(self) -> usize {
        self.table().1.len()
    }

    /// Is font pixel `(col, row)` of `ch` lit?
    ///
    /// Rows are stored with the leftmost column in the HIGH bit of the face's
    /// own width, so a 3-wide glyph is written `0b111` and a 5-wide one
    /// `0b11111` — both read as pixel art in the source, which is the entire
    /// reason for storing them as bits at all.
    #[inline]
    pub fn lit(self, ch: char, col: i32, row: i32) -> bool {
        if col < 0 || row < 0 || col >= self.cell_w() || row >= self.cell_h() {
            return false;
        }
        let bits = self.rows(ch)[row as usize];
        (bits >> (self.cell_w() - 1 - col)) & 1 == 1
    }
}

/// A face and a whole-number magnification of it. The port's `ctx.font`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextStyle {
    /// Which authored face.
    pub face: Face,
    /// Buffer pixels per font pixel. Always at least 1, never fractional.
    pub scale: i32,
}

impl TextStyle {
    /// The closest thing this font has to a CSS `{px}px` face.
    ///
    /// See the module header's table. The `+ 5` is a round-to-nearest on
    /// `px / 10`, and 10 is there because [`Face::Regular`] at scale 1 IS the
    /// TypeScript's 10px face — that equivalence is what `digitsW` pins down.
    pub const fn for_px(px: u32) -> TextStyle {
        if px <= 9 {
            TextStyle {
                face: Face::Small,
                scale: 1,
            }
        } else if px <= 17 {
            TextStyle {
                face: Face::Regular,
                scale: 1,
            }
        } else {
            let k = ((px + 5) / 10) as i32;
            TextStyle {
                face: Face::Regular,
                scale: if k < 2 { 2 } else { k },
            }
        }
    }

    /// Buffer pixels from one glyph's left edge to the next's.
    ///
    /// Always EVEN, for either face at any scale, and the whole-pixel guarantee
    /// for centred text rests on that — see the module header.
    #[inline]
    pub const fn advance(self) -> i32 {
        (self.face.cell_w() + GLYPH_GAP) * self.scale
    }

    /// Cap height in buffer pixels. Also the full height of the glyph cell.
    #[inline]
    pub const fn cap_h(self) -> i32 {
        self.face.cell_h() * self.scale
    }

    /// Baseline-to-baseline distance in buffer pixels.
    #[inline]
    pub const fn line_h(self) -> i32 {
        (self.face.cell_h() + LINE_GAP) * self.scale
    }

    /// Advance width of `text` in buffer pixels — the port's `measureText`.
    ///
    /// Counts CHARACTERS, not bytes: `×`, `·`, `—` and the three arrows are all
    /// multi-byte and all appear in ported strings, and measuring `"×3"` as four
    /// glyphs would push the whole label row along by two cells.
    ///
    /// Includes the trailing gap after the last glyph, which is what a Canvas2D
    /// advance width does too and what makes `measure` compose: the original's
    /// label row is a chain of `cx += ctx.measureText(part).width`, and every
    /// link of it works out the same here.
    #[inline]
    pub fn measure(self, text: &str) -> i32 {
        text.chars().count() as i32 * self.advance()
    }

    /// Baseline for text the original centred vertically on `cy`
    /// (`textBaseline: "middle"`).
    ///
    /// Floors rather than rounds, so a cell with an odd cap height sits half a
    /// pixel HIGH of true centre. That direction is chosen: the callers are text
    /// inside a plate, and clearing the plate's bottom edge matters more than
    /// clearing its top, which has a lit 1px highlight on it.
    #[inline]
    pub const fn baseline_from_middle(self, cy: i32) -> i32 {
        cy + self.cap_h() / 2
    }

    /// Baseline for text the original hung from `top` (`textBaseline: "top"`).
    #[inline]
    pub const fn baseline_from_top(self, top: i32) -> i32 {
        top + self.cap_h()
    }
}

/// Where a run of text sits relative to the x it was given — `ctx.textAlign`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    /// `x` is the left edge.
    Left,
    /// `x` is the centre.
    Centre,
    /// `x` is the right edge.
    Right,
}

impl Align {
    /// The left edge of a `width`-wide run anchored at `x`.
    ///
    /// Exact for all three variants: `width` is always even (see
    /// [`TextStyle::advance`]) so the `Centre` division has no remainder, and
    /// there is nothing to round.
    #[inline]
    const fn left_edge(self, x: i32, width: i32) -> i32 {
        match self {
            Align::Left => x,
            Align::Centre => x - width / 2,
            Align::Right => x - width,
        }
    }
}

/// Characters [`Face::Regular`] has glyphs for, in [`REGULAR_ROWS`] order.
const REGULAR_CHARS: &str = " !\"#%&'()*+,-./0123456789:;<=>?\
    ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_\
    abcdefghijklmnopqrstuvwxyz\
    \u{2190}\u{2191}\u{2192}\u{00b7}\u{00d7}\u{2014}";

/// [`Face::Regular`]'s glyphs: 7 rows of 5 bits, MSB leftmost, top row first.
///
/// Index-parallel with [`REGULAR_CHARS`]. Written as binary literals because at
/// this size the literal IS the pixel art, and any other encoding would have to
/// be decoded before it could be proofread.
#[rustfmt::skip]
const REGULAR_ROWS: [[u8; CELL_H as usize]; 92] = [
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // (space)
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100], // !
    [0b01010, 0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000], // "
    [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010], // #
    [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011], // %
    [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101], // &
    [0b00100, 0b00100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000], // '
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010], // (
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000], // )
    [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000], // *
    [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000], // +
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110, 0b01100], // ,
    [0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000], // -
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100], // .
    [0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000], // /
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110], // 0
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // 1
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111], // 2
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110], // 3
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010], // 4
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110], // 5
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110], // 6
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000], // 7
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110], // 8
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100], // 9
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000], // :
    [0b00000, 0b01100, 0b01100, 0b00000, 0b00110, 0b00110, 0b01100], // ;
    [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010], // <
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000], // =
    [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000], // >
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100], // ?
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], // A
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110], // B
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110], // C
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100], // D
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111], // E
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000], // F
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111], // G
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], // H
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // I
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100], // J
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001], // K
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111], // L
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001], // M
    [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001], // N
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110], // O
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000], // P
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101], // Q
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001], // R
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110], // S
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100], // T
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110], // U
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100], // V
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001], // W
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001], // X
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100], // Y
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111], // Z
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110], // [
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110], // ]
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111], // _
    [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111], // a
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110], // b
    [0b00000, 0b00000, 0b01111, 0b10000, 0b10000, 0b10000, 0b01111], // c
    [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111], // d
    [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110], // e
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000], // f
    [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110], // g
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001], // h
    [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110], // i
    [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100], // j
    [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010], // k
    [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // l
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101], // m
    [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001], // n
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110], // o
    [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000], // p
    [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b00001], // q
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000], // r
    [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110], // s
    [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110], // t
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101], // u
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100], // v
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010], // w
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001], // x
    [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110], // y
    [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111], // z
    [0b00000, 0b00100, 0b01000, 0b11111, 0b01000, 0b00100, 0b00000], // U+2190 left arrow
    [0b00100, 0b01110, 0b10101, 0b00100, 0b00100, 0b00100, 0b00100], // U+2191 up arrow
    [0b00000, 0b00100, 0b00010, 0b11111, 0b00010, 0b00100, 0b00000], // U+2192 right arrow
    [0b00000, 0b00000, 0b00000, 0b01100, 0b01100, 0b00000, 0b00000], // U+00B7 middle dot
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001], // U+00D7 multiply
    [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000], // U+2014 em dash
];

/// [`Face::Regular`]'s missing-glyph marker: the conventional hollow box.
#[rustfmt::skip]
const REGULAR_TOFU: [u8; CELL_H as usize] =
    [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111];

/// Characters [`Face::Small`] has glyphs for, in [`SMALL_ROWS`] order.
///
/// Ten digits, a space and a hyphen. See [`Face::Small`] on why that is the
/// whole set.
const SMALL_CHARS: &str = " -0123456789";

/// [`Face::Small`]'s glyphs: 5 rows of 3 bits, MSB leftmost.
///
/// Rows 5 and 6 of each entry are unused padding — the array is one type so that
/// [`Face::table`] can return either face's rows, and
/// `no_glyph_sets_a_bit_outside_its_own_cell` asserts the padding stays zero.
#[rustfmt::skip]
const SMALL_ROWS: [[u8; CELL_H as usize]; 12] = [
    [0b000, 0b000, 0b000, 0b000, 0b000, 0, 0], // (space)
    [0b000, 0b000, 0b111, 0b000, 0b000, 0, 0], // -
    [0b111, 0b101, 0b101, 0b101, 0b111, 0, 0], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111, 0, 0], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111, 0, 0], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111, 0, 0], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001, 0, 0], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111, 0, 0], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111, 0, 0], // 6
    [0b111, 0b001, 0b001, 0b010, 0b010, 0, 0], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111, 0, 0], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111, 0, 0], // 9
];

/// [`Face::Small`]'s missing-glyph marker: a solid block. See [`Face::Small`].
#[rustfmt::skip]
const SMALL_TOFU: [u8; CELL_H as usize] = [0b111, 0b111, 0b111, 0b111, 0b111, 0, 0];

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/// A CSS `rgba(r, g, b, a)` as a Bevy colour.
///
/// Every colour in the three ported files is written that way and every one of
/// them is transcribed literally below, so that a constant here can be diffed
/// against its line in the original without arithmetic in between.
pub(crate) const fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color {
    Color::Srgba(Srgba {
        red: r as f32 / 255.0,
        green: g as f32 / 255.0,
        blue: b as f32 / 255.0,
        alpha: a,
    })
}

/// A CSS `#rrggbb`.
pub(crate) const fn rgb(r: u8, g: u8, b: u8) -> Color {
    rgba(r, g, b, 1.0)
}

/// Item name colour by `tier`, clamped to the last entry.
///
/// The TypeScript's `TIER_TINT`, unchanged, including its reasoning: rarity is
/// the one item property worth reading at a glance from across the screen, and
/// colour is the only channel a one-line label row has left. This is CODE and
/// not `content/`, because "how much better is tier 3 than tier 2" is one
/// decision for the whole game rather than a property of any one item.
const TIER_TINT: [Color; 5] = [
    rgb(0xff, 0xff, 0xff), // 0 — starting gear, plain white
    rgb(0x9e, 0xe8, 0xa0), // 1
    rgb(0x87, 0xc8, 0xff), // 2
    rgb(0xd6, 0xa4, 0xff), // 3
    rgb(0xff, 0xc4, 0x6b), // 4+
];

// ---------------------------------------------------------------------------
// The display list
// ---------------------------------------------------------------------------

/// One thing to paint, in buffer pixels with the origin top-left and +y DOWN.
///
/// This exists so that the whole of the port's layout — every number the
/// original's `fillRect` and `fillText` calls carried — is a value that can be
/// built, inspected and asserted on with no window, no GPU and no Bevy `App`.
/// [`paint`] is the only thing downstream of it, and [`paint`] contains no
/// layout at all.
#[derive(Clone, Debug, PartialEq)]
pub enum UiPrim {
    /// A solid rectangle. The port's `fillRect`.
    Rect {
        /// Left edge.
        x: i32,
        /// Top edge.
        y: i32,
        /// Width. Never negative; a zero-width rect is skipped by [`paint`].
        w: i32,
        /// Height.
        h: i32,
        /// Fill colour, straight alpha.
        color: Color,
    },
    /// A run of text. The port's `fillText`, with alignment already resolved.
    Text {
        /// Left edge of the first glyph cell. Already whole-pixel.
        x: i32,
        /// Baseline, which is the BOTTOM edge of the glyph cells.
        baseline: i32,
        /// Face and magnification.
        style: TextStyle,
        /// Fill colour, straight alpha.
        color: Color,
        /// What to draw. Borrowed for the many static labels, owned for the few
        /// formatted ones.
        text: Cow<'static, str>,
    },
    /// A baked item icon, magnified by a whole number.
    ///
    /// Only emitted when [`IconAtlas::icon_cells`] answered, so `w` and `h` are
    /// always a whole multiple of the icon's own cell size.
    Icon {
        /// Left edge.
        x: i32,
        /// Top edge.
        y: i32,
        /// Width, `cells_w * icon_scale`.
        w: i32,
        /// Height, `cells_h * icon_scale`.
        h: i32,
        /// The sprite id from [`ITEM_ICONS`], for [`IconAtlas::icon_sprite`].
        sprite: &'static str,
    },
}

impl UiPrim {
    /// A rectangle, clamped to non-negative extents.
    #[inline]
    pub(crate) fn rect(x: i32, y: i32, w: i32, h: i32, color: Color) -> UiPrim {
        UiPrim::Rect {
            x,
            y,
            w: w.max(0),
            h: h.max(0),
            color,
        }
    }

    /// A run of text, with `align` resolved against `x` here and now.
    ///
    /// Resolving at BUILD time and not at paint time is what makes the
    /// whole-pixel guarantee testable: by the time a `Text` exists its left edge
    /// is an `i32`, and there is no alignment left to get wrong.
    pub(crate) fn text(
        text: impl Into<Cow<'static, str>>,
        x: i32,
        baseline: i32,
        align: Align,
        style: TextStyle,
        color: Color,
    ) -> UiPrim {
        let text = text.into();
        let x = align.left_edge(x, style.measure(&text));
        UiPrim::Text {
            x,
            baseline,
            style,
            color,
            text,
        }
    }
}

/// Where a baked item icon comes from. Implemented for `SpriteAtlases` in
/// [`crate::glue`], which is the only place that names both sides.
///
/// See the module header for the full write-up. Two methods, because the layout
/// needs the icon's size before it can place anything and [`paint`] needs the
/// texture afterwards; `id` is the validated sprite id already stored in
/// [`ITEM_ICONS`].
///
/// The trait stays declared here even though it is implemented elsewhere: it is
/// the shape of the question this module asks, and moving it next to its impl
/// would make [`build_hud`] unbuildable without a sprite baker.
///
/// An implementation that answers `None` to everything is a complete and correct
/// one — see [`NoIcons`].
pub trait IconAtlas: Send + Sync + 'static {
    /// Size of icon `id` in ICON pixels, or `None` if there is no such sprite.
    fn icon_cells(&self, id: &str) -> Option<(i32, i32)>;

    /// The still frame of icon `id`, as a whole `Sprite`.
    ///
    /// [`paint`] overwrites `custom_size` (with the whole-number magnification
    /// the layout worked out) and `color`, and touches nothing else — so an
    /// implementation is free to answer with an atlas index, a sub-rect, a flip,
    /// or a lone image, and none of that is this module's business.
    fn icon_sprite(&self, id: &str) -> Option<Sprite>;
}

/// The empty atlas: no item has art.
///
/// Not a stub for testing — this is what any app draws that has not had an atlas
/// installed by [`crate::glue`]'s `install_icons`, and it is exactly the
/// behaviour the TypeScript hotbar had before any item art existed. Nor is it a
/// mode the wired game leaves behind: `item_swatch` takes the same flat path per
/// item whenever [`ITEM_ICONS`] has no entry, atlas or no atlas. Every swatch
/// takes the flat `color` path, which the original's own comment calls "not a
/// degraded fallback but the state the hotbar is mostly in".
pub struct NoIcons;

impl IconAtlas for NoIcons {
    fn icon_cells(&self, _id: &str) -> Option<(i32, i32)> {
        None
    }

    fn icon_sprite(&self, _id: &str) -> Option<Sprite> {
        None
    }
}

// ---------------------------------------------------------------------------
// Layout tuning
// ---------------------------------------------------------------------------

/// Swatch edge, in buffer px. The TypeScript's `SW`.
///
/// 26 is what makes [`icon_scale`] land on whole numbers for the sizes icons are
/// authored at: a 2x2 icon is 13 px per icon pixel with no margin, a 3x3 is 8
/// with a 1px margin. Changing it changes both.
const SWATCH: i32 = 26;

/// Blank buffer px between one swatch and the next. The TypeScript's `GAP`.
const SWATCH_GAP: i32 = 6;

/// Height of the bottom-left plate, in buffer px.
///
/// 62 = a 26px swatch row, the 2px the selected slot lifts into, and two lines
/// of label under it. Two lines is the budget the panel has; the original's
/// comment on the label row says so.
const PANEL_H: i32 = 62;

/// Distance from the buffer edge to a plate, in buffer px.
///
/// 16 on every edge, in all three ported files. Wide enough that the plate does
/// not fuse with the frame at zoom 2, narrow enough not to eat the view.
const MARGIN: i32 = 16;

/// How far the selected slot lifts out of the strip, in buffer px.
///
/// Two. The original: selection is three signals and this is the one that
/// survives an item whose own art happens to be pale, "because it is a change in
/// geometry, not in contrast". It stays inside the plate, which starts 4px above
/// the strip, so nothing has to grow to accommodate it.
const SELECT_LIFT: i32 = 2;

/// Health bar width in buffer px.
const BAR_W: i32 = 220;

/// Health bar height in buffer px.
const BAR_H: i32 = 20;

/// Radius of the dash-readiness pip, in buffer px.
const PIP_R: i32 = 8;

/// Baseline of the toast line, as px UP from the bottom of the buffer.
///
/// 96 clears the 78px the plate occupies (`PANEL_H` plus its 4px bleed and the
/// 16px margin) with enough air that the two do not read as one object.
const TOAST_UP: i32 = 96;

/// Top of the cursor-feedback note, as px UP from the bottom of the buffer.
///
/// 122 puts it a clear 26px above the toast line, so a craft message and a
/// "too hard" warning can be on screen together and still be two messages.
const NOTE_UP: i32 = 122;

/// How long a toast stays up, in seconds, and the alpha ramp's denominator.
///
/// Three, from `Game.ts`. Alpha is `min(1, t)`, so it holds at full for two
/// seconds and fades over the last one.
pub const TOAST_LIFE_S: f32 = 3.0;

/// The 8px tier: a hotbar slot's key digit, and nothing else.
const KEY: TextStyle = TextStyle::for_px(8);

/// The 10px tier: stat labels, the flavour line, and stack counts.
const SMALL: TextStyle = TextStyle::for_px(10);

/// The 11px tier: stat values, the held-count multiplier, creative's chrome.
const MINOR: TextStyle = TextStyle::for_px(11);

/// The 12px tier: the top-right control hints and the cursor-feedback note.
const HINT: TextStyle = TextStyle::for_px(12);

/// The 13px tier: the held item's name, the HP readout, the toast.
const LABEL: TextStyle = TextStyle::for_px(13);

/// The 14px tier: the menu card's control line.
const CARD_HINT: TextStyle = TextStyle::for_px(14);

/// The 18px tier: a screen card's subtitle.
const CARD_BODY: TextStyle = TextStyle::for_px(18);

/// The 52px tier: the game-over card's headline.
const CARD_DEAD: TextStyle = TextStyle::for_px(52);

/// The 56px tier: the menu card's title.
const CARD_TITLE: TextStyle = TextStyle::for_px(56);

/// The stack count's own tier, split out because [`digits_w`] pins to it.
const COUNT: TextStyle = SMALL;

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

/// Whole-number buffer px per icon pixel for a `cells_w x cells_h` icon in a
/// [`SWATCH`] box, or 0 when even 1:1 will not fit.
///
/// The TypeScript's `iconScale`, and its reasoning is unchanged and still the
/// point of the function:
///
/// Both axes are considered and the smaller wins, because nothing forces an icon
/// to be square — a 2x3 icon scaled off its width alone would be 39px tall in a
/// 26px slot and paint over the slot above it.
///
/// 0 is a real answer and not an error: an icon too large for the slot takes the
/// flat-colour path, which is the same thing that happens for an item with no
/// icon at all. There is no worst case where the hotbar draws nothing.
///
/// A FRACTIONAL scale would drop or double rows of icon pixels depending on
/// sub-pixel position, which reads as the icon shimmering while nothing about it
/// is moving. That is the one rule here a future 4x4 or 5x5 icon could quietly
/// violate, and it is not visible in a screenshot until the icon moves — which
/// is why the original exported this, and why it is `pub` here too.
pub fn icon_scale(cells_w: i32, cells_h: i32) -> i32 {
    if cells_w <= 0 || cells_h <= 0 {
        return 0;
    }
    (SWATCH / cells_w).min(SWATCH / cells_h).max(0)
}

/// Width of a short run of digits at [`COUNT`], without measuring.
///
/// The TypeScript kept this to avoid allocating a `TextMetrics` per occupied
/// slot per frame. That reason is gone — measurement here is a multiplication —
/// but the function is not, because it is the ANCHOR that pins this font's
/// advance to the original's: `digits_w(n) == COUNT.measure(&n.to_string())`,
/// asserted below. If someone retunes [`GLYPH_GAP`] or [`CELL_W`], that test is
/// what tells them the 10px equivalence they inherited is gone.
pub fn digits_w(n: u32) -> i32 {
    let digits = if n >= 100 {
        3
    } else if n >= 10 {
        2
    } else {
        1
    };
    digits * COUNT.advance()
}

/// The [`SWATCH`]-px swatch for one item: its baked icon, or the flat colour.
///
/// The flat branch is the original's two lines, unchanged. Almost every item is
/// in it, so it is not a fallback but the state the hotbar is mostly in, and it
/// must keep covering exactly the same 26x26 px it did before any of this
/// existed.
fn item_swatch(out: &mut Vec<UiPrim>, code: ItemCode, sx: i32, sy: i32, icons: &dyn IconAtlas) {
    if let Some(id) = ITEM_ICONS.get(code as usize).copied().flatten()
        && let Some((cw, ch)) = icons.icon_cells(id)
    {
        let k = icon_scale(cw, ch);
        if k > 0 {
            let (w, h) = (cw * k, ch * k);
            out.push(UiPrim::Icon {
                // `>> 1` in the original, and the same thing here: the leftover
                // is 0 or 2 for the sizes icons are authored at, but an odd
                // leftover must land on a whole pixel anyway.
                x: sx + (SWATCH - w) / 2,
                y: sy + (SWATCH - h) / 2,
                w,
                h,
                sprite: id,
            });
            return;
        }
    }
    let [r, g, b] = item_by_code(code).color;
    out.push(UiPrim::rect(sx, sy, SWATCH, SWATCH, rgb(r, g, b)));
}

/// The same swatch for a BLOCK, which is what the creative strip is built from.
///
/// Blocks have no icons of their own and should not: the icon belongs to the
/// item you hold, and [`item_for_block`] already answers "which item is this
/// block's" — it is the drop relation, inverted once at registry init. A block
/// nothing places (fire, steam, the flow-only fluids) has no item and falls
/// through to its own material colour, which is the right answer anyway.
fn block_swatch(out: &mut Vec<UiPrim>, block: CellId, sx: i32, sy: i32, icons: &dyn IconAtlas) {
    if let Some(item) = item_for_block(block)
        && ITEM_ICONS.get(item as usize).copied().flatten().is_some()
    {
        item_swatch(out, item, sx, sy, icons);
        return;
    }
    let [r, g, b] = mat_by_code(block).color;
    out.push(UiPrim::rect(sx, sy, SWATCH, SWATCH, rgb(r, g, b)));
}

// ---------------------------------------------------------------------------
// Chrome primitives
// ---------------------------------------------------------------------------

/// A `t`-px border on whole pixels, as four fills.
///
/// The original used four `fillRect`s rather than one `strokeRect`, and its
/// reason survives the port intact even though there is no `strokeRect` here to
/// avoid: a stroke needs a half-pixel offset to land on the pixel grid at
/// `lineWidth: 1` and cannot land on it at all at `lineWidth: 2`, and under a
/// pixelated upscale that difference is a magnified grey fringe on two edges of
/// every slot. Four rects have no such problem at any thickness.
fn frame(out: &mut Vec<UiPrim>, x: i32, y: i32, w: i32, h: i32, t: i32, color: Color) {
    out.push(UiPrim::rect(x, y, w, t, color));
    out.push(UiPrim::rect(x, y + h - t, w, t, color));
    out.push(UiPrim::rect(x, y + t, t, h - 2 * t, color));
    out.push(UiPrim::rect(x + w - t, y + t, t, h - 2 * t, color));
}

/// Text with a 1px drop shadow — the only way small type survives an arbitrary
/// swatch behind it.
fn shadowed(
    out: &mut Vec<UiPrim>,
    text: impl Into<Cow<'static, str>>,
    x: i32,
    baseline: i32,
    align: Align,
    style: TextStyle,
    color: Color,
) {
    let text = text.into();
    out.push(UiPrim::text(
        text.clone(),
        x + 1,
        baseline + 1,
        align,
        style,
        rgba(0, 0, 0, 0.85),
    ));
    out.push(UiPrim::text(text, x, baseline, align, style, color));
}

/// A filled circle, as one whole-pixel span per scanline.
///
/// The original called `ctx.arc` and let the canvas antialias the rim. See the
/// module header on why that does not survive a nearest-neighbour upscale. The
/// centre is a pixel CORNER, not a pixel centre, which is why the sample point
/// for row `py` is `py + 0.5` — that is where the pixel actually is.
fn disc(out: &mut Vec<UiPrim>, cx: i32, cy: i32, r: i32, color: Color) {
    for py in (cy - r)..(cy + r) {
        let dy = py as f32 + 0.5 - cy as f32;
        let inside = (r * r) as f32 - dy * dy;
        if inside <= 0.0 {
            continue;
        }
        let half = inside.sqrt();
        let x0 = (cx as f32 - half).ceil() as i32;
        let x1 = (cx as f32 + half).floor() as i32;
        if x1 > x0 {
            out.push(UiPrim::rect(x0, py, x1 - x0, 1, color));
        }
    }
}

/// JavaScript's `Math.round`: half goes toward +infinity, not away from zero.
///
/// `f32::round` disagrees on exactly the negative halves, and the ported
/// arithmetic below reaches `Math.round` on a value that can be negative (the
/// pip row's centring, when a palette group has more members than the swatch is
/// wide). One line, so the two never have to be reasoned about together.
#[inline]
fn js_round(v: f32) -> i32 {
    (v + 0.5).floor() as i32
}

/// JavaScript's number-to-string for the values the stat line prints.
///
/// `${6}` is `"6"` in JS and `"6"` here; `${6.5}` is `"6.5"` in both. Reach is
/// an `f32` in this port and was a `number` in the original, and a stat line
/// reading `reach 6.0` where the original read `reach 6` is a visible diff.
fn js_num(v: f32) -> String {
    if v.is_finite() && v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

// ---------------------------------------------------------------------------
// Hud.ts
// ---------------------------------------------------------------------------

/// The health bar, the dash pip and the control hints.
///
/// `drawHud` in `src/ui/Hud.ts`, drawn after the world in screen space.
pub fn hud(health: f32, dash_ready: bool, view: View) -> Vec<UiPrim> {
    let mut out = Vec::new();
    let (x, y) = (MARGIN, MARGIN);
    let frac = (health / MAX_HEALTH).clamp(0.0, 1.0);

    out.push(UiPrim::rect(
        x - 4,
        y - 4,
        BAR_W + 8,
        BAR_H + 8,
        rgba(0, 0, 0, 0.6),
    ));
    out.push(UiPrim::rect(x, y, BAR_W, BAR_H, rgb(0x3c, 0x3c, 0x3c)));

    // Green -> red as health drops.
    let r = js_round(220.0 - 130.0 * frac) as u8;
    let g = js_round(60.0 + 140.0 * frac) as u8;
    // The one place the original let a fractional rect through and leant on
    // canvas antialiasing. Rounded, because the blit does not smooth.
    out.push(UiPrim::rect(
        x,
        y,
        js_round(BAR_W as f32 * frac),
        BAR_H,
        rgb(r, g, 60),
    ));

    out.push(UiPrim::text(
        format!("HP {}", js_round(health)),
        x + 8,
        LABEL.baseline_from_middle(y + BAR_H / 2 + 1),
        Align::Left,
        LABEL,
        rgb(0xff, 0xff, 0xff),
    ));

    // Dash readiness pip.
    let pip_x = x + BAR_W + 28;
    let pip_y = y + BAR_H / 2;
    disc(
        &mut out,
        pip_x,
        pip_y,
        PIP_R,
        if dash_ready {
            rgb(0x78, 0xc8, 0xff)
        } else {
            rgb(0x46, 0x50, 0x5a)
        },
    );
    out.push(UiPrim::text(
        "DASH",
        pip_x + 14,
        MINOR.baseline_from_middle(pip_y + 1),
        Align::Left,
        MINOR,
        if dash_ready {
            rgba(255, 255, 255, 0.86)
        } else {
            rgba(255, 255, 255, 0.35)
        },
    ));

    let right = view.w - MARGIN;
    out.push(UiPrim::text(
        "\u{2190}/\u{2192} move   \u{2191} jump   Shift dash",
        right,
        HINT.baseline_from_top(18),
        Align::Right,
        HINT,
        rgba(255, 255, 255, 0.6),
    ));
    for (line, top) in [
        ("LMB dig   RMB place   1-0 / wheel hotbar", 36),
        ("F use   C craft   G creative   Alt wall", 50),
    ] {
        out.push(UiPrim::text(
            line,
            right,
            MINOR.baseline_from_top(top),
            Align::Right,
            MINOR,
            rgba(255, 255, 255, 0.42),
        ));
    }
    out
}

/// One transient line above the hotbar: what you just crafted, drank, or failed
/// to.
///
/// `drawToast` in `src/ui/Hud.ts`. It fades on `alpha` rather than sliding,
/// because it sits over a world that is already moving and motion there would
/// read as something happening in it.
pub fn toast(text: &str, alpha: f32, view: View) -> Vec<UiPrim> {
    let mut out = Vec::new();
    let w = LABEL.measure(text);
    let y = view.h - TOAST_UP;
    let cx = view.w / 2;
    out.push(UiPrim::rect(
        cx - w / 2 - 8,
        y - 14,
        w + 16,
        20,
        rgba(0, 0, 0, 0.55 * alpha),
    ));
    out.push(UiPrim::text(
        text.to_owned(),
        cx,
        y,
        Align::Centre,
        LABEL,
        rgba(255, 255, 255, 0.92 * alpha),
    ));
    out
}

// ---------------------------------------------------------------------------
// Screens.ts
// ---------------------------------------------------------------------------

/// The dimming plate both screen cards are drawn on.
fn overlay(out: &mut Vec<UiPrim>, view: View) {
    out.push(UiPrim::rect(0, 0, view.w, view.h, rgba(18, 20, 30, 0.82)));
}

/// The title card. `drawMenu` in `src/ui/Screens.ts`.
pub fn menu(view: View) -> Vec<UiPrim> {
    let mut out = Vec::new();
    overlay(&mut out, view);
    let (cx, cy) = (view.w / 2, view.h / 2);
    out.push(UiPrim::text(
        "GodGame",
        cx,
        CARD_TITLE.baseline_from_middle(cy - 60),
        Align::Centre,
        CARD_TITLE,
        rgb(0xff, 0xff, 0xff),
    ));
    out.push(UiPrim::text(
        "Press Enter or Space to start",
        cx,
        CARD_BODY.baseline_from_middle(cy + 10),
        Align::Centre,
        CARD_BODY,
        rgb(0xc8, 0xc8, 0xc8),
    ));
    out.push(UiPrim::text(
        "\u{2190}/\u{2192} move   \u{2191} jump   Shift dash   L-click dig   R-click place",
        cx,
        CARD_HINT.baseline_from_middle(cy + 50),
        Align::Centre,
        CARD_HINT,
        rgb(0x96, 0x96, 0x96),
    ));
    out
}

/// The death card. `drawGameOver` in `src/ui/Screens.ts`.
pub fn game_over(view: View) -> Vec<UiPrim> {
    let mut out = Vec::new();
    overlay(&mut out, view);
    let (cx, cy) = (view.w / 2, view.h / 2);
    out.push(UiPrim::text(
        "You Died",
        cx,
        CARD_DEAD.baseline_from_middle(cy - 40),
        Align::Centre,
        CARD_DEAD,
        rgb(0xe6, 0x5a, 0x5a),
    ));
    out.push(UiPrim::text(
        "Press Enter or Space to try again",
        cx,
        CARD_BODY.baseline_from_middle(cy + 20),
        Align::Centre,
        CARD_BODY,
        rgb(0xdc, 0xdc, 0xdc),
    ));
    out
}

// ---------------------------------------------------------------------------
// BuildHud.ts
// ---------------------------------------------------------------------------

/// The bottom-left panel: the hotbar, and one line saying what the held item
/// lets you do right now.
///
/// `drawBuildHud` in `src/ui/BuildHud.ts`. Creative mode falls back to the
/// original eight-group material strip, since in that mode the inventory is not
/// what the tool is reading.
pub fn build_hud(
    tool: &BuildTool,
    inv: &Inventory,
    icons: &dyn IconAtlas,
    view: View,
) -> Vec<UiPrim> {
    if tool.creative {
        creative(tool, icons, view)
    } else {
        hotbar(tool, inv, icons, view)
    }
}

/// Top-left corner of the bottom-left plate's CONTENT for a strip of `slots`.
///
/// Returns `(x, y, panel_w)`. Both modes lay their plate out the same way and
/// differ only in how many swatches are on it, so the arithmetic is here once.
fn panel_origin(slots: i32, view: View) -> (i32, i32, i32) {
    let strip_w = slots * SWATCH + (slots - 1) * SWATCH_GAP;
    (
        MARGIN,
        view.h - PANEL_H - MARGIN,
        // The original's `stripW + 8`: 4px of plate either side of the strip.
        strip_w + 8,
    )
}

/// Survival: the inventory hotbar.
///
/// SELECTION IS THREE SIGNALS, NOT ONE. The selected slot is lifted 2px out of
/// the strip, wrapped in a bright 2px frame that sits OUTSIDE its 26px box, and
/// underlined with an accent bar. One signal was a brighter border, which is the
/// one thing that disappears the moment an item's own art happens to be pale —
/// and now that slots hold art rather than solid colour, that is common.
fn hotbar(tool: &BuildTool, inv: &Inventory, icons: &dyn IconAtlas, view: View) -> Vec<UiPrim> {
    let mut out = Vec::new();
    let (x, y, panel_w) = panel_origin(HOTBAR as i32, view);

    out.push(UiPrim::rect(
        x - 4,
        y - 4,
        panel_w + 8,
        PANEL_H + 8,
        rgba(0, 0, 0, 0.6),
    ));
    // A 1px lit top edge and a dark bottom one: the plate reads as a raised
    // object instead of a hole cut in the world, for two fills.
    out.push(UiPrim::rect(
        x - 4,
        y - 4,
        panel_w + 8,
        1,
        rgba(255, 255, 255, 0.10),
    ));
    out.push(UiPrim::rect(
        x - 4,
        y + PANEL_H + 3,
        panel_w + 8,
        1,
        rgba(0, 0, 0, 0.35),
    ));

    for i in 0..HOTBAR {
        let sx = x + i as i32 * (SWATCH + SWATCH_GAP);
        let active = i == inv.selected();
        let sy = if active { y - SELECT_LIFT } else { y };
        let stack = inv.stack_at(i);

        // The well. Icons have transparent pixels; without a constant dark field
        // behind them a wooden icon over a bright sky and the same icon over
        // stone are two different items to the eye.
        out.push(UiPrim::rect(sx, sy, SWATCH, SWATCH, rgba(0, 0, 0, 0.45)));

        match stack {
            None => out.push(UiPrim::rect(
                sx,
                sy,
                SWATCH,
                SWATCH,
                rgba(255, 255, 255, 0.06),
            )),
            Some((code, _)) => item_swatch(&mut out, code, sx, sy, icons),
        }

        if active {
            frame(
                &mut out,
                sx - 2,
                sy - 2,
                SWATCH + 4,
                SWATCH + 4,
                2,
                rgb(0xff, 0xff, 0xff),
            );
            // Accent bar in the gap the lift opened up underneath.
            out.push(UiPrim::rect(
                sx,
                sy + SWATCH + 3,
                SWATCH,
                2,
                rgba(255, 255, 255, 0.55),
            ));
        } else {
            frame(
                &mut out,
                sx,
                sy,
                SWATCH,
                SWATCH,
                1,
                rgba(255, 255, 255, 0.22),
            );
        }

        // Key label in a small corner tab, so the digit never sits on top of
        // art. Slot 10 is Digit0, which is where the key actually is.
        out.push(UiPrim::rect(
            sx + 1,
            sy + 1,
            8,
            8,
            if active {
                rgba(0, 0, 0, 0.75)
            } else {
                rgba(0, 0, 0, 0.6)
            },
        ));
        out.push(UiPrim::text(
            format!("{}", (i + 1) % 10),
            sx + 2,
            sy + 8,
            Align::Left,
            KEY,
            if active {
                rgba(255, 255, 255, 0.95)
            } else {
                rgba(255, 255, 255, 0.45)
            },
        ));

        // Count bottom-right on its own tab. A stack of one is drawn blank: the
        // number only matters when it is going down. A FULL stack is drawn gold,
        // because "this slot will not take another one" is the thing you want to
        // know before you swing at another seam, not after.
        if let Some((code, n)) = stack
            && n > 1
        {
            let full = i64::from(n) >= i64::from(item_by_code(code).stack);
            let w = digits_w(u32::from(n));
            out.push(UiPrim::rect(
                sx + SWATCH - w - 3,
                sy + SWATCH - 10,
                w + 3,
                10,
                rgba(0, 0, 0, 0.7),
            ));
            shadowed(
                &mut out,
                format!("{n}"),
                sx + SWATCH - 2,
                sy + SWATCH - 2,
                Align::Right,
                COUNT,
                if full {
                    rgb(0xff, 0xd7, 0x82)
                } else {
                    rgb(0xff, 0xff, 0xff)
                },
            );
        }
    }

    // --- Label row ----------------------------------------------------------
    // One line: what you are holding, how many, and the three numbers that
    // decide whether the next swing does anything. Two lines total including the
    // flavour below, which is the budget the panel has.
    let text_y = y + SWATCH + 20;
    let held = inv.held();

    let (name, name_color) = match held {
        None => ("Bare hands", rgba(255, 255, 255, 0.45)),
        Some(code) => {
            let def = item_by_code(code);
            let tier = def.tier.max(0) as usize;
            (def.name, TIER_TINT[tier.min(TIER_TINT.len() - 1)])
        }
    };
    out.push(UiPrim::text(
        name,
        x,
        text_y,
        Align::Left,
        LABEL,
        name_color,
    ));
    let mut cx = x + LABEL.measure(name);

    if held.is_some() && inv.held_count() > 1 {
        let mult = format!(" \u{00d7}{}", inv.held_count());
        let w = MINOR.measure(&mult);
        out.push(UiPrim::text(
            mult,
            cx,
            text_y,
            Align::Left,
            MINOR,
            rgba(255, 255, 255, 0.6),
        ));
        cx += w;
    }

    // Labels dim, values bright: the eye is looking for a number that changed
    // when it swapped pickaxes, and a uniform grey run of
    // "dig 2.0 · reach 6 · brush 1" makes it read the words to find them.
    let p = tool.profile;
    cx += 8;
    cx = stat(&mut out, "dig", format!("{:.1}", p.dig_power), cx, text_y);
    cx = stat(&mut out, "reach", js_num(p.reach), cx, text_y);
    stat(&mut out, "brush", format!("{}", p.brush_max), cx, text_y);

    // Flavour on the second line — it is the only place `desc` is surfaced, and
    // it is what tells you a Gem Pickaxe is the end of the ladder.
    if let Some(code) = held {
        let desc = item_by_code(code).desc;
        if !desc.is_empty() {
            out.push(UiPrim::text(
                desc,
                x,
                text_y + 13,
                Align::Left,
                SMALL,
                rgba(255, 255, 255, 0.34),
            ));
        }
    }

    // --- Cursor feedback ----------------------------------------------------
    // Two failure states, and they need different words: out of range is your
    // fault, too hard is your pickaxe's. Different accent colours too — amber
    // for "move", red for "you cannot" — so which one it is registers before the
    // sentence is read.
    if !tool.in_reach {
        note(&mut out, "out of reach", rgba(255, 214, 120, 0.95), view);
    } else if tool.target_too_hard {
        // Naming the block is the difference between "this is broken" and "I
        // need a better pickaxe", which is the entire tool ladder's feedback
        // loop.
        note(
            &mut out,
            format!(
                "{} \u{2014} too hard for this tool",
                mat_by_code(tool.target_block).name
            ),
            rgba(255, 132, 132, 0.95),
            view,
        );
    }

    out
}

/// One `label value` pair on the stat line. Returns the x to continue from.
fn stat(out: &mut Vec<UiPrim>, label: &'static str, value: String, x: i32, baseline: i32) -> i32 {
    out.push(UiPrim::text(
        label,
        x,
        baseline,
        Align::Left,
        SMALL,
        rgba(255, 255, 255, 0.38),
    ));
    let mut cx = x + SMALL.measure(label) + 3;

    let w = MINOR.measure(&value);
    out.push(UiPrim::text(
        value,
        cx,
        baseline,
        Align::Left,
        MINOR,
        rgba(255, 255, 255, 0.85),
    ));
    cx += w + 9;
    cx
}

/// One short warning, centred under the crosshair area at the top of the panel.
///
/// The accent bar down the left carries the severity; the text carries the
/// detail. That ordering matters because this appears and disappears while the
/// mouse is moving and is read peripherally — a colour lands in one frame, a
/// sentence takes several.
fn note(out: &mut Vec<UiPrim>, text: impl Into<Cow<'static, str>>, color: Color, view: View) {
    let text = text.into();
    let w = HINT.measure(&text);
    let x = js_round(view.w as f32 / 2.0 - w as f32 / 2.0) - 8;
    let y = view.h - NOTE_UP;
    let box_w = w + 18;

    out.push(UiPrim::rect(x, y, box_w, 18, rgba(0, 0, 0, 0.72)));
    out.push(UiPrim::rect(x, y, 2, 18, color));
    out.push(UiPrim::rect(x, y, box_w, 1, rgba(255, 255, 255, 0.10)));
    out.push(UiPrim::text(
        text,
        x + 2 + (box_w - 2) / 2,
        y + 13,
        Align::Centre,
        HINT,
        color,
    ));
}

/// Creative: the original eight-group material strip.
fn creative(tool: &BuildTool, icons: &dyn IconAtlas, view: View) -> Vec<UiPrim> {
    let mut out = Vec::new();
    let slots = tool.slots();
    let (x, y, panel_w) = panel_origin(PALETTE_SLOTS as i32, view);

    out.push(UiPrim::rect(
        x - 4,
        y - 4,
        panel_w + 8,
        PANEL_H + 8,
        rgba(40, 10, 10, 0.65),
    ));
    out.push(UiPrim::rect(
        x - 4,
        y - 4,
        panel_w + 8,
        1,
        rgba(255, 180, 180, 0.12),
    ));

    for (i, &block) in slots.iter().enumerate() {
        let sx = x + i as i32 * (SWATCH + SWATCH_GAP);
        let active = i == tool.slot_index();
        let sy = if active { y - SELECT_LIFT } else { y };

        out.push(UiPrim::rect(sx, sy, SWATCH, SWATCH, rgba(0, 0, 0, 0.45)));
        block_swatch(&mut out, block, sx, sy, icons);

        if active {
            frame(
                &mut out,
                sx - 2,
                sy - 2,
                SWATCH + 4,
                SWATCH + 4,
                2,
                rgb(0xff, 0xff, 0xff),
            );
        } else {
            frame(
                &mut out,
                sx,
                sy,
                SWATCH,
                SWATCH,
                1,
                rgba(255, 255, 255, 0.22),
            );
        }

        out.push(UiPrim::rect(
            sx + 1,
            sy + 1,
            8,
            8,
            if active {
                rgba(0, 0, 0, 0.75)
            } else {
                rgba(0, 0, 0, 0.6)
            },
        ));
        out.push(UiPrim::text(
            format!("{}", i + 1),
            sx + 2,
            sy + 8,
            Align::Left,
            KEY,
            if active {
                rgb(0xff, 0xff, 0xff)
            } else {
                rgba(255, 255, 255, 0.5)
            },
        ));
    }

    // Pips: how many materials hide behind the active key.
    let count = tool.member_count() as i32;
    let cur = tool.member_index() as i32;
    let pip_y = y + SWATCH + 5;
    let ax = x + tool.slot_index() as i32 * (SWATCH + SWATCH_GAP);
    let pips_w = count * 4 + (count - 1) * 2;
    let mut px = js_round(ax as f32 + (SWATCH - pips_w) as f32 / 2.0);
    for i in 0..count {
        out.push(UiPrim::rect(
            px,
            pip_y,
            4,
            2,
            if i == cur {
                rgb(0xff, 0xff, 0xff)
            } else {
                rgba(255, 255, 255, 0.3)
            },
        ));
        px += 6;
    }

    let text_y = y + SWATCH + 22;
    let name = tool.selected_name();
    out.push(UiPrim::text(
        name,
        x,
        text_y,
        Align::Left,
        LABEL,
        rgb(0xff, 0xff, 0xff),
    ));
    let name_w = LABEL.measure(name);

    // The original drew this unconditionally and let the empty string be a
    // no-op. An empty `Text` prim would be a quad-less entry in the display
    // list, so it is skipped instead.
    if count > 1 {
        out.push(UiPrim::text(
            format!(
                "  {}/{} \u{00b7} press {} again",
                cur + 1,
                count,
                tool.slot_index() + 1
            ),
            x + name_w,
            text_y,
            Align::Left,
            MINOR,
            rgba(255, 255, 255, 0.55),
        ));
    }

    out.push(UiPrim::text(
        format!("Brush {}", tool.brush),
        x + panel_w - 4,
        text_y,
        Align::Right,
        MINOR,
        rgba(255, 255, 255, 0.55),
    ));

    out.push(UiPrim::text(
        "CREATIVE \u{2014} G to leave",
        x + panel_w / 2,
        y - 10,
        Align::Centre,
        MINOR,
        rgba(255, 150, 150, 0.85),
    ));

    out
}

// ---------------------------------------------------------------------------
// The plugin
// ---------------------------------------------------------------------------

/// Where the overlay sits in z: above everything any other pass draws.
///
/// [`crate::effects`]'s hit flash is the current ceiling at 0.9 and it is a
/// full-view wash; the HUD has to be legible through it, which means over it.
const UI_Z: f32 = 1.0;

/// z added per quad, so the display list's ORDER survives the sort.
///
/// A 2D sprite phase sorts by z and nothing else — two quads at the same z land
/// in whatever order the renderer extracted them, which can change between
/// frames and would make the selected slot's frame flicker under its own well.
/// The original relied on call order; this is that order, made explicit.
///
/// `1e-3` is coarse enough to be exactly representable near 1.0 with room to
/// spare (an `f32` ULP there is about `1.2e-7`) and fine enough that even four
/// thousand quads only reach z 5.0, far inside the camera's range.
const UI_Z_STEP: f32 = 1.0e-3;

/// Which card, if any, is over the world.
///
/// `Game.ts` owned this three-way choice; here it is [`crate::scenes::Scene`],
/// which is a Bevy `States` type with these three variants under these three
/// names. This is deliberately NOT that type:
///
///  * A `States` type has to be registered with `init_state` before anything can
///    read it, which would make [`compose`] untestable without an `App` — and
///    the whole shape of this module is "the layout is a value you can assert
///    on". A plain resource costs nothing and keeps that true.
///  * It defaults to [`UiScreen::Playing`] rather than `Menu`, so that
///    [`UiPlugin`] on its own draws a usable HUD instead of a title card nothing
///    can dismiss. `Scene` defaults to `Menu`, which is right for `Scene` and
///    wrong for a plugin whose owner may not have wired a scene machine at all.
///
/// One system joins them, and it belongs wherever the two plugins are wired
/// together rather than in either of them: `follow_scene` in [`crate::glue`],
/// which maps the three variants onto these three and writes only when they
/// differ, so nothing watching this resource sees a change every frame. It is
/// not restated here — a copy of it in this comment is a copy that can rot.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiScreen {
    /// The title card. `drawMenu`.
    Menu,
    /// The HUD and the build panel.
    #[default]
    Playing,
    /// The death card. `drawGameOver`.
    GameOver,
}

/// The one transient line above the hotbar, and how long it has left.
///
/// `Game.ts` kept these as two fields (`toast`, `toastT`) and decremented one in
/// `update`. They are one resource here because the pair only ever makes sense
/// together: a text with no time left is not shown, and a time with no text is
/// nothing.
#[derive(Resource, Clone, Debug, Default)]
pub struct Toast {
    text: String,
    t: f32,
}

impl Toast {
    /// Put `text` up for [`TOAST_LIFE_S`], replacing whatever was there.
    ///
    /// Replacing and not queueing, exactly as the original: the toast reports
    /// what just happened, and two things that just happened are one thing the
    /// player missed.
    pub fn show(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.t = TOAST_LIFE_S;
    }

    /// Count down by a frame.
    pub fn tick(&mut self, dt: f32) {
        self.t = (self.t - dt).max(0.0);
    }

    /// The line and its alpha, or `None` when nothing is up.
    ///
    /// `min(1, t)` from `Game.ts`: full for the first two seconds, fading over
    /// the last one.
    pub fn showing(&self) -> Option<(&str, f32)> {
        if self.t <= 0.0 || self.text.is_empty() {
            None
        } else {
            Some((&self.text, self.t.min(1.0)))
        }
    }
}

/// The icon source. `None` until [`crate::glue`]'s `install_icons` writes the
/// baked atlas in — see [`IconAtlas`].
#[derive(Resource, Default)]
pub struct Icons(pub Option<Box<dyn IconAtlas>>);

impl Icons {
    /// The installed atlas, or the empty one.
    fn get(&self) -> &dyn IconAtlas {
        match &self.0 {
            Some(atlas) => atlas.as_ref(),
            None => &NoIcons,
        }
    }
}

/// This frame's display list, rebuilt from scratch every frame.
///
/// Rebuilt rather than diffed because it is a few hundred `i32`s into a `Vec`
/// that is reused, and because a diff would need a model of what changed that
/// nothing else in this crate has — the original rebuilt it too, one `fillRect`
/// at a time, and never noticed.
#[derive(Resource, Clone, Debug, Default)]
pub struct UiFrame {
    /// Everything to paint, in paint order.
    pub prims: Vec<UiPrim>,
}

/// The baked font texture and where each glyph is in it.
#[derive(Resource, Clone, Debug)]
pub struct FontAtlas {
    /// One row of [`Face::Regular`] cells over one row of [`Face::Small`] cells.
    image: Handle<Image>,
}

impl FontAtlas {
    /// Source rect for glyph `index` of `face`, in texture pixels.
    ///
    /// Both faces stride by [`CELL_W`] even though `Small` is 3 wide, so that a
    /// glyph index maps to an x with no per-face table. The two unused columns
    /// are never sampled.
    fn rect(face: Face, index: usize) -> Rect {
        let x = index as f32 * CELL_W as f32;
        let top = match face {
            Face::Regular => 0.0,
            Face::Small => CELL_H as f32,
        };
        Rect::new(x, top, x + face.cell_w() as f32, top + face.cell_h() as f32)
    }
}

/// The pooled quads every prim is painted with.
///
/// Grows to the frame's high-water mark and never shrinks, on the same terms
/// [`crate::mobs`] pools its creature rectangles: a hotbar slot emptying should
/// not touch the ECS's archetypes. It is a `Vec` rather than a fixed budget
/// because an item name is content and there is no honest constant for how many
/// glyphs one can be.
#[derive(Resource, Default)]
pub struct UiQuads {
    pool: Vec<Entity>,
}

/// Marks a pooled overlay quad.
#[derive(Component)]
pub struct UiQuad;

/// One rectangle to put on screen. The output of expanding a [`UiPrim`].
#[derive(Clone, Debug)]
struct Quad {
    /// Left edge in buffer px.
    x: i32,
    /// Top edge in buffer px, +y DOWN.
    y: i32,
    /// Width in buffer px.
    w: i32,
    /// Height in buffer px.
    h: i32,
    /// Tint. Multiplies the sampled texel, so a flat fill wants the default
    /// white image and this carries the whole colour.
    color: Color,
    /// The art to sample, or `None` for a flat fill.
    ///
    /// `custom_size` and `color` on it are ignored — [`paint`] writes both from
    /// the fields above, which are the ones the layout decided.
    art: Option<Sprite>,
}

/// The drawable half of a pooled quad, named because the query type appears
/// twice and spelling it out inline is what pushes [`Painter`] past readable.
type QuadParts = (
    &'static mut Sprite,
    &'static mut Transform,
    &'static mut Visibility,
);

/// Everything [`compose`] reads.
///
/// One [`SystemParam`] rather than six parameters, for the reason
/// [`crate::sky::Frame`] is one: past seven arguments a system stops being
/// readable, and clippy stops being quiet.
#[derive(SystemParam)]
struct HudSources<'w> {
    /// The build tool, for the palette and the cursor feedback.
    tool: Res<'w, Tool>,
    /// The pack, for the hotbar.
    pack: Res<'w, Pack>,
    /// The body, for health and dash readiness. Absent until a world exists.
    body: Option<Res<'w, PlayerBody>>,
    /// The transient line above the hotbar.
    toast: Res<'w, Toast>,
    /// Which card is over the world.
    screen: Res<'w, UiScreen>,
    /// Where item art comes from.
    icons: Res<'w, Icons>,
    /// Whether the F3 panel is up.
    debug_shown: Res<'w, crate::debug::DebugOverlay>,
    /// What it would say. Gathered in `PreUpdate`, so this is THIS frame's.
    debug: Res<'w, crate::debug::DebugReadout>,
}

/// Everything [`paint`] writes through. Bundled for [`HudSources`]'s reason.
#[derive(SystemParam)]
struct Painter<'w, 's> {
    /// For growing the pool past its high-water mark.
    commands: Commands<'w, 's>,
    /// The pool itself.
    quads: ResMut<'w, UiQuads>,
    /// The pooled entities' drawable state.
    sprites: Query<'w, 's, QuadParts, With<UiQuad>>,
    /// The buffer the overlay is laid out in.
    target: Res<'w, LowResTarget>,
    /// The baked font.
    font: Res<'w, FontAtlas>,
    /// Where item art comes from.
    icons: Res<'w, Icons>,
}

/// The overlay: the bitmap font, the HUD, the build panel and the screen cards.
///
/// Deliberately NOT in [`crate::GodGameRenderPlugin`] — the group's order is the
/// binary's business, and this has to go last, after the light composite.
pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiScreen>()
            .init_resource::<Toast>()
            .init_resource::<Icons>()
            .init_resource::<UiFrame>()
            .init_resource::<UiQuads>()
            .add_systems(Startup, bake_font)
            .add_systems(
                Update,
                (
                    tick_toast,
                    compose.run_if(resource_exists::<Tool>.and_then(resource_exists::<Pack>)),
                    paint.run_if(resource_exists::<FontAtlas>),
                )
                    .chain(),
            );
    }
}

/// Rasterise both faces into one texture, once.
///
/// White with a hard 0-or-255 alpha; every colour in the overlay arrives as the
/// quad's tint. The sampler is pinned to nearest HERE rather than inherited from
/// `ImagePlugin::default_nearest()` in the binary, because a linear sampler on
/// THIS texture is the exact failure the whole module is built to prevent and it
/// should not depend on a setting made somewhere else.
fn bake_font(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let cells = Face::Regular.glyph_count().max(Face::Small.glyph_count());
    let w = cells as i32 * CELL_W;
    let h = Face::Regular.cell_h() + Face::Small.cell_h();
    let mut data = vec![0u8; (w * h) as usize * 4];

    for (face, row_top) in [(Face::Regular, 0), (Face::Small, CELL_H)] {
        for (index, ch) in face.chars().enumerate() {
            for row in 0..face.cell_h() {
                for col in 0..face.cell_w() {
                    if !face.lit(ch, col, row) {
                        continue;
                    }
                    let px = index as i32 * CELL_W + col;
                    let py = row_top + row;
                    let o = ((py * w + px) * 4) as usize;
                    data[o..o + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.sampler = ImageSampler::nearest();

    commands.insert_resource(FontAtlas {
        image: images.add(image),
    });
}

/// Count the toast down.
fn tick_toast(time: Res<Time>, mut toast: ResMut<Toast>) {
    toast.tick(time.delta_secs());
}

/// Build this frame's display list.
///
/// The whole of the port's layout runs here, and none of it touches an entity.
fn compose(sources: HudSources, target: Res<LowResTarget>, mut frame: ResMut<UiFrame>) {
    let view = target.view;
    let prims = &mut frame.prims;
    prims.clear();

    match *sources.screen {
        UiScreen::Menu => prims.extend(menu(view)),
        UiScreen::GameOver => prims.extend(game_over(view)),
        UiScreen::Playing => {
            if let Some(body) = &sources.body {
                prims.extend(hud(body.0.health, body.0.dash_ready(), view));
            }
            prims.extend(build_hud(
                &sources.tool.0,
                &sources.pack,
                sources.icons.get(),
                view,
            ));
            if let Some((text, alpha)) = sources.toast.showing() {
                prims.extend(toast(text, alpha, view));
            }
        }
    }

    // Last, and outside the `match`, because the panel is an instrument rather
    // than part of any screen: it is as useful over the death card as over the
    // world, and the one thing it must never do is be hidden by the state you
    // were trying to diagnose.
    if sources.debug_shown.0 {
        prims.extend(crate::debug::overlay(&sources.debug, view));
    }
}

/// Put the display list on screen.
///
/// Contains no layout: it expands text into glyph quads, asks the seam for icon
/// textures, and places whole-pixel rectangles. Everything it needs to decide
/// was decided in [`compose`].
fn paint(frame: Res<UiFrame>, mut p: Painter, camera: Single<Entity, With<WorldCamera>>) {
    let view = p.target.view;
    let camera = *camera;
    let font = p.font.image.clone();
    let mut n = 0usize;

    // One prim's worth of quads at a time, into a buffer that is reused across
    // prims: the alternative is a closure capturing `p` mutably while `expand`
    // borrows the icon seam out of it immutably, which does not borrow-check and
    // does not read any better if it did.
    let mut expanded: Vec<Quad> = Vec::new();
    for prim in &frame.prims {
        expanded.clear();
        expand(prim, &font, p.icons.get(), &mut expanded);
        for quad in &expanded {
            if quad.w <= 0 || quad.h <= 0 {
                continue;
            }
            let z = UI_Z + n as f32 * UI_Z_STEP;
            let centre = quad_centre(quad.x, quad.y, quad.w, quad.h, view);
            let size = Vec2::new(quad.w as f32, quad.h as f32);
            match p.quads.pool.get(n).copied() {
                Some(entity) => {
                    if let Ok((mut sprite, mut transform, mut visibility)) =
                        p.sprites.get_mut(entity)
                    {
                        *sprite = dressed(quad, size);
                        transform.translation = centre.extend(z);
                        *visibility = Visibility::Inherited;
                    }
                }
                None => {
                    let entity = p
                        .commands
                        .spawn((
                            dressed(quad, size),
                            Transform::from_translation(centre.extend(z)),
                            UiQuad,
                            ChildOf(camera),
                            WORLD_LAYERS,
                        ))
                        .id();
                    p.quads.pool.push(entity);
                }
            }
            n += 1;
        }
    }

    let idle: Vec<Entity> = p.quads.pool.iter().skip(n).copied().collect();
    for entity in idle {
        if let Ok((_, _, mut visibility)) = p.sprites.get_mut(entity) {
            *visibility = Visibility::Hidden;
        }
    }
}

/// A quad as the `Sprite` that draws it, at the size the layout worked out.
///
/// The art's own `custom_size` and `color` are overwritten and not merged: an
/// icon's still frame arrives sized for the WORLD (`cells * CELL_SIZE` px, which
/// is what puts a dropped item at one sprite pixel per world cell), and a hotbar
/// well is 26 px wide whatever the world thinks.
fn dressed(quad: &Quad, size: Vec2) -> Sprite {
    let mut sprite = quad.art.clone().unwrap_or_default();
    sprite.color = quad.color;
    sprite.custom_size = Some(size);
    sprite
}

/// One [`UiPrim`] as the quads that draw it.
fn expand(prim: &UiPrim, font: &Handle<Image>, icons: &dyn IconAtlas, out: &mut Vec<Quad>) {
    match prim {
        UiPrim::Rect { x, y, w, h, color } => out.push(Quad {
            x: *x,
            y: *y,
            w: *w,
            h: *h,
            color: *color,
            art: None,
        }),
        UiPrim::Text {
            x,
            baseline,
            style,
            color,
            text,
        } => {
            let top = baseline - style.cap_h();
            for (i, ch) in text.chars().enumerate() {
                // A space has no lit pixels; skipping it here rather than
                // emitting a transparent quad is worth roughly a fifth of the
                // overlay's quads on the control-hint lines.
                if ch == ' ' {
                    continue;
                }
                let gx = x + i as i32 * style.advance();
                match style.face.index_of(ch) {
                    Some(index) => out.push(Quad {
                        x: gx,
                        y: top,
                        w: style.face.cell_w() * style.scale,
                        h: style.cap_h(),
                        color: *color,
                        art: Some(Sprite {
                            image: font.clone(),
                            rect: Some(FontAtlas::rect(style.face, index)),
                            ..default()
                        }),
                    }),
                    None => tofu(gx, top, *style, *color, out),
                }
            }
        }
        UiPrim::Icon { x, y, w, h, sprite } => {
            if let Some(art) = icons.icon_sprite(sprite) {
                out.push(Quad {
                    x: *x,
                    y: *y,
                    w: *w,
                    h: *h,
                    color: Color::WHITE,
                    art: Some(art),
                });
            }
        }
    }
}

/// The missing-glyph marker, as four quads rather than an atlas cell.
///
/// It is not in the baked texture because it is not in either face's table, and
/// adding it there would mean a cell index that is not a character — a special
/// case in [`bake_font`], in [`FontAtlas::rect`], and in every test that walks
/// the table. Four rectangles cost nothing and are only ever drawn for content
/// this font cannot set.
fn tofu(x: i32, y: i32, style: TextStyle, color: Color, out: &mut Vec<Quad>) {
    let w = style.face.cell_w() * style.scale;
    let h = style.cap_h();
    let t = style.scale;
    for (qx, qy, qw, qh) in [
        (x, y, w, t),
        (x, y + h - t, w, t),
        (x, y + t, t, h - 2 * t),
        (x + w - t, y + t, t, h - 2 * t),
    ] {
        out.push(Quad {
            x: qx,
            y: qy,
            w: qw,
            h: qh,
            color,
            art: None,
        });
    }
}

/// **The one +y flip.** A top-left-origin buffer rect as a camera-child centre.
///
/// # Why this lands on texel boundaries for either buffer parity
///
/// The world camera sits at `c` over an `n`-pixel buffer, so it covers
/// `[c - n/2, c + n/2]`, and [`crate::lowres`]'s `snap` guarantees `c - n/2` is
/// a whole number `K`. Buffer column `i` is therefore world `[K + i, K + i + 1)`.
///
/// A child at local `lx` with width `w` spans `[c + lx - w/2, c + lx + w/2]`.
/// Put `lx = x + w/2 - n/2` and its left edge is `c + x - n/2 = K + x` — a texel
/// boundary whenever `x` is whole, which every [`UiPrim`] field is. The `w/2`
/// and `n/2` may each be fractional; they cancel, which is why this is one
/// expression and not two rounded halves.
///
/// The y axis is the same with the sign flipped: the rect's TOP edge in +y-down
/// buffer space becomes its top edge in +y-up world space at `c + n/2 - y`.
#[inline]
fn quad_centre(x: i32, y: i32, w: i32, h: i32, view: View) -> Vec2 {
    Vec2::new(
        x as f32 + w as f32 / 2.0 - view.w as f32 / 2.0,
        view.h as f32 / 2.0 - y as f32 - h as f32 / 2.0,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A 640x360 buffer — both axes even, which is the common case.
    fn view() -> View {
        View {
            zoom: 2.0,
            w: 640,
            h: 360,
        }
    }

    /// The same with both axes odd, which is the parity that exercises
    /// [`crate::lowres`]'s half-pixel camera snap.
    fn odd_view() -> View {
        View {
            zoom: 2.0,
            w: 641,
            h: 361,
        }
    }

    /// An atlas where every item's art is a `cells x cells` square.
    struct SquareIcons(i32);

    impl IconAtlas for SquareIcons {
        fn icon_cells(&self, _id: &str) -> Option<(i32, i32)> {
            Some((self.0, self.0))
        }

        fn icon_sprite(&self, _id: &str) -> Option<Sprite> {
            // Sized for the world, exactly as `SpriteAtlas::still()` is — the
            // point being that `dressed` must overwrite it.
            Some(Sprite {
                custom_size: Some(Vec2::splat(999.0)),
                ..default()
            })
        }
    }

    fn texts(prims: &[UiPrim]) -> Vec<(&str, i32, i32, TextStyle)> {
        prims
            .iter()
            .filter_map(|p| match p {
                UiPrim::Text {
                    x,
                    baseline,
                    style,
                    text,
                    ..
                } => Some((text.as_ref(), *x, *baseline, *style)),
                _ => None,
            })
            .collect()
    }

    /// Text set in `style`.
    ///
    /// Only a discriminator for [`KEY`], which is the one tier on a different
    /// FACE. Every tier from 10px to 14px is `Regular @ 1` and therefore one
    /// `TextStyle` — that is the size ladder's central compromise, and a test
    /// that tried to tell those tiers apart would be asserting the compromise
    /// away. Use [`texts_on`] for those.
    fn texts_at(prims: &[UiPrim], style: TextStyle) -> Vec<String> {
        texts(prims)
            .into_iter()
            .filter(|(_, _, _, s)| *s == style)
            .map(|(t, ..)| t.to_owned())
            .collect()
    }

    /// Text on one baseline — the only reliable way to name a row.
    fn texts_on(prims: &[UiPrim], baseline: i32) -> Vec<String> {
        texts(prims)
            .into_iter()
            .filter(|(_, _, b, _)| *b == baseline)
            .map(|(t, ..)| t.to_owned())
            .collect()
    }

    /// A tool in survival mode, whatever `START_CREATIVE` currently is.
    ///
    /// `BuildTool::new()` follows that config flag, so a test that assumed
    /// either mode would start failing the day it flipped.
    fn survival_tool() -> BuildTool {
        let mut tool = BuildTool::new();
        if tool.creative {
            tool.toggle_creative();
        }
        tool
    }

    /// A tool in creative mode, on the same terms.
    fn creative_tool() -> BuildTool {
        let mut tool = BuildTool::new();
        if !tool.creative {
            tool.toggle_creative();
        }
        tool
    }

    /// Baseline of the cursor-feedback note, which is how it is picked out.
    fn note_baseline(view: View) -> i32 {
        view.h - NOTE_UP + 13
    }

    // --- The font -----------------------------------------------------------

    #[test]
    fn a_face_has_one_row_set_per_character() {
        for face in [Face::Small, Face::Regular] {
            let (chars, rows) = face.table();
            assert_eq!(
                chars.chars().count(),
                rows.len(),
                "{face:?}'s character list and row table have drifted apart"
            );
        }
    }

    #[test]
    fn a_face_lists_no_character_twice() {
        for face in [Face::Small, Face::Regular] {
            let mut seen: Vec<char> = face.chars().collect();
            let before = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(before, seen.len(), "{face:?} lists a character twice");
        }
    }

    #[test]
    fn no_glyph_sets_a_bit_outside_its_own_cell() {
        for face in [Face::Small, Face::Regular] {
            for ch in face.chars() {
                for (r, bits) in face.rows(ch).iter().enumerate() {
                    if r as i32 >= face.cell_h() {
                        assert_eq!(*bits, 0, "{face:?} {ch:?} row {r} is padding but is not 0");
                    } else {
                        assert!(
                            *bits < (1 << face.cell_w()),
                            "{face:?} {ch:?} row {r} sets a bit past its {} columns",
                            face.cell_w()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_character_the_ported_ui_ever_draws_has_a_glyph() {
        // Every literal in the three ported files, plus the shapes the formatted
        // strings take. Item names and descriptions come from `content/` and are
        // covered by `content_supplies_no_character_the_font_lacks` below.
        const LITERALS: &[&str] = &[
            "HP 100",
            "DASH",
            "\u{2190}/\u{2192} move   \u{2191} jump   Shift dash",
            "LMB dig   RMB place   1-0 / wheel hotbar",
            "F use   C craft   G creative   Alt wall",
            "GodGame",
            "Press Enter or Space to start",
            "\u{2190}/\u{2192} move   \u{2191} jump   Shift dash   L-click dig   R-click place",
            "You Died",
            "Press Enter or Space to try again",
            "Bare hands",
            " \u{00d7}12",
            "dig",
            "reach",
            "brush",
            "2.0",
            "out of reach",
            "Granite \u{2014} too hard for this tool",
            "CREATIVE \u{2014} G to leave",
            "Brush 3",
            "  2/4 \u{00b7} press 5 again",
        ];
        for s in LITERALS {
            for ch in s.chars() {
                assert!(
                    Face::Regular.index_of(ch).is_some(),
                    "Regular has no glyph for {ch:?} (in {s:?})"
                );
            }
        }
        // The 8px tier draws exactly one thing, and Small must cover all of it.
        for i in 0..HOTBAR {
            for ch in format!("{}", (i + 1) % 10).chars() {
                assert!(
                    Face::Small.index_of(ch).is_some(),
                    "Small has no glyph for the slot-{i} key {ch:?}"
                );
            }
        }
    }

    #[test]
    fn content_supplies_no_character_the_font_lacks() {
        use godgame_core::items::registry::ITEM_DEFS;
        use godgame_core::sim::materials::BLOCKS;

        let mut missing: Vec<char> = Vec::new();
        for def in ITEM_DEFS.iter() {
            for ch in def.name.chars().chain(def.desc.chars()) {
                if Face::Regular.index_of(ch).is_none() {
                    missing.push(ch);
                }
            }
        }
        for block in BLOCKS.iter() {
            for ch in block.name.chars() {
                if Face::Regular.index_of(ch).is_none() {
                    missing.push(ch);
                }
            }
        }
        missing.sort_unstable();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "authored content uses characters this font cannot set: {missing:?}"
        );
    }

    #[test]
    fn a_character_with_no_glyph_still_takes_its_full_advance() {
        // The whole point of the missing-glyph marker: text that silently
        // shortens itself hides the problem until someone looks at a screenshot.
        assert_eq!(LABEL.measure("\u{2603}\u{2603}"), 2 * LABEL.advance());
        assert_ne!(Face::Regular.rows('\u{2603}'), Face::Regular.rows(' '));
    }

    #[test]
    fn small_marks_a_missing_glyph_as_something_that_is_not_a_digit() {
        // A hollow 3x5 box IS the digit zero, so `Small` uses a solid block.
        for ch in Face::Small.chars() {
            assert_ne!(
                Face::Small.rows(ch),
                SMALL_TOFU,
                "{ch:?} is indistinguishable from Small's missing-glyph marker"
            );
        }
    }

    #[test]
    fn the_regular_face_is_the_typescripts_ten_pixel_face() {
        // The one measurable equivalence between the authored font and the
        // browser one it replaces. See the module header.
        assert_eq!(TextStyle::for_px(10), COUNT);
        assert_eq!(COUNT.advance(), 6);
        for n in [0u32, 1, 9, 10, 42, 99, 100, 999] {
            assert_eq!(
                digits_w(n),
                COUNT.measure(&n.to_string()),
                "digits_w disagrees with the font for {n}"
            );
        }
    }

    #[test]
    fn the_size_ladder_maps_every_face_the_typescript_asked_for() {
        let expect = [
            (8, Face::Small, 1),
            (10, Face::Regular, 1),
            (11, Face::Regular, 1),
            (12, Face::Regular, 1),
            (13, Face::Regular, 1),
            (14, Face::Regular, 1),
            (18, Face::Regular, 2),
            (52, Face::Regular, 5),
            (56, Face::Regular, 6),
        ];
        for (px, face, scale) in expect {
            let style = TextStyle::for_px(px);
            assert_eq!((style.face, style.scale), (face, scale), "{px}px");
        }
    }

    #[test]
    fn no_size_on_the_ladder_produces_a_scale_below_one() {
        for px in 0..200u32 {
            assert!(TextStyle::for_px(px).scale >= 1, "{px}px scaled to nothing");
        }
    }

    #[test]
    fn measuring_the_empty_string_is_zero_under_every_alignment() {
        for style in [KEY, SMALL, MINOR, HINT, LABEL, CARD_TITLE] {
            assert_eq!(style.measure(""), 0);
            assert!(style.line_h() > style.cap_h(), "{style:?} has leading");
        }
        // An empty run resolves to its anchor whichever way it is aligned, so a
        // centred empty string does not shift by half a glyph.
        for align in [Align::Left, Align::Centre, Align::Right] {
            assert_eq!(align.left_edge(100, 0), 100);
        }
    }

    #[test]
    fn measurement_counts_characters_and_not_bytes() {
        // `×`, `·`, `—` and the arrows are all multi-byte, and all appear in
        // ported strings. Measuring bytes would push the label row along by two
        // cells per multiplication sign.
        assert_eq!(MINOR.measure(" \u{00d7}12"), 4 * MINOR.advance());
        assert_eq!(HINT.measure("\u{2190}\u{2191}\u{2192}"), 3 * HINT.advance());
        assert!(" \u{00d7}12".len() > 4, "the test string is not multi-byte");
    }

    #[test]
    fn a_glyph_always_lands_on_a_whole_pixel() {
        // Every advance is even, at every scale, for both faces — which is what
        // makes centring exact rather than rounded.
        for face in [Face::Small, Face::Regular] {
            for scale in 1..=8 {
                let style = TextStyle { face, scale };
                assert_eq!(style.advance() % 2, 0, "{face:?} @ {scale}");
                assert_eq!(style.cap_h(), face.cell_h() * scale);
            }
        }
    }

    #[test]
    fn a_centred_run_never_lands_a_glyph_on_a_half_pixel() {
        for style in [KEY, SMALL, MINOR, HINT, LABEL, CARD_BODY, CARD_TITLE] {
            for n in 0..40 {
                let text = "M".repeat(n);
                let w = style.measure(&text);
                assert_eq!(w % 2, 0, "{n} glyphs measured odd");
                // The left edge is exact: `w / 2` has no remainder to round.
                assert_eq!(Align::Centre.left_edge(321, w) * 2, 321 * 2 - w);
            }
        }
    }

    #[test]
    fn vertical_alignment_resolves_to_a_baseline_inside_the_line_it_names() {
        for style in [KEY, SMALL, MINOR, HINT, LABEL, CARD_DEAD, CARD_TITLE] {
            let middle = style.baseline_from_middle(100);
            assert!(middle > 100 && middle - style.cap_h() < 100, "{style:?}");
            assert_eq!(style.baseline_from_top(100) - style.cap_h(), 100);
        }
    }

    // --- Icons --------------------------------------------------------------

    #[test]
    fn an_icon_is_scaled_by_a_whole_number_or_not_at_all() {
        // The property the TypeScript exported `iconScale` in order to assert.
        for w in 1..=SWATCH {
            for h in 1..=SWATCH {
                let k = icon_scale(w, h);
                assert!(w * k <= SWATCH && h * k <= SWATCH, "{w}x{h} overflows");
                assert!(k > 0, "{w}x{h} fits at 1:1 and should not have given up");
                assert!(
                    w * (k + 1) > SWATCH || h * (k + 1) > SWATCH,
                    "{w}x{h} could have gone one step larger"
                );
            }
        }
        // The two sizes icons are actually authored at, from the original's own
        // header: "a 2x2 icon lands at exactly 13 px per icon pixel in a 26px
        // slot and a 3x3 at 8 with a 1px margin".
        assert_eq!(icon_scale(2, 2), 13);
        assert_eq!(icon_scale(3, 3), 8);
    }

    #[test]
    fn an_icon_too_large_for_the_slot_falls_back_rather_than_overflowing() {
        assert_eq!(icon_scale(SWATCH + 1, 1), 0);
        assert_eq!(icon_scale(1, SWATCH + 1), 0);
        // A non-square icon is scaled off its LARGER axis, so it cannot paint
        // over the slot above it.
        assert_eq!(icon_scale(2, 3), 8);
    }

    #[test]
    fn a_degenerate_icon_size_is_a_fallback_and_not_a_division_by_zero() {
        assert_eq!(icon_scale(0, 4), 0);
        assert_eq!(icon_scale(4, 0), 0);
        assert_eq!(icon_scale(-1, -1), 0);
    }

    #[test]
    fn an_icon_is_centred_in_its_well_on_whole_pixels() {
        let mut out = Vec::new();
        item_swatch(&mut out, 0, 100, 200, &SquareIcons(3));
        let icon = out
            .iter()
            .find_map(|p| match p {
                UiPrim::Icon { x, y, w, h, .. } => Some((*x, *y, *w, *h)),
                _ => None,
            })
            .expect("a 3x3 icon fits a 26px well");
        // 3 * 8 = 24 in a 26px box: 1px of margin each side.
        assert_eq!(icon, (101, 201, 24, 24));
    }

    #[test]
    fn an_item_with_no_art_draws_the_flat_swatch_the_typescript_shipped() {
        let mut out = Vec::new();
        item_swatch(&mut out, 0, 10, 20, &NoIcons);
        assert_eq!(out.len(), 1);
        match out[0] {
            UiPrim::Rect { x, y, w, h, .. } => assert_eq!((x, y, w, h), (10, 20, SWATCH, SWATCH)),
            ref other => panic!("expected a flat swatch, got {other:?}"),
        }
    }

    // --- Layout: the hotbar -------------------------------------------------

    #[test]
    fn an_empty_hotbar_still_draws_ten_wells_and_a_bare_hands_label() {
        let tool = survival_tool();
        let inv = Inventory::new();
        let prims = hotbar(&tool, &inv, &NoIcons, view());
        assert!(
            texts(&prims).iter().any(|(t, ..)| *t == "Bare hands"),
            "an empty pack holds bare hands"
        );
        assert_eq!(texts_at(&prims, KEY).len(), HOTBAR, "one key tab per slot");
    }

    #[test]
    fn a_full_hotbar_draws_a_count_tab_for_every_stack_but_never_for_a_single() {
        let tool = survival_tool();
        let mut inv = Inventory::new();
        // Slot 0 gets one, the rest get several: the original draws a count only
        // when it is going down.
        inv.add(0, 1);
        for code in 1..HOTBAR as ItemCode {
            inv.add(code, 5);
        }
        let prims = hotbar(&tool, &inv, &NoIcons, view());
        let y = panel_origin(HOTBAR as i32, view()).1;
        // Count tabs sit on the swatch row's own baseline, and the shadow one px
        // under it. The unlifted slots are the ones with stacks of five.
        let counts = texts_on(&prims, y + SWATCH - 2);
        assert!(!counts.iter().any(|t| t == "1"), "a stack of one is blank");
        assert_eq!(counts.iter().filter(|t| *t == "5").count(), 9);
        assert_eq!(
            texts_on(&prims, y + SWATCH - 1)
                .iter()
                .filter(|t| *t == "5")
                .count(),
            9,
            "every count is shadowed"
        );
    }

    #[test]
    fn the_selected_slot_lifts_out_of_the_strip_and_the_others_do_not() {
        let tool = survival_tool();
        let mut inv = Inventory::new();
        inv.select_slot(3);
        let prims = hotbar(&tool, &inv, &NoIcons, view());
        let (_, y, _) = panel_origin(HOTBAR as i32, view());

        // The well is the first full-size rect at each slot's x.
        let well_top = |slot: i32| {
            let sx = MARGIN + slot * (SWATCH + SWATCH_GAP);
            prims
                .iter()
                .find_map(|p| match p {
                    UiPrim::Rect {
                        x,
                        y,
                        w: SWATCH,
                        h: SWATCH,
                        ..
                    } if *x == sx => Some(*y),
                    _ => None,
                })
                .expect("every slot has a well")
        };
        assert_eq!(well_top(3), y - SELECT_LIFT);
        assert_eq!(well_top(2), y);
        assert_eq!(well_top(4), y);
    }

    #[test]
    fn the_hotbar_strip_fits_inside_its_own_plate_and_the_plate_inside_the_buffer() {
        let (x, y, panel_w) = panel_origin(HOTBAR as i32, view());
        // The plate is drawn at `x - 4` and `panel_w + 8` wide, and `panel_w` is
        // already `strip_w + 8`, so the original's margins are ASYMMETRIC: 4px
        // of plate to the left of the strip and 12px to the right. Ported as-is.
        let (plate_left, plate_right) = (x - 4, x - 4 + panel_w + 8);
        let strip_right = x + (HOTBAR as i32 - 1) * (SWATCH + SWATCH_GAP) + SWATCH;
        assert_eq!(x - plate_left, 4, "4px of plate to the left");
        assert_eq!(plate_right - strip_right, 12, "12px of plate to the right");
        // The lift and its 2px frame stay inside the plate's 4px top bleed.
        const {
            assert!(
                SELECT_LIFT + 2 <= 4,
                "the selection frame escapes the plate"
            )
        };
        assert!(y - 4 > 0 && y + PANEL_H + 4 <= view().h);
        assert!(plate_right <= view().w, "the plate runs off the buffer");
    }

    #[test]
    fn the_label_row_advances_by_exactly_what_it_measured() {
        let tool = survival_tool();
        let mut inv = Inventory::new();
        inv.add(0, 4);
        let prims = hotbar(&tool, &inv, &NoIcons, view());
        let text_y = panel_origin(HOTBAR as i32, view()).1 + SWATCH + 20;

        // Everything on the label row, left to right. Each must start at or
        // after the previous one's right edge — the chain of `cx +=` in the
        // original, which is the one thing a wrong `measure` would break.
        let mut row: Vec<(i32, i32)> = texts(&prims)
            .into_iter()
            .filter(|(_, _, b, _)| *b == text_y)
            .map(|(t, x, _, s)| (x, x + s.measure(t)))
            .collect();
        row.sort_unstable();
        assert!(row.len() >= 7, "name, three labels and three values");
        for pair in row.windows(2) {
            assert!(
                pair[1].0 >= pair[0].1,
                "the label row overlaps itself: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn the_longest_item_name_in_the_content_still_fits_on_the_panel() {
        use godgame_core::items::registry::ITEM_DEFS;
        let (_, _, panel_w) = panel_origin(HOTBAR as i32, view());
        let widest = ITEM_DEFS
            .iter()
            .map(|d| (d.name, LABEL.measure(d.name)))
            .max_by_key(|(_, w)| *w)
            .expect("the registry is not empty");
        // The name shares its line with the stat run, so the budget it may not
        // exceed is the plate itself, not the whole buffer.
        assert!(
            widest.1 <= panel_w,
            "{:?} is {}px, wider than the {panel_w}px plate",
            widest.0,
            widest.1
        );
    }

    #[test]
    fn the_longest_item_description_overruns_the_plate_but_not_the_buffer() {
        use godgame_core::items::registry::ITEM_DEFS;
        let (x, _, panel_w) = panel_origin(HOTBAR as i32, view());
        let widest = ITEM_DEFS
            .iter()
            .map(|d| (d.desc, SMALL.measure(d.desc)))
            .max_by_key(|(_, w)| *w)
            .expect("the registry is not empty");

        // A KNOWN AND ACCEPTED REGRESSION, pinned here so it cannot get worse
        // unnoticed. The original set this line in a 10px proportional sans,
        // where an average character advances about 4.5px; a fixed 6px advance
        // is a third wider, so the longest flavour line now runs about 100px
        // further right than it did. The original did not clip it either — it is
        // a bare `fillText` over the world with no plate behind it, and both
        // versions overrun. What must not happen is running off the BUFFER,
        // where the text would be cut mid-word.
        assert!(
            widest.1 > panel_w,
            "descriptions now fit the plate — this test is stale, tighten it"
        );
        assert!(
            x + widest.1 <= view().w,
            "{:?} is {}px and runs off a {}px buffer",
            widest.0,
            widest.1,
            view().w
        );
    }

    #[test]
    fn out_of_reach_and_too_hard_are_different_notes_and_never_both() {
        let mut tool = survival_tool();
        let inv = Inventory::new();
        let line = note_baseline(view());

        tool.in_reach = false;
        tool.target_too_hard = true;
        let both = texts_on(&hotbar(&tool, &inv, &NoIcons, view()), line);
        assert_eq!(both, vec!["out of reach".to_owned()], "reach wins the tie");

        tool.in_reach = true;
        let hard = texts_on(&hotbar(&tool, &inv, &NoIcons, view()), line);
        assert_eq!(hard.len(), 1);
        assert!(hard[0].ends_with("too hard for this tool"), "{hard:?}");
        assert!(
            hard[0].starts_with(mat_by_code(tool.target_block).name),
            "the note must name the block: {hard:?}"
        );

        tool.target_too_hard = false;
        let quiet = texts_on(&hotbar(&tool, &inv, &NoIcons, view()), line);
        assert!(quiet.is_empty(), "no note when nothing is wrong");
    }

    #[test]
    fn the_note_box_is_centred_on_the_buffer_for_either_parity() {
        for v in [view(), odd_view()] {
            let mut out = Vec::new();
            note(&mut out, "out of reach", Color::WHITE, v);
            let (bx, bw) = match out[0] {
                UiPrim::Rect { x, w, .. } => (x, w),
                ref other => panic!("{other:?}"),
            };
            let (tx, tw) = match out.last() {
                Some(UiPrim::Text { x, style, text, .. }) => (*x, style.measure(text)),
                other => panic!("{other:?}"),
            };
            // The text lands a couple of px RIGHT of the buffer's midpoint in
            // the original too, and deliberately: the box is placed 8px left of
            // the centred text to make room for the accent bar, and the text is
            // then re-centred inside a box 18px wider than itself. How many px
            // depends on the buffer's parity, because the original rounds the
            // half-width. Ported as it was; pinned as a range, not a number.
            let drift = tx + tw / 2 - v.w / 2;
            assert!((2..=3).contains(&drift), "note text drifted {drift}px");
            // And the text sits inside its own box, with the accent bar clear.
            assert!(tx >= bx + 2 && tx + tw <= bx + bw, "text escapes its box");
        }
    }

    // --- Layout: creative ---------------------------------------------------

    #[test]
    fn the_creative_strip_draws_one_swatch_per_palette_slot() {
        let tool = creative_tool();
        let prims = creative(&tool, &NoIcons, view());
        assert_eq!(texts_at(&prims, KEY).len(), PALETTE_SLOTS);
        assert!(
            texts(&prims)
                .iter()
                .any(|(t, ..)| *t == "CREATIVE \u{2014} G to leave")
        );
    }

    #[test]
    fn the_member_pips_stay_centred_under_the_active_slot() {
        let tool = creative_tool();
        let prims = creative(&tool, &NoIcons, view());
        let (x, y, _) = panel_origin(PALETTE_SLOTS as i32, view());
        let pip_y = y + SWATCH + 5;
        let pips: Vec<i32> = prims
            .iter()
            .filter_map(|p| match p {
                UiPrim::Rect {
                    x, y, w: 4, h: 2, ..
                } if *y == pip_y => Some(*x),
                _ => None,
            })
            .collect();
        assert_eq!(pips.len(), tool.member_count(), "one pip per member");
        let (&first, &last) = (pips.first().unwrap(), pips.last().unwrap());
        let ax = x + tool.slot_index() as i32 * (SWATCH + SWATCH_GAP);
        let run_centre = first + (last + 4 - first) / 2;
        assert!(
            (run_centre - (ax + SWATCH / 2)).abs() <= 1,
            "pips centred at {run_centre}, slot centre {}",
            ax + SWATCH / 2
        );
    }

    #[test]
    fn build_hud_picks_the_strip_that_matches_the_tools_mode() {
        let inv = Inventory::new();
        let survival = build_hud(&survival_tool(), &inv, &NoIcons, view());
        assert!(texts(&survival).iter().any(|(t, ..)| *t == "Bare hands"));

        let creative_prims = build_hud(&creative_tool(), &inv, &NoIcons, view());
        assert!(
            texts(&creative_prims)
                .iter()
                .any(|(t, ..)| t.starts_with("CREATIVE")),
            "creative mode draws the palette, not the pack"
        );
        assert!(
            !texts(&creative_prims)
                .iter()
                .any(|(t, ..)| *t == "Bare hands")
        );
    }

    // --- Layout: the health bar and the cards -------------------------------

    #[test]
    fn the_health_bar_fills_whole_pixels_at_every_health() {
        for hp in 0..=100 {
            // The fill is the third rect: plate, track, fill.
            match hud(hp as f32, true, view())[2] {
                UiPrim::Rect { w, .. } => assert!((0..=BAR_W).contains(&w), "{hp} hp filled {w}px"),
                ref other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn health_outside_its_range_clamps_rather_than_overflowing_the_track() {
        for hp in [-50.0f32, 0.0, MAX_HEALTH, MAX_HEALTH * 2.0] {
            match hud(hp, false, view())[2] {
                UiPrim::Rect { w, .. } => assert!((0..=BAR_W).contains(&w), "{hp} hp"),
                ref other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn the_dash_pip_is_a_symmetric_disc_of_whole_pixel_spans() {
        let mut out = Vec::new();
        disc(&mut out, 100, 50, PIP_R, Color::WHITE);
        assert_eq!(out.len(), (PIP_R * 2) as usize, "one span per scanline");
        let mut widths = Vec::new();
        for prim in &out {
            match prim {
                UiPrim::Rect { y, w, h, .. } => {
                    assert_eq!(*h, 1);
                    assert!(*w > 0 && *w <= PIP_R * 2, "span {w}px wide");
                    assert!((50 - PIP_R..50 + PIP_R).contains(y));
                    widths.push(*w);
                }
                other => panic!("{other:?}"),
            }
        }
        // Symmetric about its centre, which a rounding slip would break.
        let mirrored: Vec<i32> = widths.iter().rev().copied().collect();
        assert_eq!(widths, mirrored);
    }

    #[test]
    fn the_control_hints_are_right_aligned_against_the_buffer_edge() {
        for v in [view(), odd_view()] {
            for (text, x, _, style) in texts(&hud(50.0, true, v)) {
                if text.starts_with("HP") || text == "DASH" {
                    continue;
                }
                assert_eq!(
                    x + style.measure(text),
                    v.w - MARGIN,
                    "{text:?} does not end on the margin"
                );
            }
        }
    }

    #[test]
    fn both_screen_cards_dim_the_whole_buffer_before_anything_else() {
        for v in [view(), odd_view()] {
            for prims in [menu(v), game_over(v)] {
                match prims[0] {
                    UiPrim::Rect { x, y, w, h, .. } => assert_eq!((x, y, w, h), (0, 0, v.w, v.h)),
                    ref other => panic!("the card must dim first, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn a_card_headline_is_centred_and_fits_the_narrowest_buffer_in_use() {
        // Deliberately smaller than anything `View::for_screen` produces, as the
        // worst case.
        let v = View {
            zoom: 2.0,
            w: 480,
            h: 270,
        };
        for prims in [menu(v), game_over(v)] {
            for (text, x, _, style) in texts(&prims) {
                assert_eq!(x, v.w / 2 - style.measure(text) / 2, "{text:?} off centre");
            }
        }
        // The title itself, at scale 6, is the thing most likely to run off.
        assert!(
            CARD_TITLE.measure("GodGame") < v.w,
            "the title does not fit a 480px buffer"
        );
    }

    #[test]
    fn a_toast_fades_on_alpha_and_never_on_geometry() {
        let a = toast("Crafted a Gem Pickaxe", 1.0, view());
        let b = toast("Crafted a Gem Pickaxe", 0.25, view());
        let geometry = |prims: &[UiPrim]| -> Vec<(i32, i32)> {
            prims
                .iter()
                .map(|p| match p {
                    UiPrim::Rect { x, y, .. } => (*x, *y),
                    UiPrim::Text { x, baseline, .. } => (*x, *baseline),
                    UiPrim::Icon { x, y, .. } => (*x, *y),
                })
                .collect()
        };
        assert_eq!(geometry(&a), geometry(&b), "the toast moved as it faded");
        assert_ne!(a, b, "the toast did not fade at all");
    }

    #[test]
    fn a_toast_holds_full_for_two_seconds_and_then_fades_to_nothing() {
        let mut t = Toast::default();
        assert!(t.showing().is_none(), "nothing is up to start with");
        t.show("Crafted a Gem Pickaxe");
        assert_eq!(t.showing().map(|(_, a)| a), Some(1.0));
        t.tick(2.0);
        assert_eq!(
            t.showing().map(|(_, a)| a),
            Some(1.0),
            "still full with 1s left"
        );
        t.tick(0.5);
        assert_eq!(t.showing().map(|(_, a)| a), Some(0.5));
        t.tick(10.0);
        assert!(t.showing().is_none(), "the toast outlived its life");
    }

    // --- The bridge to Bevy -------------------------------------------------

    #[test]
    fn a_quad_lands_on_texel_boundaries_for_either_buffer_parity() {
        for v in [view(), odd_view()] {
            for (x, y, w, h) in [(0, 0, 1, 1), (16, 16, 220, 20), (313, 7, 26, 26)] {
                let c = quad_centre(x, y, w, h, v);
                // `lowres::snap` makes `camera - n/2` whole, so the rect's edges
                // are whole exactly when these are.
                let left = c.x + v.w as f32 / 2.0 - w as f32 / 2.0;
                let top = v.h as f32 / 2.0 - (c.y + h as f32 / 2.0);
                assert_eq!(left, x as f32, "left edge of {x},{y} {w}x{h} in {v:?}");
                assert_eq!(top, y as f32, "top edge of {x},{y} {w}x{h} in {v:?}");
            }
        }
    }

    #[test]
    fn the_y_flip_puts_the_top_of_the_buffer_above_its_bottom() {
        let v = view();
        let top = quad_centre(0, 0, 10, 10, v);
        let bottom = quad_centre(0, v.h - 10, 10, 10, v);
        assert!(top.y > bottom.y, "+y is not up in Bevy space");
        // And the buffer's own extent maps to the camera-relative origin.
        assert_eq!(quad_centre(0, 0, v.w, v.h, v), Vec2::ZERO);
    }

    #[test]
    fn every_glyph_in_a_run_gets_its_own_atlas_cell_at_whole_pixels() {
        let mut out = Vec::new();
        let prim = UiPrim::text("Hi 5", 40, 30, Align::Left, LABEL, Color::WHITE);
        expand(&prim, &Handle::default(), &NoIcons, &mut out);
        // The space contributes no quad.
        assert_eq!(out.len(), 3);
        for (i, quad) in out.iter().enumerate() {
            assert_eq!(quad.h, LABEL.cap_h());
            assert_eq!(quad.y, 30 - LABEL.cap_h());
            let art = quad.art.as_ref().expect("glyph {i} has no art");
            assert!(art.rect.is_some(), "glyph {i} has no atlas cell");
        }
        // "H", "i" then "5": the third glyph skips the space's advance.
        assert_eq!(out[0].x, 40);
        assert_eq!(out[1].x, 40 + LABEL.advance());
        assert_eq!(out[2].x, 40 + 3 * LABEL.advance());
    }

    #[test]
    fn a_missing_glyph_expands_to_a_visible_box_rather_than_nothing() {
        let mut out = Vec::new();
        let prim = UiPrim::text("\u{2603}", 0, 7, Align::Left, LABEL, Color::WHITE);
        expand(&prim, &Handle::default(), &NoIcons, &mut out);
        assert_eq!(out.len(), 4, "the marker is a four-rect frame");
        assert!(
            out.iter().all(|q| q.art.is_none()),
            "the marker needs no art"
        );
    }

    #[test]
    fn an_icon_prim_expands_to_nothing_when_the_seam_has_no_texture() {
        let mut out = Vec::new();
        let prim = UiPrim::Icon {
            x: 0,
            y: 0,
            w: 24,
            h: 24,
            sprite: "block_cube",
        };
        expand(&prim, &Handle::default(), &NoIcons, &mut out);
        assert!(out.is_empty(), "no art, no quad — the well shows through");

        expand(&prim, &Handle::default(), &SquareIcons(3), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].w, out[0].h), (24, 24));
    }

    #[test]
    fn an_icons_own_world_size_never_leaks_into_the_hotbar_well() {
        // `SpriteAtlas::still()` arrives sized for the WORLD — `cells *
        // CELL_SIZE` — which is what puts a dropped item at one sprite pixel per
        // world cell. A 26px well is not that, and `dressed` must say so.
        let mut out = Vec::new();
        let prim = UiPrim::Icon {
            x: 0,
            y: 0,
            w: 24,
            h: 24,
            sprite: "block_cube",
        };
        expand(&prim, &Handle::default(), &SquareIcons(3), &mut out);
        let sprite = dressed(&out[0], Vec2::splat(24.0));
        assert_eq!(sprite.custom_size, Some(Vec2::splat(24.0)));
        assert_eq!(sprite.color, Color::WHITE);
    }

    #[test]
    fn a_flat_rect_paints_with_the_default_white_texture_and_carries_its_own_colour() {
        let quad = Quad {
            x: 0,
            y: 0,
            w: 10,
            h: 4,
            color: rgba(0, 0, 0, 0.6),
            art: None,
        };
        let sprite = dressed(&quad, Vec2::new(10.0, 4.0));
        assert_eq!(sprite.image, Handle::default());
        assert_eq!(sprite.rect, None);
        assert_eq!(sprite.color, rgba(0, 0, 0, 0.6));
        assert_eq!(sprite.custom_size, Some(Vec2::new(10.0, 4.0)));
    }

    #[test]
    fn the_font_atlas_gives_every_glyph_a_distinct_cell_inside_the_texture() {
        let cells = Face::Regular.glyph_count().max(Face::Small.glyph_count());
        let w = cells as f32 * CELL_W as f32;
        let h = (Face::Regular.cell_h() + Face::Small.cell_h()) as f32;
        let mut seen: Vec<(u32, u32)> = Vec::new();
        for face in [Face::Regular, Face::Small] {
            for index in 0..face.glyph_count() {
                let r = FontAtlas::rect(face, index);
                assert!(r.max.x <= w && r.max.y <= h, "{face:?} {index} is outside");
                assert_eq!(r.width(), face.cell_w() as f32);
                assert_eq!(r.height(), face.cell_h() as f32);
                seen.push((r.min.x as u32, r.min.y as u32));
            }
        }
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(before, seen.len(), "two glyphs share an atlas cell");
    }

    #[test]
    fn the_z_step_keeps_a_whole_frames_worth_of_quads_distinct_and_in_order() {
        // A worst case far past what the busiest panel emits.
        let mut last = f32::NEG_INFINITY;
        for n in 0..4000 {
            let z = UI_Z + n as f32 * UI_Z_STEP;
            assert!(z > last, "quad {n} did not advance past {last}");
            last = z;
        }
        assert!(
            last < 100.0,
            "the overlay left the camera's range at {last}"
        );
    }

    // --- The judgement calls ------------------------------------------------

    #[test]
    fn js_round_rounds_halves_the_way_javascript_does_and_not_the_way_rust_does() {
        // `f32::round` sends -2.5 to -3; `Math.round` sends it to -2. The pip
        // row's centring can reach a negative half.
        assert_eq!(js_round(2.5), 3);
        assert_eq!(js_round(-2.5), -2);
        assert_eq!(js_round(-2.6), -3);
        assert_eq!(js_round(0.5), 1);
        assert_eq!(js_round(-0.5), 0);
    }

    #[test]
    fn a_whole_number_stat_prints_without_a_decimal_point() {
        // `${6}` is "6" in JavaScript, and the stat line was written against it.
        assert_eq!(js_num(6.0), "6");
        assert_eq!(js_num(4.0), "4");
        assert_eq!(js_num(6.5), "6.5");
        assert_eq!(js_num(-3.0), "-3");
    }

    #[test]
    fn every_layout_number_survives_a_buffer_small_enough_to_be_pathological() {
        // Not a size `View::for_screen` produces, but the layout must not panic
        // or emit a negative extent on one.
        let v = View {
            zoom: 1.0,
            w: 120,
            h: 90,
        };
        let inv = Inventory::new();
        let mut all = build_hud(&survival_tool(), &inv, &NoIcons, v);
        all.extend(build_hud(&creative_tool(), &inv, &NoIcons, v));
        all.extend(hud(50.0, true, v));
        all.extend(toast("x", 1.0, v));
        all.extend(menu(v));
        all.extend(game_over(v));
        for prim in &all {
            if let UiPrim::Rect { w, h, .. } = prim {
                assert!(*w >= 0 && *h >= 0, "negative extent in {prim:?}");
            }
        }
    }
}
