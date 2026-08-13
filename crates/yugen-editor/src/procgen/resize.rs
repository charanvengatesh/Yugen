//! Changing the grid a drawing is on.
//!
//! # The one operation that is allowed to change shape
//!
//! Everything in [`crate::procgen::ops`] preserves the frame's shape, because a
//! frame is only meaningful against the `cellsW`/`cellsH` its record declares.
//! A resize is the deliberate exception: it moves both at once, which is why it
//! takes the target as an argument and why [`resize_sprite`] returns the new
//! declaration alongside the new frames.
//!
//! # Feet, and centred — and that is not a taste
//!
//! [`Anchor::FeetCentre`] is the default because it is the **identity** for both
//! hosts that draw these records.
//!
//! - `yugen-render`'s `ArtGrid::from_def` computes `pad_top_px` as ALL of the
//!   vertical surplus, so a sprite is drawn with its bottom row on the ground.
//! - `MobDef::build` does the same and additionally centres the horizontal
//!   overhang, asserting that the difference is even.
//!
//! So bottom-aligned and horizontally centred is where the art already sits
//! relative to the body. Any other anchor moves the drawing against the
//! collision box, which is a gameplay change wearing a migration's clothes.
//!
//! # Pad, not scale
//!
//! Every mob in `content/` today has `art.cellsW == bodyCellsW` and
//! `art.cellsH == bodyCellsH` — `yugen-render`'s mob module says so outright:
//! *"every mob in current content has zero art padding."* Padding such a record
//! to a bigger grid leaves the creature **pixel-identical** and only grows the
//! canvas around it. Scaling it 2x would draw a creature at twice the size of
//! the thing the player can actually hit.
//!
//! [`Mode::ScaleNearest`] exists because 4→8 is a clean doubling and somebody
//! will want it, but [`Mode::PadCrop`] is the default and the one a migration
//! should use.
//!
//! # The report is the point
//!
//! A resize that crops can destroy art, and it destroys it in a file whose diff
//! is a wall of digits. [`Report`] names every frame that lost ink and how much,
//! so a migration can be reviewed as "nothing was lost" rather than trusted.

use crate::procgen::ops::{self, CLEAR};
use crate::sprite::{Frame, Seq, Sprite};

/// Where the old drawing sits inside the new grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Anchor {
    /// Bottom edge and horizontal centre — where both hosts already put the art
    /// relative to the body box. The identity, and the default.
    #[default]
    FeetCentre,
    /// Centred on both axes. For an icon, which stands on nothing.
    Centre,
    /// Top-left. Rarely what is wanted; here because it is the one anchor that
    /// needs no arithmetic and so is the one to reach for when debugging.
    TopLeft,
}

impl Anchor {
    pub const ALL: [Anchor; 3] = [Anchor::FeetCentre, Anchor::Centre, Anchor::TopLeft];

    pub fn name(self) -> &'static str {
        match self {
            Anchor::FeetCentre => "feet, centred",
            Anchor::Centre => "centred",
            Anchor::TopLeft => "top left",
        }
    }

    /// Where the old grid's origin lands in the new one. May be negative, which
    /// is a crop.
    fn offset(self, from: (usize, usize), to: (usize, usize)) -> (i32, i32) {
        let (fw, fh) = (from.0 as i32, from.1 as i32);
        let (tw, th) = (to.0 as i32, to.1 as i32);
        match self {
            // The horizontal is halved and the vertical is not: the bottom edges
            // meet, so the whole vertical difference goes above the art.
            Anchor::FeetCentre => ((tw - fw) / 2, th - fh),
            Anchor::Centre => ((tw - fw) / 2, (th - fh) / 2),
            Anchor::TopLeft => (0, 0),
        }
    }
}

/// How the drawing is fitted to the new grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Keep every texel its own size; add empty space or cut it away.
    #[default]
    PadCrop,
    /// Stretch the drawing to fill the new grid, nearest-neighbour.
    ///
    /// Only honest at an integer ratio — 4→8 is a clean doubling, 6→8 is not and
    /// will drop or repeat rows unevenly. Pixel art resampled at a fractional
    /// ratio stops being pixel art.
    ScaleNearest,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::PadCrop, Mode::ScaleNearest];

    pub fn name(self) -> &'static str {
        match self {
            Mode::PadCrop => "pad / crop",
            Mode::ScaleNearest => "scale",
        }
    }
}

/// What a resize cost, per frame.
///
/// A migration you cannot review is a migration you have to trust. `lost` is
/// what makes "this changed nothing visible" a checkable claim rather than an
/// assurance.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Report {
    pub frames: usize,
    /// `(sequence, frame, texels)` for every frame that lost ink. Empty means
    /// the resize was lossless.
    pub lost: Vec<(usize, usize, usize)>,
}

impl Report {
    pub fn lossless(&self) -> bool {
        self.lost.is_empty()
    }

    /// How many texels went missing in total.
    pub fn total_lost(&self) -> usize {
        self.lost.iter().map(|&(_, _, n)| n).sum()
    }
}

/// Resize one frame. Texels, not cells — the caller multiplies by `grain`.
pub fn resize(f: &Frame, to: (usize, usize), anchor: Anchor, mode: Mode) -> Frame {
    let (tw, th) = (to.0.max(1), to.1.max(1));
    let (fw, fh) = (ops::width(f), f.len());
    if fw == 0 || fh == 0 {
        return vec![CLEAR.to_string().repeat(tw); th];
    }

    let src: Vec<Vec<char>> = f
        .iter()
        .map(|row| {
            let mut cs: Vec<char> = row.chars().collect();
            cs.resize(fw, CLEAR);
            cs
        })
        .collect();

    let mut out = vec![vec![CLEAR; tw]; th];
    match mode {
        Mode::PadCrop => {
            let (ox, oy) = anchor.offset((fw, fh), (tw, th));
            for (y, row) in src.iter().enumerate() {
                for (x, &ch) in row.iter().enumerate() {
                    let (tx, ty) = (x as i32 + ox, y as i32 + oy);
                    if tx >= 0 && ty >= 0 && (tx as usize) < tw && (ty as usize) < th {
                        out[ty as usize][tx as usize] = ch;
                    }
                }
            }
        }
        Mode::ScaleNearest => {
            for (ty, out_row) in out.iter_mut().enumerate() {
                for (tx, ch) in out_row.iter_mut().enumerate() {
                    // Sample from the middle of the target texel, so a 2x
                    // doubling picks each source texel exactly twice instead of
                    // shifting the whole image half a texel.
                    let sx = ((tx * 2 + 1) * fw) / (tw * 2);
                    let sy = ((ty * 2 + 1) * fh) / (th * 2);
                    *ch = src[sy.min(fh - 1)][sx.min(fw - 1)];
                }
            }
        }
    }
    out.into_iter().map(|r| r.into_iter().collect()).collect()
}

/// How many ink texels a resize of this frame would drop.
///
/// Counted by resizing back and comparing, rather than by arithmetic on the
/// offsets: it is the same question the caller actually cares about — "is
/// anything gone" — and it cannot disagree with [`resize`] because it uses it.
fn lost_ink(f: &Frame, to: (usize, usize), anchor: Anchor, mode: Mode) -> usize {
    let before = f
        .iter()
        .flat_map(|r| r.chars())
        .filter(|&c| !ops::is_clear(c))
        .count();
    let after = resize(f, to, anchor, mode)
        .iter()
        .flat_map(|r| r.chars())
        .filter(|&c| !ops::is_clear(c))
        .count();
    before.saturating_sub(after)
}

/// Resize a whole record: every frame of every sequence, and the declaration.
///
/// `cells` is in CELLS, matching what the record writes. The frames are resized
/// in texels, so a `grain > 1` record scales by its grain on both axes — which
/// is the point of the field, and getting it wrong here would silently rewrite
/// what a cell means for that sprite.
pub fn resize_sprite(
    s: &Sprite,
    cells: (u32, u32),
    anchor: Anchor,
    mode: Mode,
) -> (Sprite, Report) {
    let to = ((cells.0 * s.grain) as usize, (cells.1 * s.grain) as usize);
    let mut report = Report::default();
    let mut out = s.clone();
    out.cells_w = cells.0;
    out.cells_h = cells.1;

    out.seqs = s
        .seqs
        .iter()
        .enumerate()
        .map(|(si, seq)| Seq {
            state: seq.state.clone(),
            mode: seq.mode.clone(),
            fps: seq.fps,
            frames: seq
                .frames
                .iter()
                .enumerate()
                .map(|(fi, frame)| {
                    report.frames += 1;
                    let lost = lost_ink(frame, to, anchor, mode);
                    if lost > 0 {
                        report.lost.push((si, fi, lost));
                    }
                    resize(frame, to, anchor, mode)
                })
                .collect(),
        })
        .collect();

    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: &[&str]) -> Frame {
        rows.iter().map(|r| (*r).to_string()).collect()
    }

    #[test]
    fn a_resize_to_the_same_size_is_the_identity_for_every_anchor() {
        // The property the migration leans on: running the tool over a record
        // that is already the target size must produce the same bytes, so a
        // second pass is provably a no-op.
        let f = frame(&["12..", "3..4", "..56", "7..8"]);
        for anchor in Anchor::ALL {
            for mode in Mode::ALL {
                assert_eq!(
                    resize(&f, (4, 4), anchor, mode),
                    f,
                    "{} / {} moved something",
                    anchor.name(),
                    mode.name()
                );
            }
        }
    }

    #[test]
    fn padding_puts_the_feet_on_the_bottom_and_centres_the_width() {
        // The identity anchor: this is exactly where `ArtGrid::from_def` and
        // `MobDef::build` already place the art relative to the body.
        let f = frame(&["12", "34"]);
        assert_eq!(
            resize(&f, (4, 4), Anchor::FeetCentre, Mode::PadCrop),
            frame(&["....", "....", ".12.", ".34."])
        );
    }

    #[test]
    fn centring_splits_the_surplus_and_top_left_spends_none() {
        let f = frame(&["12", "34"]);
        assert_eq!(
            resize(&f, (4, 4), Anchor::Centre, Mode::PadCrop),
            frame(&["....", ".12.", ".34.", "...."])
        );
        assert_eq!(
            resize(&f, (4, 4), Anchor::TopLeft, Mode::PadCrop),
            frame(&["12..", "34..", "....", "...."])
        );
    }

    #[test]
    fn padding_a_real_mob_grid_loses_nothing() {
        // 4x4 and 6x6 to 8x8 are the two shapes the migration actually performs.
        // Both differences are even, which is what `MobDef::build` asserts about
        // the horizontal overhang.
        for from in [frame(&["12", "34"]), frame(&["123", "456", "789"])] {
            let n = from
                .iter()
                .flat_map(|r| r.chars())
                .filter(|&c| !ops::is_clear(c))
                .count();
            let out = resize(&from, (8, 8), Anchor::FeetCentre, Mode::PadCrop);
            let m = out
                .iter()
                .flat_map(|r| r.chars())
                .filter(|&c| !ops::is_clear(c))
                .count();
            assert_eq!(n, m, "padding dropped ink");
            assert_eq!(out.len(), 8);
            assert!(out.iter().all(|r| r.chars().count() == 8));
        }
    }

    #[test]
    fn a_crop_that_clips_ink_is_reported_and_a_pad_is_not() {
        // The whole reason `Report` exists. A migration is reviewable only if
        // "nothing was lost" is checkable.
        let f = frame(&["11", "11"]);
        assert_eq!(lost_ink(&f, (4, 4), Anchor::FeetCentre, Mode::PadCrop), 0);
        // Cropping to one row keeps the bottom row — feet stay planted — and
        // drops the two texels in the row above.
        assert_eq!(lost_ink(&f, (2, 1), Anchor::FeetCentre, Mode::PadCrop), 2);
    }

    #[test]
    fn a_crop_takes_from_the_top_because_the_bottom_is_the_ground() {
        let f = frame(&["12", "34"]);
        assert_eq!(
            resize(&f, (2, 1), Anchor::FeetCentre, Mode::PadCrop),
            frame(&["34"]),
            "the row standing on the ground is the one that survives"
        );
    }

    #[test]
    fn doubling_repeats_every_texel_exactly_twice() {
        // The one ratio `ScaleNearest` is honest at, and the reason the sampler
        // takes the middle of the target texel rather than its corner.
        let f = frame(&["12", "34"]);
        assert_eq!(
            resize(&f, (4, 4), Anchor::TopLeft, Mode::ScaleNearest),
            frame(&["1122", "1122", "3344", "3344"])
        );
    }

    #[test]
    fn a_resized_record_declares_the_grid_it_was_given() {
        let s = Sprite {
            id: "x".into(),
            name: "X".into(),
            prefix: "",
            cells_w: 2,
            cells_h: 2,
            grain: 1,
            fps: 8.0,
            pal: vec![".".into(), "#ffffff".into()],
            seqs: vec![Seq {
                state: "idle".into(),
                mode: "hold".into(),
                fps: 0.0,
                frames: vec![frame(&["11", "11"])],
            }],
        };
        let (out, report) = resize_sprite(&s, (8, 8), Anchor::FeetCentre, Mode::PadCrop);
        assert_eq!((out.cells_w, out.cells_h), (8, 8));
        assert_eq!(out.seqs[0].frames[0].len(), 8);
        assert_eq!(report.frames, 1);
        assert!(report.lossless(), "a pad lost {}", report.total_lost());
    }

    #[test]
    fn grain_multiplies_the_texel_grid_and_not_the_cell_count() {
        // `cellsW` is in cells and a frame row is `cellsW * grain` characters.
        // A resize that forgot the grain would rewrite what a cell means for
        // that sprite while appearing to only change its size.
        let s = Sprite {
            id: "x".into(),
            name: "X".into(),
            prefix: "",
            cells_w: 2,
            cells_h: 2,
            grain: 2,
            fps: 8.0,
            pal: vec![".".into(), "#ffffff".into()],
            seqs: vec![Seq {
                state: "idle".into(),
                mode: "hold".into(),
                fps: 0.0,
                frames: vec![frame(&["1111", "1111", "1111", "1111"])],
            }],
        };
        let (out, _) = resize_sprite(&s, (4, 4), Anchor::FeetCentre, Mode::PadCrop);
        assert_eq!((out.cells_w, out.cells_h), (4, 4));
        assert_eq!(
            out.seqs[0].frames[0].len(),
            8,
            "4 cells at grain 2 is 8 rows"
        );
        assert_eq!(out.seqs[0].frames[0][0].chars().count(), 8);
    }

    #[test]
    fn a_resize_keeps_everything_about_the_record_it_was_not_asked_to_change() {
        // The sequences' poses, modes and rates are not a resize's business, and
        // a resize that reset one would be a silent retiming.
        let s = Sprite {
            id: "x".into(),
            name: "X".into(),
            prefix: "art.",
            cells_w: 2,
            cells_h: 2,
            grain: 1,
            fps: 12.0,
            pal: vec![".".into(), "#ffffff".into()],
            seqs: vec![Seq {
                state: "run".into(),
                mode: "loop".into(),
                fps: 6.0,
                frames: vec![frame(&["11", "11"]), frame(&["1.", ".1"])],
            }],
        };
        let (out, _) = resize_sprite(&s, (4, 4), Anchor::FeetCentre, Mode::PadCrop);
        assert_eq!(out.prefix, "art.");
        assert_eq!(out.fps, 12.0);
        assert_eq!(out.pal, s.pal);
        assert_eq!(out.seqs[0].state, "run");
        assert_eq!(out.seqs[0].mode, "loop");
        assert_eq!(out.seqs[0].fps, 6.0);
        assert_eq!(out.seqs[0].frames.len(), 2);
    }
}
