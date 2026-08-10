//! Frame text to RGBA, by the same rules the game bakes with.
//!
//! # Why this is written twice
//!
//! `yugen-render/src/sprite/baked.rs::bake_frame` is the authority: it is what
//! actually paints the character you play. This is a second implementation of
//! the same forty lines, and the duplication is deliberate.
//!
//! The alternative is for the editor to link `yugen-render`, and that crate is
//! Bevy — a windowing stack, a render graph and a GPU, pulled in so a text grid
//! can become bytes. It would also be the wrong bytes: the renderer bakes from
//! `yugen-data`, which is COMPILED content, and an editor must show the file as
//! it is being typed, before `contentc` has run and while it may not even be
//! valid. There is nothing in the render path that takes unsaved text.
//!
//! So the rules are restated here, and the honest statement of the risk is that
//! nothing yet PROVES the two agree. The rules are small and stable — they have
//! not changed since the TypeScript original — but "small and stable" is a reason
//! to expect agreement, not a mechanism that enforces it. The mechanism is a
//! shared golden fixture that both crates rasterise and compare against, and it
//! is not written yet. Until it is, treat `bake_frame` as normative and this as
//! the copy that must follow.
//!
//! # The rules
//!
//! - `.` and `0` are transparent. They are not the same character and they mean
//!   the same thing, because the palette's index 0 is written `.` and a frame may
//!   spell it either way.
//! - `1`-`9` index the palette. Index 0 is never painted, so a `pal` entry there
//!   is a placeholder rather than a colour.
//! - Anything else is an ERROR, not a colour. The original's header records the
//!   bug that motivated this: an unrecognised character used to be painted with
//!   whatever colour was last set.
//! - A frame is exactly `cellsH * grain` rows of exactly `cellsW * grain`
//!   characters. Dimensions arrive here in TEXELS, already multiplied.

/// Everything that can be wrong with a frame or a palette.
///
/// Carries positions rather than a message because the editor draws them: a bad
/// character is highlighted in the canvas at `row`/`col`, which is a better
/// answer than a sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RasterError {
    /// A palette entry that is not `#rrggbb`.
    Palette { index: usize, entry: String },
    /// The frame has the wrong number of rows.
    FrameRows { rows: usize, expected: u32 },
    /// One row is the wrong width, counted in CHARACTERS.
    RowWidth {
        row: usize,
        chars: usize,
        expected: u32,
    },
    /// A character that is neither transparent nor a live palette index.
    PaletteIndex { row: usize, col: usize, ch: char },
}

impl std::fmt::Display for RasterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RasterError::Palette { index, entry } => {
                write!(f, "pal[{index}] = {entry:?} is not #rrggbb")
            }
            RasterError::FrameRows { rows, expected } => {
                write!(f, "frame has {rows} rows, wants {expected}")
            }
            RasterError::RowWidth {
                row,
                chars,
                expected,
            } => write!(f, "row {row} is {chars} characters, wants {expected}"),
            RasterError::PaletteIndex { row, col, ch } => {
                write!(f, "row {row} col {col}: {ch:?} is not a palette index")
            }
        }
    }
}

impl std::error::Error for RasterError {}

/// `#rrggbb` and nothing else. The transparent slot is written `.` and is the one
/// entry allowed not to be a colour.
pub fn parse_hex(hex: &str) -> Option<[u8; 3]> {
    let b = hex.as_bytes();
    if b.len() != 7 || b[0] != b'#' {
        return None;
    }
    let mut out = [0u8; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let s = hex.get(1 + i * 2..3 + i * 2)?;
        *slot = u8::from_str_radix(s, 16).ok()?;
    }
    Some(out)
}

/// Resolve an authored palette to RGB triples.
///
/// Index 0 is never painted, so whatever it holds is passed through as black
/// rather than rejected — `.` is the convention and refusing it would make every
/// real file an error.
pub fn palette(pal: &[String]) -> Result<Vec<[u8; 3]>, RasterError> {
    pal.iter()
        .enumerate()
        .map(|(i, entry)| {
            if i == 0 {
                return Ok([0, 0, 0]);
            }
            parse_hex(entry).ok_or_else(|| RasterError::Palette {
                index: i,
                entry: entry.clone(),
            })
        })
        .collect()
}

/// Rasterise one frame into a fresh RGBA8 buffer of `w * h` texels.
pub fn raster(frame: &[String], pal: &[[u8; 3]], w: u32, h: u32) -> Result<Vec<u8>, RasterError> {
    if frame.len() != h as usize {
        return Err(RasterError::FrameRows {
            rows: frame.len(),
            expected: h,
        });
    }
    let mut out = vec![0u8; (w * h * 4) as usize];
    for (row, line) in frame.iter().enumerate() {
        // Counted, not measured with `len()`, which is bytes. Content is ASCII so
        // the two agree today and this keeps agreeing if it ever is not.
        let chars = line.chars().count();
        if chars != w as usize {
            return Err(RasterError::RowWidth {
                row,
                chars,
                expected: w,
            });
        }
        for (col, ch) in line.chars().enumerate() {
            if ch == '.' || ch == '0' {
                continue;
            }
            let idx = (ch as u32).wrapping_sub('0' as u32) as usize;
            if idx < 1 || idx >= pal.len() {
                return Err(RasterError::PaletteIndex { row, col, ch });
            }
            let c = pal[idx];
            let px = ((row * w as usize) + col) * 4;
            out[px] = c[0];
            out[px + 1] = c[1];
            out[px + 2] = c[2];
            out[px + 3] = 255;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn both_spellings_of_transparent_leave_alpha_at_zero() {
        // `.` and `0` are different characters meaning the same thing, and a real
        // file uses both — the palette's slot 0 is written `.`, and frames in
        // `icons.toml` use `0` inside a run of digits for legibility.
        let pal = palette(&rows(&[".", "#ff0000"])).expect("palette");
        let out = raster(&rows(&[".0", "11"]), &pal, 2, 2).expect("rasters");
        assert_eq!(&out[0..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..8], &[0, 0, 0, 0]);
        assert_eq!(&out[8..12], &[255, 0, 0, 255]);
    }

    #[test]
    fn a_digit_indexes_the_palette_in_row_major_order() {
        let pal = palette(&rows(&[".", "#010203", "#0a0b0c"])).expect("palette");
        let out = raster(&rows(&["12", "21"]), &pal, 2, 2).expect("rasters");
        assert_eq!(&out[0..4], &[1, 2, 3, 255]);
        assert_eq!(&out[4..8], &[10, 11, 12, 255]);
        assert_eq!(&out[8..12], &[10, 11, 12, 255]);
        assert_eq!(&out[12..16], &[1, 2, 3, 255]);
    }

    #[test]
    fn an_unknown_character_is_refused_rather_than_painted() {
        // The bug the original's header records: an unrecognised glyph used to be
        // drawn in whatever colour was last set, so a typo became art.
        let pal = palette(&rows(&[".", "#ffffff"])).expect("palette");
        assert_eq!(
            raster(&rows(&["1x"]), &pal, 2, 1),
            Err(RasterError::PaletteIndex {
                row: 0,
                col: 1,
                ch: 'x'
            })
        );
        // And an index past the end of the palette is the same kind of mistake.
        assert_eq!(
            raster(&rows(&["19"]), &pal, 2, 1),
            Err(RasterError::PaletteIndex {
                row: 0,
                col: 1,
                ch: '9'
            })
        );
    }

    #[test]
    fn a_frame_of_the_wrong_shape_is_refused_before_it_is_drawn() {
        let pal = palette(&rows(&[".", "#ffffff"])).expect("palette");
        assert_eq!(
            raster(&rows(&["11"]), &pal, 2, 2),
            Err(RasterError::FrameRows {
                rows: 1,
                expected: 2
            })
        );
        assert_eq!(
            raster(&rows(&["111", "11"]), &pal, 2, 2),
            Err(RasterError::RowWidth {
                row: 0,
                chars: 3,
                expected: 2
            })
        );
    }

    #[test]
    fn the_transparent_slot_may_be_a_dot_and_every_other_slot_may_not() {
        assert!(palette(&rows(&[".", "#ffffff"])).is_ok());
        assert_eq!(
            palette(&rows(&[".", "teal"])),
            Err(RasterError::Palette {
                index: 1,
                entry: "teal".to_string()
            })
        );
        assert_eq!(parse_hex("#1b1220"), Some([0x1b, 0x12, 0x20]));
        assert_eq!(parse_hex("#1b122"), None);
        assert_eq!(parse_hex("1b1220"), None);
    }
}
