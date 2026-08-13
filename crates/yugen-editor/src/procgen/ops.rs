//! Transformations of one frame, none of which change its shape.
//!
//! # Why shape-preservation is the rule and not a convention
//!
//! A frame is `cellsW * grain` characters by `cellsH * grain` rows, and the
//! numbers live on the record rather than on the frame. So a frame is only
//! meaningful against a grid it does not carry, and an operator that returned a
//! different shape would produce a record whose art and whose declaration
//! disagree. `bake_frame` answers that with `SpriteError::FrameRows`, which
//! `bake_sprite_table` turns into a **panic at `PreStartup`** — the game does not
//! start, and the message is about a row count rather than about the button that
//! was pressed.
//!
//! Holding the invariant inside each function means a panel can bind any of
//! these to a button with no check, and it means the property is testable
//! against all of `content/` at once.
//!
//! # Transparency has two spellings
//!
//! FORMAT.md §5 makes both `.` and `0` transparent. Reading has to accept both;
//! writing picks one, and everything here writes [`CLEAR`] so a generated frame
//! looks like the hand-authored ones next to it.

use crate::sprite::Frame;

/// The character written for a transparent texel.
///
/// `0` means the same thing to the raster and appears in some authored art, but
/// a generator has to choose one, and `.` is what `content/sprites/icons.toml`
/// uses for all 87 of its records.
pub const CLEAR: char = '.';

/// Whether a texel is transparent — either spelling.
pub fn is_clear(ch: char) -> bool {
    ch == '.' || ch == '0'
}

/// The frame's width in texels, taken from its first row.
///
/// Zero for an empty frame. Every function here reads the width once rather than
/// per row: a ragged frame is a bug upstream, and reading per row would paper
/// over it by producing a differently-ragged output.
pub fn width(f: &Frame) -> usize {
    f.first().map_or(0, |r| r.chars().count())
}

/// A frame as a grid of characters, padded to `width` with [`CLEAR`].
///
/// Everything below works on this rather than on `String`, because a row is
/// indexed by texel and a `String` is indexed by byte. The art is ASCII today
/// and `chars()` would agree, but "today" is not an invariant and an off-by-one
/// in a paint tool is invisible until it is in a file.
fn grid(f: &Frame) -> Vec<Vec<char>> {
    let w = width(f);
    f.iter()
        .map(|row| {
            let mut cs: Vec<char> = row.chars().collect();
            cs.resize(w, CLEAR);
            cs
        })
        .collect()
}

/// Back to rows.
fn rows(g: Vec<Vec<char>>) -> Frame {
    g.into_iter().map(|r| r.into_iter().collect()).collect()
}

/// Mirror left-to-right.
pub fn mirror_x(f: &Frame) -> Frame {
    let mut g = grid(f);
    for row in &mut g {
        row.reverse();
    }
    rows(g)
}

/// Mirror top-to-bottom.
pub fn mirror_y(f: &Frame) -> Frame {
    let mut g = grid(f);
    g.reverse();
    rows(g)
}

/// Copy the left half onto the right, mirrored.
///
/// The symmetry a creature usually wants, applied to art that is already drawn —
/// as opposed to [`mirror_x`], which flips it. An odd width keeps its centre
/// column, which is the only reading of "half" that does not lose a texel.
pub fn fold_x(f: &Frame) -> Frame {
    let mut g = grid(f);
    let w = width(f);
    for row in &mut g {
        for x in 0..w / 2 {
            row[w - 1 - x] = row[x];
        }
    }
    rows(g)
}

/// Rotate a quarter turn clockwise, or `None` if the frame is not square.
///
/// Returning `None` rather than resizing is the point. A rotate that changed
/// `cellsW` and `cellsH` would be a resize wearing a rotate's name, and would
/// need everything a resize needs — an anchor, a report of what was lost, a
/// guard against a mob's body box. On a square frame it is none of those, and
/// once every drawing in the game is 8x8 it is total.
pub fn rotate_cw(f: &Frame) -> Option<Frame> {
    let g = grid(f);
    let (w, h) = (width(f), g.len());
    if w != h || w == 0 {
        return None;
    }
    let mut out = vec![vec![CLEAR; w]; h];
    for (y, row) in g.iter().enumerate() {
        for (x, &ch) in row.iter().enumerate() {
            out[x][h - 1 - y] = ch;
        }
    }
    Some(rows(out))
}

/// Move the picture, either wrapping or dropping what leaves the grid.
///
/// `wrap` is the difference between nudging a drawing into place — where
/// anything that falls off an edge is gone — and scrolling a pattern, where it
/// comes back on the other side.
pub fn shift(f: &Frame, dx: i32, dy: i32, wrap: bool) -> Frame {
    let g = grid(f);
    let (w, h) = (width(f), g.len());
    if w == 0 || h == 0 {
        return f.clone();
    }
    let mut out = vec![vec![CLEAR; w]; h];
    for (y, row) in g.iter().enumerate() {
        for (x, &ch) in row.iter().enumerate() {
            let (tx, ty) = (x as i32 + dx, y as i32 + dy);
            let (tx, ty) = if wrap {
                (
                    tx.rem_euclid(w as i32) as usize,
                    ty.rem_euclid(h as i32) as usize,
                )
            } else {
                if tx < 0 || ty < 0 || tx >= w as i32 || ty >= h as i32 {
                    continue;
                }
                (tx as usize, ty as usize)
            };
            out[ty][tx] = ch;
        }
    }
    rows(out)
}

/// Draw `ink` in every transparent texel orthogonally touching an opaque one.
///
/// Four-neighbour rather than eight: a diagonal outline thickens corners into
/// blobs, and at eight texels across a corner is a noticeable fraction of the
/// whole silhouette.
///
/// Only transparent texels are written, so an outline never eats the drawing —
/// which also makes it idempotent in the useful sense: outlining twice grows the
/// border by two, and outlining the same frame twice in a row does not silently
/// recolour what the first pass drew.
pub fn outline(f: &Frame, ink: char) -> Frame {
    let g = grid(f);
    let (w, h) = (width(f), g.len());
    let mut out = g.clone();
    for y in 0..h {
        for x in 0..w {
            if !is_clear(g[y][x]) {
                continue;
            }
            let touching = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)]
                .iter()
                .any(|(dx, dy)| {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    nx >= 0
                        && ny >= 0
                        && (nx as usize) < w
                        && (ny as usize) < h
                        && !is_clear(g[ny as usize][nx as usize])
                });
            if touching {
                out[y][x] = ink;
            }
        }
    }
    rows(out)
}

/// Light the edges facing `from` and shade the ones facing away.
///
/// `from` is a direction, not a position: `(-1, -1)` is the top-left, which is
/// where this game's art is lit from. A texel is lit if the neighbour in that
/// direction is transparent — it is on the rim facing the light — and shaded if
/// the neighbour in the opposite direction is.
///
/// Only opaque texels are touched, and the interior is left alone, so this reads
/// as a rim light over whatever base colour was already there rather than as a
/// repaint.
pub fn auto_shade(f: &Frame, lit: char, shade: char, from: (i32, i32)) -> Frame {
    let g = grid(f);
    let (w, h) = (width(f), g.len());
    let mut out = g.clone();
    let clear_at = |x: i32, y: i32| -> bool {
        x < 0 || y < 0 || x >= w as i32 || y >= h as i32 || is_clear(g[y as usize][x as usize])
    };
    for (y, out_row) in out.iter_mut().enumerate() {
        for (x, ch) in out_row.iter_mut().enumerate() {
            if is_clear(*ch) {
                continue;
            }
            let (x, y) = (x as i32, y as i32);
            if clear_at(x + from.0, y + from.1) {
                *ch = lit;
            } else if clear_at(x - from.0, y - from.1) {
                *ch = shade;
            }
        }
    }
    rows(out)
}

/// Checkerboard `a` and `b` over the opaque texels.
///
/// `phase` selects which of the two colours lands on the even squares, so
/// dithering a region and then dithering it again with the other phase gives the
/// inverse pattern rather than the same one.
///
/// Transparent texels are left transparent: a dither is a shading tool, and one
/// that filled the background would be a fill.
pub fn dither(f: &Frame, a: char, b: char, phase: u8) -> Frame {
    let mut g = grid(f);
    for (y, row) in g.iter_mut().enumerate() {
        for (x, ch) in row.iter_mut().enumerate() {
            if is_clear(*ch) {
                continue;
            }
            *ch = if (x + y + phase as usize).is_multiple_of(2) {
                a
            } else {
                b
            };
        }
    }
    rows(g)
}

/// Flood the contiguous region of like texels at `(x, y)` with `ink`.
///
/// Four-connected, matching [`outline`], and matching what a diagonal gap looks
/// like to the eye at this size: a one-texel diagonal reads as a wall, so a fill
/// that leaked through it would read as a bug.
///
/// The two transparent spellings are one region — clicking a `.` next to a `0`
/// fills both, because they draw the same and telling them apart would be a
/// distinction only the file makes.
pub fn flood(f: &Frame, x: usize, y: usize, ink: char) -> Frame {
    let mut g = grid(f);
    let (w, h) = (width(f), g.len());
    if x >= w || y >= h {
        return f.clone();
    }
    let from = g[y][x];
    if from == ink || (is_clear(from) && is_clear(ink)) {
        return f.clone();
    }
    let matches = |ch: char| {
        if is_clear(from) {
            is_clear(ch)
        } else {
            ch == from
        }
    };

    let mut stack = vec![(x, y)];
    while let Some((x, y)) = stack.pop() {
        if !matches(g[y][x]) {
            continue;
        }
        g[y][x] = ink;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    rows(g)
}

/// Rewrite palette indices through a lookup, `map[i]` replacing digit `i`.
///
/// What a palette edit needs when the colours moved but the drawing did not —
/// reordering a ramp, or freeing index 1 for an outline. Characters that are not
/// digits are left alone, which in a well-formed frame means `.` and nothing
/// else.
pub fn remap(f: &Frame, map: &[char; 10]) -> Frame {
    let mut g = grid(f);
    for row in &mut g {
        for ch in row.iter_mut() {
            if let Some(d) = ch.to_digit(10) {
                *ch = map[d as usize];
            }
        }
    }
    rows(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: &[&str]) -> Frame {
        rows.iter().map(|r| (*r).to_string()).collect()
    }

    /// Asymmetric on both axes, so a flip that did nothing would be visible.
    fn sample() -> Frame {
        frame(&["12..", "3...", "44..", "...5"])
    }

    #[test]
    fn every_operator_keeps_the_shape_it_was_given() {
        // The invariant the whole module exists to hold. A frame whose row count
        // or row width moved is a record whose art and whose `cellsH` disagree,
        // and the renderer answers that with a panic rather than an error.
        let f = sample();
        let (w, h) = (width(&f), f.len());
        let outs = [
            mirror_x(&f),
            mirror_y(&f),
            fold_x(&f),
            rotate_cw(&f).expect("square"),
            shift(&f, 1, -2, false),
            shift(&f, 1, -2, true),
            outline(&f, '9'),
            auto_shade(&f, '8', '7', (-1, -1)),
            dither(&f, '1', '2', 0),
            flood(&f, 3, 0, '6'),
            remap(&f, &['.', '2', '1', '3', '4', '5', '6', '7', '8', '9']),
        ];
        for (i, out) in outs.iter().enumerate() {
            assert_eq!(out.len(), h, "operator {i} changed the row count");
            assert_eq!(width(out), w, "operator {i} changed the row width");
            for row in out {
                assert_eq!(row.chars().count(), w, "operator {i} left a ragged row");
            }
        }
    }

    #[test]
    fn mirroring_twice_is_the_identity() {
        let f = sample();
        assert_eq!(mirror_x(&mirror_x(&f)), f);
        assert_eq!(mirror_y(&mirror_y(&f)), f);
    }

    #[test]
    fn four_quarter_turns_come_back() {
        let f = sample();
        let mut g = f.clone();
        for _ in 0..4 {
            g = rotate_cw(&g).expect("square");
        }
        assert_eq!(g, f);
    }

    #[test]
    fn a_quarter_turn_actually_turns() {
        // Guards against the identity passing the round-trip test above.
        assert_eq!(
            rotate_cw(&frame(&["12", "34"])).expect("square"),
            frame(&["31", "42"])
        );
    }

    #[test]
    fn a_non_square_frame_refuses_to_rotate_rather_than_resizing() {
        // The player is 8x10 until the migration lands. A rotate that "worked"
        // there would have silently rewritten `cellsW` and `cellsH`.
        assert_eq!(rotate_cw(&frame(&["12", "34", "56"])), None);
    }

    #[test]
    fn folding_copies_the_left_half_and_keeps_an_odd_centre() {
        assert_eq!(fold_x(&frame(&["12.."])), frame(&["1221"]));
        // Odd width: the middle column is its own mirror and must survive.
        assert_eq!(fold_x(&frame(&["12..."])), frame(&["12.21"]));
    }

    #[test]
    fn a_wrapping_shift_returns_and_a_dropping_one_does_not() {
        let f = frame(&["12", "34"]);
        assert_eq!(shift(&f, 1, 0, true), frame(&["21", "43"]));
        assert_eq!(shift(&f, 1, 0, false), frame(&[".1", ".3"]));
        // Wrapping a full turn is the identity; dropping one is empty.
        assert_eq!(shift(&f, 2, 2, true), f);
        assert_eq!(shift(&f, 2, 0, false), frame(&["..", ".."]));
    }

    #[test]
    fn an_outline_only_writes_into_transparent_texels() {
        // Otherwise outlining would eat the drawing it was meant to frame.
        let f = frame(&["...", ".1.", "..."]);
        let out = outline(&f, '9');
        assert_eq!(out, frame(&[".9.", "919", ".9."]));
        assert_eq!(out[1].chars().nth(1), Some('1'), "the art survived");
    }

    #[test]
    fn an_outline_is_orthogonal_and_not_diagonal() {
        // Eight-neighbour would fill the corners too, and at eight texels across
        // that turns a corner into a blob.
        let out = outline(&frame(&["...", ".1.", "..."]), '9');
        assert_eq!(out[0].chars().next(), Some('.'), "the corner stayed clear");
    }

    #[test]
    fn shading_lights_the_side_the_light_is_on() {
        // A solid block lit from the top-left: the top and left rim are lit, the
        // bottom and right rim are shaded, and the interior keeps its colour.
        let f = frame(&["111", "111", "111"]);
        let out = auto_shade(&f, '8', '7', (-1, -1));
        assert_eq!(out[0].chars().next(), Some('8'), "the top-left rim is lit");
        assert_eq!(out[2].chars().nth(2), Some('7'), "the far rim is shaded");
        assert_eq!(
            out[1].chars().nth(1),
            Some('1'),
            "the interior is untouched"
        );
    }

    #[test]
    fn shading_never_paints_the_background() {
        let f = frame(&["1.", ".."]);
        let out = auto_shade(&f, '8', '7', (-1, -1));
        assert!(is_clear(out[1].chars().next().expect("a row")));
        assert!(is_clear(out[0].chars().nth(1).expect("a row")));
    }

    #[test]
    fn a_dither_leaves_the_background_alone_and_phase_inverts_it() {
        let f = frame(&["11", "1."]);
        let a = dither(&f, '1', '2', 0);
        let b = dither(&f, '1', '2', 1);
        assert_eq!(a, frame(&["12", "2."]));
        assert_eq!(b, frame(&["21", "1."]));
        assert!(is_clear(a[1].chars().nth(1).expect("a row")));
    }

    #[test]
    fn a_flood_stops_at_a_different_index() {
        let f = frame(&["1122", "1122"]);
        assert_eq!(flood(&f, 0, 0, '3'), frame(&["3322", "3322"]));
    }

    #[test]
    fn a_flood_does_not_leak_through_a_diagonal() {
        // A one-texel diagonal reads as a wall at this size, so a fill that got
        // through it would read as a bug rather than as a rule.
        let f = frame(&["..1", ".1.", "1.."]);
        let out = flood(&f, 2, 2, '9');
        // The three clear texels below the diagonal, and only those. The three
        // above it are a separate region and keep their spelling.
        assert_eq!(out, frame(&["..1", ".19", "199"]));
    }

    #[test]
    fn both_spellings_of_transparent_flood_as_one_region() {
        // `.` and `0` draw identically, so treating them as two regions would be
        // a distinction only the file makes.
        let f = frame(&[".0", "0."]);
        assert_eq!(flood(&f, 0, 0, '1'), frame(&["11", "11"]));
    }

    #[test]
    fn a_flood_that_would_change_nothing_changes_nothing() {
        let f = frame(&["11", "11"]);
        assert_eq!(flood(&f, 0, 0, '1'), f);
        assert_eq!(
            flood(&f, 9, 9, '2'),
            f,
            "a click outside the grid is a no-op"
        );
    }

    #[test]
    fn a_remap_moves_indices_and_leaves_the_background() {
        let map = ['.', '2', '1', '3', '4', '5', '6', '7', '8', '9'];
        assert_eq!(remap(&frame(&["12.", "21."]), &map), frame(&["21.", "12."]));
    }

    #[test]
    fn a_ragged_row_is_padded_rather_than_propagated() {
        // `sprite::read` checks a frame's row COUNT but not its row WIDTH, so a
        // short row can reach these functions. Padding it makes the output
        // rectangular, which is what the record declares; propagating it would
        // hand a ragged frame to the next operator.
        let out = mirror_x(&frame(&["12", "1"]));
        assert_eq!(out, frame(&["21", ".1"]));
    }
}
