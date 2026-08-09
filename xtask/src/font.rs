//! `cargo xtask font` — bake the shipped typeface into a Rust glyph table.
//!
//! ```text
//! cargo xtask font           rasterise `content/fonts/`, writing the table
//! cargo xtask font --check   exit 1 if the committed table is stale (the gate)
//! ```
//!
//! # Why the font is baked and not loaded
//!
//! Everything this game draws goes into one low-resolution buffer that is then
//! upscaled with a nearest sampler, so a glyph is either on a whole buffer pixel
//! or it is wrong. A runtime rasteriser would have to be told that, every frame,
//! at every size, and the first thing it would do with the instruction is
//! antialias an edge anyway. Baking to one bit per pixel makes the constraint
//! unrepresentable rather than merely documented.
//!
//! It also keeps the boundary `crates/yugen-render` already has: the layout
//! functions in `ui` are pure `View -> Vec<UiPrim>` and are tested with no GPU,
//! no window and no `AssetServer`. `TextStyle::measure` has to answer in that
//! world, which means the metrics must be constants, which means they are
//! resolved here rather than at startup.
//!
//! # Why Departure Mono, at eleven pixels, once
//!
//! Departure Mono is a PIXEL font: its outlines are square pixels on a fixed
//! grid, and its documentation says to set it in multiples of 11 px. Rasterised
//! at exactly that grid, every curve in the file lands on a pixel boundary and
//! the threshold below has nothing to round — the bitmap this produces is the
//! bitmap the designer drew, not a sample of it.
//!
//! That is also why there is one bake and not one per size. `TextStyle` carries
//! an integer scale and the painter multiplies, so a title at scale 4 is the
//! same glyph with each pixel four times as wide. Baking a second size would
//! produce a DIFFERENT set of pixels for the same letter, which is the one thing
//! a pixel face must not do.
//!
//! # Why this is xtask and not a build script
//!
//! A `build.rs` would run this on every clean build in every environment,
//! including the ones with no font file because someone checked out with a
//! sparse filter — and it would need `ab_glyph` in the render crate's
//! dependency tree to do it. Generating on demand into a committed file, with a
//! `--check` gate to keep it honest, is what `contentc` already does for
//! `content/` and there is no reason for the font to be the exception.

use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use ab_glyph::{Font, FontRef, ScaleFont};

/// The typeface, relative to the repository root.
const FONT: &str = "content/fonts/DepartureMono-Regular.otf";

/// The generated table, relative to the repository root.
const OUT: &str = "crates/yugen-render/src/ui/font_table.rs";

/// Font units per logical pixel, read off Departure Mono's own outlines.
///
/// **This is the number the whole bake turns on, and it is measured, not
/// chosen.** Dumping the raw outlines settles it in one look — every coordinate
/// in the face, on both axes, is a multiple of fifty:
///
/// ```text
/// glyph                       advance      x                    y
/// U+2588 FULL BLOCK             350     0, 350            -150, 550
/// U+2500 LIGHT HORIZONTAL       350     0, 350             150, 200
/// U+004D 'M'                    350     50..300 by 50      0, 150, 250, 300, 400
/// U+007C '|'                    350     150, 200           -50, 450
/// ```
///
/// So the design grid is 50 units to the pixel, the em cell is 7 x 14 of them
/// (advance 350; block from 150 below the baseline to 550 above), and letters
/// occupy the middle five columns with an 8-pixel cap height.
///
/// Getting here took three wrong answers, all of them from trusting a px figure
/// instead of the outline. `ab_glyph`'s `PxScale` is **not** pixels per em — it
/// scales by `ascent - descent`, which is 700 units here, not the 550 the em is.
/// So `as_scaled(11.0)` — the size Departure Mono's own README asks for — does
/// not put a font pixel on a buffer pixel at all; it reports a 5.5 px advance,
/// and every "measurement" taken through it was of a face rasterised off its
/// own grid. That is what made a stem look 0.875 px wide, which in turn made
/// centre-sampling erase `U+007C` and area-sampling erase the crossbar of `A`.
///
/// Scaling from the design grid instead makes all of it go away: edges land on
/// integers, coverage comes back 0.0 or 1.0, and the bitmap is the one the
/// designer drew.
const UNITS_PER_PIXEL: f32 = 50.0;

/// Fraction of a font pixel that must be inked for it to be lit.
///
/// A half, applied to the pixel's AREA — see [`SS`]. On the design grid nothing
/// is ever close to the threshold; it is here so that a future face whose
/// outlines are not perfectly square degrades into "mostly covered wins"
/// rather than into noise.
const INK: f32 = 0.5;

/// Linear supersampling factor for the rasteriser.
///
/// Each font pixel is the average coverage of an `SS` x `SS` block. On the
/// design grid this is redundant — every block comes back all-ink or no-ink —
/// and that redundancy is the point: it is the assertion that the grid is
/// right. If a change to [`UNITS_PER_PIXEL`] ever put the face back off its
/// grid, this degrades to a soft answer instead of silently dropping the
/// columns that a single sample would have missed.
const SS: i32 = 8;

/// Every character the game can set, in table order.
///
/// Order is not arbitrary — it is the order glyphs land in the baked atlas, and
/// `FontAtlas::rect` indexes by position. Appending is free; reordering rebakes
/// the atlas, which is fine, but do it in one commit.
///
/// The four groups after ASCII each earn their place:
///
/// - **Latin-1 and `ū`.** `ū` is load-bearing: the title card spells `Yūgen`,
///   and without the glyph the game greets the player with a tofu box. The rest
///   of the range is there so an authored item name with an accent in it stops
///   being a bug waiting to happen.
/// - **Arrows and typographic punctuation.** The control hints are written with
///   real arrows, and the em dash appears in every "none — no world yet" row.
/// - **Box drawing.** Panel frames drawn from glyphs rather than from rectangles
///   snap to the text grid for free, which is what makes a bordered panel line
///   up with the rows inside it.
/// - **Blocks and stars.** Bars, meters and the dash pip. A block glyph fills a
///   partial cell in whole-pixel steps, which a scaled rectangle cannot.
const CHARS: &str = concat!(
    // ASCII, 0x20..=0x7E, whole and in order.
    " !\"#$%&'()*+,-./0123456789:;<=>?",
    "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_",
    "`abcdefghijklmnopqrstuvwxyz{|}~",
    // Latin-1 punctuation and symbols.
    "\u{00a1}\u{00a2}\u{00a3}\u{00a5}\u{00a7}\u{00ab}\u{00ac}\u{00b0}\u{00b1}",
    "\u{00b7}\u{00bb}\u{00bf}\u{00d7}\u{00f7}",
    // Latin-1 letters, plus the one Latin Extended-A glyph the title needs.
    "\u{00c0}\u{00c1}\u{00c2}\u{00c3}\u{00c4}\u{00c5}\u{00c6}\u{00c7}",
    "\u{00c8}\u{00c9}\u{00ca}\u{00cb}\u{00cc}\u{00cd}\u{00ce}\u{00cf}",
    "\u{00d1}\u{00d2}\u{00d3}\u{00d4}\u{00d5}\u{00d6}\u{00d8}\u{00d9}",
    "\u{00da}\u{00db}\u{00dc}\u{00dd}\u{00df}",
    "\u{00e0}\u{00e1}\u{00e2}\u{00e3}\u{00e4}\u{00e5}\u{00e6}\u{00e7}",
    "\u{00e8}\u{00e9}\u{00ea}\u{00eb}\u{00ec}\u{00ed}\u{00ee}\u{00ef}",
    "\u{00f1}\u{00f2}\u{00f3}\u{00f4}\u{00f5}\u{00f6}\u{00f8}\u{00f9}",
    "\u{00fa}\u{00fb}\u{00fc}\u{00fd}\u{00ff}\u{016b}",
    // Arrows.
    "\u{2190}\u{2191}\u{2192}\u{2193}",
    // Typographic punctuation.
    "\u{2013}\u{2014}\u{2018}\u{2019}\u{201c}\u{201d}\u{2022}\u{2026}",
    // Box drawing: light single, light double, and the four corners of each.
    "\u{2500}\u{2502}\u{250c}\u{2510}\u{2514}\u{2518}\u{251c}\u{2524}",
    "\u{252c}\u{2534}\u{253c}",
    "\u{2550}\u{2551}\u{2554}\u{2557}\u{255a}\u{255d}",
    // Shade and full blocks, then the eighth-height bar ramp.
    "\u{2591}\u{2592}\u{2593}\u{2588}",
    "\u{2581}\u{2582}\u{2583}\u{2584}\u{2585}\u{2586}\u{2587}",
    // Eighth-width bar ramp, for a meter that fills horizontally.
    "\u{258f}\u{258e}\u{258d}\u{258c}\u{258b}\u{258a}\u{2589}",
    // Stars.
    "\u{2605}\u{2606}\u{2726}\u{2727}",
);

/// What to do with the table.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Rasterise and write, reporting what changed.
    Write,
    /// Rasterise and compare, failing if the committed file differs.
    Check,
}

/// One glyph's pixels, and the character it belongs to.
struct Baked {
    ch: char,
    /// One `u16` per row, MSB the leftmost column of the cell.
    rows: Vec<u16>,
}

/// The cell every glyph is baked into, in font pixels.
///
/// Uniform across the face, because the painter places glyphs on a fixed
/// advance and reads a fixed sub-rect out of the atlas. A per-glyph box would
/// buy a few atlas columns and cost the whole-pixel guarantee.
struct Cell {
    /// Leftmost column any glyph reaches, as a pixel offset. Usually zero, and
    /// negative for the rare glyph whose ink starts left of its origin.
    left: i32,
    w: i32,
    /// Rows above the baseline.
    ascent: i32,
    /// Rows below it. Departure Mono has real descenders; the face this
    /// replaces did not, which is why `cap_h` and `cell_h` used to be the same
    /// number and no longer are.
    descent: i32,
    /// Pen movement from one glyph's origin to the next. Monospaced, so one
    /// number for the face.
    advance: i32,
    /// Cap height: rows from the baseline to the top of a capital.
    ///
    /// Distinct from [`Cell::ascent`], which also has to clear the accents on
    /// `A` with a ring over it. Every baseline in the tree is worked out from
    /// the cap, because that is the part of a line the eye centres on — a row
    /// centred on the ascent sits visibly low.
    cap: i32,
}

impl Cell {
    fn h(&self) -> i32 {
        self.ascent + self.descent
    }
}

/// Rasterise the face and reconcile it with the committed table.
pub fn run(root: &Path, mode: Mode) -> Result<String, String> {
    let path = root.join(FONT);
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("xtask font: cannot read {}: {e}\n", path.display()))?;
    let font = FontRef::try_from_slice(&bytes)
        .map_err(|e| format!("xtask font: {FONT} is not a font this can read: {e}\n"))?;

    let (cell, baked, missing) = bake(&font)?;
    let source = render(&cell, &baked);
    let source = rustfmt(&source)?;

    let out = root.join(OUT);
    let current = std::fs::read_to_string(&out).unwrap_or_default();

    let note = if missing.is_empty() {
        String::new()
    } else {
        format!(
            "  note: {} character(s) the font has no glyph for were skipped: {}\n",
            missing.len(),
            missing.iter().collect::<String>()
        )
    };

    let summary = format!(
        "font: {} glyphs, {}x{} cell ({} up, {} down), {} px advance, {} px cap\n{note}",
        baked.len(),
        cell.w,
        cell.h(),
        cell.ascent,
        cell.descent,
        cell.advance,
        cell.cap,
    );

    match mode {
        Mode::Check if current != source => Err(format!(
            "xtask font: {OUT} is stale.\n  run `cargo xtask font` and commit the result.\n"
        )),
        Mode::Check => Ok(format!("{summary}  {OUT} is current\n")),
        Mode::Write if current == source => Ok(format!("{summary}  {OUT} unchanged\n")),
        Mode::Write => {
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("xtask font: cannot create {}: {e}\n", dir.display()))?;
            }
            std::fs::write(&out, &source)
                .map_err(|e| format!("xtask font: cannot write {}: {e}\n", out.display()))?;
            Ok(format!("{summary}  wrote {OUT}\n"))
        }
    }
}

/// Rasterise every character in [`CHARS`] onto one uniform cell.
///
/// Two passes, and they cannot be one: the cell is the union of every glyph's
/// bounds, so nothing can be placed until everything has been measured.
fn bake(font: &FontRef) -> Result<(Cell, Vec<Baked>, Vec<char>), String> {
    // `PxScale` is `ascent - descent` in pixels, NOT the em — see
    // [`UNITS_PER_PIXEL`]. This is the scale at which one design pixel is one
    // buffer pixel, times the supersampling factor.
    let one_to_one = font.height_unscaled() / UNITS_PER_PIXEL;
    let scaled = font.as_scaled(one_to_one * SS as f32);

    let mut present: Vec<char> = Vec::new();
    let mut missing: Vec<char> = Vec::new();
    for ch in CHARS.chars() {
        // Glyph id 0 is `.notdef`. A face that answers with it is a face that
        // does not have the character, whatever the lookup pretended.
        if scaled.glyph_id(ch).0 == 0 && ch != ' ' {
            missing.push(ch);
        } else {
            present.push(ch);
        }
    }

    /// One glyph's coverage, and where in the over-scaled plane it sits.
    struct Ink {
        cov: Vec<f32>,
        w: i32,
        h: i32,
        /// Top-left of `cov`, in over-scaled pixels from the origin/baseline.
        x0: i32,
        y0: i32,
    }

    // Pass one: rasterise everything over-scale, and take the union of the ink
    // in FONT pixels. Nothing can be placed until everything has been measured,
    // because the cell is that union.
    let (mut left, mut right, mut top, mut bottom) = (0i32, 0i32, 0i32, 0i32);
    let mut ink: Vec<Option<Ink>> = Vec::with_capacity(present.len());
    for &ch in &present {
        let Some(outline) = scaled.outline_glyph(scaled.scaled_glyph(ch)) else {
            ink.push(None); // A space, or anything else with no ink.
            continue;
        };
        // `px_bounds` is already rounded outward to whole over-scaled pixels,
        // so these are exact and the buffer is exactly the size `draw` indexes.
        let b = outline.px_bounds();
        let (x0, y0) = (b.min.x as i32, b.min.y as i32);
        let (w, h) = (b.max.x as i32 - x0, b.max.y as i32 - y0);
        let mut cov = vec![0.0f32; (w * h) as usize];
        outline.draw(|x, y, coverage| {
            let (x, y) = (x as i32, y as i32);
            if x < w && y < h {
                cov[(y * w + x) as usize] = coverage;
            }
        });

        // Down to font pixels. `div_euclid` and not integer division: these
        // coordinates go negative above the baseline, and `-1 / 8` is 0 while
        // `(-1).div_euclid(8)` is -1, which is the row the pixel is actually in.
        left = left.min(x0.div_euclid(SS));
        right = right.max((x0 + w + SS - 1).div_euclid(SS));
        top = top.min(y0.div_euclid(SS));
        bottom = bottom.max((y0 + h + SS - 1).div_euclid(SS));
        ink.push(Some(Ink { cov, w, h, x0, y0 }));
    }

    let cell = Cell {
        left,
        w: right - left,
        // `px_bounds` is +y down from the baseline, so ink above it is negative.
        ascent: -top,
        descent: bottom,
        advance: (scaled.h_advance(scaled.glyph_id('M')) / SS as f32).round() as i32,
        // Measured off a capital rather than declared: `M` has no overshoot and
        // no accent, so its ink height IS the cap height.
        cap: scaled
            .outline_glyph(scaled.scaled_glyph('M'))
            .map(|o| (-o.px_bounds().min.y / SS as f32).round() as i32)
            .unwrap_or(0),
    };

    if cell.w <= 0 || cell.h() <= 0 {
        return Err("xtask font: the face rasterised to nothing at this size\n".into());
    }
    // The grid check. If the advance is not a whole number of pixels the face
    // is being rasterised off its own design grid, and every glyph after the
    // first on a line lands on a fraction — see [`UNITS_PER_PIXEL`] for how
    // that failure looks from the inside. Better a stopped gate than a table
    // full of plausible, subtly wrong pixels.
    let exact = scaled.h_advance(scaled.glyph_id('M')) / SS as f32;
    if (exact - cell.advance as f32).abs() > 1e-3 {
        return Err(format!(
            "xtask font: advance is {exact} px, not a whole number — \
             UNITS_PER_PIXEL ({UNITS_PER_PIXEL}) does not match this face's grid\n"
        ));
    }
    if cell.w > 16 {
        return Err(format!(
            "xtask font: a {}px cell does not fit the u16 row the table uses\n",
            cell.w
        ));
    }

    // Pass two: sample the centre of each font pixel of the cell that now
    // exists.
    let mut baked = Vec::with_capacity(present.len());
    for (&ch, ink) in present.iter().zip(&ink) {
        let mut rows = vec![0u16; cell.h() as usize];
        if let Some(ink) = ink {
            for row in 0..cell.h() {
                for col in 0..cell.w {
                    // This font pixel's top-left, in over-scaled pixels.
                    let px0 = (cell.left + col) * SS - ink.x0;
                    let py0 = (row - cell.ascent) * SS - ink.y0;
                    let mut sum = 0.0f32;
                    for dy in 0..SS {
                        for dx in 0..SS {
                            let (sx, sy) = (px0 + dx, py0 + dy);
                            if sx >= 0 && sy >= 0 && sx < ink.w && sy < ink.h {
                                sum += ink.cov[(sy * ink.w + sx) as usize];
                            }
                        }
                    }
                    if sum / (SS * SS) as f32 >= INK {
                        // MSB is the leftmost column, matching `Face::lit`.
                        rows[row as usize] |= 1 << (cell.w - 1 - col);
                    }
                }
            }
        }
        baked.push(Baked { ch, rows });
    }

    Ok((cell, baked, missing))
}

/// The hollow box drawn for a character the face has no glyph for.
///
/// Generated rather than authored so it is always exactly the cell: a tofu
/// marker that is the wrong size is a marker that reads as a punctuation glyph.
fn tofu(cell: &Cell) -> Vec<u16> {
    let full = if cell.w >= 16 {
        u16::MAX
    } else {
        (1u16 << cell.w) - 1
    };
    let edges = full & !(full >> 1) | 1;
    // Only the part above the baseline, so tofu sits where a capital would.
    (0..cell.h())
        .map(|row| {
            if row >= cell.ascent {
                0
            } else if row == 0 || row == cell.ascent - 1 {
                full
            } else {
                edges
            }
        })
        .collect()
}

/// The generated source, as text.
fn render(cell: &Cell, baked: &[Baked]) -> String {
    let chars: String = baked.iter().map(|b| b.ch).collect();
    let mut s = String::with_capacity(baked.len() * cell.h() as usize * 24);

    s.push_str(
        "//! The baked typeface: Departure Mono, one bit per pixel.\n\
         //!\n\
         //! @generated by `cargo xtask font` from `content/fonts/`. Do not edit — edit\n\
         //! the character set in `xtask/src/font.rs` and re-run, or the `font` gate will\n\
         //! put it back. See that module for why the font is baked at all.\n\
         //!\n\
         //! Departure Mono is (c) 2022-2024 Helena Zhang, under the SIL Open Font\n\
         //! License 1.1. The licence travels with the source file in\n\
         //! `content/fonts/LICENSE`.\n\n",
    );

    let _ = writeln!(
        s,
        "/// Font units per pixel of the design grid this was rasterised on.\n\
         pub const UNITS_PER_PIXEL: i32 = {};\n",
        UNITS_PER_PIXEL as i32
    );
    let _ = writeln!(
        s,
        "/// Cell width in font pixels.\npub const CELL_W: i32 = {};\n",
        cell.w
    );
    let _ = writeln!(
        s,
        "/// Cell height in font pixels: [`ASCENT`] plus [`DESCENT`].\n\
         pub const CELL_H: i32 = {};\n",
        cell.h()
    );
    let _ = writeln!(
        s,
        "/// Rows of the cell above the baseline.\npub const ASCENT: i32 = {};\n",
        cell.ascent
    );
    let _ = writeln!(
        s,
        "/// Rows of the cell below the baseline. Non-zero: this face has real\n\
         /// descenders, so the cell is taller than the cap height.\n\
         pub const DESCENT: i32 = {};\n",
        cell.descent
    );
    let _ = writeln!(
        s,
        "/// Pen movement from one glyph to the next, in font pixels. The face is\n\
         /// monospaced, so this is a constant rather than a per-glyph table.\n\
         pub const ADVANCE: i32 = {};\n",
        cell.advance
    );
    let _ = writeln!(
        s,
        "/// Baseline to the top of a capital, in font pixels.\n\
         ///\n\
         /// Less than [`ASCENT`], which also clears accents. This is the number\n\
         /// vertical centring uses: a line centred on the ascent sits low by the\n\
         /// height of an accent nothing in the string had.\n\
         pub const CAP: i32 = {};\n",
        cell.cap
    );

    s.push_str("/// Every character the face has a glyph for, in [`ROWS`] order.\n");
    let _ = writeln!(s, "pub const CHARS: &str = {};\n", quoted(&chars));

    s.push_str(
        "/// The missing-glyph marker: a hollow box the height of a capital.\n\
         #[rustfmt::skip]\n",
    );
    let _ = writeln!(
        s,
        "pub const TOFU: [u16; CELL_H as usize] = [{}];\n",
        tofu(cell)
            .iter()
            .map(|bits| bin(*bits, cell.w))
            .collect::<Vec<_>>()
            .join(", ")
    );

    s.push_str(
        "/// Every glyph, [`CELL_H`] rows each, in [`CHARS`] order.\n\
         ///\n\
         /// One `u16` per row with the leftmost column in the high bit of\n\
         /// [`CELL_W`], so the literal reads as the pixel art it is — the same\n\
         /// encoding the authored face used, and for the same reason: it has to be\n\
         /// proofreadable in a diff.\n\
         #[rustfmt::skip]\n\
         pub const ROWS: &[u16] = &[\n",
    );
    for b in baked {
        let _ = writeln!(s, "    // {}", label(b.ch));
        for bits in &b.rows {
            let _ = writeln!(s, "    {}, // {}", bin(*bits, cell.w), art(*bits, cell.w));
        }
    }
    s.push_str("];\n");
    s
}

/// A row as a binary literal exactly [`Cell::w`] digits wide.
fn bin(bits: u16, w: i32) -> String {
    format!("0b{bits:0width$b}", width = w as usize)
}

/// A row as the pixels it lights, for the eye rather than the compiler.
fn art(bits: u16, w: i32) -> String {
    (0..w)
        .map(|col| {
            if (bits >> (w - 1 - col)) & 1 == 1 {
                '#'
            } else {
                '.'
            }
        })
        .collect()
}

/// A character named so the comment survives being a quote, a backslash or a
/// codepoint nothing renders.
fn label(ch: char) -> String {
    if ch.is_ascii_graphic() && ch != '\'' && ch != '\\' {
        format!("'{ch}'  U+{:04X}", ch as u32)
    } else {
        format!("U+{:04X}", ch as u32)
    }
}

/// A Rust string literal for `s`, escaping anything that is not plain ASCII.
///
/// Escaped rather than pasted: the set includes a backslash, a quote, and sixty
/// characters whose bytes would make the generated file depend on the editor
/// that next opens it.
fn quoted(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_ascii_graphic() || c == ' ' => out.push(c),
            c => {
                let _ = write!(out, "\\u{{{:04x}}}", c as u32);
            }
        }
    }
    out.push('"');
    out
}

/// Run the generated source through `rustfmt`.
///
/// So that the `fmt` gate and this one cannot disagree. Without it the first
/// `cargo fmt` after a bake would reformat the file and the next `font --check`
/// would call the result stale — two gates, each undoing the other.
fn rustfmt(source: &str) -> Result<String, String> {
    use std::io::Write as _;

    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout", "--quiet"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("xtask font: cannot run rustfmt: {e}\n"))?;

    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(source.as_bytes())
        .map_err(|e| format!("xtask font: cannot write to rustfmt: {e}\n"))?;

    let out = child
        .wait_with_output()
        .map_err(|e| format!("xtask font: rustfmt did not finish: {e}\n"))?;
    if !out.status.success() {
        return Err("xtask font: rustfmt rejected the generated source\n".into());
    }
    String::from_utf8(out.stdout)
        .map_err(|e| format!("xtask font: rustfmt output is not UTF-8: {e}\n"))
}
