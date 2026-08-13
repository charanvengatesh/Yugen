//! A whole drawing from a seed.
//!
//! # What this is for, and what it is not
//!
//! It is a way to stop starting from an empty grid. At eight texels across there
//! are not many silhouettes, and the useful thing a generator does here is
//! produce forty of them in a minute so the one worth keeping can be recognised
//! and then drawn properly. Nothing this emits is finished art, and the pipeline
//! below is tuned for "plausible and varied" rather than for "good".
//!
//! # The pipeline, and why each step is there
//!
//! 1. **Noise one half of the grid** at `density`. Half, because the mirror in
//!    step 4 is what makes the result read as a creature rather than as static —
//!    bilateral symmetry is the single cheapest cue for "this is a body".
//! 2. **Smooth it twice.** Raw noise at any density is speckle. A cellular pass
//!    — a texel lives if enough of its neighbours do — pulls speckle into
//!    regions, and two passes is where the shapes stop changing much.
//! 3. **Keep only the largest connected region.** Smoothing leaves islands, and
//!    an island reads as dirt on the screen rather than as part of the creature.
//! 4. **Mirror.**
//! 5. **Shade**, by rim-lighting from the top left, which is where this game's
//!    art is lit from.
//! 6. **Outline**, optionally, in the darkest index.
//!
//! Every step is a pure function of the seed, so a roll is reproducible from the
//! four bytes shown next to the button.

use crate::procgen::ops::{self, CLEAR};
use crate::procgen::palette::{self, Hsl};
use crate::rng::Rng;
use crate::sprite::Frame;

/// Which axis the drawing is mirrored across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// Left and right match. What a creature seen face-on looks like, and the
    /// default for that reason.
    MirrorX,
    /// Top and bottom match. Rare in a creature and useful for an item — a gem,
    /// a coin, a rune.
    MirrorY,
    /// No symmetry. Reads as debris more often than as a body, which is
    /// occasionally what is wanted.
    None,
}

impl Symmetry {
    pub const ALL: [Symmetry; 3] = [Symmetry::MirrorX, Symmetry::MirrorY, Symmetry::None];

    pub fn name(self) -> &'static str {
        match self {
            Symmetry::MirrorX => "left/right",
            Symmetry::MirrorY => "top/bottom",
            Symmetry::None => "none",
        }
    }
}

/// What to draw. Every field is an input to a pure function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recipe {
    pub seed: u32,
    /// Texels, not cells — `cellsW * grain`. The caller multiplies.
    pub w: usize,
    pub h: usize,
    /// How much of the half-grid starts filled, `0..=1`. Low reads as spindly,
    /// high as a solid block; the interesting band is roughly 0.4 to 0.65.
    pub density: f32,
    pub symmetry: Symmetry,
    /// How many palette indices the shading uses, `2..=9`. Two is a silhouette
    /// with a rim; more gives the ramp room to read as form.
    pub inks: usize,
    pub outline: bool,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe {
            seed: 1,
            w: 8,
            h: 8,
            // Fitted against the reference sheet, not chosen — see `smooth`.
            // This was 0.52, which produced creatures filling a quarter of the
            // canvas where the sheet's fill two thirds of theirs.
            density: 0.65,
            symmetry: Symmetry::MirrorX,
            inks: 4,
            outline: true,
        }
    }
}

impl Recipe {
    /// The palette index the body is filled with before shading.
    ///
    /// The middle of the ramp, so the rim light has somewhere to go in both
    /// directions. Never 1 when an outline is wanted, because 1 is the outline.
    fn body(&self) -> usize {
        let inks = self.inks.clamp(2, palette::MAX_STEPS);
        (inks / 2 + 1).max(if self.outline { 2 } else { 1 })
    }

    fn lit(&self) -> usize {
        self.inks.clamp(2, palette::MAX_STEPS)
    }

    fn shade(&self) -> usize {
        self.body()
            .saturating_sub(1)
            .max(if self.outline { 2 } else { 1 })
    }
}

/// A digit for a palette index. Indices above 9 cannot be named by a frame
/// character, and every caller here has already clamped.
fn digit(i: usize) -> char {
    char::from_digit(i as u32, 10).unwrap_or(CLEAR)
}

/// Draw one frame.
///
/// Pure in `r`: the same recipe always gives the same frame, which is what makes
/// a seed worth writing into a comment.
pub fn generate(r: &Recipe) -> Frame {
    let (w, h) = (r.w.max(1), r.h.max(1));
    let mut rng = Rng::seeded(r.seed);

    // The region actually noised. The other half is a copy, so noising all of it
    // would spend draws on texels about to be overwritten — and, worse, would
    // make the result depend on which half won.
    let (nw, nh) = match r.symmetry {
        Symmetry::MirrorX => (w.div_ceil(2), h),
        Symmetry::MirrorY => (w, h.div_ceil(2)),
        Symmetry::None => (w, h),
    };

    let mut cells = vec![vec![false; nw]; nh];
    for row in &mut cells {
        for cell in row.iter_mut() {
            *cell = rng.chance(r.density);
        }
    }
    for _ in 0..SMOOTH_PASSES {
        cells = smooth(&cells);
    }
    keep_largest(&mut cells);
    touch_seam(&mut cells, r.symmetry);

    // Out to the full grid, mirroring as it goes.
    let body = digit(r.body());
    let mut frame = vec![vec![CLEAR; w]; h];
    for (y, row) in cells.iter().enumerate() {
        for (x, &on) in row.iter().enumerate() {
            if !on {
                continue;
            }
            frame[y][x] = body;
            match r.symmetry {
                Symmetry::MirrorX => frame[y][w - 1 - x] = body,
                Symmetry::MirrorY => frame[h - 1 - y][x] = body,
                Symmetry::None => {}
            }
        }
    }

    let frame: Frame = frame.into_iter().map(|r| r.into_iter().collect()).collect();
    let frame = ops::auto_shade(&frame, digit(r.lit()), digit(r.shade()), (-1, -1));
    if r.outline {
        ops::outline(&frame, '1')
    } else {
        frame
    }
}

/// A palette the shading indices land inside.
///
/// Index 1 is the darkest step and is what [`generate`] outlines with; the rest
/// run up to the lightest, which is what it rim-lights with. So the two have to
/// be generated together or the drawing names colours the palette does not have.
pub fn generate_pal(r: &Recipe, base: Hsl) -> Vec<String> {
    palette::ramp(
        base,
        r.inks.clamp(2, palette::MAX_STEPS),
        // Five degrees a step, measured off `palette::REFERENCE` — see the
        // default on the panel. This was 14, which walked a four-step ramp
        // across forty-two degrees and gave every generated creature a hue
        // gradient no hand-authored record in the tree has.
        GENERATED_HUE_SHIFT,
        // The reference's ramps run roughly 0.15 to 0.85 in lightness and stop
        // short of both black and white, which is what leaves room for an
        // outline underneath and a specular above.
        (0.14, 0.86),
    )
}

/// Degrees of hue a generated ramp turns per step. See [`generate_pal`].
const GENERATED_HUE_SHIFT: f32 = 5.0;

/// How many cellular passes the noise gets.
///
/// One leaves speckle; three has usually eaten the interesting detail along with
/// it. Two is where the shapes stop changing much between passes.
const SMOOTH_PASSES: usize = 2;

/// How many filled neighbours a texel needs to be filled after a pass.
///
/// Four of eight — the majority rule. Three grows everything into a blob and
/// five erodes everything to nothing.
const SMOOTH_THRESHOLD: usize = 4;

/// One cellular pass.
///
/// # The bottom edge is ground, the other three are air
///
/// Out-of-bounds counts as empty, which is not a detail: it means the edges
/// erode and the drawing pulls away from the border of its own accord. A
/// creature touching all four edges of an 8x8 grid reads as a texture, not as a
/// body.
///
/// **Directly below the grid is the exception, and it is worth one neighbour.**
/// Treating it as empty like the rest eroded the feet away: the generator
/// produced a row-7 occupancy of 0.04 against the reference sheet's 0.46, so its
/// creatures were hovering one texel above a floor they were meant to stand on.
/// Below the bottom row is the ground, and ground supports.
///
/// The weight is one neighbour rather than three because it was fitted, not
/// guessed. Sweeping the generator against the reference's mean row occupancy —
/// its 54 characters, bottom-aligned — gives:
///
/// ```text
///   row          0     1     2     3     4     5     6     7
///   reference   .21   .51   .58   .65   .77   .76   .67   .46
///   here        .13   .47   .73   .72   .72   .73   .65   .56
/// ```
///
/// Counting all three cells below as ground overfilled the bottom row; counting
/// none left it bare. One is what tracks the sheet.
///
/// The same sweep is where [`Recipe::default`]'s density comes from, and it is
/// also what killed a per-row noise bias that had looked obviously necessary:
/// with this rule in place the bias earned nothing measurable, so it is not
/// here. Standing on the ground turns out to be the whole of what made the
/// silhouettes bottom-heavy.
fn smooth(cells: &[Vec<bool>]) -> Vec<Vec<bool>> {
    let h = cells.len();
    let w = cells.first().map_or(0, Vec::len);
    let mut out = vec![vec![false; w]; h];
    for (y, out_row) in out.iter_mut().enumerate() {
        for (x, cell) in out_row.iter_mut().enumerate() {
            let mut n = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    let inside = nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h;
                    if inside && cells[ny as usize][nx as usize] {
                        n += 1;
                    } else if ny >= h as i32 && dx == 0 {
                        // Directly below the bottom row: ground, and ground
                        // supports. Only straight down — see the header for the
                        // sweep that says three is too many.
                        n += 1;
                    }
                }
            }
            *cell = n >= SMOOTH_THRESHOLD;
        }
    }
    out
}

/// Slide the drawing over until it reaches the mirror line.
///
/// # Why this is needed at all
///
/// [`keep_largest`] guarantees the half-grid is one piece, and mirroring a
/// connected shape gives two connected shapes — which is two shapes, not one, if
/// the region never touches the seam. A blob sitting in the outer corner of the
/// half becomes a matched pair of blobs with a gap down the middle, and that
/// reads as two creatures rather than one.
///
/// A translate rather than a grow: the shape that survived the smoothing is the
/// shape worth keeping, and stretching it to the seam would undo the erosion
/// that made it a shape. Nothing can fall off the far edge, because the shift is
/// exactly the distance from the region's near edge to the seam.
///
/// [`Symmetry::None`] has no seam and is left alone.
fn touch_seam(cells: &mut [Vec<bool>], symmetry: Symmetry) {
    let h = cells.len();
    let w = cells.first().map_or(0, Vec::len);
    if w == 0 || h == 0 {
        return;
    }
    match symmetry {
        // The seam is the last column: the half is mirrored about its right edge.
        Symmetry::MirrorX => {
            let Some(max_x) = (0..w).rev().find(|&x| (0..h).any(|y| cells[y][x])) else {
                return;
            };
            let by = w - 1 - max_x;
            if by == 0 {
                return;
            }
            for row in cells.iter_mut() {
                row.rotate_right(by);
                row[..by].fill(false);
            }
        }
        // The seam is the last row.
        Symmetry::MirrorY => {
            let Some(max_y) = (0..h).rev().find(|&y| cells[y].iter().any(|&c| c)) else {
                return;
            };
            let by = h - 1 - max_y;
            if by == 0 {
                return;
            }
            cells.rotate_right(by);
            for row in cells.iter_mut().take(by) {
                row.fill(false);
            }
        }
        Symmetry::None => {}
    }
}

/// Clear everything except the biggest 4-connected region.
///
/// Four-connected to match [`ops::outline`] and [`ops::flood`]: a one-texel
/// diagonal reads as a gap at this size, so two blobs joined only at a corner
/// are two blobs.
fn keep_largest(cells: &mut [Vec<bool>]) {
    let h = cells.len();
    let w = cells.first().map_or(0, Vec::len);
    let mut seen = vec![vec![false; w]; h];
    let mut best: Vec<(usize, usize)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if !cells[y][x] || seen[y][x] {
                continue;
            }
            let mut region = Vec::new();
            let mut stack = vec![(x, y)];
            seen[y][x] = true;
            while let Some((x, y)) = stack.pop() {
                region.push((x, y));
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx < w && ny < h && cells[ny][nx] && !seen[ny][nx] {
                        seen[ny][nx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            if region.len() > best.len() {
                best = region;
            }
        }
    }

    for row in cells.iter_mut() {
        row.fill(false);
    }
    for (x, y) in best {
        cells[y][x] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster;
    use std::collections::BTreeSet;

    #[test]
    fn a_seed_is_the_whole_input() {
        // The claim the seed field on the panel is making. If this were false,
        // writing a seed into a record's comment would be a lie.
        for seed in [0u32, 1, 2, 99, 65_536, u32::MAX] {
            let r = Recipe {
                seed,
                ..Recipe::default()
            };
            assert_eq!(generate(&r), generate(&r), "seed {seed} was not the input");
        }
    }

    #[test]
    fn every_roll_is_the_shape_it_was_asked_for() {
        // A generated frame goes straight into a record that already declares
        // its `cellsW` and `cellsH`. One row short is a panic at `PreStartup`.
        for seed in 0..256u32 {
            for (w, h) in [(8usize, 8usize), (8, 10), (4, 4), (6, 6), (1, 1)] {
                let f = generate(&Recipe {
                    seed,
                    w,
                    h,
                    ..Recipe::default()
                });
                assert_eq!(f.len(), h, "seed {seed} at {w}x{h}");
                for row in &f {
                    assert_eq!(row.chars().count(), w, "seed {seed} at {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn two_seeds_are_two_drawings() {
        // A generator that collapsed to a handful of outputs would be a reroll
        // button that does nothing, which is the failure mode a determinism test
        // alone would happily pass.
        //
        // The bar is not 1024, and the gap is the pipeline working rather than
        // failing. At the default recipe this measures 691 distinct drawings;
        // dropping the mirror takes it to exactly 1024 and dropping the
        // smoothing would too, because both of those are what turn 32 bits of
        // noise into a shape. Symmetry halves the grid that carries the entropy
        // and the cellular passes pull neighbouring rolls onto the same
        // attractor, deliberately. `touch_seam` costs 45 of the 1024 and buys
        // connectedness, which is the better side of that trade.
        //
        // 600 is the floor rather than the measurement, so retuning the passes
        // or the threshold has room to move without editing a test — but a
        // change that halved the variety would be caught.
        let seen: BTreeSet<Frame> = (0..1024u32)
            .map(|seed| {
                generate(&Recipe {
                    seed,
                    ..Recipe::default()
                })
            })
            .collect();
        assert!(
            seen.len() > 600,
            "1024 seeds gave only {} distinct drawings",
            seen.len()
        );
    }

    #[test]
    fn a_mirrored_roll_crosses_its_own_midline() {
        // What `touch_seam` buys, stated as the property rather than as the
        // mechanism: a shape that stopped short of the seam would be mirrored
        // into a matched PAIR of shapes with a gap between them, which reads as
        // two creatures. `a_roll_is_one_piece` covers the same ground from the
        // other side; this one names the cause.
        for seed in 0..128u32 {
            let f = generate(&Recipe {
                seed,
                outline: false,
                symmetry: Symmetry::MirrorX,
                ..Recipe::default()
            });
            let w = f[0].chars().count();
            let crosses = f.iter().any(|row| {
                let cs: Vec<char> = row.chars().collect();
                !ops::is_clear(cs[w / 2 - 1]) && !ops::is_clear(cs[w / 2])
            });
            let empty = f.iter().all(|row| row.chars().all(ops::is_clear));
            assert!(crosses || empty, "seed {seed} left a gap down the middle");
        }
    }

    #[test]
    fn a_mirrored_roll_really_is_symmetric() {
        // The cue the whole pipeline is built around. Shading breaks it — the
        // light comes from one side — so the symmetry is asserted on the step
        // that establishes it, with shading and outline off.
        for seed in 0..64u32 {
            let f = generate(&Recipe {
                seed,
                inks: 2,
                outline: false,
                symmetry: Symmetry::MirrorX,
                ..Recipe::default()
            });
            let filled: Vec<Vec<bool>> = f
                .iter()
                .map(|r| r.chars().map(|c| !ops::is_clear(c)).collect())
                .collect();
            for row in &filled {
                let mut back = row.clone();
                back.reverse();
                assert_eq!(*row, back, "seed {seed} is not left/right symmetric");
            }
        }
    }

    #[test]
    fn a_roll_is_one_piece() {
        // Islands read as dirt on the screen. `keep_largest` is what removes
        // them, and this is what says it ran.
        for seed in 0..128u32 {
            let f = generate(&Recipe {
                seed,
                outline: false,
                ..Recipe::default()
            });
            let g: Vec<Vec<bool>> = f
                .iter()
                .map(|r| r.chars().map(|c| !ops::is_clear(c)).collect())
                .collect();
            let (h, w) = (g.len(), g[0].len());

            let Some(start) = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .find(|&(x, y)| g[y][x])
            else {
                continue; // an empty roll is allowed; it is not disconnected
            };

            let mut seen = vec![vec![false; w]; h];
            let mut stack = vec![start];
            seen[start.1][start.0] = true;
            let mut reached = 0;
            while let Some((x, y)) = stack.pop() {
                reached += 1;
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx < w && ny < h && g[ny][nx] && !seen[ny][nx] {
                        seen[ny][nx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            let total: usize = g.iter().flatten().filter(|&&b| b).count();
            assert_eq!(reached, total, "seed {seed} left an island");
        }
    }

    #[test]
    fn a_roll_only_names_indices_its_palette_has() {
        // `generate` and `generate_pal` are two halves of one thing: the frame
        // writes digits and the palette is what those digits mean. A frame that
        // named index 5 of a four-colour ramp draws magenta.
        for inks in 2..=palette::MAX_STEPS {
            for outline in [true, false] {
                for seed in 0..64u32 {
                    let r = Recipe {
                        seed,
                        inks,
                        outline,
                        ..Recipe::default()
                    };
                    let pal = generate_pal(&r, Hsl::new(30.0, 0.6, 0.5));
                    let f = generate(&r);
                    for row in &f {
                        for ch in row.chars() {
                            if ops::is_clear(ch) {
                                continue;
                            }
                            let i = ch.to_digit(10).expect("a frame is digits") as usize;
                            assert!(
                                i < pal.len(),
                                "seed {seed} inks {inks} named index {i} of a {}-entry palette",
                                pal.len()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_generated_frame_rasterises() {
        // The end-to-end claim: what comes out of here goes through the same
        // rasteriser the canvas draws with and the game bakes with.
        for seed in 0..128u32 {
            let r = Recipe {
                seed,
                ..Recipe::default()
            };
            let pal = generate_pal(&r, Hsl::new(200.0, 0.55, 0.5));
            let colours = raster::palette(&pal).expect("a generated palette is colours");
            let f = generate(&r);
            let px = raster::raster(&f, &colours, r.w as u32, r.h as u32)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            assert_eq!(px.len(), r.w * r.h * 4);
        }
    }

    #[test]
    fn an_outlined_roll_reserves_index_one_for_the_outline() {
        // The body and the rim must not be drawn in the outline's colour, or the
        // outline stops being visible against the thing it outlines.
        for seed in 0..64u32 {
            let r = Recipe {
                seed,
                outline: true,
                inks: 4,
                ..Recipe::default()
            };
            assert!(r.body() >= 2 && r.shade() >= 2 && r.lit() >= 2);
            let _ = generate(&r);
        }
    }

    /// The reference sheet's mean row occupancy, top to bottom. See [`smooth`].
    const REFERENCE_ROWS: [f32; 8] = [0.21, 0.51, 0.58, 0.65, 0.77, 0.76, 0.67, 0.46];

    /// What a default roll actually fills, averaged over `n` seeds.
    fn measured_rows(n: u32) -> ([f32; 8], f32) {
        let mut rows = [0.0f32; 8];
        let mut ink = 0usize;
        for seed in 0..n {
            let f = generate(&Recipe {
                seed,
                outline: false,
                ..Recipe::default()
            });
            for (y, row) in f.iter().enumerate() {
                let c = row.chars().filter(|&c| !ops::is_clear(c)).count();
                rows[y] += c as f32 / 8.0;
                ink += c;
            }
        }
        for r in &mut rows {
            *r /= n as f32;
        }
        (rows, ink as f32 / (n as f32 * 64.0))
    }

    #[test]
    fn a_roll_fills_about_as_much_of_its_canvas_as_the_reference_does() {
        // The generator used to fill a quarter of the 8x8 while the reference
        // sheet's characters fill two thirds of theirs, which made every roll
        // read as a small thing in a large box rather than as a creature. The
        // band is wide because this is a calibration, not a golden — it exists
        // to catch a change that halves or doubles the coverage.
        let (_, fill) = measured_rows(512);
        assert!(
            (0.45..=0.72).contains(&fill),
            "a default roll fills {fill:.2} of its canvas; the reference fills 0.62"
        );
    }

    #[test]
    fn a_roll_stands_on_the_ground_rather_than_floating_above_it() {
        // The property the ground-support rule in `smooth` exists for, and the
        // one a symmetric blob generator cannot have. The reference's bottom row
        // carries more than twice its top row; before the rule this generator
        // had that backwards-to-flat, with row 7 at 0.04 against row 0 at 0.04.
        let (rows, _) = measured_rows(512);
        assert!(
            rows[7] > rows[0] * 2.0,
            "bottom row {:.2} against top row {:.2} — the silhouette is not standing",
            rows[7],
            rows[0]
        );
        // And the widest part is in the middle, not at an edge: a shape that
        // peaks at row 0 or row 7 is a wedge, not a body.
        let widest = (0..8)
            .max_by(|&a, &b| rows[a].partial_cmp(&rows[b]).expect("no NaN"))
            .expect("eight rows");
        assert!(
            (2..=5).contains(&widest),
            "widest row is {widest}; the reference peaks at 4-5"
        );
    }

    #[test]
    fn no_row_is_wildly_off_the_reference() {
        // The whole profile at once, loosely. Any single row more than 0.25 out
        // means the shape has drifted into something the sheet would not
        // recognise — which is the failure the two tests above would each miss
        // on their own.
        let (rows, _) = measured_rows(512);
        for y in 0..8 {
            let d = (rows[y] - REFERENCE_ROWS[y]).abs();
            assert!(
                d < 0.25,
                "row {y} is {:.2} against the reference's {:.2}",
                rows[y],
                REFERENCE_ROWS[y]
            );
        }
    }

    #[test]
    fn density_moves_how_much_is_drawn() {
        // The knob has to do something monotone, or it is not a knob.
        let ink = |density: f32| -> usize {
            (0..64u32)
                .map(|seed| {
                    generate(&Recipe {
                        seed,
                        density,
                        outline: false,
                        ..Recipe::default()
                    })
                    .iter()
                    .flat_map(|r| r.chars())
                    .filter(|&c| !ops::is_clear(c))
                    .count()
                })
                .sum()
        };
        let (sparse, dense) = (ink(0.35), ink(0.70));
        assert!(sparse < dense, "density did nothing: {sparse} vs {dense}");
    }
}
