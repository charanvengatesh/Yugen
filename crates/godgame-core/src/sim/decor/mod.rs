//! Decoration passes — what gets scattered on generated terrain.
//!
//! Terrain is generated per column as a pure function of absolute coordinates.
//! Decorations are the things that are NOT columnar — a tree, an ore pocket, a
//! ruin — and they are the part of worldgen easiest to get catastrophically
//! wrong, so the rule is worth stating plainly:
//!
//! > A decoration is authored from a POSITIONAL ORIGIN, and every chunk it can
//! > possibly touch recomputes it independently and paints only its own share.
//!
//! That is what makes a tree straddling a chunk boundary agree with itself. The
//! chunk holding the trunk and the chunk holding the overhanging canopy never
//! talk; they each decide "is there a tree at column X?" from noise and position
//! alone and both get the same answer. The moment a decorator asks what a
//! neighbouring chunk contains, or keeps state between calls, chunks generated
//! in a different order disagree and you get seams that only appear when the
//! player approaches from an unusual direction — the worst class of bug to
//! chase.
//!
//! Concretely, every decorator must be:
//!   - deterministic in `(seed, absolute coords)` and nothing else,
//!   - free of neighbour reads and of state carried between chunks,
//!   - bounded by a declared reach, and it must scan candidate origins that far
//!     PAST each chunk edge so an overhanging neighbour still paints its share.
//!
//! Rust enforces two thirds of that for free. [`DecorContext`] hands out no way
//! to read another chunk, and a decorator takes `&self`, so it has nowhere to
//! keep state between calls even if it wanted to.

pub mod ores;
pub mod structures;
pub mod trees;

use crate::config::world::{CHUNK_CELLS, pmod};
use crate::sim::biomes::{ColumnProfile, column_profile_at};
use crate::sim::materials::{CellId, EMPTY, MAT_COLLIDE};
use crate::sim::noise::Noise;
use crate::sim::worldgen::heightmap::Heightmap;

/// Everything a decorator is allowed to know, and nothing else.
///
/// The absence of a "read a neighbouring chunk" method is the design. So is the
/// absence of a mutable PRNG: [`DecorContext::hash`] is positional, so every
/// chunk that recomputes a decoration draws the same number in the same place.
pub struct DecorContext<'a> {
    pub noise: &'a Noise,
    pub seed: u32,
    /// Absolute cell x of this chunk's top-left.
    pub base_x: i32,
    /// Absolute cell y of this chunk's top-left.
    pub base_y: i32,
    /// The chunk being generated, row-major, `CHUNK_CELLS` square.
    out: &'a mut [CellId],
    /// Surface rows come from here, never from a chunk's contents.
    heightmap: &'a mut Heightmap,
}

impl<'a> DecorContext<'a> {
    pub fn new(
        noise: &'a Noise,
        seed: u32,
        base_x: i32,
        base_y: i32,
        out: &'a mut [CellId],
        heightmap: &'a mut Heightmap,
    ) -> DecorContext<'a> {
        debug_assert_eq!(out.len(), (CHUNK_CELLS * CHUNK_CELLS) as usize);
        DecorContext {
            noise,
            seed,
            base_x,
            base_y,
            out,
            heightmap,
        }
    }

    /// Index of an absolute cell within this chunk, or `None` if it is outside.
    ///
    /// Returning `None` rather than clamping is the whole discard mechanism: a
    /// decoration paints every cell it covers and the ones that miss this chunk
    /// simply fall on the floor, to be painted by whichever chunk does own them.
    #[inline]
    fn index(&self, wcx: i32, wcy: i32) -> Option<usize> {
        let lx = wcx - self.base_x;
        let ly = wcy - self.base_y;
        if lx < 0 || ly < 0 || lx >= CHUNK_CELLS || ly >= CHUNK_CELLS {
            return None;
        }
        Some((ly * CHUNK_CELLS + lx) as usize)
    }

    /// Write a cell if it lands inside the chunk being generated; else discard.
    #[inline]
    pub fn plot(&mut self, wcx: i32, wcy: i32, code: CellId) {
        if let Some(i) = self.index(wcx, wcy) {
            self.out[i] = code;
        }
    }

    /// As [`DecorContext::plot`], but only over empty space — for canopy that
    /// must not eat rock.
    #[inline]
    pub fn plot_if_empty(&mut self, wcx: i32, wcy: i32, code: CellId) {
        if let Some(i) = self.index(wcx, wcy)
            && self.out[i] == EMPTY
        {
            self.out[i] = code;
        }
    }

    /// As [`DecorContext::plot`], but only over solid terrain — for ore
    /// replacing rock.
    #[inline]
    pub fn plot_if_solid(&mut self, wcx: i32, wcy: i32, code: CellId) {
        if let Some(i) = self.index(wcx, wcy)
            && MAT_COLLIDE[self.out[i] as usize] != 0
        {
            self.out[i] = code;
        }
    }

    /// What is currently in this cell? Only meaningful inside the chunk; a cell
    /// outside it reads as empty, because this chunk genuinely does not know.
    #[inline]
    pub fn peek(&self, wcx: i32, wcy: i32) -> CellId {
        self.index(wcx, wcy).map_or(EMPTY, |i| self.out[i])
    }

    /// Surface row for an absolute column — recomputed, never read from a chunk.
    #[inline]
    pub fn surface_at(&mut self, wcx: i32) -> i32 {
        // `None` for the profile: the decorator asked for a column, not for a
        // column it has already profiled, so let the memo do its job rather
        // than paying for a profile the caller does not have.
        self.heightmap.surface_row_at(self.noise, wcx, None)
    }

    /// Full climate/biome/layer profile for an absolute column.
    #[inline]
    pub fn profile_at(&self, wcx: i32) -> ColumnProfile {
        column_profile_at(self.noise, wcx)
    }

    /// Stable hash in [0,1) from any two integers.
    ///
    /// THE way to make a random decision inside a decorator: it depends only on
    /// position, so every chunk that recomputes this decoration draws the same
    /// number. There is deliberately no stateful alternative reachable from
    /// here — `Noise::rand` needs `&mut Noise`, which a decorator cannot get.
    #[inline]
    pub fn hash(&self, x: i32, y: i32) -> f64 {
        self.noise.hash2(x, y)
    }
}

/// A decoration pass over one chunk.
///
/// `&self` is load-bearing: a decorator has nowhere to keep state between
/// chunks, which is one of the three contract rules made unbreakable rather
/// than merely documented.
pub trait Decorator {
    /// For diagnostics only.
    fn name(&self) -> &'static str;

    /// Farthest a decoration may extend from its origin, in cells, horizontally.
    ///
    /// The generator scans candidate origins this far past every chunk edge, so
    /// an under-declared reach shows up as decorations clipped at chunk
    /// boundaries.
    fn reach_x(&self) -> i32;

    /// Farthest a decoration may extend from its origin, in cells, vertically.
    fn reach_y(&self) -> i32;

    /// Paint this chunk's share of every decoration overlapping it.
    fn decorate(&self, ctx: &mut DecorContext<'_>);
}

/// Every candidate origin column within reach of this chunk, ascending.
///
/// `stride`/`phase` thin candidates out positionally: origin columns satisfy
/// `wcx ≡ phase (mod stride)`. The scan starts at the first candidate at or
/// after the left edge rather than testing every column and rejecting most.
///
/// An iterator rather than the TypeScript's callback, because a callback taking
/// `&mut DecorContext` while the helper also borrowed it would not compile —
/// and an iterator is the shape that wanted to exist anyway.
pub fn origin_columns(
    base_x: i32,
    reach_x: i32,
    stride: i32,
    phase: i32,
) -> impl Iterator<Item = i32> {
    let from = base_x - reach_x;
    let to = base_x + CHUNK_CELLS + reach_x;
    let first = from + pmod(phase - from, stride);
    (0..)
        .map(move |k| first + k * stride)
        .take_while(move |&x| x < to)
}

/// Which origins a 2D decorator considers: a stride and a phase per axis.
///
/// A struct rather than four more positional arguments. Stride and phase always
/// travel together, and eight bare `i32`s in a row is a transposition bug
/// waiting to happen — swapping `stride_x` and `stride_y` compiles cleanly and
/// silently produces a different world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lattice {
    pub stride_x: i32,
    pub stride_y: i32,
    pub phase_x: i32,
    pub phase_y: i32,
}

/// As [`origin_columns`], but over a 2D lattice — for structures and pockets.
///
/// Row-major, y outer, matching the TypeScript so that two decorations
/// competing for the same cell resolve the same way.
pub fn origin_cells(
    base_x: i32,
    base_y: i32,
    reach_x: i32,
    reach_y: i32,
    lat: Lattice,
) -> impl Iterator<Item = (i32, i32)> {
    let Lattice {
        stride_x,
        stride_y,
        phase_x,
        phase_y,
    } = lat;
    let from_x = base_x - reach_x;
    let to_x = base_x + CHUNK_CELLS + reach_x;
    let from_y = base_y - reach_y;
    let to_y = base_y + CHUNK_CELLS + reach_y;
    let first_x = from_x + pmod(phase_x - from_x, stride_x);
    let first_y = from_y + pmod(phase_y - from_y, stride_y);
    (0..)
        .map(move |j| first_y + j * stride_y)
        .take_while(move |&y| y < to_y)
        .flat_map(move |y| {
            (0..)
                .map(move |i| first_x + i * stride_x)
                .take_while(move |&x| x < to_x)
                .map(move |x| (x, y))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_scans_are_phase_aligned_in_absolute_space() {
        // Two adjacent chunks must agree on which columns are candidates, or a
        // decoration straddling their edge is generated by one and not the
        // other.
        let (stride, phase, reach) = (3, 1, 8);
        let a: Vec<i32> = origin_columns(0, reach, stride, phase).collect();
        let b: Vec<i32> = origin_columns(CHUNK_CELLS, reach, stride, phase).collect();
        for x in &a {
            assert_eq!(
                pmod(*x, stride),
                pmod(phase, stride),
                "column {x} off-phase"
            );
        }
        // The overlap region is scanned identically by both.
        let overlap: Vec<i32> = a.iter().copied().filter(|x| b.contains(x)).collect();
        assert!(
            !overlap.is_empty(),
            "chunks that share a reach must share candidates"
        );
    }

    #[test]
    fn origin_scans_work_west_of_the_origin() {
        let (stride, phase, reach) = (7, 3, 11);
        for base in [-CHUNK_CELLS * 5, -CHUNK_CELLS, 0, CHUNK_CELLS * 5] {
            let cols: Vec<i32> = origin_columns(base, reach, stride, phase).collect();
            assert!(!cols.is_empty());
            for x in &cols {
                assert_eq!(pmod(*x, stride), pmod(phase, stride));
            }
            assert!(cols[0] >= base - reach);
            assert!(cols[0] - stride < base - reach);
            assert!(*cols.last().unwrap() < base + CHUNK_CELLS + reach);
        }
    }

    #[test]
    fn the_2d_scan_is_row_major_and_phase_aligned() {
        let cells: Vec<(i32, i32)> = origin_cells(
            0,
            0,
            4,
            4,
            Lattice {
                stride_x: 5,
                stride_y: 6,
                phase_x: 2,
                phase_y: 3,
            },
        )
        .collect();
        assert!(!cells.is_empty());
        for (x, y) in &cells {
            assert_eq!(pmod(*x, 5), pmod(2, 5));
            assert_eq!(pmod(*y, 6), pmod(3, 6));
        }
        // y is the outer loop: the sequence is non-decreasing in y.
        for w in cells.windows(2) {
            assert!(w[0].1 <= w[1].1, "scan is not row-major");
        }
    }

    #[test]
    fn plot_discards_everything_outside_the_chunk() {
        let noise = Noise::new(1);
        let mut hm = Heightmap::new();
        let mut out = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let mut ctx = DecorContext::new(&noise, 1, 100, 200, &mut out, &mut hm);

        ctx.plot(100, 200, 7); // top-left corner, inside
        ctx.plot(99, 200, 9); // one west, outside
        ctx.plot(100, 199, 9); // one north, outside
        ctx.plot(100 + CHUNK_CELLS, 200, 9); // one east, outside
        assert_eq!(ctx.peek(100, 200), 7);
        assert_eq!(
            out.iter().filter(|&&c| c == 9).count(),
            0,
            "wrote outside the chunk"
        );
    }
}
