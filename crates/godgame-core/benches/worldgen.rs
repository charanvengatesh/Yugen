//! Worldgen microbenchmarks.
//!
//! Chunk generation is the one worldgen cost the player can feel: it runs on the
//! streaming path while they are walking, and a slow chunk shows up as a hitch at
//! the edge of the window, not as a lower frame rate. So it needs a number, and
//! the number needs to be broken out by DEPTH BAND — a sky chunk and an
//! underworld chunk do completely different amounts of work, and an average over
//! a random spread hides a regression in either one.
//!
//! `cave_stats` exists for tuning rather than timing: cave thresholds are the
//! kind of constant that is impossible to eyeball ("is 0.80 too open?") and
//! trivial to measure. Open fraction by band is the number the tuning comments
//! cite. It and `terrain_stats` print once, before the timings.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use godgame_core::config::{CAVERN_DEPTH, CHUNK_CELLS, DEEP_DEPTH, SEA_LEVEL_Y};
use godgame_core::sim::biomes::column_profile_at;
use godgame_core::sim::materials::{CellId, block};
use godgame_core::sim::worldgen::heightmap::Heightmap;
use godgame_core::sim::worldgen::{ChunkGen, world_noise};

/// Fixed seed so a bench run is comparable against the previous one. Named
/// `BENCH_SEED` rather than `SEED` so it cannot shadow the world seed in
/// `config::worldgen`, which is a different number with a different job.
const BENCH_SEED: u32 = 1337;

/// Codes that count as "open but wet" when measuring cave volume.
const LIQUID_CODES: [CellId; 4] = [block::WATER, block::LAVA, block::ACID, block::OIL];

fn is_liquid(m: CellId) -> bool {
    LIQUID_CODES.contains(&m)
}

/// Time chunk generation over chunk rows `[cy0, cy1)`.
///
/// Touches a result so the generation cannot be optimised away, and strides the
/// columns so the surface memo and the noise permutation cache see realistic
/// (not artificially perfect) locality.
///
/// The TypeScript ran 400 throwaway generations first, because V8 needs a few
/// hundred calls before the inner loops are optimised and measuring the
/// interpreter is measuring nothing. There is no interpreter here; criterion's
/// own warm-up phase covers the cache effects that remain.
fn band(c: &mut Criterion, name: &str, cy0: i32, cy1: i32) {
    let span = i64::from((cy1 - cy0).max(1));
    // ONE generator across the whole band, as the streaming path would hold: the
    // heightmap memo and the noise permutation table are built once per world,
    // not once per chunk, and a bench that rebuilt them per iteration would be
    // timing `Noise::new`.
    let mut cg = ChunkGen::new(BENCH_SEED);
    let mut i: i64 = 0;
    c.bench_function(name, |b| {
        b.iter(|| {
            let cx = (i % 64) as i32 - 32;
            let cy = cy0 + ((i / 64) % span) as i32;
            i += 1;
            let chunk = cg.generate(cx, cy);
            black_box(u32::from(chunk[0]) + u32::from(chunk[chunk.len() >> 1]))
        });
    });
}

/// Measure the open fraction of the underground by band, over a wide sample of
/// columns. This is the tuning instrument for `TUN_T0` / `CHEESE_T0`: at cavern
/// depth, ~10–15% open reads as a cave system you can travel through, under ~5%
/// reads as solid rock with occasional pockets, and over ~25% reads as rubble.
fn print_cave_stats(chunk_span_x: i32) {
    let mut cg = ChunkGen::new(BENCH_SEED);
    let noise = world_noise(BENCH_SEED);
    let mut hm = Heightmap::new();

    println!("cave stats (seed {BENCH_SEED}):");
    for (name, cy0, cy1) in [("cavern", 3, 6), ("deep", 8, 14), ("underworld", 17, 21)] {
        let mut open = 0u64;
        let mut liquid = 0u64;
        let mut cells = 0u64;
        for cx in -chunk_span_x / 2..chunk_span_x / 2 {
            for cy in cy0..cy1 {
                let chunk = cg.generate(cx, cy);
                let base_x = cx * CHUNK_CELLS;
                let base_y = cy * CHUNK_CELLS;
                for lx in 0..CHUNK_CELLS {
                    let surf = hm.surface_row_at(&noise, base_x + lx, None);
                    for ly in 0..CHUNK_CELLS {
                        if base_y + ly <= surf {
                            continue; // sky/sea is not "cave"
                        }
                        cells += 1;
                        let m = chunk[(ly * CHUNK_CELLS + lx) as usize];
                        if m == 0 {
                            open += 1;
                        } else if is_liquid(m) {
                            open += 1;
                            liquid += 1;
                        }
                    }
                }
            }
        }
        // Four decimals, not two: these percentages are the cheapest available
        // proof that a change to the cave fields moved nothing, and at two
        // decimals a drift of a few hundred cells in a hundred thousand rounds
        // away.
        let d = cells.max(1) as f64;
        println!(
            "  {name:<11} open {:>7.4}% ({open})  liquid {:>7.4}% ({liquid})  over {cells} cells",
            open as f64 / d * 100.0,
            liquid as f64 / d * 100.0
        );
    }
}

/// Height-profile summary for a stretch of world: how much of it is ocean, how
/// much is shore, and the spread of the ground line. The check that the
/// continental spline is doing its job — a world that is 60% ocean or 0% ocean
/// means the spline's waterline crossing is in the wrong place.
fn print_terrain_stats(columns: i32) {
    let noise = world_noise(BENCH_SEED);
    let mut hm = Heightmap::new();
    let mut ocean = 0i64;
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut sum = 0i64;
    for i in 0..columns {
        let wcx = i - (columns >> 1);
        let col = column_profile_at(&noise, wcx);
        let s = hm.surface_row_at(&noise, wcx, Some(&col));
        if s > SEA_LEVEL_Y {
            ocean += 1;
        }
        min = min.min(s);
        max = max.max(s);
        sum += i64::from(s);
    }
    println!(
        "terrain stats (seed {BENCH_SEED}, {columns} columns): ocean {:.1}%  rows {min}..{max}  \
         mean {:.1}  cavern depth {CAVERN_DEPTH}  deep depth {DEEP_DEPTH}",
        ocean as f64 / f64::from(columns) * 100.0,
        sum as f64 / f64::from(columns)
    );
}

fn worldgen(c: &mut Criterion) {
    print_terrain_stats(20_000);
    print_cave_stats(24);

    // The depth bands the world actually has.
    band(c, "sky (above surface)", -3, 0);
    band(c, "surface", 0, 3);
    band(c, "cavern", 3, 6);
    band(c, "deep", 8, 14);
    band(c, "underworld", 17, 24);
    band(c, "MIXED (0..24)", 0, 24);
}

criterion_group!(benches, worldgen);
criterion_main!(benches);
