//! Worldgen's golden baseline: the generator must still produce, cell for cell,
//! what it produced the day it was locked down.
//!
//! `tests/worldgen.golden.json` was produced by running the original
//! `src/sim/worldgen.ts` under node and hashing 357 whole chunks — every band
//! from two chunks above the surface anchor down past `UNDERWORLD_FLOOR`, across
//! 21 chunk columns — plus 7 820 `materialAt` probes and both spawn points.
//! **That provenance is history now**: the port is over and this is no longer an
//! authority on TypeScript, it is this project's own baseline. See
//! `registry_golden.rs` for the full argument and the rule.
//!
//! # This one WILL have to be blessed one day, and that is the interesting case
//!
//! The other four baselines are indifferent to new content. This is not: the
//! moment a new underground layer or surface biome lands, 357 chunks legitimately
//! change and every hash here goes red at once. That is not a failure, it is the
//! whole point of the file — it is telling you the world moved, and you have to
//! be the one to decide the world was *supposed* to move.
//!
//! When that day comes: bless it in its own commit, with nothing else in it, and
//! read the diff for chunks you did not expect to touch. A new deep layer that
//! quietly changed the beaches is exactly the bug this catches, and it can only
//! catch it if the bless is small enough to read. `registry_golden.rs` has the
//! `GODGAME_BLESS` pattern to copy.
//!
//! This is the check `noise_golden` cannot be. Noise parity says the
//! primitives agree; this says the whole pipeline does — heightmap, biome blend,
//! cave lattice, depth bands, cap and shore, veins and strata, and all four
//! decorator passes composed in order. Nothing else in the suite would notice a
//! divergence that is deterministic, seamless and order-independent, and simply
//! wrong: `worldgen_purity` proves the generator is a pure function, and a pure
//! function of the wrong thing passes every one of its checks.
//!
//! It has already earned its keep. The first draft of the orchestrator floored
//! the fractional BAND DEPTH before handing it to `layers`, on the correct
//! observation that every band test is `bd >= T` against an integer `T`. What
//! that missed is the one place `solid_at` reads the band depth CONTINUOUSLY —
//! the ramp hardening the underworld crust to obsidian over the last 45 cells of
//! the world. This fixture found the 67 cells, out of 226 304, where that showed.
//!
//! A per-chunk hash rather than the cells themselves: 357 chunks of `u16` is
//! 700 KiB of fixture, and a hash localises a failure to the chunk anyway, which
//! is as far as a diff of 1 024 cells would help.

use std::collections::BTreeMap;

use godgame_core::sim::biomes::column_profile_at;
use godgame_core::sim::materials::CellId;
use godgame_core::sim::worldgen::heightmap::Heightmap;
use godgame_core::sim::worldgen::{ChunkGen, SPAWN_COL, material_at, spawn_point, world_noise};
use serde_json::Value as J;

fn fixture() -> J {
    serde_json::from_str(include_str!("worldgen.golden.json"))
        .expect("worldgen.golden.json is not valid JSON")
}

/// FNV-1a, 32 bit, over the little-endian bytes of the cell codes.
///
/// The same three lines on both sides, which is the point: a checksum whose
/// implementation could itself differ between the two languages would turn a
/// parity failure into a debugging session about the checksum.
fn fnv(cells: &[CellId]) -> String {
    let mut h: u32 = 0x811c_9dc5;
    for &v in cells {
        h = (h ^ u32::from(v & 0xff)).wrapping_mul(0x0100_0193);
        h = (h ^ u32::from(v >> 8)).wrapping_mul(0x0100_0193);
    }
    format!("{h:08x}")
}

fn ints(v: &J) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[test]
fn every_chunk_matches_the_typescript_cell_for_cell() {
    let f = fixture();
    let seed = f["seed"].as_u64().unwrap() as u32;
    let cys = ints(&f["cy"]);
    let cxs = ints(&f["cx"]);
    let want: BTreeMap<&str, &str> = f["chunks"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
        .collect();

    let mut cg = ChunkGen::new(seed);
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0;
    for cx in cxs[0] as i32..=cxs[1] as i32 {
        for &cy in &cys {
            let cy = cy as i32;
            let key = format!("{cx},{cy}");
            let got = fnv(&cg.generate(cx, cy));
            let expect = want
                .get(key.as_str())
                .unwrap_or_else(|| panic!("fixture has no chunk {key}"));
            checked += 1;
            if got != *expect {
                bad.push(format!("({cx},{cy}) rust {got} != ts {expect}"));
            }
        }
    }

    assert_eq!(
        checked,
        want.len(),
        "the sweep and the fixture disagree in size"
    );
    assert!(
        bad.is_empty(),
        "{} of {checked} chunks differ from the TypeScript generator:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

#[test]
fn the_arbitrary_coordinate_probe_matches_the_typescript() {
    let f = fixture();
    let seed = f["seed"].as_u64().unwrap() as u32;
    let noise = world_noise(seed);
    let mut hm = Heightmap::new();

    // The same lattice of probes the fixture was dumped over: a column stride
    // that is coprime with everything in the generator, and a row stride that
    // walks from sky to bedrock.
    let mut probes: Vec<CellId> = Vec::new();
    let mut wcx = -400;
    while wcx <= 400 {
        let col = column_profile_at(&noise, wcx);
        let surf = hm.surface_row_at(&noise, wcx, Some(&col));
        let mut wcy = -40;
        while wcy <= 700 {
            probes.push(material_at(&noise, wcx, wcy, &col, surf));
            wcy += 11;
        }
        wcx += 7;
    }

    assert_eq!(probes.len() as u64, f["probes"]["count"].as_u64().unwrap());
    assert_eq!(
        fnv(&probes),
        f["probes"]["hash"].as_str().unwrap(),
        "material_at has drifted from the TypeScript materialAt"
    );
}

#[test]
fn spawn_lands_where_the_typescript_put_it() {
    let f = fixture();
    let seed = f["seed"].as_u64().unwrap() as u32;
    for (key, col) in [("spawn", SPAWN_COL), ("spawn100", 100)] {
        let want = ints(&f[key]);
        let got = spawn_point(seed, col);
        assert_eq!(
            (got.x as i64, got.y as i64),
            (want[0], want[1]),
            "spawn_point(seed, {col}) moved"
        );
    }
}
