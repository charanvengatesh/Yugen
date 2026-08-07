//! The cell rasteriser's golden baseline: the blit must still produce, pixel for
//! pixel, what it produced the day it was locked down.
//!
//! # How `tests/cells.golden.json` was produced
//!
//! `tools/dump-cells-fixture.mjs` in the TypeScript repo copies
//! `src/render/ChunkCanvas.ts` verbatim, turns `const TEX_A` / `const TEX_B` /
//! `const SHADE32` into `export const` on the copy (they are module-private in
//! the original and this test wants them checked at their source), bundles the
//! copy with esbuild, and runs it under node. It fills two real `CellGrid`
//! windows — 352 x 256 cells each, 88 chunks apiece — from the TypeScript
//! worldgen at seed 2334, one straddling the surface and one deep in the lava
//! band, then dumps:
//!
//!   - both pattern tiles, per-pattern hashed plus row 0 verbatim,
//!   - the whole shade table, per-material hashed plus 848 sampled entries,
//!   - the same after `updateShimmer(t)` at six points on the animation clock,
//!   - twelve `paintCells` results, hashed per row and whole, with the emitter
//!     census each one published, and two small ones dumped verbatim.
//!
//! **That provenance is history now**: the port is over and this is no longer an
//! authority on TypeScript, it is this project's own baseline. See
//! `registry_golden.rs` for the full argument and the rule.
//!
//! # What had to change here so the game could grow a block
//!
//! The shade table is `MAT_COUNT * 512` words, so it lengthens every time a
//! material is authored. `perMaterial` walks only the materials the baseline
//! names, so it became a prefix walk for free — but the whole-table hash beside
//! it did not, and that single line was one of the three things that made adding
//! a workbench impossible. It now hashes the baselined slice.
//!
//! Everything else is untouched, and that is not luck: the pattern tiles are not
//! a function of the registry, and the twelve `paint_cells` results are a
//! function of what WORLDGEN placed, not of what exists. A new block that nothing
//! generates changes nothing here. A new block that worldgen starts placing
//! changes these AND `worldgen_golden`, in agreement, which is the pair of
//! failures that means "the world moved" rather than "the rasteriser broke".
//!
//! # Why the tables are checked separately from the pixels
//!
//! `TEX_A`, `TEX_B` and `SHADE32` are the two inputs every pixel is a lookup
//! into. If a box-blur radius, a `Math.round` tie or a pack byte order is wrong,
//! every one of the 200 000 cells below is wrong with it, and the report would
//! be a wall of failed row hashes that says nothing about the cause. Checking
//! the tables first means the failure is reported where it happens: "pattern 4
//! of TEX_B differs" is a bug you can go and find.
//!
//! # Why a full grid rather than a synthetic one
//!
//! Every interesting branch in the blit is a property of the MATERIAL
//! DISTRIBUTION: the run hoist only fires on long runs, `depth_above` only
//! saturates under a thick mass, `side_open` only fires at a cave wall, and the
//! emitter census only fires on lava and torches. A checkerboard would exercise
//! none of them. The two windows here carry 18 distinct materials each, run the
//! census to its 512 cap in the depths and leave it empty at the surface, and
//! sit at NEGATIVE grid origins on both axes so every `ox - origin_x` in the
//! blit is a real subtraction.

use std::collections::BTreeMap;

use godgame_core::config::CHUNK_CELLS;
use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::materials::CellId;
use godgame_core::sim::worldgen::ChunkGen;
use godgame_render::cells::{CellShades, TEX_A, TEX_B, paint_cells};
use serde_json::Value as J;

fn fixture() -> J {
    serde_json::from_str(include_str!("cells.golden.json"))
        .expect("cells.golden.json is not valid JSON")
}

/// FNV-1a, 32 bit, over the little-endian bytes of the values.
///
/// The same three lines on both sides, which is the point: a checksum whose
/// implementation could itself differ between the two languages would turn a
/// parity failure into a debugging session about the checksum.
fn fnv(bytes: &[u8]) -> String {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    format!("{h:08x}")
}

fn fnv_u32(vals: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fnv(&bytes)
}

fn fnv_u16(vals: &[u16]) -> String {
    let mut bytes = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fnv(&bytes)
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Decode one hex-encoded IEEE754 double.
///
/// The fixture carries bit patterns rather than decimal literals for the same
/// reason `noise.golden.json` does: a decimal round-trip only preserves a value if
/// both parsers are correctly rounded to the last bit, and `serde_json`'s is
/// not. The shimmer clock `t` is multiplied by 325.9 and then truncated to a
/// table index, so a last-bit difference in `t` is a visible difference in the
/// palette. Bits are the value.
fn f64_bits(v: &J) -> f64 {
    f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
}

fn i(v: &J) -> i64 {
    v.as_i64().unwrap()
}

/// Rebuild one of the fixture's grid windows: `chunk_cols` x `chunk_rows`
/// chunks from the Rust worldgen at the fixture's seed, with the window origin
/// set to the same absolute cell as the TypeScript's.
fn build_grid(spec: &J, seed: u32) -> CellGrid {
    let cx0 = i(&spec["cx0"]) as i32;
    let cy0 = i(&spec["cy0"]) as i32;
    let chunk_cols = i(&spec["chunkCols"]) as i32;
    let chunk_rows = i(&spec["chunkRows"]) as i32;
    let cols = chunk_cols * CHUNK_CELLS;
    let rows = chunk_rows * CHUNK_CELLS;

    let mut grid = CellGrid::new(cols, rows);
    grid.set_origin(cx0 * CHUNK_CELLS, cy0 * CHUNK_CELLS);

    let mut cg = ChunkGen::new(seed);
    for j in 0..chunk_rows {
        for k in 0..chunk_cols {
            let chunk = cg.generate(cx0 + k, cy0 + j);
            let base_x = k * CHUNK_CELLS;
            let base_y = j * CHUNK_CELLS;
            for ly in 0..CHUNK_CELLS {
                let src = (ly * CHUNK_CELLS) as usize;
                let dst = ((base_y + ly) * cols + base_x) as usize;
                grid.material[dst..dst + CHUNK_CELLS as usize]
                    .copy_from_slice(&chunk[src..src + CHUNK_CELLS as usize]);
            }
        }
    }

    // The grid is the shared premise of every case below. If worldgen has
    // drifted, the pixel failures that follow would be blamed on the blit.
    assert_eq!(
        fnv_u16(&grid.material),
        spec["materialHash"].as_str().unwrap(),
        "the worldgen window differs from the TypeScript's before a single \
         pixel is painted — fix worldgen parity first"
    );
    assert_eq!(grid.origin_cell_x(), i(&spec["originCellX"]) as i32);
    assert_eq!(grid.origin_cell_y(), i(&spec["originCellY"]) as i32);
    grid
}

fn grids(f: &J) -> BTreeMap<String, CellGrid> {
    let seed = f["meta"]["seed"].as_u64().unwrap() as u32;
    f["grids"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, spec)| (name.clone(), build_grid(spec, seed)))
        .collect()
}

/// Read the fixture's per-material hashes and sampled entries against a shade
/// table. Shared by the at-rest check and every shimmer snapshot.
fn check_shade_table(
    what: &str,
    rec: &J,
    table: &[u32],
    stride: usize,
    problems: &mut Vec<String>,
) {
    let per: Vec<&str> = rec["perMaterial"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for (id, want) in per.iter().enumerate() {
        let got = fnv_u32(&table[id * stride..(id + 1) * stride]);
        if got != *want {
            problems.push(format!("{what}: material {id} slice {got} != ts {want}"));
        }
    }
    for s in rec["samples"].as_array().unwrap() {
        let a = s.as_array().unwrap();
        let (id, e, p, want) = (
            i(&a[0]) as usize,
            i(&a[1]) as usize,
            i(&a[2]) as usize,
            i(&a[3]) as u32,
        );
        let got = table[(id << 9) | (e << 6) | p];
        if got != want {
            problems.push(format!(
                "{what}: SHADE32[mat {id}, edge {e}, pat {p}] = {got:#010x} != ts {want:#010x}"
            ));
        }
    }
    // The BASELINED slice, not the whole table.
    //
    // `perMaterial` above already walks only the materials the baseline knows
    // about, so it grew a prefix for free; this line did not, and it was the one
    // thing in this file that made adding a block impossible. A new material
    // appends `stride` words past `per.len() * stride` and cannot move anything
    // below it — the per-material hashes are what prove that, one material at a
    // time. This hash is the whole-slice version of the same claim and has to be
    // cut to the same length or it is a different claim.
    let pinned = per.len() * stride;
    let all = fnv_u32(&table[..pinned]);
    if all != rec["all"].as_str().unwrap() {
        problems.push(format!(
            "{what}: the baselined {pinned} words hash {all} != {}",
            rec["all"].as_str().unwrap()
        ));
    }
}

/// The two coprime pattern tiles.
///
/// Checked first and on their own, because they are the deepest thing in the
/// file: eight patterns each built from wrapping box blurs, integer-harmonic
/// sines, plateau quantisation and a variance normalisation, and every pixel in
/// the game is a lookup into two of them. A per-pattern hash localises a failure
/// to one `match` arm; row 0 verbatim gives that arm something to diff.
#[test]
fn both_pattern_tiles_match_the_typescript_original() {
    let f = fixture();
    let mut problems: Vec<String> = Vec::new();

    for (key, tile, p) in [
        ("a", &**TEX_A, i(&f["meta"]["pa"]) as usize),
        ("b", &**TEX_B, i(&f["meta"]["pb"]) as usize),
    ] {
        let rec = &f["tex"][key];
        assert_eq!(i(&rec["p"]) as usize, p);
        let n = p * p;
        let per: Vec<&str> = rec["perPattern"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let row0: Vec<&str> = rec["row0"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for t in 0..per.len() {
            let slab = &tile[t * n..t * n + n];
            let got = fnv(slab);
            if got != per[t] {
                let got_row = hex(&slab[..p]);
                problems.push(format!(
                    "TEX_{}: pattern {t} hashes {got} != ts {}\n    rust row 0 {got_row}\n    ts   row 0 {}",
                    key.to_uppercase(),
                    per[t],
                    row0[t]
                ));
            }
        }
        let all = fnv(tile);
        if all != rec["all"].as_str().unwrap() {
            problems.push(format!(
                "TEX_{}: whole tile {all} != ts {}",
                key.to_uppercase(),
                rec["all"].as_str().unwrap()
            ));
        }
    }

    assert!(problems.is_empty(), "{}", problems.join("\n  "));
}

/// The 53 x 8 x 64 shade table at rest, and after each of the six shimmer clock
/// samples.
///
/// The shimmer snapshots are applied in the fixture's own order and to the same
/// table, because `update_shimmer` is a stateless function of `t` — if it were
/// not, this test would find that too.
#[test]
fn the_shade_table_matches_the_typescript_at_rest_and_animated() {
    let f = fixture();
    let stride = i(&f["meta"]["shadeStride"]) as usize;
    let mut problems: Vec<String> = Vec::new();

    let mut shades = CellShades::new();
    check_shade_table(
        "at rest",
        &f["shadeAtRest"],
        shades.table(),
        stride,
        &mut problems,
    );

    for snap in f["shimmer"].as_array().unwrap() {
        let t = f64_bits(&snap["t"]);
        shades.update_shimmer(t);
        check_shade_table(
            &format!("shimmer t={t}"),
            snap,
            shades.table(),
            stride,
            &mut problems,
        );
    }

    assert!(
        problems.is_empty(),
        "{} shade-table divergences:\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}

/// Every frozen `paintCells` result, row by row, plus the emitter census each
/// one published.
///
/// The at-rest cases are painted before any `update_shimmer` call, exactly as
/// the fixture generator did — the TypeScript rewrites `SHADE32` in place and
/// has no way back to the resting palette, so the order is part of the fixture.
#[test]
fn every_painted_viewport_matches_the_typescript_pixel_for_pixel() {
    let f = fixture();
    let grids = grids(&f);
    let mut shades = CellShades::new();
    let mut problems: Vec<String> = Vec::new();
    let mut cells_checked = 0usize;

    for case in f["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let grid = &grids[case["grid"].as_str().unwrap()];
        let w = i(&case["w"]) as i32;
        let h = i(&case["h"]) as i32;
        let (ox, oy) = (i(&case["ox"]) as i32, i(&case["oy"]) as i32);

        if !case["shimmer"].is_null() {
            shades.update_shimmer(f64_bits(&case["shimmer"]));
        }

        let mut px = vec![0u32; (w * h) as usize];
        let emit = paint_cells(&mut px, w, h, grid, ox, oy, &shades);
        cells_checked += px.len();

        let rows: Vec<&str> = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(rows.len(), h as usize, "{name}: fixture row count");
        let mut bad_rows = Vec::new();
        for y in 0..h as usize {
            let got = fnv_u32(&px[y * w as usize..(y + 1) * w as usize]);
            if got != rows[y] {
                bad_rows.push(y);
            }
        }
        if !bad_rows.is_empty() {
            problems.push(format!(
                "{name}: {} of {h} rows differ (first: {:?})",
                bad_rows.len(),
                &bad_rows[..bad_rows.len().min(8)]
            ));
        }

        let all = fnv_u32(&px);
        if all != case["all"].as_str().unwrap() {
            problems.push(format!(
                "{name}: whole buffer {all} != ts {}",
                case["all"].as_str().unwrap()
            ));
        }

        // The verbatim cases. A hash says "wrong"; these say "wrong how".
        if let Some(raw) = case["raw"].as_str() {
            let mut bytes = Vec::with_capacity(px.len() * 4);
            for v in &px {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let got = hex(&bytes);
            if got != raw {
                let at = got
                    .bytes()
                    .zip(raw.bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                problems.push(format!(
                    "{name}: verbatim buffer differs at nibble {at} (cell {})\n    rust {}\n    ts   {}",
                    at / 8,
                    &got[at.saturating_sub(16)..(at + 16).min(got.len())],
                    &raw[at.saturating_sub(16)..(at + 16).min(raw.len())],
                ));
            }
        }

        // The census. Order matters: it is produced by the scan order of the
        // blit, and a light pass that truncates it would take the first N.
        let want_n = i(&case["emit"]["n"]) as usize;
        let want_x: Vec<i32> = case["emit"]["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| i(v) as i32)
            .collect();
        let want_y: Vec<i32> = case["emit"]["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| i(v) as i32)
            .collect();
        if emit.len() != want_n {
            problems.push(format!("{name}: emitters {} != ts {want_n}", emit.len()));
        } else if emit.x() != want_x || emit.y() != want_y {
            let at = emit
                .x()
                .iter()
                .zip(emit.y())
                .zip(want_x.iter().zip(&want_y))
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            problems.push(format!(
                "{name}: emitter {at} is ({}, {}), ts says ({}, {})",
                emit.x()[at],
                emit.y()[at],
                want_x[at],
                want_y[at]
            ));
        }
    }

    assert!(cells_checked > 150_000, "the sweep shrank: {cells_checked}");
    assert!(
        problems.is_empty(),
        "{} viewports differ from the TypeScript blit:\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}

/// The premise the whole fixture rests on: the Rust worldgen still produces the
/// windows the TypeScript painted. Broken out so a worldgen regression is
/// reported as one line rather than as twelve failed viewports.
#[test]
fn the_fixture_grids_still_generate() {
    let f = fixture();
    let g = grids(&f);
    assert_eq!(g.len(), 2);
    for (name, spec) in f["grids"].as_object().unwrap() {
        let grid = &g[name];
        let distinct = grid
            .material
            .iter()
            .collect::<std::collections::BTreeSet<&CellId>>()
            .len();
        assert_eq!(
            distinct,
            i(&spec["distinctMaterials"]) as usize,
            "{name}: material variety changed"
        );
    }
}
