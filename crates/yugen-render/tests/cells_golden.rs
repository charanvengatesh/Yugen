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

use serde_json::Value as J;
use yugen_core::config::CHUNK_CELLS;
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, MAT_COUNT};
use yugen_core::sim::worldgen::ChunkGen;
use yugen_render::cells::{CellShades, TEX_A, TEX_B, paint_cells};

fn fixture() -> J {
    serde_json::from_str(include_str!("cells.golden.json"))
        .expect("cells.golden.json is not valid JSON")
}

/// Where the baseline lives, for the blesser to write back to.
fn fixture_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cells.golden.json")
}

/// Whether this run should rewrite the baseline instead of comparing.
///
/// `YUGEN_BLESS=1`, the same switch `registry_golden.rs` and
/// `worldgen_golden.rs` use.
fn blessing() -> bool {
    std::env::var_os("YUGEN_BLESS").is_some_and(|v| v != "0" && !v.is_empty()) || world_blessing()
}

/// Whether this bless may absorb a LEGITIMATE worldgen move.
///
/// `YUGEN_BLESS_WORLD=1` (implies a bless). The ordinary bless treats a moved
/// worldgen window as a failure to report, because for the whole life of this
/// fixture the windows were the TypeScript's and nothing was allowed to move
/// them. The day worldgen legitimately changes — a new cave shape, a new biome —
/// that stance inverts: the windows, the emitter census and the coverage masks
/// all move as a CONSEQUENCE, and refusing to re-derive them would make worldgen
/// unchangeable forever.
///
/// The mode still refuses unless `worldgen.golden.json` has ALSO changed in the
/// working tree. The two fixtures are this file's header calls "in agreement":
/// a worldgen move that shows up here but not in the 357-chunk baseline is not a
/// worldgen move, it is a bug in one of the two, and no bless may pick a winner.
fn world_blessing() -> bool {
    std::env::var_os("YUGEN_BLESS_WORLD").is_some_and(|v| v != "0" && !v.is_empty())
}

/// Rewrite the palette-derived fields of the baseline from the live build.
///
/// # Why this exists, and what it costs
///
/// Before it, the appearance of every material below the baselined prefix could
/// not be changed at all: an art edit reddened two tests here with no supported
/// way to move them, and hand-editing 164 KB of minified JSON is not a way.
///
/// The cost is real and permanent. The shade table and the twelve painted
/// viewports were pinned to an INDEPENDENT implementation — the TypeScript, via a
/// `tools/dump-cells-fixture.mjs` that no longer exists and cannot be rebuilt.
/// After a bless they are re-derived from `yugen_render::cells` itself and
/// assert only "unchanged since the last bless"; `paint_cells` becomes pinned
/// against itself. **The last commit at which the pixel baseline was TypeScript-
/// authoritative is the one before this function landed.** `git log` it if you
/// ever need to know what the original actually painted.
///
/// What survives undiminished, and is why this file is still worth its size:
///
///   - `TEX_A` and `TEX_B`. Never blessed, still TypeScript-authored, and every
///     pixel in the game is a lookup into two of them.
///   - Both worldgen windows: `materialHash`, `distinctMaterials`, `histogram`.
///   - **The emitter census.** Order-sensitive, 512-capped in the depths, empty
///     at the surface. That is the blit's CONTROL FLOW, and no palette change can
///     reach it — which is also the strongest argument for leaving `lightEmit`
///     alone through an art pass.
///   - `cover`, the per-case coverage mask: see the guard list below.
///   - The sweep shape — negative origins on both axes, the fully-outside
///     viewport, 848 sample points, six shimmer clock samples.
///
/// # It is a CHECKED bless
///
/// It regenerates exactly what a palette can move and refuses to write if
/// anything else did. A bless that quietly absorbed a worldgen drift, a pattern
/// tile change or a moved emitter would launder the very failures the file
/// exists to report.
///
/// # The fixture is one line, and that decides the writer
///
/// 163 996 bytes, no newlines. `worldgen.golden.json` is pretty-printed at one
/// space and its blesser matches that; doing the same here would reformat all
/// 164 KB on the first bless and bury the handful of values that moved. So:
/// compact, and no trailing newline. Confirmed by round-tripping the file
/// through a compact serialiser and diffing it against itself, byte for byte.
///
/// Because the diff is therefore unreadable by construction, this prints a
/// REPORT instead, and that is what belongs in the commit message. The
/// human-readable record of an art change is `registry_golden`'s diff, which is
/// pretty-printed with one table value per line.
fn bless() {
    let world = world_blessing();
    let mut root = fixture();
    let stride = i(&root["meta"]["shadeStride"]) as usize;
    let mut refuse: Vec<String> = Vec::new();
    let mut report: Vec<String> = Vec::new();

    // ---- The world-bless precondition: the OTHER fixture moved too.
    //
    // Checked against git rather than against the file's content, because the
    // question is not "does worldgen.golden.json exist" but "did the same tree
    // that is asking me to absorb a worldgen move also re-derive the 357-chunk
    // baseline". A worldgen change that only one of the two fixtures sees is a
    // bug in one of them, and this bless refuses to pick a winner.
    if world {
        let repo_diff = std::process::Command::new("git")
            .args(["diff", "--name-only", "HEAD"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("git runs");
        let changed = String::from_utf8_lossy(&repo_diff.stdout);
        assert!(
            changed.lines().any(|l| l.ends_with("worldgen.golden.json")),
            "REFUSING YUGEN_BLESS_WORLD: worldgen.golden.json is unchanged in \
             this tree. A world-bless re-derives this fixture's windows, census \
             and coverage masks on the premise that worldgen moved DELIBERATELY \
             — and the proof of that is a re-blessed worldgen.golden.json in the \
             same tree (YUGEN_BLESS=1 cargo test -p yugen-core --test \
             worldgen_golden). Bless that first, read its diff, then run this."
        );
    }

    // ---- Guards. Everything a palette CANNOT move, re-checked before writing.
    //
    // `build_grid` already asserts both origins and the material hash, so calling
    // `grids` is itself the worldgen guard; it panics rather than returning here.
    // Under a world-bless the hash premise is the thing being re-derived, so the
    // guard inverts: the windows are rebuilt unchecked and their derived fields
    // rewritten below.
    let grids = grids_expecting(&root, !world);
    if world {
        for (name, spec) in root["grids"].as_object_mut().unwrap() {
            let grid = &grids[name.as_str()];
            let hash = fnv_u16(&grid.material);
            let moved = hash != spec["materialHash"].as_str().unwrap();
            let distinct = grid
                .material
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let mut histogram = serde_json::Map::new();
            let mut counts: BTreeMap<u16, u64> = BTreeMap::new();
            for &m in grid.material.iter() {
                *counts.entry(m).or_default() += 1;
            }
            for (m, n) in counts {
                histogram.insert(m.to_string(), J::from(n));
            }
            spec["materialHash"] = J::from(hash);
            spec["distinctMaterials"] = J::from(distinct as u64);
            spec["histogram"] = J::Object(histogram);
            report.push(format!(
                "grid {name}: window {} (distinct {distinct})",
                if moved {
                    "REWRITTEN — worldgen moved"
                } else {
                    "unchanged"
                },
            ));
        }
    }
    if MAT_COUNT < i(&root["meta"]["matCount"]) as usize {
        refuse.push(format!(
            "the registry SHRANK to {MAT_COUNT} materials, below the baselined \
             prefix of {} — that is not something a bless may absorb",
            i(&root["meta"]["matCount"]),
        ));
    }
    if !world {
        for (name, spec) in root["grids"].as_object().unwrap() {
            let distinct = grids[name]
                .material
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            if distinct != i(&spec["distinctMaterials"]) as usize {
                refuse.push(format!(
                    "grid {name}: {distinct} distinct materials, baseline says {}",
                    i(&spec["distinctMaterials"]),
                ));
            }
        }
    }
    for (key, tile) in [("a", &**TEX_A), ("b", &**TEX_B)] {
        if fnv(tile) != root["tex"][key]["all"].as_str().unwrap() {
            refuse.push(format!(
                "TEX_{}: the pattern tile moved. The tiles are not a function of \
                 the registry, so this is a change to the tile generator and this \
                 bless has no business hiding it",
                key.to_uppercase(),
            ));
        }
    }

    // ---- Regenerate. Three families, mutated IN PLACE.
    //
    // In place, and never by rebuilding the root, so that every field this
    // function does not name is not merely written back identically — it is not
    // re-serialised from live data at all, and cannot drift through the blesser.

    // The at-rest table, then the six shimmer snapshots against the same table in
    // the fixture's own order, exactly as `the_shade_table_...` walks them.
    let mut shades = CellShades::new();
    let moved = rewrite_shade_record(&mut root["shadeAtRest"], shades.table(), stride);
    report.push(format!("shadeAtRest: {moved} material slices moved"));
    for k in 0..root["shimmer"].as_array().unwrap().len() {
        let t = f64_bits(&root["shimmer"][k]["t"]);
        shades.update_shimmer(t);
        let moved = rewrite_shade_record(&mut root["shimmer"][k], shades.table(), stride);
        report.push(format!("shimmer t={t}: {moved} material slices moved"));
    }

    // The twelve viewports, from a FRESH table — the painting test builds its own
    // `CellShades` and the at-rest cases are painted before any shimmer call.
    let mut shades = CellShades::new();
    for k in 0..root["cases"].as_array().unwrap().len() {
        let case = &root["cases"][k];
        let name = case["name"].as_str().unwrap().to_string();
        let grid = &grids[case["grid"].as_str().unwrap()];
        let (w, h) = (i(&case["w"]) as i32, i(&case["h"]) as i32);
        let (ox, oy) = (i(&case["ox"]) as i32, i(&case["oy"]) as i32);
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
        let had_raw = case["raw"].is_string();
        if !case["shimmer"].is_null() {
            shades.update_shimmer(f64_bits(&case["shimmer"]));
        }

        let mut px = vec![0u32; (w * h) as usize];
        let emit = paint_cells(&mut px, w, h, grid, ox, oy, &shades);

        // The census is a guard, not an output: it is a function of WHICH cells
        // emit light, which no colour can change. WORLDGEN can — different rock
        // means different emitters in the window — so the world-bless re-derives
        // it as a consequence of the windows it just rewrote.
        let census_moved = emit.len() != want_n || emit.x() != want_x || emit.y() != want_y;
        if census_moved && !world {
            refuse.push(format!(
                "{name}: the emitter census moved ({} emitters, baseline {want_n}). \
                 A palette cannot do that — check `lightEmit` and `emissive`",
                emit.len(),
            ));
        }
        if world {
            root["cases"][k]["emit"]["n"] = J::from(emit.len() as u64);
            root["cases"][k]["emit"]["x"] =
                J::Array(emit.x().iter().map(|&v| J::from(v)).collect());
            root["cases"][k]["emit"]["y"] =
                J::Array(emit.y().iter().map(|&v| J::from(v)).collect());
            if census_moved {
                report.push(format!(
                    "{name}: emitter census REWRITTEN ({want_n} -> {} emitters)",
                    emit.len(),
                ));
            }
        }
        // The coverage mask is SEEDED ONCE and is a guard forever after.
        //
        // Not "regenerated when it changes" — written only when the field is
        // absent. That asymmetry is the whole value of it: once a `cover` exists
        // in the fixture, no bless can ever overwrite it, so the one invariant
        // that survives a palette change cannot be laundered by the machine that
        // launders the others. Seeding it while the row hashes still matched the
        // TypeScript is what gives it its authority — the buffers it was derived
        // from were provably the TypeScript's.
        let cover = cover_hash(&px);
        match root["cases"][k]["cover"].as_str() {
            None => {
                root["cases"][k]["cover"] = J::from(cover);
                report.push(format!("{name}: cover SEEDED"));
            }
            Some(want) if cover != want && !world => refuse.push(format!(
                "{name}: the coverage mask moved. That is window GEOMETRY — \
                 clipping, row spans, the negative-origin subtractions — and no \
                 palette can reach it, so this is a blit change and the bless has \
                 no business hiding it"
            )),
            // The one writer `cover` has ever had besides its seeding. The mask
            // is opacity, opacity is material, and material is worldgen's — so a
            // deliberate worldgen move REWRITES it, and the authority the seeded
            // mask inherited from the TypeScript buffers ends at the first
            // world-bless. That end is recorded here rather than papered over:
            // from that bless on, `cover` pins the blit's geometry against the
            // LAST DELIBERATE worldgen, which is the strongest claim still
            // available once the world itself is allowed to change.
            Some(want) if cover != want => {
                root["cases"][k]["cover"] = J::from(cover);
                report.push(format!(
                    "{name}: cover REWRITTEN — worldgen moved the opaque set"
                ));
            }
            Some(_) => {}
        }

        let mut changed_rows = 0;
        let mut rows: Vec<J> = Vec::with_capacity(h as usize);
        for y in 0..h as usize {
            let got = fnv_u32(&px[y * w as usize..(y + 1) * w as usize]);
            if got != root["cases"][k]["rows"][y].as_str().unwrap() {
                changed_rows += 1;
            }
            rows.push(J::from(got));
        }
        let all = fnv_u32(&px);
        let all_moved = all != root["cases"][k]["all"].as_str().unwrap();
        root["cases"][k]["rows"] = J::Array(rows);
        root["cases"][k]["all"] = J::from(all);
        if had_raw {
            let mut bytes = Vec::with_capacity(px.len() * 4);
            for v in &px {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            root["cases"][k]["raw"] = J::from(hex(&bytes));
        }
        report.push(format!(
            "{name}: {changed_rows}/{h} rows moved, whole-buffer {}",
            if all_moved { "moved" } else { "unchanged" },
        ));
    }

    assert!(
        refuse.is_empty(),
        "REFUSING TO BLESS — {} thing(s) moved that a palette cannot move:\n  {}",
        refuse.len(),
        refuse.join("\n  "),
    );

    // Compact, and no trailing newline. See the header.
    let out = serde_json::to_string(&root).expect("serialises");
    std::fs::write(fixture_path(), out.as_bytes()).expect("the baseline is writable");

    // The count of coverage masks is reported rather than assumed, because the
    // guard SKIPS a case that has no `cover` field. Claiming "every coverage mask
    // held" when the fixture carries none would be the report lying about the
    // strongest invariant in the file.
    let cases = root["cases"].as_array().unwrap();
    let covers = cases.iter().filter(|c| c["cover"].is_string()).count();
    println!(
        "BLESSED {}\n  {}",
        fixture_path().display(),
        report.join("\n  ")
    );
    println!(
        "  invariants HELD: emitter census in all {} cases, both pattern tiles, \
         both grid hashes, {covers} of {} coverage masks",
        cases.len(),
        cases.len(),
    );
}

/// Overwrite one shade record's `perMaterial`, `samples` and `all` from a live
/// table. Returns how many material slices actually moved, for the report.
///
/// The prefix length comes from the FIXTURE, never from `MAT_COUNT`. A bless that
/// derived it from the live registry would silently widen the baseline from 53
/// materials to 56 the first time it ran — changing what the file claims without
/// changing a word of the prose that says what it claims.
///
/// Deliberately NOT `check_shade_table` with a flag. That function is the thing
/// under test; a blesser sharing it could only ever agree with itself, and the
/// one bug neither would catch is the one they share.
fn rewrite_shade_record(rec: &mut J, table: &[u32], stride: usize) -> usize {
    let n = rec["perMaterial"].as_array().unwrap().len();
    let mut moved = 0;
    let mut per: Vec<J> = Vec::with_capacity(n);
    for id in 0..n {
        let got = fnv_u32(&table[id * stride..(id + 1) * stride]);
        if got != rec["perMaterial"][id].as_str().unwrap() {
            moved += 1;
        }
        per.push(J::from(got));
    }
    rec["perMaterial"] = J::Array(per);

    for s in rec["samples"].as_array_mut().unwrap() {
        let a = s.as_array().unwrap();
        let (id, e, p) = (i(&a[0]) as usize, i(&a[1]) as usize, i(&a[2]) as usize);
        s[3] = J::from(table[(id << 9) | (e << 6) | p]);
    }

    rec["all"] = J::from(fnv_u32(&table[..n * stride]));
    moved
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

/// FNV over one bit per pixel: is it opaque?
///
/// A pixel is opaque exactly when its cell is not air, so this value is a pure
/// function of the material distribution and the window geometry — clipping, the
/// row spans, the negative-origin subtractions, the fully-outside case — and is
/// INVARIANT UNDER ANY PALETTE CHANGE. That is what makes it the one thing here
/// that a bless is not allowed to touch, and why it was frozen while the row
/// hashes still matched the TypeScript: the buffers it was derived from were
/// provably the TypeScript's, so it inherits that authority for the blit's
/// geometry and keeps it after the palette-derived fields have lost theirs.
fn cover_hash(px: &[u32]) -> String {
    let mut bits: Vec<u8> = Vec::with_capacity(px.len().div_ceil(8));
    let mut acc = 0u8;
    for (n, v) in px.iter().enumerate() {
        acc = (acc << 1) | u8::from(v >> 24 != 0);
        if n % 8 == 7 {
            bits.push(acc);
            acc = 0;
        }
    }
    if !px.len().is_multiple_of(8) {
        bits.push(acc << (8 - px.len() % 8));
    }
    fnv(&bits)
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
fn build_grid(spec: &J, seed: u32, expect_baselined_world: bool) -> CellGrid {
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
    // drifted, the pixel failures that follow would be blamed on the blit. The
    // ONE caller allowed to skip this is the world-bless, whose whole job is to
    // re-derive the fixture after a deliberate worldgen change; the window
    // GEOMETRY below is asserted unconditionally, because no worldgen change can
    // move where a window sits.
    if expect_baselined_world {
        assert_eq!(
            fnv_u16(&grid.material),
            spec["materialHash"].as_str().unwrap(),
            "the worldgen window differs from the baseline before a single \
             pixel is painted — fix worldgen first, or bless it deliberately \
             with YUGEN_BLESS_WORLD=1 alongside a worldgen.golden.json bless"
        );
    }
    assert_eq!(grid.origin_cell_x(), i(&spec["originCellX"]) as i32);
    assert_eq!(grid.origin_cell_y(), i(&spec["originCellY"]) as i32);
    grid
}

fn grids(f: &J) -> BTreeMap<String, CellGrid> {
    grids_expecting(f, true)
}

/// [`grids`], with the world-hash premise made explicit. `false` is for the
/// world-bless alone.
fn grids_expecting(f: &J, expect_baselined_world: bool) -> BTreeMap<String, CellGrid> {
    let seed = f["meta"]["seed"].as_u64().unwrap() as u32;
    f["grids"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, spec)| (name.clone(), build_grid(spec, seed, expect_baselined_world)))
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
    if blessing() {
        return;
    }
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
    if blessing() {
        return bless();
    }
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
    if blessing() {
        return;
    }
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

        // The coverage mask: which pixels are opaque, one bit each.
        //
        // The ONLY thing in this test that a repaint cannot move, and therefore
        // the only pixel-level claim here that a bless has never touched. It is a
        // pure function of `material == 0` and the window geometry, so it still
        // asserts — against the TypeScript, not against ourselves — that the
        // clipping, the row spans, the negative-origin subtractions and the
        // fully-outside case all resolve exactly where they used to.
        //
        // Which makes it the check that catches the class of bug the rest of this
        // test went blind to the day it gained a blesser: a blit that paints the
        // right colours into the wrong rectangle.
        let want_cover = case["cover"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: fixture has no cover mask"));
        let got_cover = cover_hash(&px);
        if got_cover != want_cover {
            problems.push(format!(
                "{name}: coverage mask {got_cover} != {want_cover} — the opaque \
                 pixels moved, which is geometry, not palette"
            ));
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
