//! Every region's authored materials actually reach the world.
//!
//! The gap this closes is narrow and was wide open. `biomes.rs` proves each
//! biome and layer is wired into the tables, that its weights are sane, and --
//! since the Badlands and Rime Hollows went in -- that it wins a fair share of
//! columns. `worldgen_purity` proves the generator is a pure function.
//! `worldgen_golden` proves it has not drifted.
//!
//! None of them looks at what a region is MADE OF. A biome whose `cap` and
//! `rock` are overwritten by the depth bands, the shore rule or a decorator pass
//! is authored, selected, blended, hashed and pure -- and invisible. It would
//! read in game as the biome next door, and the only thing that would ever have
//! caught it is somebody flying 35 000 cells out to look.
//!
//! So this generates the actual chunks at the actual surface of each biome's own
//! territory and asserts its two signature materials are in THE COLUMN, in the
//! band each belongs to.
//!
//! The first draft asserted only that each material appeared somewhere in the
//! two chunks. That is 2 048 cells of a whole neighbourhood, and it passed with
//! Badlands' cap set to LAVA -- there is lava down there anyway. A test that
//! survives having its subject replaced by molten rock is not testing anything.
//! Hence the column, and the bands.
//!
//! # How to check this test still bites
//!
//! Not by changing a biome's `cap` in `biomes.rs`. That is an INPUT to the
//! generator, so the world dutifully grows the new material and this test finds
//! it and passes -- correctly. Editing the content cannot fault-inject a test
//! about whether content is honoured.
//!
//! Break the pipeline instead. Replacing the `cap` return in
//! `worldgen/layers.rs` with a constant is the check that works: it fails here
//! with "desert: cap material 12 is nowhere in the 12 rows below its own
//! surface", which is exactly the class of bug the file exists for.

use std::collections::BTreeMap;
use yugen_core::sim::biomes::{BIOMES, Biome, biome_mix_at, column_profile_at};
use yugen_core::sim::materials::CellId;
use yugen_core::sim::noise::Noise;
use yugen_core::sim::worldgen::heightmap::Heightmap;
use yugen_core::sim::worldgen::{ChunkGen, world_noise};

/// The seed the rest of the suite uses, and the game's own.
const SEED: u32 = 2334;

/// How far out to look for a column a biome dominates.
///
/// Badlands is the rarest of the nine and its first column is 4 426 cells from
/// spawn, so this has to be generous. It is a search bound, not a claim about
/// the world: a biome not found inside it fails, which is the correct outcome
/// for content this hard to reach.
const SEARCH: i32 = 40_000;

/// Find a column this biome owns outright, searching outward from spawn.
///
/// Nothing but this biome, because an ecotone cap is DESIGNED to replace the
/// biome's own and sampling one would fail this test for a feature working
/// correctly.
///
/// The first draft asked for `second_w < 0.25` and picked column 4 457 for
/// Badlands, which is on the Badlands/Desert margin: the column caps in SAND
/// over SANDSTONE, both of them Desert's, arriving through exactly the ecotone
/// this repo authors for that pair. A quarter of a neighbour is not a little
/// bit of a neighbour -- it is a border. `items().len() == 1` is the only
/// honest reading of "heartland", so that is what this asks for.
fn heartland(noise: &Noise, b: Biome) -> Option<i32> {
    for d in 0..SEARCH {
        for x in [d, -d] {
            let m = biome_mix_at(noise, x);
            if m.top == b && m.items().len() == 1 {
                return Some(x);
            }
        }
    }
    None
}

#[test]
fn every_biome_puts_its_own_materials_on_the_ground() {
    let noise = world_noise(SEED);
    let mut hm = Heightmap::new();
    let mut cg = ChunkGen::new(SEED);

    for b in Biome::ALL {
        let def = &BIOMES[b.index()];
        // Volcanic does not compete on climate, so `biome_mix_at` is not how it
        // is placed and there is no heartland to find. `volcanic_is_rare_but_real`
        // in `biomes.rs` is its check.
        if def.climate.is_none() {
            continue;
        }
        let x = heartland(&noise, b)
            .unwrap_or_else(|| panic!("no column within {SEARCH} of spawn is solidly {}", def.id));

        let col = column_profile_at(&noise, x);
        let surf = hm.surface_row_at(&noise, x, Some(&col));

        // This column only, from the surface down, out of the chunk that
        // actually generated it -- decorators, shore rule and all.
        let (cx, lx) = (x.div_euclid(32), x.rem_euclid(32) as usize);
        let cy = surf.div_euclid(32);
        let mut column: Vec<CellId> = Vec::with_capacity(64);
        for ccy in cy..=cy + 1 {
            let cells = cg.generate(cx, ccy);
            for row in 0..32 {
                column.push(cells[row * 32 + lx]);
            }
        }
        // Rows of `column`, in world terms.
        let top = cy * 32;
        let at = |wy: i32| column[(wy - top) as usize];

        // The cap is a thin band starting at the surface row. CAP_BAND is
        // generous against `cap_scale`, which runs from 0.6 (badlands) to 1.6
        // (desert) -- the check is "the cap is at the top", not a thickness.
        const CAP_BAND: i32 = 12;
        let cap_hits = (surf..surf + CAP_BAND)
            .filter(|&y| at(y) == def.cap)
            .count();
        // Below the cap, the biome's rock signature.
        let rock_hits = (surf + CAP_BAND..top + 64)
            .filter(|&y| at(y) == def.rock)
            .count();

        let seen: BTreeMap<CellId, usize> = column.iter().fold(BTreeMap::new(), |mut m, &c| {
            *m.entry(c).or_insert(0) += 1;
            m
        });
        assert!(
            cap_hits > 0,
            "{}: cap material {} is nowhere in the {CAP_BAND} rows below its own \
             surface (column {x}, surface row {surf}). Column holds: {seen:?}",
            def.id,
            def.cap,
        );
        assert!(
            rock_hits > 0,
            "{}: rock material {} is nowhere under its own cap (column {x}, \
             surface row {surf}). Column holds: {seen:?}",
            def.id,
            def.rock,
        );
    }
}
