//! Depth banding and rock selection — everything about WHAT a solid cell is made
//! of, once caves.rs has decided that it IS solid.
//!
//! Bands, measured downward from the column's own surface row:
//!
//! | depth | band |
//! |---|---|
//! | `< cap_thickness` | topsoil cap, dithered with the neighbouring biome, overridden by a sand/sandstone shore band near sea level |
//! | `< CAVERN_DEPTH` | cavern: rock crossfading in depth from the SURFACE biome's signature to the independent UNDERGROUND layer |
//! | `< DEEP_DEPTH` | deep: the underground layer's rock and vein palette |
//! | `< UNDERWORLD_DEPTH` | strata: the same rock, cut by near-horizontal bands of contrasting stone, so a dug shaft reads as geology |
//! | `< UNDERWORLD_FLOOR` | the underworld — ash and basalt over open lava |
//! | else | bedrock. The world has a bottom. |
//!
//! The old generator ended at "endless deep, all lava", which is a non-place: no
//! landmark, no reason to stop descending, and nothing to find. Giving the world
//! a floor turns the descent into a journey with an end, which is the whole point
//! of Terraria's underworld.

use super::fields::{clamp01, smooth_ramp};
use crate::config::{CAVERN_DEPTH, UNDERWORLD_DEPTH, UNDERWORLD_FLOOR, WorldScale};
use crate::sim::biomes::{Biome, ColumnProfile, UndergroundLayerId, pick_from_mix};
use crate::sim::materials::{CellId, block};
use crate::sim::noise::Noise;

// --- Material resolution -----------------------------------------------------
//
// The TypeScript resolved each of these through a preference list against the
// live material registry, because the registry was grown by other work and a
// missing id had to degrade to something that had always existed rather than
// throw at module load. That discipline has no job here: `block::*` is a
// compile-time constant emitted by `contentc` from `content/`, so a material
// that stopped existing is a build error naming the line that wanted it —
// strictly earlier and strictly louder than the runtime fallback was.

/// Open-air code, re-exported so worldgen has one source for it.
pub const AIR: CellId = 0;

const M_SAND: CellId = block::SAND;
const M_SANDSTONE: CellId = block::SANDSTONE;
const M_ASH: CellId = block::ASH;
const M_BASALT: CellId = block::BASALT;
const M_OBSIDIAN: CellId = block::OBSIDIAN;
const M_LAVA: CellId = block::LAVA;

/// The light strata band material.
///
/// The two strata band materials are chosen for CONTRAST against the layer rocks
/// (which are mostly grey stone or dark basalt) rather than for realism: a tan
/// band and a near-black band are legible at 5px cells from across the screen,
/// which is the only thing that makes strata worth generating. Both are inert
/// solids — a powder band would avalanche into the first cave that clipped it and
/// the geology would erase itself in the first second of simulation.
const STRATA_LIGHT: CellId = M_SANDSTONE;
/// The dark strata band material. See [`STRATA_LIGHT`].
const STRATA_DARK: CellId = M_OBSIDIAN;
/// Strata thresholds against `g2`, whose measured distribution is p85 = 0.31,
/// p90 = 0.39, p95 = 0.49 (it is nowhere near uniform on [-1,1]). These put ~11%
/// of deep rock in the light band and ~8% in the dark one — enough that a shaft
/// cuts through two or three bands on the way down, sparse enough that the bulk
/// rock still identifies the underground layer.
const STRATA_LIGHT_T: f64 = 0.38;
/// See [`STRATA_LIGHT_T`].
const STRATA_DARK_T: f64 = -0.42;

// --- Boundary dithering ------------------------------------------------------
// A blend weight `t` in [0, 0.5] says "this fraction of cells here should come
// from the OTHER biome". Turning that into pixels needs a per-cell value in
// [0,1] to threshold against. Two sources are mixed:
//
//   * a mid-frequency lattice noise, which clusters the swapped cells into
//     fingers and islands so the boundary reads as an interlock, and
//   * a stateless coordinate hash, which is uniform and ragged, so the fingers
//     get bitten edges instead of smooth noise contours (and which flattens the
//     bell-shaped noise distribution, so a weight of `t` really does swap about
//     `t` of the cells).
//
// Both are pure functions of (wcx, wcy) — no RNG, no neighbour lookups.

const CAP_DITHER_FX: f64 = 0.19;
const CAP_DITHER_FY: f64 = 0.31;
const CAP_DITHER_ANCHOR: f64 = 71.5;

/// Per-cell dither value in ~[0,1] for the topsoil cap interlock. The cap band is
/// the one place with no fBm value already in hand, so it pays for its own
/// lattice sample — and only when the column is actually near a boundary. Value
/// noise on purpose: this is dither, nobody reads the field itself, and it is the
/// cheapest thing in noise.rs.
#[inline]
fn cap_dither(noise: &Noise, wcx: i32, wcy: i32, scale: WorldScale) -> f64 {
    let nv = clamp01(
        0.5 + 0.85
            * noise.n2(
                scale.coord(wcx) * CAP_DITHER_FX + CAP_DITHER_ANCHOR,
                scale.coord(wcy) * CAP_DITHER_FY,
            ),
    );
    nv * 0.62 + noise.hash2(wcx, wcy) * 0.38
}

/// Underground dither, built from a noise value the caller ALREADY sampled (the
/// vein field, or the strata field). Costs one integer hash and no extra noise.
/// `salt` decorrelates the several independent choices that share one source, so
/// e.g. the surface/underground handover does not line up with which of two
/// underground layers a cell picks.
///
/// The reuse is not only cheaper — it is what the terrain was tuned against.
/// Taking a second independent sample here would decorrelate the handover from
/// the veins it is supposed to interleave with, and the boundary would read as
/// static rather than as strata.
#[inline]
fn reuse_dither(noise: &Noise, wcx: i32, wcy: i32, v: f64, salt: i32) -> f64 {
    clamp01(0.5 + v * 0.9) * 0.6 + noise.hash2(wcx + salt, wcy) * 0.4
}

// --- Cap ---------------------------------------------------------------------

/// Topsoil cap material. Each contributing biome's cap gets a share of the dither
/// range proportional to its climate weight, so the boundary breaks into fingers
/// and islands of both instead of a ruled line.
///
/// The ECOTONE — transitional ground authored for that specific adjacency — is
/// painted as a contour band hugging every interface in dither space. Its width
/// grows as the two climates converge and reaches zero well before either
/// biome's interior, so it reads as a seam of its own rather than a third region.
///
/// SHORE overrides everything: a column whose ground line is within `SHORE_BAND`
/// of sea level gets sand over sandstone regardless of biome, dithered in by
/// weight so the beach frays into the biome instead of ending on a ruled column.
/// Sand is capped at the top 3 cells — a deep sand bank would avalanche into the
/// first cave that clipped it and the beach would drain away in the first second
/// of simulation.
pub fn cap_at(
    noise: &Noise,
    wcx: i32,
    wcy: i32,
    depth: i32,
    col: &ColumnProfile,
    shore: f64,
    scale: WorldScale,
) -> CellId {
    // The sand/sandstone split is authored as a legacy-cell thickness.
    let ld = scale.depth(f64::from(depth));
    if shore > 0.0 {
        let d = noise.hash2(wcx + 5501, wcy);
        if d < shore {
            return if ld < 3.0 { M_SAND } else { M_SANDSTONE };
        }
    }

    let mix = &col.surf;
    if mix.len() == 1 {
        return mix.items()[0].def().cap;
    }

    let v = cap_dither(noise, wcx, wcy, scale);
    if let Some(eco) = col.eco_cap {
        let cum = mix.cum();
        // Split the budget across however many interfaces the mix has, so a
        // three-way junction does not turn entirely into ecotone.
        let half = ((col.surf_t - 0.18) * 0.42) / (cum.len() - 1) as f64;
        if half > 0.0 {
            for c in &cum[..cum.len() - 1] {
                if v > c - half && v < c + half {
                    return eco;
                }
            }
        }
    }
    pick_from_mix(mix, v).def().cap
}

// --- Layer / biome selection at a cell ---------------------------------------

/// Which of the blended underground layers owns this cell.
pub fn layer_at(
    noise: &Noise,
    wcx: i32,
    wcy: i32,
    col: &ColumnProfile,
    src: f64,
) -> UndergroundLayerId {
    if col.ug.len() == 1 {
        return col.ug.items()[0];
    }
    pick_from_mix(&col.ug, reuse_dither(noise, wcx, wcy, src, 9176))
}

/// Which of the blended surface biomes owns this cell's subsurface rock.
fn surf_biome_at(noise: &Noise, wcx: i32, wcy: i32, col: &ColumnProfile, src: f64) -> Biome {
    if col.surf.len() == 1 {
        return col.surf.items()[0];
    }
    pick_from_mix(&col.surf, reuse_dither(noise, wcx, wcy, src, 271))
}

/// Depth crossfade weight from the surface biome's signature to the layer's.
#[inline]
pub fn ug_fade_at(depth: i32, col: &ColumnProfile, scale: WorldScale) -> f64 {
    // `ug_fade_start`/`end` are legacy cells, like every other depth in the
    // profile, so the world depth crosses inward here.
    clamp01(
        (scale.depth(f64::from(depth)) - col.ug_fade_start) / (col.ug_fade_end - col.ug_fade_start),
    )
}

// --- Solid rock --------------------------------------------------------------

/// Vein field frequency — high enough to read as a seam, not a region.
const VEIN_FREQ: f64 = 0.2;

/// The vein / handover-dither source field. Two octaves of value noise at
/// [`VEIN_FREQ`], anchored well away from the origin. Named because four call
/// sites share it and every one of them also feeds it to [`reuse_dither`] — the
/// value has to be the same float in both uses or the reuse stops being a reuse.
#[inline]
fn vein_field(noise: &Noise, wcx: i32, wcy: i32, scale: WorldScale) -> f64 {
    noise.fbm2(
        scale.coord(wcx) * VEIN_FREQ - 50.0,
        scale.coord(wcy) * VEIN_FREQ - 50.0,
        2,
    )
}

/// Solid material for an underground cell. `u` is the surface->layer depth
/// crossfade ([`ug_fade_at`]); `strata` is the near-horizontal banding field,
/// which the caller supplies because in the chunk path it is a lattice bilerp
/// (`CaveLattice::strata_at`) and on the probe path an exact sample
/// (`strata_exact`). Only read below `DEEP_DEPTH`, so the caller may pass 0 above
/// that.
///
/// `depth` is the BAND depth — the cell's true depth plus the column's slow
/// `band_shift` — and it is FRACTIONAL for the same reason `caves` carves against
/// a fractional one: every band interface has to be an undulating surface rather
/// than a ruled line, and the two have to undulate together or every interface
/// shows as a double edge. Rounding it to a whole cell before the call is not
/// free: the hardening ramp below reads it continuously, and quantising the input
/// moves the obsidian dither by up to 1/45 over the last 45 cells of the world.
///
/// The arity is what it is: every one of these is an independent input the rock
/// choice reads, and bundling them into a struct would only move the same eight
/// values behind a name that explains none of them.
#[allow(clippy::too_many_arguments)]
pub fn solid_at(
    noise: &Noise,
    wcx: i32,
    wcy: i32,
    depth: f64,
    col: &ColumnProfile,
    u: f64,
    strata: f64,
    scale: WorldScale,
) -> CellId {
    if depth >= f64::from(UNDERWORLD_FLOOR) {
        return M_OBSIDIAN; // bedrock
    }

    if depth >= f64::from(UNDERWORLD_DEPTH) {
        // Underworld crust: ash flats over basalt, banded, hardening to obsidian
        // as the floor approaches so the last stretch reads as "you are near the
        // end".
        let to_floor = smooth_ramp(
            f64::from(UNDERWORLD_FLOOR - 45),
            f64::from(UNDERWORLD_FLOOR),
            depth,
        );
        if to_floor > 0.0 && noise.hash2(wcx + 2207, wcy) < to_floor {
            return M_OBSIDIAN;
        }
        if strata > STRATA_LIGHT_T {
            return M_ASH;
        }
        return M_BASALT;
    }

    if depth < f64::from(CAVERN_DEPTH) {
        // Cavern: crossfade the rock signature in DEPTH from the surface biome's
        // to the underground layer's, reusing the vein value as the dither source
        // so the handover reads as interleaved strata rather than static.
        //
        // The unblended cases are checked BEFORE the vein field is sampled: away
        // from a biome boundary and outside the handover window there is exactly
        // one answer, and that is the majority of the world's cavern cells.
        // Sampling the vein field first (what the old generator did) spent two
        // noise evaluations per cell to dither between one option and itself.
        let surf_single = col.surf.len() == 1;
        let ug_single = col.ug.len() == 1;
        if u <= 0.002 && surf_single {
            return col.surf.items()[0].def().rock;
        }
        if u >= 0.998 && ug_single {
            return col.ug.items()[0].def().rock;
        }

        let v = vein_field(noise, wcx, wcy, scale);
        if u <= 0.002 {
            return surf_biome_at(noise, wcx, wcy, col, v).def().rock;
        }
        if u >= 0.998 {
            return layer_at(noise, wcx, wcy, col, v).def().rock;
        }
        return if reuse_dither(noise, wcx, wcy, v, 4099) < u {
            layer_at(noise, wcx, wcy, col, v).def().rock
        } else {
            surf_biome_at(noise, wcx, wcy, col, v).def().rock
        };
    }

    let vein = vein_field(noise, wcx, wcy, scale);
    let layer = layer_at(noise, wcx, wcy, col, vein).def();

    // Veins first: a seam of crystal or obsidian should cut THROUGH a stratum, the
    // way an intrusion cuts through sedimentary rock, not be interrupted by it.
    if vein > 0.8 {
        return layer.vein_rich;
    }
    if vein > 0.64 {
        return layer.vein_common;
    }
    if vein < -0.64 {
        return layer.vein_soft;
    }

    // Strata: near-horizontal bands of contrasting rock. Only below DEEP_DEPTH —
    // shallower than that the cavern crossfade already provides visual variety,
    // and stacking both makes the first hundred cells read as noise.
    if strata > STRATA_LIGHT_T {
        return STRATA_LIGHT;
    }
    if strata < STRATA_DARK_T {
        return STRATA_DARK;
    }
    layer.rock
}

// --- Liquids -----------------------------------------------------------------

/// The liquid that fills an open cell below the liquid table. Which liquid is the
/// underground LAYER's business — water in flooded grottos, acid in the fungal
/// depths, lava in magma chambers — crossfaded with the surface biome's pocket
/// liquid in the cavern band exactly like the rock is.
///
/// `depth` is the BAND depth, fractional, exactly as in [`solid_at`] — the
/// liquid table has to change band on the same undulating line the rock does.
pub fn liquid_at(
    noise: &Noise,
    wcx: i32,
    wcy: i32,
    depth: f64,
    col: &ColumnProfile,
    u: f64,
    scale: WorldScale,
) -> CellId {
    if depth >= f64::from(UNDERWORLD_DEPTH) {
        return M_LAVA; // the underworld is one lava sea
    }

    let src = vein_field(noise, wcx, wcy, scale);
    if depth >= f64::from(CAVERN_DEPTH) {
        return layer_at(noise, wcx, wcy, col, src).def().pocket;
    }

    if u <= 0.002 {
        return surf_biome_at(noise, wcx, wcy, col, src).def().pocket;
    }
    if u >= 0.998 {
        return layer_at(noise, wcx, wcy, col, src).def().pocket;
    }
    if reuse_dither(noise, wcx, wcy, src, 4099) < u {
        layer_at(noise, wcx, wcy, col, src).def().pocket
    } else {
        surf_biome_at(noise, wcx, wcy, col, src).def().pocket
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SEED;
    use crate::config::WorldScale;
    use crate::sim::biomes::column_profile_at;
    use crate::sim::materials::MAT_COUNT;

    #[test]
    fn both_dithers_stay_inside_the_unit_interval() {
        // Both are thresholded against a blend weight in [0, 0.5] (or a shore
        // weight in [0,1]). A value outside [0,1] would silently make one side of
        // a boundary unreachable.
        let noise = Noise::new(SEED);
        for wcy in -40..40 {
            for wcx in -40..40 {
                let d = cap_dither(&noise, wcx, wcy, WorldScale::LEGACY);
                assert!((0.0..=1.0).contains(&d), "cap_dither({wcx},{wcy}) = {d}");
                for v in [-1.0, -0.5, 0.0, 0.37, 1.0] {
                    let d = reuse_dither(&noise, wcx, wcy, v, 4099);
                    assert!((0.0..=1.0).contains(&d), "reuse_dither = {d}");
                }
            }
        }
    }

    #[test]
    fn reuse_dither_is_decorrelated_by_its_salt() {
        // The whole point of the salt: the same source value must not make the
        // surface/underground handover line up with the layer choice.
        let noise = Noise::new(SEED);
        let mut same = 0;
        for wcx in 0..500 {
            let v = vein_field(&noise, wcx, 300, WorldScale::LEGACY);
            let a = reuse_dither(&noise, wcx, 300, v, 9176);
            let b = reuse_dither(&noise, wcx, 300, v, 271);
            if (a - b).abs() < 1e-12 {
                same += 1;
            }
        }
        assert_eq!(same, 0, "two salts agreed exactly — the hash is not salted");
    }

    #[test]
    fn reuse_dither_takes_no_second_noise_sample() {
        // It is a pure function of (wcx, wcy, v, salt) and the noise's stateless
        // hash. Interleaving other work must not move it — that is what makes it
        // cheaper than a second lattice sample AND what the terrain was tuned
        // against.
        let noise = Noise::new(SEED);
        let a = reuse_dither(&noise, 17, -93, 0.42, 4099);
        for i in 0..1000 {
            noise.n2(f64::from(i) * 0.3, 11.0);
        }
        assert_eq!(reuse_dither(&noise, 17, -93, 0.42, 4099), a);
    }

    #[test]
    fn every_band_yields_a_real_material() {
        let noise = Noise::new(SEED);
        for wcx in [-4001, -37, 0, 91, 5000] {
            let col = column_profile_at(&noise, wcx, WorldScale::LEGACY);
            for depth in [
                0,
                20,
                60,
                CAVERN_DEPTH,
                250,
                UNDERWORLD_DEPTH,
                UNDERWORLD_FLOOR,
                900,
            ] {
                let wcy = 48 + depth;
                let u = ug_fade_at(depth, &col, WorldScale::LEGACY);
                for strata in [-0.9, 0.0, 0.9] {
                    let m = solid_at(
                        &noise,
                        wcx,
                        wcy,
                        f64::from(depth),
                        &col,
                        u,
                        strata,
                        WorldScale::LEGACY,
                    );
                    assert!(m != AIR && (m as usize) < MAT_COUNT, "solid_at gave {m}");
                    let l = liquid_at(
                        &noise,
                        wcx,
                        wcy,
                        f64::from(depth),
                        &col,
                        u,
                        WorldScale::LEGACY,
                    );
                    assert!(l != AIR && (l as usize) < MAT_COUNT, "liquid_at gave {l}");
                }
                let c = cap_at(&noise, wcx, wcy, depth, &col, 0.0, WorldScale::LEGACY);
                assert!(c != AIR && (c as usize) < MAT_COUNT, "cap_at gave {c}");
            }
        }
    }

    #[test]
    fn the_world_has_a_bottom_and_an_underworld_ceiling() {
        let noise = Noise::new(SEED);
        let col = column_profile_at(&noise, 128, WorldScale::LEGACY);
        for depth in [UNDERWORLD_FLOOR, UNDERWORLD_FLOOR + 1, 10_000] {
            assert_eq!(
                solid_at(
                    &noise,
                    128,
                    48 + depth,
                    f64::from(depth),
                    &col,
                    1.0,
                    0.0,
                    WorldScale::LEGACY
                ),
                M_OBSIDIAN,
                "bedrock is not bedrock at depth {depth}"
            );
        }
        for depth in [UNDERWORLD_DEPTH, UNDERWORLD_DEPTH + 50] {
            assert_eq!(
                liquid_at(
                    &noise,
                    128,
                    48 + depth,
                    f64::from(depth),
                    &col,
                    1.0,
                    WorldScale::LEGACY
                ),
                M_LAVA
            );
        }
    }

    #[test]
    fn the_shore_cap_is_sand_over_sandstone_and_only_near_the_top() {
        let noise = Noise::new(SEED);
        let col = column_profile_at(&noise, 64, WorldScale::LEGACY);
        // shore = 1 makes the hash test certain, so the branch is deterministic.
        for depth in 0..3 {
            assert_eq!(
                cap_at(&noise, 64, 48 + depth, depth, &col, 1.0, WorldScale::LEGACY),
                M_SAND
            );
        }
        for depth in 3..8 {
            assert_eq!(
                cap_at(&noise, 64, 48 + depth, depth, &col, 1.0, WorldScale::LEGACY),
                M_SANDSTONE
            );
        }
    }

    #[test]
    fn the_underground_fade_is_a_weight() {
        let noise = Noise::new(SEED);
        let col = column_profile_at(&noise, -777, WorldScale::LEGACY);
        let mut prev = -1.0;
        for depth in 0..200 {
            let u = ug_fade_at(depth, &col, WorldScale::LEGACY);
            assert!(
                (0.0..=1.0).contains(&u),
                "ug_fade_at({depth}, WorldScale::LEGACY) = {u}"
            );
            assert!(u >= prev, "the fade went backwards at depth {depth}");
            prev = u;
        }
        assert_eq!(ug_fade_at(0, &col, WorldScale::LEGACY), 0.0);
        assert_eq!(ug_fade_at(1000, &col, WorldScale::LEGACY), 1.0);
    }
}
