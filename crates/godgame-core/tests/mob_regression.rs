//! Regression gate: redrawing a creature must not move a single tuned number.
//!
//! A port of `tools/verify-mobs.mjs` from the TypeScript build, which is a
//! STANDING gate there and stays one here.
//!
//! `OLD` below is a snapshot of `MOB_DEFS` taken from
//! `src/entities/mobs/MobDefs.ts` as it stood BEFORE the content compiler owned
//! it — the 8 creatures then in the game, with every derived field including the
//! body-scaled motion numbers. It is copied across verbatim, digit for digit.
//! Creatures added since are ignored by design.
//!
//! The scaled fields are the ones that matter. `speed`, `gravity`, `knockback`
//! and friends are the product of an authored number and
//! `PHYS_SCALE * (body_cells_h * CELL_SIZE / PLAYER_H)`; comparing them exactly
//! is what proves the multiply still happens AT LOAD in the facade and was NOT
//! baked into `godgame-data`, where it would have frozen against whatever
//! `PHYS_SCALE` happened to be on the day of the migration.
//!
//! ---- WHAT THE TYPESCRIPT GATE RETIRED, AND WHY IT MATTERS HERE -------------
//! That file used to ALSO re-extract every compiled pose heredoc and compare
//! palettes, fps and frame tables against the 2026 snapshot. All of it was
//! removed when art and body were decoupled (`body_cells_*` in content;
//! `art_w_px`/`art_h_px` on the def). Those checks pinned the ART to a
//! particular drawing, which is precisely what a redraw is supposed to change.
//!
//! What is left is not a weaker claim but a STRONGER one. Before the split,
//! "the stats match" and "the art matches" were entangled: `h_px` came off the
//! sprite, so the numbers could only be equal while the picture was. Now the
//! picture is free to change and the assertion reads: **THE ART WAS REDRAWN AND
//! NOT ONE TUNED NUMBER MOVED.**
//!
//! ---- ON EXACTNESS ----------------------------------------------------------
//! JavaScript has one number type, so the snapshot literals are `f64` and carry
//! the full decimal expansion of each product. This crate computes in `f32` (see
//! the width note at the head of `entities/player.rs`). Each expected value is
//! therefore written as the `f64` the TypeScript produced and compared to the
//! `f32` the port produces after `as f32` — "the same number, at this width".
//! That is an EXACT comparison, not a tolerance: a value that differs by one
//! `f32` ulp fails, which is what makes the gate able to see a multiply that
//! moved from load time to bake time.

use godgame_core::entities::mobs::defs::{
    BAND_SURFACE_MAX, MOB_DEFS, MobBand, MobBrain, MobDef, band_at_depth, def_index,
};

/// Must match `CELL_SIZE` in `config::world`. **Restated, not imported** — the
/// checker must not be able to inherit the bug it is looking for.
const CELL: f32 = 5.0;

/// One pre-migration creature, field for field.
///
/// The TypeScript gate listed 25 field NAMES and compared them by string key.
/// Rust has no such reflection, so the list is this struct's fields and the
/// comparison is [`check`] — which means a field added to `MobDef` and
/// forgotten here is visible as a struct that no longer describes the def,
/// rather than as a silently skipped string.
struct Old {
    id: &'static str,
    brain: MobBrain,
    w_px: f64,
    h_px: f64,
    /// The pre-migration ART footprint, which back then WAS the collision box —
    /// art and hitbox were the same rectangle by construction. Recorded here as
    /// the body it became, so the box check states the thing that must stay true
    /// (the creature occupies the same space) rather than the thing that is now
    /// free to change (how much of that space is drawn).
    body_cells: [i32; 2],
    max_health: f64,
    contact_damage: f64,
    attack_cooldown: f64,
    speed: f64,
    chase_speed: f64,
    jump_speed: f64,
    gravity: f64,
    max_fall: f64,
    step_up_max: f64,
    knockback: f64,
    aggro_px: f64,
    aggro_y_px: f64,
    flee_px: f64,
    beat: f64,
    glow: f64,
    lava_immune: bool,
    blood: [u8; 3],
    bands: &'static [MobBand],
    biomes: Option<&'static [&'static str]>,
    nocturnal: bool,
    needs_lava: bool,
    weight: f64,
}

use MobBand::{Cavern, Deep, Shallow, Surface};

#[rustfmt::skip]
const OLD: &[Old] = &[
    Old {
        id: "grubling", brain: MobBrain::Walker, w_px: 10.0, h_px: 10.0, body_cells: [2, 2],
        max_health: 14.0, contact_damage: 8.0, attack_cooldown: 0.85,
        speed: 25.208333333333332, chase_speed: 49.27083333333333, jump_speed: 0.0,
        gravity: 550.0, max_fall: 320.8333333333333, step_up_max: 5.0,
        knockback: 59.58333333333333, aggro_px: 130.0, aggro_y_px: 40.0,
        flee_px: 0.0, beat: 2.4, glow: 0.0, lava_immune: false,
        blood: [169, 117, 61], bands: &[Surface, Shallow],
        biomes: Some(&["plains", "savanna", "jungle", "swamp", "tundra", "desert"]),
        nocturnal: false, needs_lava: false, weight: 26.0,
    },
    Old {
        id: "critter", brain: MobBrain::Skitter, w_px: 10.0, h_px: 10.0, body_cells: [2, 2],
        max_health: 6.0, contact_damage: 0.0, attack_cooldown: 0.85,
        speed: 27.5, chase_speed: 75.625, jump_speed: 128.33333333333331,
        gravity: 550.0, max_fall: 320.8333333333333, step_up_max: 5.0,
        knockback: 59.58333333333333, aggro_px: 0.0, aggro_y_px: 60.0,
        flee_px: 105.0, beat: 1.6, glow: 0.0, lava_immune: false,
        blood: [242, 227, 200], bands: &[Surface],
        biomes: Some(&["plains", "savanna", "jungle", "swamp", "tundra", "glacier"]),
        nocturnal: false, needs_lava: false, weight: 22.0,
    },
    Old {
        id: "slime", brain: MobBrain::Hopper, w_px: 15.0, h_px: 10.0, body_cells: [3, 2],
        max_health: 20.0, contact_damage: 10.0, attack_cooldown: 0.85,
        speed: 43.541666666666664, chase_speed: 43.541666666666664, jump_speed: 146.66666666666666,
        gravity: 550.0, max_fall: 320.8333333333333, step_up_max: 5.0,
        knockback: 59.58333333333333, aggro_px: 145.0, aggro_y_px: 70.0,
        flee_px: 0.0, beat: 1.05, glow: 0.0, lava_immune: false,
        blood: [73, 176, 119], bands: &[Shallow, Cavern], biomes: None,
        nocturnal: false, needs_lava: false, weight: 20.0,
    },
    Old {
        id: "bloat", brain: MobBrain::Hopper, w_px: 20.0, h_px: 15.0, body_cells: [4, 3],
        max_health: 46.0, contact_damage: 18.0, attack_cooldown: 1.1,
        speed: 51.5625, chase_speed: 51.5625, jump_speed: 192.5,
        gravity: 825.0, max_fall: 481.25, step_up_max: 5.0,
        knockback: 89.375, aggro_px: 165.0, aggro_y_px: 90.0,
        flee_px: 0.0, beat: 1.7, glow: 0.0, lava_immune: false,
        blood: [107, 79, 160], bands: &[Cavern, Deep], biomes: None,
        nocturnal: false, needs_lava: false, weight: 8.0,
    },
    Old {
        id: "bat", brain: MobBrain::Flyer, w_px: 15.0, h_px: 10.0, body_cells: [3, 2],
        max_health: 10.0, contact_damage: 6.0, attack_cooldown: 0.7,
        speed: 45.83333333333333, chase_speed: 82.5, jump_speed: 0.0,
        gravity: 0.0, max_fall: 320.8333333333333, step_up_max: 5.0,
        knockback: 59.58333333333333, aggro_px: 170.0, aggro_y_px: 130.0,
        flee_px: 0.0, beat: 0.9, glow: 0.0, lava_immune: false,
        blood: [74, 61, 92], bands: &[Surface, Shallow, Cavern, Deep], biomes: None,
        nocturnal: true, needs_lava: false, weight: 12.0,
    },
    Old {
        id: "driftling", brain: MobBrain::Flyer, w_px: 10.0, h_px: 10.0, body_cells: [2, 2],
        max_health: 5.0, contact_damage: 0.0, attack_cooldown: 0.85,
        speed: 16.041666666666664, chase_speed: 16.041666666666664, jump_speed: 0.0,
        gravity: 0.0, max_fall: 320.8333333333333, step_up_max: 5.0,
        knockback: 59.58333333333333, aggro_px: 0.0, aggro_y_px: 60.0,
        flee_px: 70.0, beat: 2.2, glow: 0.55, lava_immune: false,
        blood: [158, 242, 255], bands: &[Cavern, Deep], biomes: None,
        nocturnal: false, needs_lava: false, weight: 12.0,
    },
    Old {
        id: "sandworm", brain: MobBrain::Burrower, w_px: 20.0, h_px: 10.0, body_cells: [4, 2],
        max_health: 30.0, contact_damage: 16.0, attack_cooldown: 1.0,
        speed: 34.375, chase_speed: 48.125, jump_speed: 160.41666666666666,
        gravity: 550.0, max_fall: 320.8333333333333, step_up_max: 5.0,
        knockback: 59.58333333333333, aggro_px: 190.0, aggro_y_px: 110.0,
        flee_px: 0.0, beat: 2.8, glow: 0.0, lava_immune: false,
        blood: [195, 154, 90], bands: &[Surface, Shallow],
        biomes: Some(&["desert", "savanna", "tundra", "glacier"]),
        nocturnal: false, needs_lava: false, weight: 10.0,
    },
    Old {
        id: "emberling", brain: MobBrain::Walker, w_px: 10.0, h_px: 15.0, body_cells: [2, 3],
        max_health: 26.0, contact_damage: 22.0, attack_cooldown: 0.9,
        speed: 44.6875, chase_speed: 85.9375, jump_speed: 220.0,
        gravity: 825.0, max_fall: 481.25, step_up_max: 5.0,
        knockback: 89.375, aggro_px: 175.0, aggro_y_px: 80.0,
        flee_px: 0.0, beat: 1.4, glow: 0.45, lava_immune: true,
        blood: [255, 138, 60], bands: &[Shallow, Cavern, Deep], biomes: None,
        nocturnal: false, needs_lava: true, weight: 14.0,
    },
];

/// One field, compared exactly. `want` is the TypeScript `f64` at this crate's
/// width; see the note on exactness at the head of the file.
#[track_caller]
fn eq_f(id: &str, field: &str, got: f32, want: f64) {
    assert_eq!(got, want as f32, "{id}.{field}: {got} != {want}");
}

#[test]
fn band_surface_max_is_eight() {
    assert_eq!(BAND_SURFACE_MAX, 8);
}

#[test]
fn band_at_depth_answers_the_same_at_every_boundary() {
    // The same eight probe points the TypeScript gate used. `9e9` there is
    // "arbitrarily deep"; an `i32` depth spells that `i32::MAX`.
    let probes: [(i32, MobBand); 8] = [
        (0, Surface),
        (7, Surface),
        (8, Shallow),
        (109, Shallow),
        (110, Cavern),
        (209, Cavern),
        (210, Deep),
        (i32::MAX, Deep),
    ];
    for (depth, band) in probes {
        assert_eq!(band_at_depth(depth), band, "band_at_depth({depth})");
    }
}

#[test]
fn every_pre_migration_creature_matches_field_for_field() {
    for old in OLD {
        let code = def_index(old.id)
            .unwrap_or_else(|| panic!("{}: missing from the migrated bestiary", old.id));
        let now: &MobDef = &MOB_DEFS[code];
        let id = old.id;

        assert_eq!(now.brain, old.brain, "{id}.brain");
        eq_f(id, "w_px", now.w_px, old.w_px);
        eq_f(id, "h_px", now.h_px, old.h_px);
        eq_f(id, "max_health", now.max_health, old.max_health);
        eq_f(id, "contact_damage", now.contact_damage, old.contact_damage);
        eq_f(
            id,
            "attack_cooldown",
            now.attack_cooldown,
            old.attack_cooldown,
        );

        // The body-scaled block — the whole reason this file exists.
        eq_f(id, "speed", now.speed, old.speed);
        eq_f(id, "chase_speed", now.chase_speed, old.chase_speed);
        eq_f(id, "jump_speed", now.jump_speed, old.jump_speed);
        eq_f(id, "gravity", now.gravity, old.gravity);
        eq_f(id, "max_fall", now.max_fall, old.max_fall);
        eq_f(id, "step_up_max", now.step_up_max, old.step_up_max);
        eq_f(id, "knockback", now.knockback, old.knockback);

        eq_f(id, "aggro_px", now.aggro_px, old.aggro_px);
        eq_f(id, "aggro_y_px", now.aggro_y_px, old.aggro_y_px);
        eq_f(id, "flee_px", now.flee_px, old.flee_px);
        eq_f(id, "beat", now.beat, old.beat);
        eq_f(id, "glow", now.glow, old.glow);
        assert_eq!(now.lava_immune, old.lava_immune, "{id}.lava_immune");
        assert_eq!(now.blood, old.blood, "{id}.blood");
        assert_eq!(now.bands, old.bands, "{id}.bands");
        assert_eq!(now.biomes, old.biomes, "{id}.biomes");
        assert_eq!(now.nocturnal, old.nocturnal, "{id}.nocturnal");
        assert_eq!(now.needs_lava, old.needs_lava, "{id}.needs_lava");
        eq_f(id, "weight", now.weight, old.weight);
    }
}

#[test]
fn the_collision_box_is_still_a_whole_number_of_cells() {
    // Restated in CELLS and checked independently of the px-space fields above.
    // A whole number of cells is a load-bearing invariant of the whole sim, and
    // this is the one assertion that would catch `w_px`/`h_px` quietly picking
    // up the art rect again.
    for old in OLD {
        let now = &MOB_DEFS[def_index(old.id).unwrap()];
        assert_eq!(
            now.w_px,
            old.body_cells[0] as f32 * CELL,
            "{}.w_px is not {} cells",
            old.id,
            old.body_cells[0]
        );
        assert_eq!(
            now.h_px,
            old.body_cells[1] as f32 * CELL,
            "{}.h_px is not {} cells",
            old.id,
            old.body_cells[1]
        );
    }
}

#[test]
fn the_art_may_have_grown_but_never_shrunk_past_the_body() {
    // The other half of the decoupling: the picture is free to change, and this
    // is the only thing it may not do. A creature that gets a wing widens
    // `art_w_px` and leaves `w_px` alone; the test above is what notices if the
    // two ever fuse back together.
    for old in OLD {
        let now = &MOB_DEFS[def_index(old.id).unwrap()];
        assert!(
            now.art_w_px >= now.w_px && now.art_h_px >= now.h_px,
            "{}: art {}x{} is smaller than its {}x{} body",
            old.id,
            now.art_w_px,
            now.art_h_px,
            now.w_px,
            now.h_px
        );
    }
}
