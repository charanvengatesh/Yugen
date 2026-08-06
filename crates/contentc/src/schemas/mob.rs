//! The `mob` schema — the contract for everything in `content/mobs/*.toml`.
//!
//! This is the file that used to be the `Spec` interface, the `ART_*` tables and
//! the `MOB_DEFS` array at the bottom of `src/entities/mobs/MobDefs.ts`. All
//! three are derived from it now: the emitted `MobSpec` struct comes from
//! `fields`, and the flat tables the spawn picker runs over come from `hot`
//! declarations. Adding a creature means adding a record to `content/mobs/`.
//!
//! ---- HOW THE NUMBERS ARE WRITTEN -------------------------------------------
//! Every distance-bearing stat is authored in the SAME reference frame as
//! `src/config/physics.ts`: the value that would feel right on the 24px-tall
//! character the player's physics were tuned against. `MobDefs.ts` then scales it
//! to the creature's own body by `PHYS_SCALE * (bodyCellsH * CELL / PLAYER_H)`,
//! which is exactly the rule constants.ts uses — lengths, velocities and
//! accelerations scale with body height; durations and ratios do not. The upshot
//! is that a stat can be read directly against a player stat: a walker at `speed 110` covers
//! ground at roughly a third of the player's `MAX_RUN_SPEED: 360` *relative to
//! its own size*, and it stays true if PHYS_SCALE or the player's size is ever
//! retuned.
//!
//! THE COMPILER MUST EMIT THE RAW AUTHORED NUMBERS. That multiply depends on
//! runtime constants and on the sprite's measured height, so it belongs in the
//! facade and nowhere else. Baking PHYS_SCALE into generated output would
//! silently desync the bestiary from the player the next time anyone retunes
//! movement — the failure mode being that the numbers in these files stop meaning
//! what their comments say they mean, which is the one thing a content format
//! exists to prevent.
//!
//! Radii (`aggroPx`, `fleePx`, `ranged.range`) are deliberately NOT body-scaled:
//! they are about how far away something notices you ON SCREEN, which is a
//! property of the camera, not of the creature's legs. They are plain world px.
//!
//! ---- HABITAT ---------------------------------------------------------------
//! A creature is eligible at a candidate cell when the depth band matches, the
//! depth is inside any explicit `minDepth`/`maxDepth` window, the dominant biome
//! is in its list (absent list = anywhere), and any extra predicate (`needsLava`)
//! holds. Eligible defs are then picked by weight, so population composition
//! follows the terrain without any per-biome spawn table. `bands` is the coarse
//! knob and stays the primary one; the depth window is for the handful of species
//! that want a narrower slice than a whole band.
//!
//! ---- ART -------------------------------------------------------------------
//! A pose is ONE `art.seq` record: its attributes as keys, every frame of that
//! loop in the `frames` body, frames separated by a blank line. A TOML literal
//! string keeps those blank lines verbatim, so the source reads as a filmstrip:
//!
//! ```text
//!     [[grubling.art.seq]]
//!     state = "move"
//!     mode = "loop"
//!     frames = '''
//!     23
//!     1.
//!
//!     23
//!     .1
//!     '''
//! ```
//!
//! The facade splits on the blank lines and checks every frame against
//! `art.cellsH`. One character is one WORLD CELL (see MobSprite.ts), so what you
//! see in the file is the silhouette the game draws.
//!
//! The pose used to be three separate keys (`art.idle`/`art.move`/`art.air`),
//! which could carry pixels and nothing else. Making each one a `record[]`
//! element is what lets a pose also say HOW it plays — `mode`, its own `fps`,
//! the ambient blink timings — without inventing a parallel `art.moveMode` key
//! per attribute, and it is the same shape the player's sprite uses.
//!
//! Rates: a sequence with no `fps` inherits `art.fps`. Do NOT pre-halve the
//! idle rate in content, even though idle does play at half speed — that halving
//! is a rendering decision the facade applies to every creature at once, and
//! baking it into 14 files would mean 14 places to change it and 14 chances to
//! disagree.

use super::sprite::{SpriteArtOpts, sprite_art_fields};
use crate::schema::{ArrayKind, BitConstants, Field, Schema, table};
use crate::value::Value;

/// The pose vocabulary, closed. Three, because three is what the brains can
/// produce: standing, locomoting, and off the ground. `air` is optional per
/// creature and the FACADE borrows `move` for anything that lacks one — that
/// rule is about how the game animates creatures, not a fact about any one of
/// them, so it is deliberately not authorable here (see MobDefs' `fallback`).
const MOB_POSES: &[&str] = &["idle", "move", "air"];

/// How a creature moves and decides. Mirrors `MobBrain` in MobDefs.ts.
const BRAINS: &[&str] = &["walker", "hopper", "flyer", "skitter", "burrower"];

/// Depth band relative to the local surface row. Mirrors `Band` in MobDefs.ts.
/// Bit order here is the bit order of MOB_BANDS, so it is a stable property of
/// this file rather than of whatever is authored today.
const BANDS: &[&str] = &["surface", "shallow", "cavern", "deep"];

/// The damage-tag vocabulary a creature can be immune to. Declared rather than
/// open-ended so a typo is a compile error: `immune fier` on a fire elemental
/// that then burns to death in its own home biome is exactly the kind of silent
/// content rot an open string list invites.
const IMMUNITIES: &[&str] = &["fire", "acid", "ice", "impact", "poison"];

/// Boolean habitat/behaviour flags, packed into MOB_FLAGS. Bit order is fixed.
const FLAGS: &[&str] = &["nocturnal", "needsLava", "lavaImmune", "ranged"];

/// What a ranged attacker throws. An enum rather than a free string because the
/// projectile's speed, size, colour and on-hit rider live in a table in
/// `MobSystem.ts` — a new kind needs code as well as content, so content should
/// not be able to invent one.
const PROJECTILES: &[&str] = &["ice_shard", "ember", "spore", "stone"];

pub fn schema() -> Schema {
    let mut fields: Vec<(String, Field)> = vec![
        (
            "brain".into(),
            Field::new(&format!("enum({})", BRAINS.join("|")))
                .doc("Which movement/decision routine drives this creature.")
                .required()
                .alias("MobBrain"),
        ),
        (
            "name".into(),
            Field::new("string")
                .doc("Display name (bestiary UI, debug HUD).")
                .default_fn(|d| Some(Value::Str(d.str_of("id").to_string()))),
        ),
        // --- combat -------------------------------------------------------------
        (
            "maxHealth".into(),
            Field::new("float").doc("Hit points.").required(),
        ),
        (
            "contactDamage".into(),
            Field::new("float")
                .doc("Health taken from the player per contact hit. 0 = harmless.")
                .default_float(0.0),
        ),
        (
            "attackCooldown".into(),
            Field::new("float")
                .doc(
                    "Seconds this creature must wait between contact hits. A DURATION — never \
                     body-scaled.",
                )
                .default_float(0.85),
        ),
        (
            "armor".into(),
            Field::new("float")
                .doc(
                    "Flat damage subtracted from every hit this creature takes. Flat rather \
                     than a percentage so a heavily armoured creature is still killable with \
                     a weak weapon, just slowly; the combat code floors the result at 1.",
                )
                .default_float(0.0)
                .check(|v| {
                    if v.as_num().unwrap_or(0.0) < 0.0 {
                        Some("must be >= 0".to_string())
                    } else {
                        None
                    }
                }),
        ),
        (
            "immune".into(),
            Field::new(&format!("list<enum({})>", IMMUNITIES.join("|")))
                .doc(
                    "Damage tags this creature ignores entirely — a fire elemental should not \
                     burn.",
                )
                .default(Value::List(vec![]))
                .hot(vec![table(
                    "MOB_IMMUNE",
                    ArrayKind::U32,
                    0.0,
                    "Packed immunity bitmask: `MOB_IMMUNE[c] & DMG.fire`.",
                    |d, ctx| Some(ctx.mask("DMG", &d.strings_of("immune"))),
                )]),
        ),
        // --- body, the physics geometry (NOT art) -------------------------------
        // These are top-level, not under `art.`, and that placement is the whole
        // point. `MobDefs.ts` body-scales speed/jump/gravity/knockback by
        // `PHYS_SCALE * (bodyHeightPx / PLAYER_H)`, and it used to take that height
        // from `art.cellsH` — so redrawing a creature one cell taller silently made
        // it faster, jump higher and fall harder. Art and physics are now separate
        // authored facts: `art.cellsH` says how big the picture is, `bodyCellsH`
        // says how big the creature is.
        //
        // REQUIRED, with deliberately NO default. A default of `art.cellsH` would
        // reproduce the exact coupling this pair exists to remove, and would do it
        // invisibly — every existing record would keep working and the next redraw
        // would retune physics again. Making it a compile error is the only version
        // of this change that actually holds.
        (
            "bodyCellsW".into(),
            Field::new("int")
                .doc(
                    "Collision box width in world cells. Independent of `art.cellsW`: a \
                     sprite may overhang its own hitbox (wings, antennae) without widening \
                     what the world collides with.",
                )
                .required()
                .check(|v| {
                    if v.as_num().unwrap_or(0.0) < 1.0 {
                        Some("must be at least 1 cell".to_string())
                    } else {
                        None
                    }
                }),
        ),
        (
            "bodyCellsH".into(),
            Field::new("int")
                .doc(
                    "Collision box height in world cells, AND the body-scale anchor every \
                     distance-bearing stat below is multiplied against. Change this and the \
                     creature's whole motion budget moves with it — which is what you want \
                     when a creature genuinely gets bigger, and is exactly what you do not \
                     want when only its drawing changes.",
                )
                .required()
                .check(|v| {
                    if v.as_num().unwrap_or(0.0) < 1.0 {
                        Some("must be at least 1 cell".to_string())
                    } else {
                        None
                    }
                }),
        ),
        // --- motion, in the player's reference frame (see the header) -----------
        (
            "speed".into(),
            Field::new("float")
                .doc("Patrol speed. Body-scaled by the facade.")
                .required(),
        ),
        (
            "chaseSpeed".into(),
            Field::new("float").doc(
                "Speed while alerted. Defaults to `speed` — a creature that does not change \
                 gear.",
            ),
        ),
        (
            "jumpSpeed".into(),
            Field::new("float")
                .doc(
                    "Initial upward velocity for a hop, ledge-leap or eruption. 0 = cannot \
                     leave the ground.",
                )
                .default_float(0.0),
        ),
        (
            "gravityScale".into(),
            Field::new("float")
                .doc(
                    "Multiplier on the player's raw gravity. 0 = flight (the flyer brain \
                     assumes it).",
                )
                .default_float(1.0),
        ),
        // --- perception, plain world px (NOT body-scaled) -----------------------
        (
            "aggroPx".into(),
            Field::new("float")
                .doc("Horizontal notice radius for a hostile. 0 = never aggresses.")
                .default_float(0.0),
        ),
        (
            "aggroYPx".into(),
            Field::new("float")
                .doc("Vertical half-height of the notice box.")
                .default_float(60.0),
        ),
        (
            "fleePx".into(),
            Field::new("float")
                .doc(
                    "Horizontal panic radius. Non-zero makes the creature run AWAY and \
                     overrides aggroPx.",
                )
                .default_float(0.0),
        ),
        // --- ranged attack ------------------------------------------------------
        (
            "ranged.projectile".into(),
            Field::new(&format!("enum({})", PROJECTILES.join("|")))
                .doc(
                    "What it throws. Speed, size and on-hit rider come from the table in \
                     MobSystem.ts.",
                )
                .required()
                .alias("MobProjectile"),
        ),
        (
            "ranged.cooldown".into(),
            Field::new("float")
                .doc("Seconds between shots. A DURATION — never body-scaled.")
                .required(),
        ),
        (
            "ranged.range".into(),
            Field::new("float")
                .doc("Horizontal firing radius, world px. Perception, so not body-scaled.")
                .required(),
        ),
        (
            "ranged.damage".into(),
            Field::new("float")
                .doc("Health taken on a projectile hit.")
                .required(),
        ),
        // --- rhythm and look ----------------------------------------------------
        (
            "beat".into(),
            Field::new("float")
                .doc(
                    "Seconds between hops / dives / surfacings, depending on brain. A \
                     DURATION.",
                )
                .default_float(1.0),
        ),
        (
            "glow".into(),
            Field::new("float")
                .doc("Additive self-illumination drawn in the post-light overlay pass, 0..1.")
                .default_float(0.0)
                .check(|v| {
                    let n = v.as_num().unwrap_or(0.0);
                    if (0.0..=1.0).contains(&n) {
                        None
                    } else {
                        Some("must be 0..1".to_string())
                    }
                }),
        ),
        (
            "lavaImmune".into(),
            Field::new("bool")
                .doc(
                    "Skip the material-hazard probe outright. Blunter than `immune fire` and \
                     kept separate from it: this one says 'nothing the world is made of can \
                     hurt me', which is what a creature that LIVES in lava needs.",
                )
                .default_bool(false),
        ),
        (
            "blood".into(),
            Field::new("color")
                .doc("Particle colour for hits and death bursts — the creature's body tone.")
                .required(),
        ),
        // --- rewards ------------------------------------------------------------
        (
            "xp".into(),
            Field::new("int")
                .doc("Experience awarded for a kill.")
                .default_int(0),
        ),
        (
            "value".into(),
            Field::new("int")
                .doc("Currency awarded for a kill, separate from xp.")
                .default_int(0),
        ),
        (
            "drops".into(),
            Field::new("record[]")
                .doc(
                    "What killing this yields. Rolled independently, so several entries can \
                     all drop.",
                )
                .element(
                    "MobDrop",
                    vec![
                        // The item registry is being built concurrently, so an unknown id warns
                        // rather than failing the build; the ref is re-checked for real once
                        // items land. Content must be authorable ahead of the code it names.
                        (
                            "item".to_string(),
                            Field::new("ref?(item)")
                                .doc(
                                    "Item id, or a `a|b|c` preference chain. Unvalidated until \
                                     the item registry exists.",
                                )
                                .required()
                                .lenient(),
                        ),
                        (
                            "count".to_string(),
                            Field::new("range")
                                .doc("Inclusive stack size range.")
                                .default(Value::Range(1.0, 1.0)),
                        ),
                        (
                            "chance".to_string(),
                            Field::new("chance")
                                .doc("Probability this entry drops at all.")
                                .default_float(1.0),
                        ),
                    ],
                ),
        ),
        // --- habitat ------------------------------------------------------------
        (
            "bands".into(),
            Field::new(&format!("list<enum({})>", BANDS.join("|")))
                .doc("Depth bands this may spawn in. The coarse habitat knob.")
                .required()
                .hot(vec![table(
                    "MOB_BANDS",
                    ArrayKind::U8,
                    0.0,
                    "Packed band bitmask: `MOB_BANDS[c] & BAND.cavern`. The spawn picker tests \
                     every def against one band per candidate cell, so this replaces an \
                     indexOf over a string array in the inner loop.",
                    |d, ctx| Some(ctx.mask("BAND", &d.strings_of("bands"))),
                )]),
        ),
        (
            "minDepth".into(),
            Field::new("int")
                .doc(
                    "Cells below the local surface before this may spawn. Overlays `bands` for \
                     the few species that want a narrower slice than a whole band.",
                )
                .default_int(0)
                .check(depth_check)
                .hot(vec![table(
                    "MOB_MINDEPTH",
                    ArrayKind::U16,
                    0.0,
                    "",
                    |d, _ctx| Some(d.num("minDepth")),
                )]),
        ),
        (
            "maxDepth".into(),
            Field::new("int")
                .doc(
                    "Last depth this may spawn at. Default 65535 = no upper bound beyond \
                     `bands`.",
                )
                .default_int(0xffff)
                .check(depth_check)
                .hot(vec![table(
                    "MOB_MAXDEPTH",
                    ArrayKind::U16,
                    0xffff as f64,
                    "",
                    |d, _ctx| Some(d.num("maxDepth")),
                )]),
        ),
        (
            "biomes".into(),
            Field::new("list<string>").doc(
                "Dominant SURFACE biome ids this may spawn in (src/sim/biomes.ts). Absent = \
                 any biome. Not a `ref(biome)` yet: biomes are still hand-written, and a \
                 forward reference that fails the build is worse than one that does not.",
            ),
        ),
        (
            "nocturnal".into(),
            Field::new("bool")
                .doc(
                    "Weight is multiplied by darkness at the surface, so these thin out in \
                     daylight.",
                )
                .default_bool(false),
        ),
        (
            "needsLava".into(),
            Field::new("bool")
                .doc("Requires lava within a short radius of the spawn point.")
                .default_bool(false),
        ),
        (
            "packSize".into(),
            Field::new("range")
                .doc(
                    "How many spawn together. The extras are placed around the first one and \
                     each must pass the same footing test, so a pack in bad terrain simply \
                     arrives smaller rather than clipping into rock.",
                )
                .default(Value::Range(1.0, 1.0))
                .check(|v| {
                    let (lo, hi) = match v {
                        Value::Range(a, b) => (*a, *b),
                        _ => return None,
                    };
                    if lo < 1.0 {
                        return Some("pack size must be at least 1".to_string());
                    }
                    if hi < lo {
                        Some("max must be >= min".to_string())
                    } else {
                        None
                    }
                }),
        ),
        (
            "weight".into(),
            Field::new("float")
                .doc("Relative spawn weight among eligible defs. 0 = never spawns naturally.")
                .default_float(1.0)
                .hot(vec![table(
                    "MOB_WEIGHT",
                    ArrayKind::F32,
                    0.0,
                    "Relative spawn weight.",
                    |d, _ctx| Some(d.num("weight")),
                )]),
        ),
    ];

    // --- art ----------------------------------------------------------------
    // Spliced in from schema/sprite.ts under the `art.` prefix, so a creature and
    // the player describe their art with the SAME fields and only the pose
    // vocabulary differs. That is the point of the shared fragment: `variants`
    // arrived for free here the day it was added for the player, and neither host
    // can grow a field the other silently lacks.
    //
    // Everything is required WITHIN the group: a record that sets no `art.*` key
    // at all has no art object, which is how a tombstone gets
    // emitted without having to carry a sprite. The facade gives those a
    // placeholder.
    fields.extend(sprite_art_fields(SpriteArtOpts {
        prefix: "art.",
        // Three poses, because three is what the brains can produce. No
        // `default_state`: on a multi-pose host an unlabelled sequence is an
        // authoring mistake, not a shorthand, so `state=` is required.
        states: MOB_POSES,
        ts_alias: "MobPose",
        ts_element: "MobSpecSeq",
        default_state: None,
    }));

    Schema {
        kind: "mob".into(),
        prefix: "MOB".into(),
        iface: "MobSpec".into(),
        // "MobSpec" rather than "Mob", so the generated group struct for `art` is
        // `MobSpecArt` and does not collide with MobSprite's own `MobArt` — the
        // two describe the same art at different stages (flat authored lines vs.
        // parsed frame lists) and sharing a name would make the facade
        // unreadable.
        iface_prefix: "MobSpec".into(),
        fields,

        tables: vec![table(
            "MOB_FLAGS",
            ArrayKind::U8,
            0.0,
            "Packed habitat/behaviour flags: `MOB_FLAGS[c] & MOBF.nocturnal`. Reading \
             presence of the `ranged` group off a bit keeps the spawn and combat \
             early-outs to one array read.",
            |d, ctx| {
                let mut bits: u32 = 0;
                if d.flag("nocturnal") {
                    bits |= 1 << ctx.bit("MOBF", "nocturnal") as u32;
                }
                if d.flag("needsLava") {
                    bits |= 1 << ctx.bit("MOBF", "needsLava") as u32;
                }
                if d.flag("lavaImmune") {
                    bits |= 1 << ctx.bit("MOBF", "lavaImmune") as u32;
                }
                if d.contains("ranged") {
                    bits |= 1 << ctx.bit("MOBF", "ranged") as u32;
                }
                Some(bits as f64)
            },
        )],

        matrices: vec![],

        constants: vec![
            BitConstants {
                name: "BAND".into(),
                doc: Some(
                    "Depth-band bits for MOB_BANDS. Bit order is fixed by \
                     tools/contentc/schema/mob.ts."
                        .into(),
                ),
                values: BANDS.iter().map(|s| s.to_string()).collect(),
            },
            BitConstants {
                name: "DMG".into(),
                doc: Some("Damage-tag bits for MOB_IMMUNE.".into()),
                values: IMMUNITIES.iter().map(|s| s.to_string()).collect(),
            },
            BitConstants {
                name: "MOBF".into(),
                doc: Some("Flag bits for MOB_FLAGS.".into()),
                values: FLAGS.iter().map(|s| s.to_string()).collect(),
            },
        ],

        // A retired id keeps its code forever so an old save still resolves whatever
        // it had spawned; weight 0 is what keeps the placeholder out of the spawn
        // picker, and no `art.*` key means no art group, which the facade renders as
        // a loud magenta nothing — exactly what you want to see if one is ever drawn.
        tombstone: vec![
            ("brain".into(), "\"walker\"".into()),
            ("name".into(), "\"(removed)\"".into()),
            ("maxHealth".into(), "1".into()),
            ("bodyCellsW".into(), "1".into()),
            ("bodyCellsH".into(), "1".into()),
            ("speed".into(), "0".into()),
            ("blood".into(), "[255, 0, 255]".into()),
            ("bands".into(), "[\"surface\"]".into()),
            ("weight".into(), "0".into()),
        ],
    }
}

fn depth_check(v: &Value) -> Option<String> {
    let n = v.as_num().unwrap_or(0.0);
    if (0.0..=65535.0).contains(&n) {
        None
    } else {
        Some("must be 0..65535".to_string())
    }
}
