//! The `feature` schema — the contract for `content/worldgen/*.feature`.
//!
//! A `struct` is a TEMPLATE: the content file says what every cell is. A
//! `feature` is PARAMETRIC: the content file declares the knobs — how big, how
//! often, how deep, out of what — and the generator in `src/sim/gen/features.ts`
//! grows the shape from them. Anything whose silhouette should vary with its size
//! (an island, a mineshaft, a lake) has to be parametric, because a template of a
//! 46-cell island is 46 cells of authoring that then only ever appears at one size.
//!
//! `kind` is therefore the load-bearing field: it names the GENERATOR, and it is
//! an enum rather than a free string because a new shape needs code as well as
//! content and the format should not be able to invent one. Everything else is a
//! knob that generator reads, so tuning a feature — rarer, bigger, deeper, out of
//! different rock — is a content edit with no code change and no recompile of the
//! sim.
//!
//! ---- THE PLACEMENT GRID ----------------------------------------------------
//! `cellW`/`cellH` are the feature's own DETERMINISTIC GRID: the world is divided
//! into cells of that size, each cell is hashed to decide whether it hosts an
//! instance and where inside itself the origin sits. That is what makes placement
//! a pure function of absolute coordinates with a bounded extent — every chunk
//! within `reachX`/`reachY` of a hosting cell independently derives the identical
//! instance and paints only its own share. Grid size is the spacing knob and
//! `rarity` is the "and even then, usually not" knob; both are content, because
//! "common enough to meet, rare enough to be a find" is a tuning question.
//!
//! `reachX`/`reachY` are a CONTRACT, not a hint: the generator clamps itself to
//! them and the decorator declares them, so an under-declared reach shows up as a
//! feature clipped at a chunk boundary. They are authored rather than derived
//! because a generator's true extent depends on its own arithmetic, and a number
//! the author has to write down is a number somebody has thought about.

use crate::schema::{ArrayKind, Field, Schema, table};
use crate::value::{Def, Value};

/// Which generator grows this feature. Numeric because the pass switches on it
/// per candidate; the codes are load-bearing and mirrored in
/// src/sim/gen/features.ts.
const KIND: &[(&str, i64)] = &[
    ("island", 0),
    ("mineshaft", 1),
    ("dungeon", 2),
    ("lake", 3),
    ("geode", 4),
    ("grove", 5),
    ("oreblob", 6),
];

/// How the origin is found. `sky` and `surface` scan a 1D COLUMN grid and derive
/// the row from the ground line — a lake has to be at the surface by definition,
/// and scanning a 2D grid for it would spend most of its hashes underground.
/// `underground` scans a real 2D grid and gates on depth.
const PLACE: &[(&str, i64)] = &[("sky", 0), ("surface", 1), ("underground", 2)];

/// One half of a `range` field. A `range` is split across two parallel flat
/// arrays, so every such table asks for one end of it.
fn range_of(d: &Def, key: &str, i: usize) -> f64 {
    match d.get(key) {
        Some(Value::Range(a, b)) => {
            if i == 0 {
                *a
            } else {
                *b
            }
        }
        _ => 0.0,
    }
}

/// A `ref?(block)` knob on the material palette. Absent = the generator's default.
fn mat(doc: &str) -> Field {
    Field::new("ref?(block)").doc(doc)
}

fn enum_of(pairs: &[(&str, i64)]) -> String {
    format!(
        "enum({})",
        pairs.iter().map(|(k, _)| *k).collect::<Vec<_>>().join("|")
    )
}

pub fn schema() -> Schema {
    Schema {
        kind: "feature".into(),
        prefix: "FEATURE".into(),
        iface: "FeatureDef".into(),
        iface_prefix: "Feature".into(),

        fields: vec![
            (
                "name".into(),
                Field::new("string")
                    .doc("Display name — debug HUD, map markers.")
                    .default_fn(|d| Some(Value::Str(d.str_of("id").to_string()))),
            ),
            (
                "kind".into(),
                Field::new(&enum_of(KIND))
                    .doc(
                        "Which generator in src/sim/gen/features.ts grows this. Adding one needs code.",
                    )
                    .map(KIND)
                    .required()
                    .hot(vec![table(
                        "FEAT_KIND",
                        ArrayKind::U8,
                        0.0,
                        "Generator code.",
                        |d, _ctx| Some(d.num("kind")),
                    )]),
            ),
            (
                "place".into(),
                Field::new("enum(sky|surface|underground)")
                    .doc(
                        "Which lattice offers a site: a 1D column grid for sky/surface, 2D for underground.",
                    )
                    .map(PLACE)
                    .required()
                    .hot(vec![table(
                        "FEAT_PLACE",
                        ArrayKind::U8,
                        0.0,
                        "Placement class code.",
                        |d, _ctx| Some(d.num("place")),
                    )]),
            ),
            // --- the grid -----------------------------------------------------
            (
                "cellW".into(),
                Field::new("int")
                    .doc("Grid cell width in cells — the mean horizontal SPACING between instances.")
                    .required()
                    .check(|v| match v.as_num() {
                        Some(n) if n < 16.0 => Some("must be at least 16 cells".into()),
                        _ => None,
                    })
                    .hot(vec![table("FEAT_CELLW", ArrayKind::U16, 0.0, "", |d, _ctx| {
                        Some(d.num("cellW"))
                    })]),
            ),
            (
                "cellH".into(),
                Field::new("int")
                    .doc("Grid cell height. Unread for sky/surface places, where the row is derived.")
                    .default_fn(|d| d.get("cellW").cloned())
                    .hot(vec![table("FEAT_CELLH", ArrayKind::U16, 0.0, "", |d, _ctx| {
                        Some(d.num("cellH"))
                    })]),
            ),
            (
                "rarity".into(),
                Field::new("chance")
                    .doc("Probability a grid cell actually hosts an instance. The 'is it a find?' knob.")
                    .required()
                    .hot(vec![table("FEAT_RARITY", ArrayKind::F32, 0.0, "", |d, _ctx| {
                        Some(d.num("rarity"))
                    })]),
            ),
            (
                "reachX".into(),
                Field::new("int")
                    .doc(
                        "Hard bound on how far the shape may extend from its origin on X. The \
                         generator clamps to it and the decorator declares it, so this is the \
                         number chunk independence rests on — see src/sim/decor/decor.ts.",
                    )
                    .required()
                    .hot(vec![table("FEAT_REACHX", ArrayKind::U16, 0.0, "", |d, _ctx| {
                        Some(d.num("reachX"))
                    })]),
            ),
            (
                "reachY".into(),
                Field::new("int")
                    .doc("Hard bound on vertical extent from the origin, both directions.")
                    .required()
                    .hot(vec![table("FEAT_REACHY", ArrayKind::U16, 0.0, "", |d, _ctx| {
                        Some(d.num("reachY"))
                    })]),
            ),
            // --- shape knobs --------------------------------------------------
            (
                "size".into(),
                Field::new("range")
                    .doc(
                        "Primary size, drawn per instance. Half-width for a blob, length for a corridor.",
                    )
                    .required()
                    .hot(vec![
                        table(
                            "FEAT_SIZE0",
                            ArrayKind::U16,
                            0.0,
                            "Smallest primary size.",
                            |d, _ctx| Some(range_of(d, "size", 0)),
                        ),
                        table(
                            "FEAT_SIZE1",
                            ArrayKind::U16,
                            0.0,
                            "Largest primary size.",
                            |d, _ctx| Some(range_of(d, "size", 1)),
                        ),
                    ]),
            ),
            (
                "height".into(),
                Field::new("range")
                    .doc("Secondary size — thickness, depth, room height. Defaults to half of `size`.")
                    .default_fn(|d| {
                        Some(Value::Range(
                            (range_of(d, "size", 0) / 2.0).round(),
                            (range_of(d, "size", 1) / 2.0).round(),
                        ))
                    })
                    .hot(vec![
                        table("FEAT_HEIGHT0", ArrayKind::U16, 0.0, "", |d, _ctx| {
                            Some(range_of(d, "height", 0))
                        }),
                        table("FEAT_HEIGHT1", ArrayKind::U16, 0.0, "", |d, _ctx| {
                            Some(range_of(d, "height", 1))
                        }),
                    ]),
            ),
            (
                "clearance".into(),
                Field::new("range")
                    .doc(
                        "For `sky` places only: cells of open air between the ground line and the \
                         feature's bottom, drawn per instance. It is a separate knob from `height` \
                         because how high something hangs and how thick it is are independent, and \
                         an island that scaled its altitude with its size would put every big one \
                         out of reach.",
                    )
                    .default(Value::Range(0.0, 0.0))
                    .hot(vec![
                        table("FEAT_CLEAR0", ArrayKind::U16, 0.0, "", |d, _ctx| {
                            Some(range_of(d, "clearance", 0))
                        }),
                        table("FEAT_CLEAR1", ArrayKind::U16, 0.0, "", |d, _ctx| {
                            Some(range_of(d, "clearance", 1))
                        }),
                    ]),
            ),
            (
                "density".into(),
                Field::new("chance")
                    .doc(
                        "Generator-specific fill fraction: props per cell, supports per run, ore per void.",
                    )
                    .default_float(0.5)
                    .hot(vec![table("FEAT_DENSITY", ArrayKind::F32, 0.0, "", |d, _ctx| {
                        Some(d.num("density"))
                    })]),
            ),
            // --- habitat ------------------------------------------------------
            (
                "minDepth".into(),
                Field::new("int")
                    .doc("Cells below the local surface before this may appear.")
                    .default_int(0)
                    .check(|v| match v.as_num() {
                        Some(n) if n < 0.0 || n > 0xffff as f64 => Some("must be 0..65535".into()),
                        _ => None,
                    })
                    .hot(vec![table("FEAT_MINDEPTH", ArrayKind::U16, 0.0, "", |d, _ctx| {
                        Some(d.num("minDepth"))
                    })]),
            ),
            (
                "maxDepth".into(),
                Field::new("int")
                    .doc("Last depth this may appear at. 65535 = no bound.")
                    .default_int(0xffff)
                    .check(|v| match v.as_num() {
                        Some(n) if n < 0.0 || n > 0xffff as f64 => Some("must be 0..65535".into()),
                        _ => None,
                    })
                    .hot(vec![table(
                        "FEAT_MAXDEPTH",
                        ArrayKind::U16,
                        0xffff as f64,
                        "",
                        |d, _ctx| Some(d.num("maxDepth")),
                    )]),
            ),
            (
                "biomes".into(),
                Field::new("list<string>").doc(
                    "Dominant SURFACE biome ids this may appear under (src/sim/biomes.ts). Absent \
                     = any. Plain strings for the same reason mobs use them: biomes are still \
                     hand-written and a forward ref that fails the build is worse than one that does not.",
                ),
            ),
            (
                "layers".into(),
                Field::new("list<string>").doc(
                    "Underground layer ids this may appear in (src/sim/biomes.ts). Absent = any.",
                ),
            ),
            // --- material palette ---------------------------------------------
            // Every generator reads the same five slots, so a lake and a dungeon are the
            // same three lines of palette resolution. Absent slots fall back to something
            // the generator picks from the terrain, which is what keeps a feature looking
            // like it belongs to whatever rock it grew in.
            (
                "mat.shell".into(),
                mat("Outer skin: island rock, dungeon wall, geode crust, lake bed lining."),
            ),
            ("mat.fill".into(), mat("Bulk interior: island soil cap, room floor, blob ore.")),
            (
                "mat.accent".into(),
                mat("The eye-catching bit: crystal lining, glowing cap, gilded trim."),
            ),
            (
                "mat.frame".into(),
                mat("Structural timber/brick: mine supports, dungeon pillars, grove stems."),
            ),
            ("mat.liquid".into(), mat("What pools inside: lake water, flooded vault, geode brine.")),
        ],

        tables: vec![],
        matrices: vec![],
        constants: vec![],

        // A retired feature keeps its code so a save's "seen features" set still
        // resolves. rarity 0 means the placement grid can never host it.
        tombstone: vec![
            ("name".into(), "(removed)".into()),
            ("kind".into(), "oreblob".into()),
            ("place".into(), "underground".into()),
            ("cellW".into(), "4096".into()),
            ("rarity".into(), "0".into()),
            ("reachX".into(), "0".into()),
            ("reachY".into(), "0".into()),
            ("size".into(), "0".into()),
        ],
    }
}
