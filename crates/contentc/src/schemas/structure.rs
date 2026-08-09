//! The `struct` schema — the contract for everything in `content/structures/*.toml`.
//!
//! A struct is a TEMPLATE: a `body` of glyphs (FORMAT.md §5) plus a `legend`
//! mapping each glyph to a block. Everything else on the record is PLACEMENT
//! metadata — where in the world the template is allowed to land, how often, and
//! how it is allowed to vary.
//!
//! ---- WHY THE LEGEND IS A RECORD PER GLYPH AND NOT ONE MAPPING --------------
//! The obvious spelling is one table, `legend = { W = "wood", G = "glass" }`.
//! That form cannot be validated: the compiler only checks an id against another
//! kind's registry when the FIELD is declared `ref(kind)` (schema.rs
//! `reference()`), and a table of glyph -> id is a bag of strings, so a typo'd
//! block id would ship as a silent hole in a building. One `legend` entry per
//! glyph makes it a `record[]` whose `block` attribute is a real `ref?(block)`,
//! so an unknown material is a build failure at the line that named it — which is
//! the whole reason the format has refs. The entry also carries the `mark` the
//! loot pass needs, which a bare glyph -> id pair has nowhere to put.
//!
//! ---- GLYPH SEMANTICS -------------------------------------------------------
//!   `.`   force air — carve whatever terrain is here away
//!   `_`   keep whatever terrain is here (also what a space and a short row mean)
//!   any other glyph must be declared by a `legend` row
//! Both reserved glyphs are rejected in a `legend` row so a template cannot
//! redefine them out from under a reader.
//!
//! ---- VARIATION -------------------------------------------------------------
//! Three axes, in increasing order of how much they earn:
//!   `weight`   picks among the ALTERNATE structs eligible at a placement, so a
//!              set of cabins can be authored as separate records instead of one
//!              record with a switch in it;
//!   `mirror`   allows the horizontal flip, which doubles the silhouette count of
//!              every asymmetric template for the cost of one index subtraction;
//!   `repeat`   names a horizontal SLICE of the body that may be stamped several
//!              times, so one tower template covers every tower height. It is a
//!              row-index remap at stamp time, not an expanded copy, so a varying
//!              height costs no allocation and stays a pure function of the origin.

use crate::bail;
use crate::error::{ContentError, Result};
use crate::schema::{ArrayKind, BitConstants, Field, Schema, TableCtx, table};
use crate::value::{Def, Value};

/// Where in the world a template may stand. Numeric because the placement pass
/// buckets templates by it and then tests one integer per candidate; the codes are
/// load-bearing and mirrored in src/sim/gen/structs.ts.
const PLACE: &[(&str, i64)] = &[
    ("surface", 0),
    ("shore", 1),
    ("floating", 2),
    ("underground", 3),
    ("cavern", 4),
    ("underworld", 5),
];

/// Which cell of the template the placement origin refers to. `bottom_center` is
/// the default because almost everything is authored standing on the ground and
/// the ground line is the one coordinate placement actually derives.
const ANCHOR: &[(&str, i64)] = &[
    ("bottom_center", 0),
    ("bottom_left", 1),
    ("center", 2),
    ("top_center", 3),
    ("top_left", 4),
];

/// What a later pass should do with this cell. `none` is just masonry.
const MARK: &[(&str, i64)] = &[("none", 0), ("loot", 1), ("spawn", 2)];

/// `MARK.loot` and `MARK.spawn` as the RESOLVED def carries them — a mapped enum
/// lands on the def as its number, so the flags table compares numbers.
const MARK_LOOT: f64 = 1.0;
const MARK_SPAWN: f64 = 2.0;

/// Depth bands, same vocabulary (and bit order) as schemas/mob.rs.
const BANDS: [&str; 5] = ["surface", "shallow", "cavern", "deep", "underworld"];

/// Bits of STRUCT_FLAGS. Bit order is fixed by this file.
const FLAGS: [&str; 3] = ["mirror", "loot", "spawn"];

/// Glyphs the stamper interprets itself; a legend may not shadow them.
const RESERVED: [char; 3] = ['.', '_', ' '];

// ---------------------------------------------------------------------------
// Derived geometry
// ---------------------------------------------------------------------------

fn body_of(d: &Def) -> &[String] {
    match d.get("body") {
        Some(Value::Text(lines)) => lines,
        _ => &[],
    }
}

fn glyphs_of(d: &Def) -> &[Def] {
    d.records_of("legend")
}

/// Both halves of a `range` living inside a group, i.e. `repeat.times`.
fn group_range(d: &Def, group: &str, leaf: &str) -> (f64, f64) {
    match d.path(group, leaf) {
        Some(Value::Range(a, b)) => (*a, *b),
        _ => (0.0, 0.0),
    }
}

/// Widest authored row. Short rows are padded with `keep`, so this is the width.
fn body_w(d: &Def) -> f64 {
    let mut w = 0usize;
    for row in body_of(d) {
        let n = row.chars().count();
        if n > w {
            w = n;
        }
    }
    w as f64
}

/// Rows of the repeatable slice, or 0 when the template declares none.
fn rep_rows(d: &Def) -> f64 {
    if d.contains("repeat") {
        d.group_num("repeat", "rows")
    } else {
        0.0
    }
}

/// Highest copy count the repeatable slice may be stamped at.
fn rep_max(d: &Def) -> f64 {
    if d.contains("repeat") {
        group_range(d, "repeat", "times").1
    } else {
        1.0
    }
}

/// Tallest the template can ever be, with the repeatable slice at its maximum.
/// This is the number the placement pass turns into the decorator's declared
/// `reachY`, so it has to be an honest upper bound rather than the authored height.
fn max_h(d: &Def) -> f64 {
    let rows = rep_rows(d);
    body_of(d).len() as f64
        + if rows > 0.0 {
            rows * (rep_max(d) - 1.0)
        } else {
            0.0
        }
}

/// Validate the template's internal consistency, and do it HERE — inside a table
/// callback — because that is the only hook the compiler gives a schema that runs
/// once per def after resolution, and because `TableCtx::bit` already establishes
/// raising a ContentError from one as the way a schema rejects a record it could
/// not reject during field coercion (`Field::check` never runs for `record[]`, and
/// nothing else can see the body and the legend at the same time).
///
/// Called from the first table so it runs exactly once per struct.
fn validate(d: &Def) -> Result<()> {
    let id = d.str_of("id");
    let mut seen: Vec<char> = Vec::new();
    for g in glyphs_of(d) {
        let c = g.str_of("c");
        if c.chars().count() != 1 {
            bail!("struct '{id}': legend glyph '{c}' must be exactly one character");
        }
        let ch = c.chars().next().unwrap();
        if RESERVED.contains(&ch) {
            bail!(
                "struct '{id}': legend glyph '{c}' is reserved ('.' = air, '_' and ' ' = keep terrain)"
            );
        }
        if seen.contains(&ch) {
            bail!("struct '{id}': legend glyph '{c}' declared twice");
        }
        seen.push(ch);
    }

    let body = body_of(d);
    for (r, row) in body.iter().enumerate() {
        for ch in row.chars() {
            if RESERVED.contains(&ch) || seen.contains(&ch) {
                continue;
            }
            bail!(
                "struct '{id}': body row {} uses glyph '{ch}', which no legend row declares",
                r + 1
            );
        }
    }

    if !d.contains("repeat") {
        return Ok(());
    }
    let from = d.group_num("repeat", "from");
    let rows = d.group_num("repeat", "rows");
    let times = group_range(d, "repeat", "times");
    if rows < 1.0 {
        bail!("struct '{id}': repeat.rows must be at least 1");
    }
    if from < 0.0 || from + rows > body.len() as f64 {
        bail!(
            "struct '{id}': repeat slice rows {from}..{} falls outside a {}-row body",
            from + rows - 1.0,
            body.len()
        );
    }
    if times.0 < 1.0 || times.1 < times.0 {
        bail!("struct '{id}': repeat.times must be a range of at least 1");
    }
    Ok(())
}

/// A table callback returns `Option<f64>` and so cannot fail outright, where the
/// TS version simply threw. Stash the rejection in exactly the slot
/// `TableCtx::bit` stashes its own, for the emitter to raise once the pass is
/// done; first error wins, as it does there.
fn reject(ctx: &TableCtx, e: ContentError) {
    let mut slot = ctx.error.borrow_mut();
    if slot.is_none() {
        *slot = Some(e);
    }
}

pub fn schema() -> Schema {
    Schema {
        kind: "struct".into(),
        prefix: "STRUCT".into(),
        iface: "StructDef".into(),
        iface_prefix: "Struct".into(),

        fields: vec![
            (
                "name".into(),
                Field::new("string")
                    .doc("Display name — map markers, debug HUD.")
                    .default_fn(|d| Some(Value::Str(d.str_of("id").to_string()))),
            ),
            // --- placement ----------------------------------------------------
            (
                "place".into(),
                Field::new("enum(surface|shore|floating|underground|cavern|underworld)")
                    .doc(
                        "Which placement lattice offers this template a site. `surface`/`shore`/\
                         `floating` are scanned as COLUMNS (the row is derived from the ground \
                         line); the rest are scanned on a 2D lattice and gated on depth.",
                    )
                    .alias("StructPlace")
                    .map(PLACE)
                    .required()
                    .hot(vec![table(
                        "STRUCT_PLACE",
                        ArrayKind::U8,
                        0.0,
                        "Placement class code.",
                        |d, _ctx| Some(d.num("place")),
                    )]),
            ),
            (
                "rarity".into(),
                Field::new("chance")
                    .doc(
                        "Probability an eligible site actually builds this, applied AFTER the \
                         weighted pick — so `weight` says 'which of these', `rarity` says 'or \
                         nothing at all', and a rare template does not starve its neighbours.",
                    )
                    .default_float(1.0)
                    .hot(vec![table("STRUCT_RARITY", ArrayKind::F32, 0.0, "", |d, _ctx| {
                        Some(d.num("rarity"))
                    })]),
            ),
            (
                "weight".into(),
                Field::new("float")
                    .doc("Relative pick weight among the templates eligible at one site. 0 = never.")
                    .default_float(1.0)
                    .check(|v| match v.as_num() {
                        Some(n) if n < 0.0 => Some("must be >= 0".into()),
                        _ => None,
                    })
                    .hot(vec![table("STRUCT_WEIGHT", ArrayKind::F32, 0.0, "", |d, _ctx| {
                        Some(d.num("weight"))
                    })]),
            ),
            (
                "biomes".into(),
                Field::new("list<string>").doc(
                    "Dominant SURFACE biome ids this may appear in (src/sim/biomes.ts). Absent \
                     = any. Not a `ref(biome)`: biomes are still hand-written, and a forward \
                     reference that fails the build is worse than one that does not.",
                ),
            ),
            (
                "bands".into(),
                Field::new(&format!("list<enum({})>", BANDS.join("|")))
                    .doc("Depth bands this may appear in. Absent = whatever `place` implies.")
                    .default(Value::List(Vec::new()))
                    .hot(vec![table(
                        "STRUCT_BANDS",
                        ArrayKind::U8,
                        0.0,
                        "Packed band bitmask: `STRUCT_BANDS[c] & SBAND.cavern`. 0 = unrestricted.",
                        |d, ctx| {
                            let mut bits: u32 = 0;
                            for b in d.strings_of("bands") {
                                bits |= 1 << (ctx.bit("SBAND", b) as u32);
                            }
                            Some(bits as f64)
                        },
                    )]),
            ),
            (
                "minDepth".into(),
                Field::new("int")
                    .doc(
                        "Cells below the local surface before this may appear. Ignored for surface places.",
                    )
                    .default_int(0)
                    .check(|v| match v.as_num() {
                        Some(n) if n < 0.0 || n > 0xffff as f64 => Some("must be 0..65535".into()),
                        _ => None,
                    })
                    .hot(vec![table("STRUCT_MINDEPTH", ArrayKind::U16, 0.0, "", |d, _ctx| {
                        Some(d.num("minDepth"))
                    })]),
            ),
            (
                "maxDepth".into(),
                Field::new("int")
                    .doc("Last depth this may appear at. 65535 = no bound beyond `bands`.")
                    .default_int(0xffff)
                    .check(|v| match v.as_num() {
                        Some(n) if n < 0.0 || n > 0xffff as f64 => Some("must be 0..65535".into()),
                        _ => None,
                    })
                    .hot(vec![table(
                        "STRUCT_MAXDEPTH",
                        ArrayKind::U16,
                        0xffff as f64,
                        "",
                        |d, _ctx| Some(d.num("maxDepth")),
                    )]),
            ),
            // --- how it sits on the ground -----------------------------------
            (
                "anchor".into(),
                Field::new("enum(bottom_center|bottom_left|center|top_center|top_left)")
                    .doc("Which template cell the placement origin names.")
                    .alias("StructAnchor")
                    .map(ANCHOR)
                    // ANCHOR.bottom_center
                    .default_int(0)
                    .hot(vec![table(
                        "STRUCT_ANCHOR",
                        ArrayKind::U8,
                        0.0,
                        "Anchor code.",
                        |d, _ctx| Some(d.num("anchor")),
                    )]),
            ),
            (
                "sink".into(),
                Field::new("int")
                    .doc(
                        "Rows the whole template is pushed DOWN past its anchor — half-buried ruins.",
                    )
                    .default_int(0)
                    .hot(vec![table("STRUCT_SINK", ArrayKind::U8, 0.0, "", |d, _ctx| {
                        Some(d.num("sink"))
                    })]),
            ),
            (
                "clearance".into(),
                Field::new("int")
                    .doc(
                        "For `floating`: cells of open sky between the template's bottom and the ground.",
                    )
                    .default_int(0)
                    .hot(vec![table("STRUCT_CLEARANCE", ArrayKind::U8, 0.0, "", |d, _ctx| {
                        Some(d.num("clearance"))
                    })]),
            ),
            (
                "flatness".into(),
                Field::new("int")
                    .doc(
                        "Largest ground-line spread, in cells, the footprint tolerates. Nothing is \
                         built on a cliff; a cairn tolerates more than a cabin. Surface places only.",
                    )
                    .default_int(6)
                    .hot(vec![table("STRUCT_FLATNESS", ArrayKind::U8, 0.0, "", |d, _ctx| {
                        Some(d.num("flatness"))
                    })]),
            ),
            (
                "mirror".into(),
                Field::new("bool")
                    .doc(
                        "May be stamped flipped horizontally. Free variety for an asymmetric template.",
                    )
                    .default_bool(true),
            ),
            // --- the template itself ------------------------------------------
            (
                "legend".into(),
                Field::new("record[]")
                    .doc(
                        "One entry per glyph: `{ c = \"W\", block = \"wood\" }`. `.` `_` and space are reserved.",
                    )
                    .default(Value::Records(Vec::new()))
                    .element(
                        "StructGlyph",
                        vec![
                            (
                                "c".into(),
                                Field::new("string")
                                    .doc("The single character used in `body`.")
                                    .required(),
                            ),
                            (
                                "block".into(),
                                Field::new("ref?(block)").doc(
                                    "Block id, or an `a|b|c` preference chain. Absent = air (for a bare marker).",
                                ),
                            ),
                            (
                                "mark".into(),
                                Field::new(&format!(
                                    "enum({})",
                                    MARK.iter().map(|(k, _)| *k).collect::<Vec<_>>().join("|")
                                ))
                                .doc(
                                    "Tags the cell for a later pass: `loot` = a cache goes here, `spawn` = a mob does.",
                                )
                                .alias("StructMark")
                                .map(MARK)
                                // MARK.none
                                .default_int(0),
                            ),
                            (
                                "soft".into(),
                                Field::new("bool")
                                    .doc(
                                        "Only write over open space — for foliage that must not eat masonry.",
                                    )
                                    .default_bool(false),
                            ),
                        ],
                    ),
            ),
            (
                "body".into(),
                Field::new("text")
                    .doc(
                        "The layout, one line per world row, top row first. Short rows are padded \
                         with `keep`. Empty = a tombstone, which the facade skips.",
                    )
                    .default(Value::Text(Vec::new())),
            ),
            (
                "repeat.from".into(),
                Field::new("int")
                    .doc("0-based body row where the repeatable slice starts.")
                    .required(),
            ),
            (
                "repeat.rows".into(),
                Field::new("int").doc("Height of the repeatable slice, in rows.").required(),
            ),
            (
                "repeat.times".into(),
                Field::new("range")
                    .doc("How many copies of the slice to stamp. 1 = exactly as authored.")
                    .required(),
            ),
        ],

        tables: vec![
            table(
                "STRUCT_W",
                ArrayKind::U8,
                0.0,
                "Template width in cells — the widest authored body row.",
                |d, ctx| {
                    // First table in emission order, so this is the once-per-def hook the
                    // schema uses to reject an inconsistent template. See `validate`.
                    if let Err(e) = validate(d) {
                        reject(ctx, e);
                    }
                    Some(body_w(d))
                },
            ),
            table(
                "STRUCT_H",
                ArrayKind::U8,
                0.0,
                "Authored template height in rows.",
                |d, _ctx| Some(body_of(d).len() as f64),
            ),
            table(
                "STRUCT_MAXH",
                ArrayKind::U8,
                0.0,
                "Height with the repeatable slice at `repeat.times` max — the honest vertical \
                 extent the placement pass must declare as its reach.",
                |d, _ctx| Some(max_h(d)),
            ),
            table(
                "STRUCT_REP_FROM",
                ArrayKind::U8,
                0.0,
                "Repeatable slice start row.",
                |d, _ctx| {
                    Some(if d.contains("repeat") { d.group_num("repeat", "from") } else { 0.0 })
                },
            ),
            table(
                "STRUCT_REP_ROWS",
                ArrayKind::U8,
                0.0,
                "Repeatable slice height; 0 = no repeat.",
                |d, _ctx| Some(rep_rows(d)),
            ),
            table("STRUCT_REP_MIN", ArrayKind::U8, 0.0, "Fewest copies of the slice.", |d, _ctx| {
                Some(if d.contains("repeat") { group_range(d, "repeat", "times").0 } else { 1.0 })
            }),
            table("STRUCT_REP_MAX", ArrayKind::U8, 0.0, "Most copies of the slice.", |d, _ctx| {
                Some(rep_max(d))
            }),
            table(
                "STRUCT_FLAGS",
                ArrayKind::U8,
                0.0,
                "Packed template flags: `STRUCT_FLAGS[c] & SFLAG.loot`. Lets the pass answer \
                 'does this template have a cache?' without walking its legend.",
                |d, ctx| {
                    let mut bits: u32 = 0;
                    if d.flag("mirror") {
                        bits |= 1 << (ctx.bit("SFLAG", "mirror") as u32);
                    }
                    for g in glyphs_of(d) {
                        if g.num("mark") == MARK_LOOT {
                            bits |= 1 << (ctx.bit("SFLAG", "loot") as u32);
                        }
                        if g.num("mark") == MARK_SPAWN {
                            bits |= 1 << (ctx.bit("SFLAG", "spawn") as u32);
                        }
                    }
                    Some(bits as f64)
                },
            ),
        ],

        matrices: vec![],

        constants: vec![
            BitConstants {
                name: "SBAND".into(),
                doc: Some(
                    "Depth-band bits for STRUCT_BANDS. Bit order is fixed by tools/contentc/schema/struct.ts."
                        .into(),
                ),
                values: BANDS.iter().map(|s| s.to_string()).collect(),
            },
            BitConstants {
                name: "SFLAG".into(),
                doc: Some("Flag bits for STRUCT_FLAGS.".into()),
                values: FLAGS.iter().map(|s| s.to_string()).collect(),
            },
        ],

        // A retired template keeps its code so a save that recorded "you looted the
        // cabin at (x,y)" still resolves the id. Weight 0 and an empty body mean the
        // placement pass can never pick it and the facade skips it outright.
        tombstone: vec![
            ("name".into(), "\"(removed)\"".into()),
            ("place".into(), "\"surface\"".into()),
            ("rarity".into(), "0".into()),
            ("weight".into(), "0".into()),
        ],
    }
}
