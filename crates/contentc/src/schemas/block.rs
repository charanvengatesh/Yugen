//! The `block` schema — the contract for everything in `content/blocks/*.toml`.
//!
//! This is the file that used to be the `MaterialDef` interface plus the loop at
//! the bottom of `src/sim/materials.ts`. Both are derived from it now: the
//! emitted `BlockDef` struct comes from `fields`, and every flat table comes
//! from a `hot` declaration. A field with `hot` is one a per-cell loop reads;
//! everything else lives only on the def object and never costs the sim
//! anything.
//!
//! Ordering is significant twice over: `default` callbacks see the def built so
//! far (so `collides` can key off `state`, `lightEmit` off `emissive`), and the
//! emitted struct follows this order.

use crate::schema::{ArrayKind, BitConstants, Field, Matrix, NEVER, Schema, table};
use crate::value::{Def, Value};

/// Mirrors `MaterialState` in src/game/types.ts. Codes are load-bearing.
const STATE: &[(&str, i64)] = &[
    ("empty", 0),
    ("solid", 1),
    ("powder", 2),
    ("liquid", 3),
    ("gas", 4),
];

/// The two `STATE` codes `collides` keys off, by value rather than by name.
const STATE_SOLID: f64 = 1.0;
const STATE_POWDER: f64 = 2.0;

/// The tag vocabulary. Declared rather than open-ended so a typo is a compile
/// error, and so bit positions are a stable property of this file rather than of
/// whatever happens to be authored today.
const TAGS: [&str; 11] = [
    "rock",
    "soil",
    "ore",
    "liquid",
    "gas",
    "flora",
    "foliage",
    "ice",
    "hazard",
    "diggable",
    "buildable",
];

/// Per-cell noise patterns the renderer knows how to draw. Closed for the same
/// reason `MobProjectile` is: each name is a branch of real drawing code, so a
/// new one is a code change reviewed against the whole look of the world, not a
/// string a content file can invent and have silently ignored.
///
/// Eight, which is exactly a byte's worth of bits — see MAT_TEXTURE.
const TEXTURES: [&str; 8] = [
    "flat",
    "speckle",
    "grain",
    "crystalline",
    "layered",
    "fibrous",
    "molten",
    "glassy",
];

/// meltAt only takes effect when a product is named — matches the old loop.
fn melt_at(d: &Def) -> Option<f64> {
    d.path("heat", "meltInto")?;
    d.path("heat", "meltAt").and_then(Value::as_num)
}

/// Autoignition only means anything for something that can actually burn.
fn ignite_at(d: &Def) -> Option<f64> {
    if !d.contains("flammable") {
        return None;
    }
    d.path("heat", "igniteAt").and_then(Value::as_num)
}

/// One channel of the required `color` field.
fn channel(d: &Def, i: usize) -> Option<f64> {
    match d.get("color") {
        Some(Value::Color(c)) => Some(c[i] as f64),
        _ => Some(0.0),
    }
}

/// An inclusive numeric bound that keeps the exact message the TS check used.
fn bounded(lo: f64, hi: f64, msg: &'static str) -> impl Fn(&Value) -> Option<String> {
    move |v| {
        let n = v.as_num().unwrap_or(0.0);
        if n < lo || n > hi {
            Some(msg.to_string())
        } else {
            None
        }
    }
}

pub fn schema() -> Schema {
    Schema {
        kind: "block".into(),
        prefix: "BLOCK".into(),
        iface: "BlockDef".into(),
        iface_prefix: "Block".into(),

        fields: vec![
            (
                "name".into(),
                Field::new("string")
                    .doc("Display name (HUD material picker).")
                    .default_fn(|d| Some(Value::Str(d.str_of("id").to_string()))),
            ),
            (
                "state".into(),
                Field::new("enum(empty|solid|powder|liquid|gas)")
                    .doc("Physical class — drives the automata. Value is a `MaterialState`.")
                    .map(STATE)
                    .required()
                    .hot(vec![table(
                        "MAT_STATE",
                        ArrayKind::U8,
                        0.0,
                        "MaterialState value.",
                        |d, _ctx| Some(d.num("state")),
                    )]),
            ),
            (
                "color".into(),
                Field::new("color")
                    .doc("Base colour before the per-cell dither.")
                    .required()
                    .hot(vec![
                        table(
                            "MAT_R",
                            ArrayKind::U8,
                            0.0,
                            "Colour for the renderer's per-cell blit.",
                            |d, _ctx| channel(d, 0),
                        ),
                        table("MAT_G", ArrayKind::U8, 0.0, "", |d, _ctx| channel(d, 1)),
                        table("MAT_B", ArrayKind::U8, 0.0, "", |d, _ctx| channel(d, 2)),
                    ]),
            ),
            (
                "colorVar".into(),
                Field::new("int")
                    .doc("Per-channel random dither amount for cell texture (renderer).")
                    .default_int(12)
                    .check(bounded(0.0, 255.0, "must be 0..255"))
                    .hot(vec![table(
                        "MAT_COLORVAR",
                        ArrayKind::U8,
                        0.0,
                        "",
                        |d, _ctx| Some(d.num("colorVar")),
                    )]),
            ),
            (
                "emissive".into(),
                Field::new("float")
                    .doc("0..1 glow strength for bloom on fire/lava (renderer).")
                    .default_float(0.0)
                    .check(bounded(0.0, 1.0, "must be 0..1"))
                    .hot(vec![table(
                        "MAT_EMISSIVE",
                        ArrayKind::F32,
                        0.0,
                        "0..1 glow.",
                        |d, _ctx| Some(d.num("emissive")),
                    )]),
            ),
            (
                "density".into(),
                Field::new("float")
                    .doc("Relative weight for displacement/buoyancy (liquids & powders).")
                    .default_float(1.0)
                    .hot(vec![table(
                        "MAT_DENSITY",
                        ArrayKind::F32,
                        0.0,
                        "",
                        |d, _ctx| Some(d.num("density")),
                    )]),
            ),
            (
                "collides".into(),
                Field::new("bool")
                    .doc("Does the player collide with this material?")
                    .default_fn(|d| {
                        let state = d.num("state");
                        Some(Value::Bool(state == STATE_SOLID || state == STATE_POWDER))
                    })
                    .hot(vec![table(
                        "MAT_COLLIDE",
                        ArrayKind::U8,
                        0.0,
                        "1 = blocks the player.",
                        |d, _ctx| Some(if d.flag("collides") { 1.0 } else { 0.0 }),
                    )]),
            ),
            (
                "liquidSpread".into(),
                Field::new("int")
                    .doc("Horizontal flow reach per tick, in cells (liquids only).")
                    .default_int(0)
                    .hot(vec![table(
                        "MAT_SPREAD",
                        ArrayKind::U8,
                        0.0,
                        "Liquid horizontal reach.",
                        |d, _ctx| Some(d.num("liquidSpread")),
                    )]),
            ),
            (
                "viscosity".into(),
                Field::new("float")
                    .doc(
                        "How reluctantly a liquid flows, 0..1 (liquids only). 0 -- water \
                         -- moves every tick it can and pays nothing for this field \
                         existing. Above 0, each tick the liquid skips its whole move \
                         with this probability (staying AWAKE, so it oozes rather than \
                         freezes) and its sideways reach shrinks by the same fraction: \
                         a viscous liquid falls late, pools tall, and levels slowly.",
                    )
                    .default_float(0.0)
                    .check(|v| {
                        let n = v.as_num().unwrap_or(-1.0);
                        if (0.0..=1.0).contains(&n) {
                            None
                        } else {
                            Some("must be 0..1".to_string())
                        }
                    })
                    .hot(vec![table(
                        "MAT_VISCOSITY",
                        ArrayKind::F32,
                        0.0,
                        "Liquid flow reluctance, 0..1.",
                        |d, _ctx| Some(d.num("viscosity")),
                    )]),
            ),
            // --- Surface appearance -------------------------------------------------
            // Three knobs the renderer's per-cell pass reads on EVERY visible cell,
            // which is why all three are hot and all three are one byte. `colorVar`
            // above already scatters a cell's colour; these say what that scatter
            // should look like, whether the cell pulses, and how hard its silhouette
            // edge is drawn. Together they are what stops forty materials that differ
            // only in RGB from reading as forty shades of the same substance.
            (
                "texture".into(),
                Field::new(&format!("enum({})", TEXTURES.join("|")))
                    .doc(
                        "Which per-cell noise pattern the renderer applies. Names a LOOK, not a \
                         material: sandstone and strata rock are both `layered` because they are \
                         drawn the same way, and nothing in the sim reads this.",
                    )
                    .default_str("flat")
                    .alias("BlockTexture")
                    .hot(vec![table(
                        "MAT_TEXTURE",
                        ArrayKind::U8,
                        0.0,
                        "Texture pattern, stored as its `TEX` bit VALUE (not its index), so \
                         `MAT_TEXTURE[c] === TEX.grain` and `MAT_TEXTURE[c] & TEX.grain` both \
                         work — the first for a per-cell dispatch, the second for a 'is this \
                         any of these three patterns' test with no branch.",
                        |d, ctx| {
                            Some((1u32 << (ctx.bit("TEX", d.str_of("texture")) as u32)) as f64)
                        },
                    )]),
            ),
            (
                "shimmer".into(),
                Field::new("int")
                    .doc(
                        "0..255 animated emissive strength — how much the cell PULSES, as opposed \
                         to `emissive`, which is how much it constantly glows. Kept separate \
                         because lava wants both and a crystal wants only this one: a gem that \
                         brightened the room would light caves it is merely embedded in.",
                    )
                    .default_int(0)
                    .check(bounded(0.0, 255.0, "must be 0..255"))
                    .hot(vec![table(
                        "MAT_SHIMMER",
                        ArrayKind::U8,
                        0.0,
                        "0..255 animated emissive strength.",
                        |d, _ctx| Some(d.num("shimmer")),
                    )]),
            ),
            (
                "edge".into(),
                Field::new("int")
                    .doc(
                        "0..255 rim-highlight strength on the boundary with a different material. \
                         High on hard angular rock, where a crisp lit edge is what makes cut \
                         stone read as cut; low on powders, whose whole character is that they \
                         have no edge to catch the light.",
                    )
                    .default_int(0)
                    .check(bounded(0.0, 255.0, "must be 0..255"))
                    .hot(vec![table(
                        "MAT_EDGE",
                        ArrayKind::U8,
                        0.0,
                        "0..255 rim-highlight strength.",
                        |d, _ctx| Some(d.num("edge")),
                    )]),
            ),
            (
                "surface".into(),
                Field::new("enum(ice|bounce|conveyorL|conveyorR|sticky)")
                    .doc("Player-contact effect on a solid surface.")
                    .alias("BlockSurface"),
            ),
            // --- Traversal and interaction ------------------------------------------
            // Three booleans that all answer "what does the player's body do when it
            // reaches this cell", and all three are deliberately ORTHOGONAL to
            // `collides` rather than modes of it.
            //
            // `collides` is consumed as one flat truth — `MAT_COLLIDE[c] === 1` is the
            // whole `solid()` predicate in src/physics/collision.ts, and the AABB
            // resolver pushes the body out of any such cell from every side. A ladder or
            // a platform whose `collides` were true would therefore be a wall first and
            // a ladder second, and no amount of climb/one-way code downstream could
            // recover the cell the resolver already ejected the player from. So
            // climbable and one-way materials author `collides false` and these tables
            // are what re-adds the *specific* interaction that material wants.
            //
            // None of the three carries a `default`, matching `damage` below: a bool
            // that defaults to false would be materialised onto all fifty-odd defs as
            // dead weight, when what every consumer actually reads is the flat table —
            // which `fill: 0` already covers, tombstones included.
            (
                "climb".into(),
                Field::new("bool")
                    .doc(
                        "1 = the player can climb this cell (ladder, rope, vine). Implies the \
                         cell is NOT collidable — see the note above.",
                    )
                    .hot(vec![table(
                        "MAT_CLIMB",
                        ArrayKind::U8,
                        0.0,
                        "1 = climbable by the player.",
                        |d, _ctx| Some(if d.flag("climb") { 1.0 } else { 0.0 }),
                    )]),
            ),
            (
                "oneWay".into(),
                Field::new("bool")
                    .doc(
                        "1 = collides only from above: the player lands on it but walks and \
                         jumps up through it (platform). Kept off `surface`, which is a contact \
                         EFFECT on a solid; this changes whether the contact happens at all.",
                    )
                    .hot(vec![table(
                        "MAT_ONEWAY",
                        ArrayKind::U8,
                        0.0,
                        "1 = collides only from above.",
                        |d, _ctx| Some(if d.flag("oneWay") { 1.0 } else { 0.0 }),
                    )]),
            ),
            (
                "container".into(),
                Field::new("bool")
                    .doc(
                        "1 = opening this yields loot (chest). Unlike the other two this one IS \
                         usually solid — furniture you bump into — so it says nothing about \
                         collision, only that the use verb has somewhere to land.",
                    )
                    .hot(vec![table(
                        "MAT_CONTAINER",
                        ArrayKind::U8,
                        0.0,
                        "1 = opening yields loot.",
                        |d, _ctx| Some(if d.flag("container") { 1.0 } else { 0.0 }),
                    )]),
            ),
            (
                "damage".into(),
                Field::new("float")
                    .doc("hp/s while the player overlaps (spike, lava).")
                    .hot(vec![table(
                        "MAT_DAMAGE",
                        ArrayKind::F32,
                        0.0,
                        "hp/s on overlap.",
                        |d, _ctx| Some(d.num("damage")),
                    )]),
            ),
            // --- Fire ---------------------------------------------------------------
            (
                "flammable.igniteChance".into(),
                Field::new("chance")
                    .doc("Chance per tick to ignite when touching fire.")
                    .required()
                    .hot(vec![table(
                        "MAT_IGNITE",
                        ArrayKind::F32,
                        0.0,
                        "Ignite chance/tick.",
                        |d, _ctx| {
                            if d.contains("flammable") {
                                Some(d.group_num("flammable", "igniteChance"))
                            } else {
                                None
                            }
                        },
                    )]),
            ),
            (
                "flammable.burnTime".into(),
                Field::new("int")
                    .doc("Ticks the cell burns before it is consumed.")
                    .required()
                    .hot(vec![table(
                        "MAT_BURNTIME",
                        ArrayKind::U16,
                        0.0,
                        "Ticks to burn out.",
                        |d, _ctx| {
                            if d.contains("flammable") {
                                Some(d.group_num("flammable", "burnTime"))
                            } else {
                                None
                            }
                        },
                    )]),
            ),
            (
                "flammable.burnsInto".into(),
                Field::new("ref(block)")
                    .doc("Block the cell becomes once burnt out (\"smoke\", \"empty\", ...).")
                    .required()
                    .hot(vec![table(
                        "MAT_BURNINTO",
                        ArrayKind::U16,
                        0.0,
                        "Resulting material code.",
                        |d, ctx| {
                            if d.contains("flammable") {
                                Some(ctx.code(d.group_str("flammable", "burnsInto")))
                            } else {
                                None
                            }
                        },
                    )]),
            ),
            // --- Heat field ---------------------------------------------------------
            (
                "heat.emit".into(),
                Field::new("int")
                    .doc(
                        "Floor this material holds itself at each tick (lava/fire/ember). \
                         0 = inert.",
                    )
                    .hot(vec![table(
                        "MAT_HEATEMIT",
                        ArrayKind::U8,
                        0.0,
                        "Temperature floor the material holds itself at (0 = not a heat source).",
                        |d, _ctx| d.path("heat", "emit").and_then(Value::as_num),
                    )]),
            ),
            (
                "heat.conduct".into(),
                Field::new("int")
                    .doc(
                        "Explicit-diffusion coefficient times 256. 0 makes a perfect insulator. \
                         The scheme is unconditionally stable up to 64 (alpha <= 1/4); the metal \
                         ores deliberately sit above it and rely on the clamped diffusion step.",
                    )
                    .check(bounded(0.0, 255.0, "must fit a byte (0..255)"))
                    .hot(vec![table(
                        "MAT_CONDUCT",
                        ArrayKind::U8,
                        34.0,
                        "Diffusion weight (0 = insulator). Default: mildly conductive.",
                        |d, _ctx| d.path("heat", "conduct").and_then(Value::as_num),
                    )]),
            ),
            (
                "heat.cool".into(),
                Field::new("int")
                    .doc("Degrees shed per tick toward ambient. Sinks (ice/snow) use a big value.")
                    .hot(vec![table(
                        "MAT_COOL",
                        ArrayKind::U8,
                        1.0,
                        "Degrees shed per tick toward ambient — the \"sink\" knob.",
                        |d, _ctx| d.path("heat", "cool").and_then(Value::as_num),
                    )]),
            ),
            (
                "heat.meltAt".into(),
                Field::new("int")
                    .doc("At/above this temperature the cell turns into `meltInto`.")
                    .hot(vec![table(
                        "MAT_MELTAT",
                        ArrayKind::U16,
                        0xffff as f64,
                        "Melt/boil threshold (NEVER = this material never melts).",
                        |d, _ctx| melt_at(d),
                    )]),
            ),
            (
                "heat.meltInto".into(),
                Field::new("ref(block)")
                    .doc("Block produced by melting/boiling.")
                    .hot(vec![table(
                        "MAT_MELTINTO",
                        ArrayKind::U16,
                        0.0,
                        "Melt/boil product.",
                        |d, ctx| {
                            if d.path("heat", "meltAt").is_none()
                                || d.path("heat", "meltInto").is_none()
                            {
                                return None;
                            }
                            Some(ctx.code(d.group_str("heat", "meltInto")))
                        },
                    )]),
            ),
            (
                "heat.igniteAt".into(),
                Field::new("int")
                    .doc(
                        "At/above this temperature a flammable cell autoignites (no flame \
                         contact).",
                    )
                    .hot(vec![table(
                        "MAT_IGNITEAT",
                        ArrayKind::U16,
                        0xffff as f64,
                        "Autoignition threshold (NEVER = only ignites by contact).",
                        |d, _ctx| ignite_at(d),
                    )]),
            ),
            // --- Growth -------------------------------------------------------------
            (
                "growth.chance".into(),
                Field::new("chance")
                    .doc("Per-tick chance to convert one eligible neighbour.")
                    .required()
                    .hot(vec![table(
                        "MAT_GROWCHANCE",
                        ArrayKind::F32,
                        0.0,
                        "",
                        |d, _ctx| {
                            if d.contains("growth") {
                                Some(d.group_num("growth", "chance"))
                            } else {
                                None
                            }
                        },
                    )]),
            ),
            (
                "growth.into".into(),
                Field::new("ref(block)")
                    .doc("Block the converted neighbour becomes. Defaults to the grower itself.")
                    .hot(vec![table(
                        "MAT_GROWINTO",
                        ArrayKind::U16,
                        0.0,
                        "Block a converted neighbour becomes.",
                        |d, ctx| {
                            if !d.contains("growth") {
                                return None;
                            }
                            let into = match d.path("growth", "into") {
                                Some(Value::Str(s)) => s.as_str(),
                                _ => d.str_of("id"),
                            };
                            Some(ctx.code(into))
                        },
                    )]),
            ),
            (
                "growth.onto".into(),
                Field::new("list<ref(block)>")
                    .doc("Blocks this can spread into (\"empty\" allowed).")
                    .required(),
            ),
            (
                "growth.downward".into(),
                Field::new("bool")
                    .doc("Only grow into the cell directly below (vines). Default: any orthogonal.")
                    .hot(vec![table(
                        "MAT_GROWDOWN",
                        ArrayKind::U8,
                        0.0,
                        "1 = only grows straight down (vines).",
                        |d, _ctx| {
                            if !d.contains("growth") {
                                return None;
                            }
                            let down =
                                matches!(d.path("growth", "downward"), Some(Value::Bool(true)));
                            Some(if down { 1.0 } else { 0.0 })
                        },
                    )]),
            ),
            (
                "growth.maxGen".into(),
                Field::new("int")
                    .doc("Generations from the original seed before the lineage stops spreading.")
                    .required()
                    .hot(vec![table(
                        "MAT_GROWMAX",
                        ArrayKind::U8,
                        0.0,
                        "Generations from a seed cell before the lineage stops spreading.",
                        |d, _ctx| {
                            if d.contains("growth") {
                                Some(d.group_num("growth", "maxGen"))
                            } else {
                                None
                            }
                        },
                    )]),
            ),
            // --- Mining / world interaction -----------------------------------------
            (
                "hardness".into(),
                Field::new("float")
                    .doc("Dig time multiplier. Infinity (or -1) means unbreakable.")
                    .default_float(1.0)
                    .hot(vec![table(
                        "MAT_HARDNESS",
                        ArrayKind::F32,
                        0.0,
                        "Dig time multiplier; Infinity = unbreakable.",
                        |d, _ctx| {
                            let h = d.num("hardness");
                            Some(if h < 0.0 { f64::INFINITY } else { h })
                        },
                    )]),
            ),
            (
                "blastResist".into(),
                Field::new("float")
                    .doc(
                        "Explosion resistance multiplier; scales the radius an explosion eats \
                         through.",
                    )
                    .default_float(1.0)
                    .hot(vec![table(
                        "MAT_BLASTRESIST",
                        ArrayKind::F32,
                        0.0,
                        "",
                        |d, _ctx| Some(d.num("blastResist")),
                    )]),
            ),
            (
                "lightEmit".into(),
                Field::new("int")
                    .doc(
                        "0..15 light level this block radiates. Defaults from `emissive`, which \
                         is the renderer's bloom knob — the two describe the same glow at \
                         different resolutions, so authoring one is normally enough.",
                    )
                    .default_fn(|d| Some(Value::Int((d.num("emissive") * 12.0).round() as i64)))
                    .check(bounded(0.0, 15.0, "must be 0..15"))
                    .hot(vec![table(
                        "MAT_LIGHT",
                        ArrayKind::U8,
                        0.0,
                        "0..15 emitted light level.",
                        |d, _ctx| Some(d.num("lightEmit")),
                    )]),
            ),
            (
                "drop".into(),
                Field::new("record[]")
                    .doc("What mining this yields. Absent means the block drops its own item.")
                    .element(
                        "BlockDrop",
                        vec![
                            // The item registry does not exist yet, so unknown ids warn rather
                            // than fail the build; the ref is re-checked for real once items
                            // land.
                            (
                                "item".into(),
                                Field::new("ref(item)")
                                    .doc("Item id. Unvalidated until the item registry exists.")
                                    .required()
                                    .lenient(),
                            ),
                            (
                                "count".into(),
                                Field::new("range")
                                    .doc("Inclusive stack size range.")
                                    .default(Value::Range(1.0, 1.0)),
                            ),
                            (
                                "chance".into(),
                                Field::new("chance")
                                    .doc("Probability this entry drops at all.")
                                    .default_float(1.0),
                            ),
                        ],
                    ),
            ),
            (
                "tags".into(),
                Field::new(&format!("list<enum({})>", TAGS.join("|")))
                    .doc("Classification bits. Also packed into MAT_TAGS for hot-path tests.")
                    .default(Value::List(vec![])),
            ),
        ],

        // Tables that read a group's presence or more than one field, so they cannot
        // hang off a single field declaration.
        tables: vec![
            table(
                "MAT_FLAMMABLE",
                ArrayKind::U8,
                0.0,
                "1 = can ignite.",
                |d, _ctx| Some(if d.contains("flammable") { 1.0 } else { 0.0 }),
            ),
            table(
                "MAT_HEATTHRESH",
                ArrayKind::U16,
                0xffff as f64,
                "min(meltAt, igniteAt) — the ONE value the hot loop compares a cell's \
                 temperature against. Below it no thermal transition is possible, so the \
                 common (cold / inert) case costs a single array read and a compare.",
                |d, _ctx| {
                    if !d.contains("heat") {
                        return None;
                    }
                    Some(
                        melt_at(d)
                            .unwrap_or(NEVER)
                            .min(ignite_at(d).unwrap_or(NEVER)),
                    )
                },
            ),
            table(
                "MAT_GROWS",
                ArrayKind::U8,
                0.0,
                "1 = this material spreads (single-read early-out in the hot loop).",
                |d, _ctx| Some(if d.contains("growth") { 1.0 } else { 0.0 }),
            ),
            table(
                "MAT_TAGS",
                ArrayKind::U32,
                0.0,
                "Packed tag bitmask: `MAT_TAGS[c] & TAG.rock`.",
                |d, ctx| Some(ctx.mask("TAG", &d.strings_of("tags"))),
            ),
        ],

        matrices: vec![Matrix {
            name: "GROW_ONTO".into(),
            doc: Some(
                "Substrate matrix: GROW_ONTO[grower * BLOCK_COUNT + target] === 1 when \
                 `grower` may convert a `target` cell. Flat so the check is one indexed read."
                    .into(),
            ),
            row: Box::new(|d, _ctx| match d.path("growth", "onto") {
                Some(Value::List(items)) => Some(
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect(),
                ),
                _ => None,
            }),
        }],

        constants: vec![
            BitConstants {
                name: "TAG".into(),
                doc: Some(
                    "Tag bits for MAT_TAGS. Bit order is fixed by tools/contentc/schema/block.ts."
                        .into(),
                ),
                values: TAGS.iter().map(|s| s.to_string()).collect(),
            },
            BitConstants {
                name: "TEX".into(),
                doc: Some(
                    "Texture pattern values for MAT_TEXTURE. One bit each, so a value is also \
                     a mask: `TEX.grain | TEX.speckle` is a legal 'either of these' test. Bit \
                     order is fixed by tools/contentc/schema/block.ts."
                        .into(),
                ),
                values: TEXTURES.iter().map(|s| s.to_string()).collect(),
            },
        ],

        // A retired id keeps its code forever so old chunks still load; the def is a
        // loud magenta nothing, which is exactly what you want to see if one is ever
        // actually rendered.
        tombstone: vec![
            ("name".into(), "\"(removed)\"".into()),
            ("state".into(), "\"empty\"".into()),
            ("color".into(), "[255, 0, 255]".into()),
            ("collides".into(), "false".into()),
        ],
    }
}
