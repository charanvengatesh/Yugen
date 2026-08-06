//! The `item` schema — the contract for `content/items/*.toml`.
//!
//! An item is what a block turns INTO when you break it, what you spend to put a
//! block back, and what decides how deep you can break in the first place. That
//! makes this schema the other half of `block.rs`: `block.drop` names an id from
//! here, and `item.places` names an id from there. The compiler parses both kinds
//! before compiling either (see the driver), so the two references validate in
//! both directions with no ordering rule to remember.
//!
//! WHAT IS HOT, AND WHY SO LITTLE. `hot` buys a flat typed array indexed by
//! code. That is only worth it when a loop indexes it with a VARYING code. Three
//! fields qualify:
//!
//!   stack   Inventory.add walks slots looking for a partial stack of some code;
//!           the cap it compares against changes per slot. Genuinely indexed.
//!   tags    Same argument as MAT_TAGS: one masked read instead of a string scan.
//!
//! A third, `places`, wants to be here and cannot be — see the note on the field.
//!
//! Everything else — digPower, digSpeed, reach, damage, heal, colour — is read
//! ONCE per frame off the single held item, or once per swing, or once per HUD
//! repaint. A flat table for those would be pure module weight, so they live on
//! the def and nowhere else. Adding a table nobody indexes is how a generated
//! module quietly doubles in size.
//!
//! CRAFTING LIVES HERE, on the output item, as the `craft` record[] field. The
//! `recipe` kind FORMAT.md's directory table would need would be a row this
//! schema does not own; putting the recipe on the thing it produces costs one
//! field, keeps the ingredient list under the same compile-time `ref(item)`
//! validation, and has the pleasant property that a product and its recipe can
//! never drift into separate files.

use crate::schema::{ArrayKind, BitConstants, Field, Schema, table};
use crate::value::Value;

/// What an item fundamentally IS, which decides how the inventory uses it.
const CATEGORIES: [&str; 6] = [
    "material",
    "placeable",
    "tool",
    "weapon",
    "consumable",
    "accessory",
];

/// The tag vocabulary, declared rather than open-ended for the same reason
/// block.rs declares its own: a typo becomes a compile error, and bit positions
/// are a property of THIS file rather than of whatever happens to be authored
/// today. Tags answer set questions that `category` cannot — "is this fuel", "is
/// this a pickaxe" — without a second enum per question.
const TAGS: [&str; 17] = [
    "ore",
    "bar",
    "gem",
    "stone",
    "soil",
    "wood",
    "plant",
    "ice",
    "fuel",
    "pickaxe",
    "sword",
    "food",
    "relic",
    "buildable",
    // Appended, never inserted: a tag's bit position is its index here, and
    // ITEM_TAGS is already baked into a generated module. Adding to the end is
    // additive; adding to the middle silently re-labels every existing bit.
    "bow",
    "ammo",
    "light",
];

/// Crafting stations. `hand` means anywhere, which is where this pass stops.
const STATIONS: [&str; 4] = ["hand", "workbench", "furnace", "anvil"];

/// Effects a consumable can apply. Names are the contract with the buff code.
const EFFECTS: [&str; 5] = ["none", "regen", "fireward", "haste", "light"];

pub fn schema() -> Schema {
    Schema {
        kind: "item".into(),
        prefix: "ITEM".into(),
        iface: "ItemDef".into(),
        iface_prefix: "Item".into(),

        fields: vec![
            (
                "name".into(),
                Field::new("string")
                    .doc("Display name (inventory, tooltips).")
                    .required(),
            ),
            (
                "category".into(),
                Field::new(&format!("enum({})", CATEGORIES.join("|")))
                    .doc("Decides which systems act on this item.")
                    .required()
                    .alias("ItemCategory"),
            ),
            (
                "color".into(),
                Field::new("color")
                    .doc("Icon tint until real item art exists.")
                    .default(Value::Color([200, 200, 200])),
            ),
            // NO HOT TABLE, deliberately — an `ITEM_ICON: [u16]` would be indexed by
            // the ten hotbar slots once per HUD repaint and once per drop spawn, and
            // that is not a varying-code inner loop. It is precisely the case the
            // header of this file rules out, and adding the table would be one more
            // generated array nobody reads.
            //
            // `ref?` is a preference chain and `lenient`, matching `places`: sprites
            // are authored in a later pass than the items that name them, so an id
            // that does not exist yet warns and degrades to the flat colour swatch
            // instead of failing the build. Content stays authorable ahead of the art.
            (
                "icon".into(),
                Field::new("ref?(sprite)").lenient().doc(
                    "Icon art. Absent = the flat `color` swatch, which is what all 66 items had \
                     before art existed.",
                ),
            ),
            (
                "stack".into(),
                Field::new("int")
                    .doc("Max held per inventory slot.")
                    // Slot counts are stored in a u16 array, and 999 is already absurd.
                    .default_int(99)
                    .check(|v| {
                        let n = v.as_num().unwrap_or(0.0);
                        if (1.0..=999.0).contains(&n) {
                            None
                        } else {
                            Some("must be 1..999".into())
                        }
                    })
                    .hot(vec![table(
                        "ITEM_STACK",
                        ArrayKind::U16,
                        1.0,
                        "Max per slot.",
                        |d, _ctx| Some(d.num("stack")),
                    )]),
            ),
            // `ref?` is a preference chain, so an item may name a block that does not
            // exist yet without breaking the build.
            //
            // The table is a CROSS-KIND lookup — it holds BLOCK codes, indexed by ITEM
            // code — so it goes through `ctx.foreign_code` rather than `ctx.code`,
            // which resolves against this schema's own kind and would hand back 0
            // (air) for every entry. The driver assigns every kind's codes before
            // emitting any of them, which is what makes that well-defined here.
            (
                "places".into(),
                Field::new("ref?(block)")
                    .doc("Block this places when used. Only meaningful for category=placeable.")
                    .lenient()
                    .hot(vec![table(
                        "ITEM_PLACES",
                        ArrayKind::U16,
                        0.0,
                        "Block code this item places; 0 (air) means it places nothing, which is \
                         also the cheapest 'is this placeable' test there is.",
                        // `lenient` means an id that no longer exists is content that
                        // degraded, not a crash: foreign_code returns 0 and the item
                        // places nothing.
                        |d, ctx| match d.get("places") {
                            Some(Value::Str(id)) => Some(ctx.foreign_code("block", id)),
                            _ => None,
                        },
                    )]),
            ),
            (
                "tier".into(),
                Field::new("int")
                    .doc("Progression tier, 0 = starting gear.")
                    .default_int(0),
            ),
            (
                "value".into(),
                Field::new("int")
                    .doc("Trade value in coins. Also the loot-quality knob.")
                    .default_int(0),
            ),
            (
                "desc".into(),
                Field::new("string")
                    .doc("One line of flavour, shown under the name.")
                    .default_str(""),
            ),
            // --- Tools ---------------------------------------------------------------
            // The group is present iff the item is a tool, exactly like `flammable` on a
            // block: the presence of `tool` is the whole "is this a pickaxe" test at
            // emit time.
            (
                "tool.digPower".into(),
                Field::new("float")
                    .doc(
                        "Highest MAT_HARDNESS this tool can break. A cell whose hardness exceeds \
                         it does not break at all — this is the ore gate, not a speed penalty.",
                    )
                    .required(),
            ),
            (
                "tool.digSpeed".into(),
                Field::new("float")
                    .doc("Multiplier on the dig cadence. 1 = the base interval in BuildTool.")
                    .default_float(1.0),
            ),
            (
                "tool.reach".into(),
                Field::new("int")
                    .doc("How far from the player, in cells, the cursor may act.")
                    .default_int(6),
            ),
            (
                "tool.brushMax".into(),
                Field::new("int")
                    .doc("Largest brush RADIUS in cells. Bites scale with the tool, not the mouse.")
                    .default_int(1),
            ),
            // --- Weapons -------------------------------------------------------------
            //
            // RANGED IS A MODE OF THE SAME GROUP, NOT A SECOND ONE. A bow and a sword
            // both have damage, knockback and a commitment time; a bow additionally
            // launches something. Splitting that into a parallel `ranged.` group would
            // duplicate three fields and force every consumer to ask which group to
            // read before it could ask anything useful. `weapon.ranged` is the one bit
            // that decides, and it is defaulted (not optional) precisely so the melee
            // path reads `def.weapon.ranged == false` rather than `unwrap_or(false)`.
            //
            // Note the asymmetry with `mob.ranged.projectile`: a creature's projectile
            // is a closed enum because its speed, size and on-hit rider live in a table
            // in the mob system. A player weapon carries its own `projectileSpeed` and
            // names its own ammo item, so there is nothing left for an enum to select.
            (
                "weapon.damage".into(),
                Field::new("float")
                    .doc(
                        "Hit points removed per connecting swing. On a RANGED weapon this is the \
                         launcher's contribution to the shot; see `ammo` for how the two add.",
                    )
                    .required(),
            ),
            (
                "weapon.knockback".into(),
                Field::new("float")
                    .doc("Impulse in px/s applied along the swing.")
                    .default_float(120.0),
            ),
            (
                "weapon.swingTime".into(),
                Field::new("float")
                    .doc(
                        "Seconds between swings. The real DPS knob. On a ranged weapon it is the \
                         draw: the interval between shots, measured the same way.",
                    )
                    .default_float(0.4),
            ),
            (
                "weapon.ranged".into(),
                Field::new("bool")
                    .doc(
                        "Does using this launch a projectile instead of sweeping an arc? \
                         It carries a `default` rather than being left unset, so every item \
                         that has a weapon group at all has this key MATERIALISED on it — a \
                         sword emits `ranged: false` rather than nothing. The emitter still \
                         types every group sub-field optional (same as `knockback`), but the \
                         value is never actually absent, so `w.ranged` is a real boolean at \
                         runtime and the melee path never has to distinguish false from unset.",
                    )
                    .default_bool(false),
            ),
            (
                "weapon.projectileSpeed".into(),
                Field::new("float")
                    .doc(
                        "Launch speed in world px/s. Authored in the PLAYER's frame like every \
                         other distance-bearing number in content (src/config/physics.ts: \
                         MAX_RUN_SPEED 360), so 520 is 'clearly faster than you can run'. 0 on a \
                         melee weapon, where it means nothing.",
                    )
                    .default_float(0.0),
            ),
            // A LIST, not a `ref?` chain, and the difference is the whole design.
            //
            // A `ref?` preference chain (`arrow_iron|arrow`) is resolved by the
            // COMPILER — FORMAT.md §3 is explicit that it "takes the first id that
            // exists" — so it collapses to a single string in the emitted def and the
            // second candidate is gone before the game ever runs. That is exactly right
            // for what chains are for (content that names something a later pass will
            // add, degrading instead of breaking the build) and exactly wrong here,
            // where both candidates always exist and the choice between them is a
            // runtime question about what is in the quiver.
            //
            // So this is an ordered list and the runtime does the picking: the first
            // entry the player is actually holding is the one spent. One bow spans
            // flint and iron arrows, iron gets used up first because it is listed
            // first, and nothing has to be swapped by hand.
            (
                "weapon.ammo".into(),
                Field::new("list<ref(item)>").lenient().doc(
                    "Items this can be loaded with, in PREFERENCE order — the runtime spends \
                     the first entry the player is holding. Absent = needs no ammunition. \
                     This list is also the whole acceptance test: a launcher fires what it \
                     names and nothing else, which is why the sling takes `stone_chunk` and \
                     cannot be fed arrows. The chosen ammo item may itself declare a \
                     `weapon.damage`, and the contract is ADDITIVE: shot damage = launcher \
                     `weapon.damage` + ammo `weapon.damage`, the second term being 0 when the \
                     ammo declares no weapon group at all (as `stone_chunk` does not).",
                ),
            ),
            // --- Consumables ---------------------------------------------------------
            (
                "use.heal".into(),
                Field::new("float")
                    .doc("Hit points restored on use.")
                    .default_float(0.0),
            ),
            (
                "use.effect".into(),
                Field::new(&format!("enum({})", EFFECTS.join("|")))
                    .doc("Status applied on use. Names are the contract with the buff system.")
                    .default_str("none"),
            ),
            (
                "use.duration".into(),
                Field::new("float")
                    .doc("Seconds the effect lasts. Meaningless without `effect`.")
                    .default_float(0.0),
            ),
            // --- Crafting ------------------------------------------------------------
            (
                "craft".into(),
                Field::new("record[]")
                    .doc(
                        "Recipes producing THIS item. One entry per alternate route (ore or bar, \
                         workbench or anvil). Absent = not craftable.",
                    )
                    .element(
                        "ItemCraft",
                        vec![
                            // A `record[]` element is one line of `key=value`, so it cannot
                            // nest an (id, count) pair list. Ingredients are therefore two
                            // parallel lists: `in` carries the ids (and gets full ref(item)
                            // validation), `n` the counts. The registry pairs them and treats
                            // a missing count as 1, so a short `n` degrades instead of
                            // desyncing.
                            (
                                "in".into(),
                                Field::new("list<ref(item)>")
                                    .doc("Ingredient item ids.")
                                    .required(),
                            ),
                            (
                                "n".into(),
                                Field::new("list<int>")
                                    .doc("Count per ingredient, positionally. Missing = 1.")
                                    .default(Value::List(vec![])),
                            ),
                            (
                                "out".into(),
                                Field::new("int")
                                    .doc("How many of this item one craft yields.")
                                    .default_int(1),
                            ),
                            (
                                "station".into(),
                                Field::new(&format!("enum({})", STATIONS.join("|")))
                                    .doc("Where it can be made. `hand` = anywhere.")
                                    .default_str("hand"),
                            ),
                        ],
                    ),
            ),
            (
                "tags".into(),
                Field::new(&format!("list<enum({})>", TAGS.join("|")))
                    .doc("Classification bits. Also packed into ITEM_TAGS for masked tests.")
                    .default(Value::List(vec![])),
            ),
        ],

        tables: vec![table(
            "ITEM_TAGS",
            ArrayKind::U32,
            0.0,
            "Packed tag bitmask: `ITEM_TAGS[c] & ITEM_TAG.fuel`.",
            |d, ctx| Some(ctx.mask("ITEM_TAG", &d.strings_of("tags"))),
        )],

        matrices: vec![],

        constants: vec![BitConstants {
            name: "ITEM_TAG".into(),
            doc: Some(
                "Tag bits for ITEM_TAGS. Bit order is fixed by tools/contentc/schema/item.ts."
                    .into(),
            ),
            values: TAGS.iter().map(|s| s.to_string()).collect(),
        }],

        // A retired id keeps its code forever so old saves and old block drop tables
        // still resolve; the def is a loud magenta nothing that stacks alone.
        tombstone: vec![
            ("name".into(), "\"(removed)\"".into()),
            ("category".into(), "\"material\"".into()),
            ("color".into(), "[255, 0, 255]".into()),
            ("stack".into(), "1".into()),
        ],
    }
}
