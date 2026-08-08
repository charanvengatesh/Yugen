//! The `sprite` schema, and the reusable art fields every host splices in.
//!
//! There are two hosts for pixel art in this game and there will be more: the
//! player (a standalone `sprite` record), and every creature (an `art.` group
//! inside a `mob`). They want the SAME fields — a cell grid, a palette, a playback rate,
//! a list of named sequences — and a different pose vocabulary. Duplicating the
//! field list per host is how the two quietly diverge: someone adds `variants` to
//! one and not the other, and six months later half the art pipeline supports
//! tinting.
//!
//! So the fields are a function. `sprite_art_fields()` returns the same
//! `Vec<(String, Field)>` under any prefix, with the host choosing only the
//! things that genuinely differ:
//!
//! ```text
//!   prefix         ""      for a top-level sprite, "art." for a mob group
//!   states         the CLOSED pose vocabulary — a typo is a compile error
//!   ts_alias       what the emitted union is called: AnimState / MobPose
//!   ts_element     what the emitted element interface is called
//!   default_state  set when one unnamed sequence is the normal case (icons)
//! ```
//!
//! ---- WHY A CLOSED VOCABULARY PER HOST --------------------------------------
//! The player has twelve states because its movement code can produce twelve; a
//! creature has three because its brains produce three. Neither list is a subset
//! of the other in any useful way, and merging them would mean a mob could
//! legally author a `wallSlide` pose that nothing will ever select. The union is
//! generated from `states`, so the type the game codes against and the text the
//! artist writes are the same list by construction — `AnimState` cannot drift
//! from `content/sprites/player.sprite` because one is emitted from the other.
//!
//! ---- WHAT IS HOT, AND WHY NOTHING IS ---------------------------------------
//! Nothing here declares `hot`, deliberately. A flat typed array indexed by code
//! only pays for itself when a loop indexes it with a VARYING code every frame
//! (see the note at the top of schema/item.ts). Sprite art is read once, at
//! bake time, to rasterise canvases — after that the runtime holds canvases and
//! never looks at this data again. A `SPRITE_CELLSW` table would be module weight
//! that no inner loop ever reads.
//!
//! ---- ONE ART PIXEL IS ONE WORLD CELL ---------------------------------------
//! `cellsW`/`cellsH` are in CELLS, not pixels, and that invariant is load-bearing
//! (see the header of src/entities/PlayerSprite.ts for the history: art on a
//! finer grid than the sim reads as a hi-res figure pasted onto a chunky world).
//! A frame is exactly `cellsH` rows of exactly `cellsW` characters, and the
//! heredoc in the source file is the silhouette the game draws.

use crate::schema::{Field, Schema};
use crate::value::Value;

/// How a sequence's frames are stepped. Names are the contract with the runtime's
/// frame picker, so this list is code as much as it is content.
///
/// ```text
///   hold     one frame, or a set the host selects from by other means
///   loop     free-running, `fps` frames per second, wraps
///   once     one-shot, holds the last frame — the shape every reactive pose wants
///   phase    indexed by an externally supplied 0..1 phase (the run cycle, driven
///            by real horizontal speed, so footfalls land where the feet are)
///   ambient  a slow cycle with an occasional interrupt on top: frames[0..n-2]
///            alternate at `fps`, and the LAST frame is shown for `blinkFor`
///            seconds every `blinkEvery` seconds. Idle breathing plus a blink.
/// ```
const MODES: &[&str] = &["hold", "loop", "once", "phase", "ambient"];

fn cells(v: &Value) -> Option<String> {
    let n = v.as_num().unwrap_or(0.0);
    if (1.0..=64.0).contains(&n) {
        None
    } else {
        Some("must be 1..64 cells".to_string())
    }
}

/// What a host chooses when it splices the art fields in.
pub struct SpriteArtOpts {
    /// "" for a top-level sprite, "art." to splice into a group.
    pub prefix: &'static str,
    /// The closed pose vocabulary for this host. Emitted as `ts_alias`.
    pub states: &'static [&'static str],
    /// What the pose union is CALLED, e.g. "AnimState" / "MobPose".
    ///
    /// Currently documentation-only, and that is a limitation of the emitter, not
    /// a design choice. `emit.ts` declares `pub type <ts_alias> = …` only for
    /// top-level and group fields; it never walks the sub-fields of a `record[]`
    /// element, so setting the alias on `seq.state` would emit `state: AnimState`
    /// into a module that declares no such type — a generated file that does not
    /// compile. Until that loop learns about element sub-fields, the union is
    /// emitted INLINE and the facade names it in one derived line:
    ///
    /// ```text
    ///     pub type AnimState = <the `state` type of SpriteSeq>;
    /// ```
    ///
    /// That is just as drift-proof — the alias is derived from the same generated
    /// union, so content and code still cannot disagree — and the day the emitter
    /// grows the four-line walk, the only change here is passing this through to
    /// the field descriptor.
    pub ts_alias: &'static str,
    /// Name of the emitted element struct, e.g. "SpriteSeq".
    pub ts_element: &'static str,
    /// Pose a sequence gets when it omits `state=`. Set it only where one unnamed
    /// sequence is the overwhelmingly common case — an item icon is one still
    /// frame, and making it write `state=idle` would be ceremony. Left unset,
    /// `state=` is REQUIRED, which is what a multi-pose host wants: an unlabelled
    /// sequence there is an authoring mistake, not a shorthand.
    pub default_state: Option<&'static str>,
}

/// The art fields, spliceable under any prefix.
///
/// Returned as a fresh vector each call — a schema's field list is mutated by
/// nobody today, but sharing one object between two schemas would make the next
/// person who adds a host-specific tweak break the other host silently.
pub fn sprite_art_fields(opts: SpriteArtOpts) -> Vec<(String, Field)> {
    let p = opts.prefix;

    // NOTE: the type alias is deliberately NOT set on `state`, and the omission
    // is not an oversight — see the note on `SpriteArtOpts::ts_alias`. The union
    // is emitted inline, which is exactly as drift-proof; only the NAME is
    // missing, and the facade supplies that in one derived line.
    let state = Field::new(&format!("enum({})", opts.states.join("|"))).doc(&format!(
        "Which pose this sequence supplies. Closed vocabulary — a typo is a \
         build error. The facade names this union `{}`.",
        opts.ts_alias
    ));
    let state = match opts.default_state {
        None => state.required(),
        Some(d) => state.default_str(d),
    };

    vec![
        (
            format!("{p}cellsW"),
            Field::new("int")
                .doc("Art grid width in world cells. One art pixel is one world cell.")
                .required()
                .check(cells),
        ),
        (
            format!("{p}cellsH"),
            Field::new("int")
                .doc(
                    "Art grid height in world cells. Every frame must have exactly this many rows.",
                )
                .required()
                .check(cells),
        ),
        (
            format!("{p}grain"),
            Field::new("int")
                .doc(
                    "Art pixels per world cell, per axis. 1 — the default, and the \
                     rule everywhere — is ONE ART PIXEL IS ONE WORLD CELL. 2 doubles \
                     the art resolution inside the SAME world rectangle: the grid is \
                     still `cellsW` x `cellsH` CELLS, the drawn rect and every \
                     collision quantity are untouched, but each frame row is \
                     `cellsW * grain` characters and there are `cellsH * grain` rows. \
                     A finer-grained sprite is drawn on a finer grid than the terrain \
                     it stands on, which is exactly the mismatch the invariant \
                     exists to prevent — so this is an EXPERIMENT'S knob, not a \
                     default to drift toward. It exists so one creature can be \
                     redrawn finer and judged against its neighbours in a picture \
                     before any fleet-wide decision.",
                )
                .default_int(1)
                .check(|v| {
                    let n = v.as_num().unwrap_or(0.0);
                    if (1.0..=4.0).contains(&n) {
                        None
                    } else {
                        Some("must be 1..4".to_string())
                    }
                }),
        ),
        (
            format!("{p}pal"),
            Field::new("list<string>")
                .doc(
                    "Palette, \"#rrggbb\" per index. Index 0 is the transparent slot and is \
                     written \".\" — it is never painted, so its value is a placeholder, not a \
                     colour. Frame characters are digits indexing this list.",
                )
                .required(),
        ),
        (
            format!("{p}fps"),
            Field::new("float")
                .doc(
                    "Sprite-wide playback rate. A sequence with its own non-zero `fps` \
                     overrides it, so this is the value for everything that has no reason to \
                     be special.",
                )
                .default_float(8.0),
        ),
        (
            format!("{p}variants"),
            Field::new("int")
                .doc(
                    "How many tinted copies of the whole sprite to bake. 1 = just the \
                     palette as authored. More than one lets a crowd of the same creature \
                     read as individuals without authoring a second palette, at the cost of \
                     one extra baked canvas set each.",
                )
                .default_int(1)
                .check(|v| {
                    let n = v.as_num().unwrap_or(0.0);
                    if (1.0..=16.0).contains(&n) {
                        None
                    } else {
                        Some("must be 1..16".to_string())
                    }
                }),
        ),
        (
            format!("{p}seq"),
            Field::new("record[]")
                .doc(
                    "One animation sequence per entry, in file order. Attributes as keys, \
                     frames in the `frames` body.",
                )
                .element(
                    opts.ts_element,
                    vec![
                        ("state".to_string(), state),
                        (
                            "mode".to_string(),
                            Field::new(&format!("enum({})", MODES.join("|")))
                                .doc(
                                    "How the frames are stepped. Names are the contract with the \
                                     runtime's picker.",
                                )
                                .default_str("hold"),
                        ),
                        (
                            "fps".to_string(),
                            Field::new("float")
                                .doc(
                                    "Playback rate for THIS sequence. 0 inherits the sprite's \
                                     `fps`, which is why the default is 0 and not 8 — 0 is \
                                     'unset', and an explicit 8 here would pin the sequence if \
                                     the sprite were ever retuned.",
                                )
                                .default_float(0.0),
                        ),
                        (
                            "blinkEvery".to_string(),
                            Field::new("float")
                                .doc(
                                    "`ambient` only: seconds between interrupts. 0 = never \
                                     interrupt. The interrupt frame is the sequence's LAST frame, \
                                     so the ambient cycle itself is frames[0..n-2].",
                                )
                                .default_float(0.0),
                        ),
                        (
                            "blinkFor".to_string(),
                            Field::new("float")
                                .doc("`ambient` only: seconds the interrupt frame is held.")
                                .default_float(0.1),
                        ),
                        (
                            "frames".to_string(),
                            Field::new("text")
                                .doc(
                                    "The frames, as a `'''` body: `cellsH` rows per frame, frames \
                                     separated by a blank line. Read it as a filmstrip.",
                                )
                                .required(),
                        ),
                    ],
                ),
        ),
    ]
}

/// The pose vocabulary of the standalone `sprite` kind — art with no creature
/// attached to it.
///
/// Two consumers today: the player, whose art moved out of PlayerSprite.ts so it
/// stops being source nobody can redraw without a rebuild; and item icons, which
/// reference a sprite id from `item.icon`. Both want exactly the same fields,
/// which is the argument for one kind rather than an `@icon` kind that would be
/// a `sprite` with a shorter enum.
///
/// The pose vocabulary is the PLAYER's twelve, and `default_state: "idle"` is
/// what lets a one-frame icon write a `seq` entry with only its `frames` and
/// mean it. An icon that
/// happens to animate can therefore use the whole player vocabulary; nothing
/// reads those poses on an icon, and forbidding them would need a second kind to
/// buy nothing.
const ANIM_STATES: &[&str] = &[
    "idle",
    "run",
    "skid",
    "jump",
    "doubleJump",
    "fall",
    "land",
    "dash",
    "wallSlide",
    "swim",
    "punch",
    "hurt",
];

pub fn schema() -> Schema {
    let mut fields: Vec<(String, Field)> = vec![(
        "name".into(),
        Field::new("string")
            .doc("Display name, for the art debug overlay. Defaults to the id.")
            .default_fn(|d| Some(Value::Str(d.str_of("id").to_string()))),
    )];
    fields.extend(sprite_art_fields(SpriteArtOpts {
        prefix: "",
        states: ANIM_STATES,
        ts_alias: "AnimState",
        ts_element: "SpriteSeq",
        default_state: Some("idle"),
    }));

    Schema {
        kind: "sprite".into(),
        prefix: "SPRITE".into(),
        iface: "SpriteDef".into(),
        iface_prefix: "Sprite".into(),
        fields,
        tables: vec![],
        matrices: vec![],
        constants: vec![],

        // A retired sprite keeps its code so an item that still names it resolves
        // to something rather than to nothing. One magenta cell: loud, and the
        // smallest legal sprite there is.
        tombstone: vec![
            ("name".into(), "\"(removed)\"".into()),
            ("cellsW".into(), "1".into()),
            ("cellsH".into(), "1".into()),
            ("pal".into(), "[\".\", \"#ff00ff\"]".into()),
        ],
    }
}
