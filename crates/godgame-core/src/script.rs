//! A line-per-verb scripting language for driving the game without a human.
//!
//! # Why this exists
//!
//! The binary has exactly one way to play itself: `--drive SECS`, which holds
//! right and raises a jump edge every `SECS`. That is the entire vocabulary. It
//! cannot dig, place, craft, wait, or aim, so every scenario more specific than
//! "run right" has to be hand-built as a bespoke Rust test — which means the
//! scenario lives in a test binary, cannot be run against the real renderer, and
//! cannot end in a screenshot without someone writing more Rust.
//!
//! This module is the parser behind `--script FILE`. A scenario becomes a text
//! file that can be committed, diffed and re-run: the same file produces the
//! same [`Intent`] on the same frame every time, so it can be pointed at a
//! screenshot, a state dump, or a regression gate and be trusted to mean the
//! same thing next month.
//!
//! # The language
//!
//! One verb per line. `#` starts a comment that runs to end of line, blank lines
//! are ignored, verbs are lowercase and case-sensitive.
//!
//! ```text
//! # walk to the ledge and dig down into it
//! right 2s
//! aim 120 -64
//! hold punch 1.5s
//! wait 30f
//! jump
//! ```
//!
//! | line | meaning |
//! |---|---|
//! | `wait 30f` / `wait 1.5s` | no input at all for that long |
//! | `left 2s` / `right 2s` | hold that direction |
//! | `up 1s` / `down 1s` | hold up / down: climbing, dropping through platforms |
//! | `jump` | one rising edge, one frame |
//! | `dash` | one rising edge, one frame |
//! | `punch` | one rising edge, one frame (dig / swing / shoot) |
//! | `hold punch 1s` | the punch key held down, for auto-repeat |
//! | `dig 1s` | left mouse held: remove cells under the aim point |
//! | `place 1s` | right mouse held: place the held item under the aim point |
//! | `aim <dx> <dy>` | where to point, in px FROM THE BODY, for every later frame |
//!
//! A duration is REQUIRED where the table shows one and FORBIDDEN where it does
//! not. `jump 2s` is an error rather than a two-second jump, because jump is an
//! edge and there is no such thing as a long one; writing it means the author
//! believed something false about the language and should be told so.
//!
//! # The limitation worth stating up front
//!
//! A script is a SEQUENCE. Each line runs to completion before the next one
//! begins, and there is no way to express "hold right while punching" — that
//! would need concurrency, a second axis in the file format, and a rule for what
//! happens when two lines disagree. None of that is here. What can be expressed
//! is any interleaving at frame granularity, which covers the scenarios this was
//! built for; anything genuinely simultaneous still needs Rust.
//!
//! [`aim`](Script::parse) is the one exception, and it is an exception because it
//! is not an input event at all — it is where the cursor is pointing, which has
//! to persist across the lines that act on it.
//!
//! # `aim` is relative to the body, and that is not a convenience
//!
//! [`Intent::aim_x`] is a WORLD point, and an early draft of this language took
//! one. It was unusable, for a reason that only shows up when a script is run:
//! the author does not know where the body will be. A script's whole job is to
//! move it, so the position at the moment `aim` takes effect is the output of
//! everything above that line — gravity, terrain, a jump that clipped a ledge.
//! `aim 0 40` is a sentence anybody can write and means "forty pixels below my
//! feet"; the world point that turns into is not knowable when the file is
//! written.
//!
//! So `aim` is an OFFSET from the body's centre, resolved by the driver on the
//! frame it is used. [`ScriptFrame::intent`]'s `aim_x`/`aim_y` carry the offset,
//! not the point, and a driver that forgot to resolve them would aim at the
//! world origin — which `Intent::aim_x` documents as the "no aim point"
//! sentinel, so the failure is a shot along the facing rather than a wild one.

use crate::input::Intent;
use std::fmt;

/// Frames per second the script's durations are measured in.
///
/// This is a constant of the LANGUAGE, not a reading of the engine. Physics runs
/// at [`STEP_DT`](crate::config::physics::STEP_DT) and the host presents at
/// whatever rate the display gives it; if a script's `1s` were derived from
/// either, retuning the simulation or moving to a faster monitor would silently
/// change what every committed scenario file does, and the artefacts they are
/// checked against would rot without a single line of the script changing. 60 is
/// the frame cadence input is sampled at, and pinning it here means `2s` means
/// 120 frames in this file forever.
pub const SCRIPT_HZ: f32 = 60.0;

/// Ceiling on a single duration, in frames: one hour.
///
/// Frames are expanded eagerly, so a fat-fingered `wait 10000s` is not a slow
/// script, it is several gigabytes of `Intent`. Refusing it at parse time turns
/// a typo into an error message instead of an out-of-memory kill.
const MAX_DURATION_FRAMES: u32 = 60 * 60 * 60;

/// One frame of a script: what the keyboard says, and what the pointer is doing.
///
/// Two halves rather than one, because the game reads them through two different
/// paths and neither can express the other. Movement, jumping, dashing and
/// swinging arrive as an [`Intent`] sampled from the keyboard. DIGGING AND
/// PLACING DO NOT — they go through the pointer and the mouse buttons, and
/// `Intent` has no field for either. A script that only produced `Intent`s could
/// walk a body around a world it was unable to touch, which was the first thing
/// this module could not do and the reason this struct exists.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScriptFrame {
    /// The keyboard half.
    ///
    /// `aim_x` / `aim_y` are an OFFSET FROM THE BODY, not a world point — see
    /// [`Script::parse`]. The driver resolves them; a parser has no idea where
    /// the body is and could not resolve them if it tried.
    pub intent: Intent,
    /// The left mouse button: remove cells under the pointer.
    pub dig: bool,
    /// The right mouse button: place the held item under the pointer.
    pub place: bool,
}

/// A parsed script: a flat list of frames.
///
/// Frames are stored EXPANDED rather than as steps plus a cursor. A scenario is
/// at most a few thousand frames — under a hundred kilobytes of `Intent` — so
/// the compressed representation would buy nothing but a seek routine that can
/// be wrong. Expanded, [`Script::intent`] is an index, which is impossible to
/// get subtly wrong and trivially replayable from any frame.
#[derive(Clone, Debug, Default)]
pub struct Script {
    frames: Vec<ScriptFrame>,
}

/// One problem with one line of a script, with the line number a text editor
/// shows: 1-based, counting blank and comment lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptError {
    /// 1-based line number in the source text.
    pub line: usize,
    /// What is wrong, in the author's terms rather than the parser's.
    pub message: String,
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// What one non-blank line compiles to.
enum Line {
    /// `count` frames of held state. The first frame carries `first`, which is
    /// where any rising edge lives; every later frame carries `rest`. For a
    /// direction hold the two are identical, and for an edge verb `count` is 1
    /// and `rest` is never reached.
    Frames {
        first: Intent,
        rest: Intent,
        count: u32,
        /// Held for every frame of this line, not just the first. Digging is a
        /// button you hold down, and the brush's own cadence decides how often
        /// it bites — see `BuildTool`.
        dig: bool,
        /// As `dig`, for the right button.
        place: bool,
    },
    /// Move the aim point. Produces no frames of its own.
    SetAim(f32, f32),
}

impl Script {
    /// Parse a script, reporting EVERY problem found rather than only the first.
    ///
    /// A scenario file is written in one go and then run, so a parser that stops
    /// at the first bad line turns a five-typo file into five edit-run cycles.
    /// Collecting them costs a `Vec` and one pass, and the pass has to happen
    /// anyway. Lines that failed to compile contribute no frames; a script with
    /// any error is not returned at all, because a partially-applied scenario is
    /// worse than no scenario.
    ///
    /// `aim` is the only verb that outlives its own line: it sets the aim offset
    /// for every frame emitted after it, until another `aim` replaces it. The
    /// offset is measured from the body's centre and resolved by the driver —
    /// see the module header for why it cannot be a world point. `aim 0 0` aims
    /// at the body itself, which is also the state a script starts in.
    pub fn parse(text: &str) -> Result<Script, Vec<ScriptError>> {
        let mut frames: Vec<ScriptFrame> = Vec::new();
        let mut errors: Vec<ScriptError> = Vec::new();
        let mut aim = (0.0f32, 0.0f32);

        for (index, raw) in text.lines().enumerate() {
            let line = index + 1;
            let body = raw.split_once('#').map_or(raw, |(before, _)| before).trim();
            if body.is_empty() {
                continue;
            }

            let mut words = body.split_whitespace();
            let Some(verb) = words.next() else { continue };
            let args: Vec<&str> = words.collect();

            match compile(verb, &args) {
                Ok(Line::SetAim(x, y)) => aim = (x, y),
                Ok(Line::Frames {
                    first,
                    rest,
                    count,
                    dig,
                    place,
                }) => {
                    for n in 0..count {
                        let mut intent = if n == 0 { first } else { rest };
                        intent.aim_x = aim.0;
                        intent.aim_y = aim.1;
                        frames.push(ScriptFrame { intent, dig, place });
                    }
                }
                Err(message) => errors.push(ScriptError { line, message }),
            }
        }

        if errors.is_empty() {
            Ok(Script { frames })
        } else {
            Err(errors)
        }
    }

    /// Total frames the script runs for.
    pub fn frames(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Frame `n`, or `None` once the script has run out.
    ///
    /// Running out is reported rather than saturated at the last frame. Holding
    /// the final intent forever would mean a script ending in `right 2s` walks
    /// off into the world indefinitely, and the caller could never tell the
    /// difference between "still driving" and "finished". `None` is the signal
    /// to take the screenshot, dump the state, or hand control back.
    pub fn frame(&self, n: u32) -> Option<ScriptFrame> {
        self.frames.get(n as usize).copied()
    }

    /// Frame `n`'s keyboard half.
    ///
    /// Shorthand for `frame(n).map(|f| f.intent)`. Most questions about a script
    /// are about the keys, and a caller that only drives movement should not
    /// have to know the pointer half exists.
    pub fn intent(&self, n: u32) -> Option<Intent> {
        self.frame(n).map(|f| f.intent)
    }
}

/// Compile one line's worth of words, or say what is wrong with it.
fn compile(verb: &str, args: &[&str]) -> Result<Line, String> {
    match verb {
        "wait" => held(args, verb, Intent::default()),
        "left" => held(
            args,
            verb,
            Intent {
                dir_x: -1.0,
                ..Intent::default()
            },
        ),
        "right" => held(
            args,
            verb,
            Intent {
                dir_x: 1.0,
                ..Intent::default()
            },
        ),
        "up" => held(
            args,
            verb,
            Intent {
                up: true,
                ..Intent::default()
            },
        ),
        "down" => held(
            args,
            verb,
            Intent {
                down: true,
                ..Intent::default()
            },
        ),

        // The edge verbs. A real key going down is queued AND held on that
        // frame, so the held flag is set too; it is gone the next frame, which
        // for jump means a scripted jump is the cut-short minimum-height one.
        // There is no verb for a long jump, and inventing `hold jump` would be
        // guessing at a mechanic rather than describing one.
        "jump" => edge(
            args,
            verb,
            Intent {
                jump_queued: true,
                jump_held: true,
                up: true,
                ..Intent::default()
            },
        ),
        "dash" => edge(
            args,
            verb,
            Intent {
                dash_queued: true,
                ..Intent::default()
            },
        ),
        "punch" => edge(
            args,
            verb,
            Intent {
                punch_queued: true,
                punch_held: true,
                ..Intent::default()
            },
        ),

        // `hold punch 1s` is the key going down and STAYING down: the rising
        // edge on the first frame starts the swing, the held flag on the rest
        // auto-repeats it. Dropping the edge would hold a key that never
        // triggered anything, which is not what holding a key does.
        "hold" => {
            let Some(&what) = args.first() else {
                return Err("`hold` needs something to hold: `hold punch 1s`".to_string());
            };
            if what != "punch" {
                return Err(format!(
                    "`hold` only takes `punch`; `{what}` is either its own verb or not a verb"
                ));
            }
            let count = duration(&args[1..], "hold punch")?;
            Ok(Line::Frames {
                first: Intent {
                    punch_queued: true,
                    punch_held: true,
                    ..Intent::default()
                },
                rest: Intent {
                    punch_held: true,
                    ..Intent::default()
                },
                count,
                dig: false,
                place: false,
            })
        }

        "dig" => pointer(args, verb, true, false),
        "place" => pointer(args, verb, false, true),

        "aim" => {
            if args.len() != 2 {
                return Err(format!(
                    "`aim` takes an x and y offset from the body: `aim 0 40`, not {} argument(s)",
                    args.len()
                ));
            }
            let x = coordinate(args[0])?;
            let y = coordinate(args[1])?;
            Ok(Line::SetAim(x, y))
        }

        _ => Err(format!("unknown verb `{verb}`")),
    }
}

/// A verb that takes a duration and holds `state` for it.
fn held(args: &[&str], verb: &str, state: Intent) -> Result<Line, String> {
    let count = duration(args, verb)?;
    Ok(Line::Frames {
        first: state,
        rest: state,
        count,
        dig: false,
        place: false,
    })
}

/// `dig 1s` / `place 1s`: a mouse button held down over the aim point.
///
/// Held rather than an edge, unlike [`edge`]'s verbs, because that is what the
/// buttons are: `swing_brush` reads `buttons.pressed(..)` every frame and the
/// tool's own cadence decides how often the brush actually bites. A one-frame
/// dig would usually remove nothing at all.
fn pointer(args: &[&str], verb: &str, dig: bool, place: bool) -> Result<Line, String> {
    let count = duration(args, verb)?;
    Ok(Line::Frames {
        first: Intent::default(),
        rest: Intent::default(),
        count,
        dig,
        place,
    })
}

/// A verb that is a rising edge, and therefore refuses a duration.
fn edge(args: &[&str], verb: &str, state: Intent) -> Result<Line, String> {
    if !args.is_empty() {
        return Err(format!(
            "`{verb}` is a rising edge and takes no duration; it occupies exactly one frame"
        ));
    }
    Ok(Line::Frames {
        first: state,
        rest: state,
        count: 1,
        dig: false,
        place: false,
    })
}

/// The single duration argument a held verb requires.
fn duration(args: &[&str], verb: &str) -> Result<u32, String> {
    match args {
        [] => Err(format!(
            "`{verb}` needs a duration: `{verb} 30f` or `{verb} 0.5s`"
        )),
        [one] => frames_of(one),
        _ => Err(format!(
            "`{verb}` takes one duration, got {} arguments",
            args.len()
        )),
    }
}

/// `30f` or `1.5s`, in frames.
///
/// The unit is mandatory. A bare `30` is ambiguous between half a second and
/// half a minute, and the two are far enough apart that guessing is worse than
/// asking.
fn frames_of(token: &str) -> Result<u32, String> {
    let count = if let Some(digits) = token.strip_suffix('f') {
        digits
            .parse::<u32>()
            .map_err(|_| format!("`{token}` is not a whole number of frames"))?
    } else if let Some(number) = token.strip_suffix('s') {
        let seconds = number
            .parse::<f32>()
            .map_err(|_| format!("`{token}` is not a number of seconds"))?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(format!("`{token}` is not a duration"));
        }
        // Round rather than truncate: at 60 Hz a written `0.25s` is 15 frames
        // exactly, but a value that lands on 119.999 frames should be the 120
        // the author meant, not 119.
        (seconds * SCRIPT_HZ).round() as u32
    } else {
        return Err(format!(
            "`{token}` needs a unit: `f` for frames or `s` for seconds"
        ));
    };

    if count > MAX_DURATION_FRAMES {
        return Err(format!(
            "`{token}` is longer than the one-hour ceiling of {MAX_DURATION_FRAMES} frames"
        ));
    }
    Ok(count)
}

/// One component of an `aim` offset, in px from the body's centre.
fn coordinate(token: &str) -> Result<f32, String> {
    match token.parse::<f32>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(format!("`{token}` is not an aim offset in px")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Script {
        Script::parse(text).expect("script should parse")
    }

    fn errors(text: &str) -> Vec<ScriptError> {
        Script::parse(text).expect_err("script should not parse")
    }

    #[test]
    fn every_verb_produces_the_intent_it_claims_to() {
        let s = parsed("wait 1f");
        assert_eq!(s.intent(0), Some(Intent::default()));

        let s = parsed("left 1f");
        assert_eq!(s.intent(0).unwrap().dir_x, -1.0);

        let s = parsed("right 1f");
        assert_eq!(s.intent(0).unwrap().dir_x, 1.0);

        let s = parsed("up 1f");
        let i = s.intent(0).unwrap();
        assert!(i.up && !i.down);

        let s = parsed("down 1f");
        let i = s.intent(0).unwrap();
        assert!(i.down && !i.up);

        let s = parsed("dash");
        assert!(s.intent(0).unwrap().dash_queued);

        let s = parsed("punch");
        let i = s.intent(0).unwrap();
        assert!(i.punch_queued && i.punch_held);
    }

    #[test]
    fn a_jump_is_one_frame_and_not_a_held_state() {
        let s = parsed("jump\nwait 1f");
        assert_eq!(s.frames(), 2);

        let edge = s.intent(0).unwrap();
        assert!(edge.jump_queued, "the jump frame carries the edge");
        assert!(edge.jump_held, "a key going down is also down");

        let after = s.intent(1).unwrap();
        assert!(
            !after.jump_queued,
            "a queued flag that survives a frame pogos"
        );
        assert!(!after.jump_held);
    }

    #[test]
    fn holding_punch_raises_the_edge_once_and_then_only_holds() {
        let s = parsed("hold punch 3f");
        assert_eq!(s.frames(), 3);

        let first = s.intent(0).unwrap();
        assert!(first.punch_queued, "holding a key starts with pressing it");
        assert!(first.punch_held);

        for n in 1..3 {
            let later = s.intent(n).unwrap();
            assert!(!later.punch_queued, "frame {n} would start a second swing");
            assert!(later.punch_held, "frame {n} is still holding the key");
        }
    }

    #[test]
    fn two_seconds_of_right_is_a_hundred_and_twenty_frames_of_full_deflection() {
        let s = parsed("right 2s");
        assert_eq!(s.frames(), 120, "2s at {SCRIPT_HZ} Hz");
        for n in 0..s.frames() {
            assert_eq!(s.intent(n).unwrap().dir_x, 1.0, "frame {n}");
        }
    }

    #[test]
    fn frames_and_seconds_are_the_same_clock() {
        assert_eq!(parsed("wait 90f").frames(), parsed("wait 1.5s").frames());
        assert_eq!(parsed("wait 0f").frames(), 0, "a zero duration is nothing");
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let s = parsed(
            "# walk right\n\
             \n\
             right 1f   # then stop\n\
             \n\
             # trailing comment\n",
        );
        assert_eq!(s.frames(), 1);
        assert_eq!(s.intent(0).unwrap().dir_x, 1.0);
    }

    #[test]
    fn aim_persists_into_later_lines_and_is_overwritten_by_a_second_aim() {
        let s = parsed(
            "wait 1f\n\
             aim 120 -64\n\
             punch\n\
             wait 1f\n\
             aim -8 16\n\
             wait 1f\n",
        );
        assert_eq!(s.frames(), 4, "an aim line emits no frames of its own");

        let before = s.intent(0).unwrap();
        assert!(!before.has_aim(), "a script starts with no cursor");

        for n in 1..3 {
            let i = s.intent(n).unwrap();
            assert_eq!((i.aim_x, i.aim_y), (120.0, -64.0), "frame {n}");
        }

        let last = s.intent(3).unwrap();
        assert_eq!((last.aim_x, last.aim_y), (-8.0, 16.0));
    }

    #[test]
    fn a_duration_on_an_edge_verb_is_an_error() {
        let e = errors("jump 2s");
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].line, 1);
        assert!(e[0].message.contains("rising edge"), "{}", e[0].message);
    }

    #[test]
    fn a_held_verb_without_a_duration_is_an_error() {
        for text in ["wait", "left", "right", "up", "down", "hold punch"] {
            let e = errors(text);
            assert_eq!(e.len(), 1, "`{text}`");
            assert!(e[0].message.contains("duration"), "{}", e[0].message);
        }
    }

    #[test]
    fn a_duration_without_a_unit_is_an_error() {
        let e = errors("wait 30");
        assert!(e[0].message.contains("unit"), "{}", e[0].message);
    }

    #[test]
    fn every_error_in_a_file_is_reported_with_its_own_line_number() {
        let e = errors(
            "right 1s\n\
             jump 2s\n\
             # a comment, which still counts as a line\n\
             sprint 1s\n\
             wait\n\
             aim 4\n",
        );
        let lines: Vec<usize> = e.iter().map(|x| x.line).collect();
        assert_eq!(
            lines,
            vec![2, 4, 5, 6],
            "one report per bad line, 1-based, comments counted"
        );
        assert!(e[1].message.contains("unknown verb"), "{}", e[1].message);
    }

    #[test]
    fn an_error_prints_the_line_it_came_from() {
        let e = errors("\n\ndash 1s");
        assert_eq!(e[0].to_string(), format!("line 3: {}", e[0].message));
    }

    #[test]
    fn a_duration_past_the_ceiling_is_refused_rather_than_allocated() {
        let e = errors("wait 100000s");
        assert!(e[0].message.contains("ceiling"), "{}", e[0].message);
    }

    #[test]
    fn digging_and_placing_are_held_buttons_rather_than_edges() {
        // `swing_brush` reads `buttons.pressed(..)` every frame and the tool's
        // own cadence decides when the brush bites, so a one-frame dig would
        // usually remove nothing at all.
        let s = parsed("dig 3f\nplace 2f");
        for n in 0..3 {
            let f = s.frame(n).expect("a dig frame");
            assert!(f.dig && !f.place, "frame {n}: {f:?}");
        }
        for n in 3..5 {
            let f = s.frame(n).expect("a place frame");
            assert!(f.place && !f.dig, "frame {n}: {f:?}");
        }
        assert_eq!(s.frames(), 5);
    }

    #[test]
    fn the_pointer_half_is_clear_on_every_frame_that_did_not_ask_for_it() {
        // The half nobody set must be OFF, not left over from a line above.
        // A script that dug for a second and then walked away still holding the
        // button would carve a trench nobody wrote.
        let s = parsed("dig 2f\nright 2f\njump");
        assert!(s.frame(1).unwrap().dig);
        for n in 2..5 {
            let f = s.frame(n).expect("a frame");
            assert!(!f.dig && !f.place, "frame {n} still holds a button: {f:?}");
        }
    }

    #[test]
    fn an_aim_offset_rides_on_every_later_frame_including_a_dig() {
        // The offset is resolved against the body by the driver; the parser's
        // job is only to make sure it is still attached when the driver sees it.
        let s = parsed("aim 0 40\ndig 2f");
        for n in 0..2 {
            let f = s.frame(n).expect("a dig frame");
            assert!(f.dig);
            assert_eq!((f.intent.aim_x, f.intent.aim_y), (0.0, 40.0));
        }
    }

    #[test]
    fn past_the_end_the_script_reports_nothing_rather_than_repeating() {
        let s = parsed("right 3f");
        assert_eq!(s.frames(), 3);
        assert!(s.intent(2).is_some());
        assert_eq!(s.intent(3), None);
        assert_eq!(s.intent(9_999), None);

        let empty = parsed("# nothing at all\n");
        assert_eq!(empty.frames(), 0);
        assert_eq!(empty.intent(0), None);
    }
}
