//! Which frame, at what time: the animation model as arithmetic.
//!
//! [`super`]'s "The pure animation model" section, and pure is the operative
//! word — a frame index is a function of a [`SeqSpec`] and a [`SpriteClock`]
//! and nothing else. No texture, no atlas, no Bevy, no clock of its own.
//!
//! That is what lets every play mode be tested by handing it a time and reading
//! back a number, which is the only way the phase and ambient modes are
//! checkable at all: both are about what happens at a particular moment, and a
//! test that had to render a frame to find out would be measuring the renderer.

use super::vocab::{Frame, PlayMode, SeqSpec, SpriteClock, SpriteError};

// ---------------------------------------------------------------------------
// The pure animation model
// ---------------------------------------------------------------------------

/// Split one pose body into frames on blank lines, and validate the row count.
///
/// Frames are separated by a blank line so the source reads as a filmstrip. The
/// row count of every frame must match `cells_h` or the art and the collision box
/// disagree — the one error in a sprite table that renders as a plausible-looking
/// creature instead of a crash, which is why it fails here at load rather than
/// being noticed six months later. Row WIDTH is re-checked when [`BakedSprite`]
/// rasterises, because only the baker knows the palette.
pub fn split_frames(
    lines: &'static [&'static str],
    cells_h: u32,
    who: &str,
) -> Result<Vec<Frame>, SpriteError> {
    let mut out: Vec<Frame> = Vec::new();
    let mut cur: Frame = Vec::new();
    for line in lines {
        if line.is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        cur.push(line);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    for (i, frame) in out.iter().enumerate() {
        if frame.len() != cells_h as usize {
            return Err(SpriteError::FrameRows {
                who: who.to_string(),
                frame: i,
                rows: frame.len(),
                expected: cells_h,
            });
        }
    }
    Ok(out)
}

/// Resolve a sequence to a frame INDEX. Pure, integer-returning, no Bevy.
///
/// This is the entire animation model, and it is deliberately a free function:
/// pure, integer-returning, and therefore assertable without a texture, an app or
/// a GPU. [`BakedSprite::tile`] calls it and indexes an already-built array.
///
/// Every mode clamps its input to `>= 0` and wraps negatives back into range
/// rather than trusting the caller's clock, because a single frame of negative dt
/// (a window regaining focus, a debug scrub) would otherwise index out of the
/// array and blank the sprite for one frame — a bug that looks like a flicker and
/// reads like a GPU problem.
pub fn pick_frame_index(spec: &SeqSpec, clock: &SpriteClock) -> usize {
    let n = spec.frames.len();
    if n <= 1 {
        return 0;
    }

    match spec.mode {
        PlayMode::Hold => 0,

        // Cadence comes from the world (distance travelled), not from time, so a
        // half-speed run plays its contacts at half rate with no extra state.
        PlayMode::Phase => wrap_index(clock.phase * n as f32, n),

        // Holds the last frame: a one-shot that wrapped would replay its own
        // impact forever, which is precisely wrong for land/punch/double-jump.
        PlayMode::Once => {
            let raw = floor_to_i64(clock.state_t.max(0.0) * spec.fps);
            (n as i64 - 1).min(raw.max(0)) as usize
        }

        PlayMode::Ambient => ambient_index(spec, clock.clock_t),

        PlayMode::Loop => wrap_index(clock.state_t.max(0.0) * spec.fps, n),
    }
}

/// Idle: a slow breath with an occasional blink punched over the top.
///
/// The blink is the LAST frame by convention, so the breath frames are `0..n-2`
/// and adding a blink to an existing loop is an append rather than an index
/// rewrite. Both cycles are read off the SAME wrapped `t`, not off raw `clock_t`
/// — the breath therefore restarts in step with every blink instead of drifting
/// against it, which is what makes the whole idle read as one repeating gesture
/// rather than two unrelated ones beating against each other.
///
/// # Why this divides where every other mode multiplies
///
/// [`PlayMode::Loop`] and [`PlayMode::Once`] compute `floor(t * fps)`; this
/// computes `floor(t / beat)` with `beat = 1 / fps`. That is NOT an oversight.
/// The animation this replaces divided by a PERIOD (`IDLE_BREATH = 0.62`), so
/// dividing here is what makes the two exactly equivalent rather than 99.99%
/// equivalent: `t * (1/0.62)` and `t / 0.62` disagree at ties, and over an hour of
/// 60fps frame times the multiply form picks a different breath frame on eight of
/// them.
///
/// The TypeScript could close that gap completely, because in f64
/// `1/(1/0.62) === 0.62` exactly and its facade authored `fps: 1 / 0.62`. Here
/// the content compiler emits an `f32` (`1.6129032`) and the round trip is exact
/// only to about one part in `1e7`. The division form is still the right one and
/// is kept; what the port loses is the last bit of that equivalence, which is
/// invisible and is written down here rather than quietly dropped. See
/// `the_ambient_beat_divides_where_every_other_mode_multiplies`.
pub(super) fn ambient_index(spec: &SeqSpec, clock_t: f32) -> usize {
    let n = spec.frames.len();
    let raw = clock_t.max(0.0);
    let period = spec.blink_every;
    let beat = 1.0 / spec.fps;

    // No blink period means no reserved frame: loop the whole strip. The NaN test
    // is not redundant — the TypeScript wrote this `!(period > 0)` precisely so
    // that a NaN period fell in HERE rather than dividing by it below, and
    // `period <= 0.0` alone would silently drop that half of the guard.
    if period.is_nan() || period <= 0.0 {
        return wrap_index(raw / beat, n);
    }

    let t = raw % period;
    if t < spec.blink_for {
        return n - 1;
    }

    let breaths = n - 1;
    if breaths <= 1 {
        return 0;
    }
    wrap_index(t / beat, breaths)
}

/// `floor(v)` wrapped into `0..n`, the way every mode above needs it.
///
/// The TypeScript spelled this `let i = Math.floor(v) % n; if (i < 0) i += n;`
/// three times over; `rem_euclid` is the same arithmetic with the sign correction
/// built in.
///
/// A non-finite `v` returns 0. In the TypeScript a `NaN` clock produced a `NaN`
/// index, `baked[…][NaN]` was `undefined`, and `draw` bailed — one blank frame.
/// Frame 0 is the authored neutral pose of every sequence in the game, so falling
/// back to it is strictly the better of the two, and it is the outcome the
/// clamping above was reaching for anyway.
pub(super) fn wrap_index(v: f32, n: usize) -> usize {
    let i = v.floor();
    if !i.is_finite() {
        return 0;
    }
    (i as i64).rem_euclid(n as i64) as usize
}

/// `Math.floor` into an integer, saturating rather than wrapping on a huge input
/// and landing on 0 for a `NaN` one.
pub(super) fn floor_to_i64(v: f32) -> i64 {
    let f = v.floor();
    if f.is_nan() { 0 } else { f as i64 }
}
