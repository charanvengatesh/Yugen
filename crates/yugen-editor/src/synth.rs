//! Parameters to samples, so the editor can hear a sound it has not saved.
//!
//! # Why this is written twice
//!
//! The same argument as [`crate::raster`], and it lands harder here.
//! `yugen-render/src/sound/synth.rs` is the authority — it is what the game
//! actually plays — but it renders from `yugen_data::sounds`, which are `static`
//! arrays indexed by a compiled code. A sound that has not been through
//! `contentc` has no code, so there is *nothing to pass it*: the whole point of
//! a knob is hearing the value before it is committed, and the game's renderer
//! cannot take an uncommitted value at all.
//!
//! So the model is restated as a plain [`Params`] struct and the algorithm is
//! copied. Same honest caveat as the raster: nothing yet PROVES the two agree,
//! and the mechanism that would — a shared fixture both crates render and
//! compare — is not written. `yugen-render`'s copy is normative.
//!
//! The algorithm itself, and why each part of it is there, is documented at
//! length in that file. What follows is the same code with the same reasons, and
//! the reasons are not repeated in full.

/// Samples per second. 44100, matching the renderer — a different rate here
/// would make the preview a different sound from the one the game plays, which
/// is the one thing this module must not be.
pub const SAMPLE_RATE: u32 = 44_100;

/// Forced fade on both ends, whatever the envelope says.
///
/// `attack = 0` is authored on purpose — an impact wants an instant onset — but
/// "instant" at sample resolution is a DC step, and the pop is louder than the
/// sound carrying it. A millisecond turns the step into a ramp without any onset
/// in `content/sounds/` losing its character.
const DECLICK: usize = SAMPLE_RATE as usize / 1000;

/// The oscillators the synthesiser has. Closed — a name here is a branch of real
/// DSP, and content may not invent one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wave {
    Square,
    Saw,
    Sine,
    Triangle,
    Noise,
}

impl Wave {
    /// Every wave, in the schema's own order. What the editor's picker lists.
    pub const ALL: [Wave; 5] = [
        Wave::Square,
        Wave::Saw,
        Wave::Sine,
        Wave::Triangle,
        Wave::Noise,
    ];

    /// The name content writes. Round-trips with [`Wave::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Wave::Square => "square",
            Wave::Saw => "saw",
            Wave::Sine => "sine",
            Wave::Triangle => "triangle",
            Wave::Noise => "noise",
        }
    }

    pub fn parse(s: &str) -> Option<Wave> {
        Wave::ALL.into_iter().find(|w| w.name() == s)
    }
}

/// One sound, as the eight numbers that make it.
///
/// The defaults are the schema's, so a `Params` built with [`Default`] is what a
/// record writing nothing but `wave`, `hz` and `seconds` actually means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub wave: Wave,
    pub hz: f32,
    /// Pitch at the end. Equal to `hz` is no sweep.
    pub hz_to: f32,
    pub seconds: f32,
    /// Fraction of the length spent rising to full volume.
    pub attack: f32,
    /// Fraction of the length spent falling to silence.
    pub release: f32,
    /// White noise mixed over the oscillator, 0..1.
    pub noise: f32,
    pub gain: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            wave: Wave::Square,
            hz: 440.0,
            hz_to: 440.0,
            seconds: 0.1,
            attack: 0.0,
            release: 0.5,
            noise: 0.0,
            gain: 0.7,
        }
    }
}

/// One sound, rendered. Mono, `-1.0..=1.0`.
pub type Pcm = Vec<f32>;

/// Deterministic white noise for a sample index. xorshift over the index, not a
/// running stream, so a sample's value depends on nothing but its position.
#[inline]
fn noise_at(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9e37_79b9) | 1;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (f64::from(x) * 2.328_306_436_538_696_3e-10) as f32 * 2.0 - 1.0
}

/// One cycle of an oscillator at `phase`, `0.0..1.0`.
///
/// `hold` is the number of times the phase has wrapped, and it is what gives
/// noise a PITCH — a new random value per cycle rather than per sample. Without
/// it `hz` and `hzTo` are dead on every noise sound, which is most of them.
#[inline]
fn osc(wave: Wave, phase: f32, hold: u32) -> f32 {
    match wave {
        Wave::Square => {
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        Wave::Saw => phase * 2.0 - 1.0,
        Wave::Sine => (phase * core::f32::consts::TAU).sin(),
        Wave::Triangle => 1.0 - (phase - 0.5).abs() * 4.0,
        Wave::Noise => noise_at(hold),
    }
}

/// The volume envelope at `t`, a `0.0..=1.0` position through the sound.
///
/// An authored pair summing over 1 is scaled until the ramps meet, producing a
/// peak rather than a negative-width sustain — content can say `attack = 0.8,
/// release = 0.8` and that is a reasonable reading of it.
#[inline]
pub fn envelope(t: f32, attack: f32, release: f32) -> f32 {
    let a = attack.clamp(0.0, 1.0);
    let r = release.clamp(0.0, 1.0);
    let (a, r) = if a + r > 1.0 {
        let k = 1.0 / (a + r);
        (a * k, r * k)
    } else {
        (a, r)
    };
    if a > 0.0 && t < a {
        t / a
    } else if r > 0.0 && t > 1.0 - r {
        (1.0 - t) / r
    } else {
        1.0
    }
}

/// The de-click ramp at sample `n` of `total`. See [`DECLICK`].
#[inline]
fn declick(n: usize, total: usize) -> f32 {
    let d = DECLICK.min(total / 2).max(1);
    let rise = (n as f32 / d as f32).min(1.0);
    let fall = ((total - 1 - n) as f32 / d as f32).min(1.0);
    rise.min(fall)
}

/// Render these parameters to samples.
///
/// The pitch sweeps LINEARLY from `hz` to `hz_to`. Linear and not exponential
/// because the sweeps here are short and shallow, the difference is inaudible at
/// these lengths, and a linear ramp is the one a person tuning two numbers will
/// predict correctly.
pub fn render(p: &Params) -> Pcm {
    let total = (p.seconds * SAMPLE_RATE as f32) as usize;
    if total == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(total);
    // Phase is accumulated rather than computed from the sample index, because
    // the frequency moves: `sin(i * hz)` with a moving `hz` bends the waveform
    // behind the cursor instead of continuing it, which clicks at every step.
    let mut phase = 0.0f32;
    let mut hold = 0u32;
    for n in 0..total {
        let t = n as f32 / total as f32;
        let f = p.hz + (p.hz_to - p.hz) * t;
        phase += f / SAMPLE_RATE as f32;
        while phase >= 1.0 {
            phase -= 1.0;
            hold = hold.wrapping_add(1);
        }

        let tone = osc(p.wave, phase, hold);
        // The overlay noise is per-SAMPLE, unlike the noise oscillator: grit on
        // top of a tone, and grit that moved with the pitch would read as a
        // second voice.
        let s = tone * (1.0 - p.noise) + noise_at(n as u32) * p.noise;

        let ends = declick(n, total);
        out.push((s * envelope(t, p.attack, p.release) * ends * p.gain).clamp(-1.0, 1.0));
    }
    out
}

/// Peak absolute sample. What the editor shows as a level, and what a test uses
/// to say a sound is audible at all.
pub fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().fold(0.0f32, |a, s| a.max(s.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dig() -> Params {
        // `content/sounds/world.toml`'s `[dig]`, which is the shape most of the
        // bank has: short, noisy, swept down, quiet.
        Params {
            wave: Wave::Noise,
            hz: 300.0,
            hz_to: 200.0,
            seconds: 0.04,
            attack: 0.0,
            release: 0.85,
            noise: 0.0,
            gain: 0.2,
        }
    }

    #[test]
    fn a_sound_starts_and_ends_at_silence() {
        // Both ends, because a waveform cut off mid-cycle is a click, and a click
        // at the start of a footstep is the one artefact that survives being
        // quiet. This is what `DECLICK` buys, and `dig` authors `attack = 0`, so
        // without it this sound opens on a step.
        let pcm = render(&dig());
        assert!(!pcm.is_empty());
        assert!(pcm[0].abs() < 0.05, "opens on a click: {}", pcm[0]);
        assert!(
            pcm[pcm.len() - 1].abs() < 0.05,
            "ends on a click: {}",
            pcm[pcm.len() - 1]
        );
    }

    #[test]
    fn nothing_clips_even_with_noise_over_a_full_oscillator() {
        let p = Params {
            wave: Wave::Square,
            noise: 1.0,
            gain: 1.0,
            ..dig()
        };
        for s in render(&p) {
            assert!((-1.0..=1.0).contains(&s), "clipped at {s}");
        }
    }

    #[test]
    fn a_swept_sound_really_changes_pitch() {
        // Counting zero crossings in the first and last thirds is a blunt
        // frequency estimate, and exactly blunt enough to catch the sweep being
        // dropped — which is what an edit to the phase accumulator would do.
        let p = Params {
            wave: Wave::Sine,
            hz: 900.0,
            hz_to: 260.0,
            seconds: 0.3,
            ..dig()
        };
        let pcm = render(&p);
        let third = pcm.len() / 3;
        let crossings = |w: &[f32]| w.windows(2).filter(|q| q[0] * q[1] < 0.0).count();
        let (early, late) = (
            crossings(&pcm[..third]),
            crossings(&pcm[pcm.len() - third..]),
        );
        assert!(early > late, "pitch did not fall: {early} then {late}");
    }

    #[test]
    fn the_envelope_is_a_ramp_up_a_plateau_and_a_ramp_down() {
        assert_eq!(envelope(0.0, 0.25, 0.25), 0.0);
        assert_eq!(envelope(0.25, 0.25, 0.25), 1.0);
        assert_eq!(envelope(0.5, 0.25, 0.25), 1.0, "the plateau");
        assert_eq!(envelope(1.0, 0.25, 0.25), 0.0);
        // No envelope at all is a flat one, not a silent one.
        assert_eq!(envelope(0.0, 0.0, 0.0), 1.0);
        // And an overlong pair meets at a peak rather than going negative.
        assert!((envelope(0.5, 0.8, 0.8) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn the_render_is_deterministic() {
        // Determinism is a design choice — see the renderer's header. Two renders
        // of the same parameters must be identical, or the preview a designer
        // judges is not the sound they saved.
        assert_eq!(render(&dig()), render(&dig()));
        assert_ne!(noise_at(0), noise_at(1));
    }

    #[test]
    fn every_wave_is_audible_and_round_trips_through_its_name() {
        for w in Wave::ALL {
            let p = Params {
                wave: w,
                seconds: 0.1,
                gain: 0.7,
                ..dig()
            };
            assert!(peak(&render(&p)) > 0.05, "{} is inaudible", w.name());
            assert_eq!(Wave::parse(w.name()), Some(w));
        }
        assert_eq!(Wave::parse("supersaw"), None);
    }

    #[test]
    fn a_length_of_zero_renders_nothing_rather_than_panicking() {
        // The slider can reach it while being dragged, and `declick` divides by a
        // window derived from the total.
        let p = Params {
            seconds: 0.0,
            ..dig()
        };
        assert!(render(&p).is_empty());
    }
}
