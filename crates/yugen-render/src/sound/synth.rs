//! Parameters to samples. No Bevy, no device, no clock.
//!
//! [`render`] is a pure function of a sound's code and a sample rate, which is
//! the whole design: every property worth asserting about a sound — that it
//! starts at silence, that it ends at silence, that a sweep actually sweeps,
//! that nothing clips — is a property of a `Vec<f32>` and can be checked by a
//! test that never opens an audio device. The tests below do exactly that, and
//! they are the reason the synthesiser is a separate file from the plugin that
//! plays it.
//!
//! # The model
//!
//! One oscillator, one envelope, one optional noise mix. That is deliberately
//! less than sfxr offers — no vibrato, no repeat, no phaser, no filters — and
//! the missing pieces are missing because nothing in `content/sounds/` has
//! wanted one yet. Each would arrive as a schema field and an arm here, and the
//! rule that keeps this honest is the one the block schema uses: a knob nothing
//! authored has ever set is a knob whose default is untested.
//!
//! # Why the noise is deterministic
//!
//! A hash of the sample index, not a random stream. Two plays of `dig` are
//! byte-identical, which sounds slightly worse than a fresh noise seed per play
//! and buys something worth more: a golden test can assert on the samples at
//! all, and a sound that is wrong is wrong the same way every time somebody
//! goes looking for it.

use yugen_data::sounds::{
    SND_ATTACK, SND_GAIN, SND_HZ, SND_HZ_TO, SND_NOISE, SND_RELEASE, SND_SECONDS, SND_WAVE,
    SOUND_COUNT, SoundWave,
};

/// Samples per second everything here is rendered at.
///
/// 44100 because it is what every backend accepts without resampling, and
/// because these sounds are short enough that the memory a higher rate would
/// cost buys nothing: the whole bank is well under a megabyte.
pub const SAMPLE_RATE: u32 = 44_100;

/// One sound, rendered. Mono, `-1.0..=1.0`.
pub type Pcm = Vec<f32>;

/// Deterministic white noise for a sample index.
///
/// xorshift over the index rather than a running stream, so a sample's value
/// depends on nothing but its position — see the header for why that matters
/// more than the variety a live stream would give.
#[inline]
fn noise_at(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9e37_79b9) | 1;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    // `u32` to `-1..1`. The `f64` divide is `JuiceRng::rand`'s, for its reason:
    // 24 bits of `f32` mantissa cannot represent the whole range, so doing this
    // at `f32` would quantise the noise itself.
    (f64::from(x) * 2.328_306_436_538_696_3e-10) as f32 * 2.0 - 1.0
}

/// Samples of fade forced onto both ends, whatever the envelope says.
///
/// An `attack` of 0 is authored on purpose — an impact wants an instant onset,
/// and ramping it up over any audible time makes it read as a swell instead of a
/// hit. But "instant" at sample resolution is a step from silence to
/// full amplitude, which is a DC discontinuity: it pops, and the pop is louder
/// and nastier than the sound carrying it.
///
/// A millisecond is short enough that no onset in `content/sounds/` loses its
/// character and long enough that the step becomes a ramp. It is applied on top
/// of the envelope rather than folded into it so that `attack = 0` keeps
/// meaning what it says.
const DECLICK: usize = SAMPLE_RATE as usize / 1000;

/// One cycle of an oscillator at `phase`, which is `0.0..1.0`.
///
/// `hold` is the number of times the phase has wrapped, and it is what gives
/// noise a PITCH: a new random value per cycle rather than per sample, so a
/// noise sound that sweeps audibly moves. Without it `hz` and `hzTo` are dead
/// fields on every noise sound, which is most of them, and the sweeps authored
/// across `content/sounds/` would do nothing at all.
///
/// No particular record is named here on purpose. This comment used to cite
/// `dash` falling from 900 Hz to 260; `dash` has since been retuned to rise from
/// 55, and a doc comment that quotes content is a doc comment that goes quietly
/// wrong the first time somebody turns a knob.
#[inline]
fn osc(wave: SoundWave, phase: f32, hold: u32) -> f32 {
    match wave {
        SoundWave::Square => {
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        SoundWave::Saw => phase * 2.0 - 1.0,
        SoundWave::Sine => (phase * core::f32::consts::TAU).sin(),
        // Up for the first half, down for the second.
        SoundWave::Triangle => 1.0 - (phase - 0.5).abs() * 4.0,
        SoundWave::Noise => noise_at(hold),
    }
}

/// The volume envelope at `t`, a `0.0..=1.0` position through the sound.
///
/// Attack and release are fractions of the length rather than seconds, so
/// retuning a sound's duration does not silently turn its envelope into a click
/// or into a sound that never reaches full volume.
///
/// They are clamped so that an authored pair summing over 1 still produces a
/// peak instead of a negative-width sustain — content can say `attack = 0.8,
/// release = 0.8` and get a triangle, which is a reasonable reading of it and a
/// better answer than a compile error about a combination no schema field can
/// see on its own.
#[inline]
fn envelope(t: f32, attack: f32, release: f32) -> f32 {
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
    // A sound shorter than two ramps gets a triangle rather than a negative
    // window. Nothing authored is that short, and the alternative is arithmetic
    // that silently goes negative if something ever is.
    let d = DECLICK.min(total / 2).max(1);
    let rise = (n as f32 / d as f32).min(1.0);
    let fall = ((total - 1 - n) as f32 / d as f32).min(1.0);
    rise.min(fall)
}

/// One sound's numbers, lifted out of the compiled tables.
///
/// This exists so the synthesiser can be driven by something other than a code.
/// [`render`] looks a code up in `yugen-data` and calls [`render_params`], and
/// the split matters for two callers that are not the game:
///
/// - the pin test at the bottom of this file, which has to render a FIXED case
///   that no content record supplies, so that `yugen-editor`'s second copy of
///   this algorithm can be held to the same samples;
/// - `crates/yugen-editor`, which tunes sounds that have never been compiled and
///   therefore have no code to look up at all.
///
/// The editor cannot link this crate — it would be linking Bevy to turn eight
/// floats into a `Vec<f32>` — so it carries its own copy of the struct and the
/// loop. What it must not carry is its own ANSWER, which is what the pin is for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub wave: SoundWave,
    pub hz: f32,
    pub hz_to: f32,
    pub seconds: f32,
    pub attack: f32,
    pub release: f32,
    pub noise: f32,
    pub gain: f32,
}

/// Render the sound with this code, or `None` if there is no such sound.
pub fn render(code: u16) -> Option<Pcm> {
    let i = code as usize;
    if i >= SOUND_COUNT {
        return None;
    }
    Some(render_params(&Params {
        wave: SoundWave::from_code(SND_WAVE[i])?,
        hz: SND_HZ[i],
        hz_to: SND_HZ_TO[i],
        seconds: SND_SECONDS[i],
        attack: SND_ATTACK[i],
        release: SND_RELEASE[i],
        noise: SND_NOISE[i],
        gain: SND_GAIN[i],
    }))
}

/// Render these parameters to samples. The synthesiser itself.
///
/// The pitch sweeps linearly from `hz` to `hz_to` across the length. Linear and
/// not exponential: the sweeps here are short and shallow, the difference is
/// inaudible at these lengths, and a linear ramp is the one a person tuning
/// two numbers in a TOML file will predict correctly.
pub fn render_params(p: &Params) -> Pcm {
    let Params {
        wave,
        hz,
        hz_to,
        seconds,
        attack,
        release,
        noise: noise_mix,
        gain,
    } = *p;
    let total = (seconds * SAMPLE_RATE as f32) as usize;
    if total == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(total);
    // Phase is accumulated rather than computed from the sample index, because
    // the frequency changes as we go: `sin(i * hz)` with a moving `hz` bends the
    // whole waveform behind the cursor instead of continuing it, which is
    // audible as a click at every step of the sweep.
    let mut phase = 0.0f32;
    let mut hold = 0u32;
    for n in 0..total {
        let t = n as f32 / total as f32;
        let f = hz + (hz_to - hz) * t;
        phase += f / SAMPLE_RATE as f32;
        // Count the wraps rather than flooring in place, so `hold` is the cycle
        // index the noise oscillator samples on. A sweep changes how often this
        // advances, which is exactly what makes pitched noise pitched.
        while phase >= 1.0 {
            phase -= 1.0;
            hold = hold.wrapping_add(1);
        }

        let tone = osc(wave, phase, hold);
        // The overlay noise is per-SAMPLE, unlike the noise oscillator. It is
        // grit on top of a tone rather than the tone itself, and grit that
        // moved with the pitch would read as a second voice.
        let s = tone * (1.0 - noise_mix) + noise_at(n as u32) * noise_mix;

        let ends = declick(n, total);
        out.push((s * envelope(t, attack, release) * ends * gain).clamp(-1.0, 1.0));
    }
    out
}

/// Every sound in the bank, indexed by code.
///
/// Rendered once. The whole bank is a few hundred kilobytes of `f32`, which is
/// less than one of the sprite atlases, and rendering on demand would put a
/// synthesiser on the path of a footstep.
pub fn render_all() -> Vec<Pcm> {
    (0..SOUND_COUNT as u16)
        .map(|c| render(c).unwrap_or_default())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_data::sounds::sound;

    /// Loudest sample, ignoring sign. What "audible" is measured with.
    fn peak_of(pcm: &[f32]) -> f32 {
        pcm.iter().fold(0.0f32, |a, s| a.max(s.abs()))
    }

    #[test]
    fn every_authored_sound_renders_to_audible_samples() {
        // The bank as a whole, so a sound authored with a combination that
        // silently produces nothing is caught by existing rather than by
        // somebody noticing it never plays.
        //
        // `gain = 0` is exempt, and the exemption is a principle rather than a
        // name: it is the ONE way to author silence that is unambiguous. Every
        // other route to an inaudible sound — a length rounding to no samples, an
        // envelope that never opens, a noise mix cancelling a tone — is a
        // combination nobody meant, and those are exactly what this catches. A
        // record with a gain and no sound is a bug; a record with no gain is a
        // decision. `content/sounds/player.toml`'s `land` is the decision.
        for code in 0..SOUND_COUNT as u16 {
            let pcm = render(code).expect("a code in range renders");
            assert!(!pcm.is_empty(), "sound {code} rendered no samples");
            let peak = pcm.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            if SND_GAIN[code as usize] == 0.0 {
                assert_eq!(peak, 0.0, "sound {code} has no gain but is not silent");
                continue;
            }
            assert!(peak > 0.05, "sound {code} is inaudible: peak {peak}");
        }
    }

    #[test]
    fn the_silence_exemption_only_covers_a_gain_of_zero() {
        // The exemption above is the only hole in the audibility guard, so this
        // states its edges. A sound that is quiet is still a sound; only an
        // explicit zero opts out, and it opts out by being ACTUALLY silent
        // rather than by being skipped.
        let quiet = Params {
            gain: 0.01,
            ..pin_case()
        };
        assert!(peak_of(&render_params(&quiet)) > 0.0, "0.01 still sounds");

        let silent = Params {
            gain: 0.0,
            ..pin_case()
        };
        let pcm = render_params(&silent);
        assert!(
            !pcm.is_empty(),
            "silence is still the right NUMBER of samples"
        );
        assert_eq!(peak_of(&pcm), 0.0);
    }

    #[test]
    fn nothing_clips() {
        // Noise mixed over a full-amplitude oscillator is the case that would,
        // and the clamp is what stops it. A sample outside -1..1 is not merely
        // loud: most backends wrap or hard-fault on it.
        for code in 0..SOUND_COUNT as u16 {
            for s in render(code).expect("renders") {
                assert!((-1.0..=1.0).contains(&s), "sound {code} clipped at {s}");
            }
        }
    }

    #[test]
    fn a_sound_starts_and_ends_at_silence() {
        // Both ends, because a waveform cut off mid-cycle is a click, and a
        // click at the start of a footstep is the one artefact that survives
        // being quiet.
        for code in 0..SOUND_COUNT as u16 {
            let pcm = render(code).expect("renders");
            let (first, last) = (pcm[0], pcm[pcm.len() - 1]);
            assert!(first.abs() < 0.05, "sound {code} opens on a click: {first}");
            assert!(last.abs() < 0.05, "sound {code} ends on a click: {last}");
        }
    }

    #[test]
    fn the_envelope_is_a_ramp_up_a_plateau_and_a_ramp_down() {
        assert_eq!(envelope(0.0, 0.25, 0.25), 0.0);
        assert_eq!(envelope(0.125, 0.25, 0.25), 0.5);
        assert_eq!(envelope(0.25, 0.25, 0.25), 1.0);
        assert_eq!(envelope(0.5, 0.25, 0.25), 1.0, "the plateau");
        assert_eq!(envelope(0.875, 0.25, 0.25), 0.5);
        assert_eq!(envelope(1.0, 0.25, 0.25), 0.0);

        // No envelope at all is a flat one, not a silent one.
        assert_eq!(envelope(0.0, 0.0, 0.0), 1.0);
        assert_eq!(envelope(1.0, 0.0, 0.0), 1.0);
    }

    #[test]
    fn an_overlong_envelope_becomes_a_peak_rather_than_going_negative() {
        // `attack + release > 1` has no sustain to share, and the reading that
        // keeps it a sound is "scale both until they meet". Content can author
        // it and the schema cannot see the combination, so the synthesiser has
        // to have an answer.
        let peak = envelope(0.5, 0.8, 0.8);
        assert!(
            (peak - 1.0).abs() < 1.0e-5,
            "the two ramps should meet at 1"
        );
        assert_eq!(envelope(0.0, 0.8, 0.8), 0.0);
        assert_eq!(envelope(1.0, 0.8, 0.8), 0.0);
    }

    /// The case `yugen-editor` is pinned to. See `tests/pin/README.md`.
    ///
    /// Deliberately not any record in `content/sounds/`: a pin tied to authored
    /// content would break every time somebody retuned a footstep, which trains
    /// people to bless it without reading, and a golden nobody reads is not a
    /// pin. Fixed numbers instead, chosen to touch every branch — a real
    /// oscillator rather than pure noise, a sweep that rises, both ends of the
    /// envelope, and a noise mix so BOTH noise paths run.
    fn pin_case() -> Params {
        Params {
            wave: SoundWave::Triangle,
            hz: 300.0,
            hz_to: 900.0,
            seconds: 0.02,
            attack: 0.3,
            release: 0.4,
            noise: 0.25,
            gain: 0.8,
        }
    }

    #[test]
    fn the_synth_matches_the_pin_that_yugen_editor_is_held_to() {
        let pcm = render_params(&pin_case());
        assert_eq!(pcm.len(), 882, "0.02 s at 44100");
        crate::pin::check("sound_synth.hex", &crate::pin::samples_to_bytes(&pcm));
    }

    #[test]
    fn the_pin_case_covers_what_the_synth_decides() {
        // A golden nobody reads can quietly stop covering what it was written
        // for, so the properties the case was chosen for are asserted here.
        let pcm = render_params(&pin_case());
        assert!(pcm[0].abs() < 0.05, "the de-click ramp opens it");
        assert!(pcm[pcm.len() - 1].abs() < 0.05, "and closes it");
        assert!(
            pcm.iter().fold(0.0f32, |a, s| a.max(s.abs())) > 0.3,
            "the plateau is audible"
        );
        assert!(
            pcm.iter().all(|s| (-1.0..=1.0).contains(s)),
            "nothing clips"
        );

        // The sweep RISES here, unlike every sound in the bank, so a synth that
        // silently ran the ramp backwards would still pass the `dash` test.
        //
        // Counted on a noise-free copy of the same case, because zero crossings
        // cannot see the sweep through the mix: at 25% noise over 0.02 s the
        // grit contributes far more sign changes than six-to-eighteen cycles of
        // triangle do, and the count measures the noise instead. The mix stays
        // in the PIN itself — it is exactly the branch coverage the golden wants
        // — and only this one property is read off the quieter twin.
        let clean = render_params(&Params {
            noise: 0.0,
            ..pin_case()
        });
        let third = clean.len() / 3;
        let crossings = |w: &[f32]| w.windows(2).filter(|p| p[0] * p[1] < 0.0).count();
        let (early, late) = (
            crossings(&clean[..third]),
            crossings(&clean[clean.len() - third..]),
        );
        assert!(late > early, "the pitch did not rise: {early} then {late}");
    }

    #[test]
    fn rendering_a_code_and_rendering_its_parameters_are_the_same_thing() {
        // `render` is now a table lookup in front of `render_params`, and the
        // split is only safe while the lookup passes every field through.
        let i = sound::DIG as usize;
        let direct = render_params(&Params {
            wave: SoundWave::from_code(SND_WAVE[i]).expect("a real wave"),
            hz: SND_HZ[i],
            hz_to: SND_HZ_TO[i],
            seconds: SND_SECONDS[i],
            attack: SND_ATTACK[i],
            release: SND_RELEASE[i],
            noise: SND_NOISE[i],
            gain: SND_GAIN[i],
        });
        assert_eq!(render(sound::DIG), Some(direct));
    }

    #[test]
    fn a_swept_sound_really_changes_pitch() {
        // Counting zero crossings in the first and last thirds is a blunt
        // frequency estimate, and exactly blunt enough to catch the sweep being
        // dropped — which is what a refactor of the phase accumulator would do.
        //
        // The DIRECTION is read from the record rather than written down here.
        // This test used to say "`dash` sweeps 900 Hz down to 260" and assert a
        // fall, which made it a test of that record's tuning as much as of the
        // synthesiser: the day `dash` was retuned to rise, it failed while
        // nothing was wrong with the code under test. Content is allowed to
        // change its mind about a sweep; the synthesiser is not allowed to
        // ignore one.
        //
        // The widest sweep in the bank is used because the estimate is coarse,
        // and a two-hertz sweep would not move the count.
        let (code, hz, hz_to) = (0..SOUND_COUNT)
            .map(|i| (i as u16, SND_HZ[i], SND_HZ_TO[i]))
            .max_by(|a, b| (a.1 - a.2).abs().total_cmp(&(b.1 - b.2).abs()))
            .expect("the bank is not empty");
        assert!(
            (hz - hz_to).abs() > 50.0,
            "no sound in the bank sweeps far enough to measure: {hz} -> {hz_to}"
        );

        let pcm = render(code).expect("renders");
        let third = pcm.len() / 3;
        let crossings = |w: &[f32]| w.windows(2).filter(|p| p[0] * p[1] < 0.0).count();
        let early = crossings(&pcm[..third]);
        let late = crossings(&pcm[pcm.len() - third..]);
        if hz_to > hz {
            assert!(
                late > early,
                "sound {code} should rise: {early} then {late}"
            );
        } else {
            assert!(
                early > late,
                "sound {code} should fall: {early} then {late}"
            );
        }
    }

    #[test]
    fn the_noise_is_the_same_every_time() {
        // Determinism is a design choice, not an accident — see the header. Two
        // renders of the same code must be byte-identical, or no test above can
        // assert on samples at all.
        assert_eq!(render(sound::DIG), render(sound::DIG));
        assert_ne!(
            noise_at(0),
            noise_at(1),
            "a hash that returns the same value for every index is not noise"
        );
    }

    #[test]
    fn a_code_the_bank_does_not_have_renders_nothing() {
        assert!(render(SOUND_COUNT as u16).is_none());
        assert!(render(u16::MAX).is_none());
        assert_eq!(render_all().len(), SOUND_COUNT);
    }
}
