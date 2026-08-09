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
/// noise sound sweeping 900 Hz down to 260 audibly falls. Without it `hz` and
/// `hzTo` are dead fields on every noise sound, which is most of them — and
/// `content/sounds/` authors sweeps on `dash`, `land` and `splash` that would
/// have done nothing at all.
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

/// Render the sound with this code, or `None` if there is no such sound.
///
/// The pitch sweeps linearly from `hz` to `hzTo` across the length. Linear and
/// not exponential: the sweeps here are short and shallow, the difference is
/// inaudible at these lengths, and a linear ramp is the one a person tuning
/// two numbers in a TOML file will predict correctly.
pub fn render(code: u16) -> Option<Pcm> {
    let i = code as usize;
    if i >= SOUND_COUNT {
        return None;
    }
    let wave = SoundWave::from_code(SND_WAVE[i])?;
    let seconds = SND_SECONDS[i];
    let total = (seconds * SAMPLE_RATE as f32) as usize;
    if total == 0 {
        return Some(Vec::new());
    }

    let (hz, hz_to) = (SND_HZ[i], SND_HZ_TO[i]);
    let (attack, release) = (SND_ATTACK[i], SND_RELEASE[i]);
    let (noise_mix, gain) = (SND_NOISE[i], SND_GAIN[i]);

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
    Some(out)
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

    #[test]
    fn every_authored_sound_renders_to_audible_samples() {
        // The bank as a whole, so a sound authored with a combination that
        // silently produces nothing is caught by existing rather than by
        // somebody noticing it never plays.
        for code in 0..SOUND_COUNT as u16 {
            let pcm = render(code).expect("a code in range renders");
            assert!(!pcm.is_empty(), "sound {code} rendered no samples");
            let peak = pcm.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            assert!(peak > 0.05, "sound {code} is inaudible: peak {peak}");
        }
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

    #[test]
    fn a_swept_sound_really_changes_pitch() {
        // `dash` sweeps 900 Hz down to 260. Counting zero crossings in the first
        // and last thirds is a blunt frequency estimate and exactly blunt enough
        // to catch the sweep being dropped, which is what a refactor would do to
        // it.
        let pcm = render(sound::DASH).expect("renders");
        let third = pcm.len() / 3;
        let crossings = |w: &[f32]| w.windows(2).filter(|p| p[0] * p[1] < 0.0).count();
        let early = crossings(&pcm[..third]);
        let late = crossings(&pcm[pcm.len() - third..]);
        assert!(
            early > late,
            "the pitch did not fall: {early} crossings early, {late} late"
        );
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
