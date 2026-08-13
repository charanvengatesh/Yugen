//! Rolling a sound instead of typing eight floats.
//!
//! # Categories are ranges, not presets
//!
//! A preset is one sound; press it twice and get the same thing, so it answers
//! "what does a pickup sound like" once and is then furniture. A [`Category`] is
//! a **region of the parameter space** — which waveforms are plausible, which
//! way the pitch sweeps, how long it lasts — so pressing it twice gives two
//! different pickups that are both recognisably pickups. That is the difference
//! between a starting point and a decoration.
//!
//! # Everything lands inside the schema
//!
//! Every draw is clamped to [`crate::sound::HZ_RANGE`] and
//! [`crate::sound::SECONDS_RANGE`], which are the schema's own bounds restated.
//! So nothing this produces can fail the next `contentc` run — a generator that
//! could author an out-of-range value would be a generator whose output has to
//! be checked before it can be saved, and the save is a mouse click.
//!
//! # Rolls are reproducible
//!
//! Same seed, same sound, for the same reason a sprite roll is: the seed is
//! small enough to write down, so "I liked the third one" survives the fourth.

use crate::rng::Rng;
use crate::sound::{HZ_RANGE, LOWPASS_RANGE, REPEAT_RANGE, SECONDS_RANGE, VIBRATO_HZ_RANGE};
use crate::synth::{Params, Wave};

/// The shapes a game sound comes in.
///
/// Not an exhaustive taxonomy — it is the set this game actually has, read off
/// `content/sounds/`: things picked up, things landed on, things hurt, things
/// broken. `Laser` and `Powerup` are the two that are not in the tree yet and
/// are here because they are the two most people reach for next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Pickup,
    Jump,
    Hurt,
    Blip,
    Explosion,
    Laser,
    Powerup,
}

impl Category {
    pub const ALL: [Category; 7] = [
        Category::Pickup,
        Category::Jump,
        Category::Hurt,
        Category::Blip,
        Category::Explosion,
        Category::Laser,
        Category::Powerup,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::Pickup => "pickup",
            Category::Jump => "jump",
            Category::Hurt => "hurt",
            Category::Blip => "blip",
            Category::Explosion => "explosion",
            Category::Laser => "laser",
            Category::Powerup => "powerup",
        }
    }

    /// The waveforms this category is drawn from.
    ///
    /// A closed list per category rather than a free choice: a noise "blip" is
    /// not a blip and a sine "explosion" is not an explosion, and letting the
    /// roll find that out means most rolls are wasted.
    fn waves(self) -> &'static [Wave] {
        match self {
            Category::Pickup => &[Wave::Sine, Wave::Triangle, Wave::Square],
            Category::Jump => &[Wave::Square, Wave::Saw, Wave::Triangle],
            Category::Hurt => &[Wave::Saw, Wave::Square, Wave::Noise],
            Category::Blip => &[Wave::Square, Wave::Triangle],
            Category::Explosion => &[Wave::Noise],
            Category::Laser => &[Wave::Saw, Wave::Square],
            Category::Powerup => &[Wave::Square, Wave::Triangle, Wave::Sine],
        }
    }
}

/// One category's region of the parameter space.
///
/// `hz` is the starting pitch; `sweep` multiplies it to get the end, so the
/// relationship is an interval rather than a second absolute pitch — a sound
/// that rises by a fifth should still rise by a fifth when the roll starts it an
/// octave lower.
struct Region {
    hz: (f32, f32),
    sweep: (f32, f32),
    seconds: (f32, f32),
    attack: (f32, f32),
    release: (f32, f32),
    noise: (f32, f32),
    gain: (f32, f32),
    /// Vibrato depth and rate, drawn together or not at all — half a vibrato is
    /// no vibrato, so a category either wobbles or it does not.
    vibrato: Option<((f32, f32), (f32, f32))>,
    repeat: Option<(f32, f32)>,
    lowpass: Option<(f32, f32)>,
}

fn region(cat: Category) -> Region {
    match cat {
        // Rising, brief, clean. The one sound in a game that is a small reward.
        Category::Pickup => Region {
            hz: (420.0, 900.0),
            sweep: (1.3, 2.0),
            seconds: (0.05, 0.12),
            attack: (0.0, 0.08),
            release: (0.4, 0.7),
            noise: (0.0, 0.05),
            gain: (0.25, 0.4),
            vibrato: None,
            repeat: None,
            lowpass: None,
        },
        // Rising and shorter, with body. A jump is a launch, not a chime.
        Category::Jump => Region {
            hz: (180.0, 380.0),
            sweep: (1.5, 2.6),
            seconds: (0.08, 0.16),
            attack: (0.0, 0.04),
            release: (0.3, 0.6),
            noise: (0.0, 0.12),
            gain: (0.3, 0.45),
            vibrato: None,
            repeat: None,
            lowpass: None,
        },
        // Falling, harsh, with grit. Damage reads as something breaking.
        Category::Hurt => Region {
            hz: (200.0, 460.0),
            sweep: (0.35, 0.65),
            seconds: (0.12, 0.28),
            attack: (0.0, 0.03),
            release: (0.45, 0.8),
            noise: (0.15, 0.45),
            gain: (0.3, 0.5),
            vibrato: None,
            repeat: None,
            lowpass: None,
        },
        // Flat, tiny, cheap to repeat. A UI tick, not an event.
        Category::Blip => Region {
            hz: (600.0, 1400.0),
            sweep: (0.95, 1.05),
            seconds: (0.02, 0.05),
            attack: (0.0, 0.05),
            release: (0.3, 0.6),
            noise: (0.0, 0.06),
            gain: (0.18, 0.3),
            vibrato: None,
            repeat: None,
            lowpass: None,
        },
        // Noise, falling, long release. All body and no pitch.
        Category::Explosion => Region {
            hz: (90.0, 260.0),
            sweep: (0.25, 0.5),
            seconds: (0.25, 0.6),
            attack: (0.0, 0.02),
            release: (0.6, 0.95),
            noise: (0.5, 1.0),
            gain: (0.35, 0.55),
            vibrato: None,
            repeat: None,
            // Noise with the top taken off is a thud; noise without is a hiss.
            lowpass: Some((600.0, 2600.0)),
        },
        // Fast, steep, downward. The classic descending zap.
        Category::Laser => Region {
            hz: (700.0, 1600.0),
            sweep: (0.25, 0.45),
            seconds: (0.06, 0.14),
            attack: (0.0, 0.02),
            release: (0.35, 0.65),
            noise: (0.0, 0.1),
            gain: (0.25, 0.4),
            vibrato: None,
            // A zap that restarts a few times inside its own length is the
            // difference between one shot and a weapon.
            repeat: Some((14.0, 34.0)),
            lowpass: None,
        },
        // Long, rising, clean — the only one here that is allowed to take its
        // time, because it is the only one that is a reward rather than feedback.
        Category::Powerup => Region {
            hz: (260.0, 520.0),
            sweep: (1.8, 3.2),
            seconds: (0.25, 0.55),
            attack: (0.05, 0.2),
            release: (0.4, 0.7),
            noise: (0.0, 0.06),
            gain: (0.3, 0.45),
            // A powerup is the only sound here long enough for a wobble to
            // complete a cycle, which is exactly what makes it read as magical
            // rather than as a mistake.
            vibrato: Some(((0.05, 0.18), (6.0, 14.0))),
            repeat: None,
            lowpass: None,
        },
    }
}

fn clamp_hz(v: f32) -> f32 {
    v.clamp(HZ_RANGE.0, HZ_RANGE.1)
}

/// A sound of category `cat`, at `seed`.
pub fn roll(cat: Category, seed: u32) -> Params {
    let mut rng = Rng::seeded(seed);
    let r = region(cat);
    let hz = clamp_hz(rng.range(r.hz.0, r.hz.1));
    let sweep = rng.range(r.sweep.0, r.sweep.1);
    Params {
        wave: *rng.pick(cat.waves()),
        hz,
        hz_to: clamp_hz(hz * sweep),
        seconds: rng
            .range(r.seconds.0, r.seconds.1)
            .clamp(SECONDS_RANGE.0, SECONDS_RANGE.1),
        attack: rng.range(r.attack.0, r.attack.1).clamp(0.0, 1.0),
        release: rng.range(r.release.0, r.release.1).clamp(0.0, 1.0),
        noise: rng.range(r.noise.0, r.noise.1).clamp(0.0, 1.0),
        gain: rng.range(r.gain.0, r.gain.1).clamp(0.0, 1.0),
        // Drawn only where the category asks for them, so a pickup is never
        // quietly given a wobble it did not want. Off is the default and off is
        // what most categories keep.
        vibrato: r
            .vibrato
            .map_or(0.0, |(d, _)| rng.range(d.0, d.1).clamp(0.0, 1.0)),
        vibrato_hz: r.vibrato.map_or(0.0, |(_, h)| {
            rng.range(h.0, h.1)
                .clamp(VIBRATO_HZ_RANGE.0, VIBRATO_HZ_RANGE.1)
        }),
        repeat_hz: r.repeat.map_or(0.0, |x| {
            rng.range(x.0, x.1).clamp(REPEAT_RANGE.0, REPEAT_RANGE.1)
        }),
        lowpass: r.lowpass.map_or(0.0, |x| {
            rng.range(x.0, x.1).clamp(LOWPASS_RANGE.0, LOWPASS_RANGE.1)
        }),
    }
}

/// Anything, anywhere in the schema's ranges. The "surprise me" button.
///
/// Pitch is drawn logarithmically for the reason the panel's sliders are
/// logarithmic: 20 Hz to 8 kHz is nearly nine octaves, and a uniform draw spends
/// seven eighths of its rolls above the octave anybody is actually tuning.
pub fn randomize(seed: u32) -> Params {
    let mut rng = Rng::seeded(seed);
    let log = |rng: &mut Rng, lo: f32, hi: f32| -> f32 {
        (lo.ln() + rng.unit() * (hi.ln() - lo.ln())).exp()
    };
    let hz = clamp_hz(log(&mut rng, HZ_RANGE.0, HZ_RANGE.1));
    Params {
        wave: *rng.pick(&Wave::ALL),
        hz,
        hz_to: clamp_hz(log(&mut rng, HZ_RANGE.0, HZ_RANGE.1)),
        seconds: log(&mut rng, SECONDS_RANGE.0, SECONDS_RANGE.1)
            .clamp(SECONDS_RANGE.0, SECONDS_RANGE.1),
        attack: rng.unit(),
        release: rng.unit(),
        noise: rng.unit(),
        gain: rng.range(0.15, 0.6),
        // Each shaping field is switched on only a fraction of the time. All
        // four on at once is a novelty rather than a sound, and "surprise me"
        // should still mostly produce something usable.
        vibrato: if rng.chance(0.3) {
            rng.range(0.05, 0.4)
        } else {
            0.0
        },
        vibrato_hz: if rng.chance(0.5) {
            rng.range(VIBRATO_HZ_RANGE.0, VIBRATO_HZ_RANGE.1)
        } else {
            0.0
        },
        repeat_hz: if rng.chance(0.2) {
            rng.range(REPEAT_RANGE.0, REPEAT_RANGE.1)
        } else {
            0.0
        },
        lowpass: if rng.chance(0.3) {
            rng.range(LOWPASS_RANGE.0, LOWPASS_RANGE.1)
        } else {
            0.0
        },
    }
}

/// `p`, jittered by `amount` of each field's own range.
///
/// The useful amounts are small. This is the move that turns a sound which is
/// nearly right into one that is right, and it is why rolling is worth doing at
/// all: a roll finds the neighbourhood and a mutation finds the house.
///
/// The waveform changes only rarely and only at large amounts — it is the one
/// parameter that is categorical rather than continuous, so nudging it is not a
/// nudge, it is a different sound.
pub fn mutate(p: &Params, amount: f32, seed: u32) -> Params {
    let a = amount.clamp(0.0, 1.0);
    if a == 0.0 {
        return *p;
    }
    let mut rng = Rng::seeded(seed);
    // Symmetric about the current value, so a mutation is as likely to undo the
    // last one as to compound it.
    let mut jitter = |v: f32, lo: f32, hi: f32| -> f32 {
        let span = (hi - lo) * a;
        (v + rng.range(-span, span)).clamp(lo, hi)
    };
    Params {
        hz: jitter(p.hz, HZ_RANGE.0, HZ_RANGE.1),
        hz_to: jitter(p.hz_to, HZ_RANGE.0, HZ_RANGE.1),
        seconds: jitter(p.seconds, SECONDS_RANGE.0, SECONDS_RANGE.1),
        attack: jitter(p.attack, 0.0, 1.0),
        release: jitter(p.release, 0.0, 1.0),
        noise: jitter(p.noise, 0.0, 1.0),
        gain: jitter(p.gain, 0.0, 1.0),
        // A field that is OFF stays off. Jittering zero would switch on a
        // wobble or a filter nobody asked for, and "mutate" means "near this
        // sound", not "near this sound plus a feature".
        vibrato: if p.vibrato > 0.0 {
            jitter(p.vibrato, 0.0, 1.0)
        } else {
            0.0
        },
        vibrato_hz: if p.vibrato_hz > 0.0 {
            jitter(p.vibrato_hz, VIBRATO_HZ_RANGE.0, VIBRATO_HZ_RANGE.1)
        } else {
            0.0
        },
        repeat_hz: if p.repeat_hz > 0.0 {
            jitter(p.repeat_hz, REPEAT_RANGE.0, REPEAT_RANGE.1)
        } else {
            0.0
        },
        lowpass: if p.lowpass > 0.0 {
            jitter(p.lowpass, LOWPASS_RANGE.0, LOWPASS_RANGE.1)
        } else {
            0.0
        },
        // Drawn last so the numbers above do not depend on whether it changed.
        wave: if rng.chance(a * 0.25) {
            *rng.pick(&Wave::ALL)
        } else {
            p.wave
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    fn inside_schema(p: &Params, what: &str) {
        assert!(
            (HZ_RANGE.0..=HZ_RANGE.1).contains(&p.hz),
            "{what}: hz {} is outside the schema",
            p.hz
        );
        assert!(
            (HZ_RANGE.0..=HZ_RANGE.1).contains(&p.hz_to),
            "{what}: hzTo {} is outside the schema",
            p.hz_to
        );
        assert!(
            (SECONDS_RANGE.0..=SECONDS_RANGE.1).contains(&p.seconds),
            "{what}: seconds {} is outside the schema",
            p.seconds
        );
        for (k, v) in [
            ("attack", p.attack),
            ("release", p.release),
            ("noise", p.noise),
            ("gain", p.gain),
        ] {
            assert!((0.0..=1.0).contains(&v), "{what}: {k} {v} is outside 0..1");
        }
    }

    #[test]
    fn a_seed_is_the_whole_input() {
        for cat in Category::ALL {
            for seed in [0u32, 1, 77, u32::MAX] {
                assert_eq!(roll(cat, seed), roll(cat, seed));
            }
        }
        assert_eq!(randomize(9), randomize(9));
    }

    #[test]
    fn nothing_any_generator_produces_can_fail_the_next_contentc_run() {
        // The property that lets a roll go straight to a save. Four thousand
        // draws across every category, plus the unconstrained randomiser.
        for cat in Category::ALL {
            for seed in 0..512u32 {
                inside_schema(&roll(cat, seed), cat.name());
            }
        }
        for seed in 0..2048u32 {
            inside_schema(&randomize(seed), "randomize");
        }
    }

    #[test]
    fn mutating_stays_inside_the_schema_however_hard_it_is_pushed() {
        // The case that would escape: a value already at the top of its range,
        // jittered upward at full strength.
        let hot = Params {
            hz: HZ_RANGE.1,
            hz_to: HZ_RANGE.1,
            seconds: SECONDS_RANGE.1,
            attack: 1.0,
            release: 1.0,
            noise: 1.0,
            gain: 1.0,
            wave: Wave::Saw,
            ..Params::default()
        };
        for seed in 0..512u32 {
            inside_schema(&mutate(&hot, 1.0, seed), "mutate from the top");
            inside_schema(
                &mutate(&Params::default(), 1.0, seed),
                "mutate from default",
            );
        }
    }

    #[test]
    fn mutating_by_nothing_is_exactly_the_identity() {
        // Not approximately: the panel's slider starts at zero and a "mutation"
        // that moved the numbers at 0% would mark the record dirty for nothing.
        let p = roll(Category::Pickup, 3);
        for seed in 0..64u32 {
            assert_eq!(mutate(&p, 0.0, seed), p);
        }
    }

    #[test]
    fn a_small_mutation_stays_in_the_neighbourhood() {
        // The whole point of mutate over reroll. At 5% the pitch must not move
        // by an octave, or it is not a nudge.
        let p = roll(Category::Pickup, 11);
        for seed in 0..256u32 {
            let m = mutate(&p, 0.05, seed);
            let ratio = m.hz / p.hz;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "5% moved the pitch by {ratio:.2}x"
            );
        }
    }

    #[test]
    fn the_waveform_survives_a_small_mutation() {
        // Categorical, not continuous: nudging the wave is not a nudge, it is a
        // different sound. At 5% it should almost never change.
        let p = roll(Category::Jump, 5);
        let changed = (0..512u32)
            .filter(|&s| mutate(&p, 0.05, s).wave != p.wave)
            .count();
        assert!(
            changed < 40,
            "{changed} of 512 small mutations changed wave"
        );
    }

    #[test]
    fn every_category_is_recognisably_itself() {
        // The claim that makes categories worth having. Averaged over 256 rolls,
        // each category's defining property has to hold — otherwise the button
        // is a randomiser with a label on it.
        let mean = |cat: Category, f: fn(&Params) -> f32| -> f32 {
            (0..256u32).map(|s| f(&roll(cat, s))).sum::<f32>() / 256.0
        };
        // Rising things rise and falling things fall.
        for up in [Category::Pickup, Category::Jump, Category::Powerup] {
            assert!(
                mean(up, |p| p.hz_to / p.hz) > 1.2,
                "{} does not rise",
                up.name()
            );
        }
        for down in [Category::Hurt, Category::Explosion, Category::Laser] {
            assert!(
                mean(down, |p| p.hz_to / p.hz) < 0.8,
                "{} does not fall",
                down.name()
            );
        }
        // A blip is flat, and it is the only one that is.
        let flat = mean(Category::Blip, |p| p.hz_to / p.hz);
        assert!((0.9..=1.1).contains(&flat), "a blip swept by {flat:.2}x");
        // An explosion is noise; a pickup is not.
        assert!(mean(Category::Explosion, |p| p.noise) > 0.6);
        assert!(mean(Category::Pickup, |p| p.noise) < 0.1);
        // A blip is the shortest thing here and a powerup the longest.
        assert!(mean(Category::Blip, |p| p.seconds) < mean(Category::Powerup, |p| p.seconds) / 4.0);
    }

    #[test]
    fn every_category_is_audible() {
        // A roll that renders silence is a wasted click, and `gain` is not the
        // only way to get there — a long attack on a short sound can eat it.
        for cat in Category::ALL {
            for seed in 0..64u32 {
                let pcm = synth::render(&roll(cat, seed));
                assert!(
                    !pcm.is_empty(),
                    "{} seed {seed} rendered nothing",
                    cat.name()
                );
                assert!(
                    synth::peak(&pcm) > 0.05,
                    "{} seed {seed} is inaudible",
                    cat.name()
                );
            }
        }
    }

    #[test]
    fn two_seeds_are_two_sounds() {
        // A category is a region, not a preset: pressing it twice has to give
        // two different sounds that are both the category.
        for cat in Category::ALL {
            let n = (0..128u32)
                .map(|s| {
                    let p = roll(cat, s);
                    format!("{:?}{:.0}{:.3}", p.wave, p.hz, p.seconds)
                })
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            assert!(
                n > 100,
                "{} gave only {n} distinct rolls of 128",
                cat.name()
            );
        }
    }
}
