//! The world clock.
//!
//! Ported from `src/render/daynight.ts`.
//!
//! # Where the clock lives, and why it is not in the sim
//!
//! Here, on the render side, ticked with the frame dt. Deliberately NOT in
//! `yugen-core`'s sim: the cellular automata is the sole writer of world state
//! and must stay reproducible from `(seed, edits)`, and a wall clock feeding into
//! it would make every chunk's history depend on when you happened to walk past.
//! `crates/yugen-core/tests/worldgen_purity.rs` is the test that would fail if
//! that ever stopped being true.
//!
//! Time of day is purely a lighting and mood parameter, so it lives where
//! nothing downstream can perturb a cell. The one consumer outside this crate's
//! draw passes is creature spawning, which reads it through
//! [`Daylight`](crate::mobs::Daylight) — a weight on WHICH species may spawn,
//! never on whether the world generates the same way twice.
//!
//! # What the port changed
//!
//! The TypeScript recomputed one reused `DayPhase` object in place each tick and
//! said so in its header: "nothing here allocates after construction". A
//! [`DayPhase`] is fourteen floats and `Copy`, so there is nothing to reuse and
//! nothing to allocate. The "read it, don't retain it" warning on the getter went
//! with the sharing it was warning about — a copy cannot go stale.

use bevy::prelude::*;

use crate::mobs::Daylight;

/// Real seconds in one in-game day.
///
/// Five minutes: long enough to feel like a cycle rather than a strobe, short
/// enough that a play session sees several.
///
/// A module constant and not a `config` export, which is where the port puts a
/// number describing the whole game. It stays here because it describes ONE
/// algorithm — the mapping from accumulated seconds to a cycle position — and
/// because the TypeScript kept it next to that mapping for the same reason.
pub const DAY_LENGTH_S: f32 = 300.0;

/// Where the cycle starts a fresh world. Mid-morning.
///
/// Not midnight, and not noon: a run that opens in the dark reads as broken, and
/// one that opens at full noon never shows the player that there is a cycle at
/// all until five minutes in.
pub const DEFAULT_START: f32 = 0.34;

/// Sampled state of the cycle.
///
/// Every field is derived from [`DayPhase::t`] and nothing else, so this is a
/// pure function of the accumulated time — which is what makes the whole module
/// testable without a clock, a frame, or a GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DayPhase {
    /// Cycle position in `[0, 1)`: 0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset.
    pub t: f32,
    /// Sun elevation, -1 (deep night) to +1 (noon).
    pub elevation: f32,
    /// Master daylight weight, 0 at night to 1 at midday. Drives sky and ambient.
    pub day: f32,
    /// `1 - day`. Drives stars, fireflies, ember visibility.
    pub night: f32,
    /// Peaks at sunrise and sunset, 0 otherwise. Drives the warm horizon wash.
    pub twilight: f32,

    /// Sun disc position as a view fraction across, 0..1.
    pub sun_x: f32,
    /// Sun disc position as a view fraction down, 0..1.
    pub sun_y: f32,
    /// Sun disc visibility, 0..1.
    pub sun_a: f32,

    /// Moon disc position across, same convention as the sun.
    pub moon_x: f32,
    /// Moon disc position down.
    pub moon_y: f32,
    /// Moon disc visibility, 0..1.
    pub moon_a: f32,
}

impl DayPhase {
    /// What a human would call this time of day.
    ///
    /// Classified from the same weights everything else reads, rather than from
    /// `t` directly, so the word and the light always agree — a readout that
    /// said "night" over a lit sky would be worse than no readout. Twilight
    /// wins over both because it is the narrow band and the other two overlap
    /// it on either side.
    pub fn name(self) -> &'static str {
        if self.twilight > 0.5 {
            if self.elevation > 0.0 { "dusk" } else { "dawn" }
        } else if self.day > 0.5 {
            "day"
        } else {
            "night"
        }
    }

    /// The phase at cycle position `t`, which need not be in `[0, 1)`.
    ///
    /// This is the whole model. [`DayNight`] is an accumulator around it and
    /// holds no state this cannot be derived from.
    pub fn at(t: f32) -> DayPhase {
        let t = t - t.floor();

        // Elevation: -1 at midnight, 0 at the horizon crossings, +1 at noon.
        let elevation = -(t * core::f32::consts::TAU).cos();

        // Daylight ramps THROUGH the horizon crossing rather than at it, so dawn
        // and dusk are lit but not full day.
        let day = smoothstep01((elevation + 0.16) / 0.46);
        // Warmth is a narrow band around the horizon, symmetric on both sides.
        let twilight = smoothstep01(1.0 - elevation.abs() / 0.34);

        // The sun arcs left to right across the upper view between sunrise and
        // sunset. `sun_u` is 0 at sunrise and 1 at sunset.
        let sun_u = (t - 0.25) * 2.0;
        // The moon runs the same arc half a cycle out of phase, wrapped into
        // `[0, 2)` so the wrap lands off the right edge and it re-enters from the
        // left as it rises.
        let moon_u = ((t + 0.25) * 2.0) % 2.0;

        DayPhase {
            t,
            elevation,
            day,
            night: 1.0 - day,
            twilight,
            sun_x: sun_u,
            sun_y: 0.88 - (clamp01(sun_u) * core::f32::consts::PI).sin() * 0.74,
            sun_a: clamp01((elevation + 0.1) / 0.22),
            moon_x: moon_u,
            moon_y: 0.88 - (clamp01(moon_u) * core::f32::consts::PI).sin() * 0.74,
            moon_a: clamp01((-elevation + 0.1) / 0.22),
        }
    }
}

#[inline]
fn clamp01(v: f32) -> f32 {
    // `f32::clamp` panics only on a NaN *bound*, and both bounds here are
    // literals, so it cannot. A NaN input propagates, which is what the
    // TypeScript's ternary chain did too — neither branch of `v < 0 ? … : v > 1`
    // is taken for a NaN, so it fell through to `v`.
    v.clamp(0.0, 1.0)
}

#[inline]
fn smoothstep01(v: f32) -> f32 {
    let t = clamp01(v);
    t * t * (3.0 - 2.0 * t)
}

/// The accumulator: real seconds in, a cycle position out.
#[derive(Clone, Copy, Debug)]
pub struct DayNight {
    t: f32,
}

impl DayNight {
    /// A clock at cycle position `start`.
    pub fn new(start: f32) -> DayNight {
        DayNight {
            t: start - start.floor(),
        }
    }

    /// Advance by a frame.
    ///
    /// The wrap is applied every tick rather than left to accumulate, which is
    /// what keeps `t` small enough that adding a 120 Hz frame's worth of cycle
    /// (about `2.8e-5`) stays exactly representable in `f32`. Letting it grow
    /// would quietly lose the tick entirely after a few hours of play.
    pub fn update(&mut self, dt: f32) {
        let t = self.t + dt / DAY_LENGTH_S;
        self.t = t - t.floor();
    }

    /// The live phase.
    pub fn phase(&self) -> DayPhase {
        DayPhase::at(self.t)
    }

    /// The raw cycle position, `[0, 1)`.
    pub fn t(&self) -> f32 {
        self.t
    }
}

impl Default for DayNight {
    fn default() -> DayNight {
        DayNight::new(DEFAULT_START)
    }
}

// ---------------------------------------------------------------------------
// The one clock
// ---------------------------------------------------------------------------

/// The world clock, as a resource.
///
/// [`DayNight`] and [`DayPhase`] above carry no Bevy at all — that is what lets
/// the model be tested without an app, a frame or a GPU — so the one line that
/// makes it a resource lives down here, next to the plugin that ticks it.
///
/// `init_resource` rather than `insert_resource` in [`DayNightPlugin`], so an app
/// that wants to open a world at a particular hour can insert the clock first and
/// have the plugin respect it.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct WorldClock(pub DayNight);

/// Ticks the world clock, and publishes the one value the sim is allowed to see.
///
/// # There must be exactly one of these
///
/// This plugin exists because there were nearly three. The sky, the light
/// composite and the ambience emitters each need the phase, none of them owns
/// the others, and each was independently written to tick a clock of its own —
/// which would have run the day at three times speed the moment all three were
/// in the app. Each also wrote the hazard down rather than assuming it away, so
/// the fix is this: one owner, in the module the clock is named after, and every
/// consumer reads [`WorldClock`] without advancing it.
///
/// If you are adding a pass that needs the time of day: take `Res<WorldClock>`
/// and call `.0.phase()`. Do not call `update`.
///
/// # The one wire into the sim
///
/// [`Daylight`] is the single value that crosses back out of the render side:
/// the creature spawner weights nocturnal species by it. That is a weight on
/// WHICH species may spawn, never on whether a chunk generates the same way
/// twice, so worldgen stays a pure function of `(seed, edits)` — see this
/// module's header, and `crates/yugen-core/tests/worldgen_purity.rs`, which is
/// the test that would fail if that ever stopped being true.
pub struct DayNightPlugin;

impl Plugin for DayNightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldClock>()
            .init_resource::<Daylight>()
            .add_systems(Update, advance_clock);
    }
}

/// Advance the world clock and mirror its daylight weight into [`Daylight`].
///
/// One write a frame, and it is what closes the seam `crate::mobs` left open:
/// nocturnal creatures start preferring the night the moment this plugin is in
/// the app, with no change on that side.
fn advance_clock(time: Res<Time>, mut clock: ResMut<WorldClock>, mut daylight: ResMut<Daylight>) {
    clock.0.update(time.delta_secs());
    daylight.0 = clock.0.phase().day;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cycle_position_wraps_into_the_unit_interval() {
        assert_eq!(DayNight::new(1.25).t(), 0.25);
        assert_eq!(DayNight::new(0.25).t(), 0.25);
        // A negative start wraps forward, not to zero: `-0.25` is dusk, and
        // `f32::floor` is what makes that work where truncation would not.
        assert_eq!(DayNight::new(-0.25).t(), 0.75);
    }

    #[test]
    fn a_full_day_of_updates_returns_to_the_start() {
        let mut clock = DayNight::new(0.0);
        // 120 Hz for exactly one day.
        let dt = 1.0 / 120.0;
        for _ in 0..(DAY_LENGTH_S * 120.0) as u32 {
            clock.update(dt);
        }
        assert!(
            clock.t() < 1e-3 || clock.t() > 1.0 - 1e-3,
            "after one day the clock should be back at midnight, was {}",
            clock.t()
        );
    }

    #[test]
    fn the_four_quarters_are_where_their_names_say() {
        // These four are the contract every other module reads. Midnight is the
        // darkest, noon the brightest, and the two crossings sit between.
        assert_eq!(DayPhase::at(0.0).elevation, -1.0);
        assert!(
            DayPhase::at(0.25).elevation.abs() < 1e-6,
            "sunrise crossing"
        );
        assert_eq!(DayPhase::at(0.5).elevation, 1.0);
        assert!(DayPhase::at(0.75).elevation.abs() < 1e-6, "sunset crossing");

        assert_eq!(DayPhase::at(0.0).day, 0.0, "midnight is not lit");
        assert_eq!(DayPhase::at(0.5).day, 1.0, "noon is fully lit");
    }

    #[test]
    fn day_and_night_always_sum_to_one() {
        for i in 0..1000 {
            let p = DayPhase::at(i as f32 / 1000.0);
            assert!(
                (p.day + p.night - 1.0).abs() < 1e-6,
                "day+night drifted at t={}",
                p.t
            );
            assert!((0.0..=1.0).contains(&p.day));
        }
    }

    #[test]
    fn twilight_peaks_at_the_horizon_crossings_and_not_between() {
        let sunrise = DayPhase::at(0.25).twilight;
        let sunset = DayPhase::at(0.75).twilight;
        assert_eq!(sunrise, 1.0);
        assert_eq!(sunset, 1.0);
        assert_eq!(DayPhase::at(0.0).twilight, 0.0, "midnight is not warm");
        assert_eq!(DayPhase::at(0.5).twilight, 0.0, "noon is not warm");
        // Symmetric about both crossings — the wash reads the same going down as
        // it did coming up.
        assert!((DayPhase::at(0.22).twilight - DayPhase::at(0.28).twilight).abs() < 1e-6);
    }

    #[test]
    fn the_two_discs_cross_fade_rather_than_double_expose() {
        // Both ARE briefly visible at once, and that is correct — near a horizon
        // crossing one is setting while the other rises, which is what a real sky
        // does. What must never happen is both at full brightness, which would
        // read as two suns.
        //
        // The alpha ramps are `(±elev + 0.1) / 0.22`, so wherever both are
        // unclamped they sum to `0.2 / 0.22`, and clamping only ever caps the
        // total at 1. That bound is the actual invariant.
        for i in 0..1000 {
            let p = DayPhase::at(i as f32 / 1000.0);
            assert!(
                p.sun_a + p.moon_a <= 1.0 + 1e-6,
                "the discs over-expose at t={}: sun {} moon {}",
                p.t,
                p.sun_a,
                p.moon_a
            );
            assert!(
                p.sun_a.min(p.moon_a) < 0.5 + 1e-6,
                "both discs near full at t={}: sun {} moon {}",
                p.t,
                p.sun_a,
                p.moon_a
            );
        }
    }

    #[test]
    fn each_disc_is_alone_in_the_sky_at_its_own_extreme() {
        // The overlap is confined to the horizon band: at noon and at midnight
        // exactly one disc is up.
        let noon = DayPhase::at(0.5);
        assert_eq!((noon.sun_a, noon.moon_a), (1.0, 0.0));
        let midnight = DayPhase::at(0.0);
        assert_eq!((midnight.sun_a, midnight.moon_a), (0.0, 1.0));
    }

    #[test]
    fn the_sun_crosses_the_sky_from_left_to_right() {
        // Sunrise on the left edge, noon overhead, sunset on the right.
        assert_eq!(DayPhase::at(0.25).sun_x, 0.0);
        assert!((DayPhase::at(0.5).sun_x - 0.5).abs() < 1e-6);
        assert_eq!(DayPhase::at(0.75).sun_x, 1.0);
        // And it is HIGHEST at noon, which in view fractions means smallest y.
        let noon = DayPhase::at(0.5).sun_y;
        assert!(noon < DayPhase::at(0.25).sun_y);
        assert!(noon < DayPhase::at(0.75).sun_y);
    }

    #[test]
    fn the_default_start_is_lit_but_not_noon() {
        // A fresh world opens in mid-morning: bright enough to see, far enough
        // from noon that the cycle is visibly moving.
        let p = DayNight::default().phase();
        assert!(p.day > 0.5, "a fresh world should open lit, was {}", p.day);
        assert!(p.t < 0.5, "and before noon, was {}", p.t);
    }

    #[test]
    fn the_phase_is_a_pure_function_of_the_accumulated_time() {
        // Two clocks that reach the same position by different routes agree.
        // This is the property that lets every consumer take a copy.
        let mut stepped = DayNight::new(0.0);
        for _ in 0..600 {
            stepped.update(0.1);
        }
        let direct = DayNight::new(stepped.t());
        assert_eq!(stepped.phase(), direct.phase());
    }
}
