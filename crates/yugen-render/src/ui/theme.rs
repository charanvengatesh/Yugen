//! The design tokens: every colour and every type size the overlay may use.
//!
//! # Why this module exists now and did not before
//!
//! The port transcribed its colours literally — seventy-five `rgba(...)` calls
//! in `ui` alone, each one the exact CSS the TypeScript wrote at that spot — and
//! [`super`]'s own comment says why: *"every one of them is transcribed
//! literally below, so that a constant here can be diffed against its line in
//! the original."* That was the right call while the port was being proved. It
//! stops being the right call the moment somebody wants to change how the game
//! LOOKS, because there is no such thing as changing a colour: there is only
//! finding all seventy-five and hoping.
//!
//! The port is over. These are names.
//!
//! # The direction
//!
//! 幽玄 — *yūgen* — is the aesthetic of what is suggested rather than stated:
//! the shape under the mist, not the mist and not the shape. That is a strange
//! thing to ask of a HUD, whose entire job is to state things. The reading this
//! module takes is about HIERARCHY, not obscurity:
//!
//! - **The world is the subject and the overlay is not.** Plates are dark and
//!   desaturated, they sit UNDER the text rather than beside it, and none of
//!   them carries a colour of its own. Nothing in the chrome competes with a
//!   lava glow or a snowfield for attention.
//! - **Ink is graded, not coloured.** Four steps from full white down to a hint
//!   that is nearly gone. The TypeScript had this instinct already — its comment
//!   *"Labels dim, values bright"* is a statement about alpha, not hue — and
//!   this makes it a scale instead of a habit.
//! - **Colour means something or it is absent.** Exactly one accent, one warm
//!   and one cold state hue, and the vitality ramp. An element with no state to
//!   report is grey. This is the rule the old palette broke most often: the
//!   armour readout was blue and the XP readout was gold for no reason either
//!   of them could have named.
//!
//! # What is NOT here
//!
//! `content/PALETTE.md` governs the colour of the WORLD — materials, their luma
//! and chroma bands, their edge gains — and is a far more rigorous document than
//! this one. It has nothing to do with the overlay and the two must not be
//! merged: a rule that makes granite read as granite has no opinion about a
//! health bar, and vice versa.

use bevy::prelude::*;

use super::{TextStyle, rgb, rgba};

// ---------------------------------------------------------------------------
// Ink
// ---------------------------------------------------------------------------

/// Primary text: a value the player is meant to read.
///
/// Full white and not off-white. This is drawn on plates that are already dark
/// and over a world that can be any brightness at all; every step away from
/// white here is a step towards unreadable over snow.
pub const INK: Color = rgb(0xff, 0xff, 0xff);

/// Secondary text: a label naming the value beside it.
///
/// The port's *"labels dim, values bright"*, made a token. Cool rather than
/// neutral grey, so a label reads as subordinate to its value rather than as
/// the same text at lower contrast.
pub const INK_DIM: Color = rgb(0x8c, 0xa0, 0xb4);

/// Tertiary text: a control hint, a caption, a count.
pub const INK_FAINT: Color = rgba(255, 255, 255, 0.6);

/// Text that is present but not currently applicable — a dash that is still
/// recharging, a recipe that cannot be afforded.
///
/// Deliberately close to the plate it sits on. Something unavailable should
/// take effort to read; that IS the signal, and it costs no colour to send.
pub const INK_MUTED: Color = rgba(255, 255, 255, 0.35);

// ---------------------------------------------------------------------------
// Ground
// ---------------------------------------------------------------------------

/// The standard plate behind a readout.
///
/// Black at 60%, which is the value the port arrived at empirically and there
/// is no reason to move: it is opaque enough to carry 11px text over lava and
/// transparent enough that the player can still see what is coming.
pub const PLATE: Color = rgba(0, 0, 0, 0.6);

/// A plate that has to carry small text over an unknown background — the F3
/// panel, a tooltip.
pub const PLATE_DENSE: Color = rgba(0, 0, 0, 0.66);

/// The full-screen scrim a card is drawn on.
///
/// Faintly blue rather than neutral, and that is the one place this palette
/// spends chroma on something that is not state: a pure black scrim reads as
/// the game having switched off, where a blue one reads as night falling over
/// it. The world is still there underneath and should still look like a world.
pub const SCRIM: Color = rgba(18, 20, 30, 0.82);

/// A recessed slot: a hotbar well, an empty inventory square.
pub const WELL: Color = rgb(0x3c, 0x3c, 0x3c);

/// A hairline between two surfaces, or the frame around a selected slot.
pub const EDGE: Color = rgba(255, 255, 255, 0.18);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// The one accent: a thing is ready, selected, or yours.
///
/// Cold, because every warm colour in this game already means damage, fire or
/// lava, and an accent that collides with a hazard is an accent that lies.
pub const ACCENT: Color = rgb(0x78, 0xc8, 0xff);

/// The accent, unavailable.
pub const ACCENT_SPENT: Color = rgb(0x46, 0x50, 0x5a);

/// Something is wrong but not yet harmful: a recipe short of one item, a block
/// too hard for the tool in hand.
pub const WARN: Color = rgb(0xd2, 0xc8, 0x8c);

/// Harm, and the death card.
pub const DANGER: Color = rgb(0xe6, 0x5a, 0x5a);

/// Health at full, and at nothing.
///
/// The port lerped these channel by channel with two hard-coded ramps
/// (`220 - 130 * frac`, `60 + 140 * frac`). [`vitality`] is that arithmetic with
/// its endpoints given names, so the ramp can be retuned by moving a colour
/// rather than by editing a coefficient.
pub const VITAL_FULL: Color = rgb(0x5a, 0xc8, 0x3c);

/// See [`VITAL_FULL`].
pub const VITAL_EMPTY: Color = rgb(0xdc, 0x3c, 0x3c);

/// The health ramp at `frac` of full, `0.0..=1.0`.
///
/// Linear in sRGB and not in a perceptual space, because that is what the
/// original did and the two endpoints are far enough apart that nothing
/// interesting happens in between. The clamp is load-bearing: `health` can
/// exceed `MAX_HEALTH` for a frame after a heal.
pub fn vitality(frac: f32) -> Color {
    let frac = frac.clamp(0.0, 1.0);
    let a = VITAL_EMPTY.to_srgba();
    let b = VITAL_FULL.to_srgba();
    Color::srgb(
        a.red + (b.red - a.red) * frac,
        a.green + (b.green - a.green) * frac,
        a.blue + (b.blue - a.blue) * frac,
    )
}

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

/// A hotbar slot's key digit, and nothing else. [`super::Face::Small`].
pub const KEY: TextStyle = TextStyle::for_px(8);

/// Stack counts and the flavour line.
pub const CAPTION: TextStyle = TextStyle::for_px(10);

/// Stat values and secondary chrome.
pub const MINOR: TextStyle = TextStyle::for_px(11);

/// Control hints and the cursor-feedback note.
pub const HINT: TextStyle = TextStyle::for_px(12);

/// The default: an item name, the HP readout, a toast.
pub const BODY: TextStyle = TextStyle::for_px(13);

/// A card's control line.
pub const CARD_HINT: TextStyle = TextStyle::for_px(14);

/// A card's subtitle.
pub const CARD_BODY: TextStyle = TextStyle::for_px(18);

/// The death card's headline.
pub const CARD_DEAD: TextStyle = TextStyle::for_px(52);

/// The title card's headline.
pub const CARD_TITLE: TextStyle = TextStyle::for_px(56);

// ---------------------------------------------------------------------------
// Space
// ---------------------------------------------------------------------------

/// The spacing scale, in buffer pixels.
///
/// Four steps and a base unit, rather than the fourteen distinct offsets the
/// port accumulated (`+4`, `+8`, `+12`, `+14`, `+28`...). Every one of those was
/// right where it stood and none of them was chosen relative to any other, which
/// is why the HUD, the toast and the F3 panel each ended up with their own idea
/// of what "clear of that" means.
pub const UNIT: i32 = 4;

/// Inside a plate, between its edge and its content.
pub const PAD: i32 = 2 * UNIT;

/// Between two related things — a label and its value, a pip and its caption.
pub const GAP: i32 = 3 * UNIT;

/// Between two unrelated things, and from the buffer edge to a plate.
///
/// Sixteen, which is what all three ported files used on every edge. Wide
/// enough that a plate does not fuse with the frame at zoom 2, narrow enough
/// not to eat the view.
pub const MARGIN: i32 = 4 * UNIT;

#[cfg(test)]
mod tests {
    use super::*;

    /// Same colour to within a 255th, which is the finest distinction an 8-bit
    /// channel can carry. `a + (b - a) * 1.0` is not bit-identical to `b`.
    fn same(a: Color, b: Color) -> bool {
        let (a, b) = (a.to_srgba(), b.to_srgba());
        (a.red - b.red).abs() < 1.0 / 255.0
            && (a.green - b.green).abs() < 1.0 / 255.0
            && (a.blue - b.blue).abs() < 1.0 / 255.0
    }

    #[test]
    fn the_vitality_ramp_hits_both_of_its_endpoints() {
        assert!(same(vitality(1.0), VITAL_FULL));
        assert!(same(vitality(0.0), VITAL_EMPTY));
    }

    #[test]
    fn the_vitality_ramp_clamps_rather_than_extrapolating() {
        // `health` can exceed `MAX_HEALTH` for a frame after a heal, and a
        // ramp that extrapolated would hand the painter a colour outside sRGB.
        assert!(same(vitality(2.0), VITAL_FULL));
        assert!(same(vitality(-1.0), VITAL_EMPTY));
    }

    #[test]
    fn the_ramp_is_monotone_from_empty_to_full() {
        // Health going up must never make the bar look worse.
        let mut last = vitality(0.0).to_srgba().green;
        for step in 1..=20 {
            let g = vitality(step as f32 / 20.0).to_srgba().green;
            assert!(g >= last, "the ramp doubles back at {step}/20");
            last = g;
        }
    }

    #[test]
    fn the_spacing_scale_is_whole_multiples_of_the_unit() {
        // Every plate edge lands on a whole buffer pixel by construction, and
        // the scale is what keeps two plates from disagreeing by one.
        for step in [PAD, GAP, MARGIN] {
            assert_eq!(step % UNIT, 0, "{step} is not on the spacing scale");
        }
        const { assert!(PAD < GAP && GAP < MARGIN, "the scale is not ascending") };
    }

    #[test]
    fn the_type_scale_ascends() {
        let ladder = [
            KEY, CAPTION, MINOR, HINT, BODY, CARD_HINT, CARD_BODY, CARD_DEAD, CARD_TITLE,
        ];
        for pair in ladder.windows(2) {
            assert!(
                pair[1].cap_h() >= pair[0].cap_h(),
                "the type scale doubles back: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
    }
}
