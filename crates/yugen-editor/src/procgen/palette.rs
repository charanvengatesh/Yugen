//! Building a palette instead of picking nine colours by hand.
//!
//! # Why HSL and not RGB
//!
//! The operation a pixel artist actually wants is "the same colour, darker",
//! and in RGB that is a multiply, which is exactly the thing that makes
//! generated ramps look dead. Lightness is a coordinate in HSL, so a ramp is a
//! walk along one axis and the hue is free to move independently — which is the
//! whole trick, see [`ramp`].
//!
//! # Slot 0 is not a colour
//!
//! FORMAT.md §5 makes index 0 the transparent slot, written `"."`. Everything
//! here preserves that: a generated palette starts with `"."` and the generated
//! colours begin at index 1. A generator that wrote a colour there would author
//! a value nothing ever draws.

use crate::raster;

/// The ceiling on generated ramps.
///
/// A frame character is one digit and index 0 is transparent, so nine colours is
/// what the format can name. Restated here rather than reached for from the
/// palette panel because this module has no UI above it — the constraint is the
/// file format's, not the window's.
pub const MAX_STEPS: usize = 9;

/// A colour as hue, saturation and lightness.
///
/// `h` is degrees and wraps; `s` and `l` are `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hsl {
    pub h: f32,
    pub s: f32,
    pub l: f32,
}

impl Hsl {
    pub fn new(h: f32, s: f32, l: f32) -> Hsl {
        Hsl {
            h: h.rem_euclid(360.0),
            s: s.clamp(0.0, 1.0),
            l: l.clamp(0.0, 1.0),
        }
    }
}

/// RGB to HSL.
pub fn to_hsl(rgb: [u8; 3]) -> Hsl {
    let (r, g, b) = (
        f32::from(rgb[0]) / 255.0,
        f32::from(rgb[1]) / 255.0,
        f32::from(rgb[2]) / 255.0,
    );
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    let d = max - min;
    if d <= f32::EPSILON {
        // Grey. Hue is undefined rather than zero, and 0 is the conventional
        // stand-in; saturation of 0 is what makes it not matter.
        return Hsl { h: 0.0, s: 0.0, l };
    }
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    Hsl {
        h: (h * 60.0).rem_euclid(360.0),
        s,
        l,
    }
}

/// HSL to RGB.
pub fn to_rgb(c: Hsl) -> [u8; 3] {
    let (h, s, l) = (
        c.h.rem_euclid(360.0),
        c.s.clamp(0.0, 1.0),
        c.l.clamp(0.0, 1.0),
    );
    if s <= f32::EPSILON {
        let v = (l * 255.0).round() as u8;
        return [v, v, v];
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue = |mut t: f32| -> f32 {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    let h = h / 360.0;
    [
        (hue(h + 1.0 / 3.0) * 255.0).round() as u8,
        (hue(h) * 255.0).round() as u8,
        (hue(h - 1.0 / 3.0) * 255.0).round() as u8,
    ]
}

/// A colour as the `#rrggbb` a record writes.
pub fn hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

/// An `n`-step ramp from `base`, as a whole palette including the `"."` slot.
///
/// Returns `n + 1` entries: index 0 transparent, then `n` colours running from
/// `l.0` to `l.1`. `n` is clamped to [`MAX_STEPS`].
///
/// # The hue shift is the point
///
/// `hue_shift` is degrees **per step**, and it is what separates a ramp from a
/// brightness slider. Shading that only moves lightness is a greyscale multiply
/// over one hue, and it reads as plastic; every palette in `content/` shifts
/// warm into the lights or cool into the darks instead. Passing `0.0` gives the
/// plastic version, which is occasionally what a metal wants.
///
/// Saturation moves only at the SHADOW end, toward [`RAMP_SHADOW_SATURATION`].
/// This used to pull both ends down symmetrically, and measuring [`REFERENCE`]
/// says that is wrong — see the constant.
pub fn ramp(base: Hsl, n: usize, hue_shift: f32, l: (f32, f32)) -> Vec<String> {
    let n = n.clamp(1, MAX_STEPS);
    let mut out = Vec::with_capacity(n + 1);
    out.push(".".to_string());
    for i in 0..n {
        // `n == 1` has no span to walk, and dividing by zero would put the one
        // colour at NaN rather than at the dark end.
        let t = if n == 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        let light = l.0 + (l.1 - l.0) * t;
        // 1 at the darkest step, 0 at the lightest. Squared so the pull is felt
        // in the last step or two rather than spread evenly up the ramp, which
        // is how the reference's ramps actually behave: the top three steps of
        // its red hold saturation within 0.06 of each other and the bottom one
        // drops by 0.34.
        let shadow = (1.0 - t) * (1.0 - t);
        let sat = base.s + (RAMP_SHADOW_SATURATION - base.s) * shadow;
        out.push(hex(to_rgb(Hsl::new(
            base.h + hue_shift * (i as f32 - (n as f32 - 1.0) / 2.0),
            sat,
            light,
        ))));
    }
    out
}

/// What saturation a ramp's darkest step is pulled toward.
///
/// Measured, not chosen. Three ramps out of [`REFERENCE`], lightest to darkest:
///
/// ```text
///   red         s = .87 .89 .81 .69 .64 .30     l = .74 -> .17
///   purple      s = .84 .98 .53 .45 .46 .54     l = .75 -> .17
///   blue-grey   s = .24 .22 .18 .25 .30 .33     l = .82 -> .15
/// ```
///
/// Two things fall out of that and neither was what this module did before.
///
/// **The highlight end does not desaturate.** The old code pulled both ends down
/// symmetrically; the reference holds its lightest steps at full saturation and
/// spends the change entirely in shadow. A symmetric pull makes highlights
/// chalky, which is the single most recognisable tell of a generated ramp.
///
/// **The pull is toward a value, not downward.** The saturated red loses 0.57 of
/// its saturation going dark; the near-neutral blue-grey GAINS 0.09. One rule
/// produces both: shadows converge on a middling saturation. A rule that only
/// subtracted would have to special-case every near-grey ramp.
///
/// 0.38 sits between the red's 0.30 and the blue-grey's 0.33 and reproduces the
/// direction of both.
const RAMP_SHADOW_SATURATION: f32 = 0.38;

/// The reference palette: 64 colours, read out of the indexed PNG of the
/// character sheet this tree's art is being drawn against.
///
/// # Provenance, because it matters here
///
/// These are not eyeballed. The source is a colour-indexed PNG and this is its
/// `PLTE` chunk, entries 1..64 — the transparent slot excluded. So the numbers
/// are exact rather than sampled, which is what makes [`snap`] worth having: a
/// palette recovered from a screenshot would carry compression error into every
/// record that snapped to it.
///
/// The other sheet in the same folder — the item icons — is **not** this
/// palette and was not used for any number in this module. It is an upscaled
/// JPEG: 63 000 distinct colours over a 64-colour-looking image, only 0.5% of
/// its pixels within 12 RGB of an entry here, and a background grey that is not
/// in this list at all. It is a good look at how icons are OUTLINED and lit and
/// a bad source for a colour, and it was used for exactly the first of those.
///
/// # Why a palette is worth hardcoding
///
/// Not to be imposed — nothing here forces a record to use it, and
/// `content/PALETTE.md` remains the tree's own authority. It is here so that
/// [`snap`] has something to snap TO. A generated ramp is arithmetic and lands
/// wherever the arithmetic lands; art that reads as one set is art whose colours
/// come from one small list. Giving the generator that list is the difference
/// between "plausible colours" and "colours that match the other eighty
/// records".
///
/// The order is the source palette's own: greys, then blue-greys, then a hue
/// wheel of ramps. Nothing depends on the order — [`snap`] searches the whole
/// list — but keeping it makes the ramps legible when read as a table.
pub const REFERENCE: [&str; 64] = [
    "#131313", "#1b1b1b", "#272727", "#3d3d3d", "#5d5d5d", "#858585", "#b4b4b4", "#ffffff",
    "#c7cfdd", "#92a1b9", "#657392", "#424c6e", "#2a2f4e", "#1a1932", "#0e071b", "#1c121c",
    "#391f21", "#5d2c28", "#8a4836", "#bf6f4a", "#e69c69", "#f6ca9f", "#f9e6cf", "#edab50",
    "#e07438", "#c64524", "#8e251d", "#ff5000", "#ed7614", "#ffa214", "#ffc825", "#ffeb57",
    "#d3fc7e", "#99e65f", "#5ac54f", "#33984b", "#1e6f50", "#134c4c", "#0c2e44", "#00396d",
    "#0069aa", "#0098dc", "#00cdf9", "#0cf1ff", "#94fdff", "#fdd2ed", "#f389f5", "#db3ffd",
    "#7a09fa", "#3003d9", "#0c0293", "#03193f", "#3b1443", "#622461", "#93388f", "#ca52c9",
    "#c85086", "#f68187", "#f5555d", "#ea323c", "#c42430", "#891e2b", "#571c27", "#ff0040",
];

/// Move every colour in `pal` to its nearest entry in `to`.
///
/// # Nearest in what
///
/// In HSL, with lightness weighted heaviest and hue weighted by how saturated
/// the colour is. Plain RGB distance is the obvious choice and it is wrong here
/// for one specific reason: it treats a lightness error and a hue error as the
/// same size, and they are not. Getting the lightness wrong breaks the SHADING —
/// the thing that carries the form of an 8x8 drawing — while getting the hue
/// slightly wrong just makes it a neighbouring colour.
///
/// Weighting hue by saturation is the other half: two near-greys can sit 180
/// degrees apart and look identical, so hue distance has to stop mattering as
/// the colour approaches grey, or every dark neutral snaps somewhere arbitrary.
///
/// The `"."` slot and any entry that is not a colour are copied through
/// untouched, exactly as in [`harmonize`].
pub fn snap(pal: &[String], to: &[&str]) -> Vec<String> {
    let targets: Vec<(String, Hsl)> = to
        .iter()
        .filter_map(|s| raster::parse_hex(s).map(|rgb| ((*s).to_string(), to_hsl(rgb))))
        .collect();
    if targets.is_empty() {
        return pal.to_vec();
    }

    pal.iter()
        .map(|entry| {
            let Some(rgb) = raster::parse_hex(entry) else {
                return entry.clone();
            };
            let c = to_hsl(rgb);
            let mut best = &targets[0];
            let mut best_d = f32::MAX;
            for t in &targets {
                // Around the circle the short way — 350 and 10 are 20 apart.
                let dh = ((c.h - t.1.h + 540.0).rem_euclid(360.0) - 180.0) / 180.0;
                let ds = c.s - t.1.s;
                let dl = c.l - t.1.l;
                let hue_weight = c.s.min(t.1.s);
                let d = (dl * dl) * 4.0 + (ds * ds) + (dh * dh) * hue_weight * 2.0;
                if d < best_d {
                    best_d = d;
                    best = t;
                }
            }
            best.0.clone()
        })
        .collect()
}

/// Pull every colour in `pal` toward one hue.
///
/// What turns nine colours picked at different times into a palette. `amount` is
/// `0..=1` and **0 is exactly the identity** — including on the `"."` slot and on
/// any entry that is not a colour, both of which are copied through untouched
/// rather than parsed and re-emitted.
///
/// Only hue and saturation move. Lightness is what carries the drawing's form,
/// and harmonising it would flatten the shading the art depends on.
pub fn harmonize(pal: &[String], toward: Hsl, amount: f32) -> Vec<String> {
    let amount = amount.clamp(0.0, 1.0);
    pal.iter()
        .map(|entry| {
            let Some(rgb) = raster::parse_hex(entry) else {
                // `"."`, or something malformed. Either way this is not the
                // module that gets to have an opinion about it.
                return entry.clone();
            };
            let c = to_hsl(rgb);
            // Around the circle the short way: 350 and 10 are twenty degrees
            // apart, and a linear blend would send it the other three hundred
            // and forty.
            let delta = (toward.h - c.h + 540.0).rem_euclid(360.0) - 180.0;
            hex(to_rgb(Hsl::new(
                c.h + delta * amount,
                c.s + (toward.s - c.s) * amount,
                c.l,
            )))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_colour_survives_the_round_trip() {
        // Within rounding: the conversion goes through f32 and back to bytes.
        for rgb in [
            [0u8, 0, 0],
            [255, 255, 255],
            [128, 128, 128],
            [231, 180, 113],
            [25, 36, 46],
            [33, 135, 126],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ] {
            let back = to_rgb(to_hsl(rgb));
            for i in 0..3 {
                let d = i32::from(back[i]) - i32::from(rgb[i]);
                assert!(d.abs() <= 1, "{rgb:?} came back as {back:?}");
            }
        }
    }

    #[test]
    fn every_real_palette_survives_the_round_trip() {
        // The colours actually in the tree, not colours I chose. A conversion
        // that was wrong about one corner of the space would show up here as a
        // shifted hue in a record somebody drew.
        let pal = ["#19242e", "#e7b471", "#21877e", "#ffcf5c", "#b46a27"];
        for entry in pal {
            let rgb = raster::parse_hex(entry).expect("a colour");
            let back = to_rgb(to_hsl(rgb));
            for i in 0..3 {
                assert!(
                    (i32::from(back[i]) - i32::from(rgb[i])).abs() <= 1,
                    "{entry}"
                );
            }
        }
    }

    #[test]
    fn a_ramp_is_monotone_in_lightness() {
        // The property that makes it a ramp. If the hue shift or the end
        // desaturation could reorder the steps, index 5 would not reliably be
        // lighter than index 4 and shading would stop meaning anything.
        let pal = ramp(Hsl::new(30.0, 0.6, 0.5), 6, 12.0, (0.15, 0.9));
        assert_eq!(pal.len(), 7, "the transparent slot plus six colours");
        assert_eq!(pal[0], ".");

        let mut last = -1.0;
        for entry in &pal[1..] {
            let l = to_hsl(raster::parse_hex(entry).expect("a colour")).l;
            assert!(l > last, "{pal:?} is not monotone at {entry}");
            last = l;
        }
    }

    #[test]
    fn a_ramp_shifts_hue_as_it_walks() {
        // Without this a ramp is a brightness slider over one hue, which is what
        // makes a generated palette read as generated.
        let pal = ramp(Hsl::new(30.0, 0.6, 0.5), 5, 15.0, (0.2, 0.85));
        let first = to_hsl(raster::parse_hex(&pal[1]).expect("a colour")).h;
        let last = to_hsl(raster::parse_hex(&pal[5]).expect("a colour")).h;
        assert!(
            (last - first).abs() > 20.0,
            "hue barely moved: {first} to {last}"
        );

        // And zero really is no shift, for the cases that want it.
        let flat = ramp(Hsl::new(30.0, 0.6, 0.5), 5, 0.0, (0.2, 0.85));
        let a = to_hsl(raster::parse_hex(&flat[1]).expect("a colour")).h;
        let b = to_hsl(raster::parse_hex(&flat[5]).expect("a colour")).h;
        assert!(
            (a - b).abs() < 2.0,
            "hue moved without being asked: {a} to {b}"
        );
    }

    #[test]
    fn a_ramp_keeps_its_highlight_saturated_and_spends_it_in_shadow() {
        // The shape measured off `REFERENCE`, and the thing the old symmetric
        // desaturation got wrong. A chalky highlight is the loudest tell that a
        // ramp was generated.
        let pal = ramp(Hsl::new(357.0, 0.87, 0.5), 6, 0.0, (0.17, 0.74));
        let sats: Vec<f32> = pal[1..]
            .iter()
            .map(|e| to_hsl(raster::parse_hex(e).expect("a colour")).s)
            .collect();

        let (dark, light) = (sats[0], sats[sats.len() - 1]);
        assert!(
            light > 0.80,
            "the lightest step desaturated to {light} — highlights must hold"
        );
        assert!(
            dark < 0.50,
            "the darkest step held {dark} — shadows are where saturation is spent"
        );
        assert!(dark < light, "{sats:?} is not spending saturation downward");
    }

    #[test]
    fn a_near_grey_ramp_gains_saturation_going_dark_instead_of_losing_it() {
        // The other half of why the rule is "converge on a value" and not
        // "subtract". `REFERENCE`'s blue-grey walks s .24 -> .33 as it darkens;
        // a rule that only subtracted would drive it to zero and produce six
        // identical greys.
        let pal = ramp(Hsl::new(218.0, 0.24, 0.5), 6, 5.0, (0.15, 0.82));
        let sats: Vec<f32> = pal[1..]
            .iter()
            .map(|e| to_hsl(raster::parse_hex(e).expect("a colour")).s)
            .collect();
        assert!(
            sats[0] > sats[sats.len() - 1],
            "a near-grey should gain saturation in shadow, got {sats:?}"
        );
    }

    #[test]
    fn the_reference_palette_is_sixty_four_distinct_colours() {
        // A duplicate would be a transcription slip, and `snap` would silently
        // prefer whichever came first.
        let mut seen = std::collections::BTreeSet::new();
        for entry in REFERENCE {
            assert!(
                raster::parse_hex(entry).is_some(),
                "{entry} is not a colour"
            );
            assert!(seen.insert(entry), "{entry} appears twice");
        }
        assert_eq!(seen.len(), 64);
    }

    #[test]
    fn snapping_a_palette_to_itself_is_the_identity() {
        // The property that makes `snap` safe to press twice. If a colour did
        // not choose ITSELF as its own nearest entry, the metric would be
        // broken and repeated presses would walk the palette around.
        let pal: Vec<String> = REFERENCE.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(snap(&pal, &REFERENCE), pal);
    }

    #[test]
    fn snapping_lands_everything_on_the_reference() {
        // Whatever goes in, what comes out is drawn from the list — which is the
        // whole point: art reads as one set when its colours come from one small
        // list rather than from arithmetic.
        let generated = ramp(Hsl::new(30.0, 0.6, 0.5), MAX_STEPS, 12.0, (0.1, 0.9));
        let out = snap(&generated, &REFERENCE);
        assert_eq!(out[0], ".", "the transparent slot is not a colour");
        for entry in &out[1..] {
            assert!(
                REFERENCE.contains(&entry.as_str()),
                "{entry} is not in the reference"
            );
        }
    }

    #[test]
    fn snapping_preserves_the_order_of_a_ramps_lightness() {
        // A snap that reordered the ramp would scramble the shading of every
        // frame drawn against it — the digits in the art do not move, so index 5
        // must still be lighter than index 4 afterwards.
        let generated = ramp(Hsl::new(210.0, 0.5, 0.5), 6, 6.0, (0.12, 0.88));
        let out = snap(&generated, &REFERENCE);
        let mut last = -1.0;
        for entry in &out[1..] {
            let l = to_hsl(raster::parse_hex(entry).expect("a colour")).l;
            assert!(l >= last, "{out:?} is no longer monotone in lightness");
            last = l;
        }
    }

    #[test]
    fn snapping_a_near_grey_does_not_fly_across_the_hue_wheel() {
        // Why hue distance is weighted by saturation. Two near-greys can sit 180
        // degrees apart and look the same, so an unweighted metric sends every
        // dark neutral somewhere arbitrary.
        let grey = vec!["#3d3d3f".to_string()];
        let out = snap(&grey, &REFERENCE);
        let c = to_hsl(raster::parse_hex(&out[0]).expect("a colour"));
        assert!(c.s < 0.35, "{} is not a neutral", out[0]);
        assert!((c.l - 0.24).abs() < 0.12, "{} is the wrong value", out[0]);
    }

    #[test]
    fn a_ramp_never_exceeds_what_a_frame_character_can_name() {
        // A frame character is one digit, so nine colours plus transparent is
        // the ceiling. Asking for twenty has to give nine, not twenty.
        let pal = ramp(Hsl::new(200.0, 0.5, 0.5), 20, 8.0, (0.1, 0.9));
        assert_eq!(pal.len(), MAX_STEPS + 1);
        assert!(pal.len() <= 10);
    }

    #[test]
    fn a_one_step_ramp_is_a_colour_and_not_a_nan() {
        // `n - 1` is the divisor for the walk, and one step has no span.
        let pal = ramp(Hsl::new(200.0, 0.5, 0.5), 1, 8.0, (0.1, 0.9));
        assert_eq!(pal.len(), 2);
        assert!(
            raster::parse_hex(&pal[1]).is_some(),
            "{:?} is not a colour",
            pal[1]
        );
    }

    #[test]
    fn harmonizing_by_nothing_changes_nothing() {
        // The identity has to be exact, because the palette splice compares the
        // strings: a "no-op" that re-emitted `#FFF` as `#ffffff` would rewrite
        // the `pal` line and put churn in a diff.
        let pal: Vec<String> = [".", "#19242e", "#e7b471", "#21877e"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(harmonize(&pal, Hsl::new(200.0, 0.5, 0.5), 0.0), pal);
    }

    #[test]
    fn harmonizing_leaves_the_transparent_slot_alone() {
        let pal: Vec<String> = [".", "#19242e"].iter().map(|s| (*s).to_string()).collect();
        let out = harmonize(&pal, Hsl::new(200.0, 0.9, 0.5), 1.0);
        assert_eq!(out[0], ".", "slot 0 is a placeholder, not a colour");
        assert_ne!(out[1], pal[1], "and everything else did move");
    }

    #[test]
    fn harmonizing_all_the_way_lands_on_the_target_hue() {
        let pal: Vec<String> = [".", "#e7b471", "#21877e"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let out = harmonize(&pal, Hsl::new(200.0, 0.5, 0.5), 1.0);
        for entry in &out[1..] {
            let h = to_hsl(raster::parse_hex(entry).expect("a colour")).h;
            assert!((h - 200.0).abs() < 3.0, "{entry} landed at {h}");
        }
    }

    #[test]
    fn harmonizing_takes_the_short_way_round_the_circle() {
        // 350 and 10 are twenty degrees apart. A linear blend would walk the
        // other three hundred and forty and pass through every hue on the way,
        // so a half-strength harmonise would land on cyan instead of on red.
        let from = vec![hex(to_rgb(Hsl::new(350.0, 0.8, 0.5)))];
        let out = harmonize(&from, Hsl::new(10.0, 0.8, 0.5), 0.5);
        let h = to_hsl(raster::parse_hex(&out[0]).expect("a colour")).h;
        // The midpoint of 350 and 10 is 0, so the landing zone straddles the
        // wrap — which is the whole point of the test and why this is two
        // comparisons rather than one `contains`.
        assert!(
            !(5.0..=355.0).contains(&h),
            "the blend went the long way round: landed at {h}"
        );
    }

    #[test]
    fn harmonizing_does_not_touch_lightness() {
        // Lightness is what carries the form of the drawing. Harmonising it
        // would flatten the shading the art depends on.
        let pal: Vec<String> = [".", "#19242e", "#e7b471"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let out = harmonize(&pal, Hsl::new(200.0, 0.9, 0.5), 1.0);
        for (before, after) in pal[1..].iter().zip(&out[1..]) {
            let a = to_hsl(raster::parse_hex(before).expect("a colour")).l;
            let b = to_hsl(raster::parse_hex(after).expect("a colour")).l;
            assert!((a - b).abs() < 0.02, "{before} -> {after} moved lightness");
        }
    }

    #[test]
    fn every_generated_entry_is_a_colour_the_raster_accepts() {
        // The generators feed `pal`, and `raster::palette` is what reads it back.
        // A `#rrggbb` this emitted that the raster refused would be a magenta
        // canvas rather than an error.
        for h in (0..360).step_by(17) {
            let pal = ramp(Hsl::new(h as f32, 0.7, 0.5), MAX_STEPS, 10.0, (0.05, 0.95));
            raster::palette(&pal).unwrap_or_else(|e| panic!("hue {h}: {e}"));
        }
    }
}
