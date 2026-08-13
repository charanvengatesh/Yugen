//! Extra frames derived from one that is already drawn.
//!
//! # Everything here is feet-anchored
//!
//! A sprite is drawn inside a rectangle whose bottom edge is where the creature
//! stands. `ArtGrid::from_def` puts all of a sprite's vertical surplus above the
//! art and `MobDef::build` does the same, so **the bottom row of a frame is the
//! ground** and a derived frame that moved it would make the creature sink or
//! hover for those frames only.
//!
//! That is why [`squash`] drops a row from the top and doubles the bottom one
//! rather than scaling: the row count is fixed by `cellsH`, so something has to
//! give, and the top is the end that is allowed to.
//!
//! # None of these change the shape
//!
//! Same invariant as [`crate::procgen::ops`], for the same reason: these frames go
//! into a sequence of a record that already declares its grid.

use crate::procgen::ops::{self, CLEAR};
use crate::sprite::Frame;

/// A two-frame bob: the drawing, and the drawing lifted by `amp`.
///
/// The cheapest idle animation there is, and the one most of this game's
/// creatures want. Lifting rather than dropping keeps the base frame as the
/// resting pose, so a sequence that is paused looks like the drawing that was
/// made.
///
/// Nothing wraps — a lift that wrapped would bring the feet in at the top.
pub fn bob(base: &Frame, amp: i32) -> Vec<Frame> {
    vec![base.clone(), ops::shift(base, 0, -amp, false)]
}

/// A two-frame squash: the drawing, and the drawing compressed by one row.
///
/// The row count cannot change, so the compression takes a row off the top and
/// repeats the bottom one. Repeating rather than blending because there is
/// nothing to blend: a texel is a palette index, and the average of index 3 and
/// index 7 is not a colour between them.
pub fn squash(base: &Frame) -> Vec<Frame> {
    if base.len() < 2 {
        return vec![base.clone(), base.clone()];
    }
    let w = ops::width(base);
    let mut out: Vec<String> = vec![CLEAR.to_string().repeat(w)];
    // Rows 1.. of the original, then its last row again: the drawing loses its
    // top row and gains a duplicate of the row standing on the ground.
    out.extend(base[1..].iter().cloned());
    out.push(base[base.len() - 1].clone());
    // Two rows were added and one removed, so trim from the top — which is the
    // end that is allowed to move.
    let over = out.len() - base.len();
    vec![base.clone(), out[over..].to_vec()]
}

/// Runs of opaque texels along the bottom row, as `(start, len)`.
///
/// The bottom row is where a creature meets the ground, so its runs are its
/// feet. Measured across the 54 characters of the reference sheet: 29 have two
/// separate groups — the classic pair of one-texel stubs with a gap — 22 have
/// one solid base, and 3 have three.
fn feet(base: &Frame) -> Vec<(usize, usize)> {
    let Some(row) = base.last() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut start = None;
    for (x, ch) in row.chars().enumerate() {
        match (ops::is_clear(ch), start) {
            (false, None) => start = Some(x),
            (true, Some(s)) => {
                out.push((s, x - s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, row.chars().count() - s));
    }
    out
}

/// A four-frame walk, made by lifting alternate feet.
///
/// # Why the feet and not the whole body
///
/// At eight texels a walk cycle has no room to swing a limb — there is no limb,
/// there are one or two texels of foot. What reads as walking at this size is
/// **contact**: which foot is on the ground. So the cycle is
/// `[rest, lift evens, rest, lift odds]`, and the rest frame between steps is
/// what keeps it from reading as a jitter.
///
/// A foot lifts by leaving the bottom row and appearing one row up, if that cell
/// is free. If it is not — the body already fills it — the foot simply comes off
/// the ground, which at this scale is still the whole of the signal.
///
/// Degrades rather than refuses. A creature with one solid base has nothing to
/// alternate, so this gives it a hop; one with three groups alternates them
/// odd against even. Neither is wrong, and both beat having no `run` pose.
pub fn walk(base: &Frame) -> Vec<Frame> {
    let groups = feet(base);
    if groups.is_empty() || base.len() < 2 {
        return vec![base.clone(), base.clone()];
    }
    let step = |parity: usize| -> Frame {
        let mut rows: Vec<Vec<char>> = base.iter().map(|r| r.chars().collect()).collect();
        let last = rows.len() - 1;
        // The ground row and the one above it are held at the same time, so they
        // are split rather than indexed twice.
        let (upper, lower) = rows.split_at_mut(last);
        let above = &mut upper[last - 1];
        let ground = &mut lower[0];
        for (i, &(start, len)) in groups.iter().enumerate() {
            if i % 2 != parity {
                continue;
            }
            for (x, cell) in ground.iter_mut().enumerate().skip(start).take(len) {
                let foot = *cell;
                *cell = CLEAR;
                if ops::is_clear(above[x]) {
                    above[x] = foot;
                }
            }
        }
        rows.into_iter().map(|r| r.into_iter().collect()).collect()
    };
    vec![base.clone(), step(0), base.clone(), step(1)]
}

/// The drawing with `eye` replaced by `lid` — one frame, not a sequence.
///
/// A blink is a single different frame that a sequence's `blinkEvery` and
/// `blinkFor` schedule; the schema already owns the timing, so producing a
/// two-frame sequence here would be inventing a second way to say it.
///
/// This is a [`ops::remap`] of one index and is written as its own function
/// because "which index is the eye" is the entire question and a remap table
/// hides it.
pub fn blink(base: &Frame, eye: char, lid: char) -> Frame {
    base.iter()
        .map(|row| {
            row.chars()
                .map(|c| if c == eye { lid } else { c })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: &[&str]) -> Frame {
        rows.iter().map(|r| (*r).to_string()).collect()
    }

    fn shape_held(base: &Frame, outs: &[Frame], what: &str) {
        let w = ops::width(base);
        for (i, f) in outs.iter().enumerate() {
            assert_eq!(
                f.len(),
                base.len(),
                "{what} frame {i} changed the row count"
            );
            for row in f {
                assert_eq!(row.chars().count(), w, "{what} frame {i} changed the width");
            }
        }
    }

    #[test]
    fn every_derived_frame_keeps_the_grid_it_came_from() {
        let base = frame(&["1111", "1221", "1221", "1111"]);
        shape_held(&base, &bob(&base, 1), "bob");
        shape_held(&base, &squash(&base), "squash");
        shape_held(&base, &[blink(&base, '2', '1')], "blink");
    }

    #[test]
    fn a_bob_keeps_the_drawing_as_its_first_frame() {
        // A paused sequence should look like the art somebody made.
        let base = frame(&["..", "11"]);
        let out = bob(&base, 1);
        assert_eq!(out[0], base);
        assert_eq!(out[1], frame(&["11", ".."]), "the second frame is lifted");
    }

    #[test]
    fn a_bob_does_not_wrap_the_feet_in_at_the_top() {
        // Wrapping would bring the bottom row — the ground contact — round to
        // the ceiling for half of every idle cycle.
        let base = frame(&["..", "..", "11"]);
        let out = bob(&base, 2);
        assert_eq!(out[1], frame(&["11", "..", ".."]));
        let out = bob(&base, 3);
        assert_eq!(out[1], frame(&["..", "..", ".."]), "lifted clean off");
    }

    #[test]
    fn a_squash_keeps_the_bottom_row_on_the_ground() {
        // The bottom row is where the creature stands. A compression that moved
        // it would make the creature sink for those frames.
        let base = frame(&["12", "34", "56"]);
        let out = squash(&base);
        assert_eq!(out[0], base);
        assert_eq!(
            out[1].last().expect("a row"),
            base.last().expect("a row"),
            "the ground contact moved"
        );
        assert_eq!(out[1], frame(&["34", "56", "56"]));
    }

    #[test]
    fn a_squash_of_a_one_row_frame_is_not_a_panic() {
        let base = frame(&["12"]);
        let out = squash(&base);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], base);
        assert_eq!(out[1], base);
    }

    #[test]
    fn a_walk_alternates_which_foot_is_down() {
        // The whole signal at eight texels: not a swinging limb, but which foot
        // is touching the ground on which frame.
        let base = frame(&["1111", "1111", "1..1"]);
        let out = walk(&base);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], base, "the cycle rests on the drawing it came from");
        assert_eq!(out[2], base);
        // Frame 1 lifts the left foot, frame 3 the right — and each lifts one,
        // never both, or the creature is jumping rather than walking.
        assert_eq!(out[1], frame(&["1111", "1111", "...1"]));
        assert_eq!(out[3], frame(&["1111", "1111", "1..."]));
    }

    #[test]
    fn a_lifted_foot_appears_one_row_up_when_there_is_room() {
        // A foot that only vanished would read as the creature losing a leg;
        // showing it raised is what makes it a step.
        let base = frame(&["..", "1."]);
        let out = walk(&base);
        assert_eq!(out[1], frame(&["1.", ".."]), "the foot moved up, not away");
    }

    #[test]
    fn a_foot_with_no_room_above_still_leaves_the_ground() {
        // The body already fills the cell above, so there is nowhere to draw the
        // raised foot. Coming off the ground is still the signal.
        let base = frame(&["111", "1.1"]);
        let out = walk(&base);
        assert_eq!(out[1], frame(&["111", "..1"]));
        assert_eq!(out[3], frame(&["111", "1.."]));
    }

    #[test]
    fn a_solid_base_hops_because_it_has_nothing_to_alternate() {
        // 22 of the reference sheet's 54 characters have one unbroken bottom row
        // — a blob, not a biped. There is no second foot to put down, so the
        // cycle lifts the whole thing on one beat and rests on the other. That
        // is a hop, and a hop is the honest reading of a creature with no legs.
        let base = frame(&["11", "11"]);
        let out = walk(&base);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], base);
        assert_eq!(
            out[1],
            frame(&["11", ".."]),
            "the whole base left the ground"
        );
        assert_eq!(out[3], base, "and the odd beat has nothing to lift");
    }

    #[test]
    fn a_walk_keeps_the_grid_it_came_from() {
        let base = frame(&["1111", "1221", "1..1"]);
        shape_held(&base, &walk(&base), "walk");
    }

    #[test]
    fn a_creature_with_nothing_on_the_ground_does_not_panic() {
        // Degrading rather than refusing: an airborne or empty frame has no feet
        // to alternate, and a `run` pose that is two still frames beats none.
        let base = frame(&["11", ".."]);
        let out = walk(&base);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], base);
    }

    #[test]
    fn feet_are_found_as_separate_groups() {
        assert_eq!(feet(&frame(&["..1..1.."])), vec![(2, 1), (5, 1)]);
        assert_eq!(feet(&frame(&[".111111."])), vec![(1, 6)]);
        assert_eq!(feet(&frame(&["1......1"])), vec![(0, 1), (7, 1)]);
        assert_eq!(feet(&frame(&["........"])), vec![]);
    }

    #[test]
    fn a_blink_only_touches_the_eye() {
        let base = frame(&["1121", "1211"]);
        assert_eq!(blink(&base, '2', '1'), frame(&["1111", "1111"]));
    }

    #[test]
    fn a_blink_for_an_index_that_is_not_there_changes_nothing() {
        // Picking the wrong index should be a no-op you can see, not a frame
        // that silently lost something else.
        let base = frame(&["1111"]);
        assert_eq!(blink(&base, '7', '1'), base);
    }
}
