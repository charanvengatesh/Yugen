//! Where things go: rectangles, anchored to the buffer, that know about each
//! other.
//!
//! # The problem this replaces
//!
//! Five parts of the overlay draw into the same two corners, and until now each
//! one carried its own arithmetic for staying out of the others' way:
//! `MARGIN`, `PANEL_H`, `TOAST_UP = 96`, `NOTE_UP = 122`, and the F3 panel's
//! `Y = 52`. Every one of those is a correct number and none of them is derived
//! from any other. The comment on `Y` is the honest account of how they were
//! obtained:
//!
//! > The plate is drawn 10px above this, so 52 puts its top edge at 42 — clear
//! > of `ui::hud`'s health plate, which runs from `MARGIN - 4` = 12 to
//! > `MARGIN - 4 + BAR_H + 8` = 40. **The first value here was 46, which
//! > overlapped it by four pixels; that was found by looking at a capture, not
//! > by arithmetic.**
//!
//! That is a layout system made of constants that have read each other's source
//! code. It works until something moves — and Departure Mono just moved every
//! run of text a seventh wider, which is exactly the kind of change that turns
//! four such numbers into four overlaps nobody notices until a screenshot.
//!
//! # The shape
//!
//! A [`Region`] is a whole-pixel rectangle. [`Chrome`] is the set of regions the
//! standing parts of the overlay have claimed, derived ONCE from the [`View`],
//! so that "below the health plate" is a field read rather than a number copied.
//! Everything is pure arithmetic on `i32` — no Bevy, no resources — so a layout
//! question can be answered in a unit test with no app.
//!
//! Regions do not enforce anything. Nothing stops a caller drawing outside the
//! one it was given, and adding a clip would mean [`super::UiPrim`] carrying a
//! scissor rect that the painter would have to honour, for a class of bug that
//! `the_standing_claims_do_not_overlap` catches for free.

use yugen_core::config::View;

use super::theme::MARGIN;

/// Which corner of the parent an anchored region is measured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// A whole-pixel rectangle in buffer space, `+y` DOWN.
///
/// `+y` down to match [`super::UiPrim`] and every layout function in this
/// crate. Bevy's own `+y` is up and the flip happens once, in `quad_centre`, at
/// the very end — see [`super`]'s header on why that boundary is where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Region {
    /// The whole buffer.
    pub const fn of(view: View) -> Region {
        Region {
            x: 0,
            y: 0,
            w: view.w,
            h: view.h,
        }
    }

    /// The buffer inside its standard margin — where chrome is allowed to live.
    pub const fn safe(view: View) -> Region {
        Region::of(view).inset(MARGIN)
    }

    /// This region, pulled in by `by` on every edge.
    ///
    /// Saturating rather than panicking: a region can legitimately be inset to
    /// nothing on a small buffer, and a zero-sized region draws zero-sized
    /// prims, which the painter already skips.
    pub const fn inset(self, by: i32) -> Region {
        Region {
            x: self.x + by,
            y: self.y + by,
            w: if self.w > 2 * by { self.w - 2 * by } else { 0 },
            h: if self.h > 2 * by { self.h - 2 * by } else { 0 },
        }
    }

    /// A `w` x `h` box in one corner of this region.
    pub const fn corner(self, at: Corner, w: i32, h: i32) -> Region {
        let (x, y) = match at {
            Corner::TopLeft => (self.x, self.y),
            Corner::TopRight => (self.right() - w, self.y),
            Corner::BottomLeft => (self.x, self.bottom() - h),
            Corner::BottomRight => (self.right() - w, self.bottom() - h),
        };
        Region { x, y, w, h }
    }

    /// Take `h` pixels off the top, and what is left below them.
    pub const fn split_top(self, h: i32) -> (Region, Region) {
        let h = if h < self.h { h } else { self.h };
        (
            Region { h, ..self },
            Region {
                y: self.y + h,
                h: self.h - h,
                ..self
            },
        )
    }

    /// This region moved `dy` down. Negative moves it up.
    pub const fn offset_y(self, dy: i32) -> Region {
        Region {
            y: self.y + dy,
            ..self
        }
    }

    /// The x just past the right edge.
    pub const fn right(self) -> i32 {
        self.x + self.w
    }

    /// The y just past the bottom edge.
    pub const fn bottom(self) -> i32 {
        self.y + self.h
    }

    /// Horizontal centre, floored.
    pub const fn cx(self) -> i32 {
        self.x + self.w / 2
    }

    /// Vertical centre, floored.
    pub const fn cy(self) -> i32 {
        self.y + self.h / 2
    }

    /// Does any pixel belong to both?
    ///
    /// Half-open on both axes, so two regions that share an edge do not
    /// overlap — which is what makes a `split_top` pair report as disjoint.
    pub const fn overlaps(self, other: Region) -> bool {
        self.w > 0
            && self.h > 0
            && other.w > 0
            && other.h > 0
            && self.x < other.right()
            && other.x < self.right()
            && self.y < other.bottom()
            && other.y < self.bottom()
    }
}

/// The regions the standing parts of the overlay have claimed.
///
/// Derived once from the [`View`] and passed down, so that no two callers can
/// hold different opinions about where the health plate ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chrome {
    /// The health bar and its plate, top-left.
    pub vitals: Region,
    /// The stat row under the vitals.
    pub stats: Region,
    /// The control hints, top-right.
    pub hints: Region,
    /// The hotbar and its label rows, bottom-left.
    pub panel: Region,
    /// Where a diagnostic panel may start: below everything the player needs.
    pub instrument: Region,
}

/// Height of the health plate, including its bleed. `BAR_H` plus 4 px either
/// side, which is the plate the port drew.
const VITALS_H: i32 = 28;

/// Width of the health plate, including its bleed. `BAR_W` plus 4 px either
/// side.
const VITALS_W: i32 = 228;

/// Height of the stat row under the vitals.
const STATS_H: i32 = 12;

/// Height of the bottom-left hotbar plate.
///
/// 62: a 26px swatch row, the 2px the selected slot lifts into, and two lines
/// of label under it. Two lines is the budget the panel has.
pub const PANEL_H: i32 = 62;

/// Lines of control hint in the top-right stack.
const HINT_LINES: i32 = 3;

/// Pitch of the hint stack, in buffer px.
const HINT_PITCH: i32 = 14;

impl Chrome {
    /// Work out every standing claim for this buffer.
    ///
    /// Allocated by SPLITTING a band off the safe region for each claim in
    /// turn, rather than by anchoring each one independently and checking
    /// afterwards that they missed each other. Two regions carved out of a
    /// split cannot overlap — there is no arithmetic left for them to disagree
    /// about — which is the whole point of replacing the hand-tuned constants
    /// this module's header quotes.
    ///
    /// It also means the layout degrades rather than inverts on a buffer too
    /// small to hold everything. `split_top` yields at most what is there, so a
    /// claim that does not fit comes back empty and the painter skips it. The
    /// order below is the priority order: vitals before stats before the
    /// hotbar, and the instrument panel last, out of whatever is left.
    pub fn of(view: View) -> Chrome {
        let safe = Region::safe(view);

        let (top, rest) = safe.split_top(VITALS_H);
        let vitals = top.corner(Corner::TopLeft, VITALS_W.min(top.w), top.h);

        let (stats, rest) = rest.split_top(STATS_H);

        // The hints hang in the top-right across both of those bands. They are
        // the one claim NOT carved out of the stack: they are right-aligned
        // chrome over the top-left readouts, they are the first thing the HUD
        // fades out, and reserving a band for them would push the instrument
        // panel down the screen for text that is about to disappear.
        let hints = safe.corner(
            Corner::TopRight,
            (safe.w / 2).min(safe.w),
            (HINT_LINES * HINT_PITCH).min(safe.h),
        );

        // The hotbar off the bottom, then the instrument panel gets the gap
        // between the stat row and it.
        let panel_h = PANEL_H.min(rest.h);
        let (instrument, panel) = rest.split_top(rest.h - panel_h);

        Chrome {
            vitals,
            stats,
            hints,
            panel,
            instrument,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View::for_screen(1280, 720)
    }

    /// A buffer narrow enough to make every claim fight for room.
    fn cramped() -> View {
        View::for_screen(320, 240)
    }

    #[test]
    fn the_standing_claims_do_not_overlap() {
        // The property `Y = 52` was hand-tuned to satisfy, now checked rather
        // than eyeballed — and checked at both ends of the size range, which
        // eyeballing one capture could never do.
        for v in [view(), cramped(), View::for_screen(2560, 1440)] {
            let c = Chrome::of(v);
            for (a, an, b, bn) in [
                (c.vitals, "vitals", c.stats, "stats"),
                (c.vitals, "vitals", c.instrument, "instrument"),
                (c.stats, "stats", c.instrument, "instrument"),
                (c.vitals, "vitals", c.panel, "panel"),
                (c.instrument, "instrument", c.panel, "panel"),
            ] {
                assert!(!a.overlaps(b), "{an} overlaps {bn} at {}x{}", v.w, v.h);
            }
        }
    }

    #[test]
    fn every_claim_stays_inside_the_buffer() {
        for v in [view(), cramped()] {
            let whole = Region::of(v);
            let c = Chrome::of(v);
            for (r, name) in [
                (c.vitals, "vitals"),
                (c.stats, "stats"),
                (c.hints, "hints"),
                (c.panel, "panel"),
                (c.instrument, "instrument"),
            ] {
                assert!(r.x >= 0 && r.y >= 0, "{name} starts outside");
                assert!(
                    r.right() <= whole.right() && r.bottom() <= whole.bottom(),
                    "{name} runs past the buffer at {}x{}",
                    v.w,
                    v.h
                );
            }
        }
    }

    #[test]
    fn a_region_inset_past_its_own_size_is_empty_rather_than_inverted() {
        // The cramped case reaches this. A negative width would sign-flip every
        // `right()` downstream and put prims at plausible-looking coordinates
        // on the wrong side of the screen.
        let r = Region {
            x: 0,
            y: 0,
            w: 10,
            h: 4,
        };
        assert_eq!(
            r.inset(9),
            Region {
                x: 9,
                y: 9,
                w: 0,
                h: 0
            }
        );
    }

    #[test]
    fn an_empty_region_overlaps_nothing_including_itself() {
        let empty = Region {
            x: 5,
            y: 5,
            w: 0,
            h: 0,
        };
        let big = Region {
            x: 0,
            y: 0,
            w: 20,
            h: 20,
        };
        assert!(!empty.overlaps(big));
        assert!(!big.overlaps(empty));
        assert!(!empty.overlaps(empty));
    }

    #[test]
    fn splitting_gives_two_regions_that_tile_the_original() {
        let r = Region {
            x: 3,
            y: 7,
            w: 20,
            h: 30,
        };
        let (top, rest) = r.split_top(12);
        assert_eq!(top.h + rest.h, r.h);
        assert_eq!(top.y, r.y);
        assert_eq!(rest.bottom(), r.bottom());
        assert!(!top.overlaps(rest), "a split must not double-count a row");
    }

    #[test]
    fn corners_sit_against_the_edges_they_name() {
        let r = Region {
            x: 10,
            y: 20,
            w: 100,
            h: 50,
        };
        assert_eq!(r.corner(Corner::TopLeft, 8, 4).x, 10);
        assert_eq!(r.corner(Corner::TopLeft, 8, 4).y, 20);
        assert_eq!(r.corner(Corner::TopRight, 8, 4).right(), r.right());
        assert_eq!(r.corner(Corner::BottomLeft, 8, 4).bottom(), r.bottom());
        let br = r.corner(Corner::BottomRight, 8, 4);
        assert_eq!((br.right(), br.bottom()), (r.right(), r.bottom()));
    }
}
