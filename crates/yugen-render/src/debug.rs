//! The F3 panel: what the game thinks is happening, on screen, while it happens.
//!
//! # Why this exists
//!
//! The HUD tells the player their health, their dash and their hotbar, which is
//! everything a PLAYER needs and nothing a developer does. Every question worth
//! asking while looking at the game — *which biome is this, how deep am I, what
//! is that block, why is this corner dark, is anything alive nearby* — has until
//! now been answered by adding a `println!`, rebuilding, and reading a terminal
//! next to a window. `docs/HANDOFF.md` §5 recommends bisecting by commenting out
//! plugins, which is the same thing at a larger scale.
//!
//! Three bugs in `HANDOFF.md` §7.1 were of the shape *a resource is declared,
//! read, and written by nothing*, so every biome lit identically for two
//! milestones and nobody could see it. A panel that shows the resolved biome
//! weight next to the frame it produced is the cheapest instrument that would
//! have caught all three.
//!
//! # The shape, and why the readout is a plain struct
//!
//! [`DebugReadout`] is data with no Bevy in it, [`overlay`] is a pure function
//! from that data to [`UiPrim`]s, and [`gather`] is the only part that touches
//! the world. That split is not ceremony: it means the layout can be tested
//! headlessly with no GPU, no window and no world — the same property that makes
//! `ui::hud` and `ui::build_hud` testable, and the reason this module can afford
//! to be opinionated about its formatting.
//!
//! It also means the panel cannot lie by omission. Every field is filled by
//! `gather` from a live resource or explicitly marked absent; there is no
//! "compute it in the layout" path where a stale value can hide.
//!
//! # What it deliberately does not do
//!
//! It does not sample anything the frame did not already compute. The biome
//! weights come from the mood `crate::ambience` published this frame, the light
//! value from the grid `crate::light` already solved, the material from the
//! streaming window. Nothing here re-derives a value from noise, because a panel
//! that computed its own answer would agree with itself and not with the game.

use bevy::prelude::*;
use yugen_core::config::{CELL_SIZE, CHUNK_CELLS};
use yugen_core::input::{KEYS, KeyState};
use yugen_core::sim::biomes::{Biome, UndergroundLayerId};
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::materials::CellId;

use crate::ambience::AmbientLife;
use crate::input::BevyKeys;
use crate::input::CursorWorld;
use crate::light::{LightPass, depth_at};
use crate::mobs::Creatures;
use crate::player::PlayerBody;
use crate::ui::layout::Chrome;
use crate::ui::{Align, TextStyle, UiPrim, theme};
use crate::world::{SimWorld, WorldFocus};

/// Frames the frame-time average is taken over.
///
/// A single frame's delta is dominated by whatever the OS was doing during it
/// and flickers too fast to read. Half a second at 120 Hz is long enough to
/// settle and short enough that a stall is still visible as a bump rather than
/// being averaged into nothing.
const SMOOTHING_FRAMES: f32 = 60.0;

// --- The readout -------------------------------------------------------------

/// One cell, as far as anything can say what is there.
///
/// The pointer's when there is one, and the camera focus's when there is not.
/// That fallback is the difference between a panel that is useful in a headless
/// `--screenshot` capture and one that says "pointer is off the window" in every
/// capture anybody automates — which is most of them, and exactly the ones where
/// nobody is available to move a mouse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CursorReadout {
    /// Whether this came from the pointer. False means it is the focus cell.
    pub from_pointer: bool,
    /// Absolute cell coordinate.
    pub cell: WorldCell,
    /// The play plane's material there.
    pub material: CellId,
    /// The background wall plane's material there. `0` is open sky.
    pub wall: CellId,
    /// Solved light, 0..1, or `None` if the light pass has not run.
    pub light: Option<f32>,
}

/// Everything the panel shows, resolved. No Bevy, no borrows, no work left.
///
/// `Default` is a world that does not exist yet, which is a real state: the
/// panel can be toggled on the menu screen, and it should say so rather than
/// print a plausible-looking zero.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct DebugReadout {
    /// Smoothed frame time in milliseconds.
    pub frame_ms: f32,
    /// Whether a world exists. Everything below is meaningless when false.
    pub live: bool,
    /// The world seed.
    pub seed: u32,
    /// Camera focus in world px, sim convention (+y down).
    pub focus: Vec2,
    /// The cell the focus is in.
    pub cell: WorldCell,
    /// The chunk that cell is in.
    pub chunk: (i32, i32),
    /// 0 at the surface anchor, 1 at the bottom of the depth range.
    pub depth: f32,
    /// Dominant surface biome and its normalised weight.
    pub biome: (&'static str, f32),
    /// Dominant underground layer and its normalised weight.
    pub layer: (&'static str, f32),
    /// 0 outdoors, 1 in the underground layers, smooth between.
    pub underground: f32,
    /// Live creatures.
    pub mobs: usize,
    /// XP the mob system has banked. Nothing spends it yet, which is exactly
    /// why it is worth showing: a counter with no sink is invisible otherwise.
    pub xp: i32,
    /// The player's health, or `None` with no body (free camera).
    pub health: Option<f32>,
    /// The body's velocity in px/s, and whether it is standing on something.
    ///
    /// The first thing asked of any platformer bug and the last thing a
    /// screenshot can answer. "Why will it not jump" is `on_ground` and nothing
    /// else; "why is it drifting" is `vx` at rest.
    pub motion: Option<(f32, f32, bool)>,
    /// The buffer size and the zoom that produced it.
    ///
    /// `View::for_screen` is a pure function of the window size and everything
    /// in the overlay is laid out against its output, so a layout that looks
    /// wrong is a layout being given a buffer somebody did not expect.
    pub view: (i32, i32, f32),
    /// Time of day, `0.0..1.0`, and the phase name it falls in.
    pub clock: Option<(f32, &'static str)>,
    /// Live particles.
    pub particles: usize,
    /// Prims in this frame's display list, and quads in the painter's pool.
    ///
    /// The overlay's own cost, which nothing else reports. The pool only ever
    /// grows, so a gap between the two is the high-water mark of some screen
    /// that is no longer up.
    pub draw: (usize, usize),
    /// What the pointer is over, if it is over anything.
    pub cursor: Option<CursorReadout>,
}

/// Whether the panel is up. Off by default — this is an instrument, not chrome.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DebugOverlay(pub bool);

// --- The layout --------------------------------------------------------------

/// Widest the panel gets, in buffer px.
///
/// Clamped against `layout::Chrome`'s instrument region, so a buffer narrower
/// than this gets a narrower panel rather than one running off the edge. The
/// old code clamped against `view.w` while starting at x = 2, which overran any
/// view under 342 px.
const PANEL_W: i32 = 340;

/// Inset from the plate's edge to its text.
const PAD: i32 = 6;

/// Line pitch. [`ROW`] is 11px, so this is a comfortable single space.
const LINE: i32 = 13;

/// Column the values start in, so the labels do not have to be padded.
const VALUE_X: i32 = 92;

/// The 11px face, matching `ui`'s stat rows.
const ROW: TextStyle = TextStyle::for_px(11);

/// The panel, as a display list.
///
/// Pure: same readout in, same primitives out. Test it by calling it.
pub fn overlay(r: &DebugReadout, chrome: Chrome) -> Vec<UiPrim> {
    let mut rows: Vec<(&str, String)> = Vec::with_capacity(12);

    // Always true, world or no world, which is why it is first: a panel whose
    // top line is blank on the menu looks broken rather than idle.
    rows.push((
        "frame",
        format!(
            "{:.2} ms  ({:.0} fps)",
            r.frame_ms,
            1000.0 / r.frame_ms.max(0.001)
        ),
    ));

    if !r.live {
        rows.push(("world", "none — no world yet".into()));
    } else {
        rows.push(("seed", format!("{}", r.seed)));
        rows.push(("focus", format!("{:.0}, {:.0} px", r.focus.x, r.focus.y)));
        rows.push((
            "cell",
            format!(
                "{}, {}   chunk {}, {}",
                r.cell.x, r.cell.y, r.chunk.0, r.chunk.1
            ),
        ));
        rows.push((
            "depth",
            format!("{:.3}   underground {:.2}", r.depth, r.underground),
        ));
        rows.push(("biome", format!("{} {:.0}%", r.biome.0, r.biome.1 * 100.0)));
        rows.push(("layer", format!("{} {:.0}%", r.layer.0, r.layer.1 * 100.0)));
        if let Some((vx, vy, grounded)) = r.motion {
            rows.push((
                "motion",
                format!(
                    "{vx:.0}, {vy:.0} px/s   {}",
                    if grounded { "grounded" } else { "airborne" }
                ),
            ));
        }
        if let Some((t, phase)) = r.clock {
            // As a 24-hour clock as well as the raw phase: "0.72" is only
            // meaningful to somebody who already knows the answer.
            let mins = (t * 24.0 * 60.0) as i32;
            rows.push((
                "clock",
                format!("{:02}:{:02}  {phase}  ({t:.3})", mins / 60, mins % 60),
            ));
        }
        rows.push((
            "mobs",
            match r.health {
                Some(hp) => format!("{}   hp {hp:.0}   xp {}", r.mobs, r.xp),
                None => format!("{}   (no body)   xp {}", r.mobs, r.xp),
            },
        ));

        if let Some(c) = &r.cursor {
            rows.push((
                if c.from_pointer {
                    "pointer"
                } else {
                    "at focus"
                },
                format!("{}, {}   {}", c.cell.x, c.cell.y, material_name(c.material)),
            ));
            rows.push((
                "wall",
                match c.wall {
                    0 => "open sky".into(),
                    id => material_name(id).to_string(),
                },
            ));
            rows.push((
                "light",
                match c.light {
                    Some(l) => format!("{l:.3}"),
                    None => "off the light grid".into(),
                },
            ));
        }
    }

    // The region `layout::Chrome` set aside for an instrument, rather than the
    // hand-tuned `Y = 52` this used to carry. That constant was found by
    // looking at a capture after the first value overlapped the health plate by
    // four pixels; the region is derived from the plate instead, so it cannot.
    rows.push((
        "view",
        format!("{} x {} @ {:.2}x", r.view.0, r.view.1, r.view.2),
    ));
    rows.push((
        "draw",
        format!(
            "{} prims   {} quads   {} particles",
            r.draw.0, r.draw.1, r.particles
        ),
    ));

    let region = chrome.instrument;
    let w = PANEL_W.min(region.w);
    let h = (LINE * rows.len() as i32 + 8).min(region.h);
    let mut out = Vec::with_capacity(rows.len() * 2 + 1);

    // One plate behind the lot. The frame under this panel is arbitrary — snow,
    // lava, a lit cave — and 11px text on an unknown background is unreadable
    // exactly when it matters most.
    out.push(UiPrim::rect(region.x, region.y, w, h, theme::PLATE_DENSE));

    for (i, (label, value)) in rows.iter().enumerate() {
        let baseline = region.y + PAD + ROW.cap_h() + LINE * i as i32;
        out.push(UiPrim::text(
            label.to_string(),
            region.x + PAD,
            baseline,
            Align::Left,
            ROW,
            theme::INK_DIM,
        ));
        out.push(UiPrim::text(
            value.clone(),
            region.x + PAD + VALUE_X,
            baseline,
            Align::Left,
            ROW,
            theme::INK,
        ));
    }
    out
}

/// A material's authored display name, or a placeholder for an id the registry
/// does not know.
///
/// The fallback is not defensive padding: `CellId` is a `u16` read out of a cell,
/// and a bug that writes a wild id is exactly the kind of thing this panel exists
/// to make visible. Printing the number beats panicking inside a debug overlay.
fn material_name(id: CellId) -> &'static str {
    yugen_data::blocks::BLOCKS
        .get(id as usize)
        .map_or("?", |d| d.name)
}

// --- Gathering ---------------------------------------------------------------

/// Everything the panel reads. Bundled so [`gather`] stays inside clippy's
/// argument budget, the same way `light`'s solve does it.
#[derive(bevy::ecs::system::SystemParam)]
struct Sources<'w> {
    time: Res<'w, Time<Real>>,
    world: Option<Res<'w, SimWorld>>,
    focus: Res<'w, WorldFocus>,
    cursor: Res<'w, CursorWorld>,
    life: Option<Res<'w, AmbientLife>>,
    // Every one of these is optional for the reason the three above it are: a
    // host may run `DebugPlugin` without the rest of the game, and `gather`
    // runs in `PreUpdate` on the very first frame, before `Startup` has
    // inserted the render target. A missing one means that system is not in the
    // app, not that something failed.
    target: Option<Res<'w, crate::lowres::LowResTarget>>,
    clock: Option<Res<'w, crate::daynight::WorldClock>>,
    particles: Option<Res<'w, crate::particles::ParticleSystem>>,
    frame: Option<Res<'w, crate::ui::UiFrame>>,
    quads: Option<Res<'w, crate::ui::UiQuads>>,
    light: Option<Res<'w, LightPass>>,
    creatures: Option<Res<'w, Creatures>>,
    body: Option<Res<'w, PlayerBody>>,
}

/// Show and hide the panel.
///
/// # Which keys, and why two
///
/// `KEYS.debug` lists BOTH `F3` and `Backquote`, and both are needed. F3 is
/// where a decade of block games have put it, so it is the one a player will
/// try first. Backquote is the one that always works: on macOS, F3 is Mission
/// Control, and the window server takes the press before winit ever sees it —
/// a binding that silently does nothing on one of the three platforms this
/// builds for.
///
/// This used to be a hardcoded `KeyCode::F3` while `KEYS.debug` declared
/// Backquote and was read by nothing — precisely the "declared, read, and
/// written by nothing" shape this module's own header cites as its reason for
/// existing. It goes through the binding table now, like every other key.
fn toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut shown: ResMut<DebugOverlay>,
    mut settings: ResMut<crate::settings::Settings>,
) {
    if !BevyKeys(&keys).any_pressed(KEYS.debug) {
        return;
    }
    // BOTH, and they must stay equal. `settings::apply` pushes the preference
    // into [`DebugOverlay`] whenever `Settings` changes, so a key press that
    // moved only the resource would be silently undone by the next time the
    // player touched any other option. Writing both also means F3 survives a
    // restart, which is the behaviour somebody who leaves the panel up wants.
    shown.0 = !shown.0;
    settings.debug_overlay = shown.0;
}

/// Fill the readout from the live world.
///
/// The frame time comes from `Time<Real>` and not `Time`. `main.rs` installs
/// `TimeUpdateStrategy::ManualDuration` under `--script`, so the virtual clock
/// reports a synthetic 16.67 ms there — meaning the panel would claim a steady
/// 60 fps in exactly the runs used to measure what a frame costs.
///
/// Runs every frame regardless of whether the panel is up, and that is
/// deliberate: the frame-time average has to keep converging while the panel is
/// hidden, or the first number a developer sees after pressing F3 is a spike
/// from the frame that pressed it. The rest is a dozen resource reads and no
/// allocation.
fn gather(src: Sources, mut out: ResMut<DebugReadout>) {
    let dt = src.time.delta_secs() * 1000.0;
    // An exponential average, weighted so it settles over `SMOOTHING_FRAMES`.
    // Seeded on the first frame rather than ramping from zero, which would
    // otherwise read as an impossible 0.02 ms for the first half second.
    out.frame_ms = if out.frame_ms <= 0.0 {
        dt
    } else {
        out.frame_ms + (dt - out.frame_ms) / SMOOTHING_FRAMES
    };

    // Read before the early return: these are true whether or not a world
    // exists, and a panel that went blank on the menu would be a panel that
    // could not be used to diagnose the menu.
    if let Some(target) = src.target.as_ref() {
        out.view = (target.view.w, target.view.h, target.view.zoom);
    }
    out.particles = src.particles.as_ref().map_or(0, |p| p.live_count());
    out.draw = (
        src.frame.as_ref().map_or(0, |f| f.prims.len()),
        src.quads.as_ref().map_or(0, |q| q.pool_len()),
    );
    out.clock = src.clock.as_ref().map(|c| (c.0.t(), c.0.phase().name()));

    let Some(world) = src.world.as_ref() else {
        out.live = false;
        return;
    };
    out.live = true;
    out.seed = world.seed;
    out.focus = Vec2::new(src.focus.x, src.focus.y);
    out.cell = WorldCell::new(
        (src.focus.x / CELL_SIZE as f32).floor() as i32,
        (src.focus.y / CELL_SIZE as f32).floor() as i32,
    );
    out.chunk = (
        out.cell.x.div_euclid(CHUNK_CELLS),
        out.cell.y.div_euclid(CHUNK_CELLS),
    );
    out.depth = depth_at(src.focus.y);
    out.mobs = src.creatures.as_ref().map_or(0, |c| c.0.count());
    out.xp = src.creatures.as_ref().map_or(0, |c| c.0.xp_banked());
    out.health = src.body.as_ref().map(|b| b.0.health);
    out.motion = src.body.as_ref().map(|b| (b.0.vx, b.0.vy, b.0.on_ground));

    if let Some(life) = src.life.as_ref() {
        let mood = life.ambience.mood();
        out.underground = mood.underground;
        out.biome = dominant(Biome::ALL.iter().map(|b| (b.def().name, mood.biome(*b))));
        out.layer = dominant(
            UndergroundLayerId::ALL
                .iter()
                .map(|l| (l.def().name, mood.layer_of(*l))),
        );
    }

    // The pointer if there is one, the camera focus if there is not. Never
    // `None`: the focus always exists, so this section is never blank and a
    // headless capture reports the same three facts an interactive session does.
    let (at, from_pointer) = src.cursor.0.map_or((out.focus, false), |p| (p, true));
    let cell = WorldCell::new(
        (at.x / CELL_SIZE as f32).floor() as i32,
        (at.y / CELL_SIZE as f32).floor() as i32,
    );
    out.cursor = Some(CursorReadout {
        from_pointer,
        cell,
        material: world.level.grid.get_world(cell),
        wall: world.level.grid.get_back_world(cell),
        light: src
            .light
            .as_ref()
            .and_then(|p| p.light_at_world(at.x, at.y)),
    });
}

/// The heaviest-weighted entry of a normalised set.
///
/// Reported rather than the whole vector because a blend of eleven biomes is a
/// wall of numbers that answers no question; "Tundra 82%" answers the one that
/// gets asked, and the percentage is what says whether you are standing on a
/// boundary.
fn dominant(it: impl Iterator<Item = (&'static str, f32)>) -> (&'static str, f32) {
    it.fold(
        ("—", 0.0),
        |best, next| if next.1 > best.1 { next } else { best },
    )
}

// --- The plugin --------------------------------------------------------------

/// The F3 panel. Add it or do not; nothing else depends on it.
pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugOverlay>()
            .init_resource::<DebugReadout>()
            // `PreUpdate`, before `crate::ui`'s `compose` reads the readout in
            // `Update`, so the panel shows this frame's numbers rather than the
            // previous one's. `crate::ambience::publish_mood` is in the same
            // schedule for the same reason.
            //
            // `.after(InputSystems)` because that set is what repopulates
            // `just_pressed`, and `toggle` reads it. `input::gather_intent`
            // pins itself the same way and for the same reason; this did not,
            // and was relying on the scheduler's topological order to come out
            // the right way round.
            .add_systems(PreUpdate, (toggle.after(bevy::input::InputSystems), gather));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> yugen_core::config::View {
        yugen_core::config::View::for_screen(1280, 720)
    }

    fn chrome() -> Chrome {
        Chrome::of(view())
    }

    /// The whole point of the pure split: no world, no GPU, no app.
    #[test]
    fn a_dead_world_says_so_instead_of_printing_zeroes() {
        let r = DebugReadout {
            frame_ms: 8.3,
            ..DebugReadout::default()
        };
        let text = texts(&overlay(&r, chrome()));
        assert!(
            text.iter().any(|t| t.contains("no world yet")),
            "expected an explicit 'no world' line, got {text:?}"
        );
        assert!(
            !text.iter().any(|t| t.contains("seed")),
            "a dead world must not print a plausible-looking seed: {text:?}"
        );
    }

    #[test]
    fn a_live_world_reports_every_field_it_was_given() {
        let r = DebugReadout {
            frame_ms: 8.0,
            live: true,
            seed: 2334,
            focus: Vec2::new(300.0, 1000.0),
            cell: WorldCell::new(60, 200),
            chunk: (1, 6),
            depth: 0.25,
            biome: ("Tundra", 0.82),
            layer: ("Magma", 0.4),
            underground: 1.0,
            mobs: 7,
            xp: 130,
            health: Some(84.0),
            cursor: Some(CursorReadout {
                from_pointer: true,
                cell: WorldCell::new(61, 201),
                material: 0,
                wall: 0,
                light: Some(0.125),
            }),
            ..DebugReadout::default()
        };
        let text = texts(&overlay(&r, chrome())).join(" | ");
        for want in [
            "2334",
            "60, 200",
            "chunk 1, 6",
            "0.250",
            "Tundra 82%",
            "Magma 40%",
            "7",
            "hp 84",
            "xp 130",
            "61, 201",
            "open sky",
            "0.125",
            "125 fps",
        ] {
            assert!(text.contains(want), "missing {want:?} in: {text}");
        }
    }

    /// The cell section says WHERE it is reading from, so a capture with no
    /// mouse is not silently reporting the focus as if it were the pointer.
    #[test]
    fn the_cell_section_names_its_own_source() {
        let mut r = DebugReadout {
            live: true,
            frame_ms: 8.0,
            cursor: Some(CursorReadout {
                from_pointer: false,
                cell: WorldCell::new(4, 5),
                material: 0,
                wall: 0,
                light: Some(0.5),
            }),
            ..DebugReadout::default()
        };
        let focus = texts(&overlay(&r, chrome())).join(" | ");
        assert!(focus.contains("at focus"), "{focus}");
        assert!(!focus.contains("pointer"), "{focus}");

        r.cursor.as_mut().unwrap().from_pointer = true;
        let pointer = texts(&overlay(&r, chrome())).join(" | ");
        assert!(pointer.contains("pointer"), "{pointer}");
        assert!(!pointer.contains("at focus"), "{pointer}");

        // Both ways round, the light row is present. It is the field a headless
        // capture most needs and it used to be gated behind having a mouse.
        assert!(focus.contains("0.500") && pointer.contains("0.500"));
    }

    /// Every primitive is whole-pixel and inside the buffer, which is the
    /// invariant `crate::ui`'s painter assumes and never re-checks.
    #[test]
    fn the_panel_stays_inside_the_buffer() {
        let v = view();
        let r = DebugReadout {
            live: true,
            frame_ms: 8.0,
            biome: ("Tundra", 1.0),
            layer: ("Caverns", 1.0),
            cursor: Some(CursorReadout {
                from_pointer: false,
                cell: WorldCell::new(1, 1),
                material: 0,
                wall: 0,
                light: Some(1.0),
            }),
            ..DebugReadout::default()
        };
        for p in overlay(&r, Chrome::of(v)) {
            match p {
                UiPrim::Rect { x, y, w, h, .. } => {
                    assert!(x >= 0 && y >= 0 && x + w <= v.w && y + h <= v.h, "{p:?}");
                }
                UiPrim::Text { x, baseline, .. } => {
                    assert!(x >= 0 && baseline > 0 && baseline < v.h, "{p:?}");
                }
                _ => {}
            }
        }
    }

    fn texts(prims: &[UiPrim]) -> Vec<String> {
        prims
            .iter()
            .filter_map(|p| match p {
                UiPrim::Text { text, .. } => Some(text.to_string()),
                _ => None,
            })
            .collect()
    }
}
