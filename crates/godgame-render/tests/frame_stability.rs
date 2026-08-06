//! An idle world must not thrash the screen.
//!
//! # The gap this fills
//!
//! `frame_capture.rs` proves a frame IS drawn: it has colour, contrast, and a
//! sky brighter than its underground. It says nothing whatever about whether the
//! frame is STABLE from one moment to the next, and that is not a theoretical
//! gap — it is the one the worst bug in this project walked straight through.
//!
//! `MobSystem::events()` used to hand back its whole fixed backing buffer, and
//! the host drained all of it, so about fourteen phantom `MobHurt` events
//! arrived every frame at (0, 0), each worth 0.08 trauma against a decay of 4/s.
//! **The screen shake was pinned at maximum from the first frame of the game.**
//! 896 tests and a purpose-built image gate were green throughout, because a
//! violently shaking camera still renders a varied, correctly-lit frame with a
//! bright sky at the top. A human opening the game found it.
//!
//! `frame_capture`'s `an_undisturbed_world_never_shakes_the_camera` now catches
//! that specific bug by reading `Screenshake::trauma()` directly. This is the
//! general form: it does not know what shake is, or bloom, or a light haze. It
//! photographs the world twice and asks whether the picture moved.
//!
//! # Why this can assert something so strict
//!
//! It could not have been written before `common::FRAME_DT`. Eight systems on the
//! render side read wall-clock time, so two runs of the same test used to differ
//! by 0.045%-0.132% of pixels for no reason at all — noise that would have
//! swallowed any threshold worth setting. With the frame delta pinned, two runs
//! of `frame_capture` are now BIT-IDENTICAL, measured over three runs.
//!
//! That reproducibility is what makes the numbers below meaningful. They are not
//! guesses with a safety factor bolted on; they are measured, and the margin over
//! them is stated.
//!
//! # What it deliberately does not do
//!
//! It does not demand two frames be identical. The world is alive: lava
//! shimmers, the sun moves, motes drift. Demanding stillness would be demanding
//! a broken game. What it demands is that the still parts stay still — that
//! nothing is TRANSLATING the frame or repainting large regions of it while
//! nobody is doing anything.

use std::path::PathBuf;

use bevy::prelude::*;

use godgame_render::scenes::Scene;

mod common;
use common::{Frame, artefact_path, gpu_is_available, headless_game};

/// Frames to let the world stream, settle and light before the first shutter.
///
/// Longer than `frame_capture`'s warmup because this test asks a question about
/// REST, and a world still streaming chunks in is not at rest. The camera ease
/// also has to converge: `player::follow_player` moves the focus 12% of the way
/// to the body every frame and asymptotically never arrives, so the residual has
/// to fall under the pixel snap or the frame steps sideways on its own.
const SETTLE_FRAMES: u32 = 240;

/// Frames between the two shutters.
///
/// Long enough that a slow drift has room to show — a camera creeping one pixel
/// every twenty frames is invisible across two consecutive ones — and short
/// enough that the day/night cycle has not meaningfully moved the sun. At
/// [`common::FRAME_DT`] this is a third of a second against a `DAY_LENGTH_S` of
/// 300.
const GAP_FRAMES: u32 = 20;

/// The share of the frame allowed to change across [`GAP_FRAMES`] idle frames.
///
/// # Where this number comes from
///
/// **Measured at 1.54%** on a healthy build, and mapped rather than trusted. On a
/// 16x10 grid over the frame, every cell outside one contiguous blob reads under
/// 1.2%; the blob — up to 51.8% of a cell — sits exactly on the sun's dithered
/// halo. `SKY_DITHER` spreads the sky's quantisation with an ordered Bayer
/// threshold, so as the sun creeps the halo's texels legitimately flip. The rest
/// is drifting snow and twinkling stars.
///
/// So the world is genuinely alive here and a tight bound would be a demand for a
/// broken game. The margin is about 5x, and it is affordable because the failure
/// this exists to catch is not subtle: a camera translated by
/// `SHAKE_MAX_OFFSET` = 14 world px moves essentially EVERY pixel, not 8% of
/// them. Between the honest floor and the bug there are four orders of magnitude.
///
/// Re-measure rather than nudge if a legitimate change moves it. The test prints
/// the number on every run for exactly that reason.
///
/// # This bound has been seen to fail
///
/// A gate never watched to go red is unproven. Injecting a SINGLE
/// `Screenshake::hurt()` — 0.42 trauma, decaying away over about a tenth of a
/// second, far milder than the bug that motivated this file — took the frame from
/// 1.54% to **42.5%** moved and the mean from 0.142 to 1.84. Both assertions
/// fired, with the numbers in the message. That is the gap this bound sits in.
const MAX_MOVED_SHARE: f64 = 0.08;

/// Mean absolute channel change allowed across [`GAP_FRAMES`] idle frames.
///
/// **Measured at 0.142/255.** The second bound exists because the first cannot
/// see intensity: a haze that repaints a third of the frame by one LSB and a haze
/// that repaints it by 100 are the same number of moved pixels. Margin ~7x, for
/// the reason [`MAX_MOVED_SHARE`] gives.
const MAX_MEAN_ABS: f64 = 1.0;

#[test]
fn an_idle_world_does_not_thrash_the_frame() {
    let Some((first, second)) = two_frames_apart() else {
        println!("SKIPPED frame stability: no wgpu adapter on this machine");
        return;
    };

    let moved = first.diff(&second);
    println!("frame-to-frame over {GAP_FRAMES} idle frames: {moved}");
    println!("  first  frame: {}", first.png.display());
    println!("  second frame: {}", second.png.display());

    assert!(
        moved.share() <= MAX_MOVED_SHARE,
        "the frame moved while nothing was happening: {moved}. The bound is \
         {:.1}% and a healthy build measures ~1.5%, nearly all of it the sun's \
         dithered halo. Something is translating or repainting the frame — a \
         camera shake with no cause, a flickering composite, a drifting view. \
         Compare {} against {}.",
        100.0 * MAX_MOVED_SHARE,
        first.png.display(),
        second.png.display()
    );

    assert!(
        moved.mean_abs <= MAX_MEAN_ABS,
        "the frame changed too hard while nothing was happening: {moved}. The \
         bound is {MAX_MEAN_ABS}/255 and a healthy build measures ~0.14. \
         Compare {} against {}.",
        first.png.display(),
        second.png.display()
    );
}

/// Boot, settle, and photograph the world twice [`GAP_FRAMES`] apart.
///
/// `None` means there is no GPU here, which is a SKIP and never a pass — the
/// same discipline `frame_capture` and `shader_matches_cpu` hold to.
fn two_frames_apart() -> Option<(Frame, Frame)> {
    if !gpu_is_available() {
        return None;
    }

    let mut app = headless_game("GodGame frame stability");
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    for _ in 0..SETTLE_FRAMES {
        app.update();
    }

    let first_png = png_path("first.png");
    let first = Frame::from_image(&common::capture_low_res(&mut app, &first_png), first_png);

    for _ in 0..GAP_FRAMES {
        app.update();
    }

    let second_png = png_path("second.png");
    let second = Frame::from_image(&common::capture_low_res(&mut app, &second_png), second_png);

    Some((first, second))
}

/// Where the two frames land, side by side, so a human can flick between them.
fn png_path(name: &str) -> PathBuf {
    artefact_path(env!("CARGO_TARGET_TMPDIR"), "frame-stability", name)
}
