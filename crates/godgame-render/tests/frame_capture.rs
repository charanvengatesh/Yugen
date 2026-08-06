//! The frame the game draws must look like a game, and a machine must say so.
//!
//! # What this is
//!
//! Every other test in this tree asserts about a NUMBER on its way to the
//! screen: `ts_cells_parity.rs` freezes the rasteriser's output, and
//! `shader_matches_cpu.rs` proves the shader agrees with it. None of them
//! assert anything about the IMAGE. The whole stack can be arithmetically
//! perfect and still composite a black rectangle over it, and until this file
//! existed the only thing that noticed was a human opening a PNG.
//!
//! So this is the last link: it boots the real [`GodGameRenderPlugin`] group
//! headlessly, lets the world exist for a while, captures the low-resolution
//! buffer the entire game renders into, writes it to disk where a human can
//! look at it, and then asserts the handful of things that are true of every
//! frame this game has ever drawn and false of a broken one.
//!
//! # What is captured, and why that is the right buffer
//!
//! [`LowResTarget::canvas`] is the offscreen `view.w x view.h` image that
//! [`lowres`](godgame_render::lowres) points the world camera at. Everything —
//! cells, player, mobs, sky, weather, particles, and the light composite over
//! all of them — is drawn into it, and the only thing that happens afterwards
//! is a nearest-neighbour upscale to the window. Capturing it means:
//!
//!   - no window and no swapchain, so the test runs with `WinitPlugin` disabled
//!     and does not need a display;
//!   - a small deterministic size, because [`View::for_screen`] is a pure
//!     function of the physical window size, which is fixed below;
//!   - the actual shipping image, not a re-render of it through a private path
//!     that could pass while the real one was broken.
//!
//! `Screenshot::image` is what reads it back: for an image render target the
//! engine swaps in its own attachment for that frame, so the camera draws
//! straight into the texture that is then copied to the CPU. It is the same
//! mechanism `--screenshot` uses in the binary, pointed at the buffer instead
//! of at the window.
//!
//! # What this covers and what it does not
//!
//! Covered: the whole plugin group builds, its `Startup` and `PostStartup`
//! systems run, the world streams in, the cell pass uploads and specialises,
//! and the composite produces a varied, correctly-oriented image. That is the
//! entire path from worldgen to pixels.
//!
//! NOT covered, and deliberately: the upscale blit to the window (there is no
//! window), input, anything that needs hundreds of frames to populate — the
//! creatures spawn on a 0.15 s timer, so a run this short will not show them —
//! and the ARTISTIC question of whether the frame is any good. The assertions
//! below are floors, not a golden image: they are the properties whose loss is
//! a bug in every case, and nothing narrower, because a strict pixel
//! comparison against a stored frame would fail on every legitimate tuning
//! change and would be deleted within a month.
//!
//! # Skipping
//!
//! If this machine has no GPU wgpu will talk to, every test here SKIPS and says
//! so on stdout. It does not pass. This is the same discipline
//! `shader_matches_cpu.rs` holds to and for the same reason: a green tick from
//! a machine that never rendered a frame is worse than a red one, because a
//! green tick is precisely the claim this file exists to make.

use std::path::PathBuf;
use std::sync::LazyLock;

use bevy::app::{RunFixedMainLoop, RunFixedMainLoopSystems};
use bevy::prelude::*;

use godgame_render::effects::Screenshake;
use godgame_render::input::PlayerIntent;
use godgame_render::player::PlayerBody;
use godgame_render::player_art::PlayerFigure;
use godgame_render::scenes::Scene;

mod common;
use common::{Frame, artefact_path, error_count, error_sources, gpu_is_available, headless_game};

/// Frames to render before capturing.
///
/// The binary's `SCREENSHOT_WARMUP_FRAMES` is 30, which is what the terrain
/// needs: the cell upload runs in `Update` after `Startup` inserted the world,
/// and the pipeline specialises lazily. This is three times that because the
/// atmosphere passes want a world that has been ALIVE, not merely uploaded —
/// the clock has to advance, the light grid has to solve and blur, and the
/// weather and ambience pools have to emit their first motes. It is still far
/// short of the several hundred a creature census needs, which is why the
/// assertions below say nothing about mobs.
///
/// It is also short enough that the sky does not move under the test. A headless
/// frame takes a few milliseconds and `Time<Virtual>` is clamped to
/// `MAX_STEPS_PER_FRAME`, so even on a machine an order of magnitude slower than
/// this one the warmup burns a few seconds of a
/// [`DAY_LENGTH_S`](godgame_render::daynight::DAY_LENGTH_S) of 300 — the clock
/// is still mid-morning when the shutter opens, which is what
/// [`the_sky_is_brighter_than_the_underground_at_the_morning_default`] assumes.
const WARMUP_FRAMES: u32 = 90;

/// Fraction of the frame the single most common colour may occupy.
///
/// This is THE black-frame gate. A frame that is one colour puts 100% of its
/// pixels in one histogram bucket; the bug this file was written for put 100% in
/// the black bucket. The bound is loose on purpose: a legitimate frame can be
/// mostly sky, or mostly stone, and 85% of one colour is a picture that has lost
/// its detail rather than a picture with a large flat region in it.
const MAX_DOMINANT_SHARE: f64 = 0.85;

/// Distinct RGBA colours a real frame must show.
///
/// A floor, and a low one. The sky gradient alone is dozens of bands, the shade
/// table is 32 steps per material, and the light composite multiplies both. This
/// is the number below which the frame has stopped being a rendering of
/// anything — a flat fill, a two-tone fill, a cleared buffer with a HUD on it.
const MIN_DISTINCT_COLOURS: usize = 64;

/// Standard deviation of luminance, in 0-255 units, a real frame must show.
///
/// Separate from the colour count because they fail differently: a frame can
/// carry many nearly-identical colours (a gradient washed to black by a broken
/// composite) and still be, to the eye, a flat surface. This asserts the frame
/// has CONTRAST, not merely variety.
const MIN_LUMA_STDDEV: f64 = 8.0;

/// How much of the frame's height is sampled at the top and at the bottom for
/// the sky/underground comparison.
///
/// An eighth of a 400 px buffer is 50 px, or 10 cells. The camera sits on the
/// player, who spawns on the surface, so the top band is 40 cells above the
/// ground line and the bottom band is 40 below it — both comfortably clear of
/// the ridge, and neither so thin that one bright mote could carry it.
const BAND_FRACTION: u32 = 8;

/// How much brighter the sky band must be than the underground band, in 0-255
/// luminance units.
///
/// At the clock's [`DEFAULT_START`](godgame_render::daynight::DEFAULT_START) of
/// 0.34 the world is mid-morning: the sky is lit and the underground is in
/// shadow with only the ambient floor and whatever emitters are in frame. The
/// gap is enormous in a working build, so this bound is not a measurement of it
/// — it is the smallest gap that still means the light composite is the right
/// way up. An inverted or missing skylight flood collapses it to zero or below.
const MIN_SKY_ADVANTAGE: f64 = 20.0;

// --- Capturing the frame this file asserts about ----------------------------

/// A captured frame plus the errors the engine logged while drawing IT.
///
/// The error tally in [`common`] is process-wide, and any other test that boots
/// an app contributes to it — which is exactly what happened the moment a second
/// one did. So the count carried here is a DELTA taken around this capture; a
/// count that is not scoped to the capture it is reported against is not a
/// measurement of that capture.
///
/// It derefs to the [`Frame`], so every assertion below reads as though the two
/// were one thing — which is how they are used.
struct Capture {
    frame: Frame,
    errors: u64,
}

impl std::ops::Deref for Capture {
    type Target = Frame;

    fn deref(&self) -> &Frame {
        &self.frame
    }
}

/// Boot the game, draw [`WARMUP_FRAMES`] frames, and read the low-res buffer
/// back.
///
/// `finish` and `cleanup` are called by hand because this drives the app with
/// `update` instead of `run`; without them the render app never initialises its
/// device and the first `update` panics.
fn capture_one_frame() -> Option<Capture> {
    if !gpu_is_available() {
        return None;
    }

    let before_errors = error_count();
    let mut app = headless_game("GodGame frame capture");
    app.finish();
    app.cleanup();

    // Leave the title card before drawing anything worth asserting on.
    //
    // `Scene` defaults to `Menu`, and the menu is a dimmed world under a title
    // and a prompt. Every assertion below still PASSED against it — the world is
    // visible through the dim, so it has colour, contrast and a bright sky — but
    // it was measuring the menu, and the assertions were quietly worth much less
    // than they looked. Sky luminance fell from 134 to 63 and the distinct
    // colour count from 8833 to 2229 the moment the menu appeared, and nothing
    // failed. Loose floors are the right call for a gate that must survive
    // legitimate tuning (see `png_path`'s notes on why this is not a golden
    // image), but they only mean anything if the frame is the one that matters.
    //
    // Set directly rather than by synthesising a key press:
    // `confirm_advances_the_scene` is the game's business and is tested where it
    // lives, and a capture that depended on it would fail for two unrelated
    // reasons.
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    for _ in 0..WARMUP_FRAMES {
        app.update();
    }

    assert_eq!(
        app.world().resource::<State<Scene>>().get(),
        &Scene::Playing,
        "the capture must be of gameplay, not of the menu over it"
    );

    let png = png_path();
    let image = common::capture_low_res(&mut app, &png);
    Some(Capture {
        frame: Frame::from_image(&image, png),
        errors: error_count() - before_errors,
    })
}

/// Where the artefact lands: `<workspace>/target/tmp/frame-capture/lowres.png`.
///
/// The file is overwritten every run on purpose — the interesting frame is
/// always the one from the failure you are looking at. This is not a golden
/// image and nothing compares against a stored copy of it; a strict pixel
/// comparison would fail on every legitimate tuning change and would be deleted
/// within a month.
fn png_path() -> PathBuf {
    artefact_path(env!("CARGO_TARGET_TMPDIR"), "frame-capture", "lowres.png")
}

/// The one captured frame, shared by every test in this file.
///
/// Booting the whole engine is seconds of work and the tests in a file run in
/// parallel, so the app is brought up exactly once and each test asks a
/// different question of the same image. `None` means there is no GPU here.
static FRAME: LazyLock<Option<Capture>> = LazyLock::new(capture_one_frame);

/// The frame, or `None` and a printed note. See the module docs on skipping.
fn frame_or_skip(what: &str) -> Option<&'static Capture> {
    match FRAME.as_ref() {
        Some(frame) => Some(frame),
        None => {
            println!("SKIPPED {what}: no wgpu adapter on this machine");
            None
        }
    }
}

// --- The gate ----------------------------------------------------------------

/// THE test — the one the black frame would have failed.
///
/// A frame that is a single colour is not a rendering. It is what a missing
/// composite, a camera pointed at nothing, an unspecialised pipeline or a light
/// pass multiplying by zero all produce, and it is what shipped: 100% of the
/// buffer in one histogram bucket, with 660 green unit tests either side of it.
#[test]
fn the_rendered_frame_is_not_a_single_flat_colour() {
    let Some(frame) = frame_or_skip("frame capture") else {
        return;
    };

    let (colour, share) = frame.dominant();
    println!(
        "{}x{} frame: {} distinct colours, most common {colour:?} at {:.2}%",
        frame.width,
        frame.height,
        frame.colours.len(),
        share * 100.0
    );

    assert!(
        frame.colours.len() > 1,
        "the frame is a SINGLE FLAT COLOUR, {colour:?} across all {} pixels — nothing was \
         drawn, or something drew over everything. {}",
        frame.width * frame.height,
        frame.artefact()
    );
    assert!(
        share <= MAX_DOMINANT_SHARE,
        "{:.2}% of the frame is one colour, {colour:?} (limit {:.0}%), over {} distinct \
         colours in {}x{}. {}",
        share * 100.0,
        MAX_DOMINANT_SHARE * 100.0,
        frame.colours.len(),
        frame.width,
        frame.height,
        frame.artefact()
    );
}

/// The frame carries detail, not merely more than one colour.
///
/// Two measurements because they fail differently. The colour count catches a
/// frame that has collapsed to a handful of fills; the luminance spread catches
/// one that kept its variety but lost its contrast — a gradient crushed towards
/// black, which reads as flat to the eye while the histogram still looks busy.
#[test]
fn the_rendered_frame_has_the_variety_and_contrast_of_a_drawn_image() {
    let Some(frame) = frame_or_skip("frame variety") else {
        return;
    };

    let distinct = frame.colours.len();
    let (mean, stddev) = frame.luma_spread();
    println!("luminance: mean {mean:.1}, stddev {stddev:.1} over {distinct} distinct colours");

    assert!(
        distinct >= MIN_DISTINCT_COLOURS,
        "only {distinct} distinct colours in the frame (floor {MIN_DISTINCT_COLOURS}); mean \
         luminance {mean:.1}, stddev {stddev:.1}. {}",
        frame.artefact()
    );
    assert!(
        stddev >= MIN_LUMA_STDDEV,
        "the frame has almost no contrast: luminance stddev {stddev:.1} (floor \
         {MIN_LUMA_STDDEV}) around a mean of {mean:.1}, over {distinct} distinct colours. {}",
        frame.artefact()
    );
}

/// The picture is the right way up: sky above, ground below.
///
/// The clock starts at `DEFAULT_START` = 0.34, mid-morning, and `LightPlugin`
/// composites last over everything the other passes drew. So the top of the
/// frame is lit sky and the bottom is underground in shadow, and the gap between
/// them is the single cheapest statement that the light pass ran, ran the right
/// way round, and was not applied to a frame that had nothing in it.
///
/// It is also the assertion that a uniformly bright frame cannot pass: a wash of
/// any single colour puts both bands at the same luminance, so this fails
/// alongside the flat-colour test rather than being redundant with it.
#[test]
fn the_sky_is_brighter_than_the_underground_at_the_morning_default() {
    let Some(frame) = frame_or_skip("sky/underground contrast") else {
        return;
    };

    let band = (frame.height / BAND_FRACTION).max(1);
    let sky = frame.band_luma(0, band);
    let underground = frame.band_luma(frame.height - band, band);
    println!(
        "sky (top {band} rows) mean luminance {sky:.1}, underground (bottom {band} rows) \
         {underground:.1}, gap {:.1}",
        sky - underground
    );

    assert!(
        sky - underground >= MIN_SKY_ADVANTAGE,
        "the sky is not brighter than the underground: top {band} rows mean luminance \
         {sky:.1} (sample {:?}), bottom {band} rows {underground:.1} (sample {:?}), gap \
         {:.1} against a required {MIN_SKY_ADVANTAGE:.1}. {}",
        frame.pixel(frame.width / 2, band / 2),
        frame.pixel(frame.width / 2, frame.height - band / 2),
        sky - underground,
        frame.artefact()
    );
}

/// Drawing a frame must not make the engine complain.
///
/// An ERROR line is the engine saying it did something it could not do, and
/// there is no such thing as a routine one. This counts every ERROR the tracing
/// subscriber saw across the whole run — the same lines that scroll past when
/// the binary is run by hand — and names the module that emitted them, because
/// a count alone is a number nobody can act on.
///
/// It rides on the same single boot as the image assertions: [`FRAME`] is what
/// produces the log lines, so this test asks for it first even though it does
/// not look at the pixels.
///
/// # What it does not catch, honestly
///
/// The `bevy_render::slab_allocator` use-after-free flood that a windowed run
/// emits does NOT reproduce here — this capture logs zero ERROR lines. That is
/// not the bug being absent; it is the bug living on a path this test does not
/// walk. There is no window, so the canvas sprite is never drawn to a swapchain
/// and the blit's instance data never reaches the allocator that complains. So
/// treat a green tick from this test as "the offscreen pass is quiet", not as
/// "the game is quiet" — closing the rest of that gap needs a windowed run,
/// which is exactly what this file was written to avoid needing.
#[test]
fn drawing_a_frame_logs_no_engine_errors() {
    let Some(frame) = frame_or_skip("engine error log") else {
        return;
    };

    let total = frame.errors;
    let by_source: Vec<String> = error_sources()
        .iter()
        .map(|(target, n)| format!("{target}: {n}"))
        .collect();
    println!("engine ERROR lines during the capture: {total}");

    assert_eq!(
        total,
        0,
        "the engine logged {total} ERROR lines while drawing {WARMUP_FRAMES} frames:\n  {}\n{}",
        by_source.join("\n  "),
        frame.artefact()
    );
}

/// A body that is actually running reaches a non-zero lean.
///
/// This closes a hole that was flagged when the sheared quad landed: the lean is
/// a shear applied to the drawn figure, `crate::shear` proves the geometry, and
/// `crate::player_art` proves the maths — but every OTHER test in this file
/// captures a body standing still, where `lean` is exactly zero. So the whole
/// shear path was verified in pieces and never once as a running game.
///
/// What this asserts is the join: drive the intent the keyboard drives, step the
/// real schedule, and confirm the figure the renderer builds has actually leant.
/// It deliberately does NOT assert on pixels — a lean of 0.16 moves a 2x3
/// character's top edge by under two pixels, which is real but far too small to
/// separate from the terrain behind it by counting colours.
///
/// It boots its own app rather than sharing the cached frame, because it needs to
/// drive input and the other four tests want a pristine standing capture.
#[test]
fn a_running_body_reaches_a_non_zero_lean() {
    if !gpu_is_available() {
        println!("SKIPPED lean check: no wgpu adapter on this machine");
        return;
    }

    let mut app = headless_game("GodGame frame capture");
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    // Settle first: the body spawns in the air and has to land before running.
    for _ in 0..WARMUP_FRAMES {
        app.update();
    }

    // Hold right, in the ONE slot where it survives.
    //
    // Writing `PlayerIntent` from the test body does not work, and the reason is
    // worth recording: `crate::input` samples the keyboard in `PreUpdate`, so
    // anything written before `app.update()` is overwritten by an unpressed
    // keyboard before the fixed step ever reads it. The first version of this
    // test did exactly that and reported a peak lean of 0.0000 — which looked
    // like the shear never firing, and was really the intent never arriving.
    //
    // `BeforeFixedMainLoop` is after the keyboard is read and before the body is
    // stepped, which is precisely where the binary's `--drive` injects for the
    // same reason.
    app.add_systems(
        RunFixedMainLoop,
        (|mut intent: ResMut<PlayerIntent>| intent.dir_x = 1.0)
            .in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
    );

    let mut peak: f32 = 0.0;
    for _ in 0..WARMUP_FRAMES {
        app.update();
        let body = app.world().resource::<PlayerBody>();
        peak = peak.max(PlayerFigure::of(body).lean.abs());
    }

    println!("peak |lean| while running: {peak:.4}");
    assert!(
        peak > 0.0,
        "a body held to the right for {WARMUP_FRAMES} steps never leant — the \
         shear path is never exercised by a real run, however well its pieces test"
    );
}

/// A world nobody is touching does not shake the camera.
///
/// This is a regression test for a bug that 896 other tests did not see, and the
/// shape of it is worth keeping in mind.
///
/// `MobSystem::events()` returns a FIXED backing buffer and documents that only
/// the first `event_count` entries are valid; the rest are the filler it was
/// constructed with, whose `kind` happens to be `MobHurt`. The host drained the
/// whole buffer. So roughly fourteen phantom creature-hits arrived every frame,
/// at (0, 0), for creatures that did not exist — each adding 0.08 trauma against
/// a decay of 4.0/s, which pinned the shake at maximum from the instant the game
/// opened and never let go.
///
/// Nothing else could have caught it. The unit tests cover `Screenshake`'s decay
/// curve and `MobSystem`'s event contract, and both were correct in isolation —
/// the defect lived entirely in the join. The frame assertions above could not
/// see it either: a shaking camera still renders a varied, correctly-lit frame,
/// so every colour and luminance floor stayed green while the screen was
/// unplayable.
///
/// What this asserts is therefore deliberately about REST rather than about
/// pixels: with no input, no damage and nothing dying, trauma must stay at zero.
/// A single real event would legitimately break that, which is why the body is
/// left alone rather than driven.
#[test]
fn an_undisturbed_world_never_shakes_the_camera() {
    if !gpu_is_available() {
        println!("SKIPPED shake check: no wgpu adapter on this machine");
        return;
    }

    let mut app = headless_game("GodGame frame capture");
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    let mut peak = 0.0f32;
    let mut shaking_frames = 0;
    for _ in 0..WARMUP_FRAMES {
        app.update();
        let trauma = app.world().resource::<Screenshake>().trauma();
        peak = peak.max(trauma);
        if trauma > 0.0 {
            shaking_frames += 1;
        }
    }

    let body = app.world().resource::<PlayerBody>();
    assert!(
        !body.dead() && body.health > 0.0,
        "the body died during an idle run, so a shake would be legitimate — \
         health {}, which makes this test's premise wrong rather than the shake",
        body.health
    );

    println!("peak trauma over {WARMUP_FRAMES} idle frames: {peak:.4}");
    assert_eq!(
        shaking_frames, 0,
        "the camera shook on {shaking_frames} of {WARMUP_FRAMES} idle frames \
         (peak trauma {peak:.4}) with a live, undamaged body and nothing \
         attacking it — something is raising events that did not happen"
    );
}
