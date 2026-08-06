//! What every test that photographs the game needs.
//!
//! # Why this module exists
//!
//! Rust compiles each file in `tests/` into its OWN crate, so nothing in one
//! integration test is reachable from another. Before this module there were two
//! capture tests and they had already started diverging by copy: `lit_scene.rs`
//! carried a byte-identical duplicate of `frame_capture.rs`'s `gpu_is_available`,
//! and a second, subtly different copy of the app boot — one that did NOT guard
//! `LogPlugin` and so contributed to exactly the process-wide collision
//! `frame_capture` had already been bitten by and written a comment about.
//!
//! Two copies is where that stops. `tests/common/mod.rs` is the conventional
//! answer: it is not a test binary itself (Cargo only treats top-level files in
//! `tests/` as targets), so it is compiled into whichever binaries `mod common;`
//! it.
//!
//! # What belongs here
//!
//! The mechanics of getting a picture out of a headless engine — booting it,
//! counting what it complains about, waiting for the readback, and unpacking the
//! bytes. Not thresholds, and not scenes. What a given frame is SUPPOSED to look
//! like is the individual test's business and belongs in the file that asserts
//! it.

// Each test binary that takes this module uses a different part of it —
// `lit_scene` never unpacks a `Frame`, `frame_capture` never asks for a pixel
// diff — and the compiler sees the unused remainder as dead in every one of
// them. Suppressing it here is the standard price of a shared test module, and
// the alternative (a feature flag per consumer) would be worse than the warning.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use bevy::app::PluginGroup;
use bevy::image::Image;
use bevy::log::tracing::Event;
use bevy::log::tracing_subscriber::Layer;
use bevy::log::tracing_subscriber::layer::Context;
use bevy::log::tracing_subscriber::registry::Registry;
use bevy::log::{BoxedLayer, Level, LogPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::tasks::block_on;
use bevy::time::TimeUpdateStrategy;
use bevy::window::{WindowPlugin, WindowResolution};

use godgame_render::GodGameRenderPlugin;
use godgame_render::lowres::LowResTarget;

/// The physical window the buffer is sized from.
///
/// A real number rather than a round one because `View::for_screen` derives the
/// low-res extent and the zoom from it, and a size no real display has would
/// exercise a zoom no player ever sees.
pub const WINDOW_PX: (u32, u32) = (1280, 800);

/// Frames to wait for the readback before calling it a failure.
///
/// The screenshot is asynchronous — the engine copies the texture at the end of
/// a frame and triggers the observer some frames later — so the wait is real.
/// Exceeding this is a FAILURE and not a skip: the capture path is what these
/// tests exist to exercise, and a silent skip when it stops working would be the
/// gate quietly switching itself off.
pub const CAPTURE_DEADLINE_FRAMES: u32 = 120;

/// The wall-clock delta every `update()` is told it took.
///
/// A 60 Hz frame against the game's 120 Hz fixed step, so each frame runs exactly
/// two sim steps and the accumulator carries no remainder. See
/// [`headless_game`] for why pinning this at all is the whole enabler for
/// comparing one frame against another.
pub const FRAME_DT: Duration = Duration::from_nanos(1_000_000_000 / 60);

// --- Counting what the engine complains about --------------------------------

/// ERROR lines the engine logged, process-wide.
///
/// A static because `LogPlugin::custom_layer` is a plain `fn` pointer — it
/// cannot close over anything — and because the subscriber outlives the `App`.
static ERROR_LINES: AtomicU64 = AtomicU64::new(0);

/// The same count, split by the module that emitted it, so a failure names the
/// culprit instead of just counting it.
static ERROR_SOURCES: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

/// Whether `LogPlugin` has already claimed this process's tracing subscriber.
static LOG_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Tallies every ERROR the tracing subscriber sees.
///
/// It sits UNDER the `EnvFilter` layer `LogPlugin` installs, so it counts
/// exactly the lines that were printed — not the ones the filter suppressed.
struct CountErrorLines;

impl Layer<Registry> for CountErrorLines {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, Registry>) {
        if *event.metadata().level() == Level::ERROR {
            ERROR_LINES.fetch_add(1, Ordering::Relaxed);
            *ERROR_SOURCES
                .lock()
                .expect("the error tally is never held across a panic")
                .entry(event.metadata().target())
                .or_default() += 1;
        }
    }
}

/// The running ERROR total. Take one before a capture and one after: the count
/// is process-wide, so only a DELTA is a statement about a particular frame.
pub fn error_count() -> u64 {
    ERROR_LINES.load(Ordering::Relaxed)
}

/// The per-module tally, copied out so the lock is not held by a caller.
pub fn error_sources() -> BTreeMap<&'static str, u64> {
    ERROR_SOURCES
        .lock()
        .expect("the error tally is never held across a panic")
        .clone()
}

// --- Bringing the game up headlessly -----------------------------------------

/// Whether this machine has a GPU wgpu will hand out an adapter for.
///
/// Asked BEFORE the app is built, because `RenderPlugin` panics when it cannot
/// find one and a panic is not a skip. This is the same request
/// `shader_matches_cpu.rs` makes, with no surface attached, so every capture
/// test in the crate agrees about what "has a GPU" means.
pub fn gpu_is_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_ok()
}

/// Where the captured image is parked between the observer and the test.
#[derive(Resource, Default)]
struct CapturedFrame(Option<Image>);

/// The observer half of the capture: keep the image the engine handed us.
fn keep_the_captured_frame(captured: On<ScreenshotCaptured>, mut slot: ResMut<CapturedFrame>) {
    slot.0 = Some(captured.image.clone());
}

/// The shipping plugin group, with the window system removed and nothing else.
///
/// `WinitPlugin` is the ONE plugin disabled. Everything else — assets, the task
/// pools, the render app, the sprite and text pipelines, and all of the game's
/// own plugins — is exactly what `main.rs` adds, so a capture test fails when the
/// game fails rather than when a hand-picked subset of it does.
///
/// The primary `Window` entity still exists; it is just data with no surface
/// behind it. It has to, because `lowres::setup` sizes the buffer from it, and
/// because sizing the buffer from the real window is behaviour under test.
///
/// # The `LogPlugin` guard is load-bearing
///
/// `LogPlugin` installs a PROCESS-WIDE tracing subscriber, and a second install
/// fails and logs an ERROR about it. With one app per process that never came up;
/// the moment a second test booted its own app, whichever booted last logged that
/// error and `frame_capture`'s `drawing_a_frame_logs_no_engine_errors` blamed the
/// engine for it.
///
/// So the plugin goes in exactly once per process. The subscriber it installs is
/// global anyway, so the counting layer keeps working for every app after it.
///
/// # Time is pinned, and that is the point
///
/// Eight systems on the render side read wall-clock `Res<Time>` — the shake decay
/// (which also resamples an RNG per call), the world clock that moves the sun,
/// the mote emitters, the toast timer, and three `elapsed_secs` clocks driving
/// the cell shimmer, the emitter flicker and the sky twinkle. Every one of them
/// therefore depended on how fast this machine happened to render, and so did the
/// fixed-step accumulator: `Time<Virtual>` is clamped to `MAX_STEPS_PER_FRAME`,
/// so a headless frame ran anywhere from 0 to 5 sim steps.
///
/// [`TimeUpdateStrategy::ManualDuration`] replaces all of that with a fixed
/// delta, in one line, and nothing in this repo used it before. Measured: two
/// runs of `frame_capture` on one tree differed by 0.045%–0.132% of pixels before
/// this, which is small enough to look like nothing and large enough to hide a
/// regression in.
///
/// [`FRAME_DT`] is a 60 Hz frame, so each `update()` runs exactly two 120 Hz sim
/// steps — a real frame rate rather than an arbitrary one, and an integer ratio
/// so the accumulator carries no remainder between frames.
pub fn headless_game(title: &str) -> App {
    let first = !LOG_INSTALLED.swap(true, Ordering::Relaxed);
    let log = LogPlugin {
        custom_layer: |_| -> Option<BoxedLayer> { Some(Box::new(CountErrorLines)) },
        ..default()
    };

    let mut app = App::new();
    let plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: title.into(),
                resolution: WindowResolution::new(WINDOW_PX.0, WINDOW_PX.1),
                ..default()
            }),
            ..default()
        })
        .disable::<bevy::winit::WinitPlugin>();

    if first {
        app.add_plugins(plugins.set(log));
    } else {
        app.add_plugins(plugins.disable::<LogPlugin>());
    }

    app.add_plugins(GodGameRenderPlugin)
        .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME_DT))
        .init_resource::<CapturedFrame>();
    app
}

/// Read the low-res buffer back, write it to `png`, and hand back the image.
///
/// This is the whole capture mechanic and nothing else: it does not decide when
/// the world is ready to be photographed, which is the caller's judgement and
/// differs per test.
///
/// [`LowResTarget::canvas`] is the buffer the entire game renders into —
/// cells, player, mobs, sky, weather, particles, and the light composite over all
/// of them — and the only thing that happens afterwards is a nearest-neighbour
/// upscale to a window that does not exist here. So it is both the right image
/// and the one that needs no display.
pub fn capture_low_res(app: &mut App, png: &Path) -> Image {
    let target = app
        .world()
        .get_resource::<LowResTarget>()
        .expect("LowResPlugin inserts LowResTarget in Startup")
        .clone();

    if let Some(dir) = png.parent() {
        std::fs::create_dir_all(dir).expect("the artefact directory is creatable");
    }
    // Owned, because the observer outlives this call. Clippy suggests passing the
    // borrow straight in; the borrow checker disagrees, and it is the one that
    // is right.
    let write_to = png.to_path_buf();
    app.world_mut()
        .spawn(Screenshot::image(target.canvas.clone()))
        .observe(save_to_disk(write_to))
        .observe(keep_the_captured_frame);

    let mut waited = 0;
    let image = loop {
        app.update();
        if let Some(image) = app.world_mut().resource_mut::<CapturedFrame>().0.take() {
            break image;
        }
        waited += 1;
        assert!(
            waited < CAPTURE_DEADLINE_FRAMES,
            "the low-res buffer never came back: {CAPTURE_DEADLINE_FRAMES} frames after the \
             Screenshot entity was spawned, no ScreenshotCaptured had been triggered"
        );
    };

    // `save_to_disk` reports its own failures to the log and returns nothing, so
    // this is the only place that can notice. Failure messages end with this
    // path; one pointing at a file that is not there would send the reader
    // looking for a picture nobody wrote.
    assert!(
        png.exists(),
        "the frame was captured but no PNG reached {} — save_to_disk logged the reason",
        png.display()
    );

    // The buffer's size is a pure function of the window, so a mismatch means the
    // capture read something OTHER than the game's canvas — which would make
    // every assertion about it a statement about the wrong image.
    assert_eq!(
        (image.width(), image.height()),
        (target.view.w as u32, target.view.h as u32),
        "the captured image is not the low-res buffer: LowResTarget says {}x{} at zoom {}",
        target.view.w,
        target.view.h,
        target.view.zoom
    );
    image
}

/// An artefact path under `<workspace>/target/tmp/<dir>/<name>`.
///
/// `CARGO_TARGET_TMPDIR` is the one absolute path Cargo hands an integration
/// test, it is inside `target/`, and `target/` is the first line of `.gitignore`.
/// Files are overwritten every run on purpose — the interesting frame is always
/// the one from the failure you are looking at.
pub fn artefact_path(target_tmpdir: &str, dir: &str, name: &str) -> PathBuf {
    PathBuf::from(target_tmpdir).join(dir).join(name)
}

// --- The frame ---------------------------------------------------------------

/// One captured frame, unpacked into straight RGBA8 and measured.
///
/// The histogram is built once, here, because several assertions want it and
/// booting the game once per question would multiply the slowest test in the
/// crate.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8, top row first, four bytes per pixel.
    pub rgba: Vec<u8>,
    /// Per-pixel luminance, in the same order.
    pub luma: Vec<f64>,
    /// How many pixels each distinct RGBA colour claims.
    pub colours: HashMap<[u8; 4], u32>,
    /// Where the PNG of this frame was written.
    pub png: PathBuf,
}

impl Frame {
    /// Unpack a captured image, normalising the channel order.
    ///
    /// The low-res canvas is `Bgra8UnormSrgb` — the format `lowres::new_canvas`
    /// asks for — and the screenshot readback hands the texture's own format
    /// straight back. `Rgba8UnormSrgb` is accepted too so that changing the
    /// canvas format does not silently invert this module's idea of red and
    /// blue; anything else fails loudly rather than being measured as noise.
    pub fn from_image(image: &Image, png: PathBuf) -> Frame {
        let format = image.texture_descriptor.format;
        let swap = match format {
            TextureFormat::Bgra8UnormSrgb | TextureFormat::Bgra8Unorm => true,
            TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => false,
            other => panic!(
                "the low-res buffer came back as {other:?}; this module only knows how to read \
                 8-bit BGRA and RGBA, and guessing the channel order would make every \
                 assertion against it meaningless"
            ),
        };
        let data = image
            .data
            .as_ref()
            .expect("a captured screenshot always carries its bytes");
        let width = image.width();
        let height = image.height();
        let pixels = (width * height) as usize;
        assert_eq!(
            data.len(),
            pixels * 4,
            "the readback is not tightly packed: {} bytes for {width}x{height}",
            data.len()
        );

        let mut rgba = Vec::with_capacity(pixels * 4);
        let mut luma = Vec::with_capacity(pixels);
        let mut colours: HashMap<[u8; 4], u32> = HashMap::new();
        for chunk in data.chunks_exact(4) {
            let px = if swap {
                [chunk[2], chunk[1], chunk[0], chunk[3]]
            } else {
                [chunk[0], chunk[1], chunk[2], chunk[3]]
            };
            rgba.extend_from_slice(&px);
            luma.push(luminance(px));
            *colours.entry(px).or_default() += 1;
        }

        Frame {
            width,
            height,
            rgba,
            luma,
            colours,
            png,
        }
    }

    /// How far this frame is from another of the same size.
    ///
    /// Alpha is ignored: the low-res canvas is opaque everywhere, so an alpha
    /// term would only ever contribute zero and would make the mean look smaller
    /// than the colours actually moved.
    pub fn diff(&self, other: &Frame) -> FrameDiff {
        assert_eq!(
            (self.width, self.height),
            (other.width, other.height),
            "two frames of different sizes cannot be compared"
        );

        let mut differing = 0usize;
        let mut worst = 0u8;
        let mut total = 0u64;
        for (a, b) in self.rgba.chunks_exact(4).zip(other.rgba.chunks_exact(4)) {
            let mut moved = false;
            for k in 0..3 {
                let d = a[k].abs_diff(b[k]);
                if d != 0 {
                    moved = true;
                    worst = worst.max(d);
                    total += u64::from(d);
                }
            }
            if moved {
                differing += 1;
            }
        }

        let pixels = (self.width * self.height) as usize;
        FrameDiff {
            pixels,
            differing,
            worst,
            mean_abs: total as f64 / (pixels * 3) as f64,
        }
    }

    /// The most common colour and how much of the frame it covers.
    pub fn dominant(&self) -> ([u8; 4], f64) {
        let (&colour, &count) = self
            .colours
            .iter()
            .max_by_key(|&(_, &n)| n)
            .expect("a frame has at least one pixel");
        (
            colour,
            f64::from(count) / f64::from(self.width * self.height),
        )
    }

    /// Mean luminance over a horizontal band, given as a row range.
    pub fn band_luma(&self, first_row: u32, rows: u32) -> f64 {
        let start = (first_row * self.width) as usize;
        let end = start + (rows * self.width) as usize;
        let band = &self.luma[start..end];
        band.iter().sum::<f64>() / band.len() as f64
    }

    /// Mean and standard deviation of luminance over the whole frame.
    pub fn luma_spread(&self) -> (f64, f64) {
        let n = self.luma.len() as f64;
        let mean = self.luma.iter().sum::<f64>() / n;
        let variance = self
            .luma
            .iter()
            .map(|l| (l - mean) * (l - mean))
            .sum::<f64>()
            / n;
        (mean, variance.sqrt())
    }

    /// The colour at a pixel, for a failure message that wants an example.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * self.width + x) * 4) as usize;
        [
            self.rgba[at],
            self.rgba[at + 1],
            self.rgba[at + 2],
            self.rgba[at + 3],
        ]
    }

    /// The line every failure ends with: go and look at the picture.
    pub fn artefact(&self) -> String {
        format!("the captured frame is at {}", self.png.display())
    }
}

/// How far two frames are apart.
///
/// Three numbers rather than one, because they fail differently and a single
/// scalar hides which. A whole-frame translation — a shaking camera — moves a
/// LOT of pixels by a little. A single flickering light moves a few pixels by a
/// lot. `mean_abs` alone would report those two as similar.
pub struct FrameDiff {
    /// Pixels in each frame, so `differing` can be read as a share.
    pub pixels: usize,
    /// How many pixels changed in any colour channel.
    pub differing: usize,
    /// The largest single-channel change, 0-255.
    pub worst: u8,
    /// Mean absolute channel change over the whole frame, 0-255.
    pub mean_abs: f64,
}

impl FrameDiff {
    /// The share of the frame that moved, 0.0 to 1.0.
    pub fn share(&self) -> f64 {
        self.differing as f64 / self.pixels as f64
    }
}

impl std::fmt::Display for FrameDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} of {} pixels moved ({:.4}%), worst channel {}/255, mean {:.5}/255",
            self.differing,
            self.pixels,
            100.0 * self.share(),
            self.worst,
            self.mean_abs
        )
    }
}

/// Perceptual luminance of an sRGB pixel, in the same 0-255 units as the bytes.
///
/// Deliberately the cheap Rec. 601 weighting on the ENCODED bytes rather than a
/// linearised luminance. Every comparison here is between two regions of the
/// same frame through the same transfer function, so linearising would move both
/// sides and change no outcome — and the numbers a failure prints stay in the
/// units of the pixels you would read off the PNG.
pub fn luminance(px: [u8; 4]) -> f64 {
    0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2])
}
