//! How long a whole frame takes. The instrument `docs/PERF.md` did not have.
//!
//! # Why this exists
//!
//! Every other perf number in this repo is a criterion bench around ONE CPU
//! function. That is the right tool for "is `paint_cells` faster than it was",
//! and `PERF.md` §6 has always said so. It cannot answer the only question that
//! decides whether an optimisation was worth doing: **did the frame get
//! shorter.**
//!
//! The gap is not academic. It let a whole plan item be justified by a number
//! that turned out to mean nothing — see [`what_this_measured`] below, and
//! `PERF.md` §8.6.
//!
//! # It is not a gate, and it is `#[ignore]`d
//!
//! A timing threshold in `cargo xtask check` would fail on a busy laptop and
//! teach everyone to ignore it. This prints and asserts nothing:
//!
//! ```text
//! cargo test -p yugen-render --release --test frame_cost -- --ignored --nocapture
//! ```
//!
//! **`--release` is not optional.** A debug build spends its frame somewhere
//! else entirely and the ranking of costs inside it is not the game's.
//!
//! # The protocol, which is most of the value
//!
//! This instrument lies if used casually. Measured on one M3 Pro:
//!
//!   - **On a busy machine it drifts 4x.** The same binary measured 1.53 ms and
//!     6.59 ms in one afternoon. Any A/B that ran one side before the other and
//!     compared is worthless, and one did.
//!   - **On a quiet machine it is tight**: medians within ±15 µs across runs.
//!   - So: build both sides first (`--no-run`), then run the two BINARIES
//!     alternately in one loop, and compare medians. Never compare a number
//!     from one session to a number from another.
//!   - Its floor is about ±30 µs. It cannot resolve 98 µs — which is itself a
//!     finding, not a limitation, and the next section is why.
//!
//! # What this measured
//!
//! Three builds, interleaved on a quiet machine, four runs each, medians in µs:
//!
//! | build | median | vs base |
//! |---|---|---|
//! | base — the CPU blur, as shipped | 1313 | — |
//! | base with `LightGrid::blur` DELETED | 1314 | **+1** |
//! | the blur moved to two GPU passes | 1550 | **+237** |
//!
//! **Deleting 98 µs of main-world CPU work changed the frame by nothing.** The
//! main world is pipelined against the render app, so main-world CPU under the
//! render app's cost is free — and every "per-frame CPU render cost" in
//! `PERF.md` is main-world CPU. That total is a budget, not a critical path, and
//! `PERF.md` §8.6 now says so.
//!
//! The GPU version lost 237 µs, of which ~190 µs was the two extra `Camera2d`s
//! alone (measured by spawning the quads without the cameras). At 10 660 texels
//! the blur is nanoseconds of shading and the framework around it is everything.
//!
//! # What it cannot see
//!
//! GPU time. `RenderDiagnosticsPlugin` was tried and records timestamps only on
//! Vulkan and DX12 — on Metal it reports CPU spans, and the built-in ones do not
//! cover the render app's extract and queue phases, which is exactly where the
//! 190 µs went. So this measures the cost of a frame from the main world's
//! point of view, which is the number a player feels, and attributes nothing.

use std::time::Instant;

use bevy::prelude::*;
use yugen_render::scenes::Scene;

mod common;
use common::{gpu_is_available, headless_game};

/// Frames to let the world stream, settle and light before timing starts.
///
/// [`crate::frame_stability`]'s figure, for its reasons: a world still streaming
/// chunks in is not the world the game spends its time being.
const WARMUP: u32 = 240;

/// Frames timed. At ~1.3 ms each this is under two seconds, and it is enough
/// that the median is stable to ±15 µs on a quiet machine.
const SAMPLES: u32 = 1200;

/// Boot a headless game, run it, and report the distribution of frame times.
///
/// The percentiles are printed rather than just the mean because the two are
/// different questions: a change that moves the median is a change to what the
/// frame costs, and a change that only moves p99 is a change to how often
/// something occasional happens. `docs/PERF.md`'s window-shift work turned
/// entirely on that distinction.
#[test]
#[ignore = "an instrument, not a gate — see the module docs for the protocol"]
fn what_this_measured() {
    if !gpu_is_available() {
        println!("SKIPPED frame cost: no wgpu adapter on this machine");
        return;
    }

    let mut app = headless_game("Yūgen frame cost");
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    for _ in 0..WARMUP {
        app.update();
    }

    // `App::update` returns once the main world's frame is submitted, so this
    // includes the render app's extract and whatever backpressure the previous
    // frame left. That is the point: it is the only number here that contains
    // the parts a criterion bench cannot reach.
    let mut ticks = Vec::with_capacity(SAMPLES as usize);
    for _ in 0..SAMPLES {
        let at = Instant::now();
        app.update();
        ticks.push(at.elapsed().as_secs_f64() * 1e6);
    }

    let mean = ticks.iter().sum::<f64>() / ticks.len() as f64;
    ticks.sort_by(|a, b| a.partial_cmp(b).expect("a frame time is never NaN"));
    let at = |n: usize, d: usize| ticks[(ticks.len() * n / d).min(ticks.len() - 1)];
    println!(
        "frame over {SAMPLES}: mean {mean:.1}  median {:.1}  p10 {:.1}  p90 {:.1}  \
         p99 {:.1}  worst {:.1} µs",
        at(1, 2),
        at(1, 10),
        at(9, 10),
        at(99, 100),
        ticks[ticks.len() - 1],
    );
}
