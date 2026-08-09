//! Sound: parameters in, samples out, and eventually a speaker.
//!
//! # Why this lives in the render crate
//!
//! It is not a draw pass, and the crate's own description says "every draw
//! pass". The honest reading is that `yugen-render` is the HOST half of the
//! game — everything that turns simulation state into something a person
//! perceives — and audio is that, in the one sense that matters here: it hangs
//! off the same event drains the particles do, in `crate::glue`, and it must
//! not be reachable from `yugen-core`.
//!
//! `yugen-core` staying free of Bevy is the boundary this repo protects. A
//! separate `yugen-audio` crate would protect it just as well and cost a crate
//! nobody has needed yet; if the audio grows a mixer, a music track or a
//! streaming decoder, that is when it earns one.
//!
//! # The split
//!
//! [`synth`] is pure: a sound code and a sample rate in, a `Vec<f32>` out, no
//! Bevy and no device. That is what lets the bank be tested for exact samples
//! by a suite that never opens an audio backend, which is the only way any of
//! this is testable in CI at all.
//!
//! Playback is not here yet. When it arrives it is a plugin beside `synth`,
//! reading the drained `PlayerEvent`s and `MobEvent`s that `crate::particles`
//! and `crate::effects` already read, and the split stays: the plugin decides
//! WHEN, this decides WHAT.

pub mod synth;

pub use synth::{Pcm, SAMPLE_RATE, render, render_all};
