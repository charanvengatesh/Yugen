//! Making art and sound instead of typing it.
//!
//! # What is in here and what is not
//!
//! Every function in this module tree is pure and takes its randomness as a
//! seed. Nothing here imports `eframe`, opens a file or knows a panel exists —
//! [`crate::ui`] calls these and puts the results into the open document, and
//! [`crate::app`] is what eventually writes them.
//!
//! That boundary is not tidiness. It is what makes the determinism tests
//! possible: `generate(recipe) == generate(recipe)` is a claim about a function,
//! and asserting it in a test that had to open a window would be asserting it
//! about something else. It is the same argument `yugen-core` makes for holding
//! no Bevy, at a much smaller scale.
//!
//! # The invariant the operators share
//!
//! [`ops`] is a set of `Frame -> Frame` functions and **every one of them
//! preserves the row count and the row width**. A frame's shape is not a
//! property of the frame; it is `cellsW * grain` by `cellsH * grain`, declared
//! on the record, and a frame that disagrees with it is one `bake_frame`
//! answers with a panic at `PreStartup`. Keeping the shape inside the operators
//! is what lets a button call one without a check, and it is asserted against
//! every frame in `content/` rather than against a fixture.
//!
//! A resize is the deliberate exception, and it will be a different signature
//! for exactly that reason: it takes the target shape as an argument and reports
//! what the change cost.

pub mod anim;
pub mod ops;
pub mod palette;
pub mod resize;
pub mod sfx;
pub mod sprite;
