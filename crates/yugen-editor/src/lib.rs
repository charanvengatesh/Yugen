//! Authoring tools for `content/`.
//!
//! The editor this crate exists for is a mouse-driven pixel and SFX tool. Its
//! binary is `src/bin/yugen-editor.rs`; everything the binary stands on is here,
//! because the library is where the parts that can be wrong SILENTLY live.
//!
//! `content/FORMAT.md` §9 is the contract, and its first clause is the one that
//! decides the design: **never round-trip a file through a TOML serialiser.** A
//! serialiser is correct about TOML and wrong about this format. §5 makes `#`
//! and trailing whitespace DATA inside `'''`, and §6 makes the comments the most
//! valuable thing in the file — a diff of `content/blocks/terrain.toml` is
//! mostly prose explaining why each number is what it is. Re-emitting the file
//! normalises the first two and deletes the third, and it does it silently,
//! producing a diff that looks like "the whole file changed" and cannot be
//! reviewed.
//!
//! So the rule is textual: find the range, replace exactly that, leave every
//! other byte alone. It is applied at two scales, and both are needed:
//!
//! - [`splice`] replaces a whole RECORD, protecting the file around it. That is
//!   the granularity for adding a record or rewriting one wholesale.
//! - [`field`] replaces one FIELD inside a record, protecting the record around
//!   it. That is the granularity a pixel editor needs, because a real record has
//!   prose between its sequences and a redraw must not take it.
//!
//! The rest is reading, which is safe: [`sprite`] parses a record into something
//! drawable and [`raster`] turns a frame's digits into RGBA. Neither ever writes.
//!
//! # The three layers, and why they do not mix
//!
//! - [`app`] and [`doc`] own what is open and the four functions that touch the
//!   disk. Every content write in this crate goes through `Editor::save`, which
//!   is what makes the stale check and the temp-and-rename unforgettable.
//! - [`ui`] draws. Panels mutate the open document and set the status; none of
//!   them opens a file.
//!
//! The boundary that matters is the one the tests depend on: nothing outside
//! [`ui`] imports `eframe`. A generator is a function from frames to frames and
//! from parameters to parameters, so its determinism can be asserted in a test
//! that never opens a window.

pub mod app;
pub mod doc;
pub mod field;
pub mod procgen;
pub mod raster;
pub mod rng;
pub mod sound;
pub mod splice;
pub mod sprite;
pub mod synth;
pub mod template;
pub mod ui;
