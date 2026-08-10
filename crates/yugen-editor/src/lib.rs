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

pub mod app;
pub mod field;
pub mod raster;
pub mod sound;
pub mod splice;
pub mod sprite;
pub mod synth;
