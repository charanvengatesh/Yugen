//! The content compiler: TOML -> schema -> emit.
//!
//! Compiles the records under `content/` into the Rust tables in
//! `yugen-data`. A schema is data — a record of dotted field name to
//! descriptor — and it is the single source of truth for three things that must
//! never disagree: validation of the source text, the shape of the emitted Rust,
//! and which flat tables get built.

pub mod driver;
pub mod emit;
pub mod error;
pub mod lock;
pub mod names;
pub mod schema;
pub mod schemas;
pub mod toml_in;
pub mod value;

pub use error::{ContentError, Loc, Result};
