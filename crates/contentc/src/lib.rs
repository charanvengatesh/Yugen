//! The content compiler: lexer -> parser -> schema -> emit.
//!
//! Compiles the records under `content/` into the Rust tables in
//! `godgame-data`. A schema is data — a record of dotted field name to
//! descriptor — and it is the single source of truth for three things that must
//! never disagree: validation of the source text, the shape of the emitted Rust,
//! and which flat tables get built.

pub mod driver;
pub mod emit;
pub mod error;
pub mod lexer;
pub mod lock;
pub mod names;
pub mod parser;
pub mod schema;
pub mod schemas;
pub mod value;

pub use error::{ContentError, Loc, Result};
