//! Compiler output.
//!
//! Every module in this crate is emitted by `cargo run -p contentc` from the
//! records under `content/`. **Do not edit these files by hand** — the next
//! compile overwrites them, and `cargo run -p contentc -- --check` is the CI
//! gate that fails if they are stale.
//!
//! The arrow points one way: `content/` describes what things ARE, this crate
//! is the compiled form of that, and `godgame-core` reads it and never writes
//! it. That is what makes adding a block a content change rather than a code
//! change.
