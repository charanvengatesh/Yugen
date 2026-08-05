//! Worldgen: the pure function from `(chunk_x, chunk_y, seed)` to a chunk of
//! cells.
//!
//! The whole subsystem is order-independent by contract. A chunk never reads a
//! neighbour; anything that spans a chunk boundary is recomputed identically by
//! every chunk it touches and each paints only its own share. That is what makes
//! the world infinite, the generation parallelisable, and the purity suite in
//! `tests/` meaningful.
//!
//! (The TypeScript build called this directory `gen/`. `gen` is a reserved
//! keyword in Rust 2024.)

pub mod caves;
pub mod fields;
pub mod heightmap;
pub mod layers;
pub mod spline;
