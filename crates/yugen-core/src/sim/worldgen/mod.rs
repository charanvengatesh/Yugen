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
pub mod chunk;
pub mod containers;
pub mod features;
pub mod fields;
pub mod heightmap;
pub mod layers;
pub mod loot;
pub mod spline;
pub mod structs;

// The orchestrator is the face of the subsystem: `sim::worldgen::generate_chunk`
// is what a chunk store calls, and `chunk` is an implementation detail of where
// it happens to live.
pub use chunk::{
    ChunkGen, DECORATORS, SPAWN_COL, SpawnPoint, generate_chunk, generate_chunk_scaled,
    generate_chunk_terrain, material_at, spawn_ground_runs, spawn_point, walkable_spawn,
    world_noise,
};
