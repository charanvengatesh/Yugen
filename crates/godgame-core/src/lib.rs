//! The engine — what things DO.
//!
//! Deliberately free of Bevy, wgpu and windowing. That boundary is load-bearing:
//! it keeps the simulation headless-testable, keeps `cargo test` fast, and means
//! the worldgen purity suite and the benches never link a renderer.
