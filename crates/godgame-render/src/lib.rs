//! Every draw pass.
//!
//! The whole game renders into a low-resolution offscreen target at
//! `VIEW_W x VIEW_H` and is then blitted to the window with a nearest-neighbour
//! pass. That is not a shortcut — the art, and especially the 8-13px UI type,
//! is authored to be upscaled with hard pixel edges.
