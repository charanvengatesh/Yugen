//! Generate a block of chunks and write them out as a PNG, so a human can LOOK
//! at the terrain.
//!
//! The purity suite proves the generator is a pure function of its coordinates.
//! It cannot tell you the world is 90% ocean, that the beaches are ten cells
//! deep, or that the mineshafts all spawned inside a lava sea. Those are eyeball
//! bugs and they need an eyeball, so: one image, one cell per pixel, coloured by
//! the material registry's own `MAT_R`/`MAT_G`/`MAT_B`.
//!
//! ```text
//! cargo run --release --bin worldgen-dump -- --cols 24 --rows 26 --out world.png
//! ```
//!
//! # Why the PNG encoder is in here
//!
//! `godgame-core` has a deliberately short dependency list — it is the crate the
//! purity suite and the benches link, and it must stay free of anything that is
//! not the simulation. A PNG with no compression is a fixed header, a zlib
//! stream of STORED deflate blocks, and two checksums; that is cheaper than an
//! image crate and it cannot rot.
//!
//! # Why rayon
//!
//! Each chunk COLUMN is generated on its own worker with its own [`ChunkGen`].
//! That is only possible because worldgen keeps no shared scratch — no static
//! with interior mutability, no thread local — and `tests/worldgen_purity.rs`
//! asserts the results are identical to the serial ones. This binary is the
//! demonstration.

use std::path::PathBuf;

use rayon::prelude::*;

use godgame_core::config::{CHUNK_CELLS, SEED};
use godgame_core::sim::materials::{CellId, MAT_B, MAT_G, MAT_R};
use godgame_core::sim::worldgen::ChunkGen;

// --- Arguments ---------------------------------------------------------------

struct Args {
    seed: u32,
    x0: i32,
    y0: i32,
    cols: i32,
    rows: i32,
    out: PathBuf,
}

impl Default for Args {
    fn default() -> Args {
        Args {
            seed: SEED,
            // A block that starts two chunks above the surface anchor and runs
            // past UNDERWORLD_FLOOR, so one image spans sky, sea, cavern, deep
            // and the lava floor.
            x0: -8,
            y0: -2,
            cols: 16,
            rows: 26,
            out: PathBuf::from("worldgen.png"),
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: worldgen-dump [--seed N] [--x0 CHUNK] [--y0 CHUNK] [--cols N] [--rows N] \
         [--out PATH]"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(v) = it.next() else { usage() };
        match flag.as_str() {
            "--seed" => a.seed = v.parse().unwrap_or_else(|_| usage()),
            "--x0" => a.x0 = v.parse().unwrap_or_else(|_| usage()),
            "--y0" => a.y0 = v.parse().unwrap_or_else(|_| usage()),
            "--cols" => a.cols = v.parse().unwrap_or_else(|_| usage()),
            "--rows" => a.rows = v.parse().unwrap_or_else(|_| usage()),
            "--out" => a.out = PathBuf::from(v),
            _ => usage(),
        }
    }
    if a.cols < 1 || a.rows < 1 {
        usage();
    }
    a
}

// --- PNG ---------------------------------------------------------------------

/// CRC-32/ISO-HDLC, computed on the fly. A 256-entry table would be faster and
/// this runs once per file.
fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in bytes {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// Adler-32, the checksum a zlib stream ends with.
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// One PNG chunk: length, four-byte type, payload, CRC over type+payload.
fn png_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// An 8-bit truecolour PNG with no compression.
///
/// `rgb` is `w * h * 3` bytes, row-major. Deflate's STORED block type is a
/// legal, if lazy, member of the format: a two-byte zlib header, then blocks of
/// up to 65535 raw bytes each prefixed with `BFINAL/BTYPE=00` and a LEN/~LEN
/// pair, then the Adler-32. Every PNG reader in existence accepts it.
fn encode_png(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), (w * h * 3) as usize);

    // Scanlines with a leading filter byte (0 = None).
    let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
    for y in 0..h as usize {
        raw.push(0);
        let off = y * (w * 3) as usize;
        raw.extend_from_slice(&rgb[off..off + (w * 3) as usize]);
    }

    let mut z = vec![0x78, 0x01]; // CM=8 CINFO=7, FLEVEL=0, no dict, FCHECK ok
    for (i, block) in raw.chunks(65_535).enumerate() {
        let last = u8::from((i + 1) * 65_535 >= raw.len());
        let n = block.len() as u16;
        z.push(last);
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(!n).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits, truecolour, no interlace

    let mut png = Vec::with_capacity(z.len() + 64);
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    png_chunk(&mut png, b"IHDR", &ihdr);
    png_chunk(&mut png, b"IDAT", &z);
    png_chunk(&mut png, b"IEND", &[]);
    png
}

// --- Main --------------------------------------------------------------------

fn main() {
    let a = parse_args();
    let cc = CHUNK_CELLS as usize;
    let w = a.cols as usize * cc;
    let h = a.rows as usize * cc;

    let t0 = std::time::Instant::now();

    // One vertical strip of `rows` chunks per worker, each worker holding its own
    // `ChunkGen`. `map_init` reuses that generator across whatever columns the
    // worker steals, which is exactly the shape a streaming loader wants.
    let strips: Vec<Vec<CellId>> = (0..a.cols)
        .into_par_iter()
        .map_init(
            || ChunkGen::new(a.seed),
            |cg, ci| {
                let cx = a.x0 + ci;
                let mut strip = vec![0 as CellId; cc * h];
                for rj in 0..a.rows {
                    let chunk = cg.generate(cx, a.y0 + rj);
                    let top = rj as usize * cc;
                    for ly in 0..cc {
                        let dst = (top + ly) * cc;
                        strip[dst..dst + cc].copy_from_slice(&chunk[ly * cc..ly * cc + cc]);
                    }
                }
                strip
            },
        )
        .collect();

    let gen_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut rgb = vec![0u8; w * h * 3];
    for (ci, strip) in strips.iter().enumerate() {
        let x0 = ci * cc;
        for y in 0..h {
            for lx in 0..cc {
                let m = strip[y * cc + lx] as usize;
                let p = (y * w + x0 + lx) * 3;
                rgb[p] = MAT_R[m];
                rgb[p + 1] = MAT_G[m];
                rgb[p + 2] = MAT_B[m];
            }
        }
    }

    let png = encode_png(w as u32, h as u32, &rgb);
    match std::fs::write(&a.out, &png) {
        Ok(()) => println!(
            "{} — {w}x{h} cells, {} chunks in {gen_ms:.1} ms ({:.1} us/chunk), {} KiB",
            a.out.display(),
            a.cols * a.rows,
            gen_ms * 1000.0 / f64::from(a.cols * a.rows),
            png.len() / 1024
        ),
        Err(e) => {
            eprintln!("worldgen-dump: {}: {e}", a.out.display());
            std::process::exit(1);
        }
    }
}
