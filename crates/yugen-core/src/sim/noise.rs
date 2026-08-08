//! Dependency-free deterministic noise for worldgen. Same seed, same world, on
//! any machine — everything here is pure and derived from the seed passed into
//! [`Noise::new`], with no global state and no thread-local anything.
//!
//! TWO FAMILIES live here, and picking the wrong one is the difference between
//! "procedural terrain" and "grey mush":
//!
//!   VALUE noise ([`Noise::n1`]/[`Noise::n2`]/[`Noise::fbm1`]/[`Noise::fbm2`])
//!   interpolates a random SCALAR stored at each lattice point. It is one table
//!   lookup per corner, so it is the cheapest thing in this file — but every
//!   lattice point is an extremum, which plants a local max/min on a regular
//!   grid. The eye reads that as axis-aligned blobbing and a soft, "melted"
//!   character. Fine for dither and jitter, where nobody is looking at the field
//!   itself. Bad for anything structural.
//!
//!   GRADIENT noise ([`Noise::g2`]/[`Noise::gfbm2`] and everything built on
//!   them) interpolates a random DIRECTION per lattice point and takes a dot
//!   product with the offset, so the field is exactly zero at every lattice
//!   point and its extrema land at irrational positions between them. No grid
//!   signature, and the zero-set is a smooth wandering contour — which is
//!   precisely the thing the cave carver exploits (a thickened zero-set in 2D IS
//!   a tunnel).
//!
//! COST, because these run per cell over ~1000 cells per chunk:
//!
//! | call | work | relative |
//! |---|---|---|
//! | `n2` | 4 table lookups + 3 lerps | cheapest |
//! | `g2` | 4 lookups + 4 dot products + 3 lerps | ~1.6x n2 |
//! | `gfbm2c` | 3 unrolled octaves of g2 | ~5x n2 — the workhorse |
//! | `ridged2` | octaves x g2 + abs/mul | ~1.1x gfbm2 per octave |
//! | `worley2` | 9 cells x 2 hashes + 9 distances | ~6x n2 — lattice only |
//!
//! Anything with an unbounded `octaves` argument should be hoisted onto a coarse
//! lattice and bilerped (see `gen::caves`) unless it is genuinely high-frequency.
//!
//! # Arithmetic
//!
//! Every integer step is done in `u32` with wrapping multiplies, which is what
//! JavaScript's `Math.imul` and `>>>` did in the original. The bit patterns are
//! identical, so a seed produces the same permutation table and the same hash
//! stream here as it did there. All floating point is `f64`, also as it was —
//! the thresholds downstream are tuned against that precision, and narrowing to
//! `f32` moves boundary outcomes.

/// 16 unit vectors evenly spaced on the circle, indexed by 4 bits out of the
/// permutation table.
///
/// Classic Perlin uses 4 axis-diagonal gradients, which is fast but leaves a
/// visible 45-degree bias — every ridge in the field wants to run diagonally.
/// 16 directions costs two extra table entries and kills that bias, which
/// matters here because the cave system reads the ORIENTATION of the zero-set
/// directly: with 4 gradients every tunnel would trend diagonally across the
/// screen.
const GRAD_N: usize = 16;

/// 2D Perlin with unit gradients peaks at +/- sqrt(2)/2. Scaling by sqrt(2) puts
/// the field in ~[-1,1] so every threshold in worldgen can be read as a fraction
/// of full range. (Real samples rarely exceed +/-0.85; thresholds are tuned
/// against the observed distribution, not the theoretical bound.)
const G2_NORM: f64 = std::f64::consts::SQRT_2;

/// 1 + 0.5 + 0.25 — precomputed so the hot path does no division.
const GFBM3_INV: f64 = 1.0 / 1.75;

/// Result of [`Noise::worley2`].
///
/// The TypeScript version returned a shared mutable record to avoid allocating
/// in a tight loop. Two `f64`s are returned in registers here, so the sharing —
/// and the "read it before the next call" contract that came with it — is gone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cellular {
    /// Distance to the nearest feature point, in cell-size units. ~[0, 1.4].
    pub f1: f64,
    /// F2 - F1, the classic "edge" field: ~0 on a cell boundary, large inside.
    pub edge: f64,
}

/// Result of [`Noise::warp2`]: a displaced coordinate pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Warped {
    pub x: f64,
    pub y: f64,
}

/// xmur3 string-to-u32 hash — used to spread a single seed into good PRNG state.
///
/// Note that it hashes the seed's DECIMAL STRING, not its bytes. That is not an
/// accident worth cleaning up: the permutation table, and therefore every
/// terrain feature in the world, is a function of that exact string.
struct Xmur3 {
    h: u32,
}

impl Xmur3 {
    fn new(s: &str) -> Xmur3 {
        let mut h = 1779033703u32 ^ (s.len() as u32);
        for c in s.chars() {
            h = (h ^ (c as u32)).wrapping_mul(3432918353);
            h = h.rotate_left(13);
        }
        Xmur3 { h }
    }

    fn next(&mut self) -> u32 {
        let mut h = self.h;
        h = (h ^ (h >> 16)).wrapping_mul(2246822507);
        h = (h ^ (h >> 13)).wrapping_mul(3266489909);
        h ^= h >> 16;
        self.h = h;
        h
    }
}

/// mulberry32 — tiny fast PRNG, returns a float in [0,1). Deterministic per state.
#[derive(Clone, Copy, Debug)]
pub struct Mulberry32 {
    a: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Mulberry32 {
        Mulberry32 { a: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6d2b79f5);
        let a = self.a;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

/// Smootherstep — C2-continuous ease so interpolated noise has no visible seams.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Value noise built on a shuffled 256-entry permutation table (Perlin-style
/// hashing). Hashing lattice coords through the table gives a repeatable
/// pseudo-random gradient per point without storing a full 2D field — cheap and
/// stable.
///
/// `Noise` is `Send + Sync` and every sampling method takes `&self`, which is
/// what lets chunk generation run on a rayon pool. The one stateful thing, the
/// scatter PRNG, takes `&mut self` so that it cannot be used by accident from a
/// place that must stay order-independent — the decorators are contractually
/// positional and use [`Noise::hash2`].
pub struct Noise {
    perm: [u8; 512],
    grad_x: [f64; GRAD_N],
    grad_y: [f64; GRAD_N],
    /// Per-seed salt for the coordinate hash, drawn from the seed's own stream
    /// so two seeds dither their biome boundaries differently.
    salt: u32,
    rng: Mulberry32,
}

impl Noise {
    pub fn new(seed: u32) -> Noise {
        // Derive independent streams so lattice hashing and rand() don't
        // interfere. The DRAW ORDER is load-bearing: shuffle, then rand, then
        // salt. Reordering them changes every world.
        let mut seed_gen = Xmur3::new(&seed.to_string());

        let mut base: [u8; 256] = [0; 256];
        for (i, b) in base.iter_mut().enumerate() {
            *b = i as u8;
        }
        // Fisher-Yates shuffle the permutation with a dedicated PRNG stream.
        let mut shuffle = Mulberry32::new(seed_gen.next());
        let mut i = 255usize;
        while i > 0 {
            let j = (shuffle.next_f64() * (i as f64 + 1.0)).floor() as usize;
            base.swap(i, j);
            i -= 1;
        }
        let mut perm = [0u8; 512];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = base[i & 255];
        }

        let rng = Mulberry32::new(seed_gen.next());

        let mut grad_x = [0.0f64; GRAD_N];
        let mut grad_y = [0.0f64; GRAD_N];
        for i in 0..GRAD_N {
            let a = (i as f64 / GRAD_N as f64) * std::f64::consts::PI * 2.0;
            grad_x[i] = a.cos();
            grad_y[i] = a.sin();
        }

        let salt = seed_gen.next();

        Noise {
            perm,
            grad_x,
            grad_y,
            salt,
            rng,
        }
    }

    // -- value noise --------------------------------------------------------

    /// Hash a lattice point to a gradient value in [-1,1] (value-noise style).
    #[inline]
    fn grad2(&self, ix: i32, iy: i32) -> f64 {
        let p = self.perm[(ix & 255) as usize] as i32;
        let h = self.perm[((p + iy) & 511) as usize];
        h as f64 / 127.5 - 1.0
    }

    #[inline]
    fn grad1(&self, ix: i32) -> f64 {
        self.perm[(ix & 255) as usize] as f64 / 127.5 - 1.0
    }

    /// 2D value noise in [-1,1]. Cheap; use for dither, not for structure.
    pub fn n2(&self, x: f64, y: f64) -> f64 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = fade(x - x0);
        let fy = fade(y - y0);
        let (ix, iy) = (x0 as i32, y0 as i32);
        let v00 = self.grad2(ix, iy);
        let v10 = self.grad2(ix + 1, iy);
        let v01 = self.grad2(ix, iy + 1);
        let v11 = self.grad2(ix + 1, iy + 1);
        lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy)
    }

    /// 1D value noise in [-1,1] — legacy heightmaps sample this.
    pub fn n1(&self, x: f64) -> f64 {
        let x0 = x.floor();
        let fx = fade(x - x0);
        let ix = x0 as i32;
        lerp(self.grad1(ix), self.grad1(ix + 1), fx)
    }

    /// Fractal (fBm) sum of `octaves` of [`Noise::n2`], normalised to ~[-1,1].
    pub fn fbm2(&self, x: f64, y: f64, octaves: u32) -> f64 {
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        for _ in 0..octaves {
            sum += amp * self.n2(x * freq, y * freq);
            norm += amp;
            amp *= 0.5; // each octave adds finer, weaker detail
            freq *= 2.0;
        }
        sum / norm
    }

    /// Fractal 1D noise, normalised to ~[-1,1].
    pub fn fbm1(&self, x: f64, octaves: u32) -> f64 {
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        for _ in 0..octaves {
            sum += amp * self.n1(x * freq);
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    // -- gradient noise -----------------------------------------------------

    /// 2D GRADIENT (Perlin) noise in ~[-1,1]. Zero at every lattice point,
    /// extrema between them — no grid signature, and a smooth wandering
    /// zero-contour. This is the primitive every terrain and cave field is built
    /// on.
    ///
    /// `perm[(perm[ix & 255] + iy) & 511] & 15` picks one of 16 directions. The
    /// `& 511` is what makes negative coordinates work without a branch: the sum
    /// is an `i32` and only its low bits are consulted, so -1 and 511 land in
    /// the same slot deterministically.
    ///
    /// `fade` and `lerp` are written out longhand here rather than called. This
    /// is the single hottest function in worldgen — several samples per cell,
    /// ~1000 cells per chunk — and at that call density the calls it would
    /// otherwise make showed up as ~6% of total generation time in a profile.
    pub fn g2(&self, x: f64, y: f64) -> f64 {
        let xf = x.floor();
        let yf = y.floor();
        let fx = x - xf;
        let fy = y - yf;
        let big_x = xf as i32;
        let big_y = yf as i32;
        let u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
        let v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

        let p0 = self.perm[(big_x & 255) as usize] as i32;
        let p1 = self.perm[((big_x + 1) & 255) as usize] as i32;
        let h00 = (self.perm[((p0 + big_y) & 511) as usize] & 15) as usize;
        let h10 = (self.perm[((p1 + big_y) & 511) as usize] & 15) as usize;
        let h01 = (self.perm[((p0 + big_y + 1) & 511) as usize] & 15) as usize;
        let h11 = (self.perm[((p1 + big_y + 1) & 511) as usize] & 15) as usize;

        let gx = fx - 1.0;
        let gy = fy - 1.0;
        let n00 = self.grad_x[h00] * fx + self.grad_y[h00] * fy;
        let n10 = self.grad_x[h10] * gx + self.grad_y[h10] * fy;
        let n01 = self.grad_x[h01] * fx + self.grad_y[h01] * gy;
        let n11 = self.grad_x[h11] * gx + self.grad_y[h11] * gy;

        let a = n00 + (n10 - n00) * u;
        let b = n01 + (n11 - n01) * u;
        (a + (b - a) * v) * G2_NORM
    }

    /// fBm over [`Noise::g2`], normalised to ~[-1,1]. Lacunarity 2, gain 0.5.
    pub fn gfbm2(&self, x: f64, y: f64, octaves: u32) -> f64 {
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        for _ in 0..octaves {
            sum += amp * self.g2(x * freq, y * freq);
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    /// Unrolled 3-octave [`Noise::gfbm2`] — the per-cell workhorse.
    #[inline]
    pub fn gfbm2c(&self, x: f64, y: f64) -> f64 {
        (self.g2(x, y) + 0.5 * self.g2(x * 2.0, y * 2.0) + 0.25 * self.g2(x * 4.0, y * 4.0))
            * GFBM3_INV
    }

    /// Ridged multifractal in ~[0,1]: `(1-|g2|)^2` per octave, with each octave
    /// WEIGHTED by the previous one so detail only survives where the coarse
    /// octave was already close to a ridge.
    ///
    /// That feedback is what turns fBm from "bumpy" into "structured": ridges
    /// get sharp crests with smooth flanks (mountains) and, read as a level set,
    /// the high band is a thin connected filament (cave tunnels). `gain`
    /// controls how aggressively an octave gates the next — higher = thinner,
    /// more branching crests.
    pub fn ridged2(&self, x: f64, y: f64, octaves: u32, gain: f64) -> f64 {
        // Single-octave fast path. Worldgen's tunnel fields run at one octave
        // and are the most-evaluated noise in the generator; the general loop's
        // setup, weight feedback and final divide are all dead work for that
        // case.
        if octaves == 1 {
            let n = 1.0 - self.g2(x, y).abs();
            return n * n;
        }
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        let mut weight = 1.0f64;
        for _ in 0..octaves {
            let mut n = 1.0 - self.g2(x * freq, y * freq).abs();
            n *= n; // square: sharpens the crest, widens the flanks
            n *= weight; // this octave only exists where the last was near a ridge
            weight = (n * gain).clamp(0.0, 1.0);
            sum += n * amp;
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    /// Unrolled 3-octave [`Noise::ridged2`] at the default gain.
    #[inline]
    pub fn ridged2c(&self, x: f64, y: f64) -> f64 {
        self.ridged2(x, y, 3, 2.0)
    }

    /// Billow in ~[-1,1]: `2|g2|-1` per octave. The inverse character to ridged
    /// — rounded lumps with creased valleys. Dunes and rolling hills.
    pub fn billow2(&self, x: f64, y: f64, octaves: u32) -> f64 {
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        for _ in 0..octaves {
            sum += amp * (2.0 * self.g2(x * freq, y * freq).abs() - 1.0);
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    // -- hashing ------------------------------------------------------------

    /// Stateless hash of an INTEGER cell coordinate pair into [0,1).
    ///
    /// Unlike [`Noise::rand`] this depends only on `(x, y)` and the seed, never
    /// on call order, so worldgen can dither a boundary per cell and still
    /// regenerate bit-identically in any chunk order. Handles negative
    /// coordinates: the multiplies wrap and only the bit pattern matters.
    #[inline]
    pub fn hash2(&self, x: i32, y: i32) -> f64 {
        let mut h =
            (x as u32).wrapping_mul(0x27d4eb2d) ^ (y as u32).wrapping_mul(0x165667b1) ^ self.salt;
        h = (h ^ (h >> 15)).wrapping_mul(0x2c1b3c6d);
        h = (h ^ (h >> 12)).wrapping_mul(0x297a2d39);
        h ^= h >> 15;
        (h as f64) / 4294967296.0
    }

    /// Next PRNG float in [0,1) — for scatter and placement decisions.
    ///
    /// Takes `&mut self` on purpose. Worldgen is contractually a pure function
    /// of `(chunkX, chunkY, seed)`, and anything order-dependent breaks that, so
    /// a decorator that reaches for this instead of [`Noise::hash2`] should not
    /// compile.
    pub fn rand(&mut self) -> f64 {
        self.rng.next_f64()
    }

    /// Worley / cellular noise over a jittered grid of period `cell_size` (in
    /// the SAME units as x/y).
    pub fn worley2(&self, x: f64, y: f64, cell_size: f64) -> Cellular {
        let inv = 1.0 / cell_size;
        let gx = (x * inv).floor() as i32;
        let gy = (y * inv).floor() as i32;
        let mut f1 = f64::INFINITY;
        let mut f2 = f64::INFINITY;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = gx + dx;
                let cy = gy + dy;
                // Feature point jittered inside its own cell — two decorrelated
                // hashes of the same integer cell, so the point set is a pure
                // function of position and identical from any chunk that reaches
                // this cell.
                let px = (cx as f64 + self.hash2(cx, cy)) * cell_size;
                let py = (cy as f64 + self.hash2(cx + 7919, cy - 104729)) * cell_size;
                let ex = px - x;
                let ey = py - y;
                let d = (ex * ex + ey * ey).sqrt();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
        Cellular {
            f1: f1 * inv,
            edge: (f2 - f1) * inv,
        }
    }

    /// Domain warp: offset `(x,y)` by a low-frequency vector field before
    /// sampling something else at the result.
    ///
    /// This is the highest-value-per-flop technique in the file. fBm alone is
    /// statistically isotropic — it has no *shapes*, only texture. Warping the
    /// input coordinates makes the field stretch, fold and shear, which is what
    /// produces overhangs, meanders and gnarled cave outlines instead of
    /// evenly-scattered blobs.
    ///
    /// The offsets below are deliberately large and non-integral: `g2` hashes
    /// `floor(x)`/`floor(y)`, so an integer offset would just re-use the same
    /// lattice row and correlate the two components into a pure shear.
    #[inline]
    pub fn warp2(&self, x: f64, y: f64, strength: f64, freq: f64) -> Warped {
        Warped {
            x: x + strength * self.g2(x * freq + 137.31, y * freq - 41.77),
            y: y + strength * self.g2(x * freq - 613.19, y * freq + 917.53),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_permutation_table_is_a_permutation() {
        let n = Noise::new(2334);
        let mut seen = [false; 256];
        for i in 0..256 {
            seen[n.perm[i] as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "the shuffle lost a value");
        // The upper half mirrors the lower, which is what makes `& 511` safe.
        for i in 0..512 {
            assert_eq!(n.perm[i], n.perm[i & 255]);
        }
    }

    #[test]
    fn gradient_noise_is_zero_at_every_lattice_point() {
        // The defining property of gradient noise, and the reason the cave
        // carver can treat a thickened zero-set as a tunnel.
        let n = Noise::new(2334);
        for x in -4..4 {
            for y in -4..4 {
                let v = n.g2(x as f64, y as f64);
                assert!(v.abs() < 1e-12, "g2({x},{y}) = {v}");
            }
        }
    }

    #[test]
    fn hash2_is_stateless_and_handles_negative_coordinates() {
        let n = Noise::new(2334);
        let a = n.hash2(-7, -13);
        for _ in 0..100 {
            n.hash2(999, -999);
        }
        assert_eq!(n.hash2(-7, -13), a, "hash2 depends on call order");
        for x in -50..50 {
            for y in -50..50 {
                let h = n.hash2(x, y);
                assert!((0.0..1.0).contains(&h), "hash2({x},{y}) = {h}");
            }
        }
    }

    #[test]
    fn two_seeds_disagree_everywhere_that_matters() {
        let a = Noise::new(2334);
        let b = Noise::new(2335);
        assert_ne!(a.salt, b.salt);
        assert_ne!(a.perm, b.perm);
        let differs = (0..64).filter(|&i| a.hash2(i, i) != b.hash2(i, i)).count();
        assert_eq!(differs, 64);
    }

    #[test]
    fn noise_stays_inside_its_advertised_range() {
        let n = Noise::new(2334);
        let mut worst_g2: f64 = 0.0;
        let mut x = -20.0;
        while x < 20.0 {
            let mut y = -20.0;
            while y < 20.0 {
                worst_g2 = worst_g2.max(n.g2(x, y).abs());
                assert!((0.0..=1.0).contains(&n.ridged2(x, y, 3, 2.0)));
                assert!(n.n2(x, y).abs() <= 1.0);
                y += 0.37;
            }
            x += 0.37;
        }
        // The doc claims real samples rarely exceed 0.85 and never leave [-1,1].
        assert!(worst_g2 <= 1.0, "g2 escaped its normalisation: {worst_g2}");
        assert!(worst_g2 > 0.5, "g2 never got near full range: {worst_g2}");
    }

    #[test]
    fn the_scatter_prng_is_reproducible_per_seed() {
        let mut a = Noise::new(2334);
        let mut b = Noise::new(2334);
        for _ in 0..1000 {
            let v = a.rand();
            assert_eq!(v, b.rand());
            assert!((0.0..1.0).contains(&v));
        }
    }
}
