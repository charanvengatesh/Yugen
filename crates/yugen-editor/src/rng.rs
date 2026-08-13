//! A seeded stream, so a roll can be written down and rolled again.
//!
//! # Why this is not `rand`
//!
//! Three reasons, in ascending order of weight.
//!
//! `Cargo.toml` argues each of this crate's dependencies at paragraph length,
//! and a fourth would need the same argument. It does not have one: what is
//! wanted here is twenty lines of shift-and-xor.
//!
//! The house has already restated this pattern — `yugen-core`'s `SimRng`, the
//! mob spawner's stream, and [`crate::synth::noise_at`]'s xorshift over the
//! sample index. A fourth restatement is the conventional move in this tree,
//! not the lazy one.
//!
//! And decisively: **`rand` does not promise stream stability across minor
//! versions.** A generator whose seed is written into a record's comment is
//! making a promise that the seed reproduces the drawing, and a `cargo update`
//! that silently reshuffled the stream would break every one of those promises
//! at once, invisibly, in files nobody was editing. Twenty lines that will
//! behave identically in ten years is the cheaper side of that trade.
//!
//! # Where the non-determinism is
//!
//! In exactly one function, [`fresh_seed`], and it surfaces as a number rather
//! than as an effect: a reroll draws a new seed, shows it, and every generator
//! downstream is a pure function of it. So "I liked that one" is recoverable by
//! typing four bytes back in, and a generated record can carry the seed that
//! made it.

/// A xorshift32 stream. See the module header for why this is not `rand`.
///
/// Not cryptographic and not trying to be. What it has to be is *the same
/// sequence forever*, which is a property a three-line shift register has and a
/// dependency does not.
pub struct Rng {
    state: u32,
}

impl Rng {
    /// The state a zero seed falls back to.
    ///
    /// Zero is a fixed point of xorshift — it would emit zero forever — so it
    /// cannot be a state. The golden-ratio constant is the same one
    /// [`crate::synth::noise_at`] multiplies by, for the same reason: it has a
    /// well-mixed bit pattern and no other significance.
    pub const DEFAULT_STATE: u32 = 0x9e37_79b9;

    /// A stream from a seed.
    ///
    /// The first few outputs of a xorshift seeded with a small number are
    /// themselves small — seed 1 and seed 2 would start with visibly similar
    /// draws, which for a sprite generator means two "different" rolls that look
    /// alike. Discarding a few fixes it, and doing it here means no caller has
    /// to know.
    pub fn seeded(seed: u32) -> Rng {
        let mut rng = Rng {
            state: if seed == 0 { Self::DEFAULT_STATE } else { seed },
        };
        for _ in 0..8 {
            rng.next_u32();
        }
        rng
    }

    /// The next raw draw.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// A draw in `[0, 1)`.
    ///
    /// Scaled from the top 24 bits, which is every bit an `f32` mantissa can
    /// hold. Taking the whole `u32` and dividing would round the last byte away
    /// and let the result reach exactly 1.0, which every `lo + t * (hi - lo)`
    /// below would then push one step past its range.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A draw in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }

    /// A draw in `0..n`. Zero for `n == 0`, so a caller with an empty range gets
    /// an index it can clamp rather than a panic.
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        // Modulo, and the bias it carries is real but bounded by `n / 2^32`.
        // For the counts here — palette steps, grid coordinates, a handful of
        // waveforms — that is far below anything an eye or a test could see,
        // and rejection sampling would make the stream's length depend on the
        // draws, which is a worse property for something meant to reproduce.
        self.next_u32() % n
    }

    /// True with probability `p`. Anything at or below 0 is never, at or above
    /// 1 is always.
    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }

    /// One of `xs`. Panics on an empty slice, because every caller here picks
    /// from a `const` table and an empty one is a bug in the table.
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        assert!(!xs.is_empty(), "nothing to pick from");
        &xs[self.below(xs.len() as u32) as usize]
    }
}

/// A seed for a fresh roll.
///
/// The one place non-determinism enters the generators, and it enters as a
/// number the designer can read off the screen and type back in.
///
/// `counter` is mixed in and incremented because the clock is not fine enough on
/// its own: a burst of clicks on `Reroll` can land inside one tick, and two
/// identical seeds in a row read as a broken button.
pub fn fresh_seed(counter: &mut u32) -> u32 {
    *counter = counter.wrapping_add(1);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() ^ d.as_secs() as u32)
        .unwrap_or(Rng::DEFAULT_STATE);
    // Through a stream rather than straight out: the low bits of a clock walk
    // upward, and a seed that walks gives rolls that drift instead of rolls that
    // differ.
    Rng::seeded(nanos.wrapping_mul(2_654_435_761).wrapping_add(*counter)).next_u32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_is_the_whole_input() {
        // The property every generator downstream inherits, and the reason a
        // seed is worth writing into a comment.
        for seed in [0u32, 1, 2, 7, 4096, u32::MAX] {
            let a: Vec<u32> = (0..64).map(|_| Rng::seeded(seed).next_u32()).collect();
            let b: Vec<u32> = (0..64).map(|_| Rng::seeded(seed).next_u32()).collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn a_zero_seed_is_a_stream_and_not_a_fixed_point() {
        // Zero is a fixed point of xorshift: without the fallback, seed 0 would
        // emit zero forever and "roll 0" would be a blank sprite.
        let mut rng = Rng::seeded(0);
        let draws: Vec<u32> = (0..16).map(|_| rng.next_u32()).collect();
        assert!(draws.iter().any(|&x| x != 0));
        assert!(draws.windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn neighbouring_seeds_do_not_start_alike() {
        // What the discard in `seeded` buys. A raw xorshift seeded with 1 and
        // with 2 opens with near-identical small numbers, and for a sprite
        // generator that means two rolls that look like the same drawing.
        let a = Rng::seeded(1).next_u32();
        let b = Rng::seeded(2).next_u32();
        let apart = (a ^ b).count_ones();
        assert!(apart > 8, "seeds 1 and 2 differ in only {apart} bits");
    }

    #[test]
    fn unit_stays_inside_its_range() {
        // `range` is `lo + unit * (hi - lo)`, so a `unit` that could reach 1.0
        // would put every derived draw one step outside the schema range it was
        // clamped to.
        let mut rng = Rng::seeded(12345);
        for _ in 0..100_000 {
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u), "unit produced {u}");
        }
    }

    #[test]
    fn range_and_below_stay_inside_theirs() {
        let mut rng = Rng::seeded(99);
        for _ in 0..50_000 {
            let x = rng.range(20.0, 8000.0);
            assert!((20.0..8000.0).contains(&x), "range produced {x}");
            assert!(rng.below(7) < 7);
        }
        assert_eq!(rng.below(0), 0, "an empty range is an index, not a panic");
    }

    #[test]
    fn chance_is_roughly_its_probability() {
        let mut rng = Rng::seeded(7);
        let hits = (0..100_000).filter(|_| rng.chance(0.25)).count();
        assert!((24_000..26_000).contains(&hits), "{hits} of 100000");

        let mut rng = Rng::seeded(7);
        assert_eq!((0..1000).filter(|_| rng.chance(0.0)).count(), 0);
        let mut rng = Rng::seeded(7);
        assert_eq!((0..1000).filter(|_| rng.chance(1.0)).count(), 1000);
    }

    #[test]
    fn a_fresh_seed_differs_from_the_one_before_it() {
        // Two clicks inside one clock tick must not roll the same sprite. The
        // counter is what guarantees it; the clock alone does not.
        let mut counter = 0;
        let seeds: Vec<u32> = (0..64).map(|_| fresh_seed(&mut counter)).collect();
        let unique: std::collections::BTreeSet<u32> = seeds.iter().copied().collect();
        assert_eq!(unique.len(), seeds.len(), "a burst of rerolls repeated");
    }
}
