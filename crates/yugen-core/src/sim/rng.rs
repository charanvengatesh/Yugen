//! The simulation's random source.
//!
//! `Math.random` is unseedable, so an identical seed + edit sequence would not
//! replay identically. This is a plain xorshift32: one multiply-free integer
//! step, no allocation, and reproducible — same seed and same sequence of ticks
//! and edits give the same world every time.
//!
//! It lived in its own module in the TypeScript because BOTH automata.ts and
//! reactions.ts drew from it and automata already imported reactions; putting
//! the state in either of them would have made the import graph circular.
//!
//! # What changed in the port
//!
//! The TypeScript held `state` as a module-level `let` and exported
//! `seedSim`/`nextU32`/`chance`/`randInt` as free functions over it — a global
//! mutable singleton, which is exactly the shape Rust has no clean spelling for
//! (a `static mut` is `unsafe`, a `thread_local` is a hidden global, and a
//! `Mutex` would put a lock in the hottest loop in the game).
//!
//! So the stream became a value: [`SimRng`] is threaded through the sweep by
//! `&mut`. **The arithmetic and, critically, the ORDER of consumption are
//! unchanged** — the automata's determinism is order-dependent, one global
//! stream drawn in one fixed sweep order, and moving the state from a module
//! binding into a struct field must not perturb a single draw.

/// 2^-32, the scale that turns a raw `u32` into a float in `[0, 1)`.
///
/// Written out rather than computed so it reads identically to the TypeScript
/// literal it came from.
const INV_2_POW_32: f64 = 2.3283064365386963e-10;

/// A xorshift32 stream. Never seeded to 0 — xorshift is stuck at zero.
#[derive(Clone, Debug)]
pub struct SimRng {
    state: u32,
}

impl SimRng {
    /// The state a fresh, unseeded stream starts from (the golden-ratio
    /// constant the TypeScript initialised its module `state` to). Also the
    /// fallback a zero seed falls back to.
    pub const DEFAULT_STATE: u32 = 0x9e37_79b9;

    /// A stream at the default state, before any `seed` call.
    #[inline]
    pub fn new() -> SimRng {
        SimRng {
            state: SimRng::DEFAULT_STATE,
        }
    }

    /// A stream already reseeded to `seed`. See [`SimRng::seed`].
    #[inline]
    pub fn seeded(seed: u32) -> SimRng {
        let mut rng = SimRng::new();
        rng.seed(seed);
        rng
    }

    /// Reseed the sim's RNG (the worker called this with the world seed at init).
    ///
    /// A zero seed falls back to the default state, mirroring the TypeScript's
    /// `seed >>> 0 || 0x9e3779b9` — xorshift cannot leave state 0.
    pub fn seed(&mut self, seed: u32) {
        self.state = if seed != 0 {
            seed
        } else {
            SimRng::DEFAULT_STATE
        };
        // Discard a few steps so nearby seeds don't produce correlated openings.
        for _ in 0..8 {
            self.next_u32();
        }
    }

    /// Next raw 32-bit value.
    ///
    /// The three shifts are the whole algorithm. In JavaScript they relied on
    /// `<<`/`>>>` being defined on a 32-bit view of a double; in Rust the type
    /// IS `u32`, so `<<` and `>>` are already the same operations — `>>` on an
    /// unsigned is the logical shift `>>>` was.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        self.state
    }

    /// True with probability `p` (0..1).
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        if p <= 0.0 {
            return false;
        }
        if p >= 1.0 {
            return true;
        }
        // 2^-32 scaling keeps this a single float multiply.
        //
        // Note what the two early-outs mean for the STREAM: a certainty or an
        // impossibility draws nothing. Several rules lean on that (a reaction
        // with probability 1 never touches the RNG), so the guards are part of
        // the consumption order, not just an optimisation.
        f64::from(self.next_u32()) * INV_2_POW_32 < p
    }

    /// Uniform integer in `[0, n)`. `n` must be >= 1.
    #[inline]
    pub fn rand_int(&mut self, n: i32) -> i32 {
        // `as i32` truncates toward zero exactly as the TypeScript's `| 0` did;
        // the product is in `[0, n)` so it never saturates.
        (f64::from(self.next_u32()) * INV_2_POW_32 * f64::from(n)) as i32
    }
}

impl Default for SimRng {
    #[inline]
    fn default() -> SimRng {
        SimRng::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One xorshift32 step, computed the long way, as an oracle for `next_u32`.
    fn step(mut x: u32) -> u32 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    }

    #[test]
    fn the_stream_matches_a_hand_computed_xorshift32_sequence() {
        // Hand-computed from the documented default state, one step at a time.
        // If the port ever "improves" the shifts, every world diverges.
        let mut want = SimRng::DEFAULT_STATE;
        let mut rng = SimRng::new();
        for i in 0..64 {
            want = step(want);
            assert_eq!(rng.next_u32(), want, "step {i} diverged");
        }

        // The first three values, written out, so a broken shift is caught even
        // if the oracle above were wrong in the same way.
        let mut rng = SimRng::new();
        let a = 0x9e37_79b9u32;
        let a = a ^ (a << 13);
        let a = a ^ (a >> 17);
        let a = a ^ (a << 5);
        assert_eq!(rng.next_u32(), a);
    }

    #[test]
    fn seeding_discards_exactly_eight_steps() {
        let mut oracle = SimRng { state: 12345 };
        for _ in 0..8 {
            oracle.next_u32();
        }
        let seeded = SimRng::seeded(12345);
        assert_eq!(seeded.state, oracle.state);
    }

    #[test]
    fn a_zero_seed_falls_back_to_the_default_state() {
        // Not just "some non-zero state" — the SAME state an unseeded stream
        // would have had after its eight discards.
        let mut oracle = SimRng::new();
        for _ in 0..8 {
            oracle.next_u32();
        }
        assert_eq!(SimRng::seeded(0).state, oracle.state);
    }

    #[test]
    fn certainty_and_impossibility_draw_nothing() {
        // Part of the consumption order: a p>=1 or p<=0 roll must not advance
        // the stream, or every reaction with probability 1 shifts the world.
        let mut rng = SimRng::new();
        let before = rng.state;
        assert!(rng.chance(1.0));
        assert!(rng.chance(2.0));
        assert!(!rng.chance(0.0));
        assert!(!rng.chance(-1.0));
        assert_eq!(rng.state, before);

        let _ = rng.chance(0.5);
        assert_ne!(rng.state, before, "a real roll must advance the stream");
    }

    #[test]
    fn rand_int_stays_in_range() {
        let mut rng = SimRng::seeded(0xdead_beef);
        for n in 1..=8 {
            for _ in 0..500 {
                let v = rng.rand_int(n);
                assert!((0..n).contains(&v), "rand_int({n}) returned {v}");
            }
        }
    }

    #[test]
    fn chance_is_roughly_calibrated() {
        let mut rng = SimRng::seeded(7);
        let mut hits = 0;
        for _ in 0..100_000 {
            if rng.chance(0.25) {
                hits += 1;
            }
        }
        assert!(
            (24_000..26_000).contains(&hits),
            "p=0.25 gave {hits}/100000"
        );
    }
}
