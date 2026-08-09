//! The render side's own random stream.
//!
//! One type, and it is here rather than in [`super::system`] because its whole
//! reason for existing is a boundary: juice must never be able to move a cell.
//! A file of its own is the cheapest way to keep that argument next to the only
//! code it constrains, instead of thirty lines into a module about pools.
//!
//! Shared with [`crate::effects`], which shakes the screen on the same events
//! this pool bursts on and has the same reason not to reach for a sim stream.

/// The render side's own random stream.
///
/// Deliberately not the sim's [`SimRng`](yugen_core::sim::rng::SimRng) and not
/// the creatures' [`MobRng`](yugen_core::entities::mobs::MobRng): both of those
/// are consumed in a fixed ORDER that a replay of the same seed has to reproduce
/// exactly, and a spark drawn from either would desync the world against a
/// change in how much juice happened to be on screen. Juice must never be able
/// to move a cell.
///
/// It is xorshift32 with a different default state, which is the same answer
/// `MobRng` gave to the same question one layer down.
///
/// Shared with [`crate::effects`], which shakes the screen on the same events
/// this pool bursts on and has the same reason not to reach for a sim stream.
#[derive(Clone, Copy, Debug)]
pub struct JuiceRng {
    state: u32,
}

impl JuiceRng {
    /// The state a fresh stream starts from. Xorshift is stuck at zero, so it
    /// can never be that.
    pub const DEFAULT_STATE: u32 = 0x9e37_79b9;

    /// A stream at the default state.
    #[inline]
    pub fn new() -> JuiceRng {
        JuiceRng {
            state: JuiceRng::DEFAULT_STATE,
        }
    }

    /// A stream reseeded to `seed`. A zero seed falls back to the default state.
    #[inline]
    pub fn seeded(seed: u32) -> JuiceRng {
        JuiceRng {
            state: if seed != 0 {
                seed
            } else {
                JuiceRng::DEFAULT_STATE
            },
        }
    }

    /// Next raw 32-bit value.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform float in `[0, 1)` — the `Math.random()` every call site used.
    ///
    /// The `f64` multiply is deliberate, and is `MobRng`'s: 24 bits of `f32`
    /// mantissa cannot represent the whole `u32` range, so doing the divide at
    /// `f32` would quantise the stream itself. The result narrows immediately,
    /// because everything it feeds is `f32`.
    #[inline]
    pub fn rand(&mut self) -> f32 {
        (f64::from(self.next_u32()) * 2.328_306_436_538_696_3e-10) as f32
    }

    /// Uniform float in `[a, b)`.
    #[inline]
    pub fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand()
    }
}

impl Default for JuiceRng {
    #[inline]
    fn default() -> JuiceRng {
        JuiceRng::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_zero_seed_still_produces_a_stream() {
        // Xorshift is stuck at zero forever, so the fallback is not a nicety.
        let mut z = JuiceRng::seeded(0);
        assert_ne!(z.next_u32(), 0);
        let mut r = JuiceRng::new();
        for _ in 0..64 {
            let v = r.rand();
            assert!((0.0..1.0).contains(&v), "rand out of range: {v}");
        }
    }
}
