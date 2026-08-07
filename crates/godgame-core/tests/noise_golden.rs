//! The noise field's golden baseline: every entry point must still return what
//! it returned the day it was locked down.
//!
//! `tests/noise.golden.json` was produced by running the original
//! `src/sim/noise.ts` under node and dumping every entry point at 40 sample
//! points for four seeds. **That provenance is history now** — the port is over,
//! the original is not coming back, and this file is no longer an authority on
//! TypeScript. It is this project's own baseline, and the question it answers has
//! changed from *"does this match the original"* to **"has any of this moved"*.
//!
//! Nothing here needs to change for new content: noise is a function of a seed
//! and a coordinate and knows nothing about the registry. It is renamed for
//! consistency with the other four baselines and for one substantive reason —
//! `registry_golden.rs`'s rule now applies to all of them. **A baseline is
//! changed by blessing it deliberately and reading the diff, never by editing it
//! to make a red test go green.** There is no bless path in this file because
//! nothing has ever needed one; `registry_golden.rs` is the pattern if that
//! changes.
//!
//! Worldgen was ported at behavioral parity, not bit-exactness, so nothing
//! *requires* the two to agree. They agree anyway, because every integer step
//! was translated as a `u32` wrapping operation and every float kept at `f64` —
//! and having them agree turns "does my cave carver look right" into a
//! mechanical check. Every threshold downstream is tuned against this
//! distribution; if the primitives drift, the terrain drifts with them and there
//! is no way to see it by eye.

use godgame_core::sim::noise::Noise;
use serde_json::Value as J;

fn samples() -> J {
    serde_json::from_str(include_str!("noise.golden.json"))
        .expect("noise.golden.json is not valid JSON")
}

/// Decode one hex-encoded IEEE754 double.
///
/// The fixture carries bit patterns rather than decimal literals on purpose. A
/// decimal round-trip only preserves a value if both parsers are correctly
/// rounded to the last bit, and `serde_json`'s is not — it was reporting
/// one-ULP "divergences" on `hash2`, a value that is an exact `n / 2^32` on both
/// sides and cannot actually differ. Bits are the value.
fn floats(v: &J) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f64::from_bits(u64::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
        .collect()
}

/// Distance between two `f64`s counted in representable steps.
///
/// A raw difference is useless for judging a port: 1e-17 is enormous near zero
/// and invisible near one. ULP distance is the only scale-free way to say "these
/// are the same number, modulo the last bit".
fn ulps_apart(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    let (ai, bi) = (a.to_bits(), b.to_bits());
    ai.abs_diff(bi)
}

/// How far each entry point may differ from the original, in ULP.
///
/// The split is not arbitrary. Anything decided by INTEGER arithmetic must be
/// bit-exact, because there is no rounding freedom in it: if `hash2` or the
/// scatter PRNG drifts by even one bit, the port got a width, a shift or a
/// wrapping multiply wrong, and every world would differ for a reason that is a
/// bug rather than a rounding choice. `hash2` is literally `n / 2^32` on both
/// sides — an exactly representable value.
///
/// The float entry points are allowed a little slack, because LLVM on aarch64
/// and V8 are each free to contract `a * b + c` into a fused multiply-add or
/// not, and neither answer is more correct. `g2` is a chain of four dot products
/// and three lerps, so a single contraction decision moves the last bit; the
/// fBm sums stack three or four of those; and `warp2` multiplies a `g2` result
/// by a strength of 90 before adding it to a coordinate, which scales a
/// last-bit difference up by the same factor.
///
/// Even the loosest budget here is around 1e-14 on a value of order 1 — eleven
/// orders of magnitude below the smallest threshold any generator compares
/// against, so none of it can change a cell.
fn ulp_budget(what: &str) -> u64 {
    match what {
        // Integer arithmetic. No rounding freedom exists.
        "hash2" | "rand" => 0,
        // One fused multiply-add's worth of disagreement.
        "g2" | "n1" | "n2" => 2,
        // A domain warp amplifies a `g2` last-bit by its strength (90).
        "warp.x" | "warp.y" => 64,
        // Sums and products of several `g2` samples.
        _ => 8,
    }
}

/// Rounding noise is rare; an algorithmic divergence is not. If the port had a
/// real mistake — a wrong constant, a wrong octave count, a wrong lerp order —
/// it would move most samples, not a handful. Requiring the overwhelming
/// majority to be bit-identical is what keeps the ULP budgets above from
/// becoming a blanket excuse.
const MIN_EXACT_FRACTION: f64 = 0.95;

#[test]
fn every_noise_entry_point_matches_the_typescript_original() {
    let all = samples();
    let mut compared = 0usize;
    let mut exact = 0usize;
    let mut worst: Vec<(String, u64, f64, f64)> = Vec::new();
    let mut problems: Vec<String> = Vec::new();

    for (seed_str, rec) in all.as_object().unwrap() {
        let seed: u32 = seed_str.parse().unwrap();
        let n = Noise::new(seed);
        let mut rng = Noise::new(seed);

        for i in 0..40usize {
            let x = (i as f64 * 7.31) - 140.0;
            let y = (i as f64 * -4.77) + 61.3;
            let c = n.worley2(x, y, 13.0);
            let w = n.warp2(x, y, 90.0, 0.0013);

            let checks: [(&str, f64, f64); 14] = [
                ("g2", floats(&rec["g2"])[i], n.g2(x, y)),
                ("n2", floats(&rec["n2"])[i], n.n2(x * 0.31, y * 0.17)),
                ("n1", floats(&rec["n1"])[i], n.n1(x * 0.013)),
                (
                    "fbm2",
                    floats(&rec["fbm2"])[i],
                    n.fbm2(x * 0.01, y * 0.01, 4),
                ),
                (
                    "gfbm2c",
                    floats(&rec["gfbm2c"])[i],
                    n.gfbm2c(x * 0.003, y * 0.003),
                ),
                (
                    "ridged1",
                    floats(&rec["ridged1"])[i],
                    n.ridged2(x * 0.02, y * 0.02, 1, 2.0),
                ),
                (
                    "ridged3",
                    floats(&rec["ridged3"])[i],
                    n.ridged2(x * 0.02, y * 0.02, 3, 2.0),
                ),
                (
                    "billow2",
                    floats(&rec["billow2"])[i],
                    n.billow2(x * 0.02, y * 0.02, 2),
                ),
                (
                    "hash2",
                    floats(&rec["hash2"])[i],
                    n.hash2(x.round() as i32, y.round() as i32),
                ),
                ("worley.f1", floats(&rec["worleyF1"])[i], c.f1),
                ("worley.edge", floats(&rec["worleyEdge"])[i], c.edge),
                ("warp.x", floats(&rec["warpX"])[i], w.x),
                ("warp.y", floats(&rec["warpY"])[i], w.y),
                // The scatter PRNG is a stream, so it is walked in step.
                ("rand", floats(&rec["rand"])[i], rng.rand()),
            ];

            for (what, want, got) in checks {
                let ulps = ulps_apart(want, got);
                let budget = ulp_budget(what);
                if ulps == 0 {
                    exact += 1;
                }
                if ulps > budget {
                    problems.push(format!(
                        "seed {seed} {what}[{i}]: TS {want:?}, Rust {got:?} ({ulps} ulp, budget {budget})"
                    ));
                }
                match worst.iter_mut().find(|(w, ..)| w == what) {
                    Some(slot) if slot.1 < ulps => *slot = (what.into(), ulps, want, got),
                    Some(_) => {}
                    None => worst.push((what.into(), ulps, want, got)),
                }
                compared += 1;
            }
        }
    }

    if !problems.is_empty() {
        let shown: Vec<&str> = problems.iter().take(10).map(String::as_str).collect();
        panic!(
            "{} divergences (showing {}):\n{}",
            problems.len(),
            shown.len(),
            shown.join("\n")
        );
    }

    worst.sort_by_key(|(_, u, ..)| std::cmp::Reverse(*u));
    for (what, ulps, ..) in &worst {
        println!("  {what:<12} worst {ulps} ulp");
    }

    // 4 seeds x 40 points x 14 entry points.
    assert_eq!(compared, 2240, "the fixture lost samples");

    let fraction = exact as f64 / compared as f64;
    println!(
        "  {exact}/{compared} samples bit-identical ({:.2}%)",
        fraction * 100.0
    );
    assert!(
        fraction >= MIN_EXACT_FRACTION,
        "only {:.1}% of samples are bit-identical — that is an algorithmic divergence, \
         not rounding",
        fraction * 100.0
    );
}
