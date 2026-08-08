#!/usr/bin/env python3
"""Check content/blocks/*.toml against content/PALETTE.md's invariants.

    python3 tools/palette_check.py            # report, exit 1 on a violation
    python3 tools/palette_check.py --baseline # report only, always exit 0

The band-independent invariants (I3, I4, I5, I7) run with no configuration and
are the ones that fix the game. The band-dependent ones (I1, I2, I6 and the
per-band ranges) need to know which band each block is in, which is AUTHORED
rather than inferred -- PALETTE.md is explicit that inferring it lets two bands'
ranges overlap and a block become ambiguous. Until BANDS below is filled in by
the repaint, those checks report as SKIPPED rather than passing silently, which
is the difference between "checked" and "not checked yet".

Chroma, never HSL saturation. PALETTE.md 2 has the argument: snow reports
HSL-S 58% at a chroma of 5.5, so an HSL rule flags the least colourful material
in the game and waves through copper ore.
"""

import glob
import os
import statistics
import sys
import tomllib

# Gases never occupy a rim or occlusion class, so `edge` does nothing for them.
# Derived from each block's own `state` rather than a name list -- a hardcoded
# set silently flagged the first gas added after it was written.
def is_gas(rec):
    return rec.get("state") == "gas"
# A flame has no meaningful rim either.
E1 = {"fire", "ember", "lava", "torch", "campfire", "lantern"}
# The rock every ore is embedded in. I5 is measured against it.
ORE_HOST = "stone"
ORES = ["coalOre", "copperOre", "ironOre", "goldOre", "gemOre"]

# block id -> band. Filled in by the repaint; see PALETTE.md 4.
BANDS: dict[str, str] = {}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "content/blocks/*.toml"))):
        with open(path, "rb") as fh:
            for key, rec in tomllib.load(fh).items():
                if isinstance(rec, dict) and "color" in rec:
                    out[key] = rec
    out.pop("empty", None)  # air is never drawn
    return out


def rgb(rec):
    c = rec["color"]
    if isinstance(c, str):
        return [int(c[i : i + 2], 16) for i in (1, 3, 5)]
    return list(c)


def luma(rec):
    r, g, b = rgb(rec)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def chroma(rec):
    c = rgb(rec)
    return (max(c) - min(c)) / 255 * 100


def var(rec):
    return rec.get("colorVar", 12)


def edge(rec):
    return rec.get("edge", 0)


def main():
    baseline = "--baseline" in sys.argv
    blocks = load()
    fails, skips = [], []

    # I3 -- proves MIN_LUMA_STDDEV per material: sigma is exactly 0.72*colorVar.
    bad = sorted(((k, var(v)) for k, v in blocks.items() if var(v) < 12), key=lambda x: x[1])
    if bad:
        fails.append(f"I3 colorVar >= 12: {len(bad)} violations {bad}")

    # I4 -- total internal form is 0.1998*edge, so edge 8 is 1.6 RGB of shape.
    bad = sorted(
        ((k, edge(v)) for k, v in blocks.items() if not is_gas(v) and k not in E1 and edge(v) < 40),
        key=lambda x: x[1],
    )
    if bad:
        fails.append(f"I4 edge >= 40 (non-gas, non-E1): {len(bad)} violations {bad}")

    # I5 -- an ore must be findable in the dark, by luma AND by texture.
    host = blocks[ORE_HOST]
    for ore in ORES:
        if ore not in blocks:
            continue
        dl = abs(luma(blocks[ore]) - luma(host))
        dv = var(blocks[ore]) - var(host)
        if dl < 28:
            fails.append(f"I5 {ore}: luma delta {dl:.1f} < 28 against {ORE_HOST}")
        if dv < 6:
            fails.append(f"I5 {ore}: colorVar delta {dv:+d} < +6 against {ORE_HOST}")

    # I7 -- the palette's centre of mass.
    lumas = [luma(v) for v in blocks.values()]
    mean, median = statistics.mean(lumas), statistics.median(lumas)
    if mean > 105:
        fails.append(f"I7 mean luma {mean:.1f} > 105")
    if median > 95:
        fails.append(f"I7 median luma {median:.1f} > 95")

    if BANDS:
        missing = sorted(set(blocks) - set(BANDS))
        if missing:
            fails.append(f"unbanded blocks: {missing}")
        env = [k for k, b in BANDS.items() if b in ("B0", "B1", "B2", "B3", "B4", "B5")]
        acc = [k for k, b in BANDS.items() if b in ("B4", "E1", "E2")]
        if env and acc:
            gap = min(luma(blocks[k]) for k in BANDS if BANDS[k] == "E1") - max(
                luma(blocks[k]) for k in env
            )
            if gap < 12:
                fails.append(f"I1 emitter headroom {gap:.1f} < 12")
            hi = max(chroma(blocks[k]) for k in BANDS if BANDS[k] in ("B0", "B1", "B2", "B3"))
            lo = min(chroma(blocks[k]) for k in acc)
            if hi >= lo:
                fails.append(f"I2 environment chroma {hi:.1f} >= accent chroma {lo:.1f}")
        if len(acc) > 16:
            fails.append(f"I6 {len(acc)} accents > 16")
    else:
        skips.append("I1, I2, I6 and the per-band ranges: BANDS is empty")

    print(f"{len(blocks)} blocks | mean luma {mean:.1f} | median {median:.1f} "
          f"| mean chroma {statistics.mean([chroma(v) for v in blocks.values()]):.1f}")
    for s in skips:
        print(f"  SKIPPED {s}")
    for f in fails:
        print(f"  FAIL {f}")
    if not fails:
        print("  all checked invariants hold")
    return 1 if fails and not baseline else 0


if __name__ == "__main__":
    sys.exit(main())
