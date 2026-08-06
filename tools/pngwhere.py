#!/usr/bin/env python3
"""Where do two PNGs differ? A coarse grid, so noise and regressions look different.

A run-to-run difference in this game is atmosphere: stars, weather motes, the
lava shimmer, the sky gradient. Those are scattered and live in specific bands.
A real regression is structural — terrain, sprites, the HUD — and shows up as a
dense contiguous block. The count alone cannot tell the two apart; the map can.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pngdiff import decode  # noqa: E402

ROWS, COLS = 10, 16


def main(a, b):
    wa, ha, ca, pa_ = decode(a)
    wb, hb, cb, pb_ = decode(b)
    assert (wa, ha, ca) == (wb, hb, cb), "geometry differs"

    colour = min(ca, 3)
    grid = [[0] * COLS for _ in range(ROWS)]
    total = 0
    for y in range(ha):
        gy = y * ROWS // ha
        row = grid[gy]
        for x in range(wa):
            base = (y * wa + x) * ca
            for k in range(colour):
                if pa_[base + k] != pb_[base + k]:
                    row[x * COLS // wa] += 1
                    total += 1
                    break

    cell_px = (ha / ROWS) * (wa / COLS)
    print(f"{a.rsplit('/', 1)[-1]}  vs  {b.rsplit('/', 1)[-1]}")
    print(f"{total} differing pixels; each cell below is % of that cell's area\n")
    for gy, row in enumerate(grid):
        band = "".join(
            f"{100.0 * n / cell_px:5.1f}" if n else "    ." for n in row
        )
        print(f"{gy:2}|{band}")
    print()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
