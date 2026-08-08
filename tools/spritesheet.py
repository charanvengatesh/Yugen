#!/usr/bin/env python3
"""Render every authored sprite and mob drawing onto one PNG.

    python3 tools/spritesheet.py [out.png]

One tile per record -- first frame of the first sequence, at the record's own
`grain` -- magnified so one WORLD PX is a fixed number of screen px. That last
choice is the point of the whole tool: two creatures drawn at different grains
occupy the same world rectangle in game, so they are compared here at the same
WORLD scale, not the same texel scale. A finer-grain sprite shows up as a finer
drawing in an equally-sized box, which is exactly how the game shows it.

Reads content/ directly rather than the baked atlases: this reviews what an
artist edits, and it needs no GPU and no build. The bake tests already prove
the pipeline agrees with the content byte for byte.
"""

import glob
import os
import struct
import sys
import tomllib
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = 5          # world px per cell
SCALE = 6         # screen px per world px
BG = (58, 58, 66)
WELL = (40, 40, 48)
PAD = 4           # px between tiles


def records():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "content/sprites/*.toml"))):
        with open(path, "rb") as fh:
            for name, rec in tomllib.load(fh).items():
                if isinstance(rec, dict) and "pal" in rec:
                    out.append((name, rec))
    for path in sorted(glob.glob(os.path.join(ROOT, "content/mobs/*.toml"))):
        with open(path, "rb") as fh:
            for name, rec in tomllib.load(fh).items():
                if isinstance(rec, dict) and isinstance(rec.get("art"), dict):
                    out.append((name, rec["art"]))
    return out


def first_frame(rec):
    seq = rec["seq"][0]
    return seq["frames"].strip("\n").split("\n\n")[0].split("\n")


def rgb(hexs):
    return tuple(int(hexs[i : i + 2], 16) for i in (1, 3, 5))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "target", "spritesheet.png")
    recs = records()

    tiles = []
    for name, rec in recs:
        w_cells, h_cells = rec["cellsW"], rec["cellsH"]
        grain = rec.get("grain", 1)
        pal = rec["pal"]
        frame = first_frame(rec)
        # world-px dimensions of the drawn rect
        wpx, hpx = w_cells * CELL, h_cells * CELL
        # screen px per TEXEL: world px per texel is CELL/grain
        tpx = CELL * SCALE // grain
        tile = [[WELL] * (wpx * SCALE) for _ in range(hpx * SCALE)]
        for ty, row in enumerate(frame):
            for tx, ch in enumerate(row):
                if ch in ".0":
                    continue
                col = rgb(pal[int(ch)])
                for dy in range(tpx):
                    for dx in range(tpx):
                        tile[ty * tpx + dy][tx * tpx + dx] = col
        tiles.append((name, grain, tile))

    # lay out in rows of 10
    per_row = 10
    cell_w = max(len(t[0]) for _, _, t in tiles) + PAD
    cell_h = max(len(t) for _, _, t in tiles) + PAD + 8
    cols = min(per_row, len(tiles))
    rows = (len(tiles) + per_row - 1) // per_row
    W, H = cols * cell_w + PAD, rows * cell_h + PAD
    img = [[BG] * W for _ in range(H)]
    for i, (name, grain, tile) in enumerate(tiles):
        ox = PAD + (i % per_row) * cell_w
        oy = PAD + (i // per_row) * cell_h
        for y, r in enumerate(tile):
            for x, c in enumerate(r):
                img[oy + y][ox + x] = c
        # grain > 1 gets a thin marker line under the tile so the experiment is
        # findable on the sheet without reading names
        if grain > 1:
            for x in range(len(tile[0])):
                img[oy + len(tile) + 2][ox + x] = (255, 200, 80)

    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in img)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(png)
    print(f"{len(tiles)} drawings -> {out_path} ({W}x{H})")


if __name__ == "__main__":
    main()
