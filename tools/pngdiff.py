#!/usr/bin/env python3
"""Pixel-diff two PNGs with nothing but the standard library.

No PIL, no numpy, no imagemagick on this box. PNG is simple enough that a
correct decoder for the 8-bit truecolour cases is forty lines, and a real pixel
diff is worth far more here than "the hashes differ".
"""
import sys
import zlib
import struct


def decode(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    pos = 8
    idat = bytearray()
    w = h = depth = ctype = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + length

    assert depth == 8, f"{path}: only 8-bit supported, got {depth}"
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(bytes(idat))

    stride = w * channels
    out = bytearray(h * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ft = raw[p]
        p += 1
        line = bytearray(raw[p : p + stride])
        p += stride
        if ft == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ft == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:  # Average
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ft != 0:
            raise ValueError(f"{path}: bad filter type {ft}")
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return w, h, channels, bytes(out)


def main(a, b):
    wa, ha, ca, pa_ = decode(a)
    wb, hb, cb, pb_ = decode(b)
    if (wa, ha, ca) != (wb, hb, cb):
        print(f"DIFFERENT GEOMETRY: {wa}x{ha}x{ca} vs {wb}x{hb}x{cb}")
        return 2

    # Compare the colour channels only; alpha on an opaque capture is constant.
    colour = min(ca, 3)
    n = wa * ha
    differing = 0
    total = 0
    worst = 0
    for i in range(n):
        base = i * ca
        hit = 0
        for k in range(colour):
            d = abs(pa_[base + k] - pb_[base + k])
            if d:
                hit = max(hit, d)
                total += d
        if hit:
            differing += 1
            worst = max(worst, hit)

    print(f"{wa}x{ha}, {n} pixels")
    print(f"pixels differing : {differing} ({100.0 * differing / n:.4f}%)")
    print(f"worst channel    : {worst}/255")
    print(f"mean abs diff    : {total / (n * colour):.5f}/255")
    return 0 if differing == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
