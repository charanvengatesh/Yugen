# tools/

Small standalone scripts. Nothing here is part of the build; nothing here is run
by `cargo xtask check`.

## `pngdiff.py` / `pngwhere.py`

Compare two PNGs. `pngdiff` reports how much moved, `pngwhere` reports where.

```
python3 tools/pngdiff.py  a.png b.png
python3 tools/pngwhere.py a.png b.png
```

Pure standard library — no PIL, no numpy, no imagemagick, none of which is on the
machine this was written on. PNG is simple enough that a correct decoder for the
8-bit truecolour cases is forty lines, and a real pixel diff is worth far more
than "the hashes differ".

### What they are for

`tests/frame_stability.rs` asserts two bounds on how much an idle frame may
change. Those bounds are measured, not guessed, and these are what measured them
— including the region map that showed the movement was the sun's dithered halo
rather than something wrong.

### The rule they exist to enforce

**Never claim two frames are identical from looking at them.** That claim was
made once in this repo's history and was wrong: neither capture rig is bit-stable
by default, `lit_scene` varies by ~16% run to run, and the full binary by 54.8%,
because the world clock advances on wall time.

**Always diff against a same-tree control.** Two runs of the same build first,
then the build you care about. A difference only means something once you know
what zero looks like.

`tests/common::FRAME_DT` pins the frame delta, which takes `frame_capture` to
bit-identical across runs — so inside the test harness the control is now exactly
zero. Outside it, running the real binary, it is not.
