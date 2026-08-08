# PALETTE.md — the normative palette

What a material may look like. Every art commit is checked against this file, and
the checks in §4 are meant to be run by a script rather than read by a reviewer.

`FORMAT.md` says how to author a block. This says what to author.

---

## 1. The engine arithmetic, which decides everything else

A material is ONE flat colour and five modifiers. There is no texture map. The
rendered value of a cell is, from `crates/yugen-render/src/cells.rs`:

```
pixel = color
      + colorVar * pattern_d * TEX_GAIN      TEX_GAIN = 0.72
      + edge     * EDGE_GAIN[class] * 52/255
      + shimmer  (animated, warm-biased, emitters only)
```

Three consequences follow from that arithmetic and are worth more than any
opinion about colour. All three were computed from the constants, not estimated:

**Luma sigma is exactly `0.72 * colorVar`.** `colorVar` is added equally to R, G
and B, and the Rec.709 weights sum to 1, so a uniform RGB offset moves luma by
exactly that offset. The pattern spans +/-2.579 sigma, so the extremes are
`+/-1.857 * colorVar`. This is why §4's `colorVar >= 12` floor is a *proof* that
`frame_capture`'s `MIN_LUMA_STDDEV = 8.0` is met — 12 * 0.72 = 8.64 — for every
material in isolation, before any between-material contrast exists.

**A material's total internal form is `0.1998 * edge` RGB.** `EDGE_GAIN` runs
from +0.74 (exposed corner) to -0.24 (fully buried), and `(0.74 - -0.24) * 52/255
= 0.1998`. Per class:

| class | what it is | RGB per unit of `edge` |
|---|---|---|
| 4 | top face, side open — an exposed corner | +0.1509 |
| 0 | top face | +0.1264 |
| 5 | one below top, side open | +0.0734 |
| 1 | one below top | +0.0530 |
| 6 | wall face | +0.0387 |
| 7 | buried, side open — a cliff face | +0.0163 |
| 2 | two down | +0.0143 |
| 3 | fully buried | **-0.0489** |

So `edge` is bidirectional: it brightens exposed faces AND darkens buried
interiors. For bulk terrain, which is mostly class 3, raising `edge` is net
DARKENING. For a placed object, which is mostly exposed, it is net brightening.
That asymmetry is why worked things carry high `edge` and bulk things moderate
`edge` — it falls out of the class distribution rather than being a style rule.

**`texture` never changes amplitude.** All eight tiles are normalised to the same
sigma. Texture selects the spatial *structure* of the noise; `colorVar` alone
sets how strong it is. Texture is therefore free to change for legibility.

### Clip-ramping, which the content does not currently use

Because `colorVar` and `shimmer` add equally to R, G and B and then clamp at 255,
a base colour with R near 255 clips red first on positive excursions: it
desaturates toward yellow-white as it brightens and deepens toward red as it
darkens. That is a free hue ramp out of one flat colour, and it is exactly how
hot matter behaves. Every E1 FLAME below is authored to exploit it — base R in
250..255. Note that lava is NOT, in the end: see I1 for why the one emitter that
is also bulk had to be left alone.

---

## 2. Measure chroma, never HSL saturation

**`C = (max(R,G,B) - min(R,G,B)) / 255 * 100`, in 0..100.**

HSL saturation is meaningless at extreme lightness and will actively mislead
here. `snow` at `(236,240,250)` reports HSL-S 58% and is one of the least
colourful materials in the game — its chroma is 5.5. A rule written on HSL-S
flags snow and waves through `copperOre`. Every band in §4 is written against
chroma, and any script checking this file must compute chroma.

---

## 3. What was wrong with the palette this replaces

Measured over all 56 blocks: mean luma 128.7, median 118.7, mean chroma 32.8.

**The bright half of the palette had already spent the range that coloured light
needs to write into.** The lighting solve multiplies for darkness and ADDS for
coloured point light. Ten blocks sat above luma 175 and none of them were light
sources — `snow` 240, `steam` 209, `glass` 206, `acid` 206, `sand` 196 — while
`lava` was 131 and `gemOre` 101. A torch beside snow cannot look orange, because
all three channels saturate and the result is white. The predicted artefact is
hot-spot desaturation around every emitter, and that is what the game showed.

The fix is to LOWER THE BULK, not to raise the lights. Rec.709 weights green at
0.7152 and blue at 0.0722, so an orange physically cannot reach high luma without
turning into cream: `lava` at a fully saturated `(255,110,35)` is luma 141 and
that is near its ceiling *as an orange*. Raising emitters destroys their hue.

**And `edge` was allocated in inverse proportion to screen area.** The materials
covering the most pixels had almost no form — `ash` 8, `sand` 12, `wetSand` 18,
`mud` 20, `snow` 20 — which at `0.1998 * edge` is 1.6 RGB of internal contrast
for ash and 2.4 for sand. A desert or a snowfield was a shadeless field of one
hue with a dither on it. Meanwhile the small, rare, placed objects were rimmed
hardest: `obsidian` 210, `spike` 200, `anvil` 170. Fixing this is cheaper than a
recolour and worth doing on its own.

---

## 4. The bands, and the invariants a script must enforce

Band membership is AUTHORED, not inferred from the numbers. The band list is the
source of truth and the ranges are the assertion — otherwise two bands' ranges
overlap and a block becomes ambiguous. Every block sits in exactly one band.

| band | what it holds | luma | chroma | `edge` | `colorVar` |
|---|---|---|---|---|---|
| B0 | deep bulk | 18..55 | <= 12 | >= 40 | >= 12 |
| B1 | bulk environment | 38..132 | <= 34 | 40..118 | >= 12 |
| B2 | worked / structural | 45..125 | <= 30 | >= 120 | >= 12 |
| B3 | bright bulk — snow, ice, glass, vapour | 115..142 | <= 26 | >= 90 | >= 12 |
| B4 | signal and fluid accent | 70..132 | 40..62 | >= 55 | >= 12 |
| B5 | ore / vein | 24..142 | — | host edge + 24 +/- 8 | >= 22 |
| E1 | primary emissive — flame | 155..220 | >= 45 | >= 0 | >= 12 |
| E1* | `lava` alone — emitter AND bulk, see I1 | ~131 | >= 45 | >= 0 | >= 12 |
| E2 | minor emissive — cold glow | 85..140 | >= 40 | >= 80 | >= 18 |

Gases are exempt from the `edge` floors: they are not solid, so they never
occupy a rim or occlusion class and `edge` does nothing for them.

### The invariants

These are the point of the document. I1 and I5 are the two that fix the game.

- **I1 — the emitters must have somewhere to go.**
  `min(luma over E1 EXCEPT lava) - max(luma over B0..B5) >= 12`.

  **`lava` is exempt, and finding out why cost a blown-out frame.** It is the
  only material that is an emitter AND bulk, so I1 and the area rule in section 7
  — over 20% of a frame must stay dark and low-chroma — point in opposite
  directions on it and nothing else. I1 was written as if every emitter were a
  torch. Measured on the lava-sea frame, raising lava's base from luma 131 to the
  ladder's 159 does not read as brighter lava, it reads as a yellow-white wash:
  distinct colours fall monotonically 25 642 -> 23 962 -> 20 425 as the base
  climbs, and the dominant share doubles.

  The resolution is the one that costs nothing: **lava did not need to change at
  all.** The world around it got 27% darker, so it gained all the relative
  headroom the invariant was asking for without moving. Contrast is a ratio, and
  I1 was measuring an absolute.

  Before the repaint the number was **-108.7** (`snow` 239.9 against `ember`
  131.2). It is now **+19.1** (`snow` 139.1 against `ember` 158.2).
- **I2 — accents must out-colour the environment.**
  `max(chroma over B0..B3) < min(chroma over B4, E1, E2)`.
- **I3 — every block `colorVar >= 12`.** This is what proves the frame's
  `MIN_LUMA_STDDEV = 8.0` floor per material (§1). **Nine** blocks currently
  violate it: `glass` 6, `anvil` 8, `snow` 8, then `lantern`, `furnace`, `spike`,
  both conveyors and `clay` at 10.
- **I4 — every non-gas, non-E1 block `edge >= 40`.** E1 is exempt because a
  flame has no meaningful rim; gases never occupy a rim class at all. **Twelve**
  blocks currently violate it: `ash` 8, `sand` 12, `wetSand` 18, `snow` 20,
  `mud` 20, `rope` 25, `sticky` 25, then `oil`, `dirt`, `vine`, `acid`, `moss`.
- **I5 — an ore must be findable in the dark.**
  `|luma(ore) - luma(host)| >= 28` AND `colorVar(ore) >= colorVar(host) + 6`.
  **THREE of the five ores fail this today, and the worst is not the one anyone
  suspected.** Against `stone` at luma 92.3, `colorVar` 16:

  | ore | luma delta | colorVar delta | |
  |---|---|---|---|
  | `gemOre` | **9.1** | +10 | fails badly — a precious ore all but invisible by luminance |
  | `copperOre` | **20.0** | +8 | fails |
  | `goldOre` | 77.9 | **+4** | fails the texture half |
  | `ironOre` | 38.8 | +8 | passes |
  | `coalOre` | 43.1 | +6 | passes |

  `gemOre` survives today only on hue — it is the one magenta in the world — and
  hue is exactly what the underworld's red cast takes away. That is the deepest
  legibility bug in the palette and nothing in the frame captures would have
  shown it.
- **I6 — the accent list is capped.** At most 16 blocks may carry chroma >= 40,
  and every one must appear in §6.
- **I7 — mean luma over all blocks <= 105, median <= 95.**
- **I8 — E1 chroma is monotone non-increasing in luma.** Hotter reads whiter;
  the top of the hot ramp must not look like a highlighter.

### The one real risk, stated in advance

This palette is ~25% darker in mean base luma, and the deep-chamber frame
measures a solve multiplier of about 0.24 (mean luma 30.9 against a base mean of
128.7). If that multiplier is untouched, rendered stddev falls by roughly the
same 25%: the measured worst case — ore-chamber at 23.0 — lands near 16-17.
Still twice the 8.0 floor, but the margin halves.

**Re-measure after each step. If any frame lands under 12, raise the deep ambient
floor. Do NOT re-brighten the palette**, which reintroduces exactly the problem
this file exists to fix.

---

## 5. The ramps

Three hue poles, so the world reads as one place: COOL at H 205-225 (`rock`,
`ice`, `deep`), WARM at H 25-42 (`earth`, `sand`, `wood`), GREEN at H 85-100
(`flora`).

Chroma rises with luma in the warm ramps — a lit warm surface is more colourful —
and stays near-flat in the cool ones, where lighting adds luminance rather than
colour. That asymmetry is most of what makes a limited palette feel designed.

**`rock`** H 217 — the load-bearing ramp.
`rock.0 #1c1e23` L30 · `rock.1 #2a2d34` L45 · `rock.2 #333841` L56 ·
`rock.3 #434952` L72 · `rock.4 #565d68` L92 · `rock.5 #6b7280` L114

**`deep`** H 263 — the bottom of the world.
`deep.0 #17121f` L20 · `deep.1 #201a2b` L29 · `deep.2 #2b2338` L38 ·
`deep.3 #382e47` L50

**`earth`** H 29-34 — chroma climbs with luma.
`earth.0 #191612` L22 · `earth.1 #251c11` L29 · `earth.2 #34281a` L42 ·
`earth.3 #4a3520` L56 · `earth.4 #6f4823` L78

**`sand`** H 36-42 — deliberately flatter in chroma than `earth`.
`sand.0 #3a3123` L50 · `sand.1 #4e4230` L67 · `sand.2 #695c3e` L93 ·
`sand.3 #8a7853` L121 · `sand.4 #a3906c` L145

`sand.4` is RESERVED and unused, so the trees-and-structures pass has a legal
step above `sand.3` without renegotiating the bands.

**`wood`** H 25-27 — redder than `earth`, so timber never reads as soil.
`wood.0 #241a12` L28 · `wood.1 #342317` L38 · `wood.2 #472f1e` L51 ·
`wood.3 #5e3f28` L68 · `wood.4 #785134` L87

**`ice`** H 205-210.
`ice.0 #2c3742` L54 · `ice.1 #3b4956` L71 · `ice.2 #5d7d94` L120 ·
`ice.3 #6a8ba3` L134 · `ice.4 #808d99` L139

`ice.4` is deliberately LESS chromatic (C 9.8) than `ice.2` and `ice.3` (C 21.6).
Snow-white is a luminance event, not a colour event, and a saturated pale blue
reads as plastic. This is the one non-monotone ramp. Do not "fix" it.

**`flora`** H 85-98.
`flora.0 #1a2410` L32 · `flora.1 #253213` L45 · `flora.2 #31441d` L61 ·
`flora.3 #425a26` L81 · `flora.4 #4a7a2e` L106

Sanctioned off-hue variants, which take no ramp index:
`flora.pine #223a2a` (L52, H 140) · `flora.grey #3f5a34` (L82, H 103)

### Accent families

**`hot`** — every one clip-ramps (base R >= 250). Chroma FALLS as luma rises,
which is physically right and is I8.
`lava #eb6e23` L131 (unchanged, see I1) · `ember #fc8c3e` L158 · `campfire #fa983e` L166 ·
`fire #ffb054` L186 · `torch #ffba60` L194 · `lantern #ffd684` L217

**`glow`** — cold or weak emitters, and precious ore.
`mushroomCap #8f42b4` L91 · `gemOre #c04a86` L103 · `crystal #a077e8` L136 ·
`goldOre #b2872a` L137

**`signal`** — hazard and interactable.
`spike #c8323c` L83 · `conveyor #a87c1e` L127 · `bounce #2f9e52` L129

**`fluid`** — hazardous liquid.
`water #23528f` L76 · `acid #4f9422` L125

---

## 6. The accent allow-list

Chroma >= 40 is permitted ONLY to these. Sixteen of 55 blocks by count, but the
number that matters is area: six are point-sized fixtures, two are veins a few
cells wide, and only `water` and `acid` occupy real volume — which is why both
sit at the bottom of their luma band.

| block | family | why it earns it |
|---|---|---|
| `spike` | signal.red | Kills you. Red is reserved for this and nothing else. |
| `bounce` | signal.green | Changes your movement; must never read as terrain. |
| `conveyorRight`/`Left` | signal.amber | Machinery that moves you. |
| `water` | fluid.blue | Drowns you, and the only material entered deliberately. |
| `acid` | fluid.green | Kills you, and must not be mistaken for water in a dark cave, where hue is the only surviving cue. |
| `goldOre` | glow.gold | Loot. A game that desaturates its reward has desaturated its reward loop. |
| `gemOre` | glow.magenta | Loot, and the only magenta in the world — the one ore that survives the underworld's red cast. |
| `crystal` | glow.violet | Loot and a light source. |
| `mushroomCap` | glow.violet-dim | Interactable and a light source. |
| `fire` `ember` `lava` `torch` `campfire` `lantern` | hot | Light. This is what the whole tonal system exists to make room for. |

### Denied, and why

- **`leavesAutumn`** (chroma 60). An accent that can tile a whole biome canopy is
  not an accent, it is a background. Reduced to chroma 29.8 on `earth.4`. It only
  has to beat other foliage to read as autumn.
- **`copperOre`** (chroma 41). Its legibility problem is LUMA, not chroma — the
  delta was 20 against stone. Spending an accent slot would have masked the real
  fix. Note `gemOre` KEEPS its accent slot for the opposite reason: at a luma
  delta of 9.1 it has no luminance separation at all, so hue is the only thing
  holding it up until the repaint gives it some.
- **`ice` `packedIce` `glass` `snow`**. Their apparent vividness is the HSL-S
  artefact of §2. They are luminance materials and belong in B3.
- **`mushroomStem`**. The cap glows and the stem does not; splitting the pair
  across the accent line is the visual point of the pair.

---

## 7. What does not transfer from hand-authored pixel art

**The rim is hue-blind.** Form here is generated by an 8-class stencil that knows
only "how many cells above me, is a side open". `edge` is the entire silhouette
budget and there is nothing else. A very warm material and a very cool one get an
identical grey-neutral lift, where a hand artist would tint each rim. The
compensation is to keep bulk chroma low so a neutral rim never looks like a
wrong-coloured rim — **the low-chroma bulk is a technical requirement of a
hue-blind rim term, not only a stylistic import.** That is the strongest
argument for this palette if it ever has to be defended.

**The occlusion term hits thin structures.** Class 3 applies to every buried
cell, including the interior of a one-cell-thick pillar, so high-`edge` materials
used as thin structures look hollow. `wood` at edge 120 is right for a trunk and
wrong for a plank, and the fix is a second block, not a tuned constant. Watch
this in the trees-and-structures pass.

**A material can be 60% of the screen and also move.** Sand, water and smoke can
each fill a viewport, and being simulated, their per-cell noise animates as a
function of position. So, with no equivalent in hand-authored practice:

> A material that can occupy more than ~20% of a frame must sit in B0, B1 or B3
> and carry chroma <= 34. High chroma over large moving areas produces crawl and
> colour fatigue that no static mock-up reveals.

`water` at chroma 42.4 is the one uncomfortable entry on the accent list, and is
pinned to the darkest luma in its band for exactly this reason.

---

## 8. The order to land a repaint in

Bands, not content files. Each step is then a coherent visual claim that can be
judged on its own, and `registry_golden`'s diff stays readable.

1. **`edge` and `colorVar` only** — I3 and I4, no colour changes at all. The
   cheapest and largest visual return in this document: it gives ash, sand, snow,
   mud, rope, sticky and wetSand form for the first time.
2. **B0/B1/B2** — the bulk. Re-measure the frame floors here; this is where the
   stddev margin is spent.
3. **B3/B4/B5** — brights, accents, ores. Re-run the ore-chamber capture
   specifically.
4. **E1/E2** — last, because their job is to sit in the headroom the first three
   steps create. Judging them before that headroom exists gives the wrong answer.

   **`lava`'s `colorVar` 22 -> 30 belongs to THIS step and was deliberately left
   out of step 1.** It breaks `shader_matches_cpu`, and understanding why matters
   because the same trap is waiting for every other amplitude increase on an
   animated material. That test asserts the GPU and CPU shimmer paths agree
   EXACTLY for any clock under 1800 s. `colorVar` scales the whole pattern term,
   so a larger amplitude multiplies f32's representation error: a difference that
   sat below half a least-significant bit is pushed above it, and one cell in
   38 183 disagreed by 1. Nothing was wrong with the arithmetic. The test's
   "under half an hour is exact" claim is implicitly conditioned on the
   amplitudes the content carried when it was written, and raising one
   invalidates it honestly.

   Verified by bisection: `edge` 20 -> 44 alone keeps the test green; `colorVar`
   30 alone breaks it. So when step 4 raises it, expect to argue the exactness
   bound rather than to hunt a bug.

Every step: `cargo run -p contentc`, then bless `registry_golden` and read that
diff — it is the human-readable record — then bless `cells_golden` and paste its
report. No step may touch `crates/yugen-render/src/`.

**`emissive` and `lightEmit` are not palette fields and must not move in any of
these steps.** They create light sources, change the lighting solve, and
`the_bloom_threshold_admits_lamps_and_rejects_glints` asserts `goldOre`'s bloom
weight is exactly 0.0. Glow comes from `shimmer` and from the bloom constants,
which live in `crates/yugen-render/src/light.rs` and are a separate commit.

Two inconsistencies found while writing this, neither acted on:

- `goldOre` declares `emissive 0.12` and contributes zero bloom — a light-solve
  slot spent for no visual return.
- `crystal` at `emissive 0.5` genuinely lights a room and is competing with the
  E1 band it was not designed to join.

Both are worth revisiting during the structures pass.
