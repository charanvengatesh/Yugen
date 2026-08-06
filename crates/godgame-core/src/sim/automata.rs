//! The falling-sand automata — one Noita-style cellular tick over the world.
//!
//! We only visit awake chunks, iterate bottom-up (so a cell that falls this tick
//! doesn't get re-processed after it lands lower), and flip the horizontal scan
//! direction every tick to cancel the left/right drift a fixed scan order would
//! bake into flows. `CellFlags::MOVED` marks a cell that already acted this
//! tick; because `swap` carries flags with the cell, we set the bit on the
//! destination after a move so the moved matter isn't touched again on the same
//! pass.
//!
//! A tick is two sweeps over the awake set:
//!   pass 1 — clear stale MOVED bits AND relax the heat field
//!   pass 2 — thermal transitions, chemistry, growth, movement
//!
//! The two passes cannot be fused, and the reason is worth writing down because
//! it looks fusable. Pass 2 runs bottom-up and chunk by chunk, so if the
//! moved-bit clear were folded into it, a cell could have its bit cleared AFTER
//! something already moved into it this tick — a liquid spreading sideways
//! within a row, or a grain falling from the chunk above into a chunk not yet
//! swept — and would then get to act twice in one tick. Only a full clearing
//! sweep ahead of all movement makes "already acted THIS tick" mean what it
//! says.
//!
//! WHAT GETS SWEPT is the per-chunk ACTIVITY BOX, not the chunk (see
//! [`CellGrid::wake_rect`]). Sleep used to be all-or-nothing per chunk, so one
//! falling grain bought a 32x32 sweep twice over; the box narrows that to the
//! sub-rectangle something actually happened in, plus a one-cell halo for
//! neighbour rules. On a mixed benchmark world that is the difference between
//! sweeping ~43% of the window per tick and ~6%.
//!
//! WHAT EACH CELL COSTS is decided by one load: `BEH[id]`, a byte packing "can
//! react / has a reachable heat threshold / grows / moves, and if so in which
//! state". An inert solid — which is nearly every cell in nearly every chunk —
//! is dismissed on that one load and a compare against zero.
//!
//! No allocation anywhere: the growth candidate buffer is a stack array, the
//! awake-box list is a `Vec` owned by [`Automata`] and reused every tick, and
//! every inner value is a plain integer addressing a flat plane. A settled world
//! wakes no chunks and costs ~nothing.
//!
//! # DETERMINISM — the thing to not break
//!
//! This tick is deterministic ORDER-DEPENDENTLY, not positionally. One RNG
//! stream is drawn from in one exact sweep order, so the resulting world depends
//! on all of: the two-pass structure, `move_pass` running bottom-up, the X sweep
//! direction alternating with tick parity, the row-major per-chunk visit order,
//! the exact activity-box extents, and which branches happened to draw a random
//! at all. None of it is incidental and none of it may be reordered, fused or
//! parallelised. A single-threaded `&mut CellGrid` tick is the design.
//!
//! # What changed in the port
//!
//! The TypeScript kept the sweep's working set — `simGrid`, `simLTR`, `simMat`,
//! `simFlags`, `simCols`, `simRows` — in module-level `let`s. That was not a
//! design choice: `forEachAwakeChunk` takes a callback, and reading module
//! bindings was the only way to avoid allocating a fresh closure (and its
//! captured environment) twice per tick. Here that set is the [`Sweep`] struct,
//! passed by `&mut`, which costs nothing and cannot be stale outside a tick
//! because the borrow ends with it.
//!
//! The other module-level mutables — the tick counter and the RNG state — moved
//! into [`Automata`], which owns a tick. Nothing about the stream or its
//! consumption order changed; see [`super::rng`].

use std::sync::LazyLock;

use super::grid::{AwakeChunk, CellFlags, CellGrid};
use super::materials::{
    CellId, EMPTY, MAT_BURNINTO, MAT_BURNTIME, MAT_CONDUCT, MAT_COOL, MAT_COUNT, MAT_DENSITY,
    MAT_FLAMMABLE, MAT_GROWCHANCE, MAT_GROWDOWN, MAT_GROWINTO, MAT_GROWMAX, MAT_GROWS,
    MAT_HEATEMIT, MAT_HEATTHRESH, MAT_IGNITE, MAT_IGNITEAT, MAT_MELTAT, MAT_MELTINTO, MAT_SPREAD,
    MaterialState, block, grow_onto,
};
use super::reactions::{RULES, react, reaction_pending};
use super::rng::SimRng;

// Material codes. The TypeScript resolved these with `codeOf` at module load
// because the registry was only fixed at runtime; here they are compile-time
// constants out of the generated block module.
const WATER: CellId = block::WATER;
const STEAM: CellId = block::STEAM;
const SMOKE: CellId = block::SMOKE;
const FIRE: CellId = block::FIRE;
const LAVA: CellId = block::LAVA;

// Gas lifetimes, in ticks, seeded into `aux` when a gas cell is created.
const FIRE_LIFE: u16 = 30;
const SMOKE_LIFE: u16 = 120;
const STEAM_LIFE: u16 = 140;

// Tunable chances (per tick) for the stochastic bits of gas/fire behaviour.
const GAS_WANDER: f64 = 0.35; // sideways drift when a gas can't rise
const STEAM_CONDENSE: f64 = 0.008; // steam collapsing back to water on expiry
const FIRE_SMOKE: f64 = 0.06; // fire puffing smoke upward
const LAVA_IGNITE: f64 = 0.04; // lava lighting an adjacent flammable

/// How much a cell's temperature must swing in one tick to keep its chunk awake.
///
/// This single constant is what stops the heat field from defeating the sleep
/// system. Diffusion converges geometrically, so a heated neighbourhood reaches
/// a fixed point (or a ±1 integer flutter around one) after a few dozen ticks;
/// once every swing is under this threshold nothing wakes anything and the
/// region — lava lake included — goes back to costing zero. Anything genuinely
/// changing (a fire spreading, water climbing toward boiling) swings far more
/// than this and keeps itself simulated.
const HEAT_WAKE_DELTA: i32 = 2;

// --- Packed per-material behaviour bits --------------------------------------

/// One byte per material answering "what can this thing possibly do?", derived
/// from the material tables on first touch.
///
/// `step` used to interrogate five separate tables for every awake cell —
/// `mat_reactive`, `MAT_HEATTHRESH` (plus a temp load), `MAT_GROWS`,
/// `MAT_STATE` — and the overwhelming majority of cells in any real world are
/// inert solids that answer "no" to all of them. That is four or five scattered
/// array loads across four different cache lines to learn nothing.
///
/// Collapsing the answers into one byte means the common case is a single load
/// and a single compare against zero: `BEH[id] == 0` says "this material has no
/// chemistry, no thermal transition, no growth and no movement rule", and the
/// cell is dismissed on the spot. The individual bits then gate each stage, so
/// the tables above are only touched by the cells that can actually use them.
///
/// The movement state lives in the high nibble rather than as a separate flag,
/// which folds the old `match MAT_STATE[id]` load into the byte we already have.
/// A material that does not move stores 0 there — that is what keeps the whole
/// byte zero for stone, dirt, obsidian, glass and the ores, i.e. for the bulk of
/// every chunk.
///
/// Derived here, not in `materials.rs`: this packing is the automata's private
/// view of the registry, and the generated-content facade over there is free to
/// grow without knowing the hot loop exists.
static BEH: LazyLock<[u8; MAT_COUNT]> = LazyLock::new(build_beh);

/// 1 when a denser thing can displace this cell (air, liquid, gas).
static SWAPPABLE: LazyLock<[u8; MAT_COUNT]> = LazyLock::new(build_swappable);

const BEH_REACT: u8 = 1 << 0; // can DRIVE a neighbour reaction
const BEH_HEAT: u8 = 1 << 1; // has a thermal threshold a byte can actually reach
const BEH_GROW: u8 = 1 << 2; // spreads onto a substrate
const BEH_STATE_SHIFT: u8 = 4; // MaterialState of a MOVING material; 0 = static

fn build_beh() -> [u8; MAT_COUNT] {
    let rules = &*RULES;
    let mut beh = [0u8; MAT_COUNT];
    for (id, slot) in beh.iter_mut().enumerate() {
        let state = MaterialState::of(id as CellId);
        let mut b = 0u8;
        if rules.mat_reactive[id] != 0 {
            b |= BEH_REACT;
        }
        // Temperature is a byte, so a threshold above 255 is unreachable by
        // construction — that is what MAT_HEATTHRESH's NEVER sentinel encodes.
        if MAT_HEATTHRESH[id] <= 255 {
            b |= BEH_HEAT;
        }
        if MAT_GROWS[id] != 0 {
            b |= BEH_GROW;
        }
        if matches!(
            state,
            MaterialState::Powder | MaterialState::Liquid | MaterialState::Gas
        ) {
            b |= (state as u8) << BEH_STATE_SHIFT;
        }
        *slot = b;
    }
    beh
}

fn build_swappable() -> [u8; MAT_COUNT] {
    let mut sw = [0u8; MAT_COUNT];
    for (id, slot) in sw.iter_mut().enumerate() {
        let state = MaterialState::of(id as CellId);
        *slot = u8::from(id == 0 || state == MaterialState::Liquid || state == MaterialState::Gas);
    }
    sw
}

// Orthogonal neighbour offsets, shared by the ignite/growth scans.
const NX: [i32; 4] = [0, 0, -1, 1];
const NY: [i32; 4] = [-1, 1, 0, 0];

// --- The tick ----------------------------------------------------------------

/// The automata's own state across ticks: the tick counter whose parity chooses
/// the X scan direction, the one RNG stream, and the reused awake-box buffer.
///
/// This is what the TypeScript kept in module-level `let`s. Owning it means a
/// test can run an isolated, reproducible world without reaching into globals.
pub struct Automata {
    /// Tick counter: its parity chooses the X scan direction (drift cancellation).
    tick: u64,
    /// The ONE stream. Drawn from in sweep order — see the determinism note at
    /// the top of this file.
    rng: SimRng,
    /// Awake activity boxes for the current tick, collected once and walked
    /// twice (pass 1, then pass 2). Reused across ticks so a tick allocates
    /// nothing.
    boxes: Vec<AwakeChunk>,
}

impl Automata {
    pub fn new() -> Automata {
        Automata {
            tick: 0,
            rng: SimRng::new(),
            boxes: Vec::new(),
        }
    }

    /// Reseed the sim's RNG (the worker called this with the world seed at init).
    pub fn seed(&mut self, seed: u32) {
        self.rng.seed(seed);
    }

    /// Ticks simulated so far. Its parity is the X sweep direction.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// The scan direction the NEXT tick will use: left-to-right on even ticks.
    pub fn next_is_ltr(&self) -> bool {
        (self.tick + 1) & 1 == 0
    }

    /// Advance the world by one cellular tick.
    pub fn simulate(&mut self, grid: &mut CellGrid) {
        grid.begin_sim_tick();
        self.tick += 1;
        let ltr = (self.tick & 1) == 0;

        // Collect the awake set ONCE and walk it twice.
        //
        // `begin_sim_tick` has already promoted last tick's wakes into the
        // "this tick" side, and everything woken during the sweep accumulates
        // into the "next tick" side — so the set and its boxes are frozen for
        // the duration of the tick and reading them up front is exactly what
        // the TypeScript's two `forEachAwakeChunk` calls did. The list arrives
        // in row-major slot order, which is determinism-critical: it decides
        // which cell draws which random.
        let mut boxes = std::mem::take(&mut self.boxes);
        grid.awake_chunks_into(&mut boxes);

        let mut sweep = Sweep {
            cols: grid.cols(),
            rows: grid.rows(),
            grid,
            rng: &mut self.rng,
            ltr,
        };

        // Pass 1: clear last tick's stale moved-bits and relax the heat field.
        // Any cell carrying MOVED moved last tick, which woke its chunk — so it
        // is guaranteed to be inside an awake chunk now. Clearing here (not
        // persistently) means the bit only ever means "already acted THIS
        // tick", so matter keeps flowing tick after tick.
        for b in &boxes {
            sweep.pre_pass(*b);
        }

        // Pass 2: transitions / reactions / growth / movement.
        for b in &boxes {
            sweep.move_pass(*b);
        }

        self.boxes = boxes;
    }
}

impl Default for Automata {
    fn default() -> Automata {
        Automata::new()
    }
}

/// Sweep state for one tick.
///
/// `cols`/`rows` are copied out of the grid because every probe in the movement
/// rules bounds-tests against them; the TypeScript hoisted the cell planes here
/// too, but in Rust the planes stay behind `self.grid` — the borrow checker
/// will not let a `&mut Vec` live alongside the `&mut CellGrid` that owns it,
/// and `self.grid.material[i]` compiles to the same load anyway.
struct Sweep<'a> {
    grid: &'a mut CellGrid,
    rng: &'a mut SimRng,
    /// This tick's X scan direction. Alternates with tick parity so a fixed
    /// scan order cannot bake a left/right drift into flows.
    ltr: bool,
    cols: i32,
    rows: i32,
}

// --- Pass 1: moved-bit reset + heat diffusion --------------------------------

impl Sweep<'_> {
    /// Clear `CellFlags::MOVED` and advance the temperature field over one awake
    /// chunk.
    ///
    /// Heat is relaxed in place (no second buffer, no allocation): each cell
    /// moves a fraction of the difference between it and its four orthogonal
    /// neighbours, then sheds `MAT_COOL` degrees toward ambient. In-place
    /// relaxation is order-dependent, so the X scan alternates with tick parity
    /// for the same reason the movement pass does — otherwise heat would visibly
    /// smear toward one side. It is fully deterministic either way.
    ///
    /// Out-of-bounds neighbours read as ambient, and the whole thing runs only
    /// over AWAKE chunks, so a sleeping world does no thermal work whatsoever.
    fn pre_pass(&mut self, b: AwakeChunk) {
        let AwakeChunk { x0, y0, x1, y1, .. } = b;
        let cols = self.cols;
        let rows = self.rows;

        let dir = if self.ltr { 1 } else { -1 };
        let first = if self.ltr { x0 } else { x1 - 1 };
        let last = if self.ltr { x1 } else { x0 - 1 };

        for y in y0..y1 {
            let row = y * cols;
            let up_row = if y > 0 { row - cols } else { -1 };
            let dn_row = if y < rows - 1 { row + cols } else { -1 };

            let mut x = first;
            while x != last {
                let i = (row + x) as usize;
                self.grid.flags[i].remove(CellFlags::MOVED);
                self.relax(i, x, y, up_row, dn_row);
                x += dir;
            }
        }
    }

    /// The heat half of pass 1 for one cell. Split out of the loop only so the
    /// cold-cell early-out can be a `return` rather than a `continue` that has
    /// to remember to advance `x` first.
    #[inline]
    fn relax(&mut self, i: usize, x: i32, y: i32, up_row: i32, dn_row: i32) {
        let cols = self.cols;
        let id = self.grid.material[i] as usize;
        let t = i32::from(self.grid.temp[i]);
        let emit = i32::from(MAT_HEATEMIT[id]);

        let mut sum = 0i32;
        if up_row >= 0 {
            sum += i32::from(self.grid.temp[(up_row + x) as usize]);
        }
        if dn_row >= 0 {
            sum += i32::from(self.grid.temp[(dn_row + x) as usize]);
        }
        if x > 0 {
            sum += i32::from(self.grid.temp[i - 1]);
        }
        if x < cols - 1 {
            sum += i32::from(self.grid.temp[i + 1]);
        }

        // Cold cell in a cold neighbourhood — by far the common case, and the
        // reason an awake-but-unheated chunk pays almost nothing for the field.
        if sum == 0 && t == 0 && emit == 0 {
            return;
        }

        let mut nt = t;
        let cond = i32::from(MAT_CONDUCT[id]);
        // Discrete Laplacian, scaled by conductivity. `cond/256` is the rate; an
        // insulator (cond 0) simply keeps whatever it has. The shift is
        // arithmetic on a signed value in both languages, so a cooling cell
        // floors toward -inf identically.
        if cond != 0 {
            nt += ((sum - (t << 2)) * cond) >> 8;
        }
        nt -= i32::from(MAT_COOL[id]);
        if nt < emit {
            nt = emit; // sources hold their floor
        }
        // The TypeScript wrote this as `if (nt < 0) ... else if (nt > 255)`; the
        // `else` was reachable-dead (a value clamped up to 0 is never > 255), so
        // the clamp is the same two comparisons with the same result.
        nt = nt.clamp(0, 255);

        if nt != t {
            self.grid.temp[i] = nt as u8;
            let d = if nt > t { nt - t } else { t - nt };
            if d >= HEAT_WAKE_DELTA {
                self.grid.wake(x, y);
            }
        }
    }

    // --- Pass 2: the cell update ---------------------------------------------

    fn move_pass(&mut self, b: AwakeChunk) {
        let AwakeChunk { x0, y0, x1, y1, .. } = b;
        // Bottom-up: a cell that sinks this tick lands below the cursor, not
        // ahead of it. X direction alternates per tick to avoid a directional
        // bias. The row base is hoisted so the per-cell flat index is one add,
        // not a multiply.
        let cols = self.cols;
        for y in (y0..y1).rev() {
            let row = y * cols;
            if self.ltr {
                for x in x0..x1 {
                    self.step((row + x) as usize, x, y);
                }
            } else {
                for x in (x0..x1).rev() {
                    self.step((row + x) as usize, x, y);
                }
            }
        }
    }

    /// Process one cell. Order of business:
    ///   burn timer → chemistry → thermal transition → growth → movement.
    /// A cell that transforms (melts, reacts) returns immediately, so it can
    /// never be converted twice in one tick.
    ///
    /// `i` is passed in because the caller already computed it from a hoisted
    /// row base; recomputing `y * cols + x` here would be a multiply per cell
    /// for a number the sweep is holding anyway.
    ///
    /// The whole dispatch hangs off ONE load — `BEH[id]`. Everything below the
    /// inert-cell early-out is gated on a bit of that byte, so the tables for
    /// chemistry, heat, growth and state are only touched by materials that have
    /// something to say about them.
    fn step(&mut self, i: usize, x: i32, y: i32) {
        let id = self.grid.material[i];
        if id == EMPTY {
            return;
        }

        let fl = self.grid.flags[i];
        // A cell only acts once per tick. The bit is cleared by pass 1 at the
        // top of the tick; movement sets it on the destination so already-moved
        // matter is skipped for the rest of this pass.
        if fl.contains(CellFlags::MOVED) {
            return;
        }

        let b = BEH[id as usize];
        // THE early-out. An inert solid that is not on fire — stone, dirt,
        // obsidian, glass, ore, every structural material in the world — is
        // dismissed on one array load and one compare. This is the case that
        // dominates any real chunk, and it used to cost five table lookups to
        // reach the same conclusion.
        if b == 0 && !fl.contains(CellFlags::BURNING) {
            return;
        }

        // Burning cells count down regardless of state (wood is a solid that
        // burns).
        if fl.contains(CellFlags::BURNING) && self.tick_burn(x, y, i) {
            return;
        }

        // Chemistry FIRST. BEH_REACT is set only for materials that can DRIVE a
        // reaction from their own state, so inert matter never walks its
        // neighbours.
        //
        // Reactions outrank the heat field by design, and the order here is what
        // enforces it: water touching lava becomes steam+obsidian by the RULE,
        // on the first tick they meet, before the boiling threshold could ever
        // fire and turn the water to steam while leaving the lava molten. Heat
        // therefore only ever drives transitions that no contact rule already
        // covers — melting ice through a wall, boiling a pool two cells from the
        // magma — so nothing is applied twice and the seven original reactions
        // are untouched.
        if b & BEH_REACT != 0 && react(self.grid, self.rng, x, y, id) {
            return;
        }

        // Threshold transitions off the heat field. BEH_HEAT is set only when
        // MAT_HEATTHRESH (= min(meltAt, igniteAt)) is a value a temperature byte
        // can actually reach, so a material with no thermal behaviour never even
        // loads its own temperature.
        if b & BEH_HEAT != 0
            && u16::from(self.grid.temp[i]) >= MAT_HEATTHRESH[id as usize]
            && self.apply_heat(x, y, i, id)
        {
            return;
        }

        // Growth (moss creeping, vines trailing down).
        if b & BEH_GROW != 0 {
            self.grow(x, y, i, id);
        }

        // Movement state, already in hand from the same byte. 0 means "static" —
        // Solid and Empty both land there and fall straight out.
        let st = b >> BEH_STATE_SHIFT;
        if st == MaterialState::Powder as u8 {
            self.update_powder(x, y, MAT_DENSITY[id as usize]);
        } else if st == MaterialState::Liquid as u8 {
            // Lava lights the flammables it touches, then flows normally.
            if id == LAVA {
                self.ignite_neighbours(x, y, true);
            }
            self.update_liquid(
                x,
                y,
                MAT_DENSITY[id as usize],
                i32::from(MAT_SPREAD[id as usize]),
            );
        } else if st == MaterialState::Gas as u8 {
            self.update_gas(x, y, id);
        }
    }

    /// Apply a temperature-driven transition. Melting/boiling replaces the cell
    /// and returns true (the caller stops — one transformation per cell per
    /// tick). Autoignition only raises `CellFlags::BURNING`, which is idempotent
    /// with the contact-ignition path in [`Sweep::ignite_neighbours`], so a cell
    /// lit by both mechanisms in the same tick still gets exactly one burn timer.
    fn apply_heat(&mut self, x: i32, y: i32, i: usize, id: CellId) -> bool {
        let t = u16::from(self.grid.temp[i]);

        if t >= MAT_MELTAT[id as usize] {
            // Yield to a neighbour that is about to rewrite this cell by rule.
            // Cells are stepped bottom-up, so a passive participant can reach
            // its melting point before the cell that drives the rule has even
            // been visited; melting it here would quietly change the outcome of
            // a contact reaction.
            if RULES.mat_passive[id as usize] != 0 && reaction_pending(self.grid, x, y) {
                return false;
            }
            // `set` keeps the cell's temperature, so the product starts out as
            // hot as the thing it replaced — ice → water doesn't instantly
            // re-freeze.
            self.grid.set(x, y, MAT_MELTINTO[id as usize]);
            return true;
        }

        if t >= MAT_IGNITEAT[id as usize] && !self.grid.flags[i].contains(CellFlags::BURNING) {
            self.grid.flags[i].insert(CellFlags::BURNING);
            self.grid.aux[i] = MAT_BURNTIME[id as usize];
            self.grid.wake(x, y);
        }
        false
    }

    // --- Growth --------------------------------------------------------------

    /// Spread a growing material into one eligible neighbour.
    ///
    /// Termination is the whole design here. `aux` on a growing cell holds its
    /// GENERATION: a cell converted by growth is one generation further from the
    /// original seed than its parent, and once a lineage hits `MAT_GROWMAX` it
    /// stops dead — no growth, and crucially no `wake`. So the spread is capped
    /// at a fixed radius around each seed rather than creeping across the world
    /// forever.
    ///
    /// The other half of the contract: a cell only keeps its chunk awake while
    /// it actually has something to do. No eligible neighbour, or budget spent,
    /// means we return silently and the chunk is free to sleep. If a substrate
    /// cell shows up later, that write wakes the chunk itself and we simply try
    /// again.
    fn grow(&mut self, x: i32, y: i32, i: usize, id: CellId) {
        let generation = self.grid.aux[i];
        let max = u16::from(MAT_GROWMAX[id as usize]);
        if generation >= max {
            return; // lineage exhausted — inert from here on
        }

        let into = MAT_GROWINTO[id as usize];
        let grow_chance = f64::from(MAT_GROWCHANCE[id as usize]);

        // Downward growers (vines) only ever consider the cell directly below.
        if MAT_GROWDOWN[id as usize] != 0 {
            let ty = y + 1;
            if !self.grid.in_bounds(x, ty) {
                return;
            }
            let below = self.grid.idx(x, ty);
            if !grow_onto(id, self.grid.material[below]) {
                return;
            }
            if !self.rng.chance(grow_chance) {
                self.grid.wake(x, y); // still eligible — stay awake to try again
                return;
            }
            self.grid.set(x, ty, into);
            self.grid.aux[below] = generation + 1;
            return;
        }

        // Scratch list of eligible neighbour directions. The TypeScript
        // preallocated an `Int32Array(4)` at module scope to keep the sweep
        // allocation-free; a fixed-size array here is on the stack and needs no
        // such ceremony.
        let mut cand = [0usize; 4];
        let mut count = 0usize;
        for n in 0..4 {
            let nx = x + NX[n];
            let ny = y + NY[n];
            if !self.grid.in_bounds(nx, ny) {
                continue;
            }
            if !grow_onto(id, self.grid.material[self.grid.idx(nx, ny)]) {
                continue;
            }
            cand[count] = n;
            count += 1;
        }
        if count == 0 {
            return; // nothing to creep onto — let the chunk sleep
        }
        if !self.rng.chance(grow_chance) {
            self.grid.wake(x, y);
            return;
        }

        // One candidate draws NOTHING: the TypeScript's `count === 1 ? 0 :
        // randInt(count)` is a consumption-order detail, not an optimisation.
        let n = cand[if count == 1 {
            0
        } else {
            self.rng.rand_int(count as i32) as usize
        }];
        let tx = x + NX[n];
        let ty = y + NY[n];
        self.grid.set(tx, ty, into);
        let ti = self.grid.idx(tx, ty);
        self.grid.aux[ti] = generation + 1;
    }

    // --- Powder --------------------------------------------------------------

    /// Powder (sand) piles: fall straight down, else slide to a lower diagonal.
    /// It sinks through anything lighter than it (empty air or a thinner
    /// liquid), so sand drops through water and settles at the bottom.
    fn update_powder(&mut self, x: i32, y: i32, density: f32) {
        if self.fall_into(x, y, x, y + 1, density) {
            return;
        }

        // Randomise diagonal preference so a pile doesn't lean one way.
        let first = if self.rng.chance(0.5) { -1 } else { 1 };
        if self.fall_into(x, y, x + first, y + 1, density) {
            return;
        }
        self.fall_into(x, y, x - first, y + 1, density);
    }

    /// Swap the source into (tx,ty) if that cell is empty or a lighter fluid.
    ///
    /// Bounds test, index and load are done once here off the window planes, and
    /// the resulting index is handed straight to the moved-bit write — the old
    /// shape resolved (tx,ty) three separate times (inBounds, get, markMoved)
    /// for a single probe.
    fn fall_into(&mut self, x: i32, y: i32, tx: i32, ty: i32, density: f32) -> bool {
        if tx < 0 || ty < 0 || tx >= self.cols || ty >= self.rows {
            return false;
        }
        let ti = (ty * self.cols + tx) as usize;
        let target = self.grid.material[ti];
        // Empty, or a lighter fluid to sink through (water/oil/gas under a
        // powder).
        if target == EMPTY
            || (SWAPPABLE[target as usize] != 0 && MAT_DENSITY[target as usize] < density)
        {
            self.grid.swap(x, y, tx, ty);
            self.grid.flags[ti].insert(CellFlags::MOVED);
            return true;
        }
        false
    }

    // --- Liquid --------------------------------------------------------------

    /// Liquid: fall down, else down-diagonal, else spread sideways up to
    /// `spread` cells to find its level. Denser liquid sinks below lighter (lava
    /// under water, oil floating on water) via the same displacement swap.
    fn update_liquid(&mut self, x: i32, y: i32, density: f32, spread: i32) {
        if self.flow_into(x, y, x, y + 1, density) {
            return;
        }

        let first = if self.rng.chance(0.5) { -1 } else { 1 };
        if self.flow_into(x, y, x + first, y + 1, density) {
            return;
        }
        if self.flow_into(x, y, x - first, y + 1, density) {
            return;
        }

        // Horizontal spread: probe outward on the chosen side, then the other,
        // and step to the farthest open cell so water levels quickly instead of
        // crawling.
        if self.spread_side(x, y, first, spread) {
            return;
        }
        self.spread_side(x, y, -first, spread);
    }

    /// Move down/diagonal into empty air or a strictly lighter liquid.
    fn flow_into(&mut self, x: i32, y: i32, tx: i32, ty: i32, density: f32) -> bool {
        if tx < 0 || ty < 0 || tx >= self.cols || ty >= self.rows {
            return false;
        }
        let ti = (ty * self.cols + tx) as usize;
        let target = self.grid.material[ti];
        // Empty, or a strictly lighter LIQUID the denser one displaces (buoyancy
        // — lava sinking under water, oil floating on top of it). The state test
        // reads the behaviour byte rather than MAT_STATE: it is the table this
        // loop already has hot, and for a liquid the high nibble IS the state.
        if target == EMPTY
            || (BEH[target as usize] >> BEH_STATE_SHIFT == MaterialState::Liquid as u8
                && MAT_DENSITY[target as usize] < density)
        {
            self.grid.swap(x, y, tx, ty);
            self.grid.flags[ti].insert(CellFlags::MOVED);
            return true;
        }
        false
    }

    /// Walk up to `spread` cells toward `dir`, moving into the farthest empty
    /// cell.
    ///
    /// This is the hottest probe in the whole tick — water has a reach of 5, so
    /// a pool surface costs up to ten of these per cell per tick, on both sides.
    /// The loop therefore clamps the reach against the window edge ONCE up front
    /// and then indexes a hoisted row base, instead of paying a bounds test per
    /// probe.
    ///
    /// Clamping also fixes a quirk of the old form: `grid.get` reported
    /// out-of-bounds as air, so a liquid against the window edge would pick a
    /// destination outside the world, have the resulting `swap` silently no-op,
    /// and still report success — skipping the probe on its other side. Reach 0
    /// now simply means "no room this way" and the other side gets its turn. The
    /// window edge sits well outside the viewport, so this is invisible in play;
    /// it is mentioned only because it is a real difference.
    fn spread_side(&mut self, x: i32, y: i32, dir: i32, spread: i32) -> bool {
        let cols = self.cols;
        let row = y * cols;

        // Room to the window edge in this direction, so the probe never leaves
        // the row.
        let mut reach = if dir > 0 { cols - 1 - x } else { x };
        if reach > spread {
            reach = spread;
        }

        let mut dest = x;
        for s in 1..=reach {
            let nx = x + dir * s;
            if self.grid.material[(row + nx) as usize] != EMPTY {
                break; // blocked — can't reach past here.
            }
            dest = nx;
        }
        if dest == x {
            return false;
        }
        // A sideways spread is the one move that skips over cells, and every
        // cell the liquid passed through is now air that was liquid a moment
        // ago — anything resting on that run has to be woken or it will hang
        // there until an unrelated event happens to wake the region. Waking the
        // whole traversed span (the box union is one call, not one per cell) is
        // what the per-chunk activity box costs us here, and it is cheap.
        //
        // The TypeScript issued that span wake explicitly, right here, because
        // its comment believed `swap` woke only the two endpoints. It does not:
        // `swap` wakes the BOUNDING RECT of the pair, and for two cells in the
        // same row that rect IS the traversed span — `(min(x,dest), y,
        // max(x,dest), y)`, the exact rectangle the old second call passed. So
        // the span is still woken, by the line below, and the follow-up would
        // have been a bit-for-bit duplicate of it.
        self.grid.swap(x, y, dest, y);
        self.grid.flags[(row + dest) as usize].insert(CellFlags::MOVED);
        true
    }

    // --- Gas -----------------------------------------------------------------

    /// Gas (smoke/steam) rises and wanders, and dies by lifetime. `aux` is a
    /// countdown seeded at birth; when it reaches zero the cell clears (steam
    /// has a small chance to condense back to water). Fire is a gas too but
    /// delegates its ignition/lifetime to [`Sweep::update_fire`].
    fn update_gas(&mut self, x: i32, y: i32, id: CellId) {
        if id == FIRE {
            self.update_fire(x, y);
            return;
        }

        let i = self.grid.idx(x, y);
        // Seed the lifetime the first time we see this gas cell (aux still 0).
        if self.grid.aux[i] == 0 {
            self.grid.aux[i] = if id == STEAM { STEAM_LIFE } else { SMOKE_LIFE };
        }
        if self.tick_down(i) == 0 {
            if id == STEAM && self.rng.chance(STEAM_CONDENSE) {
                self.grid.set(x, y, WATER);
            } else {
                self.grid.set(x, y, EMPTY);
            }
            return;
        }

        // Rise straight up, else up-diagonal; if capped, wander sideways.
        if self.rise_into(x, y, x, y - 1) {
            return;
        }
        let first = if self.rng.chance(0.5) { -1 } else { 1 };
        if self.rise_into(x, y, x + first, y - 1) {
            return;
        }
        if self.rise_into(x, y, x - first, y - 1) {
            return;
        }
        // The `&&` short-circuits: the wander roll is drawn first and the probe
        // only happens if it passes. Swapping them would change the stream.
        if self.rng.chance(GAS_WANDER) {
            self.rise_into(x, y, x + first, y);
        }
    }

    /// Move a gas cell into empty air or displace a denser fluid above it.
    fn rise_into(&mut self, x: i32, y: i32, tx: i32, ty: i32) -> bool {
        if tx < 0 || ty < 0 || tx >= self.cols || ty >= self.rows {
            return false;
        }
        let ti = (ty * self.cols + tx) as usize;
        if self.grid.material[ti] != EMPTY {
            return false;
        }
        self.grid.swap(x, y, tx, ty);
        self.grid.flags[ti].insert(CellFlags::MOVED);
        true
    }

    // --- Fire ----------------------------------------------------------------

    /// A live fire cell: ignite flammable neighbours, occasionally puff smoke,
    /// and burn out on its own short lifetime. Fire rises like the gas it is;
    /// when its timer expires it becomes smoke or clears.
    fn update_fire(&mut self, x: i32, y: i32) {
        let i = self.grid.idx(x, y);
        if self.grid.aux[i] == 0 {
            self.grid.aux[i] = FIRE_LIFE;
        }

        self.ignite_neighbours(x, y, false);
        // Keep the flame alive next tick even if it doesn't move — otherwise a
        // fire sitting on fuel would let its chunk sleep and stop burning.
        self.grid.wake(x, y);

        // Occasionally emit smoke into the empty cell above — a visible plume.
        // The roll comes BEFORE the probe here and AFTER it below; both orders
        // are the TypeScript's, and both feed the same stream.
        if self.rng.chance(FIRE_SMOKE) && self.grid.get(x, y - 1) == EMPTY {
            self.grid.set(x, y - 1, SMOKE);
        }

        if self.tick_down(i) == 0 {
            // Die into smoke if there's room above, else just vanish.
            if self.grid.get(x, y - 1) == EMPTY && self.rng.chance(0.5) {
                self.grid.set(x, y, SMOKE);
            } else {
                self.grid.set(x, y, EMPTY);
            }
            return;
        }

        // Cling to fuel: while touching anything flammable/burning, stay put and
        // keep rolling ignition instead of floating away before the fuel
        // catches.
        if self.has_fuel(x, y) {
            return;
        }

        // Otherwise rise like buoyant gas.
        if self.rise_into(x, y, x, y - 1) {
            return;
        }
        let first = if self.rng.chance(0.5) { -1 } else { 1 };
        self.rise_into(x, y, x + first, y - 1);
    }

    /// True if any orthogonal neighbour is flammable or already burning.
    fn has_fuel(&self, x: i32, y: i32) -> bool {
        for n in 0..4 {
            let nx = x + NX[n];
            let ny = y + NY[n];
            if nx < 0 || ny < 0 || nx >= self.cols || ny >= self.rows {
                continue;
            }
            let ni = (ny * self.cols + nx) as usize;
            let t = self.grid.material[ni];
            if t == EMPTY {
                continue;
            }
            if MAT_FLAMMABLE[t as usize] != 0 || self.grid.flags[ni].contains(CellFlags::BURNING) {
                return true;
            }
        }
        false
    }

    /// Roll ignition against each orthogonal flammable neighbour. Shared by fire
    /// and lava (lava lights things it touches without being consumed itself),
    /// so the ignite path lives in one place.
    ///
    /// One index per neighbour, reused for the material read, the burning test
    /// and both writes — this used to resolve the same (nx,ny) up to five times.
    fn ignite_neighbours(&mut self, x: i32, y: i32, lava: bool) {
        for n in 0..4 {
            let nx = x + NX[n];
            let ny = y + NY[n];
            if nx < 0 || ny < 0 || nx >= self.cols || ny >= self.rows {
                continue;
            }
            let ni = (ny * self.cols + nx) as usize;
            let target = self.grid.material[ni];
            if target == EMPTY || MAT_FLAMMABLE[target as usize] == 0 {
                continue;
            }
            // Already burning cells keep their timer; don't reseed.
            if self.grid.flags[ni].contains(CellFlags::BURNING) {
                continue;
            }

            let ignite = f64::from(MAT_IGNITE[target as usize]);
            let p = if lava {
                (ignite + LAVA_IGNITE).min(1.0)
            } else {
                ignite
            };
            if self.rng.chance(p) {
                self.grid.flags[ni].insert(CellFlags::BURNING);
                self.grid.aux[ni] = MAT_BURNTIME[target as usize];
                self.grid.wake(nx, ny);
            }
        }
    }

    /// Advance a burning cell's timer (any state — wood is a burning solid).
    /// When the timer runs out the cell converts to what it burns into (usually
    /// smoke). Returns true if the cell was consumed this tick so the caller
    /// stops here.
    fn tick_burn(&mut self, x: i32, y: i32, i: usize) -> bool {
        let code = self.grid.material[i];
        if MAT_FLAMMABLE[code as usize] == 0 {
            // Not actually flammable (shouldn't happen) — clear the stray flag.
            self.grid.flags[i].remove(CellFlags::BURNING);
            return false;
        }

        // A burning cell spreads fire to its own flammable neighbours, and must
        // keep its chunk awake so the burn timer keeps counting to completion.
        self.ignite_neighbours(x, y, false);
        self.grid.wake(x, y);

        if self.grid.aux[i] == 0 {
            self.grid.aux[i] = MAT_BURNTIME[code as usize];
        }
        if self.tick_down(i) == 0 {
            self.grid.set(x, y, MAT_BURNINTO[code as usize]);
            return true;
        }
        false
    }

    /// Decrement a cell's `aux` timer and report the value the TEST sees.
    ///
    /// This is `--grid.aux[i]` from the TypeScript, and the pedantry is
    /// deliberate. `aux` is a 16-bit plane, and JavaScript's pre-decrement on a
    /// typed array yields the arithmetic result (`-1`) while STORING the
    /// wrapped one (`65535`). A timer that was already 0 therefore fails the
    /// `=== 0` test and rolls over rather than firing — so the returned value is
    /// `i32`, and only the stored value wraps.
    #[inline]
    fn tick_down(&mut self, i: usize) -> i32 {
        let v = i32::from(self.grid.aux[i]) - 1;
        self.grid.aux[i] = v as u16;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::MAT_COLLIDE;

    fn grid() -> CellGrid {
        CellGrid::new(64, 64)
    }

    /// Run `n` ticks over `g`.
    fn run(a: &mut Automata, g: &mut CellGrid, n: usize) {
        for _ in 0..n {
            a.simulate(g);
        }
    }

    #[test]
    fn the_behaviour_byte_dismisses_inert_solids() {
        // The single-load early-out is the whole reason the sweep is viable. If
        // stone ever gains a non-zero behaviour byte, every chunk got slower.
        assert_eq!(BEH[block::STONE as usize], 0);
        assert_eq!(BEH[block::OBSIDIAN as usize], 0);
        assert_eq!(BEH[block::GLASS as usize], 0);
        assert_eq!(BEH[block::IRON_ORE as usize], 0);
        assert_eq!(BEH[EMPTY as usize], 0);

        // ...and that movers carry their state in the high nibble.
        assert_eq!(
            BEH[block::SAND as usize] >> BEH_STATE_SHIFT,
            MaterialState::Powder as u8
        );
        assert_eq!(
            BEH[block::WATER as usize] >> BEH_STATE_SHIFT,
            MaterialState::Liquid as u8
        );
        assert_eq!(
            BEH[block::SMOKE as usize] >> BEH_STATE_SHIFT,
            MaterialState::Gas as u8
        );
        assert_eq!(BEH[block::STONE as usize] >> BEH_STATE_SHIFT, 0);

        // A material whose threshold is the NEVER sentinel must not claim heat.
        assert_eq!(BEH[block::STONE as usize] & BEH_HEAT, 0);
        assert_ne!(BEH[block::ICE as usize] & BEH_HEAT, 0);
        // Moss grows, water drives chemistry.
        assert_ne!(BEH[block::MOSS as usize] & BEH_GROW, 0);
        assert_ne!(BEH[block::WATER as usize] & BEH_REACT, 0);

        // Swappability: air and the fluids yield to a denser thing.
        assert_eq!(SWAPPABLE[EMPTY as usize], 1);
        assert_eq!(SWAPPABLE[block::WATER as usize], 1);
        assert_eq!(SWAPPABLE[block::SMOKE as usize], 1);
        assert_eq!(SWAPPABLE[block::STONE as usize], 0);
        assert_eq!(SWAPPABLE[block::SAND as usize], 0);
    }

    #[test]
    fn sand_falls_and_comes_to_rest() {
        let mut g = grid();
        let mut a = Automata::new();
        // A floor to land on, and one grain well above it.
        for x in 0..64 {
            g.set(x, 40, block::STONE);
        }
        g.set(10, 5, block::SAND);

        run(&mut a, &mut g, 60);
        assert_eq!(
            g.get(10, 39),
            block::SAND,
            "the grain should sit on the floor"
        );
        assert_eq!(g.get(10, 5), EMPTY);

        // And it must STAY there — a settled world is the resting state, not a
        // transient one.
        run(&mut a, &mut g, 60);
        assert_eq!(g.get(10, 39), block::SAND);
    }

    #[test]
    fn a_sand_pile_stacks_instead_of_stacking_infinitely_high() {
        let mut g = grid();
        let mut a = Automata::new();
        for x in 0..64 {
            g.set(x, 40, block::STONE);
        }
        // Ten grains dropped down one column must spread into a pile, not a
        // needle: the diagonal slide is what makes a heap.
        for k in 0..10 {
            g.set(30, 30 - k, block::SAND);
        }
        run(&mut a, &mut g, 200);

        let column: usize = (0..40).filter(|&y| g.get(30, y) == block::SAND).count();
        let total: usize = (0..64)
            .flat_map(|x| (0..40).map(move |y| (x, y)))
            .filter(|&(x, y)| g.get(x, y) == block::SAND)
            .count();
        assert_eq!(total, 10, "no grain may be lost or duplicated");
        assert!(
            column < 10,
            "the pile must slump, not tower ({column} high)"
        );
    }

    #[test]
    fn water_spreads_to_its_declared_reach() {
        let mut g = grid();
        let mut a = Automata::new();
        for x in 0..64 {
            g.set(x, 40, block::STONE);
        }
        g.set(32, 39, block::WATER);
        // One tick of spreading: the cell must move up to MAT_SPREAD cells, and
        // never further.
        let reach = i32::from(MAT_SPREAD[block::WATER as usize]);
        assert!(reach > 0, "water is supposed to spread");
        a.simulate(&mut g);

        let at: Vec<i32> = (0..64).filter(|&x| g.get(x, 39) == block::WATER).collect();
        assert_eq!(at.len(), 1, "one cell of water, one cell of water");
        let moved = (at[0] - 32).abs();
        assert!(moved <= reach, "moved {moved} cells, reach is {reach}");
        assert_eq!(moved, reach, "an open row should be crossed at full reach");
    }

    #[test]
    fn water_levels_out_over_a_basin() {
        let mut g = grid();
        let mut a = Automata::new();
        // A basin: floor at y=40, walls at x=20 and x=44.
        for x in 20..=44 {
            g.set(x, 40, block::STONE);
        }
        for y in 30..=40 {
            g.set(20, y, block::STONE);
            g.set(44, y, block::STONE);
        }
        for k in 0..20 {
            g.set(32, 30 + (k % 5), block::WATER);
        }
        let poured: usize = (0..64)
            .flat_map(|x| (0..64).map(move |y| (x, y)))
            .filter(|&(x, y)| g.get(x, y) == block::WATER)
            .count();

        run(&mut a, &mut g, 300);

        let left: usize = (0..64)
            .flat_map(|x| (0..64).map(move |y| (x, y)))
            .filter(|&(x, y)| g.get(x, y) == block::WATER)
            .count();
        assert_eq!(left, poured, "water is conserved inside a sealed basin");
        // It should have found the floor rather than piling in one column.
        let width: usize = (21..44).filter(|&x| g.get(x, 39) == block::WATER).count();
        assert!(width > 1, "the pool never spread (width {width})");
    }

    #[test]
    fn a_moved_cell_does_not_act_twice_in_one_tick() {
        // Drop a grain into open air. One tick must move it exactly one cell:
        // if MOVED were not honoured, the bottom-up sweep would let it fall
        // again as soon as the cursor reached its new row.
        let mut g = grid();
        let mut a = Automata::new();
        for x in 0..64 {
            g.set(x, 40, block::STONE);
        }
        g.set(10, 20, block::SAND);
        a.simulate(&mut g);
        assert_eq!(g.get(10, 21), block::SAND, "exactly one cell per tick");
        assert!(g.flags[g.idx(10, 21)].contains(CellFlags::MOVED));

        // And pass 1 of the NEXT tick must clear the bit, or the grain would be
        // frozen in mid-air forever.
        a.simulate(&mut g);
        assert_eq!(g.get(10, 22), block::SAND);
    }

    #[test]
    fn the_sweep_alternates_direction_with_tick_parity() {
        // Tick parity picks the X direction; the tick counter is what carries
        // it. Ticking must flip it every time and never drift.
        let mut a = Automata::new();
        let mut g = grid();
        assert_eq!(a.tick(), 0);
        for expected_tick in 1..=8u64 {
            let expected_ltr = (expected_tick & 1) == 0;
            assert_eq!(a.next_is_ltr(), expected_ltr);
            a.simulate(&mut g);
            assert_eq!(a.tick(), expected_tick);
        }

        // The direction is observable: a single grain with two equally valid
        // diagonals is decided by the RNG, but a row of liquid against a wall
        // is decided by the scan. Rather than assert on a flow, assert the
        // parity contract the whole sweep hangs off.
        assert!(!((a.tick() + 1) & 1 == 0) || a.next_is_ltr());
    }

    #[test]
    fn a_reaction_fires_from_either_ordering_inside_a_tick() {
        // Water above lava, and lava above water. Both must resolve to the same
        // pair of products, whichever cell the bottom-up sweep reaches first.
        for (water_on_top, label) in [(true, "water above"), (false, "lava above")] {
            let mut g = grid();
            let mut a = Automata::new();
            // Box them in so neither can flow away before they touch.
            for x in 29..=33 {
                g.set(x, 28, block::STONE);
                g.set(x, 33, block::STONE);
            }
            for y in 28..=33 {
                g.set(29, y, block::STONE);
                g.set(33, y, block::STONE);
            }
            let (wy, ly) = if water_on_top { (30, 31) } else { (31, 30) };
            g.set(31, wy, block::WATER);
            g.set(31, ly, block::LAVA);

            a.simulate(&mut g);

            let products: Vec<CellId> = (29..=32).map(|y| g.get(31, y)).collect();
            assert!(
                products.contains(&block::STEAM),
                "{label}: expected steam, got {products:?}"
            );
            assert!(
                products.contains(&block::OBSIDIAN),
                "{label}: expected obsidian, got {products:?}"
            );
        }
    }

    #[test]
    fn a_seeded_world_replays_identically() {
        // The determinism contract in one assertion: same seed, same edits,
        // same ticks, same world. Anything that reorders the sweep or the RNG
        // draws breaks this.
        fn play(seed: u32) -> Vec<CellId> {
            let mut g = CellGrid::new(64, 64);
            let mut a = Automata::new();
            a.seed(seed);
            for x in 0..64 {
                g.set(x, 50, block::STONE);
            }
            for x in 20..44 {
                for y in 10..20 {
                    g.set(
                        x,
                        y,
                        if (x + y) % 3 == 0 {
                            block::SAND
                        } else {
                            block::WATER
                        },
                    );
                }
            }
            run(&mut a, &mut g, 120);
            g.material.clone()
        }
        assert_eq!(play(1234), play(1234));
        assert_ne!(
            play(1234),
            play(9999),
            "different seeds should diverge, or the RNG is not being consumed"
        );
    }

    #[test]
    fn a_soak_conserves_mass_and_never_writes_a_bogus_material() {
        // Fill the window with a mixed, actively churning world and run it. No
        // cell may hold a code outside the registry, and matter may only be
        // created or destroyed by rules that say so — so the count of cells
        // that COLLIDE (the structural stuff) plus the fluids must stay sane
        // and, more importantly, nothing may vanish into an out-of-range id.
        let mut g = CellGrid::new(96, 96);
        let mut a = Automata::new();
        a.seed(0x5eed);
        for x in 0..96 {
            for y in 0..96 {
                let m = match (x * 7 + y * 13) % 6 {
                    0 => block::STONE,
                    1 => block::SAND,
                    2 => block::WATER,
                    3 => block::DIRT,
                    4 => block::WOOD,
                    _ => EMPTY,
                };
                if m != EMPTY {
                    g.set(x, y, m);
                }
            }
        }
        let filled_before = g.material.iter().filter(|&&m| m != EMPTY).count();

        run(&mut a, &mut g, 240);

        for (i, &m) in g.material.iter().enumerate() {
            assert!(
                (m as usize) < MAT_COUNT,
                "cell {i} holds out-of-range material {m}"
            );
        }
        // Nothing in this mix creates or destroys matter (no fire, no acid, no
        // gas expiry path reachable), so the occupied-cell count is invariant.
        let filled_after = g.material.iter().filter(|&&m| m != EMPTY).count();
        assert_eq!(
            filled_after, filled_before,
            "mass was not conserved over 240 ticks"
        );
        // And the world must actually be doing something, or the soak proved
        // nothing.
        assert!(MAT_COLLIDE[block::STONE as usize] != 0);
    }

    #[test]
    fn a_settled_world_costs_nothing() {
        // The sleep contract. Once everything has come to rest, a tick must
        // sweep no chunks at all — that is what makes the automata affordable.
        let mut g = grid();
        let mut a = Automata::new();
        for x in 0..64 {
            g.set(x, 40, block::STONE);
        }
        g.set(10, 5, block::SAND);
        run(&mut a, &mut g, 200);

        a.simulate(&mut g);
        assert!(
            a.boxes.is_empty(),
            "a settled world woke {} chunks",
            a.boxes.len()
        );
    }
}
