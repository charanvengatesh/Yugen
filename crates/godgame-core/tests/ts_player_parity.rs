//! Player parity: the Rust `Player` must move the way the TypeScript one did.
//!
//! `tests/ts-player.json` was produced by `tools/dump-player-fixture.mjs` in the
//! TypeScript repository: `src/entities/Player.ts` imported UNMODIFIED, bundled
//! with esbuild, run under node against a 288x224 `CellGrid` window filled by
//! the TypeScript worldgen at seed 2334 and then stamped with an authored arena
//! — ice, a pool, a shallow puddle, a one-way platform, a ladder threaded
//! through one, a one-cell step-up ledge, a sheer wall, conveyor / sticky /
//! bounce surfaces and a lava pit. 22 scripted cases, 4 048 fixed steps. Like
//! `ts-noise.json` and `ts-worldgen.json` it is a frozen artefact of the
//! original implementation and is never regenerated from the Rust side.
//!
//! Floats in the fixture are IEEE754 DOUBLE bit patterns, never decimals, for
//! the reason `ts_noise_parity` gives: a decimal round-trip only preserves a
//! value if both parsers are correctly rounded to the last bit, and
//! `serde_json`'s is not.
//!
//! # What this proves that the other suites cannot
//!
//! `ts_worldgen_parity` proves the two builds agree about the WORLD.
//! `physics::collision`'s own tests prove a box is pushed out of a wall
//! correctly. Neither notices that coyote time is read one phase too late, that
//! `wall_dir` is cleared after the collide instead of before, that the jump cut
//! is applied to a velocity that has already been clamped, or that the liquid
//! drag has quietly gone back to being a per-step multiplier. Those are all
//! ORDER, and order is only visible in a long replay. `Player::step` is a pure
//! function of `(grid, state, intent)`, so a scripted intent stream is a
//! complete specification of it.
//!
//! # The two tiers, and the widths behind them
//!
//! TypeScript has one number type and ran every value here as `f64`. This port
//! is `f32` throughout, because `crate::config` is `f32` and because
//! `physics::collision` takes and returns `f32`. Bit equality is therefore
//! impossible by construction, and widening the player to chase it would have
//! made it the one `f64` consumer of an `f32` config. So:
//!
//! - **Continuous quantities are bounded.** Position, velocity and the derived
//!   scalars are compared against the `f64` reference with the budgets in
//!   [`budget`]. The measured worst case over all 4 048 steps is 4.8e-4 px of
//!   position and 1.5e-4 px/s of velocity — a ten-thousandth of a 5 px cell.
//!   The table is printed on every run, passing or not.
//! - **Discrete quantities are exact, with a frozen exception list.** Every
//!   flag, the animation state, the air-jump count, the swing id and the
//!   per-step event list are compared for equality on all 4 048 steps, and the
//!   46 comparisons that disagree are enumerated in [`KNOWN_TIES`] with a cause
//!   apiece. Anything else is a failure, and an entry that stops disagreeing is
//!   a failure too — the list cannot silently rot in either direction.
//!
//! `on_ground`, `on_ice`, `in_liquid`, `touching_liquid`, `sticky_this_step`,
//! `climbing`, `dashing`, `punching`, `dead`, `facing`, `air_jumps` and
//! `swing_id` agree on every single step. The collision resolver snaps a blocked
//! body to an exact cell face, which discards the accumulated rounding on the
//! axis that hit — so the state that comes from GEOMETRY cannot drift, and only
//! the state that comes from a free-running TIMER can.
//!
//! Do not relax either tier to make a change pass. If this fails, the port is
//! wrong, not the test.

use godgame_core::config::{CHUNK_CELLS, STEP_DT};
use godgame_core::entities::{AnimState, Loadout, NoProjectiles, Player, PlayerEvent};
use godgame_core::input::Intent;
use godgame_core::sim::coords::WorldCell;
use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::materials::CellId;
use godgame_core::sim::worldgen::{SpawnPoint, generate_chunk};
use serde_json::Value as J;

fn fixture() -> J {
    serde_json::from_str(include_str!("ts-player.json")).expect("ts-player.json is not valid JSON")
}

/// Decode one hex-encoded IEEE754 double. See the module header.
fn f(v: &J) -> f64 {
    f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
}

fn floats(v: &J) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

fn ints(v: &J) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// FNV-1a, 32 bit, over the little-endian bytes of the cell codes. The same
/// three lines the fixture used, for the reason `ts_worldgen_parity` gives: a
/// checksum whose implementation could itself differ between the two languages
/// would turn a parity failure into a debugging session about the checksum.
fn fnv(cells: &[CellId]) -> String {
    let mut h: u32 = 0x811c_9dc5;
    for &v in cells {
        h = (h ^ u32::from(v & 0xff)).wrapping_mul(0x0100_0193);
        h = (h ^ u32::from(v >> 8)).wrapping_mul(0x0100_0193);
    }
    format!("{h:08x}")
}

// ---------------------------------------------------------------------------
// The world
// ---------------------------------------------------------------------------

/// Rebuild the fixture's window: the real generator over the same chunk range,
/// then the arena edits replayed in the order the fixture recorded them.
///
/// The hash check at the end is what keeps a GRID disagreement from being
/// misread as a PLAYER disagreement — two failures that look identical from a
/// diff of positions and have nothing to do with each other.
fn build_grid(meta: &J, edits: &J) -> CellGrid {
    let g = &meta["grid"];
    let cx0 = g["cx0"].as_i64().unwrap() as i32;
    let cy0 = g["cy0"].as_i64().unwrap() as i32;
    let chunk_cols = g["chunkCols"].as_i64().unwrap() as i32;
    let chunk_rows = g["chunkRows"].as_i64().unwrap() as i32;
    let cols = g["cols"].as_i64().unwrap() as i32;
    let rows = g["rows"].as_i64().unwrap() as i32;
    let seed = meta["seed"].as_u64().unwrap() as u32;

    let mut grid = CellGrid::new(cols, rows);
    grid.set_origin(cx0 * CHUNK_CELLS, cy0 * CHUNK_CELLS);
    assert_eq!(
        grid.origin_cell_x(),
        g["originCellX"].as_i64().unwrap() as i32
    );
    assert_eq!(
        grid.origin_cell_y(),
        g["originCellY"].as_i64().unwrap() as i32
    );

    for j in 0..chunk_rows {
        for i in 0..chunk_cols {
            let chunk = generate_chunk(cx0 + i, cy0 + j, seed);
            let (base_x, base_y) = (i * CHUNK_CELLS, j * CHUNK_CELLS);
            for ly in 0..CHUNK_CELLS {
                for lx in 0..CHUNK_CELLS {
                    let idx = grid.idx(base_x + lx, base_y + ly);
                    grid.material[idx] = chunk[(ly * CHUNK_CELLS + lx) as usize];
                }
            }
        }
    }

    for e in edits.as_array().unwrap() {
        let e = e.as_array().unwrap();
        grid.set_world(
            WorldCell::new(e[0].as_i64().unwrap() as i32, e[1].as_i64().unwrap() as i32),
            e[2].as_i64().unwrap() as CellId,
        );
    }

    assert_eq!(
        fnv(&grid.material),
        g["materialHash"].as_str().unwrap(),
        "the replayed window is not the one the fixture was recorded against — \
         that is a worldgen or edit-replay difference, not a player difference"
    );
    grid
}

// ---------------------------------------------------------------------------
// Decoding the recorded state
// ---------------------------------------------------------------------------

// Intent bits, mirroring `meta.intentBits`.
const I_JUMP_HELD: i64 = 1;
const I_JUMP_QUEUED: i64 = 2;
const I_DASH_QUEUED: i64 = 4;
const I_PUNCH_QUEUED: i64 = 8;
const I_PUNCH_HELD: i64 = 16;
const I_DOWN: i64 = 32;
const I_UP: i64 = 64;

// Flag bits, mirroring `meta.flagBits`.
const F_ON_GROUND: i64 = 1;
const F_ON_ICE: i64 = 2;
const F_IN_LIQUID: i64 = 4;
const F_TOUCHING_LIQUID: i64 = 8;
const F_STICKY: i64 = 16;
const F_CLIMBING: i64 = 32;
const F_DASHING: i64 = 64;
const F_DASH_READY: i64 = 128;
const F_PUNCHING: i64 = 256;
const F_DEAD: i64 = 512;
const F_FACING_LEFT: i64 = 1024;

/// The fixture carries `dir_x` as an integer in {-1, 0, 1} and the rest as a
/// bitfield. Rebuilding the [`Intent`] from those numbers, rather than
/// re-deriving it from a script written twice, is what makes "the identical
/// intent sequence" a fact instead of a claim.
fn intent_of(dir: i64, bits: i64) -> Intent {
    Intent {
        dir_x: dir as f32,
        jump_held: bits & I_JUMP_HELD != 0,
        jump_queued: bits & I_JUMP_QUEUED != 0,
        dash_queued: bits & I_DASH_QUEUED != 0,
        punch_queued: bits & I_PUNCH_QUEUED != 0,
        punch_held: bits & I_PUNCH_HELD != 0,
        down: bits & I_DOWN != 0,
        up: bits & I_UP != 0,
        aim_x: 0.0,
        aim_y: 0.0,
    }
}

fn flags_of(p: &Player) -> i64 {
    let mut f = 0;
    if p.on_ground {
        f |= F_ON_GROUND;
    }
    if p.on_ice {
        f |= F_ON_ICE;
    }
    if p.in_liquid {
        f |= F_IN_LIQUID;
    }
    if p.touching_liquid {
        f |= F_TOUCHING_LIQUID;
    }
    if p.sticky_this_step {
        f |= F_STICKY;
    }
    if p.climbing {
        f |= F_CLIMBING;
    }
    if p.dashing() {
        f |= F_DASHING;
    }
    if p.dash_ready() {
        f |= F_DASH_READY;
    }
    if p.punching() {
        f |= F_PUNCHING;
    }
    if p.dead() {
        f |= F_DEAD;
    }
    if p.facing < 0.0 {
        f |= F_FACING_LEFT;
    }
    f
}

/// Index into `meta.animStates`. The FIXTURE's order, which is deliberately not
/// assumed to be the enum's: the fixture is frozen and the enum is not.
fn anim_index(a: AnimState) -> i64 {
    match a {
        AnimState::Idle => 0,
        AnimState::Run => 1,
        AnimState::Skid => 2,
        AnimState::Jump => 3,
        AnimState::DoubleJump => 4,
        AnimState::Fall => 5,
        AnimState::Land => 6,
        AnimState::Dash => 7,
        AnimState::WallSlide => 8,
        AnimState::Swim => 9,
        AnimState::Punch => 10,
        AnimState::Hurt => 11,
    }
}

/// Index into `meta.events`.
fn event_index(e: PlayerEvent) -> i64 {
    match e {
        PlayerEvent::Land => 0,
        PlayerEvent::Jump => 1,
        PlayerEvent::DoubleJump => 2,
        PlayerEvent::Dash => 3,
        PlayerEvent::WallJump => 4,
        PlayerEvent::Splash => 5,
        PlayerEvent::Step => 6,
        PlayerEvent::Hurt => 7,
    }
}

// ---------------------------------------------------------------------------
// The continuous budgets
// ---------------------------------------------------------------------------

/// How far each continuous quantity may sit from the `f64` reference, in
/// absolute units (px, px/s, or a bare fraction).
///
/// These are the measured worst case over all 4 048 steps rounded up to a round
/// number, with roughly a decade of headroom, and the test prints the measured
/// value on every run so the next person can see how much is left. The whole
/// budget for a POSITION is a thousandth of a 5 px cell — three orders of
/// magnitude below anything that could move a collision decision, which is
/// exactly why the discrete tier can afford to be exact.
fn budget(what: &str) -> f64 {
    match what {
        // Bare fractions of one, each accumulated by a single multiply-add per
        // step, so they get the tightest budgets.
        "submersion" | "run_phase" => 1e-5,
        // Positions. An f32 near 1 000 px has an ULP of ~6e-5, so a handful of
        // steps' worth of rounding is the floor here whatever we do.
        "x" | "y" | "step_up_visual" => 5e-3,
        // Velocities, and the health an integrated damage rate produces. These
        // are what the per-step accumulation lands on directly.
        _ => 5e-2,
    }
}

// ---------------------------------------------------------------------------
// The frozen tie list
// ---------------------------------------------------------------------------

/// Which discrete quantity a divergence is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Discrete {
    Flags,
    Anim,
    AirJumps,
    SwingId,
    Events,
}

/// Every step, out of 4 048, where the `f32` port and the `f64` reference reach
/// a DIFFERENT discrete answer — as `(case, quantity, first_step, last_step)`,
/// inclusive.
///
/// # Why there are any at all, and why they are all the same one
///
/// `STEP_DT` is 1/120, which is not representable in binary. In `f64` the error
/// is small enough that `GRAVITY * STEP_DT` rounds to EXACTLY 6.875 px/s; in
/// `f32` the same product is 2.6e-7 over the halfway point and rounds UP by one
/// ULP, to 6.8750005. So the port's falling body gains one ULP of extra speed
/// per step and the reference's does not.
///
/// On its own that is invisible — it is the 1e-4 px/s in the continuous table.
/// What makes it visible is that every duration and threshold in the game is a
/// ROUND number against a 120 Hz step, so the `f64` build lands on the boundary
/// EXACTLY and a strict comparison against it falls the other way for the sake
/// of that one ULP:
///
/// - `DASH_COOLDOWN` is 0.55 s = exactly 66 steps, so `dash_cooldown <= 0`
///   flips a step apart. (`dash`, step 86.)
/// - `HURT_TIME` is 0.3 s = 36 steps and `HURT_REPEAT` is 0.45 s = 54 steps, so
///   the hurt pose and the hurt event's rate limit both land on the tie, four
///   times over as the player dies in the lava. (`hurt`.)
/// - At the apex of a jump the reference's `vy` is exactly 0.0, so `vy > 0`
///   — the wall-slide test — is false there and true one ULP later.
///   (`wall-slide-jump`, step 84.)
/// - Landing on the one-way platform, the reference's impact speed is exactly
///   `LAND_IMPACT_MIN` (89.375 px/s) so `vy > LAND_IMPACT_MIN` is false; the
///   port is one ULP over and plays the whole 0.13 s landing crouch.
///   (`one-way-from-below`, 16 steps — the longest run here, and still one
///   comparison.)
/// - `run_phase` crosses 0.5 and 1.0 by accumulation with nothing to snap it
///   back, so a footstep event fires one step early or late. (Five cases, two
///   steps each: the event MOVES, it is never lost or doubled.)
///
/// Every one is a tie on a strict inequality, every one resolves by itself, and
/// not one of them is a difference in what the player DOES: no entry here
/// changes a position, a velocity, a jump, a landing or a collision. That is the
/// whole claim, and it is why `on_ground` and its eleven companions are exact on
/// all 4 048 steps while `dash_ready` is not.
///
/// # Maintaining this list
///
/// It is asserted in BOTH directions. A new divergence fails; an entry that has
/// stopped diverging fails too. If a deliberate change moves one, read the
/// `case / quantity / step` triples out of the failure and edit this table — do
/// not widen a budget and do not drop a check.
const KNOWN_TIES: &[(&str, Discrete, usize, usize)] = &[
    // run_phase crossing a footstep threshold: the event moves one step.
    ("run-and-stop", Discrete::Events, 108, 109),
    ("jump-full", Discrete::Events, 128, 129),
    ("jump-cut", Discrete::Events, 102, 103),
    ("dash", Discrete::Events, 60, 61),
    ("bounce", Discrete::Events, 192, 193),
    // DASH_COOLDOWN is exactly 66 steps.
    ("dash", Discrete::Flags, 86, 86),
    // vy is exactly 0.0 at the apex, so `vy > 0` picks a different pose.
    ("wall-slide-jump", Discrete::Anim, 84, 84),
    // Impact speed is exactly LAND_IMPACT_MIN, so the port crouches and the
    // reference does not, for one LAND_HOLD.
    ("one-way-from-below", Discrete::Anim, 62, 77),
    // HURT_TIME (36 steps) and HURT_REPEAT (54 steps) are both exact.
    ("hurt", Discrete::Anim, 54, 54),
    ("hurt", Discrete::Anim, 90, 90),
    ("hurt", Discrete::Anim, 108, 109),
    ("hurt", Discrete::Anim, 144, 145),
    ("hurt", Discrete::Anim, 162, 164),
    ("hurt", Discrete::Anim, 198, 200),
    ("hurt", Discrete::Events, 54, 55),
    ("hurt", Discrete::Events, 108, 108),
    ("hurt", Discrete::Events, 110, 110),
    ("hurt", Discrete::Events, 162, 162),
    ("hurt", Discrete::Events, 165, 165),
];

fn tie_index(case: &str, what: Discrete, step: usize) -> Option<usize> {
    KNOWN_TIES
        .iter()
        .position(|&(c, w, lo, hi)| c == case && w == what && step >= lo && step <= hi)
}

// ---------------------------------------------------------------------------
// The replay
// ---------------------------------------------------------------------------

/// The largest divergence seen for one continuous quantity, and where.
#[derive(Default, Clone)]
struct Worst {
    diff: f64,
    case: String,
    step: usize,
    ts: f64,
    rs: f64,
}

impl Worst {
    fn see(&mut self, diff: f64, case: &str, step: usize, ts: f64, rs: f64) {
        if diff > self.diff {
            *self = Worst {
                diff,
                case: case.to_string(),
                step,
                ts,
                rs,
            };
        }
    }
}

/// The continuous quantities: the fixture's key, then this crate's name for it.
const QUANTITIES: [(&str, &str); 10] = [
    ("x", "x"),
    ("y", "y"),
    ("vx", "vx"),
    ("vy", "vy"),
    ("submersion", "submersion"),
    ("conveyorVx", "conveyor_vx"),
    ("health", "health"),
    ("landImpact", "land_impact"),
    ("stepUpVisual", "step_up_visual"),
    ("runPhase", "run_phase"),
];

#[test]
fn the_player_moves_the_way_the_typescript_one_did() {
    let fx = fixture();
    let meta = &fx["meta"];
    let grid = build_grid(meta, &fx["edits"]);

    let dt = f(&meta["dt"]) as f32;
    assert_eq!(
        dt, STEP_DT,
        "the fixture was recorded at a different timestep"
    );

    let spawn = SpawnPoint {
        x: f(&meta["spawn"]["x"]) as f32,
        y: f(&meta["spawn"]["y"]) as f32,
    };

    let mut worst: Vec<Worst> = vec![Worst::default(); QUANTITIES.len()];
    let mut tie_hit = vec![false; KNOWN_TIES.len()];
    let mut problems: Vec<String> = Vec::new();
    let mut total_steps = 0usize;
    let mut compared = 0usize;
    let mut diverged = 0usize;

    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;

        // Each case sets its own start; `Player::new` runs the real `reset`
        // first, so the fields a case does NOT set are the ones a respawn leaves
        // behind, which is the state the original was recorded from.
        let mut p = Player::new(spawn);
        p.x = f(&case["start"]["x"]) as f32;
        p.y = f(&case["start"]["y"]) as f32;
        p.vx = f(&case["start"]["vx"]) as f32;
        p.vy = f(&case["start"]["vy"]) as f32;
        p.facing = case["start"]["facing"].as_i64().unwrap() as f32;

        let dir = ints(&case["dir"]);
        let inb = ints(&case["in"]);
        let ref_flags = ints(&case["flags"]);
        let ref_anim = ints(&case["anim"]);
        let ref_air = ints(&case["airJumps"]);
        let ref_swing = ints(&case["swingId"]);
        let ref_events = case["events"].as_array().unwrap();
        let refs: Vec<Vec<f64>> = QUANTITIES
            .iter()
            .map(|(fixture_key, _)| floats(&case[*fixture_key]))
            .collect();

        let mut drained: Vec<PlayerEvent> = Vec::new();

        for i in 0..n {
            // The fixture was dumped from a TypeScript player with no projectile
            // pool wired up, so the body fires into the one that refuses every
            // shot. This is the only line the port's `Loadout` change touched in
            // this file: `Player::new` used to install `NoProjectiles` itself,
            // and the replay it drives is unchanged — 22 cases, 4048 steps.
            let mut nowhere = NoProjectiles;
            p.step(
                dt,
                intent_of(dir[i], inb[i]),
                &grid,
                &mut Loadout::new(&mut nowhere),
            );
            total_steps += 1;

            // Drained every step, exactly as the fixture did: an undrained
            // buffer would start dropping its oldest past MAX_EVENTS and the two
            // sides would be comparing different queues.
            drained.clear();
            p.drain_events(&mut drained);
            let got_events: Vec<i64> = drained.iter().map(|e| event_index(*e)).collect();

            // --- tier 1: discrete, exact but for the frozen tie list ------
            let checks: [(Discrete, String, String); 5] = [
                (
                    Discrete::Flags,
                    format!("{:#013b}", flags_of(&p)),
                    format!("{:#013b}", ref_flags[i]),
                ),
                (
                    Discrete::Anim,
                    anim_index(p.anim()).to_string(),
                    ref_anim[i].to_string(),
                ),
                (
                    Discrete::AirJumps,
                    p.air_jumps.to_string(),
                    ref_air[i].to_string(),
                ),
                (
                    Discrete::SwingId,
                    p.swing_id().to_string(),
                    ref_swing[i].to_string(),
                ),
                (
                    Discrete::Events,
                    format!("{got_events:?}"),
                    format!("{:?}", ints(&ref_events[i])),
                ),
            ];
            for (what, got, want) in &checks {
                compared += 1;
                if got == want {
                    continue;
                }
                diverged += 1;
                match tie_index(name, *what, i) {
                    Some(k) => tie_hit[k] = true,
                    None if problems.len() < 40 => problems.push(format!(
                        "{name} step {i}: {what:?} is {got} but the reference says {want} \
                         — not in KNOWN_TIES"
                    )),
                    None => {}
                }
            }

            // --- tier 2: continuous, bounded ------------------------------
            let got_f: [f32; 10] = [
                p.x,
                p.y,
                p.vx,
                p.vy,
                p.submersion,
                p.conveyor_vx,
                p.health,
                p.land_impact,
                p.step_up_visual(),
                p.run_phase(),
            ];
            for (q, _) in QUANTITIES.iter().enumerate() {
                let ts = refs[q][i];
                let rs = f64::from(got_f[q]);
                worst[q].see((rs - ts).abs(), name, i, ts, rs);
            }
        }
    }

    assert_eq!(
        total_steps,
        meta["totalSteps"].as_u64().unwrap() as usize,
        "replayed a different number of steps than the fixture recorded"
    );

    // Printed unconditionally: the size of the f32-vs-f64 difference belongs in
    // the log of a passing run as much as a failing one, because it is the only
    // place anyone can see how much headroom the budgets have left.
    let mut table = String::from("\nf32 port vs f64 reference, worst absolute divergence:\n");
    for (q, (_, name)) in QUANTITIES.iter().enumerate() {
        let w = &worst[q];
        table.push_str(&format!(
            "  {name:<15} {:>10.3e}  (budget {:>8.0e})  at {}[{}]  ref={} port={}\n",
            w.diff,
            budget(name),
            if w.case.is_empty() { "-" } else { &w.case },
            w.step,
            w.ts,
            w.rs
        ));
    }
    table.push_str(&format!(
        "  discrete: {diverged} of {compared} comparisons diverged ({:.3}%), all in KNOWN_TIES\n",
        100.0 * diverged as f64 / compared as f64
    ));
    println!("{table}");

    for (q, (_, name)) in QUANTITIES.iter().enumerate() {
        let w = &worst[q];
        if w.diff > budget(name) {
            problems.push(format!(
                "{name} drifted {:.3e} at {}[{}] (reference {}, port {}), over its {:.0e} budget",
                w.diff,
                w.case,
                w.step,
                w.ts,
                w.rs,
                budget(name)
            ));
        }
    }

    for (k, hit) in tie_hit.iter().enumerate() {
        if !hit {
            let (c, w, lo, hi) = KNOWN_TIES[k];
            problems.push(format!(
                "KNOWN_TIES entry ({c}, {w:?}, {lo}..={hi}) no longer diverges — \
                 if that was deliberate, delete the entry"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "the port does not reproduce the TypeScript player\n{table}\n{} problem(s):\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}
