//! Worldgen geometry — where the ground, the sea and the depth bands sit.
//!
//! Everything here is in absolute CELL units and measured either from the world
//! origin (rows) or downward FROM the local surface height (depths). That split
//! is what makes vertical infinity well-defined: the world is generated as a
//! pure function of absolute cell coordinates, with no finite top or bottom to
//! measure a fraction of.
//!
//! These are the numbers a level designer turns. The per-generator coefficients
//! (fBm octaves, cave thresholds, tree shapes) stay private to their own modules
//! under `sim::gen` and `sim::decor`.

use super::world::CHUNK_CELLS;

/// Default generation seed.
pub const SEED: u32 = 2334;

/// Cell row the ground surface averages around.
///
/// The surface oscillates around this by up to [`SURFACE_AMPLITUDE`]. Depth
/// bands below are measured downward from that local surface height, not from a
/// fixed world top.
pub const SURFACE_ANCHOR_Y: i32 = 48;

/// Cells the heightmap swings either side of [`SURFACE_ANCHOR_Y`].
pub const SURFACE_AMPLITUDE: i32 = 24;

/// Dirt cap thickness below the surface, in cells.
pub const TOPSOIL: i32 = 6;

/// Cells below the surface where the cavern band ends.
pub const CAVERN_DEPTH: i32 = 110;

/// Cells below the surface where the "endless deep" begins.
pub const DEEP_DEPTH: i32 = 210;

/// Sea level, as an absolute cell ROW (bigger = lower).
///
/// Every column whose ground line falls below this row is a basin, and worldgen
/// fills the gap with water — that is the whole mechanism behind oceans and
/// lakes, and it works only because it is a comparison against a global constant
/// rather than a flood fill (a flood fill would need neighbour reads and would
/// break chunk independence).
///
/// Sits 6 cells BELOW the mean surface anchor so the default column is dry land:
/// the continental spline has to actively dig to reach water. Measured over 20k
/// columns, about 17% of the world is below this line — enough that walking
/// finds a coast, little enough that most of a journey is on foot.
pub const SEA_LEVEL_Y: i32 = SURFACE_ANCHOR_Y + 6;

/// Half-width, in cells, of the height band around sea level that gets a
/// sand/gravel shore cap instead of the biome's own topsoil.
///
/// Wide enough to read as a beach at 5px cells, narrow enough that an inland
/// lake still gets a distinct rim rather than a sandy valley.
pub const SHORE_BAND: i32 = 5;

/// Depth (below the local surface) over which cave openness fades in from zero.
///
/// Without this the topsoil is a sponge — every fBm cave that clips the surface
/// punches a hole, and the ground reads as rotten rather than solid. 22 cells is
/// about three player heights, so you must actually dig or find a chasm. Ravines
/// deliberately ignore this fade — breaking the surface is their whole point.
pub const CAVE_SURFACE_FADE: i32 = 22;

/// Cells below the surface where the underworld's ash and lava begins.
///
/// A real bottom to the world rather than "endless deep, all lava": ash flats
/// and basalt over open lava seas between here and [`UNDERWORLD_FLOOR`], then
/// solid bedrock forever. Depths are measured below the local surface, like
/// every other band, so the underworld follows the terrain.
pub const UNDERWORLD_DEPTH: i32 = 470;

/// Cells below the surface where impenetrable bedrock begins.
pub const UNDERWORLD_FLOOR: i32 = 640;

/// Stride of the coarse lattice worldgen samples its low-frequency fields on
/// before bilinear-interpolating per cell.
///
/// MUST divide [`CHUNK_CELLS`], so the lattice is globally aligned and two
/// chunks sharing an edge interpolate from the identical corner samples — that
/// is what keeps the optimisation invisible to determinism rather than a source
/// of seams. The cave sampler is hand-unrolled for exactly this stride.
pub const GEN_LATTICE: usize = 4;

const _: () = assert!(
    (CHUNK_CELLS as usize).is_multiple_of(GEN_LATTICE),
    "GEN_LATTICE must divide CHUNK_CELLS or chunk edges stop sharing lattice corners"
);

/// How many cells one unit of authored world-feature length occupies.
///
/// Every frequency in `sim::worldgen` is *cycles per cell* and every threshold
/// and amplitude above is *cells*, so making the world bigger is not a matter of
/// editing them. It is one transform applied at the boundary of the field layer:
///
/// > divide the coordinate going in, multiply the cell length coming out.
///
/// Nothing between those two crossings moves. That is why no `*_FREQ` and none of
/// the depth constants above are touched by a scale change, and it is what keeps
/// the field layer readable — inside it, every number still means what its doc
/// comment says it means, in the same units it was authored in.
///
/// The unscaled space the field layer works in is called LEGACY cells throughout
/// worldgen. A world cell is [`WorldScale::coord`] of a legacy one.
///
/// # Why this is a parameter and not a constant
///
/// `tests/player_golden.rs` replays 4 048 fixed steps against an arena stamped
/// into a world the fixture cannot regenerate — its provenance is a TypeScript
/// tool in a repository that no longer exists, and it has no bless path. It pins
/// its world with a material hash, so the generator moving underneath it is a
/// failure it cannot absorb. It therefore generates at [`WorldScale::LEGACY`]
/// forever, which is the identity, while the game runs at [`WorldScale::LIVE`].
///
/// A const would have forced that fixture to be retired. A parameter costs one
/// argument on the worldgen call path and keeps the only long-replay net over
/// `Player::step` alive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldScale(f64);

impl WorldScale {
    /// The identity. What `player_golden` generates at, permanently.
    ///
    /// Every crossing below is a no-op at this scale, which is what lets the
    /// threading be proven byte-identical by leaving the goldens green and
    /// UNBLESSED rather than by review.
    pub const LEGACY: WorldScale = WorldScale(1.0);

    /// What the game generates at.
    ///
    /// Still the identity: this commit threads the parameter and proves the
    /// threading is byte-identical by leaving `worldgen_golden`, `player_golden`
    /// and `noise_golden` green with their fixtures untouched. Raising it to 2.0
    /// is the next commit, and the one that actually changes the world.
    pub const LIVE: WorldScale = WorldScale(1.0);

    /// A world cell coordinate, in the field layer's legacy-cell space.
    ///
    /// This is the ONLY way a world coordinate may enter a noise sample. Dividing
    /// here is exactly equivalent to halving every frequency, and it is one place
    /// rather than sixty.
    #[inline]
    pub fn coord(self, wc: i32) -> f64 {
        f64::from(wc) / self.0
    }

    /// A depth measured in world cells, in legacy cells — so every band threshold
    /// in this module is compared against the depth it was authored against.
    #[inline]
    pub fn depth(self, cells: f64) -> f64 {
        cells / self.0
    }

    /// A length authored in legacy cells, in world cells. The outward crossing:
    /// surface rows and procedural extents come back through here.
    #[inline]
    pub fn len(self, cells: f64) -> f64 {
        cells * self.0
    }

    /// An absolute row authored in legacy cells, as a world cell ROW.
    ///
    /// [`SEA_LEVEL_Y`] and [`SURFACE_ANCHOR_Y`] are the two of these. They are
    /// positions rather than depths, so they scale outward like a length — the
    /// waterline sits twice as far from the origin in a world whose columns are
    /// twice as tall.
    #[inline]
    pub fn row(self, legacy_row: i32) -> i32 {
        (f64::from(legacy_row) * self.0 + 0.5).floor() as i32
    }

    /// The raw factor, for the few callers that must scale something this type
    /// has no better name for.
    #[inline]
    pub fn factor(self) -> f64 {
        self.0
    }
}
