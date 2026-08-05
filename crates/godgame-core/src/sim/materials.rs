//! The materials facade — the one place `godgame-data`'s block tables are named
//! the way the engine speaks about them.
//!
//! Hand-written code never imports `godgame_data::blocks` directly; it comes
//! through here. That indirection is what let the block registry become
//! compiled content without touching a single call site, and it is the seam that
//! keeps the generated module free to change shape.
//!
//! Everything re-exported below is a flat array indexed by material code. A
//! per-cell loop reads `MAT_DENSITY[id as usize]`, never `BLOCKS[id].density` —
//! an array read is a load, a def read is a load plus a struct offset, and the
//! automata does several million of them a second.

pub use godgame_data::blocks::{
    BLOCK_COUNT as MAT_COUNT, BLOCK_IDS, BLOCKS, BlockDef, BlockDrop, BlockFlammable, BlockGrowth,
    BlockHeat, BlockSurface, BlockTexture, MAT_B, MAT_BLASTRESIST, MAT_BURNINTO, MAT_BURNTIME,
    MAT_CLIMB, MAT_COLLIDE, MAT_COLORVAR, MAT_CONDUCT, MAT_CONTAINER, MAT_COOL, MAT_DAMAGE,
    MAT_DENSITY, MAT_EDGE, MAT_EMISSIVE, MAT_FLAMMABLE, MAT_G, MAT_GROWCHANCE, MAT_GROWDOWN,
    MAT_GROWINTO, MAT_GROWMAX, MAT_GROWS, MAT_HARDNESS, MAT_HEATEMIT, MAT_HEATTHRESH, MAT_IGNITE,
    MAT_IGNITEAT, MAT_LIGHT, MAT_MELTAT, MAT_MELTINTO, MAT_ONEWAY, MAT_R, MAT_SHIMMER, MAT_SPREAD,
    MAT_STATE, MAT_TAGS, MAT_TEXTURE, NEVER, Tag, Tex, block, grow_onto,
};

/// A cell's material code. 0 is empty.
///
/// A newtype would be nicer, but this is the value stored in every one of the
/// ~90,000 cells in the window and indexed into a dozen flat arrays per tick;
/// the alias keeps the intent legible without adding a conversion to the hot
/// path.
pub type CellId = u16;

/// Physical class of a material — what the automata does with it.
///
/// The discriminants are load-bearing: they are the values `MAT_STATE` holds,
/// assigned by the `state` enum's `map` in the block schema, and they are packed
/// into the behaviour byte's high nibble.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MaterialState {
    Empty = 0,
    Solid = 1,
    Powder = 2,
    Liquid = 3,
    Gas = 4,
}

impl MaterialState {
    /// Read a material's state out of the flat table.
    ///
    /// Total: an out-of-range code reads as [`MaterialState::Empty`], which is
    /// what an unloaded or corrupt cell should behave as.
    #[inline]
    pub fn of(id: CellId) -> MaterialState {
        match MAT_STATE.get(id as usize) {
            Some(1) => MaterialState::Solid,
            Some(2) => MaterialState::Powder,
            Some(3) => MaterialState::Liquid,
            Some(4) => MaterialState::Gas,
            _ => MaterialState::Empty,
        }
    }

    /// Powders and liquids fall; gases rise; solids and empty do neither.
    #[inline]
    pub fn is_mobile(self) -> bool {
        matches!(
            self,
            MaterialState::Powder | MaterialState::Liquid | MaterialState::Gas
        )
    }
}

/// The empty material — air. Code 0, and always will be.
pub const EMPTY: CellId = 0;

/// Definition for a material code. Out-of-range reads air rather than panicking.
#[inline]
pub fn mat_by_code(code: CellId) -> &'static BlockDef {
    BLOCKS.get(code as usize).unwrap_or(&BLOCKS[0])
}

/// Definition for an authoring id, or air if there is no such block.
pub fn mat_by_id(id: &str) -> &'static BlockDef {
    BLOCKS.iter().find(|b| b.id == id).unwrap_or(&BLOCKS[0])
}

/// Code for an authoring id, or 0 if there is no such block.
///
/// This is a linear scan and belongs in load-time code only. In a hot path use
/// the generated constants — `block::STONE` is resolved at compile time.
pub fn code_of(id: &str) -> CellId {
    BLOCKS.iter().find(|b| b.id == id).map_or(0, |b| b.code)
}

/// Does this material carry every one of these tags?
#[inline]
pub fn has_tags(id: CellId, tags: Tag) -> bool {
    Tag::from_bits_truncate(MAT_TAGS[id as usize]).contains(tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_code_zero_and_empty_forever() {
        assert_eq!(EMPTY, 0);
        assert_eq!(BLOCKS[0].code, 0);
        assert_eq!(MaterialState::of(0), MaterialState::Empty);
        assert_eq!(MAT_COLLIDE[0], 0, "air must never collide");
    }

    #[test]
    fn every_table_is_as_long_as_the_registry() {
        // A short table is an out-of-bounds read in the automata's inner loop.
        assert_eq!(MAT_STATE.len(), MAT_COUNT);
        assert_eq!(MAT_DENSITY.len(), MAT_COUNT);
        assert_eq!(MAT_TAGS.len(), MAT_COUNT);
        assert_eq!(MAT_HARDNESS.len(), MAT_COUNT);
        assert_eq!(BLOCK_IDS.len(), MAT_COUNT);
    }

    #[test]
    fn state_codes_agree_with_the_compiled_table() {
        // The enum restates the discriminants the block schema's `map` assigns.
        // If those ever drift, the automata dispatches on the wrong class.
        for (code, def) in BLOCKS.iter().enumerate() {
            let from_table = MaterialState::of(code as CellId);
            assert_eq!(from_table as u8, def.state, "{} disagrees", def.id);
        }
    }

    #[test]
    fn lookups_are_total() {
        assert_eq!(mat_by_code(9999).id, BLOCKS[0].id);
        assert_eq!(mat_by_id("no_such_block").id, BLOCKS[0].id);
        assert_eq!(code_of("no_such_block"), 0);
        assert_eq!(code_of("stone"), block::STONE);
    }
}
