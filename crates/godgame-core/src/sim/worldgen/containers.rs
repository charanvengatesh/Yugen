//! The container block — the one thing worldgen places that HOLDS something.
//!
//! WHY THIS IS A WHOLE MODULE for three constants: three files that must not
//! import each other need the same answer to "which block is a chest".
//! `worldgen/structs.rs` needs it to STAMP one, and must stay clear of the item
//! model because it is pulled into the sim worker; `worldgen/loot.rs` needs it
//! to roll what comes out, and does import the item model; the game layer needs
//! it to notice one being broken. A leaf module over the material registry is
//! the only shape that serves all three without a cycle.
//!
//! `MAT_CONTAINER` is the authority on what a container IS — the id below is
//! only a preference for what to PLACE. That split is deliberate: content can
//! add a barrel or an urn and every chest that already exists in the world keeps
//! working, because breaking one is a table lookup and not an id compare.

use std::sync::LazyLock;

use crate::sim::materials::{CellId, MAT_CONTAINER, code_of};

const CONTAINER_ID: &str = "chest";

/// Resolved once, exactly where the TypeScript resolved it: at module load.
///
/// [`LazyLock`] rather than a `const`: `block::CHEST` would be a compile-time
/// constant, but then a content edit that dropped the block would be a build
/// error instead of the graceful degradation the two flags below exist to
/// express. This is initialise-once and `Sync`, so it is safe to read from a
/// rayon worker; it is not mutable state.
struct Container {
    code: CellId,
    present: bool,
}

static CONTAINER: LazyLock<Container> = LazyLock::new(|| {
    let code = code_of(CONTAINER_ID);
    Container {
        code,
        // Code 0 is air, which can never be a container, so this also catches a
        // registry that has lost the block entirely rather than painting holes
        // where chests should be.
        present: code != 0 && MAT_CONTAINER.get(code as usize) == Some(&1),
    }
});

/// The block a `mark=loot` glyph becomes.
#[inline]
pub fn container_code() -> CellId {
    CONTAINER.code
}

/// The gate every container feature hangs off — the stamp pass, the break scan
/// and the loot roll all collapse to one boolean test when it is false.
#[inline]
pub fn containers_present() -> bool {
    CONTAINER.present
}

/// Does this cell hold loot?
///
/// Total, unlike the TypeScript's raw index: an out-of-range code reads as "not
/// a container", which is what the `undefined !== 1` comparison did there and
/// what a corrupt cell should behave as here.
#[inline]
pub fn is_container(code: CellId) -> bool {
    MAT_CONTAINER.get(code as usize) == Some(&1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_placed_block_is_one_the_registry_calls_a_container() {
        // If content has a chest at all, the id we place and the table that
        // decides what a container IS must agree, or worldgen stamps a block the
        // break scan will never notice.
        if containers_present() {
            assert!(is_container(container_code()));
            assert_ne!(container_code(), 0);
        }
    }

    #[test]
    fn air_is_never_a_container_and_lookups_are_total() {
        assert!(!is_container(0));
        assert!(!is_container(u16::MAX));
    }
}
