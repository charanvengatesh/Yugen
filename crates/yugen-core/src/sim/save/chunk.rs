//! One chunk of diverged cells: `GGCH`, and the five planes behind it.
//!
//! A chunk is five flat arrays of fixed length over primitive integers — no
//! graph, no optionality, no field coming and going — which is why the format
//! is hand-rolled rather than derived, and why it is exactly [`ENCODED`] bytes
//! every time. See `docs/SAVE.md` for the byte layout and for the versioning
//! rule this file is currently on the wrong side of.

use crate::config::CHUNK_CELLS;
use crate::sim::chunk::ChunkSnapshot;
use crate::sim::grid::CellFlags;
use crate::sim::materials::CellId;

/// File signature. `GGCH` — Yūgen CHunk.
///
/// Four bytes at offset zero, checked on every read. Without it, pointing the
/// loader at the wrong directory produces a world assembled out of whatever
/// those bytes happened to mean, which is far worse than a refusal: it looks
/// like a corrupted save rather than a mistake.
const MAGIC: [u8; 4] = *b"GGCH";

/// Format version. Bump on ANY layout change, including adding a plane.
///
/// A reader that meets a version it does not know refuses the file and the chunk
/// regenerates from the seed. That is the right failure: a pristine chunk is
/// exactly what worldgen would produce, so an unreadable save costs the player
/// the edits in that chunk and nothing else.
const VERSION: u16 = 1;

/// Header bytes before the planes: magic, version, cell count, coordinates.
const HEADER: usize = 4 + 2 + 2 + 4 + 4;

/// Cells in one chunk's plane.
pub(super) const CELLS: usize = (CHUNK_CELLS * CHUNK_CELLS) as usize;

/// Bytes one encoded chunk occupies: header plus `u16`, `u8`, `u16`, `u8`, `u16`
/// planes.
const ENCODED: usize = HEADER + CELLS * (2 + 1 + 2 + 1 + 2);

// --- The format --------------------------------------------------------------

/// Encode a snapshot. Always exactly [`ENCODED`] bytes.
///
/// Little-endian throughout, stated once here rather than at each field: the
/// only machines this runs on are little-endian, and picking the native order
/// would make the file's meaning depend on where it was written — which is the
/// one thing a save format may not do.
pub fn encode_chunk(snap: &ChunkSnapshot) -> Vec<u8> {
    let mut out = Vec::with_capacity(ENCODED);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(CHUNK_CELLS as u16).to_le_bytes());
    out.extend_from_slice(&snap.chunk_x.to_le_bytes());
    out.extend_from_slice(&snap.chunk_y.to_le_bytes());

    for v in &snap.material {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &snap.flags {
        out.push(v.bits());
    }
    for v in &snap.aux {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&snap.temp);
    for v in &snap.back {
        out.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!(out.len(), ENCODED);
    out
}

/// Decode a snapshot, or say why not.
///
/// Every rejection is a `None` rather than a panic. This runs on bytes that came
/// off a disk somebody else's program may also have written to, and the only
/// safe reading of "these bytes are not a chunk" is to regenerate the chunk.
pub fn decode_chunk(bytes: &[u8]) -> Option<ChunkSnapshot> {
    if bytes.len() != ENCODED || bytes[..4] != MAGIC {
        return None;
    }
    if u16::from_le_bytes([bytes[4], bytes[5]]) != VERSION {
        return None;
    }
    // The cell count is checked rather than assumed: an old save written when
    // CHUNK_CELLS was a different number would otherwise be read as this one's
    // planes at the wrong stride, which is garbage that parses.
    if u16::from_le_bytes([bytes[6], bytes[7]]) != CHUNK_CELLS as u16 {
        return None;
    }

    let mut at = 8;
    let i32_at = |at: &mut usize| {
        let v = i32::from_le_bytes(bytes[*at..*at + 4].try_into().expect("4 bytes"));
        *at += 4;
        v
    };
    let chunk_x = i32_at(&mut at);
    let chunk_y = i32_at(&mut at);

    let u16s = |at: &mut usize| {
        let v: Vec<u16> = bytes[*at..*at + CELLS * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        *at += CELLS * 2;
        v
    };
    let material: Vec<CellId> = u16s(&mut at);

    let flags: Vec<CellFlags> = bytes[at..at + CELLS]
        .iter()
        // `from_bits_truncate` and not `from_bits`: an unknown bit in a save
        // written by a newer build is a flag this build does not have, and
        // dropping it is right. The cell keeps its material, which is the part
        // the player put there.
        .map(|b| CellFlags::from_bits_truncate(*b))
        .collect();
    at += CELLS;

    let aux = u16s(&mut at);
    let temp = bytes[at..at + CELLS].to_vec();
    at += CELLS;
    let back: Vec<CellId> = u16s(&mut at);

    Some(ChunkSnapshot {
        chunk_x,
        chunk_y,
        material,
        flags,
        aux,
        temp,
        back,
    })
}

#[cfg(test)]
mod tests {
    use super::super::testing::a_snapshot;
    use super::*;

    #[test]
    fn a_chunk_survives_a_round_trip_plane_for_plane() {
        let snap = a_snapshot(-7, 12);
        let back = decode_chunk(&encode_chunk(&snap)).expect("decodes");
        assert_eq!(back, snap);
    }

    #[test]
    fn an_encoded_chunk_is_exactly_the_size_the_header_promises() {
        assert_eq!(encode_chunk(&a_snapshot(0, 0)).len(), ENCODED);
    }

    #[test]
    fn every_way_a_file_can_be_wrong_decodes_to_nothing() {
        let good = encode_chunk(&a_snapshot(1, 1));

        assert!(decode_chunk(&[]).is_none(), "empty");
        assert!(decode_chunk(&good[..good.len() - 1]).is_none(), "truncated");

        let mut longer = good.clone();
        longer.push(0);
        assert!(decode_chunk(&longer).is_none(), "trailing bytes");

        let mut wrong_magic = good.clone();
        wrong_magic[0] = b'X';
        assert!(decode_chunk(&wrong_magic).is_none(), "magic");

        let mut wrong_version = good.clone();
        wrong_version[4] = VERSION.wrapping_add(1) as u8;
        assert!(decode_chunk(&wrong_version).is_none(), "version");

        let mut wrong_cells = good.clone();
        wrong_cells[6] = (CHUNK_CELLS as u16).wrapping_add(1) as u8;
        assert!(decode_chunk(&wrong_cells).is_none(), "cell count");
    }

    /// An unknown flag bit is dropped rather than refusing the file. The cell
    /// keeps its material, which is the part the player put there.
    #[test]
    fn a_flag_bit_this_build_does_not_know_costs_the_flag_and_not_the_chunk() {
        let snap = a_snapshot(0, 0);
        let mut bytes = encode_chunk(&snap);
        let flags_at = HEADER + CELLS * 2;
        bytes[flags_at] = 0xff;
        let back = decode_chunk(&bytes).expect("still decodes");
        assert_eq!(back.material, snap.material);
        assert_eq!(back.flags[0], CellFlags::from_bits_truncate(0xff));
    }
}
