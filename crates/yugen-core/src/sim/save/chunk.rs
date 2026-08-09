//! One chunk of diverged cells: `GGCH`, and the planes behind it.
//!
//! A chunk is a handful of flat arrays of fixed length over primitive integers —
//! no graph, no optionality, no field coming and going within one plane — which
//! is why the format is hand-rolled rather than derived. See `docs/SAVE.md` for
//! the byte layout.
//!
//! # The plane table, and the accident it removes
//!
//! Version 1 wrote five planes back to back at fixed offsets, and its version
//! comment said to bump on any layout change *including adding a plane*. That
//! instruction was correct and its consequence was not survivable: a reader that
//! refuses a chunk regenerates it, and a regenerated chunk is indistinguishable
//! from one that was never touched. So adding a sixth plane would have silently
//! deleted every edit in every world — not with an error, not with a warning,
//! just terrain quietly back the way worldgen first drew it.
//!
//! Version 2 writes a table first: how many planes, and for each one a tag and
//! how wide its elements are. A reader can then step over a plane it does not
//! know without understanding it, and supply a default for one it expected and
//! did not find. Adding a plane stops being a version bump at all, which is the
//! same trade [`super::run`] makes with sections and for the same reason.

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

/// Format version.
///
/// **Adding a plane is no longer a reason to move this.** That is what the plane
/// table is for; see the header. What would still move it is a change to the
/// header or to the table's own shape — the framing, not the contents — and
/// a version this reader does not know is still refused, because a chunk it
/// cannot parse is one it cannot partially trust.
///
/// A version-1 chunk is migrated by [`super::legacy::decode_chunk_v1`] rather
/// than refused. Refusing was defensible when it cost the edits in one chunk;
/// it is not, now that a short function avoids it entirely.
const VERSION: u16 = 2;

/// Header bytes before the plane table: magic, version, cell count, coordinates,
/// plane count.
const HEADER: usize = 4 + 2 + 2 + 4 + 4 + 1;

/// Cells in one chunk's plane.
pub(super) const CELLS: usize = (CHUNK_CELLS * CHUNK_CELLS) as usize;

/// What a plane is, on disk.
///
/// A tag and an element width, one byte each. The width is what lets a reader
/// skip a plane it does not recognise — it does not need to know what `6` means
/// to know that `6` is two bytes a cell and therefore `2 * CELLS` bytes to step
/// over.
///
/// Tags are assigned once and never reused, exactly as content ids are. A
/// removed plane leaves its tag burned rather than freeing it for something
/// else, because an old file still has that tag in it and the two meanings would
/// be indistinguishable.
mod plane {
    pub const MATERIAL: u8 = 1;
    pub const FLAGS: u8 = 2;
    pub const AUX: u8 = 3;
    pub const TEMP: u8 = 4;
    pub const BACK: u8 = 5;
}

// --- The format --------------------------------------------------------------

/// Bytes an encoded chunk occupies: header, the plane table, and the planes.
///
/// Computed rather than a constant, because the plane table is what says how
/// many there are. The strictness the old fixed [`ENCODED`] bought is not lost —
/// [`decode_chunk`] still refuses anything that is not exactly this many bytes
/// — it is just now a length the file itself declares rather than one this build
/// assumes.
pub(super) fn encoded_len(planes: usize, payload_bytes: usize) -> usize {
    HEADER + planes * 2 + payload_bytes
}

/// Encode a snapshot.
///
/// Little-endian throughout, stated once here rather than at each field: the
/// only machines this runs on are little-endian, and picking the native order
/// would make the file's meaning depend on where it was written — which is the
/// one thing a save format may not do.
///
/// Planes are written in tag order, which makes one snapshot have exactly one
/// encoding — the same canonicality [`super::run`]'s sections have, for the same
/// reason: a save diff is only readable if the bytes do not shuffle.
pub fn encode_chunk(snap: &ChunkSnapshot) -> Vec<u8> {
    let table: [(u8, u8); 5] = [
        (plane::MATERIAL, 2),
        (plane::FLAGS, 1),
        (plane::AUX, 2),
        (plane::TEMP, 1),
        (plane::BACK, 2),
    ];
    let payload: usize = table.iter().map(|(_, w)| *w as usize * CELLS).sum();

    let mut out = Vec::with_capacity(encoded_len(table.len(), payload));
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(CHUNK_CELLS as u16).to_le_bytes());
    out.extend_from_slice(&snap.chunk_x.to_le_bytes());
    out.extend_from_slice(&snap.chunk_y.to_le_bytes());
    out.push(table.len() as u8);
    for (tag, width) in table {
        out.push(tag);
        out.push(width);
    }

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
    debug_assert_eq!(out.len(), encoded_len(table.len(), payload));
    out
}

/// Decode a snapshot, or say why not.
///
/// Every rejection is a `None` rather than a panic. This runs on bytes that came
/// off a disk somebody else's program may also have written to, and the only
/// safe reading of "these bytes are not a chunk" is to regenerate the chunk.
///
/// A plane this build does not know is stepped over by the width the table gives
/// it. A plane this build expects and does not find takes its documented default
/// — an empty material is `0`, which is what an untouched cell already is — so a
/// file written before a plane existed loads with everything it did have.
pub fn decode_chunk(bytes: &[u8]) -> Option<ChunkSnapshot> {
    if bytes.len() < HEADER || bytes[..4] != MAGIC {
        return None;
    }
    match u16::from_le_bytes([bytes[4], bytes[5]]) {
        VERSION => {}
        1 => return super::legacy::decode_chunk_v1(bytes),
        _ => return None,
    }
    // The cell count is checked rather than assumed: an old save written when
    // CHUNK_CELLS was a different number would otherwise be read as this one's
    // planes at the wrong stride, which is garbage that parses.
    if u16::from_le_bytes([bytes[6], bytes[7]]) != CHUNK_CELLS as u16 {
        return None;
    }

    let chunk_x = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let chunk_y = i32::from_le_bytes(bytes[12..16].try_into().ok()?);

    let count = bytes[16] as usize;
    let table_end = HEADER + count * 2;
    let table = bytes.get(HEADER..table_end)?;

    // Total up the planes before reading any, so a table that promises more than
    // the file holds is refused here rather than part way through a plane.
    let payload: usize = table.chunks_exact(2).map(|p| p[1] as usize * CELLS).sum();
    if bytes.len() != encoded_len(count, payload) {
        return None;
    }

    let mut material = vec![0 as CellId; CELLS];
    let mut flags = vec![CellFlags::empty(); CELLS];
    let mut aux = vec![0u16; CELLS];
    let mut temp = vec![0u8; CELLS];
    let mut back = vec![0 as CellId; CELLS];

    let mut at = table_end;
    let mut seen = 0u32;
    for p in table.chunks_exact(2) {
        let (tag, width) = (p[0], p[1] as usize);
        let span = bytes.get(at..at + width * CELLS)?;
        at += width * CELLS;

        // A plane whose width is not the width this build reads it at is a
        // disagreement about what the tag MEANS, and reading the first two bytes
        // of a four-byte material would be reinterpreting it. Refuse, exactly as
        // the run file refuses a mis-sized section.
        let want = match tag {
            plane::MATERIAL | plane::AUX | plane::BACK => 2,
            plane::FLAGS | plane::TEMP => 1,
            // Not ours. Already stepped over.
            _ => continue,
        };
        if width != want {
            return None;
        }
        // A repeated plane is two answers to one question.
        if seen & (1 << tag) != 0 {
            return None;
        }
        seen |= 1 << tag;

        let u16s = || -> Vec<u16> {
            span.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        match tag {
            plane::MATERIAL => material = u16s(),
            plane::AUX => aux = u16s(),
            plane::BACK => back = u16s(),
            // `from_bits_truncate` and not `from_bits`: an unknown bit in a save
            // written by a newer build is a flag this build does not have, and
            // dropping it is right. The cell keeps its material, which is the
            // part the player put there.
            plane::FLAGS => {
                flags = span
                    .iter()
                    .map(|b| CellFlags::from_bits_truncate(*b))
                    .collect();
            }
            plane::TEMP => temp = span.to_vec(),
            _ => unreachable!("skipped above"),
        }
    }

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

    /// Bytes-to-planes, read out of a file's own table.
    fn table_of(bytes: &[u8]) -> Vec<(u8, u8)> {
        let count = bytes[16] as usize;
        bytes[HEADER..HEADER + count * 2]
            .chunks_exact(2)
            .map(|p| (p[0], p[1]))
            .collect()
    }

    #[test]
    fn an_encoded_chunk_is_exactly_the_size_its_own_table_promises() {
        // The strictness the old fixed `ENCODED` bought is kept, and is now a
        // length the FILE declares rather than one this build assumes. It still
        // has to be exact.
        let bytes = encode_chunk(&a_snapshot(0, 0));
        let table = table_of(&bytes);
        let payload: usize = table.iter().map(|(_, w)| *w as usize * CELLS).sum();
        assert_eq!(bytes.len(), encoded_len(table.len(), payload));
        assert_eq!(
            table,
            vec![
                (plane::MATERIAL, 2),
                (plane::FLAGS, 1),
                (plane::AUX, 2),
                (plane::TEMP, 1),
                (plane::BACK, 2)
            ]
        );
    }

    #[test]
    fn a_plane_this_build_does_not_know_is_stepped_over() {
        // The whole reason version 2 exists. A chunk written by a build with a
        // sixth plane in it must load here, minus that plane — not be refused,
        // and above all not be refused SILENTLY, which for a chunk means
        // regenerating as pristine terrain and erasing the player's work.
        let snap = a_snapshot(3, -4);
        let good = encode_chunk(&snap);
        let count = good[16] as usize;
        let table_end = HEADER + count * 2;

        let mut with_extra = Vec::new();
        with_extra.extend_from_slice(&good[..16]);
        with_extra.push(count as u8 + 1);
        with_extra.extend_from_slice(&good[HEADER..table_end]);
        with_extra.push(99); // a tag from a future this build has not seen
        with_extra.push(4); // four bytes a cell
        with_extra.extend_from_slice(&good[table_end..]);
        with_extra.extend(std::iter::repeat_n(0xab, 4 * CELLS));

        let back = decode_chunk(&with_extra).expect("an unknown plane is not fatal");
        assert_eq!(back, snap, "every plane this build DOES know still read");
    }

    #[test]
    fn a_plane_that_is_missing_takes_its_documented_default() {
        // The other direction: a file written before a plane existed. Dropping
        // `back` leaves it zero, which is what an untouched back layer already
        // is, and everything else survives.
        let snap = a_snapshot(0, 0);
        let good = encode_chunk(&snap);
        let table_end = HEADER + good[16] as usize * 2;
        let payload_before_back: usize = (2 + 1 + 2 + 1) * CELLS;

        let mut without = Vec::new();
        without.extend_from_slice(&good[..16]);
        without.push(4);
        without.extend_from_slice(&good[HEADER..table_end - 2]); // drop BACK's row
        without.extend_from_slice(&good[table_end..table_end + payload_before_back]);

        let back = decode_chunk(&without).expect("a missing plane is not fatal");
        assert_eq!(back.material, snap.material);
        assert_eq!(back.temp, snap.temp);
        assert_eq!(back.back, vec![0; CELLS], "the absent plane defaulted");
    }

    #[test]
    fn a_plane_at_the_wrong_width_or_written_twice_is_refused() {
        // A width this build does not read the tag at is a disagreement about
        // what the tag MEANS, and reading two bytes of a four-byte material
        // would be reinterpreting it. A repeated plane is two answers to one
        // question. Neither is something a reader should have an opinion about.
        let good = encode_chunk(&a_snapshot(0, 0));

        let mut wrong_width = good.clone();
        wrong_width[HEADER + 1] = 4; // MATERIAL claims four bytes a cell
        assert!(decode_chunk(&wrong_width).is_none(), "width");

        let mut repeated = good.clone();
        repeated[HEADER + 2] = plane::MATERIAL; // FLAGS' row becomes a second MATERIAL
        repeated[HEADER + 3] = 2;
        assert!(decode_chunk(&repeated).is_none(), "repeat");
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
        // Past the header, past the table, past the material plane.
        let flags_at = HEADER + bytes[16] as usize * 2 + CELLS * 2;
        bytes[flags_at] = 0xff;
        let back = decode_chunk(&bytes).expect("still decodes");
        assert_eq!(back.material, snap.material);
        assert_eq!(back.flags[0], CellFlags::from_bits_truncate(0xff));
    }
}
