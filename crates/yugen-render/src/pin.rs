//! Writing and checking the goldens in `tests/pin/`.
//!
//! Two algorithms in this tree are written twice — the sprite rasteriser and the
//! sound synthesiser — because `yugen-editor` cannot link this crate and could
//! not use it if it did: the renderer works from COMPILED content, and an editor
//! has to show a file as it is being typed. `tests/pin/README.md` has the full
//! argument.
//!
//! Neither crate can call the other, so they meet at a file. This is the side
//! that WRITES it. `yugen-editor` only reads, which is deliberate: a copy that
//! could bless its own output would only ever prove it agreed with itself.
//!
//! Test-only. Nothing here ships.

use std::path::PathBuf;

/// Whether to rewrite the goldens rather than check them. `YUGEN_BLESS=1`, the
/// same switch the worldgen and player baselines use.
fn blessing() -> bool {
    std::env::var_os("YUGEN_BLESS").is_some_and(|v| v != "0" && !v.is_empty())
}

/// `tests/pin/<name>`, from this crate's manifest directory.
fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/pin")
        .join(name)
}

/// Bytes as hex, wrapped so the file is reviewable in a diff rather than being
/// one line thousands of characters wide.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2 + bytes.len() / 48);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && i % 48 == 0 {
            out.push('\n');
        }
        out.push_str(&format!("{b:02x}"));
    }
    out.push('\n');
    out
}

/// Hex back to bytes, ignoring any whitespace the wrapping introduced.
fn unhex(text: &str) -> Vec<u8> {
    let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    digits
        .chunks_exact(2)
        .map(|pair| {
            let s = std::str::from_utf8(pair).expect("hex is ASCII");
            u8::from_str_radix(s, 16).expect("the golden is hex")
        })
        .collect()
}

/// `f32` samples as their raw bits, little-endian.
///
/// The bits and not a rounded decimal, because the pin exists to catch a
/// DIFFERENCE, and a golden written to four places would hide exactly the small
/// numeric drift a reordered expression produces.
pub fn samples_to_bytes(pcm: &[f32]) -> Vec<u8> {
    pcm.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Compare `bytes` against the golden, or write it when blessing.
///
/// # Panics
///
/// When the bytes differ, which is the point. The message says how far in the
/// first difference is, because "the sound changed" and "the sound changed at
/// sample 3" are different bugs.
pub fn check(name: &str, bytes: &[u8]) {
    let file = path(name);
    if blessing() {
        std::fs::create_dir_all(file.parent().expect("tests/pin has a parent"))
            .expect("the pin directory is writable");
        std::fs::write(&file, hex(bytes)).expect("the golden is writable");
        return;
    }

    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\n  bless it with `YUGEN_BLESS=1 cargo test -p yugen-render pin`",
            file.display()
        )
    });
    let want = unhex(&text);
    if want == bytes {
        return;
    }

    let at = want
        .iter()
        .zip(bytes)
        .position(|(a, b)| a != b)
        .unwrap_or(want.len().min(bytes.len()));
    panic!(
        "{} does not match: {} bytes wanted, {} got, first difference at byte {at}\n  \
         this crate is the AUTHORITY — if the change was deliberate, bless it AND port it \
         to yugen-editor in the same commit",
        file.display(),
        want.len(),
        bytes.len(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_ignores_the_wrapping() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let text = hex(&bytes);
        assert!(text.contains('\n'), "the golden is wrapped");
        assert_eq!(unhex(&text), bytes);
    }

    #[test]
    fn samples_keep_their_exact_bits() {
        // The reason the golden is bits and not decimals: these two are
        // different sounds and round to the same four places.
        let a = 0.1234567_f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_ne!(samples_to_bytes(&[a]), samples_to_bytes(&[b]));
    }
}
