//! Reading a sprite record out of content text, for a canvas to draw.
//!
//! # Reading is safe; writing is what needs the care
//!
//! This module parses with the `toml` crate and that is not a contradiction of
//! FORMAT.md §9. The rule there is about ROUND-TRIPPING — read text, emit text —
//! because a serialiser normalises `'''` bodies and drops comments. Reading alone
//! loses nothing, because nothing is written back through it: every edit leaves
//! by [`crate::field`], which puts lines where lines were. The parser is a
//! reader, and the file is edited as text.
//!
//! # Two hosts, one shape
//!
//! `contentc` emits art fields under two prefixes — a standalone `sprite` record
//! writes `cellsW`, and a `mob` writes `art.cellsW` with its sequences under
//! `[[id.art.seq]]`. `yugen-render` solves this with a trait and two impls; here
//! the difference is one string, because this module reads a dynamic
//! `toml::Value` rather than two generated structs. [`Sprite::prefix`] is that
//! string, and it is carried on the model rather than re-derived at write time so
//! a mob edit cannot be spliced at a sprite's key path.

use crate::raster::{self, RasterError};
use crate::splice::record_ids;

/// One frame: its rows, top to bottom, each `cellsW * grain` characters.
pub type Frame = Vec<String>;

/// One animation sequence, as authored.
#[derive(Debug, Clone, PartialEq)]
pub struct Seq {
    pub state: String,
    pub mode: String,
    /// This sequence's own rate. **0 means inherit** the sprite-wide `fps` — the
    /// schema's convention, and the reason this is not defaulted to 8 here: 0 is
    /// "unset", and materialising it would pin a sequence that wanted to follow
    /// the sprite if the sprite were ever retuned.
    pub fps: f32,
    pub frames: Vec<Frame>,
}

impl Seq {
    /// The rate to actually play at, resolving the inherit.
    pub fn rate(&self, sprite_fps: f32) -> f32 {
        if self.fps > 0.0 { self.fps } else { sprite_fps }
    }
}

/// A record's art, enough to draw it and enough to splice it back.
#[derive(Debug, Clone, PartialEq)]
pub struct Sprite {
    pub id: String,
    pub name: String,
    /// `""` for a standalone sprite, `"art."` for a mob's inline art.
    pub prefix: &'static str,
    pub cells_w: u32,
    pub cells_h: u32,
    pub grain: u32,
    /// The sprite-wide rate a sequence's `fps = 0` inherits. 8.0 is the schema
    /// default and is what a file that never mentions `fps` means.
    pub fps: f32,
    pub pal: Vec<String>,
    pub seqs: Vec<Seq>,
}

impl Sprite {
    /// Art width in TEXELS — what a frame row is counted in.
    ///
    /// `cellsW` is in CELLS and `grain` is texels per cell per axis, and keeping
    /// the two apart is the whole content of the `grain` field: a grain-2 sprite
    /// is drawn on a finer grid inside the SAME world rectangle.
    pub fn texel_w(&self) -> u32 {
        self.cells_w * self.grain
    }

    /// Art height in texels. See [`Sprite::texel_w`].
    pub fn texel_h(&self) -> u32 {
        self.cells_h * self.grain
    }

    /// The dotted tail [`crate::field::seq_spans`] wants: `seq` or `art.seq`.
    pub fn sub(&self) -> String {
        format!("{}seq", self.prefix)
    }

    /// The head key a palette edit is spliced at: `pal` or `art.pal`.
    pub fn pal_key(&self) -> String {
        format!("{}pal", self.prefix)
    }

    /// The palette as RGB, or the first entry that is not a colour.
    pub fn colours(&self) -> Result<Vec<[u8; 3]>, RasterError> {
        raster::palette(&self.pal)
    }
}

/// Everything that can stop a record being drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// The file does not parse as TOML at all.
    Toml(String),
    /// No record by that id.
    NoSuchRecord(String),
    /// The record parses but has no art fields under either prefix.
    ///
    /// Not an error the editor reports as a fault — most records in `content/`
    /// are blocks and items with no art — it is how the file browser decides what
    /// is openable.
    NoArt(String),
    /// A required art field is missing or the wrong type.
    Field { id: String, key: String },
    /// A frame is the wrong shape. Carries which sequence and which frame,
    /// because the canvas needs to say where.
    Shape {
        id: String,
        seq: usize,
        frame: usize,
        rows: usize,
        expected: u32,
    },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Toml(e) => write!(f, "{e}"),
            ReadError::NoSuchRecord(id) => write!(f, "no record `[{id}]`"),
            ReadError::NoArt(id) => write!(f, "`[{id}]` has no art"),
            ReadError::Field { id, key } => write!(f, "`[{id}]`: `{key}` is missing or wrong"),
            ReadError::Shape {
                id,
                seq,
                frame,
                rows,
                expected,
            } => write!(
                f,
                "`[{id}]` sequence {seq} frame {frame} has {rows} rows, wants {expected}"
            ),
        }
    }
}

impl std::error::Error for ReadError {}

/// Split a `'''` body into frames. A blank line separates frames — that is the
/// filmstrip convention of FORMAT.md §5, not formatting.
///
/// Mirrors `yugen-render/src/sprite/anim.rs::split_frames`, minus the row-count
/// check, which happens in [`read`] where the id is known well enough to say
/// which frame is wrong.
pub fn split_frames(body: &str) -> Vec<Frame> {
    let mut out: Vec<Frame> = Vec::new();
    let mut cur: Frame = Vec::new();
    for line in body.lines() {
        if line.is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        cur.push(line.to_string());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Frames back to body lines, ready for [`crate::field::replace_frames`].
///
/// The inverse of [`split_frames`], and the only place the editor decides what a
/// body looks like: one blank line between frames, no blank line at either end.
/// That is what the files already do, so a redraw of one frame produces a diff of
/// that frame and not of the whitespace around it.
pub fn frames_body(frames: &[Frame]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        if i > 0 {
            out.push(String::new());
        }
        out.extend(frame.iter().cloned());
    }
    out
}

/// Every record in the file that has art, in file order.
///
/// Used by the browser to list what can be opened. A file of blocks yields
/// nothing, which is the correct answer rather than an error.
pub fn art_ids(text: &str) -> Vec<String> {
    record_ids(text)
        .into_iter()
        .filter(|id| !matches!(read(text, id), Err(ReadError::NoArt(_))))
        .collect()
}

fn as_u32(v: Option<&toml::Value>) -> Option<u32> {
    v?.as_integer()?.try_into().ok()
}

/// Read one record's art.
pub fn read(text: &str, id: &str) -> Result<Sprite, ReadError> {
    let table: toml::Table = text.parse().map_err(|e| ReadError::Toml(format!("{e}")))?;
    let record = table
        .get(id)
        .ok_or_else(|| ReadError::NoSuchRecord(id.to_string()))?;

    // A standalone sprite writes `cellsW`; a mob writes `art.cellsW`, which
    // parses to a nested table. `cellsW` is the probe because the schema makes it
    // required on both hosts — a record that has it has art.
    let (prefix, art): (&'static str, &toml::Value) = if record.get("cellsW").is_some() {
        ("", record)
    } else if record.get("art").and_then(|a| a.get("cellsW")).is_some() {
        ("art.", record.get("art").expect("probed above"))
    } else {
        return Err(ReadError::NoArt(id.to_string()));
    };

    let field = |key: &str| ReadError::Field {
        id: id.to_string(),
        key: format!("{prefix}{key}"),
    };

    let cells_w = as_u32(art.get("cellsW")).ok_or_else(|| field("cellsW"))?;
    let cells_h = as_u32(art.get("cellsH")).ok_or_else(|| field("cellsH"))?;
    // `grain` and `variants` carry schema defaults, so a file that omits them is
    // correct and must not be an error here.
    let grain = match art.get("grain") {
        None => 1,
        Some(_) => as_u32(art.get("grain")).ok_or_else(|| field("grain"))?,
    };

    let pal: Vec<String> = art
        .get("pal")
        .and_then(|p| p.as_array())
        .ok_or_else(|| field("pal"))?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| field("pal"))?;

    let mut seqs = Vec::new();
    for (i, entry) in art
        .get("seq")
        .and_then(|s| s.as_array())
        .ok_or_else(|| field("seq"))?
        .iter()
        .enumerate()
    {
        let frames = split_frames(
            entry
                .get("frames")
                .and_then(|f| f.as_str())
                .ok_or_else(|| field("seq[].frames"))?,
        );
        for (f, frame) in frames.iter().enumerate() {
            if frame.len() != (cells_h * grain) as usize {
                return Err(ReadError::Shape {
                    id: id.to_string(),
                    seq: i,
                    frame: f,
                    rows: frame.len(),
                    expected: cells_h * grain,
                });
            }
        }
        seqs.push(Seq {
            // `state` has a default on the standalone host (`idle`) and is
            // required on mobs, so a missing one is shown rather than refused —
            // the editor is looking at a file that may be mid-edit.
            state: entry
                .get("state")
                .and_then(|s| s.as_str())
                .unwrap_or("idle")
                .to_string(),
            mode: entry
                .get("mode")
                .and_then(|s| s.as_str())
                .unwrap_or("hold")
                .to_string(),
            fps: entry.get("fps").and_then(|f| f.as_float()).unwrap_or(0.0) as f32,
            frames,
        });
    }

    Ok(Sprite {
        id: id.to_string(),
        name: record
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or(id)
            .to_string(),
        prefix,
        cells_w,
        cells_h,
        grain,
        fps: art.get("fps").and_then(|f| f.as_float()).unwrap_or(8.0) as f32,
        pal,
        seqs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPRITE: &str = "\
[player]\n\
name = \"Player\"\n\
cellsW = 2\n\
cellsH = 2\n\
grain = 2\n\
pal = [\".\", \"#1b1220\", \"#3ec8b4\"]\n\
[[player.seq]]\n\
state = \"idle\"\n\
mode = \"ambient\"\n\
frames = '''\n\
.11.\n\
1..1\n\
.11.\n\
1..1\n\
\n\
.22.\n\
2..2\n\
.22.\n\
2..2\n\
'''\n\
\n\
[stone]\n\
name = \"Stone\"\n\
hardness = 3.0\n\
";

    const MOB: &str = "\
[grubling]\n\
name = \"Grubling\"\n\
art.cellsW = 2\n\
art.cellsH = 1\n\
art.pal = [\".\", \"#6f4a2a\"]\n\
[[grubling.art.seq]]\n\
state = \"move\"\n\
frames = '''\n\
11\n\
'''\n\
";

    #[test]
    fn grain_multiplies_the_texel_grid_and_not_the_cell_box() {
        // The invariant `grain` exists to protect: the drawn rectangle is still
        // cellsW x cellsH CELLS, and only the art grid inside it gets finer.
        let s = read(SPRITE, "player").expect("reads");
        assert_eq!((s.cells_w, s.cells_h, s.grain), (2, 2, 2));
        assert_eq!((s.texel_w(), s.texel_h()), (4, 4));
        assert_eq!(s.seqs[0].frames[0].len(), 4);
    }

    #[test]
    fn a_blank_line_separates_frames_and_does_not_end_the_art() {
        let s = read(SPRITE, "player").expect("reads");
        assert_eq!(s.seqs.len(), 1);
        assert_eq!(s.seqs[0].frames.len(), 2);
        assert_eq!(s.seqs[0].frames[1][0], ".22.");
        assert_eq!(s.seqs[0].mode, "ambient");
    }

    #[test]
    fn a_mobs_art_reads_through_the_same_path_under_a_prefix() {
        // One editor for both hosts. The prefix is carried so a mob's palette is
        // spliced at `art.pal` and not at `pal`, which would add a second key
        // that nothing reads.
        let s = read(MOB, "grubling").expect("reads");
        assert_eq!(s.prefix, "art.");
        assert_eq!(s.sub(), "art.seq");
        assert_eq!(s.pal_key(), "art.pal");
        assert_eq!(
            s.grain, 1,
            "an omitted grain is the schema default, not an error"
        );
        assert_eq!(s.seqs[0].state, "move");

        let sprite = read(SPRITE, "player").expect("reads");
        assert_eq!(sprite.prefix, "");
        assert_eq!(sprite.sub(), "seq");
        assert_eq!(sprite.pal_key(), "pal");
    }

    #[test]
    fn a_record_with_no_art_is_reported_as_such_and_not_as_a_fault() {
        // How the browser filters. Most of `content/` is blocks and items.
        assert_eq!(read(SPRITE, "stone"), Err(ReadError::NoArt("stone".into())));
        assert_eq!(art_ids(SPRITE), vec!["player"]);
        assert!(matches!(
            read(SPRITE, "ghost"),
            Err(ReadError::NoSuchRecord(_))
        ));
    }

    #[test]
    fn a_frame_of_the_wrong_height_says_which_frame() {
        let bad = SPRITE.replace(".22.\n2..2\n.22.\n2..2\n", ".22.\n2..2\n");
        assert_eq!(
            read(&bad, "player"),
            Err(ReadError::Shape {
                id: "player".into(),
                seq: 0,
                frame: 1,
                rows: 2,
                expected: 4
            })
        );
    }

    #[test]
    fn frames_survive_a_round_trip_through_the_body_form() {
        // `split_frames` and `frames_body` are inverses on anything the editor
        // produces, which is what makes "redraw one pixel" a one-line diff.
        let s = read(SPRITE, "player").expect("reads");
        let body = frames_body(&s.seqs[0].frames);
        assert_eq!(
            body,
            vec![
                ".11.", "1..1", ".11.", "1..1", "", ".22.", "2..2", ".22.", "2..2"
            ]
        );
        assert_eq!(split_frames(&body.join("\n")), s.seqs[0].frames);
    }
}
