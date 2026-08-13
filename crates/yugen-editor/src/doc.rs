//! What is open: the record, the file it came from, and what has moved in it.
//!
//! # Why the file text lives next to the model
//!
//! [`Open`] carries both the parsed record and [`Open::text`], the file exactly
//! as it was read. That looks redundant and is the whole design: a save is a
//! splice into `text`, not a serialisation of the model, so the model only ever
//! has to describe *what changed* and never has to be able to reproduce the
//! parts that did not. The prose between sequences survives because nothing here
//! is capable of writing it.
//!
//! # The dirty flags are three, and that is deliberate
//!
//! `dirty`, `pal_dirty` and `head_dirty` are separate because they gate three
//! different splices, and a splice that runs when nothing moved is not free: it
//! rewrites a line that may be wrapped, spaced or aligned in a way this tool has
//! no opinion about. A session that painted a pixel must not rewrite the palette
//! table, and a session that never resized must not rewrite `cellsW`.

use std::path::PathBuf;

use crate::sound::Sound;
use crate::sprite::{Frame, Sprite};
use crate::synth::{self, Params};

/// How many strokes can be taken back. A stroke is one press-drag-release, so
/// this is deeper than it looks — the number that matters is "more than a session
/// of tinkering", not "one per pixel".
pub(crate) const UNDO_DEPTH: usize = 64;

/// The message under the canvas. Kept as state rather than drawn and forgotten
/// because the useful ones — what was saved, what to run next — are worth
/// leaving on screen.
pub(crate) enum Status {
    None,
    Note(String),
    Bad(String),
}

/// What a browser row opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Art,
    Sfx,
}

/// What a click on the canvas does.
///
/// Paint is the default and is what the editor did before there was a choice.
/// Flood is here rather than as a button because it needs a point, and the only
/// place a point comes from is the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Tool {
    #[default]
    Paint,
    Flood,
}

/// One undo step: everything an edit is able to move about a sequence.
///
/// Frames alone was enough while every edit was a stroke. A resize moves the
/// GRID and a palette generator moves the COLOURS, and an undo that restored ten
/// rows into a record now declaring eight would author a file `bake_frame`
/// refuses — which is a panic at `PreStartup`, not an error. So a snapshot
/// carries the shape and the colours a frame is only meaningful against.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Snapshot {
    pub(crate) frames: Vec<Frame>,
    pub(crate) cells_w: u32,
    pub(crate) cells_h: u32,
    pub(crate) pal: Vec<String>,
}

/// A sprite record, open for editing.
pub(crate) struct Art {
    pub(crate) sprite: Sprite,
    pub(crate) seq: usize,
    pub(crate) frame: usize,
    /// Set by any edit to the art. Tracked rather than derived because comparing
    /// two `Vec<Frame>` every repaint to decide whether to enable a button is
    /// work the frame budget of an editor does not need to do.
    pub(crate) dirty: bool,
    /// Set only by a palette edit. Separate from `dirty` so a session that never
    /// touched a colour does not rewrite the `pal` line — some files wrap it, and
    /// re-emitting would collapse the wrap into churn.
    pub(crate) pal_dirty: bool,
    /// Set only by a resize. Separate for the same reason `pal_dirty` is: a
    /// session that never changed the grid must not rewrite `cellsW`/`cellsH`.
    pub(crate) head_dirty: bool,
    pub(crate) undo: Vec<Snapshot>,
}

impl Art {
    /// Open a freshly-read record, with nothing pending and nothing to undo.
    pub(crate) fn new(sprite: Sprite) -> Self {
        Self {
            sprite,
            seq: 0,
            frame: 0,
            dirty: false,
            pal_dirty: false,
            head_dirty: false,
            undo: Vec::new(),
        }
    }

    /// The character at a texel, or `.` outside the art.
    pub(crate) fn glyph(&self, x: usize, y: usize) -> char {
        self.sprite.seqs[self.seq].frames[self.frame][y]
            .chars()
            .nth(x)
            .unwrap_or('.')
    }

    /// Paint one texel. Returns whether anything actually changed, so a drag
    /// across a pixel it already painted does not mark the file dirty twice or
    /// spend a repaint.
    pub(crate) fn paint(&mut self, x: usize, y: usize, ch: char) -> bool {
        let row = &mut self.sprite.seqs[self.seq].frames[self.frame][y];
        let mut chars: Vec<char> = row.chars().collect();
        if x >= chars.len() || chars[x] == ch {
            return false;
        }
        chars[x] = ch;
        *row = chars.into_iter().collect();
        self.dirty = true;
        true
    }

    /// Snapshot the current sequence before a stroke.
    ///
    /// The unit of undo is the SEQUENCE, not the pixel: a stroke can cross a
    /// frame boundary and an undo that restored one frame would leave the other
    /// half of the stroke behind.
    pub(crate) fn snapshot(&mut self) {
        if self.undo.len() == UNDO_DEPTH {
            self.undo.remove(0);
        }
        self.undo.push(Snapshot {
            frames: self.sprite.seqs[self.seq].frames.clone(),
            cells_w: self.sprite.cells_w,
            cells_h: self.sprite.cells_h,
            pal: self.sprite.pal.clone(),
        });
    }

    /// Replace the frame being edited, snapshotting first.
    ///
    /// Every generator and every operator goes through this, so "remembered to
    /// snapshot" is a property of one function rather than of twenty buttons.
    /// A generator that forgot would destroy a drawing unrecoverably — it
    /// replaces the whole frame, which is far more than a stroke does.
    pub(crate) fn put_current(&mut self, frame: Frame) {
        self.snapshot();
        self.sprite.seqs[self.seq].frames[self.frame] = frame;
        self.dirty = true;
    }

    /// Replace the current sequence's frames, snapshotting first.
    ///
    /// The frame index is clamped because a derived sequence can be shorter
    /// than the one it replaced.
    pub(crate) fn put_frames(&mut self, frames: Vec<Frame>) {
        self.snapshot();
        self.frame = self.frame.min(frames.len().saturating_sub(1));
        self.sprite.seqs[self.seq].frames = frames;
        self.dirty = true;
    }

    pub(crate) fn undo(&mut self) {
        let Some(prev) = self.undo.pop() else {
            return;
        };
        // Each field is compared before it is restored, so an undo of a STROKE
        // does not arm the palette and grid splices. The flags are one-way
        // within a session by design — an undo back to what the file already
        // says still writes the line, but writes it identically, so the diff is
        // empty either way and the cheap check is the honest one.
        if prev.cells_w != self.sprite.cells_w || prev.cells_h != self.sprite.cells_h {
            self.sprite.cells_w = prev.cells_w;
            self.sprite.cells_h = prev.cells_h;
            self.head_dirty = true;
        }
        if prev.pal != self.sprite.pal {
            self.sprite.pal = prev.pal;
            self.pal_dirty = true;
        }
        self.frame = self.frame.min(prev.frames.len().saturating_sub(1));
        self.sprite.seqs[self.seq].frames = prev.frames;
        self.dirty = true;
    }
}

/// A sound record, open for editing.
pub(crate) struct Sfx {
    pub(crate) sound: Sound,
    /// The last render, kept so the scope does not re-synthesise every repaint —
    /// a slider drag would otherwise render 44 100 samples per frame of UI.
    pub(crate) pcm: Vec<f32>,
    /// The parameters `pcm` was rendered from. Compared rather than a dirty flag
    /// so nothing can forget to set one.
    pub(crate) rendered: Params,
}

impl Sfx {
    pub(crate) fn new(sound: Sound) -> Self {
        let pcm = synth::render(&sound.params);
        let rendered = sound.params;
        Self {
            sound,
            pcm,
            rendered,
        }
    }

    /// Re-synthesise if the knobs have moved since the last render.
    pub(crate) fn refresh(&mut self) {
        if self.rendered != self.sound.params {
            self.pcm = synth::render(&self.sound.params);
            self.rendered = self.sound.params;
        }
    }
}

/// Whichever kind of record is open.
pub(crate) enum Doc {
    Art(Art),
    Sfx(Sfx),
}

/// One record, open for editing.
pub(crate) struct Open {
    pub(crate) path: PathBuf,
    /// The file as it was last read from disk, byte for byte. The base every
    /// splice is applied to, and what a save compares against to notice that
    /// something else changed the file meanwhile.
    pub(crate) text: String,
    pub(crate) doc: Doc,
}

impl Open {
    pub(crate) fn id(&self) -> &str {
        match &self.doc {
            Doc::Art(a) => &a.sprite.id,
            Doc::Sfx(s) => &s.sound.id,
        }
    }

    pub(crate) fn kind(&self) -> Kind {
        match &self.doc {
            Doc::Art(_) => Kind::Art,
            Doc::Sfx(_) => Kind::Sfx,
        }
    }

    pub(crate) fn dirty(&self) -> bool {
        match &self.doc {
            Doc::Art(a) => a.dirty || a.pal_dirty || a.head_dirty,
            Doc::Sfx(s) => s.sound.dirty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Editor;
    use crate::sound;
    use crate::sprite;
    use std::path::Path;

    fn content() -> PathBuf {
        Editor::find_content(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("the repo has one")
    }

    fn open_player() -> Art {
        let text =
            std::fs::read_to_string(content().join("sprites/player.toml")).expect("readable");
        Art::new(sprite::read(&text, "player").expect("reads"))
    }

    #[test]
    fn painting_reports_whether_anything_moved() {
        // A drag sends the same texel many times, and marking the file dirty on
        // a no-op would offer to save a file nothing changed in.
        let mut art = open_player();
        let was = art.glyph(0, 0);
        assert!(
            !art.paint(0, 0, was),
            "repainting the same glyph is a no-op"
        );
        assert!(!art.dirty);
        assert!(art.paint(0, 0, '1'));
        assert!(art.dirty);
        assert_eq!(art.glyph(0, 0), '1');
    }

    #[test]
    fn undo_restores_the_whole_sequence() {
        let mut art = open_player();
        let before = art.sprite.seqs[0].frames.clone();
        art.snapshot();
        art.paint(0, 0, '1');
        art.paint(1, 1, '2');
        art.undo();
        assert_eq!(art.sprite.seqs[0].frames, before);
    }

    #[test]
    fn undo_restores_the_grid_and_the_colours_a_frame_only_means_anything_against() {
        // The reason `Snapshot` is not just frames. A resize and a palette
        // generator both move state a frame is interpreted against, and an undo
        // that restored ten rows into a record declaring eight would author a
        // file the renderer panics on rather than reports.
        let mut art = open_player();
        let (w, h) = (art.sprite.cells_w, art.sprite.cells_h);
        let pal = art.sprite.pal.clone();

        art.snapshot();
        art.sprite.cells_h = h + 4;
        art.sprite.pal.push("#ffffff".into());
        art.head_dirty = true;
        art.pal_dirty = true;

        art.undo();
        assert_eq!((art.sprite.cells_w, art.sprite.cells_h), (w, h));
        assert_eq!(art.sprite.pal, pal);
    }

    #[test]
    fn a_stroke_undo_does_not_arm_the_palette_or_the_grid_splice() {
        // Three dirty flags exist so a session that painted one pixel rewrites
        // one thing. An undo that set them all would make every stroke rewrite
        // the `pal` table and the `cellsW` line.
        let mut art = open_player();
        art.snapshot();
        art.paint(0, 0, '1');
        art.undo();
        assert!(art.dirty, "the frames moved and moved back — still a write");
        assert!(!art.pal_dirty, "no colour was touched");
        assert!(!art.head_dirty, "the grid did not move");
    }

    #[test]
    fn the_undo_stack_is_bounded() {
        // Deep enough for a session, not unbounded: a snapshot now carries the
        // palette and the grid as well as the frames.
        let mut art = open_player();
        for _ in 0..UNDO_DEPTH + 8 {
            art.snapshot();
        }
        assert_eq!(art.undo.len(), UNDO_DEPTH);
    }

    #[test]
    fn the_scope_only_re_synthesises_when_a_knob_moves() {
        // A slider drag repaints continuously and each render is 44 100 samples
        // per second of sound. Re-rendering per frame of UI would make the drag
        // itself the expensive part.
        let text = std::fs::read_to_string(content().join("sounds/player.toml")).expect("readable");
        let mut sfx = Sfx::new(sound::read(&text, "step").expect("reads"));
        let first = sfx.pcm.clone();
        assert!(!first.is_empty());

        sfx.refresh();
        assert_eq!(sfx.pcm, first, "nothing moved, so nothing was rendered");

        sfx.sound.params.gain = 0.9;
        sfx.refresh();
        assert_ne!(sfx.pcm, first, "the knob moved, so the sound did");
        assert_eq!(sfx.rendered, sfx.sound.params);
    }
}
