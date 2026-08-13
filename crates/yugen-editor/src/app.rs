//! The editor: what is open, and the four things that touch the disk.
//!
//! # What this is for
//!
//! Two kinds of content in this game are authored as things you have to imagine
//! from numbers. A sprite is digits in a `'''` body; a sound is eight floats and
//! a waveform name. Both were drawn and tuned that way — all of
//! `content/sprites/player.toml` and all of `content/sounds/` — and both cost
//! more per change than they should. This is the tool that makes a redraw a drag
//! of the mouse and a retune a slider you can hear.
//!
//! # The one rule everything is built around
//!
//! **The file is the document, not the model.** [`crate::doc::Open::text`] holds
//! the file exactly as it was read, start to finish, and it is never regenerated.
//! A save feeds the edit through [`crate::field`], which puts lines where lines
//! were and copies every other byte through. So the prose between sequences, the
//! banner at the top, the paragraph explaining why `place` is a triangle and not
//! noise, and the trailing spaces inside a body all survive an edit made with a
//! mouse by someone who never saw them.
//!
//! That has a visible consequence, and it is the right one: **what you did not
//! touch produces no diff.** An untouched sequence splices back byte-identical
//! and an untouched sound key is never rewritten at all, so `git diff` after a
//! session shows what actually moved.
//!
//! # Why the disk lives here and not in [`crate::ui`]
//!
//! [`Editor::save`] is the only function in the crate that writes a content
//! file, and that is a rule rather than an accident. The stale check, the
//! temp-and-rename and the "run contentc" status all have to happen on every
//! write; putting the write behind one function is what makes "every write does
//! them" a property of the code instead of a thing to remember. Panels draw and
//! mutate the model. They never reach the filesystem.
//!
//! # What it does not do
//!
//! It does not write `content/ids.lock.json` and it does not run `contentc`.
//! FORMAT.md §9 is explicit that the lock is the compiler's, and the status bar
//! says so after every save instead.

use std::num::NonZero;
use std::path::{Path, PathBuf};

use crate::doc::{Art, Doc, Kind, Open, Sfx, Status, Tool};
use crate::field::{replace_frames, replace_head_key};
use crate::procgen::sprite::Recipe;
use crate::sound;
use crate::sprite::{self, frames_body};
use crate::synth;
use crate::ui::{NewState, ResizeState, Rolls};

/// The files under `content/` holding at least one editable record, with those
/// records and what they are.
pub(crate) type Files = Vec<(PathBuf, Vec<(String, Kind)>)>;

/// Walk `content/` for records this tool can open.
///
/// One directory level deep, because that is the shape `content/` has and
/// `LAYOUT.toml` is what says so. Pulled out of [`Editor::new`] because creating
/// a record has to re-run it: a file that gained its first sprite was not in the
/// list a moment ago.
pub(crate) fn scan(root: &Path) -> Files {
    let mut files = Files::new();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    for dir in dirs {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        paths.sort();
        for path in paths {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let mut rows: Vec<(String, Kind)> = sprite::art_ids(&text)
                .into_iter()
                .map(|id| (id, Kind::Art))
                .collect();
            rows.extend(
                sound::sound_ids(&text)
                    .into_iter()
                    .map(|id| (id, Kind::Sfx)),
            );
            if !rows.is_empty() {
                files.push((path, rows));
            }
        }
    }
    files
}

/// The application.
pub struct Editor {
    pub(crate) root: PathBuf,
    /// Scanned at startup and after a record is created — `content/` is a few
    /// dozen files and re-walking it every repaint would be work for nothing.
    pub(crate) files: Files,
    pub(crate) open: Option<Open>,
    pub(crate) status: Status,
    /// The palette index the left button paints. The right button always erases,
    /// which is why index 0 is selectable but not very useful.
    pub(crate) ink: usize,
    pub(crate) zoom: f32,
    pub(crate) grid: bool,
    pub(crate) playing: bool,
    /// What a canvas click does.
    pub(crate) tool: Tool,

    // -- the generate panel --------------------------------------------------
    //
    // Workspace state, not document state: it outlives the record being edited,
    // exactly like `ink` and `zoom` do. A recipe carried on the document would
    // reset every time a different icon was opened, which is the opposite of
    // what a designer rolling through eighty of them wants.
    /// The recipe the generator draws from. Its `w`/`h` are overwritten from the
    /// open record every frame — see `Editor::genart` for why they are not knobs.
    pub(crate) recipe: Recipe,
    /// Bumped by every reroll so two clicks inside one clock tick differ.
    pub(crate) seed_counter: u32,
    /// The hue the ramp generators walk from, held as RGB because that is what
    /// egui's colour picker edits.
    pub(crate) base_rgb: [u8; 3],
    /// Degrees of hue per ramp step.
    pub(crate) hue_shift: f32,
    /// How far `harmonize` pulls. 0 is the identity.
    pub(crate) harmonize: f32,
    /// Texels the bob generator lifts by.
    pub(crate) bob_amp: i32,
    /// Whether the nudge buttons wrap or drop what leaves the grid.
    pub(crate) wrap_shift: bool,
    /// Flipped after each dither so pressing it twice inverts rather than repeats.
    pub(crate) dither_phase: u8,
    /// The resize dialog, and whether it is up.
    pub(crate) resize: ResizeState,
    /// The new-record dialog.
    pub(crate) new_rec: NewState,
    /// Sounds rolled this session. Not persisted — see `ui::gensfx`.
    pub(crate) rolls: Rolls,
    /// The speaker, opened on the first play rather than at startup.
    ///
    /// Lazily, for two reasons: opening a device is slow enough to be visible in
    /// a window's first frame, and it can fail — no default device, a machine
    /// with no sound at all — and an editor that refused to start because it
    /// could not find a speaker would be unusable for the half of its job that
    /// is pixels.
    pub(crate) audio: Option<rodio::MixerDeviceSink>,
    /// Set once the device has been tried and failed, so the failure is reported
    /// once instead of retried on every click.
    pub(crate) audio_dead: bool,
}

impl Editor {
    /// Open the editor on a `content/` directory.
    pub fn new(root: PathBuf) -> Self {
        let files = scan(&root);
        Self {
            root,
            files,
            open: None,
            status: Status::None,
            ink: 1,
            zoom: 24.0,
            grid: true,
            playing: false,
            tool: Tool::default(),
            recipe: Recipe::default(),
            seed_counter: 0,
            // A mid warm brown: something a ramp reads well from, and not a
            // colour anything in `content/` already is, so a generated palette
            // is visibly generated until it is tuned.
            base_rgb: [180, 120, 70],
            // Measured off the reference rather than picked: its blue-grey ramp
            // walks 218 degrees to 242 over five steps, and its red barely moves
            // at all. Five is the honest middle. Twelve, which this was, made
            // every generated ramp cross two hues.
            hue_shift: 5.0,
            harmonize: 0.0,
            bob_amp: 1,
            wrap_shift: false,
            dither_phase: 0,
            resize: ResizeState::default(),
            new_rec: NewState::default(),
            rolls: Rolls::default(),
            audio: None,
            audio_dead: false,
        }
    }

    /// Find `content/` by walking up from `start`.
    ///
    /// `LAYOUT.toml` is the probe rather than the directory name because it is
    /// the thing that makes a directory THIS tree's content — a `content/` folder
    /// belonging to something else would not have one.
    pub fn find_content(start: &Path) -> Option<PathBuf> {
        let mut dir = Some(start);
        while let Some(d) = dir {
            let candidate = d.join("content");
            if candidate.join("LAYOUT.toml").is_file() {
                return Some(candidate);
            }
            dir = d.parent();
        }
        None
    }

    pub(crate) fn open_record(&mut self, path: &Path, id: &str, kind: Kind) {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                self.status = Status::Bad(format!("{}: {e}", path.display()));
                return;
            }
        };
        let doc = match kind {
            Kind::Art => match sprite::read(&text, id) {
                Ok(s) => {
                    self.ink = self.ink.clamp(1, s.pal.len().saturating_sub(1).max(1));
                    Doc::Art(Art::new(s))
                }
                Err(e) => {
                    self.status = Status::Bad(format!("[{id}]: {e}"));
                    return;
                }
            },
            Kind::Sfx => match sound::read(&text, id) {
                Ok(s) => Doc::Sfx(Sfx::new(s)),
                Err(e) => {
                    self.status = Status::Bad(format!("[{id}]: {e}"));
                    return;
                }
            },
        };
        self.playing = false;
        self.open = Some(Open {
            path: path.to_path_buf(),
            text,
            doc,
        });
        self.status = Status::None;
    }

    /// Send samples to the speaker, opening it if this is the first play.
    ///
    /// Takes the PCM rather than reading it off the open document, because
    /// auditioning a roll from the history plays samples that are not the open
    /// sound's and must not become them.
    pub(crate) fn emit(&mut self, pcm: Vec<f32>) {
        if pcm.is_empty() {
            self.status = Status::Bad("this sound renders no samples".into());
            return;
        }

        if self.audio.is_none() && !self.audio_dead {
            match rodio::DeviceSinkBuilder::open_default_sink() {
                Ok(mut sink) => {
                    // The drop message is a development aid in a library and
                    // noise in an application that owns its own speaker.
                    sink.log_on_drop(false);
                    self.audio = Some(sink);
                }
                Err(e) => {
                    self.audio_dead = true;
                    self.status = Status::Bad(format!("no audio device: {e}"));
                    return;
                }
            }
        }
        let Some(sink) = self.audio.as_ref() else {
            return;
        };

        // Mono, at the rate the synthesiser renders. The mixer resamples if the
        // device disagrees, which is exactly the case this must not try to
        // second-guess: the samples are the game's, and the device is the OS's.
        sink.mixer().add(rodio::buffer::SamplesBuffer::new(
            NonZero::new(1).expect("1 is not zero"),
            NonZero::new(synth::SAMPLE_RATE).expect("the sample rate is not zero"),
            pcm,
        ));
    }

    /// Render the open sound and send it to the speaker.
    pub(crate) fn play_sound(&mut self) {
        let Some(Open {
            doc: Doc::Sfx(sfx), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        sfx.refresh();
        let pcm = sfx.pcm.clone();
        self.emit(pcm);
    }

    /// Write the open record back, as splices.
    pub(crate) fn save(&mut self) {
        let Some(open) = self.open.as_mut() else {
            return;
        };

        // Somebody else may have edited the file since it was read — the game's
        // own author, in a text editor, which is the expected workflow and not an
        // exotic one. Refusing is the only safe answer: this tool holds line
        // spans into the text it read, and applying them to a file that has moved
        // would splice over whatever now occupies those lines.
        match std::fs::read_to_string(&open.path) {
            Ok(disk) if disk != open.text => {
                self.status = Status::Bad(format!(
                    "{} changed on disk since it was opened — reload before saving",
                    open.path.display()
                ));
                return;
            }
            Err(e) => {
                self.status = Status::Bad(format!("{}: {e}", open.path.display()));
                return;
            }
            _ => {}
        }

        let text = match &open.doc {
            Doc::Art(art) => {
                let id = art.sprite.id.clone();
                let sub = art.sprite.sub();
                let mut text = open.text.clone();

                // Every sequence, not only the edited one. The untouched ones
                // splice back byte-identical, so this costs nothing in the diff
                // and removes the need to track which sequences moved.
                for (i, seq) in art.sprite.seqs.iter().enumerate() {
                    match replace_frames(&text, &id, &sub, i, &frames_body(&seq.frames)) {
                        Ok(next) => text = next,
                        Err(e) => {
                            self.status = Status::Bad(format!("[{id}] sequence {i}: {e}"));
                            return;
                        }
                    }
                }

                if art.pal_dirty {
                    let key = art.sprite.pal_key();
                    let entries: Vec<String> =
                        art.sprite.pal.iter().map(|c| format!("{c:?}")).collect();
                    let line = format!("{key} = [{}]", entries.join(", "));
                    match replace_head_key(&text, &id, &key, &line) {
                        Ok(next) => text = next,
                        Err(e) => {
                            self.status = Status::Bad(format!("[{id}] {key}: {e}"));
                            return;
                        }
                    }
                }

                if art.head_dirty {
                    // `replace_head_key`, not `set_head_key`: both are
                    // `required()` in the schema, so a record missing one is a
                    // fault to report and not a line for this tool to invent.
                    // The prefix is what makes a mob splice at `art.cellsW`.
                    for (name, n) in [
                        ("cellsW", art.sprite.cells_w),
                        ("cellsH", art.sprite.cells_h),
                    ] {
                        let key = format!("{}{name}", art.sprite.prefix);
                        let line = format!("{key} = {n}");
                        match replace_head_key(&text, &id, &key, &line) {
                            Ok(next) => text = next,
                            Err(e) => {
                                self.status = Status::Bad(format!("[{id}] {key}: {e}"));
                                return;
                            }
                        }
                    }
                }
                text
            }
            // A sound writes only the keys that moved — see `sound::Sound::write`
            // for why materialising the rest would be wrong.
            Doc::Sfx(sfx) => match sfx.sound.write(&open.text) {
                Ok(text) => text,
                Err(e) => {
                    self.status = Status::Bad(format!("[{}]: {e}", sfx.sound.id));
                    return;
                }
            },
        };

        // Temp-and-rename, the same shape `save.rs` writes world files with: a
        // half-written content file is one `contentc` cannot read, and a crash
        // mid-write should cost the edit and not the record.
        let tmp = open.path.with_extension("toml.tmp");
        if let Err(e) = std::fs::write(&tmp, &text) {
            self.status = Status::Bad(format!("{}: {e}", tmp.display()));
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &open.path) {
            self.status = Status::Bad(format!("{}: {e}", open.path.display()));
            return;
        }

        open.text = text;
        match &mut open.doc {
            Doc::Art(art) => {
                art.dirty = false;
                art.pal_dirty = false;
                art.head_dirty = false;
            }
            // The new baseline: what was just written is now what the file says,
            // so nothing is pending until a knob moves again.
            Doc::Sfx(sfx) => sfx.sound.original = sfx.sound.params,
        }
        self.status = Status::Note(format!(
            "saved {} — run `cargo run -p contentc` to compile it",
            open.path.display()
        ));
    }

    pub(crate) fn reload(&mut self) {
        if let Some(open) = self.open.as_ref() {
            let (path, id, kind) = (open.path.clone(), open.id().to_string(), open.kind());
            self.open_record(&path, &id, kind);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content() -> PathBuf {
        Editor::find_content(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("the repo has one")
    }

    #[test]
    fn the_content_root_is_found_by_its_manifest_and_not_by_its_name() {
        // Walking up for a directory literally called `content` would match one
        // belonging to something else entirely; `LAYOUT.toml` is what makes it
        // this tree's.
        let found = content();
        assert!(found.join("LAYOUT.toml").is_file());
        assert!(found.join("sprites/player.toml").is_file());
        assert_eq!(Editor::find_content(Path::new("/")), None);
    }

    #[test]
    fn the_browser_lists_both_kinds_and_marks_which_is_which() {
        // One window over the whole of `content/`: the sprite files and the sound
        // files are the same list, because they are the same job.
        let editor = Editor::new(content());
        let kinds: Vec<Kind> = editor
            .files
            .iter()
            .flat_map(|(_, rows)| rows.iter().map(|(_, k)| *k))
            .collect();
        assert!(kinds.contains(&Kind::Art), "no art records found");
        assert!(kinds.contains(&Kind::Sfx), "no sound records found");

        let sounds = editor
            .files
            .iter()
            .find(|(p, _)| p.ends_with("sounds/player.toml"))
            .expect("content/sounds/player.toml is listed");
        assert!(sounds.1.iter().all(|(_, k)| *k == Kind::Sfx));
        assert!(sounds.1.iter().any(|(id, _)| id == "step"));
    }

    #[test]
    fn the_scan_is_what_the_browser_shows_and_can_be_run_again() {
        // Phase-6 creation needs to re-list after appending a record, so the walk
        // is a function rather than a step inside the constructor.
        let root = content();
        assert_eq!(
            scan(&root).len(),
            Editor::new(root.clone()).files.len(),
            "the constructor and the rescan see the same tree"
        );
    }
}
