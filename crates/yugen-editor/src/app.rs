//! The editor window: a canvas, a set of knobs, and a save that is a splice.
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
//! **The file is the document, not the model.** [`Open::text`] holds the file
//! exactly as it was read, start to finish, and it is never regenerated. A save
//! feeds the edit through [`crate::field`], which puts lines where lines were and
//! copies every other byte through. So the prose between sequences, the banner at
//! the top, the paragraph explaining why `place` is a triangle and not noise, and
//! the trailing spaces inside a body all survive an edit made with a mouse by
//! someone who never saw them.
//!
//! That has a visible consequence, and it is the right one: **what you did not
//! touch produces no diff.** An untouched sequence splices back byte-identical
//! and an untouched sound key is never rewritten at all, so `git diff` after a
//! session shows what actually moved.
//!
//! # What it does not do
//!
//! It does not write `content/ids.lock.json` and it does not run `contentc`.
//! FORMAT.md §9 is explicit that the lock is the compiler's, and the status bar
//! says so after every save instead.

use std::num::NonZero;
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::field::replace_frames;
use crate::field::replace_head_key;
use crate::raster;
use crate::sound::{self, HZ_RANGE, SECONDS_RANGE, Sound};
use crate::sprite::{self, Frame, Sprite, frames_body};
use crate::synth::{self, Params, Wave};

/// How many strokes can be taken back. A stroke is one press-drag-release, so
/// this is deeper than it looks — the number that matters is "more than a session
/// of tinkering", not "one per pixel".
const UNDO_DEPTH: usize = 64;

/// Palette indices `1`-`9` are the only ones a frame character can name, so the
/// list tops out at ten entries including the transparent slot. FORMAT.md §5 is
/// where that ceiling comes from: a frame is digits, and there are nine of them.
const MAX_PAL: usize = 10;

/// How tall the waveform scope is drawn, in points.
///
/// Enough to read an envelope's shape and not so much that it pushes the knobs
/// off the screen — the scope answers "what did that do", and the knobs are what
/// you are looking at while asking.
///
/// The scope itself draws min and max per column rather than one sample per
/// column: at 44.1 kHz a tenth-of-a-second sound is 4410 samples across a few
/// hundred pixels, and picking one sample per column would alias the envelope
/// into whatever the stride happened to land on.
const SCOPE_HEIGHT: f32 = 90.0;

/// The message under the canvas. Kept as state rather than drawn and forgotten
/// because the useful ones — what was saved, what to run next — are worth
/// leaving on screen.
enum Status {
    None,
    Note(String),
    Bad(String),
}

/// What a browser row opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Art,
    Sfx,
}

/// A sprite record, open for editing.
struct Art {
    sprite: Sprite,
    seq: usize,
    frame: usize,
    /// Set by any edit to the art. Tracked rather than derived because comparing
    /// two `Vec<Frame>` every repaint to decide whether to enable a button is
    /// work the frame budget of an editor does not need to do.
    dirty: bool,
    /// Set only by a palette edit. Separate from `dirty` so a session that never
    /// touched a colour does not rewrite the `pal` line — some files wrap it, and
    /// re-emitting would collapse the wrap into churn.
    pal_dirty: bool,
    undo: Vec<Vec<Frame>>,
}

impl Art {
    /// The character at a texel, or `.` outside the art.
    fn glyph(&self, x: usize, y: usize) -> char {
        self.sprite.seqs[self.seq].frames[self.frame][y]
            .chars()
            .nth(x)
            .unwrap_or('.')
    }

    /// Paint one texel. Returns whether anything actually changed, so a drag
    /// across a pixel it already painted does not mark the file dirty twice or
    /// spend a repaint.
    fn paint(&mut self, x: usize, y: usize, ch: char) -> bool {
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
    fn snapshot(&mut self) {
        if self.undo.len() == UNDO_DEPTH {
            self.undo.remove(0);
        }
        self.undo.push(self.sprite.seqs[self.seq].frames.clone());
    }

    fn undo(&mut self) {
        if let Some(frames) = self.undo.pop() {
            self.frame = self.frame.min(frames.len().saturating_sub(1));
            self.sprite.seqs[self.seq].frames = frames;
            self.dirty = true;
        }
    }
}

/// A sound record, open for editing.
struct Sfx {
    sound: Sound,
    /// The last render, kept so the scope does not re-synthesise every repaint —
    /// a slider drag would otherwise render 44 100 samples per frame of UI.
    pcm: Vec<f32>,
    /// The parameters `pcm` was rendered from. Compared rather than a dirty flag
    /// so nothing can forget to set one.
    rendered: Params,
}

impl Sfx {
    fn new(sound: Sound) -> Self {
        let pcm = synth::render(&sound.params);
        let rendered = sound.params;
        Self {
            sound,
            pcm,
            rendered,
        }
    }

    /// Re-synthesise if the knobs have moved since the last render.
    fn refresh(&mut self) {
        if self.rendered != self.sound.params {
            self.pcm = synth::render(&self.sound.params);
            self.rendered = self.sound.params;
        }
    }
}

/// Whichever kind of record is open.
enum Doc {
    Art(Art),
    Sfx(Sfx),
}

/// One record, open for editing.
struct Open {
    path: PathBuf,
    /// The file as it was last read from disk, byte for byte. The base every
    /// splice is applied to, and what a save compares against to notice that
    /// something else changed the file meanwhile.
    text: String,
    doc: Doc,
}

impl Open {
    fn id(&self) -> &str {
        match &self.doc {
            Doc::Art(a) => &a.sprite.id,
            Doc::Sfx(s) => &s.sound.id,
        }
    }

    fn dirty(&self) -> bool {
        match &self.doc {
            Doc::Art(a) => a.dirty || a.pal_dirty,
            Doc::Sfx(s) => s.sound.dirty(),
        }
    }
}

/// The application.
pub struct Editor {
    root: PathBuf,
    /// Files under `content/` holding at least one editable record, with those
    /// records and what they are. Scanned once at startup — `content/` is a few
    /// dozen files and re-walking it every repaint would be work for nothing.
    files: Vec<(PathBuf, Vec<(String, Kind)>)>,
    open: Option<Open>,
    status: Status,
    /// The palette index the left button paints. The right button always erases,
    /// which is why index 0 is selectable but not very useful.
    ink: usize,
    zoom: f32,
    grid: bool,
    playing: bool,
    /// The speaker, opened on the first play rather than at startup.
    ///
    /// Lazily, for two reasons: opening a device is slow enough to be visible in
    /// a window's first frame, and it can fail — no default device, a machine
    /// with no sound at all — and an editor that refused to start because it
    /// could not find a speaker would be unusable for the half of its job that
    /// is pixels.
    audio: Option<rodio::MixerDeviceSink>,
    /// Set once the device has been tried and failed, so the failure is reported
    /// once instead of retried on every click.
    audio_dead: bool,
}

impl Editor {
    /// Open the editor on a `content/` directory.
    pub fn new(root: PathBuf) -> Self {
        let mut files = Vec::new();
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
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

        Self {
            root,
            files,
            open: None,
            status: Status::None,
            ink: 1,
            zoom: 24.0,
            grid: true,
            playing: false,
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

    fn open_record(&mut self, path: &Path, id: &str, kind: Kind) {
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
                    Doc::Art(Art {
                        sprite: s,
                        seq: 0,
                        frame: 0,
                        dirty: false,
                        pal_dirty: false,
                        undo: Vec::new(),
                    })
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

    /// Render the open sound and send it to the speaker.
    fn play_sound(&mut self) {
        let Some(Open {
            doc: Doc::Sfx(sfx), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        sfx.refresh();
        if sfx.pcm.is_empty() {
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
            sfx.pcm.clone(),
        ));
    }

    /// Write the open record back, as splices.
    fn save(&mut self) {
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

    fn reload(&mut self) {
        if let Some(open) = self.open.as_ref() {
            let (path, id) = (open.path.clone(), open.id().to_string());
            let kind = match open.doc {
                Doc::Art(_) => Kind::Art,
                Doc::Sfx(_) => Kind::Sfx,
            };
            self.open_record(&path, &id, kind);
        }
    }

    // -- panels -------------------------------------------------------------

    fn browser(&mut self, ui: &mut egui::Ui) {
        let mut pick: Option<(PathBuf, String, Kind)> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (path, rows) in &self.files {
                let label = path
                    .strip_prefix(&self.root)
                    .unwrap_or(path)
                    .display()
                    .to_string();
                ui.label(egui::RichText::new(label).strong());
                for (id, kind) in rows {
                    let open_here = self
                        .open
                        .as_ref()
                        .is_some_and(|o| &o.path == path && o.id() == id);
                    let mark = match kind {
                        Kind::Art => "▦",
                        Kind::Sfx => "♪",
                    };
                    if ui
                        .selectable_label(open_here, format!("  {mark} {id}"))
                        .clicked()
                    {
                        pick = Some((path.clone(), id.clone(), *kind));
                    }
                }
                ui.add_space(4.0);
            }
        });
        if let Some((path, id, kind)) = pick {
            self.open_record(&path, &id, kind);
        }
    }

    fn palette(&mut self, ui: &mut egui::Ui) {
        let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        ui.heading("Palette");
        ui.label(
            egui::RichText::new("left button paints · right button erases")
                .small()
                .weak(),
        );
        ui.add_space(4.0);

        for i in 0..art.sprite.pal.len() {
            ui.horizontal(|ui| {
                let selected = self.ink == i;
                if ui
                    .selectable_label(selected, format!("{i}"))
                    .on_hover_text(if i == 0 {
                        "transparent — never painted"
                    } else {
                        "paint with this index"
                    })
                    .clicked()
                {
                    self.ink = i;
                }
                if i == 0 {
                    // Slot 0 is written `.` and is a placeholder, not a colour, so
                    // it gets no colour picker — offering one would invite
                    // authoring a value that is never drawn.
                    ui.label(egui::RichText::new(".  transparent").weak());
                    return;
                }
                let mut rgb = raster::parse_hex(&art.sprite.pal[i]).unwrap_or([255, 0, 255]);
                if ui.color_edit_button_srgb(&mut rgb).changed() {
                    art.sprite.pal[i] = format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2]);
                    art.pal_dirty = true;
                }
                ui.monospace(&art.sprite.pal[i]);
            });
        }

        ui.add_space(6.0);
        let full = art.sprite.pal.len() >= MAX_PAL;
        if ui
            .add_enabled(!full, egui::Button::new("add colour"))
            .on_disabled_hover_text("a frame character is one digit, so nine is the ceiling")
            .clicked()
        {
            art.sprite.pal.push("#ffffff".to_string());
            art.pal_dirty = true;
            self.ink = art.sprite.pal.len() - 1;
        }
    }

    fn sequences(&mut self, ui: &mut egui::Ui) {
        let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        ui.heading("Sequences");
        for (i, seq) in art.sprite.seqs.iter().enumerate() {
            let label = format!("{}  ·  {}  ·  {} fr", seq.state, seq.mode, seq.frames.len());
            if ui.selectable_label(art.seq == i, label).clicked() && art.seq != i {
                art.seq = i;
                art.frame = 0;
                // Undo does not cross sequences: a snapshot is a `Vec<Frame>` and
                // restoring one into a different sequence would replace art that
                // was never edited.
                art.undo.clear();
            }
        }
    }

    /// The sound knobs. Every field the schema has, with its range.
    fn knobs(&mut self, ui: &mut egui::Ui) {
        let Some(Open {
            doc: Doc::Sfx(sfx), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        let p = &mut sfx.sound.params;

        ui.heading("Sound");
        ui.add_space(4.0);

        ui.label("wave");
        ui.horizontal_wrapped(|ui| {
            for w in Wave::ALL {
                if ui.selectable_label(p.wave == w, w.name()).clicked() {
                    p.wave = w;
                }
            }
        });
        ui.add_space(6.0);

        // Logarithmic on the pitches, because the range is 20 Hz to 8 kHz and a
        // linear slider spends nine tenths of its travel above the octave anyone
        // is actually tuning.
        ui.add(
            egui::Slider::new(&mut p.hz, HZ_RANGE.0..=HZ_RANGE.1)
                .logarithmic(true)
                .text("hz"),
        )
        .on_hover_text("starting pitch");
        ui.add(
            egui::Slider::new(&mut p.hz_to, HZ_RANGE.0..=HZ_RANGE.1)
                .logarithmic(true)
                .text("hzTo"),
        )
        .on_hover_text("pitch at the end — equal to hz is no sweep");
        if ui.button("no sweep").on_hover_text("hzTo = hz").clicked() {
            p.hz_to = p.hz;
        }

        ui.add_space(6.0);
        ui.add(
            egui::Slider::new(&mut p.seconds, SECONDS_RANGE.0..=SECONDS_RANGE.1)
                .logarithmic(true)
                .text("seconds"),
        )
        .on_hover_text("total length — this is feedback, not music");
        ui.add(egui::Slider::new(&mut p.attack, 0.0..=1.0).text("attack"))
            .on_hover_text("fraction of the length spent rising — 0 is an impact");
        ui.add(egui::Slider::new(&mut p.release, 0.0..=1.0).text("release"))
            .on_hover_text("fraction of the length spent falling to silence");
        ui.add(egui::Slider::new(&mut p.noise, 0.0..=1.0).text("noise"))
            .on_hover_text("white noise mixed over the oscillator — grit");
        ui.add(egui::Slider::new(&mut p.gain, 0.0..=1.0).text("gain"))
            .on_hover_text("per-sound volume, before the player's own setting");

        ui.add_space(8.0);
        // Which keys a save would touch, spelled out. The rule that keeps these
        // files terse is invisible otherwise, and a designer who cannot see it
        // has no way to know the tool is not about to rewrite the record.
        let changes = sfx.sound.changes();
        if changes.is_empty() {
            ui.label(egui::RichText::new("nothing to write").weak());
        } else {
            ui.label(egui::RichText::new("a save writes:").strong());
            for line in changes.values() {
                ui.monospace(line);
            }
        }
    }

    fn frames(&mut self, ui: &mut egui::Ui) {
        let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        let count = art.sprite.seqs[art.seq].frames.len();
        let mut stop = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Frame");
            for i in 0..count {
                if ui
                    .selectable_label(art.frame == i, format!("{i}"))
                    .clicked()
                {
                    art.frame = i;
                    stop = true;
                }
            }
            ui.separator();
            if ui
                .button("duplicate")
                .on_hover_text("copy this frame in after itself — how an animation grows")
                .clicked()
            {
                let frame = art.sprite.seqs[art.seq].frames[art.frame].clone();
                art.snapshot();
                art.sprite.seqs[art.seq].frames.insert(art.frame + 1, frame);
                art.frame += 1;
                art.dirty = true;
            }
            if ui
                .add_enabled(count > 1, egui::Button::new("delete"))
                .on_disabled_hover_text("a sequence needs at least one frame")
                .clicked()
            {
                art.snapshot();
                art.sprite.seqs[art.seq].frames.remove(art.frame);
                art.frame = art.frame.min(count - 2);
                art.dirty = true;
            }
        });
        if stop {
            self.playing = false;
        }
    }

    /// The waveform scope: what the knobs above it actually produce.
    ///
    /// Drawn from the same samples the speaker is handed, so it is a picture of
    /// the sound rather than a picture of the parameters — the envelope clamp,
    /// the de-click ramps and the clip are all visible in it, and none of them
    /// are visible in the numbers.
    fn scope(&mut self, ui: &mut egui::Ui) {
        let Some(Open {
            doc: Doc::Sfx(sfx), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        sfx.refresh();

        let width = ui.available_width().max(120.0);
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(width, SCOPE_HEIGHT), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(24));

        let mid = rect.center().y;
        painter.line_segment(
            [egui::pos2(rect.min.x, mid), egui::pos2(rect.max.x, mid)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(60)),
        );

        if sfx.pcm.is_empty() {
            return;
        }
        let cols = width as usize;
        let per = (sfx.pcm.len() / cols.max(1)).max(1);
        let half = SCOPE_HEIGHT / 2.0 - 2.0;
        let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(90, 200, 180));
        for c in 0..cols {
            let from = c * per;
            if from >= sfx.pcm.len() {
                break;
            }
            let window = &sfx.pcm[from..(from + per).min(sfx.pcm.len())];
            // Min and max per column, not one sample per column — see
            // `SCOPE_HEIGHT`'s note on why picking one aliases the envelope.
            let lo = window.iter().copied().fold(f32::MAX, f32::min);
            let hi = window.iter().copied().fold(f32::MIN, f32::max);
            let x = rect.min.x + c as f32;
            painter.line_segment(
                [
                    egui::pos2(x, mid - hi * half),
                    egui::pos2(x, mid - lo * half),
                ],
                stroke,
            );
        }

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "{} samples · {:.0} ms · peak {:.2}",
                    sfx.pcm.len(),
                    sfx.sound.params.seconds * 1000.0,
                    synth::peak(&sfx.pcm),
                ))
                .weak()
                .small(),
            );
        });
    }

    /// The canvas. One rectangle per texel, which is the honest way to draw this:
    /// the art is 80 texels for the player and 169 for the largest icon, so a
    /// texture upload per repaint would be more machinery than the drawing costs.
    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (ink, zoom, grid) = (self.ink, self.zoom, self.grid);
        let mut stop = false;

        let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        else {
            return;
        };

        let pal = match art.sprite.colours() {
            Ok(p) => p,
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(220, 90, 90), format!("{e}"));
                return;
            }
        };
        let (tw, th) = (art.sprite.texel_w() as usize, art.sprite.texel_h() as usize);
        let px = zoom;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(tw as f32 * px, th as f32 * px),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);

        for y in 0..th {
            for x in 0..tw {
                let at = egui::Rect::from_min_size(
                    rect.min + egui::vec2(x as f32 * px, y as f32 * px),
                    egui::vec2(px, px),
                );
                let ch = art.glyph(x, y);
                let idx = (ch as u32).wrapping_sub('0' as u32) as usize;
                let fill = if ch == '.' || ch == '0' {
                    // A checkerboard, because transparent has to be visibly
                    // different from black and index 1 in this game is very
                    // nearly black.
                    if (x + y) % 2 == 0 {
                        egui::Color32::from_gray(48)
                    } else {
                        egui::Color32::from_gray(40)
                    }
                } else if idx < pal.len() {
                    let c = pal[idx];
                    egui::Color32::from_rgb(c[0], c[1], c[2])
                } else {
                    // An index the palette does not have. Loud rather than
                    // skipped: it is exactly what the raster would refuse, and the
                    // canvas is where it can be pointed at.
                    egui::Color32::from_rgb(255, 0, 255)
                };
                painter.rect_filled(at, 0.0, fill);
            }
        }

        if grid {
            let line = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(60));
            // A heavier line every `grain` texels: that is one world CELL, and the
            // cell is the unit the simulation thinks in. Drawing it is what keeps
            // a grain-2 sprite honest about the grid it stands on.
            let cell = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(140));
            for x in 0..=tw {
                let s = if (x as u32).is_multiple_of(art.sprite.grain) {
                    cell
                } else {
                    line
                };
                let at = rect.min.x + x as f32 * px;
                painter.line_segment([egui::pos2(at, rect.min.y), egui::pos2(at, rect.max.y)], s);
            }
            for y in 0..=th {
                let s = if (y as u32).is_multiple_of(art.sprite.grain) {
                    cell
                } else {
                    line
                };
                let at = rect.min.y + y as f32 * px;
                painter.line_segment([egui::pos2(rect.min.x, at), egui::pos2(rect.max.x, at)], s);
            }
        }

        if response.drag_started() || response.clicked() {
            art.snapshot();
            stop = true;
        }
        if response.is_pointer_button_down_on() || response.clicked() {
            let erase = ui.input(|i| i.pointer.secondary_down());
            if let Some(pos) = response.interact_pointer_pos() {
                let x = ((pos.x - rect.min.x) / px).floor();
                let y = ((pos.y - rect.min.y) / px).floor();
                if x >= 0.0 && y >= 0.0 && (x as usize) < tw && (y as usize) < th {
                    let ch = if erase {
                        '.'
                    } else {
                        // `ink` is an index and a frame character is its digit.
                        char::from_digit(ink as u32, 10).unwrap_or('.')
                    };
                    art.paint(x as usize, y as usize, ch);
                }
            }
        }

        if stop {
            self.playing = false;
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let has = self.open.is_some();
            let dirty = self.open.as_ref().is_some_and(Open::dirty);
            let is_art = matches!(self.open.as_ref().map(|o| &o.doc), Some(Doc::Art(_)));

            if ui
                .add_enabled(dirty, egui::Button::new("Save"))
                .on_hover_text("splice the edit back — every other byte is kept")
                .clicked()
            {
                self.save();
            }
            if ui
                .add_enabled(has, egui::Button::new("Reload"))
                .on_hover_text("throw the edit away and re-read the file")
                .clicked()
            {
                self.reload();
            }

            if is_art {
                let can_undo = matches!(
                    self.open.as_ref().map(|o| &o.doc),
                    Some(Doc::Art(a)) if !a.undo.is_empty()
                );
                if ui
                    .add_enabled(can_undo, egui::Button::new("Undo"))
                    .clicked()
                    && let Some(Open {
                        doc: Doc::Art(art), ..
                    }) = self.open.as_mut()
                {
                    art.undo();
                }
                ui.separator();
                ui.add(egui::Slider::new(&mut self.zoom, 4.0..=48.0).text("zoom"));
                ui.checkbox(&mut self.grid, "grid");
                if ui
                    .add_enabled(
                        true,
                        egui::Button::new(if self.playing { "Stop" } else { "Play" }),
                    )
                    .on_hover_text("step the sequence at its own rate")
                    .clicked()
                {
                    self.playing = !self.playing;
                }
            } else if has {
                ui.separator();
                if ui
                    .button("▶ Play")
                    .on_hover_text("synthesise these parameters and listen (space)")
                    .clicked()
                {
                    self.play_sound();
                }
            }

            if let Some(open) = self.open.as_ref() {
                ui.separator();
                let what = match &open.doc {
                    Doc::Art(a) => format!(
                        "{}  ·  {}x{} cells{}",
                        a.sprite.name,
                        a.sprite.cells_w,
                        a.sprite.cells_h,
                        if a.sprite.grain > 1 {
                            format!("  ·  grain {}", a.sprite.grain)
                        } else {
                            String::new()
                        },
                    ),
                    Doc::Sfx(s) => format!("{}  ·  {}", s.sound.name, s.sound.params.wave.name()),
                };
                ui.label(format!("{what}{}", if dirty { "  ·  unsaved" } else { "" }));
            }
        });
    }
}

impl eframe::App for Editor {
    // `ui`, not `update`: eframe 0.35 hands the app a `Ui` for the whole window
    // rather than a `Context`, and panels are shown INSIDE it. The context is
    // still reachable through it, and is cloned here because the panel closures
    // below borrow `ui` mutably.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Keyboard: the three an editor is unusable without.
        let (undo, save, play) = ctx.input(|i| {
            (
                i.modifiers.command && i.key_pressed(egui::Key::Z),
                i.modifiers.command && i.key_pressed(egui::Key::S),
                i.key_pressed(egui::Key::Space),
            )
        });
        if undo
            && let Some(Open {
                doc: Doc::Art(art), ..
            }) = self.open.as_mut()
        {
            art.undo();
        }
        if save {
            self.save();
        }
        if play && matches!(self.open.as_ref().map(|o| &o.doc), Some(Doc::Sfx(_))) {
            self.play_sound();
        }

        if self.playing
            && let Some(Open {
                doc: Doc::Art(art), ..
            }) = self.open.as_mut()
        {
            let seq = &art.sprite.seqs[art.seq];
            let rate = seq.rate(art.sprite.fps).max(0.1);
            let n = seq.frames.len();
            if n > 1 {
                let t = ctx.input(|i| i.time);
                art.frame = ((t * rate as f64) as usize) % n;
                // Repaint at roughly the sequence's own rate rather than as fast
                // as the screen will go: the point is to see the animation at the
                // speed the game plays it.
                ctx.request_repaint_after(std::time::Duration::from_secs_f32(1.0 / rate));
            }
        }

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));

        egui::Panel::left("browser")
            .default_size(240.0)
            .show(ui, |ui| {
                ui.heading("content/");
                self.browser(ui);
            });

        if matches!(self.open.as_ref().map(|o| &o.doc), Some(Doc::Art(_))) {
            egui::Panel::right("tools")
                .default_size(240.0)
                .show(ui, |ui| {
                    self.palette(ui);
                    ui.separator();
                    self.sequences(ui);
                });
        }

        egui::Panel::bottom("status").show(ui, |ui| {
            match &self.status {
                Status::None => {
                    ui.label(
                        egui::RichText::new(
                            "ids.lock.json is the compiler's — this tool never writes it",
                        )
                        .weak(),
                    );
                }
                Status::Note(s) => {
                    ui.label(s);
                }
                Status::Bad(s) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 90, 90), s);
                }
            };
        });

        egui::CentralPanel::default().show(ui, |ui| match self.open.as_ref().map(|o| &o.doc) {
            None => {
                ui.centered_and_justified(|ui| {
                    ui.label("pick a record on the left");
                });
            }
            Some(Doc::Art(_)) => {
                self.frames(ui);
                ui.separator();
                egui::ScrollArea::both().show(ui, |ui| self.canvas(ui));
            }
            Some(Doc::Sfx(_)) => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.scope(ui);
                    ui.separator();
                    self.knobs(ui);
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content() -> PathBuf {
        Editor::find_content(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("the repo has one")
    }

    fn open_player() -> (Art, String) {
        let text =
            std::fs::read_to_string(content().join("sprites/player.toml")).expect("readable");
        let sprite = sprite::read(&text, "player").expect("reads");
        (
            Art {
                sprite,
                seq: 0,
                frame: 0,
                dirty: false,
                pal_dirty: false,
                undo: Vec::new(),
            },
            text,
        )
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
    fn painting_reports_whether_anything_moved() {
        // A drag sends the same texel many times, and marking the file dirty on
        // a no-op would offer to save a file nothing changed in.
        let (mut art, _) = open_player();
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
        let (mut art, _) = open_player();
        let before = art.sprite.seqs[0].frames.clone();
        art.snapshot();
        art.paint(0, 0, '1');
        art.paint(1, 1, '2');
        art.undo();
        assert_eq!(art.sprite.seqs[0].frames, before);
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
