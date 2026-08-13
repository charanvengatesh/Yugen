//! Creating a record — the one write that adds rather than replaces.
//!
//! # It picks a file; it never makes one
//!
//! FORMAT.md §9's fourth clause puts a new record in the file
//! `content/LAYOUT.toml` names for it, and `xtask`'s layout gate fails on a file
//! no row claims. So the target is chosen from the files the browser already
//! found, and there is no "new file" button: adding a file means adding a
//! LAYOUT row, which is a decision about how the content is organised and not
//! one a dialog should make on somebody's behalf.
//!
//! # The id has to be unique in the DIRECTORY, not the file
//!
//! `contentc` concatenates every `*.toml` under `content/mobs/`, so two files
//! there cannot both hold a `grubling`. Checking only the target file would let
//! the editor create a record that compiles nowhere, and the error would arrive
//! from the compiler with no idea which of the two was the new one.
//!
//! # It still does not touch the lock
//!
//! `content/ids.lock.json` is the compiler's, for the reason `app.rs` gives: the
//! lock is what keeps new content above the baseline boundary instead of
//! renumbering what is below it. This writes a record and says what to run.

use std::collections::BTreeSet;
use std::path::Path;

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Kind, Status};
use crate::procgen::sprite as gensprite;
use crate::splice::{append_record, record_ids};
use crate::synth::Params;
use crate::template;

/// What the dialog is holding while it is open.
pub(crate) struct NewState {
    pub(crate) open: bool,
    pub(crate) kind: Kind,
    pub(crate) id: String,
    pub(crate) name: String,
    /// Index into [`Editor::files`], not a path: the list is rebuilt after a
    /// create, and an index into the list the user actually picked from is what
    /// the combo box is showing.
    pub(crate) file: usize,
    /// Whether a new sprite starts from a generator roll or from an empty grid.
    pub(crate) generate: bool,
}

impl Default for NewState {
    fn default() -> Self {
        NewState {
            open: false,
            kind: Kind::Art,
            id: String::new(),
            name: String::new(),
            file: 0,
            generate: true,
        }
    }
}

impl Editor {
    /// Every id already taken anywhere in the same kind directory.
    fn ids_in_kind_dir(&self, target: &Path) -> BTreeSet<String> {
        let Some(dir) = target.parent() else {
            return BTreeSet::new();
        };
        let mut out = BTreeSet::new();
        for (path, _) in &self.files {
            if path.parent() == Some(dir)
                && let Ok(text) = std::fs::read_to_string(path)
            {
                out.extend(record_ids(&text));
            }
        }
        out
    }

    pub(super) fn new_record_dialog(&mut self, ctx: &egui::Context) {
        if !self.new_rec.open {
            return;
        }
        if self.files.is_empty() {
            self.new_rec.open = false;
            self.status = Status::Bad("no content files to add a record to".into());
            return;
        }
        self.new_rec.file = self.new_rec.file.min(self.files.len() - 1);

        let target = self.files[self.new_rec.file].0.clone();
        let taken = self.ids_in_kind_dir(&target);
        let label = target
            .strip_prefix(&self.root)
            .unwrap_or(&target)
            .display()
            .to_string();

        let mut create = false;
        let mut close = false;
        egui::Window::new("New record")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("kind");
                    for (k, t) in [(Kind::Art, "▦ sprite"), (Kind::Sfx, "♪ sound")] {
                        if ui.selectable_label(self.new_rec.kind == k, t).clicked() {
                            self.new_rec.kind = k;
                        }
                    }
                });

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("into");
                    egui::ComboBox::from_id_salt("newfile")
                        .selected_text(&label)
                        .width(280.0)
                        .show_ui(ui, |ui| {
                            for (i, (path, _)) in self.files.iter().enumerate() {
                                let t = path
                                    .strip_prefix(&self.root)
                                    .unwrap_or(path)
                                    .display()
                                    .to_string();
                                ui.selectable_value(&mut self.new_rec.file, i, t);
                            }
                        });
                });
                ui.label(
                    egui::RichText::new(
                        "an existing file — a new one needs a LAYOUT.toml row, \
                         which is a decision about how content is filed",
                    )
                    .small()
                    .weak(),
                );

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("id  ");
                    ui.add(egui::TextEdit::singleline(&mut self.new_rec.id).desired_width(200.0));
                });
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.add(egui::TextEdit::singleline(&mut self.new_rec.name).desired_width(200.0));
                });

                if self.new_rec.kind == Kind::Art {
                    ui.add_space(4.0);
                    ui.checkbox(&mut self.new_rec.generate, "start from a generator roll")
                        .on_hover_text("otherwise the grid starts empty");
                }

                ui.separator();
                let id = self.new_rec.id.trim().to_string();
                let problem = if id.is_empty() {
                    Some("give it an id".to_string())
                } else if !template::is_legal_id(&id) {
                    Some(
                        "lowercase letters, digits and underscores; must start with a letter"
                            .into(),
                    )
                } else if taken.contains(&id) {
                    // Named rather than merely refused: the clash may be in a
                    // different file, which is the case a per-file check misses.
                    Some(format!(
                        "`{id}` is already taken in {} — ids are unique per kind, not per file",
                        target
                            .parent()
                            .and_then(|p| p.file_name())
                            .map_or_else(|| "this directory".into(), |n| n.to_string_lossy())
                    ))
                } else {
                    None
                };

                match &problem {
                    Some(why) => {
                        ui.colored_label(egui::Color32::from_rgb(220, 90, 90), why);
                    }
                    None => {
                        ui.label(format!("appends [{id}] to the bottom of {label}"));
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(problem.is_none(), egui::Button::new("create"))
                        .clicked()
                    {
                        create = true;
                    }
                    if ui.button("cancel").clicked() {
                        close = true;
                    }
                });
            });

        if create {
            self.create_record();
            close = true;
        }
        if close {
            self.new_rec.open = false;
        }
    }

    /// Compose the record, append it, and open it.
    fn create_record(&mut self) {
        let target = self.files[self.new_rec.file].0.clone();
        let id = self.new_rec.id.trim().to_string();
        let name = {
            let n = self.new_rec.name.trim();
            if n.is_empty() {
                id.clone()
            } else {
                n.to_string()
            }
        };

        let (record, kind) = match self.new_rec.kind {
            Kind::Art => {
                let mut recipe = self.recipe;
                recipe.w = 8;
                recipe.h = 8;
                let (frames, pal, seed) = if self.new_rec.generate {
                    recipe.seed = crate::rng::fresh_seed(&mut self.seed_counter);
                    (
                        vec![gensprite::generate(&recipe)],
                        gensprite::generate_pal(
                            &recipe,
                            crate::procgen::palette::to_hsl(self.base_rgb),
                        ),
                        Some(recipe.seed),
                    )
                } else {
                    (
                        vec![vec![".".repeat(8); 8]],
                        vec![".".to_string(), "#ffffff".to_string()],
                        None,
                    )
                };
                (
                    template::sprite_record(&id, &name, (8, 8), &pal, &frames, seed),
                    Kind::Art,
                )
            }
            Kind::Sfx => (
                template::sound_record(&id, &name, &Params::default(), None),
                Kind::Sfx,
            ),
        };

        // Read at the last moment and write immediately: this is the same
        // stale-window `Editor::save` guards, and the file was listed at startup
        // rather than just now.
        let Ok(text) = std::fs::read_to_string(&target) else {
            self.status = Status::Bad(format!("{}: cannot read", target.display()));
            return;
        };
        let next = append_record(&text, &record);

        let tmp = target.with_extension("toml.tmp");
        if let Err(e) = std::fs::write(&tmp, &next) {
            self.status = Status::Bad(format!("{}: {e}", tmp.display()));
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &target) {
            self.status = Status::Bad(format!("{}: {e}", target.display()));
            return;
        }

        // The file gained a record, so the list the browser draws is stale.
        self.files = crate::app::scan(&self.root);
        self.open_record(&target, &id, kind);
        self.status = Status::Note(format!(
            "added [{id}] to {} — run `cargo run -p contentc` to assign its code",
            target.display()
        ));
    }
}
