//! The top row: save, reload, undo, and what is open.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};

impl Editor {
    pub(super) fn toolbar(&mut self, ui: &mut egui::Ui) {
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
                .button("New…")
                .on_hover_text("append a record to a file that already exists")
                .clicked()
            {
                self.new_rec.open = true;
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
                if ui
                    .button("Resize…")
                    .on_hover_text("change cellsW/cellsH — says what it would cost first")
                    .clicked()
                {
                    self.resize.open = true;
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
