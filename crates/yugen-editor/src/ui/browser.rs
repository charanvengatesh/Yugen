//! The file list: every record in `content/` this tool can open.

use std::path::PathBuf;

use eframe::egui;

use crate::app::Editor;
use crate::doc::Kind;

impl Editor {
    pub(super) fn browser(&mut self, ui: &mut egui::Ui) {
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
}
