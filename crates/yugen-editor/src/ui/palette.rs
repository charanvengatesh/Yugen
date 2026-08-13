//! The colour picker, and the ceiling it enforces.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};
use crate::raster;

/// Palette indices `1`-`9` are the only ones a frame character can name, so the
/// list tops out at ten entries including the transparent slot. FORMAT.md §5 is
/// where that ceiling comes from: a frame is digits, and there are nine of them.
pub(crate) const MAX_PAL: usize = 10;

impl Editor {
    pub(super) fn palette(&mut self, ui: &mut egui::Ui) {
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
}
