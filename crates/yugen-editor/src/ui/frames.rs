//! The filmstrip: which frame is being drawn, and how a sequence grows.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};

impl Editor {
    pub(super) fn frames(&mut self, ui: &mut egui::Ui) {
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
}
