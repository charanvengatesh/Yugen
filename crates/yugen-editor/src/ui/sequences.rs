//! The pose list. One entry per `[[id.seq]]`, in file order.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};

impl Editor {
    pub(super) fn sequences(&mut self, ui: &mut egui::Ui) {
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
                // Undo does not cross sequences: a snapshot holds one sequence's
                // frames and restoring it into a different sequence would replace
                // art that was never edited.
                art.undo.clear();
            }
        }
    }
}
