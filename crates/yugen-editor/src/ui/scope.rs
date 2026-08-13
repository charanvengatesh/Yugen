//! The waveform scope: what the knobs above it actually produce.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};
use crate::synth;

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

impl Editor {
    /// Drawn from the same samples the speaker is handed, so it is a picture of
    /// the sound rather than a picture of the parameters — the envelope clamp,
    /// the de-click ramps and the clip are all visible in it, and none of them
    /// are visible in the numbers.
    pub(super) fn scope(&mut self, ui: &mut egui::Ui) {
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
}
