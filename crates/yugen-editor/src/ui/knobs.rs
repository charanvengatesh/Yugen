//! The sound knobs. Every field the schema has, with its range.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};
use crate::sound::{HZ_RANGE, LOWPASS_RANGE, REPEAT_RANGE, SECONDS_RANGE, VIBRATO_HZ_RANGE};
use crate::synth::Wave;

impl Editor {
    pub(super) fn knobs(&mut self, ui: &mut egui::Ui) {
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
        // Collapsed by default, because these four are the ones a sound does not
        // need. Eleven sliders open at once reads as eleven decisions to make;
        // eight and a fold reads as the sound, plus shaping if you want it.
        egui::CollapsingHeader::new("Shaping")
            .default_open(p.vibrato > 0.0 || p.repeat_hz > 0.0 || p.lowpass > 0.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("every one of these is off at 0")
                        .small()
                        .weak(),
                );
                ui.add(egui::Slider::new(&mut p.vibrato, 0.0..=1.0).text("vibrato"))
                    .on_hover_text("wobble depth as a fraction of the pitch — 0 is none");
                ui.add(
                    egui::Slider::new(&mut p.vibrato_hz, 0.0..=VIBRATO_HZ_RANGE.1)
                        .text("vibratoHz"),
                )
                .on_hover_text("wobble rate — 0 is none, and depth alone does nothing");
                if p.vibrato > 0.0 && p.vibrato_hz == 0.0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 170, 70),
                        "depth with no rate is silent — set vibratoHz too",
                    );
                }

                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut p.repeat_hz, 0.0..=REPEAT_RANGE.1).text("repeatHz"))
                    .on_hover_text("how often the envelope and sweep restart — 0 plays once");
                ui.add(
                    egui::Slider::new(&mut p.lowpass, 0.0..=LOWPASS_RANGE.1)
                        .logarithmic(true)
                        .text("lowpass"),
                )
                .on_hover_text("one-pole cutoff — 0 is BYPASS, not silence");
            });

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
}
