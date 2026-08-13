//! The window: which panels exist, where they sit, and what the keyboard does.
//!
//! # Panels are methods, not a registry
//!
//! Every panel below is an inherent method on [`Editor`] living in its own
//! module. Rust lets an `impl` block sit in any module of the crate, so the split
//! costs nothing at the call site and buys the thing that matters: a panel is a
//! file, and adding one is a new file plus one line here rather than a hundred
//! lines into the middle of something else.
//!
//! There is deliberately no trait and no registry. A registry pays for itself
//! when the set of panels is open — plugins, user extensions, panels discovered
//! at runtime — and this set is closed and small. What it would cost is real:
//! every panel would have to be reachable through the same interface, and they
//! are not alike. `canvas` needs the pointer, `scope` needs the PCM, `browser`
//! needs the file list and none of them need each other.
//!
//! # No panel writes a file
//!
//! Panels mutate the open document and set the status. The disk belongs to
//! [`crate::app`], for the reason its header gives: the stale check and the
//! temp-and-rename have to happen on every write, and one function is how that
//! stops being a thing to remember.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open, Status};

mod browser;
mod canvas;
mod frames;
mod genart;
mod gensfx;
mod knobs;
mod newrecord;
mod palette;
mod resize;
mod scope;
mod sequences;
mod toolbar;

pub(crate) use gensfx::Rolls;
pub(crate) use newrecord::NewState;
pub(crate) use resize::ResizeState;

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

        // Before the panels, so a modal that is up draws over them rather than
        // under.
        self.resize_dialog(&ctx);
        self.new_record_dialog(&ctx);

        egui::Panel::left("browser")
            .default_size(240.0)
            .show(ui, |ui| {
                ui.heading("content/");
                self.browser(ui);
            });

        // The right-hand panel is the tools for whatever is open: a palette and
        // the generators for art, the roll list for a sound.
        if self.open.is_some() {
            egui::Panel::right("tools")
                .default_size(260.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if matches!(self.open.as_ref().map(|o| &o.doc), Some(Doc::Art(_))) {
                            self.palette(ui);
                            ui.separator();
                            self.sequences(ui);
                            ui.separator();
                            self.genart(ui);
                        } else {
                            self.gensfx(ui);
                        }
                    });
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
