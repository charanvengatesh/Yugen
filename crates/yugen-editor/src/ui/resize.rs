//! The resize dialog: the one edit that moves `cellsW` and `cellsH`.
//!
//! # It refuses before it warns
//!
//! Two of the checks here guard `MobDef::build`'s assertions, which fire as
//! panics at load rather than as errors anybody can read — see
//! [`crate::sprite::body_cells`] for why that is the editor's problem. Those two
//! disable the button.
//!
//! The third, losing ink, only warns: cropping is sometimes exactly what is
//! wanted, and the tool's job there is to say *which frames* so the redraw has
//! a list.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open, Status};
use crate::procgen::resize::{Anchor, Mode, resize_sprite};
use crate::sprite;

/// What the dialog is holding while it is open.
pub(crate) struct ResizeState {
    pub(crate) open: bool,
    pub(crate) cells_w: u32,
    pub(crate) cells_h: u32,
    pub(crate) anchor: Anchor,
    pub(crate) mode: Mode,
}

impl Default for ResizeState {
    fn default() -> Self {
        // 8x8 is the default because it is the grid the whole tree is moving to,
        // and because a dialog that opened on the record's current size would
        // make the common case two extra clicks.
        ResizeState {
            open: false,
            cells_w: 8,
            cells_h: 8,
            anchor: Anchor::default(),
            mode: Mode::default(),
        }
    }
}

impl Editor {
    pub(super) fn resize_dialog(&mut self, ctx: &egui::Context) {
        if !self.resize.open {
            return;
        }
        let Some(Open {
            doc: Doc::Art(art),
            path,
            text,
        }) = self.open.as_ref()
        else {
            self.resize.open = false;
            return;
        };

        let id = art.sprite.id.clone();
        let from = (art.sprite.cells_w, art.sprite.cells_h);
        let grain = art.sprite.grain;
        let body = sprite::body_cells(text, &id);
        let _ = path;

        // What the resize would cost, computed live so the numbers move with the
        // spinners rather than appearing after the fact.
        let (_, report) = resize_sprite(
            &art.sprite,
            (self.resize.cells_w, self.resize.cells_h),
            self.resize.anchor,
            self.resize.mode,
        );

        let mut apply = false;
        let mut close = false;
        egui::Window::new("Resize")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(format!(
                    "[{id}] is {}x{} cells{}",
                    from.0,
                    from.1,
                    if grain > 1 {
                        format!(" at grain {grain}")
                    } else {
                        String::new()
                    }
                ));
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label("to");
                    ui.add(egui::DragValue::new(&mut self.resize.cells_w).range(1..=64));
                    ui.label("x");
                    ui.add(egui::DragValue::new(&mut self.resize.cells_h).range(1..=64));
                    ui.label("cells");
                });
                if ui.button("8 x 8").clicked() {
                    self.resize.cells_w = 8;
                    self.resize.cells_h = 8;
                }

                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label("anchor");
                    for a in Anchor::ALL {
                        if ui
                            .selectable_label(self.resize.anchor == a, a.name())
                            .clicked()
                        {
                            self.resize.anchor = a;
                        }
                    }
                });
                ui.label(
                    egui::RichText::new(
                        "feet-centred is where both hosts already draw the art \
                         relative to the body — any other anchor moves the picture \
                         against the collision box",
                    )
                    .small()
                    .weak(),
                );

                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label("fit");
                    for m in Mode::ALL {
                        if ui
                            .selectable_label(self.resize.mode == m, m.name())
                            .clicked()
                        {
                            self.resize.mode = m;
                        }
                    }
                });
                ui.label(
                    egui::RichText::new(
                        "pad keeps the creature exactly the size it is drawn; \
                         scale draws it bigger than the box the player can hit",
                    )
                    .small()
                    .weak(),
                );

                ui.separator();

                // -- the guards ----------------------------------------------
                let mut blocked: Option<String> = None;
                if let Some((bw, bh)) = body {
                    if self.resize.cells_w < bw || self.resize.cells_h < bh {
                        blocked = Some(format!(
                            "art cannot be smaller than the body box ({bw}x{bh}) — \
                             MobDef::build asserts it and the game would panic at load"
                        ));
                    } else if !(self.resize.cells_w - bw).is_multiple_of(2) {
                        blocked = Some(format!(
                            "width minus the body's {bw} must be even — the overhang is \
                             centred, and MobDef::build asserts that too"
                        ));
                    }
                }

                if let Some(why) = &blocked {
                    ui.colored_label(egui::Color32::from_rgb(220, 90, 90), why);
                } else if report.lossless() {
                    ui.label(format!(
                        "{} frames, nothing lost — the drawing is unchanged and only \
                         the canvas grows",
                        report.frames
                    ));
                } else {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 170, 70),
                        format!(
                            "{} of {} frames lose ink ({} texels) — these need redrawing:",
                            report.lost.len(),
                            report.frames,
                            report.total_lost()
                        ),
                    );
                    egui::ScrollArea::vertical()
                        .max_height(120.0)
                        .show(ui, |ui| {
                            for &(si, fi, n) in &report.lost {
                                let pose =
                                    art.sprite.seqs.get(si).map_or("?", |s| s.state.as_str());
                                ui.monospace(format!("  {pose} frame {fi} — {n} texels"));
                            }
                        });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(blocked.is_none(), egui::Button::new("resize"))
                        .clicked()
                    {
                        apply = true;
                    }
                    if ui.button("cancel").clicked() {
                        close = true;
                    }
                });
            });

        if apply {
            self.apply_resize();
            close = true;
        }
        if close {
            self.resize.open = false;
        }
    }

    /// Do the resize, snapshotting so it can be taken back.
    ///
    /// Undo restores the grid as well as the frames — that is what
    /// [`crate::doc::Snapshot`] carries the dimensions for, and without it an
    /// undo here would leave frames of the new shape inside a record declaring
    /// the old one.
    fn apply_resize(&mut self) {
        let (cells, anchor, mode) = (
            (self.resize.cells_w, self.resize.cells_h),
            self.resize.anchor,
            self.resize.mode,
        );
        let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        else {
            return;
        };
        let (out, report) = resize_sprite(&art.sprite, cells, anchor, mode);
        art.snapshot();
        art.sprite = out;
        art.frame = art
            .frame
            .min(art.sprite.seqs[art.seq].frames.len().saturating_sub(1));
        art.dirty = true;
        art.head_dirty = true;

        self.status = if report.lossless() {
            Status::Note(format!(
                "resized to {}x{} — {} frames, nothing lost",
                cells.0, cells.1, report.frames
            ))
        } else {
            Status::Note(format!(
                "resized to {}x{} — {} frame(s) lost {} texels and need redrawing",
                cells.0,
                cells.1,
                report.lost.len(),
                report.total_lost()
            ))
        };
    }
}
