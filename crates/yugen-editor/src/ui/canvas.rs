//! The pixel grid: one rectangle per texel, and the cell lines over it.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open, Tool};
use crate::procgen::ops;

impl Editor {
    /// The canvas. One rectangle per texel, which is the honest way to draw this:
    /// the art is 80 texels for the player and 169 for the largest icon, so a
    /// texture upload per repaint would be more machinery than the drawing costs.
    pub(super) fn canvas(&mut self, ui: &mut egui::Ui) {
        let (ink, zoom, grid, tool) = (self.ink, self.zoom, self.grid, self.tool);
        let mut stop = false;

        let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        else {
            return;
        };

        let pal = match art.sprite.colours() {
            Ok(p) => p,
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(220, 90, 90), format!("{e}"));
                return;
            }
        };
        let (tw, th) = (art.sprite.texel_w() as usize, art.sprite.texel_h() as usize);
        let px = zoom;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(tw as f32 * px, th as f32 * px),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);

        for y in 0..th {
            for x in 0..tw {
                let at = egui::Rect::from_min_size(
                    rect.min + egui::vec2(x as f32 * px, y as f32 * px),
                    egui::vec2(px, px),
                );
                let ch = art.glyph(x, y);
                let idx = (ch as u32).wrapping_sub('0' as u32) as usize;
                let fill = if ch == '.' || ch == '0' {
                    // A checkerboard, because transparent has to be visibly
                    // different from black and index 1 in this game is very
                    // nearly black.
                    if (x + y) % 2 == 0 {
                        egui::Color32::from_gray(48)
                    } else {
                        egui::Color32::from_gray(40)
                    }
                } else if idx < pal.len() {
                    let c = pal[idx];
                    egui::Color32::from_rgb(c[0], c[1], c[2])
                } else {
                    // An index the palette does not have. Loud rather than
                    // skipped: it is exactly what the raster would refuse, and the
                    // canvas is where it can be pointed at.
                    egui::Color32::from_rgb(255, 0, 255)
                };
                painter.rect_filled(at, 0.0, fill);
            }
        }

        if grid {
            let line = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(60));
            // A heavier line every `grain` texels: that is one world CELL, and the
            // cell is the unit the simulation thinks in. Drawing it is what keeps
            // a grain-2 sprite honest about the grid it stands on.
            let cell = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(140));
            for x in 0..=tw {
                let s = if (x as u32).is_multiple_of(art.sprite.grain) {
                    cell
                } else {
                    line
                };
                let at = rect.min.x + x as f32 * px;
                painter.line_segment([egui::pos2(at, rect.min.y), egui::pos2(at, rect.max.y)], s);
            }
            for y in 0..=th {
                let s = if (y as u32).is_multiple_of(art.sprite.grain) {
                    cell
                } else {
                    line
                };
                let at = rect.min.y + y as f32 * px;
                painter.line_segment([egui::pos2(rect.min.x, at), egui::pos2(rect.max.x, at)], s);
            }
        }

        if response.drag_started() || response.clicked() {
            art.snapshot();
            stop = true;
        }
        if response.is_pointer_button_down_on() || response.clicked() {
            let erase = ui.input(|i| i.pointer.secondary_down());
            if let Some(pos) = response.interact_pointer_pos() {
                let x = ((pos.x - rect.min.x) / px).floor();
                let y = ((pos.y - rect.min.y) / px).floor();
                if x >= 0.0 && y >= 0.0 && (x as usize) < tw && (y as usize) < th {
                    let (x, y) = (x as usize, y as usize);
                    let ch = if erase {
                        '.'
                    } else {
                        // `ink` is an index and a frame character is its digit.
                        char::from_digit(ink as u32, 10).unwrap_or('.')
                    };
                    match tool {
                        Tool::Paint => {
                            art.paint(x, y, ch);
                        }
                        // Only on the press. A fill repeated for every frame a
                        // drag is held would be a no-op each time — the region
                        // is already the new colour — but it would spend a
                        // whole flood per repaint to discover that.
                        Tool::Flood if response.clicked() => {
                            // The snapshot was already taken above, so this
                            // writes through rather than calling `put_current`
                            // and stacking a second undo step for one click.
                            let out =
                                ops::flood(&art.sprite.seqs[art.seq].frames[art.frame], x, y, ch);
                            if out != art.sprite.seqs[art.seq].frames[art.frame] {
                                art.sprite.seqs[art.seq].frames[art.frame] = out;
                                art.dirty = true;
                            }
                        }
                        Tool::Flood => {}
                    }
                }
            }
        }

        if stop {
            self.playing = false;
        }
    }
}
