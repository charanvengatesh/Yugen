//! The generate panel: a recipe, a seed, and buttons that transform a frame.
//!
//! # Everything here goes through `Art::put_*`
//!
//! A generator replaces a whole frame or a whole sequence, which is the most
//! destructive thing this editor can do to a drawing — far more than a stroke.
//! So nothing in this file writes to `sprite.seqs` directly; every path goes
//! through [`crate::doc::Art::put_current`] or [`crate::doc::Art::put_frames`],
//! which snapshot first. "Remembered to snapshot" is then a property of two
//! functions rather than of twenty buttons.
//!
//! # The seed is shown because it IS the drawing
//!
//! A reroll draws a fresh seed and displays it in hex. That is not decoration:
//! [`crate::procgen::sprite::generate`] is pure in the seed, so those four bytes
//! are the whole drawing, and a designer who liked roll `3f2a91c4` gets it back
//! by typing it in.
//!
//! # Why every button reads its inputs into locals first
//!
//! The helpers below take `&mut self` and a closure. A closure that captured
//! `self` to read a slider would borrow it twice, so each button copies what it
//! needs out first. It reads as ceremony and it is what keeps the snapshot in
//! one place instead of inlining `put_current` at every call site.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open, Tool};
use crate::procgen::palette::{self, Hsl};
use crate::procgen::sprite::{self as gensprite, Symmetry};
use crate::procgen::{anim, ops};
use crate::sprite::Frame;

impl Editor {
    pub(super) fn genart(&mut self, ui: &mut egui::Ui) {
        // The recipe follows the record's own grid rather than being a knob.
        // Generating at a size the record does not declare is the one mistake
        // this panel must not allow: `contentc` would compile it and the game
        // would panic at `PreStartup`.
        let Some((tw, th, pal_len)) = self.art_shape() else {
            return;
        };
        self.recipe.w = tw;
        self.recipe.h = th;

        egui::CollapsingHeader::new("Generate")
            .default_open(true)
            .show(ui, |ui| self.generate_section(ui));
        egui::CollapsingHeader::new("Operators")
            .default_open(true)
            .show(ui, |ui| self.operators_section(ui, pal_len));
        egui::CollapsingHeader::new("Colours")
            .default_open(false)
            .show(ui, |ui| self.colours_section(ui));
        egui::CollapsingHeader::new("Animate")
            .default_open(false)
            .show(ui, |ui| self.animate_section(ui, pal_len));
    }

    /// The open art's texel grid and palette length, or `None` if no art is open.
    fn art_shape(&self) -> Option<(usize, usize, usize)> {
        match self.open.as_ref() {
            Some(Open {
                doc: Doc::Art(art), ..
            }) => Some((
                art.sprite.texel_w() as usize,
                art.sprite.texel_h() as usize,
                art.sprite.pal.len(),
            )),
            _ => None,
        }
    }

    /// The palette base the ramp generators walk from.
    fn base_hsl(&self) -> Hsl {
        palette::to_hsl(self.base_rgb)
    }

    fn generate_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("symmetry");
            for s in Symmetry::ALL {
                if ui
                    .selectable_label(self.recipe.symmetry == s, s.name())
                    .clicked()
                {
                    self.recipe.symmetry = s;
                }
            }
        });
        ui.add(
            egui::Slider::new(&mut self.recipe.density, 0.2..=0.8)
                .text("density")
                .fixed_decimals(2),
        )
        .on_hover_text("how much of the half-grid starts filled — 0.4 to 0.65 is the useful band");
        ui.add(egui::Slider::new(&mut self.recipe.inks, 2..=palette::MAX_STEPS).text("inks"))
            .on_hover_text("how many palette steps the shading uses");
        ui.checkbox(&mut self.recipe.outline, "outline")
            .on_hover_text("draw index 1 around the silhouette");

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("seed");
            // Hex, and editable. The seed IS the drawing — see the module header
            // — so it has to be readable off the screen and typeable back in,
            // not hidden behind the button that drew it.
            let mut text = format!("{:08x}", self.recipe.seed);
            if ui
                .add(egui::TextEdit::singleline(&mut text).desired_width(80.0))
                .changed()
                && let Ok(seed) = u32::from_str_radix(text.trim(), 16)
            {
                self.recipe.seed = seed;
            }
            if ui
                .button("reroll")
                .on_hover_text("a new seed, and the drawing that comes with it")
                .clicked()
            {
                self.recipe.seed = crate::rng::fresh_seed(&mut self.seed_counter);
                self.draw_frame();
            }
        });

        ui.add_space(4.0);
        if ui
            .button("draw this frame")
            .on_hover_text("replace the frame being edited — undoable")
            .clicked()
        {
            self.draw_frame();
        }
        if ui
            .button("draw frame and palette")
            .on_hover_text("also replace `pal` with the ramp the drawing is shaded against")
            .clicked()
        {
            self.draw_frame_and_palette();
        }
    }

    fn draw_frame(&mut self) {
        let frame = gensprite::generate(&self.recipe);
        if let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        {
            art.put_current(frame);
        }
    }

    /// Draw and re-ramp together.
    ///
    /// The two halves have to move as one: [`gensprite::generate`] writes the
    /// indices that [`gensprite::generate_pal`] gives meaning to, so a frame
    /// generated against a four-step ramp and dropped onto a two-entry palette
    /// draws magenta.
    fn draw_frame_and_palette(&mut self) {
        let frame = gensprite::generate(&self.recipe);
        let pal = gensprite::generate_pal(&self.recipe, self.base_hsl());
        if let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        {
            art.put_current(frame);
            art.sprite.pal = pal;
            art.pal_dirty = true;
        }
        self.clamp_ink();
    }

    /// Apply a `Frame -> Frame` operator to the frame being edited.
    ///
    /// One helper rather than a dozen copies of the same `let-else`, and the
    /// only place the operator buttons reach the document — so an operator
    /// cannot be added without its undo.
    fn apply(&mut self, f: impl Fn(&Frame) -> Frame) {
        if let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        {
            let out = f(&art.sprite.seqs[art.seq].frames[art.frame]);
            art.put_current(out);
        }
    }

    /// Replace the whole sequence with frames derived from the one being edited.
    fn apply_seq(&mut self, f: impl Fn(&Frame) -> Vec<Frame>) {
        if let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        {
            let out = f(&art.sprite.seqs[art.seq].frames[art.frame]);
            art.put_frames(out);
        }
    }

    fn operators_section(&mut self, ui: &mut egui::Ui, pal_len: usize) {
        let ink = digit(self.ink.clamp(1, pal_len.saturating_sub(1).max(1)));
        let top = digit(pal_len.saturating_sub(1).max(1));

        ui.horizontal_wrapped(|ui| {
            if ui.button("flip ↔").clicked() {
                self.apply(ops::mirror_x);
            }
            if ui.button("flip ↕").clicked() {
                self.apply(ops::mirror_y);
            }
            if ui
                .button("fold ↔")
                .on_hover_text("copy the left half onto the right")
                .clicked()
            {
                self.apply(ops::fold_x);
            }
            // A rotate needs a square grid, and until the 8x8 migration lands
            // not every record has one. Disabled rather than hidden, with the
            // reason on the hover: "why is this greyed out" has an answer worth
            // reading.
            let square = self.recipe.w == self.recipe.h;
            if ui
                .add_enabled(square, egui::Button::new("rotate ⟳"))
                .on_disabled_hover_text("a quarter turn needs a square grid")
                .clicked()
            {
                // `rotate_cw` already refused anything but a square, so the
                // fallback is unreachable while the button is enabled.
                self.apply(|f| ops::rotate_cw(f).unwrap_or_else(|| f.clone()));
            }
        });

        ui.horizontal_wrapped(|ui| {
            let wrap = self.wrap_shift;
            for (label, dx, dy) in [("←", -1, 0), ("→", 1, 0), ("↑", 0, -1), ("↓", 0, 1)] {
                if ui.button(label).clicked() {
                    self.apply(|f| ops::shift(f, dx, dy, wrap));
                }
            }
            ui.checkbox(&mut self.wrap_shift, "wrap")
                .on_hover_text("bring what leaves one edge back on the other");
        });

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("outline")
                .on_hover_text("draw the selected index around the silhouette")
                .clicked()
            {
                self.apply(|f| ops::outline(f, ink));
            }
            if ui
                .button("shade")
                .on_hover_text("rim-light from the top left, where this game's art is lit from")
                .clicked()
            {
                self.apply(|f| ops::auto_shade(f, top, ink, (-1, -1)));
            }
            if ui
                .button("dither")
                .on_hover_text("checkerboard the selected index with the one below it")
                .clicked()
            {
                let other = digit(self.ink.saturating_sub(1).max(1));
                let phase = self.dither_phase;
                self.apply(|f| ops::dither(f, ink, other, phase));
                // Flip the phase so pressing it twice gives the inverse pattern
                // rather than the same one again.
                self.dither_phase ^= 1;
            }
        });

        ui.horizontal(|ui| {
            ui.label("click:");
            for (t, label, hint) in [
                (
                    Tool::Paint,
                    "paint",
                    "the left button paints, the right erases",
                ),
                (
                    Tool::Flood,
                    "fill",
                    "flood the region under the pointer — the right button fills with transparent",
                ),
            ] {
                if ui
                    .selectable_label(self.tool == t, label)
                    .on_hover_text(hint)
                    .clicked()
                {
                    self.tool = t;
                }
            }
        });
    }

    fn colours_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("base");
            ui.color_edit_button_srgb(&mut self.base_rgb);
            ui.monospace(palette::hex(self.base_rgb));
        });
        ui.add(egui::Slider::new(&mut self.hue_shift, -30.0..=30.0).text("hue/step"))
            .on_hover_text("degrees per step — 0 leaves the ramp a brightness slider");
        if ui
            .button("ramp")
            .on_hover_text("replace the palette with a generated ramp")
            .clicked()
        {
            let pal = palette::ramp(
                self.base_hsl(),
                self.recipe.inks,
                self.hue_shift,
                (0.12, 0.88),
            );
            self.put_palette(pal);
        }

        ui.add_space(6.0);
        if ui
            .button("snap to reference")
            .on_hover_text(
                "move every colour to its nearest of the 64 reference colours — \
                 what makes a new record read as part of the same set",
            )
            .clicked()
        {
            let Some(current) = self.current_palette() else {
                return;
            };
            let pal = palette::snap(&current, &palette::REFERENCE);
            self.put_palette(pal);
        }

        ui.add_space(6.0);
        ui.add(egui::Slider::new(&mut self.harmonize, 0.0..=1.0).text("toward base"))
            .on_hover_text("0 changes nothing at all");
        if ui
            .button("harmonize")
            .on_hover_text("pull every colour toward the base hue, leaving lightness alone")
            .clicked()
        {
            let Some(current) = self.current_palette() else {
                return;
            };
            let pal = palette::harmonize(&current, self.base_hsl(), self.harmonize);
            self.put_palette(pal);
        }
    }

    fn current_palette(&self) -> Option<Vec<String>> {
        match self.open.as_ref() {
            Some(Open {
                doc: Doc::Art(art), ..
            }) => Some(art.sprite.pal.clone()),
            _ => None,
        }
    }

    /// Replace the palette, snapshotting so it can be taken back.
    ///
    /// A palette change is why [`crate::doc::Snapshot`] carries colours: undoing
    /// a ramp has to restore the colours the frames were drawn against, not only
    /// the frames.
    fn put_palette(&mut self, pal: Vec<String>) {
        if let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_mut()
        {
            art.snapshot();
            art.sprite.pal = pal;
            art.pal_dirty = true;
        }
        self.clamp_ink();
    }

    /// Keep the selected index inside the palette it indexes.
    ///
    /// A ramp can be shorter than the palette it replaced, and an `ink` past the
    /// end paints a digit no colour answers to — which the canvas draws magenta
    /// and the raster refuses.
    fn clamp_ink(&mut self) {
        if let Some(Open {
            doc: Doc::Art(art), ..
        }) = self.open.as_ref()
        {
            self.ink = self
                .ink
                .clamp(1, art.sprite.pal.len().saturating_sub(1).max(1));
        }
    }

    fn animate_section(&mut self, ui: &mut egui::Ui, pal_len: usize) {
        ui.add(egui::Slider::new(&mut self.bob_amp, 1..=3).text("bob by"));
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("bob")
                .on_hover_text("two frames: the drawing, and the drawing lifted")
                .clicked()
            {
                let amp = self.bob_amp;
                self.apply_seq(move |f| anim::bob(f, amp));
            }
            if ui
                .button("squash")
                .on_hover_text("two frames, feet planted — the top row is what gives")
                .clicked()
            {
                self.apply_seq(anim::squash);
            }
            if ui
                .button("walk")
                .on_hover_text(
                    "four frames alternating which foot is down — a hop if the \
                     bottom row is one unbroken group",
                )
                .clicked()
            {
                self.apply_seq(anim::walk);
            }
            if ui
                .add_enabled(pal_len > 2, egui::Button::new("blink"))
                .on_disabled_hover_text("needs an index to swap and one to swap it for")
                .clicked()
            {
                let eye = digit(self.ink);
                let lid = digit(self.ink.saturating_sub(1).max(1));
                self.apply(move |f| anim::blink(f, eye, lid));
            }
        });
        ui.label(
            egui::RichText::new("bob and squash replace the sequence; blink replaces the frame")
                .small()
                .weak(),
        );
    }
}

/// A palette index as the character a frame writes.
fn digit(i: usize) -> char {
    char::from_digit(i as u32, 10).unwrap_or(ops::CLEAR)
}
