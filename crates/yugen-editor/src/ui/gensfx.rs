//! The sound generate panel: categories, a mutation slider, and what you rolled.
//!
//! # History exists because rolling is lossy
//!
//! The roll button's failure mode is not producing a bad sound, it is producing
//! a good one and then producing another. Without somewhere for the previous
//! thirty to go, "that one was nearly right" is a sound that no longer exists —
//! and the seed is only a recovery route if you noticed in time to write it down.
//! So every roll is kept, and clicking one auditions it without committing it.
//!
//! # Nothing here is persisted
//!
//! This editor writes content files and nothing else. A favourites file would be
//! the first exception, and it does not earn one: a favourite worth keeping
//! becomes a record, which is what the New button is for and where it belongs.
//! Closing the window loses the list, and that is the correct amount of
//! ceremony for something you have not decided to keep.
//!
//! # Auditioning must not clobber the scope
//!
//! `Sfx::refresh` re-renders when `params != rendered`, so writing an audition's
//! samples into `sfx.pcm` would leave the scope drawing a sound the knobs do not
//! describe. Auditions render into a scratch buffer and go straight to the
//! speaker.

use eframe::egui;

use crate::app::Editor;
use crate::doc::{Doc, Open};
use crate::procgen::sfx::{self, Category};
use crate::synth::{self, Params};

/// One thing that was rolled, and what produced it.
#[derive(Clone)]
pub(crate) struct Roll {
    pub(crate) params: Params,
    pub(crate) seed: u32,
    /// How it was made — "pickup", "random", "mutate 12%". Shown rather than
    /// stored as an enum because it is a label, and the mutation carries a
    /// number that only matters as text.
    pub(crate) what: String,
    pub(crate) favourite: bool,
}

/// The roll list, newest first.
#[derive(Default)]
pub(crate) struct Rolls {
    pub(crate) history: Vec<Roll>,
    /// How hard the mutate button pushes.
    pub(crate) amount: f32,
}

/// How many rolls are kept.
///
/// Deep enough that a session of pressing a category button does not lose the
/// good one; shallow enough that the list stays scannable. Favourites are
/// exempt — they are the ones somebody said they wanted.
const HISTORY: usize = 32;

impl Rolls {
    fn push(&mut self, params: Params, seed: u32, what: String) {
        self.history.insert(
            0,
            Roll {
                params,
                seed,
                what,
                favourite: false,
            },
        );
        if self.history.len() > HISTORY {
            // Drop the oldest non-favourite rather than the oldest, so starring
            // something makes it survive the scroll.
            if let Some(i) = self.history.iter().rposition(|r| !r.favourite) {
                self.history.remove(i);
            } else {
                self.history.pop();
            }
        }
    }
}

impl Editor {
    pub(super) fn gensfx(&mut self, ui: &mut egui::Ui) {
        if !matches!(self.open.as_ref().map(|o| &o.doc), Some(Doc::Sfx(_))) {
            return;
        }

        egui::CollapsingHeader::new("Roll a sound")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for cat in Category::ALL {
                        if ui.button(cat.name()).clicked() {
                            let seed = crate::rng::fresh_seed(&mut self.seed_counter);
                            let p = sfx::roll(cat, seed);
                            self.rolls.push(p, seed, cat.name().to_string());
                            self.take(p);
                        }
                    }
                });
                ui.add_space(4.0);
                if ui
                    .button("surprise me")
                    .on_hover_text("anywhere inside the schema's ranges")
                    .clicked()
                {
                    let seed = crate::rng::fresh_seed(&mut self.seed_counter);
                    let p = sfx::randomize(seed);
                    self.rolls.push(p, seed, "random".into());
                    self.take(p);
                }

                ui.add_space(8.0);
                ui.add(
                    egui::Slider::new(&mut self.rolls.amount, 0.0..=0.5)
                        .text("mutate by")
                        .fixed_decimals(2),
                )
                .on_hover_text("a fraction of each field's own range — small is the useful end");
                if ui
                    .button("mutate")
                    .on_hover_text("jitter what is loaded — how a nearly-right sound becomes right")
                    .clicked()
                {
                    let Some(current) = self.current_params() else {
                        return;
                    };
                    let seed = crate::rng::fresh_seed(&mut self.seed_counter);
                    let amount = self.rolls.amount;
                    let p = sfx::mutate(&current, amount, seed);
                    self.rolls
                        .push(p, seed, format!("mutate {:.0}%", amount * 100.0));
                    self.take(p);
                }
            });

        egui::CollapsingHeader::new(format!("Rolls ({})", self.rolls.history.len()))
            .default_open(true)
            .show(ui, |ui| self.roll_list(ui));
    }

    fn current_params(&self) -> Option<Params> {
        match self.open.as_ref() {
            Some(Open {
                doc: Doc::Sfx(sfx), ..
            }) => Some(sfx.sound.params),
            _ => None,
        }
    }

    /// Load a roll into the open record and play it.
    fn take(&mut self, p: Params) {
        if let Some(Open {
            doc: Doc::Sfx(sfx), ..
        }) = self.open.as_mut()
        {
            sfx.sound.params = p;
        }
        self.play_sound();
    }

    /// Play a roll without loading it.
    ///
    /// Straight to the speaker from a scratch render — see the module header on
    /// why this must not go through `sfx.pcm`.
    fn audition(&mut self, p: &Params) {
        let pcm = synth::render(p);
        self.emit(pcm);
    }

    fn roll_list(&mut self, ui: &mut egui::Ui) {
        if self.rolls.history.is_empty() {
            ui.label(egui::RichText::new("nothing rolled yet").small().weak());
            return;
        }

        let mut audition: Option<Params> = None;
        let mut load: Option<Params> = None;
        let mut star: Option<usize> = None;

        egui::ScrollArea::vertical()
            .max_height(220.0)
            .show(ui, |ui| {
                for (i, r) in self.rolls.history.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(r.favourite, if r.favourite { "★" } else { "☆" })
                            .on_hover_text("keep this one when the list scrolls")
                            .clicked()
                        {
                            star = Some(i);
                        }
                        if ui
                            .button("▶")
                            .on_hover_text("hear it without loading it")
                            .clicked()
                        {
                            audition = Some(r.params);
                        }
                        if ui
                            .button(format!("{} · {:08x}", r.what, r.seed))
                            .on_hover_text("load it into the knobs")
                            .clicked()
                        {
                            load = Some(r.params);
                        }
                    });
                }
            });

        if let Some(i) = star {
            self.rolls.history[i].favourite = !self.rolls.history[i].favourite;
        }
        if let Some(p) = audition {
            self.audition(&p);
        }
        if let Some(p) = load {
            self.take(p);
        }
    }
}
