//! What the player has chosen, and the file it survives in.
//!
//! # Every one of these does something
//!
//! That is the whole rule, and it is worth stating because a settings screen is
//! the easiest place in a game to ship furniture. There is no audio anywhere in
//! this tree, so there are no volume sliders; there is no mod loader, no
//! language table and no chat, so none of those appear either. Each field below
//! names a system that reads it, and `settings::apply` is where they are read.
//!
//! # Why a line file and not TOML
//!
//! `content/` is TOML and `contentc` parses it, but that parser is a build-time
//! tool and `toml` is not a dependency of any crate that ships. Adding one to
//! read nine values would be a strange trade.
//!
//! So this is `key=value`, one per line, which is what Minecraft's `options.txt`
//! has been for fifteen years and for the same reason. It is also the format
//! that degrades best: an unknown key is skipped, a malformed value keeps the
//! default, and a file from a newer build loads in an older one. A settings file
//! that refuses to load is a settings file that has silently reset everything
//! the player chose.
//!
//! # Where it lives
//!
//! Beside the saves directory rather than inside it. Settings are the PLAYER's
//! and worlds are the game's — deleting a world should not reset the controls,
//! and copying a world to another machine should not carry someone else's
//! fullscreen preference with it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::window::{MonitorSelection, PresentMode, WindowMode};

/// The file, inside the directory `saves` lives in.
const FILE: &str = "options.txt";

/// How the control hints behave.
///
/// Three states and not a bool, because the useful answers are not two. A
/// player who has just learned the game wants them gone; a player who comes
/// back after a month wants them back; the default wants neither and fades.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HintMode {
    /// Always on screen.
    Always,
    /// Hold, then fade out. The default, and what the HUD does.
    #[default]
    Fade,
    /// Never drawn. They are still on the pause card.
    Never,
}

impl HintMode {
    /// The whole cycle, in the order the option button steps through it.
    pub const ALL: [HintMode; 3] = [HintMode::Always, HintMode::Fade, HintMode::Never];

    /// What the option button says.
    pub fn label(self) -> &'static str {
        match self {
            HintMode::Always => "Always",
            HintMode::Fade => "Fade out",
            HintMode::Never => "Never",
        }
    }

    fn key(self) -> &'static str {
        match self {
            HintMode::Always => "always",
            HintMode::Fade => "fade",
            HintMode::Never => "never",
        }
    }

    fn parse(s: &str) -> Option<HintMode> {
        HintMode::ALL.into_iter().find(|m| m.key() == s)
    }
}

/// How big a buffer pixel is on screen.
///
/// The equivalent of Minecraft's GUI Scale, and it does the same job: the whole
/// game is drawn into a low-resolution buffer and then upscaled by a whole
/// number, so this IS the size of everything. `Auto` is what the game has always
/// done — fit the buffer to the window and take whatever zoom that implies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderScale {
    /// Fit the window, as [`yugen_core::config::View::for_screen`] does.
    #[default]
    Auto,
    /// A fixed whole-number zoom.
    Fixed(u32),
}

impl RenderScale {
    /// The cycle, in the order the option button steps through it.
    pub const ALL: [RenderScale; 5] = [
        RenderScale::Auto,
        RenderScale::Fixed(2),
        RenderScale::Fixed(3),
        RenderScale::Fixed(4),
        RenderScale::Fixed(5),
    ];

    /// What the option button says.
    pub fn label(self) -> String {
        match self {
            RenderScale::Auto => "Auto".into(),
            RenderScale::Fixed(n) => format!("{n}x"),
        }
    }

    /// The zoom floor to hand `View::for_screen_at`, or `None` for automatic.
    pub fn zoom(self) -> Option<f32> {
        match self {
            RenderScale::Auto => None,
            RenderScale::Fixed(n) => Some(n as f32),
        }
    }
}

/// Everything the player has chosen.
///
/// Plain data, `Copy`, no Bevy inside the values — so the options screen can be
/// laid out and driven in a test with no app, on the same terms as every other
/// pure layout function in `ui`.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    // --- Video ---
    /// Borderless fullscreen.
    pub fullscreen: bool,
    /// Wait for the display's refresh. Off trades tearing for latency.
    pub vsync: bool,
    /// Buffer pixel size. See [`RenderScale`].
    pub scale: RenderScale,
    /// Fraction of the particle budget that may be alive at once, `0.0..=1.0`.
    ///
    /// A budget and not a switch: particles are most of what makes digging feel
    /// like digging, and the honest low setting is fewer of them rather than
    /// none.
    pub particles: f32,
    /// Screenshake magnitude, `0.0..=1.0`. Zero is off.
    ///
    /// Present because camera shake is the single most common accessibility
    /// complaint about games that have it, and this one shakes on every hit.
    pub shake: f32,

    // --- Interface ---
    /// What the control hints do. See [`HintMode`].
    pub hints: HintMode,
    /// Whether the F3 panel starts up.
    pub debug_overlay: bool,

    // --- World ---
    /// Real minutes in one in-game day.
    pub day_minutes: f32,
    /// Seconds between autosaves.
    pub autosave_s: f32,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            fullscreen: false,
            vsync: true,
            scale: RenderScale::Auto,
            particles: 1.0,
            shake: 1.0,
            hints: HintMode::Fade,
            debug_overlay: false,
            // The values the game already had as constants, so a fresh install
            // and a build without this module play identically. Asserted by
            // `the_defaults_are_the_constants_the_game_already_shipped` — the
            // first version of this had an invented 8 minutes against a
            // `DAY_LENGTH_S` of 300 seconds, which would have silently changed
            // the day length for every existing player.
            day_minutes: crate::daynight::DAY_LENGTH_S / 60.0,
            autosave_s: crate::world::AUTOSAVE_EVERY_S,
        }
    }
}

/// The bounds each numeric setting is clamped to on load and on edit.
///
/// Stated once, here, rather than at the slider and again at the parser: a file
/// hand-edited to `shake=900` must land somewhere sane, and it must land in the
/// same place the slider would have put it.
impl Settings {
    /// Real minutes in a day, at the ends of the slider.
    pub const DAY_MINUTES: (f32, f32) = (2.0, 60.0);
    /// Seconds between autosaves, at the ends of the slider.
    pub const AUTOSAVE_S: (f32, f32) = (10.0, 300.0);

    /// Pull every value back inside its range.
    pub fn clamp(&mut self) {
        self.particles = self.particles.clamp(0.0, 1.0);
        self.shake = self.shake.clamp(0.0, 1.0);
        self.day_minutes = self
            .day_minutes
            .clamp(Settings::DAY_MINUTES.0, Settings::DAY_MINUTES.1);
        self.autosave_s = self
            .autosave_s
            .clamp(Settings::AUTOSAVE_S.0, Settings::AUTOSAVE_S.1);
    }

    /// The file, as text.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Yugen settings. Delete a line to restore its default.");
        let _ = writeln!(s, "fullscreen={}", self.fullscreen);
        let _ = writeln!(s, "vsync={}", self.vsync);
        let _ = writeln!(
            s,
            "scale={}",
            match self.scale {
                RenderScale::Auto => "auto".to_string(),
                RenderScale::Fixed(n) => n.to_string(),
            }
        );
        let _ = writeln!(s, "particles={:.2}", self.particles);
        let _ = writeln!(s, "shake={:.2}", self.shake);
        let _ = writeln!(s, "hints={}", self.hints.key());
        let _ = writeln!(s, "debug_overlay={}", self.debug_overlay);
        let _ = writeln!(s, "day_minutes={:.1}", self.day_minutes);
        let _ = writeln!(s, "autosave_s={:.0}", self.autosave_s);
        s
    }

    /// Parse `text`, keeping the default for anything missing or malformed.
    ///
    /// Never fails. Every branch that could reject something keeps the default
    /// instead, because the alternative — refusing the file — silently resets
    /// every choice the player made, which is worse than ignoring one bad line.
    pub fn from_text(text: &str) -> Settings {
        let mut out = Settings::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "fullscreen" => out.fullscreen = value == "true",
                "vsync" => out.vsync = value == "true",
                "debug_overlay" => out.debug_overlay = value == "true",
                "scale" => {
                    out.scale = if value == "auto" {
                        RenderScale::Auto
                    } else {
                        match value.parse::<u32>() {
                            Ok(n) if (1..=8).contains(&n) => RenderScale::Fixed(n),
                            _ => RenderScale::Auto,
                        }
                    }
                }
                "hints" => {
                    if let Some(m) = HintMode::parse(value) {
                        out.hints = m;
                    }
                }
                "particles" => {
                    if let Ok(v) = value.parse() {
                        out.particles = v;
                    }
                }
                "shake" => {
                    if let Ok(v) = value.parse() {
                        out.shake = v;
                    }
                }
                "day_minutes" => {
                    if let Ok(v) = value.parse() {
                        out.day_minutes = v;
                    }
                }
                "autosave_s" => {
                    if let Ok(v) = value.parse() {
                        out.autosave_s = v;
                    }
                }
                _ => {}
            }
        }
        out.clamp();
        out
    }
}

/// Where the settings file lives, given where saves do.
pub fn path_beside(saves: &Path) -> PathBuf {
    saves.parent().unwrap_or(saves).join(FILE)
}

/// Where this app reads and writes its settings. Empty until the host sets it.
///
/// A resource rather than a constant because the tests and `--saves` both point
/// the game at a different directory, and a settings file that ignored that
/// would have every test in the suite writing over the developer's own.
#[derive(Resource, Clone, Debug, Default)]
pub struct SettingsFile(pub Option<PathBuf>);

/// Read the file into [`Settings`], if there is one to read.
fn load(mut settings: ResMut<Settings>, file: Res<SettingsFile>) {
    let Some(path) = file.0.as_deref() else {
        return;
    };
    match std::fs::read_to_string(path) {
        Ok(text) => {
            *settings = Settings::from_text(&text);
            info!("settings: loaded {}", path.display());
        }
        // Not an error: the first run has no file, and that is the common case.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!("settings: cannot read {}: {e}", path.display()),
    }
}

/// Write the file whenever [`Settings`] changes.
///
/// On change and not on quit. A crash after an hour of play should not lose the
/// fact that the player turned screenshake off in the first minute, and the
/// file is under a kilobyte.
fn store(settings: Res<Settings>, file: Res<SettingsFile>) {
    if !settings.is_changed() || settings.is_added() {
        return;
    }
    let Some(path) = file.0.as_deref() else {
        return;
    };
    if let Some(dir) = path.parent()
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        warn!("settings: cannot create {}: {e}", dir.display());
        return;
    }
    if let Err(e) = std::fs::write(path, settings.to_text()) {
        warn!("settings: cannot write {}: {e}", path.display());
    }
}

/// Push every setting to the module that owns the behaviour it names.
///
/// # Why one system and not eight
///
/// Because the alternative is eight modules that each read `Settings`, and the
/// whole point of the resources below is that `lowres`, `particles`, `effects`
/// and `daynight` keep knowing nothing about menus. This is the only place in
/// the tree that reads a preference and writes a behaviour, so "what does this
/// setting actually do" has one answer and it is a line of this function.
///
/// Runs only when [`Settings`] changes. Every write below would otherwise mark
/// its target changed every frame, which for `ZoomFloor` means `fit_canvas`
/// rebuilding the buffer forever.
fn apply(settings: Res<Settings>, mut into: Applied) {
    let Applied {
        windows,
        floor,
        autosave,
        shake,
        particles,
        clock,
        overlay,
    } = &mut into;
    for mut window in &mut *windows {
        let want = if settings.fullscreen {
            // Borderless on the current monitor, which is the fullscreen mode
            // that does not change the display's resolution out from under the
            // compositor — the game already scales to any window size.
            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
        } else {
            WindowMode::Windowed
        };
        if window.mode != want {
            window.mode = want;
        }
        let present = if settings.vsync {
            PresentMode::AutoVsync
        } else {
            PresentMode::AutoNoVsync
        };
        if window.present_mode != present {
            window.present_mode = present;
        }
    }

    // `set_if_neq` by hand for the same reason `glue::follow_scene` does it:
    // `ZoomFloor` changing is what makes `fit_canvas` rebuild the buffer, so it
    // must change only when the value does.
    let want = settings.scale.zoom().unwrap_or(AUTO_ZOOM);
    if floor.0 != want {
        floor.0 = want;
    }
    if autosave.0 != settings.autosave_s {
        autosave.0 = settings.autosave_s;
    }
    if let Some(shake) = shake.as_mut() {
        shake.scale = settings.shake;
    }
    if let Some(particles) = particles.as_mut() {
        particles.budget = settings.particles;
    }
    if let Some(clock) = clock.as_mut() {
        clock.0.length_s = settings.day_minutes * 60.0;
    }
    if overlay.0 != settings.debug_overlay {
        overlay.0 = settings.debug_overlay;
    }
}

/// Everything [`apply`] writes into. Bundled so the system stays inside
/// clippy's argument budget, the same way `debug::Sources` and `ui::HudSources`
/// do it.
///
/// The four `Option`s are for a host that runs some of the game and not all of
/// it — the capture tests and the examples both do — and a missing one means
/// that system is not in the app rather than that something failed.
#[derive(bevy::ecs::system::SystemParam)]
struct Applied<'w, 's> {
    windows: Query<'w, 's, &'static mut Window>,
    floor: ResMut<'w, crate::lowres::ZoomFloor>,
    autosave: ResMut<'w, crate::world::AutosaveEvery>,
    overlay: ResMut<'w, crate::debug::DebugOverlay>,
    shake: Option<ResMut<'w, crate::effects::Screenshake>>,
    particles: Option<ResMut<'w, crate::particles::ParticleSystem>>,
    clock: Option<ResMut<'w, crate::daynight::WorldClock>>,
}

/// The zoom floor `RenderScale::Auto` means.
///
/// The value `View::for_screen` has always used, so Auto is exactly the
/// behaviour every build before this one had.
const AUTO_ZOOM: f32 = 2.0;

/// Installs [`Settings`] and keeps the file in step with it.
pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Settings>()
            .init_resource::<SettingsFile>()
            // `PreStartup`, so every `Startup` system already sees the loaded
            // values rather than the defaults — `lowres` builds its target in
            // `Startup` and has to be told the render scale before it does.
            .add_systems(PreStartup, load)
            // `PostStartup`, so the resources it writes into exist, and then on
            // every change. `resource_changed` covers the `PostStartup` run
            // too, because a freshly inserted resource counts as changed.
            .add_systems(PostStartup, apply)
            .add_systems(Update, apply.run_if(resource_changed::<Settings>))
            .add_systems(Last, store);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_trip_through_the_file_changes_nothing() {
        let mut want = Settings {
            fullscreen: true,
            vsync: false,
            scale: RenderScale::Fixed(3),
            particles: 0.25,
            shake: 0.0,
            hints: HintMode::Never,
            debug_overlay: true,
            day_minutes: 12.0,
            autosave_s: 60.0,
        };
        want.clamp();
        assert_eq!(Settings::from_text(&want.to_text()), want);
    }

    #[test]
    fn the_defaults_are_the_constants_the_game_already_shipped() {
        // A default that differs from the constant it replaces is a silent
        // change to how the game plays for everyone who never opens Options.
        let d = Settings::default();
        assert_eq!(d.day_minutes * 60.0, crate::daynight::DAY_LENGTH_S);
        assert_eq!(d.autosave_s, crate::world::AUTOSAVE_EVERY_S);
        assert_eq!(d.scale.zoom(), None, "Auto must not pin a zoom");
    }

    #[test]
    fn an_empty_file_is_the_defaults() {
        assert_eq!(Settings::from_text(""), Settings::default());
    }

    #[test]
    fn a_bad_line_costs_that_line_and_nothing_else() {
        // The whole reason this parser cannot fail: a file with one unreadable
        // value must not reset the other eight.
        let text = "shake=banana\nshake_scale\n=\nhints=never\nnonsense=1\n";
        let got = Settings::from_text(text);
        assert_eq!(got.hints, HintMode::Never, "a later key was lost");
        assert_eq!(got.shake, Settings::default().shake);
    }

    #[test]
    fn a_key_from_a_newer_build_is_skipped_rather_than_fatal() {
        let text = format!("{}\nray_tracing=true\n", Settings::default().to_text());
        assert_eq!(Settings::from_text(&text), Settings::default());
    }

    #[test]
    fn a_hand_edited_value_out_of_range_lands_where_the_slider_would_put_it() {
        let got = Settings::from_text("shake=900\nparticles=-4\nday_minutes=100000\n");
        assert_eq!(got.shake, 1.0);
        assert_eq!(got.particles, 0.0);
        assert_eq!(got.day_minutes, Settings::DAY_MINUTES.1);
    }

    #[test]
    fn an_unknown_scale_falls_back_to_auto_rather_than_to_nothing() {
        assert_eq!(Settings::from_text("scale=99").scale, RenderScale::Auto);
        assert_eq!(Settings::from_text("scale=x").scale, RenderScale::Auto);
        assert_eq!(Settings::from_text("scale=4").scale, RenderScale::Fixed(4));
    }

    #[test]
    fn the_file_sits_beside_the_saves_directory_and_not_inside_it() {
        // Deleting a world must not reset the controls.
        let p = path_beside(Path::new("/home/someone/.local/share/yugen/saves"));
        assert_eq!(p, Path::new("/home/someone/.local/share/yugen").join(FILE));
    }
}
