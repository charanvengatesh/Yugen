//! A setting chosen in one run is the setting the next run starts with.
//!
//! The unit tests in `settings` prove the parser and the writer agree with each
//! other. That is not the same as proving the FILE is read at startup and
//! written on change, which is what a player actually depends on, and which no
//! amount of pure testing can reach.

mod common;

use bevy::prelude::*;
use yugen_render::settings::{HintMode, RenderScale, Settings, SettingsFile};

#[test]
fn a_setting_survives_a_restart() {
    let dir = std::env::temp_dir().join(format!("yugen-settings-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let file = dir.join("options.txt");

    // Run one: change three settings and let `store` write them.
    {
        let mut app = common::headless_game("settings-write");
        app.insert_resource(SettingsFile(Some(file.clone())));
        app.finish();
        app.cleanup();
        app.update();
        {
            let mut s = app.world_mut().resource_mut::<Settings>();
            s.shake = 0.0;
            s.hints = HintMode::Never;
            s.scale = RenderScale::Fixed(3);
        }
        // One update to carry the change into `Last`, where `store` runs.
        app.update();
    }
    assert!(file.exists(), "no settings file was written to {file:?}");

    // Run two: a fresh app pointed at the same file starts where run one left
    // off, and the values reached the systems that own the behaviour.
    {
        let mut app = common::headless_game("settings-read");
        app.insert_resource(SettingsFile(Some(file.clone())));
        app.finish();
        app.cleanup();
        app.update();

        let s = *app.world().resource::<Settings>();
        assert_eq!(s.shake, 0.0, "screenshake did not survive");
        assert_eq!(s.hints, HintMode::Never, "the hint mode did not survive");
        assert_eq!(
            s.scale,
            RenderScale::Fixed(3),
            "render scale did not survive"
        );

        // And the preference actually reached the module that owns it — the
        // half of this that `settings`' own unit tests cannot see.
        let shake = app.world().resource::<yugen_render::effects::Screenshake>();
        assert_eq!(shake.scale, 0.0, "the shake setting never reached effects");
        let floor = app.world().resource::<yugen_render::lowres::ZoomFloor>();
        assert_eq!(floor.0, 3.0, "the render scale never reached lowres");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
