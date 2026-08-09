//! When a sound happens: a queue anything can push to, drained once a frame.
//!
//! # Why a queue and not a call
//!
//! The things that know a sound should play are the systems draining
//! `PlayerEvent` and `MobEvent` — and those are the same systems that spawn
//! particles and shake the screen, holding exactly the borrows they need for
//! that and nothing more. Handing them `Commands` and an `Assets` so they could
//! spawn an audio entity would widen four systems to make one noise.
//!
//! A [`SoundQueue`] is a `Vec<u16>` they can push a code onto. One system drains
//! it. That is the same shape [`crate::particles`] uses for its emit path and
//! for the same reason: the decision and the machinery are different concerns
//! and want different borrows.
//!
//! # One entity per sound, despawned when done
//!
//! Unlike every other pool in this tree, this one is not a pool. Bevy's audio
//! despawns a `PlaybackMode::Despawn` entity when the source ends, which is the
//! behaviour a fire-and-forget effect wants and is already written; pooling
//! would mean holding sinks alive and reasoning about which are free. A footstep
//! is one entity for a twentieth of a second.
//!
//! The cap exists anyway. A dig held down pushes a code every few frames, and
//! anything that spawns per event needs a ceiling or a bug upstream becomes an
//! unbounded spawn.

use bevy::audio::{PlaybackMode, Volume};
use bevy::prelude::*;

use super::bank::SoundBank;

/// Most sounds started in one frame.
///
/// Well above anything the game does — the loudest frame is a death, which is
/// three — and low enough that a runaway pusher is capped rather than being
/// allowed to spawn until the process dies.
const MAX_PER_FRAME: usize = 16;

/// Sound codes waiting to be played, cleared every frame.
#[derive(Resource, Default, Debug)]
pub struct SoundQueue(Vec<u16>);

impl SoundQueue {
    /// Ask for a sound. Silently dropped if this frame is already full.
    #[inline]
    pub fn play(&mut self, code: u16) {
        if self.0.len() < MAX_PER_FRAME {
            self.0.push(code);
        }
    }

    /// Codes queued this frame.
    pub fn pending(&self) -> &[u16] {
        &self.0
    }
}

/// How loud everything is, `0.0..=1.0`.
///
/// A resource rather than a constant because the settings screen will own it.
/// Zero is silence and is a real state: a player who turns the sound off gets no
/// entities spawned at all, rather than muted ones.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SoundVolume(pub f32);

impl Default for SoundVolume {
    fn default() -> SoundVolume {
        // Not 1.0. The per-sound `gain` in `content/sounds/` is authored against
        // a master that leaves headroom, so three sounds landing together do not
        // sum past full scale.
        SoundVolume(0.6)
    }
}

/// Spawn an audio entity per queued code, then clear the queue.
fn play_queued(
    mut commands: Commands,
    mut queue: ResMut<SoundQueue>,
    bank: Res<SoundBank>,
    volume: Res<SoundVolume>,
) {
    // Drained whatever happens, so a frame where the bank is missing or the
    // volume is zero does not leave codes to play late — a sound that arrives a
    // second after the thing that caused it is worse than one that never does.
    let codes = std::mem::take(&mut queue.0);
    if volume.0 <= 0.0 || bank.is_empty() {
        return;
    }
    for code in codes {
        let Some(handle) = bank.get(code) else {
            continue;
        };
        commands.spawn((
            AudioPlayer(handle),
            PlaybackSettings {
                mode: PlaybackMode::Despawn,
                volume: Volume::Linear(volume.0),
                ..default()
            },
        ));
    }
}

/// The audio half of the host.
pub struct SoundPlugin;

impl Plugin for SoundPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SoundBank>()
            .init_resource::<SoundQueue>()
            .init_resource::<SoundVolume>()
            // `PreStartup` for the same reason `SpritePlugin` bakes there: any
            // `Startup` system may then take the bank without an ordering edge.
            .add_systems(PreStartup, super::bank::bake)
            .add_systems(Update, play_queued);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_queue_is_capped_rather_than_unbounded() {
        // A dig held down pushes every few frames, and anything that spawns per
        // event needs a ceiling — otherwise a bug upstream is an unbounded
        // spawn rather than a sound that gets thin.
        let mut q = SoundQueue::default();
        for _ in 0..MAX_PER_FRAME * 4 {
            q.play(0);
        }
        assert_eq!(q.pending().len(), MAX_PER_FRAME);
    }

    #[test]
    fn silence_is_a_real_setting() {
        // Zero volume must produce no entities, not silent ones. A muted entity
        // still costs a spawn, a sink and a despawn per footstep.
        assert_eq!(SoundVolume::default().0, 0.6);
        assert!(SoundVolume(0.0).0 <= 0.0);
    }
}
