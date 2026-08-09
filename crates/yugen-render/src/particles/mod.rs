//! The pooled particle system: impact debris, dust, splashes, embers.
//!
//! Ported from `src/render/Particles.ts`.
//!
//! # The pool is the design
//!
//! Particles live in a fixed-length struct-of-arrays store allocated once.
//! Emitting reuses a dead slot instead of growing; the step and the draw touch
//! only live slots. Nothing on the per-frame path allocates.
//!
//! That property was load-bearing in the TypeScript for a reason that does not
//! survive the port — there is no garbage collector here to wake up mid-game —
//! and it is kept anyway, for the reason that does: the cap is STRUCTURAL. A
//! screen full of explosions cannot make this module cost more than a screen
//! with one spark on it, because there is no code path that would let it. This
//! is the same trade [`yugen_core::entities::projectiles`] documents at
//! length, at 85x the slot count.
//!
//! # Full pool: dropped, not recycled
//!
//! The player's arrow pool recycles its OLDEST slot when it saturates, because
//! the player has spent countable ammo and an arrow that silently vanished is an
//! input that was eaten. This pool does the opposite and DROPS the overflow,
//! because juice is cosmetic: under load the game stops adding new sparks, which
//! nobody can see, rather than deleting the oldest half of a burst that is
//! currently on screen, which everybody can. The two pools differ deliberately,
//! and the difference is a statement about what the contents are worth.
//!
//! # The sprite pool mirrors the particle pool
//!
//! [`ParticlesPlugin`] spawns [`MAX_PARTICLES`] hidden sprite entities at
//! startup and never spawns or despawns another. A particle dying toggles a
//! [`Visibility`], not an ECS archetype — the same pattern, and the same
//! reasoning, as [`crate::mobs`]'s creature placeholders. It matters more here:
//! a burst is fourteen particles arriving and leaving inside half a second, and
//! at that rate spawn/despawn would be a steady archetype churn for something
//! the player experiences as a puff of dust.
//!
//! # What the port changed
//!
//! - **The clock.** The TypeScript stepped particles once per rendered FRAME
//!   with the frame `dt`. Here they step in `FixedUpdate` at
//!   [`STEP_DT`]. That is the clock the things that emit them run on — a
//!   creature's death, a landing, an arrow's impact all arrive from the fixed
//!   step — so a particle now ages on the same clock as the event that made it,
//!   and its arc no longer changes shape with the frame rate.
//! - **Collision. This is a deliberate divergence from the original, not an
//!   oversight — it was raised and kept on purpose.** The TypeScript's particles
//!   were pure ballistics and fell through the floor; nothing in a Canvas2D
//!   `fillRect` cared. Here debris opts into the cell grid through
//!   [`EmitOpts::collide`], and the presets that represent MATTER —
//!   [`ParticleSystem::burst`], `dust`, `splash`, `puff`, `scrape` — have it ON.
//!   The ones that represent air or light (`trail`, `smear`, `ember`) stay
//!   ballistic, because an ember that stops at a ceiling reads as a bug and an
//!   ember that drifts through one reads as heat. See
//!   [`ParticleSystem::update`] for how the test is kept to one branch.
//!
//!   Every other deviation in this file exists because Canvas2D and wgpu are
//!   different machines. This one does not: the original could have done it and
//!   chose not to. It is the one place where this port knowingly plays better
//!   than the thing it is porting, and no parity suite covers particles, so
//!   nothing but this paragraph would tell you. Flipping `collide` to `false` on
//!   the five matter presets restores the original behaviour exactly and changes
//!   nothing else.
//! - **Randomness is a value.** `Math.random()` became [`JuiceRng`], threaded
//!   through the system it belongs to. Cosmetic randomness has no reason to be
//!   entropy-seeded, and making it a value is what lets every test below assert
//!   on a real burst instead of on a mock.
//! - **Two draw passes became two z levels.** `draw` and `drawGlow` were Canvas2D
//!   and did not come across. What separated them was the compositing mode, not
//!   the geometry; see below for what is left of that.
//!
//! # What the port dropped on the way in
//!
//! - ~~**Additive compositing.**~~ Recovered. `drawGlow` set
//!   `globalCompositeOperation = "lighter"` so that self-luminous motes survived
//!   the lighting multiply, and for several milestones this port drew them as
//!   ordinary sprites at a higher z instead, because a Bevy [`Sprite`]
//!   alpha-blends and has no per-sprite blend state. The flag and the pass
//!   separation were kept intact against the day a material could be swapped in.
//!   That day came: [`crate::sky::AdditiveMaterial`] now has four consumers, and
//!   [`place_particle_glow`] is the fourth. Luminous particles leave the sprite
//!   pool entirely and draw as one vertex-coloured mesh above the composite.
//! - **`COLOR_CACHE`.** A `Map` of packed rgb to `rgb(r,g,b)` strings, so the
//!   draw loop did not allocate 2048 strings a frame for a palette of about six
//!   colours. There is no string here — a colour is three bytes — so the cache
//!   has nothing to cache. The "skip the assignment when the colour has not
//!   changed" trick went with it: a `Sprite`'s colour is a field, not a state
//!   machine.
//!
//! # Wiring
//!
//! Nothing calls [`ParticleSystem::emit`] yet. The two seams are already
//! draining the events that should feed it, and both say so in their own
//! comments:
//!
//! | Event source | Where it is drained | The call to add |
//! |---|---|---|
//! | [`MobEvent`] | `crate::mobs::step_creatures` | `particles.mob_event(e)` |
//! | [`PlayerEvent`](yugen_core::entities::player::PlayerEvent) | `crate::player::step_player` | one preset per arm — see [`ParticleSystem::puff`] |
//!
//! [`ParticleSystem::mob_event`] is one line because a [`MobEvent`] carries
//! everything a burst needs: where, what colour, how hard. A `PlayerEvent` is a
//! bare enum and carries none of that, so the caller supplies the feet position
//! and the tint of whatever is underfoot — which is exactly what `Game.ts` did,
//! and why the presets are public.

mod presets;
mod rng;
mod system;

pub use presets::{EmitOpts, Preset};
pub use rng::JuiceRng;
pub use system::{MAX_PARTICLES, Particle, ParticleSystem};

use bevy::camera::visibility::NoFrustumCulling;
use bevy::prelude::*;
use bevy::sprite_render::Material2dPlugin;

use yugen_core::config::STEP_DT;

use crate::lowres::WORLD_LAYERS;
use crate::player::PlayerSet;
use crate::sky::{AdditiveMaterial, VertexBuf, dynamic_mesh, linear};
use crate::world::{SimSet, SimWorld};

/// Where ordinary particles sit in z: over the shots (0.55), under the brush
/// preview (1.0).
///
/// Over the shots deliberately. Debris is what tells the player that the shot
/// connected, and a spark behind the arrow that made it is a frame of feedback
/// spent on nothing.
const PARTICLE_Z: f32 = 0.6;

/// Where the self-luminous pass sits: ABOVE the whole light composite.
///
/// This is `drawGlow`, and it is now the real thing rather than the geometric
/// half of it — see the header.
///
/// The height is the original's overlay order. `Game.ts` ran its overlay callback
/// after `light.render`, particles first and then creatures, so this sits under
/// [`crate::mobs`]'s creature glow at 0.80 and its shot glow at 0.81, over
/// `crate::light`'s composite (which ends at 0.76 with the flat washes), and
/// below the HUD at 1.0 — a HUD dimmed by the cave behind it is the bug the
/// overlay layer exists to avoid.
///
/// # The number this replaces was a live z-fight
///
/// Glowing particles used to draw as ordinary sprites at `GLOW_Z = 0.7`, which is
/// EXACTLY `crate::light`'s `SHADOW_Z` of 0.70. Two quads at one depth sort
/// arbitrarily, so whether a spark landed in front of the darkness multiply or
/// behind it was undefined — and behind it is invisible, which is the precise
/// failure the glow pass exists to prevent. Nobody had named it.
const PARTICLE_GLOW_Z: f32 = 0.79;

/// The two orderings above that are pure constants, checked when the crate is
/// compiled rather than when its tests run.
///
/// `crate::light`'s composite ends at 0.76 with the flat washes and
/// `crate::mobs`'s creature glow begins at 0.80; the luminous particle pass has
/// to land between them, and the ordinary particles have to stay below the
/// composite where they are meant to be LIT rather than added.
const _: () = assert!(
    PARTICLE_GLOW_Z > 0.76,
    "the glow pass sank into the light composite, which would multiply it away"
);
const _: () = assert!(
    PARTICLE_GLOW_Z < 0.80,
    "the glow pass rose above the creatures', reversing the original's overlay order"
);
const _: () = assert!(
    PARTICLE_Z < 0.70,
    "ordinary particles rose above the darkness multiply and stopped being lit"
);

// ---------------------------------------------------------------------------
// The Bevy half
// ---------------------------------------------------------------------------

/// One pooled particle sprite. `slot` indexes the pool it mirrors.
#[derive(Component, Clone, Copy)]
pub struct ParticleSprite {
    /// Index into the [`ParticleSystem`] arrays.
    pub slot: usize,
}

/// The pool, its fixed step, and [`MAX_PARTICLES`] sprites to draw it with.
///
/// Deliberately not in [`YugenRenderPlugin`](crate::YugenRenderPlugin) yet:
/// until the two seams in the header are wired, adding it would spawn 2048
/// entities to draw nothing.
pub struct ParticlesPlugin;

impl Plugin for ParticlesPlugin {
    fn build(&self, app: &mut App) {
        // Registered by whichever plugin is added first — the same guard
        // `SkyPlugin`, `WeatherPlugin` and `MobsPlugin` carry, for the same
        // reason: adding a plugin twice is a panic.
        if !app.is_plugin_added::<Material2dPlugin<AdditiveMaterial>>() {
            app.add_plugins(Material2dPlugin::<AdditiveMaterial>::default());
        }

        app.init_resource::<ParticleSystem>()
            .add_systems(Startup, spawn_pool)
            .add_systems(
                FixedUpdate,
                step_particles
                    // After the body and the creatures have both taken their
                    // step, so a particle emitted by either is integrated on the
                    // step it was born on rather than sitting still for one; and
                    // after the automata, so the cells it collides against are
                    // this step's and not the previous one's.
                    .after(PlayerSet::Step)
                    .after(SimSet::Simulate)
                    .run_if(crate::scenes::running)
                    .run_if(resource_exists::<SimWorld>),
            )
            .add_systems(Update, (place_particles, place_particle_glow));
    }
}

/// The one mesh every luminous particle is drawn into.
#[derive(Component)]
pub struct ParticleGlow;

/// One hidden entity per slot, once. See the header on why they are pooled.
///
/// Plus the single additive quad the glow pass draws into — one mesh rewritten
/// per frame rather than a second pool of 2048 entities, which is the trade
/// [`crate::weather`] and [`crate::mobs`] both make for the same reason: the
/// count varies every frame and the geometry is trivial.
fn spawn_pool(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut additive: ResMut<Assets<AdditiveMaterial>>,
) {
    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(additive.add(AdditiveMaterial {})),
        Transform::from_xyz(0.0, 0.0, PARTICLE_GLOW_Z),
        ParticleGlow,
        // Rewritten in place every frame, so the bounding box Bevy computed from
        // the first version of the mesh is stale immediately.
        NoFrustumCulling,
        WORLD_LAYERS,
    ));

    for slot in 0..MAX_PARTICLES {
        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::ZERO),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, PARTICLE_Z),
            ParticleSprite { slot },
            Visibility::Hidden,
            WORLD_LAYERS,
        ));
    }
}

/// Draw every self-luminous particle additively, over the light composite.
///
/// This is `drawGlow`. The `lighter` composite it set is
/// [`crate::sky::AdditiveMaterial`]'s whole content, and the reason the pass has
/// to sit above [`crate::light`] rather than beside the ordinary particles is the
/// one the original had: a mote that is the brightest thing in a cave is
/// multiplied away by the darkness the light stack composites over the world, in
/// exactly the cave where it is the only thing the player can see.
///
/// One mesh for all of them, rebuilt per frame from a `Local<VertexBuf>`, so a
/// screenful of sparks costs one draw call and — once the buffer has reached its
/// steady size — no allocation.
fn place_particle_glow(
    particles: Res<ParticleSystem>,
    mut meshes: ResMut<Assets<Mesh>>,
    quad: Query<&Mesh2d, With<ParticleGlow>>,
    mut buf: Local<VertexBuf>,
) {
    let Ok(quad) = quad.single() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&quad.0) else {
        return;
    };

    buf.clear();
    // `glow_count` is maintained incrementally, so an ordinary frame — which has
    // no luminous particles at all — skips the walk entirely. `write` stands a
    // degenerate triangle in for the empty buffer.
    if particles.glow_count() > 0 {
        for slot in 0..MAX_PARTICLES {
            let Some(p) = particles.slot(slot) else {
                continue;
            };
            if !p.glow {
                continue;
            }
            push_particle_glow(&mut buf, &p);
        }
    }
    buf.write(&mut mesh);
}

/// One luminous particle, as an additive quad. See [`place_particle_glow`].
///
/// The rounding is the sprite pool's: `Particle::x`/`y` are the TOP-LEFT of the
/// drawn square, and rounding them is what keeps a mote on the pixel grid the
/// whole game is snapped to.
fn push_particle_glow(buf: &mut VertexBuf, p: &Particle) {
    let (left, top) = (p.x.round(), p.y.round());
    let side = p.size;
    let colour = linear(
        [
            f32::from(p.rgb[0]),
            f32::from(p.rgb[1]),
            f32::from(p.rgb[2]),
        ],
        p.alpha.clamp(0.0, 1.0),
    );
    // The one convention flip: +y is up in Bevy and down in the sim.
    buf.quad(
        [
            Vec2::new(left, -top),
            Vec2::new(left + side, -top),
            Vec2::new(left + side, -(top + side)),
            Vec2::new(left, -(top + side)),
        ],
        [colour; 4],
    );
}

/// One fixed step of every live particle, against the world it can hit.
fn step_particles(mut particles: ResMut<ParticleSystem>, world: Res<SimWorld>) {
    particles.update(STEP_DT, Some(&world.level.grid));
}

/// Put every sprite on its slot, or hide it.
///
/// The TypeScript drew `fillRect(round(px), round(py), s, s)`: the TOP-LEFT
/// rounded to a whole world pixel, so a square stays crisp against the cell art
/// instead of resampling. A Bevy sprite is positioned by its CENTRE, so the
/// round happens on the corner and the half-extent is added afterwards — the
/// same rule [`crate::mobs`] places a body box with.
fn place_particles(
    particles: Res<ParticleSystem>,
    mut sprites: Query<(
        &ParticleSprite,
        &mut Sprite,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    for (which, mut sprite, mut transform, mut visibility) in &mut sprites {
        let Some(p) = particles.slot(which.slot) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        // A luminous particle is drawn by `place_particle_glow` instead, on an
        // additive mesh above the light composite. Drawing it here as well would
        // put an alpha-blended copy of it UNDER the darkness multiply, which is
        // both a double-draw and the exact thing the glow pass exists to avoid.
        if p.glow {
            *visibility = Visibility::Hidden;
            continue;
        }
        *visibility = Visibility::Inherited;
        sprite.color = Color::srgb_u8(p.rgb[0], p.rgb[1], p.rgb[2]).with_alpha(p.alpha);
        sprite.custom_size = Some(Vec2::splat(p.size));
        let half = p.size * 0.5;
        transform.translation.x = p.x.round() + half;
        // +y is up in Bevy and down in the sim: the one convention flip, in the
        // one place, exactly as `crate::player`'s body placement does it.
        transform.translation.y = -(p.y.round() + half);
        transform.translation.z = PARTICLE_Z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The glow pass sits above the whole light composite, and shares z with
    /// nothing.
    ///
    /// This is a regression test for a live z-fight, not a style rule. Luminous
    /// particles used to draw as ordinary sprites at 0.7, which is EXACTLY
    /// `crate::light::SHADOW_Z`. Two quads at one depth sort arbitrarily, so
    /// whether a spark landed in front of the darkness multiply or behind it —
    /// invisible, the precise failure the glow pass exists to prevent — was
    /// undefined.
    ///
    /// The `light` constants are private, so the numbers are restated here rather
    /// than imported. That is the weakness of this test and it is worth naming:
    /// it pins the ORDER, and it cannot see `light` moving its own quads. What it
    /// does catch is this module drifting back down into them.
    #[test]
    fn the_glow_pass_sits_above_the_light_composite_and_alone() {
        // `crate::light`'s composite quads, in order, ending with the flat washes.
        const LIGHT_COMPOSITE: [(f32, &str); 5] = [
            (0.70, "shadow"),
            (0.71, "colour"),
            (0.72, "bloom"),
            (0.75, "vignette"),
            (0.76, "wash"),
        ];
        for (z, what) in LIGHT_COMPOSITE {
            assert!(
                PARTICLE_GLOW_Z > z,
                "the glow pass at {PARTICLE_GLOW_Z} is not above light's {what} \
                 quad at {z}, so the darkness multiply would erase it"
            );
            assert_ne!(
                PARTICLE_GLOW_Z, z,
                "the glow pass shares a depth with light's {what} quad — two quads \
                 at one z sort arbitrarily"
            );
        }

        // The two orderings that are pure constants are asserted at COMPILE time
        // instead — see the `const _` beside the z constants. A constant that has
        // to wait for `cargo test` to report is a constant that can be wrong in a
        // build somebody already shipped.
    }
}
