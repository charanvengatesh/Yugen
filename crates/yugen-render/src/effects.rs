//! Screenshake, the damage wash, and the additive glow blob.
//!
//! Ported from `src/render/effects.ts`.
//!
//! The three small effects that sit on top of the pipeline rather than inside
//! it. None of them owns a camera or a buffer; each is a few floats of state
//! that something else reads.
//!
//! # The feel of a hit lives here, not in the sim
//!
//! Shake and the red wash are the whole of the game's damage feedback, and they
//! are deliberately on the render side of the line: a creature landing a blow
//! writes hit points, and what that FEELS like is a decision about the screen.
//! Nothing in this module can perturb a cell, which is the same rule
//! [`crate::daynight`] states for the world clock and for the same reason.
//!
//! # Where the numbers come from
//!
//! The TypeScript kept the magnitudes at the CALL sites: `Game.spawnMobJuice`
//! wrote `shake.add(0.42)` next to `hitFlash.add(0.85)`, and the same pair of
//! literals appeared again in `spawnPlayerJuice`. They are here instead, as
//! named constants on the events that cause them, because "how hard does being
//! hit shake the screen" is one tuning decision and it was previously written
//! down in two places that had to be kept equal by hand.
//!
//! # What the port changed
//!
//! - **Shake is applied, not just exposed.** The TypeScript handed an x/y offset
//!   to `Game`, which copied it into `camera.shakeX/shakeY`; the camera stayed
//!   the single source of truth for the view transform. Here the view transform
//!   IS the [`WorldCamera`](crate::lowres::WorldCamera)'s `Transform`, so
//!   [`shake_camera`] adds the offset to it directly, in `PostUpdate` after
//!   `crate::lowres` has put the camera on the focus and before Bevy propagates
//!   it. The offset is still readable ([`Screenshake::x`]) and the resource is
//!   still the only writer of it.
//! - **The offset is rounded to whole pixels.** `crate::lowres` snaps the camera
//!   so that buffer texels land on world-pixel centres — a fractional camera
//!   resamples the entire frame. An unrounded shake would undo that snap on
//!   every shaken frame, which is a much worse artefact than a shake quantised
//!   to a pixel that is itself two or three screen pixels wide.
//! - **Randomness is a value.** `Math.random()` became [`JuiceRng`], for the
//!   reasons its own doc gives — chiefly that a screenshake nobody can predict
//!   is also a screenshake nobody can test.
//! - **The wash is an ellipse, not a circle.** `createRadialGradient` took pixel
//!   radii of `min(w,h) * 0.18` and `max(w,h) * 0.62`, describing a circle over
//!   a rectangular frame. Here it is one square texture stretched to the view,
//!   which makes it an ellipse on the same axes. At 16:10 the two differ by a few
//!   percent of alpha near the corners and not at all in what the effect
//!   communicates — and the ellipse is arguably the more correct shape, since
//!   what is being edge-lit is the VIEW, not a circle inscribed in it.
//!
//! # What the port dropped on the way in
//!
//! - **`additiveGlow`'s compositing.** The function set
//!   `globalCompositeOperation = "lighter"` and filled a radial gradient, so
//!   overlapping glows summed toward white instead of muddying. A Bevy [`Sprite`]
//!   alpha-blends and there is no per-sprite blend state without a custom
//!   material, so what survives is the SHAPE — [`glow_alpha`], the falloff, as a
//!   pure function — and not the blend. The milestone that owns the darkness
//!   multiply ([`crate::light`]) is the one that will own the material, and it
//!   should call this rather than re-deriving the ramp.
//! - **`ctx.createRadialGradient` per flash.** The TypeScript rebuilt the
//!   gradient object every frame a flash was running and said so: a healthy
//!   player pays nothing. Here the ramp is one texture built once at startup, so
//!   a hurt player does not pay either.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::transform::TransformSystems;

use yugen_core::entities::mobs::MobEventKind;

use crate::lowres::{LowResTarget, WORLD_LAYERS, WorldCamera};
use crate::particles::JuiceRng;

// ---------------------------------------------------------------------------
// Tuning. Each of these describes ONE algorithm — how trauma decays, how the
// wash falls off, how hard one event hits the screen — so they live here rather
// than in `yugen-core`'s config.
// ---------------------------------------------------------------------------

/// Camera offset at full trauma, in world px.
///
/// 14px against a ~720x450 buffer is about 2% of the view: unmistakable, and
/// still small enough that the thing that just hurt you does not leave the
/// screen while you are reacting to it.
const SHAKE_MAX_OFFSET: f32 = 14.0;

/// Trauma shed per second. At 4, a full-trauma hit is over in a quarter second.
const SHAKE_DECAY: f32 = 4.0;

/// Flash amount shed per second — a little over a third of a second from full.
const FLASH_DECAY: f32 = 3.4;

/// Peak alpha of the wash at the frame edge, at full flash.
///
/// Not 1: even at its hardest the wash is something you see THROUGH. A player
/// who has just been hit is a player who needs to see what hit them.
const FLASH_PEAK_ALPHA: f32 = 0.72;

/// Below this the flash is not drawn at all, rather than drawn at an alpha no
/// display can resolve.
const FLASH_MIN_VISIBLE: f32 = 0.01;

/// The wash's colour: a dark, desaturated red.
///
/// Dark deliberately. A saturated red reads as a UI element; this has to read as
/// blood in the corners of the player's vision.
const FLASH_RGB: [u8; 3] = [190, 20, 20];

/// Where the wash begins, as a fraction of the distance from the view centre to
/// the edge. Inside this it is fully transparent.
///
/// The whole design of this effect is that it is edge-weighted rather than a
/// flat tint: a full-screen red overlay hides the thing that just hurt you,
/// which is the one moment the player most needs to see the world.
const FLASH_INNER: f32 = 0.30;

/// Where the wash reaches full strength, in the same units. Past 1.0 — the edge
/// midpoint — so the corners are the only part that ever saturates.
const FLASH_OUTER: f32 = 1.15;

/// Edge of the square ramp texture the wash is drawn with, in texels.
///
/// One alpha step is 1/64 of the peak across the radius, which is far below what
/// a display resolves — so a smooth gradient survives being upscaled by a
/// nearest sampler, which is the only kind this game has.
const RAMP_TEXELS: u32 = 128;

/// Where the wash sits in z: over the world and the particles, under the brush
/// preview at 1.0.
///
/// Under the cursor on purpose. The wash is feedback about the player's body;
/// the cursor is feedback about the player's aim, and hiding the second behind
/// the first is how a hit turns into two mistakes.
const FLASH_Z: f32 = 0.9;

// --- How hard each event hits the screen ------------------------------------

/// Trauma from being hit by a creature, and from any other source of damage.
///
/// The biggest single number here by some way: taking a hit is the one event the
/// player must never miss, and it is the one they can do something about.
const SHAKE_HURT: f32 = 0.42;

/// Flash from the same. Nearly full — the wash is what says the damage was
/// YOURS and not something you watched happen.
const FLASH_HURT: f32 = 0.85;

/// Trauma from landing a non-fatal hit on a creature. Small: a confirmation, not
/// an event.
const SHAKE_MOB_HURT: f32 = 0.08;

/// Trauma from killing one. Larger than a hit and much smaller than being hit.
const SHAKE_MOB_DIE: f32 = 0.14;

/// Trauma from a dash. Weight, not damage.
const SHAKE_DASH: f32 = 0.1;

/// Landing impact, 0..1, below which a touchdown does not shake at all.
///
/// Walking off a ledge is not an event. Without a floor here every step on
/// uneven ground would rattle the camera.
const LAND_MIN_IMPACT: f32 = 0.2;

/// Trauma per unit of landing impact, and the cap on it.
///
/// Capped below [`SHAKE_HURT`] so that the hardest possible landing still reads
/// as softer than being hit, which is the ordering the player learns.
const LAND_SHAKE_SCALE: f32 = 0.34;
/// See [`LAND_SHAKE_SCALE`].
const LAND_SHAKE_MAX: f32 = 0.5;

// ---------------------------------------------------------------------------
// The models
// ---------------------------------------------------------------------------

/// Trauma-based screenshake.
///
/// [`Screenshake::add`] accumulates magnitude; [`Screenshake::update`] decays it
/// and resamples a fresh offset. Trauma is clamped at 1 so that a pile-up of
/// events cannot leave the camera rattling long after the fight ended — the
/// classic failure of a shake that sums durations instead of energy.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Screenshake {
    /// Current shake energy, 0..1, decaying to 0.
    trauma: f32,
    /// The resampled offset, world px. See [`Screenshake::x`].
    off: Vec2,
    /// The stream the offset is drawn from — see [`JuiceRng`] on why it is not
    /// the sim's.
    rng: JuiceRng,
    /// Player-set magnitude, 0..1. Applied where trauma is ADDED rather than
    /// where the offset is read, so turning shake down also shortens how long
    /// it lasts — a decaying half-strength shake, not a full-length one drawn
    /// smaller. Zero means the camera never moves at all.
    pub scale: f32,
}

impl Screenshake {
    /// A camera at rest.
    pub fn new() -> Screenshake {
        Screenshake {
            trauma: 0.0,
            off: Vec2::ZERO,
            rng: JuiceRng::new(),
            scale: 1.0,
        }
    }

    /// A camera at rest whose jitter comes off `seed`.
    pub fn seeded(seed: u32) -> Screenshake {
        Screenshake {
            rng: JuiceRng::seeded(seed),
            ..Screenshake::new()
        }
    }

    /// Add shake energy. Clamped, so nothing rattles forever.
    pub fn add(&mut self, magnitude: f32) {
        self.trauma = (self.trauma + magnitude * self.scale.clamp(0.0, 1.0)).min(1.0);
    }

    /// Decay the trauma and resample the offset.
    ///
    /// The offset is squared trauma, not trauma: the shake ramps in hard on the
    /// frame of the hit and then tails off smoothly, instead of falling linearly
    /// and reading as a mechanical slide back to centre.
    pub fn update(&mut self, dt: f32) {
        if self.trauma <= 0.0 {
            self.off = Vec2::ZERO;
            return;
        }
        self.trauma = (self.trauma - SHAKE_DECAY * dt).max(0.0);
        let amt = self.trauma * self.trauma * SHAKE_MAX_OFFSET;
        self.off = Vec2::new(
            self.rng.rand_range(-1.0, 1.0) * amt,
            self.rng.rand_range(-1.0, 1.0) * amt,
        );
    }

    /// Horizontal offset, world px.
    ///
    /// Sim-space and Bevy-space agree on x, and the y offset is a SYMMETRIC
    /// random jitter — there is no down for it to point at. So this module has no
    /// convention flip in it, which is the only reason a screen-space effect gets
    /// away without one.
    #[inline]
    pub fn x(&self) -> f32 {
        self.off.x
    }

    /// Vertical offset, world px. See [`Screenshake::x`] on the missing flip.
    #[inline]
    pub fn y(&self) -> f32 {
        self.off.y
    }

    /// Remaining shake energy, 0..1. For a test, and for a HUD that wants to know
    /// whether anything is happening.
    #[inline]
    pub fn trauma(&self) -> f32 {
        self.trauma
    }

    /// Shake for one creature event.
    ///
    /// The mapping, not just the magnitudes: a creature hurting the PLAYER shakes
    /// several times harder than the player killing a creature, because the two
    /// events need to be told apart without looking at a health bar.
    pub fn mob_event(&mut self, kind: MobEventKind) {
        match kind {
            MobEventKind::PlayerHit => self.add(SHAKE_HURT),
            MobEventKind::MobHurt => self.add(SHAKE_MOB_HURT),
            MobEventKind::MobDie => self.add(SHAKE_MOB_DIE),
            // A chill is a status with a duration. Shaking for the whole of it
            // would be a rattle, and shaking once at the start would say
            // "impact", which is the wrong thing to say.
            MobEventKind::PlayerChill => {}
        }
    }

    /// Shake for taking damage from anything — a creature, lava, a fall.
    pub fn hurt(&mut self) {
        self.add(SHAKE_HURT);
    }

    /// Shake for a touchdown. `impact` is the landing speed normalised to 0..1.
    ///
    /// Soft landings do not shake at all; see [`LAND_MIN_IMPACT`].
    pub fn land(&mut self, impact: f32) {
        if impact > LAND_MIN_IMPACT {
            self.add((impact * LAND_SHAKE_SCALE).min(LAND_SHAKE_MAX));
        }
    }

    /// Shake for a dash.
    pub fn dash(&mut self) {
        self.add(SHAKE_DASH);
    }
}

impl Default for Screenshake {
    fn default() -> Screenshake {
        Screenshake::new()
    }
}

/// Damage feedback: a red wash that floods in from the frame edges and drains
/// away over a few tenths of a second. Screen space.
///
/// Deliberately edge-weighted rather than a full-screen tint — see
/// [`FLASH_INNER`] for the argument, which is the one thing about this effect
/// that must not be "simplified" later.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct HitFlash {
    /// How much flash is left, 0..1.
    amt: f32,
}

impl HitFlash {
    /// A screen with no flash on it.
    pub fn new() -> HitFlash {
        HitFlash { amt: 0.0 }
    }

    /// Trigger, or reinforce, a flash. `magnitude` is 0..1.
    pub fn add(&mut self, magnitude: f32) {
        self.amt = (self.amt + magnitude).min(1.0);
    }

    /// Drain.
    pub fn update(&mut self, dt: f32) {
        self.amt = (self.amt - dt * FLASH_DECAY).max(0.0);
    }

    /// How much flash is left, 0..1, before the ease.
    #[inline]
    pub fn amount(&self) -> f32 {
        self.amt
    }

    /// Peak alpha of the wash this frame, 0 when there is nothing to draw.
    ///
    /// Squared, so the flash punches on the first frame and then lingers faintly
    /// rather than draining at a constant rate. The visibility floor is the
    /// TypeScript's early return, kept because it is what makes a healthy player
    /// cost nothing at all.
    #[inline]
    pub fn alpha(&self) -> f32 {
        if self.amt <= FLASH_MIN_VISIBLE {
            return 0.0;
        }
        self.amt * self.amt * FLASH_PEAK_ALPHA
    }

    /// Flash for one creature event.
    ///
    /// Only the player being hit flashes. The screen is the PLAYER's point of
    /// view, and washing it red because something else was hurt would be lying
    /// about whose blood it is.
    pub fn mob_event(&mut self, kind: MobEventKind) {
        if kind == MobEventKind::PlayerHit {
            self.add(FLASH_HURT);
        }
    }

    /// Flash for taking damage from anything.
    pub fn hurt(&mut self) {
        self.add(FLASH_HURT);
    }
}

/// Wash weight at a point, given its distance from the view centre as a fraction
/// of the distance to the edge.
///
/// `0` at and inside [`FLASH_INNER`], ramping to `1` at [`FLASH_OUTER`]. This is
/// the whole shape of the effect, and it is a free function so that the texture
/// builder and the tests are looking at the same ramp.
#[inline]
fn edge_wash(u: f32) -> f32 {
    ((u - FLASH_INNER) / (FLASH_OUTER - FLASH_INNER)).clamp(0.0, 1.0)
}

/// Alpha of an additive glow blob at `dist` from its centre.
///
/// Linear from `alpha` at the centre to fully transparent at `radius`, which is
/// what `createRadialGradient(x, y, 0, x, y, radius)` with two stops was. This is
/// what is left of `additiveGlow`; see the header for what is not, and why the
/// shape was worth keeping without it.
#[inline]
pub fn glow_alpha(dist: f32, radius: f32, alpha: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    alpha * (1.0 - dist / radius).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// The Bevy half
// ---------------------------------------------------------------------------

/// The sprite the damage wash is drawn with. One, for the whole run.
#[derive(Component)]
pub struct HitFlashSprite;

/// Everything that reacts to a hit, in one place a caller can ask for.
///
/// A [`SystemParam`] rather than two `ResMut`s at every call site: shake and
/// flash are never triggered independently — every event that causes one is a
/// candidate for the other — and bundling them means the seams that will consume
/// [`MobEvent`](yugen_core::entities::mobs::MobEvent)s take ONE parameter and
/// make ONE call.
///
/// It carries no judgement of its own. Which event is worth how much is on
/// [`Screenshake`] and [`HitFlash`], where it can be tested without an ECS.
#[derive(SystemParam)]
pub struct Feedback<'w> {
    shake: ResMut<'w, Screenshake>,
    flash: ResMut<'w, HitFlash>,
}

impl Feedback<'_> {
    /// React to one creature event.
    ///
    /// This is the wiring point: `crate::mobs::step_creatures` drains these
    /// events and discards them, and this call plus
    /// [`ParticleSystem::mob_event`](crate::particles::ParticleSystem::mob_event)
    /// is the whole of what it should do with them instead.
    pub fn mob_event(&mut self, kind: MobEventKind) {
        self.shake.mob_event(kind);
        self.flash.mob_event(kind);
    }

    /// React to the player taking damage from any source. The other wiring
    /// point: `crate::player::step_player`'s drained `PlayerEvent::Hurt`.
    pub fn hurt(&mut self) {
        self.shake.hurt();
        self.flash.hurt();
    }

    /// React to a touchdown. `impact` is 0..1; see [`Screenshake::land`].
    pub fn land(&mut self, impact: f32) {
        self.shake.land(impact);
    }

    /// React to a dash.
    pub fn dash(&mut self) {
        self.shake.dash();
    }
}

/// The two feedback resources, their decay, and the wash on screen.
///
/// Not in [`YugenRenderPlugin`](crate::YugenRenderPlugin) yet: nothing calls
/// [`Feedback`] until the mob and player event drains are wired, so the screen
/// would be permanently calm.
pub struct EffectsPlugin;

impl Plugin for EffectsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Screenshake>()
            .init_resource::<HitFlash>()
            .add_systems(Startup, spawn_flash)
            // Once per FRAME, not per fixed step. Both of these are decays of a
            // thing the eye is watching, and a 120 Hz decay under a 60 Hz screen
            // would resample the shake offset twice for every frame that could
            // show it — half the samples drawn, and a shake that reads as noise
            // rather than as motion.
            .add_systems(Update, tick_effects)
            .add_systems(
                PostUpdate,
                // After `crate::lowres` has put the camera on the focus in
                // `Update`, and before Bevy propagates the transform, so the
                // offset lands on the frame that caused it.
                (shake_camera, place_flash)
                    .chain()
                    .before(TransformSystems::Propagate),
            );
    }
}

/// Decay both effects by one frame.
fn tick_effects(time: Res<Time>, mut shake: ResMut<Screenshake>, mut flash: ResMut<HitFlash>) {
    let dt = time.delta_secs();
    shake.update(dt);
    flash.update(dt);
}

/// Build the wash's ramp texture and the one sprite that draws it.
fn spawn_flash(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let ramp = images.add(ramp_texture());
    commands.spawn((
        Sprite {
            image: ramp,
            color: Color::NONE,
            custom_size: Some(Vec2::ZERO),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, FLASH_Z),
        HitFlashSprite,
        Visibility::Hidden,
        WORLD_LAYERS,
    ));
}

/// A white square whose ALPHA is [`edge_wash`] of the distance from its centre.
///
/// White, and not red: the colour arrives through the sprite's tint, so the same
/// texture can serve any wash a later effect wants without a second allocation.
fn ramp_texture() -> Image {
    let n = RAMP_TEXELS as usize;
    let mut data = vec![0u8; n * n * 4];
    let half = (RAMP_TEXELS as f32 - 1.0) * 0.5;
    for y in 0..n {
        for x in 0..n {
            // Normalised to the half-extent on each axis, so stretching the
            // square to the view turns this circle into the view's own ellipse.
            let dx = (x as f32 - half) / half;
            let dy = (y as f32 - half) / half;
            let a = edge_wash(dx.hypot(dy));
            let px = (y * n + x) * 4;
            data[px] = 255;
            data[px + 1] = 255;
            data[px + 2] = 255;
            data[px + 3] = (a * 255.0).round() as u8;
        }
    }
    Image::new(
        Extent3d {
            width: RAMP_TEXELS,
            height: RAMP_TEXELS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Offset the world camera by the current shake.
///
/// Additive on top of whatever `crate::lowres` put there this frame, and safe to
/// be: that system writes the translation absolutely from the focus every frame,
/// so the offset cannot accumulate across frames.
///
/// Rounded, because a fractional camera resamples the whole low-res buffer. See
/// the header.
fn shake_camera(shake: Res<Screenshake>, camera: Single<&mut Transform, With<WorldCamera>>) {
    let mut transform = camera.into_inner();
    transform.translation.x += shake.x().round();
    transform.translation.y += shake.y().round();
}

/// The one sprite the wash is drawn with, as [`place_flash`] borrows it.
///
/// Named rather than written inline, and the `Without` in it is the reason it
/// needs a name: the same system also reads the world camera's `Transform`, and
/// Bevy will not hand out two overlapping accesses to one component without
/// being told, in the filters, that the two sets cannot intersect.
type WashSprite<'w, 's> = Single<
    'w,
    's,
    (
        &'static mut Sprite,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    (With<HitFlashSprite>, Without<WorldCamera>),
>;

/// Size the wash to the view, put it on the camera, and set its strength.
///
/// It follows the camera — including the shake applied a moment ago — because it
/// is a SCREEN-space effect living on a world-space layer. There is no HUD camera
/// yet; when there is, this sprite moves to it and this system loses its camera
/// query and nothing else.
fn place_flash(
    flash: Res<HitFlash>,
    target: Res<LowResTarget>,
    camera: Single<&Transform, With<WorldCamera>>,
    wash: WashSprite,
) {
    let (mut sprite, mut transform, mut visibility) = wash.into_inner();
    let alpha = flash.alpha();
    if alpha <= 0.0 {
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Inherited;
    sprite.color = Color::srgb_u8(FLASH_RGB[0], FLASH_RGB[1], FLASH_RGB[2]).with_alpha(alpha);
    sprite.custom_size = Some(Vec2::new(target.view.w as f32, target.view.h as f32));
    transform.translation.x = camera.translation.x;
    transform.translation.y = camera.translation.y;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 60 Hz frame — the clock both effects decay on.
    const FRAME: f32 = 1.0 / 60.0;

    #[test]
    fn trauma_accumulates_and_clamps_at_one() {
        let mut s = Screenshake::new();
        s.add(0.4);
        s.add(0.4);
        assert!((s.trauma() - 0.8).abs() < 1e-6);
        s.add(0.9);
        assert_eq!(s.trauma(), 1.0, "a pile-up of hits went past full trauma");
    }

    #[test]
    fn a_camera_at_rest_has_no_offset() {
        let mut s = Screenshake::new();
        s.update(FRAME);
        assert_eq!((s.x(), s.y()), (0.0, 0.0));
    }

    #[test]
    fn trauma_decays_to_nothing_and_the_offset_goes_with_it() {
        let mut s = Screenshake::new();
        s.add(1.0);
        // Full trauma at SHAKE_DECAY per second is a quarter of a second. Plus
        // one frame: fifteen subtractions of `4/60` land a few ULPs above zero
        // rather than on it, and the clamp is what turns that into a stop.
        let frames = (1.0 / SHAKE_DECAY / FRAME).ceil() as u32 + 1;
        for _ in 0..frames {
            s.update(FRAME);
        }
        assert_eq!(s.trauma(), 0.0, "still shaking after the decay window");
        // One more step to resample: trauma reaching zero is what zeroes the
        // offset, and the camera must not be left parked off-centre.
        s.update(FRAME);
        assert_eq!((s.x(), s.y()), (0.0, 0.0));
    }

    #[test]
    fn the_offset_never_exceeds_the_maximum_in_either_axis() {
        let mut s = Screenshake::seeded(7);
        for _ in 0..500 {
            s.add(1.0);
            s.update(FRAME);
            assert!(s.x().abs() <= SHAKE_MAX_OFFSET, "x was {}", s.x());
            assert!(s.y().abs() <= SHAKE_MAX_OFFSET, "y was {}", s.y());
        }
    }

    #[test]
    fn the_shake_ramps_in_hard_and_tails_off_smoothly() {
        // Squared trauma: at half trauma the bound is a QUARTER of the maximum,
        // not half of it. That curve is the difference between a punch and a
        // mechanical slide back to centre.
        let bound = |trauma: f32| trauma * trauma * SHAKE_MAX_OFFSET;
        assert!((bound(1.0) - SHAKE_MAX_OFFSET).abs() < 1e-6);
        assert!((bound(0.5) - SHAKE_MAX_OFFSET * 0.25).abs() < 1e-6);

        // And the sampled offsets obey it: after one step from full trauma the
        // bound has already dropped below the maximum.
        let mut s = Screenshake::seeded(11);
        s.add(1.0);
        s.update(FRAME);
        assert!(s.x().abs() <= bound(s.trauma()) + 1e-4);
        assert!(s.y().abs() <= bound(s.trauma()) + 1e-4);
    }

    #[test]
    fn the_offset_is_resampled_every_step_rather_than_eased() {
        // A shake that moved smoothly would be a camera drift. Two consecutive
        // samples at the same trauma must be independent draws.
        let mut s = Screenshake::seeded(3);
        s.add(1.0);
        s.update(FRAME);
        let first = (s.x(), s.y());
        s.add(1.0); // back to full, so only the resample can have changed it
        s.update(FRAME);
        assert_ne!(first, (s.x(), s.y()));
    }

    #[test]
    fn a_flash_clamps_at_one_and_drains_to_nothing() {
        let mut f = HitFlash::new();
        f.add(0.85);
        f.add(0.85);
        assert_eq!(f.amount(), 1.0);
        let frames = (1.0 / FLASH_DECAY / FRAME).ceil() as u32;
        for _ in 0..frames {
            f.update(FRAME);
        }
        assert_eq!(f.amount(), 0.0);
        assert_eq!(f.alpha(), 0.0);
    }

    #[test]
    fn the_flash_is_not_drawn_below_the_visibility_floor() {
        let mut f = HitFlash::new();
        f.add(FLASH_MIN_VISIBLE);
        assert_eq!(f.alpha(), 0.0, "a flash nobody could see was still drawn");
        f.add(0.5);
        assert!(f.alpha() > 0.0);
    }

    #[test]
    fn the_flash_eases_out_so_it_punches_on_the_first_frame() {
        let mut full = HitFlash::new();
        full.add(1.0);
        assert!((full.alpha() - FLASH_PEAK_ALPHA).abs() < 1e-6);
        // Half-drained is a QUARTER as bright, not half: the tail is faint.
        let mut half = HitFlash::new();
        half.add(0.5);
        assert!((half.alpha() - FLASH_PEAK_ALPHA * 0.25).abs() < 1e-6);
        assert!(full.alpha() > half.alpha() * 2.0);
    }

    #[test]
    fn the_wash_is_clear_at_the_centre_and_solid_at_the_corners() {
        assert_eq!(edge_wash(0.0), 0.0, "the middle of the screen was tinted");
        assert_eq!(edge_wash(FLASH_INNER), 0.0);
        // The edge midpoint is partway up the ramp; only the corner saturates.
        let edge = edge_wash(1.0);
        assert!(edge > 0.0 && edge < 1.0, "edge weight was {edge}");
        assert_eq!(edge_wash(FLASH_OUTER), 1.0);
        assert_eq!(edge_wash(9.0), 1.0, "the ramp ran past full");

        // Monotonic, which is what makes it read as coming FROM the edges.
        let mut last = 0.0;
        for i in 0..=100 {
            let w = edge_wash(i as f32 / 50.0);
            assert!(w >= last, "the wash dipped at u = {}", i as f32 / 50.0);
            last = w;
        }
    }

    #[test]
    fn a_glow_blob_is_brightest_at_its_centre_and_gone_at_its_radius() {
        assert_eq!(glow_alpha(0.0, 10.0, 0.8), 0.8);
        assert!((glow_alpha(5.0, 10.0, 0.8) - 0.4).abs() < 1e-6);
        assert_eq!(glow_alpha(10.0, 10.0, 0.8), 0.0);
        assert_eq!(glow_alpha(99.0, 10.0, 0.8), 0.0, "the blob had a tail");
        // A degenerate radius is a blob with no area, not a division by zero.
        assert_eq!(glow_alpha(0.0, 0.0, 0.8), 0.0);
    }

    #[test]
    fn being_hit_shakes_harder_than_anything_the_player_does_to_a_creature() {
        // The ordering IS the feedback. A player who cannot tell "I was hit" from
        // "I hit something" by feel alone is reading a health bar instead of
        // playing.
        let trauma_from = |kind| {
            let mut s = Screenshake::new();
            s.mob_event(kind);
            s.trauma()
        };
        let hit = trauma_from(MobEventKind::PlayerHit);
        let die = trauma_from(MobEventKind::MobDie);
        let hurt = trauma_from(MobEventKind::MobHurt);
        assert!(hit > die, "being hit shook no harder than landing a kill");
        assert!(die > hurt, "a kill shook no harder than a scratch");
        assert!(hurt > 0.0, "a landed hit did not register at all");
        assert_eq!(trauma_from(MobEventKind::PlayerChill), 0.0);
    }

    #[test]
    fn only_the_players_own_wound_washes_the_screen_red() {
        let flash_from = |kind| {
            let mut f = HitFlash::new();
            f.mob_event(kind);
            f.amount()
        };
        assert!(flash_from(MobEventKind::PlayerHit) > 0.0);
        assert_eq!(flash_from(MobEventKind::MobHurt), 0.0);
        assert_eq!(flash_from(MobEventKind::MobDie), 0.0, "whose blood is that");
        assert_eq!(flash_from(MobEventKind::PlayerChill), 0.0);

        // And any other source of damage flashes exactly as hard as a creature
        // does: as far as the screen is concerned, lava hurts the same.
        let mut lava = HitFlash::new();
        lava.hurt();
        assert_eq!(lava.amount(), flash_from(MobEventKind::PlayerHit));
    }

    #[test]
    fn a_soft_landing_does_not_shake_the_camera_at_all() {
        let mut soft = Screenshake::new();
        soft.land(LAND_MIN_IMPACT);
        assert_eq!(soft.trauma(), 0.0, "walking off a kerb shook the screen");

        let mut hard = Screenshake::new();
        hard.land(1.0);
        assert!(hard.trauma() > 0.0);
        assert!(
            hard.trauma() <= LAND_SHAKE_MAX,
            "a landing went past its cap"
        );

        // Even the hardest landing stays under being hit — the player has to be
        // able to tell falling from being attacked with their eyes shut.
        let mut wounded = Screenshake::new();
        wounded.hurt();
        assert!(hard.trauma() < wounded.trauma());
    }

    #[test]
    fn the_ramp_texture_is_transparent_in_the_middle_and_opaque_in_the_corners() {
        let image = ramp_texture();
        let data = image.data.as_ref().expect("the ramp has no pixels");
        let n = RAMP_TEXELS as usize;
        assert_eq!(data.len(), n * n * 4);

        let alpha_at = |x: usize, y: usize| data[(y * n + x) * 4 + 3];
        assert_eq!(alpha_at(n / 2, n / 2), 0, "the centre was tinted");
        assert_eq!(alpha_at(0, 0), 255, "the corner never saturated");
        assert_eq!(alpha_at(n - 1, n - 1), 255);
        // Symmetric about both axes, or the wash would visibly lean.
        assert_eq!(alpha_at(0, n / 2), alpha_at(n - 1, n / 2));
        assert_eq!(alpha_at(n / 2, 0), alpha_at(n / 2, n - 1));
        // And the edge midpoint is partway up, not saturated: the corners are
        // the only part that ever reaches full.
        let edge = alpha_at(n / 2, 0);
        assert!(edge > 0 && edge < 255, "edge alpha was {edge}");
    }
}
