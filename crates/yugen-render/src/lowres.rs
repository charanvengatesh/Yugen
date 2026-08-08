//! The low-resolution render target — the look of the game, not an optimisation.
//!
//! The world and the UI render into an offscreen buffer of
//! [`View::for_screen`] logical pixels, and that buffer is then upscaled to the
//! window with hard pixel edges. The TypeScript did it with a small `<canvas>`
//! and `image-rendering: pixelated`; here it is two cameras:
//!
//!   1. [`WorldCamera`] — `order: -1`, [`RenderTarget::Image`], renders
//!      [`WORLD_LAYERS`] into the canvas [`Image`] at 1 world unit per logical
//!      pixel. Everything in the game is on this camera.
//!   2. [`CanvasCamera`] — `order: 0`, renders one sprite (the canvas) to the
//!      window on [`CANVAS_LAYERS`]. Nothing else is ever on that layer.
//!
//! The nearest sampler comes from `ImagePlugin::default_nearest()` in the
//! binary. Every sampler in this game is nearest; the art is pixel art.
//!
//! # Rounding
//!
//! A fractional buffer edge resamples the whole frame, so:
//!
//!   - [`View::for_screen`] rounds the logical size to whole pixels. It already
//!     did; this module must not undo it.
//!   - The upscale is snapped to an integer whenever the zoom is within
//!     [`INTEGER_SNAP_EPS`] of one, which covers the common `zoom == 2` and
//!     `zoom == 3` cases exactly. Off an integer, the blit fills the window at
//!     the exact fractional zoom rather than letterboxing: nearest sampling
//!     keeps every edge hard, some source pixels just land two window pixels
//!     wide and some three. That is what the TypeScript's stretched canvas did,
//!     and losing a quarter of the screen to black bars is the worse trade.
//!   - The world camera is snapped to whole logical pixels, offset by a half
//!     pixel on any odd buffer axis so texel centres land on world-pixel
//!     centres rather than straddling them.
//!
//! # Resizing
//!
//! In the TypeScript, `VIEW_W`/`VIEW_H` were frozen from `window.innerWidth` at
//! import time and a resize did nothing until reload. `View::for_screen` is a
//! function here, so [`fit_canvas`] recomputes it and resizes the buffer when
//! the window changes.

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{RenderTarget, ScalingMode};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::window::PrimaryWindow;

use yugen_core::config::View;

use crate::world::WorldFocus;

/// Everything in the game world renders on this layer, into the low-res buffer.
pub const WORLD_LAYERS: RenderLayers = RenderLayers::layer(0);

/// The upscale blit renders on this layer, and nothing else ever does.
pub const CANVAS_LAYERS: RenderLayers = RenderLayers::layer(1);

/// How close to a whole number a zoom has to be to be treated as one.
///
/// `View::for_screen` divides physical extent by a float zoom, so an exactly
/// integral zoom can arrive a few ULPs off. This is wide enough to catch that
/// and far too narrow to catch a genuinely fractional zoom.
const INTEGER_SNAP_EPS: f32 = 1.0e-3;

/// What the window is cleared to outside the buffer, and what shows through
/// empty cells: the TypeScript's sky.
const SKY: Color = Color::srgb(0.42, 0.65, 0.88);

/// Renders the game world into the low-resolution buffer.
#[derive(Component)]
pub struct WorldCamera;

/// Renders the low-resolution buffer to the window.
#[derive(Component)]
pub struct CanvasCamera;

/// The sprite the buffer is blitted with.
#[derive(Component)]
pub struct CanvasSprite;

/// The offscreen buffer and the geometry that produced it.
#[derive(Resource, Clone, Debug)]
pub struct LowResTarget {
    /// The offscreen colour buffer, `view.w x view.h` logical px.
    pub canvas: Handle<Image>,
    /// Logical size and zoom for the current window.
    pub view: View,
    /// Physical window size the [`LowResTarget::view`] was computed from.
    pub physical: UVec2,
}

impl LowResTarget {
    /// Upscale factor actually used by the blit — integral when it can be.
    #[inline]
    pub fn blit_scale(&self) -> f32 {
        let z = self.view.zoom;
        let rounded = z.round();
        if rounded >= 1.0 && (z - rounded).abs() < INTEGER_SNAP_EPS {
            rounded
        } else {
            z
        }
    }

    /// Size of the blitted canvas on screen, in physical px.
    #[inline]
    pub fn blit_size(&self) -> Vec2 {
        let s = self.blit_scale();
        Vec2::new(self.view.w as f32 * s, self.view.h as f32 * s)
    }
}

/// The offscreen buffer, the two cameras, and the upscale blit.
pub struct LowResPlugin;

impl Plugin for LowResPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::BLACK))
            .add_systems(Startup, setup)
            .add_systems(Update, (fit_canvas, follow_focus));
    }
}

/// Allocate the buffer sized for the window as it opens, and spawn both cameras.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window, With<PrimaryWindow>>,
) {
    let physical = UVec2::new(window.physical_width(), window.physical_height());
    let view = View::for_screen(physical.x, physical.y);
    let canvas = images.add(new_canvas(view));
    let target = LowResTarget {
        canvas: canvas.clone(),
        view,
        physical,
    };

    commands.spawn((
        Camera2d,
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(SKY),
            ..default()
        },
        RenderTarget::Image(canvas.clone().into()),
        Msaa::Off,
        WorldCamera,
        WORLD_LAYERS,
    ));

    commands.spawn((
        Sprite {
            image: canvas,
            // Without an explicit size a sprite draws one world unit per texel,
            // which at 1 world unit per physical pixel is the buffer at 1:1 —
            // a small picture in the middle of a black window. THIS is the
            // upscale.
            custom_size: Some(target.blit_size()),
            ..default()
        },
        CanvasSprite,
        CANVAS_LAYERS,
    ));

    commands.spawn((
        Camera2d,
        Msaa::Off,
        // One world unit per PHYSICAL pixel, so the blit is expressed in the
        // same units `View::for_screen` was given. The default `WindowSize`
        // mode works in logical px, which on a 2x display would double every
        // scale computed here.
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: physical.x.max(1) as f32,
                height: physical.y.max(1) as f32,
            },
            ..OrthographicProjection::default_2d()
        }),
        CanvasCamera,
        CANVAS_LAYERS,
    ));

    commands.insert_resource(target);
}

/// A colour buffer that a camera can render into and a sprite can sample.
fn new_canvas(view: View) -> Image {
    let size = Extent3d {
        width: view.w.max(1) as u32,
        height: view.h.max(1) as u32,
        depth_or_array_layers: 1,
    };
    let mut canvas = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("yugen_lowres_canvas"),
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    canvas.resize(size);
    canvas
}

/// Follow the window: recompute the view, resize the buffer, rescale the blit.
///
/// Driven by comparing the stored physical size rather than by `WindowResized`,
/// because the window can also arrive at its real size a frame or two after
/// [`setup`] runs — on a scale-factor change, or when the compositor picks the
/// size — and that is not a resize message.
fn fit_canvas(
    window: Single<&Window, With<PrimaryWindow>>,
    mut target: ResMut<LowResTarget>,
    mut images: ResMut<Assets<Image>>,
    mut blit: Single<&mut Sprite, With<CanvasSprite>>,
    mut projection: Single<&mut Projection, With<CanvasCamera>>,
) {
    let physical = UVec2::new(window.physical_width(), window.physical_height());
    if physical == target.physical || physical.x == 0 || physical.y == 0 {
        return;
    }

    let view = View::for_screen(physical.x, physical.y);
    if view != target.view
        && let Some(mut canvas) = images.get_mut(&target.canvas)
    {
        canvas.resize(Extent3d {
            width: view.w.max(1) as u32,
            height: view.h.max(1) as u32,
            depth_or_array_layers: 1,
        });
    }

    target.view = view;
    target.physical = physical;

    if let Projection::Orthographic(ortho) = &mut **projection {
        ortho.scaling_mode = ScalingMode::Fixed {
            width: physical.x as f32,
            height: physical.y as f32,
        };
    }
    blit.custom_size = Some(target.blit_size());
}

/// Put the world camera on [`WorldFocus`], snapped so texels land on pixels.
///
/// Bevy's +y is up and the sim's is down, so this is the one place the two
/// conventions meet: everything on the world layer is placed at `-world_y`.
fn follow_focus(
    focus: Res<WorldFocus>,
    target: Res<LowResTarget>,
    mut camera: Single<&mut Transform, With<WorldCamera>>,
) {
    camera.translation.x = snap(focus.x, target.view.w);
    camera.translation.y = snap(-focus.y, target.view.h);
}

/// Snap a camera axis to the grid its buffer samples on.
///
/// An orthographic camera at `c` over an `n`-pixel buffer covers
/// `[c - n/2, c + n/2]`, so texel centres sit at `c - n/2 + i + 0.5`. Those land
/// on world-pixel centres exactly when `c - n/2` is a whole number: `c` integral
/// for an even `n`, half-integral for an odd one. Off by half a pixel and every
/// edge in the frame picks the wrong side of the source texel.
#[inline]
fn snap(v: f32, extent: i32) -> f32 {
    if extent % 2 == 0 {
        v.round()
    } else {
        (v - 0.5).round() + 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_even_buffer_snaps_to_whole_pixels_and_an_odd_one_to_halves() {
        assert_eq!(snap(10.4, 720), 10.0);
        assert_eq!(snap(10.6, 720), 11.0);
        assert_eq!(snap(10.4, 721), 10.5);
        assert_eq!(snap(10.6, 721), 10.5);
        assert_eq!(snap(-3.2, 720), -3.0);
        assert_eq!(snap(-3.2, 721), -3.5);
    }

    #[test]
    fn a_whole_zoom_blits_at_a_whole_scale() {
        // 1440x900 is zoom 2.0 exactly: 720x450 logical.
        let t = LowResTarget {
            canvas: Handle::default(),
            view: View::for_screen(1440, 900),
            physical: UVec2::new(1440, 900),
        };
        assert_eq!(t.view.zoom, 2.0);
        assert_eq!(t.blit_scale(), 2.0);
        assert_eq!(t.blit_size(), Vec2::new(1440.0, 900.0));
    }

    #[test]
    fn a_fractional_zoom_still_fills_the_window() {
        let physical = UVec2::new(2560, 1440);
        let view = View::for_screen(physical.x, physical.y);
        let t = LowResTarget {
            canvas: Handle::default(),
            view,
            physical,
        };
        assert!(t.blit_scale() > 3.0 && t.blit_scale() < 3.5, "zoom moved");
        let size = t.blit_size();
        assert!((size.x - physical.x as f32).abs() <= 2.0, "{size:?}");
        assert!((size.y - physical.y as f32).abs() <= 2.0, "{size:?}");
    }
}
