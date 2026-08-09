//! The cell shader must produce what `cells::paint_cells` produces.
//!
//! # What this is
//!
//! `cells.rs` is a byte-for-byte port of the TypeScript rasteriser, frozen
//! against it by `cells_golden.rs`. `cells.wgsl` is that same pass on the GPU.
//! This file closes the chain: it compiles the SHIPPING shader source, runs it on
//! a headless adapter over a real worldgen window, reads the framebuffer back and
//! diffs it against `paint_cells` over the same grid.
//!
//! There is no window, no Bevy app and no swapchain here. `wgpu` will hand out an
//! adapter with no surface attached, the pass renders into an ordinary
//! `Rgba8Unorm` texture, and `copy_texture_to_buffer` brings the pixels back.
//!
//! # Why the target is `Rgba8Unorm` and not an sRGB format
//!
//! `paint_cells` packs the AUTHORED sRGB bytes — `MAT_R`/`MAT_G`/`MAT_B` are sRGB
//! and every clamp in `cells.rs` happens on them. `cell_color` returns those same
//! bytes over 255, so a plain `Rgba8Unorm` target stores them back unchanged and
//! the comparison is exact by construction, with no colour-space round trip in
//! the middle to argue about. The GAME's target is `Bgra8UnormSrgb`, so
//! `cellmap.wgsl` converts to linear on the way out and the hardware encodes it
//! back; that conversion is the one line of the game shader this harness does not
//! exercise, and it is the one line that is not arithmetic from `cells.rs`.
//!
//! # What is exact and what is not
//!
//! Two claims, and between them they account for every pixel:
//!
//!   - A material that does not declare shimmer is a pure TABLE LOOKUP in both
//!     implementations — the same `SHADE32` word, fetched by the same
//!     `(id, edge class, pattern)` index. Those pixels must be EXACTLY equal, and
//!     [`at_rest_materials_are_byte_identical_to_the_cpu_blit`] asserts it over
//!     ~180 000 cells of real terrain. Everything structural rides on this: the
//!     three-tap `depth_above`, `side_open`, both coprime tile lookups, the
//!     shade-table indexing, the window-edge rules and the transparent air.
//!
//!   - A material that DOES declare shimmer has its colour recomputed per
//!     fragment, because that is the whole point of moving the animation to a
//!     uniform. The CPU does that arithmetic in f64 and the GPU in f32, so the two
//!     cannot agree to the last bit and it would be dishonest to assert they do.
//!     [`the_shimmer_matches_the_cpu_blit_to_a_measured_bound`] quantifies the
//!     gap instead: max channel delta, how many pixels move, and where.
//!
//! If no GPU adapter is available the tests SKIP rather than pass — see
//! [`Harness::new`]. A silent pass on a machine with no GPU would be the worst
//! outcome of the three.

use std::borrow::Cow;
use std::collections::BTreeMap;

use bevy::math::{IVec2, Vec4};
use bevy::render::render_resource::ShaderType;
use bevy::render::render_resource::encase::UniformBuffer;
use bevy::tasks::block_on;
use yugen_core::config::{CHUNK_CELLS, WINDOW_COLS, WINDOW_ROWS, WorldScale};
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::MAT_COUNT;
use yugen_core::sim::worldgen::ChunkGen;
use yugen_render::cellmap::{CellShadeParams, MATERIAL_SLOTS, build_params, new_shade_texture};
use yugen_render::cells::{
    CellShades, SHADE_EDGE_GAIN, SHADE_EDGE_SCALE, SHADE_MATERIAL_STRIDE, SHADE_PAT_MID,
    SHADE_PAT_SIGMA, SHIMMER_SIN_SCALE, SHIMMER_SIN_SIZE, TEX_A, TEX_A_PERIOD, TEX_B, TEX_B_PERIOD,
    TEX_GRAIN, TEX_PATTERN_COUNT, paint_cells_fine, shimmer_params,
};

/// The shipping shading module, included as SOURCE rather than re-implemented.
/// If this file and the game ever compile different text, the harness is
/// worthless — so it reads the same bytes the plugin embeds.
const SHADING_WGSL: &str = include_str!("../src/cells.wgsl");

/// The seed the TypeScript fixture was dumped at. Reused so the windows below
/// are the ones `cells_golden.rs` has already frozen pixel for pixel.
const SEED: u32 = 2334;

/// The two windows, as chunk coordinates of their top-left chunk.
///
/// These are the parity fixture's own, and they are chosen for the same reason:
/// every interesting branch in the blit is a property of the MATERIAL
/// DISTRIBUTION. `surface` straddles the ground line, so it carries air, top
/// faces, overhangs and cave walls — the entire edge-class range. `depths` is
/// 38 000 cells of lava, which is the only way the shimmer branch gets exercised
/// at all. Both sit at NEGATIVE grid origins on both axes, so every `pmod` in the
/// pattern lookup is a real negative-input modulo.
/// The `depths` chunk ROW is scaled: the underworld sits `UNDERWORLD_DEPTH` cells
/// below the local surface and that is a legacy depth, so a fixed chunk row stops
/// naming the lava the moment the world scales. At `WorldScale::LIVE` the
/// unscaled row 14 put this window in ordinary deep stone and the shimmer branch
/// went from >100 000 animated texels to 5 960 — a guard on its way to passing
/// vacuously, which the coverage assertion below caught.
fn windows() -> [(&'static str, i32, i32); 2] {
    let deep = (14.0 * WorldScale::LIVE.factor()) as i32;
    [("surface", -5, -1), ("depths", -5, deep)]
}

/// Clock samples, chosen to cover BOTH regimes of the f64/f32 gap.
///
/// The shimmer phase is `(t * 2.1 + phase) * SIN_SCALE`, truncated to a
/// sine-table index. `SIN_SCALE` is 326, so the index grows at ~685 per second
/// of clock and an f32 holds it exactly until it passes 2^24 — a little under an
/// hour of play. Below that the two implementations agree to the byte; above it
/// the index can land one slot either side and the truncated channel can move by
/// one.
///
/// So: one frame, a few seconds, two minutes, thirty minutes — all exact — then
/// one hour and ten hours, which are where the divergence actually lives. The
/// numbers the test prints are the honest statement of where the line is.
const CLOCKS: [f32; 8] = [0.0, 0.0166667, 1.25, 7.5, 123.456, 1800.0, 3600.0, 36000.0];

// --- The grid ----------------------------------------------------------------

/// One streaming window filled from the Rust worldgen.
fn window_grid(chunk_x0: i32, chunk_y0: i32) -> CellGrid {
    let chunk_cols = WINDOW_COLS / CHUNK_CELLS;
    let chunk_rows = WINDOW_ROWS / CHUNK_CELLS;
    let mut grid = CellGrid::new(WINDOW_COLS, WINDOW_ROWS);
    grid.set_origin(chunk_x0 * CHUNK_CELLS, chunk_y0 * CHUNK_CELLS);

    let mut chunks = ChunkGen::new(SEED);
    for j in 0..chunk_rows {
        for k in 0..chunk_cols {
            let chunk = chunks.generate(chunk_x0 + k, chunk_y0 + j);
            for ly in 0..CHUNK_CELLS {
                let src = (ly * CHUNK_CELLS) as usize;
                let dst = ((j * CHUNK_CELLS + ly) * WINDOW_COLS + k * CHUNK_CELLS) as usize;
                grid.material[dst..dst + CHUNK_CELLS as usize]
                    .copy_from_slice(&chunk[src..src + CHUNK_CELLS as usize]);
            }
        }
    }
    grid
}

/// Whether a material's colour is recomputed per fragment rather than read from
/// the shade table. The shader branches on exactly this, via `shim.x > 0`.
fn animates(id: u16) -> bool {
    id != 0 && shimmer_params(id as usize).amp > 0.0
}

// --- The shader --------------------------------------------------------------

/// The harness's own bindings and entry points.
///
/// Deliberately tiny. Everything below the `fs_main` call is `cells.wgsl`, which
/// is the file under test; the only thing stated twice is the binding table,
/// which differs between Bevy and raw wgpu by construction. `fs_main` derives the
/// cell from `@builtin(position)` where `cellmap.wgsl` derives it from the quad's
/// uv — one integer truncation either way, and the ONE thing this harness cannot
/// check for the game.
const HARNESS_WGSL: &str = r#"
@group(0) @binding(0) var cell_ids: texture_2d<u32>;
@group(0) @binding(1) var tex_a: texture_2d<u32>;
@group(0) @binding(2) var tex_b: texture_2d<u32>;
@group(0) @binding(3) var shade: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: CellShadeParams;

// A full-screen triangle: (-1,-1), (-1,3), (3,-1). The target is GRAIN texels
// per cell per axis, so `@builtin(position)` lands on FINE texel centres; the
// cell and the sub-offset are derived from the integer fine coordinate — no
// float uv in sight, which is what makes this half of the comparison exact by
// construction. (The game's `cellmap.wgsl` derives fine from a float uv; that
// quantisation is the one line this harness cannot exercise, as its header has
// always said of the uv path.)
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let i = i32(index);
    return vec4<f32>(f32(i / 2) * 4.0 - 1.0, f32(i % 2) * 4.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let fine = vec2<i32>(pos.xy);
    let cell = fine / GRAIN;
    let sub = fine - cell * GRAIN;
    let id = min(textureLoad(cell_ids, cell, 0).r, 63u);
    return cell_color(
        cell_ids,
        tex_a,
        tex_b,
        shade,
        cell,
        sub,
        params.origin,
        id,
        params.base[id],
        params.shim[id],
        params.clock,
    );
}
"#;

/// `cells.wgsl` with its one naga_oil directive removed, plus the harness entry
/// points.
///
/// The strip is asserted to remove EXACTLY the `#define_import_path` line. If a
/// `#ifdef` is ever added to the shared module, this fails loudly rather than
/// quietly testing a different shader from the one the game runs.
fn shader_source() -> String {
    let directives: Vec<&str> = SHADING_WGSL
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .collect();
    assert_eq!(
        directives,
        vec!["#define_import_path yugen::cells"],
        "cells.wgsl grew a preprocessor directive; the harness would be compiling \
         different source from the game"
    );
    let body: String = SHADING_WGSL
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{body}\n{HARNESS_WGSL}")
}

// --- The headless device -----------------------------------------------------

/// A device, a pipeline, and the four lookup textures — built once per render.
struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    views: Vec<wgpu::TextureView>,
}

impl Harness {
    /// Bring up a headless adapter, or return `None` if this machine has no GPU
    /// wgpu will talk to.
    ///
    /// The callers SKIP on `None` and say so on stdout. They do not pass: a
    /// green tick from a machine that never ran the shader is worse than a red
    /// one, because it is the claim this whole file exists to make.
    fn new() -> Option<Harness> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // No `compatible_surface`: there is no window, and an adapter does not
        // need one to render into a texture.
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yugen cell parity"),
            ..Default::default()
        }))
        .ok()?;

        // A panic beats a silently wrong frame: a validation error here would
        // otherwise surface as a black readback and be read as a parity failure.
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e}")));

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cells.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });

        let uint_tex = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cell bindings"),
            entries: &[
                entry(0, uint_tex),
                entry(1, uint_tex),
                entry(2, uint_tex),
                entry(
                    3,
                    wgpu::BindingType::Texture {
                        // `filterable: false` — the shade table is read with
                        // `textureLoad` and one texel is one table entry.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    4,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cell pass"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                // No blending: the harness wants the shader's own output bytes,
                // not the result of compositing them over anything.
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let mut harness = Harness {
            device,
            queue,
            pipeline,
            layout,
            views: Vec::new(),
        };
        // The three static lookups: both pattern tiles and the shade table. They
        // are uploaded ONCE, here, exactly as the game uploads them once at
        // startup — they are pure functions of the content build.
        harness.views.push(harness.upload(
            "tex_a",
            &TEX_A,
            TEX_A_PERIOD as u32,
            TEX_A_PERIOD as u32 * TEX_PATTERN_COUNT as u32,
            wgpu::TextureFormat::R8Uint,
            1,
        ));
        harness.views.push(harness.upload(
            "tex_b",
            &TEX_B,
            TEX_B_PERIOD as u32,
            TEX_B_PERIOD as u32 * TEX_PATTERN_COUNT as u32,
            wgpu::TextureFormat::R8Uint,
            1,
        ));
        let shade = shade_bytes();
        harness.views.push(harness.upload(
            "shade",
            &shade,
            SHADE_MATERIAL_STRIDE as u32,
            MAT_COUNT as u32,
            wgpu::TextureFormat::Rgba8Unorm,
            4,
        ));
        Some(harness)
    }

    fn upload(
        &self,
        label: &str,
        data: &[u8],
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        bytes_per_texel: u32,
    ) -> wgpu::TextureView {
        assert_eq!(
            data.len() as u32,
            width * height * bytes_per_texel,
            "{label}"
        );
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // `write_texture`, not `copy_buffer_to_texture`: it has no 256-byte row
        // alignment rule, and 61-byte and 704-byte rows are exactly what this
        // pass has.
        self.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_texel),
                rows_per_image: Some(height),
            },
            size,
        );
        texture.create_view(&Default::default())
    }

    /// Render one window at one clock and read the framebuffer back as RGBA
    /// bytes, row-major, `WINDOW_COLS x WINDOW_ROWS`.
    fn render(&self, grid: &CellGrid, clock: f32) -> Vec<u8> {
        let (w, h) = (grid.cols() as u32, grid.rows() as u32);
        // The TARGET is fine texels; the id texture stays one texel per cell.
        // That asymmetry IS the feature under test.
        let g = TEX_GRAIN as u32;
        let (fw, fh) = (w * g, h * g);

        let mut ids = Vec::with_capacity(grid.material.len() * 2);
        for id in &grid.material {
            ids.extend_from_slice(&id.to_le_bytes());
        }
        let id_view = self.upload("cell_ids", &ids, w, h, wgpu::TextureFormat::R16Uint, 2);

        let mut params = build_params();
        params.clock = Vec4::new(clock, 0.0, 0.0, 0.0);
        params.origin = IVec2::new(grid.origin_cell_x(), grid.origin_cell_y());
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: CellShadeParams::min_size().get(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoded = UniformBuffer::new(Vec::<u8>::new());
        encoded.write(&params).expect("params encode");
        self.queue.write_buffer(&uniform, 0, &encoded.into_inner());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&id_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cell framebuffer"),
            size: wgpu::Extent3d {
                width: fw,
                height: fh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());

        // A buffer copy DOES need 256-byte rows, so the readback is padded and
        // unpadded again below.
        let padded = fw * 4 + (256 - (fw * 4) % 256) % 256;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded * fh),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cells"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Clear to something that is NOT a legal cell colour, so
                        // an unwritten fragment cannot be mistaken for air.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(fh),
                },
            },
            wgpu::Extent3d {
                width: fw,
                height: fh,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll the readback");

        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((fw * fh * 4) as usize);
        for row in 0..fh {
            let at = (row * padded) as usize;
            out.extend_from_slice(&mapped[at..at + (fw * 4) as usize]);
        }
        drop(mapped);
        readback.unmap();
        out
    }
}

/// The shade table as the shader sees it, straight out of the shipping upload
/// path — so a bug in `new_shade_texture` fails here too.
fn shade_bytes() -> Vec<u8> {
    new_shade_texture(&CellShades::new())
        .data
        .expect("the shade texture carries its data")
}

// --- Comparison --------------------------------------------------------------

/// One cell that differs, with everything needed to go and look at it.
#[derive(Debug)]
struct Diff {
    x: i32,
    y: i32,
    id: u16,
    cpu: [u8; 4],
    gpu: [u8; 4],
}

impl Diff {
    fn worst(&self) -> i32 {
        (0..4)
            .map(|c| (i32::from(self.cpu[c]) - i32::from(self.gpu[c])).abs())
            .max()
            .unwrap_or(0)
    }
}

/// Walk both FINE buffers and split the disagreements by whether the material
/// animates. Returns `(static_diffs, animated_diffs, animated_texels)`. The
/// buffers are `TEX_GRAIN` texels per cell per axis; the material — and with
/// it the static/animated split — is derived from the CELL under each texel.
fn compare(grid: &CellGrid, cpu: &[u32], gpu: &[u8]) -> (Vec<Diff>, Vec<Diff>, usize) {
    let w = grid.cols();
    let g = TEX_GRAIN;
    let fw = w * g;
    let mut stat = Vec::new();
    let mut anim = Vec::new();
    let mut animated_cells = 0usize;
    for (i, &word) in cpu.iter().enumerate() {
        let (fx, fy) = (i as i32 % fw, i as i32 / fw);
        let (cx, cy) = (fx / g, fy / g);
        let id = grid.material[(cy * w + cx) as usize];
        let is_animated = animates(id);
        animated_cells += usize::from(is_animated);
        // `cells::pack` puts the red channel in whichever byte this target calls
        // lowest; `Rgba8Unorm` wants r, g, b, a in that order.
        let cpu_px = if cfg!(target_endian = "little") {
            word.to_le_bytes()
        } else {
            word.to_be_bytes()
        };
        let gpu_px: [u8; 4] = gpu[i * 4..i * 4 + 4].try_into().unwrap();
        if cpu_px == gpu_px {
            continue;
        }
        let diff = Diff {
            x: fx,
            y: fy,
            id,
            cpu: cpu_px,
            gpu: gpu_px,
        };
        if is_animated { &mut anim } else { &mut stat }.push(diff);
    }
    (stat, anim, animated_cells)
}

/// Every window, painted by the CPU and by the GPU at one clock.
fn sweep(harness: &Harness, clock: f32) -> Vec<(&'static str, CellGrid, Vec<u32>, Vec<u8>)> {
    windows()
        .iter()
        .map(|&(name, cx, cy)| {
            let grid = window_grid(cx, cy);
            let mut shades = CellShades::new();
            // The CPU oracle is put on the same clock. `update_shimmer` rewrites
            // only the materials that declare shimmer, which is exactly the set
            // the shader recomputes — so the static half of the table stays at
            // rest on both sides.
            shades.update_shimmer(f64::from(clock));
            let mut cpu = vec![0u32; (WINDOW_COLS * TEX_GRAIN * WINDOW_ROWS * TEX_GRAIN) as usize];
            paint_cells_fine(
                &mut cpu,
                grid.cols(),
                grid.rows(),
                &grid,
                grid.origin_cell_x(),
                grid.origin_cell_y(),
                &shades,
            );
            let gpu = harness.render(&grid, clock);
            (name, grid, cpu, gpu)
        })
        .collect()
}

/// Opacity is CELL geometry, and no grain may change that.
///
/// The four texels of every cell must agree in alpha: the air test lives on
/// the cell, so a cell is wholly transparent or wholly opaque, and the fine
/// pattern can never punch sub-cell holes in the world. This is the GPU-side
/// restatement of `cells_golden`'s `cover` mask — the harness has no cover
/// hash, but it can assert the property the hash freezes.
#[test]
fn every_cells_texels_agree_in_alpha() {
    let Some(harness) = harness_or_skip("sub-cell alpha") else {
        return;
    };
    let g = TEX_GRAIN;
    for (name, grid, _, gpu) in sweep(&harness, 0.0) {
        let (w, h) = (grid.cols(), grid.rows());
        let fw = w * g;
        for cy in 0..h {
            for cx in 0..w {
                let a0 = gpu[(((cy * g) * fw + cx * g) * 4 + 3) as usize];
                for sy in 0..g {
                    for sx in 0..g {
                        let at = (((cy * g + sy) * fw + cx * g + sx) * 4 + 3) as usize;
                        assert_eq!(
                            gpu[at], a0,
                            "{name}: cell ({cx},{cy}) sub ({sx},{sy}) alpha                              differs from its cell — the grain punched a                              sub-cell hole in the world"
                        );
                    }
                }
            }
        }
    }
    println!("every cell's texels agree in alpha at grain {g}");
}

/// `None` and a printed note when there is no GPU. See [`Harness::new`].
fn harness_or_skip(what: &str) -> Option<Harness> {
    match Harness::new() {
        Some(h) => Some(h),
        None => {
            println!("SKIPPED {what}: no wgpu adapter on this machine");
            None
        }
    }
}

// --- The gate ----------------------------------------------------------------

/// THE test. Every cell of a material that does not animate must be the exact
/// byte `paint_cells` packed.
///
/// This is the whole structural claim in one assertion: the three-tap
/// `depth_above`, the two horizontal taps, both coprime tile lookups at their own
/// periods, the `(id, edge, pattern)` shade index, the solid-outside-the-window
/// rule, and transparent air. Any of them wrong and this fails with the cell
/// coordinate and both colours.
#[test]
fn at_rest_materials_are_byte_identical_to_the_cpu_blit() {
    let Some(harness) = harness_or_skip("shader/CPU parity") else {
        return;
    };

    let mut checked = 0usize;
    let mut problems = Vec::new();
    for &clock in &CLOCKS {
        for (name, grid, cpu, gpu) in sweep(&harness, clock) {
            let (stat, _, animated) = compare(&grid, &cpu, &gpu);
            checked += cpu.len() - animated;
            if !stat.is_empty() {
                let worst = stat.iter().max_by_key(|d| d.worst()).unwrap();
                problems.push(format!(
                    "{name} @ t={clock}: {} of {} static cells differ; worst is \
                     cell ({}, {}) material {} — cpu {:?}, gpu {:?}",
                    stat.len(),
                    cpu.len() - animated,
                    worst.x,
                    worst.y,
                    worst.id,
                    worst.cpu,
                    worst.gpu,
                ));
            }
        }
    }

    // Re-keyed for TEX_GRAIN = 1 and the scaled depths window: 1 141 248 static
    // texels measured, against 1.2M at grain 2. Both causes are intended — a
    // texel is a whole cell again, and the depths window now genuinely holds
    // lava, whose cells are ANIMATED and so counted by the shimmer test rather
    // than this one. The floor exists to catch the sweep silently shrinking, so
    // it sits just under the measurement rather than at a round number.
    assert!(
        checked > 1_100_000,
        "the sweep shrank: only {checked} static texels compared"
    );
    assert!(
        problems.is_empty(),
        "the shader disagrees with `paint_cells` on cells it should be reading \
         the same table for:\n  {}",
        problems.join("\n  ")
    );
    println!("{checked} static cells: shader == paint_cells, byte for byte");
}

/// The animated materials, quantified rather than asserted equal.
///
/// `update_shimmer` evaluates its wave in f64 and the shader in f32, so the two
/// are not required to agree to the last bit and it would be dishonest to assert
/// they do. What they actually do, measured:
///
/// | clock | lava cells differing | max channel delta |
/// |---|---|---|
/// | 0 s .. 1800 s | **0 of 38 183** | 0 |
/// | 3600 s | 1 of 38 183 | 1 |
/// | 36000 s | 1 455 of 38 183 (3.8%) | 1 |
///
/// EXACT FOR THE FIRST HALF HOUR, then one least-significant bit on a few per
/// cent of lava. The cause is not the sine and not the pack: it is
/// `i32(a) & 2047` on a phase that grows at ~685 index units per second. An f32
/// holds that index exactly up to 2^24 — 6.8 hours' worth of raw magnitude, and
/// in practice about an hour once `t * 2.1` and the phase offset are in — and
/// after that consecutive indices stop being representable, so the truncation can
/// pick the neighbouring table slot. One slot is 0.003 rad, which moves the lift
/// by at most ~0.04 of a colour unit: enough to cross an integer boundary, never
/// enough to be visible.
///
/// The bound asserted below is therefore ONE LSB, not zero. A larger delta is a
/// bug in the translation, not a rounding order.
#[test]
fn the_shimmer_matches_the_cpu_blit_to_a_measured_bound() {
    let Some(harness) = harness_or_skip("shimmer parity") else {
        return;
    };

    let mut report: BTreeMap<String, (usize, usize, i32)> = BTreeMap::new();
    let mut worst_overall = 0;
    let mut examples: Vec<String> = Vec::new();
    let mut animated_total = 0usize;

    for &clock in &CLOCKS {
        for (name, grid, cpu, gpu) in sweep(&harness, clock) {
            let (_, anim, animated) = compare(&grid, &cpu, &gpu);
            animated_total += animated;
            let worst = anim.iter().map(Diff::worst).max().unwrap_or(0);
            worst_overall = worst_overall.max(worst);
            report.insert(format!("{name} t={clock}"), (anim.len(), animated, worst));
            if let Some(d) = anim.iter().max_by_key(|d| d.worst())
                && examples.len() < 4
            {
                examples.push(format!(
                    "{name} t={clock}: cell ({}, {}) material {} cpu {:?} gpu {:?}",
                    d.x, d.y, d.id, d.cpu, d.gpu
                ));
            }
        }
    }

    // The depths window has to be MOSTLY LAVA or the branch is not being tested
    // at all and the numbers below mean nothing.
    //
    // 36 344 animated texels measured at WorldScale::LIVE, against >100 000 at
    // the 1x world. Two intended causes and no unintended one: a texel is a whole
    // cell again at TEX_GRAIN 1, and the same relative depth in a 4x world holds
    // a different amount of lava because the liquid table scaled with everything
    // else. What matters is that it is tens of thousands rather than the 5 960
    // this read when the window row was left unscaled and slid out of the
    // underworld entirely — that is the failure this floor exists to catch, and
    // it caught it.
    assert!(
        animated_total > 30_000,
        "the shimmer branch is barely exercised: {animated_total} animated cells"
    );

    let summary: Vec<String> = report
        .iter()
        .map(|(k, (differing, total, worst))| {
            let pct = 100.0 * *differing as f64 / (*total).max(1) as f64;
            format!("{k}: {differing}/{total} cells ({pct:.2}%), max channel delta {worst}")
        })
        .collect();
    println!(
        "shimmer: f64 CPU vs f32 GPU over {animated_total} animated cells\n  {}\n  {}",
        summary.join("\n  "),
        examples.join("\n  ")
    );

    assert!(
        worst_overall <= 1,
        "the shimmer differs by more than one least-significant bit — that is a \
         bug, not a rounding order:\n  {}\n  {}",
        summary.join("\n  "),
        examples.join("\n  ")
    );

    // The regime that matters: a session under half an hour must be EXACT. If
    // this ever starts failing, something has gone wrong with the arithmetic
    // rather than with f32's reach.
    for (key, (differing, _, _)) in &report {
        let early = CLOCKS
            .iter()
            .filter(|&&t| t <= 1800.0)
            .any(|t| key.ends_with(&format!("t={t}")));
        assert!(
            !early || *differing == 0,
            "{key}: {differing} cells differ inside the range where f32 still \
             holds the sine index exactly"
        );
    }
}

// --- The constants -----------------------------------------------------------

/// Every scalar `cells.wgsl` states as a literal must be the one `cells.rs`
/// actually uses.
///
/// The shader cannot import a Rust constant, so the numbers are written twice.
/// This is what makes that safe: it parses the shader source for each `const`
/// and compares it, as an `f32`, against the re-exported value. A tuning change
/// on the CPU side that forgets the shader fails HERE, where the message says
/// which number — rather than in the pixel diff, where it says "everything".
#[test]
fn the_shader_states_the_same_constants_as_the_blit() {
    let mut found: BTreeMap<&str, f32> = BTreeMap::new();
    for line in SHADING_WGSL.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("const ") else {
            continue;
        };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let name = name.split(':').next().unwrap().trim();
        let value = value.trim().trim_end_matches(';');
        found.insert(
            Box::leak(name.to_owned().into_boxed_str()),
            value.parse::<f32>().unwrap_or_else(|_| {
                panic!("cells.wgsl `const {name}` is not a plain float literal: {value}")
            }),
        );
    }

    let expect: [(&str, f64); 9] = [
        ("GRAIN", f64::from(TEX_GRAIN)),
        ("PA", f64::from(TEX_A_PERIOD)),
        ("PB", f64::from(TEX_B_PERIOD)),
        ("PAT_MID", SHADE_PAT_MID),
        ("PAT_SIGMA", SHADE_PAT_SIGMA),
        ("EDGE_SCALE", SHADE_EDGE_SCALE),
        ("SIN_SCALE", SHIMMER_SIN_SCALE),
        ("SIN_SIZE", SHIMMER_SIN_SIZE as f64),
        ("SIN_MASK", (SHIMMER_SIN_SIZE - 1) as f64),
    ];
    for (name, want) in expect {
        let got = found
            .get(name)
            .unwrap_or_else(|| panic!("cells.wgsl no longer declares `{name}`"));
        assert_eq!(
            *got, want as f32,
            "cells.wgsl says {name} = {got}, cells.rs says {want}"
        );
    }

    // The eight edge gains are a `var<private>` array, not consts, so they are
    // read out of the initialiser.
    let start = SHADING_WGSL
        .find("var<private> EDGE_GAIN")
        .expect("cells.wgsl no longer declares EDGE_GAIN");
    let body = &SHADING_WGSL[start..];
    let body = &body[body.find('(').unwrap() + 1..body.find(");").unwrap()];
    let gains: Vec<f32> = body
        .lines()
        .filter_map(|l| {
            l.split("//")
                .next()
                .unwrap()
                .trim()
                .trim_end_matches(',')
                .parse()
                .ok()
        })
        .collect();
    assert_eq!(
        gains,
        SHADE_EDGE_GAIN.map(|g| g as f32).to_vec(),
        "cells.wgsl EDGE_GAIN has drifted from cells.rs"
    );
}

/// The uniform the shader declares must be the one Rust uploads.
///
/// `encase` decides the layout on the Rust side and naga on the shader side; if
/// the two disagree on a single offset, every material reads its neighbour's
/// colour and the pixel diff would be a wall of noise with no cause in it.
#[test]
fn the_uniform_layout_matches_on_both_sides() {
    assert_eq!(MATERIAL_SLOTS, 64, "cells.wgsl writes 64 as a literal");
    // 64 vec4s, twice, then a vec4 and an ivec2 padded to a 16-byte stride.
    assert_eq!(
        CellShadeParams::min_size().get(),
        (MATERIAL_SLOTS as u64 * 16) * 2 + 16 + 16
    );
    let mut buf = UniformBuffer::new(Vec::<u8>::new());
    buf.write(&build_params()).expect("params encode");
    assert_eq!(
        buf.into_inner().len() as u64,
        CellShadeParams::min_size().get()
    );
}
