//! The light blur shader must produce what `light::blur_one` produces.
//!
//! # What this is, and why it exists for a shader nothing renders
//!
//! The blur was 55% of the whole per-frame CPU render cost and moving it to the
//! GPU was the largest remaining item in `docs/PERF.md`. It was also the
//! riskiest, for one reason that had nothing to do with the arithmetic:
//! `shader_matches_cpu.rs` covers `cells.wgsl` and nothing covered the light
//! stack, so a GPU blur would have been a pass with no oracle at all — verified
//! by looking at a cave and deciding it seemed about right.
//!
//! So this landed first, and it is what let the move be judged rather than
//! guessed. **The move lost**: wired in, `lightblur.wgsl` cost 237 µs of
//! whole-frame time to remove CPU work worth 1 µs of it, and the game still runs
//! `blur_one`. See `PERF.md` §8.6, and `lightblur.wgsl`'s own header.
//!
//! That leaves this file testing a shader the frame does not draw, which is the
//! position `scan_emitters` has held for two milestones and is defensible for
//! the same reason: the implementation is correct, the day it becomes cheap is a
//! render-graph node away, and a `.wgsl` nobody compiles is a `.wgsl` that rots.
//! It runs in 0.6 s.
//!
//! # The three claims
//!
//! - [`the_shader_reproduces_the_cpu_blur`] — same bytes in, same field out, to
//!   a measured bound. This is the fidelity claim and it covers every texel,
//!   including the border, which is where the interesting failure lives.
//! - [`a_fused_triangle_would_be_wrong_at_the_border`] — the shader's nested
//!   loops are load-bearing rather than naive, and this is the control that
//!   proves it. Nine taps of `[1,2,3,4,5,4,3,2,1] / 25` are the SAME kernel in
//!   the interior and a different one at the edges, because the CPU clamps
//!   between its two boxes. Without this test the cheaper shader would look
//!   correct in every screenshot and be wrong on the four borders of every
//!   frame.
//! - [`quantising_the_field_before_the_blur_costs_under_a_byte`] — the one real
//!   behavioural change. The CPU blurred `f32` and quantised once, at the end;
//!   the GPU is handed an 8-bit texture and quantises at the START. This bounds
//!   what that costs in the composited output byte, on the CPU, with no GPU
//!   involved.
//!
//! If no adapter is available the GPU tests SKIP rather than pass, for the
//! reason `shader_matches_cpu` gives: a green tick from a machine that never ran
//! the shader is the worst of the three outcomes.

use std::borrow::Cow;

use bevy::tasks::block_on;
use yugen_core::config::{CHUNK_CELLS, View, WINDOW_COLS, WINDOW_ROWS};
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::worldgen::ChunkGen;
use yugen_render::light::{
    EmitterScan, LightFrame, LightGrid, blur_one, blur_window, grid_size, scan_emitters,
};

/// The shipping blur, included as SOURCE. If this file and the game ever compile
/// different text the harness is worthless.
const BLUR_WGSL: &str = include_str!("../src/lightblur.wgsl");

/// The seed the other parity fixtures are dumped at, reused so the terrain under
/// the light field is terrain those suites have already frozen.
const SEED: u32 = 2334;

/// Top-left chunk of the window the field is solved over.
///
/// The `surface` window of `shader_matches_cpu`: it straddles the ground line,
/// so the skylight flood produces the full range — open sky at 1.0, a hard
/// vertical falloff behind the first solid row, and unlit cave underneath. A
/// window of pure rock would blur a constant and prove nothing.
const WINDOW: (i32, i32) = (-5, -1);

/// The view the field is solved over.
///
/// A real 2560x1440 display and not `View::default`, whose grid is 102x52.
/// The bigger grid is 162x92 — the size `docs/PERF.md` measures the blur at —
/// and three times the texels is three times the chances for a worst case to
/// exist at all. It costs a millisecond.
fn view() -> View {
    View::for_screen(2560, 1440)
}

/// The intermediate and final targets' format, matching the game's.
///
/// `Rgba16Float` and not `Rgba8Unorm`: the pass between the two blurs is not
/// something anybody looks at, and rounding it to bytes would put a second
/// quantisation in a chain whose first one this file already has to account for.
/// Half floats carry the 0..1 field to about one part in 2000, so the
/// intermediate contributes nothing measurable and the harness can attribute the
/// whole delta to arithmetic.
const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// --- Building the field ------------------------------------------------------

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

/// The four light planes as the GPU will be handed them: `[r, g, b, light]` per
/// texel, unblurred, over a real solved window.
///
/// Deliberately NOT `LightGrid::solve`: this is every pass `solve` runs EXCEPT
/// the blur, which is the input the shader is supposed to consume. Assembling it
/// from the same three public passes rather than from a synthetic pattern is
/// what makes the test's value distribution the game's — a hard sky/rock step,
/// coloured splats on top of it, and long flat runs either side.
fn unblurred_field(view: View) -> (i32, i32, Vec<f32>) {
    let cells = window_grid(WINDOW.0, WINDOW.1);
    let (lw, lh) = grid_size(view);

    // Somewhere inside the window, straddling its ground line.
    let frame = LightFrame {
        ox: cells.origin_cell_x() + 20,
        oy: cells.origin_cell_y() + 40,
        day: 1.0,
        t: 0.0,
    };

    let mut light = LightGrid::new(view, SEED);
    let mut census = EmitterScan::default();
    scan_emitters(
        &cells,
        frame.ox - 1,
        frame.oy - 1,
        lw + 2,
        lh + 2,
        &mut census,
    );
    light.compute_skylight(&cells, frame);
    light.add_emissive(&cells, frame);
    light.add_census_emitters(&cells, census.x(), census.y(), frame);

    let mut out = vec![0.0f32; (lw * lh * 4) as usize];
    for ly in 0..lh {
        for lx in 0..lw {
            let at = ((ly * lw + lx) * 4) as usize;
            let rgb = light.colour_at(lx, ly);
            out[at..at + 3].copy_from_slice(&rgb);
            out[at + 3] = light.light_at(lx, ly);
        }
    }
    out.iter().for_each(|v| {
        assert!(
            (0.0..=1.0).contains(v),
            "the solve produced {v}, and the blur's whole edge-clamp argument \
             assumes the field is a fraction"
        );
    });
    (lw, lh, out)
}

/// Round-trip a field through the 8-bit texture the game uploads.
///
/// Both sides of the fidelity test get THIS, not the `f32` field, so the only
/// difference left between them is arithmetic. What the round trip itself costs
/// is a separate question with its own test.
fn quantised(field: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let bytes: Vec<u8> = field.iter().map(|v| unit_byte(*v)).collect();
    let back = bytes.iter().map(|b| f32::from(*b) / 255.0).collect();
    (bytes, back)
}

/// Float to byte, rounding rather than truncating.
///
/// `light.rs`'s own `unit_byte` truncates, and it is right to: those bytes are a
/// final answer and the error lands on one output texel. These bytes would be
/// the INPUT to a blur, so truncation's half-step downward bias would be smeared
/// over every texel the kernel reaches. Any real upload path for this shader
/// would round, so the harness rounds.
fn unit_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// The CPU oracle: `blur_one` over each of the four interleaved channels.
fn cpu_blur(field: &[f32], lw: i32, lh: i32) -> Vec<f32> {
    let n = (lw * lh) as usize;
    let mut out = vec![0.0f32; n * 4];
    let mut plane = vec![0.0f32; n];
    let mut scratch = vec![0.0f32; n];
    let mut acc = vec![0.0f32; lw as usize];
    for c in 0..4 {
        for (i, p) in plane.iter_mut().enumerate() {
            *p = field[i * 4 + c];
        }
        blur_one(&mut plane, &mut scratch, &mut acc, lw, lh);
        for (i, p) in plane.iter().enumerate() {
            out[i * 4 + c] = *p;
        }
    }
    out
}

// --- The headless device -----------------------------------------------------

/// The vertex stage and the bindings the game's `Material2d` supplies. The
/// FRAGMENT stage is the shipping file, unedited.
const HARNESS_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    // One oversized triangle. `uv` is the clip position mapped to 0..1 with y
    // flipped, which is what a quad's uv means in Bevy and what makes texel
    // (0, 0) the TOP-left of both the source and the target.
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return out;
}
"#;

/// Strip the shipping file's preprocessor directives and bind-group macro.
///
/// Bevy substitutes `#{MATERIAL_BIND_GROUP}` and resolves the `#import`; wgpu
/// does neither. Everything else — every line of arithmetic — is the game's.
fn shader_source() -> String {
    let directives: Vec<&str> = BLUR_WGSL
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .collect();
    assert_eq!(
        directives,
        vec!["#import bevy_sprite::mesh2d_vertex_output::VertexOutput"],
        "lightblur.wgsl grew a preprocessor directive; the harness would be \
         compiling different source from the game"
    );
    let body: String = BLUR_WGSL
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("#{MATERIAL_BIND_GROUP}", "0");
    format!("{HARNESS_WGSL}\n{body}")
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Harness {
    /// Bring up a headless adapter, or `None` if this machine has no GPU wgpu
    /// will talk to.
    fn new() -> Option<Harness> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yugen light blur parity"),
            ..Default::default()
        }))
        .ok()?;
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e}")));

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lightblur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light blur bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // `filterable: false` — every read is a `textureLoad`.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("light blur"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(TARGET.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Some(Harness {
            device,
            queue,
            pipeline,
            layout,
        })
    }

    /// Both passes, exactly as the two cameras run them: rows then columns.
    fn blur(&self, bytes: &[u8], lw: i32, lh: i32) -> Vec<f32> {
        let (w, h) = (lw as u32, lh as u32);
        let (back, fwd) = blur_window();

        let src = self.upload(bytes, w, h);
        let rows = self.pass(&src, w, h, [back as f32, fwd as f32, 1.0, 0.0]);
        let cols = self.pass(
            &rows.create_view(&Default::default()),
            w,
            h,
            [back as f32, fwd as f32, 0.0, 1.0],
        );
        self.readback(&cols, w, h)
    }

    /// The 8-bit source texture the game uploads from the CPU.
    fn upload(&self, bytes: &[u8], w: u32, h: u32) -> wgpu::TextureView {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light raw"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        texture.create_view(&Default::default())
    }

    /// One axis, into a fresh `Rgba16Float` target.
    fn pass(&self, src: &wgpu::TextureView, w: u32, h: u32, params: [f32; 4]) -> wgpu::Texture {
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let encoded: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&uniform, 0, &encoded);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light blurred"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("light blur pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Negative, so a texel the quad failed to cover reads as
                        // something no blur of a 0..1 field can produce.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: -1.0,
                            g: -1.0,
                            b: -1.0,
                            a: -1.0,
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
        self.queue.submit([encoder.finish()]);
        target
    }

    /// Pull a `Rgba16Float` target back as interleaved `f32`.
    fn readback(&self, target: &wgpu::Texture, w: u32, h: u32) -> Vec<f32> {
        let row = w * 8;
        let padded = row + (256 - row % 256) % 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll the readback");

        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let at = (y * padded) as usize;
            for i in 0..(w * 4) as usize {
                let bits = u16::from_le_bytes([mapped[at + i * 2], mapped[at + i * 2 + 1]]);
                out.push(half_to_f32(bits));
            }
        }
        out
    }
}

/// IEEE binary16 to `f32`. No infinities or NaNs can reach here — the shader's
/// last statement clamps to 0..1 — so the two exponent extremes are handled for
/// completeness rather than because a field could hit them.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    let mag = match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    };
    sign * mag
}

// --- The tests ---------------------------------------------------------------

/// Largest absolute difference between the shader and `blur_one`, given the
/// same bytes.
///
/// # Where this number comes from
///
/// **Measured at 9.1e-4** over a 162x92 grid, and printed on every run. Three
/// things separate the two sides and none of them is the kernel: the shader sums
/// its taps fresh where the CPU carries a running window, so the two accumulate
/// `f32` rounding differently; the intermediate is a half float; and the final
/// target is a half float too. A half carries eleven significand bits, so a
/// value near 1 is quantised at about 4.9e-4, and two passes of that is most of
/// what is measured — the floor here is the format, not the port.
///
/// The bound is 2.2x that, and still under a QUARTER of one 8-bit step (3.9e-3),
/// which is the difference that would actually reach a pixel.
///
/// # This bound has been seen to fail
///
/// Not by injection into a scratch build that then got thrown away:
/// [`a_fused_triangle_would_be_wrong_at_the_border`] is the injection, and it is
/// in the tree permanently. It runs the cheapest plausible wrong kernel — the
/// one anybody moving this pass to a shader would write first — through the same
/// comparison and measures **10.3 8-bit steps** of error on the border, four
/// orders of magnitude above this bound. That is the gap this number sits in.
const MAX_DELTA: f32 = 2.0e-3;

#[test]
fn the_shader_reproduces_the_cpu_blur() {
    let Some(harness) = Harness::new() else {
        println!("SKIPPED light blur parity: no wgpu adapter on this machine");
        return;
    };
    let view = view();
    let (lw, lh, field) = unblurred_field(view);
    let (bytes, dequantised) = quantised(&field);

    let cpu = cpu_blur(&dequantised, lw, lh);
    let gpu = harness.blur(&bytes, lw, lh);
    assert_eq!(cpu.len(), gpu.len());

    // Reported separately because they are different claims. The interior is the
    // kernel; the border is the edge rule, and the border is the half a fused
    // triangle gets wrong — see the control below.
    let (back, fwd) = blur_window();
    let margin = (back + fwd) as i32;
    let mut worst = (0.0f32, 0usize);
    let mut worst_border = 0.0f32;
    for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
        let d = (a - b).abs();
        if d > worst.0 {
            worst = (d, i);
        }
        let texel = i / 4;
        let (x, y) = ((texel as i32) % lw, (texel as i32) / lw);
        if x < margin || y < margin || x >= lw - margin || y >= lh - margin {
            worst_border = worst_border.max(d);
        }
    }
    let texel = worst.1 / 4;
    println!(
        "light blur, {lw}x{lh}, window ({back}, {fwd}):\n  \
         worst delta {:.3e} at ({}, {}) channel {}\n  \
         worst delta on the border {worst_border:.3e}\n  \
         one 8-bit step is {:.3e}",
        worst.0,
        (texel as i32) % lw,
        (texel as i32) / lw,
        worst.1 % 4,
        1.0 / 255.0,
    );

    assert!(
        worst.0 <= MAX_DELTA,
        "the light blur shader has drifted from `blur_one`: worst delta {:.3e} \
         against a bound of {MAX_DELTA:.3e}. That is arithmetic, not noise — a \
         half float and a re-associated sum together account for about 1e-3, so \
         anything above this is the kernel changing.",
        worst.0
    );
}

#[test]
fn a_fused_triangle_would_be_wrong_at_the_border() {
    let view = view();
    let (lw, lh, field) = unblurred_field(view);
    let (_, mut dequantised) = quantised(&field);

    // A bright one-texel ring on the outermost edge, stamped AFTER the real
    // field is built. The border disagreement this control exists to show only
    // appears when bright content sits inside the blur margin of the edge, and
    // this test used to get that for free from the worldgen window -- canopy
    // skylight made the border sharp. Then the canopy stopped being solid, the
    // skylight flooded through it, the border went flat, and the control's gap
    // collapsed from 10.3 8-bit steps to 1.6: the fault injection almost
    // stopped injecting, because its fault was borrowed from CONTENT. A control
    // must construct what it demonstrates. This ring is that construction, and
    // it is immune to every future change in what a block is made of.
    for y in 0..lh {
        for x in 0..lw {
            if x == 0 || y == 0 || x == lw - 1 || y == lh - 1 {
                let at = ((y * lw + x) * 4) as usize;
                dequantised[at..at + 4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
            }
        }
    }

    let reference = cpu_blur(&dequantised, lw, lh);
    let fused = fused_triangle(&dequantised, lw, lh);

    let (back, fwd) = blur_window();
    let margin = (back + fwd) as i32;
    let (mut interior, mut border) = (0.0f32, 0.0f32);
    for (i, (a, b)) in reference.iter().zip(&fused).enumerate() {
        let d = (a - b).abs();
        let texel = (i / 4) as i32;
        let (x, y) = (texel % lw, texel / lw);
        if x < margin || y < margin || x >= lw - margin || y >= lh - margin {
            border = border.max(d);
        } else {
            interior = interior.max(d);
        }
    }
    println!(
        "fused triangle vs the two clamped boxes: interior {interior:.3e}, \
         border {border:.3e} ({} 8-bit steps)",
        border * 255.0
    );

    assert!(
        interior <= MAX_DELTA,
        "the fused triangle disagrees with `blur_one` in the INTERIOR by \
         {interior:.3e}. The whole premise of this test is that the two kernels \
         are identical away from the edges; if that is false the triangle \
         weights are wrong, not the edge rule."
    );
    assert!(
        border > 4.0 / 255.0,
        "a fused triangle now agrees with `blur_one` at the border to within \
         {border:.3e}, so `lightblur.wgsl`'s nested loops are no longer buying \
         anything and its comment is a lie. Either the grid stopped clamping at \
         its edges or this control has stopped constructing one."
    );
}

/// The nine-tap kernel `lightblur.wgsl` deliberately does NOT use.
///
/// A box convolved with a box, evaluated in one pass with edge replication on
/// the ORIGINAL field. Identical to `blur_one` in the interior and wrong within
/// `back + fwd` of any edge, which is what the control above asserts.
fn fused_triangle(field: &[f32], lw: i32, lh: i32) -> Vec<f32> {
    let (back, fwd) = blur_window();
    let w = (back + fwd + 1) as i32;
    let inv = 1.0 / (w * w) as f32;
    let weights: Vec<f32> = (-(w - 1)..=(w - 1))
        .map(|k| (w - k.abs()) as f32 * inv)
        .collect();

    let axis = |src: &[f32], dx: i32, dy: i32| -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for y in 0..lh {
            for x in 0..lw {
                let at = ((y * lw + x) * 4) as usize;
                for (j, weight) in weights.iter().enumerate() {
                    let k = j as i32 - (w - 1);
                    let sx = (x + dx * k).clamp(0, lw - 1);
                    let sy = (y + dy * k).clamp(0, lh - 1);
                    let from = ((sy * lw + sx) * 4) as usize;
                    for c in 0..4 {
                        out[at + c] += src[from + c] * weight;
                    }
                }
            }
        }
        out
    };
    axis(&axis(field, 1, 0), 0, 1)
}

/// What quantising the light field to 8 bits BEFORE the blur costs, in
/// composited output bytes.
///
/// # Why this is the honest way to ask
///
/// The CPU path blurred `f32` and rounded once, on the way into the texture. The
/// GPU path is handed a texture, so it rounds first and blurs the rounded field.
/// Comparing the two blurred FIELDS would understate it, because a light value
/// is not what reaches the frame: `bake_shadow`'s `mix(shadow, 1, l)` is what
/// reaches the frame, and it is that byte a player could see move.
///
/// Blurring is an average, so an input rounding of at most half a step shrinks
/// rather than compounds — but only if the errors are independent, and a smooth
/// field's are not. Hence a measurement.
const MAX_QUANTISATION_BYTES: f32 = 1.0;

#[test]
fn quantising_the_field_before_the_blur_costs_under_a_byte() {
    let view = view();
    let (lw, lh, field) = unblurred_field(view);
    let (_, dequantised) = quantised(&field);

    let exact = cpu_blur(&field, lw, lh);
    let rounded = cpu_blur(&dequantised, lw, lh);

    // The darkest shadow tint the game reaches, so the multiply below is the
    // steepest mapping any frame applies to a light value — the worst case for
    // an error in it. A tint nearer white would flatten the difference.
    let shadow = yugen_render::light::shadow_tint(1.0, 0.0);
    let steepest = 1.0 - shadow.iter().cloned().fold(1.0f32, f32::min);

    let mut worst = 0.0f32;
    let mut sum = 0.0f64;
    for (a, b) in exact.iter().zip(&rounded) {
        let d = (a - b).abs() * steepest * 255.0;
        worst = worst.max(d);
        sum += f64::from(d);
    }
    let mean = sum / exact.len() as f64;
    println!(
        "quantising before the blur, in output bytes: worst {worst:.3}, mean \
         {mean:.4} (steepest multiply {steepest:.3})"
    );

    assert!(
        worst <= MAX_QUANTISATION_BYTES,
        "rounding the light field to 8 bits before blurring it now moves a \
         composited byte by {worst:.3}, against a bound of \
         {MAX_QUANTISATION_BYTES}. The GPU blur reads an 8-bit texture, so this \
         is the price of the whole move; if it has grown past a byte the upload \
         format is the thing to change, not this bound."
    );
}
