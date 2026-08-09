//! The rendered bank, in a form Bevy will play.
//!
//! # Why WAV, of all things
//!
//! `AudioSource` holds bytes and hands them to rodio, so it wants a CONTAINER
//! rather than samples — there is no "here is a `Vec<f32>` at 44.1 kHz" door.
//! Of the formats rodio decodes, WAV is the only one that can be written here
//! without a dependency: a 44-byte header and the samples as little-endian
//! `i16`. Encoding OGG by hand is not a thing anyone should do.
//!
//! So the pipeline is parameters -> `f32` -> WAV bytes -> `AudioSource`, and the
//! middle step is the one worth keeping honest: `synth` produces the samples and
//! is tested on them, and this file only wraps.
//!
//! It costs the `wav` feature on `bevy`, which is not in its defaults. That is
//! declared in the workspace manifest with a comment pointing here.
//!
//! # Baked once
//!
//! Every sound is rendered and encoded at `PreStartup`, exactly as the sprite
//! atlases are, and for the same reason: a footstep must not have a synthesiser
//! on its path. The whole bank is a few hundred kilobytes.

use bevy::audio::AudioSource;
use bevy::prelude::*;

use yugen_data::sounds::SOUND_COUNT;

use super::synth::{SAMPLE_RATE, render_all};

/// Wrap mono `f32` samples as a 16-bit PCM WAV file.
///
/// The header is the canonical 44-byte one. `i16` rather than `f32` samples
/// because it is the encoding every decoder handles without a format extension,
/// and 16 bits is far more than these sounds resolve — they are square waves and
/// noise, not orchestral recordings.
fn wav(samples: &[f32]) -> Vec<u8> {
    let bytes = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + bytes) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // format: PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels: mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(bytes as u32).to_le_bytes());
    for s in samples {
        // `32767.0` and a clamp: `i16::MIN` has no positive counterpart, so
        // scaling by 32768 lets a sample of exactly 1.0 wrap to the most
        // negative value there is — a full-scale click in place of a peak.
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    out
}

/// Every sound, decoded-ready, indexed by code.
#[derive(Resource, Default)]
pub struct SoundBank {
    handles: Vec<Handle<AudioSource>>,
}

impl SoundBank {
    /// The asset for a sound code, or `None` for a code this build lacks.
    pub fn get(&self, code: u16) -> Option<Handle<AudioSource>> {
        self.handles.get(code as usize).cloned()
    }

    /// How many sounds are loaded. Zero before `PreStartup`.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Whether the bank is empty (Rust convention).
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

/// Render and encode the whole bank into assets.
pub fn bake(mut bank: ResMut<SoundBank>, mut assets: ResMut<Assets<AudioSource>>) {
    bank.handles = render_all()
        .iter()
        .map(|pcm| {
            assets.add(AudioSource {
                bytes: wav(pcm).into(),
            })
        })
        .collect();
    debug_assert_eq!(bank.handles.len(), SOUND_COUNT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_says_what_the_data_is() {
        // 100 samples is 200 bytes of data and 244 of file. Checking the two
        // length fields rather than eyeballing the header is the point: a WAV
        // whose sizes disagree with its payload is the one failure a decoder
        // reports as "unrecognised format" with nothing else to go on.
        let bytes = wav(&vec![0.0; 100]);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + 200);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            (bytes.len() - 8) as u32,
            "the RIFF size must be the file minus its own 8-byte prefix"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
            200,
            "the data size must be the payload"
        );
    }

    #[test]
    fn a_full_scale_sample_does_not_wrap_to_the_bottom() {
        // Scaling by 32768 would send 1.0 to -32768, turning the loudest sample
        // in a sound into the most negative one there is. It is one character
        // in the source and a click at every peak.
        let bytes = wav(&[1.0, -1.0]);
        let s = |i: usize| i16::from_le_bytes(bytes[44 + i * 2..46 + i * 2].try_into().unwrap());
        assert_eq!(s(0), 32767);
        assert_eq!(s(1), -32767);
    }

    #[test]
    fn an_empty_sound_is_still_a_valid_file() {
        let bytes = wav(&[]);
        assert_eq!(bytes.len(), 44);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 0);
    }
}
