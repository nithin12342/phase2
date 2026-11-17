//! WAV PCM parse + normalized-autocorrelation pitch contour (std-only).
//!
//! Replaces librosa.pyin on the hot path. The estimator is a McLeod-style
//! normalized square difference function (NSDF): cheaper than pyin's
//! probabilistic Viterbi, and compiled. Voiced frames only; silence is skipped.
//! Pitch values differ slightly from pyin by design (a different estimator).

/// Parse a RIFF/WAVE 16-bit PCM file. Returns (sample_rate, mono samples in
/// [-1, 1]). Stereo is averaged to mono. Any other format yields (0, empty).
pub fn parse_wav(bytes: &[u8]) -> (u32, Vec<f32>) {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return (0, Vec::new());
    }
    let mut pos = 12usize;
    let mut sr = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7],
        ]) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            break;
        }
        match id {
            b"fmt " if size >= 16 => {
                channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
                sr = u32::from_le_bytes([
                    bytes[body + 4], bytes[body + 5], bytes[body + 6], bytes[body + 7],
                ]);
                bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            }
            b"data" => data = &bytes[body..body + size],
            _ => {}
        }
        pos = body + size + (size & 1); // RIFF chunks are word-aligned
    }
    if sr == 0 || bits != 16 || channels == 0 || data.is_empty() {
        return (0, Vec::new());
    }
    let ch = channels as usize;
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n / ch);
    let mut i = 0usize;
    while i + ch <= n {
        let mut acc = 0i32;
        for c in 0..ch {
            let s = i16::from_le_bytes([data[(i + c) * 2], data[(i + c) * 2 + 1]]);
            acc += s as i32;
        }
        out.push((acc / ch as i32) as f32 / 32768.0);
        i += ch;
    }
    (sr, out)
}

/// Dot product of two equal-length slices. Eight independent accumulators let
/// LLVM pack the inner loop into one AVX2 (8-wide f32) multiply-add, and
/// chunks_exact proves the bounds so no per-element check survives. This is the
/// hot kernel: it runs once per lag per frame.
fn dot(x: &[f32], y: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut xi = x.chunks_exact(8);
    let mut yi = y.chunks_exact(8);
    for (xc, yc) in xi.by_ref().zip(yi.by_ref()) {
        for k in 0..8 {
            acc[k] += xc[k] * yc[k];
        }
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    for (&a, &b) in xi.remainder().iter().zip(yi.remainder()) {
        s += a * b;
    }
    s
}

/// Parabolic interpolation around the peak lag for sub-sample accuracy. Reads
/// the NSDF values already computed by the scan, so refinement is O(1).
fn refine(cache: &[f64], lag: usize, lo: usize, hi: usize) -> f64 {
    if lag <= lo || lag >= hi {
        return lag as f64;
    }
    let y0 = cache[lag - 1 - lo];
    let y1 = cache[lag - lo];
    let y2 = cache[lag + 1 - lo];
    let denom = y0 - 2.0 * y1 + y2;
    if denom == 0.0 {
        return lag as f64;
    }
    let delta = 0.5 * (y0 - y2) / denom;
    lag as f64 + delta.clamp(-1.0, 1.0)
}

/// Voiced F0 contour in Hz. 40 ms frames, 20 ms hop (50 frames/s). A frame is
/// voiced when its RMS clears the silence gate and its NSDF peak clears 0.4.
///
/// Two cuts make this ~8x faster than the naive form:
///   * Energy is a prefix sum, so the NSDF denominator is O(1) per lag instead
///     of a second pass over the frame.
///   * The autocorrelation numerator is the vectorized `dot`, and every NSDF
///     value is cached so refinement never recomputes a lag.
pub fn f0_contour(samples: &[f32], sr: u32) -> Vec<f64> {
    if sr == 0 || samples.is_empty() {
        return Vec::new();
    }
    let sr_f = sr as f64;
    let frame = sr as usize * 40 / 1000;
    let hop = sr as usize * 20 / 1000;
    let min_lag = (sr_f / 400.0) as usize; // fmax 400 Hz
    let max_lag = (sr_f / 60.0) as usize; // fmin 60 Hz
    if frame == 0 || hop == 0 || max_lag <= min_lag || frame <= max_lag {
        return Vec::new();
    }
    let nlag = max_lag - min_lag + 1;
    let mut cache = vec![0.0f64; nlag]; // NSDF per lag, reused each frame
    let mut prefix = vec![0.0f64; frame + 1]; // energy prefix, reused each frame
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + frame <= samples.len() {
        let x = &samples[start..start + frame];
        let mut acc = 0.0f64;
        for i in 0..frame {
            let v = x[i] as f64;
            acc += v * v;
            prefix[i + 1] = acc;
        }
        let rms = (acc / frame as f64).sqrt();
        if rms >= 0.01 {
            let total = prefix[frame];
            let mut best_lag = min_lag;
            let mut best_val = 0.0f64;
            for lag in min_lag..=max_lag {
                let n = frame - lag;
                // left energy = prefix[n]; right energy = total - prefix[lag]
                let e = prefix[n] + total - prefix[lag];
                let r = dot(&x[..n], &x[lag..lag + n]) as f64;
                let v = if e > 0.0 { 2.0 * r / e } else { 0.0 };
                cache[lag - min_lag] = v;
                if v > best_val {
                    best_val = v;
                    best_lag = lag;
                }
            }
            if best_val > 0.4 {
                // Pick the smallest-lag strong local peak: the fundamental, not
                // a subharmonic. A tone peaks at every period multiple, so the
                // raw global max flips octaves on rounding noise.
                let thresh = 0.9 * best_val;
                let mut chosen = best_lag;
                for lag in (min_lag + 1)..max_lag {
                    let c = cache[lag - min_lag];
                    if c >= thresh
                        && c >= cache[lag - 1 - min_lag]
                        && c >= cache[lag + 1 - min_lag]
                    {
                        chosen = lag;
                        break;
                    }
                }
                out.push(sr_f / refine(&cache, chosen, min_lag, max_lag));
            }
        }
        start += hop;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, sr: u32, secs: f64) -> Vec<f32> {
        let n = (sr as f64 * secs) as usize;
        (0..n)
            .map(|i| (0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin()) as f32)
            .collect()
    }

    fn make_wav(samples: &[f32], sr: u32) -> Vec<u8> {
        let mut data = Vec::new();
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&1u16.to_le_bytes()); // mono
        w.extend_from_slice(&sr.to_le_bytes());
        w.extend_from_slice(&(sr * 2).to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(&data);
        w
    }

    #[test]
    fn wav_roundtrip_sr_and_length() {
        let s = sine(200.0, 16000, 0.1);
        let (sr, out) = parse_wav(&make_wav(&s, 16000));
        assert_eq!(sr, 16000);
        assert_eq!(out.len(), s.len());
        assert!((out[0] - s[0]).abs() < 1e-3);
    }

    #[test]
    fn pitch_of_sine_is_accurate() {
        let s = sine(200.0, 16000, 0.5);
        let f0 = f0_contour(&s, 16000);
        assert!(!f0.is_empty());
        let mean = f0.iter().sum::<f64>() / f0.len() as f64;
        assert!((mean - 200.0).abs() < 5.0, "mean f0 = {}", mean);
    }

    #[test]
    fn silence_yields_empty() {
        let s = vec![0.0f32; 16000];
        assert!(f0_contour(&s, 16000).is_empty());
    }

    #[test]
    fn bad_wav_is_empty() {
        assert!(parse_wav(b"not a wav file at all").1.is_empty());
        assert!(f0_contour(&[], 16000).is_empty());
    }
}