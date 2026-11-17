//! Mean absolute frame difference over a raw grayscale plane stream (std-only).
//!
//! Replaces cv2 read-all-JPGs on the hot path. Input is ffmpeg rawvideo gray
//! (W*H bytes per frame, consecutive frames). No JPEG decode happens here.

/// Motion trace: mean |frame_t - frame_{t-1}| over each W*H plane. The first
/// frame has no predecessor, so the trace has (n_frames - 1) values.
pub fn motion_from_raw(bytes: &[u8], width: usize, height: usize) -> Vec<f64> {
    let plane = width * height;
    if plane == 0 || bytes.len() < plane {
        return Vec::new();
    }
    let n = bytes.len() / plane;
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    for f in 1..n {
        let prev = &bytes[(f - 1) * plane..f * plane];
        let cur = &bytes[f * plane..(f + 1) * plane];
        let mut acc = 0u64;
        for i in 0..plane {
            acc += (cur[i] as i32 - prev[i] as i32).unsigned_abs() as u64;
        }
        out.push(acc as f64 / plane as f64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_frames_zero_motion() {
        let bytes = vec![128u8; 4 * 4 * 3]; // 3 identical 4x4 frames
        let m = motion_from_raw(&bytes, 4, 4);
        assert_eq!(m.len(), 2);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn known_difference() {
        // frame0 all 0, frame1 all 10 -> mean abs diff = 10
        let mut bytes = vec![0u8; 16];
        bytes.extend_from_slice(&[10u8; 16]);
        let m = motion_from_raw(&bytes, 4, 4);
        assert_eq!(m, vec![10.0]);
    }

    #[test]
    fn empty_or_short_is_empty() {
        assert!(motion_from_raw(&[], 4, 4).is_empty());
        assert!(motion_from_raw(&[0u8; 8], 4, 4).is_empty());
        assert!(motion_from_raw(&[0u8; 16], 0, 0).is_empty());
    }
}