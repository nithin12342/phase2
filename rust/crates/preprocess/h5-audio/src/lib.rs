//! Audio bounded context. Pure F0/loudness/pause extractors (std-only).
use h5_schema::{FeatureValue, Reliability};

pub mod pitch;

/// Port: a participant source of voiced F0 frames (Hz).
pub trait PitchSource {
    fn f0_frames(&self) -> Vec<f64>;
}

/// AUD-005 f0_cv = std / mean. NaN when the mean is zero or input is empty.
pub fn f0_cv(frames: &[f64]) -> FeatureValue {
    let m = h5_kernel::stats::mean(frames);
    let s = h5_kernel::stats::std_dev(frames);
    let v = if m == 0.0 { f64::NAN } else { s / m };
    FeatureValue::new(v, Reliability::Exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cv_of_constant_is_zero() {
        assert_eq!(f0_cv(&[2.0, 2.0, 2.0]).value, 0.0);
    }
    #[test]
    fn cv_of_empty_is_absent() {
        assert!(f0_cv(&[]).is_absent());
    }
}
