//! Video bounded context. Pure motion/quality extractors (std-only).
use h5_schema::{FeatureValue, Reliability};

pub mod motion;

/// Port: a participant source of per-frame motion magnitudes.
pub trait MotionSource {
    fn motion(&self) -> Vec<f64>;
}

/// VID-005 stillness_ratio = fraction of frames below the motion threshold.
pub fn stillness_ratio(motion: &[f64], threshold: f64) -> FeatureValue {
    if motion.is_empty() {
        return FeatureValue::absent();
    }
    let still = motion.iter().filter(|&&m| m < threshold).count();
    FeatureValue::new(still as f64 / motion.len() as f64, Reliability::Exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_still() {
        assert_eq!(stillness_ratio(&[0.0, 1.0, 0.0, 1.0], 0.5).value, 0.5);
    }
}
