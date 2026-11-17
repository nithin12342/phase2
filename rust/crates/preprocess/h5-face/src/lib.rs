//! Face bounded context. Pure action-unit extractors (std-only).
use h5_schema::{FeatureValue, Reliability};

/// Port: a participant source of per-frame AU intensities.
pub trait AuSource {
    fn au_intensity(&self, au: u8) -> Vec<f64>;
}

/// FAC-018 au_active_ratio = fraction of frames with AU > 0.05.
pub fn au_active_ratio(intensity: &[f64]) -> FeatureValue {
    if intensity.is_empty() {
        return FeatureValue::absent();
    }
    let active = intensity.iter().filter(|&&v| v > 0.05).count();
    FeatureValue::new(active as f64 / intensity.len() as f64, Reliability::Exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn active_fraction() {
        assert_eq!(au_active_ratio(&[0.0, 0.1, 0.2, 0.0]).value, 0.5);
    }
}
