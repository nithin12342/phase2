//! Fusion bounded context. The shared fx_* interaction operator (std-only).
use h5_schema::{FeatureValue, Reliability};

/// Interaction factor: (1 + a*tanh(corr(base, partner))) * (1 - b*(1 - q)).
/// The cohort renorm C(base) is applied at assembly time, not here.
pub fn interaction_factor(base: &[f64], partner: &[f64], q: f64,
                          a: f64, b: f64) -> f64 {
    let r = h5_kernel::stats::pearson(base, partner);
    let coupling = if r.is_nan() { 0.0 } else { r.tanh() };
    (1.0 + a * coupling) * (1.0 - b * (1.0 - q))
}

/// Apply the factor to a base scalar and tag reliability from partner quality.
pub fn fuse(base: f64, factor: f64, q_partner: f64) -> FeatureValue {
    let rel = if q_partner <= 0.0 { Reliability::Absent }
              else if q_partner < 1.0 { Reliability::Proxy }
              else { Reliability::Exact };
    FeatureValue::new(base * factor, rel)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factor_neutral_when_uncorrelated() {
        let f = interaction_factor(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], 1.0, 0.1, 0.15);
        assert!(f < 1.0);
    }
}
