//! Higher-order cross-modal interaction math. Pure, std-only.
//!
//! Evidence is a per-modality depression proxy in [0, 1]. Quality is in [0, 1].
//! A missing modality contributes NaN evidence; any NaN makes the result NaN
//! (Absent), never a silent zero.

/// Congruence: geometric mean of evidence, gated by quality consensus.
///
/// High only when every modality in the combination agrees on depression.
/// NaN when empty, lengths differ, or any evidence is NaN or negative.
pub fn congruence(evidence: &[f64], quality: &[f64]) -> f64 {
    if evidence.is_empty() || evidence.len() != quality.len() {
        return f64::NAN;
    }
    if evidence.iter().any(|e| e.is_nan() || *e < 0.0) {
        return f64::NAN;
    }
    let q = crate::quality::consensus(quality);
    if q.is_nan() {
        return f64::NAN;
    }
    let k = evidence.len() as f64;
    let product: f64 = evidence.iter().product();
    product.powf(1.0 / k) * q
}

/// Dispersion: population std of evidence, gated by quality consensus.
///
/// High when the modalities disagree (masking / atypical presentation).
/// NaN when empty, lengths differ, or any evidence is NaN.
pub fn dispersion(evidence: &[f64], quality: &[f64]) -> f64 {
    if evidence.is_empty() || evidence.len() != quality.len() {
        return f64::NAN;
    }
    if evidence.iter().any(|e| e.is_nan()) {
        return f64::NAN;
    }
    let q = crate::quality::consensus(quality);
    if q.is_nan() {
        return f64::NAN;
    }
    crate::stats::std_dev(evidence) * q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn congruence_full_agreement() {
        assert!((congruence(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn congruence_one_zero_is_zero() {
        assert_eq!(congruence(&[1.0, 0.0, 1.0], &[1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn congruence_geometric_mean() {
        let c = congruence(&[0.25, 0.25], &[1.0, 1.0]);
        assert!((c - 0.25).abs() < 1e-12);
    }

    #[test]
    fn congruence_quality_gated() {
        let c = congruence(&[1.0, 1.0], &[1.0, 0.5]);
        assert!((c - 0.75).abs() < 1e-12);
    }

    #[test]
    fn congruence_nan_evidence_is_nan() {
        assert!(congruence(&[1.0, f64::NAN], &[1.0, 1.0]).is_nan());
    }

    #[test]
    fn congruence_mismatch_is_nan() {
        assert!(congruence(&[1.0, 1.0], &[1.0]).is_nan());
        assert!(congruence(&[], &[]).is_nan());
    }

    #[test]
    fn dispersion_agreement_is_zero() {
        assert_eq!(dispersion(&[0.7, 0.7], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn dispersion_disagreement() {
        let d = dispersion(&[0.0, 1.0], &[1.0, 1.0]);
        assert!((d - 0.5).abs() < 1e-12);
    }

    #[test]
    fn dispersion_quality_gated() {
        let d = dispersion(&[0.0, 1.0], &[1.0, 0.5]);
        assert!((d - 0.375).abs() < 1e-12);
    }

    #[test]
    fn dispersion_nan_is_nan() {
        assert!(dispersion(&[f64::NAN, 1.0], &[1.0, 1.0]).is_nan());
        assert!(dispersion(&[1.0], &[1.0, 1.0]).is_nan());
    }
}