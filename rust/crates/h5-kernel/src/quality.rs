//! Quality-aware consensus gate (ADV8). Pure, std-only.

/// Mean of the present (non-NaN) quality scores. NaN when none are present.
///
/// A missing modality contributes NaN and is excluded, so it downweights the
/// consensus rather than silently acting as a zero.
pub fn consensus(qualities: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for q in qualities {
        if !q.is_nan() {
            sum += q;
            n += 1.0;
        }
    }
    if n == 0.0 {
        return f64::NAN;
    }
    sum / n
}

/// Downweight a base value by a quality score. `b` is the penalty strength.
pub fn gate(base: f64, quality: f64, b: f64) -> f64 {
    base * (1.0 - b * (1.0 - quality))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consensus_ignores_missing() {
        let c = consensus(&[1.0, 1.0, f64::NAN]);
        assert!((c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn consensus_partial() {
        let c = consensus(&[1.0, 0.5]);
        assert!((c - 0.75).abs() < 1e-12);
    }

    #[test]
    fn consensus_all_missing_is_nan() {
        assert!(consensus(&[f64::NAN]).is_nan());
    }

    #[test]
    fn gate_full_quality_is_identity() {
        assert_eq!(gate(10.0, 1.0, 0.15), 10.0);
    }
}