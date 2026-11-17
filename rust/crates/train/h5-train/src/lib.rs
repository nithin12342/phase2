//! Training domain (separate from preprocessing). Pure metrics + loss math.
/// Mean absolute error for PHQ-8 regression. NaN when empty.
pub fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    if pred.is_empty() || pred.len() != truth.len() {
        return f64::NAN;
    }
    let s: f64 = pred.iter().zip(truth).map(|(p, t)| (p - t).abs()).sum();
    s / pred.len() as f64
}

/// Mean squared error. NaN when empty or mismatched.
pub fn mse(pred: &[f64], truth: &[f64]) -> f64 {
    if pred.is_empty() || pred.len() != truth.len() {
        return f64::NAN;
    }
    let s: f64 = pred.iter().zip(truth).map(|(p, t)| (p - t) * (p - t)).sum();
    s / pred.len() as f64
}

/// Binary F1 at a probability threshold.
pub fn f1_binary(prob: &[f64], truth: &[bool], thr: f64) -> f64 {
    let mut tp = 0.0; let mut fp = 0.0; let mut fn_ = 0.0;
    for (p, t) in prob.iter().zip(truth) {
        let yh = *p >= thr;
        if yh && *t { tp += 1.0; }
        else if yh && !*t { fp += 1.0; }
        else if !yh && *t { fn_ += 1.0; }
    }
    let prec = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
    let rec = if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };
    if prec + rec == 0.0 { 0.0 } else { 2.0 * prec * rec / (prec + rec) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mae_basic() { assert_eq!(mae(&[1.0, 2.0], &[1.0, 4.0]), 1.0); }
    #[test]
    fn f1_perfect() { assert_eq!(f1_binary(&[0.9, 0.1], &[true, false], 0.5), 1.0); }
}
