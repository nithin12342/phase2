//! Pure statistics over f64 slices. Empty input yields NaN, never a silent zero.

/// Arithmetic mean. NaN for an empty slice.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Population variance. NaN for an empty slice.
pub fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

/// Population standard deviation.
pub fn std_dev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

/// Pearson correlation. NaN when lengths differ, n < 2, or either side constant.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    let ma = mean(a);
    let mb = mean(b);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va == 0.0 || vb == 0.0 {
        return f64::NAN;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Average ranks with ties shared. Input order is preserved in the output.
fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&i, &j| {
        xs[i].partial_cmp(&xs[j]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut r = vec![0.0; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation (Pearson on average ranks).
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    pearson(&ranks(a), &ranks(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_basic() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
    }

    #[test]
    fn mean_empty_is_nan() {
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn std_constant_is_zero() {
        assert_eq!(std_dev(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn pearson_perfect_positive() {
        let r = pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_constant_is_nan() {
        assert!(pearson(&[1.0, 1.0], &[1.0, 2.0]).is_nan());
    }

    #[test]
    fn spearman_monotonic_is_one() {
        let r = spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn spearman_handles_ties() {
        let r = spearman(&[1.0, 1.0, 2.0], &[1.0, 1.0, 2.0]);
        assert!((r - 1.0).abs() < 1e-12);
    }
}