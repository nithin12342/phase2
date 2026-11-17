//! Selective state-space scan (Mamba S6). Pure Rust, std-only.
//!
//! The recurrence is input-dependent (selective): delta, A, B, C vary per step.
//! Discretization uses ZOH for A and Euler for B:
//!   h_t = exp(delta_t * A_t) * h_{t-1} + (delta_t * B_t) * x_t
//!   y_t = C_t * h_t + D * x_t

/// Run the selective scan over a sequence. Returns one output per input step.
///
/// If any parameter slice length differs from `x`, returns NaN per step instead
/// of panicking. This keeps the kernel safe on malformed adapter input.
pub fn selective_scan(
    x: &[f64],
    delta: &[f64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: f64,
) -> Vec<f64> {
    let n = x.len();
    if delta.len() != n || a.len() != n || b.len() != n || c.len() != n {
        return vec![f64::NAN; n];
    }
    let mut y = Vec::with_capacity(n);
    let mut h = 0.0;
    for t in 0..n {
        let dt = delta[t];
        let a_bar = (dt * a[t]).exp();
        let b_bar = dt * b[t];
        h = a_bar * h + b_bar * x[t];
        y.push(c[t] * h + d * x[t]);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrator_is_cumsum() {
        // A=0 => a_bar=1; B=1, delta=1, C=1, D=0 => h accumulates x.
        let y = selective_scan(
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
            0.0,
        );
        assert_eq!(y, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn decay_shrinks_state() {
        let y = selective_scan(
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
        );
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - (-1.0f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn mismatched_lengths_are_nan_not_panic() {
        let y = selective_scan(&[1.0, 2.0], &[1.0], &[0.0], &[1.0], &[1.0], 0.0);
        assert!(y.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn empty_input_empty_output() {
        let y = selective_scan(&[], &[], &[], &[], &[], 0.0);
        assert!(y.is_empty());
    }
}