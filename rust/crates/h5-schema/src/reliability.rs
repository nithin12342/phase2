//! Reliability flag for every emitted feature. Never a silent zero.

/// How a feature value was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Raw input present; the value is computed directly.
    Exact,
    /// Estimated from a partial or proxy signal.
    Proxy,
    /// Raw input missing; the value is NaN.
    Absent,
}