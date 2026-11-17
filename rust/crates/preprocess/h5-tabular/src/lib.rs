//! Tabular bounded context. Pure session-metadata extractors (std-only).
use h5_schema::{FeatureValue, Reliability};

/// TAB-001 interview_duration = last - first timestamp (seconds).
pub fn interview_duration(first_ts: f64, last_ts: f64) -> FeatureValue {
    let v = last_ts - first_ts;
    let rel = if v < 0.0 { Reliability::Proxy } else { Reliability::Exact };
    FeatureValue::new(v, rel)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duration_positive() {
        assert_eq!(interview_duration(10.0, 70.0).value, 60.0);
    }
}
