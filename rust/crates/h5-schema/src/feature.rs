//! Feature value objects and identifiers.
use crate::reliability::Reliability;

/// A measured feature with its reliability flag. Absent => NaN + Absent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureValue {
    pub value: f64,
    pub reliability: Reliability,
}

impl FeatureValue {
    /// Construct a value with an explicit reliability flag.
    pub fn new(value: f64, reliability: Reliability) -> Self {
        Self { value, reliability }
    }

    /// The canonical absent value: NaN flagged Absent.
    pub fn absent() -> Self {
        Self { value: f64::NAN, reliability: Reliability::Absent }
    }

    /// True when the value is missing (flagged Absent or NaN).
    pub fn is_absent(&self) -> bool {
        self.reliability == Reliability::Absent || self.value.is_nan()
    }
}

/// A feature identifier such as "AUD-005" or "fx_aud_080".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureId(pub String);

impl FeatureId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Fusion-derived interactions use the `fx_` prefix.
    pub fn is_fusion(&self) -> bool {
        self.0.starts_with("fx_")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_is_nan_and_flagged() {
        let fv = FeatureValue::absent();
        assert!(fv.value.is_nan());
        assert!(fv.is_absent());
        assert_eq!(fv.reliability, Reliability::Absent);
    }

    #[test]
    fn exact_value_not_absent() {
        let fv = FeatureValue::new(0.5, Reliability::Exact);
        assert!(!fv.is_absent());
    }

    #[test]
    fn fusion_id_detected() {
        assert!(FeatureId::new("fx_aud_080").is_fusion());
        assert!(!FeatureId::new("AUD-005").is_fusion());
    }
}