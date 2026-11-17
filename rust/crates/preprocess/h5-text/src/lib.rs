//! Text bounded context. Pure lexical ratio extractors (std-only).
use h5_schema::{FeatureValue, Reliability};

/// Port: a participant source of word tokens.
pub trait TokenSource {
    fn tokens(&self) -> Vec<String>;
}

/// TXT-033 lexical_diversity (type-token ratio) = unique / total. NaN at zero.
pub fn lexical_diversity(unique: usize, total: usize) -> FeatureValue {
    let v = if total == 0 { f64::NAN } else { unique as f64 / total as f64 };
    FeatureValue::new(v, Reliability::Exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diversity_half() {
        assert_eq!(lexical_diversity(5, 10).value, 0.5);
    }
    #[test]
    fn diversity_empty_is_absent() {
        assert!(lexical_diversity(0, 0).is_absent());
    }
}
