//! # Text Tokenization
//!
//! Converts raw text into token stream with frequency statistics.
//!
//! ## Primitive Grounding
//! - σ Sequence: text → ordered token stream
//! - N Quantity: token counts, type-token ratio
//! - μ Mapping: token → frequency map

use std::collections::BTreeMap;

/// Tier: T2-C (domain composite)
///
/// Tokenized text with frequency statistics.
#[derive(Debug, Clone)]
pub struct TokenStats {
    /// Total token count
    pub total_tokens: usize,
    /// Unique token count (types)
    pub unique_tokens: usize,
    /// Type-token ratio (unique / total)
    pub ttr: f64,
    /// Token frequency map (lowercased, deterministic ordering)
    pub frequencies: BTreeMap<String, usize>,
    /// Ordered tokens (lowercased)
    pub tokens: Vec<String>,
}

/// Tokenize text into lowercased words, stripping punctuation.
///
/// Uses whitespace splitting + punctuation trimming. Sufficient for
/// statistical fingerprinting — we care about distribution shapes,
/// not linguistic precision.
#[must_use]
pub fn tokenize(text: &str) -> TokenStats {
    let tokens: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();

    let total_tokens = tokens.len();
    let mut frequencies: BTreeMap<String, usize> = BTreeMap::new();
    for token in &tokens {
        *frequencies.entry(token.clone()).or_insert(0) += 1;
    }
    let unique_tokens = frequencies.len();
    let ttr = if total_tokens > 0 {
        unique_tokens as f64 / total_tokens as f64
    } else {
        0.0
    };

    TokenStats {
        total_tokens,
        unique_tokens,
        ttr,
        frequencies,
        tokens,
    }
}

/// Compute TTR deviation from human baseline (0.7).
///
/// Returns absolute deviation: |TTR - 0.7|
/// Higher deviation = more suspicious.
#[must_use]
pub fn ttr_deviation(ttr: f64) -> f64 {
    (ttr - 0.7).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let stats = tokenize("The quick brown fox jumps over the lazy dog");
        assert_eq!(stats.total_tokens, 9);
        assert_eq!(stats.unique_tokens, 8); // "the" appears twice
        assert!(stats.frequencies.get("the").is_some_and(|&c| c == 2));
    }

    #[test]
    fn test_tokenize_punctuation() {
        let stats = tokenize("Hello, world! This is a test.");
        // Punctuation stripped, all lowered
        assert!(stats.tokens.contains(&"hello".to_string()));
        assert!(stats.tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let stats = tokenize("");
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.unique_tokens, 0);
        assert!((stats.ttr).abs() < 1e-10);
    }

    #[test]
    fn test_ttr_calculation() {
        let stats = tokenize("a b c d e f g h i j");
        // All unique: TTR = 10/10 = 1.0
        assert!((stats.ttr - 1.0).abs() < 1e-10);

        let stats2 = tokenize("the the the the the");
        // All same: TTR = 1/5 = 0.2
        assert!((stats2.ttr - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_ttr_deviation_from_baseline() {
        assert!((ttr_deviation(0.7)).abs() < 1e-10); // perfect human
        assert!((ttr_deviation(0.5) - 0.2).abs() < 1e-10); // below baseline
        assert!((ttr_deviation(0.9) - 0.2).abs() < 1e-10); // above baseline
    }
}
