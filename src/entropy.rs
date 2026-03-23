//! # Sliding Window Shannon Entropy
//!
//! Computes Shannon entropy over sliding windows of text.
//! Human text shows high entropy variance (creative bursts + simple passages).
//! LLM text has suspiciously uniform entropy across windows.
//!
//! ## Primitive Grounding
//! - Σ Sum: entropy aggregation across windows
//! - N Quantity: probability distributions, statistics

use std::collections::BTreeMap;

/// Tier: T2-C (domain composite)
///
/// Entropy profile statistics.
#[derive(Debug, Clone)]
pub struct EntropyProfile {
    /// Mean entropy across all windows
    pub mean: f64,
    /// Standard deviation of window entropies
    pub std_dev: f64,
    /// Range (max - min) of window entropies
    pub range: f64,
    /// Number of windows analyzed
    pub window_count: usize,
    /// Individual window entropy values
    pub values: Vec<f64>,
}

/// Compute Shannon entropy of a token slice.
///
/// H = -Σ p(x) × log2(p(x))
#[must_use]
pub fn shannon_entropy(tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }

    let total = tokens.len() as f64;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute entropy profile over sliding windows.
///
/// # Arguments
/// * `tokens` - Full token sequence
/// * `window_size` - Tokens per window (default: 50)
/// * `step` - Window step size (default: 25, 50% overlap)
pub fn entropy_profile(tokens: &[String], window_size: usize, step: usize) -> EntropyProfile {
    let window_size = window_size.max(1);
    let step = step.max(1);

    if tokens.len() < window_size {
        let h = shannon_entropy(tokens);
        return EntropyProfile {
            mean: h,
            std_dev: 0.0,
            range: 0.0,
            window_count: if tokens.is_empty() { 0 } else { 1 },
            values: if tokens.is_empty() { vec![] } else { vec![h] },
        };
    }

    let mut values = Vec::new();
    let mut start = 0;
    while start + window_size <= tokens.len() {
        let window = &tokens[start..start + window_size];
        values.push(shannon_entropy(window));
        start += step;
    }

    let window_count = values.len();
    if window_count == 0 {
        return EntropyProfile {
            mean: 0.0,
            std_dev: 0.0,
            range: 0.0,
            window_count: 0,
            values: vec![],
        };
    }

    let mean = values.iter().sum::<f64>() / window_count as f64;

    let variance = if window_count > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (window_count - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    EntropyProfile {
        mean,
        std_dev,
        range,
        window_count,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // 4 equally likely tokens -> H = log2(4) = 2.0
        let t = tokens(&["a", "b", "c", "d"]);
        let h = shannon_entropy(&t);
        assert!((h - 2.0).abs() < 1e-10, "H={h}");
    }

    #[test]
    fn test_shannon_entropy_single_type() {
        // All same token -> H = 0
        let t = tokens(&["same", "same", "same"]);
        let h = shannon_entropy(&t);
        assert!(h.abs() < 1e-10, "H={h}");
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let t: Vec<String> = vec![];
        assert!(shannon_entropy(&t).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_profile_uniform_text() {
        // Uniform text should have low std_dev
        let t: Vec<String> = (0..200).map(|i| format!("word{}", i % 20)).collect();
        let profile = entropy_profile(&t, 50, 25);
        assert!(profile.window_count > 1);
        // All windows drawn from same distribution -> low variance
        assert!(
            profile.std_dev < 0.5,
            "std_dev={} (expected low for uniform)",
            profile.std_dev
        );
    }

    #[test]
    fn test_entropy_profile_varied_text() {
        // First half: low entropy (repetitive), second half: high entropy (varied)
        let mut t = Vec::new();
        for _ in 0..100 {
            t.push("the".to_string());
        }
        for i in 0..100 {
            t.push(format!("unique{i}"));
        }
        let profile = entropy_profile(&t, 50, 25);
        assert!(
            profile.std_dev > 0.5,
            "std_dev={} (expected high for varied)",
            profile.std_dev
        );
        assert!(profile.range > 1.0, "range={}", profile.range);
    }

    #[test]
    fn test_entropy_non_negative() {
        let t = tokens(&["hello", "world", "hello"]);
        let h = shannon_entropy(&t);
        assert!(h >= 0.0);
    }
}
