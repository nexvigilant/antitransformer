//! # Perplexity Variance
//!
//! Computes per-sentence entropy and measures variance.
//! LLM text has suspiciously consistent perplexity (uniform surprise).
//! Human text shows high variance (some sentences are predictable, others creative).
//!
//! ## Primitive Grounding
//! - ν Frequency: per-sentence information content
//! - κ Comparison: variance across sentences

use crate::entropy::shannon_entropy;

/// Tier: T2-C (domain composite)
///
/// Perplexity variance result.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Mean sentence entropy
    pub mean_entropy: f64,
    /// Variance of sentence entropies
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of sentences analyzed
    pub sentence_count: usize,
    /// Per-sentence entropy values
    pub sentence_entropies: Vec<f64>,
}

/// Split text into sentences (simple heuristic).
///
/// Splits on `.`, `!`, `?` followed by whitespace or end.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if (ch == '.' || ch == '!' || ch == '?') && current.split_whitespace().count() > 1 {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }

    // Remaining text as final sentence if substantial
    let trimmed = current.trim().to_string();
    if trimmed.split_whitespace().count() > 1 {
        sentences.push(trimmed);
    }

    sentences
}

/// Tokenize a sentence into lowercased words (simple).
fn sentence_tokens(sentence: &str) -> Vec<String> {
    sentence
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Analyze perplexity variance across sentences.
///
/// Computes Shannon entropy per sentence, then measures
/// variance of these entropies.
pub fn perplexity_variance(text: &str) -> PerplexityResult {
    let sentences = split_sentences(text);

    if sentences.is_empty() {
        return PerplexityResult {
            mean_entropy: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            sentence_count: 0,
            sentence_entropies: vec![],
        };
    }

    let sentence_entropies: Vec<f64> = sentences
        .iter()
        .map(|s| {
            let tokens = sentence_tokens(s);
            shannon_entropy(&tokens)
        })
        .collect();

    let n = sentence_entropies.len() as f64;
    let mean_entropy = sentence_entropies.iter().sum::<f64>() / n;

    let variance = if sentence_entropies.len() > 1 {
        sentence_entropies
            .iter()
            .map(|h| (h - mean_entropy).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };

    PerplexityResult {
        mean_entropy,
        variance,
        std_dev: variance.sqrt(),
        sentence_count: sentence_entropies.len(),
        sentence_entropies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_varied_sentences() {
        let text = "The the the the the. \
                     Quantum mechanics describes the behavior of particles at subatomic scales. \
                     A a a a a a. \
                     The implications of this discovery are far-reaching and profound.";
        let result = perplexity_variance(text);
        assert!(result.sentence_count >= 2);
        // Mixed repetitive + varied -> non-zero variance
        assert!(result.variance > 0.0, "var={}", result.variance);
    }

    #[test]
    fn test_perplexity_uniform_sentences() {
        // Sentences with similar word distributions
        let text = "The cat sat on the mat. The dog sat on the rug. The bird sat on the branch.";
        let result = perplexity_variance(text);
        // Similar structure -> lower variance than varied text
        assert!(result.sentence_count >= 2);
        assert!(result.std_dev < 1.5, "std={}", result.std_dev);
    }

    #[test]
    fn test_perplexity_empty() {
        let result = perplexity_variance("");
        assert_eq!(result.sentence_count, 0);
        assert!((result.variance).abs() < 1e-10);
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? Fine thanks!");
        assert_eq!(sentences.len(), 3);
    }
}
