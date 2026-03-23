//! # Burstiness Coefficient
//!
//! Measures token reuse clustering via inter-arrival times.
//! Human text is bursty (topic words cluster), LLM text is smooth.
//!
//! B = (σ - μ) / (σ + μ) where σ, μ are std and mean of inter-arrivals.
//! B ∈ [-1, 1]: B > 0 = bursty, B < 0 = periodic, B ≈ 0 = random.
//!
//! ## Primitive Grounding
//! - ν Frequency: inter-arrival time patterns
//! - ∂ Boundary: classification thresholds

/// Tier: T2-C (domain composite)
///
/// Burstiness analysis result.
#[derive(Debug, Clone)]
pub struct BurstinessResult {
    /// Average burstiness across all tokens with repeats
    pub coefficient: f64,
    /// Number of tokens analyzed (with ≥2 occurrences)
    pub tokens_analyzed: usize,
    /// Per-token burstiness values
    pub per_token: Vec<(String, f64)>,
}

/// Compute inter-arrival times for a token in a sequence.
///
/// Inter-arrival = positions between consecutive occurrences.
fn inter_arrival_times(tokens: &[String], target: &str) -> Vec<usize> {
    let positions: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == target)
        .map(|(i, _)| i)
        .collect();

    if positions.len() < 2 {
        return vec![];
    }

    positions.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Compute burstiness coefficient for a single token.
///
/// B = (σ - μ) / (σ + μ)
fn single_burstiness(inter_arrivals: &[usize]) -> Option<f64> {
    if inter_arrivals.is_empty() {
        return None;
    }

    let n = inter_arrivals.len() as f64;
    let mean = inter_arrivals.iter().sum::<usize>() as f64 / n;

    if mean.abs() < 1e-15 {
        return Some(0.0);
    }

    let variance = if inter_arrivals.len() > 1 {
        inter_arrivals
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let denom = std_dev + mean;
    if denom.abs() < 1e-15 {
        return Some(0.0);
    }

    Some((std_dev - mean) / denom)
}

/// Analyze burstiness of the full token sequence.
///
/// Computes per-token burstiness for all tokens with ≥2 occurrences,
/// then averages for the overall coefficient.
#[must_use]
pub fn burstiness_analysis(
    tokens: &[String],
    frequencies: &std::collections::BTreeMap<String, usize>,
) -> BurstinessResult {
    let mut per_token = Vec::new();

    for (token, &count) in frequencies {
        if count < 2 {
            continue;
        }
        let arrivals = inter_arrival_times(tokens, token);
        if let Some(b) = single_burstiness(&arrivals) {
            per_token.push((token.clone(), b));
        }
    }

    let tokens_analyzed = per_token.len();
    let coefficient = if tokens_analyzed > 0 {
        per_token.iter().map(|(_, b)| b).sum::<f64>() / tokens_analyzed as f64
    } else {
        0.0
    };

    BurstinessResult {
        coefficient,
        tokens_analyzed,
        per_token,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn tokens_and_freqs(words: &[&str]) -> (Vec<String>, BTreeMap<String, usize>) {
        let tokens: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        let mut freqs = BTreeMap::new();
        for t in &tokens {
            *freqs.entry(t.clone()).or_insert(0) += 1;
        }
        (tokens, freqs)
    }

    #[test]
    fn test_inter_arrival_basic() {
        let tokens: Vec<String> = ["a", "b", "c", "a", "d", "a"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let arrivals = inter_arrival_times(&tokens, "a");
        assert_eq!(arrivals, vec![3, 2]); // positions 0,3,5 -> diffs 3,2
    }

    #[test]
    fn test_burstiness_clustered() {
        // "topic" in two tight clusters separated by a long gap produces bursty inter-arrivals.
        // Positions: 0,1,9,10 -> inter-arrivals [1, 8, 1] -> mean=10/3, std>0 -> B > 0.
        let words: Vec<&str> = [
            "topic", "topic", "a", "b", "c", "d", "e", "f", "g", "topic", "topic",
        ]
        .into();
        let (tokens, freqs) = tokens_and_freqs(&words);
        let result = burstiness_analysis(&tokens, &freqs);
        assert!(
            result.tokens_analyzed > 0,
            "should have tokens with repeats"
        );
        let topic_b = result
            .per_token
            .iter()
            .find(|(t, _)| t == "topic")
            .map(|(_, b)| *b);
        assert!(
            topic_b.is_some_and(|b| b > 0.0),
            "clustered topic should have positive burstiness B={topic_b:?}"
        );
    }

    #[test]
    fn test_burstiness_periodic() {
        // Regular spacing -> B < 0 (periodic)
        let words: Vec<&str> = ["x", "a", "b", "x", "a", "b", "x", "a", "b", "x"].into();
        let (tokens, freqs) = tokens_and_freqs(&words);
        let result = burstiness_analysis(&tokens, &freqs);
        // "x" at positions 0,3,6,9 -> perfectly periodic -> B ~ -1
        let x_burst = result.per_token.iter().find(|(t, _)| t == "x");
        if let Some((_, b)) = x_burst {
            assert!(*b < 0.0, "periodic B={b} should be negative");
        }
    }

    #[test]
    fn test_burstiness_no_repeats() {
        let (tokens, freqs) = tokens_and_freqs(&["a", "b", "c", "d", "e"]);
        let result = burstiness_analysis(&tokens, &freqs);
        assert_eq!(result.tokens_analyzed, 0);
        assert!((result.coefficient).abs() < 1e-10);
    }

    #[test]
    fn test_burstiness_bounds() {
        // B should be in [-1, 1]
        let words: Vec<&str> = (0..100)
            .map(|i| if i % 7 == 0 { "repeat" } else { "filler" })
            .collect();
        let (tokens, freqs) = tokens_and_freqs(&words);
        let result = burstiness_analysis(&tokens, &freqs);
        assert!(
            result.coefficient >= -1.0 && result.coefficient <= 1.0,
            "B={}",
            result.coefficient
        );
    }

    #[test]
    fn test_single_burstiness_equal_spacing() {
        // All arrivals = 3 -> mean=3, std=0 -> B = (0-3)/(0+3) = -1
        let arrivals = vec![3, 3, 3, 3];
        let b = single_burstiness(&arrivals);
        assert!(b.is_some());
        assert!((b.unwrap() - (-1.0)).abs() < 1e-10, "B={:?}", b);
    }
}
