//! # Signal Aggregation
//!
//! Combines 5 detection features into a single composite score using:
//! 1. Feature normalization to [0, 1]
//! 2. Beer-Lambert weighted summation
//! 3. Hill cooperative amplification
//!
//! ## Primitive Grounding
//! - Σ Sum: Beer-Lambert linear combination
//! - ρ Recursion: Hill cooperative feedback loop

use crate::chemistry;

/// Feature weights (Beer-Lambert absorptivity coefficients).
///
/// Higher weight = stronger indicator of AI-generated text.
pub const WEIGHTS: [f64; 5] = [
    2.5, // Zipf deviation (strongest: LLMs smooth the power law)
    2.0, // Entropy std (low entropy variance = suspicious)
    1.8, // Burstiness (low burstiness = suspiciously smooth)
    2.2, // Perplexity variance (low = consistent surprise)
    1.5, // TTR deviation (weakest: TTR varies naturally)
];

/// Hill equation parameters for cooperative amplification.
pub const HILL_K_HALF: f64 = 0.5;
pub const HILL_N: f64 = 2.5;

/// Tier: T2-C (domain composite)
///
/// Raw feature values before normalization.
#[derive(Debug, Clone)]
pub struct RawFeatures {
    /// Zipf alpha deviation from 1.0
    pub zipf_deviation: f64,
    /// Entropy standard deviation (inverted: low = suspicious)
    pub entropy_std: f64,
    /// Burstiness coefficient (inverted: low = suspicious)
    pub burstiness: f64,
    /// Perplexity variance (inverted: low = suspicious)
    pub perplexity_var: f64,
    /// TTR deviation from 0.7 baseline
    pub ttr_deviation: f64,
}

/// Tier: T2-C (domain composite)
///
/// Aggregation result with intermediate values for transparency.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Normalized feature values [0, 1]
    pub normalized: [f64; 5],
    /// Beer-Lambert weighted sum
    pub beer_lambert_score: f64,
    /// Normalized composite (beer_lambert / max_possible)
    pub composite: f64,
    /// Hill-amplified score
    pub hill_score: f64,
}

/// Normalize a "low = suspicious" feature.
///
/// Maps from expected human range to [0, 1] where 1 = most suspicious.
/// Uses soft clamping via tanh for smooth boundaries.
fn normalize_inverted(value: f64, human_typical: f64) -> f64 {
    if human_typical <= 0.0 {
        return 0.5;
    }
    // Ratio: how far below typical is this value?
    let ratio = 1.0 - (value / human_typical).min(1.0);
    ratio.clamp(0.0, 1.0)
}

/// Normalize a "deviation" feature.
///
/// Maps absolute deviation to [0, 1] where 1 = max deviation.
fn normalize_deviation(deviation: f64, max_expected: f64) -> f64 {
    if max_expected <= 0.0 {
        return 0.0;
    }
    (deviation / max_expected).clamp(0.0, 1.0)
}

/// Aggregate 5 features into composite score.
///
/// Pipeline: normalize → Beer-Lambert → Hill → composite
pub fn aggregate(features: &RawFeatures) -> AggregationResult {
    // Normalize each feature to [0, 1] (1 = more likely generated)
    let normalized = [
        normalize_deviation(features.zipf_deviation, 1.0), // |alpha - 1.0|, max ~1.0
        normalize_inverted(features.entropy_std, 1.0),     // typical human std ~1.0
        normalize_inverted(features.burstiness, 0.3),      // typical human burstiness ~0.3
        normalize_inverted(features.perplexity_var, 0.5),  // typical human var ~0.5
        normalize_deviation(features.ttr_deviation, 0.3),  // max expected deviation ~0.3
    ];

    // Beer-Lambert weighted sum
    let beer_lambert_score = chemistry::beer_lambert_weighted_sum(&WEIGHTS, &normalized);

    // Normalize to [0, 1]: divide by max possible score
    let max_score: f64 = WEIGHTS.iter().sum();
    let composite = (beer_lambert_score / max_score).clamp(0.0, 1.0);

    // Hill cooperative amplification
    let hill_score = chemistry::hill_amplify(composite, HILL_K_HALF, HILL_N);

    AggregationResult {
        normalized,
        beer_lambert_score,
        composite,
        hill_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_all_suspicious() {
        let features = RawFeatures {
            zipf_deviation: 1.0, // max deviation
            entropy_std: 0.0,    // zero variance (suspicious)
            burstiness: 0.0,     // no burstiness (suspicious)
            perplexity_var: 0.0, // zero variance (suspicious)
            ttr_deviation: 0.3,  // max deviation
        };
        let result = aggregate(&features);
        assert!(result.composite > 0.8, "composite={}", result.composite);
        assert!(result.hill_score > 0.8, "hill={}", result.hill_score);
    }

    #[test]
    fn test_aggregate_all_human() {
        let features = RawFeatures {
            zipf_deviation: 0.0, // perfect Zipf
            entropy_std: 1.0,    // high variance (human)
            burstiness: 0.3,     // bursty (human)
            perplexity_var: 0.5, // varied perplexity (human)
            ttr_deviation: 0.0,  // perfect TTR
        };
        let result = aggregate(&features);
        assert!(result.composite < 0.2, "composite={}", result.composite);
        assert!(result.hill_score < 0.1, "hill={}", result.hill_score);
    }

    #[test]
    fn test_aggregate_mixed() {
        let features = RawFeatures {
            zipf_deviation: 0.5,
            entropy_std: 0.5,
            burstiness: 0.15,
            perplexity_var: 0.25,
            ttr_deviation: 0.15,
        };
        let result = aggregate(&features);
        // Should be middling
        assert!(result.composite > 0.2 && result.composite < 0.8);
    }

    #[test]
    fn test_normalize_inverted() {
        assert!((normalize_inverted(0.0, 1.0) - 1.0).abs() < 1e-10); // zero = max suspicious
        assert!((normalize_inverted(1.0, 1.0) - 0.0).abs() < 1e-10); // at typical = not suspicious
        assert!((normalize_inverted(2.0, 1.0) - 0.0).abs() < 1e-10); // above typical = clamped
    }

    #[test]
    fn test_normalize_deviation() {
        assert!((normalize_deviation(0.0, 1.0) - 0.0).abs() < 1e-10); // no deviation
        assert!((normalize_deviation(1.0, 1.0) - 1.0).abs() < 1e-10); // max deviation
        assert!((normalize_deviation(2.0, 1.0) - 1.0).abs() < 1e-10); // clamped
    }

    #[test]
    fn test_hill_amplification_effect() {
        // Hill should sharpen the distinction between low and high scores
        let low = RawFeatures {
            zipf_deviation: 0.1,
            entropy_std: 0.8,
            burstiness: 0.25,
            perplexity_var: 0.4,
            ttr_deviation: 0.05,
        };
        let high = RawFeatures {
            zipf_deviation: 0.8,
            entropy_std: 0.1,
            burstiness: 0.05,
            perplexity_var: 0.05,
            ttr_deviation: 0.25,
        };
        let low_result = aggregate(&low);
        let high_result = aggregate(&high);
        // Hill should widen the gap between low and high composites
        let composite_gap = high_result.composite - low_result.composite;
        let hill_gap = high_result.hill_score - low_result.hill_score;
        assert!(
            hill_gap > composite_gap * 0.5,
            "Hill should amplify gap: composite_gap={composite_gap}, hill_gap={hill_gap}"
        );
    }
}
