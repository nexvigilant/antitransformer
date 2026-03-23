//! # Classification via Arrhenius Threshold
//!
//! Maps Hill-amplified score through Arrhenius activation gate
//! to produce final verdict (human/generated) with confidence.
//!
//! ## Primitive Grounding
//! - ∂ Boundary: classification threshold
//! - → Causality: score → verdict mapping

use crate::chemistry;
use serde::Serialize;

/// Arrhenius parameters for classification gate.
pub const ACTIVATION_ENERGY: f64 = 3.0;
pub const SCALE_FACTOR: f64 = 10.0;
pub const DECISION_THRESHOLD: f64 = 0.5;

/// Tier: T3 (domain-specific)
///
/// Classification verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Classified as human-written
    Human,
    /// Classified as AI-generated
    Generated,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Human => write!(f, "human"),
            Verdict::Generated => write!(f, "generated"),
        }
    }
}

/// Tier: T3 (domain-specific)
///
/// Full classification result with transparency.
#[derive(Debug, Clone, Serialize)]
pub struct Classification {
    /// Final verdict
    pub verdict: Verdict,
    /// Classification probability (0.0 = certainly human, 1.0 = certainly generated)
    pub probability: f64,
    /// Confidence in the classification (distance from decision boundary)
    pub confidence: f64,
}

/// Classify a Hill-amplified score through the Arrhenius gate.
///
/// # Arguments
/// * `hill_score` - Output from Hill cooperative amplification [0, 1]
///
/// # Returns
/// Classification with verdict, probability, and confidence
#[must_use]
pub fn classify(hill_score: f64) -> Classification {
    let probability = chemistry::arrhenius_probability(ACTIVATION_ENERGY, hill_score, SCALE_FACTOR);

    let verdict = if probability > DECISION_THRESHOLD {
        Verdict::Generated
    } else {
        Verdict::Human
    };

    // Confidence = distance from decision boundary, scaled to [0, 1]
    let confidence = ((probability - DECISION_THRESHOLD).abs() * 2.0).min(1.0);

    Classification {
        verdict,
        probability,
        confidence,
    }
}

/// Classify with custom threshold.
#[must_use]
pub fn classify_with_threshold(hill_score: f64, threshold: f64) -> Classification {
    let probability = chemistry::arrhenius_probability(ACTIVATION_ENERGY, hill_score, SCALE_FACTOR);

    let verdict = if probability > threshold {
        Verdict::Generated
    } else {
        Verdict::Human
    };

    let confidence = ((probability - threshold).abs() * 2.0).min(1.0);

    Classification {
        verdict,
        probability,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_high_score() {
        // High hill score -> generated
        let result = classify(0.9);
        assert_eq!(result.verdict, Verdict::Generated);
        assert!(result.probability > DECISION_THRESHOLD);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_classify_low_score() {
        // Low hill score -> human
        let result = classify(0.1);
        assert_eq!(result.verdict, Verdict::Human);
        assert!(result.probability < DECISION_THRESHOLD);
    }

    #[test]
    fn test_classify_zero() {
        let result = classify(0.0);
        assert_eq!(result.verdict, Verdict::Human);
        assert!((result.probability).abs() < 1e-10);
    }

    #[test]
    fn test_probability_bounds() {
        for score_pct in 0..=100 {
            let score = score_pct as f64 / 100.0;
            let result = classify(score);
            assert!(
                result.probability >= 0.0 && result.probability <= 1.0,
                "score={score} -> prob={}",
                result.probability
            );
        }
    }

    #[test]
    fn test_custom_threshold() {
        let hill_score = 0.6;
        let strict = classify_with_threshold(hill_score, 0.8);
        let lenient = classify_with_threshold(hill_score, 0.3);
        // Same score, different thresholds can yield different verdicts
        assert!(strict.probability == lenient.probability);
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", Verdict::Human), "human");
        assert_eq!(format!("{}", Verdict::Generated), "generated");
    }
}
