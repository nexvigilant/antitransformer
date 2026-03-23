//! # Analysis Pipeline
//!
//! Reusable 7-stage detection pipeline extracted for use by both
//! batch CLI and HTTP daemon modes.
//!
//! ## Pipeline Stages (σ Sequence)
//! 1. Tokenize (σ + N)
//! 2. Zipf analysis (κ + N)
//! 3. Entropy profile (Σ + N)
//! 4. Burstiness (ν + ∂)
//! 5. Perplexity variance (ν + κ)
//! 6. Aggregation (Σ + ρ) — Beer-Lambert + Hill
//! 7. Classification (∂ + →) — Arrhenius gate
//!
//! ## Primitive Grounding
//! - σ Sequence: ordered stage execution
//! - μ Mapping: text → features → verdict

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{aggregation, burstiness, classify, entropy, perplexity, tokenize, zipf};

/// Tier: T2-P (cross-domain primitive)
///
/// Configuration for the analysis pipeline.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Decision threshold (0.0-1.0). Default: 0.5
    pub threshold: f64,
    /// Entropy sliding window size. Default: 50
    pub window_size: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            window_size: 50,
        }
    }
}

/// Tier: T3 (domain-specific)
///
/// Feature detail for output transparency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDetail {
    pub zipf_alpha: f64,
    pub zipf_deviation: f64,
    pub entropy_std: f64,
    pub burstiness: f64,
    pub perplexity_var: f64,
    pub ttr: f64,
    pub ttr_deviation: f64,
    pub normalized: [f64; 5],
    pub beer_lambert: f64,
    pub composite: f64,
    pub hill_score: f64,
}

/// Tier: T3 (domain-specific)
///
/// Single-text analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub verdict: String,
    pub probability: f64,
    pub confidence: f64,
    pub features: FeatureDetail,
}

impl std::fmt::Display for AnalysisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "verdict={}, prob={:.3}, conf={:.3}",
            self.verdict, self.probability, self.confidence
        )
    }
}

/// Tier: T3 (domain-specific)
///
/// Input sample format.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputSample {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub label: Option<String>,
}

/// Tier: T3 (domain-specific)
///
/// Output verdict format (batch mode — includes id + label).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputVerdict {
    pub id: String,
    pub verdict: String,
    pub probability: f64,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correct: Option<bool>,
    pub features: FeatureDetail,
}

/// Tier: T2-P (cross-domain primitive)
///
/// Pipeline statistics.
#[derive(Debug, Default)]
pub struct PipelineStats {
    pub records_processed: u64,
    pub human_count: u64,
    pub generated_count: u64,
    pub correct_count: u64,
    pub labeled_count: u64,
    pub duration_secs: f64,
}

/// Analyze a single text sample through the 7-stage pipeline.
///
/// Returns `AnalysisResult` with verdict, probability, confidence, and full feature detail.
/// For texts shorter than 10 tokens, returns `insufficient_data` verdict.
#[must_use]
pub fn analyze(text: &str, config: &AnalysisConfig) -> AnalysisResult {
    // Stage 1: Tokenize (σ)
    let token_stats = tokenize::tokenize(text);

    if token_stats.total_tokens < 10 {
        return AnalysisResult {
            verdict: "insufficient_data".to_string(),
            probability: 0.0,
            confidence: 0.0,
            features: FeatureDetail {
                zipf_alpha: 0.0,
                zipf_deviation: 0.0,
                entropy_std: 0.0,
                burstiness: 0.0,
                perplexity_var: 0.0,
                ttr: token_stats.ttr,
                ttr_deviation: 0.0,
                normalized: [0.0; 5],
                beer_lambert: 0.0,
                composite: 0.0,
                hill_score: 0.0,
            },
        };
    }

    // Stage 2: Zipf analysis (κ + N)
    let zipf_result = zipf::zipf_analysis(&token_stats.frequencies);

    // Stage 3: Entropy profile (Σ + N)
    let entropy_result = entropy::entropy_profile(
        &token_stats.tokens,
        config.window_size,
        config.window_size / 2,
    );

    // Stage 4: Burstiness (ν + ∂)
    let burst_result =
        burstiness::burstiness_analysis(&token_stats.tokens, &token_stats.frequencies);

    // Stage 5: Perplexity variance (ν + κ)
    let perp_result = perplexity::perplexity_variance(text);

    // Stage 6: Aggregation (Σ + ρ) — Beer-Lambert + Hill
    let raw_features = aggregation::RawFeatures {
        zipf_deviation: zipf_result.deviation,
        entropy_std: entropy_result.std_dev,
        burstiness: burst_result.coefficient.max(0.0),
        perplexity_var: perp_result.variance,
        ttr_deviation: tokenize::ttr_deviation(token_stats.ttr),
    };
    let agg_result = aggregation::aggregate(&raw_features);

    // Stage 7: Classification (∂ + →) — Arrhenius gate
    let classification = classify::classify_with_threshold(agg_result.hill_score, config.threshold);

    AnalysisResult {
        verdict: classification.verdict.to_string(),
        probability: classification.probability,
        confidence: classification.confidence,
        features: FeatureDetail {
            zipf_alpha: zipf_result.alpha,
            zipf_deviation: zipf_result.deviation,
            entropy_std: entropy_result.std_dev,
            burstiness: burst_result.coefficient,
            perplexity_var: perp_result.variance,
            ttr: token_stats.ttr,
            ttr_deviation: tokenize::ttr_deviation(token_stats.ttr),
            normalized: agg_result.normalized,
            beer_lambert: agg_result.beer_lambert_score,
            composite: agg_result.composite,
            hill_score: agg_result.hill_score,
        },
    }
}

/// Analyze a batch of samples, tracking correctness against labels.
///
/// Returns `(verdicts, stats)` where stats include accuracy if labels present.
pub fn analyze_batch(
    samples: &[InputSample],
    config: &AnalysisConfig,
) -> (Vec<OutputVerdict>, PipelineStats) {
    let start = std::time::Instant::now();
    let mut stats = PipelineStats::default();
    let mut verdicts = Vec::with_capacity(samples.len());

    for sample in samples {
        stats.records_processed += 1;

        let result = analyze(&sample.text, config);

        info!(
            id = %sample.id,
            verdict = %result.verdict,
            prob = result.probability,
            conf = result.confidence,
            "Analyzed"
        );

        // Track verdict counts
        match result.verdict.as_str() {
            "human" => stats.human_count += 1,
            "generated" => stats.generated_count += 1,
            _ => {} // insufficient_data etc.
        }

        // Check against label if provided
        let correct = sample.label.as_ref().map(|label| {
            let is_correct = match result.verdict.as_str() {
                "human" => label == "human",
                "generated" => label == "generated" || label == "ai" || label == "llm",
                _ => false,
            };
            if is_correct {
                stats.correct_count += 1;
            }
            stats.labeled_count += 1;
            is_correct
        });

        verdicts.push(OutputVerdict {
            id: sample.id.clone(),
            verdict: result.verdict,
            probability: result.probability,
            confidence: result.confidence,
            label: sample.label.clone(),
            correct,
            features: result.features,
        });
    }

    stats.duration_secs = start.elapsed().as_secs_f64();
    (verdicts, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_short_text() {
        let config = AnalysisConfig::default();
        let result = analyze("too short", &config);
        assert_eq!(result.verdict, "insufficient_data");
        assert!((result.probability - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_returns_valid_result() {
        let config = AnalysisConfig::default();
        let text = "The quick brown fox jumps over the lazy dog. \
                    This is a sample text that should be long enough \
                    to produce a real analysis result with meaningful \
                    statistical features extracted from the content.";
        let result = analyze(text, &config);
        // Verdict is one of the two real classifications (not insufficient_data)
        assert!(
            result.verdict == "human" || result.verdict == "generated",
            "unexpected verdict: {}",
            result.verdict
        );
        // Probability is a valid probability
        assert!(
            (0.0..=1.0).contains(&result.probability),
            "probability out of range: {}",
            result.probability
        );
        assert!(
            (0.0..=1.0).contains(&result.confidence),
            "confidence out of range: {}",
            result.confidence
        );
        // Feature pipeline actually ran: Zipf alpha must be non-zero for real text,
        // entropy std must be non-negative, Hill score must be in [0,1].
        assert!(
            result.features.zipf_alpha > 0.0,
            "zipf_alpha should be positive for real text, got {}",
            result.features.zipf_alpha
        );
        assert!(
            result.features.entropy_std >= 0.0,
            "entropy_std must be non-negative, got {}",
            result.features.entropy_std
        );
        assert!(
            (0.0..=1.0).contains(&result.features.hill_score),
            "hill_score out of [0,1]: {}",
            result.features.hill_score
        );
        assert!(
            (0.0..=1.0).contains(&result.features.composite),
            "composite out of [0,1]: {}",
            result.features.composite
        );
        // Normalized features must all be in [0, 1]
        for (i, &n) in result.features.normalized.iter().enumerate() {
            assert!((0.0..=1.0).contains(&n), "normalized[{i}]={n} out of [0,1]");
        }
    }

    #[test]
    fn test_analyze_batch_empty() {
        let config = AnalysisConfig::default();
        let (verdicts, stats) = analyze_batch(&[], &config);
        assert!(verdicts.is_empty());
        assert_eq!(stats.records_processed, 0);
    }

    #[test]
    fn test_analyze_batch_with_labels() {
        let config = AnalysisConfig::default();
        let samples = vec![InputSample {
            id: "t1".to_string(),
            text: "The quick brown fox jumps over the lazy dog. \
                   This is a sample text that should be long enough \
                   to produce analysis results for testing the batch \
                   processing pipeline with label tracking."
                .to_string(),
            label: Some("human".to_string()),
        }];
        let (verdicts, stats) = analyze_batch(&samples, &config);
        assert_eq!(verdicts.len(), 1);
        assert_eq!(stats.records_processed, 1);
        assert_eq!(stats.labeled_count, 1);
        assert_eq!(verdicts[0].id, "t1");
    }

    #[test]
    fn test_config_default() {
        let config = AnalysisConfig::default();
        assert!((config.threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.window_size, 50);
    }
}
