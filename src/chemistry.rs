//! # Inlined Chemistry Primitives
//!
//! Three chemistry equations transferred to AI text detection:
//! - **Beer-Lambert**: Weighted feature summation (signal intensity ∝ concentration)
//! - **Hill**: Cooperative amplification (sigmoidal response from composite score)
//! - **Arrhenius**: Threshold gating (activation energy barrier for classification)
//!
//! ## Primitive Grounding
//! - Beer-Lambert: Σ Sum + N Quantity (linear accumulation)
//! - Hill: ρ Recursion + κ Comparison (cooperative feedback)
//! - Arrhenius: ∂ Boundary + → Causality (threshold gate)

/// Tier: T2-P (cross-domain transfer from chemistry)
///
/// Beer-Lambert weighted sum: A = Σ(ε_i × feature_i)
///
/// Each feature contributes proportionally to its weight (absorptivity).
/// Linear combination — no interaction between features.
///
/// Uses const generics to enforce equal-length arrays at compile time,
/// eliminating the need for runtime length assertions.
#[must_use]
pub fn beer_lambert_weighted_sum<const N: usize>(weights: &[f64; N], features: &[f64; N]) -> f64 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(w, f)| w * f)
        .sum()
}

/// Tier: T2-P (cross-domain transfer from chemistry)
///
/// Hill cooperative amplification: Y = x^nH / (K^nH + x^nH)
///
/// Produces sigmoidal response — gradual at extremes, steep near K_half.
/// nH > 1 = positive cooperativity (amplification).
///
/// # Arguments
/// * `x` - Composite signal score (0.0 to 1.0 typical)
/// * `k_half` - Half-saturation point (score at 50% response)
/// * `n_hill` - Hill coefficient (cooperativity, >1 = amplification)
#[must_use]
pub fn hill_amplify(x: f64, k_half: f64, n_hill: f64) -> f64 {
    if x <= 0.0 || k_half <= 0.0 || n_hill <= 0.0 {
        return 0.0;
    }
    let x_n = x.powf(n_hill);
    let k_n = k_half.powf(n_hill);
    x_n / (k_n + x_n)
}

/// Tier: T2-P (cross-domain transfer from chemistry)
///
/// Arrhenius activation probability: p = exp(-Ea / (score × scale))
///
/// Maps Hill-amplified score to classification probability.
/// High activation energy = harder to classify as generated.
/// Capped at 1.0 for valid probability.
///
/// # Arguments
/// * `activation_energy` - Ea (detection barrier)
/// * `score` - Amplified composite score
/// * `scale` - Temperature-like scaling factor
#[must_use]
pub fn arrhenius_probability(activation_energy: f64, score: f64, scale: f64) -> f64 {
    let effective = score * scale;
    if effective <= 0.0 {
        return 0.0;
    }
    let raw = (-activation_energy / effective).exp();
    raw.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beer_lambert_basic() {
        let weights = [2.0, 1.0, 3.0];
        let features = [0.5, 0.8, 0.2];
        let result = beer_lambert_weighted_sum(&weights, &features);
        // 2*0.5 + 1*0.8 + 3*0.2 = 1.0 + 0.8 + 0.6 = 2.4
        assert!((result - 2.4).abs() < 1e-10);
    }

    #[test]
    fn test_beer_lambert_zeros() {
        let weights = [1.0, 2.0];
        let features = [0.0, 0.0];
        assert!((beer_lambert_weighted_sum(&weights, &features)).abs() < 1e-10);
    }

    #[test]
    fn test_hill_at_k_half() {
        // At x = K_half, response should be exactly 0.5
        let y = hill_amplify(0.5, 0.5, 2.5);
        assert!((y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hill_monotonic() {
        // Higher input -> higher output (monotonically increasing)
        let y1 = hill_amplify(0.3, 0.5, 2.5);
        let y2 = hill_amplify(0.5, 0.5, 2.5);
        let y3 = hill_amplify(0.8, 0.5, 2.5);
        assert!(y1 < y2);
        assert!(y2 < y3);
    }

    #[test]
    fn test_hill_bounds() {
        // Output should be in [0, 1]
        let y = hill_amplify(100.0, 0.5, 2.5);
        assert!(y >= 0.0 && y <= 1.0);
        let y_zero = hill_amplify(0.0, 0.5, 2.5);
        assert!((y_zero).abs() < 1e-10);
    }

    #[test]
    fn test_arrhenius_high_score() {
        // High score should approach 1.0
        let p = arrhenius_probability(3.0, 0.9, 10.0);
        assert!(p > 0.5);
        assert!(p <= 1.0);
    }

    #[test]
    fn test_arrhenius_low_score() {
        // Low score should give low probability
        let p = arrhenius_probability(3.0, 0.05, 10.0);
        assert!(p < 0.5);
    }

    #[test]
    fn test_arrhenius_zero_score() {
        let p = arrhenius_probability(3.0, 0.0, 10.0);
        assert!((p).abs() < 1e-10);
    }
}
