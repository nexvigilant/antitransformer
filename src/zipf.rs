//! # Zipf's Law Deviation
//!
//! Human text follows Zipf's law: frequency ∝ 1/rank^α, with α ≈ 1.0.
//! LLM-generated text deviates — smoother distributions from attention
//! mechanism's softmax averaging.
//!
//! ## Primitive Grounding
//! - κ Comparison: rank ordering of frequencies
//! - N Quantity: frequency counts and regression coefficients

use std::collections::BTreeMap;

/// Tier: T2-C (domain composite)
///
/// Zipf analysis result.
#[derive(Debug, Clone)]
pub struct ZipfResult {
    /// Zipf exponent (alpha). Human text ≈ 1.0
    pub alpha: f64,
    /// R-squared of log-log regression fit
    pub r_squared: f64,
    /// Absolute deviation of alpha from 1.0
    pub deviation: f64,
}

/// Analyze rank-frequency distribution against Zipf's law.
///
/// Performs log-log linear regression on (rank, frequency) pairs.
/// Returns alpha (slope), R^2 (goodness of fit), and deviation from 1.0.
#[must_use]
pub fn zipf_analysis(frequencies: &BTreeMap<String, usize>) -> ZipfResult {
    if frequencies.len() < 2 {
        return ZipfResult {
            alpha: 0.0,
            r_squared: 0.0,
            deviation: 1.0,
        };
    }

    // Sort frequencies descending to get rank ordering
    let mut freq_vec: Vec<usize> = frequencies.values().copied().collect();
    freq_vec.sort_unstable_by(|a, b| b.cmp(a));

    // Log-log transform: ln(rank) vs ln(frequency)
    let n = freq_vec.len() as f64;
    let mut sum_x = 0.0_f64; // ln(rank)
    let mut sum_y = 0.0_f64; // ln(freq)
    let mut sum_xy = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;

    for (i, &freq) in freq_vec.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        let x = ((i + 1) as f64).ln();
        let y = (freq as f64).ln();
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    // Linear regression: y = a + b*x (slope b is negative alpha)
    let denom = n.mul_add(sum_x2, -(sum_x * sum_x));
    if denom.abs() < 1e-15 {
        return ZipfResult {
            alpha: 0.0,
            r_squared: 0.0,
            deviation: 1.0,
        };
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let alpha = -slope; // Zipf alpha is negation of log-log slope

    // R-squared
    let ss_tot = sum_y2 - (sum_y * sum_y) / n;
    let r_squared = if ss_tot.abs() < 1e-15 {
        0.0
    } else {
        let intercept = (sum_y - slope * sum_x) / n;
        let mut ss_res = 0.0;
        for (i, &freq) in freq_vec.iter().enumerate() {
            if freq == 0 {
                continue;
            }
            let x = ((i + 1) as f64).ln();
            let y = (freq as f64).ln();
            let predicted = intercept + slope * x;
            ss_res += (y - predicted).powi(2);
        }
        1.0 - (ss_res / ss_tot)
    };

    let deviation = (alpha - 1.0).abs();

    ZipfResult {
        alpha,
        r_squared: r_squared.clamp(0.0, 1.0),
        deviation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_freq_map(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_zipf_perfect_zipf() {
        // Perfect Zipf distribution: freq = C/rank
        // rank 1 -> 100, rank 2 -> 50, rank 3 -> 33, rank 4 -> 25
        let freqs = make_freq_map(&[("a", 100), ("b", 50), ("c", 33), ("d", 25), ("e", 20)]);
        let result = zipf_analysis(&freqs);
        // Alpha should be close to 1.0
        assert!(
            (result.alpha - 1.0).abs() < 0.15,
            "alpha={} expected ~1.0",
            result.alpha
        );
        assert!(result.r_squared > 0.95, "r2={}", result.r_squared);
    }

    #[test]
    fn test_zipf_flat_distribution() {
        // Flat distribution (all same frequency) — deviates from Zipf
        let freqs = make_freq_map(&[("a", 10), ("b", 10), ("c", 10), ("d", 10)]);
        let result = zipf_analysis(&freqs);
        // Flat = alpha near 0
        assert!(result.alpha.abs() < 0.5, "alpha={}", result.alpha);
    }

    #[test]
    fn test_zipf_empty() {
        let freqs: BTreeMap<String, usize> = BTreeMap::new();
        let result = zipf_analysis(&freqs);
        assert!((result.alpha).abs() < 1e-10);
        assert!((result.deviation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zipf_single_token() {
        let freqs = make_freq_map(&[("only", 42)]);
        let result = zipf_analysis(&freqs);
        assert!((result.deviation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zipf_deviation() {
        let freqs = make_freq_map(&[
            ("the", 200),
            ("a", 100),
            ("is", 60),
            ("in", 40),
            ("it", 30),
            ("to", 20),
            ("of", 15),
            ("on", 10),
        ]);
        let result = zipf_analysis(&freqs);
        // Should have reasonable alpha and high R^2
        assert!(result.alpha > 0.5, "alpha={}", result.alpha);
        assert!(result.r_squared > 0.8, "r2={}", result.r_squared);
    }
}
