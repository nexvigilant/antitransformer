//! # Antitransformer
//!
//! AI text detector via statistical fingerprints.
//!
//! Detects transformer-generated text through 5 statistical features:
//! 1. Zipf's law deviation (power law smoothing)
//! 2. Entropy uniformity (suspiciously consistent information density)
//! 3. Burstiness dampening (loss of natural word clustering)
//! 4. Perplexity consistency (uniform surprise level)
//! 5. TTR anomaly (type-token ratio deviation from human baseline)
//!
//! Features aggregated through chemistry-primitive transfer:
//! - Beer-Lambert weighted summation
//! - Hill cooperative amplification
//! - Arrhenius threshold gating
//!
//! ## Primitive Grounding (T1 → Detection)
//!
//! | Module | Dominant Primitives |
//! |--------|-------------------|
//! | tokenize | σ Sequence, N Quantity |
//! | zipf | κ Comparison, N Quantity |
//! | entropy | Σ Sum, N Quantity |
//! | burstiness | ν Frequency, ∂ Boundary |
//! | perplexity | ν Frequency, κ Comparison |
//! | aggregation | Σ Sum, ρ Recursion |
//! | classify | ∂ Boundary, → Causality |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![allow(
    clippy::exhaustive_enums,
    clippy::exhaustive_structs,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::indexing_slicing,
    reason = "Detection feature pipeline uses explicit statistical structs and bounded numeric transforms"
)]

pub mod aggregation;
pub mod burstiness;
pub mod chemistry;
pub mod classify;
pub mod daemon;
pub mod entropy;
pub mod perplexity;
pub mod pipeline;
pub mod tokenize;
pub mod zipf;
