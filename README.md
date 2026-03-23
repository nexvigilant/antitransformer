# antitransformer

AI text detection engine for the NexVigilant Core kernel. It identifies transformer-generated content through a multi-feature statistical fingerprinting process, analyzing linguistic patterns that deviate from natural human variance.

## Intent
To provide a non-semantic, statistical layer of defense against machine-generated misinformation or automated "hallucinations." It ensures that safety-critical reports or code comments have the expected high-entropy characteristics of expert human input.

## T1 Grounding (Lex Primitiva)
Dominant Primitives:
- **ν (Frequency)**: Primary for analyzing word frequency distributions and perplexity.
- **κ (Comparison)**: Compares observed statistical features against a validated human baseline.
- **Σ (Sum)**: Aggregates the five core features into a weighted probability score.
- **∂ (Boundary)**: Defines the classification boundaries (Human, Generated, or insufficient_data for texts under 10 tokens).

## The 5 Statistical Features
1. **Zipf's Law Deviation**: Detects suspiciously smooth power-law distributions.
2. **Entropy Uniformity**: Identifies consistent information density (a hallmark of AI).
3. **Burstiness Dampening**: Flags the loss of natural word-clustering "bursts."
4. **Perplexity Consistency**: Detects uniform "surprise" levels across text segments.
5. **TTR Anomaly**: Measures type-token ratio deviations from human norms.

## SOPs for Use
### Analyzing Text
```rust
use antitransformer::pipeline::{analyze, AnalysisConfig};

let config = AnalysisConfig::default();
let result = analyze("Sample text to evaluate...", &config);

if result.verdict == "generated" {
    println!("High probability of AI generation detected: {:.2}%", result.probability * 100.0);
}
```

### Aggregating Results
Use the `aggregation` module to combine multiple analyses into a single session-level fingerprint.

## License
Proprietary. Copyright (c) 2026 NexVigilant LLC. All Rights Reserved.
