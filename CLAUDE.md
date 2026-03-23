# AI Guidance — antitransformer

AI text detection via statistical fingerprinting.

## Use When
- Verifying the authenticity of incoming safety reports or research papers.
- Detecting "hallucination loops" where an AI is re-ingesting its own smooth output.
- Measuring the linguistic diversity of a codebase or documentation set.
- Implementing "Yield-gate" verification for documented public items.

## Grounding Patterns
- **Feature Convergence (Σ)**: A single feature (e.g., Zipf) is rarely conclusive. Always use the `pipeline` to aggregate all 5 features for a reliable verdict.
- **Human Baseline (κ)**: The baseline is derived from a 1.2M word corpus of medical and technical literature.
- **T1 Primitives**:
  - `ν + κ`: Root primitives for frequency analysis and baseline comparison.
  - `Σ + ∂`: Root primitives for result aggregation and boundary classification.

## Maintenance SOPs
- **Model Drift**: As LLM generation strategies evolve (e.g., better top-p sampling), the `zipf` thresholds may need re-calibration against a new "Current AI" corpus.
- **Chemistry Transfer**: The aggregation logic uses a Beer-Lambert weighted sum. If adding a 6th feature, you must update the `chemistry.rs` balancing equations.
- **Tokenization**: The `tokenize` module uses a custom fast-lexer; ensure it remains consistent with the `nexcore-prima` lexer patterns.

## Key Entry Points
- `src/pipeline.rs`: The main entry point for text analysis.
- `src/zipf.rs`: Power-law distribution analysis logic.
- `src/classify.rs`: The final classification boundary and verdict engine.
