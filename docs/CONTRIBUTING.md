# Antitransformer Technical Guide & Roadmap

This document outlines the architecture, setup, and key development goals for the `antitransformer` crate.

## Architecture

This is a **High-Performance Rust Application** designed for low-latency AI detection. No Python, no heavy neural networks at runtime. Pure Math.

### The Pipeline (7-Stage T2-P)

1. **Tokenization**: Splits text using simple heuristics (soon: Subword/BPE).
2. **Zipf Analysis**: Computes deviation from Power Law distribution.
3. **Entropy Windowing**: Slides a window (default: 50 tokens) to calculate local Shannon Entropy.
4. **Burstiness**: Measures inter-arrival time variance of tokens.
5. **Perplexity**: Estimates sentence-level surprise (simulated currently, target: KenLM).
6. **Aggregation (Chemistry Transfer)**: Normalizes features [0,1], applies Beer-Lambert weighted sum, then Hill cooperative amplification.
7. **Classification (Arrhenius Gate)**: Arrhenius activation probability produces `P(AI)`; threshold at 0.5 → verdict.

### Project Structure

```
src/
├── bin/          # Tools
│   └── tune-weights.rs  # Weight Calibration Utility
├── lib.rs        # Main library exports
├── pipeline.rs   # Core orchestration logic
├── aggregation.rs# Weight constants & LR model implementation
├── perplexity.rs # Entropy variance implementation
└── ...           # Individual metric modules
```

## Running & Calibration

### 1. Build and Run CLI

```bash
cargo run --release -- batch < input.json
```

### 2. Calibrate Weights (CRITICAL)

The weights in `src/aggregation.rs` are Beer-Lambert absorptivity coefficients. The `tune-weights` binary calibrates them via Logistic Regression gradient descent on labeled data. To re-train them:

1. Prepare a dataset in JSONL format: `{"text": "...", "label": "human"|"ai"}`.
2. Run the tuner:

    ```bash
    cargo run --release --bin tune-weights -- < my_dataset.jsonl
    ```

3. Copy the output (`pub const WEIGHTS = ...`) and replace the block in `src/aggregation.rs`.
4. Recompile and verify accuracy.

## Roadmap for Future LLMs

If you are an LLM tasked with improving this codebase, prioritize these tasks:

### P0: Weight Calibration

- **Current State**: Weights are manually tuned/heuristic.
- **Task**: Run `tune-weights` on a high-quality dataset (e.g., OpenWebText + GPT-4o output) to find optimal parameters.

### P1: True Perplexity (KenLM)

- **Current State**: Calculating Shannon Entropy of word distribution, which ignores word *order*.
- **Task**: Integrate a lightweight N-gram model (like `kenlm-rs` or a simple tri-gram lookup). The "surprisal" of a token given previous context is the strongest AI signal.

### P2: Subword Tokenization

- **Current State**: `split_whitespace()`.
- **Task**: Integrate `tokenizers` crate with a BPE model (e.g., GPT-2 tokenizer). This normalizes punctuation and handles agglutinative languages correctly.

### P3: Performance (SIMD)

- **Current State**: Scalar loops.
- **Task**: Use `Simd<f64>` for entropy and dot product calculations to speed up batch processing of massive datasets.
