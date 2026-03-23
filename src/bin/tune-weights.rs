//! # Weight Tuner
//!
//! Calibrates the 5 feature weights using Logistic Regression (Gradient Descent)
//! on a labeled dataset.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --bin tune-weights -- < dataset.jsonl
//! ```
//!
//! ## Algorithm
//!
//! - Reads labeled samples (human/ai)
//! - Extracts normalized features [0..1]
//! - Trains a Logistic Regression model: P(ai) = sigmoid(W*X + b)
//! - Minimizes Log-Loss via Gradient Descent
//! - Outputs optimal weights and bias as Rust code constants

use antitransformer::pipeline::{self, AnalysisConfig, InputSample};
use std::io::{self, BufRead};

const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 2000;

#[derive(Debug)]
struct TrainingSample {
    features: [f64; 5],
    label: f64, // 1.0 = AI, 0.0 = Human
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn predict(features: &[f64; 5], weights: &[f64; 5], bias: f64) -> f64 {
    let z: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum::<f64>()
        + bias;
    sigmoid(z)
}

fn main() -> nexcore_error::Result<()> {
    // 1. Load Data
    eprintln!(">>> Loading dataset from stdin...");
    let stdin = io::stdin();
    let mut samples = Vec::new();
    let config = AnalysisConfig::default(); // Use default window sizes

    let mut count_ai = 0;
    let mut count_human = 0;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let input: InputSample = serde_json::from_str(&line)?;

        let label_val = match input.label.as_deref() {
            Some(l) if l.eq_ignore_ascii_case("generated") || l.eq_ignore_ascii_case("ai") => {
                count_ai += 1;
                1.0
            }
            Some(l) if l.eq_ignore_ascii_case("human") => {
                count_human += 1;
                0.0
            }
            _ => {
                continue;
            } // Skip unlabeled or unknown labels
        };

        // Extract normalized features [0..1]
        let result = pipeline::analyze(&input.text, &config);

        // Skip short texts that resulted in insufficient_data (all 0s)
        if result.verdict == "insufficient_data" {
            continue;
        }

        samples.push(TrainingSample {
            features: result.features.normalized,
            label: label_val,
        });
    }

    eprintln!(
        "Loaded {} samples ({} AI, {} Human)",
        samples.len(),
        count_ai,
        count_human
    );
    if samples.is_empty() {
        eprintln!("No labeled samples found! Ensure input is JSONL with 'label': 'human'/'ai'");
        return Ok(());
    }

    // 2. Train (Gradient Descent)
    eprintln!(
        ">>> Training model (LR={}, Epochs={})...",
        LEARNING_RATE, EPOCHS
    );

    // Initialize weights randomly or with current defaults
    // Current defaults: [2.5, 2.0, 1.8, 2.2, 1.5]
    // Let's start from 0.0 to be unbiased, or small randoms.
    // Actually, let's start with 1.0 to encourage feature usage.
    let mut weights = [1.0; 5];
    let mut bias = 0.0;

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0;
        let mut dw = [0.0; 5];
        let mut db = 0.0;
        let n = samples.len() as f64;

        for sample in &samples {
            let pred = predict(&sample.features, &weights, bias);
            let error = pred - sample.label; // derivative of loss w.r.t activation functions

            // Log-Loss accumulation
            // loss = -[y*log(p) + (1-y)*log(1-p)]
            let epsilon = 1e-15;
            let p_safe = pred.clamp(epsilon, 1.0 - epsilon);
            total_loss -= sample.label * p_safe.ln() + (1.0 - sample.label) * (1.0 - p_safe).ln();

            // Gradients
            for i in 0..5 {
                dw[i] += error * sample.features[i]; // dLoss/dw_i = error * x_i
            }
            db += error; // dLoss/db = error
        }

        // Update weights
        for i in 0..5 {
            weights[i] -= LEARNING_RATE * (dw[i] / n);
        }
        bias -= LEARNING_RATE * (db / n);

        if epoch % 100 == 0 || epoch == 1 {
            eprintln!(
                "Epoch {:4}: Loss = {:.4} | Acc = {:.1}%",
                epoch,
                total_loss / n,
                evaluate_accuracy(&samples, &weights, bias) * 100.0
            );
        }
    }

    // 3. Output Results
    eprintln!("\n=== Calibration Complete ===");
    eprintln!(
        "Final Accuracy: {:.1}%",
        evaluate_accuracy(&samples, &weights, bias) * 100.0
    );
    eprintln!("Bias: {:.4}", bias);
    eprintln!("Weights:");
    let names = [
        "Zipf Deviation",
        "Entropy Std",
        "Burstiness",
        "Perplexity Var",
        "TTR Deviation",
    ];
    for (i, w) in weights.iter().enumerate() {
        eprintln!("  {}: {:.4}", names[i], w);
    }

    println!("\n// Paste WEIGHTS into src/aggregation.rs to replace the existing constant.");
    println!("// NOTE: The production pipeline uses Beer-Lambert + Hill + Arrhenius, not a");
    println!("// logistic regression with a bias term. BIAS is informational only — to use");
    println!("// it, you would need to add a bias term to the aggregation::aggregate() function.");
    println!("pub const WEIGHTS: [f64; 5] = [");
    println!("    {:.4}, // Zipf deviation", weights[0]);
    println!("    {:.4}, // Entropy std", weights[1]);
    println!("    {:.4}, // Burstiness", weights[2]);
    println!("    {:.4}, // Perplexity variance", weights[3]);
    println!("    {:.4}, // TTR deviation", weights[4]);
    println!("];");
    println!(
        "// pub const BIAS: f64 = {:.4}; // not used by current pipeline",
        bias
    );

    Ok(())
}

fn evaluate_accuracy(samples: &[TrainingSample], weights: &[f64; 5], bias: f64) -> f64 {
    let mut correct = 0;
    for sample in samples {
        let p = predict(&sample.features, weights, bias);
        let predicted_label = if p >= 0.5 { 1.0 } else { 0.0 };
        if (predicted_label - sample.label).abs() < 1e-5 {
            correct += 1;
        }
    }
    (correct as f64) / (samples.len() as f64)
}
