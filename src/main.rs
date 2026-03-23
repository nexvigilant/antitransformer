//! # Antitransformer CLI
//!
//! Reads JSON text samples from stdin (batch mode) or serves HTTP (daemon mode).
//!
//! ## Subcommands
//! - `batch` (default) — stdin JSON → stdout JSON
//! - `daemon --port 3100` — persistent HTTP server
//!
//! ## Input Format (batch)
//! ```json
//! [{"id": "h1", "text": "...", "label": "human"}]
//! ```
//!
//! ## Output Format (batch)
//! ```json
//! [{"id": "h1", "verdict": "human", "probability": 0.12, "confidence": 0.76, "features": {...}}]
//! ```

use clap::{Parser, Subcommand};
use nexcore_error::{Context, Result};
use std::io::{self, BufRead, Write};
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use antitransformer::pipeline::{self, AnalysisConfig, InputSample};

/// Antitransformer: AI text detector via statistical fingerprints.
#[derive(Parser, Debug)]
#[command(name = "antitransformer")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Detect AI-generated text through statistical fingerprints")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Batch mode: read JSON from stdin, write verdicts to stdout (default).
    Batch {
        /// Verbose output (-v, -vv, -vvv).
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,

        /// Dry run — validate configuration only.
        #[arg(long)]
        dry_run: bool,

        /// Custom decision threshold (default: 0.5).
        #[arg(long, default_value = "0.5")]
        threshold: f64,

        /// Entropy window size (default: 50).
        #[arg(long, default_value = "50")]
        window_size: usize,
    },

    /// Daemon mode: persistent HTTP server.
    Daemon {
        /// Port to listen on.
        #[arg(long, default_value = "3100")]
        port: u16,

        /// Verbose output (-v, -vv, -vvv).
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,

        /// Custom decision threshold (default: 0.5).
        #[arg(long, default_value = "0.5")]
        threshold: f64,

        /// Entropy window size (default: 50).
        #[arg(long, default_value = "50")]
        window_size: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Default to Batch if no subcommand given (backward compat)
    let command = cli.command.unwrap_or(Command::Batch {
        verbose: 0,
        dry_run: false,
        threshold: 0.5,
        window_size: 50,
    });

    match command {
        Command::Batch {
            verbose,
            dry_run,
            threshold,
            window_size,
        } => {
            init_tracing(verbose);
            info!(
                pipeline = "antitransformer",
                version = env!("CARGO_PKG_VERSION"),
                mode = "batch",
                "Starting pipeline"
            );

            if dry_run {
                info!("Dry run — configuration valid");
                return Ok(());
            }

            let config = AnalysisConfig {
                threshold,
                window_size,
            };
            run_batch(config).await
        }
        Command::Daemon {
            port,
            verbose,
            threshold,
            window_size,
        } => {
            init_tracing(verbose);
            info!(
                pipeline = "antitransformer",
                version = env!("CARGO_PKG_VERSION"),
                mode = "daemon",
                port,
                "Starting daemon"
            );

            let config = AnalysisConfig {
                threshold,
                window_size,
            };
            antitransformer::daemon::serve(port, config).await
        }
    }
}

fn init_tracing(verbose: u8) {
    let level = match verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(io::stderr),
        )
        .with(tracing_subscriber::filter::LevelFilter::from_level(level))
        .init();
}

async fn run_batch(config: AnalysisConfig) -> Result<()> {
    // Read JSON from stdin
    let stdin = io::stdin();
    let lines: Vec<String> = stdin
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .collect();

    let json_str = if lines.len() == 1 && lines[0].trim().starts_with('[') {
        lines[0].clone()
    } else {
        format!("[{}]", lines.join(","))
    };

    let samples: Vec<InputSample> =
        serde_json::from_str(&json_str).context("Failed to parse JSON input")?;

    info!(samples = samples.len(), "Ingested samples");

    let (verdicts, stats) = pipeline::analyze_batch(&samples, &config);

    // Write JSON to stdout
    let output = serde_json::to_string_pretty(&verdicts).context("Failed to serialize output")?;
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{output}").context("Failed to write output")?;

    info!(
        processed = stats.records_processed,
        human = stats.human_count,
        generated = stats.generated_count,
        duration = stats.duration_secs,
        "Pipeline completed"
    );

    if stats.labeled_count > 0 {
        let accuracy = stats.correct_count as f64 / stats.labeled_count as f64;
        eprintln!(
            "Accuracy: {}/{} ({:.1}%)",
            stats.correct_count,
            stats.labeled_count,
            accuracy * 100.0
        );
    }

    Ok(())
}
