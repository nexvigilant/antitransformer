//! # HTTP Daemon
//!
//! Persistent Axum server for the antitransformer analysis pipeline.
//! Eliminates cold-start overhead for external service integrations.
//!
//! ## Routes (μ Mapping)
//! - `GET  /health`  → service liveness probe (∃ Existence)
//! - `POST /analyze` → single text analysis (σ Sequence)
//! - `POST /batch`   → batch text analysis (σ + Σ)
//!
//! ## Primitive Grounding
//! - μ Mapping: HTTP route → analysis function
//! - ς State: server lifecycle (Starting → Running → Shutting Down)
//! - ∂ Boundary: graceful shutdown via SIGTERM/ctrl_c
//! - λ Location: TCP address:port binding

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::pipeline::{self, AnalysisConfig};

/// Tier: T2-P (cross-domain primitive)
///
/// Shared application state — immutable config behind Arc.
#[derive(Debug, Clone)]
struct AppState {
    config: AnalysisConfig,
}

/// Health response body.
#[derive(Debug, Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    version: String,
}

/// Single-text analysis request.
#[derive(Debug, Deserialize)]
struct AnalyzeRequest {
    text: String,
}

/// Batch analysis request item.
#[derive(Debug, Deserialize)]
struct BatchItem {
    id: String,
    text: String,
}

/// Batch analysis response item.
#[derive(Debug, Serialize, Deserialize)]
struct BatchResponse {
    id: String,
    verdict: String,
    probability: f64,
    confidence: f64,
    features: pipeline::FeatureDetail,
}

/// Start the HTTP daemon on the specified port.
///
/// Binds to `127.0.0.1:{port}` (localhost only) and serves until ctrl_c/SIGTERM.
/// Set `BIND_ADDR=0.0.0.0` to listen on all interfaces.
///
/// # Errors
/// Returns error if port binding fails.
pub async fn serve(port: u16, config: AnalysisConfig) -> nexcore_error::Result<()> {
    use axum::http::{Method, header};

    let state = Arc::new(AppState { config });

    // Restricted CORS — localhost only (override via CORS_ORIGINS env var)
    let allowed_origins: Vec<_> = std::env::var("CORS_ORIGINS")
        .unwrap_or_else(|_| format!("http://localhost:{port},http://127.0.0.1:{port}"))
        .split(',')
        .filter_map(|s| s.trim().parse::<axum::http::HeaderValue>().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/health", get(health))
        .route("/analyze", post(analyze))
        .route("/batch", post(batch))
        .route("/openapi.json", get(openapi_spec))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    let bind_host = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1".into());
    let addr = format!("{bind_host}:{port}");
    info!(addr = %addr, "Starting antitransformer daemon");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!(addr = %addr, "Listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Daemon stopped");
    Ok(())
}

/// Graceful shutdown: await ctrl_c (∂ Boundary crossing).
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("Shutdown signal received");
}

/// `GET /health` — liveness probe (∃ Existence).
async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// `GET /openapi.json` — OpenAPI 3.0 spec (μ Mapping: code → documentation).
async fn openapi_spec() -> impl IntoResponse {
    let spec = serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "Antitransformer API",
            "description": "AI text detector via statistical fingerprints — Zipf deviation, entropy uniformity, burstiness dampening, perplexity consistency, TTR anomaly.",
            "version": env!("CARGO_PKG_VERSION").to_string(),
            "license": { "name": "Proprietary - NexVigilant" }
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Liveness probe",
                    "operationId": "health",
                    "responses": {
                        "200": {
                            "description": "Service is alive",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/HealthResponse" } } }
                        }
                    }
                }
            },
            "/analyze": {
                "post": {
                    "summary": "Analyze a single text for AI generation",
                    "operationId": "analyze",
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AnalyzeRequest" } } }
                    },
                    "responses": {
                        "200": {
                            "description": "Analysis result",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AnalysisResult" } } }
                        }
                    }
                }
            },
            "/batch": {
                "post": {
                    "summary": "Analyze a batch of texts",
                    "operationId": "batch",
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/BatchItem" } } } }
                    },
                    "responses": {
                        "200": {
                            "description": "Batch results",
                            "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/BatchResponse" } } } }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string", "example": "ok" },
                        "version": { "type": "string", "example": "0.1.0" }
                    }
                },
                "AnalyzeRequest": {
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": { "type": "string", "description": "Text to analyze for AI generation" }
                    }
                },
                "AnalysisResult": {
                    "type": "object",
                    "properties": {
                        "verdict": { "type": "string", "enum": ["human", "generated", "insufficient_data"] },
                        "probability": { "type": "number", "format": "double", "description": "0.0 = certainly human, 1.0 = certainly generated" },
                        "confidence": { "type": "number", "format": "double", "description": "Distance from decision boundary" },
                        "features": { "$ref": "#/components/schemas/FeatureDetail" }
                    }
                },
                "FeatureDetail": {
                    "type": "object",
                    "properties": {
                        "zipf_alpha": { "type": "number" },
                        "zipf_deviation": { "type": "number" },
                        "entropy_std": { "type": "number" },
                        "burstiness": { "type": "number" },
                        "perplexity_var": { "type": "number" },
                        "ttr": { "type": "number" },
                        "ttr_deviation": { "type": "number" },
                        "normalized": { "type": "array", "items": { "type": "number" }, "minItems": 5, "maxItems": 5 },
                        "beer_lambert": { "type": "number" },
                        "composite": { "type": "number" },
                        "hill_score": { "type": "number" }
                    }
                },
                "BatchItem": {
                    "type": "object",
                    "required": ["id", "text"],
                    "properties": {
                        "id": { "type": "string" },
                        "text": { "type": "string" }
                    }
                },
                "BatchResponse": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "verdict": { "type": "string" },
                        "probability": { "type": "number" },
                        "confidence": { "type": "number" },
                        "features": { "$ref": "#/components/schemas/FeatureDetail" }
                    }
                }
            }
        }
    });
    Json(spec)
}

/// `POST /analyze` — single text analysis (σ Sequence).
async fn analyze(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnalyzeRequest>,
) -> impl IntoResponse {
    let result = pipeline::analyze(&req.text, &state.config);
    Json(result)
}

/// `POST /batch` — batch text analysis (σ + Σ).
async fn batch(
    State(state): State<Arc<AppState>>,
    Json(items): Json<Vec<BatchItem>>,
) -> Result<impl IntoResponse, StatusCode> {
    if items.is_empty() {
        return Ok(Json(Vec::<BatchResponse>::new()));
    }

    let responses: Vec<BatchResponse> = items
        .iter()
        .map(|item| {
            let result = pipeline::analyze(&item.text, &state.config);
            BatchResponse {
                id: item.id.clone(),
                verdict: result.verdict,
                probability: result.probability,
                confidence: result.confidence,
                features: result.features,
            }
        })
        .collect();

    Ok(Json(responses))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_app() -> Router {
        let state = Arc::new(AppState {
            config: AnalysisConfig::default(),
        });

        Router::new()
            .route("/health", get(health))
            .route("/analyze", post(analyze))
            .route("/batch", post(batch))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = test_app();
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .expect("failed to build request");

        let response = app.oneshot(req).await.expect("failed to call service");
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read body");
        let health: HealthResponse =
            serde_json::from_slice(&body).expect("failed to parse health response");
        assert_eq!(health.status, "ok");
        assert_eq!(health.version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_analyze_endpoint() {
        let app = test_app();
        let body = serde_json::json!({
            "text": "The quick brown fox jumps over the lazy dog. \
                     This is a sufficiently long text to produce a real \
                     analysis result with statistical features extracted \
                     from the content for testing the HTTP endpoint."
        });

        let req = Request::builder()
            .method("POST")
            .uri("/analyze")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).expect("json")))
            .expect("failed to build request");

        let response = app.oneshot(req).await.expect("failed to call service");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read body");
        let result: pipeline::AnalysisResult =
            serde_json::from_slice(&bytes).expect("failed to parse analysis result");
        assert!(result.verdict == "human" || result.verdict == "generated");
        assert!(result.probability >= 0.0 && result.probability <= 1.0);
    }

    #[tokio::test]
    async fn test_analyze_short_text() {
        let app = test_app();
        let body = serde_json::json!({"text": "too short"});

        let req = Request::builder()
            .method("POST")
            .uri("/analyze")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).expect("json")))
            .expect("failed to build request");

        let response = app.oneshot(req).await.expect("failed to call service");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read body");
        let result: pipeline::AnalysisResult =
            serde_json::from_slice(&bytes).expect("failed to parse result");
        assert_eq!(result.verdict, "insufficient_data");
    }

    #[tokio::test]
    async fn test_batch_endpoint() {
        let app = test_app();
        let body = serde_json::json!([
            {
                "id": "t1",
                "text": "The quick brown fox jumps over the lazy dog. \
                         This is a sufficiently long text to produce a real \
                         analysis result with statistical features extracted."
            },
            {
                "id": "t2",
                "text": "short"
            }
        ]);

        let req = Request::builder()
            .method("POST")
            .uri("/batch")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).expect("json")))
            .expect("failed to build request");

        let response = app.oneshot(req).await.expect("failed to call service");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read body");
        let results: Vec<BatchResponse> =
            serde_json::from_slice(&bytes).expect("failed to parse batch response");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "t1");
        assert_eq!(results[1].id, "t2");
        assert_eq!(results[1].verdict, "insufficient_data");
    }

    #[tokio::test]
    async fn test_batch_empty() {
        let app = test_app();
        let body = serde_json::json!([]);

        let req = Request::builder()
            .method("POST")
            .uri("/batch")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).expect("json")))
            .expect("failed to build request");

        let response = app.oneshot(req).await.expect("failed to call service");
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read body");
        let results: Vec<BatchResponse> =
            serde_json::from_slice(&bytes).expect("failed to parse empty batch");
        assert!(results.is_empty());
    }
}
