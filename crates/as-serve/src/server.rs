//! Multi-model HTTP/1.1 + SSE Inference Server with Prometheus telemetry.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use as_core::{DType, Error, Result};
use as_io::{JsonValue, parse_json};
use as_runtime::InferenceSession;
use as_tensor::Tensor;

use crate::batcher::{BatchConfig, DynamicBatcher};
use crate::http::{
    HttpRequest, HttpResponse, read_http_request, write_sse_data, write_sse_headers,
};

/// Configuration for [`InferenceServer`].
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// TCP bind address (e.g. `"127.0.0.1:8080"` or `"127.0.0.1:0"` for ephemeral port).
    pub bind_addr: String,
    /// Maximum concurrent in-flight HTTP connections.
    pub max_concurrent_requests: usize,
    /// Maximum permitted HTTP header bytes.
    pub max_header_bytes: usize,
    /// Maximum permitted HTTP request body bytes.
    pub max_body_bytes: usize,
    /// Socket read timeout in milliseconds.
    pub read_timeout_ms: u64,
    /// Dynamic batching configuration applied per registered model.
    pub batch_config: BatchConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".to_owned(),
            max_concurrent_requests: 64,
            max_header_bytes: 16 * 1024,
            max_body_bytes: 8 * 1024 * 1024,
            read_timeout_ms: 5_000,
            batch_config: BatchConfig::default(),
        }
    }
}

/// Atomic server-wide request and inference telemetry (exported at `GET /metrics`).
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Total HTTP requests received.
    pub requests_total: AtomicU64,
    /// Total HTTP 4xx/5xx error responses.
    pub errors_total: AtomicU64,
    /// Total `/v1/infer` model executions completed.
    pub inferences_total: AtomicU64,
    /// Total `/v1/chat/completions` requests completed.
    pub chat_completions_total: AtomicU64,
    /// Currently active in-flight connections.
    pub inflight_connections: AtomicUsize,
}

impl ServerMetrics {
    /// Formats metrics in Prometheus text exposition format.
    pub fn to_prometheus(&self, model_count: usize) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# TYPE tarsix_requests_total counter");
        let _ = writeln!(
            out,
            "tarsix_requests_total {}",
            self.requests_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE tarsix_errors_total counter");
        let _ = writeln!(
            out,
            "tarsix_errors_total {}",
            self.errors_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE tarsix_inferences_total counter");
        let _ = writeln!(
            out,
            "tarsix_inferences_total {}",
            self.inferences_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE tarsix_chat_completions_total counter");
        let _ = writeln!(
            out,
            "tarsix_chat_completions_total {}",
            self.chat_completions_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE tarsix_loaded_models gauge");
        let _ = writeln!(out, "tarsix_loaded_models {model_count}");
        out
    }
}

/// Chat completion handler returning token chunks for a `(model, prompt)` pair.
pub type ChatHandler = Arc<dyn Fn(&str, &str) -> Result<Vec<String>> + Send + Sync>;

struct ServerState {
    config: ServerConfig,
    models: BTreeMap<String, Arc<DynamicBatcher>>,
    chat_handler: Option<ChatHandler>,
    metrics: Arc<ServerMetrics>,
}

/// Zero-dependency HTTP/1.1 + SSE Inference Server builder.
pub struct InferenceServer {
    config: ServerConfig,
    models: BTreeMap<String, Arc<DynamicBatcher>>,
    chat_handler: Option<ChatHandler>,
}

impl InferenceServer {
    /// Creates a new [`InferenceServer`] with `config`.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            models: BTreeMap::new(),
            chat_handler: None,
        }
    }

    /// Registers an [`InferenceSession`] under `model_name`, backed by its own [`DynamicBatcher`].
    pub fn register_model(
        &mut self,
        model_name: impl Into<String>,
        session: InferenceSession,
    ) -> Result<&mut Self> {
        let name = model_name.into();
        if name.trim().is_empty() {
            return Err(Error::InvalidArgument {
                context: "InferenceServer::register_model",
                reason: "model_name cannot be empty".to_owned(),
            });
        }
        let batcher = DynamicBatcher::start(session, self.config.batch_config)?;
        self.models.insert(name, Arc::new(batcher));
        Ok(self)
    }

    /// Registers a token-streaming handler for `POST /v1/chat/completions`.
    pub fn with_chat_handler(&mut self, handler: ChatHandler) -> &mut Self {
        self.chat_handler = Some(handler);
        self
    }

    /// Binds the TCP listener and spawns the server accept loop, returning a [`ServerHandle`].
    pub fn start(self) -> Result<ServerHandle> {
        let listener =
            TcpListener::bind(&self.config.bind_addr).map_err(|e| Error::InvalidArgument {
                context: "InferenceServer::start",
                reason: format!("failed to bind {}: {e}", self.config.bind_addr),
            })?;
        let local_addr = listener.local_addr().map_err(|e| Error::InvalidArgument {
            context: "InferenceServer::start",
            reason: format!("failed reading local_addr: {e}"),
        })?;
        listener
            .set_nonblocking(false)
            .map_err(|e| Error::InvalidArgument {
                context: "InferenceServer::start",
                reason: format!("failed configuring listener: {e}"),
            })?;

        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(ServerMetrics::default());
        let state = Arc::new(ServerState {
            config: self.config,
            models: self.models,
            chat_handler: self.chat_handler,
            metrics: Arc::clone(&metrics),
        });

        let flag_clone = Arc::clone(&shutdown_flag);
        let accept_thread = thread::spawn(move || {
            for incoming in listener.incoming() {
                if flag_clone.load(Ordering::Acquire) {
                    break;
                }
                let Ok(stream) = incoming else {
                    continue;
                };
                if flag_clone.load(Ordering::Acquire) {
                    break;
                }
                let st = Arc::clone(&state);
                let active = st
                    .metrics
                    .inflight_connections
                    .fetch_add(1, Ordering::AcqRel);
                if active >= st.config.max_concurrent_requests {
                    st.metrics
                        .inflight_connections
                        .fetch_sub(1, Ordering::AcqRel);
                    st.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                    let mut s = stream;
                    let _ = HttpResponse::json(429, "{\"error\":\"too many concurrent requests\"}")
                        .write_to(&mut s);
                    continue;
                }
                thread::spawn(move || {
                    handle_connection(stream, &st);
                    st.metrics
                        .inflight_connections
                        .fetch_sub(1, Ordering::AcqRel);
                });
            }
        });

        Ok(ServerHandle {
            addr: local_addr,
            shutdown_flag,
            metrics,
            accept_thread: Some(accept_thread),
        })
    }
}

/// Running handle to a started [`InferenceServer`].
pub struct ServerHandle {
    addr: SocketAddr,
    shutdown_flag: Arc<AtomicBool>,
    metrics: Arc<ServerMetrics>,
    accept_thread: Option<JoinHandle<()>>,
}

impl ServerHandle {
    /// Returns the actual bound [`SocketAddr`] (useful when binding `:0`).
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Returns shared [`ServerMetrics`].
    pub fn metrics(&self) -> &Arc<ServerMetrics> {
        &self.metrics
    }

    /// Signals graceful shutdown and joins the accept thread.
    pub fn shutdown(mut self) {
        self.shutdown_flag.store(true, Ordering::Release);
        // Unblock `listener.incoming()` cleanly with a loopback connection
        let _ = TcpStream::connect(self.addr);
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if !self.shutdown_flag.swap(true, Ordering::AcqRel) {
            let _ = TcpStream::connect(self.addr);
        }
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
    }
}

fn handle_connection(mut stream: TcpStream, state: &ServerState) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(state.config.read_timeout_ms)));
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    let req = match read_http_request(
        &stream,
        state.config.max_header_bytes,
        state.config.max_body_bytes,
    ) {
        Ok(r) => r,
        Err(Error::AllocationLimitExceeded { .. }) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            let _ = HttpResponse::json(413, "{\"error\":\"payload too large\"}").write_to(&mut stream);
            return;
        }
        Err(e) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            let _ = HttpResponse::json(400, format!("{{\"error\":\"{e}\"}}")).write_to(&mut stream);
            return;
        }
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            let body = format!("{{\"status\":\"ok\",\"models\":{}}}", state.models.len());
            let _ = HttpResponse::json(200, body).write_to(&mut stream);
        }
        ("GET", "/metrics") => {
            let body = state.metrics.to_prometheus(state.models.len());
            let _ = HttpResponse::text(200, body).write_to(&mut stream);
        }
        ("GET", "/v1/models") => {
            let mut body = String::from("{\"object\":\"list\",\"data\":[");
            for (idx, name) in state.models.keys().enumerate() {
                if idx > 0 {
                    body.push(',');
                }
                let _ = write!(
                    body,
                    "{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"tarsix\"}}",
                    escape_json(name)
                );
            }
            body.push_str("]}");
            let _ = HttpResponse::json(200, body).write_to(&mut stream);
        }
        ("POST", "/v1/infer") => {
            let resp = handle_infer(&req, state);
            if resp.status >= 400 {
                state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            }
            let _ = resp.write_to(&mut stream);
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completions(&req, state, &mut stream);
        }
        _ => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            let _ = HttpResponse::json(404, "{\"error\":\"not found\"}").write_to(&mut stream);
        }
    }
}

fn handle_infer(req: &HttpRequest, state: &ServerState) -> HttpResponse {
    let Ok(body_str) = core::str::from_utf8(&req.body) else {
        return HttpResponse::json(400, "{\"error\":\"request body must be UTF-8 JSON\"}");
    };
    let json = match parse_json(body_str) {
        Ok(v) => v,
        Err(e) => return HttpResponse::json(400, format!("{{\"error\":\"{e}\"}}")),
    };
    let Some(model_name) = json.get("model").and_then(JsonValue::as_str) else {
        return HttpResponse::json(400, "{\"error\":\"missing string field `model`\"}");
    };
    let Some(batcher) = state.models.get(model_name) else {
        return HttpResponse::json(404, format!("{{\"error\":\"unknown model `{model_name}`\"}}"));
    };
    let Some(inputs_obj) = json.get("inputs").and_then(JsonValue::as_object) else {
        return HttpResponse::json(400, "{\"error\":\"missing object field `inputs`\"}");
    };

    let mut named_inputs = BTreeMap::new();
    for (name, spec) in inputs_obj {
        let Some(shape_arr) = spec.get("shape").and_then(JsonValue::as_array) else {
            return HttpResponse::json(400, format!("{{\"error\":\"input `{name}` missing `shape`\"}}"));
        };
        let Some(vals_arr) = spec.get("values").and_then(JsonValue::as_array) else {
            return HttpResponse::json(400, format!("{{\"error\":\"input `{name}` missing `values`\"}}"));
        };
        let mut dims = Vec::with_capacity(shape_arr.len());
        for d in shape_arr {
            let Some(u) = d.as_u64() else {
                return HttpResponse::json(400, "{\"error\":\"shape dimensions must be unsigned integers\"}");
            };
            dims.push(u as usize);
        }
        let mut vals = Vec::with_capacity(vals_arr.len());
        for v in vals_arr {
            let Some(f) = v.as_f64() else {
                return HttpResponse::json(400, "{\"error\":\"tensor values must be numbers\"}");
            };
            vals.push(f as f32);
        }
        match Tensor::from_f32(&vals, dims) {
            Ok(t) => {
                named_inputs.insert(name.clone(), t);
            }
            Err(e) => return HttpResponse::json(400, format!("{{\"error\":\"{e}\"}}")),
        }
    }

    match batcher.infer(named_inputs) {
        Ok(outputs) => {
            state.metrics.inferences_total.fetch_add(1, Ordering::Relaxed);
            let mut body = format!("{{\"model\":\"{}\",\"outputs\":{{", escape_json(model_name));
            let mut sorted: Vec<_> = outputs.into_iter().collect();
            sorted.sort_by_key(|(id, _)| id.0);
            for (idx, (node_id, tensor)) in sorted.into_iter().enumerate() {
                if idx > 0 {
                    body.push(',');
                }
                let f64_vals = tensor
                    .to_dtype(DType::F64)
                    .and_then(|t| t.to_f64_vec())
                    .unwrap_or_default();
                let _ = write!(
                    body,
                    "\"node_{}\":{{\"shape\":{:?},\"values\":{:?}}}",
                    node_id.0,
                    tensor.dims(),
                    f64_vals
                );
            }
            body.push_str("}}");
            HttpResponse::json(200, body)
        }
        Err(e) => HttpResponse::json(500, format!("{{\"error\":\"{e}\"}}")),
    }
}

fn handle_chat_completions(req: &HttpRequest, state: &ServerState, stream: &mut TcpStream) {
    let Ok(body_str) = core::str::from_utf8(&req.body) else {
        state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
        let _ = HttpResponse::json(400, "{\"error\":\"invalid UTF-8\"}").write_to(stream);
        return;
    };
    let json = match parse_json(body_str) {
        Ok(v) => v,
        Err(e) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            let _ = HttpResponse::json(400, format!("{{\"error\":\"{e}\"}}")).write_to(stream);
            return;
        }
    };
    let model = json
        .get("model")
        .and_then(JsonValue::as_str)
        .unwrap_or("tarsix-default");
    let is_stream = json
        .get("stream")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let prompt = json
        .get("messages")
        .and_then(JsonValue::as_array)
        .and_then(|arr| arr.last())
        .and_then(|m| m.get("content"))
        .and_then(JsonValue::as_str)
        .unwrap_or("");

    let tokens = if let Some(handler) = &state.chat_handler {
        match handler(model, prompt) {
            Ok(t) => t,
            Err(e) => {
                state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                let _ = HttpResponse::json(500, format!("{{\"error\":\"{e}\"}}")).write_to(stream);
                return;
            }
        }
    } else {
        vec!["Tarsix ".to_owned(), "inference ".to_owned(), "ready.".to_owned()]
    };

    state
        .metrics
        .chat_completions_total
        .fetch_add(1, Ordering::Relaxed);

    if is_stream {
        if write_sse_headers(stream).is_err() {
            return;
        }
        for tok in &tokens {
            let chunk = format!(
                "{{\"object\":\"chat.completion.chunk\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}}}}]}}",
                escape_json(model),
                escape_json(tok)
            );
            if write_sse_data(stream, &chunk).is_err() {
                return;
            }
        }
        let _ = write_sse_data(stream, "[DONE]");
    } else {
        let full: String = tokens.concat();
        let body = format!(
            "{{\"object\":\"chat.completion\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"stop\"}}]}}",
            escape_json(model),
            escape_json(&full)
        );
        let _ = HttpResponse::json(200, body).write_to(stream);
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}
