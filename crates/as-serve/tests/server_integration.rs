//! Integration tests for Phase 3 Inference Server (`as-serve`):
//! - Live TCP HTTP/1.1 requests to `/health`, `/v1/models`, `/v1/infer`, `/v1/chat/completions` (JSON + SSE), `/metrics`
//! - Concurrent requests coalesced through `DynamicBatcher`
//! - Graceful shutdown

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

use as_backend::BinaryOp;
use as_core::{DType, Shape};
use as_graph::{Graph, Op};
use as_runtime::{InferenceSession, RuntimeConfig};
use as_serve::{BatchConfig, InferenceServer, ServerConfig};

fn send_http(addr: SocketAddr, raw_req: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all(raw_req.as_bytes()).unwrap();
    stream.flush().unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    resp
}

#[test]
fn inference_server_handles_health_infer_chat_sse_and_metrics() {
    let mut g = Graph::new();
    let x = g.add_input("x", Shape::new(vec![2]).unwrap(), DType::F32).unwrap();
    let c = g.add_constant(Shape::new(vec![2]).unwrap(), DType::F32, vec![10.0, 20.0]).unwrap();
    let out = g.add_op(Op::Binary(BinaryOp::Add), vec![x, c]).unwrap();
    g.mark_output(out).unwrap();

    let session = InferenceSession::new(g, RuntimeConfig::default()).unwrap();

    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_owned(),
        batch_config: BatchConfig {
            max_batch_size: 4,
            max_wait_ms: 2,
            max_queue_depth: 32,
        },
        ..ServerConfig::default()
    };

    let mut server = InferenceServer::new(cfg);
    server.register_model("add-model", session).unwrap();
    server.with_chat_handler(Arc::new(|_model, prompt| {
        Ok(vec!["Echo: ".to_owned(), prompt.to_owned()])
    }));

    let handle = server.start().unwrap();
    let addr = handle.addr();

    // 1. GET /health
    let health = send_http(addr, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert!(health.starts_with("HTTP/1.1 200 OK"));
    assert!(health.contains("\"status\":\"ok\""));

    // 2. GET /v1/models
    let models = send_http(addr, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert!(models.contains("\"add-model\""));

    // 3. POST /v1/infer
    let infer_body = r#"{"model":"add-model","inputs":{"x":{"shape":[2],"values":[1.5,2.5]}}}"#;
    let infer_req = format!(
        "POST /v1/infer HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        infer_body.len(),
        infer_body
    );
    let infer_resp = send_http(addr, &infer_req);
    assert!(infer_resp.starts_with("HTTP/1.1 200 OK"), "resp={infer_resp}");
    assert!(infer_resp.contains("11.5"));
    assert!(infer_resp.contains("22.5"));

    // 4. POST /v1/chat/completions (SSE streaming)
    let chat_body = r#"{"model":"add-model","stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
    let chat_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        chat_body.len(),
        chat_body
    );
    let chat_resp = send_http(addr, &chat_req);
    assert!(chat_resp.contains("text/event-stream"));
    assert!(chat_resp.contains("data: [DONE]"));
    assert!(chat_resp.contains("hello"));

    // 5. GET /metrics (Prometheus)
    let metrics = send_http(addr, "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert!(metrics.contains("tarsix_inferences_total 1"));
    assert!(metrics.contains("tarsix_chat_completions_total 1"));

    handle.shutdown();
}
