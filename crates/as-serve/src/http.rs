//! Hardened zero-dependency HTTP/1.1 parser, response writer, and SSE streaming utilities.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use as_core::{Error, Result};

/// Parsed HTTP/1.1 request with enforced header and body byte ceilings.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP verb (`GET`, `POST`, etc.).
    pub method: String,
    /// Request target path (without query string).
    pub path: String,
    /// Lowercased HTTP headers.
    pub headers: BTreeMap<String, String>,
    /// Raw UTF-8 or binary request body.
    pub body: Vec<u8>,
}

/// HTTP/1.1 response ready to serialize onto a [`TcpStream`].
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code (e.g. `200`, `400`, `404`, `413`, `500`).
    pub status: u16,
    /// Content-Type MIME header.
    pub content_type: &'static str,
    /// Response body bytes.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Builds a JSON response with status code `status`.
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json; charset=utf-8",
            body: body.into().into_bytes(),
        }
    }

    /// Builds a plain-text response (used for Prometheus `/metrics`).
    pub fn text(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "text/plain; version=0.0.4; charset=utf-8",
            body: body.into().into_bytes(),
        }
    }

    /// Writes the formatted HTTP/1.1 response to `stream`.
    pub fn write_to(&self, stream: &mut TcpStream) -> Result<()> {
        let reason = status_reason(self.status);
        let header = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.status,
            reason,
            self.content_type,
            self.body.len()
        );
        stream
            .write_all(header.as_bytes())
            .and_then(|()| stream.write_all(&self.body))
            .and_then(|()| stream.flush())
            .map_err(|e| Error::InvalidArgument {
                context: "HttpResponse::write_to",
                reason: format!("socket write failed: {e}"),
            })
    }
}

/// Reads and validates an HTTP/1.1 request from `stream`, rejecting oversized headers/bodies.
pub fn read_http_request(
    stream: &TcpStream,
    max_header_bytes: usize,
    max_body_bytes: usize,
) -> Result<HttpRequest> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    let n = reader
        .read_line(&mut request_line)
        .map_err(|e| Error::InvalidArgument {
            context: "read_http_request",
            reason: format!("failed reading request line: {e}"),
        })?;
    if n == 0 {
        return Err(Error::InvalidArgument {
            context: "read_http_request",
            reason: "connection closed before request line".to_owned(),
        });
    }

    let mut total_header_bytes = n;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| Error::InvalidArgument {
            context: "read_http_request",
            reason: "missing HTTP method".to_owned(),
        })?
        .to_owned();
    let raw_target = parts.next().ok_or_else(|| Error::InvalidArgument {
        context: "read_http_request",
        reason: "missing HTTP target path".to_owned(),
    })?;
    let path = raw_target
        .split('?')
        .next()
        .unwrap_or(raw_target)
        .to_owned();

    let mut headers = BTreeMap::new();
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|e| Error::InvalidArgument {
                context: "read_http_request",
                reason: format!("failed reading header line: {e}"),
            })?;
        if read == 0 {
            break;
        }
        total_header_bytes = total_header_bytes.saturating_add(read);
        if total_header_bytes > max_header_bytes {
            return Err(Error::AllocationLimitExceeded {
                requested_bytes: total_header_bytes,
                limit_bytes: max_header_bytes,
            });
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
        }
    }

    let content_len: usize = headers
        .get("content-length")
        .map_or(Ok(0), |s| s.parse::<usize>())
        .map_err(|_| Error::InvalidArgument {
            context: "read_http_request",
            reason: "invalid Content-Length header".to_owned(),
        })?;

    if content_len > max_body_bytes {
        return Err(Error::AllocationLimitExceeded {
            requested_bytes: content_len,
            limit_bytes: max_body_bytes,
        });
    }

    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| Error::InvalidArgument {
                context: "read_http_request",
                reason: format!("failed reading request body: {e}"),
            })?;
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

/// Writes HTTP/1.1 Server-Sent Events (`text/event-stream`) headers to `stream`.
pub fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream
        .write_all(hdr.as_bytes())
        .and_then(|()| stream.flush())
        .map_err(|e| Error::InvalidArgument {
            context: "write_sse_headers",
            reason: format!("failed writing SSE headers: {e}"),
        })
}

/// Writes a single SSE `data: <payload>\n\n` event to `stream`.
pub fn write_sse_data(stream: &mut TcpStream, data: &str) -> Result<()> {
    let frame = format!("data: {data}\n\n");
    stream
        .write_all(frame.as_bytes())
        .and_then(|()| stream.flush())
        .map_err(|e| Error::InvalidArgument {
            context: "write_sse_data",
            reason: format!("failed writing SSE event: {e}"),
        })
}

const fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Status",
    }
}
