//! Minimal, recursion-bounded, zero-dependency JSON parser for untrusted model headers.

use std::collections::BTreeMap;

use as_core::{Error, Result};

/// Maximum JSON nesting depth to prevent stack exhaustion on adversarial inputs.
pub const MAX_JSON_DEPTH: usize = 32;

/// Maximum JSON input size (16 MiB).
pub const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

/// Parsed JSON value tree.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number (stored as `f64`).
    Number(f64),
    /// JSON UTF-8 string.
    String(String),
    /// JSON array.
    Array(Vec<JsonValue>),
    /// JSON object with deterministic key ordering.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Borrows this value as a JSON object map.
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Borrows this value as a JSON array slice.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Borrows this value as a string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns this value as an `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns this value as a non-negative integer `usize` if exact.
    pub fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        if !n.is_finite() || n < 0.0 || n.fract() != 0.0 || n > (usize::MAX as f64) {
            return None;
        }
        Some(n as usize)
    }
}

/// Parses a UTF-8 JSON string into a [`JsonValue`] with strict depth and size bounds.
pub fn parse_json(input: &str) -> Result<JsonValue> {
    if input.len() > MAX_JSON_BYTES {
        return Err(Error::InvalidArgument {
            context: "parse_json",
            reason: format!(
                "JSON payload {} bytes exceeds ceiling {MAX_JSON_BYTES}",
                input.len()
            ),
        });
    }
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    parser.skip_ws();
    let val = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(Error::InvalidArgument {
            context: "parse_json",
            reason: "unexpected trailing tokens in JSON".to_owned(),
        });
    }
    Ok(val)
}

/// Escapes a string for safe JSON emission.
pub fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue> {
        if depth > MAX_JSON_DEPTH {
            return Err(Error::InvalidArgument {
                context: "parse_json",
                reason: format!("JSON nesting depth exceeds limit {MAX_JSON_DEPTH}"),
            });
        }
        self.skip_ws();
        let b = self.peek().ok_or_else(|| Error::InvalidArgument {
            context: "parse_json",
            reason: "unexpected EOF while parsing JSON value".to_owned(),
        })?;
        match b {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => Ok(JsonValue::String(self.parse_string()?)),
            b't' => self.expect_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.expect_literal(b"false", JsonValue::Bool(false)),
            b'n' => self.expect_literal(b"null", JsonValue::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(Error::InvalidArgument {
                context: "parse_json",
                reason: format!("unexpected byte 0x{other:02X} at position {}", self.pos),
            }),
        }
    }

    fn expect_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue> {
        let end = self.pos.checked_add(lit.len()).ok_or(Error::ArithmeticOverflow {
            context: "parse_json literal",
        })?;
        if self.bytes.get(self.pos..end) == Some(lit) {
            self.pos = end;
            Ok(val)
        } else {
            Err(Error::InvalidArgument {
                context: "parse_json",
                reason: "invalid JSON literal".to_owned(),
            })
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue> {
        self.next(); // consume '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.next();
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next() != Some(b':') {
                return Err(Error::InvalidArgument {
                    context: "parse_json",
                    reason: "expected ':' after object key".to_owned(),
                });
            }
            let val = self.parse_value(depth)?;
            map.insert(key, val);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => {
                    return Err(Error::InvalidArgument {
                        context: "parse_json",
                        reason: "expected ',' or '}' in JSON object".to_owned(),
                    });
                }
            }
        }
        Ok(JsonValue::Object(map))
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue> {
        self.next(); // consume '['
        self.skip_ws();
        let mut arr = Vec::new();
        if self.peek() == Some(b']') {
            self.next();
            return Ok(JsonValue::Array(arr));
        }
        loop {
            let val = self.parse_value(depth)?;
            arr.push(val);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => {
                    return Err(Error::InvalidArgument {
                        context: "parse_json",
                        reason: "expected ',' or ']' in JSON array".to_owned(),
                    });
                }
            }
        }
        Ok(JsonValue::Array(arr))
    }

    fn parse_string(&mut self) -> Result<String> {
        if self.next() != Some(b'"') {
            return Err(Error::InvalidArgument {
                context: "parse_json",
                reason: "expected '\"' to start JSON string".to_owned(),
            });
        }
        let mut out = String::new();
        while let Some(b) = self.next() {
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.next().ok_or_else(|| Error::InvalidArgument {
                        context: "parse_json",
                        reason: "unterminated escape sequence".to_owned(),
                    })?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex_bytes = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| Error::InvalidArgument {
                                    context: "parse_json",
                                    reason: "incomplete \\uXXXX escape".to_owned(),
                                })?;
                            self.pos += 4;
                            let hex_str = core::str::from_utf8(hex_bytes).map_err(|_| {
                                Error::InvalidArgument {
                                    context: "parse_json",
                                    reason: "invalid UTF-8 in \\u escape".to_owned(),
                                }
                            })?;
                            let code = u32::from_str_radix(hex_str, 16).map_err(|_| {
                                Error::InvalidArgument {
                                    context: "parse_json",
                                    reason: "invalid hex in \\u escape".to_owned(),
                                }
                            })?;
                            let ch = char::from_u32(code).ok_or_else(|| Error::InvalidArgument {
                                context: "parse_json",
                                reason: "invalid unicode scalar in \\u escape".to_owned(),
                            })?;
                            out.push(ch);
                        }
                        _ => {
                            return Err(Error::InvalidArgument {
                                context: "parse_json",
                                reason: "unsupported escape character".to_owned(),
                            });
                        }
                    }
                }
                c if c < 0x20 => {
                    return Err(Error::InvalidArgument {
                        context: "parse_json",
                        reason: "unescaped control character in JSON string".to_owned(),
                    });
                }
                _ => {
                    // Rewind 1 byte and decode a full UTF-8 character safely
                    self.pos -= 1;
                    let rest = core::str::from_utf8(self.bytes.get(self.pos..).unwrap_or(&[]))
                        .map_err(|_| Error::InvalidArgument {
                            context: "parse_json",
                            reason: "invalid UTF-8 in JSON string".to_owned(),
                        })?;
                    let ch = rest.chars().next().ok_or(Error::InternalInvariant {
                        message: "empty UTF-8 slice in parse_string",
                    })?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        Err(Error::InvalidArgument {
            context: "parse_json",
            reason: "unterminated JSON string".to_owned(),
        })
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice = self.bytes.get(start..self.pos).unwrap_or(&[]);
        let s = core::str::from_utf8(slice).map_err(|_| Error::InvalidArgument {
            context: "parse_json",
            reason: "invalid UTF-8 in JSON number".to_owned(),
        })?;
        let n: f64 = s.parse().map_err(|_| Error::InvalidArgument {
            context: "parse_json",
            reason: format!("invalid JSON number `{s}`"),
        })?;
        if !n.is_finite() {
            return Err(Error::InvalidArgument {
                context: "parse_json",
                reason: "non-finite JSON number".to_owned(),
            });
        }
        Ok(JsonValue::Number(n))
    }
}
