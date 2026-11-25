# Contributing to Tarsix

Thank you for your interest in contributing to **Tarsix**!

## Engineering Standards

1. **Zero `unsafe` in Core Crates**: All core crates (`as-core`, `as-backend`, `as-cpu`, `as-tensor`, `as-graph`, `as-autograd`, `as-nn`, `as-optim`, `as-train`, `as-io`, `as-runtime`, `as-llm`, `as-vision`, `as-accel`, `as-cli`, `as-bench`) enforce `#![forbid(unsafe_code)]`. Hardware FFI (`as-metal`, `as-wasm`) must isolate `unsafe` behind validated boundaries.
2. **Panic-Free Library Code**: Avoid `unwrap()`, `expect()`, and `panic!()` in library paths; return `as_core::Result<T>`.
3. **Zero Unnecessary Dependencies**: Core computation crates must remain dependency-free.
4. **Numerical & Security Verification**: Every new operator or file parser must include numerical reference tests (or finite-difference `gradcheck`) and untrusted-input bounds tests.

## Running Checks Locally

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
