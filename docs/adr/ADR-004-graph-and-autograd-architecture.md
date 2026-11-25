# ADR-004: Computational Graph IR, Operation Fusion, and Panic-Free Autodiff Tape

- **Status**: Accepted
- **Date**: 2026-11-25

## Context

Phase 2 introduces a computational graph engine (`as-graph`) and an automatic differentiation engine (`as-autograd`). Many Rust autodiff libraries rely on `Rc<RefCell<Tape>>`, which can panic at runtime on overlapping borrows and obscures graph ownership.

## Decision

1. **Explicit Panic-Free `Tape` + `VarId` Handles (`as-autograd`)**:
   - Eliminates `RefCell` runtime borrow panics and `Mutex` contention.
   - Supports reverse-mode VJP with broadcast unbroadcasting, multi-step gradient accumulation (`backward_into`), `no_grad` mode, and true recomputing gradient checkpointing (`Tape::checkpoint`).
   - Provides forward-mode differentiation via `DualTensor` (JVP) and central finite-difference `gradcheck` in `F64`.
2. **DAG IR with Optimization & Safe Binary Serialization (`as-graph`)**:
   - Iterative Kahn topological sort enforces acyclicity without stack recursion.
   - Optimization pipeline executes Dead-Code Elimination (DCE) and Unary Operation Fusion (`FusedUnaryChain`) into single-pass host execution.
   - Binary serialization (`ASGR` v1) validates magic header, version, FNV-1a 64-bit integrity checksum, string/node/constant ceilings, and DAG acyclicity on untrusted byte slices.
