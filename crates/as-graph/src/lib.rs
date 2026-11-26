//! Independent computational graph engine.
//!
//! Provides:
//! - Static and dynamic DAG construction ([`Graph`], [`Node`], [`NodeId`], [`Op`])
//! - Iterative topological sorting with cycle detection
//! - Graph optimization passes: Dead-Code Elimination (DCE) and Unary Operation Fusion
//! - Memory-scheduled forward execution (releasing dead intermediates early)
//! - Checksum-verified, bounds-checked binary graph serialization for untrusted inputs

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod compiler;
pub mod executor;
pub mod infer;
pub mod ir;
pub mod optimize;
pub mod serialize;

pub use compiler::{
    AlgebraicSimplifyPass, CompilationReport, CompiledGraph, CompilerPass, ConstantFoldingPass,
    CsePass, DeadCodeEliminationPass, LinearActivationFusionPass, OptLevel, PassPipeline,
    PassStats, UnaryFusionPass, compile_graph,
};
pub use executor::{ExecutionPlan, NodeExecutionRecord, execute_graph, execute_graph_profiled};
pub use infer::{TensorMeta, build_memory_plan, infer_graph_metadata};
pub use ir::{Graph, MAX_GRAPH_NODES, Node, NodeId, Op};
pub use optimize::{OptimizationReport, optimize_graph};
pub use serialize::{deserialize_graph, serialize_graph};
