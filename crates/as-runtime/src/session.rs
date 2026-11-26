//! Inference session, dynamic batching, and streaming execution.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use as_core::{Device, Error, MemoryLimits, Result};
use as_graph::{
    CompilationReport, Graph, NodeId, Op, OptLevel, OptimizationReport, compile_graph,
    deserialize_graph, execute_graph, execute_graph_profiled, serialize_graph,
};
use as_io::{SafeTensors, deserialize_safetensors};
use as_memory::MemoryPlan;
use as_tensor::Tensor;

use crate::profiler::InferenceProfile;

/// Configuration for creating an [`InferenceSession`].
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Target execution device (Milestone 1–4: [`Device::Cpu`]).
    pub device: Device,
    /// Security and resource ceilings enforced during model loading and execution.
    pub limits: MemoryLimits,
    /// Whether to run graph optimization upon session creation.
    pub enable_optimization: bool,
    /// Compiler optimization level used when `enable_optimization` is `true`.
    pub opt_level: OptLevel,
    /// Whether to record detailed per-operator execution profiles.
    pub enable_profiling: bool,
    /// Maximum number of requests combined in a single dynamic batch chunk.
    pub max_batch_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            limits: MemoryLimits::default(),
            enable_optimization: true,
            opt_level: OptLevel::O3,
            enable_profiling: true,
            max_batch_size: 32,
        }
    }
}

/// Independent model inference session backed by a compiled [`Graph`] and bound weights.
#[derive(Debug, Clone)]
pub struct InferenceSession {
    graph: Graph,
    config: RuntimeConfig,
    optimization: Option<OptimizationReport>,
    compilation: Option<CompilationReport>,
    memory_plan: Option<MemoryPlan>,
    input_name_map: BTreeMap<String, NodeId>,
    bound_weights: HashMap<NodeId, Tensor>,
}

impl InferenceSession {
    /// Builds an [`InferenceSession`] from a [`Graph`] and [`RuntimeConfig`].
    pub fn new(graph: Graph, config: RuntimeConfig) -> Result<Self> {
        if config.device != Device::Cpu {
            return Err(Error::InvalidArgument {
                context: "InferenceSession::new",
                reason: format!(
                    "device {} is not yet enabled (CPU backend active)",
                    config.device
                ),
            });
        }
        if config.max_batch_size == 0 {
            return Err(Error::InvalidArgument {
                context: "InferenceSession::new",
                reason: "max_batch_size must be >= 1".to_owned(),
            });
        }

        let effective_level = if config.enable_optimization {
            config.opt_level
        } else {
            OptLevel::O0
        };
        let compiled = compile_graph(&graph, effective_level)?;
        let optimization = config.enable_optimization.then(|| {
            let dead_removed: usize = compiled
                .report
                .pass_stats
                .iter()
                .filter(|p| p.pass_name == "dead_code_elimination")
                .map(|p| p.rewrites)
                .sum();
            let unary_fused: usize = compiled
                .report
                .pass_stats
                .iter()
                .filter(|p| p.pass_name == "unary_fusion")
                .map(|p| p.rewrites)
                .sum();
            OptimizationReport {
                nodes_before: compiled.report.nodes_before,
                nodes_after: compiled.report.nodes_after,
                dead_nodes_removed: dead_removed,
                unary_ops_fused: unary_fused,
            }
        });

        let active_graph = compiled.graph;
        let mut input_name_map = BTreeMap::new();
        for node in active_graph.nodes() {
            if let Op::Input { name, .. } = &node.op {
                input_name_map.insert(name.clone(), node.id);
            }
        }

        Ok(Self {
            graph: active_graph,
            config,
            optimization,
            compilation: Some(compiled.report),
            memory_plan: Some(compiled.memory_plan),
            input_name_map,
            bound_weights: HashMap::new(),
        })
    }

    /// Deserializes a serialized `ASGR` graph and creates an [`InferenceSession`].
    pub fn from_graph_bytes(bytes: &[u8], config: RuntimeConfig) -> Result<Self> {
        let graph = deserialize_graph(bytes)?;
        Self::new(graph, config)
    }

    /// Serializes the session's active (optimized) graph.
    pub fn to_graph_bytes(&self) -> Vec<u8> {
        serialize_graph(&self.graph)
    }

    /// Loads weights from a raw Safetensors byte slice and binds any matching graph inputs by name.
    pub fn load_safetensors_bytes(&mut self, bytes: &[u8]) -> Result<usize> {
        let archive = deserialize_safetensors(bytes, self.config.limits)?;
        self.bind_safetensors(&archive)
    }

    /// Binds tensors from a decoded [`SafeTensors`] archive to matching named graph inputs.
    pub fn bind_safetensors(&mut self, archive: &SafeTensors) -> Result<usize> {
        let mut bound = 0usize;
        for (name, tensor) in &archive.tensors {
            if let Some(&node_id) = self.input_name_map.get(name) {
                self.bound_weights.insert(node_id, tensor.clone());
                bound += 1;
            }
        }
        Ok(bound)
    }

    /// Returns the graph's named input slots.
    pub fn input_names(&self) -> &BTreeMap<String, NodeId> {
        &self.input_name_map
    }

    /// Returns the graph's output node IDs.
    pub fn output_ids(&self) -> &[NodeId] {
        self.graph.outputs()
    }

    /// Returns the active [`RuntimeConfig`].
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns the static buffer-reuse [`MemoryPlan`] computed during compilation.
    pub fn memory_plan(&self) -> Option<&MemoryPlan> {
        self.memory_plan.as_ref()
    }

    /// Returns the compiler [`CompilationReport`] produced during session creation.
    pub fn compilation_report(&self) -> Option<&CompilationReport> {
        self.compilation.as_ref()
    }

    /// Executes a single inference pass with named runtime inputs merged with bound weights.
    pub fn run_named(
        &self,
        named_inputs: &BTreeMap<String, Tensor>,
    ) -> Result<(HashMap<NodeId, Tensor>, Option<InferenceProfile>)> {
        let mut feed = self.bound_weights.clone();
        for (name, tensor) in named_inputs {
            let node_id =
                self.input_name_map
                    .get(name)
                    .copied()
                    .ok_or_else(|| Error::InvalidArgument {
                        context: "InferenceSession::run_named",
                        reason: format!("unknown graph input `{name}`"),
                    })?;
            feed.insert(node_id, tensor.clone());
        }

        if self.config.enable_profiling {
            let t0 = Instant::now();
            let (outputs, records) = execute_graph_profiled(&self.graph, &feed)?;
            let profile = InferenceProfile::from_records(
                t0.elapsed().as_micros(),
                records,
                self.optimization,
                self.compilation.clone(),
                self.memory_plan.clone(),
            );
            Ok((outputs, Some(profile)))
        } else {
            let outputs = execute_graph(&self.graph, &feed)?;
            Ok((outputs, None))
        }
    }

    /// Executes a dynamic batch of requests in chunks up to `config.max_batch_size`.
    pub fn run_batched(
        &self,
        requests: &[BTreeMap<String, Tensor>],
    ) -> Result<Vec<HashMap<NodeId, Tensor>>> {
        let mut responses = Vec::with_capacity(requests.len());
        for chunk in requests.chunks(self.config.max_batch_size) {
            for req in chunk {
                let (out, _) = self.run_named(req)?;
                responses.push(out);
            }
        }
        Ok(responses)
    }

    /// Executes a multi-step streaming inference loop, invoking `on_step` after each step
    /// until `max_steps` is reached or `on_step` returns `Ok(false)` to stop early.
    pub fn stream_steps<F, G>(
        &self,
        max_steps: usize,
        mut next_input: F,
        mut on_step: G,
    ) -> Result<usize>
    where
        F: FnMut(usize) -> Result<BTreeMap<String, Tensor>>,
        G: FnMut(usize, &HashMap<NodeId, Tensor>) -> Result<bool>,
    {
        let mut completed = 0usize;
        for step in 0..max_steps {
            let feed = next_input(step)?;
            let (out, _) = self.run_named(&feed)?;
            completed += 1;
            let should_continue = on_step(step, &out)?;
            if !should_continue {
                break;
            }
        }
        Ok(completed)
    }
}
