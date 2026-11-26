//! Model cards, artifact descriptors, and search queries for [`crate::ModelHub`].

use as_core::DType;

/// High-level AI task domain for model discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTask {
    /// Autoregressive / causal language model.
    Llm,
    /// Computer vision (classification, ViT, object detection).
    Vision,
    /// Tabular / general MLP / regression & classification.
    Tabular,
    /// Dense vector embedding model.
    Embedding,
    /// Custom computation graph.
    Custom,
}

/// Weight quantization format of a model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QuantizationKind {
    /// Full or half floating-point weights (F32/F16/BF16).
    #[default]
    None,
    /// 8-bit integer symmetric quantization.
    Int8,
    /// 4-bit block quantization (`Q4Linear`).
    Q4,
}

/// A single cryptographically hashed file belonging to a model (weights, config, tokenizer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    /// Logical filename (e.g. `model.safetensors`, `config.json`).
    pub filename: String,
    /// Expected lowercase hex SHA-256 digest.
    pub sha256: String,
    /// Exact byte size.
    pub size_bytes: usize,
}

/// Metadata card describing a registered model in [`crate::ModelHub`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCard {
    /// Unique model identifier (e.g. `"tarsix/smollm2-135m"`).
    pub id: String,
    /// Architecture family (e.g. `"llama"`, `"qwen2"`, `"vit"`, `"mlp"`).
    pub architecture: String,
    /// Primary task domain.
    pub task: ModelTask,
    /// Total parameter count.
    pub num_parameters: u64,
    /// Primary weight data type.
    pub dtype: DType,
    /// Quantization scheme.
    pub quantization: QuantizationKind,
    /// License identifier (e.g. `"Apache-2.0"`, `"MIT"`).
    pub license: String,
    /// Searchable tags.
    pub tags: Vec<String>,
    /// Integrity-verified artifact files.
    pub artifacts: Vec<ModelArtifact>,
}

/// Structured filter query for [`crate::ModelHub::search`].
#[derive(Debug, Clone, Default)]
pub struct ModelQuery {
    /// Case-insensitive substring matched against `id`, `architecture`, and `tags`.
    pub text: Option<String>,
    /// Optional exact task filter.
    pub task: Option<ModelTask>,
    /// Optional exact architecture filter.
    pub architecture: Option<String>,
    /// Optional maximum parameter count ceiling.
    pub max_parameters: Option<u64>,
    /// Optional quantization filter.
    pub quantization: Option<QuantizationKind>,
    /// Required tag (must be present in `card.tags`).
    pub tag: Option<String>,
}

impl ModelQuery {
    /// Returns `true` if `card` satisfies every non-`None` filter in this query.
    pub fn matches(&self, card: &ModelCard) -> bool {
        if let Some(t) = self.task
            && card.task != t
        {
            return false;
        }
        if let Some(arch) = &self.architecture
            && !card.architecture.eq_ignore_ascii_case(arch)
        {
            return false;
        }
        if let Some(max_p) = self.max_parameters
            && card.num_parameters > max_p
        {
            return false;
        }
        if let Some(q) = self.quantization
            && card.quantization != q
        {
            return false;
        }
        if let Some(req_tag) = &self.tag
            && !card.tags.iter().any(|t| t.eq_ignore_ascii_case(req_tag))
        {
            return false;
        }
        if let Some(needle) = &self.text {
            let n = needle.to_ascii_lowercase();
            let in_id = card.id.to_ascii_lowercase().contains(&n);
            let in_arch = card.architecture.to_ascii_lowercase().contains(&n);
            let in_tags = card
                .tags
                .iter()
                .any(|t| t.to_ascii_lowercase().contains(&n));
            if !in_id && !in_arch && !in_tags {
                return false;
            }
        }
        true
    }
}
