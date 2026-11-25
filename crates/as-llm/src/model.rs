//! Independent Llama-style Transformer Decoder LLM (`RmsNorm` + `RoPE` + `KvCache` + `SwiGLU`).

use as_core::{DType, Error, Result};
use as_io::SafeTensors;
use as_nn::{Rng, xavier_uniform};
use as_tensor::Tensor;

use crate::kv_cache::KvCache;
use crate::rope::RoPECache;
use crate::sampler::{SamplingConfig, sample_next_token};

/// Configuration for a Transformer decoder language model (supports MHA, MQA, and Llama-3/Qwen GQA).
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of stacked decoder layers.
    pub num_layers: usize,
    /// Number of query attention heads (must divide `hidden_size` evenly, and `head_dim` must be even).
    pub num_heads: usize,
    /// Number of Key/Value heads for Grouped-Query Attention (must divide `num_heads` evenly).
    /// Equal to `num_heads` for standard MHA, `< num_heads` for GQA (Llama-3 / Qwen).
    pub num_kv_heads: usize,
    /// SwiGLU feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Maximum supported context length.
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency (`theta`).
    pub rope_theta: f64,
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attn_norm: Tensor,    // [hidden]
    w_q: Tensor,          // [hidden, hidden]
    b_q: Option<Tensor>,  // [hidden] (used by Qwen2 / Qwen2.5)
    w_k: Tensor,          // [kv_dim, hidden]
    b_k: Option<Tensor>,  // [kv_dim]
    w_v: Tensor,          // [kv_dim, hidden]
    b_v: Option<Tensor>,  // [kv_dim]
    w_o: Tensor,          // [hidden, hidden]
    ffn_norm: Tensor,     // [hidden]
    w_gate: Tensor,       // [intermediate, hidden]
    w_up: Tensor,         // [intermediate, hidden]
    w_down: Tensor,       // [hidden, intermediate]
}

/// Complete autoregressive Transformer decoder language model.
#[derive(Debug, Clone)]
pub struct LlmModel {
    config: LlmConfig,
    token_embeddings: Tensor, // [vocab_size, hidden_size]
    layers: Vec<DecoderLayer>,
    final_norm: Tensor,       // [hidden_size]
    lm_head: Tensor,          // [vocab_size, hidden_size]
    rope: RoPECache,
}

impl LlmModel {
    /// Constructs a deterministically initialized [`LlmModel`] from `config` and `seed`.
    pub fn new(config: LlmConfig, seed: u64) -> Result<Self> {
        if config.num_heads == 0 || config.hidden_size % config.num_heads != 0 {
            return Err(Error::InvalidArgument {
                context: "LlmModel::new",
                reason: "hidden_size must be divisible by num_heads".to_owned(),
            });
        }
        if config.num_kv_heads == 0 || config.num_heads % config.num_kv_heads != 0 {
            return Err(Error::InvalidArgument {
                context: "LlmModel::new",
                reason: "num_heads must be divisible by num_kv_heads (GQA constraint)".to_owned(),
            });
        }
        let head_dim = config.hidden_size / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        let rope = RoPECache::new(config.max_seq_len, head_dim, config.rope_theta)?;
        let mut rng = Rng::from_seed(seed);

        let token_embeddings = xavier_uniform(
            vec![config.vocab_size, config.hidden_size],
            config.vocab_size,
            config.hidden_size,
            DType::F32,
            &mut rng,
        )?;
        let lm_head = xavier_uniform(
            vec![config.vocab_size, config.hidden_size],
            config.hidden_size,
            config.vocab_size,
            DType::F32,
            &mut rng,
        )?;
        let final_norm = Tensor::ones(vec![config.hidden_size], DType::F32)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(DecoderLayer {
                attn_norm: Tensor::ones(vec![config.hidden_size], DType::F32)?,
                w_q: xavier_uniform(
                    vec![config.hidden_size, config.hidden_size],
                    config.hidden_size,
                    config.hidden_size,
                    DType::F32,
                    &mut rng,
                )?,
                b_q: None,
                w_k: xavier_uniform(
                    vec![kv_dim, config.hidden_size],
                    config.hidden_size,
                    kv_dim,
                    DType::F32,
                    &mut rng,
                )?,
                b_k: None,
                w_v: xavier_uniform(
                    vec![kv_dim, config.hidden_size],
                    config.hidden_size,
                    kv_dim,
                    DType::F32,
                    &mut rng,
                )?,
                b_v: None,
                w_o: xavier_uniform(
                    vec![config.hidden_size, config.hidden_size],
                    config.hidden_size,
                    config.hidden_size,
                    DType::F32,
                    &mut rng,
                )?,
                ffn_norm: Tensor::ones(vec![config.hidden_size], DType::F32)?,
                w_gate: xavier_uniform(
                    vec![config.intermediate_size, config.hidden_size],
                    config.hidden_size,
                    config.intermediate_size,
                    DType::F32,
                    &mut rng,
                )?,
                w_up: xavier_uniform(
                    vec![config.intermediate_size, config.hidden_size],
                    config.hidden_size,
                    config.intermediate_size,
                    DType::F32,
                    &mut rng,
                )?,
                w_down: xavier_uniform(
                    vec![config.hidden_size, config.intermediate_size],
                    config.intermediate_size,
                    config.hidden_size,
                    DType::F32,
                    &mut rng,
                )?,
            });
        }

        Ok(Self {
            config,
            token_embeddings,
            layers,
            final_norm,
            lm_head,
            rope,
        })
    }

    /// Loads matching model weights from a [`SafeTensors`] archive (architecture adapter).
    pub fn load_from_safetensors(&mut self, archive: &SafeTensors) -> Result<()> {
        if let Some(t) = archive.tensors.get("model.embed_tokens.weight") {
            validate_same_shape(t, &self.token_embeddings, "model.embed_tokens.weight")?;
            self.token_embeddings = t.to_dtype(DType::F32)?;
        }
        if let Some(t) = archive.tensors.get("model.norm.weight") {
            validate_same_shape(t, &self.final_norm, "model.norm.weight")?;
            self.final_norm = t.to_dtype(DType::F32)?;
        }
        if let Some(t) = archive.tensors.get("lm_head.weight") {
            validate_same_shape(t, &self.lm_head, "lm_head.weight")?;
            self.lm_head = t.to_dtype(DType::F32)?;
        } else if archive.tensors.contains_key("model.embed_tokens.weight") {
            // Tied word embeddings (used by SmolLM2, Qwen2.5-0.5B, Gemma, etc.)
            self.lm_head = self.token_embeddings.clone();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let load_opt = |suffix: &str, target: &mut Tensor| -> Result<()> {
                let key = format!("model.layers.{i}.{suffix}");
                if let Some(t) = archive.tensors.get(&key) {
                    validate_same_shape(t, target, &key)?;
                    *target = t.to_dtype(DType::F32)?;
                }
                Ok(())
            };
            load_opt("input_layernorm.weight", &mut layer.attn_norm)?;
            load_opt("self_attn.q_proj.weight", &mut layer.w_q)?;
            load_opt("self_attn.k_proj.weight", &mut layer.w_k)?;
            load_opt("self_attn.v_proj.weight", &mut layer.w_v)?;
            load_opt("self_attn.o_proj.weight", &mut layer.w_o)?;
            load_opt("post_attention_layernorm.weight", &mut layer.ffn_norm)?;
            load_opt("mlp.gate_proj.weight", &mut layer.w_gate)?;
            load_opt("mlp.up_proj.weight", &mut layer.w_up)?;
            load_opt("mlp.down_proj.weight", &mut layer.w_down)?;

            // Optional Q/K/V biases (Qwen2 / Qwen2.5 architectures)
            for (suffix, slot) in [
                ("self_attn.q_proj.bias", &mut layer.b_q),
                ("self_attn.k_proj.bias", &mut layer.b_k),
                ("self_attn.v_proj.bias", &mut layer.b_v),
            ] {
                let key = format!("model.layers.{i}.{suffix}");
                if let Some(t) = archive.tensors.get(&key) {
                    *slot = Some(t.to_dtype(DType::F32)?);
                }
            }
        }
        Ok(())
    }

    /// Returns the model's [`LlmConfig`].
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Allocates a fresh [`KvCache`] sized for this model (`num_kv_heads` for GQA memory savings).
    pub fn new_kv_cache(&self, batch_size: usize) -> KvCache {
        let head_dim = self.config.hidden_size / self.config.num_heads;
        KvCache::new(
            self.config.num_layers,
            batch_size,
            self.config.num_kv_heads,
            head_dim,
            self.config.max_seq_len,
        )
    }

    /// Runs a forward pass over `token_ids` (batch size 1, length `seq_len`),
    /// optionally updating `kv_cache`, and returns logits of shape `[1, seq_len, vocab_size]`.
    pub fn forward(&self, token_ids: &[usize], kv_cache: Option<&mut KvCache>) -> Result<Tensor> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(Error::InvalidArgument {
                context: "LlmModel::forward",
                reason: "token_ids cannot be empty".to_owned(),
            });
        }
        let pos_offset = kv_cache.as_ref().map_or(0, |c| c.seq_len());
        let mut h = embed_lookup(&self.token_embeddings, token_ids, self.config.hidden_size)?;

        let mut kv_opt = kv_cache;
        let head_dim = self.config.hidden_size / self.config.num_heads;
        let scale = Tensor::full(vec![], 1.0 / (head_dim as f64).sqrt(), DType::F32)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 1. Pre-Attention RMSNorm
            let normed = rms_norm(&h, &layer.attn_norm, self.config.rms_norm_eps)?;

            // 2. Q, K, V projections (GQA + optional Qwen2 biases)
            let q = linear_opt_bias(&normed, &layer.w_q, layer.b_q.as_ref())?
                .reshape(vec![1, seq_len, self.config.num_heads, head_dim])?
                .transpose(1, 2)?; // [1, H, T, Hd]
            let k = linear_opt_bias(&normed, &layer.w_k, layer.b_k.as_ref())?
                .reshape(vec![1, seq_len, self.config.num_kv_heads, head_dim])?
                .transpose(1, 2)?; // [1, H_kv, T, Hd]
            let v = linear_opt_bias(&normed, &layer.w_v, layer.b_v.as_ref())?
                .reshape(vec![1, seq_len, self.config.num_kv_heads, head_dim])?
                .transpose(1, 2)?;

            // 3. Apply RoPE at pos_offset
            let q_rot = self.rope.apply(&q, pos_offset)?;
            let k_rot = self.rope.apply(&k, pos_offset)?;

            // 4. Update or bypass KV-Cache
            let (k_cached, v_cached) = if let Some(cache) = kv_opt.as_deref_mut() {
                cache.append(layer_idx, &k_rot, &v)?
            } else {
                (k_rot, v)
            };
            let total_seq = k_cached.shape().dim(2)?;

            // 5. GQA head expansion: repeat KV heads to match query heads when num_kv_heads < num_heads
            let group_size = self.config.num_heads / self.config.num_kv_heads;
            let k_full = repeat_kv_heads(&k_cached, group_size)?;
            let v_full = repeat_kv_heads(&v_cached, group_size)?;

            // 5. Scaled Dot-Product Causal Attention
            let kt = k_full.transpose(2, 3)?;
            let mut scores = q_rot.matmul(&kt)?.mul(&scale)?; // [1, H, T, total_seq]
            if seq_len > 1 {
                let mask = causal_mask(seq_len, total_seq, pos_offset)?;
                scores = scores.add(&mask)?;
            }
            let probs = softmax_last(&scores)?;
            let attn_out = probs
                .matmul(&v_full)?
                .transpose(1, 2)?
                .reshape(vec![1, seq_len, self.config.hidden_size])?;
            let proj = linear_no_bias(&attn_out, &layer.w_o)?;
            h = h.add(&proj)?;

            // 6. Pre-FFN RMSNorm + SwiGLU FFN
            let normed2 = rms_norm(&h, &layer.ffn_norm, self.config.rms_norm_eps)?;
            let gate = silu_tensor(&linear_no_bias(&normed2, &layer.w_gate)?)?;
            let up = linear_no_bias(&normed2, &layer.w_up)?;
            let ffn = linear_no_bias(&gate.mul(&up)?, &layer.w_down)?;
            h = h.add(&ffn)?;
        }

        let final_h = rms_norm(&h, &self.final_norm, self.config.rms_norm_eps)?;
        linear_no_bias(&final_h, &self.lm_head)
    }

    /// Runs autoregressive streaming generation using KV-Cache, invoking `on_token`
    /// for every newly generated token until `max_new_tokens`, `eos_id`, or `on_token` returns `false`.
    pub fn generate_stream<F>(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        eos_id: usize,
        sampling: SamplingConfig,
        rng: &mut Rng,
        mut on_token: F,
    ) -> Result<Vec<usize>>
    where
        F: FnMut(usize) -> bool,
    {
        if prompt_ids.is_empty() {
            return Err(Error::InvalidArgument {
                context: "LlmModel::generate_stream",
                reason: "prompt_ids cannot be empty".to_owned(),
            });
        }
        let mut cache = self.new_kv_cache(1);
        let mut history = prompt_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        // Prefill prompt into KV-Cache
        let mut logits = self.forward(prompt_ids, Some(&mut cache))?;

        for _ in 0..max_new_tokens {
            let last_logits = extract_last_token_logits(&logits, self.config.vocab_size)?;
            let next_tok = sample_next_token(&last_logits, &history, sampling, rng)?;
            history.push(next_tok);
            generated.push(next_tok);

            let keep_going = on_token(next_tok);
            if !keep_going || next_tok == eos_id || cache.seq_len() >= self.config.max_seq_len {
                break;
            }
            // Incremental 1-token decode step
            logits = self.forward(&[next_tok], Some(&mut cache))?;
        }

        Ok(generated)
    }
}

fn validate_same_shape(src: &Tensor, dst: &Tensor, name: &str) -> Result<()> {
    if src.shape() != dst.shape() {
        return Err(Error::ShapeMismatch {
            context: if name.is_empty() {
                "LlmModel weight"
            } else {
                "LlmModel::load_from_safetensors"
            },
            expected: dst.dims().to_vec(),
            actual: src.dims().to_vec(),
        });
    }
    Ok(())
}

fn embed_lookup(table: &Tensor, ids: &[usize], hidden_size: usize) -> Result<Tensor> {
    let vocab_size = table.shape().dim(0)?;
    let raw = table.to_f32_vec()?;
    let mut out = Vec::with_capacity(ids.len() * hidden_size);
    for &id in ids {
        if id >= vocab_size {
            return Err(Error::IndexOutOfBounds {
                index: id,
                len: vocab_size,
            });
        }
        let start = id * hidden_size;
        let end = start + hidden_size;
        let row = raw.get(start..end).ok_or(Error::IndexOutOfBounds {
            index: end,
            len: raw.len(),
        })?;
        out.extend_from_slice(row);
    }
    Tensor::from_f32(&out, vec![1, ids.len(), hidden_size])
}

fn linear_no_bias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let wt = weight.transpose(0, 1)?;
    x.matmul(&wt)
}

fn linear_opt_bias(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let out = linear_no_bias(x, weight)?;
    match bias {
        Some(b) => out.add(b),
        None => Ok(out),
    }
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let last_axis = x.rank().saturating_sub(1);
    let sq = x.mul(x)?;
    let ms = sq.mean_axis(last_axis, true)?;
    let eps_t = Tensor::full(vec![], eps, DType::F32)?;
    let rms = ms.add(&eps_t)?.sqrt()?;
    x.div(&rms)?.mul(weight)
}

fn silu_tensor(x: &Tensor) -> Result<Tensor> {
    let vals: Vec<f32> = x
        .to_f32_vec()?
        .into_iter()
        .map(|v| v / (1.0 + (-v).exp()))
        .collect();
    Tensor::from_f32(&vals, x.dims().to_vec())
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let last_axis = x.rank().saturating_sub(1);
    let max_v = x.max_axis(last_axis, true)?;
    let shifted = x.sub(&max_v)?;
    let exps = shifted.exp()?;
    let sums = exps.sum_axis(last_axis, true)?;
    exps.div(&sums)
}

fn causal_mask(seq_len: usize, total_seq: usize, pos_offset: usize) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * total_seq];
    for i in 0..seq_len {
        let abs_pos = pos_offset + i;
        for j in (abs_pos + 1)..total_seq {
            if let Some(slot) = mask.get_mut(i * total_seq + j) {
                *slot = -1e9;
            }
        }
    }
    Tensor::from_f32(&mask, vec![1, 1, seq_len, total_seq])
}

fn extract_last_token_logits(logits: &Tensor, vocab_size: usize) -> Result<Vec<f32>> {
    let seq_len = logits.shape().dim(1)?;
    let all = logits.to_f32_vec()?;
    let start = (seq_len - 1) * vocab_size;
    let end = start + vocab_size;
    let slice = all.get(start..end).ok_or(Error::IndexOutOfBounds {
        index: end,
        len: all.len(),
    })?;
    Ok(slice.to_vec())
}

/// Expands `[B, H_kv, T, Hd]` to `[B, H_kv * group_size, T, Hd]` for Grouped-Query Attention (GQA).
fn repeat_kv_heads(x: &Tensor, group_size: usize) -> Result<Tensor> {
    if group_size == 1 {
        return Ok(x.clone());
    }
    let b = x.shape().dim(0)?;
    let h_kv = x.shape().dim(1)?;
    let seq = x.shape().dim(2)?;
    let hd = x.shape().dim(3)?;
    let src = x.to_f32_vec()?;
    let head_stride = seq * hd;
    let mut out = Vec::with_capacity(b * h_kv * group_size * head_stride);

    for batch in 0..b {
        for kv_h in 0..h_kv {
            let start = (batch * h_kv + kv_h) * head_stride;
            let end = start + head_stride;
            let slice = src.get(start..end).ok_or(Error::IndexOutOfBounds {
                index: end,
                len: src.len(),
            })?;
            for _ in 0..group_size {
                out.extend_from_slice(slice);
            }
        }
    }
    Tensor::from_f32(&out, vec![b, h_kv * group_size, seq, hd])
}
