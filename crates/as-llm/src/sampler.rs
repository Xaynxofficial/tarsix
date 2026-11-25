//! Autoregressive token sampling: greedy, temperature, top-k, top-p, and repetition penalty.

use as_core::{Error, Result};
use as_nn::Rng;

/// Sampling hyperparameters for token generation.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Softmax temperature (`0.0` = deterministic greedy argmax).
    pub temperature: f64,
    /// Optional top-k truncation (`0` = disabled).
    pub top_k: usize,
    /// Nucleus sampling cumulative probability threshold in `(0.0, 1.0]`.
    pub top_p: f64,
    /// Repetition penalty (`1.0` = disabled, `> 1.0` penalizes previously generated tokens).
    pub repetition_penalty: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.0,
        }
    }
}

/// Samples the next token ID from a 1-D logit slice using `config` and deterministic `rng`.
pub fn sample_next_token(
    logits: &[f32],
    history: &[usize],
    config: SamplingConfig,
    rng: &mut Rng,
) -> Result<usize> {
    if logits.is_empty() {
        return Err(Error::InvalidArgument {
            context: "sample_next_token",
            reason: "logits slice cannot be empty".to_owned(),
        });
    }

    let mut adjusted: Vec<f64> = logits.iter().map(|&x| f64::from(x)).collect();

    // Repetition penalty
    if (config.repetition_penalty - 1.0).abs() > 1e-9 && config.repetition_penalty > 0.0 {
        for &tok in history {
            if let Some(slot) = adjusted.get_mut(tok) {
                if *slot > 0.0 {
                    *slot /= config.repetition_penalty;
                } else {
                    *slot *= config.repetition_penalty;
                }
            }
        }
    }

    // Greedy decoding when temperature <= 0
    if config.temperature <= 0.0 {
        let mut best_id = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (idx, &v) in adjusted.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = idx;
            }
        }
        return Ok(best_id);
    }

    // Temperature scaling + sort descending by logit
    let inv_temp = 1.0 / config.temperature;
    let mut indexed: Vec<(usize, f64)> = adjusted
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i, v * inv_temp))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // Top-k truncation
    if config.top_k > 0 && config.top_k < indexed.len() {
        indexed.truncate(config.top_k);
    }

    // Numerically stable softmax over remaining candidates
    let max_logit = indexed.first().map_or(0.0, |x| x.1);
    let mut probs: Vec<(usize, f64)> = indexed
        .into_iter()
        .map(|(i, v)| (i, (v - max_logit).exp()))
        .collect();
    let sum_exp: f64 = probs.iter().map(|x| x.1).sum();
    if sum_exp > 0.0 {
        for p in &mut probs {
            p.1 /= sum_exp;
        }
    }

    // Top-p (nucleus) filtering
    if config.top_p > 0.0 && config.top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = 0usize;
        for &(_, p) in &probs {
            cum += p;
            keep += 1;
            if cum >= config.top_p {
                break;
            }
        }
        probs.truncate(keep.max(1));
        let renorm: f64 = probs.iter().map(|x| x.1).sum();
        if renorm > 0.0 {
            for p in &mut probs {
                p.1 /= renorm;
            }
        }
    }

    // Sample from categorical distribution
    let r = rng.next_f64();
    let mut acc = 0.0;
    for &(tok_id, p) in &probs {
        acc += p;
        if r <= acc {
            return Ok(tok_id);
        }
    }

    Ok(probs.last().map_or(0, |x| x.0))
}
