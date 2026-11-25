//! Independent HuggingFace `tokenizer.json` BPE tokenizer loader & encoder/decoder.
//!
//! Parses standard `tokenizer.json` files using our recursion-bounded `as_io::parse_json`
//! without external dependencies. Supports GPT-2 / Llama-3 / Qwen byte-level markers (`Ġ`, `Ċ`).

use std::collections::HashMap;

use as_core::{Error, Result};
use as_io::parse_json;

use crate::tokenizer::Tokenizer;

/// Byte-Pair Encoding (BPE) tokenizer loaded from HuggingFace `tokenizer.json` format.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: Vec<String>,
    merge_ranks: HashMap<(String, String), usize>,
    bos_id: usize,
    eos_id: usize,
    unk_id: usize,
}

impl BpeTokenizer {
    /// Parses a HuggingFace `tokenizer.json` string into a [`BpeTokenizer`].
    pub fn from_json_str(json_str: &str) -> Result<Self> {
        let root = parse_json(json_str)?;
        let obj = root.as_object().ok_or_else(|| Error::InvalidArgument {
            context: "BpeTokenizer::from_json_str",
            reason: "tokenizer.json root must be a JSON object".to_owned(),
        })?;

        let model = obj
            .get("model")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::InvalidArgument {
                context: "BpeTokenizer::from_json_str",
                reason: "missing `model` object in tokenizer.json".to_owned(),
            })?;

        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::InvalidArgument {
                context: "BpeTokenizer::from_json_str",
                reason: "missing `model.vocab` object in tokenizer.json".to_owned(),
            })?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0usize;
        for (tok, id_val) in vocab_obj {
            let id = id_val.as_usize().ok_or_else(|| Error::InvalidArgument {
                context: "BpeTokenizer::from_json_str",
                reason: format!("non-integer token id for `{tok}`"),
            })?;
            vocab.insert(tok.clone(), id);
            if id > max_id {
                max_id = id;
            }
        }

        // Also register added_tokens (e.g. <|begin_of_text|>, <|end_of_text|>)
        let mut bos_id = vocab
            .get("<s>")
            .or_else(|| vocab.get("<|begin_of_text|>"))
            .copied()
            .unwrap_or(0);
        let mut eos_id = vocab
            .get("</s>")
            .or_else(|| vocab.get("<|end_of_text|>"))
            .or_else(|| vocab.get("<|endoftext|>"))
            .copied()
            .unwrap_or(0);
        let unk_id = vocab.get("<unk>").copied().unwrap_or(0);

        if let Some(added) = obj.get("added_tokens").and_then(|v| v.as_array()) {
            for item in added {
                if let Some(ao) = item.as_object()
                    && let (Some(content), Some(id)) = (
                        ao.get("content").and_then(|v| v.as_str()),
                        ao.get("id").and_then(|v| v.as_usize()),
                    )
                {
                    vocab.insert(content.to_owned(), id);
                    if id > max_id {
                        max_id = id;
                    }
                    if content == "<s>" || content == "<|begin_of_text|>" {
                        bos_id = id;
                    }
                    if content == "</s>"
                        || content == "<|end_of_text|>"
                        || content == "<|endoftext|>"
                    {
                        eos_id = id;
                    }
                }
            }
        }

        let vocab_len = if vocab.is_empty() { 0 } else { max_id + 1 };
        let mut id_to_token = vec![String::new(); vocab_len];
        for (tok, &id) in &vocab {
            if let Some(slot) = id_to_token.get_mut(id) {
                *slot = tok.clone();
            }
        }

        let mut merge_ranks = HashMap::new();
        if let Some(merges) = model.get("merges").and_then(|v| v.as_array()) {
            for (rank, entry) in merges.iter().enumerate() {
                if let Some(pair_str) = entry.as_str() {
                    if let Some((a, b)) = pair_str.split_once(' ') {
                        merge_ranks.insert((a.to_owned(), b.to_owned()), rank);
                    }
                } else if let Some(pair_arr) = entry.as_array()
                    && pair_arr.len() == 2
                    && let (Some(a), Some(b)) = (
                        pair_arr.first().and_then(|v| v.as_str()),
                        pair_arr.get(1).and_then(|v| v.as_str()),
                    )
                {
                    merge_ranks.insert((a.to_owned(), b.to_owned()), rank);
                }
            }
        }

        Ok(Self {
            vocab,
            id_to_token,
            merge_ranks,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    fn bpe_word(&self, word: &str) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.len() <= 1 {
            return symbols;
        }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let (Some(a), Some(b)) = (symbols.get(i), symbols.get(i + 1))
                    && let Some(&rank) = self.merge_ranks.get(&(a.clone(), b.clone()))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_idx = Some(i);
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            if let (Some(a), Some(b)) = (symbols.get(idx).cloned(), symbols.get(idx + 1).cloned()) {
                let merged = format!("{a}{b}");
                if let Some(slot) = symbols.get_mut(idx) {
                    *slot = merged;
                }
                symbols.remove(idx + 1);
            }
            if symbols.len() <= 1 {
                break;
            }
        }
        symbols
    }
}

impl Tokenizer for BpeTokenizer {
    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn bos_id(&self) -> usize {
        self.bos_id
    }

    fn eos_id(&self) -> usize {
        self.eos_id
    }

    fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }

        // Pre-tokenize into whitespace-prefixed chunks (GPT-2 / Llama-3 `Ġ` convention)
        let mut chunks = Vec::new();
        let mut cur = String::new();
        for ch in text.chars() {
            if ch == ' ' {
                if !cur.is_empty() {
                    chunks.push(core::mem::take(&mut cur));
                }
                cur.push('Ġ');
            } else if ch == '\n' {
                if !cur.is_empty() {
                    chunks.push(core::mem::take(&mut cur));
                }
                chunks.push("Ċ".to_owned());
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }

        for chunk in chunks {
            for piece in self.bpe_word(&chunk) {
                let id = self.vocab.get(&piece).copied().unwrap_or(self.unk_id);
                ids.push(id);
            }
        }
        ids
    }

    fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut raw = String::new();
        for &id in ids {
            if id == self.bos_id || id == self.eos_id {
                continue;
            }
            let tok = self
                .id_to_token
                .get(id)
                .ok_or(Error::IndexOutOfBounds {
                    index: id,
                    len: self.id_to_token.len(),
                })?;
            raw.push_str(tok);
        }
        Ok(raw.replace('Ġ', " ").replace('Ċ', "\n"))
    }
}
