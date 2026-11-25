//! Tokenizer abstraction and zero-dependency [`ByteTokenizer`].

use as_core::{Error, Result};

/// Common tokenizer interface for encoding text to token IDs and decoding back.
pub trait Tokenizer {
    /// Total vocabulary size including special tokens.
    fn vocab_size(&self) -> usize;

    /// Beginning-of-sequence token ID.
    fn bos_id(&self) -> usize;

    /// End-of-sequence token ID.
    fn eos_id(&self) -> usize;

    /// Encodes UTF-8 text into token IDs.
    fn encode(&self, text: &str, add_bos: bool) -> Vec<usize>;

    /// Decodes token IDs back into a UTF-8 string (skipping special tokens).
    fn decode(&self, ids: &[usize]) -> Result<String>;
}

/// Lossless byte-level tokenizer (vocab = 259: 0..256 raw bytes, 256=BOS, 257=EOS, 258=PAD).
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// BOS token ID.
    pub const BOS: usize = 256;
    /// EOS token ID.
    pub const EOS: usize = 257;
    /// PAD token ID.
    pub const PAD: usize = 258;
    /// Total vocabulary size.
    pub const VOCAB_SIZE: usize = 259;
}

impl Tokenizer for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        Self::VOCAB_SIZE
    }

    fn bos_id(&self) -> usize {
        Self::BOS
    }

    fn eos_id(&self) -> usize {
        Self::EOS
    }

    fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        let mut out = Vec::with_capacity(text.len() + usize::from(add_bos));
        if add_bos {
            out.push(Self::BOS);
        }
        for &b in text.as_bytes() {
            out.push(usize::from(b));
        }
        out
    }

    fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            if id == Self::BOS || id == Self::EOS || id == Self::PAD {
                continue;
            }
            if id >= 256 {
                return Err(Error::IndexOutOfBounds {
                    index: id,
                    len: Self::VOCAB_SIZE,
                });
            }
            bytes.push(id as u8);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
