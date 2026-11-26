//! Integration tests for Faz B `ContinuousBatcher` (vLLM-style iteration-level batching).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use as_core::Result;
use as_serve::{
    ActiveSequence, ContinuousBatchConfig, ContinuousBatcher, SequenceRequest, StepGenerator,
};

struct DeterministicGenerator;

impl StepGenerator for DeterministicGenerator {
    fn step_batch(&mut self, active: &[ActiveSequence]) -> Result<Vec<usize>> {
        // Emit next token = last_token + 1
        Ok(active
            .iter()
            .map(|s| s.tokens.last().copied().unwrap_or(0) + 1)
            .collect())
    }
}

#[test]
fn continuous_batcher_admits_waiting_seq_immediately_when_short_seq_finishes() {
    let mut batcher = ContinuousBatcher::new(ContinuousBatchConfig {
        max_running_seqs: 2,
        max_waiting_seqs: 16,
    })
    .unwrap();

    // Seq 1: short (2 tokens), Seq 2: long (5 tokens), Seq 3: waiting (2 tokens)
    batcher
        .submit(SequenceRequest {
            seq_id: 1,
            prompt_tokens: vec![10],
            max_new_tokens: 2,
            eos_token_id: None,
        })
        .unwrap();
    batcher
        .submit(SequenceRequest {
            seq_id: 2,
            prompt_tokens: vec![20],
            max_new_tokens: 5,
            eos_token_id: None,
        })
        .unwrap();
    batcher
        .submit(SequenceRequest {
            seq_id: 3,
            prompt_tokens: vec![30],
            max_new_tokens: 2,
            eos_token_id: None,
        })
        .unwrap();

    let mut generator = DeterministicGenerator;
    let events = batcher.run_until_empty(&mut generator).unwrap();

    // Seq 1 finishes at iteration 1; Seq 3 MUST start at iteration 2 (while Seq 2 is still running!)
    // Total iterations = max(5, 2+2) = 5 instead of static batching's 5 + 2 = 7!
    let stats = batcher.stats();
    assert_eq!(stats.iterations, 5);
    assert_eq!(stats.completed_sequences, 3);
    assert_eq!(stats.tokens_generated, 9);
    assert_eq!(stats.peak_running_seqs, 2);

    let seq3_first_iter = events
        .iter()
        .find(|e| e.seq_id == 3)
        .map(|e| e.iteration)
        .unwrap();
    assert_eq!(seq3_first_iter, 2, "seq 3 must be admitted immediately on iteration 2");
    assert_eq!(batcher.completed()[&1], vec![11, 12]);
    assert_eq!(batcher.completed()[&2], vec![21, 22, 23, 24, 25]);
    assert_eq!(batcher.completed()[&3], vec![31, 32]);
}
