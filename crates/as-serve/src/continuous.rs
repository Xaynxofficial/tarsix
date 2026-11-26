//! Continuous (Iteration-Level) Batching scheduler for LLM token generation.
//!
//! Unlike static batching (which stalls all short requests until the longest sequence
//! completes), [`ContinuousBatcher`] schedules at the **single-token iteration** boundary:
//! - As soon as any active sequence emits `eos_token_id` or reaches `max_new_tokens`,
//!   it is immediately retired on that iteration.
//! - Waiting requests are admitted into freed slots on the very next token step,
//!   eliminating head-of-line blocking and maximizing GPU/CPU utilization.

use std::collections::{BTreeMap, VecDeque};

use as_core::{Error, Result};

/// Configuration for [`ContinuousBatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuousBatchConfig {
    /// Maximum number of sequences generating concurrently in one iteration.
    pub max_running_seqs: usize,
    /// Maximum number of queued waiting requests allowed.
    pub max_waiting_seqs: usize,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_running_seqs: 16,
            max_waiting_seqs: 256,
        }
    }
}

/// Incoming autoregressive generation request.
#[derive(Debug, Clone)]
pub struct SequenceRequest {
    /// Unique request/sequence identifier.
    pub seq_id: u64,
    /// Input prompt token IDs.
    pub prompt_tokens: Vec<usize>,
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Optional end-of-sequence token ID that stops generation early.
    pub eos_token_id: Option<usize>,
}

/// Live state of an active sequence passed to [`StepGenerator::step_batch`].
#[derive(Debug, Clone)]
pub struct ActiveSequence {
    /// Sequence identifier.
    pub seq_id: u64,
    /// Prompt tokens + generated tokens so far.
    pub tokens: Vec<usize>,
    /// Number of newly generated tokens so far.
    pub generated_count: usize,
    /// Maximum new tokens requested.
    pub max_new_tokens: usize,
    /// Optional EOS stop token.
    pub eos_token_id: Option<usize>,
}

/// Output emitted for a sequence on a single decode iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenStepOutput {
    /// Sequence identifier.
    pub seq_id: u64,
    /// Newly sampled token ID.
    pub token_id: usize,
    /// `true` if this sequence finished on this iteration (EOS or `max_new_tokens`).
    pub finished: bool,
    /// Iteration index at which this token was emitted.
    pub iteration: usize,
}

/// Telemetry counters for [`ContinuousBatcher`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContinuousBatchStats {
    /// Total decode iterations executed.
    pub iterations: usize,
    /// Total tokens generated across all sequences.
    pub tokens_generated: usize,
    /// Total sequences completed and retired.
    pub completed_sequences: usize,
    /// Peak concurrent running sequences in any single iteration.
    pub peak_running_seqs: usize,
}

/// Backend contract that generates 1 next token for each sequence in the current active batch.
pub trait StepGenerator {
    /// Given the currently active sequences, returns one next token ID per sequence in the same order.
    fn step_batch(&mut self, active: &[ActiveSequence]) -> Result<Vec<usize>>;
}

/// vLLM-style iteration-level continuous batching scheduler.
#[derive(Debug)]
pub struct ContinuousBatcher {
    config: ContinuousBatchConfig,
    waiting: VecDeque<SequenceRequest>,
    running: Vec<ActiveSequence>,
    completed: BTreeMap<u64, Vec<usize>>,
    stats: ContinuousBatchStats,
}

impl ContinuousBatcher {
    /// Creates a new [`ContinuousBatcher`] with `config`.
    pub fn new(config: ContinuousBatchConfig) -> Result<Self> {
        if config.max_running_seqs == 0 {
            return Err(Error::InvalidArgument {
                context: "ContinuousBatcher::new",
                reason: "max_running_seqs must be >= 1".to_owned(),
            });
        }
        Ok(Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            completed: BTreeMap::new(),
            stats: ContinuousBatchStats::default(),
        })
    }

    /// Enqueues a new [`SequenceRequest`] into the waiting queue.
    pub fn submit(&mut self, req: SequenceRequest) -> Result<()> {
        if self.waiting.len() >= self.config.max_waiting_seqs {
            return Err(Error::InvalidArgument {
                context: "ContinuousBatcher::submit",
                reason: "waiting queue is full".to_owned(),
            });
        }
        self.waiting.push_back(req);
        Ok(())
    }

    /// Returns `true` if there are any waiting or currently running sequences.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Returns the number of currently active (running) sequences.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Returns the number of queued waiting sequences.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Returns current [`ContinuousBatchStats`].
    pub const fn stats(&self) -> ContinuousBatchStats {
        self.stats
    }

    /// Returns completed sequences (`seq_id -> generated_tokens`).
    pub fn completed(&self) -> &BTreeMap<u64, Vec<usize>> {
        &self.completed
    }

    /// Executes a single iteration-level decode step:
    /// 1. Admits waiting requests into any open running slots (`<= max_running_seqs`)
    /// 2. Invokes `generator.step_batch(&running)` to produce 1 token per active sequence
    /// 3. Immediately retires any sequence hitting EOS or `max_new_tokens`
    pub fn step<G: StepGenerator>(&mut self, generator: &mut G) -> Result<Vec<TokenStepOutput>> {
        // 1. Admit waiting requests into freed slots
        while self.running.len() < self.config.max_running_seqs {
            let Some(req) = self.waiting.pop_front() else {
                break;
            };
            if req.max_new_tokens == 0 {
                self.completed.insert(req.seq_id, Vec::new());
                self.stats.completed_sequences += 1;
                continue;
            }
            self.running.push(ActiveSequence {
                seq_id: req.seq_id,
                tokens: req.prompt_tokens,
                generated_count: 0,
                max_new_tokens: req.max_new_tokens,
                eos_token_id: req.eos_token_id,
            });
        }

        if self.running.is_empty() {
            return Ok(Vec::new());
        }

        let iter_idx = self.stats.iterations;
        self.stats.iterations += 1;
        self.stats.peak_running_seqs = self.stats.peak_running_seqs.max(self.running.len());

        // 2. Generate 1 token for every active sequence in a single batched step
        let next_tokens = generator.step_batch(&self.running)?;
        if next_tokens.len() != self.running.len() {
            return Err(Error::BufferLengthMismatch {
                expected: self.running.len(),
                actual: next_tokens.len(),
            });
        }

        // 3. Apply tokens and retire finished sequences immediately
        let mut events = Vec::with_capacity(self.running.len());
        let mut still_running = Vec::with_capacity(self.running.len());

        for (mut seq, next_tok) in self.running.drain(..).zip(next_tokens.into_iter()) {
            seq.tokens.push(next_tok);
            seq.generated_count += 1;
            self.stats.tokens_generated += 1;

            let hit_eos = seq.eos_token_id == Some(next_tok);
            let hit_limit = seq.generated_count >= seq.max_new_tokens;
            let finished = hit_eos || hit_limit;

            events.push(TokenStepOutput {
                seq_id: seq.seq_id,
                token_id: next_tok,
                finished,
                iteration: iter_idx,
            });

            if finished {
                let prompt_len = seq.tokens.len().saturating_sub(seq.generated_count);
                let generated = seq.tokens.get(prompt_len..).unwrap_or(&[]).to_vec();
                self.completed.insert(seq.seq_id, generated);
                self.stats.completed_sequences += 1;
            } else {
                still_running.push(seq);
            }
        }

        self.running = still_running;
        Ok(events)
    }

    /// Runs continuous batching iterations until all queued and active sequences finish.
    pub fn run_until_empty<G: StepGenerator>(
        &mut self,
        generator: &mut G,
    ) -> Result<Vec<TokenStepOutput>> {
        let mut all_events = Vec::new();
        while self.has_work() {
            let mut step_events = self.step(generator)?;
            all_events.append(&mut step_events);
        }
        Ok(all_events)
    }
}
