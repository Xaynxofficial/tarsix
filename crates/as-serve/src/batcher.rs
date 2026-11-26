//! Concurrent dynamic batching scheduler for [`InferenceSession`].

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use as_core::{Error, Result};
use as_graph::NodeId;
use as_runtime::InferenceSession;
use as_tensor::Tensor;

/// Configuration for [`DynamicBatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchConfig {
    /// Maximum number of concurrent requests combined into one batch.
    pub max_batch_size: usize,
    /// Maximum time to wait for additional requests once the first arrives (milliseconds).
    pub max_wait_ms: u64,
    /// Maximum queued requests allowed before applying backpressure.
    pub max_queue_depth: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            max_wait_ms: 5,
            max_queue_depth: 256,
        }
    }
}

/// Snapshot of [`DynamicBatcher`] throughput statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatcherStats {
    /// Total inference requests served.
    pub total_requests: u64,
    /// Total batches dispatched to the session.
    pub total_batches: u64,
}

struct PendingItem {
    inputs: BTreeMap<String, Tensor>,
    reply: mpsc::SyncSender<Result<HashMap<NodeId, Tensor>>>,
}

/// Background dynamic batching engine that coalesces concurrent inference calls.
pub struct DynamicBatcher {
    tx: Option<mpsc::SyncSender<PendingItem>>,
    worker: Option<JoinHandle<()>>,
    total_requests: Arc<AtomicU64>,
    total_batches: Arc<AtomicU64>,
}

impl DynamicBatcher {
    /// Spawns a [`DynamicBatcher`] worker owning `session`.
    pub fn start(session: InferenceSession, config: BatchConfig) -> Result<Self> {
        if config.max_batch_size == 0 || config.max_queue_depth == 0 {
            return Err(Error::InvalidArgument {
                context: "DynamicBatcher::start",
                reason: "max_batch_size and max_queue_depth must be >= 1".to_owned(),
            });
        }

        let (tx, rx) = mpsc::sync_channel::<PendingItem>(config.max_queue_depth);
        let total_requests = Arc::new(AtomicU64::new(0));
        let total_batches = Arc::new(AtomicU64::new(0));
        let req_counter = Arc::clone(&total_requests);
        let batch_counter = Arc::clone(&total_batches);

        let worker = thread::spawn(move || {
            let max_wait = Duration::from_millis(config.max_wait_ms);
            while let Ok(first) = rx.recv() {
                let mut batch = Vec::with_capacity(config.max_batch_size);
                batch.push(first);
                let deadline = Instant::now() + max_wait;

                while batch.len() < config.max_batch_size {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let rem = deadline.saturating_duration_since(now);
                    match rx.recv_timeout(rem) {
                        Ok(item) => batch.push(item),
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }

                let inputs: Vec<BTreeMap<String, Tensor>> =
                    batch.iter().map(|p| p.inputs.clone()).collect();
                let batch_len = inputs.len() as u64;
                let outcome = session.run_batched(&inputs);

                req_counter.fetch_add(batch_len, Ordering::Relaxed);
                batch_counter.fetch_add(1, Ordering::Relaxed);

                match outcome {
                    Ok(outputs) => {
                        for (item, out) in batch.into_iter().zip(outputs.into_iter()) {
                            let _ = item.reply.send(Ok(out));
                        }
                    }
                    Err(err) => {
                        for item in batch {
                            let _ = item.reply.send(Err(err.clone()));
                        }
                    }
                }
            }
        });

        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
            total_requests,
            total_batches,
        })
    }

    /// Enqueues a single inference request and waits for its batched result.
    pub fn infer(&self, inputs: BTreeMap<String, Tensor>) -> Result<HashMap<NodeId, Tensor>> {
        let tx = self.tx.as_ref().ok_or(Error::InternalInvariant {
            message: "DynamicBatcher already shut down",
        })?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        tx.try_send(PendingItem {
            inputs,
            reply: reply_tx,
        })
        .map_err(|_| Error::InvalidArgument {
            context: "DynamicBatcher::infer",
            reason: "inference batch queue full or closed".to_owned(),
        })?;
        reply_rx.recv().map_err(|_| Error::InternalInvariant {
            message: "DynamicBatcher worker dropped response channel",
        })?
    }

    /// Returns current [`BatcherStats`].
    pub fn stats(&self) -> BatcherStats {
        BatcherStats {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_batches: self.total_batches.load(Ordering::Relaxed),
        }
    }
}

impl Drop for DynamicBatcher {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}
