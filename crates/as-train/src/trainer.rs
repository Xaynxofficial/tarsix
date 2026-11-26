//! Declarative Training Pipeline Engine (`Trainer`, `TrainConfig`, `Callback`, `TrainState`).
//!
//! Provides a reusable, panic-free training loop supporting:
//! - Microbatch gradient accumulation (`grad_accum_steps`)
//! - Global L2 gradient clipping (`max_grad_norm`)
//! - Learning-rate scheduling (`LrScheduler`)
//! - Validation evaluation & `MetricsTracker` integration
//! - Pluggable [`Callback`] hooks ([`EarlyStopping`], [`BestCheckpointCallback`], [`JsonlLoggerCallback`])
//! - Checksummed full-resume [`TrainState`] serialization

use std::fmt::Write as _;

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, MemoryLimits, Result};
use as_nn::{Module, Rng};
use as_optim::{LrScheduler, Optimizer, clip_grad_norm};
use as_tensor::Tensor;

use crate::checkpoint::{load_checkpoint, save_checkpoint};
use crate::data::{Batch, DataLoader, Dataset};
use crate::metrics::{EpochMetrics, MetricsTracker};

const STATE_MAGIC: &[u8; 4] = b"ASTS";
const STATE_VERSION: u32 = 1;

/// Configuration for [`Trainer`].
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Total number of training epochs.
    pub epochs: usize,
    /// Number of samples per microbatch.
    pub batch_size: usize,
    /// Number of microbatches accumulated before each optimizer step (`>= 1`).
    pub grad_accum_steps: usize,
    /// Optional global L2 gradient norm clipping threshold.
    pub max_grad_norm: Option<f64>,
    /// Whether to shuffle training data each epoch.
    pub shuffle: bool,
    /// Base deterministic PRNG seed for epoch shuffling.
    pub seed: u64,
    /// Data type used by [`DataLoader`] feature tensors.
    pub dtype: DType,
    /// Run validation every `eval_every` epochs (`>= 1`).
    pub eval_every: usize,
    /// Optional loss scaling factor (`1.0` = disabled).
    pub loss_scale: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            grad_accum_steps: 1,
            max_grad_norm: Some(1.0),
            shuffle: true,
            seed: 42,
            dtype: DType::F32,
            eval_every: 1,
            loss_scale: 1.0,
        }
    }
}

/// Telemetry emitted after each optimizer update step.
#[derive(Debug, Clone, Copy)]
pub struct StepContext {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Monotonic global optimizer step count.
    pub global_step: usize,
    /// Mean loss over the accumulated microbatches for this step.
    pub loss: f64,
    /// Active learning rate.
    pub learning_rate: f64,
    /// Global L2 gradient norm before clipping (if clipping enabled).
    pub grad_norm: Option<f64>,
}

/// Telemetry emitted at the end of each epoch.
#[derive(Debug, Clone, Copy)]
pub struct EpochContext {
    /// Completed epoch metrics.
    pub metrics: EpochMetrics,
    /// Current global optimizer step.
    pub global_step: usize,
    /// Current learning rate.
    pub learning_rate: f64,
}

/// Control decision returned by [`Callback::on_epoch_end`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training normally.
    Continue,
    /// Halt training gracefully after the current epoch.
    StopEarly {
        /// Human-readable reason for stopping early.
        reason: String,
    },
}

/// Lifecycle hook trait for customizing [`Trainer`] behavior.
pub trait Callback {
    /// Invoked once before epoch 0 starts.
    fn on_train_begin(&mut self, _config: &TrainConfig) {}

    /// Invoked after every optimizer parameter update.
    fn on_step_end(&mut self, _ctx: &StepContext) {}

    /// Invoked after each epoch completes; may request early stopping.
    fn on_epoch_end(&mut self, _ctx: &EpochContext, _model: &dyn Module) -> Result<CallbackAction> {
        Ok(CallbackAction::Continue)
    }

    /// Invoked once after training finishes (or stops early).
    fn on_train_end(&mut self, _history: &[EpochMetrics]) {}
}

/// Built-in callback that halts training when monitored loss stops improving.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best_loss: f64,
    stale_epochs: usize,
}

impl EarlyStopping {
    /// Creates an [`EarlyStopping`] callback with `patience` epochs and `min_delta` improvement threshold.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience: patience.max(1),
            min_delta: min_delta.max(0.0),
            best_loss: f64::INFINITY,
            stale_epochs: 0,
        }
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(&mut self, ctx: &EpochContext, _model: &dyn Module) -> Result<CallbackAction> {
        let monitored = ctx.metrics.val_loss.unwrap_or(ctx.metrics.train_loss);
        if monitored + self.min_delta < self.best_loss {
            self.best_loss = monitored;
            self.stale_epochs = 0;
            Ok(CallbackAction::Continue)
        } else {
            self.stale_epochs += 1;
            if self.stale_epochs >= self.patience {
                Ok(CallbackAction::StopEarly {
                    reason: format!(
                        "early stopping: no improvement for {} epochs (best loss {:.6})",
                        self.patience, self.best_loss
                    ),
                })
            } else {
                Ok(CallbackAction::Continue)
            }
        }
    }
}

/// Built-in callback that retains the best integrity-checked `ASCK` model checkpoint in memory.
#[derive(Debug, Clone, Default)]
pub struct BestCheckpointCallback {
    best_loss: f64,
    best_epoch: Option<usize>,
    checkpoint_bytes: Option<Vec<u8>>,
}

impl BestCheckpointCallback {
    /// Creates a new [`BestCheckpointCallback`].
    pub fn new() -> Self {
        Self {
            best_loss: f64::INFINITY,
            best_epoch: None,
            checkpoint_bytes: None,
        }
    }

    /// Returns the best captured `ASCK` checkpoint bytes, if any.
    pub fn best_bytes(&self) -> Option<&[u8]> {
        self.checkpoint_bytes.as_deref()
    }

    /// Returns the epoch at which the best checkpoint was captured.
    pub const fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }
}

impl Callback for BestCheckpointCallback {
    fn on_epoch_end(&mut self, ctx: &EpochContext, model: &dyn Module) -> Result<CallbackAction> {
        let monitored = ctx.metrics.val_loss.unwrap_or(ctx.metrics.train_loss);
        if monitored < self.best_loss {
            self.best_loss = monitored;
            self.best_epoch = Some(ctx.metrics.epoch);
            let params = model.parameters();
            self.checkpoint_bytes = Some(save_checkpoint(&params)?);
        }
        Ok(CallbackAction::Continue)
    }
}

/// Built-in callback that records structured JSON Lines (JSONL) epoch telemetry.
#[derive(Debug, Clone, Default)]
pub struct JsonlLoggerCallback {
    lines: Vec<String>,
}

impl JsonlLoggerCallback {
    /// Creates an empty [`JsonlLoggerCallback`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all recorded JSONL log lines.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Joins all recorded lines into a single newline-delimited JSONL string.
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for line in &self.lines {
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

impl Callback for JsonlLoggerCallback {
    fn on_epoch_end(&mut self, ctx: &EpochContext, _model: &dyn Module) -> Result<CallbackAction> {
        let mut line = String::new();
        let _ = write!(
            line,
            "{{\"epoch\":{},\"step\":{},\"lr\":{:.8},\"train_loss\":{:.6}",
            ctx.metrics.epoch, ctx.global_step, ctx.learning_rate, ctx.metrics.train_loss
        );
        if let Some(vl) = ctx.metrics.val_loss {
            let _ = write!(line, ",\"val_loss\":{vl:.6}");
        }
        if let Some(va) = ctx.metrics.val_accuracy {
            let _ = write!(line, ",\"val_accuracy\":{va:.6}");
        }
        line.push('}');
        self.lines.push(line);
        Ok(CallbackAction::Continue)
    }
}

/// Complete summary returned by [`Trainer::fit`].
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Epoch-by-epoch training and validation metrics.
    pub history: Vec<EpochMetrics>,
    /// Whether training terminated early via a callback.
    pub stopped_early: bool,
    /// Optional reason string if `stopped_early == true`.
    pub stop_reason: Option<String>,
    /// Lowest monitored loss observed across epochs.
    pub best_loss: f64,
    /// Zero-based epoch achieving `best_loss`.
    pub best_epoch: usize,
    /// Total optimizer update steps executed.
    pub total_steps: usize,
    /// Checksummed `ASCK` checkpoint captured at `best_epoch`.
    pub best_checkpoint: Option<Vec<u8>>,
}

/// Serializable, checksum-verified full training state for resuming interrupted runs.
#[derive(Debug, Clone)]
pub struct TrainState {
    /// Next epoch index to resume from.
    pub next_epoch: usize,
    /// Monotonic global optimizer step count.
    pub global_step: usize,
    /// Best monitored loss observed so far.
    pub best_loss: f64,
    /// Deterministic PRNG seed.
    pub rng_seed: u64,
    /// Embedded `ASCK` model parameter checkpoint bytes.
    pub model_checkpoint: Vec<u8>,
}

impl TrainState {
    /// Serializes [`TrainState`] into an integrity-checked `ASTS` binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(36 + self.model_checkpoint.len());
        payload.extend_from_slice(&(self.next_epoch as u64).to_le_bytes());
        payload.extend_from_slice(&(self.global_step as u64).to_le_bytes());
        payload.extend_from_slice(&self.best_loss.to_bits().to_le_bytes());
        payload.extend_from_slice(&self.rng_seed.to_le_bytes());
        payload.extend_from_slice(&(self.model_checkpoint.len() as u64).to_le_bytes());
        payload.extend_from_slice(&self.model_checkpoint);

        let checksum = fnv1a_64(&payload);
        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// Safely deserializes and verifies an untrusted `ASTS` training state byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 + 40 {
            return Err(Error::InvalidArgument {
                context: "TrainState::from_bytes",
                reason: "truncated training state buffer".to_owned(),
            });
        }
        let magic = bytes.get(0..4).unwrap_or(&[]);
        if magic != STATE_MAGIC {
            return Err(Error::InvalidArgument {
                context: "TrainState::from_bytes",
                reason: "invalid training state magic header".to_owned(),
            });
        }
        let ver_arr: [u8; 4] = bytes
            .get(4..8)
            .and_then(|s| s.try_into().ok())
            .ok_or(Error::InternalInvariant {
                message: "TrainState version slice",
            })?;
        let version = u32::from_le_bytes(ver_arr);
        if version != STATE_VERSION {
            return Err(Error::InvalidArgument {
                context: "TrainState::from_bytes",
                reason: format!("unsupported training state version {version}"),
            });
        }
        let chk_arr: [u8; 8] = bytes
            .get(8..16)
            .and_then(|s| s.try_into().ok())
            .ok_or(Error::InternalInvariant {
                message: "TrainState checksum slice",
            })?;
        let expected_chk = u64::from_le_bytes(chk_arr);
        let payload = bytes.get(16..).unwrap_or(&[]);
        if fnv1a_64(payload) != expected_chk {
            return Err(Error::InvalidArgument {
                context: "TrainState::from_bytes",
                reason: "training state integrity checksum mismatch".to_owned(),
            });
        }

        let next_epoch = read_u64_at(payload, 0)? as usize;
        let global_step = read_u64_at(payload, 8)? as usize;
        let best_loss = f64::from_bits(read_u64_at(payload, 16)?);
        let rng_seed = read_u64_at(payload, 24)?;
        let ckpt_len = read_u64_at(payload, 32)? as usize;
        let ckpt_end = 40usize.checked_add(ckpt_len).ok_or(Error::ArithmeticOverflow {
            context: "TrainState::from_bytes ckpt_end",
        })?;
        if ckpt_end != payload.len() {
            return Err(Error::InvalidArgument {
                context: "TrainState::from_bytes",
                reason: "checkpoint length mismatch in training state".to_owned(),
            });
        }
        let model_checkpoint = payload.get(40..ckpt_end).unwrap_or(&[]).to_vec();

        Ok(Self {
            next_epoch,
            global_step,
            best_loss,
            rng_seed,
            model_checkpoint,
        })
    }

    /// Restores the embedded `ASCK` model weights into `model`.
    pub fn restore_model(&self, model: &mut dyn Module, limits: MemoryLimits) -> Result<()> {
        let mut params = model.parameters_mut();
        load_checkpoint(&self.model_checkpoint, &mut params, limits)
    }
}

/// Declarative, reproducible training pipeline orchestrator.
pub struct Trainer {
    config: TrainConfig,
    callbacks: Vec<Box<dyn Callback>>,
    start_epoch: usize,
    global_step: usize,
    best_loss: f64,
}

impl Trainer {
    /// Creates a new [`Trainer`] with `config`.
    pub fn new(config: TrainConfig) -> Result<Self> {
        if config.batch_size == 0 {
            return Err(Error::InvalidArgument {
                context: "Trainer::new",
                reason: "batch_size must be >= 1".to_owned(),
            });
        }
        if config.grad_accum_steps == 0 {
            return Err(Error::InvalidArgument {
                context: "Trainer::new",
                reason: "grad_accum_steps must be >= 1".to_owned(),
            });
        }
        if config.loss_scale <= 0.0 || !config.loss_scale.is_finite() {
            return Err(Error::InvalidArgument {
                context: "Trainer::new",
                reason: "loss_scale must be finite and > 0".to_owned(),
            });
        }
        Ok(Self {
            config,
            callbacks: Vec::new(),
            start_epoch: 0,
            global_step: 0,
            best_loss: f64::INFINITY,
        })
    }

    /// Attaches a [`Callback`] to the training lifecycle.
    pub fn add_callback(&mut self, callback: Box<dyn Callback>) -> &mut Self {
        self.callbacks.push(callback);
        self
    }

    /// Resumes trainer counters and model parameters from a verified [`TrainState`].
    pub fn resume_from(
        &mut self,
        state: &TrainState,
        model: &mut dyn Module,
        limits: MemoryLimits,
    ) -> Result<()> {
        state.restore_model(model, limits)?;
        self.start_epoch = state.next_epoch;
        self.global_step = state.global_step;
        self.best_loss = state.best_loss;
        self.config.seed = state.rng_seed;
        Ok(())
    }

    /// Captures a resumable [`TrainState`] snapshot for `model`.
    pub fn snapshot_state(&self, model: &dyn Module, completed_epoch: usize) -> Result<TrainState> {
        let params = model.parameters();
        let model_checkpoint = save_checkpoint(&params)?;
        Ok(TrainState {
            next_epoch: completed_epoch + 1,
            global_step: self.global_step,
            best_loss: self.best_loss,
            rng_seed: self.config.seed,
            model_checkpoint,
        })
    }

    /// Runs the full training loop over `train_data` (and optional `val_data`).
    ///
    /// - `step_fn(model, tape, batch)` must return `(loss_var, param_var_ids)` matching `model.parameters()` order.
    /// - `eval_fn(model, batch)` (optional) returns `(val_loss, optional_accuracy)`.
    #[allow(clippy::too_many_arguments)]
    pub fn fit<M, O, F, E>(
        &mut self,
        model: &mut M,
        optimizer: &mut O,
        train_data: &dyn Dataset,
        val_data: Option<&dyn Dataset>,
        lr_scheduler: Option<&dyn LrScheduler>,
        step_fn: F,
        eval_fn: Option<E>,
    ) -> Result<TrainReport>
    where
        M: Module,
        O: Optimizer,
        F: Fn(&M, &mut Tape, &Batch) -> Result<(VarId, Vec<VarId>)>,
        E: Fn(&M, &Batch) -> Result<(f64, Option<f64>)>,
    {
        let train_loader = DataLoader::new(
            self.config.batch_size,
            self.config.shuffle,
            false,
            self.config.dtype,
        )?;
        let val_loader = DataLoader::new(self.config.batch_size, false, false, self.config.dtype)?;

        let mut tracker = MetricsTracker::new();
        let mut stopped_early = false;
        let mut stop_reason = None;
        let mut best_epoch = self.start_epoch;
        let mut best_checkpoint: Option<Vec<u8>> = None;

        for cb in &mut self.callbacks {
            cb.on_train_begin(&self.config);
        }

        let num_params = model.parameters().len();

        for epoch in self.start_epoch..self.config.epochs {
            let epoch_seed = self.config.seed.wrapping_add((epoch as u64).wrapping_mul(0x9e37_79b9));
            let mut epoch_rng = Rng::from_seed(epoch_seed);
            let batches = train_loader.epoch_batches(train_data, &mut epoch_rng)?;
            if batches.is_empty() {
                break;
            }

            let mut epoch_loss_sum = 0.0f64;
            let mut epoch_batches = 0usize;

            let mut accum_grads: Vec<Option<Tensor>> = vec![None; num_params];
            let mut accum_count = 0usize;
            let mut accum_loss_sum = 0.0f64;

            for (batch_idx, batch) in batches.iter().enumerate() {
                let mut tape = Tape::new();
                let (loss_var, param_vars) = step_fn(model, &mut tape, batch)?;
                if param_vars.len() != num_params {
                    return Err(Error::BufferLengthMismatch {
                        expected: num_params,
                        actual: param_vars.len(),
                    });
                }

                let raw_loss = tape.value(loss_var)?.to_dtype(DType::F64)?.to_f64_vec()?;
                let loss_val = raw_loss.first().copied().unwrap_or(0.0);
                epoch_loss_sum += loss_val;
                epoch_batches += 1;
                accum_loss_sum += loss_val;

                let scaled_loss_var = if (self.config.loss_scale - 1.0).abs() > f64::EPSILON {
                    let s = tape.var(
                        Tensor::full(vec![], self.config.loss_scale, self.config.dtype)?,
                        false,
                    );
                    tape.mul(loss_var, s)?
                } else {
                    loss_var
                };

                let grads = tape.backward(scaled_loss_var)?;
                for (i, &pvar) in param_vars.iter().enumerate() {
                    if let Some(g) = grads.get(pvar) {
                        if let Some(slot) = accum_grads.get_mut(i) {
                            match slot {
                                Some(existing) => {
                                    *existing = existing.add(g)?;
                                }
                                None => {
                                    *slot = Some(g.clone());
                                }
                            }
                        }
                    }
                }
                accum_count += 1;

                let is_last_batch = batch_idx + 1 == batches.len();
                if accum_count >= self.config.grad_accum_steps || is_last_batch {
                    let inv_scale = 1.0 / ((accum_count as f64) * self.config.loss_scale);
                    if (inv_scale - 1.0).abs() > f64::EPSILON {
                        for g_opt in &mut accum_grads {
                            if let Some(g) = g_opt {
                                let factor = Tensor::full(vec![], inv_scale, g.dtype())?;
                                *g = g.mul(&factor)?;
                            }
                        }
                    }

                    let grad_norm = if let Some(max_norm) = self.config.max_grad_norm {
                        let mut dense: Vec<Tensor> =
                            accum_grads.iter().filter_map(Clone::clone).collect();
                        let norm = clip_grad_norm(&mut dense, max_norm)?;
                        let mut iter = dense.into_iter();
                        for g_opt in &mut accum_grads {
                            if g_opt.is_some() {
                                *g_opt = iter.next();
                            }
                        }
                        Some(norm)
                    } else {
                        None
                    };

                    if let Some(sched) = lr_scheduler {
                        optimizer.set_learning_rate(sched.lr_at(self.global_step));
                    }

                    {
                        let grad_refs: Vec<Option<&Tensor>> =
                            accum_grads.iter().map(Option::as_ref).collect();
                        let mut params_mut = model.parameters_mut();
                        optimizer.step(&mut params_mut, &grad_refs)?;
                    }

                    let step_ctx = StepContext {
                        epoch,
                        global_step: self.global_step,
                        loss: accum_loss_sum / (accum_count as f64),
                        learning_rate: optimizer.learning_rate(),
                        grad_norm,
                    };
                    for cb in &mut self.callbacks {
                        cb.on_step_end(&step_ctx);
                    }

                    self.global_step += 1;
                    accum_grads = vec![None; num_params];
                    accum_count = 0;
                    accum_loss_sum = 0.0;
                }
            }

            let train_loss = epoch_loss_sum / (epoch_batches.max(1) as f64);

            // Optional validation pass
            let (val_loss, val_accuracy) = if let (Some(vdata), Some(efn)) = (val_data, &eval_fn)
                && epoch % self.config.eval_every.max(1) == 0
            {
                let mut val_rng = Rng::from_seed(self.config.seed);
                let vbatches = val_loader.epoch_batches(vdata, &mut val_rng)?;
                let mut vl_sum = 0.0f64;
                let mut va_sum = 0.0f64;
                let mut va_cnt = 0usize;
                for vb in &vbatches {
                    let (vl, va_opt) = efn(model, vb)?;
                    vl_sum += vl;
                    if let Some(va) = va_opt {
                        va_sum += va;
                        va_cnt += 1;
                    }
                }
                let mean_vl = if vbatches.is_empty() {
                    None
                } else {
                    Some(vl_sum / (vbatches.len() as f64))
                };
                let mean_va = if va_cnt == 0 {
                    None
                } else {
                    Some(va_sum / (va_cnt as f64))
                };
                (mean_vl, mean_va)
            } else {
                (None, None)
            };

            let metrics = EpochMetrics {
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
            };
            tracker.record(metrics);

            let monitored = val_loss.unwrap_or(train_loss);
            if monitored < self.best_loss {
                self.best_loss = monitored;
                best_epoch = epoch;
                let params = model.parameters();
                best_checkpoint = Some(save_checkpoint(&params)?);
            }

            let epoch_ctx = EpochContext {
                metrics,
                global_step: self.global_step,
                learning_rate: optimizer.learning_rate(),
            };
            for cb in &mut self.callbacks {
                if let CallbackAction::StopEarly { reason } = cb.on_epoch_end(&epoch_ctx, model)? {
                    stopped_early = true;
                    stop_reason = Some(reason);
                    break;
                }
            }
            if stopped_early {
                break;
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end(tracker.history());
        }

        Ok(TrainReport {
            history: tracker.history().to_vec(),
            stopped_early,
            stop_reason,
            best_loss: self.best_loss,
            best_epoch,
            total_steps: self.global_step,
            best_checkpoint,
        })
    }
}

fn read_u64_at(buf: &[u8], offset: usize) -> Result<u64> {
    let slice = buf.get(offset..offset + 8).ok_or(Error::InvalidArgument {
        context: "TrainState::from_bytes",
        reason: "unexpected EOF reading u64".to_owned(),
    })?;
    let arr: [u8; 8] = slice.try_into().map_err(|_| Error::InternalInvariant {
        message: "read_u64_at slice length",
    })?;
    Ok(u64::from_le_bytes(arr))
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}
