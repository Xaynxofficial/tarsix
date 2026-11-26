//! Distributed Data Parallel (DDP) parameter and gradient synchronization helpers.

use as_core::Result;
use as_nn::Module;
use as_tensor::Tensor;

use crate::comm::{AllReduceOp, Communicator};

/// Distributed Data Parallel (DDP) coordinator for synchronizing any [`Module`] across workers.
pub struct DistributedDataParallel<'a, C: Communicator> {
    comm: &'a C,
}

impl<'a, C: Communicator> DistributedDataParallel<'a, C> {
    /// Creates a new [`DistributedDataParallel`] helper bound to `comm`.
    pub const fn new(comm: &'a C) -> Self {
        Self { comm }
    }

    /// Broadcasts every trainable parameter of `model` from rank `0` to all ranks so
    /// every worker starts training from identical weights.
    pub fn broadcast_parameters(&self, model: &mut dyn Module) -> Result<()> {
        for param in model.parameters_mut() {
            let synced = self.comm.broadcast(param.data(), 0)?;
            param.set_data(synced);
        }
        Ok(())
    }

    /// Averages a slice of parameter gradients across all workers using [`AllReduceOp::Mean`].
    pub fn sync_gradients(&self, grads: &mut [Option<Tensor>]) -> Result<()> {
        for g_opt in grads.iter_mut() {
            if let Some(g) = g_opt {
                *g = self.comm.all_reduce(g, AllReduceOp::Mean)?;
            }
        }
        Ok(())
    }
}
