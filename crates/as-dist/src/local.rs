//! In-memory multi-thread ring communicator for local distributed execution and testing.

use std::sync::{Mutex, mpsc};

use as_core::{Error, Result};
use as_tensor::Tensor;

use crate::comm::{
    AllReduceOp, Communicator, RingTransport, ring_all_gather, ring_all_reduce, ring_barrier,
    ring_broadcast,
};

/// In-memory ring communicator connecting local worker threads.
pub struct LocalCommunicator {
    rank: usize,
    world_size: usize,
    send_next: Mutex<mpsc::Sender<Vec<u8>>>,
    recv_prev: Mutex<mpsc::Receiver<Vec<u8>>>,
}

impl RingTransport for LocalCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn ring_exchange(&self, send_buf: &[u8]) -> Result<Vec<u8>> {
        {
            let tx = self.send_next.lock().map_err(|_| Error::InternalInvariant {
                message: "LocalCommunicator send mutex poisoned",
            })?;
            tx.send(send_buf.to_vec()).map_err(|_| Error::InvalidArgument {
                context: "LocalCommunicator::ring_exchange",
                reason: "ring peer channel disconnected".to_owned(),
            })?;
        }
        let rx = self.recv_prev.lock().map_err(|_| Error::InternalInvariant {
            message: "LocalCommunicator recv mutex poisoned",
        })?;
        rx.recv().map_err(|_| Error::InvalidArgument {
            context: "LocalCommunicator::ring_exchange",
            reason: "ring peer closed before delivering frame".to_owned(),
        })
    }
}

impl Communicator for LocalCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn broadcast(&self, tensor: &Tensor, root: usize) -> Result<Tensor> {
        ring_broadcast(self, tensor, root)
    }

    fn all_reduce(&self, tensor: &Tensor, op: AllReduceOp) -> Result<Tensor> {
        ring_all_reduce(self, tensor, op)
    }

    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        ring_all_gather(self, tensor)
    }

    fn barrier(&self) -> Result<()> {
        ring_barrier(self)
    }
}

/// Factory for creating a connected ring of `world_size` [`LocalCommunicator`]s.
pub struct LocalCluster;

impl LocalCluster {
    /// Creates `world_size` connected [`LocalCommunicator`] handles (`0..world_size`).
    pub fn create(world_size: usize) -> Result<Vec<LocalCommunicator>> {
        if world_size == 0 {
            return Err(Error::InvalidArgument {
                context: "LocalCluster::create",
                reason: "world_size must be >= 1".to_owned(),
            });
        }
        let mut senders = Vec::with_capacity(world_size);
        let mut receivers = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            let (tx, rx) = mpsc::channel::<Vec<u8>>();
            senders.push(Some(tx));
            receivers.push(Some(rx));
        }

        let mut comms = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            // Rank `r` sends to `(r + 1) % P`, and receives from `r` (which was filled by `(r - 1) % P`)
            let next_rank = (rank + 1) % world_size;
            let tx = senders
                .get_mut(next_rank)
                .and_then(Option::take)
                .ok_or(Error::InternalInvariant {
                    message: "LocalCluster sender slot",
                })?;
            let rx = receivers
                .get_mut(rank)
                .and_then(Option::take)
                .ok_or(Error::InternalInvariant {
                    message: "LocalCluster receiver slot",
                })?;
            comms.push(LocalCommunicator {
                rank,
                world_size,
                send_next: Mutex::new(tx),
                recv_prev: Mutex::new(rx),
            });
        }
        Ok(comms)
    }
}
