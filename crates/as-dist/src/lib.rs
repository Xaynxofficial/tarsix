//! Zero-dependency, memory-safe Distributed Compute engine (`as-dist`).
//!
//! Provides:
//! - [`Communicator`] collective contract (`all_reduce`, `broadcast`, `all_gather`, `barrier`)
//! - Bandwidth-optimal **Ring-AllReduce** over any [`RingTransport`]
//! - [`LocalCluster`] / [`LocalCommunicator`] for multi-thread local distributed execution
//! - [`TcpCommunicator`] for multi-machine TCP clusters with integrity-checked `ASDC` framing
//! - [`DistributedDataParallel`] for parameter broadcast and gradient averaging across workers

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod comm;
pub mod ddp;
pub mod local;
pub mod tcp;

pub use comm::{AllReduceOp, Communicator, RingTransport};
pub use ddp::DistributedDataParallel;
pub use local::{LocalCluster, LocalCommunicator};
pub use tcp::{DEFAULT_MAX_FRAME_BYTES, TcpCommunicator};
