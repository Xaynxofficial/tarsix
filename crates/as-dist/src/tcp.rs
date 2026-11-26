//! Multi-machine TCP Ring communicator with integrity-checked `ASDC` binary framing.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use as_core::{Error, Result};
use as_tensor::Tensor;

use crate::comm::{
    AllReduceOp, Communicator, RingTransport, ring_all_gather, ring_all_reduce, ring_barrier,
    ring_broadcast,
};

const FRAME_MAGIC: &[u8; 4] = b"ASDC";

/// Default maximum single collective frame size (256 MiB) to guard against DoS OOM.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

/// Multi-machine TCP Ring communicator supporting Ring-AllReduce, Broadcast, AllGather, and Barrier.
pub struct TcpCommunicator {
    rank: usize,
    world_size: usize,
    max_frame_bytes: usize,
    send_stream: Mutex<TcpStream>,
    recv_stream: Mutex<TcpStream>,
}

impl TcpCommunicator {
    /// Connects a node into a TCP ring using an already-bound `listener` (accepting from `rank - 1`)
    /// and connecting outbound to `next_peer` (`rank + 1`).
    pub fn connect_ring(
        rank: usize,
        world_size: usize,
        listener: TcpListener,
        next_peer: SocketAddr,
        max_frame_bytes: usize,
    ) -> Result<Self> {
        if world_size < 2 || rank >= world_size {
            return Err(Error::InvalidArgument {
                context: "TcpCommunicator::connect_ring",
                reason: format!("invalid rank {rank} for world_size {world_size}"),
            });
        }

        // Perform accept and outbound connect concurrently to avoid deadlock
        let accept_handle = thread::spawn(move || listener.accept());

        let mut outbound = None;
        for _ in 0..50 {
            match TcpStream::connect_timeout(&next_peer, Duration::from_millis(200)) {
                Ok(s) => {
                    outbound = Some(s);
                    break;
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        }
        let send_stream = outbound.ok_or_else(|| Error::InvalidArgument {
            context: "TcpCommunicator::connect_ring",
            reason: format!("failed to connect to ring neighbor {next_peer}"),
        })?;
        let _ = send_stream.set_nodelay(true);

        let (recv_stream, _) = accept_handle
            .join()
            .map_err(|_| Error::InternalInvariant {
                message: "TCP accept thread panicked",
            })?
            .map_err(|e| Error::InvalidArgument {
                context: "TcpCommunicator::connect_ring",
                reason: format!("failed accepting ring predecessor: {e}"),
            })?;
        let _ = recv_stream.set_nodelay(true);

        Ok(Self {
            rank,
            world_size,
            max_frame_bytes,
            send_stream: Mutex::new(send_stream),
            recv_stream: Mutex::new(recv_stream),
        })
    }
}

impl RingTransport for TcpCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn ring_exchange(&self, send_buf: &[u8]) -> Result<Vec<u8>> {
        if send_buf.len() > self.max_frame_bytes {
            return Err(Error::AllocationLimitExceeded {
                requested_bytes: send_buf.len(),
                limit_bytes: self.max_frame_bytes,
            });
        }

        // Send and receive concurrently on separate TCP streams so large frames never deadlock on socket buffers
        let max_bytes = self.max_frame_bytes;
        let (send_res, recv_res) = thread::scope(|scope| {
            let s_handle = scope.spawn(|| {
                let mut stream = self.send_stream.lock().map_err(|_| Error::InternalInvariant {
                    message: "TcpCommunicator send mutex poisoned",
                })?;
                write_frame(&mut stream, send_buf)
            });
            let r_handle = scope.spawn(|| {
                let mut stream = self.recv_stream.lock().map_err(|_| Error::InternalInvariant {
                    message: "TcpCommunicator recv mutex poisoned",
                })?;
                read_frame(&mut stream, max_bytes)
            });
            let sr = s_handle.join().map_err(|_| Error::InternalInvariant {
                message: "TCP send thread panicked",
            });
            let rr = r_handle.join().map_err(|_| Error::InternalInvariant {
                message: "TCP recv thread panicked",
            });
            (sr, rr)
        });

        send_res??;
        recv_res?
    }
}

impl Communicator for TcpCommunicator {
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

fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<()> {
    let checksum = fnv1a_64(payload);
    let mut header = [0u8; 20];
    header[0..4].copy_from_slice(FRAME_MAGIC);
    header[4..12].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    header[12..20].copy_from_slice(&checksum.to_le_bytes());

    stream
        .write_all(&header)
        .and_then(|()| stream.write_all(payload))
        .and_then(|()| stream.flush())
        .map_err(|e| Error::InvalidArgument {
            context: "TcpCommunicator::write_frame",
            reason: format!("TCP frame write failed: {e}"),
        })
}

fn read_frame(stream: &mut TcpStream, max_frame_bytes: usize) -> Result<Vec<u8>> {
    let mut header = [0u8; 20];
    stream
        .read_exact(&mut header)
        .map_err(|e| Error::InvalidArgument {
            context: "TcpCommunicator::read_frame",
            reason: format!("TCP frame header read failed: {e}"),
        })?;
    if header.get(0..4).unwrap_or(&[]) != FRAME_MAGIC {
        return Err(Error::InvalidArgument {
            context: "TcpCommunicator::read_frame",
            reason: "invalid ASDC frame magic header".to_owned(),
        });
    }
    let len_arr: [u8; 8] = header
        .get(4..12)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::InternalInvariant {
            message: "frame len slice",
        })?;
    let chk_arr: [u8; 8] = header
        .get(12..20)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::InternalInvariant {
            message: "frame chk slice",
        })?;
    let payload_len = u64::from_le_bytes(len_arr) as usize;
    let expected_chk = u64::from_le_bytes(chk_arr);

    if payload_len > max_frame_bytes {
        return Err(Error::AllocationLimitExceeded {
            requested_bytes: payload_len,
            limit_bytes: max_frame_bytes,
        });
    }

    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        stream
            .read_exact(&mut payload)
            .map_err(|e| Error::InvalidArgument {
                context: "TcpCommunicator::read_frame",
                reason: format!("TCP frame payload read failed: {e}"),
            })?;
    }
    if fnv1a_64(&payload) != expected_chk {
        return Err(Error::InvalidArgument {
            context: "TcpCommunicator::read_frame",
            reason: "ASDC frame checksum mismatch (corrupted network packet)".to_owned(),
        });
    }
    Ok(payload)
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}
