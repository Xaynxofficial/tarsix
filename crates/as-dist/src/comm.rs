//! Collective communication traits and bandwidth-optimal Ring-AllReduce algorithm.

use as_core::{DType, Error, Result};
use as_tensor::Tensor;

/// Reduction operator applied during [`Communicator::all_reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceOp {
    /// Element-wise sum across all ranks.
    Sum,
    /// Element-wise arithmetic mean across all ranks.
    Mean,
    /// Element-wise maximum across all ranks.
    Max,
    /// Element-wise minimum across all ranks.
    Min,
}

/// Collective communication contract across `world_size` workers.
pub trait Communicator: Send + Sync {
    /// Zero-based rank of this worker (`0..world_size`).
    fn rank(&self) -> usize;

    /// Total number of workers in the communicator group.
    fn world_size(&self) -> usize;

    /// Broadcasts `tensor` from `root` rank to all other ranks.
    fn broadcast(&self, tensor: &Tensor, root: usize) -> Result<Tensor>;

    /// Reduces `tensor` across all ranks using `op` and distributes the result to every rank.
    fn all_reduce(&self, tensor: &Tensor, op: AllReduceOp) -> Result<Tensor>;

    /// Gathers tensors from all ranks (ordered `0..world_size`) onto every rank.
    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>>;

    /// Synchronizes all ranks until every worker reaches this barrier.
    fn barrier(&self) -> Result<()>;
}

/// Point-to-point ring neighbor transport used to implement Ring-AllReduce.
pub trait RingTransport: Send + Sync {
    /// Zero-based rank in the ring.
    fn rank(&self) -> usize;

    /// Total number of nodes in the ring.
    fn world_size(&self) -> usize;

    /// Simultaneously sends `send_buf` to `(rank + 1) % world_size` and receives
    /// a payload from `(rank + world_size - 1) % world_size` without deadlocking.
    fn ring_exchange(&self, send_buf: &[u8]) -> Result<Vec<u8>>;
}

/// Bandwidth-optimal Ring-AllReduce (`2 * (P - 1) / P` volume) over any [`RingTransport`].
pub fn ring_all_reduce<T: RingTransport>(
    transport: &T,
    tensor: &Tensor,
    op: AllReduceOp,
) -> Result<Tensor> {
    let p = transport.world_size();
    let rank = transport.rank();
    if p <= 1 {
        return Ok(tensor.clone());
    }

    let orig_dtype = tensor.dtype();
    let orig_dims = tensor.dims().to_vec();
    let orig_numel = tensor.numel();
    let mut data = tensor.to_dtype(DType::F64)?.to_f64_vec()?;

    // Pad to a multiple of world_size so every ring chunk has equal length
    let chunk_len = orig_numel.div_ceil(p).max(1);
    let padded_len = chunk_len.checked_mul(p).ok_or(Error::ArithmeticOverflow {
        context: "ring_all_reduce padded_len",
    })?;
    data.resize(padded_len, 0.0);

    // Phase 1: Scatter-Reduce (P - 1 steps)
    for step in 0..(p - 1) {
        let send_chunk_idx = (rank + p - step) % p;
        let recv_chunk_idx = (rank + p - step - 1) % p;

        let send_slice = data
            .get(send_chunk_idx * chunk_len..(send_chunk_idx + 1) * chunk_len)
            .ok_or(Error::InternalInvariant {
                message: "ring_all_reduce scatter send slice",
            })?;
        let send_bytes = encode_f64_slice(send_slice);
        let recv_bytes = transport.ring_exchange(&send_bytes)?;
        let recv_vals = decode_f64_slice(&recv_bytes, chunk_len)?;

        let dst_slice = data
            .get_mut(recv_chunk_idx * chunk_len..(recv_chunk_idx + 1) * chunk_len)
            .ok_or(Error::InternalInvariant {
                message: "ring_all_reduce scatter recv slice",
            })?;
        for (dst, &src) in dst_slice.iter_mut().zip(recv_vals.iter()) {
            *dst = apply_reduce(*dst, src, op);
        }
    }

    // Phase 2: All-Gather (P - 1 steps)
    for step in 0..(p - 1) {
        let send_chunk_idx = (rank + p - step + 1) % p;
        let recv_chunk_idx = (rank + p - step) % p;

        let send_slice = data
            .get(send_chunk_idx * chunk_len..(send_chunk_idx + 1) * chunk_len)
            .ok_or(Error::InternalInvariant {
                message: "ring_all_reduce gather send slice",
            })?;
        let send_bytes = encode_f64_slice(send_slice);
        let recv_bytes = transport.ring_exchange(&send_bytes)?;
        let recv_vals = decode_f64_slice(&recv_bytes, chunk_len)?;

        let dst_slice = data
            .get_mut(recv_chunk_idx * chunk_len..(recv_chunk_idx + 1) * chunk_len)
            .ok_or(Error::InternalInvariant {
                message: "ring_all_reduce gather recv slice",
            })?;
        dst_slice.copy_from_slice(&recv_vals);
    }

    data.truncate(orig_numel);
    if op == AllReduceOp::Mean {
        let inv = 1.0 / (p as f64);
        for v in &mut data {
            *v *= inv;
        }
    }

    let out = Tensor::from_f64(&data, orig_dims)?;
    if orig_dtype == DType::F64 {
        Ok(out)
    } else {
        out.to_dtype(orig_dtype)
    }
}

/// Ring-based broadcast of `tensor` from `root` to all ranks.
pub fn ring_broadcast<T: RingTransport>(
    transport: &T,
    tensor: &Tensor,
    root: usize,
) -> Result<Tensor> {
    let p = transport.world_size();
    let rank = transport.rank();
    if root >= p {
        return Err(Error::InvalidArgument {
            context: "ring_broadcast",
            reason: format!("root rank {root} >= world_size {p}"),
        });
    }
    if p <= 1 {
        return Ok(tensor.clone());
    }

    let orig_dtype = tensor.dtype();
    let orig_dims = tensor.dims().to_vec();
    let orig_numel = tensor.numel();
    let mut payload = if rank == root {
        let vals = tensor.to_dtype(DType::F64)?.to_f64_vec()?;
        encode_f64_slice(&vals)
    } else {
        vec![0u8; orig_numel * 8]
    };
    let dist = (rank + p - root) % p;
    for step in 0..(p - 1) {
        let recv = transport.ring_exchange(&payload)?;
        if step + 1 == dist {
            payload = recv;
        }
    }

    let vals = decode_f64_slice(&payload, orig_numel)?;
    let out = Tensor::from_f64(&vals, orig_dims)?;
    if orig_dtype == DType::F64 {
        Ok(out)
    } else {
        out.to_dtype(orig_dtype)
    }
}

/// Ring-based All-Gather returning `[rank_0, rank_1, ..., rank_{P-1}]` on every rank.
pub fn ring_all_gather<T: RingTransport>(transport: &T, tensor: &Tensor) -> Result<Vec<Tensor>> {
    let p = transport.world_size();
    let rank = transport.rank();
    if p <= 1 {
        return Ok(vec![tensor.clone()]);
    }

    let orig_dtype = tensor.dtype();
    let orig_dims = tensor.dims().to_vec();
    let orig_numel = tensor.numel();
    let my_vals = tensor.to_dtype(DType::F64)?.to_f64_vec()?;

    let mut slots: Vec<Option<Vec<u8>>> = vec![None; p];
    if let Some(s) = slots.get_mut(rank) {
        *s = Some(encode_f64_slice(&my_vals));
    }

    for step in 0..(p - 1) {
        let send_idx = (rank + p - step) % p;
        let recv_idx = (rank + p - step - 1) % p;
        let send_buf = slots
            .get(send_idx)
            .and_then(Option::as_deref)
            .ok_or(Error::InternalInvariant {
                message: "ring_all_gather missing send slot",
            })?;
        let recv_buf = transport.ring_exchange(send_buf)?;
        if let Some(s) = slots.get_mut(recv_idx) {
            *s = Some(recv_buf);
        }
    }

    let mut out = Vec::with_capacity(p);
    for slot in slots {
        let bytes = slot.ok_or(Error::InternalInvariant {
            message: "ring_all_gather missing final slot",
        })?;
        let vals = decode_f64_slice(&bytes, orig_numel)?;
        let t = Tensor::from_f64(&vals, orig_dims.clone())?;
        if orig_dtype == DType::F64 {
            out.push(t);
        } else {
            out.push(t.to_dtype(orig_dtype)?);
        }
    }
    Ok(out)
}

/// Ring barrier synchronizing all ranks.
pub fn ring_barrier<T: RingTransport>(transport: &T) -> Result<()> {
    let p = transport.world_size();
    if p <= 1 {
        return Ok(());
    }
    let token = [0xA5u8; 8];
    for _ in 0..p {
        let _ = transport.ring_exchange(&token)?;
    }
    Ok(())
}

#[inline]
fn apply_reduce(a: f64, b: f64, op: AllReduceOp) -> f64 {
    match op {
        AllReduceOp::Sum | AllReduceOp::Mean => a + b,
        AllReduceOp::Max => a.max(b),
        AllReduceOp::Min => a.min(b),
    }
}

fn encode_f64_slice(vals: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn decode_f64_slice(bytes: &[u8], expected_len: usize) -> Result<Vec<f64>> {
    if bytes.len() != expected_len * 8 {
        return Err(Error::BufferLengthMismatch {
            expected: expected_len * 8,
            actual: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        let slice = bytes
            .get(i * 8..(i + 1) * 8)
            .ok_or(Error::InternalInvariant {
                message: "decode_f64_slice bounds",
            })?;
        let arr: [u8; 8] = slice.try_into().map_err(|_| Error::InternalInvariant {
            message: "decode_f64_slice array",
        })?;
        out.push(f64::from_bits(u64::from_le_bytes(arr)));
    }
    Ok(out)
}
