//! Integrity-checked native binary checkpoint format (`ASCK` v1).
//!
//! Treats loaded checkpoint bytes as untrusted data:
//! - Magic header (`ASCK`) & version verification
//! - FNV-1a 64-bit integrity checksum verification
//! - Strict bounds, rank, parameter count, and shape-matching validation

use std::collections::HashMap;

use as_core::{DType, Error, MAX_RANK, MemoryLimits, Result, Shape};
use as_nn::Param;
use as_tensor::Tensor;

const MAGIC: &[u8; 4] = b"ASCK";
const VERSION: u32 = 1;
const MAX_PARAMS: usize = 100_000;
const MAX_NAME_LEN: usize = 4096;

/// Serializes model parameters into an integrity-checked binary checkpoint.
pub fn save_checkpoint(params: &[&Param]) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    write_u32(&mut payload, params.len() as u32);
    for p in params {
        let name_bytes = p.name().as_bytes();
        write_u32(&mut payload, name_bytes.len() as u32);
        payload.extend_from_slice(name_bytes);

        let dims = p.shape().dims();
        write_u32(&mut payload, dims.len() as u32);
        for &d in dims {
            payload.extend_from_slice(&(d as u64).to_le_bytes());
        }

        payload.push(dtype_tag(p.dtype()));

        let data_f64 = p.data().to_dtype(DType::F64)?.to_f64_vec()?;
        write_u32(&mut payload, data_f64.len() as u32);
        for v in data_f64 {
            payload.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }

    let checksum = fnv1a_64(&payload);
    let mut out = Vec::with_capacity(16 + payload.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Inspects parameter names, shapes, and dtypes from an integrity-checked checkpoint
/// without requiring pre-allocated destination parameters.
pub fn inspect_checkpoint_shapes(
    bytes: &[u8],
    limits: MemoryLimits,
) -> Result<HashMap<String, (Shape, DType)>> {
    let mut reader = SafeReader::new(bytes);
    let magic = reader.read_bytes(4)?;
    if magic != MAGIC {
        return Err(Error::InvalidArgument {
            context: "inspect_checkpoint_shapes",
            reason: "invalid checkpoint magic header".to_owned(),
        });
    }
    let version = reader.read_u32()?;
    if version != VERSION {
        return Err(Error::InvalidArgument {
            context: "inspect_checkpoint_shapes",
            reason: format!("unsupported checkpoint version {version}"),
        });
    }
    let expected_checksum = reader.read_u64()?;
    if fnv1a_64(reader.remaining()) != expected_checksum {
        return Err(Error::InvalidArgument {
            context: "inspect_checkpoint_shapes",
            reason: "checkpoint integrity checksum mismatch".to_owned(),
        });
    }
    let count = reader.read_u32()? as usize;
    if count > MAX_PARAMS {
        return Err(Error::InvalidArgument {
            context: "inspect_checkpoint_shapes",
            reason: format!("parameter count {count} exceeds limit {MAX_PARAMS}"),
        });
    }
    let mut out = HashMap::with_capacity(count.min(1024));
    for _ in 0..count {
        let name = reader.read_string()?;
        let rank = reader.read_u32()? as usize;
        if rank > MAX_RANK {
            return Err(Error::InvalidRank {
                context: "inspect_checkpoint_shapes",
                rank,
                expected: MAX_RANK,
            });
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(reader.read_u64()? as usize);
        }
        let shape = Shape::new(dims)?;
        let dtype = reader.read_dtype()?;
        limits.check_allocation(shape.numel(), dtype)?;
        let numel = reader.read_u32()? as usize;
        let skip_bytes = numel.checked_mul(8).ok_or(Error::ArithmeticOverflow {
            context: "inspect_checkpoint_shapes skip",
        })?;
        let _ = reader.read_bytes(skip_bytes)?;
        out.insert(name, (shape, dtype));
    }
    Ok(out)
}

/// Validates and loads checkpoint bytes into `params`, matching by parameter name,
/// shape, and dtype while enforcing `limits`.
pub fn load_checkpoint(
    bytes: &[u8],
    params: &mut [&mut Param],
    limits: MemoryLimits,
) -> Result<()> {
    let mut reader = SafeReader::new(bytes);
    let magic = reader.read_bytes(4)?;
    if magic != MAGIC {
        return Err(Error::InvalidArgument {
            context: "load_checkpoint",
            reason: "invalid checkpoint magic header".to_owned(),
        });
    }
    let version = reader.read_u32()?;
    if version != VERSION {
        return Err(Error::InvalidArgument {
            context: "load_checkpoint",
            reason: format!("unsupported checkpoint version {version}"),
        });
    }
    let expected_checksum = reader.read_u64()?;
    let actual_checksum = fnv1a_64(reader.remaining());
    if actual_checksum != expected_checksum {
        return Err(Error::InvalidArgument {
            context: "load_checkpoint",
            reason: "checkpoint integrity checksum mismatch".to_owned(),
        });
    }

    let count = reader.read_u32()? as usize;
    if count > MAX_PARAMS {
        return Err(Error::InvalidArgument {
            context: "load_checkpoint",
            reason: format!("parameter count {count} exceeds limit {MAX_PARAMS}"),
        });
    }

    let mut loaded: HashMap<String, Tensor> = HashMap::with_capacity(count.min(1024));
    for _ in 0..count {
        let name = reader.read_string()?;
        let rank = reader.read_u32()? as usize;
        if rank > MAX_RANK {
            return Err(Error::InvalidRank {
                context: "load_checkpoint",
                rank,
                expected: MAX_RANK,
            });
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(reader.read_u64()? as usize);
        }
        let shape = Shape::new(dims)?;
        let dtype = reader.read_dtype()?;
        limits.check_allocation(shape.numel(), dtype)?;

        let numel = reader.read_u32()? as usize;
        if numel != shape.numel() {
            return Err(Error::BufferLengthMismatch {
                expected: shape.numel(),
                actual: numel,
            });
        }
        let mut vals = Vec::with_capacity(numel.min(4096));
        for _ in 0..numel {
            vals.push(f64::from_bits(reader.read_u64()?));
        }
        let t_f64 = Tensor::from_f64(&vals, shape.dims().to_vec())?;
        let tensor = if dtype == DType::F64 {
            t_f64
        } else {
            t_f64.to_dtype(dtype)?
        };
        loaded.insert(name, tensor);
    }

    if !reader.remaining().is_empty() {
        return Err(Error::InvalidArgument {
            context: "load_checkpoint",
            reason: "trailing bytes after checkpoint payload".to_owned(),
        });
    }

    for p in params.iter_mut() {
        let Some(tensor) = loaded.remove(p.name()) else {
            return Err(Error::InvalidArgument {
                context: "load_checkpoint",
                reason: format!("missing parameter `{}` in checkpoint", p.name()),
            });
        };
        if tensor.shape() != p.shape() {
            return Err(Error::ShapeMismatch {
                context: "load_checkpoint parameter",
                expected: p.shape().dims().to_vec(),
                actual: tensor.dims().to_vec(),
            });
        }
        if tensor.dtype() != p.dtype() {
            return Err(Error::DTypeMismatch {
                expected: p.dtype(),
                actual: tensor.dtype(),
            });
        }
        p.set_data(tensor);
    }

    Ok(())
}

struct SafeReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SafeReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> &'a [u8] {
        self.data.get(self.pos..).unwrap_or(&[])
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::ArithmeticOverflow {
            context: "checkpoint read_bytes",
        })?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| Error::InvalidArgument {
            context: "load_checkpoint",
            reason: "unexpected EOF in checkpoint".to_owned(),
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        b.first().copied().ok_or(Error::InternalInvariant {
            message: "empty u8 slice",
        })
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        let arr: [u8; 4] = b.try_into().map_err(|_| Error::InternalInvariant {
            message: "bad u32 slice",
        })?;
        Ok(u32::from_le_bytes(arr))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        let arr: [u8; 8] = b.try_into().map_err(|_| Error::InternalInvariant {
            message: "bad u64 slice",
        })?;
        Ok(u64::from_le_bytes(arr))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        if len > MAX_NAME_LEN {
            return Err(Error::InvalidArgument {
                context: "load_checkpoint",
                reason: format!("parameter name length {len} exceeds {MAX_NAME_LEN}"),
            });
        }
        let b = self.read_bytes(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::InvalidArgument {
            context: "load_checkpoint",
            reason: "invalid UTF-8 in parameter name".to_owned(),
        })
    }

    fn read_dtype(&mut self) -> Result<DType> {
        match self.read_u8()? {
            0 => Ok(DType::F32),
            1 => Ok(DType::F64),
            2 => Ok(DType::F16),
            3 => Ok(DType::BF16),
            4 => Ok(DType::I32),
            5 => Ok(DType::I64),
            t => Err(Error::InvalidArgument {
                context: "load_checkpoint",
                reason: format!("unknown dtype tag {t}"),
            }),
        }
    }
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I32 => 4,
        DType::I64 => 5,
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}
