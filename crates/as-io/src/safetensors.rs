//! Independent, zero-dependency, security-hardened **Safetensors** reader and writer.
//!
//! Specification format:
//! - `[0..8]`: `u64` little-endian header length `N`
//! - `[8..8+N]`: UTF-8 JSON header mapping tensor names -> `{dtype, shape, data_offsets}`
//! - `[8+N..]`: raw little-endian tensor byte buffers
//!
//! Security validations enforced on untrusted inputs:
//! - Header length capped at [`MAX_SAFETENSORS_HEADER_BYTES`]
//! - Offset range bounds (`start <= end <= data_len`) and non-overlapping interval check
//! - Exact byte-length verification (`end - start == shape.numel() * dtype.size_in_bytes()`)
//! - [`MemoryLimits`] checked before allocating any tensor buffer

use std::collections::BTreeMap;

use as_core::{BF16, DType, Error, F16, MemoryLimits, Result, Shape};
use as_tensor::Tensor;

use crate::json::{escape_json_string, parse_json};

/// Maximum permitted Safetensors JSON header length (16 MiB).
pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 16 * 1024 * 1024;

/// Metadata for a single tensor inside a Safetensors archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Element data type.
    pub dtype: DType,
    /// Tensor dimensions.
    pub shape: Shape,
    /// Byte offset range `[start, end)` relative to the start of the data section.
    pub data_offsets: (usize, usize),
}

/// Decoded Safetensors archive containing named tensors and optional string metadata.
#[derive(Debug, Clone)]
pub struct SafeTensors {
    /// Deterministic ordered map of tensor names to [`Tensor`]s.
    pub tensors: BTreeMap<String, Tensor>,
    /// Optional `__metadata__` string map.
    pub metadata: BTreeMap<String, String>,
}

/// Serializes a map of named tensors and optional metadata into standard Safetensors format.
pub fn serialize_safetensors(
    tensors: &BTreeMap<String, Tensor>,
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let mut data_section = Vec::new();
    let mut header_entries = Vec::new();

    if !metadata.is_empty() {
        let meta_pairs: Vec<String> = metadata
            .iter()
            .map(|(k, v)| format!("{}:{}", escape_json_string(k), escape_json_string(v)))
            .collect();
        header_entries.push(format!("\"__metadata__\":{{{}}}", meta_pairs.join(",")));
    }

    for (name, tensor) in tensors {
        let start = data_section.len();
        encode_tensor_bytes(tensor, &mut data_section)?;
        let end = data_section.len();

        let dims_str: Vec<String> = tensor.dims().iter().map(usize::to_string).collect();
        let entry = format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            escape_json_string(name),
            dtype_to_st_str(tensor.dtype()),
            dims_str.join(","),
            start,
            end
        );
        header_entries.push(entry);
    }

    let header_json = format!("{{{}}}", header_entries.join(","));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_section.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_section);
    Ok(out)
}

/// Safely parses and validates an untrusted Safetensors byte slice.
pub fn deserialize_safetensors(bytes: &[u8], limits: MemoryLimits) -> Result<SafeTensors> {
    let len_bytes = bytes.get(0..8).ok_or_else(|| Error::InvalidArgument {
        context: "deserialize_safetensors",
        reason: "file shorter than 8-byte header length prefix".to_owned(),
    })?;
    let len_arr: [u8; 8] = len_bytes.try_into().map_err(|_| Error::InternalInvariant {
        message: "8-byte slice conversion failed",
    })?;
    let header_len_u64 = u64::from_le_bytes(len_arr);
    let header_len = usize::try_from(header_len_u64).map_err(|_| Error::ArithmeticOverflow {
        context: "safetensors header_len",
    })?;

    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(Error::InvalidArgument {
            context: "deserialize_safetensors",
            reason: format!(
                "header length {header_len} exceeds security ceiling {MAX_SAFETENSORS_HEADER_BYTES}"
            ),
        });
    }

    let data_start = 8usize
        .checked_add(header_len)
        .ok_or(Error::ArithmeticOverflow {
            context: "safetensors data_start",
        })?;
    let header_slice = bytes.get(8..data_start).ok_or_else(|| Error::InvalidArgument {
        context: "deserialize_safetensors",
        reason: "unexpected EOF inside safetensors header".to_owned(),
    })?;
    let data_section = bytes.get(data_start..).ok_or_else(|| Error::InvalidArgument {
        context: "deserialize_safetensors",
        reason: "unexpected EOF before data section".to_owned(),
    })?;

    let header_str = core::str::from_utf8(header_slice).map_err(|_| Error::InvalidArgument {
        context: "deserialize_safetensors",
        reason: "safetensors header is not valid UTF-8".to_owned(),
    })?;
    let json = parse_json(header_str)?;
    let root = json.as_object().ok_or_else(|| Error::InvalidArgument {
        context: "deserialize_safetensors",
        reason: "safetensors header root must be a JSON object".to_owned(),
    })?;

    let mut metadata = BTreeMap::new();
    let mut infos: Vec<(String, TensorInfo)> = Vec::new();

    for (key, val) in root {
        if key == "__metadata__" {
            let meta_obj = val.as_object().ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: "__metadata__ must be a JSON object".to_owned(),
            })?;
            for (mk, mv) in meta_obj {
                let s = mv.as_str().ok_or_else(|| Error::InvalidArgument {
                    context: "deserialize_safetensors",
                    reason: "__metadata__ values must be strings".to_owned(),
                })?;
                metadata.insert(mk.clone(), s.to_owned());
            }
            continue;
        }

        let obj = val.as_object().ok_or_else(|| Error::InvalidArgument {
            context: "deserialize_safetensors",
            reason: format!("tensor entry `{key}` must be a JSON object"),
        })?;
        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` missing string dtype"),
            })?;
        let dtype = st_str_to_dtype(dtype_str)?;

        let shape_arr = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` missing shape array"),
            })?;
        let mut dims = Vec::with_capacity(shape_arr.len());
        for d in shape_arr {
            dims.push(d.as_usize().ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` contains non-integer dimension"),
            })?);
        }
        let shape = Shape::new(dims)?;

        let offsets_arr = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` missing data_offsets"),
            })?;
        if offsets_arr.len() != 2 {
            return Err(Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` data_offsets must have length 2"),
            });
        }
        let start = offsets_arr
            .first()
            .and_then(|v| v.as_usize())
            .ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` invalid start offset"),
            })?;
        let end = offsets_arr
            .get(1)
            .and_then(|v| v.as_usize())
            .ok_or_else(|| Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!("tensor `{key}` invalid end offset"),
            })?;

        if start > end || end > data_section.len() {
            return Err(Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!(
                    "tensor `{key}` offsets [{start}, {end}) out of bounds for data length {}",
                    data_section.len()
                ),
            });
        }

        let expected_bytes = limits.check_allocation(shape.numel(), dtype)?;
        if end - start != expected_bytes {
            return Err(Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: format!(
                    "tensor `{key}` byte length {} does not match expected {expected_bytes}",
                    end - start
                ),
            });
        }

        infos.push((
            key.clone(),
            TensorInfo {
                dtype,
                shape,
                data_offsets: (start, end),
            },
        ));
    }

    // Security check: reject overlapping tensor byte intervals
    let mut intervals: Vec<(usize, usize)> = infos.iter().map(|(_, i)| i.data_offsets).collect();
    intervals.sort_unstable_by_key(|&(s, e)| (s, e));
    for pair in intervals.windows(2) {
        if let (Some(&(_, prev_end)), Some(&(next_start, _))) = (pair.first(), pair.get(1))
            && next_start < prev_end
        {
            return Err(Error::InvalidArgument {
                context: "deserialize_safetensors",
                reason: "overlapping tensor data_offsets detected in safetensors header".to_owned(),
            });
        }
    }

    let mut tensors = BTreeMap::new();
    for (name, info) in infos {
        let (s, e) = info.data_offsets;
        let raw = data_section.get(s..e).ok_or(Error::IndexOutOfBounds {
            index: e,
            len: data_section.len(),
        })?;
        let tensor = decode_tensor_bytes(raw, info.dtype, &info.shape)?;
        tensors.insert(name, tensor);
    }

    Ok(SafeTensors { tensors, metadata })
}

fn dtype_to_st_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::I32 => "I32",
        DType::I64 => "I64",
    }
}

fn st_str_to_dtype(s: &str) -> Result<DType> {
    match s {
        "F32" => Ok(DType::F32),
        "F64" => Ok(DType::F64),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I32" => Ok(DType::I32),
        "I64" => Ok(DType::I64),
        other => Err(Error::InvalidArgument {
            context: "deserialize_safetensors",
            reason: format!("unsupported safetensors dtype `{other}`"),
        }),
    }
}

fn encode_tensor_bytes(tensor: &Tensor, out: &mut Vec<u8>) -> Result<()> {
    match tensor.dtype() {
        DType::F32 => {
            for v in tensor.to_f32_vec()? {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::F64 => {
            for v in tensor.to_f64_vec()? {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => {
            for v in tensor.to_f16_vec()? {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        DType::BF16 => {
            for v in tensor.to_bf16_vec()? {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        DType::I32 => {
            for v in tensor.to_i32_vec()? {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::I64 => {
            for v in tensor.to_i64_vec()? {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn decode_tensor_bytes(raw: &[u8], dtype: DType, shape: &Shape) -> Result<Tensor> {
    let dims = shape.dims().to_vec();
    match dtype {
        DType::F32 => {
            let mut vals = Vec::with_capacity(shape.numel());
            for chunk in raw.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| Error::InternalInvariant {
                    message: "f32 chunk len",
                })?;
                vals.push(f32::from_le_bytes(arr));
            }
            Tensor::from_f32(&vals, dims)
        }
        DType::F64 => {
            let mut vals = Vec::with_capacity(shape.numel());
            for chunk in raw.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().map_err(|_| Error::InternalInvariant {
                    message: "f64 chunk len",
                })?;
                vals.push(f64::from_le_bytes(arr));
            }
            Tensor::from_f64(&vals, dims)
        }
        DType::F16 => {
            let mut vals = Vec::with_capacity(shape.numel());
            for chunk in raw.chunks_exact(2) {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| Error::InternalInvariant {
                    message: "f16 chunk len",
                })?;
                vals.push(F16::from_bits(u16::from_le_bytes(arr)));
            }
            Tensor::from_f16(&vals, dims)
        }
        DType::BF16 => {
            let mut vals = Vec::with_capacity(shape.numel());
            for chunk in raw.chunks_exact(2) {
                let arr: [u8; 2] = chunk.try_into().map_err(|_| Error::InternalInvariant {
                    message: "bf16 chunk len",
                })?;
                vals.push(BF16::from_bits(u16::from_le_bytes(arr)));
            }
            Tensor::from_bf16(&vals, dims)
        }
        DType::I32 => {
            let mut vals = Vec::with_capacity(shape.numel());
            for chunk in raw.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| Error::InternalInvariant {
                    message: "i32 chunk len",
                })?;
                vals.push(i32::from_le_bytes(arr));
            }
            Tensor::from_i32(&vals, dims)
        }
        DType::I64 => {
            let mut vals = Vec::with_capacity(shape.numel());
            for chunk in raw.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().map_err(|_| Error::InternalInvariant {
                    message: "i64 chunk len",
                })?;
                vals.push(i64::from_le_bytes(arr));
            }
            Tensor::from_i64(&vals, dims)
        }
    }
}
