//! Safe, checksum-verified binary serialization for computational [`Graph`]s.
//!
//! Treats deserialized byte streams as untrusted input:
//! - Magic header (`ASGR`) and version check
//! - FNV-1a 64-bit integrity checksum over payload
//! - Strict bounds checks, string-length caps, and [`MAX_GRAPH_NODES`] enforcement
//! - Cycle detection upon reconstruction via `Graph::from_raw`

use as_backend::{BinaryOp, ReduceOp, UnaryOp};
use as_core::{DType, Error, MAX_RANK, Result, Shape};

use crate::ir::{Graph, MAX_GRAPH_NODES, Node, NodeId, Op};

const MAGIC: &[u8; 4] = b"ASGR";
const VERSION: u32 = 1;
const MAX_STRING_BYTES: usize = 4096;
const MAX_CONSTANT_ELEMENTS: usize = 16_777_216;
const MAX_FUSED_CHAIN_LEN: usize = 256;

/// Serializes `graph` into a self-contained, integrity-checked binary format.
pub fn serialize_graph(graph: &Graph) -> Vec<u8> {
    let mut payload = Vec::new();
    write_u32(&mut payload, graph.nodes().len() as u32);
    for node in graph.nodes() {
        write_u32(&mut payload, node.id.0 as u32);
        payload.push(u8::from(node.checkpoint));
        write_u32(&mut payload, node.inputs.len() as u32);
        for &inp in &node.inputs {
            write_u32(&mut payload, inp.0 as u32);
        }
        encode_op(&mut payload, &node.op);
    }

    write_u32(&mut payload, graph.outputs().len() as u32);
    for &out in graph.outputs() {
        write_u32(&mut payload, out.0 as u32);
    }

    let checksum = fnv1a_64(&payload);
    let mut out = Vec::with_capacity(4 + 4 + 8 + payload.len());
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Safely deserializes and validates a [`Graph`] from an untrusted byte slice.
pub fn deserialize_graph(bytes: &[u8]) -> Result<Graph> {
    let mut reader = SafeReader::new(bytes);
    let magic = reader.read_bytes(4)?;
    if magic != MAGIC {
        return Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: "invalid graph magic header".to_owned(),
        });
    }
    let version = reader.read_u32()?;
    if version != VERSION {
        return Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: format!("unsupported graph format version {version}"),
        });
    }
    let expected_checksum = reader.read_u64()?;
    let payload = reader.remaining();
    let actual_checksum = fnv1a_64(payload);
    if actual_checksum != expected_checksum {
        return Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: "graph integrity checksum mismatch (corrupted or tampered data)".to_owned(),
        });
    }

    let node_count = reader.read_u32()? as usize;
    if node_count > MAX_GRAPH_NODES {
        return Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: format!("node count {node_count} exceeds limit {MAX_GRAPH_NODES}"),
        });
    }

    let mut nodes = Vec::with_capacity(node_count.min(1024));
    for _ in 0..node_count {
        let id = NodeId(reader.read_u32()? as usize);
        let checkpoint = reader.read_u8()? != 0;
        let inp_count = reader.read_u32()? as usize;
        if inp_count > 16 {
            return Err(Error::InvalidArgument {
                context: "deserialize_graph",
                reason: format!("suspicious input arity {inp_count}"),
            });
        }
        let mut inputs = Vec::with_capacity(inp_count);
        for _ in 0..inp_count {
            inputs.push(NodeId(reader.read_u32()? as usize));
        }
        let op = decode_op(&mut reader)?;
        nodes.push(Node {
            id,
            op,
            inputs,
            checkpoint,
        });
    }

    let out_count = reader.read_u32()? as usize;
    if out_count > node_count {
        return Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: "output count exceeds node count".to_owned(),
        });
    }
    let mut outputs = Vec::with_capacity(out_count);
    for _ in 0..out_count {
        outputs.push(NodeId(reader.read_u32()? as usize));
    }

    if !reader.remaining().is_empty() {
        return Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: "trailing unexpected bytes in graph payload".to_owned(),
        });
    }

    Graph::from_raw(nodes, outputs)
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
            context: "SafeReader::read_bytes",
        })?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| Error::InvalidArgument {
            context: "deserialize_graph",
            reason: "unexpected EOF while reading graph bytes".to_owned(),
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        b.first().copied().ok_or(Error::InternalInvariant {
            message: "read_u8 empty slice",
        })
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        let arr: [u8; 4] = b.try_into().map_err(|_| Error::InternalInvariant {
            message: "read_u32 slice len",
        })?;
        Ok(u32::from_le_bytes(arr))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        let arr: [u8; 8] = b.try_into().map_err(|_| Error::InternalInvariant {
            message: "read_u64 slice len",
        })?;
        Ok(u64::from_le_bytes(arr))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        if len > MAX_STRING_BYTES {
            return Err(Error::InvalidArgument {
                context: "deserialize_graph",
                reason: format!("string length {len} exceeds ceiling {MAX_STRING_BYTES}"),
            });
        }
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::InvalidArgument {
            context: "deserialize_graph",
            reason: "invalid UTF-8 in graph string".to_owned(),
        })
    }

    fn read_shape(&mut self) -> Result<Shape> {
        let rank = self.read_u32()? as usize;
        if rank > MAX_RANK {
            return Err(Error::InvalidRank {
                context: "deserialize_graph shape",
                rank,
                expected: MAX_RANK,
            });
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(self.read_u64()? as usize);
        }
        Shape::new(dims)
    }

    fn read_dtype(&mut self) -> Result<DType> {
        match self.read_u8()? {
            0 => Ok(DType::F32),
            1 => Ok(DType::F64),
            2 => Ok(DType::F16),
            3 => Ok(DType::BF16),
            4 => Ok(DType::I32),
            5 => Ok(DType::I64),
            tag => Err(Error::InvalidArgument {
                context: "deserialize_graph",
                reason: format!("unknown dtype tag {tag}"),
            }),
        }
    }
}

fn encode_op(buf: &mut Vec<u8>, op: &Op) {
    match op {
        Op::Input { name, shape, dtype } => {
            buf.push(0);
            write_u32(buf, name.len() as u32);
            buf.extend_from_slice(name.as_bytes());
            write_shape(buf, shape);
            buf.push(dtype_tag(*dtype));
        }
        Op::Constant {
            shape,
            dtype,
            values,
        } => {
            buf.push(1);
            write_shape(buf, shape);
            buf.push(dtype_tag(*dtype));
            write_u32(buf, values.len() as u32);
            for &v in values {
                buf.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        Op::Binary(b) => {
            buf.push(2);
            buf.push(match b {
                BinaryOp::Add => 0,
                BinaryOp::Sub => 1,
                BinaryOp::Mul => 2,
                BinaryOp::Div => 3,
            });
        }
        Op::Unary(u) => {
            buf.push(3);
            encode_unary(buf, *u);
        }
        Op::FusedUnaryChain(chain) => {
            buf.push(4);
            write_u32(buf, chain.len() as u32);
            for &u in chain {
                encode_unary(buf, u);
            }
        }
        Op::MatMul => buf.push(5),
        Op::Reshape(shape) => {
            buf.push(6);
            write_shape(buf, shape);
        }
        Op::Transpose { axis_a, axis_b } => {
            buf.push(7);
            write_u32(buf, *axis_a as u32);
            write_u32(buf, *axis_b as u32);
        }
        Op::Reduce { op, axis, keepdim } => {
            buf.push(8);
            buf.push(match op {
                ReduceOp::Sum => 0,
                ReduceOp::Mean => 1,
                ReduceOp::Max => 2,
                ReduceOp::Min => 3,
            });
            match axis {
                None => buf.push(0),
                Some(ax) => {
                    buf.push(1);
                    write_u32(buf, *ax as u32);
                }
            }
            buf.push(u8::from(*keepdim));
        }
        Op::FusedLinear {
            has_bias,
            activation,
        } => {
            buf.push(9);
            buf.push(u8::from(*has_bias));
            match activation {
                None => buf.push(0),
                Some(u) => {
                    buf.push(1);
                    encode_unary(buf, *u);
                }
            }
        }
    }
}

fn decode_op(r: &mut SafeReader<'_>) -> Result<Op> {
    match r.read_u8()? {
        0 => {
            let name = r.read_string()?;
            let shape = r.read_shape()?;
            let dtype = r.read_dtype()?;
            Ok(Op::Input { name, shape, dtype })
        }
        1 => {
            let shape = r.read_shape()?;
            let dtype = r.read_dtype()?;
            let len = r.read_u32()? as usize;
            if len > MAX_CONSTANT_ELEMENTS || len != shape.numel() {
                return Err(Error::InvalidArgument {
                    context: "deserialize_graph constant",
                    reason: format!("invalid constant length {len}"),
                });
            }
            let mut values = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                values.push(r.read_f64()?);
            }
            Ok(Op::Constant {
                shape,
                dtype,
                values,
            })
        }
        2 => {
            let bin = match r.read_u8()? {
                0 => BinaryOp::Add,
                1 => BinaryOp::Sub,
                2 => BinaryOp::Mul,
                3 => BinaryOp::Div,
                t => {
                    return Err(Error::InvalidArgument {
                        context: "deserialize_graph",
                        reason: format!("unknown binary op tag {t}"),
                    });
                }
            };
            Ok(Op::Binary(bin))
        }
        3 => Ok(Op::Unary(decode_unary(r)?)),
        4 => {
            let len = r.read_u32()? as usize;
            if len > MAX_FUSED_CHAIN_LEN {
                return Err(Error::InvalidArgument {
                    context: "deserialize_graph",
                    reason: format!("fused chain length {len} exceeds limit"),
                });
            }
            let mut chain = Vec::with_capacity(len);
            for _ in 0..len {
                chain.push(decode_unary(r)?);
            }
            Ok(Op::FusedUnaryChain(chain))
        }
        5 => Ok(Op::MatMul),
        6 => Ok(Op::Reshape(r.read_shape()?)),
        7 => {
            let axis_a = r.read_u32()? as usize;
            let axis_b = r.read_u32()? as usize;
            Ok(Op::Transpose { axis_a, axis_b })
        }
        8 => {
            let op = match r.read_u8()? {
                0 => ReduceOp::Sum,
                1 => ReduceOp::Mean,
                2 => ReduceOp::Max,
                3 => ReduceOp::Min,
                t => {
                    return Err(Error::InvalidArgument {
                        context: "deserialize_graph",
                        reason: format!("unknown reduce op tag {t}"),
                    });
                }
            };
            let axis = match r.read_u8()? {
                0 => None,
                1 => Some(r.read_u32()? as usize),
                _ => {
                    return Err(Error::InvalidArgument {
                        context: "deserialize_graph",
                        reason: "invalid optional axis tag".to_owned(),
                    });
                }
            };
            let keepdim = r.read_u8()? != 0;
            Ok(Op::Reduce { op, axis, keepdim })
        }
        9 => {
            let has_bias = r.read_u8()? != 0;
            let activation = match r.read_u8()? {
                0 => None,
                1 => Some(decode_unary(r)?),
                _ => {
                    return Err(Error::InvalidArgument {
                        context: "deserialize_graph",
                        reason: "invalid fused linear activation tag".to_owned(),
                    });
                }
            };
            Ok(Op::FusedLinear {
                has_bias,
                activation,
            })
        }
        tag => Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: format!("unknown op tag {tag}"),
        }),
    }
}

fn encode_unary(buf: &mut Vec<u8>, op: UnaryOp) {
    match op {
        UnaryOp::Neg => buf.push(0),
        UnaryOp::Abs => buf.push(1),
        UnaryOp::Exp => buf.push(2),
        UnaryOp::Ln => buf.push(3),
        UnaryOp::Sqrt => buf.push(4),
        UnaryOp::Relu => buf.push(5),
        UnaryOp::Powf(p) => {
            buf.push(6);
            buf.extend_from_slice(&p.to_bits().to_le_bytes());
        }
    }
}

fn decode_unary(r: &mut SafeReader<'_>) -> Result<UnaryOp> {
    match r.read_u8()? {
        0 => Ok(UnaryOp::Neg),
        1 => Ok(UnaryOp::Abs),
        2 => Ok(UnaryOp::Exp),
        3 => Ok(UnaryOp::Ln),
        4 => Ok(UnaryOp::Sqrt),
        5 => Ok(UnaryOp::Relu),
        6 => Ok(UnaryOp::Powf(r.read_f64()?)),
        tag => Err(Error::InvalidArgument {
            context: "deserialize_graph",
            reason: format!("unknown unary op tag {tag}"),
        }),
    }
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_shape(buf: &mut Vec<u8>, shape: &Shape) {
    write_u32(buf, shape.rank() as u32);
    for &d in shape.dims() {
        buf.extend_from_slice(&(d as u64).to_le_bytes());
    }
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

/// Deterministic 64-bit FNV-1a integrity checksum.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}
