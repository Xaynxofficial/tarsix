//! Numeric data type system (`F32`, `F64`, `F16`, `BF16`, `I32`, `I64`).
//!
//! Includes independent, zero-dependency IEEE-754 binary16 ([`F16`]) and
//! brain floating-point ([`BF16`]) representations with round-to-nearest-even
//! semantics, subnormal handling, and NaN/Inf preservation.

use core::fmt;

/// Supported tensor element data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE-754 floating point.
    F32,
    /// 64-bit IEEE-754 floating point.
    F64,
    /// 16-bit IEEE-754 binary16 floating point.
    F16,
    /// 16-bit Brain Floating Point (truncated IEEE-754 f32 with round-to-nearest-even).
    BF16,
    /// 32-bit signed two's-complement integer.
    I32,
    /// 64-bit signed two's-complement integer.
    I64,
}

impl DType {
    /// Returns the storage size of a single element in bytes.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    /// Returns `true` if this dtype is a floating-point representation.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F16 | Self::BF16)
    }

    /// Returns `true` if this dtype is a signed integer representation.
    pub const fn is_int(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I32 => "i32",
            Self::I64 => "i64",
        };
        f.write_str(name)
    }
}

/// Independent IEEE-754 binary16 (half-precision) floating-point type.
///
/// Layout: 1 sign bit | 5 exponent bits (bias 15) | 10 significand bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    /// Positive zero (`0.0`).
    pub const ZERO: Self = Self(0x0000);
    /// Positive one (`1.0`).
    pub const ONE: Self = Self(0x3C00);
    /// Canonical quiet NaN.
    pub const NAN: Self = Self(0x7E00);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFC00);

    /// Constructs an `F16` from its raw IEEE-754 bit pattern.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw IEEE-754 bit pattern.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f32` to `F16` using IEEE-754 round-to-nearest-even.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        // NaN or Infinity
        if exp == 0xFF {
            if mant == 0 {
                return Self(sign | 0x7C00);
            }
            // Preserve quiet NaN bit and top payload bits; ensure non-zero payload.
            let payload = ((mant >> 13) as u16) | 0x0200;
            return Self(sign | 0x7C00 | payload);
        }

        // Unbiased exponent: f32 bias is 127, f16 bias is 15 -> shift by -112
        let half_exp = exp - 112;

        // Overflow to infinity
        if half_exp >= 31 {
            return Self(sign | 0x7C00);
        }

        // Subnormal or underflow to zero
        if half_exp <= 0 {
            if half_exp < -10 {
                return Self(sign);
            }
            // Add implicit leading 1 bit to f32 mantissa
            let sig = mant | 0x0080_0000;
            let shift = (14 - half_exp) as u32;
            let half_mant = sig >> shift;
            // Round to nearest even
            let remainder_mask = (1u32 << shift) - 1;
            let remainder = sig & remainder_mask;
            let halfway = 1u32 << (shift - 1);
            let rounded = if remainder > halfway || (remainder == halfway && (half_mant & 1) != 0) {
                half_mant + 1
            } else {
                half_mant
            };
            return Self(sign | (rounded as u16));
        }

        // Normalized f16 number with round-to-nearest-even
        let half_mant = mant >> 13;
        let remainder = mant & 0x1FFF;
        let mut combined = ((half_exp as u32) << 10) | half_mant;
        if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
            combined += 1;
        }
        // If rounding overflowed into exponent 31, result is infinity (0x7C00)
        Self(sign | (combined as u16))
    }

    /// Converts this `F16` exactly to `f32`.
    pub fn to_f32(self) -> f32 {
        let bits = u32::from(self.0);
        let sign = (bits & 0x8000) << 16;
        let exp = (bits >> 10) & 0x1F;
        let mant = bits & 0x03FF;

        let f32_bits = if exp == 0 {
            if mant == 0 {
                // Signed zero
                sign
            } else {
                // Subnormal f16 -> normalize for f32
                let mut m = mant;
                let mut e: u32 = 113; // 127 - 14
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                sign | (e << 23) | (m << 13)
            }
        } else if exp == 31 {
            if mant == 0 {
                // Infinity
                sign | 0x7F80_0000
            } else {
                // Quiet NaN preserving payload
                sign | 0x7FC0_0000 | (mant << 13)
            }
        } else {
            let f32_exp = exp + 112;
            sign | (f32_exp << 23) | (mant << 13)
        };

        f32::from_bits(f32_bits)
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F16({})", self.to_f32())
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// Independent Brain Floating Point (`bfloat16`) type.
///
/// Layout: 1 sign bit | 8 exponent bits (bias 127, same as `f32`) | 7 significand bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct BF16(u16);

impl BF16 {
    /// Positive zero (`0.0`).
    pub const ZERO: Self = Self(0x0000);
    /// Positive one (`1.0`).
    pub const ONE: Self = Self(0x3F80);
    /// Canonical quiet NaN.
    pub const NAN: Self = Self(0x7FC0);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7F80);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFF80);

    /// Constructs a `BF16` from its raw bit pattern.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit pattern.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f32` to `BF16` using round-to-nearest-even.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            // Preserve sign and quiet NaN bit
            return Self(((bits >> 16) as u16) | 0x0040);
        }
        // Round-to-nearest-even bias on lower 16 bits
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7FFF + lsb;
        let rounded = bits.wrapping_add(rounding_bias);
        Self((rounded >> 16) as u16)
    }

    /// Converts this `BF16` exactly to `f32`.
    pub const fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl fmt::Debug for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BF16({})", self.to_f32())
    }
}

impl fmt::Display for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for super::F16 {}
    impl Sealed for super::BF16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
}

/// Sealed trait mapping Rust scalar types to [`DType`].
pub trait TensorElement: sealed::Sealed + Copy + Send + Sync + 'static {
    /// Corresponding [`DType`] variant.
    const DTYPE: DType;

    /// Zero value for this scalar type.
    fn zero() -> Self;

    /// One value for this scalar type.
    fn one() -> Self;

    /// Lossy conversion to `f64` for diagnostics and unified numeric helpers.
    fn to_f64(self) -> f64;
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl TensorElement for f64 {
    const DTYPE: DType = DType::F64;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl TensorElement for F16 {
    const DTYPE: DType = DType::F16;
    fn zero() -> Self {
        Self::ZERO
    }
    fn one() -> Self {
        Self::ONE
    }
    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

impl TensorElement for BF16 {
    const DTYPE: DType = DType::BF16;
    fn zero() -> Self {
        Self::ZERO
    }
    fn one() -> Self {
        Self::ONE
    }
    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

impl TensorElement for i32 {
    const DTYPE: DType = DType::I32;
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl TensorElement for i64 {
    const DTYPE: DType = DType::I64;
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_special_and_normal_values() {
        assert_eq!(F16::from_f32(0.0).to_bits(), 0x0000);
        assert_eq!(F16::from_f32(-0.0).to_bits(), 0x8000);
        assert_eq!(F16::from_f32(1.0).to_bits(), 0x3C00);
        assert_eq!(F16::from_f32(-2.0).to_bits(), 0xC000);
        assert_eq!(F16::from_f32(65504.0).to_bits(), 0x7BFF); // Max finite f16
        assert_eq!(F16::from_f32(65536.0).to_bits(), 0x7C00); // Overflow -> +Inf
        assert_eq!(F16::from_f32(f32::INFINITY).to_bits(), 0x7C00);
        assert_eq!(F16::from_f32(f32::NEG_INFINITY).to_bits(), 0xFC00);
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn f16_all_65536_bit_patterns_roundtrip() {
        for raw in 0u16..=u16::MAX {
            let h = F16::from_bits(raw);
            let f = h.to_f32();
            let back = F16::from_f32(f);
            if f.is_nan() {
                assert!(back.to_f32().is_nan());
            } else {
                assert_eq!(back.to_bits(), raw, "failed roundtrip for 0x{raw:04X}");
            }
        }
    }

    #[test]
    fn bf16_all_65536_bit_patterns_roundtrip() {
        for raw in 0u16..=u16::MAX {
            let b = BF16::from_bits(raw);
            let f = b.to_f32();
            let back = BF16::from_f32(f);
            if f.is_nan() {
                assert!(back.to_f32().is_nan());
            } else {
                assert_eq!(back.to_bits(), raw, "failed roundtrip for 0x{raw:04X}");
            }
        }
    }

    #[test]
    fn bf16_round_to_nearest_even() {
        // 1.0 in f32 is 0x3F80_0000; halfway to next bf16 (0x3F81_0000) is 0x3F80_8000.
        // Ties to even -> stays 0x3F80.
        let half_even = f32::from_bits(0x3F80_8000);
        assert_eq!(BF16::from_f32(half_even).to_bits(), 0x3F80);

        // Halfway from odd 0x3F81 to 0x3F82 is 0x3F81_8000 -> rounds up to even 0x3F82.
        let half_odd = f32::from_bits(0x3F81_8000);
        assert_eq!(BF16::from_f32(half_odd).to_bits(), 0x3F82);
    }
}
