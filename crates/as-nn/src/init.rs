//! Deterministic, zero-dependency PRNG (Xoshiro256++) and neural network initializers.

use core::f64::consts::PI;

use as_core::{DType, Result};
use as_tensor::Tensor;

/// Deterministic 256-bit PRNG (Xoshiro256++ seeded via SplitMix64).
///
/// Guarantees reproducible initialization, dropout masks, and dataset shuffling
/// across all platforms without external dependencies.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Seeds a new deterministic [`Rng`] from a 64-bit seed.
    pub fn from_seed(mut seed: u64) -> Self {
        let mut next_splitmix = || -> u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let s0 = next_splitmix();
        let s1 = next_splitmix();
        let s2 = next_splitmix();
        let s3 = next_splitmix();
        Self {
            state: [s0, s1, s2, s3],
        }
    }

    /// Generates the next uniformly distributed `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for full IEEE-754 f64 significand precision
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Generates a uniform `f64` in `[low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Generates a standard normal `N(0, 1)` sample via Box-Muller transform.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z0
    }

    /// Generates a uniform `usize` in `[0, bound)`.
    pub fn gen_range(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    /// Deterministic Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}

/// Xavier / Glorot uniform initialization: `U(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(
    shape: Vec<usize>,
    fan_in: usize,
    fan_out: usize,
    dtype: DType,
    rng: &mut Rng,
) -> Result<Tensor> {
    let numel: usize = shape.iter().copied().product();
    let denom = (fan_in + fan_out).max(1) as f64;
    let limit = (6.0 / denom).sqrt();
    let mut vals = Vec::with_capacity(numel);
    for _ in 0..numel {
        vals.push(rng.uniform(-limit, limit));
    }
    let t = Tensor::from_f64(&vals, shape)?;
    if dtype == DType::F64 {
        Ok(t)
    } else {
        t.to_dtype(dtype)
    }
}

/// Kaiming / He normal initialization: `N(0, sqrt(2 / fan_in))`.
pub fn kaiming_normal(
    shape: Vec<usize>,
    fan_in: usize,
    dtype: DType,
    rng: &mut Rng,
) -> Result<Tensor> {
    let numel: usize = shape.iter().copied().product();
    let std = (2.0 / (fan_in.max(1) as f64)).sqrt();
    let mut vals = Vec::with_capacity(numel);
    for _ in 0..numel {
        vals.push(rng.normal(0.0, std));
    }
    let t = Tensor::from_f64(&vals, shape)?;
    if dtype == DType::F64 {
        Ok(t)
    } else {
        t.to_dtype(dtype)
    }
}
