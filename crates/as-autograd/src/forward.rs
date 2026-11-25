//! Forward-mode automatic differentiation using dual tensors (Jacobian-Vector Products).

use as_core::{DType, Error, Result};
use as_tensor::Tensor;

/// A dual tensor pairing a `primal` value with its directional derivative `tangent` (JVP).
#[derive(Debug, Clone)]
pub struct DualTensor {
    primal: Tensor,
    tangent: Tensor,
}

impl DualTensor {
    /// Creates a new [`DualTensor`], validating matching shapes and dtypes.
    pub fn new(primal: Tensor, tangent: Tensor) -> Result<Self> {
        if primal.shape() != tangent.shape() {
            return Err(Error::ShapeMismatch {
                context: "DualTensor::new",
                expected: primal.dims().to_vec(),
                actual: tangent.dims().to_vec(),
            });
        }
        if primal.dtype() != tangent.dtype() {
            return Err(Error::DTypeMismatch {
                expected: primal.dtype(),
                actual: tangent.dtype(),
            });
        }
        Ok(Self { primal, tangent })
    }

    /// Creates a constant dual tensor with zero tangent.
    pub fn constant(primal: Tensor) -> Result<Self> {
        let tangent = Tensor::zeros(primal.dims().to_vec(), primal.dtype())?;
        Ok(Self { primal, tangent })
    }

    /// Returns the primal tensor.
    pub fn primal(&self) -> &Tensor {
        &self.primal
    }

    /// Returns the tangent (directional derivative) tensor.
    pub fn tangent(&self) -> &Tensor {
        &self.tangent
    }

    /// Forward-mode addition: `d(a + b) = da + db`.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            primal: self.primal.add(&rhs.primal)?,
            tangent: self.tangent.add(&rhs.tangent)?,
        })
    }

    /// Forward-mode subtraction: `d(a - b) = da - db`.
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        Ok(Self {
            primal: self.primal.sub(&rhs.primal)?,
            tangent: self.tangent.sub(&rhs.tangent)?,
        })
    }

    /// Forward-mode product rule: `d(a * b) = da * b + a * db`.
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        let primal = self.primal.mul(&rhs.primal)?;
        let tangent = self
            .tangent
            .mul(&rhs.primal)?
            .add(&self.primal.mul(&rhs.tangent)?)?;
        Ok(Self { primal, tangent })
    }

    /// Forward-mode quotient rule: `d(a / b) = (da * b - a * db) / b^2`.
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        let primal = self.primal.div(&rhs.primal)?;
        let num = self
            .tangent
            .mul(&rhs.primal)?
            .sub(&self.primal.mul(&rhs.tangent)?)?;
        let denom = rhs.primal.mul(&rhs.primal)?;
        Ok(Self {
            primal,
            tangent: num.div(&denom)?,
        })
    }

    /// Forward-mode negation.
    pub fn neg(&self) -> Result<Self> {
        Ok(Self {
            primal: self.primal.neg()?,
            tangent: self.tangent.neg()?,
        })
    }

    /// Forward-mode exponential: `d(exp(x)) = exp(x) * dx`.
    pub fn exp(&self) -> Result<Self> {
        let primal = self.primal.exp()?;
        let tangent = primal.mul(&self.tangent)?;
        Ok(Self { primal, tangent })
    }

    /// Forward-mode natural logarithm: `d(ln(x)) = dx / x`.
    pub fn ln(&self) -> Result<Self> {
        Ok(Self {
            primal: self.primal.ln()?,
            tangent: self.tangent.div(&self.primal)?,
        })
    }

    /// Forward-mode square root: `d(sqrt(x)) = dx / (2 * sqrt(x))`.
    pub fn sqrt(&self) -> Result<Self> {
        let primal = self.primal.sqrt()?;
        let two = Tensor::full(vec![], 2.0, primal.dtype())?;
        let tangent = self.tangent.div(&primal.mul(&two)?)?;
        Ok(Self { primal, tangent })
    }

    /// Forward-mode ReLU.
    pub fn relu(&self) -> Result<Self> {
        let primal = self.primal.relu()?;
        let mask = match self.primal.dtype() {
            DType::F32 => {
                let v: Vec<f32> = self
                    .primal
                    .to_f32_vec()?
                    .into_iter()
                    .map(|x| if x > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                Tensor::from_f32(&v, self.primal.dims().to_vec())?
            }
            DType::F64 => {
                let v: Vec<f64> = self
                    .primal
                    .to_f64_vec()?
                    .into_iter()
                    .map(|x| if x > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                Tensor::from_f64(&v, self.primal.dims().to_vec())?
            }
            dtype => {
                return Err(Error::UnsupportedDType {
                    op: "DualTensor::relu",
                    dtype,
                });
            }
        };
        Ok(Self {
            primal,
            tangent: self.tangent.mul(&mask)?,
        })
    }

    /// Forward-mode power rule: `d(x^p) = p * x^(p-1) * dx`.
    pub fn powf(&self, p: f64) -> Result<Self> {
        let primal = self.primal.powf(p)?;
        let p_t = Tensor::full(vec![], p, self.primal.dtype())?;
        let tangent = self
            .primal
            .powf(p - 1.0)?
            .mul(&p_t)?
            .mul(&self.tangent)?;
        Ok(Self { primal, tangent })
    }

    /// Forward-mode matrix multiplication: `d(A @ B) = dA @ B + A @ dB`.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let primal = self.primal.matmul(&rhs.primal)?;
        let tangent = self
            .tangent
            .matmul(&rhs.primal)?
            .add(&self.primal.matmul(&rhs.tangent)?)?;
        Ok(Self { primal, tangent })
    }

    /// Forward-mode transpose.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        Ok(Self {
            primal: self.primal.transpose(axis_a, axis_b)?,
            tangent: self.tangent.transpose(axis_a, axis_b)?,
        })
    }

    /// Forward-mode reshape.
    pub fn reshape(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = new_shape.into();
        Ok(Self {
            primal: self.primal.reshape(dims.clone())?,
            tangent: self.tangent.reshape(dims)?,
        })
    }

    /// Forward-mode global sum.
    pub fn sum(&self) -> Result<Self> {
        Ok(Self {
            primal: self.primal.sum()?,
            tangent: self.tangent.sum()?,
        })
    }

    /// Forward-mode global mean.
    pub fn mean(&self) -> Result<Self> {
        Ok(Self {
            primal: self.primal.mean()?,
            tangent: self.tangent.mean()?,
        })
    }
}
