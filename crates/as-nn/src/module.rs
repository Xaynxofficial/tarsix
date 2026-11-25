//! Trainable parameter wrapper and [`Module`] trait contract.

use as_autograd::{Tape, VarId};
use as_core::{DType, Shape};
use as_tensor::Tensor;

/// A named trainable model parameter.
#[derive(Debug, Clone)]
pub struct Param {
    name: String,
    data: Tensor,
}

impl Param {
    /// Creates a new trainable [`Param`].
    pub fn new(name: impl Into<String>, data: Tensor) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }

    /// Returns the parameter's hierarchical name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrows the underlying [`Tensor`].
    pub fn data(&self) -> &Tensor {
        &self.data
    }

    /// Returns the parameter's [`Shape`].
    pub fn shape(&self) -> &Shape {
        self.data.shape()
    }

    /// Returns the parameter's [`DType`].
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Replaces the parameter's data tensor (used by optimizers and checkpoint loading).
    pub fn set_data(&mut self, new_data: Tensor) {
        self.data = new_data;
    }

    /// Registers this parameter on `tape` with `requires_grad = true`.
    pub fn register(&self, tape: &mut Tape) -> VarId {
        tape.var(self.data.clone(), true)
    }
}

/// Trait implemented by all neural network layers and composite models.
pub trait Module {
    /// Returns immutable references to all trainable parameters in deterministic order.
    fn parameters(&self) -> Vec<&Param>;

    /// Returns mutable references to all trainable parameters in deterministic order.
    fn parameters_mut(&mut self) -> Vec<&mut Param>;

    /// Total number of scalar trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.data().numel()).sum()
    }
}
