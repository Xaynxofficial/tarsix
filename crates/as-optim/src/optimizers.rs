//! SGD (with momentum & weight decay), Adam, and AdamW (decoupled weight decay).

use std::collections::HashMap;

use as_core::{Error, Result};
use as_nn::Param;
use as_tensor::Tensor;

/// Common optimizer contract.
pub trait Optimizer {
    /// Applies one parameter update step given matched slices of parameters and optional gradients.
    fn step(&mut self, params: &mut [&mut Param], grads: &[Option<&Tensor>]) -> Result<()>;

    /// Returns the current learning rate.
    fn learning_rate(&self) -> f64;

    /// Updates the learning rate (used by LR schedulers).
    fn set_learning_rate(&mut self, lr: f64);
}

/// Stochastic Gradient Descent with optional momentum and L2 weight decay.
#[derive(Debug, Clone)]
pub struct Sgd {
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    velocities: HashMap<String, Tensor>,
}

impl Sgd {
    /// Creates a new [`Sgd`] optimizer.
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [&mut Param], grads: &[Option<&Tensor>]) -> Result<()> {
        validate_param_grad_len(params.len(), grads.len())?;
        for (param, grad_opt) in params.iter_mut().zip(grads.iter()) {
            let Some(grad) = grad_opt else {
                continue;
            };
            let dtype = param.dtype();
            let mut g = (*grad).clone();
            if self.weight_decay != 0.0 {
                let wd = Tensor::full(vec![], self.weight_decay, dtype)?;
                g = g.add(&param.data().mul(&wd)?)?;
            }
            let update = if self.momentum != 0.0 {
                let v = if let Some(prev) = self.velocities.get(param.name()) {
                    let mu = Tensor::full(vec![], self.momentum, dtype)?;
                    prev.mul(&mu)?.add(&g)?
                } else {
                    g
                };
                self.velocities.insert(param.name().to_owned(), v.clone());
                v
            } else {
                g
            };
            let lr_t = Tensor::full(vec![], self.lr, dtype)?;
            let new_data = param.data().sub(&update.mul(&lr_t)?)?;
            param.set_data(new_data);
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Adaptive Moment Estimation (`Adam`) with bias correction.
#[derive(Debug, Clone)]
pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: u64,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl Adam {
    /// Creates a new [`Adam`] optimizer with standard defaults (`beta1=0.9, beta2=0.999, eps=1e-8`).
    pub fn new(lr: f64) -> Self {
        Self::with_config(lr, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Creates a new [`Adam`] optimizer with explicit hyperparameters.
    pub fn with_config(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Param], grads: &[Option<&Tensor>]) -> Result<()> {
        validate_param_grad_len(params.len(), grads.len())?;
        self.step_count += 1;
        let t = self.step_count as i32;
        let bias_corr1 = 1.0 - self.beta1.powi(t);
        let bias_corr2 = 1.0 - self.beta2.powi(t);

        for (param, grad_opt) in params.iter_mut().zip(grads.iter()) {
            let Some(grad) = grad_opt else {
                continue;
            };
            let dtype = param.dtype();
            let mut g = (*grad).clone();
            if self.weight_decay != 0.0 {
                let wd = Tensor::full(vec![], self.weight_decay, dtype)?;
                g = g.add(&param.data().mul(&wd)?)?;
            }

            let (new_m, new_v) = update_adam_moments(
                &self.m,
                &self.v,
                param.name(),
                &g,
                self.beta1,
                self.beta2,
            )?;

            let m_hat = new_m.div(&Tensor::full(vec![], bias_corr1, dtype)?)?;
            let v_hat = new_v.div(&Tensor::full(vec![], bias_corr2, dtype)?)?;
            let denom = v_hat.sqrt()?.add(&Tensor::full(vec![], self.eps, dtype)?)?;
            let step_t = m_hat
                .div(&denom)?
                .mul(&Tensor::full(vec![], self.lr, dtype)?)?;

            param.set_data(param.data().sub(&step_t)?);
            self.m.insert(param.name().to_owned(), new_m);
            self.v.insert(param.name().to_owned(), new_v);
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Adam with decoupled weight decay (`AdamW`).
#[derive(Debug, Clone)]
pub struct AdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: u64,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl AdamW {
    /// Creates a new [`AdamW`] optimizer (`beta1=0.9, beta2=0.999, eps=1e-8`).
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self::with_config(lr, 0.9, 0.999, 1e-8, weight_decay)
    }

    /// Creates a new [`AdamW`] optimizer with explicit hyperparameters.
    pub fn with_config(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Param], grads: &[Option<&Tensor>]) -> Result<()> {
        validate_param_grad_len(params.len(), grads.len())?;
        self.step_count += 1;
        let t = self.step_count as i32;
        let bias_corr1 = 1.0 - self.beta1.powi(t);
        let bias_corr2 = 1.0 - self.beta2.powi(t);

        for (param, grad_opt) in params.iter_mut().zip(grads.iter()) {
            let Some(grad) = grad_opt else {
                continue;
            };
            let dtype = param.dtype();

            // Decoupled weight decay: p = p * (1 - lr * wd)
            let decayed = if self.weight_decay != 0.0 {
                let factor = Tensor::full(vec![], 1.0 - self.lr * self.weight_decay, dtype)?;
                param.data().mul(&factor)?
            } else {
                param.data().clone()
            };

            let (new_m, new_v) = update_adam_moments(
                &self.m,
                &self.v,
                param.name(),
                grad,
                self.beta1,
                self.beta2,
            )?;

            let m_hat = new_m.div(&Tensor::full(vec![], bias_corr1, dtype)?)?;
            let v_hat = new_v.div(&Tensor::full(vec![], bias_corr2, dtype)?)?;
            let denom = v_hat.sqrt()?.add(&Tensor::full(vec![], self.eps, dtype)?)?;
            let step_t = m_hat
                .div(&denom)?
                .mul(&Tensor::full(vec![], self.lr, dtype)?)?;

            param.set_data(decayed.sub(&step_t)?);
            self.m.insert(param.name().to_owned(), new_m);
            self.v.insert(param.name().to_owned(), new_v);
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

fn update_adam_moments(
    m_map: &HashMap<String, Tensor>,
    v_map: &HashMap<String, Tensor>,
    name: &str,
    g: &Tensor,
    beta1: f64,
    beta2: f64,
) -> Result<(Tensor, Tensor)> {
    let dtype = g.dtype();
    let b1 = Tensor::full(vec![], beta1, dtype)?;
    let one_m_b1 = Tensor::full(vec![], 1.0 - beta1, dtype)?;
    let b2 = Tensor::full(vec![], beta2, dtype)?;
    let one_m_b2 = Tensor::full(vec![], 1.0 - beta2, dtype)?;

    let prev_m = match m_map.get(name) {
        Some(t) => t.clone(),
        None => Tensor::zeros(g.dims().to_vec(), dtype)?,
    };
    let prev_v = match v_map.get(name) {
        Some(t) => t.clone(),
        None => Tensor::zeros(g.dims().to_vec(), dtype)?,
    };

    let new_m = prev_m.mul(&b1)?.add(&g.mul(&one_m_b1)?)?;
    let g_sq = g.mul(g)?;
    let new_v = prev_v.mul(&b2)?.add(&g_sq.mul(&one_m_b2)?)?;
    Ok((new_m, new_v))
}

fn validate_param_grad_len(p_len: usize, g_len: usize) -> Result<()> {
    if p_len != g_len {
        return Err(Error::BufferLengthMismatch {
            expected: p_len,
            actual: g_len,
        });
    }
    Ok(())
}
