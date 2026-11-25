//! Patch Embedding and Vision Transformer (ViT) architecture.

use as_autograd::{Tape, VarId};
use as_core::{DType, Error, Result};
use as_nn::{LayerNorm, Linear, Module, Param, Rng, TransformerBlock, xavier_uniform};
use as_tensor::Tensor;

/// Configuration for a [`VisionTransformer`] model.
#[derive(Debug, Clone)]
pub struct VitConfig {
    /// Input image height and width (square images).
    pub image_size: usize,
    /// Square patch dimension (must divide `image_size` evenly).
    pub patch_size: usize,
    /// Input image channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Transformer embedding dimension.
    pub embed_dim: usize,
    /// Number of Transformer encoder blocks.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feed-forward hidden dimension.
    pub ff_hidden_dim: usize,
    /// Number of output classification classes.
    pub num_classes: usize,
}

/// Non-overlapping image patch extractor and linear projection `[N, C, H, W] -> [N, P, D]`.
#[derive(Debug, Clone)]
pub struct PatchEmbed {
    /// Linear projection from flattened patch `C * patch_size^2` to `embed_dim`.
    pub proj: Linear,
    patch_size: usize,
    in_channels: usize,
    num_patches: usize,
}

impl PatchEmbed {
    /// Creates a new [`PatchEmbed`] layer.
    pub fn new(
        prefix: &str,
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        dtype: DType,
        rng: &mut Rng,
    ) -> Result<Self> {
        if patch_size == 0 || image_size == 0 || image_size % patch_size != 0 {
            return Err(Error::InvalidArgument {
                context: "PatchEmbed::new",
                reason: format!(
                    "image_size ({image_size}) must be a positive multiple of patch_size ({patch_size})"
                ),
            });
        }
        let grid = image_size / patch_size;
        let num_patches = grid * grid;
        let patch_dim = in_channels * patch_size * patch_size;
        let proj = Linear::new(&format!("{prefix}.proj"), patch_dim, embed_dim, true, dtype, rng)?;
        Ok(Self {
            proj,
            patch_size,
            in_channels,
            num_patches,
        })
    }

    /// Returns the number of patches per image.
    pub const fn num_patches(&self) -> usize {
        self.num_patches
    }

    /// Extracts patches from `images` (`[N, C, H, W]`) and projects to `[N, num_patches, embed_dim]`.
    pub fn forward(&self, tape: &mut Tape, images: &Tensor) -> Result<(VarId, Vec<VarId>)> {
        if images.rank() != 4 {
            return Err(Error::InvalidRank {
                context: "PatchEmbed::forward",
                rank: images.rank(),
                expected: 4,
            });
        }
        let n = images.shape().dim(0)?;
        let c = images.shape().dim(1)?;
        let h = images.shape().dim(2)?;
        let w = images.shape().dim(3)?;
        if c != self.in_channels || h % self.patch_size != 0 || w % self.patch_size != 0 {
            return Err(Error::InvalidArgument {
                context: "PatchEmbed::forward",
                reason: format!(
                    "image shape [{n}, {c}, {h}, {w}] incompatible with channels={} patch_size={}",
                    self.in_channels, self.patch_size
                ),
            });
        }
        let gh = h / self.patch_size;
        let gw = w / self.patch_size;
        let num_p = gh * gw;
        let patch_dim = c * self.patch_size * self.patch_size;
        let src = images.to_f32_vec()?;

        let mut patches = Vec::with_capacity(n * num_p * patch_dim);
        for batch in 0..n {
            for py in 0..gh {
                for px in 0..gw {
                    for ch in 0..c {
                        for dy in 0..self.patch_size {
                            let y = py * self.patch_size + dy;
                            for dx in 0..self.patch_size {
                                let x = px * self.patch_size + dx;
                                let idx = ((batch * c + ch) * h + y) * w + x;
                                patches.push(src.get(idx).copied().unwrap_or(0.0));
                            }
                        }
                    }
                }
            }
        }

        let patch_tensor = Tensor::from_f32(&patches, vec![n, num_p, patch_dim])?;
        let patch_var = tape.var(patch_tensor, false);
        self.proj.forward(tape, patch_var)
    }
}

impl Module for PatchEmbed {
    fn parameters(&self) -> Vec<&Param> {
        self.proj.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        self.proj.parameters_mut()
    }
}

/// Complete Vision Transformer (ViT) for image feature extraction and classification.
#[derive(Debug, Clone)]
pub struct VisionTransformer {
    /// Patch embedding projection.
    pub patch_embed: PatchEmbed,
    /// Learnable positional embeddings `[1, num_patches, embed_dim]`.
    pub pos_embed: Param,
    /// Stack of bidirectional Transformer encoder blocks.
    pub blocks: Vec<TransformerBlock>,
    /// Final LayerNorm before pooling.
    pub norm: LayerNorm,
    /// Linear classification head `[num_classes, embed_dim]`.
    pub head: Linear,
}

impl VisionTransformer {
    /// Constructs a deterministically initialized [`VisionTransformer`].
    pub fn new(cfg: &VitConfig, seed: u64) -> Result<Self> {
        let mut rng = Rng::from_seed(seed);
        let patch_embed = PatchEmbed::new(
            "patch_embed",
            cfg.image_size,
            cfg.patch_size,
            cfg.in_channels,
            cfg.embed_dim,
            DType::F32,
            &mut rng,
        )?;
        let pos_data = xavier_uniform(
            vec![1, patch_embed.num_patches(), cfg.embed_dim],
            patch_embed.num_patches(),
            cfg.embed_dim,
            DType::F32,
            &mut rng,
        )?;
        let pos_embed = Param::new("pos_embed", pos_data);

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(TransformerBlock::new(
                &format!("blocks.{i}"),
                cfg.embed_dim,
                cfg.num_heads,
                cfg.ff_hidden_dim,
                DType::F32,
                &mut rng,
            )?);
        }
        let norm = LayerNorm::new("norm", cfg.embed_dim, 1e-5, DType::F32)?;
        let head = Linear::new(
            "head",
            cfg.embed_dim,
            cfg.num_classes,
            true,
            DType::F32,
            &mut rng,
        )?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            head,
        })
    }

    /// Extracts pooled image embeddings of shape `[N, embed_dim]` (mean pooling over patch tokens).
    pub fn extract_embeddings(&self, tape: &mut Tape, images: &Tensor) -> Result<(VarId, Vec<VarId>)> {
        let (tokens, mut params) = self.patch_embed.forward(tape, images)?;
        let pos_var = self.pos_embed.register(tape);
        params.push(pos_var);
        let mut h = tape.add(tokens, pos_var)?;

        for blk in &self.blocks {
            let (next_h, p_blk) = blk.forward(tape, h, false)?;
            params.extend(p_blk);
            h = next_h;
        }

        let (normed, p_norm) = self.norm.forward(tape, h)?;
        params.extend(p_norm);

        // Mean pool across patch dimension (axis 1): [N, P, D] -> [N, D]
        let pooled = tape.mean_axis(normed, 1, false)?;
        Ok((pooled, params))
    }

    /// Runs full ViT classification returning logits of shape `[N, num_classes]`.
    pub fn forward(&self, tape: &mut Tape, images: &Tensor) -> Result<(VarId, Vec<VarId>)> {
        let (emb, mut params) = self.extract_embeddings(tape, images)?;
        let (logits, p_head) = self.head.forward(tape, emb)?;
        params.extend(p_head);
        Ok((logits, params))
    }
}

impl Module for VisionTransformer {
    fn parameters(&self) -> Vec<&Param> {
        let mut p = self.patch_embed.parameters();
        p.push(&self.pos_embed);
        for blk in &self.blocks {
            p.extend(blk.parameters());
        }
        p.extend(self.norm.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Param> {
        let mut p = self.patch_embed.parameters_mut();
        p.push(&mut self.pos_embed);
        for blk in &mut self.blocks {
            p.extend(blk.parameters_mut());
        }
        p.extend(self.norm.parameters_mut());
        p.extend(self.head.parameters_mut());
        p
    }
}
