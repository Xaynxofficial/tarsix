//! Searchable Model Hub combining [`ModelCache`] and [`ModelCard`] metadata registry.

use std::collections::BTreeMap;

use as_core::{Error, MemoryLimits, Result};
use as_io::{SafeTensors, deserialize_safetensors};

use crate::cache::ModelCache;
use crate::manifest::{ModelArtifact, ModelCard, ModelQuery};

/// Local & enterprise Model Hub for discovering, verifying, and loading models.
#[derive(Debug)]
pub struct ModelHub {
    cache: ModelCache,
    registry: BTreeMap<String, ModelCard>,
}

impl ModelHub {
    /// Opens a [`ModelHub`] backed by `cache`.
    pub fn new(cache: ModelCache) -> Self {
        Self {
            cache,
            registry: BTreeMap::new(),
        }
    }

    /// Returns a reference to the underlying [`ModelCache`].
    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }

    /// Registers or updates a [`ModelCard`] in the hub index.
    pub fn register(&mut self, card: ModelCard) -> Result<()> {
        if card.id.trim().is_empty() {
            return Err(Error::InvalidArgument {
                context: "ModelHub::register",
                reason: "model id cannot be empty".to_owned(),
            });
        }
        self.registry.insert(card.id.clone(), card);
        Ok(())
    }

    /// Publishes raw Safetensors weight bytes into the content-addressable cache and
    /// attaches the resulting SHA-256 [`ModelArtifact`] to `card`, registering it in the hub.
    pub fn publish_safetensors(
        &mut self,
        mut card: ModelCard,
        filename: impl Into<String>,
        safetensors_bytes: &[u8],
    ) -> Result<ModelArtifact> {
        let digest = self.cache.put_bytes(safetensors_bytes)?;
        let artifact = ModelArtifact {
            filename: filename.into(),
            sha256: digest,
            size_bytes: safetensors_bytes.len(),
        };
        card.artifacts.push(artifact.clone());
        self.register(card)?;
        Ok(artifact)
    }

    /// Looks up a registered [`ModelCard`] by its exact ID.
    pub fn get(&self, id: &str) -> Option<&ModelCard> {
        self.registry.get(id)
    }

    /// Lists all registered models in deterministic ID order.
    pub fn list(&self) -> Vec<&ModelCard> {
        self.registry.values().collect()
    }

    /// Searches the hub using a structured [`ModelQuery`].
    pub fn search(&self, query: &ModelQuery) -> Vec<&ModelCard> {
        self.registry
            .values()
            .filter(|card| query.matches(card))
            .collect()
    }

    /// Verifies the SHA-256 integrity of every artifact registered under `model_id`.
    pub fn verify_model(&self, model_id: &str) -> Result<usize> {
        let card = self.get(model_id).ok_or_else(|| Error::InvalidArgument {
            context: "ModelHub::verify_model",
            reason: format!("unknown model `{model_id}`"),
        })?;
        for art in &card.artifacts {
            let bytes = self.cache.get_verified(&art.sha256)?;
            if bytes.len() != art.size_bytes {
                return Err(Error::InvalidArgument {
                    context: "ModelHub::verify_model",
                    reason: format!(
                        "size mismatch for `{}`: expected {}, got {}",
                        art.filename,
                        art.size_bytes,
                        bytes.len()
                    ),
                });
            }
        }
        Ok(card.artifacts.len())
    }

    /// Loads and SHA-256 verifies the first `.safetensors` artifact of `model_id` into [`SafeTensors`].
    pub fn load_safetensors(&self, model_id: &str, limits: MemoryLimits) -> Result<SafeTensors> {
        let card = self.get(model_id).ok_or_else(|| Error::InvalidArgument {
            context: "ModelHub::load_safetensors",
            reason: format!("unknown model `{model_id}`"),
        })?;
        let art = card
            .artifacts
            .iter()
            .find(|a| a.filename.ends_with(".safetensors"))
            .ok_or_else(|| Error::InvalidArgument {
                context: "ModelHub::load_safetensors",
                reason: format!("model `{model_id}` has no .safetensors artifact"),
            })?;
        let bytes = self.cache.get_verified(&art.sha256)?;
        deserialize_safetensors(&bytes, limits)
    }
}
