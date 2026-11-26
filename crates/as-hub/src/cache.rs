//! Content-addressable, SHA-256 verified local artifact cache.

use std::fs;
use std::path::{Path, PathBuf};

use as_core::{Error, Result};

use crate::sha256::sha256_hex;

/// Summary statistics of a [`ModelCache`] directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of verified blob files stored in the cache.
    pub blob_count: usize,
    /// Total bytes occupied by cached blobs.
    pub total_bytes: u64,
}

/// Content-addressable local disk cache keyed by lowercase hex SHA-256 digest.
#[derive(Debug, Clone)]
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    /// Opens or creates a [`ModelCache`] at `root`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).map_err(|e| Error::InvalidArgument {
            context: "ModelCache::open",
            reason: format!("failed to create cache directory {}: {e}", blobs.display()),
        })?;
        Ok(Self { root })
    }

    /// Returns the root cache directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Stores `bytes` in the cache, returning its computed SHA-256 hex digest.
    pub fn put_bytes(&self, bytes: &[u8]) -> Result<String> {
        let digest = sha256_hex(bytes);
        let path = self.blob_path(&digest)?;
        if !path.exists() {
            fs::write(&path, bytes).map_err(|e| Error::InvalidArgument {
                context: "ModelCache::put_bytes",
                reason: format!("failed to write blob {}: {e}", path.display()),
            })?;
        }
        Ok(digest)
    }

    /// Reads the blob identified by `expected_sha256` and verifies its cryptographic integrity.
    pub fn get_verified(&self, expected_sha256: &str) -> Result<Vec<u8>> {
        let path = self.blob_path(expected_sha256)?;
        let data = fs::read(&path).map_err(|e| Error::InvalidArgument {
            context: "ModelCache::get_verified",
            reason: format!("missing cached blob `{expected_sha256}`: {e}"),
        })?;
        let actual = sha256_hex(&data);
        if actual != expected_sha256 {
            return Err(Error::InvalidArgument {
                context: "ModelCache::get_verified",
                reason: format!(
                    "SHA-256 integrity mismatch for blob: expected {expected_sha256}, got {actual}"
                ),
            });
        }
        Ok(data)
    }

    /// Returns `true` if a blob for `sha256` exists in the cache.
    pub fn contains(&self, digest: &str) -> bool {
        self.blob_path(digest).is_ok_and(|p| p.exists())
    }

    /// Removes a cached blob by its SHA-256 digest.
    pub fn remove(&self, digest: &str) -> Result<()> {
        let path = self.blob_path(digest)?;
        if path.exists() {
            fs::remove_file(&path).map_err(|e| Error::InvalidArgument {
                context: "ModelCache::remove",
                reason: format!("failed to remove blob: {e}"),
            })?;
        }
        Ok(())
    }

    /// Computes total blob count and byte usage in the cache.
    pub fn stats(&self) -> Result<CacheStats> {
        let blobs_dir = self.root.join("blobs");
        let entries = fs::read_dir(&blobs_dir).map_err(|e| Error::InvalidArgument {
            context: "ModelCache::stats",
            reason: format!("failed to read blobs dir: {e}"),
        })?;
        let mut blob_count = 0usize;
        let mut total_bytes = 0u64;
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata()
                && meta.is_file()
            {
                blob_count += 1;
                total_bytes = total_bytes.saturating_add(meta.len());
            }
        }
        Ok(CacheStats {
            blob_count,
            total_bytes,
        })
    }

    fn blob_path(&self, digest: &str) -> Result<PathBuf> {
        if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::InvalidArgument {
                context: "ModelCache::blob_path",
                reason: "invalid SHA-256 digest format (must be 64 hex chars)".to_owned(),
            });
        }
        Ok(self.root.join("blobs").join(digest.to_ascii_lowercase()))
    }
}
