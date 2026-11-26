//! Complete model training pipeline: [`Dataset`], [`DataLoader`], [`MetricsTracker`],
//! and integrity-verified binary [`checkpoint`] persistence.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod checkpoint;
pub mod csv;
pub mod data;
pub mod metrics;
pub mod plot;
pub mod trainer;

pub use checkpoint::{inspect_checkpoint_shapes, load_checkpoint, save_checkpoint};
pub use csv::{
    format_csv_dataset, generate_synthetic_classification_dataset, parse_csv_dataset,
    train_val_split,
};
pub use data::{Batch, DataLoader, Dataset, InMemoryDataset, Sample};
pub use metrics::{EpochMetrics, MetricsTracker, classification_accuracy};
pub use plot::render_training_svg;
pub use trainer::{
    BestCheckpointCallback, Callback, CallbackAction, EarlyStopping, EpochContext,
    JsonlLoggerCallback, StepContext, TrainConfig, TrainReport, TrainState, Trainer,
};
