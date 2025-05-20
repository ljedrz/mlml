use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    process::Command,
    str,
};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct MlmlConfig {
    pub dataset: DatasetConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub train_samples_count: usize,
    pub valid_samples_count: usize,
    pub test_samples_count: usize,
    pub max_seq_length: usize,
    pub max_variables: usize,
    pub max_depth: usize,
    pub db_path: PathBuf,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,

    pub dropout: f64,
    pub weight_decay: f32,
    pub gradient_clipping_norm: f32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    pub initial_lr: f64,
    pub min_lr: f64,

    pub batch_size: usize,
    pub num_epochs: usize,
    pub early_stopping_epochs: usize,
}

pub fn config_path() -> PathBuf {
    let output = Command::new(env!("CARGO"))
        .arg("locate-project")
        .arg("--workspace")
        .arg("--message-format=plain")
        .output()
        .unwrap()
        .stdout;
    let cargo_path = Path::new(str::from_utf8(&output).unwrap().trim());

    let mut config_path = cargo_path.parent().unwrap().to_path_buf();
    config_path.push("config.json");

    config_path
}
