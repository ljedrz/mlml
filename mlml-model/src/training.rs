// This module trains a classification model using the provided training and testing datasets,
// as well as the provided configuration. It first initializes a tokenizer and batchers for the datasets,
// then initializes the model and data loaders for the datasets. The function then initializes
// an optimizer and a learning rate scheduler, and uses them along with the model and datasets
// to build a learner, which is used to train the model. The trained model and the configuration are
// then saved to the specified directory.

use std::sync::Arc;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamWConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, IterationSpeedMetric, LearningRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use mlml_util::MlmlConfig;

use crate::{
    data::{MlmlBatcher, MlmlDataset, MlmlTokenizer, Tokenizer},
    model::MlmlModelConfig,
};

// Define configuration struct for the experiment
#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamWConfig,
}

// Define train function
pub fn train<B: AutodiffBackend, D: MlmlDataset + 'static>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,        // Training dataset
    dataset_valid: D,        // Validation dataset
    config: ExperimentConfig, // Experiment configuration
    artifact_dir: &str,      // Directory to save model and config files
    mlml_config: MlmlConfig,
) {
    // Initialize tokenizer
    let tokenizer = Arc::new(MlmlTokenizer::new(mlml_config.dataset.max_seq_length));

    // Initialize batcher
    let batcher = MlmlBatcher::new(tokenizer.clone(), mlml_config.dataset.max_seq_length);

    // Initialize model
    let model = MlmlModelConfig::new(
        config.transformer.clone(),
        2,
        tokenizer.vocab_size(),
        mlml_config.dataset.max_seq_length,
    )
    .init::<B>(&devices[0]);

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(mlml_config.training.batch_size)
        .num_workers(1)
        .shuffle(7777777) // FIXME
        .build(SamplerDataset::new(
            dataset_train,
            mlml_config.dataset.train_samples_count,
        ));
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(mlml_config.training.batch_size)
        .num_workers(1)
        .shuffle(7777777) // FIXME
        .build(SamplerDataset::new(
            dataset_valid,
            mlml_config.dataset.valid_samples_count,
        ));

    // Initialize optimizer
    let optim = config.optimizer.init();

    // Initialize learning rate scheduler
    let iters = mlml_config.training.num_epochs
        * mlml_config
            .dataset
            .train_samples_count
            .div_ceil(mlml_config.training.batch_size);
    let lr_scheduler =
        CosineAnnealingLrSchedulerConfig::new(mlml_config.training.initial_lr, iters)
            .with_min_lr(mlml_config.training.min_lr)
            .init()
            .unwrap();

    let early_stopping = MetricEarlyStoppingStrategy::new(
        &AccuracyMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Highest,
        Split::Valid,
        StoppingCondition::NoImprovementSince {
            n_epochs: mlml_config.training.early_stopping_epochs,
        },
    );

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(IterationSpeedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices)
        .num_epochs(mlml_config.training.num_epochs)
        .early_stopping(early_stopping)
        .summary()
        .build(model, optim, lr_scheduler);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
