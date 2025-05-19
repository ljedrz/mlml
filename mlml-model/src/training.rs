// This module trains a text classification model using the provided training and testing datasets,
// as well as the provided configuration. It first initializes a tokenizer and batchers for the datasets,
// then initializes the model and data loaders for the datasets. The function then initializes
// an optimizer and a learning rate scheduler, and uses them along with the model and datasets
// to build a learner, which is used to train the model. The trained model and the configuration are
// then saved to the specified directory.

use crate::{
    data::{CharTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer},
    model::TextClassificationModelConfig,
};
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
use std::sync::Arc;

// Define configuration struct for the experiment
#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 256)]
    pub max_seq_length: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 50)]
    pub num_epochs: usize,
}

// Define train function
pub fn train<B: AutodiffBackend, D: TextClassificationDataset + 'static>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,        // Training dataset
    dataset_test: D,         // Testing dataset
    config: ExperimentConfig, // Experiment configuration
    artifact_dir: &str,      // Directory to save model and config files
) {
    // Initialize tokenizer
    let tokenizer = Arc::new(CharTokenizer::default());

    // Initialize batcher
    let batcher = TextClassificationBatcher::new(tokenizer.clone(), config.max_seq_length);

    // Initialize model
    let model = TextClassificationModelConfig::new(
        config.transformer.clone(),
        2,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init::<B>(&devices[0]);

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(1)
        .shuffle(1234567) // FIXME
        .build(SamplerDataset::new(dataset_train, 21_250));
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(1)
        .shuffle(7777777) // FIXME
        .build(SamplerDataset::new(dataset_test, 2_500));

    // Initialize optimizer
    let optim = config.optimizer.init();

    // Initialize learning rate scheduler
    let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(2e-5, 50)
        .init()
        .unwrap();

    let early_stopping = MetricEarlyStoppingStrategy::new(
        &AccuracyMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Highest,
        Split::Valid,
        StoppingCondition::NoImprovementSince { n_epochs: 5 },
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
        .num_epochs(config.num_epochs)
        .early_stopping(early_stopping)
        .summary()
        .build(model, optim, lr_scheduler);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
