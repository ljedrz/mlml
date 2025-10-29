// This module defines the inference process for a classification model.
// It loads a model and its configuration from a directory, and uses a tokenizer
// and a batcher to prepare the input data. The model is then used to make predictions
// on the input samples, and the results are printed out for each sample.

use std::{str::FromStr, sync::Arc};

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use mlml_util::MlmlConfig;

use crate::{
    data::{MlmlBatcher, MlmlDataset, MlmlTokenizer, Tokenizer},
    model::MlmlModelConfig,
    training::ExperimentConfig,
};

// Define inference function
pub fn infer<B: Backend, D: MlmlDataset + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    test_samples: Vec<(String, String, usize, f32)>, // Text samples for inference
    mlml_config: MlmlConfig,
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(MlmlTokenizer::new(
        mlml_config.dataset.max_seq_length,
        mlml_config.dataset.max_variables,
    ));

    // Get number of classes from dataset
    let n_classes = 2;

    // Initialize batcher for batching samples
    let batcher = Arc::new(MlmlBatcher::new(
        tokenizer.clone(),
        mlml_config.dataset.max_seq_length,
    ));

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights tb");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = MlmlModelConfig::new(
        config.transformer,
        n_classes,
        tokenizer.vocab_size(),
        mlml_config.dataset.max_seq_length,
    )
    .init::<B>(&device)
    .load_record(record); // Initialize model with loaded weights

    // Run inference on the given text samples
    println!("Running inference ...");
    let item = batcher.batch(
        test_samples.iter().map(|(expr, ..)| expr.clone()).collect(),
        &device,
    ); // Batch samples using the batcher
    let predictions = model.infer(item); // Get model predictions

    // Print out predictions for each sample
    let mut misses = Vec::new();
    for (i, (expr, expected_ret, complexity, rarity)) in test_samples.iter().enumerate() {
        #[allow(clippy::single_range_in_vec_init)]
        let prediction = predictions.clone().slice([i..i + 1]); // Get prediction for current sample
        let logits = prediction.to_data(); // Convert prediction tensor to data
        let class_index = prediction.argmax(1).squeeze_dim::<1>(1).into_scalar(); // Get class index with the highest value
        let class = (class_index.elem::<i32>() as usize) == 1; // Get class name

        let correct = <bool>::from_str(expected_ret).unwrap() == class;
        let marker = if correct { "" } else { "in" };

        // Print sample text, predicted logits and predicted class
        println!(
            "\n=== Item {i} ===\n- Expr: {expr}\n- Logits: {logits}\n- Prediction: \
             {class} ({marker}correct)\n================"
        );

        if !correct {
            misses.push((i, expr, complexity, rarity));
        }
    }

    let mut complexity_sum = 0;
    let mut rarity_sum = 0.0;
    for (_i, _miss, complexity, rarity) in &misses {
        complexity_sum += *complexity;
        rarity_sum += *rarity;
    }

    println!();
    println!(
        "\nmisses: {} ({}%); avg. complexity: {}, avg. rarity: {}",
        misses.len(),
        (misses.len() as f64 / test_samples.len() as f64) * 100.0,
        complexity_sum as f32 / misses.len() as f32,
        rarity_sum / misses.len() as f32,
    );
    println!(
        "min complexity: {}",
        misses.iter().map(|m| m.2).min().unwrap()
    );
}
