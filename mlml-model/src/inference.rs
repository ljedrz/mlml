// This module defines the inference process for a text classification model.
// It loads a model and its configuration from a directory, and uses a tokenizer
// and a batcher to prepare the input data. The model is then used to make predictions
// on the input samples, and the results are printed out for each sample.
// Import required modules and types

use crate::{
    data::{CharTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer},
    model::TextClassificationModelConfig,
    training::ExperimentConfig,
};
use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use std::{str::FromStr, sync::Arc};

// Define inference function
pub fn infer<B: Backend, D: TextClassificationDataset + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    test_samples: Vec<(String, String)>, // Text samples for inference
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(CharTokenizer::default());

    // Get number of classes from dataset
    let n_classes = 2;

    // Initialize batcher for batching samples
    let batcher = Arc::new(TextClassificationBatcher::new(
        tokenizer.clone(),
        config.max_seq_length,
    ));

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights tb");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = TextClassificationModelConfig::new(
        config.transformer,
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init::<B>(&device)
    .load_record(record); // Initialize model with loaded weights

    // Run inference on the given text samples
    println!("Running inference ...");
    let item = batcher.batch(
        test_samples.iter().map(|(expr, _)| expr.clone()).collect(),
        &device,
    ); // Batch samples using the batcher
    let predictions = model.infer(item); // Get model predictions

    // Print out predictions for each sample
    let mut misses = Vec::new();
    for (i, (expr, expected_ret)) in test_samples.iter().enumerate() {
        #[allow(clippy::single_range_in_vec_init)]
        let prediction = predictions.clone().slice([i..i + 1]); // Get prediction for current sample
        let logits = prediction.to_data(); // Convert prediction tensor to data
        let class_index = prediction.argmax(1).squeeze::<1>(1).into_scalar(); // Get class index with the highest value
        let class = (class_index.elem::<i32>() as usize) == 1; // Get class name

        // Print sample text, predicted logits and predicted class
        println!(
            "\n=== Item {i} ===\n- Expr: {expr}\n- Logits: {logits}\n- Prediction: \
             {class}\n================"
        );

        let expected: bool = FromStr::from_str(&expected_ret).unwrap();
        if class != expected {
            misses.push((i, expr));
        }
    }

    println!();
    for (i, miss) in &misses {
        println!("{i}: {miss}");
    }
    println!(
        "\nmisses: {} ({}%)",
        misses.len(),
        (misses.len() as f64 / test_samples.len() as f64) * 100.0
    );
}
