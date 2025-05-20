// In each implementation, the batch function is defined to convert a vector of items into a batch.
// For training, the items are instances of Item and include both the text and the corresponding
// label. For inference, the items are simply strings without labels. The function tokenizes the
// text, generates a padding mask, and returns a batch object.

use std::sync::Arc;

use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, prelude::*};

use super::{dataset::MlmlItem, tokenizer::Tokenizer};

/// Struct for batching text classification items
#[derive(Clone, new)]
pub struct MlmlBatcher {
    tokenizer: Arc<dyn Tokenizer>, // Tokenizer for converting text to token IDs
    max_seq_length: usize,         // Maximum sequence length for tokenized text
}

#[derive(Debug, Clone, new)]
pub struct TrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub labels: Tensor<B, 1, Int>,    // Labels of the text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

#[derive(Debug, Clone, new)]
pub struct InferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

/// Implement Batcher trait for Batcher struct for training
impl<B: Backend> Batcher<B, MlmlItem, TrainingBatch<B>> for MlmlBatcher {
    /// Batches a vector of text classification items into a training batch
    fn batch(&self, items: Vec<MlmlItem>, device: &B::Device) -> TrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        // Tokenize text and create label tensor for each item
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                device,
            ));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            device,
        );

        // Create and return training batch
        TrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_pad: mask.mask,
        }
    }
}

/// Implement Batcher trait for Batcher struct for inference
impl<B: Backend> Batcher<B, String, InferenceBatch<B>> for MlmlBatcher {
    /// Batches a vector of strings into an inference batch
    fn batch(&self, items: Vec<String>, device: &B::Device) -> InferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            device,
        );

        // Create and return inference batch
        InferenceBatch {
            tokens: mask.tensor.to_device(device),
            mask_pad: mask.mask.to_device(device),
        }
    }
}
