// This trait represents the common interface for all tokenizer types.
// The `Send + Sync` bounds are necessary for allowing these operations
// to work across thread boundaries.

use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;

#[allow(dead_code)]
pub trait Tokenizer: Send + Sync {
    /// Converts a text string into a sequence of tokens.
    fn encode(&self, value: &str) -> Vec<usize>;

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Gets the size of the tokenizer's vocabulary.
    fn vocab_size(&self) -> usize;

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize;

    /// Gets the string representation of the padding token.
    /// The default implementation uses `decode` on the padding token.
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

pub struct CharTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: Vec<String>,
    pad_token: usize,
}

impl Default for CharTokenizer {
    fn default() -> Self {
        let data = [" (),:[]abcdefghijklmnopqrstuvwxyz¬→↔∧∨"];
        CharTokenizer::from_dataset(&data, "␣")
    }
}

impl CharTokenizer {
    pub fn from_dataset(dataset: &[&str], pad_token_str: &str) -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = Vec::new();

        // Insert padding token first
        vocab.insert(pad_token_str.to_string(), 0);
        inv_vocab.push(pad_token_str.to_string());

        for entry in dataset {
            for g in entry.graphemes(true) {
                if !vocab.contains_key(g) {
                    vocab.insert(g.to_string(), inv_vocab.len());
                    inv_vocab.push(g.to_string());
                }
            }
        }

        Self {
            vocab,
            inv_vocab,
            pad_token: 0,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, value: &str) -> Vec<usize> {
        value
            .graphemes(true)
            .map(|g| *self.vocab.get(g).unwrap_or(&self.pad_token)) // unknowns go to pad token
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| self.inv_vocab.get(id).cloned().unwrap_or("?".to_string()))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.inv_vocab.len()
    }

    fn pad_token(&self) -> usize {
        self.pad_token
    }
}
