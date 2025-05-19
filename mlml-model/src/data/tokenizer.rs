// This trait represents the common interface for all tokenizer types.
// The `Send + Sync` bounds are necessary for allowing these operations
// to work across thread boundaries.

use std::collections::HashMap;

#[derive(Debug)]
pub struct CharTokenizer {
    vocab: HashMap<String, usize>,     // Token to ID
    inv_vocab: HashMap<usize, String>, // ID to token
    max_seq_length: usize,
}

const VALUES: &[&str] = &["true", "false"];
const OPERATORS: &[&str] = &["∧", "∨", "¬", "→", "↔"];
const ALPHABET: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z",
];
const MISC: &[&str] = &["[", "]", ":", ",", "(", ")"];
const STRUCT: &[&str] = &[
    "<pad>",
    "<state_start>",
    "<state_end>",
    "<assign_start>",
    "<assign_end>",
    "<expr_start>",
    "<expr_end>",
];

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

impl CharTokenizer {
    fn new(max_seq_length: usize) -> Self {
        let mut tokens = STRUCT.to_vec();
        tokens.extend_from_slice(MISC);
        tokens.extend_from_slice(VALUES);
        tokens.extend_from_slice(OPERATORS);
        tokens.extend_from_slice(ALPHABET);

        let vocab: HashMap<String, usize> = tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| (t.to_string(), i))
            .collect();
        let inv_vocab: HashMap<usize, String> = tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| (i, t.to_string()))
            .collect();
        CharTokenizer {
            vocab,
            inv_vocab,
            max_seq_length,
        }
    }
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new(150)
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, input: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = input.chars().collect();

        let mut in_state = false;
        let mut in_expr = false;
        let mut in_assignment = false;

        while i < chars.len() {
            // Handle multi-char tokens (true, false, operators)
            if i + 3 < chars.len() && &chars[i..i + 4] == ['t', 'r', 'u', 'e'] {
                tokens.push(self.vocab["true"]);
                if in_assignment {
                    tokens.push(self.vocab["<assign_end>"]);
                    in_assignment = false;
                }
                i += 4;
            } else if i + 4 < chars.len() && &chars[i..i + 5] == ['f', 'a', 'l', 's', 'e'] {
                tokens.push(self.vocab["false"]);
                if in_assignment {
                    tokens.push(self.vocab["<assign_end>"]);
                    in_assignment = false;
                }
                i += 5;
            } else if OPERATORS.contains(&&*chars[i].to_string()) {
                tokens.push(self.vocab[&chars[i].to_string()]);
                i += 1;
            } else if chars[i].is_whitespace() {
                i += 1;
            } else if chars[i] == '[' {
                in_state = true;
                tokens.push(self.vocab["<state_start>"]);
                i += 1;
            } else if chars[i] == ']' {
                tokens.push(self.vocab["<state_end>"]);
                in_state = false;
                tokens.push(self.vocab["<expr_start>"]);
                in_expr = true;
                i += 1;
            } else if ALPHABET.contains(&&*chars[i].to_string()) {
                if in_state {
                    in_assignment = true;
                    tokens.push(self.vocab["<assign_start>"]);
                }
                tokens.push(
                    self.vocab
                        .get(&chars[i].to_string())
                        .copied()
                        .expect(&format!("missing token: '{}'", chars[i])),
                );
                i += 1;
            } else {
                // Remaining single-char tokens
                let c = chars[i].to_string();
                tokens.push(
                    self.vocab
                        .get(&c)
                        .copied()
                        .expect(&format!("missing token: '{c}'")),
                );
                i += 1;
            }
        }

        if in_expr {
            tokens.push(self.vocab["<expr_end>"]);
        }

        // Pad to max_seq_length
        while tokens.len() < self.max_seq_length {
            tokens.push(self.vocab["<pad>"]);
        }
        tokens.truncate(self.max_seq_length);
        tokens
    }

    fn decode(&self, token_ids: &[usize]) -> String {
        token_ids
            .iter()
            .map(|&id| {
                self.inv_vocab
                    .get(&id)
                    .expect(&format!("missing token id: '{id}'"))
                    .clone()
            })
            .collect::<Vec<String>>()
            .join(" ")
    }

    /// Gets the size of the tokenizer's vocabulary.
    fn vocab_size(&self) -> usize {
        self.inv_vocab.len()
    }

    fn pad_token(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer() {
        let expr_with_state_str = "[m: false, q: false, t: true, t: true, t: false, w: true, x: false, y: false] (((w → t) ∧ (x ∧ t)) ∨ ((q → y) ∧ (m ↔ t)))";
        let tokenizer = CharTokenizer::default();
        let tokens = tokenizer.encode(expr_with_state_str);
        let decoded = tokenizer.decode(&tokens);
        println!("{decoded}");
    }
}
