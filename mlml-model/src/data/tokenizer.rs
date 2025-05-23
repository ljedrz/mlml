// This trait represents the common interface for all tokenizer types.
// The `Send + Sync` bounds are necessary for allowing these operations
// to work across thread boundaries.

use std::collections::HashMap;

use smol_str::{SmolStr, ToSmolStr};

#[derive(Debug)]
pub struct MlmlTokenizer {
    vocab: HashMap<SmolStr, usize>,     // Token to ID
    inv_vocab: HashMap<usize, SmolStr>, // ID to token
    max_seq_length: usize,
}

const VALUES: &[&str] = &["true", "false"];
const OPERATORS: &[&str] = &["∧", "∨", "¬", "→", "↔"];
const MISC: &[&str] = &["[", "]", ":", ",", "(", ")"];
const STRUCT: &[&str] = &[
    "<pad>",
    "<assign>",
    "</assign>",
    "<operator_prefix>",
    "<value_prefix>",
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

impl MlmlTokenizer {
    pub fn new(max_seq_length: usize, max_vars: usize) -> Self {
        let mut tokens = Vec::new();
        tokens.extend(STRUCT.iter().map(|s| SmolStr::new_inline(s)));
        tokens.extend(MISC.iter().map(|s| SmolStr::new_inline(s)));
        tokens.extend(VALUES.iter().map(|s| SmolStr::new_inline(s)));
        tokens.extend(OPERATORS.iter().map(|s| SmolStr::new_inline(s)));
        let vars = (0..max_vars).map(|n| SmolStr::from(format!("<var{n}>")));
        tokens.extend(vars);

        let vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        let inv_vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.clone()))
            .collect();
        MlmlTokenizer {
            vocab,
            inv_vocab,
            max_seq_length,
        }
    }
}

impl Tokenizer for MlmlTokenizer {
    fn encode(&self, input: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = input.chars().collect();
        let mut vars: HashMap<char, usize> = Default::default();

        let mut in_state = false;
        let mut in_expr = false;
        let mut in_assignment = false;

        while i < chars.len() {
            if i + 3 < chars.len() && &chars[i..i + 4] == ['t', 'r', 'u', 'e'] {
                tokens.push(self.vocab["<value_prefix>"]);
                tokens.push(self.vocab["true"]);
                if in_assignment {
                    tokens.push(self.vocab["</assign>"]);
                    in_assignment = false;
                }
                i += 4;
            } else if i + 4 < chars.len() && &chars[i..i + 5] == ['f', 'a', 'l', 's', 'e'] {
                tokens.push(self.vocab["<value_prefix>"]);
                tokens.push(self.vocab["false"]);
                if in_assignment {
                    tokens.push(self.vocab["</assign>"]);
                    in_assignment = false;
                }
                i += 5;
            } else if OPERATORS.contains(&&*chars[i].to_smolstr()) {
                tokens.push(self.vocab["<operator_prefix>"]);
                tokens.push(self.vocab[&chars[i].to_smolstr()]);
                i += 1;
            } else if chars[i].is_whitespace() || chars[i] == ';' {
                i += 1;
            } else if chars[i] == '[' {
                in_state = true;
                tokens.push(self.vocab["["]);
                i += 1;
            } else if chars[i] == ']' {
                tokens.push(self.vocab["]"]);
                in_state = false;
                tokens.push(self.vocab["("]);
                in_expr = true;
                i += 1;
            } else if chars[i].is_alphabetic() {
                if in_state && !in_assignment {
                    in_assignment = true;
                    tokens.push(self.vocab["<assign>"]);
                }
                if !vars.contains_key(&chars[i]) {
                    let pos = vars.len();
                    vars.insert(chars[i], pos);
                }
                let var = SmolStr::from(format!("<var{}>", vars.get(&chars[i]).unwrap()));
                tokens.push(
                    self.vocab
                        .get(&var)
                        .copied()
                        .expect(&format!("missing token: '{}'", chars[i])),
                );
                i += 1;
            } else {
                // Remaining single-char tokens
                let c = chars[i].to_smolstr();
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
            tokens.push(self.vocab[")"]);
        }

        assert!(tokens.len() <= self.max_seq_length);
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
            .collect::<Vec<SmolStr>>()
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
        let expr_with_state_str = "[i, f: true; g, j: false] ((g ∧ (¬j → i)) ∧ (f ∨ j))";
        let tokenizer = MlmlTokenizer::new(64);
        let tokens = tokenizer.encode(expr_with_state_str);
        let decoded = tokenizer.decode(&tokens);
        println!("{decoded}");
    }
}
