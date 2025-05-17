use std::ascii::AsciiExt;
use std::collections::HashSet;

use crate::parser::Expr;
use rand::Rng;
use rand::distributions::{Alphanumeric, Distribution, WeightedIndex};
use rand::seq::IteratorRandom;

// --- Generator ---
pub struct ExprGenerator {
    max_depth: usize,
    max_vars: usize,
    weights: Weights,
}

#[derive(Clone)]
pub struct Weights {
    pub var: f32,
    pub and: f32,
    pub or: f32,
    pub not: f32,
}

impl ExprGenerator {
    pub fn new(max_depth: usize, max_vars: usize, weights: Weights) -> Self {
        ExprGenerator {
            max_depth,
            max_vars,
            weights,
        }
    }

    pub fn generate(&self) -> Expr {
        let mut vars = HashSet::new();
        self.generate_with_depth(0, &mut vars)
    }

    fn generate_with_depth(&self, depth: usize, vars: &mut HashSet<char>) -> Expr {
        let mut rng = rand::thread_rng();

        // At max depth, only generate variables
        if depth >= self.max_depth {
            let var = if vars.len() < self.max_vars {
                let v = self.random_variable(&mut rng);
                vars.insert(v);
                v
            } else {
                *vars.iter().choose(&mut rng).unwrap()
            };

            return Expr::Var(var);
        }

        // Choose production based on weights
        let choices = [
            ("var", self.weights.var),
            ("and", self.weights.and),
            ("or", self.weights.or),
            ("not", self.weights.not),
        ];
        let dist = WeightedIndex::new(choices.iter().map(|x| x.1)).unwrap();
        let choice = choices[dist.sample(&mut rng)].0;

        match choice {
            "var" => {
                let var = if vars.len() < self.max_vars {
                    let v = self.random_variable(&mut rng);
                    vars.insert(v);
                    v
                } else {
                    *vars.iter().choose(&mut rng).unwrap()
                };
                Expr::Var(var)
            },
            "and" => Expr::And(
                Box::new(self.generate_with_depth(depth + 1, vars)),
                Box::new(self.generate_with_depth(depth + 1, vars)),
            ),
            "or" => Expr::Or(
                Box::new(self.generate_with_depth(depth + 1, vars)),
                Box::new(self.generate_with_depth(depth + 1, vars)),
            ),
            "not" => Expr::Not(Box::new(self.generate_with_depth(depth + 1, vars))),
            _ => unreachable!(),
        }
    }

    fn random_variable(&self, rng: &mut impl Rng) -> char {
        let mut c = rng
            .sample_iter(&Alphanumeric)
            .filter(|c| (*c as char).is_alphabetic())
            .take(1)
            .next()
            .unwrap() as char;
        c.make_ascii_lowercase();
        
        c
    }
}

// --- String Conversion ---
impl Expr {
    pub fn to_string(&self) -> String {
        match self {
            Expr::Var(s) => s.to_string(),
            Expr::And(l, r) => format!("({} ∧ {})", l.to_string(), r.to_string()),
            Expr::Or(l, r) => format!("({} ∨ {})", l.to_string(), r.to_string()),
            Expr::Not(e) => format!("¬{}", e.to_string()),
        }
    }
}
