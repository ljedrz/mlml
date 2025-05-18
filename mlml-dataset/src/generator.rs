use std::collections::{HashMap, HashSet};

use crate::parser::Expr;
use rand::Rng;
use rand::distributions::{Alphanumeric, Distribution, WeightedIndex};
use rand::seq::IteratorRandom;

pub struct ExprGenerator {
    max_depth: usize,
    max_vars: usize,
    weights: Weights,
}

#[derive(Clone)]
pub struct Weights {
    pub var: f32,
    pub not: f32,
    pub and: f32,
    pub or: f32,
    pub implies: f32,
    pub equivalent: f32,
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
            ("not", self.weights.not),
            ("and", self.weights.and),
            ("or", self.weights.or),
            ("impl", self.weights.implies),
            ("equiv", self.weights.equivalent),
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
            }
            "not" => Expr::Not(Box::new(self.generate_with_depth(depth + 1, vars))),
            "and" => Expr::And(
                Box::new(self.generate_with_depth(depth + 1, vars)),
                Box::new(self.generate_with_depth(depth + 1, vars)),
            ),
            "or" => Expr::Or(
                Box::new(self.generate_with_depth(depth + 1, vars)),
                Box::new(self.generate_with_depth(depth + 1, vars)),
            ),
            "impl" => Expr::Implies(
                Box::new(self.generate_with_depth(depth + 1, vars)),
                Box::new(self.generate_with_depth(depth + 1, vars)),
            ),
            "equiv" => Expr::Equivalent(
                Box::new(self.generate_with_depth(depth + 1, vars)),
                Box::new(self.generate_with_depth(depth + 1, vars)),
            ),
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

pub fn generate_state(expr: &Expr) -> Box<[(char, bool)]> {
    let mut state = Vec::new();
    let mut rng = rand::thread_rng();
    generate_state_recurse(expr, &mut state, &mut rng);
    state.sort_unstable_by_key(|(c, _)| *c);

    state.into_boxed_slice()
}

fn generate_state_recurse(expr: &Expr, state: &mut Vec<(char, bool)>, rng: &mut impl Rng) {
    match expr {
        Expr::Var(c) => {
            state.push((*c, rng.gen_bool(0.5)));
        }
        Expr::Not(e) => generate_state_recurse(e, state, rng),
        Expr::And(l, r) | Expr::Or(l, r) | Expr::Implies(l, r) | Expr::Equivalent(l, r) => {
            generate_state_recurse(l, state, rng);
            generate_state_recurse(r, state, rng);
        }
    }
}
