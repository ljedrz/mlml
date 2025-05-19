use std::{collections::HashSet, ops::RangeInclusive};

use crate::parser::{BinaryOp, BinaryOpType, Expr};
use rand::Rng;
use rand::seq::IteratorRandom;

pub struct ExprGenerator {
    max_depth: usize,
    max_vars: usize,
}

impl ExprGenerator {
    pub fn new(max_depth: usize, max_vars: usize) -> Self {
        ExprGenerator {
            max_depth,
            max_vars,
        }
    }

    pub fn generate(&self, range: &RangeInclusive<char>) -> Expr {
        let mut vars = HashSet::new();
        self.generate_with_depth(0, range, &mut vars)
    }

    fn generate_with_depth(
        &self,
        depth: usize,
        range: &RangeInclusive<char>,
        vars: &mut HashSet<char>,
    ) -> Expr {
        let mut rng = rand::thread_rng();

        // At max depth, only generate variables
        if depth >= self.max_depth || rng.gen_bool(0.65) {
            let var = if vars.len() < self.max_vars {
                let v = self.random_variable(&range, &mut rng);
                vars.insert(v);
                v
            } else {
                *vars.iter().choose(&mut rng).unwrap()
            };

            return Expr::Var(var);
        }

        // Choose production based on weights
        let choices = ["var", "not", "and", "or", "impl", "equiv"];
        let choice = choices.iter().choose(&mut rng).unwrap();

        match &**choice {
            "var" => {
                let var = if vars.len() < self.max_vars {
                    let v = self.random_variable(range, &mut rng);
                    vars.insert(v);
                    v
                } else {
                    *vars.iter().choose(&mut rng).unwrap()
                };
                Expr::Var(var)
            }
            "not" => Expr::Not(Box::new(self.generate_with_depth(depth + 1, &range, vars))),
            "and" => Expr::BinaryOp(Box::new(BinaryOp::new(
                BinaryOpType::And,
                self.generate_with_depth(depth + 1, &range, vars),
                self.generate_with_depth(depth + 1, &range, vars),
            ))),
            "or" => Expr::BinaryOp(Box::new(BinaryOp::new(
                BinaryOpType::Or,
                self.generate_with_depth(depth + 1, &range, vars),
                self.generate_with_depth(depth + 1, &range, vars),
            ))),
            "impl" => Expr::BinaryOp(Box::new(BinaryOp::new(
                BinaryOpType::Implies,
                self.generate_with_depth(depth + 1, &range, vars),
                self.generate_with_depth(depth + 1, &range, vars),
            ))),
            "equiv" => Expr::BinaryOp(Box::new(BinaryOp::new(
                BinaryOpType::Equivalent,
                self.generate_with_depth(depth + 1, &range, vars),
                self.generate_with_depth(depth + 1, &range, vars),
            ))),
            _ => unreachable!(),
        }
    }

    fn random_variable(&self, range: &RangeInclusive<char>, rng: &mut impl Rng) -> char {
        range.clone().choose(rng).unwrap()
    }
}

pub fn generate_state(expr: &Expr) -> Box<[(char, bool)]> {
    let mut state = Vec::new();
    let mut rng = rand::thread_rng();
    generate_state_recurse(expr, &mut state, &mut rng);

    state.into_boxed_slice()
}

fn generate_state_recurse(expr: &Expr, state: &mut Vec<(char, bool)>, rng: &mut impl Rng) {
    match expr {
        Expr::Var(v) => {
            if state.iter().map(|(c, _)| c).find(|c| *c == v).is_none() {
                state.push((*v, rng.gen_bool(0.5)));
            }
        }
        Expr::Not(e) => generate_state_recurse(e, state, rng),
        Expr::BinaryOp(bop) => {
            let BinaryOp { ty: _, l, r } = &**bop;
            generate_state_recurse(l, state, rng);
            generate_state_recurse(r, state, rng);
        }
    }
}
