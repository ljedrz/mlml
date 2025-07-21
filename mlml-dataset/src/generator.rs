use std::collections::HashSet;

use rand::Rng;
use rand::seq::{IndexedRandom, IteratorRandom};

use crate::expr::{BinaryOp, BinaryOpType, Expr};

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

    pub fn generate<R: Rng>(&self, range: &[char], rng: &mut R) -> Expr {
        let mut vars = HashSet::new();
        self.generate_with_depth(0, range, &mut vars, rng)
    }

    fn generate_with_depth<R: Rng>(
        &self,
        depth: usize,
        range: &[char],
        vars: &mut HashSet<char>,
        rng: &mut R,
    ) -> Expr {
        if depth >= self.max_depth {
            let var = if vars.len() < self.max_vars {
                let v = self.random_variable(range, rng);
                vars.insert(v);
                v
            } else {
                *vars.iter().choose(rng).unwrap()
            };

            return Expr::Var(var);
        }

        let choices = [
            ("var", 5),
            ("not", 1),
            ("and", 1),
            ("or", 1),
            ("impl", 3),
            ("equiv", 1),
        ];
        let choice = choices
            .choose_weighted(rng, |(_, w)| *w)
            .map(|(c, _)| c)
            .unwrap();

        match &**choice {
            "var" => {
                let var = if vars.len() < self.max_vars {
                    let v = self.random_variable(range, rng);
                    vars.insert(v);
                    v
                } else {
                    *vars.iter().choose(rng).unwrap()
                };
                Expr::Var(var)
            }
            "not" => Expr::Not(Box::new(self.generate_with_depth(
                depth + 1,
                range,
                vars,
                rng,
            ))),
            bop => {
                let op = match bop {
                    "and" => BinaryOpType::And,
                    "or" => BinaryOpType::Or,
                    "impl" => BinaryOpType::Implies,
                    "equiv" => BinaryOpType::Equivalent,
                    _ => unreachable!(),
                };

                Expr::BinaryOp(Box::new(BinaryOp::new(
                    op,
                    self.generate_with_depth(depth + 1, range, vars, rng),
                    self.generate_with_depth(depth + 1, range, vars, rng),
                )))
            }
        }
    }

    fn random_variable<R: Rng>(&self, range: &[char], rng: &mut R) -> char {
        range.iter().copied().choose(rng).unwrap()
    }
}

pub fn generate_state<R: Rng>(expr: &Expr, rng: &mut R) -> Box<[(char, bool)]> {
    let mut state = Vec::new();
    generate_state_recurse(expr, &mut state, rng);

    state.into_boxed_slice()
}

fn generate_state_recurse<R: Rng>(expr: &Expr, state: &mut Vec<(char, bool)>, rng: &mut R) {
    match expr {
        Expr::Var(v) => {
            if !state.iter().map(|(c, _)| c).any(|c| c == v) {
                state.push((*v, rng.random_bool(0.5)));
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
