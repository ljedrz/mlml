mod eval;
mod generator;
mod parser;

use std::collections::{HashMap, HashSet};

use crate::{eval::evaluate, generator::*, parser::*};

fn main() {
    let weights = Weights {
        var: 0.4,
        not: 0.2,
        and: 0.1,
        or: 0.1,
        implies: 0.1,
        equivalent: 0.1,
    };
    let generator = ExprGenerator::new(2, 5, weights);

    let mut seen_exprs: HashSet<Expr> = HashSet::new();

    let mut i = 0;
    while i < 12_000 {
        let expr = generator.generate();
        if seen_exprs.contains(&expr) {
            continue;
        }
        let expr_str = expr.to_string();
        assert!(Parser::new(&expr_str).parse().is_ok());

        let state = generate_state(&expr);
        let ret = evaluate(&expr, &state);
        println!("{} {};{}", stringify_state(&state), expr_str, ret);

        seen_exprs.insert(expr);
        i += 1;
    }
}

fn stringify_state(state: &HashMap<char, bool>) -> String {
    let mut ret = String::from("[");
    let mut pairs = state.iter().map(|(c, b)| (*c, *b)).collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|(c, _)| *c);

    let mut iter = pairs.iter().peekable();
    while let Some((c, b)) = iter.next() {
        ret.push_str(&format!("{c}: {b}"));
        if iter.peek().is_some() {
            ret.push_str(", ");
        }
    }
    ret.push(']');

    ret
}
