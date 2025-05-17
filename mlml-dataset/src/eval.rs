use std::collections::HashMap;

use crate::parser::Expr;

pub fn evaluate(expr: &Expr, state: &HashMap<char, bool>) -> bool {
    match expr {
        Expr::Var(c) => *state.get(c).unwrap(),
        Expr::Not(e) => !evaluate(e, state),
        Expr::And(l, r) => evaluate(l, state) && evaluate(r, state),
        Expr::Or(l, r) => evaluate(l, state) || evaluate(r, state),
        Expr::Implies(l, r) => !evaluate(l, state) || evaluate(r, state),
        Expr::Equivalent(l, r) => evaluate(l, state) == evaluate(r, state),
    }
}
