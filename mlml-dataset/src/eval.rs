use crate::parser::{BinaryOpType, Expr};

pub fn evaluate(expr: &Expr, state: &[(char, bool)]) -> bool {
    match expr {
        Expr::Var(c) => state.iter().find(|(c2, _)| c == c2).unwrap().1,
        Expr::Not(e) => !evaluate(e, state),
        Expr::BinaryOp(bop) => match bop.ty {
            BinaryOpType::And => evaluate(&bop.l, state) && evaluate(&bop.r, state),
            BinaryOpType::Or => evaluate(&bop.l, state) || evaluate(&bop.r, state),
            BinaryOpType::Implies => !evaluate(&bop.l, state) || evaluate(&bop.r, state),
            BinaryOpType::Equivalent => evaluate(&bop.l, state) == evaluate(&bop.r, state),
        },
    }
}
