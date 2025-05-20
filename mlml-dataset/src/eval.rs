use crate::parser::{BinaryOpType, Expr};

pub fn evaluate(expr: &Expr, state: &[(char, bool)]) -> bool {
    match expr {
        Expr::Var(c) => state.iter().find(|(c2, _)| c == c2).unwrap().1,
        Expr::Not(e) => !evaluate(e, state),
        Expr::BinaryOp(bop) => {
            let l = evaluate(&bop.l, state);
            let r = evaluate(&bop.r, state);

            match bop.ty {
                BinaryOpType::And => l && r,
                BinaryOpType::Or => l || r,
                BinaryOpType::Implies => !l || r,
                BinaryOpType::Equivalent => l == r,
            }
        }
    }
}
