use std::{collections::HashMap, fmt};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Var(char),
    Not(Box<Expr>),
    BinaryOp(Box<BinaryOp>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOpType {
    And,
    Or,
    Implies,
    Equivalent,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryOp {
    pub ty: BinaryOpType,
    pub l: Expr,
    pub r: Expr,
}

impl BinaryOp {
    pub fn new(ty: BinaryOpType, l: Expr, r: Expr) -> Self {
        Self { ty, l, r }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write!(f, "({}, {})", self.x, self.y)
        write!(f, "{}", self._to_string(false))
    }
}

impl Expr {
    fn _to_string(&self, is_deep: bool) -> String {
        match self {
            Expr::Var(s) => s.to_string(),
            Expr::Not(e) => format!("¬{}", e._to_string(true)),
            Expr::BinaryOp(bop) => {
                let (l, r) = (bop.l._to_string(true), bop.r._to_string(true));
                let inner = match bop.ty {
                    BinaryOpType::And => format!("{l} ∧ {r}"),
                    BinaryOpType::Or => format!("{l} ∨ {r}"),
                    BinaryOpType::Implies => format!("{l} → {r}"),
                    BinaryOpType::Equivalent => {
                        format!("{l} ↔ {r}")
                    }
                };
                if is_deep { format!("({inner})") } else { inner }
            }
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Expr::Var(_) => 0,
            Expr::Not(e) => 1 + e.depth(),
            Expr::BinaryOp(bop) => 1 + bop.l.depth() + bop.r.depth(),
        }
    }

    pub fn num_variables(&self) -> usize {
        match self {
            Expr::Var(_) => 1,
            Expr::Not(e) => e.num_variables(),
            Expr::BinaryOp(bop) => bop.l.num_variables() + bop.r.num_variables(),
        }
    }

    pub fn complexity(&self) -> usize {
        self.num_variables() + self.depth() * 2
    }

    pub fn to_structure(&self) -> ExprStructure {
        match self {
            Expr::Var(_) => ExprStructure::Var,
            Expr::Not(e) => ExprStructure::Not(Box::new(e.to_structure())),
            Expr::BinaryOp(bop) => ExprStructure::BinaryOp(Box::new(BinaryOpStructure {
                ty: bop.ty,
                l: bop.l.to_structure(),
                r: bop.r.to_structure(),
            })),
        }
    }

    pub fn rarity(
        &self,
        num_all_samples: usize,
        structures: &HashMap<ExprStructure, usize>,
    ) -> f32 {
        *structures.get(&self.to_structure()).unwrap() as f32 / num_all_samples as f32
    }

    pub fn evaluate(&self, state: &[(char, bool)]) -> bool {
        match self {
            Expr::Var(c) => state.iter().find(|(c2, _)| c == c2).unwrap().1,
            Expr::Not(e) => !e.evaluate(state),
            Expr::BinaryOp(bop) => {
                let l = bop.l.evaluate(state);
                let r = bop.r.evaluate(state);

                match bop.ty {
                    BinaryOpType::And => l && r,
                    BinaryOpType::Or => l || r,
                    BinaryOpType::Implies => !l || r,
                    BinaryOpType::Equivalent => l == r,
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprStructure {
    Var,
    Not(Box<ExprStructure>),
    BinaryOp(Box<BinaryOpStructure>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryOpStructure {
    pub ty: BinaryOpType,
    pub l: ExprStructure,
    pub r: ExprStructure,
}
