use std::iter::Peekable;

use crate::expr::{BinaryOp, BinaryOpType, Expr};

pub struct Parser<'a> {
    chars: Peekable<std::str::Chars<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Parser {
            chars: input.chars().peekable(),
        }
    }

    pub fn parse(&mut self) -> Result<Expr, String> {
        self.parse_expr()
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.skip_whitespace();
        match self.peek() {
            Some('(') => self.parse_binary(),
            Some('¬') => self.parse_not(),
            Some(c) if c.is_alphabetic() => self.parse_var(),
            _ => Err("Expected variable, '(', or '¬'".to_string()),
        }
    }

    fn parse_binary(&mut self) -> Result<Expr, String> {
        self.expect('(')?;
        let left = self.parse_expr()?;
        self.skip_whitespace();
        let op = self.next_char()?.ok_or("Expected operator".to_string())?;
        let right = self.parse_expr()?;
        self.skip_whitespace();
        self.expect(')')?;

        let op = match op {
            '∧' => BinaryOpType::And,
            '∨' => BinaryOpType::Or,
            '→' => BinaryOpType::Implies,
            '↔' => BinaryOpType::Equivalent,
            _ => {
                return Err(format!("Unknown operator: {}", op));
            }
        };

        Ok(Expr::BinaryOp(Box::new(BinaryOp::new(op, left, right))))
    }

    fn parse_not(&mut self) -> Result<Expr, String> {
        self.expect('¬')?;
        let expr = self.parse_expr()?;
        Ok(Expr::Not(Box::new(expr)))
    }

    fn parse_var(&mut self) -> Result<Expr, String> {
        if let Some(c) = self.next_char()? {
            if !c.is_alphabetic() {
                return Err("Variable must start with a letter".to_string());
            }
            Ok(Expr::Var(c))
        } else {
            unreachable!();
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if !c.is_whitespace() {
                break;
            }
            let _ = self.next_char();
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn next_char(&mut self) -> Result<Option<char>, String> {
        Ok(self.chars.next())
    }

    fn expect(&mut self, expected: char) -> Result<(), String> {
        match self.next_char()? {
            Some(c) if c == expected => Ok(()),
            Some(c) => Err(format!("Expected '{}', got '{}'", expected, c)),
            None => Err(format!("Expected '{}', got EOF", expected)),
        }
    }
}
