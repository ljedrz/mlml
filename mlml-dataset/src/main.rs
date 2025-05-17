mod generator;
mod parser;

use crate::{generator::*, parser::*};

fn main() {
    // Configure generator
    let weights = Weights {
        var: 0.4,
        not: 0.2,
        and: 0.1,
        or: 0.1,
        implies: 0.1,
        equivalent: 0.1,
    };
    let generator = ExprGenerator::new(2, 4, weights);

    // Generate and test 5 expressions
    for i in 0..5 {
        let expr = generator.generate();
        let expr_str = expr.to_string();
        println!("Generated {}: {}", i + 1, expr_str);

        // Parse to verify
        let mut parser = Parser::new(&expr_str);
        match parser.parse() {
            Ok(parsed) => println!("Parsed successfully: {:?}", parsed),
            Err(e) => println!("Parse error: {}", e),
        }
    }
}
