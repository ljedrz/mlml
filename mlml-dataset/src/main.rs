mod expr;
mod generator;
mod parser;

use std::{
    collections::{HashMap, HashSet},
    fs,
};

use mlml_util::{MlmlConfig, config_path};
use rand::{SeedableRng, seq::IteratorRandom};
use rand_xorshift::XorShiftRng;

use crate::{expr::Expr, generator::*, parser::*};

#[derive(Clone, PartialEq, Eq, Hash)]
struct Entry {
    expr: Expr,
    state: Box<[(char, bool)]>,
    ret: bool,
}

fn main() {
    let config_str = fs::read_to_string(config_path()).unwrap();
    let config: MlmlConfig = serde_json::from_str(&config_str).unwrap();

    let generator = ExprGenerator::new(config.dataset.max_depth, config.dataset.max_variables);

    let _ = std::fs::remove_file(&config.dataset.db_path);
    let connection = rusqlite::Connection::open(&config.dataset.db_path).unwrap();

    let mut rng = XorShiftRng::from_rng(&mut rand::rng());
    let mut seen_all_entries = HashSet::new();
    let mut seen_all_structures = HashMap::new();

    for ty in ["train", "valid", "test"] {
        let wanted_num_samples = match ty {
            "train" => config.dataset.train_samples_count,
            "valid" => config.dataset.valid_samples_count,
            "test" => config.dataset.test_samples_count,
            _ => unreachable!(),
        };

        let mut wanted_result = true;
        let mut split_samples = HashSet::new();
        while split_samples.len() < wanted_num_samples {
            let range = ('a'..='z').choose_multiple(&mut rng, config.dataset.max_variables);

            let expr = generator.generate(&range, &mut rng);
            let state = generate_state(&expr, &mut rng);
            let ret = expr.evaluate(&state);
            let entry = Entry { expr, state, ret };

            if ret == wanted_result && seen_all_entries.insert(entry.clone()) {
                wanted_result = !ret;
            } else {
                continue;
            }

            *seen_all_structures
                .entry(entry.expr.to_structure())
                .or_default() += 1;
            split_samples.insert(entry);
        }

        let table_creation_query = format!(
            "
            CREATE TABLE {ty} (
                expression TEXT,
                result TEXT,
                complexity INTEGER,
                rarity REAL,
                row_id INTEGER PRIMARY KEY
            )
        "
        );

        connection.execute(&table_creation_query, ()).unwrap();

        let mut data_query =
            format!("INSERT INTO {ty} (expression, result, complexity, rarity) VALUES ");

        let mut iter = split_samples.iter().peekable();
        let mut row = String::new();
        while let Some(entry) = iter.next() {
            let expr_str = entry.expr.to_string();
            assert!(Parser::new(&expr_str).parse().is_ok());

            row.push_str(&format!(
                "('{} {}', '{}', {}, {})",
                stringify_state(&entry.state),
                expr_str,
                entry.ret,
                entry.expr.complexity(),
                entry
                    .expr
                    .rarity(seen_all_entries.len(), &seen_all_structures),
            ));
            if iter.peek().is_some() {
                row.push_str(", ");
            }
            data_query.push_str(&row);
            row.clear();
        }

        connection.execute(&data_query, ()).unwrap();
    }
}

fn stringify_state(state: &[(char, bool)]) -> String {
    let (mut ts, mut fs) = (Vec::new(), Vec::new());
    for (c, b) in state {
        if *b {
            ts.push(c);
        } else {
            fs.push(c);
        }
    }

    let mut ret = String::from("[");

    for (vars, val) in [(&ts, "true"), (&fs, "false")] {
        if !vars.is_empty() {
            let mut iter = vars.iter().peekable();
            while let Some(c) = iter.next() {
                ret.push(**c);
                if iter.peek().is_some() {
                    ret.push_str(", ");
                }
            }
            ret.push_str(&format!(": {val}"));
            if !ts.is_empty() && !fs.is_empty() && val == "true" {
                ret.push_str("; ");
            }
        }
    }

    ret.push(']');

    ret
}
