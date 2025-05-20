mod eval;
mod generator;
mod parser;

use std::{collections::HashSet, fs};

use mlml_util::{MlmlConfig, config_path};
use rand::{SeedableRng, seq::IteratorRandom};
use rand_xorshift::XorShiftRng;

use crate::{eval::evaluate, generator::*, parser::*};

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

    let mut seen_all = HashSet::new();

    let mut rng = XorShiftRng::from_rng(&mut rand::rng());

    for ty in ["train", "valid", "test"] {
        let num_samples = match ty {
            "train" => config.dataset.train_samples_count,
            "valid" => config.dataset.valid_samples_count,
            "test" => config.dataset.test_samples_count,
            _ => unreachable!(),
        };

        let mut samples = HashSet::new();

        let mut i = 0;
        while i < num_samples * 3 / 2 {
            let range = ('a'..='z').choose_multiple(&mut rng, config.dataset.max_variables);

            let expr = generator.generate(&range, &mut rng);
            let expr_str = expr.to_string();
            assert!(Parser::new(&expr_str).parse().is_ok());

            let state = generate_state(&expr, &mut rng);
            let ret = evaluate(&expr, &state);
            let entry = Entry { expr, state, ret };

            if !seen_all.insert(entry.clone()) {
                continue;
            }

            samples.insert(entry);
            i += 1;
        }

        let mut samples_normalized = HashSet::new();
        samples_normalized.extend(
            samples
                .iter()
                .filter(|e| e.ret)
                .cloned()
                .choose_multiple(&mut rng, num_samples / 2),
        );
        samples_normalized.extend(
            samples
                .iter()
                .filter(|e| !e.ret)
                .cloned()
                .choose_multiple(&mut rng, num_samples / 2),
        );

        let table_creation_query = format!(
            "
            CREATE TABLE {ty} (
                expression TEXT,
                result TEXT,
                row_id INTEGER PRIMARY KEY
            )
        "
        );

        connection.execute(&table_creation_query, ()).unwrap();

        let mut data_query = format!("INSERT INTO {ty} (expression, result) VALUES ");

        let mut iter = samples_normalized.iter().peekable();
        let mut row = String::new();
        while let Some(entry) = iter.next() {
            row.push_str(&format!(
                "('{} {}', '{}')",
                stringify_state(&entry.state),
                entry.expr.to_string(),
                entry.ret
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
