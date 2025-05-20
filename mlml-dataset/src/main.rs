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

    let mut seen_all = HashSet::new();
    let mut seen_train = HashSet::new();
    let mut seen_valid = HashSet::new();
    let mut seen_test = HashSet::new();
    let mut set_train = HashSet::new();
    let mut set_valid = HashSet::new();
    let mut set_test = HashSet::new();

    let mut rng = XorShiftRng::from_rng(&mut rand::rng());

    for ty in ["train", "valid", "test"] {
        let (seen, set, samples) = match ty {
            "train" => (
                &mut seen_train,
                &mut set_train,
                config.dataset.train_samples_count,
            ),
            "valid" => (
                &mut seen_valid,
                &mut set_valid,
                config.dataset.valid_samples_count,
            ),
            "test" => (
                &mut seen_test,
                &mut set_test,
                config.dataset.test_samples_count,
            ),
            _ => unreachable!(),
        };
        let range = ('a'..'z').choose_multiple(&mut rng, config.dataset.max_variables);

        let mut i = 0;
        while i < config.dataset.train_samples_count {
            let expr = generator.generate(&range, &mut rng);
            let expr_str = expr.to_string();
            assert!(Parser::new(&expr_str).parse().is_ok());

            let state = generate_state(&expr, &mut rng);
            let ret = evaluate(&expr, &state);
            let entry = Entry { expr, state, ret };

            if seen.contains(&entry) || seen_all.contains(&entry) {
                continue;
            }

            seen.insert(entry.clone());
            seen_all.insert(entry);
            i += 1;
        }

        set.extend(
            seen.iter()
                .filter(|e| e.ret)
                .cloned()
                .choose_multiple(&mut rng, samples / 2),
        );
        set.extend(
            seen.iter()
                .filter(|e| !e.ret)
                .cloned()
                .choose_multiple(&mut rng, samples / 2),
        );
    }

    let _ = std::fs::remove_file(&config.dataset.db_path);
    let connection = rusqlite::Connection::open(&config.dataset.db_path).unwrap();

    for table in &["train", "valid", "test"] {
        let table_creation_query = format!(
            "
            CREATE TABLE {table} (
                expression TEXT,
                result TEXT,
                row_id INTEGER PRIMARY KEY
            )
        "
        );

        connection.execute(&table_creation_query, ()).unwrap();

        let mut data_query = format!("INSERT INTO {table} (expression, result) VALUES ");

        let dataset = match &**table {
            "train" => &set_train,
            "valid" => &set_valid,
            "test" => &set_test,
            _ => unreachable!(),
        };
        let mut iter = dataset.iter().peekable();
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
