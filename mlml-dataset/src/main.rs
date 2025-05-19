mod eval;
mod generator;
mod parser;

use std::collections::HashSet;

use rand::seq::IteratorRandom;

use crate::{eval::evaluate, generator::*, parser::*};

#[derive(Clone, PartialEq, Eq, Hash)]
struct Entry {
    expr: Expr,
    state: Box<[(char, bool)]>,
    ret: bool,
}

fn main() {
    let weights = Weights {
        var: 0.2,
        not: 0.2,
        and: 0.2,
        or: 0.2,
        implies: 0.1,
        equivalent: 0.1,
    };
    let generator = ExprGenerator::new(2, 5, weights);

    let mut seen_entries: HashSet<Entry> = HashSet::new();

    let mut i = 0;
    while i < 20_000 {
        let expr = generator.generate();
        let expr_str = expr.to_string();
        assert!(Parser::new(&expr_str).parse().is_ok());

        let state = generate_state(&expr);
        let ret = evaluate(&expr, &state);
        let entry = Entry { expr, state, ret };

        if seen_entries.contains(&entry) {
            continue;
        }

        seen_entries.insert(entry);
        i += 1;
    }

    let mut rng = rand::thread_rng();

    // train
    let mut train_set = HashSet::new();
    train_set.extend(
        seen_entries
            .iter()
            .filter(|e| e.ret)
            .cloned()
            .choose_multiple(&mut rng, 5_000),
    );
    train_set.extend(
        seen_entries
            .iter()
            .filter(|e| !e.ret)
            .cloned()
            .choose_multiple(&mut rng, 5_000),
    );

    // validate
    let mut valid_set = HashSet::new();
    valid_set.extend(
        seen_entries
            .difference(&train_set)
            .filter(|e| e.ret)
            .cloned()
            .choose_multiple(&mut rng, 500),
    );
    valid_set.extend(
        seen_entries
            .difference(&train_set)
            .filter(|e| !e.ret)
            .cloned()
            .choose_multiple(&mut rng, 500),
    );

    let connection = rusqlite::Connection::open("dataset.db").unwrap();

    for table in &["train", "test"] {
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
            "train" => &train_set,
            "test" => &valid_set,
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
    let mut ret = String::from("[");

    let mut iter = state.iter().peekable();
    while let Some((c, b)) = iter.next() {
        ret.push_str(&format!("{c}: {b}"));
        if iter.peek().is_some() {
            ret.push_str(", ");
        }
    }
    ret.push(']');

    ret
}
