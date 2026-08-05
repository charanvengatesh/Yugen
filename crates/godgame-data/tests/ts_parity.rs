//! The port's parity gate: every compiled table must equal the one the
//! TypeScript compiler produced from the same `content/`.
//!
//! `tests/ts-snapshot.json` is a frozen extract of `src/generated/*.gen.ts` from
//! the TypeScript repo — 224 id->code mappings, 79 flat tables and 1 pair matrix.
//! It is never regenerated from the Rust side; that would turn the proof into a
//! tautology, which is exactly the trap the TS repo's own `verify.mjs` calls out.
//!
//! This is the strongest evidence available that the Rust content compiler —
//! its lexer, parser, schema resolution, coercion, defaults, ref chains, code
//! assignment and every one of the six schemas — agrees with the original.

use serde_json::Value as J;
use std::collections::BTreeMap;

fn snapshot() -> J {
    let raw = include_str!("ts-snapshot.json");
    serde_json::from_str(raw).expect("ts-snapshot.json is not valid JSON")
}

/// The snapshot stores non-finite floats as the strings "inf" / "-inf",
/// because JSON has no way to spell them.
fn number(v: &J) -> f64 {
    match v {
        J::Number(n) => n.as_f64().expect("not an f64"),
        J::String(s) if s == "inf" => f64::INFINITY,
        J::String(s) if s == "-inf" => f64::NEG_INFINITY,
        other => panic!("snapshot value is neither a number nor an infinity: {other}"),
    }
}

/// A table slot compares equal when the two agree at the width it is stored in.
/// The TS side held every table as a JS `number` (f64) before the typed array
/// narrowed it; the Rust side narrowed at compile time. Comparing at f32 for a
/// float table is therefore comparing what both actually store.
fn slots_agree(expected: f64, got: f64, kind: &str) -> bool {
    if expected.is_nan() && got.is_nan() {
        return true;
    }
    if kind == "f32" {
        return (expected as f32).to_bits() == (got as f32).to_bits();
    }
    expected == got
}

#[test]
fn every_id_kept_its_code() {
    let snap = snapshot();
    let mut got: BTreeMap<(String, String), u16> = BTreeMap::new();
    for (module, id, code) in godgame_data::all_codes() {
        got.insert((module.to_string(), id.to_string()), code);
    }

    let mut checked = 0usize;
    let mut problems: Vec<String> = Vec::new();
    for (module, body) in snap.as_object().unwrap() {
        for (id, code) in body["ids"].as_object().unwrap() {
            let want = code.as_u64().unwrap() as u16;
            match got.get(&(module.clone(), id.clone())) {
                Some(&have) if have == want => checked += 1,
                Some(&have) => {
                    problems.push(format!("{module}: '{id}' is code {have}, TS had {want}"))
                }
                None => problems.push(format!("{module}: '{id}' is missing entirely")),
            }
        }
    }

    assert!(
        problems.is_empty(),
        "{} code mismatches:\n{}",
        problems.len(),
        problems.join("\n")
    );
    assert_eq!(
        checked, 224,
        "the TS build had 224 records across six kinds"
    );
}

#[test]
fn every_flat_table_matches_the_typescript_build() {
    let snap = snapshot();
    let tables = godgame_data::all_tables();

    let mut problems: Vec<String> = Vec::new();
    let mut compared_tables = 0usize;
    let mut compared_slots = 0usize;

    for (module, body) in snap.as_object().unwrap() {
        let want_tables = body["tables"].as_object().unwrap();
        for (name, spec) in want_tables {
            let Some((_, _, table)) = tables.iter().find(|(m, n, _)| m == module && n == name)
            else {
                problems.push(format!("{module}::{name} is missing from the Rust build"));
                continue;
            };
            let want: Vec<f64> = spec["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            if want.len() != table.len() {
                problems.push(format!(
                    "{module}::{name} has {} slots, TS had {}",
                    table.len(),
                    want.len()
                ));
                continue;
            }
            compared_tables += 1;
            for (i, &w) in want.iter().enumerate() {
                let g = table.get(i);
                compared_slots += 1;
                if !slots_agree(w, g, table.kind()) {
                    problems.push(format!("{module}::{name}[{i}] = {g}, TS had {w}"));
                }
            }
        }
    }

    // Truncate the report: one wrong default can move a whole table, and 500
    // identical lines bury the other failures.
    if !problems.is_empty() {
        let shown: Vec<&String> = problems.iter().take(25).collect();
        panic!(
            "{} table mismatches (showing {}):\n{}",
            problems.len(),
            shown.len(),
            shown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    assert_eq!(compared_tables, 79, "the TS build had 79 flat tables");
    // 36*53 blocks + 3*79 items + 6*20 mobs + 18*14 structs + 16*7 features.
    // Pinned exactly so a table that silently stops being emitted is caught
    // even if nothing it holds is wrong.
    assert_eq!(compared_slots, 2629, "the TS build had 2629 table slots");
}

#[test]
fn every_pair_matrix_matches_the_typescript_build() {
    let snap = snapshot();
    let mats = godgame_data::all_matrices();
    let mut compared = 0usize;

    for (module, body) in snap.as_object().unwrap() {
        for (name, spec) in body["matrices"].as_object().unwrap() {
            let Some((_, _, side, bits)) =
                mats.iter().find(|(m, n, _, _)| m == module && n == name)
            else {
                panic!("{module}::{name} is missing from the Rust build");
            };
            let n = spec["n"].as_u64().unwrap() as usize;
            assert_eq!(side * side, n, "{module}::{name} side length disagrees");

            let want: Vec<usize> = spec["set"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let got: Vec<usize> = (0..n)
                .filter(|&i| (bits[i >> 3] >> (i & 7)) & 1 != 0)
                .collect();
            assert_eq!(got, want, "{module}::{name} has different bits set");
            compared += 1;
        }
    }
    assert_eq!(compared, 1, "the TS build had one pair matrix (GROW_ONTO)");
}

#[test]
fn record_counts_match() {
    let snap = snapshot();
    let codes = godgame_data::all_codes();
    for (module, body) in snap.as_object().unwrap() {
        let want = body["count"].as_u64().unwrap() as usize;
        let got = codes.iter().filter(|(m, _, _)| m == module).count();
        assert_eq!(got, want, "{module} has {got} records, TS had {want}");
    }
}
