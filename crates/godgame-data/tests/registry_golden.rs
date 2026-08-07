//! The registry's golden baseline: every compiled table must still hold what it
//! held the day it was locked down.
//!
//! # What this used to be, and what changed
//!
//! This was `ts_parity.rs`, and its fixture was a frozen extract of
//! `src/generated/*.gen.ts` from the TypeScript original. Its header said the
//! fixture is *"never regenerated from the Rust side; that would turn the proof
//! into a tautology"*, and while the port was in progress that was exactly right:
//! it was the strongest evidence in the repo that the content compiler — lexer,
//! parser, schema resolution, coercion, defaults, ref chains, code assignment and
//! all six schemas — agreed with the original.
//!
//! **The port is over and the original is not coming back.** What that rule cost,
//! once the port was done, was the ability to add anything at all: it asserted
//! the registry holds EXACTLY 53 blocks, 79 items, 20 mobs, 51 sprites, 14
//! structures and 7 features, so a single new block failed three suites. A game
//! that cannot grow a workbench is finished in the wrong sense.
//!
//! So the fixture is the same bytes, renamed, and it is no longer an authority on
//! TypeScript. It is this project's own baseline, and the question it answers has
//! changed from *"does this match the original"* to **"has any of this moved
//! since we locked it"**. Be clear-eyed about the trade: nothing in the tree can
//! prove the port is faithful any more. It can only prove it has not drifted.
//!
//! # The one rule that makes growth safe
//!
//! **The baseline is a PREFIX, and a prefix cannot move.** Every baselined record
//! keeps its exact code; no new record may take a code below the boundary; every
//! baselined table slot and matrix bit still reads the same. New records are
//! appended above the boundary and are not compared against anything — until
//! somebody blesses them, below.
//!
//! That is strictly stronger per record than the count it replaces. A count says
//! "there are 53 blocks". This says "`stone` is code 41 and `MAT_DENSITY[41]` is
//! 1.6", for all 224 of them.
//!
//! # Blessing
//!
//! ```text
//! GODGAME_BLESS=1 cargo test -p godgame-data --test registry_golden
//! ```
//!
//! Rewrites `registry.golden.json` from the live build and leaves the diff in the
//! working tree. **This is the only supported way to change a baseline, and it is
//! never how a red test gets fixed** — it is how new content, once it is right,
//! gets locked down so the next change cannot move it. Read the diff. A bless
//! that touches a line you did not expect to touch is a bug you were about to
//! commit.
//!
//! # The file was blessed once, on the day it was renamed
//!
//! Otherwise the FIRST real bless would have been a whole-file reformat with the
//! change buried in it, and "read the diff" would have been advice nobody could
//! take. The reformat was checked rather than trusted: 2 508 of the 2 629 slots
//! came out byte-identical, 121 differ only by an f32 round trip, and every id,
//! count and matrix bit is unchanged. Those 121 are float tables whose values the
//! dump held at f64 and this build narrows at compile time — [`slots_agree`]
//! already compared them at f32, so the claim they carry is exactly what it was.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value as J;

/// Where the baseline lives, for the bless path. Reading goes through
/// `include_str!` so an ordinary run needs no filesystem at all.
fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("registry.golden.json")
}

fn baseline() -> J {
    let raw = include_str!("registry.golden.json");
    serde_json::from_str(raw).expect("registry.golden.json is not valid JSON")
}

/// Whether this run should rewrite the baseline instead of comparing against it.
fn blessing() -> bool {
    std::env::var_os("GODGAME_BLESS").is_some_and(|v| v != "0" && !v.is_empty())
}

/// The baseline stores non-finite floats as the strings "inf" / "-inf", because
/// JSON has no way to spell them. [`write_number`] is the other half.
fn number(v: &J) -> f64 {
    match v {
        J::Number(n) => n.as_f64().expect("not an f64"),
        J::String(s) if s == "inf" => f64::INFINITY,
        J::String(s) if s == "-inf" => f64::NEG_INFINITY,
        other => panic!("baseline value is neither a number nor an infinity: {other}"),
    }
}

/// A table slot compares equal when the two agree at the width it is STORED in.
///
/// The values were dumped as f64 and the Rust side narrows at compile time, so
/// comparing a float table at f32 is comparing what is actually kept. Inherited
/// from the TypeScript comparison and still correct for the same reason.
fn slots_agree(expected: f64, got: f64, kind: &str) -> bool {
    if expected.is_nan() && got.is_nan() {
        return true;
    }
    if kind == "f32" {
        return (expected as f32).to_bits() == (got as f32).to_bits();
    }
    expected == got
}

/// How many records the baseline pins for a module — the prefix boundary.
fn baseline_count(body: &J) -> usize {
    body["count"].as_u64().expect("count is not a number") as usize
}

// --- The four claims ---------------------------------------------------------

#[test]
fn every_baselined_id_kept_its_code() {
    if blessing() {
        return bless();
    }
    let base = baseline();
    let mut got: BTreeMap<(String, String), u16> = BTreeMap::new();
    for (module, id, code) in godgame_data::all_codes() {
        got.insert((module.to_string(), id.to_string()), code);
    }

    let mut checked = 0usize;
    let mut expected = 0usize;
    let mut problems: Vec<String> = Vec::new();
    for (module, body) in base.as_object().unwrap() {
        expected += baseline_count(body);
        for (id, code) in body["ids"].as_object().unwrap() {
            let want = code.as_u64().unwrap() as u16;
            match got.get(&(module.clone(), id.clone())) {
                Some(&have) if have == want => checked += 1,
                Some(&have) => problems.push(format!(
                    "{module}: '{id}' is code {have}, baseline has {want}"
                )),
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
    // Derived from the baseline rather than a literal, so blessing new content
    // moves it without an edit here — and a TRUNCATED baseline file is still
    // caught, because `count` and the `ids` map would have to be truncated in
    // agreement for this to pass.
    assert_eq!(
        checked, expected,
        "the baseline's `ids` maps and its `count` fields disagree — the file is \
         damaged, not the build"
    );
}

#[test]
fn the_baseline_records_are_an_unmoved_prefix() {
    if blessing() {
        return;
    }
    let base = baseline();
    let codes = godgame_data::all_codes();

    let mut problems: Vec<String> = Vec::new();
    for (module, body) in base.as_object().unwrap() {
        let n = baseline_count(body);

        // What the baseline says sits at each code below the boundary.
        let mut want: BTreeMap<u16, &str> = BTreeMap::new();
        for (id, code) in body["ids"].as_object().unwrap() {
            want.insert(code.as_u64().unwrap() as u16, id.as_str());
        }

        for (m, id, code) in codes.iter().filter(|(m, _, _)| *m == module) {
            if (*code as usize) >= n {
                // Appended content. Not compared — that is the whole point of
                // the boundary. It becomes compared the moment somebody blesses.
                continue;
            }
            match want.get(code) {
                Some(&baselined) if baselined == *id => {}
                Some(&baselined) => problems.push(format!(
                    "{m}: code {code} is now '{id}', baseline has '{baselined}' — a \
                     baselined record has been renumbered or renamed"
                )),
                None => problems.push(format!(
                    "{m}: '{id}' took code {code}, which is inside the baselined \
                     range 0..{n}. New records must be APPENDED; check \
                     content/ids.lock.json"
                )),
            }
        }
    }

    assert!(
        problems.is_empty(),
        "{} record(s) moved inside the frozen prefix:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

#[test]
fn every_baselined_table_slot_still_reads_the_same() {
    if blessing() {
        return;
    }
    let base = baseline();
    let tables = godgame_data::all_tables();

    let mut problems: Vec<String> = Vec::new();
    let mut compared_tables = 0usize;
    let mut compared_slots = 0usize;

    for (module, body) in base.as_object().unwrap() {
        for (name, spec) in body["tables"].as_object().unwrap() {
            let Some((_, _, table)) = tables.iter().find(|(m, n, _)| m == module && n == name)
            else {
                problems.push(format!("{module}::{name} is missing from the build"));
                continue;
            };
            let want: Vec<f64> = spec["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            // A table may only GROW. It grows by exactly as many slots as the
            // module grew records, because every one of these is one slot per
            // record; a table that shrank has lost content that was pinned.
            if table.len() < want.len() {
                problems.push(format!(
                    "{module}::{name} has {} slots, the baseline pins {} — a pinned \
                     slot cannot be removed",
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
                    problems.push(format!("{module}::{name}[{i}] = {g}, baseline has {w}"));
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
    // Both derived from the baseline rather than written as literals — the old
    // suite pinned 79 and 2629 by hand, and the reason it gave still applies: a
    // table that silently stops being emitted has to be caught even when nothing
    // it holds is wrong. Derived, the same claim survives a bless.
    let (want_tables, want_slots) = base.as_object().unwrap().values().fold((0, 0), |acc, b| {
        let t = b["tables"].as_object().unwrap();
        (
            acc.0 + t.len(),
            acc.1
                + t.values()
                    .map(|s| s["values"].as_array().unwrap().len())
                    .sum::<usize>(),
        )
    });
    assert_eq!(
        compared_tables, want_tables,
        "a table named by the baseline was skipped without being reported"
    );
    assert_eq!(
        compared_slots, want_slots,
        "the baseline's tables are not all the length it says they are"
    );
}

#[test]
fn every_baselined_matrix_bit_still_reads_the_same() {
    if blessing() {
        return;
    }
    let base = baseline();
    let mats = godgame_data::all_matrices();
    let mut compared = 0usize;

    for (module, body) in base.as_object().unwrap() {
        for (name, spec) in body["matrices"].as_object().unwrap() {
            let Some((_, _, side, bits)) =
                mats.iter().find(|(m, n, _, _)| m == module && n == name)
            else {
                panic!("{module}::{name} is missing from the build");
            };
            let want_n = spec["n"].as_u64().unwrap() as usize;
            let want_side = isqrt(want_n);
            assert!(
                *side >= want_side,
                "{module}::{name} is now {side}x{side}, the baseline pins \
                 {want_side}x{want_side} — a pair matrix cannot shrink"
            );

            let want: std::collections::BTreeSet<usize> = spec["set"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();

            // A matrix is a SQUARE, so its frozen part is not a prefix of the
            // flat array — row r of the baseline starts at `r * want_side` and
            // row r of the live one at `r * side`. Comparing the flat prefix
            // would silently compare row 2 of one against row 1 of the other the
            // moment a block is added, which is the trap this loop exists to
            // avoid.
            for r in 0..want_side {
                for c in 0..want_side {
                    let live = r * side + c;
                    let set = (bits[live >> 3] >> (live & 7)) & 1 != 0;
                    assert_eq!(
                        set,
                        want.contains(&(r * want_side + c)),
                        "{module}::{name}[{r}][{c}] disagrees with the baseline"
                    );
                }
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 1, "the baseline pins one pair matrix (GROW_ONTO)");
}

/// Integer square root, for turning a flattened matrix length back into a side.
fn isqrt(n: usize) -> usize {
    let mut s = (n as f64).sqrt() as usize;
    while s * s > n {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= n {
        s += 1;
    }
    assert_eq!(s * s, n, "a pair matrix length {n} is not a square");
    s
}

// --- Blessing ----------------------------------------------------------------

/// Rewrite the baseline from the live build.
///
/// Deliberately NOT a `#[test]` of its own: a test that rewrites a fixture is one
/// `cargo test` away from a tautology, so it can only run when
/// `GODGAME_BLESS` says so, and every comparison above returns early in that
/// mode rather than comparing against a file it is about to overwrite.
fn bless() {
    let mut root = serde_json::Map::new();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut ids: BTreeMap<&str, serde_json::Map<String, J>> = BTreeMap::new();
    for (module, id, code) in godgame_data::all_codes() {
        *counts.entry(module).or_default() += 1;
        ids.entry(module)
            .or_default()
            .insert(id.to_string(), J::from(code));
    }

    let mut tables: BTreeMap<&str, serde_json::Map<String, J>> = BTreeMap::new();
    for (module, name, table) in godgame_data::all_tables() {
        let values: Vec<J> = (0..table.len())
            .map(|i| write_number(table.get(i)))
            .collect();
        let mut spec = serde_json::Map::new();
        spec.insert("values".into(), J::Array(values));
        tables
            .entry(module)
            .or_default()
            .insert(name.into(), J::Object(spec));
    }

    let mut matrices: BTreeMap<&str, serde_json::Map<String, J>> = BTreeMap::new();
    for (module, name, side, bits) in godgame_data::all_matrices() {
        let n = side * side;
        let set: Vec<J> = (0..n)
            .filter(|&i| (bits[i >> 3] >> (i & 7)) & 1 != 0)
            .map(J::from)
            .collect();
        let mut spec = serde_json::Map::new();
        spec.insert("n".into(), J::from(n));
        spec.insert("set".into(), J::Array(set));
        matrices
            .entry(module)
            .or_default()
            .insert(name.into(), J::Object(spec));
    }

    for (module, count) in counts {
        let mut body = serde_json::Map::new();
        body.insert("count".into(), J::from(count));
        body.insert(
            "ids".into(),
            J::Object(ids.remove(module).unwrap_or_default()),
        );
        body.insert(
            "tables".into(),
            J::Object(tables.remove(module).unwrap_or_default()),
        );
        body.insert(
            "matrices".into(),
            J::Object(matrices.remove(module).unwrap_or_default()),
        );
        root.insert(module.to_string(), J::Object(body));
    }

    let path = baseline_path();
    let text = serde_json::to_string_pretty(&J::Object(root)).expect("the baseline serialises");
    std::fs::write(&path, text + "\n").expect("the baseline is writable");
    println!(
        "BLESSED {} — read the diff before committing it",
        path.display()
    );
}

/// [`number`]'s inverse: JSON has no infinities, so they go out as strings.
fn write_number(v: f64) -> J {
    if v.is_infinite() {
        return J::from(if v > 0.0 { "inf" } else { "-inf" });
    }
    serde_json::Number::from_f64(v).map_or_else(|| J::from("nan"), J::Number)
}
