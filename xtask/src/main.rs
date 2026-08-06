//! Repo gates. `cargo xtask check` is what to run before committing.
//!
//! ```text
//! cargo xtask check          every gate, cheapest first, stops at the first failure
//! cargo xtask check clippy   one gate by name, for the middle of a fix
//! cargo xtask tuning         rewrite docs/TUNING.md from the source
//! cargo xtask tuning --check exit 1 if it is stale, undocumented or shadowed
//! ```
//!
//! # Why a binary and not a shell script
//!
//! Four of the five gates are a `cargo` invocation and could be a Makefile line.
//! The fifth — the tuning index — has to parse every `.rs` file in the tree, and
//! putting it here means the gate that WRITES `docs/TUNING.md` and the gate that
//! checks it is current are the same code path, so they cannot disagree. Once
//! one gate lives in Rust the rest may as well be driven from the same place,
//! where the ordering, the timings and the failure message are one thing to
//! maintain instead of five.

mod check;
mod tuning;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The repository root, resolved at compile time from this crate's manifest.
///
/// Not `current_dir`: `cargo xtask` is run from wherever the developer happens
/// to be standing, and a gate runner that only works from the root is a gate
/// runner people stop running.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask/ always has a parent")
        .to_path_buf()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();

    match args.first().map(String::as_str) {
        Some("check") => check::run(&root, args.get(1).map(String::as_str)),
        Some("tuning") => {
            let mode = if args.iter().any(|a| a == "--check") {
                tuning::Mode::Check
            } else {
                tuning::Mode::Write
            };
            match tuning::run(&root, mode) {
                Ok(summary) => {
                    println!("{summary}");
                    ExitCode::SUCCESS
                }
                Err(report) => {
                    eprint!("{report}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("help" | "--help" | "-h") | None => {
            print!("{}", usage());
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("xtask: unknown task `{other}`\n");
            eprint!("{}", usage());
            ExitCode::FAILURE
        }
    }
}

fn usage() -> String {
    let mut s = String::from(
        "usage: cargo xtask <task>\n\
         \n\
         tasks:\n\
         \x20 check [gate]    run every gate, or just one of them\n\
         \x20 tuning          rewrite docs/TUNING.md from the source\n\
         \x20 tuning --check  fail if docs/TUNING.md is stale\n\
         \n\
         gates, in the order `check` runs them:\n",
    );
    for gate in check::GATES {
        s.push_str(&format!("  {:<8}  {}\n", gate.name, gate.what));
    }
    s
}
