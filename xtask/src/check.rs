//! The aggregate gate.
//!
//! `crates/`-wide correctness in this tree is five separate commands, and the
//! failure mode this module exists to prevent is a developer running four of
//! them. `cargo xtask check` runs all five, in one order, and exits non-zero the
//! moment one fails.
//!
//! # The order is cheapest-first, and that is the whole design
//!
//! Every gate here can fail on its own, so the only question is which one you
//! want to hear about first. `fmt` needs no build at all and answers in under a
//! second; `test` compiles every target in the workspace and then runs the
//! parity suites. Putting `fmt` last would mean waiting three minutes to be told
//! about a missing blank line. The list runs in ascending cost, which is also
//! roughly descending frequency of failure, so the common case is the fast case.
//!
//! # Why it stops at the first failure
//!
//! Because the gates are not independent in practice. A stale `content/` compile
//! makes `clippy` and `test` fail with errors about generated code that are
//! noise, not signal — the real report is four lines from `contentc`. Running on
//! would bury it. The summary names every gate that did not run, so a stop is
//! never mistaken for a pass: this is the failure the task of writing a gate
//! runner is most likely to introduce, and the summary is the guard against it.

use std::path::Path;
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use crate::tuning;

/// One gate: a name to invoke it by, a line saying what it proves, and the work.
pub struct Gate {
    /// What `cargo xtask check <name>` matches on.
    pub name: &'static str,
    /// One line for the usage text and the per-gate banner.
    pub what: &'static str,
    body: Body,
}

enum Body {
    /// Shell out to cargo. The child inherits stdout and stderr, so rustc's own
    /// diagnostics — spans, suggestions, colour — reach the terminal untouched.
    /// A gate runner that captures compiler output and reprints "clippy failed"
    /// has thrown away the only part a developer needed.
    Cargo(&'static [&'static str]),
    /// The tuning index, run in this process. See [`crate::tuning`].
    Tuning,
}

/// The gate list from `CLAUDE.md`, plus the tuning index, cheapest first.
///
/// `clippy` carries `-D warnings` because "must be zero findings" is only a
/// standard if something enforces it; without the flag clippy prints its
/// findings and exits 0, which is a report, not a gate.
pub const GATES: &[Gate] = &[
    Gate {
        name: "fmt",
        what: "every file is rustfmt-clean",
        body: Body::Cargo(&["fmt", "--all", "--check"]),
    },
    Gate {
        name: "tuning",
        what: "docs/TUNING.md is current, config is documented, no name is shadowed",
        body: Body::Tuning,
    },
    Gate {
        name: "content",
        what: "crates/godgame-data/ matches content/",
        body: Body::Cargo(&["run", "--quiet", "-p", "contentc", "--", "--check"]),
    },
    Gate {
        name: "clippy",
        what: "zero clippy findings across the workspace",
        body: Body::Cargo(&[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]),
    },
    Gate {
        name: "test",
        what: "the whole suite, including the three TypeScript parity suites",
        body: Body::Cargo(&["test", "--workspace"]),
    },
];

impl Gate {
    /// The exact command line that reproduces this gate on its own.
    ///
    /// Printed on failure and nowhere else. The point is that the developer can
    /// copy it rather than reconstruct it — a failure message naming a gate but
    /// not the command behind it makes them read this file to find out.
    fn reproduce(&self) -> String {
        match self.body {
            Body::Cargo(args) => format!("cargo {}", args.join(" ")),
            Body::Tuning => "cargo xtask tuning --check".to_string(),
        }
    }

    fn execute(&self, root: &Path) -> Result<(), String> {
        match self.body {
            Body::Cargo(args) => {
                // `CARGO` is set by cargo itself, so a gate run under a pinned
                // toolchain stays on that toolchain instead of falling back to
                // whatever `cargo` resolves to on PATH.
                let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
                let status = Command::new(&cargo)
                    .args(args)
                    .current_dir(root)
                    .status()
                    .map_err(|e| format!("could not launch `{cargo}`: {e}"))?;
                if status.success() {
                    Ok(())
                } else {
                    Err(match status.code() {
                        Some(code) => format!("exit code {code}"),
                        None => "killed by a signal".to_string(),
                    })
                }
            }
            Body::Tuning => match tuning::run(root, tuning::Mode::Check) {
                Ok(summary) => {
                    println!("{summary}");
                    Ok(())
                }
                Err(report) => {
                    eprint!("{report}");
                    Err("the tuning index rejected the tree".to_string())
                }
            },
        }
    }
}

/// Outcome of one gate, for the summary table.
enum Outcome {
    Passed(Duration),
    Failed(Duration, String),
    NotRun,
}

pub fn run(root: &Path, only: Option<&str>) -> ExitCode {
    let selected: Vec<&Gate> = match only {
        None => GATES.iter().collect(),
        Some(name) => match GATES.iter().find(|g| g.name == name) {
            Some(g) => vec![g],
            None => {
                eprintln!("check: no gate named `{name}`.");
                eprintln!(
                    "       known gates: {}",
                    GATES.iter().map(|g| g.name).collect::<Vec<_>>().join(", ")
                );
                return ExitCode::FAILURE;
            }
        },
    };

    let total = Instant::now();
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(selected.len());
    let mut failed_at: Option<usize> = None;

    for (i, gate) in selected.iter().enumerate() {
        if failed_at.is_some() {
            outcomes.push(Outcome::NotRun);
            continue;
        }
        println!();
        println!(
            "=== [{}/{}] {} — {}",
            i + 1,
            selected.len(),
            gate.name,
            gate.what
        );
        println!("    $ {}", gate.reproduce());
        let started = Instant::now();
        match gate.execute(root) {
            Ok(()) => outcomes.push(Outcome::Passed(started.elapsed())),
            Err(why) => {
                outcomes.push(Outcome::Failed(started.elapsed(), why));
                failed_at = Some(i);
            }
        }
    }

    print_summary(&selected, &outcomes, total.elapsed());

    match failed_at {
        None => ExitCode::SUCCESS,
        Some(i) => {
            let gate = selected[i];
            let not_run = selected.len() - i - 1;
            let why = match &outcomes[i] {
                Outcome::Failed(_, why) => why.as_str(),
                _ => "",
            };
            println!();
            println!(
                "check: FAILED at `{}` ({why}) after {}.",
                gate.name,
                secs(total.elapsed())
            );
            println!();
            println!("  reproduce:    {}", gate.reproduce());
            println!("  or just it:   cargo xtask check {}", gate.name);
            if not_run > 0 {
                println!();
                println!(
                    "  {not_run} later gate(s) did not run — the summary lists them as `not run`,"
                );
                println!("  not as passing. Fix this one and run `cargo xtask check` again.");
            }
            ExitCode::FAILURE
        }
    }
}

fn print_summary(gates: &[&Gate], outcomes: &[Outcome], total: Duration) {
    println!();
    println!("----------------------------");
    println!("{:<10}{:<10}{:>8}", "gate", "result", "time");
    for (gate, outcome) in gates.iter().zip(outcomes) {
        let (result, time) = match outcome {
            Outcome::Passed(d) => ("ok", secs(*d)),
            Outcome::Failed(d, _) => ("FAILED", secs(*d)),
            Outcome::NotRun => ("not run", "-".to_string()),
        };
        println!("{:<10}{result:<10}{time:>8}", gate.name);
    }
    println!("----------------------------");
    println!("{:<10}{:<10}{:>8}", "", "total", secs(total));
}

/// Wall time, at the only precision anyone reads off a gate summary.
fn secs(d: Duration) -> String {
    format!("{:.1}s", d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_gate_from_claude_md_is_in_the_list() {
        // The four commands CLAUDE.md names as the gate, verbatim enough to
        // notice if one is quietly dropped. This test is the honesty check on
        // the list itself: `check` is only worth running if it is all of them.
        let all: Vec<String> = GATES.iter().map(|g| g.reproduce()).collect();
        let joined = all.join("\n");
        for required in [
            "cargo test --workspace",
            "-p contentc -- --check",
            "cargo clippy --workspace --all-targets",
            "cargo fmt --all --check",
        ] {
            assert!(
                joined.contains(required),
                "gate list is missing `{required}`:\n{joined}"
            );
        }
    }

    #[test]
    fn clippy_is_enforced_at_zero_findings() {
        let clippy = GATES
            .iter()
            .find(|g| g.name == "clippy")
            .expect("a clippy gate");
        assert!(
            clippy.reproduce().ends_with("-- -D warnings"),
            "clippy without -D warnings exits 0 on findings and gates nothing"
        );
    }

    #[test]
    fn gate_names_are_unique_so_running_one_is_unambiguous() {
        let mut names: Vec<&str> = GATES.iter().map(|g| g.name).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two gates share a name");
    }
}
