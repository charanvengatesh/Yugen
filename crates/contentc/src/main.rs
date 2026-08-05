//! `contentc` — the content compiler CLI.
//!
//! ```text
//! cargo run -p contentc              compile, writing changed files
//! cargo run -p contentc -- --check   exit 1 if anything WOULD change (the gate)
//! ```
//!
//! There is no `--watch`: `cargo watch -x 'run -p contentc'` already does it,
//! and the TS version only carried one because the Vite plugin needed it.

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(contentc::driver::main(&argv));
}
