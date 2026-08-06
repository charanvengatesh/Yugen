//! The tuning index — every tuned number in the game, in one place.
//!
//! ```text
//! cargo xtask tuning          rewrite docs/TUNING.md from the source
//! cargo xtask tuning --check  exit 1 if it is stale, undocumented, or shadowed
//! ```
//!
//! # The three-tier standard
//!
//! Every tuned number lives in exactly one of three tiers, and which tier is
//! decided by WHAT THE NUMBER IS ABOUT, never by who happens to read it:
//!
//! 1. `content/` describes one THING — a block's hardness, a mob's speed, a
//!    weapon's damage. Compiled by `contentc` into `crates/godgame-data/`.
//!    Adding a thing is adding a record.
//! 2. `crates/godgame-core/src/config/` describes the WHOLE GAME — the cell
//!    grid, gravity, sea level, the swing window. Anything two modules must
//!    agree on, or that a designer would reach for. One domain per file.
//! 3. A module constant describes ONE ALGORITHM — an fBm octave count, a
//!    gradient stop, a hysteresis band. It stays with the code it explains.
//!
//! # Why this is an index and not a linter
//!
//! The obvious tool here is a rule that forces every tier-3 constant to the top
//! of its file. The TypeScript original tried that and threw it away: the tier-3
//! constants are deliberately interleaved with the prose that explains them in
//! the context of the function below, each with its derivation written out.
//! Hoisting them would separate every number from its reason, trading real
//! documentation for a cosmetic property.
//!
//! The actual problem a scattered constant causes is that you cannot FIND it. So
//! this generates the finding aid instead — one page listing all three tiers
//! with file, line, value and meaning — which makes the code's own organisation
//! the thing you navigate rather than the thing you fight.
//!
//! # What is enforced
//!
//! Two rules only, both chosen because a human reviewer reliably misses them.
//!
//! **NO SHADOWING.** A module constant must not reuse the name of a `config`
//! export. In Rust a glob import loses to a local item of the same name without
//! a word from the compiler, so the day someone adds `use config::*` to that
//! file, the local declaration wins, the module keeps compiling, and it is now
//! quietly running on a different number than the rest of the game. Rust makes
//! this hazard sharper than TypeScript did, not softer: there is no import list
//! to read, and no error.
//!
//! **TIER 2 IS DOCUMENTED.** Every `pub` item under `config/` needs a doc
//! comment. Asked of tier 2 and not of tier 3 because tier 2 is the
//! designer-facing surface: a number with no stated meaning is one nobody can
//! safely turn, which defeats the only reason it was promoted out of a module in
//! the first place.
//!
//! # Staleness
//!
//! `--check` compares the rendered page byte for byte against the committed one,
//! the same contract `contentc --check` holds for the generated tables. The
//! render is a pure function of the tree, so regenerating twice cannot differ;
//! anything else would make the gate a coin flip.

mod content;
mod render;
mod rust_consts;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rust_consts::Konst;

/// Where tier 2 lives. The one path in the tree with a rule attached to it.
const CONFIG_DIR: &str = "crates/godgame-core/src/config";

/// Crates the tier-3 sweep does not read.
///
/// - `godgame-data` is `contentc` output. Its numbers are tier 1 already, seen
///   through the compiler; indexing the emitted form would double-count the
///   whole content tier and invite someone to edit a generated file.
/// - `contentc` and `xtask` are build tooling, not the game. The TypeScript drew
///   this line by keeping its compiler under `tools/` and scanning only `src/`.
///   A schema's internal enum encoding (`STATE_SOLID = 1.0`) is not a knob.
const SKIP_CRATES: &[&str] = &["godgame-data", "contentc", "xtask"];

/// Where the page is written. Also the path `--check` compares against.
const OUT: &str = "docs/TUNING.md";

pub enum Mode {
    Write,
    Check,
}

/// One `config/` module and the constants it owns.
pub struct Domain {
    /// File stem: `physics`, `world`, ...
    pub module: String,
    pub path: String,
    pub consts: Vec<Konst>,
}

/// One tier-3 source file and the constants it owns.
pub struct SourceFile {
    pub path: String,
    pub consts: Vec<Konst>,
}

/// The whole index, built from the tree.
pub struct Index {
    pub tier1: Vec<content::Kind>,
    pub tier2: Vec<Domain>,
    pub tier3: Vec<SourceFile>,
    /// Every module-level name declared under `config/`, numeric or not, mapped
    /// to the file that declares it.
    ///
    /// Wider than the tier-2 table on purpose. The table is a tuning index and
    /// only lists numbers; the shadowing hazard is about NAMES, and a config
    /// export holding a table or a string is shadowed by a local declaration
    /// exactly as silently as one holding a float.
    pub config_names: std::collections::BTreeMap<String, String>,
}

impl Index {
    pub fn tier2_count(&self) -> usize {
        self.tier2.iter().map(|d| d.consts.len()).sum()
    }

    pub fn tier3_count(&self) -> usize {
        self.tier3.iter().map(|f| f.consts.len()).sum()
    }

    pub fn record_count(&self) -> usize {
        self.tier1.iter().map(|k| k.records).sum()
    }
}

/// Run the index. `Ok` carries the one-line summary, `Err` the whole report.
///
/// # Why the page is written before the rules are judged
///
/// The TypeScript bailed out on a violation and wrote nothing. That is the wrong
/// order here. The index and the rules answer different questions — "where is
/// this number" and "is this tree sound" — and refusing to write the finding aid
/// because a name collides somewhere else means the first thing a developer
/// wants while fixing the collision is the thing they cannot have. `--check`
/// writes nothing either way, so the gate is unaffected: a violation fails it,
/// and the failure text is the whole report, not just the first problem found.
pub fn run(root: &Path, mode: Mode) -> Result<String, String> {
    let index = build(root)?;
    let rendered = render::page(&index);
    let out = root.join(OUT);
    let current = std::fs::read_to_string(&out).unwrap_or_default();
    let counts = format!(
        "{} content records, {} config constants, {} module constants",
        index.record_count(),
        index.tier2_count(),
        index.tier3_count()
    );

    let mut headline = format!("tuning: up to date — {counts}");
    let mut report = String::new();

    match mode {
        Mode::Write if current != rendered => {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}\n", parent.display()))?;
            }
            std::fs::write(&out, &rendered).map_err(|e| format!("{}: {e}\n", out.display()))?;
            headline = format!("tuning: wrote {OUT} — {counts}");
        }
        Mode::Check if current != rendered => report.push_str(&format!(
            "tuning: {OUT} is stale.\n\
             \n\
             The page is generated from the source, so a constant was added,\n\
             moved, renamed, retyped or re-documented without regenerating it.\n\
             \n\
             \x20 fix:  cargo xtask tuning\n\
             \n"
        )),
        _ => {}
    }

    let undocumented = undocumented(root)?;
    if !undocumented.is_empty() {
        report.push_str(&undocumented_report(&undocumented));
    }
    let shadows = shadows(&index);
    if !shadows.is_empty() {
        report.push_str(&shadow_report(&shadows));
    }

    if report.is_empty() {
        Ok(headline)
    } else {
        Err(format!("{headline}\n\n{report}"))
    }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

fn build(root: &Path) -> Result<Index, String> {
    let tier1 = content::scan(&root.join("content"))?;

    let mut tier2 = Vec::new();
    let mut config_names = std::collections::BTreeMap::new();
    for file in rust_files(&root.join(CONFIG_DIR))? {
        let rel = relative(root, &file);
        let module = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let all = scan_file(&file, &rel)?;
        for k in &all {
            config_names.insert(k.name.clone(), rel.clone());
        }
        // `mod.rs` is the domain map and the re-exports; it declares no numbers.
        if module == "mod" {
            continue;
        }
        let consts: Vec<Konst> = all.into_iter().filter(|k| k.tuned).collect();
        if !consts.is_empty() {
            tier2.push(Domain {
                module,
                path: rel,
                consts,
            });
        }
    }
    tier2.sort_by(|a, b| a.module.cmp(&b.module));

    let config_dir = root.join(CONFIG_DIR);
    let mut tier3 = Vec::new();
    for src_dir in crate_src_dirs(root)? {
        for file in rust_files(&src_dir)? {
            if file.starts_with(&config_dir) {
                continue;
            }
            let rel = relative(root, &file);
            let consts: Vec<Konst> = scan_file(&file, &rel)?
                .into_iter()
                .filter(|k| k.tuned)
                .collect();
            if !consts.is_empty() {
                tier3.push(SourceFile { path: rel, consts });
            }
        }
    }
    tier3.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(Index {
        tier1,
        tier2,
        tier3,
        config_names,
    })
}

fn scan_file(file: &Path, rel: &str) -> Result<Vec<Konst>, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("{rel}: {e}\n"))?;
    rust_consts::scan(&text).map_err(|e| {
        format!("tuning: {rel} does not parse as Rust — {e}\n(the tree must build before it can be indexed)\n")
    })
}

/// `crates/*/src`, minus the tooling and the generated crate, in a fixed order.
fn crate_src_dirs(root: &Path) -> Result<Vec<PathBuf>, String> {
    let crates = root.join("crates");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&crates)
        .map_err(|e| format!("{}: {e}\n", crates.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned());
            p.is_dir() && !name.is_some_and(|n| SKIP_CRATES.contains(&n.as_str()))
        })
        .map(|p| p.join("src"))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    Ok(dirs)
}

/// Every `.rs` file under a directory, sorted, so the page is reproducible.
///
/// `tests/` and `benches/` sit beside `src/` and are never reached: a fixture
/// threshold and a bench's iteration count are not the game's tuning, and
/// indexing them would make the page churn whenever a test grew.
fn rust_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}\n", d.display()))? {
            let path = entry.map_err(|e| format!("{}: {e}\n", d.display()))?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

// ---------------------------------------------------------------------------
// The two rules
// ---------------------------------------------------------------------------

struct Missing {
    path: String,
    item: rust_consts::Undocumented,
}

fn undocumented(root: &Path) -> Result<Vec<Missing>, String> {
    let mut out = Vec::new();
    for file in rust_files(&root.join(CONFIG_DIR))? {
        let rel = relative(root, &file);
        let text = std::fs::read_to_string(&file).map_err(|e| format!("{rel}: {e}\n"))?;
        let items = rust_consts::undocumented_pub_items(&text)
            .map_err(|e| format!("tuning: {rel} does not parse as Rust — {e}\n"))?;
        out.extend(items.into_iter().map(|item| Missing {
            path: rel.clone(),
            item,
        }));
    }
    Ok(out)
}

struct Shadow {
    path: String,
    konst: Konst,
    config: String,
}

fn shadows(index: &Index) -> Vec<Shadow> {
    let mut out = Vec::new();
    for file in &index.tier3 {
        for k in &file.consts {
            if let Some(config) = index.config_names.get(&k.name) {
                out.push(Shadow {
                    path: file.path.clone(),
                    konst: k.clone(),
                    config: config.clone(),
                });
            }
        }
    }
    out
}

fn undocumented_report(missing: &[Missing]) -> String {
    let mut s = format!(
        "tuning: {} `pub` item(s) under {CONFIG_DIR}/ have no doc comment.\n\
         \n\
         Tier 2 is the designer-facing surface — a number with no stated meaning\n\
         is one nobody can safely turn. Add a doc comment, or demote it to a\n\
         module constant if it is really about one algorithm.\n\
         \n",
        missing.len()
    );
    for m in missing {
        let _ = writeln!(
            s,
            "  {}:{}  {} {}",
            m.path, m.item.line, m.item.kind, m.item.name
        );
    }
    s
}

fn shadow_report(shadows: &[Shadow]) -> String {
    let mut s = format!(
        "tuning: {} module constant(s) reuse the name of a config export.\n\
         \n\
         Rename the local one. Today they merely coexist; the day that module\n\
         writes `use crate::config::*`, the local declaration wins the name\n\
         silently — a glob import loses to a local item with no warning — and the\n\
         module runs on a different number than the rest of the game.\n\
         \n",
        shadows.len()
    );
    for sh in shadows {
        let _ = writeln!(
            s,
            "  {}:{}  {}  shadows {}",
            sh.path, sh.konst.line, sh.konst.name, sh.config
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn konst(name: &str) -> Konst {
        Konst {
            name: name.to_string(),
            value: "1.0".into(),
            line: 7,
            doc: "A knob.".into(),
            exported: true,
            tuned: true,
        }
    }

    fn index_with(config: &[&str], module: &[&str]) -> Index {
        let path = format!("{CONFIG_DIR}/world.rs");
        Index {
            tier1: Vec::new(),
            tier2: vec![Domain {
                module: "world".into(),
                path: path.clone(),
                consts: config.iter().map(|n| konst(n)).collect(),
            }],
            tier3: vec![SourceFile {
                path: "crates/godgame-core/src/sim/caves.rs".into(),
                consts: module.iter().map(|n| konst(n)).collect(),
            }],
            config_names: config
                .iter()
                .map(|n| ((*n).to_string(), path.clone()))
                .collect(),
        }
    }

    #[test]
    fn a_module_constant_reusing_a_config_name_is_reported_as_a_shadow() {
        let found = shadows(&index_with(&["CELL_SIZE"], &["CELL_SIZE", "OCTAVES"]));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].konst.name, "CELL_SIZE");
        assert!(found[0].config.ends_with("world.rs"));
    }

    #[test]
    fn distinct_names_across_the_two_tiers_are_not_a_shadow() {
        assert!(shadows(&index_with(&["CELL_SIZE"], &["OCTAVES"])).is_empty());
    }

    #[test]
    fn the_shadow_report_names_the_file_the_line_and_the_config_it_collides_with() {
        let report = shadow_report(&shadows(&index_with(&["CELL_SIZE"], &["CELL_SIZE"])));
        assert!(report.contains("caves.rs:7"));
        assert!(report.contains("CELL_SIZE"));
        assert!(report.contains("world.rs"));
    }

    #[test]
    fn the_undocumented_report_names_every_offender_with_its_kind() {
        let report = undocumented_report(&[Missing {
            path: format!("{CONFIG_DIR}/worldgen.rs"),
            item: rust_consts::Undocumented {
                kind: "const",
                name: "SEED".into(),
                line: 16,
            },
        }]);
        assert!(report.contains("worldgen.rs:16  const SEED"));
        assert!(report.contains("1 `pub` item"));
    }

    #[test]
    fn the_generated_crate_and_the_build_tooling_are_out_of_the_tier_three_sweep() {
        // Naming them here so removing one from the list breaks a test rather
        // than silently doubling the content tier into the page.
        assert!(SKIP_CRATES.contains(&"godgame-data"));
        assert!(SKIP_CRATES.contains(&"contentc"));
        assert!(SKIP_CRATES.contains(&"xtask"));
    }
}
