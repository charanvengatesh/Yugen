//! The compile driver: find the repo, parse everything, then emit everything.
//!
//! The two pre-passes are the whole trick. Kinds reference each other in both
//! directions — a block drops an item, an item places a block — so no compile
//! order can make one kind's registry available to the other. Parsing is cheap
//! and order-free, so the id sets are collected first, then the CODES for every
//! kind, and only then does any kind emit a table.

use crate::emit::{Foreign, assign_kind_codes, compile};
use crate::error::Result;
use crate::lock::Lock;
use crate::schemas::{Kind, kinds};
use crate::toml_in::{RawRecord, parse_file};
use crate::{bail, err};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct Layout {
    pub root: PathBuf,
    pub content: PathBuf,
    pub lock: PathBuf,
    /// Where the generated crate's `src/` lives.
    pub data_src: PathBuf,
}

/// Locate the repo root by walking up for the `content/` + `Cargo.toml` pair.
///
/// Cannot be a fixed `../..`: the compiler is run through `cargo run` from
/// anywhere in the workspace, and from the xtask gate, so searching for the
/// thing we actually need is the only stable answer.
pub fn find_root() -> Result<Layout> {
    let start =
        std::env::current_dir().map_err(|e| err!("cannot read the working directory: {e}"))?;
    let mut dir = start.as_path();
    loop {
        if dir.join("content").is_dir() && dir.join("Cargo.toml").is_file() {
            return Ok(Layout {
                root: dir.to_path_buf(),
                content: dir.join("content"),
                lock: dir.join("content").join("ids.lock.json"),
                data_src: dir.join("crates").join("godgame-data").join("src"),
            });
        }
        match dir.parent() {
            Some(up) => dir = up,
            None => bail!("could not locate the repo root (no content/ + Cargo.toml)"),
        }
    }
}

/// Source files for one kind, sorted so code assignment — and therefore the lock
/// file — does not depend on directory iteration order.
fn sources(layout: &Layout, kind: &Kind) -> Result<Vec<PathBuf>> {
    let dir = layout.content.join(kind.dir);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| err!("cannot read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    out.sort();
    Ok(out)
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

pub struct RunResult {
    /// repo-relative path -> desired contents, for every file the compile owns.
    pub files: Vec<(String, String)>,
    pub warnings: Vec<String>,
    /// Records compiled per kind, for the CLI summary.
    pub counts: Vec<(String, usize)>,
}

/// Compile everything in memory. No writes — the caller decides what to do.
pub fn run() -> Result<RunResult> {
    let layout = find_root()?;
    let lock = if layout.lock.is_file() {
        let text = std::fs::read_to_string(&layout.lock)
            .map_err(|e| err!("cannot read {}: {e}", layout.lock.display()))?;
        Lock::from_json(&text)?
    } else {
        Lock::default()
    };

    let all = kinds();
    let mut files: Vec<(String, String)> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut counts: Vec<(String, usize)> = Vec::new();
    let mut next_lock = lock.clone();

    // Pre-pass 1: parse every kind and collect its id set.
    let mut parsed: Vec<(Vec<RawRecord>, Vec<String>)> = Vec::new();
    let mut foreign: Foreign = Foreign::new();
    for kind in &all {
        let paths = sources(&layout, kind)?;
        let mut records = Vec::new();
        for p in &paths {
            records.extend(parse_file(p, &layout.content, &kind.schema.kind)?);
        }
        // Tombstoned ids stay referenceable: a retired item that some block
        // still lists as a drop should degrade to the placeholder def, not fail
        // the build.
        let mut ids: HashSet<String> = lock
            .kind(&kind.schema.kind)
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        for rec in &records {
            ids.insert(rec.id.clone());
        }
        foreign.insert(kind.schema.kind.clone(), ids);
        parsed.push((
            records,
            paths.iter().map(|p| rel(&layout.root, p)).collect(),
        ));
    }

    // Pre-pass 2: assign every kind's CODES before emitting any kind, so a
    // generated table may reference another kind (`ITEM_PLACES` holds block
    // codes). Ids alone are not enough for that — only the id set is order-free,
    // the codes are not.
    for kind in &all {
        let k = &kind.schema.kind;
        let mut ids: Vec<String> = foreign[k].iter().cloned().collect();
        // The set has no order of its own, so impose one: ids already in the
        // lock first, by code, then the rest alphabetically. Without this, a new
        // id's code would depend on hash iteration order.
        let prior = next_lock.kind(k).to_vec();
        ids.sort_by_key(|id| {
            let locked = prior.iter().find(|(k2, _)| k2 == id).map(|(_, c)| *c);
            (locked.is_none(), locked.unwrap_or(0), id.clone())
        });
        let assigned = assign_kind_codes(k, &ids, &next_lock);
        next_lock.set_kind(k, assigned);
    }

    for (kind, (records, rel_paths)) in all.iter().zip(parsed.iter()) {
        let result = compile(&kind.schema, records, &next_lock, rel_paths, &foreign)?;
        next_lock = result.lock;
        warnings.extend(result.warnings);
        counts.push((kind.schema.kind.clone(), result.count));
        files.push((
            format!("crates/godgame-data/src/{}.rs", kind.out),
            result.source,
        ));
    }

    files.push(("crates/godgame-data/src/lib.rs".to_string(), lib_rs(&all)));

    // Format the generated Rust before anyone compares it to disk.
    //
    // Without this, `--check` fails on a pristine checkout forever: the tree is
    // rustfmt'd, the emitter is not, and every run reports six stale files. A
    // gate that always fires is a gate people learn to ignore, which is worse
    // than not having one. Formatting here also means the emitter never has to
    // think about line width or trailing commas.
    for (path, source) in &mut files {
        if path.ends_with(".rs") {
            *source = rustfmt(source);
        }
    }

    // The lock is JSON with its own byte-stable writer, so it is added after
    // formatting rather than being excluded by the filter above.
    files.push((rel(&layout.root, &layout.lock), next_lock.to_json()));

    Ok(RunResult {
        files,
        warnings,
        counts,
    })
}

/// Run generated source through `rustfmt`, or return it unchanged.
///
/// Falling back rather than failing is deliberate: `rustfmt` is a toolchain
/// component that a minimal CI image can be missing, and the compiler's job is
/// to produce correct tables. An unformatted table is ugly; a build that cannot
/// run without an optional component is broken.
fn rustfmt(source: &str) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let Ok(mut child) = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout", "--quiet"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return source.to_string();
    };

    if let Some(mut stdin) = child.stdin.take()
        && stdin.write_all(source.as_bytes()).is_err()
    {
        return source.to_string();
    }

    match child.wait_with_output() {
        Ok(out) if out.status.success() => {
            String::from_utf8(out.stdout).unwrap_or_else(|_| source.to_string())
        }
        _ => source.to_string(),
    }
}

/// `godgame-data`'s root module. Generated too, so adding a kind is still one
/// row in `schemas::kinds()`.
fn lib_rs(all: &[Kind]) -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by contentc — DO NOT EDIT\n\n");
    out.push_str("//! Compiler output.\n//!\n");
    out.push_str("//! Every module here is emitted by `cargo run -p contentc` from the records\n");
    out.push_str("//! under `content/`. **Do not edit these files by hand** — the next compile\n");
    out.push_str("//! overwrites them, and `cargo run -p contentc -- --check` is the gate that\n");
    out.push_str("//! fails if they are stale.\n//!\n");
    out.push_str(
        "//! The arrow points one way: `content/` describes what things ARE, this crate\n",
    );
    out.push_str(
        "//! is the compiled form of that, and `godgame-core` reads it and never writes\n",
    );
    out.push_str("//! it. That is what makes adding a block a content change rather than a code\n");
    out.push_str("//! change.\n\n");
    for kind in all {
        out.push_str(&format!("pub mod {};\n", kind.out));
    }
    out.push_str(
        r#"
/// One flat table, type-erased so the gates can walk them by name.
///
/// The game never touches this — it indexes the concrete `[u8; N]` statics
/// directly. It exists because Rust cannot look a static up by name and two
/// consumers need to: the parity test that diffs these tables against the
/// TypeScript build, and the tuning index.
#[derive(Clone, Copy, Debug)]
pub enum Table {
    U8(&'static [u8]),
    U16(&'static [u16]),
    U32(&'static [u32]),
    F32(&'static [f32]),
}

impl Table {
    pub fn len(&self) -> usize {
        match self {
            Table::U8(v) => v.len(),
            Table::U16(v) => v.len(),
            Table::U32(v) => v.len(),
            Table::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The slot as an `f64`, which is lossless for every width used here.
    pub fn get(&self, i: usize) -> f64 {
        match self {
            Table::U8(v) => v[i] as f64,
            Table::U16(v) => v[i] as f64,
            Table::U32(v) => v[i] as f64,
            Table::F32(v) => v[i] as f64,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Table::U8(_) => "u8",
            Table::U16(_) => "u16",
            Table::U32(_) => "u32",
            Table::F32(_) => "f32",
        }
    }
}

/// Every module's tables, for the gates. `(module, name, table)`.
pub fn all_tables() -> Vec<(&'static str, &'static str, Table)> {
    let mut out = Vec::new();
"#,
    );
    for kind in all {
        out.push_str(&format!(
            "    for (n, t) in {}::{}_TABLES {{\n        out.push((\"{}\", *n, *t));\n    }}\n",
            kind.out, kind.schema.prefix, kind.out
        ));
    }
    out.push_str("    out\n}\n");

    out.push_str(
        "\n/// Every module's authoring id -> code, for the gates. `(module, id, code)`.\npub fn all_codes() -> Vec<(&'static str, &'static str, u16)> {\n    let mut out = Vec::new();\n",
    );
    for kind in all {
        out.push_str(&format!(
            "    for (id, c) in {}::{}_CODES {{\n        out.push((\"{}\", *id, *c));\n    }}\n",
            kind.out, kind.schema.prefix, kind.out
        ));
    }
    out.push_str("    out\n}\n");

    out.push_str(
        "\n/// Every module's pair matrices, for the gates. `(module, name, side, bits)`.\npub fn all_matrices() -> Vec<(&'static str, &'static str, usize, &'static [u8])> {\n    #[allow(unused_mut)]\n    let mut out = Vec::new();\n",
    );
    for kind in all {
        if kind.schema.matrices.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "    for (n, side, bits) in {}::{}_MATRICES {{\n        out.push((\"{}\", *n, *side, *bits));\n    }}\n",
            kind.out, kind.schema.prefix, kind.out
        ));
    }
    out.push_str("    out\n}\n");
    out
}

/// Write (or check) the compile output. Returns the process exit code.
pub fn main(argv: &[String]) -> i32 {
    let check = argv.iter().any(|a| a == "--check");

    let result = match run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("contentc: {e}");
            return 1;
        }
    };
    for w in &result.warnings {
        eprintln!("contentc: warning: {w}");
    }

    let layout = match find_root() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("contentc: {e}");
            return 1;
        }
    };

    let mut stale: Vec<String> = Vec::new();
    for (path, text) in &result.files {
        let abs = layout.root.join(path);
        let current = std::fs::read_to_string(&abs).ok();
        if current.as_deref() == Some(text.as_str()) {
            continue;
        }
        stale.push(path.clone());
        if !check {
            if let Some(parent) = abs.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                eprintln!("contentc: cannot create {}: {e}", parent.display());
                return 1;
            }
            if let Err(e) = std::fs::write(&abs, text) {
                eprintln!("contentc: cannot write {}: {e}", abs.display());
                return 1;
            }
        }
    }

    let summary: Vec<String> = result
        .counts
        .iter()
        .map(|(k, n)| format!("{n} {k}"))
        .collect();

    if check {
        if !stale.is_empty() {
            eprintln!("contentc --check: out of date: {}", stale.join(", "));
            eprintln!("run `cargo run -p contentc` and commit the result");
            return 1;
        }
        println!("contentc --check: up to date ({})", summary.join(", "));
        return 0;
    }
    if stale.is_empty() {
        println!("contentc: up to date ({})", summary.join(", "));
    } else {
        println!(
            "contentc: wrote {} ({})",
            stale.join(", "),
            summary.join(", ")
        );
    }
    0
}
