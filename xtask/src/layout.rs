//! The filing gate — `content/LAYOUT.toml` against `content/`.
//!
//! # What this is for
//!
//! `contentc` reads every `*.toml` in a kind's directory and concatenates them,
//! so filenames mean nothing to the build: a new ore compiles identically from
//! `ores.toml` or `terrain.toml`. The filing is convention, and until
//! `content/LAYOUT.toml` there was nowhere it was written down. `FORMAT.md` §0
//! is the rule; this is the thing that makes the rule true.
//!
//! # What it actually catches
//!
//! Not, mostly, records in the wrong file. Only two rows in the manifest can
//! carry a checkable `rule`, because the fields that look like discriminators
//! are lists that overlap (`bands`, `biomes`) or span several files by design
//! (`category`, `place`). See `LAYOUT.toml`'s own header.
//!
//! The teeth are the **completeness** check: every file has a row and every row
//! has a file. That is what stops the failure mode that actually happens — a
//! record dropped into a new file, or into whichever file was already open,
//! with nobody stating what that file is for. An unclaimed file fails the gate,
//! so the choice has to be made out loud.
//!
//! # Why every error, not the first
//!
//! A filing pass is a batch job: someone adds four records and gets four
//! placements wrong in the same sitting. Stopping at the first would make that
//! four edit-run cycles. Everything here accumulates into one report.
//!
//! # No write mode
//!
//! Unlike [`crate::tuning`] and [`crate::font`], there is nothing to generate.
//! A manifest this gate could write from the tree would only ever assert that
//! the tree looks like itself; the `about` strings — the part with the value in
//! them — are a human's to write.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A row of the manifest: what a file is for, and optionally a checkable rule.
struct Row {
    about: String,
    rule: Option<Rule>,
}

/// A rule the gate can prove. `field` is a dotted path into a record.
struct Rule {
    field: String,
    equals: String,
}

/// Directories under `content/` that are not content kinds, from
/// `LAYOUT.toml`'s `not_a_kind`.
///
/// Public because [`crate::tuning`] needs the same answer and must not hold a
/// second copy of it: `content/fonts/` is raw assets read by `cargo xtask
/// font`, and a tuning index that lists it as a kind with zero records is
/// reporting a directory that does not exist as a concept. One declaration,
/// read twice.
pub fn not_a_kind(root: &Path) -> Result<BTreeSet<String>, String> {
    let manifest = parse_manifest(root)?;
    Ok(manifest.0)
}

/// `(not_a_kind, kind -> file stem -> row)`.
type Manifest = (BTreeSet<String>, BTreeMap<String, BTreeMap<String, Row>>);

fn parse_manifest(root: &Path) -> Result<Manifest, String> {
    let path = root.join("content/LAYOUT.toml");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let table: toml::Table = text
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let mut not_a_kind = BTreeSet::new();
    let mut kinds: BTreeMap<String, BTreeMap<String, Row>> = BTreeMap::new();

    for (key, value) in &table {
        if key == "not_a_kind" {
            let list = value
                .as_array()
                .ok_or_else(|| "LAYOUT.toml: `not_a_kind` must be an array".to_string())?;
            for entry in list {
                let name = entry.as_str().ok_or_else(|| {
                    "LAYOUT.toml: every `not_a_kind` entry must be a string".to_string()
                })?;
                not_a_kind.insert(name.to_string());
            }
            continue;
        }

        let files = value
            .as_table()
            .ok_or_else(|| format!("LAYOUT.toml: `{key}` must be a table of files"))?;
        let mut rows = BTreeMap::new();
        for (stem, row) in files {
            let row = row
                .as_table()
                .ok_or_else(|| format!("LAYOUT.toml: `{key}.{stem}` must be a table"))?;
            let about = row
                .get("about")
                .and_then(|a| a.as_str())
                .ok_or_else(|| format!("LAYOUT.toml: `{key}.{stem}` needs an `about` string"))?;
            if about.trim().is_empty() {
                return Err(format!(
                    "LAYOUT.toml: `{key}.{stem}` has an empty `about`. A row that says \
                     nothing claims the file without explaining it, which is the state \
                     this manifest exists to end."
                ));
            }
            let rule = match row.get("rule") {
                None => None,
                Some(r) => {
                    let r = r.as_table().ok_or_else(|| {
                        format!(
                            "LAYOUT.toml: `{key}.{stem}.rule` must be a table of \
                             `field` and `equals`. Omit `rule` entirely for a row \
                             enforced by convention."
                        )
                    })?;
                    let field = r.get("field").and_then(|f| f.as_str()).ok_or_else(|| {
                        format!("LAYOUT.toml: `{key}.{stem}.rule` needs a `field` string")
                    })?;
                    let equals = r.get("equals").and_then(|f| f.as_str()).ok_or_else(|| {
                        format!("LAYOUT.toml: `{key}.{stem}.rule` needs an `equals` string")
                    })?;
                    Some(Rule {
                        field: field.to_string(),
                        equals: equals.to_string(),
                    })
                }
            };
            rows.insert(
                stem.to_string(),
                Row {
                    about: about.to_string(),
                    rule,
                },
            );
        }
        kinds.insert(key.to_string(), rows);
    }

    Ok((not_a_kind, kinds))
}

/// One record's fields, flattened enough to answer a dotted rule.
type Records = BTreeMap<String, toml::Table>;

fn records_of(path: &Path) -> Result<Records, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let table: toml::Table = text
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = BTreeMap::new();
    for (id, value) in &table {
        if let Some(t) = value.as_table() {
            out.insert(id.clone(), t.clone());
        }
    }
    Ok(out)
}

/// Follow a dotted path (`state`, `heat.conduct`) and render the leaf as a
/// string, or `None` if the path is absent or is not a scalar.
///
/// Deliberately string-based. `xtask` links nothing from the workspace, so it
/// cannot import the schema's enums — and making it try would put the block
/// state vocabulary in a third place, which is the seam this repo is trying to
/// close rather than widen. A misspelled variant is `contentc`'s error to
/// raise, at the next gate along.
fn field_str(record: &toml::Table, path: &str) -> Option<String> {
    let mut cur = record.get(path.split('.').next()?)?;
    for seg in path.split('.').skip(1) {
        cur = cur.as_table()?.get(seg)?;
    }
    match cur {
        toml::Value::String(s) => Some(s.clone()),
        toml::Value::Integer(i) => Some(i.to_string()),
        toml::Value::Boolean(b) => Some(b.to_string()),
        _ => None,
    }
}

pub fn run(root: &Path) -> Result<String, String> {
    let (not_a_kind, manifest) = parse_manifest(root)?;
    let content = root.join("content");

    let mut problems: Vec<String> = Vec::new();
    let mut files_seen = 0usize;
    let mut records_seen = 0usize;
    let mut rules_checked = 0usize;

    // Every directory under `content/` is a declared kind, or declared not to be
    // one. A kind that appears on disk and nowhere in the manifest is the most
    // important thing this gate can catch, because it is a whole category of
    // content nobody has said anything about.
    let mut on_disk: BTreeSet<String> = BTreeSet::new();
    let entries = std::fs::read_dir(&content).map_err(|e| format!("content/: {e}"))?;
    for entry in entries.filter_map(Result::ok).filter(|e| e.path().is_dir()) {
        if let Some(name) = entry.file_name().to_str() {
            on_disk.insert(name.to_string());
        }
    }
    for dir in &on_disk {
        if !manifest.contains_key(dir) && !not_a_kind.contains(dir) {
            problems.push(format!(
                "content/{dir}/ has no section in LAYOUT.toml. Add one, or add \
                 \"{dir}\" to `not_a_kind` if it is raw assets rather than records."
            ));
        }
    }
    for kind in manifest.keys() {
        if !on_disk.contains(kind) {
            problems.push(format!(
                "LAYOUT.toml declares a `[{kind}]` section but content/{kind}/ does not exist."
            ));
        }
    }

    for (kind, rows) in &manifest {
        let dir = content.join(kind);
        if !dir.is_dir() {
            continue; // already reported above
        }

        let mut stems: BTreeSet<String> = BTreeSet::new();
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for path in entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                stems.insert(stem.to_string());
            }
        }

        for stem in &stems {
            if !rows.contains_key(stem) {
                // Print the sibling rows and what each is for. The author of an
                // unclaimed file is deciding between "this is a new category"
                // and "this belongs in one of the existing ones", and that is
                // not a decision anyone should have to open another file to
                // make.
                let mut choices = String::new();
                for (other, row) in rows {
                    choices.push_str(&format!("\n      {other}.toml — {}", row.about));
                }
                problems.push(format!(
                    "content/{kind}/{stem}.toml has no row in LAYOUT.toml. Either add \
                     one saying what belongs in it, or move its records into a file \
                     that is already claimed:{choices}"
                ));
            }
        }
        for stem in rows.keys() {
            if !stems.contains(stem) {
                problems.push(format!(
                    "LAYOUT.toml claims content/{kind}/{stem}.toml, which does not exist."
                ));
            }
        }

        // Load once: both the emptiness check and the partition check read them.
        let mut loaded: BTreeMap<&String, Records> = BTreeMap::new();
        for stem in stems.intersection(&rows.keys().cloned().collect()) {
            let path = dir.join(format!("{stem}.toml"));
            let recs = records_of(&path)?;
            files_seen += 1;
            records_seen += recs.len();
            if recs.is_empty() {
                problems.push(format!(
                    "content/{kind}/{stem}.toml has no records. An empty claimed file \
                     is a category with nothing in it — delete the file and its row, \
                     or author the record it was made for."
                ));
            }
            let key = rows.get_key_value(stem).expect("intersection").0;
            loaded.insert(key, recs);
        }

        // A rule must PARTITION: everything in the file matches, and nothing in a
        // sibling does. Half of that is the half people forget — a rule that only
        // checks its own file is satisfied by a tree where the same value is
        // scattered across five others.
        for (stem, row) in rows {
            let Some(rule) = &row.rule else { continue };
            let Some(mine) = loaded.get(stem) else {
                continue;
            };
            rules_checked += 1;

            for (id, record) in mine {
                match field_str(record, &rule.field) {
                    Some(v) if v == rule.equals => {}
                    Some(v) => problems.push(format!(
                        "content/{kind}/{stem}.toml claims `{} = \"{}\"` but `{id}` has \
                         `{} = \"{v}\"`.",
                        rule.field, rule.equals, rule.field
                    )),
                    None => problems.push(format!(
                        "content/{kind}/{stem}.toml claims `{} = \"{}\"` but `{id}` does \
                         not set `{}` at all.",
                        rule.field, rule.equals, rule.field
                    )),
                }
            }

            for (other, records) in &loaded {
                if *other == stem {
                    continue;
                }
                for (id, record) in records {
                    if field_str(record, &rule.field).as_deref() == Some(rule.equals.as_str()) {
                        problems.push(format!(
                            "`{id}` in content/{kind}/{other}.toml has `{} = \"{}\"`, which \
                             {stem}.toml claims. A rule that does not partition is a rule \
                             the gate cannot enforce — move the record, or drop the `rule` \
                             from the {stem} row and say the filing is a convention.",
                            rule.field, rule.equals
                        ));
                    }
                }
            }
        }
    }

    if problems.is_empty() {
        let kinds = manifest.len();
        return Ok(format!(
            "layout: ok — {files_seen} files, {records_seen} records, {kinds} kinds, \
             {rules_checked} rule(s) enforced"
        ));
    }

    let mut report = String::from("\nlayout: content/ and LAYOUT.toml disagree.\n\n");
    for problem in &problems {
        report.push_str(&format!("  - {problem}\n"));
    }
    report.push_str(
        "\nThe manifest is the rule for which file a record goes in; see\n\
         content/FORMAT.md §0. When no row claims a record, ADD A FILE AND A ROW\n\
         rather than growing an unrelated file.\n\n  fix:  edit content/LAYOUT.toml \
         or move the record\n",
    );
    Err(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(src: &str) -> toml::Table {
        src.parse().expect("valid toml")
    }

    #[test]
    fn a_dotted_field_reaches_into_a_group() {
        // `heat.conduct` is the shape FORMAT.md §2 authors groups in, so a rule
        // has to be able to name one. Without this a manifest could only ever
        // discriminate on flat keys.
        let t = table("state = \"solid\"\nheat.conduct = 56\n");
        assert_eq!(field_str(&t, "state").as_deref(), Some("solid"));
        assert_eq!(field_str(&t, "heat.conduct").as_deref(), Some("56"));
        assert_eq!(field_str(&t, "heat.meltAt"), None);
        assert_eq!(field_str(&t, "nope"), None);
    }

    #[test]
    fn a_list_valued_field_is_not_a_discriminator() {
        // The reason most rows in LAYOUT.toml carry no rule: a mob's `bands` and
        // a structure's `biomes` are lists, and a creature that is shallow AND
        // cavern cannot be partitioned by one. Returning None here is what makes
        // such a rule fail loudly at the gate instead of quietly matching
        // nothing.
        let t = table("bands = [\"shallow\", \"cavern\"]\n");
        assert_eq!(field_str(&t, "bands"), None);
    }

    #[test]
    fn the_manifest_in_the_tree_parses_and_covers_every_kind() {
        // Not a unit test of the parser so much as a check that the committed
        // manifest is the shape the parser expects — the one file this gate
        // cannot function without.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask/ always has a parent")
            .to_path_buf();
        let (not_a_kind, kinds) = parse_manifest(&root).expect("the committed LAYOUT.toml parses");
        assert!(
            not_a_kind.contains("fonts"),
            "content/fonts/ is raw assets, and the tuning index reads this set to \
             know it is not a kind"
        );
        for kind in [
            "blocks",
            "items",
            "mobs",
            "sprites",
            "structures",
            "worldgen",
        ] {
            assert!(kinds.contains_key(kind), "no `[{kind}]` section");
        }
    }

    #[test]
    fn the_tree_satisfies_its_own_manifest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask/ always has a parent")
            .to_path_buf();
        if let Err(report) = run(&root) {
            panic!("{report}");
        }
    }
}
