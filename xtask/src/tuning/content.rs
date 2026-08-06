//! Tier 1 — `content/`, summarised.
//!
//! The other two tiers are indexed constant by constant because a constant is
//! the unit you go looking for. A content record is not: nobody greps for the
//! hardness of gravel, they open `content/blocks/` and read the file, which is
//! written to be read. What a person cannot get from the files is the SHAPE of
//! the tier — how many kinds there are, how big each one is, and how wide its
//! schema has grown — so that is what this counts.
//!
//! It parses with the same `toml` crate `contentc` uses, so "53 records" here is
//! the number the compiler emits and not a guess from counting `[` characters.

use std::collections::BTreeSet;
use std::path::Path;

/// One directory under `content/`.
pub struct Kind {
    /// Directory name, e.g. `blocks`.
    pub dir: String,
    pub files: usize,
    pub records: usize,
    /// Distinct dotted field paths across every record in the directory.
    ///
    /// Dotted and flattened: `heat.conduct` counts once however many records
    /// carry it, and `art.seq.frames` counts once however long the array is.
    /// This is the width of the kind's schema as authored, which is the number
    /// that says whether a kind has quietly grown a second job.
    pub fields: usize,
}

pub fn scan(content_dir: &Path) -> Result<Vec<Kind>, String> {
    let mut dirs: Vec<_> = std::fs::read_dir(content_dir)
        .map_err(|e| format!("content/: {e}"))?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    dirs.sort();

    let mut kinds = Vec::new();
    for dir in dirs {
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        files.sort();

        let mut records = 0usize;
        let mut fields = BTreeSet::new();
        for file in &files {
            let text =
                std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
            let table: toml::Table = text
                .parse()
                .map_err(|e| format!("{}: {e}", file.display()))?;
            for value in table.values() {
                match value {
                    // `[gravel]` — one record.
                    toml::Value::Table(t) => {
                        records += 1;
                        walk(t, "", &mut fields);
                    }
                    // `[[icons]]` — a record each.
                    toml::Value::Array(items) => {
                        for item in items {
                            if let toml::Value::Table(t) = item {
                                records += 1;
                                walk(t, "", &mut fields);
                            }
                        }
                    }
                    // A bare top-level key is file metadata, not a record.
                    _ => {}
                }
            }
        }

        kinds.push(Kind {
            dir: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            files: files.len(),
            records,
            fields: fields.len(),
        });
    }
    Ok(kinds)
}

fn walk(table: &toml::Table, prefix: &str, out: &mut BTreeSet<String>) {
    for (key, value) in table {
        let path = format!("{prefix}{key}");
        out.insert(path.clone());
        match value {
            toml::Value::Table(t) => walk(t, &format!("{path}."), out),
            toml::Value::Array(items) => {
                for item in items {
                    if let toml::Value::Table(t) = item {
                        walk(t, &format!("{path}."), out);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(toml_text: &str) -> (usize, Vec<String>) {
        let table: toml::Table = toml_text.parse().expect("parses");
        let mut records = 0;
        let mut fields = BTreeSet::new();
        for value in table.values() {
            match value {
                toml::Value::Table(t) => {
                    records += 1;
                    walk(t, "", &mut fields);
                }
                toml::Value::Array(items) => {
                    for item in items {
                        if let toml::Value::Table(t) = item {
                            records += 1;
                            walk(t, "", &mut fields);
                        }
                    }
                }
                _ => {}
            }
        }
        (records, fields.into_iter().collect())
    }

    #[test]
    fn each_top_level_table_is_one_record() {
        let (records, _) = counts("[gravel]\nname = \"Gravel\"\n\n[clay]\nname = \"Clay\"\n");
        assert_eq!(records, 2);
    }

    #[test]
    fn a_nested_table_is_a_dotted_field_and_not_a_second_record() {
        let (records, fields) =
            counts("[gravel]\nname = \"Gravel\"\n[gravel.heat]\nconduct = 50\n");
        assert_eq!(records, 1);
        assert_eq!(fields, ["heat", "heat.conduct", "name"]);
    }

    #[test]
    fn an_array_of_tables_inside_a_record_flattens_to_one_field_path() {
        let (records, fields) = counts(
            "[bloat]\nname = \"Bloat\"\n[[bloat.art.seq]]\nframes = 2\n[[bloat.art.seq]]\nframes = 3\n",
        );
        assert_eq!(records, 1);
        assert_eq!(fields, ["art", "art.seq", "art.seq.frames", "name"]);
    }

    #[test]
    fn the_same_field_in_two_records_is_counted_once() {
        let (_, fields) = counts("[a]\nhardness = 1.0\n\n[b]\nhardness = 2.0\n");
        assert_eq!(fields, ["hardness"]);
    }
}
