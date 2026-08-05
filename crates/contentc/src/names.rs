//! Identifier conversion between the content vocabulary and Rust's.
//!
//! Content ids are authored in camelCase or snake_case (`goldOre`,
//! `pick_copper`). Rust wants `GOLD_ORE` for a constant, `GoldOre` for an enum
//! variant and `gold_ore` for a struct field. Doing that conversion in one place
//! means a collision — two distinct ids mapping to one Rust name — is detectable
//! rather than a confusing duplicate-definition error out of rustc.

/// `goldOre` / `gold_ore` / `GoldOre` -> `["gold", "ore"]`.
fn words(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        if c == '_' || c == '-' || c == ' ' || c == '.' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        // A capital starts a new word, except inside a run of capitals that is
        // not immediately followed by a lowercase letter — so `RGBColor` splits
        // as `RGB` + `Color`, not `R` + `G` + `B` + `Color`.
        let starts_word = c.is_ascii_uppercase()
            && !cur.is_empty()
            && (chars[i - 1].is_ascii_lowercase()
                || chars[i - 1].is_ascii_digit()
                || chars.get(i + 1).is_some_and(char::is_ascii_lowercase));
        if starts_word {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// `goldOre` -> `GOLD_ORE`. For generated constants.
pub fn screaming_snake(s: &str) -> String {
    let w = words(s);
    if w.is_empty() {
        return "_".to_string();
    }
    let joined = w.join("_").to_ascii_uppercase();
    if joined.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{joined}")
    } else {
        joined
    }
}

/// `goldOre` -> `gold_ore`. For generated struct fields.
pub fn snake(s: &str) -> String {
    let w = words(s);
    if w.is_empty() {
        return "_".to_string();
    }
    let joined = w.join("_").to_ascii_lowercase();
    if joined.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{joined}")
    } else {
        escape_keyword(&joined)
    }
}

/// `gold_ore` -> `GoldOre`. For generated enum variants and struct names.
pub fn pascal(s: &str) -> String {
    let mut out = String::new();
    for w in words(s) {
        let mut cs = w.chars();
        if let Some(c) = cs.next() {
            out.extend(c.to_uppercase());
            out.push_str(&cs.as_str().to_ascii_lowercase());
        }
    }
    if out.is_empty() {
        return "_".to_string();
    }
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{out}")
    } else {
        out
    }
}

/// Rust keywords a content id could plausibly collide with as a field name.
fn escape_keyword(s: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false",
        "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
        "ref", "return", "self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "async", "await", "box", "final", "macro", "override", "priv",
        "try", "typeof", "unsized", "virtual", "yield", "abstract", "become", "do",
    ];
    if KEYWORDS.contains(&s) {
        format!("r#{s}")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_camel_and_snake_alike() {
        assert_eq!(screaming_snake("goldOre"), "GOLD_ORE");
        assert_eq!(screaming_snake("pick_copper"), "PICK_COPPER");
        assert_eq!(screaming_snake("stone"), "STONE");
        assert_eq!(screaming_snake("_reserved3"), "RESERVED3");
    }

    #[test]
    fn keeps_capital_runs_together() {
        assert_eq!(pascal("RGBColor"), "RgbColor");
        assert_eq!(screaming_snake("maxHP"), "MAX_HP");
    }

    #[test]
    fn field_names_avoid_keywords() {
        assert_eq!(snake("type"), "r#type");
        assert_eq!(snake("colorVar"), "color_var");
        assert_eq!(snake("liquidSpread"), "liquid_spread");
    }

    #[test]
    fn variants_are_pascal() {
        assert_eq!(pascal("bottomCenter"), "BottomCenter");
        assert_eq!(pascal("flat"), "Flat");
    }
}
