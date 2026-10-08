//! Initial recall-time value of a fragment, in [`VALUE_FLOOR`, `VALUE_CEILING`].
//!
//! Port of Python v0.2.2 `wevex/value.py`: a deterministic, no-LLM scorer
//! combining a provenance prior (where the fragment came from), a type prior
//! and content heuristics. Behavioural signals (recall hits, decay) adjust
//! the stored value later; this is only the starting point.

use serde_json::{Map, Value};

pub const VALUE_FLOOR: f64 = 0.05;
pub const VALUE_CEILING: f64 = 1.0;

const PASSIVE_TOOLS: &[&str] = &["code-scanner", "scanner", "docs-watcher"];
const TRANSCRIPT_TOOLS: &[&str] = &["transcript-claude", "transcript-cursor", "transcript-codex"];
const TOOL_EVENT_VERBS: &[&str] = &[
    "Edit",
    "Write",
    "Read",
    "Bash",
    "Grep",
    "Glob",
    "Task",
    "MultiEdit",
    "NotebookEdit",
];

pub struct FragmentFacts<'a> {
    pub kind: &'a str,
    pub content: &'a str,
    pub extraction_method: &'a str,
    pub created_by_tool: Option<&'a str>,
    /// The fragment's `metadata` JSON. Anything but an object counts as empty.
    pub metadata: &'a Value,
}

pub fn compute(f: &FragmentFacts<'_>) -> f64 {
    let empty = Map::new();
    let md = f.metadata.as_object().unwrap_or(&empty);
    let value = provenance_base(f.extraction_method, f.created_by_tool, md)
        + type_adjustment(f.kind)
        + content_adjustment(f.content);
    value.clamp(VALUE_FLOOR, VALUE_CEILING)
}

fn provenance_base(extraction_method: &str, tool: Option<&str>, md: &Map<String, Value>) -> f64 {
    let em = if extraction_method.is_empty() {
        "explicit".to_string()
    } else {
        extraction_method.to_lowercase()
    };
    let tool = tool.unwrap_or("").to_lowercase();

    match md.get("promoted_via").and_then(Value::as_str) {
        Some("inbox-auto-approve") => return 0.55,
        Some("inbox-approve") => return 0.65,
        _ => {}
    }

    if em == "explicit" {
        if truthy(md.get("has_alternatives")) || truthy(md.get("structured_decision")) {
            return 0.90;
        }
        // CLI-typed writes arrive without a tool label; MCP stamps the client.
        return if matches!(tool.as_str(), "" | "cli" | "wevex-cli" | "human") {
            1.00
        } else {
            0.70
        };
    }
    if em == "code-scan" || em == "scanner" || PASSIVE_TOOLS.contains(&tool.as_str()) {
        return 0.35;
    }
    if em.starts_with("transcript") || TRANSCRIPT_TOOLS.contains(&tool.as_str()) {
        return 0.30;
    }
    if em == "tool-event" || em == "hook-observation" {
        return 0.10;
    }
    0.40
}

/// Python truthiness for a JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn type_adjustment(kind: &str) -> f64 {
    match kind {
        "decision" | "requirement" | "procedure" | "preference" => 0.10,
        "state" => -0.10,
        "observation" | "conversation" => -0.20,
        _ => 0.0,
    }
}

/// Negative-only: content can pull value down, never lift it.
fn content_adjustment(content: &str) -> f64 {
    let mut adj = 0.0;

    if is_tool_event(content) {
        adj -= 0.30;
    }

    let n = content.chars().count();
    // Very short or very long content is usually low-information.
    if !(20..=1500).contains(&n) {
        adj -= 0.05;
    }

    if n >= 40 {
        let tokens: Vec<&str> = content.split_whitespace().collect();
        if !tokens.is_empty() {
            let specific = tokens.iter().filter(|t| is_specific_token(t)).count();
            if (specific as f64) / (tokens.len() as f64) < 0.10 {
                adj -= 0.10;
            }
        }
    }
    adj
}

/// Activity-log shapes like `Edit on /path/file.py`.
fn is_tool_event(content: &str) -> bool {
    TOOL_EVENT_VERBS.iter().any(|verb| {
        content
            .strip_prefix(verb)
            .is_some_and(|rest| rest.starts_with(" on "))
    })
}

/// A token that carries information: has a path char, digit or underscore,
/// or is proper-cased (leading ASCII uppercase, a later ASCII lowercase).
fn is_specific_token(t: &str) -> bool {
    if t.chars().count() < 3 {
        return false;
    }
    if t.chars()
        .any(|c| matches!(c, '/' | '\\' | '.' | '_') || c.is_numeric())
    {
        return true;
    }
    let mut chars = t.chars();
    chars.next().is_some_and(|c| c.is_ascii_uppercase()) && chars.any(|c| c.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn value(kind: &str, content: &str, method: &str, tool: Option<&str>, md: Value) -> f64 {
        compute(&FragmentFacts {
            kind,
            content,
            extraction_method: method,
            created_by_tool: tool,
            metadata: &md,
        })
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // Expected values computed with Python v0.2.2 value.compute_fragment_value.
    #[test]
    fn matches_python_rubric() {
        let dense = "Decided to use rusqlite 0.37 for wevex-rs storage layer";
        assert!(approx(
            value("decision", dense, "explicit", None, json!({})),
            1.0
        ));
        assert!(approx(
            value("fact", dense, "explicit", Some("claude-code"), json!({})),
            0.70
        ));
        assert!(approx(
            value(
                "decision",
                dense,
                "explicit",
                Some("claude-code"),
                json!({"has_alternatives": true})
            ),
            1.0
        ));
        assert!(approx(
            value(
                "fact",
                dense,
                "explicit",
                Some("x"),
                json!({"promoted_via": "inbox-approve"})
            ),
            0.65
        ));
        assert!(approx(
            value(
                "state",
                dense,
                "docs-scanner",
                Some("docs-scanner"),
                json!({})
            ),
            0.30
        ));
        assert!(approx(
            value(
                "state",
                dense,
                "code-scanner",
                Some("code-scanner"),
                json!({})
            ),
            0.25
        ));
        assert!(approx(
            value("observation", dense, "transcript-claude", None, json!({})),
            0.10
        ));
        assert!(approx(
            value(
                "observation",
                "Edit on /Users/x/file.py",
                "tool-event",
                None,
                json!({})
            ),
            VALUE_FLOOR
        ));
        // Low-density filler loses 0.10; very short content loses 0.05.
        let filler = "we should probably think about this and maybe do it later on";
        assert!(approx(
            value("fact", filler, "explicit", None, json!({})),
            0.90
        ));
        assert!(approx(
            value("fact", "ok", "explicit", None, json!({})),
            0.95
        ));
        // Non-object metadata is treated as empty instead of failing.
        assert!(approx(
            value("fact", dense, "explicit", None, json!([1, 2])),
            1.0
        ));
    }

    #[test]
    fn specific_tokens() {
        assert!(is_specific_token("wevex.db"));
        assert!(is_specific_token("v0.2"));
        assert!(is_specific_token("Rust"));
        assert!(!is_specific_token("RUST"));
        assert!(!is_specific_token("the"));
        assert!(!is_specific_token("a1"));
    }
}
