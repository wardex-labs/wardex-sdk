//! Provider usage trees, flattened whole. Not `wardex_protocol::usage` — that
//! module owns [`crate::usage::TokenUsage`], the NORMALIZED totals; this one
//! owns the raw pass-through mirror (`wardex.usage.*`): every scalar leaf of
//! the provider's `usage` object, provider spelling preserved, bounded by
//! `max_extra_keys`.
//!
//! The two coexist because their consumers differ. The normalized fields are
//! the semconv contract a dashboard renders; the mirror is what a cost
//! calculator reads without knowing the mapping — and what keeps a NEW
//! provider counter (a cache tier, a search charge, a thinking tier) from
//! being silently discarded by a typed struct that predates it.

/// One scalar leaf of a provider usage object, exactly as reported.
#[derive(Debug, Clone, PartialEq)]
pub enum UsageLeaf {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

/// Bounds on the flattened family. `max_leaves` is `limits.max_extra_keys`;
/// the other two are structural sanity bounds no real provider usage object
/// approaches (defense against pathological bodies) — drops from all three
/// land in the same counter, so the "emitted + dropped == leaves" arithmetic
/// stays whole.
#[derive(Debug, Clone, Copy)]
pub struct UsageBounds {
    pub max_leaves: usize,
    pub max_depth: usize,
    pub max_path_bytes: usize,
}

impl UsageBounds {
    pub fn from_limits(limits: &wardex_limits::Limits) -> Self {
        Self {
            max_leaves: limits.max_extra_keys,
            max_depth: 6,
            max_path_bytes: 120,
        }
    }
}

/// The flattening result: surviving leaves plus how many were dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Flattened {
    pub leaves: Vec<(String, UsageLeaf)>,
    pub dropped: u32,
}

/// Every scalar leaf of `usage`, as `(dotted path, value)`.
///
/// Leaves are SORTED by path BEFORE the bounds apply, so both the output
/// order and which leaves survive the cap are properties of this code rather
/// than of `serde_json`'s map order — a workspace dependency enabling
/// `preserve_order` flips `Map` to an IndexMap, and iteration order with it.
/// `null` is skipped (a non-value, not a drop). Arrays flatten through index
/// segments (`.0`).
pub fn flatten_usage(usage: &serde_json::Value, b: UsageBounds) -> Flattened {
    let mut all: Vec<(String, UsageLeaf)> = Vec::new();
    let mut structural_dropped: u32 = 0;
    walk(
        usage,
        String::new(),
        0,
        &b,
        &mut all,
        &mut structural_dropped,
    );
    all.sort_by(|a, z| a.0.cmp(&z.0));
    let mut dropped = structural_dropped;
    if all.len() > b.max_leaves {
        dropped += (all.len() - b.max_leaves) as u32;
        all.truncate(b.max_leaves);
    }
    Flattened {
        leaves: all,
        dropped,
    }
}

fn walk(
    v: &serde_json::Value,
    path: String,
    depth: usize,
    b: &UsageBounds,
    out: &mut Vec<(String, UsageLeaf)>,
    dropped: &mut u32,
) {
    use serde_json::Value;
    let leaf = match v {
        Value::Null => return, // not a value, not a drop
        Value::Object(m) => {
            for (k, child) in m {
                let child_path = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                walk(child, child_path, depth + 1, b, out, dropped);
            }
            return;
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                let child_path = if path.is_empty() {
                    i.to_string()
                } else {
                    format!("{path}.{i}")
                };
                walk(child, child_path, depth + 1, b, out, dropped);
            }
            return;
        }
        Value::Bool(x) => UsageLeaf::Bool(*x),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                UsageLeaf::Int(i)
            } else if let Some(f) = n.as_f64() {
                UsageLeaf::Float(f)
            } else {
                return;
            }
        }
        Value::String(s) => UsageLeaf::Str(s.clone()),
    };
    // A scalar at the ROOT has no path; a usage object is an object, so this
    // only happens for a non-object `usage` value — nothing to mirror.
    if path.is_empty() {
        return;
    }
    if depth > b.max_depth || path.len() > b.max_path_bytes {
        *dropped += 1;
        return;
    }
    out.push((path, leaf));
}

/// Streaming Anthropic: `message_start.usage` <- each `message_delta.usage`.
/// Key-wise overwrite, recursing into nested objects — Anthropic documents the
/// delta values as cumulative, so the last delta is authoritative.
pub fn deep_merge(base: &mut serde_json::Value, delta: &serde_json::Value) {
    if let (serde_json::Value::Object(b), serde_json::Value::Object(d)) = (base, delta) {
        for (k, dv) in d {
            match b.get_mut(k) {
                Some(bv) if bv.is_object() && dv.is_object() => deep_merge(bv, dv),
                _ => {
                    b.insert(k.clone(), dv.clone());
                }
            }
        }
    }
}

/// Dotted-path reads over the raw usage `Value` — the normalization helper
/// each provider `fill()` uses with its `NORMALIZED_USAGE_PATHS` table, so a
/// path spelled in the table and the path the extraction reads are one string.
pub struct UsageView<'a>(pub &'a serde_json::Value);

impl UsageView<'_> {
    fn at(&self, path: &str) -> Option<&serde_json::Value> {
        let mut cur = self.0;
        for seg in path.split('.') {
            cur = cur.get(seg)?;
        }
        Some(cur)
    }

    pub fn i64_at(&self, path: &str) -> Option<i64> {
        self.at(path).and_then(serde_json::Value::as_i64)
    }

    pub fn str_at(&self, path: &str) -> Option<&str> {
        self.at(path).and_then(serde_json::Value::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> UsageBounds {
        UsageBounds {
            max_leaves: 64,
            max_depth: 6,
            max_path_bytes: 120,
        }
    }

    fn count_scalar_leaves(v: &serde_json::Value) -> usize {
        // The independent walker G4 checks the flattener against: counts every
        // non-null scalar under a non-empty path, ignoring all bounds.
        fn go(v: &serde_json::Value, root: bool) -> usize {
            match v {
                serde_json::Value::Null => 0,
                serde_json::Value::Object(m) => m.values().map(|c| go(c, false)).sum(),
                serde_json::Value::Array(a) => a.iter().map(|c| go(c, false)).sum(),
                _ => usize::from(!root),
            }
        }
        go(v, true)
    }

    /// G4 — emitted + dropped == the tree's scalar-leaf count, on realistic
    /// and synthetic shapes alike.
    #[test]
    fn usage_flatten_is_total_over_leaves() {
        let cases = [
            serde_json::json!({"prompt_tokens": 12, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 1}}),
            serde_json::json!({"input_tokens": 1, "output_tokens": 2,
                "cache_read_input_tokens": 3, "cache_creation_input_tokens": 4,
                "cache_creation": {"ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 0},
                "server_tool_use": {"web_search_requests": 2},
                "output_tokens_details": {"thinking_tokens": 7},
                "service_tier": "standard", "inference_geo": "us"}),
            serde_json::json!({"input_tokens": 10, "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 2}, "total_tokens": 15}),
            serde_json::json!({"a": {"b": {"c": [1, 2, {"d": true}]}}, "e": 0.5}),
            serde_json::json!({"nothing": null, "some": 1}),
        ];
        for usage in &cases {
            let f = flatten_usage(usage, bounds());
            assert_eq!(
                f.leaves.len() + f.dropped as usize,
                count_scalar_leaves(usage),
                "not total over {usage}"
            );
        }
    }

    /// T-R7 — determinism is a property of the explicit sort, not of map
    /// order: the same tree with keys inserted in reverse yields the same
    /// output, byte for byte.
    #[test]
    fn usage_flatten_is_deterministic_under_key_order() {
        let a: serde_json::Value =
            serde_json::from_str(r#"{"a": 1, "b": {"c": 2, "d": 3}, "e": "x"}"#).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(r#"{"e": "x", "b": {"d": 3, "c": 2}, "a": 1}"#).unwrap();
        assert_eq!(flatten_usage(&a, bounds()), flatten_usage(&b, bounds()));
        let flat = flatten_usage(&a, bounds());
        let paths: Vec<&str> = flat.leaves.iter().map(|(p, _)| p.as_str()).collect();
        let mut sorted = paths.clone();
        sorted.sort_unstable();
        assert_eq!(paths, sorted, "output is path-sorted");
    }

    /// T-R7 — the cap victim selection is deterministic too (sorted first,
    /// truncated after), and the count is exact.
    #[test]
    fn usage_flatten_caps_and_counts() {
        let usage = serde_json::json!({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5});
        let f = flatten_usage(
            &usage,
            UsageBounds {
                max_leaves: 3,
                ..bounds()
            },
        );
        assert_eq!(f.dropped, 2);
        let paths: Vec<&str> = f.leaves.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["a", "b", "c"], "survivors are the sort's head");
    }

    #[test]
    fn usage_flatten_skips_null_and_indexes_arrays() {
        let usage = serde_json::json!({"z": null, "arr": [7, null, {"k": "v"}]});
        let f = flatten_usage(&usage, bounds());
        assert_eq!(f.dropped, 0, "null is a non-value, not a drop");
        let got: Vec<(String, UsageLeaf)> = f.leaves;
        assert_eq!(
            got,
            vec![
                ("arr.0".to_string(), UsageLeaf::Int(7)),
                ("arr.2.k".to_string(), UsageLeaf::Str("v".to_string())),
            ]
        );
    }

    /// §3.5 — the two structural sanity bounds drop INTO the same counter, so
    /// the totality arithmetic holds whichever bound fired.
    #[test]
    fn usage_flatten_counts_structural_drops() {
        // depth 7 leaf
        let deep = serde_json::json!(
            {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}, "ok": 2}
        );
        let f = flatten_usage(&deep, bounds());
        assert_eq!(f.dropped, 1);
        assert_eq!(f.leaves.len(), 1);
        assert_eq!(f.leaves[0].0, "ok");
        // path over 120 bytes
        let long_key = "k".repeat(121);
        let long = serde_json::json!({long_key: 1, "ok": 2});
        let f = flatten_usage(&long, bounds());
        assert_eq!(f.dropped, 1);
        assert_eq!(f.leaves.len(), 1);
    }

    #[test]
    fn deep_merge_overwrites_by_key_and_recurses() {
        let mut base = serde_json::json!({"input_tokens": 3, "output_tokens": 1,
            "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 0}});
        let delta = serde_json::json!({"output_tokens": 55,
            "cache_creation": {"ephemeral_1h_input_tokens": 7},
            "server_tool_use": {"web_search_requests": 1}});
        deep_merge(&mut base, &delta);
        assert_eq!(
            base,
            serde_json::json!({"input_tokens": 3, "output_tokens": 55,
                "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 7},
                "server_tool_use": {"web_search_requests": 1}})
        );
    }

    #[test]
    fn usage_view_reads_dotted_paths() {
        let u = serde_json::json!({"a": {"b": 7}, "tier": "flex"});
        let v = UsageView(&u);
        assert_eq!(v.i64_at("a.b"), Some(7));
        assert_eq!(v.str_at("tier"), Some("flex"));
        assert_eq!(v.i64_at("a.missing"), None);
        assert_eq!(v.i64_at("tier"), None, "type mismatch is None, not a panic");
    }
}
