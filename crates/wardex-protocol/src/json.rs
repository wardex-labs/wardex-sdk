//! Field reads over a parsed `serde_json::Value`, shared by the line-oriented
//! parsers (`claude_stream_json`, `codex_exec_json`) that read one key at a
//! time and treat a missing or mistyped key as absent.

use serde_json::Value;

/// The string at `key`, owned; `None` when the key is absent or not a string.
pub(crate) fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// The integer at `key`; `None` when the key is absent or not an integer that
/// fits in `i64`.
pub(crate) fn i64_field(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(Value::as_i64)
}
