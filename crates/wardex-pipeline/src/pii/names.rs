//! Argument-name rules: a value is a secret because of the NAME it sits under.
//!
//! The value rules in `patterns.rs` recognise a credential by its shape
//! (`sk-…`, `AKIA…`). Most credentials have no shape — `appid=4f1c…`,
//! `"password": "hunter2"` — and are secrets only because of the argument they
//! are passed as. This module judges the name and returns the byte range of the
//! value to replace; the engine merges those ranges with the value rules' hits.
//!
//! One rule, every text field, every transport. The scanner does not know what
//! protocol a text came from; it recognises two text shapes wherever they
//! occur:
//!
//! * `name=value` — a URL query, a form body, a `.env` line, a WebSocket
//!   upgrade target;
//! * `"name": value` — JSON, including JSON escaped inside a JSON string (a
//!   tool call's `arguments`), and `'name': value` (a Python dict's repr).
//!
//! The judgement is by WORDS, not substrings: a name is split at `_ - .`,
//! spaces, brackets, `%XX` escapes and camelCase boundaries, then lowercased.
//! That is what keeps `keyword` and `monkey` out while `apiKey`, `X-Api-Key`
//! and `API_KEY` all read as the same two words. The lists below are the whole
//! rule; the SDK documentation prints them verbatim.

use std::sync::OnceLock;

use wardex_codec::proto::wardex::v1::RedactionRule as Rule;

/// Rule 1: a strong secret word ANYWHERE in the name. These words are never a
/// debugging value, so their position does not matter (`password_confirmation`
/// carries a password).
pub const STRONG_WORDS: &[&str] = &[
    "password",
    "passwd",
    "pwd",
    "passphrase",
    "secret",
    "credential",
    "credentials",
    "jwt",
    "bearer",
    "signature",
    "authorization",
    "cookie",
];

/// Rule 2: the name's LAST word, when the name has two or more words. `key`
/// and `token` name many harmless things when they are not the head noun
/// (`key_id`, `token_type`); as the head noun they name the credential
/// (`api_key`, `access_token`). Singular only: `max_tokens` is a count.
pub const LAST_WORDS: &[&str] = &["key", "token"];

/// Rule 3: whole names, compared word for word. Mostly names written as one
/// word, which the word split cannot see inside (`apikey`, `appid`), plus a
/// few whose words are harmless on their own.
pub const EXACT_NAMES: &[&str] = &[
    "token",
    "auth",
    "apikey",
    "apitoken",
    "hapikey",
    "appid",
    "accesstoken",
    "authtoken",
    "privatetoken",
    "accesskey",
    "secretkey",
    "privatekey",
    "clientsecret",
    "apisecret",
    "sessionid",
    "jsessionid",
    "phpsessid",
    "csrf",
    "xsrf",
    "csrfmiddlewaretoken",
    "SAMLResponse",
    "code_verifier",
];

/// Rule 4: whole names that are credentials only in the `name=value` shape.
/// OAuth sends `code` and Azure SAS sends `sig` as URL or form arguments, and
/// Google's `key` is a query argument; in JSON the same names are an error
/// code (`"code": -32601`), a code interpreter's source, or a map entry's key,
/// all of which a debugger needs.
pub const URL_FORM_ONLY_NAMES: &[&str] = &["code", "sig", "key"];

/// What every name-rule hit writes in place of the value.
pub const PLACEHOLDER: &str = "[SECRET]";

/// Longest name the scanner will read backwards. A name longer than this is
/// not an argument name, and the bound keeps the scan linear.
const MAX_NAME_LEN: usize = 128;

/// Which text shape a name was found in. Rule 4 depends on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// `name=value`
    UrlForm,
    /// `"name": value`, or a structured attribute's key
    Json,
}

/// One value the name rules want replaced.
pub struct NameHit {
    pub start: usize,
    pub end: usize,
    pub replacement: String,
    pub rule: Rule,
    pub name: String,
}

/// The name rules with an application's additions and exemptions applied.
#[derive(Debug, Default)]
pub struct NameRules {
    extra: Vec<Vec<String>>,
    reveal: Vec<Vec<String>>,
}

fn exact_names() -> &'static [Vec<String>] {
    static EXACT: OnceLock<Vec<Vec<String>>> = OnceLock::new();
    EXACT.get_or_init(|| EXACT_NAMES.iter().map(|n| words(n)).collect())
}

impl NameRules {
    /// `extra` names are secrets too; `reveal` names are exempt from every
    /// name rule (the value rules still apply). Both compare by words, so
    /// `page_token` also covers `pageToken` and `Page-Token`. A name that
    /// splits into no words cannot match anything and is an error, as is a
    /// name listed in both — the caller validates, this only normalises.
    pub fn new(extra: &[String], reveal: &[String]) -> NameRules {
        NameRules {
            extra: extra.iter().map(|n| words(n)).collect(),
            reveal: reveal.iter().map(|n| words(n)).collect(),
        }
    }

    /// The rule a value under `name` falls to, if any.
    pub fn judge(&self, name: &str, shape: Shape) -> Option<Rule> {
        let w = words(name);
        if w.is_empty() || self.reveal.contains(&w) {
            return None;
        }
        if self.extra.contains(&w) {
            return Some(Rule::SecretUserName);
        }
        if exact_names().contains(&w) {
            return Some(Rule::SecretExactName);
        }
        if shape == Shape::UrlForm && w.len() == 1 && URL_FORM_ONLY_NAMES.contains(&w[0].as_str()) {
            return Some(Rule::SecretExactName);
        }
        if w.iter().any(|x| STRONG_WORDS.contains(&x.as_str())) {
            return Some(Rule::SecretWord);
        }
        if w.len() >= 2 && LAST_WORDS.contains(&w[w.len() - 1].as_str()) {
            return Some(Rule::SecretLastWord);
        }
        None
    }

    /// Every value in `text` that sits under a secret name.
    pub fn scan(&self, text: &str, hits: &mut Vec<NameHit>) {
        for (i, &c) in text.as_bytes().iter().enumerate() {
            match c {
                b'=' => self.scan_form(text, i, hits),
                b':' => self.scan_json(text, i, hits),
                _ => {}
            }
        }
    }

    fn scan_form(&self, text: &str, eq: usize, hits: &mut Vec<NameHit>) {
        let b = text.as_bytes();
        let mut start = eq;
        while start > 0 && is_form_name_byte(b[start - 1]) {
            start -= 1;
            if eq - start > MAX_NAME_LEN {
                return;
            }
        }
        if start == eq {
            return;
        }
        let name = &text[start..eq];
        let Some(rule) = self.judge(name, Shape::UrlForm) else {
            return;
        };
        let mut end = eq + 1;
        while end < b.len() && !is_form_value_end(b[end]) {
            end += 1;
        }
        let value = &text[eq + 1..end];
        if value.is_empty() || value == PLACEHOLDER {
            return;
        }
        hits.push(NameHit {
            start: eq + 1,
            end,
            replacement: PLACEHOLDER.to_string(),
            rule,
            name: name.to_string(),
        });
    }

    fn scan_json(&self, text: &str, colon: usize, hits: &mut Vec<NameHit>) {
        let b = text.as_bytes();
        // The name's closing quote, skipping whitespace before the colon.
        let mut k = colon;
        while k > 0 && b[k - 1].is_ascii_whitespace() {
            k -= 1;
        }
        if k == 0 {
            return;
        }
        let close = k - 1;
        let quote = b[close];
        if quote != b'"' && quote != b'\'' {
            return;
        }
        // How deeply the JSON is escaped: 0 for plain JSON, 1 for JSON inside
        // a JSON string, 3 for one level further. A single quote is only ever
        // plain (a Python repr).
        let level = backslashes_before(b, close);
        if !matches!(level, 0 | 1 | 3 | 7) || (quote == b'\'' && level != 0) {
            return;
        }
        let name_end = close - level;
        let mut p = name_end;
        while p > 0 && b[p - 1] != quote && b[p - 1] != b'\\' {
            p -= 1;
            if name_end - p > MAX_NAME_LEN {
                return;
            }
        }
        if p == 0 || b[p - 1] != quote || p == name_end {
            return;
        }
        let open = p - 1;
        if backslashes_before(b, open) != level {
            return;
        }
        let name = &text[p..name_end];
        let Some(rule) = self.judge(name, Shape::Json) else {
            return;
        };
        let mut v = colon + 1;
        while v < b.len() && b[v].is_ascii_whitespace() {
            v += 1;
        }
        if v < b.len() && b[v] == b'[' {
            let mut at = v + 1;
            loop {
                while at < b.len() && (b[at].is_ascii_whitespace() || b[at] == b',') {
                    at += 1;
                }
                match scalar_at(b, at, quote, level) {
                    Some((s, e, next)) => {
                        push_value(text, s, e, quote, level, rule, name, hits);
                        at = next;
                    }
                    None => break,
                }
            }
        } else if let Some((s, e, _)) = scalar_at(b, v, quote, level) {
            push_value(text, s, e, quote, level, rule, name, hits);
        }
    }
}

/// A JSON scalar starting at `at`: `(start, end, next)` where `start..end` is
/// the range to replace (a string's content, or a number's digits) and `next`
/// is the first byte after it. `None` for anything else — an object is not
/// replaced, its own names are judged when the scan reaches them.
fn scalar_at(b: &[u8], at: usize, quote: u8, level: usize) -> Option<(usize, usize, usize)> {
    if at >= b.len() {
        return None;
    }
    let first = b[at];
    if first == b'-' || first.is_ascii_digit() {
        let mut e = at + 1;
        while e < b.len() && matches!(b[e], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
            e += 1;
        }
        return Some((at, e, e));
    }
    // A string opens with `level` backslashes and the quote.
    if at + level >= b.len() || b[at..at + level].iter().any(|&c| c != b'\\') {
        return None;
    }
    if b[at + level] != quote {
        return None;
    }
    let start = at + level + 1;
    let modulus = 2 * (level + 1);
    let mut i = start;
    while i < b.len() {
        if b[i] == quote {
            let run = backslashes_before(b, i);
            // At escape depth `level` a closing quote carries exactly `level`
            // backslashes modulo the next depth's escaping; any other run is
            // a quote INSIDE the string.
            if run % modulus == level && i >= start + level {
                return Some((start, i - level, i + 1));
            }
        }
        i += 1;
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn push_value(
    text: &str,
    start: usize,
    end: usize,
    quote: u8,
    level: usize,
    rule: Rule,
    name: &str,
    hits: &mut Vec<NameHit>,
) {
    let value = &text[start..end];
    if value.is_empty() || value == PLACEHOLDER {
        return;
    }
    let is_string = start > 0 && text.as_bytes()[start - 1] == quote;
    let replacement = if is_string {
        PLACEHOLDER.to_string()
    } else {
        // A number becomes a string: the placeholder is text, and the
        // document must stay parseable at its own escape depth.
        let esc = "\\".repeat(level);
        let q = quote as char;
        format!("{esc}{q}{PLACEHOLDER}{esc}{q}")
    };
    hits.push(NameHit {
        start,
        end,
        replacement,
        rule,
        name: name.to_string(),
    });
}

fn backslashes_before(b: &[u8], pos: usize) -> usize {
    let mut n = 0;
    while n < pos && b[pos - 1 - n] == b'\\' {
        n += 1;
    }
    n
}

/// Bytes a `name=` may be made of. Anything else ends the name scanning
/// backwards, which is also what rejects `a != b` and `x == y`.
fn is_form_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b'[' | b']' | b'%')
}

/// Bytes that end a `name=value` value.
fn is_form_value_end(c: u8) -> bool {
    c.is_ascii_whitespace()
        || matches!(
            c,
            b'&' | b'#' | b';' | b'"' | b'\'' | b'\\' | b'<' | b'>' | b')' | b'}'
        )
}

/// A name's words: split at every non-alphanumeric byte, at `%XX` escapes and
/// at camelCase boundaries (`apiKey` → api·key, `AWSAccessKeyId` →
/// aws·access·key·id), lowercased. Digits stay with their letters
/// (`oauth2Token` → oauth2·token). Non-ASCII letters are kept as word bytes
/// and never split on case.
pub fn words(name: &str) -> Vec<String> {
    let chars: Vec<char> = name.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '%'
            && i + 2 < chars.len()
            && chars[i + 1].is_ascii_hexdigit()
            && chars[i + 2].is_ascii_hexdigit()
        {
            flush(&mut cur, &mut out);
            i += 3;
            continue;
        }
        if !(c.is_alphanumeric()) {
            flush(&mut cur, &mut out);
            i += 1;
            continue;
        }
        if c.is_ascii_uppercase() && !cur.is_empty() {
            let prev = chars[i - 1];
            let next_lower = chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
            if prev.is_ascii_lowercase()
                || prev.is_ascii_digit()
                || (prev.is_ascii_uppercase() && next_lower)
            {
                flush(&mut cur, &mut out);
            }
        }
        cur.extend(c.to_lowercase());
        i += 1;
    }
    flush(&mut cur, &mut out);
    out
}

fn flush(cur: &mut String, out: &mut Vec<String>) {
    if !cur.is_empty() {
        out.push(std::mem::take(cur));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> NameRules {
        NameRules::default()
    }

    fn mask(text: &str) -> String {
        let mut hits = Vec::new();
        rules().scan(text, &mut hits);
        let mut out = text.to_string();
        hits.sort_by_key(|h| std::cmp::Reverse(h.start));
        for h in hits {
            out.replace_range(h.start..h.end, &h.replacement);
        }
        out
    }

    #[test]
    fn words_split_at_separators_and_case() {
        assert_eq!(words("api_key"), ["api", "key"]);
        assert_eq!(words("apiKey"), ["api", "key"]);
        assert_eq!(words("X-Api-Key"), ["x", "api", "key"]);
        assert_eq!(words("api.key"), ["api", "key"]);
        assert_eq!(words("API_KEY"), ["api", "key"]);
        assert_eq!(words("AWSAccessKeyId"), ["aws", "access", "key", "id"]);
        assert_eq!(words("oauth2Token"), ["oauth2", "token"]);
        assert_eq!(words("api%5Fkey"), ["api", "key"]);
        assert_eq!(words("auth[token]"), ["auth", "token"]);
        assert_eq!(words("keyword"), ["keyword"]);
        assert!(words("__").is_empty());
    }

    #[test]
    fn judged_by_words_not_substrings() {
        let r = rules();
        for kept in [
            "keyword",
            "monkey",
            "inside",
            "country_code",
            "key_id",
            "token_type",
            "max_tokens",
            "session_id",
            "sessionId",
            "app_id",
            "q",
            "query",
            "api_keys",
        ] {
            assert_eq!(r.judge(kept, Shape::UrlForm), None, "{kept}");
            assert_eq!(r.judge(kept, Shape::Json), None, "{kept}");
        }
        for (name, rule) in [
            ("apiKey", Rule::SecretLastWord),
            ("X-Api-Key", Rule::SecretLastWord),
            ("api.key", Rule::SecretLastWord),
            ("API_KEY", Rule::SecretLastWord),
            ("page_token", Rule::SecretLastWord),
            ("idempotency_key", Rule::SecretLastWord),
            ("SAMLResponse", Rule::SecretExactName),
            ("password_confirmation", Rule::SecretWord),
            ("X-Amz-Signature", Rule::SecretWord),
            ("sessionid", Rule::SecretExactName),
            ("appid", Rule::SecretExactName),
            ("token", Rule::SecretExactName),
        ] {
            assert_eq!(r.judge(name, Shape::Json), Some(rule), "{name}");
        }
    }

    #[test]
    fn url_form_only_names_are_kept_in_json() {
        let r = rules();
        for name in ["code", "sig", "key"] {
            assert_eq!(r.judge(name, Shape::UrlForm), Some(Rule::SecretExactName));
            assert_eq!(r.judge(name, Shape::Json), None);
        }
        assert_eq!(
            mask("/cb?code=abc123&state=xyz"),
            "/cb?code=[SECRET]&state=xyz"
        );
        assert_eq!(
            mask(r#"{"code": -32601, "message": "x"}"#),
            r#"{"code": -32601, "message": "x"}"#
        );
        assert_eq!(
            mask(r#"{"error":{"code":"rate_limit_exceeded"}}"#),
            r#"{"error":{"code":"rate_limit_exceeded"}}"#
        );
    }

    #[test]
    fn masks_form_and_json_values() {
        assert_eq!(
            mask("GET /tool?q=seoul&api_key=SECRET123&page=2"),
            "GET /tool?q=seoul&api_key=[SECRET]&page=2"
        );
        assert_eq!(
            mask(
                r#"{"query":"seoul weather","limit":5,"api_key":"abc123xyz","appid":"owm999key"}"#
            ),
            r#"{"query":"seoul weather","limit":5,"api_key":"[SECRET]","appid":"[SECRET]"}"#
        );
        assert_eq!(
            mask(r#"{"pin_token": 1234}"#),
            r#"{"pin_token": "[SECRET]"}"#
        );
        assert_eq!(
            mask(r#"{"api_keys_token": ["a1", "b2"]}"#),
            r#"{"api_keys_token": ["[SECRET]", "[SECRET]"]}"#
        );
        assert_eq!(mask("{'password': 'hunter2'}"), "{'password': '[SECRET]'}");
        assert_eq!(mask("API_KEY=abc\nDEBUG=1"), "API_KEY=[SECRET]\nDEBUG=1");
    }

    #[test]
    fn masks_json_escaped_inside_a_json_string() {
        let text = r#"{"arguments":"{\"city\":\"seoul\",\"api_key\":\"abc\\\"x\"}"}"#;
        assert_eq!(
            mask(text),
            r#"{"arguments":"{\"city\":\"seoul\",\"api_key\":\"[SECRET]\"}"}"#
        );
        let num = r#"{"arguments":"{\"pin_token\":42}"}"#;
        assert_eq!(mask(num), r#"{"arguments":"{\"pin_token\":\"[SECRET]\"}"}"#);
    }

    #[test]
    fn placeholders_are_not_masked_twice() {
        let once = mask("a?api_key=x&b=1");
        let mut hits = Vec::new();
        rules().scan(&once, &mut hits);
        assert!(hits.is_empty());
    }

    #[test]
    fn expressions_are_not_names() {
        for text in ["if token == x", "a != b", "x>=y", "token = 5"] {
            let mut hits = Vec::new();
            rules().scan(text, &mut hits);
            assert!(hits.is_empty(), "{text}");
        }
    }

    /// The README's "what masking does not catch" list, pinned: if one of
    /// these starts being masked, the documentation is what has to change.
    #[test]
    fn documented_limits_are_real() {
        for text in [
            r#"{"value": "hunter2"}"#,
            "api_key: abc123",
            r#"{"api_keys": ["abc123"]}"#,
            "/bot123456:ABCdef/sendMessage",
            r#"{"code": "4/0AX4XfWh"}"#,
        ] {
            let mut hits = Vec::new();
            rules().scan(text, &mut hits);
            assert!(hits.is_empty(), "{text}");
        }
    }

    #[test]
    fn user_names_extend_and_reveal() {
        let r = NameRules::new(
            &["x_corp_widget".into()],
            &["page_token".into(), "code".into()],
        );
        assert_eq!(
            r.judge("xCorpWidget", Shape::Json),
            Some(Rule::SecretUserName)
        );
        assert_eq!(r.judge("pageToken", Shape::Json), None);
        assert_eq!(r.judge("code", Shape::UrlForm), None);
    }
}
