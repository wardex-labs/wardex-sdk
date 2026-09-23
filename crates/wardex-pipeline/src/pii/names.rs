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
    "cookies",
];

/// Rule 2: the name's LAST word, when the name has two or more words. These
/// words name many harmless things when they are not the head noun
/// (`key_id`, `token_type`, `pass_through`, `otp_length`); as the head noun
/// they name the credential (`api_key`, `access_token`, `db_pass`,
/// `sms_otp`, `card_cvv`). Singular only: `max_tokens` is a count.
pub const LAST_WORDS: &[&str] = &[
    "key", "token", "pass", "passcode", "otp", "totp", "cvv", "cvc",
];

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
    "pw",
    "pass",
    "passcode",
    "otp",
    "totp",
    "cvv",
    "cvc",
    "pin",
    "pincode",
    "pin_code",
    "csrftoken",
    "connect.sid",
    "auth_code",
    "device_code",
    "mfa_code",
];

/// Rule 4: whole names that are credentials only in the `name=value` shape.
/// OAuth sends `code` and Azure SAS sends `sig` as URL or form arguments,
/// Google's `key` is a query argument, and `sid` is a session cookie's name;
/// in JSON the same names are an error code (`"code": -32601`), a code
/// interpreter's source, a map entry's key or a resource id, all of which a
/// debugger needs.
pub const URL_FORM_ONLY_NAMES: &[&str] = &["code", "sig", "key", "sid"];

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
        self.scan_multipart(text, hits);
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let resume = match b[i] {
                b'=' => self.scan_form(text, i, hits),
                b':' => self.scan_json(text, i, hits),
                _ => None,
            };
            // Resume past a replaced value: its bytes are masked whatever
            // they hold, and rescanning them made `password=password=...`
            // quadratic.
            i = resume.map_or(i + 1, |r| r.max(i + 1));
        }
    }

    /// `name=value`. Returns where scanning resumes when a value was
    /// replaced.
    fn scan_form(&self, text: &str, eq: usize, hits: &mut Vec<NameHit>) -> Option<usize> {
        let b = text.as_bytes();
        let mut start = eq;
        while start > 0 && is_form_name_byte(b[start - 1]) {
            start -= 1;
            if eq - start > MAX_NAME_LEN {
                return None;
            }
        }
        if start == eq {
            return None;
        }
        let name = &text[start..eq];
        let rule = self.judge(name, Shape::UrlForm);
        // A secret's value runs to a terminator. An ordinary value is only
        // read for `%` escapes, so it also stops at the next `=` — a raw `=`
        // is never inside an encoded value, and stopping there keeps a long
        // run of `name=name=...` linear.
        let mut end = eq + 1;
        while end < b.len() && !is_form_value_end(b[end]) && (rule.is_some() || b[end] != b'=') {
            end += 1;
        }
        let value = &text[eq + 1..end];
        match rule {
            Some(rule) => {
                if value.is_empty() || value == PLACEHOLDER {
                    return None;
                }
                hits.push(NameHit {
                    start: eq + 1,
                    end,
                    replacement: PLACEHOLDER.to_string(),
                    rule,
                    name: name.to_string(),
                });
                Some(end)
            }
            None => {
                // An ordinary argument can carry another request inside it,
                // percent-encoded: `url=https%3A%2F%2Fh%2Fp%3Fapi_key%3D...`,
                // `filter=%7B%22api_key%22...`. Judge the decoded text.
                if value.contains('%') {
                    self.scan_encoded(text, eq + 1, end, hits);
                }
                None
            }
        }
    }

    /// Scan the percent-decoded form of `text[start..end]` and map each hit
    /// back onto the encoded bytes it came from.
    fn scan_encoded(&self, text: &str, start: usize, end: usize, hits: &mut Vec<NameHit>) {
        let enc = &text.as_bytes()[start..end];
        let mut dec = Vec::with_capacity(enc.len());
        let mut map = Vec::with_capacity(enc.len() + 1);
        let mut j = 0;
        while j < enc.len() {
            map.push(j);
            match enc[j] {
                b'%' if j + 2 < enc.len()
                    && enc[j + 1].is_ascii_hexdigit()
                    && enc[j + 2].is_ascii_hexdigit() =>
                {
                    dec.push(hex(enc[j + 1]) << 4 | hex(enc[j + 2]));
                    j += 3;
                }
                b'+' => {
                    dec.push(b' ');
                    j += 1;
                }
                c => {
                    dec.push(c);
                    j += 1;
                }
            }
        }
        map.push(enc.len());
        if dec.len() == enc.len() {
            return; // nothing was encoded
        }
        // A Latin-1 `%E9` must not hide the rest of the value: judge a
        // same-length stand-in, whose offsets are the decoded bytes' own.
        let decoded = super::ascii_stand_in(&dec);
        let mut inner = Vec::new();
        self.scan(&decoded, &mut inner);
        for h in inner {
            hits.push(NameHit {
                start: start + map[h.start],
                end: start + map[h.end],
                replacement: PLACEHOLDER.to_string(),
                rule: h.rule,
                name: h.name,
            });
        }
    }

    /// A `multipart/form-data` field: `form-data; name="password"` then a
    /// blank line, then the value up to the next boundary. A part with a
    /// `filename` is a file, not an argument.
    fn scan_multipart(&self, text: &str, hits: &mut Vec<NameHit>) {
        // Linear however many `form-data;` a text holds: the next line end
        // and the next blank line are each found once and reused until the
        // scan passes them, and a header block or header line longer than
        // its bound is not a multipart header.
        const HEADER_MAX: usize = 1024;
        const HEADERS_MAX: usize = 8192;
        let next = |cache: &mut Option<Option<usize>>, from: usize, needle: &str| {
            if cache.is_none_or(|c| c.is_some_and(|at| at < from)) {
                *cache = Some(text[from..].find(needle).map(|x| from + x));
            }
            cache.flatten()
        };
        let mut crlf: Option<Option<usize>> = None;
        let mut blank: Option<Option<usize>> = None;
        let mut from = 0;
        while let Some(pos) = text[from..].find("form-data;").map(|x| from + x) {
            from = pos + "form-data;".len();
            let Some(line_end) = next(&mut crlf, pos, "\r\n") else {
                break; // no line end anywhere after this: no header can end
            };
            if line_end - pos > HEADER_MAX {
                continue;
            }
            let header = &text[pos..line_end];
            if header.contains("filename=") {
                continue;
            }
            let Some(ns) = header.find("name=\"").map(|x| pos + x + 6) else {
                continue;
            };
            let Some(ne) = text[ns..line_end].find('"').map(|x| ns + x) else {
                continue;
            };
            let Some(body) = next(&mut blank, line_end, "\r\n\r\n").map(|x| x + 4) else {
                break; // no blank line anywhere after this: no part has a body
            };
            if body - line_end > HEADERS_MAX {
                continue;
            }
            let end = text[body..].find("\r\n--").map_or(text.len(), |x| body + x);
            from = from.max(end);
            let name = &text[ns..ne];
            let Some(rule) = self.judge(name, Shape::UrlForm) else {
                continue;
            };
            let value = &text[body..end];
            if value.is_empty() || value == PLACEHOLDER {
                continue;
            }
            hits.push(NameHit {
                start: body,
                end,
                replacement: PLACEHOLDER.to_string(),
                rule,
                name: name.to_string(),
            });
        }
    }

    /// `"name": value`. Returns where scanning resumes when a value was
    /// replaced.
    fn scan_json(&self, text: &str, colon: usize, hits: &mut Vec<NameHit>) -> Option<usize> {
        let b = text.as_bytes();
        // The name's closing quote, skipping whitespace before the colon.
        let mut k = colon;
        while k > 0 && b[k - 1].is_ascii_whitespace() {
            k -= 1;
        }
        if k == 0 {
            return None;
        }
        let close = k - 1;
        let quote = b[close];
        if quote != b'"' && quote != b'\'' {
            return None;
        }
        // How deeply the JSON is escaped: 0 for plain JSON, 1 for JSON inside
        // a JSON string, 3 for one level further. A single quote is only ever
        // plain (a Python repr).
        let level = backslashes_before(b, close);
        if !matches!(level, 0 | 1 | 3 | 7) || (quote == b'\'' && level != 0) {
            return None;
        }
        let name_end = close - level;
        let mut p = name_end;
        while p > 0 && b[p - 1] != quote && b[p - 1] != b'\\' {
            p -= 1;
            if name_end - p > MAX_NAME_LEN {
                return None;
            }
        }
        if p == 0 || b[p - 1] != quote || p == name_end {
            return None;
        }
        let open = p - 1;
        if backslashes_before(b, open) != level {
            return None;
        }
        let name = &text[p..name_end];
        let rule = self.judge(name, Shape::Json)?;
        let mut v = colon + 1;
        while v < b.len() && b[v].is_ascii_whitespace() {
            v += 1;
        }
        if v < b.len() && (b[v] == b'{' || b[v] == b'[') {
            // An object or a list under a secret name is the secret, every
            // leaf of it (`"client_secret": {"value": "..."}`); its own keys
            // stay readable.
            return Some(mask_container(text, v, quote, level, rule, name, hits));
        }
        let (s, e, next, is_string) = scalar_at(b, v, quote, level)?;
        push_value(text, s, e, is_string, quote, level, rule, name, hits);
        Some(next)
    }
}

/// A JSON scalar starting at `at`: `(start, end, next, is_string)` where
/// `start..end` is the range to replace (a string's content, or a number's
/// digits) and `next` is the first byte after it. A string that never closes
/// — a body cut at the capture cap — runs to the end of the text: its tail is
/// still the secret. `None` for anything else.
fn scalar_at(b: &[u8], at: usize, quote: u8, level: usize) -> Option<(usize, usize, usize, bool)> {
    if at >= b.len() {
        return None;
    }
    let first = b[at];
    if first == b'-' || first.is_ascii_digit() {
        let mut e = at + 1;
        while e < b.len() && matches!(b[e], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
            e += 1;
        }
        return Some((at, e, e, false));
    }
    // A string opens with `level` backslashes and a quote. Plain text may use
    // either quote whatever the name used: a Python repr switches to `"` for
    // a value that contains `'`.
    if at + level >= b.len() || b[at..at + level].iter().any(|&c| c != b'\\') {
        return None;
    }
    let q = b[at + level];
    if !(q == quote || (level == 0 && (q == b'"' || q == b'\''))) {
        return None;
    }
    let start = at + level + 1;
    let modulus = 2 * (level + 1);
    let mut i = start;
    while i < b.len() {
        if b[i] == q {
            let run = backslashes_before(b, i);
            // At escape depth `level` a closing quote carries exactly `level`
            // backslashes modulo the next depth's escaping; any other run is
            // a quote INSIDE the string.
            if run % modulus == level && i >= start + level {
                return Some((start, i - level, i + 1, true));
            }
        }
        i += 1;
    }
    Some((start, b.len(), b.len(), true))
}

/// Replace every scalar leaf of the object or list opening at `open`; keys
/// stay. Returns the byte after the matching close (or the end of the text,
/// for a container the capture cut short).
fn mask_container(
    text: &str,
    open: usize,
    quote: u8,
    level: usize,
    rule: Rule,
    name: &str,
    hits: &mut Vec<NameHit>,
) -> usize {
    let b = text.as_bytes();
    let mut depth = 0usize;
    let mut prev = b'[';
    let mut i = open;
    while i < b.len() {
        let c = b[i];
        match c {
            b'{' | b'[' => {
                depth += 1;
                prev = c;
                i += 1;
                continue;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
                if depth == 0 {
                    return i;
                }
                prev = c;
                continue;
            }
            b':' | b',' => {
                prev = c;
                i += 1;
                continue;
            }
            _ => {}
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let is_number_start = c == b'-' || c.is_ascii_digit();
        if is_number_start && !matches!(prev, b':' | b',' | b'[') {
            i += 1;
            continue;
        }
        if let Some((s, e, next, is_string)) = scalar_at(b, i, quote, level) {
            let mut k = next;
            while k < b.len() && b[k].is_ascii_whitespace() {
                k += 1;
            }
            let is_key = is_string && k < b.len() && b[k] == b':';
            if !is_key {
                push_value(text, s, e, is_string, quote, level, rule, name, hits);
            }
            prev = b'"';
            i = next.max(i + 1);
            continue;
        }
        prev = c;
        i += 1;
    }
    b.len()
}

#[allow(clippy::too_many_arguments)]
fn push_value(
    text: &str,
    start: usize,
    end: usize,
    is_string: bool,
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

fn hex(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => c - b'A' + 10,
    }
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

/// Bytes that end a `name=value` value. `,` ends one too, so a list of
/// pairs (`tags=key=a,color=red`) keeps its other members.
fn is_form_value_end(c: u8) -> bool {
    c.is_ascii_whitespace()
        || matches!(
            c,
            b'&' | b'#' | b';' | b',' | b'"' | b'\'' | b'\\' | b'<' | b'>' | b'}'
        )
}

/// A name's words: `%XX` escapes decoded, then split at every
/// non-alphanumeric character and at camelCase boundaries (`apiKey` →
/// api·key, `AWSAccessKeyId` → aws·access·key·id), lowercased, with trailing
/// digits dropped (`password1` → password, `oauth2Token` → oauth·token).
/// Non-ASCII letters are kept as word characters and never split on case.
pub fn words(name: &str) -> Vec<String> {
    // Decode `%XX` first (`%5F` is `_`, `%70` is `p`): judge the name meant.
    let raw: Vec<char> = name.chars().collect();
    let mut chars = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let c = raw[i];
        if c == '%'
            && i + 2 < raw.len()
            && raw[i + 1].is_ascii_hexdigit()
            && raw[i + 2].is_ascii_hexdigit()
        {
            chars.push(char::from(
                hex(raw[i + 1] as u8) << 4 | hex(raw[i + 2] as u8),
            ));
            i += 3;
        } else {
            chars.push(c);
            i += 1;
        }
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_alphanumeric() {
            flush(&mut cur, &mut out);
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
    }
    flush(&mut cur, &mut out);
    out
}

/// End a word. Trailing digits are a version or a counter, not part of what
/// the word names: `password1` is a password, `api_key2` an API key.
fn flush(cur: &mut String, out: &mut Vec<String>) {
    let word = std::mem::take(cur);
    let trimmed = word.trim_end_matches(|c: char| c.is_ascii_digit());
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
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
        assert_eq!(words("oauth2Token"), ["oauth", "token"]);
        assert_eq!(words("password1"), ["password"]);
        assert_eq!(words("%70assword"), ["password"]);
        assert_eq!(words("api_ke%79"), ["api", "key"]);
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
            ("next_token", Rule::SecretLastWord),
            ("public_key", Rule::SecretLastWord),
            ("object_key", Rule::SecretLastWord),
            ("partition_key", Rule::SecretLastWord),
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

    #[test]
    fn a_value_cut_by_the_capture_cap_is_masked_to_the_end() {
        assert_eq!(
            mask(r#"{"q": "x", "api_key": "TRUNC"#),
            r#"{"q": "x", "api_key": "[SECRET]"#
        );
    }

    #[test]
    fn every_leaf_under_a_secret_name_is_masked_and_keys_stay() {
        assert_eq!(
            mask(r#"{"client_secret": {"value": "ek_1", "expires_at": 17}, "id": "s1"}"#),
            r#"{"client_secret": {"value": "[SECRET]", "expires_at": "[SECRET]"}, "id": "s1"}"#
        );
        assert_eq!(
            mask(r#"{"api_key": ["A1", {"v": "A2"}, "A3"], "q": "k"}"#),
            r#"{"api_key": ["[SECRET]", {"v": "[SECRET]"}, "[SECRET]"], "q": "k"}"#
        );
        // Cut short by the capture cap: the leaves that arrived are masked.
        assert_eq!(
            mask(r#"{"cookies": [{"value": "C1"#),
            r#"{"cookies": [{"value": "[SECRET]"#
        );
    }

    #[test]
    fn a_multipart_field_is_a_form_field() {
        let body = "--b\r\nContent-Disposition: form-data; name=\"password\"\r\n\r\nMP1\r\n\
                    --b\r\nContent-Disposition: form-data; name=\"q\"\r\n\r\nseoul\r\n\
                    --b\r\nContent-Disposition: form-data; name=\"key\"; filename=\"a.txt\"\r\n\r\nfile body\r\n--b--";
        let out = mask(body);
        assert!(out.contains("\r\n\r\n[SECRET]\r\n"), "{out}");
        assert!(out.contains("seoul") && out.contains("file body"), "{out}");
    }

    #[test]
    fn a_percent_encoded_request_inside_an_argument_is_judged() {
        assert_eq!(
            mask("url=https%3A%2F%2Fh%2Fp%3Fapi_key%3DSEC1%26q%3Dseoul&x=1"),
            "url=https%3A%2F%2Fh%2Fp%3Fapi_key%3D[SECRET]%26q%3Dseoul&x=1"
        );
        assert_eq!(
            mask("filter=%7B%22api_key%22%3A%22SEC2%22%2C%22q%22%3A1%7D"),
            "filter=%7B%22api_key%22%3A%22[SECRET]%22%2C%22q%22%3A1%7D"
        );
    }

    #[test]
    fn repr_quotes_names_and_pair_lists() {
        assert_eq!(
            mask(r#"{'password': "it'sSEC"}"#),
            r#"{'password': "[SECRET]"}"#
        );
        assert_eq!(
            mask("tags=key=a,key=b,color=red&page=2"),
            "tags=key=[SECRET],key=[SECRET],color=red&page=2"
        );
        for name in [
            "pass",
            "pin",
            "otp",
            "cvv",
            "csrftoken",
            "connect.sid",
            "password1",
            "api_key2",
            "%70assword",
        ] {
            assert_eq!(
                mask(&format!("{name}=S1")),
                format!("{name}=[SECRET]"),
                "{name}"
            );
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
            "x-auth=S1",
            "x_api_key_v2=S1",
            "<password>S1</password>",
            r#"{"api\u005fkey": "S1"}"#,
            r#"{"code": "4/0AX4XfWh"}"#,
        ] {
            let mut hits = Vec::new();
            rules().scan(text, &mut hits);
            assert!(hits.is_empty(), "{text}");
        }
    }

    #[test]
    fn documented_consequences_hold() {
        // `name=value` is judged wherever it appears, a SQL string included.
        assert_eq!(
            mask("query=SELECT * FROM t WHERE token=1"),
            "query=SELECT * FROM t WHERE token=[SECRET]"
        );
        // An object under a secret name is masked whole.
        assert_eq!(
            mask(r#"{"Credentials": {"Expiration": "2026"}}"#),
            r#"{"Credentials": {"Expiration": "[SECRET]"}}"#
        );
        // A name longer than 128 characters is not read.
        let long = format!("{}_password=S1", "a".repeat(130));
        assert_eq!(mask(&long), long);
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
