//! PII masking engine.
//!
//! Detection is fully deterministic: regex proposes, checksum validators
//! confirm. No ML or entropy heuristics, so the same text is always masked
//! the same way. Two families of rules feed one replacement pass: VALUE rules
//! recognise a value by its shape (`patterns.rs`), NAME rules recognise it by
//! the argument it is passed as (`names.rs`). Every replacement is recorded
//! against the rule that made it, so a span can say what was masked and why.

mod names;
mod patterns;
mod walk;

pub use names::{
    words, NameRules, Shape, EXACT_NAMES, LAST_WORDS, PLACEHOLDER, STRONG_WORDS,
    URL_FORM_ONLY_NAMES,
};
pub use walk::{mask_envelope, mask_otlp};
pub use wardex_codec::proto::wardex::v1::RedactionRule as Rule;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use regex::Regex;

use patterns::{Replacement, BUILTINS, CATEGORIES};

/// Fail-closed placeholder — what a span's text becomes when masking itself
/// failed and we refuse to ship the original.
pub const PII_FILTER_ERROR: &str = "[PII_FILTER_ERROR]";

/// What replaces the credentials in `scheme://user:password@host` — the value
/// OTel's semantic conventions prescribe for `url.full`.
pub const USERINFO_REDACTED: &str = "REDACTED:REDACTED";

/// Most distinct argument names one span records. The count and the rules are
/// never capped; only this list is, so a body with thousands of secret-named
/// fields cannot grow a span without bound.
pub const MAX_REPORTED_NAMES: usize = 32;

/// Why an engine could not be built from a masking configuration.
#[derive(Debug, thiserror::Error)]
pub enum PiiError {
    /// A category to disable that no built-in pattern belongs to, as the host
    /// spelled it. Rejected rather than ignored, so a misspelled category
    /// cannot leave the intended one switched on.
    #[error("unknown PII category `{0}` (known: {known})", known = CATEGORIES.join(", "))]
    UnknownCategory(String),
    /// A built-in pattern failed to compile. The patterns are constants of
    /// this crate, so this is a defect in the SDK, never in the host's input.
    #[error("invalid builtin pattern for category {category}: {message}")]
    InvalidPattern {
        /// The category whose pattern failed, as listed in the category table.
        category: String,
        /// The regex compiler's own explanation.
        message: String,
    },
}

#[derive(Debug)]
struct Compiled {
    regex: Regex,
    validator: Option<fn(&str) -> bool>,
    /// `PatternDef::context`: the match judged against the text around it.
    context: Option<fn(&str, usize, usize) -> bool>,
    replacement: &'static Replacement,
    rule: Rule,
    /// Rescan policy after a validator rejection. A rejected candidate
    /// normally suppresses its whole span — that is how the `{3,}` IPv4 tail
    /// rejects version strings like "1.2.3.4.5" outright. Credit cards are
    /// the exception: the separator-tolerant regex can greedily bleed into a
    /// neighboring token (e.g. the last octet of a preceding IP), and that
    /// spurious Luhn failure must not swallow the valid card starting inside
    /// the rejected span. Luhn re-confirms every retried candidate.
    retry_on_reject: bool,
}

/// What masking replaced, accumulated over every text of one record (a span).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    /// Replacements made. Two secrets in one body count two.
    pub count: u32,
    /// Each rule that replaced at least one value, in enum order.
    pub rules: Vec<Rule>,
    /// Argument names a name rule matched, first-seen order, deduplicated,
    /// at most `MAX_REPORTED_NAMES`.
    pub names: Vec<String>,
}

impl Report {
    /// True when nothing was replaced. `rules` and `names` are then empty as
    /// well, since each is recorded only alongside a replacement.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn record(&mut self, rule: Rule, name: Option<&str>) {
        self.count = self.count.saturating_add(1);
        if let Err(at) = self.rules.binary_search(&rule) {
            self.rules.insert(at, rule);
        }
        if let Some(n) = name {
            if self.names.len() < MAX_REPORTED_NAMES && !self.names.iter().any(|x| x == n) {
                self.names.push(n.to_string());
            }
        }
    }
}

struct Hit {
    start: usize,
    end: usize,
    replacement: String,
    rule: Rule,
    name: Option<String>,
}

/// Compiled pattern set. Build once per config via `engine_for`: compiling
/// the regexes is the expensive part.
#[derive(Debug)]
pub struct PiiEngine {
    active: Vec<Compiled>,
    /// `None` when the `secret` category is disabled: the name rules belong to
    /// it, as the value shapes of secrets do.
    names: Option<NameRules>,
}

fn category_rule(category: &str) -> Rule {
    match category {
        "email" => Rule::Email,
        "phone_number" => Rule::PhoneNumber,
        "credit_card" => Rule::CreditCard,
        "us_ssn" => Rule::UsSsn,
        "ip_address" => Rule::IpAddress,
        "us_bank_routing" => Rule::UsBankRouting,
        "iban" => Rule::Iban,
        _ => Rule::SecretValue,
    }
}

/// The userinfo of every `scheme://userinfo@host` in `text`: after `://`,
/// everything up to the last `@` before any path, query, fragment, whitespace
/// or quoting byte — a raw password may itself contain `@`. A scanner rather
/// than a regex so there is no compile step that could fail on the export
/// path.
fn userinfo_hits(text: &str, hits: &mut Vec<Hit>, spans: &mut Vec<(usize, usize)>) {
    let b = text.as_bytes();
    for (sep, _) in text.match_indices("://") {
        let mut s = sep;
        while s > 0 && (b[s - 1].is_ascii_alphanumeric() || matches!(b[s - 1], b'+' | b'.' | b'-'))
        {
            s -= 1;
        }
        // A scheme starts with a letter; digits glued on before it are not
        // part of it (`…:8080http://` when a target is appended to a URL).
        while s < sep && !b[s].is_ascii_alphabetic() {
            s += 1;
        }
        if s == sep {
            continue; // no scheme before `://`
        }
        let start = sep + 3;
        let mut stop = start;
        let mut at = None;
        while stop < b.len()
            && !b[stop].is_ascii_whitespace()
            && !matches!(
                b[stop],
                b'/' | b'?'
                    | b'#'
                    | b'['
                    | b']'
                    | b'"'
                    | b'\''
                    | b'<'
                    | b'>'
                    | b'\\'
                    | b','
                    | b'('
                    | b')'
                    // An argument boundary: `cb=http://h&password=p@ss` is
                    // two arguments, not one URL's userinfo.
                    | b'&'
                    | b'='
                    | b';'
            )
        {
            if b[stop] == b'@' {
                at = Some(stop);
            }
            stop += 1;
        }
        let Some(end) = at else {
            continue;
        };
        if end == start {
            continue;
        }
        // `start` follows the ASCII `://` and `end` is the offset of an ASCII
        // `@`, so this never misses; `get` keeps a slicing bug from becoming
        // a panic that prints the text.
        let Some(userinfo) = text.get(start..end) else {
            continue;
        };
        let replacement = if userinfo.contains(':') {
            USERINFO_REDACTED
        } else {
            "REDACTED"
        };
        if userinfo == replacement {
            continue; // already redacted — not a second replacement
        }
        spans.push((start, end));
        hits.push(Hit {
            start,
            end,
            replacement: replacement.to_string(),
            rule: Rule::UrlUserinfo,
            name: None,
        });
    }
}

/// `bytes` as text of the same length: each byte of an invalid UTF-8
/// sequence becomes `?`, a byte no rule treats as a boundary.
pub(crate) fn ascii_stand_in(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for chunk in bytes.utf8_chunks() {
        out.push_str(chunk.valid());
        out.extend(std::iter::repeat_n('?', chunk.invalid().len()));
    }
    out
}

impl PiiEngine {
    /// Build an engine with `disabled` categories removed.
    /// Unknown category names are a hard error — the FFI contract is fail-loud.
    pub fn new(disabled: &[String]) -> Result<PiiEngine, PiiError> {
        PiiEngine::with_names(disabled, &[], &[])
    }

    /// `new`, plus the application's own secret names (`extra`) and names
    /// exempted from the name rules (`reveal`).
    pub fn with_names(
        disabled: &[String],
        extra: &[String],
        reveal: &[String],
    ) -> Result<PiiEngine, PiiError> {
        for d in disabled {
            if !CATEGORIES.contains(&d.as_str()) {
                return Err(PiiError::UnknownCategory(d.clone()));
            }
        }
        let mut active = Vec::new();
        for def in BUILTINS {
            if disabled.iter().any(|d| d == def.category) {
                continue;
            }
            let regex = Regex::new(def.regex).map_err(|e| PiiError::InvalidPattern {
                category: def.category.to_string(),
                message: e.to_string(),
            })?;
            active.push(Compiled {
                regex,
                validator: def.validator,
                context: def.context,
                replacement: &def.replacement,
                rule: category_rule(def.category),
                retry_on_reject: def.retry_on_reject,
            });
        }
        let names =
            (!disabled.iter().any(|d| d == "secret")).then(|| NameRules::new(extra, reveal));
        Ok(PiiEngine { active, names })
    }

    /// The engine `PIIMode.OFF` still runs: URL credentials only. OTel's
    /// semantic conventions require them out of `url.full`, and a
    /// `user:password@` has no debugging value an opt-out could be protecting.
    pub fn userinfo_only() -> PiiEngine {
        PiiEngine {
            active: Vec::new(),
            names: None,
        }
    }

    /// The name rules this engine applies, if the `secret` category is on.
    pub fn name_rules(&self) -> Option<&NameRules> {
        self.names.as_ref()
    }

    /// Mask every validated match. `None` means "no change".
    #[must_use = "`None` means no change; ignoring `Some` ships the unmasked text"]
    pub fn mask_text(&self, text: &str) -> Option<String> {
        self.mask_text_into(text, &mut Report::default())
    }

    /// `mask_text`, recording each replacement into `report`.
    /// Single pass over the original text (retry-on-reject patterns may
    /// re-probe within a rejected candidate); replaced regions are never
    /// rescanned, so placeholders cannot re-match.
    #[must_use = "`None` means no change; ignoring `Some` ships the unmasked text"]
    pub fn mask_text_into(&self, text: &str, report: &mut Report) -> Option<String> {
        rewrite(text, self.edits(text, report, None))
    }

    /// `mask_text_into` for a value that is, whole, a host or peer address
    /// (`server.address`): a private address is where the traffic went, not
    /// anyone's data, and stays as written. Anything else is masked as text.
    #[must_use = "`None` means no change; ignoring `Some` ships the unmasked text"]
    fn mask_host_into(&self, host: &str, report: &mut Report) -> Option<String> {
        if is_private_address(host) {
            return None;
        }
        self.mask_text_into(host, report)
    }

    /// `mask_text_into` for a URL: a private address as the URL's own host
    /// stays as written; every other part of it (credentials, path, query)
    /// and a public host are masked as text.
    #[must_use = "`None` means no change; ignoring `Some` ships the unmasked text"]
    fn mask_url_into(&self, url: &str, report: &mut Report) -> Option<String> {
        let keep = url_host(url)
            .filter(|&(start, end)| url.get(start..end).is_some_and(is_private_address));
        rewrite(url, self.edits(url, report, keep))
    }

    /// Bytes that may not be UTF-8 as a whole. Each invalid byte is judged as
    /// a `?` — one byte for one byte, so every offset holds — and the edits
    /// land on the original bytes. A Latin-1 `é` inside a password therefore
    /// does not split the value in two and ship its tail.
    #[must_use = "`None` means no change; ignoring `Some` ships the unmasked bytes"]
    pub fn mask_bytes_into(&self, bytes: &[u8], report: &mut Report) -> Option<Vec<u8>> {
        let text = std::str::from_utf8(bytes)
            .map(std::borrow::Cow::Borrowed)
            .unwrap_or_else(|_| std::borrow::Cow::Owned(ascii_stand_in(bytes)));
        let edits = self.edits(&text, report, None);
        if edits.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(bytes.len());
        let mut pos = 0usize;
        for (start, end, replacement) in edits {
            out.extend_from_slice(&bytes[pos..start]);
            out.extend_from_slice(replacement.as_bytes());
            pos = end;
        }
        out.extend_from_slice(&bytes[pos..]);
        Some(out)
    }

    /// Every replacement `text` needs, as non-overlapping `(start, end,
    /// replacement)` in order, each recorded into `report`. An address match
    /// covering exactly `keep` is left out (a URL's private host).
    fn edits(
        &self,
        text: &str,
        report: &mut Report,
        keep: Option<(usize, usize)>,
    ) -> Vec<(usize, usize, String)> {
        let mut hits: Vec<Hit> = Vec::new();
        for p in &self.active {
            let mut at = 0;
            while let Some(m) = p.regex.find_at(text, at) {
                if p.confirms(text, m.start(), m.end()) {
                    hits.push(Hit {
                        start: m.start(),
                        end: m.end(),
                        replacement: apply(p.replacement, m.as_str()),
                        rule: p.rule,
                        name: None,
                    });
                    at = m.end();
                } else if let Some((end, rep)) = p
                    .retry_on_reject
                    .then(|| shrink_to_valid(p, text, m.start(), m.end()))
                    .flatten()
                {
                    // The greedy candidate bled into a trailing digit run; a
                    // shorter end at the same start re-validated (e.g. the
                    // 16-digit PAN inside "4111 1111 1111 1111 999").
                    hits.push(Hit {
                        start: m.start(),
                        end,
                        replacement: rep,
                        rule: p.rule,
                        name: None,
                    });
                    at = end;
                } else if p.retry_on_reject {
                    // Resume just past the rejected candidate's first char so
                    // a valid match starting inside the span is still found
                    // (leading bleed, e.g. out of a preceding IP octet).
                    at = m.start()
                        + text
                            .get(m.start()..)
                            .and_then(|rest| rest.chars().next())
                            .map_or(1, char::len_utf8);
                } else {
                    // Rejection consumes the span (find_iter semantics).
                    at = m.end();
                }
                if at >= text.len() {
                    break;
                }
            }
        }
        if keep.is_some() {
            hits.retain(|h| !(h.rule == Rule::IpAddress && Some((h.start, h.end)) == keep));
        }
        if let Some(rules) = &self.names {
            let mut found = Vec::new();
            rules.scan(text, &mut found);
            hits.extend(found.into_iter().map(|h| Hit {
                start: h.start,
                end: h.end,
                replacement: h.replacement,
                rule: h.rule,
                name: Some(h.name),
            }));
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        userinfo_hits(text, &mut hits, &mut spans);
        if !spans.is_empty() {
            // A userinfo is the credential and nothing else: a VALUE rule
            // that starts inside it (`PW@api.example.com` read as an e-mail)
            // would otherwise run its replacement over the host. Name-rule
            // hits stay — they are secrets whatever they overlap. `spans` is
            // in text order, so each lookup is a binary search.
            hits.retain(|h| {
                if h.rule == Rule::UrlUserinfo || h.name.is_some() {
                    return true;
                }
                let i = spans.partition_point(|&(s, _)| s <= h.start);
                i == 0 || h.start > spans[i - 1].1
            });
        }
        // Leftmost first; on ties the longer match wins, so the output does
        // not depend on pattern order; and on an identical range the name
        // rule's hit wins, so the report keeps the name.
        hits.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then(b.end.cmp(&a.end))
                .then(a.name.is_none().cmp(&b.name.is_none()))
        });
        let mut edits = Vec::with_capacity(hits.len());
        let mut pos = 0usize;
        for h in hits {
            if h.start < pos {
                // Overlapped by an earlier winner. An overlapped-but-validated
                // hit still masks its remainder — never emit a validated
                // span's tail raw (e.g. a Luhn-passing card that bled
                // backwards into a preceding IP match).
                if h.end > pos {
                    edits.push((pos, h.end, h.replacement));
                    pos = h.end;
                    report.record(h.rule, h.name.as_deref());
                }
                continue;
            }
            pos = h.end;
            report.record(h.rule, h.name.as_deref());
            edits.push((h.start, h.end, h.replacement));
        }
        edits
    }
}

impl Compiled {
    /// The validator and the context check both accept `text[start..end]`.
    fn confirms(&self, text: &str, start: usize, end: usize) -> bool {
        let Some(matched) = text.get(start..end) else {
            return false;
        };
        self.validator.is_none_or(|v| v(matched))
            && self.context.is_none_or(|c| c(text, start, end))
    }
}

/// `text` with `edits` applied, or None when there are none.
#[expect(
    clippy::string_slice,
    reason = "every edit bound is a regex match bound, the offset of an ASCII \
              delimiter found by the userinfo or name scanner, or the end of an \
              earlier edit, so each is a char boundary of `text`"
)]
fn rewrite(text: &str, edits: Vec<(usize, usize, String)>) -> Option<String> {
    if edits.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut pos = 0usize;
    for (start, end, replacement) in edits {
        out.push_str(&text[pos..start]);
        out.push_str(&replacement);
        pos = end;
    }
    out.push_str(&text[pos..]);
    Some(out)
}

/// A loopback (`127.0.0.0/8`, `::1`), RFC 1918 private (`10/8`,
/// `172.16/12`, `192.168/16`) or link-local (`169.254/16`) address literal,
/// an IPv6 one with or without brackets. Only these: a public address names
/// a party outside the host's own network.
fn is_private_address(text: &str) -> bool {
    let bare = text
        .strip_prefix('[')
        .and_then(|t| t.strip_suffix(']'))
        .unwrap_or(text);
    match bare.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        Ok(std::net::IpAddr::V6(v6)) => v6.is_loopback(),
        Err(_) => false,
    }
}

/// The byte range of a URL's host in `scheme://userinfo@host:port/...`:
/// after the last `@` of the authority and before its port, brackets kept
/// around an IPv6 literal. None without a `://`.
fn url_host(url: &str) -> Option<(usize, usize)> {
    let start = url.find("://")?.checked_add(3)?;
    let rest = url.get(start..)?;
    let authority = rest.get(..rest.find(['/', '?', '#']).unwrap_or(rest.len()))?;
    let from = authority.rfind('@').map_or(0, |at| at + 1);
    let host_port = authority.get(from..)?;
    let len = if host_port.starts_with('[') {
        host_port
            .find(']')
            .map_or(host_port.len(), |close| close + 1)
    } else {
        host_port.find(':').unwrap_or(host_port.len())
    };
    Some((start + from, start + from + len))
}

/// After a rejected retry-on-reject candidate, probe progressively shorter
/// end positions at the same start. The separator-tolerant card regex can
/// greedily bleed into a trailing digit run ("4111 1111 1111 1111 999"), and
/// the Luhn failure of that overlong candidate must not ship the embedded
/// valid PAN unmasked. Probes only end on a digit-run boundary (never cut a
/// digit run in half) and must keep at least 13 digits — the regex's own
/// minimum (`[0-9]` + `{12,18}` tail). The validator re-confirms each probe,
/// longest first; the first pass wins.
fn shrink_to_valid(p: &Compiled, text: &str, start: usize, end: usize) -> Option<(usize, String)> {
    p.validator?;
    // The matched span is pure ASCII (digits and `[ -]` separators), so byte
    // positions inside it are always char boundaries.
    let bytes = text.as_bytes();
    for e in (start + 1..end).rev() {
        if !bytes[e - 1].is_ascii_digit() || bytes[e].is_ascii_digit() {
            continue; // candidate must end a digit run, not split one
        }
        let Some(cand) = text.get(start..e) else {
            break;
        };
        if cand.bytes().filter(u8::is_ascii_digit).count() < 13 {
            break; // shorter probes only lose more digits
        }
        if p.confirms(text, start, e) {
            return Some((e, apply(p.replacement, cand)));
        }
    }
    None
}

fn apply(r: &Replacement, matched: &str) -> String {
    match r {
        Replacement::Label(l) => (*l).to_string(),
        Replacement::CardLast4 => {
            let digits: String = matched.chars().filter(|c| c.is_ascii_digit()).collect();
            // Invariant: only a Luhn-valid candidate reaches here, and Luhn
            // requires >=13 digits (patterns::luhn_valid) — the >=4 floor for
            // the last-4 slice below is guaranteed, never a real underflow.
            debug_assert!(
                digits.len() >= 4,
                "CardLast4 requires the >=13-digit Luhn floor"
            );
            let last4 = digits
                .get(digits.len().saturating_sub(4)..)
                .unwrap_or_default();
            format!("****-****-****-{last4}")
        }
        Replacement::AfterColon(label) => {
            // The pattern guarantees a colon; the name and the spacing after
            // it are the writer's and stay as written.
            // What follows the colon, less its leading spacing, is a suffix
            // of `matched`; stripping it leaves the name, the colon and the
            // spacing.
            let head = matched
                .split_once(':')
                .and_then(|(_, after)| matched.strip_suffix(after.trim_start()))
                .unwrap_or_default();
            format!("{head}{label}")
        }
    }
}

/// Process-wide engine cache keyed by the sorted disabled list.
/// Regex compilation is the expensive part — pay it once per config.
pub fn engine_for(disabled: &[String]) -> Result<Arc<PiiEngine>, PiiError> {
    engine_for_policy(disabled, &[], &[])
}

/// `engine_for` with the application's secret names and exemptions, all three
/// lists part of the cache key.
pub fn engine_for_policy(
    disabled: &[String],
    extra: &[String],
    reveal: &[String],
) -> Result<Arc<PiiEngine>, PiiError> {
    type Key = (Vec<String>, Vec<String>, Vec<String>);
    static CACHE: OnceLock<Mutex<HashMap<Key, Arc<PiiEngine>>>> = OnceLock::new();
    let canon = |v: &[String]| {
        let mut k = v.to_vec();
        k.sort();
        k.dedup();
        k
    };
    let key = (canon(disabled), canon(extra), canon(reveal));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // A poisoned cache is still consistent: it only ever gains whole entries,
    // each an immutable `Arc`, so no panic can leave one half-written. The
    // host calls this on every export, so panicking on poison here would turn
    // one past panic into a failure of every later export.
    let lookup = |key: &Key| {
        cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
    };
    if let Some(engine) = lookup(&key) {
        return Ok(engine);
    }
    // Compiled with the lock released: compilation takes milliseconds, and a
    // thread asking for an engine that is already cached must not wait for
    // it. Two threads may race to build the same engine; the first insert
    // wins and the other copy is dropped.
    let engine = Arc::new(PiiEngine::with_names(&key.0, &key.1, &key.2)?);
    let mut guard = cache.lock().unwrap_or_else(PoisonError::into_inner);
    Ok(Arc::clone(guard.entry(key).or_insert(engine)))
}

/// The shared `userinfo_only` engine.
pub fn userinfo_engine() -> Arc<PiiEngine> {
    static ENGINE: OnceLock<Arc<PiiEngine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| Arc::new(PiiEngine::userinfo_only()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    fn engine() -> PiiEngine {
        PiiEngine::new(&[]).unwrap()
    }

    fn mask(text: &str) -> String {
        engine().mask_text(text).unwrap_or_else(|| text.to_string())
    }

    // --- true positives ---

    #[test]
    fn masks_each_category() {
        assert_eq!(mask("mail john.doe@acme.com now"), "mail [EMAIL] now");
        assert_eq!(mask("call (555) 123-4567"), "call [PHONE]");
        assert_eq!(mask("call +1 555 123 4567"), "call [PHONE]");
        assert_eq!(
            mask("card 4111-1111-1111-1111 ok"),
            "card ****-****-****-1111 ok"
        );
        assert_eq!(mask("ssn 123-45-6789"), "ssn [SSN]");
        assert_eq!(mask("host 10.0.0.5 down"), "host [IP_ADDRESS] down");
        assert_eq!(mask("v6 2001:db8::1 up"), "v6 [IP_ADDRESS] up");
        assert_eq!(mask("routing 021000021"), "routing [BANK_ROUTING]");
        assert_eq!(mask("iban GB82WEST12345698765432"), "iban [IBAN]");
        assert_eq!(mask("key sk-abcdefghijklmnop1234"), "key [SECRET]");
        assert_eq!(mask("aws AKIAIOSFODNN7EXAMPLE"), "aws [SECRET]");
        assert_eq!(
            mask("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3ZHgifQ.c2lnbmF0dXJl ok"),
            "jwt [SECRET] ok"
        );
        assert_eq!(
            mask("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"),
            "Authorization: [SECRET]"
        );
    }

    // --- tricky true negatives: what the checksum validators reject ---

    #[test]
    fn does_not_mask_near_misses() {
        let negatives = [
            "order 1234567890",         // bare 10 digits: not a phone (no separators)
            "card 4111-1111-1111-1112", // Luhn fails
            "version 1.2.3.4.5",        // not an IPv4
            "time 12:30:45",            // not an IPv6
            "id 123456789 mismatch",    // 9 digits failing ABA? -> see note below
            "ssn 666-45-6789",          // invalid area
            "GB00WEST12345698765432",   // IBAN mod-97 fails
        ];
        for text in negatives {
            // "id 123456789": fails the ABA checksum (3*12 + 7*15 + 1*18 = 159, 159 % 10 != 0)
            assert_eq!(engine().mask_text(text), None, "false positive on: {text}");
        }
    }

    /// A dotted quad in a version context is a version, not an address: as
    /// part of a longer dotted number, touching a `v`, after `version ` or
    /// `version=`; or, where the context could as well hold an address (after
    /// a pip comparison operator, `v=`, `ver: `, a name and `/` or `-`, or
    /// before a pre-release suffix), when its numbers read as a version. An
    /// address in free text, in a filter on an address field or in a URL path
    /// is still an address.
    #[test]
    fn version_strings_are_not_ip_addresses() {
        for text in [
            "pkg numpy==1.26.4.1",
            "requires torch>=2.1.0.1",
            "pin foo~=1.2.3.4, bar!=1.2.3.4, baz<=1.2.3.4",
            "build v1.2.3.4",
            "v=1.2.3.4",
            "version 1.2.3.4",
            "Version=10.0.0.1",
            "Server: nginx/1.2.3.4",
            "ua python-requests/2.31.0.1",
            "openssl-1.1.1.1",
            "name-1.2.3.4",
            "ver: 1.2.3.4",
            "release 2.10.1.0-rc1",
            "build 1.2.3.4+local",
            "release 1.2.3.4.5",
            "release 1.2.3.4.1000",
            "release 1000.1.2.3.4",
        ] {
            let mut r = Report::default();
            let _masked = engine().mask_text_into(text, &mut r);
            assert!(!r.rules.contains(&Rule::IpAddress), "{text}: {r:?}");
        }
        for (text, masked) in [
            ("host 10.0.0.5 down", "host [IP_ADDRESS] down"),
            (
                "server 10.1.2.3, build 1.2.3.4",
                "server [IP_ADDRESS], build [IP_ADDRESS]",
            ),
            ("ssh root@8.8.8.8", "ssh root@[IP_ADDRESS]"),
            ("ends at 10.0.0.1.", "ends at [IP_ADDRESS]."),
            ("port 10.0.0.1:8080", "port [IP_ADDRESS]:8080"),
            ("dev 10.0.0.1", "dev [IP_ADDRESS]"),
            ("nginx v 203.0.113.9", "nginx v [IP_ADDRESS]"),
            ("version: 203.0.113.9", "version: [IP_ADDRESS]"),
            ("subversion 10.0.0.1", "subversion [IP_ADDRESS]"),
            ("ip.addr==192.168.1.1", "ip.addr==[IP_ADDRESS]"),
            ("filter src_ip>=10.0.0.1", "filter src_ip>=[IP_ADDRESS]"),
            ("GET /hosts/10.0.0.5", "GET /hosts/[IP_ADDRESS]"),
            ("see http://10.0.0.5/x", "see http://[IP_ADDRESS]/x"),
            ("ops@10.0.0.5", "ops@[IP_ADDRESS]"),
            ("range 10.0.0.1-10.0.0.9", "range [IP_ADDRESS]-[IP_ADDRESS]"),
            // A name and `-` or `/`, a comparison or `v=` holds an address as
            // readily as a version: a public address there is still masked.
            (
                "login from client-203.0.113.9 failed",
                "login from client-[IP_ADDRESS] failed",
            ),
            ("peer/203.0.113.9 timed out", "peer/[IP_ADDRESS] timed out"),
            (
                "blocked attacker at host-198.51.100.7",
                "blocked attacker at host-[IP_ADDRESS]",
            ),
            (
                "example.com/203.0.113.9 returned 500",
                "example.com/[IP_ADDRESS] returned 500",
            ),
            ("remote==203.0.113.9", "remote==[IP_ADDRESS]"),
            ("?v=203.0.113.9", "?v=[IP_ADDRESS]"),
            ("peer/10.0.0.5", "peer/[IP_ADDRESS]"),
            ("ver: 203.0.113.9", "ver: [IP_ADDRESS]"),
            ("host 10.0.0.5-east", "host [IP_ADDRESS]-east"),
        ] {
            assert_eq!(mask(text), masked, "{text}");
        }
    }

    /// A card, SSN or routing number glued into a longer token is part of an
    /// id or a number, not a value of its own: joined by a letter, digit or
    /// `_`, joined by a `-` or `.` to a hex group with a letter in it, to
    /// another group of the same UUID, or going on a dotted number. The same
    /// digits standing alone, joined to a word, or beside a plain digit group
    /// outside a UUID (an expiry, a CVV, an account number, a year, cents)
    /// are still masked.
    #[test]
    fn digits_inside_a_longer_token_are_not_cards_ssns_or_routing_numbers() {
        for text in [
            "id ord_4111111111111111x",
            "uuid 41111111-1111-1111-1234-567890123456",
            "uuid 48620579-8682-4828-a0e5-0454f31af317",
            "id 4111111111111111-a0e5",
            "ref 021000021_b",
            "ref 2.021000021",
        ] {
            let mut r = Report::default();
            let _masked = engine().mask_text_into(text, &mut r);
            for rule in [Rule::CreditCard, Rule::UsSsn, Rule::UsBankRouting] {
                assert!(!r.rules.contains(&rule), "{text}: {r:?}");
            }
        }
        for (text, masked) in [
            ("order 4111111111111111", "order ****-****-****-1111"),
            ("card: 4111-1111-1111-1111.", "card: ****-****-****-1111."),
            (
                "\"pan\":\"4111111111111111\"",
                "\"pan\":\"****-****-****-1111\"",
            ),
            ("id 123-45-6789", "id [SSN]"),
            ("(123-45-6789)", "([SSN])"),
            ("acct 021000021", "acct [BANK_ROUTING]"),
            ("price -4111111111111111", "price -****-****-****-1111"),
            ("card-4111111111111111", "card-****-****-****-1111"),
            ("SSN-123-45-6789", "SSN-[SSN]"),
            ("SSN 123-45-6789.Thanks", "SSN [SSN].Thanks"),
            ("x-4111 1111 1111 1111", "x-****-****-****-1111"),
            ("id.4111111111111111", "id.****-****-****-1111"),
            ("4111111111111111.json", "****-****-****-1111.json"),
            ("routing-021000021", "routing-[BANK_ROUTING]"),
            ("id ord-4111111111111111", "id ord-****-****-****-1111"),
            ("id 4111111111111111-a", "id ****-****-****-1111-a"),
            ("ref INV-123-45-6789", "ref INV-[SSN]"),
            ("ref acct-021000021", "ref acct-[BANK_ROUTING]"),
            (
                "card 4111111111111111-1227",
                "card ****-****-****-1111-1227",
            ),
            (
                "card 4111-1111-1111-1111-123",
                "card ****-****-****-1111-123",
            ),
            (
                "paid with 4111111111111111.00",
                "paid with ****-****-****-1111.00",
            ),
            ("row 7-4111111111111111", "row 7-****-****-****-1111"),
            ("ach 021000021-123456789", "ach [BANK_ROUTING]-123456789"),
            ("ssn 123-45-6789-1990", "ssn [SSN]-1990"),
            ("ref 123-45-6789-01", "ref [SSN]-01"),
        ] {
            assert_eq!(mask(text), masked, "{text}");
        }
    }

    #[test]
    fn masks_south_korean_mobile_numbers() {
        for (text, masked) in [
            ("연락처 010-2481-7730", "연락처 [PHONE]"),
            ("010-2481-7730으로 연락주세요", "[PHONE]으로 연락주세요"),
            ("번호:010-2481-7730.", "번호:[PHONE]."),
            ("phone 01024817730", "phone [PHONE]"),
            ("phone 010 2481 7730", "phone [PHONE]"),
            ("phone 010.2481.7730", "phone [PHONE]"),
            ("call +82 10-2481-7730", "call [PHONE]"),
            ("call +82-10-2481-7730", "call [PHONE]"),
            ("call +821024817730", "call [PHONE]"),
            ("call +82 010-2481-7730", "call [PHONE]"),
            ("call +82 (0)10-2481-7730", "call [PHONE]"),
            ("call (+82) 10-2481-7730", "call [PHONE]"),
            ("call (+82)10-2481-7730", "call [PHONE]"),
            ("call +82)10-2481-7730", "call [PHONE]"),
            ("call 0082-10-2481-7730", "call [PHONE]"),
            ("call 0082 10 2481 7730", "call [PHONE]"),
            ("call (010) 2481-7730", "call [PHONE]"),
            ("call (010)2481-7730", "call [PHONE]"),
            ("call 010)2481-7730", "call [PHONE]"),
            ("call 010 - 2481 - 7730", "call [PHONE]"),
            ("call 010\u{2013}2481\u{2013}7730", "call [PHONE]"),
            ("call 010\u{2011}2481\u{2011}7730", "call [PHONE]"),
            // Only a pair of parentheses is taken into the match.
            ("(010-2481-7730)", "([PHONE])"),
            ("(+82 10-2481-7730)", "([PHONE])"),
            ("legacy 011-234-5678", "legacy [PHONE]"),
            ("legacy 016-2345-6789", "legacy [PHONE]"),
            ("legacy 019 234 5678", "legacy [PHONE]"),
            (r#"{"phone": "01024817730"}"#, r#"{"phone": "[PHONE]"}"#),
            ("ONCALL_PHONE=010-2481-7730\n", "ONCALL_PHONE=[PHONE]\n"),
        ] {
            let mut r = Report::default();
            assert_eq!(
                engine().mask_text_into(text, &mut r).as_deref(),
                Some(masked),
                "{text}"
            );
            assert_eq!(r.rules, vec![Rule::PhoneNumber], "{text}");
        }
    }

    #[test]
    fn digit_runs_that_are_not_korean_mobile_numbers_are_kept() {
        for text in [
            "date 2026-10-10",
            "ts 2026-10-10T01:01:01Z",
            "order 1234567890",
            "version 1.2.3.4.5",
            "build 2010.1234.5678",
            // Ten digits under 010 with no separators: no number has that.
            "id 0101234567",
            // A three-digit middle runs on only as an id.
            "id 0112345678",
            // The number inside a longer run of digits or letters.
            "id 010-2481-77301",
            "id 1010-2481-7730",
            "id a01024817730",
            "id 01024817730_x",
            "uuid 01012345-6789-0123-4567-890123456789",
            // Not a mobile prefix.
            "landline 02-2481-7730",
            "prefix 012-2481-7730",
        ] {
            let mut r = Report::default();
            let _masked = engine().mask_text_into(text, &mut r);
            assert!(!r.rules.contains(&Rule::PhoneNumber), "{text}: {r:?}");
        }
    }

    /// Only North American and South Korean mobile numbers are phone
    /// numbers to the rules; the documentation lists these other forms as
    /// limits. If one starts being masked, the documentation changes too.
    #[test]
    fn a_phone_number_in_another_form_is_a_documented_limit() {
        for text in [
            "call +44 20 7946 0958",
            "call 02-2481-7730",
            "call 82-10-2481-7730",
            "call \u{ff10}\u{ff11}\u{ff10}-\u{ff12}\u{ff14}\u{ff18}\u{ff11}-\u{ff17}\u{ff17}\u{ff13}\u{ff10}",
            "file user_01024817730.json",
            "tel_01024817730",
        ] {
            assert_eq!(engine().mask_text(text), None, "{text}");
        }
    }

    // --- overlap / determinism / idempotency ---

    #[test]
    fn leftmost_longest_wins_and_no_rescan() {
        // "Bearer sk-..." — bearer starts earlier and swallows the sk- token.
        let out = mask("Bearer sk-abcdefghijklmnop1234");
        assert_eq!(out, "[SECRET]");
        assert_eq!(out.matches("[SECRET]").count(), 1);
    }

    #[test]
    fn trailing_digit_bleed_still_masks_the_card() {
        // The greedy candidate "4111-1111-1111-1111 999" fails Luhn; the
        // engine must shrink to the embedded 16-digit PAN, not ship it raw.
        assert_eq!(
            mask("card 4111-1111-1111-1111 999 total"),
            "card ****-****-****-1111 999 total"
        );
    }

    #[test]
    fn leading_digit_bleed_still_masks_the_card() {
        // The candidate "5 4111 1111 1111 1111" (IP octet bleed) fails Luhn;
        // the start+1 retry must still find and mask the real PAN.
        assert_eq!(
            mask("john@x.io 10.0.0.5 4111 1111 1111 1111"),
            "[EMAIL] [IP_ADDRESS] ****-****-****-1111"
        );
    }

    #[test]
    fn bleed_on_both_sides_still_masks_the_card() {
        // Leading "5 " forces the start+1 retry; the retried candidate then
        // bleeds into the trailing "999" and needs the shrink probe too.
        assert_eq!(
            mask("5 4111 1111 1111 1111 999"),
            "5 ****-****-****-1111 999"
        );
    }

    #[test]
    fn luhn_passing_bleed_never_ships_the_pan() {
        // The leading "0" from the IP octet keeps the Luhn sum valid, but that
        // candidate is joined to the address by its dot, so the retry one
        // character on finds the card itself. Whatever the path, the raw PAN
        // never ships.
        let out = mask("10.0.0.0 4111 1111 1111 1111");
        assert!(!out.contains("4111"), "raw PAN leaked: {out}");
        assert_eq!(out, "[IP_ADDRESS] ****-****-****-1111");
    }

    #[test]
    fn masking_is_deterministic_and_idempotent() {
        let text = "john@x.io 10.0.0.5 4111 1111 1111 1111";
        let once = mask(text);
        assert_eq!(once, mask(text));
        // placeholders must not re-match anything
        assert_eq!(engine().mask_text(&once), None);
    }

    // --- config surface ---

    #[test]
    fn disabled_category_passes_through() {
        let e = PiiEngine::new(&["ip_address".to_string()]).unwrap();
        let out = e.mask_text("john@x.io at 10.0.0.5").unwrap();
        assert_eq!(out, "[EMAIL] at 10.0.0.5");
    }

    #[test]
    fn unknown_disabled_category_is_a_hard_error() {
        let err = PiiEngine::new(&["not_a_category".to_string()]).unwrap_err();
        assert!(err.to_string().contains("not_a_category"));
    }

    #[test]
    fn engine_cache_returns_same_instance() {
        let a = engine_for(&["email".to_string()]).unwrap();
        let b = engine_for(&["email".to_string()]).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn url_userinfo_is_redacted_and_counted_once() {
        let e = PiiEngine::userinfo_only();
        let mut r = Report::default();
        let out = e
            .mask_text_into("GET http://user:SECRET@host/p and ftp://tok@h/x", &mut r)
            .unwrap();
        assert_eq!(
            out,
            "GET http://REDACTED:REDACTED@host/p and ftp://REDACTED@h/x"
        );
        assert_eq!(r.count, 2);
        assert_eq!(r.rules, vec![Rule::UrlUserinfo]);
        assert_eq!(
            e.mask_text(&out),
            None,
            "redacted userinfo is not replaced again"
        );
        assert_eq!(
            e.mask_text("http://bob:p@ss@h/p").unwrap(),
            "http://REDACTED:REDACTED@h/p"
        );
        // With every rule on, the e-mail rule must not eat the host.
        assert_eq!(
            engine()
                .mask_text("https://alice:PW1@api.example.com/v1?x=1")
                .unwrap(),
            "https://REDACTED:REDACTED@api.example.com/v1?x=1"
        );
        let glued = e.mask_text("http://h:80http://u:pw@h/p").unwrap();
        assert_eq!(glued, "http://h:80http://REDACTED:REDACTED@h/p");
        // An e-mail in a path is not userinfo: a `/` comes before the `@`.
        assert_eq!(e.mask_text("http://h/u/john@x.com"), None);
    }

    #[test]
    fn provider_token_shapes_are_masked() {
        // Built from pieces so no whole provider token sits in the source,
        // where a repository secret scanner would stop the push.
        let cases = [
            ["ASIA", "IOSFODNN7EXAMPLE"].concat(),
            ["sk_", "live_", "abcdefghijklmnop1234"].concat(),
            ["ya29", ".", "a0AfH6SMBxxxxxxxxxxxxxxxxxxxx"].concat(),
            ["glpat", "-", "xxxxxxxxxxxxxxxxxxxx"].concat(),
            ["hf_", "abcdefghijklmnopqrstuvwxyzABCDEF"].concat(),
        ];
        for token in cases {
            assert_eq!(mask(&format!("t {token} ok")), "t [SECRET] ok", "{token}");
        }
        assert_eq!(
            mask("curl -H 'Authorization: Bearer abc123' x"),
            "curl -H 'Authorization: [SECRET]' x"
        );
        assert_eq!(
            mask("authorization:Basic YWxpY2U6aHVudGVyMg=="),
            "authorization:[SECRET]"
        );
    }

    #[test]
    fn a_url_argument_does_not_swallow_its_neighbours() {
        let e = engine();
        // `&` ends an argument, so the URL has no userinfo and the password
        // after it is masked by its name.
        assert_eq!(
            e.mask_text("redirect=http://app.example.com&password=p@ssw0rd&x=1")
                .unwrap(),
            "redirect=http://app.example.com&password=[SECRET]&x=1"
        );
        assert_eq!(
            e.mask_text("callback=http://myhost.example.com&user=john@corp.com&page=2")
                .unwrap(),
            "callback=http://myhost.example.com&user=[EMAIL]&page=2"
        );
        assert_eq!(
            PiiEngine::userinfo_only().mask_text("cb=http://h&mail=a@b.co&page=2"),
            None
        );
    }

    #[test]
    fn a_byte_that_is_not_utf8_inside_a_secret_does_not_split_it() {
        let e = engine();
        let mut r = Report::default();
        assert_eq!(
            e.mask_bytes_into(b"password=p\xe4ssword&q=seoul", &mut r)
                .unwrap(),
            b"password=[SECRET]&q=seoul".to_vec()
        );
        assert_eq!(
            e.mask_text("url=https%3A%2F%2Fh%2Fs%3Fq%3Dcaf%E9%26api_key%3DSEC1&page=2")
                .unwrap(),
            "url=https%3A%2F%2Fh%2Fs%3Fq%3Dcaf%E9%26api_key%3D[SECRET]&page=2"
        );
        assert_eq!(r.count, 1);
    }

    #[test]
    fn head_nouns_are_not_prefixes() {
        let e = engine();
        assert_eq!(
            e.mask_text("pass_through=true&otp_length=6&cvv_required=false&db_pass=S1&sms_otp=S2"),
            Some("pass_through=true&otp_length=6&cvv_required=false&db_pass=[SECRET]&sms_otp=[SECRET]".into())
        );
        assert_eq!(
            e.mask_text("curl -H 'Authorization: token abc123'")
                .unwrap(),
            "curl -H 'Authorization: [SECRET]'"
        );
    }

    #[test]
    fn a_secret_right_after_a_json_escape_is_masked_under_its_own_name() {
        // A log or a tool result carried as a JSON string: the same lines a
        // plain body masks, with each line break written `\n` or `\t`.
        let e = engine();
        let mut r = Report::default();
        let body = r#"{"content": "log\npassword=hunter2xyz\n\tsecret=s3cr3tvalue\nGITHUB_TOKEN=abcdefg12345\n"}"#;
        assert_eq!(
            e.mask_text_into(body, &mut r).unwrap(),
            r#"{"content": "log\npassword=[SECRET]\n\tsecret=[SECRET]\nGITHUB_TOKEN=[SECRET]\n"}"#
        );
        assert_eq!(r.count, 3);
        assert_eq!(r.names, vec!["password", "secret", "GITHUB_TOKEN"]);
        assert_eq!(r.rules, vec![Rule::SecretWord, Rule::SecretLastWord]);
    }

    #[test]
    fn reported_names_are_capped_but_the_count_is_not() {
        let e = engine();
        let mut body = String::new();
        for i in 0..40 {
            write!(body, "{{\"svc{i}_token\": \"v{i}\"}} ")
                .expect("writing to a String cannot fail");
        }
        let mut r = Report::default();
        e.mask_text_into(&body, &mut r).unwrap();
        assert_eq!(r.count, 40);
        assert_eq!(r.names.len(), MAX_REPORTED_NAMES);
        assert_eq!(r.names[0], "svc0_token");
    }

    #[test]
    fn no_match_returns_none() {
        assert_eq!(engine().mask_text("hello plain world"), None);
    }
}
