//! PII masking engine — design doc 2026-07-07-phase3-pii-masking.
//!
//! Detection is fully deterministic: regex proposes, checksum validators
//! confirm (design §5.1). No ML/entropy heuristics. Two families of rules feed
//! one replacement pass: VALUE rules recognise a value by its shape
//! (`patterns.rs`), NAME rules recognise it by the argument it is passed as
//! (`names.rs`). Every replacement is recorded against the rule that made it,
//! so a span can say what was masked and why.

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
use std::sync::{Arc, Mutex, OnceLock};

use regex::Regex;

use patterns::{Replacement, BUILTINS, CATEGORIES};

/// Fail-closed placeholder — what a span's text becomes when masking itself
/// failed and we refuse to ship the original (design §8).
pub const PII_FILTER_ERROR: &str = "[PII_FILTER_ERROR]";

/// What replaces the credentials in `scheme://user:password@host` — the value
/// OTel's semantic conventions prescribe for `url.full`.
pub const USERINFO_REDACTED: &str = "REDACTED:REDACTED";

/// Most distinct argument names one span records. The count and the rules are
/// never capped; only this list is, so a body with thousands of secret-named
/// fields cannot grow a span without bound.
pub const MAX_REPORTED_NAMES: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum PiiError {
    #[error("unknown PII category: {0:?} (known: email, phone_number, credit_card, us_ssn, ip_address, us_bank_routing, iban, secret)")]
    UnknownCategory(String),
    #[error("invalid builtin pattern for category {category}: {message}")]
    InvalidPattern { category: String, message: String },
}

#[derive(Debug)]
struct Compiled {
    regex: Regex,
    validator: Option<fn(&str) -> bool>,
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

/// Compiled pattern set. Build once per config via `engine_for` (design §4.3).
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
/// everything up to the LAST `@` before any path, query, fragment, whitespace
/// or quoting byte — a raw password may itself contain `@`. A scanner rather
/// than a regex so there is no compile step that could fail on the export
/// path.
fn userinfo_hits(text: &str, hits: &mut Vec<Hit>) {
    let b = text.as_bytes();
    let mut from = 0;
    while let Some(off) = text[from..].find("://") {
        let sep = from + off;
        from = sep + 3;
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
        let userinfo = &text[start..end];
        let replacement = if userinfo.contains(':') {
            USERINFO_REDACTED
        } else {
            "REDACTED"
        };
        if userinfo == replacement {
            continue; // already redacted — not a second replacement
        }
        hits.push(Hit {
            start,
            end,
            replacement: replacement.to_string(),
            rule: Rule::UrlUserinfo,
            name: None,
        });
    }
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
    pub fn mask_text(&self, text: &str) -> Option<String> {
        self.mask_text_into(text, &mut Report::default())
    }

    /// `mask_text`, recording each replacement into `report`.
    /// Single pass over the original text (retry-on-reject patterns may
    /// re-probe within a rejected candidate); replaced regions are never
    /// rescanned, so placeholders cannot re-match (design §5.2).
    pub fn mask_text_into(&self, text: &str, report: &mut Report) -> Option<String> {
        let mut hits: Vec<Hit> = Vec::new();
        for p in &self.active {
            let mut at = 0;
            while let Some(m) = p.regex.find_at(text, at) {
                if p.validator.is_none_or(|v| v(m.as_str())) {
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
                    at = m.start() + text[m.start()..].chars().next().map_or(1, char::len_utf8);
                } else {
                    // Rejection consumes the span (find_iter semantics).
                    at = m.end();
                }
                if at >= text.len() {
                    break;
                }
            }
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
        let before = hits.len();
        userinfo_hits(text, &mut hits);
        if hits.len() > before {
            // A userinfo is the credential and nothing else: a value rule
            // that starts inside it (`PW@api.example.com` read as an e-mail)
            // would otherwise run its replacement over the host.
            let spans: Vec<(usize, usize)> =
                hits[before..].iter().map(|h| (h.start, h.end)).collect();
            hits.retain(|h| {
                h.rule == Rule::UrlUserinfo
                    || !spans.iter().any(|&(s, e)| h.start >= s && h.start <= e)
            });
        }
        if hits.is_empty() {
            return None;
        }
        // leftmost first; on ties the longer match wins (§5.2)
        hits.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        let mut out = String::with_capacity(text.len());
        let mut pos = 0usize;
        for h in hits {
            if h.start < pos {
                // Overlapped by an earlier winner. An overlapped-but-validated
                // hit still masks its remainder — never emit a validated
                // span's tail raw (e.g. a Luhn-passing card that bled
                // backwards into a preceding IP match).
                if h.end > pos {
                    out.push_str(&h.replacement);
                    pos = h.end;
                    report.record(h.rule, h.name.as_deref());
                }
                continue;
            }
            out.push_str(&text[pos..h.start]);
            out.push_str(&h.replacement);
            pos = h.end;
            report.record(h.rule, h.name.as_deref());
        }
        out.push_str(&text[pos..]);
        Some(out)
    }
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
    let validator = p.validator?;
    // The matched span is pure ASCII (digits and `[ -]` separators), so byte
    // positions inside it are always char boundaries.
    let bytes = text.as_bytes();
    for e in (start + 1..end).rev() {
        if !bytes[e - 1].is_ascii_digit() || bytes[e].is_ascii_digit() {
            continue; // candidate must end a digit run, not split one
        }
        let cand = &text[start..e];
        if cand.bytes().filter(u8::is_ascii_digit).count() < 13 {
            break; // shorter probes only lose more digits
        }
        if validator(cand) {
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
            format!("****-****-****-{}", &digits[digits.len() - 4..])
        }
        Replacement::AfterColon(label) => {
            // The pattern guarantees a colon; the name and the spacing after
            // it are the writer's and stay as written.
            let keep = matched.find(':').map_or(0, |c| {
                c + 1 + matched[c + 1..].len() - matched[c + 1..].trim_start().len()
            });
            format!("{}{label}", &matched[..keep])
        }
    }
}

/// Process-wide engine cache keyed by the sorted disabled list.
/// Regex compilation is the expensive part — pay it once per config (§4.3).
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
    let mut guard = cache.lock().expect("pii engine cache poisoned");
    if let Some(e) = guard.get(&key) {
        return Ok(e.clone());
    }
    let engine = Arc::new(PiiEngine::with_names(&key.0, &key.1, &key.2)?);
    guard.insert(key, engine.clone());
    Ok(engine)
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

    // --- tricky true negatives: the promise of §5.1's validators ---

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

    // --- overlap / determinism / idempotency (§5.2) ---

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
        // leading "0" from the IP octet keeps the Luhn sum valid -> the bled candidate
        // is accepted, overlaps the IP hit, and must still mask its remainder.
        // Output is deterministic: the IP hit consumes "10.0.0.0" (0..8), the
        // card hit (7..28, overlapping) contributes only its replacement.
        let out = mask("10.0.0.0 4111 1111 1111 1111");
        assert!(!out.contains("4111"), "raw PAN leaked: {out}");
        assert_eq!(out, "[IP_ADDRESS]****-****-****-1111");
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
        assert!(std::sync::Arc::ptr_eq(&a, &b));
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
    fn reported_names_are_capped_but_the_count_is_not() {
        let e = engine();
        let body: String = (0..40)
            .map(|i| format!("{{\"svc{i}_token\": \"v{i}\"}} "))
            .collect();
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
