//! PII masking engine — design doc 2026-07-07-phase3-pii-masking.
//!
//! Detection is fully deterministic: regex proposes, checksum validators
//! confirm (design §5.1). No ML/entropy heuristics.

mod patterns;
mod walk;

pub use walk::mask_envelope;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use regex::Regex;

use patterns::{Replacement, BUILTINS, CATEGORIES};

/// Fail-closed placeholder — what a span's text becomes when masking itself
/// failed and we refuse to ship the original (design §8).
pub const PII_FILTER_ERROR: &str = "[PII_FILTER_ERROR]";

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
    /// Rescan policy after a validator rejection. A rejected candidate
    /// normally suppresses its whole span — that is how the `{3,}` IPv4 tail
    /// rejects version strings like "1.2.3.4.5" outright. Credit cards are
    /// the exception: the separator-tolerant regex can greedily bleed into a
    /// neighboring token (e.g. the last octet of a preceding IP), and that
    /// spurious Luhn failure must not swallow the valid card starting inside
    /// the rejected span. Luhn re-confirms every retried candidate.
    retry_on_reject: bool,
}

/// Compiled pattern set. Build once per config via `engine_for` (design §4.3).
#[derive(Debug)]
pub struct PiiEngine {
    active: Vec<Compiled>,
}

impl PiiEngine {
    /// Build an engine with `disabled` categories removed.
    /// Unknown category names are a hard error — the FFI contract is fail-loud.
    pub fn new(disabled: &[String]) -> Result<PiiEngine, PiiError> {
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
                retry_on_reject: def.category == "credit_card",
            });
        }
        Ok(PiiEngine { active })
    }

    /// Mask every validated match. `None` means "no change".
    /// Single pass over the original text (retry-on-reject patterns may
    /// re-probe within a rejected candidate); replaced regions are never
    /// rescanned, so placeholders cannot re-match (design §5.2).
    pub fn mask_text(&self, text: &str) -> Option<String> {
        let mut hits: Vec<(usize, usize, String)> = Vec::new();
        for p in &self.active {
            let mut at = 0;
            while let Some(m) = p.regex.find_at(text, at) {
                if p.validator.is_none_or(|v| v(m.as_str())) {
                    hits.push((m.start(), m.end(), apply(p.replacement, m.as_str())));
                    at = m.end();
                } else if let Some((end, rep)) = p
                    .retry_on_reject
                    .then(|| shrink_to_valid(p, text, m.start(), m.end()))
                    .flatten()
                {
                    // The greedy candidate bled into a trailing digit run; a
                    // shorter end at the same start re-validated (e.g. the
                    // 16-digit PAN inside "4111 1111 1111 1111 999").
                    hits.push((m.start(), end, rep));
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
        if hits.is_empty() {
            return None;
        }
        // leftmost first; on ties the longer match wins (§5.2)
        hits.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut out = String::with_capacity(text.len());
        let mut pos = 0usize;
        for (start, end, rep) in hits {
            if start < pos {
                continue; // overlapped by an earlier winner
            }
            out.push_str(&text[pos..start]);
            out.push_str(&rep);
            pos = end;
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
            format!("****-****-****-{}", &digits[digits.len() - 4..])
        }
    }
}

/// Process-wide engine cache keyed by the sorted disabled list.
/// Regex compilation is the expensive part — pay it once per config (§4.3).
pub fn engine_for(disabled: &[String]) -> Result<Arc<PiiEngine>, PiiError> {
    static CACHE: OnceLock<Mutex<HashMap<Vec<String>, Arc<PiiEngine>>>> = OnceLock::new();
    let mut key: Vec<String> = disabled.to_vec();
    key.sort();
    key.dedup();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("pii engine cache poisoned");
    if let Some(e) = guard.get(&key) {
        return Ok(e.clone());
    }
    let engine = Arc::new(PiiEngine::new(&key)?);
    guard.insert(key, engine.clone());
    Ok(engine)
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
    fn no_match_returns_none() {
        assert_eq!(engine().mask_text("hello plain world"), None);
    }
}
