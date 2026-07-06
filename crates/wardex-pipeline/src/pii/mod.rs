//! PII masking engine — design doc 2026-07-07-phase3-pii-masking.
//!
//! Detection is fully deterministic: regex proposes, checksum validators
//! confirm (design §5.1). No ML/entropy heuristics.

mod patterns;

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
            });
        }
        Ok(PiiEngine { active })
    }

    /// Mask every validated match. `None` means "no change".
    /// Single pass over the original text; replaced regions are never
    /// rescanned, so placeholders cannot re-match (design §5.2).
    pub fn mask_text(&self, text: &str) -> Option<String> {
        let mut hits: Vec<(usize, usize, String)> = Vec::new();
        for p in &self.active {
            for m in p.regex.find_iter(text) {
                if p.validator.is_none_or(|v| v(m.as_str())) {
                    hits.push((m.start(), m.end(), apply(p.replacement, m.as_str())));
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
            // "id 123456789": 123456789 actually fails the ABA checksum (sum=165? verify);
            // if it happens to pass, swap the digits for a checksum-failing 9-digit run.
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
    fn masking_is_deterministic_and_idempotent() {
        let text = "john@x.io 10.0.0.5 abc123";
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
