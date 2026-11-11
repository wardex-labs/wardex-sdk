//! Declarative PII pattern registry: the VALUE rules, one row per pattern.
//!
//! Each row is one detection pattern; several rows may share a category
//! (secret prefixes, IPv4/IPv6). Adding a category = adding rows here plus a
//! `PIICategory` member on the Python side — engine/FFI/marshalling untouched.

/// What replaces a validated match.
#[derive(Debug)]
pub(crate) enum Replacement {
    /// Fixed label such as `[EMAIL]`.
    Label(&'static str),
    /// PCI display rule: keep only the last four digits.
    CardLast4,
    /// A header written as text: the name through its colon stays, the value
    /// becomes the label (`Authorization: [SECRET]`).
    AfterColon(&'static str),
}

pub(crate) struct PatternDef {
    pub category: &'static str,
    pub regex: &'static str,
    pub validator: Option<fn(&str) -> bool>,
    /// Confirms a match against the text around it: `(text, start, end)` of
    /// the match. The regex crate has no look-around, and a validator sees
    /// only the match, so whether the same digits are a value or a piece of
    /// a longer token is decided here.
    pub context: Option<fn(&str, usize, usize) -> bool>,
    pub replacement: Replacement,
    /// Rescan policy after a validator rejection (see `Compiled::retry_on_reject`
    /// in `mod.rs`). Declarative per-row so adding a category never requires
    /// touching the engine.
    pub retry_on_reject: bool,
}

/// Category identifiers — the FFI contract with the Python `PIICategory` enum.
pub(crate) const CATEGORIES: [&str; 8] = [
    "email",
    "phone_number",
    "credit_card",
    "us_ssn",
    "ip_address",
    "us_bank_routing",
    "iban",
    "secret",
];

/// A separator inside a South Korean mobile number: a space, a dot, a hyphen
/// or a typographic dash (U+2010 to U+2013), or a hyphen or dash with a
/// space on each side.
macro_rules! kr_sep {
    () => {
        r"(?:[ .\-\x{2010}-\x{2013}]| [\-\x{2010}-\x{2013}] )"
    };
}

pub(crate) static BUILTINS: &[PatternDef] = &[
    PatternDef {
        category: "email",
        regex: r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        validator: None,
        context: None,
        replacement: Replacement::Label("[EMAIL]"),
        retry_on_reject: false,
    },
    // NANP with mandatory separators (or +1): bare 10-digit runs are not matched
    // to avoid masking ids and timestamps, which are bare digit runs too.
    PatternDef {
        category: "phone_number",
        regex: r"(?:\+1[ .-]?)?\(?[2-9][0-9]{2}\)?[ .-][0-9]{3}[ .-][0-9]{4}\b",
        validator: None,
        context: None,
        replacement: Replacement::Label("[PHONE]"),
        retry_on_reject: false,
    },
    // South Korean mobile numbers: `010`, and the `011` and `016`-`019`
    // prefixes still in service. A four-digit middle group may run on
    // without separators (`01024817730` is how forms and databases keep
    // one); a three-digit middle, which only older numbers have, needs both
    // separators, since a bare ten-digit run is an id far more often than a
    // number. The country code is `+82`, `(+82)` or `0082`, after which the
    // leading `0` is dropped or written back as `0` or `(0)`; a prefix may
    // sit in parentheses (`(010) 2481-7730`), and only a pair of them is
    // taken into the match, so `(010-2481-7730)` keeps both. The boundaries
    // are ASCII ones, so a Korean particle written straight after the number
    // (`010-2481-7730으로`) still ends it while a longer run of digits or
    // letters, an underscore included, does not.
    PatternDef {
        category: "phone_number",
        regex: concat!(
            r"(?:(?:\(\+82\)|\+82\)?|(?-u:\b)0082)",
            kr_sep!(),
            r"?(?:\(0\)",
            kr_sep!(),
            r"?|0)?|(?-u:\b)0)1[016789]\)?(?:",
            kr_sep!(),
            r"?[0-9]{4}",
            kr_sep!(),
            r"?|",
            kr_sep!(),
            r"[0-9]{3}",
            kr_sep!(),
            r")[0-9]{4}(?-u:\b)|\((?-u:\b)01[016789]\)",
            kr_sep!(),
            r"?[0-9]{4}",
            kr_sep!(),
            r"?[0-9]{4}(?-u:\b)",
        ),
        validator: None,
        context: None,
        replacement: Replacement::Label("[PHONE]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "credit_card",
        regex: r"\b[0-9](?:[ -]?[0-9]){12,18}\b",
        validator: Some(luhn_valid),
        context: Some(stands_alone),
        replacement: Replacement::CardLast4,
        retry_on_reject: true,
    },
    PatternDef {
        category: "us_ssn",
        regex: r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
        validator: Some(ssn_valid),
        context: Some(stands_alone),
        replacement: Replacement::Label("[SSN]"),
        retry_on_reject: false,
    },
    // The {3,} tail makes version strings ("1.2.3.4.5") match as a whole and
    // then fail Ipv4Addr parsing — rejecting the false positive outright.
    PatternDef {
        category: "ip_address",
        regex: r"\b[0-9]{1,3}(?:\.[0-9]{1,3}){3,}\b",
        validator: Some(ipv4_valid),
        context: Some(ipv4_in_context),
        replacement: Replacement::Label("[IP_ADDRESS]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "ip_address",
        regex: r"\b(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]*[0-9A-Fa-f]\b",
        validator: Some(ipv6_valid),
        context: None,
        replacement: Replacement::Label("[IP_ADDRESS]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "us_bank_routing",
        regex: r"\b[0-9]{9}\b",
        validator: Some(aba_valid),
        context: Some(stands_alone),
        replacement: Replacement::Label("[BANK_ROUTING]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "iban",
        regex: r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
        validator: Some(iban_valid),
        context: None,
        replacement: Replacement::Label("[IBAN]"),
        retry_on_reject: false,
    },
    // Secrets: fixed prefixes only. Deterministic, no entropy guessing, so the
    // same text is always masked the same way.
    PatternDef {
        category: "secret",
        regex: r"\bsk-[A-Za-z0-9_-]{16,}", // covers sk- and sk-ant-
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", // long-term and STS key ids
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bgh[pousr]_[A-Za-z0-9]{36}\b",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bxox[abpos]-[A-Za-z0-9-]{10,}",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bAIza[0-9A-Za-z_-]{35}\b",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}=*",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    // Stripe secret and restricted keys (`sk_live_…`, `rk_test_…`).
    PatternDef {
        category: "secret",
        regex: r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    // Google OAuth access tokens, GitLab personal access tokens, Hugging Face
    // tokens.
    PatternDef {
        category: "secret",
        regex: r"\b(?:ya29\.[A-Za-z0-9_-]{20,}|glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,})",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    // A Basic, Bearer or `token` credential written as a header in free text (a curl
    // command in a tool's arguments), whatever its length: the header names
    // it. As a JSON or form value it is the name rule's.
    PatternDef {
        category: "secret",
        regex: r"(?i)\bauthorization\s*:\s*(?:basic|bearer|token)\s+[A-Za-z0-9._~+/-]+=*",
        validator: None,
        context: None,
        replacement: Replacement::AfterColon("[SECRET]"),
        retry_on_reject: false,
    },
    // A JWT: two base64url JSON segments (`eyJ` is `{"`) and a signature,
    // which is empty for an unsigned token. The shape alone, not a decode.
    PatternDef {
        category: "secret",
        regex: r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        validator: None,
        context: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
];

// --- context checks: the text around a match ---

/// A byte that continues an id-like token.
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The run of `is_word` bytes that ends just before `at` (`rev`) or starts
/// at `at`. Word bytes are ASCII, so both ends are char boundaries.
fn word_run(text: &str, at: usize, rev: bool) -> &str {
    let b = text.as_bytes();
    let (lo, hi) = if rev {
        let n = b
            .get(..at)
            .map_or(0, |h| h.iter().rev().take_while(|&&c| is_word(c)).count());
        (at - n, at)
    } else {
        let n = b
            .get(at..)
            .map_or(0, |t| t.iter().take_while(|&&c| is_word(c)).count());
        (at, at + n)
    };
    text.get(lo..hi).unwrap_or_default()
}

/// A digit or hex group the way a number or an id writes one: hex digits
/// only, at least one of them decimal (`1111`, `a0e5`, `01`). A word
/// (`card`, `SSN`, `json`) is not one.
fn is_id_group(run: &str) -> bool {
    !run.is_empty()
        && run.bytes().all(|c| c.is_ascii_hexdigit())
        && run.bytes().any(|c| c.is_ascii_digit())
}

/// An id group with a letter in it (`a0e5`): a piece of a hex id. A plain
/// digit group beside a card or SSN is an expiry, a CVV, an account number,
/// a year or cents, so it does not make the digits part of an id.
fn is_hex_id_group(run: &str) -> bool {
    is_id_group(run) && run.bytes().any(|c| c.is_ascii_alphabetic())
}

/// Is the match at `start..end` inside a UUID: the token of word bytes and
/// `-` around it is five hex groups of 8, 4, 4, 4 and 12.
fn in_uuid(text: &str, start: usize, end: usize) -> bool {
    let b = text.as_bytes();
    let part = |c: &&u8| is_word(**c) || **c == b'-';
    let lo = start
        - b.get(..start)
            .map_or(0, |h| h.iter().rev().take_while(part).count());
    let hi = end
        + b.get(end..)
            .map_or(0, |t| t.iter().take_while(part).count());
    text.get(lo..hi).is_some_and(|token| {
        let groups = || token.trim_matches('-').split('-');
        groups().all(|g| g.bytes().all(|c| c.is_ascii_hexdigit()))
            && groups().map(str::len).eq([8, 4, 4, 4, 12])
    })
}

/// Is the match at `start..end` glued into a longer token: a letter, digit or
/// `_` right beside it; a `.` and a digit group before it (the digits go on
/// a dotted number: `2.021000021`, the last octet of `10.0.0.0 4111…`); a
/// `-` or `.` and a hex group with a letter in it; or, inside a UUID, a `-`
/// and any of its groups with a decimal digit. A label joined by a `-` or
/// `.` (`card-4111…`, `4111….json`), a plain digit group outside a UUID
/// (`4111…-1227`, `7-4111…`, `4111….00`, `021000021-123456789`), a `.`
/// ending a sentence and a `-` starting a negative number join nothing.
fn joined(text: &str, start: usize, end: usize) -> bool {
    let b = text.as_bytes();
    let near = |i: Option<usize>| i.and_then(|i| b.get(i).copied());
    let group = |run: &str| is_hex_id_group(run) || (is_id_group(run) && in_uuid(text, start, end));
    let before = match near(start.checked_sub(1)) {
        Some(c) if is_word(c) => true,
        Some(b'.') => is_id_group(word_run(text, start - 1, true)),
        Some(b'-') => group(word_run(text, start - 1, true)),
        _ => false,
    };
    let after = match near(Some(end)) {
        Some(c) if is_word(c) => true,
        Some(b'-' | b'.') => group(word_run(text, end + 1, false)),
        _ => false,
    };
    before || after
}

/// A card, SSN or routing number is a value of its own, not digits inside
/// an id. Detection is not weakened: the same digits standing alone, set
/// off by spaces, quotes, colons, `=` or brackets, or joined to a word, are
/// still matched.
pub(crate) fn stands_alone(text: &str, start: usize, end: usize) -> bool {
    !joined(text, start, end)
}

/// A filter field naming an address (`ip.addr==`, `src_ip>=`): what follows
/// its comparison operator is an address, not a version pin.
fn names_an_address(token: &str) -> bool {
    token.split(['.', '_', '-']).any(|part| {
        [
            "ip", "ipv4", "addr", "address", "host", "src", "dst", "peer",
        ]
        .iter()
        .any(|w| part.eq_ignore_ascii_case(w))
    })
}

/// The package-name token that ends `head` (`[A-Za-z][A-Za-z0-9_.-]*`), or
/// None. A token that is itself part of a URL path, an authority or a
/// `user@host` (preceded by `/`, `:` or `@`) is none.
fn package_token(head: &str) -> Option<&str> {
    let token = word_run_dotted(head);
    let lead = head
        .len()
        .checked_sub(token.len() + 1)
        .and_then(|i| head.as_bytes().get(i));
    (token
        .bytes()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && !matches!(lead, Some(b'/' | b':' | b'@')))
    .then_some(token)
}

/// Numbers that read as a version rather than an address: every part under
/// 100, and a first part other than 10, where private networks start. A
/// random public address passes this about one time in fifty, so it decides
/// only where the context could hold either.
fn reads_as_version(quad: &str) -> bool {
    let mut parts = quad
        .split('.')
        .map(|p| p.parse::<u16>().unwrap_or(u16::MAX));
    parts.next().is_some_and(|n| n < 100 && n != 10) && parts.all(|n| n < 100)
}

/// A dotted quad is an address unless its context makes it a version. Part
/// of a longer dotted number (`1.2.3.4.1000`), touching a `v` (`v1.2.3.4`),
/// or after `version ` or `version=`: always a version. After a pip
/// comparison operator whose field does not name an address
/// (`numpy==1.26.4.1`), after `v=` or `ver: `, after a name and `/` or `-`
/// (`nginx/1.2.3.4`), or before a pre-release suffix (`2.10.1.0-rc1`): the
/// same place holds an address as readily (`client-203.0.113.9`,
/// `remote==203.0.113.9`), so only numbers that read as a version are kept.
pub(crate) fn ipv4_in_context(text: &str, start: usize, end: usize) -> bool {
    let b = text.as_bytes();
    let dotted_after = b.get(end) == Some(&b'.') && b.get(end + 1).is_some_and(u8::is_ascii_digit);
    let dotted_before = start.checked_sub(1).and_then(|i| b.get(i)) == Some(&b'.')
        && start
            .checked_sub(2)
            .and_then(|i| b.get(i))
            .is_some_and(u8::is_ascii_digit);
    if dotted_after || dotted_before {
        return false;
    }
    let Some(head) = text.get(..start) else {
        return true;
    };
    let pip = ["==", ">=", "<=", "~=", "!="]
        .iter()
        .find_map(|op| head.strip_suffix(op));
    if pip.is_some_and(|field| names_an_address(word_run_dotted(field))) {
        return true;
    }
    // A lone word: nothing alphabetic right before it.
    let ends_with_word = |w: &str| {
        head.len() >= w.len()
            && head
                .get(head.len() - w.len()..)
                .is_some_and(|tail| tail.eq_ignore_ascii_case(w))
            && !head
                .len()
                .checked_sub(w.len() + 1)
                .and_then(|i| b.get(i))
                .is_some_and(u8::is_ascii_alphabetic)
    };
    if ["v", "version ", "version="]
        .iter()
        .any(|w| ends_with_word(w))
    {
        return false;
    }
    let label = ["v=", "ver ", "ver=", "ver: ", "version: "]
        .iter()
        .any(|w| ends_with_word(w));
    let named = head
        .strip_suffix('/')
        .or_else(|| head.strip_suffix('-'))
        .and_then(package_token)
        .is_some();
    let suffix = matches!(b.get(end), Some(b'-' | b'+'))
        && b.get(end + 1).is_some_and(u8::is_ascii_alphabetic);
    let either = pip.is_some() || label || named || suffix;
    !(either && text.get(start..end).is_some_and(reads_as_version))
}

/// The field name ending `head`, dots and dashes included (`ip.addr`).
fn word_run_dotted(head: &str) -> &str {
    let n = head
        .bytes()
        .rev()
        .take_while(|&c| is_word(c) || c == b'.' || c == b'-')
        .count();
    head.get(head.len() - n..).unwrap_or_default()
}

// --- validators: regex proposes, these confirm ---

fn digits_of(s: &str) -> Vec<u8> {
    s.bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| b - b'0')
        .collect()
}

pub(crate) fn luhn_valid(m: &str) -> bool {
    let d = digits_of(m);
    if !(13..=19).contains(&d.len()) {
        return false;
    }
    let mut sum = 0u32;
    let mut double = false;
    for &x in d.iter().rev() {
        let mut v = u32::from(x);
        if double {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
        double = !double;
    }
    sum % 10 == 0
}

pub(crate) fn ssn_valid(m: &str) -> bool {
    // matched shape is fixed by the regex: ddd-dd-dddd
    let field = |range: std::ops::Range<usize>| -> u32 {
        m.get(range).and_then(|f| f.parse().ok()).unwrap_or(0)
    };
    let (area, group, serial) = (field(0..3), field(4..6), field(7..11));
    area != 0 && area != 666 && area < 900 && group != 0 && serial != 0
}

pub(crate) fn ipv4_valid(m: &str) -> bool {
    m.parse::<std::net::Ipv4Addr>().is_ok()
}

pub(crate) fn ipv6_valid(m: &str) -> bool {
    m.parse::<std::net::Ipv6Addr>().is_ok()
}

pub(crate) fn aba_valid(m: &str) -> bool {
    let d = digits_of(m);
    if d.len() != 9 || d.iter().all(|&x| x == 0) {
        return false;
    }
    // ABA checksum: 3(d1+d4+d7) + 7(d2+d5+d8) + 1(d3+d6+d9) mod 10 == 0
    let sum = 3 * u32::from(d[0] + d[3] + d[6])
        + 7 * u32::from(d[1] + d[4] + d[7])
        + u32::from(d[2] + d[5] + d[8]);
    sum % 10 == 0
}

pub(crate) fn iban_valid(m: &str) -> bool {
    // ISO 13616: move the first 4 chars to the end, map A..Z to 10..35, mod 97 == 1
    let (Some(head), Some(tail)) = (m.get(..4), m.get(4..)) else {
        return false;
    };
    let mut rem: u32 = 0;
    for c in tail.chars().chain(head.chars()) {
        let v = if c.is_ascii_digit() {
            c as u32 - '0' as u32
        } else if c.is_ascii_uppercase() {
            c as u32 - 'A' as u32 + 10
        } else {
            return false;
        };
        rem = if v < 10 {
            (rem * 10 + v) % 97
        } else {
            (rem * 100 + v) % 97
        };
    }
    rem == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luhn_accepts_valid_and_rejects_invalid() {
        assert!(luhn_valid("4111-1111-1111-1111")); // classic Visa test PAN
        assert!(luhn_valid("4111 1111 1111 1111"));
        assert!(!luhn_valid("4111-1111-1111-1112")); // checksum off by one
        assert!(!luhn_valid("1234")); // too short
    }

    #[test]
    fn ssn_validates_area_group_serial_ranges() {
        assert!(ssn_valid("123-45-6789"));
        assert!(!ssn_valid("000-45-6789")); // area 000
        assert!(!ssn_valid("666-45-6789")); // area 666
        assert!(!ssn_valid("900-45-6789")); // area >= 900
        assert!(!ssn_valid("123-00-6789")); // group 00
        assert!(!ssn_valid("123-45-0000")); // serial 0000
    }

    #[test]
    fn ip_validators_use_std_parsing() {
        assert!(ipv4_valid("10.0.0.5"));
        assert!(!ipv4_valid("1.2.3.4.5")); // version string, not an address
        assert!(!ipv4_valid("999.0.0.1")); // octet out of range
        assert!(ipv6_valid("2001:db8::1"));
        assert!(!ipv6_valid("12:30:45")); // clock time, not an address
    }

    #[test]
    fn aba_checksum() {
        assert!(aba_valid("021000021")); // JPMorgan Chase routing number
        assert!(!aba_valid("021000022")); // checksum broken
        assert!(!aba_valid("000000000")); // all zeros pass mod10 but are not a routing number
    }

    #[test]
    fn iban_mod97() {
        assert!(iban_valid("GB82WEST12345698765432")); // ISO 13616 example
        assert!(!iban_valid("GB82WEST12345698765433"));
    }

    #[test]
    fn registry_covers_exactly_the_eight_categories() {
        for def in BUILTINS {
            assert!(
                CATEGORIES.contains(&def.category),
                "unknown category {}",
                def.category
            );
        }
        for cat in CATEGORIES {
            assert!(
                BUILTINS.iter().any(|d| d.category == cat),
                "no pattern for {cat}"
            );
        }
    }
}
