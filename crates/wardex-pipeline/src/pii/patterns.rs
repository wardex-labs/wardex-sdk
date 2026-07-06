//! Declarative PII pattern registry — design §7.
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
}

pub(crate) struct PatternDef {
    pub category: &'static str,
    pub regex: &'static str,
    pub validator: Option<fn(&str) -> bool>,
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

pub(crate) static BUILTINS: &[PatternDef] = &[
    PatternDef {
        category: "email",
        regex: r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        validator: None,
        replacement: Replacement::Label("[EMAIL]"),
        retry_on_reject: false,
    },
    // NANP with mandatory separators (or +1): bare 10-digit runs are NOT matched
    // to avoid masking ids/timestamps (design §5.1).
    PatternDef {
        category: "phone_number",
        regex: r"(?:\+1[ .-]?)?\(?[2-9][0-9]{2}\)?[ .-][0-9]{3}[ .-][0-9]{4}\b",
        validator: None,
        replacement: Replacement::Label("[PHONE]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "credit_card",
        regex: r"\b[0-9](?:[ -]?[0-9]){12,18}\b",
        validator: Some(luhn_valid),
        replacement: Replacement::CardLast4,
        retry_on_reject: true,
    },
    PatternDef {
        category: "us_ssn",
        regex: r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
        validator: Some(ssn_valid),
        replacement: Replacement::Label("[SSN]"),
        retry_on_reject: false,
    },
    // The {3,} tail makes version strings ("1.2.3.4.5") match as a whole and
    // then fail Ipv4Addr parsing — rejecting the false positive outright.
    PatternDef {
        category: "ip_address",
        regex: r"\b[0-9]{1,3}(?:\.[0-9]{1,3}){3,}\b",
        validator: Some(ipv4_valid),
        replacement: Replacement::Label("[IP_ADDRESS]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "ip_address",
        regex: r"\b(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]*[0-9A-Fa-f]\b",
        validator: Some(ipv6_valid),
        replacement: Replacement::Label("[IP_ADDRESS]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "us_bank_routing",
        regex: r"\b[0-9]{9}\b",
        validator: Some(aba_valid),
        replacement: Replacement::Label("[BANK_ROUTING]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "iban",
        regex: r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
        validator: Some(iban_valid),
        replacement: Replacement::Label("[IBAN]"),
        retry_on_reject: false,
    },
    // Secrets: fixed prefixes only — deterministic, no entropy guessing (§5.1).
    PatternDef {
        category: "secret",
        regex: r"\bsk-[A-Za-z0-9_-]{16,}", // covers sk- and sk-ant-
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bAKIA[0-9A-Z]{16}\b",
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bgh[pousr]_[A-Za-z0-9]{36}\b",
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bxox[abpos]-[A-Za-z0-9-]{10,}",
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"\bAIza[0-9A-Za-z_-]{35}\b",
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}=*",
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
    PatternDef {
        category: "secret",
        regex: r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        validator: None,
        replacement: Replacement::Label("[SECRET]"),
        retry_on_reject: false,
    },
];

// --- validators: regex proposes, these confirm (design §5.1) ---

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
    let area: u32 = m[0..3].parse().unwrap_or(0);
    let group: u32 = m[4..6].parse().unwrap_or(0);
    let serial: u32 = m[7..11].parse().unwrap_or(0);
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
    let mut rem: u32 = 0;
    for c in m[4..].chars().chain(m[..4].chars()) {
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
