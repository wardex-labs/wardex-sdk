//! Performance guard — design §9: masking 1 MiB of mixed text stays under
//! 100 ms in release mode. Run explicitly:
//! `cargo test -p wardex-pipeline --release --test pii_perf -- --ignored`

use wardex_pipeline::pii::PiiEngine;

#[test]
#[ignore = "release-mode perf guard; debug builds of the regex crate are ~10x slower"]
fn masks_one_mebibyte_under_100ms() {
    let engine = PiiEngine::new(&[]).unwrap();
    // mixed corpus: prose + PII hits + near-misses, repeated to ~1 MiB
    let chunk = "Deploy log: host 10.0.42.7 responded in 42ms. Contact ops@corp.example \
                 or +1 555 123 4567. Order 1234567890 charged to 4111-1111-1111-1111. \
                 Version 1.2.3.4.5 rollout at 12:30:45. Token sk-abcdefghijklmnop1234. ";
    let corpus = chunk.repeat(1024 * 1024 / chunk.len() + 1);
    assert!(corpus.len() >= 1024 * 1024);

    let start = std::time::Instant::now();
    let masked = engine.mask_text(&corpus).expect("corpus contains PII");
    let elapsed = start.elapsed();

    assert!(masked.contains("[EMAIL]"));
    assert!(
        elapsed.as_millis() < 100,
        "masking 1 MiB took {elapsed:?} (budget: 100ms)"
    );
}
