//! Performance guard — design §9: masking 1 MiB of mixed text stays under
//! 100 ms in release mode. Run explicitly:
//! `cargo test -p wardex-pipeline --release --test pii_perf -- --ignored`

use wardex_pipeline::pii::PiiEngine;

#[test]
#[ignore = "release-mode perf guard; debug builds of the regex crate are ~10x slower"]
fn masks_one_mebibyte_under_100ms() {
    let engine = PiiEngine::new(&[]).unwrap();
    // mixed corpus: prose + PII hits + near-misses + call arguments (a JSON
    // body, a query, a tool call's escaped JSON — every `=` and `:` is a stop
    // for the name rules), repeated to ~1 MiB
    let chunk = "Deploy log: host 10.0.42.7 responded in 42ms. Contact ops@corp.example \
                 or +1 555 123 4567. Order 1234567890 charged to 4111-1111-1111-1111. \
                 Version 1.2.3.4.5 rollout at 12:30:45. Token sk-abcdefghijklmnop1234. \
                 {\"query\": \"seoul weather\", \"limit\": 5, \"api_key\": \"abc123\", \
                 \"page_size\": 20, \"filters\": {\"lang\": \"ko\", \"units\": \"metric\"}} \
                 GET /v1/search?q=seoul&page=2&appid=owm999key&sort=asc \
                 {\"arguments\":\"{\\\"city\\\":\\\"seoul\\\",\\\"token\\\":\\\"t1\\\"}\"} ";
    let corpus = chunk.repeat(1024 * 1024 / chunk.len() + 1);
    assert!(corpus.len() >= 1024 * 1024);

    let start = std::time::Instant::now();
    let masked = engine.mask_text(&corpus).expect("corpus contains PII");
    let elapsed = start.elapsed();

    assert!(masked.contains("[EMAIL]"));
    assert!(masked.contains(r#""api_key": "[SECRET]""#));
    assert!(masked.contains("appid=[SECRET]"));
    eprintln!("masked {} bytes in {elapsed:?}", corpus.len());
    assert!(
        elapsed.as_millis() < 100,
        "masking 1 MiB took {elapsed:?} (budget: 100ms)"
    );
}
