//! Phase 0: proto/wardex/v1/*.proto → OUT_DIR/wardex.v1.rs

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("protoc-bin-vendored: no binary for this platform");

    let proto_root = "../../proto";
    let proto_files = [
        "../../proto/wardex/v1/common.proto",
        "../../proto/wardex/v1/span.proto",
        "../../proto/wardex/v1/state.proto",
        "../../proto/wardex/v1/envelope.proto",
        "../../proto/wardex/v1/ingest.proto",
    ];

    for f in &proto_files {
        println!("cargo:rerun-if-changed={f}");
    }

    prost_build::Config::new()
        .protoc_executable(&protoc)
        .compile_protos(&proto_files, &[proto_root])
        .expect("failed to compile protos");

    // --- OTLP (opentelemetry-proto vendoring) ---
    let otlp_root = "proto"; // import path starts with opentelemetry/proto/...
    let otlp_files = [
        "proto/opentelemetry/proto/common/v1/common.proto",
        "proto/opentelemetry/proto/resource/v1/resource.proto",
        "proto/opentelemetry/proto/trace/v1/trace.proto",
        "proto/opentelemetry/proto/collector/trace/v1/trace_service.proto",
    ];
    for f in &otlp_files {
        println!("cargo:rerun-if-changed={f}");
    }
    prost_build::Config::new()
        .protoc_executable(&protoc)
        // The vendored OTLP proto's doc comments include JSON examples without a
        // language tag, which rustdoc mistakes for Rust code, breaking the
        // `cargo test` doctest step (e.g. the Span.attributes comment in
        // trace.proto). Code generation is unaffected — only the comments are omitted.
        .disable_comments(["."])
        .compile_protos(&otlp_files, &[otlp_root])
        .expect("failed to compile OTLP protos");
}
