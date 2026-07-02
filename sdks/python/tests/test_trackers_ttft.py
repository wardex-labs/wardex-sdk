from wardex_sdk.interceptors._trackers import _ttft_from_marks


def test_ttft_picks_first_mark_past_header_boundary():
    # marks: (cumulative bytes, ns). header_len=40 → the 50-byte mark is the first
    # to exceed the boundary
    start = 1_000_000_000
    marks = [(40, start + 1_000_000), (50, start + 5_000_000)]  # +1ms, +5ms
    # 40 > 40 is false → the second one (50>40) is picked = +5ms
    assert _ttft_from_marks(marks, 40, start) == 5.0


def test_ttft_zero_when_no_mark_past_boundary():
    start = 0
    marks = [(10, 1_000_000), (20, 2_000_000)]
    assert _ttft_from_marks(marks, 100, start) == 0.0


def test_ttft_from_marks_empty():
    assert _ttft_from_marks([], 10, 0) == 0.0


def test_ttft_zero_when_request_start_unknown():
    # mid-connection capture: _req_start_ns=0 → 0.0 instead of a bogus Unix timestamp value
    assert _ttft_from_marks([(50, 1_750_000_000_000_000_000)], 40, 0) == 0.0
