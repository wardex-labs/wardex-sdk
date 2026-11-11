"""What `set_tag` / `set_user` / `set_account` / `set_context` mean across tasks, threads and
blocks, observed on the spans a host's code ships.

The rule under test, stated once:

* Work started in a context READS what that context had set before it started.
* A write lands on the scope of the context that makes it. A context writing to a scope it did
  not create writes to a copy of its own, so a write never reaches a sibling or the context that
  started the work.
* Inside `new_scope()` / `isolation_scope()` a write belongs to the block and is gone at exit.
* A plain `threading.Thread` starts with no tags at all: Python gives it an empty context.
  `wardex.bind_context` carries the starting context into it.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import warnings

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation
from wardex_sdk.testing import RecordingTransport


@pytest.fixture
def rec():
    # Both "say once" tables are process-global; without this the order of
    # tests decides whether a line under test was already spent.
    from wardex_sdk._assembly import _diag

    for key in [k for k in _diag._REPORTED if k.startswith("wardex.set_context")]:
        del _diag._REPORTED[key]
    _hub._said.clear()
    _hub.reset_for_test()
    transport = RecordingTransport()
    wardex.init(transport=transport, intercept=False)
    yield transport
    wardex.close()
    _hub.reset_for_test()


def _spans(transport: RecordingTransport) -> dict[str, object]:
    wardex.flush()
    return {s.name: s for e in transport.envelopes for s in e.spans}


def _extra(span) -> dict[str, object]:
    return dict(span.extra or ())


def _markers(span) -> tuple[Limitation, ...]:
    return span.capture_integrity.limitations if span.capture_integrity else ()


# --------------------------------------------------------------------------
# concurrent tasks and threads
# --------------------------------------------------------------------------


def test_concurrent_tasks_never_see_each_others_user_after_a_startup_tag(rec):
    async def job(uid: str) -> None:
        wardex.set_user(wardex.UserInfo(id=uid))
        wardex.set_tag("account", "acct-" + uid)
        await asyncio.sleep(0.01)
        with wardex.span("work-" + uid):
            pass

    async def main() -> None:
        wardex.set_tag("startup", "1")
        await asyncio.gather(job("A"), job("B"))

    asyncio.run(main())
    spans = _spans(rec)
    assert _extra(spans["work-A"]) == {"startup": "1", "account": "acct-A", "user.id": "A"}
    assert _extra(spans["work-B"]) == {"startup": "1", "account": "acct-B", "user.id": "B"}


def test_a_task_write_never_reaches_the_context_that_started_it(rec):
    async def child() -> None:
        wardex.set_tag("request", "child")

    async def main() -> None:
        wardex.set_tag("startup", "1")
        await asyncio.create_task(child())
        with wardex.span("parent-after"):
            pass

    asyncio.run(main())
    assert _extra(_spans(rec)["parent-after"]) == {"startup": "1"}


def test_startup_tags_reach_a_bound_thread_and_its_writes_stay_there(rec):
    wardex.set_tag("startup", "1")

    def work() -> None:
        wardex.set_user("worker-user")
        with wardex.span("in-thread"):
            pass

    thread = threading.Thread(target=wardex.bind_context(work))
    thread.start()
    thread.join()
    with wardex.span("main-after"):
        pass
    spans = _spans(rec)
    assert _extra(spans["in-thread"]) == {"startup": "1", "user.id": "worker-user"}
    assert _extra(spans["main-after"]) == {"startup": "1"}


def test_a_plain_thread_starts_with_no_tags(rec):
    """Documented, not hidden: Python starts a thread with an empty context."""
    wardex.set_tag("startup", "1")

    def work() -> None:
        with wardex.span("plain-thread"):
            pass

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert "startup" not in _extra(_spans(rec)["plain-thread"])


def test_two_copies_of_one_context_do_not_share_writes(rec):
    wardex.set_tag("startup", "1")
    first, second = contextvars.copy_context(), contextvars.copy_context()

    def tag_and_span(name: str) -> None:
        wardex.set_tag("who", name)
        with wardex.span(name):
            pass

    first.run(tag_and_span, "first")
    second.run(tag_and_span, "second")
    first.run(tag_and_span, "first-again")
    spans = _spans(rec)
    assert _extra(spans["first"])["who"] == "first"
    assert _extra(spans["second"])["who"] == "second"
    assert _extra(spans["first-again"])["who"] == "first-again"


def test_a_dispatcher_that_sets_identity_then_starts_each_task_gives_each_its_own(rec):
    """The accept-loop shape: the parent sets the request's user, then starts it."""

    async def handle(uid: str) -> None:
        await asyncio.sleep(0.01)
        with wardex.span("req-" + uid):
            pass

    async def main() -> None:
        wardex.set_tag("startup", "1")
        tasks = []
        for uid in "ABC":
            wardex.set_user(uid)
            tasks.append(asyncio.create_task(handle(uid)))
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)

    asyncio.run(main())
    spans = _spans(rec)
    for uid in "ABC":
        assert _extra(spans["req-" + uid]) == {"startup": "1", "user.id": uid}


def test_a_dispatcher_that_sets_identity_then_submits_bound_work_gives_each_its_own(rec):
    import time
    from concurrent.futures import ThreadPoolExecutor

    def work(uid: str) -> None:
        time.sleep(0.01)
        with wardex.span("req-" + uid):
            pass

    with ThreadPoolExecutor(3) as pool:
        for uid in "ABC":
            wardex.set_account("acct-" + uid)
            pool.submit(wardex.bind_context(work), uid)
    spans = _spans(rec)
    for uid in "ABC":
        assert _extra(spans["req-" + uid]) == {"wardex.account.id": "acct-" + uid}


def test_each_continued_request_keeps_its_own_user(rec):
    async def request(uid: str) -> None:
        with wardex.continue_trace({}):
            wardex.set_user(uid)
            await asyncio.sleep(0.01)
            with wardex.span("req-" + uid):
                pass

    async def main() -> None:
        wardex.set_tag("startup", "1")
        await asyncio.gather(request("A"), request("B"))
        with wardex.span("after"):
            pass

    asyncio.run(main())
    spans = _spans(rec)
    assert _extra(spans["req-A"]) == {"startup": "1", "user.id": "A"}
    assert _extra(spans["req-B"]) == {"startup": "1", "user.id": "B"}
    assert _extra(spans["after"]) == {"startup": "1"}


# --------------------------------------------------------------------------
# set_user: accepted forms, loud mistakes, and the stamp's independence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda: wardex.set_user(wardex.UserInfo(id="u1", email="a@b.c")), ("u1", "a@b.c")),
        (lambda: wardex.set_user("u1"), ("u1", None)),
        (lambda: wardex.set_user(id="u1", email="a@b.c"), ("u1", "a@b.c")),
        (lambda: wardex.set_user(wardex.UserInfo(id=42)), ("42", None)),
        (lambda: wardex.set_user(42), ("42", None)),
        (lambda: wardex.set_user(id=42, email="a@b.c"), ("42", "a@b.c")),
    ],
    ids=["userinfo", "plain-id", "keywords", "userinfo-int-id", "int-id", "keyword-int-id"],
)
def test_set_user_accepts_userinfo_a_plain_id_or_keyword_fields(rec, call, expected):
    wardex.set_tag("customer", "acme")
    call()
    with wardex.span("s"):
        pass
    extra = _extra(_spans(rec)["s"])
    assert extra["customer"] == "acme"
    assert (extra.get("user.id"), extra.get("user.email")) == expected


@pytest.mark.parametrize(
    "call",
    [
        lambda: wardex.set_user({"id": "u"}),
        lambda: wardex.set_user(4.2),
        lambda: wardex.set_user(True),
        lambda: wardex.set_user(wardex.UserInfo(id="u"), id="v"),
        lambda: wardex.UserInfo(id={"pk": 1}),
        lambda: wardex.UserInfo(email=42),
        lambda: wardex.set_user(id=["u"]),
    ],
    ids=[
        "dict",
        "float",
        "bool",
        "both-forms",
        "userinfo-dict",
        "userinfo-int-email",
        "keyword-list",
    ],
)
def test_a_mistyped_set_user_raises_at_the_call_and_keeps_the_tags(rec, call):
    wardex.set_tag("customer", "acme")
    wardex.set_user("u1")
    with pytest.raises((TypeError, ValueError)) as caught:
        call()
    assert "set_user" in str(caught.value) or "UserInfo" in str(caught.value)
    with wardex.span("s"):
        pass
    assert _extra(_spans(rec)["s"]) == {"customer": "acme", "user.id": "u1"}
    assert wardex.diagnostics().losses == ()


@pytest.mark.parametrize(
    "call",
    [
        lambda: wardex.set_user(id=None, email=None),
        lambda: wardex.set_user(id="", email=""),
        lambda: wardex.set_user(""),
    ],
    ids=["keywords-all-none", "keywords-all-empty", "empty-id"],
)
def test_an_anonymous_user_clears_the_user_instead_of_failing_the_call(rec, call):
    wardex.set_user("u1")
    call()
    with wardex.span("s"):
        pass
    assert not any(k.startswith("user.") for k in _extra(_spans(rec)["s"]))


def test_the_yielded_scope_checks_set_user_like_the_module_level_call(rec):
    wardex.set_tag("customer", "acme")
    with wardex.isolation_scope() as scope:
        scope.set_user("user-42")
        with wardex.span("a"):
            pass
        with pytest.raises(TypeError, match="set_user"):
            scope.set_user({"id": "u"})
        with wardex.span("b"):
            pass
    spans = _spans(rec)
    assert _extra(spans["a"]) == {"customer": "acme", "user.id": "user-42"}
    assert _extra(spans["b"]) == {"customer": "acme", "user.id": "user-42"}
    assert wardex.diagnostics().losses == ()


@pytest.mark.parametrize("opener", [wardex.isolation_scope, wardex.new_scope], ids=["iso", "new"])
def test_the_yielded_scope_and_the_module_level_setters_write_the_same_block(rec, opener):
    with opener() as scope:
        wardex.set_tag("module", "1")
        scope.set_tag("direct", "1")
        assert scope.tags["module"] == "1"
        with wardex.span("inside"):
            pass
    with wardex.span("after"):
        pass
    spans = _spans(rec)
    assert _extra(spans["inside"]) == {"module": "1", "direct": "1"}
    assert _extra(spans["after"]) == {}


def test_a_write_through_the_yielded_scope_does_not_reach_a_task_started_before_it(rec):
    async def child() -> None:
        await asyncio.sleep(0.01)
        with wardex.span("child"):
            pass

    async def main() -> None:
        with wardex.isolation_scope() as scope:
            scope.set_user("first")
            task = asyncio.create_task(child())
            await asyncio.sleep(0)
            scope.set_user("second")
            with wardex.span("block"):
                pass
            await task

    asyncio.run(main())
    spans = _spans(rec)
    assert _extra(spans["child"]) == {"user.id": "first"}
    assert _extra(spans["block"]) == {"user.id": "second"}


def test_set_user_none_clears_the_user(rec):
    wardex.set_user("u1")
    wardex.set_user(None)
    with wardex.span("s"):
        pass
    assert not any(k.startswith("user.") for k in _extra(_spans(rec)["s"]))


def test_a_user_that_fails_to_stamp_does_not_take_the_tags_with_it(rec):
    class Exploding:
        @property
        def id(self):
            raise RuntimeError("boom")

        email = username = ip_address = None

    from wardex_sdk._types import InternalSpan  # noqa: F401  (type of the span below)

    with wardex.span("s"):
        pass
    span = _spans(rec)["s"]
    stamped = _hub.get_client()._stamp_scope(span, ({"customer": "acme"}, Exploding()))
    assert ("customer", "acme") in stamped.extra


# --------------------------------------------------------------------------
# set_account
# --------------------------------------------------------------------------


def test_set_account_is_public_and_stamps_spans_under_its_fixed_key(rec):
    assert "set_account" in wardex.__all__
    wardex.set_account("acct-42")
    with wardex.span("s"):
        pass
    assert _extra(_spans(rec)["s"])["wardex.account.id"] == "acct-42"


def test_set_account_is_scoped_like_set_user(rec):
    async def request(account: str) -> None:
        with wardex.isolation_scope():
            wardex.set_account(account)
            await asyncio.sleep(0.01)
            with wardex.span("req-" + account):
                pass

    async def main() -> None:
        await asyncio.gather(request("a1"), request("a2"))
        with wardex.span("after"):
            pass

    asyncio.run(main())
    spans = _spans(rec)
    assert _extra(spans["req-a1"])["wardex.account.id"] == "a1"
    assert _extra(spans["req-a2"])["wardex.account.id"] == "a2"
    assert "wardex.account.id" not in _extra(spans["after"])


@pytest.mark.parametrize(("bad", "error"), [(42, TypeError), (None, TypeError), ("", ValueError)])
def test_set_account_rejects_a_non_string_or_empty_id(rec, bad, error):
    with pytest.raises(error, match="set_account"):
        wardex.set_account(bad)


def test_the_account_id_survives_masking_on_the_wire(rec):
    from wardex_sdk.transport._codec import decode, encode

    wardex.set_account("acct-42")
    with wardex.span("s"):
        pass
    wardex.flush()
    wire = decode(encode(rec.envelopes[-1], "mask"))
    assert "acct-42" in repr(wire)


# --------------------------------------------------------------------------
# set_context
# --------------------------------------------------------------------------


def test_set_context_scalars_reach_spans_as_flattened_attributes(rec):
    wardex.set_context("order", {"id": "o-1", "items": 3, "rush": True, "total": 9.5})
    with wardex.span("s"):
        pass
    extra = _extra(_spans(rec)["s"])
    assert extra["wardex.context.order.id"] == "o-1"
    assert extra["wardex.context.order.items"] == 3
    assert extra["wardex.context.order.rush"] is True
    assert extra["wardex.context.order.total"] == 9.5


def test_set_context_replaces_the_previous_payload_of_that_name(rec):
    wardex.set_context("order", {"id": "o-1", "items": 3})
    wardex.set_context("order", {"id": "o-2"})
    with wardex.span("s"):
        pass
    extra = _extra(_spans(rec)["s"])
    assert extra["wardex.context.order.id"] == "o-2"
    assert "wardex.context.order.items" not in extra


def test_set_context_skips_non_scalar_values_and_says_so_once(rec, caplog):
    lock = threading.Lock()
    with caplog.at_level("WARNING", logger="wardex_sdk"):
        wardex.set_context("runtime", {"lock": lock, "nested": {"a": 1}, "name": "w"})
        wardex.set_context("runtime", {"lock": lock})
    with wardex.span("s"):
        pass
    extra = _extra(_spans(rec)["s"])
    assert "wardex.context.runtime.lock" not in extra
    assert "wardex.context.runtime.nested" not in extra
    warned = [r for r in caplog.records if "set_context" in r.getMessage()]
    assert len(warned) == 1


def test_set_context_stamps_a_bounded_number_of_keys(rec):
    from wardex_sdk._scope import MAX_CONTEXT_KEYS

    wardex.set_context("big", {f"k{i:03d}": i for i in range(MAX_CONTEXT_KEYS + 10)})
    with wardex.span("s"):
        pass
    keys = [k for k in _extra(_spans(rec)["s"]) if k.startswith("wardex.context.big.")]
    assert len(keys) == MAX_CONTEXT_KEYS


@pytest.mark.parametrize(
    "call", [lambda: wardex.set_context(1, {}), lambda: wardex.set_context("x", 5)]
)
def test_set_context_rejects_a_wrong_type_at_the_call(rec, call):
    with pytest.raises(TypeError, match="set_context"):
        call()


def test_set_context_values_are_masked_on_the_wire(rec):
    from wardex_sdk.transport._codec import decode, encode

    wardex.set_context("customer", {"email": "alice@example.com"})
    with wardex.span("s"):
        pass
    wardex.flush()
    wire = decode(encode(rec.envelopes[-1], "mask"))
    assert "alice@example.com" not in repr(wire)
    assert "wardex.context.customer.email" in repr(wire)


# --------------------------------------------------------------------------
# isolation_scope / new_scope
# --------------------------------------------------------------------------


def test_a_span_inside_isolation_scope_keeps_its_parent_and_conversation(rec):
    @wardex.workflow
    def handle() -> None:
        with wardex.isolation_scope():
            wardex.set_user("u")
            with wardex.span("inside-iso"):
                pass

    with wardex.conversation("chat", id="conv-1"):
        handle()
    spans = _spans(rec)
    inside, wf = spans["inside-iso"], spans["invoke_workflow handle"]
    assert inside.parent_span_id == wf.context.span_id
    assert inside.context.trace_id == wf.context.trace_id
    assert inside.conversation is not None and inside.conversation.conversation_id == "conv-1"
    assert _extra(inside)["user.id"] == "u"
    assert "user.id" not in _extra(wf)


def test_a_tag_set_inside_new_scope_is_gone_after_the_block(rec):
    wardex.set_tag("base", "1")
    with wardex.new_scope():
        wardex.set_tag("inner", "x")
        wardex.set_user("block-user")
        with wardex.span("inside"):
            pass
    with wardex.span("after"):
        pass
    spans = _spans(rec)
    assert _extra(spans["inside"]) == {"base": "1", "inner": "x", "user.id": "block-user"}
    assert _extra(spans["after"]) == {"base": "1"}


def test_nested_new_scopes_compose_and_unwind(rec):
    with wardex.new_scope():
        wardex.set_tag("outer", "1")
        with wardex.new_scope():
            wardex.set_tag("inner", "2")
            with wardex.span("deep"):
                pass
        with wardex.span("middle"):
            pass
    spans = _spans(rec)
    assert _extra(spans["deep"]) == {"outer": "1", "inner": "2"}
    assert _extra(spans["middle"]) == {"outer": "1"}


def test_a_tag_set_inside_a_span_inside_new_scope_reaches_that_span(rec):
    with wardex.new_scope():
        with wardex.span("s"):
            wardex.set_tag("late", "yes")
    assert _extra(_spans(rec)["s"])["late"] == "yes"


def test_a_task_started_inside_new_scope_reads_it_and_keeps_its_writes(rec):
    async def child() -> None:
        wardex.set_tag("child", "1")
        with wardex.span("child-span"):
            pass

    async def main() -> None:
        with wardex.new_scope():
            wardex.set_tag("block", "1")
            await asyncio.create_task(child())
            with wardex.span("block-span"):
                pass

    asyncio.run(main())
    spans = _spans(rec)
    assert _extra(spans["child-span"]) == {"block": "1", "child": "1"}
    assert _extra(spans["block-span"]) == {"block": "1"}


# --------------------------------------------------------------------------
# continue_trace input
# --------------------------------------------------------------------------

_TRACE = "0af7651916cd43dd8448eb211c80319c"
_PARENT = "b7ad6b7169203331"


@pytest.mark.parametrize(
    "carrier",
    [f"00-{_TRACE}-{_PARENT}-01", {"traceparent": f"00-{_TRACE}-{_PARENT}-01"}],
    ids=["string", "mapping"],
)
def test_continue_trace_joins_from_a_traceparent_string_or_a_header_mapping(rec, carrier):
    with wardex.continue_trace(carrier):
        with wardex.span("joined"):
            pass
    joined = _spans(rec)["joined"]
    assert joined.context.trace_id.hex() == _TRACE
    assert joined.parent_span_id.hex() == _PARENT


@pytest.mark.parametrize("bad", [42, ["traceparent"], object()], ids=["int", "list", "object"])
def test_continue_trace_rejects_a_wrong_type_at_the_call(rec, bad):
    with pytest.raises(TypeError, match="continue_trace"):
        wardex.continue_trace(bad)


def test_a_malformed_network_traceparent_starts_a_new_trace_and_says_so_once(rec, caplog):
    with caplog.at_level("WARNING", logger="wardex_sdk"), warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            with wardex.continue_trace({"traceparent": "garbage"}):
                with wardex.span("fresh"):
                    pass
    fresh = _spans(rec)["fresh"]
    assert fresh.parent_span_id is None
    assert fresh.correlation.strategy.value == "unresolved"
    assert Limitation.PARENT_UNRESOLVED in _markers(fresh)
    said = [r for r in caplog.records if "traceparent" in r.getMessage()]
    assert len(said) == 1


def test_a_readable_or_absent_traceparent_leaves_a_root_unmarked(rec):
    with wardex.continue_trace({}):
        with wardex.span("fresh"):
            pass
    fresh = _spans(rec)["fresh"]
    assert fresh.correlation.strategy.value == "trace_root"
    assert Limitation.PARENT_UNRESOLVED not in _markers(fresh)


# --------------------------------------------------------------------------
# work on a thread that carries no wardex context
# --------------------------------------------------------------------------


def _in_plain_thread(fn) -> None:
    thread = threading.Thread(target=fn)
    thread.start()
    thread.join()


def _unbound_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "bind_context" in r.getMessage()]


def test_a_root_on_a_plain_thread_while_a_span_is_open_is_unresolved_and_said_once(rec, caplog):
    def work() -> None:
        with wardex.span("thread-root"):
            pass

    with caplog.at_level("WARNING", logger="wardex_sdk"):
        with wardex.span("request"):
            _in_plain_thread(work)
            _in_plain_thread(work)
    root = _spans(rec)["thread-root"]
    assert root.parent_span_id is None
    assert root.correlation.strategy.value == "unresolved"
    assert Limitation.PARENT_UNRESOLVED in _markers(root)
    said = _unbound_warnings(caplog)
    assert len(said) == 1
    assert "isolation_scope" in said[0]


def test_a_plain_thread_with_nothing_open_elsewhere_is_a_trace_root(rec, caplog):
    def work() -> None:
        with wardex.span("background-job"):
            pass

    with caplog.at_level("WARNING", logger="wardex_sdk"):
        _in_plain_thread(work)
    assert _spans(rec)["background-job"].correlation.strategy.value == "trace_root"
    assert not _unbound_warnings(caplog)


def test_a_plain_thread_that_opens_its_own_isolation_scope_is_a_trace_root(rec, caplog):
    def work() -> None:
        with wardex.isolation_scope(), wardex.span("job"):
            pass

    with caplog.at_level("WARNING", logger="wardex_sdk"):
        with wardex.span("request"):
            _in_plain_thread(work)
    assert _spans(rec)["job"].correlation.strategy.value == "trace_root"
    assert not _unbound_warnings(caplog)


def test_a_bound_thread_is_parented_and_not_warned_about(rec, caplog):
    def work() -> None:
        with wardex.span("bound-child"):
            pass

    with caplog.at_level("WARNING", logger="wardex_sdk"):
        with wardex.span("request"):
            _in_plain_thread(wardex.bind_context(work))
    spans = _spans(rec)
    assert spans["bound-child"].parent_span_id == spans["request"].context.span_id
    assert not _unbound_warnings(caplog)


def test_overlapping_requests_through_the_wsgi_middleware_are_not_warned_about(rec, caplog):
    import time

    def app(environ, start_response):
        with wardex.span("handler"):
            time.sleep(0.05)
        start_response("200 OK", [])
        return [b"ok"]

    middleware = wardex.WardexWsgiMiddleware(app)
    environ = {"HTTP_TRACEPARENT": f"00-{_TRACE}-{_PARENT}-01"}

    def request() -> None:
        list(middleware(dict(environ), lambda *a: None))

    with caplog.at_level("WARNING", logger="wardex_sdk"):
        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
            time.sleep(0.01)
        for thread in threads:
            thread.join()
    wardex.flush()
    handlers = [s for e in rec.envelopes for s in e.spans if s.name == "handler"]
    assert [s.parent_span_id.hex() for s in handlers] == [_PARENT, _PARENT]
    assert not _unbound_warnings(caplog)


def test_a_plain_thread_that_records_nothing_is_not_warned_about(rec, caplog):
    def health() -> None:
        assert wardex.get_traceparent() is None
        wardex.set_tag("probe", "1")

    with caplog.at_level("WARNING", logger="wardex_sdk"):
        with wardex.span("request"):
            _in_plain_thread(health)
    assert list(_spans(rec)) == ["request"]
    assert not _unbound_warnings(caplog)
