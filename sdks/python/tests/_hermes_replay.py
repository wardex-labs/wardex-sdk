"""`Replay`: plays a recorded Hermes fixture through a `FakePluginContext`, with the threads,
contexts and order Hermes delivered it in.

Fixture format (`tests/fixtures/hermes/<scenario>.jsonl`, `"version": 1`). Line 1 is a header:

* `format`, `version`, `scenario`, `hermes` (the tag recorded from), `mode` (`library` or `cli`).
* `root`: the thread that called the turn and the probe value its context held (`["caller"]` in
  library mode, where the lab runner set it before `run_conversation`; `null` in a gateway job).
* `threads`: every other thread that issued an event, with `role`, `copier` (the thread whose
  context it copied), `copy_before` (the copy is taken on the copier right before that event
  `seq`) and `context` (the probe its first event saw). The copier and copy point are DERIVED from
  the measured thread model, so each entry says `"derived": ["copier", "copy_before"]`.

Each further line is one event, in recorded order (`seq`; `rseq` is the raw recording's number,
`null` for a derived event):

* `kind` `hook`: `name`, `lane` (from `HOOK_LANES`), `thread` (where the callback ran; omitted when
  it is `on`), `on` (the thread that fired it), `probe` (the context the callback saw), `payload`.
* `kind` `mw`: `name` is the middleware kind. Request kinds carry `payload`. Execution kinds come
  as an `enter` (`frame`, `payload`) and an `exit` (`frame`, `outcome` `returned` or `raised`,
  and for a raise the original error's `module`, `type` and `text` under `raised`).
* `kind` `http`: a `request` (`attempt`, `reply`, `status`, `stream`, `usage`) and a `response`
  (`attempt`, `status`), both on the thread that made the attempt.
* `derived`: the fields of this line that were not recorded but derived; see the README.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _hermes_fake import (
    Answer,
    Exchange,
    FakePluginContext,
    InternalServerError,
    Reply,
    ScriptedProvider,
    Usage,
)
from _hermes_turns import TOOL_RESULT

FIXTURES = Path(__file__).parent / "fixtures" / "hermes"
FORMAT_VERSION = 1
SCENARIOS = (
    "single_turn one_tool one_tool-cli parallel_tools inline_seq mixed_batch mcp_seq parallel_mcp "
    "retry two_subagents two_subagents_joined auxiliary_title interrupted failed_turn"
).split()


@dataclass(frozen=True)
class Recorded:
    """One fixture line after the header. See the module docstring for the fields."""

    seq: int
    kind: str  # "hook", "mw" or "http"
    name: str
    on: str
    probe: tuple[str, ...] | None
    thread: str
    rseq: int | None = None
    phase: str | None = None
    lane: str | None = None
    frame: str | None = None
    outcome: str | None = None
    raised: dict[str, str] | None = None
    attempt: int | None = None
    reply: str | None = None
    status: int | None = None
    stream: bool | None = None
    usage: dict[str, int] | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    derived: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fixture:
    header: dict[str, Any]
    events: tuple[Recorded, ...]

    @property
    def root(self) -> str:
        return self.header["root"]["thread"]

    def threads(self) -> dict[str, dict[str, Any]]:
        return {t["name"]: t for t in self.header["threads"]}


def load_fixture(name: str) -> Fixture:
    lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    if header.get("format") != "wardex.hermes.fixture" or header.get("version") != FORMAT_VERSION:
        raise ValueError(f"{name}: not a version {FORMAT_VERSION} Hermes fixture")
    events = []
    for line in lines[1:]:
        raw = json.loads(line)
        raw.setdefault("thread", raw["on"])
        probe = raw.pop("probe")
        events.append(
            Recorded(
                probe=None if probe is None else tuple(probe),
                derived=tuple(raw.pop("derived", ())),
                **raw,
            )
        )
    return Fixture(header, tuple(events))


class ReplayError(AssertionError):
    pass


class _Clock:
    """Lets each thread act only when its next step is the next in recorded order."""

    def __init__(self, keys: list[tuple[int, int]]) -> None:
        self._keys = sorted(keys)
        self._i = 0
        self._cond = threading.Condition()
        self.error: BaseException | None = None

    def wait(self, key: tuple[int, int], timeout: float = 20.0) -> None:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self.error is not None or self._keys[self._i] == key, timeout
            )
            if self.error is not None:
                raise ReplayError("another replay thread failed") from self.error
            if not ok:
                raise ReplayError(f"step {key} never came; stuck at {self._keys[self._i]}")

    def done(self, key: tuple[int, int]) -> None:
        with self._cond:
            assert self._keys[self._i] == key, (self._keys[self._i], key)
            self._i += 1
            self._cond.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._cond:
            if self.error is None:
                self.error = error
            self._cond.notify_all()

    @property
    def finished(self) -> bool:
        return self._i == len(self._keys)


@dataclass
class Replay:
    """Play one recorded fixture through `ctx`'s callbacks with Hermes's thread structure.

    Every recorded thread is a real thread with the recorded name. Each starts from a copy of its
    copier's context taken at the header's `copy_before` point, so a callback sees the context it
    saw under Hermes. Steps run strictly in recorded order: a middleware `enter` runs the chain on
    its thread, and the chain's terminal plays the steps recorded inside the frame, then returns
    or raises as the `exit` says; hooks fire through `ctx.invoke_hook` from the thread that fired
    them (stream hooks are drained before the next step, so the order is the recorded one); HTTP
    steps send and read a real request to `provider` from the recorded thread. The root thread
    starts in a copy of the context `run()` is called in.
    """

    ctx: FakePluginContext
    provider: ScriptedProvider

    def run(self, name: str) -> Fixture:
        fixture = load_fixture(name)
        steps: dict[str, list[tuple[tuple[int, int], Any]]] = {}
        for ev in fixture.events:
            actor = _STREAM_ACTOR if ev.lane == "stream" else ev.on
            steps.setdefault(actor, []).append(((ev.seq, 1), ev))
        for spec in fixture.header["threads"]:
            steps.setdefault(spec["copier"], []).append(((spec["copy_before"], 0), spec))
        for actor in steps.values():
            actor.sort(key=lambda step: step[0])
        clock = _Clock([key for actor in steps.values() for key, _ in actor])
        held: list[Exchange] = []
        exchanges: dict[int, Exchange] = {}
        threads: list[threading.Thread] = []
        snapshots: dict[str | None, contextvars.Context] = {None: contextvars.copy_context()}

        def actor_main(actor: str) -> None:
            try:
                plan = steps.get(actor, [])
                index = play(plan, 0, None)
                if index != len(plan):
                    raise ReplayError(f"{actor}: stray exit at step {plan[index][0]}")
            except BaseException as exc:
                clock.fail(exc)

        def start(spec: dict[str, Any]) -> None:
            copied = contextvars.copy_context()
            thread = threading.Thread(
                target=copied.run, args=(actor_main, spec["name"]), name=spec["name"], daemon=True
            )
            threads.append(thread)
            thread.start()

        def play(plan: list, index: int, frame: str | None) -> int:
            while index < len(plan):
                key, step = plan[index]
                if isinstance(step, dict):
                    clock.wait(key)
                    start(step)
                    clock.done(key)
                    index += 1
                    continue
                if step.kind == "mw" and step.phase == "exit":
                    if step.frame != frame:
                        raise ReplayError(f"exit of {step.frame} inside {frame}")
                    return index
                clock.wait(key)
                if step.kind == "mw" and step.phase == "enter":
                    index = enter(plan, index, step, key)
                    continue
                if step.kind == "hook" and step.lane == "stream":
                    frames = [f for f in step.probe or () if f in snapshots]
                    snapshot = snapshots[frames[-1] if frames else None]
                    snapshot.copy().run(self.ctx.invoke_hook, step.name, **dict(step.payload))
                    self.ctx.drain_streams()
                elif step.kind == "hook":
                    self.ctx.invoke_hook(step.name, **dict(step.payload))
                elif step.kind == "mw":
                    payload_key = "request" if step.name.startswith("llm") else "args"
                    payload = dict(step.payload)
                    payload.setdefault(payload_key, {})
                    self.ctx.run_request_chain(step.name, payload_key, **payload)
                elif step.name == "request":
                    reply = Reply(step.reply)
                    usage = step.usage or {}
                    answer = Answer(
                        reply,
                        Usage(
                            usage["prompt_tokens"],
                            usage["completion_tokens"],
                            usage["cached_tokens"],
                        )
                        if usage
                        else None,
                    )
                    exchanges[step.attempt] = self.provider.send(answer)
                else:
                    exchange = exchanges.pop(step.attempt)
                    status = exchange.head()
                    if status != step.status:
                        raise ReplayError(f"attempt {step.attempt}: {status} != {step.status}")
                    if Reply(steps_reply[step.attempt]) is Reply.HOLD:
                        held.append(exchange)
                    else:
                        exchange.finish()
                clock.done(key)
                index += 1
            if frame is not None:
                raise ReplayError(f"frame {frame} has no exit")
            return index

        steps_reply = {e.attempt: e.reply for e in fixture.events if e.name == "request"}

        def enter(plan: list, index: int, step: Recorded, key: tuple[int, int]) -> int:
            box: dict[str, Any] = {}

            def terminal(payload: Any) -> Any:
                snapshots[step.frame] = contextvars.copy_context()
                clock.done(key)
                box["exit"] = exit_at = play(plan, index + 1, step.frame)
                exit_key, exit_step = plan[exit_at]
                clock.wait(exit_key)
                if exit_step.outcome == "raised":
                    for exchange in held:
                        exchange.cut()
                    self.provider.release()
                    raise _replayed(exit_step.raised or {})
                return (
                    SimpleNamespace(replayed=True) if step.name == "llm_execution" else TOOL_RESULT
                )

            payload = dict(step.payload)
            payload_key = "request" if step.name == "llm_execution" else "args"
            payload.setdefault(payload_key, {})
            payload.setdefault("original_" + payload_key, payload[payload_key])
            try:
                self.ctx.run_execution_chain(step.name, terminal, **payload)
                raised = None
            except Exception as exc:
                raised = exc
            if "exit" not in box:
                raise ReplayError(f"{step.frame}: the chain never reached Hermes's own call")
            exit_key, exit_step = plan[box["exit"]]
            wanted = exit_step.outcome == "raised"
            if wanted != (raised is not None) or (
                wanted and type(raised).__name__ != (exit_step.raised or {}).get("type")
            ):
                raise ReplayError(f"{step.frame}: recorded {exit_step.outcome}, got {raised!r}")
            clock.done(exit_key)
            return box["exit"] + 1

        streams = threading.Thread(
            target=actor_main, args=(_STREAM_ACTOR,), name=_STREAM_ACTOR, daemon=True
        )
        threads.append(streams)
        streams.start()
        root = threading.Thread(
            target=contextvars.copy_context().run,
            args=(actor_main, fixture.root),
            name=fixture.root,
            daemon=True,
        )
        threads.append(root)
        root.start()
        for thread in itertools.chain([root], threads):
            thread.join(30)
        for thread in list(threads):
            thread.join(30)
        for exchange in held:
            exchange.cut()
        if clock.error is not None:
            raise clock.error
        if not clock.finished:
            raise ReplayError(f"{name}: replay stopped early")
        return fixture


#: Stream hooks are queued by Hermes with a snapshot of the enqueuing thread's context and
#: delivered later, sometimes after their frame closed (measured: `on_stream_end` after the
#: `llm_execution` exit, still showing the frame). This actor delivers each at its recorded place,
#: from a snapshot taken when the frame named last in its probe reached Hermes's own call.
_STREAM_ACTOR = "replay-stream-enqueue"

_ERRORS: dict[tuple[str, str], type[Exception]] = {
    ("builtins", "InterruptedError"): InterruptedError,
    ("openai", "InternalServerError"): InternalServerError,
}


def _replayed(raised: dict[str, str]) -> Exception:
    """The original error Hermes's provider call raised, by its recorded module and type."""
    kind = _ERRORS.get((raised.get("module", ""), raised.get("type", "")))
    if kind is None:
        kind = type(raised.get("type", "RecordedError"), (Exception,), {})
    return kind(raised.get("text", ""))
