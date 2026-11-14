"""Read by a type checker, never run: the decorators keep the wrapped signature.

Each `# type: ignore[arg-type]` below marks a call with a wrong argument. Under
`--warn-unused-ignores` an ignore that suppresses nothing is itself an error,
so a decorator that erased the signature to `Callable[..., Any]` fails the
check: the wrong call would no longer be an error to ignore.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import wardex_sdk as wardex


@wardex.tool
def search(query: str, limit: int = 3) -> list[str]:
    return [query] * limit


@wardex.agent(name="planner")
async def plan(goal: str) -> str:
    return goal


@wardex.step
def produce(count: int) -> Iterator[int]:
    yield from range(count)


@wardex.workflow(name="stream")
async def stream(count: int) -> AsyncIterator[int]:
    for i in range(count):
        yield i


def sync_calls() -> list[str]:
    search(1)  # type: ignore[arg-type]
    numbers: list[int] = list(produce(2))
    return search("cats", limit=len(numbers))


async def async_calls() -> str:
    await plan(goal=3)  # type: ignore[arg-type]
    total = 0
    async for item in stream(2):
        total += item
    return await plan("ship")
