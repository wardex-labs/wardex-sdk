"""Sketch only: does an async TLS call fail on anyio 4.7.0 without wardex too?"""

import asyncio

import httpx
import pytest
from test_ssl_interceptor import _verify_ctx

import wardex_sdk as wardex
from wardex_sdk import _hub


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _who() -> str:
    task = asyncio.current_task()
    return f"TASK {type(task).__module__}.{type(task).__qualname__} is_c={isinstance(task, asyncio.Task)}"


async def _call(tls_server: str) -> int:
    async with httpx.AsyncClient(verify=_verify_ctx()) as client:
        resp = await client.post(f"{tls_server}/v1/messages", json={"model": "y"})
    return resp.status_code


@pytest.mark.asyncio
async def test_a_without_wardex(tls_server):
    print(_who())
    assert await _call(tls_server) == 200


@pytest.mark.asyncio
async def test_b_wardex_without_interception(tls_server):
    wardex.init(transport=wardex.NoOpTransport(), intercept=False)
    print(_who())
    assert await _call(tls_server) == 200


@pytest.mark.asyncio
async def test_c_wardex_with_interception(tls_server):
    wardex.init(transport=wardex.NoOpTransport(), intercept=True)
    print(_who())
    assert await _call(tls_server) == 200
