"""Interceptors protocol parsers (Rust native wrapper)."""

from .._wardex_native import protocol as _native_protocol
from ._base import ProtocolParserInterface
from ._http1 import Http1RequestParser, Http1ResponseParser
from ._http2 import Http2Parser

parse_llm_semantics = _native_protocol.parse_llm_semantics
normalize_finish_reason = _native_protocol.normalize_finish_reason
parse_grpc_frames = _native_protocol.parse_grpc_frames
grpc_status_name = _native_protocol.grpc_status_name
JsonRpcParser = _native_protocol.JsonRpcParser
WsParser = _native_protocol.WsParser

__all__ = [
    "ProtocolParserInterface",
    "Http1RequestParser",
    "Http1ResponseParser",
    "Http2Parser",
    "parse_llm_semantics",
    "normalize_finish_reason",
    "parse_grpc_frames",
    "grpc_status_name",
    "JsonRpcParser",
    "WsParser",
]
