from enum import Enum


class CaptureMode(Enum):
    """Which intercepted traffic becomes spans (design §5.1).

    AGENT (default): LLM-semantic traffic always; generic HTTP/gRPC/WS only
    inside a *local* wardex span. ALL: everything (pre-Phase-4 behavior).
    """

    AGENT = "agent"
    ALL = "all"


class PIIMode(Enum):
    MASK = "mask"
    REDACT = "redact"
    HASH = "hash"
    OFF = "off"


class PIICategory(Enum):
    """Built-in PII detection categories (design §5.1). Values are the FFI
    contract with the Rust pattern registry — never rename casually."""

    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    CREDIT_CARD = "credit_card"
    US_SSN = "us_ssn"
    IP_ADDRESS = "ip_address"
    US_BANK_ROUTING = "us_bank_routing"
    IBAN = "iban"
    SECRET = "secret"


class AdapterName(Enum):
    ANTHROPIC_AGENT_SDK = "anthropic_agent_sdk"
    LANGGRAPH = "langgraph"
    LANGCHAIN = "langchain"
    OPENAI_AGENTS = "openai_agents"


class InterceptorName(Enum):
    SSL = "ssl"
    MCP_STDIO = "mcp_stdio"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    SSE = "sse"


class StatusCode(Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanKind(Enum):
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"


class OperationName(Enum):
    """CLOSED at twelve — design §6.2. The `operation` half of a span name.

    The three newest members (`EXECUTE_STEP`, `HANDOFF`, `EVALUATE`) each cover a
    concept at least two surveyed frameworks have and no existing member can hold
    honestly: a graph node is not an agent and not a tool; an agent transition
    needs to be a span or every dashboard needs a special case for it; a
    guardrail/judge already has `EvaluationAttributes` and needed only a span
    kind. There is no fourth — protocol variants are attributes
    (`ToolExecutionType.IPC`), never new operations.
    """

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    INVOKE_WORKFLOW = "invoke_workflow"
    GENERATE_CONTENT = "generate_content"
    RETRIEVAL = "retrieval"
    EXECUTE_STEP = "execute_step"
    HANDOFF = "handoff"
    EVALUATE = "evaluate"


class ProviderName(Enum):
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI_INFERENCE = "azure.ai.inference"
    AZURE_AI_OPENAI = "azure.ai.openai"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    GCP_GEMINI = "gcp.gemini"
    GCP_GEN_AI = "gcp.gen_ai"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    GROQ = "groq"
    IBM_WATSONX_AI = "ibm.watsonx.ai"
    MISTRAL_AI = "mistral_ai"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"


class OutputType(Enum):
    IMAGE = "image"
    JSON = "json"
    SPEECH = "speech"
    TEXT = "text"


class ToolType(Enum):
    FUNCTION = "function"
    EXTENSION = "extension"
    DATASTORE = "datastore"


class AgentType(Enum):
    PRIMARY = "primary"
    SUB_AGENT = "sub_agent"
    TEAM_MEMBER = "team_member"


class ToolExecutionType(Enum):
    """How the tool body actually ran — design §6.2.

    `IPC` and `UNKNOWN` are the newest members and they exist for the same
    reason: the two values this enum had forced a guess. Every adapter tool span
    said `NETWORK`, which is false for an MCP call over a subprocess pipe (that
    is `IPC`) and unknowable for a CLI's built-in tools, whose implementation
    wardex never observes (that is `UNKNOWN`). A wrong value is worse than an
    absent one — `UNKNOWN` is a legitimate, honest answer.
    """

    NETWORK = "network"
    IN_PROCESS = "in_process"
    IPC = "ipc"
    UNKNOWN = "unknown"


class Protocol(Enum):
    HTTP = "http"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MCP_STDIO = "mcp_stdio"
    SSE = "sse"


class Direction(Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


class SnapshotType(Enum):
    """CLOSED, 1:1 with `proto/wardex/v1/common.proto`'s `SNAPSHOT_TYPE_*`.

    `InternalStateSnapshot.snapshot_type` is still typed `str` for wire and
    signature compatibility; this enum is what the SDK coerces to before the
    record is built, so an unrecognized value degrades to
    `Limitation.SNAPSHOT_TYPE_UNKNOWN` at one place instead of reaching the
    codec, where `map_snap` would silently map it to `UNSPECIFIED` with nothing
    recorded (design §4.5).
    """

    SPAN_START = "span_start"
    SPAN_END = "span_end"
    TURN_START = "turn_start"


class SessionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CRASHED = "crashed"


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"


class CaptureSource(Enum):
    # aligned with proto common.proto CAPTURE_SOURCE_* (layer prefix removed)
    ADAPTER = "adapter"
    SSL = "ssl"
    SOCKET = "socket"  # plaintext raw-socket seam; proto sync is a follow-up task
    STDIO = "stdio"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MANUAL = "manual"


class RetentionClass(Enum):
    SUMMARY_ONLY = "summary_only"
    REPLAYABLE = "replayable"
    FORENSIC = "forensic"


class CaptureTrigger(Enum):
    ERROR = "error"
    HIGH_LATENCY = "high_latency"
    HIGH_COST = "high_cost"
    POLICY_VIOLATION = "policy_violation"
    MANUAL_MARK = "manual_mark"
    USER_REPORT = "user_report"
