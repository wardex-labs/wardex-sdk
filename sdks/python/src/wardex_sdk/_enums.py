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
    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    INVOKE_WORKFLOW = "invoke_workflow"
    GENERATE_CONTENT = "generate_content"
    RETRIEVAL = "retrieval"


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
    NETWORK = "network"
    IN_PROCESS = "in_process"


class Protocol(Enum):
    HTTP = "http"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MCP_STDIO = "mcp_stdio"
    SSE = "sse"


class Direction(Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


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
