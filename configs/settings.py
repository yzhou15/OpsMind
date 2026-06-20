"""
OpsMind 智能运维助手 - 配置文件
所有可调整参数集中管理，便于后续优化和运维
"""

from pathlib import Path
from typing import Literal

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
CONFIG_DIR = BASE_DIR / "configs"

# 确保目录存在
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, CHROMA_DB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ==================== 文档切片配置 ====================
class ChunkConfig:
    """文档切片配置"""
    CHUNK_SIZE: int = 1000  # 每个切片的目标字符数
    CHUNK_OVERLAP: int = 200  # 相邻切片的重叠字符数
    SEPARATORS: list[str] = [
        "\n\n",  # 优先按段落分割
        "\n",    # 然后按行分割
        "。", "！", "？",  # 中文句子
        ".", "!", "?",    # 英文句子
        " ", ""
    ]


# ==================== Embedding 模型配置 ====================
class EmbeddingConfig:
    """向量嵌入模型配置"""
    # 推荐使用 BGE-M3（多语言支持）或本地 text2vec 模型
    # 为避免外网依赖，建议使用 HuggingFace 镜像
    MODEL_NAME: str = "BAAI/bge-m3"
    # 备选: "shibing624/text2vec-base-chinese"
    # 备选: "nomic-ai/nomic-embed-text-v1.5"

    # HuggingFace 镜像（国内网络）
    HF_ENDPOINT: str = "https://hf-mirror.com"
    HF_HOME: str = str(BASE_DIR / ".cache" / "huggingface")


# ==================== LLM 模型配置 ====================
class LLMConfig:
    """大语言模型配置"""
    # 支持通过 Ollama 或 vLLM 调用本地模型
    PROVIDER: Literal["ollama", "vllm"] = "ollama"

    # Ollama 配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"  # 或 "llama3:8b"
    OLLAMA_TEMPERATURE: float = 0.3
    OLLAMA_TIMEOUT: int = 120

    # vLLM 配置（备选）
    VLLM_BASE_URL: str = "http://localhost:8000"
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"


# ==================== RAG 检索配置 ====================
class RAGConfig:
    """RAG 检索配置"""
    TOP_K: int = 3  # 检索返回的最相关文档数
    MAX_CONTEXT_CHARS: int = 3000  # 上下文最大字符数（防止超出模型窗口）
    SIMILARITY_THRESHOLD: float = 0.5  # 相似度阈值（低于此值不返回）


# ==================== ETL 清洗配置 ====================
class ETLConfig:
    """数据清洗配置"""
    # 日志级别过滤
    LOG_LEVELS_TO_KEEP: list[str] = ["ERROR", "WARN", "WARNING", "INFO", "FATAL", "CRITICAL"]
    LOG_LEVELS_TO_DROP: list[str] = ["DEBUG", "TRACE", "VERBOSE"]

    # 敏感信息正则（会被替换为 [REDACTED]）
    SENSITIVE_PATTERNS: list[str] = [
        r'\b\d{3}-\d{2,4}-\d{4,}\b',  # 社会保险号
        r'\b\d{16,19}\b',             # 信用卡号
        r'password["\s:=]+\S+',       # 密码明文
        r'api[_-]?key["\s:=]+\S+',    # API Key
        r'token["\s:=]+\S+',          # Token
    ]

    # 支持的文件格式
    SUPPORTED_FORMATS: list[str] = [".json", ".xml", ".log", ".txt", ".md", ".pdf"]

    # 输出格式
    OUTPUT_FORMAT: Literal["markdown", "txt"] = "markdown"


# ==================== Docker 部署配置 ====================
class DockerConfig:
    """Docker 容器化配置"""
    IMAGE_NAME: str = "opsmind-rag"
    IMAGE_TAG: str = "latest"
    CONTAINER_NAME: str = "opsmind"
    HTTP_PORT: int = 8501  # Streamlit 默认端口
    OLLAMA_PORT: int = 11434

    # 安全配置：禁止暴露公网
    BIND_ADDRESS: str = "127.0.0.1"

    # 数据卷挂载
    VOLUME_DATA: str = "opsmind_data"
    VOLUME_MODELS: str = "opsmind_models"


# ==================== 系统配置 ====================
class SystemConfig:
    """系统级配置"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 中文编码
    DEFAULT_ENCODING: str = "utf-8"
