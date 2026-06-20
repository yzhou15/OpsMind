import os

# 在模块导入时就设置 HuggingFace 镜像，避免后续网络请求走官方地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_BASE_URL"] = "https://hf-mirror.com"

from pathlib import Path

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

DEFAULT_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_db(chunks, persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """构建向量数据库并持久化到本地。"""
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )


def load_vector_db(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """从本地加载已有向量库。"""
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def load_or_create_vector_db(
    chunks,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    rebuild: bool = False,
) -> Chroma:
    """加载已有向量库；不存在或指定 rebuild 时重新创建。"""
    persist_path = Path(persist_directory)
    if not rebuild and persist_path.exists() and any(persist_path.iterdir()):
        return load_vector_db(persist_directory)
    return create_vector_db(chunks, persist_directory)