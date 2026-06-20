from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen2.5:7b"
MAX_CONTEXT_CHARS = 3000
TOP_K = 3

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个智能运维助手。请严格根据提供的知识库内容回答问题。"
        "如果知识库中没有相关信息，请如实说明，不要编造。",
    ),
    ("human", "【知识库内容】\n{context}\n\n【用户问题】\n{question}"),
])


def _truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """滑动窗口截断：限制 Prompt 上下文长度，避免超出模型窗口。"""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n...(内容已截断)"


def get_rag_response(
    vector_db,
    question: str,
    model: str = DEFAULT_MODEL,
    top_k: int = TOP_K,
) -> str:
    """检索相关知识片段，结合本地大模型生成回答。"""
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    context = _truncate_context("\n\n".join(doc.page_content for doc in docs))

    llm = ChatOllama(model=model, temperature=0.3)
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
