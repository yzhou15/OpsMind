#!/usr/bin/env python3
"""OpsMind API 服务 - FastAPI 后端接口"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_BASE_URL"] = "https://hf-mirror.com"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.vector_store import load_vector_db
from src.rag_chain import get_rag_response

app = FastAPI(title="OpsMind 智能运维助手 API", version="1.0")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    model: Optional[str] = "qwen2.5:7b"

class QueryResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "opsmind-api"

db = None

@app.on_event("startup")
async def startup_event():
    global db
    try:
        db = load_vector_db()
        print("✅ 向量数据库加载成功")
    except Exception as e:
        print(f"❌ 向量数据库加载失败: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not db:
        raise HTTPException(status_code=503, detail="向量数据库未就绪")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        answer = get_rag_response(db, request.question, model=request.model, top_k=request.top_k)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    if not db:
        raise HTTPException(status_code=503, detail="向量数据库未就绪")
    doc_count = db._collection.count()
    return {"document_count": doc_count, "persist_directory": db._persist_directory}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
