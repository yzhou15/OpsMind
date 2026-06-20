#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建向量库 - 清洗数据并入库（优化版）"""

import os
import sys
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl.cleaner import create_cleaner
from src.rag.retriever import DocumentRetriever
from src.rag.vector_store import VectorStoreManager
from configs.settings import ChunkConfig, EmbeddingConfig


def get_file_tags(file_name: str) -> list:
    """根据文件名推断问题类型标签"""
    tags = []
    
    if "mysql" in file_name.lower() or "database" in file_name.lower():
        tags.extend(["mysql", "database", "sql"])
    if "redis" in file_name.lower() or "cache" in file_name.lower():
        tags.extend(["redis", "cache"])
    if "kubernetes" in file_name.lower() or "k8s" in file_name.lower():
        tags.extend(["kubernetes", "k8s", "pod"])
    if "payment" in file_name.lower() or "pay" in file_name.lower():
        tags.extend(["payment", "支付"])
    if "error" in file_name.lower() or "exception" in file_name.lower():
        tags.extend(["error", "exception"])
    if "alert" in file_name.lower() or "monitor" in file_name.lower():
        tags.extend(["alert", "monitoring"])
    if "incident" in file_name.lower():
        tags.extend(["incident", "故障"])
    if "network" in file_name.lower():
        tags.extend(["network", "网络"])
    if "linux" in file_name.lower():
        tags.extend(["linux", "运维"])
    
    return tags


def get_data_type(file_name: str) -> str:
    """根据扩展名推断数据类型"""
    ext = os.path.splitext(file_name)[1].lower()
    type_map = {
        ".json": "json",
        ".xml": "xml",
        ".log": "log",
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
    }
    return type_map.get(ext, "unknown")


def main():
    print("=" * 60)
    print("OpsMind 向量库构建（优化版）")
    print("=" * 60)

    # 删除旧向量库（确保使用新维度重建）
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir):
        try:
            import shutil
            shutil.rmtree(persist_dir)
            print(f"✓ 已删除旧向量库目录")
        except Exception as e:
            print(f"⚠️ 删除旧向量库失败: {e}")
            print("  请手动删除 chroma_db 目录后重新运行")
            sys.exit(1)

    # 1. 清洗数据
    print("\n[1/3] 数据清洗...")
    cleaner = create_cleaner()
    raw_files = glob.glob("data/raw/*")
    processed_data = []

    for f in raw_files:
        if os.path.isfile(f):
            result = cleaner.process_file(f)
            if result.success:
                file_name = os.path.basename(f)
                data_type = get_data_type(file_name)
                tags = get_file_tags(file_name)
                
                processed_data.append({
                    "content": result.content,
                    "file_name": file_name,
                    "data_type": data_type,
                    "tags": tags,
                    "record_count": result.stats.get("total_records", 0)
                })
                print(f"  ✓ {file_name}: {result.stats.get('total_records', 0)} 条记录, 标签: {tags}")
            else:
                print(f"  ✗ {os.path.basename(f)}: {result.error}")

    print(f"\n清洗完成: {len(processed_data)} 个文件")

    # 2. 文档切片（使用配置参数）
    print("\n[2/3] 文档切片...")
    retriever = DocumentRetriever(
        chunk_size=ChunkConfig.CHUNK_SIZE,
        chunk_overlap=ChunkConfig.CHUNK_OVERLAP
    )
    
    all_chunks = []
    all_metadatas = []
    global_chunk_index = 0

    for item in processed_data:
        chunks = retriever.split_text(item["content"])
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "source_file": item["file_name"],
                "data_type": item["data_type"],
                "tags": ",".join(item["tags"]),
                "chunk_index": global_chunk_index,
                "file_chunk_index": i,
                "total_chunks_in_file": len(chunks),
                "chunk_length": len(chunk),
                "added_at": datetime.now().isoformat()
            }
            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            global_chunk_index += 1

        print(f"  {item['file_name']}: {len(chunks)} 个片段")

    info = retriever.get_chunk_info(all_chunks)
    print(f"\n切片统计:")
    print(f"  总片段数: {info['total_chunks']}")
    print(f"  平均长度: {info['avg_length']} 字符")
    print(f"  配置: chunk_size={ChunkConfig.CHUNK_SIZE}, chunk_overlap={ChunkConfig.CHUNK_OVERLAP}")

    # 3. 向量入库（使用配置中的 embedding 模型）
    print("\n[3/3] 向量入库...")
    persist_dir = "./chroma_db"

    store = VectorStoreManager(
        persist_dir=persist_dir,
        embedding_model=EmbeddingConfig.MODEL_NAME
    )

    # 先清空
    try:
        store.clear_all()
        print("  已清空旧向量库")
    except Exception as e:
        print(f"  注意: {e}")

    # 批量添加
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]
        store.add_documents(batch, batch_metadatas)
        print(f"  批次 {i//batch_size + 1}: 添加 {len(batch)} 个片段")

    count = store.get_document_count()
    print(f"\n✅ 向量库构建完成: {count} 个文档片段")
    print(f"   Embedding 模型: {EmbeddingConfig.MODEL_NAME}")

    # 4. 检索测试
    print("\n" + "=" * 60)
    print("检索测试")
    print("=" * 60)

    questions = [
        "MySQL 主从延迟",
        "Redis 缓存雪崩",
        "Kubernetes OOMKilled",
        "支付超时",
        "数据库连接池",
        "Nginx 502",
        "磁盘空间耗尽",
    ]

    for q in questions:
        results = store.search(q, k=2)
        print(f"\n❓ {q}")
        if results:
            for j, doc in enumerate(results, 1):
                preview = doc["content"][:80].replace("\n", " ")
                tags = doc["metadata"].get("tags", "")
                source = doc["metadata"].get("source_file", "")
                print(f"  [{j}] {preview}...")
                print(f"     来源: {source}, 标签: {tags}")
        else:
            print("  ⚠️ 无结果")

    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
