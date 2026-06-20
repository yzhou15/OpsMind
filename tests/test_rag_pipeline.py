#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpsMind RAG 端到端测试脚本

测试流程:
1. 数据清洗 - 使用 ETL cleaner 处理原始数据
2. 文档切片 - 使用 DocumentRetriever 滑动窗口切分
3. 向量入库 - 使用 VectorStoreManager 增量添加
4. 检索问答 - 使用 PromptManager 构建 Prompt 并调用 LLM
"""

import os
import sys
import glob
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl.cleaner import create_cleaner
from src.rag.retriever import DocumentRetriever
from src.rag.vector_store import VectorStoreManager
from src.rag.prompt_templates import PromptManager
from src.rag_chain import get_rag_response


def clean_raw_data():
    """Step 1: 清洗原始数据"""
    print("=" * 60)
    print("Step 1: 数据清洗")
    print("=" * 60)

    cleaner = create_cleaner()
    raw_files = glob.glob("data/raw/*")
    cleaned_contents = []
    file_metadata = []

    for file_path in raw_files:
        if os.path.isfile(file_path):
            print(f"\n处理文件: {os.path.basename(file_path)}")
            result = cleaner.process_file(file_path)

            if result.success:
                print(f"  ✅ 成功 - 记录数: {result.stats.get('total_records', 'N/A')}")
                cleaned_contents.append(result.content)
                file_metadata.append({
                    "source": os.path.basename(file_path),
                    "type": result.metadata.get("data_type", "unknown"),
                    "records": result.stats.get("total_records", 0)
                })
            else:
                print(f"  ❌ 失败 - {result.error}")

    print(f"\n清洗完成: {len(cleaned_contents)}/{len(raw_files)} 个文件")
    return cleaned_contents, file_metadata


def split_documents(contents):
    """Step 2: 使用滑动窗口切分文档"""
    print("\n" + "=" * 60)
    print("Step 2: 文档切片 (滑动窗口)")
    print("=" * 60)

    retriever = DocumentRetriever(chunk_size=500, chunk_overlap=100)
    all_chunks = []

    for i, content in enumerate(contents):
        chunks = retriever.split_text(content)
        all_chunks.extend(chunks)
        print(f"文档 {i+1}: {len(chunks)} 个片段")

    # 验证重叠
    if len(all_chunks) >= 2:
        retriever.verify_overlap(all_chunks, sample_size=3)

    # 统计信息
    info = retriever.get_chunk_info(all_chunks)
    print(f"\n切片统计:")
    print(f"  总片段数: {info['total_chunks']}")
    print(f"  平均长度: {info['avg_length']} 字符")
    print(f"  最小长度: {info['min_length']} 字符")
    print(f"  最大长度: {info['max_length']} 字符")

    return all_chunks


def build_vector_store(chunks):
    """Step 3: 构建向量库"""
    print("\n" + "=" * 60)
    print("Step 3: 向量库构建")
    print("=" * 60)

    # 使用现有向量库或重建
    persist_dir = "./chroma_db"

    # 为了测试，先清空重新建
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print("已清空旧向量库")

    store = VectorStoreManager(
        persist_dir=persist_dir,
        embedding_model="shibing624/text2vec-base-chinese"
    )

    # 分批添加，避免一次性加载过多
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        metadatas = [{"chunk_index": i + j, "source": "rag_test"} for j in range(len(batch))]
        ids = store.add_documents(batch, metadatas)
        print(f"  批次 {i//batch_size + 1}: 添加 {len(ids)} 个片段")

    count = store.get_document_count()
    print(f"\n向量库构建完成: 共 {count} 个文档片段")
    return store


def test_retrieval(store):
    """Step 4: 测试检索功能"""
    print("\n" + "=" * 60)
    print("Step 4: 检索测试")
    print("=" * 60)

    test_questions = [
        "MySQL 主从延迟怎么解决？",
        "Redis 缓存雪崩是什么原因？",
        "Kubernetes Pod 被 OOMKilled 怎么处理？",
        "支付网关超时怎么处理？",
        "数据库连接池耗尽了怎么办？",
        "Nginx 502 错误可能是什么原因？",
    ]

    for question in test_questions:
        print(f"\n❓ 问题: {question}")
        results = store.search(question, k=3)

        if results:
            print(f"  检索到 {len(results)} 条相关文档:")
            for i, doc in enumerate(results, 1):
                preview = doc["content"][:100].replace("\n", " ")
                print(f"    [{i}] {preview}...")
        else:
            print("  ⚠️ 未检索到相关文档")


def test_qa_with_llm(store):
    """Step 5: 使用 LLM 进行问答测试"""
    print("\n" + "=" * 60)
    print("Step 5: LLM 问答测试")
    print("=" * 60)

    questions = [
        "MySQL 主从延迟超过 30 秒怎么排查和解决？",
        "Redis 缓存雪崩有什么预防措施？",
        "Java 应用在 Kubernetes 中 OOMKilled 怎么解决？",
    ]

    for question in questions:
        print(f"\n{'=' * 50}")
        print(f"❓ 问题: {question}")
        print("=" * 50)

        try:
            start_time = time.time()
            answer = get_rag_response(store._db, question, model="qwen2.5:7b")
            elapsed = time.time() - start_time

            print(f"\n💡 回答:")
            print(answer)
            print(f"\n⏱️  响应时间: {elapsed:.2f} 秒")

        except Exception as e:
            print(f"\n❌ 问答失败: {e}")


def main():
    """主测试流程"""
    print("\n" + "🚀" * 30)
    print("  OpsMind RAG 端到端测试")
    print("🚀" * 30 + "\n")

    # Step 1: 数据清洗
    contents, metadata = clean_raw_data()

    if not contents:
        print("\n❌ 没有成功清洗的数据，测试终止")
        return

    # Step 2: 文档切片
    chunks = split_documents(contents)

    # Step 3: 向量库构建
    store = build_vector_store(chunks)

    # Step 4: 检索测试
    test_retrieval(store)

    # Step 5: LLM 问答测试
    print("\n" + "⚠️" * 20)
    print("  注意: LLM 问答测试需要 Ollama 服务运行")
    print("  模型: qwen2.5:7b")
    print("⚠️" * 20 + "\n")

    try:
        test_qa_with_llm(store)
    except Exception as e:
        print(f"\n⚠️ LLM 测试跳过: {e}")
        print("请确保 Ollama 已启动: ollama serve")

    print("\n" + "=" * 60)
    print("RAG 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
