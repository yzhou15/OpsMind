#!/usr/bin/env python3
"""初始化知识库：批量加载文档并构建向量库"""

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.loader import load_and_split_docs
from src.vector_store import create_vector_db


def load_all_docs(data_dir: str = "data") -> list:
    """加载数据目录下的所有文档"""
    all_chunks = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误：数据目录 {data_dir} 不存在")
        return []
    
    supported_extensions = (".txt", ".pdf", ".json", ".xml", ".log", ".md")
    
    for file_path in sorted(data_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            print(f"正在加载：{file_path.name}")
            chunks = load_and_split_docs(str(file_path))
            all_chunks.extend(chunks)
            print(f"  -> 生成 {len(chunks)} 个片段")
    
    return all_chunks


def main():
    print("=" * 60)
    print("初始化 OpsMind 智能运维知识库")
    print("=" * 60)
    
    chunks = load_all_docs()
    
    if not chunks:
        print("没有加载到任何文档，请检查 data 目录")
        return
    
    print(f"\n共加载 {len(chunks)} 个文档片段")
    
    print("\n正在构建向量库...")
    db = create_vector_db(chunks)
    print("向量库构建完成！")
    
    print("\n验证向量库：")
    print(f"  - 文档数量：{db._collection.count()}")
    print(f"  - 持久化目录：{db._persist_directory}")
    
    print("\n" + "=" * 60)
    print("知识库初始化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()