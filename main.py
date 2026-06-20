import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.loader import load_and_split_docs
from src.rag_chain import get_rag_response
from src.vector_store import load_or_create_vector_db


def main():
    parser = argparse.ArgumentParser(description="OpsMind 智能运维 RAG 助手")
    parser.add_argument("--file", default="data/test_doc.txt", help="知识库文档路径")
    parser.add_argument("--question", "-q", help="直接提问（不传则进入交互模式）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    args = parser.parse_args()

    print(">>> 加载并切分文档...")
    chunks = load_and_split_docs(args.file)
    if not chunks:
        print(f"错误：未能从 {args.file} 加载任何内容，请检查文件是否存在。")
        return

    print(f">>> 构建/加载向量库（共 {len(chunks)} 个片段）...")
    db = load_or_create_vector_db(chunks, rebuild=args.rebuild)

    if args.question:
        print(get_rag_response(db, args.question))
        return

    print(">>> 进入问答模式，输入 q 退出")
    while True:
        question = input("\n请输入你的问题：").strip()
        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            break
        print(get_rag_response(db, question))


if __name__ == "__main__":
    main()
