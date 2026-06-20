from pathlib import Path
from typing import List

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_split_docs(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """读取文档并切分为片段"""
    try:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix in ('.txt', '.md', '.log'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif suffix == '.pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            content = '\n\n'.join(doc.page_content for doc in docs)
        elif suffix == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = str(data)
        elif suffix == '.xml':
            from langchain_community.document_loaders import UnstructuredXMLLoader
            loader = UnstructuredXMLLoader(str(file_path))
            docs = loader.load()
            content = '\n\n'.join(doc.page_content for doc in docs)
        else:
            logger.warning(f"不支持的文件格式: {suffix}")
            return []
        
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            if end < len(content):
                last_newline = content.rfind('\n', start, end)
                if last_newline != -1 and last_newline > start:
                    end = last_newline
            chunks.append(content[start:end].strip())
            start = end - chunk_overlap
        
        logger.info(f"文档切分完成: {len(chunks)} 个片段")
        return chunks
        
    except Exception as e:
        logger.error(f"加载文件失败: {str(e)}")
        return []