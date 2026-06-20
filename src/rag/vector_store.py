"""向量存储模块 - 基于 ChromaDB 的本地向量库管理"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    向量存储管理器 - 处理文档向量化、存储和检索
    
    特性：
    - 支持本地 Embedding 模型（如 bge-m3, nomic-embed-text）
    - 增量更新：新文档入库不覆盖旧索引
    - ChromaDB 本地持久化
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-m3",
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None
    ):
        """
        初始化向量存储管理器
        
        Args:
            persist_dir: 向量库持久化目录
            embedding_model: Embedding 模型名称
            model_kwargs: 模型参数
            encode_kwargs: 编码参数
        """
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        
        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.model_kwargs = model_kwargs or {"device": "cpu"}
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}
        
        self._embeddings = None
        self._db = None
        
    def _get_embeddings(self):
        """
        获取 Embedding 模型
        
        Returns:
            HuggingFaceEmbeddings 实例
        """
        if self._embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs
                )
                logger.info(f"成功加载 Embedding 模型: {self.embedding_model}")
            except ImportError:
                logger.warning("langchain-huggingface 未安装，使用 langchain_community")
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs
                )
        
        return self._embeddings
    
    def _get_db(self):
        """
        获取或创建 ChromaDB 实例
        
        Returns:
            ChromaDB 实例
        """
        if self._db is None:
            try:
                from langchain_chroma import Chroma
                
                embeddings = self._get_embeddings()
                expected_dim = self._get_embedding_dimension(embeddings)
                
                if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
                    try:
                        self._db = Chroma(
                            persist_directory=str(self.persist_dir),
                            embedding_function=embeddings
                        )
                        logger.info(f"从本地加载向量库: {self.persist_dir}")
                    except Exception as e:
                        if "dimension" in str(e).lower():
                            logger.warning(f"向量库维度不匹配，重建向量库: {e}")
                            self._recreate_db(Chroma, embeddings)
                        else:
                            raise
                else:
                    self.persist_dir.mkdir(parents=True, exist_ok=True)
                    self._db = Chroma(
                        persist_directory=str(self.persist_dir),
                        embedding_function=embeddings
                    )
                    logger.info(f"创建新向量库: {self.persist_dir}")
                    
            except ImportError:
                logger.warning("langchain-chroma 未安装，使用 langchain_community")
                from langchain_community.vectorstores import Chroma
                
                embeddings = self._get_embeddings()
                
                if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
                    try:
                        self._db = Chroma(
                            persist_directory=str(self.persist_dir),
                            embedding_function=embeddings
                        )
                    except Exception as e:
                        if "dimension" in str(e).lower():
                            logger.warning(f"向量库维度不匹配，重建向量库: {e}")
                            self._recreate_db(Chroma, embeddings)
                        else:
                            raise
                else:
                    self.persist_dir.mkdir(parents=True, exist_ok=True)
                    self._db = Chroma(
                        persist_directory=str(self.persist_dir),
                        embedding_function=embeddings
                    )
        
        return self._db
    
    def _get_embedding_dimension(self, embeddings) -> int:
        """获取 Embedding 模型的维度"""
        try:
            sample_embedding = embeddings.embed_query("test")
            return len(sample_embedding)
        except Exception:
            return 0
    
    def _recreate_db(self, chroma_class, embeddings):
        """重建向量库（删除旧数据并创建新库）"""
        import shutil
        try:
            shutil.rmtree(str(self.persist_dir), ignore_errors=True)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._db = chroma_class(
                persist_directory=str(self.persist_dir),
                embedding_function=embeddings
            )
            logger.info(f"成功重建向量库: {self.persist_dir}")
        except Exception as e:
            logger.error(f"重建向量库失败: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        向向量库添加文档（增量更新）
        
        Args:
            texts: 文本内容列表
            metadatas: 元数据列表，与 texts 一一对应
            
        Returns:
            添加的文档 ID 列表
        """
        if not texts:
            logger.warning("没有提供要添加的文本")
            return []
        
        db = self._get_db()
        
        try:
            ids = db.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"成功添加 {len(ids)} 个文档片段")
            return ids
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"添加文档时维度不匹配，重建向量库: {e}")
                self._recreate_db_from_error()
                db = self._get_db()
                ids = db.add_texts(texts=texts, metadatas=metadatas)
                logger.info(f"重建后成功添加 {len(ids)} 个文档片段")
                return ids
            else:
                raise
    
    def _recreate_db_from_error(self):
        """从错误中重建向量库"""
        import shutil
        try:
            self._db = None
            shutil.rmtree(str(self.persist_dir), ignore_errors=True)
            logger.info(f"已删除旧向量库目录: {self.persist_dir}")
        except Exception as e:
            logger.error(f"删除旧向量库失败: {e}")
            raise
    
    def add_documents_from_files(self, file_paths: List[Union[str, Path]]):
        """
        从文件添加文档到向量库
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            添加的文档 ID 列表
        """
        all_ids = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if pd is not None:
                    timestamp = str(pd.Timestamp.now())
                else:
                    timestamp = str(datetime.now())
                
                metadata = {
                    "source": file_path.name,
                    "path": str(file_path),
                    "added_at": timestamp
                }
                
                ids = self.add_documents([content], [metadata])
                all_ids.extend(ids)
                
                logger.info(f"成功从文件添加文档: {file_path.name}")
                
            except Exception as e:
                logger.error(f"从文件添加文档失败 {file_path}: {e}")
        
        return all_ids
    
    def search(self, query: str, k: int = 3) -> List[dict]:
        """
        检索与查询最相关的文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            检索结果列表，每个结果包含 page_content 和 metadata
        """
        db = self._get_db()
        retriever = db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return results
    
    def get_document_count(self) -> int:
        """
        获取向量库中的文档数量
        
        Returns:
            文档数量
        """
        db = self._get_db()
        return db._collection.count()
    
    def delete_documents(self, ids: List[str]):
        """
        删除指定 ID 的文档
        
        Args:
            ids: 要删除的文档 ID 列表
        """
        db = self._get_db()
        db.delete(ids=ids)
        logger.info(f"成功删除 {len(ids)} 个文档")
    
    def clear_all(self):
        """
        清空向量库中的所有文档
        """
        db = self._get_db()
        db.delete(ids=db.get()['ids'])
        logger.info("向量库已清空")