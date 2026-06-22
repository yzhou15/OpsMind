"""Embedding模型对比实验 - 测试不同模型的效果"""

import time
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingComparison:
    """Embedding模型对比实验器"""
    
    MODELS = [
        {
            "name": "bge-m3",
            "model_name": "BAAI/bge-m3",
            "description": "多语言通用Embedding模型，支持向量维度1024",
            "dimensions": 1024
        },
        {
            "name": "text2vec-base-chinese",
            "model_name": "shibing624/text2vec-base-chinese",
            "description": "中文文本向量化模型，支持向量维度768",
            "dimensions": 768
        },
        {
            "name": "nomic-embed-text",
            "model_name": "nomic-ai/nomic-embed-text-v1.5",
            "description": "开源Embedding模型，支持多语言",
            "dimensions": 768
        },
        {
            "name": "all-mpnet-base-v2",
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "description": "通用语义相似度模型",
            "dimensions": 768
        }
    ]
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        初始化对比实验器
        
        Args:
            data_dir: 测试数据目录
        """
        self.data_dir = Path(data_dir)
        self.results = []
    
    def _load_test_data(self) -> List[str]:
        """加载测试数据"""
        texts = []
        for file_path in self.data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                logger.warning(f"读取文件失败 {file_path}: {e}")
        
        for file_path in self.data_dir.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                logger.warning(f"读取文件失败 {file_path}: {e}")
        
        return texts
    
    def _load_test_queries(self) -> List[str]:
        """加载测试查询"""
        return [
            "MySQL主从延迟如何排查",
            "Redis缓存雪崩怎么处理",
            "Kubernetes Pod OOMKilled解决方法",
            "Nginx返回502错误怎么办",
            "CPU使用率过高如何定位",
            "TCP连接数过多怎么处理",
            "内存泄漏排查方法",
            "磁盘IO性能优化",
            "网络丢包问题分析",
            "日志分析方法"
        ]
    
    def _create_vector_store(self, model_name: str, persist_dir: str) -> 'VectorStoreManager':
        """创建向量存储"""
        from src.rag.vector_store import VectorStoreManager
        
        # 删除旧的向量库
        persist_path = Path(persist_dir)
        if persist_path.exists():
            shutil.rmtree(str(persist_path), ignore_errors=True)
        
        return VectorStoreManager(
            persist_dir=persist_dir,
            embedding_model=model_name
        )
    
    def compare_models(self) -> List[Dict]:
        """对比测试所有模型"""
        logger.info("=" * 60)
        logger.info("开始Embedding模型对比实验")
        logger.info("=" * 60)
        
        test_texts = self._load_test_data()
        test_queries = self._load_test_queries()
        
        logger.info(f"测试数据: {len(test_texts)}个文档")
        logger.info(f"测试查询: {len(test_queries)}个问题")
        
        all_results = []
        
        for model_info in self.MODELS:
            model_name = model_info["model_name"]
            short_name = model_info["name"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"测试模型: {short_name} ({model_name})")
            logger.info(f"描述: {model_info['description']}")
            logger.info(f"{'='*60}")
            
            try:
                persist_dir = f"./chroma_db_{short_name}"
                vector_db = self._create_vector_store(model_name, persist_dir)
                
                # 测试模型加载时间
                load_start = time.time()
                embeddings = vector_db._get_embeddings()
                load_time = time.time() - load_start
                logger.info(f"模型加载时间: {load_time:.4f}s")
                
                # 测试向量入库时间
                if test_texts:
                    import uuid
                    texts = []
                    metadatas = []
                    for i, text in enumerate(test_texts):
                        texts.append(text[:2000])
                        metadatas.append({"source": f"test_{i}"})
                    
                    insert_start = time.time()
                    ids = vector_db.add_documents(texts, metadatas)
                    insert_time = time.time() - insert_start
                    logger.info(f"向量入库时间: {insert_time:.4f}s ({len(ids)}个文档)")
                
                # 测试检索速度
                query_times = []
                for query in test_queries:
                    query_start = time.time()
                    try:
                        vector_db.search(query, k=3)
                        query_times.append(time.time() - query_start)
                    except Exception as e:
                        logger.warning(f"检索失败 {query}: {e}")
                
                avg_query_time = sum(query_times) / len(query_times) if query_times else 0
                logger.info(f"检索速度: {avg_query_time:.4f}s/query")
                
                # 获取文档数量
                doc_count = vector_db.get_document_count()
                
                # 获取向量维度
                try:
                    sample_embedding = embeddings.embed_query("test")
                    actual_dim = len(sample_embedding)
                except Exception as e:
                    actual_dim = model_info["dimensions"]
                
                result = {
                    "model_name": model_name,
                    "short_name": short_name,
                    "description": model_info["description"],
                    "expected_dimensions": model_info["dimensions"],
                    "actual_dimensions": actual_dim,
                    "model_load_time": load_time,
                    "insert_time": insert_time if test_texts else 0,
                    "avg_query_time": avg_query_time,
                    "doc_count": doc_count,
                    "test_queries": len(test_queries),
                    "success": True
                }
                
                all_results.append(result)
                
                # 清理临时向量库
                persist_path = Path(persist_dir)
                if persist_path.exists():
                    shutil.rmtree(str(persist_path), ignore_errors=True)
                
            except Exception as e:
                logger.error(f"测试模型 {short_name} 失败: {e}")
                all_results.append({
                    "model_name": model_name,
                    "short_name": short_name,
                    "description": model_info["description"],
                    "expected_dimensions": model_info["dimensions"],
                    "actual_dimensions": 0,
                    "model_load_time": 0,
                    "insert_time": 0,
                    "avg_query_time": 0,
                    "doc_count": 0,
                    "test_queries": 0,
                    "success": False,
                    "error": str(e)
                })
        
        logger.info("\n" + "=" * 60)
        logger.info("Embedding模型对比实验完成")
        logger.info("=" * 60)
        
        return all_results
    
    def print_comparison_report(self, results: List[Dict]):
        """打印对比报告"""
        print("\n" + "=" * 80)
        print("Embedding模型对比实验报告")
        print("=" * 80)
        
        print(f"\n{'模型名称':<25} {'维度':>8} {'加载时间':>10} {'入库时间':>10} {'检索速度':>10} {'状态':>8}")
        print("-" * 80)
        
        for result in results:
            status = "✓" if result["success"] else "✗"
            print(f"{result['short_name']:<25} {result['actual_dimensions']:>8} "
                  f"{result['model_load_time']:>10.4f}s {result['insert_time']:>10.4f}s "
                  f"{result['avg_query_time']:>10.4f}s {status:>8}")
        
        print("\n模型详情:")
        for result in results:
            print(f"\n  {result['short_name']}:")
            print(f"    完整名称: {result['model_name']}")
            print(f"    描述: {result['description']}")
            print(f"    向量维度: {result['actual_dimensions']}")
            if result["success"]:
                print(f"    模型加载时间: {result['model_load_time']:.4f}s")
                print(f"    向量入库时间: {result['insert_time']:.4f}s")
                print(f"    平均检索速度: {result['avg_query_time']:.4f}s")
            else:
                print(f"    状态: 失败 - {result['error']}")
        
        # 推荐最佳模型
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            best_model = min(successful_results, key=lambda x: x["avg_query_time"])
            print(f"\n  推荐模型: {best_model['short_name']} (检索速度最快)")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
    
    comparison = EmbeddingComparison(data_dir="./data/raw")
    results = comparison.compare_models()
    comparison.print_comparison_report(results)