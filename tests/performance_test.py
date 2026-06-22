"""性能测试模块 - 测试检索速度、回答延迟、命中率和准确率"""

import time
import logging
from typing import List, Dict, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """性能测试器 - 评估RAG系统性能指标"""
    
    def __init__(self, vector_db, rag_chain_func):
        """
        初始化性能测试器
        
        Args:
            vector_db: 向量数据库实例
            rag_chain_func: RAG问答函数
        """
        self.vector_db = vector_db
        self.rag_chain_func = rag_chain_func
        self.results = []
    
    def test_retrieval_speed(self, queries: List[str], k: int = 3, iterations: int = 5) -> Dict:
        """
        测试检索速度
        
        Args:
            queries: 查询列表
            k: 返回文档数量
            iterations: 迭代次数
            
        Returns:
            性能指标字典
        """
        logger.info(f"开始检索速度测试，{len(queries)}个查询，{iterations}次迭代")
        
        total_times = []
        per_query_times = []
        
        for query in queries:
            query_times = []
            for _ in range(iterations):
                start_time = time.time()
                try:
                    self.vector_db.search(query, k=k)
                except Exception as e:
                    logger.error(f"检索失败: {e}")
                    continue
                elapsed = time.time() - start_time
                query_times.append(elapsed)
            
            avg_query_time = sum(query_times) / len(query_times) if query_times else 0
            per_query_times.append({"query": query, "avg_time": avg_query_time})
            total_times.extend(query_times)
        
        result = {
            "metric": "retrieval_speed",
            "avg_time": sum(total_times) / len(total_times) if total_times else 0,
            "min_time": min(total_times) if total_times else 0,
            "max_time": max(total_times) if total_times else 0,
            "p95_time": self._calculate_percentile(total_times, 95),
            "queries": per_query_times,
            "test_time": datetime.now().isoformat()
        }
        
        logger.info(f"检索速度测试完成: 平均 {result['avg_time']:.4f}s")
        return result
    
    def test_response_latency(self, queries: List[str], iterations: int = 3) -> Dict:
        """
        测试回答延迟（端到端）
        
        Args:
            queries: 查询列表
            iterations: 迭代次数
            
        Returns:
            性能指标字典
        """
        logger.info(f"开始回答延迟测试，{len(queries)}个查询，{iterations}次迭代")
        
        total_times = []
        per_query_times = []
        
        for query in queries:
            query_times = []
            for _ in range(iterations):
                start_time = time.time()
                try:
                    self.rag_chain_func(self.vector_db, query)
                except Exception as e:
                    logger.error(f"回答失败: {e}")
                    continue
                elapsed = time.time() - start_time
                query_times.append(elapsed)
            
            avg_query_time = sum(query_times) / len(query_times) if query_times else 0
            per_query_times.append({"query": query, "avg_time": avg_query_time})
            total_times.extend(query_times)
        
        result = {
            "metric": "response_latency",
            "avg_time": sum(total_times) / len(total_times) if total_times else 0,
            "min_time": min(total_times) if total_times else 0,
            "max_time": max(total_times) if total_times else 0,
            "p95_time": self._calculate_percentile(total_times, 95),
            "queries": per_query_times,
            "test_time": datetime.now().isoformat()
        }
        
        logger.info(f"回答延迟测试完成: 平均 {result['avg_time']:.4f}s")
        return result
    
    def test_accuracy(self, test_cases: List[Dict]) -> Dict:
        """
        测试准确率（基于标注数据）
        
        Args:
            test_cases: 测试用例列表，每个用例包含 question, expected_answers
            
        Returns:
            准确率指标字典
        """
        logger.info(f"开始准确率测试，{len(test_cases)}个测试用例")
        
        correct_count = 0
        results = []
        
        for test_case in test_cases:
            question = test_case["question"]
            expected_answers = test_case["expected_answers"]
            
            try:
                response = self.rag_chain_func(self.vector_db, question)
                response_lower = response.lower()
                
                is_correct = any(
                    answer.lower() in response_lower 
                    for answer in expected_answers
                )
                
                if is_correct:
                    correct_count += 1
                
                results.append({
                    "question": question,
                    "expected_answers": expected_answers,
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "is_correct": is_correct
                })
            except Exception as e:
                logger.error(f"测试用例失败 {question}: {e}")
                results.append({
                    "question": question,
                    "expected_answers": expected_answers,
                    "response": f"Error: {e}",
                    "is_correct": False
                })
        
        accuracy = correct_count / len(test_cases) if test_cases else 0
        
        result = {
            "metric": "accuracy",
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(test_cases),
            "test_cases": results,
            "test_time": datetime.now().isoformat()
        }
        
        logger.info(f"准确率测试完成: {accuracy:.2%} ({correct_count}/{len(test_cases)})")
        return result
    
    def test_hit_rate(self, queries: List[str], k: int = 3, threshold: float = 0.7) -> Dict:
        """
        测试检索命中率
        
        Args:
            queries: 查询列表
            k: 返回文档数量
            threshold: 相关性阈值（0-1）
            
        Returns:
            命中率指标字典
        """
        logger.info(f"开始命中率测试，{len(queries)}个查询")
        
        hit_count = 0
        results = []
        
        for query in queries:
            try:
                docs = self.vector_db.search(query, k=k)
                
                has_hit = len(docs) > 0
                if has_hit:
                    hit_count += 1
                
                results.append({
                    "query": query,
                    "has_hit": has_hit,
                    "doc_count": len(docs)
                })
            except Exception as e:
                logger.error(f"命中率测试失败 {query}: {e}")
                results.append({
                    "query": query,
                    "has_hit": False,
                    "doc_count": 0
                })
        
        hit_rate = hit_count / len(queries) if queries else 0
        
        result = {
            "metric": "hit_rate",
            "hit_rate": hit_rate,
            "hit_count": hit_count,
            "total_count": len(queries),
            "test_cases": results,
            "test_time": datetime.now().isoformat()
        }
        
        logger.info(f"命中率测试完成: {hit_rate:.2%} ({hit_count}/{len(queries)})")
        return result
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(percentile / 100 * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def run_all_tests(self, config: Dict) -> Dict:
        """运行所有测试"""
        logger.info("=" * 60)
        logger.info("开始运行完整性能测试套件")
        logger.info("=" * 60)
        
        all_results = {}
        
        if "retrieval_queries" in config:
            all_results["retrieval_speed"] = self.test_retrieval_speed(
                config["retrieval_queries"],
                k=config.get("k", 3),
                iterations=config.get("iterations", 5)
            )
        
        if "latency_queries" in config:
            all_results["response_latency"] = self.test_response_latency(
                config["latency_queries"],
                iterations=config.get("iterations", 3)
            )
        
        if "accuracy_test_cases" in config:
            all_results["accuracy"] = self.test_accuracy(
                config["accuracy_test_cases"]
            )
        
        if "hit_rate_queries" in config:
            all_results["hit_rate"] = self.test_hit_rate(
                config["hit_rate_queries"],
                k=config.get("k", 3)
            )
        
        logger.info("=" * 60)
        logger.info("性能测试套件运行完成")
        logger.info("=" * 60)
        
        return all_results


def print_test_report(results: Dict):
    """打印测试报告"""
    print("\n" + "=" * 60)
    print("性能测试报告")
    print("=" * 60)
    
    if "retrieval_speed" in results:
        rs = results["retrieval_speed"]
        print(f"\n[检索速度]")
        print(f"  平均耗时: {rs['avg_time']:.4f}s")
        print(f"  最小耗时: {rs['min_time']:.4f}s")
        print(f"  最大耗时: {rs['max_time']:.4f}s")
        print(f"  P95耗时: {rs['p95_time']:.4f}s")
    
    if "response_latency" in results:
        rl = results["response_latency"]
        print(f"\n[回答延迟]")
        print(f"  平均耗时: {rl['avg_time']:.4f}s")
        print(f"  最小耗时: {rl['min_time']:.4f}s")
        print(f"  最大耗时: {rl['max_time']:.4f}s")
        print(f"  P95耗时: {rl['p95_time']:.4f}s")
    
    if "accuracy" in results:
        acc = results["accuracy"]
        print(f"\n[准确率]")
        print(f"  准确率: {acc['accuracy']:.2%}")
        print(f"  正确数: {acc['correct_count']}/{acc['total_count']}")
    
    if "hit_rate" in results:
        hr = results["hit_rate"]
        print(f"\n[命中率]")
        print(f"  命中率: {hr['hit_rate']:.2%}")
        print(f"  命中数: {hr['hit_count']}/{hr['total_count']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
    
    from src.rag.vector_store import VectorStoreManager
    from src.rag_chain import get_rag_response
    
    vector_db = VectorStoreManager(persist_dir="./chroma_db")
    
    tester = PerformanceTester(vector_db, get_rag_response)
    
    test_config = {
        "k": 3,
        "iterations": 3,
        "retrieval_queries": [
            "MySQL主从延迟如何排查",
            "Redis缓存雪崩怎么处理",
            "Kubernetes Pod OOMKilled解决方法",
            "Nginx返回502错误怎么办",
            "CPU使用率过高如何定位"
        ],
        "latency_queries": [
            "MySQL主从延迟如何排查",
            "Redis缓存雪崩怎么处理",
            "Kubernetes Pod OOMKilled解决方法"
        ],
        "hit_rate_queries": [
            "MySQL主从延迟如何排查",
            "Redis缓存雪崩怎么处理",
            "Kubernetes Pod OOMKilled解决方法",
            "Nginx返回502错误怎么办",
            "CPU使用率过高如何定位",
            "TCP连接数过多怎么处理",
            "内存泄漏排查方法",
            "磁盘IO性能优化"
        ],
        "accuracy_test_cases": [
            {
                "question": "MySQL主从延迟如何排查？",
                "expected_answers": ["show slave status", "seconds_behind_master", "主从同步"]
            },
            {
                "question": "Redis缓存雪崩怎么处理？",
                "expected_answers": ["多级缓存", "热点数据", "熔断", "限流"]
            },
            {
                "question": "Kubernetes Pod OOMKilled怎么解决？",
                "expected_answers": ["资源限制", "内存分配", "requests", "limits"]
            },
            {
                "question": "Nginx返回502错误怎么办？",
                "expected_answers": ["上游服务器", "backend", "proxy_pass", "重启"]
            },
            {
                "question": "CPU使用率过高如何定位？",
                "expected_answers": ["top", "pidstat", "perf", "进程"]
            }
        ]
    }
    
    results = tester.run_all_tests(test_config)
    print_test_report(results)