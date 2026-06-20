"""检索器模块 - 实现滑动窗口文档切片和语义检索"""

import logging
from typing import List, Optional, Union
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    文档检索器 - 处理文档切片、滑动窗口和语义检索
    
    特性：
    - 实现滑动窗口机制 (Sliding Window)
    - 支持自定义 chunk_size 和 chunk_overlap
    - 保留跨段落语义连贯性
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        初始化文档检索器
        
        Args:
            chunk_size: 每个文档片段的最大字符数
            chunk_overlap: 相邻片段之间的重叠字符数
            separators: 自定义分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            self.separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        else:
            self.separators = separators
        
        logger.info(f"初始化检索器: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """
        使用滑动窗口机制切分文本
        
        Args:
            text: 原始文本内容
            
        Returns:
            切分后的文档片段列表
        """
        if not text:
            logger.warning("输入文本为空")
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            if end < text_length:
                best_split = -1
                
                for sep in self.separators:
                    idx = text.rfind(sep, start, end)
                    if idx != -1 and idx > start:
                        best_split = idx + len(sep)
                        break
                
                if best_split != -1:
                    end = best_split
                else:
                    end = start + self.chunk_size
            else:
                end = text_length
            
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            
            start = next_start
        
        logger.info(f"文本切分完成: 共 {len(chunks)} 个片段")
        return chunks
    
    def split_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        切分文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            切分后的文档片段列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.split_text(content)
            
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            return []
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return []
    
    def batch_split(self, file_paths: List[Union[str, Path]]) -> List[str]:
        """
        批量切分多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            所有文件切分后的文档片段列表
        """
        all_chunks = []
        
        for file_path in file_paths:
            chunks = self.split_file(file_path)
            all_chunks.extend(chunks)
            logger.info(f"处理文件: {file_path}, 生成 {len(chunks)} 个片段")
        
        return all_chunks
    
    def verify_overlap(self, chunks: List[str], sample_size: int = 3) -> None:
        """
        验证滑动窗口重叠是否生效
        
        Args:
            chunks: 切分后的文档片段列表
            sample_size: 要检查的相邻片段对数量
        """
        if len(chunks) < 2:
            logger.warning("片段数量不足，无法验证重叠")
            return
        
        logger.info("验证滑动窗口重叠效果：")
        
        for i in range(min(sample_size, len(chunks) - 1)):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            if len(current_chunk) >= self.chunk_overlap and len(next_chunk) >= self.chunk_overlap:
                overlap_candidate = current_chunk[-self.chunk_overlap:]
                
                if overlap_candidate in next_chunk[:self.chunk_overlap * 2]:
                    logger.info(f"✓ 片段 {i+1} 和 {i+2} 重叠验证通过")
                    logger.info(f"  重叠内容: {overlap_candidate[:50]}...")
                else:
                    logger.warning(f"✗ 片段 {i+1} 和 {i+2} 未检测到预期重叠")
    
    def get_chunk_info(self, chunks: List[str]) -> dict:
        """
        获取切分统计信息
        
        Args:
            chunks: 切分后的文档片段列表
            
        Returns:
            统计信息字典
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0
            }
        
        lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) // len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }