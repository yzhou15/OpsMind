"""
OpsMind 数据清洗模块 (Data ETL Pipeline)

本模块负责将原始异构数据（JSON/XML/日志）转换为 LLM 可理解的高质量文本。

功能特性:
    - 多格式解析: JSON, XML, LOG, TXT
    - 日志级别过滤: 保留 ERROR/WARN/INFO, 丢弃 DEBUG/TRACE
    - 敏感信息脱敏: 自动识别并替换密码、API Key、身份证等
    - Markdown 格式化输出: 包含元数据头（来源、时间、类型）

典型用法:
    >>> from src.etl.cleaner import DataCleaner
    >>> cleaner = DataCleaner()
    >>> result = cleaner.process_file("data/raw/app.log")
    >>> print(result)
"""

import json
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """支持的数据类型枚举"""
    JSON = "json"
    XML = "xml"
    LOG = "log"
    TXT = "txt"
    MD = "md"
    PDF = "pdf"
    UNKNOWN = "unknown"


@dataclass
class CleaningResult:
    """数据清洗结果容器

    Attributes:
        content: 清洗后的文本内容
        metadata: 元数据字典（来源、时间、类型等）
        success: 处理是否成功
        error: 错误信息（如果有）
        stats: 统计信息（原始行数、过滤行数等）
    """
    content: str
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    stats: dict = field(default_factory=dict)


class SensitiveDataMasker:
    """敏感信息脱敏处理器

    使用正则表达式识别并替换常见的敏感信息模式。
    """

    # 默认敏感信息模式（可配置）
    DEFAULT_PATTERNS = [
        # 社会保险号: XXX-XX-XXXX 或 XXXXXXXXX
        (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED-SSN]'),
        # 信用卡号: 16-19位数字
        (r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED-CARD]'),
        # 身份证号: 18位
        (r'\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b', '[REDACTED-ID]'),
        # 密码明文: password=xxx 或 "password": "xxx"
        (r'(?:password|pwd|passwd)["\s:=]+\S+', '[REDACTED-PWD]', re.IGNORECASE),
        # API Key: api_key=xxx 或 "apiKey": "xxx"
        (r'(?:api[_-]?key|apikey|api_secret)["\s:=]+\S+', '[REDACTED-APIKEY]', re.IGNORECASE),
        # Token: token=xxx 或 Authorization: Bearer xxx
        (r'(?:token|authorization)["\s:=]+\S+', '[REDACTED-TOKEN]', re.IGNORECASE),
        # IP 地址（可选脱敏）
        # (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[REDACTED-IP]'),
        # 邮箱地址（可选脱敏）
        # (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED-EMAIL]'),
    ]

    def __init__(self, patterns: Optional[list] = None):
        """初始化脱敏器

        Args:
            patterns: 自定义正则模式列表，每项为 (pattern, replacement) 或 (pattern, replacement, flags)
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译所有正则表达式以提升性能"""
        self._compiled = []
        for p in self.patterns:
            if len(p) == 2:
                pattern, replacement = p
                flags = 0
            else:
                pattern, replacement, flags = p
            try:
                compiled = re.compile(pattern, flags)
                self._compiled.append((compiled, replacement))
            except re.error as e:
                logger.warning(f"无效的正则表达式: {pattern}, 错误: {e}")

    def mask(self, text: str) -> str:
        """对文本进行敏感信息脱敏

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        result = text
        for pattern, replacement in self._compiled:
            result = pattern.sub(replacement, result)
        return result


class BaseParser(ABC):
    """数据解析器基类"""

    @abstractmethod
    def parse(self, content: str) -> list[dict]:
        """解析内容为结构化数据

        Args:
            content: 原始文本内容

        Returns:
            解析后的字典列表
        """
        pass

    @abstractmethod
    def to_markdown(self, data: list[dict], metadata: dict) -> str:
        """将结构化数据转换为 Markdown 格式

        Args:
            data: 解析后的数据
            metadata: 元数据

        Returns:
            Markdown 格式的文本
        """
        pass


class JSONParser(BaseParser):
    """JSON 格式解析器

    支持解析嵌套的 JSON 结构，提取关键字段并扁平化。
    """

    # JSON 到 Markdown 的字段映射（可扩展）
    FIELD_ALIASES = {
        "timestamp": ["timestamp", "time", "created_at", "datetime", "@timestamp"],
        "level": ["level", "severity", "loglevel", "log_level"],
        "message": ["message", "msg", "content", "description", "detail", "error"],
        "source": ["source", "host", "server", "service", "component"],
    }

    def __init__(self, key_path: Optional[str] = None):
        """初始化 JSON 解析器

        Args:
            key_path: jq 风格的路径表达式，用于提取嵌套数据，如 ".data.items[]"
        """
        self.key_path = key_path

    def parse(self, content: str) -> list[dict]:
        """解析 JSON 文本

        Args:
            content: JSON 格式的字符串

        Returns:
            提取的记录列表

        Raises:
            json.JSONDecodeError: JSON 格式错误
        """
        try:
            data = json.loads(content)
            records = self._extract_records(data)
            logger.info(f"JSON 解析成功，提取 {len(records)} 条记录")
            return records
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            raise

    def _extract_records(self, data) -> list[dict]:
        """递归提取记录，支持嵌套结构"""
        if isinstance(data, list):
            records = []
            for item in data:
                records.extend(self._extract_records(item))
            return records
        elif isinstance(data, dict):
            # 如果包含 timestamp/level/message 等关键字段，视为记录
            if self._is_log_record(data):
                return [self._normalize_record(data)]
            # 否则尝试提取嵌套的数组
            records = []
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            records.extend(self._extract_records(item))
            return records if records else [self._normalize_record(data)]
        return []

    def _is_log_record(self, record: dict) -> bool:
        """判断是否为日志记录（包含关键字段）"""
        key_count = sum(
            1 for aliases in self.FIELD_ALIASES.values()
            for alias in aliases
            if alias in record
        )
        return key_count >= 2

    def _normalize_record(self, record: dict) -> dict:
        """标准化记录字段名"""
        normalized = {}
        for standard_name, aliases in self.FIELD_ALIASES.items():
            for alias in aliases:
                if alias in record:
                    normalized[standard_name] = record[alias]
                    break
        # 保留其他未识别的字段
        for key, value in record.items():
            if key not in [a for aliases in self.FIELD_ALIASES.values() for a in aliases]:
                normalized[key] = value
        return normalized

    def to_markdown(self, data: list[dict], metadata: dict) -> str:
        """将 JSON 数据转换为 Markdown

        Args:
            data: 解析后的数据列表
            metadata: 元数据

        Returns:
            Markdown 格式文本
        """
        lines = [
            "---",
            f"**数据来源**: {metadata.get('source_file', 'unknown')}",
            f"**数据格式**: JSON",
            f"**处理时间**: {metadata.get('process_time', datetime.now().isoformat())}",
            f"**记录数量**: {len(data)}",
            "---\n"
        ]

        for i, record in enumerate(data, 1):
            lines.append(f"### 记录 {i}")

            # 提取关键字段
            timestamp = record.get("timestamp", "N/A")
            level = record.get("level", "INFO").upper()
            message = record.get("message", "")
            source = record.get("source", "unknown")

            if timestamp != "N/A":
                lines.append(f"**时间**: {timestamp}")
            if level != "INFO":
                lines.append(f"**级别**: {level}")
            if source != "unknown":
                lines.append(f"**来源**: {source}")

            if message:
                lines.append(f"\n**内容**: {message}")

            # 其他字段
            other_fields = {
                k: v for k, v in record.items()
                if k not in ("timestamp", "level", "message", "source") and v
            }
            if other_fields:
                lines.append("\n**附加信息**:")
                for k, v in other_fields.items():
                    lines.append(f"- {k}: {v}")

            lines.append("")
        return "\n".join(lines)


class XMLParser(BaseParser):
    """XML 格式解析器

    支持解析 XML 文档，提取元素内容和属性。
    """

    def __init__(self, record_tag: Optional[str] = None):
        """初始化 XML 解析器

        Args:
            record_tag: 记录标签名，如 "log", "entry", "item"。None 则自动检测。
        """
        self.record_tag = record_tag

    def parse(self, content: str) -> list[dict]:
        """解析 XML 文本

        Args:
            content: XML 格式的字符串

        Returns:
            提取的记录列表

        Raises:
            ET.ParseError: XML 格式错误
        """
        try:
            root = ET.fromstring(content)
            records = self._extract_records(root)
            logger.info(f"XML 解析成功，提取 {len(records)} 条记录")
            return records
        except ET.ParseError as e:
            logger.error(f"XML 解析失败: {e}")
            raise

    def _extract_records(self, root) -> list[dict]:
        """从 XML 根元素提取记录"""
        records = []

        # 自动检测记录标签
        if self.record_tag:
            elements = root.findall(f".//{self.record_tag}")
        else:
            # 查找可能的记录标签
            for tag in ["log", "entry", "item", "record", "event", "row"]:
                elements = root.findall(f".//{tag}")
                if elements:
                    self.record_tag = tag
                    break
            else:
                elements = [root]

        for elem in elements:
            record = self._element_to_dict(elem)
            records.append(record)

        return records

    def _element_to_dict(self, element) -> dict:
        """将 XML 元素转换为字典"""
        result = {"_tag": element.tag}

        # 提取属性
        if element.attrib:
            result["_attributes"] = dict(element.attrib)

        # 提取子元素
        for child in element:
            child_data = self._element_to_dict(child)

            if child.tag in result and not isinstance(result[child.tag], list):
                # 多个同名子元素，转换为列表
                result[child.tag] = [result.pop(child.tag), child_data]
            elif child.tag in result:
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        # 提取文本内容（如果没有子元素）
        if not list(element) and element.text:
            result["_text"] = element.text.strip()

        return result

    def to_markdown(self, data: list[dict], metadata: dict) -> str:
        """将 XML 数据转换为 Markdown"""
        lines = [
            "---",
            f"**数据来源**: {metadata.get('source_file', 'unknown')}",
            f"**数据格式**: XML",
            f"**处理时间**: {metadata.get('process_time', datetime.now().isoformat())}",
            f"**记录数量**: {len(data)}",
            "---\n"
        ]

        for i, record in enumerate(data, 1):
            lines.append(f"### 记录 {i}")
            lines.append(self._format_dict_as_markdown(record))
            lines.append("")

        return "\n".join(lines)

    def _format_dict_as_markdown(self, d: dict, indent: int = 0) -> str:
        """递归格式化字典为 Markdown"""
        lines = []
        prefix = "  " * indent

        for key, value in d.items():
            if key.startswith("_"):
                if key == "_text":
                    lines.append(f"{prefix}{value}")
                elif key == "_attributes":
                    for attr, attr_val in value.items():
                        lines.append(f"{prefix}- **{attr}**: {attr_val}")
                elif key == "_tag":
                    pass  # 标签名已在标题中显示
            elif isinstance(value, dict):
                lines.append(f"{prefix}**{key}**:")
                lines.append(self._format_dict_as_markdown(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}**{key}**:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._format_dict_as_markdown(item, indent + 1))
                    else:
                        lines.append(f"{prefix}- {item}")
            else:
                lines.append(f"{prefix}- **{key}**: {value}")

        return "\n".join(lines)


class LogParser(BaseParser):
    """日志格式解析器

    支持常见日志格式的解析：
    - 标准格式: [时间] [级别] [来源] 消息
    - JSON 日志
    - Apache/Nginx 访问日志
    - 自定义正则
    """

    # 常见日志格式的正则表达式
    COMMON_PATTERNS = [
        # 标准格式: 2024-01-01 12:00:00 ERROR [service] message
        re.compile(r'^(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\s]*)\s+(\w+)\s+(?:\[([^\]]+)\])?\s*(.*)$'),
        # 简化格式: [ERROR] message
        re.compile(r'^\[(\w+)\]\s+(.*)$'),
        # 带百分比的进度日志: 100% [INFO] message
        re.compile(r'^\d+%?\s*\[?(\w+)\]?\s*[:\-]?\s*(.*)$'),
    ]

    # 日志级别（按严重程度排序）
    LEVEL_PRIORITY = {
        "DEBUG": 0,
        "TRACE": 0,
        "INFO": 1,
        "WARN": 2,
        "WARNING": 2,
        "ERROR": 3,
        "FATAL": 4,
        "CRITICAL": 4,
    }

    def __init__(
        self,
        levels_to_keep: Optional[list[str]] = None,
        levels_to_drop: Optional[list[str]] = None,
        custom_pattern: Optional[str] = None
    ):
        """初始化日志解析器

        Args:
            levels_to_keep: 保留的日志级别（与 levels_to_drop 互斥）
            levels_to_drop: 丢弃的日志级别（与 levels_to_keep 互斥）
            custom_pattern: 自定义正则表达式（Python re 格式）
        """
        self.levels_to_keep = set(levels_to_keep or ["INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"])
        self.levels_to_drop = set(levels_to_drop or ["DEBUG", "TRACE", "VERBOSE"])
        self.custom_pattern = re.compile(custom_pattern) if custom_pattern else None

    def parse(self, content: str) -> list[dict]:
        """解析日志文本

        Args:
            content: 日志文本内容

        Returns:
            解析后的记录列表
        """
        lines = content.splitlines()
        records = []
        dropped_count = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = self._parse_line(line, line_num)
                if record:
                    # 级别过滤
                    level = record.get("level", "INFO").upper()
                    if level in self.levels_to_drop:
                        dropped_count += 1
                        continue
                    records.append(record)
            except Exception as e:
                logger.debug(f"行 {line_num} 解析失败: {e}，保留原始文本")
                # 解析失败时保留原始文本
                records.append({
                    "line_num": line_num,
                    "raw": line,
                    "level": "UNKNOWN",
                    "message": line
                })

        logger.info(f"日志解析完成: {len(lines)} 行 -> {len(records)} 条记录（丢弃 {dropped_count} 条）")
        return records

    def _parse_line(self, line: str, line_num: int) -> Optional[dict]:
        """解析单行日志"""
        # 优先尝试自定义正则
        if self.custom_pattern:
            match = self.custom_pattern.match(line)
            if match:
                groups = match.groups()
                return {
                    "line_num": line_num,
                    "level": groups[1].upper() if len(groups) > 1 else "INFO",
                    "timestamp": groups[0] if len(groups) > 0 else "",
                    "source": groups[2] if len(groups) > 2 else "",
                    "message": groups[-1] if groups else line,
                    "raw": line
                }

        # 尝试通用模式
        for pattern in self.COMMON_PATTERNS:
            match = pattern.match(line)
            if match:
                groups = match.groups()
                level = groups[1].upper() if len(groups) > 1 else "INFO"
                message = groups[-1] if groups else line

                # 提取堆栈跟踪（如果存在）
                return {
                    "line_num": line_num,
                    "level": level,
                    "timestamp": groups[0] if len(groups) > 0 else "",
                    "source": groups[2] if len(groups) > 2 else "",
                    "message": message,
                    "raw": line
                }

        # 无法解析，返回基础记录
        return {
            "line_num": line_num,
            "level": "INFO",
            "timestamp": "",
            "message": line,
            "raw": line
        }

    def to_markdown(self, data: list[dict], metadata: dict) -> str:
        """将日志数据转换为 Markdown"""
        lines = [
            "---",
            f"**数据来源**: {metadata.get('source_file', 'unknown')}",
            f"**数据格式**: 日志",
            f"**处理时间**: {metadata.get('process_time', datetime.now().isoformat())}",
            f"**记录数量**: {len(data)}",
            f"**日志级别**: {', '.join(sorted(self.levels_to_keep))}",
            "---\n"
        ]

        # 按级别分组显示（错误优先）
        error_records = [r for r in data if r.get("level") in ("ERROR", "FATAL", "CRITICAL")]
        warn_records = [r for r in data if r.get("level") in ("WARN", "WARNING")]
        other_records = [r for r in data if r not in error_records and r not in warn_records]

        if error_records:
            lines.append("## 错误日志\n")
            for record in error_records:
                lines.append(self._format_log_entry(record))
            lines.append("")

        if warn_records:
            lines.append("## 警告日志\n")
            for record in warn_records:
                lines.append(self._format_log_entry(record))
            lines.append("")

        if other_records and len(other_records) <= 20:  # 限制显示数量
            lines.append("## 其他日志\n")
            for record in other_records:
                lines.append(self._format_log_entry(record))

        return "\n".join(lines)

    def _format_log_entry(self, record: dict) -> str:
        """格式化单条日志条目"""
        level = record.get("level", "INFO")
        timestamp = record.get("timestamp", "")
        source = record.get("source", "")
        message = record.get("message", record.get("raw", ""))

        # 根据级别添加 emoji
        level_icon = {
            "ERROR": "❌",
            "FATAL": "☠️",
            "CRITICAL": "☠️",
            "WARN": "⚠️",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
            "DEBUG": "🔍",
        }.get(level, "📝")

        lines = [
            f"### {level_icon} [{level}] {timestamp}".rstrip(),
            ""
        ]

        if source:
            lines.append(f"**来源**: {source}")

        lines.append(f"```\n{message}\n```")
        lines.append("")

        return "\n".join(lines)


class TextParser(BaseParser):
    """纯文本解析器

    简单的文本分割器，按段落组织。
    """

    def parse(self, content: str) -> list[dict]:
        """解析纯文本

        Args:
            content: 文本内容

        Returns:
            按段落分割的记录列表
        """
        paragraphs = content.split("\n\n")
        records = [
            {
                "index": i,
                "content": p.strip(),
                "char_count": len(p.strip())
            }
            for i, p in enumerate(paragraphs)
            if p.strip()
        ]
        logger.info(f"文本解析完成: {len(paragraphs)} 段落 -> {len(records)} 条记录")
        return records

    def to_markdown(self, data: list[dict], metadata: dict) -> str:
        """将文本数据转换为 Markdown"""
        lines = [
            "---",
            f"**数据来源**: {metadata.get('source_file', 'unknown')}",
            f"**数据格式**: 文本",
            f"**处理时间**: {metadata.get('process_time', datetime.now().isoformat())}",
            f"**段落数量**: {len(data)}",
            "---\n"
        ]

        for record in data:
            lines.append(f"### 段落 {record['index'] + 1}")
            lines.append(f"\n{record['content']}\n")

        return "\n".join(lines)


class DataCleaner:
    """数据清洗主类

    整合各类解析器和脱敏器，提供统一的数据清洗接口。
    """

    # 文件扩展名到解析器的映射
    PARSER_MAP = {
        ".json": JSONParser,
        ".xml": XMLParser,
        ".log": LogParser,
        ".txt": TextParser,
        ".md": TextParser,
    }

    def __init__(
        self,
        levels_to_keep: Optional[list[str]] = None,
        levels_to_drop: Optional[list[str]] = None,
        sensitive_patterns: Optional[list] = None,
        enable_mask: bool = True
    ):
        """初始化数据清洗器

        Args:
            levels_to_keep: 保留的日志级别
            levels_to_drop: 丢弃的日志级别
            sensitive_patterns: 敏感信息正则模式
            enable_mask: 是否启用敏感信息脱敏
        """
        self.masker = SensitiveDataMasker(sensitive_patterns) if enable_mask else None
        self.levels_to_keep = levels_to_keep
        self.levels_to_drop = levels_to_drop

    def detect_format(self, file_path: str) -> DataType:
        """检测文件格式

        Args:
            file_path: 文件路径

        Returns:
            检测到的数据类型
        """
        suffix = Path(file_path).suffix.lower()

        # 优先根据扩展名判断（对于 .log 文件）
        if suffix == ".log":
            return DataType.LOG
        elif suffix in self.PARSER_MAP:
            return DataType(suffix.lstrip("."))

        # 从内容推断格式（对于无扩展名或 .txt 文件）
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(1024)  # 读取前 1KB

            if content.strip().startswith("{") or content.strip().startswith("["):
                return DataType.JSON
            elif content.strip().startswith("<"):
                return DataType.XML
            else:
                return DataType.TXT
        except Exception:
            return DataType.UNKNOWN

    def _get_parser(self, data_type: DataType, **kwargs):
        """获取对应的解析器"""
        parser_classes = {
            DataType.JSON: JSONParser,
            DataType.XML: XMLParser,
            DataType.LOG: LogParser,
            DataType.TXT: TextParser,
            DataType.UNKNOWN: TextParser,
        }

        parser_class = parser_classes.get(data_type, TextParser)

        # 日志解析器需要特殊参数
        if data_type == DataType.LOG:
            return parser_class(
                levels_to_keep=self.levels_to_keep,
                levels_to_drop=self.levels_to_drop,
                **kwargs
            )

        return parser_class(**kwargs)

    def process_file(
        self,
        file_path: str,
        output_format: str = "markdown",
        output_dir: Optional[str] = None
    ) -> CleaningResult:
        """处理单个文件

        Args:
            file_path: 输入文件路径
            output_format: 输出格式 ("markdown" 或 "txt")
            output_dir: 输出目录（None 则不保存文件）

        Returns:
            CleaningResult: 清洗结果对象
        """
        file_path = Path(file_path)
        start_time = datetime.now()

        # 构建元数据
        metadata = {
            "source_file": file_path.name,
            "source_path": str(file_path.absolute()),
            "source_size": file_path.stat().st_size,
            "process_time": start_time.isoformat(),
            "data_type": self.detect_format(str(file_path)).value,
        }

        try:
            # 读取文件内容
            content = file_path.read_text(encoding="utf-8")

            # 敏感信息脱敏
            if self.masker:
                content = self.masker.mask(content)

            # 解析
            data_type = self.detect_format(str(file_path))
            parser = self._get_parser(data_type)
            records = parser.parse(content)

            # 转换为目标格式
            if output_format == "markdown":
                output_content = parser.to_markdown(records, metadata)
            else:
                # 纯文本格式
                output_content = "\n\n".join(
                    r.get("message", r.get("content", str(r)))
                    for r in records
                )

            # 保存输出文件
            if output_dir:
                output_path = Path(output_dir) / f"{file_path.stem}_cleaned.md"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_content, encoding="utf-8")
                logger.info(f"清洗结果已保存: {output_path}")

            # 统计信息
            stats = {
                "total_lines": len(content.splitlines()),
                "total_records": len(records),
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000),
            }

            return CleaningResult(
                content=output_content,
                metadata={**metadata, "output_path": str(output_path) if output_dir else ""},
                success=True,
                stats=stats
            )

        except Exception as e:
            logger.error(f"文件处理失败 {file_path}: {e}")
            return CleaningResult(
                content="",
                metadata=metadata,
                success=False,
                error=str(e)
            )

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        recursive: bool = True,
        file_patterns: Optional[list[str]] = None
    ) -> list[CleaningResult]:
        """批量处理目录中的所有文件

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            recursive: 是否递归处理子目录
            file_patterns: 文件名匹配模式（glob 风格）

        Returns:
            所有文件的处理结果列表
        """
        input_path = Path(input_dir)
        results = []

        if recursive:
            files = input_path.rglob("*")
        else:
            files = input_path.glob("*")

        for file_path in files:
            if not file_path.is_file():
                continue

            # 文件类型过滤
            if file_path.suffix.lower() not in self.PARSER_MAP:
                continue

            # 模式匹配过滤
            if file_patterns:
                if not any(file_path.match(p) for p in file_patterns):
                    continue

            logger.info(f"处理文件: {file_path}")
            result = self.process_file(str(file_path), output_dir=output_dir)
            results.append(result)

        logger.info(f"目录处理完成: {len(results)} 个文件")
        return results


def create_cleaner(
    levels_to_keep: Optional[list[str]] = None,
    enable_mask: bool = True
) -> DataCleaner:
    """工厂函数：创建预配置的数据清洗器

    Args:
        levels_to_keep: 保留的日志级别
        enable_mask: 是否启用敏感信息脱敏

    Returns:
        配置好的 DataCleaner 实例
    """
    return DataCleaner(
        levels_to_keep=levels_to_keep or ["INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"],
        levels_to_drop=["DEBUG", "TRACE", "VERBOSE"],
        enable_mask=enable_mask
    )
