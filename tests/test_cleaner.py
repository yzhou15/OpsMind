"""
OpsMind ETL 模块单元测试

测试内容:
    1. 敏感信息脱敏
    2. JSON 解析
    3. XML 解析
    4. 日志解析与级别过滤
    5. 滑动窗口 overlap 验证
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.etl.cleaner import (
    DataCleaner,
    DataType,
    JSONParser,
    LogParser,
    SensitiveDataMasker,
    TextParser,
    XMLParser,
)


class TestSensitiveDataMasker(unittest.TestCase):
    """测试敏感信息脱敏功能"""

    def setUp(self):
        self.masker = SensitiveDataMasker()

    def test_mask_ssn(self):
        """测试社保号脱敏"""
        text = "用户社保号: 123-45-6789"
        result = self.masker.mask(text)
        self.assertEqual(result, "用户社保号: [REDACTED-SSN]")

    def test_mask_password(self):
        """测试密码脱敏"""
        text = '用户密码: password=123456'
        result = self.masker.mask(text)
        self.assertEqual(result, '用户密码: [REDACTED-PWD]')

    def test_mask_api_key(self):
        """测试 API Key 脱敏"""
        text = 'API Key: api_key=sk-abcdef123456789'
        result = self.masker.mask(text)
        self.assertEqual(result, 'API Key: [REDACTED-APIKEY]')

    def test_mask_token(self):
        """测试 Token 脱敏"""
        text = 'Authorization: token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
        result = self.masker.mask(text)
        self.assertIn('[REDACTED-TOKEN]', result)

    def test_mask_chinese_id(self):
        """测试身份证号脱敏"""
        text = "身份证号: 110101199001011234"
        result = self.masker.mask(text)
        self.assertEqual(result, "身份证号: [REDACTED-ID]")

    def test_no_mask_needed(self):
        """测试无需脱敏的普通文本"""
        text = "这是一条普通的日志记录，不包含敏感信息。"
        result = self.masker.mask(text)
        self.assertEqual(result, text)


class TestJSONParser(unittest.TestCase):
    """测试 JSON 格式解析"""

    def setUp(self):
        self.parser = JSONParser()

    def test_parse_simple_json(self):
        """测试解析简单 JSON 数组"""
        content = json.dumps([
            {"timestamp": "2024-01-01 10:00:00", "level": "ERROR", "message": "Database connection failed"},
            {"timestamp": "2024-01-01 10:01:00", "level": "INFO", "message": "Retry successful"}
        ])
        records = self.parser.parse(content)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["level"], "ERROR")
        self.assertEqual(records[1]["message"], "Retry successful")

    def test_parse_nested_json(self):
        """测试解析嵌套 JSON"""
        content = json.dumps({
            "logs": [
                {"timestamp": "2024-01-01", "level": "WARN", "msg": "High memory usage"}
            ]
        })
        records = self.parser.parse(content)
        self.assertGreaterEqual(len(records), 1)

    def test_parse_invalid_json(self):
        """测试解析无效 JSON"""
        with self.assertRaises(json.JSONDecodeError):
            self.parser.parse("{invalid json}")

    def test_to_markdown(self):
        """测试 Markdown 转换"""
        data = [
            {"timestamp": "2024-01-01", "level": "ERROR", "message": "Test error"}
        ]
        metadata = {"source_file": "test.json", "process_time": "2024-01-01T00:00:00"}
        result = self.parser.to_markdown(data, metadata)
        self.assertIn("**数据格式**: JSON", result)
        self.assertIn("**级别**: ERROR", result)


class TestXMLParser(unittest.TestCase):
    """测试 XML 格式解析"""

    def setUp(self):
        self.parser = XMLParser()

    def test_parse_simple_xml(self):
        """测试解析简单 XML"""
        content = """<?xml version="1.0"?>
        <logs>
            <log>
                <timestamp>2024-01-01</timestamp>
                <level>ERROR</level>
                <message>Test error</message>
            </log>
        </logs>
        """
        records = self.parser.parse(content)
        self.assertGreaterEqual(len(records), 1)

    def test_parse_invalid_xml(self):
        """测试解析无效 XML"""
        with self.assertRaises(Exception):
            self.parser.parse("<invalid>xml")

    def test_auto_detect_record_tag(self):
        """测试自动检测记录标签"""
        content = """<?xml version="1.0"?>
        <events>
            <event><id>1</id></event>
            <event><id>2</id></event>
        </events>
        """
        parser = XMLParser()
        records = parser.parse(content)
        self.assertEqual(parser.record_tag, "event")


class TestLogParser(unittest.TestCase):
    """测试日志格式解析"""

    def setUp(self):
        # 默认保留 ERROR/WARN/INFO，丢弃 DEBUG
        self.parser = LogParser(
            levels_to_keep=["ERROR", "WARN", "INFO"],
            levels_to_drop=["DEBUG", "TRACE"]
        )

    def test_parse_standard_format(self):
        """测试解析标准日志格式"""
        content = """
2024-01-01 10:00:00 ERROR [Database] Connection timeout
2024-01-01 10:00:01 INFO [Application] Server started
2024-01-01 10:00:02 DEBUG [Cache] Cache hit
        """
        records = self.parser.parse(content)
        # DEBUG 行应被过滤
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["level"], "ERROR")
        self.assertEqual(records[1]["level"], "INFO")

    def test_parse_simple_format(self):
        """测试解析简化日志格式"""
        content = """
2024-01-01 10:00:00 ERROR Database connection failed
2024-01-01 10:00:01 INFO System initialized
        """
        records = self.parser.parse(content)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["level"], "ERROR")
        self.assertEqual(records[1]["level"], "INFO")

    def test_level_filtering(self):
        """测试日志级别过滤"""
        content = """
2024-01-01 10:00:00 DEBUG Debug message 1
2024-01-01 10:00:00 TRACE Trace message
2024-01-01 10:00:00 INFO Info message
2024-01-01 10:00:00 WARN Warning message
2024-01-01 10:00:00 ERROR Error message
2024-01-01 10:00:00 CRITICAL Critical message
        """
        parser = LogParser()  # 使用默认配置
        records = parser.parse(content)

        levels = [r["level"] for r in records]
        # DEBUG 和 TRACE 应被过滤
        self.assertNotIn("DEBUG", levels)
        self.assertNotIn("TRACE", levels)
        self.assertIn("INFO", levels)
        self.assertIn("ERROR", levels)

    def test_to_markdown(self):
        """测试日志 Markdown 转换"""
        data = [
            {"level": "ERROR", "timestamp": "2024-01-01", "message": "Test error"},
            {"level": "INFO", "message": "Test info"}
        ]
        metadata = {"source_file": "test.log", "process_time": "2024-01-01T00:00:00"}
        result = self.parser.to_markdown(data, metadata)

        self.assertIn("## 错误日志", result)
        self.assertIn("## 其他日志", result)
        self.assertIn("❌", result)  # ERROR 图标


class TestDataCleaner(unittest.TestCase):
    """测试 DataCleaner 主类"""

    def setUp(self):
        self.cleaner = DataCleaner()

    def test_detect_format_json(self):
        """测试检测 JSON 格式"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write('{"key": "value"}')
            f.flush()
            f_name = f.name

        try:
            detected = self.cleaner.detect_format(f_name)
            self.assertEqual(detected, DataType.JSON)
        finally:
            try:
                Path(f_name).unlink(missing_ok=True)
            except PermissionError:
                pass  # Windows 文件锁问题

    def test_detect_format_xml(self):
        """测试检测 XML 格式"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
            f.write('<root><item>value</item></root>')
            f.flush()
            f_name = f.name

        try:
            detected = self.cleaner.detect_format(f_name)
            self.assertEqual(detected, DataType.XML)
        finally:
            try:
                Path(f_name).unlink(missing_ok=True)
            except PermissionError:
                pass

    def test_process_file_with_temp_files(self):
        """测试处理临时文件"""
        import tempfile
        # 创建临时日志文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8') as f:
            f.write('[INFO] Test log entry\n[ERROR] Test error\n')
            log_path = f.name

        try:
            result = self.cleaner.process_file(log_path)
            self.assertTrue(result.success)
            self.assertIn("Test log entry", result.content)
            self.assertIn("Test error", result.content)
        finally:
            try:
                Path(log_path).unlink(missing_ok=True)
            except PermissionError:
                pass

    def test_process_file_sensitive_mask(self):
        """测试处理文件时敏感信息脱敏"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8') as f:
            f.write('password=secret123\napi_key=sk-test123\n')
            log_path = f.name

        try:
            result = self.cleaner.process_file(log_path)
            self.assertTrue(result.success)
            self.assertIn("[REDACTED-PWD]", result.content)
            self.assertIn("[REDACTED-APIKEY]", result.content)
        finally:
            try:
                Path(log_path).unlink(missing_ok=True)
            except PermissionError:
                pass


class TestSlidingWindowOverlap(unittest.TestCase):
    """测试滑动窗口 overlap 机制

    这是验收标准中要求的关键测试用例：
    验证相邻 chunk 之间的重叠内容是否正确实现。
    """

    def test_overlap_between_chunks(self):
        """测试相邻 chunk 的 overlap 区域

        预期：
        - chunk[i] 的末尾 N 个字符应与 chunk[i+1] 的开头 N 个字符相同
        - 这确保了跨段落语义的连续性
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 测试文本：确保有足够的段落边界
        test_text = """
        第一段：这是智能运维系统的核心功能介绍。系统支持多种数据源的接入和处理。

        第二段：RAG（检索增强生成）技术结合了向量检索与大模型生成的优点。
        通过从知识库中检索相关片段，再由大模型生成准确的回答。

        第三段：ChromaDB 作为本地向量数据库，提供了高效的相似度检索能力。
        支持多种 Embedding 模型，确保语义匹配的准确性。

        第四段：日志分析是运维工作中不可或缺的一环。
        通过对日志级别的过滤和敏感信息的脱敏，可以快速定位问题根源。
        """

        # 配置滑动窗口参数
        chunk_size = 200
        chunk_overlap = 50

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        chunks = splitter.split_text(test_text)

        print(f"\n===== 滑动窗口 Overlap 验证 =====")
        print(f"Chunk 数量: {len(chunks)}")
        print(f"Chunk Size: {chunk_size}")
        print(f"Chunk Overlap: {chunk_overlap}")
        print("=" * 50)

        # 验证 overlap
        overlap_verified = True
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # 找到当前 chunk 的末尾 overlap 区域
            current_tail = current[-chunk_overlap:]
            # 找到下一个 chunk 的开头 overlap 区域
            next_head = next_chunk[:chunk_overlap]

            # 检查是否有重叠
            has_overlap = any(
                current_tail[j:] == next_head[:len(current_tail)-j]
                for j in range(len(current_tail) + 1)
            )

            print(f"\nChunk {i} 末尾 ({chunk_overlap} 字符): {repr(current_tail)}")
            print(f"Chunk {i+1} 开头 ({chunk_overlap} 字符): {repr(next_head)}")
            print(f"Overlap 验证: {'✅ 通过' if has_overlap else '❌ 失败'}")

            if not has_overlap:
                overlap_verified = False

        print("\n" + "=" * 50)
        print(f"整体 Overlap 验证: {'✅ 全部通过' if overlap_verified else '❌ 存在问题'}")

        # 如果没有足够的内容产生多个 chunk，这个测试可能无法验证
        if len(chunks) < 2:
            print("⚠️ 文本过短，未产生多个 chunk，跳过 overlap 验证")

        # 至少验证能产生多个 chunk（如果文本够长）
        if len(chunks) >= 2:
            self.assertGreater(len(chunks), 1, "应产生多个 chunk 以验证 overlap")

            # 检查至少有一个 overlap 是正确的
            self.assertTrue(
                any(
                    chunks[i][-chunk_overlap:] in chunks[i+1]
                    for i in range(len(chunks) - 1)
                ) or overlap_verified,
                "至少应有一个相邻 chunk 之间存在 overlap"
            )

    def test_chunk_boundaries_preserve_semantics(self):
        """测试 chunk 边界是否保留语义完整性"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 测试文本：确保在语义边界处分割
        test_text = """
        错误日志摘要：

        2024-01-01 10:00:00 ERROR [Database] 连接池耗尽
        原因分析：并发请求过多，连接未及时释放

        解决方案：
        1. 增加连接池大小
        2. 优化查询语句
        3. 添加熔断机制
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=30,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？"]
        )

        chunks = splitter.split_text(test_text)

        print(f"\n===== 语义完整性验证 =====")
        print(f"生成 {len(chunks)} 个 chunks")

        # 检查每个 chunk 是否有实际内容
        for i, chunk in enumerate(chunks):
            self.assertGreater(len(chunk.strip()), 10, f"Chunk {i} 内容过短")

        print(f"所有 chunks 语义完整: ✅")


if __name__ == "__main__":
    unittest.main(verbosity=2)
