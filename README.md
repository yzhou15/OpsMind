# OpsMind 智能运维助手

基于 RAG（检索增强生成）架构的智能运维问答系统，支持本地私有化部署。

## 🎯 项目亮点

- ✅ **本地化部署**：数据完全自主可控，无需依赖云服务
- ✅ **滑动窗口机制**：解决长文本上下文限制，保留跨段落语义连贯性
- ✅ **多格式支持**：ETL管道统一处理JSON/XML/日志/Markdown等格式
- ✅ **增量更新**：新文档入库不覆盖旧索引
- ✅ **维度自动适配**：模型切换时自动重建向量库

## 🏗️ 技术架构

```
用户提问
   ↓
┌─────────────────┐
│  Streamlit前端   │  ← 用户交互界面
└────────┬────────┘
         ↓
┌─────────────────┐
│   RAG 检索链     │  ← 核心业务逻辑（检索+生成）
└────────┬────────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌─────────┐ ┌─────────┐
│ChromaDB │ │ Ollama  │ ← qwen2.5:7b
│向量库   │ │本地大模型│
└─────────┘ └─────────┘
```

## 🛠️ 技术栈

| 模块 | 技术选型 | 版本 |
|------|---------|------|
| 编程语言 | Python | 3.10+ |
| RAG框架 | LangChain | 最新版 |
| 向量数据库 | ChromaDB | 本地持久化 |
| Embedding模型 | BAAI/bge-m3 | 1024维 |
| 大语言模型 | Qwen 2.5 | 7B |
| 前端交互 | Streamlit | 最新版 |
| 容器化 | Docker + Docker Compose | - |

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 Ollama（用于本地运行大语言模型）
# 参考：https://ollama.com/download
```

### 2. 下载模型

```bash
# 下载 Qwen 2.5 模型（推荐）
ollama pull qwen2.5:7b
```

### 3. 初始化知识库

```bash
# 加载 data 目录下的文档到向量库
python init_knowledge_base.py
```

### 4. 启动应用

```bash
# 启动 Streamlit 前端
streamlit run src/app.py
```

### 5. 访问应用

打开浏览器访问：**http://localhost:8501**

## 📊 性能测试报告

### 测试环境

- **CPU**: Intel i7-10750H
- **内存**: 16GB DDR4
- **模型**: Qwen 2.5 7B (CPU模式)
- **向量库**: ChromaDB (2516个文档片段)

### 测试结果

| 指标 | 数值 |
|------|------|
| 检索速度（平均） | 0.12s/query |
| 回答延迟（平均） | 8.5s/response |
| 检索命中率 | 95% |
| 回答准确率 | 88% |
| P95检索耗时 | 0.25s |
| P95回答耗时 | 12.3s |

### 测试用例

```python
# 测试查询示例
test_queries = [
    "MySQL主从延迟如何排查",
    "Redis缓存雪崩怎么处理",
    "Kubernetes Pod OOMKilled解决方法",
    "Nginx返回502错误怎么办",
    "CPU使用率过高如何定位"
]
```

## 🧪 Embedding模型对比实验

### 实验结果

| 模型 | 向量维度 | 模型加载时间 | 检索速度 | 状态 |
|------|---------|------------|---------|------|
| bge-m3 | 1024 | 15.2s | 0.12s | ✅ |
| text2vec-base-chinese | 768 | 8.5s | 0.08s | ✅ |
| nomic-embed-text-v1.5 | 768 | 12.3s | 0.15s | ✅ |
| all-mpnet-base-v2 | 768 | 10.8s | 0.11s | ✅ |

### 结论

- **bge-m3**：多语言支持好，向量维度高，检索效果最佳（推荐）
- **text2vec-base-chinese**：中文效果好，模型小，速度快
- **nomic-embed-text**：多语言支持，开源友好

## 💬 使用场景示例

### 场景1：MySQL故障排查

**用户提问：**
> "MySQL主从延迟如何排查？"

**系统回答：**
> 根据知识库内容，MySQL主从延迟排查步骤如下：
>
> 1. **查看复制状态**：
>    ```sql
>    SHOW SLAVE STATUS\G
>    ```
>    关注 `Seconds_Behind_Master` 字段
>
> 2. **检查网络延迟**：
>    - 在从库执行 `ping` 测试主从网络连通性
>    - 检查主从之间的带宽是否足够
>
> 3. **分析慢查询**：
>    - 开启慢查询日志
>    - 使用 `pt-query-digest` 分析慢查询
>
> 4. **检查主库写入压力**：
>    - 查看主库 `Threads_connected` 和 `Threads_running`
>    - 分析主库的写入吞吐量
>
> **引用来源**：ops_troubleshooting.md

### 场景2：Redis缓存雪崩

**用户提问：**
> "Redis缓存雪崩怎么处理？"

**系统回答：**
> 根据知识库内容，Redis缓存雪崩的处理方案：
>
> **预防措施：**
>
> 1. **设置过期时间随机化**：
>    - 避免大量key同时过期
>    - 设置过期时间时添加随机偏移量
>
> 2. **多级缓存架构**：
>    - 本地缓存（如Caffeine）+ Redis缓存
>    - 降级策略：缓存失效时先查询本地缓存
>
> 3. **热点数据永不过期**：
>    - 对于热点数据不设置过期时间
>    - 通过后台定时任务更新缓存
>
> **应急处理：**
>
> 1. **熔断限流**：
>    - 使用Sentinel或Hystrix进行熔断
>    - 限制对后端数据库的请求量
>
> 2. **缓存预热**：
>    - 在低峰期提前加载热点数据
>    - 使用脚本批量预热缓存
>
> **引用来源**：ops_troubleshooting.md

### 场景3：Kubernetes Pod故障

**用户提问：**
> "Kubernetes Pod OOMKilled怎么解决？"

**系统回答：**
> 根据知识库内容，Kubernetes Pod OOMKilled问题的解决方案：
>
> **排查步骤：**
>
> 1. **查看Pod事件**：
>    ```bash
>    kubectl describe pod <pod-name>
>    ```
>    查看Events中的OOMKilled信息
>
> 2. **查看资源使用情况**：
>    ```bash
>    kubectl top pod <pod-name>
>    ```
>    检查内存使用是否接近limits
>
> **解决方案：**
>
> 1. **调整资源限制**：
>    ```yaml
>    resources:
>      requests:
>        memory: "512Mi"
>      limits:
>        memory: "1Gi"
>    ```
>    - 增大memory limits
>    - 合理设置requests和limits比例
>
> 2. **优化应用代码**：
>    - 检查是否有内存泄漏
>    - 优化大对象处理逻辑
>
> 3. **配置质量服务等级**：
>    - 设置 `QoS Class` 为 Guaranteed
>    - 确保Pod获得足够的资源分配
>
> **引用来源**：ops_troubleshooting.md

## 📁 项目结构

```
OpsMind/
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   │   ├── application_error.log   # 应用错误日志
│   │   ├── linux_ops.txt           # Linux运维手册
│   │   ├── network_ops.txt         # 网络运维手册
│   │   ├── monitoring_rules.json   # 监控告警规则
│   │   ├── network_incidents.log   # 网络事件日志
│   │   ├── ops_troubleshooting.md  # 运维故障排查手册
│   │   └── ...
│   └── processed/           # 清洗后数据
├── src/                     # 源代码
│   ├── etl/                 # 数据清洗模块
│   │   └── cleaner.py       # 多格式解析、敏感信息脱敏
│   ├── rag/                 # RAG核心模块
│   │   ├── vector_store.py  # ChromaDB向量存储
│   │   ├── retriever.py     # 滑动窗口文档切片
│   │   └── prompt_templates.py  # Prompt模板
│   ├── loader.py            # 文档加载器
│   ├── vector_store.py      # 向量存储工具
│   ├── rag_chain.py         # RAG问答链
│   └── app.py               # Streamlit前端入口
├── configs/                 # 配置文件
│   ├── config.yaml          # 主配置文件
│   └── settings.py          # 配置类定义
├── tests/                   # 测试模块
│   ├── test_cleaner.py      # ETL清洗测试
│   ├── test_rag_pipeline.py # RAG管道测试
│   ├── performance_test.py  # 性能测试（检索速度、延迟、准确率）
│   ├── embedding_comparison.py  # Embedding模型对比
│   └── build_vector_db.py   # 向量库构建脚本
├── docker/                  # Docker配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── OFFLINE_DEPLOYMENT.md
├── init_knowledge_base.py   # 知识库初始化脚本
├── main.py                  # 命令行入口
└── requirements.txt         # 依赖清单
```

## ⚙️ 配置说明

主要配置项（configs/config.yaml）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| chunk.chunk_size | 文档切片大小 | 1000 |
| chunk.chunk_overlap | 片段重叠大小 | 200 |
| vector_store.embedding_model | Embedding模型 | BAAI/bge-m3 |
| llm.model_name | LLM模型名称 | qwen2.5:7b |
| llm.temperature | 温度参数 | 0.3 |

## 🧪 测试验证

### 运行性能测试

```bash
python tests/performance_test.py
```

测试内容：
- 检索速度测试
- 回答延迟测试
- 命中率测试
- 准确率测试

### 运行Embedding模型对比

```bash
python tests/embedding_comparison.py
```

测试内容：
- 不同Embedding模型的加载时间对比
- 检索速度对比
- 向量维度对比

### 运行单元测试

```bash
python -m pytest tests/ -v
```

## 🐳 Docker部署

```bash
# 构建并启动服务
cd docker
docker-compose up -d

# 访问应用
# http://localhost:8501
```

## 🔒 安全特性

- ✅ 所有服务默认监听 localhost/内网IP，禁止暴露公网端口
- ✅ 环境变量管理敏感配置（API Keys, DB Path）
- ✅ 数据目录使用Volume挂载，确保容器重启后知识库存留
- ✅ 敏感信息自动脱敏（IP地址、密码等）

## 📈 技术亮点

### 1. 滑动窗口文档切片

```python
chunk_size = 1000      # 每段1000字符
chunk_overlap = 200    # 重叠200字符
```

**设计意图**：解决大模型上下文窗口限制问题，通过重叠保留跨段落语义连贯性。

### 2. 维度自动适配

```python
# 检测到维度不匹配时自动重建向量库
if "dimension" in str(e).lower():
    logger.warning("向量库维度不匹配，重建向量库")
    self._recreate_db(Chroma, embeddings)
```

**设计意图**：支持在不同Embedding模型之间无缝切换。

### 3. 增量更新机制

```python
# 新文档入库不覆盖旧索引
ids = db.add_texts(texts=texts, metadatas=metadatas)
```

**设计意图**：支持知识库的持续迭代更新。

## 📝 License

MIT License