# OpsMind 智能运维助手

基于 RAG（检索增强生成）架构的智能运维问答系统，支持本地私有化部署。

## 技术栈

- **编程语言**: Python 3.10+
- **RAG 框架**: LangChain
- **向量数据库**: ChromaDB（本地持久化）
- **大语言模型**: Llama-3 / Qwen（通过 Ollama 本地调用）
- **前端交互**: Streamlit
- **容器化**: Docker + Docker Compose

## 核心功能

1. **数据清洗与预处理**：支持 JSON/XML/日志/PDF/TXT 等格式的解析与格式化
2. **文档切片与向量化**：基于滑动窗口机制（chunk_size=500, chunk_overlap=100）
3. **语义检索**：从知识库中快速检索相关文档片段
4. **智能问答**：结合检索结果生成准确回答，避免幻觉

## 快速开始

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

# 或下载 Llama 3 模型
ollama pull llama3:7b
```

### 3. 初始化知识库

```bash
# 加载 data 目录下的文档到向量库
python init_knowledge_base.py
```

### 4. 启动应用

```bash
# 启动 Streamlit 前端
streamlit run app.py
```

### 5. 访问应用

打开浏览器访问：http://localhost:8501

## 项目结构

```
OpsMind/
├── data/                  # 文档数据
│   ├── linux_ops.txt      # Linux 运维手册
│   ├── network_ops.txt    # 网络运维手册
│   └── test_doc.txt       # 测试文档
├── src/                   # 源代码
│   ├── etl/               # 数据清洗模块
│   │   └── cleaner.py
│   ├── rag/               # RAG 核心模块
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── prompt_templates.py
│   ├── loader.py          # 文档加载器
│   ├── vector_store.py    # 向量存储工具
│   └── rag_chain.py       # RAG 问答链
├── configs/               # 配置文件
│   └── config.yaml
├── docker/                # Docker 配置
│   ├── Dockerfile
│   └── docker-compose.yml
├── app.py                 # Streamlit 前端入口
├── init_knowledge_base.py # 知识库初始化脚本
└── requirements.txt       # 依赖清单
```

## 配置说明

主要配置项（configs/config.yaml）：

| 配置项                       | 说明           | 默认值      |
| ---------------------------- | -------------- | ----------- |
| chunk.chunk_size             | 文档切片大小   | 500         |
| chunk.chunk_overlap          | 片段重叠大小   | 100         |
| vector_store.embedding_model | Embedding 模型 | BAAI/bge-m3 |
| llm.model_name               | LLM 模型名称   | qwen2.5:7b  |
| llm.temperature              | 温度参数       | 0.3         |

## 测试验证

### 测试问答功能

```bash
python test_rag.py
```

### 测试滑动窗口机制

```bash
python test_sliding_window.py
```

## Docker 部署

```bash
# 构建并启动服务
cd docker
docker-compose up -d

# 访问应用
# http://localhost:8501
```

## 许可证

MIT License
