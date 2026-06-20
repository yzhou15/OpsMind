# OpsMind RAG 智能运维助手 - Dockerfile
# 基于 Python 3.10+ 的多阶段构建优化镜像大小

# ============ 第一阶段：构建阶段 ============
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖到独立目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============ 第二阶段：运行阶段 ============
FROM python:3.10-slim

# 标签
LABEL maintainer="OpsMind Team"
LABEL description="OpsMind - RAG-based Intelligent Operations Assistant"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/models/.cache/huggingface

# 创建非 root 用户（安全最佳实践）
RUN groupadd -r opsmind && useradd -r -g opsmind opsmind

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /install /usr/local

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /data/raw /data/processed /chroma_db /models/.cache

# 复制模型缓存（如果存在）
# VOLUME ["/models/.cache/huggingface"]

# 数据卷挂载点
VOLUME ["/data", "/chroma_db"]

# 暴露端口
# Streamlit 默认端口
EXPOSE 8501

# 切换到非 root 用户
USER opsmind

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动命令
# 默认启动 Streamlit Web 界面
CMD ["streamlit", "run", "app.py", "--server.address=127.0.0.1", "--server.port=8501"]
