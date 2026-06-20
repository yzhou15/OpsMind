# OpsMind 内网离线部署指南

## 问题说明

内网环境无法访问 Docker Hub，因此需要手动部署 Ollama 服务。

---

## 部署步骤

### 步骤 1：安装 Ollama（离线方式）

**Windows（推荐）:**
```powershell
# 1. 在有外网的机器上下载 Ollama 安装包
# 下载地址：https://ollama.com/download/windows

# 2. 将安装包拷贝到内网机器，双击安装

# 3. 验证安装
ollama --version
```

**Linux:**
```bash
# 1. 在有外网的机器上下载安装脚本
curl -fsSL https://ollama.com/install.sh -o install.sh

# 2. 拷贝到内网机器并安装
bash install.sh

# 3. 验证安装
ollama --version
```

### 步骤 2：下载模型（离线方式）

**方式一：在外网机器下载后拷贝**
```bash
# 在有外网的机器上
ollama pull qwen2.5:7b

# 模型默认存储路径：
#   Linux: ~/.ollama/models
#   Windows: C:\Users\<用户名>\.ollama\models

# 将整个 models 目录打包拷贝到内网机器相同路径
```

**方式二：使用模型文件（推荐）**
```bash
# 从 HuggingFace 下载 GGUF 格式模型：
# https://huggingface.co/models?search=qwen2.5%20gguf

# 创建模型文件
ollama create qwen2.5:7b -f Modelfile
```

创建 `Modelfile`:
```
FROM ./qwen2_5-7b-instruct-q4_0.gguf
PARAMETER temperature 0.3
PARAMETER context-window 8192
SYSTEM 你是一位资深运维专家，请根据提供的上下文信息回答问题。
```

### 步骤 3：启动 Ollama 服务

```bash
# Linux/macOS
ollama serve

# Windows（后台运行）
ollama serve
```

验证服务：
```bash
curl http://localhost:11434/api/tags
```

### 步骤 4：启动 OpsMind 应用

```bash
# 进入 docker 目录
cd docker

# 构建并启动应用
docker-compose up -d --build
```

### 步骤 5：验证服务

```bash
# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f opsmind-app

# 访问 Web 界面
# http://localhost:8501
```

---

## 完整配置示例

如果 Ollama 不在 Docker 网络中，需要修改环境变量：

```bash
# 修改 docker-compose.yml 中的 OLLAMA_BASE_URL
# 将 http://ollama:11434 改为 http://host.docker.internal:11434
```

或者直接使用宿主机 IP：
```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

---

## 故障排查

### 问题 1：应用无法连接 Ollama
```bash
# 检查网络连通性
docker exec opsmind curl http://ollama:11434/api/tags

# 如果不在同一网络，使用 host.docker.internal
docker exec opsmind curl http://host.docker.internal:11434/api/tags
```

### 问题 2：模型下载慢或失败
```bash
# 使用国内镜像加速（如果内网允许访问）
export OLLAMA_HOST=https://api.ollama.cn
ollama pull qwen2.5:7b
```

### 问题 3：内存不足
```bash
# 降低模型规格
ollama pull qwen2.5:4b  # 4B 参数版本
```

---

## 快速启动脚本

创建 `start_opsmind.ps1`:
```powershell
# 启动 Ollama
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow

# 等待 Ollama 启动
Start-Sleep -Seconds 10

# 启动应用
cd docker
docker-compose up -d --build
```

---

## 卸载

```bash
# 停止容器
docker-compose down

# 删除数据卷（谨慎操作）
docker-compose down -v
```
