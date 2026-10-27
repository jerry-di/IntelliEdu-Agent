#!/bin/bash

# ============================================================
# EduAgent 一键部署脚本
# 适用系统：CentOS / Rocky Linux / AlmaLinux / RHEL 系统
# 部署目录：/root/EduAgent/docker
# 部署服务：PostgreSQL + etcd + MinIO + Milvus + Attu
# ============================================================

set -e

BASE_DIR="/root/EduAgent/docker"
COMPOSE_VERSION="v5.1.3"

# 手动设置 IP 的情况（例如：192.168.88.100）
# SERVER_IP="192.168.88.100"

# 自动获取 eth0 的 IP 地址（内网地址）
SERVER_IP=$(ifconfig eth0 | grep broadcast | grep -v 127 | awk '{print $2}')
echo "当前服务器内网IP: $SERVER_IP"

echo "============================================================"
echo "EduAgent 基础服务一键部署脚本"
echo "============================================================"

# 1. 必须使用 root 执行
if [ "$(id -u)" -ne 0 ]; then
  echo "错误：请使用 root 用户执行该脚本。"
  exit 1
fi

# 2. 检查系统包管理工具
if ! command -v yum >/dev/null 2>&1; then
  echo "错误：当前脚本仅支持 yum 包管理工具。"
  echo "请确认是否 RHEL 兼容系统（例如 CentOS / Rocky Linux / AlmaLinux）。"
  exit 1
fi

# 3. 安装基础工具
printf "\n[1/8] 安装基础工具..."
yum install -y wget curl yum-utils

# 4. 配置 Docker-CE yum 源
printf "\n[2/8] 配置 Docker-CE yum 源..."
wget -q https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo -O /etc/yum.repos.d/docker-ce.repo

# 5. 安装 Docker
printf "\n[3/8] 安装 Docker..."
yum install -y docker-ce

# 6. 启动 Docker
printf "\n[4/8] 启动 Docker 并设置开机自启..."
systemctl start docker
systemctl enable docker

# 7. 安装 Docker Compose 独立二进制
printf "\n[5/8] 安装 Docker Compose..."
if [ ! -f /usr/local/bin/docker-compose ]; then
  wget -q "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" -O /usr/local/bin/docker-compose
  chmod +x /usr/local/bin/docker-compose
else
  echo "docker-compose 已存在，跳过下载。"
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif docker-compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "错误：Docker Compose 安装失败。"
  exit 1
fi

# 8. 创建部署目录
printf "\n[6/8] 创建部署目录..."
mkdir -p "${BASE_DIR}/scripts"
cd "${BASE_DIR}"

# 9. 将配置写入 .env 文件
printf "\n[7/8] 写入配置和脚本文件写入内容..."
cat > "${BASE_DIR}/.env" <<'ENVEOF'
# ===== 数据库（PostgreSQL）=====
DB_HOST=localhost
DB_PORT=5433
DB_NAME=eduagent
DB_USER=eduagent_user
DB_PASSWORD=123456

# ===== Milvus 向量数据库 =====
MILVUS_HOST=localhost
MILVUS_PORT=19531

# ===== DeepSeek 大模型 =====
DEEPSEEK_API_KEY=sk-[Your DeepSeek API Key]
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_CHAT=deepseek-v4-flash
DEEPSEEK_MODEL_CODER=deepseek-v4-flash

# ===== 本地模型路径 =====
RERANKER_MODEL_PATH=./models/reranker/bge-reranker-large
CLASSIFIER_MODEL_PATH=./models/classifier/all-MiniLM-L6-v2
BGE_M3_MODEL_PATH=./models/embedding/bge-m3

# ===== JWT 认证参数 =====
JWT_SECRET_KEY=5f4cbfb2bd164363ca0688ae4a3c1b0078ff9bfc8f8297d0c871cc05cf3b132f
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080

# ===== Web 搜索（可选，留空则用免费的 DuckDuckGo）=====
TAVILY_API_KEY=

# ===== 应用配置参数 =====
APP_ENV=local
APP_DEBUG=true
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
DEFAULT_TENANT_ID=tenant_default
ENVEOF

# 10. 将容器编排脚本写入 docker-compose.yml
cat > "${BASE_DIR}/docker-compose.yml" <<'YAMLEOF'
# docker-compose.yml — EduAgent 本地开发环境
services:
  # --- PostgreSQL 数据库（对外 5433 端口）---
  postgres:
    image: postgres:15-alpine
    container_name: edu_agent_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${DB_NAME:-eduagent}
      POSTGRES_USER: ${DB_USER:-eduagent_user}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-123456}
      TZ: Asia/Shanghai
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/01_init_db.sql:ro
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # --- etcd（Milvus 元数据存储，仅 Docker 内部使用）---
  etcd:
    image: quay.io/coreos/etcd:v3.5.14
    container_name: edu_agent_etcd
    restart: unless-stopped
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
      TZ: Asia/Shanghai
    command: >
      etcd
      --advertise-client-urls=http://etcd:2379
      --listen-client-urls=http://0.0.0.0:2379
      --data-dir=/etcd
    volumes:
      - etcd_data:/etcd

  # --- MinIO（Milvus 向量文件存储，仅 Docker 内部使用口）---
  minio:
    image: minio/minio:latest
    container_name: edu_agent_minio
    restart: unless-stopped
    command: server /data
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY:-minioadmin}
      TZ: Asia/Shanghai
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 15s
      timeout: 5s
      retries: 5

  # --- Milvus 向量数据库（对外 19531 端口）---
  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: edu_agent_milvus
    restart: unless-stopped
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY:-minioadmin}
      MINIO_SECRET_ACCESS_KEY: ${MINIO_SECRET_KEY:-minioadmin}
      TZ: Asia/Shanghai
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19531:19530"
    depends_on:
      etcd:
        condition: service_started
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 15s
      timeout: 10s
      retries: 10

  # --- Attu（Milvus 可视化管理界面，对外 30000 端口）---
  attu:
    image: zilliz/attu:v2.4.12
    container_name: edu_agent_attu
    restart: unless-stopped
    environment:
      MILVUS_URL: milvus:19530
      SERVER_PORT: "3000"
      TZ: Asia/Shanghai
    ports:
      - "30000:3000"
    depends_on:
      milvus:
        condition: service_healthy

# --- 数据卷（重启容器数据不丢失）---
volumes:
  postgres_data:
  etcd_data:
  minio_data:
  milvus_data:
YAMLEOF

# 11. 创建 PostgreSQL 初始化脚本。
cat > "${BASE_DIR}/scripts/init_db.sql" <<'SQLEOF'
-- EduAgent PostgreSQL 初始化脚本
-- 当前仅保证初始化脚本文件存在，业务表可后续按项目模型补充。
SELECT 1;
SQLEOF

# 12. 启动服务
printf "\n[8/8] 启动 EduAgent 基础服务..."
${COMPOSE_CMD} down >/dev/null 2>&1 || true
${COMPOSE_CMD} up -d

# 13. 输出结果
printf "\n"
echo "============================================================"
echo "EduAgent 基础服务部署完成"
echo "============================================================"
echo "部署目录：${BASE_DIR}"
echo ""
echo "服务访问地址："
echo "  PostgreSQL : ${SERVER_IP}:5433"
echo "  Milvus     : ${SERVER_IP}:19531"
echo "  Attu       : http://${SERVER_IP}:30000"
echo ""
echo "常用命令："
echo "  cd ${BASE_DIR}"
echo "  ${COMPOSE_CMD} ps"
echo "  ${COMPOSE_CMD} logs -f"
echo "  ${COMPOSE_CMD} down"
echo "  ${COMPOSE_CMD} down -v"
echo ""
echo "注意："
echo "  1. .env 中 DEEPSEEK_API_KEY 需要改成真实 Key。"
echo "  2. 当前 compose 只部署 PostgreSQL、Milvus、etcd、MinIO、Attu。"
echo "============================================================"

${COMPOSE_CMD} ps