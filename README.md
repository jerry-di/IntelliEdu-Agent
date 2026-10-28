

#  IntelliEdu — AI 原生教学辅助系统

## 项目简介

IntelliEdu 是一套 AI 原生的教学辅助系统，将学员答疑、试卷批改、简历审查、模拟面试四类教学业务分别构建为四个专门的智能体（Agent）。每个 Agent 都融合了企业私有知识、内置了完整业务流程，并配备了工程化的容错机制，助力教学场景实现智能化升级。

## 系统架构

EduAgent 采用分层架构设计，各层职责清晰、协同高效：

```
┌─────────────────────────────────────────────────────────┐
│                      前端 / 客户端                        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    API 层（FastAPI）                     │
│  身份校验  ·  请求路由  ·  流式/非流式响应                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│              编排层 Orchestrator                      │
│  意图识别  ·  请求路由  ·  多 Agent 流水线编排            │
└─────┬──────────┬──────────┬──────────┬───────────────┘
      │          │          │          │
┌─────▼───┐ ┌────▼───┐ ┌────▼───┐ ┌────▼────┐
│ QA Agent│ │  Exam  │ │ Resume │ │Interview│
│ 智能问答 │ │ 试卷批改 │ │ 简历审查│ │ 模拟面试 │
└────┬────┘ └───┬────┘ └───┬────┘ └────┬────┘
     │          │          │           │
┌────▼──────────▼──────────▼───────────▼──────────────────┐
│                      公共层                              │
│  LLM 工厂  ·  本地模型  ·  记忆管理  ·  容错/重试            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│                      MCP 层                            │
│  知识库检索工具  ·  联网搜索工具                           │
└─────────────┬─────────────────┬────────────────────────┘
              │                 │
     ┌────────▼──────┐  ┌───────▼────────┐
     │    Milvus     │  │    PostgreSQL  │
     └───────────────┘  └────────────────┘
```

## 核心特性

### 四大专业 Agent

| Agent | 业务场景 | 核心能力 |
|-------|---------|---------|
| **智能问答（QA）** | 学员答疑辅导 | RAG 检索增强生成，融合私有知识库，支持 BGE-M3 向量检索与 BGE-Reranker 精排，实现 SSE 流式输出 |
| **试卷批改（Exam）** | 自动批改试卷 | 三轨并行批改策略，HitL（Human-in-the-Loop）人在环机制，确保老师可最终确认批改结果 |
| **简历审查（Resume）** | 简历评估建议 | 结构化信息抽取，多维度并行评分，生成针对性优化建议 |
| **模拟面试（Interview）** | 技术面试陪练 | 状态机驱动的多阶段对话设计，双轨评估（知识掌握+表达能力），全程流式输出 |

### 技术亮点

- **意图路由智能调度**：基于本地轻量模型实现精准意图分类，自动将用户请求路由至对应的专业 Agent
- **本地化部署能力**：关键模型（BGE 系列、MiniLM）本地运行，保障数据隐私与响应速度
- **容错与降级机制**：三级降级重试策略，确保复杂场景下的服务稳定性
- **标准化 MCP 协议**：知识库检索与联网搜索采用 MCP 协议封装，便于扩展与集成

## 技术栈

| 分类 | 技术选型 |
|------|---------|
| **Web 框架** | FastAPI + Uvicorn + SSE（Server-Sent Events） |
| **Agent 框架** | LangGraph（状态图）+ LangChain |
| **大模型** | DeepSeek V4（Flash / Pro），通过 OpenAI 兼容 API 调用 |
| **本地模型** | BGE-M3（向量嵌入）、BGE-Reranker（精排）、all-MiniLM-L6-v2（意图分类） |
| **向量数据库** | Milvus 2.4（Standalone 模式） |
| **关系数据库** | PostgreSQL 15（通过 SQLAlchemy + asyncpg） |
| **MCP 协议** | MCP SDK（知识库检索 + DuckDuckGo 联网搜索） |
| **身份认证** | JWT（python-jose）+ bcrypt 密码哈希 |
| **文档解析** | PyMuPDF（PDF）+ python-docx（Word） |
| **基础设施** | Docker Compose（etcd + MinIO + Milvus + Attu） |
| **测试** | pytest + pytest-asyncio |

## 项目结构

```
EduAgent/
├── backend/                      # 后端服务核心代码
│   ├── main.py                   # 应用入口，启动 FastAPI 服务
│   ├── config.py                 # 配置管理（基于 pydantic-settings）
│   ├── dependencies.py           # FastAPI 依赖注入定义
│   ├── api/                      # API 路由层
│   │   ├── router.py             # 路由聚合入口
│   │   └── v1/                   # V1 版本接口
│   │       ├── auth.py           # 用户登录 / JWT 认证接口
│   │       ├── qa.py             # 智能问答接口
│   │       ├── exam.py           # 试卷批改接口
│   │       ├── interview.py      # 模拟面试接口
│   │       ├── resume.py         # 简历审查接口
│   │       └── unified_chat.py   # 统一对话入口（含意图路由）
│   ├── agents/                   # Agent 核心实现
│   │   ├── qa/                   # 问答 Agent
│   │   │   ├── graph.py          #   LangGraph 状态图定义
│   │   │   ├── nodes.py          #   图节点函数实现
│   │   │   ├── prompts.py        #   提示词模板
│   │   │   └── state.py          #   类型化状态 Schema
│   │   ├── exam/                 # 批改 Agent（结构同上）
│   │   ├── resume/               # 简历 Agent（结构同上）
│   │   └── interview/            # 面试 Agent（结构同上）
│   ├── core/                     # 核心公共组件
│   │   ├── orchestrator.py       # Agent 编排调度器
│   │   ├── query_classifier.py   # 意图分类（基于 MiniLM）
│   │   ├── llm_factory.py        # LLM 工厂（DeepSeek 封装）
│   │   ├── knowlege_base.py      # 向量知识库（BGE-M3 + Milvus）
│   │   ├── reranker.py           # 重排序（BGE-Reranker）
│   │   ├── memory.py             # LangGraph MemorySaver
│   │   ├── retry.py              # 三级降级重试机制
│   │   ├── exceptions.py         # 自定义异常类
│   │   └── logger.py             # 日志配置
│   ├── mcp/                      # MCP 协议实现
│   │   ├── client.py             # MCP 协议客户端
│   │   ├── knowledge_base_server.py  # 知识库检索 MCP 工具
│   │   └── web_search_server.py  # 联网搜索 MCP 工具
│   └── db/                       # 数据库相关
│       └── migrations.py         # 数据库迁移脚本
├── models/                       # 本地模型文件（需单独下载）
│   ├── classifier/               # 意图分类模型
│   ├── embedding/                # 嵌入模型
│   └── reranker/                 # 重排序模型
├── scripts/                      # 实用脚本
│   ├── init_db.sql               # PostgreSQL 初始化脚本
│   ├── init_milvus.py            # Milvus 集合初始化
│   ├── build_knowledge_base.py   # 知识库文档导入
│   ├── seed_data.py              # 测试数据填充
│   └── verify_env.py             # 环境自检脚本
├── docker-compose.yml            # Docker 基础设施编排
├── requirements.txt              # Python 依赖清单
└── .env.local                    # 本地环境变量模板
```

## 快速开始

### 环境要求

- **Python**: 3.11 或更高版本
- **Docker**: 20.10+ 及 Docker Compose
- **conda**: 推荐用于环境管理（也可使用 venv）

### 步骤 1：克隆项目

```bash
git clone https://github.com/jerry-di/IntelliEdu-Agent.git
cd EduAgent
```

### 步骤 2：创建虚拟环境

```bash
# 使用 conda 创建
conda create -n edu_agent python=3.11
conda activate edu_agent

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 步骤 3：启动基础设施

```bash
docker compose up -d
```

启动成功后，以下服务将运行在后台：

| 服务 | 端口 | 说明 |
|------|------|------|
| PostgreSQL | 5433 | 关系数据库 |
| Milvus | 19531 | 向量数据库 |
| etcd | 2379 | Milvus 元数据存储 |
| MinIO | 9000/30001 | 对象存储 |
| Attu | 30000 | Milvus 可视化管理界面 |

### 步骤 4：下载本地模型

```bash
# 下载 BGE-M3 嵌入模型
python scripts/download_models.py --model bge-m3

# 下载 BGE-Reranker 重排序模型
python scripts/download_models.py --model bge-reranker

# 下载意图分类模型
python scripts/download_models.py --model minilm
```

### 步骤 5：初始化数据库和知识库

```bash
# 创建 Milvus 集合
python scripts/init_milvus.py

# 导入知识库文档（需提前准备文档资料）
python scripts/build_knowledge_base.py

# （可选）填充测试数据
python scripts/seed_data.py
```

### 步骤 6：配置环境变量

复制环境变量模板并编辑：

```bash
cp .env.local .env
```

在 `.env` 文件中填入必要的配置：

```bash
# DeepSeek API 配置
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 数据库连接
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/edu_agent

# Milvus 连接
MILVUS_URI=http://localhost:19530
```

### 步骤 7：启动服务

```bash
python backend/main.py
```

服务启动后，API 文档可访问：`http://localhost:8000/docs`

### 步骤 8：环境自检（可选）

```bash
python scripts/verify_env.py
```

该脚本将检查所有依赖服务状态、模型文件完整性以及环境变量配置。

## API 接口概览

| 接口路径 | 方法 | 功能说明 |
|---------|------|---------|
| `/api/v1/auth/login` | POST | 用户登录，获取 JWT Token |
| `/api/v1/chat` | POST | 统一对话入口，由 Orchestrator 自动路由到对应 Agent |
| `/api/v1/qa/ask` | POST | 知识库问答，支持 SSE 流式响应 |
| `/api/v1/exam/grade` | POST | 提交试卷图片/文档进行智能批改 |
| `/api/v1/resume/review` | POST | 提交简历文档进行审查与评估 |
| `/api/v1/interview/start` | POST | 开始一轮模拟面试会话 |
| `/api/v1/interview/respond` | POST | 面试过程中提交回答，继续对话 |

## License

本项目仅用于学习和研究目的。

---

**IntelliEdu** — 让 AI 助力教学场景智能化升级
