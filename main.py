import os, sys

# Windows conda：把 base 的 Library/bin 加入 DLL 搜索路径
if sys.platform == "win32":
    _conda_base_lib_bin = os.path.normpath(
        os.path.join(os.path.dirname(sys.executable), "..", "..", "Library", "bin"))
    if os.path.isdir(_conda_base_lib_bin):
        os.add_dll_directory(_conda_base_lib_bin)

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.core.logger import setup_logger, get_logger
from backend.api.router import api_router

settings = get_settings()

# MCP 子应用（提前创建，让 lifespan 和 mount 共享同一实例）
from backend.mcp.knowledge_base_server import mcp as kb_mcp
from backend.mcp.web_search_server import mcp as ws_mcp
_kb_app = kb_mcp.streamable_http_app()
_ws_app = ws_mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger = get_logger(__name__)
    logger.info("app.starting | env=%s port=%s", settings.app_env, settings.app_port)

    # ① DB Schema 自动迁移
    try:
        from backend.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning("app.migrations_failed | error=%s", e)

    # ③ 驱动 MCP Server 的 lifespan
    async with kb_mcp._session_manager.run():
        async with ws_mcp._session_manager.run():
            logger.info("app.started")
            yield

            logger.info("app.shutting_down")
            from backend.core.llm_factory import LLMFactory
            LLMFactory.clear_cache()
            logger.info("app.shutdown_complete")


app = FastAPI(
    title="EduAgent API",
    description="AI 助教智能体系统 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080",
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

app.mount("/mcp/kb",         _kb_app)
app.mount("/mcp/web-search", _ws_app)


@app.get("/health", tags=["系统"])
async def health():
    return {"status": "ok", "env": settings.app_env}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
