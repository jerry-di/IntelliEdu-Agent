"""
router - API 路由总入口（六个子模块聚合）

"""
from fastapi import APIRouter
from backend.api.v1 import auth, qa, exam, resume, interview, unified_chat

api_router = APIRouter()

api_router.include_router(auth.router,          prefix="/auth",      tags=["认证"])
api_router.include_router(unified_chat.router,  prefix="/chat",      tags=["AI助手"])
api_router.include_router(qa.router,            prefix="/qa",        tags=["智能问答"])
api_router.include_router(exam.router,          prefix="/exam",      tags=["试卷批改"])
api_router.include_router(resume.router,        prefix="/resume",    tags=["简历审查"])
api_router.include_router(interview.router,     prefix="/interview", tags=["模拟面试"])
