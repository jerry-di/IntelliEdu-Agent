# backend/api/v1/interview.py

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import text

from backend.agents.interview.graph import build_interview_graph
from backend.agents.interview.state import InterviewStage
from backend.core.logger import get_logger
from backend.core.memory import build_thread_id, build_config
from backend.dependencies import AsyncSessionLocal, get_current_user

router = APIRouter()
logger = get_logger(__name__)
_graph = build_interview_graph()


class StartSessionRequest(BaseModel):
    target_position:  str
    resume_review_id: str | None = None


class ChatRequest(BaseModel):
    message: str


def _get_last_ai_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            return str(content)
    return "面试已开始，请等待面试官回应..."


# ════════════════════════════════════════════════════════════════
# POST /sessions — 创建面试会话（首轮）
# ════════════════════════════════════════════════════════════════

@router.post("/sessions", status_code=201)
async def start_session(req: StartSessionRequest, current_user: dict = Depends(get_current_user)):
    student_id = current_user["user_id"]
    tenant_id  = current_user["tenant_id"]
    session_id = str(uuid.uuid4())
    thread_id  = build_thread_id(student_id, session_id)

    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO interview_sessions
                    (id, tenant_id, student_id, session_id, thread_id,
                     target_position, resume_review_id, status)
                VALUES
                    (:id, :tenant_id, :student_id, :session_id, :thread_id,
                     :target_position, :resume_review_id, 'in_progress')
            """),
            {"id": str(uuid.uuid4()), "tenant_id": tenant_id, "student_id": student_id,
             "session_id": session_id, "thread_id": thread_id,
             "target_position": req.target_position, "resume_review_id": req.resume_review_id},
        )
        await session.commit()

    initial_state = {
        "messages": [HumanMessage(content="[开始面试]")],
        "student_id": student_id, "tenant_id": tenant_id,
        "session_id": session_id, "target_position": req.target_position,
        "resume_review_id": req.resume_review_id,
        "resume_projects": [], "resume_skills": [],
        "current_stage": InterviewStage.WARMUP.value,
        "stage_turn_count": 0, "total_turn_count": 0, "max_turns": 40,
        "question_bank": [], "current_question": None, "projects_asked": [],
        "last_answer_quality": "adequate", "followup_count": 0,
        "existing_summary": None, "should_summarize": False,
        "report": None, "fallback_used": False, "structured_output": None,
    }

    config = build_config(student_id, session_id)
    result = await _graph.ainvoke(initial_state, config=config)
    opening_msg = _get_last_ai_message(result.get("messages", []))

    logger.info("interview.session_started", session_id=session_id, target_position=req.target_position)
    return {"session_id": session_id, "target_position": req.target_position, "status": "in_progress", "message": opening_msg}


# ════════════════════════════════════════════════════════════════
# POST /sessions/{session_id}/chat — 发送消息
# ════════════════════════════════════════════════════════════════

@router.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest, current_user: dict = Depends(get_current_user)):
    student_id = current_user["user_id"]
    tenant_id  = current_user["tenant_id"]

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT session_id FROM interview_sessions WHERE session_id = :session_id AND tenant_id = :tenant_id AND student_id = :student_id"),
            {"session_id": session_id, "tenant_id": tenant_id, "student_id": student_id},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="面试会话不存在")

    config = build_config(student_id, session_id)
    state_update = {"messages": [HumanMessage(content=req.message)], "student_id": student_id, "session_id": session_id, "tenant_id": tenant_id}
    result = await _graph.ainvoke(state_update, config=config)

    current_stage = result.get("current_stage", InterviewStage.WARMUP.value)
    total_turns   = result.get("total_turn_count", 0)
    is_finished   = current_stage == InterviewStage.FINISHED.value
    reply         = _get_last_ai_message(result.get("messages", []))

    response = {"session_id": session_id, "reply": reply, "current_stage": current_stage, "total_turns": total_turns, "is_finished": is_finished}
    if is_finished:
        report = result.get("report") or {}
        response["report_summary"] = {"overall_score": report.get("overall_score", 0), "strengths": report.get("strengths", []), "improvements": report.get("improvements", [])}
    return response


# ════════════════════════════════════════════════════════════════
# GET /sessions/{session_id}/report — 查询完整报告
# ════════════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}/report")
async def get_report(session_id: str, current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT session_id, target_position, status, overall_score, report, finished_at FROM interview_sessions WHERE session_id = :session_id AND student_id = :student_id"),
            {"session_id": session_id, "student_id": current_user["user_id"]},
        )
        row = result.mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="面试记录不存在")
    if row["status"] != "finished":
        raise HTTPException(status_code=400, detail="面试尚未结束，报告生成中")

    report = row["report"] if isinstance(row["report"], dict) else (json.loads(row["report"]) if row["report"] else {})
    return {
        "session_id": session_id, "target_position": row["target_position"],
        "overall_score": row["overall_score"], "dimensions": report.get("dimensions", []),
        "strengths": report.get("strengths", []), "improvements": report.get("improvements", []),
        "overall_comment": report.get("overall_comment", ""),
        "recommended_topics": report.get("recommended_topics", []),
        "next_step_advice": report.get("next_step_advice", ""),
    }


# ════════════════════════════════════════════════════════════════
# GET /sessions — 获取历史面试列表
# ════════════════════════════════════════════════════════════════

@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT session_id, target_position, overall_score, status, finished_at, created_at FROM interview_sessions WHERE student_id = :student_id ORDER BY created_at DESC LIMIT 50"),
            {"student_id": current_user["user_id"]},
        )
        rows = result.mappings().all()

    items = [{"session_id": row["session_id"], "target_position": row["target_position"], "overall_score": row["overall_score"], "status": row["status"], "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None, "created_at": row["created_at"].isoformat() if row["created_at"] else None} for row in rows]
    return {"items": items, "total": len(items)}


# ════════════════════════════════════════════════════════════════
# POST /sessions/{session_id}/chat/stream — SSE 流式
# ════════════════════════════════════════════════════════════════

@router.post("/sessions/{session_id}/chat/stream")
async def chat_stream(session_id: str, req: ChatRequest, current_user: dict = Depends(get_current_user)):
    student_id = current_user["user_id"]
    tenant_id  = current_user["tenant_id"]

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT session_id FROM interview_sessions WHERE session_id = :sid AND tenant_id = :tid"),
            {"sid": session_id, "tid": tenant_id},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="面试会话不存在")

    config = build_config(student_id, session_id)
    state_update = {"messages": [HumanMessage(content=req.message)], "student_id": student_id, "session_id": session_id, "tenant_id": tenant_id}

    async def event_generator():
        try:
            async for event in _graph.astream_events(state_update, config=config, version="v2"):
                evt  = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                if evt == "on_chat_model_stream" and node == "generate_response":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        yield {"data": json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False)}

            final = await _graph.aget_state(config)
            state = final.values if final else {}
            current_stage = state.get("current_stage", InterviewStage.WARMUP.value)
            total_turns   = state.get("total_turn_count", 0)
            is_finished   = current_stage == InterviewStage.FINISHED.value
            reply         = _get_last_ai_message(state.get("messages", []))

            done_payload = {"type": "done", "reply": reply, "current_stage": current_stage, "total_turns": total_turns, "is_finished": is_finished}
            if is_finished:
                report = state.get("report") or {}
                done_payload["report_summary"] = {"overall_score": report.get("overall_score", 0), "strengths": report.get("strengths", []), "improvements": report.get("improvements", [])}
            yield {"data": json.dumps(done_payload, ensure_ascii=False)}

        except Exception as e:
            logger.error("interview.chat_stream_error", error=str(e), exc_info=True)
            yield {"data": json.dumps({"type": "error", "message": "流式输出异常，请重试"}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())
