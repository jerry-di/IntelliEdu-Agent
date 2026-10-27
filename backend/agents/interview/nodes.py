"""
nodes - 模拟面试 Agent 节点实现

"""
import asyncio
import json
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from sqlalchemy import text

from backend.agents.interview.state import (
    InterviewState, InterviewStage, AnswerQuality, ReportWrapper,
)
from backend.agents.interview.prompts import (
    SYSTEM_PROMPT, WARMUP_PROMPT, INTRO_EVAL_TECH_FIRST_PROMPT,
    TECH_BASE_PROMPT, PROJECT_PROMPT, CLOSING_PROMPT,
    STAGE_TRANSITION_PROMPTS, EVALUATE_ANSWER_PROMPT,
    EVALUATE_THINK_PROMPT, GENERATE_REPORT_PROMPT,
    TECH_BASE_GENERATE_PROMPT,
)
from backend.core.llm_factory import get_llm, get_structured_llm
from backend.core.memory import (
    build_thread_id, should_trigger_summary,
    compress_to_summary, trim_messages_to_window,
)
from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal

logger = get_logger(__name__)

DEFAULT_MAX_TURNS = 40
MAX_FOLLOWUP_PER_QUESTION = 2

STAGE_MIN_TURNS = {
    InterviewStage.WARMUP.value:    1,
    InterviewStage.TECH_BASE.value: 6,
    InterviewStage.PROJECT.value:   2,
    InterviewStage.CLOSING.value:   2,
}
STAGE_MAX_TURNS = {
    InterviewStage.WARMUP.value:    4,
    InterviewStage.TECH_BASE.value: 12,
    InterviewStage.PROJECT.value:   14,
    InterviewStage.CLOSING.value:   4,
}


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _msg_text(msg: BaseMessage) -> str:
    """安全提取消息文本，兼容 str 和 list（多模态）content。"""
    content = msg.content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


# ════════════════════════════════════════════════════════════════
# LLM 动态出题
# ════════════════════════════════════════════════════════════════

async def _generate_questions_by_llm(target_position: str, count: int = 8) -> list:
    """根据目标岗位用 LLM 动态生成技术面试题。"""
    try:
        llm = get_llm("interview", temperature=0.3)
        prompt = TECH_BASE_GENERATE_PROMPT.format(position=target_position, count=count)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = _msg_text(response).strip()

        if "```" in text:
            parts = text.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("["):
                    text = stripped
                    break

        questions_raw = json.loads(text)
        result = []
        for i, q in enumerate(questions_raw):
            if not isinstance(q, dict) or not q.get("content"):
                continue
            result.append({
                "id":         f"llm_{i}",
                "content":    q["content"],
                "difficulty": q.get("difficulty", "medium"),
                "tags":       q.get("tags", [target_position]),
                "asked":      False,
            })
        logger.info("generate_questions_llm.done", count=len(result), position=target_position)
        return result
    except Exception as e:
        logger.warning("generate_questions_llm.failed", error=str(e))
        return []


# ════════════════════════════════════════════════════════════════
# load_context_node — 上下文加载（每轮第一个执行）
# ════════════════════════════════════════════════════════════════

async def load_context_node(state: InterviewState) -> dict:
    """
    上下文加载节点（每轮第一个执行）。

    每轮都执行：加载历史摘要（from DB）。
    首轮额外执行：初始化 State、加载简历联动数据、从题库拉题。
    """
    student_id = state.get("student_id") or ""
    session_id = state.get("session_id") or ""
    if not student_id or not session_id:
        logger.error("load_context.missing_ids", state_keys=list(state.keys()))
        return {}
    thread_id = build_thread_id(student_id, session_id)

    is_first_turn = state.get("total_turn_count", 0) == 0
    updates: dict = {}

    # ── 内部查询函数 ─────────────────────────────────────────

    async def _fetch_resume(resume_review_id: str) -> dict:
        try:
            async with AsyncSessionLocal() as s:
                result = await s.execute(
                    text("SELECT structured_data FROM resume_reviews WHERE id = :id"),
                    {"id": resume_review_id},
                )
                row = result.mappings().fetchone()
            if row and row["structured_data"]:
                structured = (
                    row["structured_data"]
                    if isinstance(row["structured_data"], dict)
                    else json.loads(row["structured_data"])
                )
                return {
                    "resume_projects": structured.get("projects", []),
                    "resume_skills":   structured.get("skills_list", []),
                }
        except Exception as e:
            logger.warning("load_context.resume_load_failed", error=str(e))
        return {}

    async def _fetch_questions(target: str) -> list:
        try:
            async with AsyncSessionLocal() as s:
                result = await s.execute(
                    text("""
                        SELECT id, content, difficulty, tags
                        FROM interview_questions
                        WHERE (target_position = :target OR target_position = 'general')
                          AND is_active = TRUE
                        ORDER BY
                            CASE difficulty
                                WHEN 'medium' THEN 1
                                WHEN 'easy'   THEN 2
                                ELSE 3
                            END,
                            RANDOM()
                        LIMIT 20
                    """),
                    {"target": target},
                )
                rows = result.mappings().all()
            return [
                {
                    "id":         str(r["id"]),
                    "content":    r["content"],
                    "difficulty": r["difficulty"],
                    "tags":       r["tags"] if isinstance(r["tags"], list) else (r["tags"] or []),
                    "asked":      False,
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("load_context.question_bank_failed", error=str(e))
            return []

    async def _fetch_summary() -> str | None:
        try:
            async with AsyncSessionLocal() as s:
                result = await s.execute(
                    text("SELECT summary FROM interview_sessions WHERE thread_id = :tid"),
                    {"tid": thread_id},
                )
                row = result.mappings().fetchone()
            return row["summary"] if row else None
        except Exception as e:
            logger.warning("load_context.summary_load_failed", error=str(e))
            return None

    # ── 首轮 vs 非首轮 ─────────────────────────────────────

    if is_first_turn:
        updates.update({
            "current_stage":       InterviewStage.WARMUP.value,
            "stage_turn_count":    0,
            "total_turn_count":    0,
            "followup_count":      0,
            "projects_asked":      [],
            "last_answer_quality": AnswerQuality.ADEQUATE.value,
            "max_turns":           DEFAULT_MAX_TURNS,
            "report":              None,
            "should_summarize":    False,
            "resume_projects":     state.get("resume_projects") or [],
            "resume_skills":       state.get("resume_skills") or [],
        })

        resume_review_id = state.get("resume_review_id")
        target = state.get("target_position", "")

        async def _noop_resume():
            return {}

        resume_coro    = _fetch_resume(resume_review_id) if resume_review_id else _noop_resume()
        questions_coro = _fetch_questions(target)
        llm_q_coro     = _generate_questions_by_llm(target)
        summary_coro   = _fetch_summary()

        resume_data, db_questions, llm_questions, existing_summary = await asyncio.gather(
            resume_coro, questions_coro, llm_q_coro, summary_coro
        )

        question_bank = llm_questions + db_questions if llm_questions else db_questions

        if resume_data:
            updates["resume_projects"] = resume_data.get("resume_projects", [])
            updates["resume_skills"]   = resume_data.get("resume_skills", [])
            logger.info("load_context.resume_loaded", projects=len(updates["resume_projects"]))

        updates["question_bank"]    = question_bank
        updates["existing_summary"] = existing_summary
        logger.info("load_context.question_bank_loaded", count=len(question_bank))
    else:
        updates["existing_summary"] = await _fetch_summary()

    return updates


# ════════════════════════════════════════════════════════════════
# 阶段推进辅助函数
# ════════════════════════════════════════════════════════════════

def _check_advance_condition(stage: str, state: InterviewState, latest_msg: str) -> bool:
    """判断当前阶段「内容上」是否满足推进条件（轮数已达下限后才调用）。"""
    if stage == InterviewStage.WARMUP.value:
        return True

    if stage == InterviewStage.TECH_BASE.value:
        bank = state.get("question_bank", [])
        asked_count = sum(1 for q in bank if q.get("asked", False))
        return asked_count >= min(8, len(bank))

    if stage == InterviewStage.PROJECT.value:
        projects       = state.get("resume_projects", [])
        projects_asked = state.get("projects_asked", [])
        if not projects:
            return True
        return len(projects_asked) >= len(projects)

    return False


def _next_stage(current: str) -> str:
    """返回下一个阶段的枚举值字符串（按固定顺序往后走一格）。"""
    order = [
        InterviewStage.WARMUP.value,
        InterviewStage.TECH_BASE.value,
        InterviewStage.PROJECT.value,
        InterviewStage.CLOSING.value,
        InterviewStage.FINISHED.value,
    ]
    try:
        idx = order.index(current)
        return order[idx + 1] if idx + 1 < len(order) else InterviewStage.FINISHED.value
    except ValueError:
        return InterviewStage.CLOSING.value


def _get_last_student_answer(messages: list) -> str:
    """提取消息列表中最后一条学员（Human）消息的文本。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _msg_text(msg)
    return ""


# ════════════════════════════════════════════════════════════════
# check_stage_node — 阶段推进（纯逻辑）
# ════════════════════════════════════════════════════════════════

async def check_stage_node(state: InterviewState) -> dict:
    """
    阶段推进节点（纯逻辑，不调 LLM）：判断当前阶段是否要推进 / 强制结束。

    推进条件：
        WARMUP → TECH_BASE: turns >= 1
        TECH_BASE → PROJECT: stage_turns >= 6 且已问 >= 8 题
        PROJECT → CLOSING:   stage_turns >= 2 且所有项目已深挖
        CLOSING → FINISHED:  stage_turns >= 2
        强制终止：total_turns >= max_turns-2 或命中结束关键词
    """
    current_stage = state.get("current_stage", InterviewStage.WARMUP.value)
    stage_turns   = state.get("stage_turn_count", 0)
    total_turns   = state.get("total_turn_count", 0)
    max_turns     = state.get("max_turns", DEFAULT_MAX_TURNS)
    messages      = state.get("messages", [])

    if current_stage == InterviewStage.FINISHED.value:
        return {}

    latest_student_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_student_msg = _msg_text(msg)
            break

    force_end_keywords = ["结束面试", "不想继续", "面试结束", "结束吧", "到此为止"]
    is_forced_end = (
        total_turns >= max_turns - 2
        or any(kw in latest_student_msg for kw in force_end_keywords)
    )

    if is_forced_end:
        logger.info("check_stage.forced_to_finished", reason="keyword_or_max_turns")
        return {"current_stage": InterviewStage.FINISHED.value}

    if current_stage == InterviewStage.CLOSING.value:
        if stage_turns >= STAGE_MIN_TURNS[InterviewStage.CLOSING.value]:
            logger.info("check_stage.advance", from_stage="closing", to_stage="finished")
            return {"current_stage": InterviewStage.FINISHED.value}
        return {}

    stage_min = STAGE_MIN_TURNS.get(current_stage, 2)
    stage_max = STAGE_MAX_TURNS.get(current_stage, 10)

    should_advance = False
    if stage_turns >= stage_max:
        should_advance = True
    elif stage_turns >= stage_min:
        should_advance = _check_advance_condition(current_stage, state, latest_student_msg)

    if should_advance:
        next_stage = _next_stage(current_stage)
        logger.info("check_stage.advance", from_stage=current_stage, to_stage=next_stage)
        return {
            "current_stage":    next_stage,
            "stage_turn_count": 0,
            "followup_count":   0,
            "current_question": None,
        }

    return {}


# ════════════════════════════════════════════════════════════════
# evaluate_answer_node — 回答质量评估
# ════════════════════════════════════════════════════════════════

async def evaluate_answer_node(state: InterviewState) -> dict:
    """
    评估学员上一轮回答质量（EXCELLENT / ADEQUATE / WEAK / NO_ANSWER）。

    两个职责：
        1. 评估回答质量 → last_answer_quality
        2. 维护轮次计数 → total_turn_count += 1, stage_turn_count += 1
    """
    total_turns   = state.get("total_turn_count", 0)
    current_stage = state.get("current_stage", InterviewStage.WARMUP.value)
    messages      = state.get("messages", [])

    if total_turns == 0 or current_stage == InterviewStage.WARMUP.value:
        return {
            "last_answer_quality": AnswerQuality.ADEQUATE.value,
            "total_turn_count":    total_turns + 1,
            "stage_turn_count":    state.get("stage_turn_count", 0) + 1,
        }

    student_answer = _get_last_student_answer(messages)

    if not student_answer.strip() or student_answer.strip() in ["不知道", "不清楚", "没学过"]:
        return {
            "last_answer_quality": AnswerQuality.NO_ANSWER.value,
            "total_turn_count":    total_turns + 1,
            "stage_turn_count":    state.get("stage_turn_count", 0) + 1,
        }

    current_q     = state.get("current_question") or {}
    question_text = current_q.get("content", "上一个问题")
    answer_text   = student_answer[:800]

    reasoning_trace = ""
    try:
        think_prompt = EVALUATE_THINK_PROMPT.format(question=question_text, answer=answer_text)
        think_llm  = get_llm("qa", temperature=0)
        think_resp = await think_llm.ainvoke([HumanMessage(content=think_prompt)])
        reasoning_trace = _msg_text(think_resp).strip()
        logger.debug("evaluate_think.done", stage=current_stage)
    except Exception as e:
        logger.warning("evaluate_think.failed", error=str(e))

    think_context = f"\n\n【评估前分析】\n{reasoning_trace}" if reasoning_trace else ""
    prompt = EVALUATE_ANSWER_PROMPT.format(question=question_text, answer=answer_text) + think_context

    try:
        llm = get_llm("qa", temperature=0)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        quality_str = _msg_text(response).strip().lower()

        if "excellent" in quality_str or "优秀" in quality_str:
            quality = AnswerQuality.EXCELLENT.value
        elif "weak" in quality_str or "较弱" in quality_str:
            quality = AnswerQuality.WEAK.value
        elif "no_answer" in quality_str or "未作答" in quality_str:
            quality = AnswerQuality.NO_ANSWER.value
        else:
            quality = AnswerQuality.ADEQUATE.value
    except Exception as e:
        logger.warning("evaluate_answer.failed", error=str(e))
        quality = AnswerQuality.ADEQUATE.value

    logger.info("evaluate_answer.done", stage=current_stage, quality=quality, total_turns=total_turns + 1)

    return {
        "last_answer_quality": quality,
        "total_turn_count":    total_turns + 1,
        "stage_turn_count":    state.get("stage_turn_count", 0) + 1,
    }


# ════════════════════════════════════════════════════════════════
# generate_response 工具函数
# ════════════════════════════════════════════════════════════════

def _build_system_with_summary(state: InterviewState) -> str:
    """构建包含历史摘要的 System Prompt。"""
    position = state.get("target_position", "软件工程师")
    system   = SYSTEM_PROMPT.format(position=position)
    summary  = state.get("existing_summary")
    if summary:
        system += f"\n\n【学员面试历史摘要】\n{summary}"
    return system


def _build_messages(system: str, history: list, current_prompt: str, window: int = 10) -> list:
    """构建发送给 LLM 的消息列表：[SystemMessage] + 滑动窗口历史 + current_prompt。"""
    result = [SystemMessage(content=system)]
    windowed = trim_messages_to_window(history[:-1], window_size=window)
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            result.append(msg)
    result.append(HumanMessage(content=current_prompt))
    return result


# ════════════════════════════════════════════════════════════════
# _respond_warmup — 热身阶段
# ════════════════════════════════════════════════════════════════

async def _respond_warmup(state: InterviewState) -> tuple[str, dict]:
    """破冰热身阶段：只生成欢迎开场白，邀请自我介绍。"""
    messages = state.get("messages", [])
    position = state.get("target_position", "软件工程师")
    system   = _build_system_with_summary(state)

    prompt = WARMUP_PROMPT["opening"].format(position=position)
    llm = get_llm("interview", temperature=0.7)
    response = await llm.ainvoke(_build_messages(system, messages, prompt, window=4))
    return _msg_text(response).strip(), {}


# ════════════════════════════════════════════════════════════════
# _respond_tech_base — 技术基础阶段
# ════════════════════════════════════════════════════════════════

async def _respond_tech_base(state: InterviewState) -> tuple[str, dict]:
    """技术基础阶段：出题或追问。三条路径：首轮过渡 / 追问 / 换题。"""
    messages       = state.get("messages", [])
    quality        = state.get("last_answer_quality", AnswerQuality.ADEQUATE.value)
    followup_count = state.get("followup_count", 0)
    current_q      = state.get("current_question")
    bank           = state.get("question_bank", [])
    stage_turns    = state.get("stage_turn_count", 0)
    position       = state.get("target_position", "软件工程师")
    system         = _build_system_with_summary(state)
    updates: dict  = {}

    # 路径1：刚从 WARMUP 推进（stage_turns==1 且尚无 current_question）
    if stage_turns == 1 and current_q is None:
        unasked = [q for q in bank if not q.get("asked", False)]
        if not unasked:
            return ("感谢你的自我介绍！接下来我们开始技术基础环节。", {})

        first_q  = unasked[0]
        new_bank = [{**q, "asked": True} if q["id"] == first_q["id"] else q for q in bank]
        updates["question_bank"]    = new_bank
        updates["current_question"] = first_q
        updates["followup_count"]   = 0

        intro_text = _get_last_student_answer(messages)
        prompt = INTRO_EVAL_TECH_FIRST_PROMPT.format(
            intro=intro_text[:500], position=position, first_question=first_q["content"])
        llm = get_llm("interview", temperature=0.5)
        response = await llm.ainvoke(_build_messages(system, messages, prompt, window=4))
        return _msg_text(response).strip(), updates

    # 判断是否追问：只有 EXCELLENT 且未满 2 次
    should_followup = (
        current_q is not None
        and followup_count < MAX_FOLLOWUP_PER_QUESTION
        and quality == AnswerQuality.EXCELLENT.value
    )

    if should_followup:                                  # 路径2·追问
        tech_stack_str = ", ".join(current_q.get("tags", []))
        prompt = TECH_BASE_PROMPT["followup"].format(
            question=current_q["content"],
            answer=_get_last_student_answer(messages)[:600],
            quality=quality,
            tech_stack=tech_stack_str,
        )
        updates["followup_count"] = followup_count + 1
    else:                                                # 路径3·换题
        unasked = [q for q in bank if not q.get("asked", False)]
        if not unasked:
            return ("好的，基础技术问题我们就聊到这里，接下来我想了解一下你的项目经历。",
                    {"current_question": None})

        next_q = unasked[0]
        new_bank = [{**q, "asked": True} if q["id"] == next_q["id"] else q for q in bank]
        updates["question_bank"]    = new_bank
        updates["current_question"] = next_q
        updates["followup_count"]   = 0

        prev_question = current_q["content"] if current_q else ""
        prev_answer   = _get_last_student_answer(messages)[:600]
        prompt = TECH_BASE_PROMPT["ask_with_feedback"].format(
            question=prev_question, answer=prev_answer,
            quality=quality, next_question=next_q["content"],
        )

    llm = get_llm("interview", temperature=0.4)
    response = await llm.ainvoke(_build_messages(system, messages, prompt, window=8))
    return _msg_text(response).strip(), updates


# ════════════════════════════════════════════════════════════════
# _respond_project — 项目深挖阶段（占位，7.8 实现）
# ════════════════════════════════════════════════════════════════

async def _respond_project(state: InterviewState) -> tuple[str, dict]:
    """
    项目深挖阶段：基于简历项目针对性追问。

    追问门槛比技术环节低：EXCELLENT 或 ADEQUATE 都可追问。
    """
    messages       = state.get("messages", [])
    quality        = state.get("last_answer_quality", AnswerQuality.ADEQUATE.value)
    followup_count = state.get("followup_count", 0)
    projects       = state.get("resume_projects", [])
    projects_asked = state.get("projects_asked", [])
    current_q      = state.get("current_question")
    position       = state.get("target_position", "工程师")
    system         = _build_system_with_summary(state)
    updates: dict  = {}

    if projects:
        next_project = None
        for p in projects:
            if p.get("name") not in projects_asked:
                next_project = p
                break

        should_followup = (
            current_q is not None
            and followup_count < MAX_FOLLOWUP_PER_QUESTION
            and quality in (AnswerQuality.EXCELLENT.value, AnswerQuality.ADEQUATE.value)
        )

        if should_followup:
            tech_stack_str = ", ".join(current_q.get("tech_stack", []))
            prompt = PROJECT_PROMPT["followup_with_feedback"].format(
                project_name=current_q.get("project_name", ""),
                question=current_q["content"],
                answer=_get_last_student_answer(messages)[:600],
                quality=quality,
                tech_stack=tech_stack_str,
                question_num=followup_count + 1,
            )
            updates["followup_count"] = followup_count + 1

        elif next_project:
            tech_stack_str = ", ".join(next_project.get("tech_stack", []))
            highlights_str = "\n".join(f"- {h}" for h in next_project.get("highlights", [])) or "（无量化数据）"

            prompt = PROJECT_PROMPT["new_project"].format(
                project_name=next_project["name"],
                project_role=next_project.get("role", "开发者"),
                tech_stack=tech_stack_str,
                highlights=highlights_str,
                description=next_project.get("description", "")[:200],
            )
            updates["current_question"] = {
                "content":      f"项目深挖：{next_project['name']}",
                "project_name": next_project["name"],
                "tech_stack":   next_project.get("tech_stack", []),
            }
            updates["projects_asked"] = projects_asked + [next_project["name"]]
            updates["followup_count"] = 0

        else:
            prompt = PROJECT_PROMPT["synthesis"].format(position=position)

    else:
        prompt = PROJECT_PROMPT["no_resume"].format(
            position=position,
            answer=_get_last_student_answer(messages)[:400],
        )

    llm = get_llm("interview", temperature=0.5)
    response = await llm.ainvoke(_build_messages(system, messages, prompt, window=10))
    return _msg_text(response).strip(), updates


# ════════════════════════════════════════════════════════════════
# _respond_closing — 反问收尾阶段
# ════════════════════════════════════════════════════════════════

async def _respond_closing(state: InterviewState) -> tuple[str, dict]:
    """反问收尾阶段：开放学员提问。"""
    messages    = state.get("messages", [])
    stage_turns = state.get("stage_turn_count", 0)
    position    = state.get("target_position", "工程师")
    system      = _build_system_with_summary(state)

    if stage_turns <= 1:
        prompt = CLOSING_PROMPT["opening"]
    else:
        last_answer = _get_last_student_answer(messages)
        prompt = CLOSING_PROMPT["respond_question"].format(
            question=last_answer[:300],
            position=position,
        )

    llm = get_llm("interview", temperature=0.6)
    response = await llm.ainvoke(_build_messages(system, messages, prompt, window=4))
    return _msg_text(response).strip(), {}


# ════════════════════════════════════════════════════════════════
# generate_response_node — 分派主函数
# ════════════════════════════════════════════════════════════════

async def generate_response_node(state: InterviewState) -> dict:
    """分派主函数：根据当前阶段调用对应子函数生成面试官回应。"""
    current_stage = state.get("current_stage", InterviewStage.WARMUP.value)
    stage_turns   = state.get("stage_turn_count", 0)
    messages      = state.get("messages", [])

    transition_prefix = ""
    if stage_turns == 1 and current_stage not in (
        InterviewStage.WARMUP.value, InterviewStage.TECH_BASE.value
    ):
        transition_prefix = STAGE_TRANSITION_PROMPTS.get(current_stage, "")

    if current_stage == InterviewStage.WARMUP.value:
        response_text, updates = await _respond_warmup(state)
    elif current_stage == InterviewStage.TECH_BASE.value:
        response_text, updates = await _respond_tech_base(state)
    elif current_stage == InterviewStage.PROJECT.value:
        response_text, updates = await _respond_project(state)
    elif current_stage == InterviewStage.CLOSING.value:
        response_text, updates = await _respond_closing(state)
    else:
        response_text = "感谢您参加本次模拟面试，请稍等，正在生成评估报告..."
        updates = {}

    if transition_prefix:
        response_text = transition_prefix + "\n\n" + response_text

    updates["messages"]         = [AIMessage(content=response_text)]
    updates["should_summarize"] = should_trigger_summary(messages)

    logger.info("generate_response.done", stage=current_stage, response_length=len(response_text))
    return updates


# ════════════════════════════════════════════════════════════════
# generate_report_node — 五维度报告生成
# ════════════════════════════════════════════════════════════════

async def generate_report_node(state: InterviewState) -> dict:
    """
    基于完整对话记录和历史摘要，生成五维度结构化评估报告。

    双层输入：历史摘要（早期对话）+ 最近 40 条消息（近期细节）。
    """
    messages  = state.get("messages", [])
    summary   = state.get("existing_summary", "")
    position  = state.get("target_position", "软件工程师")

    recent_messages = [
        msg for msg in messages[-40:]
        if isinstance(msg, (HumanMessage, AIMessage))
    ]
    conversation_text = "\n".join([
        f"{'面试官' if isinstance(m, AIMessage) else '学员'}：{_msg_text(m)}"
        for m in recent_messages
    ])

    prompt = GENERATE_REPORT_PROMPT.format(
        position=position,
        history_summary=summary or "（无历史摘要）",
        conversation=conversation_text[:4000],
        total_turns=state.get("total_turn_count", 0),
    )

    structured_llm = get_structured_llm("interview", InterviewReport)

    report_dict = None
    for attempt in range(2):
        try:
            result: InterviewReport = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT.format(position=position)),
                HumanMessage(content=prompt),
            ])
            if result is None:
                raise ValueError("structured output returned None")
            report_dict = result.model_dump()
            break
        except Exception as e:
            if attempt == 0:
                logger.warning("generate_report.retry", error=str(e))
                await asyncio.sleep(1)
            else:
                logger.warning("generate_report.failed", error=str(e))

    if report_dict is None:
        report_dict = {
            "dimensions":         [],
            "overall_score":      60,
            "strengths":          ["参与了完整的模拟面试流程"],
            "improvements":       ["建议深化技术基础知识", "加强项目经历的量化描述"],
            "overall_comment":    "感谢参加本次模拟面试，请参考各维度建议持续提升。",
            "recommended_topics": [],
            "next_step_advice":   "建议针对薄弱点系统复习，1-2周后再次模拟面试。",
        }

    overall_score = report_dict.get("overall_score", 0)
    closing_message = (
        f"好的，本次模拟面试到这里就结束了，感谢你的参与！\n\n"
        f"下面是你的整体表现总结：\n\n"
        f"**综合评分：{overall_score} / 100**\n\n"
        f"详细的各维度分析和改进建议可以在面试记录中查看。"
    )

    logger.info("generate_report.done", overall_score=overall_score,
                dimensions=len(report_dict.get("dimensions", [])))

    return {
        "report":   report_dict,
        "messages": [AIMessage(content=closing_message)],
        "structured_output": {
            "session_id":  state["session_id"],
            "report":      report_dict,
            "total_turns": state.get("total_turn_count", 0),
        },
    }


# ════════════════════════════════════════════════════════════════
# save_report_node — 报告写入数据库
# ════════════════════════════════════════════════════════════════

async def save_report_node(state: InterviewState) -> dict:
    """将 InterviewReport 写入 interview_sessions，更新 status=finished。"""
    thread_id = build_thread_id(state["student_id"], state["session_id"])
    report    = state.get("report", {})

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("""
                    UPDATE interview_sessions
                    SET report        = :report,
                        overall_score = :overall_score,
                        status        = 'finished',
                        finished_at   = NOW(),
                        updated_at    = NOW()
                    WHERE thread_id = :thread_id
                """),
                {
                    "report":        json.dumps(report, ensure_ascii=False),
                    "overall_score": report.get("overall_score", 0),
                    "thread_id":     thread_id,
                },
            )
            await session.commit()
            logger.info("save_report.done", thread_id=thread_id)
        except Exception as e:
            await session.rollback()
            logger.error("save_report.db_failed", error=str(e))

    return {}


# ════════════════════════════════════════════════════════════════
# save_memory_node — 摘要压缩 + 持久化
# ════════════════════════════════════════════════════════════════

async def save_memory_node(state: InterviewState) -> dict:
    """按需触发摘要压缩，将最新摘要 UPSERT 到 interview_sessions。"""
    messages   = state.get("messages", [])
    student_id = state["student_id"]
    session_id = state["session_id"]
    tenant_id  = state["tenant_id"]
    thread_id  = build_thread_id(student_id, session_id)
    summary    = state.get("existing_summary")

    if state.get("should_summarize", False):
        try:
            summary = await compress_to_summary(messages, summary)
        except Exception as e:
            logger.warning("save_memory.compress_failed", error=str(e))

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("""
                    INSERT INTO interview_sessions
                        (id, tenant_id, student_id, session_id, thread_id,
                         target_position, resume_review_id, summary, status)
                    VALUES
                        (:id, :tenant_id, :student_id, :session_id, :thread_id,
                         :target_position, :resume_review_id, :summary, 'in_progress')
                    ON CONFLICT (thread_id) DO UPDATE
                        SET summary    = EXCLUDED.summary,
                            updated_at = NOW()
                """),
                {
                    "id":               str(uuid.uuid4()),
                    "tenant_id":        tenant_id,
                    "student_id":       student_id,
                    "session_id":       session_id,
                    "thread_id":        thread_id,
                    "target_position":  state.get("target_position", ""),
                    "resume_review_id": state.get("resume_review_id"),
                    "summary":          summary,
                },
            )
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.warning("save_memory.db_failed", error=str(e))

    return {}
