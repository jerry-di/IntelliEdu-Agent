"""
unified_chat - 统一 AI 助手入口
前置拦截 + LLM 路由 → 单 Agent 直达 / 引导跳转 / Pipeline 串联

"""
import json
import re
from dataclasses import dataclass
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from backend.core.orchestrator import AgentType, ExecutionMode
from backend.core.memory import build_thread_id
from backend.core.llm_factory import get_llm
from backend.dependencies import get_current_user
from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_AGENT_DISPLAY = {
    AgentType.QA: "智能问答", AgentType.EXAM: "试卷批改",
    AgentType.RESUME: "简历审查", AgentType.INTERVIEW: "模拟面试",
}

_GUIDANCE = {
    AgentType.EXAM: {"message": "检测到您需要进行试卷批改。请前往「试卷批改」页面上传 Word 格式答卷，AI 将自动完成三轨批改并分析知识薄弱点。", "action_label": "前往试卷批改", "action_url": "/exam"},
    AgentType.RESUME: {"message": "检测到您需要进行简历审查。请前往「简历审查」页面上传 PDF 格式简历，AI 将从六个维度进行评分并给出改进建议。", "action_label": "前往简历审查", "action_url": "/resume"},
    AgentType.INTERVIEW: {"message": "检测到您需要进行模拟面试。请前往「模拟面试」页面选择目标岗位和技术方向，AI 将进行技术原理与项目深挖双轨考察。", "action_label": "前往模拟面试", "action_url": "/interview"},
}

_PROGRESS_LABELS = {"classify_query": "理解问题中...", "load_memory_and_embed": "检索知识库...", "retrieve": "召回相关文档...", "rerank": "精排中...", "hyde_generate": "理解问题中...", "multi_query_rewrite": "改写查询中...", "generate_general": "思考中..."}
_GENERATE_NODES = {"generate_rag", "generate_direct", "generate_general"}


# ════════════════════════════════════════════════════════════════
# 规则前置拦截
# ════════════════════════════════════════════════════════════════

_STRIP_TAIL_RE = re.compile(r"[\s!！?？。~～,.，。]+$")

_HELLO_KEYWORDS = frozenset(["你好", "您好", "hi", "hello", "hey", "哈喽", "嗨", "在吗", "在不在", "在线吗", "有人吗"])
_THANKS_KEYWORDS = frozenset(["谢谢", "感谢", "多谢", "谢了", "非常感谢", "万分感谢", "辛苦了", "辛苦", "麻烦了", "不好意思", "太棒了", "太好了", "厉害", "厉害了", "牛", "牛啊", "牛逼", "好的好的", "明白了", "懂了", "知道了", "收到"])
_BYE_KEYWORDS = frozenset(["再见", "拜拜", "拜", "88", "886", "bye", "goodbye", "byebye", "下次见", "下次再聊", "先走了", "先撤了", "溜了", "闪了"])

_IDENTITY_RE = re.compile(r"(你|您)(是谁|叫什么|的名字|是什么|是.*AI|是.*机器人|是.*助手)|介绍.{0,4}(你自己|自己|一下)|你是谁|你叫(啥|什么名)", re.IGNORECASE)
_CAPABILITY_RE = re.compile(r"(你|您)(能|可以|会).{0,6}(做|帮|干)|(你|您).{0,4}(功能|用途|能力|特点)|怎么(用|使用)(你|您|这个)?|(使用说明|帮助菜单|help|usage)|你能帮(我|忙)吗", re.IGNORECASE)

_MULTI_AGENT_TIP = "- **多 Agent 协同**：描述综合需求（如「帮我准备求职」），AI 将自动串联简历审查 + 模拟面试两个 Agent 协同为您服务\n"

_REPLY_HELLO = "您好！我是 EduAgent AI 助手。\n\n我可以帮您：\n- **技术问答**：直接输入问题，我会从知识库中检索并解答\n- **试卷批改**：告诉我「我要提交试卷」，AI 完成三轨批改和知识薄弱点分析\n- **简历审查**：告诉我「帮我看看简历」，AI 给出六维度评分与改进建议\n- **模拟面试**：告诉我「我要模拟面试」，开启技术双轨考察\n" + _MULTI_AGENT_TIP + "\n请问有什么可以帮到您？"
_REPLY_THANKS = "不客气，很高兴能帮到您！\n\n如果还有其他问题，随时告诉我，我随时在线。"
_REPLY_BYE = "再见！希望今天的学习对您有所帮助，期待下次与您交流。\n\n祝您学习顺利！"
_REPLY_IDENTITY = "我是 **EduAgent AI 助手**，一个面向 IT 教育培训的智能学习辅助系统。\n\n我由多个专业 Agent 协同构成：\n- **智能问答 Agent**：基于 RAG 知识库，7x24 即时解答技术问题\n- **试卷批改 Agent**：AI 三轨并行批改 + 知识薄弱点分析\n- **简历审查 Agent**：六维度质量评审，提供原文定位的修改建议\n- **模拟面试 Agent**：技术原理 + 项目深挖双轨考察，生成结构化报告\n" + _MULTI_AGENT_TIP + "\n有什么可以帮到您吗？"
_REPLY_CAPABILITY = "我能为您提供以下功能：\n\n**单 Agent 直达**\n- 直接输入技术问题 → 智能问答（RAG 知识库检索）\n- 「提交试卷批改」 → 上传 Word 答卷，AI 完成批改并分析薄弱点\n- 「审查我的简历」 → 上传 PDF 简历，六维度评分 + 改进建议\n- 「开始模拟面试」 → 选择岗位方向，进入双轨面试考察\n\n**多 Agent 协同**\n" + _MULTI_AGENT_TIP + "\n直接告诉我您的需求，我会自动路由到最合适的 Agent。"


def _pre_filter(text: str) -> str | None:
    t = text.strip()
    t_lower = _STRIP_TAIL_RE.sub("", t.lower())
    if t_lower in _HELLO_KEYWORDS: return _REPLY_HELLO
    if t_lower in _THANKS_KEYWORDS: return _REPLY_THANKS
    if t_lower in _BYE_KEYWORDS: return _REPLY_BYE
    if _IDENTITY_RE.search(t): return _REPLY_IDENTITY
    if _CAPABILITY_RE.search(t): return _REPLY_CAPABILITY
    return None


# ════════════════════════════════════════════════════════════════
# LLM 意图路由
# ════════════════════════════════════════════════════════════════

_ROUTE_PROMPT = """判断用户需求应路由到哪个功能。

可选功能：
- qa          : 技术知识问答（用户直接提问，不涉及文件上传）
- exam        : 试卷/作业批改（需上传 Word 答卷）
- resume      : 简历审查（需上传 PDF 简历）
- interview   : 模拟面试
- multi_agent : 综合求职准备（同时提到简历 + 面试，或"全套""一条龙""求职准备"）
- clarify     : 意图不明确，无法判断，需要追问

严格按以下 JSON 格式返回，不要有其他内容：
{{"label": "功能名", "reason": "一句话说明判断依据"}}

用户输入：{message}"""

_LABEL_TO_AGENT: dict[str, AgentType] = {"qa": AgentType.QA, "exam": AgentType.EXAM, "resume": AgentType.RESUME, "interview": AgentType.INTERVIEW, "multi_agent": AgentType.QA, "clarify": AgentType.QA}
_LABEL_TO_MODE: dict[str, ExecutionMode] = {"qa": ExecutionMode.SINGLE, "exam": ExecutionMode.SINGLE, "resume": ExecutionMode.SINGLE, "interview": ExecutionMode.SINGLE, "multi_agent": ExecutionMode.PIPELINE, "clarify": ExecutionMode.CLARIFY}
_VALID_LABELS = frozenset(_LABEL_TO_AGENT.keys())


@dataclass
class _RouteResult:
    label: str; agent_type: AgentType; execution_mode: ExecutionMode; confidence: float; reason: str


async def _llm_route(message: str) -> _RouteResult:
    try:
        llm = get_llm("intent", temperature=0)
        resp = await llm.ainvoke([HumanMessage(content=_ROUTE_PROMPT.format(message=message))])
        raw = resp.text.strip()
        parsed = json.loads(raw)
        label = parsed.get("label", "qa").strip().lower()
        reason = parsed.get("reason", "LLM 路由判断")
        if label not in _VALID_LABELS:
            logger.warning("unified_chat.llm_route_unknown_label", label=label, fallback="qa")
            label = "qa"
        logger.info("unified_chat.llm_route_result", label=label, reason=reason)
    except Exception as e:
        logger.warning("unified_chat.llm_route_failed", error=str(e), fallback="qa")
        label, reason = "qa", "路由判断异常，默认转入智能问答"
    return _RouteResult(label=label, agent_type=_LABEL_TO_AGENT[label], execution_mode=_LABEL_TO_MODE[label], confidence=0.85, reason=reason)


# ════════════════════════════════════════════════════════════════
# SSE 小工具
# ════════════════════════════════════════════════════════════════

def _sse(data: dict) -> dict:
    """dict → SSE 数据帧（ensure_ascii=False 保留中文）"""
    return {"data": json.dumps(data, ensure_ascii=False)}


# ════════════════════════════════════════════════════════════════
# QA 流式执行器（复用编排器缓存的 QA 图）
# ════════════════════════════════════════════════════════════════

async def _stream_qa_agent(req: "UnifiedChatRequest", current_user: dict):
    """在统一入口中流式执行 QA Agent。"""
    from backend.core.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    graph = orchestrator._get_agent_graph(AgentType.QA)

    thread_id = build_thread_id(current_user["user_id"], req.session_id)
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "student_id": current_user["user_id"],
        "tenant_id": current_user["tenant_id"],
        "session_id": req.session_id,
        "course_id": None,
        "query_type": "PRECISE",
    }
    config = {"configurable": {"thread_id": thread_id}}

    answer_mode = None
    confidence = 0.0
    sources: list[str] = []

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            evt  = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            if evt == "on_chain_start" and node in _PROGRESS_LABELS:
                yield _sse({"type": "progress", "stage": _PROGRESS_LABELS[node]})

            elif evt == "on_chat_model_stream" and node in _GENERATE_NODES:
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})

            elif evt == "on_chain_end" and node in _GENERATE_NODES:
                output = event["data"].get("output", {})
                if output.get("answer_mode"):
                    answer_mode = output["answer_mode"]
                if output.get("sources"):
                    sources = output["sources"]
                confidence = (output.get("structured_output") or {}).get("confidence", confidence)

    except Exception as e:
        logger.error("unified_chat.qa_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "问答服务异常，请稍后重试"})
        return

    if answer_mode:
        yield _sse({"type": "meta", "answer_mode": answer_mode, "confidence": confidence, "sources": sources})


# ════════════════════════════════════════════════════════════════
# 请求模型
# ════════════════════════════════════════════════════════════════

class UnifiedChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    message:    str = Field(..., min_length=1, max_length=2000, description="用户输入")


# ════════════════════════════════════════════════════════════════
# POST /stream — 统一流式入口
# ════════════════════════════════════════════════════════════════

@router.post("/stream")
async def unified_chat_stream(req: UnifiedChatRequest, current_user: dict = Depends(get_current_user)):
    """统一 AI 助手流式接口（SSE）。1.前置拦截 → 2.LLM路由 → 3.分发处理。"""

    async def event_generator():
        pre_reply = _pre_filter(req.message)
        if pre_reply is not None:
            yield _sse({"type": "token", "content": pre_reply})
            yield _sse({"type": "done"})
            return

        decision = await _llm_route(req.message)

        yield _sse({
            "type": "routing_decision",
            "agent_type": decision.agent_type.value,
            "agent_display": _AGENT_DISPLAY.get(decision.agent_type, ""),
            "confidence": round(decision.confidence, 4),
            "reason": decision.reason,
            "execution_mode": decision.execution_mode.value,
        })

        label = decision.label

        if label == "qa":
            async for event in _stream_qa_agent(req, current_user):
                yield event

        elif label in ("exam", "resume", "interview"):
            guidance = _GUIDANCE.get(decision.agent_type, {})
            yield _sse({
                "type": "guidance",
                "message": guidance.get("message", "请前往对应功能页面操作"),
                "action_label": guidance.get("action_label", ""),
                "action_url": guidance.get("action_url", "/dashboard"),
            })

        elif label == "multi_agent":
            yield _sse({
                "type": "pipeline_plan",
                "title": "求职全链路",
                "intro": "已为您规划「求职全链路」，建议按顺序完成以下两个步骤，简历审查结果将作为模拟面试的参考依据。",
                "steps": [
                    {"step": 1, "agent_type": "resume", "label": "简历审查",
                     "desc": "上传 PDF 简历，AI 从六个维度评分并给出带原文定位的改进建议",
                     "action_label": "开始简历审查", "action_url": "/resume",
                     "tip": "建议先完成此步骤，审查结果将作为面试的参考依据"},
                    {"step": 2, "agent_type": "interview", "label": "模拟面试",
                     "desc": "基于简历背景，进行技术原理 + 项目深挖双轨考察，生成结构化雷达图报告",
                     "action_label": "开始模拟面试", "action_url": "/interview",
                     "tip": "完成简历审查后进行效果更佳"},
                ],
            })

        else:
            yield _sse({
                "type": "guidance",
                "message": "您的问题我还不太确定应该用哪个功能来帮您，能否描述得更具体一些？例如：您是想提问技术知识、提交试卷批改、审查简历，还是进行模拟面试？",
                "action_label": "", "action_url": "",
            })

        yield _sse({"type": "done"})

    return EventSourceResponse(event_generator())
