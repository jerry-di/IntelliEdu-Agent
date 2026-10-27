"""
state - 模拟面试 Agent 状态定义
InterviewState + 枚举 + 结构化输出模型

"""
from enum import Enum
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════
# 枚举
# ════════════════════════════════════════════════════════════════

class InterviewStage(str, Enum):
    """面试的五个阶段（状态机的状态取值）。"""
    WARMUP    = "warmup"      # 热身：邀请自我介绍
    TECH_BASE = "tech_base"   # 技术基础：题库问答
    PROJECT   = "project"     # 项目深挖：针对简历项目追问
    CLOSING   = "closing"     # 反问收尾：让学员提问
    FINISHED  = "finished"    # 终态：触发报告生成


class AnswerQuality(str, Enum):
    """学员单次回答的质量标签（驱动追问/换题决策）。"""
    EXCELLENT = "excellent"   # 优秀：有技术细节/量化/原理
    ADEQUATE  = "adequate"    # 基本及格：方向对但缺深度
    WEAK      = "weak"        # 较弱：方向偏或太表面
    NO_ANSWER = "no_answer"   # 未作答：明说不知道或为空


# ════════════════════════════════════════════════════════════════
# 结构化输出模型
# ════════════════════════════════════════════════════════════════

class DimensionEval(BaseModel):
    """单个维度的评估结果"""
    dimension:  str       = Field(description="评估维度名称")
    score:      int       = Field(description="得分 0-100")
    comment:    str       = Field(description="1-2句评语")
    highlights: list[str] = Field(default_factory=list, description="亮点列举")
    weaknesses: list[str] = Field(default_factory=list, description="薄弱点列举")


class InterviewReport(BaseModel):
    """面试五维度评估报告"""
    dimensions:         list[DimensionEval]
    overall_score:      int       = Field(description="综合得分 0-100（五维度加权平均）")
    strengths:          list[str] = Field(description="2-3条核心优势（真实面试表现）")
    improvements:       list[str] = Field(description="2-3条重点提升方向")
    overall_comment:    str       = Field(description="2-3句综合评语")
    recommended_topics: list[str] = Field(description="建议重点复习的3-5个知识点")
    next_step_advice:   str       = Field(description="下一步备考建议（1-2句）")


class ReportWrapper(BaseModel):
    """InterviewReport 顶层包装（with_structured_output 需要顶层对象）"""
    report: InterviewReport


class AnswerEval(BaseModel):
    """单轮回答质量评估（LLM 结构化输出）"""
    quality:           str        = Field(description="AnswerQuality 枚举值")
    reasoning:         str        = Field(description="评估理由（1-2句）")
    knowledge_tags:    list[str]  = Field(default_factory=list, description="涉及的知识点")
    missed_points:     list[str]  = Field(default_factory=list, description="遗漏的要点")
    is_worth_followup: bool       = Field(default=False, description="是否值得追问")


# ════════════════════════════════════════════════════════════════
# InterviewState（22 字段，5 组）
# ════════════════════════════════════════════════════════════════

class InterviewState(TypedDict):
    # ── ① 请求上下文 ──
    messages:         Annotated[list[BaseMessage], add_messages]
    student_id:       str
    tenant_id:        str
    session_id:       str
    target_position:  str

    # ── ② 简历联动 ──
    resume_review_id: Optional[str]
    resume_projects:  list[dict]
    resume_skills:    list[str]

    # ── ③ 阶段控制 ──
    current_stage:    str
    stage_turn_count: int
    total_turn_count: int
    max_turns:        int

    # ── ④ 题目与追问 ──
    question_bank:      list[dict]
    current_question:   Optional[dict]
    projects_asked:     list[str]
    last_answer_quality: str
    followup_count:     int

    # ── ⑤ 记忆与输出 ──
    existing_summary:  Optional[str]
    should_summarize:  bool
    report:            Optional[dict]
    fallback_used:     bool
    structured_output: Optional[dict]
