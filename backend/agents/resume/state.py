"""
state - 简历审查 Agent 工作流状态 & Pydantic 结构化输出模型

"""

# ── 标准库 / 第三方导入 ──
from typing import Annotated               # 类型注解元信息：给 TypedDict 字段打上 reducer 标记

from langchain_core.messages import BaseMessage   # LangChain 消息基类（HumanMessage / AIMessage 的父类）
from langgraph.graph.message import add_messages  # LangGraph 消息归并器：新消息追加到列表末尾而非覆盖
from pydantic import BaseModel, Field             # Pydantic v2：声明式数据校验 + LLM Function Calling Schema
from typing_extensions import TypedDict           # TypedDict：定义 LangGraph 工作流全局 State 的字段


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                   ① 结构化简历子模型（ResumeStructured 的零件）                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class EducationItem(BaseModel):
    """单条教育经历 —— LLM Function Calling 会按此 Schema 输出结构化 JSON"""
    school:   str = Field(description='学校名称')            # 如 "四川大学"
    major:    str = Field(description='专业名称')            # 如 "计算机科学与技术"
    degree:   str = Field(description='学历：本科/专科/硕士等')  # 如 "本科"
    duration: str = Field(description='在校时间，如 2020.09 - 2024.06')  # 统一格式 YYYY.MM
    gpa:      str = Field(default='', description='GPA 或成绩（可选）')   # 可选字段，没写就空字符串


class ProjectItem(BaseModel):
    """单条项目经历 —— 简历中最核心的评估素材"""
    name:        str       = Field(description='项目名称')          # 如 "电商秒杀系统"
    role:        str       = Field(description='担任角色，如：后端开发/全栈/负责人')  # 角色反映参与深度
    duration:    str       = Field(description='项目时间，如 2023.06 - 2023.12')
    tech_stack:  list[str] = Field(description='使用的技术栈列表，如 [Spring Boot, MySQL, Redis]')
    description: str       = Field(description='项目描述原文（保留原始表述）')  # 不改写，保留原汁原味
    highlights:  list[str] = Field(default_factory=list,
                                    description='量化亮点句子列表（含数字的句子）')
    # 量化亮点示例：["QPS 从 500 提升到 5000", "支撑 10 万 DAU"]


class WorkItem(BaseModel):
    """单条工作/实习经历"""
    company:     str       = Field(description='公司名称')          # 如 "阿里巴巴"
    position:    str       = Field(description='职位名称')          # 如 "Java 后端开发"
    duration:    str       = Field(description='工作时间')          # 用于时间线推理（判断项目是否属于该段工作）
    tech_stack:  list[str] = Field(default_factory=list, description='涉及技术栈')
    description: str       = Field(description='工作内容原文')


class ResumeStructured(BaseModel):
    """
    简历结构化提取的根模型 —— extract_structured_node（节点④）的输出结构。

    设计意图：
      PDF 原始文本（~4000 字）→ LLM Function Calling → 这个结构化 dict
      后续六个维度的评审不再看全文，而是结合这个结构化摘要 + 原文前 3000 字
    """
    # ── 基本信息 ──
    name:            str = Field(description='姓名')
    phone:           str = Field(default='', description='手机号')
    email:           str = Field(default='', description='邮箱')
    target_position: str = Field(default='', description='求职意向岗位')  # 用于匹配度评估

    # ── 教育经历（时间倒序，最近的在前面）──
    education: list[EducationItem] = Field(default_factory=list)

    # ── 技能 ──
    skills_raw:  str       = Field(default='', description='技能栏原始文本')      # 保留原文
    skills_list: list[str] = Field(default_factory=list, description='解析后的技术标签列表')  # 拆成标签用于匹配度计算

    # ── 项目经历（简历评审的核心评估点）──
    projects: list[ProjectItem] = Field(default_factory=list)

    # ── 工作/实习经历 ──
    work_experience: list[WorkItem] = Field(default_factory=list)

    # ── 其他补充信息 ──
    certificates: list[str] = Field(default_factory=list, description='证书列表')   # 如 ["CET-6", "AWS SAA"]
    self_intro:   str       = Field(default='', description='个人简介/自我评价原文')


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                   ② 六维度评审输出模型                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class DimensionScore(BaseModel):
    """
    单个维度的评审结果 —— run_six_dimensions_node（节点⑤）的输出。

    注意：dimension 和 weight 由代码层覆盖（SIX_DIMENSIONS 表），LLM 只需填 score/issues/suggestions。
    """
    dimension:   str       = Field(default='', description="维度名称（代码层覆盖，LLM 可留空）")
    score:       int       = Field(description='得分 0-100')          # LLM 按评分标准打分
    weight:      float     = Field(default=0.0, description='权重（代码层覆盖，LLM 可填 0）')
    issues:      list[str] = Field(default_factory=list, description='该维度问题列表')      # 每个问题一串
    suggestions: list[str] = Field(default_factory=list, description='改进建议列表')        # 与 issues 一一对应


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                   ③ 问题诊断输出模型                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class IssueItem(BaseModel):
    """
    单条诊断问题 —— diagnose_issues_node（节点⑥）的输出条目。

    每条问题要求：
      - 定位到简历的具体位置（如 "项目经历-电商系统-第2句"）
      - 给出可操作的修改建议（不要 "请改进" 这种空话）
      - 明确优先级（high → 直接影响通过率 / medium → 影响竞争力 / low → 锦上添花）
    """
    priority:    str = Field(description='优先级：high / medium / low')
    dimension:   str = Field(description='所属维度')                        # 对应 SIX_DIMENSIONS 中的 name
    description: str = Field(description='问题描述（1句话）')                 # 简洁明了
    location:    str = Field(description='问题在简历中的定位，如：项目经历-电商系统-第2句')
    suggestion:  str = Field(description='具体修改建议（可操作）')             # 最好给出修改示例


class IssueList(BaseModel):
    """
    IssueItem 的容器 —— LLM Function Calling 需要一个根对象来包装数组。
    直接让 LLM 输出 list 不被所有模型支持，所以包一层。
    """
    items: list[IssueItem]     # 问题清单，5-15 条


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                   ④ 整体评价输出模型                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class ResumeSummary(BaseModel):
    """
    简历整体评价 —— generate_summary_node（节点⑦）的输出。

    面向学员，语气正面 + 建设性，不打击也不过度夸奖。
    """
    highlights:       list[str] = Field(description='2-3 条核心亮点')          # 真实存在的优势
    core_improvements: list[str] = Field(description='2-3 条最重要的改进方向')  # 优先级 high 的问题转化
    overall_comment:  str       = Field(description='1-2 句综合评语')         # 客观概括整体水平
    fit_assessment:   str       = Field(description='对目标岗位的匹配度评估（1句话）')


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                   ⑤ LangGraph 工作流全局 State（共享内存）                      ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class ResumeState(TypedDict):
    """
    简历审查 Agent 的全局状态 —— 在 8 个节点间流转的共享字典。

    设计原则：
      - 每个字段有明确的产出节点，避免 "谁都可以改" 的混乱
      - 所有字段可选（通过 | None 或默认值），允许节点只更新自己负责的字段
      - messages 用 add_messages reducer → 追加而非覆盖
    """

    # ── 请求上下文（API 层传入，全程只读）──
    messages:        Annotated[list[BaseMessage], add_messages]  # 对话历史，add_messages = 追加到末尾
    student_id:      str                                         # 学员 ID
    tenant_id:       str                                         # 租户 ID（多租户隔离）
    review_id:       str                                         # 本次审查的唯一 ID（对应 DB 行）
    pdf_minio_path:  str                                         # PDF 在对象存储的路径（预留）
    pdf_local_path:  str                                         # PDF 本地路径（上传接口已写入）

    # ── 节点③ 产出：PDF 解析 ──
    raw_text:        str            # 纯文本（含 PAGE BREAK 分页标记）
    page_count:      int            # PDF 总页数

    # ── 节点④ 产出：结构化提取 ──
    structured:      dict | None    # ResumeStructured.model_dump() 的结果

    # ── 节点⑤ 产出：六维度评审 ──
    dimension_scores: list[dict]    # 6 个 DimensionScore.model_dump() 的列表
    weighted_score:   float         # 加权综合分 = Σ(维分 × 维权)

    # ── 节点⑥ 产出：问题诊断 ──
    issues: list[dict]              # IssueItem.model_dump() 的列表，按优先级排序

    # ── 节点⑦ 产出：整体评价 ──
    summary: dict | None            # ResumeSummary.model_dump() 的结果

    # ── 节点⑧ 写入的标记位 ──
    fallback_used:     bool         # 是否有节点触发了降级兜底（如 LLM 失败用默认值）
    structured_output: dict | None  # 最终写入 DB + 返回给 API 的完整结果
