from typing import List, Literal

from pydantic import BaseModel, Field

SkillConfirmationAction = Literal[
    "generate", "revise", "install", "cancel", "regenerate", "retry"
]


class Message(BaseModel):
    """用户传递的消息。

    ``language`` 是 bootstrap hint（用户期望工作语言），不是 source of truth。
    由 ``AgentService._create_task`` 从 session 历史的最近 ``PlanEvent.plan.language``
    派生，经 ``AgentTaskRunner.__init__(initial_language=...)`` + ``self._current_language``
    传到此字段。一旦 planner_node 解析出新的 ``plan.language``，后者通过
    ``language_callback → set_language()`` 接管为 authoritative。
    默认 "zh" 用于全新 session 首条消息或无 PlanEvent 历史场景。
    """

    message: str = ""  # 用户发送的消息
    attachments: List[str] = Field(default_factory=list)  # 用户发送的附件
    image_content_blocks: List[dict] = Field(default_factory=list)  # 图片附件的多模态内容块
    skill_confirmation_action: SkillConfirmationAction | None = None
    language: str = "zh"  # B5 #29: bootstrap hint (see docstring)
    # [S4 §7] per-run team selection (transient; no Session column). Default None
    # ⇒ INV-0. Message has no explicit model_config (inherits extra="ignore"),
    # so this additive field is regression-safe.
    team_slug: str | None = None
