"""Shared fixtures and helpers for application service tests."""
from app.application.services.agent_service import _ConfigSnapshot
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig, SkillRiskPolicy
from app.domain.models.context_overflow_config import ContextOverflowConfig


def default_snapshot() -> _ConfigSnapshot:
    """Create a minimal _ConfigSnapshot for tests that don't care about config content."""
    return _ConfigSnapshot(
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
        overflow_config=ContextOverflowConfig(),
        summary_llm=None,
        vision_fallback_model=None,
        skill_creator_service=None,
        supports_vision=True,
        supports_pdf_input=False,
        file_understanding_config=None,
    )
