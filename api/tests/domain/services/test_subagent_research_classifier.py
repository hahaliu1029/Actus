"""Preflight classifier: blocks coding/shared-state prompts via static rules + LLM."""
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.domain.services.subagent_research_classifier import (
    SubagentResearchClassifier,
    ClassifierResult,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_llm():
    """LLM returns yes/no + reason per prompt."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def classifier(mock_llm):
    return SubagentResearchClassifier(llm=mock_llm)


async def test_static_block_coding_keywords(classifier, mock_llm):
    """Static keyword filter catches obvious coding requests without LLM call."""
    prompts = ["请帮我修改 user.py 文件中的 bug"]
    results = await classifier.classify_batch(prompts)

    assert len(results) == 1
    assert results[0].approved is False
    assert "code" in results[0].reason.lower() or "修改" in results[0].reason
    mock_llm.ainvoke.assert_not_called()


async def test_static_block_write_side_effects(classifier, mock_llm):
    """Block 'save', 'publish', 'deploy', 'delete' keywords."""
    prompts = ["发布这个版本到生产环境"]
    results = await classifier.classify_batch(prompts)
    assert results[0].approved is False


async def test_llm_classifies_research_prompts(classifier, mock_llm):
    """LLM call for prompts that don't hit static rules."""
    mock_llm.ainvoke.return_value = MagicMock(
        content="1. yes - independent research about LangChain multi-agent patterns"
    )
    prompts = ["研究 LangChain 的多智能体编排方案"]
    results = await classifier.classify_batch(prompts)
    assert results[0].approved is True
    mock_llm.ainvoke.assert_called_once()


async def test_batch_classifier_single_llm_call(classifier, mock_llm):
    """N prompts → 1 batched LLM call for cost efficiency."""
    mock_llm.ainvoke.return_value = MagicMock(
        content="1. yes - research X\n2. yes - research Y\n3. yes - research Z"
    )
    prompts = ["研究 X", "研究 Y", "研究 Z"]
    results = await classifier.classify_batch(prompts)

    assert len(results) == 3
    assert all(r.approved for r in results)
    mock_llm.ainvoke.assert_called_once()


async def test_classifier_llm_error_fail_open(classifier, mock_llm):
    """If classifier LLM fails, fail OPEN (allow) — runtime tool_filter is still gate."""
    mock_llm.ainvoke.side_effect = Exception("LLM down")
    prompts = ["研究 LangChain"]
    results = await classifier.classify_batch(prompts)

    assert results[0].approved is True
    assert "classifier_error" in results[0].reason.lower() or "fail_open" in results[0].reason.lower()
