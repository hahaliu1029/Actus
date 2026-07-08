"""R2 CS2 golden matrix fixture generator.

One-shot script that builds 22 canonical ``ToolArtifact`` JSON files
covering the full (``source`` × ``variant``) matrix that the R2 Day-7
merge gate requires:

- native (8):   allow_success, allow_error_exception, allow_error_timeout,
                denied_ast_validator, denied_approval_policy,
                asked_smart_approve, passthrough_image, passthrough_pdf
- mcp (5):      allow_success, allow_error_exception, allow_error_timeout,
                denied_approval_policy, asked_smart_approve
- a2a (4):      allow_success, allow_error_exception,
                denied_approval_policy, asked_smart_approve
- skill (5):    allow_success, allow_error_exception,
                denied_approval_policy, asked_risk_enforce,
                asked_approval_policy

Run this script **once** to regenerate fixtures after any deliberate
change to the ``ToolArtifact`` / ``ToolOutcome`` wire format. The
companion ``test_r2_golden_matrix.py`` then loads each JSON, validates
it via ``TOOL_ARTIFACT_ADAPTER``, and round-trips it back to bytes to
catch wire-format drift.

Execution:

    cd api && uv run python tests/golden/r2_tool_outcome_matrix/_generate_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    DocumentPreview,
    FileBlock,
    FilePayload,
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalPayload,
    Passthrough,
    ToolArtifact,
)
from app.domain.services.tools.tool_source_resolver import ToolSource


OUT = Path(__file__).parent


# Canonical tool sources used across the 22 fixtures.
NATIVE_SHELL = ToolSource(
    source="native", category="shell", canonical_name="shell_execute"
)
NATIVE_FILE = ToolSource(
    source="native", category="file", canonical_name="file_view"
)
MCP_SRC = ToolSource(
    source="mcp", category="mcp", canonical_name="mcp_slack_post"
)
A2A_SRC = ToolSource(
    source="a2a", category="a2a", canonical_name="call_remote_agent"
)
SKILL_SRC = ToolSource(
    source="skill", category="skill", canonical_name="skill_example"
)


def write(fname: str, artifact: ToolArtifact) -> None:
    (OUT / fname).write_text(
        json.dumps(
            artifact.model_dump(mode="json", by_alias=True),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


FIXTURES: list[tuple[str, ToolArtifact]] = [
    # -------------------- native (8) -------------------- #
    (
        "native_allow_success.json",
        ToolArtifact(
            tool_call_id="n_ok",
            tool_name="shell_execute",
            tool_source=NATIVE_SHELL,
            outcome=AllowSuccess(content="total 0\n"),
        ),
    ),
    (
        "native_allow_error_exception.json",
        ToolArtifact(
            tool_call_id="n_err",
            tool_name="shell_execute",
            tool_source=NATIVE_SHELL,
            outcome=AllowError(
                content="Permission denied",
                reason=DecisionReason(
                    type="exception", code="PermissionError"
                ),
            ),
        ),
    ),
    (
        "native_allow_error_timeout.json",
        ToolArtifact(
            tool_call_id="n_to",
            tool_name="shell_execute",
            tool_source=NATIVE_SHELL,
            outcome=AllowError(
                content="Command timeout",
                reason=DecisionReason(
                    type="timeout", code="sandbox_shell_timeout"
                ),
                retryable=True,
            ),
        ),
    ),
    (
        "native_denied_ast_validator.json",
        ToolArtifact(
            tool_call_id="n_ast",
            tool_name="shell_execute",
            tool_source=NATIVE_SHELL,
            outcome=Denied(
                content="AST validator blocked: rm -rf pattern detected",
                reason=DecisionReason(
                    type="ast_validator", code="dangerous_rm_pattern"
                ),
            ),
        ),
    ),
    (
        "native_denied_approval_policy.json",
        ToolArtifact(
            tool_call_id="n_dp",
            tool_name="shell_execute",
            tool_source=NATIVE_SHELL,
            outcome=Denied(
                content="Denied by user approval policy",
                reason=DecisionReason(
                    type="approval_policy", code="always_deny"
                ),
            ),
        ),
    ),
    (
        "native_asked_smart_approve.json",
        ToolArtifact(
            tool_call_id="n_sa",
            tool_name="shell_execute",
            tool_source=NATIVE_SHELL,
            outcome=Asked(
                content="SmartApprove flagged this command for review",
                reason=DecisionReason(
                    type="smart_approve", code="llm_flagged_high_risk"
                ),
            ),
        ),
    ),
    (
        "native_passthrough_image.json",
        ToolArtifact(
            tool_call_id="n_img",
            tool_name="file_view",
            tool_source=NATIVE_FILE,
            outcome=Passthrough(
                content="[file_view: file_view — 1 image(s) loaded]",
                data=MultimodalPayload(
                    blocks=[
                        ImageUrlBlock(
                            image_url=ImageUrlPayload(
                                url="data:image/png;base64,iVBOR..."
                            )
                        )
                    ]
                ),
            ),
        ),
    ),
    (
        "native_passthrough_image_mediatype.json",
        ToolArtifact(
            tool_call_id="n_img_mt",
            tool_name="file_view",
            tool_source=NATIVE_FILE,
            outcome=Passthrough(
                content="[file_view: file_view — 1 image(s) loaded]",
                data=MultimodalPayload(
                    blocks=[
                        ImageUrlBlock(image_url=ImageUrlPayload(
                            url="data:image/png;base64,iVBOR..."))
                    ],
                    media_type="image/png",
                ),
            ),
        ),
    ),
    (
        "native_passthrough_video_mediatype.json",
        ToolArtifact(
            tool_call_id="n_vid_mt",
            tool_name="file_view",
            tool_source=NATIVE_FILE,
            outcome=Passthrough(
                content="[file_view: file_view — 1 image(s) loaded]",
                data=MultimodalPayload(
                    blocks=[
                        ImageUrlBlock(image_url=ImageUrlPayload(
                            url="data:image/jpeg;base64,/9j/4AAQ..."))
                    ],
                    # B12 PR-1: video keyframe blocks are image_url (JPEG); payload
                    # media_type is the SOURCE video mime (document-level, mirrors
                    # pdf.py). Projector → render_style=image + media_type=video/mp4.
                    media_type="video/mp4",
                ),
            ),
        ),
    ),
    (
        "native_passthrough_pdf.json",
        ToolArtifact(
            tool_call_id="n_pdf",
            tool_name="file_view",
            tool_source=NATIVE_FILE,
            outcome=Passthrough(
                content="[file_view: file_view — 0 image(s) loaded]",
                data=MultimodalPayload(
                    blocks=[
                        FileBlock(
                            file=FilePayload(
                                filename="report.pdf",
                                file_data=(
                                    "data:application/pdf;base64,JVBERi..."
                                ),
                            )
                        )
                    ]
                ),
            ),
        ),
    ),
    (
        "native_passthrough_pdf_docpreview.json",
        ToolArtifact(
            tool_call_id="n_pdf_dp",
            tool_name="file_view",
            tool_source=NATIVE_FILE,
            outcome=Passthrough(
                content="[PDF: report.pdf, 3 pages]",
                data=MultimodalPayload(
                    blocks=[
                        FileBlock(file=FilePayload(
                            filename="report.pdf",
                            file_data="data:application/pdf;base64,JVBERi..."))
                    ],
                    document_preview=DocumentPreview(
                        filename="report.pdf", media_type="application/pdf", page_count=3),
                ),
            ),
        ),
    ),
    # -------------------- mcp (5) -------------------- #
    (
        "mcp_allow_success.json",
        ToolArtifact(
            tool_call_id="m_ok",
            tool_name="mcp_slack_post",
            tool_source=MCP_SRC,
            outcome=AllowSuccess(
                content="message sent",
                data={"message_id": "m123"},
            ),
        ),
    ),
    (
        "mcp_allow_error_exception.json",
        ToolArtifact(
            tool_call_id="m_err",
            tool_name="mcp_slack_post",
            tool_source=MCP_SRC,
            outcome=AllowError(
                content="MCP connection lost",
                reason=DecisionReason(
                    type="exception", code="ConnectionError"
                ),
                retryable=True,
            ),
        ),
    ),
    (
        "mcp_allow_error_timeout.json",
        ToolArtifact(
            tool_call_id="m_to",
            tool_name="mcp_slack_post",
            tool_source=MCP_SRC,
            outcome=AllowError(
                content="MCP call timeout",
                reason=DecisionReason(
                    type="timeout", code="mcp_client_timeout"
                ),
                retryable=True,
            ),
        ),
    ),
    (
        "mcp_denied_approval_policy.json",
        ToolArtifact(
            tool_call_id="m_dp",
            tool_name="mcp_slack_post",
            tool_source=MCP_SRC,
            outcome=Denied(
                content="User denied MCP tool",
                reason=DecisionReason(
                    type="approval_policy", code="session_deny"
                ),
            ),
        ),
    ),
    (
        "mcp_asked_smart_approve.json",
        ToolArtifact(
            tool_call_id="m_sa",
            tool_name="mcp_slack_post",
            tool_source=MCP_SRC,
            outcome=Asked(
                content="SmartApprove flagged cross-channel post for review",
                reason=DecisionReason(
                    type="smart_approve", code="llm_cross_channel"
                ),
            ),
        ),
    ),
    # -------------------- a2a (4) -------------------- #
    (
        "a2a_allow_success.json",
        ToolArtifact(
            tool_call_id="a_ok",
            tool_name="call_remote_agent",
            tool_source=A2A_SRC,
            outcome=AllowSuccess(
                content="Remote agent responded: task completed"
            ),
        ),
    ),
    (
        "a2a_allow_error_exception.json",
        ToolArtifact(
            tool_call_id="a_err",
            tool_name="call_remote_agent",
            tool_source=A2A_SRC,
            outcome=AllowError(
                content="Remote agent returned error: internal failure",
                reason=DecisionReason(
                    type="exception", code="a2a_remote_error"
                ),
            ),
        ),
    ),
    (
        "a2a_denied_approval_policy.json",
        ToolArtifact(
            tool_call_id="a_dp",
            tool_name="call_remote_agent",
            tool_source=A2A_SRC,
            outcome=Denied(
                content="A2A call denied by policy",
                reason=DecisionReason(
                    type="approval_policy", code="deny_cross_agent"
                ),
            ),
        ),
    ),
    (
        "a2a_asked_smart_approve.json",
        ToolArtifact(
            tool_call_id="a_sa",
            tool_name="call_remote_agent",
            tool_source=A2A_SRC,
            outcome=Asked(
                content="SmartApprove requires confirmation for A2A delegation",
                reason=DecisionReason(
                    type="smart_approve", code="llm_delegation_risk"
                ),
            ),
        ),
    ),
    # -------------------- skill (5) -------------------- #
    (
        "skill_allow_success.json",
        ToolArtifact(
            tool_call_id="s_ok",
            tool_name="skill_example",
            tool_source=SKILL_SRC,
            outcome=AllowSuccess(
                content="Skill ran successfully",
                data={"output": "value"},
            ),
        ),
    ),
    (
        "skill_allow_error_exception.json",
        ToolArtifact(
            tool_call_id="s_err",
            tool_name="skill_example",
            tool_source=SKILL_SRC,
            outcome=AllowError(
                content="Skill script failed: NameError",
                reason=DecisionReason(
                    type="exception", code="skill_runtime_error"
                ),
            ),
        ),
    ),
    (
        "skill_denied_approval_policy.json",
        ToolArtifact(
            tool_call_id="s_dp",
            tool_name="skill_example",
            tool_source=SKILL_SRC,
            outcome=Denied(
                content="Skill denied",
                reason=DecisionReason(
                    type="approval_policy", code="user_deny"
                ),
            ),
        ),
    ),
    (
        "skill_asked_risk_enforce.json",  # ★ R3 unlock evidence
        ToolArtifact(
            tool_call_id="s_re",
            tool_name="skill_example",
            tool_source=SKILL_SRC,
            outcome=Asked(
                content=(
                    "Skill 'skill_example' marked high risk — "
                    "awaiting user confirmation"
                ),
                reason=DecisionReason(
                    type="risk_enforce", code="skill_example_risk_high"
                ),
            ),
        ),
    ),
    (
        "skill_asked_approval_policy.json",
        ToolArtifact(
            tool_call_id="s_ap",
            tool_name="skill_example",
            tool_source=SKILL_SRC,
            outcome=Asked(
                content="Pending user confirmation",
                reason=DecisionReason(
                    type="approval_policy", code="pending_user_decision"
                ),
            ),
        ),
    ),
]


def main() -> None:
    assert len(FIXTURES) == 25, (
        f"Expected 25 fixtures, got {len(FIXTURES)} — update the FIXTURES "
        f"list in tests/golden/r2_tool_outcome_matrix/_generate_fixtures.py"
    )
    for fname, artifact in FIXTURES:
        write(fname, artifact)
    print(f"Wrote {len(FIXTURES)} golden fixtures to {OUT}")


if __name__ == "__main__":
    main()
