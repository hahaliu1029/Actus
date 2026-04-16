"""Shared helper for running sandbox commands that may take > 5 seconds.

The sandbox exec_command API only waits 5 seconds for a command to complete.
If the command is still running, it returns status="running" with output=None.
File processors (PDF extraction, audio transcription, video keyframe extraction)
often take much longer. This helper polls wait_process + read_shell_output to
get the final result.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.domain.external.sandbox import SandboxHandle

logger = logging.getLogger(__name__)


async def exec_and_wait(
    sandbox: SandboxHandle,
    command: str,
    *,
    session_id: str | None = None,
    timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> dict:
    """Execute a command in the sandbox and wait for completion.

    Returns a dict with {"returncode": int, "output": str, "status": str}.
    Handles the sandbox's 5-second initial wait by polling until completion or timeout.
    """
    # Use a unique session ID per call to avoid conflicts with other sandbox commands.
    # Sandbox kills old processes when a new command arrives on the same session.
    import uuid
    if session_id is None:
        session_id = f"fv_{uuid.uuid4().hex[:8]}"

    result = await sandbox.exec_command(session_id, "", command)

    # Extract data from ToolResult
    data: dict = {}
    if hasattr(result, "data") and isinstance(result.data, dict):
        data = result.data
    else:
        return {"returncode": -1, "output": str(result), "status": "unknown"}

    status = data.get("status", "unknown")

    # If completed OR status not present (e.g. mock/old sandbox that doesn't
    # return status field — treat as completed if returncode is present)
    if status == "completed" or (status == "unknown" and data.get("returncode") is not None):
        return {
            "returncode": data.get("returncode") if data.get("returncode") is not None else -1,
            "output": data.get("output") or "",
            "status": "completed",
        }

    # Command still running — poll until done or timeout
    logger.info("exec_and_wait: command still running on session %s, polling...", session_id)
    elapsed = 0.0
    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        try:
            wait_result = await sandbox.wait_process(
                session_id=session_id, seconds=int(min(poll_interval, 5))
            )
            wait_data = {}
            if hasattr(wait_result, "data") and isinstance(wait_result.data, dict):
                wait_data = wait_result.data

            rc = wait_data.get("returncode")
            logger.debug("exec_and_wait: poll rc=%s elapsed=%.0fs session=%s", rc, elapsed, session_id)
            if rc is not None:
                # Process finished — read full output
                read_result = await sandbox.read_shell_output(session_id=session_id)
                output = ""
                if hasattr(read_result, "data") and isinstance(read_result.data, dict):
                    output = read_result.data.get("output") or ""
                elif hasattr(read_result, "message"):
                    output = read_result.message or ""
                return {
                    "returncode": rc,
                    "output": output,
                    "status": "completed",
                }
        except Exception as e:
            logger.debug("Poll wait_process error (will retry): %s", e)
            continue

    # Timeout — return whatever we have
    logger.warning("exec_and_wait timed out after %.0fs for command: %s", timeout, command[:100])
    try:
        read_result = await sandbox.read_shell_output(session_id=session_id)
        output = ""
        if hasattr(read_result, "data") and isinstance(read_result.data, dict):
            output = read_result.data.get("output") or ""
        return {"returncode": -1, "output": output, "status": "timeout"}
    except Exception:
        return {"returncode": -1, "output": "", "status": "timeout"}
