"""Skill Creator Service — AI 驱动 Skill 创建流水线。"""

from __future__ import annotations

import ast
import json
import logging
import shlex
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.models.skill import SkillSourceType
from app.domain.models.skill_creator import (
    GitHubRepoInfo,
    ScriptFile,
    SkillBlueprint,
    SkillCreationProgress,
    SkillCreationResult,
    SkillGeneratedFiles,
)
from app.infrastructure.external.github_search_client import GitHubSearchClient

logger = logging.getLogger(__name__)

MAX_FIX_RETRIES = 2

ANALYZE_SYSTEM_PROMPT = """\
你是 Skill 架构师。请将用户自然语言需求解析为严格 JSON：
- skill_name
- description
- tools（name/description/parameters）
- search_keywords（2-4 组英文关键词）
- estimated_deps（Python 依赖名）
仅返回 JSON，不要返回其他文本。"""

GENERATE_SYSTEM_PROMPT = """\
你是 Skill 代码生成器。请根据需求蓝图和 GitHub 调研结果，生成可安装的 Native Skill。

必须返回严格 JSON，包含以下四个字段：

### 1. skill_md (string)
完整的 SKILL.md 内容，格式必须严格如下（注意 --- 分隔符）：
```
---
name: skill-name
version: "0.1.0"
description: 功能描述
runtime_type: native
tools:
  - name: tool_name
    description: 工具描述
    parameters:
      param1:
        type: string
        description: 参数描述
    required:
      - param1
    entry:
      command: python bundle/tool_name.py
activation: {}
policy:
  risk_level: low
---

# Skill 名称

使用说明文档...
```

### 2. manifest (object)
必须符合以下结构：
{
  "name": "skill-name",
  "slug": "skill-name",
  "version": "0.1.0",
  "description": "功能描述",
  "runtime_type": "native",
  "tools": [
    {
      "name": "tool_name",
      "description": "工具描述",
      "parameters": {
        "param1": {"type": "string", "description": "参数描述"}
      },
      "required": ["param1"],
      "entry": {
        "command": "python bundle/tool_name.py"
      }
    }
  ],
  "activation": {},
  "policy": {"risk_level": "low"},
  "security": {}
}
注意：
- parameters 必须是 object 格式（键为参数名，值为 {type, description}），不是数组
- entry.command 指向 bundle/ 下的脚本，格式为 "python bundle/<script>.py"
- runtime_type 必须为 "native"

### 3. scripts (array)
每个元素包含 path 和 content：
[{"path": "bundle/tool_name.py", "content": "...python code..."}]

脚本要求：
- 接受一个 JSON 字符串作为命令行参数: sys.argv[1]
- 输出 JSON 格式结果到 stdout
- 支持 --help 参数（用 argparse 或简单判断）
- 优先使用调研到的成熟开源库

### 4. dependencies (array)
需要 pip install 的包名列表。

只返回 JSON，不要返回其他文字。"""


class SkillCreatorService:
    """五步流水线：分析、调研、生成、验证、安装。"""

    def __init__(
        self,
        llm: LLM,
        github_client: GitHubSearchClient,
        skill_service,
    ) -> None:
        self._llm = llm
        self._github = github_client
        self._skill_service = skill_service

    async def create(
        self,
        description: str,
        sandbox: Sandbox | None = None,
        installed_by: str = "",
    ) -> AsyncGenerator[SkillCreationProgress | SkillCreationResult, None]:
        from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

        temp_sandbox: Sandbox | None = None

        try:
            yield SkillCreationProgress(step="analyzing", message="正在分析需求...")
            blueprint = await self._analyze_requirement(description)

            yield SkillCreationProgress(
                step="researching",
                message="正在调研 GitHub 方案...",
            )
            repos = await self._research(blueprint)
            report = self._github.format_research_report(repos)
            yield SkillCreationProgress(
                step="researching",
                message=f"调研完成，找到 {len(repos)} 个参考仓库",
                references=repos or None,
            )

            yield SkillCreationProgress(step="generating", message="正在生成 Skill 文件...")
            files = await self._generate_files(blueprint, report)
            yield SkillCreationProgress(
                step="generating",
                message=f"生成完成，共 {len(files.scripts)} 个脚本",
            )

            yield SkillCreationProgress(step="validating", message="正在执行沙箱验证...")
            if sandbox is None:
                temp_sandbox = await DockerSandbox.create()
                sandbox = temp_sandbox

            errors = await self._validate_in_sandbox(files, sandbox)
            fix_round = 0
            while errors and fix_round < MAX_FIX_RETRIES:
                yield SkillCreationProgress(
                    step="validating",
                    message=f"检测到问题，正在自动修复（第 {fix_round + 1} 次）...",
                    detail=errors[0],
                )
                files = await self._fix_files(files, errors, blueprint, report)
                errors = await self._validate_in_sandbox(files, sandbox)
                fix_round += 1

            if errors:
                yield SkillCreationProgress(
                    step="validating",
                    message="沙箱验证失败",
                    detail=errors[0],
                )
                return

            yield SkillCreationProgress(step="validating", message="沙箱验证通过")

            yield SkillCreationProgress(step="installing", message="正在安装 Skill...")
            installed_skill = await self._install(files, installed_by)
            tool_names = [
                str(tool.get("name") or "")
                for tool in files.manifest.get("tools", [])
                if isinstance(tool, dict)
            ]
            yield SkillCreationProgress(step="installing", message="安装完成")
            yield SkillCreationResult(
                skill_id=installed_skill.id,
                skill_name=installed_skill.name,
                tools=tool_names,
                files_count=len(files.scripts) + 2,
                summary=f"Skill '{installed_skill.name}' 创建成功",
            )
        finally:
            if temp_sandbox:
                try:
                    await temp_sandbox.destroy()
                except Exception as exc:
                    logger.warning("销毁临时沙箱失败: %s", exc)

    async def _analyze_requirement(self, description: str) -> SkillBlueprint:
        response = await self._llm.invoke(
            messages=[
                {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            response_format={"type": "json_object"},
        )
        payload = self._parse_llm_json(response)
        return SkillBlueprint.model_validate(payload)

    async def _research(self, blueprint: SkillBlueprint) -> list[GitHubRepoInfo]:
        try:
            return await self._github.research_keywords(blueprint.search_keywords, top_n=3)
        except Exception as exc:
            logger.warning("GitHub 调研失败，降级为空结果: %s", exc)
            return []

    async def _generate_files(
        self,
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> SkillGeneratedFiles:
        tool_payload = [tool.model_dump() for tool in blueprint.tools]
        prompt = (
            "## 需求蓝图\n"
            f"- 名称: {blueprint.skill_name}\n"
            f"- 描述: {blueprint.description}\n"
            f"- 工具: {json.dumps(tool_payload, ensure_ascii=False)}\n"
            f"- 依赖: {json.dumps(blueprint.estimated_deps, ensure_ascii=False)}\n\n"
            f"{research_report}"
        )
        response = await self._llm.invoke(
            messages=[
                {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        payload = self._parse_llm_json(response)
        return SkillGeneratedFiles(
            skill_md=str(payload.get("skill_md") or ""),
            manifest=dict(payload.get("manifest") or {}),
            scripts=[ScriptFile.model_validate(item) for item in payload.get("scripts", [])],
            dependencies=[
                str(dep).strip()
                for dep in payload.get("dependencies", [])
                if str(dep).strip()
            ],
        )

    async def _validate_in_sandbox(
        self,
        files: SkillGeneratedFiles,
        sandbox: Sandbox,
    ) -> list[str]:
        errors: list[str] = []

        for script in files.scripts:
            try:
                ast.parse(script.content)
            except SyntaxError as exc:
                errors.append(f"语法错误 ({script.path}): {exc}")
        if errors:
            return errors

        session_id = "skill_creator_validate"
        skill_root = "/tmp/skill_creator_validate"

        mkdir_result = await sandbox.exec_command(
            session_id=session_id,
            exec_dir="/tmp",
            command=f"mkdir -p {skill_root}/bundle",
        )
        if not mkdir_result.success:
            errors.append(f"创建验证目录失败: {mkdir_result.message or ''}".strip())
            return errors

        for script in files.scripts:
            write_result = await sandbox.write_file(
                filepath=f"{skill_root}/{script.path}",
                content=script.content,
            )
            if not write_result.success:
                errors.append(
                    f"写入脚本失败 ({script.path}): {write_result.message or ''}".strip()
                )
                return errors

        if files.dependencies:
            deps_str = " ".join(shlex.quote(dep) for dep in files.dependencies)
            install_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=f"python -m pip install {deps_str}",
            )
            if not install_result.success:
                errors.append(f"依赖安装失败: {install_result.message or ''}".strip())
                return errors

        for script in files.scripts:
            run_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=f"python {shlex.quote(script.path)} --help",
            )
            if not run_result.success:
                errors.append(f"脚本运行失败 ({script.path}): {run_result.message or ''}".strip())
                return errors

        return errors

    async def _fix_files(
        self,
        files: SkillGeneratedFiles,
        errors: list[str],
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> SkillGeneratedFiles:
        del files
        fix_patch = (
            "\n\n## 验证错误\n"
            + "\n".join(f"- {item}" for item in errors)
            + "\n请修复后返回完整 JSON。"
        )
        return await self._generate_files(blueprint, research_report + fix_patch)

    async def _install(self, files: SkillGeneratedFiles, installed_by: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir)

            (skill_dir / "SKILL.md").write_text(files.skill_md, encoding="utf-8")
            (skill_dir / "manifest.json").write_text(
                json.dumps(files.manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            for script in files.scripts:
                script_path = skill_dir / script.path
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(script.content, encoding="utf-8")

            return await self._skill_service.install_skill(
                source_type=SkillSourceType.LOCAL,
                source_ref=f"local:{skill_dir.as_posix()}",
                manifest=files.manifest,
                skill_md=files.skill_md,
                installed_by=installed_by,
            )

    @staticmethod
    def _parse_llm_json(response: dict) -> dict:
        content = response.get("content", "{}")
        if isinstance(content, dict):
            return content
        text = str(content or "{}").strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return json.loads(text or "{}")
