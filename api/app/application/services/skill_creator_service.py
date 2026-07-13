"""Skill Creator Service — AI 驱动 Skill 创建流水线。"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

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


class ManifestOutput(BaseModel):
    """Phase 1 输出：manifest + dependencies。仅用于生成管线内部。"""

    model_config = ConfigDict(extra="ignore")

    manifest: dict[str, Any] = Field(
        description='manifest.json 对象，必须包含 name, runtime_type("native"), '
        "tools 数组。每个 tool 必须有 name, description, parameters(object), "
        'required, entry(含 command: "python bundle/<name>.py")'
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="pip 包名列表（如 requests, beautifulsoup4），不要写中文描述",
    )


def _normalize_generated_manifest(
    manifest: dict[str, Any],
    blueprint: Any = None,
) -> dict[str, Any]:
    """归一化 LLM 生成的 manifest：补全缺失字段、runtime_type 和 entry.command。

    当 LLM 输出不完整时（缺少 name/tools），从 blueprint 回填。
    """
    manifest = dict(manifest)
    if "runtime_type" not in manifest:
        manifest["runtime_type"] = "native"

    # 从 blueprint 回填 name（当 LLM 输出缺失时）
    if not manifest.get("name") and blueprint is not None:
        skill_name = getattr(blueprint, "skill_name", None)
        if skill_name:
            manifest["name"] = skill_name

    # 从 blueprint 回填 description
    if not manifest.get("description") and blueprint is not None:
        desc = getattr(blueprint, "description", None)
        if desc:
            manifest["description"] = desc

    # 从 blueprint 回填 tools（当 LLM 完全未生成 tools 时）
    if not manifest.get("tools") and blueprint is not None:
        bp_tools = getattr(blueprint, "tools", None)
        if bp_tools:
            manifest["tools"] = [
                {
                    "name": t.name if hasattr(t, "name") else str(t),
                    "description": getattr(t, "description", "") or "",
                    "parameters": {
                        p.name: {"type": p.type or "string", "description": p.description or ""}
                        for p in (getattr(t, "parameters", None) or [])
                        if hasattr(p, "name")
                    } if hasattr(t, "parameters") else {},
                    "required": [
                        p.name for p in (getattr(t, "parameters", None) or [])
                        if hasattr(p, "name") and getattr(p, "required", False)
                    ] if hasattr(t, "parameters") else [],
                }
                for t in bp_tools
                if hasattr(t, "name")
            ]
            logger.warning(
                "LLM 未生成 tools，从 blueprint 回填 %d 个工具",
                len(manifest["tools"]),
            )

    for tool in manifest.get("tools", []):
        if isinstance(tool, dict):
            entry = tool.get("entry")
            if isinstance(entry, dict) and "command" not in entry:
                tool_name = tool.get("name", "tool")
                entry["command"] = f"python bundle/{tool_name}.py"
            elif entry is None:
                tool_name = tool.get("name", "tool")
                tool["entry"] = {"command": f"python bundle/{tool_name}.py"}
    return manifest


def _normalize_generated_dependencies(raw: Any) -> list[str]:
    """归一化 LLM 生成的 dependencies：dict/list-of-dicts → list[str]，过滤中文。"""
    if isinstance(raw, dict):
        extracted: list[str] = []
        for _key, val in raw.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        extracted.append(item)
            elif isinstance(val, str):
                extracted.append(val)
        return [d for d in extracted if not any('\u4e00' <= c <= '\u9fff' for c in d)]
    if isinstance(raw, list):
        result: list[str] = []
        for dep in raw:
            if isinstance(dep, str):
                if not any('\u4e00' <= c <= '\u9fff' for c in dep):
                    result.append(dep)
            elif isinstance(dep, dict):
                name = dep.get("name", "")
                if name and not any('\u4e00' <= c <= '\u9fff' for c in name):
                    result.append(name)
        return result
    return []

ANALYZE_SYSTEM_PROMPT = """\
你是 Skill 架构师。你的任务是：将用户的"创建 Skill"需求解析为结构化 JSON 蓝图。

⚠️ 重要区分：你在**设计 Skill 的架构蓝图**，而不是执行 Skill 描述的功能。
例如用户说"创建一个翻译工具 Skill"，你应该输出翻译工具的架构蓝图（包含工具名、参数定义等），
而不是执行翻译操作。用户消息中的示例数据（如 URL、文本）仅用于理解需求，不要把它们当作实际输入。

必须严格遵循以下 JSON 结构（注意字段名必须完全一致）：
{
  "skill_name": "english-kebab-case-name",
  "description": "该 Skill 的功能描述",
  "tools": [
    {
      "name": "tool_function_name",
      "description": "工具功能描述",
      "parameters": [
        {"name": "param1", "type": "string", "description": "参数说明", "required": true}
      ]
    }
  ],
  "search_keywords": ["keyword1", "keyword2"],
  "estimated_deps": ["requests", "beautifulsoup4"]
}

关键约束：
- 顶层字段必须是 skill_name（不是 name/skill/title），使用英文 kebab-case
- 直接返回 JSON 对象，不要嵌套在其他键下
- 仅返回上述结构的 JSON，不要返回其他文本或执行实际操作"""

MANIFEST_SYSTEM_PROMPT = """\
你是 Skill manifest 生成器。请根据需求蓝图和调研报告，生成 manifest.json 和 pip 依赖列表。

必须返回包含两个字段的 JSON：

### 1. manifest (object)
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
- 每个蓝图中的工具都必须有对应的 tool 条目

### 2. dependencies (array)
需要 pip install 的包名列表（如 ["requests", "beautifulsoup4"]）。
只写 pip 包名，不要写中文描述。

仅返回 JSON，不要返回其他文字。"""

SCRIPT_SYSTEM_PROMPT = """\
你是 Python 脚本生成器。请为指定的工具生成一个完整的 Python 脚本。

脚本要求：
1. 接受一个 JSON 字符串作为命令行参数: sys.argv[1]
2. 解析 JSON 输入，执行工具逻辑
3. 输出 JSON 格式结果到 stdout（使用 print(json.dumps(...))）
4. 支持 --help 参数（使用 argparse 或简单的 sys.argv 判断）
5. 优先使用调研报告中推荐的成熟开源库
6. 合理处理异常，出错时也输出 JSON 格式的错误信息

直接返回 Python 代码，不要包含任何解释性文字。
如果使用 markdown 代码块，请用 ```python 标记。"""

SKILL_MD_SYSTEM_PROMPT = """\
你是 SKILL.md 文档生成器。请根据已生成的 manifest 和脚本列表，生成完整的 SKILL.md 文件。

SKILL.md 格式必须严格如下：
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

## 功能说明
描述该 Skill 的用途和适用场景。

## 工具列表
列出每个工具的使用方式和参数说明。

## 使用示例
提供 1-2 个典型使用场景。

直接返回 SKILL.md 的完整内容，不要包含额外解释。"""


def _parse_llm_json(text: str) -> dict[str, Any]:
    """从 LLM 文本响应中提取 JSON 对象。

    处理常见格式：
    - 纯 JSON
    - ```json ... ``` markdown 代码块
    - 混合文本中的 JSON 对象
    """
    text = text.strip()

    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 markdown 代码块中的 JSON
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 找到第一个 { 到最后一个 } 之间的内容
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    # 4. 使用 json_repair 库兜底
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    raise ValueError(f"无法从 LLM 响应中提取 JSON（前 200 字符）: {text[:200]}")


def _extract_python_code(text: str) -> str:
    """从 LLM 响应中提取 Python 代码。

    优先级：```python 代码块 > ``` 通用代码块 > 原始文本（去掉首尾非代码行）。
    """
    text = text.strip()
    if not text:
        raise ValueError("LLM 响应为空，无法提取 Python 代码")

    # 1. 提取 ```python ... ``` 代码块
    py_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if py_match:
        return py_match.group(1).strip()

    # 2. 提取 ``` ... ``` 通用代码块
    generic_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    # 3. 原始文本：去掉首尾的非 Python 行
    lines = text.splitlines()
    code_pattern = re.compile(
        r"^(import |from |def |class |#|@|if |for |while |try:|except |return |"
        r"with |async |await |print\(|raise |    )"
    )
    start = None
    for i, line in enumerate(lines):
        if code_pattern.match(line):
            start = i
            break

    if start is None:
        raise ValueError(f"无法从 LLM 响应中提取 Python 代码（前 200 字符）: {text[:200]}")

    code_lines = lines[start:]
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    result = "\n".join(code_lines).strip()
    if not result:
        raise ValueError(f"提取到空的 Python 代码（前 200 字符）: {text[:200]}")
    return result


def _validate_generated_files(files: SkillGeneratedFiles) -> None:
    """校验 LLM 生成结果的结构完整性。

    检查 LLM 是否真正生成了可执行的 Skill 文件，而不是仅返回文档或 JSON 描述符。
    """
    errors: list[str] = []

    # 1. manifest 必须包含安装所需的关键字段
    if not files.manifest:
        errors.append("manifest 为空")
    else:
        for required_key in ("name", "runtime_type", "tools"):
            if required_key not in files.manifest:
                errors.append(f"manifest 缺少必填字段: {required_key}")
        manifest_tools = files.manifest.get("tools", [])
        if isinstance(manifest_tools, list) and manifest_tools:
            for i, tool in enumerate(manifest_tools):
                if isinstance(tool, dict) and not tool.get("entry", {}).get("command"):
                    errors.append(
                        f"manifest.tools[{i}]({tool.get('name', '?')}) "
                        f"缺少 entry.command"
                    )

    # 2. 必须有 bundle/ 下的 Python 脚本
    py_scripts = [s for s in files.scripts if s.path.endswith(".py")]
    bundle_scripts = [s for s in py_scripts if s.path.startswith("bundle/")]
    if not bundle_scripts:
        errors.append(
            f"缺少 bundle/*.py 可执行脚本（当前 scripts: "
            f"{[s.path for s in files.scripts] or '无'}）"
        )

    # 3. dependencies 应为 pip 包名，不应为中文描述
    non_pip_deps = [
        d for d in files.dependencies
        if any('\u4e00' <= c <= '\u9fff' for c in d)
    ]
    if non_pip_deps:
        errors.append(
            f"dependencies 应为 pip 包名，不应为中文描述: {non_pip_deps[:3]}"
        )

    # 4. skill_md 不应为空
    if not files.skill_md.strip():
        errors.append("skill_md 为空")

    if errors:
        raise ValueError(
            f"LLM 生成结果不符合 Skill 结构要求（{len(errors)} 个问题）：\n"
            + "\n".join(f"- {e}" for e in errors)
        )


_SAMPLE_VALUES = {
    "string": "test",
    "str": "test",
    "number": 1,
    "integer": 1,
    "int": 1,
    "float": 1.0,
    "boolean": True,
    "bool": True,
    "array": [],
    "list": [],
    "object": {},
    "dict": {},
}


def _build_sample_input(tool_def: dict[str, Any]) -> dict[str, Any]:
    """根据 manifest tool 定义构造最小测试 JSON 输入。"""
    params_raw = tool_def.get("parameters", {})
    required_raw = tool_def.get("required", [])
    required_names = set(required_raw) if isinstance(required_raw, list) else set()

    sample: dict = {}

    # parameters 可能是 {"param": {type, desc}} 或 [{name, type, desc, required}]
    if isinstance(params_raw, dict):
        for name, spec in params_raw.items():
            if not isinstance(spec, dict):
                continue
            param_type = str(spec.get("type", "string")).lower()
            sample[name] = _SAMPLE_VALUES.get(param_type, "test")
    elif isinstance(params_raw, list):
        for spec in params_raw:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name", "")
            if not name:
                continue
            param_type = str(spec.get("type", "string")).lower()
            sample[name] = _SAMPLE_VALUES.get(param_type, "test")

    return sample


class SkillCreatorService:
    """五步流水线：分析、调研、生成、验证、安装。"""

    def __init__(
        self,
        llm: BaseChatModel,
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
            try:
                files = await self._generate_files(blueprint, report)
            except Exception as exc:
                logger.exception("Skill 文件生成失败: %s", exc)
                yield SkillCreationProgress(
                    step="generating",
                    message="生成失败",
                    detail=str(exc),
                )
                return
            yield SkillCreationProgress(
                step="generating",
                message=f"生成完成，共 {len(files.scripts)} 个脚本",
            )

            yield SkillCreationProgress(step="validating", message="正在执行沙箱验证...")
            if sandbox is None:
                # C5c: harden the generated-code validation sandbox too (the more
                # security-sensitive path — it runs LLM-generated scripts). Flag OFF
                # → compile_runtime_policy returns None → create() byte-identical.
                from app.application.services.sandbox_runtime_policy import (
                    compile_runtime_policy,
                )
                from core.config import get_settings

                temp_sandbox = await DockerSandbox.create(
                    runtime_policy=compile_runtime_policy(get_settings()),
                )
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

    async def analyze(self, description: str) -> SkillBlueprint:
        """公共分析接口，供 brainstorm_skill 工具调用。"""
        return await self._analyze_requirement(description)

    async def generate(
        self,
        description: str,
        sandbox: Sandbox,
        blueprint: SkillBlueprint | None = None,
    ) -> AsyncGenerator[SkillCreationProgress | SkillGeneratedFiles, None]:
        """执行 RESEARCHING → GENERATING → VALIDATING，返回生成文件，不安装。"""
        if blueprint is None:
            yield SkillCreationProgress(step="analyzing", message="正在分析需求...")
            blueprint = await self._analyze_requirement(description)

        yield SkillCreationProgress(step="researching", message="正在调研 GitHub 方案...")
        repos = await self._research(blueprint)
        report = self._github.format_research_report(repos)
        yield SkillCreationProgress(
            step="researching",
            message=f"调研完成，找到 {len(repos)} 个参考仓库",
            references=repos or None,
        )

        yield SkillCreationProgress(step="generating", message="正在生成 Skill 文件...")
        try:
            files = await self._generate_files(blueprint, report)
        except Exception as exc:
            logger.exception("Skill 文件生成失败: %s", exc)
            yield SkillCreationProgress(
                step="generating",
                message="生成失败",
                detail=str(exc),
            )
            return
        yield SkillCreationProgress(
            step="generating",
            message=f"生成完成，共 {len(files.scripts)} 个脚本",
        )

        yield SkillCreationProgress(step="validating", message="正在执行沙箱验证...")
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
        yield files

    async def install(self, files: SkillGeneratedFiles, installed_by: str) -> SkillCreationResult:
        """公共安装接口，供 install_skill 工具调用。"""
        installed_skill = await self._install(files, installed_by)
        tool_names = [
            str(tool.get("name") or "")
            for tool in files.manifest.get("tools", [])
            if isinstance(tool, dict)
        ]
        return SkillCreationResult(
            skill_id=installed_skill.id,
            skill_name=installed_skill.name,
            tools=tool_names,
            files_count=len(files.scripts) + 2,
            summary=f"Skill '{installed_skill.name}' 创建成功",
        )

    async def _analyze_requirement(self, description: str) -> SkillBlueprint:
        messages = [
            SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
            HumanMessage(content=description),
        ]
        structured = self._llm.with_structured_output(SkillBlueprint)

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await structured.ainvoke(messages)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "蓝图分析失败 (attempt %d/3), %ds 后重试: %s",
                        attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _research(self, blueprint: SkillBlueprint) -> list[GitHubRepoInfo]:
        try:
            return await self._github.research_keywords(blueprint.search_keywords, top_n=3)
        except Exception as exc:
            logger.warning("GitHub 调研失败，降级为空结果: %s", exc)
            return []

    # ---- Phase 方法 -------------------------------------------------------- #

    async def _generate_manifest(
        self,
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> ManifestOutput:
        """Phase 1: 生成 manifest + dependencies。"""
        tool_payload = [tool.model_dump() for tool in blueprint.tools]
        prompt = (
            "## 需求蓝图\n"
            f"- 名称: {blueprint.skill_name}\n"
            f"- 描述: {blueprint.description}\n"
            f"- 工具: {json.dumps(tool_payload, ensure_ascii=False)}\n"
            f"- 预计依赖: {json.dumps(blueprint.estimated_deps, ensure_ascii=False)}\n\n"
            f"{research_report}"
        )
        messages = [
            SystemMessage(content=MANIFEST_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 先尝试 with_structured_output（3 次），全部失败后回退到 ainvoke + JSON 解析
        structured = self._llm.with_structured_output(ManifestOutput)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = await structured.ainvoke(messages)
                result.manifest = _normalize_generated_manifest(result.manifest, blueprint)
                result.dependencies = _normalize_generated_dependencies(result.dependencies)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "Phase 1 structured output 失败 (attempt %d/3), %ds 后重试: %s",
                        attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)

        # structured output 全部失败，回退到 ainvoke + JSON 解析（3 次）
        logger.warning("Phase 1 structured output 全部失败，回退到 ainvoke + JSON 解析")
        for attempt in range(3):
            try:
                response = await self._llm.ainvoke(
                    messages,
                    response_format={"type": "json_object"},
                )
                text = response.content if hasattr(response, "content") else str(response)
                parsed = _parse_llm_json(text)
                manifest = _normalize_generated_manifest(parsed.get("manifest", {}), blueprint)
                dependencies = _normalize_generated_dependencies(
                    parsed.get("dependencies", [])
                )
                return ManifestOutput(manifest=manifest, dependencies=dependencies)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "Phase 1 fallback 失败 (attempt %d/3), %ds 后重试: %s",
                        attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _generate_script(
        self,
        tool_def: dict,
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> ScriptFile:
        """Phase 2: 为单个工具生成 Python 脚本。"""
        tool_name = tool_def.get("name", "tool")
        params_desc = json.dumps(tool_def.get("parameters", {}), ensure_ascii=False)
        required_desc = json.dumps(tool_def.get("required", []), ensure_ascii=False)

        prompt = (
            f"## 工具信息\n"
            f"- 工具名: {tool_name}\n"
            f"- 描述: {tool_def.get('description', '')}\n"
            f"- 参数: {params_desc}\n"
            f"- 必填参数: {required_desc}\n\n"
            f"## 所属 Skill\n"
            f"- 名称: {blueprint.skill_name}\n"
            f"- 描述: {blueprint.description}\n\n"
            f"{research_report}\n\n"
            f"请生成 {tool_name} 的完整 Python 脚本。"
        )
        messages = [
            SystemMessage(content=SCRIPT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 提取脚本路径
        entry = tool_def.get("entry", {})
        command = entry.get("command", "") if isinstance(entry, dict) else ""
        if command.startswith("python "):
            script_path = command[len("python "):].strip()
        else:
            script_path = f"bundle/{tool_name}.py"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._llm.ainvoke(messages)
                text = response.content if hasattr(response, "content") else str(response)
                code = _extract_python_code(text)
                ast.parse(code)  # 语法校验
                return ScriptFile(path=script_path, content=code)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "Phase 2 脚本生成失败 (%s, attempt %d/3), %ds 后重试: %s",
                        tool_name, attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _generate_skill_md(
        self,
        manifest: dict[str, Any],
        scripts: list[ScriptFile],
        blueprint: SkillBlueprint,
    ) -> str:
        """Phase 3: 生成 SKILL.md 文档。"""
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        script_list = ", ".join(s.path for s in scripts)
        prompt = (
            f"## 已生成的 Manifest\n```json\n{manifest_json}\n```\n\n"
            f"## 脚本文件列表\n{script_list}\n\n"
            f"## Skill 描述\n{blueprint.description}\n\n"
            f"请生成完整的 SKILL.md 文件。"
        )
        messages = [
            SystemMessage(content=SKILL_MD_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._llm.ainvoke(messages)
                text = response.content if hasattr(response, "content") else str(response)
                text = text.strip()

                # 去掉 markdown 代码块包裹
                for fence in ("```yaml\n", "```markdown\n", "```\n"):
                    if text.startswith(fence):
                        text = text[len(fence):]
                        if text.endswith("```"):
                            text = text[:-3]
                        text = text.strip()
                        break

                if not text or "---" not in text:
                    raise ValueError(
                        f"SKILL.md 缺少 --- frontmatter（前 100 字符）: {text[:100]}"
                    )

                # 合并：标准 frontmatter（来自 Exporter）+ LLM body
                try:
                    from app.domain.services.skill_md_parser import SkillMdParser
                    from app.domain.services.skill_md_exporter import SkillMdExporter

                    llm_parse = SkillMdParser.parse(text)
                    # Derive slug from name if not in manifest
                    slug = manifest.get("slug") or re.sub(
                        r"[^a-z0-9]+", "-", manifest.get("name", "").lower()
                    ).strip("-")
                    standard_md = SkillMdExporter.export(
                        meta={
                            "name": manifest.get("name", ""),
                            "slug": slug,
                            "description": manifest.get("description", ""),
                            "version": manifest.get("version", "0.1.0"),
                            "runtime_type": manifest.get("runtime_type", "native"),
                        },
                        manifest={**manifest, "skill_md": llm_parse.body or ""},
                    )
                    text = standard_md
                except Exception as merge_exc:
                    logger.debug("SKILL.md frontmatter merge 失败，使用 LLM 原始输出: %s", merge_exc)

                return text
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2 * (2 ** attempt)
                    logger.warning(
                        "Phase 3 SKILL.md 生成失败 (attempt %d/3), %ds 后重试: %s",
                        attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _generate_files(
        self,
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> SkillGeneratedFiles:
        """三阶段生成 Skill 文件：manifest → 逐个脚本 → SKILL.md。"""
        # Phase 1: manifest + dependencies
        manifest_output = await self._generate_manifest(blueprint, research_report)
        manifest = manifest_output.manifest
        dependencies = manifest_output.dependencies

        # Phase 2: 逐个工具生成脚本
        scripts: list[ScriptFile] = []
        for tool_def in manifest.get("tools", []):
            if not isinstance(tool_def, dict):
                continue
            script = await self._generate_script(tool_def, blueprint, research_report)
            scripts.append(script)

        # Phase 3: SKILL.md
        skill_md = await self._generate_skill_md(manifest, scripts, blueprint)

        # 交叉校验：manifest 中每个工具都有对应脚本
        tool_commands = {
            t["entry"]["command"]
            for t in manifest.get("tools", [])
            if isinstance(t, dict) and isinstance(t.get("entry"), dict)
        }
        script_commands = {f"python {s.path}" for s in scripts}
        missing = tool_commands - script_commands
        if missing:
            logger.warning("Manifest 工具缺少对应脚本: %s", missing)

        files = SkillGeneratedFiles(
            skill_md=skill_md,
            manifest=manifest,
            scripts=scripts,
            dependencies=dependencies,
        )
        _validate_generated_files(files)
        return files

    async def _validate_in_sandbox(
        self,
        files: SkillGeneratedFiles,
        sandbox: Sandbox,
    ) -> list[str]:
        """沙箱端到端验证：语法 → 部署 → 依赖安装 → --help → 烟雾测试。

        烟雾测试会用 manifest 中每个工具的参数构造最小 JSON 输入，
        实际执行脚本并检查是否能跑通（输出 JSON / 不崩溃）。
        """
        errors: list[str] = []

        # ---- 1. 语法检查 ---- #
        for script in files.scripts:
            if not script.path.endswith(".py"):
                errors.append(
                    f"脚本 {script.path} 不是 Python 文件，"
                    "entry.command 要求 bundle/*.py 格式"
                )
                continue
            try:
                ast.parse(script.content)
            except SyntaxError as exc:
                errors.append(f"语法错误 ({script.path}): {exc}")
        if errors:
            return errors

        # ---- 2. 部署到沙箱 ---- #
        run_id = uuid.uuid4().hex[:8]
        session_id = f"skill_creator_validate_{run_id}"
        skill_root = f"/tmp/skill_creator_validate_{run_id}"

        # B5 PR-S1-7a (Q4 category a): runtime business path → logger.info
        # so the line goes through RedactingFormatter + LogRecord factory
        # and carries trace_id / request_id / session_id.
        logger.info(
            "[沙箱验证] 开始: scripts=%s, deps=%s, manifest_tools=%d",
            [s.path for s in files.scripts],
            files.dependencies,
            len(files.manifest.get("tools", [])),
        )

        mkdir_result = await sandbox.exec_command(
            session_id=session_id,
            exec_dir="/tmp",
            command=f"rm -rf {skill_root} && mkdir -p {skill_root}/bundle",
        )
        logger.info("[沙箱验证] mkdir: success={mkdir_result.success}, message={(mkdir_result.message or '')[:100]}")
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

        # ---- 3. 依赖安装 ---- #
        if files.dependencies:
            deps_str = " ".join(shlex.quote(dep) for dep in files.dependencies)
            install_cmd = f"python -m pip install {deps_str}"
            logger.info("[沙箱验证] pip install: {install_cmd}")
            install_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=install_cmd,
            )
            logger.info("[沙箱验证] pip install: success={install_result.success}, message={(install_result.message or '')[:200]}")
            if not install_result.success:
                errors.append(f"依赖安装失败: {install_result.message or ''}".strip())
                return errors

        # ---- 4. --help 测试 ---- #
        for script in files.scripts:
            help_cmd = f"python {shlex.quote(script.path)} --help"
            logger.info("[沙箱验证] --help: {help_cmd}")
            run_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=help_cmd,
            )
            logger.info("[沙箱验证] --help: success={run_result.success}, message={(run_result.message or '')[:200]}")
            if not run_result.success:
                errors.append(f"脚本 --help 失败 ({script.path}): {run_result.message or ''}".strip())
                return errors

        # ---- 5. 烟雾测试（仅警告，不阻断） ---- #
        # 烟雾测试用伪造的最小输入运行工具，对需要网络/API 的工具（如视频下载、
        # 翻译等）必然失败，因此仅记录日志不作为阻断条件。
        smoke_warnings = await self._smoke_test_tools(files, sandbox, session_id, skill_root)
        for warning in smoke_warnings:
            logger.warning("[烟雾测试] %s", warning)
            logger.info("[沙箱验证] 烟雾测试警告（不阻断）: {warning[:200]}")

        return errors

    async def _smoke_test_tools(
        self,
        files: SkillGeneratedFiles,
        sandbox: Sandbox,
        session_id: str,
        skill_root: str,
    ) -> list[str]:
        """对 manifest 中每个 tool 做实际运行烟雾测试。

        根据 tool 的 parameters 构造最小 JSON 输入，执行 entry.command，
        验证脚本能正常启动、处理输入并输出 JSON。
        """
        errors: list[str] = []
        manifest_tools = files.manifest.get("tools", [])
        if not manifest_tools:
            return errors

        for tool_def in manifest_tools:
            if not isinstance(tool_def, dict):
                continue
            tool_name = tool_def.get("name", "unknown")
            entry = tool_def.get("entry", {})
            command = entry.get("command", "") if isinstance(entry, dict) else ""
            if not command:
                continue

            # 构造最小测试输入
            sample_input = _build_sample_input(tool_def)
            sample_json = json.dumps(sample_input, ensure_ascii=False)

            # 执行: python bundle/tool.py '{"param": "test"}'
            full_cmd = f"{command} {shlex.quote(sample_json)}"
            logger.info("[沙箱验证] 烟雾测试 {tool_name}: {full_cmd}")
            run_result = await sandbox.exec_command(
                session_id=session_id,
                exec_dir=skill_root,
                command=full_cmd,
            )

            output = (run_result.message or "").strip()
            logger.info("[沙箱验证] 烟雾测试 {tool_name}: success={run_result.success}, output={output[:200]}")

            if not run_result.success:
                errors.append(
                    f"工具 {tool_name} 烟雾测试失败 "
                    f"(cmd: {command}):\n{output[:500]}"
                )
                continue

            # 检查 stdout 是否包含 JSON 输出
            if not output:
                errors.append(
                    f"工具 {tool_name} 烟雾测试无输出 "
                    f"(预期输出 JSON 到 stdout)"
                )
                continue

            # 尝试解析 JSON（允许输出中有前缀日志，取最后一行）
            json_found = False
            for line in reversed(output.splitlines()):
                line = line.strip()
                if line.startswith(("{", "[")):
                    try:
                        json.loads(line)
                        json_found = True
                        break
                    except json.JSONDecodeError:
                        pass
            if not json_found:
                errors.append(
                    f"工具 {tool_name} 烟雾测试输出非 JSON:\n{output[:300]}"
                )

        return errors

    async def _fix_files(
        self,
        files: SkillGeneratedFiles,
        errors: list[str],
        blueprint: SkillBlueprint,
        research_report: str,
    ) -> SkillGeneratedFiles:
        """智能修复：脚本错误只重生成坏脚本，manifest 错误全量重生成。"""
        fix_context = (
            research_report
            + "\n\n## 验证错误\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n请修复上述问题。"
        )

        # 分类错误
        script_errors = [
            e for e in errors
            if "脚本" in e or "bundle/" in e or "烟雾测试" in e or "语法错误" in e
        ]
        manifest_errors = [e for e in errors if "manifest" in e]

        if manifest_errors or not script_errors:
            # manifest 问题或无法分类 → 全量重生成
            return await self._generate_files(blueprint, fix_context)

        # 仅脚本错误 → 定向修复
        new_scripts = list(files.scripts)
        for error in script_errors:
            broken_path = self._extract_script_path_from_error(error)
            if not broken_path:
                return await self._generate_files(blueprint, fix_context)
            tool_def = self._find_tool_def_by_path(broken_path, files.manifest)
            if not tool_def:
                return await self._generate_files(blueprint, fix_context)
            new_script = await self._generate_script(tool_def, blueprint, fix_context)
            new_scripts = [
                new_script if s.path == broken_path else s
                for s in new_scripts
            ]

        return SkillGeneratedFiles(
            skill_md=files.skill_md,
            manifest=files.manifest,
            scripts=new_scripts,
            dependencies=files.dependencies,
        )

    @staticmethod
    def _extract_script_path_from_error(error: str) -> str | None:
        """从错误消息中提取 bundle/*.py 路径。"""
        match = re.search(r"(bundle/\S+\.py)", error)
        return match.group(1) if match else None

    @staticmethod
    def _find_tool_def_by_path(script_path: str, manifest: dict) -> dict | None:
        """根据脚本路径在 manifest 中找到对应的 tool 定义。"""
        target_command = f"python {script_path}"
        for tool in manifest.get("tools", []):
            if not isinstance(tool, dict):
                continue
            entry = tool.get("entry", {})
            if isinstance(entry, dict) and entry.get("command") == target_command:
                return tool
        return None

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
                trust_origin="agent_created",
                actor_id=installed_by,   # D1a §6.1-3：agent_created 会话用户即 actor
            )

