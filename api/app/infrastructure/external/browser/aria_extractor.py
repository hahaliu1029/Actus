"""ARIA snapshot 文本解析 + LLM 可见格式化。"""

import re
from typing import Final

from app.infrastructure.external.browser.snapshot_bundle import ElementDescriptor


# 仅这些 ARIA role 视为可交互；其他（heading/paragraph/generic 等）过滤掉。
_INTERACTIVE_ROLES: Final[frozenset[str]] = frozenset(
    {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "listbox",
        "option",
        "checkbox",
        "radio",
        "switch",
        "tab",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "slider",
        "spinbutton",
    }
)

# Tag 推断：ARIA role → HTML 标签近似映射（仅用于 LLM 显示，不影响 fresh resolve）
_ROLE_TO_TAG: Final[dict[str, str]] = {
    "button": "button",
    "link": "a",
    "textbox": "input",
    "searchbox": "input",
    "combobox": "select",
    "listbox": "select",
    "option": "option",
    "checkbox": "input",
    "radio": "input",
    "switch": "input",
    "tab": "button",
    "menuitem": "button",
    "menuitemcheckbox": "input",
    "menuitemradio": "input",
    "slider": "input",
    "spinbutton": "input",
}

# Descriptor line forms:
#   - role
#   - role "name"
#   - role: scalar
#   - role "name":
#   - role "name" [checked]              ← Playwright state attrs in brackets
#   - role "name" [checked] [disabled]   ← multiple bracket-attrs allowed
#   - role "name" [checked]:              ← bracket-attrs + trailing colon (children follow)
# The bracket-attr block carries state info (`checked`, `expanded`, `selected`,
# `disabled`, `pressed`, `level=N`, etc.). Phase 1 narrow does not surface that
# state to the descriptor — it just must not silently drop these lines.
_DESCRIPTOR_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^(?P<indent>\s*)-\s+
    (?P<role>[a-z]+)
    (?:\s+"(?P<name>[^"]*)")?         # optional quoted accessible name
    (?P<attrs>(?:\s+\[[^\]]*\])*)     # zero or more `[attr]` state blocks
    (?:\s*:(?:\s+(?P<scalar>.+))?)?   # optional trailing ':' with optional scalar value
    \s*$
    """,
    re.VERBOSE,
)

# Attribute child line: "- /placeholder: value" / "- /url: /x" / "- /checked"
_ATTR_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^\s*-\s+
    /(?P<attr>[a-z]+)
    (?:\s*:\s*(?P<value>.*?))?
    \s*$
    """,
    re.VERBOSE,
)


def parse_aria_snapshot(snapshot: str) -> list[ElementDescriptor]:
    """解析 ARIA snapshot 文本 → ElementDescriptor 列表。

    bbox 字段在解析阶段填 `(0,0,0,0)` 占位；调用方在拿到 page 后用
    `_resolve_bboxes_for_descriptors()` 异步填充真实 bbox。

    Caveat: `option` descriptors are emitted only when their parent
    combobox/listbox is open in the DOM at snapshot time. T11 downstream
    `get_by_role("option", name=...)` will fail if the dropdown closed
    between snapshot and click — caller is responsible for re-opening
    the parent before resolving an option.
    """
    intermediate: list[dict] = []
    last_idx = -1
    last_indent = -1

    for raw_line in snapshot.splitlines():
        if not raw_line.strip():
            continue

        attr_match = _ATTR_LINE_RE.match(raw_line)
        if attr_match:
            attr_indent = len(raw_line) - len(raw_line.lstrip())
            if last_idx >= 0 and attr_indent > last_indent:
                if attr_match.group("attr") == "placeholder":
                    intermediate[last_idx]["placeholder"] = (
                        attr_match.group("value") or ""
                    ).strip()
            continue

        desc_match = _DESCRIPTOR_RE.match(raw_line)
        if not desc_match:
            continue
        role = desc_match.group("role")
        if role not in _INTERACTIVE_ROLES:
            # Don't update last_idx/last_indent — keeps any subsequent attribute
            # children from attaching to a non-interactive parent. But also
            # reset so attributes don't leak from prior interactive descriptor.
            last_idx = -1
            last_indent = -1
            continue

        indent = len(desc_match.group("indent") or "")
        name = desc_match.group("name") or ""
        scalar = (desc_match.group("scalar") or "").strip()
        # text precedence: name > scalar > ""
        text = name if name else scalar

        intermediate.append(
            {
                "role": role,
                "name": name,
                "text": text,
                "placeholder": "",
            }
        )
        last_idx = len(intermediate) - 1
        last_indent = indent

    descriptors: list[ElementDescriptor] = []
    role_name_counter: dict[tuple[str, str], int] = {}
    for item in intermediate:
        role = item["role"]
        name = item["name"]
        key = (role, name)
        nth = role_name_counter.get(key, 0)
        role_name_counter[key] = nth + 1
        descriptors.append(
            ElementDescriptor(
                role=role,
                name=name,
                text=item["text"],
                tag=_ROLE_TO_TAG.get(role, role),
                placeholder=item["placeholder"],
                bbox=(0.0, 0.0, 0.0, 0.0),
                nth=nth,
            )
        )

    return descriptors


def format_descriptors_for_llm(descriptors: list[ElementDescriptor]) -> list[str]:
    """格式化为 LLM 看到的 `index:<tag>text</tag>` 字符串列表（外部接口零变更）。"""
    out: list[str] = []
    for idx, d in enumerate(descriptors):
        if d.tag == "input":
            display_text = (
                f"[Placeholder: {d.placeholder}]" if d.placeholder else d.text
            )
            out.append(f"{idx}:<{d.tag}>{display_text}</{d.tag}>")
        else:
            out.append(f"{idx}:<{d.tag}>{d.text}</{d.tag}>")
    return out
