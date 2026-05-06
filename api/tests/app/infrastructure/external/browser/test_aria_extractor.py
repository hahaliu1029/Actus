import pytest

from app.infrastructure.external.browser.aria_extractor import (
    parse_aria_snapshot,
    format_descriptors_for_llm,
)
from app.infrastructure.external.browser.snapshot_bundle import ElementDescriptor


# bbox 由 _resolve_bboxes_for_descriptors() 异步阶段填充；本测试只验证 parser 产出结构。
EMPTY_BBOX = (0.0, 0.0, 0.0, 0.0)


def test_parse_simple_button() -> None:
    snapshot = '- button "提交"'
    out = parse_aria_snapshot(snapshot)
    assert len(out) == 1
    assert out[0].role == "button"
    assert out[0].name == "提交"
    assert out[0].text == "提交"
    assert out[0].tag == "button"
    assert out[0].nth == 0


def test_parse_unlabeled_textboxes_share_empty_name_nth() -> None:
    """Multiple textboxes with no accessible name → keyed on (role, '') → nth=0,1.

    Pins the (role, name) keying invariant so a future refactor that
    folds `placeholder` into the key can't silently break T11 fresh-resolve.
    """
    snapshot = (
        '- textbox "":\n'
        '  - /placeholder: Email\n'
        '- textbox "":\n'
        '  - /placeholder: Password'
    )
    out = parse_aria_snapshot(snapshot)
    assert len(out) == 2
    assert [(d.role, d.name, d.nth) for d in out] == [
        ("textbox", "", 0),
        ("textbox", "", 1),
    ]
    assert out[0].placeholder == "Email"
    assert out[1].placeholder == "Password"


def test_role_to_tag_covers_every_interactive_role() -> None:
    """Guard against `_INTERACTIVE_ROLES` and `_ROLE_TO_TAG` drifting apart."""
    from app.infrastructure.external.browser import aria_extractor

    missing = aria_extractor._INTERACTIVE_ROLES - aria_extractor._ROLE_TO_TAG.keys()
    assert missing == frozenset(), f"Roles missing from _ROLE_TO_TAG: {missing}"


def test_parse_handles_textbox_with_placeholder() -> None:
    """Playwright emits placeholder as a child '- /placeholder: value' line, not bracket-attr."""
    snapshot = (
        '- textbox "搜索":\n'
        '  - /placeholder: 搜索关键词'
    )
    out = parse_aria_snapshot(snapshot)
    assert len(out) == 1
    assert out[0].role == "textbox"
    assert out[0].name == "搜索"
    assert out[0].placeholder == "搜索关键词"


def test_parse_skips_non_interactive_roles() -> None:
    """heading / paragraph / generic 等非交互 role 应被过滤。"""
    snapshot = (
        '- heading "页面标题"\n'
        '- paragraph "说明文本"\n'
        '- button "操作"'
    )
    out = parse_aria_snapshot(snapshot)
    assert len(out) == 1
    assert out[0].role == "button"


def test_parse_assigns_nth_for_duplicate_role_name() -> None:
    """相同 role+name 的多个元素分配递增 nth。"""
    snapshot = (
        '- button "确认"\n'
        '- button "确认"\n'
        '- button "确认"'
    )
    out = parse_aria_snapshot(snapshot)
    assert [d.nth for d in out] == [0, 1, 2]
    assert all(d.role == "button" and d.name == "确认" for d in out)


def test_parse_independent_nth_for_different_role_name() -> None:
    snapshot = (
        '- button "提交"\n'
        '- button "取消"\n'
        '- button "提交"'
    )
    out = parse_aria_snapshot(snapshot)
    nths = [(d.role, d.name, d.nth) for d in out]
    assert nths == [("button", "提交", 0), ("button", "取消", 0), ("button", "提交", 1)]


def test_parse_empty_snapshot_returns_empty_list() -> None:
    assert parse_aria_snapshot("") == []
    assert parse_aria_snapshot("\n\n  \n") == []


def test_format_descriptors_produces_index_tag_text_strings() -> None:
    descs = [
        ElementDescriptor(role="button", name="提交", text="提交", tag="button", placeholder="", bbox=EMPTY_BBOX, nth=0),
        ElementDescriptor(role="textbox", name="搜索", text="", tag="input", placeholder="搜索关键词", bbox=EMPTY_BBOX, nth=0),
        ElementDescriptor(role="combobox", name="语言", text="语言", tag="select", placeholder="", bbox=EMPTY_BBOX, nth=0),
    ]
    out = format_descriptors_for_llm(descs)
    assert out == [
        "0:<button>提交</button>",
        "1:<input>[Placeholder: 搜索关键词]</input>",
        "2:<select>语言</select>",
    ]


def test_format_descriptors_handles_empty_list() -> None:
    assert format_descriptors_for_llm([]) == []


def test_parse_textbox_scalar_form_treats_value_as_text_not_name() -> None:
    """Playwright emits 'role: scalar' for textbox WITH a value but no accessible name.

    The scalar must NOT become `name` (would break get_by_role(name=value) resolution).
    It becomes `text` only; `name` stays empty.
    """
    snapshot = "- textbox: Enter your name"
    out = parse_aria_snapshot(snapshot)
    assert len(out) == 1
    assert out[0].role == "textbox"
    assert out[0].name == ""
    assert out[0].text == "Enter your name"


def test_parse_link_with_url_child_skips_url_attribute_line() -> None:
    """`- /url: /x` is a Playwright attribute-child, NOT a descriptor — must be skipped."""
    snapshot = (
        '- link "Read more":\n'
        '  - /url: /x'
    )
    out = parse_aria_snapshot(snapshot)
    assert len(out) == 1
    assert out[0].role == "link"
    assert out[0].name == "Read more"


def test_parse_full_real_playwright_fixture() -> None:
    """End-to-end fixture from real Playwright 1.57 aria_snapshot() output.

    Mirrors codex's local Playwright run (button + 3 textboxes + link).
    """
    snapshot = (
        '- button "Submit"\n'
        '- textbox: Enter your name\n'
        '- textbox "Email"\n'
        '- textbox "Search":\n'
        '  - /placeholder: keyword\n'
        '- link "Read more":\n'
        '  - /url: /x'
    )
    out = parse_aria_snapshot(snapshot)
    assert [(d.role, d.name, d.text, d.placeholder, d.nth) for d in out] == [
        ("button", "Submit", "Submit", "", 0),
        ("textbox", "", "Enter your name", "", 0),
        ("textbox", "Email", "Email", "", 0),
        ("textbox", "Search", "Search", "keyword", 0),
        ("link", "Read more", "Read more", "", 0),
    ]


def test_parse_handles_bracket_attr_state_lines() -> None:
    """Real Playwright emits state via bracket attrs:
        - checkbox "Agree" [checked]
        - button "Menu" [expanded]
        - button "Disabled" [disabled]
    These lines must produce ElementDescriptors. Codex audit P1 #1 caught the
    parser silently dropping them — interactive elements with state would
    disappear from the LLM-visible list, breaking common stateful controls.
    """
    snapshot = (
        '- checkbox "Agree" [checked]\n'
        '- button "Menu" [expanded]\n'
        '- button "Disabled" [disabled]'
    )
    out = parse_aria_snapshot(snapshot)
    assert [(d.role, d.name) for d in out] == [
        ("checkbox", "Agree"),
        ("button", "Menu"),
        ("button", "Disabled"),
    ]


def test_parse_handles_multiple_bracket_attrs_and_trailing_colon() -> None:
    """Multiple bracket-attrs and trailing colon (parent with children) coexist."""
    snapshot = (
        '- checkbox "Toggle" [checked] [disabled]\n'
        '- button "Open" [expanded]:\n'
        '  - /placeholder: extra'
    )
    out = parse_aria_snapshot(snapshot)
    assert [(d.role, d.name) for d in out] == [
        ("checkbox", "Toggle"),
        ("button", "Open"),
    ]
