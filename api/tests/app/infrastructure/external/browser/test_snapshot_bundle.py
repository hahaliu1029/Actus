# api/tests/app/infrastructure/external/browser/test_snapshot_bundle.py
import pytest
from dataclasses import FrozenInstanceError

from app.infrastructure.external.browser.snapshot_bundle import (
    ElementDescriptor,
    ViewportInfo,
    SnapshotBundle,
)


def test_element_descriptor_is_frozen() -> None:
    desc = ElementDescriptor(
        role="button",
        name="提交",
        text="提交",
        tag="button",
        placeholder="",
        bbox=(10.0, 20.0, 100.0, 40.0),
        nth=0,
    )
    assert desc.role == "button"
    assert desc.bbox == (10.0, 20.0, 100.0, 40.0)
    with pytest.raises(FrozenInstanceError):
        desc.role = "link"  # type: ignore[misc]


def test_element_descriptor_bbox_is_4_tuple() -> None:
    desc = ElementDescriptor(
        role="textbox",
        name="搜索",
        text="",
        tag="input",
        placeholder="搜索",
        bbox=(0.0, 0.0, 200.0, 30.0),
        nth=0,
    )
    x, y, w, h = desc.bbox
    assert (x, y, w, h) == (0.0, 0.0, 200.0, 30.0)


def test_viewport_info_frozen() -> None:
    vp = ViewportInfo(
        width=1280,
        height=720,
        scroll_x=0.0,
        scroll_y=0.0,
        device_pixel_ratio=2.0,
    )
    assert vp.width == 1280
    with pytest.raises(FrozenInstanceError):
        vp.width = 1024  # type: ignore[misc]


def test_snapshot_bundle_frozen_and_carries_required_fields() -> None:
    vp = ViewportInfo(width=1280, height=720, scroll_x=0.0, scroll_y=0.0, device_pixel_ratio=1.0)
    bundle = SnapshotBundle(
        aria_snapshot="- button \"提交\"",
        screenshot=b"\x89PNG\r\n",
        viewport=vp,
        page_url="https://example.com/",
        page_title="Example",
        timestamp=1714896000.0,
    )
    assert bundle.page_url == "https://example.com/"
    assert bundle.viewport.width == 1280
    assert bundle.screenshot == b"\x89PNG\r\n"
    assert isinstance(bundle.screenshot, bytes)
    with pytest.raises(FrozenInstanceError):
        bundle.page_url = "https://other.com/"  # type: ignore[misc]
