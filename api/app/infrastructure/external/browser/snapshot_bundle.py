# api/app/infrastructure/external/browser/snapshot_bundle.py
"""ElementDescriptor + SnapshotBundle 数据类，BR1 Phase 1 narrow 切共享数据结构。

`data-manus-id` 索引方案的替代物：服务器侧缓存 element 的 ARIA descriptor，
click/input/select 时通过 fresh resolve 重新定位，规避 SPA 重渲染与自定义组件失效。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementDescriptor:
    """ARIA-based element identity，缓存于 PlaywrightBrowser 服务器侧，供 fresh resolve 使用。"""

    role: str
    name: str
    text: str
    tag: str
    placeholder: str
    bbox: tuple[float, float, float, float]  # (x, y, width, height) CSS 像素
    nth: int


@dataclass(frozen=True)
class ViewportInfo:
    """视口尺寸 + 滚动偏移。Phase 1 坐标统一用 CSS 像素，不消费 device_pixel_ratio。"""

    width: int
    height: int
    scroll_x: float
    scroll_y: float
    device_pixel_ratio: float


@dataclass(frozen=True)
class SnapshotBundle:
    """每次浏览器交互前统一采集的页面快照证据集。"""

    aria_snapshot: str
    screenshot: bytes
    viewport: ViewportInfo
    page_url: str
    page_title: str
    timestamp: float
