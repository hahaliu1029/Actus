"""D1a Task 13 — D1-0 ①层行为半：lifespan off gate 构造器零调用（R1#21/#22）。

注入拓扑 AST 门（test_d1a_injection_topology.py）证明**结构**（main.py 透传 kwarg）；
本文件证明**行为**：`EXTENSION_GOVERNANCE_MODE` 缺省（off）下真实驱动 lifespan
启动段，断言治理 port 构造器**零调用** + 三 `app.state.extension_*_port` 均为 None。

harness 照抄 tests/interfaces/test_memory_embedding_startup.py：patch 掉重基础设施
（Redis/Postgres/MinIO/Alembic/CheckpointerPool），真实跑 `lifespan(app)`。治理段
（main.py DB-ready 后、首次 `_load_app_config` 之前）跑完后，用 `_load_app_config`
的 sentinel side_effect **确定性**停在治理段之后——避免驱动到需真 PG 的下游段。

false-green 防护：off 断言（ctor==[]）后，在同一 patch 作用域内直接构造
DbExtensionAdmissionPort，断言计数器确实自增 → 证明 patch 命中的是真符号
（否则 off 的「零调用」可能来自 patch 目标写错）。
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.main import app, lifespan

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _StopLifespan(Exception):
    """确定性哨兵：治理段之后第一处 `_load_app_config()` 抛出，停在治理段之后。"""


def _enter_infra_patches(stack: ExitStack) -> None:
    """重基础设施 no-op（照抄 test_memory_embedding_startup._enter_base_patches 的子集，
    覆盖治理段之前必经的 Redis/Postgres/MinIO/Alembic/CheckpointerPool）。"""
    stack.enter_context(
        patch("app.main.get_redis", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock()))
    )
    stack.enter_context(
        patch("app.main.get_postgres", return_value=MagicMock(
            init=AsyncMock(), shutdown=AsyncMock(), session_factory=MagicMock(),
        ))
    )
    stack.enter_context(
        patch("app.main.get_minio", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock()))
    )
    stack.enter_context(patch("app.main.command"))  # skip Alembic migrations
    stack.enter_context(
        patch("app.infrastructure.checkpointer_pool.CheckpointerPool", return_value=MagicMock(
            open=AsyncMock(), close=AsyncMock(), pool=MagicMock(),
        ))
    )


async def test_off_mode_constructs_no_governance_ports():
    """mode=off（缺省）→ 治理 port 构造器零调用 + 三 app.state port 均 None。"""
    from app.main import settings

    # 前置：缺省 off（本 epic flag default-OFF 不翻）。若被 env 污染则本测试无意义。
    assert settings.extension_governance_mode == "off"

    ctor_calls: list = []

    def _counting_init(self, *args, **kwargs) -> None:
        ctor_calls.append((args, kwargs))

    with ExitStack() as stack:
        _enter_infra_patches(stack)
        # 计治理 admission port 的构造次数（off 分支不 import 该类 → 恒零）。
        stack.enter_context(
            patch(
                "app.infrastructure.external.governance.db_extension_admission."
                "DbExtensionAdmissionPort.__init__",
                _counting_init,
            )
        )
        # 治理段之后第一处 `_load_app_config()`（main.py:362-363）抛哨兵，确定性停机。
        stack.enter_context(
            patch(
                "app.interfaces.service_dependencies._load_app_config",
                side_effect=_StopLifespan,
            )
        )
        with pytest.raises(_StopLifespan):
            async with lifespan(app):
                pass

        # ① 行为断言：off 分支未构造任何治理 admission port。
        assert ctor_calls == [], f"off 模式不得构造 admission port，实际={ctor_calls}"
        # ② 三 port 均被 off 分支显式置 None（INV-D1-0）。
        assert app.state.extension_admission_port is None
        assert app.state.extension_registry_write_port is None
        assert app.state.extension_registry_read_port is None

        # ③ false-green 防护：同一 patch 作用域内直接构造 → 计数器自增，证明 patch
        #    命中真符号（否则 ①的「零」可能是 patch 目标写错导致的假绿）。
        from app.infrastructure.external.governance.db_extension_admission import (
            DbExtensionAdmissionPort,
        )

        DbExtensionAdmissionPort(MagicMock(), mode="shadow")
        assert len(ctor_calls) == 1, "patch 未命中真实 DbExtensionAdmissionPort.__init__ 符号"


async def test_off_mode_emits_no_governance_mode_log(caplog):
    """#1：off = 零新增日志 I/O——治理 mode 行仅 mode≠off 才发（off 下不 emit）。"""
    import logging as _logging

    from app.main import settings

    assert settings.extension_governance_mode == "off"
    with ExitStack() as stack:
        _enter_infra_patches(stack)
        stack.enter_context(
            patch(
                "app.interfaces.service_dependencies._load_app_config",
                side_effect=_StopLifespan,
            )
        )
        with caplog.at_level(_logging.INFO):
            with pytest.raises(_StopLifespan):
                async with lifespan(app):
                    pass
    assert not any(
        "D1a extension governance" in r.getMessage() for r in caplog.records
    ), "off 模式不得发治理 mode 日志行（off=零新增日志 I/O）"
