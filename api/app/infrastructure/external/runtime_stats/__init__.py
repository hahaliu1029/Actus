"""B9 运行时扩展统计（infrastructure 实现层）。

domain 端口在 ``app.domain.external.extension_stats``（P-9 钉子，无 Redis 概念）；
本包提供 Redis-backed 实现 ``RedisExtensionStats``（结构化满足
``ExtensionStatsRecorder`` / ``ExtensionStatsReader`` 两个 Protocol，不继承）。

**分层不变式（INV，结构门 tests/structure/test_b9_layering.py 锁死）**：
``app.application`` / ``app.domain`` 任何模块**禁止** import 本包——只有 ``app.main``
（lifespan 接线）与 ``app.interfaces.service_dependencies``（DI）可触达。
"""
