"""C7 PR7 — INV-C7-1..8 测试资产清单锁（防收尾丢测试文件；Acceptance→INV→PR 矩阵 §12）。"""
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

# INV → 承载测试文件（存在性 + 非空锁；具体断言在各文件内）
INV_MANIFEST = {
    "INV-C7-1": ["domain/models/test_lifecycle_contract.py", "domain/models/test_lifecycle_event_union.py"],
    "INV-C7-2": ["invariants/test_inv_c7_2_lifecycle_single_constructor.py"],
    "INV-C7-3": ["domain/services/test_agent_task_runner_lifecycle_hook.py",
                 "domain/services/test_lifecycle_flag_off_zero_diff.py"],
    "INV-C7-4": ["interfaces/schemas/test_event_mapper_wire_snapshot.py"],
    "INV-C7-5": ["domain/services/test_agent_task_runner_lifecycle_subagent_dedup.py"],
    "INV-C7-6": ["domain/services/test_lifecycle_pair_audit.py",
                 "infrastructure/test_lifecycle_recovery_replay.py"],
    "INV-C7-7": [],  # 前端 reducer 侧：ui/src/lib/lifecycle/__tests__/reducer.test.ts（见下）
    "INV-C7-8": ["domain/models/test_lifecycle_event_union.py",
                 "domain/services/test_agent_task_runner_lifecycle_retried.py"],
}

FE_TESTS = [
    "ui/src/lib/lifecycle/__tests__/reducer.test.ts",       # INV-C7-7 + 规则 0-5
    "ui/src/lib/lifecycle/__tests__/wire-contract.test.ts",
    "ui/src/lib/store/__tests__/lifecycle-store.test.ts",
]


def test_backend_inv_test_files_exist_and_nonempty():
    missing = []
    for inv, files in INV_MANIFEST.items():
        for rel in files:
            p = TESTS / rel
            if not p.exists() or p.stat().st_size == 0:
                missing.append(f"{inv}: {rel}")
    assert missing == [], f"INV test assets missing: {missing}"


def test_frontend_inv_test_files_exist():
    repo_root = TESTS.parents[1]
    missing = [rel for rel in FE_TESTS if not (repo_root / rel).exists()]
    assert missing == [], f"FE INV test assets missing: {missing}"
