"""M2-PR1: tests for eval harness path resolution.

Covers:
- public dataset path resolves to a real file under the repo
- private dir defaults to ``~/.actus/eval/memory_gate/real/`` when env
  var absent
- ``ACTUS_EVAL_DATA_DIR`` override wins when set
- ``private_available()`` reports False when the file is absent (the
  CI / contributor case)
"""
from __future__ import annotations

from pathlib import Path

from tests.eval.memory_gate.paths import (
    _DEFAULT_PRIVATE_ROOT,
    _ENV_VAR,
    control_set_available,
    control_set_path,
    private_available,
    private_dataset_path,
    private_dir,
    resolve_private_root,
    synthetic_dataset_path,
    synthetic_dir,
)


class TestPublicDataset:
    def test_synthetic_dir_points_to_repo_subdir(self) -> None:
        d = synthetic_dir()
        assert d.name == "synthetic"
        assert d.parent.name == "memory_gate"
        assert d.is_dir(), f"synthetic dir must exist at {d}"

    def test_synthetic_dataset_file_exists(self) -> None:
        """M1 shipped with a 25-sample dataset; it must still be there."""
        p = synthetic_dataset_path()
        assert p.is_file(), (
            f"synthetic/dataset.jsonl missing at {p} — M1 regressed?"
        )
        # Verify it's non-empty (>= 10 samples, defensive lower bound)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 10, (
            f"synthetic dataset shrunk unexpectedly: {len(lines)} lines"
        )


class TestPrivateRootResolution:
    def test_default_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(_ENV_VAR, raising=False)
        root = resolve_private_root()
        # Default expands ~/.actus/eval
        expected = _DEFAULT_PRIVATE_ROOT.expanduser().resolve()
        assert root == expected

    def test_env_override_wins(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv(_ENV_VAR, str(tmp_path))
        root = resolve_private_root()
        assert root == tmp_path.resolve()

    def test_env_override_expands_tilde(self, monkeypatch) -> None:
        """Bash-style ``~`` in the override is respected."""
        monkeypatch.setenv(_ENV_VAR, "~/some-eval-dir")
        root = resolve_private_root()
        assert str(root).startswith(str(Path.home()))
        assert root.name == "some-eval-dir"

    def test_empty_env_falls_back_to_default(self, monkeypatch) -> None:
        """Empty string is treated as unset (common shell confusion)."""
        monkeypatch.setenv(_ENV_VAR, "")
        root = resolve_private_root()
        assert root == _DEFAULT_PRIVATE_ROOT.expanduser().resolve()


class TestPrivateDirStructure:
    def test_private_dir_nests_suite_under_root(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv(_ENV_VAR, str(tmp_path))
        d = private_dir()
        # <root>/memory_gate/real/
        assert d == tmp_path.resolve() / "memory_gate" / "real"

    def test_private_dataset_path_and_control_set_path(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(_ENV_VAR, str(tmp_path))
        ds = private_dataset_path()
        cs = control_set_path()
        assert ds == tmp_path.resolve() / "memory_gate" / "real" / "dataset.jsonl"
        assert cs == tmp_path.resolve() / "memory_gate" / "real" / "control_set.jsonl"


class TestAvailabilityChecks:
    def test_private_available_is_false_when_dir_missing(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """CI / contributor machine: private dir doesn't exist → skip signal."""
        monkeypatch.setenv(_ENV_VAR, str(tmp_path / "does-not-exist"))
        assert private_available() is False
        assert control_set_available() is False

    def test_private_available_is_true_after_creating_file(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(_ENV_VAR, str(tmp_path))
        ds = private_dataset_path()
        ds.parent.mkdir(parents=True, exist_ok=True)
        ds.write_text('{"id": "p1", "text": "stub"}\n')
        assert private_available() is True

    def test_control_set_available_independent_of_dataset(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Control set and dataset availability are tracked separately —
        having a dataset doesn't imply the control set has been carved out."""
        monkeypatch.setenv(_ENV_VAR, str(tmp_path))
        ds = private_dataset_path()
        ds.parent.mkdir(parents=True, exist_ok=True)
        ds.write_text('{"id": "p1"}\n')
        assert private_available() is True
        # Control set not created yet
        assert control_set_available() is False
