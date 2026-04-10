"""B5 C9: ContextAssembler signature shim — effective_window vs context_window.

The pre-C9 API accepted ``context_window: int`` and internally subtracted
``reserved_output_tokens`` in ``_compute_budget``. The new API accepts
``effective_window: int`` (already net of system + reserved). The shim
lets both coexist until B5.5/B5.6 deletes the legacy parameter.

Rules:
- Exactly one of ``effective_window`` / ``context_window`` must be provided
- Passing both → ``effective_window`` wins
- Passing neither → ``ValueError``
- ``context_window`` path emits a ``DeprecationWarning`` but still works
- Both paths must produce the SAME budget when the inputs describe the
  same effective window
"""
from __future__ import annotations

import warnings

import pytest

from app.domain.services.graphs.context_assembler import ContextAssembler
from app.domain.services.graphs.token_estimator import TokenEstimator


def _estimator() -> TokenEstimator:
    return TokenEstimator(strategy="hybrid")


class TestContextAssemblerSignature:
    def test_effective_window_only_constructs_cleanly(self) -> None:
        a = ContextAssembler(
            estimator=_estimator(),
            effective_window=5000,
            reserved_output_tokens=1000,
            safety_factor=1.15,
        )
        # Internal budget = effective_window / safety_factor
        assert a._compute_budget() == int(5000 / 1.15)

    def test_context_window_only_emits_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a = ContextAssembler(
                estimator=_estimator(),
                context_window=8000,
                reserved_output_tokens=1000,
                safety_factor=1.15,
            )
            # At least one DeprecationWarning mentioning the shim
            deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert deprecation, "Expected a DeprecationWarning for context_window= usage"
            assert "context_window" in str(deprecation[0].message)
            assert "effective_window" in str(deprecation[0].message)

        # Deprecated path still works: internal budget should match the
        # pre-C9 formula (context_window - reserved) / safety_factor
        assert a._compute_budget() == int((8000 - 1000) / 1.15)

    def test_both_parameters_missing_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ContextAssembler(
                estimator=_estimator(),
                reserved_output_tokens=1000,
            )
        assert "effective_window" in str(exc_info.value)
        assert "context_window" in str(exc_info.value)

    def test_both_parameters_present_favors_effective_window(self) -> None:
        # effective_window=5000 should win over context_window=99999
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress in case shim still warns
            a = ContextAssembler(
                estimator=_estimator(),
                effective_window=5000,
                context_window=99999,
                reserved_output_tokens=1000,
                safety_factor=1.15,
            )
        assert a._compute_budget() == int(5000 / 1.15)
        # And it must NOT be the context_window-derived value
        assert a._compute_budget() != int((99999 - 1000) / 1.15)

    def test_both_paths_agree_when_inputs_describe_same_window(self) -> None:
        """If the caller passes ``context_window=9000`` with ``reserved=1000``,
        the deprecated shim internally yields ``effective = 9000 - 1000 = 8000``.
        A new-API caller passing ``effective_window=8000`` directly should
        land at the exact same ``_compute_budget()`` value.
        """
        new_api = ContextAssembler(
            estimator=_estimator(),
            effective_window=8000,
            reserved_output_tokens=1000,
            safety_factor=1.15,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = ContextAssembler(
                estimator=_estimator(),
                context_window=9000,
                reserved_output_tokens=1000,
                safety_factor=1.15,
            )
        assert new_api._compute_budget() == legacy._compute_budget()

    def test_effective_window_preserves_uses_shim_flag_false(self) -> None:
        """Internal flag for observability: new API path → _uses_shim=False."""
        a = ContextAssembler(
            estimator=_estimator(),
            effective_window=5000,
        )
        assert a._uses_shim is False

    def test_context_window_sets_uses_shim_true(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ContextAssembler(
                estimator=_estimator(),
                context_window=8000,
                reserved_output_tokens=1000,
            )
        assert a._uses_shim is True
