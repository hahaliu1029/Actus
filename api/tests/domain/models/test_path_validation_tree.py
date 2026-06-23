# api/tests/domain/models/test_path_validation_tree.py
import pytest

from app.domain.models.path_validation import (
    CoordinatorPathContractError,
    tree_contains,
    validate_coordinator_tree_prefix,
)


class TestValidateCoordinatorTreePrefix:
    def test_single_segment_dir_accepted(self):
        # UNLIKE validate_coordinator_path, a single-segment dir is a valid
        # tree prefix (it names a directory, not a bare file).
        assert validate_coordinator_tree_prefix("workspace") == "workspace"

    def test_multi_segment_dir_accepted(self):
        assert validate_coordinator_tree_prefix("api/gen") == "api/gen"

    def test_dot_slash_canonicalized(self):
        assert validate_coordinator_tree_prefix("./workspace") == "workspace"

    def test_double_slash_canonicalized(self):
        assert validate_coordinator_tree_prefix("api//gen") == "api/gen"

    def test_absolute_under_root_stripped(self):
        assert validate_coordinator_tree_prefix("/home/ubuntu/workspace") == "workspace"

    def test_trailing_slash_canonicalized(self):
        assert validate_coordinator_tree_prefix("workspace/") == "workspace"

    def test_absolute_outside_root_rejected(self):
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_tree_prefix("/etc")

    def test_workspace_root_itself_rejected(self):
        # "/home/ubuntu" -> empty rel -> a tree prefix cannot be the root.
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_tree_prefix("/home/ubuntu")

    def test_empty_rejected(self):
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_tree_prefix("")

    def test_dot_alone_rejected(self):
        # "." normalizes to "" -> the root -> not a valid prefix.
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_tree_prefix(".")

    def test_traversal_rejected(self):
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_tree_prefix("../escape")


class TestTreeContains:
    def test_path_under_prefix(self):
        assert tree_contains("workspace", "workspace/foo.py") is True

    def test_path_deep_under_prefix(self):
        assert tree_contains("workspace", "workspace/sub/foo.py") is True

    def test_exact_equal_is_not_contained(self):
        # A tree prefix is a DIRECTORY; the prefix path itself is the dir,
        # not a file under it. ADD-only leases target files strictly inside.
        assert tree_contains("workspace", "workspace") is False

    def test_sibling_prefix_not_contained(self):
        # component-aware: workspace-foo is NOT under workspace.
        assert tree_contains("workspace", "workspace-foo/bar.py") is False

    def test_unrelated_not_contained(self):
        assert tree_contains("workspace", "api/foo.py") is False

    def test_prefix_under_prefix_overlap(self):
        # api/gen is contained by api (used by dispatch overlap detection).
        assert tree_contains("api", "api/gen/x.py") is True
        assert tree_contains("api", "api/gen") is True
