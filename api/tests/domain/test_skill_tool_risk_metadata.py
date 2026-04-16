"""CI enforcement: AST scan ensures skill tools carry risk metadata"""
import ast
from pathlib import Path


def test_langchain_dynamic_skill_tools_sets_risk_level():
    """Confirm create_dynamic_skill_langchain_tools sets metadata['risk_level']"""
    source_path = Path("app/domain/services/tools/langchain_dynamic_skill_tools.py")
    source = source_path.read_text()
    assert "risk_level" in source
    assert "metadata" in source


def test_skill_tool_no_evaluate_risk_enforce():
    """Confirm SkillTool no longer has _evaluate_risk_enforce"""
    source_path = Path("app/domain/services/tools/skill.py")
    source = source_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.name != "_evaluate_risk_enforce", \
                f"_evaluate_risk_enforce still exists at line {node.lineno}"
