import importlib.util
import json
import sys
from pathlib import Path


def _load_checker_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "tools" / "check_function_coverage.py"
    )
    spec = importlib.util.spec_from_file_location("check_function_coverage", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_function_collector_tracks_first_body_line_for_decorated_functions(tmp_path):
    checker = _load_checker_module()
    source = "\n".join(
        [
            "@decorator",
            "def foo():",
            "    '''doc'''",
            "    return 1",
            "",
        ]
    )
    tree = checker.ast.parse(source)
    collector = checker.FunctionCollector(Path("alphapulldown/example.py"))

    collector.visit(tree)

    function = collector.functions[0]
    assert function.lineno == 2
    assert function.body_lineno == 3
    assert function.end_lineno == 4


def test_check_function_coverage_ignores_definition_only_execution(
    tmp_path,
    monkeypatch,
    capsys,
):
    checker = _load_checker_module()
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "files": {
                    "alphapulldown/example.py": {
                        "executed_lines": [10],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checker,
        "iter_package_functions",
        lambda: [
            checker.FunctionSpan(
                path=Path("alphapulldown/example.py"),
                qualname="foo",
                lineno=10,
                body_lineno=11,
                end_lineno=12,
            )
        ],
    )

    exit_code = checker.check_function_coverage(coverage_json)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "alphapulldown/example.py:10 foo" in captured.out


def test_check_function_coverage_accepts_executed_function_body(tmp_path, monkeypatch, capsys):
    checker = _load_checker_module()
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "files": {
                    "alphapulldown/example.py": {
                        "executed_lines": [11],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checker,
        "iter_package_functions",
        lambda: [
            checker.FunctionSpan(
                path=Path("alphapulldown/example.py"),
                qualname="foo",
                lineno=10,
                body_lineno=11,
                end_lineno=12,
            )
        ],
    )

    exit_code = checker.check_function_coverage(coverage_json)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Function coverage check passed" in captured.out
