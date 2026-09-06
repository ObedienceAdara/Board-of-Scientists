import pytest

from board_of_scientists.execution.sandbox import execute_python_code


@pytest.mark.parametrize(
    "code,needle",
    [
        ("import os\nprint('no')", "Blocked module import detected: os"),
        ("import subprocess\nprint('no')", "Blocked module import detected: subprocess"),
        ("open('secret.txt')", "File I/O via open()"),
        ("eval('1 + 1')", "Dynamic code execution via eval()"),
    ],
)
def test_sandbox_rejects_known_unsafe_payloads(code, needle):
    result = execute_python_code(code)
    assert result["success"] is False
    assert result["code"] == -2
    assert needle in result["stderr"]


def test_sandbox_runs_safe_deterministic_code():
    result = execute_python_code("print(6 * 7)")
    assert result["success"] is True
    assert result["stdout"].strip() == "42"


def test_sandbox_allows_whitelisted_math_imports():
    result = execute_python_code("import math\nprint(math.sqrt(81))")
    assert result["success"] is True
    assert result["stdout"].strip() == "9.0"
