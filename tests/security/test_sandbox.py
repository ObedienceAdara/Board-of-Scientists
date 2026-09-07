import pytest

from board_of_scientists.execution.sandbox import execute_python_code


@pytest.mark.parametrize(
    "code,needle",
    [
        ("import os\nprint('no')", "Blocked module import detected: os"),
        ("import subprocess\nprint('no')", "Blocked module import detected: subprocess"),
        ("from pathlib import Path\nprint(Path('.').resolve())", "Blocked module import detected: pathlib"),
        ("import io\nprint(io.open('x'))", "Blocked module import detected: io"),
        ("open('secret.txt')", "Blocked call detected: open()"),
        ("eval('1 + 1')", "Blocked call detected: eval()"),
        ("__import__('os')", "Blocked call detected: __import__()"),
        ("print((1).__class__)", "Blocked attribute access detected: .__class__"),
        ("import numpy as np\nnp.load('secret.npy')", "Blocked operation detected: .load()"),
        ("import torch\ntorch.load('secret.pt')", "Blocked operation detected: .load()"),
        ("import pandas as pd\npd.read_csv('secret.csv')", "Blocked operation detected: .read_csv()"),
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


def test_sandbox_timeout_kills_process_group():
    result = execute_python_code("while True: pass", timeout=1)
    assert result["success"] is False
    assert result["code"] == -1
    assert "timed out" in result["stderr"].lower()
