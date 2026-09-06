from board_of_scientists.execution.sandbox import execute_python_code


def test_generated_python_module_can_be_executed_by_sandbox():
    generated_code = "def forward(x):\n    return x * 2\nprint(forward(21))\n"
    result = execute_python_code(generated_code)
    assert result["success"] is True
    assert result["stdout"].strip() == "42"


def test_generated_code_with_network_access_is_rejected_before_execution():
    generated_code = "import socket\nsocket.create_connection(('example.com', 80))\n"
    result = execute_python_code(generated_code)
    assert result["success"] is False
    assert "Blocked module import detected: socket" in result["stderr"]
