"""Secure, bounded execution primitives.

This module intentionally owns execution only. PDF parsing and artifact
persistence live in ``ingestion`` and ``reports`` respectively.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from textwrap import dedent


ALLOWED_MODULES = frozenset({
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "itertools", "functools", "operator", "collections", "copy",
    "pprint", "textwrap", "re", "string", "difflib", "enum", "numbers",
    "struct", "codecs", "unicodedata", "warnings", "datetime", "calendar",
    "time", "typing", "typing_extensions", "dataclasses", "abc",
    "hashlib", "hmac", "secrets", "json", "csv", "contextlib", "ast",
    "numpy", "scipy", "torch", "torchvision", "tqdm", "matplotlib", "PIL",
    "sklearn", "pandas",
    "_io", "_warnings", "_weakref", "_abc", "_codecs", "_collections_abc",
    "_frozen_importlib", "_frozen_importlib_external", "_imp", "_stat",
    "encodings", "errno", "genericpath", "posixpath", "ntpath", "stat",
})

BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "multiprocessing", "concurrent", "threading", "asyncio",
    "shutil", "glob", "fnmatch", "socket", "http", "urllib", "requests", "urllib3",
    "aiohttp", "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc", "ctypes", "cffi",
    "platform", "site", "sitecustomize", "usercustomize", "builtins", "__builtin__",
    "pdb", "cmd", "shlex", "getpass", "signal", "mmap", "importlib", "pkgutil",
    "zipimport", "pickle", "pickletools", "shelve", "marshal", "zipfile", "tarfile",
    "gzip", "bz2", "lzma", "zlib", "resource", "gc", "pathlib", "tempfile", "io",
})

BLOCKED_BUILTINS = frozenset({
    "open", "input", "eval", "exec", "compile", "breakpoint", "exit", "quit", "help",
})
BLOCKED_CALL_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "vars", "dir", "globals", "locals", "memoryview", "breakpoint",
})
BLOCKED_ATTRIBUTE_NAMES = frozenset({
    "__class__", "__bases__", "__base__", "__subclasses__", "__globals__", "__builtins__",
    "__getattribute__", "__reduce__", "__reduce_ex__", "__code__", "__loader__", "__spec__",
    "read_text", "read_bytes", "write_text", "write_bytes", "read_csv", "read_json",
    "read_pickle", "read_excel", "to_csv", "to_json", "to_pickle", "tofile", "fromfile",
    "load", "loads", "save", "savetxt", "savefig", "getenv", "environ",
})


def _top_level(module: str) -> str:
    return module.split(".", 1)[0]


def _check_code_safety(code: str) -> list[str]:
    """AST-based rejection of imports, calls, and introspection with escape/file surfaces."""
    try:
        tree = ast.parse(code, filename="<sandbox>")
    except SyntaxError:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if top in BLOCKED_MODULES or top not in ALLOWED_MODULES:
                    violations.append(f"Blocked module import detected: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append("Relative imports are not allowed in standalone sandbox snippets.")
                continue
            module = node.module or ""
            top = _top_level(module)
            if top in BLOCKED_MODULES or top not in ALLOWED_MODULES:
                violations.append(f"Blocked module import detected: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES:
                violations.append(f"Blocked call detected: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRIBUTE_NAMES:
                violations.append(f"Blocked operation detected: .{node.func.attr}()")
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRIBUTE_NAMES:
            violations.append(f"Blocked attribute access detected: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in {"__builtins__", "__import__"}:
            violations.append(f"Blocked builtin access detected: {node.id}")

    return list(dict.fromkeys(violations))


def _resource_setup_source(cpu_seconds: int, memory_bytes: int) -> str:
    """Child-process setup without ``preexec_fn`` (safe when caller has threads)."""
    return dedent(
        f"""
        try:
            import resource as _resource
            _resource.setrlimit(_resource.RLIMIT_CPU, (max({int(cpu_seconds)}, 5), max({int(cpu_seconds)} + 2, 7)))
            _resource.setrlimit(_resource.RLIMIT_AS, ({int(memory_bytes)}, {int(memory_bytes)}))
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (128, 128))
            _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
        except (ImportError, ValueError, OSError):
            pass
        """
    )


def _child_environment(project_dir: str | None, tmp_dir: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "HOME": tmp_dir,
        "TMPDIR": tmp_dir,
        "TEMP": tmp_dir,
        "TMP": tmp_dir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if project_dir:
        env["PYTHONPATH"] = project_dir
    return env


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def execute_python_code(code: str, timeout: int = 60, project_dir: str | None = None, trusted: bool = False) -> dict:
    """Run Python in a bounded child process with static and interpreter-level controls."""
    if not isinstance(code, str):
        return {"success": False, "stdout": "", "stderr": "Code must be a string.", "code": -2}
    if timeout <= 0 or timeout > 600:
        return {"success": False, "stdout": "", "stderr": "Timeout must be between 1 and 600 seconds.", "code": -2}

    if not trusted:
        violations = _check_code_safety(code)
        if violations:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Code rejected by safety check:\n" + "\n".join(f"  - {v}" for v in violations),
                "code": -2,
            }

    with tempfile.TemporaryDirectory() as tmp_dir:
        payload_path = os.path.join(tmp_dir, "payload.py")
        wrapper_path = os.path.join(tmp_dir, "run.py")
        Path(payload_path).write_text(code, encoding="utf-8")
        setup = _resource_setup_source(timeout, 4 * 1024 * 1024 * 1024)

        if trusted:
            wrapper = setup + f"""
with open({payload_path!r}, 'r', encoding='utf-8') as _f:
    _src = _f.read()
exec(compile(_src, '<trusted>', 'exec'), {{'__name__': '__main__', '__file__': {payload_path!r}}})
"""
        else:
            allowed = repr(sorted(ALLOWED_MODULES))
            blocked = repr(sorted(BLOCKED_MODULES))
            blocked_builtins = repr(sorted(BLOCKED_BUILTINS))
            wrapper = setup + f"""
import builtins as _bt
_allowed = set({allowed})
_blocked_modules = set({blocked})
_real_import = _bt.__import__
_real_open = _bt.open
_real_compile = _bt.compile
_real_exec = _bt.exec

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError('Relative imports are blocked in standalone sandbox execution.')
    top = name.split('.', 1)[0]
    if top in _blocked_modules or top not in _allowed:
        raise ImportError('Import of ' + name + ' is blocked in this sandbox.')
    return _real_import(name, globals, locals, fromlist, level)

def _blocked(*_a, **_k):
    raise PermissionError('Operation disabled in sandbox.')
_bt.__import__ = _guarded_import
for _name in {blocked_builtins}:
    if hasattr(_bt, _name):
        setattr(_bt, _name, _blocked)
with _real_open({payload_path!r}, 'r', encoding='utf-8') as _f:
    _src = _f.read()
_real_exec(_real_compile(_src, '<sandbox>', 'exec'), {{'__name__': '__main__', '__file__': {payload_path!r}}})
"""

        Path(wrapper_path).write_text(wrapper, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-B", wrapper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_dir or tmp_dir,
            env=_child_environment(project_dir, tmp_dir),
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return {
                "success": False,
                "stdout": stdout[:6000],
                "stderr": f"Execution timed out after {timeout}s; process group terminated.\n{stderr[:2500]}",
                "code": -1,
            }

        for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "API_AUTH_TOKEN"):
            secret = os.environ.get(key)
            if secret:
                stdout = stdout.replace(secret, "[REDACTED]")
                stderr = stderr.replace(secret, "[REDACTED]")

        return {"success": proc.returncode == 0, "stdout": stdout[:6000], "stderr": stderr[:3000], "code": proc.returncode}


_VALIDATION_SENTINEL = "===VALIDATION_RESULT_JSON==="


def run_codebase_validation(code_modules: dict, timeout: int = 90) -> dict:
    """Perform real syntax/import checks and best-effort Torch smoke tests."""
    if not code_modules:
        return {"measured": None, "error": "No code_modules to validate.", "raw_stdout": "", "raw_stderr": ""}

    with tempfile.TemporaryDirectory() as project_dir:
        project_dir = os.path.realpath(project_dir)
        target_files: list[tuple[str, str]] = []
        for filename, module in code_modules.items():
            if not isinstance(filename, str):
                continue
            cleaned = filename.replace("\\", "/")
            parts = [p for p in PurePosixPath(cleaned).parts if p not in ("", ".")]
            if not parts or any(p == ".." for p in parts) or cleaned.startswith("/") or (len(cleaned) > 1 and cleaned[1] == ":"):
                return {"measured": None, "error": f"Unsafe generated filename: {filename!r}", "raw_stdout": "", "raw_stderr": ""}
            rel = "/".join(parts)
            if not rel.endswith(".py"):
                continue
            code = module.get("code", "") if isinstance(module, dict) else str(module)
            full_path = os.path.join(project_dir, rel)
            Path(full_path).parent.mkdir(parents=True, exist_ok=True)
            Path(full_path).write_text(code, encoding="utf-8")
            target_files.append((rel, rel[:-3].replace("/", ".")))

        if not target_files:
            return {"measured": None, "error": "No .py files found among code_modules.", "raw_stdout": "", "raw_stderr": ""}

        for root, _dirs, files in os.walk(project_dir):
            if root == project_dir or any(f.endswith(".py") for f in files):
                Path(root, "__init__.py").touch(exist_ok=True)

        harness = dedent(f"""
            import ast, builtins, contextlib, importlib, inspect, io, json, os, sys, traceback
            PROJECT_DIR = os.path.realpath({project_dir!r})
            TARGET_FILES = {target_files!r}
            results = {{}}
            _real_import = builtins.__import__
            _real_open = builtins.open
            _allowed = set({sorted(ALLOWED_MODULES)!r})
            _blocked = set({sorted(BLOCKED_MODULES)!r})
            _blocked_builtins = set({sorted(BLOCKED_BUILTINS)!r})
            _stdlib_root = os.path.realpath(os.path.dirname(os.__file__))
            _prefix_root = os.path.realpath(sys.prefix)

            def _inside(path, root):
                path = os.path.realpath(path)
                root = os.path.realpath(root)
                return path == root or root in os.path.commonpath((path, root))

            def _safe_open(file, mode='r', *args, **kwargs):
                path = os.path.realpath(os.fspath(file))
                write_mode = any(flag in mode for flag in ('w', 'a', 'x', '+'))
                if not (_inside(path, PROJECT_DIR) or _inside(path, _stdlib_root) or _inside(path, _prefix_root)):
                    raise PermissionError('Filesystem access outside the validation project/runtime is blocked.')
                if write_mode and not _inside(path, PROJECT_DIR):
                    raise PermissionError('Writes outside the validation project are blocked.')
                return _real_open(file, mode, *args, **kwargs)

            def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if level:
                    return _real_import(name, globals, locals, fromlist, level)
                top = name.split('.', 1)[0]
                if top in _blocked or top not in _allowed:
                    if name not in sys.modules or not top.startswith('_'):
                        raise ImportError('Import of ' + name + ' is blocked in validation sandbox.')
                return _real_import(name, globals, locals, fromlist, level)

            def _blocked(*_args, **_kwargs):
                raise PermissionError('Operation disabled during generated-code validation.')

            builtins.__import__ = _guarded_import
            builtins.open = _safe_open
            for _name in _blocked_builtins:
                if hasattr(builtins, _name):
                    setattr(builtins, _name, _blocked)

            for rel_path, dotted in TARGET_FILES:
                entry = {{'syntax_ok': False, 'syntax_error': None, 'import_ok': False, 'import_error': None, 'classes_found': [], 'instantiation_attempts': []}}
                full = os.path.join(PROJECT_DIR, rel_path)
                try:
                    source = _real_open(full, 'r', encoding='utf-8').read()
                    try:
                        ast.parse(source, filename=rel_path)
                        entry['syntax_ok'] = True
                    except SyntaxError as exc:
                        entry['syntax_error'] = type(exc).__name__ + ': ' + str(exc)
                        results[rel_path] = entry
                        continue

                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            module = importlib.import_module(dotted)
                        entry['import_ok'] = True
                    except Exception:
                        entry['import_error'] = traceback.format_exc()[-1500:]
                        results[rel_path] = entry
                        continue

                    try:
                        import torch
                        import torch.nn as nn
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if obj.__module__ != dotted:
                                continue
                            entry['classes_found'].append(name)
                            if issubclass(obj, nn.Module):
                                attempt = {{'class': name, 'instantiated': False, 'forward_ok': False, 'error': None}}
                                try:
                                    instance = obj()
                                    attempt['instantiated'] = True
                                    for shape in ((1, 16), (1, 3, 8, 8), (1, 8, 16)):
                                        try:
                                            instance(torch.randn(*shape))
                                            attempt['forward_ok'] = True
                                            attempt['tried_shape'] = list(shape)
                                            break
                                        except Exception:
                                            pass
                                    if not attempt['forward_ok']:
                                        attempt['error'] = 'No generic dummy input shape worked.'
                                except Exception as exc:
                                    attempt['error'] = type(exc).__name__ + ': ' + str(exc)
                                entry['instantiation_attempts'].append(attempt)
                    except Exception:
                        pass
                except Exception as exc:
                    entry['import_error'] = 'Validation harness error: ' + type(exc).__name__ + ': ' + str(exc)
                results[rel_path] = entry

            builtins.__import__ = _real_import
            builtins.open = _real_open
            summary = {{'python_version': sys.version.split()[0], 'torch_available': 'torch' in sys.modules, 'files_checked': len(TARGET_FILES), 'files_syntax_ok': sum(1 for x in results.values() if x['syntax_ok']), 'files_import_ok': sum(1 for x in results.values() if x['import_ok']), 'per_file': results}}
            print({_VALIDATION_SENTINEL!r} + json.dumps(summary))
        """)

        executed = execute_python_code(harness, timeout=timeout, project_dir=project_dir, trusted=True)
        measured = None
        for line in executed.get("stdout", "").splitlines():
            if line.startswith(_VALIDATION_SENTINEL):
                try:
                    measured = json.loads(line[len(_VALIDATION_SENTINEL):])
                except json.JSONDecodeError:
                    measured = None
                break
        return {
            "measured": measured,
            "error": None if measured is not None else "Validation harness did not produce parseable results.",
            "raw_stdout": executed.get("stdout", ""),
            "raw_stderr": executed.get("stderr", ""),
            "harness_success": executed.get("success", False),
        }


def format_measured_results(validation: dict) -> str:
    measured = validation.get("measured")
    if measured is None:
        return "MEASURED RESULTS: unavailable.\nReason: " + validation.get("error", "unknown")
    lines = [
        f"Python version (measured): {measured['python_version']}",
        f"torch available (measured): {measured['torch_available']}",
        f"Files checked: {measured['files_checked']}",
        f"Files with valid syntax: {measured['files_syntax_ok']}/{measured['files_checked']}",
        f"Files that imported successfully: {measured['files_import_ok']}/{measured['files_checked']}",
        "Per-file results:",
    ]
    for filename, info in measured['per_file'].items():
        lines.append(f"  - {filename}: syntax_ok={info['syntax_ok']} import_ok={info['import_ok']}")
        if info.get('syntax_error'):
            lines.append(f"      syntax_error={info['syntax_error']}")
        if info.get('import_error'):
            lines.append(f"      import_error={info['import_error'][:500]}")
        for attempt in info.get('instantiation_attempts', []):
            lines.append(f"      {attempt['class']}: instantiated={attempt['instantiated']} forward_ok={attempt['forward_ok']} error={attempt.get('error')}")
    return "\n".join(lines)


__all__ = [
    "ALLOWED_MODULES", "BLOCKED_MODULES", "execute_python_code",
    "run_codebase_validation", "format_measured_results",
]
