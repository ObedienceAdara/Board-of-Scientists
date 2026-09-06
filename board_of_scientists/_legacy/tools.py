"""
tools.py — All tools available to the AI Research Team.

Tools:
  - PDF extraction (text, images, tables, equations) via PyMuPDF
  - Python REPL for code execution (sandboxed)
  - Web search via Tavily
  - File writer for saving code modules
"""

import os
import re
import json
import sys
import ast
import subprocess
import tempfile
import signal
from pathlib import Path, PurePosixPath
from datetime import datetime

from langchain_tavily import TavilySearch


# ══════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_pdf_pages(pdf_path: str) -> list:
    """
    Extract content from PDF page by page.
    Returns list of dicts: { page, text, has_figures, has_tables }

    Uses PyMuPDF (fitz) for extraction.
    Falls back to pdfplumber if fitz is unavailable.
    """
    pages = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        print(f"   📖 PDF loaded: {len(doc)} pages")

        for page_num in range(len(doc)):
            page     = doc[page_num]
            text     = page.get_text("text")

            # Detect figures/tables heuristically
            text_lower    = text.lower()
            has_figures   = any(kw in text_lower for kw in ["figure", "fig.", "fig "])
            has_tables    = any(kw in text_lower for kw in ["table", "tab."])
            has_equations = any(kw in text_lower for kw in ["equation", "eq.", "theorem", "proof", "lemma"])

            # Extract image list
            image_list = page.get_images(full=True)
            images = []
            for idx, img in enumerate(image_list):
                images.append(f"[Image {idx+1} on page {page_num+1}]")

            pages.append({
                "page":         page_num + 1,
                "text":         text.strip(),
                "has_figures":  has_figures,
                "has_tables":   has_tables,
                "has_equations":has_equations,
                "image_count":  len(image_list),
                "images":       images,
                "char_count":   len(text)
            })

        doc.close()

    except ImportError:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                print(f"   📖 PDF loaded via pdfplumber: {len(pdf.pages)} pages")
                for page_num, page in enumerate(pdf.pages):
                    text       = page.extract_text() or ""
                    text_lower = text.lower()
                    tables     = page.extract_tables() or []

                    pages.append({
                        "page":         page_num + 1,
                        "text":         text.strip(),
                        "has_figures":  "figure" in text_lower or "fig." in text_lower,
                        "has_tables":   len(tables) > 0,
                        "has_equations":any(kw in text_lower for kw in ["equation", "theorem", "proof"]),
                        "image_count":  0,
                        "images":       [],
                        "char_count":   len(text)
                    })
        except ImportError:
            print("   ⚠️  Neither PyMuPDF nor pdfplumber found. Install with: pip install pymupdf pdfplumber")
            pages = [{
                "page":         1,
                "text":         f"[Could not extract PDF: {pdf_path}]",
                "has_figures":  False,
                "has_tables":   False,
                "has_equations":False,
                "image_count":  0,
                "images":       [],
                "char_count":   0
            }]

    return pages


def extract_equations(text: str) -> list:
    """
    Heuristically extract equation-like patterns from text.
    Looks for LaTeX, numbered equations, theorem/lemma blocks.
    """
    equations = []
    lines     = text.split('\n')

    # Numbered equations pattern: (1), (2), etc.
    eq_pattern   = re.compile(r'\((\d+)\)')
    math_keywords = ["∀", "∃", "∑", "∏", "∫", "→", "←", "⟹", "≤", "≥",
                     "argmax", "argmin", "softmax", "sigmoid", "relu",
                     "\\mathcal", "\\mathbb", "\\frac", "\\sum", "\\prod"]

    for i, line in enumerate(lines):
        if eq_pattern.search(line) or any(kw in line for kw in math_keywords):
            equations.append({
                "line":    i + 1,
                "content": line.strip()
            })

    return equations


def get_paper_metadata(pages: list) -> dict:
    """Extract title, abstract, and section headings from first pages."""
    if not pages:
        return {"title": "Unknown", "abstract": "", "sections": []}

    # Title is usually in first 1-2 pages
    first_page_text = pages[0]["text"] if pages else ""
    lines           = [l.strip() for l in first_page_text.split('\n') if l.strip()]

    title    = lines[0] if lines else "Unknown Paper"
    abstract = ""

    # Find abstract
    full_text = "\n".join([p["text"] for p in pages[:3]])
    if "abstract" in full_text.lower():
        abs_start = full_text.lower().find("abstract")
        abs_end   = full_text.lower().find("introduction", abs_start)
        if abs_end == -1:
            abs_end = abs_start + 2000
        abstract = full_text[abs_start:abs_end].strip()[:1500]

    # Find section headings (ALL CAPS or numbered lines)
    sections = []
    heading_pattern = re.compile(r'^(\d+\.?\s+[A-Z][A-Za-z\s]+|[A-Z]{2,}[A-Z\s]+)$')
    for page in pages:
        for line in page["text"].split('\n'):
            line = line.strip()
            if heading_pattern.match(line) and len(line) < 80:
                sections.append({"page": page["page"], "heading": line})

    return {
        "title":    title[:200],
        "abstract": abstract,
        "sections": sections[:30]  # Cap at 30 sections
    }


# ══════════════════════════════════════════════════════════════
# PYTHON CODE EXECUTOR (SANDBOXED)
# ══════════════════════════════════════════════════════════════
#
# HONESTY NOTE (please read before trusting this as a security boundary)
# ------------------------------------------------------------------------
# The previous version of this file claimed five defense-in-depth layers,
# including "restricted builtins" and Unix "resource limits via the resource
# module." Neither was real: _build_safe_builtins() was defined but never
# called, and the `resource` module was never imported anywhere. What
# actually ran was a completely ordinary, unrestricted `python` subprocess,
# gated only by a regex/blocklist scan of the source text — which is well
# known to be bypassable (string concatenation, getattr() indirection,
# module aliasing, etc.).
#
# This version makes the claimed layers real:
#   1. Static analysis        — same regex pre-filter as before (fast-fail,
#                                imperfect, kept as one layer among several).
#   2. Guarded builtins        — dangerous builtins (open/eval/exec/compile/
#                                breakpoint/exit/quit) are actually removed,
#                                and __import__ is actually replaced with a
#                                whitelist-checking version, enforced by the
#                                interpreter's real import machinery — not by
#                                pattern-matching source text.
#   3. Temp directory isolation — unchanged: ephemeral dir, no project access.
#   4. Timeout enforcement     — unchanged: subprocess.run(timeout=...).
#   5. Resource limits (POSIX) — REAL now: resource.setrlimit() caps address
#                                space, CPU time, and file descriptors via a
#                                preexec_fn, enforced by the OS/kernel.
#
# What this is still NOT: a hard security boundary against a determined,
# sophisticated adversary. Pure-Python "restricted execution" has a long,
# well-documented history of being escaped via object introspection (e.g.
# walking `().__class__.__base__.__subclasses__()` to reach a useful gadget)
# — there is no way to close that off from inside the interpreter itself.
# This sandbox is appropriate for validating code THIS SAME PIPELINE
# generated (semi-trusted, not adversarial third-party input) and for
# catching honest bugs plus unsophisticated attempts. If this is ever used
# to run genuinely untrusted, adversarial code, wrap it in real OS-level
# isolation (a container with dropped capabilities, gVisor, Firecracker,
# nsjail) before exposing it to the outside world. See SECURITY.md.

# Third-party / stdlib modules considered safe for research code.
ALLOWED_MODULES = {
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "itertools", "functools", "operator", "collections", "copy",
    "pprint", "textwrap", "re", "string", "difflib", "enum",
    "numbers", "struct", "codecs", "unicodedata", "warnings",
    "datetime", "calendar", "time",
    "typing", "typing_extensions",
    "dataclasses", "abc",
    "hashlib", "hmac", "secrets",
    "pathlib",
    "json", "csv", "io",
    "contextlib",
    "inspect", "dis", "ast",
    # Scientific computing
    "numpy", "scipy",
    # Deep learning
    "torch", "torchvision",
    # Utilities
    "tqdm", "matplotlib", "PIL", "sklearn", "pandas",
    # CPython's own import machinery pulls these in lazily while loading any
    # ordinary source file (e.g. _io backs file reads during import) — they
    # are harmless bootstrap/path/encoding internals with no dangerous
    # surface on their own (unlike e.g. _socket or _ctypes, which stay
    # blocked). Without allowing these, even importing a completely benign
    # target file fails before user code is ever reached.
    "_io", "_warnings", "_weakref", "_abc", "_codecs", "_collections_abc",
    "_frozen_importlib", "_frozen_importlib_external", "_imp", "_stat",
    "encodings", "codecs", "abc", "errno", "genericpath", "posixpath",
    "ntpath", "stat", "marshal",
}

# ENFORCED allow-list: only the TOP-LEVEL package name matters for the
# guarded import (Python's import machinery gives full access to a module's
# entire attribute tree once any part of it is imported — there is no way to
# grant "os.path" without also granting "os", so we don't pretend to).
# `os` and `sys` are deliberately NOT derived/allowed here even though they
# used to appear as "os.path" in ALLOWED_MODULES — that entry never actually
# restricted anything (see note above). Use `pathlib` for path operations.
ALLOWED_TOP_LEVEL_MODULES = frozenset(m.split(".")[0] for m in ALLOWED_MODULES)

# Modules explicitly and always blocked, regardless of ALLOWED_TOP_LEVEL_MODULES.
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "multiprocessing", "concurrent", "threading", "asyncio",
    "shutil", "glob", "fnmatch",
    "socket", "http", "urllib", "requests", "urllib3", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc",
    "ctypes", "cffi",
    "platform", "site", "sitecustomize", "usercustomize",
    "builtins", "__builtin__",
    "pdb", "cmd", "shlex", "getpass",
    "signal", "mmap",
    "importlib", "pkgutil", "zipimport",
    "pickle", "pickletools", "shelve", "marshal",
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "zlib",
    "resource", "gc",
}

BLOCKED_BUILTINS = frozenset({
    "open", "input", "eval", "exec", "compile", "breakpoint", "exit", "quit", "help",
})


def _check_code_safety(code: str) -> list:
    """
    Fast-fail static analysis. This is layer 1 of several — kept because it's
    cheap and catches obviously-bad code before we even spin up a subprocess
    — but it is NOT relied upon as the only defense (see module docstring).
    """
    violations = []

    for module in BLOCKED_MODULES:
        patterns = [
            rf"^\s*import\s+{re.escape(module)}\b",
            rf"^\s*from\s+{re.escape(module)}\b",
            rf"__import__\s*\(\s*['\"]{re.escape(module)}(\.[a-zA-Z0-9_.]*)?['\"]",
            rf"importlib\.import_module\s*\(\s*['\"]{re.escape(module)}",
        ]
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE):
                violations.append(f"Blocked module import detected: {module}")
                break

    dangerous_calls = [
        (r"\bopen\s*\(", "File I/O via open()"),
        (r"\beval\s*\(", "Dynamic code execution via eval()"),
        (r"\bexec\s*\(", "Dynamic code execution via exec()"),
        (r"\bcompile\s*\(", "Dynamic code compilation via compile()"),
        (r"os\.system\s*\(", "System command execution via os.system()"),
        (r"os\.popen\s*\(", "Process creation via os.popen()"),
        (r"subprocess\.", "Subprocess execution"),
        (r"socket\.", "Network socket access"),
        (r"requests\.", "HTTP requests"),
        (r"urllib\.", "URL handling"),
        (r"shutil\.", "File system operations via shutil"),
        (r"ctypes\.", "C library access via ctypes"),
        (r"pickle\.", "Object serialization via pickle"),
        (r"pdb\.set_trace", "Debugger invocation"),
        (r"breakpoint\s*\(", "Debugger invocation"),
        (r"\bexit\s*\(", "Process termination via exit()"),
        (r"\bquit\s*\(", "Process termination via quit()"),
    ]
    for pattern, description in dangerous_calls:
        if re.search(pattern, code):
            violations.append(f"Dangerous pattern detected: {description}")

    traversal_patterns = [
        r"\.\./", r"\.\.\\", r"/etc/", r"/proc/", r"/sys/", r"/dev/",
        r"\\windows\\", r"\\system32\\",
    ]
    for pattern in traversal_patterns:
        if re.search(pattern, code):
            violations.append("Path traversal or system path access detected")
            break

    return violations


# ── Real resource limits (POSIX only) ───────────────────────────
def _posix_resource_limits(cpu_seconds: int):
    """
    preexec_fn for subprocess.run: runs in the forked child, before exec(),
    and applies REAL, kernel-enforced limits via resource.setrlimit(). This
    is the piece that was previously only claimed in a docstring and never
    implemented.

    Memory is capped generously (4 GB virtual address space) rather than
    tightly, because RLIMIT_AS limits *virtual* address space, and libraries
    like PyTorch/NumPy routinely reserve large virtual mappings (e.g. during
    CUDA availability checks) that don't reflect real memory pressure — a
    tight cap would produce false-negative failures on legitimate ML code,
    which would undermine the goal of honest measurement. CPU time and wall
    clock (via subprocess timeout) are the primary, reliable backstops here.
    """
    def _limiter():
        import resource
        try:
            mem_bytes = 4 * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        try:
            cpu_cap = max(int(cpu_seconds), 5)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap + 2))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
    return _limiter


def _preexec_fn_for(timeout: int):
    """Returns a real preexec_fn on POSIX, or None on platforms without the resource module (e.g. Windows)."""
    if os.name != "posix":
        return None
    return _posix_resource_limits(timeout)


# ── Guarded builtins, actually wired in (fixes the dead-code bug) ──
_GUARDED_IMPORT_PREAMBLE = '''
import os as _os_ref, builtins as _bt

_ALLOWED_TOP_LEVEL = {allowed!r}
_PROJECT_DIR = {project_dir!r}
_real_import = _bt.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Uses the pre-imported _os_ref, never "import os" here — that would
    # recurse through this function once it's installed as __import__.
    top = name.split(".")[0]
    intra_project = False
    if _PROJECT_DIR:
        intra_project = (
            _os_ref.path.isdir(_os_ref.path.join(_PROJECT_DIR, top))
            or _os_ref.path.isfile(_os_ref.path.join(_PROJECT_DIR, top + ".py"))
        )
    if top in _ALLOWED_TOP_LEVEL or intra_project:
        return _real_import(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of '{{name}}' is blocked in this sandbox.")

_bt.__import__ = _guarded_import
for _blocked_name in {blocked_builtins!r}:
    if hasattr(_bt, _blocked_name):
        def _make_blocker(_n):
            def _blocked(*a, **k):
                raise NameError(f"'{{_n}}' is disabled in this sandbox.")
            return _blocked
        setattr(_bt, _blocked_name, _make_blocker(_blocked_name))
'''


def _build_safe_builtins() -> str:
    """
    Returns the Python SOURCE (as a string) that installs guarded builtins
    when executed at the top of a subprocess script. We generate source
    rather than a live dict because the restriction has to take effect
    *inside the child subprocess's own interpreter* — a dict of live
    function objects in this (parent) process can't be handed across a
    process boundary. Kept as a function (not a module-level constant) so
    every call site is explicit that this text still needs `.format(...)`.
    """
    return _GUARDED_IMPORT_PREAMBLE


def execute_python_code(code: str, timeout: int = 60, project_dir: str = None, trusted: bool = False) -> dict:
    """
    Execute Python code in a sandboxed subprocess with real, layered isolation:

    1. Static analysis     — regex pre-filter (layer for fast, cheap rejection).
    2. Guarded builtins     — open/eval/exec/compile/breakpoint/exit/quit are
                              actually removed; __import__ is actually replaced
                              with a whitelist-checking version enforced by the
                              interpreter's real import machinery.
    3. Temp dir isolation   — ephemeral directory, no access to project files
                              (unless `project_dir` is given for intra-project
                              imports — see run_codebase_validation()).
    4. Timeout enforcement  — subprocess.run(timeout=...).
    5. Resource limits      — REAL on POSIX via resource.setrlimit(), applied
                              through preexec_fn.

    `trusted=True` is for scripts THIS CODEBASE authored (e.g. the validation
    harness below) rather than LLM-generated payloads: it skips the regex
    pre-filter and the whole-script builtins restriction (the harness needs
    real open()/exec() for its own logic, and applies its own fine-grained
    guarded-import scope around just the target code it's inspecting — see
    _GUARDED_SCOPE_SNIPPET). Resource limits, timeout, and temp-dir isolation
    still apply regardless of `trusted` — those are unconditional OS-level
    backstops, not a judgment call about the code's trustworthiness.

    See the module-level docstring above for what this is and isn't.
    """
    if not trusted:
        violations = _check_code_safety(code)
        if violations:
            return {
                "success": False,
                "stdout":  "",
                "stderr":  "Code rejected by safety check:\n" + "\n".join(f"  - {v}" for v in violations),
                "code":    -2,
            }

    with tempfile.TemporaryDirectory() as tmp_dir:
        payload_path = os.path.join(tmp_dir, "_payload.py")
        script_path = os.path.join(tmp_dir, "run.py")

        with open(payload_path, "w", encoding="utf-8") as f:
            f.write(code)

        if trusted:
            # Run with normal, unrestricted builtins — this is our own code,
            # not an LLM payload. Resource limits/timeout/temp-dir isolation
            # (applied below, unconditionally) remain the real backstop.
            wrapper = f'''
with open({payload_path!r}, "r", encoding="utf-8") as _f:
    _src = _f.read()
_real_exec, _real_compile = exec, compile
_real_exec(_real_compile(_src, "<harness_code>", "exec"), {{"__name__": "__main__"}})
'''
        else:
            guarded_preamble = _build_safe_builtins().format(
                allowed=ALLOWED_TOP_LEVEL_MODULES,
                project_dir=project_dir or "",
                blocked_builtins=BLOCKED_BUILTINS,
            )
            # Order matters: read the payload with the REAL open() first,
            # capture real exec/compile before they get patched out, THEN
            # install guarded builtins, THEN exec the payload. Guarding
            # first would make the guarded open()/exec() block this very
            # wrapper's own trusted read-and-run steps.
            wrapper = f'''
with open({payload_path!r}, "r", encoding="utf-8") as _f:
    _src = _f.read()
_real_exec, _real_compile = exec, compile
''' + guarded_preamble + '''
_real_exec(_real_compile(_src, "<user_code>", "exec"), {"__name__": "__main__"})
'''
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapper)

        try:
            result = subprocess.run(
                [sys.executable, "-u", "-B", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=(project_dir or tmp_dir),
                preexec_fn=_preexec_fn_for(timeout),
                env={
                    **{k: v for k, v in os.environ.items()
                       if k not in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
                                     "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
                                     "API_AUTH_TOKEN")},
                    "PYTHONPATH": project_dir or "",
                    "HOME": tmp_dir,
                    "TMPDIR": tmp_dir,
                    "TEMP": tmp_dir,
                    "TMP": tmp_dir,
                }
            )

            stdout = result.stdout[:6000]
            stderr = result.stderr[:3000]

            for key_name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
                              "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
                              "API_AUTH_TOKEN"):
                actual_key = os.environ.get(key_name)
                if actual_key:
                    stdout = stdout.replace(actual_key, "[REDACTED]")
                    stderr = stderr.replace(actual_key, "[REDACTED]")

            return {
                "success": result.returncode == 0,
                "stdout":  stdout,
                "stderr":  stderr,
                "code":    result.returncode,
            }

        except subprocess.TimeoutExpired as e:
            if hasattr(e, "process") and e.process is not None:
                try:
                    e.process.kill()
                except Exception:
                    pass
            return {
                "success": False,
                "stdout":  "",
                "stderr":  f"Execution timed out after {timeout}s. Process killed.",
                "code":    -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout":  "",
                "stderr":  f"Execution failed: {type(e).__name__}: {e}",
                "code":    -1,
            }


# ══════════════════════════════════════════════════════════════
# REAL CODEBASE VALIDATION (fixes the "fake validation" bug)
# ══════════════════════════════════════════════════════════════
#
# Previously, experiment_engineer_agent ran a fixed, generic smoke test that
# only checked the Python version and whether `torch`/`numpy` import — it
# never touched the actual generated code. Every "forward pass result",
# "discrepancy percentage", and "gradient check" the LLM reported was
# invented text, not a measurement, even though it was written into a
# polished PDF report with a "FINAL VERDICT."
#
# run_codebase_validation() below actually writes the generated files to a
# real project directory and runs a harness INSIDE the sandbox that performs
# genuine, measured checks: syntax validity, real import success/failure,
# and a best-effort instantiation/forward-pass smoke test for any
# torch.nn.Module subclasses it finds. Everything in the returned dict is
# something that was actually executed — nothing here is inferred.

_VALIDATION_SENTINEL = "===VALIDATION_RESULT_JSON==="

# The harness itself runs TRUSTED (real open(), real exec()) — but the
# TARGET files it imports are semi-trusted, LLM-generated code. This
# provides a context manager the harness uses to apply the same guarded
# __import__ / blocked-builtins treatment as execute_python_code(), scoped
# tightly around just "import the target module" and "instantiate/call it",
# so the harness's own file I/O outside that scope is unaffected.
_GUARDED_SCOPE_SNIPPET = '''
import os as _os_ref, builtins as _bt, contextlib as _cl

_ALLOWED_TOP_LEVEL = {allowed!r}
_PROJECT_DIR_FOR_GUARD = {project_dir!r}
# NOTE: this is deliberately a NARROWER blocklist than execute_python_code()
# uses for plain snippets. compile()/exec() cannot be blocked here: Python's
# own import machinery calls compile() internally to turn a .py file's
# source into bytecode, so blocking it would break importing ANY file, not
# just malicious ones. The primary defense for this path is the __import__
# allow-list below (still blocks os/subprocess/socket/ctypes/pickle/etc.),
# backed by the resource limits and timeout applied to the whole subprocess.
_BLOCKED_NAMES = {{"open", "input", "breakpoint", "exit", "quit"}}

@_cl.contextmanager
def _guarded_scope():
    _real_import = _bt.__import__
    _saved = {{}}
    for _n in _BLOCKED_NAMES:
        if hasattr(_bt, _n):
            _saved[_n] = getattr(_bt, _n)

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # NOTE: uses the pre-imported _os_ref, never "import os" here — doing
        # the import inline would recurse through this very function, since
        # it's what builtins.__import__ now points to.
        top = name.split(".")[0]
        intra_project = False
        if _PROJECT_DIR_FOR_GUARD:
            intra_project = (
                _os_ref.path.isdir(_os_ref.path.join(_PROJECT_DIR_FOR_GUARD, top))
                or _os_ref.path.isfile(_os_ref.path.join(_PROJECT_DIR_FOR_GUARD, top + ".py"))
            )
        if top in _ALLOWED_TOP_LEVEL or intra_project:
            return _real_import(name, globals, locals, fromlist, level)
        raise ImportError(f"Import of '{{name}}' is blocked in this sandbox.")

    def _make_blocker(_n):
        def _blocked(*a, **k):
            raise NameError(f"'{{_n}}' is disabled while executing sandboxed target code.")
        return _blocked

    _bt.__import__ = _guarded_import
    for _n in _BLOCKED_NAMES:
        if hasattr(_bt, _n):
            setattr(_bt, _n, _make_blocker(_n))
    try:
        yield
    finally:
        _bt.__import__ = _real_import
        for _n, _orig in _saved.items():
            setattr(_bt, _n, _orig)
'''

_VALIDATION_HARNESS_TEMPLATE = '''
import ast, json, sys, traceback, importlib, io, contextlib

PROJECT_DIR = {project_dir!r}
TARGET_FILES = {target_files!r}  # list of (relative_path, dotted_module_name)

''' + _GUARDED_SCOPE_SNIPPET + '''

results = {{}}

def _torch_available():
    try:
        import torch  # noqa
        return True
    except Exception:
        return False

TORCH_OK = _torch_available()

for rel_path, dotted in TARGET_FILES:
    entry = {{
        "syntax_ok": False, "syntax_error": None,
        "import_ok": False, "import_error": None,
        "classes_found": [], "instantiation_attempts": [],
    }}
    full_path = PROJECT_DIR + "/" + rel_path
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        entry["syntax_error"] = f"Could not read file: {{e}}"
        results[rel_path] = entry
        continue

    # 1. REAL syntax check (no execution at all).
    try:
        ast.parse(source, filename=rel_path)
        entry["syntax_ok"] = True
    except SyntaxError as e:
        entry["syntax_error"] = f"{{e.__class__.__name__}}: {{e}} (line {{e.lineno}})"
        results[rel_path] = entry
        continue

    # 2. REAL import attempt (this is the actual interpreter importing the
    #    actual file — success/failure and the traceback are both measured,
    #    not guessed).
    buf_out, buf_err = io.StringIO(), io.StringIO()
    module = None
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            with _guarded_scope():
                module = importlib.import_module(dotted)
        entry["import_ok"] = True
    except Exception:
        entry["import_error"] = traceback.format_exc()[-1500:]
        results[rel_path] = entry
        continue

    # 3. Best-effort discovery + instantiation smoke test. Heuristic: many
    #    classes require constructor args we can't guess, so "failed to
    #    instantiate with no arguments" is reported as-is, not as a bug.
    try:
        import inspect as _inspect
        for name, obj in _inspect.getmembers(module, _inspect.isclass):
            if obj.__module__ != dotted:
                continue  # skip re-exported / imported classes
            entry["classes_found"].append(name)
            if not TORCH_OK:
                continue
            import torch.nn as _nn
            if not issubclass(obj, _nn.Module):
                continue
            attempt = {{"class": name, "instantiated": False, "forward_ok": False, "error": None}}
            try:
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err), _guarded_scope():
                    instance = obj()
                attempt["instantiated"] = True
                import torch as _torch
                for shape in [(1, 16), (1, 3, 8, 8), (1, 8, 16)]:
                    try:
                        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err), _guarded_scope():
                            dummy = _torch.randn(*shape)
                            _ = instance(dummy)
                        attempt["forward_ok"] = True
                        attempt["tried_shape"] = list(shape)
                        break
                    except Exception:
                        continue
                if not attempt["forward_ok"]:
                    attempt["error"] = "No generic dummy input shape worked (expected for non-trivial signatures)."
            except Exception as e:
                attempt["error"] = f"{{type(e).__name__}}: {{e}}"[:300]
            entry["instantiation_attempts"].append(attempt)
    except Exception as e:
        entry["classes_found"].append(f"[inspection error: {{e}}]")

    results[rel_path] = entry

summary = {{
    "python_version": sys.version.split()[0],
    "torch_available": TORCH_OK,
    "files_checked": len(TARGET_FILES),
    "files_syntax_ok": sum(1 for r in results.values() if r["syntax_ok"]),
    "files_import_ok": sum(1 for r in results.values() if r["import_ok"]),
    "per_file": results,
}}

print({sentinel!r} + json.dumps(summary))
'''


def run_codebase_validation(code_modules: dict, timeout: int = 90) -> dict:
    """
    Actually execute the generated codebase and return REAL, measured
    results — not an LLM's guess at what execution would probably show.

    Writes every file in `code_modules` into a real, importable project
    directory (auto-creating __init__.py so intra-project imports like
    'from models.backbone import Encoder' resolve correctly), then runs a
    harness inside the same hardened sandbox (guarded builtins + resource
    limits + timeout) that performs, per .py file: a real syntax check, a
    real import attempt, and — for any torch.nn.Module subclasses found and
    only if torch is actually installed — a best-effort instantiation and
    forward-pass smoke test.

    Returns a dict with a top-level "measured" key holding this structured,
    real data, plus "raw_stdout"/"raw_stderr" for debugging. Callers should
    treat everything under "measured" as ground truth and never let an LLM
    silently overwrite it.
    """
    if not code_modules:
        return {
            "measured": None,
            "error": "No code_modules to validate — the Engineer produced no files.",
            "raw_stdout": "", "raw_stderr": "",
        }

    with tempfile.TemporaryDirectory() as project_dir:
        project_dir = os.path.realpath(project_dir)
        target_files = []
        for filename, module in code_modules.items():
            code = module.get("code", "") if isinstance(module, dict) else str(module)
            rel_path = sanitize_relative_path(filename)
            if not rel_path.endswith(".py"):
                continue
            full_path = os.path.join(project_dir, rel_path)
            Path(full_path).parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(code)
            dotted = rel_path[:-3].replace("/", ".")
            target_files.append((rel_path, dotted))

        # Make every intermediate directory a real package so intra-project
        # imports (e.g. `from models.backbone import Encoder`) resolve.
        for root, _dirs, files in os.walk(project_dir):
            if any(f.endswith(".py") for f in files) or root == project_dir:
                init_path = os.path.join(root, "__init__.py")
                if not os.path.exists(init_path):
                    with open(init_path, "w", encoding="utf-8") as f:
                        f.write("")

        if not target_files:
            return {
                "measured": None,
                "error": "No .py files found among code_modules to validate.",
                "raw_stdout": "", "raw_stderr": "",
            }

        harness_source = _VALIDATION_HARNESS_TEMPLATE.format(
            project_dir=project_dir,
            target_files=target_files,
            sentinel=_VALIDATION_SENTINEL,
            allowed=ALLOWED_TOP_LEVEL_MODULES,
        )

        # trusted=True: this is our own hardcoded harness, not an LLM
        # payload — it needs real open()/exec() for its own logic, and
        # applies its own fine-grained _guarded_scope() around just the
        # target-file import/instantiation steps (see above).
        exec_result = execute_python_code(harness_source, timeout=timeout, project_dir=project_dir, trusted=True)

        measured = None
        for line in exec_result.get("stdout", "").splitlines():
            if line.startswith(_VALIDATION_SENTINEL):
                try:
                    measured = json.loads(line[len(_VALIDATION_SENTINEL):])
                except json.JSONDecodeError:
                    measured = None
                break

        return {
            "measured": measured,
            "error": None if measured is not None else "Harness did not produce a parseable result.",
            "raw_stdout": exec_result.get("stdout", ""),
            "raw_stderr": exec_result.get("stderr", ""),
            "harness_success": exec_result.get("success", False),
        }


def format_measured_results(validation: dict) -> str:
    """
    Render run_codebase_validation()'s output as a clearly-labeled,
    human/LLM-readable block. This text is what gets embedded in the
    Experiment Engineer's prompt as ground truth — everything in it was
    actually executed, nothing was inferred.
    """
    measured = validation.get("measured")
    if measured is None:
        return (
            "MEASURED RESULTS: unavailable.\n"
            f"Reason: {validation.get('error', 'unknown')}\n"
            f"Harness stderr (truncated): {validation.get('raw_stderr', '')[:800]}"
        )

    lines = [
        f"Python version (measured): {measured['python_version']}",
        f"torch available (measured): {measured['torch_available']}",
        f"Files checked: {measured['files_checked']}",
        f"Files with valid syntax: {measured['files_syntax_ok']}/{measured['files_checked']}",
        f"Files that imported successfully: {measured['files_import_ok']}/{measured['files_checked']}",
        "",
        "Per-file results:",
    ]
    for rel_path, info in measured["per_file"].items():
        lines.append(f"  - {rel_path}")
        lines.append(f"      syntax_ok={info['syntax_ok']}" + (f"  syntax_error={info['syntax_error']}" if info["syntax_error"] else ""))
        lines.append(f"      import_ok={info['import_ok']}" + (f"  import_error={(info['import_error'] or '')[:300]}" if info["import_error"] else ""))
        if info["classes_found"]:
            lines.append(f"      classes_found={info['classes_found']}")
        for attempt in info["instantiation_attempts"]:
            lines.append(
                f"      [{attempt['class']}] instantiated={attempt['instantiated']} "
                f"forward_ok={attempt['forward_ok']}"
                + (f" error={attempt['error']}" if attempt.get("error") else "")
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# WEB SEARCH
# ══════════════════════════════════════════════════════════════

def get_search_tool(max_results: int = 5) -> TavilySearch:
    return TavilySearch(max_results=max_results)


def parse_search_results(results) -> str:
    """Safe parsing of Tavily results regardless of return type."""
    if isinstance(results, str):
        return results
    if isinstance(results, dict):
        return f"URL: {results.get('url', '')}\nContent: {results.get('content', str(results))}"
    if isinstance(results, list):
        parts = []
        for r in results:
            if isinstance(r, dict):
                parts.append(f"URL: {r.get('url','')}\nContent: {r.get('content', r.get('snippet', str(r)))}")
            else:
                parts.append(str(r))
        return "\n\n".join(parts)
    return str(results)


_search_tool = get_search_tool()

def web_search(query: str) -> str:
    """Search the web and return parsed results."""
    try:
        results = _search_tool.invoke(query)
        return parse_search_results(results)
    except Exception as e:
        return f"Search error: {e}"


# ══════════════════════════════════════════════════════════════
# FILE WRITER
# ══════════════════════════════════════════════════════════════

def sanitize_relative_path(filename: str) -> str:
    """
    Validate and normalize a filename that may include subdirectories (e.g.
    'models/backbone.py'), rejecting path traversal, absolute paths, drive
    letters, and null bytes. Returns a safe, relative, POSIX-style path.

    Shared by save_code_file() (final on-disk output) and
    run_codebase_validation() (the sandbox validation writer) so "what gets
    validated" and "what gets saved to disk" can never silently diverge.

    NOTE: the previous version of save_code_file() used
    os.path.basename(filename), which discarded every subdirectory the
    Architect designed (e.g. 'models/backbone.py' was saved as flat
    'backbone.py') — silently flattening the whole proposed package
    structure. This preserves safe subdirectories instead of stripping them.
    """
    if not filename or "\x00" in filename:
        return "unnamed_module.py"

    cleaned = filename.replace("\\", "/").strip().lstrip("/")
    if len(cleaned) > 1 and cleaned[1] == ":":  # strip a Windows drive prefix like "C:"
        cleaned = cleaned[2:].lstrip("/")

    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return "unnamed_module.py"

    safe_path = "/".join(parts)

    # Defense-in-depth: resolve against a throwaway root and confirm we
    # haven't somehow escaped it.
    probe_root = os.sep + "__sandbox_probe_root__"
    resolved = os.path.normpath(os.path.join(probe_root, safe_path))
    if not (resolved == probe_root or resolved.startswith(probe_root + os.sep)):
        return "unnamed_module.py"

    return safe_path


def save_code_file(output_dir: str, filename: str, code: str) -> str:
    """
    Save a code module to the output directory, preserving safe relative
    subdirectories so the Architect's proposed package structure survives to
    disk instead of being silently flattened.

    Security: rejects path traversal ('../'), absolute paths, and null bytes
    via sanitize_relative_path().
    """
    output_dir_real = os.path.realpath(output_dir)
    Path(output_dir_real).mkdir(parents=True, exist_ok=True)

    safe_rel_path = sanitize_relative_path(filename)
    full_path = os.path.realpath(os.path.join(output_dir_real, safe_rel_path))

    if not (full_path == output_dir_real or full_path.startswith(output_dir_real + os.sep)):
        raise ValueError(
            f"Rejected filename '{filename}' — resolved path '{full_path}' "
            f"is outside output directory '{output_dir_real}'"
        )

    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"   💾 Saved: {safe_rel_path}")
    return full_path


def save_all_modules(output_dir: str, code_modules: dict) -> list:
    """Save all code modules from state to disk."""
    saved = []
    for filename, module in code_modules.items():
        if isinstance(module, dict):
            code = module.get("code", "")
        else:
            code = str(module)
        path = save_code_file(output_dir, filename, code)
        saved.append(path)
    return saved


# ══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════

from reportlab.lib.pagesizes   import A4
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units       import cm
from reportlab.lib             import colors
from reportlab.platypus        import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable, Preformatted
from reportlab.lib.enums       import TA_CENTER


def generate_implementation_report(data: dict, output_path: str) -> str:
    """
    Generate a PDF implementation report.
    data keys: paper_title, date, sections (list of {title, content})
    """
    doc    = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                              textColor=colors.HexColor("#0d1117"), alignment=TA_CENTER, spaceAfter=10)
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=11,
                              textColor=colors.HexColor("#555"), alignment=TA_CENTER, spaceAfter=6)
    head_s  = ParagraphStyle("H", parent=styles["Heading1"], fontSize=14,
                              textColor=colors.HexColor("#0d1117"), spaceBefore=16, spaceAfter=8)
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=9,
                              leading=15, textColor=colors.HexColor("#222"), spaceAfter=6)
    code_s  = ParagraphStyle("C", parent=styles["Code"], fontSize=7,
                              leading=11, textColor=colors.HexColor("#1a1a2e"), spaceAfter=4)

    story = []

    # Cover
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("AI RESEARCH IMPLEMENTATION TEAM", sub_s))
    story.append(Paragraph("Implementation Report", title_s))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0d1117")))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(data.get("paper_title", "Research Paper"), sub_s))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"Generated: {data.get('date', datetime.now().strftime('%Y-%m-%d'))}", sub_s))
    story.append(PageBreak())

    for section in data.get("sections", []):
        story.append(Paragraph(section["title"], head_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.3*cm))

        content = (section["content"]
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("\n", "<br/>"))
        story.append(Paragraph(content, body_s))
        story.append(PageBreak())

    doc.build(story)
    print(f"   ✅ PDF report: {output_path}")
    return output_path
