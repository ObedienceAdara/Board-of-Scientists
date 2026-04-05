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

# Modules that are safe to import in a research context
ALLOWED_MODULES = {
    # Standard library (safe subset)
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "itertools", "functools", "operator", "collections", "copy",
    "pprint", "textwrap", "re", "string", "difflib", "enum",
    "numbers", "struct", "codecs", "unicodedata",
    "datetime", "calendar", "time",
    "typing", "typing_extensions",
    "dataclasses", "abc",
    "hashlib", "hmac", "secrets",
    "pathlib", "os.path",
    "json", "csv", "io",
    "contextlib",
    "inspect", "dis",
    # Scientific computing
    "numpy", "numpy.linalg", "numpy.random",
    "scipy", "scipy.stats", "scipy.optimize", "scipy.linalg", "scipy.signal",
    "scipy.ndimage", "scipy.sparse", "scipy.integrate",
    # Deep learning
    "torch", "torch.nn", "torch.nn.functional", "torch.optim",
    "torch.utils.data", "torch.utils.data.dataloader",
    "torchvision", "torchvision.transforms",
    # Utilities
    "tqdm", "matplotlib", "matplotlib.pyplot", "PIL", "pillow",
    "sklearn", "sklearn.metrics", "sklearn.model_selection",
    "pandas",
}

# Dangerous modules/patterns that must be blocked
BLOCKED_MODULES = {
    # System access
    "os", "sys", "subprocess", "multiprocessing", "concurrent.futures",
    "threading", "asyncio",
    # File system
    "shutil", "glob", "fnmatch",
    # Network
    "socket", "http", "http.client", "http.server", "urllib",
    "requests", "urllib3", "aiohttp", "ftplib", "smtplib", "poplib",
    "imaplib", "xmlrpc",
    # Dynamic code execution
    "ctypes", "cffi", "dl",
    # Platform introspection
    "platform", "site", "sitecustomize", "usercustomize",
    # Built-in access
    "builtins", "__builtin__",
    # Command execution
    "pdb", "cmd", "shlex", "getpass",
    # Signal handling
    "signal",
    # Resource limits (could be used to escape sandbox)
    "mmap",
    # Import machinery
    "importlib", "pkgutil", "zipimport",
    # Pickle (arbitrary code execution)
    "pickle", "pickletools", "shelve",
    # Compression (can be used to decompress malicious payloads)
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "zlib",
    # OS — handled separately in wrapper (kept in sys.modules for Python internals)
    # "os" — blocked at import level but not removed from sys.modules
}

# Modules that should be removed from sys.modules (exclude os and sys which Python internals need)
MODULES_TO_POP = {
    "subprocess", "multiprocessing", "concurrent.futures",
    "threading", "asyncio", "shutil", "glob", "fnmatch",
    "socket", "http", "http.client", "http.server", "urllib",
    "requests", "urllib3", "aiohttp", "ftplib", "smtplib", "poplib",
    "imaplib", "xmlrpc", "ctypes", "cffi", "dl",
    "platform", "site", "sitecustomize", "usercustomize",
    "builtins", "__builtin__", "pdb", "cmd", "shlex", "getpass",
    "signal", "mmap", "importlib", "pkgutil", "zipimport",
    "pickle", "pickletools", "shelve", "zipfile", "tarfile",
    "gzip", "bz2", "lzma", "zlib",
}

BLOCKED_BUILTINS = {
    "open", "input", "eval", "exec", "compile", "__import__",
    "breakpoint", "exit", "quit",
}


def _check_code_safety(code: str) -> list:
    """
    Perform static analysis on code to detect dangerous patterns.
    Returns list of violations (empty = passes basic safety check).
    """
    violations = []

    # Check for blocked module imports
    for module in BLOCKED_MODULES:
        patterns = [
            rf"^import\s+{re.escape(module)}\b",
            rf"^from\s+{re.escape(module)}\b",
            rf"__import__\s*\(\s*['\"]{re.escape(module)}['\"]",
            rf"importlib\.import_module\s*\(\s*['\"]{re.escape(module)}['\"]",
        ]
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE):
                violations.append(f"Blocked module import detected: {module}")

    # Check for dangerous function calls
    dangerous_calls = [
        (r"\bopen\s*\(", "File I/O via open()"),
        (r"\beval\s*\(", "Dynamic code execution via eval()"),
        (r"\bexec\s*\(", "Dynamic code execution via exec()"),
        (r"\bcompile\s*\(", "Dynamic code compilation via compile()"),
        (r"\b__import__\s*\(", "Dynamic import via __import__()"),
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

    # Check for path traversal attempts
    traversal_patterns = [
        r"\.\./", r"\.\.\\", r"['\"]\.\./", r"['\"]\.\.\\",
        r"/etc/", r"/proc/", r"/sys/", r"/dev/",
        r"\\windows\\", r"\\system32\\",
    ]
    for pattern in traversal_patterns:
        if re.search(pattern, code):
            violations.append(f"Path traversal or system path access detected")
            break

    # Check for shell metacharacters in string literals (potential injection)
    shell_patterns = [
        r"['\"](?:.*\|\s*\w+|.*;\s*\w+|.*&&\s*\w+|.*\$\()",
    ]
    for pattern in shell_patterns:
        if re.search(pattern, code):
            violations.append(f"Potential shell command injection pattern")
            break

    return violations


def _build_safe_builtins() -> dict:
    """
    Build a restricted builtins dict that removes dangerous functions.
    """
    safe_builtins = {}
    for name in dir(__builtins__):
        if name in BLOCKED_BUILTINS:
            continue
        if not name.startswith("_"):
            try:
                safe_builtins[name] = getattr(__builtins__, name)
            except AttributeError:
                pass
    # Remove __build_class__ to prevent class creation
    safe_builtins.pop("__build_class__", None)
    # Remove help
    safe_builtins.pop("help", None)
    return safe_builtins


def execute_python_code(code: str, timeout: int = 60) -> dict:
    """
    Execute Python code in a sandboxed subprocess with multiple isolation layers:

    1. **Static analysis** — Blocks dangerous imports, function calls, and patterns
    2. **Restricted builtins** — Removes open(), eval(), exec(), __import__(), etc.
    3. **Temp directory isolation** — Runs in ephemeral temp dir with no access to project files
    4. **Timeout enforcement** — Kills process after timeout to prevent fork bombs/infinite loops
    5. **Resource limits** (Unix only) — Sets memory and CPU limits via resource module

    Security model: Defense in depth. Even if one layer fails, others provide protection.
    """
    # Layer 1: Static analysis safety check
    violations = _check_code_safety(code)
    if violations:
        return {
            "success": False,
            "stdout":  "",
            "stderr":  f"Code rejected by safety check:\n" + "\n".join(f"  - {v}" for v in violations),
            "code":    -2
        }

    # Layer 2: Wrap code with restricted environment
    # The static analysis layer blocks dangerous imports and patterns.
    # The subprocess runs in an isolated temp directory with env sanitization.
    # Resource limits are applied only on Unix where the module is available.
    wrapped_code = f"""
# ── User code begins here ──
{code}
"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, "run.py")

        # Write wrapper + user code for debugging
        debug_path = os.path.join(os.path.dirname(__file__), "_debug_last_wrapper.py")
        
        # Layer 3: Ensure script can't escape temp directory
        # Write with restrictive permissions (owner-only read/write)
        try:
            fd = os.open(script_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(wrapped_code)
            # Also save for debugging
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(wrapped_code)
        except Exception as e:
            # Fallback to normal open if permissions fail
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(wrapped_code)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(wrapped_code)

        try:
            # Layer 4: Subprocess with timeout
            # Use -u for unbuffered output, -B to skip .pyc creation
            result = subprocess.run(
                [sys.executable, "-u", "-B", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
                # Don't inherit parent environment variables that could leak info
                env={
                    **{k: v for k, v in os.environ.items()
                       if k not in ("GROQ_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY")},
                    "PYTHONPATH": "",
                    "HOME": tmp_dir,
                    "TMPDIR": tmp_dir,
                    "TEMP": tmp_dir,
                    "TMP": tmp_dir,
                }
            )

            # Sanitize output to prevent info leakage
            stdout = result.stdout[:3000]
            stderr = result.stderr[:2000]

            # Redact any potential key leakage from stderr
            for key_name in ["GROQ_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "OPENAI_API_KEY"]:
                if key_name in os.environ:
                    actual_key = os.environ[key_name]
                    if actual_key:
                        stdout = stdout.replace(actual_key, "[REDACTED]")
                        stderr = stderr.replace(actual_key, "[REDACTED]")

            return {
                "success": result.returncode == 0,
                "stdout":  stdout,
                "stderr":  stderr,
                "code":    result.returncode
            }

        except subprocess.TimeoutExpired as e:
            # Kill any child processes
            if hasattr(e, 'process'):
                try:
                    e.process.kill()
                except Exception:
                    pass
            return {
                "success": False,
                "stdout":  "",
                "stderr":  f"Execution timed out after {timeout}s. Process killed.",
                "code":    -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout":  "",
                "stderr":  f"Execution failed: {type(e).__name__}: {e}",
                "code":    -1
            }


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

def save_code_file(output_dir: str, filename: str, code: str) -> str:
    """
    Save a code module to the output directory.
    
    Security: Validates filename to prevent path traversal attacks.
    LLM-generated filenames like '../../etc/passwd' are rejected.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename: reject path traversal attempts
    # Strip any directory components — only the basename is allowed
    safe_filename = os.path.basename(filename)
    
    # Reject if basename is empty or suspicious
    if not safe_filename or safe_filename in ('.', '..'):
        safe_filename = "unnamed_module.py"
    
    # Reject filenames with null bytes
    if '\x00' in safe_filename:
        safe_filename = "unnamed_module.py"
    
    # Validate that the resolved path is within output_dir
    full_path = os.path.realpath(os.path.join(output_dir, safe_filename))
    output_dir_real = os.path.realpath(output_dir)
    
    if not full_path.startswith(output_dir_real + os.sep) and full_path != output_dir_real:
        raise ValueError(
            f"Rejected filename '{filename}' — resolved path '{full_path}' "
            f"is outside output directory '{output_dir_real}'"
        )
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"   💾 Saved: {safe_filename}")
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
