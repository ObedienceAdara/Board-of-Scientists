"""
agent_registry.py — Canonical agent identity registry.

THE BUG THIS FILE FIXES
------------------------
Previously, agents addressed each other on the message board using human
display names as free-text strings — e.g. the Architect posted messages to
recipient="Senior ML Engineer", but the Engineer looked up its own inbox with
get_messages_for(state, "engineer"). Because those two strings are not equal
("ENGINEER" != "SENIOR ML ENGINEER"), the lookup silently returned nothing.
The Engineer never saw the Architect's directives, the Reviewer's follow-ups,
or the Experiment Engineer's bug reports. The whole "team communication"
concept was broken by a one-line string mismatch.

THE FIX
-------
Every agent is now addressed everywhere — message routing, evaluation
feedback, revision tracking — by a single, stable, lowercase key (e.g.
"engineer"). Display names exist ONLY for human-readable output (printed
logs, the message_board dump, the PDF report) and are looked up from this
one registry, so the routing key and the pretty name can never drift apart
again.
"""

from typing import Final

# ── Canonical keys ────────────────────────────────────────────────
# These are the ONLY strings that should ever be used to route a message,
# key an evaluation, or key a revision count. Never use a display name for
# routing — always go through normalize_key() first if a string of unknown
# provenance shows up.
CRO: Final[str] = "cro"
ANALYST: Final[str] = "analyst"
THEORIST: Final[str] = "theorist"
ARCHITECT: Final[str] = "architect"
ENGINEER: Final[str] = "engineer"
REVIEWER: Final[str] = "reviewer"
EXPERIMENT: Final[str] = "experiment"
WRITER: Final[str] = "writer"
ALL: Final[str] = "all"

# ── Human-readable display names (for logs / reports only) ───────
DISPLAY_NAMES: Final[dict] = {
    CRO:        "Dr. Aria Chen — Chief Research Officer",
    ANALYST:    "Dr. Marcus Webb — Paper Analyst",
    THEORIST:   "Prof. Elena Vasquez — Theorist",
    ARCHITECT:  "Dr. James Okafor — ML Architect",
    ENGINEER:   "Dr. Kai Nakamura — Senior ML Engineer",
    REVIEWER:   "Dr. Priya Sharma — Code Reviewer",
    EXPERIMENT: "Dr. Santiago Reyes — Experiment Engineer",
    WRITER:     "Dr. Amara Osei — Technical Writer",
    ALL:        "ALL",
}

VALID_KEYS: Final[frozenset] = frozenset(DISPLAY_NAMES.keys())

# Short role labels used inside prompts (e.g. "You have just received a
# deliverable from your {role}").
ROLE_LABELS: Final[dict] = {
    CRO:        "Chief Research Officer",
    ANALYST:    "Paper Analyst",
    THEORIST:   "Theorist",
    ARCHITECT:  "ML Architect",
    ENGINEER:   "Senior ML Engineer",
    REVIEWER:   "Code Reviewer",
    EXPERIMENT: "Experiment Engineer",
    WRITER:     "Technical Writer",
}


def display_name(key: str) -> str:
    """Human-readable name for a canonical agent key. Falls back to the raw key."""
    return DISPLAY_NAMES.get(normalize_key(key), key or "unknown")


def role_label(key: str) -> str:
    """Short role label for a canonical agent key, for embedding in prompts."""
    return ROLE_LABELS.get(normalize_key(key), key or "unknown")


def normalize_key(key: str) -> str:
    """
    Normalize any agent identifier to its canonical key.

    Accepts canonical keys (any case), or — defensively — legacy display
    names, so any old saved state, hand-written test fixtures, or a stray
    LLM output that used a display name instead of a key doesn't silently
    disappear into a routing black hole the way it used to.
    """
    if not key:
        return ""
    k = key.strip().lower()
    if k in VALID_KEYS:
        return k
    for canonical, display in DISPLAY_NAMES.items():
        if k == display.lower() or k in display.lower() or display.lower() in k:
            return canonical
    for canonical, role in ROLE_LABELS.items():
        if k == role.lower():
            return canonical
    return k
