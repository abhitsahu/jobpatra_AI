"""Keyword extractor — pull important terms from plain text.

This module has ONE responsibility: identify meaningful tokens (words and
short multi-word phrases) from any text — resume or job description.

It does NOT:
  - classify or compare keywords
  - score keywords
  - call AI or external services
  - look up a skills reference list (that is skill_extractor's job)

Algorithm:
  1. Tokenise the text into words using a simple regex word-boundary split.
  2. Filter out a curated stop-word list (function words, prepositions,
     common verbs, etc.) that carry no information value.
  3. Deduplicate while preserving the first-seen order.
  4. Return a list of unique, non-trivial tokens.

All functions are pure. No I/O. No state. No FastAPI imports.
"""

import re

# ---------------------------------------------------------------------------
# Stop-words — words that add no keyword value
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        # Articles / determiners
        "a", "an", "the",
        # Prepositions
        "at", "by", "for", "from", "in", "into", "of", "on", "onto",
        "out", "over", "to", "under", "up", "with", "about", "as",
        "between", "through", "during", "before", "after", "above",
        "below", "within", "without",
        # Conjunctions
        "and", "but", "or", "nor", "so", "yet", "both", "either",
        "neither", "not", "although", "because", "since", "unless",
        "while", "if", "than",
        # Pronouns
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
        "his", "she", "her", "it", "its", "they", "them", "their",
        "this", "that", "these", "those",
        # Common verbs with no keyword value
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "must", "can", "could",
        "need", "used",
        # Resume filler words
        "responsible", "responsibilities", "role", "position",
        "experience", "work", "worked", "working",
        "team", "teams", "member", "members",
        "strong", "good", "excellent", "great", "ability", "able",
        "skills", "knowledge", "understanding", "proficient",
        "various", "multiple", "different", "other", "also", "including",
        "well", "using", "use", "make", "made", "help", "helped",
        "ensure", "ensured", "support", "supported", "manage", "managed",
        "lead", "led", "build", "built", "create", "created",
        "develop", "developed", "implement", "implemented",
        "maintain", "maintained", "improve", "improved",
        "provide", "provided", "review", "reviewed",
    }
)

# Tokenisation: keep alphanumeric runs and dots (for "Node.js", "ASP.NET")
_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9.#+\-_]*")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(text: str) -> list[str]:
    """Extract unique, meaningful keywords from plain text.

    Tokenises ``text``, removes stop-words and very short tokens (≤ 1 char),
    deduplicates case-insensitively while preserving the first-seen casing,
    and returns the result in first-seen order.

    Args:
        text: Any plain text — resume section body, job description, etc.

    Returns:
        Ordered list of unique keyword strings.  May be empty if the text
        contains only stop-words or punctuation.
    """
    tokens = _TOKEN_RE.findall(text)
    seen_lower: set[str] = set()
    keywords: list[str] = []

    for token in tokens:
        if len(token) <= 1:
            continue
        if token.lower() in _STOP_WORDS:
            continue
        if token.lower() in seen_lower:
            continue
        seen_lower.add(token.lower())
        keywords.append(token)

    return keywords
