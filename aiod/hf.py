"""Hugging Face link parsing (pure, no I/O).

Kept separate from cli.py so the URL-variant matrix is unit-testable.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Reserved leading path segments on huggingface.co that are NOT model repos.
# We reject these loudly rather than silently producing a bad repo id.
_RESERVED = {
    "datasets",
    "spaces",
    "models",
    "organizations",
    "settings",
    "join",
    "login",
    "logout",
    "blog",
    "docs",
    "pricing",
    "api",
    "new",
    "search",
}

_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}

# A single repo-id segment: alphanumerics plus . _ -
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_repo_id(link: str) -> str:
    """Normalize a Hugging Face model reference to ``'org/repo'``.

    Accepts:
      * full URLs: ``https://huggingface.co/org/repo`` (with ``/tree/...``,
        ``/resolve/...``, ``?query`` or ``#fragment`` suffixes),
      * bare ``org/repo``.

    Raises ``ValueError`` on unparseable input and on Spaces/datasets (or any
    other non-model namespace) URLs — never returns a bad id silently.
    """
    if not isinstance(link, str) or not link.strip():
        raise ValueError(f"empty Hugging Face reference: {link!r}")

    s = link.strip()

    if "://" in s or s.lower().startswith(tuple(h + "/" for h in _HF_HOSTS)):
        candidate = s if "://" in s else "https://" + s
        parsed = urlparse(candidate)
        host = parsed.netloc.lower()
        if host not in _HF_HOSTS:
            raise ValueError(f"not a Hugging Face URL (host {host!r}): {link!r}")
        path = parsed.path
    else:
        # Bare form; drop any query/fragment that snuck in.
        path = s.split("#", 1)[0].split("?", 1)[0]

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"could not parse 'org/repo' from {link!r}")

    org, repo = parts[0], parts[1]

    if org.lower() in _RESERVED:
        raise ValueError(
            f"not a model repo (looks like a '{org}' page, not a model): {link!r}"
        )

    # Strip a revision pin (org/repo@sha) if present.
    repo = repo.split("@", 1)[0]

    if not _SEGMENT.fullmatch(org) or not _SEGMENT.fullmatch(repo):
        raise ValueError(f"invalid org/repo in {link!r}")

    return f"{org}/{repo}"
