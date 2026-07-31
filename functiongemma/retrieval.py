"""Tool retrieval  —  making function calling scale past a full context window.

The whole pipeline so far assumes every tool fits in the prompt. Real
deployments don't: a phone assistant can expose *hundreds* of tools (every app
registers its own), and a 2B edge model has a small context window. Stuffing 300
tool schemas into the prompt is slow, blows the budget, and *lowers* accuracy —
the model has more wrong options to pick from.

The industrial fix is **tool retrieval** (a.k.a. tool RAG): given the request,
retrieve the top-k relevant tools first, then prompt the model with only those.
This module is a dependency-free BM25 retriever over the tool catalogue —
lexical, deterministic, and tiny enough to run on-device before the model does.

    request ── BM25 over {name, description, params} ──> top-k tools ──> prompt

BM25 (not plain keyword overlap) matters here: it rewards rare, discriminative
terms ("weather", "timer") and down-weights common ones, and it saturates term
frequency so a tool that repeats a word ten times doesn't dominate. With a real
system you'd swap the lexical index for an embedding index behind the same
``select`` interface — the retriever is a seam, exactly like the model backend.
"""

import math
import re
from collections import Counter

from .tools import TOOLS

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _WORD.findall(text.lower())


def _tool_text(tool):
    """The searchable text of a tool: name (split), description, param names."""
    parts = [tool["name"].replace("_", " "), tool.get("description", "")]
    parts.extend(tool.get("parameters", {}).keys())
    return " ".join(parts)


class ToolRetriever:
    """BM25 index over a tool catalogue. Build once, ``select`` per request."""

    def __init__(self, tools=TOOLS, k1=1.5, b=0.75):
        self.tools = list(tools)
        self.k1 = k1
        self.b = b
        self._docs = [Counter(_tokens(_tool_text(t))) for t in self.tools]
        self._doc_len = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._doc_len) / len(self._docs)) if self._docs else 0.0
        self._idf = self._compute_idf()

    def _compute_idf(self):
        """Standard BM25 idf with the +1 floor so scores stay non-negative."""
        n = len(self._docs)
        df = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        return {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, request, doc_idx):
        """BM25 score of one tool document against the request."""
        doc = self._docs[doc_idx]
        length = self._doc_len[doc_idx]
        s = 0.0
        for term in _tokens(request):
            if term not in doc:
                continue
            tf = doc[term]
            denom = tf + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
            s += self._idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
        return s

    def select(self, request, k=8):
        """Return the top-k most relevant tools for ``request``.

        Deterministic: ties break by catalogue order. If nothing matches (an
        empty or off-topic request) we return the first ``k`` tools rather than
        an empty set, so the downstream prompt is never toolless — the model can
        still abstain, which is the correct outcome for an off-topic request.
        """
        if k >= len(self.tools):
            return list(self.tools)
        scored = [(self.score(request, i), i) for i in range(len(self.tools))]
        if not any(s > 0 for s, _ in scored):
            return self.tools[:k]
        # Sort by score desc, then original index asc (stable, deterministic).
        scored.sort(key=lambda si: (-si[0], si[1]))
        return [self.tools[i] for _, i in scored[:k]]
