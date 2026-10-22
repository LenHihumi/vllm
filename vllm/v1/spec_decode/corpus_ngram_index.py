# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static corpus n-gram index for speculative decoding.

This module builds and stores a continuation table over a fixed token corpus: for
every ``order``-token context seen in the corpus, we keep the token that most often
follows it. Drafting then walks the table token by token, which is cheap enough to
run on the critical path (one dict lookup per drafted token).

The corpus is *data*, not a model: it is usually a benchmark's reference answers, or
the verified outputs a service has accumulated. The index therefore grows by
appending documents (``NgramTable.add_documents``) instead of being retrained.

Example:
    >>> table = NgramTable.build([[1, 2, 3, 4, 5, 6]], order=3)
    >>> table.propose([1, 2, 3], k=2)
    [4, 5]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence

import numpy as np

_DTYPE = np.int32
DEFAULT_ORDER = 5


def _context_key(context: Sequence[int]) -> bytes:
    """Pack a token context into bytes so it can key a dict cheaply."""
    return np.asarray(context, dtype=_DTYPE).tobytes()


class NgramTable:
    """A corpus n-gram continuation table.

    Internally we keep, per context, the observed successor counts. Counts (rather
    than a single argmax) are what make incremental growth meaningful: the same
    context may be followed by different tokens in different documents, and the
    majority successor can change as the corpus grows.
    """

    def __init__(self, order: int = DEFAULT_ORDER) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.order = order
        self._counts: dict[bytes, dict[int, int]] = {}
        self._best: dict[bytes, int] = {}
        self.num_tokens = 0

    def __len__(self) -> int:
        """Number of distinct contexts in the table."""
        return len(self._counts)

    @classmethod
    def build(
        cls, documents: Iterable[Sequence[int]], order: int = DEFAULT_ORDER
    ) -> "NgramTable":
        """Build a table from an iterable of token-id sequences."""
        table = cls(order=order)
        table.add_documents(documents)
        return table

    def add_documents(self, documents: Iterable[Sequence[int]]) -> int:
        """Add token sequences to the table; returns the number of tokens added."""
        added = 0
        order = self.order
        for doc in documents:
            tokens = [int(t) for t in doc]
            added += len(tokens)
            for i in range(len(tokens) - order):
                key = _context_key(tokens[i : i + order])
                nxt = tokens[i + order]
                per_key = self._counts.get(key)
                if per_key is None:
                    self._counts[key] = {nxt: 1}
                    self._best[key] = nxt
                    continue
                c = per_key.get(nxt, 0) + 1
                per_key[nxt] = c
                if c > per_key[self._best[key]]:
                    self._best[key] = nxt
        self.num_tokens += added
        return added

    def successor(self, context: Sequence[int]) -> int | None:
        """Return the most frequent successor of ``context``, or ``None``."""
        if len(context) < self.order:
            return None
        return self._best.get(_context_key(context[-self.order :]))

    def propose(self, context: Sequence[int], k: int) -> list[int]:
        """Draft up to ``k`` tokens by walking the table from ``context``.

        The walk is autoregressive in the corpus table itself: each drafted token
        extends the context window, so a match of length ``k`` requires the corpus
        to contain that whole continuation. The walk stops at the first context the
        corpus does not cover, which keeps false drafts cheap.
        """
        if k <= 0 or len(context) < self.order:
            return []
        window = [int(t) for t in context[-self.order :]]
        draft: list[int] = []
        for _ in range(k):
            nxt = self._best.get(_context_key(window))
            if nxt is None:
                break
            draft.append(nxt)
            window = window[1:] + [nxt]
        return draft

    def save(self, path: str) -> None:
        """Persist the table as a compressed ``.npz`` archive."""
        keys = np.empty((len(self._best), self.order), dtype=_DTYPE)
        values = np.empty((len(self._best),), dtype=_DTYPE)
        for i, (key, nxt) in enumerate(self._best.items()):
            keys[i] = np.frombuffer(key, dtype=_DTYPE)
            values[i] = nxt
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            path, keys=keys, values=values, order=np.int64(self.order),
            num_tokens=np.int64(self.num_tokens),
        )

    @classmethod
    def load(cls, path: str) -> "NgramTable":
        """Load a table written by :meth:`save`."""
        with np.load(path) as data:
            order = int(data["order"])
            table = cls(order=order)
            keys = data["keys"]
            values = data["values"]
            best = table._best
            counts = table._counts
            for i in range(keys.shape[0]):
                key = keys[i].tobytes()
                nxt = int(values[i])
                best[key] = nxt
                counts[key] = {nxt: 1}
            table.num_tokens = int(data["num_tokens"]) if "num_tokens" in data else 0
        return table


def _tokenize_jsonl(
    path: str, tokenizer: str, text_field: str, max_tokens: int
) -> list[list[int]]:
    """Tokenize a JSONL file of documents into token-id sequences."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    documents: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get(text_field)
            if not text:
                continue
            ids = tok(text, add_special_tokens=False, truncation=True,
                      max_length=max_tokens)["input_ids"]
            if ids:
                documents.append(ids)
    return documents


def _cmd_build(args: argparse.Namespace) -> None:
    documents = _tokenize_jsonl(
        args.answers, args.tokenizer, args.text_field, args.max_tokens
    )
    table = NgramTable.build(documents, order=args.order)
    table.save(args.out)
    print(
        f"built order={table.order} contexts={len(table)} "
        f"tokens={table.num_tokens} docs={len(documents)} -> {args.out}"
    )


def _cmd_info(args: argparse.Namespace) -> None:
    table = NgramTable.load(args.index)
    print(
        f"order={table.order} contexts={len(table)} tokens={table.num_tokens}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point for building and inspecting an index."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build an index from a JSONL corpus")
    build.add_argument("--answers", required=True,
                       help="JSONL file, one document per line")
    build.add_argument("--tokenizer", required=True,
                       help="tokenizer name or local path of the target model")
    build.add_argument("--out", required=True, help="output .npz path")
    build.add_argument("--order", type=int, default=DEFAULT_ORDER)
    build.add_argument("--text-field", default="text")
    build.add_argument("--max-tokens", type=int, default=1024,
                       help="truncate each document to this many tokens")
    build.set_defaults(func=_cmd_build)

    info = sub.add_parser("info", help="print index statistics")
    info.add_argument("index")
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
