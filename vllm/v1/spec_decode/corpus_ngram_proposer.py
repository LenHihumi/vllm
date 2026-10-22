# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Corpus n-gram proposer: speculative drafting from a fixed token corpus.

Most n-gram proposers draft from text the request already contains (the prompt, or
earlier output of the same request). This one drafts from an *external* corpus that
is known before the request arrives --- a benchmark's reference answers, a codebase,
or the verified outputs a service has collected --- so it helps most on the first
visit, when the model has produced nothing yet.

Usage (vLLM v1):

.. code-block:: python

    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        speculative_config={
            "method": "custom_class",
            "model": "vllm.v1.spec_decode.corpus_ngram_proposer.CorpusNgramProposer",
            "num_speculative_tokens": 8,
        },
    )

The corpus is configured through environment variables, so no change to
``SpeculativeConfig`` is needed:

- ``VLLM_CORPUS_NGRAM_PATH``: index file built by
  ``python -m vllm.v1.spec_decode.corpus_ngram_index build``. Optional: without it
  the proposer starts with an empty table and can be grown in process through
  :meth:`CorpusNgramProposer.add_documents`.
- ``VLLM_CORPUS_NGRAM_ORDER``: context order used when no index file is given
  (default 5).

Because the table is static, an updated index file is picked up between decoding
steps: rewriting the file is enough to publish newly verified answers to a running
engine, which is the "grow" step of an evaluation service.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.v1.spec_decode.corpus_ngram_index import DEFAULT_ORDER, NgramTable

if TYPE_CHECKING:
    from vllm.config import VllmConfig

try:  # keep the module importable outside a vLLM install (unit tests, tooling)
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # pragma: no cover - only hit without vLLM on the path
    import logging

    logger = logging.getLogger(__name__)

_ENV_INDEX_PATH = "VLLM_CORPUS_NGRAM_PATH"
_ENV_ORDER = "VLLM_CORPUS_NGRAM_ORDER"


class CorpusNgramProposer:
    """Propose draft tokens by walking an n-gram table built from a corpus.

    The constructor signature (``VllmConfig``) and ``propose`` signature match the
    ``custom_class`` speculative-decoding contract, so this class is used directly
    as ``speculative_config["model"]``.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        spec = vllm_config.speculative_config
        if spec is None:
            raise ValueError(
                "CorpusNgramProposer requires a speculative config; pass "
                "speculative_config={...} to LLM()."
            )
        self.num_speculative_tokens = spec.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        self._index_path = os.environ.get(_ENV_INDEX_PATH)
        self._order = int(os.environ.get(_ENV_ORDER, "0")) or DEFAULT_ORDER
        self._mtime: float | None = None
        self.table = self._load_table()
        logger.info(
            "CorpusNgramProposer: order=%d contexts=%d tokens=%d path=%s",
            self.table.order,
            len(self.table),
            self.table.num_tokens,
            self._index_path or "<empty>",
        )

    def _load_table(self) -> NgramTable:
        """Load the configured index, or start from an empty table."""
        if self._index_path and os.path.exists(self._index_path):
            self._mtime = os.stat(self._index_path).st_mtime
            return NgramTable.load(self._index_path)
        if self._index_path:
            logger.warning(
                "CorpusNgramProposer: index path %s does not exist; starting empty",
                self._index_path,
            )
        return NgramTable.build([], order=self._order)

    def _refresh_index(self) -> None:
        """Reload the index if the file on disk is newer than what we hold."""
        if not self._index_path:
            return
        try:
            mtime = os.stat(self._index_path).st_mtime
        except OSError:
            return
        if self._mtime is not None and mtime <= self._mtime:
            return
        try:
            self.table = NgramTable.load(self._index_path)
        except Exception as exc:  # a half-written file must not kill decoding
            logger.warning(
                "CorpusNgramProposer: reload of %s failed (%s); keeping old index",
                self._index_path,
                exc,
            )
            return
        self._mtime = mtime
        logger.info(
            "CorpusNgramProposer: reloaded index, contexts=%d tokens=%d",
            len(self.table),
            self.table.num_tokens,
        )

    def add_documents(self, documents: Sequence[Sequence[int]]) -> int:
        """Grow the in-memory table with more token sequences.

        This is the in-process entry point for a service that verifies answers as
        they arrive: no restart and no file rewrite is needed.
        """
        return self.table.add_documents(documents)

    def save_index(self, path: str | None = None) -> str:
        """Persist the current table, defaulting to the configured path."""
        target = path or self._index_path
        if not target:
            raise ValueError(
                "No path given and VLLM_CORPUS_NGRAM_PATH is unset; "
                "pass an explicit path to save_index()."
            )
        self.table.save(target)
        self._mtime = os.stat(target).st_mtime
        return target

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, Any] | None = None,
    ) -> list[list[int]]:
        """Draft tokens for every request in the batch.

        Args:
            sampled_token_ids: tokens sampled for each request in this step; an
                empty list marks a partial prefill, which we skip.
            num_tokens_no_spec: number of tokens each request already holds,
                including speculative ones.
            token_ids_cpu: host-side token ids, indexed by request position.
            slot_mappings: unused, accepted for interface compatibility.

        Returns:
            One list of draft token ids per request, each at most
            ``num_speculative_tokens`` long.
        """
        del slot_mappings  # the corpus table does not depend on KV layout
        self._refresh_index()
        order = self.table.order
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len or num_tokens < order:
                drafts.append([])
                continue
            # Same budget rule as other proposers: never draft past the limit.
            k = min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
            context = token_ids_cpu[i, num_tokens - order : num_tokens]
            drafts.append(self.table.propose(context.tolist(), k))
        return drafts

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """No model to load: the proposer is a lookup table."""
