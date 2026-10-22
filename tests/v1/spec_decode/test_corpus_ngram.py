# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the corpus n-gram proposer.

These tests only exercise the index and the draft walk; they do not start an engine,
so they run on CPU without a GPU.
"""

import os
import time

import numpy as np
import pytest

from vllm.v1.spec_decode.corpus_ngram_index import NgramTable
from vllm.v1.spec_decode.corpus_ngram_proposer import CorpusNgramProposer


class _StubModelConfig:
    max_model_len = 64


class _StubSpecConfig:
    num_speculative_tokens = 4


class _StubConfig:
    model_config = _StubModelConfig()
    speculative_config = _StubSpecConfig()


def _build_table():
    # "1 2 3 4 5" repeated, plus a distractor that must lose the majority vote.
    doc = [1, 2, 3, 4, 5] * 3 + [1, 2, 3, 9, 5]
    return NgramTable.build([doc], order=3)


def test_build_and_successor():
    table = _build_table()
    assert table.order == 3
    # (1, 2, 3) is followed by 4 three times and by 9 once.
    assert table.successor([1, 2, 3]) == 4
    # Unseen context has no successor.
    assert table.successor([7, 7, 7]) is None


def test_propose_walks_multiple_tokens():
    table = _build_table()
    assert table.propose([1, 2, 3], k=4) == [4, 5, 1, 2]
    # k=0 and short contexts produce no draft.
    assert table.propose([1, 2, 3], k=0) == []
    assert table.propose([1, 2], k=4) == []


def test_propose_stops_at_uncovered_context():
    table = NgramTable.build([[10, 11, 12, 13, 99]], order=3)
    # (10, 11, 12) -> 13 then (11, 12, 13) -> 99, at which point the corpus has
    # nothing more to say about (12, 13, 99).
    assert table.propose([10, 11, 12], k=8) == [13, 99]


def test_add_documents_flips_majority():
    table = NgramTable.build([[1, 2, 3, 4]], order=3)
    assert table.successor([1, 2, 3]) == 4
    table.add_documents([[1, 2, 3, 7], [1, 2, 3, 7]])
    assert table.successor([1, 2, 3]) == 7


def test_save_load_roundtrip(tmp_path):
    table = _build_table()
    path = str(tmp_path / "index.npz")
    table.save(path)
    loaded = NgramTable.load(path)
    assert loaded.order == table.order
    assert len(loaded) == len(table)
    assert loaded.num_tokens == table.num_tokens
    assert loaded.propose([1, 2, 3], k=4) == table.propose([1, 2, 3], k=4)


def test_empty_corpus_yields_no_drafts():
    table = NgramTable.build([], order=4)
    assert len(table) == 0
    assert table.propose([1, 2, 3, 4], k=8) == []


def test_proposer_batch_contract(monkeypatch):
    monkeypatch.delenv("VLLM_CORPUS_NGRAM_PATH", raising=False)
    proposer = CorpusNgramProposer(_StubConfig())  # type: ignore[arg-type]
    proposer.add_documents([[1, 2, 3, 4, 5, 6, 7, 8]])

    token_ids_cpu = np.zeros((2, 64), dtype=np.int32)
    token_ids_cpu[0, :4] = [1, 2, 3, 4]
    token_ids_cpu[1, :3] = [1, 2, 3]
    drafts = proposer.propose(
        sampled_token_ids=[[5], []],
        num_tokens_no_spec=np.array([4, 3], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    # Request 0 has the order-5 context (1, 2, 3, 4, ...) available.
    assert len(drafts) == 2
    assert drafts[1] == []  # partial prefill: skipped
    assert all(len(d) <= proposer.num_speculative_tokens for d in drafts)


def test_proposer_respects_max_model_len(monkeypatch):
    monkeypatch.delenv("VLLM_CORPUS_NGRAM_PATH", raising=False)
    proposer = CorpusNgramProposer(_StubConfig())  # type: ignore[arg-type]
    proposer.add_documents([[1, 2, 3, 4, 5]])
    token_ids_cpu = np.zeros((1, 64), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 4, 5]
    # num_tokens == max_model_len: nothing may be drafted.
    drafts = proposer.propose(
        sampled_token_ids=[[6]],
        num_tokens_no_spec=np.array([64], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert drafts == [[]]


def test_proposer_reloads_updated_index(tmp_path, monkeypatch):
    path = str(tmp_path / "index.npz")
    NgramTable.build([[1, 2, 3, 4, 5]], order=3).save(path)
    monkeypatch.setenv("VLLM_CORPUS_NGRAM_PATH", path)
    proposer = CorpusNgramProposer(_StubConfig())  # type: ignore[arg-type]
    token_ids_cpu = np.zeros((1, 64), dtype=np.int32)
    token_ids_cpu[0, :3] = [1, 2, 3]
    before = proposer.propose([[4]], np.array([3], dtype=np.int32), token_ids_cpu)[0]

    # Publish a new index; the next step must see it without a restart.
    NgramTable.build([[1, 2, 3, 7, 8]], order=3).save(path)
    newer = time.time() + 10
    os.utime(path, (newer, newer))
    after = proposer.propose([[4]], np.array([3], dtype=np.int32), token_ids_cpu)[0]
    assert before != after
    assert after[0] == 7


@pytest.mark.parametrize("order", [1, 2, 5, 8])
def test_order_variants(order):
    doc = list(range(20))
    table = NgramTable.build([doc], order=order)
    context = list(range(order))
    assert table.propose(context, k=2)[:2] == [order, order + 1]
