# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline example: speculative decoding from a corpus n-gram index.

The corpus here is a handful of reference solutions, which stands in for the case
this proposer targets: text you already hold before the first request arrives.

Run:
    python examples/features/corpus_ngram_offline.py
"""

import json
import os
import tempfile

from vllm import LLM, SamplingParams
from vllm.v1.spec_decode.corpus_ngram_index import NgramTable

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
]

PROMPTS = [
    "Complete the Python function. Output only code.\n\ndef add(a, b):\n",
    "Complete the Python function. Output only code.\n\ndef mul(a, b):\n",
]


def build_index(workdir: str) -> str:
    """Tokenize the corpus with the target tokenizer and save an index."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    documents = [tok(text, add_special_tokens=False)["input_ids"] for text in CORPUS]
    table = NgramTable.build(documents, order=3)
    path = os.path.join(workdir, "corpus_ngram.npz")
    table.save(path)
    print(f"index: contexts={len(table)} tokens={table.num_tokens} -> {path}")
    return path


def main() -> None:
    """Score a couple of prompts with and without the corpus index."""
    with tempfile.TemporaryDirectory() as workdir:
        index_path = build_index(workdir)
        os.environ["VLLM_CORPUS_NGRAM_PATH"] = index_path

        spec = {
            "method": "custom_class",
            "model": (
                "vllm.v1.spec_decode.corpus_ngram_proposer.CorpusNgramProposer"
            ),
            "num_speculative_tokens": 4,
        }
        prompts = [
            {"role": "user", "content": p} for p in PROMPTS
        ]
        params = SamplingParams(temperature=0.0, max_tokens=32)

        for label, speculative_config in (("baseline", None), ("corpus", spec)):
            llm = LLM(model=MODEL, speculative_config=speculative_config)
            outputs = llm.chat(prompts, params)
            print(f"--- {label} ---")
            for out in outputs:
                print(json.dumps(out.outputs[0].text))
            del llm


if __name__ == "__main__":
    main()
