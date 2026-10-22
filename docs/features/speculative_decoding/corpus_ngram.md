# Corpus n-gram speculative decoding

Corpus n-gram decoding drafts tokens from a **fixed token corpus** instead of from
the request itself. The corpus is anything you already hold in machine-readable
form: a benchmark's reference answers, a codebase, or the verified outputs a
service has accumulated. It is compiled once into a small continuation table, which
the proposer walks token by token during decoding.

This complements the built-in [`ngram`](n_gram.md) method, which drafts from the
prompt and earlier output of the *same request*: that method is strong on repetitive
long-form text but contributes little on the first token of a fresh request against a
new prompt. A corpus index is available from the first step, which is what an
evaluation service or a fixed-domain deployment wants.

## When to use it

| Situation | Suggested method |
|---|---|
| One prompt, long repetitive output (summarisation, copy-heavy tasks) | `ngram` |
| Many short requests against a fixed corpus (benchmark scoring, doc-grounded QA) | `corpus_ngram` |
| You have a trained draft head | `eagle` / `draft_model` |

The gain comes from token overlap between the corpus and the model's output. If the
corpus shares no surface form with the answers being generated, the index drafts
nothing and costs a few dictionary lookups per step (measured: an empty index sits
slightly *below* unaccelerated decoding, because the proposer is called every step).

## Building an index

The corpus is tokenized with the **target model's tokenizer**, because the draft
tokens must be in the target vocabulary:

```bash
# answers.jsonl: one document per line, e.g. {"text": "def add(a, b):\n    return a + b"}
python -m vllm.v1.spec_decode.corpus_ngram_index build \
    --answers answers.jsonl \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
    --order 5 \
    --out corpus_ngram.npz

python -m vllm.v1.spec_decode.corpus_ngram_index info corpus_ngram.npz
```

`--order` is the context length of the table: larger orders need longer exact matches
(cleaner drafts, more misses), smaller orders draft more but are likelier to be
rejected. 5 is a reasonable default; sweeping it on your own corpus is cheap.

## Serving

Point the engine at the index with environment variables, and select the proposer
through the `custom_class` method:

```bash
export VLLM_CORPUS_NGRAM_PATH=/path/to/corpus_ngram.npz
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --speculative-config '{"method": "custom_class", "model": "vllm.v1.spec_decode.corpus_ngram_proposer.CorpusNgramProposer", "num_speculative_tokens": 8}'
```

Python API:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    speculative_config={
        "method": "custom_class",
        "model": "vllm.v1.spec_decode.corpus_ngram_proposer.CorpusNgramProposer",
        "num_speculative_tokens": 8,
    },
)
```

`VLLM_CORPUS_NGRAM_ORDER` overrides the table order when no index file is present.
Without either variable the proposer starts with an empty table.

## Growing the corpus while the engine runs

An evaluation service verifies submissions continuously. Two ways to publish newly
verified answers, neither of which requires a restart:

- **Rewrite the index file.** The proposer compares the file's mtime between
  decoding steps and reloads it, so `build` + save is enough. A partially written
  file is ignored, and the previous index stays in use.
- **In process.** If you drive the proposer yourself, call
  `proposer.add_documents(list_of_token_id_lists)`; the table updates in place.

Reloading keeps the old table until the new one loads cleanly, so publishing never
interrupts decoding.

## Cost and limits

- Index size: one entry per distinct context, ~12 bytes plus dictionary overhead.
  1M corpus tokens at order 5 gives roughly 1M entries; a 5,000-problem corpus with
  ~200-token answers is a few tens of MB, and loading takes a second.
- Draft cost is one dict lookup per drafted token, on CPU, per request per step.
- The table stores only the majority successor per context, so a context followed
  equally often by two tokens drafts one of them. If that token is rejected, the
  engine accepts a shorter prefix, which is exactly the usual speculative-decoding
  contract: correctness never depends on the draft being right.
- Losslessness is unaffected: proposals are verified by the target model, and only
  tokens the target would itself have chosen are accepted.

## Correctness notes

- Drafts never exceed `num_speculative_tokens` or run past `max_model_len`.
- Partial prefill steps (no sampled tokens yet) are skipped.
- Dictionaries are read-only during a step, so tensor-parallel workers can each
  hold their own copy of the same index file.

### Greedy output can differ from an unaccelerated run by a few tokens

Verification guarantees that only tokens the target model itself would sample are
accepted, so the *distribution* is unchanged --- but greedy decoding is not
bit-identical across engine layouts. The verify pass scores several positions in one
batched forward, and bfloat16 reductions differ in the last bits from a
single-position decode; where two candidates are nearly tied, the argmax can flip.
This is a property of speculative decoding in general, not of this proposer: it
shows up as a handful of flipped tokens, most often at points where the model is
itself uncertain (a branch between two plausible statements), and everything after
the flip follows deterministically.

Measure it before you trust a throughput number. On a small smoke corpus with
32-token greedy completions, this proposer produced identical output on every prompt
where its drafts matched, and one prompt flipped at a high-uncertainty branch
(`return "Error: ..."` vs `raise ValueError(...)`); the same arm re-run was
reproducible. If you need exact token-for-token agreement with a vanilla baseline,
score with `temperature=0` and compare token ids, and treat a small number of
divergences at near-ties as expected rather than as a defect.
