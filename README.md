<div align="center">

# GuidedBench

**Guideline-grounded evaluation for LLM jailbreak methods**

[Paper](https://openreview.net/forum?id=ZVg8y3ibyM) ·
[Project page](https://sproutnan.github.io/GuidedBench/) ·
[Dataset](https://huggingface.co/datasets/HRXUST/GuidedBench)

</div>

GuidedBench evaluates whether a victim-model response fulfills the specific
harmful objective in a question. Every case includes verified, case-specific
entity and action guidelines, turning a holistic harmfulness judgment into a
set of explicit content checks.

The benchmark contains 200 English cases across 20 topics: a 180-case core set
and a 20-case, vendor-policy-dependent additional set. GuidedBench was accepted
as a poster at ICLR 2026.

## Installation

```bash
pip install guidedbench
```

Install the current GitHub version directly when needed:

```bash
pip install "guidedbench @ git+https://github.com/SproutNan/GuidedBench.git"
```

Install only the judge backend you need:

```bash
pip install "guidedbench[openai]"      # OpenAI and OpenAI-compatible APIs
pip install "guidedbench[anthropic]"   # Claude Messages API
pip install "guidedbench[local]"       # Local Hugging Face Transformers model
```

## Two-step workflow

GuidedBench separates jailbreak generation from evaluation. Your jailbreak
runner produces victim-model responses in Step 1; GuidedEval scores those saved
responses in Step 2. GuidedBench does not require a particular attack framework
or victim-model provider.

### Step 1: Run your jailbreak method on GuidedBench

First request access on the
[GuidedBench dataset page](https://huggingface.co/datasets/HRXUST/GuidedBench),
then authenticate once:

```bash
hf auth login
```

Load the core cases and pass each `question` through your existing jailbreak
method and victim-model client:

```python
import json
from collections.abc import Callable

from guidedbench import load_cases


def generate_responses(
    jailbreak_method: Callable[[str], str],
    query_victim: Callable[[str], str],
) -> None:
    with open("generations.jsonl", "w", encoding="utf-8") as output:
        for case in load_cases(subset="core"):
            attack_prompt = jailbreak_method(case.question)
            response = query_victim(attack_prompt)
            record = {
                "id": case.id,
                "response": response,
                "method": "method-name",
                "victim_model": "victim-model-id",
                "metadata": {"attack_prompt": attack_prompt},
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


# Supply the two callables from your existing jailbreak runner:
# generate_responses(my_jailbreak_method, my_victim_client.generate)
```

The output of Step 1 is `generations.jsonl`, with one victim response per line:

```json
{"id":"guidedbench-000","response":"...","method":"method-name","victim_model":"victim-model-id","metadata":{"attack_prompt":"..."}}
```

Every row must contain `response` plus one stable case identifier: `id`, `index`,
or the exact `question`. `method`, `victim_model`, and `metadata` are optional
and are copied into the final score records.

If your jailbreak runner consumes JSONL rather than Python objects, export the
pinned benchmark release first:

```bash
guidedbench export cases.jsonl --subset core
```

No database is required. GuidedBench downloads the JSONL files from a pinned
Hugging Face commit on first use and then reuses the standard local Hub cache.
Neither this GitHub repository nor the Python wheel contains the benchmark
records. In CI, `HF_TOKEN` can provide access without an interactive login.

### Step 2: Score the generated responses with GuidedEval

GuidedEval is exposed through the `guidedbench evaluate` command and the
`GuidedBenchEvaluator` Python class. The most portable option is an
OpenAI-compatible judge endpoint, whether it is remote or served locally by
vLLM, SGLang, LM Studio, Ollama, llama.cpp, or another compatible server:

```bash
export JUDGE_API_KEY="not-needed"

guidedbench evaluate generations.jsonl \
  --provider openai-compatible \
  --model your-judge-model \
  --base-url http://localhost:8000/v1 \
  --api-key-env JUDGE_API_KEY \
  --workers 8 \
  --output scores.jsonl
```

The compatible backend uses `POST /chat/completions` by default. Add
`--api-style responses` when the endpoint implements the Responses API.

Native OpenAI and Anthropic judges are also available:

```bash
export OPENAI_API_KEY="..."
guidedbench evaluate generations.jsonl \
  --provider openai \
  --model YOUR_OPENAI_MODEL \
  --api-key-env OPENAI_API_KEY \
  --output scores.jsonl

export ANTHROPIC_API_KEY="..."
guidedbench evaluate generations.jsonl \
  --provider anthropic \
  --model YOUR_CLAUDE_MODEL \
  --api-key-env ANTHROPIC_API_KEY \
  --output scores.jsonl
```

Or load an open-source judge directly with Transformers:

```bash
guidedbench evaluate generations.jsonl \
  --provider transformers \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --workers 1 \
  --output scores.jsonl
```

The Transformers backend uses the model's chat template and deterministic
generation (`do_sample=False`). Serving the same model through vLLM or SGLang
and using `openai-compatible` is usually faster for concurrent evaluation.

Each line in `scores.jsonl` contains the GuidedEval score from 0 to 1, every
guideline-level decision, the judge name, the raw judge output, benchmark
version metadata, and the optional Step 1 metadata. The command also prints the
mean score after a successful run. Keep each jailbreak-method/victim-model pair
in a separate output file when reporting aggregate results.

Remote judge requests can run concurrently without sharing a result file. Each
completed evaluation is persisted as an independent atomic checkpoint inside
`scores.jsonl.parts/` before the final JSONL is gathered. If a run stops midway,
rerun the same command: completed checkpoints are reused and only missing or
failed evaluations are retried.

## Repository contents

- [Hugging Face Dataset](https://huggingface.co/datasets/HRXUST/GuidedBench):
  canonical `core` and `additional` JSONL splits, access form, data card, and
  dataset license.
- [`src/guidedbench/`](src/guidedbench/): installable evaluator and provider
  integrations. It downloads a pinned dataset revision when records are needed.

## Reproducibility

Report at least the GuidedBench version, judge provider and exact model id,
generation parameters, evaluator failures/refusals, and whether scores cover
the core set, additional set, or both. Do not compare aggregate scores produced
with different benchmark versions without disclosing the difference.

## Safety and licensing

This repository is intended for controlled AI-safety research. The benchmark
contains descriptions of harmful objectives and examples. Do not use it to
facilitate real-world harm.

The software is licensed under Apache-2.0. The separately distributed
GuidedBench dataset is licensed under CC BY 4.0; see its
[dataset license](https://huggingface.co/datasets/HRXUST/GuidedBench/blob/main/LICENSE).

## Citation

```bibtex
@inproceedings{huang2026guidedbench,
  title     = {{GuidedBench}: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild {LLM} Jailbreak Methods},
  author    = {Ruixuan Huang and Xunguang Wang and Zongjie Li and Daoyuan Wu and Shuai Wang},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=ZVg8y3ibyM}
}
```
