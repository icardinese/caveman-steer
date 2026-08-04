"""Score each generated explanation for correctness and coherence with an LLM judge. Runs locally
(no GPU needed) — only reads results/generations_<split>.jsonl produced by generate.py on the GPU pod."""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from data_utils import RESULTS_DIR, read_jsonl, write_jsonl

KEY_PATH = Path(__file__).resolve().parent.parent / "openai.key"
JUDGE_MODEL = "gpt-4o-mini"
CONDITIONS = ["base", "prompt", "const", "prompt_const", "psr", "prompt_psr", "a_psr", "prompt_a_psr"]

RUBRIC = """You are grading an automatically generated explanation of a Python function.

Function:
```python
{code}
```

Reference explanation (written by the original developer, for grading only):
{reference_explanation}

Candidate explanation to grade:
{candidate}

Score the candidate explanation on two axes:
- "correct": 0 if it is wrong or misleading about what the function does, 1 if it is vague or only partially
  correct, 2 if it correctly captures the function's actual behavior (wording may differ from the reference).
- "coherent": false if the text is degenerate, repetitive, non-English gibberish, or so garbled it fails to
  read as a genuine explanation; true otherwise.

Respond with ONLY a JSON object, no other text: {{"correct": <0|1|2>, "coherent": <true|false>}}"""


def judge_one(client: OpenAI, code: str, reference_explanation: str, candidate: str) -> dict:
    prompt = RUBRIC.format(code=code, reference_explanation=reference_explanation, candidate=candidate)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main(split: str) -> None:
    api_key = KEY_PATH.read_text().strip()
    client = OpenAI(api_key=api_key)
    rows = read_jsonl(RESULTS_DIR / f"generations_{split}.jsonl")

    out_rows = []
    for i, row in enumerate(rows):
        judged = {"id": row["id"]}
        for cond in CONDITIONS:
            candidate = row[f"{cond}_response"]
            score = judge_one(client, row["code"], row["reference_explanation"], candidate)
            judged[f"{cond}_correct"] = score["correct"]
            judged[f"{cond}_coherent"] = score["coherent"]
            judged[f"{cond}_tokens"] = row[f"{cond}_tokens"]
        out_rows.append(judged)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(rows)} judged")

    write_jsonl(RESULTS_DIR / f"judged_{split}.jsonl", out_rows)
    print(f"wrote {len(out_rows)} rows to results/judged_{split}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
