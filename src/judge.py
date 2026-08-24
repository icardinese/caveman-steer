"""Score each generated explanation for correctness and coherence with an LLM judge. Runs locally
(no GPU needed) — only reads results/generations_<split>.jsonl produced by generate.py on the GPU pod.

Resumable, same pattern as generate.py: writes each row immediately and skips any row id already
present in the output file on startup. With 8 conditions x 180 rows = 1,440 real API calls, losing
partial progress to a crash mid-run means re-paying for and re-running calls that already succeeded."""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from data_utils import RESULTS_DIR, append_jsonl, read_jsonl

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


def already_done_ids(out_path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(split: str) -> None:
    api_key = KEY_PATH.read_text().strip()
    client = OpenAI(api_key=api_key)
    rows = read_jsonl(RESULTS_DIR / f"generations_{split}.jsonl")

    out_path = RESULTS_DIR / f"judged_{split}.jsonl"
    done_ids = already_done_ids(out_path)
    if done_ids:
        print(f">>> Resuming: {len(done_ids)}/{len(rows)} rows already judged in {out_path}, skipping those")

    n_processed = 0
    for i, row in enumerate(rows):
        if row["id"] in done_ids:
            continue

        judged = {"id": row["id"]}
        for cond in CONDITIONS:
            candidate = row[f"{cond}_response"]
            score = judge_one(client, row["code"], row["reference_explanation"], candidate)
            judged[f"{cond}_correct"] = score["correct"]
            judged[f"{cond}_coherent"] = score["coherent"]
            judged[f"{cond}_tokens"] = row[f"{cond}_tokens"]

        append_jsonl(out_path, judged)  # written and flushed immediately, not batched
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"{len(done_ids) + n_processed}/{len(rows)} judged")

    print(f"done. {out_path} now has {len(done_ids) + n_processed}/{len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)