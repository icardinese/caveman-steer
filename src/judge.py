"""Score the 4 new conditions (psr_conceptor, prompt_psr_conceptor, psr_proper, prompt_psr_proper)
for correctness and coherence, same rubric/model as judge.py. Runs locally (no GPU needed) --
only reads results/generations_<split>_new_conditions.jsonl (from generate_new_conditions.py)
plus data/<split>.jsonl (for code + reference_explanation, which the generations file doesn't
duplicate) and joins them on "id".

No resumability here, matching judge.py's own convention -- a full re-run is 4 conditions x
however many rows, i.e. half the API calls judge.py makes for the original 8.
"""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl, write_jsonl

KEY_PATH = Path(__file__).resolve().parent.parent / "openai.key"
JUDGE_MODEL = "gpt-4o-mini"
CONDITIONS = ["psr_conceptor", "prompt_psr_conceptor", "psr_proper", "prompt_psr_proper"]

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

    gen_rows = read_jsonl(RESULTS_DIR / f"generations_{split}_new_conditions.jsonl")
    data_rows = {row["id"]: row for row in read_jsonl(DATA_DIR / f"{split}.jsonl")}

    out_rows = []
    for i, gen_row in enumerate(gen_rows):
        data_row = data_rows.get(gen_row["id"])
        if data_row is None:
            print(f"WARNING: id {gen_row['id']} in generations file but not in data/{split}.jsonl, skipping")
            continue

        judged = {"id": gen_row["id"]}
        for cond in CONDITIONS:
            candidate = gen_row[f"{cond}_response"]
            score = judge_one(client, data_row["code"], data_row["reference_explanation"], candidate)
            judged[f"{cond}_correct"] = score["correct"]
            judged[f"{cond}_coherent"] = score["coherent"]
            judged[f"{cond}_tokens"] = gen_row[f"{cond}_tokens"]
        out_rows.append(judged)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(gen_rows)} judged")

    write_jsonl(RESULTS_DIR / f"judged_{split}_new_conditions.jsonl", out_rows)
    print(f"wrote {len(out_rows)} rows to results/judged_{split}_new_conditions.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)