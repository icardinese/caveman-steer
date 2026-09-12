"""Bootstrap CIs for judged output-quality metrics (full_correct, any_correct, coherent) across
all 12 conditions -- NOT MSE. MSE trivially decreases with more aggressive steering; it says nothing
about whether the output is still good, which is the actual question once you're past "does the
correction do anything at all." Runs entirely locally against results you already have -- no GPU.

Uses a PAIRED bootstrap for comparisons (same 180 resampled ids applied to both conditions in a
pair), not two independent bootstraps -- correct here since every condition is evaluated on the
exact same underlying test rows, so the natural per-example correlation between two conditions'
scores should be preserved, not averaged away.
"""
import argparse
import json
import random

from data_utils import RESULTS_DIR, read_jsonl

N_BOOTSTRAP = 2000
CI_LOW, CI_HIGH = 2.5, 97.5  # percentiles for a 95% CI

CONDITIONS = [
    "base", "prompt", "const", "prompt_const", "psr", "prompt_psr", "a_psr", "prompt_a_psr",
    "psr_conceptor", "prompt_psr_conceptor", "psr_proper", "prompt_psr_proper",
]
METRICS = {
    "full_correct": lambda r, c: r[f"{c}_correct"] == 2,
    "any_correct": lambda r, c: r[f"{c}_correct"] >= 1,
    "coherent": lambda r, c: bool(r[f"{c}_coherent"]),
}
# Separate from METRICS (above) because these return numbers on a 0-2 scale, not booleans --
# bootstrap_ci/paired_bootstrap_diff work identically either way (mean of a list), but printing
# and interpretation differ (percentage points vs. a raw score difference). Using the full 0/1/2
# scale instead of binarizing to "==2" keeps information a partial-credit score (1) actually
# carries, which binarizing throws away -- more information per example means tighter CIs at the
# same n, i.e. real statistical power gained for free.
GRADED_METRICS = {
    "mean_score": lambda r, c: r[f"{c}_correct"],
}

# The comparisons worth actually testing, not every pair (66 pairs x 3 metrics = 198 tests would be
# noise-mining). Each is a real question raised earlier in this investigation.
KEY_COMPARISONS = [
    ("prompt_psr_proper", "prompt_psr_conceptor", "does psr_proper actually beat psr_conceptor on output quality, not just training MSE?"),
    ("prompt_psr_proper", "prompt_psr", "does the fidelity-fixed trained-direction variant beat the OLD S-PSR baseline?"),
    ("prompt_psr_proper", "prompt", "does steering add anything over prompting alone, for the best new method?"),
    ("prompt_const", "prompt", "does constant steering actually hurt correctness relative to prompting alone? (the 37.2% collapse)"),
    ("prompt_psr", "prompt_a_psr", "does the old single-layer PSR beat the old multi-layer PSR?"),
]


def bootstrap_ci(values: list[bool], n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    n = len(values)
    point = sum(values) / n
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [values[random.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(n_bootstrap * CI_LOW / 100)]
    hi = boot_means[int(n_bootstrap * CI_HIGH / 100)]
    return point, lo, hi


def paired_bootstrap_diff(values_a: list[bool], values_b: list[bool], n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """CI on (metric_a - metric_b), resampling the SAME indices for both -- preserves pairing."""
    n = len(values_a)
    assert len(values_b) == n
    point = sum(values_a) / n - sum(values_b) / n
    diffs = []
    for _ in range(n_bootstrap):
        idx = [random.randrange(n) for _ in range(n)]
        a = sum(values_a[i] for i in idx) / n
        b = sum(values_b[i] for i in idx) / n
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(n_bootstrap * CI_LOW / 100)]
    hi = diffs[int(n_bootstrap * CI_HIGH / 100)]
    return point, lo, hi


def load_merged(split: str) -> list[dict]:
    old_rows = {r["id"]: r for r in read_jsonl(RESULTS_DIR / f"judged_{split}.jsonl")}
    new_rows = read_jsonl(RESULTS_DIR / f"judged_{split}_new_conditions.jsonl")
    merged = []
    for new_row in new_rows:
        old_row = old_rows.get(new_row["id"])
        if old_row is not None:
            merged.append({**old_row, **new_row})
    return merged


def main(split: str) -> None:
    rows = load_merged(split)
    n = len(rows)
    print(f"{n} rows with both old and new judged results\n")

    print("=" * 80)
    print("PER-CONDITION 95% BOOTSTRAP CIs")
    print("=" * 80)
    summary = {}
    for metric_name, metric_fn in {**METRICS, **GRADED_METRICS}.items():
        print(f"\n--- {metric_name} ---")
        is_graded = metric_name in GRADED_METRICS
        unit = "" if is_graded else "%"
        scale = 1 if is_graded else 100
        print(f"{'condition':<24}{'point':>8}{'95% CI':>20}")
        summary[metric_name] = {}
        for cond in CONDITIONS:
            values = [metric_fn(r, cond) for r in rows]
            point, lo, hi = bootstrap_ci(values)
            summary[metric_name][cond] = {"point": point, "ci_low": lo, "ci_high": hi}
            print(f"{cond:<24}{point*scale:>7.2f}{unit}   [{lo*scale:.2f}{unit}, {hi*scale:.2f}{unit}]")

    print("\n" + "=" * 80)
    print("KEY PAIRED COMPARISONS (95% CI on the difference; excludes 0 = likely real difference)")
    print("=" * 80)
    comparisons = []
    for cond_a, cond_b, question in KEY_COMPARISONS:
        print(f"\n--- {question} ---")
        print(f"    {cond_a} vs {cond_b}")
        for metric_name, metric_fn in {**METRICS, **GRADED_METRICS}.items():
            is_graded = metric_name in GRADED_METRICS
            unit = "" if is_graded else "pp"
            scale = 1 if is_graded else 100
            values_a = [metric_fn(r, cond_a) for r in rows]
            values_b = [metric_fn(r, cond_b) for r in rows]
            point, lo, hi = paired_bootstrap_diff(values_a, values_b)
            excludes_zero = (lo > 0) or (hi < 0)
            flag = "  <-- CI excludes 0, likely real" if excludes_zero else "  (CI includes 0, could be noise)"
            print(f"    {metric_name:<14} diff={point*scale:>6.2f}{unit}   [{lo*scale:.2f}, {hi*scale:.2f}]{unit}{flag}")
            comparisons.append({
                "cond_a": cond_a, "cond_b": cond_b, "question": question, "metric": metric_name,
                "diff": point, "ci_low": lo, "ci_high": hi, "excludes_zero": excludes_zero,
            })

    out_path = RESULTS_DIR / f"bootstrap_analysis_{split}.json"
    with out_path.open("w") as f:
        json.dump({"per_condition": summary, "comparisons": comparisons, "n": n}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)