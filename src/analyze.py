"""Merges judged_<split>.jsonl (original 8 conditions) with judged_<split>_new_conditions.jsonl
(psr_conceptor, prompt_psr_conceptor, psr_proper, prompt_psr_proper) into one summary + plot across
all 12 conditions -- this is the actual comparison that matters (new methods vs. Prompt+Steer, not
just against each other in isolation). Separate output filenames from analyze.py's own
summary_<split>.json / summary_plot_<split>.png -- doesn't overwrite the original.

CAUTION on colors: analyze.py's own comment claims its 8-color palette was checked with a
scripts/validate_palette.js for pairwise distinguishability. That script isn't present in this
checkout, so I can't confirm it actually was run, and I haven't validated the 4 new colors below
against it either. Eyeball the plot before trusting it at a glance -- if two conditions look
confusable, that's a real risk here, not something already ruled out.
"""
import argparse
import json

import matplotlib.pyplot as plt

from data_utils import RESULTS_DIR, read_jsonl

CONDITIONS = [
    "base", "prompt", "const", "prompt_const", "psr", "prompt_psr", "a_psr", "prompt_a_psr",
    "psr_conceptor", "prompt_psr_conceptor", "psr_proper", "prompt_psr_proper",
]
NEW_CONDITIONS = {"psr_conceptor", "prompt_psr_conceptor", "psr_proper", "prompt_psr_proper"}
MAX_NEW_TOKENS = 150  # must match model_common.MAX_NEW_TOKENS; not imported to keep this script torch-free
LABELS = {
    "base": "Base", "prompt": "Prompt", "const": "Steer", "prompt_const": "Prompt+Steer",
    "psr": "S-PSR", "prompt_psr": "Prompt+S-PSR", "a_psr": "A-PSR", "prompt_a_psr": "Prompt+A-PSR",
    "psr_conceptor": "PSR-Conceptor", "prompt_psr_conceptor": "Prompt+PSR-Conceptor",
    "psr_proper": "PSR-Proper", "prompt_psr_proper": "Prompt+PSR-Proper",
}
MARKERS = {
    "base": "o", "prompt": "s", "const": "^", "prompt_const": "D",
    "psr": "v", "prompt_psr": "P", "a_psr": "X", "prompt_a_psr": "*",
    "psr_conceptor": "h", "prompt_psr_conceptor": "p", "psr_proper": "<", "prompt_psr_proper": ">",
}
COLORS = {
    "base": "#2a78d6", "prompt": "#eb6834", "const": "#1baf7a", "prompt_const": "#4a3aa7",
    "psr": "#8e44ad", "prompt_psr": "#c0392b", "a_psr": "#16a085", "prompt_a_psr": "#d35400",
    "psr_conceptor": "#2c3e50", "prompt_psr_conceptor": "#e91e8c",
    "psr_proper": "#7f8c00", "prompt_psr_proper": "#00838f",
}
MODEL_NAME = "Qwen2.5-Coder-7B-Instruct"
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def load_merged(split: str) -> list[dict]:
    old_rows = {r["id"]: r for r in read_jsonl(RESULTS_DIR / f"judged_{split}.jsonl")}
    new_rows = read_jsonl(RESULTS_DIR / f"judged_{split}_new_conditions.jsonl")

    merged = []
    missing = 0
    for new_row in new_rows:
        old_row = old_rows.get(new_row["id"])
        if old_row is None:
            missing += 1
            continue
        merged.append({**old_row, **new_row})
    if missing:
        print(f"WARNING: {missing}/{len(new_rows)} new-condition rows had no matching id in judged_{split}.jsonl, dropped")
    return merged


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    n = len(rows)
    for cond in CONDITIONS:
        avg_tokens = sum(r[f"{cond}_tokens"] for r in rows) / n
        full_correct = sum(1 for r in rows if r[f"{cond}_correct"] == 2) / n
        any_correct = sum(1 for r in rows if r[f"{cond}_correct"] >= 1) / n
        coherent = sum(1 for r in rows if r[f"{cond}_coherent"]) / n
        summary[cond] = {
            "avg_tokens": avg_tokens,
            "full_correct_rate": full_correct,
            "any_correct_rate": any_correct,
            "coherent_rate": coherent,
        }
    return summary


def plot(summary: dict[str, dict], out_path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # A legend, not inline text labels. Several conditions here land at near-identical (x, y) --
    # e.g. Steer / S-PSR / A-PSR / PSR-Conceptor / PSR-Proper all sit around 95-98% correctness at
    # 100-135 tokens -- so no amount of text-repelling (adjust_text tried, still collided) can
    # separate labels sitting on top of literally overlapping points. A legend sidesteps this
    # entirely: it doesn't need to be near the point.
    handles = []
    for cond in CONDITIONS:
        s = summary[cond]
        x, y = s["avg_tokens"], s["full_correct_rate"] * 100
        ax.scatter(x, y, marker=MARKERS[cond], s=150, color=COLORS[cond], zorder=3,
                   edgecolors="#0b0b0b" if cond in NEW_CONDITIONS else "none", linewidths=0.9)
        handles.append(plt.Line2D(
            [0], [0], marker=MARKERS[cond], color="w", markerfacecolor=COLORS[cond],
            markeredgecolor="#0b0b0b" if cond in NEW_CONDITIONS else COLORS[cond],
            markersize=10, label=f"{LABELS[cond]} \u2014 {y:.1f}%",
        ))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_xlim(0, 160)

    ax.axvline(MAX_NEW_TOKENS, color="#e34948", linestyle=(0, (1, 2)), linewidth=1.5, zorder=2)
    ax.annotate(
        "MAX_NEW_TOKENS = 150",
        (MAX_NEW_TOKENS, ax.get_ylim()[1]),
        textcoords="offset points",
        xytext=(-8, -4),
        va="top",
        ha="right",
        fontsize=8,
        color="#e34948",
    )

    ax.set_xlabel("Average response tokens", color=INK)
    ax.set_ylabel("Fully-correct rate", color=INK)
    ax.set_title(f"Steering and Prompt Correctness (all 12 conditions) - {MODEL_NAME}", color=INK, fontsize=12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              frameon=False, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote plot to {out_path}")


def main(split: str) -> None:
    rows = load_merged(split)
    if not rows:
        print("no rows with both old and new judged results -- nothing to summarize")
        return
    summary = summarize(rows)

    print(f"{'condition':<22}{'avg_tokens':>12}{'full_correct':>14}{'any_correct':>13}{'coherent':>11}")
    for cond in CONDITIONS:
        s = summary[cond]
        marker = " *" if cond in NEW_CONDITIONS else ""
        print(
            f"{LABELS[cond] + marker:<22}{s['avg_tokens']:>12.1f}{s['full_correct_rate'] * 100:>13.1f}%"
            f"{s['any_correct_rate'] * 100:>12.1f}%{s['coherent_rate'] * 100:>10.1f}%"
        )
    print("(* = new PSR-conceptor/proper conditions)")

    with (RESULTS_DIR / f"summary_{split}_all12.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot(summary, RESULTS_DIR / f"summary_plot_{split}_all12.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)