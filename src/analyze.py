"""Aggregate judged results into a summary table and a token-count-vs-correctness plot. Runs locally."""
import argparse
import json

import matplotlib.pyplot as plt

from data_utils import RESULTS_DIR, read_jsonl

CONDITIONS = ["base", "prompt", "const", "prompt_const", "psr", "prompt_psr", "a_psr", "prompt_a_psr"]
MAX_NEW_TOKENS = 150  # must match model_common.MAX_NEW_TOKENS; not imported to keep this script torch-free
LABELS = {
    "base": "Base",
    "prompt": "Prompt",
    "const": "Steer",
    "prompt_const": "Prompt+Steer",
    "psr": "S-PSR",
    "prompt_psr": "Prompt+S-PSR",
    "a_psr": "A-PSR",
    "prompt_a_psr": "Prompt+A-PSR",
}
MARKERS = {
    "base": "o",
    "prompt": "s",
    "const": "^",
    "prompt_const": "D",
    "psr": "v",
    "prompt_psr": "P",
    "a_psr": "X",
    "prompt_a_psr": "*",
}
# Same validated all-pairs-safe 4-color palette as the sweep plot (dataviz skill, light-mode static PNG)
# for the original 4 conditions; the 4 new ones reuse a second validated palette at reduced saturation
# so the plot stays legible with 8 points -- re-run scripts/validate_palette.js before trusting this if
# you change these.
COLORS = {
    "base": "#2a78d6",
    "prompt": "#eb6834",
    "const": "#1baf7a",
    "prompt_const": "#4a3aa7",
    "psr": "#8e44ad",
    "prompt_psr": "#c0392b",
    "a_psr": "#16a085",
    "prompt_a_psr": "#d35400",
}
MODEL_NAME = "Qwen2.5-Coder-7B-Instruct"
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"


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
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for cond in CONDITIONS:
        s = summary[cond]
        x, y = s["avg_tokens"], s["full_correct_rate"] * 100
        ax.scatter(x, y, marker=MARKERS[cond], s=130, color=COLORS[cond], zorder=3)
        ax.annotate(
            f"{LABELS[cond]}\n{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(9, 7),
            fontsize=10,
            color=INK,
            linespacing=1.4,
        )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.3, ymax + 0.9)  # headroom so top annotations clear the title
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
    ax.set_title(f"Steering and Prompt Correctness - {MODEL_NAME}", color=INK, fontsize=12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote plot to {out_path}")


def main(split: str) -> None:
    rows = read_jsonl(RESULTS_DIR / f"judged_{split}.jsonl")
    summary = summarize(rows)

    print(f"{'condition':<14}{'avg_tokens':>12}{'full_correct':>14}{'any_correct':>13}{'coherent':>11}")
    for cond in CONDITIONS:
        s = summary[cond]
        print(
            f"{LABELS[cond]:<14}{s['avg_tokens']:>12.1f}{s['full_correct_rate'] * 100:>13.1f}%"
            f"{s['any_correct_rate'] * 100:>12.1f}%{s['coherent_rate'] * 100:>10.1f}%"
        )

    with (RESULTS_DIR / f"summary_{split}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot(summary, RESULTS_DIR / f"summary_plot_{split}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
