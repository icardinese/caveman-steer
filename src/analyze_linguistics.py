"""Caveman-rule-compliance comparisons across all conditions: sentence length, definite-article
compliance, and token usage, each as a per-condition dot with a 95% CI whisker. This is exploratory
analysis, not part of the polished README deliverable -- pass --out-dir to write it somewhere other
than results/ (e.g. the scratchpad) so it doesn't need to live in the tracked repo. Runs locally, no GPU."""
import argparse
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from data_utils import RESULTS_DIR, read_jsonl

CONDITIONS = ["base", "prompt", "const", "prompt_const", "psr", "prompt_psr", "a_psr", "prompt_a_psr"]
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
# Color encodes a grouping that matters -- has steering or not -- rather than duplicating the x-axis
# labels. NOT re-validated for CVD/contrast with the new conditions added -- re-run
# scripts/validate_palette.js before trusting this if you keep this grouping scheme.
HAS_STEERING = {
    "base": False,
    "prompt": False,
    "const": True,
    "prompt_const": True,
    "psr": True,
    "prompt_psr": True,
    "a_psr": True,
    "prompt_a_psr": True,
}
GROUP_COLOR = {False: "#2a78d6", True: "#eb6834"}
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#ececea"

DEFINITE_ARTICLE = re.compile(r"\bthe\b", re.IGNORECASE)


def avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lens = [len(s.split()) for s in sentences]
    return sum(lens) / len(lens) if lens else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return center - half, center + half


def mean_ci(vals: list[float], z: float = 1.96) -> tuple[float, float, float]:
    m = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals))
    return m, m - z * se, m + z * se


def dot_ci_panel(
    ax,
    point: dict[str, float],
    lo: dict[str, float],
    hi: dict[str, float],
    title: str,
    zero_based: bool = False,
) -> None:
    positions = list(range(len(CONDITIONS)))

    # Group shading: a light tint behind the two "has steering" conditions.
    ax.axvspan(1.5, len(CONDITIONS) - 0.5, color=GROUP_COLOR[True], alpha=0.05, zorder=0)

    for i, c in enumerate(CONDITIONS):
        err_lo = point[c] - lo[c]
        err_hi = hi[c] - point[c]
        ax.errorbar(
            [i], [point[c]], yerr=[[err_lo], [err_hi]], fmt=MARKERS[c], color=GROUP_COLOR[HAS_STEERING[c]],
            ecolor=INK, markersize=7, capsize=5, capthick=1.3, linewidth=1.3,
            markeredgecolor=INK, markeredgewidth=0.8, zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=9.5)
    # Prompt+Steer's label stands out in black/bold; the rest stay muted.
    for tick_label, c in zip(ax.get_xticklabels(), CONDITIONS):
        if c == "prompt_const":
            tick_label.set_color(INK)
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color(MUTED)
    ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="medium")

    if zero_based:
        ax.set_ylim(0, max(hi.values()) * 1.08)
    else:
        span = max(hi.values()) - min(lo.values())
        ax.set_ylim(min(lo.values()) - span * 0.15, max(hi.values()) + span * 0.15)

    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main(out_dir: Path) -> None:
    rows = read_jsonl(RESULTS_DIR / "generations_test.jsonl")
    n = len(rows)

    tokens = {c: mean_ci([r[f"{c}_tokens"] for r in rows]) for c in CONDITIONS}
    sent_len = {c: mean_ci([avg_sentence_length(r[f"{c}_response"]) for r in rows]) for c in CONDITIONS}

    definite_rate = {}
    for c in CONDITIONS:
        texts = [r[f"{c}_response"] for r in rows]
        k_def = sum(1 for t in texts if DEFINITE_ARTICLE.search(t))
        lo_d, hi_d = wilson_ci(k_def, n)
        definite_rate[c] = (k_def / n * 100, lo_d * 100, hi_d * 100)

    def bold_if_lower(metric: dict[str, tuple[float, float, float]]) -> str:
        p, ps = metric["prompt"][0], metric["prompt_const"][0]
        return f"**{ps:.2f}**" if ps < p else f"{ps:.2f}"

    print(f"{'condition':<14}{'sent_len_mean':>16}{'definite_the%':>15}{'tokens_mean':>14}")
    for c in CONDITIONS:
        print(
            f"{LABELS[c]:<14}{sent_len[c][0]:>16.2f}{definite_rate[c][0]:>14.1f}%{tokens[c][0]:>14.2f}"
        )
    print()
    print("Prompt+Steer bolded where its mean/rate is below Prompt's:")
    print(f"  sentence length:   Prompt {sent_len['prompt'][0]:.2f}  Prompt+Steer {bold_if_lower(sent_len)}")
    print(f"  definite article%: Prompt {definite_rate['prompt'][0]:.1f}  Prompt+Steer {bold_if_lower(definite_rate)}")
    print(f"  tokens:            Prompt {tokens['prompt'][0]:.2f}  Prompt+Steer {bold_if_lower(tokens)}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    sl_point = {c: sent_len[c][0] for c in CONDITIONS}
    sl_lo = {c: sent_len[c][1] for c in CONDITIONS}
    sl_hi = {c: sent_len[c][2] for c in CONDITIONS}
    dot_ci_panel(axes[0], sl_point, sl_lo, sl_hi, "Mean sentence length (words)")

    def_point = {c: definite_rate[c][0] for c in CONDITIONS}
    def_lo = {c: definite_rate[c][1] for c in CONDITIONS}
    def_hi = {c: definite_rate[c][2] for c in CONDITIONS}
    dot_ci_panel(axes[1], def_point, def_lo, def_hi, 'Responses containing "the" (%)', zero_based=True)

    tok_point = {c: tokens[c][0] for c in CONDITIONS}
    tok_lo = {c: tokens[c][1] for c in CONDITIONS}
    tok_hi = {c: tokens[c][2] for c in CONDITIONS}
    dot_ci_panel(axes[2], tok_point, tok_lo, tok_hi, "Mean response tokens")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=GROUP_COLOR[False], linestyle="", markersize=8, markeredgecolor=INK, markeredgewidth=0.8),
        plt.Line2D([0], [0], marker="o", color=GROUP_COLOR[True], linestyle="", markersize=8, markeredgecolor=INK, markeredgewidth=0.8),
    ]
    legend = fig.legend(
        legend_handles, ["No steering", "Has steering"],
        loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=True, labelcolor=INK, fontsize=10,
    )
    legend.get_frame().set_edgecolor(MUTED)
    legend.get_frame().set_facecolor("#fcfcfb")
    fig.suptitle("Caveman Rules Compliance", color=INK, fontsize=16, x=0.32, y=0.985, fontweight="bold")
    fig.text(0.5, 0.02, f"(n={n}, 95% CI)", ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.055, right=0.98, wspace=0.28)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "caveman_rules_compliance.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    main(args.out_dir)
