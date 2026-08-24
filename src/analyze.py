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


def _cluster_points_by_pixel_proximity(ax, points, min_pixel_gap=28):
    transform = ax.transData.transform
    for p in points:
        p["px"], p["py"] = transform((p["x"], p["y"]))

    remaining = sorted(points, key=lambda p: -p["py"])
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        still_remaining = []
        for p in remaining:
            near_any = any(
                ((p["px"] - m["px"]) ** 2 + (p["py"] - m["py"]) ** 2) ** 0.5 < min_pixel_gap * 2.2
                for m in cluster
            )
            (cluster if near_any else still_remaining).append(p)
        remaining = still_remaining
        clusters.append(cluster)
    return clusters


def _place_labels_without_overlap(ax, clusters, base_offset=(9, 7), stagger_step=26):
    """Default label placement is a fixed (9,7) point offset for every marker -- fine when points are
    spread out, but multiple conditions landing close together in (tokens, correctness) space is a
    normal outcome, not a rare edge case, and their labels end up directly on top of each other.
    Clusters are pre-computed by actual on-screen (pixel) proximity -- not raw data-unit distance,
    since tokens (0-160) and correctness (0-100%) are on very different scales -- so this just fans
    each cluster's labels out vertically with a short leader line back to its point."""
    for cluster in clusters:
        if len(cluster) == 1:
            p = cluster[0]
            ax.annotate(
                p["label"], (p["x"], p["y"]), textcoords="offset points", xytext=base_offset,
                fontsize=10, color=INK, linespacing=1.4,
            )
            continue
        cluster_sorted = sorted(cluster, key=lambda p: p["x"])
        for i, p in enumerate(cluster_sorted):
            dy = base_offset[1] + i * stagger_step
            dx = base_offset[0] + (i - (len(cluster_sorted) - 1) / 2) * 4
            ax.annotate(
                p["label"], (p["x"], p["y"]), textcoords="offset points", xytext=(dx, dy),
                fontsize=10, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=0, shrinkB=6),
            )


def plot(summary: dict[str, dict], out_path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    points = []
    for cond in CONDITIONS:
        s = summary[cond]
        x, y = s["avg_tokens"], s["full_correct_rate"] * 100
        ax.scatter(x, y, marker=MARKERS[cond], s=130, color=COLORS[cond], zorder=3)
        points.append({"cond": cond, "x": x, "y": y, "label": f"{LABELS[cond]}\n{y:.1f}%"})

    ax.set_xlim(0, 160)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.3, ymax + 0.9)  # base headroom, expanded further below if a cluster needs more
    fig.canvas.draw()  # need a real transData to detect pixel-space clusters against actual axis limits

    clusters = _cluster_points_by_pixel_proximity(ax, points)

    # Whichever cluster stacks the most labels determines how much extra headroom is actually needed --
    # computed from the real cluster sizes found, not a fixed guessed constant, so this doesn't quietly
    # break again the next time a run happens to produce a differently-shaped cluster of results.
    max_stagger_points = max((len(c) - 1) * 26 for c in clusters) if clusters else 0
    if max_stagger_points > 0:
        px_per_data_unit = ax.get_window_extent().height / (ax.get_ylim()[1] - ax.get_ylim()[0])
        px_per_point = fig.dpi / 72.0
        needed_data_units = (max_stagger_points + 20) * px_per_point / px_per_data_unit  # +20pt for label text height
        current_top_margin = ax.get_ylim()[1] - max(p["y"] for p in points)
        if needed_data_units > current_top_margin:
            ax.set_ylim(ax.get_ylim()[0], max(p["y"] for p in points) + needed_data_units)
            fig.canvas.draw()  # transData changed with the new ylim -- redetect clusters against it
            clusters = _cluster_points_by_pixel_proximity(ax, points)

    _place_labels_without_overlap(ax, clusters)

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