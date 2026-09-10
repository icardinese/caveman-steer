"""Layer sweep for PSR-Conceptor and PSR-Proper, plus the rank diagnostic (quota / capture
fraction) at every layer in the grid. Answers the biggest open question flagged in the README:
every result so far used layer 14, inherited from the constant-steering sweep, never validated
for PSR or the conceptor specifically.

Key efficiency: pooling activations for the conceptor/diagnostic doesn't need a separate forward
pass per candidate layer. model(..., output_hidden_states=True) returns every layer in one call,
so collect_all_layers_pooled() does exactly the same number of forward passes as the single-layer
version did (2 per row: base prompt, terse prompt) regardless of how many layers are in LAYER_GRID.
Training a gate at a given layer still needs its own live forward+backward per layer (unavoidable,
the correction changes the graph), so that part of the cost does scale with the grid.

Writes one combined results file (results/layer_sweep.jsonl), one row per (layer, method[, alpha])
config, so the whole sweep can be compared in one table at the end.
"""
import argparse
import json
import os

import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import build_prompt, load_model, num_layers
from psr_conceptor_logic import (
    collect_live_training_pair,
    forward_with_gate_hook,
    load_or_compute_responses,
    regularization_loss,
    subsequent_layers_mse,
)
from psr_conceptor_state import (
    compute_conceptor,
    conceptor_direction,
    init_gate_params,
    init_psr_trained_params,
)

N_EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
REG_COEFF = 0.1
CONCEPTOR_ALPHAS = [2.0, 4.0]  # narrowed from the full 5-point grid -- alpha=2 already won there;
# checking one neighbor (4.0) per layer instead of the full 5 keeps the grid affordable while still
# catching it if the optimal alpha shifts at a different layer.

LAYER_GRID = [int(x) for x in os.environ.get(
    "LAYER_GRID", "2,4,6,8,10,12,14,16,18,20,22,24,26"
).split(",")]


@torch.no_grad()
def collect_all_layers_pooled(model, tokenizer, rows, responses, n_layers):
    """One forward pass per row per pole (same cost as the single-layer version), returns
    {layer_idx: {"base": Tensor(N,d), "instr": Tensor(N,d)}} for every layer at once."""
    per_layer = {l: {"base": [], "instr": []} for l in range(n_layers)}
    for row in rows:
        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
        teacher_response = responses[str(row["id"])]
        resp_ids = tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        if resp_ids.shape[1] == 0:
            continue
        base_ids = tokenizer(base_prompt, return_tensors="pt")["input_ids"].to(model.device)
        instr_ids = tokenizer(terse_prompt, return_tensors="pt")["input_ids"].to(model.device)
        full_base = torch.cat([base_ids, resp_ids], dim=1)
        full_instr = torch.cat([instr_ids, resp_ids], dim=1)
        n_resp = resp_ids.shape[1]

        out_base = model(input_ids=full_base, output_hidden_states=True)
        out_instr = model(input_ids=full_instr, output_hidden_states=True)
        for l in range(n_layers):
            per_layer[l]["base"].append(out_base.hidden_states[l + 1][0, -n_resp:, :].float().cpu())
            per_layer[l]["instr"].append(out_instr.hidden_states[l + 1][0, -n_resp:, :].float().cpu())

    return {
        l: {"base": torch.cat(v["base"], dim=0), "instr": torch.cat(v["instr"], dim=0)}
        for l, v in per_layer.items()
    }


def rank_diagnostic(base_pool: torch.Tensor, instr_pool: torch.Tensor, direction: torch.Tensor) -> dict:
    """Same math as diagnose_conceptor_rank.py, inlined here so the sweep reports it per layer
    without a second pass over the data."""
    pooled = torch.cat([base_pool, instr_pool], dim=0)
    n, d = pooled.shape
    r = (pooled.T @ pooled) / n
    eigvals, eigvecs = torch.linalg.eigh(r)
    order = torch.argsort(eigvals, descending=True)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    pr = ((eigvals.sum() ** 2) / (eigvals**2).sum()).item()
    direction = direction / direction.norm()
    top1_capture = ((eigvecs[:, :1].T @ direction) ** 2).sum().item()
    quota_alpha2 = (eigvals / (eigvals + 2.0**-2)).mean().item()
    return {"participation_ratio": pr, "top1_capture_fraction": top1_capture, "quota_alpha2": quota_alpha2}


def train_gate(model, tokenizer, direction, params, layer_idx, n_layers, train_rows, dev_rows, responses_train, responses_dev):
    optimizer = torch.optim.Adam(params.as_list(), lr=LR, weight_decay=WEIGHT_DECAY)

    def eval_dev():
        losses = []
        for row in dev_rows:
            pair = collect_live_training_pair(model, tokenizer, row, layer_idx, responses_dev)
            if pair is None:
                continue
            with torch.no_grad():
                target_hidden = model(input_ids=pair["full_instr"], output_hidden_states=True).hidden_states
            pred_hidden, _ = forward_with_gate_hook(model, params, direction, layer_idx, pair["full_base"], pair["n_resp"])
            losses.append(subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers).item())
        return sum(losses) / len(losses)

    baseline_mse = eval_dev()
    for _ in range(N_EPOCHS):
        for row in train_rows:
            pair = collect_live_training_pair(model, tokenizer, row, layer_idx, responses_train)
            if pair is None:
                continue
            with torch.no_grad():
                target_hidden = model(input_ids=pair["full_instr"], output_hidden_states=True).hidden_states
            pred_hidden, fit_vals = forward_with_gate_hook(model, params, direction, layer_idx, pair["full_base"], pair["n_resp"])
            mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
            loss = mse + regularization_loss(fit_vals, REG_COEFF)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    final_mse = eval_dev()
    return baseline_mse, final_mse


def main(layer_grid: list[int]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")

    train_responses = load_or_compute_responses(model, tokenizer, train_rows, RESULTS_DIR / "teacher_responses_train.json")
    dev_responses = load_or_compute_responses(model, tokenizer, dev_rows, RESULTS_DIR / "teacher_responses_dev.json")

    print("pooling ALL layers in one pass over train+dev (this is the efficiency win -- one pass regardless of grid size)")
    pooled_train = collect_all_layers_pooled(model, tokenizer, train_rows, train_responses, n_layers)

    out_path = RESULTS_DIR / "layer_sweep.jsonl"
    results = []
    done = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            results.append(row)
            done.add((row["layer"], row["method"], row["alpha"]))
        print(f"resuming: {len(done)} (layer, method, alpha) combos already done in {out_path}")

    for layer_idx in layer_grid:
        print(f"\n=== layer {layer_idx} ===")
        base_pool = pooled_train[layer_idx]["base"].to(device)
        instr_pool = pooled_train[layer_idx]["instr"].to(device)
        diff_mean_direction = (instr_pool.mean(0) - base_pool.mean(0))
        dm_norm = diff_mean_direction.norm()
        if dm_norm < 1e-6:
            print(f"  WARNING: near-zero diff-mean norm ({dm_norm.item():.2e}) at layer {layer_idx} -- "
                  f"base and terse-prompt activations are nearly identical here, skipping this layer "
                  f"rather than dividing by ~0 and propagating NaN through training.")
            continue
        diff_mean_direction = diff_mean_direction / dm_norm

        diag = rank_diagnostic(base_pool, instr_pool, diff_mean_direction)
        print(f"  rank diagnostic: participation_ratio={diag['participation_ratio']:.2f}, "
              f"top1_capture={diag['top1_capture_fraction']:.1%}, quota(alpha=2)={diag['quota_alpha2']:.4f}")

        hidden_size = base_pool.shape[1]

        # psr_proper: jointly trained direction, no conceptor
        if (layer_idx, "psr_proper", None) in done:
            print(f"  psr_proper: already done, skipping")
        else:
            proper_params = init_psr_trained_params(hidden_size, device)
            baseline_mse, final_mse = train_gate(
                model, tokenizer, proper_params.direction, proper_params, layer_idx, n_layers,
                train_rows, dev_rows, train_responses, dev_responses,
            )
            print(f"  psr_proper: baseline={baseline_mse:.4f} -> final={final_mse:.4f}")
            results.append({
                "layer": layer_idx, "method": "psr_proper", "alpha": None,
                "baseline_mse": baseline_mse, "final_mse": final_mse, **diag,
            })

        # psr_conceptor: closed-form direction at each alpha in the (narrowed) grid
        pooled_r = torch.cat([base_pool, instr_pool], dim=0)
        for alpha in CONCEPTOR_ALPHAS:
            if (layer_idx, "psr_conceptor", alpha) in done:
                print(f"  psr_conceptor alpha={alpha}: already done, skipping")
                continue
            conceptor = compute_conceptor(pooled_r, alpha=alpha)
            direction = conceptor_direction(conceptor, diff_mean_direction)
            gate_params = init_gate_params(hidden_size, device)
            baseline_mse, final_mse = train_gate(
                model, tokenizer, direction, gate_params, layer_idx, n_layers,
                train_rows, dev_rows, train_responses, dev_responses,
            )
            print(f"  psr_conceptor alpha={alpha}: baseline={baseline_mse:.4f} -> final={final_mse:.4f}")
            results.append({
                "layer": layer_idx, "method": "psr_conceptor", "alpha": alpha,
                "baseline_mse": baseline_mse, "final_mse": final_mse, **diag,
            })

        # Checkpoint after every layer, not just at the end -- same discipline as the earlier
        # per-epoch checkpointing, for the same reason (don't lose N-1 layers to a disconnect).
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default=None, help="comma-separated layer indices, overrides LAYER_GRID env var")
    args = parser.parse_args()
    grid = [int(x) for x in args.layers.split(",")] if args.layers else LAYER_GRID
    main(grid)