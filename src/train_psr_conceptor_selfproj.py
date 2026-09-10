"""FINAL conceptor test: self-projection variant, correction(h) = coeff(h) * (C @ h - h) / scale.
No mu_instr, no population-mean confound -- isolates matrix-vs-vector cleanly.

Critical fix from the previous attempt: alpha is chosen ADAPTIVELY from R's actual eigenvalue
spectrum, not a guessed absolute number. mu_i = sigma_i / (sigma_i + alpha^-2) is close to 1 (i.e.
C ~ I, no meaningful filtering) whenever alpha^-2 << sigma_i for most eigenvalues -- confirmed on
synthetic data that alpha=4 (the value used throughout the earlier sweep) gives mu >= 0.889 even
at the SMALLEST eigenvalue in a realistically-shaped spectrum. If real activations have a similar
or larger eigenvalue scale, every alpha tested so far (2-32) may never have given the conceptor a
fair chance to differ from doing nothing. This script picks alphas at fixed PERCENTILES of the
observed eigenvalue spectrum (so alpha^-2 = eigenvalue at that percentile), guaranteeing the grid
spans from "barely filters anything" to "filters almost everything" regardless of the real scale.
"""
import json
import os

import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import build_prompt, load_model, num_layers
from psr_conceptor_logic import (
    collect_live_training_pair,
    collect_target_hidden_states,
    compute_selfproj_delta_scale,
    forward_with_gate_hook_selfproj,
    load_or_compute_responses,
    regularization_loss,
    subsequent_layers_mse,
)
from psr_conceptor_state import init_gate_params

N_EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
REG_COEFF = 0.1
PERCENTILES = [10, 30, 50, 70, 90]  # of the eigenvalue spectrum -- see module docstring
LAYER_OVERRIDE = os.environ.get("PSR_CONCEPTOR_SELFPROJ_LAYER")


@torch.no_grad()
def collect_pool(model, tokenizer, rows, layer_idx, responses):
    pooled = []
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
        pooled.append(out_base.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
        pooled.append(out_instr.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
    return torch.cat(pooled, dim=0)


def eigendecompose(pool):
    n, d = pool.shape
    r = (pool.T @ pool) / n
    eigvals, eigvecs = torch.linalg.eigh(r)
    order = torch.argsort(eigvals, descending=True)
    return eigvals[order], eigvecs[:, order], r


@torch.no_grad()
def eval_dev_mse(model, tokenizer, params, conceptor, delta_scale, layer_idx, n_layers, dev_rows, responses):
    losses = []
    for row in dev_rows:
        pair = collect_live_training_pair(model, tokenizer, row, layer_idx, responses)
        if pair is None:
            continue
        target_hidden = collect_target_hidden_states(model, pair["full_instr"])
        pred_hidden, _ = forward_with_gate_hook_selfproj(model, params, conceptor, delta_scale, layer_idx, pair["full_base"], pair["n_resp"])
        mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
        losses.append(mse.item())
    return sum(losses) / len(losses)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = num_layers(model)
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")

    if LAYER_OVERRIDE is not None:
        layer_idx = int(LAYER_OVERRIDE)
    else:
        with (RESULTS_DIR / "const_steer_config.json").open() as f:
            layer_idx = json.load(f)["layer"]
    print(f"layer = {layer_idx}")

    train_responses = load_or_compute_responses(model, tokenizer, train_rows, RESULTS_DIR / "teacher_responses_train.json")
    dev_responses = load_or_compute_responses(model, tokenizer, dev_rows, RESULTS_DIR / "teacher_responses_dev.json")

    print("collecting pooled activations for eigendecomposition...")
    pool = collect_pool(model, tokenizer, train_rows, layer_idx, train_responses).to(device)
    eigvals, eigvecs, r = eigendecompose(pool)
    d = eigvals.shape[0]
    print(f"eigenvalue range: max={eigvals[0].item():.4f}, min={eigvals[-1].item():.4f}, "
          f"median={eigvals[d//2].item():.4f}")

    # Adaptive alpha grid: alpha^-2 = eigenvalue at each percentile -> alpha = 1/sqrt(eigenvalue).
    # This guarantees the grid spans from "alpha^-2 >> all eigenvalues" (C ~ 0, extreme filtering)
    # to "alpha^-2 << all eigenvalues" (C ~ I, no filtering), regardless of the real data's scale.
    alpha_grid = []
    for p in PERCENTILES:
        idx = min(d - 1, int((100 - p) / 100 * d))  # percentile of eigenvalue magnitude (sorted descending)
        eig_at_p = max(eigvals[idx].item(), 1e-6)
        alpha_grid.append((p, 1.0 / (eig_at_p ** 0.5)))
    print("adaptive alpha grid (percentile of eigenvalue spectrum -> alpha):")
    for p, a in alpha_grid:
        print(f"  p{p}: alpha={a:.4f}")

    out_path = RESULTS_DIR / "psr_conceptor_selfproj_results.jsonl"
    results = []
    hidden_size = d

    for percentile, alpha in alpha_grid:
        identity = torch.eye(d, device=device, dtype=pool.dtype)
        conceptor = r @ torch.linalg.inv(r + (alpha ** -2) * identity)
        delta_scale = compute_selfproj_delta_scale(conceptor, pool)
        mu_mean = (eigvals / (eigvals + alpha ** -2)).mean().item()
        print(f"\n=== percentile={percentile}, alpha={alpha:.4f}, mean(mu)={mu_mean:.4f}, delta_scale={delta_scale.item():.4f} ===")
        if delta_scale.item() < 1e-3:
            print("  SKIPPING: delta_scale ~0, C is indistinguishable from identity at this alpha, nothing to learn")
            results.append({"percentile": percentile, "alpha": alpha, "mean_mu": mu_mean,
                             "delta_scale": delta_scale.item(), "skipped": True})
            continue

        params = init_gate_params(hidden_size, device)
        optimizer = torch.optim.Adam(params.as_list(), lr=LR, weight_decay=WEIGHT_DECAY)
        baseline_mse = eval_dev_mse(model, tokenizer, params, conceptor, delta_scale, layer_idx, n_layers, dev_rows, dev_responses)
        print(f"  baseline_dev_mse={baseline_mse:.4f}")

        for epoch in range(N_EPOCHS):
            for row in train_rows:
                pair = collect_live_training_pair(model, tokenizer, row, layer_idx, train_responses)
                if pair is None:
                    continue
                target_hidden = collect_target_hidden_states(model, pair["full_instr"])
                pred_hidden, fit_vals = forward_with_gate_hook_selfproj(
                    model, params, conceptor, delta_scale, layer_idx, pair["full_base"], pair["n_resp"]
                )
                mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
                loss = mse + regularization_loss(fit_vals, REG_COEFF)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        final_mse = eval_dev_mse(model, tokenizer, params, conceptor, delta_scale, layer_idx, n_layers, dev_rows, dev_responses)
        print(f"  final_dev_mse={final_mse:.4f}")
        results.append({"percentile": percentile, "alpha": alpha, "mean_mu": mu_mean,
                         "delta_scale": delta_scale.item(), "skipped": False,
                         "baseline_mse": baseline_mse, "final_mse": final_mse})
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

    print(f"\nwrote {len(results)} rows to {out_path}")
    best = min((r for r in results if not r["skipped"]), key=lambda r: r["final_mse"], default=None)
    if best:
        print(f"\nbest: percentile={best['percentile']}, alpha={best['alpha']:.4f}, "
              f"final_mse={best['final_mse']:.4f} (psr_proper's was 5.77, old fixed-vector conceptor's best was 14.87-15.89)")


if __name__ == "__main__":
    main()