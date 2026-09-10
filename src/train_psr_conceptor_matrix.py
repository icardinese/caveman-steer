"""Trains the GENUINE matrix-application conceptor variant: correction(h) = coeff(h) * (C @ (mu_instr
- h)), evaluated fresh every position -- not gate_correction()'s single fixed vector computed once
from C @ base_direction. This is the actual test of whether the conceptor's multidimensionality buys
anything when it's allowed to do ongoing work, rather than being collapsed to a rank-1 direction and
discarded immediately after one matrix-vector product.

Same fidelity fixes as train_psr_conceptor.py (answer_only masking, subsequent-layers MSE, live
gradient-tracked forward, regularization) -- only the correction mechanism itself changes.
"""
import json
import os

import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import build_prompt, load_model, num_layers
from psr_conceptor_logic import (
    collect_target_hidden_states,
    compute_delta_scale,
    forward_with_gate_hook_matrix,
    load_or_compute_responses,
    regularization_loss,
    subsequent_layers_mse,
)
from psr_conceptor_state import compute_conceptor, init_gate_params

N_EPOCHS = 3
LR = float(os.environ.get("PSR_CONCEPTOR_MATRIX_LR", 1e-3))
WEIGHT_DECAY = 1e-4
ALPHA = float(os.environ.get("PSR_CONCEPTOR_MATRIX_ALPHA", 4.0))
REG_COEFF = float(os.environ.get("PSR_CONCEPTOR_MATRIX_REG_COEFF", 0.1))
OUT_TAG = os.environ.get("PSR_CONCEPTOR_MATRIX_OUT_TAG", "")
LAYER_OVERRIDE = os.environ.get("PSR_CONCEPTOR_MATRIX_LAYER")


@torch.no_grad()
def collect_pooled_separate_poles(model, tokenizer, rows, layer_idx, responses):
    """Like collect_pooled_activations_for_conceptor, but keeps base and instr activations
    SEPARATE (not pre-concatenated) -- needed here because mu_instr must come from the instr pole
    alone, not a base+instr mix. Same forward-pass cost as the original."""
    base_list, instr_list = [], []
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
        base_list.append(out_base.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
        instr_list.append(out_instr.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu())
    return torch.cat(base_list, dim=0), torch.cat(instr_list, dim=0)


@torch.no_grad()
def eval_dev_mse(model, tokenizer, params, conceptor, mu_instr, delta_scale, layer_idx, n_layers, dev_rows, responses):
    from psr_conceptor_logic import collect_live_training_pair
    losses = []
    for row in dev_rows:
        pair = collect_live_training_pair(model, tokenizer, row, layer_idx, responses)
        if pair is None:
            continue
        target_hidden = collect_target_hidden_states(model, pair["full_instr"])
        pred_hidden, _ = forward_with_gate_hook_matrix(model, params, conceptor, mu_instr, delta_scale, layer_idx, pair["full_base"], pair["n_resp"])
        mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
        losses.append(mse.item())
    return sum(losses) / len(losses)


def main() -> None:
    final_path = RESULTS_DIR / f"psr_conceptor_matrix_probe{OUT_TAG}.pt"
    if final_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {final_path} already exists, skipping (set FORCE_RERUN=1 to redo)")
        return

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
    print(f"layer = {layer_idx}, alpha = {ALPHA}")

    train_responses = load_or_compute_responses(model, tokenizer, train_rows, RESULTS_DIR / "teacher_responses_train.json")
    dev_responses = load_or_compute_responses(model, tokenizer, dev_rows, RESULTS_DIR / "teacher_responses_dev.json")

    print("collecting base/instr pools separately (needed for mu_instr, unlike the fixed-vector variant)")
    from psr_conceptor_logic import collect_live_training_pair  # noqa: F401 (used in eval_dev_mse)
    base_pool, instr_pool = collect_pooled_separate_poles(model, tokenizer, train_rows, layer_idx, train_responses)
    base_pool, instr_pool = base_pool.to(device), instr_pool.to(device)
    pooled_for_r = torch.cat([base_pool, instr_pool], dim=0)
    conceptor = compute_conceptor(pooled_for_r, alpha=ALPHA)
    mu_instr = instr_pool.mean(dim=0)
    delta_scale = compute_delta_scale(conceptor, mu_instr, base_pool)
    print(f"delta_scale = {delta_scale.item():.2f}")

    hidden_size = base_pool.shape[1]
    params = init_gate_params(hidden_size, device)
    optimizer = torch.optim.Adam(params.as_list(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("baseline dev MSE (gate untrained)")
    baseline_mse = eval_dev_mse(model, tokenizer, params, conceptor, mu_instr, delta_scale, layer_idx, n_layers, dev_rows, dev_responses)
    print(f"baseline_dev_mse={baseline_mse:.4f}")

    def save_checkpoint(epoch, dev_mse) -> None:
        torch.save({
            "weight": params.weight.detach().cpu(), "bias": params.bias.detach().cpu(),
            "coeff_bias": params.coeff_bias.detach().cpu(),
            "conceptor": conceptor.detach().cpu(), "mu_instr": mu_instr.detach().cpu(), "delta_scale": delta_scale.detach().cpu(),
            "layer": layer_idx, "alpha": ALPHA,
            "completed_epochs": epoch if epoch is not None else N_EPOCHS,
        }, final_path)
        with (RESULTS_DIR / f"psr_conceptor_matrix_train_log{OUT_TAG}.json").open("w") as f:
            json.dump({
                "layer": layer_idx, "alpha": ALPHA, "lr": LR, "reg_coeff": REG_COEFF,
                "baseline_dev_mse": baseline_mse,
                "completed_epochs": epoch if epoch is not None else N_EPOCHS,
                "dev_mse_after_last_completed_epoch": dev_mse,
            }, f, indent=2)

    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for row in train_rows:
            pair = collect_live_training_pair(model, tokenizer, row, layer_idx, train_responses)
            if pair is None:
                continue
            target_hidden = collect_target_hidden_states(model, pair["full_instr"])
            pred_hidden, fit_vals = forward_with_gate_hook_matrix(
                model, params, conceptor, mu_instr, delta_scale, layer_idx, pair["full_base"], pair["n_resp"]
            )
            mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
            loss = mse + regularization_loss(fit_vals, REG_COEFF)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch={epoch} train_loss={avg_loss:.4f}")
        epoch_dev_mse = eval_dev_mse(model, tokenizer, params, conceptor, mu_instr, delta_scale, layer_idx, n_layers, dev_rows, dev_responses)
        print(f"epoch={epoch} dev_mse={epoch_dev_mse:.4f} -- checkpointing")
        save_checkpoint(epoch, epoch_dev_mse)

    final_mse = eval_dev_mse(model, tokenizer, params, conceptor, mu_instr, delta_scale, layer_idx, n_layers, dev_rows, dev_responses)
    print(f"final_dev_mse={final_mse:.4f} (baseline was {baseline_mse:.4f})")
    save_checkpoint(None, final_mse)
    print(f"wrote {final_path}")


if __name__ == "__main__":
    main()