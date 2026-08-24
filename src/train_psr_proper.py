"""Train "proper" PSR: same fidelity fixes as train_psr_conceptor.py (answer_only masking,
subsequent-layers MSE, live gradient-tracked forward pass, location_fit regularization), but the
direction (z_attr) is jointly gradient-trained from Nokia's actual default init -- not conceptor-
projected, not warm-started from the Const mean-diff vector.

This exists to isolate ONE variable against train_psr_conceptor.py: same everything else, only the
direction's origin differs. Comparing this against the old steering_psr.py baseline (offline MSE,
injection-layer-only, no regularization) conflates the fidelity fixes with the direction choice --
comparing THIS against train_psr_conceptor.py isolates the direction choice cleanly.
"""
import json
import os

import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import load_model, num_layers
from psr_conceptor_state import init_psr_trained_params
from psr_conceptor_logic import (
    collect_live_training_pair,
    collect_target_hidden_states,
    forward_with_gate_hook,
    load_or_compute_responses,
    regularization_loss,
    subsequent_layers_mse,
)

N_EPOCHS = 3
LR = float(os.environ.get("PSR_PROPER_LR", 1e-3))
WEIGHT_DECAY = 1e-4
REG_COEFF = float(os.environ.get("PSR_PROPER_REG_COEFF", 0.1))
OUT_TAG = os.environ.get("PSR_PROPER_OUT_TAG", "")


@torch.no_grad()
def eval_dev_mse(model, params, layer_idx, n_layers, dev_rows, tokenizer, responses) -> float:
    losses = []
    for row in dev_rows:
        pair = collect_live_training_pair(model, tokenizer, row, layer_idx, responses)
        if pair is None:
            continue
        target_hidden = collect_target_hidden_states(model, pair["full_instr"])
        pred_hidden, _ = forward_with_gate_hook(model, params, params.direction, layer_idx, pair["full_base"], pair["n_resp"])
        mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
        losses.append(mse.item())
    return sum(losses) / len(losses)


def main() -> None:
    final_path = RESULTS_DIR / f"psr_proper_probe{OUT_TAG}.pt"
    if final_path.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
        print(f">>> {final_path} already exists, skipping (set FORCE_RERUN=1 to redo)")
        return

    device = "cuda"
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model(device)
    for p in model.parameters():
        p.requires_grad_(False)  # same OOM guard as train_psr_conceptor.py -- frozen base model,
        # gate + direction are the only leaf tensors that should accumulate .grad
    n_layers = num_layers(model)
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")

    with (RESULTS_DIR / "const_steer_config.json").open() as f:
        const_config = json.load(f)
    layer_idx = const_config["layer"]  # same layer as the conceptor variant, for a clean comparison

    print(f"loading/precomputing teacher-forced responses for {len(train_rows)} train + {len(dev_rows)} dev rows "
          f"(shared cache with train_psr_conceptor.py -- same rows, same deterministic generation)")
    train_responses = load_or_compute_responses(model, tokenizer, train_rows, RESULTS_DIR / "teacher_responses_train.json")
    dev_responses = load_or_compute_responses(model, tokenizer, dev_rows, RESULTS_DIR / "teacher_responses_dev.json")

    hidden_size = model.config.hidden_size
    params = init_psr_trained_params(hidden_size, device)
    optimizer = torch.optim.Adam(params.as_list(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("baseline dev MSE (gate + direction both untrained)")
    baseline_mse = eval_dev_mse(model, params, layer_idx, n_layers, dev_rows, tokenizer, dev_responses)
    print(f"baseline_dev_mse={baseline_mse:.4f}")

    def save_checkpoint(epoch: int | None, dev_mse: float | None) -> None:
        torch.save(
            {
                "weight": params.weight.detach().cpu(),
                "bias": params.bias.detach().cpu(),
                "coeff_bias": params.coeff_bias.detach().cpu(),
                "direction": params.direction.detach().cpu(),
                "layer": layer_idx,
                "completed_epochs": epoch if epoch is not None else N_EPOCHS,
            },
            final_path,
        )
        with (RESULTS_DIR / f"psr_proper_train_log{OUT_TAG}.json").open("w") as f:
            json.dump(
                {
                    "layer": layer_idx, "lr": LR, "reg_coeff": REG_COEFF,
                    "baseline_dev_mse": baseline_mse,
                    "completed_epochs": epoch if epoch is not None else N_EPOCHS,
                    "dev_mse_after_last_completed_epoch": dev_mse,
                },
                f, indent=2,
            )

    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for row in train_rows:
            pair = collect_live_training_pair(model, tokenizer, row, layer_idx, train_responses)
            if pair is None:
                continue
            target_hidden = collect_target_hidden_states(model, pair["full_instr"])
            pred_hidden, fit_vals = forward_with_gate_hook(
                model, params, params.direction, layer_idx, pair["full_base"], pair["n_resp"]
            )
            mse = subsequent_layers_mse(pred_hidden, target_hidden, layer_idx, pair["n_resp"], n_layers)
            reg = regularization_loss(fit_vals, REG_COEFF)
            loss = mse + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch={epoch} train_loss={avg_loss:.4f}")
        epoch_dev_mse = eval_dev_mse(model, params, layer_idx, n_layers, dev_rows, tokenizer, dev_responses)
        print(f"epoch={epoch} dev_mse={epoch_dev_mse:.4f} -- checkpointing")
        save_checkpoint(epoch=epoch, dev_mse=epoch_dev_mse)

    print("final dev MSE (gate + direction both trained)")
    final_mse = eval_dev_mse(model, params, layer_idx, n_layers, dev_rows, tokenizer, dev_responses)
    print(f"final_dev_mse={final_mse:.4f} (baseline was {baseline_mse:.4f})")
    save_checkpoint(epoch=None, dev_mse=final_mse)
    print(f"wrote {final_path} and psr_proper_train_log{OUT_TAG}.json")


if __name__ == "__main__":
    main()
